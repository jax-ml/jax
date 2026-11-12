# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision tests for `jax.scipy.stats.truncnorm`.

Compares `bfloat16`, `float16`, `float32`, and `float64` values and gradients
against 250-bit `mpmath` references (using JAX's `mpmath`-verified `float64`
path as the reference for large low-precision sweeps), along with checks for
series constants, custom autodiff rules, and quadrature bounds.
"""

from collections.abc import Sequence
import fractions
import math

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.scipy.stats.truncnorm import (
    _DoubleWord,
    _FactoredExp,
    _GL4_NODES,
    _GL4_WEIGHTS,
    _NEAR_BOUND_COEFFS,
    _NEAR_ZERO_COEFFS_F32,
    _NEAR_ZERO_COEFFS_F64,
    _QUAD_MAX_EXP_DIFF,
    _QUAD_MAX_WIDTH,
    _half_sq_diff as _jax_half_sq_diff,
    _normal_prob_quad,
    _use_quad,
)
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special
import scipy.stats

config.parse_flags_with_absl()

truncnorm = jsp.stats.truncnorm
bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]

_PREC = 250  # Working precision in bits for the mpmath reference functions.
# In mpmath, unary `+x` rounds `x` to the current working precision, so each
# oracle works with extra guard precision and rounds once on return.


def _erfcx_nonneg(x):
  """Extended-precision erfcx(x) = e^{x^2} erfc(x) for x >= 0."""
  if x == 0:
    return mpmath.mpf(1)
  if x == mpmath.inf:
    return mpmath.mpf(0)
  if x < 50:
    with mpmath.extraprec(max(0, 2 * int(mpmath.mag(x))) + 32):
      return mpmath.exp(x**2) * mpmath.erfc(x)
  return mpmath.hyperu(0.5, 0.5, x**2) / mpmath.sqrt(mpmath.pi)


def _exact_sub(x, y):
  if not (mpmath.isfinite(x) and mpmath.isfinite(y)):
    return x - y
  return mpmath.mpf(0) if x == y else mpmath.fsub(x, y, exact=True)


def _exact_add(x, y):
  if not (mpmath.isfinite(x) and mpmath.isfinite(y)):
    return x + y
  return mpmath.mpf(0) if x == -y else mpmath.fadd(x, y, exact=True)


def _half_sq_diff(x, y):
  return _exact_sub(x, y) * _exact_add(x, y) / 2


def _oracle_prec():
  return max(_PREC + 32, mpmath.mp.prec)


def _mpmath_normal_prob(a, b, prec=None):
  """Computes log(Phi(b) - Phi(a)) as (anchor, log_coeff).

  Represents the log-probability as -anchor^2 / 2 + log_coeff to preserve
  precision even when Phi(b) - Phi(a) is tiny (deep tail). This is
  the arbitrary-precision analogue of _normal_prob in truncnorm.py.
  """
  if prec is None:
    prec = max(_PREC, mpmath.mp.prec)
  # Reflect intervals with `b <= 0` to `0 <= -b < -a`. Exact negation avoids
  # rounding before adding or subtracting endpoints passed from `_mpmath_ppf`.
  if b <= 0:
    a, b = mpmath.fneg(b, exact=True), mpmath.fneg(a, exact=True)
  inv_sqrt_2 = mpmath.sqrt(mpmath.mpf(0.5))
  if a < 0:  # a < 0 < b: spans zero, so anchor = 0.
    with mpmath.workprec(prec + 32):
      tail_a = mpmath.exp(-0.5 * a**2) * _erfcx_nonneg(-a * inv_sqrt_2)
      tail_b = mpmath.exp(-0.5 * b**2) * _erfcx_nonneg(b * inv_sqrt_2)
      tail = 0.5 * (tail_a + tail_b)
      if tail <= 0.5:
        log_coeff = mpmath.log1p(-tail)
      else:
        log_coeff = mpmath.log(
            0.5 * (mpmath.erf(-a * inv_sqrt_2) + mpmath.erf(b * inv_sqrt_2))
        )
    return mpmath.mpf(0), +log_coeff

  # When `0 <= a < b`, subtracting scaled `erfcx` values cancels digits on
  # narrow intervals. Retry with extra guard bits until full precision holds.
  width = _exact_sub(b, a)
  extra = 32
  if width != mpmath.inf and a != 0:
    extra += max(0, int(mpmath.mag(a) - mpmath.mag(width)))
  for _ in range(20):
    with mpmath.workprec(prec + extra):
      erfcx_a = _erfcx_nonneg(a * inv_sqrt_2)
      if b == mpmath.inf:
        val = erfcx_a
        break
      scaled_erfcx_b = (
          mpmath.exp(-0.5 * width * (b + a)) * _erfcx_nonneg(b * inv_sqrt_2)
      )
      val = erfcx_a - scaled_erfcx_b
      lost = mpmath.mag(erfcx_a) - mpmath.mag(val) if val else extra + 64
    if lost + 32 <= extra:
      break
    extra = int(lost) + 64
  else:
    raise RuntimeError(f"_mpmath_normal_prob did not converge: {a}, {b}")
  with mpmath.workprec(prec + 32):
    return a, +mpmath.log(0.5 * val)


def _mpmath_logpdf(x, a, b, prob_ab=None):
  if mpmath.isnan(x) or mpmath.isnan(a) or mpmath.isnan(b) or a >= b:
    return mpmath.nan
  if x < a or x > b or mpmath.isinf(x):
    return -mpmath.inf
  with mpmath.workprec(_oracle_prec()):
    anchor, log_coeff = (
        prob_ab if prob_ab is not None else _mpmath_normal_prob(a, b)
    )
    return +(
        _half_sq_diff(anchor, x) - 0.5 * mpmath.log(2 * mpmath.pi) - log_coeff
    )


def _mpmath_logcdf(x, a, b, prob_ab=None):
  if mpmath.isnan(x) or mpmath.isnan(a) or mpmath.isnan(b) or a >= b:
    return mpmath.nan
  if x <= a:
    return -mpmath.inf
  if x >= b:
    return mpmath.mpf(0)
  with mpmath.workprec(_oracle_prec()):
    anchor_ab, log_ab = (
        prob_ab if prob_ab is not None else _mpmath_normal_prob(a, b)
    )
    anchor_ax, log_ax = _mpmath_normal_prob(a, x)
    logcdf = _half_sq_diff(anchor_ab, anchor_ax) + (log_ax - log_ab)
    if logcdf > -mpmath.log(2):
      # cdf > 1/2: computing it as 1 - sf keeps the digits of the small sf.
      anchor_xb, log_xb = _mpmath_normal_prob(x, b)
      logsf = _half_sq_diff(anchor_ab, anchor_xb) + (log_xb - log_ab)
      return +mpmath.log1p(-mpmath.exp(logsf))
    return +logcdf


def _mpmath_pdf(x, a, b, prob_ab=None):
  with mpmath.workprec(_oracle_prec()):
    return +mpmath.exp(_mpmath_logpdf(x, a, b, prob_ab=prob_ab))


def _mpmath_cdf(x, a, b, prob_ab=None):
  with mpmath.workprec(_oracle_prec()):
    return +mpmath.exp(_mpmath_logcdf(x, a, b, prob_ab=prob_ab))


def _mpmath_logsf(x, a, b, prob_ab=None):
  neg_ab = (
      None
      if prob_ab is None
      else (mpmath.fneg(prob_ab[0], exact=True), prob_ab[1])
  )
  with mpmath.workprec(_oracle_prec()):
    neg_x = mpmath.fneg(x, exact=True)
    neg_b = mpmath.fneg(b, exact=True)
    neg_a = mpmath.fneg(a, exact=True)
    return +_mpmath_logcdf(neg_x, neg_b, neg_a, prob_ab=neg_ab)


def _mpmath_sf(x, a, b, prob_ab=None):
  with mpmath.workprec(_oracle_prec()):
    return +mpmath.exp(_mpmath_logsf(x, a, b, prob_ab=prob_ab))


def _mpmath_ppf(q, a, b, prob_ab=None):
  """Reference inverse CDF using bracketed Newton steps in log space.

  Solves `log(Phi(z) - Phi(a)) - log(Phi(b) - Phi(a)) - log(q) = 0`. Splits
  `z = base + offset` and iterates on `offset` so tiny steps near `a` or `b`
  are not lost to rounding. Runs a coarse 80-bit pass followed by full-precision
  refinement.
  """
  if mpmath.isnan(q) or mpmath.isnan(a) or mpmath.isnan(b):
    return mpmath.nan
  if q < 0 or q > 1 or a >= b:
    return mpmath.nan
  if q == 0:
    return a
  if q == 1:
    return b
  if q == mpmath.mpf("0.5") and a == -b:
    return mpmath.mpf(0)
  outer_prec = max(_PREC, mpmath.mp.prec)
  with mpmath.workprec(outer_prec + 16):
    if q > mpmath.mpf("0.5"):
      neg_ab = None if prob_ab is None else (-prob_ab[0], prob_ab[1])
      return -_mpmath_ppf(1 - q, -b, -a, prob_ab=neg_ab)

    anchor_ab_full, log_ab_full = (
        prob_ab
        if prob_ab is not None
        else _mpmath_normal_prob(a, b, prec=outer_prec)
    )
    log_q_full = mpmath.log(q)

    # Start near a finite lower bound `a` with a linear step when it stays well
    # inside `[a, b]`.
    if mpmath.isfinite(a):
      prob_over_pdf = mpmath.exp(
          log_q_full + log_ab_full + _half_sq_diff(a, anchor_ab_full)
      ) * mpmath.sqrt(2 * mpmath.pi)
      if prob_over_pdf <= 0.5 * (b - a) and not (
          a < 0 < b and a + prob_over_pdf > 0
      ):
        base, offset = a, prob_over_pdf
      elif mpmath.isfinite(b) and b - a <= 1:
        base, offset = a, 0.5 * (b - a)
      else:
        base = mpmath.mpf(0)
        offset = 0.5 * (a + min(b, mpmath.mpf(0)))
    else:  # `a == -inf`: `log Phi(z) = -anchor_ab^2 / 2 + log_ab + log_q`.
      log_phi_z = -0.5 * anchor_ab_full**2 + log_ab_full + log_q_full
      z_init = (
          -mpmath.sqrt(-2 * log_phi_z)
          if log_phi_z < -2
          else min(mpmath.mpf(0), b - 1)
      )
      if z_init >= b:
        z_init = b - 1
      base = b if (mpmath.isfinite(b) and b - z_init < 1) else mpmath.mpf(0)
      offset = z_init - base

    for stage_prec, tol in (
        (80, mpmath.mpf(2) ** -56),
        (outer_prec, mpmath.mpf(2) ** -(outer_prec // 2)),
    ):
      with mpmath.workprec(stage_prec + 16):
        lo, hi = a - base, b - base
        anchor_ab = +anchor_ab_full
        log_ab = +log_ab_full
        log_q = +log_q_full
        inv_sqrt_2pi = 1 / mpmath.sqrt(2 * mpmath.pi)
        for _ in range(200):
          z = _exact_add(base, offset)
          anchor_z, log_z = _mpmath_normal_prob(a, z, prec=stage_prec)
          residual = (
              _half_sq_diff(anchor_ab, anchor_z) + (log_z - log_ab) - log_q
          )
          if residual == 0:
            break
          if residual > 0:
            hi = offset
          else:
            lo = offset
          deriv = mpmath.exp(_half_sq_diff(anchor_z, z) - log_z) * inv_sqrt_2pi
          next_offset = (
              offset - residual / deriv
              if mpmath.isfinite(residual)
              else offset
          )
          if (
              mpmath.isfinite(residual)
              and abs(next_offset - offset)
              <= max(abs(next_offset), abs(z)) * tol
          ):
            offset = next_offset
            break
          if not lo < next_offset < hi:
            if mpmath.isfinite(lo) and mpmath.isfinite(hi):
              next_offset = 0.5 * (lo + hi)
            elif mpmath.isfinite(hi):
              next_offset = hi - 2 * (1 + abs(hi))
            else:
              next_offset = lo + 2 * (1 + abs(lo))
          offset = next_offset
        else:
          raise RuntimeError(f"_mpmath_ppf did not converge: {q}, {a}, {b}")
    return +_exact_add(base, offset)


def _mpmath_isf(q, a, b, prob_ab=None):
  neg_ab = (
      None
      if prob_ab is None
      else (mpmath.fneg(prob_ab[0], exact=True), prob_ab[1])
  )
  with mpmath.workprec(_oracle_prec()):
    neg_b = mpmath.fneg(b, exact=True)
    neg_a = mpmath.fneg(a, exact=True)
    return mpmath.fneg(
        _mpmath_ppf(q, neg_b, neg_a, prob_ab=neg_ab), exact=True
    )


def _ref_f64(fn):
  """Wraps a JAX truncnorm function as a vectorized float64 reference.

  Large low-precision sweeps evaluate up to 2^20 inputs, which would be too slow
  in pure Python mpmath. SciPy loses accuracy on narrow intervals and in the
  tails even in float64, so these sweeps compare against JAX's float64 path,
  which is verified directly against mpmath in the `_float64` tests.
  """

  def wrapped(*args: np.ndarray) -> np.ndarray:
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(True), jax.default_device(cpu):
      jax_args = [jnp.asarray(a, dtype=jnp.float64) for a in args]
      return np.array(fn(*jax_args), dtype=np.float64, copy=True)

  return wrapped


_ref_ppf = _ref_f64(truncnorm.ppf)
_SWEEP_FNS = {
    "logpdf": (truncnorm.logpdf, _ref_f64(truncnorm.logpdf), _mpmath_logpdf),
    "pdf": (truncnorm.pdf, _ref_f64(truncnorm.pdf), _mpmath_pdf),
    "logcdf": (truncnorm.logcdf, _ref_f64(truncnorm.logcdf), _mpmath_logcdf),
    "cdf": (truncnorm.cdf, _ref_f64(truncnorm.cdf), _mpmath_cdf),
    "logsf": (truncnorm.logsf, _ref_f64(truncnorm.logsf), _mpmath_logsf),
    "sf": (truncnorm.sf, _ref_f64(truncnorm.sf), _mpmath_sf),
}


def _ppf_window_ignored(
    a: np.ndarray, width: float, q: float = 0.25
) -> np.ndarray:
  """Returns where `ppf(q, a, a + width)` is too ill-conditioned to test.

  Excludes two regimes from sweep comparisons:
    1. Outputs `z` near zero (`a < 0 < a + width`): here
       `kappa = q * P(a, b) / (phi(z) * |z|)` measures how much a relative error
       in `Phi(z)` is amplified in `z`. When `kappa > 4`, a 1-ULP error in
       `Phi(z) ~ 0.5` shifts `z` by many ULPs.
    2. Overflow: when `|a| > 0.25 * sqrt(fmax)`, `a**2` overflows in `float64`
       or in the reference formula.
  """
  dt = a.dtype.type
  a64 = a.astype(np.float64)
  b64 = (a + dt(width)).astype(np.float64)
  with np.errstate(all="ignore"):
    z = scipy.stats.truncnorm.ppf(q, a64, b64)
    prob_ab = scipy.special.ndtr(b64) - scipy.special.ndtr(a64)
    kappa = q * prob_ab / (scipy.stats.norm.pdf(z) * np.abs(z))
  is_z_near_zero = (a64 < 0.0) & (b64 > 0.0) & (kappa > 4.0)
  overflow_limit = np.sqrt(float(jnp.finfo(a.dtype).max)) * 0.25
  return is_z_near_zero | (np.isfinite(a) & (np.abs(a) > overflow_limit))


def _normal_underflow_x(dtype) -> float:
  """Returns |x| where exp(-x^2 / 2) crosses `tiny` (~13.2 f32, ~37.6 f64)."""
  return math.sqrt(-2.0 * math.log(float(jnp.finfo(dtype).tiny)))


def _outside_quad_upper_bound(a: float, scale: float = 1.02) -> float:
  """Returns `b > a` with `|b^2 - a^2| / 2 = scale * _QUAD_MAX_EXP_DIFF`.

  Requires `a**2 > 2 * scale * _QUAD_MAX_EXP_DIFF` when `a < 0`.
  """
  target = scale * _QUAD_MAX_EXP_DIFF
  if a < 0.0:
    return -math.sqrt(a**2 - 2.0 * target)
  return math.sqrt(a**2 + 2.0 * target)


# Window endpoints mirrored around 0: sub-ULP offsets (1e-20), crossovers
# between quadrature and erfcx scaling (0.125 through 2.0, straddling sqrt(2)
# and _QUAD_MAX_WIDTH = 1.48), and the tails where float32 and float64 Gaussian
# densities underflow (around 13.2 and 37.6).
_WINDOW_HALF = np.array([
    1e-20,
    0.125,
    0.5,
    1.0,
    1.5,
    2.0,
    8.0,
    10.0,
    math.floor(_normal_underflow_x(f32)),
    20.0,
    25.0,
    math.floor(_normal_underflow_x(f64)),
])
_WINDOW_POINTS = np.concatenate(
    (-_WINDOW_HALF[::-1], [0.0], _WINDOW_HALF)
).tolist()

# Uniform quantile grid combined with log-spaced points near 0, 0.5, and 1. The
# extra 1e-30 and 1e-20 points probe extreme tails below logspace(-12, -2).
_QUANTILE_EPS = np.logspace(-12, -2, 6)
_QUANTILE_POINTS = np.unique(
    np.concatenate((
        [1e-30, 1e-20],
        np.linspace(0.0, 1.0, 9),
        _QUANTILE_EPS,
        0.5 - _QUANTILE_EPS,
        0.5 + _QUANTILE_EPS,
        1.0 - _QUANTILE_EPS,
    ))
).tolist()


def _check_interval_sweep(
    tc,
    dtype,
    fn_name: str,
    a: float,
    b: float,
    bounds,
    extra_x: Sequence[float] = (),
):
  """Sweeps the evaluation point `x` across a fixed `[a, b]` interval."""
  dt = np.dtype(dtype).type
  a_dt, b_dt = dt(a), dt(b)
  a64, b64 = float(a_dt), float(b_dt)
  with mpmath.workprec(_PREC + 32):
    prob_ab = _mpmath_normal_prob(mpmath.mpf(a64), mpmath.mpf(b64))

  fn, f64_ref_fn, mp_fn = _SWEEP_FNS[fn_name]

  @jax.jit
  def jax_fn(x):
    return fn(x, jnp.asarray(a_dt, x.dtype), jnp.asarray(b_dt, x.dtype))

  def ref_fn(x64):
    return f64_ref_fn(x64, np.full_like(x64, a64), np.full_like(x64, b64))

  def mpmath_fn(x_mp):
    return mp_fn(x_mp, mpmath.mpf(a64), mpmath.mpf(b64), prob_ab=prob_ab)

  base_x = (
      [a64 + t * (b64 - a64) for t in _QUANTILE_POINTS]
      if np.isfinite(a64) and np.isfinite(b64)
      else _WINDOW_POINTS
  )
  util.check_unary_precision(
      tc,
      jax_fn,
      ref_fn,
      mpmath_fn,
      dtype,
      bounds=bounds,
      input_ftz=[("gpu", False)],
      check_signed_zeros=False,
      max_samples=4000 if dtype == f64 else None,
      interesting_points=[*extra_x, *base_x],
  )


# Coordinate naming used across test tables and helpers below:
#   x: evaluation coordinate in [a, b] (for pdf / logpdf / cdf / logsf / sf)
#   q: quantile probability in [0, 1] (for ppf / isf, inverse of cdf / sf)
#   u: either x or q in shared helpers
#   xab, qab, uab: (N, 3) sequences/arrays with columns (x|q|u, a, b)
#   ab, xy:        (N, 2) sequences/arrays with columns (a, b) or (x, y)
def _sample_intervals_xab(
    intervals: Sequence[tuple[float, float]],
    fracs: Sequence[float],
    mirror: bool = False,
) -> list[tuple[float, float, float]]:
  """Samples finite `[a, b]` intervals at interior fractions `t` in (0, 1)."""
  xab = [(a + t * (b - a), a, b) for a, b in intervals for t in fracs]
  return xab + [(-x, -b, -a) for x, a, b in xab] if mirror else xab


def _cartesian_qab(
    qs: Sequence[float],
    intervals: Sequence[tuple[float, float]],
    mirror: bool = False,
) -> list[tuple[float, float, float]]:
  """Returns the Cartesian product of quantiles `qs` and `[a, b]` intervals."""
  qab = [(q, a, b) for a, b in intervals for q in qs]
  return qab + [(1.0 - q, -b, -a) for q, a, b in qab] if mirror else qab


def _ulp_interval(a: float, num_ulps: int, dtype) -> tuple[float, float]:
  """Returns the interval `(a, a + num_ulps * ulp(a))` in `dtype`.

  Callers pass an odd `num_ulps` (such as 21) so that the midpoint lies strictly
  between two representable floating-point numbers.
  """
  dt = np.dtype(dtype).type
  return (float(a), float(a) + num_ulps * float(np.spacing(dt(a))))


def _special_log_xab(dtype):
  # Edge `(x, a, b)` cases with `x` in `[a, b]`: intervals only a few ULPs or
  # subnormal steps wide, narrow intervals in the quadrature and erfcx
  # regimes, and far-tail endpoints.
  tiny = float(jnp.finfo(dtype).tiny)
  mirrored_ab = [_ulp_interval(100.0, 21, dtype)]
  # (1.2, 1.2 + 1e-5) is a narrow non-dyadic quadrature interval.
  # (19.0, 19.35) sits past where exp(-a^2/2) underflows float32 (around 13.2)
  # and exceeds _QUAD_MAX_EXP_DIFF (about 6.7), so it uses erfcx despite its
  # width being below 1.48.
  # (0.0, 8 * tiny) tests a subnormal interval width near zero.
  interior_ab = [(1.2, 1.2 + 1e-5), (19.0, 19.35), (0.0, 8.0 * tiny)]
  # Just inside the points where the Gaussian density underflows in float32
  # (around 13.2) and float64 (around 37.6).
  unbounded_xs = [math.floor(_normal_underflow_x(f32)) - 1.0]
  extra_xab = [(3.5, 3.0, np.inf), (-3.5, -np.inf, -3.0)]
  if dtype == f64:
    mirrored_ab.append(_ulp_interval(1e4, 21, f64))
    # Narrow deep-tail erfcx intervals whose exponent difference a * width = 50.
    interior_ab.extend([
        (2000.0, 2000.0 + 50.0 / 2000.0),
        (-6000.0 - 50.0 / 6000.0, -6000.0),
        (8.0, 10.0),
    ])
    unbounded_xs.append(math.floor(_normal_underflow_x(f64)) - 1.0)
    ulp_1e7 = float(np.spacing(1e7))
    extra_xab.append(
        # Extreme tail: x lies within 1/a of a = 1e7 (CDF roughly 0.089).
        (1e7 + 5.0 * ulp_1e7, 1e7, np.inf),
    )
  unbounded_xab = [
      (x, a, b)
      for x in unbounded_xs
      for a, b in ((-np.inf, np.inf), (-5.0, 40.0))
  ]
  return [
      *_sample_intervals_xab(mirrored_ab, (0.25, 0.75), mirror=True),
      *_sample_intervals_xab(interior_ab, (0.25, 0.75)),
      *unbounded_xab,
      *extra_xab,
  ]


def _check_special_log_xab(tc, dtype, *fn_pairs):
  if dtype == f64 and jtu.device_under_test() == "tpu":
    tc.skipTest("float64 on TPU is ef57 double-double")
  x, a, b = jnp.array(_special_log_xab(dtype), dtype=dtype).T
  for fn, mp_fn in fn_pairs:
    util.check_precision_at_points(tc, fn, mp_fn, x, a, b, max_ulp=5.0)


# Upper bounds on ULP error for the sweep tests, shared per function family.
# The bfloat16 and float16 sweeps are exhaustive, and check_unary_precision
# interprets a plain number there as an exact integer ULP target (errors round
# up to integers). Because the harness never reports an error below 0.5 ULPs,
# passing `(0.5, x)` means "at most `x` ULPs" for those dtypes.
_ALL_PLATFORMS = ["cpu", "gpu", "tpu"]
_LOGPDF_BOUNDS = [(_ALL_PLATFORMS, {bf16: 0.5, f16: 0.5, f32: 2.0, f64: 2.5})]
_PDF_BOUNDS = [
    (_ALL_PLATFORMS, {bf16: (0.5, 1.0), f16: (0.5, 1.0), f32: 3.5, f64: 2.5})
]
_CDF_BOUNDS = [
    (_ALL_PLATFORMS, {bf16: (0.5, 1.0), f16: (0.5, 1.0), f32: 10.0, f64: 6.5})
]
_PPF_BOUNDS = [
    (_ALL_PLATFORMS, {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 8.0, f64: 6.5})
]


# Specific float32 worst-case inputs found during earlier randomized sweeps.
# Passing them via `extra_x` guarantees the high-precision sweeps include them,
# since uniform grids alone measure roughly 2-5x lower error in float32.
@jtu.thread_unsafe_test_class()
class TruncnormPdfTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logpdf_narrow(self, dtype):
    _check_interval_sweep(
        self, dtype, "logpdf", 1.25, 1.375, _LOGPDF_BOUNDS, extra_x=[1.37475598]
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logpdf_narrow_tail(self, dtype):
    _check_interval_sweep(
        self, dtype, "logpdf", 8.0, 8.125, _LOGPDF_BOUNDS, extra_x=[8.09145164]
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_pdf_tail(self, dtype):
    _check_interval_sweep(
        self, dtype, "pdf", -1.0, 37.0, _PDF_BOUNDS, extra_x=[12.7015352]
    )

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_logpdf_pdf_special_points(self, dtype):
    _check_special_log_xab(
        self,
        dtype,
        (truncnorm.logpdf, _mpmath_logpdf),
        (truncnorm.pdf, _mpmath_pdf),
    )


@jtu.thread_unsafe_test_class()
class TruncnormLogCdfTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logcdf_narrow(self, dtype):
    _check_interval_sweep(
        self, dtype, "logcdf", 1.25, 1.375, _CDF_BOUNDS, extra_x=[1.32752168]
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logcdf_wide(self, dtype):
    _check_interval_sweep(
        self,
        dtype,
        "logcdf",
        -1.0,
        2.0,
        _CDF_BOUNDS,
        extra_x=[0.414286822, 0.48219496],
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logcdf_tail(self, dtype):
    _check_interval_sweep(
        self, dtype, "logcdf", 8.0, 10.0, _CDF_BOUNDS, extra_x=[9.1207943]
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logcdf_wide_tail(self, dtype):
    _check_interval_sweep(
        self, dtype, "logcdf", -5.0, 40.0, _CDF_BOUNDS, extra_x=[5.78175163]
    )

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_logcdf_special_points(self, dtype):
    _check_special_log_xab(
        self,
        dtype,
        (truncnorm.logcdf, _mpmath_logcdf),
        (truncnorm.logsf, _mpmath_logsf),
    )


@jtu.thread_unsafe_test_class()
class TruncnormCdfSfTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cdf_tail(self, dtype):
    _check_interval_sweep(
        self, dtype, "cdf", -40.0, 5.0, _CDF_BOUNDS, extra_x=[-4.01144791]
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sf_unbounded(self, dtype):
    _check_interval_sweep(
        self, dtype, "sf", -np.inf, np.inf, _CDF_BOUNDS, extra_x=[2.8976419]
    )

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_cdf_sf_special_points(self, dtype):
    _check_special_log_xab(
        self, dtype, (truncnorm.cdf, _mpmath_cdf), (truncnorm.sf, _mpmath_sf)
    )


@jtu.thread_unsafe_test_class()
class TruncnormPpfQuantileTest(jtu.JaxTestCase):

  def _check_quantile_sweep(
      self, dtype, a: float, b: float, bounds=_PPF_BOUNDS
  ):
    dt = np.dtype(dtype).type
    a_dt, b_dt = dt(a), dt(b)
    a64, b64 = float(a_dt), float(b_dt)
    with mpmath.workprec(_PREC + 16):
      prob_ab = _mpmath_normal_prob(mpmath.mpf(a64), mpmath.mpf(b64))

    @jax.jit
    def jax_fn(q):
      return truncnorm.ppf(
          q, jnp.asarray(a_dt, q.dtype), jnp.asarray(b_dt, q.dtype)
      )

    def ref_fn(q64):
      return _ref_ppf(q64, np.full_like(q64, a64), np.full_like(q64, b64))

    def mpmath_fn(q_mp):
      return _mpmath_ppf(
          q_mp, mpmath.mpf(a64), mpmath.mpf(b64), prob_ab=prob_ab
      )

    util.check_unary_precision(
        self,
        jax_fn,
        ref_fn,
        mpmath_fn,
        dtype,
        bounds=bounds,
        input_ftz=[("gpu", False)],
        check_signed_zeros=False,
        max_samples=2500 if dtype == f64 else None,
        interesting_points=_QUANTILE_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_wide(self, dtype):
    self._check_quantile_sweep(dtype, 0.5, 2.0)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_narrow(self, dtype):
    self._check_quantile_sweep(dtype, 1.25, 1.375)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_tail(self, dtype):
    self._check_quantile_sweep(dtype, 8.0, 10.0)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_inf_bound(self, dtype):
    self._check_quantile_sweep(dtype, 3.0, np.inf)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_unbounded(self, dtype):
    self._check_quantile_sweep(dtype, -np.inf, np.inf)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_symmetric_wide(self, dtype):
    self._check_quantile_sweep(dtype, -2.0, 2.0)


@jtu.thread_unsafe_test_class()
class TruncnormPpfWindowTest(jtu.JaxTestCase):

  def _check_window_sweep(
      self, dtype, width: float, bounds=_PPF_BOUNDS, q: float = 0.25
  ):
    dt = np.dtype(dtype).type
    q_dt, width_dt = dt(q), dt(width)
    q64 = float(q_dt)

    @jax.jit
    def jax_fn(a):
      b = lax.optimization_barrier(a + jnp.asarray(width_dt, dtype=a.dtype))
      return truncnorm.ppf(jnp.asarray(q_dt, dtype=a.dtype), a, b)

    def ref_fn(a64):
      b64 = (a64.astype(dt) + width_dt).astype(np.float64)
      return _ref_ppf(np.full_like(a64, q64), a64, b64)

    def mpmath_fn(a_mp):
      b_mp = mpmath.mpf(float(dt(float(a_mp)) + width_dt))
      return _mpmath_ppf(mpmath.mpf(q64), a_mp, b_mp)

    util.check_unary_precision(
        self,
        jax_fn,
        ref_fn,
        mpmath_fn,
        dtype,
        bounds=bounds,
        ignore_inputs=lambda a: _ppf_window_ignored(a, width, q),
        input_ftz=[("gpu", False)],
        check_signed_zeros=False,
        max_samples=4000 if dtype == f64 else None,
        interesting_points=_WINDOW_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_narrow_window(self, dtype):
    self._check_window_sweep(dtype, 0.125)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_wide_window(self, dtype):
    # Just outside the kappa > 4 mask around z = 0, subtracting q * (Phi(b) -
    # Phi(a)) and the tail offset cancels up to roughly 2 bits (4x), amplifying
    # ~1 ULP in each term into about 5-6.5 ULPs in z.
    self._check_window_sweep(dtype, 2.0)


@jtu.thread_unsafe_test_class()
class TruncnormPpfSpecialPointsTest(jtu.JaxTestCase):

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_ppf_special_points(self, dtype):
    # Edge `(q, a, b)` cases: quantiles spanning 21-ULP and sub-ULP widths,
    # infinite bounds, near-median zero crossings, and tiny tail quantiles.
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    tiny = float(jnp.finfo(dtype).tiny)
    mirrored_ab = [(0.5, 2.0), (20.0, 25.0), (3.0, np.inf), (-0.1, 2.0)]
    sub_ulp = 1e-20  # Far smaller than eps (< 0.5 ULP of Phi(0)) in float32/64.
    extra_ab = [
        (1.2, 1.2 + 1e-5),
        # exp(-19^2 / 2) underflows float32, and the exponent difference (6.7)
        # exceeds _QUAD_MAX_EXP_DIFF.
        (19.0, 19.35),
        _ulp_interval(100.0, 21, dtype),
        (0.0, sub_ulp),
        # Asymmetric zero-straddling interval narrower than 1 ULP of Phi(0);
        # exercises _ppf_near_bound with a < 0 < b inside quadrature.
        (-sub_ulp, 2.0 * sub_ulp),
        (-0.5, 0.5),
        (-10.0, 12.0),
        (-10.0, np.inf),
        (-np.inf, np.inf),
    ]
    # q = tiny lies well past Gaussian density underflow (~13.2 float32, ~37.6
    # float64).
    tail_a = -20.0 if dtype == f32 else -40.0
    extra_qab = [(tiny, tail_a, 0.0)]
    if dtype == f64:
      # Deep left tail where exp(-a^2 / 2) is subnormal in float64 (~2e-319)
      # and the exponent difference sits just outside _QUAD_MAX_EXP_DIFF.
      deep_a = -38.3
      extra_ab.extend([
          (deep_a, _outside_quad_upper_bound(deep_a, scale=1.02)),
          _ulp_interval(1e6, 21, f64),
          (0.0, 1e-300),
      ])
    # q = 0.499 probes just below the median where preserving full accuracy in
    # q - 0.5 matters.
    qs = [0.1, 0.25, 0.499, 0.5, 0.75, 0.9]
    qab = [
        *_cartesian_qab(qs, mirrored_ab, mirror=True),
        *_cartesian_qab(qs, extra_ab),
        *extra_qab,
    ]
    # For q in {0.1, 0.9} on (-0.1, 2.0) and (-2.0, 0.1), z lands near 0 with
    # cancellation factor kappa around 4.35 (~7.6 ULPs in float64; isf swaps
    # bounds at the same q). Allow 8.0 ULPs for those four cases and 5.0 ULPs
    # everywhere else.
    crossing_qab = _cartesian_qab([0.1, 0.9], [(-0.1, 2.0)], mirror=True)
    crossing_set = set(crossing_qab)
    tight_qab = [row for row in qab if row not in crossing_set]
    for rows, max_ulp in ((tight_qab, 5.0), (crossing_qab, 8.0)):
      q, a, b = jnp.array(rows, dtype=dtype).T
      util.check_precision_at_points(
          self, truncnorm.ppf, _mpmath_ppf, q, a, b, max_ulp=max_ulp
      )
      util.check_precision_at_points(
          self, truncnorm.isf, _mpmath_isf, q, a, b, max_ulp=max_ulp
      )


def _check_grads_ulp(tc, fn, mp_grads, uab, dtype, max_ulp):
  """Checks JAX grad(fn) against mpmath partials at `(u, a, b)` rows."""
  if dtype == f64 and jtu.device_under_test() == "tpu":
    tc.skipTest("float64 on TPU is ef57 double-double")
  u, a, b = jnp.asarray(uab, dtype=dtype).T
  jax_grads = jax.jit(jax.vmap(jax.grad(fn, argnums=(0, 1, 2))))(u, a, b)
  for val, mp_grad in zip(jax_grads, mp_grads):
    util.check_precision_at_points(
        tc, lambda *_, out=val: out, mp_grad, u, a, b, max_ulp=max_ulp
    )


def _eval_hessians(fn, uab, dtype):
  uab = jnp.asarray(uab, dtype=dtype)
  hess_fn = lambda v: jax.jacrev(jax.grad(lambda p: fn(p[0], p[1], p[2])))(v)
  return jax.jit(jax.vmap(hess_fn))(uab)


def _check_hessian_ulp(tc, jax_hess, mp_hess, uab, dtype, max_ulp):
  """Checks precomputed JAX Hessians against mpmath at `(u, a, b)` rows."""
  u, a, b = jnp.asarray(uab, dtype=dtype).T
  for i in range(3):
    for j in range(i, 3):
      val = jax_hess[:, i, j]
      util.check_precision_at_points(
          tc,
          lambda *_, out=val: out,
          lambda *args, i=i, j=j: mp_hess(*args)[i][j],
          u,
          a,
          b,
          max_ulp=max_ulp,
      )


def _check_single_tangent_zeros(tc, fn, uab, dtype):
  """Checks zero-tangent handling: all-zero tangents and single-input JVPs."""
  u, a, b = jnp.asarray(uab, dtype=dtype).T
  loss = lambda *args: jnp.sum(fn(*args))
  full_grad = jax.jit(jax.grad(loss, (0, 1, 2)))(u, a, b)
  zeros = (jnp.zeros_like(u),) * 3
  _, zero_jvp = jax.jit(lambda *args: jax.jvp(fn, args, zeros))(u, a, b)
  tc.assertArraysEqual(zero_jvp, jnp.zeros_like(u))
  tol = 1e-5 if dtype == f32 else 1e-12
  for k in range(3):
    single_grad = jax.jit(jax.grad(loss, k))(u, a, b)
    tc.assertArraysEqual(single_grad, full_grad[k])
    eye = tuple(
        jnp.ones_like(u) if j == k else jnp.zeros_like(u) for j in range(3)
    )
    _, jvp_k = jax.jit(lambda *args, e=eye: jax.jvp(fn, args, e))(u, a, b)
    tc.assertAllClose(jvp_k, full_grad[k], rtol=tol, atol=tol)


def _mp_partial(mp_fn, orders):
  """Builds an mpmath mixed partial derivative of a three-argument function.

  `orders` is a 3-tuple such as `(1, 0, 1)`, meaning d^2 f / (dx_0 dx_2).
  Uses `mpmath.diff` (high-precision finite differences) for generality.
  """

  def fn(*args):
    active = [(i, v) for i, (o, v) in enumerate(zip(orders, args)) if o]
    if any(mpmath.isinf(v) for _, v in active):
      return mpmath.mpf(0)

    def slice_fn(*vals):
      full = list(args)
      for (i, _), val in zip(active, vals):
        full[i] = val
      return mp_fn(*full)

    with mpmath.workprec(200):
      active_points = tuple(v for _, v in active)
      active_orders = tuple(orders[i] for i, _ in active)
      if len(active) == 1:
        return +mpmath.diff(slice_fn, active_points[0], n=active_orders[0])
      return +mpmath.diff(slice_fn, active_points, active_orders)

  return fn


def _mp_grads(mp_fn):
  return tuple(
      _mp_partial(mp_fn, tuple(1 if j == k else 0 for j in range(3)))
      for k in range(3)
  )


def _mp_hessian(mp_fn):
  cache = {}
  hess_fns = {
      (i, j): _mp_partial(
          mp_fn,
          tuple((1 if k == i else 0) + (1 if k == j else 0) for k in range(3)),
      )
      for i in range(3)
      for j in range(i, 3)
  }

  def hess(x, a, b):
    key = (x._mpf_, a._mpf_, b._mpf_)
    if key not in cache:
      rows = [[None] * 3 for _ in range(3)]
      for (i, j), fn in hess_fns.items():
        val = fn(x, a, b)
        rows[i][j] = val
        rows[j][i] = val
      cache[key] = tuple(tuple(row) for row in rows)
    return cache[key]

  return hess


def _check_density_or_cdf_second_order_rev_grads(tc, fn_names, dtype):
  """Checks 2nd-order reverse-mode Hessians for density/CDF vs mpmath."""
  if dtype == f64 and jtu.device_under_test() == "tpu":
    tc.skipTest("float64 on TPU is ef57 double-double")
  ulp_100 = float(np.spacing(np.dtype(dtype).type(100.0)))
  narrow_xab = (100.0 + 5.0 * ulp_100, *_ulp_interval(100.0, 21, dtype))
  finite_ab = [
      (0.5, 2.0),
      (1.25, 1.375),
      (8.0, 10.0),
      (19.0, 19.35),
      (-1.0, 2.0),
  ]
  inf_tail_xab = [(3.5, 3.0, np.inf)]
  common_xab = [
      # Test both left- and right-half regimes across standard, quadrature, and
      # mirrored/erfcx-scaled tail intervals.
      *_sample_intervals_xab(finite_ab, (0.2, 0.85), mirror=True),
      *inf_tail_xab,
      *[(-x, -b, -a) for x, a, b in inf_tail_xab],
      (5.0, -np.inf, np.inf),
  ]
  density_xab = [*common_xab, narrow_xab]
  # Exclude narrow_xab from the relative CDF Hessian ULP check; those points
  # are verified via assertAllClose (relative + absolute tolerance) below.
  cdf_xab = [*common_xab, (100.01, 100.0, 100.03)]

  specs = {
      "logpdf": (truncnorm.logpdf, _mp_hessian(_mpmath_logpdf), density_xab),
      "pdf": (truncnorm.pdf, _mp_hessian(_mpmath_pdf), density_xab),
      "logcdf": (truncnorm.logcdf, _mp_hessian(_mpmath_logcdf), cdf_xab),
      "cdf": (truncnorm.cdf, _mp_hessian(_mpmath_cdf), cdf_xab),
  }
  max_ulp = 256.0
  mp_narrow = tuple(mpmath.mpf(v) for v in narrow_xab)
  for fn_name in fn_names:
    fn, mp_hess, fn_xab = specs[fn_name]
    is_cdf = fn_name in ("logcdf", "cdf")
    all_xab = [*fn_xab, narrow_xab] if is_cdf else fn_xab
    jax_hess = _eval_hessians(fn, all_xab, dtype)
    num_ulp = len(fn_xab)
    _check_hessian_ulp(
        tc, jax_hess[:num_ulp], mp_hess, fn_xab, dtype, max_ulp
    )
    if is_cdf:
      # On [100, 100 + 21 * ulp], P(a, z) + P(z, b) adds roughly phi(z) and
      # -phi(z), amplifying roundoff by 1 / (x * width). The off-diagonal
      # entries, whose true values are zero, stay within 1 ULP of max|H|
      # (around 6.7e7).
      ref_hess = np.array(
          [[float(v) for v in row] for row in mp_hess(*mp_narrow)], dtype=dtype
      )
      rtol = 1e-4 if dtype == f32 else 1e-5
      atol = max(
          1e-6,
          float(np.max(np.abs(ref_hess))) * (1e-6 if dtype == f32 else 1e-8),
      )
      tc.assertAllClose(jax_hess[-1], ref_hess, rtol=rtol, atol=atol)
  if dtype == f64:
    # Test zero-tangent handling on one representative function per custom_jvp
    # (logpdf and cdf).
    rep_name = "logpdf" if "logpdf" in fn_names else "cdf"
    _check_single_tangent_zeros(
        tc, specs[rep_name][0], specs[rep_name][2][:5], dtype
    )


@jtu.thread_unsafe_test_class()
class TruncnormDensityGradTest(jtu.JaxTestCase):
  """Checks partial derivatives of density/CDF functions vs mpmath."""

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_density_and_cdf_grads(self, dtype):
    ulp_100 = float(np.spacing(np.dtype(dtype).type(100.0)))
    # Sample finite intervals near both endpoints so we cross the midpoint
    # split (`is_left`) and hit mixed quadrature/erfcx branches; asymmetric and
    # tail intervals are also mirrored across zero.
    mirrored_xab = _sample_intervals_xab(
        [(8.0, 10.0), (19.0, 19.35)], (0.15, 0.85), mirror=True
    )
    interior_xab = _sample_intervals_xab(
        [(0.5, 2.0), (1.25, 1.375), (-1.0, 2.0), (-0.1, 0.1), (-2.0, 2.0)],
        # Including 0.35 puts points between 0 and the midpoint on (-1.0, 2.0),
        # exercising mixed quadrature and normal-CDF splits.
        (0.2, 0.35, 0.5, 0.8),
    )
    inf_tail_xab = [(3.5, 3.0, np.inf)]
    xab = [
        *mirrored_xab,
        *interior_xab,
        *inf_tail_xab,
        *[(-x, -b, -a) for x, a, b in inf_tail_xab],
        (100.0 + 5.0 * ulp_100, *_ulp_interval(100.0, 21, dtype)),
        # Keep x = 5.0 within float32 density range on the wide [-5, 40] span.
        (5.0, -5.0, 40.0),
        (5.0, -np.inf, np.inf),
    ]

    for fn, mp_fn, max_ulp in (
        (truncnorm.logpdf, _mpmath_logpdf, 2.0),
        (truncnorm.pdf, _mpmath_pdf, 4.5),
        (truncnorm.logcdf, _mpmath_logcdf, 4.5 if dtype == f32 else 3.0),
        (truncnorm.logsf, _mpmath_logsf, 4.5 if dtype == f32 else 3.0),
        (truncnorm.cdf, _mpmath_cdf, 4.5 if dtype == f32 else 3.5),
        (truncnorm.sf, _mpmath_sf, 4.5 if dtype == f32 else 3.5),
    ):
      _check_grads_ulp(self, fn, _mp_grads(mp_fn), xab, dtype, max_ulp)

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_density_second_order_rev_grads(self, dtype):
    _check_density_or_cdf_second_order_rev_grads(self, ("logpdf", "pdf"), dtype)


@jtu.thread_unsafe_test_class()
class TruncnormCdfGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_cdf_second_order_rev_grads(self, dtype):
    _check_density_or_cdf_second_order_rev_grads(self, ("logcdf", "cdf"), dtype)


@jtu.thread_unsafe_test_class()
class TruncnormQuantileGradTest(jtu.JaxTestCase):
  """Checks partial derivatives of ppf and isf vs mpmath."""

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_ppf_and_isf_grads(self, dtype):
    q_grid = [1e-5, 0.25, 0.5, 0.75, 1.0 - 1e-5]
    mirrored_ab = [
        (0.5, 2.0),
        (1.25, 1.375),
        (8.0, 10.0),
        (19.0, 19.35),
        (-0.1, 2.0),
        (3.0, np.inf),
    ]
    sym_ab = [
        (-0.5, 0.5),
        (-2.0, 2.0),
        # Straddle sqrt(2), where half-erf hands off to tail erfcx.
        (-(math.sqrt(2) - 0.005), math.sqrt(2) - 0.005),
        (-(math.sqrt(2) + 0.005), math.sqrt(2) + 0.005),
        (-np.inf, np.inf),
    ]
    qab = [
        *_cartesian_qab(q_grid, mirrored_ab, mirror=True),
        *_cartesian_qab(q_grid, sym_ab),
        (1e-6, 0.5, 2.0),
        (1.0 - 1e-6, 0.5, 2.0),
        (0.75, *_ulp_interval(100.0, 21, dtype)),
        (1e-20, -np.inf, np.inf),
        (1e-20, -20.0, 1.0),
    ]
    if dtype == f64:
      qab.extend([
          (1e-300, -np.inf, np.inf),
          (1e-300, -40.0, 1.0),
      ])

    # Evaluate the implicit-function partial derivatives in closed form
    # (dz/dq = 1/pdf, dz/da = (1 - q) * exp((z^2 - a^2) / 2), etc.) rather than
    # using finite-difference mpmath.diff: in extreme tails such as
    # (1e-20, -20, 1), dz/da is around 1e-69, so resolving z(a + h) - z(a - h)
    # would require hundreds of extra bits of precision.
    ppf_cache = {}

    def _solve_ppf(q, a, b):
      key = (q._mpf_, a._mpf_, b._mpf_)
      if key not in ppf_cache:
        with mpmath.workprec(_PREC + 32):
          prob_ab = _mpmath_normal_prob(a, b)
          z = _mpmath_ppf(q, a, b, prob_ab=prob_ab)
          ppf_cache[key] = (prob_ab, z)
      return ppf_cache[key]

    def mp_ppf_dq(q, a, b):
      with mpmath.workprec(_PREC + 32):
        prob_ab, z = _solve_ppf(q, a, b)
        return +(1 / _mpmath_pdf(z, a, b, prob_ab=prob_ab))

    def mp_ppf_da(q, a, b):
      if mpmath.isinf(a):
        return mpmath.mpf(0)
      with mpmath.workprec(_PREC + 32):
        _, z = _solve_ppf(q, a, b)
        return +((1 - q) * mpmath.exp(_half_sq_diff(z, a)))

    def mp_ppf_db(q, a, b):
      if mpmath.isinf(b):
        return mpmath.mpf(0)
      with mpmath.workprec(_PREC + 32):
        _, z = _solve_ppf(q, a, b)
        return +(q * mpmath.exp(_half_sq_diff(z, b)))

    def mp_isf_dq(q, a, b):
      neg_b, neg_a = mpmath.fneg(b, exact=True), mpmath.fneg(a, exact=True)
      return -mp_ppf_dq(q, neg_b, neg_a)

    def mp_isf_da(q, a, b):
      neg_b, neg_a = mpmath.fneg(b, exact=True), mpmath.fneg(a, exact=True)
      return mp_ppf_db(q, neg_b, neg_a)

    def mp_isf_db(q, a, b):
      neg_b, neg_a = mpmath.fneg(b, exact=True), mpmath.fneg(a, exact=True)
      return mp_ppf_da(q, neg_b, neg_a)

    max_ulp = 4.0
    _check_grads_ulp(
        self,
        truncnorm.ppf,
        (mp_ppf_dq, mp_ppf_da, mp_ppf_db),
        qab,
        dtype,
        max_ulp,
    )
    _check_grads_ulp(
        self,
        truncnorm.isf,
        (mp_isf_dq, mp_isf_da, mp_isf_db),
        qab,
        dtype,
        max_ulp,
    )


@jtu.thread_unsafe_test_class()
class TruncnormQuantileSecondOrderGradTest(jtu.JaxTestCase):
  """Checks 2nd-order reverse-mode Hessians of ppf vs mpmath."""

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_ppf_second_order_rev_grads(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    # Pick tiny_q below 2^-64 in float32 (2^-256 in float64) inside [-25, 25]
    # so _FactoredExp.scale_by rescaling triggers while z * dz/dq^2 stays
    # finite and away from both bounds.
    tiny_q = 4e-20 if dtype == f32 else 1e-100
    mirrored_ab = [
        (0.5, 2.0),
        (1.25, 1.375),
        (8.0, 10.0),
        (19.0, 19.35),
        _ulp_interval(100.0, 21, dtype),
        (-2.0, 0.1),
        (3.0, np.inf),
    ]
    sym_ab = [(-2.0, 2.0), (-np.inf, np.inf)]
    qs = [0.05, 0.15, 0.25, 0.75, 0.85]
    qab = [
        *_cartesian_qab(qs, mirrored_ab, mirror=True),
        *_cartesian_qab(qs, sym_ab),
        (tiny_q, -25.0, 25.0),
    ]
    cache = {}

    def _ppf_hess(q, a, b):
      key = (q._mpf_, a._mpf_, b._mpf_)
      if key not in cache:
        with mpmath.workprec(_PREC + 32):
          prob = _mpmath_normal_prob(a, b)
          z = _mpmath_ppf(q, a, b, prob_ab=prob)
          anchor_ab, log_ab = prob
          # First derivatives: z_q = P / phi(z),
          # z_a = (1 - q) * phi(a) / phi(z), and z_b = q * phi(b) / phi(z).
          # Differentiating a second time with (1 / phi(z))' = z / phi(z) and
          # phi'(u) = -u * phi(u) gives the Hessian entries h_qq through h_bb.
          zero = mpmath.mpf(0)
          is_inf_a, is_inf_b = mpmath.isinf(a), mpmath.isinf(b)
          z_q = mpmath.sqrt(2 * mpmath.pi) * mpmath.exp(
              _half_sq_diff(z, anchor_ab) + log_ab
          )
          ratio_a = zero if is_inf_a else mpmath.exp(_half_sq_diff(z, a))
          ratio_b = zero if is_inf_b else mpmath.exp(_half_sq_diff(z, b))
          z_a, z_b = (1 - q) * ratio_a, q * ratio_b
          h_qq = +(z * z_q**2)
          h_qa = +(zero if is_inf_a else ratio_a * ((1 - q) * z * z_q - 1))
          h_qb = +(zero if is_inf_b else ratio_b * (1 + q * z * z_q))
          h_aa = +(zero if is_inf_a else z_a * (z * z_a - a))
          h_ab = +(zero if (is_inf_a or is_inf_b) else z * z_a * z_b)
          h_bb = +(zero if is_inf_b else z_b * (z * z_b - b))
          cache[key] = (
              (h_qq, h_qa, h_qb),
              (h_qa, h_aa, h_ab),
              (h_qb, h_ab, h_bb),
          )
      return cache[key]

    near_a_qab = (1e-100, -20.0, 1.0)
    all_qab = [*qab, near_a_qab] if dtype == f64 else qab
    jax_hess = _eval_hessians(truncnorm.ppf, all_qab, dtype)
    # In the right tail at small q, (1 - q) * z * dz/dq - 1 cancels near 1
    # (~6380 ULPs in float32 at (0.05, 19.0, 19.35) and ~2382 ULPs in float64
    # at (0.05, 8.0, 10.0)), while z * dz/da - a cancels roughly 8000x at
    # (0.25, 8.0, 10.0).
    max_ulp = 8192.0 if dtype == f32 else 4096.0
    num_ulp = len(qab)
    _check_hessian_ulp(
        self, jax_hess[:num_ulp], _ppf_hess, qab, dtype, max_ulp
    )
    if dtype == f64:
      _check_single_tangent_zeros(self, truncnorm.ppf, qab[:5], dtype)
      # At (q = 1e-100, -20, 1), Phi(a) is so much larger than q * P that
      # z * dz/da - a cancels by roughly 38 bits (about 3e-5 relative error in
      # float64).
      mp_near_a = tuple(mpmath.mpf(v) for v in near_a_qab)
      ref_hess = np.array(
          [[float(v) for v in row] for row in _ppf_hess(*mp_near_a)],
          dtype=dtype,
      )
      self.assertAllClose(jax_hess[-1], ref_hess, rtol=1e-4)


class _MpJet:
  """Second-order forward-mode tuple `(val, grad, hess)` in mpmath.

  Used to compute reference gradients and Hessians for `_FactoredExp.scale_by`
  and `_DoubleWord.mul` at tiny scales where finite differences in
  `mpmath.diff` cannot resolve the perturbation.
  """

  def __init__(self, val, grad, hess):
    self.val = val
    self.grad = grad
    self.hess = hess

  @classmethod
  def var(cls, val, idx, n):
    grad = [mpmath.mpf(1 if i == idx else 0) for i in range(n)]
    hess = [[mpmath.mpf(0)] * n for _ in range(n)]
    return cls(val, grad, hess)

  def __add__(self, other):
    idxs = range(len(self.grad))
    return _MpJet(
        self.val + other.val,
        [self.grad[i] + other.grad[i] for i in idxs],
        [[self.hess[i][j] + other.hess[i][j] for j in idxs] for i in idxs],
    )

  def __mul__(self, other):
    idxs = range(len(self.grad))
    val_x, grad_x, hess_x = self.val, self.grad, self.hess
    val_y, grad_y, hess_y = other.val, other.grad, other.hess
    return _MpJet(
        val_x * val_y,
        [grad_x[i] * val_y + val_x * grad_y[i] for i in idxs],
        [
            [
                hess_x[i][j] * val_y
                + grad_x[i] * grad_y[j]
                + grad_y[i] * grad_x[j]
                + val_x * hess_y[i][j]
                for j in idxs
            ]
            for i in idxs
        ],
    )

  def exp(self):
    idxs = range(len(self.grad))
    grad_x, hess_x = self.grad, self.hess
    exp_val = mpmath.exp(self.val)
    return _MpJet(
        exp_val,
        [exp_val * grad_x[i] for i in idxs],
        [
            [exp_val * (hess_x[i][j] + grad_x[i] * grad_x[j]) for j in idxs]
            for i in idxs
        ],
    )

  def log(self):
    idxs = range(len(self.grad))
    val_x, grad_x, hess_x = self.val, self.grad, self.hess
    log_hess = [
        [(hess_x[i][j] - grad_x[i] * grad_x[j] / val_x) / val_x for j in idxs]
        for i in idxs
    ]
    return _MpJet(
        mpmath.log(val_x), [grad_x[i] / val_x for i in idxs], log_hess
    )


@jtu.thread_unsafe_test_class()
class TruncnormDtypeTest(jtu.JaxTestCase):

  @parameterized.parameters(
      "logpdf", "pdf", "logcdf", "cdf", "logsf", "sf", "ppf", "isf"
  )
  @config.explicit_x64_dtypes("allow")
  def test_explicit_x64_dtypes(self, fn_name):
    """Verifies that explicit float64 inputs stay float64 with x64 disabled."""
    if jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    fn = getattr(truncnorm, fn_name)
    a = jnp.array([-1.0, 1.25, 0.5, 8.0], dtype=jnp.float64)
    b = jnp.array([1.5, 1.375, 0.625, 10.0], dtype=jnp.float64)
    u = jnp.array(
        [0.3, 0.25, 0.02, 0.7]
        if fn_name in ("ppf", "isf")
        else [0.2, 1.3, 0.55, 8.5],
        dtype=jnp.float64,
    )
    grad_fn = jax.grad(lambda u_, a_, b_: jnp.sum(fn(u_, a_, b_)), (0, 1, 2))
    eval_fn = jax.jit(lambda u_, a_, b_: (fn(u_, a_, b_), *grad_fn(u_, a_, b_)))
    with jax.enable_x64(True):
      expected = eval_fn(u, a, b)
    with jax.enable_x64(False):
      actual = eval_fn(u, a, b)
    for actual_val, expected_val in zip(actual, expected):
      self.assertEqual(actual_val.dtype, jnp.float64)
      self.assertArraysEqual(actual_val, expected_val)

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_internal_ad_and_factored_exp_grads(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    # 1. Custom JVP for _half_sq_diff: cancellation, zero tangents, Hessians,
    # and overflow.
    ulp_100 = float(np.spacing(np.dtype(dtype).type(100.0)))
    huge = 1e20 if dtype == f32 else 1e160
    huge_step = 1e10 if dtype == f32 else 1e140
    xy = jnp.array(
        [
            (3.5, -1.25),
            (100.0, 100.0 + 5.0 * ulp_100),
            (0.0, 2.5),
            (huge, huge - huge_step),
        ],
        dtype=dtype,
    )
    x, y = xy.T
    ones, zeros = jnp.ones_like(x), jnp.zeros_like(x)
    for k, expected_grad, tangent in (
        (0, x, (ones, zeros)),
        (1, -y, (zeros, ones)),
    ):
      grad_k = jax.jit(jax.vmap(jax.grad(_jax_half_sq_diff, argnums=k)))(x, y)
      _, jvp_k = jax.jit(
          lambda x_, y_, t=tangent: jax.jvp(_jax_half_sq_diff, (x_, y_), t)
      )(x, y)
      self.assertArraysEqual(grad_k, expected_grad)
      self.assertArraysEqual(jvp_k, expected_grad)
    hess_fn = jax.jit(
        jax.vmap(jax.jacrev(jax.grad(lambda p: _jax_half_sq_diff(p[0], p[1]))))
    )
    expected_hess = np.broadcast_to(
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=dtype), (len(x), 2, 2)
    )
    self.assertArraysEqual(hess_fn(xy), expected_hess)

    overflow_xy = jnp.array([(huge, 0.0), (0.0, -huge)], dtype=dtype)
    overflow_x, overflow_y = overflow_xy.T
    self.assertTrue(
        bool(jnp.all(jnp.isinf(_jax_half_sq_diff(overflow_x, overflow_y))))
    )
    overflow_grad = jax.jit(
        jax.grad(lambda x_, y_: jnp.sum(_jax_half_sq_diff(x_, y_)), (0, 1))
    )(overflow_x, overflow_y)
    self.assertArraysEqual(
        overflow_grad, (jnp.zeros_like(overflow_x), jnp.zeros_like(overflow_y))
    )
    self.assertArraysEqual(
        hess_fn(overflow_xy), jnp.zeros((2, 2, 2), dtype=dtype)
    )

    # 2. _FactoredExp.scale_by and _DoubleWord.mul stop_gradient rules at first
    # and second order.
    scaled = lambda p: _FactoredExp(p[1], _DoubleWord.mul(p[1], p[2])).scale_by(
        p[0]
    )
    scaled_val = lambda p: scaled(p).value()
    scaled_log = lambda p: scaled(p).log_value()

    def _check_scale(u_val, b_val, check_log_hess):
      params = jnp.array([u_val, 1.5, b_val], dtype=dtype)
      with mpmath.workprec(_PREC + 32):
        u, a, b = (
            _MpJet.var(mpmath.mpf(v), i, 3)
            for i, v in enumerate((u_val, 1.5, b_val))
        )
        ua, ab = u * a, a * b
        ref_val = ua * ab.exp()
        ref_log = ua.log() + ab
      rtol = 1e-5 if dtype == f32 else 1e-12
      to_dtype = lambda x: np.array(x, dtype=float).astype(dtype)
      for fn, ref, check_hess in (
          (scaled_val, ref_val, True),
          (scaled_log, ref_log, check_log_hess),
      ):
        self.assertAllClose(
            jax.jit(jax.grad(fn))(params), to_dtype(ref.grad), rtol=rtol
        )
        if check_hess:
          # Computing d^2/du da log(scaled * a) subtracts +1/(u*a) and -1/(u*a),
          # leaving O(eps / (u * a)) roundoff in the off-diagonal entries.
          hess_atol = (
              float(np.spacing(np.float64(1.0 / (u_val * 1.5)))) * 8.0
              if fn is scaled_log
              else 0.0
          )
          self.assertAllClose(
              jax.jit(jax.jacrev(jax.grad(fn)))(params),
              to_dtype(ref.hess),
              rtol=rtol,
              atol=hess_atol,
          )

    if dtype == f32:
      # At u = 1e-20 (< 2^-64), -1 / u^2 overflows float32. Check the value
      # gradient, value Hessian, and log gradient, omitting -1 / u^2 from the
      # log Hessian check.
      _check_scale(1e-20, 20.0, check_log_hess=False)
    else:
      _check_scale(1e-100, 150.0, check_log_hess=True)
      # Exercise the 2^-512 rescaling threshold without overflowing 1 / u^2.
      _check_scale(1e-160, 240.0, check_log_hess=False)

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_underflow_positive_zero(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    # The low-word correction reaches 1 + exponent.lo <= 0 once
    # x^2 / 2 >= 2 / eps, that is, x >= 2 / sqrt(eps) (about 5793 in float32,
    # 1.34e8 in float64).
    lo = 2.0 / math.sqrt(float(jnp.finfo(dtype).eps))
    x = jnp.linspace(lo, 4.0 * lo, 64, dtype=dtype)
    a = jnp.zeros((), dtype=dtype)
    b = jnp.asarray(np.inf, dtype=dtype)
    for fn in (truncnorm.pdf, truncnorm.sf):
      y = fn(x, a, b)
      self.assertArraysEqual(y, jnp.zeros_like(y))
      self.assertFalse(bool(jnp.any(jnp.signbit(y))))

  @parameterized.parameters(
      "logpdf", "pdf", "logcdf", "cdf", "logsf", "sf", "ppf", "isf"
  )
  def test_complex_inputs_raise(self, fn_name):
    fn = getattr(truncnorm, fn_name)
    with self.assertRaisesRegex(TypeError, "complex-valued inputs"):
      fn(jnp.complex64(0.5), -1.0, 1.0)


@jtu.thread_unsafe_test_class()
class TruncnormConstantsTest(jtu.JaxTestCase):

  def test_truncnorm_constants(self):
    """Verifies Taylor and quadrature constants in `truncnorm.py`."""
    frac = fractions.Fraction

    def round_frac(f, prec):
      """Rounds a positive Fraction to `prec` significant bits, as a float."""
      exp = f.numerator.bit_length() - f.denominator.bit_length()
      if f < frac(1 << max(exp, 0), 1 << max(-exp, 0)):
        exp -= 1
      shift = prec - 1 - exp
      scaled = f * (1 << shift) if shift >= 0 else f / (1 << -shift)
      return math.ldexp(round(scaled), -shift)

    # 1. Maclaurin coefficients of z'(t) = exp(z^2 / 2), z(0) = 0, via
    # z''(t) = z(t) * z'(t)^2 with z(t) = sum_{k>=0} coeffs_k * t^{2k+1}.
    coeffs, derivs, deriv_sq = [frac(1)], [frac(1)], []
    for k in range(1, len(_NEAR_ZERO_COEFFS_F64) + 1):
      deriv_sq.append(sum(derivs[i] * derivs[k - 1 - i] for i in range(k)))
      coeff_k = sum(
          coeffs[j] * deriv_sq[k - 1 - j] for j in range(k)
      ) / (2 * k * (2 * k + 1))
      coeffs.append(coeff_k)
      derivs.append((2 * k + 1) * coeff_k)
    num_f32 = len(_NEAR_ZERO_COEFFS_F32)
    self.assertEqual(
        _NEAR_ZERO_COEFFS_F64, tuple(float(c) for c in coeffs[1:])
    )
    self.assertEqual(
        tuple(float(np.float32(c)) for c in _NEAR_ZERO_COEFFS_F32),
        tuple(round_frac(c, 24) for c in coeffs[1 : num_f32 + 1]),
    )

    # 2. Bivariate series reversion for delta'(u) = exp(a * delta + delta^2 / 2)
    # with delta(0) = 0 and u = prob_over_pdf, using delta''(u) =
    # (a + delta) * delta'(u)^2. `series[n][j]` holds the coefficient of
    # u^n * a^j; rewriting as u^(2m + 1) * (a * u)^j (where n - j = 2m + 1)
    # yields _NEAR_BOUND_COEFFS.
    series, deriv_sq_terms = {1: [frac(1)]}, {}
    for n in range(2, 8):
      m = n - 2
      deriv_sq_terms[m] = [frac(0)] * (m + 1)
      for i in range(m + 1):
        for j1, c1 in enumerate(series[i + 1]):
          for j2, c2 in enumerate(series[m - i + 1]):
            deriv_sq_terms[m][j1 + j2] += (i + 1) * (m - i + 1) * c1 * c2
      coeffs_n = [frac(0)] * n
      for j, c in enumerate(deriv_sq_terms[m]):
        coeffs_n[j + 1] += c
      for k in range(1, m + 1):
        for j1, c1 in enumerate(series[k]):
          for j2, c2 in enumerate(deriv_sq_terms[m - k]):
            coeffs_n[j1 + j2] += c1 * c2
      series[n] = [c / (n * (n - 1)) for c in coeffs_n]
    expected_near_bound = tuple(
        tuple(float(series[2 * m + 1 + j][j]) for j in range(len(row)))
        for m, row in enumerate(_NEAR_BOUND_COEFFS)
    )
    self.assertEqual(_NEAR_BOUND_COEFFS, expected_near_bound)

    # 3. 4-point ([0, 1]) Gauss-Legendre nodes and weights.
    def gl_rule(n):
      nodes, weights = [], []
      with mpmath.workprec(200):
        for k in range(n):
          x = mpmath.cos(mpmath.pi * (n - k - mpmath.mpf("0.25")) / (n + 0.5))
          for _ in range(8):
            p0, p1 = mpmath.mpf(1), x
            for j in range(1, n):
              p0, p1 = p1, ((2 * j + 1) * x * p1 - j * p0) / (j + 1)
            dp = n * (x * p1 - p0) / (x * x - 1)
            x -= p1 / dp
          nodes.append(x)
          weights.append(2 / ((1 - x * x) * dp * dp))
      return nodes, weights

    nodes_gl4, weights_gl4 = gl_rule(4)
    self.assertEqual(_GL4_NODES, tuple(float((x + 1) / 2) for x in nodes_gl4))
    self.assertEqual(_GL4_WEIGHTS, tuple(float(w / 2) for w in weights_gl4))

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_quadrature_range_accuracy(self, dtype):
    """Checks _normal_prob_quad accuracy across the _use_quad domain."""
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    tiny = float(jnp.finfo(dtype).tiny)
    # Width and exponent-difference limits trade off at a = 4 / 1.48 - 0.74
    # (about 1.96), so a = 1.5 is width-limited while 2.0 through 100.0 are
    # exponent-difference-limited.
    quad_anchors = (0.0, 1.5, 2.0, 5.0, 10.0, 100.0)
    base_ab = []
    for anchor in quad_anchors:
      max_width = min(
          _QUAD_MAX_WIDTH,
          2.0
          * _QUAD_MAX_EXP_DIFF
          / (anchor + math.sqrt(anchor**2 + 2.0 * _QUAD_MAX_EXP_DIFF)),
      )
      # Pull back slightly from the _use_quad boundary so dtype rounding keeps
      # the exponent difference at or below _QUAD_MAX_EXP_DIFF.
      scale = 1.0 if anchor == 0.0 else 0.99
      base_ab.append((anchor, anchor + scale * max_width))
    quad_half = 0.49 * _QUAD_MAX_WIDTH
    ab = jnp.array(
        [
            *base_ab,
            *[(-b, -a) for a, b in base_ab],
            (-quad_half, quad_half),
            (1.25, 1.25 + 1e-6),
            (0.0, 1e-20),
            (0.0, 8.0 * tiny),
            (0.5, -0.5),
        ],
        dtype=dtype,
    )
    a, b = ab.T
    self.assertTrue(bool(jnp.all(_use_quad(a, b))))

    def mp_signed_prob(a_, b_):
      if a_ > b_:
        return -mp_signed_prob(b_, a_)
      with mpmath.workprec(_PREC + 32):
        anchor, log_coeff = _mpmath_normal_prob(a_, b_)
        return +(mpmath.exp(-0.5 * anchor**2 + log_coeff))

    jax_quad_val = jax.jit(lambda a_, b_: _normal_prob_quad(a_, b_).value())
    util.check_precision_at_points(
        self, jax_quad_val, mp_signed_prob, a, b, max_ulp=2.5
    )

    has_pos_width = a < b
    jax_quad_log = jax.jit(lambda a_, b_: _normal_prob_quad(a_, b_).log_value())

    def mp_log_prob(a_, b_):
      with mpmath.workprec(_PREC + 32):
        anchor, log_coeff = _mpmath_normal_prob(a_, b_)
        return +(-0.5 * anchor**2 + log_coeff)

    util.check_precision_at_points(
        self,
        jax_quad_log,
        mp_log_prob,
        a[has_pos_width],
        b[has_pos_width],
        max_ulp=2.0,
    )


for _name in ("logpdf", "pdf", "logcdf", "cdf", "logsf", "sf", "ppf", "isf"):
  util.register_benchmark(
      getattr(truncnorm, _name), name=f"truncnorm_{_name}", nargs=3
  )


if __name__ == "__main__":
  util.main()
