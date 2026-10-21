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

"""Precision tests for hypergeometric functions against reference implementations."""

import itertools
import math

from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp

# Under pytest, tests run against an installed wheel that does not
# include `jax.tests`, so skip before importing `jax.tests.numerics`.
if jtu.is_running_under_pytest():
  import pytest

  pytest.skip("Only runs under Bazel", allow_module_level=True)

from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special

config.parse_flags_with_absl()

f32, f64 = jnp.float32, jnp.float64
# hyp1f1 is not tested in bfloat16 and float16, which are too narrow for its
# series and gamma function ratios.
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [f32, f64]]


def _hyp1f1_series(a, b, z, t, kummer, d=None):
  """Float64 sums of the series of t * 1F1(c, b, z) for z >= 0.

  c is d = b - a where kummer and a elsewhere, and t is the first term. Returns
  1F1, its derivatives in a and b, and the sums of the absolute values of the
  terms of 1F1, of its derivative in c, and of its derivative in b at fixed c.
  The derivatives of each term are carried along with the product rule, so that
  integer c <= 0 (where the series terminates) needs no special case.
  """
  kummer = kummer_all = np.broadcast_to(kummer, t.shape)
  c = np.where(kummer, b - a if d is None else d, a)
  zero = np.zeros_like(t)
  v, w, w_c = zero, zero, zero  # The derivatives of t in c, b and b at fixed c.
  sums = [t, zero, zero, np.abs(t), zero, zero]
  out = np.empty((6, len(t)))
  idx = np.arange(len(t))
  k = 0
  while len(idx):
    r = z / ((b + k) * (k + 1))
    # The b-derivative of (c + k) / (b + k) is g / (b + k)**2.
    g = np.where(kummer, a, -(c + k))
    v = (v * (c + k) + t) * r
    w = (w * (c + k) + t * g / (b + k)) * r
    w_c = (w_c - t / (b + k)) * (c + k) * r
    t = t * (c + k) * r
    for i, term in enumerate((t, v, w, np.abs(t), np.abs(v), np.abs(w_c))):
      sums[i] = sums[i] + term
    k += 1
    if k % 16 == 0:
      # The terms may grow until k ~ z, and decrease from there on.
      done = k > z + 40
      for term, total in ((t, sums[0]), (v, sums[1]), (w, sums[2])):
        done &= np.abs(term) <= 1e-18 * np.abs(total)
      done |= ~np.isfinite(sums[0] + sums[1] + sums[2])
      out[:, idx[done]] = [total[done] for total in sums]
      keep = ~done
      idx, kummer, a, b, c, z, t, v, w, w_c = (
          arr[keep] for arr in (idx, kummer, a, b, c, z, t, v, w, w_c))
      sums = [total[keep] for total in sums]
  out[1] = np.where(kummer_all, -out[1], out[1])
  return out


def _rgamma_and_derivative(d):
  """1 / gamma(d) and its derivative, which is (-1)^n n! at the poles d = -n."""
  rgamma = scipy.special.rgamma(d)
  pole = (d <= 0) & (d == np.round(d))
  sign = np.where(d % 2 == 0, 1, -1)
  return rgamma, np.where(pole, sign * scipy.special.gamma(1 - d),
                          -scipy.special.psi(d) * rgamma)


def _hyp1f1_asymptotic(a, b, z, d):
  """Float64 1F1(a, b, -z) and its derivatives in a and b, for z >= 1000.

  d is b - a. Uses the expansion for x -> -inf (https://dlmf.nist.gov/13.7.E2)
    1F1(a, b, -z) ~ gamma(b) / gamma(b - a) z^-a
                    sum_k (a)_k (a - b + 1)_k / k! z^-k
  whose exponentially small part is negligible for |a|, |b| <= 10.
  """
  c = 1 - d
  t, u, w = np.ones_like(z), np.zeros_like(z), np.zeros_like(z)
  s, s_a, s_b = t.copy(), u.copy(), w.copy()
  for k in range(60):
    r = 1 / ((k + 1) * z)
    u = (u * (a + k) * (c + k) + t * (a + c + 2 * k)) * r
    w = (w * (c + k) - t) * (a + k) * r
    t = t * (a + k) * (c + k) * r
    s, s_a, s_b = s + t, s_a + u, s_b + w
  rgamma, rgamma_d = _rgamma_and_derivative(d)
  pole = (d <= 0) & (d == np.round(d))
  # gamma(b) z^-a, which can be in range while z^-a is not.
  p = np.sign(scipy.special.gamma(b)) * np.exp(
      scipy.special.gammaln(b) - a * np.log(z))
  # At the poles, 1F1 is exp(-z) times a polynomial, which underflows; its sign
  # is that of the leading term (-n)_n / (b)_n z^n / n!.
  sign = np.where(d % 2 == 0, 1, -1) * scipy.special.poch(b, -d)
  f = np.where(pole, np.copysign(0, sign), p * rgamma * s)
  f_a = p * (-(rgamma_d + rgamma * np.log(z)) * s + rgamma * s_a)
  f_b = p * ((scipy.special.psi(b) * rgamma + rgamma_d) * s + rgamma * s_b)
  return f, f_a, f_b


def _hyp1f1_reference(a, b, x, shift=0):
  """Float64 1F1(a, b, x) and its derivatives in a and b, for |a|, |b| <= 10.

  With shift, these are at (a + shift, b + shift, x), without rounding b - a.

  Used as the reference for float32 (float64 tests use mpmath), since scipy
  has no derivatives of hyp1f1 and mpmath is too slow for the float32 sample
  count. For x >= 0 the series of 1F1 is summed directly, and for
  -1000 <= x < 0 that of Kummer's transform
    1F1(a, b, x) = exp(x) 1F1(b - a, b, -x),
  whose terms do not alternate in sign once k > a - b. Below that the
  asymptotic expansion is used. Over the tested domain this agrees with mpmath
  to within 0.02 float32 ULP.
  """
  a, b, x = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a, b, x)))
  a, b, x = a.ravel(), b.ravel(), x.ravel()
  d, a, b = b - a, a + shift, b + shift
  f, f_a, f_b = (np.full(a.shape, np.nan) for _ in range(3))
  ok = np.isfinite(a) & np.isfinite(b)
  with np.errstate(all="ignore"):
    m = ok & (x >= 0) & (x <= 1000)
    f[m], f_a[m], f_b[m] = _hyp1f1_series(a[m], b[m], x[m], np.ones(m.sum()),
                                          kummer=False)[:3]
    m = ok & (x < 0) & (x >= -1000)
    # exp(x) is applied in two halves, so that the series does not overflow.
    h = np.exp(x[m] / 2)
    f[m], f_a[m], f_b[m] = h * _hyp1f1_series(a[m], b[m], -x[m], h,
                                              kummer=True, d=d[m])[:3]
    m = ok & (x < -1000)
    f[m], f_a[m], f_b[m] = _hyp1f1_asymptotic(a[m], b[m], -x[m], d[m])
  # The convention of hyp1f1, which also holds for x = nan.
  f[a == 0] = 1
  return f, f_a, f_b


def _hyp1f1_cancellation(a, b, x):
  """The cancellation in hyp1f1 and in its derivatives in a and b.

  That is the ratio of the sum of the absolute values of the terms that hyp1f1
  adds up to the absolute value of the result, by which the relative rounding
  error grows.
  """
  ratios = np.ones((3, len(a)))
  with np.errstate(all="ignore"):
    # For |x| < 100, hyp1f1 sums the series of 1F1(a, b, x) for x >= 0, and
    # that of 1F1(b - a, b, -x) for x < 0. The b-derivative of the latter is
    # that at fixed b - a plus the derivative in b - a.
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(x) < 100)
    kummer = x[m] < 0
    s, s_a, s_b, abs_s, abs_a, abs_b = _hyp1f1_series(
        a[m], b[m], np.abs(x[m]), np.ones(m.sum()), kummer)
    ratios[:, m] = (abs_s / np.abs(s), abs_a / np.abs(s_a),
                    (abs_b + np.where(kummer, abs_a, 0)) / np.abs(s_b))
    # For x <= -100, the derivatives of the asymptotic expansion are dominated
    # by those of its prefactor gamma(b) / gamma(b - a) (-x)^-a, which hyp1f1
    # forms from 1 / gamma(b - a), its derivative, digamma(b) and log(-x), and
    # the b-derivative again in the same way.
    m = np.isfinite(a) & np.isfinite(b) & (x <= -100)
    d, log_z = b[m] - a[m], np.log(-x[m])
    rgamma, rgamma_d = _rgamma_and_derivative(d)
    rgamma_log_z = rgamma * log_z
    rgamma_psi_b = rgamma * scipy.special.psi(b[m])
    ratios[1, m] = ((np.abs(rgamma_d) + np.abs(rgamma_log_z))
                    / np.abs(rgamma_d + rgamma_log_z))
    ratios[2, m] = ((np.abs(rgamma_d) + 2 * np.abs(rgamma_log_z)
                     + np.abs(rgamma_psi_b)) / np.abs(rgamma_d + rgamma_psi_b))
    # For 1F1 itself at x <= -100 and b - a = -n, it sums the n + 1 terms of
    # the series of 1F1(-n, b, -x) instead.
    m &= _hyp1f1_terminates(a, b, x)
    c, b, z = b[m] - a[m], b[m], -x[m]
    term = total = total_abs = np.ones_like(z)
    for k in range(20):
      term = term * (c + k) / (b + k) * z / (k + 1)
      total, total_abs = total + term, total_abs + np.abs(term)
    ratios[0, m] = total_abs / np.abs(total)
  return ratios


def _hyp1f1_terminates(a, b, x):
  """Whether hyp1f1 sums the terminating series of 1F1(b - a, b, -x)."""
  return (x < 0) & (b - a <= 0) & (b - a == np.round(b - a))


def _hyp1f1_ref(a, b, x):
  return _hyp1f1_reference(a, b, x)[0]


def _hyp1f1_grad_a_ref(a, b, x):
  return _hyp1f1_reference(a, b, x)[1]


def _hyp1f1_grad_b_ref(a, b, x):
  return _hyp1f1_reference(a, b, x)[2]


def _hyp1f1_grad_x_ref(a, b, x):
  return a / b * _hyp1f1_reference(a, b, x, shift=1)[0]


def _mpmath_hyp1f1(a, b, x):
  if a == 0:
    return mpmath.mpf(1)
  if x < -1000:
    # mpmath.hyp1f1 can take seconds per point for tiny a or b and large -x,
    # so sum the asymptotic expansion used by _hyp1f1_asymptotic, whose terms
    # here decrease at least 4-fold each.
    z = -x
    s = t = mpmath.mpf(1)
    for k in range(60):
      t *= (a + k) * (a - b + 1 + k) / ((k + 1) * z)
      s += t
    return mpmath.gamma(b) * mpmath.rgamma(b - a) * z ** -a * s
  # Tiny a, b and x need a higher working precision than the default maximum.
  return mpmath.hyp1f1(a, b, x, maxprec=max(30000, 10 * mpmath.mp.prec))


def _mpmath_derivative(f, v, relative=False):
  """f'(v) by a central difference.

  The step is relative to v if relative (which keeps b clear of the pole at 0),
  and the working precision is raised until it covers the bits lost to
  cancellation (or the derivative is 0).
  """
  try:
    f_v = abs(f(v))
  except mpmath.libmp.NoConvergence:
    f_v = 0  # f(v) == 0, as for 1F1(-1, b, b).
  h = (abs(v) if relative and v else 1) * mpmath.ldexp(1, -mpmath.mp.prec - 20)
  addprec = 20
  while True:
    derivative = mpmath.diff(f, v, h=h, addprec=addprec)
    lost = (mpmath.log(f_v / abs(derivative * h), 2) if derivative
            else mpmath.inf)
    if lost < addprec or addprec > 4000:
      return derivative
    addprec = 4 * addprec if mpmath.isinf(lost) else int(lost) + 20


def _mpmath_hyp1f1_grad_a(a, b, x):
  if mpmath.isnan(a) or mpmath.isnan(b) or mpmath.isnan(x):
    return mpmath.nan
  return _mpmath_derivative(lambda t: _mpmath_hyp1f1(t, b, x), a)


def _mpmath_hyp1f1_grad_b(a, b, x):
  if mpmath.isnan(a) or mpmath.isnan(b) or mpmath.isnan(x):
    return mpmath.nan
  return _mpmath_derivative(lambda t: _mpmath_hyp1f1(a, t, x), b, relative=True)


def _mpmath_hyp1f1_grad_x(a, b, x):
  return a / b * _mpmath_hyp1f1(a + 1, b + 1, x)


def _two_sum(a, b):
  """Returns s = a + b rounded to the dtype, and a + b - s exactly."""
  s = a + b
  v = s - a
  return s, (a - (s - v)) + (b - v)


def _near_overflow_or_underflow(f, x, finfo):
  """Whether |reference| f is within a factor 1 / eps of overflow or underflow.

  The reference itself overflows to inf or nan for finite x.
  """
  return (~(f <= finfo.max * finfo.eps) & ~np.isnan(x)) | (
      (f > 0) & (f < finfo.tiny / finfo.eps))


def _hyp1f1_ignored(a, b, x, which=0):
  """Inputs (a, b, x) excluded from the accuracy checks.

  These are for 1F1 if which is 0, and for its derivative in a or b if which is
  1 or 2. Only |a|, |b| <= 10 are tested: beyond that the 250 terms that hyp1f1
  sums are not always enough, and the asymptotic expansions lose accuracy. All
  finite x are tested, except where noted below.
  """
  finfo = np.finfo(a.dtype)
  ignored = ~((np.abs(a) <= 10) & (np.abs(b) <= 10)) | np.isinf(x)
  d, d_err = _two_sum(b, -a)
  # Whether subnormal inputs (and b - a) are flushed to zero depends on the
  # platform.
  for v in (a, b, x, d):
    ignored |= (v != 0) & (np.abs(v) < finfo.tiny)
  # Poles of 1F1 at b = 0, -1, -2, ...
  ignored |= (b <= 0) & (b == np.round(b))
  a, b, x, d_err = (v.astype(np.float64) for v in (a, b, x, d_err))
  with np.errstate(all="ignore"):
    ref = _hyp1f1_reference(a, b, x)
    f = np.abs(ref[which])
    # Results within a factor 1 / eps of overflowing or underflowing, where
    # intermediate results do.
    ignored |= _near_overflow_or_underflow(f, x, finfo)
    # For x < 0, hyp1f1 works with b - a, whose rounding error the reference
    # does not have. It matters close to the poles of gamma(b - a) (as
    # 1F1(a, b, x) ~ gamma(b) / gamma(b - a) (-x)^-a for x -> -inf) and to
    # b - a = 0, and points where it changes 1F1 by more than 64 ULP are
    # ignored.
    d_err_ulp = np.abs(ref[1] * d_err) / (finfo.eps * np.abs(ref[0]))
    ignored |= (x < 0) & (d_err_ulp > 64)
    # For x >= 100, hyp1f1 uses the expansion for x -> +inf
    #   1F1(a, b, x) ~ gamma(b) / gamma(a) exp(x) x^(a - b)
    #                    sum_k (b - a)_k (1 - a)_k / k! x^-k
    #                  + gamma(b) / gamma(b - a) (-x)^-a
    #                    sum_k (a)_k (a - b + 1)_k / k! (-x)^-k
    # without its second part, which matters when a is close to 0 or to a
    # non-positive integer. For x <= -100 it uses Kummer's transformation of
    # it, without its first part. The series diverge, and are inaccurate when
    # none of their terms is below eps times their sum.
    # (Its derivatives use them also where 1F1 itself uses the terminating
    # series.)
    a_k, x_k = np.where(x > 0, a, b - a), np.abs(x)
    asymptotic = x_k >= 100
    if which == 0:
      asymptotic &= ~_hyp1f1_terminates(a, b, x)
    log_ratio = (scipy.special.gammaln(a_k) - scipy.special.gammaln(b - a_k)
                 + (b - 2 * a_k) * np.log(x_k) - x_k)
    ignored |= asymptotic & (log_ratio > np.log(finfo.eps))
    term = total = np.ones_like(x)
    converged = ~asymptotic
    for k in range(250):
      term = term * (b - a_k + k) * (1 - a_k + k) / ((k + 1) * x_k)
      total = total + term
      converged |= np.abs(term) <= finfo.eps * np.abs(total)
    ignored |= ~converged
    # It also evaluates exp(x) on its own, which overflows for x > log(max)
    # (and so does 1F1, except for a as above).
    ignored |= x > np.log(finfo.max)
    # For x <= -100, hyp1f1 evaluates (-x)^-a and gamma(b) / gamma(b - a) on
    # their own, which can overflow or underflow before the result does, the
    # latter for b close to 0.
    power = -a * np.log(-x)
    gamma_ratio = np.abs(scipy.special.gamma(b) * scipy.special.rgamma(b - a))
    underflow = (power < np.log(finfo.tiny)) & (f >= finfo.tiny)
    ignored |= asymptotic & (x < 0) & ((power > np.log(finfo.max))
                                       | (gamma_ratio > finfo.max) | underflow)
    # Where the terms that hyp1f1 adds up cancel, the relative error grows with
    # the ratio of the sum of their absolute values to the result, and is too
    # large when it exceeds 100. For |x| < 100 those are the terms of the
    # series, scaled by exp(x / 2) for x < 0, which can also overflow before the
    # result does. The first of them, a / b, overflows for b close to 0.
    ratio = _hyp1f1_cancellation(a, b, x)[which]
    terms = ratio * f * np.exp(np.maximum(-x, 0) / 2)
    ignored |= (ratio > 100) | ((np.abs(x) < 100) & (terms > finfo.max))
    ignored |= np.abs(a / b) > finfo.max
  return ignored


def _hyp1f1_grad_a_ignored(a, b, x):
  return _hyp1f1_ignored(a, b, x, which=1)


def _hyp1f1_grad_b_ignored(a, b, x):
  return _hyp1f1_ignored(a, b, x, which=2)


def _hyp1f1_grad_x_ignored(a, b, x):
  """The x-derivative is a / b * hyp1f1(a + 1, b + 1, x)."""
  finfo = np.finfo(a.dtype)
  a_1, a_err = _two_sum(a, a.dtype.type(1))
  b_1, b_err = _two_sum(b, b.dtype.type(1))
  ignored = _hyp1f1_ignored(a_1, b_1, x)
  ignored |= ~((np.abs(a) <= 10) & (np.abs(b) <= 10)) | (b == 0)
  for v in (a, b):
    ignored |= (v != 0) & (np.abs(v) < finfo.tiny)
  a, b, x, a_1, b_1, a_err, b_err = (
      v.astype(np.float64) for v in (a, b, x, a_1, b_1, a_err, b_err))
  with np.errstate(all="ignore"):
    f = np.abs(_hyp1f1_grad_x_ref(a, b, x))
    ignored |= _near_overflow_or_underflow(f, x, finfo)
    # The rounding errors of a + 1 and b + 1 can matter, as that of b - a above.
    f, f_a, f_b = _hyp1f1_reference(a_1, b_1, x)
    err = np.abs(f_a * a_err) + np.abs(f_b * b_err)
    ignored |= err > 64 * finfo.eps * np.abs(f)
  return ignored


_HYP1F1_INTERESTING_POINTS = list(itertools.product(
    [-7.5, 1e-5, 2.5, 9.5],
    [-6.5, 0.5, 9.2],
    # hyp1f1 switches from the series to the asymptotic expansions at |x| = 100,
    # and the references at x = -1000.
    [-1e6, -1000.5, -999.5, -100.5, -99.5, -20.0, 20.0, 99.5, 100.5, 300.0],
))


@jax.jit
def hyp1f1_grad_a(a, b, x):
  return jax.vmap(jax.grad(jsp.special.hyp1f1, argnums=0))(a, b, x)


@jax.jit
def hyp1f1_grad_b(a, b, x):
  return jax.vmap(jax.grad(jsp.special.hyp1f1, argnums=1))(a, b, x)


@jax.jit
def hyp1f1_grad_x(a, b, x):
  return jax.vmap(jax.grad(jsp.special.hyp1f1, argnums=2))(a, b, x)


@jtu.thread_unsafe_test_class()
class Hyp1f1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hyp1f1_accuracy(self, dtype):
    # The errors are largest where the terms of the series partly cancel, and
    # close to |x| = 100, where the series have the most terms.
    bounds = [
        ("cpu", {f32: 150.0, f64: 75.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        jsp.special.hyp1f1,
        _hyp1f1_ref,
        _mpmath_hyp1f1,
        dtype,
        nargs=3,
        bounds=bounds,
        ignore_inputs=_hyp1f1_ignored,
        # Results that underflow are +0.0 regardless of their sign, as the
        # exp(x) factor of Kummer's transformation underflows first.
        check_signed_zeros=False,
        interesting_points=_HYP1F1_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class Hyp1f1GradTest(jtu.JaxTestCase):

  def _check_grad(self, jax_fn, ref_fn, mpmath_fn, ignored, dtype, bounds):
    util.check_nary_precision(
        self,
        jax_fn,
        ref_fn,
        mpmath_fn,
        dtype,
        nargs=3,
        bounds=bounds,
        ignore_inputs=ignored,
        # Results that underflow are +0.0 regardless of their sign, as the
        # exp(x) factor of Kummer's transformation underflows first.
        check_signed_zeros=False,
        interesting_points=_HYP1F1_INTERESTING_POINTS,
        # The gradients are slower to evaluate than 1F1 itself.
        max_samples=2**18 if dtype == f32 else None,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hyp1f1_grad_a_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 150.0, f64: 100.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    self._check_grad(hyp1f1_grad_a, _hyp1f1_grad_a_ref,
                     _mpmath_hyp1f1_grad_a, _hyp1f1_grad_a_ignored, dtype,
                     bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hyp1f1_grad_b_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 550.0, f64: 600.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    self._check_grad(hyp1f1_grad_b, _hyp1f1_grad_b_ref,
                     _mpmath_hyp1f1_grad_b, _hyp1f1_grad_b_ignored, dtype,
                     bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hyp1f1_grad_x_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 150.0, f64: 75.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    self._check_grad(hyp1f1_grad_x, _hyp1f1_grad_x_ref,
                     _mpmath_hyp1f1_grad_x, _hyp1f1_grad_x_ignored, dtype,
                     bounds)


util.register_benchmark(jsp.special.hyp1f1, nargs=3, dtypes=[f32, f64])


if __name__ == "__main__":
  util.main()
