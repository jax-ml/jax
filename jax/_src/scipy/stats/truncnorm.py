# Copyright 2022 The JAX Authors.
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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
import math
from typing import Any

from jax._src import api
from jax._src import custom_derivatives
from jax._src import dtypes
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.lax import double_word
# `jax._src.lax` exports `erf` as the function `lax.erf`, so import symbols
# from `jax._src.lax.erf` directly rather than `from jax._src.lax import erf`.
from jax._src.lax.erf import (
    _ERF32_T_M1,
    _ERF64_T_M1,
    _HALF_LN2PI_F32,
    _HALF_LN2PI_F64,
    _SQRT_HALF_PI_F32,
    _SQRT_HALF_PI_F64,
    _ndtr,
    exp_neg_half_sq,
)
from jax._src.lax.lax import AccuracyMode
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy import special
from jax._src.tree_util import register_dataclass, tree_map
from jax._src.typing import Array, ArrayLike
import numpy as np

# Every function here needs P(a, b) = Phi(b) - Phi(a), where Phi is the
# standard normal CDF. Computing ndtr(b) - ndtr(a) directly goes wrong in two
# places:
#   - Narrow intervals: Phi(a) and Phi(b) agree in most of their digits, so
#     the subtraction leaves only the few digits where they differ.
#   - Tails: for large positive a and b, Phi(a) and Phi(b) both round to 1.0;
#     for large negative a and b, both underflow to 0.0. Either way
#     ndtr(b) - ndtr(a) collapses to 0 even though ratios like phi(x) / P(a, b)
#     remain finite and well-conditioned.
#
# _normal_prob picks one of three formulas, each of which avoids both problems
# in its own region:
#   - _normal_prob_quad:  numerical integration of the density over [a, b],
#     which involves no subtraction. Used for short intervals.
#   - _normal_prob_erfcx: for intervals entirely in one tail. Pulls the tiny
#     factor exp(-anchor^2 / 2) out front and subtracts only O(1) quantities.
#   - _normal_prob_ndtr:  signed ndtr differences, for wide intervals near zero.
#
# Each returns a _FactoredExp, which keeps that exp(-anchor^2 / 2) factor
# unexponentiated (as `anchor`) so ratios can cancel it before underflow.
#
# ppf and isf solve cdf(z) = q for z, again choosing among three methods:
#   - _ppf_near_bound: z is very close to a finite bound a (or b), so solve
#     for the small distance z - a directly.
#   - _ppf_near_zero:  z is near 0, where Phi(z) is near 0.5 and its log
#     carries little information about z.
#   - _ppf_tail:       everything else, via ndtri_exp on log Phi(z).

_LOG_2 = math.log(2.0)
_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2 = math.sqrt(0.5)


def _dot_tangents(
    out: Array, tangents: tuple[Any, ...], *partial_fns: Callable[[], Array]
) -> Array:
  """Sums df/dx_i * dx_i over active inputs, skipping SymbolicZero tangents."""
  dout = jnp.zeros((), dtype=out.dtype)
  for t, fn in zip(tangents, partial_fns):
    if type(t) is not custom_derivatives.SymbolicZero:
      dout = dout + fn() * t
  return jnp.broadcast_to(dout, out.shape)


def _select(cases: Sequence[tuple[Array, Any]], default: Any) -> Any:
  """Like jnp.select, but takes (cond, value) pairs and supports PyTrees."""
  out = default
  for cond, val in reversed(cases):
    out = tree_map(partial(jnp.where, cond), val, out)
  return out


def _exp(x: Array) -> Array:
  """High-accuracy exp(x) returning 0.0 with zero derivative at x = -inf.

  Masks x = -inf before calling lax.exp so an infinite tangent on an input
  cannot combine with exp(-inf) = 0 to produce 0 * inf = NaN in autodiff.
  """
  is_ok = ~jnp.isneginf(x)
  y = lax.exp(jnp.where(is_ok, x, 0.0), accuracy=AccuracyMode.HIGHEST)
  return jnp.where(is_ok, y, 0.0)


_expm1 = partial(lax.expm1, accuracy=AccuracyMode.HIGHEST)


def _log_or_log1p(fn: Any, x: Array, cutoff: float, safe_x: float) -> Array:
  """High-accuracy log/log1p returning -inf with zero derivative at x <= cutoff.

  Masks x <= cutoff before calling fn so the 1/0 derivative at the boundary
  never enters autodiff.
  """
  is_ok = ~(x <= cutoff)
  y = fn(jnp.where(is_ok, x, safe_x), accuracy=AccuracyMode.HIGHEST)
  return jnp.where(is_ok, y, -np.inf)


_log = partial(_log_or_log1p, lax.log, cutoff=0.0, safe_x=1.0)
_log1p = partial(_log_or_log1p, lax.log1p, cutoff=-1.0, safe_x=0.0)


@custom_derivatives.custom_jvp
def _half_sq_diff(x: Array, y: Array) -> Array:
  """Computes (x^2 - y^2) / 2 without cancellation or x + y overflow.

  Halving before adding or subtracting avoids overflowing x + y when
  |x + y| > fmax (for example, x == y > fmax / 2 would otherwise give
  0 * inf = NaN).
  """
  return 2.0 * ((0.5 * x - 0.5 * y) * (0.5 * x + 0.5 * y))


@partial(_half_sq_diff.defjvp, symbolic_zeros=True)
def _half_sq_diff_jvp(
    primals: tuple[Array, Array], tangents: tuple[Any, Any]
) -> tuple[Array, Array]:
  """Differentiates (x^2 - y^2) / 2 directly as x * dx - y * dy."""
  x, y = primals
  val = _half_sq_diff(x, y)
  # Zero derivatives when the primal overflows to +-inf so inf does not leak.
  is_finite = jnp.isfinite(val)
  return val, _dot_tangents(
      val,
      tangents,
      lambda: jnp.where(is_finite, x, 0.0),
      lambda: -jnp.where(is_finite, y, 0.0),
  )


@register_dataclass
@dataclass(frozen=True)
class _DoubleWord:
  """A value stored as hi + lo, giving roughly twice the mantissa bits.

  Used for exponents: exp(x) turns an absolute error e in x into a relative
  error e in the result, so an exponent of 100 rounded to float32 (error ~4e-6)
  would already cost ~30 ULPs. Carrying the rounding error in `lo` preserves
  those bits.

  `lo` is excluded from derivatives (dropping its gradient changes results by
  at most O(eps)). Non-finite `hi` values get `lo = 0` so inf - inf cannot turn
  into NaN.

  TODO(barron/phawkins): Upstream +-inf guards and AD rules to
  jax._src.lax.double_word.two_sum/f2_add/two_prod_big and add overflow-safe
  f2_mul_big, then replace _DoubleWord with double_word calls. Also generally
  see if double_word.py can be improved by upstreaming anything else here.
  """

  hi: Array
  lo: Array

  @classmethod
  def from_array(cls, x: Array) -> "_DoubleWord":
    return cls(x, jnp.zeros((), dtype=x.dtype))

  @staticmethod
  def _two_prod_big(x: Array, y: Array) -> tuple[Array, Array]:
    # Inline double_word.two_prod_big without its @jit wrapper to avoid nested
    # jit boundaries inside truncnorm's jitted entry points.
    prod = x * y
    x_hi, x_lo = double_word.split_half_mask(x)
    y_hi, y_lo = double_word.split_half_mask(y)
    err = (((x_hi * y_hi - prod) + x_hi * y_lo) + x_lo * y_hi) + x_lo * y_lo
    return prod, err

  @staticmethod
  def _residual(hi: Array, fn: Any, *args: Array) -> Array:
    """Computes the low-order residual without propagating gradients or NaNs."""
    hi = lax.stop_gradient(hi)
    is_finite = jnp.isfinite(hi)
    safe_hi = jnp.where(is_finite, hi, 0.0)
    safe_args = [jnp.where(is_finite, lax.stop_gradient(a), 0.0) for a in args]
    return jnp.where(is_finite, fn(safe_hi, *safe_args), 0.0)

  def __neg__(self) -> "_DoubleWord":
    return _DoubleWord(-self.hi, lax.stop_gradient(-self.lo))

  def __add__(self, other: "_DoubleWord | Array") -> "_DoubleWord":
    if not isinstance(other, _DoubleWord):
      other = _DoubleWord.from_array(other)
    hi = self.hi + other.hi

    def _lo(hi, x, y, lo):
      sum_hi, sum_err = double_word.two_sum(x, y)
      # sum_hi - hi is 0 unless the compiler fused self.hi + other.hi.
      return (sum_hi - hi) + sum_err + lo

    return _DoubleWord(
        hi, self._residual(hi, _lo, self.hi, other.hi, self.lo + other.lo)
    )

  def __sub__(self, other: "_DoubleWord | Array") -> "_DoubleWord":
    return self + (-other)

  @classmethod
  def mul(cls, x: Array, y: Array) -> "_DoubleWord":
    """Evaluates x * y as a _DoubleWord with non-differentiable residual."""
    hi = x * y

    def _lo(hi, x, y):
      prod_hi, prod_err = cls._two_prod_big(x, y)
      # prod_hi - hi is 0 unless the compiler fused x * y.
      return (prod_hi - hi) + prod_err

    return cls(hi, cls._residual(hi, _lo, x, y))

  @classmethod
  def half_sq_diff(cls, x: Array, y: Array) -> "_DoubleWord":
    """Evaluates (x^2 - y^2) / 2 as a _DoubleWord without cancellation."""
    hi = _half_sq_diff(x, y)

    def _lo(hi, x, y):
      # Split each factor into hi + lo (halving is exact in IEEE float), then
      # expand 2 * (sub_hi + sub_lo) * (add_hi + add_lo) dropping sub_lo * add_lo.
      sub_hi, sub_lo = double_word.two_sum(0.5 * x, -0.5 * y)
      add_hi, add_lo = double_word.two_sum(0.5 * x, 0.5 * y)
      prod_hi, prod_err = cls._two_prod_big(sub_hi, add_hi)
      return (2.0 * prod_hi - hi) + 2.0 * (
          prod_err + (sub_hi * add_lo + sub_lo * add_hi)
      )

    return cls(hi, cls._residual(hi, _lo, x, y))


@register_dataclass
@dataclass(frozen=True)
class _FactoredExp:
  """Represents coeff * exp(exponent.hi + exponent.lo) * exp(-anchor^2 / 2).

  Densities and tail probabilities have the form exp(-x^2 / 2) times a
  moderate factor, and exp(-x^2 / 2) underflows once |x| exceeds ~38 in float64
  (~13 in float32). Ratios such as phi(z) / P(a, b) ~ exp((a^2 - z^2) / 2)
  remain finite and well-scaled even when both numerator and denominator
  underflow. Storing the Gaussian argument in `anchor` lets `__truediv__`
  compute (other.anchor^2 - self.anchor^2) / 2 in _DoubleWord before
  exponentiating.

  Probabilities and densities store an array in `anchor` (0.0 when no Gaussian
  factor is factored out); dividing two anchored values cancels their anchors
  into `exponent` and returns an unanchored value with `anchor=None`.
  """

  coeff: Array
  exponent: _DoubleWord
  anchor: Array | None = None

  @classmethod
  def from_coeff(
      cls, coeff: Array, *, anchor: Array | None = None
  ) -> "_FactoredExp":
    zero = jnp.zeros((), dtype=coeff.dtype)
    return cls(coeff, _DoubleWord.from_array(zero), anchor)

  @classmethod
  def from_log(
      cls, exponent: _DoubleWord | Array, *, anchor: Array | None = None
  ) -> "_FactoredExp":
    exp_dw = (
        exponent
        if isinstance(exponent, _DoubleWord)
        else _DoubleWord.from_array(exponent)
    )
    return cls(jnp.ones((), dtype=exp_dw.hi.dtype), exp_dw, anchor)

  @classmethod
  def one(cls, dtype: Any, *, anchor: Array | None = None) -> "_FactoredExp":
    return cls.from_coeff(jnp.ones((), dtype=dtype), anchor=anchor)

  def value(self) -> Array:
    """Evaluates the represented value in linear space."""
    # Since |lo| is O(eps), exp(lo) ~= 1 + lo. Multiplying by (1 + lo) instead
    # of adding val + val * lo avoids flushing val * lo to 0 when val is tiny
    # on hardware without subnormals (such as TPU). Clamping >= 0.5 ensures
    # extreme underflow/overflow residuals never flip +0 to -0 or +inf to NaN.
    lo_correction = jnp.maximum(0.5, 1.0 + self.exponent.lo)
    val = self.coeff * _exp(self.exponent.hi) * lo_correction
    if self.anchor is None:
      return val
    return val * exp_neg_half_sq(self.anchor)

  def log_value(self) -> Array:
    """Evaluates log(value()) without underflow or overflow."""
    base = (_log(self.coeff) + self.exponent.hi) + self.exponent.lo
    if self.anchor is None:
      return base
    return -0.5 * self.anchor * self.anchor + base

  def scale_by(self, factor: Array) -> "_FactoredExp":
    """Multiplies by factor >= 0, pulling tiny powers of 2 into the exponent.

    When factor is subnormal or near underflow, multiplying coeff * factor loses
    bits or flushes to zero. Rescaling factor * 2^k into normal range and
    subtracting k * ln(2) from `exponent` is exact since 2^k * exp(-k*ln 2) = 1.
    """
    dtype = factor.dtype
    to_dtype = lambda v: jnp.asarray(v, dtype=dtype)
    exp_shift = jnp.zeros_like(factor)
    scaled = factor
    # Float64 exponents reach 2^-1074, handled in three binary-halving steps;
    # float32 lifts any factor < 2^-64 away from subnormals in one 2^88 step.
    if dtype == np.float32:
      ln2_hi, ln2_mid, ln2_lo = double_word.LN2_SPLIT_F32
      shift_spec: Sequence[tuple[int, int]] = ((64, 88),)
    else:
      ln2_hi, ln2_mid, ln2_lo = double_word.LN2_SPLIT_F64
      shift_spec = ((512, 512), (256, 256), (128, 128))
    for thresh_bits, shift_bits in shift_spec:
      is_tiny = (scaled > 0.0) & (scaled < to_dtype(2.0**-thresh_bits))
      scaled = jnp.where(is_tiny, scaled * to_dtype(2.0**shift_bits), scaled)
      # stop_gradient is exact away from threshold boundaries because
      # 2^k * exp(-k * ln 2) == 1 identically.
      exp_shift = lax.stop_gradient(
          jnp.where(is_tiny, exp_shift - to_dtype(shift_bits), exp_shift)
      )
    # LN2_SPLIT_* zeroes trailing bits of hi/mid, making integer * ln(2) exact.
    shift_hi = exp_shift * to_dtype(ln2_hi)
    shift_lo = exp_shift * to_dtype(ln2_mid) + exp_shift * to_dtype(ln2_lo)
    return _FactoredExp(
        scaled * self.coeff,
        self.exponent + _DoubleWord(shift_hi, shift_lo),
        self.anchor,
    )

  def __le__(self, other: "_FactoredExp") -> Array:
    """Compares two non-negative anchored values in log space."""
    if self.anchor is None or other.anchor is None:
      raise ValueError("Comparison requires anchored _FactoredExp values.")
    is_self_zero = (self.coeff <= 0.0) | jnp.isneginf(self.exponent.hi)
    is_other_zero = (other.coeff <= 0.0) | jnp.isneginf(other.exponent.hi)
    dtype = jnp.result_type(self.coeff, other.coeff)
    one = _FactoredExp.one(dtype, anchor=jnp.zeros((), dtype=dtype))
    safe_self = _select([(is_self_zero, one)], self)
    safe_other = _select([(is_other_zero, one)], other)
    log_diff = (
        _half_sq_diff(safe_other.anchor, safe_self.anchor)
        + (safe_self.exponent.hi - safe_other.exponent.hi)
        + (_log(safe_self.coeff) - _log(safe_other.coeff))
    )
    return is_self_zero | (~is_other_zero & (log_diff <= 0.0))

  def __add__(self, other: "_FactoredExp") -> "_FactoredExp":
    """Adds two non-negative anchored values without underflow.

    Divides the smaller term by the larger to get a ratio <= 1, then scales
    the larger by (1 + ratio).
    """
    is_self_smaller = self <= other
    small, large = _select([(is_self_smaller, (self, other))], (other, self))
    has_small = (small.coeff > 0.0) & jnp.isfinite(small.exponent.hi)
    zero_anchor = jnp.zeros((), dtype=small.coeff.dtype)
    one = _FactoredExp.one(small.coeff.dtype, anchor=zero_anchor)
    safe_small, safe_large = _select([(has_small, (small, large))], (one,) * 2)
    ratio_val = jnp.where(has_small, (safe_small / safe_large).value(), 0.0)
    return large * (1.0 + ratio_val)

  def __mul__(self, other: "_FactoredExp | Array | float") -> "_FactoredExp":
    if not isinstance(other, _FactoredExp):
      return _FactoredExp(self.coeff * other, self.exponent, self.anchor)
    if self.anchor is not None and other.anchor is not None:
      raise ValueError("Cannot multiply two anchored _FactoredExp values.")
    coeff = self.coeff * other.coeff
    exponent = self.exponent + other.exponent
    anchor = self.anchor if self.anchor is not None else other.anchor
    return _FactoredExp(coeff, exponent, anchor)

  def __truediv__(self, other: "_FactoredExp") -> "_FactoredExp":
    """Divides two anchored values, cancelling the Gaussian anchors."""
    if self.anchor is None or other.anchor is None:
      raise ValueError("Division requires anchored _FactoredExp values.")
    exponent = self.exponent - other.exponent
    exponent = _DoubleWord.half_sq_diff(other.anchor, self.anchor) + exponent
    return _FactoredExp(self.coeff / other.coeff, exponent)


def _neg_half_log_2pi(x: Array) -> _DoubleWord:
  """Returns -0.5 * log(2 * pi) as a _DoubleWord for x.dtype."""
  hi, lo = _HALF_LN2PI_F32 if x.dtype == np.float32 else _HALF_LN2PI_F64
  hi = jnp.asarray(-hi, dtype=x.dtype)
  lo = jnp.asarray(-lo, dtype=x.dtype)
  return _DoubleWord(hi, lo)


def _normal_pdf(x: Array) -> _FactoredExp:
  """Evaluates standard normal density phi(x) in factored form."""
  return _FactoredExp.from_log(_neg_half_log_2pi(x), anchor=x)


def _bound_pdf(
    bound: Array,
    denom: _FactoredExp,
    weight: _FactoredExp | Array | None = None,
) -> Array:
  """Evaluates weight * phi(bound) / denom, or 0 where bound is infinite."""
  assert denom.anchor is not None
  is_finite = jnp.isfinite(bound)
  safe_bound = jnp.where(is_finite, bound, denom.anchor)
  ratio = _normal_pdf(safe_bound) / denom
  if isinstance(weight, _FactoredExp):
    ratio = ratio * weight
  elif weight is not None:
    ratio = ratio.scale_by(weight)
  return jnp.where(is_finite, ratio.value(), 0.0)


def _ndtr_neg_abs(x: Array) -> Array:
  """Evaluates Phi(-|x|) <= 0.5 with continuous derivative sign across 0."""
  return _ndtr(jnp.where(x > 0.0, -x, x))


# Use Gauss-Legendre quadrature when [a, b] is short and exp(-x^2 / 2) does
# not vary too steeply across it. A short width alone is not enough in the far
# tail: on [100, 100.1], width = 0.1, but x^2 / 2 increases by ~10 so the
# Gaussian drops by e^-10 and polynomial quadrature loses accuracy (while erfcx
# subtraction has no cancellation there).
_QUAD_MAX_WIDTH = 1.48
_QUAD_MAX_EXP_DIFF = 4.0


def _use_quad(a: Array, b: Array) -> Array:
  """True when Gauss-Legendre quadrature is accurate on [a, b]."""
  is_finite = jnp.isfinite(a) & jnp.isfinite(b)
  # Non-finite bounds map to width 2.0 > _QUAD_MAX_WIDTH, so they return False.
  finite_a, finite_b = _select([(is_finite, (a, b))], (0.0, 2.0))
  width = jnp.abs(finite_b - finite_a)
  # How much x^2 / 2 changes across [a, b], i.e. how many factors of e the
  # density falls by. Exactly |b^2 - a^2| / 2 for same-sign bounds; an
  # overestimate across 0, where the width limit already keeps it below 1.1.
  exp_diff = (0.5 * jnp.abs(finite_a) + 0.5 * jnp.abs(finite_b)) * width
  return (width <= _QUAD_MAX_WIDTH) & (exp_diff <= _QUAD_MAX_EXP_DIFF)


def _normal_prob_quad(a: Array, b: Array) -> _FactoredExp:
  """Computes signed Phi(b) - Phi(a) via Gauss-Legendre quadrature.

  Integrates the normal density over [a, b] directly, avoiding CDF subtraction.
  With h = (b - a) / 2, the result is written as
    (b - a) * phi(a) * exp(-a * h - h^2 / 2) * (1 + corr),
  where the exponent is kept in _DoubleWord and corr (the deviation of the
  integrand from its midpoint value) is computed directly, so neither loses
  digits to rounding.

  In detail, with midpoint m = a + h and half-width h = (b - a) / 2,
  substituting x = m + h * u for u in [-1, 1] gives:
    Phi(b) - Phi(a)
      = (b - a) / sqrt(2 * pi) * exp(-m^2 / 2)
        * (1/2) * integral_{-1}^1 exp(-h^2 * u^2 / 2 - m * h * u) du.

  Writing -m^2 / 2 = -a^2 / 2 - a * h - h^2 / 2 anchors the Gaussian at `a`
  rather than `m` (avoiding rounding error in m^2 when b - a is tiny).
  Averaging +u and -u replaces exp(-m * h * u) with cosh(-m * h * u), and
  pulling out 1 turns the integral into 1 + corr where corr integrates only the
  small shape correction:
    corr = integral_0^1 (exp(-h^2 * u^2 / 2) * cosh(-m * h * u) - 1) du.

  When called with a > b (by _ppf_near_zero), width = b - a < 0 and the
  returned coeff carries that sign.
  """
  a, b = _select([(_use_quad(a, b), (a, b))], (0.0, 1.0))
  width = b - a
  h = 0.5 * width
  neg_mh = -(a + h) * h
  neg_half_h2 = -0.5 * h**2
  # Because the integrand exp(-h^2 u^2 / 2) * cosh(-m h u) - 1 is even and
  # vanishes at u = 0, the midpoint drops out and (1/2) * int_{-1}^1 reduces
  # to int_0^1 over strictly positive nodes with their [-1, 1] weights.
  if a.dtype == np.float32:
    # 7-point Gauss-Legendre rule on [-1, 1], strictly positive nodes/weights.
    nodes = (0.4058451513773972, 0.7415311855993945, 0.9491079123427585)
    weights = (0.3818300505051189, 0.27970539148927664, 0.1294849661688697)
  else:
    # 11-point Gauss-Legendre rule on [-1, 1], strictly positive nodes/weights.
    nodes = (
        0.26954315595234496,
        0.5190961292068118,
        0.7301520055740494,
        0.8870625997680953,
        0.978228658146057,
    )
    weights = (
        0.26280454451024665,
        0.23319376459199048,
        0.18629021092773426,
        0.1255803694649046,
        0.05566856711617366,
    )
  shape = (len(nodes),) + (1,) * h.ndim
  nodes_arr = jnp.asarray(nodes, dtype=a.dtype).reshape(shape)
  weights_arr = jnp.asarray(weights, dtype=a.dtype).reshape(shape)
  expm1_h2 = _expm1(nodes_arr**2 * neg_half_h2[None])
  expm1_mh = _expm1(nodes_arr * neg_mh[None])
  # Use exp(u) * cosh(v) - 1 = expm1(u) + exp(u) * expm1(v)^2 / (2 * exp(v))
  # so tiny exponents do not cancel against 1.
  integrand = expm1_h2 + 0.5 * (1.0 + expm1_h2) * expm1_mh**2 / (1.0 + expm1_mh)
  corr = jnp.sum(weights_arr * integrand, axis=0)
  # Sum the small O(b - a) terms before adding O(1) const so low bits of
  # -a * h - h^2 / 2 + log1p(corr) are preserved when |b - a| << 1.
  quad_shift = _DoubleWord.mul(-a, h) + neg_half_h2
  small_exp_terms = quad_shift + _log1p(corr)
  exponent = _neg_half_log_2pi(a) + small_exp_terms
  return _FactoredExp(width, exponent, anchor=a)


def _normal_prob_erfcx(a: Array, b: Array) -> _FactoredExp:
  """Computes Phi(b) - Phi(a) via erfcx when [a, b] lies in one tail.

  erfcx(x) = exp(x^2) * erfc(x) is about 1 / (x * sqrt(pi)) for large x, so
  unlike erfc it never underflows. Left-tail intervals are mirrored onto
  [lo, hi] with 0 < lo < hi, and then
  1 - Phi(z) = 0.5 * exp(-z^2 / 2) * erfcx(z / sqrt(2)) lets us pull the
  Gaussian factor at the bound nearer zero (the anchor) out front:
    Phi(b) - Phi(a) = exp(-anchor^2 / 2) * 0.5 * (
        erfcx(lo / sqrt(2)) - exp(-(hi^2 - lo^2) / 2) * erfcx(hi / sqrt(2)))
  """
  in_left_tail = b <= 0.0
  lo, hi, anchor = _select([(in_left_tail, (-b, -a, b))], (a, b, a))
  in_tail = (lo > _SQRT_2) & jnp.isfinite(lo)
  safe_lo, safe_anchor = _select([(in_tail, (lo, anchor))], (2.0, 2.0))
  erfcx_lo = jnp.where(in_tail, special.erfcx(safe_lo * _INV_SQRT_2), 1.0)
  is_hi_finite = in_tail & jnp.isfinite(hi) & (hi >= lo)
  safe_hi = jnp.where(is_hi_finite, hi, safe_lo)
  scale_hi = jnp.where(
      is_hi_finite, _exp(-_half_sq_diff(safe_hi, safe_lo)), 0.0
  )
  erfcx_hi = jnp.where(
      is_hi_finite, special.erfcx(safe_hi * _INV_SQRT_2), 0.0
  )
  coeff = 0.5 * (erfcx_lo - scale_hi * erfcx_hi)
  return _FactoredExp.from_coeff(coeff, anchor=safe_anchor)


def _normal_prob_ndtr(a: Array, b: Array) -> _FactoredExp:
  """Computes Phi(b) - Phi(a) via the standard normal CDF (wide intervals).

  Evaluating ndtr(z) for z > 0 loses precision as Phi(z) rounds toward 1.0, so
  we only call ndtr at arguments <= 0 (where the result is <= 0.5), using
  Phi(z) = 1 - Phi(-z):
    - a <= 0 < b  (spans 0):   Phi(b) - Phi(a) = 1 - (Phi(a) + Phi(-b))
    - 0 < a < b   (right of 0): Phi(b) - Phi(a) = Phi(-a) - Phi(-b)
    - a < b <= 0  (left of 0):  Phi(b) - Phi(a)
  """
  cdf_neg_a = _ndtr_neg_abs(a)
  cdf_neg_b = _ndtr_neg_abs(b)
  spans_zero = (a <= 0.0) & (b > 0.0)
  tail_diff = jnp.where(a > 0.0, cdf_neg_a - cdf_neg_b, cdf_neg_b - cdf_neg_a)
  log_prob = jnp.where(
      spans_zero, _log1p(-(cdf_neg_a + cdf_neg_b)), _log(tail_diff)
  )
  anchor = jnp.zeros((), dtype=log_prob.dtype)
  return _FactoredExp.from_log(log_prob, anchor=anchor)


def _normal_prob(a: Array, b: Array) -> _FactoredExp:
  """Evaluates Phi(b) - Phi(a) in factored form for a < b."""
  use_quad = _use_quad(a, b)
  use_erfcx = (a > _SQRT_2) | (b < -_SQRT_2)
  return _select(
      [
          (use_quad, _normal_prob_quad(a, b)),
          (use_erfcx, _normal_prob_erfcx(a, b)),
      ],
      _normal_prob_ndtr(a, b),
  )


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(3,))
def _std_pdf_or_logpdf(z: Array, a: Array, b: Array, log: bool) -> Array:
  """Evaluates standard truncnorm.(log)pdf for interior z in [a, b]."""
  prob_ab = _normal_prob(a, b)
  if log:
    # Divide P(a, b) / phi(z) rather than phi(z) / P(a, b) so subnormal widths
    # b - a do not overflow 1 / (b - a).
    return -(prob_ab / _normal_pdf(z)).log_value()
  return (_normal_pdf(z) / prob_ab).value()


@partial(_std_pdf_or_logpdf.defjvp, symbolic_zeros=True)
def _std_pdf_or_logpdf_jvp(
    log: bool,
    primals: tuple[Array, Array, Array],
    tangents: tuple[Any, Any, Any],
) -> tuple[Array, Array]:
  z, a, b = primals
  prob_ab = _normal_prob(a, b)
  if log:
    val = -(prob_ab / _normal_pdf(z)).log_value()
  else:
    val = (_normal_pdf(z) / prob_ab).value()
  # d/dz logpdf = -z, d/da logpdf = phi(a) / P(a, b), d/db = -phi(b) / P(a, b).
  # For pdf, multiply by val while zeroing 0 * inf if pdf underflows/overflows.
  is_nonzero = jnp.isfinite(val) & (val > 0.0)
  dval = _dot_tangents(
      val,
      tangents,
      lambda: -z if log else jnp.where(is_nonzero, -z, 0.0),
      lambda: _bound_pdf(a, prob_ab),
      lambda: -_bound_pdf(b, prob_ab),
  )
  if not log:
    dval = jnp.where(jnp.isfinite(val), val, 0.0) * dval
  return val, dval


def _cdf_parts(
    z: Array, a: Array, b: Array
) -> tuple[Array, _FactoredExp, _FactoredExp, _FactoredExp]:
  """Splits the CDF into two sub-intervals P(a, z) and P(z, b) for a < z < b.

  Because P(a, b) = P(a, z) + P(z, b) with both terms positive, CDF, logCDF, and
  their derivatives follow from just these two pieces without a third
  _normal_prob call on [a, b].

  Returns (is_left, prob_az, denom, ratio) where:
    - is_left: True when P(a, z) <= P(z, b) (z is in the left half).
    - prob_az: P(a, z) (used by the logCDF JVP).
    - denom:   max(P(a, z), P(z, b)), the larger term.
    - ratio:   min / max <= 1, safe from overflow.
  """
  prob_az = _normal_prob(a, z)
  prob_zb = _normal_prob(z, b)
  is_left = prob_az <= prob_zb
  numer, denom = _select([(is_left, (prob_az, prob_zb))], (prob_zb, prob_az))
  return is_left, prob_az, denom, numer / denom


def _cdf_from_ratio(is_left: Array, ratio: _FactoredExp, log: bool) -> Array:
  """Converts the min/max `ratio` from _cdf_parts into CDF or logCDF.

  With ratio = min(P(a, z), P(z, b)) / max(...):
    - CDF is ratio / (1 + ratio) in the left half, else 1 / (1 + ratio).
    - logCDF is log(ratio / (1 + ratio)) in the left half, else -log(1 + ratio).
  """
  ratio_val = ratio.value()
  prob_min = ratio_val / (1.0 + ratio_val)
  if log:
    log1p_ratio = _log1p(ratio_val)
    # For ratio >= eps, log(ratio / (1 + ratio)) avoids cancellation inside
    # ratio.log_value() when log(coeff) and exponent.hi nearly cancel near
    # ratio ~ 1. Below eps, ratio.log_value() - log1p(ratio) stays finite even
    # if ratio_val underflows to 0.
    eps = jnp.asarray(dtypes.finfo(ratio_val.dtype).eps, dtype=ratio_val.dtype)
    log_prob_min = jnp.where(
        ratio_val >= eps, _log(prob_min), ratio.log_value() - log1p_ratio
    )
    return jnp.where(is_left, log_prob_min, -log1p_ratio)
  return jnp.where(is_left, prob_min, 1.0 / (1.0 + ratio_val))


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(3,))
def _std_cdf(z: Array, a: Array, b: Array, log: bool) -> Array:
  """Evaluates standard truncnorm.cdf or truncnorm.logcdf for a < z < b."""
  is_left, _, _, ratio = _cdf_parts(z, a, b)
  return _cdf_from_ratio(is_left, ratio, log)


@partial(_std_cdf.defjvp, symbolic_zeros=True)
def _std_cdf_jvp(
    log: bool,
    primals: tuple[Array, Array, Array],
    tangents: tuple[Any, Any, Any],
) -> tuple[Array, Array]:
  z, a, b = primals
  is_left, prob_az, denom, ratio = _cdf_parts(z, a, b)
  val = _cdf_from_ratio(is_left, ratio, log)

  ratio_val = ratio.value()
  inv_1p = 1.0 / (1.0 + ratio_val)
  one = _FactoredExp.one(val.dtype)
  # With F = P(a, z) / P(a, b) and 1 - F = P(z, b) / P(a, b):
  #   cdf:    dF/dz = phi(z)/P(a,b), dF/da = -phi(a)*(1-F)/P(a,b), dF/db = -phi(b)*F/P(a,b)
  #   logcdf: d/dz  = phi(z)/P(a,z), d/da  = -phi(a)*(1-F)/P(a,z), d/db  = -phi(b)/P(a,b)
  # F and 1 - F are rebuilt from the min/max ratio as factored values, so a
  # tiny F (or 1 - F) is multiplied into phi(bound) / P before either factor
  # is converted to a plain float and can underflow.
  cdf_factor = _select([(is_left, ratio)], one) * inv_1p
  sf_factor = _select([(is_left, one)], ratio) * inv_1p
  prob_ab = denom * (1.0 + ratio_val)
  z_denom = prob_az if log else prob_ab

  return val, _dot_tangents(
      val,
      tangents,
      lambda: (_normal_pdf(z) / z_denom).value(),
      lambda: -_bound_pdf(a, z_denom, sf_factor),
      lambda: -_bound_pdf(b, prob_ab, None if log else cdf_factor),
  )


# Dividing Phi(a + delta) - Phi(a) = q * P(a, b) by phi(a) gives the target
# integral prob_over_pdf = q * P(a, b) / phi(a). We use this near-bound solver
# when prob_over_pdf <= 0.08 and |a * prob_over_pdf| <= 0.08, where the
# 6th-order bivariate series reversion has < 3e-7 relative error so one Newton
# step refines delta = z - a to ~1 ULP; outside that box, z is far enough from
# the bound for _ppf_near_zero or _ppf_tail.
_NEAR_BOUND_LIMIT = 0.08

# Bivariate series-reversion coefficients of delta / prob_over_pdf in powers
# of prob_over_pdf^2 (rows) and a * prob_over_pdf (columns within each row).
_NEAR_BOUND_COEFFS = (
    (1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0),
    (1.0 / 6.0, 7.0 / 24.0, 23.0 / 60.0, 163.0 / 360.0),
    (7.0 / 120.0, 127.0 / 720.0),
)

# Nodes and weights of the 4-point Gauss-Legendre rule on [0, 1].
_GL4_NODES = (
    0.06943184420297371,
    0.33000947820757187,
    0.6699905217924281,
    0.9305681557970263,
)
_GL4_WEIGHTS = (
    0.17392742256872692,
    0.32607257743127305,
    0.32607257743127305,
    0.17392742256872692,
)


def _ppf_bound_delta(a: Array, prob_over_pdf: Array) -> Array:
  """Solves integral_0^delta exp(-a * t - t^2 / 2) dt = prob_over_pdf.

  This is Phi(a + delta) - Phi(a) = prob_over_pdf * phi(a) divided by phi(a).
  Its solution delta is the offset z - a from the bound, found without ever
  forming the difference Phi(a + delta) - Phi(a).

  Initializes delta from a 6th-order bivariate series reversion in
  (prob_over_pdf, a * prob_over_pdf), then takes one Newton step using 4-point
  Gauss-Legendre quadrature on [0, 1] via t = delta * u:
    f(delta)  = (delta - prob_over_pdf)
                + delta * int_0^1 (exp(-t * (a + t / 2)) - 1) du
    f'(delta) = exp(-delta * (a + delta / 2)).
  Separating delta - prob_over_pdf and using expm1 avoids cancelling delta
  against itself when delta ~ prob_over_pdf.
  """
  row_coeffs = [
      lax.polynomial(a * prob_over_pdf, row) for row in _NEAR_BOUND_COEFFS
  ]
  delta = prob_over_pdf * lax.polynomial(prob_over_pdf**2, row_coeffs)
  shape = (len(_GL4_NODES),) + (1,) * delta.ndim
  nodes = jnp.asarray(_GL4_NODES, dtype=delta.dtype).reshape(shape)
  weights = jnp.asarray(_GL4_WEIGHTS, dtype=delta.dtype).reshape(shape)
  t = nodes * delta[None]
  quad_terms = _expm1(-t * (a[None] + 0.5 * t))
  quad_avg = jnp.sum(weights * quad_terms, axis=0)
  fx = (delta - prob_over_pdf) + delta * quad_avg
  dfx = _exp(-delta * (a + 0.5 * delta))
  return delta - fx / dfx


def _ppf_near_bound(
    q: Array, a: Array, b: Array, prob_ab: _FactoredExp
) -> tuple[Array, Array]:
  """Inverts the CDF when the answer z is close to a finite bound.

  When q is tiny, Phi(a) + q * P rounds toward Phi(a) and loses precision in
  delta = z - a. Writing z = a + delta and dividing Phi(a + delta) - Phi(a) =
  q * (Phi(b) - Phi(a)) by phi(a) gives the integral equation solved by
  _ppf_bound_delta with prob_over_pdf = q * P(a, b) / phi(a). For q > 0.5 we
  reflect via (1 - q, -b, -a) to solve from the upper bound instead.

  Returns (is_near_bound, z_bound).
  """
  assert prob_ab.anchor is not None
  use_upper = q > 0.5
  a, q = _select([(use_upper, (-b, 1.0 - q))], (a, q))
  is_finite = jnp.isfinite(a)
  safe_a = jnp.where(is_finite, a, prob_ab.anchor)

  # Check prob_over_pdf <= _NEAR_BOUND_LIMIT in log space first so huge
  # values do not overflow, then check |safe_a * prob_over_pdf_val| against it.
  prob_over_pdf = (prob_ab / _normal_pdf(safe_a)).scale_by(q)
  is_small = is_finite & (
      prob_over_pdf.log_value() <= math.log(_NEAR_BOUND_LIMIT)
  )
  one = _FactoredExp.one(q.dtype)
  prob_over_pdf_val = _select([(is_small, prob_over_pdf)], one).value()
  in_box = jnp.abs(safe_a * prob_over_pdf_val) <= _NEAR_BOUND_LIMIT
  is_near = is_small & in_box
  safe_a, safe_prob_over_pdf = _select(
      [(is_near, (safe_a, prob_over_pdf_val))], (0.0, 0.0)
  )
  z = safe_a + _ppf_bound_delta(safe_a, safe_prob_over_pdf)
  return is_near, jnp.where(use_upper, -z, z)


_NEAR_ZERO_MAX_ERF = 0.5
# Maclaurin coefficients c_m of z(t) = t + t^3 * sum_{m>=0} c_m * t^{2m} for
# t = integral_0^z exp(-v^2 / 2) dv = sqrt(pi / 2) * erf(z / sqrt(2)), which
# satisfies z'(t) = exp(z^2 / 2), z(0) = 0, so z''(t) = z(t) * z'(t)^2
# (c_0 = 1/6, c_1 = 7/120, c_2 = 127/5040, ...; verified in truncnorm_test.py).
_NEAR_ZERO_COEFFS_F64 = (
    float.fromhex("0x1.5555555555555p-3"),
    float.fromhex("0x1.ddddddddddddep-5"),
    float.fromhex("0x1.9cd9cd9cd9cdap-6"),
    float.fromhex("0x1.8a851a18bdfdap-7"),
    float.fromhex("0x1.9006cdbe0e0b5p-8"),
    float.fromhex("0x1.a5c1b3da6c064p-9"),
    float.fromhex("0x1.c922328217d37p-10"),
    float.fromhex("0x1.f9cb87937811dp-11"),
    float.fromhex("0x1.1c5148d24cf01p-11"),
    float.fromhex("0x1.43bae454210d6p-12"),
    float.fromhex("0x1.7472621543157p-13"),
    float.fromhex("0x1.b033a9df1a262p-14"),
    float.fromhex("0x1.f934384a9456bp-15"),
    float.fromhex("0x1.291c4665cefb5p-15"),
    float.fromhex("0x1.5f58ac4f47dd4p-16"),
    float.fromhex("0x1.a17099f5c2048p-17"),
    float.fromhex("0x1.f20627ad93349p-18"),
    float.fromhex("0x1.2a2d0db9c2265p-18"),
    float.fromhex("0x1.66389885754e4p-19"),
    float.fromhex("0x1.afa143c585b51p-20"),
    float.fromhex("0x1.04bc59e13a475p-20"),
    float.fromhex("0x1.3bc5490013ac3p-21"),
    float.fromhex("0x1.7f44a27f43788p-22"),
    float.fromhex("0x1.d222cbc779e46p-23"),
    float.fromhex("0x1.1bfd007dbffc4p-23"),
)
_NEAR_ZERO_COEFFS_F32 = _NEAR_ZERO_COEFFS_F64[:11]


def _half_erf(x: Array, tail_ndtr: Array) -> Array:
  """Computes Phi(x) - 0.5 = 0.5 * erf(x / sqrt(2)) without cancellation.

  Uses the erf polynomial directly for |x| <= sqrt(2); for |x| > sqrt(2),
  computes +-(0.5 - tail_ndtr) from the precomputed tail_ndtr = Phi(-|x|).
  """
  # Share _ERF*_T_M1 with ndtr so the |x| = sqrt(2) handoff is continuous
  # (lax.erf has no AccuracyMode.HIGHEST path, measuring ~4.2 max ULP vs ~2.1
  # ULP here on normal TPU float32 inputs).
  coeffs = _ERF32_T_M1 if x.dtype == np.float32 else _ERF64_T_M1
  is_small = jnp.abs(x) <= _SQRT_2
  y = jnp.where(is_small, x, 0.0) * _INV_SQRT_2
  half_y = 0.5 * y
  one_minus_y2 = (1.0 - y) * (1.0 + y)
  poly = half_y + half_y * lax.polynomial(
      one_minus_y2, coeffs, unroll=len(coeffs)
  )
  tail = 0.5 - tail_ndtr
  return _select([(is_small, poly), (x < 0, -tail)], tail)


def _ndtri_near_zero(erf_z: Array) -> Array:
  """Inverts erf(z / sqrt(2)) = erf_z near zero via Maclaurin series.

  Sets t = sqrt(pi / 2) * erf_z (splitting sqrt(pi / 2) as scale_hi + scale_lo
  for extra precision in the leading term) and computes z(t) = t + t^3 * P(t^2).
  """
  # Avoid lax.erf_inv / erf_inv_core here: both measure ~177 max ULP on TPU
  # float32 over |erf_z| <= 0.5 (because erf_inv_core calls default-accuracy
  # lax.log), vs ~1.1 max ULP and exact 0 at erf_z = 0 for this series.
  if erf_z.dtype == np.float32:
    scale_hi, scale_lo = _SQRT_HALF_PI_F32
    coeffs = _NEAR_ZERO_COEFFS_F32
  else:
    scale_hi, scale_lo = _SQRT_HALF_PI_F64
    coeffs = _NEAR_ZERO_COEFFS_F64
  t = scale_hi * erf_z
  t2 = t * t
  poly = lax.polynomial(t2, coeffs, unroll=len(coeffs))
  return t + (scale_lo * erf_z + (t * t2) * poly)


def _ppf_near_zero(
    q: Array, a: Array, b: Array, prob_ab: _FactoredExp
) -> tuple[Array, Array]:
  """Inverts the truncated normal CDF when the answer z is near zero.

  Near z = 0, log Phi(z) rounds near -log(2) and loses relative precision in z.
  Evaluates erf(z / sqrt(2)) = 2 * Phi(z) - 1 from half_erf_a = Phi(a) - 0.5
  and half_erf_b = Phi(b) - 0.5 (with P = Phi(b) - Phi(a)), then inverts it
  with _ndtri_near_zero:
    - q <= 0.25:       2 * (half_erf_a + q * P)
    - 0.25 < q < 0.75: 2 * (q - 0.5) * P + (half_erf_a + half_erf_b)
    - q >= 0.75:       2 * (half_erf_b - (1 - q) * P), with exact 1 - q

  Returns (is_near_zero, z_zero).
  """
  tail_a = _ndtr_neg_abs(a)
  tail_b = _ndtr_neg_abs(b)
  half_erf_a = _half_erf(a, tail_a)
  half_erf_b = _half_erf(b, tail_b)
  spans_zero = (a < 0) & (b > 0)
  # P = half_erf_b - half_erf_a. For narrow same-sign intervals this subtracts
  # two close numbers, so use quadrature instead. Across 0, signs oppose so
  # nothing cancels; on wide intervals, the two terms are far apart.
  use_quad_ab = _use_quad(a, b) & ~spans_zero
  prob_val = jnp.where(use_quad_ab, prob_ab.value(), half_erf_b - half_erf_a)

  # When a < 0 < b, half_erf_a + half_erf_b = Phi(b) - Phi(-a); use
  # quadrature when -a and b are close (giving exact 0 at -a == b).
  quad_sum = _normal_prob_quad(-a, b).value()
  spans_zero_sum = jnp.where(_use_quad(-a, b), quad_sum, tail_a - tail_b)
  half_erf_sum = jnp.where(spans_zero, spans_zero_sum, half_erf_a + half_erf_b)
  erf_z = _select(
      [
          (q <= 0.25, 2.0 * (half_erf_a + q * prob_val)),
          (q >= 0.75, 2.0 * (half_erf_b - (1.0 - q) * prob_val)),
      ],
      2.0 * (q - 0.5) * prob_val + half_erf_sum,
  )
  return jnp.abs(erf_z) <= _NEAR_ZERO_MAX_ERF, _ndtri_near_zero(erf_z)


def _log_ndtr_add_exp(x: Array, log_y: Array) -> Array:
  """Evaluates log(Phi(x) + exp(log_y)) without underflow or overflow."""
  log_cdf = special.log_ndtr(x)
  hi, lo = jnp.maximum(log_cdf, log_y), jnp.minimum(log_cdf, log_y)
  is_neginf_hi = jnp.isneginf(hi)
  safe_lo, safe_hi = _select([(is_neginf_hi, (0.0, 0.0))], (lo, hi))
  safe_diff = safe_lo - safe_hi
  return jnp.where(is_neginf_hi, hi, hi + _log1p(_exp(safe_diff)))


def _ppf_tail(q: Array, a: Array, b: Array, prob_ab: _FactoredExp) -> Array:
  """Inverts the truncated normal CDF via log-space inversion (general/tail).

  Computes log Phi(z) = log(Phi(a) + q * P) and log Phi(-z) =
  log(Phi(-b) + (1 - q) * P), then inverts whichever tail is smaller (<= 0.5)
  via ndtri_exp.
  """
  log_prob = prob_ab.log_value()
  log_cdf_z = _log_ndtr_add_exp(a, _log(q) + log_prob)
  log_sf_z = _log_ndtr_add_exp(-b, _log1p(-q) + log_prob)

  # When anchor^2 overflows log_prob to -inf, the interval has zero width in
  # float precision, so collapse onto prob_ab.anchor.
  is_zero_prob = jnp.isneginf(log_prob)
  use_left = log_cdf_z <= log_sf_z
  log_tail = _select([(is_zero_prob, -_LOG_2), (use_left, log_cdf_z)], log_sf_z)
  z_inv = special.ndtri_exp(log_tail)
  return _select([(is_zero_prob, prob_ab.anchor), (use_left, z_inv)], -z_inv)


def _tail_erfcx(x: Array) -> Array:
  """Evaluates erfcx(|x| / sqrt(2)) for finite |x| > sqrt(2), guarded for AD."""
  abs_x = jnp.abs(x)
  is_ok = (abs_x > _SQRT_2) & jnp.isfinite(abs_x)
  safe_abs_x = jnp.where(is_ok, abs_x, 2.0)
  return jnp.where(is_ok, special.erfcx(safe_abs_x * _INV_SQRT_2), 0.0)


def _ppf_pdf_z(
    q: Array, a: Array, b: Array, z: Array, prob_ab: _FactoredExp
) -> _FactoredExp:
  """Evaluates phi(z) at z = ppf(q, a, b) without amplifying rounding in z.

  Evaluating exp(-z^2 / 2) directly at a rounded quantile z amplifies a
  0.5-ULP error in z by z^2 / 2 ULPs in the tails. For |z| > sqrt(2), we
  instead reconstruct Phi(-|z|) — Phi(a) + q * P(a, b) for z < 0, or
  Phi(-b) + (1 - q) * P(a, b) for z >= 0 — as a sum of two non-negative
  _FactoredExp terms, then divide out erfcx(|z| / sqrt(2)) via:
    phi(z) = 2 * Phi(-|z|) / (sqrt(2 * pi) * erfcx(|z| / sqrt(2))).
  """
  assert prob_ab.anchor is not None
  erfcx_z = _tail_erfcx(z)
  safe_erfcx_z = jnp.where(erfcx_z > 0.0, erfcx_z, 1.0)
  use_left = z < 0.0
  a, q = _select([(use_left, (a, q))], (-b, 1.0 - q))
  safe_a = jnp.where(jnp.isfinite(a), a, prob_ab.anchor)
  bound_coeff = 0.5 * _tail_erfcx(a)
  bound_tail = _FactoredExp.from_coeff(bound_coeff, anchor=safe_a)
  quantile_tail = prob_ab.scale_by(q)
  tail_prob = bound_tail + quantile_tail
  pdf_tail = tail_prob * _FactoredExp(2.0 / safe_erfcx_z, _neg_half_log_2pi(z))
  return _select([(erfcx_z > 0.0, pdf_tail)], _normal_pdf(z))


@custom_derivatives.custom_jvp
def _ppf_interior(q: Array, a: Array, b: Array) -> Array:
  """Evaluates standard truncnorm.ppf for 0 < q < 1 and a < b."""
  prob_ab = _normal_prob(a, b)
  near_bound = _ppf_near_bound(q, a, b, prob_ab)
  near_zero = _ppf_near_zero(q, a, b, prob_ab)
  z = _select([near_bound, near_zero], _ppf_tail(q, a, b, prob_ab))
  return _select([(z < a, a), (z > b, b)], z)


@partial(_ppf_interior.defjvp, symbolic_zeros=True)
def _ppf_interior_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Any, Any, Any],
) -> tuple[Array, Array]:
  """Differentiates z = ppf(q, a, b) without differentiating the solver.

  z is defined by Phi(z) = (1 - q) * Phi(a) + q * Phi(b). Differentiating both
  sides gives:
    phi(z) * dz = P(a, b) * dq + (1 - q) * phi(a) * da + q * phi(b) * db,
  so dividing by phi(z) gives each partial derivative in closed form.
  """
  q, a, b = primals
  z = _ppf_interior(q, a, b)
  prob_ab = _normal_prob(a, b)
  pdf_z = _ppf_pdf_z(q, a, b, z, prob_ab)
  return z, _dot_tangents(
      z,
      tangents,
      lambda: (prob_ab / pdf_z).value(),
      lambda: _bound_pdf(a, pdf_z, 1.0 - q),
      lambda: _bound_pdf(b, pdf_z, q),
  )


def _prepare_args(name: str, *args: ArrayLike) -> tuple[list[Array], np.dtype]:
  """Promotes inputs to a common float dtype; upcasts bf16/f16 to f32.

  Returns the promoted arrays (at their input shapes, without upfront
  broadcasting) and the original promoted dtype so callers can cast back once.
  """
  promoted = promote_args_inexact(f"truncnorm.{name}", *args)
  dtype = promoted[0].dtype
  if dtypes.issubdtype(dtype, np.complexfloating):
    raise TypeError(f"{name} does not support complex-valued inputs.")
  if dtypes.finfo(dtype).bits < 32:
    promoted = [v.astype(np.float32) for v in promoted]
  return promoted, dtype


def _pdf_or_logpdf(
    name: str,
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    *,
    log: bool = False,
) -> Array:
  """Evaluates truncnorm.pdf or truncnorm.logpdf across all inputs."""
  (x, a, b, loc, scale), dtype = _prepare_args(name, x, a, b, loc, scale)
  is_pos = scale > 0
  safe_scale = jnp.where(is_pos, scale, 1.0)
  z = (x - loc) / safe_scale
  has_pos_width = a < b
  safe_a, safe_b = _select([(has_pos_width, (a, b))], (-1.0, 1.0))
  is_invalid = ~is_pos | ~has_pos_width | jnp.isnan(z)
  in_support = ~is_invalid & (z >= a) & (z <= b) & jnp.isfinite(z)
  fallback_z = lax.stop_gradient(jnp.clip(0.0, safe_a, safe_b))
  safe_z = jnp.where(in_support, z, fallback_z)
  interior = _std_pdf_or_logpdf(safe_z, safe_a, safe_b, log)
  val = interior - _log(safe_scale) if log else interior / safe_scale
  outside = -np.inf if log else 0.0
  out = _select([(is_invalid, np.nan), (in_support, val)], outside)
  return out.astype(dtype)


def _interior_bounds_and_mid(a: Array, b: Array) -> tuple[Array, Array, Array]:
  """Returns (safe_a, safe_b, safe_mid) with safe_a < safe_mid < safe_b.

  _std_cdf splits [a, b] at z, so masked-out lanes need a strictly interior
  point even when one or both bounds are infinite (where (a + b) / 2 would be
  infinite or NaN). Falls back to (-1, 1, 0) where a < b does not hold.
  """
  a_sg, b_sg = lax.stop_gradient(a), lax.stop_gradient(b)
  fmax = jnp.asarray(dtypes.finfo(a.dtype).max, dtype=a.dtype)
  near_bound = jnp.where(a_sg >= 0.0, a_sg, b_sg)
  offset = jnp.minimum(
      jnp.maximum(1.0, 0.25 * jnp.abs(near_bound)), 0.5 * (b_sg - a_sg)
  )
  spans_zero = (a_sg < 0.0) & (b_sg > 0.0)
  same_side_mid = near_bound + jnp.where(a_sg >= 0.0, offset, -offset)
  mid = jnp.where(spans_zero, 0.0, same_side_mid)
  mid = jnp.clip(mid, -fmax, fmax)
  has_interior_mid = (a < mid) & (mid < b)
  return _select([(has_interior_mid, (a, b, mid))], (-1.0, 1.0, 0.0))


def _cdf_or_sf(
    name: str,
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    *,
    survival: bool = False,
    log: bool = False,
) -> Array:
  """Evaluates truncnorm.(log)cdf or truncnorm.(log)sf across all inputs."""
  (x, a, b, loc, scale), dtype = _prepare_args(name, x, a, b, loc, scale)
  is_pos = scale > 0
  safe_scale = jnp.where(is_pos, scale, 1.0)
  z = (x - loc) / safe_scale
  if survival:
    # By normal symmetry, sf(z; a, b) = cdf(-z; -b, -a).
    z, a, b = -z, -b, -a
  safe_a, safe_b, safe_mid = _interior_bounds_and_mid(a, b)
  has_pos_width = a < b
  is_invalid = ~is_pos | ~has_pos_width | jnp.isnan(z)
  in_support = ~is_invalid & (z > a) & (z < b)
  safe_z = jnp.where(in_support, z, safe_mid)
  val = _std_cdf(safe_z, safe_a, safe_b, log)
  lo, hi = (-np.inf, 0.0) if log else (0.0, 1.0)
  out = _select([(is_invalid, np.nan), (z <= a, lo), (z >= b, hi)], val)
  return out.astype(dtype)


def _ppf_or_isf(
    name: str,
    q: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    *,
    survival: bool = False,
) -> Array:
  """Evaluates truncnorm.ppf or truncnorm.isf across all inputs."""
  (q, a, b, loc, scale), dtype = _prepare_args(name, q, a, b, loc, scale)
  if survival:
    # By normal symmetry, isf(q; a, b) = -ppf(q; -b, -a).
    a, b = -b, -a
  has_pos_width = a < b
  safe_a, safe_b = _select([(has_pos_width, (a, b))], (-1.0, 1.0))
  safe_q = jnp.where((q > 0) & (q < 1), q, 0.5)
  is_pos = scale > 0
  is_invalid = ~(has_pos_width & (q >= 0) & (q <= 1) & is_pos)
  z = _ppf_interior(safe_q, safe_a, safe_b)
  z = _select([(q == 0, a), (q == 1, b)], z)
  if survival:
    z = -z
  safe_z, safe_scale = _select([(~is_invalid, (z, scale))], (0.0, 1.0))
  return jnp.where(is_invalid, np.nan, loc + safe_z * safe_scale).astype(dtype)


@api.jit(inline=api.Inline.JAX_EARLY)
def logpdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal log probability distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logpdf``.

  For ``loc = 0`` and ``scale = 1``, the truncated normal density is

  .. math::

     f(x, a, b) = \begin{cases}
       \frac{1}{\sqrt{2\pi}(\Phi(b) - \Phi(a))}e^{-x^2/2} & a \le x \le b \\
       0 & \mathrm{otherwise}
     \end{cases}

  where :math:`\Phi` is the standard normal CDF. Following
  :obj:`scipy.stats.truncnorm`, :math:`a` and :math:`b` are specified in
  standard deviations from ``loc`` rather than in :math:`x`-coordinates, so for
  general ``loc`` and ``scale`` the support is
  ``[loc + a * scale, loc + b * scale]`` with standardized coordinate
  ``z = (x - loc) / scale``.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return _pdf_or_logpdf("logpdf", x, a, b, loc, scale, log=True)


@api.jit(inline=api.Inline.JAX_EARLY)
def pdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal probability distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``pdf``.

  For ``loc = 0`` and ``scale = 1``, the truncated normal density is

  .. math::

     f(x, a, b) = \begin{cases}
       \frac{1}{\sqrt{2\pi}(\Phi(b) - \Phi(a))}e^{-x^2/2} & a \le x \le b \\
       0 & \mathrm{otherwise}
     \end{cases}

  where :math:`\Phi` is the standard normal CDF. Following
  :obj:`scipy.stats.truncnorm`, :math:`a` and :math:`b` are specified in
  standard deviations from ``loc`` rather than in :math:`x`-coordinates, so for
  general ``loc`` and ``scale`` the support is
  ``[loc + a * scale, loc + b * scale]`` with standardized coordinate
  ``z = (x - loc) / scale``.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return _pdf_or_logpdf("pdf", x, a, b, loc, scale)


@api.jit(inline=api.Inline.JAX_EARLY)
def logsf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  """Truncated normal distribution log survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logsf``.

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return _cdf_or_sf("logsf", x, a, b, loc, scale, survival=True, log=True)


@api.jit(inline=api.Inline.JAX_EARLY)
def sf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  """Truncated normal distribution survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return _cdf_or_sf("sf", x, a, b, loc, scale, survival=True)


@api.jit(inline=api.Inline.JAX_EARLY)
def logcdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf} = \int_{-\infty}^x f_{pdf}(y) \mathrm{d}y

  where here :math:`f_{pdf}` is the probability distribution function,
  :func:`jax.scipy.stats.truncnorm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return _cdf_or_sf("logcdf", x, a, b, loc, scale, log=True)


@api.jit(inline=api.Inline.JAX_EARLY)
def cdf(
    x: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf} = \int_{-\infty}^x f_{pdf}(y) \mathrm{d}y

  where here :math:`f_{pdf}` is the probability distribution function,
  :func:`jax.scipy.stats.truncnorm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return _cdf_or_sf("cdf", x, a, b, loc, scale)


@api.jit(inline=api.Inline.JAX_EARLY)
def ppf(
    q: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal percent point function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``ppf``.

  The percent point function is defined as the inverse of the cumulative
  distribution function, :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    q: arraylike, value at which to evaluate the PPF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of ppf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return _ppf_or_isf("ppf", q, a, b, loc, scale)


@api.jit(inline=api.Inline.JAX_EARLY)
def isf(
    q: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal inverse survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``isf``.

  Returns the inverse of the survival function,
  :func:`jax.scipy.stats.truncnorm.sf`.

  Args:
    q: arraylike, value at which to evaluate the ISF
    a: arraylike, lower truncation bound in standard deviations from ``loc``
    b: arraylike, upper truncation bound in standard deviations from ``loc``
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of isf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
  """
  return _ppf_or_isf("isf", q, a, b, loc, scale, survival=True)
