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

"""Base-2 shifted exponential and logarithmic functions (IEEE 754-2019)."""

from functools import partial

from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.lax.double_word import (
    fast_two_sum,
    split26_f64,
    split_half_mask,
    two_sum,
)
from jax._src.lax.lax import (
    RoundingMethod,
    _const,
    _float,
    _one,
    add,
    bitcast_convert_type,
    clamp,
    convert_element_type,
    div,
    exp2,
    full_like,
    max as lax_max,
    min as lax_min,
    mul,
    polynomial,
    round as lax_round,
    select,
    shift_left,
    shift_right_arithmetic,
    standard_unop,
)
from jax._src.typing import Array, ArrayLike
import numpy as np

export = util.set_module("jax.lax")

# Polynomial coefficients for exp2m1(r) = 2^r - 1 on r in [-0.5, 0.5], in
# ascending degree order for lax.polynomial.
#
# For float32 and float64, ln(2) is split into ln2_hi + ln2_lo so r_hi * ln2_hi
# is exact when r is split via split_half_mask, and _EXP2M1_{32,64} hold the
# quadratic-and-higher coefficients (degrees 2..deg). Narrow (< 32-bit) float
# dtypes promoted to float32 use the plain 5-term _EXP2M1_16 polynomial.
#
# Generated with Sollya 8.0:
#   prec = 350!;
#   display = hexadecimal!;
#   p16 = fpminimax((2^x - 1) / x, [|0, ..., 4|], [|SG...|], [-0.5; 0.5],
#                   floating, relative);
#   /* ln2_hi = 0x1.62ep-1 (float32) or 0x1.62e43p-1 (float64) */
#   ln2_lo = round(log(2) - ln2_hi, prec_bits, RN);
#   p = fpminimax((2^x - 1) / x, [|1, ..., deg|], [|SG...|], [-0.5; 0.5],
#                 floating, relative, ln2_hi + ln2_lo);

_EXP2M1_16 = (
    float.fromhex("0x1.62e42ep-1"),
    float.fromhex("0x1.ebfa4ep-3"),
    float.fromhex("0x1.c6b31ep-5"),
    float.fromhex("0x1.3cbe66p-7"),
    float.fromhex("0x1.5d413ap-10"),
)

_LN2_HI_32 = float.fromhex("0x1.62ep-1")
_LN2_LO_32 = float.fromhex("0x1.0bfbe8p-15")

_EXP2M1_32 = (
    float.fromhex("0x1.ebfbep-3"),
    float.fromhex("0x1.c6b08cp-5"),
    float.fromhex("0x1.3b2a54p-7"),
    float.fromhex("0x1.5d8a08p-10"),
    float.fromhex("0x1.44175ap-13"),
    float.fromhex("0x1.fde278p-17"),
)

_LN2_HI_64 = float.fromhex("0x1.62e43p-1")
_LN2_LO_64 = float.fromhex("-0x1.05c610ca86c39p-29")

_EXP2M1_64 = (
    float.fromhex("0x1.ebfbdff82c58fp-3"),
    float.fromhex("0x1.c6b08d704a0c1p-5"),
    float.fromhex("0x1.3b2ab6fba4d47p-7"),
    float.fromhex("0x1.5d87fe78a5dc3p-10"),
    float.fromhex("0x1.430912f8c2a04p-13"),
    float.fromhex("0x1.ffcbfc5db2bb5p-17"),
    float.fromhex("0x1.62c020e67a669p-20"),
    float.fromhex("0x1.b52529304ae02p-24"),
    float.fromhex("0x1.e4d402a565dcp-28"),
    float.fromhex("0x1.e912217fb2b55p-32"),
    float.fromhex("0x1.bdf14e7c17eb7p-36"),
    float.fromhex("0x1.228f10c3f04fbp-40"),
)

# Polynomial coefficients for log2p1(f) = log2(1 + f) on
# f in [sqrt(2)/2 - 1, sqrt(2) - 1]:
# - Narrow (< 32-bit) float dtypes promoted to float32 use _LOG2P1_16 directly.
# - float32 splits log2(e) into 12-bit _LOG2E_HI_32 + _LOG2E_LO_32 so
#   f_hi * _LOG2E_HI_32 is exact, with _LOG2P1_32 holding degrees 2..12.
# - float64 uses z = f / (1 + 0.5 * f), u = z * z, and splits log2(e) into
#   24-bit _LOG2E_HI_64 + _LOG2E_LO_64 so z_hi * _LOG2E_HI_64 is exact when z_s
#   is split via split26_f64.
#
# Generated with Sollya 8.0:
#   prec = 350!;
#   display = hexadecimal!;
#   f_lo = sqrt(2)/2 - 1; f_hi = sqrt(2) - 1;
#   p16 = fpminimax(log2(1 + x) / x, [|0, ..., 6|], [|SG...|], [f_lo; f_hi],
#                   floating, relative);
#   log2e_hi_32 = 0x1.714p0;
#   log2e_lo_32 = round(1 / log(2) - log2e_hi_32, 24, RN);
#   p32 = fpminimax(log2(1 + x) / x, [|1, ..., 11|], [|SG...|], [f_lo; f_hi],
#                   floating, relative, log2e_hi_32 + log2e_lo_32);
#   u_hi = (6 - 4 * sqrt(2))^2;
#   log2e_hi_64 = 0x1.715476p0;
#   log2e_lo_64 = round(1 / log(2) - log2e_hi_64, 53, RN);
#   p64 = fpminimax(log2((1 + 0.5 * sqrt(x)) / (1 - 0.5 * sqrt(x))) / sqrt(x),
#                   [|1, ..., 8|], [|D...|], [1b-120; u_hi],
#                   floating, relative, log2e_hi_64 + log2e_lo_64);

_LOG2P1_16 = (
    float.fromhex("0x1.71548ep0"),
    float.fromhex("-0x1.71569p-1"),
    float.fromhex("0x1.ec295ap-2"),
    float.fromhex("-0x1.6fff66p-2"),
    float.fromhex("0x1.2ecd1p-2"),
    float.fromhex("-0x1.13c69cp-2"),
    float.fromhex("0x1.5f6f62p-3"),
)

_LOG2E_HI_32 = float.fromhex("0x1.714p0")
_LOG2E_LO_32 = float.fromhex("0x1.47652cp-12")

_LOG2P1_32 = (
    float.fromhex("-0x1.715476p-1"),
    float.fromhex("0x1.ec709ap-2"),
    float.fromhex("-0x1.715496p-2"),
    float.fromhex("0x1.277834p-2"),
    float.fromhex("-0x1.ec769ap-3"),
    float.fromhex("0x1.a5cb2p-3"),
    float.fromhex("-0x1.6faa12p-3"),
    float.fromhex("0x1.49d23cp-3"),
    float.fromhex("-0x1.3fd648p-3"),
    float.fromhex("0x1.28aaa2p-3"),
    float.fromhex("-0x1.39f0b6p-4"),
)

_LOG2E_HI_64 = float.fromhex("0x1.715476p0")
_LOG2E_LO_64 = float.fromhex("0x1.4ae0bf85ddf44p-26")

_LOG2P1_64 = (
    float.fromhex("0x1.ec709dc3a03fdp-4"),
    float.fromhex("0x1.2776c50ef9e6ap-6"),
    float.fromhex("0x1.a61762a730a1ep-9"),
    float.fromhex("0x1.484b1435b8492p-11"),
    float.fromhex("0x1.0c9a5e6ce1908p-13"),
    float.fromhex("0x1.c6a12c15ac3a3p-16"),
    float.fromhex("0x1.87900c87145edp-18"),
    float.fromhex("0x1.86f5eabb6cdbbp-20"),
)

_SQRT2_OVER_2_BITS_32 = 0x3F3504F3
_SQRT2_OVER_2_BITS_64 = 0x3FE6A09E667F3BCD


def _exp2m1_narrow_f32(x: Array) -> Array:
  """Approximates `2^x - 1` in `float32` for narrower (< 32-bit) float inputs."""
  zero = full_like(x, 0.0)
  one = full_like(x, 1.0)
  x_safe = select(x == x, clamp(_const(x, -26.0), x, _const(x, 128.0)), zero)
  k = lax_round(x_safe, RoundingMethod.TO_NEAREST_EVEN)
  r = x_safe - k
  k_int = convert_element_type(k, np.int32)

  p = r * polynomial(r, _EXP2M1_16, unroll=len(_EXP2M1_16))
  s_mant = full_like(k_int, 23)
  bias_v = full_like(k_int, 127)
  neg_k = lax_max(full_like(k_int, -126), -k_int)
  t = bitcast_convert_type(shift_left(neg_k + bias_v, s_mant), np.float32)

  k1 = lax_min(k_int, full_like(k_int, 127))
  k2 = k_int - k1
  pow1 = bitcast_convert_type(shift_left(k1 + bias_v, s_mant), np.float32)
  pow2 = bitcast_convert_type(shift_left(k2 + bias_v, s_mant), np.float32)

  res = (((one - t) + p) * pow2) * pow1
  res = select(x == 0.0, x, res)
  res = select(x <= -26.0, full_like(x, -1.0), res)
  res = select(x >= 128.0, full_like(x, np.inf), res)
  return select(x == x, res, full_like(x, np.nan))


def _exp2m1_impl(x: Array) -> Array:
  """Approximates `2^x - 1`.

  Splits `x` into its nearest integer `k` and a small remainder `r = x - k` in
  `[-0.5, 0.5]`, so `2^r - 1` can be approximated with a short polynomial
  around zero:
  - When `k == 0` (`|x| <= 0.5`), `r = x` and the result is simply `2^r - 1`,
    avoiding the precision loss of computing `2^x` and subtracting `1`.
  - When `k != 0`, we rewrite `2^(k + r) - 1` as `2^k * ((1 - 2^-k) + (2^r - 1))`,
    add `1 - 2^-k` to the polynomial approximation of `2^r - 1`, and scale by
    `2^k` at the end.

  To keep rounding errors small, we split the leading linear term `r * ln(2)`
  into an exact high product and a low remainder, fold the higher-degree terms
  into `p_hi` via `fast_two_sum`, and keep any low bits rounded off from
  `1 - 2^-k` before combining the pieces once.
  """
  is_f64 = x.dtype == np.float64
  dtype = x.dtype
  if is_f64:
    int_dtype = np.int64
    mant_bits, bias, emin, emax, p_bits = 52, 1023, -1022, 1023, 53
    ln2_hi, ln2_lo, coeffs = _LN2_HI_64, _LN2_LO_64, _EXP2M1_64
  else:
    int_dtype = np.int32
    mant_bits, bias, emin, emax, p_bits = 23, 127, -126, 127, 24
    ln2_hi, ln2_lo, coeffs = _LN2_HI_32, _LN2_LO_32, _EXP2M1_32

  x_under = float(-(p_bits + 2))
  x_over = float(emax + 1)
  zero = full_like(x, 0.0)
  one = full_like(x, 1.0)

  # Clamp out-of-range and NaN inputs before converting k to an integer.
  x_safe = select(x == x, clamp(_const(x, x_under), x, _const(x, x_over)), zero)
  k = lax_round(x_safe, RoundingMethod.TO_NEAREST_EVEN)
  r = x_safe - k  # Exact subtraction; 0 at integers and x for |x| <= 0.5.

  k_int = convert_element_type(k, int_dtype)

  # For very small r (only possible when k == 0), r_lo * ln2_hi and r * ln2_lo
  # would round down to zero. Scale r up by a power of two before splitting,
  # and fold the inverse scale factor into pow2 afterward.
  small = (k_int == 0) & (abs(r) < (2.0**-960 if is_f64 else 2.0**-96))
  up = select(small, full_like(r, 2.0**512 if is_f64 else 2.0**64), one)
  r_s = r * up
  r_hi, r_lo = split_half_mask(r_s)
  ln2_hi_c = _const(r, ln2_hi)
  p_hi, q_lo = fast_two_sum(
      r_hi * ln2_hi_c,
      (r_s * r) * polynomial(r, coeffs, unroll=len(coeffs)),
  )
  p_lo = q_lo + (r_lo * ln2_hi_c + r_s * _const(r, ln2_lo))

  s_mant = full_like(k_int, mant_bits)
  bias_v = full_like(k_int, bias)

  # Construct t = 2^-k (0 when k > p_bits + 2 where 2^-k is negligible).
  neg_k_clamped = lax_max(full_like(k_int, emin), -k_int)
  t = select(
      k_int <= p_bits + 2,
      bitcast_convert_type(shift_left(neg_k_clamped + bias_v, s_mant), dtype),
      zero,
  )

  # Split 2^k = 2^k1 * 2^k2 so k = emax + 1 with 1 + poly < 1 does not overflow,
  # and absorb the tiny-input `down` scale factor into k2 when k == 0.
  k1 = lax_min(k_int, full_like(k_int, emax))
  k2 = select(small, full_like(k_int, -512 if is_f64 else -64), k_int - k1)
  pow1 = bitcast_convert_type(shift_left(k1 + bias_v, s_mant), dtype)
  pow2 = bitcast_convert_type(shift_left(k2 + bias_v, s_mant), dtype)

  # Split 1 - 2^-k into b_hi + b_lo so we don't lose -2^-k when k > p_bits
  # (where 1 - t rounds to 1) or +1 when k < -p_bits (where 1 - t rounds to -t).
  # Since |b_hi| >= 0.5 > 0.4143 >= |p_hi| whenever k != 0 (and b_hi == 0 when
  # k == 0), fast_two_sum(b_hi, p_hi) captures any bits shifted off when adding.
  b_hi = one - t
  b_lo = select(k_int > p_bits, -t, select(k_int < -p_bits, one, zero))
  s, e = fast_two_sum(b_hi, p_hi)
  m = s + (e + (b_lo + p_lo))
  res = (m * pow2) * pow1

  res = select(x == 0.0, x, res)
  res = select(x <= x_under, full_like(x, -1.0), res)
  res = select(x >= x_over, full_like(x, np.inf), res)
  return select(x == x, res, full_like(x, np.nan))


def _log2p1_narrow_f32(x: Array) -> Array:
  """Approximates `log2(1 + x)` in `float32` for narrower (< 32-bit) float inputs."""
  zero = full_like(x, 0.0)
  one = full_like(x, 1.0)
  x_safe = select((x > -1.0) & (x < np.inf), x, zero)

  u = one + x_safe
  u_bits = bitcast_convert_type(u, np.int32)
  s_mant = full_like(u_bits, 23)
  exp_off = shift_right_arithmetic(
      u_bits - _const(u_bits, _SQRT2_OVER_2_BITS_32), s_mant
  )
  m = bitcast_convert_type(u_bits - shift_left(exp_off, s_mant), np.float32)

  f = select(exp_off == 0, x_safe, m - one)
  k = convert_element_type(exp_off, np.float32)
  res = k + f * polynomial(f, _LOG2P1_16, unroll=len(_LOG2P1_16))

  res = select(x == 0.0, x, res)
  res = select(x == -1.0, full_like(x, -np.inf), res)
  res = select(x == np.inf, full_like(x, np.inf), res)
  return select((x >= -1.0) & (x == x), res, full_like(x, np.nan))


def _log2p1_impl(x: Array) -> Array:
  """Approximates `log2(1 + x)` for `x >= -1`.

  Factors `1 + x` as `2^k * (1 + f)` where `k` is an integer and `1 + f` lies in
  `[sqrt(2)/2, sqrt(2))` (so `f` is in roughly `[-0.293, 0.414)`), giving
  `log2(1 + x) = k + log2(1 + f)`:
  - When `1 + x` is already in `[sqrt(2)/2, sqrt(2))` (`exp_off == 0`), `k = 0`
    and `f = x` directly. Skipping the `1 + x` addition preserves all bits of
    `x` near zero.
  - Otherwise, adding `1 + x` in a single float can round off low bits. We use
    `two_sum(1, x)` to write `1 + x = u + c` with no lost bits, extract `k` and
    `1 + f` from the exponent and fraction bits of `u`, and add a small linear
    correction for the leftover `c` term.

  We then approximate `log2(1 + f)` with a polynomial (directly in `f` for
  `float32`, or in `z = f / (1 + 0.5 * f)` for `float64` so even powers cancel),
  splitting the leading `log2(e)` product into an exact high part and a low
  remainder so `k + hi + lo` only rounds once at the end.
  """
  is_f64 = x.dtype == np.float64
  dtype = x.dtype
  if is_f64:
    int_dtype = np.int64
    mant_bits, bias, emin, emax, p_bits = 52, 1023, -1022, 1023, 53
    sqrt2_bits = _SQRT2_OVER_2_BITS_64
  else:
    int_dtype = np.int32
    mant_bits, bias, emin, emax, p_bits = 23, 127, -126, 127, 24
    sqrt2_bits = _SQRT2_OVER_2_BITS_32

  zero = full_like(x, 0.0)
  one = full_like(x, 1.0)
  in_domain = (x > -1.0) & (x < np.inf)
  x_safe = select(in_domain, x, zero)

  u, c = two_sum(one, x_safe)

  # Extract k and m = 1 + f from the bits of u. Subtracting the bit pattern of
  # sqrt(2)/2 shifts the exponent boundary from [1, 2) to [sqrt(2)/2, sqrt(2)).
  u_bits = bitcast_convert_type(u, int_dtype)
  s_mant = full_like(u_bits, mant_bits)
  bias_v = full_like(u_bits, bias)
  exp_off = shift_right_arithmetic(u_bits - _const(u_bits, sqrt2_bits), s_mant)
  m = bitcast_convert_type(u_bits - shift_left(exp_off, s_mant), dtype)

  # When exp_off == 0 (1 + x is in [sqrt(2)/2, sqrt(2))), keep k = 0 and f = x
  # directly without forming 1 + x so small inputs and +-0.0 lose no precision.
  in_k0 = exp_off == 0
  neg_exp = clamp(full_like(exp_off, emin), -exp_off, full_like(exp_off, emax))
  two_neg_k = bitcast_convert_type(shift_left(neg_exp + bias_v, s_mant), dtype)
  c_scaled = select(in_k0 | (exp_off > p_bits + 2), zero, c * two_neg_k)

  k = select(in_k0, zero, convert_element_type(exp_off, dtype))
  f = select(in_k0, x_safe, m - one)

  # For very small f (only possible when in_k0 is True), the low split product
  # would round down to zero. Scale f up by a power of two before splitting,
  # and scale the final sum back down afterward.
  small = in_k0 & (abs(f) < (2.0**-960 if is_f64 else 2.0**-96))
  up = select(small, full_like(f, 2.0**512 if is_f64 else 2.0**64), one)
  down = select(small, full_like(f, 2.0**-512 if is_f64 else 2.0**-64), one)

  # Linear correction for the low part c of 1 + x:
  # log2(1 + c_scaled / (1 + f)) ≈ log2(e) * c_scaled / (1 + f).
  corr = _const(f, 1.4426950408889634) * c_scaled * (one - f * (one - f))

  if is_f64:
    d_hi, d_lo = fast_two_sum(one, _const(f, 0.5) * f)
    inv_d = one / d_hi
    z = f * inv_d
    u2 = z * z
    z_s = z * up
    z_hi, z_lo = split26_f64(z_s)
    dh_hi, dh_lo = split26_f64(d_hi)
    p_hi = z_s * d_hi
    p_lo = (
        ((z_hi * dh_hi - p_hi) + z_hi * dh_lo) + z_lo * dh_hi
    ) + z_lo * dh_lo
    dz = ((f * up - p_hi) - (p_lo + z_s * d_lo)) * inv_d
    log2e_hi_c = _const(z, _LOG2E_HI_64)
    hi, q_lo = fast_two_sum(
        z_hi * log2e_hi_c,
        (z_s * u2) * polynomial(u2, _LOG2P1_64, unroll=len(_LOG2P1_64)),
    )
    lo = (
        q_lo
        + ((z_lo + dz) * log2e_hi_c + z_s * _const(z, _LOG2E_LO_64))
        + corr
    )
  else:
    f_s = f * up
    f_hi, f_lo = split_half_mask(f_s)
    log2e_hi_c = _const(f, _LOG2E_HI_32)
    hi, q_lo = fast_two_sum(
        f_hi * log2e_hi_c,
        (f_s * f) * polynomial(f, _LOG2P1_32, unroll=len(_LOG2P1_32)),
    )
    lo = (
        q_lo
        + (f_lo * log2e_hi_c + f_s * _const(f, _LOG2E_LO_32))
        + corr
    )

  # |k| >= 1 > 0.5 >= |hi| whenever k != 0 (and k == 0 otherwise), so
  # fast_two_sum captures any low bits of hi shifted off when adding k.
  s, e = fast_two_sum(k, hi)
  res = (s + (e + lo)) * down

  res = select(x == 0.0, x, res)
  res = select(x == -1.0, full_like(x, -np.inf), res)
  res = select(x == np.inf, full_like(x, np.inf), res)
  return select((x >= -1.0) & (x == x), res, full_like(x, np.nan))


def _exp_log_unary_lowering(impl, narrow_impl, ctx, x):
  aval = ctx.avals_in[0]

  def _lower(v):
    if aval.dtype not in (np.float32, np.float64):
      return convert_element_type(
          narrow_impl(convert_element_type(v, np.float32)), aval.dtype
      )
    return impl(v)

  return mlir.lower_fun(_lower, multiple_results=False)(ctx, x)


@export
def exp2m1(x: ArrayLike) -> Array:
  r"""Elementwise :math:`2^{x} - 1`.

  Implements the IEEE 754-2019 ``exp2m1`` operation on real floating-point
  inputs. Compared to the naive expression ``lax.exp2(x) - 1``, it avoids
  cancellation for ``x`` near zero.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`2^{x} - 1`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.exp2`: elementwise base-2 exponential :math:`2^x`.
    - :func:`jax.lax.expm1`: elementwise :math:`e^x - 1`.
    - :func:`jax.lax.log2p1`: elementwise :math:`\mathrm{log}_2(1 + x)`.
  """
  return exp2m1_p.bind(x)


@export
def log2p1(x: ArrayLike) -> Array:
  r"""Elementwise :math:`\mathrm{log}_2(1 + x)`.

  Implements the IEEE 754-2019 ``log2p1`` operation on real floating-point
  inputs. Compared to the naive expression ``lax.log2(1 + x)``, it avoids
  loss of precision for ``x`` near zero.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    base-2 logarithm of ``1 + x``.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.log2`: elementwise base-2 logarithm :math:`\mathrm{log}_2(x)`.
    - :func:`jax.lax.log1p`: elementwise :math:`\mathrm{log}(1 + x)`.
    - :func:`jax.lax.exp2m1`: elementwise :math:`2^x - 1`.
  """
  return log2p1_p.bind(x)


exp2m1_p = standard_unop(_float, "exp2m1")
ad.defjvp(
    exp2m1_p,
    lambda g, x: mul(g, mul(_const(x, float(np.log(2.0))), exp2(x))),
)
mlir.register_lowering(
    exp2m1_p,
    partial(_exp_log_unary_lowering, _exp2m1_impl, _exp2m1_narrow_f32),
    inline=False,
)

log2p1_p = standard_unop(_float, "log2p1")
ad.defjvp(
    log2p1_p,
    lambda g, x: div(
        g, mul(add(x, _one(x)), _const(x, float(np.log(2.0))))
    ),
)
mlir.register_lowering(
    log2p1_p,
    partial(_exp_log_unary_lowering, _log2p1_impl, _log2p1_narrow_f32),
    inline=False,
)
