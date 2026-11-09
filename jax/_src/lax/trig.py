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

"""Trigonometric functions scaled by pi."""

from functools import partial

from jax._src import ad_util
from jax._src import core
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.lax.double_word import fast_two_sum, split_half_mask
from jax._src.lax.lax import (
    RoundingMethod,
    _float,
    _maybe_broadcast,
    bitcast_convert_type,
    convert_element_type,
    full_like,
    hypot,
    max as lax_max,
    min as lax_min,
    one_minus_square,
    polynomial,
    round as lax_round,
    rsqrt,
    select,
    select_n,
    sqrt,
    standard_naryop,
    standard_unop,
)
from jax._src.typing import Array, ArrayLike
import numpy as np

export = util.set_module("jax.lax")

# Polynomial coefficients for sin(pi * r), cos(pi * r), and tan(pi * r) on
# r in [-0.25, 0.25] (u = r * r in [0, 0.0625]), in ascending degree order for
# lax.polynomial.
#
# For sinpi and tanpi, pi is split into pi_hi + pi_lo (with pi_lo stored as the
# constant term of the u-polynomial) and evaluated as:
#   pi_hi * r + r * polynomial(r * r, coeffs)
# so the leading pi * r term keeps full precision when r is small.
#
# Generated with Sollya 8.0:
#   prec = 350!;
#   display = hexadecimal!;
#   u_lo = 1b-120; u_hi = 0.0625;
#   pi_hi = single(pi); pi_lo = single(pi - pi_hi);  /* or double(...) */
#   p_sin = fpminimax(sin(pi * sqrt(u)) / sqrt(u), [|1, ..., deg|],
#                     [|SG...|], [u_lo; u_hi], floating, relative,
#                     pi_hi + pi_lo);
#   p_cos = fpminimax(cos(pi * sqrt(u)), [|1, ..., deg|],
#                     [|SG...|], [u_lo; u_hi], floating, relative, 1);
#   p_tan = fpminimax(tan(pi * sqrt(u)) / sqrt(u), [|1, ..., deg|],
#                     [|SG...|], [u_lo; u_hi], floating, relative,
#                     pi_hi + pi_lo);

_PI_HI_32 = float.fromhex("0x1.921fb6p1")

_SINPI_32 = (
    float.fromhex("-0x1.777a5cp-24"),
    float.fromhex("-0x1.4abbcep2"),
    float.fromhex("0x1.466b84p1"),
    float.fromhex("-0x1.32b5ap-1"),
    float.fromhex("0x1.46c086p-4"),
)

_COSPI_32 = (
    float.fromhex("0x1p0"),
    float.fromhex("-0x1.3bd3ccp2"),
    float.fromhex("0x1.03c1b8p2"),
    float.fromhex("-0x1.55b7cep0"),
    float.fromhex("0x1.d684aap-3"),
)

_TANPI_32 = (
    float.fromhex("-0x1.777a5cp-24"),
    float.fromhex("0x1.4abbdap3"),
    float.fromhex("0x1.466758p5"),
    float.fromhex("0x1.4693p7"),
    float.fromhex("0x1.3cfa48p9"),
    float.fromhex("0x1.8fbfcep11"),
    float.fromhex("0x1.2649cp7"),
    float.fromhex("0x1.ebdc6cp16"),
)

_PI_HI_64 = float.fromhex("0x1.921fb54442d18p1")

_SINPI_64 = (
    float.fromhex("0x1.1a62633145c07p-53"),
    float.fromhex("-0x1.4abbce625be53p2"),
    float.fromhex("0x1.466bc6775ab5bp1"),
    float.fromhex("-0x1.32d2cce635e46p-1"),
    float.fromhex("0x1.5078349480d19p-4"),
    float.fromhex("-0x1.e3075f86a10fdp-8"),
    float.fromhex("0x1.e8fcbf60ee7cfp-12"),
    float.fromhex("-0x1.70298f665471fp-16"),
)

_COSPI_64 = (
    float.fromhex("0x1p0"),
    float.fromhex("-0x1.3bd3cc9be45dep2"),
    float.fromhex("0x1.03c1f081b5a67p2"),
    float.fromhex("-0x1.55d3c7e3c325cp0"),
    float.fromhex("0x1.e1f5067b90e22p-3"),
    float.fromhex("-0x1.a6d1e7297d362p-6"),
    float.fromhex("0x1.f9c89cce8e272p-10"),
    float.fromhex("-0x1.b1673f09c144cp-14"),
)

_TANPI_64 = (
    float.fromhex("0x1.1a62633145c07p-53"),
    float.fromhex("0x1.4abbce625be5bp3"),
    float.fromhex("0x1.466bc67759dd3p5"),
    float.fromhex("0x1.45fff9b50329dp7"),
    float.fromhex("0x1.45f472c2ac0d5p9"),
    float.fromhex("0x1.45f334b562128p11"),
    float.fromhex("0x1.45f26f68ec321p13"),
    float.fromhex("0x1.45febbc45c794p15"),
    float.fromhex("0x1.4553a4c27db68p17"),
    float.fromhex("0x1.4c1213a3785a3p19"),
    float.fromhex("0x1.1a2e32b762664p21"),
    float.fromhex("0x1.1448d89654eb7p24"),
    float.fromhex("-0x1.fc2b28dda3b8bp25"),
    float.fromhex("0x1.2ec954a74a288p30"),
    float.fromhex("-0x1.74d898db81474p32"),
    float.fromhex("0x1.75c0b233b2e6fp34"),
)

# Polynomial coefficients for asin(sqrt(t)) / (pi * sqrt(t)) on t in [0, 0.25]
# and atan(sqrt(t)) / (pi * sqrt(t)) on t in [0, (sqrt(2) - 1)^2], with 1 / pi
# split into hi + lo so the leading (1 / pi) * x term stays accurate without
# losing precision.
#
# Generated with Sollya 8.0:
#   prec = 350!;
#   display = hexadecimal!;
#   u_lo = 1b-120;
#   /* inv_pi_hi = 0x1.45ep-2 (float32) or 0x1.45f306cp-2 (float64) */
#   inv_pi_lo = round(1/pi - inv_pi_hi, prec_bits, RN);
#   p_asin = fpminimax(asin(sqrt(x)) / (pi * sqrt(x)), [|1, ..., deg|],
#                      [|SG...|], [u_lo; 0.25], floating, relative,
#                      inv_pi_hi + inv_pi_lo);
#   p_atan = fpminimax(atan(sqrt(x)) / (pi * sqrt(x)), [|1, ..., deg|],
#                      [|SG...|], [u_lo; (sqrt(2) - 1)^2], floating, relative,
#                      inv_pi_hi + inv_pi_lo);

_INV_PI_HI_32 = float.fromhex("0x1.45ep-2")

_ASINPI_32 = (
    float.fromhex("0x1.306dcap-14"),
    float.fromhex("0x1.b29954p-5"),
    float.fromhex("0x1.872a08p-6"),
    float.fromhex("0x1.d05ec4p-7"),
    float.fromhex("0x1.4b7716p-7"),
    float.fromhex("0x1.2f3cdcp-8"),
    float.fromhex("0x1.86021p-7"),
)

_ATANPI_32 = (
    float.fromhex("0x1.306dcap-14"),
    float.fromhex("-0x1.b2995p-4"),
    float.fromhex("0x1.04bd8p-4"),
    float.fromhex("-0x1.736f62p-5"),
    float.fromhex("0x1.1460eep-5"),
    float.fromhex("-0x1.3d9714p-6"),
)

_INV_PI_HI_64 = float.fromhex("0x1.45f306cp-2")

_ASINPI_64 = (
    float.fromhex("0x1.c9c882a53f84fp-30"),
    float.fromhex("0x1.b2995e7b7b62fp-5"),
    float.fromhex("0x1.8723a1d5820d6p-6"),
    float.fromhex("0x1.d1a452f799d7bp-7"),
    float.fromhex("0x1.3ce52908fe72dp-7"),
    float.fromhex("0x1.d2b38ad426b3fp-8"),
    float.fromhex("0x1.69f7bb4c0e961p-8"),
    float.fromhex("0x1.23a6ae626c9bap-8"),
    float.fromhex("0x1.db921c18e1584p-9"),
    float.fromhex("0x1.c1d0f6d256841p-9"),
    float.fromhex("0x1.34ef19d92f258p-10"),
    float.fromhex("0x1.c50796d56a1c3p-8"),
    float.fromhex("-0x1.b651fa28c384p-8"),
    float.fromhex("0x1.54686492e52ebp-7"),
)

_ATANPI_64 = (
    float.fromhex("0x1.c9c882a53f84fp-30"),
    float.fromhex("-0x1.b2995e7b7b5e4p-4"),
    float.fromhex("0x1.04c26be3adcddp-4"),
    float.fromhex("-0x1.7483758bfbdap-5"),
    float.fromhex("0x1.21bb93c24ffebp-5"),
    float.fromhex("-0x1.da1b860666b67p-6"),
    float.fromhex("0x1.9127cc4d03287p-6"),
    float.fromhex("-0x1.5b7e411c48d4bp-6"),
    float.fromhex("0x1.30fa02f7cfdcbp-6"),
    float.fromhex("-0x1.06cd4fb2010afp-6"),
    float.fromhex("0x1.8cff0cf68b119p-7"),
    float.fromhex("-0x1.757ff8f8fef22p-8"),
)


def _xor_signbit(y: Array, x: Array) -> Array:
  """Flips the sign bit of `y` wherever `x` has its sign bit set."""
  uint_dtype = np.uint64 if y.dtype == np.float64 else np.uint32
  y_u = bitcast_convert_type(y, uint_dtype)
  x_u = bitcast_convert_type(x, uint_dtype)
  sign_mask = full_like(x_u, 1 << (63 if y.dtype == np.float64 else 31))
  return bitcast_convert_type(y_u ^ (x_u & sign_mask), y.dtype)


def _reduce_half_turns(x: Array) -> tuple[Array, Array, Array]:
  """Reduces `|x|` into quarter-turn index `q` in `{0, 1, 2, 3}` and remainder `r` in `[-0.25, 0.25]`."""
  even_int_threshold = 2.0**53 if x.dtype == np.float64 else 2.0**24

  ax = abs(x)
  is_fin = ax < np.inf

  # Every finite float with |x| >= 2^24 (float32) or 2^53 (float64) is already
  # an even integer. Replace those and non-finite inputs with 0.0, then subtract
  # the nearest multiple of 2 so ax_mod2 is in [-1, 1] with no rounding error.
  ax_clamped = select(is_fin & (ax < even_int_threshold), ax, full_like(x, 0.0))
  two_turns = lax_round(0.5 * ax_clamped, RoundingMethod.TO_NEAREST_EVEN)
  ax_mod2 = ax_clamped - 2.0 * two_turns

  # Round 2 * ax_mod2 to the nearest integer k in {-2, -1, 0, 1, 2}, so
  # ax_mod2 = 0.5 * k + r with r in [-0.25, 0.25]. Subtracting 0.5 * k has no
  # rounding error because 0.5 * k is an exact half-integer within 0.25 of
  # ax_mod2, and for |x| < 0.25 both two_turns and k are 0 so subnormal inputs
  # pass through unchanged.
  k = lax_round(2.0 * ax_mod2, RoundingMethod.TO_NEAREST_EVEN)
  r = ax_mod2 - 0.5 * k
  q = convert_element_type(k, np.int32) & 3
  return r, q, is_fin


def _sinpi_poly(r: Array, u: Array) -> Array:
  pi_hi, coeffs = (
      (_PI_HI_64, _SINPI_64)
      if r.dtype == np.float64
      else (_PI_HI_32, _SINPI_32)
  )
  return pi_hi * r + r * polynomial(u, coeffs, unroll=len(coeffs))


def _cospi_poly(u: Array) -> Array:
  coeffs = _COSPI_64 if u.dtype == np.float64 else _COSPI_32
  return polynomial(u, coeffs, unroll=len(coeffs))


def _tanpi_poly(r: Array, u: Array) -> Array:
  pi_hi, coeffs = (
      (_PI_HI_64, _TANPI_64)
      if r.dtype == np.float64
      else (_PI_HI_32, _TANPI_32)
  )
  return pi_hi * r + r * polynomial(u, coeffs, unroll=len(coeffs))


def _sinpi_impl(x: Array) -> Array:
  r, q, is_fin = _reduce_half_turns(x)
  u = r * r
  s = _sinpi_poly(r, u)
  c = _cospi_poly(u)

  # Quarter-turns on |x| >= 0: 0 -> +s, 1 -> +c, 2 -> -s, 3 -> -c.
  res = select_n(q, s, c, -s, -c)
  # Normalize zeros on |x| >= 0 to +0.0 (at odd integers q == 2, -s(+0.0) is
  # -0.0), then apply the sign of x so sinpi(n) has the same sign as n.
  res = select(res == 0.0, full_like(x, 0.0), res)
  res = _xor_signbit(res, x)
  return select(is_fin, res, full_like(x, np.nan))


def _cospi_impl(x: Array) -> Array:
  r, q, is_fin = _reduce_half_turns(x)
  u = r * r
  s = _sinpi_poly(r, u)
  c = _cospi_poly(u)

  # Quarter-turns on |x| >= 0: 0 -> +c, 1 -> -s, 2 -> -c, 3 -> +s.
  res = select_n(q, c, -s, -c, s)
  # At half-integers with q == 1, -s(+0.0) is -0.0; IEEE 754 requires +0.0 at
  # all half-integers.
  res = select(res == 0.0, full_like(x, 0.0), res)
  return select(is_fin, res, full_like(x, np.nan))


def _tanpi_impl(x: Array) -> Array:
  r, q, is_fin = _reduce_half_turns(x)
  u = r * r
  t = _tanpi_poly(r, u)

  # For odd q (near half-integers), tan(pi * (0.5 * k + r)) = -1 / tan(pi * r).
  # Replace t with 1.0 when q is even before dividing, so small inputs near 0 do
  # not divide by 0 in the unused -1 / t branch.
  odd_q = (q & 1) != 0
  safe_t = select(odd_q, t, full_like(x, 1.0))
  res = select(odd_q, -1.0 / safe_t, t)

  # Exact zeros at integers and signed infinities at half-integers on |x| >= 0:
  #   q == 0 (even |n|):       +0.0
  #   q == 1 (even |n| + 0.5): +inf
  #   q == 2 (odd |n|):        -0.0
  #   q == 3 (odd |n| + 0.5):  -inf
  # Flipping the sign bit when x < 0 preserves tanpi(-x) = -tanpi(x) everywhere.
  zero_or_pole = select_n(
      q,
      full_like(x, 0.0),
      full_like(x, np.inf),
      full_like(x, -0.0),
      full_like(x, -np.inf),
  )
  res = select(r == 0.0, zero_or_pole, res)
  res = _xor_signbit(res, x)
  return select(is_fin, res, full_like(x, np.nan))


def _is_sign_negative(x: Array) -> Array:
  """Returns True wherever `x` is negative, including `-0.0`."""
  int_dtype = np.int64 if x.dtype == np.float64 else np.int32
  return bitcast_convert_type(x, int_dtype) < 0


def _asinpi_acospi_parts(a: Array) -> tuple[Array, Array]:
  """Approximates `asin(a) / pi` or `acos(a) / pi` for `a` in `[0, 1]`.

  Splits `[0, 1]` at `0.5` so a single polynomial on `[0, 0.25]` serves both
  `asinpi` and `acospi` without losing accuracy near `a = 1`:
  - For `a <= 0.5` (inner region), `hi + lo` approximates `asin(a) / pi`
    (evaluated at `t = a^2`). Callers that need full precision down to subnormal
    `a` (`_asinpi_impl`) scale tiny `a` by a power of two before calling this
    helper and scale `hi + lo` back down afterward.
  - For `0.5 < a <= 1` (outer region), `hi + lo` approximates
    `acos(a) / pi = 2 * asin(sqrt((1 - a) / 2)) / pi` (evaluated at
    `t = (1 - a) / 2`, kept unsummed so `_asinpi_impl` can subtract `hi` from
    `0.5` without losing low bits).
  """
  is_f64 = a.dtype == np.float64
  inv_pi_hi = _INV_PI_HI_64 if is_f64 else _INV_PI_HI_32
  coeffs = _ASINPI_64 if is_f64 else _ASINPI_32

  # Clamp out-of-domain and NaN inputs to 0.0 so intermediate products and
  # square roots stay finite (callers replace a > 1 and NaN with NaN).
  a = select(a <= 1.0, a, full_like(a, 0.0))
  outer = a > 0.5

  # For a <= 0.5 (the "inner" region), we approximate asin(a) / pi directly
  # using a polynomial evaluated at t = a^2.
  # For 0.5 < a <= 1.0 (the "outer" region), the slope of asin approaches
  # infinity, making a direct polynomial inaccurate. Instead, we use the trig
  # identity: acos(a) = 2 * asin(sqrt((1 - a) / 2)), and asin(a) = pi/2 - acos(a).
  # So we set t = (1 - a) / 2 and evaluate the same polynomial.
  t_outer = 0.5 * (1.0 - a)
  t = select(outer, t_outer, a * a)
  p = polynomial(t, coeffs, unroll=len(coeffs))

  # The polynomial computes the curve minus the linear term: asin(z)/pi - z/pi.
  # So the full result is w * (1/pi) + w * p (with w = a in the inner region and
  # w = 2 * sqrt(t_outer) in the outer region). To avoid losing precision to
  # rounding when adding these together, we split `w` and `1/pi` into high and
  # low parts. `w_hi * inv_pi_hi` has few enough bits that it multiplies
  # exactly without any rounding error, and the leftover bits are added first.
  #
  # In the outer region, w = u = 2 * sqrt(t_outer). The square root introduces
  # a small rounding error. To recover the lost precision, we split `u` into a
  # high part `u_hi` and compute the remainder using the algebraic identity:
  # u - u_hi = (u^2 - u_hi^2) / (u + u_hi) = (4 * t_outer - u_hi^2) / (u + u_hi),
  # which captures both the low bits of `u` and corrects the square-root
  # rounding error.
  s = sqrt(t_outer)
  u = 2.0 * s
  w = select(outer, u, a)
  w_hi, a_lo = split_half_mask(w)
  denom = select(t_outer > 0.0, u + w_hi, full_like(a, 1.0))
  u_lo = (4.0 * t_outer - w_hi * w_hi) / denom
  w_lo = select(outer, u_lo, a_lo)

  hi = w_hi * inv_pi_hi
  lo = w_lo * inv_pi_hi + w * p
  return hi, lo


def _asinpi_impl(x: Array) -> Array:
  is_f64 = x.dtype == np.float64
  a = abs(x)
  # For extremely small numbers, `w_lo * inv_pi_hi` and `w * p` would underflow
  # and lose the low bits of `1 / pi`. Multiply tiny inputs by a power of two
  # before splitting, then scale the inner sum back down afterward.
  small = a < (2.0**-960 if is_f64 else 2.0**-96)
  up_scale = select(
      small, full_like(x, 2.0**512 if is_f64 else 2.0**64), full_like(x, 1.0)
  )
  down_scale = select(
      small, full_like(x, 2.0**-512 if is_f64 else 2.0**-64), full_like(x, 1.0)
  )
  hi, lo = _asinpi_acospi_parts(a * up_scale)
  # In the outer region, asin(a)/pi = 0.5 - acos(a)/pi. Near |x| = 0.5,
  # `hi` is close to 0.5 and cancels, while near |x| = 1.0, `hi` is much
  # smaller than 0.5 and subtracting it shifts off its low bits. Using
  # fast_two_sum captures any bits rounded off by 0.5 - hi so we can combine
  # them with `lo` before rounding the final sum once.
  s, e = fast_two_sum(full_like(x, 0.5), -hi)
  res = select(a > 0.5, s + (e - lo), (hi + lo) * down_scale)
  # asin is an odd function: asin(-x) = -asin(x).
  res = _xor_signbit(res, x)
  return select(a <= 1.0, res, full_like(x, np.nan))


def _acospi_impl(x: Array) -> Array:
  a = abs(x)
  # For |x| <= 0.5, acospi(x) = 0.5 ± y rounds any |y| < 0.5 * ulp(0.5) to 0.5,
  # so tiny-input rescaling is unnecessary here.
  hi, lo = _asinpi_acospi_parts(a)
  y = hi + lo
  neg_x = x < 0.0
  # acos(x) = pi/2 - asin(x). So for |x| <= 0.5, acospi(x) = 0.5 - asinpi(x).
  # If x is negative, asin(x) is negative, so we add the magnitude of asinpi
  # to 0.5 instead of subtracting it. Because `y` is at most ~0.166, its
  # rounding error gets shrunk down relative to 0.5.
  inner_val = select(neg_x, 0.5 + y, 0.5 - y)
  # For x > 0.5, acospi(x) is computed directly by the outer approximation.
  # For x < -0.5, acos(x) = pi - acos(|x|), so acospi(x) = 1.0 - acospi(|x|).
  # Since `hi` is at most ~0.333, we sum `hi + lo` first before subtracting
  # from 1.0 so we don't round the large 1.0 sum twice.
  outer_val = select(neg_x, 1.0 - y, y)
  res = select(a > 0.5, outer_val, inner_val)
  return select(a <= 1.0, res, full_like(x, np.nan))


def _atanpi_poly(t: Array) -> Array:
  """Approximates `atan(t) / pi` for `|t| <= sqrt(2) - 1`."""
  is_f64 = t.dtype == np.float64
  inv_pi_hi, coeffs = (
      (_INV_PI_HI_64, _ATANPI_64) if is_f64 else (_INV_PI_HI_32, _ATANPI_32)
  )
  # The polynomial computes (atan(t)/pi - t * inv_pi_hi) / t, with `inv_pi_lo`
  # baked into `coeffs[0]`. Splitting `t` via `split_half_mask` makes
  # `t_hi * inv_pi_hi` exact so the leading `t / pi` term does not round in the
  # wider [0.125, 0.25) binade when `|t|` is in (pi/8, sqrt(2) - 1]; scaling
  # tiny `t` by a power of two prevents `t_lo * inv_pi_hi` and `t * p` from
  # underflowing.
  small = abs(t) < (2.0**-960 if is_f64 else 2.0**-96)
  up_scale = select(
      small, full_like(t, 2.0**512 if is_f64 else 2.0**64), full_like(t, 1.0)
  )
  down_scale = select(
      small, full_like(t, 2.0**-512 if is_f64 else 2.0**-64), full_like(t, 1.0)
  )
  t_scaled = t * up_scale
  t_hi, t_lo = split_half_mask(t_scaled)
  p = polynomial(t * t, coeffs, unroll=len(coeffs))
  return (t_hi * inv_pi_hi + (t_lo * inv_pi_hi + t_scaled * p)) * down_scale


def _atanpi_impl(x: Array) -> Array:
  a = abs(x)
  t1 = float(np.sqrt(2.0) - 1.0)
  t2 = float(np.sqrt(2.0) + 1.0)
  r0 = a <= t1
  r2 = a > t2
  is_fin = a < np.inf

  # We map the input `a` into the small range [0, sqrt(2) - 1] so a single
  # short polynomial can approximate it accurately.
  # - For a <= sqrt(2) - 1: t = a, and atan(a)/pi = atan(t)/pi.
  # - For a > sqrt(2) + 1: t = 1/a, and atan(a)/pi = 0.5 - atan(t)/pi.
  # - In between: t = (a - 1) / (a + 1), and atan(a)/pi = 0.25 + atan(t)/pi.
  # Select the numerator and denominator before dividing once, replacing
  # infinities and NaNs with 1.0 so the unused division stays finite.
  a_safe = select(is_fin, a, full_like(x, 1.0))
  num = select(r2, full_like(x, 1.0), a_safe - 1.0)
  den = select(r2, a_safe, a_safe + 1.0)
  t_div = select(is_fin, num / den, full_like(x, 0.0))
  t = select(r0, a, t_div)

  p = _atanpi_poly(t)
  mag = select(r0, p, select(r2, 0.5 - p, 0.25 + p))
  # atan is an odd function: atan(-x) = -atan(x).
  res = _xor_signbit(mag, x)
  # x == x is False only for NaN.
  return select(x == x, res, full_like(x, np.nan))


def _atan2pi_impl(x: Array, y: Array) -> Array:
  is_f64 = x.dtype == np.float64
  hi_thresh = 2.0**512 if is_f64 else 2.0**64
  lo_thresh = 2.0**-512 if is_f64 else 2.0**-64
  hi_scale = 2.0**-512 if is_f64 else 2.0**-64
  lo_scale = 2.0**512 if is_f64 else 2.0**64

  ax = abs(x)
  ay = abs(y)
  s = lax_min(ax, ay)
  l = lax_max(ax, ay)

  # atan2(x, y) depends only on the ratio x / y. If the inputs are extremely
  # large or extremely small, dividing them or adding them could overflow to
  # infinity or round down to zero. We multiply both by a power of two to bring
  # them into a safe range without changing their ratio or causing rounding.
  scale = select(
      l >= hi_thresh,
      full_like(l, hi_scale),
      select(l <= lo_thresh, full_like(l, lo_scale), full_like(l, 1.0)),
  )
  s_scaled = s * scale
  l_scaled = l * scale

  # Replace zeros and infinities with safe dummy values to avoid 0/0 or inf/inf.
  finite_nz = (l > 0.0) & (l < np.inf)
  l_safe = select(finite_nz, l_scaled, full_like(l, 1.0))
  s_safe = select(finite_nz, s_scaled, full_like(l, 0.0))

  # Since s <= l, the ratio s / l is in [0, 1]. We map it into [0, sqrt(2) - 1]
  # just like in atanpi. If s > (sqrt(2) - 1) * l, we use the identity
  # atan(s/l) = pi/4 + atan((s - l) / (s + l)). Computing (s - l) directly
  # (exact for s >= 0.5 * l) avoids the precision loss near s == l of dividing
  # s/l and then subtracting 1.
  t1 = float(np.sqrt(2.0) - 1.0)
  small = s_safe <= t1 * l_safe
  num = select(small, s_safe, s_safe - l_safe)
  den = select(small, l_safe, s_safe + l_safe)
  t = num / den
  p = _atanpi_poly(t)

  # Map `p` back to the full [0, 1] upper half-plane by picking the base angle
  # in {0, 0.25, 0.5, 0.75, 1} and the sign of `p` first, then adding once.
  # Doing a single addition avoids extra rounding from chaining `0.25 + p`,
  # `0.5 - u`, and `1.0 - u`. We check the sign bit of `y` rather than `y < 0`
  # so `atan2pi(±0.0, -0.0)` returns `±1.0`.
  swap = ax > ay
  neg_y = _is_sign_negative(y)
  offset_small = select(
      swap,
      full_like(l, 0.5),
      select(neg_y, full_like(l, 1.0), full_like(l, 0.0)),
  )
  offset_large = select(neg_y, full_like(l, 0.75), full_like(l, 0.25))
  offset = select(small, offset_small, offset_large)
  signed_p = select(neg_y ^ swap, -p, p)
  u = offset + signed_p

  # IEEE 754 requires exact specific angles when inputs are infinite.
  inf = full_like(l, np.inf)
  x_inf = ax == inf
  y_inf = ay == inf
  both_inf = x_inf & y_inf
  only_x_inf = x_inf & (ay < inf)
  only_y_inf = (ax < inf) & y_inf
  u = select(both_inf, select(neg_y, full_like(l, 0.75), full_like(l, 0.25)), u)
  u = select(only_x_inf, full_like(l, 0.5), u)
  u = select(only_y_inf, select(neg_y, full_like(l, 1.0), full_like(l, 0.0)), u)

  # Finally, mirror the angle into the negative half-plane if x is negative.
  res = _xor_signbit(u, x)
  not_nan = (x == x) & (y == y)
  return select(not_nan, res, full_like(l, np.nan))


def _trig_unary_lowering(impl, ctx, x):
  aval = ctx.avals_in[0]

  def _lower(v):
    if aval.dtype not in (np.float32, np.float64):
      return convert_element_type(
          impl(convert_element_type(v, np.float32)), aval.dtype
      )
    return impl(v)

  return mlir.lower_fun(_lower, multiple_results=False)(ctx, x)


def _trig_binary_lowering(impl, ctx, x, y):
  aval_in = ctx.avals_in[0]
  aval_out = ctx.avals_out[0]

  def _lower(lhs, rhs):
    lhs = _maybe_broadcast(aval_out.shape, lhs, aval_out.sharding)
    rhs = _maybe_broadcast(aval_out.shape, rhs, aval_out.sharding)
    if aval_in.dtype not in (np.float32, np.float64):
      return convert_element_type(
          impl(
              convert_element_type(lhs, np.float32),
              convert_element_type(rhs, np.float32),
          ),
          aval_in.dtype,
      )
    return impl(lhs, rhs)

  return mlir.lower_fun(_lower, multiple_results=False)(ctx, x, y)


@export
def sinpi(x: ArrayLike) -> Array:
  r"""Elementwise sine of :math:`\pi x`: :math:`\mathrm{sin}(\pi x)`.

  Implements the IEEE 754-2019 ``sinPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{sin}(\pi x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.sin`: elementwise sine in radians.
    - :func:`jax.lax.cospi`: elementwise cosine of :math:`\pi x`.
    - :func:`jax.lax.tanpi`: elementwise tangent of :math:`\pi x`.
  """
  return sinpi_p.bind(x)


@export
def cospi(x: ArrayLike) -> Array:
  r"""Elementwise cosine of :math:`\pi x`: :math:`\mathrm{cos}(\pi x)`.

  Implements the IEEE 754-2019 ``cosPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{cos}(\pi x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.cos`: elementwise cosine in radians.
    - :func:`jax.lax.sinpi`: elementwise sine of :math:`\pi x`.
    - :func:`jax.lax.tanpi`: elementwise tangent of :math:`\pi x`.
  """
  return cospi_p.bind(x)


@export
def tanpi(x: ArrayLike) -> Array:
  r"""Elementwise tangent of :math:`\pi x`: :math:`\mathrm{tan}(\pi x)`.

  Implements the IEEE 754-2019 ``tanPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{tan}(\pi x)`.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.tan`: elementwise tangent in radians.
    - :func:`jax.lax.sinpi`: elementwise sine of :math:`\pi x`.
    - :func:`jax.lax.cospi`: elementwise cosine of :math:`\pi x`.
    - :func:`jax.lax.atanpi`: elementwise arc tangent divided by :math:`\pi`.
  """
  return tanpi_p.bind(x)


@export
def asinpi(x: ArrayLike) -> Array:
  r"""Elementwise arc sine divided by :math:`\pi`: :math:`\mathrm{asin}(x) / \pi`.

  Implements the IEEE 754-2019 ``asinPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{asin}(x) / \pi` in ``[-0.5, 0.5]``.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.asin`: elementwise arc sine in radians.
    - :func:`jax.lax.acospi`: elementwise arc cosine divided by :math:`\pi`.
    - :func:`jax.lax.atanpi`: elementwise arc tangent divided by :math:`\pi`.
    - :func:`jax.lax.sinpi`: elementwise sine of :math:`\pi x`.
  """
  return asinpi_p.bind(x)


@export
def acospi(x: ArrayLike) -> Array:
  r"""Elementwise arc cosine divided by :math:`\pi`: :math:`\mathrm{acos}(x) / \pi`.

  Implements the IEEE 754-2019 ``acosPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{acos}(x) / \pi` in ``[0, 1]``.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.acos`: elementwise arc cosine in radians.
    - :func:`jax.lax.asinpi`: elementwise arc sine divided by :math:`\pi`.
    - :func:`jax.lax.atanpi`: elementwise arc tangent divided by :math:`\pi`.
    - :func:`jax.lax.cospi`: elementwise cosine of :math:`\pi x`.
  """
  return acospi_p.bind(x)


@export
def atanpi(x: ArrayLike) -> Array:
  r"""Elementwise arc tangent divided by :math:`\pi`: :math:`\mathrm{atan}(x) / \pi`.

  Implements the IEEE 754-2019 ``atanPi`` operation on real floating-point
  inputs.

  Args:
    x: input array. Must have a real floating-point dtype.

  Returns:
    Array of the same shape and dtype as ``x`` containing the element-wise
    values of :math:`\mathrm{atan}(x) / \pi` in ``[-0.5, 0.5]``.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.atan`: elementwise arc tangent in radians.
    - :func:`jax.lax.atan2pi`: elementwise two-term arc tangent divided by :math:`\pi`.
    - :func:`jax.lax.asinpi`: elementwise arc sine divided by :math:`\pi`.
    - :func:`jax.lax.acospi`: elementwise arc cosine divided by :math:`\pi`.
    - :func:`jax.lax.tanpi`: elementwise tangent of :math:`\pi x`.
  """
  return atanpi_p.bind(x)


@export
def atan2pi(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise two-term arc tangent divided by :math:`\pi`: :math:`\mathrm{atan}(x / y) / \pi`.

  Implements the IEEE 754-2019 ``atan2Pi`` operation on real floating-point
  inputs.

  Args:
    x, y: input arrays. Must have matching real floating-point dtypes. If
      neither is a scalar, the two arrays must have the same number of
      dimensions and be broadcast-compatible.

  Returns:
    Array of the same shape and dtype as ``x`` and ``y`` containing the
    element-wise values of :math:`\mathrm{atan2}(x, y) / \pi` in ``[-1, 1]``,
    respecting the quadrant indicated by the sign of each input.

  Numerical Precision:
    For accuracy bounds, see :ref:`numerical-accuracy`.

  See also:
    - :func:`jax.lax.atan2`: elementwise two-term arc tangent in radians.
    - :func:`jax.lax.atanpi`: elementwise one-term arc tangent divided by :math:`\pi`.
  """
  x, y = core.auto_insert_reshard(x, y)
  return atan2pi_p.bind(x, y)


sinpi_p = standard_unop(_float, "sinpi")
ad.defjvp(sinpi_p, lambda g, x: g * (np.pi * cospi(x)))
mlir.register_lowering(
    sinpi_p, partial(_trig_unary_lowering, _sinpi_impl), inline=False
)

cospi_p = standard_unop(_float, "cospi")
ad.defjvp(cospi_p, lambda g, x: -g * (np.pi * sinpi(x)))
mlir.register_lowering(
    cospi_p, partial(_trig_unary_lowering, _cospi_impl), inline=False
)

tanpi_p = standard_unop(_float, "tanpi")
ad.defjvp2(tanpi_p, lambda g, ans, x: g * (np.pi * (1.0 + ans * ans)))
mlir.register_lowering(
    tanpi_p, partial(_trig_unary_lowering, _tanpi_impl), inline=False
)

asinpi_p = standard_unop(_float, "asinpi")
ad.defjvp(
    asinpi_p,
    lambda g, x: g * (float(1.0 / np.pi) * rsqrt(one_minus_square(x))),
)
mlir.register_lowering(
    asinpi_p, partial(_trig_unary_lowering, _asinpi_impl), inline=False
)

acospi_p = standard_unop(_float, "acospi")
ad.defjvp(
    acospi_p,
    lambda g, x: -g * (float(1.0 / np.pi) * rsqrt(one_minus_square(x))),
)
mlir.register_lowering(
    acospi_p, partial(_trig_unary_lowering, _acospi_impl), inline=False
)

atanpi_p = standard_unop(_float, "atanpi")


def _atanpi_jvp(g, x):
  r, _, u2 = hypot(x, full_like(x, 1.0))
  return ((u2 * g) * float(1.0 / np.pi)) / r


ad.defjvp(atanpi_p, _atanpi_jvp)
mlir.register_lowering(
    atanpi_p, partial(_trig_unary_lowering, _atanpi_impl), inline=False
)

atan2pi_p = standard_naryop([_float, _float], "atan2pi")


def _atan2pi_jvp(primals, tangents):
  x, y = primals
  dx, dy = tangents
  out = atan2pi(x, y)
  r, u1, u2 = hypot(x, y)
  num = ad.add_tangents(
      dx if type(dx) is ad_util.Zero else u2 * dx,
      dy if type(dy) is ad_util.Zero else -(u1 * dy),
  )
  return (
      out,
      ad_util.p2tz(out)
      if type(num) is ad_util.Zero
      else (num * float(1.0 / np.pi)) / r,
  )


ad.primitive_jvps[atan2pi_p] = _atan2pi_jvp
mlir.register_lowering(
    atan2pi_p, partial(_trig_binary_lowering, _atan2pi_impl), inline=False
)
