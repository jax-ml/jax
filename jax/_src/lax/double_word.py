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

"""Double-word `(hi, lo)` floating-point arithmetic and elementary kernels.

All operations represent extended-precision values as unevaluated sums `hi + lo`
in either `float32` (~48 significand bits) or `float64` (~106 significand bits)
without requiring hardware fused multiply-add (FMA) or subnormal arithmetic
(flushed to zero on GPU and TPU).
"""

from __future__ import annotations

from typing import NamedTuple

from jax._src.api import jit
from jax._src.lax import lax
import numpy as np


class HybridPoly(NamedTuple):
  """Hybrid polynomial coefficients: descending `(hi, lo)` low degrees + ascending plain high degrees."""

  lo_deg_hi: tuple[float, ...]
  lo_deg_lo: tuple[float, ...]
  hi_deg: tuple[float, ...]


# Four-interval exact 4-bit dyadic reciprocals `r_i in {1.25, 1.0625, 0.9375, 0.75}`
# map `m in [sqrt(2)/2, sqrt(2))` to `m * r_i in [0.88388, 1.11488]`, shrinking
# `s = (m * r_i - 1) / (m * r_i + 1)` to `w = s^2 <= 0.00380` (~8x smaller than
# unreduced `w <= 0.02944`), so `P_log(w) = (ln((1+s)/(1-s)) / s - 2) / w`
# needs only 1 `(hi, lo)` constant in float32 (zero `f2_mul` Horner steps) and
# 4 `(hi, lo)` constants in float64.
_NEG_LNR4_F32 = (
    (-0.2231435477733612, -3.540848503291727e-09),
    (-0.0606246218085289, -7.905942393804377e-12),
    (0.06453852355480194, -2.417230859919073e-09),
    (0.28768208622932434, -1.3777543550475002e-08),
)
_NEG_LNR4_F64 = (
    (-0.22314355131420976, 9.091270597324799e-18),
    (-0.06062462181643484, -2.6424025938726934e-18),
    (0.06453852113757118, -6.470486661692933e-18),
    (0.2876820724517809, 2.607160616442564e-17),
)

# Float32: 5 terms after peeling 2.0 (1 `(hi, lo)` + 4 float32, ~54-bit rel accuracy).
_LOG_POLY_F32 = HybridPoly(
    lo_deg_hi=(0.6666666865348816,),
    lo_deg_lo=(-1.9868215517249155e-08,),
    hi_deg=(
        0.4000000059604645,
        0.2857142984867096,
        0.22221730649471283,
        0.1832914650440216,
    ),
)

# Float64: 10 terms after peeling 2.0 (5 `(hi, lo)` + 5 float64, ~108-bit accuracy).
_LOG_POLY_F64 = HybridPoly(
    lo_deg_hi=(
        0.18181818181818155,
        0.2222222222222222,
        0.2857142857142857,
        0.4,
        0.6666666666666666,
    ),
    lo_deg_lo=(
        -1.0849770252736922e-17,
        1.2501947479184836e-17,
        1.5860279463478475e-17,
        -2.2204460486792272e-17,
        3.700743415417178e-17,
    ),
    hi_deg=(
        0.15384615384643138,
        0.13333333316787233,
        0.1176471198512322,
        0.10524956497710565,
        0.09691143935120411,
    ),
)

# `(exp(r) - 1 - r) / r^2` on `[-ln(2)/2, ln(2)/2]` for double-word `exp`.
_EXP_POLY_F32 = HybridPoly(
    lo_deg_hi=(0.0416666679084301, 0.1666666716337204, 0.5),
    lo_deg_lo=(
        -1.1221643614334198e-09,
        -4.967094735519595e-09,
        -4.4893255779498986e-13,
    ),
    hi_deg=(
        0.008333333767950535,
        0.0013888839166611433,
        0.00019841225002892315,
        2.486786979716271e-05,
        2.761756377367419e-06,
    ),
)
_EXP_POLY_F64 = HybridPoly(
    lo_deg_hi=(
        0.041666666666666664,
        0.16666666666666666,
        0.5,
    ),
    lo_deg_lo=(
        2.3112453133981343e-18,
        9.251858662483088e-18,
        2.1072866792986803e-24,
    ),
    hi_deg=(
        0.008333333333333333,
        0.0013888888888888892,
        0.00019841269841269844,
        2.4801587301575863e-05,
        2.7557319223979163e-06,
        2.7557319251195953e-07,
        2.5052108401445554e-08,
        2.0876723766810705e-09,
        1.6059024297298586e-10,
        1.1490855116406605e-11,
        7.658991821701572e-13,
    ),
)

# `ln(2)` split into `(hi, mid, lo)` where `hi` and `mid` have at most 15
# significand bits in float32 (since `|k| <= 256`, 9 + 15 = 24 bits) and 41
# significand bits in float64 (since `|k| <= 2048`, 12 + 41 = 53 bits), so
# `k * hi` and `k * mid` multiply with zero rounding error across all normal
# and subnormal exponent offsets.
LN2_SPLIT_F32 = (
    0.693145751953125,
    1.4285906217992306e-06,
    1.619850954759361e-11,
)
LN2_SPLIT_F64 = (
    0.6931471805598903,
    5.49792301870721e-14,
    1.1612227229362532e-26,
)

_SQRT2_OVER_2_BITS_F32 = 0x3F3504F3
_SQRT2_OVER_2_BITS_F64 = 0x3FE6A09E667F3BCD


def two_sum(a, b):
  """Computes `(s, e)` with `s + e == a + b` without rounding loss."""
  s = a + b
  v = s - a
  return s, (a - (s - v)) + (b - v)


def fast_two_sum(a, b):
  """Computes `(s, e)` with `s + e == a + b` when `|a| >= |b|` (or `a == 0`)."""
  s = a + b
  return s, b - (s - a)


def split26_f64(a):
  """Splits float64 `a` (`|a| < 2^996`) into two 26-bit parts `ah + al == a`."""
  c = lax._const(a, 134217729.0) * a
  ah = c - (c - a)
  return ah, a - ah


def split_bits(a, keep_bits: int):
  """Splits normal `a` via bitmask truncation so `ah` keeps `keep_bits` significand bits.

  For finite normal `a` (and zero), `ah + al == a` and `ah` has at most
  `keep_bits` significand bits. Subnormal inputs retain fewer than `keep_bits`
  significand bits in `ah` (down to `0.0` when all set bits fall in the masked
  low bits).
  """
  if a.dtype == np.float64:
    int_ty, nmant = np.int64, 52
  elif a.dtype == np.float32:
    int_ty, nmant = np.int32, 23
  else:
    raise ValueError(
        f"split_bits requires float32 or float64 input, got {a.dtype}"
    )
  if not 1 <= keep_bits <= nmant + 1:
    raise ValueError(
        f"keep_bits must be in [1, {nmant + 1}] for {a.dtype}, got {keep_bits}"
    )
  mask = -(1 << (nmant + 1 - keep_bits))
  ai = lax.bitcast_convert_type(a, int_ty)
  ah = lax.bitcast_convert_type(ai & lax._const(ai, mask), a.dtype)
  return ah, a - ah


def split_half(a):
  """Splits float32 (12+12 bits) or float64 (26+27 bits) `a` into `ah + al == a`."""
  if a.dtype == np.float32:
    # Float32 has 24 significand bits (even), so masking the low 12 bits splits
    # into 12 + 12 bits where all four partial products fit in 24 bits.
    return split_bits(a, 12)
  # Float64 has 53 significand bits (odd), so bitmask truncation gives 26 + 27
  # bits where `al * bl` (54 bits) would round. Multiplying by `2^27 + 1` and
  # subtracting `c - a` rounds `ah` to 26 bits so `|al| <= 2^-27 * |ah|` also
  # fits in 26 bits (26 + 26 = 52 <= 53).
  return split26_f64(a)


def split_half_mask(a):
  """Splits `a` via bitmask truncation, safe up to `FLT_MAX` / `DBL_MAX`."""
  return split_bits(a, 12 if a.dtype == np.float32 else 26)


@jit
def two_prod(a, b):
  """Computes `(p, e)` with `p + e == a * b` (assuming no overflow) without FMA."""
  p = a * b
  ah, al = split_half(a)
  bh, bl = split_half(b)
  return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


@jit
def two_prod_big(a, b):
  """Same as `two_prod`, safe up to `FLT_MAX` / `DBL_MAX` via bitmask splitting."""
  p = a * b
  ah, al = split_half_mask(a)
  bh, bl = split_half_mask(b)
  return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


@jit
def f2_square(a0, a1):
  """Squares `(a0 + a1)^2` as a `(hi, lo)` pair with a single half-width split."""
  p = a0 * a0
  ah, al = split_half(a0)
  ah_al = ah * al
  e = ((ah * ah - p) + (ah_al + ah_al)) + al * al
  a0_a1 = a0 * a1
  return fast_two_sum(p, e + (a0_a1 + a0_a1))


@jit
def f2_add(a0, a1, b0, b1):
  """Adds two `(hi, lo)` pairs `(a0 + a1) + (b0 + b1)`."""
  s, e = two_sum(a0, b0)
  return fast_two_sum(s, e + (a1 + b1))


@jit
def f2_add_ordered(a0, a1, b0, b1):
  """Adds `(a0 + a1) + (b0 + b1)` when `|a0| >= |b0|` (or `a0 == 0`)."""
  s, e = fast_two_sum(a0, b0)
  return fast_two_sum(s, e + (a1 + b1))


def f2_sum(a0, a1, b0, b1):
  """Adds two `(hi, lo)` pairs and rounds the sum to a single word."""
  s, e = two_sum(a0, b0)
  return s + (e + (a1 + b1))


@jit
def f2_mul(a0, a1, b0, b1):
  """Multiplies two `(hi, lo)` pairs `(a0 + a1) * (b0 + b1)` without FMA."""
  p, e = two_prod(a0, b0)
  return fast_two_sum(p, e + (a0 * b1 + a1 * b0))


def f2_mul_exact_high(a0, a1, b0, b1):
  """Multiplies `(a0, a1)` and `(b0, b1)` when `a0 * b0` is exact."""
  ph = a0 * b0
  pl = a0 * b1 + a1 * b0 + a1 * b1
  return fast_two_sum(ph, pl)


def recip_est(b0):
  """Computes a full-precision single-word reciprocal estimate `1 / b0`."""
  r = lax.reciprocal(b0)
  if b0.dtype == np.float32:
    # Refine hardware float32 reciprocal (~14-23 bits on accelerators) to full
    # 24-bit accuracy before computing the low residual term.
    two = lax.full_like(b0, 2.0)
    r = r * (two - b0 * r)
    r = r * (two - b0 * r)
  return r


@jit
def f2_recip(a0, a1):
  """Computes `1 / (a0 + a1)` as a `(hi, lo)` pair without FMA."""
  r = recip_est(a0)
  p0, e0 = two_prod(a0, r)
  t = r * (((lax._const(a0, 1.0) - p0) - e0) - a1 * r)
  return fast_two_sum(r, t)


@jit
def f2_div(a0, a1, b0, b1):
  """Divides two `(hi, lo)` pairs `(a0 + a1) / (b0 + b1)` without FMA."""
  r = recip_est(b0)
  q0 = a0 * r
  p0, e0 = two_prod(q0, b0)
  q1 = (((a0 - p0) - e0) + (a1 - q0 * b1)) * r
  return fast_two_sum(q0, q1)


def poly_hybrid(x0, x1, poly: HybridPoly):
  """Evaluates `poly` via plain Horner on high degrees and `@jit` `(hi, lo)` on low degrees."""
  pt = lax.polynomial(x0, poly.hi_deg, unroll=len(poly.hi_deg))
  p0, p1 = two_prod(pt, x0)
  p0, p1 = f2_add_ordered(
      lax._const(x0, poly.lo_deg_hi[0]),
      lax._const(x0, poly.lo_deg_lo[0]),
      p0,
      p1,
  )
  for hi_c, lo_c in zip(poly.lo_deg_hi[1:], poly.lo_deg_lo[1:]):
    m0, m1 = f2_mul(p0, p1, x0, x1)
    p0, p1 = f2_add_ordered(
        lax._const(x0, hi_c), lax._const(x0, lo_c), m0, m1
    )
  return p0, p1


@jit
def log_f2(z0, z1=None, iz=None, k_adj=None):
  """Computes `ln(z0 + z1) + k_adj * ln(2)` for normal `z0 > 0` as `(hi, lo)`.

  Args:
    z0: High part of the positive normal argument.
    z1: Optional low residual part (`|z1| <= ulp(z0)`).
    iz: Optional precomputed `(iz0, iz1) = 1 / (z0 + z1)` pair to avoid a
      duplicate reciprocal when computing the `z1 / z0` correction.
    k_adj: Optional integer power-of-two exponent adjustment added to `k` before
      multiplying by `ln(2)`.
  """
  dtype = z0.dtype
  zero = lax.full_like(z0, 0.0)
  one = lax.full_like(z0, 1.0)
  two = lax.full_like(z0, 2.0)
  if dtype == np.float32:
    int_dtype = np.int32
    mant_bits = 23
    sqrt2_bits = _SQRT2_OVER_2_BITS_F32
    ln2_hi, ln2_mid, ln2_lo = LN2_SPLIT_F32
    neg_lnr = _NEG_LNR4_F32
    log_poly = _LOG_POLY_F32
  else:
    int_dtype = np.int64
    mant_bits = 52
    sqrt2_bits = _SQRT2_OVER_2_BITS_F64
    ln2_hi, ln2_mid, ln2_lo = LN2_SPLIT_F64
    neg_lnr = _NEG_LNR4_F64
    log_poly = _LOG_POLY_F64

  # Write z0 = 2^k * m with m in [sqrt(2)/2, sqrt(2)), then reduce m into
  # [0.88388, 1.11488] using one of four exact 4-bit dyadic factors r_i so
  # `mh * r_i` and `ml * r_i` incur zero rounding error and `w = s^2 <= 0.00380`.
  z_bits = lax.bitcast_convert_type(z0, int_dtype)
  s_mant = lax.full_like(z_bits, mant_bits)
  exp_off = lax.shift_right_arithmetic(
      z_bits - lax._const(z_bits, sqrt2_bits), s_mant
  )
  k = lax.convert_element_type(exp_off, dtype)
  if k_adj is not None:
    k = k + k_adj
  m = lax.bitcast_convert_type(
      z_bits - lax.shift_left(exp_off, s_mant), dtype
  )
  mh, ml = split_half(m)

  b1 = m >= one
  b0 = m >= lax.select(
      b1,
      lax.full_like(m, 1.189207115002721),
      lax.full_like(m, 0.8408964152537145),
  )
  r = lax.select(
      b1,
      lax.select(b0, lax.full_like(m, 0.75), lax.full_like(m, 0.9375)),
      lax.select(b0, lax.full_like(m, 1.0625), lax.full_like(m, 1.25)),
  )
  lr_h = lax.select(
      b1,
      lax.select(
          b0, lax.full_like(m, neg_lnr[3][0]), lax.full_like(m, neg_lnr[2][0])
      ),
      lax.select(
          b0, lax.full_like(m, neg_lnr[1][0]), lax.full_like(m, neg_lnr[0][0])
      ),
  )
  lr_l = lax.select(
      b1,
      lax.select(
          b0, lax.full_like(m, neg_lnr[3][1]), lax.full_like(m, neg_lnr[2][1])
      ),
      lax.select(
          b0, lax.full_like(m, neg_lnr[1][1]), lax.full_like(m, neg_lnr[0][1])
      ),
  )
  fh, fl = two_sum(mh * r - one, ml * r)

  kln_h = k * lax._const(k, ln2_hi)
  kln_m = k * lax._const(k, ln2_mid)
  kln_s, kln_e = fast_two_sum(kln_h, kln_m)
  kln_e = kln_e + k * lax._const(k, ln2_lo)
  kln_s, kln_e = f2_add(kln_s, kln_e, lr_h, lr_l)

  d0, d1 = fast_two_sum(two, fh)
  d1 = d1 + fl
  s0, s1 = f2_div(fh, fl, d0, d1)
  w0, w1 = f2_square(s0, s1)
  lp0, lp1 = poly_hybrid(w0, w1, log_poly)
  lp0, lp1 = f2_mul(lp0, lp1, w0, w1)
  lp0, lp1 = f2_add_ordered(two, zero, lp0, lp1)
  lp0, lp1 = f2_mul(lp0, lp1, s0, s1)
  if z1 is not None:
    if iz is not None:
      zc_corr0, zc_corr1 = f2_mul(z1, zero, iz[0], iz[1])
    else:
      zc_corr0, zc_corr1 = f2_div(z1, zero, z0, zero)
    lp0, lp1 = f2_add_ordered(lp0, lp1, zc_corr0, zc_corr1)
  return f2_add(kln_s, kln_e, lp0, lp1)


@jit
def exp_f2_mant_exp(x0, x1):
  """Computes `(eh, el, kf)` with `exp(x0 + x1) = 2**kf * (eh + el)` (`|x0| <= 1418`)."""
  dtype = x0.dtype
  zero = lax.full_like(x0, 0.0)
  one = lax.full_like(x0, 1.0)
  if dtype == np.float32:
    ln2_hi, ln2_mid, ln2_lo = LN2_SPLIT_F32
    exp_poly = _EXP_POLY_F32
  else:
    ln2_hi, ln2_mid, ln2_lo = LN2_SPLIT_F64
    exp_poly = _EXP_POLY_F64
  kf = lax.round(
      x0 * lax._const(x0, 1.4426950408889634),
      lax.RoundingMethod.TO_NEAREST_EVEN,
  )
  r0, e0 = two_sum(x0, -kf * lax._const(x0, ln2_hi))
  s1, e1 = two_sum(e0, -kf * lax._const(x0, ln2_mid))
  r0, t0 = fast_two_sum(r0, s1)
  r1 = t0 + e1 + x1 - kf * lax._const(x0, ln2_lo)
  p0, p1 = poly_hybrid(r0, r1, exp_poly)
  r2_0, r2_1 = f2_square(r0, r1)
  q0, q1 = f2_mul(r2_0, r2_1, p0, p1)
  o0, o1 = f2_add(one, zero, r0, r1)
  eh, el = f2_add(o0, o1, q0, q1)
  return eh, el, kf


@jit
def scale_f2_pow2(m0, m1, k_int):
  """Scales `(m0 + m1) * 2**k_int` to a single float across normal and overflow exponents."""
  dtype = m0.dtype
  if dtype == np.float32:
    mant_bits, bias, emin, emax = 23, 127, -126, 127
  else:
    mant_bits, bias, emin, emax = 52, 1023, -1022, 1023
  s_mant = lax.full_like(k_int, mant_bits)
  bias_v = lax.full_like(k_int, bias)
  k_clamped = lax.clamp(
      lax.full_like(k_int, 2 * emin), k_int, lax.full_like(k_int, 2 * emax)
  )
  k1 = lax.clamp(
      lax.full_like(k_int, emin), k_clamped, lax.full_like(k_int, emax)
  )
  k2 = k_clamped - k1
  pow1 = lax.bitcast_convert_type(lax.shift_left(k1 + bias_v, s_mant), dtype)
  pow2 = lax.bitcast_convert_type(lax.shift_left(k2 + bias_v, s_mant), dtype)
  return ((m0 + m1) * pow2) * pow1
