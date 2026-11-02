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

"""Float32 and float64 implementations of digamma psi(x) = d/dx ln|Gamma(x)|.

Algorithm overview
------------------
Both precisions follow the same structure, combining three standard identities
to evaluate psi(x) within 1 ULP across all inputs:

1. Asymptotic expansion for z >= z_min (4.5 in float32, 12.5 in float64):
     psi(z) = ln(z) - 1 / (2z) - sum_{k >= 1} B_{2k} / (2k * z^{2k}),
   where B_{2k} are the Bernoulli numbers (B_2 = 1/6, B_4 = -1/30, ...).
   - For float32 (z >= 4.5, 1 / z^2 <= 0.0494), an 8-term polynomial in
     1 / z^2 reaches ~51-bit accuracy.
   - For float64 (z >= 12.5, 1 / z^2 <= 0.0064), a 15-term polynomial in
     1 / z^2 reaches ~113-bit accuracy.

2. Recurrence shift for 0.5 <= z < z_min:
     psi(z) = psi(z + 4k) - sum_{j = 0}^{4k - 1} 1 / (z + j),
   with at most one +4 shift in float32 and at most three +4 shifts in float64,
   moving z into [z_min, inf). Centering each 4-term block around m = c + 1.5
   with u = m^2 - 1.25 >= 2.75 collapses four reciprocals into one cancellation-
   free division:
     1/c + 1/(c+1) + 1/(c+2) + 1/(c+3) = 4 * m * u / (u^2 - 1).

3. Euler's reflection formula for x < 0.5:
     psi(x) = psi(1 - x) - pi * cot(pi * x).
   Writing x = n + d with integer n = round(x) and d = x - n in [-0.5, 0.5],
   periodicity of cot gives cot(pi * x) = cot(pi * d), while y = 1 - x =
   (1 - n) - d >= 0.5 lands on the positive half-line.

   Evaluating -pi * cot(pi * d) via sin/cos loses accuracy near d = 0 (where
   cot has a pole) and near |d| = 0.5 (where cot vanishes and is closest to its
   neighboring poles at d = +-1). Subtracting the poles at d = 0, +-1, and +-2
   from the partial-fraction expansion of pi * cot(pi * d) gives
     -pi * cot(pi * d) = -1/d + d * ((10 - 4*d^2) / ((d^2 - 5)*d^2 + 4) + R(d^2)),
   leaving a smooth remainder R(d^2) whose nearest poles are at d = +-3, so a
   polynomial in d^2 on [0, 0.25] converges rapidly (~53 bits at 8 terms for
   float32; ~112 bits at 16 terms for float64).

Precision and cancellation near zeros
-------------------------------------
psi(x) has a unique positive zero at r_+ ~= 1.461632145 and one zero inside
each negative interval (-k - 1, -k) for integer k >= 0. Subtracting nearly
equal terms near a zero cancels k leading bits and loses k bits of relative
precision, so intermediates need more than p + k bits of precision (p = 24 for
float32, p = 53 for float64) to keep the rounded result within 1 ULP:

- Float32 on CPU (`use_f64_for_f32=True`): intermediate calculations run in
  native float64 (53-bit significand, 29 extra bits over float32). That leaves
  at least 24 accurate bits everywhere except at 3 negative float32 inputs where
  up to 29.2 bits cancel, which are handled by table lookup.
- Float32 on GPU and TPU (`use_f64_for_f32=False`) and float64 on all backends:
  intermediate values are represented as unevaluated sums `hi + lo` of two
  floating-point numbers (double-word arithmetic: ~48 bits in float32, ~106 bits
  in float64), using operations that require neither hardware fused multiply-add
  (FMA) nor subnormal numbers (flushed to zero on GPU and TPU). That extra
  precision keeps the error within 1 ULP everywhere except:
  * in a narrow window around r_+ ~= 1.461632145 (`2e-5` in float32, `1e-4` in
    float64), where psi(x) is evaluated directly by a short Taylor polynomial
    around the floating-point value nearest r_+; and
  * at the negative inputs where cancellation near a zero of psi(x) leaves too
    few accurate bits in `(hi, lo)` precision: 3 float32 inputs (up to 29.2
    canceled bits, found by exhaustive float32 search) and 10 float64 inputs on
    (-2^52, 0) (up to 55.75 canceled bits, found by root search over
    (-2^52, 0)), which return precomputed correctly rounded values.

Special values (with subnormal inputs flushed to zero):
- `|x| < finfo.tiny` (`2^-126` in float32, `2^-1022` in float64):
  `-copysign(inf, x)` (-inf for +0, +inf for -0).
- Negative integers (`x < 0` and `x == floor(x)`, including `-inf`): `NaN`.
- `+inf`: `+inf`; `NaN`: `NaN`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from jax._src import cloud_tpu_init
from jax._src import config
from jax._src.api import jit
from jax._src.lax import lax
from jax._src.lax.double_word import (
    HybridPoly,
    f2_add,
    f2_div,
    f2_mul,
    f2_recip,
    f2_square,
    fast_two_sum,
    log_f2,
    poly_hybrid,
    two_prod,
    two_sum,
)
import numpy as np


class _DigammaConfig(NamedTuple):
  """Precision-specific parameters for the shared `(hi, lo)` digamma pipeline."""

  dtype: Any
  int_dtype: Any
  tiny: float
  all_int_below: float
  z_min: float
  n_shifts: int
  z_big: float
  recip_scale: float
  stir_poly: HybridPoly
  cot_poly: HybridPoly
  rpos_hi: float
  rpos_tol: float
  overrides: tuple[tuple[float, float], ...]


# ---------------------------------------------------------------------------
# Polynomial generation & precision-specific constants
# ---------------------------------------------------------------------------
#
# All Chebyshev polynomials below were generated by Chebyshev-node interpolation
# in `mpmath`, splitting dominant low-degree terms into descending `(hi, lo)`
# pairs and leaving high-degree tail terms in ascending plain floating-point
# form (where powers of `w` or `u` suppress single-word rounding error well
# below `(hi, lo)` working precision):
#
#   import mpmath, numpy as np
#   def cheb_poly(fn, a, b, n, dps=140):
#     with mpmath.workdps(dps):
#       a, b = mpmath.mpf(a), mpmath.mpf(b)
#       nodes = [(a + b)/2 + (b - a)/2 * mpmath.cos(mpmath.pi * (2*j + 1) / (2*n))
#                for j in range(n)]
#       m = mpmath.matrix([[x**k for k in range(n)] for x in nodes])
#       c = mpmath.lu_solve(m, mpmath.matrix([fn(x) for x in nodes]))
#       return [c[k] for k in range(n)]  # ascending powers x^0 .. x^{n-1}
#
#   def split_desc(coeffs, dtype, dps=140):
#     with mpmath.workdps(dps):
#       hi = tuple(float(dtype(float(c))) for c in reversed(coeffs))
#       lo = tuple(float(dtype(float(c - mpmath.mpf(h))))
#                  for c, h in zip(reversed(coeffs), hi))
#       return hi, lo

# --- Float32 constants ---

# Taylor expansion psi(x) ~= psi(_RPOS_HI) + psi'(_RPOS_HI) * t + 0.5 * psi'' * t^2
# around _RPOS_HI (the float32 nearest r_+ = 1.4616321449683623) for |t| < 2e-5.
# _PSI1_HI16 has 16 mantissa bits and t has at most 8 nonzero mantissa bits, so
# `t * _PSI1_HI16` (8 + 16 = 24 bits) multiplies with zero rounding error.
_RPOS_HI_F32 = 1.4616321325302124
_PSI1_HI16 = 0.9676666259765625
_PSI1_LO16 = 5.619471266982146e-06
_PSI_OFFSET = -1.203605215494008e-08
_HALF_PSI2 = -0.44276317954063416

# Float32 asymptotic tail `-(psi(z) - ln(z) + 1/(2z)) * z^2` in `u = 1 / z^2`
# on `z >= 4.5` (`u in [0, 1/20.25]`, 8 terms: 4 `(hi, lo)` + 4 float32).
_STIR_POLY_F32 = HybridPoly(
    lo_deg_hi=(
        -0.004166500177234411,
        0.0039682528004050255,
        -0.008333333767950535,
        0.0833333358168602,
    ),
    lo_deg_lo=(
        1.80291295781565e-10,
        1.6523876711360685e-10,
        4.369439809970288e-10,
        -2.483527827834564e-09,
    ),
    hi_deg=(
        0.0075619034469127655,
        -0.02044805884361267,
        0.06567689776420593,
        -0.1522468477487564,
    ),
)

# Float32 reflection remainder `R(w) = (-pi*cot(pi*d) + 1/d - 2d/(1-w) - 2d/(4-w)) / d`
# in `w = d^2` on `[0, 0.25]` (8 terms: 4 `(hi, lo)` + 4 float32).
_COT_POLY_F32 = HybridPoly(
    lo_deg_hi=(
        0.00034221241367049515,
        0.003436123952269554,
        0.03964646905660629,
        0.7898681163787842,
    ),
    lo_deg_lo=(
        -1.4072926851627088e-11,
        1.6512574294025306e-11,
        -1.6343285613729108e-09,
        1.7317669076533093e-08,
    ),
    hi_deg=(
        3.602519791456871e-05,
        3.892349468515022e-06,
        4.2365297758806264e-07,
        5.249261292306073e-08,
    ),
)

# Across all 2^32 float32 bit patterns, only these three negative inputs exceed
# 1 ULP under ~24.8-29.2 bits of cancellation in `psi(1 - x) - pi * cot(pi * d)`:
_OVERRIDES_F32 = (
    (-2.6107208728790283, -4.811107601199183e-08),  # 0xc027160d
    (-66.79590606689453, -6.681547759512796e-09),  # 0xc2859781
    (-95.80821990966797, 5.399198954592066e-08),  # 0xc2bf9dcf
)

# Ascending float64 asymptotic polynomial on `[1e-35, 1/20.25]` (`z >= 4.5`) for
# the CPU float32 path (`use_f64_for_f32=True`).
_STIR_CPU_F64 = (
    0.08333333333333333,
    -0.008333333333331052,
    0.00396825396638545,
    -0.004166666064322241,
    0.007575656825057211,
    -0.021082807398581014,
    0.08270110356222089,
    -0.4165436317406144,
    2.2764606013219155,
    -10.32528257297648,
    25.54399691440876,
)

# --- Float64 constants ---

# Degree-4 Taylor expansion around `_RPOS_HI_F64` (the float64 nearest
# r_+ = 1.46163214496836234126...) for `|x - _RPOS_HI_F64| < 1e-4`.
_RPOS_HI_F64 = 1.4616321449683622
_PSI0_F64 = (-9.241265521729427e-17, -2.5907356508198256e-33)
_PSI1_F64 = (0.9676722454476212, 5.0688981776619267e-17)
_PSI2_F64 = -0.4427631689835922
_PSI3_F64 = 0.2584997609556511
_PSI4_F64 = -0.16394270544240658

# Float64 asymptotic tail `-(psi(z) - ln(z) + 1/(2z)) * z^2` in `u = 1 / z^2`
# on `z >= 12.5` (`u in [0, 1/12.5^2]`, 15 terms: 6 `(hi, lo)` + 9 float64).
_STIR_POLY_F64 = HybridPoly(
    lo_deg_hi=(
        -0.02109279609279136,
        0.0075757575757575725,
        -0.004166666666666667,
        0.003968253968253968,
        -0.008333333333333333,
        0.08333333333333333,
    ),
    lo_deg_lo=(
        6.6846026287507115e-19,
        1.7779357872760644e-20,
        -5.648576985608088e-20,
        2.2028205636971673e-19,
        -1.1564823170698995e-19,
        4.625929269271485e-18,
    ),
    hi_deg=(
        0.08333333332881014,
        -0.44325980095164014,
        3.053952947844877,
        -26.455747897251968,
        281.34693751040305,
        -3587.5352658291686,
        52314.90878434773,
        -755855.2580681203,
        7355272.314586399,
    ),
)

# Float64 reflection remainder `R(w)` in `w = d^2` on `[0, 0.25]`
# (16 terms: 10 `(hi, lo)` + 6 float64).
_COT_POLY_F64 = HybridPoly(
    lo_deg_hi=(
        5.754349334325642e-10,
        5.19199874964571e-09,
        4.694069230374593e-08,
        4.259576174096584e-07,
        3.8918566160965975e-06,
        3.6025255636170674e-05,
        0.00034221239588867877,
        0.0034361239688982794,
        0.039646467422276385,
        0.7898681336964529,
    ),
    lo_deg_lo=(
        4.4934487685539166e-26,
        5.9592513086368146e-27,
        1.6102102731493155e-24,
        -1.2003802057223964e-23,
        -1.798675399939698e-22,
        4.7500665878142945e-22,
        -1.1175797492901849e-20,
        -1.5592281451156182e-20,
        -2.174273797593867e-18,
        -5.020885545454614e-17,
    ),
    hi_deg=(
        6.384724984320336e-11,
        7.088565158962753e-12,
        7.872169220727059e-13,
        8.759464478499956e-14,
        9.459278579589876e-15,
        1.3512023505691749e-15,
    ),
)

# First 12 ascending float64 terms of `_COT_POLY_F64` (54-bit accuracy on
# `[0, 0.25]`), reused on the CPU float32 path (`use_f64_for_f32=True`).
_COT_CPU_F64 = (*reversed(_COT_POLY_F64.lo_deg_hi), *_COT_POLY_F64.hi_deg[:2])

# Negative float64 inputs on (-2^52, 0) where `psi(1 - x) - pi * cot(pi * d)`
# cancels more than 51.8 bits near a zero of psi(x) (up to 55.75 bits). Found by
# bracketing each zero on (-2^24, 0) and solving `psi(1 + N - d) = pi * cot(pi * d)`
# for integer `N` at each representable fractional offset `d` on (-2^52, -2^24):
_OVERRIDES_F64 = (
    (-0.5040830082644554, 7.289763902976895e-17),  # 0xbfe02172b05ee260
    (-1.5734984731623904, 1.5649788481838454e-16),  # 0xbff92d0cbc289d4a
    (-5.6671624415568855, 4.1867794464524804e-17),  # 0xc016ab2ca18e6ce3
    (-14.727334416018529, 6.305749002764493e-16),  # 0xc02d74652d341c14
    (-87.80536027839507, -3.819390762310924e-16),  # 0xc055f38b05d64d13
    (-2977.8808777877402, 1.3170578140239663e-16),  # 0xc0a743c30269d43c
    (-6066919.936815942, -1.089898921099485e-15),  # 0xc15724b9fbf4cad9
    (-204097275.94819948, 5.82590269634041e-16),  # 0xc1a8548df7e57a67
    (-216428973.94835505, 3.682396316799126e-15),  # 0xc1a9cce35be58ecb
    (-71243556817555.97, -3.810910311192602e-15),  # 0xc2d032ebb962a4fe
)

_F32_CFG = _DigammaConfig(
    dtype=np.float32,
    int_dtype=np.int32,
    tiny=1.1754943508222875e-38,  # 2^-126
    all_int_below=-8388608.0,  # -2^23
    z_min=4.5,
    n_shifts=1,
    z_big=1e18,
    recip_scale=16777216.0,  # 2^24
    stir_poly=_STIR_POLY_F32,
    cot_poly=_COT_POLY_F32,
    rpos_hi=_RPOS_HI_F32,
    rpos_tol=2e-5,
    overrides=_OVERRIDES_F32,
)

_F64_CFG = _DigammaConfig(
    dtype=np.float64,
    int_dtype=np.int64,
    tiny=2.2250738585072014e-308,  # 2^-1022
    all_int_below=-4503599627370496.0,  # -2^52
    z_min=12.5,
    n_shifts=3,
    z_big=1e80,
    recip_scale=18014398509481984.0,  # 2^54
    stir_poly=_STIR_POLY_F64,
    cot_poly=_COT_POLY_F64,
    rpos_hi=_RPOS_HI_F64,
    rpos_tol=1e-4,
    overrides=_OVERRIDES_F64,
)


@jit
def _shift4_f2(c0, c1):
  """Computes `1/c + 1/(c+1) + 1/(c+2) + 1/(c+3)` for `c = c0 + c1` as `(hi, lo)`.

  Centering around `m = c + 1.5` with `u = m^2 - 1.25 >= 2.75` gives
  `4 * m * u / (u^2 - 1)`, replacing four reciprocals with one division.
  """
  s, e = two_sum(c0, lax._const(c0, 1.5))
  m0, m1 = fast_two_sum(s, e + c1)
  msq0, msq1 = f2_square(m0, m1)
  u0, u1 = fast_two_sum(msq0, lax._const(c0, -1.25))
  u1 = u1 + msq1
  usq0, usq1 = f2_square(u0, u1)
  den0, den1 = fast_two_sum(usq0, lax._const(c0, -1.0))
  den1 = den1 + usq1
  num0, num1 = f2_mul(4.0 * m0, 4.0 * m1, u0, u1)
  return f2_div(num0, num1, den0, den1)


def _classify_and_reflect(x, cfg: _DigammaConfig, use_optimization_barrier: bool):
  """Classifies special regions and splits `x in (all_int_below, 0.5)` into `n + delta`."""
  one = lax.full_like(x, 1.0)
  inf = lax.full_like(x, np.inf)
  half_neg = lax.full_like(x, -0.5)
  # Work around an older XLA:TPU constant-folding bug in libtpu < 0.0.51.
  # TODO(phawkins): Remove once minimum libtpu version is at least 0.0.51.
  if use_optimization_barrier and not cloud_tpu_init.is_libtpu_at_least(
      "0.0.51"
  ):
    one = lax.optimization_barrier(one)

  # Substitute finite placeholders (`1.0` on the positive path, `-0.5` on the
  # reflection path) so inactive lanes never divide by zero or overflow
  # intermediate products before the final special-value `select`s.
  ax = lax.abs(x)
  is_zero_or_sub = ax < lax._const(ax, cfg.tiny)
  is_pos = x >= 0.5
  gt_int_lim = x > lax._const(x, cfg.all_int_below)
  x_safe = lax.select(is_zero_or_sub | ~(ax < inf), one, x)
  x_neg_safe = lax.select(
      (~is_pos) & gt_int_lim & (~is_zero_or_sub), x_safe, half_neg
  )

  # Split inputs x in (all_int_below, 0.5) as x = n + delta with integer
  # n = round(x) <= 0 and delta in [-0.5, 0.5]. Substitute `-0.5` on non-positive
  # integer poles (`delta == 0`) so both `1 / delta` and `(1 - w)(4 - w)` stay
  # away from zero on inactive lanes.
  n = lax.round(x_neg_safe, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = x_neg_safe - n
  is_delta_zero = delta == 0.0
  delta_nz = lax.select(is_delta_zero, half_neg, delta)
  return (
      one,
      inf,
      half_neg,
      is_zero_or_sub,
      is_pos,
      gt_int_lim,
      is_delta_zero,
      x_safe,
      n,
      delta,
      delta_nz,
  )


def _restore_special_outputs(
    x,
    res,
    inf,
    cfg: _DigammaConfig,
    is_pos,
    is_delta_zero,
    gt_int_lim,
    is_zero_or_sub,
):
  """Applies near-root overrides and restores special outputs for NaN, poles, +-0, and +inf."""
  for x_i, y_i in cfg.overrides:
    res = lax.select(
        lax.eq(x, lax._const(x, x_i)), lax.full_like(x, y_i), res
    )
  is_nan_out = (x != x) | ((~is_pos) & (is_delta_zero | (~gt_int_lim)))
  res = lax.select(is_nan_out, lax.full_like(x, np.nan), res)
  bits = lax.bitcast_convert_type(x, cfg.int_dtype)
  zero_inf = lax.select(bits < 0, inf, -inf)
  res = lax.select(is_zero_or_sub, zero_inf, res)
  return lax.select(x == inf, inf, res)


def _digamma_cpu_f32(x, *, use_optimization_barrier: bool = False):
  """Computes float32 digamma(x) to <= 1 ULP on CPU using native float64."""
  x = lax.convert_element_type(x, np.float32)
  (
      _,
      inf,
      _,
      is_zero_or_sub,
      is_pos,
      gt_int_lim,
      is_delta_zero,
      x_safe,
      n,
      _,
      delta_nz,
  ) = _classify_and_reflect(x, _F32_CFG, use_optimization_barrier)

  xd = lax.convert_element_type(x_safe, np.float64)
  dd = lax.convert_element_type(delta_nz, np.float64)
  z = lax.select(
      is_pos, xd, (1.0 - lax.convert_element_type(n, np.float64)) - dd
  )

  # Shift z < 4.5 to z + 4 >= 4.5 using symmetric pairing around m = z + 1.5.
  need_shift = z < 4.5
  zs = lax.select(need_shift, z, lax.full_like(z, 1.0))
  m = zs + 1.5
  q = m * m
  sh = (4.0 * m * (q - 1.25)) / ((q - 2.5) * q + 0.5625)
  acc = lax.select(need_shift, -sh, lax.full_like(z, 0.0))
  z = lax.select(need_shift, z + 4.0, z)

  iz = lax.reciprocal(z)
  u = iz * iz
  st = lax.polynomial(u, _STIR_CPU_F64, unroll=len(_STIR_CPU_F64))
  psi_z = acc + lax.log(z) - 0.5 * iz - u * st

  w = dd * dd
  ct = lax.polynomial(w, _COT_CPU_F64, unroll=len(_COT_CPU_F64))
  cot = -lax.reciprocal(dd) + dd * (
      (10.0 - 4.0 * w) / ((w - 5.0) * w + 4.0) + ct
  )
  res = lax.convert_element_type(
      lax.select(is_pos, psi_z, psi_z + cot), np.float32
  )
  return _restore_special_outputs(
      x,
      res,
      inf,
      _F32_CFG,
      is_pos,
      is_delta_zero,
      gt_int_lim,
      is_zero_or_sub,
  )


def _digamma_f2(
    x, cfg: _DigammaConfig, *, use_optimization_barrier: bool = False
):
  """Computes float32 or float64 digamma(x) to <= 1 ULP using `(hi, lo)` pairs."""
  x = lax.convert_element_type(x, cfg.dtype)
  zero = lax.full_like(x, 0.0)
  four = lax.full_like(x, 4.0)
  (
      one,
      inf,
      half_neg,
      is_zero_or_sub,
      is_pos,
      gt_int_lim,
      is_delta_zero,
      x_safe,
      n,
      delta,
      delta_nz,
  ) = _classify_and_reflect(x, cfg, use_optimization_barrier)

  # Since n <= 0 on all lanes and |n| <= -all_int_below, `1 - n >= 1 > 0.5 >=
  # |delta|` is an exact integer in `cfg.dtype`, so `fast_two_sum(1 - n, -delta)`
  # forms `y = 1 - x` without rounding loss.
  yh, yl = fast_two_sum(one - n, -delta)
  z0 = lax.select(is_pos, x_safe, yh)
  z1 = lax.select(is_pos, zero, yl)

  # Shift z < z_min into [z_min, inf) in `cfg.n_shifts` steps of +4 (`1` for
  # float32, `3` for float64), clamping inactive lanes to `1.0` before `_shift4_f2`.
  acc0, acc1 = zero, zero
  z_min = lax._const(x, cfg.z_min)
  for _ in range(cfg.n_shifts):
    mask = (z0 < z_min) | ((z0 == z_min) & (z1 < 0.0))
    sc0 = lax.select(mask, z0, one)
    sc1 = lax.select(mask, z1, zero)
    h0, h1 = _shift4_f2(sc0, sc1)
    na0, na1 = f2_add(acc0, acc1, -h0, -h1)
    nz0, nz1 = f2_add(sc0, sc1, four, zero)
    acc0 = lax.select(mask, na0, acc0)
    acc1 = lax.select(mask, na1, acc1)
    z0 = lax.select(mask, nz0, z0)
    z1 = lax.select(mask, nz1, z1)

  # Subtract `-1 / (2z)`. For `z0 >= cfg.z_big`, `1 / (2z)` is >65 bits (float32)
  # or >270 bits (float64) below `ln(z)`, so clamping avoids subnormal `1 / z`
  # at `FLT_MAX` and `split26_f64` overflow at `DBL_MAX`.
  z_big = z0 >= lax._const(x, cfg.z_big)
  zc0 = lax.select(z_big, one, z0)
  zc1 = lax.select(z_big, zero, z1)
  iz0, iz1 = f2_recip(zc0, zc1)
  iz0 = lax.select(z_big, zero, iz0)
  iz1 = lax.select(z_big, zero, iz1)
  acc0, acc1 = f2_add(acc0, acc1, half_neg * iz0, half_neg * iz1)

  # Add shared `(hi, lo)` natural logarithm `ln(z0 + z1)`.
  lg0, lg1 = log_f2(z0, z1, iz=(iz0, iz1))
  acc0, acc1 = f2_add(acc0, acc1, lg0, lg1)

  # Subtract the asymptotic Bernoulli series `(1 / z^2) * P_stir(1 / z^2)`.
  u0, u1 = f2_square(iz0, iz1)
  sp0, sp1 = poly_hybrid(u0, u1, cfg.stir_poly)
  sp0, sp1 = f2_mul(sp0, sp1, -u0, -u1)
  acc0, acc1 = f2_add(acc0, acc1, sp0, sp1)

  # Euler reflection for x < 0.5: scale `delta_nz` (`tiny <= |delta_nz| <= 0.5`)
  # by `cfg.recip_scale` (`2^24` or `2^54`) before `f2_recip` so intermediate
  # products never overflow near `+-tiny`, then rescale by `-cfg.recip_scale`.
  scale = lax.full_like(x, cfg.recip_scale)
  ids0, ids1 = f2_recip(delta_nz * scale, zero)
  neg_scale = -scale
  id0 = lax.select(is_pos, zero, neg_scale * ids0)
  id1 = lax.select(is_pos, zero, neg_scale * ids1)
  acc0, acc1 = f2_add(acc0, acc1, id0, id1)

  # Smooth reflection terms `delta * ((10 - 4*w) / ((w - 5)*w + 4) + R(w))`
  # with `w = delta^2 in [0, 0.25]`, zeroed on `x >= 0.5` by `delta_refl`.
  dw0, dw1 = two_prod(delta_nz, delta_nz)
  rn0, rn1 = f2_add(lax._const(x, 10.0), zero, -4.0 * dw0, -4.0 * dw1)
  wm5_0, wm5_1 = f2_add(lax._const(x, -5.0), zero, dw0, dw1)
  rd0, rd1 = f2_mul(wm5_0, wm5_1, dw0, dw1)
  rd0, rd1 = f2_add(four, zero, rd0, rd1)
  rat0, rat1 = f2_div(rn0, rn1, rd0, rd1)

  ct0, ct1 = poly_hybrid(dw0, dw1, cfg.cot_poly)
  cp0, cp1 = f2_add(ct0, ct1, rat0, rat1)
  delta_refl = lax.select(is_pos, zero, delta_nz)
  cp0, cp1 = f2_mul(cp0, cp1, delta_refl, zero)
  s_fin, e_fin = two_sum(acc0, cp0)
  res = s_fin + (e_fin + (acc1 + cp1))

  # Direct Taylor expansion around the positive zero r_+ ~= 1.461632145 inside
  # `|x - rpos_hi| < rpos_tol`, avoiding cancellation in `acc + cp`.
  rpos_hi = lax._const(x, cfg.rpos_hi)
  in_rpos = lax.abs(x_safe - rpos_hi) < lax._const(x, cfg.rpos_tol)
  t = lax.select(in_rpos, x_safe - rpos_hi, zero)
  if cfg.dtype == np.float32:
    res_pos = t * lax._const(t, _PSI1_HI16) + (
        lax._const(t, _PSI_OFFSET)
        + t * (lax._const(t, _PSI1_LO16) + lax._const(t, _HALF_PSI2) * t)
    )
  else:
    p0, p1 = two_prod(t, lax._const(t, _PSI1_F64[0]))
    s0, s1 = two_sum(lax._const(t, _PSI0_F64[0]), p0)
    tail = lax._const(t, _PSI0_F64[1]) + t * (
        lax._const(t, _PSI1_F64[1])
        + t
        * (
            lax._const(t, _PSI2_F64)
            + t * (lax._const(t, _PSI3_F64) + t * lax._const(t, _PSI4_F64))
        )
    )
    res_pos = s0 + (s1 + (p1 + tail))
  res = lax.select(in_rpos, res_pos, res)

  return _restore_special_outputs(
      x,
      res,
      inf,
      cfg,
      is_pos,
      is_delta_zero,
      gt_int_lim,
      is_zero_or_sub,
  )


@config.explicit_x64_dtypes("allow")
@jit(static_argnames=["use_f64_for_f32", "use_optimization_barrier"])
def digamma_impl(
    x, *, use_f64_for_f32: bool = False, use_optimization_barrier: bool = False
):
  """Computes float32 or float64 digamma(x) to <= 1 ULP."""
  if x.dtype == np.float64:
    return _digamma_f2(
        x, _F64_CFG, use_optimization_barrier=use_optimization_barrier
    )
  if use_f64_for_f32:
    return _digamma_cpu_f32(
        x, use_optimization_barrier=use_optimization_barrier
    )
  return _digamma_f2(
      x, _F32_CFG, use_optimization_barrier=use_optimization_barrier
  )
