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

"""High-accuracy real gamma implementation.

Intermediate values are represented as `(hi, lo)` pairs of floats ("double-word"
arithmetic from `jax._src.lax.double_word`), giving roughly twice the working
precision of the input type (~48 bits in float32, ~106 bits in float64) without
needing hardware fused multiply-add or subnormal support. That extra precision
is needed for two reasons:
- In `exp(ln(Gamma))`, a 1-ULP error in `ln(Gamma)` produces `|ln(Gamma)|` ULPs
  of error in `Gamma`—up to ~88 ULPs in float32 and ~710 ULPs in float64 near
  the overflow limit.
- Subtracting `x` from `1` or reducing arguments modulo `ln(2)` can cancel
  leading bits when two numbers are close, so the remaining bits of the
  difference come from the lower bits kept in `lo`.
Carrying ~24 extra bits in float32 (~53 in float64) leaves enough accurate bits
at the end to round the final result within 1 ULP.

Strategy for `gamma_real(x)`:
- For `x >= 0.5`, shift `x` up by `N` (`4` in float32, `8` in float64) so
  `z_s = x + N >= 4.5` (or `8.5`), where Stirling's asymptotic series for
  `ln(Gamma(z_s))` converges rapidly:
    `ln(Gamma(z_s)) ~= (z_s - 0.5) * (ln(z_s) - 1) + (0.5 * ln(2*pi) - 0.5)`
    `                  + (1 / z_s) * S(1 / z_s^2)`.
  Exponentiate that sum into a `(hi, lo)` mantissa plus an integer power of two
  (`exp_f2_mant_exp`), divide by the shift product
  `P = x * (x + 1) * ... * (x + N - 1)`, and apply the power-of-two scale at the
  very end (`scale_f2_pow2`). Keeping the power-of-two exponent separate avoids
  premature overflow when `Gamma(x + N)` overflows float32/float64 even though
  `Gamma(x)` itself is still finite (for example near `x = 35` in float32 or
  `x = 171` in float64).
- For `x < 0.5`, apply Euler's reflection formula
  `Gamma(x) = pi / (sin(pi * x) * Gamma(1 - x))`. Splitting `x = n + delta` with
  integer `n = round(x)` and `delta in [-0.5, 0.5]` gives
  `sin(pi * x) = (-1)^n * pi * delta * sinc(delta)`, canceling `pi` analytically
  so `Gamma(x) = (-1)^n / (delta * sinc(delta) * Gamma(1 - x))`. Both `x >= 0.5`
  and `x < 0.5` share the same shifted Stirling evaluation (on `x` or
  `y = 1 - x`) and the same final division and power-of-two scaling.
"""

from __future__ import annotations

from jax._src import config
from jax._src import custom_derivatives
from jax._src.api import jit
from jax._src.lax import lax
from jax._src.lax import special as lax_special
from jax._src.lax.double_word import (
    HybridPoly,
    exp_f2_mant_exp,
    f2_add,
    f2_div,
    f2_mul,
    f2_recip,
    f2_square,
    fast_two_sum,
    log_f2,
    poly_hybrid,
    scale_f2_pow2,
    two_prod,
)
from jax._src.typing import Array
import numpy as np

# Constant term `0.5 * ln(2 * pi) - 0.5` in Stirling's expansion when the
# leading term is written in factored form `(z_s - 0.5) * (ln(z_s) - 1)`.
_HLN2PI_MHALF_F32 = (0.418938547372818, -1.4168144879533884e-08)
_HLN2PI_MHALF_F64 = (0.4189385332046727, 1.6728209650585413e-17)

# Polynomial approximations for the Stirling series remainder `S(u)` in
# `u = 1 / z_s^2` on `[0, 1 / 4.5^2]` (float32) and `[0, 1 / 8.5^2]` (float64).
# Low-degree coefficients are stored as `(hi, lo)` pairs (`lo_deg_*`) and
# smaller higher-degree coefficients as single floats (`hi_deg`).
_STIR_REAL_F32 = HybridPoly(
    lo_deg_hi=(0.0833333358168602,),
    lo_deg_lo=(-2.4835524747857107e-09,),
    hi_deg=(
        -0.0027777778450399637,
        0.0007936418405734003,
        -0.0005944418953731656,
        0.000808869837783277,
        -0.001249697059392929,
    ),
)
_STIR_REAL_F64 = HybridPoly(
    lo_deg_hi=(0.0007936507936507933, -0.002777777777777778, 0.08333333333333333),
    lo_deg_lo=(
        3.316689196962006e-20,
        1.0609560759990016e-19,
        4.625929265202287e-18,
    ),
    hi_deg=(
        -0.0005952380952376979,
        0.0008417508414713099,
        -0.0019175268010075124,
        0.006410225439100511,
        -0.029545187123491443,
        0.17899168633582968,
        -1.3396758536832984,
        10.561873979935012,
        -57.9286343353279,
    ),
)

# Polynomial approximations for `(sinc(delta) - 1) / w` in `w = delta^2` on
# `[0, 0.25]`, where `sinc(delta) = sin(pi * delta) / (pi * delta)`.
_SINC_DIRECT_F32 = HybridPoly(
    lo_deg_hi=(0.8117424249649048, -1.644934058189392),
    lo_deg_lo=(8.535201018178284e-11, -8.6580254077262e-09),
    hi_deg=(
        -0.1907518059015274,
        0.026147661730647087,
        -0.002344642300158739,
        0.00014328490942716599,
    ),
)
_SINC_DIRECT_F64 = HybridPoly(
    lo_deg_hi=(
        0.0261478478176548,
        -0.19075182412208422,
        0.8117424252833536,
        -1.6449340668482264,
    ),
    lo_deg_lo=(
        6.310877563167436e-19,
        4.4195868159601715e-18,
        3.5613840315191636e-17,
        -3.0406723503979357e-17,
    ),
    hi_deg=(
        -0.0023460810354558235,
        0.00014842879303107092,
        -6.9758736616552575e-06,
        2.531217404030389e-07,
        -7.304711754636685e-09,
        1.7165356452629683e-10,
        -3.347381652129664e-12,
        5.3924959290663136e-14,
    ),
)


def _nan_where(cond: Array, valid: Array) -> Array:
  """Returns NaN where `cond` is True without producing NaN when `cond` is False."""
  inf_if_nan = lax.select(
      cond, lax.full_like(valid, np.inf), lax.full_like(valid, 0.0)
  )
  return lax.select(cond, inf_if_nan - inf_if_nan, valid)


@custom_derivatives.custom_jvp
def gamma_real(x: Array) -> Array:
  return _gamma_real_impl(x)


@config.explicit_x64_dtypes("allow")
@jit
def _gamma_real_impl(x: Array) -> Array:
  dtype = x.dtype
  zero = lax.full_like(x, 0.0)
  one = lax.full_like(x, 1.0)
  two = lax.full_like(x, 2.0)
  half = lax.full_like(x, 0.5)
  neg_half = lax.full_like(x, -0.5)
  neg_one = lax.full_like(x, -1.0)
  if dtype == np.float32:
    int_dtype = np.int32
    mant_bits, bias = 23, 127
    abs_mask, inf_bits = 0x7FFFFFFF, 0x7F800000
    min_norm_bits, mant_mask, one_bits = 0x00800000, 0x007FFFFF, 0x3F800000
    # Outside [x_under, x_over], Gamma(x) underflows to +/-0 or overflows to +inf.
    x_under = lax.full_like(x, -42.5)
    x_over = lax.full_like(x, 36.0)
    n_shifts = 4
    c_h = _HLN2PI_MHALF_F32
    stir_poly = _STIR_REAL_F32
    sinc_poly = _SINC_DIRECT_F32
  else:
    int_dtype = np.int64
    mant_bits, bias = 52, 1023
    abs_mask, inf_bits = 0x7FFFFFFFFFFFFFFF, 0x7FF0000000000000
    min_norm_bits = 0x0010000000000000
    mant_mask = 0x000FFFFFFFFFFFFF
    one_bits = 0x3FF0000000000000
    x_under = lax.full_like(x, -190.5)
    x_over = lax.full_like(x, 173.0)
    n_shifts = 8
    c_h = _HLN2PI_MHALF_F64
    stir_poly = _STIR_REAL_F64
    sinc_poly = _SINC_DIRECT_F64

  # Classify +/-0, subnormals (flushed to signed zero -> +/-inf, matching
  # `lax.lgamma` and `lax.digamma`), +/-inf, and NaN via raw IEEE bits.
  xi = lax.bitcast_convert_type(x, int_dtype)
  zero_i = lax.full_like(xi, 0)
  inf_bits_v = lax.full_like(xi, inf_bits)
  abs_bits = xi & lax.full_like(xi, abs_mask)
  sign_neg = lax.lt(xi, zero_i)
  is_zero = lax.lt(abs_bits, lax.full_like(xi, min_norm_bits))
  is_inf = lax.eq(abs_bits, inf_bits_v)
  is_nan_in = lax.gt(abs_bits, inf_bits_v)
  is_finite_in = lax.lt(abs_bits, inf_bits_v)

  # Clamp inputs into the finite output range and split x = n + delta for the
  # x < 0.5 reflection branch, with integer n = round(x) and delta in [-0.5, 0.5].
  is_pos = (~sign_neg) & (~is_zero) & lax.ge(x, half)
  x_clamped = lax.select(
      is_nan_in | is_zero, zero, lax.clamp(x_under, x, x_over)
  )
  x_neg = lax.select(is_pos, neg_half, x_clamped)
  n = lax.round(x_neg, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = x_neg - n
  ad = lax.abs(delta)

  x_fin = lax.select(is_finite_in & (~is_zero), x, zero)
  x_floor = lax.floor(x_fin)
  is_int = is_finite_in & (is_zero | lax.eq(x, x_floor))
  is_zero_or_neg_int = (~is_pos) & is_int

  # Split normal `|delta| = ad_mant * 2^e_ad` with `ad_mant in [1, 2)` so
  # `P / (|delta| * ...)` cannot overflow near `|x| ~ finfo.tiny` before `2^-kf`
  # is applied.
  ad_bits = lax.bitcast_convert_type(ad, int_dtype)
  raw_exp = lax.shift_right_logical(ad_bits, lax.full_like(xi, mant_bits))
  ad_mant = lax.bitcast_convert_type(
      lax.full_like(xi, one_bits) | (ad_bits & lax.full_like(xi, mant_mask)),
      dtype,
  )
  e_ad = raw_exp - lax.full_like(xi, bias)
  d_factor = lax.select(is_pos | is_zero_or_neg_int, one, ad_mant)
  e_ad_eff = lax.select(is_pos | is_zero_or_neg_int, zero_i, e_ad)

  # Evaluate the shifted argument `z = x` (for `x >= 0.5`) or `z = 1 - x =
  # (1 - n) - delta` (for `x < 0.5`) as a `(hi, lo)` pair `(z0, z1)`, then form
  # the recurrence product `P = z * (z + 1) * ... * (z + N - 1)` in a tree.
  yh, yl = fast_two_sum(one - n, -delta)
  z0 = lax.select(is_pos, x_clamped, yh)
  z1 = lax.select(is_pos, zero, yl)

  terms = [
      f2_add(z0, z1, lax._const(x, float(k)), zero) if k > 0 else (z0, z1)
      for k in range(n_shifts)
  ]
  while len(terms) > 1:
    terms = [
        f2_mul(*terms[2 * i], *terms[2 * i + 1]) for i in range(len(terms) // 2)
    ]
  p0, p1 = terms[0]

  # Evaluate Stirling's series for `ln(Gamma(z_s))` at `z_s = z + N >= N + 0.5`,
  # and exponentiate into `(eh + el) * 2^kf` with mantissa `eh + el` near `[0.7, 1.42]`.
  zs0, zs1 = f2_add(z0, z1, lax._const(x, float(n_shifts)), zero)
  iz0, iz1 = f2_recip(zs0, zs1)
  lnz0, lnz1 = log_f2(zs0, zs1, iz=(iz0, iz1))
  lnm1_0, lnm1_1 = f2_add(lnz0, lnz1, neg_one, zero)
  zmh0, zmh1 = f2_add(zs0, zs1, neg_half, zero)
  acc0, acc1 = f2_mul(zmh0, zmh1, lnm1_0, lnm1_1)
  acc0, acc1 = f2_add(acc0, acc1, lax._const(x, c_h[0]), lax._const(x, c_h[1]))
  u0, u1 = f2_square(iz0, iz1)
  sp0, sp1 = poly_hybrid(u0, u1, stir_poly)
  sp0, sp1 = f2_mul(sp0, sp1, iz0, iz1)
  acc0, acc1 = f2_add(acc0, acc1, sp0, sp1)

  eh, el, kf = exp_f2_mant_exp(acc0, acc1)
  k_int = lax.convert_element_type(kf, int_dtype)

  # For `x >= 0.5`, `|Gamma(x)| = (eh + el) / P * 2^kf`.
  # For `x < 0.5`, `|Gamma(x)| = P / (d_factor * sinc(delta) * (eh + el))`
  # scaled by `2^(-kf - e_ad)`.
  w0, w1 = two_prod(delta, delta)
  sc0, sc1 = poly_hybrid(w0, w1, sinc_poly)
  sc0, sc1 = f2_mul(sc0, sc1, w0, w1)
  sinc0, sinc1 = f2_add(one, zero, sc0, sc1)
  den_neg0, den_neg1 = f2_mul(d_factor, zero, sinc0, sinc1)
  den_neg0, den_neg1 = f2_mul(den_neg0, den_neg1, eh, el)

  m0, m1 = f2_div(
      lax.select(is_pos, eh, p0),
      lax.select(is_pos, el, p1),
      lax.select(is_pos, p0, den_neg0),
      lax.select(is_pos, p1, den_neg1),
  )
  k_tot = lax.select(is_pos, k_int, -k_int - e_ad_eff)
  mag = scale_f2_pow2(m0, m1, k_tot)

  # Apply sign `(-1)^floor(x)` on negative intervals and substitute special values:
  # `|x| < finfo.tiny -> +/-inf`, `+inf -> +inf`, negative integers / `-inf -> NaN`.
  neg_odd = lax.ne(lax.rem(x_floor, two), zero)
  sign = lax.select(is_pos, one, lax.select(neg_odd, neg_one, one))
  res = sign * mag
  pos_inf = lax.full_like(x, np.inf)
  neg_inf = lax.full_like(x, -np.inf)
  zero_inf = lax.select(sign_neg, neg_inf, pos_inf)
  res = lax.select(is_zero, zero_inf, res)
  res = lax.select(is_inf & (~sign_neg), pos_inf, res)
  is_nan_out = is_nan_in | (is_inf & sign_neg) | (sign_neg & is_int & (~is_zero))
  return _nan_where(is_nan_out, res)


@gamma_real.defjvp
def _gamma_real_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  g = gamma_real(x)
  return g, x_dot * g * lax_special.digamma(x)
