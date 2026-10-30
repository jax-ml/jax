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

"""Real log-beta implementation (`jax.scipy.special.betaln`).

Strategy for `_betaln(a, b)`:
- For strictly positive inputs `u, v > 0` (`_betaln_pos(u, v)` with
  `p = min(u, v)` and `q = max(u, v)`):
  * Regimes 1 & 2 (`q < 8`): evaluate via three `lax.lgamma` calls, shifting any
    argument `< 0.5` up by `+1` (`lgamma(x) = lgamma(x + 1) - log(x)`) so
    `lgamma` is never called near `0`, and shifting `u + v` down by `1` when
    `p` is near `1` (`[0.75, 1.25]`) to avoid cancellation and binade-crossing
    rounding in `u + v`. Very near `(1, 1)` (`|u - 1|, |v - 1| < 1e-3`), use the
    power series in `du = u - 1, dv = v - 1` whose linear Euler-Mascheroni terms
    cancel analytically.
  * Regime 3 (`p < 8 <= q`): combine `lgamma(p)` with Stirling's expansion for
    `lgamma(q) - lgamma(p + q)`. Decomposing `p = m_p * 2^e_p` and
    `q = m_q * 2^e_q` with `m_p, m_q in [0.75, 1.5)` and splitting `p` into
    high/low parts cancels `e_p` against `-p * e_q` exactly in `log(2)` before
    multiplying by `ln(2)`.
  * Regime 4 (`8 <= p <= q`): combine Stirling's series analytically using a
    symmetric formulation in `(u, v)` when `p / q >= 0.0625` (preserving smooth
    derivatives across the diagonal `u == v`) and a cancellation-free expansion
    in `h = p / q` (`_betaln_delta`) when `p / q < 0.0625`.
- For non-positive inputs (`p <= 0`), reduce arguments modulo `1` before
  evaluating `sin(pi * x)` and apply Euler's reflection formula to route all
  sign regimes into the single `_betaln_pos` evaluation:
  * `p < 0 < q, a + b < 0.5`: `betaln(1 - (a + b), q) + log|sin(pi*(a + b)) / sin(pi*p)|`
  * `p < 0 < q, a + b >= 0.5`: `-betaln(1 - p, a + b) - log(q) + log(pi) - log|sin(pi*p)|`
  * `p <= q < 0`: `-betaln(1 - a, 1 - b) - log(1 - (a + b)) + log(pi) + log|sin(pi*(a + b)) / (sin(pi*a) * sin(pi*b))|`
"""

from __future__ import annotations

import math

from jax._src import dtypes
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.api import jit
from jax._src.lax.lax import (
    AccuracyMode,
    _const as _lax_const,
    bitcast_convert_type,
)
from jax._src.numpy.ufuncs import isneginf, isposinf
from jax._src.typing import Array
import numpy as np

# Bernoulli coefficients B_{2k+2} / ((2k+1)(2k+2)) for the Stirling remainder
# of log(Gamma(z)) for z >= 7, in ascending degree order for lax.polynomial.
_STIRLING_REM_COEFFS_F64 = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
    -3617.0 / 122400.0,
)
_STIRLING_REM_COEFFS_F32 = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
)

# Taylor coefficients (-1)^{k-1} / (k * (k + 1)) for ((1 + h) * log(1 + h) - h) / h^2,
# in ascending degree order for lax.polynomial.
_BETALN_DELTA_COEFFS_F64 = tuple(
    ((-1) ** (k - 1)) / (k * (k + 1)) for k in range(1, 11)
)
_BETALN_DELTA_COEFFS_F32 = tuple(
    ((-1) ** (k - 1)) / (k * (k + 1)) for k in range(1, 7)
)

# Taylor coefficients (-1)^k * zeta(k) / k (with zeta(1) := gamma) for
# lgamma(1 + x) / x around x = 0, in ascending degree order for lax.polynomial.
_LGAMMA1P_SMALL_COEFFS_F64 = (
    -0.5772156649015329,
    0.8224670334241132,
    -0.40068563438653143,
    0.2705808084277845,
    -0.207385551028674,
    0.16955717699740816,
    -0.14404989676884614,
    0.125509669524743,
    -0.1113342658695647,
    0.10009945751278178,
)
_LGAMMA1P_SMALL_COEFFS_F32 = _LGAMMA1P_SMALL_COEFFS_F64[:6]


def _stirling_remainder(x: Array) -> Array:
  """Computes the Stirling series remainder R(x) = log(Gamma(x)) - (x - 0.5)*log(x) + x - 0.5*log(2*pi)."""
  # For x >= max_x, R(x) = 1 / (12 * x) << eps^2 is negligible in both value
  # and derivatives; clamping avoids 1 / inf and subnormal 1 / x^2 in autodiff.
  max_x = _lax_const(x, 2.0**500 if x.dtype == np.float64 else 2.0**60)
  x_safe = jnp.where(x < max_x, x, max_x)
  inv_x = 1.0 / x_safe
  inv_x2 = lax.square(inv_x)
  coeffs = (
      _STIRLING_REM_COEFFS_F64
      if x.dtype == np.float64
      else _STIRLING_REM_COEFFS_F32
  )
  rem = inv_x * lax.polynomial(inv_x2, coeffs, unroll=len(coeffs))
  return jnp.where(x < max_x, rem, 0.0)


def _betaln_delta(p: Array, q: Array, h: Array) -> Array:
  """Computes p - (p + q - 0.5) * log1p(p / q) without catastrophic cancellation."""
  coeffs = (
      _BETALN_DELTA_COEFFS_F64
      if p.dtype == np.float64
      else _BETALN_DELTA_COEFFS_F32
  )
  is_small = jnp.abs(h) <= 0.0625
  h_small = jnp.where(is_small, h, 0.0)
  poly = lax.polynomial(h_small, coeffs, unroll=len(coeffs))
  l1p = lax.log1p(h, accuracy=AccuracyMode.HIGHEST)
  small_h = 0.5 * l1p - p * (h_small * poly)
  p_large_h = jnp.where(is_small, 1.0, p)
  q_large_h = jnp.where(is_small, 8.0, q)
  l1p_large = jnp.where(is_small, 0.0, l1p)
  large_h = p_large_h - p_large_h * l1p_large - (q_large_h - 0.5) * l1p_large
  return jnp.where(is_small, small_h, large_h)


def _inv_pow2_scale(x: Array) -> Array:
  """Returns the exact power of two 2^(1 - e) such that x * 2^(1 - e) is in [2, 4) for normal x > 0."""
  min_normal = _lax_const(
      x, 2.0**-1022 if x.dtype == np.float64 else 2.0**-126
  )
  x = jnp.where(x >= min_normal, x, min_normal)
  if x.dtype == np.float64:
    bits = bitcast_convert_type(x, np.int64)
    inv_exp_bits = lax.bitwise_and(
        lax.bitwise_not(bits), _lax_const(bits, 0x7FF0000000000000)
    )
    return bitcast_convert_type(inv_exp_bits, np.float64)
  bits = bitcast_convert_type(x, np.int32)
  inv_exp_bits = lax.bitwise_and(
      lax.bitwise_not(bits), _lax_const(bits, 0x7F800000)
  )
  return bitcast_convert_type(inv_exp_bits, np.float32)


def _decompose_mant_exp(x: Array) -> tuple[Array, Array]:
  """Decomposes finite x > 0 into (m, e) with x = m * 2^e and m in [0.75, 1.5)."""
  sub_thresh = _lax_const(x, 2.0**-960 if x.dtype == np.float64 else 2.0**-60)
  huge_thresh = _lax_const(x, 2.0**960 if x.dtype == np.float64 else 2.0**60)
  is_sub = x < sub_thresh
  is_huge = x >= huge_thresh
  scale = jnp.where(
      is_sub,
      _lax_const(x, 2.0**64),
      jnp.where(is_huge, _lax_const(x, 2.0**-64), _lax_const(x, 1.0)),
  )
  exp_offset = jnp.where(
      is_sub,
      _lax_const(x, -64.0),
      jnp.where(is_huge, _lax_const(x, 64.0), _lax_const(x, 0.0)),
  )
  x_scaled = x * scale
  if x.dtype == np.float64:
    bits = bitcast_convert_type(x_scaled, np.int64)
    bits_round = bits + _lax_const(bits, 0x0008000000000000)
    e = ((bits_round >> 52) - _lax_const(bits, 1023)).astype(
        np.float64
    ) + exp_offset
    inv_exp_bits = lax.bitwise_and(
        lax.bitwise_not(bits_round), _lax_const(bits, 0x7FF0000000000000)
    )
    inv_pow2 = 0.5 * bitcast_convert_type(inv_exp_bits, np.float64)
  else:
    bits = bitcast_convert_type(x_scaled, np.int32)
    bits_round = bits + _lax_const(bits, 0x00400000)
    e = ((bits_round >> 23) - _lax_const(bits, 127)).astype(
        np.float32
    ) + exp_offset
    inv_exp_bits = lax.bitwise_and(
        lax.bitwise_not(bits_round), _lax_const(bits, 0x7F800000)
    )
    inv_pow2 = 0.5 * bitcast_convert_type(inv_exp_bits, np.float32)
  max_inv = _lax_const(x, 2.0**958 if x.dtype == np.float64 else 2.0**62)
  m = lax.stop_gradient(x_scaled * inv_pow2) + (
      x_scaled - lax.stop_gradient(x_scaled)
  ) * jnp.minimum(inv_pow2, max_inv)
  return m, e


def _betaln_pos(
    u: Array,
    v: Array,
    du_override: Array | None = None,
    dv_override: Array | None = None,
    uv_override: Array | None = None,
    absorb_log_uv_m1: Array | bool = False,
) -> Array:
  """Computes betaln(u, v) for strictly positive finite u, v > 0."""
  valid = (u > 0) & (v > 0) & jnp.isfinite(u) & jnp.isfinite(v)
  u = jnp.where(valid, u, 1.0)
  v = jnp.where(valid, v, 1.0)
  p = jnp.where(u <= v, u, v)
  q = jnp.where(u <= v, v, u)

  # Regimes 1, 2, and 3 share just three lax.lgamma evaluations.
  # Mask inputs >= 8 to 1.0 so lgamma never overflows in unselected branches,
  # and shift any argument < 0.5 by +1 via lgamma(x) = lgamma(x + 1) - log(x)
  # so lax.lgamma is never evaluated near 0 or on subnormal inputs.
  u_lt_8 = jnp.where(u < 8.0, u, 1.0)
  v_lt_8 = jnp.where(v < 8.0, v, 1.0)

  u_shift = jnp.where(u_lt_8 < 0.5, 1.0, 0.0)
  v_shift = jnp.where(v_lt_8 < 0.5, 1.0, 0.0)

  # When p = min(u, v) lies in [0.75, 1.25] and q < 8, shift u + v = 1 + dp + q
  # down by 1 via lgamma(1 + q + dp) = log(q + dp) + lgamma(q + dp) so both
  # (u, v) near (1, 1) and p near 1 with q in (1.25, 8) avoid cancellation and
  # binade-crossing rounding in u + v.
  both_near_1 = (p >= 0.75) & (q <= 1.25)
  p_near_1 = (p >= 0.75) & (p <= 1.25) & (q < 8.0)
  du = (
      u_lt_8 - 1.0
      if du_override is None
      else jnp.where(valid & (u < 8.0), du_override, 0.0)
  )
  dv = (
      v_lt_8 - 1.0
      if dv_override is None
      else jnp.where(valid & (v < 8.0), dv_override, 0.0)
  )
  dp_uv = jnp.where(u <= v, du, dv)
  d_sum = jnp.where(both_near_1, du + dv, 0.0)
  q_plus_dp = jnp.where(p_near_1, jnp.where(q < 8.0, q, 1.0) + dp_uv, 1.0)
  uv_base = (
      u_lt_8 + v_lt_8
      if uv_override is None
      else jnp.where(valid & (q < 8.0), uv_override, 1.0)
  )

  lg_u_shifted = lax.lgamma(u_lt_8 + u_shift)
  lg_v_shifted = lax.lgamma(v_lt_8 + v_shift)
  uv_arg = jnp.where(
      both_near_1,
      1.0 + d_sum,
      jnp.where(
          p_near_1,
          q_plus_dp,
          jnp.where(q < 8.0, uv_base, 1.0)
          + jnp.where((q < 0.5) & ~absorb_log_uv_m1, 1.0, 0.0),
      ),
  )
  lg_uv_shifted = lax.lgamma(uv_arg)

  log_uv_shift = jnp.where(
      (q < 0.5) & ~absorb_log_uv_m1,
      lax.log(
          jnp.where((q < 0.5) & ~absorb_log_uv_m1, uv_base, 1.0),
          accuracy=AccuracyMode.HIGHEST,
      ),
      0.0,
  )
  near1_arg = jnp.where(
      both_near_1, d_sum, jnp.where(p_near_1, q_plus_dp - 1.0, 0.0)
  )
  use_near1_shift = p_near_1 & ~absorb_log_uv_m1

  # For |du|, |dv| < 1e-3, lgamma(1 + du) + lgamma(1 + dv) - lgamma(1 + du + dv)
  # cancels its linear Euler-Mascheroni terms -gamma * (du + dv) identically,
  # leaving the series sum_{k>=2} (-1)^k zeta(k)/k * (du^k + dv^k - (du + dv)^k).
  very_near_1 = both_near_1 & (jnp.abs(du) < 1e-3) & (jnp.abs(dv) < 1e-3)
  pi2_over_6 = _lax_const(u, math.pi * math.pi / 6.0)
  zeta3 = _lax_const(u, 1.2020569031595942)
  du_tiny = jnp.where(very_near_1, du, 0.0)
  dv_tiny = jnp.where(very_near_1, dv, 0.0)
  du_dv = du_tiny * dv_tiny
  if u.dtype == np.float64:
    half_zeta4 = _lax_const(u, math.pi**4 / 180.0)
    zeta5 = _lax_const(u, 1.0369277551433699)
    zeta6 = _lax_const(u, math.pi**6 / 945.0)
    s_d2 = lax.square(d_sum)
    pos_terms = du_dv * (d_sum * (zeta3 + zeta5 * (s_d2 - du_dv)))
    neg_terms = du_dv * (
        pi2_over_6
        + half_zeta4 * (2.0 * s_d2 - du_dv)
        + zeta6
        * (s_d2 * (s_d2 - 1.5 * du_dv) + (1.0 / 3.0) * lax.square(du_dv))
    )
  else:
    pos_terms = du_dv * (d_sum * zeta3)
    neg_terms = du_dv * pi2_over_6
  lg_comb = jnp.where(
      very_near_1,
      pos_terms - neg_terms,
      (lg_u_shifted + lg_v_shifted) - lg_uv_shifted,
  )

  # Regime 3 & 4: q >= 8.
  # Multiply numerator and denominator by inv_q_scale = 2^(1 - floor(log2(q_large)))
  # so q_s = q_large * inv_q_scale is in [2, 4), preventing overflow or underflow
  # in lax.div's JVP and transpose rules across the entire float domain.
  # When absorb_log_uv_m1 is True and p < 8 <= q, evaluate
  # betaln(p, q) + log(p + q - 1) = lgamma(p) + lgamma(q) - lgamma((p - 1) + q)
  # directly using p_m1 = p - 1 so log(q) does not cancel -(1 - p_f)*log(q).
  absorb_in_mid = absorb_log_uv_m1 & (q >= 8.0) & (p < 8.0)
  q_large = jnp.where(q >= 8.0, q, 8.0)
  p_large = jnp.where(absorb_in_mid, dp_uv, jnp.where(q >= 8.0, p, 1.0))
  inv_q_scale = _inv_pow2_scale(q_large)
  scale_pq = jnp.where(
      jnp.abs(p_large) >= 1.0, inv_q_scale, _lax_const(u, 1.0)
  )
  h = (p_large * scale_pq) / (q_large * scale_pq)
  use_delta = (p < 8.0) | (h < 0.0625)
  p_delta = jnp.where(use_delta, p_large, 1.0)
  q_delta = jnp.where(use_delta, q_large, 8.0)
  h_delta = jnp.where(use_delta, h, 0.0625)
  delta = _betaln_delta(p_delta, q_delta, h_delta)

  # Regime 4 setup (8 <= p <= q) and shared log/log1p/Stirling evaluations
  # across mutually exclusive regimes.
  u_hi = jnp.where(p >= 8.0, u, 8.0)
  v_hi = jnp.where(p >= 8.0, v, 8.0)
  p_hi_8 = jnp.where(p >= 8.0, p, 8.0)
  u_s = u_hi * inv_q_scale
  v_s = v_hi * inv_q_scale
  p_s = p_hi_8 * inv_q_scale
  q_s = q_large * inv_q_scale
  h_hi = p_s / q_s
  u_sym = jnp.where(h_hi >= 0.0625, u_hi, 8.0)
  v_sym = jnp.where(h_hi >= 0.0625, v_hi, 8.0)
  u_sym_s = jnp.where(h_hi >= 0.0625, u_s, 1.0)
  v_sym_s = jnp.where(h_hi >= 0.0625, v_s, 1.0)
  p_asym = jnp.where(h_hi < 0.0625, p_hi_8, 8.0)
  h_asym = jnp.where(h_hi < 0.0625, h_hi, 0.0625)

  in_mid = (q >= 8.0) & (p < 8.0)
  need_uv_log_shift = (q < 8.0) | absorb_in_mid
  log1_val = lax.log(
      jnp.where(
          p >= 8.0,
          jnp.where(h_hi < 0.0625, q_large, u_sym),
          jnp.where(need_uv_log_shift & (u_lt_8 < 0.5), u_lt_8, 1.0),
      ),
      accuracy=AccuracyMode.HIGHEST,
  )
  log2_val = lax.log(
      jnp.where(
          p >= 8.0,
          jnp.where(h_hi < 0.0625, h_asym, v_sym),
          jnp.where(need_uv_log_shift & (v_lt_8 < 0.5), v_lt_8, 1.0),
      ),
      accuracy=AccuracyMode.HIGHEST,
  )
  log_u_shift = jnp.where(u_lt_8 < 0.5, log1_val, 0.0)
  log_v_shift = jnp.where(v_lt_8 < 0.5, log2_val, 0.0)

  # Regime 3: p < 8 <= q.
  # Decompose p = m_p * 2^e_p and q = m_q * 2^e_q with m_p, m_q in [0.75, 1.5)
  # and split p_mid = p_hi + p_lo so k_hi = e_p + p_hi * e_q cancels e_p against
  # p * e_q exactly before multiplying by log(2) when p < 0.5.
  use_p_shift = in_mid & (p < 0.5) & ~absorb_in_mid
  p_pos_mid = jnp.where(use_p_shift, p, 1.0)
  q_mid = jnp.where(in_mid, q_large, 8.0)
  p_small_thresh = _lax_const(u, 0.03125 if u.dtype == np.float64 else 0.0625)
  lg1p_coeffs = (
      _LGAMMA1P_SMALL_COEFFS_F64
      if u.dtype == np.float64
      else _LGAMMA1P_SMALL_COEFFS_F32
  )
  p_tiny_arg = jnp.where(p_pos_mid < p_small_thresh, p_pos_mid, 0.0)
  lg_p1_tiny = p_tiny_arg * lax.polynomial(
      p_tiny_arg, lg1p_coeffs, unroll=len(lg1p_coeffs)
  )
  lg_p_shifted = jnp.where(u <= v, lg_u_shifted, lg_v_shifted)
  lg_p_mid = jnp.where(
      use_p_shift & (p_pos_mid < p_small_thresh),
      lg_p1_tiny,
      jnp.where(
          absorb_in_mid & (p < 0.5),
          lg_p_shifted - jnp.where(u <= v, log_u_shift, log_v_shift),
          lg_p_shifted,
      ),
  )
  p_mid = jnp.where(p < 8.0, p_large, 1.0)
  m_p, e_p = _decompose_mant_exp(p_pos_mid)
  m_q, e_q = _decompose_mant_exp(q_mid)
  if u.dtype == np.float64:
    p_bits = bitcast_convert_type(p_mid, np.int64)
    p_hi = bitcast_convert_type(
        lax.bitwise_and(p_bits, _lax_const(p_bits, -0x4000000)), np.float64
    )
    ln2_hi = _lax_const(u, 0.6931471805599453)
    ln2_lo = _lax_const(u, 2.3190468138462996e-17)
  else:
    p_bits = bitcast_convert_type(p_mid, np.int32)
    p_hi = bitcast_convert_type(
        lax.bitwise_and(p_bits, _lax_const(p_bits, -0x1000)), np.float32
    )
    ln2_hi = _lax_const(u, 0.693145751953125)
    ln2_lo = _lax_const(u, 1.428606765330187e-06)
  p_lo = p_mid - p_hi
  k_hi = jnp.where(use_p_shift, e_p, 0.0) + p_hi * e_q
  k_lo = p_lo * e_q

  l1p1_val = lax.log1p(
      jnp.where(
          q < 8.0,
          jnp.where(use_near1_shift, near1_arg, 0.0),
          jnp.where(
              p < 8.0,
              jnp.where(use_p_shift, m_p - 1.0, 0.0),
              v_sym_s / u_sym_s,
          ),
      ),
      accuracy=AccuracyMode.HIGHEST,
  )
  l1p2_val = lax.log1p(
      jnp.where(
          q < 8.0,
          0.0,
          jnp.where(p < 8.0, m_q - 1.0, u_sym_s / v_sym_s),
      ),
      accuracy=AccuracyMode.HIGHEST,
  )
  log_near1_shift = jnp.where(use_near1_shift, l1p1_val, 0.0)
  log_mp = jnp.where(use_p_shift, l1p1_val, 0.0)
  log_mq = l1p2_val

  # Regime 1 (q < 0.5) & Regime 2 (0.5 <= q < 8):
  # Syntactically symmetric in (u, v), with lgamma evaluated only on [0.5, 16).
  res_q_lt_8 = (log_uv_shift + lg_comb) - (
      log_u_shift + log_v_shift + log_near1_shift
  )

  rem_u = _stirling_remainder(jnp.where(p < 8.0, q_mid, u_hi))
  rem_v = _stirling_remainder(v_hi)
  rem_uv = _stirling_remainder(jnp.where(p < 8.0, p_mid + q_mid, u_hi + v_hi))
  rem_mid = rem_u - rem_uv
  rem_both = (rem_u + rem_v) - rem_uv

  res_mid = (-k_hi * ln2_hi - log_mp) + (
      (lg_p_mid - p_mid * log_mq)
      - (k_lo * ln2_hi + (k_hi + k_lo) * ln2_lo)
      + delta
      + rem_mid
  )

  # Regime 4: 8 <= p <= q.
  # For h >= 0.0625 (including all points near the diagonal u == v), use the
  # symmetric Stirling form in (u, v) so values and higher-order derivatives
  # are smooth across u == v. For h < 0.0625, use the cancellation-free
  # expansion in h = p / q.
  half_log_2pi = _lax_const(u, 0.5 * math.log(2.0 * math.pi))
  res_both_sym = (
      half_log_2pi
      - 0.25 * (log1_val + log2_val)
      - ((u_sym - 0.25) * l1p1_val + (v_sym - 0.25) * l1p2_val)
      + rem_both
  )
  res_both_asym = (
      half_log_2pi
      - 0.5 * log1_val
      + (p_asym - 0.5) * log2_val
      - p_asym
      + delta
      + rem_both
  )
  res_both_large = jnp.where(h_hi < 0.0625, res_both_asym, res_both_sym)

  res_q_ge_8 = jnp.where(p < 8.0, res_mid, res_both_large)
  return jnp.where(q < 8.0, res_q_lt_8, res_q_ge_8)


def _log_abs_sinpi_reduced(d: Array) -> tuple[Array, Array]:
  """Computes (log(|sin(pi * d)|), |sin(pi * d)|) for reduced d in [-0.5, 0.5]."""
  d_safe = jnp.where(d == 0, 0.5, d)
  abs_d = jnp.abs(d_safe)
  # For tiny |d|, sin(pi * d) = pi * d * (1 - (pi * d)^2 / 6 + ...), avoiding
  # subnormal underflow of pi * d near the float underflow threshold.
  pi_d2 = lax.square(np.pi * d_safe)
  abs_sin = jnp.abs(jnp.sin(np.pi * d_safe))
  log_val = lax.log(
      jnp.where(abs_d < 1e-4, abs_d, abs_sin),
      accuracy=AccuracyMode.HIGHEST,
  )
  small = (
      _lax_const(d, math.log(math.pi))
      + log_val
      - pi_d2 * (1.0 / 6.0 + pi_d2 / 180.0)
  )
  return jnp.where(abs_d < 1e-4, small, log_val), abs_sin


@jit
def _betaln(a: Array, b: Array) -> Array:
  orig_dtype = a.dtype
  if dtypes.finfo(orig_dtype).bits < 32:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

  p = jnp.where(a <= b, a, b)
  q = jnp.where(a <= b, b, a)

  # Sanitize non-finite values before evaluating finite branches.
  finite_pq = jnp.isfinite(a) & jnp.isfinite(b)
  a_f = jnp.where(finite_pq, a, 1.0)
  b_f = jnp.where(finite_pq, b, 1.0)
  p_f = jnp.where(a_f <= b_f, a_f, b_f)
  q_f = jnp.where(a_f <= b_f, b_f, a_f)
  s_f = a_f + b_f

  # Reduce modulo 1 before forming sin(pi * x) so large magnitudes do not lose
  # fractional precision and derivatives remain pi * cot(pi * x).
  # Subtract r_d from the larger of |da|, |db| first so no low-order bit is lost
  # when |da + db| > 0.5 and r_d == +-1.
  na = lax.round(a_f, lax.RoundingMethod.TO_NEAREST_EVEN)
  nb = lax.round(b_f, lax.RoundingMethod.TO_NEAREST_EVEN)
  da = a_f - na
  db = b_f - nb
  r_d = lax.round(da + db, lax.RoundingMethod.TO_NEAREST_EVEN)
  half_r_d = 0.5 * r_d
  ds = jnp.where(
      (jnp.abs(da) >= 0.25) & (jnp.abs(db) >= 0.25),
      (da - half_r_d) + (db - half_r_d),
      jnp.where(
          jnp.abs(da) >= jnp.abs(db),
          ((da - half_r_d) + db) - half_r_d,
          ((db - half_r_d) + da) - half_r_d,
      ),
  )
  dp = jnp.where(a_f <= b_f, da, db)
  dq = jnp.where(a_f <= b_f, db, da)

  both_pos = p_f > 0
  mixed_sign = (p_f <= 0) & (q_f > 0)
  mixed_s_lt_half = mixed_sign & (s_f < 0.5)
  mixed_s_ge_half = mixed_sign & (s_f >= 0.5)

  # When |da|, |db| are near tiny with opposite signs (e.g. a = tiny + ulp,
  # b = -tiny), da + db can be subnormal and flush to zero on FTZ backends.
  # Rescaling da and db by 2^(1 - exp) keeps their sum in normal float range,
  # and splitting half_d_scale prevents XLA's AlgebraicSimplifier from factoring
  # (da + db) * d_scale.
  d_max = jnp.maximum(jnp.abs(da), jnp.abs(db))
  d_min = jnp.minimum(jnp.abs(da), jnp.abs(db))
  d_signed_max = jnp.where(jnp.abs(da) >= jnp.abs(db), da, db)
  d_signed_min = jnp.where(jnp.abs(da) >= jnp.abs(db), db, da)
  d_scale = jnp.where(
      ~both_pos & (d_max > 0) & (d_max < 0.25),
      _inv_pow2_scale(jnp.where(d_max > 0, d_max, 1.0)),
      _lax_const(a_f, 1.0),
  )
  half_d_scale = 0.5 * d_scale
  da_scaled = da * d_scale
  db_scaled = db * d_scale
  ds_scaled = jnp.where(
      r_d == 0,
      jnp.where(
          ((da < 0) != (db < 0)) & (d_min >= 0.5 * d_max),
          (d_signed_min * d_scale + d_signed_max * half_d_scale)
          + d_signed_max * half_d_scale,
          da_scaled + db_scaled,
      ),
      ds * d_scale,
  )
  dp_scaled = jnp.where(a_f <= b_f, da_scaled, db_scaled)

  p_is_int = dp == 0
  q_is_int = dq == 0
  # Since da, db are in [-0.5, 0.5], da + db is an integer in exact arithmetic
  # iff da == -db or |da| == 0.5 and da == db (avoiding FTZ false positives on ds == 0).
  s_is_int = (da == -db) | ((jnp.abs(da) == 0.5) & (da == db))
  int_pole_cancels = p_is_int & q_is_int & (q_f > 0) & (s_f <= 0)

  log_pi = _lax_const(a_f, math.log(math.pi))
  log_sin_p, abs_sin_p = _log_abs_sinpi_reduced(jnp.where(p_f < 0, dp, 0.5))
  log_sin_q, _ = _log_abs_sinpi_reduced(jnp.where(q_f < 0, dq, 0.5))
  log_sin_s, abs_sin_s = _log_abs_sinpi_reduced(
      jnp.where((p_f < 0) & ((q_f < 0) | (s_f < 0.5)), ds, 0.5)
  )

  # For |dp| < 1e-4 and |ds| < 1e-4 (with r_d == 0), evaluate
  # log(|sin(pi * ds) / sin(pi * dp)|) from the scaled ratio ds_scaled / dp_scaled
  # so log(pi) and the common exponent cancel before taking the logarithm.
  # When both |dp| >= 1e-4 and |ds| >= 1e-4, take a single logarithm of
  # |sin(pi * ds)| / |sin(pi * dp)| to avoid subtracting two logarithms.
  min_den = _lax_const(a_f, 1e-100 if a_f.dtype == np.float64 else 1e-15)
  abs_ds_s = jnp.abs(ds_scaled)
  abs_dp_s = jnp.abs(dp_scaled)
  use_ratio_sp = mixed_s_lt_half & ~int_pole_cancels
  both_d_tiny = (
      use_ratio_sp
      & (d_max < 1e-4)
      & (r_d == 0)
      & (abs_dp_s >= min_den)
      & (abs_ds_s >= min_den)
  )
  both_d_normal = (
      use_ratio_sp
      & ~both_d_tiny
      & (jnp.abs(dp) >= 1e-4)
      & (jnp.abs(ds) >= 1e-4)
  )
  ratio_sp = jnp.where(
      both_d_tiny,
      jnp.where(both_d_tiny, abs_ds_s, 1.0)
      / jnp.where(both_d_tiny, abs_dp_s, 1.0),
      jnp.where(
          both_d_normal,
          jnp.where(both_d_normal, abs_sin_s, 1.0)
          / jnp.where(both_d_normal, abs_sin_p, 1.0),
          1.0,
      ),
  )

  # Route all sign/reflection regimes into a single _betaln_pos(u, v) call:
  # 1. p_f > 0: +_betaln_pos(a_f, b_f)
  # 2. p_f < 0 < q_f, s_f < 0.5: +_betaln_pos(1 - s_f, q_f) + log|sin(pi*s)/sin(pi*p)|
  # 3. p_f < 0 < q_f, s_f >= 0.5: -_betaln_pos(1 - p_f, s_f) - log(q_f) + log(pi) - log|sin(pi*p)|
  # 4. p_f <= q_f < 0: -_betaln_pos(1 - a_f, 1 - b_f) - log(1 - s_f) + log(pi) + log|sin(pi*s)/(sin(pi*a)sin(pi*b))|

  small_ab = (jnp.abs(a_f) <= 2.0**20) & (jnp.abs(b_f) <= 2.0**20)
  n_sum = (na + nb) + r_d
  s_acc = jnp.where(small_ab, n_sum + ds, s_f)
  one_minus_s = jnp.where(small_ab, (1.0 - n_sum) - ds, 1.0 - s_f)

  u = jnp.where(
      both_pos,
      a_f,
      jnp.where(
          mixed_s_lt_half,
          one_minus_s,
          jnp.where(mixed_s_ge_half, 1.0 - p_f, 1.0 - a_f),
      ),
  )
  v = jnp.where(
      both_pos,
      b_f,
      jnp.where(
          mixed_s_lt_half,
          q_f,
          jnp.where(mixed_s_ge_half, s_acc, 1.0 - b_f),
      ),
  )
  du_override = jnp.where(
      both_pos,
      a_f - 1.0,
      jnp.where(
          mixed_s_lt_half,
          jnp.where((na + nb == 0) & (r_d == 0), -ds, -s_acc),
          jnp.where(mixed_s_ge_half, -p_f, -a_f),
      ),
  )
  dv_override = jnp.where(
      both_pos,
      b_f - 1.0,
      jnp.where(
          mixed_s_lt_half,
          q_f - 1.0,
          jnp.where(
              mixed_s_ge_half,
              jnp.where((na + nb == 1) & (r_d == 0), ds, s_acc - 1.0),
              -b_f,
          ),
      ),
  )
  uv_override = jnp.where(
      both_pos,
      s_acc,
      jnp.where(
          mixed_s_lt_half,
          1.0 - p_f,
          jnp.where(mixed_s_ge_half, q_f, one_minus_s),
      ),
  )
  absorb_log_uv_m1 = mixed_s_ge_half | ((p_f <= 0) & (q_f <= 0))
  base_pos = _betaln_pos(
      u,
      v,
      du_override=du_override,
      dv_override=dv_override,
      uv_override=uv_override,
      absorb_log_uv_m1=absorb_log_uv_m1,
  )

  p_uv = jnp.where(u <= v, u, v)
  need_log_uv_m1 = absorb_log_uv_m1 & (p_uv >= 8.0)
  uv_m1_rem = jnp.where(
      need_log_uv_m1,
      jnp.where(q_f > 0, q_f, jnp.where(s_f < 0, one_minus_s, 1.0)),
      1.0,
  )
  shared_log_arg = jnp.where(need_log_uv_m1, uv_m1_rem, ratio_sp)
  shared_log_val = lax.log(shared_log_arg, accuracy=AccuracyMode.HIGHEST)
  log_uv_m1_rem = jnp.where(need_log_uv_m1, shared_log_val, 0.0)
  log_sin_ratio_sp = jnp.where(
      both_d_tiny,
      shared_log_val - (np.pi * np.pi / 6.0) * (lax.square(ds) - lax.square(dp)),
      jnp.where(both_d_normal, shared_log_val, log_sin_s - log_sin_p),
  )

  res_neg_qpos_s_lt_half = base_pos + jnp.where(
      int_pole_cancels, 0.0, log_sin_ratio_sp
  )
  res_neg_qpos_s_ge_half = -base_pos - log_uv_m1_rem + log_pi - log_sin_p
  res_neg_qneg = (
      -base_pos
      - log_uv_m1_rem
      + log_pi
      + log_sin_s
      - (log_sin_p + log_sin_q)
  )

  res_neg = jnp.where(
      q_f > 0,
      jnp.where(s_f < 0.5, res_neg_qpos_s_lt_half, res_neg_qpos_s_ge_half),
      res_neg_qneg,
  )

  # Special values and poles for p <= 0:
  # - Any zero argument returns +inf.
  # - Negative integer p returns +inf unless q > 0 is an integer with p + q <= 0.
  # - Negative integer q (with non-integer p < 0) returns +inf.
  # - Non-positive integer p + q (with neither p nor q an integer) returns -inf.
  is_pos_inf_pole = (
      (p_f == 0)
      | (q_f == 0)
      | (p_is_int & ~int_pole_cancels)
      | ((q_f < 0) & q_is_int)
  )
  is_neg_inf_pole = ~is_pos_inf_pole & ~int_pole_cancels & (s_f <= 0) & s_is_int
  res_neg = jnp.where(
      is_pos_inf_pole,
      jnp.inf,
      jnp.where(is_neg_inf_pole, -jnp.inf, res_neg),
  )

  res_finite = jnp.where(both_pos, base_pos, res_neg)

  # Handle NaNs and infinities:
  # - nan input or (+inf, +inf) -> nan
  # - (-inf, anything) -> +inf
  # - (p, +inf) with finite p > 0 -> -inf; with finite p <= 0 -> +inf
  is_nan = jnp.isnan(a) | jnp.isnan(b) | (isposinf(p) & isposinf(q))
  res_inf = jnp.where(
      isneginf(p),
      jnp.inf,
      jnp.where(p > 0, -jnp.inf, jnp.inf),
  )
  res = jnp.where(is_nan, jnp.nan, jnp.where(finite_pq, res_finite, res_inf))
  return res.astype(orig_dtype)
