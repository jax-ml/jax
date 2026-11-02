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

from __future__ import annotations

import fractions
from functools import partial
import math
import operator
import numpy as np

from jax._src import config
from jax._src import core
from jax._src import custom_derivatives
from jax._src import dtypes
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.api import jit
from jax._src.numpy.ufuncs import cbrt, copysign, exp2
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def _jn_series(n: int, w: Array, max_iter: int, r_max: float) -> Array:
  """Ascending power series for J_n(w) for integer n >= 0 and Re(w), Im(w) >= 0."""
  real_dtype = jnp.real(w).dtype
  w_safe = jnp.where(jnp.isfinite(w) & (jnp.abs(w) <= r_max), w, 0.0)
  z2 = -0.25 * (w_safe * w_safe)

  def body(k, state):
    term, total = state
    kf = k.astype(real_dtype)
    term = term * (z2 / (kf * (n + kf)))
    total = total + term
    return term, total

  one = jnp.ones_like(w_safe)
  _, total = lax.fori_loop(1, max_iter + 1, body, (one, one))

  if n == 0:
    res = total
  elif n == 1:
    res = w_safe * (0.5 * total)
  else:
    # Factor (w / 2)^n / n! = (2^-q * w)^(n - rem) * (2^-(q+1) * w)^rem * C_n
    # where 2^k_exp / n! = C_n in [0.707, 1.414] is precomputed in Python and
    # the power-of-2 scalings of w are exact in IEEE floating-point.
    if n <= 1000:
      fact = math.factorial(n)
      k_exp = round(math.lgamma(n + 1) / math.log(2.0))
      c_n = float(fractions.Fraction(1 << k_exp, fact))
    else:
      lg2 = math.lgamma(n + 1) / math.log(2.0)
      k_exp = round(lg2)
      c_n = math.pow(2.0, k_exp - lg2)
    q, rem = divmod(k_exp + n, n)
    w1 = w_safe * float(2.0 ** (-q))
    scaled_total = total * c_n
    if rem == 0:
      n1 = n // 2
      n2 = n - n1
      res = ((w1**n1) * scaled_total) * (w1**n2)
    else:
      w2 = w_safe * float(2.0 ** (-(q + 1)))
      res = ((w1 ** (n - rem)) * scaled_total) * (w2**rem)

  zero = jnp.zeros_like(w)
  return jnp.where(w == 0, one if n == 0 else zero, res)


def _hankel_sincos_chi_real(n: int, x: Array) -> tuple[Array, Array]:
  """Computes (cos(chi), sin(chi)) for chi = x - (n/2 + 1/4)*pi with real x >= 0."""
  real_dtype = x.dtype
  x_safe = jnp.where(jnp.isfinite(x) & (x >= 0.0), x, 10.0)
  max_c2x = 8e307 if real_dtype == np.float64 else 1.7e38

  # Evaluate cos(x) +/- sin(x) stabilized by cos(2x) = cos^2(x) - sin^2(x)
  # to avoid cancellation when cos(x) and sin(x) have nearly equal magnitudes.
  sx = jnp.sin(x_safe)
  cx = jnp.cos(x_safe)
  can_c2x = x_safe <= max_c2x
  c2x = jnp.cos(2.0 * jnp.minimum(x_safe, max_c2x))
  u_sum = cx + sx
  v_diff = sx - cx
  u_stab = jnp.where(
      (cx * sx >= 0.0) | ~can_c2x,
      u_sum,
      -c2x / jnp.where(cx * sx < 0.0, v_diff, 1.0),
  )
  v_stab = jnp.where(
      (cx * sx < 0.0) | ~can_c2x,
      v_diff,
      -c2x / jnp.where(cx * sx >= 0.0, u_sum, 1.0),
  )
  inv_sqrt2 = math.sqrt(0.5)
  c0 = inv_sqrt2 * u_stab
  s0 = inv_sqrt2 * v_stab
  # Exact value at x == 0 (for purely imaginary w = i*y).
  c0 = jnp.where(x_safe == 0.0, inv_sqrt2, c0)
  s0 = jnp.where(x_safe == 0.0, -inv_sqrt2, s0)

  # Rotate by -n * pi/2 according to n mod 4.
  mod4 = n & 3
  if mod4 == 0:
    return c0, s0
  elif mod4 == 1:
    return s0, -c0
  elif mod4 == 2:
    return -c0, -s0
  else:
    return -s0, c0


def _hankel_pq(n: int, w_safe: Array, max_pairs: int) -> tuple[Array, Array]:
  """Evaluates Hankel's asymptotic series (P_n(w), Q_n(w)) for large |w|."""
  real_dtype = jnp.real(w_safe).dtype
  eps = dtypes.finfo(real_dtype).eps
  mu = float(4 * n * n)
  inv_8w = 0.125 / w_safe

  real_w = jnp.real(w_safe)
  one_w = jnp.ones_like(w_safe)
  zero_w = jnp.zeros_like(w_safe)
  inf_r = jnp.full_like(real_w, np.inf)
  true_b = jnp.ones_like(real_w, dtype=np.bool_)

  def body(m, state):
    p, q, t_prev, sign, prev_err, active = state
    mf = m.astype(real_dtype)
    k1 = 4.0 * mf - 3.0
    k2 = 4.0 * mf - 1.0
    t_odd = t_prev * ((mu - k1 * k1) / (2.0 * mf - 1.0)) * inv_8w
    t_even = t_odd * ((mu - k2 * k2) / (2.0 * mf)) * inv_8w
    err = jnp.abs(t_odd) + jnp.abs(t_even)
    still_active = (
        active
        & jnp.isfinite(err)
        & ((err <= prev_err) | (mf <= 2.0))
        & (prev_err > eps * 0.1)
    )
    q = jnp.where(still_active, q + sign * t_odd, q)
    p = jnp.where(still_active, p - sign * t_even, p)
    t_next = jnp.where(still_active, t_even, t_prev)
    next_err = jnp.where(still_active, err, prev_err)
    return p, q, t_next, -sign, next_err, still_active

  init = (one_w, zero_w, one_w, jnp.ones_like(real_w), inf_r, true_b)
  p, q, _, _, _, _ = lax.fori_loop(1, max_pairs + 1, body, init)
  return p, q


def _hankel_unscaled_complex(
    p: Array,
    q: Array,
    c_chi: Array,
    s_chi: Array,
    half_amp: Array,
    exp_neg2y: Array,
) -> Array:
  """Computes exponentially scaled e^{-y} * J_n(w) from Hankel (P, Q) for complex w."""
  i_c = lax.complex(jnp.zeros_like(c_chi), jnp.ones_like(c_chi))
  e_minus = lax.complex(c_chi, -s_chi)
  e_plus = lax.complex(c_chi, s_chi)
  h2_part = e_minus * (p - i_c * q)
  h1_part = exp_neg2y * e_plus * (p + i_c * q)
  return half_amp * (h2_part + h1_part)


def _jn_asymptotic(n: int, w: Array, max_pairs: int, r_min: float) -> Array:
  """Hankel's asymptotic expansion for J_n(w) for large |w|."""
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  w_safe = jnp.where(
      jnp.isfinite(w) & (jnp.abs(w) >= r_min), w, jnp.full_like(w, r_min)
  )
  p, q = _hankel_pq(n, w_safe, max_pairs)
  x = jnp.real(w_safe)
  c_chi, s_chi = _hankel_sincos_chi_real(n, jnp.maximum(x, 0.0))

  if not is_complex:
    amp = math.sqrt(2.0 / math.pi) / jnp.sqrt(x)
    return amp * (p * c_chi - q * s_chi)
  else:
    y = jnp.imag(w_safe)
    half_amp = (1.0 / math.sqrt(2.0 * math.pi)) / jnp.sqrt(w_safe)
    exp_neg2y = jnp.exp(-2.0 * y)
    unscaled = _hankel_unscaled_complex(
        p, q, c_chi, s_chi, half_amp, exp_neg2y
    )
    max_log = 700.0 if real_dtype == np.float64 else 85.0
    ey_half = jnp.exp(jnp.clip(0.5 * y, 0.0, max_log))
    re = (lax.real(unscaled) * ey_half) * ey_half
    im = (lax.imag(unscaled) * ey_half) * ey_half
    return lax.complex(re, im)


def _jn_asymptotic_j0_j1(w_safe: Array, max_pairs: int) -> tuple[Array, Array]:
  """Evaluates (J_0, J_1) (exponentially scaled by e^{-y} if complex) sharing trig calls."""
  is_complex = dtypes.issubdtype(w_safe.dtype, np.complexfloating)
  p0, q0 = _hankel_pq(0, w_safe, max_pairs)
  p1, q1 = _hankel_pq(1, w_safe, max_pairs)
  x = jnp.real(w_safe)
  c0, s0 = _hankel_sincos_chi_real(0, jnp.maximum(x, 0.0))
  # For n = 1, chi_1 = chi_0 - pi/2, so (cos(chi_1), sin(chi_1)) = (s0, -c0).
  c1, s1 = s0, -c0

  if not is_complex:
    amp = math.sqrt(2.0 / math.pi) / jnp.sqrt(x)
    j0 = amp * (p0 * c0 - q0 * s0)
    j1 = amp * (p1 * c1 - q1 * s1)
    return j0, j1
  else:
    y = jnp.imag(w_safe)
    half_amp = (1.0 / math.sqrt(2.0 * math.pi)) / jnp.sqrt(w_safe)
    exp_neg2y = jnp.exp(-2.0 * y)
    j0e = _hankel_unscaled_complex(p0, q0, c0, s0, half_amp, exp_neg2y)
    j1e = _hankel_unscaled_complex(p1, q1, c1, s1, half_amp, exp_neg2y)
    return j0e, j1e


def _jn_miller(
    n: int, w: Array, m_steps: int, r_min: float, rl: float
) -> Array:
  """Computes J_n(w) for r_min < |w| < rl via Miller's backward recurrence."""
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  r = jnp.abs(w)
  w_default = 0.5 * (r_min + rl)
  w_safe = jnp.where(
      jnp.isfinite(w) & (r >= r_min) & (r <= rl),
      w,
      jnp.full_like(w, w_default),
  )

  scle = 1e-100 if real_dtype == np.float64 else 1e-30
  scale_limit = 1e100 if real_dtype == np.float64 else 1e20
  scale_down = 1e-100 if real_dtype == np.float64 else 1e-20

  p_next = jnp.zeros_like(w_safe)
  p_curr = jnp.full_like(w_safe, scle)
  p_n = jnp.zeros_like(w_safe)
  sum_even = jnp.zeros_like(w_safe)
  sum_odd = jnp.zeros_like(w_safe)

  def body(i, state):
    p_curr, p_next, p_n, sum_even, sum_odd = state
    k_int = m_steps - i
    kf = k_int.astype(real_dtype)
    if n > 0:
      p_n = jnp.where(k_int == n, p_curr, p_n)
    term = 2.0 * p_curr
    rem4 = k_int & 3
    sum_even = jnp.where(
        rem4 == 0,
        sum_even + term,
        jnp.where(rem4 == 2, sum_even - term, sum_even),
    )
    sum_odd = jnp.where(
        rem4 == 1,
        sum_odd + term,
        jnp.where(rem4 == 3, sum_odd - term, sum_odd),
    )
    p_prev = (2.0 * kf * p_curr) / w_safe - p_next
    big = jnp.abs(p_prev) > scale_limit
    p_prev = jnp.where(big, p_prev * scale_down, p_prev)
    p_curr = jnp.where(big, p_curr * scale_down, p_curr)
    p_n = jnp.where(big, p_n * scale_down, p_n)
    sum_even = jnp.where(big, sum_even * scale_down, sum_even)
    sum_odd = jnp.where(big, sum_odd * scale_down, sum_odd)
    return p_prev, p_curr, p_n, sum_even, sum_odd

  init = (p_curr, p_next, p_n, sum_even, sum_odd)
  p_0, _, p_n, sum_even, sum_odd = lax.fori_loop(0, m_steps, body, init)
  if n == 0:
    p_n = p_0
  sum_even = sum_even + p_0

  if not is_complex:
    denom = sum_even * jnp.cos(w_safe) + sum_odd * jnp.sin(w_safe)
    return p_n / denom
  else:
    x = lax.real(w_safe)
    y = lax.imag(w_safe)
    i_c = lax.complex(jnp.zeros_like(x), jnp.ones_like(x))
    s_norm = sum_even - i_c * sum_odd
    e_minus_iw = jnp.exp(y) * lax.complex(jnp.cos(x), -jnp.sin(x))
    abs_s = jnp.abs(s_norm)
    inv_abs_s = 1.0 / jnp.where(abs_s == 0.0, 1.0, abs_s)
    conj_unit = jnp.conj(s_norm) * inv_abs_s
    norm = (e_minus_iw * inv_abs_s) * conj_unit
    return p_n * norm


def _jn_recur_from_asym(
    n: int,
    w: Array,
    active_mask: Array,
    r_min: float,
    r_max: float,
    asym_pairs: int,
    cf_pad: int,
) -> Array:
  """Computes J_n(w) for n >= 2 and r_min <= |w| < r_max via (J_0, J_1) + recurrence."""
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  r_raw = jnp.abs(w)
  valid = active_mask & jnp.isfinite(w) & (r_raw >= r_min) & (r_raw <= r_max)
  w_safe = jnp.where(
      valid,
      w,
      jnp.full_like(w, r_min + 1.0),
  )
  r_safe = jnp.abs(w_safe)

  j0, j1 = _jn_asymptotic_j0_j1(w_safe, asym_pairs)

  inv_w = 1.0 / w_safe

  if not is_complex and n <= int(r_min):
    if n <= 8:
      j_km1, j_k = j0, j1
      for k in range(1, n):
        j_km1, j_k = j_k, (2.0 * k * inv_w) * j_k - j_km1
      return j_k
    _, j_n = lax.fori_loop(
        1,
        n,
        lambda k, s: (s[1], (2.0 * k.astype(real_dtype) * inv_w) * s[1] - s[0]),
        (j0, j1),
    )
    return j_n

  # On or near the real axis (Im(w) <= 1), forward recurrence is stable up to
  # the turning point m_turn = min(floor(|w|), n). Away from the real axis
  # (Im(w) > 1), J_k(w) has no zeros (Lommel's theorem) and backward continued
  # fraction down to m_turn = 1 avoids forward-recurrence amplification of Y_k.
  if is_complex:
    use_fwd = jnp.imag(w_safe) <= 1.0
    m_turn = jnp.where(
        use_fwd,
        jnp.clip(jnp.floor(r_safe).astype(np.int32), 1, n),
        np.int32(1),
    )
  else:
    m_turn = jnp.clip(jnp.floor(r_safe).astype(np.int32), 1, n)

  def fwd_body(k, state):
    j_km1, j_k = state
    active = k < m_turn
    kf = k.astype(real_dtype)
    j_kp1 = (2.0 * kf * inv_w) * j_k - j_km1
    j_km1 = jnp.where(active, j_k, j_km1)
    j_k = jnp.where(active, j_kp1, j_k)
    return j_km1, j_k

  j_m_minus_1, j_m = lax.fori_loop(1, n, fwd_body, (j0, j1))

  need_cf = valid & (m_turn < n)
  r_int = jnp.floor(r_safe).astype(np.int32)
  m_top = jnp.maximum(np.int32(n), r_int)
  # Scale continued-fraction padding with the Airy transition width O(M^(1/3))
  # so damping exp((4*sqrt(2)/3) * pad^(3/2) / sqrt(M)) stays below machine eps.
  cf_coeff = 10.0 if real_dtype == np.float64 else 6.0
  dyn_pad = jnp.ceil(cf_coeff * cbrt(m_top.astype(real_dtype))).astype(np.int32)
  pad = jnp.maximum(np.int32(cf_pad), dyn_pad)
  k_top = jnp.where(need_cf, m_top + pad, m_turn)
  k_init = jnp.max(
      jnp.where(need_cf, k_top, np.int32(0)), initial=np.int32(0)
  )

  scale_thresh = 2.0 ** (-500 if real_dtype == np.float64 else -60)
  scale_up = 2.0 ** (500 if real_dtype == np.float64 else 60)
  scale_bits = np.int32(500 if real_dtype == np.float64 else 60)

  def cf_cond(state):
    k, _, _, _ = state
    return jnp.any(need_cf & (k > m_turn))

  def cf_body(state):
    k, r_next, prod, scale_exp = state
    active = need_cf & (k <= k_top) & (k > m_turn)
    kf = k.astype(real_dtype)
    denom = (2.0 * kf) * inv_w - r_next
    r_k = 1.0 / jnp.where(active, denom, jnp.ones_like(denom))
    r_next = jnp.where(active, r_k, r_next)
    in_prod_range = active & (k <= n)
    prod = jnp.where(in_prod_range, prod * r_k, prod)
    small = (
        in_prod_range & (jnp.abs(prod) < scale_thresh) & (jnp.abs(prod) > 0.0)
    )
    prod = jnp.where(small, prod * scale_up, prod)
    scale_exp = jnp.where(small, scale_exp - scale_bits, scale_exp)
    return k - np.int32(1), r_next, prod, scale_exp

  init_cf = (
      k_init,
      jnp.zeros_like(w_safe),
      jnp.ones_like(w_safe),
      jnp.zeros_like(m_turn),
  )
  _, r_m_plus_1, prod, scale_exp = lax.while_loop(cf_cond, cf_body, init_cf)

  mf = m_turn.astype(real_dtype)
  r_m_denom = (2.0 * mf) * inv_w - r_m_plus_1
  r_m = 1.0 / jnp.where(need_cf, r_m_denom, jnp.ones_like(r_m_denom))
  use_m = jnp.abs(j_m) >= jnp.abs(j_m_minus_1)
  base_val = jnp.where(use_m, j_m, j_m_minus_1 * r_m)

  unscaled_res = jnp.where(need_cf, base_val * prod, j_m)
  total_exp = jnp.where(need_cf, scale_exp, np.int32(0)).astype(real_dtype)

  if not is_complex:
    max_e = 1022.0 if real_dtype == np.float64 else 126.0
    e1 = jnp.clip(total_exp, -max_e, max_e)
    rem1 = total_exp - e1
    e2 = jnp.clip(rem1, -max_e, max_e)
    e3 = jnp.clip(rem1 - e2, -max_e, max_e)
    return ((unscaled_res * exp2(e1)) * exp2(e2)) * exp2(e3)
  else:
    # Undo the e^{-y} scaling of (j0, j1) together with 2^{scale_exp} by
    # splitting y = k_y * ln(2) + r_y so large +y and negative scale_exp cancel
    # without intermediate overflow.
    y = lax.imag(w_safe)
    k_y = jnp.floor(y * (1.0 / math.log(2.0)) + 0.5)
    if real_dtype == np.float64:
      ln2_hi = 0.6931471803691238
      ln2_lo = 1.9082149292705877e-10
    else:
      ln2_hi = 0.693115234375
      ln2_lo = 3.19461849453e-5
    r_y = (y - k_y * ln2_hi) - k_y * ln2_lo
    er = jnp.exp(r_y)
    total_exp = total_exp + k_y
    max_e = 1022.0 if real_dtype == np.float64 else 126.0
    e1 = jnp.clip(total_exp, -max_e, max_e)
    rem1 = total_exp - e1
    e2 = jnp.clip(rem1, -max_e, max_e)
    e3 = jnp.clip(rem1 - e2, -max_e, max_e)
    s1, s2, s3 = exp2(e1), exp2(e2), exp2(e3)
    re = (((lax.real(unscaled_res) * er) * s1) * s2) * s3
    im = (((lax.imag(unscaled_res) * er) * s1) * s2) * s3
    return lax.complex(re, im)


def _jn_first_quadrant(n: int, w: Array) -> Array:
  """Evaluates J_n(w) for static integer n >= 0 and w in the first quadrant."""
  real_dtype = jnp.real(w).dtype
  r = jnp.abs(w)

  if real_dtype == np.float64:
    rl = 24.0
    series_iters = 20
    asym_pairs = 10
    base_miller_steps = 64
    cf_pad = 40
  else:
    rl = 10.0
    series_iters = 12
    asym_pairs = 6
    base_miller_steps = 28
    cf_pad = 20

  if n <= 1:
    in_series = r <= 2.0
    in_miller = (r > 2.0) & (r < rl)
    res_ser = _jn_series(n, w, series_iters, 2.0)
    res_mil = _jn_miller(n, w, base_miller_steps, 2.0, rl)
    res_asy = _jn_asymptotic(n, w, asym_pairs, rl)
    return jnp.where(in_series, res_ser, jnp.where(in_miller, res_mil, res_asy))

  # n >= 2:
  r_ser = max(2.0, math.sqrt(2.0 * (n + 1)))
  r_asy = max(r_ser, rl + 0.5 * n * n)

  in_series = r <= r_ser
  res_ser = _jn_series(n, w, series_iters, r_ser)
  res_asy = _jn_asymptotic(n, w, asym_pairs, r_asy)
  res = jnp.where(in_series, res_ser, res_asy)

  if r_ser < rl:
    in_miller = (r > r_ser) & (r < rl)
    miller_steps = max(n + cf_pad, base_miller_steps)
    res_mil = _jn_miller(n, w, miller_steps, r_ser, rl)
    res = jnp.where(in_miller, res_mil, res)

  r_rec_min = max(rl, r_ser)
  if r_rec_min < r_asy:
    in_recur = (r >= r_rec_min) & (r < r_asy)
    res_rec = _jn_recur_from_asym(
        n, w, in_recur, r_rec_min, r_asy, asym_pairs, cf_pad
    )
    res = jnp.where(in_recur, res_rec, res)

  return res


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0,))
def _jn(n: int, z: Array) -> Array:
  with config.numpy_dtype_promotion("standard"):
    return _jn_impl(n, z)


def _jn_impl(n: int, z: Array) -> Array:
  is_complex = dtypes.issubdtype(z.dtype, np.complexfloating)
  n_abs = abs(n)
  is_odd = (n_abs & 1) == 1
  neg_n = n < 0 and is_odd

  if not is_complex:
    w = jnp.abs(z)
    res = _jn_first_quadrant(n_abs, w)
    if neg_n:
      res = -res
    if is_odd:
      res = jnp.where(z < 0.0, -res, res)

    # Special cases for real z:
    if n_abs == 0:
      z_zero_val = jnp.ones_like(z)
    elif is_odd:
      z_zero_val = copysign(0.0, -z if neg_n else z)
    else:
      z_zero_val = jnp.zeros_like(z)
    res = jnp.where(z == 0.0, z_zero_val, res)
    res = jnp.where(jnp.isinf(z), 0.0, res)
    if is_odd:
      res = jnp.where(
          (res == 0.0) & (jnp.abs(z) < 1.0),
          copysign(0.0, -z if neg_n else z),
          res,
      )
    res = jnp.where(jnp.isnan(z), np.nan, res)
    return res
  else:
    x = lax.real(z)
    y = lax.imag(z)
    w = lax.complex(jnp.abs(x), jnp.abs(y))
    res = _jn_first_quadrant(n_abs, w)
    if neg_n:
      res = -res

    # Map from first quadrant w = |x| + i|y| back to z = x + i*y:
    # 1. If x < 0: J_n(-|x| + i|y|) = (-1)^n * conj(J_n(w))
    res_neg_x = -jnp.conj(res) if is_odd else jnp.conj(res)
    res = jnp.where(x < 0.0, res_neg_x, res)
    # 2. If y < 0: J_n(x - i|y|) = conj(J_n(x + i|y|))
    res = jnp.where(y < 0.0, jnp.conj(res), res)
    # 3. On the imaginary axis (x == 0), J_n(i*y) = i^n * I_n(y) is purely
    # imaginary for odd n and purely real for even n; on the real axis
    # (y == 0), J_n(x) is purely real.
    if is_odd:
      res = jnp.where(
          x == 0.0, lax.complex(jnp.zeros_like(x), lax.imag(res)), res
      )
    else:
      res = jnp.where(
          x == 0.0, lax.complex(lax.real(res), jnp.zeros_like(y)), res
      )
    res = jnp.where(
        y == 0.0, lax.complex(lax.real(res), jnp.zeros_like(y)), res
    )

    nan_c = lax.complex(jnp.full_like(x, np.nan), jnp.full_like(y, np.nan))
    zero_c = jnp.zeros_like(z)
    one_c = jnp.ones_like(z)
    res = jnp.where(z == 0, one_c if n_abs == 0 else zero_c, res)
    res = jnp.where(jnp.isinf(x), zero_c, res)
    res = jnp.where(jnp.isnan(x) | jnp.isnan(y) | jnp.isinf(y), nan_c, res)
    return res


@_jn.defjvp
def _jn_jvp(
    n: int, primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
  (z,) = primals
  (z_dot,) = tangents
  y = _jn(n, z)
  if n == 0:
    dy_dz = -_jn(1, z)
  else:
    dy_dz = 0.5 * (_jn(n - 1, z) - _jn(n + 1, z))
  return y, dy_dz * z_dot


def jn(n: int, z: ArrayLike) -> Array:
  r"""Bessel function of the first kind of integer order `n`.

  JAX implementation of :obj:`scipy.special.jn`.

  .. math::

     J_n(z) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k!\,(n + k)!}
     \left(\frac{z}{2}\right)^{2k + n}

  for integer order :math:`n` (with :math:`J_{-n}(z) = (-1)^n J_n(z)`).

  Args:
    n: The integer order of the Bessel function (must be a static integer).
    z: The argument(s) at which to evaluate :math:`J_n(z)`. Real and complex
      inputs are supported.

  Returns:
    An array containing the evaluated Bessel function values.

  Notes:
    Unlike :obj:`scipy.special.jn`, ``jax.scipy.special.jn`` requires ``n`` to
    be a concrete integer order (so ``n`` cannot be traced or batched with
    :func:`jax.vmap`).

  See Also:
    - :func:`jax.scipy.special.bessel_jn`
    - :func:`jax.scipy.special.i0`
    - :func:`jax.scipy.special.i1`
  """
  n = core.concrete_or_error(
      operator.index,
      n,
      "The 'n' argument of jax.scipy.special.jn must be a static integer.",
  )
  (z,) = promote_args_inexact("jn", z)
  orig_dtype = z.dtype
  if orig_dtype in (dtypes.bfloat16, np.float16):
    z = z.astype(np.float32)
    return _jn(n, z).astype(orig_dtype)
  return _jn(n, z)
