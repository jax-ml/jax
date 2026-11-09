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
from jax._src.lax.double_word import (
    f2_add,
    f2_div,
    f2_mul,
    f2_recip,
    f2_square,
    fast_two_sum,
    scale_f2_pow2,
    split_half,
    split_half_mask,
    two_prod,
    two_sum,
)
from jax._src.lax.lax import (
    _const,
    bitcast_convert_type,
    shift_left,
    shift_right_arithmetic,
)
from jax._src.numpy.ufuncs import cbrt, copysign
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike

_INV_SQRT_2PI_F32 = (0.3989422917366028, -1.1335170135851058e-08)
_INV_SQRT_2PI_F64 = (0.3989422804014327, -2.49232720227773e-17)
# 12 + 7 + 24 bit split of ln(2) for float32 so kf * hi and kf * mid are exact
# single-word products for all |kf| <= 4096 (|w| <= 2839).
_LN2_SPLIT_12_F32 = (
    0.693115234375,
    3.1948089599609375e-05,
    -1.9046543252431068e-09,
)
# 26 + 26 + 53 bit split of ln(2) for float64 so kf * hi and kf * mid are exact
# single-word products for all |kf| <= 2**27 (|w| <= 9.3e7).
_LN2_SPLIT_26_F64 = (
    0.6931471675634384,
    1.2996506759677118e-08,
    1.3421277060097865e-16,
)
_EXP_R_POLY_F32 = tuple(1.0 / math.factorial(m) for m in range(2, 9))
_EXP_R_POLY_F64 = tuple(1.0 / math.factorial(m) for m in range(2, 15))


def _pow2_int(e: Array) -> Array:
  """Computes 2**e for integer float Array e via IEEE 754 exponent bits."""
  dtype = e.dtype
  if dtype == np.float64:
    int_ty, mant_bits, bias, emin, emax = np.int64, 52, 1023, -1022, 1023
  else:
    int_ty, mant_bits, bias, emin, emax = np.int32, 23, 127, -126, 127
  e_clamped = lax.clamp(
      lax.full_like(e, float(emin - 1)), e, lax.full_like(e, float(emax + 1))
  )
  ei = lax.convert_element_type(e_clamped, int_ty)
  bits = shift_left(
      ei + lax.full_like(ei, bias), lax.full_like(ei, mant_bits)
  )
  return bitcast_convert_type(bits, dtype)


def _f2_frexp_sqrt2(a0: Array, a1: Array) -> tuple[Array, Array, Array]:
  """Normalizes positive (a0, a1) into a0 * 2**(-e) in [sqrt(2)/2, sqrt(2))."""
  dtype = a0.dtype
  if dtype == np.float64:
    int_ty, mant_bits, bias = np.int64, 52, 1023
    sqrt2_bits = 0x3FE6A09E667F3BCD
  else:
    int_ty, mant_bits, bias = np.int32, 23, 127
    sqrt2_bits = 0x3F3504F3
  bits = bitcast_convert_type(a0, int_ty)
  s_mant = lax.full_like(bits, mant_bits)
  e = shift_right_arithmetic(bits - _const(bits, sqrt2_bits), s_mant)
  e_clamped = lax.clamp(
      lax.full_like(e, -(bias - 1)), e, lax.full_like(e, bias)
  )
  scale = bitcast_convert_type(
      shift_left(lax.full_like(bits, bias) - e_clamped, s_mant), dtype
  )
  return a0 * scale, a1 * scale, e_clamped


def _f2_ipow_mant_exp(w_pos: Array, p: int) -> tuple[Array, Array, Array]:
  """Computes w_pos**p as (m0, m1) * 2**e with m0 in [sqrt(2)/2, sqrt(2))."""
  dtype = w_pos.dtype
  if dtype == np.float64:
    int_ty, mant_bits = np.int64, 52
    sqrt2_bits = 0x3FE6A09E667F3BCD
  else:
    int_ty, mant_bits = np.int32, 23
    sqrt2_bits = 0x3F3504F3
  bits = bitcast_convert_type(w_pos, int_ty)
  s_mant = lax.full_like(bits, mant_bits)
  eb = shift_right_arithmetic(bits - _const(bits, sqrt2_bits), s_mant)
  b0 = bitcast_convert_type(bits - shift_left(eb, s_mant), dtype)
  b1 = lax.full_like(w_pos, 0.0)
  acc: tuple[Array, Array, Array] | None = None
  while p > 0:
    if p & 1:
      if acc is None:
        acc = (b0, b1, eb)
      else:
        a0, a1 = f2_mul(acc[0], acc[1], b0, b1)
        a0, a1, ea = _f2_frexp_sqrt2(a0, a1)
        acc = (a0, a1, acc[2] + eb + ea)
    p >>= 1
    if p > 0:
      b0, b1 = f2_square(b0, b1)
      b0, b1, ed = _f2_frexp_sqrt2(b0, b1)
      eb = eb + eb + ed
  assert acc is not None
  return acc


def _scale_by_w_half_pow_n_over_fact_f2(
    n: int, w_safe: Array, t0: Array, t1: Array
) -> Array:
  """Computes (t0 + t1) * (w_safe / 2)^n / n! for real w_safe >= 0."""
  if n == 0:
    return t0 + t1
  zero = lax.full_like(w_safe, 0.0)
  if n == 1:
    p0, p1 = f2_mul(w_safe, zero, 0.5 * t0, 0.5 * t1)
    return p0 + p1
  dtype = w_safe.dtype
  emin = -1022 if dtype == np.float64 else -126
  fact = math.factorial(n)
  k_exp = fact.bit_length()
  c_frac = fractions.Fraction(1 << k_exp, fact)
  if dtype == np.float32:
    c_hi = float(np.float32(float(c_frac)))
    c_lo = float(np.float32(float(c_frac - fractions.Fraction(c_hi))))
  else:
    c_hi = float(c_frac)
    c_lo = float(c_frac - fractions.Fraction(c_hi))
  w_pos = jnp.where(w_safe > 0.0, w_safe, jnp.ones_like(w_safe))
  p0, p1, ep = _f2_ipow_mant_exp(w_pos, n)
  s0, s1 = f2_mul(t0, t1, _const(w_safe, c_hi), _const(w_safe, c_lo))
  v0, v1 = f2_mul(p0, p1, s0, s1)
  k_tot = ep - _const(ep, n + k_exp)
  res = scale_f2_pow2(v0, v1, k_tot)
  return jnp.where((w_safe <= 0.0) | (k_tot < 2 * emin), 0.0, res)


def _scale_f2_exp_w(
    val0: Array,
    val1: Array | None,
    w_safe: Array,
    scale_exp: Array | None = None,
) -> Array:
  """Multiplies (val0, val1) by exp(w_safe) * 2**scale_exp without overflow.

  Args:
    val0: High-order word of the multiplicand.
    val1: Low-order word of the multiplicand, or None if the multiplicand is a
      single-word floating-point value.
    w_safe: Real exponential argument.
    scale_exp: Additional power-of-two exponent scaling factor, or None to apply
      no extra exponent scaling.

  Returns:
    The scaled result in standard floating-point precision.
  """
  real_dtype = w_safe.dtype
  int_ty = np.int64 if real_dtype == np.float64 else np.int32
  emin = -1022 if real_dtype == np.float64 else -126
  if real_dtype == np.float64:
    ln2_hi, ln2_mid, ln2_lo = _LN2_SPLIT_26_F64
    exp_coeffs = _EXP_R_POLY_F64
  else:
    ln2_hi, ln2_mid, ln2_lo = _LN2_SPLIT_12_F32
    exp_coeffs = _EXP_R_POLY_F32
  # Split w_safe = kf * ln(2) + r with integer kf and |r| <= ln(2) / 2 so
  # exp(w_safe) = 2**kf * exp(r). We combine 2**kf with 2**scale_exp at the end
  # via scale_f2_pow2 so that large w_safe values where exp(w_safe) alone would
  # overflow can still cancel against small (val0, val1) or negative scale_exp.
  kf = lax.round(
      w_safe * _const(w_safe, 1.4426950408889634),
      lax.RoundingMethod.TO_NEAREST_EVEN,
  )
  # Subtract kf * ln(2) using a three-part split of ln(2) so the remainder
  # r + r_lo retains full double-word precision even when kf is large.
  r_hi = w_safe - kf * _const(w_safe, ln2_hi)
  kl_mid = kf * _const(w_safe, ln2_mid)
  r, e1 = two_sum(r_hi, -kl_mid)
  r, r_lo = fast_two_sum(r, e1 - kf * _const(w_safe, ln2_lo))
  # Approximate exp(r + r_lo) in double-word precision as eh + el via a Taylor
  # polynomial on the narrow interval |r| <= ln(2) / 2.
  one = lax.full_like(w_safe, 1.0)
  p = lax.polynomial(r, exp_coeffs, unroll=len(exp_coeffs))
  q = (r * r) * p + r_lo * (one + r)
  s0, s1 = fast_two_sum(one, r)
  eh, el = fast_two_sum(s0, s1 + q)
  # Multiply (val0, val1) by (eh, el) and apply the combined power of two.
  p0, p1 = two_prod(val0, eh)
  m1 = p1 + val0 * el if val1 is None else p1 + (val0 * el + val1 * eh)
  k_tot = lax.convert_element_type(kf, int_ty)
  if scale_exp is None:
    return scale_f2_pow2(p0, m1, k_tot)
  k_tot = k_tot + lax.convert_element_type(scale_exp, int_ty)
  res = scale_f2_pow2(p0, m1, k_tot)
  return jnp.where(k_tot < 2 * emin, 0.0, res)


def _scale_by_w_half_pow_n_over_fact(
    n: int, w_safe: Array, val: Array
) -> Array:
  """Computes val * (w_safe / 2)^n / n! for integer n >= 0 without overflow."""
  if n == 0:
    return val
  if n == 1:
    return w_safe * (0.5 * val)
  # Evaluating (w / 2)^n and n! separately overflows float range for large n.
  # Instead, factor n! = 2^k_exp / c_n (with c_n in (1, 2]) and absorb the
  # powers of 1/2 into each factor of w before exponentiating. Power-of-2
  # scalings only shift the floating-point exponent, so they introduce no
  # rounding error.
  fact = math.factorial(n)
  k_exp = fact.bit_length()
  c_n = float(fractions.Fraction(1 << k_exp, fact))
  q, rem = divmod(k_exp, n)
  w1 = w_safe * float(0.5 ** (q + 1))
  scaled_val = val * c_n
  if rem == 0:
    n1 = n // 2
    n2 = n - n1
    return ((w1**n1) * scaled_val) * (w1**n2)
  w2 = 0.5 * w1
  return ((w1 ** (n - rem)) * scaled_val) * (w2**rem)


def _scale_exp2(
    val: Array, total_exp: Array, er: Array | None = None
) -> Array:
  """Multiplies val by (er if provided) * 2**total_exp without overflow."""
  # Split the exponent in half so that large shifts (e.g. when undoing loop
  # rescaling on tiny values) don't overflow a single 2**total_exp factor.
  e1 = jnp.floor(0.5 * total_exp)
  e2 = total_exp - e1
  s1, s2 = _pow2_int(e1), _pow2_int(e2)
  if not dtypes.issubdtype(val.dtype, np.complexfloating):
    res = val if er is None else val * er
    return (res * s1) * s2
  re = lax.real(val) if er is None else lax.real(val) * er
  im = lax.imag(val) if er is None else lax.imag(val) * er
  # Guard exact zeros so that 0.0 * inf stays 0.0 instead of turning into NaN.
  re = jnp.where(re == 0.0, 0.0, (re * s1) * s2)
  im = jnp.where(im == 0.0, 0.0, (im * s1) * s2)
  return lax.complex(re, im)


def _scale_exp_y_and_exp2(
    val: Array, y: Array, scale_exp: Array, sign_y: float
) -> Array:
  """Multiplies complex val by exp(sign_y * y) * 2**scale_exp."""
  real_dtype = y.dtype
  # Rewrite y = k_y * ln(2) + r_y with |r_y| <= ln(2)/2 so exp(sign_y * y)
  # splits into a bounded factor exp(sign_y * r_y) and a power of 2 that merges
  # directly into scale_exp. Subtracting ln(2) in high + low parts keeps r_y
  # accurate when y is large.
  k_y = jnp.floor(y * (1.0 / math.log(2.0)) + 0.5)
  if real_dtype == np.float64:
    ln2_hi = 0.6931471803691238
    ln2_lo = 1.9082149292705877e-10
  else:
    ln2_hi = 0.693115234375
    ln2_lo = 3.19461849453e-5
  r_y = (y - k_y * ln2_hi) - k_y * ln2_lo
  er = jnp.exp(sign_y * r_y)
  total_exp = scale_exp.astype(real_dtype) + sign_y * k_y
  return _scale_exp2(val, total_exp, er=er)


def _miller_norm(w_safe: Array, sum_even: Array, sum_odd: Array) -> Array:
  """Recovers the scale factor that normalizes backward-recurrence values."""
  # For real w, sum_even * cos(w) + sum_odd * sin(w) equals 1 / norm by
  # standard trigonometric identities for J_k(w).
  if not dtypes.issubdtype(w_safe.dtype, np.complexfloating):
    return 1.0 / (sum_even * jnp.cos(w_safe) + sum_odd * jnp.sin(w_safe))
  # For complex w, cos(w) and sin(w) both grow like exp(Im(w)) / 2, which would
  # cancel against exp(-Im(w)) in the sum if combined naively. Using the single
  # exponential exp(-i * w) = sum_even - i * sum_odd avoids that cancellation,
  # and dividing via conj(s) / |s| avoids overflowing |s|^2.
  x = lax.real(w_safe)
  y = lax.imag(w_safe)
  i_c = lax.complex(jnp.zeros_like(x), jnp.ones_like(x))
  s_norm = sum_even - i_c * sum_odd
  e_minus_iw = jnp.exp(y) * lax.complex(jnp.cos(x), -jnp.sin(x))
  abs_s = jnp.abs(s_norm)
  inv_abs_s = 1.0 / jnp.where(abs_s == 0.0, 1.0, abs_s)
  conj_unit = jnp.conj(s_norm) * inv_abs_s
  return (e_minus_iw * inv_abs_s) * conj_unit


def _forward_recur_rescaled(
    n: int, inv_w: Array, f0: Array, f1: Array
) -> tuple[Array, Array]:
  """Steps f_{k+1} = (2k / w) * f_k - f_{k-1} up to order n without overflow."""
  real_dtype = jnp.real(inv_w).dtype
  scale_thresh = 2.0 ** (500 if real_dtype == np.float64 else 60)
  scale_step = 2.0 ** (-500 if real_dtype == np.float64 else -60)
  scale_bits = np.int32(500 if real_dtype == np.float64 else 60)

  # When k > |w|, values can grow rapidly with k. Whenever a step approaches
  # overflow, divide the running pair by 2**scale_bits and record the exponent
  # shift in scale_exp so the caller can restore the true magnitude at the end.
  def fwd_body(k, state):
    f_km1, f_k, scale_exp = state
    kf = k.astype(real_dtype)
    f_kp1 = (2.0 * kf * inv_w) * f_k - f_km1
    big = jnp.abs(f_kp1) > scale_thresh
    f_kp1 = jnp.where(big, f_kp1 * scale_step, f_kp1)
    f_k = jnp.where(big, f_k * scale_step, f_k)
    scale_exp = jnp.where(big, scale_exp + scale_bits, scale_exp)
    return f_k, f_kp1, scale_exp

  _, f_n, scale_exp = lax.fori_loop(
      1,
      n,
      fwd_body,
      (f0, f1, jnp.zeros_like(jnp.real(inv_w), dtype=np.int32)),
  )
  return f_n, scale_exp


def _jn_series(
    n: int, w: Array, max_iter: int, r_max: float, *, modified: bool = False
) -> Array:
  """Ascending power series for J_n(w) or I_n(w) for integer n >= 0."""
  real_dtype = jnp.real(w).dtype
  w_safe = jnp.where(jnp.isfinite(w) & (jnp.abs(w) <= r_max), w, 0.0)
  coeff = 0.25 if modified else -0.25
  one = jnp.ones_like(w_safe)
  zero = jnp.zeros_like(w)

  if not dtypes.issubdtype(w.dtype, np.complexfloating):
    # Evaluate 1 + sum_{k>=1} (coeff * w^2)^k / (k! * (n + 1)...(n + k))
    # from the inside out (Horner's method) in double-word arithmetic.
    # Pre-split w_safe / 2 via bitmask truncation so wh * wh (<= 52 bits in f64,
    # 24 bits in f32) and (wh + wh) * wl (<= 53 bits in f64, 24 bits in f32) are
    # exact IEEE 754 products; this prevents XLA/LLVM from contracting
    # ah * ah - w_safe * w_safe in two_prod across separate pre-loop fusions into
    # an FMA that zeroes out the low word z2_1 on targets with hardware FMA.
    wh, wl = split_half_mask(0.5 * w_safe)
    s0, s1 = fast_two_sum(wh * wh, (wh + wh) * wl)
    z2_0, z2_1 = fast_two_sum(s0, s1 + wl * wl)
    if not modified:
      z2_0, z2_1 = -z2_0, -z2_1

    def body_f2(i, state):
      t0, t1 = state
      kf = lax.convert_element_type(max_iter - i, real_dtype)
      p0, p1 = f2_mul(t0, t1, z2_0, z2_1)
      q0, q1 = f2_div(p0, p1, kf * (_const(kf, n) + kf), zero)
      return f2_add(one, zero, q0, q1)

    t0, t1 = lax.fori_loop(0, max_iter, body_f2, (one, zero))
    res = _scale_by_w_half_pow_n_over_fact_f2(n, w_safe, t0, t1)
    return jnp.where(w == 0, one if n == 0 else zero, res)

  z2 = coeff * (w_safe * w_safe)

  def body(i, total):
    kf = lax.convert_element_type(max_iter - i, real_dtype)
    return one + total * (z2 / (kf * (n + kf)))

  total = lax.fori_loop(0, max_iter, body, one)
  res = _scale_by_w_half_pow_n_over_fact(n, w_safe, total)
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
  r = jnp.abs(w)
  w_default = 0.5 * (r_min + rl)
  w_safe = jnp.where(
      jnp.isfinite(w) & (r >= r_min) & (r <= rl),
      w,
      jnp.full_like(w, w_default),
  )

  scle = 2.0 ** (-500 if real_dtype == np.float64 else -60)
  scale_limit = 2.0 ** (500 if real_dtype == np.float64 else 60)
  scale_down = 2.0 ** (-500 if real_dtype == np.float64 else -60)

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
  return p_n * _miller_norm(w_safe, sum_even, sum_odd)


def _jn_recur_from_asym(
    n: int,
    w: Array,
    active_mask: Array,
    r_min: float,
    r_max: float,
    asym_pairs: int,
    cf_pad: int,
    j0_j1: tuple[Array, Array] | None = None,
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

  if j0_j1 is None:
    j0, j1 = _jn_asymptotic_j0_j1(w_safe, asym_pairs)
  else:
    j0, j1 = j0_j1

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
  total_exp = jnp.where(need_cf, scale_exp, np.int32(0))

  if not is_complex:
    return _scale_exp2(unscaled_res, total_exp.astype(real_dtype))
  else:
    return _scale_exp_y_and_exp2(
        unscaled_res, lax.imag(w_safe), total_exp, 1.0
    )


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
    - :func:`jax.scipy.special.yn`
    - :func:`jax.scipy.special.iv`
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


def _yn_series(
    n: int, w: Array, max_iter: int, r_max: float
) -> tuple[Array, Array]:
  """Power series around w = 0 for (Y_n(w), J_n(w)) in the first quadrant.

  The second tuple element J_n(w) is populated for complex w and zeros for real w.
  """
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  r = jnp.abs(w)
  w_safe = jnp.where(
      jnp.isfinite(w) & (r > 0.0) & (r <= r_max), w, jnp.ones_like(w)
  )
  z2_neg = -0.25 * (w_safe * w_safe)

  # Near w = 0, Y_n(w) splits into a smooth series (multiplying J_n(w) by
  # log(w/2) plus harmonic weights 1 + 1/2 + ... + 1/k) and, for n >= 1, a
  # finite sum of negative powers (w/2)^(2k - n) that diverges as w -> 0.
  # The loop below computes both the J_n series (sum_t) and the weighted series
  # (sum_ht) in one pass.
  h_n = math.fsum(1.0 / j for j in range(1, n + 1))

  def reg_body(k, state):
    term, sum_t, sum_ht, h_diff = state
    kf = k.astype(real_dtype)
    term = term * (z2_neg / (kf * (n + kf)))
    h_diff = h_diff + 0.5 * (1.0 / kf + 1.0 / (n + kf))
    sum_t = sum_t + term
    sum_ht = sum_ht + h_diff * term
    return term, sum_t, sum_ht, h_diff

  one_w = jnp.ones_like(w_safe)
  zero_w = jnp.zeros_like(w_safe)
  zero_r = jnp.zeros_like(jnp.real(w_safe))
  _, sum_t, sum_ht, _ = lax.fori_loop(
      1, max_iter + 1, reg_body, (one_w, one_w, zero_w, zero_r)
  )

  log_w_over_2 = jnp.log(w_safe) - math.log(2.0)
  reg_bracket = (log_w_over_2 + (np.euler_gamma - 0.5 * h_n)) * sum_t - sum_ht

  reg_part = _scale_by_w_half_pow_n_over_fact(
      n, w_safe, (2.0 / math.pi) * reg_bracket
  )
  j_part = (
      _scale_by_w_half_pow_n_over_fact(n, w_safe, sum_t)
      if is_complex
      else zero_w
  )

  if n == 0:
    res = reg_part
  else:
    # Finite sum of negative powers: -((n - 1)! / pi) * (w / 2)^(-n) * sum_u.
    # For large n, early terms dominate while the final terms are below machine
    # precision for |w| <= r_max, so we cap the loop at s1_iters steps.
    z2_pos = -z2_neg
    s1_iters = min(n - 1, max(30 if real_dtype == np.float64 else 24, max_iter))

    def s1_body(k, state):
      u, sum_u = state
      kf = k.astype(real_dtype)
      u = u * (z2_pos / (kf * (n - kf)))
      sum_u = sum_u + u
      return u, sum_u

    _, sum_u = lax.fori_loop(1, s1_iters + 1, s1_body, (one_w, one_w))

    # Factor (n - 1)! = d_n * pi * 2^m_exp and distribute the 2^m_exp across
    # the n factors of 2 / w so that neither (n - 1)! nor (2 / w)^n overflows
    # prematurely when n is large.
    fact_m1 = math.factorial(n - 1)
    m_exp = fact_m1.bit_length()
    d_n = float(fractions.Fraction(fact_m1, 1 << m_exp)) / math.pi
    q_u, rem_u = divmod(m_exp, n)
    if is_complex:
      # Separate unit direction conj(w)/|w| from magnitude 1/|w| and scale the
      # real and imaginary parts separately. This way, when |w| is tiny and
      # (2 / |w|)^n overflows to +inf, an exact zero component stays 0.0
      # instead of producing 0.0 * inf = NaN.
      abs_w = jnp.abs(w_safe)
      unit_conj = jnp.conj(w_safe) / abs_w
      unit_conj_n = unit_conj if n == 1 else unit_conj**n
      phase_u = (unit_conj_n * sum_u) * (-d_n)
      u1_r = float(2.0 ** (q_u + 1)) / abs_w
      if n == 1:
        r1 = u1_r
        r2 = 1.0
      elif rem_u == 0:
        n1 = n // 2
        n2 = n - n1
        r1 = u1_r**n1
        r2 = u1_r**n2
      else:
        u2_r = 2.0 * u1_r
        r1 = u1_r ** (n - rem_u)
        r2 = u2_r**rem_u
      re_p = lax.real(phase_u)
      im_p = lax.imag(phase_u)
      re = jnp.where(re_p == 0.0, 0.0, (re_p * r1) * r2)
      im = jnp.where(im_p == 0.0, 0.0, (im_p * r1) * r2)
      s1 = lax.complex(re, im)
    else:
      u1 = float(2.0 ** (q_u + 1)) / w_safe
      scaled_u = -sum_u * d_n
      if n == 1:
        s1 = u1 * scaled_u
      elif rem_u == 0:
        n1 = n // 2
        n2 = n - n1
        s1 = ((u1**n1) * scaled_u) * (u1**n2)
      else:
        u2 = 2.0 * u1
        s1 = ((u1 ** (n - rem_u)) * scaled_u) * (u2**rem_u)
    res = s1 + reg_part

  neg_inf = (
      lax.complex(
          jnp.full_like(jnp.real(w), -np.inf), jnp.zeros_like(jnp.imag(w))
      )
      if is_complex
      else jnp.full_like(w, -np.inf)
  )
  y_res = jnp.where(w == 0, neg_inf, res)
  j_res = (
      jnp.where(w == 0, one_w if n == 0 else zero_w, j_part)
      if is_complex
      else zero_w
  )
  return y_res, j_res


def _hankel_yj_unscaled_complex(
    n: int,
    x: Array,
    p: Array,
    q: Array,
    c_chi: Array,
    s_chi: Array,
    half_amp: Array,
    exp_neg2y: Array,
) -> tuple[Array, Array]:
  """Computes exp(-y) * (Y_n(w), J_n(w)) from large-|w| series terms (P, Q)."""
  i_c = lax.complex(jnp.zeros_like(c_chi), jnp.ones_like(c_chi))
  e_minus = lax.complex(c_chi, -s_chi)
  e_plus = lax.complex(c_chi, s_chi)
  # Split J_n and Y_n into the two complex-exponential waves exp(-i*chi) and
  # exp(+i*chi), with exp(y) factored out so large Im(w) = y cannot overflow
  # intermediate products. The second wave is smaller by a factor of exp(-2*y).
  h2 = half_amp * e_minus * (p - i_c * q)
  h1 = half_amp * e_plus * (p + i_c * q)
  # On the positive imaginary axis (x == 0), h2 and h1 are purely real or
  # purely imaginary depending on whether n is even or odd; zero out rounding
  # noise in the vanishing component before combining them.
  if (n & 1) == 0:
    h2 = jnp.where(
        x == 0.0, lax.complex(lax.real(h2), jnp.zeros_like(x)), h2
    )
    h1 = jnp.where(
        x == 0.0, lax.complex(jnp.zeros_like(x), lax.imag(h1)), h1
    )
  else:
    h2 = jnp.where(
        x == 0.0, lax.complex(jnp.zeros_like(x), lax.imag(h2)), h2
    )
    h1 = jnp.where(
        x == 0.0, lax.complex(lax.real(h1), jnp.zeros_like(x)), h1
    )
  i_h2 = lax.complex(-lax.imag(h2), lax.real(h2))
  i_h1 = lax.complex(-lax.imag(h1), lax.real(h1))
  # Right on the positive imaginary axis (x == 0), the large-|w| series for h2
  # gives exp(-y) * J_n(i*y) directly without its tiny exp(-2*y) tail, so h1
  # supplies the full decaying part of Y_n(i*y) (doubling its weight) and is
  # omitted from J_n(i*y).
  y_h1_scale = jnp.where(x == 0.0, 2.0 * exp_neg2y, exp_neg2y)
  j_h1_scale = jnp.where(x == 0.0, 0.0, exp_neg2y)
  y_unscaled = i_h2 - y_h1_scale * i_h1
  j_unscaled = h2 + j_h1_scale * h1
  return y_unscaled, j_unscaled


def _yn_asymptotic(
    n: int, w: Array, max_pairs: int, r_min: float
) -> tuple[Array, Array]:
  """Large-|w| expansion for (Y_n(w), J_n(w)) in powers of 1 / w.

  The second tuple element J_n(w) is populated for complex w and zeros for real w.
  """
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  w_safe = jnp.where(
      jnp.isfinite(w) & (jnp.abs(w) >= r_min), w, jnp.full_like(w, r_min)
  )
  # For large |w|, Y_n(w) behaves like sqrt(2 / (pi * w)) times a phase-shifted
  # sinusoid modulated by slowly varying series (p, q) in powers of 1 / w.
  p, q = _hankel_pq(n, w_safe, max_pairs)
  x = jnp.real(w_safe)
  c_chi, s_chi = _hankel_sincos_chi_real(n, jnp.maximum(x, 0.0))

  if not is_complex:
    amp = math.sqrt(2.0 / math.pi) / jnp.sqrt(x)
    return amp * (p * s_chi + q * c_chi), jnp.zeros_like(w_safe)
  else:
    y = jnp.imag(w_safe)
    half_amp = (1.0 / math.sqrt(2.0 * math.pi)) / jnp.sqrt(w_safe)
    exp_neg2y = jnp.exp(-2.0 * y)
    y_unscaled, j_unscaled = _hankel_yj_unscaled_complex(
        n, x, p, q, c_chi, s_chi, half_amp, exp_neg2y
    )
    # Restore the factored-out exp(y) in two half-steps exp(y/2) * exp(y/2) so
    # that y near log(fmax) does not overflow exp(y) when the 1/sqrt(|w|)
    # factor keeps the final product within float range.
    max_log = 700.0 if real_dtype == np.float64 else 85.0
    ey_half = jnp.exp(jnp.clip(0.5 * y, 0.0, max_log))
    y_re = (lax.real(y_unscaled) * ey_half) * ey_half
    y_im = (lax.imag(y_unscaled) * ey_half) * ey_half
    j_re = (lax.real(j_unscaled) * ey_half) * ey_half
    j_im = (lax.imag(j_unscaled) * ey_half) * ey_half
    return lax.complex(y_re, y_im), lax.complex(j_re, j_im)


def _yn_miller(
    n: int, w: Array, m_steps: int, r_min: float, rl: float
) -> tuple[Array, Array]:
  """Computes (Y_n(w), J_n(w)) for moderate |w| via downward recurrence.

  The second tuple element J_n(w) is populated for complex w and zeros for real w.
  """
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  r = jnp.abs(w)
  w_default = 0.5 * (r_min + rl)
  w_safe = jnp.where(
      jnp.isfinite(w) & (r >= r_min) & (r <= rl),
      w,
      jnp.full_like(w, w_default),
  )
  inv_w = 1.0 / w_safe

  scle = 2.0 ** (-500 if real_dtype == np.float64 else -60)
  scale_limit = 2.0 ** (500 if real_dtype == np.float64 else 60)
  scale_down = 2.0 ** (-500 if real_dtype == np.float64 else -60)

  # Step p_{k-1} = (2k / w) * p_k - p_{k+1} downward from k = m_steps with a
  # tiny seed. Downward steps rapidly lock onto the true ratios J_k(w) / J_0(w)
  # up to a single shared scale factor. In the same pass we accumulate:
  #   - sum_even, sum_odd: alternating sums of even/odd p_k used to find the
  #     shared scale factor (norm) at the end;
  #   - y0_sum, y1_sum: weighted sums of even/odd p_k that give the series part
  #     of Y_0(w) and Y_1(w) once multiplied by norm.
  p_next = jnp.zeros_like(w_safe)
  p_curr = jnp.full_like(w_safe, scle)
  p_n = jnp.zeros_like(w_safe)
  sum_even = jnp.zeros_like(w_safe)
  sum_odd = jnp.zeros_like(w_safe)
  y0_sum = jnp.zeros_like(w_safe)
  y1_sum = jnp.zeros_like(w_safe)

  def body(i, state):
    p_curr, p_next, p_n, sum_even, sum_odd, y0_sum, y1_sum = state
    k_int = m_steps - i
    kf = k_int.astype(real_dtype)
    if is_complex and n >= 2:
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
    y0_term = p_curr / kf
    y0_sum = jnp.where(
        rem4 == 2,
        y0_sum + y0_term,
        jnp.where(rem4 == 0, y0_sum - y0_term, y0_sum),
    )
    if n >= 1:
      kf_odd_denom = jnp.where(k_int >= 3, kf * kf - 1.0, 1.0)
      y1_term = jnp.where(
          k_int >= 3, p_curr * (kf / kf_odd_denom), jnp.zeros_like(p_curr)
      )
      y1_sum = jnp.where(
          rem4 == 3,
          y1_sum + y1_term,
          jnp.where(rem4 == 1, y1_sum - y1_term, y1_sum),
      )

    p_prev = (2.0 * kf * inv_w) * p_curr - p_next
    # If p_prev grows near overflow, rescale every running quantity by the same
    # power of 2; because everything is linear in the seed, ratios are unchanged.
    big = jnp.abs(p_prev) > scale_limit
    p_prev = jnp.where(big, p_prev * scale_down, p_prev)
    p_curr = jnp.where(big, p_curr * scale_down, p_curr)
    if is_complex and n >= 2:
      p_n = jnp.where(big, p_n * scale_down, p_n)
    sum_even = jnp.where(big, sum_even * scale_down, sum_even)
    sum_odd = jnp.where(big, sum_odd * scale_down, sum_odd)
    y0_sum = jnp.where(big, y0_sum * scale_down, y0_sum)
    y1_sum = jnp.where(big, y1_sum * scale_down, y1_sum)
    return p_prev, p_curr, p_n, sum_even, sum_odd, y0_sum, y1_sum

  init = (p_curr, p_next, p_n, sum_even, sum_odd, y0_sum, y1_sum)
  p_0, p_1, p_n, sum_even, sum_odd, y0_sum, y1_sum = lax.fori_loop(
      0, m_steps, body, init
  )
  sum_even = sum_even + p_0
  norm = _miller_norm(w_safe, sum_even, sum_odd)

  # Assemble Y_0(w) and Y_1(w) from J_0, J_1 and the normalized series sums.
  j0 = p_0 * norm
  y0_s = y0_sum * norm
  log_w_over_2 = jnp.log(w_safe) - math.log(2.0)
  y0 = (2.0 / math.pi) * ((log_w_over_2 + np.euler_gamma) * j0 + 4.0 * y0_s)
  zero_w = jnp.zeros_like(w_safe)
  if n == 0:
    return y0, j0 if is_complex else zero_w

  j1 = p_1 * norm
  y1_s = y1_sum * norm
  y1 = (2.0 / math.pi) * (
      -j0 * inv_w + (log_w_over_2 + (np.euler_gamma - 1.0)) * j1 + 4.0 * y1_s
  )
  if n == 1:
    return y1, j1 if is_complex else zero_w

  # Step upward from (Y_0, Y_1) to Y_n using the 3-term recurrence.
  j_n = p_n * norm if is_complex else zero_w
  if n <= 8:
    y_km1, y_k = y0, y1
    for k in range(1, n):
      y_km1, y_k = y_k, (2.0 * k * inv_w) * y_k - y_km1
    return y_k, j_n

  y_n, scale_exp = _forward_recur_rescaled(n, inv_w, y0, y1)
  return _scale_exp2(y_n, scale_exp.astype(real_dtype)), j_n


def _yn_recur_from_asym(
    n: int,
    w: Array,
    active_mask: Array,
    r_min: float,
    r_max: float,
    asym_pairs: int,
    cf_pad: int,
) -> tuple[Array, Array]:
  """Computes (Y_n(w), J_n(w)) for n >= 2 by stepping from orders (0, 1) to n.

  The second tuple element J_n(w) is populated for complex w and zeros for real w.
  """
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  r_raw = jnp.abs(w)
  valid = active_mask & jnp.isfinite(w) & (r_raw >= r_min) & (r_raw <= r_max)
  w_safe = jnp.where(
      valid,
      w,
      jnp.full_like(w, r_min + 1.0),
  )
  inv_w = 1.0 / w_safe

  if not is_complex:
    # For real w, Y_k(w) grows in magnitude once k > w, so stepping upward from
    # (Y_0, Y_1) does not suffer cancellation; track power-of-2 rescales when
    # n > r_min so intermediate orders cannot overflow.
    p0, q0 = _hankel_pq(0, w_safe, asym_pairs)
    p1, q1 = _hankel_pq(1, w_safe, asym_pairs)
    c0, s0 = _hankel_sincos_chi_real(0, jnp.maximum(w_safe, 0.0))
    c1, s1 = s0, -c0
    amp = math.sqrt(2.0 / math.pi) / jnp.sqrt(w_safe)
    y0 = amp * (p0 * s0 + q0 * c0)
    y1 = amp * (p1 * s1 + q1 * c1)

    if n <= int(r_min):
      if n <= 8:
        y_km1, y_k = y0, y1
        for k in range(1, n):
          y_km1, y_k = y_k, (2.0 * k * inv_w) * y_k - y_km1
        return y_k, jnp.zeros_like(y_k)
      _, y_n = lax.fori_loop(
          1,
          n,
          lambda k, s: (s[1], (2.0 * k.astype(real_dtype) * inv_w) * s[1] - s[0]),
          (y0, y1),
      )
      return y_n, jnp.zeros_like(y_n)

    y_n, scale_exp = _forward_recur_rescaled(n, inv_w, y0, y1)
    y_res = _scale_exp2(y_n, scale_exp.astype(real_dtype))
    return y_res, jnp.zeros_like(y_res)
  else:
    # For complex w with Im(w) > 0, stepping Y_k upward directly loses digits
    # when k < |w| because Y_k contains a component of size exp(+Im(w)) that
    # later shrinks at order n. Instead, write Y_n(w) = i * (J_n(w) - H_n^(1))
    # where H_k^(1) = J_k + i * Y_k has size ~ exp(-Im(w)) and grows cleanly
    # upward for every w in the first quadrant, while J_n(w) is found via
    # downward ratios in _jn_recur_from_asym reusing the same (0, 1) terms.
    p0, q0 = _hankel_pq(0, w_safe, asym_pairs)
    p1, q1 = _hankel_pq(1, w_safe, asym_pairs)
    x = lax.real(w_safe)
    y = lax.imag(w_safe)
    c0, s0 = _hankel_sincos_chi_real(0, jnp.maximum(x, 0.0))
    c1, s1 = s0, -c0
    half_amp = (1.0 / math.sqrt(2.0 * math.pi)) / jnp.sqrt(w_safe)
    exp_neg2y = jnp.exp(-2.0 * y)
    j0e = _hankel_unscaled_complex(p0, q0, c0, s0, half_amp, exp_neg2y)
    j1e = _hankel_unscaled_complex(p1, q1, c1, s1, half_amp, exp_neg2y)
    j_n = _jn_recur_from_asym(
        n,
        w,
        active_mask,
        r_min,
        r_max,
        asym_pairs,
        cf_pad,
        j0_j1=(j0e, j1e),
    )

    i_c = lax.complex(jnp.zeros_like(x), jnp.ones_like(x))
    amp_w = 2.0 * half_amp
    # Compute exp(+y) * H_k^(1)(w) for k = 0, 1 so exp(-y) does not underflow
    # before stepping upward to order n.
    h0e = amp_w * lax.complex(c0, s0) * (p0 + i_c * q0)
    h1e = amp_w * lax.complex(c1, s1) * (p1 + i_c * q1)

    h_ne, scale_exp = _forward_recur_rescaled(n, inv_w, h0e, h1e)
    h_n = _scale_exp_y_and_exp2(h_ne, y, scale_exp, -1.0)
    h_re = lax.real(h_n)
    h_im = lax.imag(h_n)
    # Along the positive imaginary axis (x == 0), J_n(i*y) and H_n^(1)(i*y) each
    # have one component that is identically zero; clear rounding noise before
    # subtracting so it cannot corrupt the exponentially smaller component.
    if (n & 1) == 0:
      j_re = lax.real(j_n)
      j_im = jnp.where(x == 0.0, 0.0, lax.imag(j_n))
      h_re = jnp.where(x == 0.0, 0.0, h_re)
    else:
      j_re = jnp.where(x == 0.0, 0.0, lax.real(j_n))
      j_im = lax.imag(j_n)
      h_im = jnp.where(x == 0.0, 0.0, h_im)
    # Combine Y_n = i * (J_n - H_n^(1)):
    #   Re(Y_n) = Im(H_n^(1)) - Im(J_n),  Im(Y_n) = Re(J_n) - Re(H_n^(1)).
    return lax.complex(h_im - j_im, j_re - h_re), lax.complex(j_re, j_im)


def _yn_first_quadrant(n: int, w: Array) -> tuple[Array, Array]:
  """Evaluates (Y_n(w), J_n(w)) for static integer n >= 0 and w in the first quadrant.

  The second tuple element J_n(w) is populated for complex w and zeros for real w.
  """
  # Split the first quadrant (Re(w) >= 0, Im(w) >= 0) by |w| into regimes where
  # each approximation is accurate without cancellation or overflow:
  #   - small |w|: power series around 0 (_yn_series)
  #   - moderate |w|: downward recurrence (_yn_miller)
  #   - intermediate |w| (when n >= 2 and rl < r_asy): order-(0, 1) large-|w|
  #     expansion stepped up to order n (_yn_recur_from_asym)
  #   - large |w| >= r_asy: direct 1/w expansion (_yn_asymptotic)
  # Each helper sanitizes out-of-range lanes so all branches can be evaluated
  # unconditionally and blended with jnp.where under jit/vmap.
  real_dtype = jnp.real(w).dtype
  is_complex = dtypes.issubdtype(w.dtype, np.complexfloating)
  r = jnp.abs(w)

  if real_dtype == np.float64:
    rl = 24.0
    series_iters = 20
    ext_series_iters = 45
    asym_pairs = 10
    base_miller_steps = 64
    cf_pad = 40
  else:
    rl = 10.0
    series_iters = 12
    ext_series_iters = 24
    asym_pairs = 6
    base_miller_steps = 28
    cf_pad = 20

  if n <= 1:
    in_series = r <= 2.0
    in_miller = (r > 2.0) & (r < rl)
    res_ser, j_ser = _yn_series(n, w, series_iters, 2.0)
    res_mil, j_mil = _yn_miller(n, w, base_miller_steps, 2.0, rl)
    res_asy, j_asy = _yn_asymptotic(n, w, asym_pairs, rl)
    y_res = jnp.where(
        in_series, res_ser, jnp.where(in_miller, res_mil, res_asy)
    )
    j_res = (
        jnp.where(in_series, j_ser, jnp.where(in_miller, j_mil, j_asy))
        if is_complex
        else jnp.zeros_like(w)
    )
    return y_res, j_res

  # For n >= 2, the 1/w series needs |w| > r_asy ~ rl + n^2 / 2 before its
  # terms shrink rapidly, while the power series around 0 stays accurate up to
  # r_ser without alternating-sign cancellation.
  r_ser = max(2.0, math.sqrt(2.0 * (n + 1)))
  r_asy = max(r_ser, rl + 0.5 * n * n)

  if is_complex and r_ser < rl:
    # For complex w with |w| < rl, stepping Y_k upward from (Y_0, Y_1) loses
    # precision when Im(w) is large or |w| is not well above n. Default to a
    # longer power series across |w| < rl, and use _yn_miller only inside the
    # wedge near the real axis where upward stepping from (Y_0, Y_1) is benign.
    res_ser, j_ser = _yn_series(n, w, ext_series_iters, rl)
    res_asy, j_asy = _yn_asymptotic(n, w, asym_pairs, r_asy)
    res = jnp.where(r < rl, res_ser, res_asy)
    j_res = jnp.where(r < rl, j_ser, j_asy)
    r_mil_min = max(r_ser, 0.9 * n)
    if r_mil_min < rl:
      in_miller = (
          (r > r_mil_min)
          & (r > 0.9 * n + 0.6 * lax.imag(w))
          & (r < rl)
          & (lax.imag(w) <= 0.8 * lax.real(w))
      )
      miller_steps = max(n + cf_pad, base_miller_steps)
      res_mil, j_mil = _yn_miller(n, w, miller_steps, r_mil_min, rl)
      res = jnp.where(in_miller, res_mil, res)
      j_res = jnp.where(in_miller, j_mil, j_res)
  else:
    in_series = r <= r_ser
    res_ser, j_ser = _yn_series(n, w, series_iters, r_ser)
    res_asy, j_asy = _yn_asymptotic(n, w, asym_pairs, r_asy)
    res = jnp.where(in_series, res_ser, res_asy)
    j_res = jnp.where(in_series, j_ser, j_asy) if is_complex else jnp.zeros_like(w)

    if r_ser < rl:
      in_miller = (r > r_ser) & (r < rl)
      res_mil, _ = _yn_miller(n, w, base_miller_steps, r_ser, rl)
      res = jnp.where(in_miller, res_mil, res)

  # In complex128, _yn_recur_from_asym already reaches full float64 precision
  # by |w| = 17, which avoids any remaining cancellation in the power series
  # near |w| = rl = 24.
  rec_floor = 17.0 if (is_complex and real_dtype == np.float64) else rl
  r_rec_min = max(rec_floor, r_ser)
  if r_rec_min < r_asy:
    in_recur = (r >= r_rec_min) & (r < r_asy)
    res_rec, j_rec = _yn_recur_from_asym(
        n, w, in_recur, r_rec_min, r_asy, asym_pairs, cf_pad
    )
    res = jnp.where(in_recur, res_rec, res)
    if is_complex:
      j_res = jnp.where(in_recur, j_rec, j_res)

  return res, j_res


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0,))
def _yn(n: int, z: Array) -> Array:
  with config.numpy_dtype_promotion("standard"):
    return _yn_impl(n, z)


def _yn_impl(n: int, z: Array) -> Array:
  is_complex = dtypes.issubdtype(z.dtype, np.complexfloating)
  n_abs = abs(n)
  is_odd = (n_abs & 1) == 1
  # Negative integer orders satisfy Y_{-k}(z) = (-1)^k * Y_k(z), so only odd
  # negative orders flip sign relative to |n|.
  neg_n = n < 0 and is_odd

  if not is_complex:
    w = jnp.where(z > 0.0, z, 1.0)
    res, _ = _yn_first_quadrant(n_abs, w)
    if neg_n:
      res = -res

    # Boundary values for real inputs (for z < 0, Y_n(z) is complex-valued so
    # real-dtype inputs return NaN, matching scipy.special.yn).
    res = jnp.where(z == np.inf, 0.0, res)
    res = jnp.where(z == 0.0, np.inf if neg_n else -np.inf, res)
    res = jnp.where((z < 0.0) | jnp.isnan(z), np.nan, res)
    return res
  else:
    x = lax.real(z)
    y = lax.imag(z)
    w = lax.complex(jnp.abs(x), jnp.abs(y))
    res, j_w = _yn_first_quadrant(n_abs, w)

    # Map from the first quadrant w = |x| + i*|y| back to arbitrary z = x + i*y
    # using reflection identities across the imaginary and real axes:
    # 1. Reflect across the imaginary axis when x < 0:
    #    Y_{|n|}(-|x| + i*|y|) = (-1)^n * (conj(Y_{|n|}(w)) + 2i*conj(J_{|n|}(w))).
    #    Because Y_n has a log(z) term, crossing into Re(z) < 0 adds 2i * J_n.
    #    When |y| is huge and both Y_n(w) and J_n(w) overflow, their sum would
    #    produce inf - inf = NaN; use Y_n(w) + 2i * J_n(w) -> -Y_n(w) instead.
    two_i_conj_j = lax.complex(2.0 * lax.imag(j_w), 2.0 * lax.real(j_w))
    j_inf = jnp.isinf(lax.real(two_i_conj_j)) | jnp.isinf(
        lax.imag(two_i_conj_j)
    )
    comb = jnp.where(j_inf, -jnp.conj(res), jnp.conj(res) + two_i_conj_j)
    res_neg_x = -comb if is_odd else comb
    res = jnp.where(x < 0.0, res_neg_x, res)
    # 2. Reflect across the real axis when y < 0 (using signbit so y = -0.0
    #    selects the lower side of the branch cut along the negative real axis):
    #    Y_{|n|}(x - i*|y|) = conj(Y_{|n|}(x + i*|y|)).
    res = jnp.where(jnp.signbit(y), jnp.conj(res), res)
    # 3. Flip sign for odd negative orders Y_{-|n|}(z) = (-1)^n * Y_{|n|}(z).
    if neg_n:
      res = -res
    # 4. On the positive real axis (x > 0, y == 0), Y_n(x) is purely real.
    res = jnp.where(
        (y == 0.0) & (x > 0.0),
        lax.complex(lax.real(res), jnp.zeros_like(y)),
        res,
    )

    # Boundary values for z = 0, |Re(z)| = inf (oscillatory decay -> 0), and
    # |Im(z)| = inf (exponential growth with undefined phase -> NaN).
    nan_c = lax.complex(jnp.full_like(x, np.nan), jnp.full_like(y, np.nan))
    zero_c = jnp.zeros_like(z)
    z_zero_c = lax.complex(
        jnp.full_like(x, np.inf if neg_n else -np.inf), jnp.zeros_like(y)
    )
    res = jnp.where(z == 0, z_zero_c, res)
    res = jnp.where(jnp.isinf(x), zero_c, res)
    res = jnp.where(jnp.isnan(x) | jnp.isnan(y) | jnp.isinf(y), nan_c, res)
    return res


@_yn.defjvp
def _yn_jvp(
    n: int, primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
  (z,) = primals
  (z_dot,) = tangents
  y = _yn(n, z)
  if n == 0:
    dy_dz = -_yn(1, z)
  else:
    # Differentiate via Y_n'(z) = -Y_{n+1}(z) + (n/z) * Y_n(z) (for n > 0) or
    # Y_{n-1}(z) - (n/z) * Y_n(z) (for n < 0). Reusing y = Y_n(z) needs only one
    # extra yn call per derivative order instead of two (so k-th derivatives
    # cost 2^k evaluations rather than 3^k), and puts the leading z^(-|n|-1)
    # divergence as z -> 0 in a single term (dom) so it cannot cancel. When
    # either term overflows near z = 0, keep dom alone to avoid inf - inf = NaN.
    dom = -_yn(n + 1, z) if n > 0 else _yn(n - 1, z)
    z_safe = jnp.where(z == 0, jnp.ones_like(z), z)
    term = (float(abs(n)) / z_safe) * y
    is_zero = z == 0
    if dtypes.issubdtype(z.dtype, np.complexfloating):
      use_dom = is_zero | ~jnp.isfinite(dom) | ~jnp.isfinite(term)
      dy_dz = jnp.where(use_dom, dom, dom + term)
    else:
      inf_or_zero = is_zero | jnp.isinf(dom) | jnp.isinf(term)
      dy_dz = jnp.where(inf_or_zero, dom, dom + term)
  return y, dy_dz * z_dot


def yn(n: int, z: ArrayLike) -> Array:
  r"""Bessel function of the second kind of integer order `n`.

  JAX implementation of :obj:`scipy.special.yn`.

  .. math::

     Y_n(z) = -\frac{1}{\pi} \sum_{k=0}^{n-1} \frac{(n - k - 1)!}{k!}
     \left(\frac{z}{2}\right)^{2k - n} + \frac{2}{\pi} \ln\left(\frac{z}{2}\right) J_n(z)
     - \frac{1}{\pi} \sum_{k=0}^{\infty} \frac{\psi(k + 1) + \psi(n + k + 1)}{k!\,(n + k)!}
     \left(-\frac{z^2}{4}\right)^k \left(\frac{z}{2}\right)^n

  for integer order :math:`n \ge 0` (with :math:`Y_{-n}(z) = (-1)^n Y_n(z)`).

  Args:
    n: The integer order of the Bessel function (must be a static integer).
    z: The argument(s) at which to evaluate :math:`Y_n(z)`. Real and complex
      inputs are supported.

  Returns:
    An array containing the evaluated Bessel function values.

  Notes:
    Unlike :obj:`scipy.special.yn`, ``jax.scipy.special.yn`` requires ``n`` to
    be a concrete integer order (so ``n`` cannot be traced or batched with
    :func:`jax.vmap`), and supports complex inputs directly (matching
    :obj:`scipy.special.yv` on the complex plane).

  See Also:
    - :func:`jax.scipy.special.jn`
    - :func:`jax.scipy.special.iv`
    - :func:`jax.scipy.special.bessel_jn`
  """
  n = core.concrete_or_error(
      operator.index,
      n,
      "The 'n' argument of jax.scipy.special.yn must be a static integer.",
  )
  (z,) = promote_args_inexact("yn", z)
  orig_dtype = z.dtype
  if orig_dtype in (dtypes.bfloat16, np.float16):
    z = z.astype(np.float32)
    return _yn(n, z).astype(orig_dtype)
  return _yn(n, z)


def _iv_asymptotic_real(
    n: int, w: Array, max_terms: int, r_min: float
) -> Array:
  """Evaluates I_n(w) for large real w > 0 using its 1/w expansion."""
  real_dtype = w.dtype
  overflow_w = 1000.0 if real_dtype == np.float64 else 120.0
  w_safe = jnp.where(
      jnp.isfinite(w) & (w >= r_min) & (w < overflow_w),
      w,
      jnp.full_like(w, r_min),
  )
  # For large w, I_n(w) ~ (exp(w) / sqrt(2*pi*w)) * sum_{k>=0} a_k(n) / (8w)^k,
  # where consecutive terms satisfy
  # a_k / a_{k-1} = -((2k - 1)^2 - 4n^2) / (8 * k * w) (DLMF 10.40.1).
  # Evaluate the finite series from the inside out (Horner's method) in
  # double-word arithmetic so rounding errors don't build up.
  mu = float(4 * n * n)
  zero = jnp.zeros_like(w_safe)
  one = jnp.ones_like(w_safe)
  inv_8w0, inv_8w1 = f2_recip(8.0 * w_safe, zero)

  def body(i, state):
    s0, s1 = state
    kf = lax.convert_element_type(max_terms - i, real_dtype)
    two_k_m1 = 2.0 * kf - 1.0
    c0, c1 = f2_div(two_k_m1 * two_k_m1 - mu, zero, kf, zero)
    f0, f1 = f2_mul(c0, c1, inv_8w0, inv_8w1)
    t0, t1 = f2_mul(s0, s1, f0, f1)
    return f2_add(one, zero, t0, t1)

  tot0, tot1 = lax.fori_loop(0, max_terms, body, (one, zero))
  # Refine sqrt(w_safe) to double-word precision (sw0, sw1) with one Newton step
  # sqrt(w) ≈ sw + (w - sw^2) / (2 * sw), then divide by sqrt(2 * pi * w) and
  # multiply by exp(w_safe).
  sw = jnp.sqrt(w_safe)
  sw2_0, sw2_1 = two_prod(sw, sw)
  sw_rem = (w_safe - sw2_0) - sw2_1
  sw0, sw1 = fast_two_sum(sw, sw_rem / (2.0 * sw))
  q0, q1 = f2_div(tot0, tot1, sw0, sw1)
  c2pi_hi, c2pi_lo = (
      _INV_SQRT_2PI_F64 if real_dtype == np.float64 else _INV_SQRT_2PI_F32
  )
  u0, u1 = f2_mul(
      q0, q1, _const(w_safe, c2pi_hi), _const(w_safe, c2pi_lo)
  )
  res = _scale_f2_exp_w(u0, u1, w_safe)
  return jnp.where(w >= overflow_w, np.inf, res)


def _iv_recur_neumann_real(
    n: int,
    w: Array,
    in_recur: Array,
    r_min: float,
    cf_pad: int,
) -> Array:
  """Evaluates I_n(w) for intermediate real w > 0 via backward recurrence."""
  # Modified Bessel functions satisfy
  # I_{k-1}(w) = (2k / w) * I_k(w) + I_{k+1}(w).
  # Stepping this recurrence forward from k = 0 to n loses precision when k > w
  # because I_k(w) shrinks rapidly with k while rounding errors grow. Stepping
  # backward from a large starting index k_top down to 1 is numerically stable.
  #
  # Dividing the recurrence by I_{k-1}(w) gives a backward step for the ratio
  # r_k = I_k(w) / I_{k-1}(w):
  #   r_k = 1 / (2k / w + r_{k+1}),  starting from r_{k_top + 1} = 0.
  # In a single backward pass from k = k_top down to 1, we also maintain:
  #   1. p = r_1 * r_2 * ... * r_n = I_n(w) / I_0(w)
  #   2. s = 1 + r_1 * (1 + r_2 * (1 + ...))
  #        = (I_0(w) + I_1(w) + ... + I_{k_top}(w)) / I_0(w)
  # Using the identity I_0(w) + 2 * sum_{k>=1} I_k(w) = exp(w) (DLMF 10.35.2),
  # 2 * s - 1 equals exp(w) / I_0(w), so p / (2 * s - 1) = exp(-w) * I_n(w),
  # which we then multiply by exp(w) via _scale_f2_exp_w.
  real_dtype = w.dtype
  w_safe = jnp.where(in_recur, w, jnp.full_like(w, r_min))
  zero = jnp.zeros_like(w_safe)
  one = jnp.ones_like(w_safe)
  inv_w0, inv_w1 = f2_recip(w_safe, zero)
  # Pre-split inv_w0 outside the loop so 2k * inv_w is computed in double-word
  # precision without calling split_half every iteration (and without XLA/LLVM
  # fusing s - 2k * inv_w0 into an FMA that would corrupt the error term).
  inv_w_hi, inv_w_mid = split_half(inv_w0)

  w_int = jnp.floor(w_safe).astype(np.int32)
  m_top = jnp.maximum(np.int32(n), w_int)
  k_top = jnp.where(in_recur, m_top + np.int32(cf_pad), np.int32(0))
  k_init = jnp.max(k_top, initial=np.int32(0))

  scale_thresh = 2.0 ** (-500 if real_dtype == np.float64 else -60)
  scale_up = 2.0 ** (500 if real_dtype == np.float64 else 60)
  scale_bits = np.int32(500 if real_dtype == np.float64 else 60)

  def cf_cond(state):
    k, *_ = state
    return k >= 1

  def cf_body(state):
    k, r0, r1, s0, s1, p0, p1, scale_exp = state
    active = in_recur & (k <= k_top)
    two_kf = lax.convert_element_type(2 * k, real_dtype)
    tk_p0, tk_e0 = fast_two_sum(two_kf * inv_w_hi, two_kf * inv_w_mid)
    tk0, tk1 = fast_two_sum(tk_p0, tk_e0 + two_kf * inv_w1)
    den0, den1 = f2_add(tk0, tk1, r0, r1)
    rk0, rk1 = f2_recip(den0, den1)
    rs0, rs1 = f2_mul(rk0, rk1, s0, s1)
    sn0, sn1 = f2_add(one, zero, rs0, rs1)
    r0 = jnp.where(active, rk0, r0)
    r1 = jnp.where(active, rk1, r1)
    s0 = jnp.where(active, sn0, s0)
    s1 = jnp.where(active, sn1, s1)
    in_prod = active & (k <= n)
    pn0, pn1 = f2_mul(p0, p1, rk0, rk1)
    p0 = jnp.where(in_prod, pn0, p0)
    p1 = jnp.where(in_prod, pn1, p1)
    # When n is large, the ratio product p = I_n(w) / I_0(w) can become very
    # small; rescale (p0, p1) by 2**scale_bits and record -scale_bits in
    # scale_exp so p does not underflow before multiplying by exp(w).
    small = in_prod & (p0 < scale_thresh) & (p0 > 0.0)
    p0 = jnp.where(small, p0 * scale_up, p0)
    p1 = jnp.where(small, p1 * scale_up, p1)
    scale_exp = jnp.where(small, scale_exp - scale_bits, scale_exp)
    return k - np.int32(1), r0, r1, s0, s1, p0, p1, scale_exp

  init_cf = (
      k_init,
      zero,
      zero,
      one,
      zero,
      one,
      zero,
      jnp.zeros_like(w_int),
  )
  _, _, _, s0, s1, p0, p1, scale_exp = lax.while_loop(cf_cond, cf_body, init_cf)
  s0, s1 = f2_add(2.0 * s0, 2.0 * s1, -one, zero)
  u0, u1 = f2_div(p0, p1, s0, s1)
  return _scale_f2_exp_w(u0, u1, w_safe, scale_exp)


def _iv_real_nonneg(n: int, w: Array) -> Array:
  """Evaluates I_n(w) for static integer n >= 0 and real w >= 0."""
  real_dtype = w.dtype
  overflow_w = 1000.0 if real_dtype == np.float64 else 120.0

  # For orders n = 0 and n = 1, delegate to the dedicated rational minimax
  # primitives bessel_i0e and bessel_i1e (which compute exp(-w) * I_n(w))
  # and multiply by exp(w).
  if n <= 1:
    w_safe = jnp.where(jnp.isfinite(w) & (w < overflow_w), w, 0.0)
    ie = lax.bessel_i0e(w_safe) if n == 0 else lax.bessel_i1e(w_safe)
    res = _scale_f2_exp_w(ie, None, w_safe)
    return jnp.where(w >= overflow_w, np.inf, res)

  # For n >= 2, split w >= 0 into up to three regimes:
  #   1. Small w (w <= r_ser): ascending power series (all terms are positive,
  #      so there is no alternating-sign cancellation).
  #   2. Intermediate w (r_ser < w < r_rec_max): backward recurrence.
  #   3. Large w (w >= r_rec_max): large-argument 1/w expansion (or +inf when
  #      r_rec_max is already past the overflow threshold).
  if real_dtype == np.float64:
    rl = 24.0
    series_iters = 38
    asym_terms = 20
    cf_pad = 40
  else:
    rl = 10.0
    series_iters = 26
    asym_terms = 12
    cf_pad = 20

  r_ser = max(12.0, math.sqrt(2.0 * (n + 1)))
  r_asy = max(r_ser, rl + 0.5 * n * n)
  r_rec_max = min(r_asy, max(overflow_w, 2.0 * n))

  in_series = w <= r_ser
  res_ser = _jn_series(n, w, series_iters, r_ser, modified=True)
  res_asy = (
      jnp.full_like(w, np.inf)
      if r_rec_max >= overflow_w
      else _iv_asymptotic_real(n, w, asym_terms, r_rec_max)
  )
  res = jnp.where(in_series, res_ser, res_asy)

  if r_ser < r_rec_max:
    in_recur = (w > r_ser) & (w < r_rec_max)
    res_rec = _iv_recur_neumann_real(n, w, in_recur, r_ser, cf_pad)
    res = jnp.where(in_recur, res_rec, res)

  return res


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(0,))
@partial(jit, static_argnums=(0,))
def _iv(v: int, z: Array) -> Array:
  with config.numpy_dtype_promotion("standard"):
    return _iv_impl(v, z)


def _iv_impl(v: int, z: Array) -> Array:
  is_complex = dtypes.issubdtype(z.dtype, np.complexfloating)
  # For integer orders, I_{-v}(z) = I_v(z) for all v in Z.
  n_abs = abs(v)
  is_odd = (n_abs & 1) == 1

  if not is_complex:
    res = _iv_real_nonneg(n_abs, jnp.abs(z))
    if is_odd:
      res = copysign(res, z)
    return jnp.where(jnp.isnan(z), np.nan, res)

  x = lax.real(z)
  y = lax.imag(z)
  # By DLMF 10.27.6, I_n(z) = i^{-n} J_n(i * z). For x >= 0 and y >= 0,
  # let w = |y| + i * |x| in the first quadrant; then
  # I_n(|x| + i * |y|) = i^n * conj(J_n(|y| + i * |x|)).
  w = lax.complex(jnp.abs(y), jnp.abs(x))
  j_w = _jn_first_quadrant(n_abs, w)
  jr = lax.real(j_w)
  ji = lax.imag(j_w)
  mod4 = n_abs & 3
  if mod4 == 0:
    res = lax.complex(jr, -ji)
  elif mod4 == 1:
    res = lax.complex(ji, jr)
  elif mod4 == 2:
    res = lax.complex(-jr, ji)
  else:
    res = lax.complex(-ji, -jr)

  # Map from the first quadrant |x| + i * |y| back to z = x + i * y:
  # 1. If x < 0: I_n(-|x| + i * |y|) = (-1)^n * conj(I_n(|x| + i * |y|))
  res_neg_x = -jnp.conj(res) if is_odd else jnp.conj(res)
  res = jnp.where(x < 0.0, res_neg_x, res)
  # 2. If y < 0: I_n(x - i * |y|) = conj(I_n(x + i * |y|))
  res = jnp.where(y < 0.0, jnp.conj(res), res)
  # 3. On the imaginary axis (x == 0), I_n(i * y) = i^n * J_n(y) is purely
  # imaginary for odd n and purely real for even n; on the real axis
  # (y == 0), I_n(x) is purely real.
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
  res = jnp.where(jnp.isinf(y), zero_c, res)
  res = jnp.where(jnp.isnan(x) | jnp.isnan(y) | jnp.isinf(x), nan_c, res)
  return res


@_iv.defjvp
def _iv_jvp(
    v: int, primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
  (z,) = primals
  (z_dot,) = tangents
  y = _iv(v, z)
  n_abs = abs(v)
  # DLMF 10.29.2: I_0'(z) = I_1(z), and
  # I_v'(z) = 0.5 * (I_{v-1}(z) + I_{v+1}(z)).
  dy_dz = (
      _iv(1, z)
      if n_abs == 0
      else _const(z, 0.5) * (_iv(n_abs - 1, z) + _iv(n_abs + 1, z))
  )
  return y, dy_dz * z_dot


def iv(v: int, z: ArrayLike) -> Array:
  r"""Modified Bessel function of the first kind of integer order `v`.

  JAX implementation of :obj:`scipy.special.iv` for integer orders.

  .. math::

     I_v(z) = \sum_{k=0}^{\infty} \frac{1}{k!\,(v + k)!}
     \left(\frac{z}{2}\right)^{2k + v}

  for integer order :math:`v \ge 0` (with :math:`I_{-v}(z) = I_v(z)`).

  Args:
    v: The integer order of the modified Bessel function (must be a static
      integer).
    z: The argument(s) at which to evaluate :math:`I_v(z)`. Real and complex
      inputs are supported.

  Returns:
    An array containing the evaluated modified Bessel function values.

  Notes:
    Unlike :obj:`scipy.special.iv`, ``jax.scipy.special.iv`` requires ``v`` to
    be a concrete integer order (so non-integer orders are not supported and
    ``v`` cannot be traced or batched with :func:`jax.vmap`).

  See Also:
    - :func:`jax.scipy.special.i0`
    - :func:`jax.scipy.special.i0e`
    - :func:`jax.scipy.special.i1`
    - :func:`jax.scipy.special.i1e`
    - :func:`jax.scipy.special.jn`
    - :func:`jax.scipy.special.yn`
  """
  v = abs(
      core.concrete_or_error(
          operator.index,
          v,
          "The 'v' argument of jax.scipy.special.iv must be a static integer.",
      )
  )
  (z,) = promote_args_inexact("iv", z)
  orig_dtype = z.dtype
  if orig_dtype in (dtypes.bfloat16, np.float16):
    z = z.astype(np.float32)
    return _iv(v, z).astype(orig_dtype)
  return _iv(v, z)
