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

"""Polynomial approximations for erf-related functions and their inverses."""

import math
from jax._src import dtypes
from jax._src.api import jit
from jax._src.custom_derivatives import custom_jvp
from jax._src.lax.double_word import (
    LN2_SPLIT_F32, LN2_SPLIT_F64, fast_two_sum, split_bits, split_half_mask,
    two_prod, two_sum,
)
from jax._src.lax.lax import (
    AccuracyMode, RoundingMethod, _const, asarray, bitcast_convert_type, clamp,
    convert_element_type, exp, exp2, expm1, full_like, log, log1p, max as lax_max,
    min as lax_min, neg, polynomial, reciprocal, round as lax_round, select,
    select_n, shift_left, sqrt, square,
)
from jax._src.typing import Array
import numpy as np


def _split_square(x: Array) -> tuple[Array, Array]:
  """Splits `x^2` into `s_hi + s_lo` where `s_hi = x_hi^2` is exact.

  When computing `exp(-x^2)` (in `erfc`) or `exp(x^2)` (in `erfcx`), a naive
  evaluation rounds `x^2` to a single floating-point number before calling
  `exp`. The rounding error in `x^2` can be up to `0.5 * ulp(x^2) ~= 0.5 * x^2 *
  eps`.
  Because `(d/du) exp(u) = exp(u)`, exponentiating `x^2` magnifies that absolute
  error into a relative error of `0.5 * x^2` ULPs in the output—up to ~45 ULPs
  at `x = 9.5` in `float32`, and ~350 ULPs at `x = 26.5` in `float64`.

  To avoid this error magnification without relying on hardware FMA, we split
  `x = x_hi + x_lo` by masking off the lower 12 mantissa bits in `float32`
  (leaving 12 significant bits) or the lower 27 mantissa bits in `float64`
  (leaving 26 significant bits). Because `x_hi` has at most half the mantissa
  bits of the format, `s_hi = x_hi * x_hi` is representable with zero rounding
  error. The remainder `s_lo = x^2 - x_hi^2 = (x - x_hi) * (x + x_hi)` captures
  the low-order bits (`|s_lo| / x^2 <= 2^-11` in `float32`, `<= 2^-25` in
  `float64`), allowing us to evaluate `exp(+-x^2) = exp(+-s_hi) * exp(+-s_lo)`
  to full precision.
  """
  x_hi, x_lo = split_half_mask(x)
  return x_hi * x_hi, x_lo * (x + x_hi)


def _expm1_lo(u: Array) -> Array:
  """Approximates `exp(u) - 1` for the small remainder `u = +-s_lo` of `x^2`.

  After splitting `x^2 = s_hi + s_lo` with `_split_square`, `|u| = |s_lo|` is
  at most ~0.03 on the non-underflowing/non-overflowing domain of `erfc` and
  `erfcx` in `float32` (`|x| <= 11`), and at most ~1.2e-5 in `float64`
  (`|x| <= 28`). A short Taylor polynomial `u * (1 + u/2 + u^2/6 + ...)` gives
  `exp(u) - 1` to full machine precision, and we then combine the factors as
  `exp(s_hi) * (y + y * _expm1_lo(u))` so adding the small correction to `1`
  does not lose the low bits of `y`.
  """
  if u.dtype == np.float32:
    return u * polynomial(u, [1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0])
  return u * polynomial(u, [1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0])


@custom_jvp
def exp_neg_sq(x: Array) -> Array:
  """Evaluates `exp(-x^2)` using `_split_square` for `float32` and `float64`."""
  if x.dtype not in (np.float32, np.float64):
    return exp(neg(square(x)))
  max_ax = 11.0 if x.dtype == np.float32 else 28.0
  ax_exp = lax_min(abs(x), _const(x, max_ax))
  s_hi, s_lo = _split_square(ax_exp)
  e = exp(-s_hi, accuracy=AccuracyMode.HIGHEST)
  return e + e * _expm1_lo(-s_lo)


@exp_neg_sq.defjvp
def _exp_neg_sq_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  # The primal may be a Python scalar, e.g. under nested `jvp` of `erfc(10.0)`.
  x = asarray(x)
  ans = exp_neg_sq(x)
  # For `|x| >= max_ax`, `ans` underflows to exactly 0. At `x = +-inf`,
  # `-2 * x * ans` would then be `inf * 0 = NaN`, although the true derivative
  # is 0. Replacing `x` with 0 there gives 0 and cannot change any finite result,
  # since `ans` is already 0.
  max_ax = 11.0 if x.dtype == np.float32 else 28.0
  safe_x = select(abs(x) < _const(x, max_ax), x, full_like(x, 0.0))
  return ans, -_const(x, 2.0) * safe_x * ans * x_dot


@custom_jvp
def exp_neg_half_sq(x: Array) -> Array:
  """Evaluates `exp(-x^2 / 2)` accurately in the tails for `float32` and `float64`."""
  if x.dtype not in (np.float32, np.float64):
    return exp(-0.5 * square(x))
  # Beyond max_ax (16 in float32, 40 in float64), exp(-x^2 / 2) underflows to 0
  # even in subnormals; clamping avoids overflow in _split_square for huge |x|.
  max_ax = 16.0 if x.dtype == np.float32 else 40.0
  ax_exp = lax_min(abs(x), _const(x, max_ax))
  s_hi, s_lo = _split_square(ax_exp)
  e = exp(-0.5 * s_hi, accuracy=AccuracyMode.HIGHEST)
  return e + e * _expm1_lo(-0.5 * s_lo)


@exp_neg_half_sq.defjvp
def _exp_neg_half_sq_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  x = asarray(x)
  ans = exp_neg_half_sq(x)
  # For `|x| >= max_ax`, `ans` underflows to 0. Replacing `x` with 0 there
  # avoids `inf * 0 = NaN` at `x = +-inf` without changing any finite result.
  max_ax = 16.0 if x.dtype == np.float32 else 40.0
  safe_x = select(abs(x) < max_ax, x, full_like(x, 0.0))
  return ans, -safe_x * ans * x_dot


# Sollya `fpminimax` polynomial coefficients for `R(1 - w) - 1 = erf(x) / x - 1`
# on `|x| <= 1` in terms of `w = 1 - x^2 = (1 - x) * (1 + x) in [0, 1]`, listed
# from lowest degree to highest degree for `lax.polynomial`. They were generated
# by this Sollya script:
#
#   prec = 300!;
#   display = hexadecimal!;
#   f = erf(sqrt(1 - x))/sqrt(1 - x) - 1;
#   /* w = 1 (x = 0) is a removable 0/0, so stop just short of it. */
#   I = [0; 1 - 1b-40];
#   p32 = fpminimax(f, 6, [|SG...|], I, absolute, floating);
#   p64 = fpminimax(f, 12, [|D...|], I, absolute, floating);
#   print("# _ERF32_T_M1");
#   for i from 0 to 6 do print(coeff(p32, i));
#   print("# _ERF64_T_M1");
#   for i from 0 to 12 do print(coeff(p64, i));
#
# The error that matters is the relative error these cause in `erfc(x)`, which
# is `x * |p(w) - f(w)| / erfc(x)`. That is at most 0.148 half-ULPs in `float32`
# and 0.169 half-ULPs in `float64`, which is exactly the error from rounding the
# constant term `erf(1) - 1` alone, so these degrees cannot do better. Going one
# degree lower raises it to 4.6 half-ULPs (`float32`) and 0.41 (`float64`).
#
# Why we expand around `w = 0` (`x^2 = 1`) rather than `x^2 = 0`:
# In `_erfc_small`, we evaluate `erfc(x) = (1 - x) - x * (erf(x) / x - 1)`.
# Near `x = 0` (`w = 1`), `1 - x ~= 1` dominates and `x * (erf(x) / x - 1)`
# vanishes like `0.128 * x`. Near `x = 1` (`w = 0`), however, `1 - x -> 0` and
# `-x * (erf(x) / x - 1)` supplies 100% of `erfc(1) ~= 0.1573`. Expanding in
# `w = (1 - x) * (1 + x)` around `w = 0` makes the constant term equal to
# `erf(1) - 1 = -erfc(1) ~= -0.1573` with zero cancellation as `x -> 1`, while
# all higher-order coefficients `c_1, ..., c_n` are positive on `w in [0, 1]`.
_ERF32_T_M1 = (
    float.fromhex("-0x1.422616p-3"),
    float.fromhex("0x1.b5daf4p-3"),
    float.fromhex("0x1.cf7114p-5"),
    float.fromhex("0x1.9ae15ap-7"),
    float.fromhex("0x1.35b662p-9"),
    float.fromhex("0x1.5853f2p-12"),
    float.fromhex("0x1.4b5ec6p-14"),
)

_ERF64_T_M1 = (
    float.fromhex("-0x1.4226162fbddd5p-3"),
    float.fromhex("0x1.b5db0451254fap-3"),
    float.fromhex("0x1.cf6d2bc57822cp-5"),
    float.fromhex("0x1.9b3c1054ff0dfp-7"),
    float.fromhex("0x1.31cd0ceac3b4dp-9"),
    float.fromhex("0x1.84227cc6c435fp-12"),
    float.fromhex("0x1.ac2510dffaf4dp-15"),
    float.fromhex("0x1.a0df95c08a7d3p-18"),
    float.fromhex("0x1.6af2214c994d3p-21"),
    float.fromhex("0x1.1d3815deb5114p-24"),
    float.fromhex("0x1.9f6d498ffeab3p-28"),
    float.fromhex("0x1.dee5605ee0a79p-32"),
    float.fromhex("0x1.f987104017746p-35"),
)


def _erfc_small(x: Array) -> Array:
  """Computes `erfc(x) = 1 - erf(x)` on `|x| <= 1`.

  Writing `r_m1(w) = erf(x) / x - 1` as a polynomial in
  `w = 1 - x^2 = (1 - x) * (1 + x)` gives
  `erfc(x) = 1 - x * (1 + r_m1(w)) = (1 - x) - x * r_m1(w)`.
  - For `x < 0.25`, `erf(x) < 0.2763`, so `1 - x * (1 + r_m1(w))` has no
    cancellation and avoids a separate rounding of `1 - x` near `x = 0`.
  - For `0.25 <= x <= 1.0`, `(1 - x) - x * r_m1(w)` avoids both subtraction
    cancellation in `1 - erf(x)` and polynomial cancellation in `r_m1(w)` as
    `x -> 1` (`w -> 0`).
  """
  d = 1.0 - x
  w = d * (1.0 + x)
  if x.dtype == np.float32:
    r_m1 = polynomial(w, _ERF32_T_M1, unroll=len(_ERF32_T_M1))
  elif x.dtype == np.float64:
    r_m1 = polynomial(w, _ERF64_T_M1, unroll=len(_ERF64_T_M1))
  else:
    raise TypeError(f"Unsupported dtype: {x.dtype}")
  return select(x < 0.25, 1.0 - x * (1.0 + r_m1), d - x * r_m1)


# Sollya `fpminimax` polynomial coefficients for the scaled derivative
# `G(x) = -x^2 * erfcx'(x)` on `x >= 1`, in ascending degree order for
# `lax.polynomial`.
#
# Why we approximate `erfcx'(x)` first and reconstruct `erfcx(x)` from it:
# The scaled complementary error function `erfcx(x) = exp(x^2) * erfc(x)`
# satisfies the linear differential equation:
#
#   erfcx'(x) = 2 * x * erfcx(x) - 2 / sqrt(pi).
#
# As `x -> inf`, `2 * x * erfcx(x) -> 2 / sqrt(pi)` (specifically,
# `2 * x * erfcx(x) = (2 / sqrt(pi)) * (1 - 1/(2*x^2) + 3/(4*x^4) - ...)`).
# If we approximate `erfcx(x)` first and compute its derivative via
# `2 * x * erfcx(x) - 2 / sqrt(pi)`, the two terms nearly cancel for large `x`,
# losing `O(x^2)` relative accuracy.
#
# If instead we approximate `erfcx'(x)` directly on `x >= 1`, we can recover the
# primal `erfcx(x)` by solving the ODE for `erfcx(x)`:
#
#   erfcx(x) = (1 / sqrt(pi) + 0.5 * erfcx'(x)) / x.
#
# Going in this direction has no cancellation at all: on `x >= 1`,
# `0.5 * erfcx'(x)` is bounded in `[-0.1366, 0)` while `1 / sqrt(pi) ~= 0.5642`,
# so their sum is always `>= 0.4275`. Better yet, adding the exact constant
# `1 / sqrt(pi)` attenuates any relative error in `erfcx'(x)` by at least `3.1x`
# (and by `O(x^2)` as `x -> inf`).
#
# Since `erfcx'(x) ~ -1 / (sqrt(pi) * x^2)` for large `x`, we factor out
# `-1 / x^2` and approximate the positive, bounded function
# `G(x) = -x^2 * erfcx'(x)` (which increases smoothly from `~0.2731` at `x = 1`
# to `1 / sqrt(pi) ~= 0.5642` as `x -> inf`) in terms of `u = 1 / x` and
# `t = u^2 = 1 / x^2`:
#
#   _ERFC32_G_1_2:   `x in [1, 2)`,   degree 7 in `s = u - 0.75`
#                                     (rel err 2.54e-8, ~0.42 ULP)
#   _ERFC32_G_2_8:   `x in [2, 8)`,   degree 7 in `s = u - 0.3125`
#                                     (rel err 4.48e-8, ~0.75 ULP)
#   _ERFC32_G_8_INF: `x in [8, inf)`, degree 3 in `t = u^2`
#                                     (rel err 5.36e-8, ~0.90 ULP)
#
# The `_ERFC32_G_*`, `_ERFC64_G_*` tables below (including `_ERFC*_G_0_1`) were
# generated by this Sollya script, which prints each table in ascending degree
# order:
#
#   prec = 300!;
#   display = hexadecimal!;
#   procedure show(name, p, n) {
#     var i;
#     print("#", name);
#     for i from 0 to n do print(coeff(p, i));
#   };
#   /* G(x) = -x^2 * erfcx'(x) as a function of u = 1/x and of t = 1/x^2. */
#   g_u = 2/sqrt(pi)/x^2 - (2/x^3)*exp(1/x^2)*erfc(1/x);
#   g_t = 2/sqrt(pi)/x - (2/(x*sqrt(x)))*exp(1/x)*erfc(1/sqrt(x));
#   /* Asymptotic series of G in t, used for the float64 tail x >= 12. */
#   g_asymp_t = 1/sqrt(pi) * (1 - 3/2*x + 15/4*x^2 - 105/8*x^3 + 945/16*x^4
#       - 10395/32*x^5 + 135135/64*x^6 - 2027025/128*x^7 + 34459425/256*x^8
#       - 654729075/512*x^9 + 13749310575/1024*x^10 - 316234143225/2048*x^11
#       + 7905853580625/4096*x^12 - 213458046676875/8192*x^13
#       + 6190283353629375/16384*x^14 - 191898783962510625/32768*x^15);
#   /* erfcx'(x) on [0, 1] as a function of s = x - 0.5. */
#   d_0_1 = 2*(x + 0.5)*exp((x + 0.5)^2)*erfc(x + 0.5) - 2/sqrt(pi);
#   show("_ERFC32_G_1_2", fpminimax(g_u(x + 0.75), 7, [|SG...|],
#        [-0.25; 0.25], relative, floating), 7);
#   show("_ERFC32_G_2_8", fpminimax(g_u(x + 0.3125), 7, [|SG...|],
#        [-0.1875; 0.1875], relative, floating), 7);
#   show("_ERFC32_G_8_INF", fpminimax(g_t, 3, [|SG...|],
#        [1b-20; 1/64], relative, floating), 3);
#   show("_ERFC64_G_1_2", fpminimax(g_u(x + 0.75), 17, [|D...|],
#        [-0.25; 0.25], relative, floating), 17);
#   show("_ERFC64_G_2_4", fpminimax(g_u(x + 0.375), 15, [|D...|],
#        [-0.125; 0.125], relative, floating), 15);
#   show("_ERFC64_G_4_12", fpminimax(g_u, 14, [|D...|],
#        [1/12; 0.25], relative, floating), 14);
#   show("_ERFC64_G_12_INF", fpminimax(g_asymp_t, 8, [|D...|],
#        [0; 1/144], relative, floating), 8);
#   show("_ERFC32_G_0_1", fpminimax(d_0_1, 9, [|SG...|],
#        [-0.5; 0.5], relative, floating), 9);
#   show("_ERFC64_G_0_1", fpminimax(d_0_1, 18, [|D...|],
#        [-0.5; 0.5], relative, floating), 18);
_ERFC32_G_1_2 = (
    float.fromhex("0x1.5d8fccp-2"),
    float.fromhex("-0x1.3c8f3ap-2"),
    float.fromhex("0x1.308bc6p-3"),
    float.fromhex("0x1.40234ep-7"),
    float.fromhex("-0x1.c6c0c8p-4"),
    float.fromhex("0x1.28d0aap-3"),
    float.fromhex("-0x1.fb5704p-4"),
    float.fromhex("0x1.07c794p-4"),
)

_ERFC32_G_2_8 = (
    float.fromhex("0x1.fcc322p-2"),
    float.fromhex("-0x1.6c3ccap-2"),
    float.fromhex("-0x1.8a59f4p-3"),
    float.fromhex("0x1.67556ep-1"),
    float.fromhex("-0x1.536338p-1"),
    float.fromhex("-0x1.be92bcp-3"),
    float.fromhex("0x1.a70c6ep0"),
    float.fromhex("-0x1.0e2840p1"),
)

_ERFC32_G_8_INF = (
    float.fromhex("0x1.20dd74p-1"),
    float.fromhex("-0x1.b14786p-1"),
    float.fromhex("0x1.0d8cd0p1"),
    float.fromhex("-0x1.9bd596p2"),
)

# Double-precision (`float64`) Sollya `fpminimax` polynomial coefficients for
# `G(x) = -x^2 * erfcx'(x)` on `x >= 1`, in ascending degree order (`u = 1 / x`,
# `t = u^2 = 1 / x^2`):
#
#   _ERFC64_G_1_2:    `x in [1, 2)`,    degree 17 in `s = u - 0.75`
#                                       (rel err 2.98e-17, ~0.27 ULP)
#   _ERFC64_G_2_4:    `x in [2, 4)`,    degree 15 in `s = u - 0.375`
#                                       (rel err 1.97e-17, ~0.18 ULP)
#   _ERFC64_G_4_12:   `x in [4, 12)`,   degree 14 in `u`
#                                       (rel err 7.20e-18, ~0.07 ULP)
#   _ERFC64_G_12_INF: `x in [12, inf)`, degree 8 in `t = u^2`
#                                       (rel err 1.36e-17, ~0.12 ULP)
_ERFC64_G_1_2 = (
    float.fromhex("0x1.5d8fcc4ff299fp-2"),
    float.fromhex("-0x1.3c8f3c0e9713fp-2"),
    float.fromhex("0x1.308b5d651d692p-3"),
    float.fromhex("0x1.40539e936fdf9p-7"),
    float.fromhex("-0x1.c6a3b77e2cbdcp-4"),
    float.fromhex("0x1.28273e28c6857p-3"),
    float.fromhex("-0x1.fb639fa6102fap-4"),
    float.fromhex("0x1.23e180fbac86ep-4"),
    float.fromhex("-0x1.333520f71b863p-7"),
    float.fromhex("-0x1.68564659148d6p-5"),
    float.fromhex("0x1.3f0ccd3895528p-4"),
    float.fromhex("-0x1.69af0c554c522p-4"),
    float.fromhex("0x1.3a785279c5119p-4"),
    float.fromhex("-0x1.94c906e0eee1bp-5"),
    float.fromhex("0x1.0d917547d50b9p-6"),
    float.fromhex("0x1.4af285199dcf0p-6"),
    float.fromhex("-0x1.2cdbbf2120b51p-4"),
    float.fromhex("0x1.61423e2bd69edp-4"),
)

_ERFC64_G_2_4 = (
    float.fromhex("0x1.e564643a22ec1p-2"),
    float.fromhex("-0x1.7d2245a3fab2dp-2"),
    float.fromhex("-0x1.3a22e2716d9b4p-4"),
    float.fromhex("0x1.11bd8bc5ed735p-1"),
    float.fromhex("-0x1.4dfa3c7c26ea4p-1"),
    float.fromhex("0x1.b7b2f445357eep-3"),
    float.fromhex("0x1.67b14c9ce96fap-1"),
    float.fromhex("-0x1.a31adbd26f844p0"),
    float.fromhex("0x1.ccbc138251430p0"),
    float.fromhex("-0x1.83c9b3a976bcbp-2"),
    float.fromhex("-0x1.71000220096a0p1"),
    float.fromhex("0x1.bdbeaa0846d38p2"),
    float.fromhex("-0x1.24148b88fc5a6p3"),
    float.fromhex("0x1.33e7aee6b38dbp2"),
    float.fromhex("0x1.b80e9a3646cb3p3"),
    float.fromhex("-0x1.34f75fb625e77p5"),
)

_ERFC64_G_4_12 = (
    float.fromhex("0x1.20dd7504362c6p-1"),
    float.fromhex("0x1.c76beba6f246cp-28"),
    float.fromhex("-0x1.b14c47c4fdd1fp-1"),
    float.fromhex("0x1.2769f70118d67p-15"),
    float.fromhex("0x1.0ead78b59c354p1"),
    float.fromhex("0x1.5842cd55f8dc8p-6"),
    float.fromhex("-0x1.ed7ea8a970b71p2"),
    float.fromhex("0x1.a8269ce322d02p1"),
    float.fromhex("0x1.8f40d19cd55a0p2"),
    float.fromhex("0x1.4dcb55c4285fbp7"),
    float.fromhex("-0x1.d85fd73493477p9"),
    float.fromhex("0x1.3224fc40f7246p11"),
    float.fromhex("-0x1.cabf0ba59224fp11"),
    float.fromhex("0x1.85ccc633f9b37p11"),
    float.fromhex("-0x1.26fdae0850b99p10"),
)

_ERFC64_G_12_INF = (
    float.fromhex("0x1.20dd750429b6dp-1"),
    float.fromhex("-0x1.b14c2f863e731p-1"),
    float.fromhex("0x1.0ecf9db3a64c3p1"),
    float.fromhex("-0x1.d9eb537ff0ebfp2"),
    float.fromhex("0x1.0a943fa7b85f5p5"),
    float.fromhex("-0x1.6e827a9457943p7"),
    float.fromhex("0x1.28f33e00c3a43p10"),
    float.fromhex("-0x1.0b1a528ddb599p13"),
    float.fromhex("0x1.94b1159b4e5f9p15"),
)

# High + low floating-point splits of `1 / sqrt(pi)`. When reconstructing
# `erfcx(x) = (1 / sqrt(pi) + 0.5 * erfcx'(x)) / x` on `x >= 1`, grouping the
# sum as `c1_hi + (c1_lo + 0.5 * erfcx'(x))` folds the low bits of `1 / sqrt(pi)`
# into the smaller `0.5 * erfcx'(x)` term before adding `c1_hi`, avoiding a half-ULP
# rounding loss near `x = 1`.
_INV_SQRT_PI_F32_HI = np.float32(float.fromhex("0x1.20dd76p-1"))
_INV_SQRT_PI_F32_LO = np.float32(float.fromhex("-0x1.f7ac92p-26"))
_INV_SQRT_PI_F64_HI = np.float64(float.fromhex("0x1.20dd750429b6dp-1"))


def _erfcx_from_grad_large(ax: Array, grad_large: Array) -> Array:
  """Reconstructs `erfcx(ax)` from its derivative `grad_large = erfcx'(ax)` on `ax >= 1`.

  Rearranging the differential equation `erfcx'(ax) = 2 * ax * erfcx(ax) - 2 /
  sqrt(pi)`
  gives:

    erfcx(ax) = (1 / sqrt(pi) + 0.5 * erfcx'(ax)) / ax.

  Because `grad_large` is negative with `|0.5 * grad_large| <= 0.1366 < 0.5642
  ~= 1 / sqrt(pi)`
  for all `ax >= 1`, the numerator is bounded away from zero (`>= 0.4275`) and
  free of catastrophic cancellation.

  In `float32`, we also scale both numerator and denominator by `2^-64` when
  `ax > 2^64`. On TPU v4, v5e, and v5p, `float32` division uses a Newton-Raphson
  refinement `y_n + y_n * (1 - d * y_n)` starting from a reciprocal estimate;
  when the divisor `d = ax > 2^111`, the intermediate correction term underflows
  below the normal `float32` threshold `2^-126` and flushes to zero, leaving
  ~200 ULPs of error. Scaling numerator and denominator by the exact power of
  two `2^-64` keeps the divisor in the normal range without changing the
  mathematical quotient.
  """
  if ax.dtype == np.float32:
    rem = _INV_SQRT_PI_F32_LO + 0.5 * grad_large
    num = _INV_SQRT_PI_F32_HI + rem
    scale = select(ax > 2.0**64, full_like(ax, 2.0**-64), full_like(ax, 1.0))
    return (num * scale) / (ax * scale)
  num = _const(grad_large, _INV_SQRT_PI_F64_HI) + 0.5 * grad_large
  return num / ax


def _erfcx_g_large(ax: Array) -> tuple[Array, Array]:
  """Returns `(t, g)` where `t = 1 / ax^2` and `g = -ax^2 * erfcx'(ax)` for `ax >= 1`."""
  u = reciprocal(ax)
  t = square(u)
  if ax.dtype == np.float32:
    g_1_2 = polynomial(u - 0.75, _ERFC32_G_1_2, unroll=len(_ERFC32_G_1_2))
    g_2_8 = polynomial(u - 0.3125, _ERFC32_G_2_8, unroll=len(_ERFC32_G_2_8))
    g_8_inf = polynomial(t, _ERFC32_G_8_INF, unroll=len(_ERFC32_G_8_INF))
    g = select(ax < 2.0, g_1_2, select(ax < 8.0, g_2_8, g_8_inf))
  elif ax.dtype == np.float64:
    g_1_2 = polynomial(u - 0.75, _ERFC64_G_1_2, unroll=len(_ERFC64_G_1_2))
    g_2_4 = polynomial(u - 0.375, _ERFC64_G_2_4, unroll=len(_ERFC64_G_2_4))
    g_4_12 = polynomial(u, _ERFC64_G_4_12, unroll=len(_ERFC64_G_4_12))
    g_12_inf = polynomial(t, _ERFC64_G_12_INF, unroll=len(_ERFC64_G_12_INF))
    g = select(
        ax < 2.0,
        g_1_2,
        select(ax < 4.0, g_2_4, select(ax < 12.0, g_4_12, g_12_inf)),
    )
  else:
    raise TypeError(f"Unsupported dtype: {ax.dtype}")
  return t, g


def erfcx_grad_large(ax: Array) -> Array:
  """Computes the derivative `erfcx'(ax)` for `ax >= 1`.

  We write `erfcx'(ax) = -(1 / ax^2) * G(ax)` where `G(ax) = -ax^2 * erfcx'(ax)`
  is positive and bounded in `[0.2731, 0.5642)` on `[1, inf)`. We evaluate
  `G(ax)` using piecewise Sollya minimax polynomials in `u = 1 / ax` on the
  near-field intervals (`[1, 2)` and `[2, 8)` in `float32`; `[1, 2)`, `[2, 4)`,
  and `[4, 12)` in `float64`) and in `t = u^2 = 1 / ax^2` on the asymptotic tail
  (`[8, inf)` in `float32`; `[12, inf)` in `float64`).
  """
  t, g = _erfcx_g_large(ax)
  return -t * g


def erfcx_large(ax: Array) -> Array:
  """Computes `erfcx(ax)` for `ax >= 1` via `erfcx_grad_large` and `_erfcx_from_grad_large`."""
  ax = lax_max(ax, _const(ax, 1.0))
  return _erfcx_from_grad_large(ax, erfcx_grad_large(ax))


# Sollya `fpminimax` polynomial coefficients for `erfcx'(x)` on `x in [0, 1]`,
# expanded in the centered variable `s = x - 0.5` in ascending degree order for
# `lax.polynomial`.
#
# Why `[0, 1]` uses a direct polynomial for `erfcx'(x)`:
# On `[0, 1]`, `2 * x * erfcx(x)` increases from `0` to `0.8552` while
# `2 / sqrt(pi) ~= 1.1284`. Near `x = 1`, subtracting `2 * x * erfcx(x) - 2 / sqrt(pi)`
# cancels ~75% of the magnitude, which would magnify any rounding error in
# `erfcx(x)` by ~4x. Evaluating a minimax polynomial directly in `s = x - 0.5`
# avoids that cancellation (`_ERFC32_G_0_1`: degree 9, rel err ~0.27 ULP;
# `_ERFC64_G_0_1`: degree 18, rel err ~0.18 ULP).
_ERFC32_G_0_1 = (
    float.fromhex("-0x1.067f26p-1"),
    float.fromhex("0x1.6ff85ep-1"),
    float.fromhex("-0x1.550206p-1"),
    float.fromhex("0x1.fc9b52p-2"),
    float.fromhex("-0x1.47907ap-2"),
    float.fromhex("0x1.7976dcp-3"),
    float.fromhex("-0x1.8c2f18p-4"),
    float.fromhex("0x1.87796cp-5"),
    float.fromhex("-0x1.8d572ap-6"),
    float.fromhex("0x1.3d5cd4p-7"),
)

_ERFC64_G_0_1 = (
    float.fromhex("-0x1.067f263ec85e7p-1"),
    float.fromhex("0x1.6ff861544dbffp-1"),
    float.fromhex("-0x1.55021bd369e1dp-1"),
    float.fromhex("0x1.fc9a0570ff834p-2"),
    float.fromhex("-0x1.4786f912f56aep-2"),
    float.fromhex("0x1.79973b6973c28p-3"),
    float.fromhex("-0x1.8e2e1451bc1b5p-4"),
    float.fromhex("0x1.85b049641853ap-5"),
    float.fromhex("-0x1.65a4048dd402ep-6"),
    float.fromhex("0x1.363682a7210e9p-7"),
    float.fromhex("-0x1.ffb88ced16238p-9"),
    float.fromhex("0x1.934c7b5caf5ecp-10"),
    float.fromhex("-0x1.30ef3121b1896p-11"),
    float.fromhex("0x1.bbfeb6475aa4cp-13"),
    float.fromhex("-0x1.382eba5243836p-14"),
    float.fromhex("0x1.a581276bdfc2ap-16"),
    float.fromhex("-0x1.1299bedad2447p-17"),
    float.fromhex("0x1.9078339c927cep-19"),
    float.fromhex("-0x1.0f0a671941893p-20"),
)


# Sollya `fpminimax` polynomial tables for `erfc(x)` (`bfloat16`, `float32`,
# and `float64`), listed in ascending degree order for `lax.polynomial`:
#
#   _ERFC_BF16_P_COEFFS: degree 6 in `u = 1 / (|x| + 2)` on `u in [0.05, 0.5]`
#                        (`|x| in [0, 18]`), approximating
#                        `erfc(|x|) * exp(x^2) * (|x| + 2)` for `bfloat16`
#                        (<= 0.5 ULP across all `bfloat16` inputs).
#   _ERFC32_EXP_COEFFS:  degree 5 in `r` on `|r| <= 0.5 * ln(2)`, approximating
#                        `(exp(r) - 1 - r) / r^2` for `float32`.
#   _ERFC32_P_COEFFS:    degree 1..10 coefficients of `P_10(z)` in
#                        `z = 1 / (|x| + 2) - 9/32` on `|x| in [0, 10.125]`,
#                        approximating `erfc(|x|) * exp(x^2) * (|x| + 2)` for
#                        `float32`. The constant and linear terms `d_0 + d_1 * z`
#                        are split into an exact 8-bit sum
#                        `s0 = 0.26953125 + 3 * u_hi` plus residual
#                        `-0.0013709686762269 + 3 * u_lo` (where
#                        `d_0 = 0.26953125 + 0.84375 - 0.0013709686762269`).
#   _ERFC64_EXP_COEFFS:  degree 9 in `r` on `|r| <= 0.5 * ln(2)`, approximating
#                        `(exp(r) - 1 - r) / r^2` for `float64` (max error
#                        `r^2 * |P_9(r) - f(r)| <= 2^-56.2`).
#   _ERFC64_P_COEFFS:    degree 22 in `z = 1 / (|x| + 4) - 9/64` on
#                        `|x| in [0, 27.25]`, approximating
#                        `erfc(|x|) * exp(x^2) * (|x| + 4)` for `float64`.
#
# They were generated by this Sollya script:
#
#   prec = 300!;
#   display = hexadecimal!;
#   procedure show(name, p, n) {
#     var i;
#     print("#", name);
#     for i from 0 to n do print(coeff(p, i));
#   };
#   /* Remainder `(exp(r) - 1 - r) / r^2` on `|r| <= 0.5 * ln(2)`. */
#   f_exp = (exp(x) - 1 - x) / (x^2);
#   I_exp = [-0.5 * log(2); 0.5 * log(2)];
#   show("_ERFC32_EXP_COEFFS", fpminimax(f_exp, 5, [|SG...|], I_exp, relative, floating), 5);
#   show("_ERFC64_EXP_COEFFS", fpminimax(f_exp, 9, [|D...|], I_exp, relative, floating), 9);
#   /* Scaled erfc `g(u) = erfc(1/u - c) * exp((1/u - c)^2) / u`. */
#   g_bf16 = erfc(1/x - 2) * exp((1/x - 2)^2) / x;
#   show("_ERFC_BF16_P_COEFFS", fpminimax(g_bf16, 6, [|SG...|], [0.05; 0.5], relative, floating), 6);
#   g32 = erfc(1/(x + 9/32) - 2) * exp((1/(x + 9/32) - 2)^2) / (x + 9/32);
#   p32 = fpminimax(g32, 10, [|D, D, SG...|], [1/(10.125 + 2) - 9/32; 7/32], relative, floating);
#   print("# _ERFC32_P_COEFFS (d0 - (0.26953125 + 0.84375) =", coeff(p32, 0) - (0.26953125 + 0.84375), ")");
#   print(coeff(p32, 1) - 3);
#   for i from 2 to 10 do print(coeff(p32, i));
#   g64 = erfc(1/(x + 9/64) - 4) * exp((1/(x + 9/64) - 4)^2) / (x + 9/64);
#   show("_ERFC64_P_COEFFS", fpminimax(g64, 22, [|D...|], [1/(27.25 + 4) - 9/64; 7/64], relative, floating), 22);
_ERFC_F16_P_COEFFS = (
    float.fromhex("0x1.1bd04p-1"),
    float.fromhex("0x1.53a09ep0"),
    float.fromhex("0x1.b86166p-2"),
    float.fromhex("0x1.107e72p3"),
    float.fromhex("-0x1.8eef68p2"),
)

_ERFC_BF16_P_COEFFS = (
    float.fromhex("0x1.20f88ap-1"),
    float.fromhex("0x1.1e4d82p0"),
    float.fromhex("0x1.1470f4p1"),
    float.fromhex("0x1.208a6p0"),
    float.fromhex("0x1.5d5cccp3"),
    float.fromhex("-0x1.48ea3ep4"),
    float.fromhex("0x1.3ecb4cp3"),
)

# Sollya `fpminimax` tables for default-accuracy `erfc` (`erfc_default_impl`),
# pinning `0.5 * P_8(0.5) == 1.0` (`float32`) and `0.25 * P_20(7/64) == 1.0`
# (`float64`) so `erfc(+-0.0) == 1.0` is exact:
#
#   prec = 200!;
#   display = hexadecimal!;
#   g_f16 = erfc(1/x - 2) * exp((1/x - 2)^2) / x;
#   p_f16 = fpminimax(g_f16, 4, [|SG...|], [1/6.125; 0.5], relative, floating);
#   q_exp32 = fpminimax(exp(x), [|1, 2, 3, 4, 5|], [|SG...|], [-0.560; 0.0171],
#                       relative, floating, 1);
#   f_expm1 = (exp(x) - 1) / x;
#   p_exp32 = fpminimax(f_expm1, 2, [|SG...|], [-0.0411; 0.0411],
#                       relative, floating);
#   c32 = 2 + 1b-24;
#   w32 = (0.5 - x) / g_f16;
#   q32 = remez(((g_f16 - c32) / (x - 0.5)) * w32, 7, [1/12.5; 0.5 - 1b-16], w32);
#   p32_def = fpminimax(g_f16, 8, [|SG...|], [1/12.5; 0.5], relative, floating,
#                       0, c32 + (x - 0.5) * q32);
#   g64_def = erfc(1/(x + 9/64) - 4) * exp((1/(x + 9/64) - 4)^2) / (x + 9/64);
#   w64 = (7/64 - x) / g64_def;
#   q64 = remez(((g64_def - 4) / (x - 7/64)) * w64, 19,
#               [1/31.5 - 9/64; 7/64 - 1b-12], w64);
#   p64_def = fpminimax(g64_def, 20, [|D...|], [1/31.5 - 9/64; 7/64],
#                       relative, floating, 0, 4 + (x - 7/64) * q64);
_ERFC32_DEFAULT_EXPM1_COEFFS = (
    float.fromhex("0x1.ffffdcp-1"),
    float.fromhex("0x1.fff74cp-2"),
    float.fromhex("0x1.54b96ap-3"),
    float.fromhex("0x1.4c12fcp-5"),
    float.fromhex("0x1.a1309cp-8"),
)

_ERFC32_DEFAULT_EXP_EXPM1_COEFFS = (
    float.fromhex("0x1p0"),
    float.fromhex("0x1.0006ecp-1"),
    float.fromhex("0x1.55590ap-3"),
)

_ERFC32_DEFAULT_P_COEFFS = (
    float.fromhex("0x1.20dbaap-1"),
    float.fromhex("0x1.211dccp0"),
    float.fromhex("0x1.f2899p0"),
    float.fromhex("0x1.9be248p1"),
    float.fromhex("-0x1.6b085cp-1"),
    float.fromhex("0x1.14e918p4"),
    float.fromhex("-0x1.eaf5dep5"),
    float.fromhex("0x1.1f07a4p6"),
    float.fromhex("-0x1.dbcdc2p4"),
)

_ERFC64_DEFAULT_P_COEFFS = (
    float.fromhex("0x1.3b35b725c71c9p0"),
    float.fromhex("0x1.32b1220caf326p3"),
    float.fromhex("0x1.02bfff5f3feafp6"),
    float.fromhex("0x1.792f364cf3722p8"),
    float.fromhex("0x1.d671b460d2ca1p10"),
    float.fromhex("0x1.eccbcbc614047p12"),
    float.fromhex("0x1.a267a162a81dap14"),
    float.fromhex("0x1.0a65e67f66f5fp16"),
    float.fromhex("0x1.8c9813aae5362p16"),
    float.fromhex("-0x1.cb16abe662d0ep15"),
    float.fromhex("-0x1.69f6c43799811p19"),
    float.fromhex("-0x1.483e78debce58p20"),
    float.fromhex("0x1.0d057c73f122ep21"),
    float.fromhex("0x1.9097e4bb63cfep23"),
    float.fromhex("0x1.5046e1bd2b525p21"),
    float.fromhex("-0x1.705b30825761bp26"),
    float.fromhex("-0x1.b13dbfa6eae42p26"),
    float.fromhex("0x1.3d91bca0a2241p29"),
    float.fromhex("0x1.2cfaaf76308ddp30"),
    float.fromhex("-0x1.9ffb97e2d173cp31"),
    float.fromhex("-0x1.f04d8174da975p32"),
)

_ERFC32_EXP_COEFFS = (
    float.fromhex("0x1p-1"),
    float.fromhex("0x1.555556p-3"),
    float.fromhex("0x1.555502p-5"),
    float.fromhex("0x1.110ff2p-7"),
    float.fromhex("0x1.6d2ep-10"),
    float.fromhex("0x1.a26762p-13"),
)

_ERFC32_P_COEFFS = (
    float.fromhex("0x1.188b2750ded4p-5"),
    float.fromhex("0x1.29fd9ap2"),
    float.fromhex("0x1.965e7p0"),
    float.fromhex("-0x1.96233ap2"),
    float.fromhex("-0x1.044916p2"),
    float.fromhex("0x1.ce4e76p3"),
    float.fromhex("0x1.e11eb8p1"),
    float.fromhex("-0x1.38a2c2p5"),
    float.fromhex("0x1.4e9696p3"),
    float.fromhex("0x1.23053ap6"),
)

_ERFC64_EXP_COEFFS = (
    float.fromhex("0x1.0000000000001p-1"),
    float.fromhex("0x1.555555555554dp-3"),
    float.fromhex("0x1.5555555553ce8p-5"),
    float.fromhex("0x1.11111111144ccp-7"),
    float.fromhex("0x1.6c16c178a5881p-10"),
    float.fromhex("0x1.a01a018c37ffbp-13"),
    float.fromhex("0x1.a019b8cc9b308p-16"),
    float.fromhex("0x1.71de59f195219p-19"),
    float.fromhex("0x1.2891bbf3b837bp-22"),
    float.fromhex("0x1.aeabc675c0f0ep-26"),
)

_ERFC64_P_COEFFS = (
    float.fromhex("0x1.3b35b725c71c9p0"),
    float.fromhex("0x1.32b1220caf334p3"),
    float.fromhex("0x1.02bfff5f3fe7fp6"),
    float.fromhex("0x1.792f364cf2bf2p8"),
    float.fromhex("0x1.d671b460d50a6p10"),
    float.fromhex("0x1.eccbcbc6676f8p12"),
    float.fromhex("0x1.a267a1621a15p14"),
    float.fromhex("0x1.0a65e66dbb26bp16"),
    float.fromhex("0x1.8c9813e99dd39p16"),
    float.fromhex("-0x1.cb166a8559fc2p15"),
    float.fromhex("-0x1.69f6ca898d5fbp19"),
    float.fromhex("-0x1.4840bb219b572p20"),
    float.fromhex("0x1.0d05b96ac3e65p21"),
    float.fromhex("0x1.90c9bf57df9d1p23"),
    float.fromhex("0x1.50d8d64c1d6a4p21"),
    float.fromhex("-0x1.730f43b9d7b97p26"),
    float.fromhex("-0x1.b517e3f5771d2p26"),
    float.fromhex("0x1.548119f231c9cp29"),
    float.fromhex("0x1.435770404322ep30"),
    float.fromhex("-0x1.3c9803f174af6p32"),
    float.fromhex("-0x1.7829928210c2cp33"),
    float.fromhex("0x1.b7c7b8628e9a6p34"),
    float.fromhex("0x1.269081bc9685bp36"),
)


def _erfc_f16(x: Array) -> Array:
  """Computes `erfc(x)` in `float16` to <= 1.0 ULP across all inputs."""
  xf = convert_element_type(x, np.float32)
  ax = abs(xf)
  e = exp(-(ax * ax))
  u = reciprocal(ax + 2.0)
  p = polynomial(u, _ERFC_F16_P_COEFFS, unroll=len(_ERFC_F16_P_COEFFS))
  res_pos = e * (u * p)
  res = select(xf < 0.0, 2.0 - res_pos, res_pos)
  return convert_element_type(res, np.float16)


def _erfc_bf16(x: Array) -> Array:
  """Computes `erfc(x)` in `bfloat16` to <= 0.5 ULPs across all inputs."""
  xf = convert_element_type(x, np.float32)
  ax = abs(xf)
  e = exp(-(ax * ax))
  u = reciprocal(ax + 2.0)
  p = polynomial(u, _ERFC_BF16_P_COEFFS, unroll=len(_ERFC_BF16_P_COEFFS))
  res_pos = e * (u * p)
  res = select(xf < 0.0, 2.0 - res_pos, res_pos)
  return convert_element_type(res, dtypes.bfloat16)


def _erfc_default_f32(x: Array, *, use_exp2: bool = True) -> Array:
  """Computes default-accuracy `erfc(x)` in `float32`."""
  # Round `safe_x` to ULP 2^-6 (`0x1.8p+17`) for `exp2` so `s_hi * (-23/16)` is
  # exact in `float32`, or to ULP 2^-8 (`0x1.8p+15`) for `exp`.
  magic = 196608.0 if use_exp2 else 49152.0
  safe_x = lax_min(abs(x), _const(x, 10.5))
  x_hi = (safe_x + magic) - magic
  x_lo = safe_x - x_hi
  s_hi = x_hi * x_hi
  s_lo = x_lo * (safe_x + x_hi)
  if use_exp2:
    # `s_hi <= 110.25` is `m^2 * 2^-12` (`m <= 672`) and `-0x1.7p+0 = -23 * 2^-4`,
    # so `s_hi * (-0x1.7p+0)` has numerator `<= 10,386,432 < 2^24` (exact in f32).
    e_hi = exp2(s_hi * float.fromhex("-0x1.7p+0"))
    # Remainder `-x^2 - (-s_hi * 0x1.7p+0 * ln(2))` in `[-0.560, 0.0171]`.
    ns = s_hi * float.fromhex("-0x1.d7fb18p-9") - s_lo
    expm1_coeffs = _ERFC32_DEFAULT_EXPM1_COEFFS
  else:
    e_hi = exp(-s_hi)
    ns = -s_lo
    expm1_coeffs = _ERFC32_DEFAULT_EXP_EXPM1_COEFFS
  e_lo_m1 = ns * polynomial(ns, expm1_coeffs, unroll=len(expm1_coeffs))
  u = reciprocal(safe_x + 2.0)
  p = polynomial(
      u, _ERFC32_DEFAULT_P_COEFFS, unroll=len(_ERFC32_DEFAULT_P_COEFFS)
  )
  up = u * p
  res_pos = e_hi * (up + up * e_lo_m1)
  return select(x < 0.0, 2.0 - res_pos, res_pos)


def _erfc_default_f64(x: Array) -> Array:
  """Computes default-accuracy `erfc(x)` in `float64`."""
  magic = float.fromhex("0x1.8p+31")  # Rounds [0, 64) to ULP 2^-21.
  safe_x = lax_min(abs(x), _const(x, 27.5))
  x_hi = (safe_x + magic) - magic
  x_lo = safe_x - x_hi
  s_hi = x_hi * x_hi
  s_lo = x_lo * (safe_x + x_hi)
  e_hi = exp(-s_hi)
  ns = -s_lo
  e_lo_m1 = ns * polynomial(ns, (1.0, 0.5, 1.0 / 6.0), unroll=3)
  u = reciprocal(safe_x + 4.0)
  z = u - 9.0 / 64.0
  p = polynomial(
      z, _ERFC64_DEFAULT_P_COEFFS, unroll=len(_ERFC64_DEFAULT_P_COEFFS)
  )
  up = u * p
  res_pos = e_hi * (up + up * e_lo_m1)
  return select(x < 0.0, 2.0 - res_pos, res_pos)


def _erfc_f32(x: Array) -> Array:
  """Computes `erfc(x)` in `float32` to <= 1.5 ULPs across all inputs."""
  magic = 49152.0  # 0x1.8p+15; rounds magnitudes in [0, 16) to ULP 2^-8.
  safe_x = lax_min(abs(x), _const(x, 10.125))

  # 1. Split `safe_x = x_hi + x_lo` so `x_hi` has <= 12 significant bits and
  # `s_hi = x_hi * x_hi` is exact in `float32`. Adding `49154 = 49152 + 2`
  # rounds `safe_x + 2` to ULP 2^-8 directly.
  d_hi = (safe_x + 49154.0) - magic
  x_hi = d_hi - 2.0
  x_lo = safe_x - x_hi
  s_hi = x_hi * x_hi
  s_lo = x_lo * (safe_x + x_hi)

  # 2. Cody-Waite range reduction `exp(-x^2) = 2^-k * exp(r)` with 16-bit
  # `ln2_hi` so `rem1 = k * ln2_hi - s_hi` is exact in `float32`.
  log2e = 1.4426950408889634
  k = lax_round(s_hi * log2e, rounding_method=RoundingMethod.TO_NEAREST_EVEN)
  ln2_hi = LN2_SPLIT_F32[0]
  ln2_lo = LN2_SPLIT_F32[1] + LN2_SPLIT_F32[2]
  rem1 = k * ln2_hi - s_hi
  rem2 = k * ln2_lo - s_lo
  r = rem1 + rem2

  r_hi = (r + magic) - magic
  e_hi = 1.0 + r_hi
  r_lo = (rem1 - r_hi) + rem2
  p_exp = polynomial(r, _ERFC32_EXP_COEFFS, unroll=len(_ERFC32_EXP_COEFFS))
  e_lo = r_lo + (r * r) * p_exp

  # 3. Reciprocal `u = 1 / (|x| + 2) = u_hi + u_lo` via one Newton-Raphson step.
  u0 = reciprocal(safe_x + 2.0)
  u_hi = (u0 + magic) - magic
  nr_rem = (1.0 - d_hi * u_hi) - x_lo * u_hi
  u_lo = u0 * nr_rem

  # 4. Degree-10 minimax polynomial `P_10(z)` with exact 8-bit leading part `s0`.
  s0 = 0.26953125 + 3.0 * u_hi
  z = (u_hi - 0.28125) + u_lo
  p_tail = polynomial(z, _ERFC32_P_COEFFS, unroll=len(_ERFC32_P_COEFFS))
  p_bot = (-0.0013709686762269 + 3.0 * u_lo) + z * p_tail
  p_sum = s0 + p_bot

  # 5. Exact 3-way product (`e_hi * u_hi * s0` is exact in `float32`).
  eu_hi = e_hi * u_hi
  eu_lo = e_hi * u_lo + e_lo * u0
  unscaled = eu_hi * s0 + (eu_hi * p_bot + eu_lo * p_sum)

  # 6. Scale by `2^-k` via `(155 - k) << 23` bitcast and `* 2^-28`.
  k_int = convert_element_type(k, np.int32)
  scale_bits = shift_left(_const(k_int, 155) - k_int, _const(k_int, 23))
  scale = bitcast_convert_type(scale_bits, np.float32)
  res_pos = (unscaled * scale) * 3.725290298461914e-09

  return select(x < 0.0, 2.0 - res_pos, res_pos)


def _erfc_f64(x: Array) -> Array:
  """Computes `erfc(x)` in `float64` to <= 2.0 ULPs across all inputs."""
  magic_x = float.fromhex("0x1.8p+31")  # Rounds [0, 64) to ULP 2^-21.
  magic_14 = float.fromhex("0x1.8p+38")  # Rounds [0, 2) to ULP 2^-14.
  safe_x = lax_min(abs(x), _const(x, 27.25))

  # 1. Split `safe_x = x_hi + x_lo` so `x_hi` and `d_hi` have <= 26 significant
  # bits (`s_hi = x_hi * x_hi` is exact in `float64`) while preserving tiny
  # `safe_x` in `x_lo`. `0x1.80000008p+31 = magic_x + 4.0`.
  d_hi = (safe_x + float.fromhex("0x1.80000008p+31")) - magic_x
  x_hi = d_hi - 4.0
  x_lo = safe_x - x_hi
  s_hi = x_hi * x_hi
  s_lo = x_lo * (safe_x + x_hi)

  # 2. Cody-Waite range reduction `exp(-x^2) = 2^-k * exp(r)` with 41-bit
  # `ln2_hi` so `rem1 = k * ln2_hi - s_hi` is exact for all `k <= 1071`.
  log2e = float.fromhex("0x1.71547652b82fep+0")
  k = lax_round(s_hi * log2e, rounding_method=RoundingMethod.TO_NEAREST_EVEN)
  ln2_hi = LN2_SPLIT_F64[0]
  ln2_lo = LN2_SPLIT_F64[1] + LN2_SPLIT_F64[2]
  rem1 = k * ln2_hi - s_hi
  rem2 = k * ln2_lo - s_lo
  r = rem1 + rem2

  r_hi = (r + magic_14) - magic_14
  e_hi = 1.0 + r_hi
  r_lo = (rem1 - r_hi) + rem2
  p_exp = polynomial(r, _ERFC64_EXP_COEFFS, unroll=len(_ERFC64_EXP_COEFFS))
  e_lo = r_lo + (r * r) * p_exp

  # 3. Reciprocal `u = 1 / (|x| + 4) = u_hi + u_lo` via one Newton-Raphson step.
  u0 = reciprocal(safe_x + 4.0)
  u_hi = (u0 + magic_14) - magic_14
  nr_rem = (1.0 - d_hi * u_hi) - x_lo * u_hi
  u_lo = u0 * nr_rem

  # 4. Degree-22 minimax polynomial `P_22(z)` with compensated low Horner steps
  # (`fast_two_sum` for `k = 3, 2, 1` where `exponent(c_k) >= exponent(t_k)`;
  # `two_sum` for `k = 0`).
  z_hi = u_hi - 9.0 / 64.0
  z = z_hi + u_lo
  c0, c1, c2, c3 = _ERFC64_P_COEFFS[:4]
  acc4 = polynomial(z, _ERFC64_P_COEFFS[4:], unroll=len(_ERFC64_P_COEFFS) - 4)

  p3_hi, p3_lo = fast_two_sum(c3, acc4 * z)

  p2_hi, p2_e = fast_two_sum(c2, p3_hi * z)
  p2_lo = p2_e + p3_lo * z

  p1_hi, p1_e = fast_two_sum(c1, p2_hi * z)
  p1_lo = p1_e + p2_lo * z

  # `p1_26` (25 bits) * `z_hi` (<= 11 bits) is exact (`36 <= 53` bits).
  p1_26, p1_26_rem = split_bits(p1_hi, 25)
  p1_rem = p1_26_rem + p1_lo
  t0_hi = p1_26 * z_hi
  t0_lo = p1_26 * u_lo + p1_rem * z

  p_hi, p_e = two_sum(c0, t0_hi)
  p_lo = p_e + t0_lo

  p_sum, p_sum_lo = fast_two_sum(p_hi, p_lo)
  p_26, p_26_rem = split_half_mask(p_sum)
  p_rem = p_26_rem + p_sum_lo

  # 5. Exact 3-way product (`e_hi * u_hi` <= 27 bits, `p_26` <= 26 bits ->
  # `e_hi * u_hi * p_26` <= 53 bits is exact).
  eu_hi = e_hi * u_hi
  eu_lo = e_hi * u_lo + e_lo * u0
  unscaled = eu_hi * p_26 + (eu_hi * p_rem + eu_lo * p_sum)

  # 6. Scale by `2^-k` via `(1079 - k) << 52` bitcast and `* 2^-56`.
  k_int = convert_element_type(k, np.int64)
  scale_bits = shift_left(_const(k_int, 1079) - k_int, _const(k_int, 52))
  scale = bitcast_convert_type(scale_bits, np.float64)
  res_pos = (unscaled * scale) * float.fromhex("0x1p-56")

  return select(x < 0.0, 2.0 - res_pos, res_pos)


def erfc_impl(x: Array) -> Array:
  """Computes high-accuracy `erfc(x)` across floating-point dtypes.

  We evaluate `erfc(|x|) = exp(-x^2) * u * P(z)` in a single branch-free regime
  across all inputs via the Mobius transformation `u = 1 / (|x| + c)`
  (`c = 2` for `bfloat16` and `float32`, `c = 4` for `float64`), and reflect
  `erfc(x) = 2 - erfc(|x|)` for `x < 0`. For `bfloat16`, upcasting to
  `float32` makes `x^2` exact and avoids double-word splitting; for `float32`
  and `float64`, we split `|x|` so `x_hi^2` is exact, reduce
  `exp(-x^2) = 2^-k * exp(r)` via Cody-Waite range reduction, and evaluate the
  minimax polynomial `P(z)` in double-word precision. Other floating-point
  dtypes (such as `float16` and `float8_*`) are evaluated via `float32`.
  """
  if x.dtype == dtypes.bfloat16:
    return _erfc_bf16(x)
  elif x.dtype == np.float32:
    return _erfc_f32(x)
  elif x.dtype == np.float64:
    return _erfc_f64(x)
  else:
    return convert_element_type(
        _erfc_f32(convert_element_type(x, np.float32)), x.dtype
    )


def erfc_default_impl(x: Array, *, use_exp2: bool = True) -> Array:
  """Computes default-accuracy `erfc(x)` preserving subnormal outputs.

  Evaluates `erfc(|x|) = exp(-x^2) * u * P(z)` via the single-branch Mobius
  transformation `u = 1 / (|x| + c)` and single-word polynomial evaluation,
  splitting only `x^2 = s_hi + s_lo` in `float32` and `float64` so rounding
  `x^2` does not magnify error by `O(x^2)` ULPs. In `float32`, `use_exp2=True`
  uses `exp2` with an exact `23 / 16` split of `log2(e)` to avoid `exp` range
  reduction error on GPU, while `use_exp2=False` uses `exp` directly.
  """
  if x.dtype == np.float16:
    return _erfc_f16(x)
  elif x.dtype == dtypes.bfloat16:
    return _erfc_bf16(x)
  elif x.dtype == np.float32:
    return _erfc_default_f32(x, use_exp2=use_exp2)
  elif x.dtype == np.float64:
    return _erfc_default_f64(x)
  else:
    return convert_element_type(
        _erfc_default_f32(
            convert_element_type(x, np.float32), use_exp2=use_exp2
        ),
        x.dtype,
    )


@custom_jvp
def _ndtr(x: Array) -> Array:
  """Computes the standard normal CDF `ndtr(x) = 0.5 * erfc(-x / sqrt(2))`.

  Calling `0.5 * erfc_impl(-x / sqrt(2))` directly would lose hundreds of ULPs
  for large negative `x`: `erfc(u)` contains a factor of `exp(-u^2)`, and
  rounding `u = -x / sqrt(2)` to a single float before squaring introduces
  `~x^2` ULPs of error in `exp(-u^2)`.

  To avoid that rounding error, `_ndtr` computes `u^2 = 0.5 * x^2` directly from
  `x` and evaluates `0.5 * erfc(-x / sqrt(2))` in three pieces:
  - `0.5 * _erfc_small(-x / sqrt(2))` for `|x| < sqrt(2)`,
  - `tail` for `x <= -sqrt(2)`, and
  - `1 - tail` for `x >= sqrt(2)`,
  where `tail = exp(-0.5 * x^2) * 0.5 * erfcx_large(|x| / sqrt(2))`. To evaluate
  `exp(-0.5 * x^2)` accurately, `_ndtr` splits `x^2 = s_hi + s_lo` into an
  exact sum of two floats via `_split_square(|x|)` and multiplies both parts by
  `-0.5`.
  """
  if x.dtype not in (np.float32, np.float64):
    return convert_element_type(
        _ndtr(convert_element_type(x, np.float32)), x.dtype
    )
  inv_sqrt2 = math.sqrt(0.5)
  ax = abs(x)
  u = -x * inv_sqrt2
  small = 0.5 * _erfc_small(clamp(_const(x, -1.0), u, _const(x, 1.0)))

  max_ax = 16.0 if x.dtype == np.float32 else 40.0
  ax_exp = lax_min(ax, _const(ax, max_ax))
  s_hi, s_lo = _split_square(ax_exp)
  y = 0.5 * erfcx_large(ax * inv_sqrt2)
  expm1_neg_lo = _expm1_lo(-0.5 * s_lo)
  tail = exp(-0.5 * s_hi, accuracy=AccuracyMode.HIGHEST) * (
      y + y * expm1_neg_lo
  )
  large = select(x < 0.0, tail, 1.0 - tail)
  return select(ax < math.sqrt(2.0), small, large)


@_ndtr.defjvp
def _ndtr_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  x = asarray(x)
  return _ndtr(x), (1.0 / math.sqrt(2.0 * math.pi)) * exp_neg_half_sq(x) * x_dot


@jit
def ndtr(x: Array) -> Array:
  return _ndtr(x)


@custom_jvp
def _log_ndtr(x: Array) -> Array:
  """Computes `log(ndtr(x))`.

  Evaluating `log(ndtr(x))` directly fails in both tails:
  - For `x > 0`, `ndtr(x)` approaches `1`, so rounding `ndtr(x)` to a single
    float before `log` loses precision (and rounds to `1`, giving `log(1) = 0`,
    for larger `x`).
  - For large negative `x`, `ndtr(x)` underflows into subnormals (below `-12.95`
    in `float32`, `-37.52` in `float64`) and then to `0` (`log(0) = -inf`, below
    `-14.12` in `float32`, `-38.47` in `float64`), even though `log(ndtr(x))`
    stays finite down to `x ~= -2.61e19` (`-1.90e154`).

  `_log_ndtr` therefore uses three cases:
  - `x > 0`: `log1p(-ndtr(-x))`, using `ndtr(x) = 1 - ndtr(-x)`.
  - `-sqrt(2) <= x <= 0`: `log(ndtr(x))` directly (`ndtr(x)` is in
    `[0.078, 0.5]`, so neither issue occurs).
  - `x < -sqrt(2)`: takes the log of
    `ndtr(x) = exp(-0.5 * x^2) * 0.5 * erfcx_large(|x| / sqrt(2))` directly:
      `log_ndtr(x) = -0.5 * x^2 + log(0.5 * erfcx_large(|x| / sqrt(2)))`
    Evaluating `-0.5 * x^2` as `-2 * (0.5 * |x|)^2` avoids `x^2` overflowing
    early when `|x|` is between `sqrt(fmax)` and `sqrt(2 * fmax)`. (Unlike
    `_ndtr`, `-0.5 * x^2` is added rather than exponentiated, and both terms
    have the same sign, so `_split_square` is not needed here.)
  """
  sqrt2 = math.sqrt(2.0)
  inv_sqrt2 = math.sqrt(0.5)
  ax = abs(x)

  # Both x > 0 (which needs ndtr(-x)) and -sqrt(2) <= x <= 0 (which needs
  # ndtr(x)) evaluate ndtr at -|x|.
  ndtr_neg_ax = _ndtr(-ax)

  tail = -2.0 * square(0.5 * ax) + log(
      0.5 * erfcx_large(ax * inv_sqrt2),
      accuracy=AccuracyMode.HIGHEST,
  )

  return select(
      x > 0.0,
      log1p(-ndtr_neg_ax, accuracy=AccuracyMode.HIGHEST),
      select(
          x >= -sqrt2,
          log(ndtr_neg_ax, accuracy=AccuracyMode.HIGHEST),
          tail,
      ),
  )


@custom_jvp
def _log_ndtr_grad_and_shift(x: Array) -> tuple[Array, Array]:
  """Returns `(r, q)` where `r = d/dx log(ndtr(x))` and `q = x + r`.

  By the chain rule, `r(x) = ndtr'(x) / ndtr(x)`, where
  `ndtr'(x) = exp(-0.5 * x^2) / sqrt(2 * pi)`:
  - For `x >= -sqrt(2)`, `ndtr(x) >= 0.078`, so dividing
    `exp_neg_half_sq(x) / (_ndtr(x) * sqrt(2 * pi))` directly is accurate.
  - For large negative `x`, both `ndtr'(x)` and `ndtr(x)` underflow to `0`
    (`0 / 0 = NaN`), even though their ratio grows like `|x|`. Substituting
    `ndtr(x) = exp(-0.5 * x^2) * 0.5 * erfcx(u)` (with `u = |x| / sqrt(2)`)
    cancels `exp(-0.5 * x^2)` analytically to give
    `r(x) = sqrt(2 / pi) / erfcx(u)`. Since
    `u * erfcx(u) = 1 / sqrt(pi) + 0.5 * erfcx'(u)` and
    `erfcx'(u) = -(1 / u^2) * G(u)`, writing `u * erfcx(u) = (1 - delta) / sqrt(pi)`
    with `delta = 0.5 * sqrt(pi) * (1 / u^2) * G(u)` gives
      `r(x) = |x| / (1 - delta)`,
    keeping the denominator in `[0.75, 1]` instead of dividing by `erfcx(u)`
    (which underflows near `-fmax`).

  Differentiating `r(x) = ndtr'(x) / ndtr(x)` (using `ndtr''(x) = -x * ndtr'(x)`)
  gives `r'(x) = -r(x) * (x + r(x))`. For large negative `x`,
  `r(x) ~= |x| + 1 / |x|`, so adding `x + r(x)` directly would cancel `-|x|`
  against `|x|`. Computing `q(x) = x + r(x)` algebraically on `x < -sqrt(2)` as
    `-|x| + |x| / (1 - delta) = sqrt(pi) * G(u) / (|x| * (1 - delta))`
  avoids that cancellation, and wrapping `(r, q)` in `@custom_jvp` gives a
  closed recurrence (`r' = -r * q`, `q' = 1 - r * q`) that never differentiates
  through the polynomials in `_ndtr` or `_erfcx_g_large`.
  """
  sqrt2 = math.sqrt(2.0)
  inv_sqrt2 = math.sqrt(0.5)
  max_ax = 16.0 if x.dtype == np.float32 else 40.0

  r_pos = exp_neg_half_sq(x) / (_ndtr(x) * math.sqrt(2.0 * math.pi))
  # For `x >= max_ax`, `r_pos` underflows to 0; zeroing `q_pos` avoids
  # `0 * inf = NaN` in `-r * q` at `x = +inf`.
  q_pos = select(x < max_ax, x + r_pos, full_like(x, 0.0))

  ax = -x
  t, g = _erfcx_g_large(ax * inv_sqrt2)
  denom = 1.0 - (0.5 * math.sqrt(math.pi)) * t * g
  r_neg = ax / denom
  if x.dtype == np.float32:
    # Avoid TPU v3-v5p float32 division subnormal flush when `ax * denom > 2^111`
    # (see `_erfcx_from_grad_large`).
    scale = select(ax > 2.0**64, full_like(ax, 2.0**-64), full_like(ax, 1.0))
    q_neg = (math.sqrt(math.pi) * g * scale) / ((ax * scale) * denom)
  else:
    q_neg = (math.sqrt(math.pi) * g) / (ax * denom)

  is_pos = x >= -sqrt2
  return select(is_pos, r_pos, r_neg), select(is_pos, q_pos, q_neg)


@_log_ndtr_grad_and_shift.defjvp
def _log_ndtr_grad_and_shift_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  x = asarray(x)
  r, q = _log_ndtr_grad_and_shift(x)
  rq = r * q
  return (r, q), (-rq * x_dot, (1.0 - rq) * x_dot)


@_log_ndtr.defjvp
def _log_ndtr_jvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  x = asarray(x)
  r, _ = _log_ndtr_grad_and_shift(x)
  return _log_ndtr(x), r * x_dot


@jit
def log_ndtr(x: Array) -> Array:
  if x.dtype not in (np.float32, np.float64):
    return convert_element_type(
        _log_ndtr(convert_element_type(x, np.float32)), x.dtype
    )
  return _log_ndtr(x)


def erfcx_impl(x: Array) -> Array:
  """Computes the scaled complementary error function `erfcx(x) = exp(x^2) * erfc(x)`.

  We divide the real line into three regimes:
  - `|x| < 1`: Evaluates `exp(x^2) * erfc(x)` using `_erfc_small(x)` for
    `erfc(x)` and `_split_square(x)` for `exp(x^2) = exp(s_hi) * (1 +
    expm1(s_lo))`.
    On `[-1, 1)`, `exp(x^2) <= e` and `erfc(x) in (0.1573, 1.8427]`, so both
    factors are well-scaled and free of underflow or overflow.
  - `x >= 1`: Evaluates `erfcx_large(x)`, which reconstructs `erfcx(x)` from
    the minimax approximation of `erfcx'(x)`. Because it never evaluates
    `exp(x^2)` or `erfc(x)` separately, it has no underflow gap where `erfc(x)`
    flushes to zero and remains accurate all the way up to `x ~ fmax`.
  - `x <= -1`: Uses the reflection identity `erfcx(x) = 2 * exp(x^2) -
  erfcx(|x|)`.
    For `x <= min_x` (`-9.382415` in `float32`, `-26.628735713751492` in
    `float64`), `2 * exp(x^2)` overflows to `+inf`, which we return directly.
  """
  ax = abs(x)
  max_exp_ax = 9.382414 if x.dtype == np.float32 else 26.62873571375149
  min_x = -9.382415 if x.dtype == np.float32 else -26.628735713751492

  ax_exp = lax_min(ax, _const(ax, max_exp_ax))
  s_hi, s_lo = _split_square(ax_exp)
  e = exp(s_hi, accuracy=AccuracyMode.HIGHEST)
  expm1_pos_lo = _expm1_lo(s_lo)

  y_small = _erfc_small(clamp(_const(x, -1.0), x, _const(x, 1.0)))
  small = e * (y_small + y_small * expm1_pos_lo)

  large_pos = erfcx_large(ax)
  large_neg = select(
      x <= min_x,
      full_like(x, np.inf),
      (e + e * expm1_pos_lo) * 2.0 - large_pos,
  )
  large = select(x < 0.0, large_neg, large_pos)
  return select(ax < 1.0, small, large)


def erfcx_grad_impl(x: Array, ans: Array) -> Array:
  """Computes `d/dx erfcx(x)` given `x` and `ans = erfcx(x)`.

  Uses three regimes to avoid subtraction cancellation everywhere on `(-inf,
  inf)`:
  - `x < 0`: Evaluates the ODE `2 * x * erfcx(x) - 2 / sqrt(pi)` directly using
    the primal value `ans = erfcx(x)`. Because `x < 0` and `ans > 1`, both
    `2 * x * ans` and `-2 / sqrt(pi)` are negative, so this is a same-sign
    addition with zero cancellation. (We replace `x` and `ans` with `0.0` when
    `x >= 0` before multiplying so `x = +inf, ans = 0` cannot produce
    `inf * 0 = NaN` in the unselected branch.)
  - `0 <= x < 1`: Evaluates the centered minimax polynomial `_ERFC32_G_0_1` or
    `_ERFC64_G_0_1` in `s = x - 0.5`, avoiding the ~2-bit cancellation in
    `2 * x * erfcx(x) - 2 / sqrt(pi)` as `x` approaches `1`.
  - `x >= 1`: Evaluates `erfcx_grad_large(x)`, which approximates
    `erfcx'(x) = -(1 / x^2) * G(x)` directly without cancellation.
  """
  is_neg = x < 0.0
  is_large = x >= 1.0
  zeros = full_like(x, 0.0)
  x_neg = select(is_neg, x, zeros)
  ans_neg = select(is_neg, ans, zeros)
  direct_neg = 2.0 * x_neg * ans_neg - 2.0 / math.sqrt(math.pi)
  s = clamp(_const(x, 0.0), x, _const(x, 1.0)) - 0.5
  coeffs_0_1 = _ERFC32_G_0_1 if x.dtype == np.float32 else _ERFC64_G_0_1
  grad_0_1 = polynomial(s, coeffs_0_1, unroll=len(coeffs_0_1))
  grad_large = erfcx_grad_large(lax_max(x, _const(x, 1.0)))
  return select(is_neg, direct_neg, select(is_large, grad_large, grad_0_1))


# Sollya `fpminimax` polynomial coefficients for `erfinv` and `erfcinv`,
# parameterized by `w = -log(q * (1 + |x|))` where `q = 1 - |x|` (`1 - |x|` for
# `erfinv(x)`, `min(y, 2 - y)` for `erfcinv(y)`), listed in ascending degree
# order for `lax.polynomial`:
#
#   float32 (degree 8):
#     _ERFINV32_P1: `w in [0, 3.5)`,         `out = x * P1(w - 1.75)`
#                                            (rel err 1.55e-8, ~0.26 ULP)
#     _ERFINV32_P2: `w in [3.5, 18.0625)`,   `out = sgn(x) * (sqrt(w) + P2(sqrt(w) - 3.0))`
#                                            (rel err 2.64e-8, ~0.44 ULP)
#     _ERFINV32_P3: `w in [18.0625, 102.6]`, `out = sgn(x) * (sqrt(w) + P3(sqrt(w) - 7.0))`
#                                            (rel err 3.37e-8, ~0.56 ULP)
#
#   float64 (degree 19):
#     _ERFINV64_P1: `w in [0, 4.5)`,         `out = x * P1(w - 2.25)`
#                                            (rel err 5.15e-17, ~0.46 ULP)
#     _ERFINV64_P2: `w in [4.5, 16.0)`,      `out = sgn(x) * (sqrt(w) + P2(sqrt(w) - 3.0))`
#                                            (rel err 1.04e-16, ~0.94 ULP)
#     _ERFINV64_P3: `w in [16.0, 37.0)`,     `out = sgn(x) * (sqrt(w) + P3(sqrt(w) - 5.0))`
#                                            (rel err 5.21e-17, ~0.47 ULP)
#     _ERFINV64_P4: `w in [37.0, 144.0)`,    `out = sgn(x) * (sqrt(w) + P4(sqrt(w) - 9.0))`
#                                            (rel err 3.65e-17, ~0.33 ULP)
#     _ERFINV64_P5: `w in [144.0, 743.8]`,   `out = sgn(x) * (sqrt(w) + P5(sqrt(w) - 19.5))`
#                                            (rel err 4.44e-17, ~0.40 ULP)
#
# Every `sqrt(w) - c` shift is exact by Sterbenz's lemma on its piece. For
# `erfinv(x)` (`has_far_tail=False`), `q = 1 - |x|` cannot go below `2^-24` in
# `float32` (`w <= 15.94 < 18.0625`) or `2^-53` in `float64` (`w <= 36.04 < 37`),
# so only the first 2 (`float32`) or 3 (`float64`) pieces are needed. The tables
# were generated by this Sollya script:
#
#   prec = 260!;
#   display = hexadecimal!;
#   procedure solve_y(w_val) {
#     var u, q_val, y, k, f_val, df_val, c, s_pi, lq;
#     s_pi = evaluate(sqrt(pi), 0);
#     u = evaluate(sqrt(-expm1(-x)), w_val);
#     q_val = evaluate(exp(-x), w_val) / (1 + u);
#     if (w_val < 4) then {
#       y = round((s_pi / 2) * u, 53, RN);
#       for k from 1 to 9 do {
#         f_val = evaluate(erfc(x), y) - q_val;
#         df_val = -(2 / s_pi) * evaluate(exp(-x * x), y);
#         y = round(y - f_val / df_val, 250, RN);
#       };
#     } else {
#       lq = evaluate(-x - log(1 + u), w_val);
#       y = evaluate(sqrt(-x), lq);
#       y = evaluate(sqrt(-(lq + log(s_pi * x))), y);
#       for k from 1 to 9 do {
#         c = evaluate(erfc(x), y);
#         f_val = evaluate(log(x), c) - lq;
#         df_val = -((2 / s_pi) * evaluate(exp(-x * x), y)) / c;
#         y = round(y - f_val / df_val, 250, RN);
#       };
#     };
#     return y;
#   };
#   procedure build_cheb(a, b, is_sqrt_w, shift_c, is_central, N) {
#     var nodes, vals, i, theta, t_i, w_i, y_i, u_i, v_i;
#     nodes = [||];
#     vals = [||];
#     for i from 0 to N do {
#       theta = evaluate(pi * (2 * i + 1) / (2 * (N + 1)), 0);
#       t_i = round(0.5 * (a + b) + 0.5 * (b - a) * evaluate(cos(x), theta), 250, RN);
#       if (is_sqrt_w) then w_i = (t_i + shift_c)^2 else w_i = t_i + shift_c;
#       if (w_i <= 1b-90) then {
#         v_i = evaluate(sqrt(pi) / 2, 0);
#       } else {
#         y_i = solve_y(w_i);
#         if (is_central) then {
#           u_i = evaluate(sqrt(-expm1(-x)), w_i);
#           v_i = y_i / u_i;
#         } else {
#           v_i = y_i;
#         };
#       };
#       nodes = t_i .: nodes;
#       vals = round(v_i, 250, RN) .: vals;
#     };
#     return interpolate(nodes, vals, 1b-150, [a; b]);
#   };
#   procedure show(name, p, n) {
#     var i;
#     print("#", name);
#     for i from 0 to n do print(coeff(p, i));
#   };
#   w32_1 = 3.5; z32_1 = evaluate(sqrt(w32_1), 0); z32_2 = 4.25;
#   z32_max = evaluate(sqrt(148 * log(2)), 0);
#   show("_ERFINV32_P1", fpminimax(build_cheb(1b-64 - 1.75, w32_1 - 1.75, false, 1.75, true, 32), 8, [|SG...|], [1b-64 - 1.75; w32_1 - 1.75], relative, floating), 8);
#   show("_ERFINV32_P2", fpminimax(build_cheb(z32_1 - 3.0, z32_2 - 3.0, true, 3.0, false, 32), 8, [|SG...|], [z32_1 - 3.0; z32_2 - 3.0], relative, floating, x + 3.0) - (x + 3.0), 8);
#   show("_ERFINV32_P3", fpminimax(build_cheb(z32_2 - 7.0, z32_max - 7.0, true, 7.0, false, 32), 8, [|SG...|], [z32_2 - 7.0; z32_max - 7.0], relative, floating, x + 7.0) - (x + 7.0), 8);
#   w64_1 = 4.5; z64_1 = evaluate(sqrt(w64_1), 0); z64_2 = 4.0;
#   z64_3 = evaluate(sqrt(37.0), 0); z64_4 = 12.0;
#   z64_max = evaluate(sqrt(1073 * log(2)), 0);
#   show("_ERFINV64_P1", fpminimax(build_cheb(1b-80 - 2.25, w64_1 - 2.25, false, 2.25, true, 48), 19, [|D...|], [1b-80 - 2.25; w64_1 - 2.25], relative, floating), 19);
#   show("_ERFINV64_P2", fpminimax(build_cheb(z64_1 - 3.0, z64_2 - 3.0, true, 3.0, false, 48), 19, [|D...|], [z64_1 - 3.0; z64_2 - 3.0], relative, floating, x + 3.0) - (x + 3.0), 19);
#   show("_ERFINV64_P3", fpminimax(build_cheb(z64_2 - 5.0, z64_3 - 5.0, true, 5.0, false, 48), 19, [|D...|], [z64_2 - 5.0; z64_3 - 5.0], relative, floating, x + 5.0) - (x + 5.0), 19);
#   show("_ERFINV64_P4", fpminimax(build_cheb(z64_3 - 9.0, z64_4 - 9.0, true, 9.0, false, 48), 19, [|D...|], [z64_3 - 9.0; z64_4 - 9.0], relative, floating, x + 9.0) - (x + 9.0), 19);
#   show("_ERFINV64_P5", fpminimax(build_cheb(z64_4 - 19.5, z64_max - 19.5, true, 19.5, false, 48), 19, [|D...|], [z64_4 - 19.5; z64_max - 19.5], relative, floating, x + 19.5) - (x + 19.5), 19);
_ERFINV32_P1 = (
    float.fromhex("0x1.508eb4p0"),
    float.fromhex("0x1.006e84p-2"),
    float.fromhex("-0x1.44d956p-11"),
    float.fromhex("-0x1.f2946ep-10"),
    float.fromhex("0x1.a737f8p-13"),
    float.fromhex("0x1.026b3ap-16"),
    float.fromhex("-0x1.64551ep-18"),
    float.fromhex("0x1.5cd332p-24"),
    float.fromhex("0x1.938558p-24"),
)

_ERFINV32_P2 = (
    float.fromhex("-0x1.566be2p-3"),
    float.fromhex("0x1.5cfceap-9"),
    float.fromhex("0x1.c06574p-8"),
    float.fromhex("-0x1.c3ce24p-9"),
    float.fromhex("0x1.2db682p-10"),
    float.fromhex("-0x1.c6429ep-13"),
    float.fromhex("-0x1.b5b046p-15"),
    float.fromhex("0x1.c345ap-15"),
    float.fromhex("-0x1.917454p-17"),
)

_ERFINV32_P3 = (
    float.fromhex("-0x1.0c3f68p-3"),
    float.fromhex("0x1.19ac7p-7"),
    float.fromhex("-0x1.0cd7bcp-11"),
    float.fromhex("0x1.7f5974p-18"),
    float.fromhex("0x1.9bff7cp-18"),
    float.fromhex("-0x1.7464bp-20"),
    float.fromhex("0x1.344676p-22"),
    float.fromhex("-0x1.7798f2p-24"),
    float.fromhex("0x1.b7293cp-27"),
)

_ERFINV64_P1 = (
    float.fromhex("0x1.7083a336cd764p0"),
    float.fromhex("0x1.fce34214befaep-3"),
    float.fromhex("-0x1.9d94b8d1cc026p-9"),
    float.fromhex("-0x1.821fe73cf0223p-10"),
    float.fromhex("0x1.cf8b8bcbdb1f1p-13"),
    float.fromhex("0x1.8d38bc4f859a4p-21"),
    float.fromhex("-0x1.1aedd3b03ed11p-18"),
    float.fromhex("0x1.c9571aa3cf69ap-22"),
    float.fromhex("0x1.57ea41d40b6f7p-25"),
    float.fromhex("-0x1.b146771d77374p-27"),
    float.fromhex("0x1.ecf346d136385p-32"),
    float.fromhex("0x1.d672a920915cfp-33"),
    float.fromhex("-0x1.0c6f0e94ac237p-35"),
    float.fromhex("-0x1.9087b763adfbfp-40"),
    float.fromhex("0x1.b747c74f0b4fcp-41"),
    float.fromhex("-0x1.9feaa0741881cp-45"),
    float.fromhex("-0x1.b73d097d71e3dp-47"),
    float.fromhex("0x1.1d26f17f2447ap-49"),
    float.fromhex("0x1.b058ee070e9a4p-54"),
    float.fromhex("-0x1.2ffccb301a77fp-55"),
)

_ERFINV64_P2 = (
    float.fromhex("-0x1.566be9baa624ep-3"),
    float.fromhex("0x1.5cf6bc26d4b0dp-9"),
    float.fromhex("0x1.c0820afc709fep-8"),
    float.fromhex("-0x1.c38c06b7f2ac1p-9"),
    float.fromhex("0x1.2b7b99b463d5ep-10"),
    float.fromhex("-0x1.cfecefc93c5cfp-13"),
    float.fromhex("-0x1.32cc7e7e51e7bp-15"),
    float.fromhex("0x1.d490539604ecbp-15"),
    float.fromhex("-0x1.9420cb7edb35ap-16"),
    float.fromhex("0x1.5677bcc9afc2ap-19"),
    float.fromhex("0x1.a67db4feb28e9p-19"),
    float.fromhex("-0x1.0e35ac15520b9p-19"),
    float.fromhex("0x1.9883ecf6aa351p-22"),
    float.fromhex("0x1.9d40c1d9402a5p-23"),
    float.fromhex("-0x1.57e6b5295de64p-23"),
    float.fromhex("0x1.315f1ab1eab22p-25"),
    float.fromhex("0x1.ffd498b130146p-27"),
    float.fromhex("-0x1.6ad67e67ce1c9p-27"),
    float.fromhex("0x1.56ca0f8fc7bebp-31"),
    float.fromhex("0x1.5f3c32a44f21dp-31"),
)

_ERFINV64_P3 = (
    float.fromhex("-0x1.336445cb78457p-3"),
    float.fromhex("0x1.5186909252081p-7"),
    float.fromhex("-0x1.22eb371abdb35p-13"),
    float.fromhex("-0x1.c2f0c4ba317f3p-13"),
    float.fromhex("0x1.3eb336279e7cbp-14"),
    float.fromhex("-0x1.49de79761feacp-16"),
    float.fromhex("0x1.2dcd131fc6114p-18"),
    float.fromhex("-0x1.014c93b9c770ap-20"),
    float.fromhex("0x1.a10fbfbd8714bp-23"),
    float.fromhex("-0x1.426c13d3d9282p-25"),
    float.fromhex("0x1.d1064f3f365e7p-28"),
    float.fromhex("-0x1.176ce2d17ac0dp-30"),
    float.fromhex("0x1.9ca8e696af3dap-35"),
    float.fromhex("0x1.0b05814447eebp-34"),
    float.fromhex("-0x1.5e5575ec8c5f6p-35"),
    float.fromhex("0x1.3d5558cee1095p-36"),
    float.fromhex("-0x1.f4afbb5188accp-38"),
    float.fromhex("0x1.068d714fd68e8p-39"),
    float.fromhex("0x1.64aead41961a4p-47"),
    float.fromhex("-0x1.db803224318bap-44"),
)

_ERFINV64_P4 = (
    float.fromhex("-0x1.da072a61ba4eap-4"),
    float.fromhex("0x1.b8e2742818d39p-8"),
    float.fromhex("-0x1.accd56c8ea75fp-12"),
    float.fromhex("0x1.58dc927ba7941p-16"),
    float.fromhex("-0x1.fc7f589c31317p-23"),
    float.fromhex("-0x1.461c7a571177p-23"),
    float.fromhex("0x1.1a167f5d1e39ep-25"),
    float.fromhex("-0x1.5e5f5e288cc6cp-28"),
    float.fromhex("0x1.7fef58ca557e1p-31"),
    float.fromhex("-0x1.89982c2ca9eeep-34"),
    float.fromhex("0x1.82f614b7d99ebp-37"),
    float.fromhex("-0x1.7166cfcfa4668p-40"),
    float.fromhex("0x1.583f8c4c2d1d8p-43"),
    float.fromhex("-0x1.3b923c3a99e19p-46"),
    float.fromhex("0x1.23838035a7f2ap-49"),
    float.fromhex("-0x1.04939cf5158e2p-52"),
    float.fromhex("0x1.7b33c11940dc3p-56"),
    float.fromhex("-0x1.473c3d074aaf5p-59"),
    float.fromhex("0x1.16a8e1e7620a6p-61"),
    float.fromhex("-0x1.dd5eb07dcdc59p-65"),
)

_ERFINV64_P5 = (
    float.fromhex("-0x1.2b971f40e5e38p-4"),
    float.fromhex("0x1.3fdeef10d1f97p-9"),
    float.fromhex("-0x1.8070a7ed3f146p-14"),
    float.fromhex("0x1.dd995aa777e37p-19"),
    float.fromhex("-0x1.29c10b551c6b3p-23"),
    float.fromhex("0x1.6d4bc74ea8be4p-28"),
    float.fromhex("-0x1.af290403d711cp-33"),
    float.fromhex("0x1.d76e99f341a57p-38"),
    float.fromhex("-0x1.b3d937cccea3dp-43"),
    float.fromhex("0x1.c1c18da7dae7ep-49"),
    float.fromhex("0x1.62b6d3c46a513p-53"),
    float.fromhex("-0x1.9cc37db7a34aap-56"),
    float.fromhex("0x1.0275a925b8617p-59"),
    float.fromhex("-0x1.1afa28919d34bp-63"),
    float.fromhex("0x1.5cd5a45f9f345p-67"),
    float.fromhex("-0x1.532d6710bbc58p-71"),
    float.fromhex("0x1.4964074ec7ae3p-77"),
    float.fromhex("-0x1.0f4635900acefp-81"),
    float.fromhex("0x1.0dcd16a13181bp-82"),
    float.fromhex("-0x1.dfdfd0a3f5a5cp-87"),
)


# Sollya `fpminimax` polynomial coefficients for `ndtri(p) = sqrt(2) * erfinv(2p - 1)`
# on the same `w` intervals and degree/shift structure as `_ERFINV*` (generated
# by scaling `solve_y(w)` by `sqrt(2)` in the Sollya script above, with tail
# pieces fitted against fixed part `g = k_h * (x + shift_c)` using
# `k_h = 0x1.6a0000p0` in float32 and `k_h = 0x1.6a09e68000000p0` in float64,
# `z32_max = sqrt(147 * log(2))`, and `z64_max = sqrt(1072 * log(2))` for
# `q = 2 * min(p, 1 - p)`):
_NDTRI32_P1 = (
    float.fromhex("0x1.dbf6cep0"),
    float.fromhex("0x1.6aa63p-2"),
    float.fromhex("-0x1.cb58eap-11"),
    float.fromhex("-0x1.608b44p-9"),
    float.fromhex("0x1.2b210ep-12"),
    float.fromhex("0x1.6d1694p-16"),
    float.fromhex("-0x1.f460ecp-18"),
    float.fromhex("0x1.fd779p-24"),
    float.fromhex("0x1.0d3e14p-23"),
)

_NDTRI32_P2 = (
    float.fromhex("-0x1.e35432p-3"),
    float.fromhex("0x1.00abfap-8"),
    float.fromhex("0x1.3d1088p-7"),
    float.fromhex("-0x1.3f797ep-8"),
    float.fromhex("0x1.aaacbep-10"),
    float.fromhex("-0x1.413624p-12"),
    float.fromhex("-0x1.354cacp-14"),
    float.fromhex("0x1.3f1a2ap-14"),
    float.fromhex("-0x1.1c2152p-16"),
)

_NDTRI32_P3 = (
    float.fromhex("-0x1.7931ap-3"),
    float.fromhex("0x1.934bc8p-7"),
    float.fromhex("-0x1.7c3996p-11"),
    float.fromhex("0x1.0fafbep-17"),
    float.fromhex("0x1.244c8cp-17"),
    float.fromhex("-0x1.07e81p-19"),
    float.fromhex("0x1.ad9bd8p-22"),
    float.fromhex("-0x1.08fd02p-23"),
    float.fromhex("0x1.3cee08p-26"),
)

_NDTRI64_P1 = (
    float.fromhex("0x1.0494328c11f68p1"),
    float.fromhex("0x1.67d684b8bd0d5p-2"),
    float.fromhex("-0x1.247225e75ba1p-8"),
    float.fromhex("-0x1.110805d0604b2p-9"),
    float.fromhex("0x1.47c6a064c0769p-12"),
    float.fromhex("0x1.18e0cb604677ep-20"),
    float.fromhex("-0x1.901f3e49ecb21p-18"),
    float.fromhex("0x1.43636db9d70a2p-21"),
    float.fromhex("0x1.e65e87a338d15p-25"),
    float.fromhex("-0x1.325f34412e34ep-26"),
    float.fromhex("0x1.5c925408877cap-31"),
    float.fromhex("0x1.4ca8526267166p-32"),
    float.fromhex("-0x1.7ba3536bbbec5p-35"),
    float.fromhex("-0x1.1b3d75563af1ap-39"),
    float.fromhex("0x1.36b513a3c3d5bp-40"),
    float.fromhex("-0x1.25f97285d3bc1p-44"),
    float.fromhex("-0x1.3728996b01934p-46"),
    float.fromhex("0x1.92e5e5fa2cb94p-49"),
    float.fromhex("0x1.34c83cb67b8a6p-53"),
    float.fromhex("-0x1.acf9c4a320a36p-55"),
)

_NDTRI64_P2 = (
    float.fromhex("-0x1.e441d8be4c3adp-3"),
    float.fromhex("0x1.ed8234b66041ap-9"),
    float.fromhex("0x1.3d249de327a56p-7"),
    float.fromhex("-0x1.3f4abbe9c170dp-8"),
    float.fromhex("0x1.a7885c3c296ffp-10"),
    float.fromhex("-0x1.480b7df92653ep-12"),
    float.fromhex("-0x1.b1e108357f884p-15"),
    float.fromhex("0x1.4b531a7fc52ffp-14"),
    float.fromhex("-0x1.1dc3003028c2fp-15"),
    float.fromhex("0x1.e4528e6b26637p-19"),
    float.fromhex("0x1.2abf090c0722bp-18"),
    float.fromhex("-0x1.7e2254f12d186p-19"),
    float.fromhex("0x1.20dd5ed0a343cp-21"),
    float.fromhex("0x1.24368bd08457fp-22"),
    float.fromhex("-0x1.e65a6ebcfc888p-23"),
    float.fromhex("0x1.afde942fab73cp-25"),
    float.fromhex("0x1.69efbb197c7aep-26"),
    float.fromhex("-0x1.0093c0225e999p-26"),
    float.fromhex("0x1.e4a0876288ad6p-31"),
    float.fromhex("0x1.f0d47baf6bcb8p-31"),
)

_NDTRI64_P3 = (
    float.fromhex("-0x1.b2b7b1a2ff71cp-3"),
    float.fromhex("0x1.dd5549ebcfecp-7"),
    float.fromhex("-0x1.9b6bdc0636cep-13"),
    float.fromhex("-0x1.3edcf340d0579p-12"),
    float.fromhex("0x1.c2b5bdb78bfcep-14"),
    float.fromhex("-0x1.d28159778d37ap-16"),
    float.fromhex("0x1.aacfa8c92a2f4p-18"),
    float.fromhex("-0x1.6be03bf46678dp-20"),
    float.fromhex("0x1.26e8395f1f627p-22"),
    float.fromhex("-0x1.c7f968bb2147fp-25"),
    float.fromhex("0x1.48d07eb481d54p-27"),
    float.fromhex("-0x1.8b2187adf74bfp-30"),
    float.fromhex("0x1.254242c3809eep-34"),
    float.fromhex("0x1.78b8f4cdb8771p-34"),
    float.fromhex("-0x1.f20e8edb60dcp-35"),
    float.fromhex("0x1.c421e187080b9p-36"),
    float.fromhex("-0x1.5cf04feb50101p-37"),
    float.fromhex("0x1.65ad647afa4f8p-39"),
    float.fromhex("-0x1.1e0712564e325p-46"),
    float.fromhex("-0x1.21aaa0ee852d4p-43"),
)

_NDTRI64_P4 = (
    float.fromhex("-0x1.4f30422c407cfp-3"),
    float.fromhex("0x1.37c0967c8044fp-7"),
    float.fromhex("-0x1.2f3578ef61b8dp-11"),
    float.fromhex("0x1.e7b53d46362c8p-16"),
    float.fromhex("-0x1.678fdeb8075acp-22"),
    float.fromhex("-0x1.cd30e1818fe2dp-23"),
    float.fromhex("0x1.8eeeb8c790b52p-25"),
    float.fromhex("-0x1.ef8067d440c86p-28"),
    float.fromhex("0x1.0f7ba63868244p-30"),
    float.fromhex("-0x1.165033b6c620dp-33"),
    float.fromhex("0x1.119f78d647a46p-36"),
    float.fromhex("-0x1.0534d25c884dap-39"),
    float.fromhex("0x1.e6d71c7211099p-43"),
    float.fromhex("-0x1.be493db6c5e64p-46"),
    float.fromhex("0x1.9c43a232d9c07p-49"),
    float.fromhex("-0x1.70813fb19231ap-52"),
    float.fromhex("0x1.0c2174a0c8ce1p-55"),
    float.fromhex("-0x1.ced2868e22e89p-59"),
    float.fromhex("0x1.8a17cd7e962p-61"),
    float.fromhex("-0x1.518570c182bf7p-64"),
)

_NDTRI64_P5 = (
    float.fromhex("-0x1.a7af65743d66p-4"),
    float.fromhex("0x1.c45d6cafd81f2p-9"),
    float.fromhex("-0x1.0fd715b65ab27p-13"),
    float.fromhex("0x1.51b6a937534b5p-18"),
    float.fromhex("-0x1.a5167dcb6d5p-23"),
    float.fromhex("0x1.024da42857645p-27"),
    float.fromhex("-0x1.30e0565828cb7p-32"),
    float.fromhex("0x1.4d5a4d0326a65p-37"),
    float.fromhex("-0x1.3430e913d10b6p-42"),
    float.fromhex("0x1.3e08cf0897759p-48"),
    float.fromhex("0x1.f599e21836bc3p-53"),
    float.fromhex("-0x1.23ec07f0c81c5p-55"),
    float.fromhex("0x1.6d98804f9ac97p-59"),
    float.fromhex("-0x1.8fcbc70762358p-63"),
    float.fromhex("0x1.ecf92b25a9dabp-67"),
    float.fromhex("-0x1.e15c809d29bf6p-71"),
    float.fromhex("0x1.d53b0574b4157p-77"),
    float.fromhex("-0x1.6fe0b0aeaa851p-81"),
    float.fromhex("0x1.7d21d290ddc3fp-82"),
    float.fromhex("-0x1.553dbd7f1f0afp-86"),
)


_ERFINV32_P = (_ERFINV32_P1, _ERFINV32_P2, _ERFINV32_P3)
_NDTRI32_P = (_NDTRI32_P1, _NDTRI32_P2, _NDTRI32_P3)
_ERFINV64_P = (
    _ERFINV64_P1, _ERFINV64_P2, _ERFINV64_P3, _ERFINV64_P4, _ERFINV64_P5
)
_NDTRI64_P = (_NDTRI64_P1, _NDTRI64_P2, _NDTRI64_P3, _NDTRI64_P4, _NDTRI64_P5)

_ERFINV32_W_CUTS = (3.5, 18.0625)
_ERFINV32_SHIFTS = (3.0, 7.0)
_NDTRI32_K_H = float.fromhex("0x1.6a0000p0")

_ERFINV64_W_CUTS = (4.5, 16.0, 37.0, 144.0)
_ERFINV64_SHIFTS = (3.0, 5.0, 9.0, 19.5)
_NDTRI64_K_H = float.fromhex("0x1.6a09e68000000p0")


def _piece_index(conds: tuple[Array, ...]) -> Array:
  if len(conds) == 1:
    return ~conds[0]
  idx = convert_element_type(~conds[0], np.int32)
  for cond in conds[1:]:
    idx = idx + convert_element_type(~cond, np.int32)
  return idx


def _select_piece(w: Array, idx: Array, vals: tuple[float, ...]) -> Array:
  return select_n(idx, *(full_like(w, v) for v in vals))


def erf_inv_core(
    x: Array,
    q: Array,
    *,
    has_far_tail: bool = True,
    is_ndtri: bool = False,
) -> Array:
  """Evaluates `erfinv(x)` (or `sqrt(2) * erfinv(x)` when `is_ndtri=True`).

  Takes both `x in (-1, 1)` and `q = 1 - |x| in (0, 1]` so that
  `1 - x^2 = q * (1 + |x|)` can be computed without cancellation near `|x| = 1`:
  - `erfinv(x)` passes `x = x` and `q = 1 - |x|` (`has_far_tail=False`).
  - `erfcinv(y)` passes `x = 1 - y` and `q = min(y, 2 - y)`
    (`has_far_tail=True`).
  - `ndtri(p)` passes `x = 2 * (p - 0.5)` and `q = 2 * min(p, 1 - p)`
    (`has_far_tail=True`, `is_ndtri=True`).

  When `has_far_tail=False` (`erfinv`), only the pieces needed for `q >= 2^-24`
  (`float32`, 2 pieces) or `q >= 2^-53` (`float64`, 3 pieces) are selected.
  """
  ax = abs(x)
  w = -log(q * (1.0 + ax))
  if x.dtype == np.float32:
    ps = _NDTRI32_P if is_ndtri else _ERFINV32_P
    n_cuts = len(_ERFINV32_W_CUTS) if has_far_tail else 1
    w_cuts = _ERFINV32_W_CUTS[:n_cuts]
    shifts = _ERFINV32_SHIFTS[:n_cuts]
    # Near |x| = 1 (large w), erfinv(x) grows like sqrt(w) and ndtri(p) grows
    # like sqrt(2) * sqrt(w). For ndtri, round sqrt(2) to half its fraction
    # bits so multiplying `k_h * z_h` below has zero rounding error.
    k_h = _NDTRI32_K_H if is_ndtri else 1.0
  elif x.dtype == np.float64:
    ps = _NDTRI64_P if is_ndtri else _ERFINV64_P
    n_cuts = len(_ERFINV64_W_CUTS) if has_far_tail else 2
    w_cuts = _ERFINV64_W_CUTS[:n_cuts]
    shifts = _ERFINV64_SHIFTS[:n_cuts]
    k_h = _NDTRI64_K_H if is_ndtri else 1.0
  else:
    raise TypeError(f"Unsupported dtype for erf_inv_core: {x.dtype}")

  conds = tuple(w < c for c in w_cuts)
  c1 = 0.5 * w_cuts[0]
  unroll = len(ps[0])
  ps = ps[: len(conds) + 1]
  idx = _piece_index(conds)

  # In the tail pieces, the polynomial tables `P(t)` store only the small
  # remainder `f(z) - k_h * z` around `t = z - shift`, leaving `k_h * sqrt(w)`
  # to be evaluated separately:
  # - Because `k_h * z` has already been subtracted from `P(t)`, the linear
  #   coefficient `p[1]` is near zero (e.g. ~0.003 for erfinv), so rounding
  #   error in `z` inside `t = z - shift` barely affects `poly`.
  # - To compute `k_h * sqrt(w)` in extra precision without needing FMA or wider
  #   floats, split `sqrt(w)` into `z_h + z_l`:
  #   * `z_h, _ = split_half_mask(z)` zeros the low half of `z`'s fraction bits,
  #     so `z_h * z_h` and `k_h * z_h` are exact.
  #   * Since `w - z_h^2 = (sqrt(w) - z_h) * (sqrt(w) + z_h)`, we can compute
  #     `sqrt(w) - z_h = (w - z_h^2) / (sqrt(w) + z_h)`. Because `z_h * z_h` is
  #     close to `w_safe`, the subtraction `w_safe - z_h * z_h` is exact.
  w_safe = select(conds[0], full_like(w, 1.0), w)
  z = sqrt(w_safe)
  z_h, _ = split_half_mask(z)
  z_l = (w_safe - z_h * z_h) / (z + z_h)
  shift = (
      full_like(w, shifts[0])
      if len(shifts) == 1
      else _select_piece(w, idx, (0.0, *shifts))
  )
  t = select(conds[0], w - c1, z - shift)

  coeffs = [_select_piece(w, idx, cs) for cs in zip(*ps)]
  poly = polynomial(t, coeffs, unroll=unroll)
  # Add the two small parts (`k_h * z_l` and `poly`) before adding them to the
  # exact main term `k_h * z_h`.
  rem = k_h * z_l + poly
  tail_mag = k_h * z_h + rem
  sign_x = select(x < 0.0, full_like(x, -1.0), full_like(x, 1.0))
  return select(conds[0], x * poly, sign_x * tail_mag)


@custom_jvp
def erf_inv_jvp_scale(x: Array, z: Array) -> Array:
  """Evaluates `d/dx erfinv(x) = (sqrt(pi) / 2) * exp(z^2)` where `z = erfinv(x)`.

  - For `|z| <= 1`, `(sqrt(pi) / 2) * exp(z^2)` has condition number `<= 2`
    with respect to `z`.
  - For `|z| > 1`, evaluating `exp(z^2)` directly amplifies any rounding error
    in `z` by `2 * z^2` ULPs. Using `1 - |x| = erfc(|z|) = exp(-z^2) * erfcx(|z|)`
    gives `(sqrt(pi) / 2) * erfcx(|z|) / (1 - |x|)`, where `1 - |x|` is exact by
    Sterbenz's lemma (`|x| > erf(1) > 0.5`) and `erfcx(|z|)` has condition
    number `< 1` with respect to `z`.
  """
  q = 1.0 - abs(x)
  u = abs(z)
  is_tail = u > 1.0

  zeros = full_like(x, 0.0)
  ones = full_like(x, 1.0)
  half_sqrt_pi = 0.5 * math.sqrt(math.pi)

  z_central = select(is_tail, zeros, z)
  central = half_sqrt_pi * exp(square(z_central))

  is_finite_tail = is_tail & (q > 0.0)
  u_tail = select(is_finite_tail, u, ones)
  q_safe = select(is_finite_tail, q, ones)
  tail = select(
      q == 0.0,
      full_like(x, np.inf),
      (half_sqrt_pi * erfcx_large(u_tail)) / q_safe,
  )
  return select(q < 0.0, full_like(x, np.nan), select(is_tail, tail, central))


@erf_inv_jvp_scale.defjvp
def _erf_inv_jvp_scale_jvp(primals, tangents):
  # `z` is the cached primal `erfinv(x)`, not an independent input, so the total
  # derivative with respect to `x` is d/dx ((sqrt(pi) / 2) * exp(z^2)) = 2 * z * g^2.
  (x, z), (x_dot, _) = primals, tangents
  g = erf_inv_jvp_scale(x, z)
  return g, x_dot * ((2.0 * z * g) * g)


# Polynomial coefficients for ndtri_exp(y) near the median y = -log(2),
# where the function crosses zero. Let t = sqrt(pi/2) * (y + log(2)); on
# [_NDTRI_EXP_MEDIAN_LO, _NDTRI_EXP_MEDIAN_HI], ndtri_exp(y) ~= t + t^2 * P(t)
# has relative error at most 0.025 ULPs (float32) / 0.01 ULPs (float64).
# Coefficients generated by Sollya fpminimax (script below).
#
# Sollya script:
#   prec = 300!;
#   display = hexadecimal!;
#   c = round(sqrt(pi / 2), 280, RN);
#   s2 = round(sqrt(2), 280, RN);
#   ln2 = round(log(2), 280, RN);
#   procedure solve_z(t_val) {
#     var z, k, q_val, f_val, df_val;
#     q_val = 2 * evaluate(exp(x), round(-ln2 + t_val / c, 280, RN));
#     z = t_val;
#     for k from 1 to 9 do {
#       f_val = evaluate(erfc(x), round(-z / s2, 280, RN)) - q_val;
#       df_val = evaluate(exp(-x * x / 2), z) / c;
#       z = round(z - f_val / df_val, 280, RN);
#     };
#     return z;
#   };
#   t_lo = round(c * (-92 / 100 + ln2), 280, RN);
#   t_hi = round(c * (-48 / 100 + ln2), 280, RN);
#   nodes = [||];
#   vals = [||];
#   for i from 0 to 47 do {
#     theta = round(pi * (2 * i + 1) / 96, 280, RN);
#     t_i = round(0.5 * (t_lo + t_hi)
#                 + 0.5 * (t_hi - t_lo) * evaluate(cos(x), theta), 280, RN);
#     z_i = solve_z(t_i);
#     nodes = t_i .: nodes;
#     vals = round((z_i - t_i) / (t_i * t_i), 280, RN) .: vals;
#   };
#   g = interpolate(nodes, vals, 1b-220, [t_lo; t_hi]);
#   p32 = fpminimax(g, 8, [|SG...|], [t_lo; t_hi], relative, floating);
#   p64 = fpminimax(g, 19, [|D...|], [t_lo; t_hi], relative, floating);
#   for i from 0 to 8 do print(coeff(p32, i));
#   for i from 0 to 19 do print(coeff(p64, i));
_NDTRI_EXP_MEDIAN_LO = -0.92
_NDTRI_EXP_MEDIAN_HI = -0.48

_NDTRI_EXP32_MEDIAN = (
    float.fromhex("0x1.988454p-2"),
    float.fromhex("0x1.1750f2p-2"),
    float.fromhex("0x1.c3dbc8p-3"),
    float.fromhex("0x1.8e187ep-3"),
    float.fromhex("0x1.716b48p-3"),
    float.fromhex("0x1.5eab06p-3"),
    float.fromhex("0x1.593b6cp-3"),
    float.fromhex("0x1.a5c942p-3"),
    float.fromhex("0x1.a932cp-3"),
)

_NDTRI_EXP64_MEDIAN = (
    float.fromhex("0x1.9884533d4365p-2"),
    float.fromhex("0x1.175102498982ep-2"),
    float.fromhex("0x1.c3dca46cd7a41p-3"),
    float.fromhex("0x1.8e0209d202216p-3"),
    float.fromhex("0x1.71419cbaddbbp-3"),
    float.fromhex("0x1.62cc54262ba58p-3"),
    float.fromhex("0x1.5da08eb813145p-3"),
    float.fromhex("0x1.5f2b1027bf2f6p-3"),
    float.fromhex("0x1.6604b4472a2fp-3"),
    float.fromhex("0x1.7167d2fb9d13fp-3"),
    float.fromhex("0x1.80efe392a31f9p-3"),
    float.fromhex("0x1.94665dc20275fp-3"),
    float.fromhex("0x1.ab7f145ffb46dp-3"),
    float.fromhex("0x1.c77db086bf382p-3"),
    float.fromhex("0x1.ed964c26ae3a6p-3"),
    float.fromhex("0x1.060c76e938cf3p-2"),
    float.fromhex("0x1.f0c00fed9f6bbp-3"),
    float.fromhex("0x1.239e8f5d5b09ap-2"),
    float.fromhex("0x1.07fd9e2758f4fp-1"),
    float.fromhex("0x1.fc10e4194114bp-2"),
)


# Full-width (hi, lo) float splits of ln(2), 0.5*ln(2*pi), and sqrt(pi/2).
# Unlike LN2_SPLIT_F{32,64} (which shorten the high word so k * ln2_hi is
# exact for the integer k in exp range reduction), the median-region
# subtraction d_hi = y + ln2_hi needs full-precision ln2_hi.
_LN2_F32 = (
    float.fromhex("0x1.62e43p-1"),
    float.fromhex("-0x1.05c61p-29"),
)
_LN2_F64 = (
    float.fromhex("0x1.62e42fefa39efp-1"),
    float.fromhex("0x1.abc9e3b39803fp-56"),
)
_HALF_LN2PI_F32 = (
    float.fromhex("0x1.d67f1cp-1"),
    float.fromhex("0x1.0c97d6p-26"),
)
_HALF_LN2PI_F64 = (
    float.fromhex("0x1.d67f1c864beb5p-1"),
    float.fromhex("-0x1.65b5a1b7ff5dfp-55"),
)
_SQRT_HALF_PI_F32 = (
    float.fromhex("0x1.40d932p0"),
    float.fromhex("-0x1.3b1f4ep-33"),
)
_SQRT_HALF_PI_F64 = (
    float.fromhex("0x1.40d931ff62706p0"),
    float.fromhex("-0x1.a6a0d6f814637p-54"),
)

# Asymptotic series coefficients for the far-left-tail root solve in
# `ndtri_exp_core`. For `u = x^2 / 2` and `u0 = -y - 0.5*log(2*pi)`, solving
# `log_ndtr(-sqrt(2*u)) = y` is equivalent to finding the root of
#   `f(u) = -log_ndtr(-sqrt(2*u)) - 0.5*log(2*pi) - u0`
#         `= u + 0.5*log(2*u) - R(1/u) - u0 = 0`,
# where `R(1/u)` is the asymptotic series of `log(sqrt(pi*u) * erfcx(sqrt(u)))`
# (`_NDTRI_EXP_ASYMP`). Differentiating gives
# `1 / f'(u) = sqrt(pi*u) * erfcx(sqrt(u))`, whose asymptotic series
# `sum_{m>=0} (-1)^m (2m - 1)!! / (2*u)^m` (`_NDTRI_EXP_INV_DF`) lets the
# Newton step multiply by `1 / f'(u)` instead of dividing.
_NDTRI_EXP_INV_DF = (1.0, -1 / 2, 3 / 4, -15 / 8, 105 / 16)
_NDTRI_EXP_ASYMP = (
    -1 / 2,
    5 / 8,
    -37 / 24,
    353 / 64,
    -4081 / 160,
    55205 / 384,
    -854197 / 896,
)


def ndtri_exp_core(y: Array) -> Array:
  """Computes `ndtri(exp(y))` for finite `y < 0` in `float32` or `float64`.

  Evaluating `ndtri(exp(y))` directly fails in three regions:
  - Near `y = -log(2)`, `2 * exp(y) - 1` cancels catastrophically.
  - Near `y = 0`, `exp(y)` approaches `1`, so rounding `exp(y)` to a single
    float loses precision (and rounds to `1`, giving `+inf`, for `y > -eps/2`).
  - For large negative `y`, `exp(y)` underflows into subnormals and then to `0`
    (`ndtri(0) = -inf`), even though `ndtri(exp(y))` stays finite across all
    finite `y < 0`.

  `ndtri_exp_core` therefore uses three regimes, sharing a single `exp`,
  `expm1`, `log`, and `sqrt` call:
  1. Near the median (`_NDTRI_EXP_MEDIAN_LO <= y <= _NDTRI_EXP_MEDIAN_HI`):
     `y + ln2_hi` has no rounding error (because `-y` and `ln2_hi` are within a
     factor of 2), so we evaluate a polynomial in `t = sqrt(pi/2) * (y +
     log(2))`.
  2. Moderate tails (`y in [-max_w_y, _NDTRI_EXP_MEDIAN_LO)` or
     `(_NDTRI_EXP_MEDIAN_HI, 0)`): computes `x = 2 * exp(y) - 1` and
     `w = -log(4 * exp(y) * (1 - exp(y)))` without subtracting nearly equal
     numbers via `exp` and `expm1`, reusing `_NDTRI{32,64}_P` from
     `erf_inv_core`.
  3. Far left tail (`y < -max_w_y`): inverts the asymptotic expansion of
     `log_ndtr` for `u = 0.5 * x^2` using `(hi, lo)` pairs of floats for extra
     precision.
  """
  if y.dtype == np.float32:
    ln2_hi, ln2_lo = _LN2_F32
    half_ln2pi_hi, half_ln2pi_lo = _HALF_LN2PI_F32
    c_hi, c_lo = _SQRT_HALF_PI_F32
    median_cs = _NDTRI_EXP32_MEDIAN
    ps = _NDTRI32_P
    w_cuts = _ERFINV32_W_CUTS
    shifts = _ERFINV32_SHIFTS
    k_h = _NDTRI32_K_H
    # Term counts for the far-tail series R(1/u), 1/f'(u), log1p(delta)/delta,
    # and 1/(1+delta); each omitted tail is below 0.05 ULPs of u0 (and below
    # 0.01 ULP of the final output x).
    n_asymp, n_inv_df, n_log1p, n_rcp = 4, 4, 3, 4
  elif y.dtype == np.float64:
    ln2_hi, ln2_lo = _LN2_F64
    half_ln2pi_hi, half_ln2pi_lo = _HALF_LN2PI_F64
    c_hi, c_lo = _SQRT_HALF_PI_F64
    median_cs = _NDTRI_EXP64_MEDIAN
    ps = _NDTRI64_P
    w_cuts = _ERFINV64_W_CUTS
    shifts = _ERFINV64_SHIFTS
    k_h = _NDTRI64_K_H
    n_asymp, n_inv_df, n_log1p, n_rcp = 7, 5, 6, 7
  else:
    raise TypeError(f"Unsupported dtype for ndtri_exp_core: {y.dtype}")

  finfo = np.finfo(y.dtype)
  max_w_y = -math.log(2.0 * float(finfo.tiny))
  tiny_cut = -0.25 * float(finfo.eps)
  shift_ln2 = float(finfo.bits)
  asymp_cap = math.ldexp(1.0, finfo.maxexp // 2)
  scale_exp = math.ldexp(4.0, finfo.bits)
  asymp_cs = _NDTRI_EXP_ASYMP[:n_asymp]
  inv_df_cs = _NDTRI_EXP_INV_DF[:n_inv_df]
  log1p_cs = [(-1.0) ** (k - 1) / k for k in range(1, n_log1p + 1)]
  rcp1p_cs = [(-1.0) ** k for k in range(n_rcp)]
  c1 = 0.5 * w_cuts[0]
  unroll = len(ps[0])
  zero = full_like(y, 0.0)
  one = full_like(y, 1.0)

  # 1. Near the median: because `-y` and `ln2_hi` are within a factor of 2 on
  # `[_NDTRI_EXP_MEDIAN_LO, _NDTRI_EXP_MEDIAN_HI]`, `d_hi = y_med + ln2_hi` has
  # no floating-point rounding error. Evaluate `t = sqrt(pi/2) * (d_hi +
  # ln2_lo)` as a `(t_hi, t_lo)` float pair, then `t + t^2 * P(t)`.
  y_med = clamp(
      _const(y, _NDTRI_EXP_MEDIAN_LO), y, _const(y, _NDTRI_EXP_MEDIAN_HI)
  )
  d_hi = y_med + ln2_hi
  t_hi, t_err = two_prod(d_hi, full_like(d_hi, c_hi))
  t_lo = (t_err + d_hi * c_lo) + ln2_lo * c_hi
  t_med = t_hi + t_lo
  p_med = polynomial(t_med, median_cs, unroll=len(median_cs))
  near_median = t_hi + (t_lo + square(t_med) * p_med)

  # 2. Moderate tails: `_NDTRI{32,64}_P` from `erf_inv_core` expects the
  # transformed coordinates `x = 2*p - 1` and `w = -log(4*p*(1 - p)) =
  # -log(1 - x^2)` for `p = exp(y)`.
  #
  # Compute `x = 2*exp(y) - 1` as a `(x_hi, x_lo)` float pair across three
  # sub-intervals so we neither lose low bits when adding a small tail to `±1`
  # nor cancel nearly equal numbers near `p = 1/2` (`y = -ln2`):
  # - Left (`y <= -2*ln2`, `p <= 1/4`): write `x = -1 + 2*exp(y)`. Here
  #   `|2*exp(y)| <= 0.5`, which is strictly smaller than `|-1|`. Because
  #   `fast_two_sum(a, b)` requires `|a| >= |b|`, this ordering lets us add `-1`
  #   and `2*p_left` with `fast_two_sum` below and capture the exact rounding
  #   error in `x_left_lo` instead of losing low bits of `2*exp(y)` against
  #   `-1`.
  # - Middle (`-2*ln2 < y < -ln2/2`, `1/4 < p < 1/sqrt(2)`): near `y = -ln2`,
  #   `-1 + 2*exp(y)` would subtract nearly equal numbers. Instead, absorb
  #   `2 = exp(ln2)` into the exponent so `x = exp(y + ln2) - 1` passes an
  #   argument near `0` to `expm1`. Splitting `ln2 = ln2_hi + ln2_lo`, let
  #   `A = y_mod + ln2_hi` and `B = ln2_lo`:
  #   * `-y_mod` and `ln2_hi` are within a factor of 2 here, so
  #     `A = y_mod + ln2_hi` has zero rounding error.
  #   * Since `|B| = |ln2_lo|` is tiny (< 2^-28), a first-order Taylor expansion
  #     around `A` gives `exp(A + B) - 1 ~= expm1(A) + exp(A) * B`. Writing
  #     `expm1_val = expm1(A)` and `exp(A) = 1 + expm1_val` gives
  #     `x_mid_hi = expm1_val` and `x_mid_lo = (1 + expm1_val) * ln2_lo`.
  # - Right (`y >= -ln2/2`, `p >= 1/sqrt(2)`): near `y = 0`, `exp(y)` rounds
  #   toward `1` and loses precision, so rewrite `2*exp(y) - 1 = 1 + 2*expm1(y)`
  #   instead. On `[-ln2/2, 0)`, `|2*expm1(y)| <= 2 - sqrt(2) ~= 0.59 < |1|`,
  #   satisfying `|a| >= |b|` so `fast_two_sum(1, 2*expm1_val)` captures the
  #   exact rounding error in `x_right_lo`.
  two_ln2 = 2.0 * ln2_hi
  half_ln2 = 0.5 * ln2_hi
  ln_3_4 = math.log(0.75)

  y_mod = lax_max(y, _const(y, -max_w_y))
  y_left = lax_min(y_mod, _const(y, -two_ln2))
  p_left = exp(y_left, accuracy=AccuracyMode.HIGHEST)

  # Share one `expm1` call between the middle (`y_mod + ln2_hi`) and right
  # (`y_mod`) branches.
  is_mid_x = (y_mod > -two_ln2) & (y_mod < -half_ln2)
  y_right_x = lax_max(y_mod, _const(y, -half_ln2))
  expm1_arg = select(is_mid_x, y_mod + ln2_hi, y_right_x)
  expm1_val = expm1(expm1_arg, accuracy=AccuracyMode.HIGHEST)

  x_left_hi, x_left_lo = fast_two_sum(_const(y, -1.0), 2.0 * p_left)
  x_right_hi, x_right_lo = fast_two_sum(_const(y, 1.0), 2.0 * expm1_val)
  x_mid_hi = expm1_val
  x_mid_lo = (1.0 + expm1_val) * ln2_lo
  is_left = y_mod <= -two_ln2
  x_hi = select(is_left, x_left_hi, select(is_mid_x, x_mid_hi, x_right_hi))
  x_lo = select(is_left, x_left_lo, select(is_mid_x, x_mid_lo, x_right_lo))

  # Far-tail target `u0 = -y - 0.5*log(2*pi)` as a `(u0_hi, u0_lo)` float pair
  # (needed ahead of section 3 so `u0_hi` can share the `log` call below; capped
  # at `asymp_cap` because huge `-y` uses `-2*sqrt(-y/2)` directly).
  is_asymp = y < -max_w_y
  abs_y = select(is_asymp, -y, full_like(y, max_w_y))
  abs_y_cap = lax_min(abs_y, _const(y, asymp_cap))
  u0_hi, u0_err = two_sum(abs_y_cap, _const(abs_y_cap, -half_ln2pi_hi))
  u0_lo = u0_err - half_ln2pi_lo

  # Evaluate a single `shared_log` serving both the far-tail `log(u0_hi)` and
  # the moderate-tail `w = -log(4*p*(1 - p))`. In the left and right tails,
  # factor `p = exp(y)` out of `-log(4*p*(1 - p))` so `-log(p) = -y` is added
  # directly in log space without round-tripping through `exp` and `log`:
  # - Left (`y <= -2*ln2`, `p <= 1/4`): `w = -y - 2*ln2 - log(1 - p)`, passing
  #   `one_minus_p_hi` (`>= 3/4`) to `shared_log`.
  # - Central (`-2*ln2 < y < log(3/4)`, `|x| < 1/2`): `w = -log(1 - x^2)`,
  #   passing `one_minus_x2` (`>= 3/4`) to `shared_log`.
  # - Right (`y >= log(3/4)`, `p >= 3/4`): `w = -y - log(-4*expm1(y))`, passing
  #   `right_arg = -4*expm1_val` to `shared_log`. When `y > tiny_cut`
  #   (`-y < eps/4`), `-expm1(y)` equals `-y` to machine precision; multiplying
  #   `-y_right` by `scale_exp = 4 * 2^bits` lifts subnormal `-y` into normal
  #   range first.
  is_right = y_mod >= ln_3_4
  y_right = lax_max(y_mod, _const(y, ln_3_4))
  is_tiny = y_right > tiny_cut
  right_arg = select(is_tiny, (-y_right) * scale_exp, -4.0 * expm1_val)
  one_minus_p_hi, one_minus_p_lo = fast_two_sum(_const(y, 1.0), -p_left)
  x2_hi, x2_lo = _split_square(x_hi)
  one_minus_x2 = (1.0 - x2_hi) - (x2_lo + 2.0 * x_hi * x_lo)
  mid_or_right_arg = select(is_right, right_arg, one_minus_x2)
  log_arg = select(is_left, one_minus_p_hi, mid_or_right_arg)
  shared_log = log(
      select(is_asymp, u0_hi, log_arg), accuracy=AccuracyMode.HIGHEST
  )

  # Left `w = -y - 2*ln2 - log(1 - p)`: `shared_log` evaluated
  # `log(one_minus_p_hi)`, and since `1 - p = one_minus_p_hi + one_minus_p_lo`,
  # restore `one_minus_p_lo` via `log(one_minus_p_hi + one_minus_p_lo) ~=
  # shared_log + one_minus_p_lo / (1 - p)` with `1 / (1 - p) ~= 1 + p`
  # (`|one_minus_p_lo * p^2| <= 0.03 * eps` on `p <= 0.25`). For `p < eps/2`,
  # `one_minus_p_hi` rounds to `1.0` (`shared_log = 0`) and
  # `one_minus_p_lo = -p` holds all of `-p`, so use the second-order Taylor step
  # `log1p_tiny = log1p(-p) ~= one_minus_p_lo * (1 - 0.5 * one_minus_p_lo)`.
  log1p_tiny = one_minus_p_lo * (1.0 - 0.5 * one_minus_p_lo)
  log_1mp_norm = shared_log + one_minus_p_lo * (1.0 + p_left)
  log_1mp = select(one_minus_p_hi == 1.0, log1p_tiny, log_1mp_norm)
  y_shift_hi, y_shift_err = two_sum(-y_left, _const(y, -two_ln2))
  w_left_hi, w_left_err = two_sum(y_shift_hi, -log_1mp)
  w_left_lo = w_left_err + (y_shift_err - 2.0 * ln2_lo)

  # Right `w = -y - log(-4*expm1(y))`: when `is_tiny` (`-y < eps/4`),
  # `scale_exp` scaled the `log` argument by `2^finfo.bits` (`2^32` or `2^64`).
  # Because `shift_ln2 = finfo.bits` (`32` or `64`) is a power of two,
  # `shift_ln2 * ln2_hi` and `shift_ln2 * ln2_lo` are exact, and `-y_right` is
  # `< 0.02` ULP of `w > 22` so it can be dropped when `is_tiny` is true.
  right_shift_hi = select(is_tiny, full_like(y, shift_ln2 * ln2_hi), -y_right)
  right_shift_lo = select(is_tiny, full_like(y, shift_ln2 * ln2_lo), zero)
  w_right_hi, w_right_err = two_sum(-shared_log, right_shift_hi)
  w_right_lo = w_right_err + right_shift_lo

  w_hi = select(is_left, w_left_hi, select(is_right, w_right_hi, -shared_log))
  w_lo = select(is_left, w_left_lo, select(is_right, w_right_lo, zero))
  w = w_hi + w_lo

  # 3. Far left tail (`y < -max_w_y`): substituting
  # `ndtr(x) = 0.5 * exp(-x^2/2) * erfcx(-x/sqrt(2))` into `log(ndtr(x)) = y`
  # with `u = x^2 / 2` and `u0 = -y - 0.5*log(2*pi)` gives the root equation
  #   `f(u) = u + 0.5*log(2*u) - R(1/u) - u0 = 0`,
  # where `R(1/u) = log(sqrt(pi*u) * erfcx(sqrt(u)))` and differentiating gives
  # `1 / f'(u) = sqrt(pi*u) * erfcx(sqrt(u))`. Since `shared_log = log(u0_hi)`,
  # `log(2*u0_hi) = shared_log + ln2 = log_2u0_hi + log_2u0_lo`. Two fixed-point
  # steps (`d0 = -0.5*log_2u0_hi`, `d1 = -0.5*(log_2u0_hi + d0/u0) + c_1/u0`)
  # give `u1 = u0_hi + d1`, refined by one Newton step
  # `u = u1 - f(u1) * (1/f'(u1))`.
  log_2u0_hi, log_2u0_err = two_sum(shared_log, _const(y, ln2_hi))
  log_2u0_lo = log_2u0_err + ln2_lo
  inv_u0 = 1.0 / u0_hi
  d0 = -0.5 * log_2u0_hi
  d1 = d0 + (-0.5 * d0 + asymp_cs[0]) * inv_u0
  u1 = u0_hi + d1
  # Write `u1 = u0_hi * (1 + delta)` with `delta = (u1 - u0_hi) / u0_hi` so we
  # can evaluate `log(2*u1) = log(2*u0_hi) + log1p(delta)` (`log1p_delta`) and
  # `1/u1 = (1/u0_hi) / (1 + delta)` (`inv_u1`) via short Taylor series in
  # `delta` instead of a second `log` or division in the Newton step.
  delta = (u1 - u0_hi) * inv_u0
  log1p_delta = delta * polynomial(delta, log1p_cs, unroll=n_log1p)
  inv_u1 = inv_u0 * polynomial(delta, rcp1p_cs, unroll=n_rcp)
  r_u1 = inv_u1 * polynomial(inv_u1, asymp_cs, unroll=n_asymp)
  d2_lo = -0.5 * (log_2u0_lo + log1p_delta) + r_u1
  # Decompose `f(u1) = (u1 - u0) + 0.5*log(2*u1) - R(1/u1)` using
  # `-d0 = 0.5*log_2u0_hi` and `-d2_lo = 0.5*(log_2u0_lo + log1p_delta) - R`.
  # Grouping `((u1 - u0_hi) - d0)` first avoids cancellation error: `u1` is
  # close to `u0_hi` and `u1 - u0_hi` is close to `d0` (both pairs are within a
  # few percent), so both subtractions are exact with zero rounding error.
  f_u1 = ((u1 - u0_hi) - d0) - (u0_lo + d2_lo)
  newton_step = -f_u1 * polynomial(inv_u1, inv_df_cs, unroll=n_inv_df)
  u_hi, u_lo = two_sum(u1, newton_step)

  # Share one `sqrt` and fraction-bit split `z = z_h + z_l` across the moderate
  # (`w`) and far (`2*u`) tails (see `erf_inv_core`). Above `asymp_cap`
  # (`2^(maxexp // 2)`), the `log(2*u)` term is smaller than float roundoff, so
  # `x = -sqrt(-2*y) = -2 * sqrt(-y / 2)`; evaluating `sqrt(-y / 2)` (with
  # `k_tail = 2` below) avoids overflowing `-2*y` near `-fmax`.
  conds = tuple(w < c for c in w_cuts)
  w_mod_hi = select(conds[0], one, w_hi)
  w_mod_lo = select(conds[0], zero, w_lo)
  is_huge = abs_y > asymp_cap
  rad_hi = select(is_asymp, select(is_huge, 0.5 * abs_y, 2.0 * u_hi), w_mod_hi)
  rad_lo = select(is_asymp, select(is_huge, zero, 2.0 * u_lo), w_mod_lo)
  z = sqrt(rad_hi + rad_lo)
  z_h, _ = split_half_mask(z)
  z_l = ((rad_hi - z_h * z_h) + rad_lo) / (z + z_h)

  idx = _piece_index(conds)
  shift = _select_piece(w, idx, (0.0, *shifts))
  t = select(conds[0], (w_hi - c1) + w_lo, select(is_asymp, zero, z - shift))
  coeffs = [_select_piece(w, idx, cs) for cs in zip(*ps)]
  poly = polynomial(t, coeffs, unroll=unroll)
  # Tail magnitude `k_tail * z + rem_tail`: `k_h * sqrt(w) + poly` in the
  # moderate tail, `sqrt(2*u)` in the far tail, or `2 * sqrt(-y/2)` above
  # `asymp_cap`.
  k_asymp = select(is_huge, full_like(y, 2.0), one)
  k_tail = select(is_asymp, k_asymp, full_like(y, k_h))
  rem_tail = select(is_asymp, zero, poly)
  tail_mag = k_tail * z_h + (k_tail * z_l + rem_tail)
  sign_x = select(y < -ln2_hi, full_like(y, -1.0), one)
  # Note that we multiply `x_hi` and `x_lo` by `poly` separately: `x_hi + x_lo`
  # rounds `x_lo` away when `(x_hi, x_lo)` comes from `fast_two_sum`, whereas
  # `x_lo * poly` can still shift `x_hi * poly` by 1 ULP.
  central = (x_hi * poly) + (x_lo * poly)  # Do not re-associate.
  outer = select(is_asymp | ~conds[0], sign_x * tail_mag, central)
  is_median = (y >= _NDTRI_EXP_MEDIAN_LO) & (y <= _NDTRI_EXP_MEDIAN_HI)
  return select(is_median, near_median, outer)
