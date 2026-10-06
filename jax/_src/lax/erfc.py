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

"""Polynomial approximations for `erfc`, `erfcx`, and the derivative of `erfcx`."""

import math
from jax._src.custom_derivatives import custom_jvp
from jax._src.lax.lax import (
    AccuracyMode, _const, asarray, bitcast_convert_type, clamp, exp, full_like,
    max as lax_max, min as lax_min, neg, optimization_barrier, polynomial,
    reciprocal, select, square,
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
  if x.dtype == np.float32:
    # Mask low 12 mantissa bits -> 12 significant bits -> x_hi^2 is exact.
    x_u = bitcast_convert_type(x, np.uint32)
    x_hi = bitcast_convert_type(x_u & np.uint32(0xFFFFF000), np.float32)
  elif x.dtype == np.float64:
    # Mask low 27 mantissa bits -> 26 significant bits -> x_hi^2 is exact.
    x_u = bitcast_convert_type(x, np.uint64)
    x_hi = bitcast_convert_type(
        x_u & _const(x_u, 0xFFFFFFFFF8000000), np.float64
    )
  else:
    raise TypeError(f"Unsupported dtype for _split_square: {x.dtype}")
  x_lo = x - x_hi
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
    num = _INV_SQRT_PI_F32_HI + optimization_barrier(
        _INV_SQRT_PI_F32_LO + 0.5 * grad_large
    )
    scale = select(ax > 2.0**64, full_like(ax, 2.0**-64), full_like(ax, 1.0))
    return (num * scale) / (ax * scale)
  num = _const(grad_large, _INV_SQRT_PI_F64_HI) + 0.5 * grad_large
  return num / ax


def erfcx_grad_large(ax: Array) -> Array:
  """Computes the derivative `erfcx'(ax)` for `ax >= 1`.

  We write `erfcx'(ax) = -(1 / ax^2) * G(ax)` where `G(ax) = -ax^2 * erfcx'(ax)`
  is positive and bounded in `[0.2731, 0.5642)` on `[1, inf)`. We evaluate
  `G(ax)` using piecewise Sollya minimax polynomials in `u = 1 / ax` on the
  near-field intervals (`[1, 2)` and `[2, 8)` in `float32`; `[1, 2)`, `[2, 4)`,
  and `[4, 12)` in `float64`) and in `t = u^2 = 1 / ax^2` on the asymptotic tail
  (`[8, inf)` in `float32`; `[12, inf)` in `float64`).
  """
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
  return -t * g


def _erfcx_large(ax: Array) -> Array:
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


def erfc_impl(x: Array) -> Array:
  """Computes `erfc(x)` for `float32` and `float64` inputs.

  We split the domain at `|x| = 1`:
  - For `|x| < 1`, `_erfc_small(x)` evaluates `erfc(x)` directly from a minimax
    polynomial for `erf(x) / x - 1`.
  - For `|x| >= 1`, we first compute `erfc(|x|) = exp(-|x|^2) * erfcx(|x|)`.
    Splitting `|x|^2 = s_hi + s_lo` via `_split_square` and evaluating
    `exp(-s_hi) * (y + y * _expm1_lo(-s_lo))` where `y = erfcx(|x|)` prevents
    the `O(x^2)` ULP error growth that would occur if `|x|^2` were rounded
    before calling `exp`. For `x <= -1`, the reflection identity
    `erfc(x) = 2 - erfc(|x|)` gives the result without cancellation since
    `erfc(|x|) <= 0.1573`.
  """
  ax = abs(x)
  small = _erfc_small(clamp(_const(x, -1.0), x, _const(x, 1.0)))

  # Beyond max_ax (11.0 in float32, 28.0 in float64), exp(-x^2) underflows to
  # 0.0 even in subnormals. Clamping ax before _split_square prevents s_hi and
  # s_lo from overflowing to inf for huge inputs, which would otherwise turn
  # exp(-s_hi) * (y + y * _expm1_lo(-s_lo)) into 0.0 * -inf = NaN.
  max_ax = 11.0 if x.dtype == np.float32 else 28.0
  ax_exp = lax_min(ax, _const(ax, max_ax))
  s_hi, s_lo = _split_square(ax_exp)
  y = _erfcx_large(ax)
  expm1_neg_lo = _expm1_lo(-s_lo)
  large_pos = exp(-s_hi, accuracy=AccuracyMode.HIGHEST) * (
      y + y * expm1_neg_lo
  )
  large = select(x < 0.0, 2.0 - large_pos, large_pos)
  return select(ax < 1.0, small, large)


def erfcx_impl(x: Array) -> Array:
  """Computes the scaled complementary error function `erfcx(x) = exp(x^2) * erfc(x)`.

  We divide the real line into three regimes:
  - `|x| < 1`: Evaluates `exp(x^2) * erfc(x)` using `_erfc_small(x)` for
    `erfc(x)` and `_split_square(x)` for `exp(x^2) = exp(s_hi) * (1 +
    expm1(s_lo))`.
    On `[-1, 1)`, `exp(x^2) <= e` and `erfc(x) in (0.1573, 1.8427]`, so both
    factors are well-scaled and free of underflow or overflow.
  - `x >= 1`: Evaluates `_erfcx_large(x)`, which reconstructs `erfcx(x)` from
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

  large_pos = _erfcx_large(ax)
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
