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

"""Polynomial approximations for `erfc`, `erfcx`, `erfinv`, and `erfcinv`."""

import math
from jax._src.custom_derivatives import custom_jvp
from jax._src.lax.lax import (
    AccuracyMode, _const, asarray, bitcast_convert_type, clamp,
    convert_element_type, exp, full_like, log, max as lax_max, min as lax_min,
    neg, optimization_barrier, polynomial, reciprocal, select, select_n, sqrt,
    square,
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
  y = erfcx_large(ax)
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


def _piece_index(conds: tuple[Array, ...]) -> Array:
  if len(conds) == 1:
    return ~conds[0]
  idx = convert_element_type(~conds[0], np.int32)
  for cond in conds[1:]:
    idx = idx + convert_element_type(~cond, np.int32)
  return idx


def _select_piece(w: Array, idx: Array, vals: tuple[float, ...]) -> Array:
  return select_n(idx, *(full_like(w, v) for v in vals))


def _half_mantissa(x: Array) -> Array:
  """Zeros the bottom half of the fraction bits of `x`.

  Because each value uses at most half the available precision (12 of 24 bits
  in float32, 26 of 53 bits in float64), multiplying two such numbers fits in a
  standard float with zero rounding error, without needing hardware FMA.
  """
  if x.dtype == np.float32:
    x_u = bitcast_convert_type(x, np.uint32)
    return bitcast_convert_type(x_u & np.uint32(0xFFFFF000), np.float32)
  elif x.dtype == np.float64:
    x_u = bitcast_convert_type(x, np.uint64)
    return bitcast_convert_type(
        x_u & _const(x_u, 0xFFFFFFFFF8000000), np.float64
    )
  else:
    raise TypeError(f"Unsupported dtype for _half_mantissa: {x.dtype}")


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
  - `erfcinv(y)` passes `x = 1 - y` and `q = min(y, 2 - y)` (`has_far_tail=True`).
  - `ndtri(p)` passes `x = 2 * (p - 0.5)` and `q = 2 * min(p, 1 - p)`
    (`has_far_tail=True`, `is_ndtri=True`).

  When `has_far_tail=False` (`erfinv`), only the pieces needed for `q >= 2^-24`
  (`float32`, 2 pieces) or `q >= 2^-53` (`float64`, 3 pieces) are selected.
  """
  ax = abs(x)
  w = -log(q * (1.0 + ax))
  if x.dtype == np.float32:
    ps = _NDTRI32_P if is_ndtri else _ERFINV32_P
    conds = (w < 3.5, w < 18.0625) if has_far_tail else (w < 3.5,)
    shifts = (3.0, 7.0) if has_far_tail else (3.0,)
    c1 = 1.75
    unroll = 9
    # Near |x| = 1 (large w), erfinv(x) grows like sqrt(w) and ndtri(p) grows
    # like sqrt(2) * sqrt(w). For ndtri, truncate sqrt(2) to half its fraction
    # bits so multiplying `k_h * z_h` below has zero rounding error.
    k_h = float.fromhex("0x1.6a0000p0") if is_ndtri else 1.0
  elif x.dtype == np.float64:
    ps = _NDTRI64_P if is_ndtri else _ERFINV64_P
    conds = (
        (w < 4.5, w < 16.0, w < 37.0, w < 144.0)
        if has_far_tail
        else (w < 4.5, w < 16.0)
    )
    shifts = (3.0, 5.0, 9.0, 19.5) if has_far_tail else (3.0, 5.0)
    c1 = 2.25
    unroll = 20
    k_h = float.fromhex("0x1.6a09e68000000p0") if is_ndtri else 1.0
  else:
    raise TypeError(f"Unsupported dtype for erf_inv_core: {x.dtype}")

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
  #   * `z_h` zeros the low fraction bits of `z`, so `z_h * z_h` is exact.
  #   * Since `w - z_h^2 = (sqrt(w) - z_h) * (sqrt(w) + z_h)`, we can compute
  #     `sqrt(w) - z_h = (w - z_h^2) / (sqrt(w) + z_h)`. Because `z_h * z_h` is
  #     close to `w_safe`, the subtraction `w_safe - z_h * z_h` is exact.
  w_safe = select(conds[0], full_like(w, 1.0), w)
  z = sqrt(w_safe)
  z_h = _half_mantissa(z)
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
  # exact main term `k_h * z_h`. Use `optimization_barrier` so XLA does not
  # regroup the sum into `(k_h * z_h + k_h * z_l) + poly` and round away `z_l`.
  tail_mag = k_h * z_h + optimization_barrier(k_h * z_l + poly)
  sign_x = select(x < 0.0, full_like(x, -1.0), full_like(x, 1.0))
  return select(conds[0], x * poly, sign_x * tail_mag)
