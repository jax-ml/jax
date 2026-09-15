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

"""High-accuracy float32 implementation of lgamma(x) = ln(|Gamma(x)|).

Why this file exists
--------------------
The Gamma function Gamma(x) extends factorials to real numbers: for positive
integers, Gamma(n) = (n - 1)!. Because factorials grow huge very quickly
(35! already overflows float32), numerical libraries compute its natural
logarithm instead:

    lgamma(x) = ln(|Gamma(x)|)

Why naive float32 evaluation loses accuracy
-------------------------------------------
Evaluating standard formulas for `lgamma(x)` directly in float32 suffers from
cancellation in three places:
  1. Near the positive roots x = 1 and x = 2 (where lgamma(1) = lgamma(2) = 0),
     summing a power series cancels the constant and linear terms against
     higher-order terms.
  2. In Stirling's series for large x, subtracting `x` from `(x - 0.5) * ln(x)`
     cancels leading bits.
  3. In Euler's reflection formula for negative x,
     `lgamma(x) = ln(pi / |sin(pi * x)|) - lgamma(1 - x)`, the two terms are
     equal at the 15 negative roots of lgamma(x) in (-9.0, -2.0), so subtracting
     them cancels leading bits.

How we achieve <= 0.70 ULP without float64 or hardware FMA
----------------------------------------------------------
Not all accelerators support `float64` or hardware fused multiply-add (`FMA`).
To maintain accuracy across the domain using only basic `float32` arithmetic:

1. Double-float `(hi, lo)` pairs:
   For sensitive intermediate operations, we represent a number as a pair of
   two `float32` values: `hi` (the main value) and `lo` (the rounding error
   from `hi`). Together, `hi + lo` holds ~48 bits of precision. Arithmetic on
   these pairs is encapsulated in `_df_add`, `_df_sub`, `_df_mul`, `_df_fma`,
   and `_horner_df`. At the end of the calculation, we add `hi + lo` back into
   a single `float32`.

2. Factoring out zeros explicitly:
   Instead of summing a series whose terms cancel to zero at x = 1 and x = 2,
   we approximate `lgamma(x) / ((x - 1)(x - 2))` with a polynomial `P(x)` and
   compute `(x - 1) * (x - 2) * P(x)`. Because `(x - 1)` and `(x - 2)` are
   exact near 1 and 2, the product preserves relative accuracy right down to
   zero. For the 15 negative roots in (-9.0, -2.0), we store their
   high-precision
   locations in a lookup table, compute the exact distance `u = x - root`, and
   evaluate a Taylor series `z * Q(z)` around the root.
"""

from __future__ import annotations

from jax._src.api import jit
from jax._src.lax import lax
from jax._src.lax import slicing
import numpy as np

# ==============================================================================
# Part 1: Exact float32 building blocks and Double-Float `(hi, lo)` arithmetic
# ==============================================================================
# `_two_sum` and `_two_prod` compute exact single-float addition and
# multiplication residuals. If JAX adds a hardware-backed `lax.fma` primitive
# in the future, `_two_prod(a, b)` reduces to `p = a * b; return p, lax.fma(a, b, -p)`.


@jit
def _two_sum(a, b):
  """Adds two float32 numbers and returns (sum, exact_rounding_error).

  When hardware computes `s = a + b`, it rounds `s` to 24 bits and discards
  the lower bits. We recover those discarded bits by subtracting `a` back out
  (`v = s - a`) to see what portion of `b` actually made it into `s`, and then
  measuring the difference from the original inputs.
  """
  s = lax.add(a, b)
  v = lax.sub(s, a)
  err = lax.add(lax.sub(a, lax.sub(s, v)), lax.sub(b, v))
  return s, err


@jit
def _split_f32(a):
  """Splits a 24-bit float32 into two 12-bit halves (hi, lo) such that hi + lo == a.

  Why 12 bits? When we multiply two 12-bit numbers together, the result has at
  most 24 bits, which fits into a standard float32 with zero rounding error!
  Multiplying `a` by 4097 (which is 2^12 + 1) shifts its bits up by 12 places;
  subtracting back rounds away the bottom 12 bits, leaving the top 12 bits in
  `hi` and the remaining bottom 12 bits in `lo = a - hi`.
  """
  c = lax.mul(lax.full_like(a, 4097.0), a)
  hi = lax.sub(c, lax.sub(c, a))
  return hi, lax.sub(a, hi)


@jit
def _two_prod(a, b):
  """Multiplies two float32 numbers and returns (product, exact_rounding_error).

  The exact mathematical product of two 24-bit numbers has 48 bits. Hardware
  `p = a * b` keeps the top 24 bits and throws away the bottom 24 bits.
  By splitting `a = ah + al` and `b = bh + bl` into 12-bit halves, each cross
  product (`ah * bh`, `ah * bl`, `al * bh`, `al * bl`) has at most 24 bits and
  is computed by float32 hardware with zero error. Subtracting `p` from their
  sum leaves the exact bottom 24 bits in `err`.
  """
  p = lax.mul(a, b)
  ah, al = _split_f32(a)
  bh, bl = _split_f32(b)
  err = lax.add(
      lax.add(
          lax.add(lax.sub(lax.mul(ah, bh), p), lax.mul(ah, bl)),
          lax.mul(al, bh),
      ),
      lax.mul(al, bl),
  )
  return p, err


@jit
def _df_add(a, b):
  """Adds two `(hi, lo)` double-float pairs: `(a_hi, a_lo) + (b_hi, b_lo)`."""
  s_hi, e = _two_sum(a[0], b[0])
  return _two_sum(s_hi, lax.add(lax.add(a[1], b[1]), e))


@jit
def _df_sub(a, b):
  """Subtracts two `(hi, lo)` double-float pairs: `(a_hi, a_lo) - (b_hi, b_lo)`."""
  return _df_add(a, (lax.neg(b[0]), lax.neg(b[1])))


@jit
def _df_mul(a, b):
  """Multiplies two `(hi, lo)` double-float pairs: `(a_hi, a_lo) * (b_hi, b_lo)`."""
  p_hi, p_lo = _two_prod(a[0], b[0])
  p_lo = lax.add(p_lo, lax.add(lax.mul(a[0], b[1]), lax.mul(a[1], b[0])))
  return _two_sum(p_hi, p_lo)


@jit
def _df_fma(a, b, c):
  """Fused multiply-add on `(hi, lo)` double-float pairs: `a * b + c`."""
  return _df_add(_df_mul(a, b), c)


@jit
def _df_select(cond, a, b):
  """Selects elementwise between two `(hi, lo)` double-float pairs."""
  return (lax.select(cond, a[0], b[0]), lax.select(cond, a[1], b[1]))


def _horner_df(x_df, coeffs_hi, coeffs_lo=()):
  """Evaluates a polynomial sum(c_i * x^i) using Horner's rule.

  High-degree terms (which contribute very little to the total sum) are
  evaluated using ordinary float32 arithmetic. The final `len(coeffs_lo)` terms
  (which dominate the sum) are evaluated using `_df_fma` so that the
  lowest-order
  terms retain ~48 bits of precision.
  """
  x_hi, _ = x_df
  as_arr = (
      lambda c: lax.full_like(x_hi, c) if isinstance(c, (int, float)) else c
  )
  n_lo = len(coeffs_lo)
  p_hi = as_arr(coeffs_hi[-1])
  for c in reversed(coeffs_hi[n_lo:-1]):
    p_hi = lax.add(lax.mul(p_hi, x_hi), as_arr(c))
  p = (p_hi, lax.full_like(x_hi, 0.0))
  for i in range(n_lo - 1, -1, -1):
    p = _df_fma(p, x_df, (as_arr(coeffs_hi[i]), as_arr(coeffs_lo[i])))
  return p


# ==============================================================================
# Part 2: Extra-precision natural logarithm ln(x)
# ==============================================================================

# Coefficients for ln(1 + f) / f = 1 - 0.5*f + (1/3)*f^2 - ... on [-0.293, 0.414].
# Including 1.0 and -0.5 directly in the coefficient table lets `_horner_df`
# evaluate the entire series in a single pass.
_LOG_C_HI = [
    1.0, -0.5, 0.3333333432674408, -0.25,
    0.20000000298023224, -0.16666659712791443, 0.142856165766716,
    -0.12500311434268951, 0.11115570366382599, -0.09996045380830765,
    0.08999034762382507, -0.08285383135080338, 0.08527489006519318,
    -0.0843496173620224, 0.045338910073041916,
]
_LOG_C_LO = [
    0.0, 0.0, -9.945067880323677e-09, -3.1426775071174973e-10,
    5.030416083684486e-09,
]


@jit
def _log_hi_lo(x):
  """Computes natural logarithm ln(x) as a high-precision (hi, lo) pair.

  How it works:
  Any positive float32 number is stored in binary scientific notation as:
      x = 2^k * m
  By basic logarithm rules:
      ln(x) = k * ln(2) + ln(m)

  To make `ln(m)` easy to approximate with a short polynomial, we want `m` to
  be as close to 1.0 as possible. Specifically, we choose integer `k` so that
  `m` falls between sqrt(2)/2 (~0.707) and sqrt(2) (~1.414).
  In IEEE-754 float32 bits, `0x3f3504f3` is the bit pattern of sqrt(2)/2.
  Subtracting `0x3f3504f3` from `x`'s bits and shifting right by 23 extracts
  that exact integer exponent `k` in one step.

  Then `f = m - 1.0` is small (|f| <= 0.414), and we compute:
      ln(1 + f) = f * P(f)
  using double-float arithmetic via `_horner_df`, and add `k * ln(2)` (where
  ln(2) is split into `0.693115234375 + 3.19461832987e-05`).
  """
  bits = lax.bitcast_convert_type(x, np.int32)
  exp_off = lax.shift_right_arithmetic(
      lax.sub(bits, lax.full_like(bits, 0x3F3504F3)),
      lax.full_like(bits, 23),
  )
  k = lax.convert_element_type(exp_off, np.float32)
  mant_bits = lax.sub(bits, lax.shift_left(exp_off, lax.full_like(bits, 23)))
  m = lax.bitcast_convert_type(mant_bits, np.float32)

  f_df = (lax.sub(m, lax.full_like(x, 1.0)), lax.full_like(x, 0.0))
  ln1pf = _df_mul(f_df, _horner_df(f_df, _LOG_C_HI, _LOG_C_LO))
  k_ln2 = (
      lax.mul(k, lax.full_like(x, 0.693115234375)),
      lax.mul(k, lax.full_like(x, 3.19461832987e-05)),
  )
  return _df_add(k_ln2, ln1pf)


# ==============================================================================
# Part 3: Coefficient tables for positive and negative regions
# ==============================================================================

# Polynomials for lgamma(x) / ((x - 1)(x - 2)) on three sub-intervals of [0.5, 4.0]:
# - _C05: [0.5, 1.0], centered at x = 0.75
# - _C10: [1.0, 2.0], centered at x = 1.50
# - _C20: [2.0, 4.0], centered at x = 3.00
_C05_HI = [
    0.6504990458488464, -0.352359414100647, 0.2940853536128998,
    -0.288911372423172, 0.3072616159915924, -0.3419567942619324,
    0.391918420791626, -0.4597609341144562, 0.5463233590126038,
    -0.6242102384567261, 0.7568618655204773, -1.3452478647232056,
    1.6633952856063843,
]
_C05_LO = [
    -1.2687012551637622e-09, 1.3159803913254109e-08, 1.3123020004002228e-08,
]

_C10_HI = [
    0.48312896490097046, -0.14595989882946014, 0.06291139870882034,
    -0.03130850940942764, 0.016797112300992012, -0.009424976073205471,
    0.005446053110063076, -0.00322078843601048, 0.001928378245793283,
    -0.0011080558178946376, 0.0006762047996744514, -0.0006071248208172619,
    0.0003772154450416565,
]
_C10_LO = [
    -1.4359989641832271e-08, 3.0976332610066493e-09, 2.2513459985162854e-09,
]

_C20_HI = [
    0.3465735912322998, -0.05846821889281273, 0.013149048201739788,
    -0.003332283115014434, 0.0009018019773066044, -0.0002543189038988203,
    7.37710070097819e-05, -2.1898746126680635e-05, 6.5816534515761305e-06,
    -1.8967072037412436e-06, 5.812576091557276e-07, -2.631753943660442e-07,
    8.212163038479048e-08,
]
_C20_LO = [
    -9.523271060629668e-10, 1.1038031599852616e-09, 1.5069562264713454e-10,
]

# Stirling asymptotic series coefficients for x > 4.0:
# lgamma(x) ~ (x - 0.5)*ln(x) - x + 0.5*ln(2*pi) + 1/(12x) - 1/(360x^3) + ...
_STIRLING_HI = [
    0.0833333358168602, -0.0027777778450399637, 0.0007936508045531809,
    -0.0005952381179668009, 0.0008417508215643466,
]

# Coefficients for ln(sin(pi * delta) / (pi * delta)) / delta^2 on [-0.5, 0.5]
_SINC_HI = [
    -1.644934058189392, -0.5411607623100281, -0.3391686975955963,
    -0.24974527955055237, -0.21417342126369476, -0.09055595844984055,
    -0.33258557319641113,
]
_SINC_LO0 = -1.0855913501472969e-08

# Lookup table for the 15 negative roots of lgamma(x) in (-9.0, -2.0).
# Between each pair of negative integers (-k - 1, -k) for k in 2..8, Gamma(x)
# crosses +1 or -1 twice, meaning lgamma(x) = ln(|Gamma(x)|) has two roots.
# (For k = 2, the right side near -2.0 is handled by reflection, so slot 1
# stores r_hi = 0.0, which also serves as the fallback slot for out-of-range k.)
#
# Shape: (15 rows, 16 columns), where column index `idx = 2*(k - 2) + side`
# selects one of the 16 slots (8 integer intervals x 2 sides):
#   Row 0..1:  r_hi, r_lo (exact root offset from nearest integer -k)
#   Row 2:     scale      (exact power-of-2 scaling factor for u = delta - r)
#   Row 3..12: c_hi[0..9] (Taylor polynomial coefficients in z = u * scale)
#   Row 13..14: c_lo[0..1] (low bits of the first two Taylor coefficients)
_NEG_TABLE_NP = np.array([
    [
        -0.4570247530937195, 0.0, -0.14358088374137878, 0.2523173391819,
        -0.03936183825135231, 0.04470571503043175, -0.008218168281018734,
        0.008455359376966953, -0.0013852944830432534, 0.001392519916407764,
        -0.00019833340775221586, 0.00019849211093969643,
        -2.4800270693958737e-05, 2.480290459061507e-05,
        -2.755714831437217e-06, 2.7557489374885336e-06,
    ],
    [
        1.4872918896458032e-08, 0.0, -4.608601056332873e-09,
        1.4090687727730256e-08, -1.4891845534492631e-09,
        1.1097032320828148e-10, -4.157478875055354e-11,
        6.298532528870027e-11, 2.988815445137405e-11,
        2.716606670519206e-12, 4.274642434725501e-13,
        -1.5773943566263493e-12, 1.1999039084651265e-14,
        -4.1127921923157784e-13, 8.786870767129619e-15,
        8.504327342166718e-14,
    ],
    [
        -2.0, 0.0, -8.0, 4.0, -32.0, 16.0, -128.0, 128.0,
        -512.0, 512.0, -4096.0, 4096.0, -32768.0, 32768.0,
        -262144.0, 262144.0,
    ],
    [
        -0.7578017115592957, 0.0, -0.972735583782196, -0.4785875380039215,
        -0.8372025489807129, -1.2953163385391235, -0.9637670516967773,
        -0.9104357957839966, -1.4135481119155884, -1.3989168405532837,
        -1.2314525842666626, -1.2294842004776, -1.2305994033813477,
        -1.2303380966186523, -1.3842945098876953, -1.3842601776123047,
    ],
    [
        1.2145802974700928, 0.0, 0.40361467003822327, 0.5984493494033813,
        0.31665557622909546, 0.983260452747345, 0.4519508183002472,
        0.42695531249046326, 0.9939132928848267, 0.9836257696151733,
        0.7576321959495544, 0.7564211487770081, 0.7571070790290833,
        0.7569462656974792, 0.9581237435340881, 0.9581000208854675,
    ],
    [
        -0.17641133069992065, 0.0, -0.2192753255367279, -0.3139864206314087,
        -0.1667996495962143, -0.9107920527458191, -0.2863674461841583,
        -0.2629375159740448, -0.9342057108879089, -0.9197388887405396,
        -0.6217434406280518, -0.620253324508667, -0.6210972666740417,
        -0.6208994388580322, -0.8842113614082336, -0.8841784596443176,
    ],
    [
        0.545111358165741, 0.0, 0.14377212524414062, 0.24463777244091034,
        0.0993209183216095, 0.955018162727356, 0.2041737586259842,
        0.1822098046541214, 0.987851619720459, 0.9675076603889465,
        0.5740062594413757, 0.5721727013587952, 0.5732110142707825,
        0.5729675889015198, 0.9180009961128235, 0.9179554581642151,
    ],
    [
        -0.18127486109733582, 0.0, -0.10001645982265472, -0.19020982086658478,
        -0.06308446824550629, -1.0681545734405518, -0.155283123254776,
        -0.13469114899635315, -1.114266276359558, -1.0856564044952393,
        -0.5652885437011719, -0.563032329082489, -0.5643097162246704,
        -0.5640101432800293, -1.016666054725647, -1.0166029930114746,
    ],
    [
        0.38790544867515564, 0.0, 0.07257077097892761, 0.1579437553882599,
        0.0417366698384285, 1.244433879852295, 0.12301554530858994,
        0.10370931029319763, 1.30917489528656, 1.268941879272461,
        0.5798759460449219, 0.5770996809005737, 0.5786712169647217,
        0.5783026218414307, 1.1728023290634155, 1.1727150678634644,
    ],
    [
        -0.18698744475841522, 0.0, -0.05395636335015297, -0.13337939977645874,
        -0.02829911559820175, -1.4858304262161255, -0.09987468272447586,
        -0.0818382129073143, -1.5763986110687256, -1.5200252532958984,
        -0.6096206903457642, -0.6062169671058655, -0.6081433296203613,
        -0.6076914668083191, -1.3865357637405396, -1.3864153623580933,
    ],
    [
        0.32012999057769775, 0.0, 0.04109508544206619, 0.1156911849975586,
        0.019655296579003334, 1.8172651529312134, 0.08306203037500381,
        0.06615249067544937, 1.9444022178649902, 1.865140438079834,
        0.6565018892288208, 0.6523144245147705, 0.6546839475631714,
        0.6541280746459961, 1.6791480779647827, 1.6789813041687012,
    ],
    [
        -0.21912552416324615, 0.0, -0.03530412167310715, -0.11308857053518295,
        -0.015398441813886166, -2.5070199966430664, -0.0779184028506279,
        -0.06031518802046776, -2.7051825523376465, -2.5814437866210938,
        -0.7974485754966736, -0.7917285561561584, -0.7949647307395935,
        -0.7942054271697998, -2.29370379447937, -2.293447494506836,
    ],
    [
        0.31931573152542114, 0.0, 0.027713356539607048, 0.10103966295719147,
        0.011023047380149364, 3.160276412963867, 0.06678906083106995,
        0.050249867141246796, 3.4390206336975098, 3.2646872997283936,
        0.8851097822189331, 0.8780583739280701, 0.8820471167564392,
        0.8811110258102417, 2.862947702407837, 2.8625924587249756,
    ],
    [
        -1.2500447787999747e-08, 0.0, 1.4702138395605857e-09,
        -8.450832922335394e-09, 2.1247268833235466e-08,
        3.56032217041502e-08, -1.525417303582799e-08,
        1.8587357208943445e-09, 6.518987127890341e-09,
        -1.5823927057567744e-08, -3.8139276625770435e-08,
        -2.1698129515357323e-08, 4.265299224925911e-09,
        6.277530939513554e-09, -1.336677435403999e-08,
        1.3375615282029685e-08,
    ],
    [
        -5.949705084162815e-08, 0.0, -7.933250500968825e-09,
        -7.124366874222687e-09, 1.2607675792253303e-08,
        2.6201584901741626e-08, 1.3788374886303245e-08,
        7.063126528095154e-09, 1.5556823385054486e-08,
        -1.558556661507282e-08, -9.824473679032053e-09,
        9.969857828195927e-09, -2.3522765957295633e-08,
        2.3668235371587798e-08, 2.9370211152013326e-08,
        -2.9186102423750526e-08,
    ],
], dtype=np.float32)


# ==============================================================================
# Part 4: Main single-pass float32 implementation
# ==============================================================================


def lgamma_impl(x, *, dtype):
  """Computes lgamma(x) in float32 with <= 0.70 ULP error across the full domain.

  How the algorithm is structured (and why it runs in a single pass):
  On vector processors (CPUs, GPUs, TPUs), `if/else` branches are compiled by
  evaluating both paths and selecting elements via `lax.select`. If we called
  our positive `lgamma` helper separately for positive `x` and again inside
  the negative reflection formula, the compiled XLA graph would duplicate
  every polynomial and logarithm operation.

  Instead, we prepare a single positive input `arg_pos`:
    - If x > 0: `arg_pos = x`
    - If x < 0: `arg_pos = 1 - x` (which is positive!)
  We then evaluate positive `lgamma(arg_pos)` ONCE, and use its output directly
  for positive `x` or feed it into Euler's reflection formula for negative `x`.
  """
  del dtype
  one = lax.full_like(x, 1.0)
  zero = lax.full_like(x, 0.0)
  half = lax.full_like(x, 0.5)

  is_zero = lax.eq(x, zero)
  is_pos = lax.gt(x, zero)
  x_safe = lax.select(is_zero, one, x)

  # ----------------------------------------------------------------------------
  # Step 1: Prepare negative reflection argument (for x < 0)
  # ----------------------------------------------------------------------------
  # Euler's reflection formula relates negative inputs to positive inputs:
  #     Gamma(x) * Gamma(1 - x) = pi / sin(pi * x)
  # Taking logs:
  #     lgamma(x) = ln(pi / |sin(pi * x)|) - lgamma(1 - x)
  #
  # Let `n = round(x)` be the nearest negative integer, and `delta = x - n`
  # be the fractional distance to that integer (`delta` is in [-0.5, 0.5]).
  # Because `sin(pi * x) = +/- sin(pi * delta)`, we can write:
  #     |sin(pi * x)| = |delta| * (pi * sin(pi * delta) / (pi * delta))
  # So:
  #     lgamma(x) = -ln(|delta|) - ln(sinc(pi * delta)) - lgamma(1 - x)
  #
  # Subtlety when computing `1 - x = (1 - n) - delta`:
  # Since `1 - n` is a larger integer than `n`, adding `-delta` to `1 - n` in
  # float32 can round away the lowest bit of `delta`. We use `_two_sum` to
  # capture that rounded-off bit in `y_lo`:
  #     y_hi + y_lo = (1 - n) - delta
  x_neg_safe = lax.select(
      lax.bitwise_and(
          lax.bitwise_not(is_pos), lax.gt(x_safe, lax.full_like(x, -8388608.0))
      ),
      x_safe,
      lax.neg(half),
  )
  n = lax.round(x_neg_safe, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = lax.sub(x_neg_safe, n)
  abs_delta = lax.abs(delta)
  y_hi, y_lo = _two_sum(lax.sub(one, n), lax.neg(delta))

  # Single positive argument evaluated by the positive pipeline:
  arg_pos = lax.select(is_pos, x_safe, y_hi)
  x_bounded = lax.select(
      lax.bitwise_and(
          lax.gt(arg_pos, zero), lax.le(arg_pos, lax.full_like(x, 4.085003e36))
      ),
      arg_pos,
      lax.full_like(x, 2.0),
  )

  # ----------------------------------------------------------------------------
  # Step 2: Positive evaluation on (0, 4.0] via root-factored polynomials
  # ----------------------------------------------------------------------------
  # For x in (0, 4.0], lgamma(x) has zeros at x = 1 and x = 2.
  # To prevent cancellation near these zeros, we factor them out explicitly:
  #     lgamma(x) = (x - 1) * (x - 2) * P(x)
  #
  # What if x < 0.5?
  # Near x = 0, Gamma(x) blows up like 1/x, so `lgamma(x)` blows up like `-ln(x)`.
  # Using the factorial step identity `Gamma(x + 1) = x * Gamma(x)`, we get:
  #     lgamma(x) = lgamma(x + 1) - ln(x)
  # So if `x < 0.5` (`is_small`), we shift `x` up by 1 into `x_eval = x + 1`
  # (which lands in [1.0, 1.5]), evaluate the polynomial there, and subtract
  # `ln(x)` at the end.
  #
  # Why `d = _two_sum(x_for_k, -center)` works uniformly across all sub-intervals:
  # For x in [0.5, 4.0], subtracting the sub-interval center `c in {0.75, 1.5, 3.0}`
  # satisfies Sterbenz's Lemma (1/2 <= x / c <= 2), so `x - c` is bit-exact and
  # `_two_sum` automatically returns `lo = 0.0`. For x < 0.5 (`center = 0.5`),
  # `(x + 1) - 1.5 == x - 0.5`, and `_two_sum(x, -0.5)` captures the exact
  # low-order bits of `x` that would otherwise be lost when adding 1.0.
  x_for_k = lax.select(
      lax.le(x_bounded, lax.full_like(x, 4.0)), x_bounded, lax.full_like(x, 2.0)
  )
  is_small = lax.lt(x_for_k, half)
  x_eval = lax.select(is_small, lax.add(x_for_k, one), x_for_k)
  in_05_10 = lax.lt(x_eval, one)
  in_10_20 = lax.bitwise_and(
      lax.ge(x_eval, one), lax.le(x_eval, lax.full_like(x, 2.0))
  )

  # First call to _log_hi_lo, shared between:
  # - positive small x < 0.5: needs ln(x) for `lgamma(x + 1) - ln(x)`
  # - negative x < 0:         needs ln(|delta|) for reflection formula
  log1_arg = lax.select(
      is_pos,
      lax.select(is_small, x_for_k, one),
      lax.select(lax.gt(abs_delta, zero), abs_delta, one),
  )
  l1 = _log_hi_lo(log1_arg)

  def _sel3(v0, v1, v2):
    return lax.select(
        in_05_10,
        lax.full_like(x, v0),
        lax.select(in_10_20, lax.full_like(x, v1), lax.full_like(x, v2)),
    )

  center = lax.select(is_small, half, _sel3(0.75, 1.5, 3.0))
  d = _two_sum(x_for_k, lax.neg(center))
  c_hi = [_sel3(a, b, c) for a, b, c in zip(_C05_HI, _C10_HI, _C20_HI)]
  c_lo = [_sel3(a, b, c) for a, b, c in zip(_C05_LO, _C10_LO, _C20_LO)]
  p = _horner_df(d, c_hi, c_lo)

  # Multiply by prefactor (u * v) = (x_eval - 1) * (x_eval - 2).
  # Note: when x < 0.5 (`is_small`), `x_eval = x + 1`, so `x_eval - 1` is
  # literally `x - 0` (zero rounding error!), and `x_eval - 2` is `x - 1`.
  u_off = lax.select(is_small, zero, one)
  u = _two_sum(x_for_k, lax.neg(u_off))
  v = _two_sum(x_for_k, lax.neg(lax.add(u_off, one)))
  res = _df_mul(_df_mul(u, v), p)

  # If x < 0.5, subtract ln(x) to undo the `x + 1` shift:
  k_pos = _df_select(is_small, _df_sub(res, l1), res)

  # ----------------------------------------------------------------------------
  # Step 3: Positive evaluation for x > 4.0 via Stirling's approximation
  # ----------------------------------------------------------------------------
  # For x > 4, factorials follow Stirling's formula:
  #     lgamma(x) = (x - 0.5) * (ln(x) - 1) - 0.5 + 0.5*ln(2*pi) + series(1/x)
  # Notice we write `(x - 0.5) * (ln(x) - 1) - 0.5` instead of
  # `(x - 0.5)*ln(x) - x` so that we avoid subtracting two large products and
  # only perform one double-float multiplication.
  #
  # Overflow protection for huge x > 2^110 (~1.3e33):
  # If x is near the float32 max (~3.4e38), multiplying `x * ln(x)` overflows
  # to infinity. To prevent intermediate overflow, we scale `x` down by 65536
  # (2^-16), multiply `(x * 2^-16) * (ln(x) - 1)`, and multiply the product
  # back by 65536 at the end. Since 65536 is an exact power of 2, scaling
  # introduces zero rounding error.
  x_gt4 = lax.select(
      lax.gt(x_bounded, lax.full_like(x, 4.0)), x_bounded, lax.full_like(x, 8.0)
  )
  is_huge = lax.gt(x_gt4, lax.full_like(x, 1.2980742e33))
  scale_down = lax.select(is_huge, lax.full_like(x, 1.52587890625e-05), one)
  scale_up = lax.select(is_huge, lax.full_like(x, 65536.0), one)

  lgx = _log_hi_lo(x_gt4)
  a = _two_sum(lax.mul(x_gt4, scale_down), lax.mul(lax.neg(half), scale_down))
  prod = _df_mul(a, _df_sub(lgx, (one, zero)))
  prod_scaled = (lax.mul(prod[0], scale_up), lax.mul(prod[1], scale_up))

  # Add constant term: 0.5 * ln(2*pi) - 0.5 = 0.4189385175704956 + 1.56012466e-8
  # and asymptotic 1/x series: (1/12)/x - (1/360)/x^3 + ...
  const_term = (
      lax.full_like(x, 0.4189385175704956),
      lax.full_like(x, 1.5601246597875267e-08),
  )
  inv_x = lax.reciprocal(x_gt4)
  sp, _ = _horner_df((lax.mul(inv_x, inv_x), zero), _STIRLING_HI)
  s = _df_add(_df_add(prod_scaled, const_term), (lax.mul(inv_x, sp), zero))

  # Select between small-x polynomial (x <= 4) and large-x Stirling (x > 4):
  lg_pos = _df_select(lax.le(x_bounded, lax.full_like(x, 4.0)), k_pos, s)
  pos_res = lax.add(lg_pos[0], lg_pos[1])

  # ----------------------------------------------------------------------------
  # Step 4: Negative reflection formula (away from negative roots)
  # ----------------------------------------------------------------------------
  # Recall that for x < 0, we evaluated positive lgamma at `y_hi`, where
  # `1 - x = y_hi + y_lo`.
  # To account for the tiny rounding bit `y_lo` that was lost when forming `y_hi`,
  # we use a first-order Taylor correction (calculus derivative rule):
  #     lgamma(y_hi + y_lo) = lgamma(y_hi) + y_lo * lgamma'(y_hi)
  # The derivative of lgamma(y) is the digamma function `psi(y)`, which for
  # y >= 1.5 is accurately approximated by `ln(y - 0.5) + 1 / (24 * (y - 0.5)^2)`.
  ym05 = lax.sub(y_hi, half)
  psi_approx = lax.add(
      lax.log(ym05), lax.div(lax.full_like(x, 0.041666668), lax.mul(ym05, ym05))
  )
  neg_lg = _df_add(lg_pos, (zero, lax.mul(y_lo, psi_approx)))

  # Compute ln(sin(pi * delta) / (pi * delta)) = delta^2 * P(delta^2)
  # Since `delta` is a single float32, `_two_prod(delta, delta)` gives its exact
  # 48-bit square without needing full double-float multiplication.
  d2 = _two_prod(delta, delta)
  sinc_term = _df_mul(d2, _horner_df(d2, _SINC_HI, [_SINC_LO0]))

  # Combine: lgamma(x) = -(ln(|delta|) + ln(sinc(pi * delta)) + lgamma(1 - x))
  refl_df = _df_add(_df_add(l1, neg_lg), sinc_term)
  refl = lax.neg(lax.add(refl_df[0], refl_df[1]))

  # ----------------------------------------------------------------------------
  # Step 5: Negative root window lookup (for x near the 15 negative roots)
  # ----------------------------------------------------------------------------
  # Between x = -9 and x = -2, `lgamma(x)` crosses zero 15 times. When `x` is
  # close to one of those roots, the reflection formula subtracts two nearly
  # equal numbers (`ln(|sin(pi*x)|)` and `lgamma(1 - x)`), which would lose
  # precision.
  #
  # Instead, for any `x` in (-9.0, -2.0), we look up the nearest root offset
  # `r_hi + r_lo` in `_NEG_TABLE_NP` based on the integer `k = -n_root` and the
  # sign of `delta_root`. We compute the exact distance from the root:
  #     u = delta_root - (r_hi + r_lo)
  # Because each column's scaling factor `scale` in `_NEG_TABLE_NP` is an exact
  # power of 2, multiplying `z = u * scale` simply shifts exponent bits and
  # incurs zero rounding error in IEEE-754 floating point. If `|u_hi| <= 0.22 * |r_hi|`,
  # we evaluate the Taylor polynomial `z * Q(z)` centered directly at that root.
  #
  # Note on the interval (-3, -2):
  # The two roots on (-3, -2) are at -2.74768 (near -3) and -2.45702 (near -2).
  # Because -2.45702 is very close to the half-integer -2.5, its root window
  # [-2.5576, -2.3565] crosses -2.5. If we used `n = round(x)`, inputs in
  # [-2.5576, -2.5) would round to n = -3 instead of n = -2 and miss the root
  # window. We therefore split (-3, -2) at -2.6 (the midpoint between the two
  # roots) when selecting `n_root` for the table lookup. Out-of-range k (< 2 or > 9)
  # map to slot 1 (where r_hi = 0.0, making `in_root_win` False automatically).
  n_root = lax.select(
      lax.bitwise_and(
          lax.gt(x_neg_safe, lax.full_like(x, -2.6)),
          lax.lt(x_neg_safe, lax.full_like(x, -2.0)),
      ),
      lax.full_like(x, -2.0),
      n,
  )
  delta_root = lax.sub(x_neg_safe, n_root)
  k = lax.convert_element_type(lax.neg(n_root), np.int32)
  in_k_range = lax.bitwise_and(
      lax.ge(k, lax.full_like(k, 2)), lax.le(k, lax.full_like(k, 9))
  )
  s_idx = lax.select(
      lax.gt(delta_root, zero), lax.full_like(k, 1), lax.full_like(k, 0)
  )
  raw_idx = lax.add(
      lax.mul(lax.sub(k, lax.full_like(k, 2)), lax.full_like(k, 2)), s_idx
  )
  idx = lax.select(in_k_range, raw_idx, lax.full_like(k, 1))

  idx_exp = lax.broadcast_in_dim(idx, (*x.shape, 1), tuple(range(x.ndim)))
  dnums = slicing.GatherDimensionNumbers(
      offset_dims=(0,), collapsed_slice_dims=(1,), start_index_map=(1,)
  )
  cols = slicing.gather(
      lax._const(x, _NEG_TABLE_NP), idx_exp, dnums, slice_sizes=(15, 1)
  )
  col = lambda i: slicing.index_in_dim(cols, i, axis=0, keepdims=False)

  r = (col(0), col(1))
  u = _df_sub((delta_root, zero), r)
  in_root_win = lax.le(
      lax.abs(u[0]), lax.mul(lax.full_like(x, 0.22), lax.abs(r[0]))
  )
  scale = lax.select(in_root_win, col(2), zero)
  z = (lax.mul(u[0], scale), lax.mul(u[1], scale))

  p_root = _horner_df(
      z, [col(3 + i) for i in range(10)], [col(13 + i) for i in range(2)]
  )
  rw_df = _df_mul(z, p_root)
  rw = lax.add(rw_df[0], rw_df[1])

  # ----------------------------------------------------------------------------
  # Step 6: Select final result and handle special values (poles, inf, NaN)
  # ----------------------------------------------------------------------------
  neg_res = lax.select(in_root_win, rw, refl)
  res = lax.select(is_pos, pos_res, neg_res)

  # Non-positive integers (0, -1, -2, ...) are poles where Gamma(x) = +/-inf,
  # so lgamma(x) = +inf. Also return +inf if x > 4.085e36 (where lgamma overflows).
  is_neg_int = lax.bitwise_and(
      lax.bitwise_not(is_pos), lax.eq(x_safe, lax.floor(x_safe))
  )
  is_pole_or_inf = lax.bitwise_or(
      lax.bitwise_or(is_zero, is_neg_int),
      lax.gt(x, lax.full_like(x, 4.085003e36)),
  )
  res = lax.select(is_pole_or_inf, lax.full_like(x, np.inf), res)
  return lax.select(lax.ne(x, x), lax.full_like(x, np.nan), res)
