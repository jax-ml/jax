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

High-level algorithm
--------------------
The Gamma function Gamma(x) extends the factorial function to real and complex
numbers, satisfying Gamma(n) = (n - 1)! for positive integers and the
recurrence Gamma(x + 1) = x * Gamma(x). Because Gamma(x) grows faster than any
exponential (Gamma(36) already overflows float32), numerical libraries compute
its natural logarithm:

    lgamma(x) = ln(|Gamma(x)|)

Across the real line, lgamma(x) is evaluated by partitioning the domain into
three primary regimes:

1. Small and moderate positive inputs (0 < x <= 4.0):
   - On [1.0, 4.0], lgamma(x) is smooth and bounded, with zeros at x = 1 and
     x = 2. We factor out the zeros explicitly as `(x - 1) * (x - 2) * P(x)` and
     evaluate a degree-12 minimax polynomial `P(x)` on each of the two
     sub-intervals [1.0, 2.0] (centered at 1.5) and (2.0, 4.0] (centered at 3.0).
   - On (0, 1.0), Gamma(x) has a simple pole at 0 (Gamma(x) ~ 1/x). We apply
     the recurrence Gamma(x + 1) = x * Gamma(x) in log space:
         lgamma(x) = lgamma(x + 1) - ln(x)
     This shifts x into [1.0, 2.0), where the [1.0, 2.0] polynomial is evaluated.
     (Note: one cannot simply shift all x in (0, 4] up to x + 4 and apply
     Stirling's series, because `lgamma(x + 4) - ln(x(x+1)(x+2)(x+3))` suffers
     catastrophic cancellation near the zeros at x = 1 and x = 2.)

2. Large positive inputs (x > 4.0):
   - Evaluated using Stirling's asymptotic expansion:
         lgamma(x) ~ (x - 0.5) * ln(x) - x + 0.5 * ln(2*pi)
                     + 1/(12*x) - 1/(360*x^3) + 1/(1260*x^5) - ...

3. Negative inputs (x < 0):
   - Reduced to positive inputs via Euler's reflection formula
     (Gamma(x) * Gamma(1 - x) = pi / sin(pi * x)):
         lgamma(x) = ln(pi) - ln(|sin(pi * x)|) - lgamma(1 - x)
     where 1 - x > 1 is evaluated using the positive domain methods above.
   - At non-positive integers (0, -1, -2, ...), Gamma(x) has simple poles and
     lgamma(x) = +inf.

Single-pass SIMD execution
--------------------------
On SIMD/SIMT hardware (CPUs, GPUs, TPUs), conditional branches evaluate both
paths and select elements via `lax.select`. To avoid evaluating the positive
lgamma pipeline twice (once for x > 0 and again for 1 - x when x < 0), we
form a single positive argument `arg_pos = x if x > 0 else 1 - x`, evaluate
the positive lgamma pipeline once on `arg_pos`, and combine the result with
`ln(|sin(pi * x)|)` for negative inputs. Similarly, `_log_hi_lo` is evaluated
only twice across the entire graph: once on `x_gt4` for Stirling's expansion,
and once on a shared argument `log_arg` that supplies `ln(x)` for `0 < x < 1`
and `ln(|delta|)` for `x < 0`.

Numerical refinements for <= 0.71 ULP accuracy across all float32 inputs
------------------------------------------------------------------------
Evaluating the formulas above in native float32 suffers from catastrophic
cancellation in three places, plus subnormal underflow near 0, which we resolve
without float64 promotion or hardware FMA requirements (maximum error 0.7057 ULP):

1. Double-float `(hi, lo)` arithmetic (~48-bit precision):
   Sensitive intermediate sums and products are represented as pairs of
   float32 values `hi + lo` (encapsulated in `_df_add`, `_df_sub`, `_df_mul`,
   and `_horner_df`). For Stirling's formula, we rearrange
   `(x - 0.5) * ln(x) - x` into `(x - 0.5) * (ln(x) - 1) - 0.5` so that only a
   single double-float multiplication is required.

2. Factoring out positive zeros (x = 1 and x = 2):
   Because lgamma(1) = lgamma(2) = 0, summing a power series near 1 or 2
   cancels leading bits. On (0, 4.0], we factor out the zeros explicitly and
   approximate `P(x) = lgamma(x) / ((x - 1)(x - 2))`, computing
   `(x - 1) * (x - 2) * P(x)`.

3. Local Taylor expansions near the 15 negative zeros in (-9.1, -2.0):
   Between each pair of negative integers (-k - 1, -k), Gamma(x) crosses +1 or
   -1 twice where `ln(pi / |sin(pi * x)|) == lgamma(1 - x)`. Near these roots,
   reflection subtracts two ~2.5-magnitude quantities; even a 48-bit
   double-float intermediate leaves ~1e-14 absolute error, which exceeds 1 ULP
   when the result itself is ~1e-7. For k in 2..9 (15 roots in (-9.000003, -2.0)),
   when `x` falls within `0.13 * |r|` of a root, we bypass Euler's reflection
   formula and evaluate a degree-8 Taylor polynomial centered at the
   high-precision root offset `(r_hi, r_lo)` stored in `_NEG_TABLE_NP`.

4. Integer bit-space normalization for subnormal inputs (0 < |x| < 2^-126):
   At the entry of `lgamma_impl`, subnormal float32 inputs are scaled into the
   normal range [2^-126, 2^-125) using integer bit manipulation (`lax.clz`) and
   compensated via an integer exponent shift `sub_shift` passed into `_log_hi_lo`.
   Normalizing in integer registers prevents hardware `FTZ`/`DAZ`
   (Flush-To-Zero / Denormals-Are-Zero) modes on TPU and CPU from flushing
   subnormal inputs to zero.
"""

from __future__ import annotations

from jax._src.api import jit
from jax._src.lax import lax
from jax._src.lax import slicing
import numpy as np


# Largest float32 x with lgamma(x) <= FLT_MAX (0x7C44AF8D); the very next
# representable float32 (4.0850034e36, 0x7C44AF8E) overflows to +inf.
_MAX_FINITE_ARG = 4.085003e36

# Below -2**23 (-8388608.0), the spacing between adjacent float32 numbers is
# >= 1.0, so every representable float32 is an exact integer (a pole of Gamma).
_ALL_INTEGERS_BELOW = -8388608.0

# IEEE-754 float32 bit pattern of sqrt(2)/2 (~0.70710677), used to center the
# mantissa m in [sqrt(2)/2, sqrt(2)) during logarithm argument reduction.
_SQRT2_OVER_2_BITS = 0x3F3504F3

# Exact split of ln(2) = _LN2_HI + _LN2_LO where _LN2_HI has 12 trailing zero
# mantissa bits so that `k * _LN2_HI` is bit-exact in float32 for all integer
# exponents k in [-150, 128].
_LN2_HI = 0.693115234375
_LN2_LO = 3.19461832987e-05


# ==============================================================================
# Part 1: Exact float32 building blocks and Double-Float `(hi, lo)` arithmetic
# ==============================================================================
# `_two_sum`, `_quick_two_sum`, and `_two_prod` compute exact single-float
# addition and multiplication residuals. If JAX adds a hardware-backed `lax.fma`
# primitive in the future, `_two_prod(a, b)` reduces to:
#     p = a * b; return p, lax.fma(a, b, -p)
#
# The leaf helpers below are decorated with `@jit` so JAX traces each building
# block once and emits a single shared subcomputation in the pre-optimization
# HLO (rather than re-tracing and inlining copies at every call site); XLA
# subsequently inlines them during compilation at zero runtime cost.


@jit
def _two_sum(a, b):
  """Adds two float32 numbers and returns (sum, exact_rounding_error).

  When hardware computes `s = a + b`, it rounds `s` to 24 bits and discards
  the lower bits. We recover those discarded bits by subtracting `a` back out
  (`v = s - a`) to see what portion of `b` actually made it into `s`, and then
  measuring the difference from the original inputs (6 HLO instructions).
  """
  s = a + b
  v = s - a
  return s, (a - (s - v)) + (b - v)


@jit
def _quick_two_sum(a, b):
  """FastTwoSum: returns (sum, exact_rounding_error) when |a| >= |b|.

  When `a` is already the high-order part and `b` is a smaller term (`|a| >= |b|`),
  or when `a + b` is already bit-exact by Sterbenz's lemma, subtracting `a` from
  `s = a + b` recovers the exact rounding error in only 3 HLO instructions
  instead of 6.
  """
  s = a + b
  return s, b - (s - a)


@jit
def _split_f32(a):
  """Splits a 24-bit float32 into two 12-bit halves (hi, lo) such that hi + lo == a.

  Why 12 bits? When we multiply two 12-bit numbers together, the result has at
  most 24 bits (`(2^12 - 1)^2 < 2^24`), which fits into a standard 24-bit
  float32 significand (1 implicit bit + 23 fraction bits) with zero rounding
  error. Because 24 is even (`24 = 12 + 12`), we can split `a` simply by
  clearing its bottom 12 mantissa bits with a bitmask (`bits & -4096`, i.e.
  `0xFFFFF000`), leaving the top 12 bits in `hi` and the remaining bottom 12
  bits in `lo = a - hi`.

  Compared to Dekker's `4097.0 * a` multiplication split, bitmask truncation:
    1. Cannot be broken if a compiler contracts `c - a` into an FMA instruction.
    2. Never overflows `4097.0 * a` when `a` is near `FLT_MAX` (`|hi| <= |a|`).
    3. Saves 1 HLO instruction per split.
  """
  bits = lax.bitcast_convert_type(a, np.int32)
  hi = lax.bitcast_convert_type(bits & -4096, np.float32)
  return hi, a - hi


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
  p = a * b
  ah, al = _split_f32(a)
  bh, bl = _split_f32(b)
  err = ((ah * bh - p) + ah * bl) + al * bh + al * bl
  return p, err


@jit
def _df_add(a, b):
  """Adds two `(hi, lo)` double-float pairs: `(a_hi, a_lo) + (b_hi, b_lo)`."""
  s_hi, e = _two_sum(a[0], b[0])
  return _quick_two_sum(s_hi, a[1] + b[1] + e)


@jit
def _df_sub(a, b):
  """Subtracts two `(hi, lo)` double-float pairs: `(a_hi, a_lo) - (b_hi, b_lo)`."""
  return _df_add(a, (-b[0], -b[1]))


@jit
def _df_mul(a, b):
  """Double-float multiplication: returns (hi, lo) ~ (a_hi + a_lo) * (b_hi + b_lo)."""
  p_hi, p_lo = _two_prod(a[0], b[0])
  return _quick_two_sum(p_hi, p_lo + (a[0] * b[1] + a[1] * b[0]))


@jit
def _df_select(cond, a, b):
  """Selects elementwise between two `(hi, lo)` double-float pairs."""
  return (lax.select(cond, a[0], b[0]), lax.select(cond, a[1], b[1]))


def _horner(x, coeffs):
  """Evaluates a polynomial sum(coeffs[i] * x^i) in float32 via Horner's rule."""
  p = (
      lax.full_like(x, coeffs[-1])
      if isinstance(coeffs[-1], (int, float))
      else coeffs[-1]
  )
  for c in reversed(coeffs[:-1]):
    p = p * x + c
  return p


def _horner_df(x_df, coeffs_hi, coeffs_lo):
  """Evaluates a polynomial sum(c_i * x^i) using Horner's rule.

  High-degree terms (which are multiplied by powers of a small offset `x` and
  contribute very little to the total sum) are evaluated using ordinary float32
  arithmetic via `_horner`. The final `k = len(coeffs_lo)` steps (where the
  leading terms dominate the sum) are accumulated using double-float arithmetic
  so that the result retains ~48 bits of precision.
  """
  if not isinstance(coeffs_lo, (list, tuple)):
    coeffs_lo = [coeffs_lo]
  k = len(coeffs_lo)
  x_hi, x_lo = x_df
  acc_hi = _horner(x_hi, coeffs_hi[k:])
  acc_df = (acc_hi, lax.full_like(x_hi, 0.0))
  for i in range(k - 1, -1, -1):
    prod_hi, prod_lo = _two_prod(acc_df[0], x_hi)
    prod_lo = prod_lo + (acc_df[0] * x_lo + acc_df[1] * x_hi)
    s_hi, e = _quick_two_sum(coeffs_hi[i], prod_hi)
    acc_df = _quick_two_sum(s_hi, prod_lo + coeffs_lo[i] + e)
  return acc_df


# ==============================================================================
# Part 2: Extra-precision natural logarithm ln(x)
# ==============================================================================

# Minimax coefficients for Q(f) in ln(1 + f) = f - 0.5*f^2 + f^3 * Q(f) on [-0.293, 0.414].
# The leading coefficient 1/3 is represented as a double-float pair
# (_LOG_Q_HI[0], _LOG_Q_LO0) so that `f^3 * Q(f)` retains >24 bits of accuracy.
_LOG_Q_HI = [
    0.3333333432674408, -0.24999989569187164, 0.19999992847442627,
    -0.16668006777763367, 0.14287430047988892, -0.12451331317424774,
    0.11021918058395386, -0.10625848919153214, 0.10618706792593002,
    -0.06354549527168274,
]
_LOG_Q_LO0 = -1.003832572621377e-08


@jit
def _log_hi_lo(x, sub_shift=0):
  """Computes natural logarithm ln(x * 2^-sub_shift) as a high-precision (hi, lo) pair.

  How it works:
  Any positive normalized float32 number is stored in binary scientific notation as:
      x = 2^k * m
  If the original input to `lgamma_impl` was subnormal, it was pre-scaled by
  `2^sub_shift` into [2^-126, 2^-125), so the true value is `2^(k - sub_shift) * m`.
  By basic logarithm rules:
      ln(x * 2^-sub_shift) = (k - sub_shift) * ln(2) + ln(m)

  To make `ln(m)` easy to approximate with a short polynomial, we want `m` to
  be as close to 1.0 as possible. Specifically, we choose integer `k` so that
  `m` falls between sqrt(2)/2 (~0.707) and sqrt(2) (~1.414).
  In IEEE-754 float32 bits, `0x3f3504f3` (`_SQRT2_OVER_2_BITS`) is the bit
  pattern of sqrt(2)/2. Subtracting `0x3f3504f3` from `x`'s bits and shifting
  right by 23 extracts that exact integer exponent `exp_off` in one step, and
  subtracting `exp_off << 23` from `bits` replaces `x`'s exponent with 0 so
  `m` is in [sqrt(2)/2, sqrt(2)).

  Then `f = m - 1.0` is bit-exact by Sterbenz's lemma (|f| <= 0.414), and we
  evaluate:
      ln(1 + f) = f - 0.5 * f^2 + f^3 * Q(f)
  Because -0.5 is an exact power of 2, `-0.5 * _two_prod(f, f)` yields the exact
  48-bit quadratic term, and `|0.5 * f^2| <= 0.207 * |f| < |f|` satisfies the
  `_quick_two_sum` precondition. Finally we add `(k - sub_shift) * ln(2)` (where
  ln(2) is split into `_LN2_HI + _LN2_LO = 0.693115234375 + 3.19461832987e-05`).
  """
  bits = lax.bitcast_convert_type(x, np.int32)
  exp_off = lax.shift_right_arithmetic(
      bits - _SQRT2_OVER_2_BITS, lax.full_like(bits, 23)
  )
  k = lax.convert_element_type(exp_off - sub_shift, np.float32)
  mant_bits = bits - lax.shift_left(exp_off, lax.full_like(bits, 23))
  m = lax.bitcast_convert_type(mant_bits, np.float32)

  f = m - 1.0
  zero = lax.full_like(f, 0.0)
  f2_hi, f2_lo = _two_prod(f, f)
  q_df = _horner_df((f, zero), _LOG_Q_HI, _LOG_Q_LO0)
  f3_hi, f3_lo = _two_prod(f2_hi, f)
  f3q_df = _df_mul((f3_hi, f3_lo + f2_lo * f), q_df)
  s1, e1 = _quick_two_sum(f, -0.5 * f2_hi)
  ln1pf = _df_add((s1, e1 - 0.5 * f2_lo), f3q_df)
  k_ln2 = (k * _LN2_HI, k * _LN2_LO)
  return _df_add(k_ln2, ln1pf)


# ==============================================================================
# Part 3: Coefficient tables for positive and negative regions
# ==============================================================================

# Minimax polynomial coefficients for lgamma(x) / ((x - 1)(x - 2)) on [1.0, 4.0]:
#   - `_POS_HI_1_2`, `_POS_LO_1_2`: sub-interval [1.0, 2.0], centered at x = 1.50.
#     Also used for x in (0, 1.0) via the shift `x -> x + 1` into [1.0, 2.0).
#   - `_POS_HI_2_4`, `_POS_LO_2_4`: sub-interval (2.0, 4.0], centered at x = 3.00.
# Each `_POS_HI_*` list contains the 13 float32 polynomial coefficients `c_hi[0..12]`,
# with low-order bits in `_POS_LO_*`. On (2.0, 4.0], `c_0..c_2` are stored in
# double-float so multiplication by `(x - 1)(x - 2)` (up to 6.0 at x = 4.0) preserves
# full precision when used in Euler's reflection formula on (-3.0, -2.0).
_POS_HI_1_2 = [
    0.48312896490097046, -0.14595989882946014, 0.06291139870882034,
    -0.03130850940942764, 0.016797112300992012, -0.009424976073205471,
    0.005446053110063076, -0.00322078843601048, 0.001928378245793283,
    -0.0011080558178946376, 0.0006762047996744514, -0.0006071248208172619,
    0.0003772154450416565,
]
_POS_LO_1_2 = [-1.4359989641832271e-08, 0.0, 0.0]

_POS_HI_2_4 = [
    0.3465735912322998, -0.05846821889281273, 0.013149048201739788,
    -0.003332283115014434, 0.0009018019773066044, -0.0002543189038988203,
    7.37710070097819e-05, -2.1898746126680635e-05, 6.5816534515761305e-06,
    -1.8967072037412436e-06, 5.812576091557276e-07, -2.631753943660442e-07,
    8.212163038479048e-08,
]
_POS_LO_2_4 = [
    -9.523271060629668e-10, 1.1056726645364279e-09, 7.500583487640711e-11,
]

# Stirling asymptotic series coefficients for x > 4.0:
# lgamma(x) ~ (x - 0.5)*ln(x) - x + 0.5*ln(2*pi) + (1/x) * S(1/x^2), where
# S(1/x^2) = 1/12 - 1/(360*x^2) + 1/(1260*x^4) - 1/(1680*x^6) + ...
_STIRLING_HI = [
    0.0833333358168602, -0.0027777778450399637, 0.0007936508045531809,
    -0.0005952381179668009, 0.0008417508215643466,
]

# Coefficients for ln(sin(pi * delta) / (pi * delta)) / delta^2 on [-0.5, 0.5],
# with leading coefficient -pi^2 / 6 = _SINC_HI[0] + _SINC_LO0.
_SINC_HI = [
    -1.644934058189392, -0.5411607623100281, -0.3391686975955963,
    -0.24974527955055237, -0.21417342126369476, -0.09055595844984055,
    -0.33258557319641113,
]
_SINC_LO0 = -1.0855913501472969e-08

# Lookup table for the 15 negative roots of lgamma(x) near negative integers -k
# for k in 2..9.
#
# Between each pair of negative integers (-k - 1, -k), Gamma(x) crosses +1 or -1
# twice, meaning lgamma(x) = ln(|Gamma(x)|) has two roots at distance ~1/k! on
# either side of the pole -k:
#   - Why the table stops at k = 9:
#     For k = 9, the root offset from -9 is ~1/9! = 2.76e-6 (~3 float32 ULPs
#     near -9, where 1 ULP = 9.54e-7), so representable float32 inputs land
#     close to the root. For k >= 10, the root offset (~1/10! = 2.76e-7) is
#     smaller than a single float32 ULP near -10 (9.54e-7), so the nearest
#     representable float32 inputs are a full ULP away from the root where
#     |lgamma(x)| is O(1) and Euler's reflection formula is already accurate.
#   - Why column 1 (k = 2, side = 1) is all zeros:
#     The largest negative root is at -2.45702 (side = 0 of k = 2); there is no
#     root in (-2, -1). Storing `r_hi = 0.0` in slot 1 makes `in_root_win`
#     (`|u| <= 0.13 * |r_hi| == 0`) evaluate to False automatically for any
#     non-integer `x`, so slot 1 deliberately doubles as the safe fallback slot
#     for out-of-range k (< 2 or > 9).
#
# Shape: (12 rows, 16 columns), where column index `idx = 2*(k - 2) + side`
# selects one of the 16 slots (8 integer poles k in 2..9 x 2 sides: side = 0 for
# delta < 0 left of -k, side = 1 for delta > 0 right of -k):
#   Row 0..1:  r_hi, r_lo (exact root offset from nearest integer -k)
#   Row 2:     scale      (exact power-of-2 scaling factor for u = delta - r)
#   Row 3..10: c_hi[0..7] (Taylor polynomial coefficients in z = u * scale)
#   Row 11:    c_lo[0]    (low bits of the leading Taylor coefficient c_0)
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
        -1.2500447787999747e-08, 0.0, 1.4702138395605857e-09,
        -8.450832922335394e-09, 2.1247268833235466e-08,
        3.56032217041502e-08, -1.525417303582799e-08,
        1.8587357208943445e-09, 6.518987127890341e-09,
        -1.5823927057567744e-08, -3.8139276625770435e-08,
        -2.1698129515357323e-08, 4.265299224925911e-09,
        6.277530939513554e-09, -1.336677435403999e-08,
        1.3375615282029685e-08,
    ],
], dtype=np.float32)


# ==============================================================================
# Part 4: Domain-specific helper functions (the four primary cases)
# ==============================================================================


def _lgamma_pos_small(x_small, log_x_or_delta):
  """Case 1: Evaluates lgamma(x) on (0, 4.0] via root-factored minimax polynomials.

  For x in [1.0, 4.0], lgamma(x) has zeros at x = 1 and x = 2. To prevent
  cancellation near these zeros, we factor them out explicitly:
      lgamma(x) = (x - 1) * (x - 2) * P(x)
  We evaluate P(x) on two sub-intervals using `_POS_HI_1_2` and `_POS_HI_2_4`:
    - Interval 1: [1.0, 2.0] centered at 1.5
    - Interval 2: (2.0, 4.0] centered at 3.0

  If x < 1.0 (`is_small`), we apply `lgamma(x) = lgamma(x + 1) - ln(x)` by
  shifting `x` up to `x_eval = x + 1` in [1.0, 2.0), evaluating the [1.0, 2.0]
  polynomial there, and subtracting `log_x_or_delta = ln(x)` at the end.
  """
  is_small = x_small < 1.0
  in_gt_20 = x_small > 2.0

  # Compute `d = x_eval - center_offset` in double-float precision:
  # When `is_small` is True (x < 1.0), `x_eval - 1.5 = (x + 1) - 1.5 = x - 0.5`,
  # so we directly subtract `center = 0.5` from `x_small`.
  # Note that `_quick_two_sum(-center, x_small)` is always exact here: for
  # `x_small < 0.25`, `|-0.5| >= |x_small|`; for `x_small in [0.25, 4.0]`,
  # `x_small` and `center` satisfy Sterbenz's lemma (`0.5 <= x_small / center <= 2.0`).
  center = lax.select(
      is_small,
      lax.full_like(x_small, 0.5),
      lax.select(in_gt_20, lax.full_like(x_small, 3.0), lax.full_like(x_small, 1.5)),
  )
  d = _quick_two_sum(-center, x_small)
  coeffs_hi = [
      lax.select(in_gt_20, lax.full_like(x_small, c2), lax.full_like(x_small, c1))
      for c1, c2 in zip(_POS_HI_1_2, _POS_HI_2_4)
  ]
  coeffs_lo = [
      lax.select(in_gt_20, lax.full_like(x_small, c2), lax.full_like(x_small, c1))
      for c1, c2 in zip(_POS_LO_1_2, _POS_LO_2_4)
  ]
  p = _horner_df(d, coeffs_hi, coeffs_lo)

  # Multiply by prefactor (u * v) = (x_eval - 1) * (x_eval - 2).
  # By Sterbenz's Lemma, one of the two factors is always bit-exact in float32:
  # - For x < 1.0:        `x_small - 0.0 == x_small` (with other factor `x_small - 1.0`)
  # - For 1.0 <= x <= 2.0: `x_small - 1.0`           (with other factor `x_small - 2.0`)
  # - For x > 2.0:        `x_small - 2.0`           (with other factor `x_small - 1.0`)
  u_off = lax.select(is_small, lax.full_like(x_small, 0.0), lax.full_like(x_small, 1.0))
  exact_fac = x_small - lax.select(in_gt_20, lax.full_like(x_small, 2.0), u_off)
  other_df = _two_sum(
      x_small, -lax.select(in_gt_20, lax.full_like(x_small, 1.0), u_off + 1.0)
  )
  uv_hi, uv_lo = _two_prod(exact_fac, other_df[0])
  uv = (uv_hi, uv_lo + exact_fac * other_df[1])
  res = _df_mul(uv, p)

  # If x < 1.0, subtract ln(x) to undo the `x + 1` recurrence shift:
  return _df_select(is_small, _df_sub(res, log_x_or_delta), res)


def _lgamma_pos_large(x_gt4):
  """Case 2: Evaluates lgamma(x) for x > 4.0 via Stirling's asymptotic expansion.

  Stirling's formula is rearranged as:
      lgamma(x) = (x - 0.5) * (ln(x) - 1) - 0.5 + 0.5*ln(2*pi) + series(1/x)
  to avoid subtracting two large products (`(x - 0.5)*ln(x) - x`) and require
  only one double-float multiplication.

  At `x_gt4 = _MAX_FINITE_ARG` (`4.085003e36`), `(x - 0.5) * (ln(x) - 1)` is
  `3.40282335e38`, only `0.57 ULP` below `FLT_MAX`. To keep all intermediate
  products inside `_df_mul` and `_two_prod` far below `FLT_MAX`, we pre-scale
  `a` by the exact power of 2 `0.5` (`a_half = 0.5 * x - 0.25`) so that every
  intermediate in `_df_mul` is bounded by `1.7014117e38 <= 0.5 * FLT_MAX`, and
  then scale `prod_half` back by `2.0` with zero rounding error.
  """
  lgx = _log_hi_lo(x_gt4)
  a_half = _quick_two_sum(0.5 * x_gt4, lax.full_like(x_gt4, -0.25))
  prod_half = _df_mul(
      a_half,
      _df_sub(lgx, (lax.full_like(x_gt4, 1.0), lax.full_like(x_gt4, 0.0))),
  )
  prod = (2.0 * prod_half[0], 2.0 * prod_half[1])

  # Add constant term: 0.5 * ln(2*pi) - 0.5 = 0.4189385175704956 + 1.56012466e-8
  # and asymptotic 1/x series: (1/12)/x - (1/360)/x^3 + ...
  inv_x = lax.reciprocal(x_gt4)
  sp = _horner(inv_x * inv_x, _STIRLING_HI)
  tail_hi, tail_e = _two_sum(
      lax.full_like(x_gt4, 0.4189385175704956), inv_x * sp
  )
  tail_lo = tail_e + 1.5601246597875267e-08
  return _df_add(prod, (tail_hi, tail_lo))


def _lgamma_neg_reflection(y_hi, y_lo, delta, log_x_or_delta, lg_pos):
  """Case 3: Evaluates Euler's reflection formula for negative x outside root windows.

  Writing x = n + delta with n = round(x) and |delta| <= 0.5, we have
  |sin(pi * x)| = |sin(pi * delta)| = pi * |delta| * sinc(pi * delta), so the
  `ln(pi)` term in `lgamma(x) = ln(pi) - ln(|sin(pi * x)|) - lgamma(1 - x)`
  cancels analytically, leaving:
      lgamma(x) = -(ln(|delta|) + ln(sinc(pi * delta)) + lgamma(1 - x))
  where `1 - x = y_hi + y_lo` and `lg_pos = lgamma(y_hi)`.

  To account for the tiny rounding residual `y_lo` that was lost when forming
  `y_hi = fl(1 - x)`, we apply a first-order Taylor correction:
      lgamma(y_hi + y_lo) = lgamma(y_hi) + y_lo * lgamma'(y_hi)
  The derivative of lgamma(y) is the digamma function `psi(y)`, approximated by
  `ln(y - 0.5) + 1 / (24 * (y - 0.5)^2)`:
    - For `|x| >= 0.5` (`y >= 1.5`), this asymptotic approximation to `psi(y)`
      is accurate to < 4e-4 and multiplies `|y_lo| <= 0.5 * ulp(y_hi)`.
    - For `-0.5 < x < 0` (`1.0 <= y < 1.5`), `psi(y)` has absolute error <= 0.06:
      when `y_hi > 1.0`, `|y_lo| <= 2^-24` so the error in `y_lo * psi(y)` is
      `<= 0.06 * 2^-24 < 3.6e-9` (< 0.03 ULP); when `y_hi == 1.0` (`x -> 0^-`,
      `y_lo = -x`), `|y_lo * psi(1)| ~ 0.58 * |x|` is added to `ln(|x|)`, so a
      6% error on `0.58 * |x|` is `< 0.04 * |x| / |ln(|x|)| < 1e-8` relative.
  """
  ym05 = y_hi - 0.5
  psi_approx = lax.log(ym05) + 0.041666668 / (ym05 * ym05)
  neg_lg = _df_add(lg_pos, (lax.full_like(delta, 0.0), y_lo * psi_approx))

  # Compute ln(sin(pi * delta) / (pi * delta)) = delta^2 * P(delta^2).
  # Since `delta` is a single float32, `_two_prod(delta, delta)` gives its exact
  # 48-bit square without needing full double-float multiplication, and
  # `|_SINC_HI[0]| = 1.645 > |d2_hi * sinc_p| <= 0.18` satisfies `_quick_two_sum`.
  d2_hi, d2_lo = _two_prod(delta, delta)
  sinc_p = _horner(d2_hi, _SINC_HI[1:])
  sinc_hi, sinc_e = _quick_two_sum(
      lax.full_like(delta, _SINC_HI[0]), d2_hi * sinc_p
  )
  sinc_term = _df_mul((d2_hi, d2_lo), (sinc_hi, sinc_e + _SINC_LO0))

  # Combine: lgamma(x) = -(ln(|delta|) + ln(sinc(pi * delta)) + lgamma(1 - x))
  refl_df = _df_add(_df_add(log_x_or_delta, neg_lg), sinc_term)
  return -refl_df[0]


def _lgamma_neg_root(x_neg_safe, n):
  """Case 4: Evaluates root-centered Taylor polynomials near the 15 negative zeros.

  Between x = -9.000003 and x = -2, `lgamma(x)` crosses zero 15 times (twice in
  each interval (-k - 1, -k) for k in 3..9, and once at -2.45702 in (-3, -2)).
  When `x` is close to one of those roots, the reflection formula subtracts two
  nearly equal numbers (`ln(pi / |sin(pi*x)|)` and `lgamma(1 - x)`), which would
  lose significant bits.

  Instead, for any `x` with integer pole `k = -n_root` in 2..9, we look up the
  nearest root offset `r_hi + r_lo` in `_NEG_TABLE_NP` based on `k` and the sign
  of `delta_root`. We compute the exact distance from the root:
      u = delta_root - (r_hi + r_lo)
  Because each column's scaling factor `scale` in `_NEG_TABLE_NP` is an exact
  power of 2, multiplying `z = u * scale` simply shifts exponent bits and
  incurs zero rounding error in IEEE-754 floating point, normalizing every
  root window `|u_hi| <= 0.13 * |r_hi|` to `|z| <= 0.14` (where the degree-8
  Taylor truncation error `c_8 * z^9` is < 0.15 ULP). When `in_root_win` is
  True, we evaluate the Taylor polynomial `z * Q(z)` centered directly at that
  root.

  Note on the interval (-3, -2):
  The two roots on (-3, -2) are at -2.74768 (near -3) and -2.45702 (near -2).
  Because -2.45702 is very close to the half-integer -2.5, its root window
  `[-2.516, -2.398]` crosses -2.5. If we used `n = round(x)`, inputs in
  `(-2.516, -2.5)` would round to `n = -3` instead of `n = -2` and miss the root
  window. We therefore split (-3, -2) at -2.6 (the midpoint between the two
  roots) when selecting `n_root` for the table lookup. Out-of-range k (< 2 or
  > 9) map to slot 1 (where `r_hi = 0.0`, making `in_root_win` False
  automatically). Returns `(in_root_win, rw)`.
  """
  n_root = lax.select(
      (x_neg_safe > -2.6) & (x_neg_safe < -2.0),
      lax.full_like(x_neg_safe, -2.0),
      n,
  )
  delta_root = x_neg_safe - n_root
  k = lax.convert_element_type(-n_root, np.int32)
  in_k_range = (k >= 2) & (k <= 9)
  side = lax.select(delta_root > 0.0, lax.full_like(k, 1), lax.full_like(k, 0))
  idx = lax.select(in_k_range, 2 * (k - 2) + side, lax.full_like(k, 1))

  table = lax._const(x_neg_safe, _NEG_TABLE_NP)
  idx_exp = lax.broadcast_in_dim(idx, (*idx.shape, 1), tuple(range(idx.ndim)))
  dnums = slicing.GatherDimensionNumbers(
      offset_dims=(0,), collapsed_slice_dims=(1,), start_index_map=(1,)
  )
  cols_arr = slicing.gather(
      table, idx_exp, dnums, slice_sizes=(_NEG_TABLE_NP.shape[0], 1)
  )
  cols = [
      slicing.index_in_dim(cols_arr, i, axis=0, keepdims=False)
      for i in range(_NEG_TABLE_NP.shape[0])
  ]
  r_hi, r_lo, scale = cols[0], cols[1], cols[2]

  # Inside the root window (`|delta_root - r_hi| <= 0.13 * |r_hi|`), `delta_root - r_hi`
  # is Sterbenz-exact, so it is either 0.0 or >= ulp(r_hi) >= 2 * |r_lo|.
  u = _quick_two_sum(delta_root - r_hi, -r_lo)
  in_root_win = lax.abs(u[0]) <= 0.13 * lax.abs(r_hi)
  z_scale = lax.select(in_root_win, scale, lax.full_like(x_neg_safe, 0.0))
  z = (u[0] * z_scale, u[1] * z_scale)

  p_root = _horner_df(z, cols[3:11], cols[11])
  return in_root_win, _df_mul(z, p_root)[0]


# ==============================================================================
# Part 5: Main single-pass float32 implementation
# ==============================================================================


def lgamma_impl(x, *, dtype):
  """Computes lgamma(x) in float32 with <= 0.71 ULP error across the full domain.

  Algorithm overview:
    Step 1: Normalize subnormal inputs (`0 < |x| < 2^-126`) in `int32` bit space
            so hardware `FTZ`/`DAZ` flags cannot flush them to zero. Map
            negative inputs to positive inputs via `arg_pos = 1 - x` (preserving
            the fractional distance `delta = x - round(x)` to the nearest
            integer for `sin(pi * x)`), and leave positive inputs as
            `arg_pos = x`.
    Step 2: Evaluate `lgamma(arg_pos)` on (0, 4.0] via `_lgamma_pos_small`,
            using piecewise minimax polynomials `P(x)` multiplied by
            `(x - 1)(x - 2)` and shifting `x < 1.0` to `x + 1` via
            `lgamma(x) = lgamma(x + 1) - ln(x)`.
    Step 3: Evaluate `lgamma(arg_pos)` on `(4.0, _MAX_FINITE_ARG]` via
            `_lgamma_pos_large`, using Stirling's asymptotic expansion
            `(x - 0.5)*(ln(x) - 1) - 0.5 + 0.5*ln(2*pi) + S(1/x)`, and select
            between Step 2 and Step 3 based on `arg_pos <= 4.0`.
    Step 4: For negative `x` outside root windows, evaluate Euler's reflection
            formula `lgamma(x) = -ln(|delta|) - ln(sinc(pi * delta)) - lgamma(1 - x)`
            via `_lgamma_neg_reflection`.
    Step 5: For negative `x` near one of the 15 roots in (-9.1, -2.0), evaluate
            a root-centered Taylor polynomial from `_NEG_TABLE_NP` via
            `_lgamma_neg_root`.
    Step 6: Select the final positive or negative result and handle special
            values (poles at non-positive integers, +inf overflow, NaN).
  """
  del dtype
  # Step 1: Normalize subnormals (0 < |x| < 2^-126) in int32 bit space so
  # hardware FTZ/DAZ flags on TPU/CPU never flush subnormal inputs to zero.
  # The true subnormal logarithm `ln(|x|)` is restored via `sub_shift` in
  # `_log_hi_lo`, while using `x_norm` (`< 2^-125`) in the O(x) polynomial terms
  # changes `lgamma(1 +- x)` by `< 1.4e-38` against `|ln(|x|)| > 87.3` (< 1e-32 ULP).
  bits = lax.bitcast_convert_type(x, np.int32)
  abs_bits = bits & 0x7FFFFFFF
  is_zero = abs_bits == 0
  is_sub = (abs_bits > 0) & (abs_bits < 0x00800000)
  sub_shift = lax.select(is_sub, lax.clz(abs_bits) - 8, lax.full_like(bits, 0))
  x_norm = lax.bitcast_convert_type(
      lax.select(
          is_sub,
          (bits & -0x80000000) | lax.shift_left(abs_bits, sub_shift),
          bits,
      ),
      np.float32,
  )

  is_pos = x_norm > 0.0
  x_safe = lax.select(is_zero, lax.full_like(x, 1.0), x_norm)

  # Prepare negative reflection argument (1 - x = y_hi + y_lo) and shared first
  # logarithm argument (ln(x) for x in (0, 1.0) or ln(|delta|) for x < 0).
  # Clamp x <= _ALL_INTEGERS_BELOW (-2^23, all exact integer poles) and NaN to
  # -0.5 so round(x) and delta = x - round(x) stay finite and in [-0.5, 0.5].
  x_neg_safe = lax.select(
      (~is_pos) & (x_safe > _ALL_INTEGERS_BELOW),
      x_safe,
      lax.full_like(x, -0.5),
  )
  n = lax.round(x_neg_safe, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = x_neg_safe - n
  abs_delta = lax.abs(delta)
  y_hi, y_lo = _quick_two_sum(1.0 - n, -delta)

  # `arg_pos` is strictly positive (> 0) across all bit patterns (including 0,
  # +-inf, and NaN); clamp only the upper bound to `_MAX_FINITE_ARG`.
  arg_pos = lax.select(is_pos, x_safe, y_hi)
  x_bounded = lax.min(arg_pos, lax.full_like(x, _MAX_FINITE_ARG))
  four = lax.full_like(x, 4.0)
  x_small = lax.min(x_bounded, four)
  x_gt4 = lax.max(x_bounded, four)

  log_arg = lax.select(
      is_pos,
      lax.select(x_small < 1.0, x_small, lax.full_like(x, 1.0)),
      lax.select(abs_delta > 0.0, abs_delta, lax.full_like(x, 1.0)),
  )
  log_x_or_delta = _log_hi_lo(log_arg, sub_shift)

  # Step 2 & Step 3: Positive evaluation on (0, 4.0] and (4.0, _MAX_FINITE_ARG]
  lg_small = _lgamma_pos_small(x_small, log_x_or_delta)
  lg_large = _lgamma_pos_large(x_gt4)
  lg_pos = _df_select(x_bounded <= four, lg_small, lg_large)

  # Step 4 & Step 5: Negative reflection formula and negative root windows
  refl = _lgamma_neg_reflection(y_hi, y_lo, delta, log_x_or_delta, lg_pos)
  in_root_win, rw = _lgamma_neg_root(x_neg_safe, n)
  neg_res = lax.select(in_root_win, rw, refl)

  # Step 6: Select final result and handle special values (poles, inf, NaN)
  res = lax.select(is_pos, lg_pos[0], neg_res)
  is_neg_int = (~is_pos) & (x_safe == lax.floor(x_safe))
  is_pole_or_inf = is_zero | is_neg_int | (x > _MAX_FINITE_ARG)
  res = lax.select(is_pole_or_inf, lax.full_like(x, np.inf), res)
  return lax.select(x != x, lax.full_like(x, np.nan), res)
