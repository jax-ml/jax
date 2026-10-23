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

"""Float32 and float64 implementations of lgamma(x).

Approach overview:
- Positive arguments (x >= 0.5 in float32, x > 0 in float64) are evaluated
  directly on the positive half-line:
  1. Near x = 1 and x = 2: evaluated via factored minimax polynomials
     (x - 1)(x - 2)P(d) that vanish identically at the roots x = 1 and x = 2.
     Float32 splits [0.5, 4] across [0.5, 1.5] (d = x - 1.0) and (1.5, 4]
     (d = x - 2.75), using 12-bit mantissa truncation (`& -0x1000`) in pure
     float32 so the product retains precision under reflection. Float64 shifts
     (0, 1) to x + 1, uses one polynomial on [1, 2.5) (d = x - 1.5), and shifts
     [2.5, 8) up to x + m >= 8.
  2. Larger x (x > 4 in float32, x >= 8 in float64): evaluated via the
     asymptotic Stirling series
     lgamma(x) = (x - 0.5)(ln(x) - 1) + (0.5 ln(2 pi) - 0.5) + (1/x) S(1/x^2),
     using 12-bit splits in pure float32 (or (hi, lo) pairs in float64) on
     (x - 0.5) and (ln(x) - 1) so the leading product p_hi = b_hi * (h_x - 1)
     is exact.
- Remaining arguments (x < 0.5 in float32, x <= 0 in float64)
  reflect to y = 1 - x via two complementary paths:
  1. Euler reflection: evaluates
     lgamma(x) = -(ln|delta| + lgamma(1 - x) - ln(pi delta / sin(pi delta))),
     where n = round(x), delta = x - n in [-0.5, 0.5], and y = (1 - n) - delta.
     In pure float32, a first-order Taylor correction psi(y) * y_lo compensates
     for rounding in y.
  2. Root windows near negative zeros: near zeros of lgamma(x) on (-10, -2) in
     float32 or (-17, -2) in float64,
     where Euler reflection suffers catastrophic cancellation, each root
     x_root = -k + r (k in {2, ..., 9} in float32, {2, ..., 16} in float64,
     side in {0, 1}) lies close to the
     logarithmic pole at x = -k (delta = 0). A direct polynomial in delta - r
     converges slowly because of that nearby pole; instead, we apply the
     rational change of variable (a Cayley transform)
     w = (delta - r) / (delta + r), which maps the root delta = r to w = 0 and
     turns the pole term -ln(delta / r) = -ln((1 + w) / (1 - w)) = -2 * atanh(w)
     into a rapidly converging odd power series. Each window is evaluated as
     w * P(w) using a lookup table of ULP-weighted minimax polynomials
     (14 x 16 degree-10 in float32; 25 x 30 degree-20 in float64).

Two intermediate-precision modes are supported:
- Pure float32 (`use_f64_for_f32=False`, used on GPU and TPU) uses float32
  polynomials, 12-bit mantissa splits, `(hi, lo)` pairs, the `psi(y) * y_lo`
  reflection correction, one residual refinement step on `w`, and `_NEG_TABLE`.
- Float64 intermediates (`use_f64_for_f32=True`, used on CPU) use higher-degree
  float64 minimax polynomials (~49-52 bits, enough to absorb up to 23.3 bits of
  cancellation near negative roots in float32) so Euler's reflection formula is
  accurate across all negative inputs without `_NEG_TABLE`.
"""

from __future__ import annotations

from jax._src import config
from jax._src.api import jit
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax.double_word import (
    f2_add,
    f2_add_ordered,
    f2_mul,
    f2_mul_exact_high,
    f2_sum,
    fast_two_sum,
    split_half,
    two_prod,
    two_prod_big,
    two_sum,
)
import numpy as np


# Largest float32 whose lgamma is finite in float32 (lgamma(4.085003e36) =
# 3.4028234e38; the next float32 up overflows to +inf).
_MAX_FINITE_ARG = 4.085003e36
_ALL_INTEGERS_BELOW = -8388608.0
_SQRT2_OVER_2_BITS = 0x3F3504F3
_LN2_HI = 0.693115234375
_LN2_LO = 3.19461832987e-05
# Constant term 0.5 * ln(2 * pi) - 0.5 in Stirling's expansion when the leading
# product is written as (x - 0.5) * (ln(x) - 1) = (x - 0.5) * ln(x) - x + 0.5.
_HALF_LN_2PI_MINUS_HALF = 0.4189385175704956
_HALF_LN_2PI_MINUS_HALF_F64 = 0.41893853320467274
_LN2_F64 = 0.6931471805599453


# Degree-9 Sollya fpminimax polynomial for (ln(1 + f) - (f - 0.5 f^2)) / f^3 on
# f = m - 1 in [sqrt(2)/2 - 1, sqrt(2) - 1], stored in ascending powers of f.
# Generated with Sollya:
#   prec = 300!;
#   p = fpminimax(log1p(x), [|3, 4, 5, 6, 7, 8, 9, 10, 11, 12|], [|SG...|],
#                 [sqrt(2)/2 - 1; sqrt(2) - 1], relative, floating,
#                 x - 0.5 * x^2);
#   for i from 3 to 12 do print(single(coeff(p, i)));
_LOG_Q_HI = [
    0.3333333134651184, -0.24999968707561493, 0.20000173151493073,
    -0.1666935533285141, 0.1428280621767044, -0.12423781305551529,
    0.11063456535339355, -0.10845641046762466, 0.10516089200973511,
    -0.05804464966058731,
]

# Degree-16 float64 minimax polynomial for (ln(1 + f) - (f - 0.5 f^2)) / f^3 on
# f = m - 1 in [sqrt(2)/2 - 1, sqrt(2) - 1] (~51.4-bit relative accuracy).
_LOG_Q_F32_F64 = [
    0.3333333333333261, -0.25000000000345535, 0.20000000000697915,
    -0.16666666594619337, 0.1428571412724866, -0.12500005436363315,
    0.11111125324064747, -0.09999801375585593, 0.09090283898105432,
    -0.08337109600190651, 0.07707064490152053, -0.07108586385305052,
    0.06478328342758392, -0.06310141921529044, 0.07058883671019928,
    -0.06663032331509593, 0.030711639087442108,
]


@jit
def _log_f32(x):
  """Computes ln(x) as a compensated (log_hi, log_lo) pair."""
  bits = lax.bitcast_convert_type(x, np.int32)
  exp_off = lax.shift_right_arithmetic(
      bits - _SQRT2_OVER_2_BITS, lax.full_like(bits, 23)
  )
  k = lax.convert_element_type(exp_off, np.float32)
  mant_bits = bits - lax.shift_left(exp_off, lax.full_like(bits, 23))
  m = lax.bitcast_convert_type(mant_bits, np.float32)

  # TODO(phawkins): To reduce worst-case error near x in (0.7, 1.0) and x ~ -5,
  # split f with _split12(f) and use 2Sum on (f, -0.5 * f_hi^2) and (h_d, q_hi).
  f = m - 1.0
  f2 = f * f
  ln1pf = f - 0.5 * f2 + (f2 * f) * lax.polynomial(
      f, _LOG_Q_HI, unroll=len(_LOG_Q_HI)
  )
  h_d = k * _LN2_HI
  l_d = ln1pf + k * _LN2_LO
  log_hi = h_d + l_d
  log_lo = (h_d - log_hi) + l_d
  return log_hi, log_lo


@jit
def _log_f32_with_f64(x):
  """Computes ln(x) in float64 using a float64 minimax polynomial."""
  bits = lax.bitcast_convert_type(x, np.int32)
  exp_off = lax.shift_right_arithmetic(
      bits - _SQRT2_OVER_2_BITS, lax.full_like(bits, 23)
  )
  k = lax.convert_element_type(exp_off, np.float64)
  mant_bits = bits - lax.shift_left(exp_off, lax.full_like(bits, 23))
  m = lax.bitcast_convert_type(mant_bits, np.float32)

  f_d = lax.convert_element_type(m - 1.0, np.float64)
  q = lax.polynomial(f_d, _LOG_Q_F32_F64, unroll=len(_LOG_Q_F32_F64))
  f2_d = f_d * f_d
  poly_d = f_d - 0.5 * f2_d + (f2_d * f_d) * q
  return k * _LN2_F64 + poly_d


# Degree-12 Sollya fpminimax polynomials for lgamma(x) / ((x - 1)(x - 2)) in
# ascending powers of d = x - center:
# - _POS_HI_05_15: fitted on x in [0.5, 1.5] around center = 1.0.
# - _POS_HI_15_4: fitted on x in [1.5, 4.0] around center = 2.75.
# Generated with Sollya (using a shifted Stirling series for lgamma(1 + x)):
#   prec = 300!;
#   z = 1 + x + 20;
#   lg1 = (z - 0.5)*log(z) - z + 0.5*log(2*pi)
#       + (1/z)*(1/12 - (1/z^2)/360 + (1/z^4)/1260 - (1/z^6)/1680 + (1/z^8)/1188
#       - 691*(1/z^10)/360360 + (1/z^12)/156 - 3617*(1/z^14)/122400
#       + 43867*(1/z^16)/244188 - 174611*(1/z^18)/125400);
#   for k from 0 to 19 do lg1 = lg1 - log(1 + x + k);
#   p1 = fpminimax(lg1 / (x - 1), [|1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13|],
#                  [|SG...|], [-0.5; 0.5], relative, floating);
#   for i from 1 to 13 do print(single(coeff(p1, i)));
#   lg2 = lg1(x + 1);
#   q2 = remez(lg2 / ((x + 1) * x), 24, [-0.5; 2.0]);
#   p2 = fpminimax(q2(x + 0.75), 12, [|SG...|], [-1.25; 1.25],
#                  relative, floating);
#   for i from 0 to 12 do print(single(coeff(p2, i)));
_POS_HI_05_15 = [
    0.5772156715393066, -0.24525131285190582, 0.15543334186077118,
    -0.1151529923081398, 0.09227604418992996, -0.0771135687828064,
    0.06608173251152039, -0.06148973107337952, 0.058332279324531555,
    -0.03065679967403412, 0.016753701493144035, -0.08664266765117645,
    0.09459882974624634,
]

_POS_HI_15_4 = [
    0.3620683252811432, -0.06572934240102768, 0.016030851751565933,
    -0.004419584292918444, 0.0013036809395998716, -0.00040008797077462077,
    0.00012487791536841542, -4.1899154894053936e-05, 1.5541731045232154e-05,
    -3.445456741246744e-06, 2.1466885868903773e-07, -9.512536394140625e-07,
    5.093689878776786e-07,
]

# Sollya fpminimax coefficients in ascending powers of 1/x^2 for the Stirling
# remainder S(1/x^2) = B_2/(1*2) + B_4/(3*4) x^-2 + ... on x >= 4.
# Generated with Sollya:
#   prec = 300!;
#   s = 1/12 - x/360 + x^2/1260 - x^3/1680 + x^4/1188
#       - 691*x^5/360360 + x^6/156 - 3617*x^7/122400 + 43867*x^8/244188;
#   p = fpminimax(s, 4, [|SG...|], [0; 1/16], relative, floating);
#   for i from 0 to 4 do print(single(coeff(p, i)));
_STIRLING_HI = [
    0.0833333358168602, -0.0027783112600445747, 0.000824464950710535,
    -0.0012502220924943686, 0.005375693552196026,
]

# Sollya fpminimax coefficients in ascending powers of delta^2 for
# -ln(pi * delta / sin(pi * delta)) / delta^2 on |delta| <= 0.5.
# Generated with Sollya:
#   prec = 300!;
#   f = log(sin(pi * sqrt(x)) / (pi * sqrt(x))) / x;
#   p = fpminimax(f, 6, [|SG...|], [1b-40; 0.25], relative, floating);
#   for i from 0 to 6 do print(single(coeff(p, i)));
_SINC_HI = [
    -1.644934058189392, -0.5411619544029236, -0.3391309082508087,
    -0.2502579689025879, -0.21078622341156006, -0.10131293535232544,
    -0.3194314241409302,
]

# Degree-23 float64 Sollya fpminimax polynomials for lgamma(x) / ((x - 1)(x - 2))
# in ascending powers of d = x - center, fitted for float32 lgamma reflection:
# - _POS_05_15_F64: x in [0.5, 1.5] around center = 1.0 (~49.5-bit rel accuracy).
# - _POS_15_4_F64: x in [1.5, 4.0] around center = 2.75 (~52.1-bit rel accuracy).
_POS_05_15_F64 = [
    0.5772156649015322, -0.2452513685225703, 0.1554342658647344,
    -0.11514654256790671, 0.09223900831635617, -0.07731816804434827,
    0.06673173918441967, -0.05877796840964052, 0.05255590945458782,
    -0.04754229079873885, 0.04342012573655991, -0.039959068720920164,
    0.03685996405243268, -0.03424304179664488, 0.033410878458216185,
    -0.03193612275579251, 0.021414012659978037, -0.017813063448699765,
    0.053308287436331876, -0.05757720110254376, -0.04170677971103274,
    0.05084438630620622, 0.09857378154450186, -0.1022005082529236,
]

_POS_15_4_F64 = [
    0.36206831764947583, -0.06572934792248143, 0.016030995033300615,
    -0.004419488380429822, 0.0013029063335328588, -0.0004005636052229375,
    0.0001266990626986635, -4.0898695804588575e-05, 1.3403831584135535e-05,
    -4.4444160863060155e-6, 1.4873885491971618e-6, -5.014860559989504e-7,
    1.6994335939762067e-7, -5.7915617286448015e-8, 2.0090005970320186e-8,
    -6.9370808324906276e-9, 2.1333154316934285e-9, -7.153446240703879e-10,
    4.207448247253875e-10, -1.5630927610098696e-10, -1.8647641985504433e-11,
    9.089155242178041e-12, 1.459639735381569e-11, -5.396724340229409e-12,
]

# Degree-6 float64 minimax coefficients in ascending powers of 1/x^2 for the
# Stirling remainder S(1/x^2) on x >= 4 (~47.7-bit absolute accuracy in S).
_STIRLING_F32_F64 = [
    0.08333333333332908, -0.002777777770848688, 0.0007936489447267786,
    -0.0005950513428842073, 0.0008325836509206451, -0.0016760106456934216,
    0.0028786348131155308,
]

# Degree-11 float64 minimax coefficients in ascending powers of delta^2 for
# -ln(pi * delta / sin(pi * delta)) / delta^2 on |delta| <= 0.5 (~49.1-bit rel
# accuracy).
_SINC_F32_F64 = [
    -1.6449340668482237, -0.5411616168587046, -0.3391143534032649,
    -0.25101938254889333, -0.20019726882272687, -0.16674427259742153,
    -0.142353419429977, -0.1296848567043034, -0.08295902231943517,
    -0.20914242809307546, 0.16492085267622705, -0.3875246085244428,
]

# Negative-argument root lookup table of shape (14, 16) for zeros x_root = -k + r
# of lgamma(x) near negative integers -k (k in {2, ..., 9}).
#
# From Gamma(-k + delta) = Gamma(delta) / prod_{i=1}^k (delta - i), one obtains
# lgamma(-k + delta) = -ln|delta / r| + [f(delta) - f(r)] with f smooth. Under
# the rational substitution w = (delta - r) / (delta + r) (equivalently
# delta / r = (1 + w) / (1 - w)), the singular part is identically
# -ln(delta / r) = -2 * atanh(w) = -2 * (w + w^3/3 + w^5/5 + ...),
# which makes the window well-conditioned in w and explains why the even-indexed
# coefficients converge to c_0 ~ -2, c_2 ~ -2/3, c_4 ~ -2/5, c_6 ~ -2/7.
#
# Column index is slot = 2 * (k - 2) + side, where side = 0 (r < 0, delta < 0)
# or side = 1 (r > 0, delta > 0). Column 1 (k = 2, side = 1) is an inactive
# sentinel slot (`w_bound = -1.0`) because lgamma(x) > 0 on (-2, -1).
# Rows contain:
#   row 0 (`r_hi`): float32 leading part of root offset r = x_root - (-k).
#   row 1 (`r_lo`): residual float32(r - r_hi) so r_hi + r_lo captures r to ~48 bits.
#   row 2 (`w_bound`): half-width threshold on the variable
#                      w = (delta - r) / (delta + r) activating the root window.
#   rows 3..13 (`c_0 .. c_10`): degree-10 polynomial coefficients in ascending
#       powers of w, fitted via iteratively reweighted least squares (Lawson/Remez)
#       to minimize output-ULP error |w * P(w) - lgamma(-k + delta)| / ulp(lgamma).
_NEG_TABLE = np.array([
    [
        -0.4570247530937195, 1.0, -0.14358088374137878, 0.2523173391819,
        -0.03936183825135231, 0.04470571503043175, -0.008218168281018734,
        0.008455359376966953, -0.0013852944830432534, 0.001392519916407764,
        -0.00019833340775221586, 0.00019849211093969643,
        -2.4800270693958737e-05, 2.480290459061507e-05,
        -2.755714831437217e-06, 2.7557489374885336e-06,
    ],
    [
        1.4872918896458032e-08, 0.0, -4.608601056332873e-09,
        1.4090687727730256e-08, -1.4891843314046582e-09,
        1.1097014973593389e-10, -4.15751669202713e-11,
        6.298560284445642e-11, 2.9888522212750956e-11,
        2.7165145133345447e-12, 4.276554967357765e-13,
        -1.5769613262786586e-12, 1.1601349492618845e-14,
        -4.116331028206771e-13, 8.029728344369734e-15,
        8.437904373695751e-14,
    ],
    [
        0.255, -1.0, 0.34, 0.37, 0.42, 0.42, 0.42, 0.42,
        0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42,
    ],
    [
        -1.3853365182876587, 0.0, -2.2346599102020264, -0.9660475254058838,
        -2.1090452671051025, -1.8530573844909668, -2.0276224613189697,
        -1.9707038402557373, -2.0051767826080322, -1.9947720766067505,
        -2.0007991790771484, -1.999199628829956, -2.0001060962677,
        -1.9998937845230103, -2.0000123977661133, -1.9999876022338867,
    ],
    [
        2.6737210750579834, 0.0, -0.10455801337957382, 1.472333550453186,
        -0.09949951618909836, 0.1592523604631424, -0.027202462777495384,
        0.02974076010286808, -0.005164650734513998, 0.005240116734057665,
        -0.0007990428712219, 0.0008006766438484192,
        -0.00010617126827128232, 0.00010619792737998068,
        -1.2454861462174449e-05, 1.2445724678400438e-05,
    ],
    [
        5.654962062835693, 0.0, -0.6329700946807861, 1.3282930850982666,
        -0.7565763592720032, -0.49503886699676514, -0.6934509873390198,
        -0.6364829540252686, -0.6718208193778992, -0.6614158749580383,
        -0.667466938495636, -0.6658671498298645, -0.6667742729187012,
        -0.6665619015693665, -0.6666803359985352, -0.6666555404663086,
    ],
    [
        13.648747444152832, 0.0, 0.18435128033161163, 2.663341760635376,
        -0.08025430887937546, 0.18398988246917725, -0.02635575272142887,
        0.030618861317634583, -0.005139326676726341, 0.005263099446892738,
        -0.0007983599789440632, 0.0008009925368241966,
        -0.00010607429430820048, 0.00010612390178721398,
        -1.0810486855916679e-05, 1.1192499187018257e-05,
    ],
    [
        29.05604362487793, 0.0, -0.047576695680618286, 3.1682286262512207,
        -0.47040924429893494, -0.2035086750984192, -0.42581096291542053,
        -0.36883100867271423, -0.405023992061615, -0.39462560415267944,
        -0.40070071816444397, -0.3991020619869232, -0.4000096917152405,
        -0.3997977674007416, -0.3999229967594147, -0.39989691972732544,
    ],
    [
        64.69268798828125, 0.0, 0.5535060167312622, 4.828585147857666,
        -0.06051565706729889, 0.2127881944179535, -0.0257291067391634,
        0.0318717323243618, -0.005158360581845045, 0.005333737935870886,
        -0.0008044287096709013, 0.0008080150000751019,
        -0.0001079421053873375, 0.00010787778592202812,
        -3.5387503885431215e-05, 2.9538590752054006e-05,
    ],
    [
        144.2744140625, 0.0, 0.5029898881912231, 6.225178241729736,
        -0.3385486900806427, -0.058972425758838654, -0.31383317708969116,
        -0.25587204098701477, -0.2933040261268616, -0.28273552656173706,
        -0.28885287046432495, -0.2872275114059448, -0.2881371080875397,
        -0.2879173457622528, -0.28795185685157776, -0.2879430055618286,
    ],
    [
        354.1711730957031, 0.0, 0.874119758605957, 9.207906723022461,
        -0.04211042821407318, 0.19476988911628723, -0.021956387907266617,
        0.027959952130913734, -0.004527992568910122, 0.004708156455308199,
        -0.0007103931275196373, 0.0007148765143938363,
        -8.80396764841862e-05, 8.913140482036397e-05,
        0.00013230105105321854, -9.354341455036774e-05,
    ],
    [
        972.521484375, 0.0, 0.8290212154388428, 15.62746810913086,
        -0.2303585410118103, -0.011887536384165287, -0.2158958464860916,
        -0.16832894086837769, -0.2014114260673523, -0.1927308589220047,
        -0.19840183854103088, -0.19707120954990387, -0.1979275345802307,
        -0.19777315855026245, -0.19833749532699585, -0.19823305308818817,
    ],
    [
        2111.91357421875, 0.0, 2.9675049781799316, 24.958375930786133,
        -0.012698682025074959, 0.4708675146102905, -0.03855019807815552,
        0.05648542195558548, -0.008209493942558765, 0.008694824762642384,
        -0.0012875236570835114, 0.0012958142906427383,
        -0.00018785701831802726, 0.00018500936857890338,
        -0.00034586386755108833, 0.0002531808859203011,
    ],
    [
        2133.65771484375, 0.0, 3.8766984939575195, 18.714889526367188,
        -0.28982892632484436, 0.24794331192970276, -0.34019771218299866,
        -0.23934946954250336, -0.30434995889663696, -0.286053866147995,
        -0.2954479455947876, -0.2926260828971863, -0.2939679026603699,
        -0.2935481369495392, -0.29285988211631775, -0.29295918345451355,
    ],
], dtype=np.float32)


def _lgamma_pos_small(x_small):
  """Evaluates lgamma(x) on [0.5, 4] and returns (lg_small, res_hi, res_lo).

  On [0.5, 4], lgamma(x) is approximated as (x - 1)(x - 2) P(d). The factors
  u = x - 1, v = x - 2, and P(d) are split into 12-bit high and low parts so
  that the unsummed pair (res_hi, res_lo) retains extra precision for Euler
  reflection.
  """
  in_15_4 = x_small > 1.5
  u = x_small - 1.0
  v = u - 1.0
  d = u - lax.select(in_15_4, lax.full_like(u, 1.75), lax.full_like(u, 0.0))
  coeffs_hi = [
      lax.select(in_15_4, lax.full_like(x_small, c2), lax.full_like(x_small, c1))
      for c1, c2 in zip(_POS_HI_05_15, _POS_HI_15_4)
  ]
  p = lax.polynomial(d, coeffs_hi, unroll=len(coeffs_hi))
  u_hi, u_lo = split_half(u)
  v_hi, v_lo = split_half(v)
  uv_hi = u_hi * v_hi
  uv_lo = u_lo * v + u_hi * v_lo
  p_hi, p_lo = split_half(p)
  res_hi = uv_hi * p_hi
  res_lo = uv_lo * p + uv_hi * p_lo
  return res_hi + res_lo, res_hi, res_lo


def _lgamma_pos_small_with_f64(u):
  """Evaluates lgamma(1 + u) on u in [-0.5, 3] in float64."""
  in_15_4 = u > 0.5
  u_d = lax.convert_element_type(u, np.float64)
  d_d = u_d - lax.select(
      in_15_4, lax.full_like(u_d, 1.75), lax.full_like(u_d, 0.0)
  )
  coeffs_f64 = [
      lax.select(in_15_4, lax.full_like(d_d, c2), lax.full_like(d_d, c1))
      for c1, c2 in zip(_POS_05_15_F64, _POS_15_4_F64)
  ]
  p_d = lax.polynomial(d_d, coeffs_f64, unroll=len(coeffs_f64))
  return (u_d * (u_d - 1.0)) * p_d


def _lgamma_pos_large(x_gt4):
  """Evaluates the asymptotic Stirling series for x >= 4.

  Returns (p_hi, s_pos, psi_large), where p_hi + s_pos approximates lgamma(x)
  with an exact 12-bit leading product p_hi = b_hi * (h_x - 1), and
  psi_large = ln(x) - 1/(2x) approximates the digamma function psi(x).
  """
  s, log_lo = _log_f32(x_gt4)
  h_x, s_lo = split_half(s)
  l_x = s_lo + log_lo

  b = x_gt4 - 0.5
  b_hi, b_lo = split_half(b)
  hx_m1 = h_x - 1.0
  p_hi = b_hi * hx_m1
  p_lo = b_lo * hx_m1 + b * l_x
  inv_x = lax.reciprocal(x_gt4)
  sp = lax.polynomial(inv_x * inv_x, _STIRLING_HI, unroll=len(_STIRLING_HI))
  s_pos = p_lo + _HALF_LN_2PI_MINUS_HALF + inv_x * sp
  return p_hi, s_pos, s - 0.5 * inv_x


def _lgamma_pos_large_with_f64(x_f32, x_d):
  """Evaluates the asymptotic Stirling series for x >= 4 in float64."""
  inv_x_0 = lax.convert_element_type(lax.reciprocal(x_f32), np.float64)
  inv_x_d = inv_x_0 * (2.0 - x_d * inv_x_0)
  du = (x_d - lax.convert_element_type(x_f32, np.float64)) * inv_x_d
  # Since du = (x_d - x_f32) / x_d = 1 - x_f32 / x_d,
  # ln(x_d) = ln(x_f32) - ln(1 - du) = ln(x_f32) + du * (1 + 0.5 * du + ...).
  ln_x = _log_f32_with_f64(x_f32) + du * (1.0 + 0.5 * du)
  sp_d = lax.polynomial(
      inv_x_d * inv_x_d, _STIRLING_F32_F64, unroll=len(_STIRLING_F32_F64)
  )
  main_d = (x_d - 0.5) * (ln_x - 1.0)
  return main_d + _HALF_LN_2PI_MINUS_HALF_F64 + inv_x_d * sp_d


def _lgamma_neg_reflection(delta, log_delta_hi, lg_pos_hi, lg_pos_lo, dy_corr):
  """Evaluates Euler's reflection formula -(ln|delta| + lgamma(1 - x) - ln(pi delta / sin(pi delta)))."""
  d2 = delta * delta
  sinc_term = d2 * lax.polynomial(d2, _SINC_HI, unroll=len(_SINC_HI))
  # TODO(phawkins): Using 2Sum on (log_delta_hi, lg_pos_hi) avoids cancellation
  # loss when log_delta_hi + lg_pos_hi cancels outside root windows.
  return -((log_delta_hi + lg_pos_hi) + (lg_pos_lo + (sinc_term + dy_corr)))


def _lgamma_neg_root(x_neg_safe, n):
  """Evaluates lgamma(x) near negative roots via rational root-window polynomials.

  The offset delta_root = x - n_root is mapped to the rational variable
  w = (delta_root - r) / (delta_root + r). Returns (in_root_win, w * P(w)).
  """
  # On (-3, -2), the two zeros of lgamma(x) lie at -2.748 and -2.457 (midpoint
  # -2.602), whereas round(x) switches from -2 to -3 at -2.5 (inside the root
  # window of the -2.457 zero). Override n_root to -2.0 on (-2.6, -2.0).
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

  idx_exp = lax.broadcast_in_dim(idx, (*idx.shape, 1), tuple(range(idx.ndim)))
  dnums = slicing.GatherDimensionNumbers(
      offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
  )
  cols = [
      slicing.gather(row, idx_exp, dnums, slice_sizes=(1,))
      for row in _NEG_TABLE
  ]
  r_hi, r_lo, w_bound = cols[0], cols[1], cols[2]
  # TODO(phawkins): To reduce worst-case root-window error near w = w_bound
  # (e.g. x ~ -6.00337), compensate (delta_root - r_hi) and the final product
  # w * (c_0 + w * P_tail(w)) using 2Sum and _split12.
  d0 = delta_root - r_hi
  d = delta_root + r_hi
  rcp = lax.reciprocal(d)
  w = (d0 - r_lo) * rcp
  w = w + ((d0 - w * d) - r_lo * (1.0 + w)) * rcp
  in_root_win = lax.abs(w) <= w_bound
  w_safe = lax.select(in_root_win, w, lax.full_like(x_neg_safe, 0.0))
  return in_root_win, w_safe * lax.polynomial(
      w_safe, cols[3:], unroll=len(cols) - 3
  )


def _lgamma_impl_f32(x, *, use_f64=False):
  """Computes float32 lgamma(x) across all normal and special inputs."""
  x = lax.convert_element_type(x, np.float32)
  # Flush subnormals to zero (< finfo(float32).tiny) so non-FTZ backends
  # behave consistently with FTZ backends without subnormal bit-normalization.
  is_zero = lax.abs(x) < float(np.finfo(np.float32).tiny)
  is_pos = x >= 0.5
  # Substitute 1.0 for 0.0 so inactive branches avoid 0/0 or log(0) NaNs.
  x_safe = lax.select(is_zero, lax.full_like(x, 1.0), x)

  # For x <= -2^23 (_ALL_INTEGERS_BELOW), every float32 is an exact integer
  # pole. Clamp inputs with x >= 0.5 and large negative integers to -0.5 so
  # round(x_neg_safe) cannot overflow and delta = x - n stays away from zero.
  x_neg_safe = lax.select(
      (~is_pos) & (x_safe > _ALL_INTEGERS_BELOW),
      x_safe,
      lax.full_like(x, -0.5),
  )
  # Decompose x < 0.5 as x = n + delta with n = round(x) and delta in [-0.5, 0.5]
  # (exact by Sterbenz's lemma).
  n = lax.round(x_neg_safe, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = x_neg_safe - n
  abs_delta = lax.abs(delta)

  if use_f64:
    # Float64 minimax polynomials provide ~49-52 bits of precision, which is
    # enough to absorb the worst-case 23.3 bits of cancellation near negative
    # roots in float32 without 12-bit splits or the _NEG_TABLE lookup.
    x_pos_bounded = lax.min(x_safe, lax.full_like(x, _MAX_FINITE_ARG))
    u = lax.select(is_pos, x_pos_bounded - 1.0, -x_neg_safe)
    x_bounded = lax.select(is_pos, x_pos_bounded, 1.0 - x_neg_safe)
    x_bounded_d = lax.select(
        is_pos,
        lax.convert_element_type(x_pos_bounded, np.float64),
        1.0 - lax.convert_element_type(x_neg_safe, np.float64),
    )
    lg_small_d = _lgamma_pos_small_with_f64(u)
    lg_large_d = _lgamma_pos_large_with_f64(x_bounded, x_bounded_d)
    lg_pos_d = lax.select(u <= 3.0, lg_small_d, lg_large_d)

    log_delta_d = _log_f32_with_f64(abs_delta)
    d_d = lax.convert_element_type(delta, np.float64)
    d2_d = d_d * d_d
    sinc_d = d2_d * lax.polynomial(
        d2_d, _SINC_F32_F64, unroll=len(_SINC_F32_F64)
    )
    refl = lax.convert_element_type(
        -(log_delta_d + lg_pos_d + sinc_d), np.float32
    )
    res = lax.select(
        is_pos, lax.convert_element_type(lg_pos_d, np.float32), refl
    )
    # At x = -1.5639554e-09 (0xb0d6f2ca), the exact value sits 0.19 float64
    # ULPs above a float32 midpoint, causing float64 -> float32 double rounding.
    res = lax.select(
        x == -1.5639554096225083e-09, lax.full_like(x, 20.27604866027832), res
    )
  else:
    # Form the reflected argument y = 1 - x = (1 - n) - delta as a Fast2Sum pair
    # (y, y_lo) because (1 - n) can be much larger than |delta|, losing low bits
    # of delta in y.
    y = (1.0 - n) - delta
    y_lo = ((1.0 - n) - y) - delta

    # Both x >= 0.5 (at x) and x < 0.5 (via reflection at y = 1 - x > 0.5)
    # evaluate lgamma at an argument >= 0.5. Clamp to _MAX_FINITE_ARG to prevent
    # inf - inf NaNs in Stirling's series.
    arg_pos = lax.select(is_pos, x_safe, y)
    x_bounded = lax.min(arg_pos, lax.full_like(x, _MAX_FINITE_ARG))

    # Compute ln|delta| for Euler's reflection formula on x < 0.5.
    log_hi, log_lo = _log_f32(abs_delta)

    # Evaluate lgamma on [0.5, 4] and [4, inf), keeping both the summed result
    # lg_pos (for x >= 0.5) and the unsummed (lg_pos_hi, lg_pos_lo) pair so
    # Euler reflection can cancel leading terms before adding low-order
    # corrections.
    lg_small, lg_small_hi, lg_small_lo = _lgamma_pos_small(x_bounded)
    p_hi, s_pos, psi = _lgamma_pos_large(x_bounded)
    in_le_four = x_bounded <= 4.0
    lg_pos = lax.select(in_le_four, lg_small, p_hi + s_pos)
    lg_pos_hi = lax.select(in_le_four, lg_small_hi, p_hi)
    lg_pos_lo = lax.select(in_le_four, lg_small_lo, s_pos)

    # Combine Euler reflection (with first-order Taylor correction psi(y) * y_lo
    # for rounding in y) and rational root-window polynomials near negative
    # zeros on (-10, -2) where reflection suffers catastrophic cancellation.
    refl = _lgamma_neg_reflection(
        delta, log_hi, lg_pos_hi, lg_pos_lo + log_lo, psi * y_lo
    )
    in_root_win, rw = _lgamma_neg_root(x_neg_safe, n)
    neg_res = lax.select(in_root_win, rw, refl)
    res = lax.select(is_pos, lg_pos, neg_res)
  # Canonicalize -0.0 at x = 1.0, 2.0 to +0.0.
  res = lax.select(res == 0.0, lax.full_like(x, 0.0), res)
  # Return +inf at poles (x = +-0 and negative integers, which also matches
  # x = -inf) and on positive overflow (x > _MAX_FINITE_ARG); preserve NaNs.
  is_neg_int = (~is_pos) & (x_safe == lax.floor(x_safe))
  is_pole_or_inf = is_zero | is_neg_int | (x > _MAX_FINITE_ARG)
  res = lax.select(is_pole_or_inf, lax.full_like(x, np.inf), res)
  return lax.select(x != x, lax.full_like(x, np.nan), res)


# ---------------------------------------------------------------------------
# Float64 implementation
# ---------------------------------------------------------------------------

# Largest float64 whose lgamma is finite in float64 (lgamma(2.5599833278516383e305)
# = 1.7976931348623157e308 <= DBL_MAX; the next float64 up overflows to +inf).
_MAX_FINITE_ARG_F64 = 2.5599833278516383e305
_ALL_INTEGERS_BELOW_F64 = -4503599627370496.0
_SQRT2_OVER_2_BITS_F64 = 0x3FE6A09E667F3BCD
# 42-bit + 53-bit split of ln(2) so k * _LN2_HI_F64 is exact for |k| <= 2048.
_LN2_HI_F64 = 0.6931471805598903
_LN2_LO_F64 = 5.497923018708371e-14
# High/low float64 pair for (ln(2 * pi) - 1) / 2.
_C_STIRLING_HI_F64 = 0.4189385332046727
_C_STIRLING_LO_F64 = 1.6728209650585413e-17

# Degree-20 Sollya fpminimax polynomial for (ln(1 + f) - (f - f^2/2 + f^3/3)) / f^4
# on f = m - 1 in [sqrt(2)/2 - 1, sqrt(2) - 1], stored in ascending powers of f.
# Generated with Sollya:
#   prec = 300!;
#   p = fpminimax(log1p(x),
#       [|4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24|],
#       [|D...|], [sqrt(2)/2 - 1; sqrt(2) - 1], relative, floating,
#       x - 0.5 * x^2 + (1/3) * x^3);
#   for i from 4 to 24 do print(double(coeff(p, i)));
_LOG_Q_F64 = [
    -0.25000000000000044, 0.20000000000000684, -0.1666666666665117,
    0.14285714285524623, -0.1250000000185486, 0.11111111132038475,
    -0.09999999896240677, 0.09090907860216561, -0.08333336179500314,
    0.07692350617660347, -0.07142830929109048, 0.06665745309174184,
    -0.062494384445847165, 0.05894350949521322, -0.0557491085999632,
    0.05176870915457102, -0.04765086647690675, 0.04986665866679927,
    -0.05835232191926365, 0.05131732430398771, -0.020844657565116175,
]

# Degree-25 minimax polynomial for lgamma(x) / ((x - 1)(x - 2)) in ascending
# powers of d = x - 1.5 on [1.0, 2.5].
_POS_1_2_F64 = [
    0.4831289505409809, -0.14595989591430608, 0.06291140107456523,
    -0.03130848750101384, 0.016797098631156464, -0.009425622448251035,
    0.005446457845678879, -0.0032120740850929658, 0.001922916122193837,
    -0.0011644143182151867, 0.0007115223504805953, -0.00043799690726877314,
    0.00027127807787894024, -0.00016888251593045404, 0.00010560593801995817,
    -6.63531280568692e-05, 4.1889018842861785e-05, -2.6381718111451702e-05,
    1.63964215276399e-05, -1.0321522858455005e-05, 7.078038835494905e-06,
    -5.126151988040559e-06, 3.239077383360151e-06, -1.4716755012110035e-06,
    4.0571328898284337e-07, -5.0333679908980265e-08,
]

# Minimax coefficients in ascending powers of 1/z^2 for the Stirling remainder
# S(1/z^2) on z >= 8.
_STIRLING_F64 = [
    0.08333333333333333, -0.002777777777777778, 0.0007936507936507791,
    -0.0005952380952234643, 0.0008417508341741269, -0.0019175246162359454,
    0.006409817242328721, -0.029496160785839165, 0.17520548011608733,
    -1.1579334089679656, 5.626868322268739,
]

# Degree-15 Sollya fpminimax polynomial for ln(sinc(pi * delta)) / delta^2 on
# delta in [-0.5, 0.5], in ascending powers of d2 = delta^2.
# Generated with Sollya:
#   prec = 300!;
#   f = log(sin(pi * sqrt(x)) / (pi * sqrt(x))) / x;
#   c0 = double(-pi^2 / 6);
#   c0_lo = double(-pi^2 / 6 - c0);
#   p = fpminimax(f + pi^2 / 6,
#       [|1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15|],
#       [|D...|], [1b-40; 0.25], absolute, floating);
#   print(c0_lo);
#   print(c0);
#   for i from 1 to 15 do print(double(coeff(p, i)));
_SINC_C0_LO_F64 = -3.040672350398476e-17
_SINC_F64 = [
    -1.6449340668482264, -0.5411616168555692, -0.33911435399478207,
    -0.2510193390542899, -0.20019891468257156, -0.16670769570769814,
    -0.1428654870357636, -0.1250096373515999, -0.11100708179151114,
    -0.1010207396483517, -0.0836486341691027, -0.12077319335821443,
    0.0604927293411325, -0.4163820190384763, 0.47863524753569425,
    -0.5052021896795194,
]

# Negative-argument root lookup table of shape (25, 30) for zeros x_root = -k + r
# of lgamma(x) near negative integers -k (k in {2, ..., 16}). For k >= 17,
# |r| ~ 1/k! < ulp(-k), so no non-integer float64 lies near a root.
# Rows contain:
#   rows 0..2 (`r0, r1, r2`): 3-float non-overlapping split of r = x_root - (-k).
#   row 3 (`w_bound`): half-width threshold on w = (delta - r) / (delta + r).
#   rows 4..24 (`c_0 .. c_20`): degree-20 minimax polynomial coefficients in w.
_NEG_TABLE_F64 = np.array([
    [-0.45702473822080064, 1.0, -0.14358088834998006, 0.2523173532725874, -0.03936183974053688, 0.04470571514140207, -0.008218168322593521, 0.008455359439952277, -0.0013852944531550972, 0.0013925199191243706, -0.00019833340732475162, 0.0001984921093623021, -2.48002706819597e-05, 2.4802904179335845e-05, -2.7557148226503463e-06, 2.7557490225318057e-06, -2.755730136466002e-07, 2.7557337083353163e-07, -2.5052106852407546e-08, 2.5052109918476226e-08, -2.087675687777539e-09, 2.087675709796081e-09, -1.6059043830108875e-10, 1.6059043843534355e-10, -1.147074559737784e-11, 1.147074559808161e-11, -7.647163731803788e-13, 7.647163731835846e-13, -4.779477332386745e-14, 4.779477332388026e-14],
    [1.843554041574456e-17, 0.0, 3.862806401714152e-18, -2.04688991691325e-17, 3.2731379225891636e-18, 8.223977942582053e-19, -5.582665915956405e-19, 8.231751957350036e-19, -2.6295731811446077e-20, -7.090441230646531e-20, 1.1820573401560963e-20, 6.263909978492175e-21, 1.0083974251164085e-21, 3.15397713595963e-22, -5.624459871432079e-23, -6.500300408654724e-23, -2.4472526783403563e-23, 1.4502658434955713e-23, 1.2924868620853802e-24, -6.807394864189793e-25, -4.849684877236172e-26, -8.658839215884982e-26, 7.046449983196074e-27, 6.465141936497467e-27, 4.194056116105366e-28, -7.830340218001236e-28, 4.886154187641447e-29, -3.8035184915187745e-29, -1.811833433216651e-30, -9.319923331229129e-31],
    [-1.3622663121726005e-33, 0.0, 3.0899838057763426e-34, 2.4127314614779214e-34, -1.2946821126563304e-34, 4.579537115999924e-35, 1.4462448897470062e-35, -1.3866184681212228e-36, 3.565474316668236e-37, 5.348580824008999e-36, 6.395218422774651e-37, -1.8863056797360293e-37, 7.203753792282653e-38, 4.313766104588735e-39, 3.482629221993885e-39, 5.853754782999141e-39, 1.3906455698886483e-39, -5.362578714050634e-40, 8.03536423808754e-41, 1.9842318548433462e-41, 4.75154991965243e-43, -2.5911350779887772e-42, 3.753098085579908e-43, -4.0581912582368964e-43, 4.450809771991351e-44, -1.2604965694501327e-44, -2.020544021944698e-45, 5.92376186432879e-46, -1.7117855162619145e-46, 6.619263486738802e-47],
    [0.06, -1.0, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
    [-1.3853365381572815, 0.0, -2.2346598245031606, -0.9660475441408101, -2.1090452304243645, -1.853057332921452, -2.0276223947689727, -1.9707038423622454, -2.005176677961308, -1.9947720577043362, -2.000799289737154, -1.9991995733979362, -2.0001061730738354, -1.9998938078459143, -2.0000124103276042, -1.9999875894216372, -2.0000012961586116, -1.9999987038387381, -2.0000001223876387, -1.9999998776123382, -2.0000000105469167, -1.999999989453083, -2.0000000008360077, -1.9999999991639925, -2.0000000000613536, -1.9999999999386464, -2.000000000004192, -1.9999999999958078, -2.000000000000268, -1.999999999999732],
    [2.673724764253963, 0.0, -0.10455894447703194, 1.4723336980240769, -0.0994995491507347, 0.159251498482983, -0.027202407519091913, 0.029740663409124374, -0.005164640325456555, 0.005240105578631542, -0.0007990413896202449, 0.000800675346647562, -0.00010616917149802439, 0.0001061960572510789, -1.241027923521866e-05, 1.2410626733054182e-05, -1.2961581262117948e-06, 1.2961617471025452e-06, -1.2238763478124098e-07, 1.2238766584163939e-07, -1.054691669750544e-08, 1.0546916920663231e-08, -8.360075136127152e-10, 8.36007514975095e-10, -6.135350038852013e-11, 6.13535003956748e-11, -4.192195542547764e-12, 4.192195542580427e-12, -2.6798656807567477e-13, 2.6798656807580563e-13],
    [5.655016965104241, 0.0, -0.6329766231481236, 1.3283223045398886, -0.7565748626623968, -0.49504265824017485, -0.6934489837338348, -0.6364814354630196, -0.6718192692091269, -0.6614143978247053, -0.6674654597085423, -0.6658657425755931, -0.6667728319358269, -0.6665604667062505, -0.6666790768975328, -0.6666542559915634, -0.6666679628243076, -0.6666653705044343, -0.6666667890542974, -0.6666665442789969, -0.6666666772135833, -0.6666666561197497, -0.6666666675026742, -0.6666666658306591, -0.6666666667280202, -0.6666666666053132, -0.6666666666708588, -0.6666666666624744, -0.6666666666669346, -0.6666666666663987],
    [13.646672250576216, 0.0, 0.18452218707533613, 2.6632833807372487, -0.08025011512795627, 0.18409537828793723, -0.02636208392847831, 0.03062990526689935, -0.005140564580448945, 0.005264432118193118, -0.0007985446939068473, 0.0008011728353554061, -0.00010616136682231183, 0.00010620386358104198, -1.2410182497171244e-05, 1.2410723473499054e-05, -1.2961571557829664e-06, 1.296162717533886e-06, -1.2238762674041126e-07, 1.223876738824711e-07, -1.0546916641545272e-08, 1.0546916976623401e-08, -8.360075132809795e-10, 8.360075153068307e-10, -6.135350038682492e-11, 6.135350039737002e-11, -4.192195542540219e-12, 4.192195542587972e-12, -2.679865680756453e-13, 2.679865680758351e-13],
    [29.035921057847425, 0.0, -0.046968881638725014, 3.164772642089249, -0.47050400560851524, -0.20329868423406788, -0.4259416685981684, -0.3689252701318118, -0.4051285264075521, -0.39472340456013233, -0.4007982963457003, -0.3991985784204936, -0.400106157464484, -0.3998937922332544, -0.40001241013412814, -0.3999875892281563, -0.40000129615667057, -0.39999870383679725, -0.4000001223876227, -0.3999998776123221, -0.40000001054691664, -0.399999989453083, -0.4000000008360075, -0.3999999991639925, -0.4000000000613535, -0.3999999999386465, -0.40000000000419217, -0.3999999999958078, -0.400000000000268, -0.399999999999732],
    [64.98702199160488, 0.0, 0.5452512522098585, 4.829798317892424, -0.06064823805914922, 0.2094782353956854, -0.02552103154455747, 0.03151974930849315, -0.005116487991898609, 0.005288758839526189, -0.0007980479972427041, 0.0008016703235399936, -0.00010615356214542476, 0.00010621166990993501, -1.2410085759122588e-05, 1.2410820213942699e-05, -1.2961561853541372e-06, 1.296163687965226e-06, -1.2238761869958152e-07, 1.223876819233028e-07, -1.0546916585585104e-08, 1.054691703258357e-08, -8.360075129492439e-10, 8.360075156385664e-10, -6.13535003851297e-11, 6.135350039906524e-11, -4.192195542532674e-12, 4.192195542595517e-12, -2.6798656807561575e-13, 2.6798656807586465e-13],
    [147.10470752566349, 0.0, 0.482784981277822, 6.371257047788926, -0.3363750555357443, -0.06325084244821723, -0.310814418916723, -0.2537492777858701, -0.2908187350159023, -0.280413363364575, -0.2865120853628063, -0.2849123666468161, -0.28582043537409224, -0.2856080701412117, -0.2857266957516758, -0.28570187484570153, -0.28571558186998586, -0.28571298955011254, -0.2857144081019004, -0.28571416332659977, -0.28571429626120226, -0.28571427516736864, -0.28571428655029324, -0.2857142848782782, -0.28571428577563923, -0.2857142856529322, -0.2857142857184779, -0.2857142857100935, -0.2857142857145537, -0.28571428571401775],
    [342.3498359519069, 0.0, 1.0317968727617497, 9.353020268805944, -0.04051905821601453, 0.23569532703710577, -0.02467893396636483, 0.0324105501161364, -0.005092410304833189, 0.005313086002964257, -0.0007975512995206722, 0.0008021678113088109, -0.00010614575746733698, 0.00010621947623778417, -1.2409989021072685e-05, 1.2410916954385124e-05, -1.2961552149253068e-06, 1.2961646583965648e-06, -1.2238761065875178e-07, 1.223876899641345e-07, -1.0546916529624936e-08, 1.0546917088543739e-08, -8.360075126175083e-10, 8.360075159703021e-10, -6.135350038343448e-11, 6.135350040076045e-11, -4.1921955425251296e-12, 4.1921955426030615e-12, -2.6798656807558627e-13, 2.6798656807589414e-13],
    [810.4476269133567, 0.0, 1.124180164833617, 13.17345819447863, -0.2524221937229917, 0.026991330169771878, -0.24647961640616886, -0.1893658019068389, -0.22730259319189677, -0.21689697239039135, -0.2230195251724517, -0.22141980566715133, -0.22232836407735004, -0.2221159988428209, -0.22223463216287426, -0.22220981125689762, -0.22222351837695192, -0.2222209260570786, -0.22222234460982887, -0.22222209983452823, -0.22222223276913872, -0.2222222116753051, -0.22222222305822972, -0.2222222213862147, -0.22222222228357572, -0.22222222216086873, -0.2222222222264144, -0.22222222221803004, -0.2222222222249022, -0.22222222222195423],
    [1948.880680942719, 0.0, 1.7264860516868281, 19.533472364182522, -0.019679697116765905, 0.2630592670989325, -0.02383547414911425, 0.03330266303341061, -0.005068331264265538, 0.005337413868855946, -0.0007970546006336089, 0.0008026652987693448, -0.00010613795278802234, 0.00010622728256461577, -1.240989228302154e-05, 1.2411013694826337e-05, -1.296154244496476e-06, 1.2961656288279034e-06, -1.2238760261792209e-07, 1.2238769800496625e-07, -1.0546916473664773e-08, 1.0546917144503914e-08, -8.36007512285773e-10, 8.360075163020381e-10, -6.135350038173931e-11, 6.13535004024557e-11, -4.1921955425175864e-12, 4.192195542610608e-12, -2.679865680755569e-13, 2.6798656807592377e-13],
    [4740.876941587141, 0.0, 2.008175676977363, 28.76274622599986, -0.19075182861387507, 0.09545713208858458, -0.20523131591357024, -0.14806885896918995, -0.18687447297491153, -0.17646860367159567, -0.18261498806890158, -0.1810152677757643, -0.18192431586862978, -0.1817119506324542, -0.18183059166209586, -0.18180577075611684, -0.18181947797194117, -0.18181688565206783, -0.1818183042057805, -0.18181805943047985, -0.18181819236509833, -0.1818181712712647, -0.1818181826541894, -0.18181818098217437, -0.18181818187953538, -0.18181818175682837, -0.18181818182237408, -0.18181818181398968, -0.18181818181844986, -0.1818181818179139],
    [11642.86968078643, 0.0, 2.7597576098155248, 43.490377156725195, 0.0020636419116818083, 0.29190645355490197, -0.02299033418912612, 0.03419644442041672, -0.005044250615187363, 0.005361742697568448, -0.0007965579004741387, 0.000803162786028846, -0.00010613014810742363, 0.00010623508889042488, -1.2409795544965517e-05, 1.2411110435262711e-05, -1.2961532740672657e-06, 1.2961665992588622e-06, -1.2238759457705666e-07, 1.2238770604576224e-07, -1.054691641770153e-08, 1.0546917200461006e-08, -8.360075119537935e-10, 8.360075166335299e-10, -6.135350038002619e-11, 6.135350040413301e-11, -4.1921955425088194e-12, 4.1921955426169305e-12, -2.679865680754492e-13, 2.679865680758751e-13],
    [28811.41764792298, 0.0, 3.311079776013495, 65.96748029985126, -0.14050754570086124, 0.1531534453976564, -0.17641318836889802, -0.11920208135256376, -0.15887836345388062, -0.1484722462917643, -0.15464246339600332, -0.15304274231650256, -0.1539522800918861, -0.1537399148540665, -0.1538585635932954, -0.15383374268731398, -0.15384744999890831, -0.153844857679035, -0.15384627623371003, -0.1538460314584094, -0.15384616439303586, -0.15384614329920224, -0.15384615468212698, -0.15384615301011195, -0.15384615390747297, -0.15384615378476596, -0.15384615385031167, -0.15384615384192726, -0.15384615384638745, -0.15384615384585149],
    [71750.78894918696, 0.0, 4.342813222201444, 101.52723250549434, 0.02491877478161391, 0.32260406404535463, -0.022143195110220452, 0.035092251914588056, -0.005020168103153817, 0.005386072750155458, -0.0007960611990301447, 0.0008036602732913377, -0.00010612234343821778, 0.00010624289522796473, -1.2409698808390581e-05, 1.2411207177180517e-05, -1.296152303792886e-06, 1.2961675698446547e-06, -1.2238758655081105e-07, 1.223877141011781e-07, -1.054691636299817e-08, 1.0546917257677984e-08, -8.360075117216795e-10, 8.360075170648872e-10, -6.135350038564209e-11, 6.135350041313933e-11, -4.192195543000832e-12, 4.192195543124032e-12, -2.6798656810735396e-13, 2.679865681078389e-13],
    [179642.50343034536, 0.0, 5.3079592907229936, 157.04841702665868, -0.09650047293718188, 0.20543848906293946, -0.15505210921818782, -0.0977923057252266, -0.13834145941458478, -0.12793509502963954, -0.13412914619381705, -0.13252942432916848, -0.1334394517871757, -0.13322708654768017, -0.13334574299651444, -0.13332092209052665, -0.1333346294978932, -0.13333203717801945, -0.1333334557336571, -0.13333321095835643, -0.1333333438929909, -0.13333332279915727, -0.13333333418208207, -0.13333333251006704, -0.13333333340742806, -0.13333333327197985, -0.13333333335026676, -0.13333333334188238, -0.13333333334634256, -0.13333333334580658],
    [451808.7224217881, 0.0, 6.821895712621164, 245.02190082688918, 0.04911089169648033, 0.3555577714629333, -0.021293736069290744, 0.03599044340355407, -0.004996083315584032, 0.0054104041099184465, -0.0007955644706782671, 0.0008041577347428386, -0.00010611453536937644, 0.00010625069815985649, -1.2409601674290051e-05, 1.2411303521493267e-05, -1.2961512919969207e-06, 1.2961684989080596e-06, -1.2238757460392744e-07, 1.223877182359493e-07, -1.0546915970428405e-08, 1.0546916977028511e-08, -8.360074847083861e-10, 8.360074907150648e-10, -6.135349842582217e-11, 6.135349845670984e-11, -4.192195409197471e-12, 4.19219540933576e-12, -2.6798655955441195e-13, 2.679865595549559e-13],
    [1140923.6661636205, 0.0, 8.446193644015102, 384.0373504048825, -0.055862709366734584, 0.25537303231293124, -0.13851509212732388, -0.0812065090220722, -0.12263109576437564, -0.11222448520279435, -0.11844237178812286, -0.11684264920342026, -0.11775316630782759, -0.11754080107510424, -0.11765946522882238, -0.11763464432381539, -0.11764835182640895, -0.11764575950663789, -0.11764717806318174, -0.11764693328789078, -0.11764706622252795, -0.11764704512869517, -0.11764705651161957, -0.11764705483960461, -0.11764705573696559, -0.11764705561425859, -0.11764705567980428, -0.11764705567141989, -0.11764705567588007, -0.1176470556753441],
    [2896925.8032902596, 0.0, 10.76950549963546, 605.281149641511, 0.0748871845856548, 0.39122337487674935, -0.020441735844097857, 0.036891603302337164, -0.004972023612300115, 0.005434768088256848, -0.0007950721885372976, 0.0008046597147495142, -0.00010610732188005472, 0.00010625909678760836, -1.240957409209317e-05, 1.2411469431620209e-05, -1.296157544926683e-06, 1.2961766928374902e-06, -1.2238824862218985e-07, 1.223884083370349e-07, -1.0546974691853985e-08, 1.0546975810382552e-08, -8.360121434007222e-10, 8.360121500709207e-10, -6.135384034382469e-11, 6.135384037810304e-11, -4.1922187720661125e-12, 4.192218772219492e-12, -2.679880530288509e-13, 2.6798805302945386e-13],
    [7366757.65228341, 0.0, 13.465058950148588, 957.453159643403, -0.016811116999590887, 0.3049643458090739, -0.12527827890102464, -0.06792041698856355, -0.11022365465823368, -0.09981673077994846, -0.1060584957716675, -0.10445876205209088, -0.1053697747445793, -0.10515740813585692, -0.10527608076731819, -0.10525125970167565, -0.10526496738875242, -0.10526237505220466, -0.10526379361889139, -0.1052635488420163, -0.10526368177752178, -0.10526366068355249, -0.10526367206655061, -0.10526367039452483, -0.10526367129189163, -0.10526367116918382, -0.10526367123472995, -0.1052636712263455, -0.10526367123080571, -0.10526367123026974],
    [18161650.23186171, 0.0, 17.108808892164763, 1513.7613872853585, 0.102416032015214, 0.4297862909239346, -0.019575769801149744, 0.03777143077439166, -0.0049449405817236156, 0.005455737627519443, -0.0007940906015108819, 0.0008046646240201446, -0.00010603469864573113, 0.00010620196294150692, -1.2401895117768077e-05, 1.2403982419738896e-05, -1.2953646067492066e-06, 1.2953856802296025e-06, -1.223134597442909e-07, 1.2231363541345543e-07, -1.0540530293817517e-08, 1.0540531523375912e-08, -8.355013285018235e-10, 8.355013358301902e-10, -6.13163523275337e-11, 6.131635236517526e-11, -4.189657270485927e-12, 4.1896572706542756e-12, -2.6782430875482035e-13, 2.6782430875548184e-13],
    [46247060.26014505, 0.0, 21.55491544484296, 2409.4605517627865, 0.021813061389277154, 0.355379040968014, -0.11433134763964477, -0.05696297483027051, -0.10011769701441664, -0.08971750642408946, -0.09597887738976531, -0.09438021555715892, -0.09529110607874244, -0.09507888186609423, -0.09519748261818653, -0.09517267819575512, -0.09518637678702269, -0.09518378618870854, -0.09518520380516736, -0.09518495919242198, -0.09518509203879848, -0.09518507095897327, -0.09518508233433884, -0.09518508066343419, -0.09518508156019927, -0.09518508143757376, -0.09518508150307593, -0.09518508149469711, -0.09518508149915432, -0.0951850814986187],
], dtype=np.float64)


# Degree-7 Sollya fpminimax polynomial for digamma(w) on [1.0, 5.0] (the only
# part of w < 8.0 where y_lo != 0 in Euler reflection).
# Generated with Sollya:
#   prec = 300!;
#   z = x + 16;
#   psi = log(z) - 1/(2*z) - (1/z^2)*(1/12 - (1/z^2)/120 + (1/z^4)/252
#         - (1/z^6)/240 + (1/z^8)/132 - 691*(1/z^10)/32760 + (1/z^12)/12);
#   for k from 0 to 15 do psi = psi - 1/(x + k);
#   p = fpminimax(psi, 7, [|D...|], [1; 5], absolute, floating);
#   for i from 0 to 7 do print(double(coeff(p, i)));
_PSI_1_5_F64 = [
    -4.248544747468387, 7.169558627936287, -5.511031392409067,
    2.6991260255235368, -0.8222847600014095, 0.1511139130995527,
    -0.015317621228517228, 0.0006569884397460105,
]


@jit
def _log_f64(x, k_adj):
  """Computes `ln(x) + k_adj * ln(2)` as a `(log_hi, log_lo)` pair."""
  # Write x = 2^exp_off * m with m in [sqrt(2)/2, sqrt(2)] so f = m - 1 stays
  # small (|f| <= 0.414). Adding k_adj into k folds 2^k_adj rescaling into
  # k * ln(2).
  bits = lax.bitcast_convert_type(x, np.int64)
  exp_off = lax.shift_right_arithmetic(
      bits - lax._const(bits, _SQRT2_OVER_2_BITS_F64), lax.full_like(bits, 52)
  )
  k = lax.convert_element_type(exp_off, np.float64) + k_adj
  mant_bits = bits - lax.shift_left(exp_off, lax.full_like(bits, 52))
  m = lax.bitcast_convert_type(mant_bits, np.float64)

  f = m - 1.0
  # Evaluate ln(1 + f) = f - f^2/2 + f^3 * (1/3 + f * P(f)). Split f into
  # 13-bit fh and 40-bit fl (c = (2^40 + 1) * f) so fh^2 (26 bits) and fh^3
  # (39 bits) have no rounding error in float64.
  cf = 1099511627777.0 * f
  fh = cf - (cf - f)
  fl = f - fh
  fh2 = fh * fh
  fh3 = fh2 * fh
  f_plus_fh = f + fh
  f2_l = fl * f_plus_fh
  f3_l = fl * (fh2 + f * f_plus_fh)
  nhf2 = -0.5 * fh2
  th, tl = fast_two_sum(f, nhf2)
  tl = tl - 0.5 * f2_l
  # 0.3333333333333333 + 1.850371707708594e-17 is the (hi, lo) split of 1/3.
  qh, ql = fast_two_sum(
      lax.full_like(f, 0.3333333333333333),
      f * lax.polynomial(f, _LOG_Q_F64, unroll=len(_LOG_Q_F64)),
  )
  ql = ql + 1.850371707708594e-17
  # Split qh into 13-bit q_hh and 40-bit q_hl so fh^3 * q_hh (39 + 13 = 52
  # bits) has no rounding error.
  cq = 1099511627777.0 * qh
  q_hh = cq - (cq - qh)
  q_hl = (qh - q_hh) + ql
  ph = fh3 * q_hh
  pl = fh3 * q_hl + f3_l * qh
  sh, sl = f2_add_ordered(th, tl, ph, pl)
  kh = k * _LN2_HI_F64
  kl = k * _LN2_LO_F64
  # Combine k * ln(2) + ln(1 + f). |kh| >= ln(2) > 0.5 * ln(2) >= |sh| when
  # k != 0, and kh == 0 when k == 0.
  return f2_add_ordered(kh, kl, sh, sl)


def _lgamma_stirling_f64(z):
  """Evaluates Stirling's series for z >= 8 as `(res_h, res_l, digamma_z)`."""
  # Stirling's expansion:
  #   lgamma(z) = (z - 0.5) * (ln(z) - 1) + (ln(2*pi) - 1)/2 + (1/z) * S(1/z^2).
  lh, ll = _log_f64(z, lax.full_like(z, 0.0))
  l_m1_h, l_m1_l = fast_two_sum(lh, lax.full_like(z, -1.0))
  l_m1_l = l_m1_l + ll
  b = z - 0.5
  # Use two_prod_big because z can reach 2.56e305, where Veltkamp splitting would
  # overflow float64.
  ph, pl = two_prod_big(b, l_m1_h)
  ph, pl = fast_two_sum(ph, pl + b * l_m1_l)
  sh, sl = f2_add_ordered(
      ph,
      pl,
      lax.full_like(z, _C_STIRLING_HI_F64),
      lax.full_like(z, _C_STIRLING_LO_F64),
  )
  inv_z = lax.reciprocal(z)
  inv_z2 = inv_z * inv_z
  sp = lax.polynomial(inv_z2, _STIRLING_F64, unroll=len(_STIRLING_F64))
  # For z >= 8, (1/z) * S(1/z^2) <= 1/96 while |lgamma| >= 0.12 everywhere this
  # path is used, so a single float64 product has <= 0.04 ULP rounding error.
  res_h, res_l = fast_two_sum(sh, sl + inv_z * sp)
  # Approximate digamma(z) = d/dz lgamma(z) for first-order rounding corrections.
  psi = (
      lh
      - 0.5 * inv_z
      - inv_z2 * (1.0 / 12.0 - inv_z2 * (1.0 / 120.0 - inv_z2 / 252.0))
  )
  return res_h, res_l, psi


def _lgamma_shifted_f64(w):
  """Evaluates Stirling lgamma(z), shift product prod, and digamma(w) for w >= 1.

  Stirling's series needs z >= 8 for full float64 accuracy. For w in [1, 8),
  pick integer m = ceil(8 - w) in {1, ..., 7} so z = w + m >= 8, and use
  lgamma(w) = lgamma(z) - ln(prod) with prod = w * (w + 1) * ... * (w + m - 1).
  Returns lgamma(z) and prod unlogged so the caller can merge logarithms.
  """
  need0 = w < 8.0
  zero = lax.full_like(w, 0.0)
  m = lax.select(need0, lax.ceil(8.0 - w), zero)
  # Split w in [1, 8] into 24-bit wh (c = (2^29 + 1) * w) and 29-bit wl.
  # Because wh has steps of >= 2^-23 and wh + k <= 8, each factor wh + k fits
  # in 26 bits, so pairs of high parts multiply with zero rounding error.
  # Factors with k >= m are replaced by 1.
  w_shift = lax.min(w, lax.full_like(w, 8.0))
  c = 536870913.0 * w_shift
  wh = c - (c - w_shift)
  wl = w_shift - wh
  one = lax.full_like(w, 1.0)
  fh = [lax.select(m > 0.0, wh, one)]
  fl = [lax.select(m > 0.0, wl, zero)]
  for k in range(1, 7):
    need = m > float(k)
    fh.append(lax.select(need, wh + float(k), one))
    fl.append(lax.select(need, wl, zero))
  p01_h, p01_l = f2_mul_exact_high(fh[0], fl[0], fh[1], fl[1])
  p23_h, p23_l = f2_mul_exact_high(fh[2], fl[2], fh[3], fl[3])
  p45_h, p45_l = f2_mul_exact_high(fh[4], fl[4], fh[5], fl[5])
  p45_hh, p45_hl = split_half(p45_h)
  p456_h = p45_h * fh[6]
  p456_l = ((p45_hh * fh[6] - p456_h) + p45_hl * fh[6]) + (
      p45_h * fl[6] + p45_l * fh[6]
  )
  p0123_h, p0123_l = two_prod(p01_h, p23_h)
  p0123_l = p0123_l + (p01_h * p23_l + p01_l * p23_h)
  prod_h, prod_l = f2_mul(p0123_h, p0123_l, p456_h, p456_l)
  # Adding m to w can round off up to 3 low bits of w (z_lo = w - (z - m));
  # add psi_z * z_lo to correct lgamma(z) for that rounding.
  z = w + m
  z_lo = w - (z - m)
  zh, zl, psi_z = _lgamma_stirling_f64(z)
  zl = zl + psi_z * z_lo
  psi_y = lax.select(
      need0,
      lax.polynomial(w_shift, _PSI_1_5_F64, unroll=len(_PSI_1_5_F64)),
      psi_z,
  )
  return zh, zl, prod_h, prod_l, psi_y


def _lgamma_neg_root_f64(x_neg_safe, n):
  """Evaluates float64 lgamma(x) near negative zeros on (-17, -2).

  lgamma(x) crosses zero twice in each interval (-k - 1, -k) for integer k >= 2,
  where Euler's reflection formula cancels. For k >= 17, the zero is within
  1 ULP of -k, so no non-integer float64 gets close to it.
  """
  # On (-3, -2), the two zeros are at -2.748 and -2.457, whereas round(x)
  # switches from -2 to -3 at -2.5 (inside the -2.457 root window). Force
  # n_root = -2 on (-2.6, -2.0).
  n_root = lax.select(
      (x_neg_safe > -2.6) & (x_neg_safe < -2.0),
      lax.full_like(x_neg_safe, -2.0),
      n,
  )
  delta_root = x_neg_safe - n_root
  k = lax.convert_element_type(-n_root, np.int64)
  in_k_range = (k >= 2) & (k <= 16)
  side = lax.select(delta_root > 0.0, lax.full_like(k, 1), lax.full_like(k, 0))
  # Slot = 2 * (k - 2) + side. Slot 1 (k = 2, side = 1) is an inactive sentinel
  # with w_bound = -1.0 because lgamma(x) > 0 on (-2, -1).
  idx = lax.select(in_k_range, 2 * (k - 2) + side, lax.full_like(k, 1))

  idx_exp = lax.broadcast_in_dim(idx, (*idx.shape, 1), tuple(range(idx.ndim)))
  dnums = slicing.GatherDimensionNumbers(
      offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
  )

  def _gather_row(i):
    return slicing.gather(
        lax._const(x_neg_safe, _NEG_TABLE_F64[i]),
        idx_exp,
        dnums,
        slice_sizes=(1,),
    )

  # Gather one 1D row at a time so GPU kernels do not keep a 25-element float64
  # slice live per thread.
  r0 = _gather_row(0)
  r1 = _gather_row(1)
  r2 = _gather_row(2)
  w_bound = _gather_row(3)
  # The root offset r = x_root - (-k) is split into three float64 parts
  # r0 + r1 + r2 (~159 bits). Inside an active window, delta_root is close
  # enough to r0 (within a factor of 2) that d0 = delta_root - r0 has no
  # rounding error, and |d0| >= ulp(r0) >= 2 * |r1| whenever d0 != 0.
  d0 = delta_root - r0
  num_h, num_l = fast_two_sum(d0, -r1)
  num_l = num_l - r2
  den_h, den_l = two_sum(delta_root, r0)
  den_l = den_l + r1
  # Compute w = (delta_root - r) / (delta_root + r) as (w_h, w_l). This maps
  # the zero delta_root = r to w = 0 and turns -ln|delta_root / r| into
  # -2 * atanh(w), which converges fast as an odd polynomial in w.
  rcp = lax.reciprocal(den_h)
  w_h = num_h * rcp
  wh_dh, wh_dl = two_prod(w_h, den_h)
  rem_num = ((num_h - wh_dh) - wh_dl) + num_l - w_h * den_l
  w_l = rem_num * rcp
  in_root_win = lax.abs(w_h) <= w_bound
  w_s = lax.select(in_root_win, w_h, lax.full_like(x_neg_safe, 0.0))
  w_sl = lax.select(in_root_win, w_l, lax.full_like(x_neg_safe, 0.0))
  p_tail = _gather_row(24)
  for i in range(23, 4, -1):
    p_tail = p_tail * w_s + _gather_row(i)
  c0 = _gather_row(4)
  pw_h, pw_l = fast_two_sum(c0, w_s * p_tail)
  rw_h, rw_l = two_prod(w_s, pw_h)
  return in_root_win, rw_h + (rw_l + (w_s * pw_l + w_sl * pw_h))


def _lgamma_impl_f64(x):
  """Computes float64 lgamma(x) across all normal and special inputs."""
  x = lax.convert_element_type(x, np.float64)
  # Flush subnormals to zero (< finfo(float64).tiny) so non-FTZ backends
  # behave consistently with FTZ backends without subnormal bit-normalization.
  is_zero = lax.abs(x) < float(np.finfo(np.float64).tiny)
  is_pos = x > 0.0
  # Replace 0.0 with 1.0 so inactive branches do not divide by zero or log(0).
  x_safe = lax.select(is_zero, lax.full_like(x, 1.0), x)

  # For x <= -2^52 (_ALL_INTEGERS_BELOW_F64), every float64 is an exact integer
  # pole. Clamp inputs with x > 0 and large negative integers to -0.5 so
  # round(x_neg_safe) cannot overflow and delta stays away from zero.
  x_neg_safe = lax.select(
      (~is_pos) & (x_safe > _ALL_INTEGERS_BELOW_F64),
      x_safe,
      lax.full_like(x, -0.5),
  )
  # Write x = n + delta with integer n = round(x) and delta in [-0.5, 0.5].
  # Form the reflected argument y = 1 - x = (1 - n) - delta and its rounding
  # remainder y_lo = ((1 - n) - y) - delta (needed when 1 - n >> |delta|).
  n = lax.round(x_neg_safe, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = x_neg_safe - n
  abs_delta = lax.abs(delta)
  y = (1.0 - n) - delta
  y_lo = ((1.0 - n) - y) - delta

  # Both positive x and negative reflection y = 1 - x evaluate lgamma on the
  # positive half-line. Clamp to _MAX_FINITE_ARG_F64 to prevent inf - inf NaNs.
  arg_pos = lax.select(is_pos, x_safe, y)
  x_bounded = lax.min(arg_pos, lax.full_like(x, _MAX_FINITE_ARG_F64))

  # Positive inputs on (0, 2.5):
  # - On [1, 2.5), evaluate (x - 1)(x - 2) * P(x - 1.5) so the zeros at x = 1
  #   and x = 2 are exact.
  # - On (0, 1), shift to x + 1 in (1, 2) via lgamma(x) = lgamma(x + 1) - ln(x),
  #   using u = x so (x + 1 - 1)(x + 1 - 2) = x * (x - 1).
  in_small = is_pos & (x_bounded < 2.5)
  x_small_in = lax.min(x_bounded, lax.full_like(x, 4.0))
  is_lt1 = x_small_in < 1.0
  u = lax.select(is_lt1, x_small_in, x_small_in - 1.0)
  v = u - 1.0
  d = u - 0.5
  p = lax.polynomial(d, _POS_1_2_F64, unroll=len(_POS_1_2_F64))
  uv_h, uv_l = two_prod(u, v)
  p_h, p_l = two_prod(uv_h, p)
  p_h, p_l = fast_two_sum(p_h, p_l + uv_l * p)

  # Positive inputs >= 2.5 and negative reflection y = 1 - x >= 1 both use
  # shifted Stirling on w_arg >= 1.
  w_arg = lax.max(x_bounded, lax.full_like(x, 1.0))
  zh, zl, prod_h, prod_l, psi_y = _lgamma_shifted_f64(w_arg)

  # Compute one shared logarithm _log_f64 for all three cases:
  # - x in (0, 1): needs ln(x) for lgamma(x) = lgamma(x + 1) - ln(x).
  # - x >= 2.5: needs ln(prod) for lgamma(x) = lgamma(z) - ln(prod).
  # - x <= 0: Euler reflection needs lgamma(y) + ln|delta|
  #   = (lgamma(z) - ln(prod)) + ln|delta| = lgamma(z) + ln(|delta| / prod).
  # When |delta| < 1e-290 on x <= 0, scale |delta| by 2^54 before dividing by
  # prod (which can be up to ~7!) so |delta| / prod cannot underflow to
  # subnormal, and pass k_adj = -54 to subtract 54 * ln(2) inside _log_f64.
  abs_delta_safe = lax.select(
      abs_delta == 0.0, lax.full_like(x, 1.0), abs_delta
  )
  is_tiny_neg = (~is_pos) & (abs_delta_safe < 1e-290)
  abs_d_s = lax.select(
      is_tiny_neg, abs_delta_safe * 18014398509481984.0, abs_delta_safe
  )
  # Share one reciprocal for both 1 / prod_h and 1 / abs_d_s: on x > 0,
  # d_for_rcp = 1 so rcp_both = 1 / prod_h; on x <= 0, rcp_both = 1 / (prod_h *
  # abs_d_s), giving 1 / prod_h = abs_d_s * rcp_both and 1 / abs_d_s = prod_h *
  # rcp_both.
  d_for_rcp = lax.select(is_pos, lax.full_like(x, 1.0), abs_d_s)
  rcp_both = lax.reciprocal(prod_h * d_for_rcp)
  rcp_prod = d_for_rcp * rcp_both
  qh = abs_d_s * rcp_prod
  ph_q, pl_q = two_prod(qh, prod_h)
  rem_q = ((abs_d_s - ph_q) - pl_q) - qh * prod_l

  pos_log_h = lax.select(in_small, x_small_in, prod_h)
  log_arg_h = lax.select(is_pos, pos_log_h, qh)
  k_adj = lax.select(is_tiny_neg, lax.full_like(x, -54.0), lax.full_like(x, 0.0))
  log_h, log_l = _log_f64(log_arg_h, k_adj)
  pos_log_corr = lax.select(in_small, lax.full_like(x, 0.0), prod_l * rcp_prod)
  neg_log_corr = rem_q * (prod_h * rcp_both)
  log_l = log_l + lax.select(is_pos, pos_log_corr, neg_log_corr)

  # Assemble positive result: subtract ln(x) on (0, 1), keep p_h + p_l on
  # [1, 2.5), and subtract ln(prod) from lgamma(z) on [2.5, inf).
  pos_small = lax.select(
      is_lt1, f2_sum(p_h, p_l, -log_h, -log_l), p_h + p_l
  )
  lg_pos = lax.select(
      in_small, pos_small, f2_sum(zh, zl, -log_h, -log_l)
  )

  # Euler's reflection formula for x <= 0:
  #   lgamma(x) = -(lgamma(1 - x) + ln|delta| + ln(sin(pi*delta) / (pi*delta))).
  # Approximate ln(sin(pi*delta) / (pi*delta)) = delta^2 * S(delta^2). Split
  # delta into 13-bit dh and 40-bit dl so dh^2 (26 bits) * s_hh (26 bits) has
  # no rounding error.
  cd = 1099511627777.0 * delta
  dh = cd - (cd - delta)
  dl = delta - dh
  dh2 = dh * dh
  d2_l = dl * (delta + dh)
  d2 = delta * delta
  sh_sinc, sl_sinc = fast_two_sum(
      lax.full_like(x, _SINC_F64[0]),
      d2 * lax.polynomial(d2, _SINC_F64[1:], unroll=len(_SINC_F64) - 1),
  )
  sl_sinc = sl_sinc + _SINC_C0_LO_F64
  s_hh, s_hl = split_half(sh_sinc)
  sinc_h = dh2 * s_hh
  sinc_l = dh2 * (s_hl + sl_sinc) + d2_l * sh_sinc
  # Add digamma(y) * y_lo to compensate for rounding in y = (1 - n) - delta.
  dy_corr = psi_y * y_lo
  sum_h, sum_l = f2_add(zh, zl, log_h, log_l)
  refl = -f2_sum(sum_h, sum_l, sinc_h, sinc_l + dy_corr)

  # Use root-window polynomials near negative zeros on (-17, -2).
  in_root_win, rw = _lgamma_neg_root_f64(x_neg_safe, n)
  neg_res = lax.select(in_root_win, rw, refl)

  res = lax.select(is_pos, lg_pos, neg_res)
  # Canonicalize -0.0 at x = 1.0, 2.0 to +0.0.
  res = lax.select(res == 0.0, lax.full_like(x, 0.0), res)
  # Return +inf at poles (x = +-0 and negative integers, which also covers
  # x = -inf) and on positive overflow (x > _MAX_FINITE_ARG_F64); keep NaNs.
  is_neg_int = (~is_pos) & (x_safe == lax.floor(x_safe))
  is_pole_or_inf = is_zero | is_neg_int | (x > _MAX_FINITE_ARG_F64)
  res = lax.select(is_pole_or_inf, lax.full_like(x, np.inf), res)
  return lax.select(x != x, lax.full_like(x, np.nan), res)


@config.explicit_x64_dtypes("allow")
@jit(static_argnames=["use_f64_for_f32"])
def lgamma_impl(x, *, use_f64_for_f32: bool = False):
  """Computes lgamma(x) for float32 and float64 inputs."""
  if x.dtype == np.float64:
    return _lgamma_impl_f64(x)
  return _lgamma_impl_f32(x, use_f64=use_f64_for_f32)
