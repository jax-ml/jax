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

Overview:
- Positive inputs (x >= 0.5 in float32, x > 0 in float64):
  1. Small x: approximated by polynomials of the form (x - 1)(x - 2) * P(d),
     which are zero at x = 1 and x = 2. Float32 uses two polynomials on
     [0.5, 1.5] (d = x - 1.0) and (1.5, 4] (d = x - 2.75), splitting the factors
     and leading terms into high and low parts so the result keeps extra
     precision when used in Euler's reflection formula. Float64 shifts (0, 1) up
     to x + 1, uses one polynomial on [1, 2.5) (d = x - 1.75), and shifts
     [2.5, 7) up to x + m >= 7.
  2. Larger x (x > 4 in float32, x >= 7 in float64): evaluated with Stirling's
     series
       lgamma(x) = (x - 0.5)(ln(x) - 1) + (0.5 ln(2 pi) - 0.5) + (1/x) S(1/x^2),
     splitting (x - 0.5) and (ln(x) - 1) into high and low parts so their
     leading product has no rounding error.
- Remaining inputs (x < 0.5 in float32, x <= 0 in float64):
  1. Euler's reflection formula:
       lgamma(x) = -(ln|delta| + lgamma(1 - x) - ln(pi delta / sin(pi delta))),
     where n = round(x), delta = x - n in [-0.5, 0.5], and y = (1 - n) - delta.
     Adding digamma(y) * y_lo corrects for rounding in y.
  2. Near negative zeros (on (-10, -2) in float32 and (-17, -2) in float64),
     the terms in Euler's reflection formula nearly cancel out. Each zero
     x_root = -k + r sits about |r| ~ 1 / k! from the integer -k. Multiplying
     delta - r by a power-of-two factor `scale` (or 5 on (-3, -2.5)) chosen so
     |r * scale| is in [1, 2) gives u = (delta - r) * scale of order 1 without
     dividing. We then evaluate u * P(u) from a table of polynomials (or
     _BASIN_MIN + u^2 * Q(u) around the local minimum on (-3, -2) in float32).

On CPU (`use_f64_for_f32=True`), float32 inputs are promoted to float64 and
shifted up to z >= 8 for Stirling's series, using short Taylor series near
x = 1 and x = 2 and Euler's reflection formula on x < 0.5 without a lookup
table.
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
    split_bits,
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
_C_STIRLING_HI = 0.418938547372818
_C_STIRLING_LO = -1.4168144879533884e-08
_SQRT2_OVER_2_BITS_F64 = 0x3FE6A09E667F3BCD
# Split of ln(2) with a 42-bit high part so k * _LN2_HI_F64 has no rounding
# error for |k| <= 2048.
_LN2_HI_F64 = 0.6931471805598903
_LN2_LO_F64 = 5.497923018708371e-14
# High/low float64 pair for 0.5 * ln(2 * pi) - 0.5.
_C_STIRLING_HI_F64 = 0.4189385332046727
_C_STIRLING_LO_F64 = 1.6728209650585413e-17


# Degree-7 polynomial for (ln(1 + f) - (f - 0.5 * f^2)) / f^3 on
# f = m - 1 in [sqrt(2)/2 - 1, sqrt(2) - 1], in ascending powers of f.
_LOG_Q = [
    0.3333333134651184, -0.250008225440979, 0.20001226663589478,
    -0.16623350977897644, 0.14201749861240387, -0.1316024214029312,
    0.12761665880680084, -0.07634450495243073,
]


@jit
def _log_f32(x):
  """Computes ln(x) as a `(log_hi, log_lo)` pair in float32."""
  bits = lax.bitcast_convert_type(x, np.int32)
  exp_off = lax.shift_right_arithmetic(
      bits - _SQRT2_OVER_2_BITS, lax.full_like(bits, 23)
  )
  k = lax.convert_element_type(exp_off, np.float32)
  mant_bits = bits - lax.shift_left(exp_off, lax.full_like(bits, 23))
  m = lax.bitcast_convert_type(mant_bits, np.float32)

  f = m - 1.0
  f2 = f * f
  th, tl = fast_two_sum(f, -0.5 * f2)
  tl = tl + (f2 * f) * lax.polynomial(f, _LOG_Q, unroll=len(_LOG_Q))
  log_hi, e_d = fast_two_sum(k * _LN2_HI, th)
  log_lo = e_d + (tl + k * _LN2_LO)
  return log_hi, log_lo


# Degree-11 polynomials for lgamma(x) / ((x - 1)(x - 2)) in ascending powers of
# d = x - center. The constant term c_0 is stored as a (hi, lo) pair, and the
# linear term c_1 is split into a 12-bit high part C1_HH plus a remainder stored
# in Q1_REM[0]:
# - _POS_05_15_*: for x in [0.5, 1.5] around center = 1.0.
# - _POS_15_4_*: for x in [1.5, 4.0] around center = 2.75.
_POS_05_15_C0_HI, _POS_05_15_C0_LO = 0.5772156715393066, -1.6366939803447167e-08
_POS_05_15_C1_HH = -0.2452392578125
_POS_05_15_Q1_REM = [
    -1.2042846719850786e-05, 0.15543700754642487, -0.11515380442142487,
    0.09211507439613342, -0.0770968347787857, 0.06875524669885635,
    -0.06163526326417923, 0.038027770817279816, -0.030093315988779068,
    0.08821531385183334, -0.08744385093450546,
]

_POS_15_4_C0_HI, _POS_15_4_C0_LO = 0.3620683252811432, -8.077157609420738e-09
_POS_15_4_C1_HH = -0.065704345703125
_POS_15_4_Q1_REM = [
    -2.4997580112540163e-05, 0.01603109948337078, -0.004419579170644283,
    0.0013021451886743307, -0.0004001049674116075, 0.0001287032209802419,
    -4.187468221061863e-05, 1.1072624147345778e-05, -3.4609686281328322e-06,
    2.6656666705093812e-06, -9.476669902142021e-07,
]

# Degree-2 polynomial in 1/x^2 for the Stirling series tail
# S(1/x^2) = 1/12 - 1/(360 * x^2) + ... on x >= 4.
_STIRLING_HI = [
    0.08333330601453781, -0.0027760176453739405, 0.0007350948872044683,
]

# Polynomial in delta^2 for ln(sin(pi * delta) / (pi * delta)) / delta^2 on
# |delta| <= 0.5. The constant term -pi^2 / 6 is split into the 9-bit value
# _SINC_C0_HH = -421/256 plus _SINC_REM[0], so (dh * dh) * _SINC_C0_HH has no
# rounding error when dh has 6 mantissa bits.
_SINC_C0_HH = -1.64453125
_SINC_REM = [
    -0.000402778823627159, -0.5411723256111145, -0.338631808757782,
    -0.25888514518737793, -0.14380806684494019, -0.34033387899398804,
]

# Degree-7 float64 polynomial in 1/z^2 for the Stirling series tail S(1/z^2)
# on z >= 8.
_STIRLING_F32_F64 = [
    0.08333333333333337, -0.0027777777777657103, 0.0007936507847927581,
    -0.0005952363828805681, 0.0008415863584863707, -0.001906033373818225,
    0.0057034901036038445, -0.005481381600466104,
]

# Degree-8 float64 Taylor polynomials for lgamma(1 + t) / t and
# lgamma(2 + t) / t on |t| <= 3/64.
_TAYLOR_1_F32_F64 = [
    -0.5772156649015329, 0.8224670334241132, -0.40068563438653143,
    0.27058080842778454, -0.20738555102867398, 0.1695571769974082,
    -0.1440498967688461, 0.12550966952474304, -0.11133426586956469,
]
_TAYLOR_2_F32_F64 = [
    0.42278433509846713, 0.3224670334241132, -0.0673523010531981,
    0.020580808427784546, -0.007385551028673986, 0.0028905103307415234,
    -0.001192753911703261, 0.0005096695247430425, -0.00022315475845357939,
]

# Degree-11 float64 polynomial in delta^2 for
# ln(sin(pi * delta) / (pi * delta)) / delta^2 on |delta| <= 0.5.
_SINC_F32_F64 = [
    -1.6449340668482264, -0.5411616168587046, -0.3391143534032649,
    -0.25101938254889333, -0.20019726882272687, -0.16674427259742153,
    -0.142353419429977, -0.1296848567043034, -0.08295902231943517,
    -0.20914242809307546, 0.16492085267622705, -0.3875246085244428,
]

# Interval (_BASIN_LO, _BASIN_HI) and minimum value _BASIN_MIN = lgamma(x_min)
# around the local minimum x_min ~= -2.61072 on (-3, -2).
_BASIN_LO, _BASIN_HI = -2.70703125, -2.5
_BASIN_MIN = -0.11862999200820923

# Lookup table of shape (18, 16) for approximating lgamma(x) near its zeros
# x_root = -k + r alongside negative integers -k (k in {2, ..., 9}).
#
# Near x = -k, lgamma(-k + delta) behaves like -ln|delta / r| plus a smooth
# remainder, where |r| ~ 1 / k! shrinks rapidly with k. Multiplying delta - r by
# `scale` (chosen with the same sign as r so |r * scale| is in [1, 2)) gives a
# rescaled offset u = (delta - r) * scale of order 1 without dividing, so
# -ln|delta / r| = -ln(1 + u / |r * scale|) can be approximated by a fixed-degree
# polynomial u * P(u) for every k.
#
# Column 2 * (k - 2) covers the zero below -k (r < 0, delta < 0), and column
# 2 * (k - 2) + 1 covers the zero above -k (r > 0, delta > 0). Because
# lgamma(x) > 0 on (-2, -1), column 1 (k = 2, delta > 0) has no zero; instead it
# stores the local minimum x_min = -3 + 0.38927913 on (_BASIN_LO, _BASIN_HI),
# where lgamma(x) = _BASIN_MIN + u^2 * Q(u).
#
# Rows:
#   row 0 (`r_hi`): float32 high part of r = x_root - (-k) (or x_min - (-3) in
#       column 1).
#   row 1 (`neg_r_lo_sc`): float32(-(r - r_hi) * scale), added to
#       (delta - r_hi) * scale to get u = (delta - r) * scale as (u, u_lo).
#   row 2 (`scale`): scale factor (signed power of two, or 5.0 in column 3).
#   row 3 (`u_center`), row 4 (`u_half`): center and half-width of the interval
#       |u - u_center| <= u_half where this column is used.
#   row 5 (`c_0_hi`), row 6 (`c_0_lo`): (hi, lo) float32 pair for the constant
#       term c_0 of P(u) (or Q(u) in column 1).
#   rows 7..17 (`c_1 .. c_11`): remaining degree-11 polynomial coefficients in
#       ascending powers of u.
_NEG_TABLE = np.array([
    [
        -0.4570247530937195, 0.3892791271209717, -0.14358088374137878, 0.2523174285888672,
        -0.03936183825135231, 0.04470571503043175, -0.008218168281018734, 0.008455359376966953,
        -0.0013852944830432534, 0.001392519916407764, -0.00019833340775221586, 0.00019849211093969643,
        -2.4800270693958737e-05, 2.480290459061507e-05, -2.755714831437217e-06, 2.7557489374885336e-06,
    ],
    [
        5.9491675585832127e-08, -3.547906857193084e-08, -3.686880845066298e-08, 3.765813971767784e-07,
        -4.765390571037642e-08, -3.5510503426650075e-09, -5.321572960070853e-09, -8.062121636953634e-09,
        3.0605470158207027e-08, -2.781805230611667e-09, 3.5017870825271302e-09, 1.2922014569483053e-08,
        7.863690254517053e-10, 2.6953594911560685e-08, 4.606850900756854e-09, -4.458716773569904e-08,
    ],
    [
        -4.0, 8.0, -8.0, 5.0,
        -32.0, 32.0, -128.0, 128.0,
        -1024.0, 1024.0, -8192.0, 8192.0,
        -65536.0, 65536.0, -524288.0, 524288.0,
    ],
    [
        -0.15932877361774445, 0.05764194577932358, 0.08905893564224243, -0.1125136986374855,
        0.13573573529720306, 0.06965170055627823, 0.12568676471710205, 0.14650428295135498,
        0.040686994791030884, 0.03328812122344971, 0.11742763221263885, 0.11752159148454666,
        0.11746834218502045, 0.1174808144569397, 0.10442119836807251, 0.10442249476909637,
    ],
    [
        0.331330806016922, 0.829135000705719, 0.5033261179924011, 0.3158716857433319,
        0.6167409420013428, 0.6234267354011536, 0.48645418882369995, 0.576084315776825,
        0.38452547788619995, 0.38452547788619995, 0.7463630437850952, 0.7469602823257446,
        0.7466217875480652, 0.7467010617256165, 0.6636950373649597, 0.6637032628059387,
    ],
    [
        -0.3789008557796478, 0.08475256711244583, -0.972735583782196, -0.38287004828453064,
        -0.8372024893760681, -0.6476581692695618, -0.9637670516967773, -0.9104357957839966,
        -0.7067740559577942, -0.6994584202766418, -0.6157262921333313, -0.6147421002388,
        -0.6152997016906738, -0.6151690483093262, -0.6921472549438477, -0.6921300888061523,
    ],
    [
        -8.088338887546342e-09, 3.0228826108924522e-09, -2.629517048546859e-09, 1.1159768220636579e-08,
        -2.9614779961661952e-08, 1.8307664717553962e-08, -2.4201131765266837e-08, -6.1257079408960635e-09,
        3.2978766384417213e-09, -7.87305065585997e-09, -1.9248458826837123e-08, -1.1028030044712978e-08,
        1.953765593754042e-09, 2.9598630213456545e-09, -6.884640413318266e-09, 6.486552184270522e-09,
    ],
    [
        0.3036450743675232, -0.008314274251461029, 0.40361469984054565, 0.38300758600234985,
        0.31665563583374023, 0.24581515789031982, 0.4519508481025696, 0.42695537209510803,
        0.24847832322120667, 0.24590644240379333, 0.189408078789711, 0.1891053169965744,
        0.189276784658432, 0.1892365962266922, 0.23953098058700562, 0.23952503502368927,
    ],
    [
        -0.022051390260457993, 0.0031260584946721792, -0.21927523612976074, -0.16076107323169708,
        -0.16679921746253967, -0.11384912580251694, -0.2863669991493225, -0.26293572783470154,
        -0.11677572876214981, -0.11496737599372864, -0.07771792262792587, -0.07753165811300278,
        -0.07763715088367462, -0.07761242240667343, -0.11052641272544861, -0.1105223000049591,
    ],
    [
        0.034069448709487915, -0.0006115938303992152, 0.1437695175409317, 0.10020365566015244,
        0.09931786358356476, 0.05968732014298439, 0.20417243242263794, 0.18220442533493042,
        0.06174071878194809, 0.06046922504901886, 0.035874247550964355, 0.03575965389609337,
        0.03582454100251198, 0.03580933064222336, 0.057373229414224625, 0.0573703832924366,
    ],
    [
        -0.005664459429681301, 0.00019508387777023017, -0.10001526027917862, -0.06232548877596855,
        -0.0630946010351181, -0.03337479010224342, -0.15530124306678772, -0.13474911451339722,
        -0.03481920436024666, -0.03392516076564789, -0.017664317041635513, -0.017593814060091972,
        -0.017633728682994843, -0.017624370753765106, -0.031769104301929474, -0.03176713362336159,
    ],
    [
        0.006060823332518339, -4.814308704226278e-05, 0.07264159619808197, 0.04140627756714821,
        0.041801027953624725, 0.01946423202753067, 0.12307784706354141, 0.10388010740280151,
        0.02045558951795101, 0.019826851785182953, 0.009074314497411251, 0.009030869230628014,
        0.00905546173453331, 0.009049694985151291, 0.018352853134274483, 0.018351485952734947,
    ],
    [
        -0.0014674802077934146, 1.4522259334626142e-05, -0.05414929240942001, -0.028049077838659286,
        -0.028297534212470055, -0.011698040179908276, -0.0998203381896019, -0.08141360431909561,
        -0.01236222218722105, -0.011920804157853127, -0.0047876955941319466, -0.004760960582643747,
        -0.004776091780513525, -0.004772542510181665, -0.010889245197176933, -0.010888298973441124,
    ],
    [
        0.0012524601770564914, -3.955991360271582e-06, 0.040391478687524796, 0.019385607913136482,
        0.019129019230604172, 0.006993971299380064, 0.08203091472387314, 0.06410042941570282,
        0.007611149922013283, 0.0073030125349760056, 0.002504174830392003, 0.0024882012512534857,
        0.002497240202501416, 0.0024951195809990168, 0.006404976826161146, 0.006404341198503971,
    ],
    [
        -0.00038447920815087855, 1.159806856776413e-06, -0.031247643753886223, -0.014391499571502209,
        -0.013998933136463165, -0.004125987179577351, -0.07254776358604431, -0.056765373796224594,
        -0.004736089147627354, -0.00451267696917057, -0.0013456688029691577, -0.00133602658752352,
        -0.0013414818095043302, -0.001340201823040843, -0.0038705654442310333, -0.0038701335433870554,
    ],
    [
        0.0003150988486595452, -3.4455067066119227e-07, 0.029085591435432434, 0.009928914718329906,
        0.012214251793920994, 0.0030908638145774603, 0.07091644406318665, 0.057515501976013184,
        0.0031194700859487057, 0.0029423751402646303, 0.0009176156017929316, 0.000910310831386596,
        0.0009144428768195212, 0.0009134731953963637, 0.002968098968267441, 0.002967730863019824,
    ],
    [
        1.947129476320697e-06, 1.2183934927634255e-07, -0.025662211701273918, -0.00039150964585132897,
        -0.008898217231035233, -0.0026070764288306236, -0.05637626722455025, -0.04427916184067726,
        -0.002250844379886985, -0.002149523701518774, -0.0006105965003371239, -0.0006052380776964128,
        -0.0006082686013542116, -0.0006075572455301881, -0.0022210015449672937, -0.0022206977009773254,
    ],
    [
        0.00018473352247383446, -2.9646693988638617e-08, 0.011288844980299473, 0.017486728727817535,
        0.003027826314792037, 0.0011142778676003218, 0.02157474309206009, 0.015372389927506447,
        0.0011180674191564322, 0.0011126288445666432, 0.0001965321134775877, 0.00019464737852104008,
        0.0001957131316885352, 0.00019546292605809867, 0.000803905597422272, 0.0008037853986024857,
    ],
], dtype=np.float32)


def _lgamma_pos_small(x_small):
  """Evaluates lgamma(x) on [0.5, 4] as a `(res_hi, res_lo)` pair.

  Approximates lgamma(x) as (x - 1)(x - 2) * P(d), splitting (x - 1), (x - 2),
  and P(d) into high and low parts so their product keeps extra precision for
  Euler's reflection formula.
  """
  x_sm = lax.min(x_small, lax.full_like(x_small, 4.0))
  in_15_4 = x_sm > 1.5
  u = x_sm - 1.0
  # On [0.5, 2], |-2| >= x_sm satisfies fast_two_sum's precondition; on
  # (2, 4], -2 + x_sm is exact by Sterbenz's lemma so v_err == 0.
  v, v_err = fast_two_sum(lax.full_like(x_sm, -2.0), x_sm)
  d = x_sm - lax.select(
      in_15_4, lax.full_like(x_sm, 2.75), lax.full_like(x_sm, 1.0)
  )
  c0_hi = lax.select(
      in_15_4,
      lax.full_like(x_sm, _POS_15_4_C0_HI),
      lax.full_like(x_sm, _POS_05_15_C0_HI),
  )
  c0_lo = lax.select(
      in_15_4,
      lax.full_like(x_sm, _POS_15_4_C0_LO),
      lax.full_like(x_sm, _POS_05_15_C0_LO),
  )
  c1_hh = lax.select(
      in_15_4,
      lax.full_like(x_sm, _POS_15_4_C1_HH),
      lax.full_like(x_sm, _POS_05_15_C1_HH),
  )
  coeffs_q1 = [
      lax.select(in_15_4, lax.full_like(x_sm, c2), lax.full_like(x_sm, c1))
      for c1, c2 in zip(_POS_05_15_Q1_REM, _POS_15_4_Q1_REM)
  ]
  d_hi, d_lo = split_half(d)
  p_sm, p_err = fast_two_sum(c0_hi, d_hi * c1_hh)
  p_tail = (
      c0_lo
      + d_lo * c1_hh
      + d * lax.polynomial(d, coeffs_q1, unroll=len(coeffs_q1))
  )
  p_sm, p_err = fast_two_sum(p_sm, p_err + p_tail)

  u_hi, u_lo = split_bits(u, 8)
  v_hi, v_lo = split_bits(v, 8)
  uv_hi = u_hi * v_hi
  uv_lo = u_lo * v + u_hi * (v_lo + v_err)
  p_hi_sm, p_lo_sm = split_bits(p_sm, 8)
  return fast_two_sum(
      uv_hi * p_hi_sm, uv_lo * p_sm + uv_hi * (p_lo_sm + p_err)
  )


def _lgamma_pos_large(x_gt4):
  """Evaluates Stirling's series for x >= 4.

  Returns `(p_hi, s_pos, psi_large)`, where `(p_hi, s_pos)` is a `(hi, lo)` pair
  for `lgamma(x)` and `psi_large = ln(x) - 1/(2x)` approximates `digamma(x)`.
  """
  s, log_lo = _log_f32(x_gt4)
  h_x, s_lo = split_half(s)
  l_x = s_lo + log_lo

  b, b_err = fast_two_sum(x_gt4, lax.full_like(x_gt4, -0.5))
  b_hi, b_lo = split_half(b)
  hx_m1 = h_x - 1.0
  p_hi = b_hi * hx_m1
  p_lo = (b_lo + b_err) * hx_m1 + b * l_x
  inv_x = lax.reciprocal(x_gt4)
  sp = lax.polynomial(inv_x * inv_x, _STIRLING_HI, unroll=len(_STIRLING_HI))
  p_hi, p_rem = fast_two_sum(p_hi, lax.full_like(x_gt4, _C_STIRLING_HI))
  s_pos = p_rem + (p_lo + (_C_STIRLING_LO + inv_x * sp))
  return p_hi, s_pos, s - 0.5 * inv_x


def _lgamma_neg_reflection(delta, log_delta_hi, lg_pos_hi, lg_pos_lo, dy_corr):
  """Evaluates Euler's reflection formula -(ln|delta| + lgamma(1 - x) - ln(pi delta / sin(pi delta)))."""
  d2 = delta * delta
  dh, dl = split_bits(delta, 6)
  sinc_hi = (dh * dh) * _SINC_C0_HH
  sinc_lo = dl * (delta + dh) * _SINC_C0_HH + d2 * lax.polynomial(
      d2, _SINC_REM, unroll=len(_SINC_REM)
  )
  neg_hi, neg_e1 = fast_two_sum(log_delta_hi, sinc_hi)
  sum_hi, sum_err = two_sum(lg_pos_hi, neg_hi)
  return -(sum_hi + (sum_err + neg_e1 + lg_pos_lo + (sinc_lo + dy_corr)))


def _lgamma_neg_root(x_neg_safe, n, delta):
  """Evaluates lgamma(x) near its zeros on (-10, -2) using `_NEG_TABLE`.

  Returns `(in_root_win, rw)`, where `in_root_win` indicates whether `x` falls
  in one of the table intervals and `rw` is the polynomial value there.
  """
  # On (-3, -2), the two zeros are at -2.748 (column 3, n = -3) and -2.457
  # (column 0, n = -2), separated by the local minimum at -2.611 (column 1,
  # n = -3). Because round-to-nearest-even switches from -2 to -3 at
  # _BASIN_HI = -2.5, n = round(x) already gives -2 on [-2.5, -2) and -3 on
  # (-3, -2.5), matching all three columns.
  in_basin = (x_neg_safe > _BASIN_LO) & (x_neg_safe < _BASIN_HI)
  n_i32 = lax.convert_element_type(n, np.int32)
  in_k_range = (x_neg_safe < -2.0) & (n_i32 >= -9)
  base_idx = lax.select(
      delta > 0.0, lax.full_like(n_i32, -3), lax.full_like(n_i32, -4)
  )
  idx = lax.clamp(
      lax.full_like(n_i32, 0),
      lax.select(
          in_basin, lax.full_like(n_i32, 1), base_idx - (n_i32 + n_i32)
      ),
      lax.full_like(n_i32, 15),
  )

  idx_exp = lax.broadcast_in_dim(idx, (*idx.shape, 1), tuple(range(idx.ndim)))
  dnums = slicing.GatherDimensionNumbers(
      offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
  )

  def _gather_row(i):
    return slicing.gather(
        lax._const(x_neg_safe, _NEG_TABLE[i]),
        idx_exp,
        dnums,
        slice_sizes=(1,),
    )

  # Gather one 1D row at a time so GPU kernels do not keep all 18 table rows
  # live per thread.
  r_hi = _gather_row(0)
  neg_r_lo_sc = _gather_row(1)
  scale = _gather_row(2)
  u_center = _gather_row(3)
  u_half = _gather_row(4)
  u, u_lo = fast_two_sum((delta - r_hi) * scale, neg_r_lo_sc)
  in_root_win = in_k_range & (lax.abs(u - u_center) <= u_half)
  u_safe = lax.select(in_root_win, u, lax.full_like(x_neg_safe, 0.0))
  # Keep 6 mantissa bits of u in column 1 (so uh * uh fits in 12 bits) and 12
  # bits elsewhere.
  u_mask = lax.select(
      in_basin,
      lax.full_like(n_i32, -0x40000),
      lax.full_like(n_i32, -0x1000),
  )
  uh = lax.bitcast_convert_type(
      lax.bitcast_convert_type(u_safe, np.int32) & u_mask, np.float32
  )
  u_tail = (u_safe - uh) + u_lo
  lead_hi = lax.select(in_basin, uh * uh, uh)
  lead_lo = lax.select(in_basin, u_tail * (u_safe + uh), u_tail)
  b_add_hi = lax.select(
      in_basin,
      lax.full_like(x_neg_safe, _BASIN_MIN),
      lax.full_like(x_neg_safe, 0.0),
  )
  p_tail = _gather_row(17)
  for i in range(16, 6, -1):
    p_tail = p_tail * u_safe + _gather_row(i)
  pw_hi, pw_lo = fast_two_sum(_gather_row(5), u_safe * p_tail)
  c0_lo = _gather_row(6)
  pw_hh, pw_hl = split_half(pw_hi)
  rw_hi, rw_err = fast_two_sum(b_add_hi, lead_hi * pw_hh)
  rw_lo = lead_hi * (pw_hl + (pw_lo + c0_lo)) + lead_lo * pw_hi
  return in_root_win, rw_hi + (rw_err + rw_lo)


def _lgamma_f32_with_f64(x, is_pos, x_neg_safe, delta, abs_delta):
  """Evaluates float32 lgamma(x) in float64 via shifted Stirling and reflection."""
  x_pos_d = lax.convert_element_type(
      lax.min(x, lax.full_like(x, _MAX_FINITE_ARG)), np.float64
  )
  y_d = lax.select(
      is_pos, x_pos_d, 1.0 - lax.convert_element_type(x_neg_safe, np.float64)
  )

  # Shift y_d in [0.5, 8) up to z_d = y_d + m >= 8 using
  # lgamma(y_d) = lgamma(z_d) - ln(prod), where prod = y_d * ... * (y_d + m - 1).
  need_shift = y_d < 8.0
  m = lax.select(need_shift, lax.ceil(8.0 - y_d), lax.full_like(y_d, 0.0))
  one = lax.full_like(y_d, 1.0)
  factors = [lax.select(need_shift, y_d, one)] + [
      lax.select(m > float(k), y_d + float(k), one) for k in range(1, 8)
  ]
  prod = ((factors[0] * factors[1]) * (factors[2] * factors[3])) * (
      (factors[4] * factors[5]) * (factors[6] * factors[7])
  )
  z_d = y_d + m

  ln_z = lax.log(z_d)
  inv_z = lax.reciprocal(z_d)
  sp = lax.polynomial(
      inv_z * inv_z, _STIRLING_F32_F64, unroll=len(_STIRLING_F32_F64)
  )
  lg_z = (z_d - 0.5) * (ln_z - 1.0) + _C_STIRLING_HI_F64 + inv_z * sp

  # Share one logarithm for ln(prod) on x >= 0.5 and ln(|delta| / prod) on
  # x < 0.5, since lgamma(1 - x) + ln|delta| = lgamma(z_d) + ln(|delta| / prod).
  abs_delta_safe = lax.select(
      abs_delta == 0.0,
      one,
      lax.convert_element_type(abs_delta, np.float64),
  )
  log_arg = lax.select(is_pos, prod, abs_delta_safe / prod)
  ln_arg = lax.log(log_arg)

  # Near x = 1 and x = 2, lgamma(z_d) and ln(prod) cancel; use short Taylor
  # series there so x = 1 and x = 2 return 0 and nearby inputs lose no bits.
  near_2 = lax.abs(x_pos_d - 2.0) <= (3.0 / 64.0)
  t = x_pos_d - lax.select(
      near_2, lax.full_like(x_pos_d, 2.0), lax.full_like(x_pos_d, 1.0)
  )
  t_coeffs = [
      lax.select(near_2, lax.full_like(x_pos_d, c2), lax.full_like(x_pos_d, c1))
      for c1, c2 in zip(_TAYLOR_1_F32_F64, _TAYLOR_2_F32_F64)
  ]
  taylor_d = t * lax.polynomial(t, t_coeffs, unroll=len(t_coeffs))
  lg_pos_d = lax.select(
      lax.abs(t) <= (3.0 / 64.0),
      taylor_d,
      lg_z - ln_arg,
  )

  d_d = lax.convert_element_type(delta, np.float64)
  d2_d = d_d * d_d
  sinc_d = d2_d * lax.polynomial(
      d2_d, _SINC_F32_F64, unroll=len(_SINC_F32_F64)
  )
  refl_d = -(lg_z + ln_arg + sinc_d)

  return lax.convert_element_type(
      lax.select(is_pos, lg_pos_d, refl_d), np.float32
  )


def _lgamma_impl_f32(x, *, use_f64=False):
  """Computes float32 lgamma(x) across all normal and special inputs."""
  x = lax.convert_element_type(x, np.float32)
  # Treat subnormals as zero so backends that flush subnormals and backends that
  # preserve them return the same result (+inf).
  is_zero = lax.abs(x) < float(np.finfo(np.float32).tiny)
  is_pos = x >= 0.5

  # Every float32 <= -2^23 (_ALL_INTEGERS_BELOW) is an integer. Clamp those and
  # inputs >= 0.5 to -0.5 so round(x_neg_safe) cannot overflow.
  x_neg_safe = lax.select(
      (~is_pos) & (x > _ALL_INTEGERS_BELOW),
      x,
      lax.full_like(x, -0.5),
  )
  # Split x < 0.5 into integer n = round(x) and remainder delta = x - n in
  # [-0.5, 0.5] (x - n has no rounding error).
  n = lax.round(x_neg_safe, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = x_neg_safe - n
  abs_delta = lax.abs(delta)

  if use_f64:
    res = _lgamma_f32_with_f64(x, is_pos, x_neg_safe, delta, abs_delta)
  else:
    # Compute y = 1 - x = (1 - n) - delta and its rounding error y_lo, since
    # subtracting delta from a much larger (1 - n) drops low bits of delta.
    one_m_n = 1.0 - n
    y = one_m_n - delta
    y_lo = (one_m_n - y) - delta

    # Both x >= 0.5 (at x) and x < 0.5 (via reflection at y = 1 - x > 0.5)
    # evaluate lgamma at an argument >= 0.5. Clamp to _MAX_FINITE_ARG to avoid
    # inf - inf NaNs in Stirling's series.
    arg_pos = lax.select(is_pos, x, y)
    x_bounded = lax.min(arg_pos, lax.full_like(x, _MAX_FINITE_ARG))

    # Compute ln|delta| for Euler's reflection formula on x < 0.5.
    log_hi, log_lo = _log_f32(abs_delta)

    # Evaluate lgamma on [0.5, 4] and [4, inf) as (hi, lo) pairs so Euler's
    # reflection formula can add the high parts before adding the low parts.
    lg_small_hi, lg_small_lo = _lgamma_pos_small(x_bounded)
    p_hi, s_pos, psi = _lgamma_pos_large(x_bounded)
    in_le_four = x_bounded <= 4.0
    lg_pos_hi = lax.select(in_le_four, lg_small_hi, p_hi)
    lg_pos_lo = lax.select(in_le_four, lg_small_lo, s_pos)
    lg_pos = lg_pos_hi + lg_pos_lo

    # Use _NEG_TABLE near negative zeros on (-10, -2) where Euler's reflection
    # formula cancels, and Euler's formula (with psi * y_lo to account for
    # rounding in y) everywhere else.
    refl = _lgamma_neg_reflection(
        delta, log_hi, lg_pos_hi, lg_pos_lo + log_lo, psi * y_lo
    )
    in_root_win, rw = _lgamma_neg_root(x_neg_safe, n, delta)
    neg_res = lax.select(in_root_win, rw, refl)
    res = lax.select(is_pos, lg_pos, neg_res)
  # Replace -0.0 at x = 1.0 and 2.0 with +0.0.
  res = lax.select(res == 0.0, lax.full_like(x, 0.0), res)
  # Return +inf at poles (x = +-0 and negative integers, which also matches
  # x = -inf) and on positive overflow (x > _MAX_FINITE_ARG); preserve NaNs.
  is_neg_int = (~is_pos) & (x == lax.floor(x))
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

# Degree-25 polynomial for lgamma(x) / ((x - 1)(x - 2)) in ascending powers of
# d = x - 1.75 on [1.0, 2.5], with a (hi, lo) split on the constant term.
_POS_1_2_C0_LO_F64 = 1.459080004873635e-17
_POS_1_2_F64 = [
    0.4501393121092563, -0.11948158662524275, 0.044520447198509394,
    -0.01905395783024671, 0.008776370425279009, -0.004226194125966872,
    0.002095533879199047, -0.0010605660739579296, 0.0005448970031937583,
    -0.0002831906437563517, 0.00014851663460304423, -7.846154955361401e-05,
    4.170628948416333e-05, -2.2282626664252453e-05, 1.1945800482551106e-05,
    -6.432056080553514e-06, 3.516938080297205e-06, -1.91393390768375e-06,
    9.415575972716519e-07, -5.001201006581048e-07, 4.357288953734357e-07,
    -2.52829877822819e-07, -3.412184404355e-08, 2.8063702745014876e-08,
    9.135129327120217e-08, -5.296021598657274e-08,
]

# Polynomial in 1/z^2 for the Stirling series tail S(1/z^2) on z >= 7.
_STIRLING_F64 = [
    0.08333333333333333, -0.002777777777777778, 0.0007936507936507791,
    -0.0005952380952234643, 0.0008417508341741269, -0.0019175246162359454,
    0.006409817242328721, -0.029496160785839165, 0.17520548011608733,
    -1.1579334089679656, 5.626868322268739,
]

# Degree-15 polynomial for ln(sin(pi * delta) / (pi * delta)) / delta^2 on
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

# Lookup table of shape (21, 30) for approximating float64 lgamma(x) near its
# zeros x_root = -k + r alongside negative integers -k (k in {2, ..., 16}). For
# k >= 17, |r| ~ 1 / k! is smaller than the float64 spacing at -k, so no
# non-integer float64 gets close to a zero.
# As in _NEG_TABLE, u = (delta_root - r) * scale rescales delta_root - r by a
# signed power of two so |r * scale| is in [1, 2), and lgamma(-k + delta_root)
# is evaluated as u * P(u) without dividing.
# Rows:
#   row 0 (`r0`): float64 high part of r = x_root - (-k).
#   row 1 (`neg_r1_sc`): float64(-(r - r0) * scale), added to
#       (delta_root - r0) * scale to get u as (u_h, u_l).
#   row 2 (`scale`): signed power-of-two scale factor.
#   row 3 (`u_bound`): max |u_h| where this column is used (-1.0 disables
#       column 1, since lgamma(x) > 0 on (-2, -1)).
#   rows 4..5 (`c_0_hi, c_0_lo`): (hi, lo) float64 pair for the constant term
#       c_0 of P(u).
#   rows 6..20 (`c_1 .. c_15`): remaining degree-15 polynomial coefficients in
#       ascending powers of u.
_NEG_TABLE_F64 = np.array([
    [
        -0.45702473822080064, 1.0, -0.14358088834998006, 0.2523173532725874,
        -0.03936183974053688, 0.04470571514140207, -0.008218168322593521, 0.008455359439952277,
        -0.0013852944531550972, 0.0013925199191243706, -0.00019833340732475162, 0.0001984921093623021,
        -2.48002706819597e-05, 2.4802904179335845e-05, -2.7557148226503463e-06, 2.7557490225318057e-06,
        -2.755730136466002e-07, 2.7557337083353163e-07, -2.5052106852407546e-08, 2.5052109918476226e-08,
        -2.087675687777539e-09, 2.087675709796081e-09, -1.6059043830108875e-10, 1.6059043843534355e-10,
        -1.147074559737784e-11, 1.147074559808161e-11, -7.647163731803788e-13, 7.647163731835846e-13,
        -4.779477332386745e-14, 4.779477332388026e-14,
    ],
    [
        7.374216166297825e-17, 0.0, 3.0902451213713215e-17, 8.187559667653e-17,
        1.0474041352285324e-16, -2.631672941626257e-17, -7.145812372424199e-17, -1.0536642505408046e-16,
        -2.6926829374920783e-17, 7.260611820182048e-17, 9.683413730558741e-17, -5.1313950543807896e-17,
        6.608633365242895e-17, -2.066990455822503e-17, -2.948836817073382e-17, 3.408029500652768e-17,
        -1.026452169777367e-16, -6.082855828436849e-17, 8.673732504947454e-17, 4.568365361352113e-17,
        -2.6036547429543917e-17, 4.648678906693535e-17, 6.052854446145378e-17, -5.553514636250946e-17,
        5.764266834003625e-17, 1.0761937648918022e-16, 1.0744766688836333e-16, 8.364025615771848e-17,
        -6.374822167728059e-17, 3.279156503253523e-17,
    ],
    [
        -4.0, 1.0, -8.0, 4.0,
        -32.0, 32.0, -128.0, 128.0,
        -1024.0, 1024.0, -8192.0, 8192.0,
        -65536.0, 65536.0, -524288.0, 524288.0,
        -4194304.0, 4194304.0, -67108864.0, 67108864.0,
        -536870912.0, 536870912.0, -8589934592.0, 8589934592.0,
        -137438953472.0, 137438953472.0, -2199023255552.0, 2199023255552.0,
        -35184372088832.0, 35184372088832.0,
    ],
    [
        0.23399666596904994, -1.0, 0.2010132436899721, 0.1766221472908112,
        0.2204263025470065, 0.2503520047918516, 0.18408697042609487, 0.189400051454931,
        0.2482447660053934, 0.2495395695070872, 0.2843307727407639, 0.2845582879817963,
        0.28442934439725936, 0.28445954745196694, 0.2528379372640983, 0.2528410751169022,
        0.2022714738502482, 0.20227173602659887, 0.29421322554254503, 0.2942132615506126,
        0.19614216133003703, 0.19614216339873214, 0.24140573819371866, 0.24140573839553567,
        0.2758922723282533, 0.27589227234518027, 0.29428509049188106, 0.29428509049311474,
        0.29428509049245843, 0.2942850904925373,
    ],
    [
        -0.37890086200541434, 0.0, -0.9727355822664189, -0.4785875464028997,
        -0.8372025276918935, -0.6476581514313658, -0.9637670669011984, -0.9104357938769033,
        -0.7067740526614064, -0.6994584281519016, -0.6157263111707555, -0.6147421110556793,
        -0.6152996995258233, -0.6151690451383635, -0.6921472615910105, -0.6921300820821211,
        -0.8651744612462748, -0.8651722184402321, -0.5948067439395661, -0.5948065983455844,
        -0.8922100161239319, -0.8922099973038025, -0.7249206310610562, -0.724920629848978,
        -0.6343055516870568, -0.6343055516092231, -0.5946614546726241, -0.5946614546676383,
        -0.5946614546702905, -0.5946614546699719,
    ],
    [
        1.0178947128065389e-17, 0.0, 1.680883852146608e-17, -1.418360844233643e-17,
        -2.1558708977211774e-17, 4.5832715730910957e-17, 1.529218842797825e-17, -4.322277107134316e-17,
        -9.891571315019126e-18, -2.799691441429464e-17, -2.4746896893354217e-17, -4.0932999483366435e-17,
        3.742043651768266e-17, -1.9687756504818555e-17, -9.262945610925148e-18, 2.6796720192843015e-17,
        9.154176558192696e-18, -2.8869195168189655e-17, 3.0652073300510556e-17, 1.2452254603382799e-17,
        4.290291251181273e-17, 2.2124159245472052e-17, 1.4738066094102168e-17, -1.2181137349759676e-17,
        4.8000714377831255e-18, -2.8300021236766846e-18, -4.139729433508783e-17, 4.324292462191617e-17,
        -4.3355558099326364e-17, 4.520118243237312e-17,
    ],
    [
        0.3036450594771248, 0.0, 0.40361466206856184, 0.5984493422318542,
        0.31665558880649136, 0.2458151197137592, 0.45195083204520764, 0.42695531951257715,
        0.24847832708653828, 0.24590643848377478, 0.18940804651307155, 0.1891052896685478,
        0.18927676385839376, 0.18923657232324684, 0.23953094320306095, 0.23952499790182796,
        0.3742629390921931, 0.37426196888419583, 0.17689750966792545, 0.17689746636778964,
        0.3980193522380564, 0.39801934384230253, 0.26275496044931484, 0.2627549600099846,
        0.20117176643816817, 0.20117176641348297, 0.17681112283593947, 0.17681112283445702,
        0.17681112283524564, 0.17681112283515088,
    ],
    [
        -0.022051424110593448, 0.0, -0.21927536386167615, -0.31398648307567195,
        -0.16679967714774294, -0.11384902086427871, -0.28636749670640116, -0.26293754475178677,
        -0.11677572963843691, -0.11496737800337307, -0.07771794262567715, -0.0775316762242418,
        -0.07763716897959962, -0.07761244176562901, -0.1105264384322145, -0.11052232345931529,
        -0.21586835122916434, -0.21586751183107097, -0.07014655019853033, -0.07014652444332069,
        -0.2367445672035108, -0.23674455971273517, -0.126984327775816, -0.12698432745733643,
        -0.08506957886034, -0.08506957884468204, -0.0700951730051343, -0.07009517300425275,
        -0.0700951730047217, -0.07009517300466535,
    ],
    [
        0.0340694632180287, 0.0, 0.14377215386670306, 0.24463782309966886,
        0.09932094116349979, 0.05968864623905906, 0.20417379720753878, 0.18220984454847963,
        0.061740735817486556, 0.06046924098154247, 0.03587539917433143, 0.03576080168532009,
        0.0358256931969031, 0.03581048016487808, 0.057375072748974236, 0.05737222461709699,
        0.14007274757786212, 0.14007202135301022, 0.03129272892671442, 0.031292713607343904,
        0.1584194047560059, 0.15841939807266092, 0.0690401692407227, 0.0690401690098503,
        0.04047007961185387, 0.04047007960192195, 0.03126217315850645, 0.031262173157982215,
        0.03126217315826109, 0.031262173158227574,
    ],
    [
        -0.005664467350190596, 0.0, -0.10001201737739722, -0.1902013235342802,
        -0.06308166436942, -0.03337834780337359, -0.1552762330761731, -0.13468516907993686,
        -0.034819275964107436, -0.033925255837705615, -0.01766448223623683, -0.017593977920452485,
        -0.0176338944847044, -0.017624534904450786, -0.03176940245437756, -0.031767431152162746,
        -0.09694982830343693, -0.09694919999305635, -0.014890500049968346, -0.01489049093790457,
        -0.11307470313855102, -0.11307469717560374, -0.040038914426010846, -0.04003891425864668,
        -0.02053631693892629, -0.020536316932626416, -0.014872327492970093, -0.014872327492658355,
        -0.014872327492824189, -0.01487232749280426,
    ],
    [
        0.00606077688522543, 0.0, 0.07256722253371145, 0.15793603508219253,
        0.0417346283318727, 0.019443328237140102, 0.1230095242729358, 0.10370423348165567,
        0.02045485655062537, 0.01982624700666358, 0.009060117908724958, 0.009016741226982476,
        0.009041295010522293, 0.00903553667500681, 0.018324140388302927, 0.01832277597380608,
        0.0698987175918336, 0.06989817399444316, 0.007380807756946222, 0.0073808023370299,
        0.08407198514844576, 0.08407197982824452, 0.02418752925040368, 0.024187529129077946,
        0.010855249870939585, 0.010855249866943539, 0.007369999917743657, 0.007369999917558277,
        0.007369999917656892, 0.007369999917645042,
    ],
    [
        -0.0014672133223432022, 0.0, -0.054149468020208855, -0.13385698207177846,
        -0.028400395864362576, -0.01164959398539593, -0.10023212804377371, -0.08213110307631302,
        -0.012359690626757517, -0.011917697385837457, -0.004779706692516949, -0.004753019863297032,
        -0.004768123575568943, -0.004764580849580038, -0.010871078566507483, -0.010870134202462039,
        -0.0518353256860078, -0.05183485538003417, -0.0037629891399383122, -0.0037629859161359,
        -0.06429417209068616, -0.06429416734394935, -0.015029176374199434, -0.015029176286247902,
        -0.00590189598330461, -0.005901895980769897, -0.003756561349409417, -0.0037565613492991793,
        -0.0037565613493578216, -0.0037565613493507747,
    ],
    [
        0.0012547173591927149, 0.0, 0.04124937107251128, 0.11612545857643224,
        0.0197290911733934, 0.00712534370206163, 0.08337387801610789, 0.0664008540130313,
        0.007623836985183246, 0.0073130582736637174, 0.0025740885543675544, 0.0025576698967421773,
        0.0025669606085471945, 0.0025647810017291116, 0.00658379800353157, 0.006583144372654936,
        0.03924074957151759, 0.03924034267532737, 0.00195846978450516, 0.001958467866965978,
        0.05019341604586345, 0.05019341181077736, 0.009533090021023346, 0.009533089957265468,
        0.0032756547162530857, 0.0032756547146453024, 0.0019546469582949204, 0.001954646958229366,
        0.0019546469582642383, 0.001954646958260048,
    ],
    [
        -0.00038394207369683836, 0.0, -0.031921056932780696, -0.10225121580142306,
        -0.013922864866249149, -0.004427307514127335, -0.07045176796092027, -0.05453540169284601,
        -0.004777255233793373, -0.004558737195864994, -0.0014082655718547915, -0.0013981642605013,
        -0.0014038792271255144, -0.0014025382598793262, -0.0040505985536201305, -0.004050146151019813,
        -0.030177801643191917, -0.03017744960792805, -0.0010354750244334196, -0.001035473883869851,
        -0.039807118453869535, -0.03980711467528841, -0.006142866102718104, -0.006142866056498764,
        -0.001846900609870171, -0.0018469006088503465, -0.0010332014623692133, -0.001033201462330231,
        -0.0010332014623509681, -0.0010332014623484762,
    ],
    [
        0.0002827037666790877, 0.0, 0.025011121756144273, 0.09118778029711502,
        0.009948227797604738, 0.002785281817866662, 0.060276687094183753, 0.0453501734995369,
        0.003030950707622491, 0.002877303434983882, 0.000780083740673173, 0.000773869065342526,
        0.0007773844995066927, 0.0007765594904118269, 0.002523233801816087, 0.0025229206765019926,
        0.023498140156041746, 0.023497835584537052, 0.0005543167040867914, 0.0005543160256720171,
        0.03196467651939325, 0.03196467314810558, 0.004007781065815049, 0.004007781032309698,
        0.0010543493089439965, 0.0010543493082971161, 0.0005529645392780244, 0.0005529645392548431,
        0.0005529645392671747, 0.0005529645392656928,
    ],
    [
        -0.00010137263820203262, 0.0, -0.01979718406422197, -0.08214388683414571,
        -0.0071808728531515586, -0.0017701619534985055, -0.05209802452141801, -0.0380972643826782,
        -0.0019426461620299623, -0.001834599091817858, -0.0004365281874411595, -0.00043270426784061797,
        -0.00043486695570866776, -0.00043435932381571165, -0.0015878526087952079, -0.0015876358579188122,
        -0.018483907326711468, -0.018483643789488552, -0.00029977175207891065, -0.00029977134850721826,
        -0.02592950874032909, -0.025929505732089176, -0.002641504358922473, -0.002641504334630981,
        -0.0006080508684267848, -0.0006080508680164182, -0.00029896748175810457, -0.00029896748174431797,
        -0.0002989674817516519, -0.0002989674817507706,
    ],
    [
        6.738160060844462e-05, 0.0, 0.015799005532099734, 0.07460790339430504,
        0.005225944843309662, 0.001134260815458071, 0.045399305142179644, 0.032267454000343046,
        0.001255349356528517, 0.0011793772692269048, 0.00024628583099747987, 0.00024393321509772693,
        0.0002452635490376543, 0.0002449512353572634, 0.0010074394397430431, 0.0010072894175567756,
        0.014659195741480939, 0.014658967735307454, 0.00016344796409525076, 0.00016344772404727976,
        0.021206760546355646, 0.021206757862364544, 0.0017553137027215729, 0.0017553136851121076,
        0.0003535504361832134, 0.00035355043592291474, 0.0001629696354352194, 0.000162969635427021,
        0.00016296963543138224, 0.00016296963543085816,
    ],
    [
        -2.6949066859520235e-05, 0.0, -0.012630333377989179, -0.06788098566177579,
        -0.0038098786289970806, -0.0007280674079966239, -0.03963107840115587, -0.027377534926473508,
        -0.0008126325957015882, -0.0007594917598625781, -0.0001391955324683597, -0.00013775565476759503,
        -0.00013856972108381867, -0.00013837857500347467, -0.0006403043392784948, -0.000640201043541705,
        -0.01164622670186048, -0.011646030463671764, -8.927442440545879e-05, -8.927428236644031e-05,
        -0.017374531820176557, -0.017374529437956077, -0.0011684681766298511, -0.0011684681639308248,
        -0.0002059309276877475, -0.00020593092752349792, -8.899142661437218e-05, -8.899142660952227e-05,
        -8.899142661210224e-05, -8.899142661179221e-05,
    ],
    [
        1.6693404663508524e-05, 0.0, 0.010209497430021234, 0.06244787013623986,
        0.002808419285553297, 0.00047253582365968804, 0.034980584464680615, 0.023487049924048826,
        0.0005318980029468086, 0.000494535948673518, 7.954550680879264e-05, 7.865972395076437e-05,
        7.916043361183697e-05, 7.904284457041913e-05, 0.0004114891878735963, 0.00041141769934314245,
        0.00935545302871356, 0.00935528326396619, 4.930365199731899e-05, 4.930356751917769e-05,
        0.014393167387193113, 0.014393165261943723, 0.000786472464007222, 0.0007864724548022657,
        0.00012128197530675587, 0.00012128197520258083, 4.913535872517595e-05, 4.913535872229216e-05,
        4.913535872382623e-05, 4.913535872364188e-05,
    ],
    [
        -7.686451671931314e-06, 0.0, -0.009302039677366073, -0.06475462967432928,
        -0.002333442083551095, -0.0003456862936860922, -0.03480186405707002, -0.022711555169645223,
        -0.00039241604315347144, -0.00036295847403056315, -5.123777745608125e-05, -5.0626706315652266e-05,
        -5.0972068655238225e-05, -5.089094806218819e-05, -0.00029806754117606176, -0.00029801205904232616,
        -0.00847088291380835, -0.00847071822112345, -3.069130769179701e-05, -3.069125134829323e-05,
        -0.01343952485288583, -0.013439522726702739, -0.0005966703096691551, -0.0005966703021868423,
        -8.051099996047143e-05, -8.051099988637696e-05, -3.057907657646825e-05, -3.057907657454535e-05,
        -3.057907657556826e-05, -3.057907657544534e-05,
    ],
    [
        4.5249946299075846e-06, 0.0, 0.007598250657076195, 0.06019851917462657,
        0.0017381758462046931, 0.00022672070696556822, 0.0310412753456066, 0.019689146081614758,
        0.0002595534190394812, 0.0002388238082300469, 2.9588743163293713e-05, 2.9212487189475324e-05,
        2.9425100816918287e-05, 2.937515231239919e-05, 0.00019356754716092978, 0.00019352911480167002,
        0.006876289303275566, 0.006876146700422936, 1.712826218808773e-05, 1.712822864751636e-05,
        0.011250531981362014, 0.011250530082827377, 0.0004058326082392421, 0.00040583260281077635,
        4.791547357609965e-05, 4.791547352906314e-05, 1.7061460580993495e-05, 1.7061460579849096e-05,
        1.7061460580457875e-05, 1.706146058038472e-05,
    ],
], dtype=np.float64)


# Degree-7 Sollya fpminimax polynomial for digamma(w) on [1.0, 5.0] (the only
# part of w < 7.0 where y_lo != 0 in Euler reflection).
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
  # 13-bit fh and 40-bit fl so fh^2 (26 bits) and fh^3 (39 bits) have no
  # rounding error in float64.
  fh, fl = split_bits(f, 13)
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
  q_hh, q_hl = split_bits(qh, 13)
  q_hl = q_hl + ql
  ph = fh3 * q_hh
  pl = fh3 * q_hl + f3_l * qh
  sh, sl = f2_add_ordered(th, tl, ph, pl)
  kh = k * _LN2_HI_F64
  kl = k * _LN2_LO_F64
  # Combine k * ln(2) + ln(1 + f). |kh| >= ln(2) > 0.5 * ln(2) >= |sh| when
  # k != 0, and kh == 0 when k == 0.
  return f2_add_ordered(kh, kl, sh, sl)


def _lgamma_stirling_f64(z):
  """Evaluates Stirling's series for z >= 7 as `(res_h, res_l, digamma_z)`."""
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
  # For z >= 7, (1/z) * S(1/z^2) <= 1/84 while |lgamma| >= 0.12 everywhere this
  # path is used, so a single float64 product has <= 0.04 ULP rounding error.
  res_h, res_l = fast_two_sum(sh, sl + inv_z * sp)
  # Approximate digamma(z) = d/dz lgamma(z) to correct for rounding in z or y.
  psi = (
      lh
      - 0.5 * inv_z
      - inv_z2 * (1.0 / 12.0 - inv_z2 * (1.0 / 120.0 - inv_z2 / 252.0))
  )
  return res_h, res_l, psi


def _lgamma_shifted_f64(w):
  """Evaluates Stirling lgamma(z), shift product prod, and digamma(w) for w >= 1.

  Stirling's series needs z >= 7 for full float64 accuracy. For w in [1, 7),
  pick integer m = ceil(7 - w) in {1, ..., 6} so z = w + m >= 7, and use
  lgamma(w) = lgamma(z) - ln(prod) with prod = w * (w + 1) * ... * (w + m - 1).
  Returns lgamma(z) and prod unlogged so the caller can merge logarithms.
  """
  need0 = w < 7.0
  zero = lax.full_like(w, 0.0)
  m = lax.select(need0, lax.ceil(7.0 - w), zero)
  # Split w in [1, 7] into 24-bit wh and 29-bit wl. Because wh has steps of
  # >= 2^-23 and wh + k <= 7 for k < m, each active factor wh + k fits in 26
  # bits, so pairs of high parts multiply with zero rounding error. Factors with
  # k >= m are replaced by 1.
  w_shift = lax.min(w, lax.full_like(w, 7.0))
  wh, wl = split_bits(w_shift, 24)
  one = lax.full_like(w, 1.0)
  fh = [lax.select(m > 0.0, wh, one)]
  fl = [lax.select(m > 0.0, wl, zero)]
  for k in range(1, 6):
    need = m > float(k)
    fh.append(lax.select(need, wh + float(k), one))
    fl.append(lax.select(need, wl, zero))
  p01_h, p01_l = f2_mul_exact_high(fh[0], fl[0], fh[1], fl[1])
  p23_h, p23_l = f2_mul_exact_high(fh[2], fl[2], fh[3], fl[3])
  p45_h, p45_l = f2_mul_exact_high(fh[4], fl[4], fh[5], fl[5])
  p0123_h, p0123_l = two_prod(p01_h, p23_h)
  p0123_l = p0123_l + (p01_h * p23_l + p01_l * p23_h)
  prod_h, prod_l = f2_mul(p0123_h, p0123_l, p45_h, p45_l)
  # Adding m to w can round off up to 3 low bits of w (z_lo = w - (z - m));
  # add psi_z * z_lo to correct lgamma(z) for that rounding.
  z = w + m
  z_lo = w - (z - m)
  zh, zl, psi_z = _lgamma_stirling_f64(z)
  zl = zl + psi_z * z_lo
  # _PSI_1_5_F64 is fitted on [1, 5]; for w in (5, 7) (x in (-6, -4)),
  # y = (1 - n) - delta is exact so y_lo == 0 and psi_y * y_lo == 0.
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
  # switches from -2 to -3 at -2.5 (inside the interval around -2.457). Use
  # n_root = -2 across (-2.6, -2.0).
  n_root = lax.select(
      (x_neg_safe > -2.6) & (x_neg_safe < -2.0),
      lax.full_like(x_neg_safe, -2.0),
      n,
  )
  delta_root = x_neg_safe - n_root
  in_k_range = (x_neg_safe < -2.0) & (n_root >= -16.0)
  n_i32 = lax.convert_element_type(
      lax.max(n_root, lax.full_like(n_root, -17.0)), np.int32
  )
  base_idx = lax.select(
      delta_root > 0.0, lax.full_like(n_i32, -3), lax.full_like(n_i32, -4)
  )
  # Column 2 * (k - 2) + side; column 1 has u_bound = -1.0 since lgamma > 0 on
  # (-2, -1).
  idx = lax.clamp(
      lax.full_like(n_i32, 0),
      base_idx - (n_i32 + n_i32),
      lax.full_like(n_i32, 29),
  )

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

  # Gather one 1D row at a time so GPU kernels do not keep a 21-element float64
  # slice live per thread.
  r0 = _gather_row(0)
  neg_r1_sc = _gather_row(1)
  scale = _gather_row(2)
  u_bound = _gather_row(3)
  # When |u_h| <= u_bound, delta_root is within 17.5% of r0 and scale is a power
  # of two, so (delta_root - r0) * scale has no rounding error; adding
  # neg_r1_sc = -(r - r0) * scale gives u = (delta_root - r) * scale as
  # (u_h, u_l).
  u_h, u_l = fast_two_sum((delta_root - r0) * scale, neg_r1_sc)
  in_root_win = in_k_range & (lax.abs(u_h) <= u_bound)
  u_s = lax.select(in_root_win, u_h, lax.full_like(x_neg_safe, 0.0))
  p_tail = _gather_row(20)
  for i in range(19, 5, -1):
    p_tail = p_tail * u_s + _gather_row(i)
  pw_h, pw_l = fast_two_sum(_gather_row(4), u_s * p_tail)
  pw_l = pw_l + _gather_row(5)
  uh_s, ul_s = split_half(u_s)
  ph_s, pl_s = split_half(pw_h)
  rw = (uh_s * ph_s) + (uh_s * (pl_s + pw_l) + (ul_s + u_l) * pw_h)
  return in_root_win, rw


def _lgamma_impl_f64(x):
  """Computes float64 lgamma(x) across all normal and special inputs."""
  x = lax.convert_element_type(x, np.float64)
  # Treat subnormals as zero so backends that flush subnormals and backends that
  # preserve them return the same result (+inf).
  is_zero = lax.abs(x) < float(np.finfo(np.float64).tiny)
  is_pos = x > 0.0
  # Replace 0.0 with 1.0 so inactive branches do not divide by zero or log(0).
  x_safe = lax.select(is_zero, lax.full_like(x, 1.0), x)

  # For x <= -2^52 (_ALL_INTEGERS_BELOW_F64), every float64 is an integer.
  # Clamp inputs with x > 0 and large negative integers to -0.5 so
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

  # Both positive x and negative reflection y = 1 - x evaluate lgamma at a
  # positive argument. Clamp to _MAX_FINITE_ARG_F64 to avoid inf - inf NaNs.
  arg_pos = lax.select(is_pos, x_safe, y)
  x_bounded = lax.min(arg_pos, lax.full_like(x, _MAX_FINITE_ARG_F64))

  # Positive inputs on (0, 2.5):
  # - On [1, 2.5), evaluate (x - 1)(x - 2) * P(x - 1.75) so the zeros at x = 1
  #   and x = 2 are exact.
  # - On (0, 1), shift to x + 1 in (1, 2) via lgamma(x) = lgamma(x + 1) - ln(x),
  #   using u = x so (x + 1 - 1)(x + 1 - 2) = x * (x - 1).
  in_small = is_pos & (x_bounded < 2.5)
  x_small_in = lax.min(x_bounded, lax.full_like(x, 2.5))
  is_lt1 = x_small_in < 1.0
  u = lax.select(is_lt1, x_small_in, x_small_in - 1.0)
  v = u - 1.0
  d = u - 0.75
  ph_poly, pl_poly = fast_two_sum(
      lax.full_like(x, _POS_1_2_F64[0]),
      d * lax.polynomial(d, _POS_1_2_F64[1:], unroll=len(_POS_1_2_F64) - 1),
  )
  pl_poly = pl_poly + _POS_1_2_C0_LO_F64
  uv_h, uv_l = two_prod(u, v)
  p_h, p_l = two_prod(uv_h, ph_poly)
  p_h, p_l = fast_two_sum(p_h, p_l + (uv_h * pl_poly + uv_l * ph_poly))

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
  # prod (which can be up to ~6!) so |delta| / prod cannot underflow to
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
  dh, dl = split_bits(delta, 13)
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
  # Add digamma(y) * y_lo to account for rounding in y = (1 - n) - delta.
  dy_corr = psi_y * y_lo
  sum_h, sum_l = f2_add(zh, zl, log_h, log_l)
  refl = -f2_sum(sum_h, sum_l, sinc_h, sinc_l + dy_corr)

  # Use _NEG_TABLE_F64 near negative zeros on (-17, -2).
  in_root_win, rw = _lgamma_neg_root_f64(x_neg_safe, n)
  neg_res = lax.select(in_root_win, rw, refl)

  res = lax.select(is_pos, lg_pos, neg_res)
  # Replace -0.0 at x = 1.0 and 2.0 with +0.0.
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
