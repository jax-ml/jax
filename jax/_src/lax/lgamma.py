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

"""Float32 implementation of lgamma(x).

Approach overview:
- Arguments x >= 0.5 are evaluated directly on the positive half-line:
  1. x in [0.5, 4]: evaluated via factored minimax polynomials (x - 1)(x - 2)P(d)
     that vanish identically at the roots x = 1 and x = 2, split across [0.5, 1.5]
     (center d = x - 1.0) and (1.5, 4] (center d = x - 2.75). 12-bit mantissa
     truncation (`& -0x1000`) splits the factors and polynomial value into
     high/low parts so the product retains precision under reflection.
  2. x > 4: evaluated via the asymptotic Stirling series
     lgamma(x) = (x - 0.5)(ln(x) - 1) + (0.5 ln(2 pi) - 0.5) + (1/x) S(1/x^2),
     using 12-bit splits on (x - 0.5) and (ln(x) - 1) so the leading product
     p_hi = b_hi * (h_x - 1) is exact in float32.
- Arguments x < 0.5 (including (0, 0.5), subnormals, and all negative inputs)
  reflect to y = 1 - x > 0.5 via two complementary paths:
  1. Euler reflection: evaluates
     lgamma(x) = -(ln|delta| + lgamma(1 - x) - ln(pi delta / sin(pi delta))),
     where n = round(x), delta = x - n in [-0.5, 0.5], y = (1 - n) - delta, and
     a first-order Taylor correction psi(y) * y_lo compensates for rounding in y.
  2. Root windows near negative zeros: near zeros of lgamma(x) on (-10, -2),
     where Euler reflection suffers catastrophic cancellation, each root
     x_root = -k + r (k in {2, ..., 9}, side in {0, 1}) lies close to the
     logarithmic pole at x = -k (delta = 0). A direct polynomial in delta - r
     converges slowly because of that nearby pole; instead, we apply the
     rational change of variable (a Cayley transform)
     w = (delta - r) / (delta + r), which maps the root delta = r to w = 0 and
     turns the pole term -ln(delta / r) = -ln((1 + w) / (1 - w)) = -2 * atanh(w)
     into a rapidly converging odd power series. Each window is evaluated as
     w * P(w) using a 14 x 16 lookup table of degree-10 ULP-weighted minimax
     polynomials.
"""

from __future__ import annotations

from jax._src.api import jit
from jax._src.lax import lax
from jax._src.lax import slicing
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


def _split12(x):
  """Splits float32 `x` into `(x_hi, x_lo)` where `x_hi` has 12 mantissa bits.

  Because 24 = 12 + 12, products `x_hi * y_hi` are exact in float32.
  """
  x_hi = lax.bitcast_convert_type(
      lax.bitcast_convert_type(x, np.int32) & -0x1000, np.float32
  )
  return x_hi, x - x_hi


@jit
def _horner(x, coeffs):
  """Evaluates polynomial sum_{i=0}^{n} coeffs[i] * x^i via Horner's rule."""
  p = (
      lax.full_like(x, coeffs[-1])
      if isinstance(coeffs[-1], (int, float))
      else coeffs[-1]
  )
  for c in reversed(coeffs[:-1]):
    p = p * x + c
  return p


# Degree-9 minimax polynomial for (ln(1 + f) - (f - 0.5 f^2)) / f^3 on
# f = m - 1 in [sqrt(2)/2 - 1, sqrt(2) - 1], stored in ascending powers of f.
_LOG_Q_HI = [
    0.3333333432674408, -0.24999989569187164, 0.19999992847442627,
    -0.16668006777763367, 0.14287430047988892, -0.12451331317424774,
    0.11021918058395386, -0.10625848919153214, 0.10618706792593002,
    -0.06354549527168274,
]


@jit
def _log_f32(x, sub_shift=0):
  """Computes ln(x * 2^-sub_shift) as a compensated (log_hi, log_lo) pair."""
  bits = lax.bitcast_convert_type(x, np.int32)
  exp_off = lax.shift_right_arithmetic(
      bits - _SQRT2_OVER_2_BITS, lax.full_like(bits, 23)
  )
  k = lax.convert_element_type(exp_off - sub_shift, np.float32)
  mant_bits = bits - lax.shift_left(exp_off, lax.full_like(bits, 23))
  m = lax.bitcast_convert_type(mant_bits, np.float32)

  # TODO(phawkins): To reduce worst-case error near x in (0.7, 1.0) and x ~ -5,
  # split f with _split12(f) and use 2Sum on (f, -0.5 * f_hi^2) and (h_d, q_hi).
  f = m - 1.0
  f2 = f * f
  ln1pf = f - 0.5 * f2 + (f2 * f) * _horner(f, _LOG_Q_HI)
  h_d = k * _LN2_HI
  l_d = ln1pf + k * _LN2_LO
  log_hi = h_d + l_d
  log_lo = (h_d - log_hi) + l_d
  return log_hi, log_lo


# Degree-12 Sollya fpminimax polynomials for lgamma(x) / ((x - 1)(x - 2)) in
# ascending powers of d = x - center:
# - _POS_HI_05_15: fitted on x in [0.5, 1.5] around center = 1.0.
# - _POS_HI_15_4: fitted on x in [1.5, 4.0] around center = 2.75.
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

# Minimax coefficients in ascending powers of 1/x^2 for the Stirling remainder
# S(1/x^2) = B_2/(1*2) + B_4/(3*4) x^-2 + ... on x >= 4.
_STIRLING_HI = [
    0.0833333358168602, -0.0027777778450399637, 0.0007936508045531809,
    -0.0005952381179668009, 0.0008417508215643466,
]

# Minimax coefficients in ascending powers of delta^2 for
# -ln(pi * delta / sin(pi * delta)) / delta^2 on |delta| <= 0.5.
_SINC_HI = [
    -1.644934058189392, -0.5411607623100281, -0.3391686975955963,
    -0.24974527955055237, -0.21417342126369476, -0.09055595844984055,
    -0.33258557319641113,
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


@jit
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
  p = _horner(d, coeffs_hi)
  u_hi, u_lo = _split12(u)
  v_hi, v_lo = _split12(v)
  uv_hi = u_hi * v_hi
  uv_lo = u_lo * v + u_hi * v_lo
  p_hi, p_lo = _split12(p)
  res_hi = uv_hi * p_hi
  res_lo = uv_lo * p + uv_hi * p_lo
  return res_hi + res_lo, res_hi, res_lo


@jit
def _lgamma_pos_large(x_gt4):
  """Evaluates the asymptotic Stirling series for x >= 4.

  Returns (p_hi, s_pos, psi_large), where p_hi + s_pos approximates lgamma(x)
  with an exact 12-bit leading product p_hi = b_hi * (h_x - 1), and
  psi_large = ln(x) - 1/(2x) approximates the digamma function psi(x).
  """
  s, log_lo = _log_f32(x_gt4)
  h_x, s_lo = _split12(s)
  l_x = s_lo + log_lo

  b = x_gt4 - 0.5
  b_hi, b_lo = _split12(b)
  hx_m1 = h_x - 1.0
  p_hi = b_hi * hx_m1
  p_lo = b_lo * hx_m1 + b * l_x
  inv_x = lax.reciprocal(x_gt4)
  sp = _horner(inv_x * inv_x, _STIRLING_HI)
  s_pos = p_lo + _HALF_LN_2PI_MINUS_HALF + inv_x * sp
  return p_hi, s_pos, s - 0.5 * inv_x


@jit
def _lgamma_neg_reflection(delta, log_delta_hi, lg_pos_hi, lg_pos_lo, dy_corr):
  """Evaluates Euler's reflection formula -(ln|delta| + lgamma(1 - x) - ln(pi delta / sin(pi delta)))."""
  d2 = delta * delta
  sinc_term = d2 * _horner(d2, _SINC_HI)
  # TODO(phawkins): Using 2Sum on (log_delta_hi, lg_pos_hi) avoids cancellation
  # loss when log_delta_hi + lg_pos_hi cancels outside root windows.
  return -((log_delta_hi + lg_pos_hi) + (lg_pos_lo + (sinc_term + dy_corr)))


@jit
def _lgamma_neg_root(x_neg_safe, n):
  """Evaluates lgamma(x) near negative roots via rational root-window polynomials.

  The offset delta_root = x - n_root is mapped to the rational variable
  w = (delta_root - r) / (delta_root + r) using one residual refinement step on
  w. Returns (in_root_win, w * P(w)).
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
      slicing.gather(
          lax._const(x_neg_safe, row), idx_exp, dnums, slice_sizes=(1,)
      )
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
  return in_root_win, w_safe * _horner(w_safe, cols[3:])


@jit
def lgamma_impl(x):
  """Computes float32 lgamma(x) across all normal, subnormal, and special inputs."""
  # Normalize subnormals in integer bitwise arithmetic before any float ops so
  # execution is exact even on hardware that flushes subnormals to zero (FTZ).
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

  is_pos = x_norm >= 0.5
  # Substitute 1.0 for 0.0 so inactive branches avoid 0/0 or log(0) NaNs.
  x_safe = lax.select(is_zero, lax.full_like(x, 1.0), x_norm)

  # For x <= -2^23 (_ALL_INTEGERS_BELOW), every float32 is an exact integer
  # pole. Clamp inputs with x >= 0.5 and large negative integers to -0.5 so
  # round(x_neg_safe) cannot overflow and delta = x - n stays away from zero.
  x_neg_safe = lax.select(
      (~is_pos) & (x_safe > _ALL_INTEGERS_BELOW),
      x_safe,
      lax.full_like(x, -0.5),
  )
  # Decompose x < 0.5 as x = n + delta with n = round(x) and delta in [-0.5, 0.5]
  # (exact by Sterbenz's lemma). Form the reflected argument y = 1 - x =
  # (1 - n) - delta as a Fast2Sum pair (y, y_lo) because (1 - n) can be much
  # larger than |delta|, losing low bits of delta in y.
  n = lax.round(x_neg_safe, lax.RoundingMethod.TO_NEAREST_EVEN)
  delta = x_neg_safe - n
  abs_delta = lax.abs(delta)
  y = (1.0 - n) - delta
  y_lo = ((1.0 - n) - y) - delta

  # Both x >= 0.5 (at x) and x < 0.5 (via reflection at y = 1 - x > 0.5) evaluate
  # lgamma at an argument >= 0.5. Clamp to _MAX_FINITE_ARG to prevent inf - inf
  # NaNs in Stirling's series.
  arg_pos = lax.select(is_pos, x_safe, y)
  x_bounded = lax.min(arg_pos, lax.full_like(x, _MAX_FINITE_ARG))

  # Compute ln|delta| for Euler's reflection formula on x < 0.5, applying
  # sub_shift when x is a positive or negative subnormal (where n = 0 and
  # |delta| = |x|).
  log_hi, log_lo = _log_f32(abs_delta, sub_shift)

  # Evaluate lgamma on [0.5, 4] and [4, inf), keeping both the summed result
  # lg_pos (for x >= 0.5) and the unsummed (lg_pos_hi, lg_pos_lo) pair so Euler
  # reflection can cancel leading terms before adding low-order corrections.
  lg_small, lg_small_hi, lg_small_lo = _lgamma_pos_small(x_bounded)
  p_hi, s_pos, psi = _lgamma_pos_large(x_bounded)
  in_le_four = x_bounded <= 4.0
  lg_pos = lax.select(in_le_four, lg_small, p_hi + s_pos)
  lg_pos_hi = lax.select(in_le_four, lg_small_hi, p_hi)
  lg_pos_lo = lax.select(in_le_four, lg_small_lo, s_pos)

  # Combine Euler reflection (with first-order Taylor correction psi(y) * y_lo
  # for rounding in y) and rational root-window polynomials near negative zeros
  # on (-10, -2) where reflection suffers catastrophic cancellation.
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
