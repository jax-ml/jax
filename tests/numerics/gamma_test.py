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

"""Precision tests for gamma-related functions against reference implementations."""

import math

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import core as jax_core
from jax._src import test_util as jtu
from jax._src.lax.digamma import digamma_impl
from jax._src.lax.double_word import split_bits
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]


def _mpmath_digamma(x):
  """Evaluates digamma with pole and infinity guards for mpmath."""
  if mpmath.isnan(x):
    return mpmath.nan
  if x == 0.0:
    return mpmath.inf if math.copysign(1.0, float(x)) < 0.0 else -mpmath.inf
  if x == -mpmath.inf or (x < 0.0 and x == mpmath.floor(x)):
    return mpmath.nan
  if x == mpmath.inf:
    return mpmath.inf
  with mpmath.workprec(200):
    return +mpmath.digamma(x)


def _digamma_reference(x: np.ndarray) -> np.ndarray:
  """Evaluates float64 reference digamma, patching scipy float64 cancellation."""
  res = scipy.special.psi(x)
  # At x = float32(-95.80821990966797) (0xc2bf9dcf), cephes::psi in float64
  # loses 26.5 bits of cancellation (0.52 ULP error in float32 ULPs).
  res[x == np.float64(np.float32(-95.80821990966797))] = 5.3991989474124756e-08
  return res


def _mpmath_lgamma(x):
  if (x <= 0 and mpmath.isint(x)) or x == -mpmath.inf:
    return mpmath.inf
  with mpmath.workprec(200):
    return +mpmath.log(abs(mpmath.gamma(x)))


_DIGAMMA_INTERESTING_POINTS = [
    # Non-positive integer poles of digamma(x):
    *(float(-k) for k in range(21)),
    -100.0,
    -1000.0,
    # Roots of digamma(x) = 0, computed via mpmath.findroot(mpmath.digamma, x0):
    1.4616321449683623,  # unique positive root on (0, inf)
    -0.5040830082644554,  # root in (-1, 0) (53.61 bits cancellation)
    -1.5734984731623904,  # root in (-2, -1) (52.51 bits cancellation)
    -2.6107208684441446,  # root in (-3, -2)
    -3.635293366436901,  # root in (-4, -3)
    -4.653237761743142,  # root in (-5, -4)
    # Closest float32 bit patterns to negative zeros across all 2^32 float32s:
    -56.78992462158203,  # 0xc26328e1
    -66.79590606689453,  # 0xc2859781 (29.23 bits cancellation)
    -95.80821990966797,  # 0xc2bf9dcf (26.33 bits cancellation)
    -347.84320068359375,  # 0xc3adebeb
    -85156.9140625,  # 0xc7a65275
    # Remaining 14 float64 inputs on (-2^52, 0) with >51 bits of negative-root
    # cancellation (besides the (-1, 0) and (-2, -1) roots above), plus -2^52:
    -5.6671624415568855,  # 55.27 bits cancellation
    -14.727334416018529,  # 51.94 bits cancellation
    -87.80536027839507,  # 53.38 bits cancellation
    -1965.8749921566869,  # 51.21 bits cancellation
    -2977.8808777877402,  # 55.75 bits cancellation (max on (-2^52, 0))
    -2401203.932943261,  # 51.63 bits cancellation
    -3114755.934076784,  # 51.37 bits cancellation
    -3400673.93445092,  # 51.12 bits cancellation
    -6066919.936815942,  # 53.67 bits cancellation
    -101181205.94626413,  # 51.43 bits cancellation
    -204097275.94819948,  # 54.87 bits cancellation
    -216428973.94835505,  # 52.21 bits cancellation
    -71243556817555.97,  # 52.89 bits cancellation (-71243556817556 + 1/32)
    -71243556817556.97,  # 51.47 bits cancellation (-71243556817557 + 1/32)
    -4503599627370495.5,  # -2**52 + 0.5 (last negative non-integer in f64)
    -4503599627370496.0,  # -2**52 (all f64 below this are even integers)
    # Reflection, positive-root Taylor window, and recurrence switch boundaries:
    0.5,
    1.4615321449683622,
    1.4617321449683622,
    4.5,
    6.5,
    8.5,
    10.0,
    12.5,
]


@jtu.thread_unsafe_test_class()
class DigammaTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_digamma_accuracy(self, dtype):
    bounds = [
        (["cpu", "gpu", "tpu"], {f16: 1.0, f32: 1.0, f64: 1.0}),
    ]
    util.check_unary_precision(
        self,
        lax.digamma,
        _digamma_reference,
        _mpmath_digamma,
        dtype,
        bounds=bounds,
        interesting_points=_DIGAMMA_INTERESTING_POINTS,
    )

  @config.explicit_x64_dtypes("allow")
  def test_no_intermediate_nonfinite_on_poles_and_extremes(self):
    if jtu.device_under_test() == "tpu":
      dtypes_and_modes = [(jnp.float32, False)]
    else:
      dtypes_and_modes = [
          (jnp.float32, True),
          (jnp.float32, False),
          (jnp.float64, False),
      ]
    for dtype, use_f64_for_f32 in dtypes_and_modes:
      finfo = jnp.finfo(dtype)
      x = jnp.array(
          [-2.0, -1.0, -0.0, 0.0, 0.5, 1.5, finfo.tiny, -finfo.tiny, finfo.max],
          dtype=dtype,
      )
      with jax.disable_jit():
        jaxpr = jax.make_jaxpr(
            lambda v: digamma_impl(v, use_f64_for_f32=use_f64_for_f32)
        )(x)
      env = dict(zip(jaxpr.jaxpr.constvars, jaxpr.consts))
      env[jaxpr.jaxpr.invars[0]] = x
      for eqn in jaxpr.jaxpr.eqns:
        invals = [
            v.val if isinstance(v, jax_core.Literal) else env[v]
            for v in eqn.invars
        ]
        ans = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
          ans = [ans]
        if eqn.primitive in (lax.div_p, lax.mul_p):
          for out in ans:
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(out))),
                f"Non-finite {eqn.primitive.name} output {out} for"
                f" dtype={dtype}, use_f64_for_f32={use_f64_for_f32}",
            )
        for v, out in zip(eqn.outvars, ans):
          env[v] = out

  @config.explicit_x64_dtypes("allow")
  def test_explicit_x64_dtypes(self):
    if jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    fns = [lax.digamma, jsp.special.digamma]
    pts = [
        -2977.8808777877402,
        -87.80536027839507,
        -5.6671624415568855,
        -1.5,
        0.5,
        1.4616321449683622,
        12.5,
    ]
    with jax.enable_x64(True):
      x_ref = jnp.array(pts, dtype=jnp.float64)
      expected = [jax.jit(fn)(x_ref) for fn in fns]
    with jax.enable_x64(False):
      x = jnp.array(pts, dtype=jnp.float64)
      actual = [jax.jit(fn)(x) for fn in fns]
    for exp_val, act_val in zip(expected, actual):
      self.assertArraysEqual(act_val, exp_val)


@jtu.thread_unsafe_test_class()
class LgammaTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_lgamma_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("tpu", {f16: 1.0, f32: 1.0}),
    ]
    ref_fn = lambda x: np.where(x == -np.inf, np.inf, scipy.special.gammaln(x))
    neg_roots = [
        # Negative zero crossings where |gamma(x)| = 1 (lgamma(x) = 0), computed
        # via mpmath.findroot(lambda x: abs(mpmath.gamma(x)) - 1, x0):
        -2.4570247382208006,
        -2.7476826467274127,
        -3.14358088834998,
        -3.955294284858598,
        -4.039361839740537,
        -4.991544640560048,
        -5.0082181683225935,
        -5.998607480080875,
        -6.001385294453155,
        -6.999801507890638,
        -7.000198333407325,
        -7.999975197095821,
        -8.000024800270682,
        -8.999997244250977,
        -9.000002755714823,
        -9.99999972442663,
        -10.000000275573013,
        -10.99999997494789,
        -11.000000025052106,
        -11.999999997912324,
        -12.000000002087676,
        -12.99999999983941,
        -13.00000000016059,
        -13.99999999998853,
        -14.00000000001147,
        -14.999999999999236,
        -15.000000000000764,
        -15.999999999999952,
        -16.000000000000046,
    ]
    # Interval endpoints around negative zeros where lgamma switches between its
    # lookup table and Euler's reflection formula in float32 and float64:
    f32_neg_root_boundaries = [
        -2.33435985,
        -2.5,
        -2.70703125,
        -2.83335972,
        -3.09179749,
        -3.21762902,
        -3.93363558,
        -3.97260000,
        -4.02433043,
        -4.06287674,
        -4.98589942,
        -4.99490073,
        -5.00539967,
        -5.01300052,
        -5.99819946,
        -5.99895049,
        -6.00104951,
        -6.00180054,
        *(
            round(x) + (x - round(x)) * factor
            for x in neg_roots[9:15]
            for factor in (19.0 / 31.0, 121.0 / 79.0)
        ),
    ]
    f64_neg_root_boundaries = [
        round(x) + (x - round(x)) * factor
        for i, x in enumerate(neg_roots)
        for factor in ((0.872, 1.128) if i == 0 else (0.825, 1.175))
    ]
    neg_root_boundaries = [*f32_neg_root_boundaries, *f64_neg_root_boundaries]
    interesting_points = [
        # Zero and negative integers (where gamma(x) has poles), plus large
        # negative half-integers:
        *(float(-k) for k in range(21)),
        -100.0,
        -1000.0,
        -8388607.5,  # -2**23 + 0.5 (last negative non-integer in f32)
        -8388608.0,  # -2**23 (all f32 below this are integers)
        -4503599627370495.5,  # -2**52 + 0.5 (last negative non-integer in f64)
        -4503599627370496.0,  # -2**52 (all f64 below this are even integers)
        # Subnormal inputs (treated as poles returning +inf):
        float(np.finfo(np.float32).tiny) / 2.0,
        -float(np.finfo(np.float32).tiny) / 2.0,
        float(np.finfo(np.float64).tiny) / 2.0,
        -float(np.finfo(np.float64).tiny) / 2.0,
        # Zeros at x = 1 and x = 2, plus points just beside them and at the
        # CPU float32 Taylor window boundaries (|x - 1| <= 3/64, |x - 2| <= 3/64):
        1.0,
        2.0,
        1.0 - 2**-26,
        1.0 + 2**-26,
        2.0 - 2**-26,
        2.0 + 2**-26,
        1.0 - 3.0 / 64.0,
        1.0 + 3.0 / 64.0,
        2.0 - 3.0 / 64.0,
        2.0 + 3.0 / 64.0,
        # Float32 inputs near float64->float32 midpoint ties or the -2.4570247 zero:
        0.0038502016104757786,
        1.0886335372924805,
        1.6304548978805542,
        6.990847948130907e33,
        -1.5639554096225083e-09,
        -2.457024574279785,
        -2.744002342224121,
        -2.9687294960021973,
        -3.143528461456299,
        # Local minimum of gamma(x) on (0, inf), where digamma(x) = 0
        # (computed via mpmath.findroot(mpmath.digamma, 1.5)):
        1.4616321449683623,
        *neg_roots,
        *neg_root_boundaries,
        # Local minimum on (-3, -2) and nearby negative test points:
        -2.70703125,
        -2.634361,
        -2.6107208684441446,
        -2.5234375,
        -3.7606792,
        -6.00337,
        # Interval boundaries between polynomial approximations and Stirling's
        # series (including immediate float32 neighbors of 1.5 and 4.0):
        0.5,
        0.75,
        float(np.nextafter(np.float32(1.5), np.float32(0.0))),
        1.5,
        float(np.nextafter(np.float32(1.5), np.float32(2.0))),
        1.75,
        2.5,
        2.75,
        3.0,
        float(np.nextafter(np.float32(4.0), np.float32(0.0))),
        4.0,
        float(np.nextafter(np.float32(4.0), np.float32(5.0))),
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
        8388607.5,  # 2**23 - 0.5 (largest positive non-integer in f32)
        # Overflow thresholds where lgamma(x) == finfo.max, computed via
        # mpmath.findroot(lambda x: mpmath.loggamma(x) - finfo.max, x0), plus
        # values just below the float32 and float64 overflow boundaries:
        2.556348e36,
        4.085003e36,  # largest float32 with finite lgamma
        2.556348163871691e305,  # finite in float64, just above 2.556348e305
        2.559983327851638e305,  # float64 overflow threshold
        # Worst-case inputs across math libraries from Gladman et al.,
        # "Accuracy of Mathematical Functions in Single, Double, Double
        # Extended, and Quadruple Precision":
        *(
            float.fromhex(h)
            for h in (
                "0x0.00000002f6806p-1022",
                "-0x1.3f62c60e23b31p+2",
                "-0x1.3a7fc9600f86cp+1",
                "-0x1.bffcbf76b86fp+2",
                "-0x1.bffe071eea741p+2",
                "-0x1.fa471547c2fe5p+1",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        lax.lgamma,
        ref_fn,
        _mpmath_lgamma,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )

  @config.explicit_x64_dtypes("allow")
  def test_explicit_x64_dtypes(self):
    if jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    fns = [
        lax.lgamma,
        jsp.special.gammaln,
        jax.vmap(jax.grad(lax.lgamma)),
    ]
    pts = [-3.14358088834998, -2.457024738220801, -1.5, 0.5, 1.5, 2.5, 10.0]
    with jax.enable_x64(True):
      x_ref = jnp.array(pts, dtype=jnp.float64)
      expected = [jax.jit(fn)(x_ref) for fn in fns]
    with jax.enable_x64(False):
      x = jnp.array(pts, dtype=jnp.float64)
      actual = [jax.jit(fn)(x) for fn in fns]
    for exp_val, act_val in zip(expected, actual):
      self.assertArraysEqual(act_val, exp_val)

  def test_split_bits(self):
    x32 = jnp.array(math.pi, dtype=jnp.float32)
    ah, al = jax.jit(lambda v: split_bits(v, 8))(x32)
    self.assertArraysEqual(ah + al, x32)
    self.assertEqual(
        int(lax.bitcast_convert_type(ah, np.uint32)) & ((1 << 16) - 1), 0
    )
    for bad_dtype in (jnp.float16, jnp.bfloat16):
      with self.assertRaisesRegex(ValueError, "float32 or float64"):
        jax.jit(lambda v: split_bits(v, 4))(jnp.ones((), dtype=bad_dtype))
    for bad_bits in (0, -1, 25):
      with self.assertRaisesRegex(ValueError, "keep_bits"):
        jax.jit(lambda v: split_bits(v, bad_bits))(x32)
    if jtu.device_under_test() != "tpu":
      with jax.enable_x64(True):
        x64 = jnp.array(math.pi, dtype=jnp.float64)
        ah64, al64 = jax.jit(lambda v: split_bits(v, 13))(x64)
        self.assertArraysEqual(ah64 + al64, x64)
        self.assertEqual(
            int(lax.bitcast_convert_type(ah64, np.uint64)) & ((1 << 40) - 1), 0
        )
        for bad_bits in (0, 54):
          with self.assertRaisesRegex(ValueError, "keep_bits"):
            jax.jit(lambda v: split_bits(v, bad_bits))(x64)


def _with_special_values(a, y, x, x_at_0, x_at_1):
  """Sets the documented special values of the JAX version in scipy's `x`.

  The inverse is `x_at_0` at y = 0 and `x_at_1` at y = 1 for any a >= 0, and
  nan for y outside [0, 1], for a < 0, and for a in {0, inf} with 0 < y < 1.
  scipy is inconsistent for a = 0 and for some y < 0, for example
  `scipy.special.gammaincinv(0.001, -6e-8) == 0`.
  """
  x = np.where(y == 0, x_at_0, np.where(y == 1, x_at_1, x))
  interior = (y > 0) & (y < 1)
  invalid = (
      np.isnan(a) | np.isnan(y) | (a < 0) | (y < 0) | (y > 1)
      | (((a == 0) | np.isinf(a)) & interior)
  )
  return np.where(invalid, np.nan, x)


def _gammaincinv_ref(a, y):
  x = scipy.special.gammaincinv(a, y)
  return _with_special_values(a, y, x, x_at_0=0.0, x_at_1=np.inf)


def _gammainccinv_ref(a, y):
  x = scipy.special.gammainccinv(a, y)
  return _with_special_values(a, y, x, x_at_0=np.inf, x_at_1=0.0)


def _ignored_inputs(a, y, x):
  """Inputs (a, y) with reference result x excluded from the accuracy checks."""
  finfo = jnp.finfo(a.dtype)
  tiny = float(finfo.tiny)
  # Whether subnormal a is flushed to zero, which changes the result from 0
  # (it underflows for all but subnormal y) to nan, depends on the platform.
  # So does whether a negative subnormal y compares as zero (giving the result
  # at y = 0) or as negative (giving nan): GPUs flush subnormals in arithmetic
  # but not in comparisons.
  ignored = (np.abs(a) > 0) & (np.abs(a) < tiny)
  ignored |= (np.abs(y) > 0) & (np.abs(y) < tiny)
  # Within a factor 1 / eps of the smallest normal number, the density of the
  # gamma distribution at the result underflows.
  near_underflow = tiny / float(finfo.eps)
  for v in (a, y, x):
    ignored |= (v > 0) & (v < near_underflow)
  if a.dtype == np.float64:
    # For finite a >= 1e6 in float64, the initial approximation is used without
    # refinement (relative error up to about 1e-6), and scipy's relative error
    # also grows beyond 1e-7 (checked against mpmath for a >= 1e7).
    ignored |= (a >= 1e6) & np.isfinite(a)
  return ignored


def _gammaincinv_ignored(a, y):
  with np.errstate(all="ignore"):
    x = scipy.special.gammaincinv(a.astype(np.float64), y.astype(np.float64))
  return _ignored_inputs(a, y, x)


def _gammainccinv_ignored(a, y):
  with np.errstate(all="ignore"):
    x = scipy.special.gammainccinv(a.astype(np.float64), y.astype(np.float64))
  return _ignored_inputs(a, y, x)


# (a, y) inputs at the special values set by _with_special_values.
_GAMMAINCINV_SPECIAL_POINTS = [
    (2.0, 0.0),
    (2.0, 1.0),
    (0.0, 0.0),
    (0.0, 1.0),
    (math.inf, 0.0),
    (math.inf, 1.0),
    (0.0, 0.5),
    (math.inf, 0.5),
    (-1.0, 0.5),
    (2.0, -0.1),
    (2.0, 1.1),
    (2.0, -math.inf),
    (2.0, math.inf),
    (2.0, math.nan),
    (math.nan, 0.5),
    (math.nan, 0.0),
]

_GAMMAINCINV_INTERESTING_POINTS = [
    *_GAMMAINCINV_SPECIAL_POINTS,
    # Thresholds in a of the initial approximation (a < 0.3, a < 1, a >= 500)
    # and of the Halley refinement (a < 200 in float32, a < 1e6 in float64):
    0.3,
    200.0,
    500.0,
    1e6,
    # Values of y near thresholds of the initial approximation for a < 1,
    # which are on b = (1 - p) * Gamma(a), close to 1 - p for a near 1:
    1e-5,
    0.15,
    0.35,
    0.6,
]


@jtu.thread_unsafe_test_class()
class GammaincinvTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gammaincinv_accuracy(self, dtype):
    # mpmath has no inverse incomplete gamma function, so scipy is the
    # reference for all dtypes. In float32 the accuracy for small a is limited
    # by that of gammainc. The result is refined against lax.igamma, so on GPU
    # and TPU it inherits the accuracy of igamma there, amplified by about 1 / a
    # for small a.
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 600.0, f64: 14500.0}),
        (["gpu", "tpu"],
         {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        jsp.special.gammaincinv,
        _gammaincinv_ref,
        mpmath_fn=None,
        dtype=dtype,
        nargs=2,
        bounds=bounds,
        ignore_inputs=_gammaincinv_ignored,
        interesting_points=_GAMMAINCINV_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class GammainccinvTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gammainccinv_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 1250.0, f64: 16500.0}),
        (["gpu", "tpu"],
         {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        jsp.special.gammainccinv,
        _gammainccinv_ref,
        mpmath_fn=None,
        dtype=dtype,
        nargs=2,
        bounds=bounds,
        ignore_inputs=_gammainccinv_ignored,
        interesting_points=_GAMMAINCINV_INTERESTING_POINTS,
    )


def _scipy_gamma(x):
  # scipy.special.gamma returns +0.0 when it underflows for negative x.
  r = scipy.special.gamma(x)
  return np.where(r == 0, np.where(np.floor(x) % 2 == 1, -0.0, 0.0), r)


def _mpmath_gamma(x):
  if x == 0:
    return -mpmath.inf if np.signbit(float(x)) else mpmath.inf
  if (x < 0 and mpmath.isint(x)) or x == -mpmath.inf:
    return mpmath.nan
  return mpmath.gamma(x)


@jtu.thread_unsafe_test_class()
class GammaTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gamma_accuracy(self, dtype):
    bounds = [
        (["cpu", "gpu", "tpu"], {f16: 1.0, f32: 1.0, f64: 1.0}),
    ]
    interesting_points = [
        # Non-positive integer poles of gamma(x):
        *(float(-k) for k in range(21)),
        -100.0,
        -1000.0,
        # Unconditional-shift boundaries and integer factorials:
        0.5,
        4.5,
        8.5,
        10.0,
        # Largest n with finite (n - 1)! in float32 and float64:
        35.0,
        171.0,
        # Overflow thresholds where gamma(x) == finfo.max, computed via
        # mpmath.findroot(lambda x: mpmath.loggamma(x) - log(finfo.max), x0):
        35.040098403112816,  # float32
        171.6243769563027,  # float64
        # Negative inputs where |gamma(x)| lands just below a power of two,
        # maximizing ULP error in the reflection formula in float64:
        -6.2469100857178725,  # gamma(x) ~= -2**-8
        -13.999975942555134,  # gamma(x) ~= +2**-21
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "0x1.8b9710d0adf83p-702",
                "-0x1.3e0001ad3bee3p+6",
                "-0x1.535175475cc8dp+7",
                "-0x1.540b170c4e65ep+7",
                "-0x1.fc4b534c8eccp+2",
                "-0x1.5456e56919a19p+7",
                "-0x1.5c00000003c15p+7",
                "-0x1.547cf3ddaaddap+7",
                "-0x1.2baa17692a3f2p+7",
                "-0x1.e7b39f8d17722p+3",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jsp.special.gamma,
        _scipy_gamma,
        _mpmath_gamma,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )

  def test_factorial_exact(self):
    n32 = jnp.arange(35, dtype=jnp.float32)
    ref32 = np.array([math.factorial(i) for i in range(35)], dtype=np.float32)
    self.assertArraysEqual(jsp.special.factorial(n32), ref32)
    self.assertArraysEqual(jsp.special.gamma(n32 + 1), ref32)
    if jtu.device_under_test() != "tpu":
      with jax.enable_x64(True):
        n64 = jnp.arange(171, dtype=jnp.float64)
        ref64 = np.array(
            [math.factorial(i) for i in range(171)], dtype=np.float64
        )
        self.assertArraysEqual(jsp.special.factorial(n64), ref64)
        self.assertArraysEqual(jsp.special.gamma(n64 + 1), ref64)

  @config.explicit_x64_dtypes("allow")
  def test_explicit_x64_dtypes(self):
    if jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    fns = [
        jsp.special.gamma,
        jsp.special.factorial,
        jax.vmap(jax.grad(jsp.special.gamma)),
    ]
    pts = [-6.2469100857178725, -1.5, 0.5, 1.5, 10.0, 35.0, 171.0]
    with jax.enable_x64(True):
      x_ref = jnp.array(pts, dtype=jnp.float64)
      expected = [jax.jit(fn)(x_ref) for fn in fns]
    with jax.enable_x64(False):
      x = jnp.array(pts, dtype=jnp.float64)
      actual = [jax.jit(fn)(x) for fn in fns]
    for exp_val, act_val in zip(expected, actual):
      self.assertArraysEqual(act_val, exp_val)


util.register_benchmark(lax.digamma)
util.register_benchmark(lax.lgamma)
util.register_benchmark(jsp.special.gammaincinv, nargs=2)
util.register_benchmark(jsp.special.gammainccinv, nargs=2)
util.register_benchmark(jsp.special.gamma, name="gamma")


if __name__ == "__main__":
  util.main()
