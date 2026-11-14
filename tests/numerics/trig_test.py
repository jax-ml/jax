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

"""Precision tests for trigonometric and inverse trigonometric functions."""

import math

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version
import jax.numpy as jnp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]

# Range-reduction breakpoint sin(pi/3) = cos(pi/6) = sqrt(3)/2 in [-1, 1]
# (0.25, 0.5, sqrt(0.5), and 1 - 2^-k are already in _common_interesting_points).
_ACOS_ASIN_INTERESTING_POINTS = [-math.sqrt(3.0) / 2.0, math.sqrt(3.0) / 2.0]

_ACOS_INTERESTING_POINTS = [
    *_ACOS_ASIN_INTERESTING_POINTS,
    # Worst-case inputs across math libraries from Gladman et al.:
    *(
        float.fromhex(h)
        for h in (
            "0x1.dffffb3488a4p-1",
            "0x1.6c05eb219ec46p-1",
            "0x1.35b03e336a82bp-1",
            "-0x1.0068b067c6feep-1",
            "-0x1.8d313198a2e03p-53",
            "-0x1.cb3b399d747f3p-55",
            "-0x1.01f97a9d82a5dp-1",
            "0x1.251869c3f7881p-1",
            "0x1.266637a3d2bbcp-1",
            "-0x1.3702dc3201849p-1",
        )
    ),
]

_ASIN_INTERESTING_POINTS = [
    *_ACOS_ASIN_INTERESTING_POINTS,
    # Worst-case inputs across math libraries from Gladman et al.:
    *(
        float.fromhex(h)
        for h in (
            "-0x1.0000045b2c904p-3",
            "0x1.6c042a6378102p-1",
            "-0x1.00d44cccfa99p-1",
            "-0x1.004d1c5a9400bp-1",
            "0x1.eaeca101513d2p-2",
            "-0x1.7137449123ef6p-26",
            "0x1.003d8c089658bp-1",
            "0x1.0479b37d95e5cp-1",
            "-0x1.2ef2481799c7cp-1",
            "0x1.df2c6032adcd8p-2",
        )
    ),
]

_ATAN_INTERESTING_POINTS = [
    # Octant and sixteenth-circle range-reduction thresholds
    # (tan(pi/8) = sqrt(2) - 1, tan(3*pi/8) = sqrt(2) + 1, tan(pi/6)).
    *(
        sign * v
        for v in (
            math.sqrt(2.0) - 1.0,
            1.0 / math.sqrt(3.0),
            math.sqrt(2.0) + 1.0,
        )
        for sign in (-1, 1)
    ),
    # Worst-case inputs across math libraries from Gladman et al.:
    *(
        float.fromhex(h)
        for h in (
            "0x1.f9004c4fef9eap-4",
            "-0x1.ffff8020d3d1dp-7",
            "-0x1.60370d15396b7p-1",
            "0x1.62ff6a1682c25p-1",
            "0x1.01e0be37af68fp+1",
            "-0x1.085b3fdb46cbp+0",
            "0x1.032b4811f3dc5p+0",
            "0x1.52184b1b9bd9bp+0",
            "0x1.00bb805435ccfp+0",
        )
    ),
]

_ATAN2_INTERESTING_POINTS = [
    (float.fromhex(y), float.fromhex(x))
    for y, x in (
        ("0x1.960176p-63", "0x1.f88dp+80"),
        ("-0x1.58a7ecp-118", "0x1.58a7bep-123"),
        ("0x1.fffe24p+59", "0x1.000adcp+73"),
        ("-0x1.f9cf48p+49", "0x1.f60598p+51"),
        ("0x1.a10104p+123", "0x1.99f182p+125"),
        ("-0x1.ce62cep-116", "0x1.cbf9bp-113"),
        ("-0x1.9be82ap-101", "0x1.92cfb2p-101"),
        ("-0x1.eda3fep-120", "0x1.e3bfa8p-118"),
        ("-0x1.00f6f2p+44", "0x1.ffe934p+46"),
        ("0x1.ed6060626eecfp-429", "0x1.f42ebb62994dcp-426"),
        ("0x1.b77ade79a36d5p-326", "0x1.ff6a37b72b52bp-319"),
        ("-0x1.ec2003fe89e36p-433", "0x1.ec173ed5fd8f5p-427"),
        ("-0x1.358bb5eb25bdcp+813", "0x1.2f86b82481a0ap+815"),
        ("-0x1.6a539153430d8p-416", "0x1.d2b5b9dc716d8p-415"),
        ("-0x1.bdb2c358b0b81p-93", "0x1.c717a5969be93p-86"),
        ("0x1.fffffffffffffp-627", "0x1.0000000000000p+448"),
        ("0x1.d5de7a294d493p-935", "0x1.d030d7b608be1p-935"),
        ("0x1.2632db402a593p+52", "0x1.255022bb76d28p+52"),
        ("0x1.401ec07d65549p+888", "0x1.3c3976605bb0cp+888"),
    )
]


def _sinc_ref(x: np.ndarray) -> np.ndarray:
  # np.sinc(x) computes sin(pi * x) / (pi * x) directly, which loses float64
  # precision near non-zero integers due to rounding in pi * x. Instead, use
  # exact additive range reduction x = n + r with n = round(x) and
  # r = x - n in [-0.5, 0.5] so sin(pi * x) = (-1)^n * sin(pi * r).
  x_dt = x.astype(np.float64)
  n = np.round(x_dt)
  r = x_dt - n
  val = np.where(n % 2 == 0, 1, -1) * np.sin(np.pi * r) / (np.pi * x_dt)
  # Special cases:
  # - x == 0: val evaluates 0 / 0 = NaN, whereas sinc(0) == 1.0.
  # - r == 0 (non-zero integers): (-1)^n * (+0.0) / (pi * x) yields -0.0
  #   when (-1)^n / x < 0; explicitly return +0.0.
  # - isinf(x): r = inf - inf = NaN, whereas lim_{|x|->inf} sinc(x) == 0.0.
  return np.where(
      x == 0, 1.0, np.where((r == 0) | np.isinf(x), 0.0, val)
  ).astype(np.float64)


def _sinpi_ref(x: np.ndarray) -> np.ndarray:
  t = np.fmod(x.astype(np.float64), 2.0)
  return np.where(
      np.abs(t) == 1.0, np.copysign(0.0, t), np.sin(np.pi * t)
  ).astype(np.float64)


def _cospi_ref(x: np.ndarray) -> np.ndarray:
  t = np.abs(np.fmod(x.astype(np.float64), 2.0))
  return np.where((t == 0.5) | (t == 1.5), 0.0, np.cos(np.pi * t)).astype(
      np.float64
  )


def _tanpi_ref(x: np.ndarray) -> np.ndarray:
  t = np.fmod(x.astype(np.float64), 2.0)
  at = np.abs(t)
  res = np.tan(np.pi * t)
  res = np.where(at == 1.0, np.copysign(0.0, -t), res)
  res = np.where(at == 0.5, np.copysign(np.inf, t), res)
  res = np.where(at == 1.5, np.copysign(np.inf, -t), res)
  return res.astype(np.float64)


def _mpmath_sinpi(x):
  fx = float(x)
  if math.isnan(fx) or math.isinf(fx):
    return mpmath.nan
  return mpmath.sinpi(x)


def _mpmath_cospi(x):
  fx = float(x)
  if math.isnan(fx) or math.isinf(fx):
    return mpmath.nan
  return mpmath.cospi(x)


def _mpmath_tanpi(x):
  fx = float(x)
  if math.isnan(fx) or math.isinf(fx):
    return mpmath.nan
  c, s = mpmath.cospi_sinpi(x)
  if c == 0:
    return mpmath.inf if s > 0 else -mpmath.inf
  return s / c


def _asinpi_ref(x: np.ndarray) -> np.ndarray:
  with np.errstate(all="ignore"):
    x64 = x.astype(np.float64)
    res = np.arcsin(x64) / np.pi
    res = np.where(np.abs(x64) == 1.0, np.copysign(0.5, x64), res)
    return res.astype(np.float64)


def _acospi_ref(x: np.ndarray) -> np.ndarray:
  with np.errstate(all="ignore"):
    x64 = x.astype(np.float64)
    res = np.arccos(x64) / np.pi
    res = np.where(x64 == 1.0, 0.0, np.where(x64 == -1.0, 1.0, res))
    return res.astype(np.float64)


def _atanpi_ref(x: np.ndarray) -> np.ndarray:
  with np.errstate(all="ignore"):
    x64 = x.astype(np.float64)
    res = np.arctan(x64) / np.pi
    res = np.where(np.isinf(x64), np.copysign(0.5, x64), res)
    return res.astype(np.float64)


def _atan2pi_ref(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  with np.errstate(all="ignore"):
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    res = np.arctan2(x64, y64) / np.pi
    neg_y = np.signbit(y64)
    both_inf = np.isinf(x64) & np.isinf(y64)
    only_x_inf = np.isinf(x64) & np.isfinite(y64)
    only_y_inf = np.isfinite(x64) & np.isinf(y64)
    res = np.where(
        both_inf, np.copysign(np.where(neg_y, 0.75, 0.25), x64), res
    )
    res = np.where(only_x_inf, np.copysign(0.5, x64), res)
    res = np.where(
        only_y_inf, np.copysign(np.where(neg_y, 1.0, 0.0), x64), res
    )
    return res.astype(np.float64)


def _mpmath_asinpi(x):
  return mpmath.asin(x) / mpmath.pi


def _mpmath_acospi(x):
  return mpmath.acos(x) / mpmath.pi


def _mpmath_atanpi(x):
  fx = float(x)
  if math.isinf(fx):
    return mpmath.mpf(-0.5 if np.signbit(fx) else 0.5)
  return mpmath.atan(x) / mpmath.pi


def _mpmath_atan2pi(x, y):
  if mpmath.isnan(x) or mpmath.isnan(y):
    return mpmath.nan
  fx, fy = float(x), float(y)
  sx = -1 if np.signbit(fx) else 1
  sy = -1 if np.signbit(fy) else 1
  if x == 0:
    return mpmath.mpf(0.0) if sy > 0 else mpmath.mpf(sx)
  if y == 0:
    return mpmath.mpf(sx) / 2
  if mpmath.isinf(x) and mpmath.isinf(y):
    return mpmath.mpf(sx) * (mpmath.mpf(1) if sy > 0 else mpmath.mpf(3)) / 4
  if mpmath.isinf(x):
    return mpmath.mpf(sx) / 2
  if mpmath.isinf(y):
    return mpmath.mpf(0.0) if sy > 0 else mpmath.mpf(sx)
  return mpmath.atan2(x, y) / mpmath.pi


_PI_TRIG_INTERESTING_POINTS = [
    *(
        sign * v
        for v in (
            0.125,
            0.25,
            0.375,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            10.0,
            100.5,
            float(2**22 + 0.5),
            float(2**22 + 1.5),
            float(2**23 - 1.5),
            float(2**23 - 0.5),
            float(2**23),
            float(2**23 + 1),
            float(2**24 - 1),
            float(2**24),
            float(2**51 + 0.5),
            float(2**51 + 1.5),
            float(2**52 - 1.5),
            float(2**52 - 0.5),
            float(2**52),
            float(2**52 + 1),
            float(2**53 - 1),
            float(2**53),
        )
        for sign in (-1, 1)
    ),
    1.4558208527321925,
    6.489487419631901,
    -7.526160955055503,
]


def _skip_if_unsupported_f64(test_case: jtu.JaxTestCase, dtype):
  if dtype == f64 and not config.enable_x64.value:
    test_case.skipTest("requires x64")
  if dtype == f64 and jtu.device_under_test() == "tpu":
    test_case.skipTest("float64 on TPU is ef57 double-double")


def _pi_trig_special_sets(dtype):
  finfo = jnp.finfo(dtype)
  even_ints = [2.0, 4.0, 2**finfo.nmant, float(finfo.max)]
  odd_ints = [1.0, 3.0, 2**finfo.nmant + 1, 2 ** (finfo.nmant + 1) - 1]
  q1_halves = [0.5, 2.5, 2 ** (finfo.nmant - 1) + 0.5, 2**finfo.nmant - 1.5]
  q3_halves = [1.5, 3.5, 2 ** (finfo.nmant - 1) + 1.5, 2**finfo.nmant - 0.5]
  interior = np.array(
      [
          float(finfo.smallest_normal),
          0.25,
          0.5 - 1e-4,
          0.5 + 1e-4,
          1.0 - 1e-4,
          1.0 + 1e-4,
          1.5 - 1e-4,
          1.5 + 1e-4,
          1.4558208527321925,
          6.489487419631901,
          -7.526160955055503,
      ],
      dtype=dtype,
  )
  return even_ints, odd_ints, q1_halves, q3_halves, interior


@jtu.thread_unsafe_test_class()
class SinTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sin_accuracy(self, dtype):
    bounds = [
        # ARM has 1.0 ULP error at the smallest normal inputs (+-2^-126 for f32,
        # +-2^-1022 for f64) where sin returns +-0.0, whereas Intel is correctly
        # rounded (0.5 ULP).
        ("cpu", {bf16: (0.5, 1.0), f16: 1.0, f32: (0.5, 1.0), f64: (0.5, 1.0)}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 2.5}),
        ("tpu", {bf16: 1.0, f16: 1.0, f32: 3.5}),
    ]
    input_ftz = [
        ("cpu", {f32: False}),
    ]
    check_signed_zeros = [
        ("cpu", {bf16: False}),
    ]
    # Worst-case inputs across math libraries from Gladman et al., "Accuracy of
    # Mathematical Functions in Single, Double, Double Extended, and Quadruple
    # Precision".
    interesting_points = [
        float.fromhex(h)
        for h in (
            "0x1p-149",
            "-0x1.f8b791cafcdefp+4",
            "-0x1.0e16eb809a35dp+944",
            "0x1.9eb7148f354d6p+21",
            "-0x1.842d8ec8f752fp+21",
            "0x1.4d84db080b9fdp+21",
            "-0x1.07e4c92b5349dp+4",
            "0x1.1031fb019618p-8",
            "-0x1.11b624b546894p+9",
            "-0x1.1c49ad613ff3bp+19",
            "-0x1.f05ed802c1cc7p+5",
        )
    ]
    util.check_unary_precision(
        self,
        jnp.sin,
        np.sin,
        mpmath.sin,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        check_signed_zeros=check_signed_zeros,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class CosTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cos_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0}),
        ("gpu", {f16: 1.0, f32: 2.0, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_v6e"], {f16: 1.0, f32: 3.5}),
        ("tpu_7x", {f16: 1.0, f32: 3.0}),
    ]
    # Worst-case inputs across math libraries from Gladman et al.
    interesting_points = [
        float.fromhex(h)
        for h in (
            "-0x1.7120161c92674p+0",
            "-0x1.d19ebc5567dcdp+311",
            "0x1.9eb7148f354d6p+20",
            "-0x1.4ae182c1ab422p+21",
            "-0x1.34e729fd08086p+21",
            "0x1.2f29eb4e99fa2p+7",
            "0x1.16e534ee3658p-4",
            "-0x1.9200634d4471fp-1",
            "0x1.25133ca3904dfp+20",
            "0x1.2a33ae1a620b1p+1",
        )
    ]
    util.check_unary_precision(
        self,
        jnp.cos,
        np.cos,
        mpmath.cos,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class TanTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tan_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0}),
        ("gpu", {f16: 1.0, f32: 3.5, f64: 2.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 6.5}),
        ("tpu_v5p", {f16: 1.0, f32: 7.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 5.5}),
    ]
    # Worst-case inputs across math libraries from Gladman et al.
    interesting_points = [
        float.fromhex(h)
        for h in (
            "0x1.000002p-14",
            "-0x1.317cd745dd37cp+9",
            "0x1.49adfd996a81dp+18",
            "0x1.93c05c9ed3cbcp+18",
            "0x1.3f9605aaeb51bp+21",
            "-0x1.a81d98fc58537p+6",
            "-0x1.13a5ccd87c9bbp+1008",
            "-0x1.4d7c8b8320237p+11",
            "0x1.da7a85a88bbecp+11",
            "0x1.5a7ee2ab3840cp+14",
        )
    ]
    util.check_unary_precision(
        self,
        jnp.tan,
        np.tan,
        mpmath.tan,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class SincTest(jtu.JaxTestCase):

  @jtu.run_on_devices("cpu")
  def test_sinc_ref(self):
    # Verify that the float64 NumPy reference implementation _sinc_ref matches
    # mpmath.sincpi within a few float64 ULPs across random full-range inputs
    # and explicit corner cases (zeros, infinities, NaNs, integers, and
    # near-integer values).
    rng = jtu.rand_fullrange(self.rng())
    random_x = rng((10_000,), np.float64)
    ints = np.array([-1000.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 1000.0])
    special_x = np.concatenate([
        np.array([0.0, -0.0, np.inf, -np.inf, np.nan]),
        ints,
        np.nextafter(ints, np.inf),
        np.nextafter(ints, -np.inf),
    ])
    x = np.concatenate([random_x, special_x])
    with np.errstate(all="ignore"):
      actual = _sinc_ref(x)
    expected = np.array(
        [
            util.eval_mpmath(mpmath.sincpi, v.item(), dtype=np.float64)
            for v in x
        ],
        dtype=object,
    )
    ulps = np.abs(util.ulp_diff(actual, expected, np.float64))
    self.assertLessEqual(np.max(ulps), 3.0)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinc_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 2.5, f64: 2.0}),
        ("gpu", {f32: 3.5, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_v6e"], {f32: 4.0}),
        ("tpu_7x", {f32: 3.5}),
    ]
    interesting_points = [
        # Non-zero integers (zeros of sinc(x) = sin(pi * x) / (pi * x)) and
        # half-integers (extrema of sin(pi * x)).
        *(
            sign * float(n)
            for n in (11, 12, 100, 1000)
            for sign in (-1, 1)
        ),
        *(
            sign * (n + 0.5)
            for n in (3, 4, 5, 10, 100)
            for sign in (-1, 1)
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.sinc,
        _sinc_ref,
        mpmath.sincpi,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class AcosTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acos_accuracy(self, dtype):
    if jaxlib_extension_version >= 512:
      bounds = [
          ("cpu", {f16: 1.0, f32: 1.0, f64: 1.5}),
          ("gpu", {f16: 1.0, f32: 1.5, f64: 2.0}),
          ("tpu", {f16: 1.0, f32: 1.0}),
      ]
    else:
      bounds = [
          ("cpu", {bf16: 1.5, f16: 1.5, f32: 1.5, f64: 1.5}),
          ("gpu", {f16: 1.0, f32: 1.5, f64: 2.0}),
          ([*TPU_EUPV1, "tpu_v5p", "tpu_7x"], {f16: 1.0, f32: 5.0}),
          ("tpu_v6e", {f16: 1.0, f32: 4.0}),
      ]
    interesting_points = _ACOS_INTERESTING_POINTS
    util.check_unary_precision(
        self,
        jnp.acos,
        np.arccos,
        mpmath.acos,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class AsinTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asin_accuracy(self, dtype):
    if jaxlib_extension_version >= 512:
      bounds = [
          ("cpu", {f32: 1.0, f64: 1.5}),
          ("gpu", {f16: 1.0, f32: 1.5, f64: 2.5}),
          ("tpu", {f32: 1.0}),
      ]
    else:
      bounds = [
          ("cpu", {bf16: 1.5, f16: 1.5, f32: 2.0, f64: 1.5}),
          ("gpu", {f16: 1.0, f32: 1.5, f64: 2.5}),
          ([*TPU_EUPV1, "tpu_v5p"], {bf16: 0.5, f32: 5.0}),
          (["tpu_v6e", "tpu_7x"], {bf16: 0.5, f32: 4.5}),
      ]
    util.check_unary_precision(
        self,
        jnp.asin,
        np.arcsin,
        mpmath.asin,
        dtype,
        bounds=bounds,
        interesting_points=_ASIN_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class AtanTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: (4.0, 5.5), f64: 3.5}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 2.5}),
        ("tpu", {f16: 1.0, f32: 2.5}),
    ]
    check_signed_zeros = [
        ("cpu", {f32: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.atan,
        np.arctan,
        mpmath.atan,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_ATAN_INTERESTING_POINTS,
    )


def _mpmath_atan2(y, x):
  if mpmath.isnan(y) or mpmath.isnan(x):
    return mpmath.nan
  fy, fx = float(y), float(x)
  sy = -1 if np.signbit(fy) else 1
  sx = -1 if np.signbit(fx) else 1
  if y == 0:
    return mpmath.mpf(0.0) if sx > 0 else sy * mpmath.pi
  if x == 0:
    return sy * (mpmath.pi / 2)
  if mpmath.isinf(y) and mpmath.isinf(x):
    return sy * (mpmath.pi / 4 if sx > 0 else 3 * mpmath.pi / 4)
  if mpmath.isinf(y):
    return sy * (mpmath.pi / 2)
  if mpmath.isinf(x):
    return mpmath.mpf(0.0) if sx > 0 else sy * mpmath.pi
  return mpmath.atan2(y, x)


@jtu.thread_unsafe_test_class()
class Atan2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan2_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 3.0, f64: 2.0}),
        ("tpu", {bf16: 1.0, f16: 1.0, f32: 3.5}),
    ]
    tiny_f64 = float(np.finfo(np.float64).tiny)
    ignore_inputs = [
        # On CPU float64, glibc's atan2 inspects y via integer bits (ignoring
        # MXCSR DAZ) while executing SSE comparisons/divisions on y and x with
        # DAZ enabled, returning +-pi/2 when |x| < tiny and -pi instead of +pi
        # when 0 < y < tiny and x < 0.
        (
            "cpu",
            {f64: lambda y, x: (np.abs(y) > 0.0) & (np.abs(y) < tiny_f64)},
        ),
    ]
    # Worst-case (y, x) inputs across math libraries from Gladman et al.
    util.check_nary_precision(
        self,
        jnp.atan2,
        np.arctan2,
        _mpmath_atan2,
        dtype,
        nargs=2,
        bounds=bounds,
        ignore_inputs=ignore_inputs,
        interesting_points=_ATAN2_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan2_grad_and_hessian(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    finfo = jnp.finfo(dtype)
    e_hi = (finfo.maxexp - 1) // 2
    # Scales where x^2 + y^2 overflows or underflows even though 1 / r is well
    # within the normal range.
    s_grad_hi = 2.0 ** (e_hi + 2)
    s_grad_lo = 2.0**finfo.minexp
    grad_fn = jax.jit(jax.grad(jnp.atan2, argnums=(0, 1)))
    tol = 8 * float(finfo.eps)
    for s in (1.0, s_grad_lo, s_grad_hi):
      y = jnp.array(3.0 * s, dtype=dtype)
      x = jnp.array(4.0 * s, dtype=dtype)
      gy, gx = grad_fn(y, x)
      # d/dy atan2(y, x) = x / r^2 = 0.8 / (5 * s) = 0.16 / s
      # d/dx atan2(y, x) = -y / r^2 = -0.6 / (5 * s) = -0.12 / s
      self.assertAllClose(float(gy), 0.16 / s, atol=0, rtol=tol)
      self.assertAllClose(float(gx), -0.12 / s, atol=0, rtol=tol)

    # Scales where (x^2 + y^2)^2 overflows or underflows in the 2nd derivative
    # even though 1 / r^2 is well within the normal range.
    s_hess_hi = 2.0 ** ((e_hi - 2) // 2 + 2)
    s_hess_lo = 2.0 ** (-(e_hi + 2))
    hess_fn = jax.jit(jax.hessian(jnp.atan2, argnums=(0, 1)))
    for s in (1.0, s_hess_lo, s_hess_hi):
      y = jnp.array(3.0 * s, dtype=dtype)
      x = jnp.array(4.0 * s, dtype=dtype)
      r = 5.0 * s
      # u1 = y / r = 0.6, u2 = x / r = 0.8:
      # d^2/dy^2 = -2 u1 u2 / r^2, d^2/dydx = (u1^2 - u2^2) / r^2,
      # d^2/dx^2 = 2 u1 u2 / r^2.
      expected_hess = (
          np.array([[-0.96, -0.28], [-0.28, 0.96]]) / (r * r)
      )
      hess = np.array([[float(h) for h in row] for row in hess_fn(y, x)])
      self.assertAllClose(hess, expected_hess, atol=0, rtol=tol)

    # Single-axis infinities: both partial derivatives and Hessian vanish;
    # (±inf, ±inf) has no well-defined derivative and produces NaN.
    for s_inf in (np.inf, -np.inf):
      for fin in (0.0, -0.0, 3.0, -3.0):
        v_inf = jnp.array(s_inf, dtype=dtype)
        v_fin = jnp.array(fin, dtype=dtype)
        for y, x in ((v_fin, v_inf), (v_inf, v_fin)):
          gy, gx = grad_fn(y, x)
          self.assertEqual(float(gy), 0.0)
          self.assertEqual(float(gx), 0.0)
          hess = np.array([[float(h) for h in row] for row in hess_fn(y, x)])
          np.testing.assert_array_equal(hess, np.zeros((2, 2)))
    for sy in (np.inf, -np.inf):
      for sx in (np.inf, -np.inf):
        gy, gx = grad_fn(jnp.array(sy, dtype=dtype), jnp.array(sx, dtype=dtype))
        self.assertTrue(np.isnan(float(gy)))
        self.assertTrue(np.isnan(float(gx)))


@jtu.thread_unsafe_test_class()
class Deg2radTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_deg2rad_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 1.0, f64: 1.0}),
        ("tpu", {bf16: 1.0, f16: 1.0, f32: 1.0}),
    ]
    util.check_unary_precision(
        self,
        jnp.deg2rad,
        np.deg2rad,
        mpmath.radians,
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class Rad2degTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_rad2deg_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.5, f32: 1.0, f64: 1.0}),
        ("gpu", {bf16: 1.0, f16: 1.5, f32: 1.0, f64: 1.0}),
        ("tpu", {bf16: 1.0, f16: 1.5, f32: 1.0}),
    ]
    util.check_unary_precision(
        self,
        jnp.rad2deg,
        np.rad2deg,
        mpmath.degrees,
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class SinpiTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinpi_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 1.0, f64: 1.0}),
        ("gpu", {f32: 2.0, f64: 1.5}),
        ("tpu", {f32: 1.5}),
    ]
    check_signed_zeros = [
        ("cpu", {bf16: False}),
    ]
    util.check_unary_precision(
        self,
        lax.sinpi,
        _sinpi_ref,
        _mpmath_sinpi,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_PI_TRIG_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinpi_special_values(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    even_ints, odd_ints, q1_halves, q3_halves, interior = _pi_trig_special_sets(
        dtype
    )
    pos_ints = [*even_ints, *odd_ints]
    cases = [
        (0.0, 0.0),
        (-0.0, -0.0),
        *((n, 0.0) for n in pos_ints),
        *((-n, -0.0) for n in pos_ints),
        *((h, 1.0) for h in q1_halves),
        *((-h, -1.0) for h in q1_halves),
        *((h, -1.0) for h in q3_halves),
        *((-h, 1.0) for h in q3_halves),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.sinpi)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    sym_x = np.concatenate([xs, interior])
    self.assertArraysEqual(
        np.asarray(jitted(-sym_x)).view(udt),
        (-np.asarray(jitted(sym_x))).view(udt),
    )
    nan_xs = np.array([np.inf, -np.inf, np.nan], dtype=dtype)
    self.assertTrue(np.all(np.isnan(np.asarray(jitted(nan_xs)))))


@jtu.thread_unsafe_test_class()
class CospiTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cospi_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 1.0, f64: 1.0}),
        ("gpu", {f32: 2.0, f64: 1.5}),
        ("tpu", {f32: 1.5}),
    ]
    util.check_unary_precision(
        self,
        lax.cospi,
        _cospi_ref,
        _mpmath_cospi,
        dtype,
        bounds=bounds,
        interesting_points=_PI_TRIG_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cospi_special_values(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    even_ints, odd_ints, q1_halves, q3_halves, interior = _pi_trig_special_sets(
        dtype
    )
    halves = [*q1_halves, *q3_halves]
    cases = [
        (0.0, 1.0),
        (-0.0, 1.0),
        *((n, 1.0) for n in even_ints),
        *((-n, 1.0) for n in even_ints),
        *((n, -1.0) for n in odd_ints),
        *((-n, -1.0) for n in odd_ints),
        *((h, 0.0) for h in halves),
        *((-h, 0.0) for h in halves),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.cospi)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    sym_x = np.concatenate([xs, interior])
    self.assertArraysEqual(
        np.asarray(jitted(-sym_x)).view(udt),
        np.asarray(jitted(sym_x)).view(udt),
    )
    nan_xs = np.array([np.inf, -np.inf, np.nan], dtype=dtype)
    self.assertTrue(np.all(np.isnan(np.asarray(jitted(nan_xs)))))


@jtu.thread_unsafe_test_class()
class TanpiTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanpi_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 2.0, f64: 1.5}),
        ("gpu", {f16: 1.0, f32: 3.0, f64: 2.0}),
        ("tpu", {f16: 1.0, f32: 3.0}),
    ]
    check_signed_zeros = [
        ("cpu", {bf16: False}),
    ]
    util.check_unary_precision(
        self,
        lax.tanpi,
        _tanpi_ref,
        _mpmath_tanpi,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_PI_TRIG_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanpi_special_values(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    even_ints, odd_ints, q1_halves, q3_halves, interior = _pi_trig_special_sets(
        dtype
    )
    cases = [
        (0.0, 0.0),
        (-0.0, -0.0),
        # IEEE 754-2019 §9.2.1: 2*n + {0.5, 1, 1.5, 2} for n >= 0,
        # and 2*n - {0.5, 1, 1.5, 2} for n <= 0.
        *((h, np.inf) for h in q1_halves),
        *((-h, -np.inf) for h in q1_halves),
        *((n, -0.0) for n in odd_ints),
        *((-n, 0.0) for n in odd_ints),
        *((h, -np.inf) for h in q3_halves),
        *((-h, np.inf) for h in q3_halves),
        *((n, 0.0) for n in even_ints),
        *((-n, -0.0) for n in even_ints),
        (0.25, 1.0),
        (-0.25, -1.0),
        (0.75, -1.0),
        (-0.75, 1.0),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.tanpi)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    sym_x = np.concatenate([xs, interior])
    self.assertArraysEqual(
        np.asarray(jitted(-sym_x)).view(udt),
        (-np.asarray(jitted(sym_x))).view(udt),
    )
    nan_xs = np.array([np.inf, -np.inf, np.nan], dtype=dtype)
    self.assertTrue(np.all(np.isnan(np.asarray(jitted(nan_xs)))))


@jtu.thread_unsafe_test_class()
class AsinpiTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinpi_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("tpu", {f16: 1.0, f32: 1.0}),
    ]
    check_signed_zeros = [
        ("cpu", {bf16: False}),
    ]
    util.check_unary_precision(
        self,
        lax.asinpi,
        _asinpi_ref,
        _mpmath_asinpi,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_ASIN_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinpi_special_values(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = jnp.finfo(dtype)
    cases = [
        (0.0, 0.0),
        (-0.0, -0.0),
        (1.0, 0.5),
        (-1.0, -0.5),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.asinpi)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    interior = np.array(
        [
            float(finfo.smallest_normal),
            float(finfo.smallest_normal) * 4.0,
            0.25,
            0.5,
            0.75,
        ],
        dtype=dtype,
    )
    sym_x = np.concatenate([xs, interior])
    self.assertArraysEqual(
        np.asarray(jitted(-sym_x)).view(udt),
        (-np.asarray(jitted(sym_x))).view(udt),
    )
    out_of_domain = [
        float(1.0 + finfo.eps),
        2.0,
        2**finfo.nmant,
        float(finfo.max),
        np.inf,
    ]
    nan_xs = np.array(
        [*out_of_domain, *(-v for v in out_of_domain), np.nan], dtype=dtype
    )
    self.assertTrue(np.all(np.isnan(np.asarray(jitted(nan_xs)))))


@jtu.thread_unsafe_test_class()
class AcospiTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acospi_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 1.0, f64: 1.0}),
        ("gpu", {f32: 1.0, f64: 1.0}),
        ("tpu", {f32: 1.0}),
    ]
    util.check_unary_precision(
        self,
        lax.acospi,
        _acospi_ref,
        _mpmath_acospi,
        dtype,
        bounds=bounds,
        interesting_points=_ACOS_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acospi_special_values(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = jnp.finfo(dtype)
    cases = [
        (1.0, 0.0),
        (0.0, 0.5),
        (-0.0, 0.5),
        (-1.0, 1.0),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.acospi)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    out_of_domain = [
        float(1.0 + finfo.eps),
        2.0,
        2**finfo.nmant,
        float(finfo.max),
        np.inf,
    ]
    nan_xs = np.array(
        [*out_of_domain, *(-v for v in out_of_domain), np.nan], dtype=dtype
    )
    self.assertTrue(np.all(np.isnan(np.asarray(jitted(nan_xs)))))


@jtu.thread_unsafe_test_class()
class AtanpiTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanpi_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 2.0, f64: 2.0}),
        ("gpu", {f16: 1.0, f32: 2.0, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_v6e"], {f16: 1.0, f32: 2.5}),
        ("tpu_7x", {f16: 1.0, f32: 2.0}),
    ]
    check_signed_zeros = [
        ("cpu", {bf16: False}),
    ]
    util.check_unary_precision(
        self,
        lax.atanpi,
        _atanpi_ref,
        _mpmath_atanpi,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_ATAN_INTERESTING_POINTS,
    )
    if dtype in (f32, f64):
      _skip_if_unsupported_f64(self, dtype)
      # On the unreduced interval |x| <= tan(pi/8) = sqrt(2) - 1, splitting x
      # against half-width 1/pi keeps _atanpi_poly within 1.0 ULP even across
      # (pi/8, sqrt(2) - 1] where x / pi > 0.125 > atanpi(x).
      t1 = float(np.sqrt(2.0) - 1.0)
      np_dtype = np.dtype(dtype)
      xs = np.linspace(0.39, t1, 512, dtype=np_dtype)
      actual = np.asarray(jax.jit(lax.atanpi)(xs), dtype=np.float64)
      expected = np.array(
          [
              util.eval_mpmath(_mpmath_atanpi, float(v), dtype=np_dtype)
              for v in xs
          ],
          dtype=object,
      )
      ulps = np.abs(util.ulp_diff(actual, expected, np_dtype))
      self.assertLessEqual(float(np.max(ulps)), 1.0)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanpi_special_values(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = jnp.finfo(dtype)
    cases = [
        (0.0, 0.0),
        (-0.0, -0.0),
        (1.0, 0.25),
        (-1.0, -0.25),
        (np.inf, 0.5),
        (-np.inf, -0.5),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.atanpi)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    interior = np.array(
        [
            float(finfo.smallest_normal),
            float(finfo.smallest_normal) * 4.0,
            0.25,
            0.5,
            2.0,
            float(finfo.max),
        ],
        dtype=dtype,
    )
    sym_x = np.concatenate([xs, interior])
    self.assertArraysEqual(
        np.asarray(jitted(-sym_x)).view(udt),
        (-np.asarray(jitted(sym_x))).view(udt),
    )
    self.assertTrue(np.isnan(float(jitted(np.array(np.nan, dtype=dtype)))))

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanpi_grad_and_hessian(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    finfo = jnp.finfo(dtype)
    e_hi = (finfo.maxexp - 1) // 2
    s_grad_hi = 2.0 ** (e_hi + 2)
    s_grad_lo = 2.0**finfo.minexp
    jvp_fn = jax.jit(lambda x, dx: jax.jvp(lax.atanpi, (x,), (dx,))[1])
    grad_fn = jax.jit(jax.grad(lax.atanpi))
    tol = 8 * float(finfo.eps)
    for s in (1.0, s_grad_hi):
      x = jnp.array(s, dtype=dtype)
      expected_jvp = (
          s / (np.pi * (1.0 + s * s)) if s <= 2.0**500 else 1.0 / (np.pi * s)
      )
      self.assertAllClose(float(jvp_fn(x, x)), expected_jvp, atol=0, rtol=tol)
    for s in (1.0, s_grad_lo):
      x = jnp.array(s, dtype=dtype)
      self.assertAllClose(
          float(grad_fn(x)), 1.0 / (np.pi * (1.0 + s * s)), atol=0, rtol=tol
      )
    if dtype == f16 or jtu.device_under_test() == "gpu":
      # Subnormals are preserved on float16 across devices and on GPU, so
      # 1 / (pi * (1 + s_grad_hi^2)) remains non-zero rather than flushing to 0.
      self.assertGreater(float(grad_fn(jnp.array(s_grad_hi, dtype=dtype))), 0.0)

    s_hess_hi = 2.0 ** ((e_hi - 2) // 2 + 2)
    s_hess_lo = 2.0 ** (-(e_hi + 2))
    hess_fn = jax.jit(jax.hessian(lax.atanpi))
    for s in (1.0, s_hess_lo, s_hess_hi):
      x = jnp.array(s, dtype=dtype)
      if s < 2.0**-100:
        expected_hess = -2.0 * s / np.pi
      elif s > 2.0**100:
        expected_hess = -2.0 / (np.pi * s * s * s)
      else:
        expected_hess = -2.0 * s / (np.pi * (1.0 + s * s) ** 2)
      self.assertAllClose(float(hess_fn(x)), expected_hess, atol=0, rtol=tol)
    self.assertEqual(float(hess_fn(jnp.array(s_grad_hi, dtype=dtype))), 0.0)

    for s_inf in (np.inf, -np.inf):
      v_inf = jnp.array(s_inf, dtype=dtype)
      self.assertEqual(float(grad_fn(v_inf)), 0.0)
      self.assertEqual(float(hess_fn(v_inf)), 0.0)


@jtu.thread_unsafe_test_class()
class Atan2piTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan2pi_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 2.0, f64: 2.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 3.0, f64: 2.0}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 3.5}),
        (["tpu_v5p", "tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 3.0}),
    ]
    util.check_nary_precision(
        self,
        lax.atan2pi,
        _atan2pi_ref,
        _mpmath_atan2pi,
        dtype,
        nargs=2,
        bounds=bounds,
        interesting_points=_ATAN2_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan2pi_special_values(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = jnp.finfo(dtype)
    pos_fin = [
        float(finfo.smallest_normal),
        0.5,
        1.0,
        2.0,
        float(finfo.max),
    ]
    all_fin = [0.0, -0.0, *pos_fin, *(-v for v in pos_fin)]
    # IEEE 754-2019 §9.2.1 special cases of atan2Pi(y, x).
    cases = [
        (0.0, -0.0, 1.0),
        (-0.0, -0.0, -1.0),
        (0.0, 0.0, 0.0),
        (-0.0, 0.0, -0.0),
        *((0.0, -v, 1.0) for v in pos_fin),
        *((-0.0, -v, -1.0) for v in pos_fin),
        *((0.0, v, 0.0) for v in pos_fin),
        *((-0.0, v, -0.0) for v in pos_fin),
        *((-v, 0.0, -0.5) for v in pos_fin),
        *((-v, -0.0, -0.5) for v in pos_fin),
        *((v, 0.0, 0.5) for v in pos_fin),
        *((v, -0.0, 0.5) for v in pos_fin),
        *((v, -np.inf, 1.0) for v in pos_fin),
        *((-v, -np.inf, -1.0) for v in pos_fin),
        *((v, np.inf, 0.0) for v in pos_fin),
        *((-v, np.inf, -0.0) for v in pos_fin),
        *((np.inf, x, 0.5) for x in all_fin),
        *((-np.inf, x, -0.5) for x in all_fin),
        (np.inf, -np.inf, 0.75),
        (-np.inf, -np.inf, -0.75),
        (np.inf, np.inf, 0.25),
        (-np.inf, np.inf, -0.25),
        *((v, v, 0.25) for v in pos_fin),
        *((-v, v, -0.25) for v in pos_fin),
        *((v, -v, 0.75) for v in pos_fin),
        *((-v, -v, -0.75) for v in pos_fin),
    ]
    ys = np.array([c[0] for c in cases], dtype=dtype)
    xs = np.array([c[1] for c in cases], dtype=dtype)
    expected = np.array([c[2] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.atan2pi)
    actual = np.asarray(jitted(ys, xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))

    grid_vals = np.array(
        [
            0.0,
            -0.0,
            float(finfo.smallest_normal),
            -float(finfo.smallest_normal),
            float(finfo.smallest_normal) * 4.0,
            -float(finfo.smallest_normal) * 4.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            float(finfo.max),
            -float(finfo.max),
            np.inf,
            -np.inf,
        ],
        dtype=dtype,
    )
    gy = np.broadcast_to(grid_vals[:, None], (len(grid_vals), len(grid_vals)))
    gx = np.broadcast_to(grid_vals[None, :], (len(grid_vals), len(grid_vals)))
    self.assertArraysEqual(
        np.asarray(jitted(-gy, gx)).view(udt),
        (-np.asarray(jitted(gy, gx))).view(udt),
    )

    nan_pairs = [
        *((np.nan, v) for v in (0.0, -0.0, 1.0, -1.0, np.inf, -np.inf)),
        *((v, np.nan) for v in (0.0, -0.0, 1.0, -1.0, np.inf, -np.inf)),
        (np.nan, np.nan),
    ]
    nan_ys = np.array([p[0] for p in nan_pairs], dtype=dtype)
    nan_xs = np.array([p[1] for p in nan_pairs], dtype=dtype)
    self.assertTrue(np.all(np.isnan(np.asarray(jitted(nan_ys, nan_xs)))))

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan2pi_unit_denom_matches_atanpi(self, dtype):
    _skip_if_unsupported_f64(self, dtype)
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = jnp.finfo(dtype)
    pos = np.concatenate([
        np.linspace(0.0, 0.5, 257),
        np.linspace(0.5, 1.0, 257),
        np.linspace(1.0, 2.5, 1025),
        np.linspace(2.5, 16.0, 257),
        np.array([float(finfo.smallest_normal), float(finfo.max), np.inf]),
    ]).astype(dtype)
    xs = np.concatenate([np.array([0.0, -0.0], dtype=dtype), pos, -pos])
    ones = np.ones_like(xs)
    actual = np.asarray(jax.jit(lax.atan2pi)(xs, ones))
    expected = np.asarray(jax.jit(lax.atanpi)(xs))
    self.assertArraysEqual(actual.view(udt), expected.view(udt))

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan2pi_grad_and_hessian(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    finfo = jnp.finfo(dtype)
    e_hi = (finfo.maxexp - 1) // 2
    s_grad_hi = 2.0 ** (e_hi + 2)
    s_grad_lo = 2.0**finfo.minexp
    grad_fn = jax.jit(jax.grad(lax.atan2pi, argnums=(0, 1)))
    tol = 8 * float(finfo.eps)
    for s in (1.0, s_grad_lo, s_grad_hi):
      y = jnp.array(3.0 * s, dtype=dtype)
      x = jnp.array(4.0 * s, dtype=dtype)
      gy, gx = grad_fn(y, x)
      self.assertAllClose(float(gy), 0.16 / (np.pi * s), atol=0, rtol=tol)
      self.assertAllClose(float(gx), -0.12 / (np.pi * s), atol=0, rtol=tol)

    s_hess_hi = 2.0 ** ((e_hi - 2) // 2 + 2)
    s_hess_lo = 2.0 ** (-(e_hi + 2))
    hess_fn = jax.jit(jax.hessian(lax.atan2pi, argnums=(0, 1)))
    for s in (1.0, s_hess_lo, s_hess_hi):
      y = jnp.array(3.0 * s, dtype=dtype)
      x = jnp.array(4.0 * s, dtype=dtype)
      r = 5.0 * s
      expected_hess = (
          np.array([[-0.96, -0.28], [-0.28, 0.96]]) / (np.pi * r * r)
      )
      hess = np.array([[float(h) for h in row] for row in hess_fn(y, x)])
      self.assertAllClose(hess, expected_hess, atol=0, rtol=tol)

    for s_inf in (np.inf, -np.inf):
      for fin in (0.0, -0.0, 3.0, -3.0):
        v_inf = jnp.array(s_inf, dtype=dtype)
        v_fin = jnp.array(fin, dtype=dtype)
        for y, x in ((v_fin, v_inf), (v_inf, v_fin)):
          gy, gx = grad_fn(y, x)
          self.assertEqual(float(gy), 0.0)
          self.assertEqual(float(gx), 0.0)
          hess = np.array([[float(h) for h in row] for row in hess_fn(y, x)])
          np.testing.assert_array_equal(hess, np.zeros((2, 2)))
    for sy in (np.inf, -np.inf):
      for sx in (np.inf, -np.inf):
        gy, gx = grad_fn(jnp.array(sy, dtype=dtype), jnp.array(sx, dtype=dtype))
        self.assertTrue(np.isnan(float(gy)))
        self.assertTrue(np.isnan(float(gx)))


util.register_benchmark(jnp.sin)
util.register_benchmark(jnp.cos)
util.register_benchmark(jnp.tan)
util.register_benchmark(lax.sinpi)
util.register_benchmark(lax.cospi)
util.register_benchmark(lax.tanpi)
util.register_benchmark(lax.asinpi)
util.register_benchmark(lax.acospi)
util.register_benchmark(lax.atanpi)
util.register_benchmark(lax.atan2pi, nargs=2)
util.register_benchmark(jnp.sinc)
util.register_benchmark(jnp.acos)
util.register_benchmark(jnp.asin)
util.register_benchmark(jnp.atan)
util.register_benchmark(jnp.atan2, nargs=2)
util.register_benchmark(jnp.deg2rad)
util.register_benchmark(jnp.rad2deg)


if __name__ == "__main__":
  util.main()
