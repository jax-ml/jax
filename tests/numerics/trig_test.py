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
        ("gpu", False),
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
    input_ftz = [
        ("gpu", False),
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
        input_ftz=input_ftz,
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
    ulps = np.abs(util.ulp_diff_mpmath(actual, expected, np.float64))
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
    interesting_points = [
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
    interesting_points = [
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
    util.check_unary_precision(
        self,
        jnp.asin,
        np.arcsin,
        mpmath.asin,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
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
    interesting_points = [
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
    util.check_unary_precision(
        self,
        jnp.atan,
        np.arctan,
        mpmath.atan,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=interesting_points,
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
    input_ftz = [
        ("gpu", False),
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
    interesting_points = [
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
    util.check_nary_precision(
        self,
        jnp.atan2,
        np.arctan2,
        _mpmath_atan2,
        dtype,
        nargs=2,
        bounds=bounds,
        input_ftz=input_ftz,
        ignore_inputs=ignore_inputs,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class Deg2radTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_deg2rad_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 1.0, f64: 1.0}),
        ("tpu", {bf16: 1.0, f16: 1.0, f32: 1.0}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        jnp.deg2rad,
        np.deg2rad,
        mpmath.radians,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
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
    input_ftz = [
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        jnp.rad2deg,
        np.rad2deg,
        mpmath.degrees,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


util.register_benchmark(jnp.sin)
util.register_benchmark(jnp.cos)
util.register_benchmark(jnp.tan)
util.register_benchmark(jnp.sinc)
util.register_benchmark(jnp.acos)
util.register_benchmark(jnp.asin)
util.register_benchmark(jnp.atan)
util.register_benchmark(jnp.atan2, nargs=2)
util.register_benchmark(jnp.deg2rad)
util.register_benchmark(jnp.rad2deg)


if __name__ == "__main__":
  util.main()
