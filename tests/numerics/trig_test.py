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

from absl.testing import absltest
from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp

# Under pytest, tests run against an installed wheel that does not
# include `jax.tests`, so skip before importing `jax.tests.numerics`.
if jtu.is_running_under_pytest():
  import pytest

  pytest.skip("Only runs under Bazel", allow_module_level=True)

from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


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
        # ARM has 1.0 ULP error at the smallest normal inputs (+-2^-126) where
        # sin returns +-0.0, whereas Intel is correctly rounded (0.5 ULP).
        ("cpu", {bf16: (0.5, 1.0), f16: 1.0, f32: (0.5, 1.0)}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 2.5}),
        ("tpu", {bf16: 1.0, f16: 1.0, f32: 3.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False, f32: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    check_signed_zeros = [
        ("cpu", {bf16: False}),
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
    )


@jtu.thread_unsafe_test_class()
class CosTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cos_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0}),
        ("gpu", {f16: 1.0, f32: 2.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_v6e"], {f16: 1.0, f32: 3.5}),
        ("tpu_7x", {f16: 1.0, f32: 3.0}),
    ]
    util.check_unary_precision(
        self, jnp.cos, np.cos, mpmath.cos, dtype, bounds=bounds
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
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.tan,
        np.tan,
        mpmath.tan,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
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
    util.check_unary_precision(
        self, jnp.sinc, _sinc_ref, mpmath.sincpi, dtype, bounds=bounds
    )


@jtu.thread_unsafe_test_class()
class AcosTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acos_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 1.5, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_7x"], {f16: 1.0, f32: 5.0}),
        ("tpu_v6e", {f16: 1.0, f32: 4.0}),
    ]
    util.check_unary_precision(
        self, jnp.acos, np.arccos, mpmath.acos, dtype, bounds=bounds
    )


@jtu.thread_unsafe_test_class()
class AsinTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asin_accuracy(self, dtype):
    # Large errors occur for normal inputs in the first exponent bin
    # (|x| < 2 * tiny) on CPU/TPU because XLA lowers asin(x) to
    # 2 * atan2(x, 1 + sqrt(1 - x^2)), where the intermediate x / 2 is
    # subnormal and flushes to zero under FTZ mode, causing asin(x) to
    # evaluate to 0.0.
    bounds = [
        (
            "cpu",
            {bf16: 128.0, f16: 1.5, f32: 8388608.0, f64: 4503599627370496.0},
        ),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 2.5}),
        ("tpu", {bf16: 128.0, f32: 8388607.0}),
    ]
    util.check_unary_precision(
        self, jnp.asin, np.arcsin, mpmath.asin, dtype, bounds=bounds
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
    )


if __name__ == "__main__":
  absltest.main(testLoader=util.ClassShardedTestLoader())
