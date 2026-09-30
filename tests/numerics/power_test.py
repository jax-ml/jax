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

"""Precision tests for power, root, and reciprocal functions."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import lax
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


@jtu.thread_unsafe_test_class()
class SqrtTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sqrt_accuracy(self, dtype):
    bounds = [
        ("gpu", {f32: 1.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 3.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2.0}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.sqrt,
        np.sqrt,
        mpmath.sqrt,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class RsqrtTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_rsqrt_accuracy(self, dtype):
    # The max ULP for float32 is either 1.0 and 2.0 depend on CPU vendor (AMD
    # vs Intel). The (min, max) tuple allows tightness checks to pass on both.
    bounds = [
        ("cpu", {f32: (1.0, 2.0), f64: 1.5}),
        ("gpu", {f32: 2.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f32: 2.5}),
        ("tpu_v6e", {f32: 1.5}),
        ("tpu_7x", {f32: 1.0}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        lax.rsqrt,
        lambda x: np.reciprocal(np.sqrt(x)),
        lambda x: mpmath.nan if x < 0 else 1 / mpmath.sqrt(x),
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class CbrtTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cbrt_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 4.5}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.cbrt,
        np.cbrt,
        lambda x: -mpmath.cbrt(-x) if x < 0 else mpmath.cbrt(x),
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class SquareTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_square_accuracy(self, dtype):
    bounds = []
    util.check_unary_precision(
        self, jnp.square, np.square, lambda x: x * x, dtype, bounds=bounds
    )


@jtu.thread_unsafe_test_class()
class ReciprocalTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_reciprocal_accuracy(self, dtype):
    bounds = [
        ("gpu", {f32: 1.0}),
        (TPU_EUPV1, {bf16: 1.0, f32: 198.0}),
        ("tpu_v5p", {bf16: 1.0, f32: 40.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.reciprocal,
        np.reciprocal,
        lambda x: 1 / x,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


if __name__ == "__main__":
  absltest.main(testLoader=util.ClassShardedTestLoader())
