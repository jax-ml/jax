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

"""Precision tests for logarithmic functions."""

import math

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

# Mantissa range-reduction breakpoints in [0.5, 2.0] for log implementations.
_LOG_INTERESTING_POINTS = [
    math.sqrt(0.5),
    math.sqrt(2.0),
    2.0 / 3.0,
    4.0 / 3.0,
]


@jtu.thread_unsafe_test_class()
class LogTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.5}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4030.5}),
        ("tpu_v5p", {f16: 1.0, f32: 62.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2.5}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        jnp.log,
        np.log,
        mpmath.log,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        interesting_points=_LOG_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class Log2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 2.5, f64: 1.5}),
        ("gpu", {bf16: 2.0, f16: 2.0, f32: 2.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.5, f32: 5159.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 57.5}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 2.5}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        jnp.log2,
        np.log2,
        mpmath.log2,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        interesting_points=_LOG_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class Log10Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log10_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 1.5, f32: 3.0, f64: 2.0}),
        ("gpu", {bf16: 2.0, f16: 1.5, f32: 2.5, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 6213.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 57.0}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 3.0}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        jnp.log10,
        np.log10,
        mpmath.log10,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        interesting_points=[
            *_LOG_INTERESTING_POINTS,
            # Exact powers of 10 where log10(x) is an integer.
            *(10.0**k for k in (-10, -3, -2, -1, 1, 2, 3, 10)),
        ],
    )


@jtu.thread_unsafe_test_class()
class Log1pTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log1p_accuracy(self, dtype):
    if jtu.device_under_test() == "tpu" and not jtu.is_libtpu_at_least("0.0.50"):
      self.skipTest("Requires libtpu >= 0.0.50")
    bounds = [
        ("cpu", {f16: 1.0, f32: 3.0, f64: 2.0}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4034.0}),
        ("tpu_v5p", {f16: 1.0, f32: 63.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 3.0}),
    ]
    # Points where 1 + x crosses range-reduction thresholds in [0.5, 2.0] or e.
    interesting_points = [
        math.sqrt(0.5) - 1.0,
        math.sqrt(2.0) - 1.0,
        math.e - 1.0,
    ]
    util.check_unary_precision(
        self,
        jnp.log1p,
        np.log1p,
        mpmath.log1p,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


util.register_benchmark(jnp.log)
util.register_benchmark(jnp.log2)
util.register_benchmark(jnp.log10)
util.register_benchmark(jnp.log1p)


if __name__ == "__main__":
  util.main()
