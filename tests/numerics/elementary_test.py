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

"""Precision tests for elementary functions against reference implementations."""

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

config.parse_flags_with_absl()


bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
tpu_devices = [
    "tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i",
    "tpu_v5e", "tpu_v5p", "tpu_v6e", "tpu_7x",
]


@jtu.skip_under_pytest("Only runs under Bazel")
@jtu.thread_unsafe_test_class()
class ElementaryTest(jtu.JaxTestCase):

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_exp(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {f16: 1, f32: 1, f64: 2}),
        (["gpu"], {bf16: 1, f16: 1, f32: 2, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 116}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 109}),
        (["tpu_v6e"], {bf16: 1, f16: 1, f32: 64}),
        (["tpu_7x"], {bf16: 1, f16: 1, f32: 65}),
    ]
    util.check_unary_precision(self, jnp.exp, mpmath.exp, dtype, bounds=bounds)

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_log(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {f16: 1, f32: 1, f64: 1}),
        (["gpu"], {f16: 1, f32: 1, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 4030}),
        (["tpu_v5p"], {f16: 1, f32: 62}),
        (["tpu_v6e", "tpu_7x"], {f16: 1, f32: 2}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.log, mpmath.log, dtype, bounds=bounds, input_ftz=input_ftz
    )

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_sin(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {bf16: 1, f16: 1, f32: 1, f64: 1}),
        (["gpu"], {f16: 1, f32: 1, f64: 2}),
        (tpu_devices, {bf16: 1, f16: 1, f32: 3}),
    ]
    input_ftz = [
        (["cpu"], {f16: False, f32: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.sin, mpmath.sin, dtype, bounds=bounds, input_ftz=input_ftz
    )

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_cos(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {f16: 1, f32: 1, f64: 1}),
        (["gpu"], {f16: 1, f32: 2, f64: 2}),
        (tpu_devices, {f16: 1, f32: 3}),
    ]
    util.check_unary_precision(self, jnp.cos, mpmath.cos, dtype, bounds=bounds)

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_tan(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {f16: 1, f64: 1}),
        (["gpu"], {f16: 1, f32: 3, f64: 2}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {f16: 1, f32: 6}),
        (["tpu_v5p"], {f16: 1, f32: 7}),
        (["tpu_v6e", "tpu_7x"], {f16: 1, f32: 5}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.tan, mpmath.tan, dtype, bounds=bounds, input_ftz=input_ftz
    )

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_sinh(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {f16: 1, f32: 24, f64: 497}),
        (["gpu"], {f32: 3, f64: 2}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 1794}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 1332}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 59}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    # XLA's composite sinh incorrectly overflows to +/-inf for two f32 input
    # values, namely +/-89.4159851 (0x42b2d4fc, 0xc2b2d4fc), due to rounding
    # error when computing x +/- log(1/2). The correct answer of 3.40281961e+38
    # (0x7f7fffec) is very close to max-float. See xla/hlo/builder/lib/math.cc.
    ignore_inputs = [
        (["cpu", *tpu_devices], {f32: [0x42B2D4FC, 0xC2B2D4FC]}),
    ]
    util.check_unary_precision(
        self, jnp.sinh, mpmath.sinh, dtype,
        bounds=bounds, input_ftz=input_ftz, ignore_inputs=ignore_inputs,
    )

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_cosh(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {f16: 1, f32: 24, f64: 496}),
        (["gpu"], {f16: 1, f32: 2, f64: 2}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {f16: 1, f32: 93}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 99}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 59}),
    ]
    # XLA's composite cosh incorrectly overflows to inf for two f32 input
    # values, namely +/-89.4159851 (0x42b2d4fc, 0xc2b2d4fc), due to rounding
    # error when computing x +/- log(1/2). The correct answer of 3.40281961e+38
    # (0x7f7fffec) is very close to max-float. See xla/hlo/builder/lib/math.cc.
    ignore_inputs = [
        (["cpu", *tpu_devices], {f32: [0x42B2D4FC, 0xC2B2D4FC]}),
    ]
    util.check_unary_precision(
        self, jnp.cosh, mpmath.cosh, dtype, bounds=bounds, ignore_inputs=ignore_inputs
    )

  @parameterized.parameters(bf16, f16, f32, f64)
  def test_tanh(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")

    bounds = [
        (["cpu"], {f32: 5, f64: 7}),
        (["gpu"], {f32: 5, f64: 3}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {f16: 1, f32: 1365}),
        (["tpu_v5p"], {f16: 1, f32: 92}),
        (["tpu_v6e", "tpu_7x"], {f32: 1}),
    ]
    input_ftz = [
        (["cpu"], {bf16: False, f16: False, f32: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.tanh, mpmath.tanh, dtype, bounds=bounds, input_ftz=input_ftz
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
