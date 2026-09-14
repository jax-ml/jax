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

config.parse_flags_with_absl()


bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
tpu_devices = [
    "tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i",
    "tpu_v5e", "tpu_v5p", "tpu_v6e", "tpu_7x",
]


@jtu.skip_under_pytest("Only runs under Bazel")
@jtu.thread_unsafe_test_class()
class ElementaryTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 1, f64: 2}),
        (["gpu"], {bf16: 1, f16: 1, f32: 2, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 116}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 109}),
        (["tpu_v6e"], {bf16: 1, f16: 1, f32: 64}),
        (["tpu_7x"], {bf16: 1, f16: 1, f32: 65}),
    ]
    util.check_unary_precision(self, jnp.exp, mpmath.exp, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log(self, dtype):
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

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sin(self, dtype):
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

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cos(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 1}),
        (["gpu"], {f16: 1, f32: 2, f64: 1}),
        (tpu_devices, {f16: 1, f32: 3}),
    ]
    util.check_unary_precision(self, jnp.cos, mpmath.cos, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tan(self, dtype):
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

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinh(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 24, f64: 496}),
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
    # Ignore x = +-89.415985: near the float32 overflow threshold, where the
    # true value is finite (~3.40282e+38) but XLA's lowering on CPU/TPU computes
    # 0.5 * exp(x), where exp(x) overflows float32 to inf.
    ignore_inputs = [
        (["cpu", "tpu", *tpu_devices], {f32: [0x42b2d4fc, 0xc2b2d4fc]}),
    ]
    util.check_unary_precision(
        self, jnp.sinh, mpmath.sinh, dtype,
        bounds=bounds, input_ftz=input_ftz, ignore_inputs=ignore_inputs,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cosh(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 24, f64: 496}),
        (["gpu"], {f16: 1, f32: 2, f64: 2}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {f16: 1, f32: 93}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 99}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 59}),
    ]
    # Ignore x = +-89.415985: near the float32 overflow threshold, where the
    # true value is finite (~3.40282e+38) but XLA's lowering on CPU/TPU computes
    # 0.5 * exp(x), where exp(x) overflows float32 to inf.
    ignore_inputs = [
        (["cpu", "tpu", *tpu_devices], {f32: [0x42b2d4fc, 0xc2b2d4fc]}),
    ]
    util.check_unary_precision(
        self, jnp.cosh, mpmath.cosh, dtype, bounds=bounds, ignore_inputs=ignore_inputs
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh(self, dtype):
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

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acos(self, dtype):
    bounds = [
        (["cpu"], {bf16: 1, f16: 1, f32: 1, f64: 1}),
        (["gpu"], {f16: 1, f32: 1, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e", "tpu_v5p", "tpu_7x"], {f16: 1, f32: 5}),
        (["tpu_v6e"], {f16: 1, f32: 4}),
    ]
    util.check_unary_precision(
        self, jnp.acos, mpmath.acos, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asin(self, dtype):
    bounds = [
        (["cpu"], {bf16: 128, f16: 1, f32: 8388608}),
        (["gpu"], {f16: 1, f32: 1, f64: 2}),
        (tpu_devices, {bf16: 128, f32: 8388607}),
    ]
    util.check_unary_precision(
        self, jnp.asin, mpmath.asin, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atan(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 4, f64: 4}),
        (["gpu"], {f16: 1, f32: 1, f64: 2}),
        (tpu_devices, {f16: 1, f32: 2}),
    ]
    util.check_unary_precision(
        self, jnp.atan, mpmath.atan, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acosh(self, dtype):
    bounds = [
        (["cpu"], {bf16: 2, f16: 2, f32: 4, f64: 4}),
        (["gpu"], {f16: 1, f32: 2, f64: 2}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 4031}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 1003}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 984}),
    ]
    util.check_unary_precision(
        self, jnp.acosh, mpmath.acosh, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinh(self, dtype):
    bounds = [
        (["cpu"], {bf16: 1, f16: 1, f32: 3, f64: 2}),
        (["gpu"], {f16: 1, f32: 2, f64: 2}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 4034}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 2082}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 2049}),
    ]
    util.check_unary_precision(
        self, jnp.asinh, mpmath.asinh, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanh(self, dtype):
    bounds = [
        (["cpu"], {bf16: 1, f16: 1, f32: 3, f64: 3}),
        (["gpu"], {f32: 3, f64: 3}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 2183}),
        (["tpu_v5p"], {f16: 1, f32: 1061}),
        (["tpu_v6e", "tpu_7x"], {f32: 1025}),
    ]
    util.check_unary_precision(
        self, jnp.atanh, mpmath.atanh, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sqrt(self, dtype):
    bounds = [
        (["gpu"], {f32: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e", "tpu_v5p"], {f16: 1, f32: 3}),
        (["tpu_v6e", "tpu_7x"], {f16: 1, f32: 2}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.sqrt, mpmath.sqrt, dtype, bounds=bounds, input_ftz=input_ftz
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_rsqrt(self, dtype):
    bounds = [
        # f64 is 1 ULP vs mpmath, but the numpy reference (1 / sqrt(x) in f64)
        # can itself be 1 ULP off in the opposite direction.
        (["cpu"], {f32: 2, f64: 2}),
        (["gpu"], {f32: 2, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e", "tpu_v5p"], {f32: 2}),
        (["tpu_v6e", "tpu_7x"], {f32: 1}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self,
        lax.rsqrt,
        lambda x: mpmath.nan if x < 0 else 1 / mpmath.sqrt(x),
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cbrt(self, dtype):
    bounds = [
        (["cpu"], {f16: 1}),
        (["gpu"], {f16: 1, f32: 1, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e", "tpu_v5p"], {f16: 1, f32: 4}),
        (["tpu_v6e", "tpu_7x"], {f16: 1, f32: 31}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.cbrt,
        lambda x: -mpmath.cbrt(-x) if x < 0 else mpmath.cbrt(x),
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_square(self, dtype):
    bounds = []
    util.check_unary_precision(
        self, jnp.square, lambda x: x * x, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_reciprocal(self, dtype):
    bounds = [
        (["gpu"], {f32: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f32: 198}),
        (["tpu_v5p"], {bf16: 1, f32: 40}),
        (["tpu_v6e", "tpu_7x"], {f32: 1}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.reciprocal,
        lambda x: 1 / x,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2(self, dtype):
    # TODO(phawkins): lax.exp2 lowers as exp(x * ln(2)) where ln(2) is cast
    # to the input dtype. In bfloat16, ln(2) has ~0.25% relative error, which
    # causes large output errors (~93 ULP) and causes x = 128.0 to evaluate to
    # finite 2.72e+38 instead of overflowing to inf.
    bounds = [
        (["cpu"], {bf16: 93, f16: 14, f32: 68, f64: 719}),
        (["gpu"], {bf16: 93, f16: 14, f32: 69, f64: 719}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 51, f16: 7, f32: 141}),
        (["tpu_v5p"], {bf16: 51, f16: 7, f32: 133}),
        (["tpu_v6e"], {bf16: 51, f16: 7, f32: 90}),
        (["tpu_7x"], {bf16: 75, f16: 7, f32: 90}),
    ]
    ignore_inputs = [
        (["cpu", "gpu", "tpu", *tpu_devices], {bf16: [0x4300]}),
    ]
    util.check_unary_precision(
        self, jnp.exp2, lambda x: mpmath.power(2, x), dtype, bounds=bounds,
        ignore_inputs=ignore_inputs,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_expm1(self, dtype):
    bounds = [
        (["cpu"], {f16: 2, f32: 6, f64: 5}),
        (["gpu"], {f16: 1, f32: 1, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 1772}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 1357}),
        (["tpu_v6e"], {bf16: 1, f32: 64}),
        (["tpu_7x"], {bf16: 1, f32: 63}),
    ]
    util.check_unary_precision(
        self, jnp.expm1, mpmath.expm1, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2(self, dtype):
    bounds = [
        (["cpu"], {bf16: 2, f16: 2, f32: 2, f64: 1}),
        (["gpu"], {f16: 1, f32: 2, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 5159}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 57}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 2}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.log2, mpmath.log2, dtype, bounds=bounds, input_ftz=input_ftz
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log10(self, dtype):
    bounds = [
        (["cpu"], {bf16: 2, f16: 1, f32: 3, f64: 2}),
        (["gpu"], {f16: 1, f32: 2, f64: 2}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 6152}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 57}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 3}),
    ]
    input_ftz = [
        (["cpu"], {f16: False}),
        (["gpu"], {bf16: False, f16: False, f32: False, f64: False}),
        (tpu_devices, {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.log10, mpmath.log10, dtype, bounds=bounds, input_ftz=input_ftz
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log1p(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 3, f64: 2}),
        (["gpu"], {f16: 1, f32: 1, f64: 1}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 4034}),
        (["tpu_v5p"], {f16: 1, f32: 2082}),
        (["tpu_v6e", "tpu_7x"], {f16: 1, f32: 2049}),
    ]
    util.check_unary_precision(
        self, jnp.log1p, mpmath.log1p, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logistic(self, dtype):
    bounds = [
        (["cpu"], {bf16: 2, f16: 2, f32: 2, f64: 4}),
        (["gpu"], {bf16: 2, f16: 2, f32: 4, f64: 4}),
        (["tpu_7x"], {bf16: 63, f16: 1, f32: 64}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 243}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 124}),
        (["tpu_v6e"], {f16: 1, f32: 65}),
    ]
    util.check_unary_precision(
        self,
        lax.logistic,
        lambda x: 1 / (1 + mpmath.exp(-x)),
        dtype,
        bounds=bounds,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinc(self, dtype):
    # TODO(phawkins): Large errors occur when |x| >= max_float / pi because
    # pi * x overflows to inf, evaluating to sin(inf) = NaN rather than 0.0.
    bounds = [
        (["cpu"], {bf16: 18446744073709551615, f16: 18446744073709551615, f32: 18446744073709551615, f64: 1}),
        (["gpu"], {bf16: 18446744073709551615, f16: 18446744073709551615, f32: 18446744073709551615, f64: 2}),
        (tpu_devices, {bf16: 18446744073709551615, f16: 18446744073709551615, f32: 18446744073709551615}),
    ]
    util.check_unary_precision(
        self, jnp.sinc, mpmath.sincpi, dtype, bounds=bounds
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
