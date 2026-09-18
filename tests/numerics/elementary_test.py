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

import math

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


@jtu.skip_under_pytest("Only runs under Bazel")
@jtu.thread_unsafe_test_class()
class ElementaryTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.5, f64: 1.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 2.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 116.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 109.5}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 64.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 65.0}),
    ]
    util.check_unary_precision(
        self, jnp.exp, np.exp, mpmath.exp, dtype, bounds=bounds)

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
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.log, np.log, mpmath.log, dtype,
        bounds=bounds, input_ftz=input_ftz)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sin_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0}),
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
        self, jnp.sin, np.sin, mpmath.sin, dtype,
        bounds=bounds, input_ftz=input_ftz,
        check_signed_zeros=check_signed_zeros)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cos_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0}),
        ("gpu", {f16: 1.0, f32: 2.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_v6e"], {f16: 1.0, f32: 3.5}),
        ("tpu_7x", {f16: 1.0, f32: 3.0}),
    ]
    util.check_unary_precision(
        self, jnp.cos, np.cos, mpmath.cos, dtype, bounds=bounds)

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
        self, jnp.tan, np.tan, mpmath.tan, dtype,
        bounds=bounds, input_ftz=input_ftz)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinh_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 25.0, f64: 496.0}),
        ("gpu", {f32: 3.0, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 1794.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 1332.5}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 59.0}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 59.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    # Ignore x = +-89.415985: near the float32 overflow threshold, where the
    # true value is finite (~3.40282e+38) but XLA's lowering on CPU/TPU computes
    # 0.5 * exp(x), where exp(x) overflows float32 to inf.
    ignore_inputs = [
        (["cpu", "tpu"], {f32: [0x42B2D4FC, 0xC2B2D4FC]}),
    ]
    check_signed_zeros = [
        ("tpu", False),
    ]
    util.check_unary_precision(
        self, jnp.sinh, np.sinh, mpmath.sinh, dtype,
        bounds=bounds, input_ftz=input_ftz, ignore_inputs=ignore_inputs,
        check_signed_zeros=check_signed_zeros)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cosh_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 25.0, f64: 496.0}),
        ("gpu", {f16: 1.0, f32: 2.5, f64: 2.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 93.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 99.0}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 59.0}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 59.5}),
    ]
    # Ignore x = +-89.415985: near the float32 overflow threshold, where the
    # true value is finite (~3.40282e+38) but XLA's lowering on CPU/TPU computes
    # 0.5 * exp(x), where exp(x) overflows float32 to inf.
    ignore_inputs = [
        (["cpu", "tpu"], {f32: [0x42B2D4FC, 0xC2B2D4FC]}),
    ]
    util.check_unary_precision(
        self, jnp.cosh, np.cosh, mpmath.cosh, dtype,
        bounds=bounds, ignore_inputs=ignore_inputs)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 5.0, f64: 6.5}),
        ("gpu", {f32: 5.5, f64: 3.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 1365.5}),
        ("tpu_v5p", {f16: 1.0, f32: 92.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {bf16: False, f16: False, f32: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.tanh, np.tanh, mpmath.tanh, dtype,
        bounds=bounds, input_ftz=input_ftz)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acos_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 1.5, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_7x"], {f16: 1.0, f32: 5.0}),
        ("tpu_v6e", {f16: 1.0, f32: 4.0}),
    ]
    util.check_unary_precision(
        self, jnp.acos, np.arccos, mpmath.acos, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asin_accuracy(self, dtype):
    # TODO(phawkins): Large errors occur for normal inputs in the first exponent
    # bin (|x| < 2 * tiny) on CPU/TPU because XLA lowers asin(x) to
    # 2 * atan2(x, 1 + sqrt(1 - x^2)), where the intermediate x / 2 is subnormal
    # and flushes to zero under FTZ mode, causing asin(x) to evaluate to 0.0.
    bounds = [
        ("cpu",
         {bf16: 128.0, f16: 1.5, f32: 8388608.0, f64: 4503599627370496.0}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 2.5}),
        ("tpu", {bf16: 128.0, f32: 8388607.0}),
    ]
    util.check_unary_precision(
        self, jnp.asin, np.arcsin, mpmath.asin, dtype, bounds=bounds)

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
        self, jnp.atan, np.arctan, mpmath.atan, dtype, bounds=bounds,
        check_signed_zeros=check_signed_zeros)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acosh_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 4.5, f64: 3.5}),
        ("gpu", {f16: 1.0, f32: 2.5, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4031.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 1003.0}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 984.0}),
    ]
    util.check_unary_precision(
        self, jnp.acosh, np.arccosh, mpmath.acosh, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinh_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.5, f64: 2.0}),
        ("gpu", {f16: 1.0, f32: 2.0, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4034.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 2082.5}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 2049.0}),
    ]
    util.check_unary_precision(
        self, jnp.asinh, np.arcsinh, mpmath.asinh, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanh_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.0, f64: 2.5}),
        ("gpu", {f32: 3.5, f64: 3.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 2183.5}),
        ("tpu_v5p", {f16: 1.0, f32: 1061.5}),
        (["tpu_v6e", "tpu_7x"], {f32: 1025.5}),
    ]
    util.check_unary_precision(
        self, jnp.atanh, np.arctanh, mpmath.atanh, dtype, bounds=bounds)

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
        self, jnp.sqrt, np.sqrt, mpmath.sqrt, dtype,
        bounds=bounds, input_ftz=input_ftz)

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
        self, lax.rsqrt, lambda x: np.reciprocal(np.sqrt(x)),
        lambda x: mpmath.nan if x < 0 else 1 / mpmath.sqrt(x),
        dtype, bounds=bounds, input_ftz=input_ftz)

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
        self, jnp.cbrt, np.cbrt,
        lambda x: -mpmath.cbrt(-x) if x < 0 else mpmath.cbrt(x),
        dtype, bounds=bounds, input_ftz=input_ftz)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_square_accuracy(self, dtype):
    bounds = []
    util.check_unary_precision(
        self, jnp.square, np.square, lambda x: x * x, dtype, bounds=bounds)

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
        self, jnp.reciprocal, np.reciprocal, lambda x: 1 / x,
        dtype, bounds=bounds, input_ftz=input_ftz)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2_accuracy(self, dtype):
    # TODO(phawkins): lax.exp2 lowers as exp(x * ln(2)) where ln(2) is cast
    # to the input dtype. In bfloat16, ln(2) has ~0.25% relative error, which
    # causes large output errors (~93 ULP) and causes x = 128.0 to evaluate to
    # finite 2.72e+38 instead of overflowing to inf.
    bounds = [
        ("cpu", {bf16: 101.0, f16: 14.0, f32: 68.5, f64: 719.0}),
        ("gpu", {bf16: 101.0, f16: 14.0, f32: 69.0, f64: 719.0}),
        (TPU_EUPV1, {bf16: 44.0, f16: 7.5, f32: 141.5}),
        ("tpu_v5p", {bf16: 44.0, f16: 7.5, f32: 133.0}),
        ("tpu_v6e", {bf16: 44.0, f16: 7.5, f32: 90.0}),
        ("tpu_7x", {bf16: 75.0, f16: 7.5, f32: 90.0}),
    ]
    ignore_inputs = [
        (["cpu", "gpu", "tpu"], {bf16: [0x4300]}),
    ]
    util.check_unary_precision(
        self, jnp.exp2, np.exp2, lambda x: mpmath.power(2, x),
        dtype, bounds=bounds, ignore_inputs=ignore_inputs)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_expm1_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 2.5, f32: 6.5, f64: 4.5}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 1772.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 1357.5}),
        ("tpu_v6e", {bf16: 1.0, f32: 64.0}),
        ("tpu_7x", {bf16: 1.0, f32: 63.5}),
    ]
    check_signed_zeros = [
        ("tpu", False),
    ]
    util.check_unary_precision(
        self, jnp.expm1, np.expm1, mpmath.expm1, dtype, bounds=bounds,
        check_signed_zeros=check_signed_zeros)

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
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.log2, np.log2, mpmath.log2, dtype,
        bounds=bounds, input_ftz=input_ftz)

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
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self, jnp.log10, np.log10, mpmath.log10, dtype,
        bounds=bounds, input_ftz=input_ftz)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log1p_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 3.0, f64: 1.5}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4034.0}),
        ("tpu_v5p", {f16: 1.0, f32: 2082.5}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2049.0}),
    ]
    util.check_unary_precision(
        self, jnp.log1p, np.log1p, mpmath.log1p, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logistic_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.5, f16: 2.0, f32: 2.5, f64: 3.5}),
        ("gpu", {bf16: 2.5, f16: 2.0, f32: 4.0, f64: 4.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 243.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 124.0}),
        ("tpu_v6e", {f16: 1.0, f32: 65.5}),
        ("tpu_7x", {bf16: 63.0, f16: 1.0, f32: 64.0}),
    ]
    util.check_unary_precision(
        self, lax.logistic, lambda x: 1.0 / (1.0 + np.exp(-x)),
        lambda x: 1 / (1 + mpmath.exp(-x)), dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinc_accuracy(self, dtype):
    # TODO(phawkins): Large errors occur when |x| >= max_float / pi because
    # pi * x overflows to inf, evaluating to sin(inf) = NaN rather than 0.0.
    bounds = [
        ("cpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("gpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("tpu", {bf16: math.inf, f16: 4145.0, f32: math.inf}),
    ]
    ref_fn = lambda x: np.where(np.isinf(x), 0.0, np.sinc(x))
    util.check_unary_precision(
        self, jnp.sinc, ref_fn, mpmath.sincpi, dtype, bounds=bounds)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
