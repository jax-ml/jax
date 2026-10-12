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

"""Precision tests for hyperbolic and inverse hyperbolic functions."""

import math

from absl.testing import parameterized
import jax
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


def _sinh_cosh_interesting_points(dtype):
  # Overflow threshold where sinh(x)/cosh(x) = 0.5 * exp(|x|) overflows
  # (~ln(max_val) + ln(2); ln(max_val) and +-0.5*ln(2), +-ln(2) are already in
  # _common_interesting_points).
  sinh_overflow = math.log(float(jnp.finfo(dtype).max)) + math.log(2.0)
  return [-sinh_overflow, sinh_overflow]


def _tanh_interesting_points(dtype):
  p = jnp.finfo(dtype).nmant + 1
  return [
      # Saturation thresholds where tanh(x) rounds to +-1.0 (~0.5 * (p + 1) * ln(2)).
      *(
          sign * 0.5 * k * math.log(2.0)
          for k in (p + 1, p + 2)
          for sign in (-1, 1)
      ),
  ]


# Note: Separate gradient tests for sinh and cosh are omitted because
# grad(sinh) lowers directly to cosh and grad(cosh) lowers directly to sinh,
# which are already covered by CoshTest and SinhTest.


def tanh_grad(x):
  return jax.vmap(jax.grad(jnp.tanh))(x)


def tanh_grad_highest(x):
  return jax.vmap(
      jax.grad(lambda x: lax.tanh(x, accuracy=lax.AccuracyMode.HIGHEST))
  )(x)


def acosh_grad(x):
  return jax.vmap(jax.grad(jnp.acosh))(x)


def asinh_grad(x):
  return jax.vmap(jax.grad(jnp.asinh))(x)


def atanh_grad(x):
  return jax.vmap(jax.grad(jnp.atanh))(x)


@jtu.thread_unsafe_test_class()
class SinhTest(jtu.JaxTestCase):

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
        ("gpu", False),
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
        self,
        jnp.sinh,
        np.sinh,
        mpmath.sinh,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        ignore_inputs=ignore_inputs,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_sinh_cosh_interesting_points(dtype),
    )


@jtu.thread_unsafe_test_class()
class CoshTest(jtu.JaxTestCase):

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
        self,
        jnp.cosh,
        np.cosh,
        mpmath.cosh,
        dtype,
        bounds=bounds,
        ignore_inputs=ignore_inputs,
        interesting_points=_sinh_cosh_interesting_points(dtype),
    )


@jtu.thread_unsafe_test_class()
class TanhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 5.0, f64: 7.0}),
        ("gpu", {f32: 5.5, f64: 3.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 1365.5}),
        ("tpu_v5p", {f16: 1.0, f32: 92.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {bf16: False, f32: False}),
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        jnp.tanh,
        np.tanh,
        mpmath.tanh,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        interesting_points=_tanh_interesting_points(dtype),
    )


@jtu.thread_unsafe_test_class()
class TanhGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh_grad_accuracy(self, dtype):
    # By default, grad(tanh)(x) is evaluated from y = tanh(x) as
    # (1 + y) * (1 - y), which suffers catastrophic cancellation as |y|
    # approaches 1 and rounds to 0.0 once tanh(x) saturates to +-1.0 (~2^p ULPs).
    bounds = [
        (
            "cpu",
            {
                bf16: 256.0,
                f16: 2045.5,
                f32: 16777206.5,
                f64: 117093590311632992.0,
            },
        ),
        (
            "gpu",
            {
                bf16: 256.0,
                f16: 2045.5,
                f32: 16777206.5,
                f64: 9007199254740964.5,
            },
        ),
        (TPU_EUPV1, {bf16: 383.5, f16: 723.5, f32: 51270903.0}),
        ("tpu_v5p", {bf16: 274.5, f16: 72.0, f32: 18925970.5}),
        ("tpu_v6e", {bf16: 256.0, f16: 3.0, f32: 16777199.0}),
        ("tpu_7x", {bf16: 256.0, f16: 2.5, f32: 16777199.0}),
    ]
    util.check_unary_precision(
        self,
        tanh_grad,
        lambda x: np.square(np.reciprocal(np.cosh(x))),
        lambda x: 1 / mpmath.cosh(x) ** 2,
        dtype,
        bounds=bounds,
        interesting_points=_tanh_interesting_points(dtype),
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh_grad_highest_accuracy(self, dtype):
    # With accuracy=AccuracyMode.HIGHEST, grad(tanh)(x) is evaluated as
    # 4 * logistic(2 * x) * logistic(-2 * x). On CPU/TPU in FTZ mode (bf16/f32),
    # the intermediate logistic(-2 * |x|) flushes to 0.0 before multiplying by 4
    # once 2 * |x| >= -ln(tiny) (|x| >= 43.67), while the true derivative is
    # still normal. In float16 on CPU/GPU, exp(2 * |x|) overflows for
    # |x| >= 5.55 while the true derivative is subnormal (~1020.5 ULPs).
    bounds = [
        ("cpu", {bf16: 154.0, f16: 1020.5, f32: 12020967.5, f64: 3.0}),
        ("gpu", {bf16: 3.5, f16: 1020.5, f32: 4.5, f64: 3.0}),
        ("tpu", {bf16: 154.0, f16: 1.0, f32: 12020967.5}),
    ]
    util.check_unary_precision(
        self,
        tanh_grad_highest,
        lambda x: np.square(np.reciprocal(np.cosh(x))),
        lambda x: 1 / mpmath.cosh(x) ** 2,
        dtype,
        bounds=bounds,
        interesting_points=_tanh_interesting_points(dtype),
    )


@jtu.thread_unsafe_test_class()
class AcoshTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acosh_accuracy(self, dtype):
    if jtu.device_under_test() == "tpu" and not jtu.is_libtpu_at_least("0.0.50"):
      self.skipTest("Requires libtpu >= 0.0.50")
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 4.5, f64: 3.5}),
        ("gpu", {f16: 1.0, f32: 2.5, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4031.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 61.0}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 4.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 5.0}),
    ]
    util.check_unary_precision(
        self,
        jnp.acosh,
        np.arccosh,
        mpmath.acosh,
        dtype,
        bounds=bounds,
        interesting_points=[math.cosh(1.0)],
    )


@jtu.thread_unsafe_test_class()
class AcoshGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acosh_grad_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 5.5, f64: 3.0}),
        ("gpu", {bf16: 2.0, f16: 2.0, f32: 5.0, f64: 2.5}),
        (TPU_EUPV1, {f32: 5.0}),
        ("tpu_v5p", {f32: 5.0}),
        ("tpu_v6e", {f32: 3.5}),
        ("tpu_7x", {f32: 3.0}),
    ]
    util.check_unary_precision(
        self,
        acosh_grad,
        lambda x: np.reciprocal(np.sqrt(x - 1.0) * np.sqrt(x + 1.0)),
        lambda x: (
            mpmath.nan
            if x < 1
            else (mpmath.inf if x == 1 else 1 / mpmath.sqrt(x * x - 1))
        ),
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class AsinhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinh_accuracy(self, dtype):
    if jtu.device_under_test() == "tpu" and not jtu.is_libtpu_at_least("0.0.50"):
      self.skipTest("Requires libtpu >= 0.0.50")
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.5, f64: 2.5}),
        ("gpu", {f16: 1.0, f32: 2.0, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4034.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 62.5}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 3.5}),
    ]
    util.check_unary_precision(
        self,
        jnp.asinh,
        np.arcsinh,
        mpmath.asinh,
        dtype,
        bounds=bounds,
        interesting_points=[-math.sinh(1.0), math.sinh(1.0)],
    )


@jtu.thread_unsafe_test_class()
class AsinhGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinh_grad_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 2.5, f64: 2.0}),
        ("gpu", {bf16: 2.0, f16: 2.0, f32: 2.5, f64: 2.0}),
        (TPU_EUPV1, {f16: 1.0, f32: 3.0}),
        ("tpu_v5p", {f16: 1.0, f32: 3.0}),
        ("tpu_v6e", {f16: 1.0, f32: 2.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 2.0}),
    ]
    util.check_unary_precision(
        self,
        asinh_grad,
        lambda x: np.reciprocal(np.sqrt(np.square(x) + 1.0)),
        lambda x: 1 / mpmath.sqrt(x * x + 1),
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class AtanhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanh_accuracy(self, dtype):
    if jtu.device_under_test() == "tpu" and not jtu.is_libtpu_at_least("0.0.50"):
      self.skipTest("Requires libtpu >= 0.0.50")
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.0, f64: 2.5}),
        ("gpu", {f32: 3.5, f64: 3.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 2183.5}),
        ("tpu_v5p", {f16: 1.0, f32: 43.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 3.5}),
    ]
    util.check_unary_precision(
        self,
        jnp.atanh,
        np.arctanh,
        mpmath.atanh,
        dtype,
        bounds=bounds,
        interesting_points=[-math.tanh(1.0), math.tanh(1.0)],
    )


@jtu.thread_unsafe_test_class()
class AtanhGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanh_grad_accuracy(self, dtype):
    # In float16 on CPU/GPU, (1 - x) * (1 + x) overflows to -inf for |x| >= 256,
    # causing the reciprocal to evaluate to -0.0 while the true value is
    # subnormal (~256 ULPs).
    bounds = [
        ("cpu", {bf16: 2.5, f16: 256.5, f32: 2.5, f64: 3.5}),
        ("gpu", {bf16: 2.5, f16: 256.5, f32: 3.0, f64: 3.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 190.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 37.5}),
        ("tpu_v6e", {f16: 1.0, f32: 3.5}),
        ("tpu_7x", {bf16: 2.5, f16: 1.0, f32: 3.5}),
    ]
    util.check_unary_precision(
        self,
        atanh_grad,
        lambda x: np.reciprocal((1.0 - x) * (1.0 + x)),
        lambda x: mpmath.inf if abs(x) == 1 else 1 / (1 - x * x),
        dtype,
        bounds=bounds,
    )


util.register_benchmark(jnp.sinh)
util.register_benchmark(jnp.cosh)
util.register_benchmark(jnp.tanh)
util.register_benchmark(tanh_grad)
util.register_benchmark(tanh_grad_highest)
util.register_benchmark(jnp.acosh)
util.register_benchmark(acosh_grad)
util.register_benchmark(jnp.asinh)
util.register_benchmark(asinh_grad)
util.register_benchmark(jnp.atanh)
util.register_benchmark(atanh_grad)


if __name__ == "__main__":
  util.main()
