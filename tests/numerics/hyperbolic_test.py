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
        ("cpu", {bf16: False, f16: False, f32: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    p = jnp.finfo(dtype).nmant + 1
    interesting_points = [
        # Saturation thresholds where tanh(x) rounds to +-1.0 (~0.5 * (p + 1) * ln(2)).
        *(
            sign * 0.5 * k * math.log(2.0)
            for k in (p + 1, p + 2)
            for sign in (-1, 1)
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.tanh,
        np.tanh,
        mpmath.tanh,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class AcoshTest(jtu.JaxTestCase):

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
        self,
        jnp.acosh,
        np.arccosh,
        mpmath.acosh,
        dtype,
        bounds=bounds,
        interesting_points=[math.cosh(1.0)],
    )


@jtu.thread_unsafe_test_class()
class AsinhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinh_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.5, f64: 2.5}),
        ("gpu", {f16: 1.0, f32: 2.0, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4034.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 2082.5}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 2049.0}),
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
class AtanhTest(jtu.JaxTestCase):

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
        self,
        jnp.atanh,
        np.arctanh,
        mpmath.atanh,
        dtype,
        bounds=bounds,
        interesting_points=[-math.tanh(1.0), math.tanh(1.0)],
    )


util.register_benchmark(jnp.sinh)
util.register_benchmark(jnp.cosh)
util.register_benchmark(jnp.tanh)
util.register_benchmark(jnp.acosh)
util.register_benchmark(jnp.asinh)
util.register_benchmark(jnp.atanh)


if __name__ == "__main__":
  util.main()
