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

"""Precision tests for exponential and logistic functions."""

import math

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


def _exp_highest(x):
  return lax.exp(x, accuracy=lax.AccuracyMode.HIGHEST)


def _expm1_highest(x):
  return lax.expm1(x, accuracy=lax.AccuracyMode.HIGHEST)


# Additional range-reduction boundaries (k * ln(2)) beyond +-0.5*ln(2) and
# +-ln(2) (which are already in _common_interesting_points).
_EXP_INTERESTING_POINTS = [k * math.log(2.0) for k in (-10, -2, 2, 10)]


def _expm1_interesting_points(dtype):
  # Thresholds where expm1(x) saturates to -1.0 for `dtype`.
  p = jnp.finfo(dtype).nmant + 1
  return [-p * math.log(2.0), -(p + 1) * math.log(2.0)]


@jtu.thread_unsafe_test_class()
class ExpTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.5, f64: 2.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 2.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 116.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 109.5}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 64.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 65.0}),
    ]
    util.check_unary_precision(
        self,
        jnp.exp,
        np.exp,
        mpmath.exp,
        dtype,
        bounds=bounds,
        interesting_points=_EXP_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp_highest_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.5, f64: 2.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 2.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {bf16: 1.0, f16: 1.0, f32: 1.5}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    util.check_unary_precision(
        self,
        _exp_highest,
        np.exp,
        mpmath.exp,
        dtype,
        bounds=bounds,
        interesting_points=_EXP_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class Exp2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2_accuracy(self, dtype):
    # lax.exp2 lowers as exp(x * ln(2)) where ln(2) is cast to the input dtype.
    # In bfloat16, ln(2) has ~0.25% relative error, which causes large output
    # errors (~93 ULP) and causes x = 128.0 to evaluate to finite 2.72e+38
    # instead of overflowing to inf.
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
        self,
        jnp.exp2,
        np.exp2,
        lambda x: mpmath.power(2, x),
        dtype,
        bounds=bounds,
        ignore_inputs=ignore_inputs,
    )


@jtu.thread_unsafe_test_class()
class Expm1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_expm1_accuracy(self, dtype):
    if jtu.is_libtpu_at_least("0.0.50"):
      tpu_bounds = [
          (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 1772.0}),
          ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 1357.5}),
          (["tpu_v6e", "tpu_7x"], {f32: 3.5}),
      ]
      check_signed_zeros = [
          ([*TPU_EUPV1, "tpu_v5p"], False),
      ]
    else:
      tpu_bounds = [
          (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 1772.0}),
          ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 1357.5}),
          ("tpu_v6e", {bf16: 1.0, f32: 64.0}),
          ("tpu_7x", {bf16: 1.0, f32: 63.5}),
      ]
      check_signed_zeros = [
          ("tpu", False),
      ]
    bounds = [
        ("cpu", {f16: 2.5, f32: 6.5, f64: 4.5}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        *tpu_bounds,
    ]
    util.check_unary_precision(
        self,
        jnp.expm1,
        np.expm1,
        mpmath.expm1,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_expm1_interesting_points(dtype),
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_expm1_highest_accuracy(self, dtype):
    if jtu.device_under_test() == "tpu" and not jtu.is_libtpu_at_least("0.0.50"):
      self.skipTest("Requires libtpu >= 0.0.50")
    bounds = [
        ("cpu", {f16: 2.5, f32: 6.5, f64: 3.5}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 2.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 2.0}),
    ]
    util.check_unary_precision(
        self,
        _expm1_highest,
        np.expm1,
        mpmath.expm1,
        dtype,
        bounds=bounds,
        interesting_points=_expm1_interesting_points(dtype),
    )


@jtu.thread_unsafe_test_class()
class LogisticTest(jtu.JaxTestCase):

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
    p = jnp.finfo(dtype).nmant + 1
    interesting_points = [
        # Saturation to 1.0 thresholds for `dtype` (underflow to 0.0 at
        # -log(fmax) / -log(tiny) is already in _common_interesting_points).
        *(sign * k * math.log(2.0) for k in (p, p + 1) for sign in (-1, 1)),
    ]
    # TODO(phawkins): lax.logistic lowers to 1 / (1 + exp(-x)), which prematurely
    # underflows to 0.0 on CPU and GPU when exp(-x) overflows in float16 for
    # x <= -11.09.
    output_ftz = [
        (["cpu", "gpu"], {f16: True}),
    ]
    util.check_unary_precision(
        self,
        lax.logistic,
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        lambda x: 1 / (1 + mpmath.exp(-x)),
        dtype,
        bounds=bounds,
        output_ftz=output_ftz,
        interesting_points=interesting_points,
    )


util.register_benchmark(jnp.exp)
util.register_benchmark(_exp_highest)
util.register_benchmark(jnp.exp2)
util.register_benchmark(jnp.expm1)
util.register_benchmark(_expm1_highest)
util.register_benchmark(lax.logistic)


if __name__ == "__main__":
  util.main()
