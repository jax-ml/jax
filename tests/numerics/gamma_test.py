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

"""Precision tests for gamma-related functions against reference implementations."""

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
import scipy.special

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]


@jtu.thread_unsafe_test_class()
class DigammaTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_digamma_accuracy(self, dtype):
    # Digamma has poles at non-positive integers (0, -1, -2, ...) where the
    # function is ill-conditioned and outputs differ across implementations
    # (NaN vs +-inf).
    bounds = [
        ("cpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("gpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("tpu", {bf16: math.inf, f16: math.inf, f32: math.inf}),
    ]
    util.check_unary_precision(
        self,
        lax.digamma,
        scipy.special.psi,
        mpmath.digamma,
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class LgammaTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_lgamma_accuracy(self, dtype):
    # Large errors occur near zero-crossings (where |gamma(x)| = 1, e.g.
    # x ~= -3.14358) where small approximation errors cause sign flips across
    # zero, and at x = -inf due to inf/NaN handling differences.
    bounds = [
        ("cpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("gpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("tpu", {bf16: math.inf, f16: math.inf, f32: math.inf}),
    ]
    util.check_unary_precision(
        self,
        lax.lgamma,
        scipy.special.gammaln,
        lambda x: mpmath.log(abs(mpmath.gamma(x))),
        dtype,
        bounds=bounds,
    )


util.register_benchmark(lax.digamma)
util.register_benchmark(lax.lgamma)


if __name__ == "__main__":
  util.main()
