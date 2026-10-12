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


def _mpmath_digamma(x):
  if x <= 0 and mpmath.isint(x):
    return mpmath.nan
  return mpmath.digamma(x)


def _mpmath_lgamma(x):
  if (x <= 0 and mpmath.isint(x)) or x == -mpmath.inf:
    return mpmath.inf
  return mpmath.log(abs(mpmath.gamma(x)))


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
    interesting_points = [
        # Non-positive integer poles of digamma(x):
        *(float(-k) for k in range(21)),
        -100.0,
        -1000.0,
        # Roots of digamma(x) = 0, computed via mpmath.findroot(mpmath.digamma, x0):
        1.4616321449683623,  # unique positive root on (0, inf)
        -0.5040830082644554,  # root in (-1, 0)
        -1.5734984731623905,  # root in (-2, -1)
        -2.6107208684441446,  # root in (-3, -2)
        -3.635293366436901,  # root in (-4, -3)
        -4.653237761743142,  # root in (-5, -4)
        # Recurrence-to-asymptotic series switch thresholds:
        8.0,
        10.0,
    ]
    util.check_unary_precision(
        self,
        lax.digamma,
        scipy.special.psi,
        _mpmath_digamma,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
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
    interesting_points = [
        # Non-positive integer poles of gamma(x):
        *(float(-k) for k in range(21)),
        -100.0,
        -1000.0,
        # Positive zero crossings where gamma(1) = gamma(2) = 1:
        1.0,
        2.0,
        # Local minimum of gamma(x) on (0, inf), where digamma(x) = 0
        # (computed via mpmath.findroot(mpmath.digamma, 1.5)):
        1.4616321449683623,
        # Negative zero crossings where |gamma(x)| = 1 (lgamma(x) = 0), computed
        # via mpmath.findroot(lambda x: abs(mpmath.gamma(x)) - 1, x0):
        -2.151610527686942,  # first root in (-3, -2)
        -2.8115278800280976,  # second root in (-3, -2)
        -3.1435830219538416,  # first root in (-4, -3)
        -3.9533289185258997,  # second root in (-4, -3)
        -4.026705717226458,  # first root in (-5, -4)
        -4.990151042516608,  # second root in (-5, -4)
        # Stirling / Lanczos asymptotic switch thresholds:
        8.0,
        10.0,
        # Overflow thresholds where lgamma(x) == finfo.max, computed via
        # mpmath.findroot(lambda x: mpmath.loggamma(x) - finfo.max, x0):
        2.556348e36,  # float32 overflow threshold
        2.559983327851638e305,  # float64 overflow threshold
    ]
    util.check_unary_precision(
        self,
        lax.lgamma,
        scipy.special.gammaln,
        _mpmath_lgamma,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


util.register_benchmark(lax.digamma)
util.register_benchmark(lax.lgamma)


if __name__ == "__main__":
  util.main()
