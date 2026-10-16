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
import numpy as np
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
  with mpmath.workprec(200):
    return +mpmath.log(abs(mpmath.gamma(x)))


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
    bounds = [
        ("cpu", {f16: 1.0, f32: 2.5, f64: 2.0}),
        ("gpu", {f16: 1.0, f32: 4.0, f64: 2.0}),
        ("tpu", {f16: 1.0, f32: 4.0}),
    ]
    ref_fn = lambda x: np.where(x == -np.inf, np.inf, scipy.special.gammaln(x))
    neg_roots = [
        # Negative zero crossings where |gamma(x)| = 1 (lgamma(x) = 0), computed
        # via mpmath.findroot(lambda x: abs(mpmath.gamma(x)) - 1, x0):
        -2.4570247382208006,
        -2.7476826467274127,
        -3.14358088834998,
        -3.955294284858598,
        -4.039361839740537,
        -4.991544640560048,
        -5.0082181683225935,
        -5.998607480080875,
        -6.001385294453155,
        -6.999801507890638,
        -7.000198333407325,
        -7.999975197095821,
        -8.000024800270682,
        -8.999997244250977,
        -9.000002755714823,
        -9.99999972442663,
        -10.000000275573013,
        -10.99999997494789,
        -11.000000025052106,
        -11.999999997912324,
        -12.000000002087676,
        -12.99999999983941,
        -13.00000000016059,
        -13.99999999998853,
        -14.00000000001147,
        -14.999999999999236,
        -15.000000000000764,
        -15.999999999999952,
        -16.000000000000046,
    ]
    # Root-window switch boundaries |w| = 0.06 for neg_roots[0] (delta / r in
    # {47/53, 53/47}) and |w| = 0.08 for the rest (delta / r in {23/27, 27/23})
    # where negative-root rational approximations hand off to Euler reflection.
    neg_root_boundaries = [
        round(x) + (x - round(x)) * factor
        for i, x in enumerate(neg_roots)
        for factor in (
            (47.0 / 53.0, 53.0 / 47.0)
            if i == 0
            else (23.0 / 27.0, 27.0 / 23.0)
        )
    ]
    interesting_points = [
        # Non-positive integer poles of gamma(x) and large negative half-integer:
        *(float(-k) for k in range(21)),
        -100.0,
        -1000.0,
        -4503599627370495.5,  # -2**52 + 0.5 (last negative non-integer in f64)
        -4503599627370496.0,  # -2**52 (all f64 below this are even integers)
        # Positive zero crossings where gamma(1) = gamma(2) = 1, plus offsets
        # where (x - 1) and (x - 2) cross half-precision scale:
        1.0,
        2.0,
        1.0 - 2**-26,
        1.0 + 2**-26,
        2.0 - 2**-26,
        2.0 + 2**-26,
        # Local minimum of gamma(x) on (0, inf), where digamma(x) = 0
        # (computed via mpmath.findroot(mpmath.digamma, 1.5)):
        1.4616321449683623,
        *neg_roots,
        *neg_root_boundaries,
        # Polynomial and recurrence-to-asymptotic switch thresholds:
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
        # Overflow thresholds where lgamma(x) == finfo.max, computed via
        # mpmath.findroot(lambda x: mpmath.loggamma(x) - finfo.max, x0), plus
        # values just below the float64 overflow boundary:
        2.556348e36,  # float32 overflow threshold
        2.556348163871691e305,  # finite in float64, just above 2.556348e305
        2.559983327851638e305,  # float64 overflow threshold
    ]
    util.check_unary_precision(
        self,
        lax.lgamma,
        ref_fn,
        _mpmath_lgamma,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


util.register_benchmark(lax.digamma)
util.register_benchmark(lax.lgamma)


if __name__ == "__main__":
  util.main()
