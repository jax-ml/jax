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
import jax.scipy as jsp

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
    # TODO(phawkins): In float64, chlo.lgamma has large errors near negative
    # zero-crossings (where |gamma(x)| = 1, e.g. x ~= -3.14358).
    bounds = [
        ("cpu", {f16: 1.0, f32: 4.0, f64: math.inf}),
        ("gpu", {f16: 1.0, f32: 4.0, f64: math.inf}),
        ("tpu", {f16: 1.0, f32: 4.0}),
    ]
    ref_fn = lambda x: np.where(x == -np.inf, np.inf, scipy.special.gammaln(x))
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
        ref_fn,
        _mpmath_lgamma,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


def _with_special_values(a, y, x, x_at_0, x_at_1):
  """Sets the documented special values of the JAX version in scipy's `x`.

  The inverse is `x_at_0` at y = 0 and `x_at_1` at y = 1 for any a >= 0, and
  nan for y outside [0, 1], for a < 0, and for a in {0, inf} with 0 < y < 1.
  scipy is inconsistent for a = 0 and for some y < 0, for example
  `scipy.special.gammaincinv(0.001, -6e-8) == 0`.
  """
  x = np.where(y == 0, x_at_0, np.where(y == 1, x_at_1, x))
  interior = (y > 0) & (y < 1)
  invalid = (
      np.isnan(a) | np.isnan(y) | (a < 0) | (y < 0) | (y > 1)
      | (((a == 0) | np.isinf(a)) & interior)
  )
  return np.where(invalid, np.nan, x)


def _gammaincinv_ref(a, y):
  x = scipy.special.gammaincinv(a, y)
  return _with_special_values(a, y, x, x_at_0=0.0, x_at_1=np.inf)


def _gammainccinv_ref(a, y):
  x = scipy.special.gammainccinv(a, y)
  return _with_special_values(a, y, x, x_at_0=np.inf, x_at_1=0.0)


def _ignored_inputs(a, y, x):
  """Inputs (a, y) with reference result x excluded from the accuracy checks."""
  finfo = jnp.finfo(a.dtype)
  tiny = float(finfo.tiny)
  # Whether subnormal a is flushed to zero, which changes the result from 0
  # (it underflows for all but subnormal y) to nan, depends on the platform.
  # So does whether a negative subnormal y compares as zero (giving the result
  # at y = 0) or as negative (giving nan): GPUs flush subnormals in arithmetic
  # but not in comparisons.
  ignored = (np.abs(a) > 0) & (np.abs(a) < tiny)
  ignored |= (np.abs(y) > 0) & (np.abs(y) < tiny)
  # Within a factor 1 / eps of the smallest normal number, the density of the
  # gamma distribution at the result underflows.
  near_underflow = tiny / float(finfo.eps)
  for v in (a, y, x):
    ignored |= (v > 0) & (v < near_underflow)
  if a.dtype == np.float64:
    # For finite a >= 1e6 in float64, the initial approximation is used without
    # refinement (relative error up to about 1e-6), and scipy's relative error
    # also grows beyond 1e-7 (checked against mpmath for a >= 1e7).
    ignored |= (a >= 1e6) & np.isfinite(a)
  return ignored


def _gammaincinv_ignored(a, y):
  with np.errstate(all="ignore"):
    x = scipy.special.gammaincinv(a.astype(np.float64), y.astype(np.float64))
  return _ignored_inputs(a, y, x)


def _gammainccinv_ignored(a, y):
  with np.errstate(all="ignore"):
    x = scipy.special.gammainccinv(a.astype(np.float64), y.astype(np.float64))
  return _ignored_inputs(a, y, x)


# (a, y) inputs at the special values set by _with_special_values.
_GAMMAINCINV_SPECIAL_POINTS = [
    (2.0, 0.0),
    (2.0, 1.0),
    (0.0, 0.0),
    (0.0, 1.0),
    (math.inf, 0.0),
    (math.inf, 1.0),
    (0.0, 0.5),
    (math.inf, 0.5),
    (-1.0, 0.5),
    (2.0, -0.1),
    (2.0, 1.1),
    (2.0, -math.inf),
    (2.0, math.inf),
    (2.0, math.nan),
    (math.nan, 0.5),
    (math.nan, 0.0),
]

_GAMMAINCINV_INTERESTING_POINTS = [
    *_GAMMAINCINV_SPECIAL_POINTS,
    # Thresholds in a of the initial approximation (a < 0.3, a < 1, a >= 500)
    # and of the Halley refinement (a < 200 in float32, a < 1e6 in float64):
    0.3,
    200.0,
    500.0,
    1e6,
    # Values of y near thresholds of the initial approximation for a < 1,
    # which are on b = (1 - p) * Gamma(a), close to 1 - p for a near 1:
    1e-5,
    0.15,
    0.35,
    0.6,
]


@jtu.thread_unsafe_test_class()
class GammaincinvTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gammaincinv_accuracy(self, dtype):
    # mpmath has no inverse incomplete gamma function, so scipy is the
    # reference for all dtypes. In float32 the accuracy for small a is limited
    # by that of gammainc. The result is refined against lax.igamma, so on GPU
    # and TPU it inherits the accuracy of igamma there, amplified by about 1 / a
    # for small a.
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 600.0, f64: 14500.0}),
        (["gpu", "tpu"],
         {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        jsp.special.gammaincinv,
        _gammaincinv_ref,
        mpmath_fn=None,
        dtype=dtype,
        nargs=2,
        bounds=bounds,
        ignore_inputs=_gammaincinv_ignored,
        interesting_points=_GAMMAINCINV_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class GammainccinvTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gammainccinv_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 1250.0, f64: 16500.0}),
        (["gpu", "tpu"],
         {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        jsp.special.gammainccinv,
        _gammainccinv_ref,
        mpmath_fn=None,
        dtype=dtype,
        nargs=2,
        bounds=bounds,
        ignore_inputs=_gammainccinv_ignored,
        interesting_points=_GAMMAINCINV_INTERESTING_POINTS,
    )


def _scipy_gamma(x):
  # scipy.special.gamma returns +0.0 when it underflows for negative x.
  r = scipy.special.gamma(x)
  return np.where(r == 0, np.where(np.floor(x) % 2 == 1, -0.0, 0.0), r)


def _mpmath_gamma(x):
  if x == 0:
    return -mpmath.inf if np.signbit(float(x)) else mpmath.inf
  if (x < 0 and mpmath.isint(x)) or x == -mpmath.inf:
    return mpmath.nan
  return mpmath.gamma(x)


@jtu.thread_unsafe_test_class()
class GammaTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gamma_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 10.0, f64: 7.0}),
        ("gpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("tpu", {bf16: math.inf, f16: math.inf, f32: math.inf}),
    ]
    interesting_points = [
        # Non-positive integer poles of gamma(x):
        *(float(-k) for k in range(21)),
        -100.0,
        -1000.0,
        # Horner switch from z to 1/z:
        10.0,
        # Largest n with finite (n - 1)! in float32 and float64:
        35.0,
        171.0,
        # Overflow thresholds where gamma(x) == finfo.max, computed via
        # mpmath.findroot(lambda x: mpmath.loggamma(x) - log(finfo.max), x0):
        35.040098403112816,  # float32
        171.6243769563027,  # float64
    ]
    util.check_unary_precision(
        self,
        jsp.special.gamma,
        _scipy_gamma,
        _mpmath_gamma,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


util.register_benchmark(lax.digamma)
util.register_benchmark(lax.lgamma)
util.register_benchmark(jsp.special.gammaincinv, nargs=2)
util.register_benchmark(jsp.special.gammainccinv, nargs=2)
util.register_benchmark(jsp.special.gamma, name="gamma")


if __name__ == "__main__":
  util.main()
