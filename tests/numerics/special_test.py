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

"""Precision tests for special functions against reference implementations."""

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
import scipy.special

config.parse_flags_with_absl()


bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


def _mpmath_erfc(x):
  """Evaluates erfc with magnitude guards to prevent mpmath hanging on |x| > 100."""
  if x > 100.0:
    return mpmath.mpf(0)
  if x < -100.0:
    return mpmath.mpf(2)
  return mpmath.erfc(x)


def _erfinv_reference(x: np.ndarray) -> np.ndarray:
  """Evaluates erfinv with a domain guard to avoid slow C++ exception handling."""
  return np.where(
      np.abs(x) <= 1.0,
      np.copysign(scipy.special.erfinv(np.clip(x, -1.0, 1.0)), x),
      np.nan,
  )


@jtu.skip_under_pytest("Only runs under Bazel")
@jtu.thread_unsafe_test_class()
class SpecialTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i0e_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 7.0, f64: 7.5}),
        ("gpu", {f32: 8.0, f64: 7.5}),
        ("tpu", {f32: 8.0}),
    ]
    util.check_unary_precision(
        self, lax.bessel_i0e, scipy.special.i0e,
        lambda x: mpmath.besseli(0, x) * mpmath.exp(-abs(x)), dtype,
        bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i1e_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 11.0, f64: 10.5}),
        ("gpu", {f16: 1.0, f32: 15.5, f64: 6.0}),
        ("tpu", {f16: 1.0, f32: 15.5}),
    ]
    ref_fn = lambda x: np.copysign(scipy.special.i1e(x), x)
    util.check_unary_precision(
        self, lax.bessel_i1e, ref_fn,
        lambda x: mpmath.besseli(1, x) * mpmath.exp(-abs(x)), dtype,
        bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_digamma_accuracy(self, dtype):
    # TODO(phawkins): Digamma has poles at non-positive integers (0, -1, -2, ...)
    # where the function is ill-conditioned and outputs differ across
    # implementations (NaN vs +-inf).
    bounds = [
        ("cpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("gpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("tpu", {bf16: math.inf, f16: math.inf, f32: math.inf}),
    ]
    util.check_unary_precision(
        self, lax.digamma, scipy.special.psi, mpmath.digamma, dtype,
        bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erf_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 131.0, f32: 7.0, f64: 2.5}),
        ("gpu",
         {bf16: 16.0, f16: 131.0, f32: 1076922.0, f64: 576922644178748.5}),
        (TPU_EUPV1, {f16: 131.0, f32: 7.5}),
        ("tpu_v5p", {f16: 131.0, f32: 8.5}),
        (["tpu_v6e", "tpu_7x"], {f16: 131.0, f32: 1.5}),
    ]
    util.check_unary_precision(
        self, lax.erf, scipy.special.erf, mpmath.erf, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfc_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 66.0, f64: 256.0}),
        ("gpu", {f16: 1.0, f32: 66.5, f64: 256.0}),
        (TPU_EUPV1, {f16: 1.0, f32: 145.0}),
        ("tpu_v5p", {f16: 1.0, f32: 157.0}),
        ("tpu_v6e", {f16: 1.0, f32: 124.5}),
        ("tpu_7x", {f16: 1.0, f32: 125.0}),
    ]
    util.check_unary_precision(
        self, lax.erfc, scipy.special.erfc, _mpmath_erfc, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfinv_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 65.0, f64: 82.5}),
        ("gpu", {f16: 1.0, f32: 65.0, f64: 83.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 427.0}),
        (["tpu_v5p", "tpu_7x"], {f16: 1.0, f32: 65.5}),
        ("tpu_v6e", {f16: 1.0, f32: 65.0}),
    ]
    util.check_unary_precision(
        self, lax.erf_inv, _erfinv_reference, mpmath.erfinv, dtype,
        bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_lgamma_accuracy(self, dtype):
    # TODO(phawkins): Large errors occur near zero-crossings (where |gamma(x)| = 1,
    # e.g. x ~= -3.14358) where small approximation errors cause sign flips across
    # zero, and at x = -inf due to inf/NaN handling differences.
    bounds = [
        ("cpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("gpu", {bf16: math.inf, f16: math.inf, f32: math.inf, f64: math.inf}),
        ("tpu", {bf16: math.inf, f16: math.inf, f32: math.inf}),
    ]
    util.check_unary_precision(
        self, lax.lgamma, scipy.special.gammaln,
        lambda x: mpmath.log(abs(mpmath.gamma(x))), dtype, bounds=bounds)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
