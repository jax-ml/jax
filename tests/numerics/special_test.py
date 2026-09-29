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
import jax
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
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


def _mpmath_erfc(x):
  """Evaluates erfc with magnitude guards to prevent mpmath hanging on |x| > 100."""
  if x > 100.0:
    return mpmath.mpf(0)
  if x < -100.0:
    return mpmath.mpf(2)
  return mpmath.erfc(x)


def _mpmath_erfcx(x):
  if x < -30.0:
    return mpmath.inf
  if x < 1.0:
    return mpmath.exp(x * x) * mpmath.erfc(x)
  return mpmath.hyperu(0.5, 0.5, x * x) / mpmath.sqrt(mpmath.pi)


def erfcx_grad(x):
  return jax.vmap(jax.grad(jsp.special.erfcx))(x)


def _mpmath_erfcx_grad(x):
  if x < -30.0:
    return -mpmath.inf
  if x < 1.0:
    return 2 * x * _mpmath_erfcx(x) - 2 / mpmath.sqrt(mpmath.pi)
  return -mpmath.hyperu(1.0, 0.5, x * x) / mpmath.sqrt(mpmath.pi)


def _erfcx_grad_reference(x: np.ndarray) -> np.ndarray:
  sx = np.where(x >= 1.0, 0.0, x)
  z = np.square(np.where(x >= 1.0, np.minimum(x, 1e75), 1.0))
  direct = 2.0 * sx * scipy.special.erfcx(sx) - 2.0 / np.sqrt(np.pi)
  asymp = -scipy.special.hyperu(1.0, 0.5, z) / np.sqrt(np.pi)
  return np.where(x >= 1e75, -0.0, np.where(x >= 1.0, asymp, direct))


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
  def test_bessel_i0e_test_accuracy(self, dtype):
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
  def test_bessel_i1e_test_accuracy(self, dtype):
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
  def test_digamma_test_accuracy(self, dtype):
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
  def test_erf_test_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 7.0, f64: 2.5}),
        ("gpu", {bf16: 0.5, f16: (0.5, 1.0), f32: 6.5, f64: 2.5}),
        (TPU_EUPV1, {f16: 0.5, f32: 7.5}),
        ("tpu_v5p", {f16: 0.5, f32: 8.5}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self, lax.erf, scipy.special.erf, mpmath.erf, dtype,
        bounds=bounds, input_ftz=input_ftz)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfc_test_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 66.0, f64: 350.0}),
        ("gpu", {f16: 1.0, f32: 66.5, f64: 350.0}),
        (TPU_EUPV1, {f16: 1.0, f32: 145.0}),
        ("tpu_v5p", {f16: 1.0, f32: 157.0}),
        ("tpu_v6e", {f16: 1.0, f32: 124.5}),
        ("tpu_7x", {f16: 1.0, f32: 125.0}),
    ]
    util.check_unary_precision(
        self, lax.erfc, scipy.special.erfc, _mpmath_erfc, dtype, bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfcx_test_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 64.5, f64: 350.0}),
        ("gpu", {f16: 1.0, f32: 65.0, f64: 350.0}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 214.0}),
        ("tpu_v5p", {f16: 1.0, f32: 155.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 125.5}),
    ]
    # Ignore x = -9.382414: near the float32 overflow threshold, where the
    # true value is finite (~3.40282e+38) but x^2 rounds up across
    # log(fmax / 2), causing exp(x^2) * erfc(x) to overflow float32 to inf.
    ignore_inputs = [
        (["cpu", "gpu", "tpu"], {f32: [0xC1161E5E]}),
    ]
    util.check_unary_precision(
        self, jsp.special.erfcx, scipy.special.erfcx, _mpmath_erfcx, dtype,
        bounds=bounds, ignore_inputs=ignore_inputs)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfcx_grad_test_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 400.0, f64: 500.0}),
        ("gpu", {f16: 1.0, f32: 400.0, f64: 500.0}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.5, f32: 8000.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 6000.0}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 350.0}),
        ("tpu_7x", {f16: 1.0, f32: 350.0}),
    ]
    util.check_unary_precision(
        self, erfcx_grad, _erfcx_grad_reference, _mpmath_erfcx_grad, dtype,
        bounds=bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfcx_test_probes(self, dtype):
    # Probe regime thresholds and erfc(x) underflow gaps that sampling may miss.
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    x = jnp.concatenate([
        jnp.linspace(7.9, 8.1, 32, dtype=dtype),        # f32 regime threshold
        jnp.linspace(9.195, 9.419, 32, dtype=dtype),    # f32 erfc underflow gap
        jnp.linspace(11.9, 12.1, 32, dtype=dtype),      # f64 regime threshold
        jnp.linspace(26.543, 26.642, 32, dtype=dtype),  # f64 erfc underflow gap
    ])
    x_np = np.asarray(x)
    for jax_fn, mp_fn, max_ulp in [
        (jsp.special.erfcx, _mpmath_erfcx, 350.0),
        (erfcx_grad, _mpmath_erfcx_grad, 8000.0),
    ]:
      actual = np.asarray(jax.jit(jax_fn)(x))
      ref = np.array(
          [util.eval_mpmath(mp_fn, v.item(), dtype=dtype) for v in x_np],
          dtype=object if dtype == f64 else np.float64,
      )
      self.assertLessEqual(np.max(util.ulp_diff(actual, ref, dtype)), max_ulp)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfinv_test_accuracy(self, dtype):
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
  def test_lgamma_test_accuracy(self, dtype):
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
