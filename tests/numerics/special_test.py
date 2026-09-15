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
tpu_devices = [
    "tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i",
    "tpu_v5e", "tpu_v5p", "tpu_v6e", "tpu_7x",
]


@jtu.skip_under_pytest("Only runs under Bazel")
@jtu.thread_unsafe_test_class()
class SpecialTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i0e_accuracy(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 7, f64: 8}),
        (["gpu"], {f32: 8, f64: 2}),
        (tpu_devices, {f32: 8}),
    ]
    util.check_unary_precision(
        self,
        lax.bessel_i0e,
        lambda x: mpmath.besseli(0, x) * mpmath.exp(-abs(x)),
        dtype,
        bounds=bounds,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i1e_accuracy(self, dtype):
    bounds = [
        (["cpu"], {bf16: 1, f16: 1, f32: 11, f64: 11}),
        (["gpu"], {bf16: 1, f16: 1, f32: 15, f64: 2}),
        (tpu_devices, {bf16: 1, f16: 1, f32: 15}),
    ]
    util.check_unary_precision(
        self,
        lax.bessel_i1e,
        lambda x: mpmath.besseli(1, x) * mpmath.exp(-abs(x)),
        dtype,
        bounds=bounds,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_digamma_accuracy(self, dtype):
    # TODO(phawkins): Digamma has poles at non-positive integers (0, -1, -2, ...)
    # where the function is ill-conditioned and outputs differ across
    # implementations (NaN vs +-inf).
    bounds = [
        (["cpu"], {bf16: 18446744073709551615, f16: 18446744073709551615, f32: 18446744073709551615, f64: 18446744073709551615}),
        (["gpu"], {bf16: 18446744073709551615, f16: 18446744073709551615, f32: 18446744073709551615, f64: 18446744073709551615}),
        (tpu_devices, {bf16: 18446744073709551615, f16: 18446744073709551615, f32: 18446744073709551615}),
    ]
    util.check_unary_precision(
        self, lax.digamma, mpmath.digamma, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erf_accuracy(self, dtype):
    bounds = [
        (["cpu"], {f16: 131, f32: 7, f64: 3}),
        (["gpu"], {bf16: 16, f16: 131, f32: 1076922, f64: 576922644178748}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {f16: 131, f32: 7}),
        (["tpu_v5p"], {f16: 131, f32: 8}),
        (["tpu_v6e", "tpu_7x"], {f16: 131, f32: 1}),
    ]
    util.check_unary_precision(
        self, lax.erf, mpmath.erf, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfc_accuracy(self, dtype):
    bounds = [
        (["cpu"], {f16: 1, f32: 66, f64: 13}),
        (["gpu"], {f16: 1, f32: 66, f64: 4}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {f16: 1, f32: 145}),
        (["tpu_v5p"], {f16: 1, f32: 157}),
        (["tpu_v6e"], {f16: 1, f32: 124}),
        (["tpu_7x"], {f16: 1, f32: 125}),
    ]
    util.check_unary_precision(
        self, lax.erfc, mpmath.erfc, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfinv_accuracy(self, dtype):
    bounds = [
        (["cpu"], {bf16: 1, f16: 1, f32: 65, f64: 83}),
        (["gpu"], {bf16: 1, f16: 1, f32: 65, f64: 83}),
        (["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"], {bf16: 1, f16: 1, f32: 427}),
        (["tpu_v5p"], {bf16: 1, f16: 1, f32: 65}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1, f16: 1, f32: 65}),
    ]
    util.check_unary_precision(
        self, lax.erf_inv, mpmath.erfinv, dtype, bounds=bounds
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_lgamma_accuracy(self, dtype):
    # TODO(phawkins): In float64, chlo.lgamma has large errors near negative
    # zero-crossings (where |gamma(x)| = 1, e.g. x ~= -3.14358).
    bounds = [
        (["cpu"], {bf16: 0, f16: 1, f32: 1, f64: 18446744073709551615}),
        (["gpu"], {bf16: 0, f16: 1, f32: 1, f64: 18446744073709551615}),
        (tpu_devices, {bf16: 0, f16: 1, f32: 1}),
    ]
    input_ftz = [
        (["cpu", "gpu"], {bf16: False, f16: False, f32: False}),
        (tpu_devices, {f16: False, f32: False}),
    ]
    util.check_unary_precision(
        self,
        lax.lgamma,
        lambda x: (
            mpmath.inf
            if x == -mpmath.inf
            else mpmath.log(abs(mpmath.gamma(x)))
        ),
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        ref_fn=lambda x: np.where(x == -np.inf, np.inf, scipy.special.gammaln(x)),
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
