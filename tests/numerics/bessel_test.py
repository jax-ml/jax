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

"""Precision tests for modified Bessel functions against reference implementations."""

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


@jtu.thread_unsafe_test_class()
class BesselI0eTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i0e_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 7.0, f64: 7.5}),
        ("gpu", {f32: 8.0, f64: 7.5}),
        ("tpu", {f32: 8.0}),
    ]
    util.check_unary_precision(
        self,
        lax.bessel_i0e,
        scipy.special.i0e,
        lambda x: mpmath.besseli(0, x) * mpmath.exp(-abs(x)),
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class BesselI1eTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i1e_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 11.0, f64: 10.5}),
        ("gpu", {f16: 1.0, f32: 15.5, f64: 6.0}),
        ("tpu", {f16: 1.0, f32: 15.5}),
    ]
    ref_fn = lambda x: np.copysign(scipy.special.i1e(x), x)
    util.check_unary_precision(
        self,
        lax.bessel_i1e,
        ref_fn,
        lambda x: mpmath.besseli(1, x) * mpmath.exp(-abs(x)),
        dtype,
        bounds=bounds,
    )


if __name__ == "__main__":
  absltest.main(testLoader=util.ClassShardedTestLoader())
