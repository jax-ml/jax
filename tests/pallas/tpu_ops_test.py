# Copyright 2024 The JAX Authors.
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

"""Tests for common JAX operations within pallas_call."""

import functools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl

# Import mosaic for flag definitions
from jax.experimental import mosaic as _  # noqa: F401


class TpuOpsTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test requires TPU device.")

    super().setUp()

  @parameterized.parameters([-3.2, -1.0, -0.4, 0., 0.72, 1.0, 2.4])
  def test_erf_inv(self, x):
    @jax.jit
    @functools.partial(
        pl.pallas_call,
        # TODO(ayx): add float64 support for `erf_inv`
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = lax.erf_inv(x_ref[...])

    x = jnp.full((4,), x)
    out = kernel(x)
    expected = lax.erf_inv(x)
    np.testing.assert_array_equal(out, expected)

  SIGN_PARAMS = [
    (jnp.int32, (-3, 0, 5)),
    (jnp.uint32, (0, 5)),
    (jnp.float32, (-3.2, -0., 0., 5.1, jnp.nan, jnp.inf, -jnp.inf)),
  ]

  @parameterized.named_parameters(
      (f"{dtype.__name__}_{value}", dtype, value)
      for dtype, values in SIGN_PARAMS
      for value in values
  )
  def test_sign(self, dtype, value):
    @jax.jit
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), dtype),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.sign(x_ref[...])

    x = jnp.full((4,), value, dtype=dtype)
    out = kernel(x)
    expected = jnp.sign(x)
    np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
