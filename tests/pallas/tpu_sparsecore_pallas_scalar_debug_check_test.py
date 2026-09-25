# Copyright 2025 The JAX Authors.
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
"""SparseCore Pallas tests for scalar debug check assertions."""

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


config.parse_flags_with_absl()


class ScalarDebugCheckTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")

    super().setUp()

  def test_scalar_debug_check(self):
    if not jtu.is_device_tpu_at_least(8):
      # TODO: b/469486032 - Figure out why the test gets stuck on v5p, v6e, and v7.
      self.skipTest("Fails on v5p, v6e, and v7.")

    x = jnp.arange(8)

    @pl.kernel(
        out_type=x,
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
    )
    def kernel(_):
      pl.debug_check(True, "Check success!")
      pl.debug_check(False, "Check failure!")

    with config.jax_pallas_enable_debug_checks(True), self.assertRaises(
        jax.errors.JaxRuntimeError
    ) as error:
      jax.block_until_ready(kernel())

    self.assertNotIn("Check success!", str(error.exception))
    self.assertIn("Check failure!", str(error.exception))
    self.assertIn(
        "check at ScalarDebugCheckTest.test_scalar_debug_check",
        str(error.exception),
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
