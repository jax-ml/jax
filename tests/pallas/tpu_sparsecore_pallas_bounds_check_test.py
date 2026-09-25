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
"""SparseCore Pallas tests with runtime bounds checking assertions."""

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


config.parse_flags_with_absl()


class BoundsCheckTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")

    super().setUp()

  def test_trigger_bounds_checker(self):
    size = plsc.get_sparse_core_info().num_lanes
    x = jnp.arange(size, dtype=jnp.int32)

    @pl.kernel(
        out_type=x,
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=1
        ),
        scratch_types=dict(
            x_ref=pltpu.VMEM.like(x),
            indices_ref=pltpu.VMEM.like(x),
            o_ref=pltpu.VMEM.like(x),
        ),
    )
    def kernel(
        x_hbm_ref, indices_hbm_ref, o_hbm_ref, *, x_ref, indices_ref, o_ref
    ):
      pltpu.sync_copy((x_hbm_ref, indices_hbm_ref), (x_ref, indices_ref))
      o_ref[...] = plsc.load_gather(x_ref, [indices_ref[...]])
      pltpu.sync_copy(o_ref, o_hbm_ref)

    compiled_kernel = jax.jit(
        kernel, compiler_options=dict(xla_sc_assert_level="all-loads-stores")
    )

    # 1. Test in-bounds access succeeds without errors.
    if not jtu.is_device_tpu(7, "x"):
      indices_valid = jnp.arange(size, dtype=jnp.int32)
      jax.block_until_ready(compiled_kernel(x, indices_valid))

    # 2. Test out-of-bounds access raises JaxRuntimeError and halts.
    indices_oob = jnp.arange(size, dtype=jnp.int32) + jnp.int32(128)
    with config.jax_pallas_enable_debug_checks(True), self.assertRaises(
        jax.errors.JaxRuntimeError
    ) as error:
      jax.block_until_ready(compiled_kernel(x, indices_oob))

    # TODO(b/479427406): Remove this once the bug is fixed.
    if not (jtu.is_cloud_tpu() and jtu.is_device_tpu_at_least(7)):
      self.assertIn(
          "Trying to perform an indexed vector load from out of bounds"
          " address.",
          str(error.exception),
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
