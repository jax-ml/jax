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

"""Tests for Pallas mesh API."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.state import discharge as state_discharge
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class ShmallasTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supported on TPU v4+")

  def test_can_create_tensorcore_mesh(self):
    _ = pltpu.create_tensorcore_mesh("x")

  def test_can_trivially_shard_map_with_pallas_mesh(self):
    mesh = pltpu.create_tensorcore_mesh("x")
    _ = shard_map.shard_map(lambda: None, mesh, in_specs=(), out_specs=None)()

  def test_can_run_basic_pallas_kernel_with_shard_map(self):
    mesh = pltpu.create_tensorcore_mesh("x")

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        def kernel():
          def alloc(sem):
            pltpu.async_copy(x_ref, y_ref, sem).wait()
          pl.run_scoped(alloc, pltpu.SemaphoreType.DMA)
        shard_map.shard_map(kernel, mesh, in_specs=(), out_specs=None,
                            check_rep=False)()
      _, y = state_discharge.run_state(inner)((x, y))
      return y
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_can_query_core_index_pallas_kernel_with_shard_map(self):
    mesh = pltpu.create_tensorcore_mesh("x")

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        def kernel():
          num_cores = jax.lax.psum(1, "x")
          slc_size = 16 // num_cores
          def alloc(x_vmem_ref, y_vmem_ref, sem):
            core_index = jax.lax.axis_index("x")
            slc = pl.ds(core_index * slc_size, slc_size)
            pltpu.async_copy(
                x_ref.at[slc],
                x_vmem_ref,
                sem,
            ).wait()
            y = x_vmem_ref[...] + jax.lax.axis_index("x")
            y_vmem_ref[...] = y
            pltpu.async_copy(y_vmem_ref, y_ref.at[slc], sem).wait()
          pl.run_scoped(
              alloc,
              pltpu.VMEM((slc_size, 128), x_ref.dtype),
              pltpu.VMEM((slc_size, 128), y_ref.dtype),
              pltpu.SemaphoreType.DMA,
          )
        shard_map.shard_map(kernel, mesh, in_specs=(), out_specs=None,
                            check_rep=False)()
      _, y = state_discharge.run_state(inner)((x, y))
      return y
    num_cores = jax.devices()[0].num_cores
    x = jnp.arange(16 * 128, dtype=jnp.int32).reshape((16, 128))
    expected_out = (
        x.reshape((num_cores, -1, 128)) + jnp.arange(num_cores)[..., None, None]
    ).reshape(x.shape)
    y = f(x)
    np.testing.assert_array_equal(y, expected_out)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
