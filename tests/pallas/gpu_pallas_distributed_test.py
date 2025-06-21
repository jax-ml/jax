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

"""Tests for distributed pallas GPU operations."""

import functools
import os

import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src import test_multiprocess as jt_multiprocess
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.experimental.mosaic.gpu as mgpu
import jax.numpy as jnp
import numpy as np


P = jax.sharding.PartitionSpec
partial = functools.partial


class PallasCallRemoteDMATest(jt_multiprocess.MultiProcessTest):

  def setUp(self):
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    if not mgpu.supports_cross_device_collectives():
      self.skipTest("NVSHMEM library unavailable.")
    if jax.process_count() == 1:
      self.skipTest("Test requires multiple processes.")
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "") == "platform":
      self.skipTest("NVSHMEM doesn't work with the platform allocator.")
    super().setUp()

  def test_basic_remote_dma(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    def kernel(x_ref, y_ref, ready_sem, recv_sem):
      other_dev_id = 1 - lax.axis_index('x')
      y_ref[...] = x_ref[...]
      pl.semaphore_signal(ready_sem, device_id=other_dev_id,
                          device_id_type=pl.DeviceIdType.LOGICAL)
      pl.semaphore_wait(ready_sem)
      neighbor_ptr = plgpu.remote_ref(
          y_ref, other_dev_id, device_id_type=pl.DeviceIdType.LOGICAL
      )
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id,
                          device_id_type=pl.DeviceIdType.LOGICAL)
      pl.semaphore_wait(recv_sem)

    x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[
              plgpu.SemaphoreType.REGULAR,
              plgpu.SemaphoreType.REGULAR,
          ],
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh, in_specs=P('x'), out_specs=P('x'), check_rep=False,
        )
    )(x)

    expected = x[8:] if jax.process_index() == 0 else x[:8]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  def test_wait_twice(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 2, device_id=other_dev_id,
                          device_id_type=pl.DeviceIdType.LOGICAL)
      pl.semaphore_wait(sem)
      pl.semaphore_wait(sem)
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_call = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            kernel_call, mesh, in_specs=(), out_specs=P(None), check_rep=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

  def test_permuted_mesh(self):
    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 1, device_id=other_dev_id,
                          device_id_type=pl.DeviceIdType.LOGICAL)
      pl.semaphore_wait(sem)

    kernel_call = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )
    mesh = jax.sharding.Mesh(jax.devices()[::-1], ['x'])  # Reverse the devices.
    f = jax.jit(
        shard_map.shard_map(
            kernel_call, mesh, in_specs=(), out_specs=P(None), check_rep=False,
        )
    )
    msg = (
        'Mosaic GPU only supports meshes with device ordering that follows'
        ' row-major device ids.'
    )
    with self.assertRaisesRegex(NotImplementedError, msg):
      f()


if __name__ == '__main__':
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  jt_multiprocess.main()
