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
"""Tests for Pallas on SparseCore with multiple devices."""
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class PallasCallRemoteDMATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('SparseCore only supported on TPU v5+')

  @parameterized.product(direction=['left', 'right'], num_devices=[2, None])
  def test_collective_permute_1d(self, direction, num_devices):
    shape = (8, 128)

    # Implements a very simple collective permute.
    @plsc.kernel(
        out_shape=jax.ShapeDtypeStruct(shape, jnp.int32),
        mesh=plsc.ScalarSubcoreMesh(axis_name='core', num_cores=1),
        scratch_shapes=(
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
    )
    def kernel(x_ref, y_ref, ready_sem, send_sem, recv_sem):

      my_id = lax.axis_index('x')
      axis_size = lax.axis_size('x')
      if direction == 'right':
        neighbor = lax.rem(my_id + 1, axis_size)
      else:
        neighbor = lax.rem(my_id + axis_size - 1, axis_size)
      pltpu.semaphore_signal(ready_sem, device_id=neighbor)
      pltpu.semaphore_wait(ready_sem)
      pltpu.async_remote_copy(
          x_ref, y_ref, send_sem, recv_sem, device_id=neighbor
      ).wait()

    num_devices = num_devices or jax.device_count()
    x = jnp.arange(num_devices * math.prod(shape), dtype=jnp.int32).reshape(
        (-1, shape[-1])
    )
    device_mesh = mesh_utils.create_device_mesh(
        (num_devices,), jax.devices()[:num_devices]
    )
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    f = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=jax.P('x'),
            out_specs=jax.P('x'),
            check_vma=False,
        )
    )
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(f(x), expected)

  @parameterized.product(direction=['left', 'right'])
  def test_collective_permute_2d(self, direction):
    shape = (8, 128)

    @plsc.kernel(
        out_shape=jax.ShapeDtypeStruct(shape, jnp.int32),
        mesh=plsc.ScalarSubcoreMesh(axis_name='core', num_cores=1),
        scratch_shapes=(
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
    )
    def kernel(x_ref, y_ref, ready_sem, send_sem, recv_sem):
      my_id = lax.axis_index('x')
      my_other_id = lax.axis_index('y')
      axis_size = lax.axis_size('x')
      if direction == 'right':
        neighbor = lax.rem(my_id + 1, axis_size)
      else:
        neighbor = lax.rem(my_id + axis_size - 1, axis_size)
      pltpu.semaphore_signal(ready_sem, device_id=(my_other_id, neighbor))
      pltpu.semaphore_wait(ready_sem)
      pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=(my_other_id, neighbor)
        ).wait()

    axis_size = jax.device_count() // 2
    x = jnp.arange(axis_size * 8 * 128).reshape((axis_size * 8, 128))

    device_mesh = mesh_utils.create_device_mesh((2, axis_size), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['y', 'x'])
    y = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=jax.P('x', None),
            out_specs=jax.P('x', None),
            check_vma=False,
        )
    )(x)
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)


if __name__ == '__main__':
  absltest.main()
