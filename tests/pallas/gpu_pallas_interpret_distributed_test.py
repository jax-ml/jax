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

import math

from absl.testing import absltest
import jax
from jax import lax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams as InterpretParams
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

P = jax.sharding.PartitionSpec


# TODO(nrink): Figure out how to safely run different instance of GPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class InterpretDistributedTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    mosaic_interpret.gpu_callbacks.reset_gpu_interpret_mode_state()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    if jax.device_count() < 2:
      self.skipTest(f'requires at least 2 devices, found {jax.device_count()}')

  @jtu.parameterized.product(
      num_devices=list(range(2, 9)),
      shard_input=[False, True],
      shard_output=[False, True],
  )
  def test_pallas_call_under_shard_map(
      self, num_devices, shard_input, shard_output
  ):
    if not shard_input and not shard_output:
      self.skipTest('Skipping test configuration where nothing is sharded.')

    devices = jax.devices()[:num_devices]
    partition_spec = P('x')
    out_shape = (1,) if shard_output else (num_devices,)

    def kernel(x_ref, o_ref):
      dev_id = lax.axis_index('x')
      x_idx = 0 if shard_input else dev_id
      o_idx = 0 if shard_output else dev_id
      o_ref[o_idx] = x_ref[x_idx]

    x = np.arange(num_devices, dtype=jnp.int32)
    if shard_input:
      devices_mesh = jax.sharding.Mesh(devices, ['x'])
      sharding = jax.sharding.NamedSharding(devices_mesh, partition_spec)
      x = jax.device_put(x, sharding)

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct(out_shape, jnp.int32),
          interpret=InterpretParams(detect_races=True),
      )(x)

    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=partition_spec if shard_input else P(),
            out_specs=partition_spec if shard_output else P(),
            check_vma=False,
        )
    )(x)

    if not shard_output:
      y = np.array(
          [y.addressable_shards[i].data[i] for i in range(num_devices)]
      )

    np.testing.assert_array_equal(y, x)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
    num_devices=list(range(2, 9)),
    shard_input=[False, True],
    shard_output=[False, True],
  )
  def test_kernel_under_shard_map(self, num_devices, shard_input, shard_output):
    if not shard_input and not shard_output:
      self.skipTest('Skipping test configuration where nothing is sharded.')

    grid = (2, 3)
    grid_names = ('g0', 'g1')
    cluster = (2, 3)
    cluster_names = ('c0', 'c1')
    num_threads = 2
    thread_name = 't'
    num_total_threads = math.prod(grid) * math.prod(cluster) * num_threads

    devices = jax.devices()[:num_devices]
    partition_spec = P('x')
    out_shape = (
        (num_total_threads,)
        if shard_output
        else (num_devices * num_total_threads,)
    )

    def kernel(x_ref, o_ref):
      axes_dims = grid + cluster + (num_threads,)
      axes_names = grid_names + cluster_names + (thread_name,)

      flat_thread_id = jnp.int32(0)
      for i, name in enumerate(axes_names):
        stride = math.prod(axes_dims[i + 1 :])
        flat_thread_id += jax.lax.axis_index(name) * stride

      dev_id = lax.axis_index('x')
      x_idx = flat_thread_id
      if not shard_input:
        x_idx += dev_id * num_total_threads
      o_idx = flat_thread_id
      if not shard_output:
        o_idx += dev_id * num_total_threads
      o_ref[o_idx] = x_ref[x_idx]

    x = jnp.arange(num_devices * num_total_threads, dtype=jnp.int32)
    if shard_input:
      devices_mesh = jax.sharding.Mesh(devices, ['x'])
      sharding = jax.sharding.NamedSharding(devices_mesh, partition_spec)
      x = jax.device_put(x, sharding)

    def body(x):
      return plgpu.kernel(
          kernel,
          out_type=jax.ShapeDtypeStruct(out_shape, jnp.int32),
          grid=grid,
          grid_names=grid_names,
          cluster=cluster,
          cluster_names=cluster_names,
          num_threads=num_threads,
          thread_name=thread_name,
          interpret=InterpretParams(detect_races=True),
      )(x)

    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=partition_spec if shard_input else P(),
            out_specs=partition_spec if shard_output else P(),
            check_vma=False,
        )
    )(x)

    if not shard_output:
      y = np.concatenate([
          y.addressable_shards[i].data[
              i * num_total_threads : (i + 1) * num_total_threads
          ]
          for i in range(num_devices)
      ])

    np.testing.assert_array_equal(y, x)
    self.assertFalse(mosaic_interpret.get_races().races_found)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
