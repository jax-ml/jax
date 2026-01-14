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

import functools
import json
import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax._src import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

partial = functools.partial


class PallasCallRemoteDMATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only TPUs v4+ are supported.')

  @parameterized.named_parameters(
      ('vmem', pltpu.VMEM),
      ('hbm', pl.ANY),
  )
  def test_basic_remote_vmem_dma(self, mem):
    # Implements very simple collective permute
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        other_dev_id = 1 - lax.axis_index('x')
        pltpu.semaphore_signal(ready_sem, device_id=other_dev_id,
                               device_id_type=pltpu.DeviceIdType.LOGICAL)
        pltpu.semaphore_wait(ready_sem)
        copy_done = pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, other_dev_id,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        copy_done.wait_send()
        copy_done.wait_recv()

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

    x = jnp.arange(2 * 8 * 128.0).reshape((2 * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=mem)],
          out_specs=pl.BlockSpec(memory_space=mem),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32, vma=frozenset('x')),
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    f = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
        )
    )
    jaxpr = f.trace(x).jaxpr
    self.assertNotIn('pvary', str(jaxpr))
    y = f(x)
    expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)

  def test_vma_error(self):
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        other_dev_id = 1 - lax.axis_index('x')
        pltpu.semaphore_signal(ready_sem, device_id=other_dev_id,
                               device_id_type=pltpu.DeviceIdType.LOGICAL)
        pltpu.semaphore_wait(ready_sem)
        copy_done = pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, other_dev_id,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        copy_done.wait_send()
        copy_done.wait_recv()

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

    x = jnp.arange(2 * 8 * 128.0).reshape((2 * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    f = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
        )
    )
    with self.assertRaisesRegex(
        ValueError,
        'When `check_vma=True` on `jax.shard_map`, `vma` on'
        ' `jax.ShapeDtypeStruct` must not be `None`'):
      f(x)

  @parameterized.named_parameters(
      ('left', 'left'),
      ('right', 'right')
  )
  def test_pallas_call_axis_index(self, direction):
    # Implements very simple collective permute
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        my_id = lax.axis_index('x')
        num_devices = lax.axis_size('x')
        if direction == 'right':
          neighbor = lax.rem(my_id + 1, num_devices)
        else:
          neighbor = lax.rem(my_id - 1, num_devices)
          # Neighbor might be negative here so we add num_devices in case
          neighbor = jnp.where(neighbor < 0, neighbor + num_devices, neighbor)
        pltpu.semaphore_signal(ready_sem, device_id=neighbor)
        pltpu.semaphore_wait(ready_sem)
        copy_done = pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=neighbor
        )
        copy_done.wait_send()
        copy_done.wait_recv()

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

    num_devices = jax.local_device_count()
    x = jnp.arange(num_devices * 8 * 128).reshape((num_devices * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          out_shape=x,
      )(x)

    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False
        )
    )(x)
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)

  @parameterized.named_parameters(('left', 'left'), ('right', 'right'))
  def test_pallas_call_axis_index_2d_mesh(self, direction):
    # Implements very simple collective permute in a 2D mesh.
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        my_id = lax.axis_index('x')
        my_other_id = lax.axis_index('y')
        axis_size = lax.axis_size('x')
        if direction == 'right':
          neighbor = lax.rem(my_id + 1, axis_size)
        else:
          neighbor = lax.rem(my_id - 1, axis_size)
          # Neighbor might be negative here so we add num_devices in case
          neighbor = jnp.where(neighbor < 0, neighbor + axis_size, neighbor)
        pltpu.semaphore_signal(ready_sem, device_id=(my_other_id, neighbor))
        pltpu.semaphore_wait(ready_sem)
        copy_done = pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=(my_other_id, neighbor)
        )
        copy_done.wait_send()
        copy_done.wait_recv()

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

    axis_size = jax.device_count() // 2
    x = jnp.arange(axis_size * 8 * 128).reshape((axis_size * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          out_shape=x,
      )(x)

    device_mesh = mesh_utils.create_device_mesh(
        (2, axis_size), jax.devices()
    )
    mesh = jax.sharding.Mesh(device_mesh, ['y', 'x'])
    y = jax.jit(
        shard_map.shard_map(
            body,
            mesh=mesh,
            in_specs=P('x', None),
            out_specs=P('x', None),
            check_vma=False,
        )
    )(x)
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)

  def test_barrier_semaphore(self):
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        my_id = lax.axis_index('x')
        num_devices = lax.axis_size('x')
        neighbor = lax.rem(my_id + 1, num_devices)
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(barrier_sem, device_id=neighbor)
        pltpu.semaphore_wait(barrier_sem)
        pltpu.semaphore_signal(ready_sem, device_id=neighbor)
        pltpu.semaphore_wait(ready_sem)
        pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=neighbor
        ).wait()

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

    num_devices = jax.local_device_count()
    x = jnp.arange(num_devices * 8 * 128).reshape((num_devices * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          out_shape=x,
          compiler_params=pltpu.CompilerParams(collective_id=0),
      )(x)

    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False
        )
    )(x)
    expected = jnp.concatenate([x[-8:], x[:-8]])
    np.testing.assert_allclose(y, expected)

  def test_barrier_semaphore_no_axis_name(self):
    def kernel(x_ref, y_ref):
      num_devices = lax.axis_size('x')
      barrier_sem = pltpu.get_barrier_semaphore()
      for i in range(num_devices):
        pltpu.semaphore_signal(barrier_sem, device_id=i)
      pltpu.semaphore_wait(barrier_sem, num_devices)
      pltpu.sync_copy(x_ref, y_ref)

    x = jnp.arange(8 * 128).reshape((8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          out_shape=x,
          compiler_params=pltpu.CompilerParams(collective_id=0),
      )(x)

    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False
        )
    )(x)
    np.testing.assert_allclose(y, x)

  @parameterized.product(joint_axis=[True, False])
  def test_axis_dict_with_core_multi_device(self, joint_axis):
    if jax.device_count() < 2:
      self.skipTest('Requires at least 2 devices for DMAs.')
    if (cdim := jax.devices()[0].num_cores) < 2:
      self.skipTest('Requires a TPU with at least 2 cores.')
    if pltpu.get_tpu_info().num_cores > 1 and joint_axis:
      self.skipTest('Joint axis is not supported on multi-core TPUs.')
    mesh = jax.make_mesh(
        (jax.device_count(),),
        ('device',),
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    ddim = jax.device_count()
    tcmesh = pltpu.create_tensorcore_mesh('core')
    pspec = P('device', None)
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    # Array is fully sharded.
    xlocal, ylocal = 8, 256
    input_arr = jnp.arange(xlocal * ddim * ylocal, dtype=jnp.int32).reshape(
        (xlocal * ddim, ylocal)
    )
    input_arr = jax.device_put(input_arr, sharding)

    def core_copy(refs):
      in_ref, out_ref = refs

      @pl.core_map(tcmesh, compiler_params=pltpu.CompilerParams(collective_id=7))
      def _():
        num_cores = jax.lax.axis_size('core')
        slc_size = ylocal // num_cores
        vmem_shape = (xlocal, slc_size)

        # This runs on every core, for every vmem iterations
        def alloc(core_sem, out_vmem_ref, sem, send_sem, recv_sem):
          core_index = jax.lax.axis_index('core')
          # Make sure all cores have entered run_scoped.
          for j in range(num_cores):
            pltpu.semaphore_signal(core_sem, 1, device_id={'core': j})
          pltpu.semaphore_wait(core_sem, num_cores)

          device_index = jax.lax.axis_index('device')
          slc = pl.ds(core_index * slc_size, slc_size)

          # Make sure all devices and cores have entered run_scoped.
          sem0 = pltpu.get_barrier_semaphore()
          for i in range(ddim):
            for j in range(num_cores):
              pltpu.semaphore_signal(
                  sem0, 1, device_id={'device': i, 'core': j}
              )
          pltpu.semaphore_wait(sem0, ddim * num_cores)

          # Identity function by default
          pltpu.async_copy(in_ref.at[:, slc], out_ref.at[:, slc], sem).wait()

          if joint_axis:
            device_id = {('device', 'core'): cdim + 1}
          else:
            device_id = {'device': 1, 'core': 1}
          copy_d0c0_to_d1c1 = pltpu.make_async_remote_copy(
              src_ref=in_ref.at[:, slc],
              dst_ref=out_vmem_ref,
              send_sem=send_sem,
              recv_sem=recv_sem,
              device_id=device_id,
              device_id_type=pltpu.DeviceIdType.MESH,
          )

          @pl.when(device_index == 0)
          def _():
            @pl.when(core_index == 0)
            def _():
              copy_d0c0_to_d1c1.start()
              copy_d0c0_to_d1c1.wait_send()

          @pl.when(device_index == 1)
          def _():
            @pl.when(core_index == 1)
            def _():
              copy_d0c0_to_d1c1.wait_recv()
              pltpu.async_copy(out_vmem_ref, out_ref.at[:, slc], sem).wait()

        pl.run_scoped(
            alloc,
            pltpu.SemaphoreType.REGULAR,
            pltpu.VMEM(vmem_shape, out_ref.dtype),
            *([pltpu.SemaphoreType.DMA] * 3),
        )

    @partial(jax.shard_map, mesh=mesh, in_specs=pspec, out_specs=pspec, check_vma=False)
    def run_core_kernel(input):
      output = jnp.zeros_like(input)
      _, output = pl.run_state(core_copy)((input, output))
      return output
    pallas_out = jax.jit(run_core_kernel)(input_arr)

    # The device=1 core=1 slice was flushed with device=0 core=0 contents
    np.testing.assert_array_equal(pallas_out[8:16, 128:], input_arr[:8, :128])
    # Mask that slice out and all should be the same.
    mask = jnp.zeros((8, 128), jnp.int32)
    masked_in = jax.lax.dynamic_update_slice(input_arr, mask, (8, 128))
    masked_out = jax.lax.dynamic_update_slice(pallas_out, mask, (8, 128))
    np.testing.assert_array_equal(masked_in, masked_out)

  def test_multi_device_core_local_kernel(self):
    num_devices = jax.device_count()
    num_cores = pltpu.get_tpu_info().num_cores
    x = jnp.arange(num_devices * num_cores * 8 * 128).reshape(
        (num_devices, num_cores, 8, 128)
    )

    def body(x):
      x_ref = jax.new_ref(x)
      y_ref = jax.new_ref(jnp.empty_like(x))

      tcmesh = pltpu.create_tensorcore_mesh('core')
      @pl.core_map(tcmesh)
      def _():
        num_cores = jax.lax.axis_size('core')
        def inner(sem):
          for i in range(num_cores):
            pltpu.semaphore_signal(sem, 1, device_id={'core': i})
          pltpu.semaphore_wait(sem, num_cores)
          core_id = jax.lax.axis_index('core')
          pltpu.sync_copy(x_ref.at[:, core_id], y_ref.at[:, core_id])
        pl.run_scoped(inner, pltpu.SemaphoreType.REGULAR)
      return jax.freeze(y_ref)

    mesh = jax.make_mesh((jax.device_count(),), ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False
        )
    )(x)
    np.testing.assert_allclose(y, x)

  def test_no_barrier_semaphore(self):
    def alloc_sem(_):
      num_devices = lax.axis_size('x')
      barrier_sem = pltpu.get_barrier_semaphore()
      for i in range(num_devices):
        pltpu.semaphore_signal(barrier_sem, device_id=i)
      pltpu.semaphore_wait(barrier_sem, num_devices)

    def barrier_kernel(x_ref, sem_ref, out_ref):
      num_devices = lax.axis_size('x')
      for i in range(num_devices):
        pltpu.semaphore_signal(sem_ref, device_id=i)
      pltpu.semaphore_wait(sem_ref, num_devices)
      out_ref[...] = x_ref[...] + 1

    x = jnp.arange(8 * 128).reshape((8, 128))

    def body(x):
      sem = pl.pallas_call(
          alloc_sem,
          in_specs=[],
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          out_shape=pltpu.SemaphoreType.REGULAR(()),
          compiler_params=pltpu.CompilerParams(collective_id=0),
      )()
      return pl.pallas_call(
          barrier_kernel,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.VMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          out_shape=x,
          compiler_params=pltpu.CompilerParams(skip_device_barrier=True),
      )(x, sem)

    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False
        )
    )(x)
    np.testing.assert_allclose(y, x + 1)


class PallasCallRemoteDMAInterpretTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu():
      self.skipTest('Test requires TPU')

  @parameterized.parameters(('left',), ('right',))
  def test_interpret_remote_dma_ppermute(self, permutation):
    if jax.device_count() <= 1:
      self.skipTest('Test requires multiple devices.')
    num_devices = jax.device_count()
    if permutation == 'left':
      permute_fn = lambda x: lax.rem(x + num_devices - 1, num_devices)
    else:
      permute_fn = lambda x: lax.rem(x + num_devices + 1, num_devices)

    # Construct a kernel which performs a ppermute based on permute_fn.
    def test_kernel(x_ref,
                    o_ref,
                    copy_send_sem,
                    copy_recv_sem,
                ):
      o_ref[...] = jnp.zeros_like(o_ref[...])
      my_id = lax.axis_index('x')
      dst_device = permute_fn(my_id)
      input_to_output_copy = pltpu.make_async_remote_copy(
          src_ref=x_ref,
          dst_ref=o_ref,
          send_sem=copy_send_sem,
          recv_sem=copy_recv_sem,
          device_id=dst_device,
          device_id_type=pltpu.DeviceIdType.LOGICAL,
      )
      input_to_output_copy.start()
      input_to_output_copy.wait()

    out_shape = (jax.ShapeDtypeStruct((8, 128), jnp.float32))
    grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA] * 2
            )
        )

    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = jax.sharding.Mesh(devices, 'x')
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'x'))
    unsharded_arr = jax.random.normal(
        jax.random.key(0), shape=(8, 128 * num_devices))
    sharded_arr = jax.device_put(unsharded_arr, sharding)

    kernel = pl.pallas_call(
        test_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=True,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_vma=False))
    result = compiled_func(sharded_arr)

    perm = tuple((src, permute_fn(src)) for src in range(num_devices))
    perm = jax.tree_util.tree_map(int, perm)
    def lax_permute(x):
      return lax.ppermute(x, 'x', perm)
    expected = jax.jit(shard_map.shard_map(lax_permute,
                                   mesh=mesh,
                                   in_specs=P(None, 'x'),
                                   out_specs=P(None, 'x')))(sharded_arr)
    np.testing.assert_array_equal(result, expected)

  def test_interpret_remote_dma_asymmetrical_indexer(self):
    # Test DMAs where destination slices are not the same.
    if jax.local_device_count() <= 1:
      self.skipTest('Test requires multiple devices.')
    if not jtu.is_device_tpu(5, 'e'):
      self.skipTest('Only works with TPU v5e.')
    num_devices = jax.local_device_count()

    def test_kernel(x_ref,
               output_ref,
               send_sem,
               recv_sem, barrier_sem):
      output_ref[...] = jnp.zeros_like(output_ref[...])
      my_id = lax.axis_index('x')
      even_device = lax.rem(my_id, 2)
      odd_device = 1 - even_device
      next_device = lax.rem(my_id + 1, num_devices)

      del barrier_sem
      # This kernel as written is racey, but remote semaphore_signal is not
      # supported in HLO interpret mode yet. HLO interpret will not race
      # because DMAs are implemented as collectives which will barrier.
      # Signal to the sender to this device that output_ref has been zeroed
      # and this device is ready to receive.
      # prev_device = (my_id - 1) % num_devices
      # pltpu.semaphore_signal(barrier_sem, 1, device_id=prev_device)
      # pltpu.semaphore_wait(barrier_sem)

      # If the device_id is even, we copy to output_ref[1].
      # If it's odd, we copy to output_ref[0].
      @pl.when(even_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=output_ref.at[1],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=next_device,
        )
        remote_dma.start()
        remote_dma.wait()
      @pl.when(odd_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=output_ref.at[0],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=next_device,
        )
        remote_dma.start()
        remote_dma.wait()

    out_shape = (jax.ShapeDtypeStruct((2, 8, 128), jnp.float32))
    grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA,
                 pltpu.SemaphoreType.DMA,
                 pltpu.SemaphoreType.REGULAR]
            )
        )

    devices = mesh_utils.create_device_mesh(( num_devices,))
    mesh = jax.sharding.Mesh(devices, P('x'))
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'x'))
    unsharded_arr = jax.random.normal(
        jax.random.key(0), shape=(8, 128 * num_devices))
    sharded_arr = jax.device_put(unsharded_arr, sharding)

    # Compare interpret mode result to non-interpret mode result.
    kernel = pl.pallas_call(
        test_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=True,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_vma=False))
    result_interpret = compiled_func(sharded_arr)

    expected = []
    zeros = jnp.zeros((8, 128), jnp.float32)
    for i in range(num_devices):
      if i == 0:
        x_slice = unsharded_arr[:, 128 * (num_devices - 1):]
      else:
        x_slice = unsharded_arr[:, 128 * (i-1):128 * i]
      if i % 2 == 0:
        expected.append(jnp.stack([zeros, x_slice], axis=0))
      else:
        expected.append(jnp.stack([x_slice, zeros], axis=0))
    expected = jnp.concatenate(expected, axis=1)

    np.testing.assert_array_equal(result_interpret,
                                  expected)

  def test_interpret_remote_dma_asymmetrical_refs(self):
    # Test DMAs where dst refs are not the same.
    self.skipTest('Known failure.')
    num_devices = jax.local_device_count()

    def test_kernel(x_ref,
               even_output,
               odd_output,
               send_sem,
               recv_sem):
      even_output[...] = jnp.zeros_like(even_output[...])
      odd_output[...] = jnp.zeros_like(odd_output[...])
      my_id = lax.axis_index('x')
      even_device = lax.rem(my_id, 2)
      odd_device = 1 - even_device
      neighbor = lax.rem(my_id + 1, num_devices)
      @pl.when(even_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=even_output,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=neighbor,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        remote_dma.start()
        remote_dma.wait()
      @pl.when(odd_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=odd_output,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=neighbor,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        remote_dma.start()
        remote_dma.wait()

    out_shape = (jax.ShapeDtypeStruct((8, 128), jnp.float32))
    grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA] * 2
            )
        )

    devices = mesh_utils.create_device_mesh((1, num_devices))
    mesh = jax.sharding.Mesh(devices, P(None, 'x'))
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'x'))
    unsharded_arr = jax.random.normal(
        jax.random.key(0), shape=(8, 128 * num_devices))
    sharded_arr = jax.device_put(unsharded_arr, sharding)

    # Compare interpret mode result to non-interpret mode result.
    kernel = pl.pallas_call(
        test_kernel,
        out_shape=(out_shape, out_shape),
        grid_spec=grid_spec,
        interpret=True,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_vma=False))
    result_interpret = compiled_func(sharded_arr)

    kernel = pl.pallas_call(
        test_kernel,
        out_shape=(out_shape, out_shape),
        grid_spec=grid_spec,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_vma=False))
    result_noninterpret = compiled_func(sharded_arr)
    np.testing.assert_allclose(result_interpret,
                               result_noninterpret,
                               atol=1e-5,
                               rtol=1e-3)


class VerificationTest(jtu.JaxTestCase):

  def test_verification(self):
    self.skipTest(
        'TODO(b/455847773): Fix MLIR layout mismatch in tpu.memref_slice (dynamic offset issue).'
    )
    if (num_devices := jax.local_device_count()) <= 1:
      self.skipTest('Test requires multiple devices.')
    if not jtu.is_device_tpu_at_least(4) or jax.devices()[0].num_cores > 1:
      self.skipTest('Test requires a new single-core TPU.')
    def kernel_body(in_ref, out_ref, scratch_ref, send_sem, recv_sem, capacity_sem):
      my_id = lax.axis_index('x')
      dst_id = jnp.where(my_id == num_devices - 1, 0, my_id + 1)
      src_id = jnp.where(my_id == 0, num_devices - 1, my_id - 1)
      pltpu.semaphore_signal(capacity_sem, 1, device_id=src_id)
      out_ref[...] = jnp.zeros_like(out_ref)
      scratch_ref[0] = in_ref[0]

      @functools.partial(lax.fori_loop, 0, num_devices - 1, init_val=None)
      def _(i, _):
        slot = i % 2
        next_slot = 1 - slot
        pltpu.semaphore_wait(capacity_sem, 1)
        copy = pltpu.async_remote_copy(
            scratch_ref.at[slot],
            scratch_ref.at[next_slot],
            send_sem,
            recv_sem,
            device_id=dst_id,
        )
        out_ref[...] += scratch_ref[slot]
        copy.wait()
        pltpu.semaphore_signal(capacity_sem, 1, device_id=src_id)
      out_ref[...] += scratch_ref[(num_devices - 1) % 2]
      pltpu.semaphore_wait(capacity_sem, 1)

    kernel = pl.pallas_call(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((2, 128, 128), jnp.float32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
        ],
    )
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = jax.sharding.Mesh(devices, ['x'])
    # This is just a smoke test to ensure that the verification does not crash.
    with tempfile.TemporaryDirectory() as tmpdir:
      previous_config = jax.config.read('jax_pallas_dump_promela_to')
      jax.config.update('jax_pallas_dump_promela_to', tmpdir)
      shard_map.shard_map(
          kernel, mesh=mesh, in_specs=P('x'), out_specs=P(None), check_vma=False
      )(jnp.ones((8, 128, 128), jnp.float32))
      jax.config.update('jax_pallas_dump_promela_to', previous_config)
      self.assertNotEmpty(os.listdir(tmpdir))


class PallasKernelMetadataDistributedTest(parameterized.TestCase):

  @parameterized.product(
      axis_names=[['x', 'y'], [('x', 'y')], ['x'], ['y']],
      op=['copy', 'signal'],
  )
  def test_mesh_axes_metadata_is_preserved(self, axis_names, op):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Remote async copy only supported on TPU v4+')
    if len(jax.devices()) < 4:
      self.skipTest('Not enough devices')
    devices = np.array(jax.devices()[:4]).reshape((2, 2))
    mesh = jax.sharding.Mesh(devices, ('x', 'y'))

    def kernel(x_ref, out_ref):
      def body(send_sem, recv_sem, sem):
        if len(jax.tree.leaves(axis_names)) > 0:
          device_id = {a: 0 for a in axis_names}
          if op == 'copy':
            pltpu.async_remote_copy(
                x_ref,
                out_ref,
                send_sem,
                recv_sem,
                device_id=device_id,
            ).wait()
          else:
            pl.semaphore_signal(sem, device_id=device_id)
        else:
          out_ref[...] = x_ref[...]
      pl.run_scoped(
          body,
          send_sem=pltpu.SemaphoreType.DMA,
          recv_sem=pltpu.SemaphoreType.DMA,
          sem=pltpu.SemaphoreType.REGULAR,
      )

    @functools.partial(
        jax.jit,
        out_shardings=jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec('x', 'y')
        ),
    )
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec('x', 'y'),
        out_specs=jax.sharding.PartitionSpec('x', 'y'),
        check_vma=False,
    )
    def f(x):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((1, 1, 1, 128), jnp.float32),
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
      )(x)

    x = jnp.zeros((2, 2, 1, 128), dtype=jnp.float32)
    hlo = f.lower(x).compile().as_text()
    axis_names_text = json.dumps(
        json.dumps(sorted(jax.tree.leaves(axis_names)))
    )
    self.assertIn(
        f'"mesh_axes":{axis_names_text}',
        hlo,
    )


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
