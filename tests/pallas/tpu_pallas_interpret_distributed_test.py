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

"""Tests for TPU-specific interpret mode.

To work around https://github.com/jax-ml/jax/issues/25671 , this file
contains only tests that use shard_map.
"""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax._src import test_util as jtu
import jax._src.pallas.mosaic.interpret as mosaic_interpret
from jax.experimental import pallas as pl
from jax._src import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

import numpy as np

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

P = jax.sharding.PartitionSpec


# TODO(jburnim): Figure out how to safely run different instance of TPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class InterpretDistributedTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest(f'requires at least 4 devices, found {jax.device_count()}')

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_right_permute_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jax.make_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    # Create an input array that shards the last dimension across
    # all devices.
    input_arr = jax.random.uniform(
      jax.random.key(0), (8, 128 * num_devices), dtype=jnp.float32)
    input_arr = jax.device_put(input_arr, sharding)

    def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem):
      my_id = lax.axis_index('x')
      left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
      right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))

      barrier_sem = pltpu.get_barrier_semaphore()
      pltpu.semaphore_signal(
          barrier_sem,
          device_id=(left_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH)
      pltpu.semaphore_signal(
          barrier_sem,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH)
      pltpu.semaphore_wait(barrier_sem, 2)

      remote_copy_op = pltpu.make_async_remote_copy(
          src_ref=input_ref,
          dst_ref=output_ref,
          send_sem=send_sem,
          recv_sem=recv_sem,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
      )
      remote_copy_op.start()
      remote_copy_op.wait()

    out_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # MemorySpace.ANY will (usually) place the tensor in HBM.
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
        scratch_shapes=(
            # We allocate DMA semaphores in scratch memory.
            [pltpu.SemaphoreType.DMA] * 2
        ),
    )
    right_permute = pl.pallas_call(
        right_permute_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=13),
        interpret=pltpu.InterpretParams(
            dma_execution_mode=dma_execution_mode, detect_races=detect_races),
    )
    # Wrap the kernel within a shard_map to call.
    pallas_result = jax.jit(
        shard_map.shard_map(
            right_permute,
            mesh=mesh,
            in_specs=partition,
            out_specs=partition,
            check_vma=False,
        )
    )(input_arr)

    # Compare Pallas result to XLA shard_map result.
    perm = tuple((src, (src + 1) % num_devices) for src in range(num_devices))
    xla_result = jax.jit(
        shard_map.shard_map(
            lambda x: lax.ppermute(x, 'x', perm),
            mesh=mesh, in_specs=partition, out_specs=partition)
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result)
    if detect_races:
      self.assertFalse(mosaic_interpret.races.races_found)

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_all_gather_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P('x', None)
    mesh = jax.make_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    # Create an input array that shards the first dimension across
    # all devices.
    input_arr = jax.random.uniform(
        jax.random.key(0), (8 * num_devices, 128), dtype=jnp.float32)
    input_arr = jax.device_put(input_arr, sharding)

    def all_gather_kernel(input_ref,
                          output_ref,
                          local_copy_sem,
                          send_sem,
                          recv_sems):
      outer_step = pl.program_id(0)
      my_id = lax.axis_index('x')
      left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
      right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
      copy_slot = my_id - outer_step
      copy_slot = lax.rem(copy_slot + num_devices, jnp.int32(num_devices))

      @pl.when(outer_step == 0)
      def _():
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(left_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, 2)

        local_copy_op = pltpu.make_async_copy(
          src_ref=input_ref,
          dst_ref=output_ref.at[my_id],
          sem=local_copy_sem,
        )
        local_copy_op.start()
        local_copy_op.wait()

      # Copy to our right neighbor.
      # Note that we will also be receiving data from our left neighbor,
      # but at `copy_slot-1` rather than `copy_slot`! This makes use of the fact
      # that the indices do not need to be symmetric between remote DMAs.
      remote_copy_op = pltpu.make_async_remote_copy(
        src_ref=output_ref.at[copy_slot],
        dst_ref=output_ref.at[copy_slot],
        send_sem=send_sem,
        recv_sem=recv_sems.at[outer_step],
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )
      remote_copy_op.start()
      remote_copy_op.wait()

    out_shape = jax.ShapeDtypeStruct((num_devices, 8, 128), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
        # MemorySpace.ANY will (usually) place the tensor in HBM.
        pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
      scratch_shapes=(
        # DMA semaphores are allocated in scratch memory.
        # We allocated one semaphore for a local HBM-VMEM copy,
        # and one for the remote send semaphore.
        [pltpu.SemaphoreType.DMA] * 2
        # We additionally allocate one receive semaphore per device.
        # This is to avoid situations where we have multiple
        # DMAs in flight, as we do not want to share a receive
        # semaphore between the DMAs.
        + [pltpu.SemaphoreType.DMA((num_devices-1,))]
      ),
      grid=(num_devices-1,)
    )

    all_gather = pl.pallas_call(
      all_gather_kernel,
      out_shape=out_shape,
      grid_spec=grid_spec,
      interpret=pltpu.InterpretParams(
          dma_execution_mode=dma_execution_mode, detect_races=detect_races),
      compiler_params=pltpu.CompilerParams(collective_id=0),
    )

    # Wrap the kernel within a shard_map to call.
    pallas_result = jax.jit(
      shard_map.shard_map(
        all_gather,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False
      )
    )(input_arr)

    # Compare Pallas result to XLA shard_map result.
    xla_result = jax.jit(
      shard_map.shard_map(
        lambda x: lax.all_gather(x, 'x'),
        mesh=mesh, in_specs=partition, out_specs=partition
      )
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result)
    if detect_races:
      self.assertFalse(mosaic_interpret.races.races_found)

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_all_reduce_sum_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jax.make_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    input_arr = jax.random.uniform(
        jax.random.key(0), shape=(8, 128 * num_devices))
    input_arr = jax.device_put(input_arr, sharding)

    def all_reduce_kernel(
        x_ref,
        o_ref,
        hbm_scratch,
        copy_sem,
        remote_recv_sem,
        remote_send_sem,
        capacity_sem,
        receive_scratch,
    ):
      outer_step = pl.program_id(0)
      working_slot = lax.rem(outer_step, jnp.int32(2))
      receiving_slot = 1 - working_slot

      my_id = lax.axis_index('x')
      right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
      left_neighbor = lax.rem(my_id - 1 + num_devices, jnp.int32(num_devices))

      @pl.when(outer_step == 0)
      def _():
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(left_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, 2)

        # Initialize o_ref, acc_scratch, and hbm_scratch.
        o_ref[...] = jnp.zeros_like(o_ref)
        receive_scratch[...] = jnp.zeros_like(receive_scratch)
        initial_copy = pltpu.make_async_remote_copy(
          src_ref=x_ref,
          dst_ref=hbm_scratch.at[working_slot],
          send_sem=remote_send_sem,
          recv_sem=remote_recv_sem,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        initial_copy.start()
        initial_copy.wait()

      # Signal to our left neighbor that we are ready to receive.
      # Without this signal, our left neighbor can be >=1 iteration ahead,
      # meaning it could write into our working slot.
      pltpu.semaphore_signal(
        capacity_sem,
        inc=1,
        device_id=(left_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )

      # Copy the partial result our left neighbor sent to us into VMEM for
      # computation.
      local_copy = pltpu.make_async_copy(
        src_ref=hbm_scratch.at[working_slot],
        dst_ref=receive_scratch,
        sem=copy_sem,
      )
      local_copy.start()

      # Block until our right neighbor is ready to receive.
      pltpu.semaphore_wait(capacity_sem, 1)
      # Pass the value to our right neighbor.
      remote_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot],
        dst_ref=hbm_scratch.at[receiving_slot],
        send_sem=remote_send_sem,
        recv_sem=remote_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )
      remote_copy.start()
      # Finish local copy and accumulate while remote_copy is happening.
      local_copy.wait()
      o_ref[...] += receive_scratch[...]
      # Block until remote copy finishes.
      remote_copy.wait()

    out_shape = (
      jax.ShapeDtypeStruct((8, 128), input_arr.dtype),
      # We allocate the double-buffer as a Pallas output so that it is
      # resident in HBM.
      jax.ShapeDtypeStruct((2, 8, 128), input_arr.dtype),  # hbm_scratch
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
        # Our input lives in VMEM
        pl.BlockSpec(memory_space=pltpu.VMEM),
      ],
      out_specs=[
        # Our output lives in VMEM
        pl.BlockSpec(memory_space=pltpu.VMEM),
        # Our double-buffer lives in HBM
        pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      grid=(num_devices,),
      scratch_shapes=(
        [pltpu.SemaphoreType.DMA] * 3
        + [pltpu.SemaphoreType.REGULAR]  # capacity_sem
        + [pltpu.VMEM((8, 128), input_arr.dtype)]  # receive_scratch
      ),
    )

    kernel = pl.pallas_call(
      all_reduce_kernel,
      out_shape=out_shape,
      grid_spec=grid_spec,
      interpret=pltpu.InterpretParams(
          dma_execution_mode=dma_execution_mode, detect_races=detect_races),
      compiler_params=pltpu.CompilerParams(collective_id=0),
    )

    pallas_result = jax.jit(
      shard_map.shard_map(
        kernel,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
      )
    )(input_arr)
    pallas_result = jax.block_until_ready(pallas_result)[0]

    def lax_sum(x):
      return lax.psum(x, 'x')

    xla_result = jax.jit(
      shard_map.shard_map(
        lax_sum, mesh=mesh, in_specs=P(None, 'x'), out_specs=P(None, 'x')
      )
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result, atol=1e-5)
    if detect_races:
      self.assertFalse(mosaic_interpret.races.races_found)

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_reduce_scatter_sum_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jax.make_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    # We need a block size of (16, 128) to ensure that a half-slice is at least
    # of size (8, 128), which is the size of a VREG. This makes tiling easier
    # for the compiler.
    block_size = (16, 128)
    input_arr = jax.random.uniform(
        jax.random.key(0),
        shape=(block_size[0] * num_devices, block_size[1] * num_devices),
        dtype=jnp.float32,
    )
    input_arr = jax.device_put(input_arr, sharding)

    LEFT = 0
    RIGHT = 1

    def mod(x, n):
      return lax.rem(x + n, n)

    def signal(left_or_right, semaphore):
      my_id = lax.axis_index('x')
      if left_or_right == LEFT:
        neighbor = mod(my_id - 1, jnp.int32(num_devices))
      else:
        neighbor = mod(my_id + 1, jnp.int32(num_devices))
      pltpu.semaphore_signal(
          semaphore,
          inc=1,
          device_id=(neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
      )

    def reduce_scatter_kernel(
        x_ref,
        o_ref,
        hbm_scratch,
        local_copy_sem,
        left_recv_sem,
        left_send_sem,
        right_recv_sem,
        right_send_sem,
        left_capacity_sem,
        right_capacity_sem,
        accum_scratch,
    ):
      outer_step = pl.program_id(0)
      phase = pl.program_id(1)
      is_start = jnp.logical_and(outer_step == 0, phase == 0)
      last_iteration = outer_step == pl.num_programs(0) - 1

      working_slot = lax.rem(outer_step, jnp.int32(2))
      receiving_slot = 1 - working_slot
      my_id = lax.axis_index('x')
      right_neighbor = mod(my_id + 1, jnp.int32(num_devices))
      left_neighbor = mod(my_id - 1, jnp.int32(num_devices))

      left_copy_device = mod(my_id + outer_step + 1, jnp.int32(num_devices))
      right_copy_device = mod(my_id - outer_step - 1, jnp.int32(num_devices))
      # Slices can be specified using pl.ds(start, size)
      left_copy_slice = pl.ds(0, block_size[0] // 2)
      right_copy_slice = pl.ds(block_size[0] // 2, block_size[0] // 2)
      current_phase_slice = pl.ds(phase * (block_size[0] // 2), block_size[0] // 2)

      @pl.when(is_start)
      def _():
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(left_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, 2)

      initial_left_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, left_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )

      initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )

      left_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot, left_copy_slice],
        dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )
      right_copy = pltpu.make_async_remote_copy(
        # Note: Right copy is flipped with regards to slots since we are copying
        # to the next outer_step iteration.
        src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )

      # --- Prologue ---
      @pl.when(is_start)
      def _():
        # Initialize o_ref, acc_scratch, and hbm_scratch with initial copies.
        o_ref[...] = jnp.zeros_like(o_ref[...])
        accum_scratch[...] = jnp.zeros_like(accum_scratch[...])

        initial_left_copy.start()
        initial_left_copy.wait()
        initial_right_copy.start()

        # We tell our left neighbor that it is allowed to send to the right.
        # (and vice versa for right neighbor)
        signal(LEFT, right_capacity_sem)
        signal(RIGHT, left_capacity_sem)

      # --- Body ---
      # At the beginning of our kernel body, we start a DMA which copies
      # the result we computed in the previous phase to our neighbor.
      # This allows us to overlap the communication of sending our previous phase
      # with the computation for the current phase.
      @pl.when(~is_start)
      def _():
        @pl.when(phase == LEFT)
        def _():
          # We block here until our right neighbor tells use we can send to
          # the right.
          pltpu.semaphore_wait(right_capacity_sem, 1)
          right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
          # We block here until our left neighbor tells use we can send to
          # the left.
          pltpu.semaphore_wait(left_capacity_sem, 1)
          left_copy.start()

      local_copy = pltpu.make_async_copy(
        src_ref=hbm_scratch.at[working_slot, current_phase_slice],
        dst_ref=accum_scratch,
        sem=local_copy_sem,
      )
      local_copy.start()
      local_copy.wait()

      @pl.when(~last_iteration)
      def _():
        @pl.when(phase == LEFT)
        def _():
          accum_scratch[...] += x_ref[left_copy_device, left_copy_slice]

        @pl.when(phase == RIGHT)
        def _():
          accum_scratch[...] += x_ref[right_copy_device, right_copy_slice]

      local_copy = pltpu.make_async_copy(
        src_ref=accum_scratch,
        dst_ref=hbm_scratch.at[working_slot, current_phase_slice],
        sem=local_copy_sem,
      )
      local_copy.start()
      local_copy.wait()

      @pl.when(is_start)
      def _():
        initial_right_copy.wait()

      # At the end of our kernel body, we wait on the DMA of the previous phase
      # to make sure the results are ready for the next phase.
      @pl.when(~is_start)
      def _():
        @pl.when(phase == LEFT)
        def _():
          right_copy.wait()
          signal(LEFT, right_capacity_sem)

        @pl.when(phase == RIGHT)
        def _():
          left_copy.wait()
          signal(RIGHT, left_capacity_sem)

      # --- Epilogue ---
      # Store result on last iteration.
      @pl.when(last_iteration)
      def _():
        # Clean up semaphores so that they exit with a value of 0.
        @pl.when(phase == LEFT)
        def _():
          o_ref[left_copy_slice, ...] = accum_scratch[...]
          pltpu.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == RIGHT)
        def _():
          o_ref[right_copy_slice, ...] = accum_scratch[...]
          pltpu.semaphore_wait(left_capacity_sem, 1)

    out_shape = (
      jax.ShapeDtypeStruct((block_size[0], block_size[1]), jnp.float32),  # output
      # Shape: [working/recv, block[0], block[1]]
      jax.ShapeDtypeStruct(
        (2, block_size[0], block_size[1]), jnp.float32
      ),  # hbm_scratch
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
        pl.BlockSpec(memory_space=pltpu.VMEM),
      ],
      out_specs=[
        pl.BlockSpec(memory_space=pltpu.VMEM),
        pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      grid=(num_devices, 2),
      scratch_shapes=(
        [pltpu.SemaphoreType.DMA] * 5
        + [pltpu.SemaphoreType.REGULAR] * 2  # Capacity semaphores
        + [
          pltpu.VMEM((block_size[0] // 2, block_size[1]), jnp.float32)
        ]  # accum_scratch
      ),
    )

    def pallas_reduce_scatter(input_arr):
      input_arr = input_arr.reshape(num_devices, block_size[0], block_size[1])
      return pl.pallas_call(
        reduce_scatter_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=pltpu.InterpretParams(
            dma_execution_mode=dma_execution_mode, detect_races=True),
        compiler_params=pltpu.CompilerParams(collective_id=7),
      )(input_arr)[0]

    pallas_result = jax.jit(
      shard_map.shard_map(
        pallas_reduce_scatter,
        mesh=mesh,
        in_specs=P(None, 'x'),
        out_specs=P('x', None),
        check_vma=False,
      )
    )(input_arr)
    pallas_result = jax.block_until_ready(pallas_result)

    # Compare our result to XLA.
    def lax_reduce_sum_scatter(x):
      x = x.reshape(num_devices, block_size[0], block_size[1])
      return lax.psum_scatter(x, 'x')

    xla_result = jax.jit(
      shard_map.shard_map(
        lax_reduce_sum_scatter,
        mesh=mesh,
        in_specs=P(None, 'x'),
        out_specs=P('x', None),
      )
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result, atol=1e-5)
    if detect_races:
      self.assertFalse(mosaic_interpret.races.races_found)

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_reduce_scatter_sum_with_emit_pipeline_example(
      self, dma_execution_mode, detect_races):
    self.skipTest('requires a patched pallas.emit_pipeline to specify/fake '
                  'the TPU generation')
    if jax.config.jax_enable_x64:
      self.skipTest('pallas.emit_pipeline + x64 is not currently supported')
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jax.make_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    # We pick a large outer kernel block size that we do not want to place
    # in VMEM. For pedagogical purposes we use (4096, 4096), although in
    # principle this can be much larger.
    outer_block_size = (512, 512)
    # We pick a smaller VMEM block size for the inner kernel.
    inner_block_size = (128, 128)
    input_arr = jax.random.uniform(
      jax.random.key(0),
      shape=(
        outer_block_size[0] * num_devices,
        outer_block_size[1] * num_devices,
      ),
      dtype=jnp.float32,
    )
    input_arr = jax.device_put(input_arr, sharding)

    inner_grid = (
      outer_block_size[0] // inner_block_size[0] // 2,
      outer_block_size[1] // inner_block_size[1],
    )
    inner_block_spec = pl.BlockSpec(
      index_map=lambda i, j: (i, j),
      block_shape=inner_block_size,
      memory_space=pltpu.ANY,
    )

    LEFT = 0
    RIGHT = 1

    def mod(x, n):
      return lax.rem(x + n, n)

    def signal(left_or_right, semaphore):
      my_id = lax.axis_index('x')
      if left_or_right == LEFT:
        neighbor = mod(my_id - 1, num_devices)
      else:
        neighbor = mod(my_id + 1, num_devices)
      pltpu.semaphore_signal(
          semaphore,
          inc=1,
          device_id=(neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
      )

    def reduce_scatter_kernel(
        x_ref,
        o_ref,
        hbm_scratch,
        left_recv_sem,
        left_send_sem,
        copy_sem,
        right_recv_sem,
        right_send_sem,
        left_capacity_sem,
        right_capacity_sem,
    ):
      outer_step = pl.program_id(0)
      phase = pl.program_id(1)
      is_start = jnp.logical_and(outer_step == 0, phase == 0)
      last_iteration = outer_step == pl.num_programs(0) - 1

      working_slot = lax.rem(outer_step, 2)
      receiving_slot = 1 - working_slot
      my_id = lax.axis_index('x')
      right_neighbor = mod(my_id + 1, num_devices)
      left_neighbor = mod(my_id - 1, num_devices)

      left_copy_device = mod(my_id + outer_step + 1, num_devices)
      right_copy_device = mod(my_id - outer_step - 1, num_devices)
      left_copy_slice = pl.ds(0, outer_block_size[0] // 2)
      right_copy_slice = pl.ds(outer_block_size[0] // 2, outer_block_size[0] // 2)
      current_phase_slice = pl.ds(
        phase * (outer_block_size[0] // 2), outer_block_size[0] // 2
      )

      initial_left_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, left_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )

      initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )

      left_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot, left_copy_slice],
        dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )
      right_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
        device_id_type=pltpu.DeviceIdType.MESH,
      )

      # --- Prologue ---
      @pl.when(is_start)
      def _():
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(left_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(right_neighbor,),
          device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, 2)

        initial_left_copy.start()
        initial_left_copy.wait()
        initial_right_copy.start()

        # We tell our left neighbor that it is allowed to send to the right.
        # (and vice versa for right neighbor)
        signal(LEFT, right_capacity_sem)
        signal(RIGHT, left_capacity_sem)

      @pl.when(~is_start)
      def _():
        @pl.when(phase == LEFT)
        def _():
          # We block here until our right neighbor tells use we can send to
          # the right.
          pltpu.semaphore_wait(right_capacity_sem, 1)
          right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
          # We block here until our left neighbor tells use we can send to
          # the left.
          pltpu.semaphore_wait(left_capacity_sem, 1)
          left_copy.start()

      # --- Body ---
      def inner_kernel(input_ref, accum_ref):
        # We do not explicitly use += because we set should_accumulate_out=True.
        accum_ref[...] = input_ref[...]

      accum_pipeline = pltpu.emit_pipeline(
        inner_kernel,
        in_specs=[inner_block_spec],
        out_specs=inner_block_spec,
        should_accumulate_out=True,
        grid=inner_grid,
      )

      @pl.when(~last_iteration)
      def _():
        @pl.when(phase == LEFT)
        def _():
          accum_pipeline(
            x_ref.at[left_copy_device, left_copy_slice],
            hbm_scratch.at[working_slot, left_copy_slice],
          )

        @pl.when(phase == RIGHT)
        def _():
          accum_pipeline(
            x_ref.at[right_copy_device, right_copy_slice],
            hbm_scratch.at[working_slot, right_copy_slice],
          )

      # --- Epilogue ---
      @pl.when(is_start)
      def _():
        initial_right_copy.wait()

      @pl.when(~is_start)
      def _():
        @pl.when(phase == LEFT)
        def _():
          right_copy.wait()
          signal(LEFT, right_capacity_sem)

        @pl.when(phase == RIGHT)
        def _():
          left_copy.wait()
          signal(RIGHT, left_capacity_sem)

      # Store result on last iteration.
      @pl.when(last_iteration)
      def _():
        output_copy = pltpu.make_async_copy(
          src_ref=hbm_scratch.at[working_slot, current_phase_slice],
          dst_ref=o_ref.at[current_phase_slice],
          sem=copy_sem,
        )
        output_copy.start()
        output_copy.wait()

        # Clean up semaphores so that they exit with a value of 0.
        @pl.when(phase == LEFT)
        def _():
          pltpu.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == RIGHT)
        def _():
          pltpu.semaphore_wait(left_capacity_sem, 1)

    out_shape = (
      jax.ShapeDtypeStruct(
        (outer_block_size[0], outer_block_size[1]), jnp.float32
      ),
      # Shape: [working/recv, block[0], block[1]]
        jax.ShapeDtypeStruct(
          (2, outer_block_size[0], outer_block_size[1]), jnp.float32
        ),  # hbm_scratch
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
        pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      out_specs=[
        pl.BlockSpec(memory_space=pltpu.ANY),
        pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      grid=(num_devices, 2),
      scratch_shapes=(
        [pltpu.SemaphoreType.DMA] * 5
        + [pltpu.SemaphoreType.REGULAR] * 2  # Capacity semaphores
      ),
    )

    def pallas_reduce_scatter(input_arr):
      input_arr = input_arr.reshape(
        num_devices, outer_block_size[0], outer_block_size[1]
      )
      return pl.pallas_call(
        reduce_scatter_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=pltpu.InterpretParams(
            dma_execution_mode=dma_execution_mode, detect_races=detect_races),
        compiler_params=pltpu.CompilerParams(collective_id=19),
      )(input_arr)[0]

    pallas_result = jax.jit(
      shard_map.shard_map(
        pallas_reduce_scatter,
        mesh=mesh,
        in_specs=P(None, 'x'),
        out_specs=P('x', None),
        check_vma=False,
      )
    )(input_arr)
    pallas_result = jax.block_until_ready(pallas_result)

    def lax_reduce_sum_scatter(x):
      x = x.reshape(num_devices, outer_block_size[0], outer_block_size[1])
      return lax.psum_scatter(x, 'x')

    xla_result = jax.jit(
        shard_map.shard_map(
            lax_reduce_sum_scatter,
            mesh=mesh,
            in_specs=P(None, 'x'),
            out_specs=P('x', None),
        )
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result, atol=1e-5)
    if detect_races:
      self.assertFalse(mosaic_interpret.races.races_found)

  def test_race_detection(self):
    num_devices = 4
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, P('x', None))

    input_arr = jax.random.uniform(jax.random.key(0), (8 * num_devices, 128))
    input_arr = jax.device_put(input_arr, sharding)

    def kernel(src_dst_ids_ref, x_ref, o_ref, send_sem, recv_sem):
      # Send the specified DMAs.
      my_id = lax.axis_index('x')
      src_dst_ids = src_dst_ids_ref[:]
      recv_count = 0
      for i in range(src_dst_ids.shape[0]):
        src_id = src_dst_ids[i, 0]
        dst_id = src_dst_ids[i, 1]
        @pl.when(src_id == my_id)
        def _():
          dma = pltpu.make_async_remote_copy(
              src_ref=x_ref,
              dst_ref=o_ref,
              send_sem=send_sem,
              recv_sem=recv_sem,
              device_id=(dst_id,),
              device_id_type=pltpu.DeviceIdType.MESH,
          )
          dma.start()
          dma.wait_send()
        recv_count += jnp.where(dst_id == my_id, 1, 0)

      # Wait until we have received all DMAs.
      @pl.when(recv_count > 0)
      def _():
        fake_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref.at[pl.ds(0, 8 * recv_count)],
            dst_ref=o_ref.at[pl.ds(0, 8 * recv_count)],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(my_id,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        fake_dma.wait_recv()

    @jax.jit
    def run(src_dst_ids):
      return shard_map.shard_map(
          pl.pallas_call(
              kernel,
              out_shape=jax.ShapeDtypeStruct((8, 128), input_arr.dtype),
              in_specs=[
                  pl.BlockSpec(memory_space=pltpu.SMEM),
                  pl.BlockSpec(memory_space=pltpu.ANY),
              ],
              out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
              scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
              interpret=pltpu.InterpretParams(
                  dma_execution_mode='eager',
                  detect_races=True,
              ),
          ),
          mesh=mesh,
          in_specs=(P(None), P('x', None)),
          out_specs=P('x', None),
          check_vma=False,
      )(src_dst_ids, input_arr)

    run(jnp.array([[0, 1], [1, 2], [2, 3]], jnp.int32)).block_until_ready()
    self.assertFalse(mosaic_interpret.races.races_found)

    # Racing writes to device 2.
    run(jnp.array([[0, 1], [1, 2], [3, 2], [3, 0]], jnp.int32)).block_until_ready()
    self.assertTrue(mosaic_interpret.races.races_found)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
