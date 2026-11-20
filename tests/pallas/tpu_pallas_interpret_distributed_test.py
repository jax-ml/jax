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

import collections
import functools
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import shard_map
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpret_lib
from jax._src.pallas.mosaic.interpret import race_detection_state
from jax._src.pallas.mosaic.interpret import vector_clock
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
    self.enter_context(
        jtu.ignore_warning(
            category=DeprecationWarning,
            message='jax.experimental.pallas.DeviceIdType is deprecated',
        )
    )

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    if jax.device_count() < 4:
      self.skipTest(f'requires at least 4 devices, found {jax.device_count()}')

  def tearDown(self):
    super().tearDown()
    pltpu.reset_tpu_interpret_mode_state()
    try:
      # If an exception was thrown by a jitted computation during the test,
      # we observe/consume exception here to avoid propagating it to the
      # next test.
      jax.effects_barrier()
    except:
      pass

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_right_permute_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jtu.create_mesh((num_devices,), ('x',))
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
      pl.semaphore_signal(barrier_sem, device_id=(left_neighbor,))
      pl.semaphore_signal(barrier_sem, device_id=(right_neighbor,))
      pl.semaphore_wait(barrier_sem, 2)

      remote_copy_op = pltpu.make_async_remote_copy(
          src_ref=input_ref,
          dst_ref=output_ref,
          send_sem=send_sem,
          recv_sem=recv_sem,
          device_id=(right_neighbor,),
      )
      remote_copy_op.start()
      remote_copy_op.wait()

    out_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        # MemorySpace.ANY will (usually) place the tensor in HBM.
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
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

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_all_gather_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P('x', None)
    mesh = jtu.create_mesh((num_devices,), ('x',))
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
        pl.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(left_neighbor,),
        )
        pl.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(right_neighbor,),
        )
        pl.semaphore_wait(barrier_sem, 2)

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
    )
      remote_copy_op.start()
      remote_copy_op.wait()

    out_shape = jax.ShapeDtypeStruct((num_devices, 8, 128), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
        # MemorySpace.ANY will (usually) place the tensor in HBM.
        pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
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

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_all_reduce_sum_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jtu.create_mesh((num_devices,), ('x',))
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
        pl.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(left_neighbor,),
        )
        pl.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(right_neighbor,),
        )
        pl.semaphore_wait(barrier_sem, 2)

        # Initialize o_ref, acc_scratch, and hbm_scratch.
        o_ref[...] = jnp.zeros_like(o_ref)
        receive_scratch[...] = jnp.zeros_like(receive_scratch)
        initial_copy = pltpu.make_async_remote_copy(
          src_ref=x_ref,
          dst_ref=hbm_scratch.at[working_slot],
          send_sem=remote_send_sem,
          recv_sem=remote_recv_sem,
          device_id=(right_neighbor,),
        )
        initial_copy.start()
        initial_copy.wait()

      # Signal to our left neighbor that we are ready to receive.
      # Without this signal, our left neighbor can be >=1 iteration ahead,
      # meaning it could write into our working slot.
      pl.semaphore_signal(
        capacity_sem,
        inc=1,
        device_id=(left_neighbor,),
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
      pl.semaphore_wait(capacity_sem, 1)
      # Pass the value to our right neighbor.
      remote_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot],
        dst_ref=hbm_scratch.at[receiving_slot],
        send_sem=remote_send_sem,
        recv_sem=remote_recv_sem,
        device_id=(right_neighbor,),
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
        pl.BlockSpec(memory_space=pl.ANY),
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

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_reduce_scatter_sum_example(self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jtu.create_mesh((num_devices,), ('x',))
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
      pl.semaphore_signal(
          semaphore,
          inc=1,
          device_id=(neighbor,),
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
        pl.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(left_neighbor,),
        )
        pl.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(right_neighbor,),
        )
        pl.semaphore_wait(barrier_sem, 2)

      initial_left_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, left_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
    )

      initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
    )

      left_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot, left_copy_slice],
        dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
    )
      right_copy = pltpu.make_async_remote_copy(
        # Note: Right copy is flipped with regards to slots since we are copying
        # to the next outer_step iteration.
        src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
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
          pl.semaphore_wait(right_capacity_sem, 1)
          right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
          # We block here until our left neighbor tells use we can send to
          # the left.
          pl.semaphore_wait(left_capacity_sem, 1)
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
          pl.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == RIGHT)
        def _():
          o_ref[right_copy_slice, ...] = accum_scratch[...]
          pl.semaphore_wait(left_capacity_sem, 1)

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
        pl.BlockSpec(memory_space=pl.ANY),
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

  @parameterized.product(
      dma_execution_mode=['eager', 'on_wait'],
      detect_races=[True, False])
  def test_reduce_scatter_sum_with_emit_pipeline_example(
      self, dma_execution_mode, detect_races):
    num_devices = jax.device_count()
    partition = P(None, 'x')
    mesh = jtu.create_mesh((num_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    self.enter_context(jax.set_mesh(mesh))

    def local_barrier(left_neighbor, right_neighbor, double_barrier=True):
      """Performs a barrier with neighbors on the global barrier semaphore.

      Optionally performs a second barrier, which prevents a potential race
      when reusing the same collective_id across kernel invocations.
      """
      barrier_sem = pltpu.get_barrier_semaphore()
      for neighbor in [left_neighbor, right_neighbor]:
        pl.semaphore_signal(
          barrier_sem,
          inc=1,
          device_id=(neighbor,),
        )
      pl.semaphore_wait(barrier_sem, 2)
      if double_barrier:
        # The double-barrier prevents a race condition where one neighbor can
        # re-enter the kernel again on a subsequent call and increment the
        # barrier semaphore a second time. This would unblock the current device
        # even if the other neighbor is not ready yet.
        # To implement a double-barrier, we stack-allocate a second REGULAR
        # semaphore using run_scoped.
        @functools.partial(pl.run_scoped,
                           second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
          for neighbor in [left_neighbor, right_neighbor]:
            pl.semaphore_signal(
              second_barrier,
              inc=1,
              device_id=(neighbor,),
                )
          pl.semaphore_wait(second_barrier, 2)

    # We pick a large outer kernel block size that we do not want to place
    # in VMEM. For pedagogical purposes we use (512, 512), although in
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
      memory_space=pltpu.VMEM,
    )

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
      pl.semaphore_signal(
          semaphore,
          inc=1,
          device_id=(neighbor,),
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

      working_slot = lax.rem(outer_step, jnp.int32(2))
      receiving_slot = 1 - working_slot
      my_id = lax.axis_index('x')
      right_neighbor = mod(my_id + 1, jnp.int32(num_devices))
      left_neighbor = mod(my_id - 1, jnp.int32(num_devices))

      left_copy_device = mod(my_id + outer_step + 1, jnp.int32(num_devices))
      right_copy_device = mod(my_id - outer_step - 1, jnp.int32(num_devices))
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
    )

      initial_right_copy = pltpu.make_async_remote_copy(
        src_ref=x_ref.at[my_id, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
    )

      left_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[working_slot, left_copy_slice],
        dst_ref=hbm_scratch.at[receiving_slot, left_copy_slice],
        send_sem=left_send_sem,
        recv_sem=left_recv_sem,
        device_id=(left_neighbor,),
    )
      right_copy = pltpu.make_async_remote_copy(
        src_ref=hbm_scratch.at[receiving_slot, right_copy_slice],
        dst_ref=hbm_scratch.at[working_slot, right_copy_slice],
        send_sem=right_send_sem,
        recv_sem=right_recv_sem,
        device_id=(right_neighbor,),
    )

      # --- Prologue ---
      @pl.when(is_start)
      def _():
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        local_barrier(left_neighbor, right_neighbor)

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
          pl.semaphore_wait(right_capacity_sem, 1)
          right_copy.start()

        @pl.when(phase == RIGHT)
        def _():
          # We block here until our left neighbor tells use we can send to
          # the left.
          pl.semaphore_wait(left_capacity_sem, 1)
          left_copy.start()

      # --- Body ---
      def inner_kernel(step, input_ref, accum_in_ref, accum_out_ref):
        del step  # Unused.
        accum_out_ref[...] = accum_in_ref[...] + input_ref[...]

      @pl.when(~last_iteration)
      def _():
        @pl.when(phase == LEFT)
        def _():
          pltpu.emit_pipeline(
            inner_kernel,
            in_specs=[inner_block_spec, inner_block_spec],
            out_specs=inner_block_spec,
            grid=inner_grid,
            _explicit_indices=True,
          )(
            x_ref.at[left_copy_device, left_copy_slice],
            hbm_scratch.at[working_slot, left_copy_slice],
            hbm_scratch.at[working_slot, left_copy_slice],
          )

        @pl.when(phase == RIGHT)
        def _():
          pltpu.emit_pipeline(
            inner_kernel,
            in_specs=[inner_block_spec, inner_block_spec],
            out_specs=inner_block_spec,
            grid=inner_grid,
            _explicit_indices=True,
          )(
            x_ref.at[right_copy_device, right_copy_slice],
            hbm_scratch.at[working_slot, right_copy_slice],
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
          pl.semaphore_wait(right_capacity_sem, 1)

        @pl.when(phase == RIGHT)
        def _():
          pl.semaphore_wait(left_capacity_sem, 1)

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
        pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=[
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
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
            dma_execution_mode=dma_execution_mode,
            detect_races=detect_races,
            uninitialized_memory='zero'
        ),
        compiler_params=pltpu.CompilerParams(collective_id=19),
      )(input_arr)[0]

    with jax.sharding.use_abstract_mesh(
          mesh.abstract_mesh.update(
              abstract_device=jax.sharding.AbstractDevice(
                  device_kind='TPU v6e', num_cores=1, platform='tpu')
          )
    ):
      pallas_result = jax.jit(
          shard_map.shard_map(
              pallas_reduce_scatter,
              in_specs=P(None, 'x'),
              out_specs=P('x', None),
              check_vma=False,
      ))(input_arr)
    pallas_result = jax.block_until_ready(pallas_result)

    def lax_reduce_sum_scatter(x):
      x = x.reshape(num_devices, outer_block_size[0], outer_block_size[1])
      return lax.psum_scatter(x, 'x')

    xla_result = jax.jit(
        shard_map.shard_map(
            lax_reduce_sum_scatter,
            in_specs=P(None, 'x'),
            out_specs=P('x', None),
        )
    )(input_arr)

    np.testing.assert_allclose(xla_result, pallas_result, atol=1e-5)

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
                  pl.BlockSpec(memory_space=pl.ANY),
              ],
              out_specs=pl.BlockSpec(memory_space=pl.ANY),
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

    # Racing writes to device 2.
    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, 'RACE DETECTED'):
      run(
          jnp.array([[0, 1], [1, 2], [3, 2], [3, 0]], jnp.int32)
      ).block_until_ready()

  def test_that_barrier_does_not_hang_after_exception(self):
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')

    partition = P(None, 'x')
    mesh = jtu.create_mesh((num_devices,), ('x',))

    def kernel(x_ref, o_ref):
      my_id = lax.axis_index('x')

      # Trigger an exception on device 0 (out-of-bounds read).
      @pl.when(my_id == 0)
      def _():
        o_ref[0, 0] = x_ref[100, 0]

      barrier_sem = pltpu.get_barrier_semaphore()
      left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
      right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
      pl.semaphore_signal(barrier_sem, device_id={'x': left_neighbor})
      pl.semaphore_signal(barrier_sem, device_id={'x': right_neighbor})
      pl.semaphore_wait(barrier_sem, 2)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[],
    )

    pallas_call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=42),
        interpret=pltpu.InterpretParams(),
    )

    run = shard_map.shard_map(
        pallas_call,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )

    input_arr = jnp.ones((8, 128 * num_devices), dtype=jnp.float32)
    input_arr = jax.device_put(
        input_arr, jax.sharding.NamedSharding(mesh, partition)
    )

    # TODO(jburnim): Check for 'o_ref[0, 0] = x_ref[100, 0]' and
    # 'Out-of-bounds read' in the error message once we can reliably propagate
    # the original exception.  (Currently, whether the final exception contains
    # the original cause depends on the thread order.)
    with self.assertRaises(jax.errors.JaxRuntimeError):
      run(input_arr).block_until_ready()

  def test_that_exception_does_not_cause_dma_to_wait_forever(self):
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')

    partition = P(None, 'x')
    mesh = jtu.create_mesh((num_devices,), ('x',))

    @pl.kernel(
        mesh=pltpu.TensorCoreMesh(axis_name='core', num_cores=2),
        out_type=jax.ShapeDtypeStruct((16, 128), jnp.float32),
        scratch_types=[pltpu.VMEM((8, 128), jnp.float32),
                       pltpu.SemaphoreType.DMA(2)],
        interpret=pltpu.InterpretParams())
    def kernel(x_ref, o_ref, vmem_ref, dma_sems):
      my_id = lax.axis_index('x')
      left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
      core_id = lax.axis_index('core')

      # Trigger an exception on device 1, core 0.
      @pl.when((my_id == 1) & (core_id == 0))
      def _():
        vmem_ref[0, 0] = vmem_ref[100, 0]

      pltpu.async_remote_copy(
          x_ref.at[pl.ds(core_id, 8)],
          o_ref.at[pl.ds(core_id, 8)],
          dma_sems.at[0],
          dma_sems.at[1],
          device_id=(left_neighbor,),
      ).wait()

    run = shard_map.shard_map(
        kernel,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )

    input_arr = jnp.ones((16, 128 * num_devices), dtype=jnp.float32)
    input_arr = jax.device_put(
        input_arr, jax.sharding.NamedSharding(mesh, partition)
    )

    # TODO(jburnim): Check for 'vmem_ref[0, 0] = vmem_ref[100, 0]' and
    # 'Out-of-bounds read' in the error message once we can reliably propagate
    # the original exception out of a thread_map.
    with self.assertRaises(jax.errors.JaxRuntimeError):
      run(input_arr).block_until_ready()

  @parameterized.parameters(1, 2, 4)
  def test_shard_map_of_mpmd_map(self, num_cores):
    num_devices = jax.device_count()
    partition = P('x', None)
    mesh = jtu.create_mesh((num_devices,), ('x',))

    core_mesh = pltpu.TensorCoreMesh(axis_name='core', num_cores=num_cores)
    interpret = pltpu.InterpretParams(detect_races=True)

    slc_size = 16 // core_mesh.shape['core']

    @jax.jit
    def f(x):
      @pl.kernel(
          out_type=x,
          mesh=core_mesh,
          scratch_types=[
              pltpu.VMEM((slc_size, 128), x.dtype),
              pltpu.VMEM((slc_size, 128), x.dtype),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.REGULAR,
          ],
          interpret=interpret,
      )
      def kernel(x_ref, y_ref, x_vmem_ref, y_vmem_ref, dma_sem, sem):
        num_cores = jax.lax.axis_size('core')
        slc_size = 16 // num_cores
        # Barrier so we deadlock unless the core_map is actually parallel.
        for i in range(num_cores):
          pl.semaphore_signal(sem, 1, core_index=i)
        pl.semaphore_wait(sem, num_cores)

        core_index = jax.lax.axis_index('core')
        slc = pl.ds(core_index * slc_size, slc_size)
        pltpu.async_copy(
            x_ref.at[slc],
            x_vmem_ref,
            dma_sem,
        ).wait()
        y = (x_vmem_ref[...] + num_cores * jax.lax.axis_index('x')
             + core_index + 1)
        y_vmem_ref[...] = y
        pltpu.async_copy(y_vmem_ref, y_ref.at[slc], dma_sem).wait()
      return kernel(x)

    x = jnp.arange(num_devices * 16 * 128, dtype=jnp.int32).reshape((-1, 128))
    y = jax.jit(
        shard_map.shard_map(f,
            mesh=mesh,
            in_specs=partition,
            out_specs=partition,
            check_vma=False,
        )
    )(x).block_until_ready()
    expected_out = (
        x.reshape((num_devices, num_cores, -1, 128)) + 1
        + jnp.arange(num_devices, dtype=jnp.int32)[..., None, None, None] * num_cores
        + jnp.arange(num_cores, dtype=jnp.int32)[None, ..., None, None]
    ).reshape(x.shape)
    np.testing.assert_array_equal(y, expected_out)

  def test_cross_kernel_collective_protocol(self):
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    for dma_mode in ('eager', 'on_wait'):
      for detect_races in (False, True):
        pltpu.reset_tpu_interpret_mode_state()

        # Step 1: allocate kernel returns (recv DMA sem, regular sem) and signals a barrier
        def alloc_k(recv_sem_out, reg_sem_out):
          my_id = lax.axis_index('x')
          peer_id = 1 - my_id
          bar = pltpu.get_barrier_semaphore('step1_barrier')
          pl.semaphore_signal(bar, device_id=(peer_id,))

        alloc_call = pl.pallas_call(
            alloc_k,
            out_shape=[
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.REGULAR(()),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ],
            interpret=pltpu.InterpretParams(
                dma_execution_mode=dma_mode,
                detect_races=detect_races,
            ),
        )

        # Step 2: start kernel waits barrier, starts remote copy into an HBM output using escaped recv sem
        def start_k(src_hbm_ref, recv_sem_in, reg_sem_in, dst_hbm_out, send_sem_scratch):
          my_id = lax.axis_index('x')
          right_neighbor = lax.rem(my_id + 1, jnp.int32(2))
          bar = pltpu.get_barrier_semaphore('step1_barrier')
          # Wait signal from alloc_k:
          pl.semaphore_wait(bar, 1)

          # Synchronize both devices within start_k before initiating remote copy:
          pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
          pl.semaphore_wait(bar, 1)

          pl.semaphore_signal(reg_sem_in, 1, device_id=(right_neighbor,))
          remote_copy_op = pltpu.make_async_remote_copy(
              src_ref=src_hbm_ref,
              dst_ref=dst_hbm_out,
              send_sem=send_sem_scratch,
              recv_sem=recv_sem_in,
              device_id=(right_neighbor,),
          )
          remote_copy_op.start()
          remote_copy_op.wait_send()
          # Ensure both devices have finished sending before either exits start_k
          pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
          pl.semaphore_wait(bar, 1)

        start_call = pl.pallas_call(
            start_k,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
            interpret=pltpu.InterpretParams(
                dma_execution_mode=dma_mode,
                detect_races=detect_races,
            ),
        )

        # Step 3: finish kernel waits recv sem and reads data
        def finish_k(dst_hbm_ref, recv_sem_in, reg_sem_in, dst_hbm_out):
          del dst_hbm_out
          my_id = lax.axis_index('x')
          left_neighbor = lax.rem(my_id + 2 - 1, jnp.int32(2))
          pl.semaphore_wait(reg_sem_in, 1)
          fake_dma = pltpu.make_async_remote_copy(
              src_ref=dst_hbm_ref,
              dst_ref=dst_hbm_ref,
              send_sem=recv_sem_in,
              recv_sem=recv_sem_in,
              device_id=(left_neighbor,),
          )
          fake_dma.wait_recv()

        finish_call = pl.pallas_call(
            finish_k,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            input_output_aliases={0: 0},
            interpret=pltpu.InterpretParams(
                dma_execution_mode=dma_mode,
                detect_races=detect_races,
            ),
        )

        def protocol_fn(src):
          recv_sem, reg_sem = alloc_call()
          dst = start_call(src, recv_sem, reg_sem)
          res = finish_call(dst, recv_sem, reg_sem)
          return res

        sharded_protocol = jax.jit(
            shard_map.shard_map(
                protocol_fn,
                mesh=mesh,
                in_specs=P(None, 'x'),
                out_specs=P(None, 'x'),
                check_vma=False,
            )
        )

        src = jnp.arange(8 * 128 * 2, dtype=jnp.float32).reshape((8, 128 * 2))
        res = sharded_protocol(src)
        np.testing.assert_array_equal(res[:, :128], src[:, 128:])
        np.testing.assert_array_equal(res[:, 128:], src[:, :128])

  def test_cross_kernel_collective_race_detected(self):
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    for dma_mode in ('eager', 'on_wait'):
      pltpu.reset_tpu_interpret_mode_state()

      def alloc_k(recv_sem_out):
        my_id = lax.axis_index('x')
        peer_id = 1 - my_id
        bar = pltpu.get_barrier_semaphore('step1_barrier')
        pl.semaphore_signal(bar, device_id=(peer_id,))

      alloc_call = pl.pallas_call(
          alloc_k,
          out_shape=pltpu.SemaphoreType.DMA(()),
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          interpret=pltpu.InterpretParams(
              dma_execution_mode=dma_mode,
              detect_races=True,
          ),
      )

      def start_k(src_hbm_ref, recv_sem_in, dst_hbm_out, send_sem_scratch):
        my_id = lax.axis_index('x')
        peer_id = 1 - my_id
        bar = pltpu.get_barrier_semaphore('step1_barrier')
        pl.semaphore_wait(bar, 1)

        # Synchronize both devices before initiating remote copy so buffers are allocated
        pl.semaphore_signal(bar, 1, device_id=(peer_id,))
        pl.semaphore_wait(bar, 1)

        remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=src_hbm_ref,
            dst_ref=dst_hbm_out,
            send_sem=send_sem_scratch,
            recv_sem=recv_sem_in,
            device_id=(peer_id,),
        )
        remote_copy_op.start()
        remote_copy_op.wait_send()
        # Barrier to ensure both sends completed before exiting start_k
        pl.semaphore_signal(bar, 1, device_id=(peer_id,))
        pl.semaphore_wait(bar, 1)

      start_call = pl.pallas_call(
          start_k,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          interpret=pltpu.InterpretParams(
              dma_execution_mode=dma_mode,
              detect_races=True,
          ),
      )

      # finish skips wait and leaves leftover unconsumed semaphore count
      def finish_k(dst_hbm_ref, recv_sem_in, dst_hbm_out):
        del dst_hbm_out
        # Skips waiting on recv_sem_in!

      finish_call = pl.pallas_call(
          finish_k,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          input_output_aliases={0: 0},
          interpret=pltpu.InterpretParams(
              dma_execution_mode=dma_mode,
              detect_races=True,
          ),
      )

      def bad_protocol_fn(src):
        recv_sem = alloc_call()
        dst = start_call(src, recv_sem)
        res = finish_call(dst, recv_sem)
        return res

      sharded_bad_protocol = jax.jit(
          shard_map.shard_map(
              bad_protocol_fn,
              mesh=mesh,
              in_specs=P(None, 'x'),
              out_specs=P(None, 'x'),
              check_vma=False,
          )
      )

      src = jnp.arange(8 * 128 * 2, dtype=jnp.float32).reshape((8, 128 * 2))
      res = sharded_bad_protocol(src)
      res.block_until_ready()
      # Proves leftover is detected: unconsumed escaped semaphore left with nonzero count
      shared_memory = interpret_lib._get_shared_memory()
      leftovers = shared_memory.get_sempahores_with_nonzero_count(0, include_escaped=True)
      self.assertTrue(len(leftovers) > 0)

  def test_cross_kernel_isolation(self):
    pltpu.reset_tpu_interpret_mode_state()
    # Ensure after reset, an independent kernel starts fresh
    def k(o_ref):
      o_ref[...] = jnp.ones_like(o_ref)

    call = pl.pallas_call(
        k,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        interpret=True,
    )
    res = call()
    np.testing.assert_array_equal(res, jnp.ones((8, 128), jnp.float32))

  def test_shard_map_pl_kernel_closed_over_ref_axis_index(self):
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((num_devices,), ('x',))

    def run_fn(x):
      out_ref = jax.new_ref(x)
      meta_ref = jax.new_ref(x)
      my_id = jnp.int32(lax.axis_index('x'))

      scratch_ty = (
          None,
          pltpu.SMEM((1,), jnp.int32),
      )
      @pl.kernel(
          mesh=pltpu.TensorCoreMesh(axis_name='core', num_cores=1),
          scratch_types=scratch_ty,
      )
      def k(_unused, meta_smem):
        pltpu.sync_copy(meta_ref, meta_smem)
        own = my_id * 2
        meta_smem[0] = own
        pltpu.sync_copy(meta_smem, out_ref)
      k()
      return out_ref[...]

    sharded_run = jax.jit(
        shard_map.shard_map(
            run_fn,
            mesh=mesh,
            in_specs=P('x'),
            out_specs=P('x'),
            check_vma=False,
        ),
        out_shardings=jax.sharding.NamedSharding(mesh, P('x')),
    )

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
      in_arr = jnp.arange(num_devices, dtype=jnp.int32) * 10
      res = sharded_run(in_arr)
      expected = jnp.arange(num_devices, dtype=jnp.int32) * 2
      np.testing.assert_array_equal(res, expected)

  def test_active_cross_kernel_dmas_cleared_after_wait(self):
    """active_cross_kernel_dmas must be cleared after dma_wait().

    If dma_wait() never removes the DMA from active_cross_kernel_dmas,
    has_escaped_semaphores_or_pending_dmas() always returns True after the
    first cross-kernel DMA, so shared memory is never freed and a subsequent
    *independent* kernel reuses the stale shared memory instead of starting
    fresh.  Concretely: run a full protocol (alloc → start → finish), then
    reset, then run a plain kernel, and confirm state is clean.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    for dma_mode in ('eager', 'on_wait'):
      pltpu.reset_tpu_interpret_mode_state()

      # --- alloc kernel: returns a DMA semaphore ---
      def alloc_k(recv_sem_out):
        my_id = lax.axis_index('x')
        peer_id = 1 - my_id
        bar = pltpu.get_barrier_semaphore('ck_dma_clear_bar')
        pl.semaphore_signal(bar, device_id=(peer_id,))

      alloc_call = pl.pallas_call(
          alloc_k,
          out_shape=pltpu.SemaphoreType.DMA(()),
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
      )

      # --- start kernel: starts a remote copy using the escaped recv sem ---
      def start_k(src_hbm_ref, recv_sem_in, dst_hbm_out, send_sem_scratch):
        my_id = lax.axis_index('x')
        right = lax.rem(my_id + 1, jnp.int32(2))
        bar = pltpu.get_barrier_semaphore('ck_dma_clear_bar')
        pl.semaphore_wait(bar, 1)
        pl.semaphore_signal(bar, 1, device_id=(right,))
        pl.semaphore_wait(bar, 1)
        copy_op = pltpu.make_async_remote_copy(
            src_ref=src_hbm_ref,
            dst_ref=dst_hbm_out,
            send_sem=send_sem_scratch,
            recv_sem=recv_sem_in,
            device_id=(right,),
        )
        copy_op.start()
        copy_op.wait_send()
        pl.semaphore_signal(bar, 1, device_id=(right,))
        pl.semaphore_wait(bar, 1)

      start_call = pl.pallas_call(
          start_k,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
      )

      # --- finish kernel: waits on the escaped recv sem ---
      def finish_k(dst_hbm_ref, recv_sem_in, dst_hbm_out):
        del dst_hbm_out
        my_id = lax.axis_index('x')
        left = lax.rem(my_id + 2 - 1, jnp.int32(2))
        fake_dma = pltpu.make_async_remote_copy(
            src_ref=dst_hbm_ref,
            dst_ref=dst_hbm_ref,
            send_sem=recv_sem_in,
            recv_sem=recv_sem_in,
            device_id=(left,),
        )
        fake_dma.wait_recv()

      finish_call = pl.pallas_call(
          finish_k,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          input_output_aliases={0: 0},
          interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
      )

      def protocol_fn(src):
        recv_sem = alloc_call()
        dst = start_call(src, recv_sem)
        return finish_call(dst, recv_sem)

      sharded = jax.jit(shard_map.shard_map(
          protocol_fn, mesh=mesh,
          in_specs=P(None, 'x'), out_specs=P(None, 'x'),
          check_vma=False,
      ))
      src = jnp.arange(8 * 128 * 2, dtype=jnp.float32).reshape((8, 256))
      res = sharded(src)
      res.block_until_ready()

      # KEY ASSERTION: after the protocol completes, active_cross_kernel_dmas
      # must be empty.  If dma_wait() didn't clear it, this will be non-empty
      # and state from this run will leak into the next kernel.
      shared_memory = interpret_lib._get_shared_memory()
      with self.subTest(f'dma_mode={dma_mode}'):
        self.assertEmpty(
            shared_memory.active_cross_kernel_dmas,
            msg=(
                'active_cross_kernel_dmas not cleared after dma_wait(); '
                f'contents={shared_memory.active_cross_kernel_dmas}'
            ),
        )

      pltpu.reset_tpu_interpret_mode_state()

      # Now run a completely independent plain kernel - it must start fresh.
      def plain_k(o_ref):
        o_ref[...] = jnp.ones_like(o_ref)

      plain_call = pl.pallas_call(
          plain_k,
          out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
          out_specs=pl.BlockSpec((4, 32), lambda: (0, 0)),
          interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
      )
      plain_res = plain_call()
      np.testing.assert_array_equal(
          plain_res,
          jnp.ones((4, 32), dtype=jnp.float32),
          err_msg='Plain kernel after reset produced wrong values; '
                  'state leaked from previous protocol run.',
      )

  def test_escaped_sem_leftover_invisible_to_per_kernel_check(self):
    """Escaped sems left nonzero must be invisible from per-kernel leftover check
    (include_escaped=False default) but visible with include_escaped=True.

    If this invariant is violated the per-kernel leftover check would falsely
    flag escaped semaphores that are legitimately consumed in a later kernel.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    pltpu.reset_tpu_interpret_mode_state()

    # alloc kernel: allocate regular sem and immediately signal it to self
    # (so the count is nonzero when the kernel exits)
    def alloc_signal_k(reg_sem_out):
      my_id = lax.axis_index('x')
      pl.semaphore_signal(reg_sem_out, 1, device_id=(my_id,))

    alloc_signal_call = pl.pallas_call(
        alloc_signal_k,
        out_shape=pltpu.SemaphoreType.REGULAR(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=pltpu.InterpretParams(detect_races=False),
    )

    def fn(dummy):
      return dummy, alloc_signal_call()

    sharded = jax.jit(shard_map.shard_map(
        fn, mesh=mesh, in_specs=P('x'), out_specs=(P('x'), P()),
        check_vma=False,
    ))
    _, reg_sem_out = sharded(jnp.zeros(2, dtype=jnp.int32))
    reg_sem_out.block_until_ready()

    shared_memory = interpret_lib._get_shared_memory()

    # With include_escaped=False: should NOT see the escaped sems (they
    # legitimately have pending count for the next kernel to consume).
    leftovers_no_escaped = shared_memory.get_sempahores_with_nonzero_count(
        0, include_escaped=False)
    self.assertEmpty(
        leftovers_no_escaped,
        msg=(
            'Per-kernel leftover check falsely reported escaped semaphores '
            'as leftovers. '
            f'Got: {leftovers_no_escaped}'
        ),
    )

    # With include_escaped=True: should see the nonzero escaped sem.
    leftovers_with_escaped = shared_memory.get_sempahores_with_nonzero_count(
        0, include_escaped=True)
    self.assertNotEmpty(
        leftovers_with_escaped,
        msg=(
            'include_escaped=True should expose escaped sems with nonzero '
            'count, but returned empty. '
            'Escaped semaphore signalling in alloc_signal_k was not tracked.'
        ),
    )

  def test_independent_kernel_after_escaped_sem_full_cycle(self):
    """After a full alloc→finish protocol + reset, an independent kernel must
    start from a completely clean slate (no semaphore IDs, no memory buffers).

    Failure mode: if escaped_semaphore_ids or active_cross_kernel_dmas are not
    cleared during reset, _clear_shared_memory() may skip wiping shared memory
    for the next run, causing the independent kernel to observe stale state.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    for _run_idx in range(2):   # Run the protocol twice to amplify any leak
      pltpu.reset_tpu_interpret_mode_state()

      def alloc_k2(recv_sem_out):
        my_id = lax.axis_index('x')
        peer = 1 - my_id
        bar = pltpu.get_barrier_semaphore('indep_bar')
        pl.semaphore_signal(bar, device_id=(peer,))

      alloc_call2 = pl.pallas_call(
          alloc_k2,
          out_shape=pltpu.SemaphoreType.DMA(()),
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          interpret=pltpu.InterpretParams(),
      )

      def start_k2(src_hbm_ref, recv_sem_in, dst_hbm_out, send_sem_scratch):
        my_id = lax.axis_index('x')
        right = lax.rem(my_id + 1, jnp.int32(2))
        bar = pltpu.get_barrier_semaphore('indep_bar')
        pl.semaphore_wait(bar, 1)
        pl.semaphore_signal(bar, 1, device_id=(right,))
        pl.semaphore_wait(bar, 1)
        copy_op = pltpu.make_async_remote_copy(
            src_ref=src_hbm_ref,
            dst_ref=dst_hbm_out,
            send_sem=send_sem_scratch,
            recv_sem=recv_sem_in,
            device_id=(right,),
        )
        copy_op.start()
        copy_op.wait_send()
        pl.semaphore_signal(bar, 1, device_id=(right,))
        pl.semaphore_wait(bar, 1)

      start_call2 = pl.pallas_call(
          start_k2,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          interpret=pltpu.InterpretParams(),
      )

      def finish_k2(dst_ref, recv_sem_in, out_ref):
        del out_ref
        my_id = lax.axis_index('x')
        left = lax.rem(my_id + 2 - 1, jnp.int32(2))
        # Use a dummy DMA wait to consume the escaped recv sem
        fake = pltpu.make_async_remote_copy(
            src_ref=dst_ref,
            dst_ref=dst_ref,
            send_sem=recv_sem_in,
            recv_sem=recv_sem_in,
            device_id=(left,),
        )
        fake.wait_recv()

      finish_call2 = pl.pallas_call(
          finish_k2,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
          input_output_aliases={0: 0},
          interpret=pltpu.InterpretParams(),
      )

      def protocol_fn2(x):
        recv_sem = alloc_call2()
        dst = start_call2(x, recv_sem)
        return finish_call2(dst, recv_sem)

      sharded2 = jax.jit(shard_map.shard_map(
          protocol_fn2, mesh=mesh,
          in_specs=P(None, 'x'), out_specs=P(None, 'x'),
          check_vma=False,
      ))
      res = sharded2(jnp.ones((4, 64), jnp.float32))
      res.block_until_ready()

    pltpu.reset_tpu_interpret_mode_state()

    # Now run a truly independent kernel; it should not raise or crash.
    def counting_k(o_ref):
      o_ref[...] = jnp.full_like(o_ref, 7)

    counting_call = pl.pallas_call(
        counting_k,
        out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
        out_specs=pl.BlockSpec((4, 32), lambda: (0, 0)),
        interpret=pltpu.InterpretParams(),
    )

    indep_res = counting_call()
    np.testing.assert_array_equal(
        indep_res,
        jnp.full((4, 32), 7, dtype=jnp.float32),
        err_msg='Independent kernel after reset+protocol saw wrong values; '
                'state leaked from previous run.',
    )

  def test_reset_clears_escaped_semaphore_ids(self):
    """reset_tpu_interpret_mode_state must fully discard escaped_semaphore_ids
    so that a subsequent run does not inherit them.

    If reset only sets _shared_memory = None but escaped_semaphore_ids
    persists in an old SharedMemory object that gets reused (which can't
    happen with the current design but would be a future regression), the
    next protocol would falsely believe cross-kernel DMAs are in-flight.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    pltpu.reset_tpu_interpret_mode_state()

    # Run a minimal kernel that escapes a semaphore.
    def alloc_k3(sem_out):
      del sem_out

    alloc_call3 = pl.pallas_call(
        alloc_k3,
        out_shape=pltpu.SemaphoreType.DMA(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=pltpu.InterpretParams(),
    )

    def alloc_fn(dummy):
      return dummy, alloc_call3()

    _, res = jax.jit(shard_map.shard_map(
        alloc_fn, mesh=mesh,
        in_specs=P('x'), out_specs=(P('x'), P()),
        check_vma=False,
    ))(jnp.zeros(2, dtype=jnp.int32))
    res.block_until_ready()

    shared_memory_before = interpret_lib._get_shared_memory()
    self.assertNotEmpty(
        shared_memory_before.escaped_semaphore_ids,
        msg='escaped_semaphore_ids should be non-empty after alloc kernel',
    )

    # Reset, then get a new shared memory from a fresh kernel.
    pltpu.reset_tpu_interpret_mode_state()

    def plain_k2(o_ref):
      o_ref[...] = jnp.zeros_like(o_ref)

    pl.pallas_call(
        plain_k2,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        out_specs=pl.BlockSpec((1,), lambda: (0,)),
        interpret=pltpu.InterpretParams(),
    )()

    # Reset discarded the old shared memory, so the plain kernel created fresh
    # shared memory without escaped semaphores and cleared it on exit.
    jax.effects_barrier()
    self.assertIsNone(interpret_lib._shared_memory)

  def test_two_cross_kernel_dmas_one_escaped_recv_sem(self):
    """Two distinct DMAs whose recv (dst) semaphore is the SAME escaped sem.

    Attack: dma_wait on that sem clears ALL entries in active_cross_kernel_dmas
    whose cross_kernel_dma_sem_ids[dma_id][0] == sem_id.  After ONE wait both
    entries are removed, so the second DMA is no longer tracked.
    has_escaped_semaphores_or_pending_dmas() may then return False and clear
    shared memory before the second DMA is consumed, causing the second recv
    to deadlock or use a stale semaphore.

    If the implementation is correct, running two successive rounds of this
    protocol without an explicit reset must NOT deadlock or raise.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    for dma_mode in ('eager', 'on_wait'):
      pltpu.reset_tpu_interpret_mode_state()

      # alloc kernel: returns ONE escaped recv sem
      def alloc_k_two(recv_sem_out):  # pylint: disable=unused-argument
        pass

      alloc_call_two = pl.pallas_call(
          alloc_k_two,
          out_shape=pltpu.SemaphoreType.DMA(()),
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
      )

      def start_k_two(src_ref, recv_sem_in, dst_out, send_sem_scratch):
        my_id = lax.axis_index('x')
        right = lax.rem(my_id + 1, jnp.int32(2))
        bar = pltpu.get_barrier_semaphore('two_dma_bar')
        pl.semaphore_signal(bar, 1, device_id=(right,))
        pl.semaphore_wait(bar, 1)
        # First remote copy using the escaped recv_sem
        copy1 = pltpu.make_async_remote_copy(
            src_ref=src_ref,
            dst_ref=dst_out,
            send_sem=send_sem_scratch,
            recv_sem=recv_sem_in,
            device_id=(right,),
        )
        copy1.start()
        copy1.wait_send()
        pl.semaphore_signal(bar, 1, device_id=(right,))
        pl.semaphore_wait(bar, 1)

      start_call_two = pl.pallas_call(
          start_k_two,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
      )

      def finish_k_two(dst_ref, recv_sem_in, out_ref):
        del out_ref
        my_id = lax.axis_index('x')
        left = lax.rem(my_id + 2 - 1, jnp.int32(2))
        fake = pltpu.make_async_remote_copy(
            src_ref=dst_ref,
            dst_ref=dst_ref,
            send_sem=recv_sem_in,
            recv_sem=recv_sem_in,
            device_id=(left,),
        )
        fake.wait_recv()

      finish_call_two = pl.pallas_call(
          finish_k_two,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
          input_output_aliases={0: 0},
          interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
      )

      # Run the protocol TWICE without reset between rounds to stress that
      # active_cross_kernel_dmas does not accumulate stale entries.
      def protocol_two(x):
        recv_sem = alloc_call_two()
        dst = start_call_two(x, recv_sem)
        return finish_call_two(dst, recv_sem)

      sharded_two = jax.jit(shard_map.shard_map(
          protocol_two, mesh=mesh,
          in_specs=P(None, 'x'), out_specs=P(None, 'x'),
          check_vma=False,
      ))
      src = jnp.ones((4, 64), jnp.float32)
      # Round 1
      res1 = sharded_two(src)
      res1.block_until_ready()
      pltpu.reset_tpu_interpret_mode_state()
      # Round 2 — if stale entries leaked, this will deadlock or raise
      res2 = sharded_two(src)
      res2.block_until_ready()

      with self.subTest(f'two_dma_one_sem_dma_mode={dma_mode}'):
        np.testing.assert_array_equal(
            res2, jnp.ones((4, 64), jnp.float32),
            err_msg=(
                f'Round 2 of two-DMA-one-sem protocol produced wrong values '
                f'(dma_mode={dma_mode}). '
                'Stale cross_kernel_dma entries likely caused wrong sem binding.'
            ),
        )

  def test_regular_sem_signal_call1_wait_call2(self):
    """A REGULAR semaphore signalled in kernel 1 is waited in kernel 2.

    This is the simplest escaped-sem cross-call pattern:
    - alloc_k: allocates REGULAR sem, signals it to self (count → 1)
    - wait_k: waits on that REGULAR sem (count → 0)
    Both kernels are separate pallas_call invocations.
    If escaped_semaphore_ids is not set or shared memory is cleared between
    calls, the wait will deadlock or reference a nonexistent semaphore.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    for detect_races in (False, True):
      pltpu.reset_tpu_interpret_mode_state()

      def alloc_signal_reg(reg_sem_out):
        my_id = lax.axis_index('x')
        # Signal self so count is 1 when kernel exits
        pl.semaphore_signal(reg_sem_out, 1, device_id=(my_id,))

      alloc_signal_call = pl.pallas_call(
          alloc_signal_reg,
          out_shape=pltpu.SemaphoreType.REGULAR(()),
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          interpret=pltpu.InterpretParams(detect_races=detect_races),
      )

      def wait_reg(reg_sem_in, out_ref):
        pl.semaphore_wait(reg_sem_in, 1)
        out_ref[0] = 1

      wait_call = pl.pallas_call(
          wait_reg,
          in_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
          out_specs=pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM),
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          interpret=pltpu.InterpretParams(detect_races=detect_races),
      )

      def protocol_reg(dummy):
        reg_sem = alloc_signal_call()
        result = wait_call(reg_sem)
        return dummy, result

      sharded_reg = jax.jit(shard_map.shard_map(
          protocol_reg, mesh=mesh,
          in_specs=P('x'), out_specs=(P('x'), P()),
          check_vma=False,
      ))

      _, result = sharded_reg(jnp.zeros(2, jnp.int32))
      result.block_until_ready()

      with self.subTest(f'reg_sem_cross_call_detect_races={detect_races}'):
        np.testing.assert_array_equal(
            result,
            jnp.ones((1,), jnp.int32),
            err_msg=(
                f'REGULAR sem cross-call signal→wait produced wrong result '
                f'(detect_races={detect_races}). Shared memory may have been '
                'cleared between alloc_signal_call and wait_call.'
            ),
        )

  def test_on_wait_src_sem_task_does_not_block_cleanup(self):
    """In on_wait mode, tasks are appended to BOTH src and dst sems for a
    cross-kernel DMA.  After dma_wait on the dst sem, the src sem task should
    either have already executed or should not prevent cleanup.

    If the src sem task lingers in sem.tasks after the dst sem wait,
    has_escaped_semaphores_or_pending_dmas() returns True forever, shared
    memory is never freed, and the next protocol run reuses stale state.

    Attack: run a full 3-kernel alloc→start→finish protocol in on_wait mode,
    confirm that after finish + reset the shared memory is actually None
    (not held alive by pending src_sem tasks).
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    pltpu.reset_tpu_interpret_mode_state()

    def alloc_k_ow(recv_sem_out):  # pylint: disable=unused-argument
      pass

    alloc_ow = pl.pallas_call(
        alloc_k_ow,
        out_shape=pltpu.SemaphoreType.DMA(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=pltpu.InterpretParams(dma_execution_mode='on_wait'),
    )

    def start_k_ow(src_ref, recv_sem_in, dst_out, send_sem):
      my_id = lax.axis_index('x')
      right = lax.rem(my_id + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore('ow_bar')
      pl.semaphore_signal(bar, 1, device_id=(right,))
      pl.semaphore_wait(bar, 1)
      copy_op = pltpu.make_async_remote_copy(
          src_ref=src_ref,
          dst_ref=dst_out,
          send_sem=send_sem,
          recv_sem=recv_sem_in,
          device_id=(right,),
      )
      copy_op.start()
      copy_op.wait_send()
      pl.semaphore_signal(bar, 1, device_id=(right,))
      pl.semaphore_wait(bar, 1)

    start_ow = pl.pallas_call(
        start_k_ow,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(dma_execution_mode='on_wait'),
    )

    def finish_k_ow(dst_ref, recv_sem_in, out_ref):
      del out_ref
      my_id = lax.axis_index('x')
      left = lax.rem(my_id + 2 - 1, jnp.int32(2))
      fake = pltpu.make_async_remote_copy(
          src_ref=dst_ref,
          dst_ref=dst_ref,
          send_sem=recv_sem_in,
          recv_sem=recv_sem_in,
          device_id=(left,),
      )
      fake.wait_recv()

    finish_ow = pl.pallas_call(
        finish_k_ow,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((4, 32), jnp.float32),
        input_output_aliases={0: 0},
        interpret=pltpu.InterpretParams(dma_execution_mode='on_wait'),
    )

    def protocol_ow(x):
      recv_sem = alloc_ow()
      dst = start_ow(x, recv_sem)
      return finish_ow(dst, recv_sem)

    sharded_ow = jax.jit(shard_map.shard_map(
        protocol_ow, mesh=mesh,
        in_specs=P(None, 'x'), out_specs=P(None, 'x'),
        check_vma=False,
    ))
    src = jnp.ones((4, 64), jnp.float32)
    res = sharded_ow(src)
    res.block_until_ready()

    # After the protocol completes, explicitly reset.
    pltpu.reset_tpu_interpret_mode_state()

    # Now run a plain independent kernel – should not deadlock or fail.
    def plain_ow(o_ref):
      o_ref[...] = jnp.full_like(o_ref, 42)

    plain_res = pl.pallas_call(
        plain_ow,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        out_specs=pl.BlockSpec((1,), lambda: (0,)),
        interpret=pltpu.InterpretParams(dma_execution_mode='on_wait'),
    )()
    np.testing.assert_array_equal(
        plain_res,
        jnp.full((1,), 42, jnp.int32),
        err_msg=(
            'Plain kernel after on_wait protocol + reset produced wrong result. '
            'Lingering src_sem tasks in on_wait mode may have blocked '
            '_clear_shared_memory, causing state leak.'
        ),
    )

  def test_escaped_sem_id_binding_with_data_hbm_input_before(self):
    """Semaphore-typed input that follows a regular HBM input.

    The input binding loop iterates i over jaxpr.invars[num_index_operands:]
    and does `sem_id = input_args[i]`.  But input_args only covers the actual
    inputs (not outputs/scratch), so when i=1 (second input is the semaphore),
    input_args[1] is the semaphore – this should be correct.

    Attack: put a float HBM buffer as the first input and the escaped sem as
    the second input, then confirm the kernel can signal the sem and the signal
    persists to a third kernel that waits it.  If the index is off by 1 the
    kernel gets the float buffer as its semaphore, which will crash or corrupt.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))
    pltpu.reset_tpu_interpret_mode_state()

    # Alloc kernel: returns a REGULAR semaphore
    def alloc_k_bind(reg_sem_out):  # pylint: disable=unused-argument
      pass

    alloc_bind = pl.pallas_call(
        alloc_k_bind,
        out_shape=pltpu.SemaphoreType.REGULAR(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=pltpu.InterpretParams(),
    )

    # Signal kernel: takes (float_data_hbm, reg_sem) – float comes FIRST
    def signal_k_bind(float_hbm_ref, reg_sem_in):
      del float_hbm_ref
      my_id = lax.axis_index('x')
      pl.semaphore_signal(reg_sem_in, 1, device_id=(my_id,))

    signal_bind = pl.pallas_call(
        signal_k_bind,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),    # float HBM – comes first
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),  # sem – second
        ],
        out_shape=[],
        interpret=pltpu.InterpretParams(),
    )

    # Wait kernel: takes (float_data_hbm, reg_sem) – waits on the sem
    def wait_k_bind(float_hbm_ref, reg_sem_in, out_ref):
      del float_hbm_ref
      pl.semaphore_wait(reg_sem_in, 1)
      out_ref[0] = 1

    wait_bind = pl.pallas_call(
        wait_k_bind,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM),
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        interpret=pltpu.InterpretParams(),
    )

    def protocol_bind(float_data, dummy):
      reg_sem = alloc_bind()
      signal_bind(float_data, reg_sem)
      result = wait_bind(float_data, reg_sem)
      return dummy, result

    sharded_bind = jax.jit(shard_map.shard_map(
        protocol_bind, mesh=mesh,
        in_specs=(P(None, 'x'), P('x')),
        out_specs=(P('x'), P()),
        check_vma=False,
    ))

    float_data = jnp.ones((4, 64), jnp.float32)
    dummy = jnp.zeros(2, jnp.int32)
    _, result = sharded_bind(float_data, dummy)
    result.block_until_ready()

    np.testing.assert_array_equal(
        result,
        jnp.ones((1,), jnp.int32),
        err_msg=(
            'HBM-input-then-sem cross-kernel signal→wait produced wrong result. '
            'The sem binding index (input_args[i]) may be off when a non-sem '
            'input precedes the escaped semaphore input.'
        ),
    )

  def test_multiple_escaped_sems_cross_kernel_cleanup(self):
    """When multiple semaphores are escaped, all of them must be cleared from
    escaped_semaphore_ids during reset, so a fresh protocol starting on the
    same shared memory does not accumulate stale IDs.

    Attack: run two back-to-back protocols each escaping a DMA sem; check that
    after the second protocol's finish + reset, shared memory is fully cleaned
    up and a plain kernel runs correctly.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    for _trial in range(3):  # Three repetitions to amplify any leak
      pltpu.reset_tpu_interpret_mode_state()

      def alloc_mk(s1_out, s2_out):  # pylint: disable=unused-argument
        pass

      alloc_multi = pl.pallas_call(
          alloc_mk,
          out_shape=[
              pltpu.SemaphoreType.DMA(()),
              pltpu.SemaphoreType.REGULAR(()),
          ],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          interpret=pltpu.InterpretParams(),
      )

      def consume_mk(dummy, s1_in, s2_in, out_ref):
        del s1_in, s2_in
        out_ref[0] = 1

      consume_multi = pl.pallas_call(
          consume_mk,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM),
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          interpret=pltpu.InterpretParams(),
      )

      def protocol_multi(x, dummy):
        s1, s2 = alloc_multi()
        result = consume_multi(x, s1, s2)
        return dummy, result

      sharded_multi = jax.jit(shard_map.shard_map(
          protocol_multi, mesh=mesh,
          in_specs=(P(None, 'x'), P('x')),
          out_specs=(P('x'), P()),
          check_vma=False,
      ))

      float_data = jnp.ones((4, 64), jnp.float32)
      dummy = jnp.zeros(2, jnp.int32)
      _, result = sharded_multi(float_data, dummy)
      result.block_until_ready()

    np.testing.assert_array_equal(
        result,
        jnp.ones((1,), jnp.int32),
        err_msg=(
            'Multi-escaped-sem protocol produced wrong result on trial '
            f'{_trial}. Stale escaped_semaphore_ids may have caused cleanup '
            'to be skipped, leading to wrong sem binding in the next trial.'
        ),
    )

  def test_aliased_semaphore_input_output(self):
    """Regression test: aliased semaphore input/output in pallas_call.

    When inout arguments include semaphores (input_output_aliases maps an input
    semaphore to an output), interpret_pallas_call must treat the aliased
    output as a semaphore (setting output_buffer_id to None and not querying
    input_buffer_ids which attempts to read from HBM using a non-numpy dma_sem dtype).
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))

    pltpu.reset_tpu_interpret_mode_state()

    def alloc_k(s_out):
      pass

    alloc_fn = pl.pallas_call(
        alloc_k,
        out_shape=pltpu.SemaphoreType.DMA(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=pltpu.InterpretParams(),
    )

    def inout_k(x_in, s_in, x_out, s_out, scratch):
      del s_in, s_out
      pltpu.sync_copy(x_in, scratch)
      scratch[...] = scratch[...] + 1
      pltpu.sync_copy(scratch, x_out)

    inout_call = pl.pallas_call(
        inout_k,
        in_specs=(
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ),
        out_specs=(
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((8, 128), jnp.float32),
            pltpu.SemaphoreType.DMA(()),
        ),
        scratch_shapes=[pltpu.VMEM((8, 128), jnp.float32)],
        input_output_aliases={0: 0, 1: 1},
        interpret=pltpu.InterpretParams(),
    )

    def step(x):
      sem = alloc_fn()
      x_out, sem_out = inout_call(x, sem)
      return x_out

    sharded = jax.jit(
        shard_map.shard_map(
            step,
            mesh=mesh,
            in_specs=P(None, 'x'),
            out_specs=P(None, 'x'),
            check_vma=False,
        )
    )
    x_in = jnp.zeros((8, 256), dtype=jnp.float32)
    res = sharded(x_in)
    np.testing.assert_array_equal(res, jnp.ones_like(x_in))

  def test_aliased_semaphore_output(self):
    """Tests that an aliased semaphore output reuses the input semaphore ID."""
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((2,), ('x',))
    pltpu.reset_tpu_interpret_mode_state()

    def alloc_k(s_out):
      pass

    alloc_fn = pl.pallas_call(
        alloc_k,
        out_shape=pltpu.SemaphoreType.REGULAR(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=pltpu.InterpretParams(),
    )

    def aliased_k(s_in, s_out):
      del s_out
      my_id = lax.axis_index('x')
      right = lax.rem(my_id + 1, jnp.int32(2))
      pl.semaphore_signal(s_in, 1, device_id=(right,))

    aliased_fn = pl.pallas_call(
        aliased_k,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
        out_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
        out_shape=[pltpu.SemaphoreType.REGULAR(())],
        input_output_aliases={0: 0},
        interpret=pltpu.InterpretParams(),
    )

    def wait_k(s_in, out_ref):
      pl.semaphore_wait(s_in, 1)
      out_ref[0] = 1

    wait_fn = pl.pallas_call(
        wait_k,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
        out_specs=pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM),
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        interpret=pltpu.InterpretParams(),
    )

    def protocol(dummy):
      s = alloc_fn()
      (s_aliased,) = aliased_fn(s)
      result = wait_fn(s_aliased)
      return dummy, result

    sharded = jax.jit(shard_map.shard_map(
        protocol, mesh=mesh,
        in_specs=P('x'), out_specs=(P('x'), P()),
        check_vma=False,
    ))

    _, res = sharded(jnp.zeros(2, jnp.int32))
    res.block_until_ready()
    np.testing.assert_array_equal(res, jnp.ones((1,), jnp.int32))

  def test_cross_kernel_tagged_barrier_race_detected(self):
    """Verifies that tagged barrier kernels do not perform vector clock sync at entry.

    If vector clocks were globally synced before kernel execution, an unsynchronized
    cross-kernel write after an un-waited signal would be masked. This test proves
    that omitting full vector clock sync at kernel entry preserves race/leftover detection.
    """
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((num_devices,), ('x',))

    for dma_mode in ('eager', 'on_wait'):
      pltpu.reset_tpu_interpret_mode_state()

      # Step 1: allocate escaped regular semaphore and signal tagged barrier
      def step1_k(dummy_in, s_out, o_ref):
        del dummy_in, s_out
        my_id = lax.axis_index('x')
        peer_id = lax.rem(my_id + 1, jnp.int32(num_devices))
        bar = pltpu.get_barrier_semaphore('test_tag')
        pl.semaphore_signal(bar, 1, device_id=(peer_id,))
        o_ref[0] = 42

      step1 = pl.pallas_call(
          step1_k,
          in_specs=[pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM)],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
              pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM),
          ],
          out_shape=[
              pltpu.SemaphoreType.REGULAR(()),
              jax.ShapeDtypeStruct((1,), jnp.int32),
          ],
          interpret=pltpu.InterpretParams(
              dma_execution_mode=dma_mode,
              detect_races=True,
          ),
      )

      # Step 2: takes escaped semaphore and tagged barrier, omits waiting on barrier
      def step2_bad_k(dummy_in, s_in, o_ref):
        del dummy_in, s_in
        bar = pltpu.get_barrier_semaphore('test_tag')
        del bar
        o_ref[0] = 99

      step2 = pl.pallas_call(
          step2_bad_k,
          in_specs=[
              pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec((1,), lambda: (0,), memory_space=pltpu.SMEM),
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          interpret=pltpu.InterpretParams(
              dma_execution_mode=dma_mode,
              detect_races=True,
          ),
      )

      def bad_protocol(dummy):
        s, x1 = step1(dummy)
        x2 = step2(dummy, s)
        return x1, x2

      sharded = jax.jit(
          shard_map.shard_map(
              bad_protocol,
              mesh=mesh,
              in_specs=P('x'),
              out_specs=(P('x'), P('x')),
              check_vma=False,
          )
      )

      res1, res2 = sharded(jnp.zeros(num_devices, jnp.int32))
      res1.block_until_ready()
      res2.block_until_ready()
      shared_memory = interpret_lib._get_shared_memory()
      leftovers = shared_memory.get_sempahores_with_nonzero_count(0, include_escaped=True)
      self.assertTrue(len(leftovers) > 0)

  def test_remote_dma_into_output_buffer_detected(self):
    """Verifies that an in-flight remote DMA into an output buffer at kernel exit is detected."""
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('requires at least 2 devices')
    mesh = jtu.create_mesh((num_devices,), ('x',))

    for dma_mode in ('eager', 'on_wait'):
      pltpu.reset_tpu_interpret_mode_state()

      def alloc_k(recv_sem_out):
        pass

      alloc_fn = pl.pallas_call(
          alloc_k,
          out_shape=pltpu.SemaphoreType.DMA(()),
          out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          interpret=pltpu.InterpretParams(),
      )

      def bad_writer_k(src_hbm_ref, recv_sem_in, dst_out, send_sem_scratch):
        my_id = lax.axis_index('x')
        peer_id = lax.rem(my_id + 1, jnp.int32(num_devices))
        # Start remote copy into peer's dst_out
        remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=src_hbm_ref,
            dst_ref=dst_out,
            send_sem=send_sem_scratch,
            recv_sem=recv_sem_in,
            device_id=(peer_id,),
        )
        remote_copy_op.start()
        # Wait send so local send is done, but peer does NOT wait recv in this kernel
        remote_copy_op.wait_send()

      writer_call = pl.pallas_call(
          bad_writer_k,
          in_specs=[
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          interpret=pltpu.InterpretParams(
              dma_execution_mode=dma_mode,
              detect_races=True,
          ),
      )

      def protocol(src):
        sem = alloc_fn()
        res = writer_call(src, sem)
        return res

      sharded = jax.jit(
          shard_map.shard_map(
              protocol,
              mesh=mesh,
              in_specs=P(None, 'x'),
              out_specs=P(None, 'x'),
              check_vma=False,
          )
      )

      src = jnp.arange(8 * 128 * num_devices, dtype=jnp.float32).reshape((8, 128 * num_devices))
      res = sharded(src)
      res.block_until_ready()
      shared_memory = interpret_lib._get_shared_memory()
      # Leftover semaphore count or leftover active DMA must be detected
      leftovers = shared_memory.get_sempahores_with_nonzero_count(0, include_escaped=True)
      has_leftovers = len(leftovers) > 0 or len(shared_memory.active_cross_kernel_dmas) > 0
      self.assertTrue(has_leftovers)

  @parameterized.parameters("eager", "on_wait")
  def test_split_kernel_smem_fetch_clean(self, dma_mode):
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def prefetch_start(arr_ref, smem_ref, sem_ref):
      pltpu.async_copy(arr_ref, smem_ref, sem_ref)

    def prefetch_done(arr_ref, _, sem_ref, smem_out_ref):
      pltpu.make_async_copy(arr_ref, smem_out_ref, sem_ref).wait()

    def fn(arr):
      smem, sem = pl.pallas_call(
          prefetch_start,
          out_shape=(
              pltpu.SMEM(arr.shape, dtype=arr.dtype),
              pltpu.SemaphoreType.DMA,
          ),
          in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
          out_specs=(
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ),
          interpret=interpret,
      )(arr)
      return pl.pallas_call(
          prefetch_done,
          out_shape=pltpu.SMEM(smem.shape, dtype=smem.dtype),
          in_specs=(
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ),
          out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
          input_output_aliases={1: 0},
          interpret=interpret,
      )(arr, smem, sem)

    x = jnp.arange(16, dtype=jnp.int32)
    mesh = jax.sharding.Mesh(jax.devices(), ('x',))
    out = jax.jit(
        jax.shard_map(
            fn, mesh=mesh, in_specs=jax.sharding.PartitionSpec(),
            out_specs=jax.sharding.PartitionSpec(),
            check_vma=False,
        )
    )(x)
    np.testing.assert_array_equal(out, x)

  def test_split_kernel_smem_fetch_read_before_wait_race(self):
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode="eager"
    )

    def prefetch_start(arr_ref, smem_ref, sem_ref):
      pltpu.async_copy(arr_ref, smem_ref, sem_ref)

    def bad_reader(smem_ref, sem_ref, out_ref):
      del sem_ref
      out_ref[...] = smem_ref[...]

    def fn(arr):
      smem, sem = pl.pallas_call(
          prefetch_start,
          out_shape=(
              pltpu.SMEM(arr.shape, dtype=arr.dtype),
              pltpu.SemaphoreType.DMA,
          ),
          in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
          out_specs=(
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ),
          interpret=interpret,
      )(arr)
      return pl.pallas_call(
          bad_reader,
          out_shape=jax.ShapeDtypeStruct(arr.shape, arr.dtype),
          in_specs=(
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ),
          out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
          interpret=interpret,
      )(smem, sem)

    x = jnp.arange(16, dtype=jnp.int32)
    mesh = jax.sharding.Mesh(jax.devices(), ('x',))
    with self.assertRaisesRegex(Exception, "RACE DETECTED"):
      jax.block_until_ready(
          jax.jit(
              jax.shard_map(
                  fn, mesh=mesh, in_specs=jax.sharding.PartitionSpec(),
                  out_specs=jax.sharding.PartitionSpec(),
            check_vma=False,
              )
          )(x)
      )

  @parameterized.parameters(('eager', False), ('eager', True), ('on_wait', False), ('on_wait', True))
  def test_cross_kernel_remote_dma_output_buffer_clean(self, dma_mode, detect_races):
    interpret = pltpu.InterpretParams(
        detect_races=detect_races, dma_execution_mode=dma_mode
    )

    def alloc_k(recv_sem_out, reg_sem_out):
      my_id = lax.axis_index('x')
      peer_id = 1 - my_id
      bar = pltpu.get_barrier_semaphore('test_barrier')
      pl.semaphore_signal(bar, device_id=(peer_id,))

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=[pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.REGULAR(())],
        out_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE), pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
        interpret=interpret,
    )

    def start_k(src_ref, recv_sem_in, reg_sem_in, dst_out, send_sem_scratch):
      my_id = lax.axis_index('x')
      right_neighbor = lax.rem(my_id + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore('test_barrier')
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(reg_sem_in, 1, device_id=(right_neighbor,))
      remote_copy_op = pltpu.make_async_remote_copy(
          src_ref=src_ref,
          dst_ref=dst_out,
          send_sem=send_sem_scratch,
          recv_sem=recv_sem_in,
          device_id=(right_neighbor,),
      )
      remote_copy_op.start()
      remote_copy_op.wait_send()
      pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
      pl.semaphore_wait(bar, 1)

    start_call = pl.pallas_call(
        start_k,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        interpret=interpret,
    )

    def finish_k(dst_ref, recv_sem_in, reg_sem_in, dst_out):
      del dst_out
      my_id = lax.axis_index('x')
      left_neighbor = lax.rem(my_id + 2 - 1, jnp.int32(2))
      pl.semaphore_wait(reg_sem_in, 1)
      fake_dma = pltpu.make_async_remote_copy(
          src_ref=dst_ref,
          dst_ref=dst_ref,
          send_sem=recv_sem_in,
          recv_sem=recv_sem_in,
          device_id=(left_neighbor,),
      )
      fake_dma.wait_recv()

    finish_call = pl.pallas_call(
        finish_k,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        input_output_aliases={0: 0},
        interpret=interpret,
    )

    def protocol_fn(src):
      recv_sem, reg_sem = alloc_call()
      dst = start_call(src, recv_sem, reg_sem)
      res = finish_call(dst, recv_sem, reg_sem)
      return res

    mesh = jax.sharding.Mesh(jax.devices()[:2], ('x',))
    sharded_protocol = jax.jit(
        jax.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=jax.sharding.PartitionSpec(None, 'x'),
            out_specs=jax.sharding.PartitionSpec(None, 'x'),
            check_vma=False,
        )
    )
    src = jnp.arange(8 * 128 * 2, dtype=jnp.float32).reshape((8, 128 * 2))
    res = sharded_protocol(src)
    np.testing.assert_array_equal(res[:, :128], src[:, 128:])
    np.testing.assert_array_equal(res[:, 128:], src[:, :128])

  @parameterized.parameters(('eager', False), ('eager', True), ('on_wait', False), ('on_wait', True))
  def test_cross_kernel_scalar_smem_count_dma_clean(self, dma_mode, detect_races):
    interpret = pltpu.InterpretParams(
        detect_races=detect_races, dma_execution_mode=dma_mode
    )

    def alloc_k(recv_sem_out, reg_sem_out):
      my_id = lax.axis_index('x')
      peer_id = 1 - my_id
      bar = pltpu.get_barrier_semaphore('test_barrier_cnt')
      pl.semaphore_signal(bar, device_id=(peer_id,))

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=[pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.REGULAR(())],
        out_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE), pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
        interpret=interpret,
    )

    def start_k(src_ref, recv_sem_in, reg_sem_in, dst_out, send_sem_scratch):
      my_id = lax.axis_index('x')
      right_neighbor = lax.rem(my_id + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore('test_barrier_cnt')
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(reg_sem_in, 1, device_id=(right_neighbor,))
      remote_copy_op = pltpu.make_async_remote_copy(
          src_ref=src_ref,
          dst_ref=dst_out,
          send_sem=send_sem_scratch,
          recv_sem=recv_sem_in,
          device_id=(right_neighbor,),
      )
      remote_copy_op.start()
      remote_copy_op.wait_send()
      pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
      pl.semaphore_wait(bar, 1)

    start_call = pl.pallas_call(
        start_k,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        out_shape=pltpu.SMEM((1,), jnp.uint32),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        interpret=interpret,
    )

    def finish_k(dst_ref, recv_sem_in, reg_sem_in, dst_out):
      del dst_out
      my_id = lax.axis_index('x')
      left_neighbor = lax.rem(my_id + 2 - 1, jnp.int32(2))
      pl.semaphore_wait(reg_sem_in, 1)
      fake_dma = pltpu.make_async_remote_copy(
          src_ref=dst_ref,
          dst_ref=dst_ref,
          send_sem=recv_sem_in,
          recv_sem=recv_sem_in,
          device_id=(left_neighbor,),
      )
      fake_dma.wait_recv()

    finish_call = pl.pallas_call(
        finish_k,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        out_shape=pltpu.SMEM((1,), jnp.uint32),
        input_output_aliases={0: 0},
        interpret=interpret,
    )

    def protocol_fn(src):
      recv_sem, reg_sem = alloc_call()
      dst = start_call(src, recv_sem, reg_sem)
      res = finish_call(dst, recv_sem, reg_sem)
      return res

    mesh = jax.sharding.Mesh(jax.devices()[:2], ('x',))
    sharded_protocol = jax.jit(
        jax.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=jax.sharding.PartitionSpec('x'),
            out_specs=jax.sharding.PartitionSpec('x'),
            check_vma=False,
        )
    )
    src = jnp.array([[42], [99]], dtype=jnp.uint32)
    res = sharded_protocol(src)
    np.testing.assert_array_equal(res, jnp.array([99, 42], dtype=jnp.uint32))

  def test_cross_kernel_remote_dma_read_before_wait_race(self):
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode='eager'
    )

    def alloc_k(recv_sem_out, reg_sem_out):
      my_id = lax.axis_index('x')
      peer_id = 1 - my_id
      bar = pltpu.get_barrier_semaphore('test_barrier_race')
      pl.semaphore_signal(bar, device_id=(peer_id,))

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=[pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.REGULAR(())],
        out_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE), pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
        interpret=interpret,
    )

    def start_k(src_ref, recv_sem_in, reg_sem_in, dst_out, send_sem_scratch):
      my_id = lax.axis_index('x')
      right_neighbor = lax.rem(my_id + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore('test_barrier_race')
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(reg_sem_in, 1, device_id=(right_neighbor,))
      remote_copy_op = pltpu.make_async_remote_copy(
          src_ref=src_ref,
          dst_ref=dst_out,
          send_sem=send_sem_scratch,
          recv_sem=recv_sem_in,
          device_id=(right_neighbor,),
      )
      remote_copy_op.start()
      remote_copy_op.wait_send()
      pl.semaphore_signal(bar, 1, device_id=(right_neighbor,))
      pl.semaphore_wait(bar, 1)

    start_call = pl.pallas_call(
        start_k,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        interpret=interpret,
    )

    def bad_finish_k(dst_ref, recv_sem_in, reg_sem_in, dst_out):
      del dst_out, recv_sem_in, reg_sem_in
      # Reads dst_ref directly in VMEM without waiting on recv_sem!
      dst_ref[...] = dst_ref[...] + 1.0

    bad_finish_call = pl.pallas_call(
        bad_finish_k,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        input_output_aliases={0: 0},
        interpret=interpret,
    )

    def protocol_fn(src):
      recv_sem, reg_sem = alloc_call()
      dst = start_call(src, recv_sem, reg_sem)
      res = bad_finish_call(dst, recv_sem, reg_sem)
      return res

    mesh = jax.sharding.Mesh(jax.devices()[:2], ('x',))
    sharded_protocol = jax.jit(
        jax.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=jax.sharding.PartitionSpec(None, 'x'),
            out_specs=jax.sharding.PartitionSpec(None, 'x'),
            check_vma=False,
        )
    )
    src = jnp.arange(8 * 128 * 2, dtype=jnp.float32).reshape((8, 128 * 2))
    with self.assertRaisesRegex(Exception, 'RACE DETECTED'):
      jax.block_until_ready(sharded_protocol(src))

  def _two_escaped_buffers_protocol(
      self, interpret, finish_k, tag, diverge_buffer_ids=False
  ):
    """Each device DMAs two same-shape buffers into its peer's kernel outputs.

    The receive semaphores escape the sending pallas_call, and `finish_k`
    receives both destination buffers as VMEM inputs, plus both semaphores.
    If `diverge_buffer_ids`, the destination buffers have different buffer ids
    on the two devices.
    """

    def alloc_k(sem_a_out, sem_b_out, reg_sem_out):
      del sem_a_out, sem_b_out
      my_id = lax.axis_index('x')
      if diverge_buffer_ids:
        # One more buffer on device 0. The escaped semaphores keep the shared
        # memory, and so the buffer id counters, alive across pallas_calls.
        @pl.when(my_id == 0)
        def _():
          @functools.partial(
              pl.run_scoped, b=pltpu.VMEM((8, 128), jnp.float32)
          )
          def _(b):
            b[...] = jnp.zeros((8, 128), dtype=jnp.float32)
            # An effect on a kernel operand, so that the allocation is not
            # removed as dead code.
            pl.semaphore_signal(reg_sem_out, 1)
            pl.semaphore_wait(reg_sem_out, 1)

      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, device_id=(1 - my_id,))

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=[
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.REGULAR(()),
        ],
        out_specs=[pl.BlockSpec(memory_space=pltpu.SEMAPHORE)] * 3,
        interpret=interpret,
    )

    def start_k(src_a, src_b, sem_a, sem_b, reg_sem, dst_a, dst_b,
                send_sem_a, send_sem_b):
      my_id = lax.axis_index('x')
      peer = lax.rem(my_id + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(reg_sem, 1, device_id=(peer,))
      copy_a = pltpu.make_async_remote_copy(
          src_ref=src_a, dst_ref=dst_a, send_sem=send_sem_a, recv_sem=sem_a,
          device_id=(peer,),
      )
      copy_b = pltpu.make_async_remote_copy(
          src_ref=src_b, dst_ref=dst_b, send_sem=send_sem_b, recv_sem=sem_b,
          device_id=(peer,),
      )
      copy_a.start()
      copy_b.start()
      copy_a.wait_send()
      copy_b.wait_send()
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    out_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    start_call = pl.pallas_call(
        start_k,
        in_specs=[vmem, vmem, sem, sem, sem],
        out_specs=[vmem, vmem],
        out_shape=[out_shape, out_shape],
        scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
        interpret=interpret,
    )
    finish_call = pl.pallas_call(
        finish_k,
        in_specs=[vmem, vmem, sem, sem, sem],
        out_specs=[vmem, vmem],
        out_shape=[out_shape, out_shape],
        input_output_aliases={0: 0, 1: 1},
        interpret=interpret,
    )

    def protocol_fn(src_a, src_b):
      sem_a, sem_b, reg_sem = alloc_call()
      dst_a, dst_b = start_call(src_a, src_b, sem_a, sem_b, reg_sem)
      return finish_call(dst_a, dst_b, sem_a, sem_b, reg_sem)

    mesh = jax.sharding.Mesh(jax.devices()[:2], ('x',))
    spec = jax.sharding.PartitionSpec(None, 'x')
    return jax.jit(
        jax.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )

  @parameterized.parameters(
      ('eager', False), ('eager', True), ('on_wait', False), ('on_wait', True)
  )
  def test_cross_kernel_dma_two_escaped_buffers_bound_one_to_one(
      self, dma_mode, detect_races
  ):
    interpret = pltpu.InterpretParams(
        detect_races=detect_races, dma_execution_mode=dma_mode
    )

    def finish_k(dst_a, dst_b, sem_a, sem_b, reg_sem, out_a, out_b):
      del out_a, out_b
      left = lax.rem(lax.axis_index('x') + 1, jnp.int32(2))
      pl.semaphore_wait(reg_sem, 1)
      for dst, recv_sem in ((dst_a, sem_a), (dst_b, sem_b)):
        pltpu.make_async_remote_copy(
            src_ref=dst, dst_ref=dst, send_sem=recv_sem, recv_sem=recv_sem,
            device_id=(left,),
        ).wait_recv()

    protocol = self._two_escaped_buffers_protocol(
        interpret, finish_k, 'test_barrier_two_bufs'
    )
    src_a = jnp.arange(8 * 256, dtype=jnp.float32).reshape((8, 256))
    src_b = src_a + 10000.0
    res_a, res_b = protocol(src_a, src_b)
    # Each device receives its peer's shard of each source.
    np.testing.assert_array_equal(res_a[:, :128], src_a[:, 128:])
    np.testing.assert_array_equal(res_a[:, 128:], src_a[:, :128])
    np.testing.assert_array_equal(res_b[:, :128], src_b[:, 128:])
    np.testing.assert_array_equal(res_b[:, 128:], src_b[:, :128])

  @parameterized.parameters('eager', 'on_wait')
  def test_cross_kernel_dma_second_escaped_buffer_read_before_wait_race(
      self, dma_mode
  ):
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def bad_finish_k(dst_a, dst_b, sem_a, sem_b, reg_sem, out_a, out_b):
      del dst_a, sem_a, sem_b, reg_sem, out_a, out_b
      # Reads dst_b without waiting on its receive semaphore.
      dst_b[...] = dst_b[...] + 1.0

    protocol = self._two_escaped_buffers_protocol(
        interpret, bad_finish_k, 'test_barrier_two_bufs_race'
    )
    src = jnp.arange(8 * 256, dtype=jnp.float32).reshape((8, 256))
    with self.assertRaisesRegex(Exception, 'RACE DETECTED'):
      jax.block_until_ready(protocol(src, src + 1.0))

  @parameterized.parameters('eager', 'on_wait')
  def test_cross_kernel_dma_into_escaped_buffers_after_buffer_ids_diverge(
      self, dma_mode
  ):
    # The escaped DMA destinations must be recorded under the receiver's
    # buffer ids, which differ from the sender's ids here.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def finish_k(dst_a, dst_b, sem_a, sem_b, reg_sem, out_a, out_b):
      del out_a, out_b
      left = lax.rem(lax.axis_index('x') + 1, jnp.int32(2))
      pl.semaphore_wait(reg_sem, 1)
      for dst, recv_sem in ((dst_a, sem_a), (dst_b, sem_b)):
        pltpu.make_async_remote_copy(
            src_ref=dst, dst_ref=dst, send_sem=recv_sem, recv_sem=recv_sem,
            device_id=(left,),
        ).wait_recv()

    protocol = self._two_escaped_buffers_protocol(
        interpret, finish_k, 'test_barrier_two_bufs_ids_diverge',
        diverge_buffer_ids=True,
    )
    src_a = jnp.arange(8 * 256, dtype=jnp.float32).reshape((8, 256))
    src_b = src_a + 10000.0
    res_a, res_b = protocol(src_a, src_b)
    np.testing.assert_array_equal(res_a[:, :128], src_a[:, 128:])
    np.testing.assert_array_equal(res_a[:, 128:], src_a[:, :128])
    np.testing.assert_array_equal(res_b[:, :128], src_b[:, 128:])
    np.testing.assert_array_equal(res_b[:, 128:], src_b[:, :128])

  def test_boundary_copy_read_vs_cross_kernel_dma_write(self):
    key = ('vmem', 3, 0, 0)
    rnge = (slice(0, 8),)

    def unordered_clocks():
      read_clock = vector_clock.NpVectorClock(4)
      read_clock.inc(0)
      write_clock = vector_clock.NpVectorClock(4)
      write_clock.inc(1)
      return read_clock, write_clock

    # The interpreter's copy-out of a kernel output does not race with a DMA
    # whose receive semaphore escaped the pallas_call.
    races = race_detection_state.RaceDetectionState()
    read_clock, write_clock = unordered_clocks()
    races.check_read((0, 0), read_clock, key, rnge, boundary_copy=True)
    races.check_write((9, 0), write_clock, key, rnge, cross_kernel_dma=True)
    self.assertFalse(races.races_found)

    # A kernel read still races with such a DMA.
    races = race_detection_state.RaceDetectionState()
    read_clock, write_clock = unordered_clocks()
    races.check_read((0, 0), read_clock, key, rnge)
    with self.assertRaisesRegex(RuntimeError, 'RACE DETECTED'):
      races.check_write((9, 0), write_clock, key, rnge, cross_kernel_dma=True)

    # The copy-out still races with any other write.
    races = race_detection_state.RaceDetectionState()
    read_clock, write_clock = unordered_clocks()
    races.check_read((0, 0), read_clock, key, rnge, boundary_copy=True)
    with self.assertRaisesRegex(RuntimeError, 'RACE DETECTED'):
      races.check_write((9, 0), write_clock, key, rnge)

    # Two such DMAs into the same buffer still race with each other.
    races = race_detection_state.RaceDetectionState()
    clock_1, clock_2 = unordered_clocks()
    races.check_write((8, 0), clock_1, key, rnge, cross_kernel_dma=True)
    with self.assertRaisesRegex(RuntimeError, 'RACE DETECTED'):
      races.check_write((9, 0), clock_2, key, rnge, cross_kernel_dma=True)

  def _run_asymmetric_buffer_id_divergence_test(self, dma_mode):
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest("requires at least 2 devices")
    mesh = jtu.create_mesh((num_devices,), ("x",))

    def kernel(input_ref, output_ref, scratch_ref, send_sem, recv_sem):
      my_id = lax.axis_index("x")
      right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
      left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))

      # Device 0 allocates an extra scoped VMEM buffer under pl.when with side-effect write
      @pl.when(my_id == 0)
      def _extra_dev0():
        @functools.partial(pl.run_scoped, b1=pltpu.VMEM((8, 128), jnp.int32))
        def _(b1):
          b1[...] = jnp.zeros((8, 128), dtype=jnp.int32)
          scratch_ref[...] = b1[...].astype(jnp.float32)

      # Destination buffer allocated inside the kernel body on all devices
      @functools.partial(pl.run_scoped, target_buf=pltpu.VMEM((8, 128), jnp.float32))
      def _scoped_region(target_buf):
        target_buf[...] = jnp.zeros((8, 128), dtype=jnp.float32)

        barrier_sem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(barrier_sem, 1, device_id=(left_neighbor,))
        pl.semaphore_signal(barrier_sem, 1, device_id=(right_neighbor,))
        pl.semaphore_wait(barrier_sem, 2)

        # Remote copy into right neighbor's target_buf
        remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=input_ref,
            dst_ref=target_buf,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(right_neighbor,),
        )
        remote_copy_op.start()
        remote_copy_op.wait_send()
        remote_copy_op.wait_recv()

        pltpu.async_copy(target_buf, output_ref, send_sem).wait()

    pallas_fn = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=[
            pltpu.VMEM((8, 128), jnp.float32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        interpret=pltpu.InterpretParams(
            dma_execution_mode=dma_mode,
            detect_races=True,
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )

    src = jnp.full((8 * num_devices, 128), 42.0, dtype=jnp.float32)
    sharding = jax.sharding.NamedSharding(mesh, P("x", None))
    src = jax.device_put(src, sharding)

    sharded_fn = jax.jit(
        shard_map.shard_map(
            pallas_fn,
            mesh=mesh,
            in_specs=P("x", None),
            out_specs=P("x", None),
            check_vma=False,
        )
    )
    res = sharded_fn(src)
    np.testing.assert_allclose(res, 42.0)

  def test_remote_dma_asymmetric_allocation_buffer_id_alignment_eager(self):
    self._run_asymmetric_buffer_id_divergence_test("eager")

  def test_remote_dma_asymmetric_allocation_buffer_id_alignment_on_wait(self):
    self._run_asymmetric_buffer_id_divergence_test("on_wait")

  @parameterized.parameters("eager", "on_wait")
  def test_remote_dma_into_hbm_output_after_buffer_ids_diverge(self, dma_mode):
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def alloc_k(sem_out):
      # One more buffer on device 0, so that buffer ids differ across devices
      # in later pallas_calls. The escaped semaphore keeps the shared memory,
      # and so the buffer id counters, alive across pallas_calls.
      @pl.when(lax.axis_index("x") == 0)
      def _():
        @functools.partial(pl.run_scoped, b=pltpu.VMEM((8, 128), jnp.float32))
        def _(b):
          b[...] = jnp.zeros((8, 128), dtype=jnp.float32)
          # An effect on a kernel operand, so that the allocation is not
          # removed as dead code.
          pl.semaphore_signal(sem_out, 1)
          pl.semaphore_wait(sem_out, 1)

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=pltpu.SemaphoreType.REGULAR(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=interpret,
    )

    num_devices = jax.device_count()

    def copy_k(x_hbm, sem_in, o_hbm, send_sem, recv_sem):
      del sem_in
      my_id = lax.axis_index("x")
      right = lax.rem(my_id + 1, jnp.int32(num_devices))
      left = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
      bar = pltpu.get_barrier_semaphore()
      pl.semaphore_signal(bar, 1, device_id=(left,))
      pl.semaphore_signal(bar, 1, device_id=(right,))
      pl.semaphore_wait(bar, 2)
      copy = pltpu.make_async_remote_copy(
          src_ref=x_hbm, dst_ref=o_hbm, send_sem=send_sem, recv_sem=recv_sem,
          device_id=(right,),
      )
      copy.start()
      copy.wait_send()
      copy.wait_recv()

    copy_call = pl.pallas_call(
        copy_k,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )

    mesh = jtu.create_mesh((num_devices,), ("x",))
    sharded = jax.jit(
        shard_map.shard_map(
            lambda x: copy_call(x, alloc_call()),
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    src = jnp.arange(8 * 128 * num_devices, dtype=jnp.float32).reshape(
        (8, 128 * num_devices)
    )
    res = sharded(src)
    # Each device receives the shard of its left neighbor.
    np.testing.assert_array_equal(res, np.roll(src, 128, axis=1))

  @parameterized.parameters("eager", "on_wait")
  def test_remote_dma_into_buffer_missing_on_peer_raises(self, dma_mode):
    num_devices = jax.device_count()

    def kernel(x_ref, o_ref, send_sem, recv_sem):
      o_ref[...] = x_ref[...]
      peer = lax.rem(lax.axis_index("x") + 1, jnp.int32(num_devices))

      # Only device 0 allocates `b`, so the peer has no buffer to receive into.
      @pl.when(lax.axis_index("x") == 0)
      def _():
        @functools.partial(pl.run_scoped, b=pltpu.VMEM((8, 128), jnp.float32))
        def _(b):
          copy = pltpu.make_async_remote_copy(
              src_ref=x_ref, dst_ref=b, send_sem=send_sem, recv_sem=recv_sem,
              device_id=(peer,),
          )
          copy.start()
          copy.wait_send()

    pallas_fn = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(dma_execution_mode=dma_mode),
    )
    mesh = jtu.create_mesh((num_devices,), ("x",))
    sharded = jax.jit(
        shard_map.shard_map(
            pallas_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    src = jnp.zeros((8, 128 * num_devices), dtype=jnp.float32)
    with self.assertRaisesRegex(Exception, "Cannot translate buffer"):
      jax.block_until_ready(sharded(src))

  @parameterized.parameters("eager", "on_wait")
  def test_remote_dma_before_peer_binds_its_buffers(self, dma_mode):
    # A collective kernel without a barrier DMAs into its peer's output buffer
    # right away. Device 1 is slow to bind that buffer, so device 0 must not
    # start the kernel body before every device has bound its buffers.
    def kernel(x_ref, o_ref, send_sem, recv_sem):
      peer = lax.axis_index("x") ^ 1
      copy = pltpu.make_async_remote_copy(
          src_ref=x_ref, dst_ref=o_ref, send_sem=send_sem, recv_sem=recv_sem,
          device_id=(peer,),
      )
      copy.start()
      copy.wait_send()
      copy.wait_recv()

    pallas_fn = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(
            detect_races=True, dma_execution_mode=dma_mode
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0, allow_collective_id_without_custom_barrier=True
        ),
    )
    num_devices = jax.device_count()
    mesh = jtu.create_mesh((num_devices,), ("x",))
    sharded = jax.jit(
        shard_map.shard_map(
            pallas_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )

    register = interpret_lib.memory.SharedMemory.register_logical_buffer

    def slow_register(sm, device_id, local_core_id, buffer_id, logical_key):
      if int(device_id) == 1 and logical_key == ("kernel_output", 0):
        time.sleep(1.0)
      return register(sm, device_id, local_core_id, buffer_id, logical_key)

    src = jnp.arange(8 * 128 * num_devices, dtype=jnp.float32).reshape(
        (8, 128 * num_devices)
    )
    with mock.patch.object(
        interpret_lib.memory.SharedMemory,
        "register_logical_buffer",
        slow_register,
    ):
      res = jax.block_until_ready(sharded(src))
    # Each device receives its peer's shard.
    expected = src.reshape(8, num_devices // 2, 2, 128)[:, :, ::-1, :]
    np.testing.assert_array_equal(res, expected.reshape(src.shape))

  def _remote_dma_waited_in_later_call_protocol(self, interpret):
    # Both semaphores escape `alloc_k`. `start_k` starts a remote DMA into the
    # HBM output of its peer, and does not wait for it. `finish_k` gets that
    # output as an aliased operand, waits for the DMA into it and for the DMA
    # it sent, and then reads it.
    def alloc_k(recv_sem_out, send_sem_out):
      del recv_sem_out, send_sem_out

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=(pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.DMA(())),
        out_specs=(pl.BlockSpec(memory_space=pltpu.SEMAPHORE),) * 2,
        interpret=interpret,
    )

    def start_k(x_hbm, recv_sem, send_sem, o_hbm):
      pltpu.make_async_remote_copy(
          src_ref=x_hbm, dst_ref=o_hbm, send_sem=send_sem, recv_sem=recv_sem,
          device_id=(lax.axis_index("x") ^ 1,),
      ).start()

    start_call = pl.pallas_call(
        start_k,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=interpret,
    )

    def finish_k(o_hbm, recv_sem, send_sem, o_hbm_out, res_ref):
      del o_hbm_out
      pltpu.make_async_copy(src_ref=o_hbm, dst_ref=o_hbm, sem=recv_sem).wait()
      pltpu.make_async_copy(src_ref=o_hbm, dst_ref=o_hbm, sem=send_sem).wait()
      pltpu.sync_copy(o_hbm, res_ref)

    finish_call = pl.pallas_call(
        finish_k,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_shape=[jax.ShapeDtypeStruct((8, 128), jnp.float32)] * 2,
        input_output_aliases={0: 0},
        interpret=interpret,
    )

    def protocol_fn(x):
      recv_sem, send_sem = alloc_call()
      o = start_call(x, recv_sem, send_sem)
      return finish_call(o, recv_sem, send_sem)[1]

    return protocol_fn

  @parameterized.product(
      dma_mode=("eager", "on_wait"), send_wait=("start_k", "finish_k")
  )
  def test_remote_dma_lands_before_peer_reads_its_outputs(
      self, dma_mode, send_wait
  ):
    # Each pair of devices DMA into each other's HBM output, and wait for the
    # DMA from the peer only in a later pallas_call, as the receive semaphore
    # escapes `alloc_k`. The DMA from device 1 is slow, so device 0 must not
    # read its outputs before device 1 has finished running `start_k`.
    #
    # With send_wait="finish_k", both DMA waits are in the later pallas_call,
    # so in "on_wait" mode the DMAs run there, and must write to its operand.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def alloc_k(recv_sem_out):
      del recv_sem_out

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=pltpu.SemaphoreType.DMA(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=interpret,
    )

    def start_k(x_hbm, recv_sem, o_hbm, send_sem):
      copy = pltpu.make_async_remote_copy(
          src_ref=x_hbm, dst_ref=o_hbm, send_sem=send_sem, recv_sem=recv_sem,
          device_id=(lax.axis_index("x") ^ 1,),
      )
      copy.start()
      copy.wait_send()

    start_call = pl.pallas_call(
        start_k,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        interpret=interpret,
    )

    def finish_k(x_hbm, recv_sem, o_ref):
      pltpu.make_async_copy(src_ref=x_hbm, dst_ref=x_hbm, sem=recv_sem).wait()
      pltpu.sync_copy(x_hbm, o_ref)

    finish_call = pl.pallas_call(
        finish_k,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=interpret,
    )

    def protocol_fn(x):
      recv_sem = alloc_call()
      return finish_call(start_call(x, recv_sem), recv_sem)

    if send_wait == "finish_k":
      protocol_fn = self._remote_dma_waited_in_later_call_protocol(interpret)

    num_devices = jax.device_count()
    mesh = jtu.create_mesh((num_devices,), ("x",))
    sharded = jax.jit(
        shard_map.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )

    execute = interpret_lib.DMA.execute_read_and_write

    def slow_execute(dma):
      if dma.src_device_id == 1 and dma.dst_device_id == 0:
        time.sleep(1.0)
      return execute(dma)

    src = jnp.arange(8 * 128 * num_devices, dtype=jnp.float32).reshape(
        (8, 128 * num_devices)
    )
    with mock.patch.object(
        interpret_lib.DMA, "execute_read_and_write", slow_execute
    ):
      res = jax.block_until_ready(sharded(src))
    # Each device receives its peer's shard.
    expected = src.reshape(8, num_devices // 2, 2, 128)[:, :, ::-1, :]
    np.testing.assert_array_equal(res, expected.reshape(src.shape))

  def _recording_hbm_binds(self):
    # Records, for each call, the buffer that each bindable operand of a
    # pallas_call is bound to, or -1.
    bind = interpret_lib._bind_hbm_inputs_to_dma_dsts
    binds = []

    def recording_bind(*args, **kwargs):
      token, bound = bind(*args, **kwargs)
      binds.append(np.asarray(bound).tolist())
      return token, bound

    return (
        mock.patch.object(
            interpret_lib, "_bind_hbm_inputs_to_dma_dsts", recording_bind
        ),
        binds,
    )

  @parameterized.product(dma_mode=("eager", "on_wait"), swap=(False, True))
  def test_remote_dmas_into_two_outputs_waited_in_later_call(
      self, dma_mode, swap
  ):
    # Each pair of devices DMA two arrays into each other's two HBM outputs,
    # and wait for the DMAs only in a later pallas_call, which gets the outputs
    # as aliased operands, in order or swapped, and reads its first operand.
    #
    # In "on_wait" mode the DMAs run in the later pallas_call, so both outputs
    # still hold the same uninitialized values when it starts, and which
    # operand is which output is not known. The operands must then not be bound
    # to the DMA dst buffers, so they do not get the DMA'd values, rather than
    # be bound by guessing, which may give an operand the other one's values.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def alloc_k(recv_sem_out, send_sem_out):
      del recv_sem_out, send_sem_out

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=(pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.DMA(())),
        out_specs=(pl.BlockSpec(memory_space=pltpu.SEMAPHORE),) * 2,
        interpret=interpret,
    )

    def start_k(x_hbm, y_hbm, recv_sem, send_sem, o_x_hbm, o_y_hbm):
      for src_ref, dst_ref in ((x_hbm, o_x_hbm), (y_hbm, o_y_hbm)):
        pltpu.make_async_remote_copy(
            src_ref=src_ref, dst_ref=dst_ref, send_sem=send_sem,
            recv_sem=recv_sem, device_id=(lax.axis_index("x") ^ 1,),
        ).start()

    start_call = pl.pallas_call(
        start_k,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 2
        + [pl.BlockSpec(memory_space=pltpu.SEMAPHORE)] * 2,
        out_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 2,
        out_shape=[jax.ShapeDtypeStruct((8, 128), jnp.float32)] * 2,
        interpret=interpret,
    )

    def finish_k(a_hbm, b_hbm, recv_sem, send_sem, a_out, b_out, res_ref):
      del b_hbm, a_out, b_out
      for sem in (recv_sem, recv_sem, send_sem, send_sem):
        pltpu.make_async_copy(src_ref=a_hbm, dst_ref=a_hbm, sem=sem).wait()
      pltpu.sync_copy(a_hbm, res_ref)

    finish_call = pl.pallas_call(
        finish_k,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 2
        + [pl.BlockSpec(memory_space=pltpu.SEMAPHORE)] * 2,
        out_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 2
        + [pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_shape=[jax.ShapeDtypeStruct((8, 128), jnp.float32)] * 3,
        input_output_aliases={0: 0, 1: 1},
        interpret=interpret,
    )

    def protocol_fn(x):
      recv_sem, send_sem = alloc_call()
      o_x, o_y = start_call(x, x + 0.5, recv_sem, send_sem)
      a, b = (o_y, o_x) if swap else (o_x, o_y)
      return finish_call(a, b, recv_sem, send_sem)[2]

    num_devices = jax.device_count()
    mesh = jtu.create_mesh((num_devices,), ("x",))
    sharded = jax.jit(
        shard_map.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    src = jnp.arange(8 * 128 * num_devices, dtype=jnp.float32).reshape(
        (8, 128 * num_devices)
    )
    recording, binds = self._recording_hbm_binds()
    with recording:
      res = jax.block_until_ready(sharded(src))
    self.assertLen(binds, num_devices)
    if dma_mode == "eager":
      # The DMAs run in `start_k`, so the outputs hold the DMA'd values, which
      # tell the operands apart, and the first operand is read as on a TPU.
      for bound in binds:
        self.assertNotIn(-1, bound)
        self.assertNotEqual(bound[0], bound[1])
      peer_src = src.reshape(8, num_devices // 2, 2, 128)[:, :, ::-1, :]
      np.testing.assert_array_equal(
          res, peer_src.reshape(src.shape) + (0.5 if swap else 0.0)
      )
    else:
      np.testing.assert_array_equal(res, np.full(res.shape, np.nan, res.dtype))
      self.assertEqual(binds, [[-1, -1]] * num_devices)

  @parameterized.product(
      dma_mode=("eager", "on_wait"),
      operands=("dst_and_equal_array", "changed_dst"),
  )
  def test_remote_dma_dst_bound_only_to_the_one_operand_it_can_be(
      self, dma_mode, operands
  ):
    # Each pair of devices DMA into each other's HBM output, and wait for the
    # DMA from the peer only in a later pallas_call, which gets two aliased
    # operands and reads the second one. The DMA runs in `start_k`, where the
    # sender waits for it, so the output holds the DMA'd values.
    #
    # With "dst_and_equal_array", the operands are an array with the same
    # values (all devices have the same input) and the output, so either
    # operand can be the output. With "changed_dst", the operands are zeros and
    # the output plus one, so neither operand has the values that `start_k`
    # returned, and neither can be the output. In both cases no operand is
    # bound to the DMA dst buffer.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def alloc_k(recv_sem_out):
      del recv_sem_out

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=pltpu.SemaphoreType.DMA(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        interpret=interpret,
    )

    def start_k(x_hbm, recv_sem, o_hbm, send_sem):
      copy = pltpu.make_async_remote_copy(
          src_ref=x_hbm, dst_ref=o_hbm, send_sem=send_sem, recv_sem=recv_sem,
          device_id=(lax.axis_index("x") ^ 1,),
      )
      copy.start()
      copy.wait_send()

    start_call = pl.pallas_call(
        start_k,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        interpret=interpret,
    )

    def finish_k(a_hbm, b_hbm, recv_sem, a_out, b_out, res_ref):
      del a_hbm, a_out, b_out
      pltpu.make_async_copy(src_ref=b_hbm, dst_ref=b_hbm, sem=recv_sem).wait()
      pltpu.sync_copy(b_hbm, res_ref)

    finish_call = pl.pallas_call(
        finish_k,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 2
        + [pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
        out_specs=[pl.BlockSpec(memory_space=pl.ANY)] * 2
        + [pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_shape=[jax.ShapeDtypeStruct((8, 128), jnp.float32)] * 3,
        input_output_aliases={0: 0, 1: 1},
        interpret=interpret,
    )

    def protocol_fn(x):
      recv_sem = alloc_call()
      o = start_call(x, recv_sem)
      if operands == "dst_and_equal_array":
        a, b = x, o
      else:
        a, b = jnp.zeros_like(o), o + 1.0
      return finish_call(a, b, recv_sem)[2]

    num_devices = jax.device_count()
    mesh = jtu.create_mesh((num_devices,), ("x",))
    sharded = jax.jit(
        shard_map.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    src = jnp.ones((8, 128 * num_devices), dtype=jnp.float32)
    recording, binds = self._recording_hbm_binds()
    with recording:
      res = jax.block_until_ready(sharded(src))
    self.assertEqual(binds, [[-1, -1]] * num_devices)
    np.testing.assert_array_equal(
        res, src if operands == "dst_and_equal_array" else src + 1.0
    )

  def test_tagged_barrier_semaphore_id_differs_from_internal_semaphore_ids(
      self
  ):
    # A tag whose hash, truncated to 15 bits, is the id of the first
    # internally allocated semaphore. Its barrier semaphore must still get an
    # id of its own.
    tag = next(
        ("barrier_tag", i) for i in range(1 << 20)
        if (hash(("barrier_tag", i)) & 0x7FFF) == 2000
    )
    num_devices = jax.device_count()

    def kernel(x_ref, o_ref, sem):
      my_id = lax.axis_index("x")
      right = lax.rem(my_id + 1, jnp.int32(num_devices))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, 1, device_id=(right,))
      pl.semaphore_wait(bar, 1)
      copy = pltpu.make_async_copy(x_ref, o_ref, sem)
      copy.start()
      copy.wait()

    pallas_fn = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    mesh = jtu.create_mesh((num_devices,), ("x",))
    sharded = jax.jit(
        shard_map.shard_map(
            pallas_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    src = jnp.arange(8 * 128 * num_devices, dtype=jnp.float32).reshape(
        (8, 128 * num_devices)
    )
    np.testing.assert_array_equal(sharded(src), src)

  def test_remote_dma_asymmetric_allocation_read_before_wait_race(self):
    """Verifies RACE DETECTED when receiver reads before wait under asymmetric allocations."""
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest("requires at least 2 devices")
    mesh = jtu.create_mesh((num_devices,), ("x",))

    def body_fn(src_ref, dst_ref, dst_out_ref, send_sem, recv_sem):
      del dst_out_ref
      my_id = lax.axis_index("x")
      peer_id = lax.rem(my_id + 1, jnp.int32(num_devices))

      @pl.when(my_id == 0)
      def _extra_alloc():
        @functools.partial(
            pl.run_scoped, extra_buf=pltpu.VMEM((8, 128), jnp.int32)
        )
        def _dummy(extra_buf):
          extra_buf[...] = jnp.zeros((8, 128), dtype=jnp.int32)

      bar = pltpu.get_barrier_semaphore()
      pl.semaphore_signal(bar, device_id=(peer_id,))
      pl.semaphore_wait(bar, 1)

      remote_copy_op = pltpu.make_async_remote_copy(
          src_ref=src_ref,
          dst_ref=dst_ref,
          send_sem=send_sem,
          recv_sem=recv_sem,
          device_id=(peer_id,),
      )

      @pl.when(my_id == 0)
      def _send():
        remote_copy_op.start()
        remote_copy_op.wait_send()

      @pl.when(my_id == 1)
      def _recv():
        dst_ref[...] = dst_ref[...] + 1.0
        remote_copy_op.wait_recv()

    pallas_fn = pl.pallas_call(
        body_fn,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        input_output_aliases={1: 0},
        scratch_shapes=[pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(
            dma_execution_mode="eager",
            detect_races=True,
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )

    src = jnp.full((8, 128 * num_devices), 42.0, dtype=jnp.float32)
    dst = jnp.zeros((8, 128 * num_devices), dtype=jnp.float32)
    sharded_fn = jax.jit(
        shard_map.shard_map(
            pallas_fn,
            mesh=mesh,
            in_specs=(P(None, "x"), P(None, "x")),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    with self.assertRaisesRegex(Exception, "RACE DETECTED"):
      jax.block_until_ready(sharded_fn(src, dst))

  def test_semaphore_wait_uninitialized_clock_no_assertion_error(self):
    """Verifies that waiting on semaphore with count >= value when clock is None doesn't raise AssertionError."""
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest("requires at least 2 devices")
    mesh = jtu.create_mesh((num_devices,), ("x",))

    def body_fn(x_ref, out_ref, sem_wait, sem_sig):
      out_ref[...] = x_ref[...]
      my_id = lax.axis_index("x")
      peer = lax.rem(my_id + 1, jnp.int32(num_devices))
      pl.semaphore_signal(sem_sig, 1, device_id=(peer,))
      pl.semaphore_wait(sem_wait, 0)
      pl.semaphore_wait(sem_sig, 1)

    pallas_fn = pl.pallas_call(
        body_fn,
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.SemaphoreType.REGULAR, pltpu.SemaphoreType.REGULAR],
        interpret=pltpu.InterpretParams(
            detect_races=True,
        ),
    )
    src = jnp.zeros((8, 128 * num_devices), dtype=jnp.float32)
    sharded_fn = jax.jit(
        shard_map.shard_map(
            pallas_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    res = sharded_fn(src)
    res.block_until_ready()

  def _recording_kernel_arg_binds(self):
    # Records, for each call, the indices of the bindable inputs whose kernel
    # args are not in HBM, and the buffer that each one is bound to, or -1.
    bind = interpret_lib._bind_kernel_args_to_dma_dsts
    binds = []

    def recording_bind(*args, **kwargs):
      token, bound = bind(*args, **kwargs)
      binds.append((kwargs["arg_idxs"], np.asarray(bound).tolist()))
      return token, bound

    return (
        mock.patch.object(
            interpret_lib, "_bind_kernel_args_to_dma_dsts", recording_bind
        ),
        binds,
    )

  def _two_chunk_vmem_protocol(self, interpret, layout, tag):
    """2 chunks, each DMA'd into the peer's VMEM output and waited for later.

    `alloc_k` returns each chunk's DMA receive and send semaphores, which
    escape. `start_k` copies each chunk's source to its `send` output and starts
    a remote copy from it into the peer's `recv` output of that chunk, and
    exits without waiting. `finish_k` waits for each chunk's semaphores and
    then reads that chunk's `recv`. With layout "send_recv", each chunk's
    inputs to `finish_k` are (send, recv, recv_sem, send_sem), so an unused
    input of the same shape comes before `recv`; with "recv_only" they are
    (recv, recv_sem, send_sem).
    """
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    out = jax.ShapeDtypeStruct((8, 128), jnp.int32)

    def alloc_k(*sems_out):
      del sems_out
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, device_id=(1 - lax.axis_index("x"),))

    alloc_call = pl.pallas_call(
        alloc_k,
        out_shape=[pltpu.SemaphoreType.DMA(())] * 4,
        out_specs=[sem] * 4,
        interpret=interpret,
    )

    def start_k(src_a, src_b, rsem_a, ssem_a, rsem_b, ssem_b, send_a, recv_a,
                send_b, recv_b):
      peer = lax.rem(lax.axis_index("x") + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)
      send_a[...] = src_a[...]
      send_b[...] = src_b[...]
      for send, recv, rsem, ssem in (
          (send_a, recv_a, rsem_a, ssem_a),
          (send_b, recv_b, rsem_b, ssem_b),
      ):
        pltpu.make_async_remote_copy(
            src_ref=send, dst_ref=recv, send_sem=ssem, recv_sem=rsem,
            device_id=(peer,),
        ).start()

    start_call = pl.pallas_call(
        start_k,
        in_specs=[vmem, vmem] + [sem] * 4,
        out_specs=[vmem] * 4,
        out_shape=[out] * 4,
        interpret=interpret,
    )

    n_leaves = 4 if layout == "send_recv" else 3

    def finish_k(*refs):
      chunks = (refs[0:n_leaves], refs[n_leaves : 2 * n_leaves])
      for (recv, rsem, ssem), o in zip(
          (c[-3:] for c in chunks), refs[2 * n_leaves :]
      ):
        pltpu.make_async_copy(src_ref=recv, dst_ref=recv, sem=rsem).wait()
        pltpu.make_async_copy(src_ref=recv, dst_ref=recv, sem=ssem).wait()
        o[...] = recv[...]
      peer = lax.rem(lax.axis_index("x") + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)

    chunk_specs = [vmem, vmem, sem, sem] if n_leaves == 4 else [vmem, sem, sem]
    finish_call = pl.pallas_call(
        finish_k,
        in_specs=chunk_specs * 2,
        out_specs=[vmem, vmem],
        out_shape=[out, out],
        interpret=interpret,
    )

    def protocol_fn(src_a, src_b):
      rsem_a, ssem_a, rsem_b, ssem_b = alloc_call()
      send_a, recv_a, send_b, recv_b = start_call(
          src_a, src_b, rsem_a, ssem_a, rsem_b, ssem_b
      )
      if n_leaves == 4:
        args = (send_a, recv_a, rsem_a, ssem_a, send_b, recv_b, rsem_b, ssem_b)
      else:
        args = (recv_a, rsem_a, ssem_a, recv_b, rsem_b, ssem_b)
      return finish_call(*args)

    mesh = jax.sharding.Mesh(jax.devices()[:2], ("x",))
    spec = P(None, "x")
    return jax.jit(
        shard_map.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )

  @parameterized.product(
      layout=("send_recv", "recv_only"), dma_mode=("eager", "on_wait")
  )
  def test_escaped_vmem_dst_bound_to_its_own_input(self, layout, dma_mode):
    # Each kernel arg for a `recv` input must be the buffer that the DMA of its
    # own chunk writes to, also when an unused input of the same shape comes
    # first. Binding each kernel arg to the first unclaimed escaped DMA dst
    # buffer of the same shape bound `send_a` to chunk a's buffer and `recv_a`
    # to chunk b's, so `finish_k` read chunk b's buffer after waiting only for
    # chunk a's DMA (a false race), and read the wrong values.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )
    protocol = self._two_chunk_vmem_protocol(
        interpret, layout, f"test_barrier_two_chunks_{layout}_{dma_mode}"
    )
    src_a = jnp.arange(8 * 256, dtype=jnp.int32).reshape((8, 256))
    src_b = src_a + 10000
    res_a, res_b = jax.block_until_ready(protocol(src_a, src_b))
    # Each device receives its peer's shard of each source.
    for res, src in ((res_a, src_a), (res_b, src_b)):
      np.testing.assert_array_equal(res[:, :128], src[:, 128:])
      np.testing.assert_array_equal(res[:, 128:], src[:, :128])

  @parameterized.product(
      layout=("send_recv", "recv_only"), dma_mode=("eager", "on_wait")
  )
  def test_escaped_vmem_dst_binds_only_recv_args(self, layout, dma_mode):
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )
    protocol = self._two_chunk_vmem_protocol(
        interpret, layout, f"test_barrier_two_chunks_binds_{layout}_{dma_mode}"
    )
    src = jnp.arange(8 * 256, dtype=jnp.int32).reshape((8, 256))
    recording, binds = self._recording_kernel_arg_binds()
    with recording:
      jax.block_until_ready(protocol(src, src + 10000))
    # The call to `finish_k` on each device binds both `recv` args, each to a
    # different buffer, and no other arg.
    finish_idxs = (0, 1, 4, 5) if layout == "send_recv" else (0, 3)
    recv_args = [1, 3] if layout == "send_recv" else [0, 1]
    finish_binds = [bound for idxs, bound in binds if idxs == finish_idxs]
    self.assertLen(finish_binds, 2)
    for bound in finish_binds:
      recv_bound = [bound[k] for k in recv_args]
      self.assertNotIn(-1, recv_bound)
      self.assertNotEqual(recv_bound[0], recv_bound[1])
      self.assertEqual(
          [b for k, b in enumerate(bound) if k not in recv_args],
          [-1] * (len(bound) - 2),
      )

  def test_escaped_vmem_dst_read_before_wait_race(self):
    # The kernel arg for chunk b's `recv` is bound to the buffer that chunk b's
    # DMA writes to, so reading it after waiting only for chunk a's DMA is a
    # race.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode="eager"
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    out = jax.ShapeDtypeStruct((8, 128), jnp.int32)
    tag = "test_barrier_two_chunks_race"

    def alloc_k(*sems_out):
      del sems_out
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, device_id=(1 - lax.axis_index("x"),))

    def start_k(src_a, src_b, rsem_a, rsem_b, recv_a, recv_b, ssem_a, ssem_b):
      peer = lax.rem(lax.axis_index("x") + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)
      for src, recv, rsem, ssem in (
          (src_a, recv_a, rsem_a, ssem_a),
          (src_b, recv_b, rsem_b, ssem_b),
      ):
        copy = pltpu.make_async_remote_copy(
            src_ref=src, dst_ref=recv, send_sem=ssem, recv_sem=rsem,
            device_id=(peer,),
        )
        copy.start()
        copy.wait_send()

    def bad_finish_k(recv_a, recv_b, rsem_a, rsem_b, out_a, out_b):
      pltpu.make_async_copy(src_ref=recv_a, dst_ref=recv_a, sem=rsem_a).wait()
      out_a[...] = recv_a[...]
      out_b[...] = recv_b[...]  # Does not wait for chunk b's DMA first.
      pltpu.make_async_copy(src_ref=recv_b, dst_ref=recv_b, sem=rsem_b).wait()
      peer = lax.rem(lax.axis_index("x") + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)

    def protocol_fn(src_a, src_b):
      rsem_a, rsem_b = pl.pallas_call(
          alloc_k,
          out_shape=[pltpu.SemaphoreType.DMA(())] * 2,
          out_specs=[sem] * 2,
          interpret=interpret,
      )()
      recv_a, recv_b = pl.pallas_call(
          start_k,
          in_specs=[vmem, vmem, sem, sem],
          out_specs=[vmem, vmem],
          out_shape=[out, out],
          scratch_shapes=[pltpu.SemaphoreType.DMA] * 2,
          interpret=interpret,
      )(src_a, src_b, rsem_a, rsem_b)
      return pl.pallas_call(
          bad_finish_k,
          in_specs=[vmem, vmem, sem, sem],
          out_specs=[vmem, vmem],
          out_shape=[out, out],
          interpret=interpret,
      )(recv_a, recv_b, rsem_a, rsem_b)

    mesh = jax.sharding.Mesh(jax.devices()[:2], ("x",))
    spec = P(None, "x")
    protocol = jax.jit(
        shard_map.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )
    src = jnp.arange(8 * 256, dtype=jnp.int32).reshape((8, 256))
    with self.assertRaisesRegex(Exception, "RACE DETECTED"):
      jax.block_until_ready(protocol(src, src + 10000))

  @parameterized.parameters("eager", "on_wait")
  def test_escaped_vmem_dst_copied_input_not_bound(self, dma_mode):
    # `finish_k` gets chunk a's `recv` output and a copy of it (computed
    # outside of the pallas_calls), which have the same value, so either input
    # can be chunk a's buffer, and neither kernel arg is bound to it.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    sem = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    out = jax.ShapeDtypeStruct((8, 128), jnp.int32)
    tag = f"test_barrier_copied_recv_{dma_mode}"

    def alloc_k(rsem_out):
      del rsem_out
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, device_id=(1 - lax.axis_index("x"),))

    def start_k(src, rsem, recv, ssem):
      peer = lax.rem(lax.axis_index("x") + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_wait(bar, 1)
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)
      copy = pltpu.make_async_remote_copy(
          src_ref=src, dst_ref=recv, send_sem=ssem, recv_sem=rsem,
          device_id=(peer,),
      )
      copy.start()
      copy.wait_send()

    def finish_k(recv, recv_copy, rsem, o):
      del recv_copy
      pltpu.make_async_copy(src_ref=recv, dst_ref=recv, sem=rsem).wait()
      o[...] = jnp.zeros_like(o)
      peer = lax.rem(lax.axis_index("x") + 1, jnp.int32(2))
      bar = pltpu.get_barrier_semaphore(tag)
      pl.semaphore_signal(bar, 1, device_id=(peer,))
      pl.semaphore_wait(bar, 1)

    def protocol_fn(src):
      rsem = pl.pallas_call(
          alloc_k,
          out_shape=pltpu.SemaphoreType.DMA(()),
          out_specs=sem,
          interpret=interpret,
      )()
      recv = pl.pallas_call(
          start_k,
          in_specs=[vmem, sem],
          out_specs=vmem,
          out_shape=out,
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          interpret=interpret,
      )(src, rsem)
      return pl.pallas_call(
          finish_k,
          in_specs=[vmem, vmem, sem],
          out_specs=vmem,
          out_shape=out,
          interpret=interpret,
      )(recv, recv + 0, rsem)

    mesh = jax.sharding.Mesh(jax.devices()[:2], ("x",))
    protocol = jax.jit(
        shard_map.shard_map(
            protocol_fn,
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )
    src = jnp.arange(8 * 256, dtype=jnp.int32).reshape((8, 256))
    recording, binds = self._recording_kernel_arg_binds()
    with recording:
      jax.block_until_ready(protocol(src))
    self.assertEqual(
        [bound for idxs, bound in binds if idxs == (0, 1)], [[-1, -1]] * 2
    )

  def test_pending_dst_fill_values_differ(self):
    # The values returned for the elements of two DMA dst buffers that pending
    # DMAs write to never match, nor match the uninitialized values.
    for dtype in (
        jnp.float32, jnp.bfloat16, jnp.float16, jnp.int32, jnp.int16,
        jnp.int8, jnp.uint32, jnp.uint8,
    ):
      fills = [interpret_lib._pending_dst_fill_value(dtype, c) for c in range(8)]
      self.assertNotIn(None, fills, msg=str(dtype))
      bits = {f.tobytes() for f in fills}
      self.assertLen(bits, 8, msg=str(dtype))
      for mode in ("nan", "zero"):
        uninit = np.array(
            interpret_lib.interpret_utils.get_uninitialized_value(dtype, mode),
            dtype=dtype,
        )
        self.assertNotIn(uninit.tobytes(), bits, msg=f"{dtype} {mode}")
      if jnp.issubdtype(dtype, jnp.floating):
        self.assertTrue(np.isnan(np.asarray(fills, np.float32)).all())
    self.assertIsNone(interpret_lib._pending_dst_fill_value(jnp.bool_, 0))

    # Buffers of the same shape and dtype on a device get different codes, and
    # a code is reused only after it is released.
    shared_memory = mock.Mock(
        pending_dst_fill_code_of={},
        pending_dst_fill_codes=collections.defaultdict(set),
        escaped_dma_dst_snapshots={},
        escaped_dma_dst_mirrors={},
    )
    value = np.zeros((8, 128), np.float32)
    keys = [("vmem", b, 0, 0) for b in range(100, 104)]
    codes = [
        interpret_lib._pending_dst_fill_code(shared_memory, k, value)
        for k in keys
    ]
    self.assertEqual(codes, [0, 1, 2, 3])
    self.assertEqual(
        interpret_lib._pending_dst_fill_code(shared_memory, keys[1], value), 1
    )
    self.assertEqual(
        interpret_lib._pending_dst_fill_code(
            shared_memory, ("vmem", 200, 1, 0), value
        ),
        0,
    )
    interpret_lib._release_escaped_dma_dst(shared_memory, keys[1])
    self.assertEqual(
        interpret_lib._pending_dst_fill_code(
            shared_memory, ("vmem", 105, 0, 0), value
        ),
        1,
    )

  @parameterized.parameters("eager", "on_wait")
  def test_escaped_smem_dst_read_via_aliased_any_input(self, dma_mode):
    # As in `prefetch_tree(..., target_memory_space=SMEM)`: `start_k` starts a
    # DMA into its SMEM output with a semaphore that escapes, and `done_k`
    # gets the output back as an aliased ANY (HBM) input and waits for the DMA.
    # In "on_wait" mode the DMA runs in `done_k`, after `start_k` returned its
    # SMEM output, and must still write to the output of `done_k`.
    interpret = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_mode
    )

    def start_k(arr_ref, smem_ref, sem_ref):
      pltpu.async_copy(arr_ref, smem_ref, sem_ref)

    def done_k(arr_ref, x_alias_ref, sem_ref, out_ref):
      del x_alias_ref
      pltpu.make_async_copy(arr_ref, out_ref, sem_ref).wait()

    def fn(arr):
      smem, sem = pl.pallas_call(
          start_k,
          out_shape=(
              pltpu.SMEM(arr.shape, dtype=arr.dtype),
              pltpu.SemaphoreType.DMA,
          ),
          in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
          out_specs=(
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ),
          interpret=interpret,
      )(arr)
      return pl.pallas_call(
          done_k,
          out_shape=pltpu.SMEM(smem.shape, dtype=smem.dtype),
          in_specs=(
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pl.ANY),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ),
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          input_output_aliases={1: 0},
          interpret=interpret,
      )(arr, smem, sem)

    x = jnp.arange(16, dtype=jnp.int32) + 7
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    out = jax.jit(
        shard_map.shard_map(
            fn, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False,
        )
    )(x)
    np.testing.assert_array_equal(out, x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
