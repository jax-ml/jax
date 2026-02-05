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

import functools
from typing import Any
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


def _maybe_reverse(arg: tuple[Any], reverse: bool) -> tuple[Any]:
  if reverse:
    return tuple(reversed(arg))
  else:
    return arg


# TODO(nrink): Figure out how to safely run different instance of GPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class InterpretTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  def test_interpret_pallas_call(self):
    def _kernel(o_ref):
      o_ref[0] = 42

    @jax.jit
    def kernel():
      return pl.pallas_call(
          _kernel,
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          interpret=mosaic_interpret.InterpretParams(detect_races=True),
      )()

    np.testing.assert_equal(kernel(), np.array([42], dtype=jnp.int32))
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.parameters(range(1, 17))
  def test_interpret_core_map(self, num_threads: int):
    @pl.run_state
    def kernel(o_ref):
      mesh = plgpu.Mesh(num_threads=num_threads, thread_name='x')

      @pl.core_map(
          mesh,
          interpret=mosaic_interpret.InterpretParams(detect_races=True),
      )
      def _():
        thread_idx = jax.lax.axis_index('x')
        o_ref[thread_idx] = thread_idx

    y = kernel(jnp.zeros((num_threads,), jnp.int32))
    np.testing.assert_equal(y, np.arange(num_threads, dtype=jnp.int32))
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_interpret_core_map_with_race(self):
    @pl.run_state
    def kernel(o_ref):
      mesh = plgpu.Mesh(num_threads=2, thread_name='x')

      @pl.core_map(
          mesh,
          interpret=mosaic_interpret.InterpretParams(detect_races=True),
      )
      def _():
        thread_idx = jax.lax.axis_index('x')
        o_ref[...] = thread_idx

    kernel(jnp.zeros((), jnp.int32))
    self.assertTrue(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.parameters(range(1, 17))
  def test_interpret_kernel(self, num_threads):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((num_threads,), jnp.int32),
        num_threads=num_threads,
        thread_name='x',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )
    def _kernel(o_ref):
      thread_idx = jax.lax.axis_index('x')
      o_ref[thread_idx] = thread_idx

    np.testing.assert_equal(jax.jit(_kernel)(), np.arange(num_threads))
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_skip_floating_point_ops(self):
    def matmul_kernel(x_ref, y_ref, z_ref):
      # TODO(nrink): Matrix multiplication with `@` is nor supported for real
      # GPU kernels (but the GPU kernel interpreter allows this). Replace this
      # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
      # kernel interpreter.
      z_ref[...] = x_ref[...] @ y_ref[...]

    def matmul(x: jax.Array, y: jax.Array):
      return pl.pallas_call(
          matmul_kernel,
          out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
          interpret=mosaic_interpret.InterpretParams(
              skip_floating_point_ops=True
          ),
      )(x, y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = jax.jit(matmul)(x, y)
    np.testing.assert_array_equal(z, jnp.full_like(z, jnp.inf))

    lowered = jax.jit(matmul).lower(x, y).as_text(dialect='stablehlo')
    self.assertNotIn('dot_general', lowered)

  @jtu.parameterized.parameters(
      (1, 1, 1),
      (2, 1, 2),
      (2, 2, 1),
      (4, 1, 4),
      (4, 2, 2),
      (4, 4, 1),
      (8, 1, 8),
      (8, 2, 4),
      (8, 4, 2),
      (8, 8, 1),
      (16, 1, 16),
      (16, 2, 8),
      (16, 4, 4),
      (16, 8, 2),
      (16, 16, 1),
  )
  def test_matmul_example(self, num_threads, num_row_blocks, num_col_blocks):
    assert num_threads == num_row_blocks * num_col_blocks

    @jax.jit
    def matmul(x: jax.Array, y: jax.Array):
      num_rows_per_block = x.shape[0] // num_row_blocks
      num_cols_per_block = y.shape[1] // num_col_blocks

      @functools.partial(
          plgpu.kernel,
          out_shape=jax.ShapeDtypeStruct(
              (
                  x.shape[0],
                  y.shape[1],
              ),
              x.dtype,
          ),
          num_threads=num_threads,
          thread_name='t',
          interpret=mosaic_interpret.InterpretParams(
              detect_races=True, num_cores_or_threads=num_threads
          ),
      )
      def _matmul_kernel(x_ref, y_ref, o_ref):
        thread_idx = jax.lax.axis_index('t')

        row_block_idx = thread_idx // num_col_blocks
        row_slice = pl.ds(
            row_block_idx * num_rows_per_block, num_rows_per_block
        )

        col_block_idx = jax.lax.rem(thread_idx, jnp.int32(num_col_blocks))
        col_slice = pl.ds(
            col_block_idx * num_cols_per_block, num_cols_per_block
        )

        # TODO(nrink): Matrix multiplication with `@` is nor supported for real
        # GPU kernels (but the GPU kernel interpreter allows this). Replace this
        # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
        # kernel interpreter.
        o_ref[row_slice, col_slice] = x_ref[row_slice, :] @ y_ref[:, col_slice]

      return _matmul_kernel(x, y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = matmul(x, y)
    np.testing.assert_allclose(z, x @ y, atol=1e-3)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.parameters(False, True)
  def test_run_scoped(self, with_race):
    mesh = plgpu.Mesh(num_threads=2, thread_name='n')

    @jax.jit
    def f(x):
      def inner(o_ref):
        @pl.core_map(
            mesh,
            interpret=mosaic_interpret.InterpretParams(
                detect_races=True,
            ),
        )  # type: ignore[wrong-arg-types]
        def _():
          def body(ref):
            @pl.when(jax.lax.axis_index('n') == 0)
            def _():
              ref[...] = jnp.zeros_like(ref[...])
              o_ref[0, ...] = ref[...]

            @pl.when(jax.lax.axis_index('n') == 1)
            def _():
              ref[...] = jnp.ones_like(ref[...])
              o_ref[1, ...] = ref[...]

          pl.run_scoped(
              body,
              plgpu.GMEM(o_ref.shape[1:], dtype=o_ref.dtype),
              collective_axes=('n',) if with_race else (),
          )

      y = pl.run_state(inner)(x)
      return y

    y = f(jnp.zeros((2, 16, 128)))

    if with_race:
      # Due to the presence of a race, we cannot expect `y` to have a
      # well-defined value. Hence, we do not assert anything about `y`.
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      np.testing.assert_array_equal(
          y, np.broadcast_to(np.arange(2).reshape(2, 1, 1), y.shape)
      )
      self.assertFalse(mosaic_interpret.get_races().races_found)

  # Test adapted from
  # https://docs.jax.dev/en/latest/pallas/gpu/reference.html#using-multiple-pallas-threads-per-cuda-block
  def test_producer_consumer_threads_with_barrier(self):
    x = jnp.arange(128, dtype=jnp.float32)

    @functools.partial(
        plgpu.kernel,
        out_shape=x,
        scratch_shapes=dict(
            smem_ref=plgpu.SMEM(x.shape, x.dtype),
            barrier_ref=plgpu.Barrier(),
        ),
        num_threads=2,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def producer_thread():
        smem_ref[...] = x_ref[...] + 1
        plgpu.barrier_arrive(barrier_ref)

      @pl.when(thread_id == 1)
      def consumer_thread():
        plgpu.barrier_wait(barrier_ref)
        out_ref[...] = smem_ref[...] + 1

    y = _kernel(x)
    np.testing.assert_array_equal(y, x + 2)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.parameters(range(2, 17))
  def test_single_barrier_with_multiple_arrival(self, num_threads):

    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=dict(
            smem_ref=plgpu.SMEM((num_threads - 1,), jnp.int32),
            barrier_ref=plgpu.Barrier(
                num_arrivals=num_threads - 1, num_barriers=1
            ),
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, smem_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_wait(barrier_ref)
        out_ref[...] = sum(smem_ref[...])

      @pl.when(thread_id > 0)
      def _():
        smem_ref[thread_id - 1] = thread_id
        plgpu.barrier_arrive(barrier_ref)

    y = _kernel()
    self.assertEqual(y, sum(range(num_threads)))
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.parameters(range(2, 17))
  def test_multiple_barriers_with_single_arrival(self, num_threads):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=dict(
            smem_ref=plgpu.SMEM((num_threads - 1,), jnp.int32),
            barrier_ref=plgpu.Barrier(
                num_arrivals=1, num_barriers=num_threads - 1
            ),
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, smem_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        for b in range(num_threads - 1):
          plgpu.barrier_wait(barrier_ref.at[b])
        out_ref[...] = sum(smem_ref[...])

      @pl.when(thread_id > 0)
      def _():
        smem_ref[thread_id - 1] = thread_id
        plgpu.barrier_arrive(barrier_ref.at[thread_id - 1])

    y = _kernel()
    self.assertEqual(y, sum(range(num_threads)))
    self.assertFalse(mosaic_interpret.get_races().races_found)

  # Test adapted from
  # https://docs.jax.dev/en/latest/pallas/gpu/reference.html#explicit-arrival-cross-thread-synchronization
  #
  # `buffer_size` has to be at least 2 so that the sizes of the `produced` and
  # `consumed` barriers are at least 2 (for each barrier). Otherwise the
  # indexing `produced.at[_]`/`consumed.at[_]` will not work.
  @jtu.parameterized.product(
      skip_floating_point_ops=[False, True],
      input_size=[1, 2, 4, 16, 64, 128],
      buffer_size=[2, 4, 8, 16],
  )
  def test_barrier_for_buffering(
      self, skip_floating_point_ops, input_size, buffer_size, seed=0
  ):
    k = jax.random.key(seed)
    x = jax.random.normal(k, (input_size,), dtype=jnp.float32)

    @functools.partial(
        plgpu.kernel,
        out_shape=x,
        scratch_shapes=dict(
            queue=plgpu.SMEM((buffer_size,), jnp.float32),
            produced=plgpu.Barrier(num_arrivals=1, num_barriers=buffer_size),
            consumed=plgpu.Barrier(num_arrivals=1, num_barriers=buffer_size),
        ),
        num_threads=2,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(
            detect_races=True, skip_floating_point_ops=skip_floating_point_ops
        ),
    )
    def _kernel(x_ref, out_ref, queue, produced, consumed):
      thread_id = jax.lax.axis_index('t')
      _get_slot = lambda i: jax.lax.rem(i, buffer_size)

      def _thread0_body(i, _):
        slot = _get_slot(i)
        @pl.when(i >= buffer_size)
        def _await_consumed():
          plgpu.barrier_wait(consumed.at[slot])
        queue[slot] = 3.0 * x_ref[i]
        plgpu.barrier_arrive(produced.at[slot])

      pl.when(thread_id == 0)(
          lambda: jax.lax.fori_loop(0, input_size, _thread0_body, None)
      )

      def _thread1_body(i, _):
        slot = _get_slot(i)
        plgpu.barrier_wait(produced.at[slot])
        out_ref[i] = queue[slot] + 42.0
        plgpu.barrier_arrive(consumed.at[slot])

      pl.when(thread_id == 1)(
          lambda: jax.lax.fori_loop(0, input_size, _thread1_body, None)
      )

      # TODO(nrink): This epilogue is needed to satisfy the requirement that all
      # completed barrier arrivals must have been waited for by the end of
      # kernel execution. Relax this requirement to match what is required when
      # the kernel is executed on real GPU hardware, where having unawaited
      # completed arrivals is acceptable immediately before returning from the
      # kernel.
      def _thread0_epilogue(i, _):
        slot = _get_slot(i)
        @pl.when(i < buffer_size)
        def _await_consumed():
          plgpu.barrier_wait(consumed.at[slot])

      pl.when(thread_id == 0)(
          lambda: jax.lax.fori_loop(0, input_size, _thread0_epilogue, None)
      )

    y = _kernel(x)
    self.assertFalse(mosaic_interpret.get_races().races_found)
    if skip_floating_point_ops:
      np.testing.assert_array_equal(y, jnp.full_like(y, jnp.inf))
    else:
      np.testing.assert_array_equal(y, 3.0 * x + 42.0)
    # TODO(nrink): Add a variant of this test case that does not correctly use
    # the `consumed` barrier and therefore has a race. Test that the race is
    # detected.

  def test_indexing_singleton_barrier_ok(self):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=(plgpu.Barrier(),),
        num_threads=1,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')
      plgpu.barrier_arrive(barrier_ref.at[thread_id])
      out_ref[...] = 42

    y = _kernel()
    self.assertEqual(y, 42)

  def test_not_indexing_multiple_barriers_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=(plgpu.Barrier(num_barriers=2),),
        num_threads=1,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      plgpu.barrier_arrive(barrier_ref)
      out_ref[...] = 42

    with self.assertRaisesRegex(
        ValueError,
        r'Attempting to operate on barrier without indexing, but `num_barriers'
        r' = 2`',
    ):
      _kernel()
    mosaic_interpret.reset_gpu_interpret_mode_state()

  def test_wait_for_barrier_twice(self):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=dict(
            barrier_ref=plgpu.Barrier(num_arrivals=1, num_barriers=2)
        ),
        num_threads=3,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        out_ref[...] = 1
        plgpu.barrier_wait(barrier_ref.at[0])
        out_ref[...] = 2
        plgpu.barrier_arrive(barrier_ref.at[1])
        plgpu.barrier_wait(barrier_ref.at[0])
        out_ref[...] = 3

      @pl.when(thread_id == 1)
      def _():
        plgpu.barrier_arrive(barrier_ref.at[0])

      @pl.when(thread_id == 2)
      def _():
        plgpu.barrier_wait(barrier_ref.at[1])
        plgpu.barrier_arrive(barrier_ref.at[0])

    y = _kernel()
    self.assertEqual(y, 3)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_completing_barrier_twice_in_same_thread_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        interpret=mosaic_interpret.InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      plgpu.barrier_arrive(barrier_ref)
      plgpu.barrier_arrive(barrier_ref)
      out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception,
        r'Barrier arrival was completed again before previous completion was'
        r' observed by a thread.',
    ):
      _kernel()
    mosaic_interpret.reset_gpu_interpret_mode_state()

  def test_completing_barrier_twice_in_different_threads_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_shapes=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        interpret=mosaic_interpret.InterpretParams(),
        num_threads=2,
        thread_name='t',
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')
      plgpu.barrier_arrive(barrier_ref)
      out_ref[thread_id] = thread_id

    with self.assertRaisesRegex(
        Exception,
        r'Barrier arrival was completed again before previous completion was'
        r' observed by a thread.',
    ):
      _kernel()
    mosaic_interpret.reset_gpu_interpret_mode_state()

  @jtu.parameterized.product(
      num_arriving_threads=list(range(1, 17)),
      num_observing_threads=list(range(1, 17)),
      num_threads=[16],
  )
  def test_barrier_wait_in_multiple_threads_ok(
      self, num_arriving_threads, num_observing_threads, num_threads
  ):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((num_threads,), jnp.int32),
        scratch_shapes=dict(
            barrier_ref=plgpu.Barrier(
                num_arrivals=num_arriving_threads, num_barriers=1
            )
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id < num_arriving_threads)
      def _():
        out_ref[thread_id] = thread_id
        plgpu.barrier_arrive(barrier_ref)

      @pl.when(thread_id >= num_threads - num_observing_threads)
      def _():
        out_ref[thread_id] = thread_id
        plgpu.barrier_wait(barrier_ref)

    _kernel()

  def test_more_barrier_completions_than_waits_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        num_threads=2,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_arrive(barrier_ref)

      @pl.when(thread_id == 1)
      def _():
        plgpu.barrier_wait(barrier_ref)
        plgpu.barrier_arrive(barrier_ref)
        out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception,
        r'Thread 1 did not observe all phases \(2\) for barrier \(but observed 1'
        r' phase\).',
    ):
      _kernel()
    mosaic_interpret.reset_gpu_interpret_mode_state()

  def test_not_waiting_for_all_barrier_completions_in_thread_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=dict(
            barrier_ref=plgpu.Barrier(num_arrivals=1, num_barriers=2)
        ),
        num_threads=3,
        thread_name='t',
        interpret=mosaic_interpret.InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')

      @pl.when(thread_id == 0)
      def _():
        plgpu.barrier_arrive(barrier_ref.at[0])

      @pl.when(thread_id == 1)
      def _():
        # This thread observes only the first completed arrival at
        # `barrier_ref.at[0]`
        plgpu.barrier_wait(barrier_ref.at[0])
        plgpu.barrier_arrive(barrier_ref.at[0])
        plgpu.barrier_arrive(barrier_ref.at[1])

      @pl.when(thread_id == 2)
      def _():
        plgpu.barrier_wait(barrier_ref.at[1])
        # This thread observes only the second completed arrival at
        # `barrier_ref.at[0]`
        plgpu.barrier_wait(barrier_ref.at[0])
        out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception,
        r'Thread 2 is awaiting phase 1, but barrier is already at phase 2.',
    ):
      _kernel()
    mosaic_interpret.reset_gpu_interpret_mode_state()

  @jtu.parameterized.product(
      num_blocks_w=[1, 2, 3],
      num_blocks_x=[1, 2, 3],
      num_blocks_y=[1, 2, 3],
      num_threads=[1, 2, 3],
  )
  def test_grid_iteration(
      self, num_blocks_w, num_blocks_x, num_blocks_y, num_threads
  ):
    def _kernel(a_gmem, out_gmem):
      w = jax.lax.axis_index('w')
      x = jax.lax.axis_index('x')
      y = jax.lax.axis_index('y')
      z = jax.lax.axis_index('z')

      offset = (
          w * num_blocks_x * num_blocks_y * num_threads
          + x * num_blocks_y * num_threads
          + y * num_threads
          + z
      )
      out_gmem[w, x, y, z] = a_gmem[w, x, y, z] + offset

    a = 42 * jnp.ones(
        (num_blocks_w, num_blocks_x, num_blocks_y, num_threads),
        dtype=jnp.int32,
    )

    kernel = plgpu.kernel(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        grid=(num_blocks_w, num_blocks_x, num_blocks_y),
        grid_names=('w', 'x', 'y'),
        num_threads=num_threads,
        thread_name='z',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )

    y = kernel(a)
    expected = a + jnp.arange(
        num_blocks_w * num_blocks_x * num_blocks_y * num_threads,
        dtype=a.dtype,
    ).reshape(a.shape)
    np.testing.assert_array_equal(y, expected)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      tile_x=[1, 2, 4],
      tile_y=[1, 2, 4],
      tile_z=[1, 2, 4],
      swap_grid_axes=[True, False],
  )
  def test_add_over_grid(self, tile_x, tile_y, tile_z, swap_grid_axes):
    dtype = jnp.int32
    x, y, z = 4, 4, 4

    assert x % tile_x == 0
    assert y % tile_y == 0
    assert z % tile_z == 0

    a = jnp.arange(x * y * z, dtype=dtype).reshape((x, y, z))
    b = jnp.ones((x, y, z), dtype=dtype)

    x_iters = x // tile_x
    y_iters = y // tile_y
    z_iters = z // tile_z

    def _kernel(a_gmem, b_gmem, out_gmem):
      xi = jax.lax.axis_index('x')
      yi = jax.lax.axis_index('y')
      zi = jax.lax.axis_index('z')
      xi_slice = pl.ds(xi * tile_x, tile_x)
      yi_slice = pl.ds(yi * tile_y, tile_y)
      zi_slice = pl.ds(zi * tile_z, tile_z)

      out_gmem[xi_slice, yi_slice, zi_slice] = (
          a_gmem[xi_slice, yi_slice, zi_slice]
          + b_gmem[xi_slice, yi_slice, zi_slice]
      )

    kernel = plgpu.kernel(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((x, y, z), dtype),
        grid=_maybe_reverse((x_iters, y_iters), swap_grid_axes),
        grid_names=_maybe_reverse(('x', 'y'), swap_grid_axes),
        num_threads=z_iters,
        thread_name='z',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )

    expected = a + b
    y = kernel(a, b)
    np.testing.assert_array_equal(y, expected)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      tile_m=[1, 2, 4],
      tile_k=[1, 2, 4],
      tile_n=[1, 2, 4],
      swap_grid_axes=[True, False],
  )
  def test_matmul_over_grid_with_barrier_and_smem(
      self, tile_m, tile_k, tile_n, swap_grid_axes
  ):
    dtype = jnp.int32
    m, k, n = 4, 4, 4

    assert m % tile_m == 0
    assert k % tile_k == 0
    assert n % tile_n == 0

    a = jnp.arange(16, dtype=dtype).reshape((m, k))
    b = jnp.ones((k, n), dtype=dtype)

    m_iters = m // tile_m
    k_iters = k // tile_k
    n_iters = n // tile_n

    def _kernel(a_gmem, b_gmem, out_gmem, acc_smem, barrier):
      mi = jax.lax.axis_index('m')
      ki = jax.lax.axis_index('k')
      ni = jax.lax.axis_index('n')
      mi_slice = pl.ds(mi * tile_m, tile_m)
      ki_slice = pl.ds(ki * tile_k, tile_k)
      ni_slice = pl.ds(ni * tile_n, tile_n)

      # We map the reduced dimension, i.e. `k`, to the thread dimension. This
      # allows us to do the accumulation into an `SMEM` buffer, i.e. `acc_smem`.
      # (Note that a fresh `SMEM` buffer is allocated for each grid point, but
      # for each fixed grid point, a single `SMEM` buffer is shared across the
      # threads.) We then need to use barriers to sequentialize access to the
      # `SMEM` buffer between the threads. (Note that the specific sequential
      # order of the updates to `acc_smem` does not matter, so long as there are
      # no races.)

      @pl.when(ki == 0)
      def _():
        acc_smem[...] = jnp.zeros(acc_smem.shape, dtype=acc_smem.dtype)
        plgpu.barrier_arrive(barrier.at[0])

      plgpu.barrier_wait(barrier.at[ki])
      # TODO(nrink): Matrix multiplication with `@` is not supported for real
      # GPU kernels (but the GPU kernel interpreter allows this). Replace this
      # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
      # kernel interpreter.
      acc_smem[...] += a_gmem[mi_slice, ki_slice] @ b_gmem[ki_slice, ni_slice]
      plgpu.barrier_arrive(barrier.at[(ki + 1) % k_iters])

      @pl.when(ki == 0)
      def _():
        plgpu.barrier_wait(barrier.at[0])
        out_gmem[mi_slice, ni_slice] = acc_smem[...]

    kernel = plgpu.kernel(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_shapes=dict(
            acc_smem=plgpu.SMEM((tile_m, tile_n), dtype),
            barrier=plgpu.Barrier(num_barriers=k_iters),
        ),
        grid=_maybe_reverse((m_iters, n_iters), swap_grid_axes),
        grid_names=_maybe_reverse(('m', 'n'), swap_grid_axes),
        num_threads=k_iters,
        thread_name='k',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )

    expected = a @ b
    y = kernel(a, b)
    np.testing.assert_array_equal(y, expected)
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_matmul_over_grid_with_race(self, tile_m=4, tile_k=2, tile_n=4):
    dtype = jnp.int32
    m, k, n = 4, 4, 4

    assert m % tile_m == 0
    assert k % tile_k == 0
    assert n % tile_n == 0

    a = jnp.arange(16, dtype=dtype).reshape((m, k))
    b = jnp.ones((k, n), dtype=dtype)

    m_iters = m // tile_m
    k_iters = k // tile_k
    n_iters = n // tile_n

    def _kernel(a_gmem, b_gmem, _, acc_smem):
      mi = jax.lax.axis_index('m')
      ki = jax.lax.axis_index('k')
      ni = jax.lax.axis_index('n')
      mi_slice = pl.ds(mi * tile_m, tile_m)
      ki_slice = pl.ds(ki * tile_k, tile_k)
      ni_slice = pl.ds(ni * tile_n, tile_n)

      # The two threads race to update `acc_smem`. We do not bother with
      # initializing `acc_mem` to zero, nor with copying the final result out to
      # `GMEM`, as is done in the correct (i.e. race-free) test above (i.e. in
      # `test_matmul_over_grid_with_barrier_and_smem`). This would only
      # introduce additional races.
      #
      # TODO(nrink): Matrix multiplication with `@` is not supported for real
      # GPU kernels (but the GPU kernel interpreter allows this). Replace this
      # with a `wgmma` or `tcgen05_mma` once these are supported by the GPU
      # kernel interpreter.
      acc_smem[...] += a_gmem[mi_slice, ki_slice] @ b_gmem[ki_slice, ni_slice]

    kernel = plgpu.kernel(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_shapes=dict(
            acc_smem=plgpu.SMEM((tile_m, tile_n), dtype),
        ),
        grid=(m_iters, n_iters),
        grid_names=('m', 'n'),
        num_threads=k_iters,
        thread_name='k',
        interpret=mosaic_interpret.InterpretParams(detect_races=True),
    )

    kernel(a, b)
    self.assertTrue(mosaic_interpret.get_races().races_found)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
