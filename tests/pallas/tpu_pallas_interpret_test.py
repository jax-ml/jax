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
contains only tests that do not use shard_map.
"""

from collections.abc import Callable
import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as mosaic_interpret
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()
jax.config.update('jax_threefry_partitionable', True)


class CountStoreCallbacksContext:
  """Wraps the I/O callback `store` into a callback that counts the number of calls to `store`."""

  def __init__(self):
    self._num_stores = 0
    self._saved = mosaic_interpret.store

  def __enter__(self):
    def _store_callback(self, *args, **kwargs):
      self._num_stores += 1
      return self._saved(*args, **kwargs)

    mosaic_interpret.store = functools.partial(_store_callback, self)
    return self

  def __exit__(self, ty, value, traceback):
    del ty, value, traceback
    mosaic_interpret.store = self._saved

  @property
  def num_stores(self):
    return self._num_stores


@dataclasses.dataclass(frozen=True)
class ProcessedGridPoint():
  """Represents a grid point and the ID of the core that has processed it."""
  grid_point: tuple[int, ...]
  core_id: int


class GridPointRecorderContext:
  """Records grid points in the order in which they are procsessed."""

  def __init__(self):
    self._grid_points: list[ProcessedGridPoint] = []

  def __enter__(self):
    return self

  def __exit__(self, ty, value, traceback):
    ...

  def get_recorder(self) -> Callable[[tuple[np.int32, ...], np.int32], None]:
    def _recorder(grid_point, core_id):
      processed_grid_point = ProcessedGridPoint(
          tuple(int(coord) for coord in grid_point), int(core_id)
      )
      self._grid_points.append(processed_grid_point)

    return _recorder

  @property
  def grid_points(self) -> list[ProcessedGridPoint]:
    return sorted(self._grid_points, key=lambda x: x.core_id)


# TODO(jburnim): Figure out how to safely run different instance of TPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class InterpretTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      # Workaround for https://github.com/jax-ml/jax/issues/25671
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  def test_revisiting_is_an_error(self):
    def kernel(x_ref, o1_ref, o2_ref):
      pass

    @jax.jit
    def run():
      return pl.pallas_call(
          kernel,
          out_shape=[
              jax.ShapeDtypeStruct((16, 256), jnp.float32),
              jax.ShapeDtypeStruct((16, 256), jnp.float32),
          ],
          grid=(4, 4),
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=[
              pl.BlockSpec((4, 128), lambda i, j: (i, j // 2)),
              pl.BlockSpec((4, 128), lambda i, j: (j // 2, i % 2)),
          ],
          interpret=pltpu.InterpretParams(),
      )(jnp.zeros((8, 128)))

    with self.assertRaisesRegex(
        Exception, r'Revisited block .* of output 1 in iteration \(2, 0\)'):
      run()[0].block_until_ready()
    pltpu.reset_tpu_interpret_mode_state()

  def test_matmul_example(self):
    def matmul_kernel(x_ref, y_ref, z_ref):
      z_ref[...] = x_ref[...] @ y_ref[...]

    @jax.jit
    def matmul(x: jax.Array, y: jax.Array):
      return pl.pallas_call(
          matmul_kernel,
          out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
          grid=(2, 2),
          in_specs=[
              pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
              pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j)),
          ],
          out_specs=pl.BlockSpec(
              (x.shape[0] // 2, y.shape[1] // 2),
              lambda i, j: (i, j),
          ),
          interpret=pltpu.InterpretParams(),
      )(x, y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = matmul(x, y)
    np.testing.assert_allclose(z, x @ y, atol=1e-3)

  @parameterized.parameters('raise', 'uninitialized')
  def test_out_of_bounds_block_spec(self, out_of_bounds_reads):
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def run(input_offset, output_offset):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
          out_specs=pl.BlockSpec((4, 128), lambda i: (i+output_offset, 0)),
          in_specs=[pl.BlockSpec((4, 128), lambda i: (i+input_offset, 0))],
          grid=(4,),
          interpret=pltpu.InterpretParams(
              out_of_bounds_reads=out_of_bounds_reads),
      )(jnp.zeros((16, 128), jnp.float32))

    # Out-of-bounds input block.
    if out_of_bounds_reads == 'uninitialized':
      out = np.array(run(1, 0))
      np.testing.assert_equal(out[:12], 0.0)
      self.assertTrue(np.isnan(out[12:]).all())
    elif out_of_bounds_reads == 'raise':
      with self.assertRaisesRegex(
          Exception, 'Out-of-bounds block index .* for input'):
        run(1, 0)
      pltpu.reset_tpu_interpret_mode_state()

    # Out-of-bounds output block.
    if out_of_bounds_reads == 'raise':
      with self.assertRaisesRegex(
          Exception, 'Out-of-bounds block index .* for output'):
        run(0, 2)
      pltpu.reset_tpu_interpret_mode_state()

  @parameterized.parameters('raise', 'uninitialized')
  def test_out_of_bounds_read_index(self, out_of_bounds_reads):
    def kernel(s_ref, x_ref, o_ref):
      def read(ref, i):
        return ref[i]
      def body(carry):
        i, accum = carry
        accum += read(x_ref, i)
        return (i + 1, accum)
      start = read(x_ref, s_ref[0])
      stop = read(x_ref, s_ref[1])
      o_ref[0] = jax.lax.while_loop(
        lambda c: c[0] < stop,
        body,
        (start, jnp.int32(0)))[1]

    @jax.jit
    def run(s, x):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SMEM)
          ],
          interpret=pltpu.InterpretParams(
              out_of_bounds_reads=out_of_bounds_reads),
      )(s, x)

    self.assertEqual(run(jnp.array([0, 1], jnp.int32),
                         jnp.array([2, 5, 9, 15, 17], jnp.int32)),
                     9 + 15 + 17)

    if out_of_bounds_reads == 'uninitialized':
      self.assertLess(run(jnp.array([0, 1], jnp.int32),
                          jnp.array([2, 6, 9, 15, 17], jnp.int32)),
                      0)  # sum includes one uninitialized value
    elif out_of_bounds_reads == 'raise':
      with self.assertRaisesRegex(Exception, 'Out-of-bounds read'):
        run(jnp.array([0, 1], jnp.int32),
            jnp.array([2, 6, 9, 15, 17], jnp.int32)),
      pltpu.reset_tpu_interpret_mode_state()

  @parameterized.parameters('raise', 'uninitialized')
  def test_out_of_bounds_read_range(self, out_of_bounds_reads):
    def kernel(x_ref, o_ref, sem):
      pltpu.async_copy(x_ref.at[pl.ds(jnp.int32(4), 8), 1], o_ref, sem).wait()

    @jax.jit
    def run():
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128,), jnp.float32),
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          interpret=pltpu.InterpretParams(
              out_of_bounds_reads=out_of_bounds_reads),
      )(jnp.zeros((8, 4, 128), jnp.float32))

    if out_of_bounds_reads == 'raise':
      with self.assertRaisesRegex(Exception, 'Out-of-bounds read'):
        run().block_until_ready()
      pltpu.reset_tpu_interpret_mode_state()
    else:
      out = run().block_until_ready()
      np.testing.assert_equal(np.array(out[:4]), 0.0)
      self.assertTrue(np.isnan(out[4:]).all())

  def test_scalar_prefetch_example(self):
    def dynamic_slice_kernel(indices, x_ref, o_ref):
      del indices
      o_ref[...] = x_ref[...]

    @functools.partial(jax.jit, static_argnums=(2,))
    def block_dynamic_slice(x, starts, sizes):
      grid_spec = pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(1, 1),
          in_specs=[
              pl.BlockSpec(
                  sizes, lambda i, j, block_idx: (block_idx[0], block_idx[1])
              )
          ],
          out_specs=pl.BlockSpec(sizes, lambda *_: (0, 0)),
      )

      kernel = pl.pallas_call(
          dynamic_slice_kernel,
          grid_spec=grid_spec,
          out_shape=jax.ShapeDtypeStruct(shape=sizes, dtype=x.dtype),
          interpret=pltpu.InterpretParams(),
      )
      block_idx = jnp.array([starts[0] // sizes[0], starts[1] // sizes[1]])
      return kernel(block_idx, x)

    shape = (512, 512)
    x = jnp.reshape(jnp.arange(np.prod(shape), dtype=jnp.int32), shape)
    result = block_dynamic_slice(
        x, starts=jnp.array([128, 256]), sizes=(128, 128)
    )
    ref = jax.lax.dynamic_slice(
        x, start_indices=(128, 256), slice_sizes=(128, 128)
    )
    np.testing.assert_allclose(result, ref)

  def test_dynamic_grid_and_aliasing(self):
    def kernel(s1_ref, s2_ref, x_ref, o_ref):
      del s2_ref
      o_ref[...] = x_ref[...] + s1_ref[0].astype(x_ref.dtype)

    iters = jax.random.randint(jax.random.key(0), (), 10, 20, dtype=jnp.int32)

    @jax.jit
    def f(s1, s2, x):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=2,
              grid=(iters,),
              in_specs=[
                  pl.BlockSpec(x.shape, lambda i, *_: (0, 0)),
              ],
              out_specs=pl.BlockSpec(x.shape, lambda i, *_: (0, 0)),
          ),
          input_output_aliases={2: 0},
          interpret=pltpu.InterpretParams(),
      )(s1, s2, x)

    s1 = jnp.array([1], dtype=jnp.int32)
    s2 = jnp.array([2], dtype=jnp.int32)
    x = jnp.arange(32 * 128.0).reshape((32, 128))
    y = f(s1, s2, x)
    # NOTE: No matter how many times the kernel body is run, the kernel input
    # buffer will only be written once by the pallas_call machinery, just
    # before the first iteration. So the output will be x + 1 , despite the
    # aliasing in HBM.
    np.testing.assert_allclose(y, x + 1.0)

  def test_aliasing(self):
    def kernel(x_ref, o_ref, s_ref):
      @pl.when((pl.program_id(0) == 0) & (pl.program_id(1) == 0))
      def _():
        s_ref[0] = jnp.int32(0)

      s = s_ref[0]
      s_ref[0] = s + 1
      o_ref[:] = x_ref[:] + s.astype(x_ref.dtype)

    x = jnp.zeros((4 * 8, 4 * 128))
    y = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(4, 4),
        in_specs=[
            pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec(
            block_shape=(8, 128), index_map=lambda i, j: (j, i)
        ),
        scratch_shapes=(pltpu.SMEM((1,), jnp.int32),),
        input_output_aliases={0: 0},
        interpret=pltpu.InterpretParams(),
    )(x)

    expected = np.zeros((4, 4))
    t = 0
    for i in range(4):
      for j in range(4):
        expected[j, i] = expected[i, j] + t
        t += 1
    # NOTE: expected is
    #   [[0, 5, 10, 15],
    #    [1, 5, 15, 20],
    #    [2, 6, 10, 25],
    #    [3, 7, 11, 15]]
    np.testing.assert_allclose(y[::8, ::128], expected)

  @parameterized.parameters('eager', 'on_wait')
  def test_race_detection(self, dma_execution_mode):
    def kernel_without_race(x_ref, o_ref, t_ref, sem):
      copy = pltpu.make_async_copy(x_ref, t_ref, sem)
      copy.start()
      copy.wait()
      o_ref[...] = t_ref[...] + 1.0

    def kernel_with_race(x_ref, o_ref, t_ref, sem):
      copy = pltpu.make_async_copy(x_ref, t_ref, sem)
      copy.start()
      # This read of t_ref races with the above DMA's write of t_ref.
      o_ref[...] = t_ref[...] + 1.0
      copy.wait()

    x = jnp.zeros((8, 128), jnp.float32)
    y = pl.pallas_call(
        kernel_without_race,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        scratch_shapes=[
            pltpu.VMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ],
        interpret=pltpu.InterpretParams(
            detect_races=True, dma_execution_mode=dma_execution_mode
        ),
    )(x).block_until_ready()
    self.assertFalse(mosaic_interpret.races.races_found)
    np.testing.assert_allclose(y, x + 1.0)

    pl.pallas_call(
        kernel_with_race,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        scratch_shapes=[
            pltpu.VMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ],
        interpret=pltpu.InterpretParams(
            detect_races=True, dma_execution_mode=dma_execution_mode
        ),
    )(x).block_until_ready()
    self.assertTrue(mosaic_interpret.races.races_found)

  def test_skip_floating_point_ops(self):
    def matmul_kernel(x_ref, y_ref, z_ref):
      z_ref[...] = x_ref[...] @ y_ref[...]

    def matmul(x: jax.Array, y: jax.Array):
      return pl.pallas_call(
          matmul_kernel,
          out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
          interpret=pltpu.InterpretParams(
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

  @parameterized.parameters('nan', 'zero')
  def test_uninitialized_memory(self, uninitialized_memory):
    def kernel(o1_ref, o2_ref, o3_ref, t1_ref, t2_ref):
      o1_ref[...] = t1_ref[...]
      o2_ref[...] = t2_ref[...]

    x, y, z = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((8, 128), jnp.int16),
            jax.ShapeDtypeStruct((8, 128), jnp.float32),
        ],
        in_specs=[],
        scratch_shapes=[
            pltpu.VMEM((8, 128), jnp.bfloat16),
            pltpu.VMEM((8, 128), jnp.int16),
        ],
        interpret=pltpu.InterpretParams(
            uninitialized_memory=uninitialized_memory
        ),
    )()
    if uninitialized_memory == 'nan':
      self.assertTrue(jnp.isnan(x).all())
      np.testing.assert_equal(np.array(y), 32767)
      self.assertTrue(jnp.isnan(z).all())
    if uninitialized_memory == 'zero':
      np.testing.assert_equal(np.array(x), 0)
      np.testing.assert_equal(np.array(y), 0)
      np.testing.assert_equal(np.array(z), 0)

  def test_correct_number_of_stores(self):
    def kernel(x_ref, s_ref, o_ref):
      s = s_ref[0]
      x_ref[:] += jax.lax.full_like(x_ref, s)
      s_ref[0] = s + 1
      o_ref[:] = x_ref[:]

    def kernel_call(x, s):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
          grid=(2, 2),
          in_specs=[
              pl.BlockSpec((8, 256), lambda i, j: (i, 0)),
              pl.BlockSpec(memory_space=pltpu.SMEM),
          ],
          out_specs=pl.BlockSpec((8, 256), lambda i, j: (i, 0)),
          interpret=pltpu.InterpretParams(),
      )(x, s)

    with CountStoreCallbacksContext() as store_callbacks_counter:
      result = jax.jit(kernel_call)(
          jnp.zeros((16, 256), jnp.float32), jnp.zeros((1,), jnp.int32)
      )
      np.testing.assert_allclose(result[::8, ::256], [[1.0], [5.0]])
      self.assertEqual(store_callbacks_counter.num_stores, 5)

  def test_randomization_of_parallel_dimensions(self):
    def kernel(s_ref, o_ref):
      s = s_ref[0]
      s_ref[0] = s + 1
      o_ref[:] = jax.lax.full_like(o_ref, s)

    def kernel_call_dimensions_parallel_arbitrary(s, grid_point_recorder):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((32, 512), jnp.float32),
          grid=(4, 4),
          in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
          out_specs=pl.BlockSpec((8, 128), lambda i, j: (i, j)),
          interpret=pltpu.InterpretParams(
              random_seed=12345, grid_point_recorder=grid_point_recorder
          ),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=('parallel', 'arbitrary')
          ),
      )(s)

    with GridPointRecorderContext() as grid_point_recorder:
      result = jax.jit(
          kernel_call_dimensions_parallel_arbitrary, static_argnums=1
      )(
          jnp.zeros((1,), jnp.int32),
          grid_point_recorder.get_recorder(),
      )
      np.testing.assert_allclose(
          result[::8, ::128],
          [
              [ 8.0,  9.0, 10.0, 11.0],
              [12.0, 13.0, 14.0, 15.0],
              [ 0.0,  1.0,  2.0,  3.0],
              [ 4.0,  5.0,  6.0,  7.0],
          ],
      )
      self.assertListEqual(
          grid_point_recorder.grid_points,
          [
              ProcessedGridPoint((2, 0), 0),
              ProcessedGridPoint((2, 1), 0),
              ProcessedGridPoint((2, 2), 0),
              ProcessedGridPoint((2, 3), 0),
              ProcessedGridPoint((3, 0), 0),
              ProcessedGridPoint((3, 1), 0),
              ProcessedGridPoint((3, 2), 0),
              ProcessedGridPoint((3, 3), 0),
              ProcessedGridPoint((0, 0), 0),
              ProcessedGridPoint((0, 1), 0),
              ProcessedGridPoint((0, 2), 0),
              ProcessedGridPoint((0, 3), 0),
              ProcessedGridPoint((1, 0), 0),
              ProcessedGridPoint((1, 1), 0),
              ProcessedGridPoint((1, 2), 0),
              ProcessedGridPoint((1, 3), 0),
          ],
      )

  def test_dimensions_arbitrary_parallel_raises(self):
    def kernel_call(s):
      def kernel(s_ref, o_ref):
        s = s_ref[0]
        o_ref[0] = s

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((32, 512), jnp.float32),
          grid=(4, 4),
          in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
          out_specs=pl.BlockSpec((8, 128), lambda i, j: (i, j)),
          interpret=pltpu.InterpretParams(random_seed=12345),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=('arbitrary', 'parallel')
          ),
      )(s)

    with self.assertRaises(ValueError):
      jax.jit(kernel_call)(
          jnp.zeros((1,), jnp.int32),
      )

  def test_dynamic_parallel_dimension_raises(self):
    def kernel(o_ref):
      o_ref[0] = 42.0

    @jax.jit
    def kernel_call_dynamic_parallel_dimension():
      dim_size = jax.random.randint(
          jax.random.key(0), (), 10, 20, dtype=jnp.int32
      )
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
          grid=(dim_size,),
          in_specs=[],
          out_specs=pl.BlockSpec((1,), lambda _: (0,)),
          interpret=pltpu.InterpretParams(),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=('parallel',)
          ),
      )()

    with self.assertRaises(jax.errors.ConcretizationTypeError):
      kernel_call_dynamic_parallel_dimension()

  @parameterized.product(
      num_cores=[1, 2, 4],
      use_context_manager=[False, True]
  )
  def test_core_map(self, num_cores, use_context_manager):
    mesh = pltpu.create_tensorcore_mesh('x', num_cores=num_cores)
    interpret = False if use_context_manager else pltpu.InterpretParams()

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        @pl.core_map(mesh, interpret=interpret)
        def _():
          num_cores = jax.lax.axis_size('x')
          slc_size = 16 // num_cores
          def alloc(x_vmem_ref, y_vmem_ref, dma_sem, sem):
            # Barrier so we deadlock unless the core_map is actually parallel.
            for i in range(num_cores):
              pl.semaphore_signal(sem, 1, core_index=i)
            pl.semaphore_wait(sem, num_cores)

            core_index = jax.lax.axis_index('x')
            slc = pl.ds(core_index * slc_size, slc_size)
            pltpu.async_copy(
                x_ref.at[slc],
                x_vmem_ref,
                dma_sem,
            ).wait()
            y = x_vmem_ref[...] + jax.lax.axis_index('x') + 1
            y_vmem_ref[...] = y
            pltpu.async_copy(y_vmem_ref, y_ref.at[slc], dma_sem).wait()
          pl.run_scoped(
              alloc,
              pltpu.VMEM((slc_size, 128), x_ref.dtype),
              pltpu.VMEM((slc_size, 128), y_ref.dtype),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.REGULAR,
          )
      _, y = pl.run_state(inner)((x, y))
      return y

    x = jnp.arange(16 * 128, dtype=jnp.int32).reshape((16, 128))
    expected_out = (
        x.reshape((num_cores, -1, 128)) + 1
        + jnp.arange(num_cores, dtype=jnp.int32)[..., None, None]
    ).reshape(x.shape)

    if use_context_manager:
      with pltpu.force_tpu_interpret_mode():
        y = f(x)
    else:
      y = f(x)
    np.testing.assert_array_equal(y, expected_out)

  def test_hbm_allocation_in_run_scoped_raises(self):
    mesh = pltpu.create_tensorcore_mesh('x', num_cores=1)

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)

      def inner(x):
        x_ref, y_ref = x

        @pl.core_map(
            mesh,
            interpret=pltpu.InterpretParams(
                allow_hbm_allocation_in_run_scoped=False
            ),
        )
        def _():
          def copy(hbm):
            pltpu.sync_copy(x_ref, hbm)
            pltpu.sync_copy(hbm, y_ref)

          pl.run_scoped(
              copy,
              pltpu.HBM(x_ref.shape, x_ref.dtype),
          )

      _, y =  pl.run_state(inner)((x, y))
      return y

    with self.assertRaisesRegex(
        ValueError, r'Cannot allocate HBM in `run_scoped`.'
    ):
      f(jnp.arange(8))

  @parameterized.product(
      first_core_to_copy=[0, 1], dma_execution_mode=['eager', 'on_wait']
  )
  def test_allocate_shared_buffer_in_core_map(
      self, first_core_to_copy, dma_execution_mode
  ):
    mesh = pltpu.create_tensorcore_mesh('x', num_cores=2)
    second_core_to_copy = 1 if first_core_to_copy == 0 else 0

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)

      def inner(refs):
        x_ref, y_ref = refs
        # Thanks to the semaphore `sem` below, this test is race-free, and both
        # cores access the shared HBM buffer entirely sequentially. If the
        # runtime management of buffers were not done carefully, two issues
        # could arise, each resulting in an attempt to access unallocated
        # memory:
        #  1. The first core to reach the `copy` function, inside the nested
        #     `run_scoped`, might find that the shared HBM buffer has not been
        #     allocated yet. This could happen if the other core were
        #     repsonsible for allocating the HBM buffer when entering the nested
        #     `run_scoped`; but that other core has not reached the nested
        #     `run_scoped` yet, and hence has not allocated the HBM buffer yet.
        #  2. The second core to reach the `copy` function might find that the
        #     shared HBM buffer has been deallocated already. This could happen
        #     if the other core (i.e. the first one to reach the `copy`
        #     function) were responsible for deallocating the HBM buffer when
        #     exiting the nested `run_scoped`. If that other core has already
        #     run ahead to the end of the nested `run_scoped`, it will have
        #     deallocated the HBM buffer.
        @pl.core_map(
            mesh,
            interpret=pltpu.InterpretParams(
                detect_races=True, allow_hbm_allocation_in_run_scoped=True,
                dma_execution_mode=dma_execution_mode,
            ),
        )
        def _():
          def body(sem):
            @pl.when(jax.lax.axis_index('x') == second_core_to_copy)
            def _():
              pltpu.semaphore_wait(sem, 1)

            def copy(x_hbm_ref):
              pltpu.sync_copy(x_ref, x_hbm_ref)
              pltpu.sync_copy(x_hbm_ref, y_ref)

            pl.run_scoped(
                copy,
                pltpu.HBM(x_ref.shape, x_ref.dtype),
            )

            @pl.when(jax.lax.axis_index('x') == first_core_to_copy)
            def _():
              pltpu.semaphore_signal(sem, 1, core_index=second_core_to_copy)

          pl.run_scoped(
              body,
              pltpu.SemaphoreType.REGULAR,
          )

      _, y = pl.run_state(inner)((x, y))
      return y

    x = jnp.arange(16 * 128, dtype=jnp.int32).reshape((16, 128))
    y = f(x)
    np.testing.assert_array_equal(y, x)
    self.assertFalse(mosaic_interpret.races.races_found)

  @parameterized.product(
      slow_core=[0, 1], dma_execution_mode=['eager', 'on_wait']
  )
  def test_allocate_shared_buffer_in_core_map_with_race(
      self, slow_core, dma_execution_mode
  ):
    mesh = pltpu.create_tensorcore_mesh('x', num_cores=2)

    @jax.jit
    def f(x, y):
      z = jnp.zeros_like(x)
      o = jnp.zeros((y.shape[0], y.shape[0]), dtype=y.dtype)

      def inner(refs):
        """Copies `x_ref` to `z_ref` and computes `y_ref @ y_ref^t` into `o_ref`."""
        x_ref, y_ref, z_ref, o_ref = refs
        @pl.core_map(
            mesh,
            interpret=pltpu.InterpretParams(
                detect_races=True,
                allow_hbm_allocation_in_run_scoped=True,
                dma_execution_mode=dma_execution_mode,
            ),
        )
        def _():
          # The slow core performs an expensive matrix multiplication, and then
          # copies from `x_ref` to `z_ref`, going through an HBM buffer that is
          # shared between the two cores. The other core, aka. the fast core,
          # proceeds directly to copying from `x_ref` to `z_ref`, going through
          # the same shared HBM buffer. If the shared buffer were, incorrectly,
          # deallocated by the fast core (once it is done copying from `x_ref`
          # to `z_ref`) and then reallocated by the slow core (before it starts
          # copying from `x_ref` to `z_ref`), we would not see any attempts of
          # accessing unallocated memory. However, we would also not detect any
          # races since the cores would operate on separate buffers.
          def body(x_hbm_ref, vmem_ref_0, vmem_ref_1):
            @pl.when(jax.lax.axis_index('x') == slow_core)
            def _():
              pltpu.sync_copy(y_ref, vmem_ref_0)
              vmem_ref_1[...] = vmem_ref_0[...] @ jnp.transpose(vmem_ref_0[...])
              pltpu.sync_copy(vmem_ref_1, o_ref)

            pltpu.sync_copy(x_ref, x_hbm_ref)
            pltpu.sync_copy(x_hbm_ref, z_ref)

          pl.run_scoped(
              body,
              pltpu.HBM(x.shape, dtype=x.dtype),
              pltpu.VMEM(y.shape, dtype=y.dtype),
              pltpu.VMEM((y.shape[0], y.shape[0]), dtype=y.dtype),
          )

      _, _, z, o = pl.run_state(inner)((x, y, z, o))
      return z, o

    x = jnp.arange(16 * 128, dtype=jnp.int32).reshape((16, 128))
    y = jax.random.randint(
        jax.random.key(0), (1024, 1024), minval=-100, maxval=100
    )
    _, o = f(x, y)
    # We do not assert that the first result of `f` must be equal to `x`. This
    # is because of the copying from `x` to the first result of `f` is racy, and
    # we should therefore not expect the first result of `f` to have a
    # well-defined value.
    np.testing.assert_array_equal(o, y @ jnp.transpose(y))
    self.assertTrue(mosaic_interpret.races.races_found)

  def test_two_cores_along_parallel_dimension_with_race(self):
    def kernel(x_ref, o_ref, vmem_ref):
      vmem_ref[...] = x_ref[...]
      o_ref[...] = x_ref[...] + vmem_ref[...]

    x = jnp.ones((8, 128), jnp.float32)
    trace_count = [0]

    @jax.jit
    def f(x):
      trace_count[0] += 1
      return pl.pallas_call(
          kernel,
          grid=(2,),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          scratch_shapes=[
              pltpu.VMEM(x.shape, x.dtype),
          ],
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=('parallel',),
          ),
          interpret=pltpu.InterpretParams(
              num_cores_or_threads=2,
              detect_races=False,
          ),
      )(x)

    y = f(x).block_until_ready()
    self.assertFalse(mosaic_interpret.races.races_found)
    np.testing.assert_allclose(y, 2.0 * x)

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(
        num_cores_or_threads=1,
        detect_races=True,
    )):
      y = f(x).block_until_ready()
    self.assertFalse(mosaic_interpret.races.races_found)
    np.testing.assert_allclose(y, 2.0 * x)
    self.assertEqual(trace_count[0], 2)

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(
        num_cores_or_threads=2,
        detect_races=True,
    )):
      y = f(x).block_until_ready()
    self.assertTrue(mosaic_interpret.races.races_found)
    np.testing.assert_allclose(y, 2.0 * x)
    self.assertEqual(trace_count[0], 3)

  def test_two_cores_along_parallel_dimension_no_race(self):
    def kernel(x_ref, o_ref, vmem_ref):
      vmem_ref[...] = x_ref[...]
      o_ref[...] = x_ref[...] + vmem_ref[...]

    x = jnp.ones((16, 128), jnp.float32)
    y = pl.pallas_call(
        kernel,
        grid=(2,),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        out_specs=pl.BlockSpec(
            (8, 128),
            lambda i: (i, 0),
        ),
        in_specs=[
            pl.BlockSpec(
                (8, 128),
                lambda i: (i, 0),
            ),
        ],
        scratch_shapes=[
            pltpu.VMEM((8, 128), x.dtype),
        ],
        interpret=pltpu.InterpretParams(
            num_cores_or_threads=2,
            detect_races=True,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel',)
        ),
    )(x).block_until_ready()
    self.assertFalse(mosaic_interpret.races.races_found)
    np.testing.assert_allclose(y, 2.0 * x)

  def test_parallel_dimension_and_multiple_cores(self):
    def kernel(s_ref, in_ref, o_ref):
      # NOTE: diff should be 0.
      diff = in_ref[...] - jnp.float32(4 * pl.program_id(0) + pl.program_id(1))

      s = s_ref[0]
      s_ref[0] = s + 1
      o_ref[:] = jax.lax.full_like(o_ref, s) + diff

    def kernel_call(s, num_cores_per_device, grid_point_recorder):
      block_input = jnp.repeat(
          jnp.repeat(
              jnp.arange(16, dtype=jnp.float32).reshape((4, 4)), 128, axis=1),
          8, axis=0)
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((32, 512), jnp.float32),
          grid=(4, 4),
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec((8, 128), lambda i, j: (i, j)),
          ],
          out_specs=pl.BlockSpec((8, 128), lambda i, j: (i, j)),
          interpret=pltpu.InterpretParams(
              random_seed=12345,
              num_cores_or_threads=num_cores_per_device,
              grid_point_recorder=grid_point_recorder,
              detect_races=True,
          ),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=('parallel', 'arbitrary')
          ),
      )(s, block_input)

    with self.subTest('num_cores_per_device=1'):
      with GridPointRecorderContext() as grid_point_recorder:
        result = jax.jit(kernel_call, static_argnums=(1, 2))(
            jnp.zeros((1,), jnp.float32), 1, grid_point_recorder.get_recorder()
        )
        np.testing.assert_allclose(
            result[::8, ::128],
            [
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        self.assertListEqual(
            grid_point_recorder.grid_points,
            # parallel_subgrid_size = 4
            # num_parallel_points_per_core = (4 + 1 - 1) // 1 = 4
            # num_iterations_per_core = 4 * (16 // 4) = 16
            [
                ProcessedGridPoint((2, 0), 0),
                ProcessedGridPoint((2, 1), 0),
                ProcessedGridPoint((2, 2), 0),
                ProcessedGridPoint((2, 3), 0),
                ProcessedGridPoint((3, 0), 0),
                ProcessedGridPoint((3, 1), 0),
                ProcessedGridPoint((3, 2), 0),
                ProcessedGridPoint((3, 3), 0),
                ProcessedGridPoint((0, 0), 0),
                ProcessedGridPoint((0, 1), 0),
                ProcessedGridPoint((0, 2), 0),
                ProcessedGridPoint((0, 3), 0),
                ProcessedGridPoint((1, 0), 0),
                ProcessedGridPoint((1, 1), 0),
                ProcessedGridPoint((1, 2), 0),
                ProcessedGridPoint((1, 3), 0),
            ],
        )

    with self.subTest('num_cores_per_device=2'):
      with GridPointRecorderContext() as grid_point_recorder:
        result = jax.jit(kernel_call, static_argnums=(1, 2))(
            jnp.zeros((1,), jnp.float32), 2, grid_point_recorder.get_recorder()
        )
        np.testing.assert_allclose(
            result[::8, ::128],
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        self.assertListEqual(
            grid_point_recorder.grid_points,
            # parallel_subgrid_size = 4
            # num_parallel_points_per_core = (4 + 2 - 1) // 2 = 2
            # num_iterations_per_core = 2 * (16 // 4) = 8
            [
                ProcessedGridPoint((2, 0), 0),
                ProcessedGridPoint((2, 1), 0),
                ProcessedGridPoint((2, 2), 0),
                ProcessedGridPoint((2, 3), 0),
                ProcessedGridPoint((3, 0), 0),
                ProcessedGridPoint((3, 1), 0),
                ProcessedGridPoint((3, 2), 0),
                ProcessedGridPoint((3, 3), 0),
                ProcessedGridPoint((0, 0), 1),
                ProcessedGridPoint((0, 1), 1),
                ProcessedGridPoint((0, 2), 1),
                ProcessedGridPoint((0, 3), 1),
                ProcessedGridPoint((1, 0), 1),
                ProcessedGridPoint((1, 1), 1),
                ProcessedGridPoint((1, 2), 1),
                ProcessedGridPoint((1, 3), 1),
            ],
        )

    with self.subTest('num_cores_per_device=3'):
      with GridPointRecorderContext() as grid_point_recorder:
        result = jax.jit(kernel_call, static_argnums=(1, 2))(
            jnp.zeros((1,), jnp.float32), 3, grid_point_recorder.get_recorder()
        )
        np.testing.assert_allclose(
            result[::8, ::128],
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        self.assertListEqual(
            grid_point_recorder.grid_points,
            # parallel_subgrid_size = 4
            # num_parallel_points_per_core = (4 + 3 - 1) // 3 = 2
            # num_iterations_per_core = 2 * (16 // 4) = 8
            [
                ProcessedGridPoint((2, 0), 0),
                ProcessedGridPoint((2, 1), 0),
                ProcessedGridPoint((2, 2), 0),
                ProcessedGridPoint((2, 3), 0),
                ProcessedGridPoint((3, 0), 0),
                ProcessedGridPoint((3, 1), 0),
                ProcessedGridPoint((3, 2), 0),
                ProcessedGridPoint((3, 3), 0),
                ProcessedGridPoint((0, 0), 1),
                ProcessedGridPoint((0, 1), 1),
                ProcessedGridPoint((0, 2), 1),
                ProcessedGridPoint((0, 3), 1),
                ProcessedGridPoint((1, 0), 1),
                ProcessedGridPoint((1, 1), 1),
                ProcessedGridPoint((1, 2), 1),
                ProcessedGridPoint((1, 3), 1),
            ],
        )

    with self.subTest('num_cores_per_device=4'):
      with GridPointRecorderContext() as grid_point_recorder:
        result = jax.jit(kernel_call, static_argnums=(1, 2))(
            jnp.zeros((1,), jnp.float32), 4, grid_point_recorder.get_recorder()
        )
        np.testing.assert_allclose(
            result[::8, ::128],
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ],
        )
        self.assertListEqual(
            grid_point_recorder.grid_points,
            # parallel_subgrid_size = 4
            # num_parallel_points_per_core = (4 + 4 - 1) // 4 = 1
            # num_iterations_per_core = 1 * (16 // 4) = 4
            [
                ProcessedGridPoint((2, 0), 0),
                ProcessedGridPoint((2, 1), 0),
                ProcessedGridPoint((2, 2), 0),
                ProcessedGridPoint((2, 3), 0),
                ProcessedGridPoint((3, 0), 1),
                ProcessedGridPoint((3, 1), 1),
                ProcessedGridPoint((3, 2), 1),
                ProcessedGridPoint((3, 3), 1),
                ProcessedGridPoint((0, 0), 2),
                ProcessedGridPoint((0, 1), 2),
                ProcessedGridPoint((0, 2), 2),
                ProcessedGridPoint((0, 3), 2),
                ProcessedGridPoint((1, 0), 3),
                ProcessedGridPoint((1, 1), 3),
                ProcessedGridPoint((1, 2), 3),
                ProcessedGridPoint((1, 3), 3),
            ],
        )

    with self.subTest('num_cores_per_device=5'):
      with GridPointRecorderContext() as grid_point_recorder:
        result = jax.jit(kernel_call, static_argnums=(1, 2))(
            jnp.zeros((1,), jnp.float32), 5, grid_point_recorder.get_recorder()
        )
        np.testing.assert_allclose(
            result[::8, ::128],
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ],
        )
        self.assertListEqual(
            grid_point_recorder.grid_points,
            # parallel_subgrid_size = 4
            # num_parallel_points_per_core = (4 + 5 - 1) // 5 = 1
            # num_iterations_per_core = 1 * (16 // 4) = 4
            [
                ProcessedGridPoint((2, 0), 0),
                ProcessedGridPoint((2, 1), 0),
                ProcessedGridPoint((2, 2), 0),
                ProcessedGridPoint((2, 3), 0),
                ProcessedGridPoint((3, 0), 1),
                ProcessedGridPoint((3, 1), 1),
                ProcessedGridPoint((3, 2), 1),
                ProcessedGridPoint((3, 3), 1),
                ProcessedGridPoint((0, 0), 2),
                ProcessedGridPoint((0, 1), 2),
                ProcessedGridPoint((0, 2), 2),
                ProcessedGridPoint((0, 3), 2),
                ProcessedGridPoint((1, 0), 3),
                ProcessedGridPoint((1, 1), 3),
                ProcessedGridPoint((1, 2), 3),
                ProcessedGridPoint((1, 3), 3),
            ],
        )

    with self.subTest('num_cores_per_device=6'):
      with GridPointRecorderContext() as grid_point_recorder:
        result = jax.jit(kernel_call, static_argnums=(1, 2))(
            jnp.zeros((1,), jnp.float32), 6, grid_point_recorder.get_recorder()
        )
        np.testing.assert_allclose(
            result[::8, ::128],
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ],
        )
        self.assertListEqual(
            grid_point_recorder.grid_points,
            # parallel_subgrid_size = 4
            # num_parallel_points_per_core = (4 + 6 - 1) // 6 = 1
            # num_iterations_per_core = 1 * (16 // 4) = 4
            [
                ProcessedGridPoint((2, 0), 0),
                ProcessedGridPoint((2, 1), 0),
                ProcessedGridPoint((2, 2), 0),
                ProcessedGridPoint((2, 3), 0),
                ProcessedGridPoint((3, 0), 1),
                ProcessedGridPoint((3, 1), 1),
                ProcessedGridPoint((3, 2), 1),
                ProcessedGridPoint((3, 3), 1),
                ProcessedGridPoint((0, 0), 2),
                ProcessedGridPoint((0, 1), 2),
                ProcessedGridPoint((0, 2), 2),
                ProcessedGridPoint((0, 3), 2),
                ProcessedGridPoint((1, 0), 3),
                ProcessedGridPoint((1, 1), 3),
                ProcessedGridPoint((1, 2), 3),
                ProcessedGridPoint((1, 3), 3),
            ],
        )

  @parameterized.parameters(pltpu.HBM, pl.ANY)
  def test_referencing_hbm_raises(self, disallowed_memory_space):
    def jax_load_and_store(in_ref, o_ref):
      o_ref[...] = in_ref[...]

    def pallas_load_and_store(in_ref, o_ref):
      t = pltpu.load(in_ref)
      pltpu.store(o_ref, t)

    def kernel_call(kernel, x, *, in_memory_space, out_memory_space):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(1,),
          in_specs=[pl.BlockSpec(memory_space=in_memory_space)],
          out_specs=pl.BlockSpec(memory_space=out_memory_space),
          interpret=pltpu.InterpretParams(),
      )(x)

    with self.assertRaisesRegex(
        ValueError,
        r'get_p: Buffers with a memory space of HBM or ANY cannot be'
        r' referenced directly. Instead, use `pltpu.sync_copy` or'
        r' `pltpu.async_copy`.',
    ):
      kernel_call(
          jax_load_and_store,
          jnp.zeros((8, 128), jnp.float32),
          in_memory_space=disallowed_memory_space,
          out_memory_space=pltpu.VMEM,
      )
    pltpu.reset_tpu_interpret_mode_state()

    with self.assertRaisesRegex(
        ValueError,
        r'load_p: Buffers with a memory space of HBM or ANY cannot be'
        r' referenced directly. Instead, use `pltpu.sync_copy` or'
        r' `pltpu.async_copy`.',
    ):
      kernel_call(
          pallas_load_and_store,
          jnp.zeros((8, 128), jnp.float32),
          in_memory_space=disallowed_memory_space,
          out_memory_space=pltpu.VMEM,
      )
    pltpu.reset_tpu_interpret_mode_state()

    with self.assertRaisesRegex(
        ValueError,
        r'swap_p: Buffers with a memory space of HBM or ANY cannot be'
        r' referenced directly. Instead, use `pltpu.sync_copy` or'
        r' `pltpu.async_copy`.',
    ):
      kernel_call(
          jax_load_and_store,
          jnp.zeros((8, 128), jnp.float32),
          in_memory_space=pltpu.VMEM,
          out_memory_space=disallowed_memory_space,
      )
    pltpu.reset_tpu_interpret_mode_state()

    with self.assertRaisesRegex(
        ValueError,
        r'swap_p: Buffers with a memory space of HBM or ANY cannot be'
        r' referenced directly. Instead, use `pltpu.sync_copy` or'
        r' `pltpu.async_copy`.',
    ):
      kernel_call(
          pallas_load_and_store,
          jnp.zeros((8, 128), jnp.float32),
          in_memory_space=pltpu.VMEM,
          out_memory_space=pltpu.HBM,
      )
    pltpu.reset_tpu_interpret_mode_state()


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
