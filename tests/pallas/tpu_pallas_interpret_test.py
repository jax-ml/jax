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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax._src.pallas.mosaic.interpret as mosaic_interpret
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class CountStoreCallbacksContext(object):
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


class InterpretTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      # Workaround for https://github.com/jax-ml/jax/issues/25671
      self.skipTest(f'requires 1 device, found {self.num_devices}')

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
          interpret=mosaic_interpret.TPUInterpretParams(),
      )(x, y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = matmul(x, y)
    np.testing.assert_allclose(z, x @ y, atol=1e-4)

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
          interpret=mosaic_interpret.TPUInterpretParams(),
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
    diff = jnp.max(jnp.abs(result - ref))
    np.testing.assert_allclose(result, ref)

  def test_dynamic_grid_and_aliasing(self):
    def kernel(s_ref, x_ref, o_ref):
      o_ref[...] = x_ref[...] + s_ref[0].astype(x_ref.dtype)

    iters = jax.random.randint(jax.random.key(0), (), 10, 20, dtype=jnp.int32)

    @jax.jit
    def f(s, x):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(iters,),
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM),
              pl.BlockSpec(x.shape, lambda i: (0, 0)),
          ],
          out_specs=pl.BlockSpec(x.shape, lambda i: (0, 0)),
          input_output_aliases={1: 0},
          interpret=mosaic_interpret.TPUInterpretParams(),
      )(s, x)

    s = jnp.array([1], dtype=jnp.int32)
    x = jnp.arange(32 * 128.0).reshape((32, 128))
    y = f(s, x)
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
        interpret=mosaic_interpret.TPUInterpretParams(),
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
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)],
        scratch_shapes=[
            pltpu.VMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ],
        interpret=mosaic_interpret.TPUInterpretParams(
            detect_races=True, dma_execution_mode=dma_execution_mode
        ),
    )(x).block_until_ready()
    self.assertFalse(mosaic_interpret.races.races_found)
    np.testing.assert_allclose(y, x + 1.0)

    pl.pallas_call(
        kernel_with_race,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)],
        scratch_shapes=[
            pltpu.VMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ],
        interpret=mosaic_interpret.TPUInterpretParams(
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
          interpret=mosaic_interpret.TPUInterpretParams(
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
        interpret=mosaic_interpret.TPUInterpretParams(
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
          interpret=mosaic_interpret.TPUInterpretParams(),
      )(x, s)

    with CountStoreCallbacksContext() as store_callbacks_counter:
      result = jax.jit(kernel_call)(
          jnp.zeros((16, 256), jnp.float32), jnp.zeros((1,), jnp.int32)
      )
      np.testing.assert_allclose(result[::8, ::256], [[1.0], [5.0]])
      self.assertEqual(store_callbacks_counter.num_stores, 5)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
