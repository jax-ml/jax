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
            pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec(
            (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j),
        ),
        interpret=mosaic_interpret.TPUInterpretParams(),
      )(x, y)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k1, (1024, 1024))
    y = jax.random.normal(k2, (1024, 1024))
    z = matmul(x, y)
    np.testing.assert_allclose(z, x @ y, atol=1e-4)

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
          interpret=mosaic_interpret.TPUInterpretParams()
      )(s, x)

    s = jnp.array([1], dtype=jnp.int32)
    x = jnp.arange(32 * 128.).reshape((32, 128))
    y = f(s, x)
    np.testing.assert_allclose(y, x + 1.0)

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
    y = pl.pallas_call(kernel_without_race,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)],
        scratch_shapes=[
            pltpu.VMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ],
        interpret=mosaic_interpret.TPUInterpretParams(
            detect_races=True, dma_execution_mode=dma_execution_mode),
    )(x).block_until_ready()
    self.assertFalse(mosaic_interpret.races.races_found)
    np.testing.assert_allclose(y, x + 1.0)

    pl.pallas_call(kernel_with_race,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)],
        scratch_shapes=[
            pltpu.VMEM(x.shape, x.dtype),
            pltpu.SemaphoreType.DMA,
        ],
        interpret=mosaic_interpret.TPUInterpretParams(
            detect_races=True, dma_execution_mode=dma_execution_mode),
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

    lowered = jax.jit(matmul).lower(x, y).as_text(dialect="stablehlo")
    self.assertNotIn("dot_general", lowered)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
