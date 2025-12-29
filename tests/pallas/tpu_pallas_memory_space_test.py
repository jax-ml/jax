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

"""Test TPU-specific uses of Pallas memory space APIs."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()
P = jax.sharding.PartitionSpec
partial = functools.partial


class TPUPallasCallMemorySpaceTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Needs a newer TPU')

  @parameterized.product(
      memory_space_color=[(pltpu.VMEM, 1),
      (pltpu.SMEM, 4),
      (pltpu.HBM, 0),
      (pl.ANY, None)],
      batched=[True, False],
  )
  def test_basic_input_memory_space_constraint(self, memory_space_color, batched):
    memory_space, color = memory_space_color
    def kernel(x_ref, y_ref):
      pltpu.sync_copy(x_ref, y_ref)

    def g(x):
      return pl.pallas_call(
          kernel,
          out_shape=x,
          in_specs=[pl.BlockSpec(memory_space=memory_space)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
      )(x)

    @jax.jit
    def f(x):
      x = pltpu.with_memory_space_constraint(x, memory_space=memory_space)
      if color is not None:
        self.assertEqual(jax.typeof(x).memory_space, memory_space)
      x = g(x)
      return x

    x = jnp.ones((8, 128), dtype=jnp.float32)
    if batched:
      x = x[None]
      f = jax.vmap(f)
    y = f(x)
    np.testing.assert_array_equal(y, x)
    hlo = jax.jit(f).lower(x).compile().as_text()
    if color is None or memory_space == pltpu.SMEM:
      self.assertIn('"input_memory_space_colors":[]', hlo)
    else:
      self.assertIn(
          f'"input_memory_space_colors":[{{"operand_index":"0","color":"{color}","shape_index":[]}}]',
          hlo,
      )

  @parameterized.parameters(
      (pltpu.VMEM, 1),
      (pltpu.SMEM, 4),
      (pltpu.HBM, 0),
      (pl.ANY, None),
      (pltpu.HOST, 5),
  )
  def test_basic_output_memory_space_constraint(self, memory_space, color):
    out_shape_ctor = memory_space
    if color is None:
      out_shape_ctor = jax.ShapeDtypeStruct

    def kernel(x_ref, y_ref):
      pltpu.sync_copy(x_ref, y_ref)

    def g(x):
      return pl.pallas_call(
          kernel,
          out_shape=out_shape_ctor(x.shape, x.dtype),
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=memory_space),
      )(x)

    if memory_space == pltpu.HOST:
      if jax.device_count() > 1:
        self.skipTest('Test only works with a single device.')
      out_sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(jax.devices(), 'x'),
          jax.sharding.PartitionSpec(),
          memory_kind='pinned_host',
      )
    else:
      out_sharding = None

    @functools.partial(jax.jit, out_shardings=out_sharding)
    def f(x):
      x = g(x)
      return x

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)
    hlo = jax.jit(f, out_shardings=out_sharding).lower(x).compile().as_text()
    if color is None:
      self.assertIn('"output_memory_colors":[]', hlo)
    else:
      self.assertIn(
          f'"output_memory_colors":["{color}"]',
          hlo,
      )


class TPUCoreMapMemorySpaceTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Needs a newer TPU')

  @parameterized.parameters(
      (pltpu.VMEM, 1),
      (pltpu.SMEM, 4),
      (pltpu.HBM, 0),
      (pl.ANY, None),
  )
  def test_basic_ref_memory_space_constraint(self, memory_space, color):
    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x, memory_space=memory_space)
      y_ref = jax.new_ref(pl.empty_like(x), memory_space=memory_space)

      self.assertEqual(jax.typeof(x_ref).memory_space, memory_space)
      self.assertEqual(jax.typeof(y_ref).memory_space, memory_space)

      @pl.core_map(mesh=pltpu.create_tensorcore_mesh('core'))
      def _():
        if jax.typeof(x_ref).memory_space is pltpu.VMEM:
          y_ref[...] = x_ref[...]
        else:
          pltpu.sync_copy(x_ref, y_ref)

      return y_ref[...]

    x = jnp.arange(1024, dtype=jnp.float32).reshape((8, 128))
    num_cores = jax.devices()[0].num_cores
    if num_cores > 1 and memory_space == pltpu.VMEM:
      with self.assertRaisesRegex(
          NotImplementedError,
          'TensorCoreMesh does not support VMEM inputs/outputs when there are'
          ' >1 cores. Use HBM or ANY instead.',
      ):
        f.lower(x).compile()
      return
    lowered = f.lower(x)
    compiled = lowered.compile()
    hlo = compiled.as_text()
    if color is None or memory_space == pltpu.SMEM:
      self.assertIn('"input_memory_space_colors":[]', hlo)
    else:
      self.assertIn(
          f'"input_memory_space_colors":[{{"operand_index":"0","color":"{color}","shape_index":[]}},{{"operand_index":"1","color":"{color}","shape_index":[]}}]',
          hlo,
      )
    y = compiled(x)
    np.testing.assert_array_equal(y, x)

  def test_smem_copy(self):
    mesh = pltpu.create_tensorcore_mesh('core')
    if len(mesh.devices) > 1:
      self.skipTest('Only one core is supported for this test.')

    kernel = pl.core_map(mesh=mesh)

    @jax.jit
    def f():
      y_ref = pl.empty_ref_like(pltpu.SMEM((8,), jnp.int32))

      @kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = i

      @kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = y_ref[i] + 1

      return y_ref[...]

    np.testing.assert_array_equal(f(), np.arange(8) + 1)

  def test_smem_async_copy(self):
    mesh = pltpu.create_tensorcore_mesh('core')
    if len(mesh.devices) > 1:
      self.skipTest('Only one core is supported for this test.')

    kernel = pl.core_map(mesh=mesh)

    @jax.jit
    def f():
      y_ref = pl.empty_ref_like(pltpu.SMEM((8,), jnp.int32))

      @kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = i

      @kernel
      def _():
        for i in range(y_ref.shape[0]):
          y_ref[i] = y_ref[i] + 1

      y_out_ref = pl.empty_ref_like(pltpu.HBM((8,), jnp.int32))

      sem = pl.empty_ref_like(pltpu.SemaphoreType.DMA(()))

      @kernel
      def _():
        pltpu.make_async_copy(y_ref, y_out_ref, sem).start()

      @kernel
      def _():
        pltpu.make_async_copy(y_ref, y_out_ref, sem).wait()

      return y_out_ref[...]

    np.testing.assert_array_equal(f(), np.arange(8) + 1)

  def test_smem_async_copy_megacore(self):
    mesh = pltpu.create_tensorcore_mesh('core')
    num_cores = len(mesh.devices)
    if num_cores == 1:
      self.skipTest('Only megacore is supported for this test.')

    kernel = pl.core_map(mesh=mesh)
    n = 256

    @jax.jit
    def f():
      y_ref = pl.empty_ref_like(pltpu.SMEM((1, n), jnp.int32))

      @kernel
      def _():
        core_i = jax.lax.axis_index('core')
        for i in range(n):
          y_ref[0, i] = i + core_i * n

      @kernel
      def _():
        for i in range(n):
          y_ref[0, i] = y_ref[0, i] + 1

      y_out_ref = pl.empty_ref_like(pltpu.HBM((num_cores, 1, n), jnp.int32))

      sem = pl.empty_ref_like(pltpu.SemaphoreType.DMA(()))

      @kernel
      def _():
        core_i = jax.lax.axis_index('core')
        pltpu.make_async_copy(y_ref, y_out_ref.at[core_i, ...], sem).start()

      @kernel
      def _():
        core_i = jax.lax.axis_index('core')
        pltpu.make_async_copy(y_ref, y_out_ref.at[core_i, ...], sem).wait()

      return y_out_ref[...]

    np.testing.assert_array_equal(
        f(), np.arange(num_cores * n).reshape((num_cores, 1, n)) + 1
    )


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
