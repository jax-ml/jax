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


class TPUPallasMemorySpaceTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.if_cloud_tpu_at_least(2025, 6, 10):
      self.skipTest('Needs a newer libTPU')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Needs a newer TPU')

  @parameterized.parameters(
      (pltpu.VMEM, 1),
      (pltpu.HBM, 0),
      (pltpu.ANY, None),
  )
  def test_basic_input_memory_space_constraint(self, memory_space, color):

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...]

    def g(x):
      return pl.pallas_call(kernel, out_shape=x)(x)

    @jax.jit
    def f(x):
      x = pltpu.with_memory_space_constraint(x, memory_space=memory_space)
      if color is None:
        self.assertIsNone(pltpu.get_memory_space(x))
      else:
        self.assertEqual(pltpu.get_memory_space(x), memory_space)
      x = g(x)
      return x

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)
    hlo = jax.jit(f).lower(x).compile().as_text()
    if color is None:
      self.assertIn('"input_memory_space_colors":[]', hlo)
    else:
      self.assertIn(
          f'"input_memory_space_colors":[{{"operand_index":"0","color":"{color}","shape_index":[]}}]',
          hlo,
      )

  @parameterized.parameters(
      (pltpu.VMEM, 1),
      (pltpu.HBM, 0),
      (pltpu.ANY, None),
  )
  def test_basic_output_memory_space_constraint(self, memory_space, color):
    if color is None:
      memory_space = jax.ShapeDtypeStruct

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...]

    def g(x):
      return pl.pallas_call(kernel, out_shape=memory_space(x.shape, x.dtype))(x)

    @jax.jit
    def f(x):
      x = g(x)
      return x

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)
    hlo = jax.jit(f).lower(x).compile().as_text()
    if color is None:
      self.assertIn('"output_memory_space_colors":[]', hlo)
    else:
      self.assertIn(
          f'"output_memory_space_colors":[{{"color":"{color}","shape_index":[]}}]',
          hlo,
      )


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
