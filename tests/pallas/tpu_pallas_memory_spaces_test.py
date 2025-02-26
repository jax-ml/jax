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

"""Test TPU-specific uses of Pallas async APIs."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()
jax.config.update("jax_traceback_filtering", "off")


def add_one(x):
  memory_space = pl.get_memory_space(x)

  def f(x_ref, y_ref):
    y_ref[...] = x_ref[...] + 1

  return pl.pallas_call(
      f,
      in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
      out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
      out_shape=memory_space(x.shape, x.dtype),
  )(x)


def add_one_core_map(x):
  mesh = pltpu.create_tensorcore_mesh("x")

  y = pl.empty_like(x, memory_space=pl.get_memory_space(x))

  @pl.run_state
  def f(refs):
    x_ref, y_ref = refs

    x_memory_space = pl.get_memory_space(x_ref)
    y_memory_space = pl.get_memory_space(y_ref)

    @pl.core_map(mesh)
    def _():
      def kernel(vmem_ref):
        if x_memory_space == pltpu.VMEM:
          vmem_ref[...] = x_ref[...]
        else:
          pltpu.sync_copy(x_ref, vmem_ref)

        vmem_ref[...] += 1

        if y_memory_space == pltpu.VMEM:
          y_ref[...] = vmem_ref[...]
        else:
          pltpu.sync_copy(vmem_ref, y_ref)

      pl.run_scoped(kernel, pltpu.VMEM(x.shape, x.dtype))

  _, y = f((x, y))
  return y


class PallasCallMemorySpacesTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.if_cloud_tpu_at_least(2025, 3, 22):
      self.skipTest("Needs a newer libTPU")
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("DMAs not supported on TPU generations <= 3")

  @parameterized.parameters([pltpu.HBM, pltpu.VMEM])
  def test_can_get_memory_space_at_trace_time(self, memory_space):
    def f(x):
      x = pl.with_memory_constraint(x, memory_space)
      self.assertEqual(pl.get_memory_space(x), memory_space)
      return x

    _ = jax.jit(f).lower(jax.ShapeDtypeStruct((1024, 1024), jnp.float32))

  def test_can_query_memory_space_in_pallas_call(self):
    x = jax.random.normal(jax.random.key(0), (1024, 1024), jnp.float32)
    y = jax.jit(add_one)(x)
    np.testing.assert_allclose(y, x + 1)

  @parameterized.parameters([pltpu.HBM, pltpu.VMEM])
  def test_constrain_memory_space(self, memory_space):
    if memory_space == pltpu.VMEM and jtu.is_device_tpu(4, " lite"):
      self.skipTest("Cannot output VMEM on TPU v4 lite.")
    num_cores = jax.devices()[0].num_cores
    if num_cores > 1 and memory_space == pltpu.VMEM:
      self.skipTest("Cannot output VMEM when there are >1 cores.")
    x = jax.random.normal(jax.random.key(0), (1024, 1024), jnp.float32)

    def f(x):
      x = pl.with_memory_constraint(x, memory_space)
      self.assertEqual(pl.get_memory_space(x), memory_space)
      return add_one(x)

    y = jax.jit(f)(x)
    np.testing.assert_allclose(y, x + 1)


class CoreMapMemorySpacesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.if_cloud_tpu_at_least(2025, 3, 22):
      self.skipTest("Needs a newer libTPU")
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("DMAs not supported on TPU generations <= 3")

  @parameterized.parameters([pltpu.HBM, pltpu.VMEM])
  def test_can_get_memory_space_of_refs_at_trace_time(self, memory_space):
    def f(x):
      x = pl.with_memory_constraint(x, memory_space)

      @pl.run_state
      def body(x_ref):
        self.assertEqual(pl.get_memory_space(x_ref), memory_space)

      return body(x)

    _ = jax.jit(f).lower(jax.ShapeDtypeStruct((1024, 1024), jnp.float32))

  def test_can_query_memory_space_in_pallas_call(self):
    x = jax.random.normal(jax.random.key(0), (1024, 1024), jnp.float32)
    y = jax.jit(add_one_core_map)(x)
    np.testing.assert_allclose(y, x + 1)

  @parameterized.parameters([pltpu.HBM, pltpu.VMEM])
  def test_constrain_memory_space(self, memory_space):
    if memory_space == pltpu.VMEM and jtu.is_device_tpu(4, " lite"):
      self.skipTest("Cannot output VMEM on TPU v4 lite.")
    x = jax.random.normal(jax.random.key(0), (1024, 1024), jnp.float32)

    def f(x):
      x = pl.with_memory_constraint(x, memory_space)
      self.assertEqual(pl.get_memory_space(x), memory_space)
      return add_one_core_map(x)

    num_cores = jax.devices()[0].num_cores
    if num_cores > 1 and memory_space == pltpu.VMEM:
      with self.assertRaisesRegex(
          NotImplementedError,
          "TensorCoreMesh does not support VMEM inputs/outputs when there are"
          " >1 cores. Use HBM or ANY instead.",
      ):
        _ = jax.jit(f)(x)
    else:
      y = jax.jit(f)(x)
      np.testing.assert_allclose(y, x + 1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
