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
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class PallasCallStatefulTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supported on TPU v4+")

  def test_basic_stateful_kernel(self):

    def copy_kernel(x_ref, y_ref):
      def body(sem):
        pltpu.make_async_copy(x_ref, y_ref, sem).start()
        pltpu.make_async_copy(x_ref, y_ref, sem).wait()
      pl.run_scoped(body, pltpu.SemaphoreType.DMA)

    def f_stateful(refs):
      x_ref, y_ref = refs

      pl.pallas_call(functools.partial(copy_kernel, x_ref, y_ref),
                     out_shape=[])()

    @jax.jit
    def f(x):
      _, y = pl.run_state(f_stateful)((x, jnp.zeros_like(x)))
      return y

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_basic_stateful_kernel_with_scratch_sem(self):

    def copy_kernel(x_ref, y_ref, sem):
      pltpu.make_async_copy(x_ref, y_ref, sem).start()
      pltpu.make_async_copy(x_ref, y_ref, sem).wait()

    def f_stateful(refs):
      x_ref, y_ref = refs

      pl.pallas_call(functools.partial(copy_kernel, x_ref, y_ref),
                     scratch_shapes=[pltpu.SemaphoreType.DMA],
                     out_shape=[])()

    @jax.jit
    def f(x):
      _, y = pl.run_state(f_stateful)((x, jnp.zeros_like(x)))
      return y

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_basic_stateful_kernel_with_scalar_prefetch(self):

    def copy_kernel(x_ref, y_ref, index_ref, sem):
      i = index_ref[0]
      pltpu.make_async_copy(x_ref.at[i], y_ref, sem).start()
      pltpu.make_async_copy(x_ref.at[i], y_ref, sem).wait()

    def f_stateful(refs):
      x_ref, y_ref = refs

      pl.pallas_call(
          functools.partial(copy_kernel, x_ref, y_ref),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              scratch_shapes=[pltpu.SemaphoreType.DMA],
          ),
          out_shape=[],
      )(jnp.array([0]))

    @jax.jit
    def f(x):
      _, y = pl.run_state(f_stateful)((x[None], jnp.zeros_like(x)))
      return y

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_basic_stateful_kernel_with_io_aliasing(self):

    def copy_kernel(x_ref, y_ref, x_old_ref, x_old_ref2, sem):
      del x_old_ref, x_old_ref2
      pltpu.make_async_copy(x_ref, y_ref, sem).start()
      pltpu.make_async_copy(x_ref, y_ref, sem).wait()

    def f_stateful(refs):
      x_ref, y_ref, o_ref = refs

      x = pl.pallas_call(
          functools.partial(copy_kernel, x_ref, y_ref),
          scratch_shapes=[pltpu.SemaphoreType.DMA],
          out_shape=jax.ShapeDtypeStruct(x_ref.shape, x_ref.dtype),
          input_output_aliases={0: 0},
      )(x_ref[...])
      o_ref[...] = x

    @jax.jit
    def f(x):
      _, y, o = pl.run_state(f_stateful)(
          (x, jnp.zeros_like(x), jnp.zeros_like(x))
      )
      return y, o

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    y, o = f(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(o, x)

  def test_stateful_matmul(self):

    m, k, n = 512, 512, 512
    bm, bk, bn = 128, 128, 128

    def matmul_kernel(acc_ref, x_ref, y_ref, o_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

      acc_ref[...] += jnp.dot(
          x_ref[...], y_ref[...], preferred_element_type=jnp.float32
      )

      @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
      def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)

    def matmul(x, y):

      def run_matmul(refs):
        x_ref, y_ref, o_ref = refs

        def matmul_pipeline_kernel(acc_ref):
          pltpu.emit_pipeline(
              functools.partial(matmul_kernel, acc_ref),
              grid=(m // bm, n // bn, k // bk),
              in_specs=[
                  pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                  pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
              ],
              out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          )(x_ref, y_ref, o_ref)

        pl.pallas_call(
            matmul_pipeline_kernel,
            out_shape=[],
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        )()

      _, _, o = pl.run_state(run_matmul)(
          (x, y, jnp.ones((m, n), dtype=x.dtype))
      )
      return o

    x = jax.random.normal(jax.random.key(0), (m, k), jnp.float32)
    y = jax.random.normal(jax.random.key(1), (k, n), jnp.float32)
    o = matmul(x, y)
    atol = 0
    if jtu.is_device_tpu(6):
      atol = 2e-5
    np.testing.assert_allclose(o, x @ y, atol=atol)


class ShmallasTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supported on TPU v4+")

  def test_can_create_tensorcore_mesh(self):
    _ = pltpu.create_tensorcore_mesh("x")

  def test_can_run_basic_pallas_kernel_with_core_map(self):
    mesh = pltpu.create_tensorcore_mesh("x")

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        @pl.core_map(mesh)
        def _():
          def alloc(sem):
            pltpu.async_copy(x_ref, y_ref, sem).wait()
          pl.run_scoped(alloc, pltpu.SemaphoreType.DMA)
      _, y = pl.run_state(inner)((x, y))
      return y
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_can_query_core_index_pallas_kernel_with_core_map(self):
    mesh = pltpu.create_tensorcore_mesh("x")

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        x_ref, y_ref = refs
        @pl.core_map(mesh)
        def _():
          num_cores = jax.lax.psum(1, "x")
          slc_size = 16 // num_cores
          def alloc(x_vmem_ref, y_vmem_ref, sem):
            core_index = jax.lax.axis_index("x")
            slc = pl.ds(core_index * slc_size, slc_size)
            pltpu.async_copy(
                x_ref.at[slc],
                x_vmem_ref,
                sem,
            ).wait()
            y = x_vmem_ref[...] + jax.lax.axis_index("x")
            y_vmem_ref[...] = y
            pltpu.async_copy(y_vmem_ref, y_ref.at[slc], sem).wait()
          pl.run_scoped(
              alloc,
              pltpu.VMEM((slc_size, 128), x_ref.dtype),
              pltpu.VMEM((slc_size, 128), y_ref.dtype),
              pltpu.SemaphoreType.DMA,
          )
      _, y = pl.run_state(inner)((x, y))
      return y
    num_cores = jax.devices()[0].num_cores
    x = jnp.arange(16 * 128, dtype=jnp.int32).reshape((16, 128))
    expected_out = (
        x.reshape((num_cores, -1, 128)) + jnp.arange(num_cores)[..., None, None]
    ).reshape(x.shape)
    y = f(x)
    np.testing.assert_array_equal(y, expected_out)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
