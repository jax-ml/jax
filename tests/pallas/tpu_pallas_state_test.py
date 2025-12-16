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
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.state.primitives import pin, unpin
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
          in_specs=[pl.BlockSpec(memory_space=pltpu.ANY)],
          out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
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
    if jtu.is_device_tpu_at_least(6):
      atol = 2e-5
    np.testing.assert_allclose(o, x @ y, atol=atol)


class PinnedBufferTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supported on TPU v4+")

  def test_basic(self):

    @jax.jit
    def f(x):
      x_pinned = pin(x)
      x_pinned = pl.pallas_call(
          lambda *_: None, out_shape=x_pinned,
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          input_output_aliases={0: 0}
          )(x_pinned)
      return unpin(x_pinned)

    x = jnp.arange(3.)
    y = f(x)
    self.assertAllClose(y, x, check_dtypes=False)

  def test_error_if_not_aliased(self):

    @jax.jit
    def f(x):
      x_pinned = pin(x)
      x_pinned = pl.pallas_call(
          lambda *_: None, out_shape=x_pinned,
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          # input_output_aliases={0: 0}  # no aliasing!
          )(x_pinned)
      return unpin(x_pinned)

    x = jnp.arange(3.)
    with self.assertRaisesRegex(ValueError, r"pinned buffers without"):
      f(x)


class CoreMapTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supported on TPU v4+")

  def test_can_create_tensorcore_mesh(self):
    _ = pltpu.create_tensorcore_mesh("x")

  def test_kernel_helper_basic(self):
    mesh = pltpu.create_tensorcore_mesh("x")
    def body(x_ref, o_ref):
      pltpu.sync_copy(x_ref, o_ref)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    with self.subTest("decorator"):
      result = pl.kernel(body, out_shape=x, mesh=mesh)(x)
      np.testing.assert_array_equal(result, x)
    with self.subTest("decorator_factory"):
      result = pl.kernel(out_shape=x, mesh=mesh)(body)(x)
      np.testing.assert_array_equal(result, x)

  def test_empty_core_map_raises_error(self):
    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def inner(refs):
        del refs  # Unused.
        @pl.core_map(pltpu.create_tensorcore_mesh("x"))
        def _():
          pass
      _, y = pl.run_state(inner)((x, y))
      return y
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    with self.assertRaisesRegex(Exception,
      "Attempted to lower core_map without discharging."):
      f(x)

  def test_can_signal_cores(self):
    @jax.jit
    def f(x):
      x_ref = jax.new_ref(x)
      y_ref = jax.new_ref(jnp.empty_like(x))
      @pl.core_map(pltpu.create_tensorcore_mesh("x"))
      def _():
        @functools.partial(pl.run_scoped, sem=pltpu.SemaphoreType.REGULAR)
        def inner(sem):
          s = jax.lax.axis_size("x")
          for i in range(s):
            pl.semaphore_signal(sem, device_id={"x": i})
          pl.semaphore_wait(sem, s)
          pltpu.sync_copy(x_ref, y_ref)
      return jax.freeze(y_ref)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    np.testing.assert_array_equal(f(x), x)

  def test_can_query_core_index(self):
    mesh = pltpu.create_tensorcore_mesh("x")
    slc_size = 16 // mesh.shape["x"]

    @jax.jit
    def f(x):
      @pl.kernel(
          out_shape=x,
          mesh=mesh,
          scratch_shapes=[
              pltpu.VMEM((slc_size, 128), x.dtype),
              pltpu.VMEM((slc_size, 128), x.dtype),
              pltpu.SemaphoreType.DMA,
          ],
      )
      def kernel(x_ref, y_ref, x_vmem_ref, y_vmem_ref, sem):
        num_cores = jax.lax.axis_size("x")
        slc_size = 16 // num_cores
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
      return kernel(x)
    num_cores = jax.devices()[0].num_cores
    x = jnp.arange(16 * 128, dtype=jnp.int32).reshape((16, 128))
    expected_out = (
        x.reshape((num_cores, -1, 128)) + jnp.arange(num_cores)[..., None, None]
    ).reshape(x.shape)
    y = f(x)
    np.testing.assert_array_equal(y, expected_out)

  def test_raises_on_captured_arrays(self):
    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)

      @pl.kernel(out_shape=x,
                 mesh=pltpu.create_tensorcore_mesh("x"),
                 scratch_shapes=dict(tmp_ref=pltpu.VMEM(x.shape, x.dtype)))
      def kernel(x_ref, out_ref, tmp_ref):
        pltpu.sync_copy(x_ref, tmp_ref)
        tmp_ref[...] += y
        out_ref[...] = tmp_ref[...]
      return kernel(x)

    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    with self.assertRaisesRegex(
        Exception, "core_map .* captures non-scalar constants"
    ):
      f(x)

  def test_capture_scalar(self):
    @jax.jit
    def f(x, i):
      @pl.kernel(out_shape=jax.ShapeDtypeStruct(x.shape[1:], jnp.int32),
                 mesh=pltpu.create_tensorcore_mesh("x", num_cores=1))
      def kernel(x_ref, out_ref):
        pltpu.sync_copy(x_ref.at[i], out_ref)
      return kernel(x)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    for i in range(x.shape[0]):
      out = f(x, i)
      np.testing.assert_array_equal(out, x[i])

    @jax.jit
    def g(x, i):
      @pl.kernel(out_shape=jax.ShapeDtypeStruct((2, *x.shape[1:]), jnp.int32),
                 mesh=pltpu.create_tensorcore_mesh("x", num_cores=1))
      def kernel(x_ref, out_ref):
        pltpu.sync_copy(x_ref.at[pl.ds(i, 2)], out_ref)
      return kernel(x)

    x = jnp.arange(4 * 8 * 128, dtype=jnp.int32).reshape((4, 8, 128))
    for i in range(3):
      out = g(x, i)
      np.testing.assert_array_equal(out, x[i:i+2])

  def test_kernel_helper_with_scratch(self):
    mesh = pltpu.create_tensorcore_mesh("x")
    def body(x_ref, o_ref, scratch_ref):
      pltpu.sync_copy(x_ref, scratch_ref)
      scratch_ref[...] += 1
      pltpu.sync_copy(scratch_ref, o_ref)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    result = pl.kernel(
        body, out_shape=x, mesh=mesh,
        scratch_shapes=dict(scratch_ref=pltpu.VMEM(x.shape, x.dtype)))(x)
    np.testing.assert_array_equal(result, x + 1)

  def test_kernel_helper_with_out_tree(self):
    mesh = pltpu.create_tensorcore_mesh("x")
    def body(x_ref, o1_ref, o2_ref, scratch_ref):
      pltpu.sync_copy(x_ref, o1_ref)
      pltpu.sync_copy(x_ref, scratch_ref)
      scratch_ref[...] += 1
      pltpu.sync_copy(scratch_ref, o2_ref)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    result1, result2 = pl.kernel(
        body, out_shape=[x, x], mesh=mesh,
        scratch_shapes=[pltpu.VMEM(x.shape, x.dtype)])(x)
    np.testing.assert_array_equal(result1, x)
    np.testing.assert_array_equal(result2, x + 1)

  @parameterized.named_parameters(
      ("HBM", pltpu.HBM, 0),
      ("VMEM", pltpu.VMEM, 1),
      ("SMEM", pltpu.SMEM, 4),
      ("SEMAPHORE", pltpu.SEMAPHORE, 2),
  )
  def test_kernel_with_output_memory_space(self, memory_space, color):
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest("Only supported on TPU v5+")
    mesh = pltpu.create_tensorcore_mesh("x", num_cores=1)
    def body(x_ref, o_ref):
      pltpu.sync_copy(x_ref, o_ref)
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
    text = pl.kernel(
        body, out_shape=memory_space(x.shape, x.dtype), mesh=mesh,
    ).lower(x).as_text()
    custom_call = [l for l in text.split("\n") if "@tpu_custom_call" in l]
    self.assertLen(custom_call, 1)
    custom_call = custom_call[0]
    self.assertRegex(custom_call,
                     r".*output_memory_colors\\22: \[" + str(color) + r"\].*")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
