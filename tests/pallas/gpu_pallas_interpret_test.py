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
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


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

  @jtu.parameterized.parameters(1, 2, 4, 8, 16)
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

  @jtu.parameterized.parameters(1, 2, 4, 8, 16)
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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
