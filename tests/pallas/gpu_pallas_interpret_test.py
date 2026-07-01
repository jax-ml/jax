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

import dataclasses
import functools
import math
from typing import Any
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams as InterpretParams
from jax._src.pallas.mosaic_gpu.interpret.params import force_gpu_interpret_mode
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu
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
    mosaic_interpret.gpu_callbacks.reset_gpu_interpret_mode_state()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  def test_hopper_matmul(self):
    (m, n, k) = 512, 512, 512
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    a = jax.random.uniform(k1, (m, k), jnp.float16)
    b = jax.random.uniform(k2, (k, n), jnp.float16)
    c = jax.random.uniform(k3, (m, n), jnp.float32)

    spec = hopper_matmul_mgpu.TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        max_concurrent_steps=2,
        epi_tile_m=64,
        epi_tile_n=64,
        wg_dimension=hopper_matmul_mgpu.MatmulDimension.M,
    )

    device = jax.sharding.AbstractDevice(
        device_kind='NVIDIA H100 80GB HBM3',
        platform='gpu',
        num_cores=8,
    )
    with (jax.sharding.use_abstract_mesh(
              jax.sharding.AbstractMesh((), (), abstract_device=device)),
          force_gpu_interpret_mode(InterpretParams())):
      res = hopper_matmul_mgpu.matmul(a, b, c, spec).block_until_ready()
    expected = (
        jnp.dot(a, b, preferred_element_type=jnp.float32) + c)
    np.testing.assert_allclose(res, expected, rtol=5e-3)

  def test_interpret_pallas_call(self):
    def _kernel(o_ref):
      o_ref[0] = 42

    @jax.jit
    def kernel():
      return pl.pallas_call(
          _kernel,
          out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
          interpret=InterpretParams(detect_races=True),
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
          interpret=InterpretParams(detect_races=True),
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
          interpret=InterpretParams(detect_races=True),
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
        out_type=jax.ShapeDtypeStruct((num_threads,), jnp.int32),
        num_threads=num_threads,
        thread_name='x',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(o_ref):
      thread_idx = jax.lax.axis_index('x')
      o_ref[thread_idx] = thread_idx

    np.testing.assert_equal(jax.jit(_kernel)(), np.arange(num_threads))
    self.assertFalse(mosaic_interpret.get_races().races_found)

  def test_tiling_and_swizzle_transforms(self):

    @jax.jit
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((128, 128), jnp.float16),
        scratch_types=dict(
            smem_ref1=plgpu.SMEM(
                (2, 128, 128),
                jnp.float16,
                transforms=(
                    plgpu.TilingTransform((8, 64)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
            smem_ref2=plgpu.SMEM(
                (4, 128, 32),
                jnp.float16,
                transforms=(plgpu.SwizzleTransform(64),),
            ),
            smem_ref3=plgpu.SMEM(
                (256, 128),
                jnp.float16,
                transforms=(plgpu.TilingTransform((16, 32)),),
            ),
        ),
        interpret=InterpretParams(),
    )
    def kernel(o_ref, smem_ref1, smem_ref2, smem_ref3):
      smem_ref1[...] = jnp.full_like(smem_ref1, 42.0)
      smem_ref2[...] = jnp.full_like(smem_ref2, 1.0)
      smem_ref3[...] = jnp.full_like(smem_ref3, 2.0)
      o_ref[...] = (smem_ref1[0, ...] + smem_ref2[...].reshape((128, 128))
                    + smem_ref3[:128])

    np.testing.assert_equal(kernel(), np.full((128, 128), 45.0, jnp.float16))

  def test_tiling_and_swizzle_transforms_with_pallas_call(self):
    def _kernel(o_ref, smem_ref1, smem_ref2, smem_ref3):
      smem_ref1[...] = jnp.full_like(smem_ref1, 42.0)
      smem_ref2[...] = jnp.full_like(smem_ref2, 1.0)
      smem_ref3[...] = jnp.full_like(smem_ref3, 2.0)
      o_ref[...] = (smem_ref1[0, ...] + smem_ref2[...].reshape((128, 128))
                    + smem_ref3[:128])

    @jax.jit
    def run():
      return pl.pallas_call(
          _kernel,
          out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float16),
          out_specs=plgpu.BlockSpec((128, 128), memory_space=plgpu.SMEM),
          scratch_shapes=dict(
              smem_ref1=plgpu.SMEM((2, 128, 128), jnp.float16, transforms=(
                  plgpu.TilingTransform((8, 64)),
                  plgpu.SwizzleTransform(128),
              )),
              smem_ref2=plgpu.SMEM((4, 128, 32), jnp.float16, transforms=(
                  plgpu.SwizzleTransform(64),
              )),
              smem_ref3=plgpu.SMEM((256, 128), jnp.float16, transforms=(
                  plgpu.TilingTransform((16, 32)),
              )),
          ),
          interpret=InterpretParams(),
      )()

    np.testing.assert_equal(run(), np.full((128, 128), 45.0, jnp.float16))

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
          interpret=InterpretParams(
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
          out_type=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
          num_threads=num_threads,
          thread_name='t',
          interpret=InterpretParams(
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

  def test_run_scoped(self):
    mesh = plgpu.Mesh(num_threads=2, thread_name='n')

    @jax.jit
    def f(x):
      def inner(o_ref):

        @pl.core_map(
            mesh,
            interpret=InterpretParams(
                detect_races=True,
            ),
        )
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
              collective_axes=('n',),
          )

      y = pl.run_state(inner)(x)
      return y

    _ = f(jnp.zeros((2, 16, 128)))
    self.assertTrue(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.parameters(
      ((),),
      (('t',),),
      (('c0',),),
      (('c1',),),
      (('c0', 't'),),
      (('c1', 't'),),
      (('c0', 'c1'),),
      (('c0', 'c1', 't'),),
  )
  def test_run_scoped_with_cluster(self, collective_axes):
    all_axis_names = ('c0', 'c1', 't')
    non_collective_axis_names = tuple(
        name for name in all_axis_names if name not in collective_axes
    )

    mesh = plgpu.Mesh(
        cluster=(2, 2),
        cluster_names=('c0', 'c1'),
        num_threads=2,
        thread_name='t',
    )

    @jax.jit
    def f(x):
      def inner(o_ref):

        @pl.core_map(
            mesh,
            interpret=InterpretParams(
                detect_races=True,
            ),
        )  # type: ignore[wrong-arg-types]
        def _():
          def body(ref):
            collective_indices = tuple(
                jax.lax.axis_index(axis_name) for axis_name in collective_axes
            )
            non_collective_indices = tuple(
                jax.lax.axis_index(axis_name)
                for axis_name in non_collective_axis_names
            )

            ref[collective_indices] = sum(
                stride * index
                for stride, index in zip(
                    (1, 2, 4), reversed(collective_indices)
                )
            )

            o_ref[non_collective_indices + collective_indices] = ref[
                collective_indices
            ]

          pl.run_scoped(
              body,
              plgpu.GMEM((2,) * len(collective_axes), dtype=jnp.int32),
              collective_axes=collective_axes,
          )

      y = pl.run_state(inner)(x)
      return y

    if 'c0' in collective_axes or 'c1' in collective_axes:
      with self.assertRaisesRegex(
          Exception,
          r'Collective allocations along cluster axes are not' r' supported\.',
      ):
        _ = f(jnp.zeros((2, 2, 2), dtype=jnp.int32))
    elif 't' not in collective_axes:
      with self.assertRaisesRegex(
          Exception,
          r'Scoped allocation must have the thread axis in its collective'
          r' axes\.',
      ):
        _ = f(jnp.zeros((2, 2, 2), dtype=jnp.int32))
    else:
      y = f(jnp.zeros((2, 2, 2), dtype=jnp.int32))
      self.assertFalse(mosaic_interpret.get_races().races_found)
      expected = np.arange(2 ** len(collective_axes)).reshape(
          (1,) * len(non_collective_axis_names) + (2,) * len(collective_axes)
      )
      expected = np.broadcast_to(expected, (2, 2, 2))
      np.testing.assert_array_equal(y, expected)

  def test_run_scoped_with_unknown_collective_axis(self):
    mesh = plgpu.Mesh(
        cluster=(2, 2),
        cluster_names=('c0', 'c1'),
        num_threads=2,
        thread_name='t',
    )

    @jax.jit
    def f(x):
      def inner(o_ref):

        @pl.core_map(
            mesh,
            interpret=InterpretParams(),
        )
        def _():
          def body(_):
            o_ref[...] = 42

          pl.run_scoped(
              body,
              plgpu.GMEM((), dtype=jnp.int32),
              collective_axes=('unknown_axis',),
          )
      y = pl.run_state(inner)(x)
      return y

    with self.assertRaisesRegex(
        Exception,
        r"Collective axis `unknown_axis` not found among axes `\['c0', 'c1', 't'\]`",
    ):
      _ = f(jnp.zeros((), dtype=jnp.int32))

  @jtu.parameterized.parameters(
      (
          True,
          ('c0',),
          r'Collective allocations along cluster axes are not'
          r' supported\.',
      ),
      (
          True,
          ('c0', 't'),
          r'Collective allocations along cluster axes are not'
          r' supported\.',
      ),
      (
          True,
          (),
          (
              r'Scoped allocation must have the thread axis in its collective'
              r' axes\.'
          ),
      ),
      (
          False,
          ('c0',),
          r'Requesting collective allocations, but no explicit thread axis'
          r' specified\.',
      ),
      (False, ('t',), r'Collective axis `t` not found among axes'),
  )
  def test_run_scoped_barrier_with_incorrect_collective_axes(
      self, has_thread_axis, collective_axes, expected_error_regex
  ):
    mesh_kwargs = dict(
        cluster=(2, 2),
        cluster_names=('c0', 'c1'),
    )
    if has_thread_axis:
      mesh_kwargs.update(
          num_threads=2,
          thread_name='t',
      )
    mesh = plgpu.Mesh(**mesh_kwargs)

    @jax.jit
    def f(x):
      def inner(o_ref):
        @pl.core_map(
            mesh,
            interpret=InterpretParams(),
        )
        def _():
          def body(_):
            o_ref[...] = 42

          pl.run_scoped(
              body,
              plgpu.Barrier(),
              collective_axes=collective_axes,
          )
      y = pl.run_state(inner)(x)
      return y

    with self.assertRaisesRegex(Exception, expected_error_regex):
      _ = f(jnp.zeros((), dtype=jnp.int32))

  # Test adapted from
  # https://docs.jax.dev/en/latest/pallas/gpu/reference.html#using-multiple-pallas-threads-per-cuda-block
  def test_producer_consumer_threads_with_barrier(self):
    x = jnp.arange(128, dtype=jnp.float32)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(x.shape, x.dtype),
            barrier_ref=plgpu.Barrier(),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
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

  @jtu.parameterized.product(with_race=[True, False])
  def test_barrier_multidimensional_1d(self, with_race):
    shape = (2,)
    x = jnp.arange(2, dtype=jnp.float32)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(shape, x.dtype),
            barrier=plgpu.Barrier(num_arrivals=1, num_barriers=shape),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier):
      thread_id = jax.lax.axis_index('t')
      for i in range(2):
        @pl.when(thread_id == 0)
        def _():
          smem_ref[i] = x_ref[i] + 1
          plgpu.barrier_arrive(barrier.at[i])

        @pl.when(thread_id == 1)
        def _():
          if not with_race:
            plgpu.barrier_wait(barrier.at[i])
          out_ref[i] = smem_ref[i] + 1

    y = _kernel(x)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      np.testing.assert_array_equal(y, x + 2)

  @jtu.parameterized.product(with_race=[True, False])
  def test_barrier_multidimensional_2d(self, with_race):
    shape = (2, 3)
    x = jnp.arange(6, dtype=jnp.float32).reshape(shape)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(shape, x.dtype),
            barrier=plgpu.Barrier(num_arrivals=1, num_barriers=shape),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier):
      thread_id = jax.lax.axis_index('t')
      for i in range(2):
        for j in range(3):
          @pl.when(thread_id == 0)
          def _():
            smem_ref[i, j] = x_ref[i, j] + 1
            plgpu.barrier_arrive(barrier.at[i, j])

          @pl.when(thread_id == 1)
          def _():
            if not with_race:
              plgpu.barrier_wait(barrier.at[i, j])
            out_ref[i, j] = smem_ref[i, j] + 1

    y = _kernel(x)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      np.testing.assert_array_equal(y, x + 2)

  @jtu.parameterized.product(with_race=[True, False])
  def test_barrier_multidimensional_3d(self, with_race):
    shape = (2, 1, 3)
    x = jnp.arange(6, dtype=jnp.float32).reshape(shape)

    @functools.partial(
        plgpu.kernel,
        out_type=x,
        scratch_types=dict(
            smem_ref=plgpu.SMEM(shape, x.dtype),
            barrier=plgpu.Barrier(num_arrivals=1, num_barriers=shape),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
    )
    def _kernel(x_ref, out_ref, smem_ref, barrier):
      thread_id = jax.lax.axis_index('t')
      for i in range(2):
        for j in range(1):
          for k in range(3):
            @pl.when(thread_id == 0)
            def _():
              smem_ref[i, j, k] = x_ref[i, j, k] + 1
              plgpu.barrier_arrive(barrier.at[i, j, k])

            @pl.when(thread_id == 1)
            def _():
              if not with_race:
                plgpu.barrier_wait(barrier.at[i, j, k])
              out_ref[i, j, k] = smem_ref[i, j, k] + 1

    y = _kernel(x)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      np.testing.assert_array_equal(y, x + 2)

  @jtu.parameterized.parameters(range(2, 17))
  def test_single_barrier_with_multiple_arrival(self, num_threads):

    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((num_threads - 1,), jnp.int32),
            barrier_ref=plgpu.Barrier(
                num_arrivals=num_threads - 1, num_barriers=1
            ),
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
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
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((num_threads - 1,), jnp.int32),
            barrier_ref=plgpu.Barrier(
                num_arrivals=1, num_barriers=num_threads - 1
            ),
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
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
        out_type=x,
        scratch_types=dict(
            queue=plgpu.SMEM((buffer_size,), jnp.float32),
            produced=plgpu.Barrier(num_arrivals=1, num_barriers=buffer_size),
            consumed=plgpu.Barrier(num_arrivals=1, num_barriers=buffer_size),
        ),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(
            detect_races=True,
            skip_floating_point_ops=skip_floating_point_ops,
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
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=(plgpu.Barrier(),),
        num_threads=1,
        thread_name='t',
        interpret=InterpretParams(),
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
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=(plgpu.Barrier(num_barriers=2),),
        num_threads=1,
        thread_name='t',
        interpret=InterpretParams(),
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

  def test_wait_for_barrier_twice(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier_ref=plgpu.Barrier(num_arrivals=1, num_barriers=2)
        ),
        num_threads=3,
        thread_name='t',
        interpret=InterpretParams(detect_races=True),
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
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        interpret=InterpretParams(),
    )
    def _kernel(out_ref, barrier_ref):
      plgpu.barrier_arrive(barrier_ref)
      plgpu.barrier_arrive(barrier_ref)
      out_ref[...] = 42

    with self.assertRaisesRegex(
        Exception,
        r'Barrier completed phase 1, but no threads observed phase 0.'
    ):
      _kernel()

  def test_completing_barrier_twice_in_different_threads_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((2,), jnp.int32),
        scratch_types=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        interpret=InterpretParams(),
        num_threads=2,
        thread_name='t',
    )
    def _kernel(out_ref, barrier_ref):
      thread_id = jax.lax.axis_index('t')
      plgpu.barrier_arrive(barrier_ref)
      out_ref[thread_id] = thread_id

    with self.assertRaisesRegex(
        Exception,
        r'Barrier completed phase 1, but no threads observed phase 0.',
    ):
      _kernel()

  @jtu.parameterized.product(
      num_arriving_threads=list(range(1, 8)),
      num_observing_threads=list(range(1, 8)),
      num_threads=[16],
  )
  def test_barrier_wait_in_multiple_threads_ok(
      self, num_arriving_threads, num_observing_threads, num_threads
  ):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((num_threads,), jnp.int32),
        scratch_types=dict(
            barrier_ref=plgpu.Barrier(
                num_arrivals=num_arriving_threads, num_barriers=1
            )
        ),
        num_threads=num_threads,
        thread_name='t',
        interpret=InterpretParams(),
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
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(barrier_ref=plgpu.Barrier(num_arrivals=1)),
        num_threads=2,
        thread_name='t',
        interpret=InterpretParams(),
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
        r'Thread 1 only observed barrier up to phase 0, but barrier'
        r' completed up to phase 1.',
    ):
      _kernel()

  def test_not_waiting_for_all_barrier_completions_in_thread_raises(self):
    @functools.partial(
        plgpu.kernel,
        out_type=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_types=dict(
            barrier_ref=plgpu.Barrier(num_arrivals=1, num_barriers=2)
        ),
        num_threads=3,
        thread_name='t',
        interpret=InterpretParams(),
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
        r'Thread 2 is waiting at barrier \w+ for the first time, but barrier is already at phase 2'
    ):
      _kernel()

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
        out_type=jax.ShapeDtypeStruct.like(a),
        grid=(num_blocks_w, num_blocks_x, num_blocks_y),
        grid_names=('w', 'x', 'y'),
        num_threads=num_threads,
        thread_name='z',
        interpret=InterpretParams(detect_races=True),
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
        out_type=jax.ShapeDtypeStruct((x, y, z), dtype),
        grid=_maybe_reverse((x_iters, y_iters), swap_grid_axes),
        grid_names=_maybe_reverse(('x', 'y'), swap_grid_axes),
        num_threads=z_iters,
        thread_name='z',
        interpret=InterpretParams(detect_races=True),
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
        out_type=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_types=dict(
            acc_smem=plgpu.SMEM((tile_m, tile_n), dtype),
            barrier=plgpu.Barrier(num_barriers=k_iters),
        ),
        grid=_maybe_reverse((m_iters, n_iters), swap_grid_axes),
        grid_names=_maybe_reverse(('m', 'n'), swap_grid_axes),
        num_threads=k_iters,
        thread_name='k',
        interpret=InterpretParams(detect_races=True),
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
        out_type=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_types=dict(
            acc_smem=plgpu.SMEM((tile_m, tile_n), dtype),
        ),
        grid=(m_iters, n_iters),
        grid_names=('m', 'n'),
        num_threads=k_iters,
        thread_name='k',
        interpret=InterpretParams(detect_races=True),
    )

    kernel(a, b)
    self.assertTrue(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_single_thread(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))

    def _kernel(in_gmem, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(in_gmem, smem, barrier)
      if not with_race:
        plgpu.barrier_wait(barrier)
      out_gmem[...] = smem[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            barrier=plgpu.Barrier(), smem=plgpu.SMEM(x.shape, x.dtype)
        ),
    )

    y = kernel(x)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      np.testing.assert_array_equal(y, x)

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_two_threads(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))

    def _kernel(in_gmem, out_gmem, barrier, smem):
      tid = jax.lax.axis_index('t')

      @pl.when(tid == 0)
      def _():
        plgpu.copy_gmem_to_smem(in_gmem, smem, barrier)

      @pl.when(tid == 1)
      def _():
        if not with_race:
          plgpu.barrier_wait(barrier)
        out_gmem[...] = smem[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            barrier=plgpu.Barrier(),
            smem=plgpu.SMEM(x.shape, x.dtype),
        ),
        num_threads=2,
        thread_name='t',
    )

    y = kernel(x)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      np.testing.assert_array_equal(y, x)

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_parallel_two_threads(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))

    def _kernel(in_gmem, out_gmem, per_thread_barrier, smem0, smem1):
      tid = jax.lax.axis_index('t')

      def _per_thread_kernel(smem):
        plgpu.copy_gmem_to_smem(in_gmem, smem, per_thread_barrier.at[tid])
        if not with_race:
          plgpu.barrier_wait(per_thread_barrier.at[tid])
        out_gmem[tid, ...] = smem[tid, ...]

      # TODO(nrink): Investigate why this does not work with a single SMEM
      # buffer and `smem.at[0]` and `smem.at[1]`.
      pl.when(tid == 0)(functools.partial(_per_thread_kernel, smem=smem0))
      pl.when(tid == 1)(functools.partial(_per_thread_kernel, smem=smem1))

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            per_thread_barrier=plgpu.Barrier(num_barriers=2),
            smem0=plgpu.SMEM(x.shape, x.dtype),
            smem1=plgpu.SMEM(x.shape, x.dtype),
        ),
        num_threads=2,
        thread_name='t',
    )

    y = kernel(x)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      np.testing.assert_array_equal(y, x)

  @jtu.parameterized.product(num_tma_threads_per_device=[2, 3, 4])
  def test_copy_gmem_to_smem_multiple_tma_threads(
      self, num_tma_threads_per_device
  ):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))
    y = 2 * x

    def _kernel(gmem0, gmem1, out_gmem, barrier, smem):
      plgpu.copy_gmem_to_smem(gmem0, smem, barrier.at[0])
      plgpu.copy_gmem_to_smem(gmem1, smem, barrier.at[1])
      plgpu.barrier_wait(barrier.at[0])
      plgpu.barrier_wait(barrier.at[1])
      out_gmem[...] = gmem0[...] + gmem1[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(
            detect_races=True,
            num_tma_threads_per_device=num_tma_threads_per_device,
        ),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_barriers=2),
            smem=plgpu.SMEM(x.shape, x.dtype),
        ),
    )

    z = kernel(x, y)
    z.block_until_ready()
    self.assertTrue(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(with_race=[True, False])
  def test_copy_gmem_to_smem_multiple_arrivals_at_barrier(self, with_race):
    x = jnp.arange(16, dtype=jnp.int32).reshape((2, 8))
    y = 2 * x

    def _kernel(gmem0, gmem1, out_gmem, barrier, smem0, smem1):
      plgpu.copy_gmem_to_smem(gmem0, smem0, barrier)
      plgpu.copy_gmem_to_smem(gmem1, smem1, barrier)
      if not with_race:
        plgpu.barrier_wait(barrier)
      out_gmem[...] = smem0[...] + smem1[...]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            barrier=plgpu.Barrier(num_arrivals=2),
            smem0=plgpu.SMEM(x.shape, x.dtype),
            smem1=plgpu.SMEM(x.shape, x.dtype),
        ),
    )

    z = kernel(x, y)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      np.testing.assert_array_equal(z, x + y)

  def test_copy_smem_to_gmem(self):
    def _kernel(out_gmem, smem_ref):
      tid = jax.lax.axis_index('t')
      i = jax.lax.axis_index('i')
      j = jax.lax.axis_index('j')

      smem_ref[tid, pl.ds(i*16, 16), pl.ds(j*32, 32)] = (
          jnp.arange((2*32*128), dtype=jnp.int32).reshape((2, 32, 128))[
              tid, pl.ds(i*16, 16), pl.ds(j*32, 32)])
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          smem_ref.at[tid, pl.ds(i*16, 16), pl.ds(j*32, 32)],
          out_gmem.at[tid, pl.ds(i*16, 16), pl.ds(j*32, 32)])
      plgpu.wait_smem_to_gmem(0)

    y = jax.jit(plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct((2, 32, 128), jnp.int32),
        interpret=InterpretParams(),
        grid=(2, 4),
        grid_names=('i', 'j'),
        scratch_types=dict(
            smem_ref=plgpu.SMEM((2, 32, 128), jnp.int32)
        ),
        num_threads=2,
        thread_name='t',
    ))()

    expected = jnp.arange((2*32*128), dtype=jnp.int32).reshape((2, 32, 128))
    np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(
      grid_dict=[
          None,
          dict(g0=1),
          dict(g0=2, g1=3),
      ],
      cluster_dict=[
          None,
          dict(c0=2),
          dict(c0=2, c1=3),
          dict(c0=2, c1=3, c2=2),
      ],
      thread_dict=[
          None,
          dict(t=1),
          dict(t=3),
      ],
  )
  def test_cluster(self, grid_dict, cluster_dict, thread_dict):
    if cluster_dict is None and thread_dict is None and grid_dict is None:

      def kernel(o_ref):
        o_ref[...] = 42

      y = plgpu.kernel(
          kernel,
          out_type=jax.ShapeDtypeStruct((), jnp.int32),
          interpret=InterpretParams(detect_races=True),
      )()

      self.assertFalse(mosaic_interpret.get_races().races_found)
      self.assertEqual(y, 42)
      return

    mesh_kwargs = {}
    axes_dims = ()
    axes_names = ()
    if grid_dict is not None:
      mesh_kwargs['grid'] = tuple(grid_dict.values())
      mesh_kwargs['grid_names'] = tuple(grid_dict.keys())
      axes_dims += tuple(grid_dict.values())
      axes_names += tuple(grid_dict.keys())
    if cluster_dict is not None:
      mesh_kwargs['cluster'] = tuple(cluster_dict.values())
      mesh_kwargs['cluster_names'] = tuple(cluster_dict.keys())
      axes_dims += tuple(cluster_dict.values())
      axes_names += tuple(cluster_dict.keys())
    if thread_dict is not None:
      (thread_name, num_threads), = thread_dict.items()
      mesh_kwargs['num_threads'] = num_threads
      mesh_kwargs['thread_name'] = thread_name
      axes_dims += (num_threads,)
      axes_names += (thread_name,)
    mesh = plgpu.Mesh(**mesh_kwargs)
    out_shape = axes_dims

    @pl.run_state
    def kernel(o_ref):
      @pl.core_map(
          mesh,
          interpret=InterpretParams(detect_races=True),
      )
      def _():
        flat_thread_id = jnp.int32(0)
        for i, name in enumerate(axes_names):
          stride = math.prod(axes_dims[i + 1 :])
          flat_thread_id += jax.lax.axis_index(name) * stride
        thread_idx = tuple(jax.lax.axis_index(name) for name in axes_names)

        o_ref[thread_idx] = flat_thread_id

    expected = np.arange(
        math.prod(out_shape), dtype=jnp.int32
    ).reshape(out_shape)
    y = kernel(jnp.zeros(out_shape, jnp.int32))
    self.assertFalse(mosaic_interpret.get_races().races_found)
    np.testing.assert_array_equal(y, expected)


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
  epilogue_tile_n: int = 64
  grid_minor_dim: int = 0
  grid_tile_width: int = 1


def matmul0(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem, acc_tmem, acc_smem, consumed_barriers):
    mi = jax.lax.axis_index("m")
    ni = jax.lax.axis_index("n")
    m_slice = pl.ds(mi * tile_m, tile_m)
    n_slice = pl.ds(ni * tile_n, tile_n)

    def do_mma(idxs, a_smem, b_smem):
      (ki,) = idxs
      arrive_barrier_slot = ki % 2
      wait_barrier_slot = 1 - arrive_barrier_slot
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier=consumed_barriers.at[arrive_barrier_slot],
          accumulate=(ki > 0),
      )
      plgpu.barrier_wait(consumed_barriers.at[wait_barrier_slot])

    # Make sure the wait succeeds in the first iteration.
    plgpu.barrier_arrive(consumed_barriers.at[1])
    block_kwargs = dict(transforms=transforms, delay_release=1)
    plgpu.emit_pipeline(
      do_mma,
      in_specs=[
          plgpu.BlockSpec((tile_m, tile_k), lambda ki: (mi, ki), **block_kwargs),
          plgpu.BlockSpec((tile_k, tile_n), lambda ki: (ki, ni), **block_kwargs),
      ],
      grid=(k_iters,),
      max_concurrent_steps=max_concurrent_steps,
    )(a_gmem, b_gmem)

    final_barrier = 1 - (k_iters % 2)
    plgpu.barrier_wait(consumed_barriers.at[final_barrier])
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      scratch_types=dict(
          acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
      ),
  )
  return f(a, b)


# TODO(nrink): Figure out how to safely run different instance of GPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class BlackwellExampleMatmulTest(jtu.JaxTestCase):

  @jtu.run_on_devices('cpu')
  def test_matmul0(self):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        max_concurrent_steps=4,
    )
    m, n, k = 512, 512, 512
    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.uniform(k1, (m, k), jnp.float16)
    b = jax.random.uniform(k2, (k, n), jnp.float16)

    device = jax.sharding.AbstractDevice(
        device_kind='NVIDIA B200',
        platform='gpu',
        num_cores=8,
    )
    with (
        jax.sharding.use_abstract_mesh(
            jax.sharding.AbstractMesh((), (), abstract_device=device)
        ),
        force_gpu_interpret_mode(InterpretParams()),
    ):
      res = matmul0(a, b, example_config).block_until_ready()

    expected = jnp.dot(a, b, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-3)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
