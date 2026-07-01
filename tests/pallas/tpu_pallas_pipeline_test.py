# Copyright 2023 The JAX Authors.
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

"""Test TPU-specific extensions to pallas_call."""
from __future__ import annotations

import dataclasses
import functools
from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax._src import hijax
from jax._src import hypothesis_test_util as htu
from jax._src import state
from jax._src import test_util as jtu
from jax._src import config
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


hp.settings.register_profile(
    'deterministic',
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=200,
    print_blob=True,
    verbosity=hp.Verbosity.verbose,
)
hp.settings.load_profile('deterministic')


jax.config.parse_flags_with_absl()


@jax.named_scope('compute')
def basic_matmul_kernel(
    lhs_ref,
    rhs_ref,
    out_ref,
    acc_scratch_ref,
    *,
    k: int,
):
  k_index = pl.program_id(2)
  num_k = pl.num_programs(2)
  bk = lhs_ref.shape[1]
  @pl.when(k_index == 0)
  def _zero_acc():
    acc_scratch_ref[...] = jnp.zeros(
        acc_scratch_ref.shape, acc_scratch_ref.dtype)

  divisible_k = k % bk == 0
  if divisible_k:
    acc_scratch_ref[...] += jnp.dot(
        lhs_ref[...],
        rhs_ref[...],
        preferred_element_type=acc_scratch_ref.dtype,
    )
  else:
    def _last_block():
      accum_dtype = acc_scratch_ref.dtype
      lhs_mask = (
          k_index * bk + jax.lax.broadcasted_iota(jnp.int32, lhs_ref.shape, 1)
          < k
      )
      rhs_mask = (
          k_index * bk + jax.lax.broadcasted_iota(jnp.int32, rhs_ref.shape, 0)
          < k
      )
      dtype = lhs_ref.dtype
      lhs = lhs_ref[...].astype(accum_dtype)
      lhs = jnp.where(lhs_mask, lhs, 0).astype(dtype)
      rhs = rhs_ref[...].astype(accum_dtype)
      rhs = jnp.where(rhs_mask, rhs, 0).astype(dtype)
      acc_scratch_ref[...] += jnp.dot(
          lhs, rhs, preferred_element_type=acc_scratch_ref.dtype)
    def _not_last_block():
      acc_scratch_ref[...] += jnp.dot(
          lhs_ref[...],
          rhs_ref[...],
          preferred_element_type=acc_scratch_ref.dtype,
      )
    jax.lax.cond(
       k_index == num_k - 1, _last_block, _not_last_block
    )

  @pl.when(k_index == num_k - 1)
  def _reduce_out():
    out_ref[...] = acc_scratch_ref[...].astype(out_ref.dtype)


class PallasCallPipelineTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5')

    super().setUp()

  @parameterized.parameters(((0, 2),), ((2, -1),))
  def test_pipeline_empty_grid(self, grid):
    def kernel(o_hbm_ref):
      def body(o_ref):
        o_ref[...] = jnp.zeros((128, 128))

      pltpu.emit_pipeline(
          body, grid=grid, out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j))
      )(o_hbm_ref)

    with self.assertRaisesRegex(
        ValueError, 'All elements in the grid must be strictly positive'
    ):
      pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
      )()

  def test_pipeline_without_inputs(self):
    def kernel(o_hbm_ref):
      def body(o_ref):
        o_ref[...] = jnp.full(o_ref.shape, 42, dtype=o_ref.dtype)

      pltpu.emit_pipeline(
          body, grid=(4,), out_specs=pl.BlockSpec((8, 128), lambda i: (0, i))
      )(o_hbm_ref)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 512), jnp.int32),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )()
    np.testing.assert_allclose(out, jnp.full_like(out, 42))

  def test_hbm_output(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 512), jnp.int32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )
    def kernel(x_hbm_ref, o_hbm_ref):
      @functools.partial(
          pltpu.emit_pipeline,
          grid=(4,),
          in_specs=pl.BlockSpec((8, 128), lambda i: (0, i)),
          out_specs=pl.BlockSpec(
              (8, 512), lambda i: (0, 0), memory_space=pltpu.HBM
          ),
      )
      def pipeline(x_ref, o_ref):
        i = pl.program_id(0)
        pltpu.sync_copy(x_ref, o_ref.at[:, pl.ds(i * 128, 128)])

      pipeline(x_hbm_ref, o_hbm_ref)

    x = jnp.arange(8 * 512).reshape(8, 512)
    np.testing.assert_allclose(kernel(x), x)

  def test_emit_pipeline_input_hbm_memory_space(self):
    def pipeline_body(x_ref, y_ref, z_ref, o_ref, scratch_ref):
      pltpu.sync_copy(y_ref, o_ref)
      pltpu.sync_copy(z_ref, scratch_ref)
      o_ref[...] = o_ref[...] + x_ref[...] + 2 * scratch_ref[...]
      assert y_ref.shape == (8, 128) and y_ref.dtype == jnp.float32

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((8 * 8, 128), jnp.float32),
        mesh=pltpu.create_tensorcore_mesh('core'),
        scratch_types=[pltpu.VMEM((8, 128), jnp.float32)],
    )
    def kernel(x_hbm_ref, y_hbm_ref, z_hbm_ref, o_hbm_ref, scratch_ref):
      pltpu.emit_pipeline(
          functools.partial(pipeline_body, scratch_ref=scratch_ref),
          grid=(8,),
          in_specs=[
              pl.BlockSpec((8, 128), lambda i: (i, 0)),
              pl.BlockSpec((8, 128), lambda i: (i, 0), memory_space=pltpu.HBM),
              pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0),
                           memory_space=pltpu.HBM),
          ],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
      )(x_hbm_ref, y_hbm_ref, z_hbm_ref, o_hbm_ref)

    x = jnp.arange(8 * 8 * 128, dtype=jnp.float32).reshape((8 * 8, 128))
    y = jnp.arange(8 * 8 * 128, dtype=jnp.float32).reshape((8 * 8, 128))
    z = jnp.arange(8 * 8 * 128, dtype=jnp.float32).reshape((8, 8, 128))
    out = kernel(x, y, z)
    np.testing.assert_allclose(out, x + y + 2 * z.reshape((8 * 8, 128)))

  def test_trivial_windowing_elides_double_buffering(self):
    def pipeline_body(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[...]

    def kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref):
      pltpu.emit_pipeline(
          pipeline_body,
          grid=(4, 1),
          in_specs=[
              pl.BlockSpec((8, 128), lambda i, j: (0, i)),
              pl.BlockSpec((8, 128), lambda i, j: (0, j)),
          ],
          out_specs=pl.BlockSpec((8, 128), lambda i, j: (0, i)),
      )(x_hbm_ref, y_hbm_ref, o_hbm_ref)

    x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
    y = jnp.ones((8, 128), jnp.float32)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )(x, y)
    expected = x + jnp.tile(y, (1, 4))
    np.testing.assert_allclose(out, expected)

  def test_trivial_windowing_matched_vmem(self):
    def pipeline_body(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def kernel(x_hbm_ref, o_hbm_ref):
      @functools.partial(
          pl.run_scoped,
          x_vmem=pltpu.VMEM((8, 512), jnp.int32),
          o_vmem=pltpu.VMEM((8, 512), jnp.int32),
      )
      def _(x_vmem, o_vmem):
        pltpu.sync_copy(x_hbm_ref, x_vmem)
        pltpu.emit_pipeline(
            pipeline_body,
            grid=(4,),
            in_specs=pl.BlockSpec(
                (8, 512), lambda i: (0, 0), memory_space=pltpu.VMEM
            ),
            out_specs=pl.BlockSpec(
                (8, 512), lambda i: (0, 0), memory_space=pltpu.VMEM
            ),
        )(x_vmem, o_vmem)
        pltpu.sync_copy(o_vmem, o_hbm_ref)

    x = jnp.arange(8 * 512, dtype=jnp.int32).reshape(8, 512)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 512), jnp.int32),
    )(x)
    np.testing.assert_allclose(out, x)

  def test_trivial_windowing_mismatched_input(self):
    def pipeline_body(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def kernel(x_hbm_ref, o_hbm_ref):
      @functools.partial(
          pl.run_scoped,
          o_vmem=pltpu.VMEM((3,), jnp.int32),
      )
      def _(o_vmem):
        pltpu.emit_pipeline(
            pipeline_body,
            grid=(1,),
            in_specs=pl.BlockSpec(
                (3,), lambda i: (0,), memory_space=pltpu.VMEM
            ),
            out_specs=pl.BlockSpec(
                (3,), lambda i: (0,), memory_space=pltpu.VMEM
            ),
        )(x_hbm_ref, o_vmem)
        pltpu.sync_copy(o_vmem, o_hbm_ref)

    x = jnp.array([1, 2, 3], dtype=jnp.int32)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((3,), jnp.int32),
    )(x)
    np.testing.assert_allclose(out, x)

  def test_trivial_windowing_mismatched_output(self):
    def pipeline_body(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def kernel(x_hbm_ref, o_hbm_ref):
      @functools.partial(
          pl.run_scoped,
          x_vmem=pltpu.VMEM((3,), jnp.int32),
      )
      def _(x_vmem):
        pltpu.sync_copy(x_hbm_ref, x_vmem)
        pltpu.emit_pipeline(
            pipeline_body,
            grid=(1,),
            in_specs=pl.BlockSpec(
                (3,), lambda i: (0,), memory_space=pltpu.VMEM
            ),
            out_specs=pl.BlockSpec(
                (3,), lambda i: (0,), memory_space=pltpu.VMEM
            ),
        )(x_vmem, o_hbm_ref)

    x = jnp.array([1, 2, 3], dtype=jnp.int32)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((3,), jnp.int32),
    )(x)
    np.testing.assert_allclose(out, x)

  def test_trivial_windowing_with_explicit_buffering(self):
    def pipeline_body(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def kernel(x_hbm_ref, o_hbm_ref):
      pltpu.emit_pipeline(
          pipeline_body,
          grid=(1,),
          in_specs=pl.BlockSpec(
              (8, 512), lambda i: (0, 0),
              pipeline_mode=pl.Buffered(buffer_count=2)
          ),
          out_specs=pl.BlockSpec((8, 512), lambda i: (0, 0)),
      )(x_hbm_ref, o_hbm_ref)

    x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
    out = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 512), jnp.float32),
    )(x)
    np.testing.assert_allclose(out, x)

  def test_prng_pallas_vs_pipeline(self):
    def kernel(key_ref, o_ref):
      o_ref[...] = jax.random.uniform(key_ref[...], shape=o_ref.shape)

    def pipeline_kernel(key_hbm, o_hbm):
      pltpu.emit_pipeline(
          kernel,
          grid=(),
          in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
          out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
      )(key_hbm, o_hbm)

    out_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    key = jax.random.key(0)
    p_key = pltpu.to_pallas_key(key)

    o_standard = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )(p_key)

    o_pipeline = pl.pallas_call(
        pipeline_kernel,
        out_shape=out_shape,
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )(p_key)

    np.testing.assert_array_equal(o_standard, o_pipeline)

  def test_key_memory_space(self):
    def kernel(key_ref, o_ref):
      o_ref[...] = jax.random.uniform(key_ref[...], shape=o_ref.shape)

    def pipeline_kernel(key_hbm, o_hbm):
      pltpu.emit_pipeline(
          kernel,
          grid=(),
          in_specs=[pl.BlockSpec(memory_space=pl.MemorySpace.KEY)],
          out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
      )(key_hbm, o_hbm)

    out_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    key = jax.random.key(0)
    p_key = pltpu.to_pallas_key(key)

    o_standard = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )(p_key)

    o_pipeline = pl.pallas_call(
        pipeline_kernel,
        out_shape=out_shape,
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )(p_key)

    np.testing.assert_array_equal(o_standard, o_pipeline)

  def test_squeezed_block_spec_has_correct_shape(self):
    def body(x_ref):
      self.assertEqual(x_ref.shape, (128, 64))

    x = jnp.zeros((1, 128, 64), jnp.float32)
    spec = pl.BlockSpec(
        (None, 128, 64),
        lambda b: (0, 0, 0),
        memory_space=pltpu.VMEM,
    )

    def run_kernel(x_ref, o_ref):
      pltpu.emit_pipeline(
          body,
          grid=(1,),
          in_specs=(spec,),
      )(x_ref)

    pl.pallas_call(
        run_kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
    )(x)

  def test_trivial_window_representation_strips_all_none_dimensions(
      self):
    def body(x_ref):
      self.assertEqual(x_ref.shape, (128, 64))

    x = jnp.zeros((1, 1, 128, 64), jnp.float32)
    spec = pl.BlockSpec(
        (None, None, 128, 64),
        lambda b: (0, 0, 0, 0),
        memory_space=pltpu.VMEM,
    )

    def run_kernel(x_ref, _):
      pltpu.emit_pipeline(
          body,
          grid=(1,),
          in_specs=(spec,),
      )(x_ref)

    pl.pallas_call(
        run_kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
    )(x)

  def test_unbuffered_vmem_output_flushes_to_hbm(self):
    def body(o_ref):
      o_ref[...] = jnp.ones((128, 64), jnp.float32)

    spec = pl.BlockSpec(
        (128, 64),
        lambda b: (0, 0),
        memory_space=pltpu.VMEM,
    )

    def run_kernel(o_ref):
      pltpu.emit_pipeline(
          body,
          grid=(1,),
          out_specs=(spec,),
      )(o_ref)

    out = pl.pallas_call(
        run_kernel,
        out_shape=jax.ShapeDtypeStruct((128, 64), jnp.float32),
    )()
    expected_out = jnp.ones((128, 64), jnp.float32)
    np.testing.assert_allclose(out, expected_out)

  @parameterized.product(no_pipelining=[False, True])
  def test_pipeline_matmul(self, no_pipelining):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (512, 512))
    y = jax.random.uniform(k2, (512, 512))

    def matmul_pipeline(x_ref, y_ref, z_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        z_ref[...] = jnp.zeros(z_ref.shape, jnp.float32)

      z_ref[...] += x_ref[...] @ y_ref[...]

    def matmul_kernel(x_ref, y_ref, z_ref):
      pltpu.emit_pipeline(
          matmul_pipeline,
          grid=(4, 4, 4),
          in_specs=[
              pl.BlockSpec((128, 128), lambda i, j, k: (i, k)),
              pl.BlockSpec((128, 128), lambda i, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((128, 128), lambda i, j, k: (i, j)),
          no_pipelining=no_pipelining,
      )(x_ref, y_ref, z_ref)

    z = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((512, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )

    jax.block_until_ready(z(x, y))
    jax.block_until_ready(jnp.dot(x, y))

    out = jax.block_until_ready(z(x, y))
    expected_out = jax.block_until_ready(jnp.dot(x, y))

    np.testing.assert_allclose(out, expected_out, atol=5e-5)

  def test_unaligned_slice_hbm_3d(self):
    M, N = 32, 5  # not aligned with 128.

    def kernel(x_hbm, out_hbm):
      hbm_slice = x_hbm.at[0, :, :]
      pltpu.sync_copy(hbm_slice, out_hbm)

    x = jnp.arange(2 * M * N, dtype=jnp.float32).reshape(2, M, N)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(),
    )(x)

    np.testing.assert_allclose(out, x[0])

  @parameterized.named_parameters(
      ('standard', (4,), 128, 2, False, (2, 128), 'first_buffer'),
      ('lookahead', (4,), 128, 2, True, (2, 128), 'first_buffer'),
      ('trivial_windowing', (1,), 512, 1, False, (512,), 'full'),
  )
  def test_prefetched_input(
      self,
      grid,
      block_size,
      buffer_count,
      use_lookahead,
      vmem_shape,
      vmem_slice_type,
  ):
    if config.use_emit_pipeline_primitive.value:
      self.skipTest(
          'allocations are not yet supported by the emit_pipeline primitive.'
      )

    def pipeline_body(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    if vmem_slice_type == 'first_buffer':
      vmem_slice_fn = lambda ref: ref.at[0]
    elif vmem_slice_type == 'full':
      vmem_slice_fn = lambda ref: ref
    else:
      raise ValueError(f'Unknown vmem_slice_type: {vmem_slice_type}')

    def prefetch_start(x_hbm_ref, x_vmem_ref, sem):
      pltpu.async_copy(
          x_hbm_ref.at[:block_size],
          vmem_slice_fn(x_vmem_ref),
          sem,
      )

    def prefetch_done(x_hbm_ref, x_vmem_ref, sem, x_vmem_out_ref):
      del x_vmem_out_ref  # alias to x_vmem_ref
      pltpu.make_async_copy(
          x_hbm_ref.at[:block_size],
          vmem_slice_fn(x_vmem_ref),
          sem,
      ).wait()

    def kernel(x_hbm_ref, x_prefetched_vmem, o_hbm_ref):
      pipeline, make_allocs = pltpu.emit_pipeline_with_allocations(
          pipeline_body,
          grid=grid,
          in_specs=[
              pl.BlockSpec(
                  (block_size,),
                  lambda i: (i,),
                  pipeline_mode=pl.Buffered(
                      buffer_count=buffer_count,
                      use_lookahead=use_lookahead,
                      prefetched_count=1,
                  ),
              )
          ],
          out_specs=[pl.BlockSpec((block_size,), lambda i: (i,))],
      )

      @functools.partial(
          pl.run_scoped, allocs=make_allocs(x_hbm_ref, o_hbm_ref)
      )
      def _(allocs):
        x_bref, o_bref = allocs
        allocs = (x_bref.with_window_ref(x_prefetched_vmem), o_bref)
        return pipeline(x_hbm_ref, o_hbm_ref, allocations=allocs)

    x = jnp.arange(512, dtype=jnp.float32)
    @jax.jit
    def run(x):
      prefetched, sem = pl.pallas_call(
          prefetch_start,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
          ],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.VMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_shape=(
              pltpu.VMEM(vmem_shape, jnp.float32),
              pltpu.SemaphoreType.DMA,
          ),
      )(x)
      prefetched = pl.pallas_call(
          prefetch_done,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.VMEM),
              pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          out_shape=pltpu.VMEM(vmem_shape, jnp.float32),
          input_output_aliases={1: 0},
      )(x, prefetched, sem)

      out = pl.pallas_call(
          kernel,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.VMEM),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          out_shape=jax.ShapeDtypeStruct((512,), jnp.float32),
      )(x, prefetched)
      return out

    out = run(x)
    np.testing.assert_allclose(out, x)

  def test_prefetched_input_unbound_error(self):
    if config.use_emit_pipeline_primitive.value:
      self.skipTest(
          'allocations are not yet supported by the emit_pipeline primitive.'
      )

    def pipeline_body(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def kernel(x_hbm_ref, o_hbm_ref):
      pipeline, make_allocs = pltpu.emit_pipeline_with_allocations(
          pipeline_body,
          grid=(4,),
          in_specs=[
              pl.BlockSpec(
                  (128,),
                  lambda i: (i,),
                  pipeline_mode=pl.Buffered(
                      buffer_count=2,
                      prefetched_count=1,
                  ),
              )
          ],
          out_specs=[pl.BlockSpec((128,), lambda i: (i,))],
      )

      @functools.partial(
          pl.run_scoped, allocs=make_allocs(x_hbm_ref, o_hbm_ref)
      )
      def _(allocs):
        # Purposely do not bind window_ref using with_window_ref!
        return pipeline(x_hbm_ref, o_hbm_ref, allocations=allocs)

    x = jnp.arange(512, dtype=jnp.float32)

    @jax.jit
    def run(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          out_shape=jax.ShapeDtypeStruct((512,), jnp.float32),
      )(x)

    with self.assertRaisesRegex(
        ValueError, 'Expected external window buffer to be bound for prefetched input'
    ):
      run(x)


@jtu.with_config(jax_pallas_poison_buffers=True)
class PallasCallPipelinePoisonTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5')
    super().setUp()

  @parameterized.product(
      dtype=[jnp.float32, jnp.int32, jnp.int8, jnp.float8_e4m3fn],
      no_pipelining=[False, True])
  def test_poison_buffers(self, dtype, no_pipelining):
    def pipeline_body(x_ref, o_ref):
      o_ref[:4, :] = x_ref[:4, :]

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((8, 128), dtype),
        mesh=pltpu.create_tensorcore_mesh('core'),
    )
    def kernel(x_hbm_ref, o_hbm_ref):
      pltpu.emit_pipeline(
          pipeline_body,
          grid=(1,),
          in_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
          out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
          no_pipelining=no_pipelining,
      )(x_hbm_ref, o_hbm_ref)

    x = jnp.ones((8, 128), dtype=dtype)
    out = kernel(x)

    # First half should be 1s (copied from x)
    np.testing.assert_allclose(
        out[:4, :].astype(jnp.float32), x[:4, :].astype(jnp.float32)
    )

    # Second half should be poisoned
    if jnp.issubdtype(dtype, jnp.floating):
      self.assertTrue(np.isnan(out[4:, :].astype(jnp.float32)).all())
    else:
      expected_poison = jnp.iinfo(dtype).min
      np.testing.assert_array_equal(
          out[4:, :].astype(jnp.int32), expected_poison
      )


class PallasCallMultipleBufferedPipelineTest(jtu.JaxTestCase):

  def setUp(self):
    if jax.device_count() > 1:
      self.skipTest('Only 1 device is supported.')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5+')
    super().setUp()

  @parameterized.product(
      in_buffer_count=[2, 4],
      out_buffer_count=[2],
  )
  def test_copy(self, in_buffer_count, out_buffer_count):
    x = jnp.reshape(jnp.arange(512 * 512), (512, 512))
    def copy_kernel(x_hbm_ref, o_hbm_ref):
      def inner_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      pltpu.emit_pipeline(
          inner_kernel,
          grid=(4, 4),
          in_specs=[
              pl.BlockSpec((128, 128), lambda i, j: (i, j),
                pipeline_mode=pl.Buffered(buffer_count=in_buffer_count)),
          ],
          out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j),
            pipeline_mode=pl.Buffered(buffer_count=out_buffer_count)),
      )(x_hbm_ref, o_hbm_ref)
    fn = pl.pallas_call(
        copy_kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )
    result = fn(x)
    np.testing.assert_allclose(result, x)

  @parameterized.product(
      x_buffer_count=[2, 4],
      y_buffer_count=[2, 4],
      out_buffer_count=[2],
  )
  def test_matmul(self, x_buffer_count, y_buffer_count, out_buffer_count):
    block_shape = (128, 128)
    x = jax.random.uniform(jax.random.key(0), (512, 512))
    y = jax.random.uniform(jax.random.key(1), (512, 512))
    def matmul_kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref):
      def pipeline_step(x_ref, y_ref, o_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
          o_ref[...] = jnp.zeros(o_ref.shape, jnp.float32)
        o_ref[...] += x_ref[...] @ y_ref[...]
      pltpu.emit_pipeline(
          pipeline_step,
          grid=(
              512 // block_shape[0],
              512 // block_shape[0],
              512 // block_shape[0],
          ),
          in_specs=[
              pl.BlockSpec(
                  block_shape,
                  lambda i, j, k: (i, k),
                  pipeline_mode=pl.Buffered(buffer_count=x_buffer_count),
              ),
              pl.BlockSpec(
                  block_shape,
                  lambda i, j, k: (k, j),
                  pipeline_mode=pl.Buffered(buffer_count=y_buffer_count),
              ),
          ],
          out_specs=pl.BlockSpec(
              block_shape,
              lambda i, j, k: (i, j),
              pipeline_mode=pl.Buffered(buffer_count=out_buffer_count),
          ),
      )(x_hbm_ref, y_hbm_ref, o_hbm_ref)
    fn = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )
    result = fn(x, y)
    np.testing.assert_allclose(result, x @ y, atol=5e-5)

  @parameterized.product(
      x_buffer_count=[2, 4],
      y_buffer_count=[2, 4],
      out_buffer_count=[2],
  )
  def test_matmul_megacore(self, x_buffer_count, y_buffer_count,
                           out_buffer_count):
    block_shape = (128, 128)
    x = jax.random.uniform(jax.random.key(0), (512, 512))
    y = jax.random.uniform(jax.random.key(1), (512, 512))
    def matmul_kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref):
      def pipeline_step(x_ref, y_ref, o_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
          o_ref[...] = jnp.zeros(o_ref.shape, jnp.float32)
        o_ref[...] += x_ref[...] @ y_ref[...]
      pltpu.emit_pipeline(
          pipeline_step,
          core_axis_name='core',
          grid=(
              512 // block_shape[0],
              512 // block_shape[0],
              512 // block_shape[0],
          ),
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
          in_specs=[
              pl.BlockSpec(
                  block_shape,
                  lambda i, j, k: (i, k),
                  pipeline_mode=pl.Buffered(buffer_count=x_buffer_count),
              ),
              pl.BlockSpec(
                  block_shape,
                  lambda i, j, k: (k, j),
                  pipeline_mode=pl.Buffered(buffer_count=y_buffer_count),
              ),
          ],
          out_specs=pl.BlockSpec(
              block_shape,
              lambda i, j, k: (i, j),
              pipeline_mode=pl.Buffered(buffer_count=out_buffer_count),
          ),
      )(x_hbm_ref, y_hbm_ref, o_hbm_ref)

    @jax.jit
    def fn(x, y):
      x_ref = jax.new_ref(x)
      y_ref = jax.new_ref(y)
      o_ref = jax.empty_ref(jax.ShapeDtypeStruct(x.shape, jnp.float32))
      mesh = pltpu.create_tensorcore_mesh('core')

      @pl.core_map(mesh)
      def _():
        matmul_kernel(x_ref, y_ref, o_ref)

      return jax.freeze(o_ref)

    result = fn(x, y)
    np.testing.assert_allclose(result, x @ y, atol=5e-5)

  @parameterized.product(
      in_buffer_count=[2, 4],
      out_buffer_count=[2],
      in_block_indices=[
          [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          [0, 0, 2, 3, 3, 3, 1, 7, 6, 6],
          [3, 3, 3, 3, 3, 3, 3, 7, 6, 6],
          [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
      ],
      use_lookahead=[True, False],
  )
  def test_block_gather(self, in_block_indices, in_buffer_count,
                        out_buffer_count, use_lookahead):
    # Excercises pipeline with repeated input block indices.
    block_size = 128
    x = jnp.reshape(jnp.arange(1024 * 128), (1024, 128))

    def copy_kernel(x_hbm_ref, blk_indices_ref, o_hbm_ref):
      def x_index_map(i):
        return (blk_indices_ref[i], 0)
      def inner_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      pltpu.emit_pipeline(
          inner_kernel,
          grid=(len(in_block_indices),),
          in_specs=[
              pl.BlockSpec(
                  (128, 128),
                  index_map=x_index_map,
                  pipeline_mode=pl.Buffered(buffer_count=in_buffer_count,
                                            use_lookahead=use_lookahead),
              ),
          ],
          out_specs=pl.BlockSpec(
              (128, 128),
              lambda i: (i, 0),
              pipeline_mode=pl.Buffered(buffer_count=out_buffer_count),
          ),
      )(x_hbm_ref, o_hbm_ref)
    fn = pl.pallas_call(
        copy_kernel,
        out_shape=jax.ShapeDtypeStruct((len(in_block_indices) * 128, 128), jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )
    result = fn(x, jnp.array(in_block_indices))

    expected = []
    for blk_idx in in_block_indices:
      expected.append(x[blk_idx * block_size:(blk_idx + 1) * block_size, :])
    expected = jnp.concatenate(expected, axis=0)
    np.testing.assert_allclose(result, expected)

  @parameterized.product(
      in_buffer_count=[2, 4],
      out_buffer_count=[2],
      out_block_indices=[
        [0, 0, 2, 2, 2, 5, 3, 3],
        [5, 5, 5, 5, 5, 5, 5, 5],
      ],
  )
  def test_block_scatter(self, out_block_indices, in_buffer_count,
                         out_buffer_count):
    # Excercises pipeline with repeated output block indices.
    block_size = 128
    x = jnp.reshape(jnp.arange(1024 * 128), (1024, 128))

    def copy_kernel(x_hbm_ref, blk_indices_ref, o_hbm_ref):
      # zero-out o_hbm_ref
      @functools.partial(pl.run_scoped,
                         o_vmem=pltpu.VMEM((1024, 128), jnp.int32))
      def _(o_vmem):
        o_vmem[...] = jnp.zeros(o_vmem.shape, jnp.int32)
        pltpu.sync_copy(o_vmem, o_hbm_ref)

      def o_index_map(i):
        return (blk_indices_ref[i], 0)
      def inner_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      pltpu.emit_pipeline(
          inner_kernel,
          grid=(8,),
          in_specs=[
              pl.BlockSpec((128, 128), index_map=lambda i: (i, 0),
                pipeline_mode=pl.Buffered(buffer_count=in_buffer_count)),
          ],
          out_specs=pl.BlockSpec((128, 128), o_index_map,
            pipeline_mode=pl.Buffered(buffer_count=out_buffer_count)),
      )(x_hbm_ref, o_hbm_ref)
    fn = pl.pallas_call(
        copy_kernel,
        out_shape=jax.ShapeDtypeStruct((1024, 128), jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )
    result = fn(x, jnp.array(out_block_indices))

    expected = [jnp.zeros((128, 128), jnp.int32)] * 8
    for i, blk_idx in enumerate(out_block_indices):
      expected[blk_idx] = x[i * block_size:(i + 1) * block_size, :]
    expected = jnp.concatenate(expected, axis=0)
    np.testing.assert_allclose(result, expected)

  def test_matmul_with_input_output(self):
    if config.use_emit_pipeline_primitive.value:
      self.skipTest(
          'allocations are not yet supported by the emit_pipeline primitive.')

    M, N, K = 512, 512, 512
    blk_m, blk_n, blk_k = 128, 128, 128
    nm, nn, nk = M // blk_m, N // blk_n, K // blk_k
    inner_allocs = [
        pltpu.BufferedRef.input(
            pl.BlockSpec((blk_m, blk_k), lambda n, m, k: (m, k)), jnp.float32),
        pltpu.BufferedRef.input(
            pl.BlockSpec((blk_k, blk_n), lambda n, m, k: (k, n)), jnp.float32),
        pltpu.BufferedRef.input_output(
            pl.BlockSpec((blk_m, blk_n), lambda n, m, k: (m, n)), jnp.float32),
        ]

    def matmul_kernel(x_hbm, y_hbm, o_hbm, x_bref, y_bref, o_bref):
      def inner_kernel(x_ref, y_ref, o_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
          o_ref[...] = jnp.zeros_like(o_ref)
        o_ref[...] += x_ref[...] @ y_ref[...]

      pltpu.emit_pipeline(
          inner_kernel,
          grid=(nm, nn, nk),
      )(
        x_hbm, y_hbm, o_hbm,
        allocations=[x_bref, y_bref, o_bref]
      )

    x = jax.random.uniform(jax.random.key(0), (M, K), jnp.float32)
    y = jax.random.uniform(jax.random.key(1), (K, N), jnp.float32)
    fn = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=inner_allocs,
    )
    result = fn(x, y)
    np.testing.assert_allclose(result, x @ y, atol=5e-5)

  def test_single_buffered_output(self):
    def body(o_ref):
      @pl.when(pl.program_id(1) == 0)
      def _():
        o_ref[...] = jnp.zeros_like(o_ref)
      o_ref[...] += 1.0

    spec = pl.BlockSpec(
        (1, 8, 128),
        lambda b, _: (b, 0, 0),
        memory_space=pltpu.VMEM,
        pipeline_mode=pl.Buffered(buffer_count=1),
    )

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((2, 8, 128), jnp.float32),
        mesh=pltpu.create_tensorcore_mesh('core'),
    )
    def run_kernel(o_ref):
      pltpu.emit_pipeline(
          body,
          grid=(2, 3),
          out_specs=(spec,),
      )(o_ref)

    out = run_kernel()
    expected_out = jnp.full((2, 8, 128), 3, jnp.float32)
    np.testing.assert_allclose(out, expected_out)


class PallasCallCollectivePipelineTest(jtu.JaxTestCase):

  def setUp(self):
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if jtu.is_device_tpu(7, variant='x') and jax.device_count() < 4:
      # v7x consists of pairs of chips that share the same ICI connection,
      # so we need at least 4 chips to test collectives.
      self.skipTest('Only >=4 devices are supported on TPU v7x.')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5')

    super().setUp()


class PallasCallMegacoreTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works with TPU v4')

    super().setUp()

  def test_can_partition_nondivisible_grid_with_dynamic_dimensions(
      self):

    def mul_pipeline(x_ref, y_ref):
      y_ref[...] = x_ref[...] * 2

    def mul_kernel(i, x_ref, y_ref):
      pltpu.emit_pipeline(
          mul_pipeline,
          grid=(i, 5),
          in_specs=[
              pl.BlockSpec((128, 128), lambda i, j: (i, j)),
          ],
          out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
          core_axis_name='core',
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
      )(x_ref, y_ref)

    @jax.jit
    def func(x, i):
      x_ref = jax.new_ref(x)
      y_ref = jax.empty_ref(jax.ShapeDtypeStruct.like(x))
      mesh = pltpu.create_tensorcore_mesh('core')

      @pl.core_map(mesh)
      def _():
        mul_kernel(i, x_ref, y_ref)

      return jax.freeze(y_ref)

    x = jax.random.uniform(jax.random.key(0), (640, 640))
    np.testing.assert_allclose(func(x, 5), x * 2)

  @parameterized.parameters(
      # 1D scenarios
      dict(
          grid_vals=(8,),
          dimension_semantics=(pltpu.PARALLEL,),
          dynamic_grid_index=None,
      ),
      dict(
          grid_vals=(5,),
          dimension_semantics=(pltpu.PARALLEL,),
          dynamic_grid_index=None,
      ),
      dict(
          grid_vals=(8,),
          dimension_semantics=(pltpu.PARALLEL,),
          dynamic_grid_index=0,
      ),
      dict(
          grid_vals=(5,),
          dimension_semantics=(pltpu.PARALLEL,),
          dynamic_grid_index=0,
      ),
      # 2D scenarios (1 Parallel, 1 Arbitrary)
      dict(
          grid_vals=(8, 3),
          dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
          dynamic_grid_index=None,
      ),
      dict(
          grid_vals=(5, 3),
          dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
          dynamic_grid_index=None,
      ),
      dict(
          grid_vals=(8, 3),
          dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
          dynamic_grid_index=0,
      ),
      dict(
          grid_vals=(5, 3),
          dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
          dynamic_grid_index=0,
      ),
      dict(
          grid_vals=(2, 7),
          dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
          dynamic_grid_index=1,
      ),
      # 2D Two Parallel (Static)
      dict(
          grid_vals=(8, 8),
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
          dynamic_grid_index=None,
      ),
      dict(
          grid_vals=(5, 7),
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
          dynamic_grid_index=None,
      ),
      dict(
          grid_vals=(1, 1),
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
          dynamic_grid_index=None,
      ),
  )
  def test_can_partition_grid(
      self, grid_vals, dimension_semantics, dynamic_grid_index=None
  ):

    def mul_kernel(grid, o_vmem_ref):
      def mul_pipeline():
        o_vmem_ref[...] += 1
      pltpu.emit_pipeline(
          mul_pipeline,
          grid=grid,
          out_specs=[],
          core_axis_name='core',
          dimension_semantics=dimension_semantics,
      )()

    num_cores = pltpu.get_tpu_info().num_cores

    @jax.jit
    def func(*args):
      o_ref = jax.empty_ref(
          jax.ShapeDtypeStruct((num_cores, 8, 128), jnp.int32)
      )
      mesh = pltpu.create_tensorcore_mesh('core')
      @pl.core_map(mesh)
      def _():
        def run(o_vmem_ref):
          o_vmem_ref[...] = jnp.zeros_like(o_vmem_ref)
          if dynamic_grid_index is None:
            grid = grid_vals
          else:
            grid = list(grid_vals)
            grid[dynamic_grid_index] = args[0]
            grid = tuple(grid)
          mul_kernel(grid, o_vmem_ref)
          pltpu.sync_copy(o_vmem_ref, o_ref.at[jax.lax.axis_index('core')])
        pl.run_scoped(run, pltpu.VMEM((8, 128), jnp.int32))
      return jax.freeze(o_ref)

    if dynamic_grid_index is not None:
      out = func(jnp.array(grid_vals[dynamic_grid_index], jnp.int32))
    else:
      out = func()

    # Calculate expected logic (matches _partition_grid)
    parallel_dims = [
        i for i, d in enumerate(dimension_semantics) if d == pltpu.PARALLEL
    ]
    div_dims = [
        i
        for i in parallel_dims
        if isinstance(grid_vals[i], int) and grid_vals[i] % num_cores == 0
    ]

    if div_dims:
      partition_dim = min(div_dims)
    else:
      static_parallel_dims = [
          i
          for i in parallel_dims
          if isinstance(grid_vals[i], int) and grid_vals[i] > 1
      ]
      if not static_parallel_dims:
        partition_dim = parallel_dims[0]  # Must be dynamic (if mixed)
      else:
        max_val = max(grid_vals[i] for i in static_parallel_dims)
        partition_dim = min(
            i for i in static_parallel_dims if grid_vals[i] == max_val
        )

    dim_size = grid_vals[partition_dim]
    other_dims_prod = int(
        np.prod([v for i, v in enumerate(grid_vals) if i != partition_dim])
    )

    q, r = divmod(dim_size, num_cores)
    expected = np.empty((num_cores, 8, 128), jnp.int32)
    for c in range(num_cores):
      count = q + (1 if c < r else 0)
      expected[c] = count * other_dims_prod

    np.testing.assert_allclose(out, expected)

  def test_megacore_mul(self):
    x = jax.random.uniform(jax.random.key(0), (512, 512))

    def matmul_pipeline(x_ref, y_ref):
      y_ref[...] = x_ref[...] * 2

    def matmul_kernel(x_ref, y_ref):
      pltpu.emit_pipeline(
          matmul_pipeline,
          grid=(4, 4),
          in_specs=[
              pl.BlockSpec((128, 128), lambda i, j: (i, j)),
          ],
          out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
          core_axis_name='core',
          dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
      )(x_ref, y_ref)

    @jax.jit
    def func(x):
      x_ref = jax.new_ref(x)
      y_ref = jax.empty_ref(jax.ShapeDtypeStruct.like(x))
      mesh = pltpu.create_tensorcore_mesh('core')

      @pl.core_map(mesh)
      def _():
        matmul_kernel(x_ref, y_ref)

      return jax.freeze(y_ref)

    np.testing.assert_allclose(func(x), x * 2)

  @parameterized.parameters(
      (1024, 1024, 1024, 256, 512, 256),
      (768, 1024, 1024, 256, 512, 256),
      (1024, 1024, 768, 256, 512, 256),
      (768, 1024, 768, 256, 512, 256),
  )
  def test_megacore_matmul(self, m, k, n, bm, bk, bn):
    k1, k2 = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(k1, (m, k))
    y = jax.random.uniform(k2, (k, n))

    def matmul_pipeline(ps, x_ref, y_ref, z_ref):
      @pl.when(ps.index[2] == 0)
      def _():
        z_ref[...] = jnp.zeros_like(z_ref)
      z_ref[...] += x_ref[...] @ y_ref[...]

    def matmul_kernel(x_ref, y_ref, z_ref, *, bm, bk, bn):
      m, k = x_ref.shape
      _, n = y_ref.shape
      assert k % bk == 0
      pltpu.emit_pipeline(
          matmul_pipeline,
          grid=(pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk)),
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          core_axis_name='core',
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
          _explicit_indices=True,
      )(x_ref, y_ref, z_ref)

    @jax.jit
    def func(x, y):
      x_ref = jax.new_ref(x)
      y_ref = jax.new_ref(y)
      o_ref = jax.empty_ref(jax.ShapeDtypeStruct((m, n), jnp.float32))
      mesh = pltpu.create_tensorcore_mesh('core')

      @pl.core_map(mesh)
      def _():
        matmul_kernel(x_ref, y_ref, o_ref, bm=bm, bk=bk, bn=bn)

      return jax.freeze(o_ref)

    np.testing.assert_allclose(func(x, y), x @ y, atol=7e-5)

  def test_pipeline_axis_queries(self):

    mesh = pltpu.create_tensorcore_mesh('core')
    num_cores = mesh.shape['core']
    grid_size = 8

    def kernel(o_ref):
      def pipeline_body(o_smem_ref):
        pid = pl.program_id(0)
        nprogs = pl.num_programs(0)
        asize = jax.lax.axis_size('core')
        aindex = jax.lax.axis_index('core')
        o_smem_ref[0], o_smem_ref[1] = pid, nprogs
        o_smem_ref[2], o_smem_ref[3] = asize, aindex

      pltpu.emit_pipeline(
          pipeline_body,
          grid=(grid_size,),
          out_specs=pl.BlockSpec(
              (None, 128), lambda i: (i, 0), memory_space=pltpu.SMEM),
          core_axis_name='core',
          dimension_semantics=(pltpu.PARALLEL,),
      )(o_ref)

    @jax.jit
    def func():
      o_ref = jax.empty_ref(jax.ShapeDtypeStruct((grid_size, 128), jnp.int32))
      @pl.core_map(mesh)
      def _():
        kernel(o_ref)
      return jax.freeze(o_ref)

    out = func()
    out_np = np.array(out)
    local_nprogs = grid_size // num_cores
    for i in range(grid_size):
      pid, nprogs = int(out_np[i, 0]), int(out_np[i, 1])
      asize, aindex = int(out_np[i, 2]), int(out_np[i, 3])
      self.assertEqual(pid, i)
      self.assertEqual(nprogs, local_nprogs)
      self.assertEqual(asize, num_cores)
      self.assertEqual(aindex, i // local_nprogs)

@jax.jit(static_argnames=['bm', 'bk', 'bn'])
def matmul(x: jax.Array, y: jax.Array, *, bm: int, bk: int, bn: int):

  m, k = x.shape
  _, n = y.shape

  def kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref):

    grid = (pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk))

    def run(acc_scratch_ref):
      pltpu.emit_pipeline(
          functools.partial(
              basic_matmul_kernel, acc_scratch_ref=acc_scratch_ref, k=k
          ),
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          grid=grid,
          core_axis=0,
          dimension_semantics=(
              pltpu.PARALLEL,
              pltpu.PARALLEL,
              pltpu.ARBITRARY,
          ),
      )(x_hbm_ref, y_hbm_ref, o_hbm_ref)

    accum_dtype = (
        jnp.float32 if jnp.issubdtype(x.dtype, jnp.floating) else jnp.int32
    )
    pl.run_scoped(run, pltpu.VMEM((bm, bn), accum_dtype))

  num_cores = jax.devices()[0].num_cores
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      in_specs=[
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      grid=(num_cores,),
  )(x, y)

@jtu.thread_unsafe_test_class(condition=not htu.hypothesis_is_thread_safe())
class PaddedPipelineEmitterTest(htu.HypothesisShardedTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only TPU v4+ allowed.')

  @parameterized.named_parameters(
      ('float32', 'float32'), ('bfloat16', 'bfloat16'), ('int8', 'int8')
  )
  @hp.given(
      hps.integers(1, 1024),
      hps.integers(1, 1024),
      hps.integers(1, 1024),
      hps.sampled_from([8, 16, 32, 128, 256, 512]),
      hps.sampled_from([128, 256, 512]),
      hps.sampled_from([128, 256, 512]),
      hps.integers(0, 4),
  )
  def test_padded_matmul(self, dtype, m, k, n, bm, bk, bn, seed):
    if dtype == 'int8' and jtu.is_device_tpu_at_least(6):
      self.skipTest('Not implemented for TPU v6.')

    hp.assume(bm <= m)
    hp.assume(bn <= n)
    hp.assume(bk <= k)
    if dtype == 'bfloat16':
      hp.assume(bm >= 16)
    if dtype == 'int8':
      if not jtu.is_device_tpu_at_least(5):
        self.skipTest('Only TPU v5+ allowed for int8.')
      hp.assume(bm >= 32)
    k1, k2 = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(k1, (m, k), jnp.float32).astype(dtype)
    y = jax.random.normal(k2, (k, n), jnp.float32).astype(dtype)

    out = matmul(x, y, bm=bm, bk=bk, bn=bn)
    expected = x @ y
    atol = rtol = 2.3e-5
    if dtype == 'bfloat16':
      out = out.astype('float32')
      expected = expected.astype('float32')
      atol = rtol = 1e-2
    np.testing.assert_allclose(out, expected, atol=atol, rtol=rtol)


class PallasCallBoundedSliceIndexingTest(jtu.JaxTestCase):

  def test_block_spec_bounded_slice_invalid_index(self):
    if not jtu.is_device_tpu():
      self.skipTest('Only works on TPU.')
    shape = (16, 8, 128)

    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def main(refs):
      x_ref, y_ref = refs

      @pl.core_map(pltpu.create_tensorcore_mesh('core'))
      def _():
        pltpu.emit_pipeline(
            kernel,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(
                    (pl.BoundedSlice(8), 8, 128),
                    lambda i: (0, 0, 0),  # first index needs to be a pl.ds
                ),
            ),
            out_specs=pl.BlockSpec(
                (8, 8, 128),
                lambda i: (0, 0, 0),
            ),
        )(x_ref, y_ref)

    @jax.jit
    def f(x):
      y = jnp.ones((8, 8, 128), dtype=jnp.int32)
      _, y = pl.run_state(main)((x, y))
      return y
    with self.assertRaisesRegex(
        ValueError,
        'Must return a (pl\\.)?ds from the index_map for a BoundedSlice'
        ' dimension.|when it should be pl.Slice'
    ):
      f.trace(jax.ShapeDtypeStruct(shape, jnp.int32))

  def test_block_spec_bounded_slice_static(self):
    if not jtu.is_device_tpu():
      self.skipTest('Only works on TPU.')
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works on TPU v4+')
    shape = (16, 8, 128)

    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def main(refs):
      x_ref, y_ref = refs

      @pl.core_map(pltpu.create_tensorcore_mesh('core'))
      def _():
        pltpu.emit_pipeline(
            kernel,
            grid=(1,),
            in_specs=(
                pl.BlockSpec(
                    (pl.BoundedSlice(8), 8, 128),
                    lambda i: (pl.ds(4, 8), 0, 0),
                ),
            ),
            out_specs=pl.BlockSpec(
                (8, 8, 128),
                lambda i: (0, 0, 0),
            ),
        )(x_ref, y_ref)

    x = jnp.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    @jax.jit
    def f(x):
      y = jnp.ones((8, 8, 128), dtype=jnp.int32)
      _, y = pl.run_state(main)((x, y))
      return y

    out = f(x)
    np.testing.assert_allclose(out, x[4:12])

  def test_block_spec_bounded_slice_dynamic(self):
    if not jtu.is_device_tpu():
      self.skipTest('Only works on TPU.')
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works on TPU v4+')
    shape = (16, 8, 128)

    slices = jnp.array([[0, 3], [3, 8], [8, 11], [11, 16]], dtype=jnp.int32)[
        ::-1
    ]

    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    def main(refs):
      x_ref, y_ref, slices_ref = refs

      @pl.core_map(pltpu.create_tensorcore_mesh('core'))
      def _():

        @functools.partial(
            pl.run_scoped, slices_smem=pltpu.SMEM(slices.shape, slices.dtype)
        )
        def _(slices_smem):
          pltpu.sync_copy(slices_ref, slices_smem)
          def index_map(i):
            return (
                pl.ds(slices_smem[i, 0], slices_smem[i, 1] - slices_smem[i, 0]),
                0,
                0,
            )
          block_spec = pl.BlockSpec(
              (pl.BoundedSlice(16), 8, 128),
              index_map,
          )
          pltpu.emit_pipeline(
              kernel,
              grid=(slices.shape[0],),
              in_specs=(block_spec,),
              out_specs=block_spec,
          )(x_ref, y_ref)

    x = jnp.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    @jax.jit
    def f(x, slices):
      y = pl.empty_like(x)
      _, y, _ = pl.run_state(main)((x, y, slices))
      return y

    out = f(x, slices)
    np.testing.assert_allclose(out, x)


class PipelineHijaxTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works on TPU v4+.')

  def test_emit_pipeline_hijax(self):
    if config.use_emit_pipeline_primitive.value:
      self.skipTest("Hijax not yet supported by the emit_pipeline primitive.")

    @dataclasses.dataclass(frozen=True)
    class ArrayTuple:
      x0: jax.Array
      x1: jax.Array

      @property
      def shape(self):
        assert self.x0.shape == self.x1.shape
        return self.x0.shape

      @property
      def dtype(self):
        assert self.x0.dtype == self.x1.dtype
        return self.x0.dtype

    @dataclasses.dataclass(frozen=True)
    class ShapedArrayTuple(hijax.HiType):
      shape: tuple[int, ...]
      dtype: jnp.dtype

      def update(self, *, shape=None, dtype=None, **kwargs):
        changes = {}
        if shape is not None: changes['shape'] = shape
        if dtype is not None: changes['dtype'] = dtype
        return dataclasses.replace(self, **changes)

      def lo_ty(self) -> list[hijax.ShapedArray]:
        return [hijax.ShapedArray(self.shape, self.dtype)] * 2

      def lower_val(self, hi_val: ArrayTuple) -> list[jax.Array]:
        return [hi_val.x0, hi_val.x1]

      def raise_val(self, x0, x1) -> ArrayTuple:
        return ArrayTuple(x0, x1)

      def ref_get_abstract_eval(self, ref_aval, *args, tree):
        arr_aval = hijax.ShapedArray(self.shape, self.dtype)
        updated_ref = ref_aval.update(inner_aval=arr_aval)
        out, effects = state_primitives.get_p.abstract_eval(
            updated_ref, *args, tree=tree
        )
        assert isinstance(out, hijax.ShapedArray)
        return ShapedArrayTuple(out.shape, out.dtype), effects

      def ref_get_to_lojax(
          self, ref: state.TransformedRef | jax.Ref, idx: indexing.NDIndexer
      ):
        tup_ref, transforms = ref._refs, ref.transforms
        assert isinstance(transforms, tuple)
        transforms += (idx,)

        flat_transforms, tree = jax.tree.flatten(transforms)
        x0_out = state_primitives.get_p.bind(
            tup_ref.x0, *flat_transforms, tree=tree
        )
        x1_out = state_primitives.get_p.bind(
            tup_ref.x1, *flat_transforms, tree=tree
        )
        return ShapedArrayTuple(x0_out, x1_out).raise_val(x0_out, x1_out)

      def ref_swap_abstract_eval(self, ref_aval, val_aval, *args, tree):
        arr_aval = hijax.ShapedArray(self.shape, self.dtype)
        val_arr_aval = hijax.ShapedArray(val_aval.shape, val_aval.dtype)
        updated_ref = ref_aval.update(inner_aval=arr_aval)
        out_aval, effects = state_primitives.swap_p.abstract_eval(
            updated_ref, val_arr_aval, *args, tree=tree
        )
        assert isinstance(out_aval, hijax.ShapedArray)
        return ShapedArrayTuple(out_aval.shape, out_aval.dtype), effects

      def ref_swap_to_lojax(
          self,
          ref: state.TransformedRef | jax.Ref,
          val: ArrayTuple,
          idx: indexing.NDIndexer,
      ):
        tup_ref, transforms = ref._refs, ref.transforms
        assert isinstance(transforms, tuple)
        transforms += (idx,)

        flat_transforms, tree = jax.tree.flatten(transforms)
        x0_out = state_primitives.swap_p.bind(
            tup_ref.x0, val.x0, *flat_transforms, tree=tree
        )
        x1_out = state_primitives.swap_p.bind(
            tup_ref.x1, val.x1, *flat_transforms, tree=tree
        )
        return self.raise_val(x0_out, x1_out)

      def lower_block_spec(
          self, block_spec: pl.BlockSpec
      ) -> list[pl.BlockSpec]:
        return [block_spec, block_spec]

      def transform_ndindexer(
          self, idx: indexing.NDIndexer
      ) -> ShapedArrayTuple:
        x0_t = idx.transform_type(self.x0)
        x1_t = idx.transform_type(self.x1)
        return ShapedArrayTuple(x0_t, x1_t)

      def dma_start(
          self,
          src_ref: state.TransformedRef,
          dst_ref: state.TransformedRef,
          src_sem: state.TransformedRef,
          dst_sem: state.TransformedRef,
          device_id: jax.Array | int | None,
          device_id_type: pl.DeviceIdType,
          priority: int,
          add: bool,
      ) -> None:
        del add
        src_aval = jax.typeof(src_ref.ref).inner_aval
        assert isinstance(src_aval, ShapedArrayTuple)
        dst_aval = jax.typeof(dst_ref.ref).inner_aval
        assert isinstance(dst_aval, ShapedArrayTuple)

        src_ref, src_transforms = src_ref.ref._refs, src_ref.transforms
        dst_ref, dst_transforms = dst_ref.ref._refs, dst_ref.transforms

        def _run_dma(
            src_ref,
            dst_ref,
            src_sem,
            dst_sem,
            device_id,
            device_id_type,
            priority,
        ):
          if src_sem is not None:
            desc = pltpu.make_async_remote_copy(
                src_ref,
                dst_ref,
                src_sem,
                dst_sem,
                device_id=device_id,
                device_id_type=device_id_type,
            )
          else:
            assert device_id is None
            desc = pltpu.make_async_copy(src_ref, dst_ref, dst_sem)
          desc.start(priority=priority)

        src_x0_ref, src_x1_ref = src_ref.x0, src_ref.x1
        dst_x0_ref, dst_x1_ref = dst_ref.x0, dst_ref.x1

        _run_dma(
            state.TransformedRef(src_x0_ref, src_transforms),
            state.TransformedRef(dst_x0_ref, dst_transforms),
            src_sem,
            dst_sem,
            device_id,
            device_id_type,
            priority,
        )
        _run_dma(
            state.TransformedRef(src_x1_ref, src_transforms),
            state.TransformedRef(dst_x1_ref, dst_transforms),
            src_sem,
            dst_sem,
            device_id,
            device_id_type,
            priority,
        )

      def dma_wait(
          self, src_ref, dst_ref, src_sem, dst_sem, device_id, device_id_type
      ):
        assert isinstance(jax.typeof(src_ref.ref).inner_aval, ShapedArrayTuple)
        assert isinstance(jax.typeof(dst_ref.ref).inner_aval, ShapedArrayTuple)

        src_ref, src_transforms = src_ref.ref._refs, src_ref.transforms
        dst_ref, dst_transforms = dst_ref.ref._refs, dst_ref.transforms

        def _run_dma(
            src_ref, dst_ref, src_sem, dst_sem, device_id, device_id_type
        ):
          if src_sem is not None:
            desc = pltpu.make_async_remote_copy(
                src_ref,
                dst_ref,
                src_sem,
                dst_sem,
                device_id=device_id,
                device_id_type=device_id_type,
            )
          else:
            assert device_id is None
            desc = pltpu.make_async_copy(src_ref, dst_ref, dst_sem)
          desc.wait()

        src_x0_ref, src_x1_ref = src_ref.x0, src_ref.x1
        dst_x0_ref, dst_x1_ref = dst_ref.x0, dst_ref.x1

        _run_dma(
            state.TransformedRef(src_x0_ref, src_transforms),
            state.TransformedRef(dst_x0_ref, dst_transforms),
            src_sem,
            dst_sem,
            device_id,
            device_id_type,
        )
        _run_dma(
            state.TransformedRef(src_x1_ref, src_transforms),
            state.TransformedRef(dst_x1_ref, dst_transforms),
            src_sem,
            dst_sem,
            device_id,
            device_id_type,
        )

    hijax.register_hitype(
        ArrayTuple, lambda q: ShapedArrayTuple(q.shape, q.dtype)
    )
    indexing.indexer_transform_type_registry.add(ShapedArrayTuple)

    def kernel(x_hbm_ref, o_hbm_ref):
      def body(x_ref, o_ref):
        o_ref[...] = x_ref[...]

      num_steps = 4
      block_shape = (x_hbm_ref.shape[0] // num_steps, x_hbm_ref.shape[1])

      pltpu.emit_pipeline(
          body,
          grid=(num_steps,),
          in_specs=(pl.BlockSpec(block_shape, lambda i: (i, 0)),),
          out_specs=pl.BlockSpec(block_shape, lambda i: (i, 0)),
      )(x_hbm_ref, o_hbm_ref)

    inp = ArrayTuple(
        jnp.arange(32 * 128, dtype=jnp.int32).reshape((32, 128)),
        jnp.arange(32 * 128, dtype=jnp.int32).reshape((32, 128)),
    )

    out_ty = ShapedArrayTuple(
        inp.shape,
        inp.dtype,
    )

    out = pl.pallas_call(
        kernel,
        in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
        out_shape=out_ty,
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )(inp)

    np.testing.assert_allclose(out.x0, inp.x0)
    np.testing.assert_allclose(out.x1, inp.x1)


class PallasCallPipelineTransformedRefsTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5')
    super().setUp()

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs(self, dynamic_grid):
    def body(x, o):
      o[...] = x[...]

    @jax.jit
    def fn(x):
      out_type = jax.ShapeDtypeStruct(x.shape[1:], x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, o_ref):
        x_ref = x_ref.at[1, ...]
        bs = x_ref.shape[:-1] + (256,)
        idx_map = lambda i: (0,) * (x_ref.ndim - 1) + (i,)
        pltpu.emit_pipeline(
            body, grid=(jnp.array(2),) if dynamic_grid else (2,),
            in_specs=[pl.BlockSpec(bs, idx_map)],
            out_specs=[pl.BlockSpec(bs, idx_map)],
            dimension_semantics=(pltpu.ARBITRARY,),
        )(x_ref, o_ref)
      return run_kernel(x)

    x = jnp.arange(2 * 16 * 512, dtype=jnp.float32).reshape((2, 16, 512))
    np.testing.assert_allclose(fn(x), x[1, ...])

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs_nested(self, dynamic_grid):
    def fn(x):
      out_type = jax.ShapeDtypeStruct(x.shape[1:], x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, o_ref):
        x_ref = x_ref.at[1, ...]

        def outer_body(outer_x, outer_o):
          def inner_body(x, o):
            o[...] = x[...]

          pltpu.emit_pipeline(
              inner_body, grid=(jnp.array(2),) if dynamic_grid else (2,),
              in_specs=[pl.BlockSpec((16, 128), lambda i: (0, i))],
              out_specs=[pl.BlockSpec((16, 128), lambda i: (0, i))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(outer_x, outer_o)

        pltpu.emit_pipeline(
            outer_body, grid=(jnp.array(2),) if dynamic_grid else (2,),
            in_specs=[pl.BlockSpec((16, 256), lambda i: (0, i))],
            out_specs=[pl.BlockSpec((16, 256), lambda i: (0, i))],
            dimension_semantics=(pltpu.PARALLEL,),
        )(x_ref, o_ref)
      return run_kernel(x)

    x = jnp.arange(2 * 16 * 512, dtype=jnp.float32).reshape((2, 16, 512))
    np.testing.assert_allclose(fn(x), x[1, ...])

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs_nested_uneven(
      self, dynamic_grid):
    def fn(x):
      out_type = jax.ShapeDtypeStruct((256, 128), x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, o_ref):
        def outer_body(outer_x, outer_o):
          outer_o[...] = jnp.zeros(outer_o.shape, outer_o.dtype)
          tx = outer_x.at[0:128, :].at[pl.ds(0, 64), :]
          to = outer_o.at[pl.ds(0, 64), :]

          def inner_body(x, o):
            o[...] = x[...]

          pltpu.emit_pipeline(
              inner_body, grid=(jnp.array(2),) if dynamic_grid else (2,),
              in_specs=[pl.BlockSpec((32, 128), lambda i: (i, 0))],
              out_specs=[pl.BlockSpec((32, 128), lambda i: (i, 0))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(tx, to)

        pltpu.emit_pipeline(
            outer_body, grid=(jnp.array(2),) if dynamic_grid else (2,),
            in_specs=[pl.BlockSpec((256, 128), lambda i: (i, 0))],
            out_specs=[pl.BlockSpec((128, 128), lambda i: (i, 0))],
            dimension_semantics=(pltpu.PARALLEL,),
        )(x_ref, o_ref)
      return run_kernel(x)

    x = jnp.arange(512 * 128, dtype=jnp.float32).reshape((512, 128))
    expected = jnp.zeros((256, 128), dtype=jnp.float32)
    expected = expected.at[0:64, :].set(x[0:64, :])
    expected = expected.at[128:192, :].set(x[256:320, :])
    np.testing.assert_allclose(fn(x), expected)

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs_nested_bitcast(
      self, dynamic_grid):
    if not config.use_emit_pipeline_primitive.value:
      self.skipTest('Requires jax_use_emit_pipeline_primitive')

    from jax._src.state import utils as state_utils

    @jax.jit
    def fn(x):
      out_type = jax.ShapeDtypeStruct((128, 128), x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, o_ref):
        def outer_body(outer_x, outer_o):
          tx = outer_x.at[0:64, :].bitcast(jnp.int16)
          to = outer_o.bitcast(jnp.int16)
          def inner_body(x, o): o[...] = x[...]
          pltpu.emit_pipeline(
              inner_body, grid=(jnp.array(4),) if dynamic_grid else (4,),
              in_specs=[pl.BlockSpec((32, 128), lambda i: (i, 0))],
              out_specs=[pl.BlockSpec((32, 128), lambda i: (i, 0))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(tx, to)

        pltpu.emit_pipeline(
            outer_body, grid=(jnp.array(2),) if dynamic_grid else (2,),
            in_specs=[pl.BlockSpec((128, 128), lambda i: (i, 0))],
            out_specs=[pl.BlockSpec((64, 128), lambda i: (i, 0))],
            dimension_semantics=(pltpu.PARALLEL,),
        )(x_ref, o_ref)
      return run_kernel(x)

    x = jnp.arange(256 * 128, dtype=jnp.float32).reshape((256, 128))
    x_i16 = state_utils.bitcast(x, jnp.int16)
    expected_i16 = jnp.zeros((256, 128), dtype=jnp.int16)
    expected_i16 = expected_i16.at[0:128, :].set(x_i16[0:128, :])
    expected_i16 = expected_i16.at[128:256, :].set(x_i16[256:384, :])
    np.testing.assert_allclose(
        fn(x), state_utils.bitcast(expected_i16, jnp.float32))

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs_nested_3level(
      self, dynamic_grid):
    if not config.use_emit_pipeline_primitive.value:
      self.skipTest('Requires jax_use_emit_pipeline_primitive')
    @jax.jit
    def fn(x):
      out_type = jax.ShapeDtypeStruct(x.shape[1:], x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, o_ref):
        x_ref = x_ref.at[1, ...]
        def outer_body(outer_x, outer_o):
          def middle_body(middle_x, middle_o):
            def inner_body(x, o):
              o[...] = x[...]

            pltpu.emit_pipeline(
                inner_body, grid=(jnp.array(8),) if dynamic_grid else (8,),
                in_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
                out_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
                dimension_semantics=(pltpu.PARALLEL,),
            )(middle_x, middle_o)

          pltpu.emit_pipeline(
              middle_body, grid=(jnp.array(8),) if dynamic_grid else (8,),
              in_specs=[pl.BlockSpec((16, 4096), lambda i: (0, i))],
              out_specs=[pl.BlockSpec((16, 4096), lambda i: (0, i))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(outer_x, outer_o)

        pltpu.emit_pipeline(
            outer_body, grid=(jnp.array(8),) if dynamic_grid else (8,),
            in_specs=[pl.BlockSpec((16, 32768), lambda i: (0, i))],
            out_specs=[pl.BlockSpec((16, 32768), lambda i: (0, i))],
            dimension_semantics=(pltpu.PARALLEL,),
        )(x_ref, o_ref)
      return run_kernel(x)

    x = jnp.arange(2 * 16 * 262144, dtype=jnp.float32).reshape((2, 16, 262144))
    np.testing.assert_allclose(fn(x), x[1, ...])

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs_nested_hbm_passthrough(
      self, dynamic_grid
  ):

    @jax.jit
    def fn(x, y):
      out_type = jax.ShapeDtypeStruct(x.shape[1:], x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, y_ref, o_ref):
        x_ref = x_ref.at[1, ...]
        y_scratch = jax.empty_ref(
            jax.ShapeDtypeStruct((16, 512), jnp.float32), pltpu.VMEM)
        def outer_body(outer_x, outer_o):
          pltpu.sync_copy(y_ref, y_scratch)
          def inner_body(x, o): o[...] = x[...] + y_scratch[...]
          pltpu.emit_pipeline(
              inner_body, grid=(jnp.array(8),) if dynamic_grid else (8,),
              in_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
              out_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(outer_x, outer_o)

        pltpu.emit_pipeline(
            outer_body, grid=(jnp.array(8),) if dynamic_grid else (8,),
            in_specs=[pl.BlockSpec((16, 4096), lambda i: (0, i))],
            out_specs=[pl.BlockSpec((16, 4096), lambda i: (0, i))],
            dimension_semantics=(pltpu.PARALLEL,),
        )(x_ref, o_ref)

      return run_kernel(x, y)

    x = jnp.arange(2 * 16 * 32768, dtype=jnp.float32).reshape((2, 16, 32768))
    y = jnp.ones((16, 512), dtype=jnp.float32)
    np.testing.assert_allclose(fn(x, y), x[1, ...] + 1)

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs_nested_mixed_grids(
      self, dynamic_grid):

    @jax.jit
    def fn(x):
      out_type = jax.ShapeDtypeStruct(x.shape[1:], x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, o_ref):
        x_ref = x_ref.at[1, ...]
        def outer_body(outer_x, outer_o):
          def inner_body(x, o):
            o[...] = x[...]

          pltpu.emit_pipeline(
              inner_body, grid=(jnp.array(8),) if dynamic_grid else (8,),
              in_specs=[pl.BlockSpec((32, 512), lambda i: (0, i))],
              out_specs=[pl.BlockSpec((32, 512), lambda i: (0, i))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(outer_x, outer_o)

        pltpu.emit_pipeline(
            outer_body, grid=(jnp.array(8), jnp.array(8)) if dynamic_grid else (8, 8),
            in_specs=[pl.BlockSpec((32, 4096), lambda i, j: (i, j))],
            out_specs=[pl.BlockSpec((32, 4096), lambda i, j: (i, j))],
            dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
        )(x_ref, o_ref)
      return run_kernel(x)

    x = jnp.arange(2 * 256 * 32768, dtype=jnp.float32).reshape((2, 256, 32768))
    np.testing.assert_allclose(fn(x), x[1, ...])

  @parameterized.product(dynamic_grid=[True, False])
  def test_transformed_refs_as_inputs_nested_sequential(
      self, dynamic_grid):

    @jax.jit
    def fn(x):
      out_type = jax.ShapeDtypeStruct(x.shape[1:], x.dtype)

      @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
      def run_kernel(x_ref, o_ref):
        x_ref = x_ref.at[1, ...]

        def outer_body(outer_x, outer_o):
          tmp = jax.empty_ref(
              jax.ShapeDtypeStruct((16, 4096), jnp.float32), pltpu.VMEM)

          def inner_body_1(x, o):
            o[...] = x[...] * 2

          def inner_body_2(x, o):
            o[...] = x[...] + 1

          pltpu.emit_pipeline(
              inner_body_1, grid=(jnp.array(8),) if dynamic_grid else (8,),
              in_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
              out_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(outer_x, tmp)
          pltpu.emit_pipeline(
              inner_body_2, grid=(jnp.array(8),) if dynamic_grid else (8,),
              in_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
              out_specs=[pl.BlockSpec((16, 512), lambda i: (0, i))],
              dimension_semantics=(pltpu.PARALLEL,),
          )(tmp, outer_o)
        pltpu.emit_pipeline(
            outer_body, grid=(jnp.array(8),) if dynamic_grid else (8,),
            in_specs=[pl.BlockSpec((16, 4096), lambda i: (0, i))],
            out_specs=[pl.BlockSpec((16, 4096), lambda i: (0, i))],
            dimension_semantics=(pltpu.PARALLEL,),
        )(x_ref, o_ref)
      return run_kernel(x)

    x = jnp.arange(2 * 16 * 32768, dtype=jnp.float32).reshape((2, 16, 32768))
    np.testing.assert_allclose(fn(x), x[1, ...] * 2 + 1)


class PallasCallPipelineNonFlatArgsTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5 or newer.')
    super().setUp()

  @parameterized.product(dynamic_grid=[True, False])
  def test_non_flat_args(self, dynamic_grid):
    out_type = [(), (jax.ShapeDtypeStruct((512,), jnp.float32),
                    jax.ShapeDtypeStruct((512,), jnp.float32))]
    x_shape = (512,)

    def body(in_refs, out_refs):
      out_refs[1][0][...] = in_refs[0][0][...] + in_refs[2][0][...]
      out_refs[1][1][...] = in_refs[2][1][...]

    @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
    def run_kernel(x_ref, y_ref, z_ref, out_xy_ref, out_z_ref):
      in_specs = ([(pl.BlockSpec((256,), lambda i: (i,)),), (),
                  (pl.BlockSpec((256,), lambda i: (i,)),
                    pl.BlockSpec((256,), lambda i: (i,)))],)
      out_specs = ( [(), (pl.BlockSpec((256,), lambda i: (i,)),
                          pl.BlockSpec((256,), lambda i: (i,)))], )
      pltpu.emit_pipeline(
          body, grid=(jnp.array(2,),) if dynamic_grid else (2,),
          in_specs=in_specs, out_specs=out_specs,
      )([(x_ref,), (), (y_ref, z_ref)], [out_xy_ref, out_z_ref])

    x = jnp.arange(np.prod(x_shape), dtype=jnp.float32).reshape(x_shape)
    out = run_kernel(x, x * 2, x * 3)
    self.assertLen(out, 2)
    self.assertEqual(out[0], ())
    np.testing.assert_allclose(out[1][0], x + x * 2)
    np.testing.assert_allclose(out[1][1], x * 3)

  @parameterized.product(dynamic_grid=[True, False])
  def test_non_flat_args_and_scratch(self, dynamic_grid):

    out_type = [(), (jax.ShapeDtypeStruct((512,), jnp.float32),
                     jax.ShapeDtypeStruct((512,), jnp.float32))]
    x_shape = (512,)

    def body(in_refs, out_refs):
      scratch1 = jax.empty_ref(
          jax.ShapeDtypeStruct((256,), jnp.float32), memory_space=pltpu.VMEM
      )
      scratch2 = jax.empty_ref(
          jax.ShapeDtypeStruct((256,), jnp.float32), memory_space=pltpu.VMEM
      )
      scratch2_list = [scratch2]
      scratch1[...] = in_refs[0][0][...] + in_refs[2][0][...]
      out_refs[1][0][...] = scratch1[...]
      scratch2_list[0][...] = in_refs[2][1][...]
      out_refs[1][1][...] = scratch2_list[0][...]

    @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
    def run_kernel(x_ref, y_ref, z_ref, out_xy_ref, out_z_ref):
      in_specs = ([(pl.BlockSpec((256,), lambda i: (i,)),), (),
                   (pl.BlockSpec((256,), lambda i: (i,)),
                    pl.BlockSpec((256,), lambda i: (i,)))],)
      out_specs = ( [(), (pl.BlockSpec((256,), lambda i: (i,)),
                          pl.BlockSpec((256,), lambda i: (i,)))],)
      pltpu.emit_pipeline(
          body, grid=(jnp.array(2,),) if dynamic_grid else (2,),
          in_specs=in_specs, out_specs=out_specs,
      )([(x_ref,), (), (y_ref, z_ref)], [out_xy_ref, out_z_ref])

    x = jnp.arange(np.prod(x_shape), dtype=jnp.float32).reshape(x_shape)
    out = run_kernel(x, x * 2, x * 3)
    self.assertLen(out, 2)
    self.assertEqual(out[0], ())
    np.testing.assert_allclose(out[1][0], x + x * 2)
    np.testing.assert_allclose(out[1][1], x * 3)

  @parameterized.product(dynamic_grid=[True, False])
  def test_non_flat_args_nested(self, dynamic_grid):
    out_type = [(), (jax.ShapeDtypeStruct((512,), jnp.float32),
                     jax.ShapeDtypeStruct((512,), jnp.float32))]

    def outer_body(outer_in, outer_out):
      def inner_body(in_refs, out_refs):
        out_refs[1][0][...] = in_refs[0][0][...] + in_refs[2][0][...]
        out_refs[1][1][...] = in_refs[2][1][...]

      pltpu.emit_pipeline(
          inner_body, grid=(jnp.array(2,),) if dynamic_grid else (2,),
          in_specs=([ (pl.BlockSpec((128,), lambda i: (i,)),), (),
                      (pl.BlockSpec((128,), lambda i: (i,)),
                       pl.BlockSpec((128,), lambda i: (i,))) ],),
          out_specs=([ (), (pl.BlockSpec((128,), lambda i: (i,)),
                            pl.BlockSpec((128,), lambda i: (i,))) ],),
      )(outer_in, outer_out)

    @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
    def run_kernel(x_ref, y_ref, z_ref, out_xy_ref, out_z_ref):
      in_specs = ([(pl.BlockSpec((256,), lambda i: (i,)),), (),
                   (pl.BlockSpec((256,), lambda i: (i,)),
                    pl.BlockSpec((256,), lambda i: (i,)))],)
      out_specs = ( [(), (pl.BlockSpec((256,), lambda i: (i,)),
                          pl.BlockSpec((256,), lambda i: (i,)))], )

      pltpu.emit_pipeline(
          outer_body, grid=(jnp.array(2,),) if dynamic_grid else (2,),
          in_specs=in_specs, out_specs=out_specs,
      )([(x_ref,), (), (y_ref, z_ref)], [out_xy_ref, out_z_ref])

    x = jnp.arange(512, dtype=jnp.float32)
    out = run_kernel(x, x * 2, x * 3)
    self.assertLen(out, 2)
    self.assertEqual(out[0], ())
    np.testing.assert_allclose(out[1][0], x + x * 2)
    np.testing.assert_allclose(out[1][1], x * 3)

  @parameterized.product(dynamic_grid=[True, False])
  def test_non_flat_args_and_scratch_nested(self, dynamic_grid):

    out_type = [(), (jax.ShapeDtypeStruct((512,), jnp.float32),
                     jax.ShapeDtypeStruct((512,), jnp.float32))]

    def outer_body(outer_in, outer_out):
      outer_scratch1 = jax.empty_ref(
          jax.ShapeDtypeStruct((256,), jnp.float32), memory_space=pltpu.VMEM
      )
      outer_scratch2 = jax.empty_ref(
          jax.ShapeDtypeStruct((256,), jnp.float32), memory_space=pltpu.VMEM
      )
      outer_scratch2_list = [outer_scratch2]
      outer_scratch1[...] = outer_in[0][0][...] + outer_in[2][0][...]
      outer_scratch2_list[0][...] = outer_in[2][1][...]

      def inner_body(in_refs, out_refs):
        inner_scratch = jax.empty_ref(
            jax.ShapeDtypeStruct((128,), jnp.float32), memory_space=pltpu.VMEM
        )
        inner_scratch[...] = in_refs[0][0][...]
        out_refs[1][0][...] = inner_scratch[...]
        out_refs[1][1][...] = in_refs[2][0][...]

      pltpu.emit_pipeline(
          inner_body, grid=(jnp.array(2,),) if dynamic_grid else (2,),
          in_specs=([ (pl.BlockSpec((128,), lambda i: (i,)),), (),
                      (pl.BlockSpec((128,), lambda i: (i,)),) ],),
          out_specs=([ (), (pl.BlockSpec((128,), lambda i: (i,)),
                            pl.BlockSpec((128,), lambda i: (i,))) ],),
      )([(outer_scratch1,), (), (outer_scratch2_list[0],)], outer_out)

    @pl.kernel(out_type=out_type, mesh=pltpu.create_tensorcore_mesh('core'))
    def run_kernel(x_ref, y_ref, z_ref, out_xy_ref, out_z_ref):
      in_specs = ([(pl.BlockSpec((256,), lambda i: (i,)),), (),
                   (pl.BlockSpec((256,), lambda i: (i,)),
                    pl.BlockSpec((256,), lambda i: (i,)))],)
      out_specs = ( [(), (pl.BlockSpec((256,), lambda i: (i,)),
                          pl.BlockSpec((256,), lambda i: (i,)))], )
      pltpu.emit_pipeline(
          outer_body, grid=(jnp.array(2,),) if dynamic_grid else (2,),
          in_specs=in_specs, out_specs=out_specs,
      )([(x_ref,), (), (y_ref, z_ref)], [out_xy_ref, out_z_ref])

    x = jnp.arange(512, dtype=jnp.float32)
    out = run_kernel(x, x * 2, x * 3)
    self.assertLen(out, 2)
    self.assertEqual(out[0], ())
    np.testing.assert_allclose(out[1][0], x + x * 2)
    np.testing.assert_allclose(out[1][1], x * 3)


class PallasCallPipelineEffectsTest(jtu.JaxTestCase):
  def setUp(self):
    self.enter_context(config.use_emit_pipeline_primitive(True))
    super().setUp()

  def test_basic_pipeline_effects(self):
    def body(x, y, o):
      o[...] = x[...] + y[...]

    def kernel(x_ref, y_ref, o_ref):
      pltpu.emit_pipeline(
          body, grid=(2,),
          in_specs=[pl.BlockSpec((256,), lambda i: (i,)), pl.BlockSpec((256,), lambda i: (i,))],
          out_specs=[pl.BlockSpec((256,), lambda i: (i,))],
      )(x_ref, y_ref, o_ref)

    jaxpr = jax.make_jaxpr(kernel)(
        state.shaped_array_ref((512,), jnp.float32),
        state.shaped_array_ref((512,), jnp.float32),
        state.shaped_array_ref((512,), jnp.float32),
    )
    x, y, o = jaxpr.jaxpr.invars
    expected_effects = {state.ReadEffect(x), state.ReadEffect(y),
                        state.WriteEffect(o)}
    self.assertSetEqual(jaxpr.effects, expected_effects)

  def test_dynamic_grid_and_consts_effects(self):
    c = jnp.array(1.5, dtype=jnp.float32)
    def body(x, o):
      o[...] = x[...] * c

    def kernel(d_grid, x_ref, o_ref):
      pltpu.emit_pipeline(
          body, grid=(d_grid,),
          in_specs=[pl.BlockSpec((256,), lambda i: (i,))],
          out_specs=[pl.BlockSpec((256,), lambda i: (i,))],
      )(x_ref, o_ref)

    jaxpr = jax.make_jaxpr(kernel)(
        jax.ShapeDtypeStruct((), jnp.int32),
        state.shaped_array_ref((512,), jnp.float32),
        state.shaped_array_ref((512,), jnp.float32),
    )
    _, x, o = jaxpr.jaxpr.invars
    expected_effects = {state.ReadEffect(x), state.WriteEffect(o)}
    self.assertSetEqual(jaxpr.effects, expected_effects)

  def test_transformed_refs_effects(self):
    def body(x, o):
      o[...] = x[...]

    def kernel(x_ref, o_ref):
      x_slice = x_ref.at[1, ...]
      pltpu.emit_pipeline(
          body, grid=(2,),
          in_specs=[pl.BlockSpec((256,), lambda i: (i,))],
          out_specs=[pl.BlockSpec((256,), lambda i: (i,))],
      )(x_slice, o_ref)

    jaxpr = jax.make_jaxpr(kernel)(
        state.shaped_array_ref((2, 512), jnp.float32),
        state.shaped_array_ref((512,), jnp.float32),
    )
    x, o = jaxpr.jaxpr.invars
    expected_effects = {state.ReadEffect(x), state.WriteEffect(o)}
    self.assertSetEqual(jaxpr.effects, expected_effects)


# Temporary subclasses for testing the primitive code path for emit_pipeline.
# TODO(rdyro): Remove once the primitive is the default.
class PallasCallPipelinePrimitiveTest(PallasCallPipelineTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


class PallasCallPipelinePoisonPrimitiveTest(PallasCallPipelinePoisonTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


class PallasCallMultipleBufferedPipelinePrimitiveTest(
    PallasCallMultipleBufferedPipelineTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


class PallasCallMegacorePrimitiveTest(PallasCallMegacoreTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


@jtu.thread_unsafe_test_class(condition=not htu.hypothesis_is_thread_safe())
class PaddedPipelineEmitterPrimitiveTest(PaddedPipelineEmitterTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


class PallasCallBoundedSliceIndexingPrimitiveTest(
    PallasCallBoundedSliceIndexingTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


class PipelineHijaxPrimitiveTest(PipelineHijaxTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


class PallasCallPipelineTransformedRefsPrimitiveTest(
    PallasCallPipelineTransformedRefsTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


class PallasCallPipelineNonFlatArgsPrimitiveTest(
    PallasCallPipelineNonFlatArgsTest):
  def setUp(self):
    super().setUp()
    self.enter_context(config.use_emit_pipeline_primitive(True))


if __name__ == '__main__':
  absltest.main(testLoader=htu.HypothesisShardedTestLoader())
