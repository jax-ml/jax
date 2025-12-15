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

import dataclasses
import functools
from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax import lax
from jax._src import hijax
from jax._src import shard_map
from jax._src import state
from jax._src import test_util as jtu
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax.experimental import mesh_utils
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
P = jax.sharding.PartitionSpec
partial = functools.partial


def mod(a, n):
  return lax.rem(a + n, n)


def make_ds(idx, stride):
  return pl.ds(pl.multiple_of(idx * stride, stride), stride)


def _grid_size(grid):
  size = jnp.array(1, jnp.int32)
  for dim in grid:
    size *= dim
  return size


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


class PallasCallPipelineTest(parameterized.TestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5')

    super().setUp()

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
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
    )()
    np.testing.assert_allclose(out, jnp.full_like(out, 42))

  @parameterized.product(
      no_pipelining=[False, True],
  )
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
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
    )

    jax.block_until_ready(z(x, y))
    jax.block_until_ready(jnp.dot(x, y))

    out = jax.block_until_ready(z(x, y))
    expected_out = jax.block_until_ready(jnp.dot(x, y))

    np.testing.assert_allclose(out, expected_out, atol=5e-5)

  @parameterized.named_parameters(
      ('vmem', pltpu.VMEM),
      ('hbm', pltpu.ANY),
  )
  def test_double_pipeline_matmul(self, memory_space):
    # TODO(b/358121809): Re-enable this test once the bug is fixed.
    self.skipTest('Broken test.')
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (512, 512))
    y = jax.random.uniform(k2, (512, 512))

    def matmul_pipeline(x_ref, y_ref, z_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        z_ref[...] = jnp.zeros(z_ref.shape, jnp.float32)

      z_ref[...] += x_ref[...] @ y_ref[...]

    def matmul_kernel(x_ref, y_ref, z_ref):

      def emit_pipeline(should_accumulate_out):
        pltpu.emit_pipeline(
            matmul_pipeline,
            grid=(4, 4, 4),
            in_specs=[
                pl.BlockSpec((128, 128), lambda i, j, k: (i, k)),
                pl.BlockSpec((128, 128), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((128, 128), lambda i, j, k: (i, j)),
            should_accumulate_out=should_accumulate_out,
        )(x_ref, y_ref, z_ref)

      emit_pipeline(False)
      emit_pipeline(True)

    z = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((512, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=memory_space),
            pl.BlockSpec(memory_space=memory_space),
        ],
        out_specs=pl.BlockSpec(memory_space=memory_space),
    )(x, y)

    np.testing.assert_allclose(z, jnp.dot(x, y) + jnp.dot(x, y))


class PallasCallMultipleBufferedPipelineTest(parameterized.TestCase):

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
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
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
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
    )
    result = fn(x, y)
    np.testing.assert_allclose(result, x @ y, atol=5e-5)

  @parameterized.product(
      x_buffer_count=[2, 4],
      y_buffer_count=[2, 4],
      out_buffer_count=[2],
  )
  def test_matmul_megacore(self,
                           x_buffer_count, y_buffer_count, out_buffer_count):
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
          core_axis=0,
          grid=(512 // block_shape[0], 512 // block_shape[0], 512 // block_shape[0]),
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
          in_specs=[
              pl.BlockSpec(block_shape, lambda i, j, k: (i, k),
                pipeline_mode=pl.Buffered(buffer_count=x_buffer_count)),
              pl.BlockSpec(block_shape, lambda i, j, k: (k, j),
                pipeline_mode=pl.Buffered(buffer_count=y_buffer_count)),
          ],
          out_specs=pl.BlockSpec(block_shape, lambda i, j, k: (i, j),
            pipeline_mode=pl.Buffered(buffer_count=out_buffer_count)),
      )(x_hbm_ref, y_hbm_ref, o_hbm_ref)
    fn = pl.pallas_call(
        matmul_kernel,
        grid=(2,),
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(pltpu.PARALLEL,)
        ),
    )
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
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
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
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
    )
    result = fn(x, jnp.array(out_block_indices))

    expected = [jnp.zeros((128, 128), jnp.int32)] * 8
    for i, blk_idx in enumerate(out_block_indices):
      expected[blk_idx] = x[i * block_size:(i + 1) * block_size, :]
    expected = jnp.concatenate(expected, axis=0)
    np.testing.assert_allclose(result, expected)

  @parameterized.product(
      in_buffer_count=[2, 4],
      out_buffer_count=[2],

  )
  def test_copy_with_multiple_cycles(self, in_buffer_count, out_buffer_count):
    x = jnp.reshape(jnp.arange(512 * 512), (512, 512))
    def copy_kernel(x_hbm_ref, o_hbm_ref):
      def inner_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      pipeline_fn, make_allocations = pltpu.emit_pipeline_with_allocations(
          inner_kernel,
          grid=(2, 4),
          in_specs=[
              pl.BlockSpec((128, 128), lambda i, j: (i, j),
                pipeline_mode=pl.Buffered(buffer_count=in_buffer_count)),
          ],
          out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j),
            pipeline_mode=pl.Buffered(buffer_count=out_buffer_count)),
      )
      def prefetch(x_bref, o_bref, scheduler):
        del o_bref
        # Prefetch will use a 0, 0 index so we need to slice x_hbm_ref
        scheduler.prefetch(x_bref, x_hbm_ref.at[256:, :])

      @functools.partial(pl.run_scoped,
        allocations=make_allocations(x_hbm_ref,
                                     o_hbm_ref,
                                     should_accumulate_out=(False,)))
      def _(allocations):
        pipeline_fn(x_hbm_ref.at[:256, :], o_hbm_ref.at[:256, :],
                allocations=allocations,
                first_cycle=True,
                last_cycle=False,
                prefetch=prefetch,
                postyeet=None,
        )
        pipeline_fn(x_hbm_ref.at[256:, :], o_hbm_ref.at[256:, :],
                allocations=allocations,
                first_cycle=False,
                last_cycle=True,
                prefetch=None,
                postyeet=None,
        )
    fn = pl.pallas_call(
        copy_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
    )
    result = fn(x)
    np.testing.assert_allclose(result, x)

  @parameterized.product(
      in_buffer_count=[2, 4],
      out_buffer_count=[2],
      in_block_indices=[
          [0, 1, 2, 3, 4, 5],
          [2, 2, 2, 2, 2, 2],
          [0, 0, 7, 7, 4, 4],
          [3, 3, 7, 7, 5, 3, 3],
          [5],
      ],
      use_lookahead=[True, False],
  )
  def test_block_gather_with_multiple_cycles(
      self, in_block_indices, in_buffer_count, out_buffer_count, use_lookahead
  ):
    # Exercises pipeline with repeated input block indices.
    block_size = 128
    x = jnp.reshape(jnp.arange(1024 * 128), (1024, 128))
    blk_len = len(in_block_indices)

    def copy_kernel(x_hbm_ref, blk_indices_ref, o_hbm_ref, blk_idx_offset):
      blk_idx_offset[0] = 0
      def inner_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      def x_index_map(i):
        return (blk_indices_ref[i], 0)
      pipeline_fn, make_allocations = pltpu.emit_pipeline_with_allocations(
          inner_kernel,
          grid=(blk_len,),
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
              lambda i: (i + blk_idx_offset[0], 0),
              pipeline_mode=pl.Buffered(buffer_count=out_buffer_count),
          ),
      )
      def prefetch(x_bref, o_bref, scheduler):
        del o_bref
        scheduler.prefetch(x_bref, x_hbm_ref)

      @functools.partial(pl.run_scoped,
        allocations=make_allocations(x_hbm_ref,
                                     o_hbm_ref,
                                     should_accumulate_out=(False,)))
      def _(allocations):
        pipeline_fn(x_hbm_ref, o_hbm_ref,
                allocations=allocations,
                first_cycle=True,
                last_cycle=False,
                prefetch=prefetch,
                postyeet=None,
        )
        blk_idx_offset[0] = blk_len
        pipeline_fn(x_hbm_ref, o_hbm_ref,
                allocations=allocations,
                first_cycle=False,
                last_cycle=True,
                prefetch=None,
                postyeet=None,
        )
    fn = pl.pallas_call(
        copy_kernel,
        out_shape=jax.ShapeDtypeStruct((blk_len * 2 * 128, 128), jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        scratch_shapes = [pltpu.SMEM((1,), dtype=jnp.int32)]
    )
    result = jax.block_until_ready(fn(x, jnp.array(in_block_indices)))

    expected = []
    for blk_idx in [*in_block_indices, *in_block_indices]:
      x_block = x[blk_idx * block_size:(blk_idx + 1) * block_size, :]
      expected.append(x_block)
    expected = jax.block_until_ready(jnp.concatenate(expected, axis=0))
    np.testing.assert_allclose(result, expected)

  @parameterized.product(
      in_buffer_count=[2, 4],
  )
  def test_pipeline_with_accumulator(self, in_buffer_count):
    x = jnp.reshape(jnp.arange(1024 * 128), (1024, 128)) // (128*128)
    accum_schedule = pltpu.get_pipeline_schedule('fixed')
    def copy_kernel(x_hbm_ref, o_hbm_ref):
      def inner_kernel(x_ref, o_ref):
        @pl.when(pl.program_id(0) == 0)
        def _():
          o_ref[...] = jnp.zeros_like(o_ref)
        o_ref[...] += x_ref[...]
      pipeline_fn, make_allocations = pltpu.emit_pipeline_with_allocations(
          inner_kernel,
          grid=(4,),
          in_specs=[
              pl.BlockSpec(
                  (128, 128),
                  lambda i: (i, 0),
                  pipeline_mode=pl.Buffered(buffer_count=in_buffer_count),
              ),
          ],
          out_specs=pl.BlockSpec((128, 128), lambda i: (0, 0)),
          should_accumulate_out=True,
      )
      def prefetch(x_bref, o_bref, scheduler):
        del o_bref
        # Prefetch will use a 0, 0 index so we need to slice x_hbm_ref
        scheduler.prefetch(x_bref, x_hbm_ref.at[512:, :])

      @functools.partial(pl.run_scoped,
        allocations=make_allocations(x_hbm_ref,
                                     o_hbm_ref,
                                     should_accumulate_out=(True,)))
      def _(allocations):
        pipeline_fn(x_hbm_ref.at[:512, :], o_hbm_ref,
                allocations=allocations,
                first_cycle=True,
                last_cycle=False,
                prefetch=prefetch,
                postyeet=None,
                init_accumulators=True,
                schedule=(None, accum_schedule)
        )
        pipeline_fn(x_hbm_ref.at[512:, :], o_hbm_ref,
                allocations=allocations,
                first_cycle=False,
                last_cycle=True,
                prefetch=None,
                postyeet=None,
                init_accumulators=False,
                schedule=(None, accum_schedule)
        )
    fn = pl.pallas_call(
        copy_kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
    )
    result = fn(x)
    expected = 0
    for i in range(x.shape[0] // 128):
      x_blk = x[i * 128:(i + 1) * 128, :]
      expected += x_blk
    np.testing.assert_allclose(result, expected)

  def test_matmul_with_input_output(self):
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
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        scratch_shapes=inner_allocs,
    )
    result = fn(x, y)
    np.testing.assert_allclose(result, x @ y, atol=5e-5)


class PallasCallCollectivePipelineTest(parameterized.TestCase):

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

  @parameterized.named_parameters(
      ('vmem', pltpu.VMEM, jnp.bfloat16, 2, 2, 2),
      ('hbm', pltpu.ANY, jnp.bfloat16, 2, 2, 2),
      ('hbm_float32', pltpu.ANY, jnp.float32, 2, 2, 2),
      ('hbm_float32_112', pltpu.ANY, jnp.float32, 1, 1, 2),
      ('hbm_float32_111', pltpu.ANY, jnp.float32, 1, 1, 1),
  )
  def test_pipeline_latency_optimized_allgather_matmul(
      self, memory_space, out_dtype, n_tiles, m_tiles, k_tiles):
    input_dtype = jnp.float32

    num_devices = jax.local_device_count()

    tn = 128 * 1
    tm = 128 * 1
    tk = 128 * 1
    n = tn * n_tiles
    m = tm * m_tiles
    k = tk * k_tiles * num_devices

    outer_steps = num_devices // 2

    sharded_k = k // num_devices
    inner_grid = (n // tn, m // tm, sharded_k // tk)

    inner_kernel = partial(basic_matmul_kernel, k=sharded_k)

    inner_allocs = [
        pltpu.BufferedRef.input(
            pl.BlockSpec((tm, tk), lambda n, m, k: (m, k)), input_dtype),
        pltpu.BufferedRef.input(
            pl.BlockSpec((tk, tn), lambda n, m, k: (k, n)), input_dtype),
        pltpu.BufferedRef.accumulator(
            pl.BlockSpec((tm, tn), lambda n, m, k: (m, n)), out_dtype),
        ]

    def all_gather_lhs_matmul_kernel(
        # in refs:
        lhs_ref,  # [m, sharded_k]
        rhs_ref,  # [k, n]
        # out refs:
        out_ref,  # [m, n]
        # used as scratch but defined as an output so that it's resident in HBM:
        lhs_scratch_ref,  # [2 (bwd/fwd), 2 (work/buffer), m, sharded_k]
        # scratch refs:
        acc_scratch_ref,  # [tm, tn]
        bwd_recv_sem,
        bwd_send_sem,
        fwd_recv_sem,
        fwd_send_sem,
        lhs_bref,
        rhs_bref,
        out_bref
    ):
      # Outer Pipeline
      # Coordinates collective rDMAs and prefetching around the inner compute
      # pipeline.

      # Given shapes:
      #   lhs: A sharded 2d, jnp.ndarray with shape [m, k // axis_size].
      #   rhs: A 2d, jnp.ndarray with shape [k, n].
      # Results:
      #   out: jnp.ndarray with shape [m, n].

      # A bidirectional collective allgather-matmul sends two "streams" of LHS
      # chunks in the forward and backward directions.  These are matmul'd with
      # their corresponding chunks in the RHS contracting dimension and added to
      # a running accumulator.

      # We run the computation in N / 2 steps with 2 "phases" per step:
      # - phase 0: in which we compute the backwards stream matmul.
      # - phase 1: in which we compute the forward stream matmul.

      # In the prologue we initialize the backwards stream by using the local
      # LHS shard, and the forwards stream by sending the local LHS shard
      # "right" along the contractive sharding axis.

      # At each step afterwards we roll the fwd copies right, and the bwd copies
      # left via rDMAs that are overlapped with matmuls.

      # At step n, phase p, we compute the following:
      # let:
      #  idx = (axis_index + step) if p == 0 else (axis_index - step - 1)
      #  contraction_slice = idx * (k // axis_size) : (idx+1) * (k // axis_size)
      #  out[m, n] += lhs[m, contraction_slice] @ rhs[contraction_slice, n]

      # Where LHS slices are the corresponding shards passed along the "fwd"
      # and "bwd" streams, and RHS is sliced directly from an unsharded array.

      outer_step = pl.program_id(0)  # range [0, steps-1]
      phase = pl.program_id(1)  # range [0, 1]  0 == BWD Matmul, 1 == FWD Matmul

      # kernel start / end booleans
      is_start = jnp.logical_and(outer_step == 0, phase == 0)
      is_end = jnp.logical_and(outer_step == outer_steps - 1, phase == 1)

      # slots for double-buffered LHS scratch accumulator
      # at each sub-step, working slot --> buffering slot
      working_slot = lax.rem(outer_step, 2)
      buffering_slot = 1 - working_slot

      # IDs of self and neighbors
      my_id = lax.axis_index('x')
      right_neighbor = mod(my_id + 1, num_devices)
      left_neighbor = mod(my_id - 1, num_devices)

      # Async copy definitions.

      # NB: The send semaphore is what the sender uses to wait until the
      # destination buffer is free. The recv semaphore is only readable by the
      # destination core to wait until all data has arrived. (The completion of
      # these sync flags can be multiple microseconds apart.)  async wait()
      # calls will only unblock after both for remote copies.

      # Initialize backwards stream by transfer of LHS chunks into local
      # working copies.
      initial_bwd_copy = pltpu.make_async_copy(
          lhs_ref,
          lhs_scratch_ref.at[0, working_slot],
          bwd_send_sem,
      )

      # Initialize forwards stream by transfer of initial LHS chunks to right
      # neighbors' working copies.
      initial_fwd_copy = pltpu.make_async_remote_copy(
          src_ref=lhs_ref,
          dst_ref=lhs_scratch_ref.at[1, working_slot],
          send_sem=fwd_send_sem,
          recv_sem=fwd_recv_sem,
          device_id=right_neighbor,
      )

      # Transfer working copies of LHS chunks backwards to left neighbors'
      # buffering copies.
      bwd_copy = pltpu.make_async_remote_copy(
          src_ref=lhs_scratch_ref.at[0, working_slot],
          dst_ref=lhs_scratch_ref.at[0, buffering_slot],
          send_sem=bwd_send_sem,
          recv_sem=bwd_recv_sem,
          device_id=left_neighbor,
      )

      # Transfer working copies of LHS chunks forwards to right neighbors'
      # buffering copies.
      fwd_copy = pltpu.make_async_remote_copy(
          src_ref=lhs_scratch_ref.at[1, working_slot],
          dst_ref=lhs_scratch_ref.at[1, buffering_slot],
          send_sem=fwd_send_sem,
          recv_sem=fwd_recv_sem,
          device_id=right_neighbor,
      )

      # Slice RHS to match LHS slices in bwd/fwd phases for contractions.
      def get_rhs_slice(outer_step, phase):
        bwd_rhs_offset = mod(my_id + outer_step, num_devices)
        fwd_rhs_offset = mod(my_id - outer_step - 1, num_devices)
        offset = jnp.where(phase, fwd_rhs_offset, bwd_rhs_offset)
        return pl.ds(
            pl.multiple_of(offset * sharded_k, sharded_k),
            sharded_k,
        )

      # Fixed Ref schedule, only really needed to prevent HBM data race in the
      # degenerate case of a trivial (single-step) inner loop.
      accum_schedule = pltpu.get_pipeline_schedule('fixed')

      # Outer loop prologue
      @pl.when(is_start)
      @jax.named_scope('sync_and_bwd_init')
      def _sync_and_bwd_init():
        # barrier at start
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
        pltpu.semaphore_signal(barrier_sem, device_id=right_neighbor)
        pltpu.semaphore_wait(barrier_sem, 2)

        # initializing copies
        initial_bwd_copy.start()
        initial_fwd_copy.start()
        initial_bwd_copy.wait()

      @pl.when(jnp.logical_and(outer_step != outer_steps - 1, phase == 0))
      @jax.named_scope('send_next_dma')
      def _send_next_dma():
        bwd_copy.start()
        @pl.when(jnp.logical_not(is_start))
        def _send_next_fwd_dma():
          fwd_copy.start()

      # Cross-loop prefetch
      def prefetch(lhs_bref, rhs_bref, out_bref, scheduler):
        @pl.when(is_start)
        @jax.named_scope('fwd_init')
        def _fwd_init():
          initial_fwd_copy.wait()
          fwd_copy.start()

        @pl.when(jnp.logical_and(outer_step != outer_steps - 1, phase == 1))
        @jax.named_scope('wait_on_prev_dma')
        def _wait_on_prev_dma():
          bwd_copy.wait()
          fwd_copy.wait()

        # prefetch next loop's inputs
        prefetch_working_slot = jnp.where(
            phase == 0, working_slot, buffering_slot)
        prefetch_step = jnp.where(phase == 0, outer_step, outer_step + 1)
        prefetch_phase = lax.rem(phase + 1, 2)
        scheduler.prefetch(
            lhs_bref, lhs_scratch_ref.at[prefetch_phase, prefetch_working_slot])
        scheduler.prefetch(
            rhs_bref, rhs_ref.at[get_rhs_slice(prefetch_step, prefetch_phase)])
        scheduler.prefetch(out_bref, out_ref, accum_schedule)

      pltpu.emit_pipeline(inner_kernel, grid=inner_grid)(
            lhs_scratch_ref.at[phase, working_slot],
            rhs_ref.at[get_rhs_slice(outer_step, phase)],
            out_ref,
            allocations=[lhs_bref, rhs_bref, out_bref],
            scratches=[acc_scratch_ref],
            first_cycle=is_start,
            last_cycle=is_end,
            init_accumulators=is_start,
            prefetch=prefetch,
            schedule=[None, None, accum_schedule]
      )

    kernel = pl.pallas_call(
        all_gather_lhs_matmul_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((m, n), out_dtype),
            jax.ShapeDtypeStruct((2, 2, m, sharded_k), input_dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=memory_space),
                pl.BlockSpec(memory_space=memory_space),
            ],
            out_specs=[pl.BlockSpec(memory_space=memory_space),
                       pl.BlockSpec(memory_space=memory_space)],
            grid=(outer_steps, 2),
            scratch_shapes=[
                pltpu.VMEM((tm, tn), jnp.float32)]
            + [pltpu.SemaphoreType.DMA] * 4
            + inner_allocs
        ),
        compiler_params=pltpu.CompilerParams(
                    collective_id=0,
                    # must set scoped vmem flag *larger* than below! e.g.:
                    # flags.FLAGS.xla_tpu_scoped_vmem_limit_kib = 131072
                    vmem_limit_bytes=int(134217728 * 0.9)  # 0.9 * 128MiB
        ),
    )

    shard = partial(
        shard_map.shard_map,
        mesh=jax.sharding.Mesh(
            mesh_utils.create_device_mesh((num_devices,), jax.devices()),
            ['x'],
        ),
        in_specs=(P(None, 'x'), P(None, None)),
        out_specs=P(None, None),
        check_vma=False,
    )

    test = jax.jit(shard(kernel))

    @jax.jit
    @shard
    def reference(x, y):
      x = jax.lax.all_gather(x, 'x', axis=1, tiled=True)
      return jnp.dot(x, y, preferred_element_type=out_dtype)

    k1, k2 = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(
        k1, (m, k), dtype=input_dtype, minval=-1, maxval=1)
    y = jax.random.uniform(
        k2, (k, n), dtype=input_dtype, minval=-1, maxval=1
    )

    out = jax.block_until_ready(test(x, y)[0])
    expected_out = jax.block_until_ready(reference(x, y))

    np.testing.assert_allclose(
        out.astype(jnp.float32),
        expected_out.astype(jnp.float32),
        atol=1 if out_dtype == jnp.float32 else 5,
    )

  @parameterized.named_parameters(
      ('vmem', pltpu.VMEM, jnp.bfloat16, 2, 2, 2),
      ('hbm', pltpu.ANY, jnp.bfloat16, 2, 2, 2),
      ('hbm_float32', pltpu.ANY, jnp.float32, 2, 2, 2),
      ('hbm_float32_122', pltpu.ANY, jnp.float32, 1, 2, 2),
      ('hbm_float32_121', pltpu.ANY, jnp.float32, 1, 2, 1),
  )
  def test_pipeline_throughput_optimized_allgather_matmul(
      self, memory_space, out_dtype, n_tiles, m_tiles, k_tiles):
    # TODO(b/358121809): Re-enable this test once the bug is fixed.
    self.skipTest('Broken test.')
    input_dtype = out_dtype
    num_devices = jax.local_device_count()

    tn = 128
    tm = 128
    tk = 128
    n = tn * n_tiles
    m = tm * m_tiles  # subsplit on this dim!
    k = tk * k_tiles * num_devices

    outer_steps = num_devices

    sharded_k = k // num_devices
    half_m = m // 2
    inner_grid = (n // tn, half_m // tm, sharded_k // tk)

    inner_kernel = partial(basic_matmul_kernel, k=sharded_k)

    inner_allocs = [
        pltpu.BufferedRef.input(
            pl.BlockSpec((tm, tk), lambda n, m, k: (m, k)), input_dtype),
        pltpu.BufferedRef.input(
            pl.BlockSpec((tk, tn), lambda n, m, k: (k, n)), input_dtype),
        pltpu.BufferedRef.accumulator(
            pl.BlockSpec((tm, tn), lambda n, m, k: (m, n)), out_dtype),
        ]

    def all_gather_lhs_matmul_kernel(
        # in refs:
        lhs_ref,  # [m, sharded_k]
        rhs_ref,  # [k, n]
        # out refs:
        out_ref,  # [m, n]
        # used as scratch but defined as an output so that it's resident in HBM:
        lhs_scratch_ref,  # [2 (bwd/fwd), 2 (work/buffer), m//2, sharded_k]
        # scratch refs:
        acc_scratch_ref,  # [tm, tn]
        bwd_recv_sem,
        bwd_send_sem,
        fwd_recv_sem,
        fwd_send_sem,
        lhs_bref,
        rhs_bref,
        out_bref
    ):
      outer_step = pl.program_id(0)  # range [0, steps-1]
      phase = pl.program_id(1)  # range [0, 1]  0 == BWD Matmul, 1 == FWD Matmul

      # kernel start / end booleans
      is_start = jnp.logical_and(outer_step == 0, phase == 0)
      is_end = jnp.logical_and(outer_step == outer_steps - 1, phase == 1)

      # slots for double-buffered LHS scratch accumulator
      # at each sub-step, working slot --> buffering slot
      working_slot = lax.rem(outer_step, 2)
      buffering_slot = 1 - working_slot

      # IDs of self and neighbors
      my_id = lax.axis_index('x')
      right_neighbor = mod(my_id + 1, num_devices)
      left_neighbor = mod(my_id - 1, num_devices)

      # Initialize backwards stream by transfer of LHS chunks into local
      # working copies.
      initial_bwd_copy = pltpu.make_async_copy(
          lhs_ref.at[make_ds(1, m//2)],
          lhs_scratch_ref.at[0, working_slot],
          bwd_send_sem,
      )

      # Initialize forwards stream by transfer of LHS chunks into local
      # working copies.
      initial_fwd_copy = pltpu.make_async_copy(
          lhs_ref.at[make_ds(0, m//2)],
          lhs_scratch_ref.at[1, working_slot],
          bwd_send_sem,
      )

      # Transfer working copies of LHS chunks backwards to left neighbors'
      # buffering copies.
      bwd_copy = pltpu.make_async_remote_copy(
          src_ref=lhs_scratch_ref.at[0, working_slot],
          dst_ref=lhs_scratch_ref.at[0, buffering_slot],
          send_sem=bwd_send_sem,
          recv_sem=bwd_recv_sem,
          device_id=left_neighbor,
      )

      # Transfer working copies of LHS chunks forwards to right neighbors'
      # buffering copies.
      fwd_copy = pltpu.make_async_remote_copy(
          src_ref=lhs_scratch_ref.at[1, working_slot],
          dst_ref=lhs_scratch_ref.at[1, buffering_slot],
          send_sem=fwd_send_sem,
          recv_sem=fwd_recv_sem,
          device_id=right_neighbor,
      )

      # Slice RHS to match LHS slices in bwd/fwd phases for contractions.
      def get_rhs_slice(outer_step, phase):
        bwd_rhs_offset = mod(my_id + outer_step, num_devices)
        fwd_rhs_offset = mod(my_id - outer_step, num_devices)
        offset = jnp.where(phase, fwd_rhs_offset, bwd_rhs_offset)
        return make_ds(offset, sharded_k)

      def get_half(phase):
        return make_ds(jnp.where(phase, 0, 1), m//2)

      # Loop Prologue
      @pl.when(is_start)
      @jax.named_scope('sync_and_bwd_init')
      def _sync_and_bwd_init():
        # barrier at start
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
        pltpu.semaphore_signal(barrier_sem, device_id=right_neighbor)
        pltpu.semaphore_wait(barrier_sem, 2)
        # initializing copies
        initial_bwd_copy.start()
        initial_fwd_copy.start()
        initial_bwd_copy.wait()

      @pl.when(jnp.logical_and(outer_step != outer_steps - 1, phase == 0))
      @jax.named_scope('send_next_dma')
      def _send_next_dma():
        bwd_copy.start()
        @pl.when(jnp.logical_not(is_start))
        def _send_next_fwd_dma():
          fwd_copy.start()

      # Loop Prefetch
      def prefetch(lhs_bref, rhs_bref, out_bref, scheduler):
        @pl.when(is_start)
        @jax.named_scope('fwd_init')
        def _fwd_init():
          initial_fwd_copy.wait()
          fwd_copy.start()

        @pl.when(jnp.logical_and(outer_step != outer_steps - 1, phase == 1))
        @jax.named_scope('wait_on_prev_dma')
        def _wait_on_prev_dma():
          bwd_copy.wait()
          fwd_copy.wait()

        # prefetch next inputs
        next_working_slot = jnp.where(
            phase == 0, working_slot, buffering_slot)
        next_step = jnp.where(phase == 0, outer_step, outer_step + 1)
        next_phase = lax.rem(phase + 1, 2)
        scheduler.prefetch(
            lhs_bref, lhs_scratch_ref.at[next_phase, next_working_slot])
        scheduler.prefetch(
            rhs_bref, rhs_ref.at[get_rhs_slice(next_step, next_phase)])
        scheduler.prefetch(
            out_bref, out_ref.at[get_half(next_phase)])

      pltpu.emit_pipeline(inner_kernel, grid=inner_grid)(
          lhs_scratch_ref.at[phase, working_slot],
          rhs_ref.at[get_rhs_slice(outer_step, phase)],
          out_ref.at[get_half(phase)],
          allocations=[lhs_bref, rhs_bref, out_bref],
          scratches=[acc_scratch_ref],
          first_cycle=is_start,
          last_cycle=is_end,
          init_accumulators=outer_step == 0,
          prefetch=prefetch,
      )

    kernel = pl.pallas_call(
        all_gather_lhs_matmul_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((m, n), out_dtype),
            jax.ShapeDtypeStruct((2, 2, half_m, sharded_k), input_dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=memory_space),
                pl.BlockSpec(memory_space=memory_space),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=memory_space),
                pl.BlockSpec(memory_space=memory_space),
            ],
            grid=(outer_steps, 2),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)]
            + [pltpu.SemaphoreType.DMA] * 4
            + inner_allocs,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            # must set scoped vmem flag *larger* than below! e.g.:
            # flags.FLAGS.xla_tpu_scoped_vmem_limit_kib = 131072
            vmem_limit_bytes=int(134217728 * 0.9),  # 0.9 * 128MiB
        ),
    )

    shard = partial(
        shard_map.shard_map,
        mesh=jax.sharding.Mesh(
            mesh_utils.create_device_mesh((num_devices,), jax.devices()),
            ['x'],
        ),
        in_specs=(P(None, 'x'), P(None, None)),
        out_specs=P(None, None),
        check_vma=False,
    )

    test = jax.jit(shard(kernel))

    @jax.jit
    @shard
    def reference(x, y):
      x = jax.lax.all_gather(x, 'x', axis=1, tiled=True)
      return jnp.dot(x, y, preferred_element_type=out_dtype)

    k1, k2 = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(
        k1, (m, k), dtype=input_dtype, minval=-1, maxval=1)
    y = jax.random.uniform(
        k2, (k, n), dtype=input_dtype, minval=-1, maxval=1
    )

    out = jax.block_until_ready(test(x, y)[0])
    expected_out = jax.block_until_ready(reference(x, y))

    np.testing.assert_allclose(
        out.astype(jnp.float32),
        expected_out.astype(jnp.float32),
        atol=1 if out_dtype == jnp.float32 else 5,
    )

  @parameterized.named_parameters(
      ('vmem', pltpu.VMEM, jnp.bfloat16, 2, 2, 2),
      ('hbm', pltpu.ANY, jnp.bfloat16, 2, 2, 2),
      ('hbm_float32', pltpu.ANY, jnp.float32, 2, 4, 2),
      ('hbm_float32_112', pltpu.ANY, jnp.float32, 1, 1, 2),
      ('hbm_float32_111', pltpu.ANY, jnp.float32, 1, 1, 1),
  )
  def test_pipeline_latency_optimized_matmul_reducescatter(
      self, memory_space, out_dtype, n_tiles, m_tiles, k_tiles):
    input_dtype = jnp.float32
    num_devices = jax.device_count()

    tn = 128 * 1
    tm = 128 * 1
    tk = 128 * 1
    n = tn * n_tiles
    m = tm * m_tiles * num_devices
    k = tk * k_tiles * num_devices

    sharded_m = m // num_devices
    sharded_k = k // num_devices
    inner_grid = (n // tn, sharded_m // tm, sharded_k // tk)
    outer_steps = num_devices // 2
    reduce_grid = (sharded_m // tm,)

    inner_kernel = partial(basic_matmul_kernel, k=sharded_k)

    def reduce_kernel(
        out_ref,  # [tm, tn]
        rs_accum_scratch_ref,  # [tm, tn]
    ):
      rs_accum_scratch_ref[...] = out_ref[...]

    inner_allocs = [
        pltpu.BufferedRef.input(
            pl.BlockSpec((tm, tk), lambda n, m, k: (m, k)), input_dtype),
        pltpu.BufferedRef.input(
            pl.BlockSpec((tk, tn), lambda n, m, k: (k, n)), input_dtype),
        pltpu.BufferedRef.accumulator(
            pl.BlockSpec((tm, tn), lambda n, m, k: (m, n)), out_dtype),
        # only used for final addition of fwd + bwd streams.
        pltpu.BufferedRef.input(
            pl.BlockSpec((tm, n), lambda m: (m, 0)), out_dtype),
        pltpu.BufferedRef.accumulator(
            pl.BlockSpec((tm, n), lambda m: (m, 0)), out_dtype),
        ]

    def reduce_scatter_lhs_matmul_kernel(
        # in refs:
        lhs_ref,  # [sharded_m, sharded_k]
        rhs_ref,  # [sharded_k, n]
        # out refs:
        accumulator_ref,  # [2 (bwd/fwd), 2 (work/buffer), sharded_m, n]
        # scratch refs:
        acc_scratch_ref,  # [tm, tn]
        bwd_recv_sem,
        bwd_send_sem,
        fwd_recv_sem,
        fwd_send_sem,
        lhs_bref,
        rhs_bref,
        out_bref,
        reduce_in_bref,
        reduce_out_bref,
    ):
      outer_step = pl.program_id(0)  # range [0, outer_steps-1]
      phase = pl.program_id(1)  # range [0, 1]  0 == BWD Matmul, 1 == FWD Matmul

      num_inner_steps = _grid_size(inner_grid)
      trivial_loop = num_inner_steps == 1

      # kernel start / end booleans
      is_start = jnp.logical_and(outer_step == 0, phase == 0)
      is_end = jnp.logical_and(outer_step == outer_steps - 1, phase == 1)

      # slots for double-buffered accumulator
      # at each sub-step, working slot --> buffering slot
      working_slot = lax.rem(outer_step, 2)
      buffering_slot = 1 - working_slot

      # IDs of self and neighbors
      my_id = lax.axis_index('x')
      right_neighbor = mod(my_id + 1, num_devices)
      left_neighbor = mod(my_id - 1, num_devices)

      # Async copy definitions:

      # Transfer accumulator chunks backwards to left neighbors
      bwd_copy = pltpu.make_async_remote_copy(
          # buffering <--> working swapped as this is run in a subsequent step.
          src_ref=accumulator_ref.at[1, buffering_slot],
          dst_ref=accumulator_ref.at[1, working_slot],
          send_sem=bwd_send_sem,
          recv_sem=bwd_recv_sem,
          device_id=left_neighbor,
      )

      # Transfer accumulator chunks forwards to right neighbors
      fwd_copy = pltpu.make_async_remote_copy(
          src_ref=accumulator_ref.at[0, working_slot],
          dst_ref=accumulator_ref.at[0, buffering_slot],
          send_sem=fwd_send_sem,
          recv_sem=fwd_recv_sem,
          device_id=right_neighbor,
      )

      # Slice RHS slices in bwd/fwd phases for contractions.
      def get_lhs_slice(step, phase):
        bwd_lhs_offset = mod(my_id + step + num_devices//2 + 1, num_devices)
        fwd_lhs_offset = mod(my_id - step - num_devices//2, num_devices)
        offset = jnp.where(phase, bwd_lhs_offset, fwd_lhs_offset)
        return (
            pl.ds(pl.multiple_of(offset * sharded_m, sharded_m), sharded_m),
            pl.ds(pl.multiple_of(0, sharded_k), sharded_k),
        )

      rhs_schedule = pltpu.get_pipeline_schedule('fixed')

      # Outer Loop Prologue
      @pl.when(is_start)
      @jax.named_scope('sync')
      def _sync_barrier():
        # barrier at start
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
        pltpu.semaphore_signal(barrier_sem, device_id=right_neighbor)
        pltpu.semaphore_wait(barrier_sem, 2)

      # Writeback previous outputs on first step of our present cycle
      def postyeet(lhs_bref, rhs_bref, out_bref, scheduler):
        del lhs_bref, rhs_bref

        @pl.when(~is_start)
        def _rdmas():
          @pl.when(phase == 1)
          @jax.named_scope('send_prev_fwd_dma')
          def _send_prev_fwd_dma():
            fwd_copy.start()
          @pl.when(phase == 0)
          @jax.named_scope('send_prev_bwd_dma')
          def _send_prev_bwd_dma():
            bwd_copy.start()

        # When the inner matmul loop consists of a single iteration, we have
        # no opportunity to overlap in this loop and must block immediately.
        @pl.when(trivial_loop)
        def _prefetch_accumulator_late():
          @pl.when(~is_start)
          def _rdmas():
            @pl.when(phase == 1)
            @jax.named_scope('send_prev_fwd_dma')
            def _send_prev_fwd_dma():
              fwd_copy.wait()
            @pl.when(phase == 0)
            @jax.named_scope('send_prev_bwd_dma')
            def _send_prev_bwd_dma():
              bwd_copy.wait()

          # deferred "prefetch"
          # Don't prefetch when accums are being zeroed.
          @pl.when(~is_start)
          def _prefetch():
            next_slot = jnp.where(phase == 0, working_slot, buffering_slot)
            next_phase = 1 - phase
            scheduler.prefetch(out_bref,
                                accumulator_ref.at[next_phase, next_slot])

      # Prefetch next inputs on last step of our present cycle
      def prefetch(lhs_bref, rhs_bref, out_bref, scheduler):
        @pl.when(~is_start & ~trivial_loop)
        def _wait_dmas():
          @pl.when(phase == 1)
          @jax.named_scope('wait_prev_fwd_dma')
          def _wait_prev_fwd_dma():
            fwd_copy.wait()
          @pl.when(phase == 0)
          @jax.named_scope('wait_prev_bwd_dma')
          def _wait_prev_bwd_dma():
            bwd_copy.wait()

        # prefetch next inputs
        next_working_slot = jnp.where(phase == 0, working_slot, buffering_slot)
        next_step = jnp.where(phase == 0, outer_step, outer_step + 1)
        next_phase = lax.rem(phase + 1, 2)
        scheduler.prefetch(
            lhs_bref, lhs_ref.at[get_lhs_slice(next_step, next_phase)])
        scheduler.prefetch(rhs_bref, rhs_ref, rhs_schedule)
        # When the inner matmul loop consists of a single iteration, we need
        # to avoid optimistic prefetch to avoid a data race.
        # Don't prefetch when accums are being zeroed.
        @pl.when(~trivial_loop & ~is_start)
        def _prefetch_accum():
          scheduler.prefetch(
              out_bref, accumulator_ref.at[next_phase, next_working_slot])

      # Run matmul pipeline
      pltpu.emit_pipeline(inner_kernel, grid=inner_grid)(
          lhs_ref.at[get_lhs_slice(outer_step, phase)],
          rhs_ref,
          accumulator_ref.at[phase, working_slot],
          allocations=[lhs_bref, rhs_bref, out_bref],
          scratches=[acc_scratch_ref],
          first_cycle=is_start,
          last_cycle=is_end,
          init_accumulators=outer_step == 0,
          prefetch=prefetch,
          postyeet=postyeet,
          schedule=[None, rhs_schedule, None],
      )

      # Add forwards and backwards stream results together
      # Is it really advantageous to do this here rather than doing a simple
      # addition outside?
      @pl.when(is_end)
      def _loop_epilogue():
        pltpu.emit_pipeline(reduce_kernel, grid=reduce_grid)(
            accumulator_ref.at[1, 1],  # <-- 1,1/0,0 always correct?
            accumulator_ref.at[0, 0],
            allocations=[reduce_in_bref, reduce_out_bref],
            scratches=[],
            first_cycle=True,
            last_cycle=True,
            init_accumulators=False,
        )

    kernel = pl.pallas_call(
        reduce_scatter_lhs_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((2, 2, sharded_m, n), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=memory_space),
                pl.BlockSpec(memory_space=memory_space),
            ],
            out_specs=pl.BlockSpec(memory_space=memory_space),
            grid=(outer_steps, 2),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)]
            + [pltpu.SemaphoreType.DMA] * 4
            + inner_allocs,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            # must set scoped vmem flag *larger* than below!
            # e.g. flags.FLAGS.xla_tpu_scoped_vmem_limit_kib = 131072
            vmem_limit_bytes=int(134217728 * 0.9),  # 0.9 * 128MiB
        ),
    )

    shard = partial(
        shard_map.shard_map,
        mesh=jax.sharding.Mesh(
            mesh_utils.create_device_mesh(
                (num_devices,), jax.devices()[:num_devices]),
            ['x'],
        ),
        in_specs=(P(None, 'x'), P('x', None)),
        out_specs=P('x', None),
        check_vma=False,
    )

    test = jax.jit(shard(lambda x, y: kernel(x, y)[0, 0]))

    @jax.jit
    @shard
    def reference(x, y):
      unreduced = jnp.dot(x, y, preferred_element_type=out_dtype)
      return lax.psum_scatter(
          unreduced, 'x', scatter_dimension=0, tiled=True)

    k1, k2 = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(
        k1, (m, k), dtype=input_dtype, minval=-1, maxval=1)
    y = jax.random.uniform(
        k2, (k, n), dtype=input_dtype, minval=-1, maxval=1
    )

    out = jax.block_until_ready(test(x, y))
    expected_out = jax.block_until_ready(reference(x, y))

    np.testing.assert_allclose(
        out.astype(jnp.float32),
        expected_out.astype(jnp.float32),
        atol=1 if out_dtype == jnp.float32 else 5,
    )

    np.mean(np.abs(out - expected_out))

  @parameterized.named_parameters(
      ('vmem', pltpu.VMEM, jnp.bfloat16, 2, 2, 2),
      ('hbm', pltpu.ANY, jnp.bfloat16, 2, 2, 2),
      ('hbm_float32', pltpu.ANY, jnp.float32, 2, 4, 2),
      ('hbm_float32_112', pltpu.ANY, jnp.float32, 1, 2, 2),
      ('hbm_float32_111', pltpu.ANY, jnp.float32, 1, 2, 1),
  )
  def test_pipeline_throughput_optimized_matmul_reducescatter(
      self, memory_space, out_dtype, n_tiles, m_tiles, k_tiles):
    # TODO(b/358121809): Re-enable this test once the bug is fixed.
    self.skipTest('Broken test.')
    input_dtype = jnp.float32
    num_devices = jax.device_count()

    tn = 128 * 1
    tm = 128 * 1
    tk = 128 * 1
    n = tn * n_tiles
    m = tm * m_tiles * num_devices  # subsplit dim
    k = tk * k_tiles * num_devices

    sharded_m = m // num_devices
    half_m = sharded_m // 2
    sharded_k = k // num_devices
    inner_grid = (n // tn, half_m // tm, sharded_k // tk)
    outer_steps = num_devices
    inner_kernel = partial(basic_matmul_kernel, k=sharded_k)

    inner_allocs = [
        pltpu.BufferedRef.input(
            pl.BlockSpec((tm, tk), lambda n, m, k: (m, k)), input_dtype),
        pltpu.BufferedRef.input(
            pl.BlockSpec((tk, tn), lambda n, m, k: (k, n)), input_dtype),
        pltpu.BufferedRef.accumulator(
            pl.BlockSpec((tm, tn), lambda n, m, k: (m, n)), out_dtype),
        ]

    def reduce_scatter_lhs_matmul_kernel(
        # in refs:
        lhs_ref,  # [sharded_m, sharded_k]
        rhs_ref,  # [sharded_k, n]
        # out refs:
        rs_accum_scratch_ref,  # [2 (work/buffer), 2*sharded_m//2 (fwd/bwd), n]
        # scratch refs:
        acc_scratch_ref,  # [tm, tn]
        bwd_recv_sem,
        bwd_send_sem,
        fwd_recv_sem,
        fwd_send_sem,
        lhs_bref,
        rhs_bref,
        out_bref,
    ):
      outer_step = pl.program_id(0)  # range [0, outer_steps-1]
      phase = pl.program_id(1)  # range [0, 1]  0 == BWD Matmul, 1 == FWD Matmul

      num_inner_steps = _grid_size(inner_grid)
      trivial_loop = num_inner_steps == 1

      # kernel start / end booleans
      is_start = jnp.logical_and(outer_step == 0, phase == 0)
      is_end = jnp.logical_and(outer_step == outer_steps - 1, phase == 1)

      # slots for double-buffered accumulator
      # at each sub-step, working slot --> buffering slot
      working_slot = lax.rem(outer_step, 2)
      buffering_slot = 1 - working_slot

      # IDs of self and neighbors
      my_id = lax.axis_index('x')
      right_neighbor = mod(my_id + 1, num_devices)
      left_neighbor = mod(my_id - 1, num_devices)

      # Async copy definitions:

      # Transfer accumulator chunks backwards to left neighbors
      bwd_copy = pltpu.make_async_remote_copy(
          # buffering <--> working swapped as this is run in a subsequent step.
          src_ref=rs_accum_scratch_ref.at[buffering_slot, make_ds(1, half_m)],
          dst_ref=rs_accum_scratch_ref.at[working_slot, make_ds(1, half_m)],
          send_sem=bwd_send_sem,
          recv_sem=bwd_recv_sem,
          device_id=left_neighbor,
      )

      # Transfer accumulator chunks forwards to right neighbors
      fwd_copy = pltpu.make_async_remote_copy(
          src_ref=rs_accum_scratch_ref.at[working_slot, make_ds(0, half_m)],
          dst_ref=rs_accum_scratch_ref.at[buffering_slot, make_ds(0, half_m)],
          send_sem=fwd_send_sem,
          recv_sem=fwd_recv_sem,
          device_id=right_neighbor,
      )

      # Slice RHS slices in bwd/fwd phases for contractions.
      def get_lhs_slice(step, phase):
        bwd_lhs_offset = 2 * mod(my_id + step + 1, num_devices) + 1
        fwd_lhs_offset = 2 * mod(my_id - step - 1, num_devices)
        offset = jnp.where(phase, bwd_lhs_offset, fwd_lhs_offset)
        return (
            pl.ds(pl.multiple_of(offset * half_m, half_m), half_m),
            pl.ds(pl.multiple_of(0, sharded_k), sharded_k),
        )
      def get_accum_slice(phase, slot):
        return (slot, make_ds(phase, half_m))

      rhs_schedule = pltpu.get_pipeline_schedule('fixed')

      # Outer Loop Prologue
      @pl.when(is_start)
      @jax.named_scope('sync')
      def _sync_and_bwd_init():
        # barrier at start
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
        pltpu.semaphore_signal(barrier_sem, device_id=right_neighbor)
        pltpu.semaphore_wait(barrier_sem, 2)

      # Writeback previous outputs on first step of our present cycle
      def postyeet(lhs_bref, rhs_bref, out_bref, scheduler):
        del lhs_bref, rhs_bref
        @pl.when(~is_start & ~is_end)
        def _rdmas():
          @pl.when(phase == 1)
          @jax.named_scope('send_prev_fwd_dma')
          def _send_prev_fwd_dma():
            fwd_copy.start()
          @pl.when(phase == 0)
          @jax.named_scope('send_prev_bwd_dma')
          def _send_prev_bwd_dma():
            bwd_copy.start()

        # When the inner matmul loop consists of a single iteration, we have
        # no opportunity to overlap in this loop and must block immediately.
        @pl.when(trivial_loop)
        def _prefetch_accumulator_late():
          @pl.when(~is_start & ~is_end)
          def _wait_dmas():
            @pl.when(phase == 1)
            @jax.named_scope('wait_prev_fwd_dma')
            def _wait_prev_fwd_dma():
              fwd_copy.wait()
            @pl.when(phase == 0)
            @jax.named_scope('wait_prev_bwd_dma')
            def _wait_prev_bwd_dma():
              bwd_copy.wait()
          # deferred "prefetch"
          # Don't prefetch when accums are being zeroed.
          @pl.when(~is_start)
          def _prefetch():
            next_working_slot = jnp.where(
                phase == 0, working_slot, buffering_slot)
            next_phase = 1 - phase
            scheduler.prefetch(
                out_bref, rs_accum_scratch_ref.at[
                    get_accum_slice(next_phase, next_working_slot)])

      # Prefetch next inputs on last step of our present cycle
      def prefetch(lhs_bref, rhs_bref, out_bref, scheduler):
        @pl.when(~is_start & ~is_end & ~trivial_loop)
        def _wait_dmas():
          @pl.when(phase == 1)
          @jax.named_scope('wait_prev_fwd_dma')
          def _wait_prev_fwd_dma():
            fwd_copy.wait()
          @pl.when(phase == 0)
          @jax.named_scope('wait_prev_bwd_dma')
          def _wait_prev_bwd_dma():
            bwd_copy.wait()

        # prefetch next inputs
        next_working_slot = jnp.where(phase == 0, working_slot, buffering_slot)
        next_step = jnp.where(phase == 0, outer_step, outer_step + 1)
        next_phase = lax.rem(phase + 1, 2)
        scheduler.prefetch(lhs_bref,
                           lhs_ref.at[get_lhs_slice(next_step, next_phase)])
        scheduler.prefetch(rhs_bref, rhs_ref, rhs_schedule)
        # Don't prefetch when accums are being zeroed.
        @pl.when(~trivial_loop & ~is_start)
        def _prefetch_accumulator():
          scheduler.prefetch(
              out_bref, rs_accum_scratch_ref.at[
                  get_accum_slice(next_phase, next_working_slot)])

      # Run matmul pipeline
      pltpu.emit_pipeline(inner_kernel, grid=inner_grid)(
          lhs_ref.at[get_lhs_slice(outer_step, phase)],
          rhs_ref,
          rs_accum_scratch_ref.at[get_accum_slice(phase, working_slot)],
          allocations=[lhs_bref, rhs_bref, out_bref],
          scratches=[acc_scratch_ref],
          first_cycle=is_start,
          last_cycle=is_end,
          init_accumulators=outer_step == 0,
          prefetch=prefetch,
          postyeet=postyeet,
          schedule=[None, rhs_schedule, None],
      )

    kernel = pl.pallas_call(
        reduce_scatter_lhs_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((2, sharded_m, n), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=memory_space),
                pl.BlockSpec(memory_space=memory_space),
            ],
            out_specs=pl.BlockSpec(memory_space=memory_space),
            grid=(outer_steps, 2),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)]
            + [pltpu.SemaphoreType.DMA] * 4
            + inner_allocs,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            # must set scoped vmem flag *larger* than below!
            # e.g. flags.FLAGS.xla_tpu_scoped_vmem_limit_kib = 131072
            vmem_limit_bytes=int(134217728 * 0.9),  # 0.9 * 128MiB
        ),
    )

    shard = partial(
        shard_map.shard_map,
        mesh=jax.sharding.Mesh(
            mesh_utils.create_device_mesh(
                (num_devices,), jax.devices()[:num_devices]),
            ['x'],
        ),
        in_specs=(P(None, 'x'), P('x', None)),
        out_specs=P('x', None),
        check_vma=False,
    )

    test = jax.jit(shard(lambda x, y: kernel(x, y)[1]))

    @jax.jit
    @shard
    def reference(x, y):
      unreduced = jnp.dot(x, y, preferred_element_type=out_dtype)
      return lax.psum_scatter(
          unreduced, 'x', scatter_dimension=0, tiled=True)

    k1, k2 = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(
        k1, (m, k), dtype=input_dtype, minval=-1, maxval=1)
    y = jax.random.uniform(
        k2, (k, n), dtype=input_dtype, minval=-1, maxval=1
    )

    out = jax.block_until_ready(test(x, y))
    expected_out = jax.block_until_ready(reference(x, y))

    np.testing.assert_allclose(
        out.astype(jnp.float32),
        expected_out.astype(jnp.float32),
        atol=1 if out_dtype == jnp.float32 else 5,
    )


class PallasCallMegacoreTest(parameterized.TestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works with TPU v4')

    super().setUp()

  def test_can_partition_nondivisible_grid_with_dynamic_dimensions(self):
    # TODO(b/358121809): Re-enable this test once the bug is fixed.
    self.skipTest('Broken test.')

    def mul_pipeline(x_ref, y_ref):
      y_ref[...] = x_ref[...] * 2

    def mul_kernel(iters_ref, x_ref, y_ref):
      pltpu.emit_pipeline(
          mul_pipeline,
          grid=(iters_ref[0], 5),
          in_specs=[
              pl.BlockSpec((128, 128), lambda i, j: (i, j)),
          ],
          out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
          core_axis=0,
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
      )(x_ref, y_ref)

    num_cores = jax.devices()[0].num_cores
    func = pl.pallas_call(
        mul_kernel,
        out_shape=jax.ShapeDtypeStruct((640, 640), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
            grid=(num_cores,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel',)
        ),
    )
    x = jax.random.uniform(jax.random.key(0), (640, 640))
    np.testing.assert_allclose(func(jnp.array([5]), x), x * 2)

  def test_megacore_mul(self):
    # TODO(b/358121809): Re-enable this test once the bug is fixed.
    self.skipTest('Broken test.')
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
          core_axis=0,
          dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL)
      )(x_ref, y_ref)

    num_cores = jax.devices()[0].num_cores
    func = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((512, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
        grid=(num_cores,),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel',)
        ),
    )
    np.testing.assert_allclose(func(x), x * 2)

  @parameterized.parameters(
      (1024, 1024, 1024, 256, 512, 256),
      (768, 1024, 1024, 256, 512, 256),
      (1024, 1024, 768, 256, 512, 256),
      (768, 1024, 768, 256, 512, 256),
  )
  def test_megacore_matmul(self, m, k, n, bm, bk, bn):
    # TODO(b/358121809): Re-enable this test once the bug is fixed.
    self.skipTest('Broken test.')
    k1, k2 = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(k1, (m, k))
    y = jax.random.uniform(k2, (k, n))

    def matmul_pipeline(x_ref, y_ref, z_ref):
      @pl.when(pl.program_id(2) == 0)
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
          core_axis=0,
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY)
      )(x_ref, y_ref, z_ref)

    num_cores = jax.devices()[0].num_cores
    func = pl.pallas_call(
        functools.partial(matmul_kernel, bm=bm, bk=bk, bn=bn),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
        grid=(num_cores,),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel',)
        ),
    )
    np.testing.assert_allclose(func(x, y), x @ y, atol=7e-5)


@partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def matmul(x: jax.Array, y: jax.Array, *, bm: int, bk: int, bn: int):

  m, k = x.shape
  _, n = y.shape

  def kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref):

    grid = (pl.cdiv(m, bm), pl.cdiv(n, bn), pl.cdiv(k, bk))

    def run(acc_scratch_ref):
      pltpu.emit_pipeline(
          partial(basic_matmul_kernel, acc_scratch_ref=acc_scratch_ref, k=k),
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
          pl.BlockSpec(memory_space=pltpu.ANY),
          pl.BlockSpec(memory_space=pltpu.ANY),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
      grid=(num_cores,),
  )(x, y)

@jtu.thread_unsafe_test_class(condition=not jtu.hypothesis_is_thread_safe())
class PaddedPipelineEmitterTest(parameterized.TestCase):

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


class PallasCallBoundedSliceIndexingTest(parameterized.TestCase):

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
        'Must return a pl.ds from the index_map for a BoundedSlice dimension.'
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


class PipelineHijaxTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works on TPU v4+.')

  def test_emit_pipeline_hijax(self):
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

      update = dataclasses.replace

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
        tup_ref, transforms = ref._refs, ref.transforms  # pylint: disable=protected-access
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
        tup_ref, transforms = ref._refs, ref.transforms  # pylint: disable=protected-access
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

        src_ref, src_transforms = src_ref.ref._refs, src_ref.transforms  # pylint: disable=protected-access
        dst_ref, dst_transforms = dst_ref.ref._refs, dst_ref.transforms  # pylint: disable=protected-access

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

        src_ref, src_transforms = src_ref.ref._refs, src_ref.transforms  # pylint: disable=protected-access
        dst_ref, dst_transforms = dst_ref.ref._refs, dst_ref.transforms  # pylint: disable=protected-access

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
        in_specs=(pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),),
        out_shape=out_ty,
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
    )(inp)

    np.testing.assert_allclose(out.x0, inp.x0)
    np.testing.assert_allclose(out.x1, inp.x1)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
