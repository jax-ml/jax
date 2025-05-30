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

import contextlib
import dataclasses
import functools
import math
import operator
import os
import re
import sys
import tempfile
import traceback
from typing import ClassVar

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import export
from jax import lax
from jax._src import checkify
from jax._src import test_util as jtu
from jax._src.pallas import core as pallas_core
from jax._src.pallas import pallas_call
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering as mgpu_lowering
from jax._src.pallas.mosaic_gpu import pipeline as mgpu_pipeline
from jax._src.pallas.mosaic_gpu import primitives as mgpu_primitives
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

try:
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib
except ImportError:
  mosaic_gpu_lib = None


jax.config.parse_flags_with_absl()


def _fori_loop(force_while: bool, lb, ub, body, init):
  if force_while:
    # using jnp.asarray make the matcher for while or scan to think
    # that the bounds are dynamic and forces the use of the while
    # primitive.
    lb, ub = jnp.asarray(lb), jnp.asarray(ub)
  return jax.lax.fori_loop(lb, ub, body, init)


def _sum_same_dtype(x):
  # TODO(slebedev): Remove this once ``FragmentedArray`` supports
  # ``reduce_sum`` for non-32-bit types.
  return jnp.sum(x, dtype=x.dtype)


class PallasTestMetaclass(parameterized.TestGeneratorMetaclass):

  def __new__(mcs, *args, lowering_semantics=plgpu.LoweringSemantics.Lane):
    cls = super().__new__(mcs, *args)
    cls.LOWERING_SEMANTICS = lowering_semantics
    return cls


class PallasTest(jtu.JaxTestCase, metaclass=PallasTestMetaclass):
  LOWERING_SEMANTICS: ClassVar[plgpu.LoweringSemantics]

  def setUp(self):
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Only works on a GPU with capability >= sm90")
    context_stack = contextlib.ExitStack()
    context_stack.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))
    self.addCleanup(context_stack.close)

    super().setUp()

  def skip_if_wg_semantics(self):
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup:
      self.skipTest("Not supported under WG semantics")

  def kernel(self, *args, **kwargs):
    compiler_params = dataclasses.replace(
        kwargs.pop("compiler_params", plgpu.CompilerParams()),
        lowering_semantics=self.LOWERING_SEMANTICS,
    )
    return plgpu.kernel(*args, compiler_params=compiler_params, **kwargs)

  def pallas_call(self, *args, **kwargs):
    compiler_params = dataclasses.replace(
        kwargs.pop("compiler_params", plgpu.CompilerParams()),
        lowering_semantics=self.LOWERING_SEMANTICS,
    )
    return pl.pallas_call(*args, compiler_params=compiler_params, **kwargs)

  @contextlib.contextmanager
  def capture_stdout(self):
    if "pytest" in sys.modules:
      self.skipTest("pytest interacts badly with GPU stdout capture")
    if mosaic_gpu_lib is None:
      raise ValueError("Running tests but missing Mosaic GPU extension")
    with jtu.capture_stdout() as stdout:
      yield stdout
      # We need to cudaDeviceSynchronize to make sure printfs are flushed.
      mosaic_gpu_lib._mosaic_gpu_ext._sync_all_devices()


class PallasSm90ATest(PallasTest, jtu.CudaArchSpecificTest):

  def setUp(self):
    self.skip_unless_sm90a()
    super().setUp()


class PallasSm100ATest(PallasTest, jtu.CudaArchSpecificTest):

  def setUp(self):
    self.skip_unless_sm100a()
    super().setUp()


class PallasCallTest(PallasTest):

  @parameterized.product(
      op=[
          lax.neg,
          lax.bitwise_not,
          lax.logistic,
          lax.exp,
          lambda x: x**2,
          lax.rsqrt,
          lax.tanh,
          lax.log,
      ],
      approx_math=[True, False],
  )
  def test_unary_op(self, op, approx_math):
    dtype = jnp.int32 if op is lax.bitwise_not else jnp.float32

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], dtype),
        compiler_params=plgpu.CompilerParams(approx_math=approx_math),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = op(x_ref[...])

    x = jnp.arange(256).astype(dtype)
    np.testing.assert_allclose(
        kernel(x), op(x), rtol=1e-5 if approx_math else 3e-7
    )

  @parameterized.product(
      op=[
          operator.add,
          lambda x, _: x + 1,  # for int->vector conversion
          operator.sub,
          operator.mul,
          lax.div,
          jnp.minimum,
          jnp.maximum,
      ],
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
  )
  def test_binary_op(self, op, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], dtype)
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = op(x_ref[...], y_ref[...])

    key0, key1 = jax.random.split(jax.random.key(0), 2)
    x = (jax.random.uniform(key0, [256]) * 42).astype(dtype)
    y = (jax.random.uniform(key1, [256]) * 42).astype(dtype)
    np.testing.assert_array_equal(kernel(x, y), op(x, y))

  @parameterized.product(
      op=[
          lax.eq,
          operator.ne,
          operator.lt,
          operator.le,
          operator.gt,
          operator.ge,
      ],
      # TODO(slebedev): Support integral types.
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
  )
  def test_comparison_op(self, op, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], dtype)
    )
    def kernel(o_ref):
      o_ref[...] = jnp.broadcast_to(
          op(dtype(42), dtype(24)).astype(dtype), o_ref.shape
      )

    np.testing.assert_array_equal(kernel(), jnp.full([256], op(42, 24), dtype))

  def test_add_first(self):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[0]

    x = jnp.arange(256).astype(jnp.float32)
    y = jnp.flip(x).reshape(1, 256)
    np.testing.assert_array_equal(kernel(x, y), x + y[0])

  @parameterized.product(shape=[(128,), (128, 128)])
  def test_reduce_sum(self, shape):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.broadcast_to(_sum_same_dtype(x_ref[...]), o_ref.shape)

    x = jnp.arange(math.prod(shape)).reshape(shape).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), jnp.sum(x))

  def test_reshape(self):
    self.skip_if_wg_semantics()

    shape1, shape2 = (128,), (2, 16, 4)

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(shape2, jnp.float32)
    )
    def kernel(x_ref, out_ref):
      x_ref_reshaped = x_ref.reshape(shape2)
      self.assertEqual(x_ref.shape, shape1)
      self.assertEqual(x_ref_reshaped.shape, shape2)
      out_ref[...] = x_ref_reshaped[...]

    x = jnp.arange(math.prod(shape1)).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x.reshape(shape2))

  def test_add_xy_indexed(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([128], jnp.float32)
    )
    def kernel(x_ref, y_ref, o_ref):
      idx = _sum_same_dtype(y_ref[...])
      o_ref[...] = x_ref[idx]

    x = jnp.arange(4 * 128).reshape(4, 128).astype(jnp.float32)
    y = jnp.zeros(128, dtype=jnp.int32)
    np.testing.assert_array_equal(kernel(x, y), x[jnp.sum(y)])

  def test_add_one_grid(self):

    @functools.partial(
        self.pallas_call,
        in_specs=[pl.BlockSpec((128,), lambda *i: i)],
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.float32),
        grid=2,
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(128 * 2).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_grid_with_scratch(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.float32),
        in_specs=[pl.BlockSpec((128,), lambda *i: i)],
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        scratch_shapes=[plgpu.SMEM((128,), jnp.float32)],
        grid=2,
    )
    def kernel(x_ref, o_ref, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      o_ref[...] = scratch_ref[...]

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  @parameterized.product(max_concurrent_steps=[1, 2, 3, 4, 16])
  def test_add_one_grid_pipelined(self, max_concurrent_steps):

    @functools.partial(
        self.pallas_call,
        in_specs=[pl.BlockSpec((128, 16), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((128, 16), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct([128 * 2, 64], jnp.float32),
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel", "sequential"],
            max_concurrent_steps=max_concurrent_steps,
        ),
        grid=(2, 4),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(128 * 2 * 64).reshape((128 * 2, 64)).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_grid_pipelined_program_id(self):

    @functools.partial(
        self.pallas_call,
        out_specs=pl.BlockSpec((16, 16), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct([16, 64], jnp.int32),
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel", "sequential"],
            max_concurrent_steps=2,
        ),
        grid=(4, 4),
    )
    def kernel(o_ref):
      o_ref[...] = jnp.broadcast_to(pl.program_id(1), o_ref.shape)

    np.testing.assert_array_equal(
        kernel(),
        jnp.repeat(jnp.repeat(jnp.arange(4), 16)[None], 16, axis=0),
    )

  def test_add_one_grid_pipelined_sequential_invariant_output(self):

    @functools.partial(
        self.pallas_call,
        in_specs=[pl.BlockSpec((32, 16), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((32, 16), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct([32 * 2, 64], jnp.float32),
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel", "sequential"],
            max_concurrent_steps=2,
        ),
        grid=(2, 4),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(32 * 2 * 64).reshape((32 * 2, 64)).astype(jnp.float32)
    y = jnp.empty_like(x)
    for i in range(2):
      i_slice = slice(32 * i, 32 * (i + 1))
      for j in range(4):
        j_slice = slice(16 * j, 16 * (j + 1))
        y = y.at[i_slice, :16].set(x[i_slice, j_slice] + 1)

    # We only compare the elements in the first 16 columns, because the rest
    # are never written to.
    np.testing.assert_array_equal(kernel(x)[:, :16], y[:, :16])

  @parameterized.parameters(jnp.float32, jnp.int32, jnp.uint32)
  def test_iota(self, dtype):
    self.skip_if_wg_semantics()

    dimension = 1

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((128, 128), dtype)
    )
    def kernel(o_ref):
      o_ref[...] = plgpu.broadcasted_iota(
          dtype, o_ref.shape, dimension, layout=plgpu.Layout.WGMMA
      )

    np.testing.assert_array_equal(
        kernel(), jax.lax.broadcasted_iota(dtype, (128, 128), dimension)
    )

  def test_inline_mgpu(self):
    dtype = jnp.dtype(jnp.bfloat16)
    self.skip_if_wg_semantics()
    shape = (128, 128)
    tile = (64, 128 // dtype.itemsize)
    tiled_shape = mgpu.tile_shape(shape, tile)
    tiled_shape_t = list(tiled_shape)
    tiled_shape_t[0], tiled_shape_t[1] = tiled_shape_t[1], tiled_shape_t[0]

    key = jax.random.key(0)
    x = (jax.random.uniform(key, (2, *shape)) * 42).astype(dtype)

    transforms = (
        plgpu.TilingTransform(tile),
        plgpu.TransposeTransform((0, 2, 1, 3, 4)),
        plgpu.SwizzleTransform(128),
    )
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM(
                x.shape,
                dtype,
                transforms=transforms,
            ),
            plgpu.Barrier(),
        ],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
    )
    def kernel(x_ref, o_ref, smem_ref, barrier):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier)
      plgpu.barrier_wait(barrier)
      # Add an indexer at the end.
      sliced_smem_ref = smem_ref.at[0]
      @plgpu.inline_mgpu(
          arg_types=(plgpu.RefType((
              plgpu.TilingTransform(tile),
              plgpu.TransposeTransform((1, 0, 2, 3)),
              plgpu.SwizzleTransform(128),
          )),),
          return_type=plgpu.ShapeDtypeStruct(
              shape, dtype, layout=plgpu.Layout.WGMMA
          ),
      )
      def foo(ctx, smem_ref):
        del ctx
        assert smem_ref.type.shape == tiled_shape_t, (smem_ref.type, tiled_shape_t)
        x = mgpu.FragmentedArray.load_tiled(smem_ref, swizzle=128)
        y = mgpu.FragmentedArray.splat(
            mgpu.c(1, x.mlir_dtype), shape=x.shape, layout=x.layout
        )
        return (x + y)

      arr = foo(sliced_smem_ref)
      @plgpu.inline_mgpu(arg_types=(plgpu.Layout.WGMMA, plgpu.RefType(transforms), plgpu.RefType()))
      def store(ctx, arr, smem_ref, o_ref):
        sliced_smem_ref = mgpu.memref_slice(smem_ref, (0,))
        arr.store_tiled(sliced_smem_ref, swizzle=128)
        mgpu.commit_shared()
        ctx.async_copy(
            src_ref=sliced_smem_ref,
            dst_ref=o_ref,
            swizzle=128,
            gmem_transform=(
                mgpu.TileTransform(tile),
                mgpu.TransposeTransform((1, 0, 2, 3)),
            ),
        )
        ctx.await_async_copy(0)

      # This time we slice inside the inline_mgpu body.
      store(arr, smem_ref, o_ref)

    np.testing.assert_array_equal(kernel(x), x[0] + 1)

  @parameterized.product(indexer=[..., slice(128), slice(None, 128)])
  def test_copy_smem_to_gmem(self, indexer):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.SMEM((256,), jnp.float32)],
    )
    def kernel(x_ref, o_ref_gmem, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_ref.at[indexer], o_ref_gmem.at[indexer])
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x)[indexer], x[indexer] + 1.0)

  @parameterized.parameters(jnp.bfloat16, jnp.float16, jnp.float32)
  def test_copy_smem_to_gmem_reduction(self, dtype):
    @functools.partial(
        pl.pallas_call,
        grid=(200,),
        in_specs=[pl.BlockSpec((128,), lambda *i: i), pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct([128], dtype),
        scratch_shapes=[plgpu.SMEM((128,), dtype)],
        input_output_aliases={1:0}
    )
    def kernel(x_ref, o_ref_gmem, o_ref_gmem_alias, scratch_ref):
      del o_ref_gmem_alias
      scratch_ref[...] = x_ref[...]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_ref.at[...], o_ref_gmem.at[...], reduction_op="add")
      plgpu.wait_smem_to_gmem(0)
    x = jnp.ones(200 * 128).astype(dtype) # 200 blocks
    output = jnp.zeros(128).astype(dtype)
    output = kernel(x, output)
    output_val = x.reshape(-1, 128).sum(axis=0)
    np.testing.assert_array_equal(output, output_val)

  @parameterized.named_parameters(
      {"testcase_name": "1d_none",
       "shape": (256,), "indexers": (slice(0, 128), slice(None, 32))},
      {"testcase_name": "1d_offset",
       "shape": (256,), "indexers": (slice(32, 96), slice(0, 32))},
      {"testcase_name": "2d_extract",
       "shape": (64, 64), "indexers": (4, slice(0, 64))},
      )
  def test_copy_smem_to_gmem_with_multiple_gmem_indexers(self, shape, indexers):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.SMEM(shape, jnp.float32)],
    )
    def kernel(x_ref, o_ref_gmem, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      plgpu.commit_smem()
      for indexer in indexers:
        scratch_ref = scratch_ref.at[indexer]
        o_ref_gmem = o_ref_gmem.at[indexer]
      plgpu.copy_smem_to_gmem(scratch_ref, o_ref_gmem)
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(np.prod(shape)).astype(jnp.float32).reshape(*shape)
    result = kernel(x)
    ref = x + 1.0
    for indexer in indexers:
      result = result[indexer]
      ref = ref[indexer]
    np.testing.assert_array_equal(result, ref)

  @parameterized.product(indexer=[..., slice(128), slice(None, 128)])
  def test_copy_gmem_to_smem(self, indexer):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM((256,), jnp.float32),
            plgpu.Barrier(),
        ],
    )
    def kernel(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(
          x_ref_gmem.at[indexer], scratch_ref.at[indexer], barrier_ref
      )
      plgpu.barrier_wait(barrier_ref)
      o_ref[...] = scratch_ref[...] + 1

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x)[indexer], x[indexer] + 1.0)

  @parameterized.named_parameters(
      {
          "testcase_name": "1d_none",
          "shape": (256,),
          "indexers": (slice(0, 128), slice(None, 32)),
      },
      {
          "testcase_name": "1d_offset",
          "shape": (256,),
          "indexers": (slice(32, 96), slice(0, 32)),
      },
      {
          "testcase_name": "2d_extract_static",
          "shape": (64, 64),
          "indexers": (4, slice(0, 64)),
      },
      {
          "testcase_name": "2d_extract_dyn",
          "shape": (64, 64),
          "indexers": lambda in_dev: (
              pl.program_id(0) + 4 if in_dev else jnp.array(4),
              slice(0, 64),
          ),
      },
  )
  def test_copy_gmem_to_smem_with_multiple_gmem_indexers(self, shape, indexers):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM(shape, jnp.float32),
            plgpu.Barrier(),
        ],
        grid=(1,),
    )
    def kernel(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      scratch_ref_sliced = scratch_ref
      for indexer in indexers(True) if callable(indexers) else indexers:
        scratch_ref_sliced = scratch_ref_sliced.at[indexer]
        x_ref_gmem = x_ref_gmem.at[indexer]
      plgpu.copy_gmem_to_smem(
          x_ref_gmem, scratch_ref_sliced, barrier_ref
      )
      plgpu.barrier_wait(barrier_ref)
      o_ref[...] = scratch_ref[...] + 1

    x = jnp.arange(np.prod(shape)).astype(jnp.float32).reshape(*shape)
    result = kernel(x)
    ref = x + 1.0
    for indexer in indexers(False) if callable(indexers) else indexers:
      result = result[indexer]
      ref = ref[indexer]
    np.testing.assert_array_equal(result, ref)

  def test_gmem_to_smem_with_multiple_smem_indexers(self):
    x = jax.random.uniform(jax.random.key(0), (2, 64, 64), dtype=jnp.float32)
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([64, 64], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM(x.shape, jnp.float32),
            plgpu.Barrier(),
        ],
    )
    def extract_x0(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref_gmem, scratch_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      x_sliced = scratch_ref.at[0, :, :]  # shape=(64, 64)
      o_ref[pl.ds(0, 32), :] = x_sliced[pl.ds(0, 32), :]
      o_ref[pl.ds(32, 32), :] = x_sliced[pl.ds(32, 32), :]
    np.testing.assert_array_equal(extract_x0(x), x[0])

  def test_gmem_to_smem_with_multiple_smem_indexers_and_transforms(self):
    self.skip_if_wg_semantics()

    x = jnp.arange(512 * 512, dtype=jnp.int32).reshape(512, 512)
    @functools.partial(
        self.pallas_call,
        grid=(4, 4),
        out_shape=jax.ShapeDtypeStruct((256, 128), jnp.int32),
        in_specs=(
            plgpu.BlockSpec(
                block_shape=(128, 128),
                index_map=lambda i, j: (i, j),
                memory_space=plgpu.SMEM,
                transforms=(
                    plgpu.TilingTransform((8, 32)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
        ),
        out_specs=(
            plgpu.BlockSpec(
                block_shape=(64, 32),
                index_map=lambda i, j: (i, j),
                memory_space=plgpu.SMEM,
            )
        ),
    )
    def kernel(x_ref, o_ref):
      x_sliced = x_ref.at[0:64, 32:96].at[:, 0:32]  # get x_ref[0:64, 32:64]
      o_ref[...] = x_sliced[...]
    ref = jnp.concatenate([x[blk:blk+64, :] for blk in range(0, 512, 128)])
    ref = jnp.concatenate(
        [ref[:, blk+32:blk+64] for blk in range(0, 512, 128)], axis=1)
    np.testing.assert_array_equal(kernel(x), ref)

  @parameterized.product(indexer=[0, 1, 2, 3])
  def test_copy_gmem_to_smem_with_indexed_barrier(self, indexer):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM((128,), jnp.float32),
            plgpu.Barrier(num_barriers=4),
        ],
    )
    def kernel(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(
          x_ref_gmem, scratch_ref, barrier_ref.at[indexer]
      )
      plgpu.barrier_wait(barrier_ref.at[indexer])
      o_ref[...] = scratch_ref[...] + 1

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  @parameterized.named_parameters(("_g2s", False), ("_s2g", True))
  def test_copy_with_transforms(self, to_smem):
    self.skip_if_wg_semantics()

    def kernel(x_ref, o_ref, barrier_ref):
      if to_smem:
        plgpu.copy_gmem_to_smem(x_ref, o_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)
      else:
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(x_ref, o_ref)
        plgpu.wait_smem_to_gmem(0)

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.BlockSpec(
        transforms=(
            plgpu.TilingTransform((8, 32)),
            plgpu.SwizzleTransform(128),
        ),
        memory_space=plgpu.SMEM,
    )
    if not to_smem:
      in_spec, out_spec = out_spec, in_spec
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([128, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(f(x), x)

  def test_scoped_copy_with_transforms(self):
    self.skip_if_wg_semantics()

    ts = (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128))
    def kernel(x_ref, o_ref, barrier_ref):
      def body(tmp_ref):
        plgpu.copy_gmem_to_smem(x_ref, tmp_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)
        o_ref[...] = tmp_ref[...] * 2
      pl.run_scoped(body, plgpu.SMEM((128, 128), jnp.float32, transforms=ts))

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.BlockSpec(transforms=ts, memory_space=plgpu.SMEM)
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([128, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(f(x), x * 2)

  def test_scoped_copy_with_user_transforms(self):
    def kernel(x_ref, o_ref, barrier_ref):
      def body(tmp_ref):
        tmp_ref = plgpu.unswizzle_ref(tmp_ref, 128)
        tmp_ref = plgpu.untile_ref(tmp_ref, (8, 32))
        plgpu.copy_gmem_to_smem(x_ref, tmp_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)
        o_ref[...] = tmp_ref[...] * 2
      pl.run_scoped(body, plgpu.SMEM((16, 4, 8, 32), jnp.float32))

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([128, 128], jnp.float32),
        in_specs=(in_spec,),
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(f(x), x * 2)

  def test_copy_with_transforms_and_indexing(self):
    self.skip_if_wg_semantics()

    def kernel(x_ref, o_ref, barrier_ref):
      for i in range(2):
        plgpu.copy_gmem_to_smem(x_ref, o_ref.at[i], barrier_ref)
        plgpu.barrier_wait(barrier_ref)

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.BlockSpec(
        transforms=(
            plgpu.TilingTransform((8, 32)),
            plgpu.TransposeTransform((0, 2, 1, 3, 4)),
            plgpu.SwizzleTransform(128),
        ),
        memory_space=plgpu.SMEM,
    )
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([2, 128, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(f(x), np.stack([x, x], axis=0))

  @parameterized.product(
      src_memory_space=[plgpu.SMEM, plgpu.GMEM],
      layout=[plgpu.Layout.WG_STRIDED((128,), vec_size=1), None,
      ]
  )
  def test_load_to_strided_layout_with_indexing(self, src_memory_space, layout):
    self.skip_if_wg_semantics()

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([2, 128], jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=src_memory_space)],
        out_specs=plgpu.BlockSpec(memory_space=plgpu.SMEM),
    )
    def kernel(x_ref, o_ref):
      for i in range(2):
        x = plgpu.load(x_ref, (i,), layout=layout)
        o_ref[i, ...] = x

    x = jnp.arange(2 * 128, dtype=jnp.float32).reshape(2, 128)
    np.testing.assert_array_equal(kernel(x), x)

  def test_indexing_before_transpose(self):
    self.skip_if_wg_semantics()

    def kernel(x_ref, o_ref, barrier_ref):
      for i in range(2):
        plgpu.copy_gmem_to_smem(
            x_ref, plgpu.transpose_ref(o_ref.at[i], (1, 0, 2)), barrier_ref
        )
        plgpu.barrier_wait(barrier_ref)

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.BlockSpec(memory_space=plgpu.SMEM)
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([2, 64, 2, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(2 * 64 * 128, dtype=jnp.float32).reshape(2, 64, 128)
    xt = x.transpose((1, 0, 2))
    np.testing.assert_array_equal(f(x), np.stack([xt, xt], axis=0))

  def test_copy_gmem_to_smem_in_run_scoped(self):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
    )
    def kernel(x_ref_gmem, o_ref):
      def body(barrier_ref):
        def inner_body(scratch_ref):
          plgpu.copy_gmem_to_smem(x_ref_gmem, scratch_ref, barrier_ref)
          plgpu.barrier_wait(barrier_ref)
          o_ref[...] = scratch_ref[...] + 1
        pl.run_scoped(inner_body, plgpu.SMEM((256,), jnp.float32))
      pl.run_scoped(body, plgpu.Barrier())

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_doubled_sum(self):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + jnp.sum(x_ref[...]) + jnp.sum(x_ref[...])

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + x.sum()*2)

  @parameterized.product(input_factor=[0.001, 1, 10, 100, 100])
  def test_layer_norm(self, input_factor):
    eps = 1e-5
    gamma = 1.0
    beta = 1.0

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def layer_norm(x_ref, o_ref):
      x_mean = jnp.mean(x_ref[...])
      x_centered = x_ref[...] - x_mean
      o_ref[...] = (
          x_centered * jax.lax.rsqrt(jnp.mean(x_centered**2) + eps) * gamma
          + beta
      )

    def layer_norm_np(x):
      x_mean = np.mean(x)
      x_centered = x - x_mean
      return (x_centered / np.sqrt(np.mean(x_centered**2) + eps) * gamma) + beta

    # Ones are always fully precise
    x = jnp.ones((256,)).astype(jnp.float32) * input_factor
    np.testing.assert_allclose(layer_norm(x), layer_norm_np(x))

    # random (and anything else is not)
    x = (
        jax.random.uniform(jax.random.key(42), shape=(256,), dtype=jnp.float32)
        * input_factor
    )
    np.testing.assert_allclose(layer_norm(x), layer_norm_np(x), rtol=5e-5)

  def test_print(self):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      del x_ref, o_ref
      pl.debug_print("It works!")

    x = jnp.arange(256).astype(jnp.float32)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
    self.assertEqual(output(), "It works!\n")

  def test_print_wgmma_tiled_layout(self):
    self.skip_if_wg_semantics()

    shape = (128, 64)
    size = math.prod(shape)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        in_specs=[
            plgpu.BlockSpec(
                transforms=(
                    plgpu.TilingTransform((8, 32)),
                    plgpu.SwizzleTransform(128),
                )
            )
        ],
    )
    def kernel(x_ref, o_ref):
      del o_ref  # Unused.
      pl.debug_print("prefix {}", x_ref[...])

    x = jnp.arange(size, dtype=jnp.float32).reshape(shape)
    with self.capture_stdout() as get_output:
      jax.block_until_ready(kernel(x))

    output = get_output()
    results = re.findall(r"prefix \[(\d+), (\d+)\]: (\d+).?\d*", output)
    self.assertLen(results, size, output)
    for i, j, v in results:
      i, j, v = map(int, (i, j, v))
      self.assertEqual(v, i * shape[1] + j)

  def test_print_scalar(self):
    self.skip_if_wg_semantics()

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x.sum() = {}", _sum_same_dtype(x_ref[...]))

    x = jnp.arange(256, dtype=jnp.int32)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x.sum() = {x.sum()}", output())

  def test_print_scalar_array(self):
    self.skip_if_wg_semantics()

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x.sum() = {}", _sum_same_dtype(x_ref[...]) + 1)

    x = jnp.arange(256, dtype=jnp.int32)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x.sum() = {x.sum() + 1}", output())

  def test_print_array(self):
    self.skip_if_wg_semantics()

    in_shape = [2, 1, 64, 64]

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(in_shape, jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x: {}", x_ref[...])

    x = jnp.arange(math.prod(in_shape), dtype=jnp.int32).reshape(in_shape)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn("x: [1, 0, 43, 23]: 6871\n", output())

  @parameterized.parameters(
          (plgpu.TilingTransform((1, 32)), plgpu.SwizzleTransform(128)),
          (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128)),
          (),
  )
  def test_get_swap_with_transforms(self, *transforms):
    self.skip_if_wg_semantics()

    shape = (128, 128)

    @functools.partial(
        self.pallas_call,
        in_specs=[plgpu.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=plgpu.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.int32),
        scratch_shapes=[
            plgpu.SMEM(shape, jnp.int32, transforms=tuple(transforms)),
            plgpu.Barrier(),
        ]
    )
    def kernel(x_ref, o_ref, scratch_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, scratch_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      scratch_ref[...] = scratch_ref[...] * 2
      plgpu.copy_smem_to_gmem(scratch_ref, o_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(math.prod(shape), dtype=jnp.int32).reshape(shape)
    np.testing.assert_array_equal(kernel(x), x * 2)

  def test_check(self):
    self.skip_if_wg_semantics()

    self.enter_context(pl.enable_debug_checks(True))

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      pl.debug_check(_sum_same_dtype(x_ref[...]) > 0, "x.sum() is negative")
      o_ref[...] = x_ref[...]

    x = jnp.arange(256, dtype=jnp.int32)
    np.testing.assert_array_equal(kernel(x), x)

  def test_load_scalar(self):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.int32),
        in_specs=[plgpu.BlockSpec(memory_space=plgpu.GMEM)],
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.broadcast_to(x_ref[10], (128,))

    np.testing.assert_array_equal(kernel(jnp.arange(11, dtype=jnp.int32)),
                                  jnp.full((128,), 10, dtype=jnp.int32))

  def test_run_scoped(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      def body(tmp_ref):
        self.assertEqual(tmp_ref.shape, (8, 128))
        tmp_ref[...] = x_ref[...] + 1.0
        return tmp_ref[...]

      tmp = pl.run_scoped(body, plgpu.SMEM((8, 128), jnp.float32))
      self.assertEqual(tmp.shape, (8, 128))
      o_ref[...] = tmp

    x = np.ones((8, 128), jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_run_scoped_in_cond(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.SMEM),
    )
    def kernel(x_ref_gmem, o_ref):
      def scoped_kernel(barrier_ref):
        plgpu.copy_gmem_to_smem(x_ref_gmem, o_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)

      def branch():
        pl.run_scoped(scoped_kernel, plgpu.Barrier())

      jax.lax.cond(x_ref_gmem[0] % 2 == 0, branch, branch)

    x = jnp.full((256,), 1234, dtype=jnp.int32)
    np.testing.assert_array_equal(kernel(x), x)

  def test_program_id(self):
    @functools.partial(
        self.pallas_call,
        in_specs=(),
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.int32),
        grid=2,
    )
    def kernel(o_ref):
      o_ref[...] = jnp.full(o_ref.shape, pl.program_id(0))

    np.testing.assert_array_equal(
        kernel(),
        jnp.array([0] * 128 + [1] * 128, dtype=jnp.int32),
    )

  def test_program_id_in_squashed_grid(self):
    # Tests whether a grid with >3 logical dimensions is correctly squashed to
    # 3 CUDA grid dimensions.
    grid = (2, 3, 4, 5)
    @functools.partial(
        self.pallas_call,
        in_specs=(),
        out_specs=pl.BlockSpec((1,) * len(grid) + (128,), lambda *i: (*i, 0)),
        out_shape=jax.ShapeDtypeStruct([*grid, 128], jnp.int32),
        grid=grid,
    )
    def kernel(o_ref):
      mult = 1
      idx = 0
      for axis in range(len(grid)-1, -1, -1):
        idx += pl.program_id(axis) * mult
        mult *= pl.num_programs(axis)
      o_ref[...] = jnp.full(o_ref.shape, idx)

    np.testing.assert_array_equal(
        kernel()[:, :, :, :, 0],
        jnp.arange(math.prod(grid), dtype=jnp.int32).reshape(*grid)
    )

  def test_program_id_in_block_spec(self):
    @functools.partial(
        self.pallas_call,
        in_specs=(pl.BlockSpec((2, 128), lambda i: (pl.program_id(0), i)),),
        out_specs=pl.BlockSpec((2, 128), lambda i: (pl.program_id(0), i)),
        out_shape=jax.ShapeDtypeStruct([2, 128], jnp.int32),
        grid=2,
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    x = jnp.arange(2 * 128, dtype=jnp.int32).reshape([2, 128])
    np.testing.assert_array_equal(kernel(x), x)

  def test_num_programs(self):
    @functools.partial(
        self.pallas_call,
        in_specs=(),
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.int32),
        grid=2,
    )
    def kernel(o_ref):
      o_ref[...] = jnp.full(o_ref.shape, pl.num_programs(0), o_ref.dtype)

    np.testing.assert_array_equal(
        kernel(),
        jnp.full([256], 2, dtype=jnp.int32),
    )

  def test_swizzled_blockspec_shapes(self):
    self.skip_if_wg_semantics()

    spec = plgpu.BlockSpec(
        (128, 64),
        lambda *i: i,
        transforms=(
            plgpu.TilingTransform((8, 64)),
            plgpu.SwizzleTransform(128),
        ),
    )
    @functools.partial(
        self.pallas_call,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float16),
        grid=(2, 2),
    )
    def kernel(x_ref, o_ref):
      assert x_ref.shape == (128, 64), x_ref.shape
      o_ref[...] = x_ref[...]

    x = jnp.arange(128 * 128).astype(jnp.float16).reshape(128, 128)
    np.testing.assert_array_equal(kernel(x), x)

  @parameterized.product(force_while=[False, True])
  def test_fori_loop_array(self, force_while):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], jnp.int32)
    )
    def kernel(x_ref, o_ref):
      # Equivalent to x_ref[...] + 2 + 3.
      o_ref[...] = _fori_loop(
          force_while, 2, 4, lambda i, x: x + i, x_ref[...]
      )

    x = jnp.arange(256, dtype=jnp.int32)
    np.testing.assert_array_equal(kernel(x), x + 2 + 3)

  @parameterized.product(unroll=[1, 2])
  def test_fori_loop_array_unrolled(self, unroll):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], jnp.int32)
    )
    def kernel(x_ref, o_ref):
      # Equivalent to x_ref[...] + 2 + 3 + 4 + 5.
      o_ref[...] = lax.fori_loop(
          2, 6, lambda i, x: x + i, x_ref[...], unroll=unroll
      )

    x = jnp.arange(256, dtype=jnp.int32)
    np.testing.assert_array_equal(kernel(x), x + 2 + 3 + 4 + 5)

  @parameterized.product(force_while=[False, True])
  def test_fori_loop_scalar(self, force_while):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], jnp.int32)
    )
    def kernel(o_ref):
      # Equivalent to 2 + 3.
      o_ref[...] = jax.lax.broadcast(
          _fori_loop(force_while, 2, 4, lambda i, x: x + i, jnp.int32(0)),
          o_ref.shape,
      )

    np.testing.assert_array_equal(kernel(), jnp.full([256], 5, jnp.int32))

  def test_fori_loop_dynamic_bounds(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
        grid=(1,)
    )
    def kernel(o_ref):
      zero = pl.program_id(0)
      # Equivalent to 2 + 3.
      o_ref[...] = jax.lax.broadcast(
          jax.lax.fori_loop(2 + zero, 4 + zero, lambda i, x: x + i, 0), o_ref.shape
      )

    np.testing.assert_array_equal(kernel(), jnp.full([256], 5, dtype=jnp.int32))

  @parameterized.product(force_while=[False, True])
  def test_fori_loop_tuple(self, force_while):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], jnp.int32)
    )
    def kernel(o_ref):
      def body(step, xs):
        return tuple(
            jax.lax.cond(step % 2 == 0, lambda x: x + 1, lambda x: x, x)
            for x in xs
        )

      # Equivalent to 3 * (0 + 1).
      o_ref[...] = jax.lax.broadcast(
          sum(_fori_loop(force_while, 2, 4, body, (jnp.int32(0),) * 3)),
          o_ref.shape,
      )

    np.testing.assert_array_equal(
        kernel(), jnp.full([256], 3 * (0 + 1), jnp.int32)
    )

  @parameterized.product(force_while=[False, True])
  def test_fori_loop_indexed_store(self, force_while):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([4, 128], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      def body(idx, _):
        o_ref[idx] = x_ref[idx] + y_ref[idx]
        return ()

      _fori_loop(force_while, 0, 4, body, ())

    x = jnp.arange(4 * 128).reshape(4, 128).astype(jnp.float32)
    y = x + 1
    np.testing.assert_array_equal(kernel(x, y), x + y)

  def test_while_loop(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([128], jnp.int32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.zeros(o_ref.shape, dtype=jnp.int32)

      def cond(acc):
        _, last_o = acc
        return _sum_same_dtype(last_o) < 128*10

      def body(acc):
        i, _ = acc
        o_ref[...] += x_ref[i]
        return i+1, o_ref[...]

      _ = jax.lax.while_loop(cond, body, (0, o_ref[...]))

    np.testing.assert_array_equal(
        kernel(jnp.ones([128, 128], jnp.int32)), jnp.full([128], 10, jnp.int32)
    )

  def test_while_loop_layout_mismatch(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([128], jnp.int32)
    )
    def kernel(o_ref):
      def cond(acc):
        return _sum_same_dtype(acc) < 128

      def body(acc):
        del acc  # Unused.

        # We deliberately do a cast here to trigger a layout mismatch.
        return plgpu.layout_cast(
            jnp.zeros(o_ref.shape, o_ref.dtype), plgpu.Layout.WGMMA_ROW
        )

      _ = jax.lax.while_loop(cond, body, o_ref[...])

    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup:
      with self.assertRaisesRegex(
          NotImplementedError,
          "Cannot convert from WGStridedFragLayout.* to TiledLayout",
      ):
        kernel()
    else:
      with self.assertRaisesRegex(
          ValueError, "has layout .*, when it should be"
      ):
        kernel()

  def test_cond(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], jnp.int32)
    )
    def kernel(x_ref, o_ref):
      jax.lax.cond(
          x_ref[0] % 2 == 0,
          lambda: pl.debug_print("acc % 2"),
          lambda: pl.debug_print("acc"),
      )
      o_ref[...] = jnp.broadcast_to(jnp.asarray(0, dtype=o_ref.dtype), o_ref.shape)

    x = jnp.full((256,), 1234, dtype=jnp.int32)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn("acc % 2", output())

  def test_cond_returning_array(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([256], jnp.int32)
    )
    def kernel(x_ref, o_ref):
      acc_sum = _sum_same_dtype(x_ref[...])
      acc2, acc = jax.lax.cond(
          acc_sum % 2 == 0,
          lambda: (acc_sum * 2, x_ref[...]),
          lambda: (acc_sum, x_ref[...]),
      )
      o_ref[...] = jnp.broadcast_to(_sum_same_dtype(acc) + acc2, o_ref.shape)

    x = jnp.arange(256, dtype=jnp.int32)
    np.testing.assert_array_equal(kernel(x), jnp.broadcast_to(jnp.sum(x) * 3, [256]))

  def test_tile_slicing(self):
    # Not testing with warpgroup semantics, because we want to enforce a layout.
    self.skip_if_wg_semantics()

    shape = (256, 128)
    block_spec = plgpu.BlockSpec(
        transforms=(plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    )
    @functools.partial(
        self.pallas_call,
        in_specs=[block_spec],
        out_specs=block_spec,
        out_shape=jax.ShapeDtypeStruct((64, 64), jnp.uint16),
    )
    def kernel(x_ref, o_ref):
      def sum_tiles(row, acc):
        row_slice = pl.ds(row * 64, 64)
        for col in range(128 // 64):
          acc += x_ref[row_slice, pl.ds(col * 64, 64)]
        return acc
      acc = plgpu.layout_cast(jnp.zeros((64, 64), jnp.uint16), plgpu.Layout.WGMMA)
      o_ref[...] = _fori_loop(False, 0, 256 // 64, sum_tiles, acc)

    x = jnp.arange(math.prod(shape), dtype=jnp.uint16).reshape(shape)
    y = x.reshape(256 // 64, 64, 128 // 64, 64).sum(axis=(0, 2), dtype=jnp.uint16)
    np.testing.assert_array_equal(kernel(x), y)

  def test_input_output_aliases(self):
    # Note that we're writing to the input pointer, which should alias b_ptr.
    def kernel(a_ref, b_ref):
      del b_ref
      a_ref[...] = jnp.ones_like(a_ref)

    a = np.zeros((64, 64), dtype=jnp.float32)
    b = self.pallas_call(
        kernel,
        in_specs=[plgpu.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=plgpu.BlockSpec(memory_space=plgpu.GMEM),
        input_output_aliases={0: 0},
        out_shape=a,
    )(a)
    np.testing.assert_array_equal(b, np.ones_like(a))

  def test_slicing(self):
    self.skip_if_wg_semantics()

    left = upper = slice(None, 64)
    right = lower = slice(64, None)
    # We rotate the four quadrants of the input clockwise.
    def rotate(src, dst):
      dst[upper, left] = src[lower, left]
      dst[upper, right] = src[upper, left]
      dst[lower, right] = src[upper, right]
      dst[lower, left] = src[lower, right]

    x = jnp.arange(128 * 128).astype(jnp.float16).reshape(128, 128)
    spec = plgpu.BlockSpec(
        transforms=(plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    )
    f = self.pallas_call(rotate, out_shape=x, in_specs=[spec], out_specs=spec)
    expected = np.empty_like(x)
    rotate(x, expected)
    np.testing.assert_array_equal(f(x), expected)

  def test_layout_cast(self, shape=(256, 64)):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
    )
    def kernel(o_ref):
      o_ref[...] = plgpu.layout_cast(jnp.full(shape, 42.0, jnp.float32), plgpu.Layout.WGMMA)

    x = jnp.full(shape, 42.0, jnp.float32)
    np.testing.assert_array_equal(kernel(), x)

  @parameterized.parameters(False, True)
  def test_wgmma_transposed_layout(self, store_transposed):
    """Tests that the result of wgmma can be store transposed using
    the WGMMA_TRNASPOSED layout.
    """

    dtype = jnp.dtype(jnp.float16)
    swizzle_elems = 128 // dtype.itemsize
    shape = (128, 128)
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[
            plgpu.SMEM(
                shape, dtype,
                transforms=(
                    plgpu.TilingTransform((8, swizzle_elems)),
                    plgpu.SwizzleTransform(128),
                ),
            )
        ]
    )
    def kernel(o_ref, smem):
      iota = plgpu.broadcasted_iota(
          dtype, o_ref.shape, 0, layout=plgpu.Layout.WGMMA
      ) * o_ref.shape[0]
      iota += plgpu.broadcasted_iota(
          dtype, o_ref.shape, 1, layout=plgpu.Layout.WGMMA
      )

      smem_trns = plgpu.transpose_ref(smem, (1, 0))
      smem_trns[...] = plgpu.layout_cast(iota, plgpu.Layout.WGMMA_TRANSPOSED)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_trns if store_transposed else smem, o_ref)

    x = jnp.arange(128 * 128, dtype=dtype).reshape((128, 128)).T
    if store_transposed:
      with self.assertRaises(ValueError):
        kernel()
    else:
      np.testing.assert_array_equal(kernel(), x)

  def test_profiler(self):
    self.skip_if_wg_semantics()  # Transform inference fails.

    def kernel(x_ref, o_ref):
      with jax.named_scope("add"):
        with jax.named_scope("load"):
          x = x_ref[...]
        o = x + x
      with jax.named_scope("store"):
        o_ref[...] = o
    with tempfile.TemporaryDirectory() as tmpdir:
      x = jnp.arange(256).astype(jnp.float32)
      y = self.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
          compiler_params=plgpu.CompilerParams(
              profile_space=16, profile_dir=tmpdir
          ),
      )(x)
      jax.block_until_ready(y)
      jax.effects_barrier()
      [name] = os.listdir(tmpdir)
      with open(os.path.join(tmpdir, name), "r") as f:
        data = f.read()
        self.assertEqual(data.count('"name": "add"'), 2)
        self.assertEqual(data.count('"name": "load"'), 2)
        self.assertEqual(data.count('"name": "store"'), 2)
      np.testing.assert_array_equal(y, x + x)

  @parameterized.product(
      dtypes=[
          (jnp.float16, jnp.float16),  # Noop
          (jnp.int16, jnp.bfloat16),
          (jnp.int16, jnp.float16),
          (jnp.uint16, jnp.float16),
          (jnp.float32, jnp.int32),
          (jnp.float32, jnp.uint32),
          (jnp.uint32, jnp.int32),
          (jnp.int32, jnp.uint32),
      ],
  )
  def test_bitcast_convert_type(self, dtypes):
    in_dtype, out_dtype = dtypes
    m, n = 16, 8
    out_shape = jax.ShapeDtypeStruct((m, n), out_dtype)

    @functools.partial(self.pallas_call, out_shape=out_shape)
    def convert(x_ref, y_ref):
      y_ref[...] = jax.lax.bitcast_convert_type(x_ref[...], out_shape)

    x = jnp.arange(m * n, dtype=in_dtype).reshape((m, n))
    np.testing.assert_array_equal(
        convert(x), jax.lax.bitcast_convert_type(x, out_dtype)
    )

  def test_optimization_barrier(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = lax.optimization_barrier(x_ref[...])

    x = jax.lax.iota(jnp.float32, 128)
    np.testing.assert_array_equal(kernel(x), x)

  def test_optimization_barrier_multiple_inputs(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      x, y = lax.optimization_barrier([x_ref[...], y_ref[...]])
      o_ref[...] = x + y

    x = jax.lax.iota(jnp.float32, 128)
    y = jax.lax.iota(jnp.float32, 128) * 3
    np.testing.assert_array_equal(kernel(x, y), x + y)

  def test_smem_aliasing_works(self):
    self.skip_if_wg_semantics()

    in_shape = (2, 256)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        in_specs=[pl.BlockSpec(in_shape)],
        out_specs=pl.BlockSpec((128,), memory_space=plgpu.GMEM),
        scratch_shapes=[
            plgpu.RefUnion(
                # Note: this test exposes internals that we don't particularly
                # want to phold for the sake of testing the functionality of the
                # API. It's expected that this test might end up breaking in the
                # future, e.g. if we decide to change our alignment requirements
                # on SMEM refs---and that's OK. Users should explicitly NOT rely
                # on this exact behaviour.
                #
                # Use a value larger than the number of bytes used for SMEM
                # alignment (1024) in order to make sure that the second ref
                # in the second group aliases the single ref in the first group.
                plgpu.SMEM(in_shape, jnp.float32),
                [
                    plgpu.SMEM((256,), jnp.bfloat16),
                    # Add an arbitrary level of nesting to make sure that we
                    # support PyTrees.
                    [
                        plgpu.SMEM(
                            (128,),
                            jnp.float32,
                            transforms=(plgpu.TilingTransform((64,)),),
                    ),
                    ]
                ],
            )
        ],
    )
    def kernel(x_ref, o_ref128, aliased_ref):
      smem_ref256, _, smem_ref128 = aliased_ref
      # Ensure that extraction via index works the same as unfolding.
      smem_ref128_2 = aliased_ref[2]
      self.assertIsInstance(smem_ref128, state_types.TransformedRef)
      self.assertIsInstance(smem_ref128_2, state_types.TransformedRef)
      self.assertIs(smem_ref128.ref, smem_ref128_2.ref)
      self.assertEqual(smem_ref128.transforms, smem_ref128_2.transforms)
      extract_alias_transform, tile_transform = smem_ref128.transforms
      # Ensure that the transforms provided in the scratch shapes have been
      # passed correctly.
      self.assertIsInstance(extract_alias_transform, gpu_core.ExtractAliasedRef)
      self.assertIsInstance(tile_transform, gpu_core.UntileRef)
      smem_ref256[...] = x_ref[...] + 1
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref128, o_ref128)

    x = jnp.arange(512).astype(jnp.float32)
    np.testing.assert_array_equal(
        kernel(x.reshape(in_shape)).reshape((128,)), x[256 : 256 + 128] + 1
    )

  def test_smem_aliasing_works_with_subbyte_dtypes(self):
    self.skip_if_wg_semantics()

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.uint4),
        in_specs=[pl.BlockSpec((128,))],
        out_specs=pl.BlockSpec((256,), memory_space=plgpu.GMEM),
        scratch_shapes=[
            plgpu.RefUnion(
                # Note: this test exposes internals that we don't particularly
                # want to phold for the sake of testing the functionality of the
                # API. It's expected that this test might end up breaking in the
                # future, e.g. if we decide to change our alignment requirements
                # on SMEM refs---and that's OK. Users should explicitly NOT rely
                # on this exact behaviour.
                #
                # This allocation scheme is a bit complicated, but serves to
                # test that
                #   1. Refs are aligned correctly (currently to 1024 bytes);
                #   2. (u)int4 references are not allocated more than 1 byte per
                #      2 elements.
                # The first group of refs serves to create two allocations, each
                # aligned to 1024 bytes. The second group serves to create two
                # allocations where the first one is exactly 1024 bytes,
                # assuming 1 byte per 2 uint4 elements. As a result, if our
                # implementation is correct, the second allocation of the second
                # group should exactly alias the second allocation of the first
                # group.
                [
                    plgpu.SMEM((128,), jnp.int8),
                    plgpu.SMEM((128,), jnp.int8),
                ],
                [plgpu.SMEM((2048,), jnp.uint4), plgpu.SMEM((256,), jnp.uint4)],
            )
        ],
    )
    def kernel(x_ref, o_refi4, aliased_ref):
      _, smem_refi8, _, smem_refi4 = aliased_ref
      smem_refi8[...] = x_ref[...]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_refi4, o_refi4)

    def unpack_i4_as_i8(x):
      x = x.reshape((128, 1))
      x_high = x >> 4
      x_low = x & 0xF
      return jnp.concatenate([x_low, x_high], axis=-1).reshape((256,))

    x = jnp.arange(128).astype(jnp.int8)
    test_as_i8 = jax.lax.convert_element_type(kernel(x), new_dtype=jnp.int8)
    np.testing.assert_array_equal(test_as_i8[:256], unpack_i4_as_i8(x))

  def test_smem_aliasing_works_for_quantization(self):
    self.skip_if_wg_semantics()
    shape = (64, 256)
    large_ty, small_ty = jnp.bfloat16, jnp.uint4
    large_swizzle = plgpu.SwizzleTransform(64 * jnp.finfo(large_ty).bits // 8)
    small_swizzle = plgpu.SwizzleTransform(64 * jnp.iinfo(small_ty).bits // 8)
    tiling = plgpu.TilingTransform((8, 64))

    def kernel(x_gmem, o_gmem):
      return pl.run_scoped(
          functools.partial(scoped_kernel, x_gmem, o_gmem),
          plgpu.RefUnion(
              plgpu.SMEM(shape, large_ty, transforms=(tiling, large_swizzle)),
              plgpu.SMEM(shape, small_ty, transforms=(tiling, small_swizzle))
          ),
          plgpu.Barrier(num_barriers=1),
      )

    def scoped_kernel(x_gmem, o_gmem, aliased_ref, barrier):
      ref_large_ty, ref_small_ty = aliased_ref
      plgpu.copy_gmem_to_smem(x_gmem, ref_small_ty, barrier=barrier)
      plgpu.barrier_wait(barrier)
      ref_large_ty[...] = ref_small_ty[...].astype(ref_large_ty.dtype) * 3
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(ref_large_ty, o_gmem)
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel_fn = self.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(shape, large_ty),
        grid=(1, 1),
    )
    key = jax.random.key(42)
    x = jax.random.randint(key, shape, 0, 4).astype(small_ty)
    expected = x * 3
    np.testing.assert_array_equal(kernel_fn(x), expected)

  def test_assigning_to_ref_union_raises(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        in_specs=[pl.BlockSpec((128,))],
        out_specs=pl.BlockSpec((128,), memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.RefUnion(plgpu.SMEM((128,), jnp.float32))],
    )
    def kernel(x_ref, o_ref128, aliased_ref):
      aliased_ref[...] = x_ref[...] + 1
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(aliased_ref, o_ref128)

    with self.assertRaisesRegex(ValueError, "can't be assigned to"):
      kernel(jnp.arange(128).astype(jnp.float32))

  @parameterized.parameters(1, 2, 3)
  def test_nd_loop(self, sm_steps):
    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((sm_steps, 132, 128), jnp.int32),
        grid=(132,),
        grid_names=("sm",),
    )
    def kernel(o_ref):
      def body(idx, _):
        assert len(idx) == 3
        # We need to use `mode="clip"`, because the indices are not static.
        flat_idx = jnp.ravel_multi_index(idx, (sm_steps, 4, 33), mode="clip")
        sm_step = lax.div(
            flat_idx, lax.convert_element_type(lax.axis_size("sm"), jnp.int32)
        )
        o_ref[sm_step, lax.axis_index("sm")] = lax.broadcast(
            flat_idx, o_ref.shape[-1:]
        )

      plgpu.nd_loop((sm_steps, 4, 33), body, None, collective_axes="sm")

    result = kernel()
    for sm_step in range(sm_steps):
      np.testing.assert_array_equal(
          result[sm_step],
          jnp.tile((132 * sm_step + jnp.arange(132))[:, None], 128),
      )

  def test_lowering_error_context(self):
    def body(x_ref, y_ref, barrier):
      plgpu.copy_gmem_to_smem(x_ref, y_ref, barrier)
      plgpu.barrier_wait(barrier)

    x = jnp.arange(127, dtype=jnp.int4)  # Size is not a multiple of bytes
    offending_line = "plgpu.copy_gmem_to_smem(x_ref, y_ref, barrier)"
    try:
      pl.pallas_call(
          body,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.SMEM),
          out_shape=x,
          scratch_shapes=[plgpu.Barrier()],
      )(x)
    except:
      # assertRaisesRegex raises does not let us match the traceback.
      self.assertIn(offending_line, traceback.format_exc())
    else:
      self.fail("Should have raised an exception")


class PallasCallWarpPrimitiveSemanticsTest(PallasTest):
  def setUp(self):
    super().setUp()
    if self.LOWERING_SEMANTICS != plgpu.LoweringSemantics.Lane:
      self.skipTest("Test only works on Lane semantics")

  def test_axis_index(self):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(plgpu.kernel,
                       out_shape=jax.ShapeDtypeStruct((2, 128), jnp.int32))
    def kernel(y_ref):
      def scope(ones_smem_ref, threes_smem_ref):
        # Prepare data to copy.
        ones_smem_ref[:] = jnp.ones((1, 128), jnp.int32)
        threes_smem_ref[:] = jnp.ones((1, 128), jnp.int32) * 3
        plgpu.commit_smem()
        @pl.core_map(warp_mesh)
        def _():
          warp_id = lax.axis_index("warp")
          # We cannot load/store inside of core_map, so we issue async
          # copies instead to produce a testable result.
          @pl.when(warp_id == 1)
          def _():
            plgpu.copy_smem_to_gmem(ones_smem_ref, y_ref.at[0:1])
          @pl.when(warp_id == 3)
          def _():
            plgpu.copy_smem_to_gmem(threes_smem_ref, y_ref.at[1:2])
        plgpu.wait_smem_to_gmem(0)
      pl.run_scoped(scope,
                    plgpu.SMEM((1, 128), jnp.int32),
                    plgpu.SMEM((1, 128), jnp.int32)
                    )
    result = kernel()
    expected = jnp.stack((jnp.ones((128,), jnp.int32),
                          jnp.ones((128,), jnp.int32) * 3), axis=0)
    np.testing.assert_array_equal(result, expected)

  def test_errors_when_closing_over_array(self):
    # We currently do not allow closing over arrays when mapping over
    # a mesh, since we would need to present a view of the array local
    # to each warp.
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(plgpu.kernel,
                       out_shape=jax.ShapeDtypeStruct((32, 32), jnp.float32),
                       scratch_shapes=[plgpu.SMEM((32, 32), jnp.float32)])
    def kernel(out_ref, smem_ref):
      arr = jnp.ones((32, 32), dtype=jnp.float32)
      @pl.core_map(warp_mesh)
      def _():
        smem_ref[...] = arr + 1
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, out_ref)
      plgpu.wait_smem_to_gmem(0)
    with self.assertRaisesRegex(
        mgpu_lowering.LoweringError,
        "Can only close over scalars and Refs when using core_map with "
        "WarpMesh",
    ):
      kernel()

  def test_single_warp_scan(self):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(plgpu.kernel,
                       out_shape=jax.ShapeDtypeStruct((10, 128), jnp.int32))
    def kernel(y_ref):
      def scope(smem_ref):
        # Prepare data to copy.
        for i in range(10):
          smem_ref[i, :] = jnp.ones_like(smem_ref.at[i]) * i
        plgpu.commit_smem()
        @pl.core_map(warp_mesh)
        def _():
          warp_id = lax.axis_index("warp")
          @pl.when(warp_id == 0)
          def _():
            def loop_body(i, _):
              _slice = pl.ds(i, 1)
              plgpu.copy_smem_to_gmem(smem_ref.at[_slice], y_ref.at[_slice])
            lax.fori_loop(0, 10, loop_body, None)
        plgpu.wait_smem_to_gmem(0)
      pl.run_scoped(scope, plgpu.SMEM((10, 128), jnp.int32))
    result = kernel()
    expected = jnp.stack(
        [jnp.ones((128,), jnp.int32) * i for i in range(10)], axis=0)
    np.testing.assert_array_equal(result, expected)

  def test_debug_print(self):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(
        plgpu.kernel,
        out_shape=jnp.zeros(128, np.int32),
    )
    def kernel(ref):
      ref[...] = ref[...]  # Prevent kernel from being DCE'd
      @pl.core_map(warp_mesh)
      def _():
        warp_id = lax.axis_index("warp")
        pl.debug_print("warp: {}", warp_id)

    with self.capture_stdout() as output:
      jax.block_until_ready(kernel())
    self.assertEqual(
        set(output().splitlines()),
        {
            "warp: 0",
            "warp: 1",
            "warp: 2",
            "warp: 3",
        },
    )

  def test_copy_gmem_to_smem_from_different_warps(self):
    # In this test, we issue a copy from from warp 0 and await it in warp 1.
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(plgpu.kernel,
                       out_shape=jax.ShapeDtypeStruct((32, 32), jnp.float32))
    def kernel(x_ref, y_ref):
      def scope(smem_ref, tma_barrier):
        @pl.core_map(warp_mesh)
        def _():
          warp_id = lax.axis_index("warp")
          @pl.when(warp_id == 0)
          def _():
            plgpu.copy_gmem_to_smem(x_ref.at[32:64], smem_ref, tma_barrier)

          @pl.when(warp_id == 1)
          def _():
            plgpu.barrier_wait(tma_barrier)
            plgpu.copy_smem_to_gmem(smem_ref, y_ref)
        plgpu.wait_smem_to_gmem(0)
      pl.run_scoped(scope,
                    smem_ref=plgpu.SMEM((32, 32), jnp.float32),
                    tma_barrier=plgpu.Barrier())
    x = jax.random.uniform(jax.random.key(42), (64, 32), jnp.float32)
    result = kernel(x)
    np.testing.assert_array_equal(result, x[32:64])


class PallasCallWGTest(
    PallasCallTest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...

  def test_missing_primitive_lowerings_are_tracked(self):
    # This test is a way to keep track of which primitives need to be adapted
    # to using warpgroup semantics. Once the set is empty, we should be able to
    # enable warpgroup semantics by default (assuming we haven't overspecialized
    # lowerings).
    rules = mgpu_lowering.mosaic_lowering_rules
    wg_wg_lowered_primitives = set(
        rules[(plgpu.LoweringSemantics.Warpgroup,
         gpu_core.PrimitiveSemantics.Warpgroup)])
    lane_wg_lowered_primitives = set(rules[
        (plgpu.LoweringSemantics.Lane, gpu_core.PrimitiveSemantics.Warpgroup)])

    actual_missing_primitives = (lane_wg_lowered_primitives -
                                 wg_wg_lowered_primitives)
    expected_missing_primitives = {
        mgpu_primitives.inline_mgpu_p,
        mgpu_primitives.broadcasted_iota_p,
        mgpu_primitives.load_p,
        mgpu_primitives.tcgen05_mma_p,
        mgpu_primitives.commit_tmem_p,
        lax.slice_p,
        pallas_core.core_map_p,
        pallas_primitives.semaphore_signal_p,
        pallas_primitives.semaphore_wait_p,
        pallas_primitives.semaphore_read_p,
        checkify.check_p,
    }

    self.assertSetEqual(actual_missing_primitives, expected_missing_primitives)


class PallasCallSm90ATest(PallasSm90ATest):

  @parameterized.parameters(False, True)
  def test_fori_loop_accumulator(self, force_while):
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    else:
      transforms = ()

    @functools.partial(
        self.pallas_call,
        in_specs=[plgpu.BlockSpec((64, 64), transforms=transforms)],
        out_shape=jax.ShapeDtypeStruct((64, 64), jnp.float16),
        out_specs=plgpu.BlockSpec((64, 64)),
    )
    def kernel(i_ref, o_ref):
      def scope(acc_ref):
        return _fori_loop(force_while, 0, 4, lambda _, v: v + acc_ref[...], acc_ref[...])
      o_ref[...] = pl.run_state(scope)(plgpu.ACC.init(i_ref[...]))

    acc_ini = jnp.ones((64, 64), dtype=jnp.float16)
    np.testing.assert_array_equal(kernel(acc_ini), jnp.full((64, 64), 5, dtype=jnp.float16))

  @parameterized.product(lhs_transpose=[False, True], rhs_transpose=[False, True])
  def test_realistic_matmul(self, lhs_transpose, rhs_transpose):
    dtype = jnp.float16
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    grid_m, grid_k, grid_n = 132, 10, 4
    tile_m = tile_n = 128
    tile_k = elems_128b
    m, k, n = grid_m * tile_m, grid_k * tile_k, grid_n * tile_n
    def kernel(a_ref, b_ref, o_ref, acc_ref):
      # Make sure tiling does not alter the shape of references
      if lhs_transpose:
        a_ref = plgpu.transpose_ref(a_ref, (1, 0))
      assert a_ref.shape == (tile_m, tile_k)
      if rhs_transpose:
        b_ref = plgpu.transpose_ref(b_ref, (1, 0))
      assert b_ref.shape == (tile_k, tile_n)
      assert o_ref.shape == acc_ref.shape == (tile_m, tile_n)
      plgpu.wgmma(acc_ref, a_ref, b_ref)
      is_last_step = pl.program_id(2) == grid_k - 1
      @pl.when(is_last_step)
      def _epilogue():
        o_ref[...] = acc_ref[...].astype(dtype)
      plgpu.wgmma_wait(1)  # We don't await the last WGMMA, hence delay_release=1

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a_shape = (k, m) if lhs_transpose else (m, k)
    a = jax.random.uniform(key1, shape=a_shape, dtype=dtype)
    b_shape = (n, k) if rhs_transpose else (k, n)
    b = jax.random.uniform(key2, shape=b_shape, dtype=dtype)

    if lhs_transpose:
      lhs_spec = pl.BlockSpec(
          (tile_k, tile_m),
          lambda m, n, k: (k, m),
      )
    else:
      lhs_spec = pl.BlockSpec(
          (tile_m, tile_k),
          lambda m, n, k: (m, k),
      )
    if rhs_transpose:
      rhs_spec = pl.BlockSpec(
          (tile_n, tile_k),
          lambda m, n, k: (n, k),
      )
    else:
      rhs_spec = pl.BlockSpec(
          (tile_k, tile_n),
          lambda m, n, k: (k, n),
      )
    out_spec = pl.BlockSpec(
        (tile_m, tile_n),
        lambda m, n, k: (m, n),
    )

    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      lhs_spec = plgpu.BlockSpec(
          lhs_spec.block_shape,
          lhs_spec.index_map,
          transforms=(
              plgpu.TilingTransform((8, elems_128b)),
              plgpu.SwizzleTransform(128),
          ),
      )
      rhs_spec = plgpu.BlockSpec(
          rhs_spec.block_shape,
          rhs_spec.index_map,
          transforms=(
              plgpu.TilingTransform((8, elems_128b)),
              plgpu.SwizzleTransform(128),
          ),
      )
      out_spec = plgpu.BlockSpec(
          out_spec.block_shape,
          out_spec.index_map,
          transforms=(
              plgpu.TilingTransform((8, elems_128b)),
              plgpu.SwizzleTransform(128),
          ),
      )

    res = self.pallas_call(
        kernel,
        in_specs=[lhs_spec, rhs_spec],
        out_specs=out_spec,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float16),
        scratch_shapes=[plgpu.ACC((tile_m, tile_n), jnp.float32)],
        grid=(grid_m, grid_n, grid_k),
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["parallel", "parallel", "sequential"],
            max_concurrent_steps=2,
            delay_release=1,
        ),
    )(a, b)
    np.testing.assert_allclose(
        res,
        (a.T if lhs_transpose else a) @ (b.T if rhs_transpose else b),
        rtol=1e-3,
    )

  @parameterized.parameters(jnp.float16, jnp.float32)
  def test_wgmma(self, dtype):
    self.skip_if_wg_semantics()

    # TensorCores can only fuse transposes of 16-bit values, and RHS
    # is expected to be column major by default.
    rhs_transpose = jnp.dtype(dtype).itemsize != 2
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    def kernel(a_ref, b_ref, o_ref):
      if rhs_transpose:
        b_ref = plgpu.transpose_ref(b_ref, (1, 0))
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref, b_ref)
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=dtype)
    b_shape = (128, 192)
    if rhs_transpose:
      b_shape = b_shape[::-1]
    b = jax.random.uniform(key2, shape=b_shape, dtype=dtype)

    rhs_transforms = (plgpu.TilingTransform((8, elems_128b)),)
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(
                (64, 128),
                lambda i, j: (i, j),
                transforms=(
                    plgpu.TilingTransform((8, elems_128b)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
            plgpu.BlockSpec(
                b_shape,
                lambda *i: i,
                transforms=(*rhs_transforms, plgpu.SwizzleTransform(128)),
            ),
        ],
        out_specs=plgpu.BlockSpec((64, 192), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_allclose(
        res, a @ (b.T if rhs_transpose else b), rtol=1e-3
    )

  def test_wgmma_registers(self):
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref[...], b_ref)
        return acc_ref[...]
      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 192), dtype=jnp.float16)

    transforms = ()
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
        ],
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
    )(a, b)
    np.testing.assert_allclose(res, a @ b, rtol=1e-3)

  def test_wgmma_registers_init(self):
    def kernel(a_ref, b_ref, i_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref[...], b_ref)
      o_ref[...] = pl.run_state(scope)(plgpu.ACC.init(i_ref[...]))

    key1, key2, key3 = jax.random.split(jax.random.key(42), 3)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 192), dtype=jnp.float16)
    i = jax.random.uniform(key3, shape=(64, 192), dtype=jnp.float16) * 10

    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
    else:
      transforms = ()
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
        ],
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float16),
    )(a, b, i)
    np.testing.assert_allclose(res, i + a @ b, rtol=2e-3)

  def test_wgmma_sliced_ref(self):
    self.skip_if_wg_semantics()  # Needs WGMMA to support slices.

    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref.at[0], b_ref.at[0])
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(2, 64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(2, 128, 192), dtype=jnp.float16)

    transforms = ()
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))

    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
        ],
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
    )(a, b)
    np.testing.assert_allclose(res, a[0] @ b[0], rtol=1e-3)

  def test_wgmma_sliced_acc(self):
    self.skip_if_wg_semantics()  # Needs WGMMA to support slices.

    swizzle = 128
    elems_128b = swizzle // jnp.dtype(jnp.float16).itemsize
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref, b_ref)
        return acc_ref[:, :64], acc_ref[:, 64:]

      o_ref[:, :64], o_ref[:, 64:] = pl.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 128), dtype=jnp.float16)
    transforms = ()
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      transforms = (
          plgpu.TilingTransform((8, elems_128b)),
          plgpu.SwizzleTransform(128),
      )
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec((64, 128), lambda *ij: ij, transforms=transforms),
            plgpu.BlockSpec((128, 128), lambda *ij: ij, transforms=transforms),
        ],
        out_specs=plgpu.BlockSpec((64, 128), lambda *ij: ij),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_allclose(res, a @ b, rtol=1e-3)

  @parameterized.product(
      src_memory_space=[plgpu.SMEM, plgpu.GMEM],
      layout=[plgpu.Layout.WGMMA_ROW, plgpu.Layout.WGMMA_COL],
      m=[64, 128, 192],
  )
  def test_load_to_wgmma_row_col_layout_with_indexing(self, src_memory_space, layout, m):
    self.skip_if_wg_semantics()

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([2, m], jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=src_memory_space)],
        out_specs=plgpu.BlockSpec(memory_space=plgpu.SMEM),
    )
    def kernel(x_ref, o_ref):
      for i in range(2):
        x = plgpu.load(
            x_ref, (i,), layout=layout, optimized=src_memory_space != plgpu.GMEM
        )
        o_ref[i, ...] = x

    x = jnp.arange(2 * m, dtype=jnp.float32).reshape(2, m)
    np.testing.assert_array_equal(kernel(x), x)

  @parameterized.product(
      src_memory_space=[plgpu.SMEM],
      layout=[plgpu.Layout.WGMMA_ROW, plgpu.Layout.WGMMA_COL],
  )
  def test_load_row_input_to_wgmma_with_transforms(self, src_memory_space, layout):
    self.skip_if_wg_semantics()

    m, k, n = 64, 128, 192
    key1, key2 = jax.random.split(jax.random.key(42), 2)
    if layout == plgpu.Layout.WGMMA_ROW:
      input_shape = (m,)
      broadcast_dim = 0
      expand_dim = 1
    else:
      input_shape = (k,)
      broadcast_dim = 1
      expand_dim = 0
    a = jax.random.uniform(key1, shape=input_shape, dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(k, n), dtype=jnp.float16)
    def kernel(x_ref, y_ref, o_ref):
      x = plgpu.load(x_ref, (), layout=layout)
      x = lax.broadcast_in_dim(x, (m, k), [broadcast_dim])

      def compute(acc_ref):
        plgpu.wgmma(acc_ref, x, y_ref)
        return acc_ref[...]

      out = pl.run_scoped(compute, plgpu.ACC((m, n), jnp.float32))
      o_ref[...] = out
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([m, n], jnp.float32),
        in_specs=(
            pl.BlockSpec(memory_space=src_memory_space),
            plgpu.BlockSpec(
                transforms=(
                    plgpu.TilingTransform((8, 64)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
        ),
        out_specs=plgpu.BlockSpec(memory_space=plgpu.SMEM),
    )

    out_ref = (
        jnp.broadcast_to(jnp.expand_dims(a, axis=expand_dim), (m, k)) @ b
    )
    np.testing.assert_allclose(f(a, b), out_ref, rtol=1e-3)


class PallasCallSm90AWGTest(
    PallasCallSm90ATest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class PallasCallSm100ATest(PallasSm100ATest):

  def test_tmem(self):
    self.skip_if_wg_semantics()  # TMEM read not wired up in the WG get rule.
    swizzle_elems = 128 // jnp.dtype(jnp.float32).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(128),
    )
    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((128, 128), jnp.float32),
        scratch_shapes=[
            plgpu.TMEM((128, 128), jnp.float32),
            plgpu.TMEM((128, 128), jnp.float32),
            plgpu.SMEM((128, 128), jnp.float32, transforms=transforms),
            plgpu.Barrier(),
        ],
        num_threads=1,
        thread_name="x",
    )
    def kernel(x_ref, y_ref, tmem_ref, tmem_ref2, smem_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      # Exercise TMEM by roundtripping SMEM -> TMEM -> TMEM -> SMEM.
      x_val = plgpu.load(smem_ref, (), layout=plgpu.Layout.TCGEN05)
      tmem_ref[...] = x_val + 1
      plgpu.commit_tmem()
      tmem_ref2[...] = tmem_ref[...]
      plgpu.commit_tmem()
      smem_ref[...] = tmem_ref2[...]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(
        jax.random.key(0), shape=(128, 128), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_array_equal(x_result, x + 1)

  @parameterized.product(shape=[(128, 128)],
                         swizzle=[128, 64, 32],
                         dtype=[jnp.float16, jnp.bfloat16],
                         lhs_tmem=[False, True],
                         transpose_rhs=[False, True],
                         transpose_lhs=[False, True])
  def test_simple_matmul(self, shape, swizzle,
                         dtype=jnp.float16,
                         lhs_tmem=False,
                         transpose_lhs=False,
                         transpose_rhs=False):
    self.skip_if_wg_semantics()
    if transpose_lhs and lhs_tmem:
      self.skipTest("TMEM transpose not supported.")
    # Test a matmul with a single block.
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(a_smem, b_smem, out_ref, acc_tmem, scratch_smem, barrier_ref,
               a_tmem_ref):
      if transpose_lhs:
        a_smem = plgpu.transpose_ref(a_smem, (1, 0))
      if transpose_rhs:
        b_smem = plgpu.transpose_ref(b_smem, (1, 0))
      if lhs_tmem:
        lhs_ref = a_tmem_ref
        lhs_ref[...] = plgpu.load(a_smem, (), layout=plgpu.Layout.TCGEN05)
        plgpu.commit_tmem()
      else:
        lhs_ref = a_smem
      plgpu.tcgen05_mma(acc_tmem,
                        lhs_ref,
                        b_smem,
                        barrier_ref,
                        accumulate=False)
      plgpu.barrier_wait(barrier_ref)
      scratch_smem[...] = acc_tmem[...].astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_smem, out_ref)
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.TMEM(shape, jnp.float32, packed=False),
        plgpu.SMEM(shape, dtype, transforms=transforms),
        plgpu.Barrier(for_tensor_core=True),
    ]
    if lhs_tmem:
      scratch_shapes.append(plgpu.TMEM(shape, dtype, packed=True))
    else:
      scratch_shapes.append(None)

    f = self.pallas_call(
        kernel,
        in_specs=(
            plgpu.BlockSpec(transforms=transforms, memory_space=plgpu.SMEM),
            plgpu.BlockSpec(transforms=transforms, memory_space=plgpu.SMEM),
        ),
        out_specs=plgpu.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        scratch_shapes=scratch_shapes,
    )
    x = jax.random.uniform(jax.random.key(0), shape=shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=shape, dtype=dtype)
    result = f(x, y)
    if transpose_lhs:
      x = jnp.transpose(x, (1, 0))
    if transpose_rhs:
      y = jnp.transpose(y, (1, 0))
    expected = x @ y
    np.testing.assert_allclose(result, expected, rtol=1e-3)

  @parameterized.parameters(
      ((256, 256), (256, 256), 128, jnp.float16),
      # Test additional shape combinations.
      ((256, 128), (128, 128), 128, jnp.float16),
      ((256, 64), (64, 256), 128, jnp.float16),
      # Test bfloat16.
      ((256, 256), (256, 256), 128, jnp.bfloat16),
      # Test additional swizzles.
      ((256, 256), (256, 256), 64, jnp.float16),
      ((256, 256), (256, 256), 32, jnp.float16),
  )
  def test_simple_collective_matmul(self, lhs_shape, rhs_shape, swizzle, dtype):
    self.skip_if_wg_semantics()
    # Test a collective (paired CTA) matmul on a single block.
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    acc_shape = (lhs_shape[0], rhs_shape[1])
    _acc_shape = (lhs_shape[0] // 2, rhs_shape[1])
    _lhs_shape = (lhs_shape[0] // 2, lhs_shape[1])
    _rhs_shape = (rhs_shape[0], rhs_shape[1] // 2)

    def kernel(a_gmem, b_gmem, out_gmem):
      cluster_idx = lax.axis_index("x")
      slice_lhs = pl.ds(cluster_idx * _lhs_shape[0], _lhs_shape[0])
      slice_rhs = pl.ds(cluster_idx * _rhs_shape[1], _rhs_shape[1])

      @functools.partial(pl.run_scoped,
        a_smem=plgpu.SMEM(_lhs_shape, dtype, transforms=transforms),
        b_smem=plgpu.SMEM(_rhs_shape, dtype, transforms=transforms),
        acc_tmem=plgpu.TMEM(_acc_shape, jnp.float32, collective=True),
        scratch_smem=plgpu.SMEM(_acc_shape, dtype, transforms=transforms),
        tma_barrier=plgpu.Barrier(),
        mma_barrier=plgpu.Barrier(for_tensor_core=True),
        cluster_barrier=plgpu.ClusterBarrier(collective_axes=("x",)),
      )
      def _scoped(a_smem, b_smem,
                  acc_tmem, scratch_smem, tma_barrier, mma_barrier, cluster_barrier):
        plgpu.copy_gmem_to_smem(a_gmem.at[slice_lhs, :], a_smem, tma_barrier)
        plgpu.barrier_wait(tma_barrier)
        plgpu.copy_gmem_to_smem(b_gmem.at[:, slice_rhs], b_smem, tma_barrier)
        plgpu.barrier_wait(tma_barrier)

        plgpu.barrier_arrive(cluster_barrier)
        plgpu.barrier_wait(cluster_barrier)

        plgpu.tcgen05_mma(acc_tmem,
                          a_smem,
                          b_smem,
                          mma_barrier,
                          accumulate=False,
                          collective_axis="x")
        plgpu.barrier_wait(mma_barrier)
        scratch_smem[...] = acc_tmem[...].astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(scratch_smem, out_gmem.at[slice_lhs, :])
        plgpu.wait_smem_to_gmem(0)

    f = self.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct(acc_shape, dtype),
      grid=(1,),
      grid_names=("_",),
      cluster=(2,),
      cluster_names=("x",),
    )
    x = jax.random.uniform(jax.random.key(0), shape=lhs_shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=rhs_shape, dtype=dtype)
    result = f(x, y)
    expected = x @ y
    np.testing.assert_allclose(result, expected, rtol=1e-3)

  @parameterized.parameters((0,), (1,))
  def test_mma_barrier_indexing(
      self, barrier_index, shape=(128, 128), swizzle=128, dtype=jnp.float16
  ):
    self.skip_if_wg_semantics()
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(a_smem, b_smem, out_ref, acc_tmem, scratch_smem, barrier_ref):
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier_ref.at[barrier_index],
          accumulate=False,
      )
      plgpu.barrier_wait(barrier_ref.at[barrier_index])
      scratch_smem[...] = acc_tmem[...].astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_smem, out_ref)
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.TMEM(shape, jnp.float32, packed=False),
        plgpu.SMEM(shape, dtype, transforms=transforms),
        plgpu.Barrier(num_barriers=2, for_tensor_core=True),
    ]
    f = self.pallas_call(
        kernel,
        in_specs=(
            plgpu.BlockSpec(transforms=transforms, memory_space=plgpu.SMEM),
            plgpu.BlockSpec(transforms=transforms, memory_space=plgpu.SMEM),
        ),
        out_specs=plgpu.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        scratch_shapes=scratch_shapes,
    )
    x = jax.random.uniform(jax.random.key(0), shape=shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=shape, dtype=dtype)
    result = f(x, y)
    expected = x @ y
    np.testing.assert_allclose(result, expected, rtol=1e-3)


class PallasCallSm100AWGTest(
    PallasCallSm100ATest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class PipelineTest(PallasTest):

  def test_pipeline_mode(self):
    def body(x_ref, y_ref, o_ref):
      x = x_ref[:]
      y = y_ref[:]
      o_ref[:] = x + y

    data_size =  64 * 256
    block_size = 256

    x = jnp.arange(data_size, dtype=jnp.float32)
    y = jnp.arange(data_size, dtype=jnp.float32)
    in_specs = [
        pl.BlockSpec((block_size,), lambda *i: i, pipeline_mode=pl.Buffered(2)),
        pl.BlockSpec((block_size,), lambda *i: i, pipeline_mode=pl.Buffered(1))
    ]
    out_specs = pl.BlockSpec((block_size,), lambda *i: i)

    @jax.jit
    def vadd(x, y):
      return self.pallas_call(
          body,
          out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=data_size // block_size,
      )(x, y)

    with self.assertRaisesRegex(Exception, "Pipeline mode is not supported"):
      vadd(x, y)

  def test_manual(self):
    max_concurrent_steps = 2
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      return pl.run_scoped(
          functools.partial(scoped_kernel, x_gmem, o_gmem),
          plgpu.SMEM((max_concurrent_steps, 32, 16), jnp.float32),
          plgpu.SMEM((max_concurrent_steps, 32, 16), jnp.float32),
          plgpu.Barrier(num_barriers=max_concurrent_steps),
      )

    def scoped_kernel(x_gmem, o_gmem, x_smem, o_smem, barrier):
      gmem_slice = pl.ds(pl.program_id(0) * 32, 32)

      def body(step, _):
        slot = step % max_concurrent_steps

        # Wait for the current GMEM->SMEM copy to complete.
        plgpu.barrier_wait(barrier.at[slot])
        # Wait for the previous output SMEM->GMEM copy to complete.
        plgpu.wait_smem_to_gmem(max_concurrent_steps - 1)

        o_smem.at[slot][...] = x_smem.at[slot][...] + 1.0

        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(
            o_smem.at[slot], o_gmem.at[gmem_slice, pl.ds(step * 16, 16)]
        )

        fetch_step = step + max_concurrent_steps
        fetch_slot = slot  # (x + y) % y == x % y
        jax.lax.cond(
            fetch_step < num_steps,
            lambda: plgpu.copy_gmem_to_smem(
                x_gmem.at[gmem_slice, pl.ds(fetch_step * 16, 16)],
                x_smem.at[fetch_slot],
                barrier.at[fetch_slot],
            ),
            lambda: None,
        )
        return ()

      # Initialize the pipeline.
      for slot in range(min(max_concurrent_steps, num_steps)):
        plgpu.copy_gmem_to_smem(
            x_gmem.at[gmem_slice, pl.ds(slot * 16, 16)],
            x_smem.at[slot],
            barrier.at[slot],
        )

      jax.lax.fori_loop(0, num_steps, body, ())

      # Finalize the pipeline.
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(32 * 4 * 64).reshape(32 * 4, 64).astype(jnp.float32)
    kernel_fn = self.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(4, 1),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  @parameterized.product(
      transforms=(
          (),
          (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128)),
      ),
      repeats=(1, 3),
  )
  def test_emit(self, transforms, repeats):
    if transforms:
      self.skip_if_wg_semantics()

    num_steps = 4

    def kernel(x_gmem, o_gmem):
      for _ in range(repeats):
        plgpu.emit_pipeline(
            kernel_body,
            in_specs=[
                plgpu.BlockSpec(
                    (64, 64), lambda i: (0, i), transforms=transforms
                )
            ],
            out_specs=[
                plgpu.BlockSpec(
                    (64, 64), lambda i: (0, i), transforms=transforms
                )
            ],
            grid=(num_steps,),
            max_concurrent_steps=2,
        )(x_gmem, o_gmem)

    def kernel_body(_, x_smem, o_smem):
      # +1 for the indexing done by ``emit_pipeline`.
      self.assertLen(x_smem.transforms, len(transforms) + 1)
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(64 * num_steps * 64)
    x = x.reshape(-1, num_steps * 64).astype(jnp.float32)
    kernel_fn = self.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  def test_nested_emit(self):
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      plgpu.emit_pipeline(
          nested_kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          grid=(),
      )(x_gmem, o_gmem)

    def nested_kernel(_, x_gmem, o_gmem):
      plgpu.emit_pipeline(
          nested_kernel_body,
          in_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          out_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          grid=(num_steps,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def nested_kernel_body(_, x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(32 * num_steps * 16)
    x = x.reshape(-1, num_steps * 16).astype(jnp.float32)
    kernel_fn = self.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  def test_emit_with_grid_invariant_output(self):
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          out_specs=[pl.BlockSpec((32, 16), lambda i: (0, 0))],
          grid=(num_steps,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(_, x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(32 * num_steps * 16)
    x = x.reshape(-1, num_steps * 16).astype(jnp.float32)
    kernel_fn = self.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    y = jnp.empty_like(x)
    for i in range(num_steps):
      i_slice = slice(16 * i, 16 * (i + 1))
      y = y.at[:, :16].set(x[:, i_slice] + 1)
    # We only compare the elements in the first 16 columns, because the rest
    # are never written to.
    np.testing.assert_array_equal(kernel_fn(x)[:, :16], y[:, :16])

  def test_emit_with_parallel_grid(self):
    num_steps1 = 4
    num_steps2 = 5

    def kernel(x_gmem, o_gmem):
      pid = pl.program_id(0)
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((32, 16), lambda i: (pid, i))],
          out_specs=[pl.BlockSpec((32, 16), lambda i: (pid, i))],
          grid=(num_steps2,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(_, x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(num_steps1 * 32 * num_steps2 * 16)
    x = x.reshape(-1, num_steps2 * 16).astype(jnp.float32)
    kernel_fn = self.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(num_steps1,),
    )
    y = x + 1.0
    np.testing.assert_array_equal(kernel_fn(x), y)

  @parameterized.product(static=[False, True], short=[False, True])
  def test_emit_with_2d_grid(self, static, short):
    num_steps1 = 4
    num_steps2 = 5
    if short:
      num_steps1 = num_steps2 = 1

    def kernel(x_gmem, o_gmem):
      grid = (num_steps1, num_steps2)
      if static:
        grid = jax.tree.map(jnp.asarray, grid)

      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((32, 16, 8), lambda i, j: (0, i, j))],
          out_specs=[pl.BlockSpec((32, 16, 8), lambda i, j: (0, i, j))],
          grid=grid,
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(_, x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(32 * num_steps1 * 16 * num_steps2 * 8)
    x = x.reshape(-1, num_steps1 * 16, num_steps2 * 8).astype(jnp.float32)
    kernel_fn = self.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  def test_emit_with_carry(self):
    num_steps = 4

    def kernel(o_gmem):
      plgpu.emit_pipeline(
          kernel_body,
          out_specs=[pl.BlockSpec((64, 64), lambda i: (0, i))],
          grid=(num_steps,),
          max_concurrent_steps=2,
          init_carry=0,
      )(o_gmem)

    def kernel_body(_, o_smem, carry):
      o_smem[...] = lax.broadcast(carry, o_smem.shape)
      return carry + 1

    kernel_fn = self.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((64, num_steps * 64), jnp.int32),
    )
    np.testing.assert_array_equal(
        kernel_fn(), jnp.tile(jnp.repeat(jnp.arange(num_steps), 64), (64, 1))
    )


class PipelineWGTest(
    PipelineTest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class PipelineSm90ATest(PallasSm90ATest):

  def test_realistic_matmul(self):
    self.skip_if_wg_semantics()  # Needs WGMMA to support slices.

    dtype = jnp.float16
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    grid_m, grid_k, grid_n = 132, 10, 4
    tile_m = tile_n = 128
    assert tile_m % elems_128b == 0
    tile_k = elems_128b
    m, k, n = grid_m * tile_m, grid_k * tile_k, grid_n * tile_n

    transforms = ()
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      transforms = (
          plgpu.TilingTransform((8, elems_128b)),
          plgpu.SwizzleTransform(128),
      )

    def kernel(a_gmem, b_gmem, o_smem, acc):
      def kernel_body(_, a_smem, b_smem):
        assert a_smem.shape == (tile_m, tile_k)
        assert b_smem.shape == (tile_k, tile_n)
        plgpu.wgmma(acc, a_smem, b_smem)
        plgpu.wgmma_wait(1)

      pid_m = pl.program_id(0)
      pid_n = pl.program_id(1)
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[
              plgpu.BlockSpec(
                  (tile_m, tile_k), lambda k: (pid_m, k), transforms=transforms
              ),
              plgpu.BlockSpec(
                  (tile_k, tile_n), lambda k: (k, pid_n), transforms=transforms
              ),
          ],
          grid=(grid_k,),
          max_concurrent_steps=2,
          delay_release=1,
      )(a_gmem, b_gmem)

      o_smem[...] = acc[...].astype(dtype)

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(m, k), dtype=dtype)
    b = jax.random.uniform(key2, shape=(k, n), dtype=dtype)

    res = self.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=plgpu.GMEM),
            pl.BlockSpec(memory_space=plgpu.GMEM),
        ],
        out_specs=plgpu.BlockSpec(
            (tile_m, tile_n), lambda m, n: (m, n), transforms=transforms
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float16),
        scratch_shapes=[plgpu.ACC((tile_m, tile_n), jnp.float32)],
        grid=(grid_m, grid_n),
    )(a, b)
    np.testing.assert_array_equal(res, a @ b)


class PipelineSm90AWGTest(
    PipelineSm90ATest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class WarpSpecializedPipelineTest(PallasTest):

  @parameterized.product(m=[512], n=[512], repeats=[1, 3],
                         manual_consumed_barriers=[False, True])
  def test_pipelined_copy(self, m, n, repeats, manual_consumed_barriers):
    self.skip_if_wg_semantics()  # Times out!

    x = jax.random.uniform(jax.random.key(0), (m, n), dtype=jnp.float16)
    blk_m = blk_n = 64

    def copy_kernel(_, x_smem, o_smem, o_last_block_smem, *consumed_barriers):
      # TODO(justinfu): Have each wg compute a separate slice
      # after multiple-indexers are supported.
      # This is currently a race, but the values written are the same.
      o_smem[...] = x_smem[...]
      o_last_block_smem[...] = x_smem[...]
      if manual_consumed_barriers:
        [x_barrier] = consumed_barriers
        plgpu.barrier_arrive(x_barrier)

    spec = pl.BlockSpec(
        block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
    )
    def body(*gmem_refs):
      pipeline = mgpu_pipeline.emit_pipeline_warp_specialized(
          copy_kernel,
          grid=(m // blk_m, n // blk_n),
          memory_registers=40,
          max_concurrent_steps=2,
          num_compute_wgs=2,
          wg_axis="wg",
          manual_consumed_barriers=manual_consumed_barriers,
          in_specs=[spec],
          out_specs=[
              spec,
              # Create an index-invariant output.
              pl.BlockSpec(
                  block_shape=(blk_m, blk_n), index_map=lambda i, j: (0, 0)
              ),
          ],
      )
      for _ in range(repeats):
        pipeline(*gmem_refs)  # Make sure we can run the pipeline multiple times
    kernel = self.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((m, n), jnp.float16),
            jax.ShapeDtypeStruct((blk_m, blk_n), jnp.float16),
        ),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=(1,),
        grid_names=("_",),
        num_threads=3,
        thread_name="wg",
    )
    out, out_last_block = kernel(x)
    np.testing.assert_array_equal(out, x)
    np.testing.assert_array_equal(out_last_block, x[-blk_m:, -blk_n:])

  @parameterized.product(
      m=[256, 64], n=[256, 64], num_compute_wgs=[1, 2], static=[False, True]
  )
  def test_elementwise_add(self, m, n, num_compute_wgs, static):
    self.skip_if_wg_semantics()  # Crashes!

    blk_m = blk_n = 64
    spec = pl.BlockSpec(
        block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
    )

    def tiled_add_kernel(_, x_smem, y_smem, o_smem):
      # TODO(justinfu): Have each wg compute a separate slice
      # after multiple-indexers are supported.
      # This is currently a race, but the values written are the same.
      o_smem[...] = x_smem[...] + y_smem[...]

    def pipeline(*gmem_refs):
      grid = (m // blk_m, n // blk_n)
      if not static:
        grid = jax.tree.map(jnp.asarray, grid)
      return mgpu_pipeline.emit_pipeline_warp_specialized(
          tiled_add_kernel,
          grid=grid,
          max_concurrent_steps=2,
          num_compute_wgs=num_compute_wgs,
          memory_registers=40,
          wg_axis="wg",
          in_specs=[spec, spec],
          out_specs=[spec],
      )(*gmem_refs)

    kernel = self.kernel(
        pipeline,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=(1,),
        grid_names=("_",),
        num_threads=num_compute_wgs + 1,
        thread_name="wg",
    )
    x = jax.random.uniform(jax.random.key(0), (m, n), dtype=jnp.float32)
    y = jax.random.uniform(jax.random.key(1), (m, n), dtype=jnp.float32)
    np.testing.assert_allclose(kernel(x, y), x + y, atol=1e-4)

  def test_carry_accumulate(self, m=256, n=256, num_compute_wgs=2):
    blk_m = blk_n = 64

    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((blk_m, blk_n), jnp.float32),
        scratch_shapes=[
            plgpu.SMEM((blk_m, blk_n), jnp.float32),
        ],
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=(1,),
        grid_names=("_",),
        num_threads=num_compute_wgs + 1,
        thread_name="wg",
    )
    def kernel(x_gmem, acc_gmem, acc_smem):
      def _compute_thread(pipeline_fn):
        # Cast the init value to the same layout as x_smem, so the pipeline loop
        # carry has a constant signature.
        o_acc = plgpu.layout_cast(
          jnp.full((blk_m, blk_n,), 0, dtype=jnp.float32),
          plgpu.Layout.WG_STRIDED((blk_m, blk_n), vec_size=2))
        # Pass control to the pipeline emitter and return the final carry.
        o_final = pipeline_fn(o_acc)
        # Note that both compute WGs are doing identical work so the potential
        # race condition on the store here won't affect the result.
        acc_smem[...] = o_final
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(acc_smem, acc_gmem)
        plgpu.wait_smem_to_gmem(0)

      def tiled_acc_kernel(_, x_smem, carry):
        new_carry = x_smem[...] + carry
        return new_carry

      pipeline = mgpu_pipeline.emit_pipeline_warp_specialized(
          tiled_acc_kernel,
          grid=(m // blk_m, n // blk_n),
          max_concurrent_steps=2,
          num_compute_wgs=num_compute_wgs,
          memory_registers=40,
          wg_axis="wg",
          compute_context=_compute_thread,
          in_specs=[
              pl.BlockSpec(
                  block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
              )
          ],
          out_specs=[],
      )
      pipeline(x_gmem)

    x = jax.random.uniform(jax.random.key(0), (m, n), dtype=jnp.float32)
    ref = jnp.sum(jnp.stack(np.split(x, m // blk_m, axis=0)), axis=0)
    ref = jnp.sum(jnp.stack(np.split(ref, n // blk_n, axis=1)), axis=0)
    np.testing.assert_allclose(kernel(x), ref, atol=1e-4)


class WarpSpecializedPipelineWGTest(
    WarpSpecializedPipelineTest,
    lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
):
  ...


class CoreMapTest(PallasTest, jtu.CudaArchSpecificTest):

  def test_multiple_wg(self):

    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((2, 128), np.int32),
        num_threads=2,
        thread_name="wg",
    )
    def kernel(o_ref):
      wg_idx = jax.lax.axis_index("wg")
      o_ref[wg_idx] = jnp.broadcast_to(wg_idx, (128,))

    np.testing.assert_array_equal(
        kernel(), np.repeat(np.arange(2), 128).reshape(2, 128)
    )

  def test_multiple_wg_with_grid(self):

    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((4, 2, 128), np.int32),
        grid=(2, 2),
        grid_names=("x", "y"),
        num_threads=2,
        thread_name="wg",
    )
    def kernel(o_ref):
      xy_idx = jax.lax.axis_index(("x", "y"))
      yx_idx = jax.lax.axis_index(("y", "x"))
      wg_idx = jax.lax.axis_index("wg")
      num_wgs = jax.lax.axis_size("wg")
      o_ref[xy_idx, wg_idx] = jnp.broadcast_to(
          yx_idx * num_wgs + wg_idx, (128,)
      )

    np.testing.assert_array_equal(
        kernel(), np.repeat([0, 1, 4, 5, 2, 3, 6, 7], 128).reshape(4, 2, 128)
    )

  def test_multiple_wg_with_squashed_grid(self):
    # Tests whether a grid with >3 logical dimensions is correctly squashed to
    # 3 CUDA grid dimensions.
    b = 4
    x_dim = 3
    y_dim = 5
    z_dim = 7
    num_threads = 2

    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros(
            (b, x_dim, y_dim, z_dim, num_threads, 128), np.int32
        ),
        grid=(b, x_dim, y_dim, z_dim),
        grid_names=("b", "x", "y", "z"),
        num_threads=num_threads,
        thread_name="wg",
    )
    def kernel(o_ref):
      b_idx = jax.lax.axis_index("b")
      x_idx = jax.lax.axis_index("x")
      y_idx = jax.lax.axis_index("y")
      z_idx = jax.lax.axis_index("z")
      wg_idx = jax.lax.axis_index("wg")
      bxyzw_idx = jax.lax.axis_index(("b", "x", "y", "z", "wg"))
      o_ref[b_idx, x_idx, y_idx, z_idx, wg_idx] = jnp.broadcast_to(
          bxyzw_idx, (128,)
      )

    result = kernel()[:, :, :, :, :, 0]
    ref = np.arange(b * x_dim * y_dim * z_dim * num_threads).reshape(
        result.shape
    )
    np.testing.assert_array_equal(result, ref)

  def test_cross_wg_barrier(self):
    self.skip_if_wg_semantics()  # Times out!

    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((2, 128), np.int32),
        # Each warpgroup is a single logical thread!
        scratch_shapes=[plgpu.Barrier(num_arrivals=2)],
        num_threads=2,
        thread_name="wg",
    )
    def kernel(o_ref, barrier):
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      wg_idx = jax.lax.axis_index("wg")
      o_ref[wg_idx] = jnp.broadcast_to(wg_idx, (128,))

    np.testing.assert_array_equal(
        kernel(), np.repeat([0, 1], 128).reshape(2, 128)
    )

  def test_cluster(self):
    self.skip_if_wg_semantics()  # Needs debug_print in the MGPU dialect.

    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros(128, np.int32),
        grid=(2,),
        grid_names=("x",),
        cluster=(2,),
        cluster_names=("cluster",),
    )
    def kernel(ref):
      block_idx = jax.lax.axis_index("x")
      cluster_idx = jax.lax.axis_index("cluster")
      pl.debug_print("block: {} cluster: {}", block_idx, cluster_idx)

      ref[...] = ref[...]

    with self.capture_stdout() as output:
      jax.block_until_ready(kernel())
    self.assertEqual(
        set(output().splitlines()),
        {
            "block: 0 cluster: 0",
            "block: 1 cluster: 0",
            "block: 0 cluster: 1",
            "block: 1 cluster: 1",
        },
    )

  def test_realistic_matmul_with_cluster(self):
    self.skip_if_wg_semantics()  # Needs WGMMA to support slices.
    self.skip_unless_sm90a()  # Requires WGMMA.

    dtype = jnp.float16
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    grid_m, grid_k, grid_n = 132, 10, 32
    # TODO(slebedev): Remove ``grid_tile_n`` to simplify the test.
    grid_tile_n = 4
    assert grid_n % grid_tile_n == 0
    cluster_m = 2
    cluster_n = 2
    cluster_tile_n = min(cluster_n, grid_tile_n)
    tile_m = tile_n = 128
    assert tile_m % elems_128b == 0
    tile_k = elems_128b
    m, k, n = grid_m * tile_m, grid_k * tile_k, grid_n * tile_n

    transforms = (
        plgpu.TilingTransform((8, elems_128b)),
        plgpu.SwizzleTransform(128),
    )

    max_concurrent_steps = 2
    delay_release = 1

    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_shapes=[
            plgpu.SMEM(
                (max_concurrent_steps, tile_m, tile_k),
                dtype,
                transforms=transforms,
            ),
            plgpu.SMEM(
                (max_concurrent_steps, tile_k, tile_n),
                dtype,
                transforms=transforms,
            ),
            plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
            plgpu.ACC((tile_m, tile_n), jnp.float32),
            plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),
            plgpu.ClusterBarrier(
                collective_axes=(("x", "z"), "y"),
                num_barriers=max_concurrent_steps,
            ),
        ],
        grid=(grid_tile_n, grid_m, grid_n // grid_tile_n),
        grid_names=("tile_n", "m", "n"),
        cluster=(cluster_tile_n, cluster_m, cluster_n // cluster_tile_n),
        cluster_names=("x", "y", "z"),
    )
    def kernel(
        a_gmem,
        b_gmem,
        o_gmem,
        a_smem,
        b_smem,
        o_smem,
        acc,
        barrier,
        cluster_barrier,
    ):
      m_slice = pl.ds(lax.axis_index("m") * tile_m, tile_m)
      n_slice = pl.ds(
          (lax.axis_index("tile_n") + lax.axis_index("n") * grid_tile_n)
          * tile_n,
          tile_n,
      )

      def fetch(step, slot):
        if not isinstance(slot, int):  # Skip in initialization.
          plgpu.barrier_arrive(cluster_barrier.at[slot])
          plgpu.barrier_wait(cluster_barrier.at[slot])

        k_slice = pl.ds(step * tile_k, tile_k)
        plgpu.copy_gmem_to_smem(
            a_gmem.at[m_slice, k_slice],
            a_smem.at[slot],
            barrier.at[slot],
            collective_axes=("x", "z"),
        )
        plgpu.copy_gmem_to_smem(
            b_gmem.at[k_slice, n_slice],
            b_smem.at[slot],
            barrier.at[slot],
            collective_axes="y",
        )

      # Initialize the pipeline.
      for slot in range(min(max_concurrent_steps, grid_k)):
        fetch(slot, slot)

      def body(step, _):
        slot = step % max_concurrent_steps
        plgpu.barrier_wait(barrier.at[slot])

        plgpu.wgmma(acc, a_smem.at[slot], b_smem.at[slot])
        plgpu.wgmma_wait(delay_release)

        fetch_step = step + (max_concurrent_steps - delay_release)
        fetch_slot = lax.rem(fetch_step, max_concurrent_steps)
        jax.lax.cond(
            lax.bitwise_and(step >= delay_release, fetch_step < grid_k),
            lambda: fetch(fetch_step, fetch_slot),
            lambda: None,
        )
        return ()

      jax.lax.fori_loop(0, grid_k, body, ())

      # Finalize the pipeline.
      o_smem[...] = acc[...].astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, o_gmem.at[m_slice, n_slice])
      plgpu.wait_smem_to_gmem(0)

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(m, k), dtype=dtype)
    b = jax.random.uniform(key2, shape=(k, n), dtype=dtype)
    np.testing.assert_array_equal(kernel(a, b), a @ b)


class CoreMapWGTest(
    CoreMapTest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class PrettyPrintingTest(PallasTest):

  def test_load(self):

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([2, 128], jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=plgpu.BlockSpec(memory_space=plgpu.SMEM),
    )
    def kernel(x_ref, o_ref):
      for i in range(2):
        x = plgpu.load(x_ref, (i,))
        o_ref[i, ...] = x

    _ = str(jax.make_jaxpr(kernel)(jax.ShapeDtypeStruct((2, 128), jnp.float32)))

  def test_copy_primitives(self):
    num_steps = 4

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((64, 64), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
    )
    def kernel(x_gmem, o_gmem):
      # ``plgpu.emit_pipeline`` is implemented in terms of async copy and
      # synchronization primitives.
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((64, 64), lambda i: (0, i))],
          out_specs=[
              pl.BlockSpec(
                  (64, 64),
                  lambda i: (0, i),
              )
          ],
          grid=(num_steps,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(_, x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    _ = str(jax.make_jaxpr(kernel)(jax.ShapeDtypeStruct((64, 64), jnp.float32)))

  def test_wgmma(self):
    transforms = ()
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
      transforms = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
        in_specs=[
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
        ],
    )
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref[...], b_ref)
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    _ = str(
        jax.make_jaxpr(kernel)(
            jax.ShapeDtypeStruct((64, 128), jnp.float16),
            jax.ShapeDtypeStruct((128, 192), jnp.float16),
        )
    )


class ExportTest(PallasTest):

  def test_export_succeeds(self):
    out_shape = jax.ShapeDtypeStruct([128], jnp.float32)

    @functools.partial(self.pallas_call, out_shape=out_shape)
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    _ = export.export(kernel)(out_shape)


class ExamplesTest(PallasTest):

  # Basic
  def test_stage0(self):
    x = jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128)

    @functools.partial(self.kernel, out_shape=x)
    def kernel(l_ref, r_ref, o_ref):
      o_ref[...] = l_ref[...] + r_ref[...]

    np.testing.assert_allclose(kernel(x, x), x + x)

  # Multi-block kernels
  def test_stage1(self):
    row_block = 64
    x = jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128)

    @functools.partial(
        self.kernel, out_shape=x, grid=(2,), grid_names=("rows",)
    )
    def kernel(l_ref, r_ref, o_ref):
      my_slice = pl.ds(lax.axis_index("rows") * row_block, row_block)
      o_ref[my_slice] = l_ref[my_slice] + r_ref[my_slice]

    np.testing.assert_allclose(kernel(x, x), x + x)

  # Async copies
  def test_stage3(self):
    row_block, col_block = 64, 128

    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float16),
        scratch_shapes=[
            *([plgpu.SMEM((row_block, col_block), jnp.float16)] * 3),
            plgpu.Barrier(num_arrivals=2),
        ],
        grid=(2,),
        grid_names=("rows",),
    )
    def kernel(l_ref, r_ref, o_ref, l_smem, r_smem, o_smem, barrier):
      my_slice = pl.ds(lax.axis_index("rows") * row_block, row_block)
      plgpu.copy_gmem_to_smem(l_ref.at[my_slice], l_smem, barrier)
      plgpu.copy_gmem_to_smem(r_ref.at[my_slice], r_smem, barrier)
      plgpu.barrier_wait(barrier)
      o_smem[...] = l_smem[...] + r_smem[...]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, o_ref.at[my_slice])
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128)
    np.testing.assert_allclose(kernel(x, x), x + x)

  # Pipelining
  def test_stage4(self):
    row_block, col_block = 64, 32
    x = jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128)

    @functools.partial(
        self.kernel, out_shape=x, grid=(2,), grid_names=("rows",)
    )
    def kernel(l_ref, r_ref, o_ref):
      def compute(_, l_smem, r_smem, o_smem):
        o_smem[...] = l_smem[...] + r_smem[...]
      r = lax.axis_index("rows")
      block = pl.BlockSpec((row_block, col_block), lambda c: (r, c))
      plgpu.emit_pipeline(
          compute,
          grid=(l_ref.shape[1] // col_block,),
          in_specs=[block] * 2,
          out_specs=[block],
      )(l_ref, r_ref, o_ref)

    np.testing.assert_allclose(kernel(x, x), x + x)

  # Transforms
  def test_stage5(self):
    self.skip_if_wg_semantics()  # Needs WGMMA to support slices.

    row_block, col_block = 64, 32
    x = jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128)

    @functools.partial(
        self.kernel, out_shape=x, grid=(2,), grid_names=("rows",)
    )
    def kernel(l_ref, r_ref, o_ref):
      def compute(_, l_smem, r_smem, o_smem):
        o_smem[...] = l_smem[...] + r_smem[...]
      r = lax.axis_index("rows")
      block = plgpu.BlockSpec(
          (row_block, col_block),
          lambda c: (r, c),
          transforms=(
              plgpu.TilingTransform((8, 32)),
              plgpu.SwizzleTransform(64),
          ),
      )
      plgpu.emit_pipeline(
          compute,
          grid=(l_ref.shape[1] // col_block,),
          in_specs=[block] * 2,
          out_specs=[block],
      )(l_ref, r_ref, o_ref)

    np.testing.assert_allclose(kernel(x, x), x + x)


class SemaphoreTest(PallasTest):

  def test_lowering(self):
    # This is a smoke test until we add support for lowering of semaphore ops.
    def body(i_ref1, i_ref2, o_ref, sem_ref):
      del i_ref2  # Only here to have a different number of inputs and outputs.
      assert sem_ref.shape == (4,)
      assert jnp.issubdtype(sem_ref.dtype, pl.semaphore)
      o_ref[...] = i_ref1[...]
    x = jnp.arange(128, dtype=jnp.float32).reshape((128,))
    kernel = self.pallas_call(
        body,
        out_shape=x,
        scratch_shapes=[plgpu.SemaphoreType.REGULAR((4,))],
    )
    text = jax.jit(kernel).lower(x, x).as_text()
    self.assertIn(
        r"output_operand_aliases ="
        r" [#stablehlo.output_operand_alias<output_tuple_indices = [1],"
        r" operand_index = 2, operand_tuple_indices = []>]",
        text,
    )
    self.assertIn(
        r"(tensor<128xf32>, tensor<128xf32>, tensor<4xi32>) ->"
        r" (tensor<128xf32>, tensor<4xi32>)",
        text,
    )

  def test_basic(self):
    def body(o_ref, sem_ref):
      assert jnp.issubdtype(sem_ref.dtype, pl.semaphore)
      pl.semaphore_signal(sem_ref)
      o_ref[...] = jnp.ones_like(o_ref)
      pl.semaphore_wait(sem_ref)
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
        grid=(2,),
        grid_names=("x",),
    )
    text = jax.jit(kernel).lower().as_text()
    np.testing.assert_array_equal(kernel(), jnp.ones((128,), jnp.float32))
    # The semaphore array is scaled up by the grid size.
    self.assertIn(
        r"(tensor<128xf32>, tensor<2xi32>) -> (tensor<128xf32>, tensor<2xi32>)",
        text,
    )

  def test_with_profiler(self):
    # Dealing with profiler and semaphores together is tricky because they both
    # add extra outputs to the HLO op.
    def body(o_ref, sem_ref):
      assert jnp.issubdtype(sem_ref.dtype, pl.semaphore)
      with jax.named_scope("output"):
        o_ref[...] = jnp.ones_like(o_ref)
    with tempfile.TemporaryDirectory() as tmp_dir:
      kernel = plgpu.kernel(
          body,
          out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
          scratch_shapes=[plgpu.SemaphoreType.REGULAR],
          grid=(2,),
          grid_names=("x",),
          compiler_params=plgpu.CompilerParams(profile_space=32, profile_dir=tmp_dir),
      )
      text = jax.jit(kernel).lower().as_text()
      np.testing.assert_array_equal(kernel(), jnp.ones((128,), jnp.float32))
    self.assertIn(
        r"(tensor<128xf32>, tensor<2xi32>) ->"
        r" (tensor<128xf32>, tensor<2xi32>, tensor<512xui32>)",
        text,
    )


class ExamplesWGTest(
    ExamplesTest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class ExamplesSm90ATest(PallasSm90ATest):

  # WGMMA
  def test_stage6(self):
    self.skip_if_wg_semantics()  # Needs WGMMA to support slices.

    m_block = n_block = 64
    k_block = 32
    x = jnp.arange(128 * 128, dtype=jnp.float16).reshape(128, 128)

    @functools.partial(
        self.kernel, out_shape=x, grid=(2, 2), grid_names=("m", "n")
    )
    def kernel(l_ref, r_ref, o_ref):
      def compute(_, l_smem, r_smem, o_smem):
        def do_wgmma(acc_ref):
          plgpu.wgmma(acc_ref, l_smem, r_smem)
          return acc_ref[...]
        o_smem[...] += pl.run_scoped(do_wgmma, plgpu.ACC((m_block, n_block), jnp.float16))
      m = lax.axis_index("m")
      n = lax.axis_index("n")
      lo_transforms = (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(64))
      r_transforms = (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(64))
      plgpu.emit_pipeline(
          compute,
          grid=(l_ref.shape[1] // k_block,),
          in_specs=[
              plgpu.BlockSpec(
                  (m_block, k_block), lambda k: (m, k), transforms=lo_transforms
              ),
              plgpu.BlockSpec(
                  (k_block, n_block), lambda k: (k, n), transforms=r_transforms
              ),
          ],
          out_specs=[
              plgpu.BlockSpec(
                  (m_block, n_block), lambda k: (m, n), transforms=lo_transforms
              )
          ],
      )(l_ref, r_ref, o_ref)

    np.testing.assert_allclose(kernel(x, x), x @ x)

  # TODO(apaszke): Clusters and multicast


class ExamplesSm90AWGTest(
    ExamplesSm90ATest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


if __name__ == "__main__":
  absltest.main()
