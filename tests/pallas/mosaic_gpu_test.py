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

from collections.abc import Sequence
import contextlib
import dataclasses
import functools
import itertools
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
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
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


def _get_linearized_cuda_grid_index():
  shape = ()
  layout = plgpu.Layout.WG_SPLAT

  @plgpu.inline_mgpu(
      arg_types=(),
      return_type=plgpu.ShapeDtypeStruct(shape, jnp.int32, layout=layout),
  )
  def fn(_):
    grid_x = gpu_dialect.grid_dim(gpu_dialect.Dimension.x)
    grid_y = gpu_dialect.grid_dim(gpu_dialect.Dimension.y)
    block_x = gpu_dialect.block_id(gpu_dialect.Dimension.x)
    block_y = gpu_dialect.block_id(gpu_dialect.Dimension.y)
    block_z = gpu_dialect.block_id(gpu_dialect.Dimension.z)

    grid_idx = arith_dialect.addi(
        block_x,
        arith_dialect.addi(
            arith_dialect.muli(block_y, grid_x),
            arith_dialect.muli(block_z, arith_dialect.muli(grid_x, grid_y)),
        ),
    )

    return mgpu.FragmentedArray.splat(
        arith_dialect.index_cast(ir.IntegerType.get_signless(32), grid_idx),
        shape=shape,
        layout=layout.to_mgpu(),
        is_signed=False
    )
  return fn()


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
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

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

  def default_transforms(
      self, *, swizzle: int = 128, dtype: jnp.dtype
  ) -> Sequence[plgpu.MemoryRefTransform]:
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup:
      return ()
    swizzle_elems = 8 * swizzle // dtypes.itemsize_bits(dtype)
    return (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )


class PallasSm90ATest(PallasTest, jtu.CudaArchSpecificTest):

  def setUp(self):
    self.skip_unless_sm90a()
    super().setUp()


class PallasSm100ATest(PallasTest, jtu.CudaArchSpecificTest):

  def setUp(self):
    self.skip_unless_sm100a()
    super().setUp()


class PallasCallTest(PallasTest):

  def test_jitted_function_containing_multiple_pallas_calls(self):
    # This test aims to ensure that execution works correctly inside CUDA
    # graphs. This is complementary to the test in
    # jaxlib/mosaic/gpu/custom_call_test.cc that checks that such jitted
    # functions do invoke CUDA graphs.
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1

    @jax.jit
    def f(x):
      # Run the kernel 10 times because CUDA graphs only trigger for >= 5 ops.
      for _ in range(10):
        x = kernel(x)
      return x

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(f(x), x + 10)

  @parameterized.product(
      op=[
          lax.neg,
          lax.bitwise_not,
          lax.logistic,
          lax.exp,
          lambda x: x**2,
          lambda x: x**5,
          lax.rsqrt,
          lax.tanh,
          lax.log,
          jax.nn.gelu,
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

  @parameterized.product(shape=[(128,), (128, 64)])
  def test_reduce_sum(self, shape):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32)
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.broadcast_to(_sum_same_dtype(x_ref[...]), o_ref.shape)

    x = jnp.arange(math.prod(shape)).reshape(shape).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), jnp.sum(x))

  def test_reshape(self):
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

  def test_reshape_tiled(self):
    self.skip_if_wg_semantics()
    shape1, shape2 = (6 * 64, 8), (2, 3, 64, 8)

    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct(shape2, jnp.float32),
    )
    def kernel(x_ref, out_ref):
      y = plgpu.load(x_ref, (), layout=plgpu.Layout.WGMMA, optimized=False).reshape(shape2)
      out_ref[...] = y

    x = jnp.arange(math.prod(shape1)).reshape(shape1).astype(jnp.float32)
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

  def test_add_one_grid_pipelined_with_leading_sequential_dimension(self):
    @functools.partial(
        self.pallas_call,
        in_specs=[pl.BlockSpec((128, 16), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((128, 16), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct([128 * 2, 64], jnp.float32),
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=["sequential", "parallel"],
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

  @parameterized.parameters(jnp.bfloat16, jnp.int16, jnp.uint16)
  def test_inline_mgpu(self, jnp_type):
    dtype = jnp.dtype(jnp_type)
    is_signed = mgpu.utils.is_signed(dtype)
    shape = (128, 128)
    tile = (64, 128 // dtype.itemsize)
    tiled_shape = list(mgpu.tile_shape(shape, tile))

    key = jax.random.key(0)
    x = jax.random.uniform(key, (2, *shape), minval=-10, maxval=10).astype(
        dtype
    )

    transforms = (
        plgpu.TilingTransform(tile),
        plgpu.SwizzleTransform(128),
    )

    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup:
      pallas_call_transforms = ()
    else:
      pallas_call_transforms = transforms

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM(
                x.shape,
                dtype,
                transforms=pallas_call_transforms,
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
              plgpu.SwizzleTransform(128),
          )),),
          return_type=plgpu.ShapeDtypeStruct(
              shape, dtype, layout=plgpu.Layout.WGMMA
          ),
      )
      def foo(ctx, smem_ref):
        del ctx
        assert smem_ref.type.shape == tiled_shape, (smem_ref.type, tiled_shape)
        x = mgpu.FragmentedArray.load_tiled(
            smem_ref, swizzle=128, is_signed=is_signed
        )
        y = mgpu.FragmentedArray.splat(
            mgpu.c(1, x.mlir_dtype),
            shape=x.shape,
            layout=x.layout,
            is_signed=is_signed,
        )
        return (x + x + y)

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
            gmem_transform=(mgpu.TileTransform(tile)),
        )
        ctx.await_async_copy(0)

        # A dummy if statement to make sure we inline nested blocks correctly.
        is_leader_thread = mgpu.utils.single_thread_predicate()
        with mgpu.utils.when(is_leader_thread):
          pass

      # This time we slice inside the inline_mgpu body.
      store(arr, smem_ref, o_ref)

    np.testing.assert_array_equal(kernel(x), x[0] + x[0] + 1)

  @parameterized.parameters(
      plgpu.Layout.WGMMA,
      plgpu.Layout.WGMMA_UPCAST_2X,
      plgpu.Layout.WGMMA_UPCAST_4X,
      plgpu.Layout.TCGEN05,
  )
  def test_inline_mgpu_layout_args(self, layout: gpu_core.SomeLayout):
    quant_dtype = jnp.int8
    dtype = jnp.bfloat16
    mgpu_layout = layout.to_mgpu()
    shape = (128, 128)

    rngs = list(jax.random.split(jax.random.key(0)))
    x = jax.random.randint(rngs.pop(), shape, minval=-10, maxval=10).astype(
        quant_dtype
    )
    x_s = jax.random.uniform(
        rngs.pop(), shape[0], minval=-100, maxval=100
    ).astype(dtype)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),
                  pl.BlockSpec(memory_space=plgpu.GMEM)),
        scratch_shapes=[
            plgpu.SMEM(
                x.shape,
                dtype,
            ),
        ],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
    )
    def kernel(
        x_ref, x_scale_ref, o_ref, o_smem_ref,
    ):
      x = plgpu.load(x_ref, (), layout=layout, optimized=False).astype(x_scale_ref.dtype)
      x_s = plgpu.load(x_scale_ref, (), layout=layout.reduce(1), optimized=False)

      @plgpu.inline_mgpu(
          arg_types=(layout,layout.reduce(1)),
          return_type=plgpu.ShapeDtypeStruct(
              shape, dtype, layout=layout
          ),
      )
      def custom_broadcast(ctx, x_fa, xs_fa):
        del ctx
        return xs_fa.broadcast_in_dim(shape, [0], layout=mgpu_layout) * x_fa

      arr = custom_broadcast(x, x_s)
      o_smem_ref[...] = arr
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem_ref, o_ref)
      plgpu.wait_smem_to_gmem(0)

    np.testing.assert_array_equal(
        kernel(x, x_s),
        x.astype(dtype) * jnp.broadcast_to(x_s[:, None], x.shape),
    )

  def test_sync_copy(self):
    shape = (128, 128)
    transforms = self.default_transforms(dtype=jnp.float32)
    @functools.partial(
        self.pallas_call,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.SMEM(shape, jnp.float32, transforms=transforms)],
    )
    def kernel(x_ref, y_ref, scratch_ref):
      layout = plgpu.Layout.SMEM_GMEM_COPY(shape, jnp.float32, swizzle=128)
      # GMEM loads require optimized=False, because we can't prove coalescing.
      # But with this layout they should be fast.
      scratch_ref[...] = plgpu.load(x_ref, (), layout=layout, optimized=False)
      y_ref[...] = plgpu.layout_cast(scratch_ref[...], layout)

    x = jnp.arange(math.prod(shape), dtype=jnp.float32).reshape(shape)
    np.testing.assert_array_equal(kernel(x), x)

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
    # TODO(b/415721295):Remove after the minimal jaxlib version is 0.8.2.
    if not hasattr(mgpu.dialect, "TMAReduction"):
      self.skip_if_wg_semantics()

    @functools.partial(
        self.pallas_call,
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

  def test_collective_copy_gmem_to_smem(self):
    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128), jnp.float32),
        scratch_shapes=dict(
            smem_ref=plgpu.SMEM((128,), jnp.float32),
            barrier_ref=plgpu.Barrier(),
        ),
        cluster=(2,),
        cluster_names=("cluster",),
    )
    def kernel(x_ref, y_ref, smem_ref, barrier_ref):
      # Specifying collective_axes will enable TMA multicast automatically.
      plgpu.copy_gmem_to_smem(
          x_ref, smem_ref, barrier_ref, collective_axes="cluster"
      )
      plgpu.barrier_wait(barrier_ref)
      plgpu.copy_smem_to_gmem(smem_ref, y_ref.at[jax.lax.axis_index("cluster")])
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(128, dtype=jnp.float32)
    y = kernel(x)
    # Each block gets the same data and writes it out.
    np.testing.assert_array_equal(y, jnp.stack([x, x], axis=0))

  @parameterized.product(indexer=[..., slice(128), slice(None, 128)])
  def test_async_prefetch(self, indexer):

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
      plgpu.async_prefetch(x_ref_gmem.at[indexer])
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
    transforms = self.default_transforms(dtype=jnp.int32)
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
                transforms=transforms,
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
    transforms = self.default_transforms(dtype=jnp.float32)

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
        transforms=transforms,
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
    ts = self.default_transforms(dtype=jnp.float32)
    def kernel(x_ref, o_ref, barrier_ref):
      def body(tmp_ref):
        plgpu.copy_gmem_to_smem(x_ref, tmp_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)
        o_ref[...] = tmp_ref[...] * 2
      pl.run_scoped(body, plgpu.SMEM((128, 64), jnp.float32, transforms=ts))

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.BlockSpec(transforms=ts, memory_space=plgpu.SMEM)
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([128, 64], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(128 * 64, dtype=jnp.float32).reshape(128, 64)
    np.testing.assert_array_equal(f(x), x * 2)

  @jtu.skip_if_mosaic_gpu_exceeds_shared_memory(device_patterns="RTX PRO 6000 Blackwell")
  def test_scoped_copy_with_user_transforms(self):
    self.skip_if_wg_semantics()

    def kernel(x_ref, o_ref, barrier_ref):
      def body(tmp_ref):
        tmp_ref = plgpu.unswizzle_ref(tmp_ref, 128)
        tmp_ref = plgpu.untile_ref(tmp_ref, (8, 32))
        plgpu.copy_gmem_to_smem(x_ref, tmp_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)
        o_ref[...] = tmp_ref[...] * 2
      pl.run_scoped(body, plgpu.SMEM((8, 4, 8, 32), jnp.float32))

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([64, 128], jnp.float32),
        in_specs=(in_spec,),
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
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
        out_shape=jax.ShapeDtypeStruct([2, 64, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    np.testing.assert_array_equal(f(x), np.stack([x, x], axis=0))

  @parameterized.parameters(
      ((),),
      ((plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128)),),
      (
          (
              plgpu.TilingTransform((8, 32)),
              plgpu.TransposeTransform((1, 0, 2, 3)),
              plgpu.SwizzleTransform(128),
          ),
      ),
  )
  def test_copy_gmem_to_smem_gather(self, transforms):
    if not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Only works on a GPU with capability >= sm100")
    self.skip_if_wg_semantics()
    dtype = jnp.int32
    out_shape = (64, 128)
    shape = (128, 64 + out_shape[-1])
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(out_shape, dtype),
        out_specs=plgpu.BlockSpec(memory_space=plgpu.SMEM, transforms=transforms),
        in_specs=(
            pl.BlockSpec(memory_space=plgpu.GMEM),
            pl.BlockSpec(memory_space=plgpu.SMEM),
        ),
        scratch_shapes=[plgpu.Barrier()],
    )
    def kernel(x_ref_gmem, idx_ref, o_ref, barrier_ref):
      idxs = plgpu.load(idx_ref, (), layout=plgpu.Layout.TMA_GATHER_INDICES)
      plgpu.copy_gmem_to_smem(x_ref_gmem.at[idxs, 64:], o_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)

    x = jnp.arange(math.prod(shape)).reshape(shape).astype(dtype)
    idx = jax.random.permutation(jax.random.key(1234), out_shape[0]).astype(jnp.uint32)
    np.testing.assert_array_equal(kernel(x, idx), x[idx, 64:])

  @parameterized.parameters(
      (plgpu.Layout.WGMMA, plgpu.Layout.WGMMA_TRANSPOSED),
      (plgpu.Layout.WGMMA_TRANSPOSED, plgpu.Layout.WGMMA),
  )
  def test_transposed_load_store(self, src_layout, dst_layout):
    def is_transposed(layout):
      return layout == plgpu.Layout.WGMMA_TRANSPOSED

    shape, dtype = (128, 128), jnp.float32

    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
    )
    def kernel(src_ref, dst_ref):
      if is_transposed(src_layout):
        src_ref = src_ref.T
      if is_transposed(dst_layout):
        dst_ref = dst_ref.T
      src = plgpu.load(src_ref, (), layout=src_layout, optimized=False)
      dst = plgpu.layout_cast(src, dst_layout)
      dst_ref[...] = dst

    x = jnp.arange(math.prod(shape), dtype=dtype).reshape(shape)
    np.testing.assert_array_equal(kernel(x), x.T)

  @parameterized.product(
      src_memory_space=[plgpu.SMEM, plgpu.GMEM],
      layout=[plgpu.Layout.WG_STRIDED((128,), vec_size=1), None,
      ]
  )
  def test_load_to_strided_layout_with_indexing(self, src_memory_space, layout):
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
        out_shape=jax.ShapeDtypeStruct([2, 32, 2, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier()],
    )
    x = jnp.arange(2 * 32 * 128, dtype=jnp.float32).reshape(2, 32, 128)
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
    # The default printf buffer on some smaller GPUs (e.g. Thor) only has space for
    # 4096 threads to printf (short) messages. Keep this shape below that.
    shape = (128, 32)
    size = math.prod(shape)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        in_specs=[
            plgpu.BlockSpec(
                transforms= self.default_transforms(dtype=jnp.float32),
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

  def test_print_layout(self):
    shape = (128,)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.bfloat16),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      x = plgpu.layout_cast(x_ref[...], plgpu.Layout.WGMMA_ROW)
      plgpu.print_layout("x: {}", x)

    x = jnp.arange(math.prod(shape), dtype=jnp.bfloat16).reshape(shape)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn("x: WGMMA_ROW\n", output())

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

  def test_swap_scalar_constant(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
    )
    def kernel(o_ref):
      o_ref[...] = jnp.array(42)

    np.testing.assert_array_equal(kernel(), jnp.array(42, jnp.int32))

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

  @parameterized.parameters(
      ((2, 3), ("a", "b"), (), ()),
      ((2, 3), ("a", "b"), (2,), ("x",)),
      ((2, 3, 4), ("a", "b", "c"), (), ()),
      ((2, 3, 4), ("a", "b", "c"), (2,), ("x",)),
      ((2, 3, 4), ("a", "b", "c"), (2, 3), ("x", "y")),
      ((2, 3, 4, 5), ("a", "b", "c", "d"), (), ()),
      ((2, 3, 4, 5), ("a", "b", "c", "d"), (2,), ("x",)),
  )
  def test_axis_indices_in_grid(self, grid, grid_names, cluster, cluster_names):
    @functools.partial(
        self.kernel,
        out_shape=[
            jax.ShapeDtypeStruct([*cluster, *grid, 128], jnp.int32),
            jax.ShapeDtypeStruct([*cluster, *grid, 128], jnp.int32)
        ],
        grid=grid,
        grid_names=grid_names,
        cluster=cluster,
        cluster_names=cluster_names,
    )
    def kernel(out1_ref, out2_ref):
      pallas_grid_idx = lax.axis_index(grid_names)
      cuda_grid_idx = _get_linearized_cuda_grid_index()

      out_indices = [lax.axis_index(ax) for ax in (*cluster_names, *grid_names)]
      out1_ref[*out_indices] = jnp.full((128,), pallas_grid_idx)
      out2_ref[*out_indices] = jnp.full((128,), cuda_grid_idx)
    out1, out2 = kernel()

    out_per_cta = jnp.arange(math.prod(grid), dtype=jnp.int32).reshape(grid)
    out1_ref = jnp.broadcast_to(out_per_cta[..., None], (*cluster, *grid, 128))
    np.testing.assert_array_equal(out1, out1_ref)

    padded_cluster = (1,) * (len(grid) - len(cluster)) + cluster
    scaled_grid = tuple(g * c for g, c in zip(grid, padded_cluster))
    original = jnp.arange(math.prod(scaled_grid), dtype=jnp.int32).reshape(
        scaled_grid
    )

    # Untile the scaled grid to get the per-cluster grid.
    interleaved_shape = tuple(val for pair in zip(grid, padded_cluster) for val in pair)
    perm = tuple(range(1, 2 * len(grid), 2)) + tuple(range(0, 2 * len(grid), 2))

    out2_ref = original.reshape(interleaved_shape).transpose(perm).squeeze()
    out2_ref = jnp.broadcast_to(out2_ref[..., None], out2_ref.shape + (128,))

    np.testing.assert_array_equal(out2, out2_ref)

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
    spec = plgpu.BlockSpec(
        (128, 64),
        lambda *i: i,
        transforms=self.default_transforms(dtype=jnp.float16),
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

  @parameterized.product(unroll=[1, 2, 4])
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
        o_ref[...] = o_ref[...]  # side-effect to prevent DCE

        # We deliberately do a cast here to trigger a layout mismatch.
        return plgpu.layout_cast(
            jnp.zeros(o_ref.shape, o_ref.dtype), plgpu.Layout.WGMMA_ROW
        )
      # Cast explicitly to cause the mismatch, otherwise layout inference will
      # succeed at constructing a working program.
      strided_input = plgpu.layout_cast(
          o_ref[...], plgpu.Layout.WG_STRIDED(shape=(128,), vec_size=1)
      )
      _ = jax.lax.while_loop(cond, body, strided_input)

    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup:
      with self.assertRaisesRegex(
          ValueError, "Failed to infer a possible set of layouts",
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
        transforms=self.default_transforms(dtype=jnp.uint16)
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
        transforms=self.default_transforms(dtype=jnp.float16)
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

  @parameterized.product(
      layouts=[
          (plgpu.Layout.WGMMA, plgpu.Layout.WGMMA_TRANSPOSED),
          (plgpu.Layout.TCGEN05, plgpu.Layout.TCGEN05_TRANSPOSED),
      ],
  )
  def test_transposed_layout(self, layouts):
    layout, transposed_layout = layouts
    dtype = jnp.dtype(jnp.float16)
    shape = (256, 192)
    transforms = self.default_transforms(dtype=dtype)
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape[::-1], dtype),
        out_specs=plgpu.BlockSpec(transforms=transforms),
    )
    def kernel(o_ref):
      iota = plgpu.broadcasted_iota(dtype, shape, 0, layout=layout)
      iota *= shape[1]
      iota += plgpu.broadcasted_iota(dtype, shape, 1, layout=layout)
      o_ref_t = plgpu.transpose_ref(o_ref, (1, 0))
      o_ref_t[...] = plgpu.layout_cast(iota, transposed_layout)

    x = jnp.arange(math.prod(shape), dtype=dtype).reshape(shape).T
    np.testing.assert_array_equal(kernel(), x)

  def test_profiler(self):
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
      with open(os.path.join(tmpdir, name)) as f:
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

  def test_smem_aliasing_works_basic(self):
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
                            transforms=(plgpu.TilingTransform((64,)),)),
                    ]
                ],
            )
        ],
    )
    def kernel(x_ref, o_ref128, aliased_ref):
      smem_ref256, [_, [smem_ref128]] = aliased_ref
      # Ensure that extraction via index works the same as unfolding.
      smem_ref128_2 = aliased_ref[1][1][0]
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
      [_, smem_refi8], [_, smem_refi4] = aliased_ref
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

  def test_loading_from_ref_union_works(self):
    self.skip_if_wg_semantics()  # Transform inference not implemented.
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        in_specs=[pl.BlockSpec((128,))] * 2,
        out_specs=pl.BlockSpec((128,), memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.RefUnion(plgpu.SMEM((128,), jnp.float32)),
                        plgpu.SMEM((128,), jnp.float32)],
    )
    def kernel(x_ref, y_ref, o_ref128, ref_union, o_smem):
      [aliased_ref] = ref_union
      aliased_ref[...] = x_ref[...]
      plgpu.commit_smem()
      load_ref = lambda r: plgpu.load(r, (), layout=plgpu.Layout.TCGEN05_ROW)
      # This is a regression test for b/423697560, where we used to fail to
      # transform the dtype correctly when processing an aliased ref.
      o_smem[...] = load_ref(aliased_ref) + load_ref(y_ref)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, o_ref128)

    x, y = (jnp.arange(128).astype(jnp.float32) for _ in range(2))
    np.testing.assert_array_equal(kernel(x, y), x + y)

  @parameterized.parameters(1, 2, 3)
  def test_nd_loop_with_carry(self, sm_steps):
    @functools.partial(
        self.kernel,
        out_shape=(
            jax.ShapeDtypeStruct((sm_steps, 132, 128), jnp.int32),
            jax.ShapeDtypeStruct((132,), jnp.int32)
        ),
        grid=(132,),
        grid_names=("sm",),
    )
    def kernel(o_ref, steps_ref):
      def body(loop_info, carry):
        idx = loop_info.index
        assert len(idx) == 3
        # We need to use `mode="clip"`, because the indices are not static.
        flat_idx = jnp.ravel_multi_index(idx, (sm_steps, 4, 33), mode="clip")
        sm_step = lax.div(
            flat_idx, lax.convert_element_type(lax.axis_size("sm"), jnp.int32)
        )
        o_ref[sm_step, lax.axis_index("sm")] = lax.broadcast(
            flat_idx, o_ref.shape[-1:]
        )
        return carry + 1

      steps_ref[lax.axis_index("sm")] = plgpu.nd_loop(
          (sm_steps, 4, 33), collective_axes="sm", init_carry=0
      )(body)

    result, steps = kernel()  # pylint: disable=unpacking-non-sequence
    for sm_step in range(sm_steps):
      np.testing.assert_array_equal(steps, jnp.full((132,), sm_steps))

      np.testing.assert_array_equal(
          result[sm_step],
          jnp.tile(
              (132 * sm_step + jnp.arange(132))[:, None],
              128,
          ),
      )

  @parameterized.product(
      sm_steps=(1, 2, 3),
      tiling=(None, 1, 2, 4),
  )
  def test_nd_loop(self, sm_steps: int, tiling: int | None):
    if tiling is not None:
      tiling = (sm_steps, tiling, 33)
    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((sm_steps, 132, 128), jnp.int32),
        grid=(132,),
        grid_names=("sm",),
    )
    def kernel(o_ref):
      @plgpu.nd_loop((sm_steps, 4, 33), tiling=tiling, collective_axes="sm")
      def _(loop_info):
        idx = loop_info.index
        assert len(idx) == 3
        # We need to use `mode="clip"`, because the indices are not static.
        grid = (sm_steps, 4, 33)
        if tiling:
          # Reconstruct the tiled grid and index.
          tiled_grid = tuple(g // t for g, t in zip(grid, tiling))
          grid = tiled_grid + tiling
          tile_idx = tuple(
              lax.div(idx, jnp.int32(t)) for idx, t in zip(idx, tiling))
          subtile_idx = tuple(
              lax.rem(idx, jnp.int32(t)) for idx, t in zip(idx, tiling))
          idx = tile_idx + subtile_idx
        flat_idx = jnp.ravel_multi_index(idx, grid, mode="clip")
        sm_step = lax.div(
            flat_idx, lax.convert_element_type(lax.axis_size("sm"), jnp.int32)
        )
        o_ref[sm_step, lax.axis_index("sm")] = lax.broadcast(
            flat_idx, o_ref.shape[-1:]
        )

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
      self.pallas_call(
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

  def test_lower_with_abstract_mesh(self):
    def kernel(y_ref, sem):
      plgpu.semaphore_signal_multicast(sem, collective_axes='x')
      # Wait for the multicast signal (each device gets signaled by all devices)
      pl.semaphore_wait(sem, 2)  # Wait for signals from both devices
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_jax = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )
    abstract_mesh = jax.sharding.AbstractMesh((2,), ('x',))
    jax.jit(jax.shard_map(
        kernel_jax, mesh=abstract_mesh, in_specs=(),
        out_specs=jax.P(), check_vma=False)).trace().lower(
            lowering_platforms=('gpu',))  # doesn't crash

  @parameterized.named_parameters(
    (
        f"_{''.join(map(str, collective_dims))}={collective_size}{'_' + ''.join(map(str, noncollective_dims)) if noncollective_dims else ''}",
        collective_dims,
        noncollective_dims,
        collective_size,
    )
    for collective_dims in itertools.chain.from_iterable(
        itertools.combinations("xyz", n) for n in range(1, 4)
    )
    for noncollective_dims in itertools.chain.from_iterable(
        itertools.combinations("xyz", n) for n in range(3)
    )
    for collective_size in (1, 2, 4)
    if all(d not in noncollective_dims for d in collective_dims)
  )
  def test_tma_load_multicast(self, collective_dims, noncollective_dims, collective_dim_size):
    """
      1. Broadcast a GMEM slice to SMEM across collective CTAs.
      2. Send a SMEM slice from each collective CTA to reconstruct the GMEM slice.
        It's not strictly necessary to use every collective CTA, but we use them
        to test that the cluster axes are used correctly.
    """

    self.skip_if_wg_semantics()  # User transforms are not supported.

    dtype = jnp.float16
    cluster = [1, 1, 1]
    for d in collective_dims:
      cluster["xyz".index(d)] = collective_dim_size
    for d in noncollective_dims:
      cluster["xyz".index(d)] = 2
    if math.prod(cluster) > jtu.get_cuda_nonportable_max_cluster_size():
      self.skipTest("Cluster is too big.")

    collective_size = math.prod(cluster["xyz".index(d)] for d in collective_dims)
    noncollective_size = math.prod(cluster) // collective_size

    swizzle = 128
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )
    shape = (noncollective_size, collective_size * 8, swizzle_elems)

    def body(x_gmem, out_gmem, smem, tma_barrier):
      # Compute the index in a subset of the cluster.
      def cluster_id(axes):
        idx, stride = 0, 1
        for d in sorted(axes):
          idx += lax.axis_index(d) * stride
          stride *= lax.axis_size(d)
        return idx

      noncollective_idx = cluster_id(noncollective_dims)
      collective_idx = cluster_id(collective_dims)

      plgpu.copy_gmem_to_smem(
            x_gmem.at[noncollective_idx],
            smem,
            tma_barrier,
            collective_axes=collective_dims)
      plgpu.barrier_wait(tma_barrier)

      plgpu.commit_smem()
      collective_slice = pl.ds(8 * collective_idx, 8)
      plgpu.copy_smem_to_gmem(
          smem.at[collective_slice],
          out_gmem.at[noncollective_idx, collective_slice, :],
      )
      plgpu.wait_smem_to_gmem(0)

    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    kernel = self.kernel(
      body,
      grid=cluster,
      grid_names=("grid_x", "grid_y", "grid_z"),
      cluster=cluster,
      cluster_names=("x", "y", "z"),
      out_shape=jax.ShapeDtypeStruct(shape, dtype),
      scratch_shapes=(
        plgpu.SMEM(shape[1:], dtype, transforms=transforms),
        plgpu.Barrier(),
      )
    )
    np.testing.assert_array_equal(kernel(x), x)

  @parameterized.product(
      layout=(
          plgpu.Layout.WGMMA,
          plgpu.Layout.TCGEN05,
          plgpu.Layout.TCGEN05_TMEM_NATIVE,
          plgpu.Layout.TCGEN05_M64_COLLECTIVE(128),
          plgpu.Layout.TILED(  # WGMMA, but defined as a custom tiling.
              plgpu.Tiling(((64, 8), (16, 8), (8, 8), (2,))),
              warp_dims=(-7,),
              lane_dims=(-3, -2),
              vector_dim=-1,
          ),
      ),
      op=(jnp.sum, jnp.max),
  )
  def test_reduce_with_layout(self, layout, op):
    self.skip_if_wg_semantics()
    axis = -1
    transforms = self.default_transforms(dtype=jnp.float32)
    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((128,), jnp.float32),
        scratch_shapes=[
            plgpu.SMEM((128, 128), jnp.float32, transforms=transforms),
            plgpu.SMEM((128,), jnp.float32),
            plgpu.Barrier(),
        ],
    )
    def kernel(x_ref, y_ref, smem_ref, smem_reduced_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      x_val = plgpu.load(smem_ref, (), layout=layout)
      smem_reduced_ref[...] = op(x_val, axis=axis)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_reduced_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(
        jax.random.key(0), shape=(128, 128), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_allclose(x_result, op(x, axis=axis), atol=1e-5)

  def _test_broadcast_in_dim_base(self, shape, layout, *, axis, hint):
    assert len(shape) == 2

    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros(shape, jnp.float32),
        scratch_shapes=[
            plgpu.SMEM((shape[1 - axis],), jnp.float32),
            plgpu.SMEM(shape, jnp.float32),
            plgpu.Barrier(),
        ],
    )
    def kernel(x_ref, y_ref, smem_ref, smem_out_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      reduced_layout = layout.reduce(axis)
      reduced = plgpu.load(smem_ref, (), layout=reduced_layout)
      broadcasted = lax.broadcast_in_dim(reduced, shape, [1 - axis])
      if hint:
        broadcasted = plgpu.layout_cast(broadcasted, layout)
      # Note that without the hint, the layout of broadcasted is not guaranteed
      # to be the same as the layout argument!
      smem_out_ref[...] = broadcasted
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_out_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(jax.random.key(0), shape=(128,), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    expected = jnp.expand_dims(x, axis=axis)
    expected = jnp.broadcast_to(expected, shape)
    np.testing.assert_array_equal(x_result, expected)

  @parameterized.product(
      layout=(
          plgpu.Layout.WGMMA,
          plgpu.Layout.TCGEN05,
          plgpu.Layout.TCGEN05_TMEM_NATIVE,
          plgpu.Layout.TCGEN05_M64_COLLECTIVE(128),
      ),
      axis=(0, 1),
      hint=(True, False),
  )
  def test_broadcast_in_dim(self, layout, axis, hint):
    self._test_broadcast_in_dim_base((128, 128), layout, axis=axis, hint=hint)

  # Regression test for a crash when using a small shape.
  def test_broadcast_in_dim_does_not_crash_on_small_shape(self):
    shape = (128, 4)
    self._test_broadcast_in_dim_base(
        shape, plgpu.Layout.TCGEN05_TMEM_NATIVE, axis=1, hint=False
    )

  def test_broadcast_in_dim_wg_strided_majormost_dim(self):
    self.skip_if_wg_semantics()
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((256, 128), jnp.float32),
    )
    def kernel(x_ref, y_ref):
      to_be_broadcasted = plgpu.load(
          x_ref, (), layout=plgpu.Layout.WG_STRIDED((128,), 1)
      )
      broadcasted = lax.broadcast_in_dim(to_be_broadcasted, (256, 128), (1,))
      y_ref[...] = broadcasted

    result = jax.random.uniform(jax.random.key(0), shape=(128,), dtype=jnp.float32)
    np.testing.assert_array_equal(kernel(result), jnp.broadcast_to(result[None,:], (256, 128)))

  def test_broadcast_in_dim_tcgen05_native_layout(self):
    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((128, 128), jnp.float32),
        scratch_shapes=[
            plgpu.SMEM((128,), jnp.float32),
            plgpu.SMEM((128, 128), jnp.float32),
            plgpu.Barrier(),
        ],
        num_threads=1,
        thread_name="x",
    )
    def kernel(x_ref, y_ref, smem_ref, smem_out_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      reduced = plgpu.load(smem_ref, (), layout=plgpu.Layout.TCGEN05_TMEM_NATIVE.reduce(1))
      broadcasted = lax.broadcast_in_dim(reduced, (128, 128), [0])
      broadcasted = plgpu.layout_cast(broadcasted, plgpu.Layout.TCGEN05_TMEM_NATIVE)
      smem_out_ref[...] = broadcasted
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_out_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(jax.random.key(0), shape=(128,), dtype=jnp.float32)
    np.testing.assert_array_equal(kernel(x), jnp.broadcast_to(x[:, None], (128, 128)))

  @parameterized.named_parameters((l.name.lower(), l) for l in plgpu.Layout)
  @jtu.skip_if_mosaic_gpu_exceeds_shared_memory(
      device_patterns=("RTX PRO 6000 Blackwell", "GB10$"))
  def test_copy_layout(self, layout):
    if layout in {
        plgpu.Layout.WG_SPLAT,
        plgpu.Layout.WGMMA_TRANSPOSED,
        plgpu.Layout.TCGEN05_TRANSPOSED,
        plgpu.Layout.TILED
    }:
      self.skipTest("Not the right layout for this test")

    # We don't infer optimized transfer-compatible transforms for load to
    # registers with TCGEN05_TMEM_NATIVE layout.
    # TODO(allanrenucci): Manually specify transforms when supported for WG
    # lowering semantic.
    optimized = (
        self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane
        or layout != plgpu.Layout.TCGEN05_TMEM_NATIVE
    ) and layout != plgpu.Layout.TCGEN05_M64_COLLECTIVE_NATIVE

    shape = (128, 128) if "tcgen05" in layout.name.lower() else (64, 128)
    dtype = jnp.float32
    swizzle = 128
    if layout in (plgpu.Layout.WGMMA_UPCAST_4X, plgpu.Layout.WGMMA_UPCAST_2X):
      dtype = jnp.float8_e5m2
      swizzle = 64
    transforms = self.default_transforms(dtype=dtype, swizzle=swizzle)

    if layout == plgpu.Layout.TCGEN05_M64_COLLECTIVE:
      layout = plgpu.Layout.TCGEN05_M64_COLLECTIVE(128)
    elif layout == plgpu.Layout.TCGEN05_M64_COLLECTIVE_NATIVE:
      layout = plgpu.Layout.TCGEN05_M64_COLLECTIVE_NATIVE(128)
      if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Lane:
        self.skipTest("Need to add support for optimized= for stores")
    elif layout == plgpu.Layout.WG_STRIDED:
      layout = plgpu.Layout.WG_STRIDED(shape, 2)
      transforms = ()
    elif layout == plgpu.Layout.SMEM_GMEM_COPY:
      layout = plgpu.Layout.SMEM_GMEM_COPY(shape, jnp.float32, swizzle=128)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        in_specs=[plgpu.BlockSpec(transforms=transforms)],
        out_specs=plgpu.BlockSpec(transforms=transforms),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plgpu.load(x_ref, (), layout=layout, optimized=optimized)

    x = jnp.arange(math.prod(shape), dtype=dtype).reshape(shape)
    np.testing.assert_array_equal(kernel(x), x)

  @parameterized.parameters(
      (((0, 0),), (128, 128), (128, 128)),
      (((0, 1),), (128, 128), (128, 128)),
      (((1, None),), (128, 128), (128,)),
      (((0, 0),), (128, 128), (128, 128)),
      (((0, 0), (0, 0)), (128, 128), (128, 128)),
  )
  def test_vmap_kernel(self, vmap_axes, x_shape, y_shape):
    rng0, rng1 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(rng0, x_shape, jnp.float32)
    y = jax.random.uniform(rng1, y_shape, jnp.float32)

    out_shape = list(x_shape)
    for x_axis, _ in vmap_axes:
      del out_shape[x_axis]
    out_shape = jax.ShapeDtypeStruct(out_shape, jnp.float32)

    @functools.partial(self.kernel, out_shape=out_shape)
    def f(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[...]

    f_ref = lambda x, y: x + y
    for in_axes in vmap_axes:
      f = jax.vmap(f, in_axes)
      f_ref = jax.vmap(f_ref, in_axes)

    np.testing.assert_array_equal(f(x, y), f_ref(x, y))

  def test_discharge_comms_effect(self):
    def body(out, sem):
      pl.semaphore_signal(sem, device_id=jnp.asarray(2, jnp.int32))

    f = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((), jnp.int32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )
    jax_core.check_jaxpr(jax.make_jaxpr(f)().jaxpr)

  @jtu.thread_unsafe_test()  # Modifies ``os.environ``.
  @jtu.skip_under_pytest("Test fails under pytest in CI")
  def test_line_info(self):
    self.skip_if_wg_semantics()

    with jtu.set_env(MOSAIC_GPU_DUMP_PTX="1"), jtu.capture_stdout() as output:
      @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
      )
      def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + x_ref[0]

      jax.block_until_ready(kernel(jnp.arange(256, dtype=jnp.float32)))

    ptx = output()
    self.assertIn(".file", ptx)
    self.assertIn(".loc", ptx)
    [path] = re.findall(r'.file\s+\d+\s+"(.+)"', ptx)
    self.assertEndsWith(__file__, path)

  def test_collective_arrival_count(self):
    def kernel(dst, collective_barrier):
      plgpu.barrier_arrive(collective_barrier)
      plgpu.barrier_arrive(collective_barrier)
      plgpu.barrier_arrive(collective_barrier)
      plgpu.barrier_arrive(collective_barrier)
      plgpu.barrier_wait(collective_barrier)
      dst[...] = jnp.ones_like(dst)
    y = self.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((128,), jnp.int32),
      scratch_shapes=[plgpu.ClusterBarrier(collective_axes=("x",), num_arrivals=4)],
      cluster=(2,),
      cluster_names=("x",)
    )()
    np.testing.assert_array_equal(y, np.ones((), dtype=np.int32))

  def test_replicated_layout(self):
    shape = (32,)
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
    )
    def kernel(src_ref, dst_ref):
      layout = plgpu.Layout.TILED(
        plgpu.Tiling(((32,), (1,))),
        warp_dims=(plgpu.Replicated(4),),
        lane_dims=(-2,),
        vector_dim=-1,
      )
      dst_ref[...] = plgpu.load(src_ref, (), layout=layout)
    src = jnp.arange(shape[0], dtype=jnp.float32)
    np.testing.assert_array_equal(kernel(src), src)


class PallasCallWarpPrimitiveSemanticsTest(PallasTest):
  def setUp(self):
    super().setUp()
    if self.LOWERING_SEMANTICS != plgpu.LoweringSemantics.Lane:
      self.skipTest("Test only works on Lane semantics")

  def test_axis_index(self):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(self.kernel,
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
            plgpu.async_prefetch(y_ref.at[1:2])
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

  def test_scalar_load(self):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(self.kernel,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def kernel(x_ref, y_ref):
      @pl.core_map(warp_mesh)
      def _():
        warp_id = lax.axis_index("warp")
        @pl.when(warp_id == 1)
        def _():
          y_ref[...] = x_ref[...]
    np.testing.assert_array_equal(kernel(4), 4)

  def test_non_scalar_load_raises(self):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(self.kernel,
                       out_shape=jax.ShapeDtypeStruct((2,), jnp.int32))
    def kernel(x_ref, y_ref):
      @pl.core_map(warp_mesh)
      def _():
        warp_id = lax.axis_index("warp")
        @pl.when(warp_id == 1)
        def _():
          y_ref[...] = x_ref[...]
    with self.assertRaisesRegex(ValueError, "Can only load scalars",):
      kernel(jnp.ones((2,), jnp.int32))

  @parameterized.parameters(
    lax.add, lax.sub, lax.mul, lax.div, lax.rem, lax.bitwise_and,
    lax.bitwise_or, lax.bitwise_xor, lax.max, lax.min,
    lax.gt, lax.lt, lax.ge, lax.le, lax.eq, lax.ne,
  )
  def test_scalar_binary_op(self, op):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(self.kernel,
                       out_shape=jax.ShapeDtypeStruct((), jnp.int32))
    def kernel(y_ref):
      @pl.core_map(warp_mesh)
      def _():
        warp_id = lax.axis_index("warp")
        @pl.when(warp_id == 1)
        def _():
          x = jnp.array(1234, dtype=jnp.int32)
          y = jnp.array(6543, dtype=jnp.int32)
          y_ref[...] = op(x, y).astype(jnp.int32)
    result = kernel()
    x = jnp.array(1234, dtype=jnp.int32)
    y = jnp.array(6543, dtype=jnp.int32)
    np.testing.assert_array_equal(result, op(x, y).astype(jnp.int32))

  def test_errors_when_closing_over_array(self):
    # We currently do not allow closing over arrays when mapping over
    # a mesh, since we would need to present a view of the array local
    # to each warp.
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(self.kernel,
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
        "Can only close over scalars and Refs .* with WarpMesh",
    ):
      kernel()

  @parameterized.parameters(True, False)
  def test_single_warp_loop(self, force_while):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(self.kernel,
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
            _fori_loop(force_while, 0, 10, loop_body, None)
        plgpu.wait_smem_to_gmem(0)
      pl.run_scoped(scope, plgpu.SMEM((10, 128), jnp.int32))
    result = kernel()
    expected = jnp.stack(
        [jnp.ones((128,), jnp.int32) * i for i in range(10)], axis=0)
    np.testing.assert_array_equal(result, expected)

  def test_debug_print(self):
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(
        self.kernel,
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

  @parameterized.parameters(False, True)
  def test_copy_gmem_to_smem_from_different_warps(self,
                                                  wait_smem_to_gmem_in_warp):
    # In this test, we issue a copy from from warp 0 and await it in warp 1.
    warp_mesh = plgpu.WarpMesh(axis_name="warp")
    @functools.partial(self.kernel,
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
            if wait_smem_to_gmem_in_warp:
              plgpu.wait_smem_to_gmem(0)
        if not wait_smem_to_gmem_in_warp:
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
        mgpu_primitives.async_copy_scales_to_tmem_p,
        mgpu_primitives.async_copy_sparse_metadata_to_tmem_p,
        mgpu_primitives.wait_load_tmem_p,
        mgpu_primitives.semaphore_signal_parallel_p,
        mgpu_primitives.semaphore_signal_multicast_p,
        mgpu_primitives.try_cluster_cancel_p,
        mgpu_primitives.query_cluster_cancel_p,
        mgpu_primitives.multimem_store_p,
        mgpu_primitives.multimem_load_reduce_p,
        pallas_core.core_map_p,
        pallas_primitives.semaphore_signal_p,
        pallas_primitives.semaphore_wait_p,
        pallas_primitives.semaphore_read_p,
        pallas_primitives.delay_p,
        checkify.check_p,
        lax.reshape_p,
    }

    self.assertSetEqual(actual_missing_primitives, expected_missing_primitives)


class PallasCallSm90ATest(PallasSm90ATest):

  @parameterized.parameters(False, True)
  def test_fori_loop_accumulator(self, force_while):
    transforms = self.default_transforms(dtype=jnp.float16)

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

    transforms = self.default_transforms(dtype=dtype)

    if lhs_transpose:
      lhs_spec = plgpu.BlockSpec(
          (tile_k, tile_m),
          lambda m, n, k: (k, m),
          delay_release=1,
          transforms=transforms,
      )
    else:
      lhs_spec = plgpu.BlockSpec(
          (tile_m, tile_k),
          lambda m, n, k: (m, k),
          delay_release=1,
          transforms=transforms,
      )
    if rhs_transpose:
      rhs_spec = plgpu.BlockSpec(
          (tile_n, tile_k),
          lambda m, n, k: (n, k),
          delay_release=1,
          transforms=transforms,
      )
    else:
      rhs_spec = plgpu.BlockSpec(
          (tile_k, tile_n),
          lambda m, n, k: (k, n),
          delay_release=1,
          transforms=transforms,
      )
    out_spec = plgpu.BlockSpec(
        (tile_m, tile_n),
        lambda m, n, k: (m, n),
        transforms=transforms,
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
        ),
    )(a, b)
    np.testing.assert_allclose(
        res,
        (a.T if lhs_transpose else a) @ (b.T if rhs_transpose else b),
        rtol=1e-3,
    )

  @parameterized.parameters(jnp.float16, jnp.float32)
  def test_wgmma(self, dtype):
    # TensorCores can only fuse transposes of 16-bit values, and RHS
    # is expected to be column major by default.
    rhs_transpose = jnp.dtype(dtype).itemsize != 2
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

    transforms = self.default_transforms(dtype=dtype)
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(
                (64, 128),
                lambda i, j: (i, j),
                transforms=transforms,
            ),
            plgpu.BlockSpec(
                b_shape,
                lambda *i: i,
                transforms=transforms,
            ),
        ],
        out_specs=plgpu.BlockSpec((64, 192), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_allclose(
        res, a @ (b.T if rhs_transpose else b), rtol=1e-3
    )

  @parameterized.parameters(jnp.int8, jnp.uint8)
  def test_wgmma_integer(self, dtype):
    m, k, n = 64, 128, 64

    is_signed = jnp.issubdtype(dtype, jnp.signedinteger)
    acc_type = jnp.int32

    def kernel(a_ref, b_ref, o_ref):

      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref, plgpu.transpose_ref(b_ref, (1, 0)))
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((m, n), acc_type))

    # use small values to avoid overflow, [0, 8) for u8 and (-8, 8) for s8
    random_int_input = lambda key, shape: jax.random.randint(
        key, minval=-8 * is_signed, maxval=8, shape=shape, dtype=dtype
    )

    a = random_int_input(jax.random.key(0), shape=(m, k))
    b = random_int_input(jax.random.key(1), shape=(n, k))

    transforms = self.default_transforms(dtype=dtype)
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(
                (m, k),
                lambda i, j: (i, j),
                transforms=transforms,
            ),
            plgpu.BlockSpec(
                (n, k),
                lambda *i: i,
                transforms=transforms,
            ),
        ],
        out_specs=plgpu.BlockSpec((m, n), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct((m, n), acc_type),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_array_equal(
        res, a.astype(acc_type) @ b.T.astype(acc_type)
    )

  def test_wgmma_sliced_acc_flip(self):
    self.skip_if_wg_semantics()
    dtype = jnp.float16

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=dtype)
    b = jax.random.uniform(key2, shape=(128, 256), dtype=dtype)

    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref.at[:, :128], a_ref, b_ref.at[:, 128:])
        plgpu.wgmma(acc_ref.at[:, 128:], a_ref, b_ref.at[:, :128])
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 256), jnp.float32))

    transforms = self.default_transforms(dtype=dtype)
    res = self.pallas_call(
        kernel,
        in_specs=[plgpu.BlockSpec(transforms=transforms)] * 2,
        out_shape=jax.ShapeDtypeStruct((64, 256), jnp.float32),
    )(a, b)

    def flip_halves(x):
      y = x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2)
      y = y[..., ::-1, :]
      return y.reshape(x.shape)

    np.testing.assert_allclose(res, a @ flip_halves(b), rtol=1e-3)

  def test_wgmma_registers(self):
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref[...], b_ref)
        return acc_ref[...]
      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 192), dtype=jnp.float16)

    transforms = self.default_transforms(dtype=jnp.float16)
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
        ],
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
    )(a, b)
    np.testing.assert_allclose(res, a @ b, rtol=1e-3)

  @parameterized.parameters(jnp.int8, jnp.float8_e4m3fn, jnp.float8_e5m2)
  def test_wgmma_registers_8bit(self, input_dtype):
    if input_dtype != jnp.int8:
      self.skip_if_wg_semantics()
    if jnp.issubdtype(input_dtype, jnp.integer):
      out_dtype = jnp.int32
    else:
      out_dtype = jnp.float32
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        a_regs = plgpu.load(a_ref, (), layout=plgpu.Layout.WGMMA_8BIT)
        plgpu.wgmma(acc_ref, a_regs, plgpu.transpose_ref(b_ref, (1, 0)))
        return acc_ref[...]
      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), out_dtype))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    m = 64
    k = 128
    n = 192
    if input_dtype == jnp.int8:
      a = jax.random.randint(key1, shape=(m, k), minval=-128, maxval=127, dtype=jnp.int8)
      b = jax.random.randint(key2, shape=(n, k), minval=-128, maxval=127, dtype=jnp.int8)
    else:
      assert jnp.issubdtype(input_dtype, jnp.floating)
      a = jax.random.uniform(key1, shape=(m, k), dtype=input_dtype)
      b = jax.random.uniform(key2, shape=(n, k), dtype=input_dtype)

    transforms = self.default_transforms(swizzle=64, dtype=input_dtype)
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
        ],
        out_shape=jax.ShapeDtypeStruct((64, 192), out_dtype),
    )(a, b)
    ref = a.astype(out_dtype) @ b.T.astype(out_dtype)
    if input_dtype == jnp.int8:
      np.testing.assert_array_equal(res, ref)
    else:
      np.testing.assert_allclose(res, ref)

  def test_wgmma_registers_init(self):
    def kernel(a_ref, b_ref, i_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref[...], b_ref)
      o_ref[...] = pl.run_state(scope)(plgpu.ACC.init(i_ref[...]))

    key1, key2, key3 = jax.random.split(jax.random.key(42), 3)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 192), dtype=jnp.float16)
    i = jax.random.uniform(key3, shape=(64, 192), dtype=jnp.float16) * 10

    transforms = self.default_transforms(dtype=jnp.float16)
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
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref.at[0], b_ref.at[0])
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(2, 64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(2, 128, 192), dtype=jnp.float16)

    transforms = self.default_transforms(dtype=jnp.float16)
    res = self.pallas_call(
        kernel,
        in_specs=[
            plgpu.BlockSpec(transforms=transforms),
            plgpu.BlockSpec(transforms=transforms),
        ],
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
    )(a, b)
    np.testing.assert_allclose(res, a[0] @ b[0], rtol=1e-3)

  def test_wgmma_sliced_acc_read(self):
    self.skip_if_wg_semantics()  # MLIR verifier error for `memref.subview`.

    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref, b_ref)
        return acc_ref[:, :64], acc_ref[:, 64:]

      o_ref[:, :64], o_ref[:, 64:] = pl.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 128), dtype=jnp.float16)
    transforms = self.default_transforms(dtype=jnp.float16)
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
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([2, m], jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=src_memory_space)],
        out_specs=plgpu.BlockSpec(memory_space=plgpu.SMEM),
    )
    def kernel(x_ref, o_ref):
      for i in range(2):
        x = plgpu.load(
            x_ref, (i,), layout=layout, optimized=src_memory_space == plgpu.SMEM
        )
        o_ref[i, ...] = x

    x = jnp.arange(2 * m, dtype=jnp.float32).reshape(2, m)
    np.testing.assert_array_equal(kernel(x), x)

  @parameterized.product(
      src_memory_space=[plgpu.SMEM],
      layout=[plgpu.Layout.WGMMA_ROW, plgpu.Layout.WGMMA_COL],
  )
  def test_load_row_input_to_wgmma_with_transforms(self, src_memory_space, layout):
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
                transforms=self.default_transforms(dtype=jnp.float16),
            ),
        ),
        out_specs=plgpu.BlockSpec(memory_space=plgpu.SMEM),
    )

    out_ref = (
        jnp.broadcast_to(jnp.expand_dims(a, axis=expand_dim), (m, k)) @ b
    )
    np.testing.assert_allclose(f(a, b), out_ref, rtol=1e-3)

  def test_load_store_wgmma_transposed(self):
    if self.LOWERING_SEMANTICS == plgpu.LoweringSemantics.Warpgroup:
      self.skipTest("Doesn't work in WG semantics")
    transforms = (plgpu.TilingTransform((8, 16)),
                  plgpu.SwizzleTransform(64))
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([8, 64], jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=plgpu.GMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[
            plgpu.SMEM((8, 64), jnp.float32, transforms=transforms),
            plgpu.Barrier(),
        ],
    )
    def kernel(x_gmem, o_ref, x_smem, barrier):
      plgpu.copy_gmem_to_smem(x_gmem, x_smem, barrier)
      plgpu.barrier_wait(barrier)
      x = plgpu.load(x_smem.T, (), layout=plgpu.Layout.WGMMA_TRANSPOSED)
      x_smem.T[...] = x + 1
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(x_smem, o_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(jax.random.key(42), shape=(8, 64), dtype=jnp.float32)
    result = kernel(x)
    np.testing.assert_array_equal(result, x + 1)


class PallasCallSm90AWGTest(
    PallasCallSm90ATest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class PallasCallSm100ATest(PallasSm100ATest):

  def test_print_layout_tmem(self):
    shape = (128, 256)

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.bfloat16),
        scratch_shapes=[plgpu.TMEM(shape, jnp.bfloat16, packed=True)],
    )
    def kernel(o_ref, tmem_ref):
      del o_ref
      # Slicing TMEM to make sure we handle transforms correctly.
      plgpu.print_layout("tmem: {}", tmem_ref.at[:, :128])

    with self.capture_stdout() as output:
      jax.block_until_ready(kernel())

    self.assertIn("tmem: TMEM_DEFAULT(packing=2)\n", output())

  def test_mixed_tmem_allocations_raise(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), jnp.float32),
        scratch_shapes=[
            plgpu.TMEM((128, 128), jnp.float32, collective=True),
            plgpu.TMEM((128, 128), jnp.float32, collective=False),
        ],
    )
    def kernel(out_ref, tmem_ref0, tmem_ref1):
      del out_ref, tmem_ref0, tmem_ref1

    with self.assertRaisesRegex(
        ValueError,
        "Can't mix collective and non-collective TMEM allocations within the"
        " same kernel.",
    ):
      kernel()

  def test_transposed_tmem_ref_raises(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct([], jnp.float32),
        scratch_shapes=[plgpu.TMEM((128, 128), jnp.float32)],
    )
    def kernel(out, tmem_ref):
      del out
      plgpu.transpose_ref(tmem_ref, (1, 0))

    with self.assertRaisesRegex(ValueError, "Can't transpose a TMEM reference"):
      kernel()

  @parameterized.parameters((False,), (True,))
  def test_tmem(self, collective):
    transforms = self.default_transforms(dtype=jnp.float32)
    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((128, 128), jnp.float32),
        scratch_shapes=[
            plgpu.TMEM((128, 128), jnp.float32, collective=collective),
            plgpu.TMEM((128, 128), jnp.float32, collective=collective),
            plgpu.SMEM((128, 128), jnp.float32, transforms=transforms),
            plgpu.Barrier(),
        ],
        num_threads=1,
        thread_name="x",
        cluster=(2,) if collective else (),
        cluster_names=("x",) if collective else (),
    )
    def kernel(x_ref, y_ref, tmem_ref, tmem_ref2, smem_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      # Exercise TMEM by roundtripping SMEM -> TMEM -> TMEM -> SMEM.
      x_val = plgpu.load(smem_ref, (), layout=plgpu.Layout.TCGEN05)
      plgpu.async_store_tmem(tmem_ref, x_val + 1)
      plgpu.commit_tmem()
      #  We don't await the load, because we never overwrite tmem_ref
      tmem_read = plgpu.async_load_tmem(tmem_ref)
      plgpu.async_store_tmem(tmem_ref2, tmem_read)
      plgpu.commit_tmem()
      #  We don't await the load, because we never overwrite tmem_ref2
      smem_ref[...] = plgpu.async_load_tmem(tmem_ref2)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(
        jax.random.key(0), shape=(128, 128), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_array_equal(x_result, x + 1)

  def test_tmem_allocation_estimation(self):
    """Make sure that we don't overestimate the TMEM allocation.

    All of the refs below are packed and should fit into TMEM at once.
    """
    transforms = self.default_transforms(dtype=jnp.bfloat16)
    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((128, 256), jnp.bfloat16),
        scratch_shapes=[
            plgpu.TMEM((128, 256), jnp.bfloat16, packed=True),
            plgpu.TMEM((128, 256), jnp.bfloat16, packed=True),
            plgpu.TMEM((128, 256), jnp.bfloat16, packed=True),
            plgpu.SMEM((128, 256), jnp.bfloat16, transforms=transforms),
            plgpu.Barrier(),
        ],
        num_threads=1,
        thread_name="x",
    )
    def kernel(x_ref, y_ref, tmem_ref1, tmem_ref2, tmem_ref3, smem_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      x_val = plgpu.load(smem_ref, (), layout=plgpu.Layout.TCGEN05)
      plgpu.async_store_tmem(tmem_ref1, x_val + 1)
      plgpu.commit_tmem()
      x_val = plgpu.async_load_tmem(tmem_ref1)
      plgpu.async_store_tmem(tmem_ref2, x_val + 1)
      plgpu.commit_tmem()
      x_val = plgpu.async_load_tmem(tmem_ref2)
      plgpu.async_store_tmem(tmem_ref3, x_val + 1)
      plgpu.commit_tmem()
      smem_ref[...] = plgpu.async_load_tmem(tmem_ref3)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(jax.random.key(0), shape=(128, 256), dtype=jnp.bfloat16)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_array_equal(x_result, x + 3)

  def test_tmem_ref_aliasing(self):
    self.skip_if_wg_semantics()
    transforms = self.default_transforms(dtype=jnp.float32)
    @functools.partial(
        self.kernel,
        out_shape=jnp.zeros((128, 128), jnp.float32),
        scratch_shapes=[
            plgpu.RefUnion(
              [plgpu.TMEM((128, 32), jnp.float32),
               plgpu.TMEM((128, 32), jnp.float32)],
              plgpu.TMEM((128, 64), jnp.float32),
            ),
            plgpu.SMEM((128, 128), jnp.float32, transforms=transforms),
            plgpu.Barrier(),
        ],
        num_threads=1,
        thread_name="x",
    )
    def kernel(x_ref, y_ref, aliased_ref, smem_ref, barrier_ref):
      [tmem_128x32a, tmem_128x32b], tmem_128x64 = aliased_ref
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      # Test tmem_128x32 a and b
      x_val = plgpu.load(smem_ref.at[:, 0:32], (), layout=plgpu.Layout.TCGEN05)
      plgpu.async_store_tmem(tmem_128x32a, x_val + 1)
      plgpu.commit_tmem()
      smem_ref[:, 0:32] = plgpu.async_load_tmem(tmem_128x32a)
      plgpu.wait_load_tmem()  # Make sure the load is done before we write to TMEM again.

      x_val = plgpu.load(smem_ref.at[:, 32:64], (), layout=plgpu.Layout.TCGEN05)
      plgpu.async_store_tmem(tmem_128x32b, x_val + 1)
      plgpu.commit_tmem()
      smem_ref[:, 32:64] = plgpu.async_load_tmem(tmem_128x32b)
      plgpu.wait_load_tmem()  # Make sure the load is done before we write to TMEM again.

      # Test tmem_128x64
      x_val = plgpu.load(smem_ref.at[:, 64:128], (), layout=plgpu.Layout.TCGEN05)
      plgpu.async_store_tmem(tmem_128x64, x_val + 1)
      plgpu.commit_tmem()
      smem_ref[:, 64:128] = plgpu.async_load_tmem(tmem_128x64)

      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)
    x = jax.random.uniform(
        jax.random.key(0), shape=(128, 128), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_array_equal(x_result, x + 1)

  @parameterized.parameters(
      plgpu.Layout.TCGEN05, plgpu.Layout.TCGEN05_TMEM_NATIVE
  )
  def test_tmem_load_layout(self, layout):
    transforms = self.default_transforms(dtype=jnp.float32)
    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32),
        scratch_shapes=[
            plgpu.TMEM((128, 128), jnp.float32),
            plgpu.SMEM((128, 128), jnp.float32, transforms=transforms),
            plgpu.Barrier(),
        ],
    )
    def kernel(x_ref, y_ref, tmem_ref, smem_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      optimized = layout != plgpu.Layout.TCGEN05_TMEM_NATIVE
      x_val = plgpu.load(smem_ref, (), layout=layout, optimized=optimized)
      plgpu.async_store_tmem(tmem_ref, x_val + 1)
      plgpu.commit_tmem()
      # We don't wait for the load to complete, because we never overwrite
      # tmem_ref.
      smem_ref[...] = plgpu.async_load_tmem(tmem_ref, layout=layout)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(
        jax.random.key(0), shape=(128, 128), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_array_equal(x_result, x + 1)

  @parameterized.parameters(
      plgpu.Layout.TCGEN05_M64_COLLECTIVE(160),
      plgpu.Layout.TCGEN05_M64_COLLECTIVE_NATIVE(160)
  )
  def test_tmem_store_load_collective(self, layout):
    self.skip_if_wg_semantics()  # Failed to infer a possible set of layouts.
    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((64, 160), jnp.float32),
        cluster=(2,),
        cluster_names=("cluster",),
        scratch_shapes=[
            plgpu.TMEM(
                (64, 160), jnp.float32, collective=True,
                layout=plgpu.TMEMLayout.M64_COLLECTIVE_LAYOUT(160),
            ),
        ],
    )
    def kernel(x_ref, y_ref, tmem_ref):
      x_val = plgpu.load(x_ref, (), layout=layout, optimized=False)
      plgpu.async_store_tmem(tmem_ref, x_val + 1)
      plgpu.commit_tmem()
      # We don't wait for the load to complete, because we never overwrite
      # tmem_ref.
      y_ref[...] = plgpu.async_load_tmem(tmem_ref, layout=layout)

    x = jax.random.uniform(
        jax.random.key(0), shape=(64, 160), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_array_equal(x_result, x + 1)

  def test_tmem_column_slicing(self):
    transforms = self.default_transforms(dtype=jnp.float32)
    @functools.partial(
        self.kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32),
        scratch_shapes=[
            plgpu.TMEM((128, 256), jnp.float32),
            plgpu.SMEM((128, 128), jnp.float32, transforms=transforms),
            plgpu.Barrier(),
        ],
        num_threads=1,
        thread_name="x",
    )
    def kernel(x_ref, y_ref, tmem_ref, smem_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(x_ref, smem_ref, barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      x_val = plgpu.load(smem_ref, (), layout=plgpu.Layout.TCGEN05)
      tmem_slice = tmem_ref.at[:, 8:208].at[:, 0:128]
      plgpu.async_store_tmem(tmem_slice, x_val + 1)
      plgpu.commit_tmem()
      smem_ref[...] = plgpu.async_load_tmem(tmem_ref.at[:, 8:136])
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(smem_ref, y_ref)
      plgpu.wait_smem_to_gmem(0)

    x = jax.random.uniform(
        jax.random.key(0), shape=(128, 128), dtype=jnp.float32)
    x_result = jax.block_until_ready(kernel(x))
    np.testing.assert_array_equal(x_result, (x + 1)[:, 0:128])

  @parameterized.product(
      m=[64, 128],
      n=[64, 128, 256],
      swizzle=[64, 32],
      dtype=[jnp.int8, jnp.uint8],
      lhs_tmem=[False, True],
  )
  def test_integer_matmul(self, m, n, swizzle, dtype, lhs_tmem):
    if n * jnp.dtype(dtype).itemsize <= swizzle:
      self.skipTest("swizzle too big")
    if lhs_tmem and m == 64:
      self.skipTest("m=64 not supported for LHS in TMEM")
    if lhs_tmem:
      self.skip_if_wg_semantics()  # Layout inference fails to find a solution.
    k = 128
    is_signed = jnp.issubdtype(dtype, jnp.signedinteger)
    o_dtype = jnp.int32

    in_transforms = self.default_transforms(dtype=dtype, swizzle=swizzle)
    out_transforms = self.default_transforms(dtype=o_dtype)

    def kernel(
        a_smem, b_smem, out_ref, acc_tmem, scratch_smem, barrier_ref, a_tmem_ref
    ):
      if lhs_tmem:
        lhs_ref = a_tmem_ref
        layout = plgpu.Layout.TCGEN05_TMEM_NATIVE(4)
        plgpu.async_store_tmem(lhs_ref, plgpu.load(a_smem, (), layout=layout, optimized=False))
        plgpu.commit_tmem()
      else:
        lhs_ref = a_smem

      plgpu.tcgen05_mma(
          acc_tmem, lhs_ref, b_smem, barrier_ref, accumulate=False
      )
      plgpu.barrier_wait(barrier_ref)
      scratch_smem[...] = plgpu.async_load_tmem(acc_tmem)
      plgpu.commit_smem()

      plgpu.copy_smem_to_gmem(scratch_smem, out_ref)
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.TMEM((m, n), o_dtype, packed=False),
        plgpu.SMEM((m, n), o_dtype, transforms=out_transforms),
        plgpu.Barrier(orders_tensor_core=True),
    ]
    if lhs_tmem:
      scratch_shapes.append(plgpu.TMEM((m, k), dtype, packed=True))
    else:
      scratch_shapes.append(None)

    f = self.pallas_call(
        kernel,
        in_specs=(
            plgpu.BlockSpec(transforms=in_transforms, memory_space=plgpu.SMEM),
            plgpu.BlockSpec(transforms=in_transforms, memory_space=plgpu.SMEM),
        ),
        out_specs=plgpu.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((m, n), o_dtype),
        scratch_shapes=scratch_shapes,
    )
    # use small values to avoid overflow, [0, 8) for u8 and (-8, 8) for s8
    random_int_input = lambda key, shape: jax.random.randint(
        key, minval=-8 * is_signed, maxval=8, shape=shape, dtype=dtype
    )

    x = random_int_input(jax.random.key(0), shape=(m, k))
    y = random_int_input(jax.random.key(1), shape=(k, n))

    result = f(x, y)
    expected = x.astype(o_dtype) @ y.astype(o_dtype)
    np.testing.assert_array_equal(result, expected)

  @parameterized.product(m=[64, 128],
                         n=[64, 128, 256],
                         swizzle=[128, 64, 32],
                         dtype=[jnp.float16, jnp.bfloat16],
                         lhs_tmem=[False, True],
                         transpose_rhs=[False, True],
                         transpose_lhs=[False, True])
  def test_simple_matmul(
      self, m, n, swizzle, dtype, lhs_tmem, transpose_lhs, transpose_rhs
  ):
    if transpose_lhs and lhs_tmem:
      self.skipTest("TMEM transpose not supported")
    if n * jnp.dtype(dtype).itemsize <= swizzle:
      self.skipTest("swizzle too big")
    if lhs_tmem and m == 64:
      self.skipTest("m=64 not supported for LHS in TMEM")
    k = 128
    # Test a matmul with a single block.
    transforms = self.default_transforms(dtype=dtype, swizzle=swizzle)

    def kernel(a_smem, b_smem, out_ref, acc_tmem, scratch_smem, barrier_ref,
               a_tmem_ref):
      if transpose_lhs:
        a_smem = plgpu.transpose_ref(a_smem, (1, 0))
      if transpose_rhs:
        b_smem = plgpu.transpose_ref(b_smem, (1, 0))
      if lhs_tmem:
        lhs_ref = a_tmem_ref
        layout = plgpu.Layout.TCGEN05 if m == 128 else plgpu.Layout.WGMMA
        plgpu.async_store_tmem(lhs_ref, plgpu.load(a_smem, (), layout=layout))
        plgpu.commit_tmem()
      else:
        lhs_ref = a_smem
      plgpu.tcgen05_mma(acc_tmem,
                        lhs_ref,
                        b_smem,
                        barrier_ref,
                        accumulate=False)
      plgpu.barrier_wait(barrier_ref)
      # We don't await the load because acc_tmem is never modified again.
      scratch_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_smem, out_ref)
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.TMEM((m, n), jnp.float32, packed=False),
        plgpu.SMEM((m, n), dtype, transforms=transforms),
        plgpu.Barrier(orders_tensor_core=True),
    ]
    if lhs_tmem:
      scratch_shapes.append(plgpu.TMEM((m, k), dtype, packed=True))
    else:
      scratch_shapes.append(None)

    f = self.pallas_call(
        kernel,
        in_specs=(
            plgpu.BlockSpec(transforms=transforms, memory_space=plgpu.SMEM),
            plgpu.BlockSpec(transforms=transforms, memory_space=plgpu.SMEM),
        ),
        out_specs=plgpu.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        scratch_shapes=scratch_shapes,
    )
    lhs_shape = (k, m) if transpose_lhs else (m, k)
    rhs_shape = (n, k) if transpose_rhs else (k, n)
    x = jax.random.uniform(jax.random.key(0), shape=lhs_shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=rhs_shape, dtype=dtype)
    result = f(x, y)
    if transpose_lhs:
      x = jnp.transpose(x, (1, 0))
    if transpose_rhs:
      y = jnp.transpose(y, (1, 0))
    expected = x @ y
    np.testing.assert_allclose(result, expected, rtol=1e-3)

  def test_matmul_alignment(self):
    m = k = n = 128
    dtype = jnp.float16
    transforms = self.default_transforms(dtype=dtype)

    def kernel(a_smem, b_smem, out_ref, _, acc_tmem, barrier_ref):
      plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, barrier_ref, accumulate=False)
      plgpu.barrier_wait(barrier_ref)
      # We don't await the load because acc_tmem is never modified again.
      out_ref[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)

    spec = plgpu.BlockSpec(transforms=transforms, memory_space=plgpu.SMEM)
    f = self.pallas_call(
        kernel,
        in_specs=(spec, spec),
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct((m, n), dtype),
        # Add a one column space to test if we align the accumulator.
        scratch_shapes=(
            plgpu.TMEM((128, 1), jnp.float32),
            plgpu.TMEM((m, n), jnp.float32),
            plgpu.Barrier(orders_tensor_core=True),
        ),
    )
    lhs_shape = (m, k)
    rhs_shape = (k, n)
    x = jax.random.uniform(jax.random.key(0), shape=lhs_shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=rhs_shape, dtype=dtype)
    result = f(x, y)
    expected = x @ y
    np.testing.assert_allclose(result, expected, rtol=1e-3)

  @parameterized.product(
      m=[128],
      n=[128, 256],
      dtype=[jnp.float8_e5m2, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
  )
  def test_simple_scaled_matmul(self, m, n, dtype):
    self.skip_if_wg_semantics()
    # TODO(apaszke): Add support for single-buffering in pallas_call.
    causes_oom = jnp.finfo(dtype).bits == 8 and n == 256
    k = 128 if causes_oom else 256
    swizzle = 128
    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)
    out_transforms = self.default_transforms(dtype=jnp.float32)

    def kernel(a_smem, b_smem, a_scale_smem, b_scale_smem, out_ref,
               barrier_ref, acc_tmem, a_scale_tmem, b_scale_tmem):
      plgpu.async_copy_scales_to_tmem(a_scale_smem, a_scale_tmem)
      plgpu.async_copy_scales_to_tmem(b_scale_smem, b_scale_tmem)
      # We don't have to await the copy because it's only used by the MMA.
      plgpu.tcgen05_mma(acc_tmem,
                        a_smem,
                        plgpu.transpose_ref(b_smem, (1, 0)),
                        a_scale=a_scale_tmem,
                        b_scale=b_scale_tmem,
                        accumulate=False)
      plgpu.tcgen05_commit_arrive(barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      # We don't await the load because acc_tmem is never modified again.
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    scratch_shapes = [
        plgpu.Barrier(orders_tensor_core=True),
        plgpu.TMEM((m, n), jnp.float32),
        plgpu.TMEM((m, k // 32), jnp.float8_e8m0fnu, layout=plgpu.TMEMLayout.SCALES_LAYOUT),
        plgpu.TMEM((n, k // 32), jnp.float8_e8m0fnu, layout=plgpu.TMEMLayout.SCALES_LAYOUT),
    ]

    f = self.pallas_call(
        kernel,
        in_specs=(
            plgpu.BlockSpec(memory_space=plgpu.SMEM, transforms=transforms),
            plgpu.BlockSpec(memory_space=plgpu.SMEM, transforms=transforms),
            plgpu.BlockSpec(memory_space=plgpu.SMEM),
            plgpu.BlockSpec(memory_space=plgpu.SMEM),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        out_specs=plgpu.BlockSpec(transforms=out_transforms),
        scratch_shapes=scratch_shapes,
    )
    x = jax.random.uniform(jax.random.key(1), shape=(m, k), dtype=jnp.float32).astype(dtype)
    y = jax.random.uniform(jax.random.key(2), shape=(n, k), dtype=jnp.float32).astype(dtype)
    ksx, ksy = jax.random.split(jax.random.key(1234), 2)
    x_scale = jax.lax.bitcast_convert_type(
        jax.random.randint(ksx, (m, k // 32), 122, 132, dtype=jnp.uint8),
        jnp.float8_e8m0fnu
    )
    y_scale = jax.lax.bitcast_convert_type(
        jax.random.randint(ksy, (n, k // 32), 122, 132, dtype=jnp.uint8),
        jnp.float8_e8m0fnu
    )
    def format_scales(scales):
      mn, k = scales.shape
      assert mn % 128 == 0 and k % 4 == 0
      return (
          scales.reshape(mn // 128, 4, 32, k // 4, 4)
          .transpose(0, 3, 2, 1, 4)
          .reshape(mn // 128, k // 4, 32, 16)
      )
    result = f(x, y, format_scales(x_scale), format_scales(y_scale))
    x_logical_scale = jnp.repeat(x_scale, 32, axis=1).astype(jnp.float32)
    y_logical_scale = jnp.repeat(y_scale, 32, axis=1).astype(jnp.float32)
    expected = jnp.dot(
        x.astype(jnp.float32) * x_logical_scale,
        (y.astype(jnp.float32) * y_logical_scale).T,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-3)

  @parameterized.product(
      m=[256],
      n=[256],
      scale_jax_dtype=[jnp.float8_e8m0fnu, jnp.float8_e4m3fn],
  )
  def test_collective_scaled_matmul(self, m, n, scale_jax_dtype):
    self.skip_if_wg_semantics()

    in_jax_dtype = jnp.float4_e2m1fn
    out_jax_dtype = jnp.float32
    scale_block = 32 if scale_jax_dtype == jnp.float8_e8m0fnu else 16
    swizzle = 128
    k_steps = 2
    swizzle_elems = 8 * swizzle // dtypes.itemsize_bits(in_jax_dtype)
    k = swizzle_elems * k_steps
    tiling = (8, swizzle_elems)
    transforms = (
        plgpu.TilingTransform(tiling), plgpu.SwizzleTransform(swizzle)
    )
    out_transforms = self.default_transforms(dtype=out_jax_dtype)

    m_block = m // 2
    n_block = n // 2

    def kernel(lhs_gmem, rhs_gmem, lhs_scales_gmem, rhs_scales_gmem, out_gmem,
               lhs_smem, rhs_smem, lhs_scales_smem, rhs_scales_smem, out_smem,
               tma_barrier, mma_barrier,
               acc_tmem, lhs_scales_tmem, rhs_scales_tmem):
      plgpu.copy_gmem_to_smem(lhs_gmem, lhs_smem, tma_barrier,
                              collective_axes="x", partitioned_axis=0)
      plgpu.copy_gmem_to_smem(rhs_gmem, rhs_smem, tma_barrier,
                              collective_axes="x", partitioned_axis=0)
      plgpu.copy_gmem_to_smem(lhs_scales_gmem, lhs_scales_smem, tma_barrier,
                              collective_axes="x", partitioned_axis=0)
      # RHS scales are replicated (multicast)
      plgpu.copy_gmem_to_smem(rhs_scales_gmem, rhs_scales_smem, tma_barrier,
                              collective_axes="x", partitioned_axis=None)
      cluster_idx = lax.axis_index("x")

      @pl.when(cluster_idx == 0)
      def _leader_block():
        plgpu.barrier_wait(tma_barrier)
        plgpu.async_copy_scales_to_tmem(lhs_scales_smem, lhs_scales_tmem, collective_axis="x")
        plgpu.async_copy_scales_to_tmem(rhs_scales_smem, rhs_scales_tmem, collective_axis="x")
        plgpu.tcgen05_mma(
            acc_tmem,
            lhs_smem,
            plgpu.transpose_ref(rhs_smem, (1, 0)),
            mma_barrier,
            a_scale=lhs_scales_tmem,
            b_scale=rhs_scales_tmem,
            accumulate=False,
            collective_axis="x"
        )
      plgpu.barrier_wait(mma_barrier)

      out_smem[...] = plgpu.async_load_tmem(acc_tmem)
      plgpu.commit_smem()
      slice_out = pl.ds(cluster_idx * m_block, m_block)
      plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[slice_out, :])
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.SMEM((m_block, k), in_jax_dtype, transforms=transforms),
        plgpu.SMEM((n_block, k), in_jax_dtype, transforms=transforms),
        plgpu.SMEM((m_block // 128, k // (scale_block * 4), 32, 16), scale_jax_dtype),
        plgpu.SMEM((n // 128, k // (scale_block * 4), 32, 16), scale_jax_dtype),
        plgpu.SMEM((m_block, n), out_jax_dtype, transforms=out_transforms),
        plgpu.Barrier(num_arrivals=4),
        plgpu.Barrier(orders_tensor_core=True),
        plgpu.TMEM((m_block, n), out_jax_dtype, collective=True),
        plgpu.TMEM((m_block, k // scale_block), scale_jax_dtype,
                   layout=plgpu.TMEMLayout.SCALES_LAYOUT, collective=True),
        plgpu.TMEM((n, k // scale_block), scale_jax_dtype,
                   layout=plgpu.TMEMLayout.SCALES_LAYOUT, collective=True),
    ]

    f = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), out_jax_dtype),
        grid=(1,),
        grid_names=("_",),
        cluster=(2,),
        cluster_names=("x",),
        scratch_shapes=scratch_shapes,
    )

    x = jax.random.uniform(jax.random.key(1), shape=(m, k), dtype=jnp.float32).astype(in_jax_dtype)
    y = jax.random.uniform(jax.random.key(2), shape=(n, k), dtype=jnp.float32).astype(in_jax_dtype)

    ka, kb = jax.random.split(jax.random.key(1234), 2)
    if scale_jax_dtype == jnp.float8_e8m0fnu:
      x_scale = jax.lax.bitcast_convert_type(
          jax.random.randint(ka, (m, k // scale_block), 122, 132, dtype=jnp.uint8),
          scale_jax_dtype
      )
      y_scale = jax.lax.bitcast_convert_type(
          jax.random.randint(kb, (n, k // scale_block), 122, 132, dtype=jnp.uint8),
          scale_jax_dtype
      )
    else:
      x_scale = jnp.abs(
          jax.random.normal(ka, (m, k // scale_block), dtype=jnp.float32).astype(scale_jax_dtype)
      )
      y_scale = jnp.abs(
          jax.random.normal(kb, (n, k // scale_block), dtype=jnp.float32).astype(scale_jax_dtype)
      )

    def format_scales(scales):
      mn, k = scales.shape
      assert mn % 128 == 0 and k % 4 == 0
      return (
          scales.reshape(mn // 128, 4, 32, k // 4, 4)
          .transpose(0, 3, 2, 1, 4)
          .reshape(mn // 128, k // 4, 32, 16)
      )

    result = f(x, y, format_scales(x_scale), format_scales(y_scale))

    x_logical_scale = jnp.repeat(x_scale, scale_block, axis=1).astype(jnp.float32)
    y_logical_scale = jnp.repeat(y_scale, scale_block, axis=1).astype(jnp.float32)
    expected = jnp.dot(
        x.astype(jnp.float32) * x_logical_scale,
        (y.astype(jnp.float32) * y_logical_scale).T,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-3)

  @parameterized.product(
      m=[128],
      n=[128, 256],
      dtype=[jnp.float16],
  )
  def test_simple_sparse_matmul(self, m, n, dtype):
    self.skip_if_wg_semantics()
    k = 128
    swizzle = 128 // jnp.dtype(dtype).itemsize
    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)
    out_transforms = self.default_transforms(dtype=jnp.float32)

    def kernel(a_smem, b_smem, a_sparse_smem, out_ref,
               barrier_ref, acc_tmem, a_sparse_tmem):
      plgpu.async_copy_sparse_metadata_to_tmem(a_sparse_smem, a_sparse_tmem)
      # We don't have to await the copy because it's only used by the MMA.
      plgpu.tcgen05_mma(acc_tmem,
                        a_smem,
                        plgpu.transpose_ref(b_smem, (1, 0)),
                        a_sparse_metadata=a_sparse_tmem,
                        accumulate=False)
      plgpu.tcgen05_commit_arrive(barrier_ref)
      plgpu.barrier_wait(barrier_ref)
      # We don't await the load because acc_tmem is never modified again.
      out_ref[...] = plgpu.async_load_tmem(acc_tmem)

    scratch_shapes = [
        plgpu.Barrier(orders_tensor_core=True),
        plgpu.TMEM((m, n), jnp.float32),
        plgpu.TMEM((m, k // 2), jnp.uint2, layout=plgpu.TMEMLayout.SPARSE_METADATA_LAYOUT),
    ]

    f = self.pallas_call(
        kernel,
        in_specs=(
            plgpu.BlockSpec(memory_space=plgpu.SMEM, transforms=transforms),
            plgpu.BlockSpec(memory_space=plgpu.SMEM, transforms=transforms),
            plgpu.BlockSpec(memory_space=plgpu.SMEM),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        out_specs=plgpu.BlockSpec(transforms=out_transforms),
        scratch_shapes=scratch_shapes,
    )
    x = jax.random.uniform(jax.random.key(1), shape=(m, k // 2), dtype=dtype)
    y = jax.random.uniform(jax.random.key(2), shape=(n, k), dtype=dtype)
    index_pairs = np.asarray(np.meshgrid(range(4), range(4))).T.reshape(-1, 2)
    valid_pairs = index_pairs[index_pairs[:, 0] < index_pairs[:, 1]]
    assert len(valid_pairs) == 6
    x_pairs = jax.random.randint(jax.random.key(1234), (m, k // 4), 0, 6, dtype=jnp.uint8)
    x_sparse = valid_pairs[x_pairs]
    assert x_sparse.shape == (m, k // 4, 2)
    z = f(x, y, plgpu.format_tcgen05_sparse_metadata(x_sparse.astype(jnp.uint2)))
    x_logical = np.zeros_like(x, shape=(m, k // 4, 4))
    np.put_along_axis(x_logical, x_sparse, x.reshape(x_sparse.shape), axis=-1)
    x_logical = x_logical.reshape(m, k)
    ref = x_logical.astype(jnp.float32) @ y.T.astype(jnp.float32)
    np.testing.assert_allclose(z, ref, atol=7e-5, rtol=5e-6)

  @parameterized.parameters(
      (128, jnp.float16)
  )
  def test_manual_tcgen05_commit_arrive(self, swizzle, dtype):
    shape = (128, 128)
    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)

    def kernel(a_gmem, b_gmem, out_gmem,
        a_smem, b_smem, out_smem, tma_barrier, mma_barrier, acc_tmem):
      plgpu.copy_gmem_to_smem(a_gmem, a_smem, tma_barrier)
      plgpu.barrier_wait(tma_barrier)
      plgpu.copy_gmem_to_smem(b_gmem, b_smem, tma_barrier)
      plgpu.barrier_wait(tma_barrier)

      plgpu.commit_tmem()
      # Don't pass a barrier directly into tcgen05_mma and arrive manually.
      plgpu.tcgen05_mma(acc_tmem,
                        a_smem,
                        b_smem,
                        accumulate=False)
      plgpu.tcgen05_commit_arrive(mma_barrier)
      plgpu.barrier_wait(mma_barrier)
      # We don't await the load because acc_tmem is never modified again.
      out_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_smem, out_gmem)
      plgpu.wait_smem_to_gmem(0)

    f = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        scratch_shapes=[
          plgpu.SMEM(shape, dtype, transforms=transforms),  # a_smem
          plgpu.SMEM(shape, dtype, transforms=transforms),  # b_smem
          plgpu.SMEM(shape, dtype, transforms=transforms),  # out_smem
          plgpu.Barrier(),  # tma_barrier
          plgpu.Barrier(orders_tensor_core=True),  # mma_barrier
          plgpu.TMEM((128, 128), jnp.float32),  # acc
        ],
    )
    x = jax.random.uniform(jax.random.key(0), shape=shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=shape, dtype=dtype)
    result = f(x, y)
    np.testing.assert_allclose(result, x @ y, rtol=1e-3)

  def test_matmul_with_sliced_accumulator(self):
    dtype = jnp.bfloat16
    shape = (128, 128)
    tmem_shape = (128, 2 * 128)
    swizzle = 128

    # Test a matmul with a single block.
    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)

    def kernel(a_smem, b_smem, out_ref, acc_tmem, scratch_smem, barrier_ref):
      acc_tmem_slice = acc_tmem.at[slice(None), pl.dslice(0, 128)]
      plgpu.tcgen05_mma(acc_tmem_slice,
                        a_smem,
                        b_smem,
                        barrier_ref,
                        accumulate=False)
      plgpu.barrier_wait(barrier_ref)
      # We don't await the load because acc_tmem is never modified again.
      scratch_smem[...] = plgpu.async_load_tmem(acc_tmem_slice).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_smem, out_ref)
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.TMEM(tmem_shape, jnp.float32, packed=False),
        plgpu.SMEM(shape, dtype, transforms=transforms),
        plgpu.Barrier(orders_tensor_core=True),
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

  @parameterized.product(
      m_n_k=[
          (256, 256, 256),
          (256, 128, 128),
          (256, 256, 64),
          (128, 64, 128),
          (128, 64, 128),
      ],
      swizzle=[128, 64, 32],
      dtype=[jnp.float16, jnp.bfloat16],
      lhs_tmem=[False, True],
  )
  def test_simple_collective_matmul(self, m_n_k, swizzle, dtype, lhs_tmem):
    m, n, k = m_n_k
    if (n // 2) * jnp.dtype(dtype).itemsize < swizzle:
      self.skipTest("swizzle too big")
    full_lhs_shape = (m, k)
    full_rhs_shape = (k, n)
    full_acc_shape = (m, n)
    block_acc_shape = (m // 2, n)
    block_lhs_shape = (m // 2, k)
    block_rhs_shape = (k, n // 2)
    # Test a collective (paired CTA) matmul on a single block.
    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)
    if lhs_tmem and m == 128:
      self.skipTest("m=128 not supported for LHS in TMEM")

    def kernel(a_gmem, b_gmem, out_gmem, a_smem, b_smem,
               scratch_smem, acc_tmem, tma_barrier, mma_barrier,
               cluster_barrier, lhs_tmem_ref):
      cluster_idx = lax.axis_index("x")
      slice_lhs = pl.ds(cluster_idx * block_lhs_shape[0], block_lhs_shape[0])
      slice_rhs = pl.ds(cluster_idx * block_rhs_shape[1], block_rhs_shape[1])

      plgpu.copy_gmem_to_smem(a_gmem.at[slice_lhs, :], a_smem, tma_barrier)
      plgpu.barrier_wait(tma_barrier)
      plgpu.copy_gmem_to_smem(b_gmem.at[:, slice_rhs], b_smem, tma_barrier)
      plgpu.barrier_wait(tma_barrier)

      if lhs_tmem:
        lhs_ref = lhs_tmem_ref
        plgpu.async_store_tmem(lhs_ref, plgpu.load(a_smem, (), layout=plgpu.Layout.TCGEN05))
        plgpu.commit_tmem()
      else:
        lhs_ref = a_smem

      plgpu.barrier_arrive(cluster_barrier)
      plgpu.barrier_wait(cluster_barrier)

      plgpu.tcgen05_mma(
          acc_tmem,
          lhs_ref,
          b_smem,
          mma_barrier,
          accumulate=False,
          collective_axis="x",
      )
      plgpu.barrier_wait(mma_barrier)
      if m == 128:
        layout = plgpu.Layout.TCGEN05_M64_COLLECTIVE(n)
      else:
        layout = plgpu.Layout.TCGEN05
      # We don't await the load because acc_tmem is never modified again.
      scratch_smem[...] = plgpu.async_load_tmem(acc_tmem, layout=layout).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_smem, out_gmem.at[slice_lhs, :])
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.SMEM(block_lhs_shape, dtype, transforms=transforms),
        plgpu.SMEM(block_rhs_shape, dtype, transforms=transforms),
        plgpu.SMEM(block_acc_shape, dtype, transforms=transforms),
        plgpu.TMEM(block_acc_shape, jnp.float32, collective=True),
        plgpu.Barrier(),
        plgpu.Barrier(orders_tensor_core=True),
        plgpu.ClusterBarrier(collective_axes=("x",)),
    ]
    if lhs_tmem:
      scratch_shapes.append(
          plgpu.TMEM(block_lhs_shape, dtype, collective=True, packed=True)
      )
    else:
      scratch_shapes.append(None)

    f = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct(full_acc_shape, dtype),
        grid=(1,),
        grid_names=("_",),
        cluster=(2,),
        cluster_names=("x",),
        scratch_shapes=scratch_shapes,
    )
    x = jax.random.uniform(jax.random.key(0), shape=full_lhs_shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=full_rhs_shape, dtype=dtype)
    result = f(x, y)
    expected = x @ y
    np.testing.assert_allclose(result, expected, rtol=1e-3)

  @parameterized.parameters(
      (128, jnp.float16)
  )
  def test_matmul_with_smem_aliasing(self, swizzle, dtype):
    # Perform a 128x128 @ 128x128 matmul and a 128x64 @ 64x128 matmul
    # using aliased Refs pointing to the same SMEM address.
    self.skip_if_wg_semantics()
    shape = (128, 128)
    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)

    def kernel(a_gmem, b_gmem, out_gmem128, out_gmem64,
        a_aliased, b_aliased, out_smem, tma_barrier, mma_barrier, acc_tmem):
      # Note: We directly copy into 128-sized refs assuming that both aliased
      # refs point to the same address, so we can skip the copy for
      # the 64-sized ref. We transpose the LHS Ref so that the 64-sized Ref
      # receives the correct slice of data from this TMA.
      # As this is implementation dependent, this test may break if we change
      # the underlying aliasing behavior.
      a_smem_128, a_smem_64 = a_aliased
      plgpu.copy_gmem_to_smem(a_gmem, a_smem_128, tma_barrier)
      plgpu.barrier_wait(tma_barrier)
      b_smem_128, b_smem_64 = b_aliased
      plgpu.copy_gmem_to_smem(b_gmem, b_smem_128, tma_barrier)
      plgpu.barrier_wait(tma_barrier)

      # Do 128x128 @ 128x128 matmul
      plgpu.tcgen05_mma(acc_tmem,
                        plgpu.transpose_ref(a_smem_128, (1, 0)),
                        b_smem_128,
                        mma_barrier,
                        accumulate=False)
      plgpu.barrier_wait(mma_barrier)
      out_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_smem, out_gmem128)
      plgpu.wait_smem_to_gmem(0)

      # Do 128x64 @ 64x128 matmul
      plgpu.wait_load_tmem()  # Make sure the loads are complete
      plgpu.tcgen05_mma(acc_tmem,
                        plgpu.transpose_ref(a_smem_64, (1, 0)),
                        b_smem_64,
                        mma_barrier,
                        accumulate=False)
      plgpu.barrier_wait(mma_barrier)
      out_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_smem, out_gmem64)
      plgpu.wait_smem_to_gmem(0)

    f = self.kernel(
        kernel,
        out_shape=[jax.ShapeDtypeStruct(shape, dtype),
                   jax.ShapeDtypeStruct(shape, dtype)],
        scratch_shapes=[
          plgpu.RefUnion(   # aliased a_smem
            plgpu.SMEM(shape, dtype, transforms=transforms),
            plgpu.SMEM((64, 128), dtype, transforms=transforms),
          ),
          plgpu.RefUnion(   # aliased b_smem
            plgpu.SMEM(shape, dtype, transforms=transforms),
            plgpu.SMEM((64, 128), dtype, transforms=transforms),
          ),
          plgpu.SMEM(shape, dtype, transforms=transforms),  # out_smem
          plgpu.Barrier(),  # tma_barrier
          plgpu.Barrier(orders_tensor_core=True),  # mma_barrier
          plgpu.TMEM(shape, jnp.float32),  # acc
        ],
    )
    x = jax.random.uniform(jax.random.key(0), shape=shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=shape, dtype=dtype)
    result_128, result_64 = f(x.T, y)
    np.testing.assert_allclose(result_128, x @ y, rtol=1e-3)
    np.testing.assert_allclose(result_64, x[:, :64] @ y[:64, :], rtol=1e-3)

  @parameterized.parameters(
      (128, jnp.float16)
  )
  def test_matmul_with_tmem_aliasing(self, swizzle, dtype):
    # Perform a 128x128 @ 128x128 matmul and a 128x64 @ 64x128 matmul
    # using aliased Refs pointing to the same TMEM address.
    self.skip_if_wg_semantics()
    shape = (128, 128)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def kernel(a_gmem, b_gmem, out_gmem128, out_gmem64,
        a_smem, b_smem, out_smem, tma_barrier, mma_barrier, aliased_refs):
      plgpu.copy_gmem_to_smem(a_gmem, a_smem, tma_barrier)
      plgpu.barrier_wait(tma_barrier)
      plgpu.copy_gmem_to_smem(b_gmem, b_smem, tma_barrier)
      plgpu.barrier_wait(tma_barrier)
      [acc_128, lhs_128], [lhs_64, acc_64], _ = aliased_refs

      # Do 128x128 @ 128x128 matmul
      plgpu.async_store_tmem(lhs_128, plgpu.load(a_smem, (), layout=plgpu.Layout.TCGEN05))
      plgpu.commit_tmem()
      plgpu.tcgen05_mma(acc_128,
                        lhs_128,
                        b_smem,
                        mma_barrier,
                        accumulate=False)
      plgpu.barrier_wait(mma_barrier)
      out_smem[...] = plgpu.async_load_tmem(acc_128).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_smem, out_gmem128)
      plgpu.wait_smem_to_gmem(0)

      # Do 128x64 @ 64x128 matmul
      plgpu.wait_load_tmem()  # Make sure the loads have completed
      plgpu.async_store_tmem(
          lhs_64,
          plgpu.load(a_smem.at[:, 0:64], (), layout=plgpu.Layout.TCGEN05),
      )
      plgpu.commit_tmem()
      plgpu.tcgen05_mma(acc_64,
                        lhs_64,
                        b_smem.at[0:64, :],
                        mma_barrier,
                        accumulate=False)
      plgpu.barrier_wait(mma_barrier)
      # We don't await the load because TMEM is never modified again.
      out_smem[...] = plgpu.async_load_tmem(acc_64).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_smem, out_gmem64)
      plgpu.wait_smem_to_gmem(0)

    f = self.kernel(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct(shape, dtype),
            jax.ShapeDtypeStruct(shape, dtype),
        ],
        scratch_shapes=[
            plgpu.SMEM(shape, dtype, transforms=transforms),  # a_smem
            plgpu.SMEM(shape, dtype, transforms=transforms),  # b_smem
            plgpu.SMEM(shape, dtype, transforms=transforms),  # out_smem
            plgpu.Barrier(),  # tma_barrier
            plgpu.Barrier(orders_tensor_core=True),  # mma_barrier
            plgpu.RefUnion(  # aliased_refs
                [
                    plgpu.TMEM((128, 128), jnp.float32),  # acc
                    plgpu.TMEM((128, 128), dtype, packed=True),  # lhs
                ],
                [
                    plgpu.TMEM((128, 64), dtype, packed=True),  # lhs
                    plgpu.TMEM((128, 128), jnp.float32),  # acc
                ],
                plgpu.TMEM((128, 128), jnp.float32),  # unused
            ),
        ],
    )
    x = jax.random.uniform(jax.random.key(0), shape=shape, dtype=dtype)
    y = jax.random.uniform(jax.random.key(1), shape=shape, dtype=dtype)
    result_128, result_64 = f(x, y)
    np.testing.assert_allclose(result_128, x @ y, rtol=1e-3)
    np.testing.assert_allclose(result_64, x[:, :64] @ y[:64, :], rtol=1e-3)

  @parameterized.parameters((0,), (1,))
  def test_mma_barrier_indexing(
      self, barrier_index, shape=(128, 128), swizzle=128, dtype=jnp.float16
  ):
    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)

    def kernel(a_smem, b_smem, out_ref, acc_tmem, scratch_smem, barrier_ref):
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier_ref.at[barrier_index],
          accumulate=False,
      )
      plgpu.barrier_wait(barrier_ref.at[barrier_index])
      scratch_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_smem, out_ref)
      plgpu.wait_smem_to_gmem(0)

    scratch_shapes = [
        plgpu.TMEM(shape, jnp.float32, packed=False),
        plgpu.SMEM(shape, dtype, transforms=transforms),
        plgpu.Barrier(num_barriers=2, orders_tensor_core=True),
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

  @parameterized.product(
      warp_level=(True, False),
      squeezed_index=(True, False),
  )
  def test_copy_gmem_to_smem_partitioned(self, warp_level, squeezed_index):
    self.skip_if_wg_semantics()  # `pl.core_map` not implemented for warpgroup.
    block_size = (128, 128)
    partitioned_block_size = (block_size[0] // 2, block_size[1])
    a = jax.random.uniform(
        jax.random.key(0), shape=block_size, dtype=jnp.float32)
    if squeezed_index:
      a = a.reshape(1, *block_size)
    b = jax.random.uniform(
        jax.random.key(1), shape=block_size, dtype=jnp.float32)
    def kernel(a_gmem, b_gmem, out_gmem,
              a_smem, b_smem, out_smem,
              a_tma_barrier, b_tma_barrier, cluster_barrier):
      if squeezed_index:
        a_gmem = a_gmem.at[0]
      cluster_idx = lax.axis_index("x")
      out_slice = pl.ds(cluster_idx * partitioned_block_size[0],
                        partitioned_block_size[0])

      if warp_level:
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")
          @pl.when(warp_id == 0)
          def _():
            plgpu.copy_gmem_to_smem(
                a_gmem,
                a_smem,
                a_tma_barrier,
                collective_axes="x",
                partitioned_axis=0,
            )
            plgpu.copy_gmem_to_smem(
                b_gmem,
                b_smem,
                b_tma_barrier,
                collective_axes="x",
                partitioned_axis=0,
            )
      else:
        plgpu.copy_gmem_to_smem(
            a_gmem,
            a_smem,
            a_tma_barrier,
            collective_axes="x",
            partitioned_axis=0,
        )
        plgpu.copy_gmem_to_smem(
            b_gmem,
            b_smem,
            b_tma_barrier,
            collective_axes="x",
            partitioned_axis=0,
        )
      @pl.when(cluster_idx == 0)
      def _():
        plgpu.barrier_wait(a_tma_barrier)
        plgpu.barrier_wait(b_tma_barrier)
      plgpu.barrier_arrive(cluster_barrier)
      plgpu.barrier_wait(cluster_barrier)
      out_smem[...] = a_smem[...] + b_smem[...]
      plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[out_slice])
      plgpu.wait_smem_to_gmem(0)
    f = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct(block_size, jnp.float32),
        grid=(1,),
        grid_names=("_"),
        cluster_names=("x",),
        cluster=(2,),
        scratch_shapes=(  # type: ignore
            plgpu.SMEM(partitioned_block_size, jnp.float32),
            plgpu.SMEM(partitioned_block_size, jnp.float32),
            plgpu.SMEM(partitioned_block_size, jnp.float32),
            plgpu.Barrier(num_arrivals=1),
            plgpu.Barrier(num_arrivals=1),
            plgpu.ClusterBarrier(collective_axes=("x",)),
        ),
    )
    result = f(a, b)
    if squeezed_index:
      a = a[0]
    np.testing.assert_array_equal(result, a + b)

  def test_arrive_wait_on_tc_barrier(self):
    def kernel(out_ref, barrier):
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)
      out_ref[...] = jnp.ones_like(out_ref)

    f = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        scratch_shapes=(  # type: ignore
            plgpu.Barrier(num_arrivals=1, orders_tensor_core=True),
        ),
    )
    np.testing.assert_array_equal(f(), np.ones((128,), np.float32))

  @parameterized.parameters(
      (0, (1,), False),
      (0, (1,), True),
      (1, (1,), False),
      (2, (1,), False),
      (0, (1, 2,), False),
      (0, (2, 1,), False),
  )
  def test_cluster_launch_control(self, dim, cluster, with_indexing):
    self.skip_if_wg_semantics()
    # We attempt to schedule 1 more CTA than can be scheduled at once. Only
    # one CTA will succeed in stealing the last block, and the others will
    # fail. Therefore we test that there is exactly 1 stolen block and the
    # others fail and return -1.

    num_sms = jax.devices()[0].core_count
    cluster_size = math.prod(cluster)

    grid = [1, 1, 1]
    grid[dim] = num_sms // cluster_size + 1

    grid_names = tuple("xyz"[: len(grid)])
    cluster_names = tuple("abc"[: len(cluster)])

    def kernel(out_ref, cancel_result_ref, barrier, _):
      if with_indexing:
        cancel_result_ref = cancel_result_ref.at[0]
      plgpu.try_cluster_cancel(cancel_result_ref, barrier)
      plgpu.barrier_wait(barrier)

      cta_ids, cancelled_launch = plgpu.query_cluster_cancel(
          cancel_result_ref, grid_names=grid_names)
      cta_id = sum(cta_ids)

      # Store a sentinel value if no work can be scheduled.
      value = lax.select(cancelled_launch, cta_id, jnp.int32(-1))

      grid_idx = lax.axis_index(grid_names) * lax.axis_size(
          cluster_names
      ) + lax.axis_index(cluster_names)
      out_ref[grid_idx] = value

    f = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_sms,), jnp.int32),
        grid=grid,
        grid_names=grid_names,
        num_threads=2,
        thread_name="wg",
        cluster=cluster,
        cluster_names=cluster_names,
        scratch_shapes=[
            plgpu.TryClusterCancelResult(2 if with_indexing else None),
            plgpu.Barrier(num_arrivals=2),
            # Requesting SMEM close to the 228kb limit to ensure that each SM
            # only schedules 1 block.
            plgpu.SMEM((220 * 1024,), jnp.int8),
        ],
    )
    result = np.sort(f())
    last_cta_id = math.ceil(num_sms / cluster_size)
    expected = np.array([-1] * (num_sms - cluster_size) + [last_cta_id] * cluster_size)
    np.testing.assert_equal(result, expected)


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
      repeats=(1, 10),
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

  def test_emit_with_no_output(self):
    m, n = 16, 128

    def kernel(x_gmem, o_gmem, o_smem):
      def acc_scope(acc_ref):
        acc_ref[...] = jnp.zeros_like(acc_ref)
        def body(_, x_smem):
          acc_ref[...] += x_smem[...]  # Can't += in a lambda...
        in_specs = [plgpu.BlockSpec((1, n), lambda i: (i, 0), delay_release=1)]
        plgpu.emit_pipeline(
            body,
            in_specs=in_specs,
            grid=(m,),
            max_concurrent_steps=2,
        )(x_gmem)
        return acc_ref[...]

      acc = pl.run_scoped(acc_scope, plgpu.SMEM((1, n), dtype=jnp.float32))

      o_smem[...] = acc[...]
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(o_smem, o_gmem)

    dtype = jnp.float32
    x = jax.random.uniform(jax.random.key(0), (m, n)).astype(dtype)

    kernel_fn = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, n), dtype),
        scratch_shapes=[plgpu.SMEM((1, n), dtype)],
    )

    np.testing.assert_allclose(kernel_fn(x), x.sum(0, keepdims=True), rtol=1e-6)

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

  def test_emit_with_dynamic_grid_smaller_than_concurrent_steps(self):
    block_x = 128
    x = jax.random.randint(jax.random.key(1234), (block_x,),
                           minval=-128, maxval=128, dtype=jnp.int32)

    def body(num_blocks_gmem, x_gmem, o_gmem):
      num_blocks = num_blocks_gmem[...]
      def pipeline_body(_, x_smem, o_smem):
        o_smem[...] = x_smem[...]
      for _ in range(2):
        plgpu.emit_pipeline(
            pipeline_body,
            grid=(num_blocks,),
            in_specs=[plgpu.BlockSpec((block_x,), lambda i: (i,))],
            out_specs=[plgpu.BlockSpec((block_x,), lambda i: (i,))],
            max_concurrent_steps=2,
        )(x_gmem, o_gmem)

    # The test only intends to check that this does not crash/hang.
    plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((block_x,), jnp.int32),
        grid=(1,),
        grid_names=("blocks",)
    )(0, x).block_until_ready()

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
    dtype = jnp.float16
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    grid_m, grid_k, grid_n = 132, 10, 4
    tile_m = tile_n = 128
    assert tile_m % elems_128b == 0
    tile_k = elems_128b
    m, k, n = grid_m * tile_m, grid_k * tile_k, grid_n * tile_n

    transforms = self.default_transforms(swizzle=swizzle, dtype=dtype)

    def kernel(a_gmem, b_gmem, o_smem, acc):
      def kernel_body(_, a_smem, b_smem):
        assert a_smem.shape == (tile_m, tile_k)
        assert b_smem.shape == (tile_k, tile_n)
        plgpu.wgmma(acc, a_smem, b_smem)
        plgpu.wgmma_wait(1)

      pid_m = pl.program_id(0)
      pid_n = pl.program_id(1)
      in_specs = [
          plgpu.BlockSpec(
              (tile_m, tile_k), lambda k: (pid_m, k), transforms=transforms,
              delay_release=1,
          ),
          plgpu.BlockSpec(
              (tile_k, tile_n), lambda k: (k, pid_n), transforms=transforms,
              delay_release=1,
          ),
      ]
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=in_specs,
          grid=(grid_k,),
          max_concurrent_steps=2,
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

  @parameterized.product(m=[512], n=[512], repeats=[1, 10],
                         manual_consumed_barriers=[False, True],
                         max_concurrent_steps=[2, 3])
  def test_pipelined_copy(
      self, m, n, repeats, manual_consumed_barriers, max_concurrent_steps
  ):
    x = jax.random.uniform(jax.random.key(0), (m, n), dtype=jnp.float16)
    blk_m = blk_n = 32

    def copy_kernel(_, x_smem, o_smem, o_last_block_smem, *consumed_barriers):
      wg_idx = lax.axis_index("wg")
      o_smem[...] = x_smem[...]
      o_last_block_smem[...] = x_smem[...]
      if manual_consumed_barriers:
        [x_barrier] = consumed_barriers
        plgpu.barrier_arrive(x_barrier)

    spec = pl.BlockSpec(
        block_shape=(2 * blk_m, blk_n), index_map=lambda i, j: (i, j)
    )
    def body(*gmem_refs):
      pipeline = mgpu_pipeline.emit_pipeline_warp_specialized(
          copy_kernel,
          grid=(m // (2 * blk_m), n // blk_n),
          memory_registers=40,
          max_concurrent_steps=max_concurrent_steps,
          num_compute_wgs=1,
          wg_axis="wg",
          manual_consumed_barriers=manual_consumed_barriers,
          in_specs=[spec],
          out_specs=[
              spec,
              # Create an index-invariant output.
              pl.BlockSpec(
                  block_shape=(2 * blk_m, blk_n), index_map=lambda i, j: (0, 0)
              ),
          ],
      )
      for _ in range(repeats):
        pipeline(*gmem_refs)  # Make sure we can run the pipeline multiple times
    kernel = self.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((m, n), jnp.float16),
            jax.ShapeDtypeStruct((2 * blk_m, blk_n), jnp.float16),
        ),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=(1,),
        grid_names=("_",),
        num_threads=2,
        thread_name="wg",
    )
    out, out_last_block = kernel(x)
    np.testing.assert_array_equal(out, x)
    np.testing.assert_array_equal(out_last_block, x[-(2 * blk_m):, -blk_n:])

  @parameterized.product(
      m=[256, 64],
      n=[256, 64],
      num_compute_wgs=[1],  # TODO(apaszke): Use 2WGs once we add support for outputs.
      static=[False, True],
      manual_consumed_barriers=[False, True],
      in_tree_template=[(0, 1), ((0, (1,), None))],
  )
  @jtu.skip_if_mosaic_gpu_exceeds_shared_memory(device_patterns="RTX PRO 6000 Blackwell")
  def test_elementwise_add(self, m, n, num_compute_wgs, static,
                           manual_consumed_barriers, in_tree_template):
    self.skip_if_wg_semantics()  # Crashes!

    blk_m = blk_n = 64
    if m % (num_compute_wgs * blk_m):
      self.skipTest(f"{m=} must be divisible by {num_compute_wgs=} * {blk_m=}")
    spec = pl.BlockSpec(
        block_shape=(num_compute_wgs * blk_m, blk_n),
        index_map=lambda i, j: (i, j),
    )
    in_treedef = jax.tree.structure(in_tree_template)
    in_specs = jax.tree.unflatten(in_treedef, (spec, spec))

    def tiled_add_kernel(_, *smems):
      flat_smems, _ = jax.tree.flatten(smems)
      x_smem, y_smem, o_smem, *consumed_barriers = flat_smems

      wg_idx = lax.axis_index("wg")
      m_slice = pl.ds(wg_idx * blk_m, blk_m)
      o_smem[m_slice] = x_smem[m_slice] + y_smem[m_slice]
      if manual_consumed_barriers:
        [x_consumed_barrier, y_consumed_barrier] = consumed_barriers
        plgpu.barrier_arrive(x_consumed_barrier)
        plgpu.barrier_arrive(y_consumed_barrier)

    def pipeline(*gmem_refs):
      grid = (m // (num_compute_wgs * blk_m), n // blk_n)
      if not static:
        grid = jax.tree.map(jnp.asarray, grid)
      return mgpu_pipeline.emit_pipeline_warp_specialized(
          tiled_add_kernel,
          grid=grid,
          max_concurrent_steps=2,
          num_compute_wgs=num_compute_wgs,
          memory_registers=40,
          wg_axis="wg",
          in_specs=in_specs,
          out_specs=[spec],
          manual_consumed_barriers=manual_consumed_barriers,
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
    inputs = jax.tree.unflatten(in_treedef, (x, y))
    np.testing.assert_allclose(kernel(*inputs), x + y, atol=1e-4)

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

  @parameterized.product(
      num_compute_wgs=[1],  # TODO(apaszke): Use 2WGs once we add support for outputs.
      static=[False, True],
      manual_consumed_barriers=[False, True],
      small_shape=[True, False],
      max_concurrent_steps=[2, 3, 4],
  )
  @jtu.skip_if_mosaic_gpu_exceeds_shared_memory(device_patterns="RTX PRO 6000 Blackwell")
  def test_delay_release(
      self, num_compute_wgs, static, manual_consumed_barriers, small_shape,
      max_concurrent_steps
  ):
    if small_shape:
      m = n = 64
    else:
      m = n = 256
    blk_m, blk_n = 32, 64
    spec = plgpu.BlockSpec(
        block_shape=(num_compute_wgs * blk_m, blk_n),
        index_map=lambda i, j: (i, j),
        delay_release=1,
    )
    out_spec = pl.BlockSpec(
        block_shape=(num_compute_wgs * blk_m, blk_n),
        index_map=lambda i, j: (i, j),
    )

    def tiled_add_kernel(idx, x_smem, y_smem, o_smem, *consumed_barriers):
      wg_idx = lax.axis_index("wg")
      m_slice = pl.ds(wg_idx * blk_m, blk_m)
      o_smem[m_slice] = x_smem[m_slice] + y_smem[m_slice]
      if manual_consumed_barriers:
        @pl.when(jnp.logical_or(idx[0] != 0, idx[1] != 0))
        def _signal_consumed():
          for b in consumed_barriers:
            plgpu.barrier_arrive(b)

    def pipeline(*gmem_refs):
      grid = (m // (num_compute_wgs * blk_m), n // blk_n)
      if not static:
        grid = jax.tree.map(jnp.asarray, grid)
      return mgpu_pipeline.emit_pipeline_warp_specialized(
          tiled_add_kernel,
          grid=grid,
          max_concurrent_steps=max_concurrent_steps,
          manual_consumed_barriers=manual_consumed_barriers,
          num_compute_wgs=num_compute_wgs,
          memory_registers=40,
          wg_axis="wg",
          in_specs=[spec, spec],
          out_specs=[out_spec],
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

  def test_different_delay_release(self):
    self.skip_if_wg_semantics()  # Crashes!
    m, n = 128, 64
    blk_m, blk_n = 32, 32
    in_specs = [
        plgpu.BlockSpec(
            block_shape=(blk_m, blk_n),
            index_map=lambda i, j: (i, j),
            delay_release=delay,
        )
        for delay in range(3)
    ]
    out_spec = pl.BlockSpec(
        block_shape=(blk_m, blk_n),
        index_map=lambda i, j: (i, j),
    )

    def tiled_add_kernel(_, x_smem, y_smem, z_smem, o_smem):
      o_smem[...] = x_smem[...] + y_smem[...] + z_smem[...]

    def pipeline(*gmem_refs):
      grid = (m // blk_m, n // blk_n)
      return mgpu_pipeline.emit_pipeline(
          tiled_add_kernel,
          grid=grid,
          max_concurrent_steps=4,
          in_specs=in_specs,
          out_specs=[out_spec],
      )(*gmem_refs)

    kernel = self.kernel(
        pipeline,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(1,),
        grid_names=("_",)
    )
    x = jax.random.uniform(jax.random.key(0), (m, n), dtype=jnp.float32)
    y = jax.random.uniform(jax.random.key(1), (m, n), dtype=jnp.float32)
    z = jax.random.uniform(jax.random.key(3), (m, n), dtype=jnp.float32)
    np.testing.assert_allclose(kernel(x, y, z), x + y + z)

  @parameterized.product(
      delay_release=[0, 1],
  )
  def test_repeat(self, delay_release):
    num_steps = 4

    def kernel_body(_, x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    def kernel(x_gmem, o_gmem):
      in_specs = [
          plgpu.BlockSpec((64, 64), lambda i: (0, i), delay_release=delay_release)
      ]
      out_specs = [plgpu.BlockSpec((64, 64), lambda i: (0, i))]
      for _ in range(3):
        plgpu.emit_pipeline_warp_specialized(
            kernel_body,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(num_steps,),
            max_concurrent_steps=2,
            num_compute_wgs=1,
            memory_registers=40,
            wg_axis="wg",
        )(x_gmem, o_gmem)

    x = jnp.arange(64 * num_steps * 64)
    x = x.reshape(-1, num_steps * 64).astype(jnp.float32)
    kernel_fn = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(1,),
        grid_names=("_",),
        num_threads=2,
        thread_name="wg",
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  @parameterized.parameters((False,), (True,))
  def test_stationary_input(self, flip):
    self.skip_if_wg_semantics()

    m = n = 256
    blk_m = blk_n = 64

    def add_kernel(_, x_smem, y_smem, o_smem):
      if flip:
        x_smem, y_smem = y_smem, x_smem
      o_smem[...] = x_smem[...] + y_smem[...]

    def body(*gmem_refs):
      mgpu_pipeline.emit_pipeline_warp_specialized(
          add_kernel,
          grid=(m // blk_m, n // blk_n),
          memory_registers=40,
          max_concurrent_steps=2,
          num_compute_wgs=1,
          wg_axis="wg",
          in_specs=[
              pl.BlockSpec(
                  block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
              ),
              pl.BlockSpec(
                  block_shape=(blk_m, blk_n), index_map=lambda i, j: (0, 0)
              )
          ][::(-1 if flip else 1)],
          out_specs=[
              pl.BlockSpec(
                  block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
              ),
          ],
      )(*gmem_refs)
    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float16),
        grid=(1,),
        grid_names=("_",),
        num_threads=2,
        thread_name="wg",
    )
    x = jax.random.uniform(jax.random.key(1), (m, n), dtype=jnp.float16)
    y = jax.random.uniform(jax.random.key(2), (blk_m, blk_n), dtype=jnp.float16)
    ref = x + np.tile(y, (m // blk_m, n // blk_n))
    if flip:
      x, y = y, x
    # TODO(apaszke,justinfu): Fix the bug (this test freezes) and remove this restriction.
    with self.assertRaisesRegex(
        NotImplementedError,
        "Only inputs with a dependency on the grid are supported.",
    ):
      out = kernel(x, y)
      np.testing.assert_array_equal(out, ref)

  def test_no_output(self):
    m = n = 256
    blk_m = blk_n = 64

    def body(x_ref, o_ref, o_scratch, barrier):
      @pl.when(lax.axis_index("wg") == 0)
      def _():
        o_scratch[...] = jnp.zeros_like(o_scratch)

      # Wait for scratch to be initialized
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)

      # Make sure we can run the pipeline many times. This also introduces
      # extra jitter into warp scheduling and has uncovered bugs in the past.
      @pl.loop(0, 10)
      def _pipeline_loop(_):
        def add(_, x_smem):
          slc = pl.ds(lax.axis_index("wg") * (blk_m // 2), blk_m // 2)
          o_scratch[slc] += x_smem[slc]
        mgpu_pipeline.emit_pipeline_warp_specialized(
            add,
            grid=(m // blk_m, n // blk_n),
            memory_registers=40,
            max_concurrent_steps=2,
            num_compute_wgs=2,
            wg_axis="wg",
            in_specs=[
                pl.BlockSpec(
                    block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
                ),
            ]
        )(x_ref)

      # Wait for both compute WGs to finish initializing the output
      plgpu.barrier_arrive(barrier)
      plgpu.barrier_wait(barrier)

      @pl.when(lax.axis_index("wg") == 0)
      def _():
        plgpu.copy_smem_to_gmem(o_scratch, o_ref)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((blk_m, blk_n), jnp.float32),
        num_threads=3,
        thread_name="wg",
        scratch_shapes=[
            plgpu.SMEM((blk_m, blk_n), jnp.float32),
            plgpu.Barrier(num_arrivals=3),
        ],
    )
    x = jax.random.uniform(jax.random.key(1234), (m, n), dtype=jnp.float32)
    ref = 10 * x.reshape(m // blk_m, blk_m, n // blk_n, blk_n).sum((0, 2))
    np.testing.assert_allclose(kernel(x), ref, rtol=5e-6)

  @parameterized.product(manual_consumed_barriers=[False, True])
  def test_pipelined_pipeline(self, manual_consumed_barriers):
    m = n = 512

    x = jax.random.randint(jax.random.key(0), (m, n), -10, 15, dtype=jnp.int32)
    blk_m = blk_n = 64

    def body(x_ref, out_gmem_ref, out_ref):
      wg_idx = jax.lax.axis_index("wg")
      @pl.when(wg_idx == 0)
      def _zero_output():
        out_ref[...] = jnp.zeros_like(out_ref)

      def pipeline_body(_, x_smem, *consumed_barriers):
        out_ref[...] += x_smem[...]
        if manual_consumed_barriers:
          [x_barrier] = consumed_barriers
          plgpu.barrier_arrive(x_barrier)

      spec = pl.BlockSpec(
          block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
      )
      pipeline = functools.partial(
          mgpu_pipeline.emit_pipeline_warp_specialized,
          body=pipeline_body,
          grid=(m // blk_m, n // blk_n),
          memory_registers=40,
          max_concurrent_steps=2,
          num_compute_wgs=1,
          wg_axis="wg",
          manual_consumed_barriers=manual_consumed_barriers,
          in_specs=[spec],
      )

      @functools.partial(
          pl.run_scoped,
          allocs=pipeline(pipeline_state=None).get_allocations(x_ref),
          collective_axes="wg",
      )
      def _pipeline_scope(allocs):
        @pl.loop(0, 2)
        def _outer_loop(_):
          @pl.loop(0, 4)
          def _pipeline_loop(i):
            state = plgpu.PipelinePipeline.START
            state = jnp.where(i > 0, plgpu.PipelinePipeline.STEADY, state)
            state = jnp.where(i == 3, plgpu.PipelinePipeline.STOP, state)
            pipeline(pipeline_state=state)(x_ref, allocations=allocs)
          # Make sure we have properly quiesced the pipeline.
          pipeline(pipeline_state=None)(x_ref, allocations=allocs)

      @pl.when(wg_idx == 0)
      def _store_out():
        out_gmem_ref[...] = out_ref[...]

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((blk_m, blk_n), jnp.int32),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        scratch_shapes=[plgpu.SMEM((blk_m, blk_n), jnp.int32)],
        grid=(1,),
        grid_names=("_",),
        num_threads=2,
        thread_name="wg",
    )
    out = kernel(x)
    np.testing.assert_array_equal(
        out, x.reshape(m // blk_m, blk_m, n // blk_n, blk_n).sum((0, 2)) * 10
    )

  @parameterized.product(manual_consumed_barriers=[False, True])
  def test_pipeline_with_manual_allocation(self, manual_consumed_barriers):
    m = n = 512

    x = jax.random.randint(jax.random.key(4), (m, n), -10, 15, dtype=jnp.int32)
    y = jax.random.randint(jax.random.key(5), (m, n), -10, 15, dtype=jnp.int32)
    blk_m = blk_n = 64

    def body(x_ref, y_ref, out_gmem_ref, out_ref):
      wg_idx = jax.lax.axis_index("wg")
      @pl.when(wg_idx == 0)
      def _zero_output():
        out_ref[...] = jnp.zeros_like(out_ref)

      def pipeline_body(_, x_smem, y_smem, *consumed_barriers):
        out_ref[...] += x_smem[...] + y_smem[...]
        for b in consumed_barriers:
          plgpu.barrier_arrive(b)

      spec = pl.BlockSpec(
          block_shape=(blk_m, blk_n), index_map=lambda i, j: (i, j)
      )
      pipeline = mgpu_pipeline.emit_pipeline_warp_specialized(
          body=pipeline_body,
          grid=(m // blk_m, n // blk_n),
          memory_registers=40,
          max_concurrent_steps=2,
          num_compute_wgs=1,
          wg_axis="wg",
          manual_consumed_barriers=manual_consumed_barriers,
          in_specs=[spec, spec],
      )

      @functools.partial(
          pl.run_scoped,
          allocs=pipeline.get_allocations(x_ref, y_ref),
          collective_axes="wg",
      )
      def _alloc_scope(allocs):
        @pl.loop(0, 4)
        def _outer_loop(_):
          pipeline(x_ref, y_ref, allocations=allocs)

      @pl.when(wg_idx == 0)
      def _store_out():
        out_gmem_ref[...] = out_ref[...]

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((blk_m, blk_n), jnp.int32),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        scratch_shapes=[plgpu.SMEM((blk_m, blk_n), jnp.int32)],
        grid=(1,),
        grid_names=("_",),
        num_threads=2,
        thread_name="wg",
    )
    np.testing.assert_array_equal(
        kernel(x, y),
        (x + y).reshape(m // blk_m, blk_m, n // blk_n, blk_n).sum((0, 2)) * 4,
    )

  @jtu.thread_unsafe_test()  # Modifies ``os.environ``.
  def test_collective(self):
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      cluster_idx = lax.axis_index("cluster")
      in_specs = [
          plgpu.BlockSpec(
              (64, 64), lambda i: (0, i), collective_axes=("cluster",)
          )
      ]
      out_specs = [plgpu.BlockSpec((1, 64, 64), lambda i: (cluster_idx, 0, i))]
      # Run a few times to make sure we leave barriers in a good state.
      for _ in range(3):
        def pipeline_body(_, x_smem, o_smem):
          o_smem[0, ...] = x_smem[...] + 1.0
        plgpu.emit_pipeline_warp_specialized(
            pipeline_body,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(num_steps,),
            max_concurrent_steps=2,
            num_compute_wgs=1,
            memory_registers=40,
            wg_axis="wg",
        )(x_gmem, o_gmem)

    x = jnp.arange(64 * num_steps * 64)
    x = x.reshape(-1, num_steps * 64).astype(jnp.float32)
    kernel_fn = self.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, *x.shape), x.dtype),
        num_threads=2,
        thread_name="wg",
        cluster=(2,),
        cluster_names=("cluster",)
    )
    with jtu.set_env(MOSAIC_GPU_DUMP_PTX="1"), self.capture_stdout() as ptx:
      y = jax.block_until_ready(kernel_fn(x))
    self.assertIn(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster",
        ptx(),
    )
    np.testing.assert_array_equal(y, np.stack([x + 1.0, x + 1.0]))


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

    transforms = self.default_transforms(dtype=dtype)

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
    transforms = self.default_transforms(dtype=jnp.float16)

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
          transforms=self.default_transforms(swizzle=64, dtype=jnp.float16),
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
    kernel = self.kernel(
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
      kernel = self.kernel(
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

  def test_global_semaphore(self):
    # We signal from block 0 and wait on block 1 to test whether the semaphore
    # is globally shared.
    def body(out_ref):
      sem_ref = pl.get_global(plgpu.SemaphoreType.REGULAR)
      block_id = lax.axis_index("x")
      @pl.when(block_id == 0)
      def _():
        pl.semaphore_signal(sem_ref)
      @pl.when(block_id == 1)
      def _():
        pl.semaphore_wait(sem_ref)
        out_ref[...] = jnp.ones_like(out_ref)

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(10,),
        grid_names=("x",),
    )
    result = kernel()
    np.testing.assert_array_equal(result, jnp.ones((128,), jnp.float32))

  def test_global_semaphore_with_multiple_threads(self):
    def body(out_ref):
      sem_ref = pl.get_global(plgpu.SemaphoreType.REGULAR)
      block_id = lax.axis_index("x")
      @pl.when(block_id == 0)
      def _():
        pl.semaphore_signal(sem_ref)
      @pl.when(block_id == 1)
      def _():
        pl.semaphore_wait(sem_ref)
        out_ref[...] = jnp.ones_like(out_ref)

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(10,),
        grid_names=("x",),
        thread_name="wg",
        num_threads=2,
    )
    result = kernel()
    np.testing.assert_array_equal(result, jnp.ones((128,), jnp.float32))

  def test_multiple_get_global_semaphores(self):
    def body(out_ref):
      sem1 = pl.get_global(plgpu.SemaphoreType.REGULAR)
      sem2 = pl.get_global(plgpu.SemaphoreType.REGULAR)
      block_id = lax.axis_index("x")
      @pl.when(block_id == 0)
      def _():
        pl.semaphore_signal(sem1, inc=5)
        pl.semaphore_signal(sem2, inc=10)
      @pl.when(block_id == 1)
      def _():
        pl.semaphore_wait(sem1, value=5, decrement=False)
        pl.semaphore_wait(sem2, value=10, decrement=False)
        val1 = pl.semaphore_read(sem1)
        val2 = pl.semaphore_read(sem2)
        out_ref[0] = val1
        out_ref[1] = val2

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        grid=(10,),
        grid_names=("x",),
    )
    result = kernel()
    np.testing.assert_array_equal(result, jnp.array([5, 10], jnp.int32))

  def test_get_global_in_and_outside_control_flow(self):
    def body(out_ref):
      sem_before = pl.get_global(plgpu.SemaphoreType.REGULAR)
      block_id = lax.axis_index("x")

      @pl.when(block_id == 0)
      def _():
        sem_inside = pl.get_global(plgpu.SemaphoreType.REGULAR)
        pl.semaphore_signal(sem_inside, 7)
        pl.semaphore_signal(sem_before, 3)
        val_inside = pl.semaphore_read(sem_inside)
        out_ref[1] = val_inside

      sem_after = pl.get_global(plgpu.SemaphoreType.REGULAR)
      pl.semaphore_signal(sem_after, 11)
      val_before = pl.semaphore_read(sem_before)
      val_after = pl.semaphore_read(sem_after)
      out_ref[0] = val_before
      out_ref[2] = val_after

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((3,), jnp.int32),
        grid=(1,),
        grid_names=("x",),
    )
    result = kernel()
    np.testing.assert_array_equal(result, jnp.array([3, 7, 11], jnp.int32))

  def test_multiple_semaphore_scopes(self):
    def body(out_ref):
      global_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)

      @functools.partial(pl.run_scoped, block_sem=plgpu.SemaphoreType.REGULAR)
      def _scope2(block_sem):
        block_id = lax.axis_index("x")
        pl.semaphore_signal(block_sem)

        @pl.when(block_id == 0)
        def _():
          pl.semaphore_signal(global_sem)

        @pl.when(block_id == 1)
        def _():
          pl.semaphore_wait(global_sem)
          out_ref[...] = jnp.ones_like(out_ref)

        pl.semaphore_wait(block_sem)

    kernel = self.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(10,),
        grid_names=("x",),
    )
    result = kernel()
    np.testing.assert_array_equal(result, jnp.ones((128,), jnp.float32))


class ExamplesWGTest(
    ExamplesTest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class ExamplesSm90ATest(PallasSm90ATest):

  # WGMMA
  def test_stage6(self):
    self.skip_if_wg_semantics()  # `fa.optimization_barrier` does not support f16 arrays.

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
      transforms = self.default_transforms(swizzle=64, dtype=jnp.float16)
      plgpu.emit_pipeline(
          compute,
          grid=(l_ref.shape[1] // k_block,),
          in_specs=[
              plgpu.BlockSpec(
                  (m_block, k_block), lambda k: (m, k), transforms=transforms
              ),
              plgpu.BlockSpec(
                  (k_block, n_block), lambda k: (k, n), transforms=transforms
              ),
          ],
          out_specs=[
              plgpu.BlockSpec(
                  (m_block, n_block), lambda k: (m, n), transforms=transforms
              )
          ],
      )(l_ref, r_ref, o_ref)

    np.testing.assert_allclose(kernel(x, x), x @ x)

  # TODO(apaszke): Clusters and multicast


class ExamplesSm90AWGTest(
    ExamplesSm90ATest, lowering_semantics=plgpu.LoweringSemantics.Warpgroup
):
  ...


class HelpersTest(PallasTest):

  @parameterized.product(
      m=[4, 16],
      n=[4, 16],
      minor_dim=[0, 1],
      tile_width=[1, 2, 4],
  )
  def test_planar_snake(self, m, n, minor_dim, tile_width):
    reference = np.full((m, n), -1)
    counter = itertools.count()
    minor_size, major_size = (m, n) if minor_dim == 0 else (n, m)
    for minor_tile in range(minor_size // tile_width):
      for major in range(major_size):
        major = major if minor_tile % 2 == 0 else major_size - 1 - major
        for minor_in_tile in range(tile_width):
          minor = minor_tile * tile_width + minor_in_tile
          idx = (minor, major) if minor_dim == 0 else (major, minor)
          reference[idx] = next(counter)
    results = np.full((m, n), -1)
    for lin in range(m * n):
      results[plgpu.planar_snake(np.int32(lin), (m, n), minor_dim, tile_width)] = lin
    np.testing.assert_array_equal(results, reference)

  def test_planar_snake_golden_with_partial_tile(self):
    m, n = 5, 5
    with self.subTest("minor_dim=0 tile_width=3"):
      results = np.full((m, n), -1)
      for lin in range(m * n):
        results[plgpu.planar_snake(np.int32(lin), (m, n), 0, 3)] = lin
      expected = np.array([
          [0, 3, 6, 9, 12],
          [1, 4, 7, 10, 13],
          [2, 5, 8, 11, 14],
          [23, 21, 19, 17, 15],
          [24, 22, 20, 18, 16]])
      np.testing.assert_array_equal(results, expected)
    with self.subTest("minor_dim=1 tile_width=3"):
      results = np.full((m, n), -1)
      for lin in range(m * n):
        results[plgpu.planar_snake(np.int32(lin), (m, n), 1, 3)] = lin
      expected = np.array([
          [0, 1, 2, 23, 24],
          [3, 4, 5, 21, 22],
          [6, 7, 8, 19, 20],
          [9, 10, 11, 17, 18],
          [12, 13, 14, 15, 16]])
      np.testing.assert_array_equal(results, expected)

  def test_planar_snake_golden(self):
    m, n = 8, 8
    with self.subTest("minor_dim=0 tile_width=2"):
      results = np.full((m, n), -1)
      for lin in range(m * n):
        results[plgpu.planar_snake(np.int32(lin), (m, n), 0, 2)] = lin
      expected = np.array([
          [0, 2, 4, 6, 8, 10, 12, 14],
          [1, 3, 5, 7, 9, 11, 13, 15],
          [30, 28, 26, 24, 22, 20, 18, 16],
          [31, 29, 27, 25, 23, 21, 19, 17],
          [32, 34, 36, 38, 40, 42, 44, 46],
          [33, 35, 37, 39, 41, 43, 45, 47],
          [62, 60, 58, 56, 54, 52, 50, 48],
          [63, 61, 59, 57, 55, 53, 51, 49],
      ])
      np.testing.assert_array_equal(results, expected)
    with self.subTest("minor_dim=1 tile_width=2"):
      results = np.full((m, n), -1)
      for lin in range(m * n):
        results[plgpu.planar_snake(np.int32(lin), (m, n), 1, 2)] = lin
      expected = np.array([
          [0, 1, 30, 31, 32, 33, 62, 63],
          [2, 3, 28, 29, 34, 35, 60, 61],
          [4, 5, 26, 27, 36, 37, 58, 59],
          [6, 7, 24, 25, 38, 39, 56, 57],
          [8, 9, 22, 23, 40, 41, 54, 55],
          [10, 11, 20, 21, 42, 43, 52, 53],
          [12, 13, 18, 19, 44, 45, 50, 51],
          [14, 15, 16, 17, 46, 47, 48, 49],
      ])
      np.testing.assert_array_equal(results, expected)
    with self.subTest("minor_dim=0 tile_width=1"):
      results = np.full((m, n), -1)
      for lin in range(m * n):
        results[plgpu.planar_snake(np.int32(lin), (m, n), 0, 1)] = lin
      expected = np.array([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [15, 14, 13, 12, 11, 10, 9, 8],
          [16, 17, 18, 19, 20, 21, 22, 23],
          [31, 30, 29, 28, 27, 26, 25, 24],
          [32, 33, 34, 35, 36, 37, 38, 39],
          [47, 46, 45, 44, 43, 42, 41, 40],
          [48, 49, 50, 51, 52, 53, 54, 55],
          [63, 62, 61, 60, 59, 58, 57, 56],
      ])
      np.testing.assert_array_equal(results, expected)

  @parameterized.parameters(
      ((100,), ()),  # grid < SM count
      ((300,), ()),  # grid > SM count
      ((3, 3, 3, 3, 3), ()),  #  squashed grid dimensions
      ((50,), (2, 1)),  # small grid w/ cluster
      ((50, 4), (1, 2)),  # large grid w/ cluster
  )
  def test_dynamic_work_scheduling(self, grid, cluster):
    if not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Only works on a GPU with capability >= sm100a")
    grid_names = tuple(str(i) for i in range(len(grid)))
    cluster_names = tuple("c"+str(i) for i in range(len(cluster)))
    def body(out_gmem, _):
      sm_idx = lax.axis_index(grid_names)
      cluster_idx = ()
      if cluster:
        cluster_idx = tuple(lax.axis_index(axis) for axis in cluster_names)
      @plgpu.dynamic_scheduling_loop(grid_names)
      def loop_body(loop_info: plgpu.NDLoopInfo):
        out_gmem[*loop_info.index, *cluster_idx] = sm_idx
    out_shape = (*grid, *cluster)
    max_shared_memory = jax.local_devices()[0].shared_memory_per_block_optin
    # Mosaic GPU uses some shared memory implicitly, so we can't
    # explicitly request the full amount.
    large_amount_of_shared_memory = int(0.9 * max_shared_memory)
    result = self.kernel(body,
                 out_shape=jax.ShapeDtypeStruct(out_shape, jnp.int32),
                 grid=grid,
                 grid_names=grid_names,
                 cluster=cluster,
                 cluster_names=cluster_names,
                 # Allocate a large amount of SMEM to prevent multiple blocks
                 # being scheduled on the same SM.
                 scratch_shapes=[
                   plgpu.SMEM((large_amount_of_shared_memory,), jnp.int8)],
                 )()

    # Result maps grid_idx -> SM that performed the work.
    # Check that each SM had at least 1 block of work.
    cluster_size = int(np.prod(cluster))
    num_sms = min(jax.devices()[0].core_count // cluster_size, np.prod(grid))
    histogram = np.histogram(result, bins=range(num_sms+1))[0]
    self.assertEqual(np.sum(histogram), np.prod(out_shape))
    self.assertGreaterEqual(np.min(histogram), 1)
    # Make sure all blocks > num_sms were stolen.
    self.assertEqual(np.max(result), jnp.int32(num_sms) - 1)

  def test_dynamic_work_scheduling_with_carry(self):
    if not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Only works on a GPU with capability >= sm100a")
    # In this test we make SM 0 run a the dynamic scheduling loop while all
    # other SMs spin. This means SM 0 should steal all of the work and we
    # keep track of the number of stolen blocks in the carry.
    blocks_to_steal = 100
    sm_count = jax.devices()[0].core_count
    def body(out_gmem, _):
      sm_idx = lax.axis_index("x")
      global_semaphore = pl.get_global(plgpu.SemaphoreType.REGULAR)

      @pl.when(sm_idx == 0)
      def _steal_loop():
        def loop_body(loop_info: plgpu.NDLoopInfo, carry: jax.Array):
          del loop_info
          return carry + jnp.int32(1)

        final_carry = plgpu.dynamic_scheduling_loop(
            ("x",), init_carry=jnp.int32(0)
        )(loop_body)
        out_gmem[0] = final_carry
        pl.semaphore_signal(global_semaphore, inc=sm_count)

      # All SMs wait until SM 0 has finished all blocks.
      pl.semaphore_wait(global_semaphore)

    max_shared_memory = jax.local_devices()[0].shared_memory_per_block_optin
    # Mosaic GPU uses some shared memory implicitly, so we can't
    # explicitly request the full amount.
    large_amount_of_shared_memory = int(0.9 * max_shared_memory)
    result = self.kernel(body,
                 out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
                 grid=(sm_count + blocks_to_steal,),
                 grid_names=("x",),
                 # Allocate a large amount of SMEM to prevent multiple blocks
                 # being scheduled on the same SM.
                 scratch_shapes=[
                   plgpu.SMEM((large_amount_of_shared_memory,), jnp.int8)],
                 )()
    self.assertEqual(result[0], blocks_to_steal + 1)


if __name__ == "__main__":
  absltest.main()
