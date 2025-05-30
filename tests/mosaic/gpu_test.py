# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Sequence
import contextlib
import dataclasses
import enum
import itertools
import math
import operator
import sys
import re
import unittest

from absl.testing import absltest, parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import cf
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
from jax.experimental.mosaic.gpu import dialect as mgpu_dialect  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import tcgen05
import jax.numpy as jnp
import numpy as np
try:
  import jax._src.lib.mosaic_gpu  as mosaic_gpu_lib # noqa: F401
  HAS_MOSAIC_GPU = True
except ImportError:
  mosaic_gpu_lib = None
  HAS_MOSAIC_GPU = False

  class Dimension(enum.IntEnum):  # Just to make parameterized tests expand ok
    x = 0
    y = 1
    z = 2
else:
  import jax.experimental.mosaic.gpu as mgpu
  from jax.experimental.mosaic.gpu import core
  from jax.experimental.mosaic.gpu import launch_context
  from jax.experimental.mosaic.gpu import layouts
  from jax.experimental.mosaic.gpu import utils as utils
  from jax.experimental.mosaic.gpu import profiler
  from jax.experimental.mosaic.gpu import inference_utils
  from jax.experimental.mosaic.gpu.utils import *  # noqa: F403
  from jax._src.lib.mlir.dialects import gpu
  from jax._src.lib.mlir.dialects import llvm
  Dimension = gpu.Dimension
try:
  import hypothesis as hp
  import hypothesis.strategies as hps
  jtu.setup_hypothesis()
except ImportError:
  hp = hps = None


# ruff: noqa: F405
# pylint: disable=g-complex-comprehension
config.parse_flags_with_absl()

def nd_loop(bounds, body, *, _idxs = ()):
  if not bounds:
    body(*_idxs)
    return
  bound, *other_bounds = bounds
  @fori(bound, ())
  def _loop_body(i, _):
    nd_loop(other_bounds, body, _idxs=(*_idxs, i))
    return ()


def mlir_sum(elems):
  assert elems
  total = elems[0]
  for elem in elems[1:]:
    total = arith.addi(total, elem)
  return total


def copy(src: ir.Value, dst: ir.Value, swizzle: int | None = None):
  index = ir.IndexType.get()
  thread_id = gpu.thread_id(gpu.Dimension.x)
  stride = gpu.block_dim(gpu.Dimension.x)
  for dim in (gpu.Dimension.y, gpu.Dimension.z):
    thread_id = arith.addi(thread_id, arith.muli(gpu.thread_id(dim), stride))
    stride = arith.muli(stride, gpu.block_dim(dim))
  is_first_thread = arith.cmpi(arith.CmpIPredicate.eq, thread_id, c(0, index))

  src_ty = ir.MemRefType(src.type)
  dst_ty = ir.MemRefType(dst.type)
  if src_ty.shape != dst_ty.shape:
    raise ValueError(
        f"src and dst shapes don't match: {src_ty.shape} != {dst_ty.shape}"
    )
  shape = src_ty.shape
  if src_ty.element_type != dst_ty.element_type:
    raise ValueError(
        f"src and dst element types don't match: {src_ty.element_type} !="
        f" {dst_ty.element_type}"
    )
  contig_strides = get_contiguous_strides(shape)
  # If swizzling is on, at least one of the memrefs must be contiguous
  # (simulating a TMA).
  if (swizzle is not None and
      src_ty.get_strides_and_offset()[0] != contig_strides and
      dst_ty.get_strides_and_offset()[0] != contig_strides):
    raise NotImplementedError(src_ty, dst_ty)

  bw = bitwidth(src_ty.element_type)
  if bw < 8:
    assert bw.bit_count() == 1
    packing = 8 // bw
    if shape[-1] % packing:
      raise NotImplementedError
    workgroup_mem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    shape = (*shape[:-1], shape[-1] // packing)
    contig_strides = get_contiguous_strides(shape)
    def bitcast(ref):
      ref_ty = ir.MemRefType(ref.type)
      old_strides = ref_ty.get_strides_and_offset()[0]
      if old_strides[-1] != 1:
        raise NotImplementedError
      new_strides = [s // packing for s in old_strides[:-1]] + [1]
      new_ref_ty = ir.MemRefType.get(
          shape,
          ir.VectorType.get((packing,), src_ty.element_type),  # noqa: F821
          ir.StridedLayoutAttr.get(0, new_strides),
          ref_ty.memory_space,
      )
      ptr_space = (
          3
          if ref_ty.memory_space is not None
          and ref_ty.memory_space == workgroup_mem
          else None
      )
      return ptr_as_memref(
          # NOTE: memref_ptr applies the offset in case there was any.
          memref_ptr(ref, memory_space=ptr_space),
          new_ref_ty,
          ptr_memory_space=ptr_space,
      )
    src = bitcast(src)
    dst = bitcast(dst)
    bw = 8
  del src_ty, dst_ty  # If you remove this, update it in the branch above
  dyn_strides = [c(s, index) for s in contig_strides]

  with ir.InsertionPoint(scf.IfOp(is_first_thread).then_block):
    def body(*idx):
      dst_idx = idx
      if swizzle is not None:
        assert swizzle.bit_count() == 1
        assert bw % 8 == 0
        bytes_per_element = bw // 8
        linear_idx = c(0, index)
        for stride, i in zip(dyn_strides, idx):
          linear_idx = arith.addi(linear_idx, arith.muli(i, stride))
        # Swizzle pattern repeats every 128 bytes.
        swizzle_src = arith.remui(
            arith.divui(linear_idx, c(128 // bytes_per_element, index)),
            c(swizzle // 16, index),
        )
        # Swizzle happens in groups of 16 bytes.
        swizzle_shift = 4 - (bytes_per_element.bit_length() - 1)
        dst_linear_idx = arith.xori(
            linear_idx, arith.shli(swizzle_src, c(swizzle_shift, index))
        )
        dst_idx = [
            arith.remui(arith.divui(dst_linear_idx, stride), c(bound, index))
            for stride, bound in zip(dyn_strides, shape)
        ]
      memref.store(memref.load(src, idx), dst, dst_idx)
    nd_loop([c(d, index) for d in shape], body)
    scf.yield_([])
  gpu.barrier()
  nvvm.fence_proxy(nvvm.ProxyKind.async_)


def iota_tensor(m, n, dtype):
  """A wgmma tensor where arr[i, j] = i * N + j."""
  index = ir.IndexType.get()
  mlir_dtype = utils.dtype_to_ir_type(dtype)
  int_ty = ir.IntegerType.get_signless(bitwidth(mlir_dtype))
  ret = mgpu.FragmentedArray.splat(
      llvm.mlir_undef(int_ty), (m, n), is_signed=False
  )
  ret = ret.to_layout(mgpu.WGMMA_LAYOUT)

  def iota_value(_, idx):
    assert len(idx) == 2
    return arith.index_cast(
        int_ty, arith.addi(idx[1], arith.muli(idx[0], c(n, index)))
    )

  ret = ret.foreach(
      iota_value,
      create_array=True,
      is_signed=False,
  )
  return ret.astype(mlir_dtype, is_signed=utils.is_signed(dtype))


class TestCase(parameterized.TestCase):

  def setUp(self):
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    super().setUp()
    self.prng = np.random.default_rng(1234)
    self.context = mlir.make_ir_context()
    mgpu_dialect.register_dialect(self.context)
    self.enter_context(config.traceback_filtering("off"))
    self.enter_context(self.context)
    self.enter_context(ir.Location.unknown())

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


class Sm90ATestCase(TestCase, jtu.CudaArchSpecificTest):

  def setUp(self):
      self.skip_unless_sm90a()
      super().setUp()


class TestUtilTest(TestCase):

  def test_copy_basic(self):
    def kernel(ctx, src, dst, _):
      copy(src, dst)
    x = jnp.arange(2 * 3 * 5).reshape(2, 5, 3)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, ())(x)
    np.testing.assert_array_equal(y, x)

  def test_copy_swizzle(self):
    def kernel(ctx, src, dst, _):
      copy(src, dst, swizzle=128)
    x = jnp.arange(8 * 32, dtype=jnp.float32).reshape(8, 32)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, ())(x)
    expected = np.zeros_like(y)
    for i in range(8):
      for j in range(8):
        js = j ^ i
        expected[i, (j * 4):(j * 4) + 4] = x[i, (js * 4):(js * 4) + 4]
    np.testing.assert_array_equal(y, expected)

  def test_copy_swizzle_noop(self):
    # Two swizzles cancel out
    def kernel(ctx, src, dst, smem):
      copy(src, smem, swizzle=128)
      copy(smem, dst, swizzle=128)
    x = jnp.arange(8 * 32, dtype=jnp.float32).reshape(8, 32)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)

  def test_iota_tensor(self):
    m = n = 64
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      index = ir.IndexType.get()
      registers = iota_tensor(m, n, jnp.float32).registers
      assert registers.size == 16, registers.size
      for i, vec_reg in enumerate(registers.flat):
        for j in range(2):
          reg = vector.extractelement(vec_reg, position=c(j, index))
          memref.store(
              reg, dst, [gpu.thread_id(gpu.Dimension.x), c(2 * i + j, index)]
          )
    out_shape = jax.ShapeDtypeStruct((128, 32), jnp.float32)
    regs = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    thread_ids = np.arange(128)
    warp_ids = thread_ids // 32
    lane_ids = thread_ids % 32
    thread_rows = warp_ids * 16 + lane_ids // 4
    thread_start_cols = (lane_ids % 4) * 2
    thread_cols = thread_start_cols[:, None] + (np.arange(n // 8)[None] * 8)
    regs = regs.reshape(128, 8, 2, 2)
    for row_half in range(2):
      for col_half in range(2):
        np.testing.assert_array_equal(
            regs[..., row_half, col_half],
            (thread_rows[:, None] + row_half * 8) * n + thread_cols + col_half
        )


class MemRefTest(TestCase):
  @parameterized.product(
      dim=tuple(range(3)),
      strided=(False, True)
  )
  def test_unsqueeze(self, dim, strided):
    def kernel(ctx, inp, out, _):
      if strided:
        for i in range(8):
          s = ds(i, 1)
          out_slice = s if dim != 0 else (slice(None), s)
          copy(
              memref_unsqueeze(memref_slice(inp, s), dim),
              memref_slice(out, out_slice),
          )
      else:
        copy(memref_unsqueeze(inp, dim), out)
    x = np.arange(8 * 16, dtype=jnp.float32).reshape(8, 16)
    out_shape = list(x.shape)
    out_shape.insert(dim, 1)
    out_ty = jax.ShapeDtypeStruct(out_shape, jnp.float32)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_shape))

  @parameterized.product(
      dim=tuple(range(2)),
      strided=(False, True)
  )
  def test_unfold(self, dim, strided):
    in_shape = (8, 16)
    def kernel(ctx, inp, out, _):
      if strided:
        # We slice the dim we don't unfold
        for i in range(in_shape[1 - dim] // 4):
          s = ds(i * 4, 4)
          in_slice = s if dim == 1 else (slice(None), s)
          out_slice = s if dim == 1 else (slice(None),) * 3 + (s,)
          copy(
              memref_unfold(memref_slice(inp, in_slice), dim, (2, 2, None)),
              memref_slice(out, out_slice),
          )
      else:
        copy(memref_unfold(inp, dim, (2, 2, None)), out)
    x = np.arange(np.prod(in_shape), dtype=jnp.float32).reshape(in_shape)
    out_shape = list(in_shape)
    out_shape[dim:dim + 1] = [2, 2, out_shape[dim] // 4]
    out_ty = jax.ShapeDtypeStruct(out_shape, jnp.float32)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_ty.shape))

  @parameterized.product(
      dim=tuple(range(2)),
  )
  def test_fold_not_strided(self, dim):
    def kernel(ctx, inp, out, _):
      copy(memref_fold(inp, dim, 2), out)

    x = np.arange(8 * 2 * 8, dtype=jnp.float32).reshape(8, 2, 8)
    out_ty = jax.ShapeDtypeStruct((16, 8) if dim == 0 else (8, 16), jnp.float32)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_ty.shape))

  @parameterized.named_parameters(
      ("drop_1s", (1, 1, 5, 1, 1, 2, 1, 1), (5, 1, 2)),
      ("add_1s", (5, 1, 2), (1, 1, 5, 1, 1, 2, 1, 1)),
      ("fold", (1, 5, 2, 1,), (1, 10, 1)),
      ("un", (1, 10, 1), (1, 5, 2, 1,)),
      ("to_scalar", (1, 1, 1), ()),
      ("from_scalar", (), (1, 1, 1)),
  )
  def test_reshape(self, inp_shape, out_shape):
    def kernel(ctx, inp, out, _):
      copy(memref_reshape(inp, out_shape), out)

    x = np.arange(math.prod(inp_shape), dtype=jnp.float32).reshape(inp_shape)
    out_ty = jax.ShapeDtypeStruct(out_shape, jnp.float32)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_shape))

  @parameterized.named_parameters([
      ("packed", (4, 4, 4), (16, 4, 1), 1, 2, False),
      ("strided_end", (4, 4, 4, 4), (256, 64, 16, 4), 1, 2, False),
      ("strided_bot", (4, 4, 4, 4), (256, 16, 4, 1), 1, 2, False),
      ("strided_top", (4, 4, 4, 4), (256, 64, 4, 1), 1, 2, True),
      ("strided_mid", (4, 4, 4, 4), (265, 64, 16, 1), 1, 3, True),
      # TODO(cperivol): Investigate why this is indexing OOB and uncomment.
      # ("overap", (2, 4, 4), (16, 1, 1), 0, 3, True),
  ])
  def test_fold_strided(
      self, shape, strides, dim, fold_rank, throws_not_impl
  ):
    expanded_shape = get_packed_shape(strides, shape)
    total_size = np.prod(expanded_shape)
    np_inp = np.arange(total_size, dtype=jnp.float32).reshape(expanded_shape)
    index = tuple(slice(0, s) for s in shape)

    # Reference implementation
    def np_fold(inp, dim, fold_rank):
      out_shape = list(inp.shape)
      out_shape[dim : dim + fold_rank] = [
          int(np.prod(inp.shape[dim : dim + fold_rank]))
      ]
      if throws_not_impl:
        return jax.ShapeDtypeStruct(shape=out_shape, dtype=inp.dtype)
      else:
        return inp.reshape(*out_shape)

    total_size = np.prod(shape) * np.prod(strides)

    def do_test():
      def kernel(ctx, inp, out, _):
        copy(memref_fold(memref_slice(inp, index), dim, fold_rank), out)

      out = np_fold(np_inp[index], dim, fold_rank)
      y = mgpu.as_gpu_kernel(
          kernel, (1, 1, 1), (128, 1, 1), np_inp, out, ()
      )(np_inp)
      assert (
          not throws_not_impl
      ), "If it should have thrown it would during the call."
      np.testing.assert_array_equal(y, out)

    if throws_not_impl:
      with self.assertRaises(NotImplementedError):
        do_test()
    else:
      do_test()

  @parameterized.parameters(jnp.uint64, jnp.uint32, jnp.uint16, jnp.uint8)
  def test_scalar_argument(self, dtype):
    if dtype == jnp.uint64 and not config.enable_x64.value:
      self.skipTest(
        "64-bit types are disabled: this leads to the input scalar being"
        " traced as a uint32 value, which causes the top 32 bits of the 64-bit"
        " values read from the 32-bit input buffer to sometimes"
        " (nondeterministically) contain garbage.")

    scalar = dtype(42)
    expected = np.full((128, 128), scalar, dtype=dtype)

    def kernel(ctx, inp, out, _):
      del ctx
      inp = memref.load(inp, [])
      mgpu.FragmentedArray.splat(inp, expected.shape, is_signed=True).store_untiled(out)

    res = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        jax.ShapeDtypeStruct(shape=(), dtype=expected.dtype),
        expected,
        (),
    )(scalar)
    np.testing.assert_array_equal(res, expected)


def get_packed_shape(strides, shape):
  perm = sorted(range(len(strides)), key=lambda i: strides[i], reverse=True)
  ordered_strides = [strides[i] for i in perm]
  ordered_shape = [shape[i] for i in perm]
  packed_shape = [ordered_shape[-1]]
  packed_shape += [
      stride0 // stride
      for stride0, stride in zip(ordered_strides, ordered_strides[1:])
  ]
  # Invert permutation
  inv_perm = [None] * len(perm)
  for i, p in enumerate(perm):
    inv_perm[p] = i
  return [packed_shape[i] for i in inv_perm]


class WGMMALayoutTest(TestCase):

  @parameterized.product(dtype=[jnp.float16, jnp.float32])
  def test_store_untiled(self, dtype):
    def kernel(ctx, out, _):
      del ctx
      iota_tensor(64, 64, dtype).store_untiled(out, optimized=False)
    expected = np.arange(64 * 64, dtype=dtype).reshape(64, 64)
    iota = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, ()
    )()
    np.testing.assert_array_equal(iota, expected)

  @parameterized.product(
      dtype=[jnp.float32, jnp.float16, jnp.int8],
      swizzle=(32, 64, 128),
      num_col_tiles=(1, 2, 3),
  )
  def test_store_tiled(self, dtype, swizzle, num_col_tiles):
    mlir_dtype = utils.dtype_to_ir_type(dtype)
    if bytewidth(mlir_dtype) > 2 and swizzle == 32:
      self.skipTest("Not implemented")
    col_tiling = swizzle // bytewidth(mlir_dtype)
    m = 128
    n = col_tiling * num_col_tiles
    tiling = (64, col_tiling)
    def kernel(ctx, out, smem):
      del ctx
      iota_tensor(m, n, dtype).store_tiled(smem, swizzle=swizzle)
      copy(smem, out, swizzle=swizzle)
    expected = (
        np.arange(m * n, dtype=dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )
    iota = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, expected
    )()
    np.testing.assert_array_equal(iota, expected)

  @parameterized.parameters(jnp.int8, jnp.int16, jnp.int32)
  def test_sub_byte_conversion(self, jax_dtype_to):
    jax_dtype_from = jnp.int4
    def kernel(ctx, inp, out, smem):
      del ctx  # Unused.
      smem_inp, smem_out = smem
      copy(inp, smem_inp, swizzle=16)
      t = mgpu.FragmentedArray.load_tiled(smem_inp, is_signed=True, swizzle=16)
      t = t.astype(utils.dtype_to_ir_type(jax_dtype_to), is_signed=True)
      t.store_tiled(smem_out, swizzle=32 * jnp.dtype(jax_dtype_to).itemsize)
      copy(smem_out, out, swizzle=32 * jnp.dtype(jax_dtype_to).itemsize)

    x = self.prng.integers(
        low=-8, high=7, size=(1, 1, 64, 64), dtype=np.int32
    ).astype(jax_dtype_from)
    y = x.astype(jax_dtype_to)
    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, y, (x, y))
    np.testing.assert_array_equal(f(x), y)

  @parameterized.product(
      jax_dtype_from_to=(
          (jnp.int8, jnp.bfloat16),
          (jnp.int4, jnp.bfloat16),
      ),
      layout=(
          fa.WGMMA_LAYOUT,
          fa.WGMMA_LAYOUT_UPCAST_2X,
          fa.WGMMA_LAYOUT_UPCAST_4X,
      ),
  )
  def test_optimized_conversion(self, jax_dtype_from_to, layout):
    jax_dtype_from, jax_dtype_to = jax_dtype_from_to
    mlir_dtype_from = utils.dtype_to_ir_type(jax_dtype_from)
    mlir_dtype_to = utils.dtype_to_ir_type(jax_dtype_to)
    m = 128
    n = 256 * 8 // bitwidth(mlir_dtype_from)
    def kernel(ctx, inp, out, smem):
      del ctx
      smem_from, smem_to = smem
      copy(inp, smem_from, swizzle=128)
      t = mgpu.FragmentedArray.load_tiled(
          smem_from,
          swizzle=128,
          is_signed=utils.is_signed(jax_dtype_from),
          layout=layout,
      )
      t = t.astype(mlir_dtype_to, is_signed=utils.is_signed(jax_dtype_to))
      t.store_tiled(smem_to, swizzle=128)
      copy(smem_to, out, swizzle=128)

    from_tiling = (64, 128 * 8 // bitwidth(mlir_dtype_from))
    to_tiling = (64, 128 * 8 // bitwidth(mlir_dtype_to))
    # We only test lossless conversions for now.
    # TODO(apaszke): Test and fix failures that appear with lossy conversions.
    int_sample_dtype = getattr(
        jnp,
        "int" + str(min(bitwidth(mlir_dtype_from), bitwidth(mlir_dtype_to))),
    )
    sample_iinfo = jnp.iinfo(int_sample_dtype)
    expected_raw = self.prng.integers(
        low=sample_iinfo.min, high=sample_iinfo.max,
        size=(m, n), dtype=np.int32
    )
    expected = lambda jax_dtype, tiling: expected_raw.reshape(
        m // tiling[0], tiling[0], n // tiling[1], tiling[1]
    ).transpose(0, 2, 1, 3).astype(jax_dtype)

    expected_from = expected(jax_dtype_from, from_tiling)
    expected_to = expected(jax_dtype_to, to_tiling)
    res = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        expected_from,
        expected_to,
        (expected_from, expected_to),
    )(expected_from)
    np.testing.assert_array_equal(res, expected_to)

  @parameterized.named_parameters(
      ("f32", jnp.float32),
      ("f16", jnp.float16),
      ("i8", jnp.int8),
  )
  def test_load_tiled(self, jax_dtype):
    mlir_dtype = utils.dtype_to_ir_type(jax_dtype)
    m = 128
    n = 256 // bytewidth(mlir_dtype)
    tiling = (64, 128 // bytewidth(mlir_dtype))
    def kernel(ctx, in_, out, smem):
      del ctx
      smem1, smem2 = smem
      copy(in_, smem1, swizzle=128)
      t = mgpu.FragmentedArray.load_tiled(
          smem1, swizzle=128, is_signed=utils.is_signed(jax_dtype)
      )
      t.store_tiled(smem2, swizzle=128)
      copy(smem2, out, swizzle=128)
    expected = (
        np.arange(m * n, dtype=jax_dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )
    iota = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), expected, expected, (expected,) * 2
    )(expected)
    np.testing.assert_array_equal(iota, expected)


class I8Type:
  """A type that represents a 8-bit signed integer.

  This is a workaround to bypass the fact that we don't have a proper 8-bit
  integer type class available in MLIR, and can't instantiate types without a
  MLIR context.
  """

  @staticmethod
  def get():  # pylint: disable=no-method-argument
    return ir.IntegerType.get_signless(8)


class WGMMATest(TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Only works on GPU with capability sm90a")

  @parameterized.product(
      lhs_transpose=(False, True),
      rhs_transpose=(False, True),
      in_mlir_dtype_cls=(
          ir.F16Type,
          ir.BF16Type,
          ir.F32Type,
          ir.Float8E5M2Type,
          ir.Float8E4M3FNType,
      ),
      m=(64, 128, 192),
      n=(64, 128, 192),
      k_steps=(1, 2),
      swizzle=(32, 64, 128),
      jax_out_dtype=(jnp.float16, jnp.float32),
      rhs_tiling_kind=("large", "small", "small+no_transpose"),
      lhs_tiling_kind=("large", "small", "small+no_transpose"),
  )
  def test_wgmma_basic_float(
      self,
      lhs_transpose,
      rhs_transpose,
      in_mlir_dtype_cls,
      m,
      n,
      k_steps,
      swizzle,
      jax_out_dtype,
      rhs_tiling_kind,
      lhs_tiling_kind,
  ):
    self._test_wgmma_basic(
        m,
        n,
        k_steps,
        in_mlir_dtype_cls,
        lhs_transpose,
        rhs_transpose,
        swizzle,
        jax_out_dtype,
        rhs_tiling_kind,
        lhs_tiling_kind,
    )

  @parameterized.product(
      in_mlir_dtype_cls=(I8Type,),
      m=(64, 128, 192),
      n=(64, 128, 192),
      k_steps=(1, 2),
      swizzle=(32, 64, 128),
      jax_out_dtype=(jnp.int32,),
      rhs_tiling_kind=("large", "small", "small+no_transpose"),
      lhs_tiling_kind=("large", "small"),
  )
  def test_wgmma_basic_int(
      self,
      in_mlir_dtype_cls,
      m,
      n,
      k_steps,
      swizzle,
      jax_out_dtype,
      rhs_tiling_kind,
      lhs_tiling_kind,
  ):
    self._test_wgmma_basic(
        m,
        n,
        k_steps,
        in_mlir_dtype_cls,
        lhs_transpose=False,
        rhs_transpose=True,
        swizzle=swizzle,
        jax_out_dtype=jax_out_dtype,
        rhs_tiling_kind=rhs_tiling_kind,
        lhs_tiling_kind=lhs_tiling_kind,
    )

  def _test_wgmma_basic(
      self,
      m,
      n,
      k_steps,
      in_mlir_dtype_cls,
      lhs_transpose,
      rhs_transpose,
      swizzle,
      jax_out_dtype,
      rhs_tiling_kind,
      lhs_tiling_kind,
  ):
    if jax_out_dtype == jnp.int32 and in_mlir_dtype_cls != I8Type:
      self.skipTest("s32 accumulator only supported with s8 inputs")
    if jax_out_dtype != jnp.int32 and in_mlir_dtype_cls == I8Type:
      self.skipTest("s8 inputs only supported with s32 accumulator")
    if jax_out_dtype == jnp.float16 and in_mlir_dtype_cls in {ir.F32Type, ir.BF16Type}:
      self.skipTest(f"{in_mlir_dtype_cls.get()} does not support f16 output.")
    if swizzle != 128 and lhs_transpose and lhs_tiling_kind == "large":
      self.skipTest("Transpose only supported in 128B swizzled WGMMA")
    if rhs_tiling_kind == "small+no_transpose" and not rhs_transpose:
      self.skipTest("No transpose happening anyway")
    if lhs_tiling_kind == "small+no_transpose" and not lhs_transpose:
      self.skipTest("No transpose happening anyway")

    in_mlir_dtype = in_mlir_dtype_cls.get()
    out_mlir_dtype = utils.dtype_to_ir_type(jax_out_dtype)
    if (lhs_transpose or not rhs_transpose) and bytewidth(in_mlir_dtype) != 2:
      self.skipTest("Transpose only supported in 16-bit WGMMA")
    if ir.F32Type.isinstance(in_mlir_dtype):  # We actually use tf32 instead
      in_jax_dtype = jnp.float32
      exponent_bits, mantissa_bits = 8, 10  # Use tf32
    elif bytewidth(in_mlir_dtype) == 2:
      if n % 64 != 0:
        self.skipTest("16-bit WGMMA only supports n % 64 == 0")
      if ir.F16Type.isinstance(in_mlir_dtype):
        in_jax_dtype = jnp.float16
        exponent_bits, mantissa_bits = 5, 10
      elif ir.BF16Type.isinstance(in_mlir_dtype):
        in_jax_dtype = jnp.bfloat16
        exponent_bits, mantissa_bits = 8, 7
      else:
        raise NotImplementedError(in_mlir_dtype)
    elif in_mlir_dtype_cls == ir.Float8E5M2Type:
      in_jax_dtype = jnp.float8_e5m2
      exponent_bits, mantissa_bits = 5, 2
    elif in_mlir_dtype_cls == ir.Float8E4M3FNType:
      in_jax_dtype = jnp.float8_e4m3fn
      exponent_bits, mantissa_bits = 4, 3
    elif in_mlir_dtype_cls == I8Type:
      in_jax_dtype = jnp.int8
      exponent_bits = mantissa_bits = None
    else:
      raise NotImplementedError(in_mlir_dtype)
    nk_tile = swizzle // bytewidth(in_mlir_dtype)
    k = nk_tile * k_steps
    if n % nk_tile:
      self.skipTest("tiling does not divide N")
    assert m % 64 == 0 and n % nk_tile == 0

    small_rhs_tile = rhs_tiling_kind != "large"
    transpose_rhs_tiles = rhs_tiling_kind != "small+no_transpose"
    rhs_tiling = (8, nk_tile) if small_rhs_tile else (nk_tile, nk_tile)
    small_lhs_tile = lhs_tiling_kind != "large"
    transpose_lhs_tiles = lhs_tiling_kind != "small+no_transpose"
    lhs_tiling = (8, nk_tile) if small_lhs_tile else (64, nk_tile)

    def kernel(ctx, lhs, rhs, out, scratch):
      lhs_smem, rhs_smem, barriers = scratch
      lhs_transform = (mgpu.TileTransform(lhs_tiling),)
      if lhs_transpose and transpose_lhs_tiles:
        lhs_transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
      rhs_transform = (mgpu.TileTransform(rhs_tiling),)
      if rhs_transpose and transpose_rhs_tiles:
        rhs_transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
      ctx.async_copy(
          src_ref=lhs,
          dst_ref=lhs_smem,
          swizzle=swizzle,
          gmem_transform=lhs_transform,
          barrier=barriers[0],
      )
      ctx.async_copy(
          src_ref=rhs,
          dst_ref=rhs_smem,
          swizzle=swizzle,
          gmem_transform=rhs_transform,
          barrier=barriers[1],
      )
      for i in range(2):
        barriers[i].wait()
      is_signed = True if ir.IntegerType.isinstance(in_mlir_dtype) else None
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n, dtype=out_mlir_dtype, is_signed=is_signed)
      if lhs_transpose:
        perm = (0, 1, 3, 2) if transpose_lhs_tiles else (1, 0, 3, 2)
        lhs_smem = memref_transpose(lhs_smem, perm)
      if rhs_transpose:
        perm = (0, 1, 3, 2) if transpose_rhs_tiles else (1, 0, 3, 2)
        rhs_smem = memref_transpose(rhs_smem, perm)
      acc = mgpu.wgmma(init_acc, lhs_smem, rhs_smem, swizzle=swizzle)
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out, optimized=False)

    def quantize(x):
      # Quantize the input to avoid rounding when feeding the WGMMA
      return jax.lax.reduce_precision(x, exponent_bits, mantissa_bits)

    x_shape = (k, m) if lhs_transpose else (m, k)
    y_shape = (n, k) if rhs_transpose else (k, n)
    if in_mlir_dtype_cls == I8Type:
      x = self.prng.integers(-128, 127, x_shape).astype(in_jax_dtype)
      y = self.prng.integers(-128, 127, y_shape).astype(in_jax_dtype)
    else:
      x = quantize(self.prng.uniform(-1, 1, x_shape)).astype(in_jax_dtype)
      y = quantize(self.prng.uniform(-1, 1, y_shape)).astype(in_jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jax_out_dtype)
    if transpose_rhs_tiles:
      rhs_tiling_t = rhs_tiling[::-1] if rhs_transpose else rhs_tiling
      rhs_smem_shape = (k // rhs_tiling_t[0], n // rhs_tiling_t[1], *rhs_tiling)
    else:
      rhs_smem_shape = tile_shape(y_shape, rhs_tiling)
    if transpose_lhs_tiles:
      lhs_tiling_t = lhs_tiling[::-1] if lhs_transpose else lhs_tiling
      lhs_smem_shape = (m // lhs_tiling_t[0], k // lhs_tiling_t[1], *lhs_tiling)
    else:
      lhs_smem_shape = tile_shape(x_shape, lhs_tiling)
    scratch_shape = [
        jax.ShapeDtypeStruct(lhs_smem_shape, in_jax_dtype),
        jax.ShapeDtypeStruct(rhs_smem_shape, in_jax_dtype),
        mgpu.TMABarrier(2),
    ]
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (x, y), out_shape, scratch_shape
    )(x, y)
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    ref = (x32.T if lhs_transpose else x32) @ (y32.T if rhs_transpose else y32)
    atol = 2e-2 if jax_out_dtype == jnp.float16 else 5e-6
    if ir.IntegerType.isinstance(in_mlir_dtype) and ir.IntegerType.isinstance(out_mlir_dtype):
      atol = 0
    elif utils.bitwidth(in_mlir_dtype) == 8:
      atol = 3e-2
    np.testing.assert_allclose(z, ref, atol=atol)

  # TODO(apaszke): Add support for f32
  @parameterized.product(
      m=(64, 128, 192),
      n=(64, 128, 192),
      k_steps=(1, 2),
      rhs_transpose=(False, True),
      swizzle=(32, 64, 128),
      dtype=[jnp.float16, jnp.bfloat16],
  )
  def test_wgmma_reg_lhs(self, m, n, k_steps, rhs_transpose, swizzle, dtype):
    index = ir.IndexType.get()

    bytewidth = 2
    nk_tile = swizzle // bytewidth
    k = nk_tile * k_steps

    def kernel(ctx, rhs, out, rhs_smem):
      del ctx
      for ki in range(k_steps):
        for ni in range(n // nk_tile):
          rhs_slice = (
              ds(c(ki * nk_tile, index), nk_tile),
              ds(c(ni * nk_tile, index), nk_tile),
          )
          if rhs_transpose:
            rhs_slice = rhs_slice[::-1]
          copy(
              src=memref_slice(rhs, rhs_slice),
              dst=memref_slice(rhs_smem, (ki, ni)),
              swizzle=swizzle,
          )
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n)
      lhs_regs = iota_tensor(m, k, dtype)
      if rhs_transpose:
        rhs_smem = memref_transpose(rhs_smem, (0, 1, 3, 2))
      acc = mgpu.wgmma(init_acc, lhs_regs, rhs_smem, swizzle=swizzle)
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out, optimized=False)

    y_shape = (n, k) if rhs_transpose else (k, n)
    y = self.prng.uniform(-1, 1, y_shape).astype(dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    scratch_shape = jax.ShapeDtypeStruct(
        (k_steps, n // nk_tile, nk_tile, nk_tile), dtype
    )
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), y, out_shape, scratch_shape
    )(y)
    x = np.arange(m * k, dtype=dtype).reshape(m, k)
    ref = jax.lax.dot(
        x, (y.T if rhs_transpose else y), preferred_element_type=jnp.float32
    )
    rtol = 5e-4
    np.testing.assert_allclose(z, ref, rtol=rtol, atol=0)

  @parameterized.product(
      rhs_transpose=(False, True),
      swizzle=(32, 64, 128),
      n=(8, 16),
      small_rhs_tile=(False, True),
  )
  def test_narrow_n(self, rhs_transpose, swizzle, n, small_rhs_tile):
    m, k_steps = 64, 2
    bytewidth = 2
    nk_tile = swizzle // bytewidth
    k = nk_tile * k_steps
    if small_rhs_tile and not rhs_transpose:
      self.skipTest("Small tiles only supported for transposed RHS")

    n_tile = 8 if small_rhs_tile else nk_tile

    def kernel(ctx, rhs, out, smem):
      rhs_smem, barrier = smem
      gmem_slice = (ds(0, k), ds(0, max(n_tile, n)))
      transform = (mgpu.TileTransform((n_tile, nk_tile)),)
      if rhs_transpose:
        gmem_slice = gmem_slice[::-1]
        transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
      ctx.async_copy(
          src_ref=rhs,
          dst_ref=rhs_smem,
          swizzle=swizzle,
          gmem_slice=gmem_slice,
          gmem_transform=transform,
          barrier=barrier,
      )
      barrier.wait()
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n)
      lhs_regs = iota_tensor(m, k, jnp.float16)
      if rhs_transpose:
        rhs_smem = memref_transpose(rhs_smem, (0, 1, 3, 2))
      if not small_rhs_tile:
        smem_slice = (slice(None), slice(None), slice(None), ds(0, n))
        rhs_smem = memref_slice(rhs_smem, smem_slice)
      acc = mgpu.wgmma(init_acc, lhs_regs, rhs_smem, swizzle=swizzle)
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out, optimized=False)

    jax_dtype = jnp.float16
    y_shape = (n, k) if rhs_transpose else (k, n)
    y = self.prng.uniform(-1, 1, y_shape).astype(jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    rhs_scratch_shape = jax.ShapeDtypeStruct(
        (k_steps, (n + n_tile - 1) // n_tile, n_tile, nk_tile), jax_dtype
    )
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), y, out_shape, (rhs_scratch_shape, mgpu.TMABarrier()),
    )(y)
    x = np.arange(m * k, dtype=jax_dtype).reshape(m, k)
    ref = jax.lax.dot(
        x, (y.T if rhs_transpose else y), preferred_element_type=jnp.float32
    )
    np.testing.assert_allclose(z, ref, rtol=1e-3, atol=0)


class TCGen05Test(TestCase):

  def setUp(self):
    super().setUp()
    capabilities = ("10.0", "10.1")
    if not any(jtu.is_cuda_compute_capability_equal(sm) for sm in capabilities):
      self.skipTest("Only works on GPU with capability sm_100a or sm_101a")

  @parameterized.parameters([(jnp.float32, 1), (jnp.float16, 1), (jnp.float16, 2)])
  def test_load_store_tmem_swizzle(self, jax_dtype, packing):
    swizzle = 128
    in_mlir_dtype = utils.dtype_to_ir_type(jax_dtype)
    swizzle_elems = swizzle // bytewidth(in_mlir_dtype)
    tiling = (8, swizzle_elems)

    def kernel(ctx, input, output, scratch):
      smem, barrier, tmem = scratch
      ctx.async_copy(
          src_ref=input,
          dst_ref=smem,
          swizzle=swizzle,
          gmem_transform=mgpu.TileTransform(tiling),
          barrier=barrier,
      )
      barrier.wait()
      tmem.store(fa.FragmentedArray.load_tiled(smem, swizzle, layout=tcgen05.LAYOUT))
      tcgen05.commit_tmem()
      tmem.load().store_tiled(smem, swizzle)
      mgpu.commit_shared()
      ctx.async_copy(
          src_ref=smem, dst_ref=output, swizzle=swizzle, gmem_transform=mgpu.TileTransform(tiling),
      )
      ctx.await_async_copy(0)

    x = self.prng.uniform(-1, 1, (128, 128)).astype(jax_dtype)
    scratch_shape = [
        jax.ShapeDtypeStruct(tile_shape(x.shape, tiling), jax_dtype),
        mgpu.TMABarrier(),
        mgpu.TMEM(x.shape, jax_dtype, packing=packing),
    ]
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, scratch_shape
    )(x)
    np.testing.assert_array_equal(x, y)

  @parameterized.parameters([(jnp.float32, 1), (jnp.float16, 1), (jnp.float16, 2)])
  def test_load_store_tmem_native(self, jax_dtype, packing):

    def kernel(ctx, input, output, scratch):
      smem, barrier, tmem = scratch
      ctx.async_copy(src_ref=input, dst_ref=smem, barrier=barrier)
      barrier.wait()
      tmem.store(fa.FragmentedArray.load_untiled(smem, layout=tcgen05.TMEM_NATIVE_LAYOUT, optimized=False))
      tcgen05.commit_tmem()
      tmem.load(tcgen05.TMEM_NATIVE_LAYOUT).store_untiled(smem, optimized=False)
      mgpu.commit_shared()
      ctx.async_copy(src_ref=smem, dst_ref=output)
      ctx.await_async_copy(0)

    x = self.prng.uniform(-1, 1, (128, 128)).astype(jax_dtype)
    scratch_shape = [
        jax.ShapeDtypeStruct(x.shape, jax_dtype),
        mgpu.TMABarrier(),
        mgpu.TMEM(x.shape, jax_dtype, packing=packing),
    ]
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, scratch_shape
    )(x)
    np.testing.assert_array_equal(x, y)

  @parameterized.parameters([
      (jnp.float32, 1, "130.0000"),
      (jnp.float16, 1, "130.0000"),
      (jnp.float16, 2, "[132.000000,133.000000]"),
  ])
  @jtu.thread_unsafe_test()
  def test_tmem_debug_print(self, jax_dtype, packing, expected):
    swizzle = 128
    in_mlir_dtype = utils.dtype_to_ir_type(jax_dtype)
    swizzle_elems = swizzle // bytewidth(in_mlir_dtype)
    tiling = (8, swizzle_elems)

    def kernel(ctx, input, output, scratch):
      smem, barrier, tmem = scratch
      ctx.async_copy(
          src_ref=input,
          dst_ref=smem,
          swizzle=swizzle,
          gmem_transform=mgpu.TileTransform(tiling),
          barrier=barrier,
      )
      barrier.wait()
      tmem.store(fa.FragmentedArray.load_tiled(smem, swizzle, layout=tcgen05.LAYOUT))
      tcgen05.commit_tmem()
      tmem.slice(slice(None), slice(0, 8))._debug_print()

    x = jnp.arange(128 * 128, dtype=jax_dtype).reshape(128, 128)
    scratch_shape = [
        jax.ShapeDtypeStruct(tile_shape(x.shape, tiling), jax_dtype),
        mgpu.TMABarrier(),
        mgpu.TMEM(x.shape, jax_dtype, packing=packing),
    ]
    with self.capture_stdout() as stdout:
      mgpu.as_gpu_kernel(
          kernel, (1, 1, 1), (128, 1, 1), x, x, scratch_shape
      )(x).block_until_ready()
    self.assertIn("[1, 2]: " + expected, stdout())

  @parameterized.product(
      lhs_transpose=(False, True),
      rhs_transpose=(False, True),
      in_jax_dtype=(jnp.float16, jnp.bfloat16, jnp.float8_e5m2, jnp.float8_e4m3fn),  # TODO(apaszke): f32
      out_jax_dtype=(jnp.float16, jnp.float32,),
      m=(128,),  # TODO(apaszke): 64, 192, 256
      n=(64, 128, 256, 512),  # TODO(apaszke): 192, other non-power-of-2
      swizzle=(32, 64, 128,),
  )
  def test_mma_basic_float(self, **kwargs):
    if kwargs["n"] * jnp.dtype(kwargs["in_jax_dtype"]).itemsize < kwargs["swizzle"]:
      self.skipTest("swizzle too large for input")
    self._basic_mma_test(
        **kwargs,
        k_steps=2,  # Reducing to 1 can be helpful while debugging.
        lhs_transpose_tiles=False,
        rhs_transpose_tiles=False,
    )

  @parameterized.product(
      lhs_transpose=(False, True),
      rhs_transpose=(False, True),
      in_jax_dtype=(jnp.int8,),
      out_jax_dtype=(jnp.int32,),
      m=(128,),  # TODO(apaszke): 64, 192, 256
      n=(64, 128, 256, 512),  # TODO(apaszke): 192, other non-power-of-2
      swizzle=(32, 64, 128,),
  )
  def test_mma_basic_int(self, **kwargs):
    if kwargs["n"] * jnp.dtype(kwargs["in_jax_dtype"]).itemsize < kwargs["swizzle"]:
      self.skipTest("swizzle too large for input")
    self._basic_mma_test(
        **kwargs,
        k_steps=2,  # Reducing to 1 can be helpful while debugging.
        lhs_transpose_tiles=False,
        rhs_transpose_tiles=False,
    )

  @parameterized.product(
      lhs_transpose=(False, True),
      rhs_transpose=(False, True),
      in_jax_dtype=(jnp.float16,),
      out_jax_dtype=(jnp.float32,),
      m=(128,),
      n=(128, 512),
      swizzle=(32, 64, 128,),
      lhs_transpose_tiles=(False, True),
      rhs_transpose_tiles=(False, True),
  )
  def test_mma_transposed_tiles(self, **kwargs):
    if not kwargs["lhs_transpose_tiles"] and not kwargs["rhs_transpose_tiles"]:
      self.skipTest("This is already tested in test_mma_basic")
    self._basic_mma_test(
        **kwargs,
        k_steps=2,  # Reducing to 1 can be helpful while debugging.
    )

  def _basic_mma_test(
      self,
      m,
      n,
      k_steps,
      swizzle,
      lhs_transpose,
      rhs_transpose,
      in_jax_dtype,
      out_jax_dtype,
      rhs_transpose_tiles,
      lhs_transpose_tiles,
  ):
    if out_jax_dtype != jnp.float32 and (
        in_jax_dtype == jnp.float32 or in_jax_dtype == jnp.bfloat16
    ):
      self.skipTest("Only f32 output is supported for f32 and bf16 input.")

    in_mlir_dtype = utils.dtype_to_ir_type(in_jax_dtype)
    swizzle_elems = swizzle // bytewidth(in_mlir_dtype)
    k = swizzle_elems * k_steps
    lhs_tiling = rhs_tiling = (8, swizzle_elems)

    def kernel(ctx, lhs, rhs, out, scratch):
      lhs_smem, rhs_smem, barriers, acc = scratch
      lhs_transform = (mgpu.TileTransform(lhs_tiling),)
      if lhs_transpose_tiles:
        lhs_transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
      rhs_transform = (mgpu.TileTransform(rhs_tiling),)
      if rhs_transpose_tiles:
        rhs_transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
      ctx.async_copy(
          src_ref=lhs,
          dst_ref=lhs_smem,
          swizzle=swizzle,
          gmem_transform=lhs_transform,
          barrier=barriers[0],
      )
      ctx.async_copy(
          src_ref=rhs,
          dst_ref=rhs_smem,
          swizzle=swizzle,
          gmem_transform=rhs_transform,
          barrier=barriers[1],
      )
      barriers[0].wait()
      barriers[1].wait()
      with mgpu.single_thread():
        if lhs_transpose_tiles:
          lhs_smem = memref_transpose(lhs_smem, (1, 0, 2, 3))
        if lhs_transpose:
          lhs_smem = memref_transpose(lhs_smem, (1, 0, 3, 2))
        if rhs_transpose_tiles:
          rhs_smem = memref_transpose(rhs_smem, (1, 0, 2, 3))
        if rhs_transpose:
          rhs_smem = memref_transpose(rhs_smem, (1, 0, 3, 2))
        tcgen05.mma(
            acc, lhs_smem, rhs_smem, a_swizzle=swizzle, b_swizzle=swizzle, accumulate=False,
        )
        tcgen05.commit_arrive(barriers[2])
      barriers[2].wait(for_tensor_core=True)
      is_signed = True if jnp.issubdtype(in_jax_dtype, jnp.integer) else None
      acc.load(is_signed=is_signed).store_untiled(out, optimized=False)

    x_shape = (k, m) if lhs_transpose else (m, k)
    x = self.prng.uniform(-1, 1, x_shape).astype(in_jax_dtype)
    y_shape = (n, k) if rhs_transpose else (k, n)
    y = self.prng.uniform(-1, 1, y_shape).astype(in_jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), out_jax_dtype)
    if rhs_transpose_tiles:
      rhs_smem_shape = (
          y_shape[1] // rhs_tiling[1], y_shape[0] // rhs_tiling[0], *rhs_tiling,
      )
    else:
      rhs_smem_shape = tile_shape(y_shape, rhs_tiling)
    if lhs_transpose_tiles:
      lhs_smem_shape = (
          x_shape[1] // lhs_tiling[1], x_shape[0] // lhs_tiling[0], *lhs_tiling,
      )
    else:
      lhs_smem_shape = tile_shape(x_shape, lhs_tiling)
    scratch_shape = [
        jax.ShapeDtypeStruct(lhs_smem_shape, in_jax_dtype),
        jax.ShapeDtypeStruct(rhs_smem_shape, in_jax_dtype),
        mgpu.TMABarrier(3),
        mgpu.TMEM((128, n), out_jax_dtype),
    ]
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (x, y), out_shape, scratch_shape
    )(x, y)
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    ref = (x32.T if lhs_transpose else x32) @ (y32.T if rhs_transpose else y32)
    atol = 2e-2 if out_jax_dtype == jnp.float16 else 2e-5
    rtol = 8e-4 if out_jax_dtype == jnp.float16 else 1e-7
    np.testing.assert_allclose(z, ref, atol=atol, rtol=rtol)

  @parameterized.product(
      in_jax_dtype=(jnp.float16, jnp.bfloat16),  # TODO(apaszke): f32
      out_jax_dtype=(jnp.float16, jnp.float32,),
      m=(128,),  # TODO(apaszke): 64, 192, 256
      n=(64, 128, 256),  # TODO(apaszke): 192, other non-power-of-2
  )
  def test_mma_lhs_tmem(self, m, n, in_jax_dtype, out_jax_dtype):
    swizzle = 128
    k_steps = 2  # Reducing to 1 can be helpful while debugging.
    if out_jax_dtype == jnp.float16 and in_jax_dtype != jnp.float16:
      self.skipTest("Only f16 input is supported for f16 output.")

    in_mlir_dtype = utils.dtype_to_ir_type(in_jax_dtype)
    swizzle_elems = swizzle // bytewidth(in_mlir_dtype)
    k = swizzle_elems * k_steps
    lhs_tiling = rhs_tiling = (8, swizzle_elems)

    def kernel(ctx, lhs, rhs, out, scratch):
      lhs_smem, rhs_smem, barriers, acc, lhs_tmem = scratch
      ctx.async_copy(
          src_ref=lhs,
          dst_ref=lhs_smem,
          swizzle=swizzle,
          gmem_transform=mgpu.TileTransform(lhs_tiling),
          barrier=barriers[0],
      )
      ctx.async_copy(
          src_ref=rhs,
          dst_ref=rhs_smem,
          swizzle=swizzle,
          gmem_transform=mgpu.TileTransform(rhs_tiling),
          barrier=barriers[1],
      )
      barriers[0].wait()
      barriers[1].wait()
      lhs_tmem.store(
          fa.FragmentedArray.load_tiled(
              lhs_smem, swizzle, layout=tcgen05.LAYOUT
          )
      )
      tcgen05.commit_tmem()
      with mgpu.single_thread():
        tcgen05.mma(
            acc, lhs_tmem, rhs_smem, a_swizzle=swizzle, b_swizzle=swizzle, accumulate=False,
        )
        tcgen05.commit_arrive(barriers[2])
      barriers[2].wait(for_tensor_core=True)
      acc.load().store_untiled(out, optimized=False)

    x_shape = (m, k)
    x = self.prng.uniform(-1, 1, x_shape).astype(in_jax_dtype)
    y_shape = (k, n)
    y = self.prng.uniform(-1, 1, y_shape).astype(in_jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), out_jax_dtype)
    scratch_shape = [
        jax.ShapeDtypeStruct(tile_shape(x_shape, lhs_tiling), in_jax_dtype),
        jax.ShapeDtypeStruct(tile_shape(y_shape, rhs_tiling), in_jax_dtype),
        mgpu.TMABarrier(3),
        mgpu.TMEM((128, n), out_jax_dtype),
        mgpu.TMEM((128, k), in_jax_dtype, packing=2),
    ]
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (x, y), out_shape, scratch_shape
    )(x, y)
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    ref = x32 @ y32
    atol = 2e-2 if out_jax_dtype == jnp.float16 else 2e-5
    rtol = 8e-4 if out_jax_dtype == jnp.float16 else 1e-7
    np.testing.assert_allclose(z, ref, atol=atol, rtol=rtol)

  @parameterized.product(
      lhs_transpose=(False, True),
      rhs_transpose=(False, True),
      in_jax_dtype=(jnp.float16,),
      out_jax_dtype=(jnp.float32,),
      m=(256,),  # TODO(apaszke): 64, 192, 256
      n=(128, 256, 512),  # TODO(apaszke): 192, other non-power-of-2
      k_steps=(1, 2),
      swizzle=(32, 64, 128,),
  )
  def test_mma_collective(
      self,
      m,
      n,
      k_steps,
      swizzle,
      lhs_transpose,
      rhs_transpose,
      in_jax_dtype,
      out_jax_dtype,
  ):
    if out_jax_dtype == jnp.float16 and in_jax_dtype != jnp.float16:
      raise self.skipTest("Only f16 input is supported for f16 output.")

    in_mlir_dtype = utils.dtype_to_ir_type(in_jax_dtype)
    m_block_tile = m // 2
    n_block_tile = n // 2
    swizzle_elems = swizzle // bytewidth(in_mlir_dtype)
    k = swizzle_elems * k_steps
    index = ir.IndexType.get()

    tiling = (8, swizzle_elems)

    def kernel(ctx, lhs, rhs, out, scratch):
      lhs_smem, rhs_smem, barriers, acc = scratch
      block_id = gpu.cluster_block_id(gpu.Dimension.x)
      ctx.async_copy(
          src_ref=lhs,
          dst_ref=lhs_smem,
          swizzle=swizzle,
          gmem_transform=mgpu.TileTransform(tiling),
          barrier=barriers[0],
          collective=gpu.Dimension.x,
          partitioned=1 if lhs_transpose else 0,  # Split non-contracting dim.
      )
      ctx.async_copy(
          src_ref=rhs,
          dst_ref=rhs_smem,
          swizzle=swizzle,
          gmem_transform=mgpu.TileTransform(tiling),
          barrier=barriers[1],
          collective=gpu.Dimension.x,
          partitioned=0 if rhs_transpose else 1,  # Split non-contracting dim.
      )
      is_leader_thread = single_thread_predicate()
      is_first_block = arith.cmpi(arith.CmpIPredicate.eq, block_id, c(0, index))
      with when(arith.andi(is_first_block, is_leader_thread)):
        barriers[0].wait()
        barriers[1].wait()
        if lhs_transpose:
          lhs_smem = memref_transpose(lhs_smem, (1, 0, 3, 2))
        if rhs_transpose:
          rhs_smem = memref_transpose(rhs_smem, (1, 0, 3, 2))
        tcgen05.mma(
            acc, lhs_smem, rhs_smem, a_swizzle=swizzle, b_swizzle=swizzle, accumulate=False, collective=True
        )
        tcgen05.commit_arrive(barriers[2], collective=True, ctx=ctx)
      barriers[2].wait(for_tensor_core=True)
      m_slice = ds(arith.muli(block_id, c(m_block_tile, index)), m_block_tile)
      acc.load().store_untiled(memref_slice(out, m_slice), optimized=False)

    in_finfo = jnp.finfo(in_jax_dtype)
    exponent_bits, mantissa_bits = in_finfo.nexp, in_finfo.nmant
    def quantize(x):
      # Quantize the input to avoid rounding when feeding the TensorCore
      return jax.lax.reduce_precision(x, exponent_bits, mantissa_bits)

    x_shape = (k, m) if lhs_transpose else (m, k)
    x_block_shape = (k, m_block_tile) if lhs_transpose else (m_block_tile, k)
    x = quantize(self.prng.uniform(-1, 1, x_shape)).astype(in_jax_dtype)
    y_shape = (n, k) if rhs_transpose else (k, n)
    y_block_shape = (n_block_tile, k) if rhs_transpose else (k, n_block_tile)
    y = quantize(self.prng.uniform(-1, 1, y_shape)).astype(in_jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), out_jax_dtype)
    tmem_layout = tcgen05.TMEM_COLLECTIVE_N512_LAYOUT if n == 512 else None
    scratch_shape = [
        jax.ShapeDtypeStruct(tile_shape(x_block_shape, tiling), in_jax_dtype),
        jax.ShapeDtypeStruct(tile_shape(y_block_shape, tiling), in_jax_dtype),
        mgpu.TMABarrier(3),
        mgpu.TMEM((128, n), out_jax_dtype, collective=True, layout=tmem_layout),
    ]
    z = mgpu.as_gpu_kernel(
        kernel, (2, 1, 1), (128, 1, 1), (x, y), out_shape, scratch_shape, cluster=(2, 1, 1)
    )(x, y)
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    ref = (x32.T if lhs_transpose else x32) @ (y32.T if rhs_transpose else y32)
    atol = 2e-2 if out_jax_dtype == jnp.float16 else 5e-6
    np.testing.assert_allclose(z, ref, atol=atol)


class BarrierTest(TestCase):

  def test_wg_communication(self):
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, dst, scratch):
      tmp, barriers = scratch
      del ctx  # Unused.
      wg_idx = arith.divui(mgpu.warp_idx(), c(4, i32))
      is_first_wg = arith.cmpi(arith.CmpIPredicate.eq, wg_idx, c(0, i32))
      is_second_wg = arith.cmpi(arith.CmpIPredicate.eq, wg_idx, c(1, i32))
      arr = mgpu.FragmentedArray.splat(
          arith.addi(wg_idx, c(1, i32)),
          (128,),
          mgpu.WGStridedFragLayout((128,), 1),
          is_signed=False,
      )
      with ir.InsertionPoint(scf.IfOp(is_first_wg).then_block):
        arr.store_untiled(tmp)
        barriers[0].arrive()  # Signal that tmp is ready.
        barriers[1].wait()  # Wait for the other warp to produce tmp.
        final_arr = arr + mgpu.FragmentedArray.load_strided(
            tmp, is_signed=False
        )
        final_arr.store_untiled(memref_slice(dst, 0), optimized=False)
        scf.yield_([])
      with ir.InsertionPoint(scf.IfOp(is_second_wg).then_block):
        barriers[0].wait()
        final_arr = arr + mgpu.FragmentedArray.load_strided(
            tmp, is_signed=False
        )
        barriers[2].arrive()
        barriers[2].wait()  # Synchronize this warpgroup before we overwrite tmp.
        arr.store_untiled(tmp)
        barriers[1].arrive()  # Signal that tmp is ready.
        final_arr.store_untiled(memref_slice(dst, 1), optimized=False)
        scf.yield_([])
    out_shape = jax.ShapeDtypeStruct((2, 128), jnp.int32)
    y = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (2 * 128, 1, 1),
        (),
        out_shape,
        (
            jax.ShapeDtypeStruct((128,), jnp.int32),
            mgpu.Barrier(arrival_count=128, num_barriers=3),
        ),
    )()
    np.testing.assert_array_equal(y, np.full_like(y, 3, dtype=np.int32))

  @parameterized.named_parameters(
      (
          f"_{''.join(map(str, collective_dims))}={collective_size}{'_' + ''.join(map(str, noncollective_dims)) if noncollective_dims else ''}{'_group' if group_dims else ''}",
          collective_dims,
          noncollective_dims,
          collective_size,
          group_dims,
      )
      for collective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(1, 4)
      )
      for noncollective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(3)
      )
      for collective_size in (1, 2, 4)
      for group_dims in (False,) + ((True,) if len(collective_dims) > 1 else ())
      if all(d not in noncollective_dims for d in collective_dims)
  )
  def test_collective_arrive(self, collective_dims, noncollective_dims, collective_size, group_dims):
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    cluster = [1, 1, 1]
    for d in collective_dims:
      cluster[d] = collective_size
    for d in noncollective_dims:
      cluster[d] = 2
    if math.prod(cluster) > 16:
      self.skipTest("Cluster too big")
    is_trivial = math.prod(cluster[d] for d in collective_dims) == 1
    def kernel(ctx, dst, mask, collective_barrier):
      cluster_idx = ctx.cluster_idx()
      if not is_trivial:
        memref.store(collective_barrier.cluster_mask, mask, [cluster_idx])
      else:
        assert collective_barrier.cluster_mask is None
      collective_barrier.arrive()
      collective_barrier.wait()
      tid = thread_idx()
      linear_idx = arith.index_cast(index, tid)
      stride = c(128, index)
      for d in gpu.Dimension:
        linear_idx = arith.addi(linear_idx, arith.muli(gpu.block_id(d), stride))
        stride = arith.muli(stride, gpu.grid_dim(d))
      memref.store(arith.index_cast(i32, linear_idx), dst, [linear_idx])
    out_shape = jax.ShapeDtypeStruct((math.prod(cluster) * 128,), jnp.int32)
    mask_shape = jax.ShapeDtypeStruct((math.prod(cluster),), jnp.int32)
    barrier_dims = collective_dims
    if group_dims:
      barrier_dims = (collective_dims[:2], *collective_dims[2:])
    scratch = mgpu.ClusterBarrier(barrier_dims)
    y, mask = mgpu.as_gpu_kernel(
        kernel, cluster, (128, 1, 1), (), (out_shape, mask_shape), scratch, cluster=cluster,
    )()
    np.testing.assert_array_equal(
        y, np.arange(math.prod(cluster) * 128, dtype=np.int32)
    )
    if not is_trivial:
      # Verify that the mask is correct. Blocks are column-major, hence the transpose.
      block_bits = 1 << np.arange(math.prod(cluster), dtype=np.int32).reshape(cluster[::-1]).T
      expected_mask = 0
      for bd in barrier_dims:
        if isinstance(bd, gpu.Dimension):
          bd = (bd,)
        least_significant_slice = tuple(
            slice(None) if d in bd else 0 for d in gpu.Dimension
        )
        mask_bits = block_bits[least_significant_slice]
        expected_mask |= np.bitwise_or.reduce(mask_bits, axis=None)
      self.assertEqual(min(mask), expected_mask)


class TMATest(TestCase):

  @parameterized.product(
      swizzle=(None, 32, 64, 128),
      shape=((64, None), (5, None), (2, 3, 5, None)),
      dtype=(jnp.float32, jnp.float16, jnp.int4),
  )
  def test_tma_load_basic(self, swizzle, shape, dtype):
    bw = bitwidth(dtype_to_ir_type(dtype))
    minor_size = 64 if swizzle is None else 8 * swizzle // bw
    shape = (*shape[:-1], minor_size)
    i1 = ir.IntegerType.get_signless(1)
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      ctx.async_copy(src_ref=src, dst_ref=tmp, swizzle=swizzle, barrier=barrier)
      barrier.wait_parity(c(0, i1))
      copy(tmp, dst, swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem = (x, mgpu.TMABarrier())
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, smem)(x)
    np.testing.assert_array_equal(y, x)

  def test_tma_with_1d_tiling(self):
    swizzle = 128
    dtype = jnp.float16
    shape = (64, 128)
    tiling = (1, swizzle // jnp.dtype(dtype).itemsize)
    def kernel(ctx, dst, smem):
      iota_tensor(*shape, dtype=dtype).store_tiled(smem, swizzle=swizzle)
      ctx.async_copy(
          src_ref=smem,
          dst_ref=dst,
          swizzle=swizzle,
          gmem_transform=mgpu.TileTransform(tiling),
      )
      ctx.await_async_copy(0)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem = jax.ShapeDtypeStruct(utils.tile_shape(shape, tiling), dtype)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), (), x, smem)()
    np.testing.assert_array_equal(y, x)

  @parameterized.named_parameters(
      (
          f"_{''.join(map(str, collective_dims))}={collective_size}{'_' + ''.join(map(str, noncollective_dims)) if noncollective_dims else ''}",
          collective_dims,
          noncollective_dims,
          collective_size,
      )
      for collective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(1, 4)
      )
      for noncollective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(3)
      )
      for collective_size in (1, 2, 4)
      if all(d not in noncollective_dims for d in collective_dims)
  )
  def test_tma_load_multicast(self, collective_dims, noncollective_dims, collective_dim_size):
    index = ir.IndexType.get()
    swizzle = 128
    dtype = jnp.float16
    cluster = [1, 1, 1]
    for d in collective_dims:
      cluster[d] = collective_dim_size
    for d in noncollective_dims:
      cluster[d] = 2
    if math.prod(cluster) > 16:
      self.skipTest("Cluster too big")
    collective_size = math.prod(cluster[d] for d in collective_dims)
    noncollective_size = math.prod(cluster) // collective_size
    # We use the 2 dimension to exercise splitting the collective over
    # multiple dimensions when the cluster is large.
    shape = (noncollective_size, 2, 16 * collective_size, 64)
    minor_size = 64 if swizzle is None else swizzle // jnp.dtype(dtype).itemsize
    shape = (*shape[:-1], minor_size)
    # Note that this kernel does not use the non-collective dimensions in any
    # interesting way and so they don't really have to be part of the cluster.
    # We use them to test that the multicast mask is generated correctly.
    def kernel(ctx, src, dst, scratch):
      tmp, barrier = scratch
      stride = 1
      noncollective_idx = c(0, index)
      for d in noncollective_dims:
        noncollective_idx = arith.addi(
            noncollective_idx,
            arith.muli(gpu.cluster_block_id(d), c(stride, index))
        )
        stride *= cluster[d]
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          gmem_slice=(noncollective_idx,),
          swizzle=swizzle,
          barrier=barrier,
          collective=collective_dims,
      )
      barrier.wait()
      # This is _not_ the real cluster block idx, because it does not consider
      # the column-major ordering of the grid dimensions.
      idx = c(0, index)
      stride = 1
      for d in collective_dims:
        idx = arith.addi(
            idx, arith.muli(gpu.cluster_block_id(d), c(stride, index))
        )
        stride *= cluster[d]
      idx_minor = arith.divui(idx, c(2, index))
      idx_major = arith.remui(idx, c(2, index))
      slc_minor = ds(
          arith.muli(idx_minor, c(16 * 2, index)), 16 * 2
      )
      copy(
          memref_slice(tmp, (idx_major, slc_minor)),
          memref_slice(dst, (noncollective_idx, idx_major, slc_minor)),
          swizzle=swizzle,
      )
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem_shape = (jax.ShapeDtypeStruct(shape[1:], dtype), mgpu.TMABarrier())
    y = mgpu.as_gpu_kernel(
        kernel, cluster, (128, 1, 1), x, x, smem_shape, cluster=cluster
    )(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 128),
      shape=((128, 128), (5, 32, 128)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_load_tiled(self, swizzle, shape, dtype):
    # TODO(apaszke): ptxas seems to freeze when generating code for copy with
    # swizzle 32 and 64.
    i1 = ir.IntegerType.get_signless(1)
    index = ir.IndexType.get()
    tiling = (32, (swizzle or 128) // jnp.dtype(dtype).itemsize)
    tiled_shape = tile_shape(shape, tiling)[:len(shape)]
    def kernel(ctx, src, dst, scratch):
      tmp, barrier = scratch
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          swizzle=swizzle,
          barrier=barrier,
          gmem_transform=mgpu.TileTransform(tiling),
      )
      barrier.wait_parity(c(0, i1))
      for idxs in np.ndindex(tiled_shape):
        untiled_idxs, tiled_idxs = idxs[:-len(tiling)], idxs[-len(tiling):]
        s = (
            *untiled_idxs,
            *(ds(c(ix * t, index), t) for ix, t in zip(tiled_idxs, tiling)),
        )
        copy(memref_slice(tmp, idxs), memref_slice(dst, s), swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem = (
        jax.ShapeDtypeStruct(tile_shape(shape, tiling), dtype),
        mgpu.TMABarrier(),
    )
    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, smem)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(swizzle=(None, 128))
  def test_tma_load_tiled_rounding(self, swizzle):
    # TODO(apaszke): ptxas seems to freeze when generating code for copy with
    # swizzle 32 and 64.
    shape = (5, 32, 144)
    dtype = jnp.float16
    i1 = ir.IntegerType.get_signless(1)
    index = ir.IndexType.get()
    tiling = (32, (swizzle or 128) // jnp.dtype(dtype).itemsize)
    rounded_shape = (*shape[:-1], shape[-1] // tiling[-1] * tiling[-1])
    tiled_shape = tile_shape(rounded_shape, tiling)[:len(shape)]
    def kernel(ctx, src, dst, scratch):
      tmp, barrier = scratch
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          swizzle=swizzle,
          barrier=barrier,
          gmem_transform=mgpu.TileTransform(
              tiling, rounding=mgpu.Rounding.DOWN
          ),
      )
      barrier.wait_parity(c(0, i1))
      for idxs in np.ndindex(tiled_shape):
        untiled_idxs, tiled_idxs = idxs[:-len(tiling)], idxs[-len(tiling):]
        s = (
            *untiled_idxs,
            *(ds(c(ix * t, index), t) for ix, t in zip(tiled_idxs, tiling)),
        )
        copy(memref_slice(tmp, idxs), memref_slice(dst, s), swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    tmp_shape = jax.ShapeDtypeStruct(tile_shape(rounded_shape, tiling), dtype)
    smem = (tmp_shape, mgpu.TMABarrier())
    out_shape = jax.ShapeDtypeStruct(rounded_shape, dtype)
    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, out_shape, smem)
    y = f(x)
    np.testing.assert_array_equal(y, x[..., :rounded_shape[-1]])

  def test_tma_load_indexed_tiled(self):
    shape = (128, 2, 128)
    tiling = mgpu.TileTransform((32, 32))
    def kernel(ctx, src, dst, scratch):
      tmp, barrier = scratch
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          barrier=barrier,
          gmem_transform=tiling,
          gmem_slice=(slice(None), 1, slice(None)),
      )
      barrier.wait()
      ctx.async_copy(src_ref=tmp, dst_ref=dst, gmem_transform=tiling)
      ctx.await_async_copy(0)
    x = np.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    smem = (
        jax.ShapeDtypeStruct((4, 4, 32, 32), jnp.float32),
        mgpu.TMABarrier(),
    )
    out_shape = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, out_shape, smem)
    np.testing.assert_array_equal(f(x), x[:, 1, :])

  @parameterized.product(
      swizzle=(None, 128),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_squeeze_indexing(self, swizzle, dtype):
    # TODO(apaszke): ptxas seems to freeze when generating code for copy with
    # swizzle 32 and 64.
    minor_size = 64 if swizzle is None else swizzle // jnp.dtype(dtype).itemsize
    shape = (4, 5, minor_size)
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      for i in range(4):
        ctx.async_copy(
            src_ref=src,
            dst_ref=memref_slice(tmp, i),
            gmem_slice=i,
            swizzle=swizzle,
            barrier=barrier,
        )
        barrier.wait()
      copy(tmp, dst, swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem = (x, mgpu.TMABarrier())
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, smem)(x)
    np.testing.assert_array_equal(y, x)

  def test_parity_tracking(self):
    shape = (16, 64)
    index = ir.IndexType.get()
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      for i in range(shape[0]):
        s = ds(c(i, index), 1)
        ctx.async_copy(
            src_ref=src, dst_ref=tmp, gmem_slice=s, barrier=barrier,
        )
        barrier.wait()
        copy(tmp, memref_slice(dst, s))
    x = np.arange(np.prod(shape), dtype=jnp.float16).reshape(shape)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, (x[0:1], mgpu.TMABarrier())
    )(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 32, 64, 128),
      shape=((64, None), (5, None), (2, 3, 5, None)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_store(self, swizzle, shape, dtype):
    minor_size = 64 if swizzle is None else swizzle // jnp.dtype(dtype).itemsize
    shape = (*shape[:-1], minor_size)
    def kernel(ctx, src, dst, tmp):
      copy(src, tmp, swizzle=swizzle)
      ctx.async_copy(src_ref=tmp, dst_ref=dst, swizzle=swizzle)
      ctx.await_async_copy(0)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.parameters(0, 1)
  def test_tma_small_tile_load(self, small_dim):
    if small_dim == 0:
      shape = (4, 128)
    elif small_dim == 1:
      shape = (128, 8)
    else:
      raise ValueError("small_dim must be 0 or 1")
    tiled_shape = ((shape[0] + 63) // 64, (shape[1] + 63) // 64, 64, 64)
    padded_shape = (math.prod(tiled_shape[0::2]), math.prod(tiled_shape[1::2]))
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          swizzle=128,
          gmem_transform=mgpu.TileTransform((64, 64)),
          gmem_slice=(ds(0, padded_shape[0]), ds(0, padded_shape[1])),
          barrier=barrier,
      )
      barrier.wait()
      copy(tmp, dst, swizzle=128)
    x = np.arange(np.prod(shape), dtype=jnp.float16).reshape(shape)
    tiled = jax.ShapeDtypeStruct(tiled_shape, jnp.float16)
    y_tiled = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, tiled, (tiled, mgpu.TMABarrier()),
    )(x)
    y = y_tiled.swapaxes(1, 2).reshape(padded_shape)
    # y should contain x and zero everywhere else.
    np.testing.assert_array_equal(y[:shape[0], :shape[1]], x)
    y_mut = np.asarray(y).copy()
    y_mut[:shape[0], :shape[1]] = 0
    np.testing.assert_array_equal(y_mut, np.zeros_like(y_mut))

  @parameterized.product(small_dim=(0, 1),)
  def test_tma_small_tile_store(self, small_dim):
    if small_dim == 0:
      shape = (4, 128)
    elif small_dim == 1:
      shape = (128, 8)
    else:
      raise ValueError("small_dim must be 0 or 1")
    tiled_shape = ((shape[0] + 63) // 64, (shape[1] + 63) // 64, 64, 64)
    m, n = (math.prod(tiled_shape[0::2]), math.prod(tiled_shape[1::2]))
    def kernel(ctx, dst, tmp):
      vals = iota_tensor(m, n, jnp.float16)
      vals.store_tiled(tmp, swizzle=128)
      ctx.async_copy(
          src_ref=tmp,
          dst_ref=dst,
          swizzle=128,
          gmem_transform=mgpu.TileTransform((64, 64)),
          gmem_slice=(ds(0, m), ds(0, n)),
      )
      ctx.await_async_copy(0)
    tiled = jax.ShapeDtypeStruct(tiled_shape, jnp.float16)
    out = jax.ShapeDtypeStruct(shape, jnp.float16)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out, tiled,
    )()
    iota = np.arange(m * n, dtype=jnp.float16).reshape([m, n])
    np.testing.assert_array_equal(y, iota[:shape[0], :shape[1]])

  def test_tma_invalid(self):
    def kernel(ctx, src, dst, tmp):
      copy(src, tmp)
      ctx.async_copy(src_ref=tmp, dst_ref=dst)
      ctx.await_async_copy(0)

    def run_kernel(shape):
      x = np.arange(np.prod(shape)).reshape(shape)
      _ = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)

    with self.assertRaisesRegex(ValueError, "all GMEM strides except the last"):
      run_kernel([1] * 6)

    with self.assertRaisesRegex(
        ValueError, "last dimension to be divisible by 128"
    ):
      run_kernel([23])


class FragmentedArrayTest(TestCase):

  @parameterized.product(
      op=(
          operator.add,
          operator.mul,
          operator.sub,
          (lambda x, y: mgpu.FragmentedArray.min(x, y), np.minimum),
          (lambda x, y: mgpu.FragmentedArray.max(x, y), np.maximum),
      ),
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
      m=(64, 128),
      n=(8, 16, 32, 64, 80, 128, 256),
  )
  @jtu.ignore_warning(
      message="(invalid value|divide by zero)", category=RuntimeWarning
  )
  def test_binary(self, op, dtype, m=64, n=32):
    if isinstance(op, tuple):
      op, np_op = op
    else:
      np_op = op

    for scalar_rhs in [None, 2]:
      def kernel(ctx, dst, _):
        mlir_dtype = utils.dtype_to_ir_type(dtype)
        iota = iota_tensor(m, n, dtype)
        rhs = iota if scalar_rhs is None else c(scalar_rhs, mlir_dtype)
        op(iota, rhs).store_untiled(dst, optimized=False)
      out_shape = jax.ShapeDtypeStruct((m, n), dtype)
      result = mgpu.as_gpu_kernel(
          kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
      )()
      ref_x = np.arange(m * n, dtype=dtype).reshape(m, n)
      ref_rhs = scalar_rhs or ref_x
      np.testing.assert_array_equal(result, np_op(ref_x, ref_rhs))

  def test_minimum_np_compatibility(self):
    one = np.ones((128, 128)).astype(np.float32)
    negz = one * -0.
    posz = one * 0.
    nan = one * np.nan
    expectation = (np.minimum(negz, posz) == negz) & (np.minimum(nan, one) != one)
    assert np.all(expectation), expectation

    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      splat = lambda i: mgpu.FragmentedArray.splat(c(i, f32), (128, 128))
      negz = splat(-0.)
      posz = splat(0.)
      nan = splat(np.nan)
      one = splat(1.)
      res = (negz.min(posz) == negz) & (one.min(nan) != one) & (nan.min(one) != one)
      i8 = ir.IntegerType.get_signless(8)
      res.astype(i8, is_signed=False).store_untiled(dst)

    out_shape = jax.ShapeDtypeStruct((128, 128), np.int8)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    np.testing.assert_array_equal(result == 1, expectation)

  @parameterized.product(
      op=[operator.truediv, operator.floordiv, operator.mod],
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
  )
  def test_division(self, op, dtype, m=64, n=32):
    if jnp.issubdtype(dtype, jnp.integer) and op is operator.truediv:
      self.skipTest("Unsupported for integer types")
    if jnp.issubdtype(dtype, jnp.floating) and op is operator.mod:
      self.skipTest("Unsupported for floating types")

    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, dtype)
      op(dtype(4.2).item() * iota, iota + 1).store_untiled(dst, optimized=False)

    out_shape = jax.ShapeDtypeStruct((m, n), dtype)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    iota = np.arange(m * n, dtype=dtype).reshape(m, n)
    np.testing.assert_allclose(
        result, op(dtype(4.2).item() * iota, iota + 1), atol=2e-7
    )

  @parameterized.product(
      op=[
          operator.lt,
          operator.le,
          operator.gt,
          operator.ge,
          operator.eq,
          operator.ne,
      ],
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
      rhs_is_literal=[False, True]
  )
  def test_comparison(self, op, dtype, rhs_is_literal, m=64, n=32):
    def kernel(ctx, dst, _):
      i8 = ir.IntegerType.get_signless(8)
      iota = iota_tensor(m, n, dtype)
      rhs = 0 if rhs_is_literal else iota + 1
      res = op(iota, rhs)
      assert not res.is_signed
      res.astype(i8, is_signed=False).store_untiled(dst, optimized=False)

    out_shape = jax.ShapeDtypeStruct((m, n), jnp.int8)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    iota = np.arange(m * n, dtype=dtype).reshape(m, n)
    rhs = 0 if rhs_is_literal else iota + 1
    np.testing.assert_array_equal(result, op(iota, rhs).astype(jnp.int8))

  def test_foreach_wgmma_row_array(self):
    def kernel(ctx, out, smem):
      del ctx, smem
      x = iota_tensor(128, 128, jnp.float32)
      row = x.reduce("add", 1)
      # Test returning an array
      row = row.foreach(
          lambda x, _: arith.addf(x, c(1, row.mlir_dtype)), create_array=True
      )
      # Test no array return
      @row.foreach
      def _(v, idx):
        memref.store(v, out, idx)

    result = mgpu.as_gpu_kernel(
        kernel,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(),
        out_shape=jax.ShapeDtypeStruct(shape=(128,), dtype=jnp.float32),
        smem_scratch_shape=(),
    )()
    iota = np.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(result, iota.sum(axis=1) + 1)

  def test_foreach(self):
    dtype = jnp.int32
    swizzle = 128
    tiling = (8, swizzle // jnp.dtype(dtype).itemsize)
    shape = 128, 192
    mlir_dtype = utils.dtype_to_ir_type(dtype)
    cst = 9999
    def causal(val, idx):
      row, col = idx
      mask = arith.cmpi(arith.CmpIPredicate.uge, row, col)
      return arith.select(mask, val, c(cst, mlir_dtype))

    def kernel(ctx, dst, smem):
      x = iota_tensor(shape[0], shape[1], dtype)
      x.foreach(causal, create_array=True, is_signed=False).store_tiled(smem, swizzle=128)
      mgpu.commit_shared()
      ctx.async_copy(
          src_ref=smem,
          dst_ref=dst,
          gmem_transform=mgpu.TileTransform(tiling),
          swizzle=128,
      )
      ctx.await_async_copy(0)

    iota = np.arange(np.prod(shape), dtype=dtype).reshape(*shape)
    result = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        (),
        jax.ShapeDtypeStruct(shape=shape, dtype=dtype),
        jax.ShapeDtypeStruct(shape=mgpu.tile_shape(shape, tiling), dtype=dtype),
    )()
    expected = jnp.tril(iota) + jnp.triu(jnp.ones(shape), k=1) * cst
    np.testing.assert_array_equal(result, expected)

  @parameterized.product(
      op=[operator.and_, operator.or_, operator.xor],
      dtype=[jnp.uint32],
  )
  def test_bitwise(self, op, dtype, m=64, n=8):
    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, dtype)
      op(iota, iota + 1).store_untiled(dst, optimized=False)

    out_shape = jax.ShapeDtypeStruct((m, n), dtype)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    iota = np.arange(m * n, dtype=dtype).reshape(m, n)
    np.testing.assert_array_equal(result, op(iota, iota + 1))

  @parameterized.product(
      ops=(
          (lambda x: -x, jax.lax.neg),
          (lambda x: x + 42, lambda x: x + 42),
          (lambda x: x.tanh(), jax.lax.tanh),
      ),
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
  )
  def test_unary(self, ops, dtype, m=64, n=32):
    op, np_op = ops
    if np_op is jax.lax.tanh and jnp.issubdtype(dtype, jnp.integer):
      raise self.skipTest("Tanh not supported for integer types")

    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, dtype)
      op(iota).store_untiled(dst, optimized=False)

    out_shape = jax.ShapeDtypeStruct((m, n), dtype)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=dtype).reshape(m, n)
    np.testing.assert_allclose(result, np_op(x), atol=2e-7, rtol=2e-7)

  def test_select(self, m=64, n=32):

    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, jnp.int32)
      (iota < 16).select(iota * 2, iota * 3).store_untiled(dst, optimized=False)

    out_shape = jax.ShapeDtypeStruct((m, n), jnp.int32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=jnp.int32).reshape(m, n)
    np.testing.assert_array_equal(result, np.where(x < 16, x * 2, x * 3))

  @parameterized.product(
      ops=[
          (lambda x: mgpu.FragmentedArray.exp(x), np.exp),
          (lambda x: mgpu.FragmentedArray.sin(x), np.sin),
          (lambda x: mgpu.FragmentedArray.cos(x), np.cos),
          (lambda x: mgpu.FragmentedArray.rsqrt(x), jax.lax.rsqrt),
      ],
      approx=[False, True],
  )
  @jtu.ignore_warning(message="overflow encountered", category=RuntimeWarning)
  def test_math(self, ops, approx, m=64, n=32):
    op, np_op = ops
    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, jnp.float32)
      op(iota).store_untiled(dst, optimized=False)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=jnp.float32).reshape(m, n)
    atol = 5e-3 if approx else 2e-7
    rtol = 4e-6 if approx else 2e-7
    np.testing.assert_allclose(result, np_op(x), atol=atol, rtol=rtol)

  @parameterized.product(
      dtype=[jnp.float32, jnp.int32],
      m=[128],
      n=[32, 64],
  )
  def test_strided_reduce_sum(self, dtype, m, n):
    def kernel(ctx, src, dst, scratch):
      src = mgpu.FragmentedArray.load_strided(
          src, is_signed=utils.is_signed(dtype)
      )
      acc = src.reduce("add", (0, 1), scratch).broadcast((m,))
      acc.store_untiled(dst, optimized=False)

    in_shape = jax.ShapeDtypeStruct((m, n), dtype)
    out_shape = jax.ShapeDtypeStruct((m,), dtype)
    kernel_fn = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        in_shape,
        out_shape,
        smem_scratch_shape=jax.ShapeDtypeStruct((4,), dtype),
    )
    x = np.arange(m * n, dtype=dtype).reshape(m, n)
    np.testing.assert_array_equal(kernel_fn(x), jnp.full((m,), x.sum()))

  @parameterized.product(
      dtype=[jnp.float32, jnp.int32],
      m=[128],
      n=[32, 64],
      reduce_both=[False, True],
  )
  def test_splat_reduce_sum(self, dtype, m, n, reduce_both):
    def kernel(ctx, dst, _):
      src = mgpu.FragmentedArray.splat(
          utils.c(1, utils.dtype_to_ir_type(dtype)),
          (m, n),
          is_signed=utils.is_signed(dtype),
      )
      if reduce_both:
        acc = src.reduce("add", (0, 1)).broadcast((m,))
      else:
        acc = src.reduce("add", 1)
      acc.store_untiled(dst, optimized=False)

    kernel_fn = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        in_shape=(),
        out_shape=jax.ShapeDtypeStruct((m,), dtype),
        smem_scratch_shape=(),
    )
    result = m * n if reduce_both else n
    np.testing.assert_array_equal(kernel_fn(), jnp.full((m,), result, dtype))

  @parameterized.product(
      op=(arith.addf, arith.maximumf),
      m=(64, 128),
      n=(8, 16, 32, 64, 80, 128, 256),
  )
  def test_reduce(self, op, m=64, n=32):
    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, jnp.float32)
      iota.reduce(op, axis=1).broadcast_minor(n).store_untiled(dst, optimized=False)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=jnp.float32).reshape(m, n)
    if op == arith.addf:
      expected = np.broadcast_to(x.sum(axis=1, keepdims=True), x.shape)
    elif op == arith.maximumf:
      expected = np.broadcast_to(x.max(axis=1, keepdims=True), x.shape)
    else:
      raise NotImplementedError(f"Unsupported op: {op}")
    np.testing.assert_array_equal(result, expected)

  def test_splat_layout(self):
    m, n = 64, 8
    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, jnp.float32)
      cte = c(1, iota.mlir_dtype)
      cte_arr = mgpu.FragmentedArray.splat(cte, ())
      cte_arr = cte_arr.reshape((1, 1)).broadcast((m, n))
      (iota + cte_arr).store_untiled(dst, optimized=False)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    expected = np.arange(m * n, dtype=jnp.float32).reshape(m, n) + 1
    np.testing.assert_array_equal(result, expected)

  def test_splat(self):
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      v = arith.constant(f32, ir.FloatAttr.get(f32, 3.14))
      t = mgpu.FragmentedArray.splat(
          v, (128,), mgpu.WGMMA_ROW_LAYOUT
      )
      t.broadcast_minor(32).store_untiled(dst, optimized=False)
    out_shape = jax.ShapeDtypeStruct((128, 32), jnp.float32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    np.testing.assert_array_equal(result, np.full((128, 32), 3.14, np.float32))

  def test_splat_binary_ops(self):
    def kernel(ctx, src, dst, _):
      f32 = ir.F32Type.get()
      pi_arr = mgpu.FragmentedArray.load_strided(src)
      assert isinstance(pi_arr.layout, mgpu.WGStridedFragLayout)
      pi_scalar = arith.constant(f32, ir.FloatAttr.get(f32, 3.14))
      pi_splat = mgpu.FragmentedArray.splat(pi_scalar, ())
      assert isinstance(pi_splat.layout, mgpu.WGSplatFragLayout)
      pi_arr_sq = pi_arr * pi_splat.broadcast(pi_arr.shape)
      assert isinstance(pi_arr_sq.layout, mgpu.WGStridedFragLayout)
      pi_arr_cube = pi_splat.broadcast(pi_arr.shape) * pi_arr_sq
      assert isinstance(pi_arr_cube.layout, mgpu.WGStridedFragLayout)
      (pi_arr == pi_arr).select(pi_splat, pi_arr_cube).store_untiled(dst, optimized=False)

    out_shape = jax.ShapeDtypeStruct((128, 32), jnp.float32)
    inp = jnp.ones_like(out_shape) * 3.14
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), inp, out_shape, ()
    )(inp)
    np.testing.assert_allclose(result, np.full((128, 32), 3.14, np.float32))

  @parameterized.product(in_shape=((128, 128), (128, 64), (64, 128)))
  def test_strided_load_store(self, in_shape):
    def kernel(ctx, *args):
      gmem_input, gmem_output, (smem_input, smem_output) = args
      copy(gmem_input, smem_input)
      t = mgpu.FragmentedArray.load_strided(smem_input)
      t.store_untiled(smem_output)
      copy(smem_output, gmem_output)

    inp = out = self.prng.uniform(-1, 1, in_shape).astype(jnp.float32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (inp,), out, [inp, out],
    )(inp)
    np.testing.assert_array_equal(inp, result)

  @parameterized.product(
      in_shape=((1024,), (256,), (128,), (64,)),
      dtype=(jnp.float16, jnp.float32),
      swizzle=(16, 32, 64, 128)
  )
  def test_wgmma_row_load_store_with_layout(self, in_shape, dtype, swizzle):
    def kernel(ctx, gmem_input, gmem_output, smem):
      smem_input, smem_output = smem
      copy(gmem_input, smem_input, swizzle=swizzle)
      t = mgpu.FragmentedArray.load_untiled(
          smem_input, layout=mgpu.WGMMA_ROW_LAYOUT, swizzle=swizzle
      )
      t.store_untiled(smem_output)
      copy(smem_output, gmem_output)

    inp = out = self.prng.uniform(-1, 1, in_shape).astype(dtype)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (inp,), out, [inp, out],
    )(inp)
    np.testing.assert_array_equal(inp, result)

  @parameterized.product(
      in_shape=((128,), (64,)),
      dtype=(jnp.float16, jnp.float32),
      swizzle=(16, 32, 64, 128),
  )
  def test_wgmma_col_load_store_with_layout(self, in_shape, dtype, swizzle):
    def kernel(ctx, *args):
      gmem_input, gmem_output, (smem_input, smem_output) = args
      copy(gmem_input, smem_input, swizzle=swizzle)
      t = mgpu.FragmentedArray.load_untiled(
          smem_input, swizzle=swizzle, layout=mgpu.WGMMA_COL_LAYOUT
      )
      t.store_untiled(smem_output)
      copy(smem_output, gmem_output)

    inp = out = self.prng.uniform(-1, 1, in_shape).astype(dtype)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (inp,), out, [inp, out],
    )(inp)
    np.testing.assert_array_equal(result, inp)

  @parameterized.parameters((128, 128), (128, 64), (64, 128))
  def test_broadcast_major(self, m, n):
    def kernel(ctx, gmem_input, gmem_output, _):
      t = mgpu.FragmentedArray.load_untiled(
          gmem_input, layout=mgpu.WGMMA_COL_LAYOUT, optimized=False
      )
      t.broadcast_major(m).store_untiled(gmem_output, optimized=False)

    inp = self.prng.uniform(-1, 1, (n,)).astype(jnp.float16)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float16)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (inp,), out_shape, inp
    )(inp)
    out_ref = jax.lax.broadcast_in_dim(inp, (m, n), (1,))
    np.testing.assert_array_equal(result, out_ref)

  def test_warp_tree_reduce(self):
    def kernel(ctx, out, *_):
      del ctx
      i32 = ir.IntegerType.get_signless(32)
      tid = gpu.thread_id(gpu.Dimension.x)
      value = arith.index_cast(i32, tid)
      grp = warp_tree_reduce(value, arith.addi, 4)
      memref.store(grp, out, [tid])

    x = np.arange(128, dtype=jnp.int32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), x, [],
    )()
    for i in range(0, 128, 4):
      x[i:i + 4] = jnp.sum(x[i:i + 4])

    np.testing.assert_array_equal(result, x)

  @parameterized.parameters(2, 4)
  def test_fast_i8_convert(self, reg_length):
    jax_dtype_to = jnp.dtype(jnp.bfloat16)
    jax_dtype_from = jnp.dtype(jnp.int8)
    mlir_dtype_to = utils.dtype_to_ir_type(jax_dtype_to)
    def kernel(ctx, inp, out, smem):
      del ctx, smem
      arr = mgpu.FragmentedArray.load_strided(inp, is_signed=True)
      assert ir.VectorType(arr.registers.flat[0].type).shape == [reg_length]
      arr.astype(mlir_dtype_to).store_untiled(out, optimized=False)

    x = jnp.arange(-128, 128, dtype=jax_dtype_from)
    x = jnp.tile(x, reg_length // 2)
    reference = x.astype(jax_dtype_to)

    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, reference, None,
    )(x)
    np.testing.assert_array_equal(result, reference)

  @parameterized.parameters(
      ([64 * 4], "WGMMA_ROW_LAYOUT"),
      ([64 * 4, 8 * 2], "WGMMA_LAYOUT"),
  )
  def test_to_layout(self, shape, new_layout):
    def kernel(ctx, _):
      # No assertions, we are just checking there are no compile-time errors.
      arr = mgpu.FragmentedArray.splat(c(42.0, ir.F32Type.get()), shape)
      arr.to_layout(getattr(mgpu, new_layout))

    _ = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), (), (), None)()

  @parameterized.parameters(
      (jnp.float16, jnp.float16),  # Noop
      (jnp.int16, jnp.bfloat16),
      (jnp.int16, jnp.float16),
      (jnp.uint16, jnp.float16),
      (jnp.float32, jnp.int32),
      (jnp.float32, jnp.uint32),
      (jnp.uint32, jnp.int32),
      (jnp.int32, jnp.uint32),
  )
  def test_bitcast(self, in_dtype, out_dtype):
    out_ir_type = utils.dtype_to_ir_type(out_dtype)
    in_is_signed = utils.is_signed(in_dtype)
    out_is_signed = utils.is_signed(out_dtype)

    def kernel(ctx, inp, out, smem):
      del ctx, smem
      arr = mgpu.FragmentedArray.load_strided(inp, is_signed=in_is_signed)
      arr = arr.bitcast(out_ir_type, output_is_signed=out_is_signed)
      arr.store_untiled(out)

    x = jnp.arange(256, dtype=in_dtype)
    reference = jax.lax.bitcast_convert_type(x, out_dtype)

    result = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        x,
        reference,
        None,
    )(x)
    np.testing.assert_array_equal(result, reference)

  @parameterized.parameters(jnp.float32, jnp.float16, jnp.bfloat16)
  def test_optimization_barrier(self, dtype):
    def kernel(ctx, inp, out, smem):
      del ctx, smem
      arr = mgpu.FragmentedArray.load_strided(inp)
      arr2 = arr * 2
      arr, arr2 = mgpu.optimization_barrier(arr, arr2)
      (arr + arr2).store_untiled(out)

    x = jnp.arange(256, dtype=dtype)

    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, None)
    np.testing.assert_array_equal(f(x), x * 3)

  def test_optimization_barrier_with_single_value(self):
    shape = (64, 64)
    value = 5.0
    dtype = jnp.float32
    def kernel(ctx, out, smem):
      del ctx, smem
      mlir_type = utils.dtype_to_ir_type(dtype)
      arr = mgpu.FragmentedArray.splat(c(value, mlir_type), shape)
      arr = mgpu.optimization_barrier(arr)
      arr.store_untiled(out)

    out_shape = jax.ShapeDtypeStruct(shape, dtype)
    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ())
    np.testing.assert_array_equal(f(), jnp.full(shape, value, dtype=dtype))

  def test_convert_bool_to_u8(self):
    m, n = 128, 128
    def kernel(ctx, dst, _):
      i8 = ir.IntegerType.get_signless(8)
      iota = iota_tensor(m, n, jnp.uint8)
      (iota > 10).astype(i8, is_signed=False).store_untiled(dst, optimized=False)

    out_shape = jax.ShapeDtypeStruct((m, n), jnp.int8)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    iota = np.arange(m * n, dtype=jnp.uint8).reshape(m, n)
    np.testing.assert_array_equal(result, (iota > 10).astype(jnp.uint8))

  @parameterized.parameters(
      (jnp.uint8, jnp.uint16, 255),
      (jnp.uint8, jnp.int16, 255),
      (jnp.int8, jnp.uint16, -127),
      (jnp.int8, jnp.int16, -127),
  )
  def test_convert_int_uint(self, from_dtype, to_dtype, value):
    m, n = 1, 128
    def kernel(ctx, dst, _):
      i8 = ir.IntegerType.get_signless(8)
      from_mlir_dtype = utils.dtype_to_ir_type(from_dtype)
      to_mlir_dtype = utils.dtype_to_ir_type(to_dtype)
      from_arr = mgpu.FragmentedArray.splat(
          c(value, from_mlir_dtype),
          (m, n),
          is_signed=utils.is_signed(from_dtype),
      )
      to_arr = from_arr.astype(to_mlir_dtype, is_signed=utils.is_signed(to_dtype))
      to_arr.store_untiled(dst)

    out_shape = jax.ShapeDtypeStruct((m, n), to_dtype)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    expected = jnp.full((m, n), value, dtype=from_dtype).astype(to_dtype)
    np.testing.assert_array_equal(result, expected)


class ProfilerTest(TestCase):

  def test_measure_events_explicit(self):
    x = jnp.arange(1024 * 1024)
    _, runtime_ms = profiler.measure(lambda x, y: x + y, mode="events")(x, x)
    self.assertIsInstance(runtime_ms, float)

  def test_profile(self):
    def kernel(ctx, src, dst, _):
      mgpu.FragmentedArray.load_strided(src).store_untiled(dst)
    x = np.arange(64 * 64, dtype=jnp.float32).reshape(64, 64)
    spec = profiler.ProfilerSpec(1024)
    # This is just a smoke test.
    f = jax.jit(mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, (), prof_spec=spec
    ))
    jax.block_until_ready(f(x))


class TorchTest(TestCase):

  def setUp(self):
    super().setUp()
    try:
      import torch
    except ImportError:
      raise unittest.SkipTest("Test requires PyTorch")
    self.torch = torch

  def test_basic(self):
    def kernel(ctx, i_gmem, o_gmem, _):
      x = mgpu.FragmentedArray.load_strided(i_gmem)
      (x + x).store_untiled(o_gmem)

    ty = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    x = self.torch.randn((128, 128), dtype=self.torch.float, device='cuda')
    f = mgpu.as_torch_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), ty, ty, ())
    y = f(x)
    np.testing.assert_allclose(y.cpu(), x.cpu() * 2)
    del y  # Make sure the destructor runs successfully.


class LayoutTest(TestCase):

  @parameterized.named_parameters(
      ("f32", jnp.float32, 256),
      ("f16", jnp.float16, 256),
      ("f16_small", jnp.float16, 128),
  )
  def test_store_untiled_splat(self, jax_dtype, size):
    mlir_dtype = utils.dtype_to_ir_type(jax_dtype)
    def kernel(ctx, out, _):
      del ctx
      arr = mgpu.FragmentedArray.splat(
          c(1.0, mlir_dtype), (size,), is_signed=utils.is_signed(jax_dtype)
      )
      arr.store_untiled(out)
    expected = np.ones((size,), jax_dtype)
    mosaic_ones = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, ()
    )()
    np.testing.assert_array_equal(mosaic_ones, expected)

  @parameterized.product(
      shape=((128, 128), (64, 8), (64, 256)),
      dtype=(jnp.int32, jnp.int16, jnp.int8),
  )
  def test_wgmma_tiled_layout(self, shape, dtype):
    def kernel(ctx, dst, _):
      iota = iota_tensor(*shape, dtype)
      tiled = iota.to_layout(fa._tiled_wgmma_layout(shape))
      # Note that WGMMA layouts are always (shape[0] // 64, shape[1] // 8, 2, 1)
      self.assertEqual(
          tiled.registers.shape,
          (shape[0] // 64, shape[1] // 8, 1, 1, 2, 1, 1, 1, 1, 1),
      )
      self.assertEqual(tiled.shape, shape)
      self.assertEqual(tiled.mlir_dtype, iota.mlir_dtype)
      tiled.store_untiled(dst, optimized=False)
    ty = jax.ShapeDtypeStruct(shape, dtype)
    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), (), ty, ())
    expected = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    np.testing.assert_array_equal(f(), expected)

  @parameterized.product(
      dtype=[jnp.int8, jnp.int16, jnp.int32],
      swizzle=[16, 32, 64, 128],
      num_col_tiles=[1, 2, 3],
      row_tiling=[8, 64],
  )
  @jtu.thread_unsafe_test()  # Modifies ``os.environ``.
  def test_copy_tiled(self, dtype, swizzle, num_col_tiles, row_tiling):
    mlir_dtype = utils.dtype_to_ir_type(dtype)
    bw = bytewidth(mlir_dtype)
    col_tiling = swizzle // bw
    if col_tiling % 8:
      self.skipTest("WGMMA layout requires col_tiling % 8 == 0")
    m, n = 128, col_tiling * num_col_tiles
    tiling = (row_tiling, col_tiling)
    def kernel(ctx, in_, out, smems):
      smem_in, smem_out, barrier = smems
      ctx.async_copy(src_ref=in_, dst_ref=smem_in, swizzle=swizzle, barrier=barrier)
      barrier.wait()
      t = mgpu.FragmentedArray.load_tiled(
          smem_in, swizzle=swizzle, is_signed=True, layout=mgpu.WGMMA_LAYOUT
      )
      t.store_tiled(smem_out, swizzle=swizzle)
      mgpu.commit_shared()
      ctx.async_copy(src_ref=smem_out, dst_ref=out, swizzle=swizzle)
      ctx.await_async_copy(0)
    expected = (
        np.arange(m * n, dtype=dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )

    with jtu.set_env(MOSAIC_GPU_DUMP_SASS="1"), self.capture_stdout() as sass:
      iota = mgpu.as_gpu_kernel(
          kernel, (1, 1, 1), (128, 1, 1), expected, expected,
          [expected, expected, mgpu.TMABarrier()],
      )(expected)
    np.testing.assert_array_equal(iota, expected)

    # Verify that we don't use too many registers for the transfers.
    # We verify LDS and STS separately, because they might use two different
    # methods of computing offsets and we don't rely on CSE between them.
    expected_regs = swizzle // bytewidth(mlir_dtype) // 8
    # When the bytewidth is smaller than 2 the swizzle pattern changes every 2
    # column tiles, so we only need half the registers.
    if bytewidth(mlir_dtype) < 2:
      expected_regs //= 2
    for instr in ("STS", "LDS"):
      with self.subTest(instr + " count"):
        addrs = re.findall(instr + r".* \[(.*)\]", sass())
        def get_reg(addr):
          if (pos := addr.find("+")) != -1:
            return addr[:pos]
          return addr
        used_regs = {get_reg(addr) for addr in addrs}
        self.assertLessEqual(len(used_regs), expected_regs)

  def test_copy_for_upcast(self):
    dtype = jnp.int8
    swizzle = 128
    col_tiling = swizzle // bytewidth(utils.dtype_to_ir_type(dtype))
    m, n = 128, col_tiling * 2
    tiling = (64, col_tiling)
    layout = fa.WGMMA_LAYOUT_UPCAST_2X
    def kernel(ctx, in_, out, smems):
      smem_in, smem_out, barrier = smems
      ctx.async_copy(src_ref=in_, dst_ref=smem_in, swizzle=swizzle, barrier=barrier)
      barrier.wait()
      t = mgpu.FragmentedArray.load_tiled(
          smem_in, swizzle=swizzle, is_signed=True, layout=layout
      )
      t.store_tiled(smem_out, swizzle=swizzle)
      mgpu.commit_shared()
      ctx.async_copy(src_ref=smem_out, dst_ref=out, swizzle=swizzle)
      ctx.await_async_copy(0)
    x = jax.random.randint(
        jax.random.key(42), tile_shape((m, n), tiling), -128, 127, dtype=dtype
    )
    f = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, [x, x, mgpu.TMABarrier()],
    )
    np.testing.assert_array_equal(f(x), x)

  @parameterized.product(
      dtype=[jnp.int16],  # TODO(apaszke): More dtypes
      # TODO(apaszke): swizzle=64 <- not implemented in transfer_tiled right now
      swizzle=[16, 32, 128],
  )
  def test_transpose_tiled(self, dtype, swizzle):
    mlir_dtype = utils.dtype_to_ir_type(dtype)
    bw = bytewidth(mlir_dtype)
    col_tiling = swizzle // bw
    m, n = 128, 256
    tiling = (8, col_tiling)
    transpose_layout = fa.WGMMA_TRANSPOSED_LAYOUT
    def kernel(ctx, in_, out, smems):
      smem_in, smem_out, barrier = smems
      ctx.async_copy(src_ref=in_, dst_ref=smem_in, swizzle=swizzle, barrier=barrier)
      barrier.wait()
      t = mgpu.FragmentedArray.load_tiled(
          smem_in, swizzle=swizzle, is_signed=True, layout=fa.WGMMA_LAYOUT
      )
      smem_out_t = memref_transpose(smem_out, (1, 0, 3, 2))
      t.to_layout(transpose_layout).store_tiled(smem_out_t, swizzle=swizzle)
      mgpu.commit_shared()
      ctx.async_copy(src_ref=smem_out, dst_ref=out, swizzle=swizzle)
      ctx.await_async_copy(0)
    x = (
        np.arange(m * n, dtype=dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )
    y_ref = (
        np.arange(m * n, dtype=dtype)
        .reshape(m, n)
        .T.reshape(n // tiling[0], tiling[0], m // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )

    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, y_ref, [x, y_ref, mgpu.TMABarrier()],
    )(x)
    np.testing.assert_array_equal(y, y_ref)

  @parameterized.parameters(
      (fa.WGMMA_LAYOUT_UPCAST_2X, fa.WGMMA_LAYOUT, jnp.int8, jnp.int8, 1),
      (fa.WGMMA_LAYOUT_UPCAST_2X, fa.WGMMA_LAYOUT, jnp.int8, jnp.int16, 1),
      (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT_UPCAST_2X, jnp.int4, jnp.int4, 1),
      (fa.WGMMA_LAYOUT_UPCAST_2X, fa.WGMMA_LAYOUT, jnp.int4, jnp.int4, 0.5),
      (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT, jnp.int4, jnp.int4, 2),
  )
  @jtu.thread_unsafe_test()  # Modifies ``os.environ``.
  def test_upcast_to_wgmma(
      self, start_layout, end_layout, in_dtype, cast_dtype, shfl_per_reg
  ):
    in_dtype = jnp.dtype(in_dtype)
    out_dtype = jnp.dtype(jnp.int16)
    out_dtype_mlir = utils.dtype_to_ir_type(out_dtype)
    swizzle = 128
    in_col_tiling = 8 * swizzle // jnp.iinfo(in_dtype).bits
    in_tiling = (8, in_col_tiling)
    out_col_tiling = swizzle // out_dtype.itemsize
    out_tiling = (8, out_col_tiling)
    m, n = 128, in_col_tiling * 2
    regs_per_thread = None
    def kernel(ctx, in_, out, smems):
      nonlocal regs_per_thread
      smem_in, smem_out, barrier = smems
      ctx.async_copy(src_ref=in_, dst_ref=smem_in, swizzle=swizzle, barrier=barrier)
      barrier.wait()
      t = mgpu.FragmentedArray.load_tiled(
          smem_in, swizzle=swizzle, is_signed=True, layout=start_layout
      )
      regs_per_thread = t.registers.size
      t = t.astype(utils.dtype_to_ir_type(cast_dtype), is_signed=True)
      t = t.to_layout(end_layout)
      t = t.astype(out_dtype_mlir, is_signed=True)
      t.store_tiled(smem_out, swizzle=swizzle)
      mgpu.commit_shared()
      ctx.async_copy(src_ref=smem_out, dst_ref=out, swizzle=swizzle)
      ctx.await_async_copy(0)
    def tile(x, tiling):
      return x.reshape(
          x.shape[0] // tiling[0], tiling[0], x.shape[1] // tiling[1], tiling[1]
      ).transpose(0, 2, 1, 3)
    in_iinfo = jnp.iinfo(in_dtype)
    x = jax.random.randint(
        jax.random.key(42), (m, n), in_iinfo.min, in_iinfo.max, dtype=jnp.int32
    ).astype(in_dtype)
    xt = tile(x, in_tiling)
    y = x.astype(out_dtype)
    yt = tile(y, out_tiling)
    f = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), xt, yt, [xt, yt, mgpu.TMABarrier()],
    )
    with jtu.set_env(MOSAIC_GPU_DUMP_SASS="1"), self.capture_stdout() as sass:
      yt_kernel = f(xt)
      jax.block_until_ready(yt_kernel)
    np.testing.assert_array_equal(yt_kernel, yt)
    self.assertEqual(sass().count("SHFL.BFLY"), regs_per_thread * shfl_per_reg)


@dataclasses.dataclass(frozen=True)
class Tile:
  """Defines a Tile transform in a TestCaseInput.

  Note that we cannot simply alias mgpu_dialect.TileTransformAttr.get, because
  we do not have an MLIR context at the point we define the TestCaseInput.
  """

  tiling: tuple[int, ...]

  def attr(self):
    return mgpu_dialect.TileTransformAttr.get(self.tiling)


@dataclasses.dataclass(frozen=True)
class Transpose:
  """Defines a Transpose transform in a TestCaseInput.

  Note that we cannot simply alias mgpu_dialect.TransposeTransformAttr.get,
  because we do not have an MLIR context at the point we define the
  TestCaseInput.
  """

  permutation: tuple[int, ...]

  def attr(self):
    return mgpu_dialect.TransposeTransformAttr.get(self.permutation)


@dataclasses.dataclass(frozen=True)
class Swizzle:
  """Defines a Swizzle transform in a TestCaseInput.

  Note that we cannot simply alias mgpu_dialect.SwizzleTransformAttr.get,
  because we do not have an MLIR context at the point we define the
  TestCaseInput.
  """

  swizzle: mgpu_dialect.SwizzlingMode

  def attr(self):
    return mgpu_dialect.SwizzleTransformAttr.get(self.swizzle)


def set_in_transforms(
    op: ir.OpView, transforms: Sequence[Sequence[Tile | Transpose | Swizzle]],
) -> None:
  """Annotates an op with in_transforms."""
  if not transforms:
    return

  in_transforms = []
  smem_refs = filter(inference_utils.is_transformable_smem_memref, op.operands)  # pylint: disable=undefined-variable
  for _, result_transforms in jax._src.util.safe_zip(smem_refs, transforms):
    in_transforms.append(
        ir.ArrayAttr.get([t.attr() for t in result_transforms])
    )

  op.attributes["in_transforms"] = ir.ArrayAttr.get(in_transforms)


class MosaicGpuDialectTest(TestCase, jtu.JaxTestCase):
  """Device tests with lowering from the MLIR dialect and layout inference."""

  def setUp(self):
    if mgpu_dialect is None:
      raise self.skipTest("Test requires Mosaic GPU dialect")
    super().setUp()

  def test_pointwise_kernel(self):
    def add(ctx, a, b, result, smem):
      del ctx, smem
      shape = ir.MemRefType(a.type).shape
      elt_type = ir.MemRefType(a.type).element_type

      zero_index = arith.constant(ir.IndexType.get(), 0)
      zero_vector_indices = [zero_index] * len(shape)

      # GMEM -> registers
      ab_type = ir.VectorType.get(shape, elt_type)
      a = vector.load(ab_type, a, zero_vector_indices)
      b = vector.load(ab_type, b, zero_vector_indices)

      # Computation
      add = arith.addf(a, b)

      # Registers -> GMEM
      vector.store(add, result, zero_vector_indices)

    dtype = jnp.bfloat16
    shape = (128, 128)
    jax_shape = jax.ShapeDtypeStruct(shape, dtype)
    kernel = mgpu.as_gpu_kernel(
        add,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(jax_shape, jax_shape),
        out_shape=jax_shape,
        smem_scratch_shape=[],
        thread_semantics=mgpu.LoweringSemantics.Warpgroup,
    )

    x = self.prng.uniform(-1, 1, shape).astype(dtype)
    y = self.prng.uniform(-1, 1, shape).astype(dtype)

    self.assertArraysEqual(jax.jit(kernel)(x, y), x + y)

  @staticmethod
  def kernel_with_tma_cases(dtype: jnp.dtype):
    @dataclasses.dataclass()
    class TestCaseInput:
      shape: tuple[int, ...]
      shape_sliced: tuple[int, ...] = ()
      slice_indices: tuple[int, ...] = ()
      slice_lengths: tuple[int, ...] = ()
      transforms: tuple[Tile | Transpose | Swizzle, ...] = ()

      def __post_init__(self):
        if not self.shape_sliced:
          self.shape_sliced = self.shape
        if not self.slice_lengths:
          self.slice_lengths = self.shape_sliced
        if not self.slice_indices:
          self.slice_indices = tuple([0] * len(self.slice_lengths))

    result = []
    for swizzle in mgpu_dialect.SwizzlingMode:
      n = swizzle * 8 // jnp.finfo(dtype).bits
      if swizzle == mgpu_dialect.SwizzlingMode.kNoSwizzle:
        #  We need at least one case with no transforms, as this is handled
        #  differently.
        result.append(TestCaseInput(shape=[128, n]))
      result.extend([
          TestCaseInput(
              shape=[128, n],
              transforms=[Swizzle(swizzle)],
          ),
          TestCaseInput(
              shape=[256, n],
              shape_sliced=[128, n],
              transforms=[Swizzle(swizzle)],
          ),
          TestCaseInput(
              shape=[2, 128, n],
              shape_sliced=[128, n],
              slice_lengths=[-1, 128, n],
              slice_indices=[1, 0, 0],
              transforms=[Swizzle(swizzle)],
          ),
          TestCaseInput(
              shape=[2, 3, 64, n],
              transforms=[Transpose([0, 1, 2, 3]), Swizzle(swizzle)],
          ),
          TestCaseInput(
              shape=[2, 3, 64, n],
              transforms=[
                  Transpose([1, 0, 2, 3]),
                  Transpose([1, 0, 2, 3]),
                  Swizzle(swizzle),
              ],
          ),
          TestCaseInput(
              shape=[2, 3, 64, n],
              transforms=[Transpose([1, 0, 2, 3]), Swizzle(swizzle)],
          ),
          TestCaseInput(
              shape=[256, n],
              shape_sliced=[128, n],
              transforms=[Tile([64, n]), Swizzle(swizzle)],
          ),
          TestCaseInput(
              shape=[2 * 64, 3 * n],
              transforms=[
                  Tile([64, n]),
                  Transpose([1, 0, 2, 3]),
                  Swizzle(swizzle),
              ],
          ),
      ])
    return result

  @parameterized.parameters(kernel_with_tma_cases(jnp.bfloat16))
  def test_kernel_with_tma(self, test_case):
    def add(
        ctx: launch_context.LaunchContext,
        in_gmem_ref: ir.Value,
        result_gmem_ref: ir.Value,
        smem: list[ir.Value],
    ):
      del ctx
      smem_ref, tma_barrier = smem
      dialect_barrier = tma_barrier.as_barrier_memref()

      elt_type = ir.MemRefType(in_gmem_ref.type).element_type
      memref_bytes = utils.bytewidth(elt_type) * math.prod(
          test_case.shape_sliced
      )
      mgpu_dialect.arrive_expect_tx(
          barrier=dialect_barrier, expect_tx= memref_bytes
      )

      i32 = ir.IntegerType.get_signless(32)
      slice_indices = [arith.constant(i32, i) for i in test_case.slice_indices]

      # GMEM -> SMEM
      load_op = mgpu_dialect.AsyncLoadOp(
          source=in_gmem_ref,
          destination=smem_ref,
          barrier=dialect_barrier,
          indices=slice_indices,
          slice_lengths=test_case.slice_lengths,
          collective=ir.ArrayAttr.get([]),
      )
      set_in_transforms(load_op, [test_case.transforms])

      parities = memref.load(tma_barrier.barrier_ref.phases, [])
      parity, _ = tma_barrier.update_parities(parities)
      mgpu_dialect.wait(dialect_barrier, parity)

      # SMEM -> GMEM
      zero_index = arith.constant(i32, 0)
      mgpu_dialect.async_store(
          source=smem_ref,
          destination=result_gmem_ref,
          indices=[zero_index] * len(test_case.shape_sliced),
          slice_lengths=test_case.shape_sliced,
      )
      nvvm.cp_async_bulk_wait_group(0)
      utils.warpgroup_barrier()

    dtype = jnp.bfloat16

    jax_shape = jax.ShapeDtypeStruct(test_case.shape, dtype)
    jax_shape_sliced = jax.ShapeDtypeStruct(test_case.shape_sliced, dtype)
    kernel = mgpu.as_gpu_kernel(
        add,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(jax_shape),
        out_shape=jax_shape_sliced,
        smem_scratch_shape=[
            jax_shape_sliced,
            core.TMABarrier(1),
        ],
        thread_semantics=mgpu.LoweringSemantics.Warpgroup,
    )

    x = self.prng.uniform(-1, 1, test_case.shape).astype(dtype)

    input_slice = tuple(
        slice(i * abs(l), (i + 1) * abs(l))
        for i, l in zip(test_case.slice_indices, test_case.slice_lengths)
    )
    self.assertArraysEqual(
        jax.jit(kernel)(x),
        (x[input_slice]).reshape(test_case.shape_sliced),
    )

  def test_pointwise_kernel_with_tma(self):
    def add(
        ctx: launch_context.LaunchContext,
        a_gmem_ref: ir.Value,
        b_gmem_ref: ir.Value,
        result_gmem_ref: ir.Value,
        smem: list[ir.Value],
    ):
      del ctx
      a_smem_ref, b_smem_ref, result_smem_ref, tma_barrier = smem
      dialect_barrier = tma_barrier.as_barrier_memref()

      memref_type = ir.MemRefType(a_gmem_ref.type)
      shape = memref_type.shape
      elt_type = memref_type.element_type

      memref_bytes = utils.bytewidth(elt_type) * math.prod(shape)
      mgpu_dialect.arrive_expect_tx(
          barrier=dialect_barrier, expect_tx=2 * memref_bytes
      )

      zero_i32 = arith.constant(ir.IntegerType.get_signless(32), 0)
      zero_slice_indices = [zero_i32] * memref_type.rank

      # GMEM -> SMEM
      mgpu_dialect.async_load(
          source=a_gmem_ref,
          destination=a_smem_ref,
          barrier=dialect_barrier,
          indices=zero_slice_indices,
          slice_lengths=shape,
          collective=ir.ArrayAttr.get([]),
      )
      mgpu_dialect.async_load(
          source=b_gmem_ref,
          destination=b_smem_ref,
          barrier=dialect_barrier,
          indices=zero_slice_indices,
          slice_lengths=shape,
          collective=ir.ArrayAttr.get([]),
      )

      parities = memref.load(tma_barrier.barrier_ref.phases, [])
      parity, _ = tma_barrier.update_parities(parities)
      mgpu_dialect.wait(dialect_barrier, parity)

      zero_index = arith.constant(ir.IndexType.get(), 0)
      zero_vector_indices = [zero_index] * memref_type.rank

      # SMEM -> registers
      ab_type = ir.VectorType.get(shape, elt_type)
      a = vector.load(ab_type, a_smem_ref, zero_vector_indices)
      b = vector.load(ab_type, b_smem_ref, zero_vector_indices)

      # Computation
      add = arith.addf(arith.addf(a, b), b)

      # Registers -> SMEM
      vector.store(add, result_smem_ref, zero_vector_indices)

      # SMEM -> GMEM
      mgpu_dialect.async_store(
          source=result_smem_ref,
          destination=result_gmem_ref,
          indices=zero_slice_indices,
          slice_lengths=shape,
      )
      nvvm.cp_async_bulk_wait_group(0)
      utils.warpgroup_barrier()

    dtype = jnp.bfloat16

    spec = jax.ShapeDtypeStruct((2, 3, 4, 64), dtype)
    kernel = mgpu.as_gpu_kernel(
        add,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(spec, spec),
        out_shape=spec,
        smem_scratch_shape=[
            spec,
            spec,
            spec,
            core.TMABarrier(1),
        ],
        thread_semantics=mgpu.LoweringSemantics.Warpgroup,
    )

    x = self.prng.uniform(-1, 1, spec.shape).astype(dtype)
    y = self.prng.uniform(-1, 1, spec.shape).astype(dtype)

    self.assertArraysEqual(jax.jit(kernel)(x, y), x + y + y)

  @parameterized.parameters(
      ((64,), (64, 128), [0]),
      ((64,), (128, 64), [1]),
  )
  def test_broadcast_in_dim(self, input_shape, output_shape, bcast_dims):
    element_value = 42.0
    def body(ctx, result_gmem_ref, smem):
      del ctx
      result_smem_ref = smem[0]

      f32 = ir.F32Type.get()
      zero_index = arith.constant(ir.IndexType.get(), 0)

      # Create input in registers
      x_type = ir.VectorType.get(input_shape, f32)
      c = arith.constant(f32, element_value)
      x = vector.splat(x_type, c)

      # Computation
      out_type = ir.VectorType.get(output_shape, f32)
      expanded = mgpu_dialect.broadcast_in_dim(out_type, x, bcast_dims)
      cast = mgpu_dialect.layout_cast(
          expanded, layouts.to_layout_attr(fa.WGMMA_LAYOUT)
      )

      # Registers -> SMEM
      vector.store(cast, result_smem_ref, [zero_index] * len(output_shape))

      # SMEM -> GMEM
      zero_i32 = arith.constant(ir.IntegerType.get_signless(32), 0)
      mgpu_dialect.async_store(
          source=result_smem_ref,
          destination=result_gmem_ref,
          indices=[zero_i32] * len(output_shape),
          slice_lengths=output_shape,
      )
      nvvm.cp_async_bulk_wait_group(0)
      utils.warpgroup_barrier()

    dtype = jnp.float32
    kernel = mgpu.as_gpu_kernel(
        body,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(),
        out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
        smem_scratch_shape=[jax.ShapeDtypeStruct(output_shape, dtype)],
        thread_semantics=mgpu.LoweringSemantics.Warpgroup,
    )

    x = np.full(input_shape, element_value, dtype=dtype)
    self.assertArraysEqual(
        jax.jit(kernel)(), jax.lax.broadcast_in_dim(x, output_shape, bcast_dims)
    )

  @parameterized.parameters(
      (jnp.float32, 5.0, 2.0, vector.CombiningKind.ADD),
      (jnp.float32, 5.0, 2.0, vector.CombiningKind.MAXIMUMF),
      (jnp.float32, 5.0, 7.0, vector.CombiningKind.MAXIMUMF),
      (jnp.int32, 5, 2, vector.CombiningKind.MAXSI),
      (jnp.int32, -5, -2, vector.CombiningKind.MAXSI),
      (jnp.int32, -2, -5, vector.CombiningKind.MAXSI),
      (jnp.uint32, 5, 2, vector.CombiningKind.MAXUI),
      (jnp.uint32, 2, 5, vector.CombiningKind.MAXUI),
      #
      # TODO(dasenov): Add tests for wgmma_col_layout output once
      # fragmented_array.reduce supports that.
  )
  def test_vector_multi_dim_reduction(
      self,
      dtype,
      input_value,
      init_value,
      kind,
  ):
    input_shape = (128, 64)
    output_shape = (128,)
    red_dims = [1]

    def body(ctx, result_gmem_ref, smem):
      del ctx
      result_smem_ref = smem[0]

      el_type = utils.dtype_to_ir_type(dtype)
      zero_index = arith.constant(ir.IndexType.get(), 0)

      # Create source in registers
      source_type = ir.VectorType.get(input_shape, el_type)
      c = arith.constant(el_type, input_value)
      source = vector.splat(source_type, c)

      # Create accumulator in registers
      acc_type = ir.VectorType.get(output_shape, el_type)
      c = arith.constant(el_type, init_value)
      acc = vector.splat(acc_type, c)

      # Cast inputs
      source = mgpu_dialect.layout_cast(
          source, layouts.to_layout_attr(fa.WGMMA_LAYOUT)
      )
      acc_layout = (
          fa.WGMMA_ROW_LAYOUT if red_dims[0] == 1 else fa.WGMMA_COL_LAYOUT
      )
      acc = mgpu_dialect.layout_cast(acc, layouts.to_layout_attr(acc_layout))

      # Computation
      reduced = vector.multi_reduction(kind, source, acc, red_dims)

      # Registers -> SMEM
      vector.store(reduced, result_smem_ref, [zero_index] * len(output_shape))

      # SMEM -> GMEM
      zero_i32 = arith.constant(ir.IntegerType.get_signless(32), 0)
      mgpu_dialect.async_store(
          source=result_smem_ref,
          destination=result_gmem_ref,
          indices=[zero_i32] * len(output_shape),
          slice_lengths=output_shape,
      )
      nvvm.cp_async_bulk_wait_group(0)
      utils.warpgroup_barrier()

    kernel = mgpu.as_gpu_kernel(
        body,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(),
        out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
        smem_scratch_shape=[jax.ShapeDtypeStruct(output_shape, dtype)],
        thread_semantics=mgpu.LoweringSemantics.Warpgroup,
    )

    source = np.full(input_shape, input_value, dtype=dtype)
    acc = np.full(output_shape, init_value, dtype=dtype)
    if kind == vector.CombiningKind.ADD:
      red = jax.lax.reduce_sum(source, red_dims)
      red = red + acc
    else:
      red = jax.lax.reduce_max(source, red_dims)
      red = jax.lax.max(red, acc)
    self.assertArraysEqual(jax.jit(kernel)(), red)

  @parameterized.parameters(fa.WGMMA_ROW_LAYOUT, fa.WGMMA_COL_LAYOUT)
  def test_wgmma_row_col_store(self, in_layout):
    element_value = 42.0
    shape = (64, )
    def body(ctx, result_gmem_ref, smem):
      del ctx
      result_smem_ref = smem[0]

      f32 = ir.F32Type.get()
      zero_index = arith.constant(ir.IndexType.get(), 0)

      # Create input in registers
      x_type = ir.VectorType.get(shape, f32)
      c = arith.constant(f32, element_value)
      x = vector.splat(x_type, c)
      cast = mgpu_dialect.layout_cast(x, layouts.to_layout_attr(in_layout))

      # Registers -> SMEM
      vector.store(cast, result_smem_ref, [zero_index])

      # SMEM -> GMEM
      zero_i32 = arith.constant(ir.IntegerType.get_signless(32), 0)
      mgpu_dialect.async_store(
          source=result_smem_ref,
          destination=result_gmem_ref,
          indices=[zero_i32],
          slice_lengths=shape,
      )
      nvvm.cp_async_bulk_wait_group(0)
      utils.warpgroup_barrier()

    dtype = jnp.float32
    kernel = mgpu.as_gpu_kernel(
        body,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(),
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        smem_scratch_shape=[jax.ShapeDtypeStruct(shape, dtype)],
        thread_semantics=mgpu.LoweringSemantics.Warpgroup,
    )

    x = np.full(shape, element_value, dtype=dtype)
    self.assertArraysEqual(jax.jit(kernel)(), x)


class MosaicGpuDialectSm90ATest(Sm90ATestCase, jtu.JaxTestCase):

  @parameterized.named_parameters(
      (
          f"swizzle={int(swizzle)}_{transpose_lhs=}_{transpose_rhs=}_{lhs_in_registers=}",
          swizzle,
          transpose_lhs,
          transpose_rhs,
          lhs_in_registers,
      )
      for swizzle in mgpu_dialect.SwizzlingMode
      for transpose_lhs in [False, True]
      for transpose_rhs in [False, True]
      for lhs_in_registers in [False, True]
  )
  def test_wgmma_kernel_with_tma(
      self, swizzle, transpose_lhs, transpose_rhs, load_a_in_registers
  ):
    if swizzle == mgpu_dialect.SwizzlingMode.kNoSwizzle:
      self.skipTest("No swizzle is not supported by wgmma")

    if transpose_lhs or transpose_rhs:
      self.skipTest("Transposes are not supported by transform inference yet.")

    swizzle_elems = swizzle // np.dtype(jnp.bfloat16).itemsize
    tiling_m, tiling_n, tiling_k = 64, swizzle_elems, swizzle_elems

    groups_m, groups_n, groups_k = 4, 1, 1
    m, n, k = groups_m * tiling_m, groups_n * tiling_n, groups_k * tiling_k

    lhs_shape = (k, m) if transpose_lhs else (m, k)
    rhs_shape = (n, k) if transpose_rhs else (k, n)
    out_shape = (m, n)

    def matmul(
        ctx: launch_context.LaunchContext,
        lhs_gmem_ref: ir.Value,
        rhs_gmem_ref: ir.Value,
        result_gmem_ref: ir.Value,
        smem: list[ir.Value],
    ):
      del ctx
      lhs_smem_ref, rhs_smem_ref, result_smem_ref, tma_barrier = smem
      dialect_barrier = tma_barrier.as_barrier_memref()

      operand_elt_type = ir.MemRefType(lhs_gmem_ref.type).element_type
      bytes_a = utils.bytewidth(operand_elt_type) * math.prod(lhs_shape)
      bytes_b = utils.bytewidth(operand_elt_type) * math.prod(rhs_shape)

      mgpu_dialect.arrive_expect_tx(
          barrier=dialect_barrier,
          expect_tx=bytes_a + bytes_b,
      )

      zero_i32 = arith.constant(ir.IntegerType.get_signless(32), 0)
      # GMEM -> SMEM
      mgpu_dialect.async_load(
          source=lhs_gmem_ref,
          destination=lhs_smem_ref,
          barrier=dialect_barrier,
          indices=[zero_i32] * len(lhs_shape),
          slice_lengths=lhs_shape,
          collective=ir.ArrayAttr.get([]),
      )
      mgpu_dialect.async_load(
          source=rhs_gmem_ref,
          destination=rhs_smem_ref,
          barrier=dialect_barrier,
          indices=[zero_i32] * len(rhs_shape),
          slice_lengths=rhs_shape,
          collective=ir.ArrayAttr.get([]),
      )

      parities = memref.load(tma_barrier.barrier_ref.phases, [])
      parity, _ = tma_barrier.update_parities(parities)
      mgpu_dialect.wait(dialect_barrier, parity)

      # Computation
      shape_result = ir.MemRefType(result_gmem_ref.type).shape
      result_elt_type = ir.MemRefType(result_gmem_ref.type).element_type
      acc_elt_type = ir.F32Type.get()
      acc_type = ir.VectorType.get(shape_result, acc_elt_type)
      zero_acc = arith.constant(
          result_elt_type, ir.FloatAttr.get(acc_elt_type, 0.0)
      )
      accumulator = vector.splat(acc_type, zero_acc)

      if transpose_lhs:
        lhs_smem_ref = utils.memref_transpose(lhs_smem_ref, (1, 0))
      if transpose_rhs:
        rhs_smem_ref = utils.memref_transpose(rhs_smem_ref, (1, 0))

      zero_index = arith.constant(ir.IndexType.get(), 0)
      if load_a_in_registers:
        # SMEM -> Registers
        lhs_ty = ir.VectorType.get(lhs_shape, operand_elt_type)
        zero_vector_indices = [zero_index] * len(lhs_shape)
        lhs_operand = vector.load(lhs_ty, lhs_smem_ref, zero_vector_indices)
      else:
        lhs_operand = lhs_smem_ref

      result = mgpu_dialect.wgmma(
          accumulator,
          lhs_operand,
          rhs_smem_ref,
      )

      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)

      # Registers -> SMEM
      vector.store(result, result_smem_ref, [zero_index] * len(shape_result))

      # SMEM -> GMEM
      mgpu_dialect.async_store(
          source=result_smem_ref,
          destination=result_gmem_ref,
          indices=[zero_i32, zero_i32],
          slice_lengths=shape_result,
      )
      nvvm.cp_async_bulk_wait_group(0)

    operand_type = jnp.bfloat16
    acctype = jnp.float32
    lhs_jax_shape = jax.ShapeDtypeStruct(lhs_shape, operand_type)
    rhs_jax_shape = jax.ShapeDtypeStruct(rhs_shape, operand_type)
    result_jax_shape = jax.ShapeDtypeStruct(out_shape, acctype)
    kernel = mgpu.as_gpu_kernel(
        matmul,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(lhs_jax_shape, rhs_jax_shape),
        out_shape=result_jax_shape,
        smem_scratch_shape=[
            lhs_jax_shape,
            rhs_jax_shape,
            result_jax_shape,
            core.TMABarrier(1),
        ],
        thread_semantics=mgpu.LoweringSemantics.Warpgroup,
    )

    prng_key = jax.random.key(1234)
    k0, k1 = jax.random.split(prng_key, 2)

    x = jax.random.randint(k0, lhs_shape, 0, 2).astype(operand_type)
    y = jax.random.randint(k1, rhs_shape, 0, 2).astype(operand_type)

    transpose = lambda x, t: x.T if t else x
    self.assertArraysAllClose(
        jax.jit(kernel)(x, y),
        np.matmul(
            transpose(x, transpose_lhs),
            transpose(y, transpose_rhs)
        ),
        atol=0,
        rtol=0,
    )


class UtilsTest(TestCase):
  @parameterized.parameters(
      (1,),
      (-1,),
      (slice(2), slice(3),),
      (slice(1), slice(1, 3)),
      (slice(-2, 0),),
      (slice(-2, -1),),
      *([(utils.DynamicSlice(0, 2),)] if HAS_MOSAIC_GPU else []),
  )
  def test_parse_indices(self, *indices):
    # We are simply making sure this does not raise.
    _, _, _ = utils.parse_indices(indices, (2, 3, 4))

  @parameterized.parameters(
      (42,),
      (-42,),
      (slice(42),),
      (slice(0, 42),),
      (slice(-42, 0),),
      (slice(-4, -42),),
      *([(utils.DynamicSlice(0, 4),)] if HAS_MOSAIC_GPU else []),
  )
  def test_parse_indices_oob(self, indices):
    with self.assertRaisesRegex(IndexError, "out of bounds"):
      utils.parse_indices(indices, (2, 3, 4))

  @jtu.thread_unsafe_test()  # Modifies ``os.environ``.
  def test_assert(self):
    if cf is None:
      self.skipTest("``cf`` is not available")

    def kernel(ctx: mgpu.LaunchContext, x_ref, out, scratch) -> None:
      del ctx, out  # Unused.
      # TODO(b/408271232): Use a False condition once the bug is fixed.
      x = mgpu.FragmentedArray.load_strided(x_ref)
      cond = x.reduce("add", 0, *scratch) != 42.0
      cf.assert_(cond.registers.item(), "OOOPS")

    f = mgpu.as_gpu_kernel(
        kernel,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        in_shape=(jax.ShapeDtypeStruct((128,), jnp.float32),),
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        smem_scratch_shape=(jax.ShapeDtypeStruct((4,), jnp.float32),),
    )

    with jtu.set_env(MOSAIC_GPU_DUMP_SASS="1"), self.capture_stdout() as sass:
      jax.block_until_ready(f(jnp.ones((128,), jnp.float32)))

    # SASS doesn't seem to include the assertion message, so we are just
    # checking that __assertfail appears in the symbol table for the kernel.
    self.assertIn("__assertfail", sass())


class SerializationTest(absltest.TestCase):

  def test_pass_is_registered(self):
    ctx = mlir.make_ir_context()
    ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx):
      module = ir.Module.create()
      pipeline = passmanager.PassManager.parse(
          "builtin.module(mosaic_gpu-serde{serialize=true})",
          ctx,
      )
      pipeline.run(module.operation)


class ApiTest(TestCase):

  def test_inout(self):
    def kernel(ctx, src, inout, dst, smem):
      val = memref.load(inout, [])
      gpu.barrier()
      new_val = arith.constant(ir.IntegerType.get_signless(32), 42)
      memref.store(new_val, inout, [])
      x = mgpu.FragmentedArray.load_strided(src, is_signed=True)
      (x + val).store_untiled(dst)
    x = jnp.arange(128, dtype=jnp.int32)
    y = jnp.asarray(2.0, dtype=jnp.int32)
    kernel = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, (), inout_shape=y,
    )
    xo, yo = kernel(x, y)
    np.testing.assert_array_equal(xo, x + 2.0)
    np.testing.assert_array_equal(yo, jnp.asarray(42, dtype=jnp.int32))


if hp is not None:
  @hps.composite
  def tiled_layouts(
      draw, initial_tile, vector_transfer: bool = False
  ) -> fa.TiledLayout:
    assert all(t.bit_count() == 1 for t in initial_tile)
    assert math.prod(initial_tile) >= 128
    tiles = [initial_tile]
    dim_offset = len(initial_tile)
    warp_dim = fa.Replicated(4)
    if draw(hps.booleans()):
      warp_dim = draw(
          hps.sampled_from(
              [i for i, t in enumerate(tiles[-1]) if t % 4 == 0]
          )
      )
      warp_tile = list(tiles[-1])
      warp_tile[warp_dim] //= 4
      warp_dim += dim_offset
      tiles.append(warp_tile)
      dim_offset += len(tiles[-1])
    lane_dims = [fa.Replicated(2) if draw(hps.booleans()) else None for _ in range(5)]
    for i, dim in enumerate(lane_dims):
      if isinstance(dim, fa.Replicated):
        continue
      lane_dim = draw(hps.sampled_from(
          [i for i, t in enumerate(tiles[-1]) if t % 2 == 0]
      ))
      lane_tile = list(tiles[-1])
      lane_tile[lane_dim] //= 2
      lane_dims[i] = dim_offset + lane_dim
      tiles.append(lane_tile)
      dim_offset += len(lane_tile)
    # Permute lane dims so that they don't always partition the data in order.
    lane_dims = draw(hps.permutations(lane_dims))
    if vector_transfer:
      min_vector_dim = len(tiles[-1]) - 1
    else:
      min_vector_dim = 0
    vector_dim = draw(hps.integers(min_vector_dim, len(tiles[-1]) - 1))
    vector_size = 2 ** draw(
        hps.integers(0, tiles[-1][vector_dim].bit_length() - 1)
    )
    vector_tile = list(tiles[-1])
    assert vector_tile[vector_dim] % vector_size == 0
    vector_tile[vector_dim] //= vector_size
    tiles.append(vector_tile)
    dim_offset += len(vector_tile)
    vector_dim += dim_offset
    dim_offset += len(vector_tile)  # This is the remainder after tiling!

    if not isinstance(warp_dim, fa.Replicated):
      warp_dim = warp_dim - dim_offset
    lane_dims = tuple(
        d if isinstance(d, fa.Replicated) else d - dim_offset
        for d in lane_dims
    )
    vector_dim = vector_dim - dim_offset
    return fa.TiledLayout(
        tiling=fa.Tiling(tuple(map(tuple, tiles))),
        warp_dim=warp_dim,
        lane_dims=lane_dims,
        vector_dim=vector_dim,
    )

  @hps.composite
  def shape_and_tiled_layout(
      draw, vector_transfer: bool = False
  ) -> tuple[tuple[int, ...], fa.TiledLayout]:
    rank = draw(hps.integers(2, 3))
    initial_tile = tuple(
        draw(hps.sampled_from([1, 2, 4, 8, 16, 32, 64, 128]))
        for _ in range(rank)
    )
    hp.assume(128 <= math.prod(initial_tile) < 128 * 32)
    shape = tuple(t * draw(hps.integers(1, 5)) for t in initial_tile)
    hp.assume(math.prod(shape) <= 128 * 128)
    layout = draw(tiled_layouts(initial_tile, vector_transfer=vector_transfer))
    return shape, layout

  class HypothesisTest(TestCase):

    def test_reduce(self):
      @hps.composite
      def strategy(draw):
        shape, layout = draw(shape_and_tiled_layout(vector_transfer=True))
        rank = len(shape)
        reduced_dims = draw(hps.sets(hps.integers(0, rank - 1), min_size=1))
        return shape, layout, tuple(reduced_dims)

      @hp.given(strategy())
      def run(args):
        shape, layout, reduced_dims = args
        out_shape = list(shape)
        for d in sorted(reduced_dims, reverse=True):
          del out_shape[d]
        def kernel(ctx, src, dst, scratch):
          arr = fa.FragmentedArray.load_untiled(src, layout=layout, optimized=False)
          arr.reduce("max", reduced_dims, scratch).store_untiled(dst, optimized=False)
        x = jax.random.normal(jax.random.key(1234), shape, jnp.float32)
        out_type = jax.ShapeDtypeStruct(out_shape, jnp.float32)
        scratch_type = jax.ShapeDtypeStruct((2048,), jnp.float32)
        hp.assume(layout.vector_length <= 16)  # Otherwise we run out of scratch
        try:
          result = mgpu.as_gpu_kernel(
              kernel, (1, 1, 1), (128, 1, 1), x, out_type, scratch_type
          )(x)
        except NotImplementedError:
          hp.assume(False)
          return
        np.testing.assert_array_equal(result, x.max(reduced_dims))
      run()

    def test_slice(self):
      i32 = ir.IntegerType.get_signless(32)
      index = ir.IndexType.get()

      @hps.composite
      def strategy(draw):
        shape, layout = draw(shape_and_tiled_layout(vector_transfer=True))
        tiling = layout.base_tile_shape
        tiled_shape = mgpu.tile_shape(shape, tiling)[:len(shape)]
        def draw_slice(size, tile):
          start = draw(hps.integers(0, size - 1))
          length = draw(hps.integers(1, size - start))
          return slice(start * tile, (start + length) * tile)
        slices = tuple(map(draw_slice, tiled_shape, tiling))
        return shape, layout, slices

      basic_slices = (slice(128, 256), slice(16, 16 + 32))
      @hp.given(strategy())
      @hp.example(((256, 256), fa.WGMMA_LAYOUT, basic_slices))
      @hp.example(((256, 256), tcgen05.LAYOUT, basic_slices))
      @hp.example(((256, 256), tcgen05.TMEM_NATIVE_LAYOUT, basic_slices))
      def run(args):
        shape, layout, slices = args
        def kernel(ctx, dst, _):
          def linear_index(*idxs):
            total = arith.constant(index, 0)
            stride = 1
            for i, size in zip(idxs[::-1], shape[::-1]):
              total = arith.addi(total, arith.muli(i, c(stride, index)))
              stride *= size
            return arith.index_cast(i32, total)
          x = mgpu.FragmentedArray.build(
              shape, layout, linear_index, is_signed=True
          )
          x[slices].store_untiled(dst, optimized=False)

        slice_shape = tuple(len(range(size)[s]) for s, size in zip(slices, shape))
        out_shape = jax.ShapeDtypeStruct(shape=slice_shape, dtype=jnp.int32)
        result = mgpu.as_gpu_kernel(
            kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
        )()
        iota = np.arange(np.prod(shape), dtype=jnp.int32).reshape(*shape)
        np.testing.assert_array_equal(result, iota[slices])
      run()


if __name__ == "__main__":
  absltest.main(argv=["python"], testLoader=jtu.JaxTestLoader())
