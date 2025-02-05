# Copyright 2025 The JAX Authors. All Rights Reserved.
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

import contextlib

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import ShapeDtypeStruct as SDS
from jax._src.interpreters import mlir
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import c, utils, create_descriptor, ds
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import vector
from jaxlib.mlir.dialects import scf

# TODO(andportnoy) refactor into a single function for Hopper/Blackwell?
def create_smem_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | None,
    memory_space: int | None = None,
):
  i64 = ir.IntegerType.get_signless(64)

  hopper_desc = create_descriptor(
    memref_arg, leading_byte_offset, stride_byte_offset, swizzle, memory_space=3)
  blackwell_bit = arith.shli(c(1,i64), c(46,i64))
  blackwell_desc = llvm.or_(hopper_desc, blackwell_bit)
  return blackwell_desc


def smem_descriptor_increment_address(desc, nbytes):
  i64 = ir.IntegerType.get_signless(64)
  return arith.addi(desc, arith.shrui(c(nbytes,i64), c(4,i64)))


def create_blackwell_fp16_mma_descriptor(
    m: int,
    n: int,
    dtype,
    transpose_b: bool = False,
):
  i32 = ir.IntegerType.get_signless(32)
  desc = c(0,i32)
  def fieldset(val, bit):
    field = arith.shli(c(val,i32), c(bit, i32))
    nonlocal desc
    desc = arith.ori(desc, field)
  # encoding is dependent on .kind::<foo>, below is for .kind::f16
  #  1: 0 - sparsity selector if sparsity is enabled
  #  2    - sparsity: dense = 0, sparse = 1
  #  3    - saturate for integer types, irrelevant for fp
  #  5: 4 - output dtype
  fieldset(1, 4)
  #  6    - reserved
  #  9: 7 - A dtype: f16 = 0, b16 = 1
  if dtype == jnp.bfloat16:
    fieldset(1, 7)
  # 12:10 - B dtype: f16 = 0, b16 = 1
  if dtype == jnp.bfloat16:
    fieldset(1, 10)
  # 13    - negate A
  # 14    - negate B
  # 15    - transpose A
  # 16    - transpose B
  if transpose_b:
    fieldset(1, 16)
  # 22:17 - N >> 3
  fieldset(n >> 3, 17)  # TODO validate field width
  # 23    - reserved
  # 28:24 - M >> 4
  fieldset(m >> 4, 24)
  # 29    - reserved
  # 31:30 - max shift under .ws: irrelevant here
  return desc

WARPSIZE = 32

m_tile = 128
n_tile = 128
k_tile = 64
m = 16*m_tile
n = 16*n_tile
k = 16*k_tile
# K = 16 is inherent to Blackwell MMA half precision instructions
blackwell_mma_fp16_k = 16
ashape = (m, k)  # k-major (row major)
ashape_tile = (m_tile, k_tile)
bshape = (n, k)  # k-major (column major)
bshape_tile = (n_tile, k_tile)
dshape = (m, n)  # n-major (row major)
in_dtype = jnp.bfloat16
ddtype = jnp.float32
grid = (n//n_tile, m//m_tile, 1)
block = (8*WARPSIZE, 1, 1)

def kernel(ctx, a, b, d, smem):
  i1 = ir.IntegerType.get_signless(1)
  i32 = ir.IntegerType.get_signless(32)
  f32 = ir.F32Type.get()
  index = ir.IndexType.get()
  ptr = ir.Type.parse("!llvm.ptr")
  ptr6 = ir.Type.parse("!llvm.ptr<6>")
  tidx = gpu.thread_id(gpu.Dimension.x)
  warpid = arith.shrui(tidx, c(5,index))
  tma_warp = 0
  mma_warp = 1
  # need a full aligned warpgroup, here it's warps 4-7
  ldtm_warp_range = (4, 8)

  @contextlib.contextmanager
  def only_warp(i):
    is_warp_i = arith.cmpi(arith.CmpIPredicate.eq, warpid, c(i,index))
    with ir.InsertionPoint(scf.IfOp(is_warp_i).then_block):
      yield
      scf.yield_([])

  @contextlib.contextmanager
  def only_warp_range(a, b):
    gea = arith.cmpi(arith.CmpIPredicate.uge, warpid, c(a,index))
    ltb = arith.cmpi(arith.CmpIPredicate.ult, warpid, c(b,index))
    predicate = arith.andi(gea, ltb)
    with ir.InsertionPoint(scf.IfOp(predicate).then_block):
      yield
      scf.yield_([])

  @contextlib.contextmanager
  def single_warp_thread():
    elected = nvvm.elect_sync(i1)
    with ir.InsertionPoint(scf.IfOp(elected).then_block):
      yield
      scf.yield_([])

  a_shared, b_shared, (ab_full_barrier, ab_empty_barrier, mma_done_barrier), tmem_addr = smem

  k_loop_iter = k//k_tile

  def bytecount(shape, dtype):
    return int(np.prod(shape) * dtype.dtype.itemsize)

  txcount = bytecount(ashape_tile, in_dtype) + bytecount(bshape_tile, in_dtype)
  with only_warp(tma_warp), single_warp_thread():
    # XXX TODO move loop iteration into MLIR, otherwise we force unroll
    for i in range(k_loop_iter):
      if i > 0:
        ab_empty_barrier.wait()
      ab_full_barrier.arrive_expect_tx(txcount)
      m_start = arith.muli(gpu.block_id(gpu.Dimension.y), c(m_tile,index))
      n_start = arith.muli(gpu.block_id(gpu.Dimension.x), c(n_tile,index))
      k_start = i*k_tile
      ctx.async_copy(
        src_ref=a,
        dst_ref=a_shared,
        gmem_slice=(ds(m_start, m_tile), ds(k_start,k_tile)),
        swizzle=128,
        barrier=ab_full_barrier,
        arrive=False,
        uniform=False,
      )
      ctx.async_copy(
        src_ref=b,
        dst_ref=b_shared,
        gmem_slice=(ds(n_start, n_tile), ds(k_start,k_tile)),
        swizzle=128,
        barrier=ab_full_barrier,
        arrive=False,
        uniform=False,
      )

  with only_warp(mma_warp):
    ncols = c(b_shared.type.shape[0], i32)
    tmem_addr_addr = utils.memref_ptr(tmem_addr, memory_space=3)
    nvvm.tcgen05_alloc(tmem_addr_addr, ncols)
    nvvm.tcgen05_relinquish_alloc_permit()
    accumulate = 0
    with single_warp_thread():
      tmem_addr_value = llvm.load(ptr6, tmem_addr_addr)
      idesc = create_blackwell_fp16_mma_descriptor(m_tile, n_tile, in_dtype)
      for i in range(k_loop_iter):
        adesc = create_smem_descriptor(
          a_shared, leading_byte_offset=16, stride_byte_offset=1024, swizzle=128)
        bdesc = create_smem_descriptor(
          b_shared, leading_byte_offset=16, stride_byte_offset=1024, swizzle=128)
        ab_full_barrier.wait()
        for _ in range(4):
          nvvm.tcgen05_mma("f16", "cta_1", tmem_addr_value, adesc, bdesc, idesc, enable_input_d=c(accumulate,i1))
          accumulate = 1
          adesc = smem_descriptor_increment_address(adesc, blackwell_mma_fp16_k*2)
          bdesc = smem_descriptor_increment_address(bdesc, blackwell_mma_fp16_k*2)
        last_iter = i == k_loop_iter-1
        barrier = mma_done_barrier if last_iter else ab_empty_barrier
        nvvm.tcgen05_commit_arrive(barrier.get_ptr())

  with only_warp_range(*ldtm_warp_range), ctx.named_region("LDTM"):
    mma_done_barrier.wait()
    tmem_ptr = llvm.inttoptr(ptr6, memref.load(tmem_addr, [c(0,index)]))
    # TODO automate type creation
    vector_i32 = nvvm.tcgen05_ld(ir.VectorType.get([n_tile], i32), shape="shape_32x32b", num=n_tile, tmem_addr=tmem_ptr)
    vector_f32 = vector.bitcast(ir.VectorType.get(vector_i32.type.shape, f32), vector_i32)
    wg_tidx = arith.andi(tidx, c((1 << 7)-1, index))  # tidx within warpgroup
    row = arith.addi(
      arith.muli(
        gpu.block_id(gpu.Dimension.y),
        c(m_tile,index)),
      wg_tidx)
    column = arith.muli(
      gpu.block_id(gpu.Dimension.x),
      c(n_tile,index))
    vector.store(vector_f32, d, [row, column])


if __name__ == '__main__':
  ka, kb = jr.split(jr.key(0), 2)
  a = jr.uniform(key=ka, shape=ashape, dtype=in_dtype)
  b = jr.uniform(key=kb, shape=bshape, dtype=in_dtype)

  asds_tile = SDS(ashape_tile, a.dtype)
  bsds_tile = SDS(bshape_tile, b.dtype)
  dsds = SDS(dshape, ddtype)
  tmem_addr = SDS((1,), np.uint32)

  smem = (asds_tile, bsds_tile, tuple(mgpu.Barrier(arrival_count=1) for _ in range(3)), tmem_addr)
  with mlir.make_ir_context(), ir.Location.unknown():
    f = mgpu.as_gpu_kernel(
        kernel,
        grid,
        block,
        (SDS(a.shape, a.dtype), SDS(b.shape, b.dtype)),
        dsds,
        smem
    )
  y = f(a, b)

  @jax.jit
  def ref_f(x, y):
    return jax.lax.dot_general(
        x,
        y,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)

  ref = ref_f(a, b)
  np.testing.assert_allclose(
      y.astype(jnp.float32), ref.astype(jnp.float32), atol=1e-3, rtol=1e-3
  )
