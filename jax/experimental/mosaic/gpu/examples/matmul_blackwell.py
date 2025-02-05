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
"""Matmul kernel for Blackwell."""

import jax
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import nvvm
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import c, ds, utils
from jax.experimental.mosaic.gpu import tcgen05
import jax.numpy as jnp
import jax.random as jr
import numpy as np


BLACKWELL_MMA_FP16_K = 16
TMA_WARP = 1
MMA_WARP = 0


def bytecount(shape, dtype):
  return int(np.prod(shape) * dtype.dtype.itemsize)


def build_kernel(
    m, n, k,
    tile_m: int = 128,
    tile_n: int = 128,
):
  i1 = ir.IntegerType.get_signless(1)
  i32 = ir.IntegerType.get_signless(32)
  f32 = ir.F32Type.get()
  index = ir.IndexType.get()
  ptr6 = ir.Type.parse("!llvm.ptr<6>")  # TMEM

  swizzle = 128
  tile_k = 64  # TODO(apaszke): I think we need to tile TMA to change this.
  in_dtype = jnp.float16
  k_loop_iter = k // tile_k

  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")

  def kernel(ctx, a, b, d, smem):
    # TODO(apaszke): Use more SMEM slots to avoid oversynchronizing warps.
    a_smem, b_smem, d_smem, barriers, tmem_addr = smem
    (ab_full_barrier, ab_empty_barrier, mma_done_barrier) = barriers

    warp_idx = mgpu.warp_idx(sync=True)
    warp_leader = nvvm.elect_sync(i1)

    is_warp = lambda i: arith.cmpi(arith.CmpIPredicate.eq, warp_idx, c(i, i32))

    m_start = arith.muli(gpu.block_id(gpu.Dimension.y), c(tile_m,index))
    n_start = arith.muli(gpu.block_id(gpu.Dimension.x), c(tile_n,index))

    with mgpu.when(arith.andi(is_warp(TMA_WARP), warp_leader)):
      @mgpu.fori(c(k_loop_iter, index), None)
      def _tma_body(ki, _):
        # TODO(apaszke): Use a predicate instead of a conditional.
        with mgpu.when(arith.cmpi(arith.CmpIPredicate.ugt, ki, c(0, index))):
          ab_empty_barrier.wait()
        ab_full_barrier.arrive_expect_tx(
            bytecount((tile_m, tile_k), in_dtype) + bytecount((tile_n, tile_k), in_dtype)
        )
        k_start = arith.muli(ki, c(tile_k, index))
        common_args = dict(
            swizzle=swizzle, barrier=ab_full_barrier, arrive=False, uniform=False,
        )
        ctx.async_copy(
            src_ref=a,
            dst_ref=a_smem,
            gmem_slice=(ds(m_start, tile_m), ds(k_start, tile_k)),
            **common_args,
        )
        ctx.async_copy(
            src_ref=b,
            dst_ref=b_smem,
            gmem_slice=(ds(n_start, tile_n), ds(k_start, tile_k)),
            **common_args,
        )

    with mgpu.when(is_warp(MMA_WARP)):
      tmem_addr_addr = utils.memref_ptr(tmem_addr, memory_space=3)
      tcgen05.tmem_alloc(tmem_addr_addr, tile_n)
      tcgen05.tmem_relinquish_alloc_permit()
      with mgpu.when(warp_leader):
        tmem_addr_value = llvm.load(ptr6, tmem_addr_addr)
        idesc = tcgen05.create_instr_descriptor(
            m=tile_n, n=tile_n, acc_dtype=jnp.float32, input_dtype=in_dtype
        )
        @mgpu.fori(c(k_loop_iter, index), arith.constant(i1, 0))
        def _mma_body(ki, accumulate):
          adesc = tcgen05.create_smem_descriptor(
            a_smem, leading_byte_offset=16, stride_byte_offset=1024, swizzle=swizzle)
          bdesc = tcgen05.create_smem_descriptor(
            b_smem, leading_byte_offset=16, stride_byte_offset=1024, swizzle=swizzle)
          ab_full_barrier.wait()

          # TODO(apaszke): Abstract this into a function.
          assert tile_k % BLACKWELL_MMA_FP16_K == 0
          def smem_descriptor_increment_address(desc, nbytes):
            i64 = ir.IntegerType.get_signless(64)
            return arith.addi(desc, arith.shrui(c(nbytes,i64), c(4,i64)))
          for _ in range(tile_k // BLACKWELL_MMA_FP16_K):
            tcgen05.mma("f16", 1, tmem_addr_value, adesc, bdesc, idesc, enable_input_d=accumulate)
            accumulate = arith.constant(i1, 1)
            adesc = smem_descriptor_increment_address(
                adesc, BLACKWELL_MMA_FP16_K * 2
            )
            bdesc = smem_descriptor_increment_address(
                bdesc, BLACKWELL_MMA_FP16_K * 2
            )

          is_last_iter = arith.cmpi(
              arith.CmpIPredicate.eq, ki, c(k_loop_iter - 1, index)
          )
          barrier_ptr = arith.select(
              is_last_iter, mma_done_barrier.get_ptr(), ab_empty_barrier.get_ptr()
          )
          tcgen05.commit_arrive(barrier_ptr)
          return accumulate

    gpu.barrier()
    mma_done_barrier.wait()

    tmem_ref = tcgen05.TMEMRef.from_alloc(tmem_addr, tcgen05.TMEMLayout.D, tile_n, f32)
    tmem_ref[:].astype(ir.F16Type.get()).store_tiled(d_smem, swizzle=128)
    mgpu.commit_shared()
    ctx.async_copy(
        src_ref=d_smem,
        dst_ref=d,
        gmem_slice=(ds(m_start, tile_m), ds(n_start, tile_n)),
        gmem_transform=mgpu.TileTransform((128, 64)),
        swizzle=128,
    )
    ctx.await_async_copy(0)

  smem = (
      jax.ShapeDtypeStruct((tile_m, tile_k), jnp.float16),
      jax.ShapeDtypeStruct((tile_n, tile_k), jnp.float16),
      jax.ShapeDtypeStruct(mgpu.tile_shape((tile_m, tile_n), (128, 64)), jnp.float16),
      [mgpu.Barrier(arrival_count=1)] * 3,
      jax.ShapeDtypeStruct((1,), np.uint32),  # TMEM address
  )
  return mgpu.as_gpu_kernel(
      kernel,
      (n // tile_n, m // tile_m, 1),
      (128, 1, 1),
      (
          jax.ShapeDtypeStruct((m, k), jnp.float16),
          jax.ShapeDtypeStruct((n, k), jnp.float16),
      ),
      jax.ShapeDtypeStruct((m, n), jnp.float16),
      smem,
  )


def main(unused_argv):
  m_tile = 128
  n_tile = 128
  k_tile = 64
  m = 16*m_tile
  n = 16*n_tile
  k = 16*k_tile

  ka, kb = jr.split(jr.key(0), 2)
  a = jr.normal(key=ka, shape=(m, k), dtype=jnp.float16)
  b = jr.normal(key=kb, shape=(n, k), dtype=jnp.float16)

  with mlir.make_ir_context(), ir.Location.unknown():
    f = build_kernel(m, n, k, tile_m=m_tile, tile_n=n_tile)
  y = f(a, b).block_until_ready()

  ref = np.asarray(a) @ np.asarray(b).T
  np.testing.assert_allclose(y, ref, atol=1e-3, rtol=1e-3)
  print("OK!")


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
