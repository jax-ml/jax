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
from jax._src.lib.mlir.dialects import nvvm
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import c, ds
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
    max_concurrent_steps: int = 2,
):
  i1 = ir.IntegerType.get_signless(1)
  i32 = ir.IntegerType.get_signless(32)
  f32 = ir.F32Type.get()
  index = ir.IndexType.get()

  swizzle = 128
  tile_k = swizzle // 2

  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")

  in_dtype = jnp.float16
  k_loop_iter = k // tile_k
  max_concurrent_steps = min(max_concurrent_steps, k_loop_iter)
  tma_tile_m = 128
  tma_tile_kn = 64

  def kernel(ctx, a, b, d, smem):
    a_smem, b_smem, d_smem, barriers, mma_done_barrier, acc = smem
    (ab_full_barriers, ab_empty_barriers) = barriers

    warp_idx = mgpu.warp_idx(sync=True)
    warp_leader = nvvm.elect_sync(i1)

    is_warp = lambda i: arith.cmpi(arith.CmpIPredicate.eq, warp_idx, c(i, i32))

    m_start = arith.muli(gpu.block_id(gpu.Dimension.y), c(tile_m,index))
    n_start = arith.muli(gpu.block_id(gpu.Dimension.x), c(tile_n,index))

    with mgpu.when(arith.andi(is_warp(TMA_WARP), warp_leader)):
      @mgpu.fori(c(k_loop_iter, index), None)
      def _tma_body(ki, _):
        slot = arith.remui(ki, c(max_concurrent_steps, index))
        # TODO(apaszke): Use a predicate instead of a conditional.
        with mgpu.when(arith.cmpi(arith.CmpIPredicate.uge, ki, c(max_concurrent_steps, index))):
          ab_empty_barriers[slot].wait()
        full_barrier = ab_full_barriers[slot]
        full_barrier.arrive_expect_tx(
            bytecount((tile_m, tile_k), in_dtype) + bytecount((tile_n, tile_k), in_dtype)
        )
        k_start = arith.muli(ki, c(tile_k, index))
        common_args = dict(
            swizzle=swizzle, barrier=full_barrier, arrive=False, uniform=False,
        )
        ctx.async_copy(
            src_ref=a,
            dst_ref=mgpu.memref_slice(a_smem, slot),
            gmem_slice=(ds(m_start, tile_m), ds(k_start, tile_k)),
            gmem_transform=mgpu.TileTransform((tma_tile_m, tma_tile_kn)),
            **common_args,
        )
        ctx.async_copy(
            src_ref=b,
            dst_ref=mgpu.memref_slice(b_smem, slot),
            gmem_slice=(ds(n_start, tile_n), ds(k_start, tile_k)),
            gmem_transform=(
                mgpu.TileTransform((tma_tile_kn, tma_tile_kn)),
                mgpu.TransposeTransform((1, 0, 2, 3)),
            ),
            **common_args,
        )

    with mgpu.when(arith.andi(is_warp(MMA_WARP), warp_leader)):
      with mgpu.when(warp_leader):
        @mgpu.fori(c(k_loop_iter, index), arith.constant(i1, 0))
        def _mma_body(ki, accumulate):
          slot = arith.remui(ki, c(max_concurrent_steps, index))
          ab_full_barriers[slot].wait()
          tcgen05.mma(
              acc,
              mgpu.memref_slice(a_smem, slot),
              mgpu.memref_transpose(mgpu.memref_slice(b_smem, slot), (0, 1, 3, 2)),
              a_swizzle=swizzle,
              b_swizzle=swizzle,
              accumulate=accumulate,
          )
          accumulate = arith.constant(i1, 1)
          is_last_iter = arith.cmpi(
              arith.CmpIPredicate.eq, ki, c(k_loop_iter - 1, index)
          )
          barrier_ptr = arith.select(
              is_last_iter,
              mma_done_barrier.get_ptr(),
              ab_empty_barriers[slot].get_ptr(),
          )
          tcgen05.commit_arrive(barrier_ptr)
          return accumulate

    gpu.barrier()
    mma_done_barrier.wait(for_tensor_core=True)

    acc[:].astype(ir.F16Type.get()).store_tiled(d_smem, swizzle=128)
    mgpu.commit_shared()
    # TODO(apaszke): Free up TMEM?
    ctx.async_copy(
        src_ref=d_smem,
        dst_ref=d,
        gmem_slice=(ds(m_start, tile_m), ds(n_start, tile_n)),
        gmem_transform=mgpu.TileTransform((128, 64)),
        swizzle=swizzle,
    )
    ctx.await_async_copy(0)

  # TODO(apaszke): Use a union for output SMEM.
  smem = (
      jax.ShapeDtypeStruct((max_concurrent_steps, *mgpu.tile_shape((tile_m, tile_k), (tma_tile_m, tma_tile_kn))), jnp.float16),
      jax.ShapeDtypeStruct((max_concurrent_steps, *mgpu.tile_shape((tile_k, tile_n), (tma_tile_kn, tma_tile_kn))), jnp.float16),
      jax.ShapeDtypeStruct(mgpu.tile_shape((tile_m, tile_n), (tma_tile_m, tma_tile_kn)), jnp.float16),
      [mgpu.Barrier(arrival_count=1, num_barriers=max_concurrent_steps)] * 2,
      mgpu.Barrier(arrival_count=1),
      mgpu.TMEM((128, tile_n), jnp.float32, tcgen05.TMEMLayout.D),
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
