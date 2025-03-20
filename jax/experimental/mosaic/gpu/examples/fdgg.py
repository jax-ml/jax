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

import itertools

import jax
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import c, ds
from jax.experimental.mosaic.gpu import tcgen05
from jax.experimental.mosaic.gpu import profiler
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from cuda.bindings.runtime import cudaDeviceGetAttribute, cudaDeviceAttr, cudaGetDeviceCount, cudaError_t


TMA_WARP = 1
MMA_WARP = 0


def get_sm_count():
  err, value = cudaGetDeviceCount()
  if err != cudaError_t.cudaSuccess:
    raise RuntimeError(err)
  assert value > 0
  err, value = cudaDeviceGetAttribute(cudaDeviceAttr.cudaDevAttrMultiProcessorCount, 0)
  if err != cudaError_t.cudaSuccess:
    raise RuntimeError(err)
  return value


def worker_for(worker_id, worker_count, work_count):
  def wrapper(f):
    for_op = scf.ForOp(worker_id, work_count, worker_count, [])
    with ir.InsertionPoint(for_op.body):
      f(for_op.induction_variable)
      scf.yield_([])
  return wrapper


def bytecount(shape, dtype):
  return int(np.prod(shape) * dtype.dtype.itemsize)


def build_kernel(
    m, n, k,
    cta_count: int,
    expert_count: int,
    group_chunk_count: int,
    expert_n: int,
    tile_m: int = 128,
    tile_n: int = 128,
    max_concurrent_steps: int = 2,
    in_dtype = jnp.float16,
):
  i1 = ir.IntegerType.get_signless(1)
  i32 = ir.IntegerType.get_signless(32)
  index = ir.IndexType.get()

  swizzle = 128
  swizzle_elems = tile_k = swizzle // 2
  tiling = (8, swizzle_elems)

  k_loop_iter = k // tile_k
  max_concurrent_steps = min(max_concurrent_steps, k_loop_iter)

  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  if n % expert_count != 0:
    raise ValueError(f"{n=} must be divisible by {expert_count=}")

  m_tile_count = m // tile_m
  n_tile_count = n // tile_n
  tile_count = m_tile_count * n_tile_count

  def kernel(ctx, a, w, group_offsets, expert_ids, d, smem):
    ((a_smem, b_smem), d_smem), barriers, mma_done_barrier, acc = smem
    (ab_full_barriers, ab_empty_barriers) = barriers

    warp_idx = mgpu.warp_idx(sync=True)
    is_warp_leader = nvvm.elect_sync(i1)
    is_leader_of = lambda i: arith.andi(arith.cmpi(arith.CmpIPredicate.eq, warp_idx, c(i, i32)), is_warp_leader)

    worker_id = gpu.block_id(gpu.Dimension.x)
    worker_count = c(cta_count, index)
    work_count = c(tile_count, index)
    @worker_for(worker_id=worker_id, worker_count=worker_count, work_count=work_count)
    def body(work_id):
      m_idx = arith.divui(work_id, c(n_tile_count, index))
      n_idx = arith.remui(work_id, c(n_tile_count, index))
      m_start = arith.muli(m_idx, c(tile_m, index))
      n_start = arith.muli(n_idx, c(tile_n, index))

      local_work_id = arith.divui(work_id, worker_count)
      get_persistent_ki = lambda ki: arith.addi(ki, arith.muli(local_work_id, c(k_loop_iter,index)))
      with mgpu.when(is_leader_of(TMA_WARP)):
        @mgpu.fori(c(k_loop_iter, index), None)
        def _tma_body(ki, _):
          persistent_ki = get_persistent_ki(ki)
          slot = arith.remui(persistent_ki, c(max_concurrent_steps, index))
          # TODO(apaszke): Use a predicate instead of a conditional.
          with mgpu.when(arith.cmpi(arith.CmpIPredicate.uge, persistent_ki, c(max_concurrent_steps, index))):
            ab_empty_barriers[slot].wait()
          full_barrier = ab_full_barriers[slot]
          full_barrier.arrive_expect_tx(
              bytecount((tile_m, tile_k), in_dtype) + bytecount((tile_n, tile_k), in_dtype)
          )
          k_start = arith.muli(ki, c(tile_k, index))
          common_args = dict(
              swizzle=swizzle,
              barrier=full_barrier,
              arrive=False,
              uniform=False,
          )
          ctx.async_copy(
              src_ref=a,
              dst_ref=mgpu.memref_slice(a_smem, slot),
              gmem_slice=(ds(m_start, tile_m), ds(k_start, tile_k)),
              gmem_transform=mgpu.TileTransform(tiling),
              **common_args,
          )
          ctx.async_copy(
              src_ref=w,
              dst_ref=mgpu.memref_slice(b_smem, slot),
              gmem_slice=(ds(n_start, tile_n), ds(k_start, tile_k)),
              gmem_transform=mgpu.TileTransform(tiling),
              **common_args,
          )

      with mgpu.when(is_leader_of(MMA_WARP)):
        @mgpu.fori(c(k_loop_iter, index), arith.constant(i1, 0))
        def _mma_body(ki, accumulate):
          persistent_ki = get_persistent_ki(ki)
          slot = arith.remui(persistent_ki, c(max_concurrent_steps, index))
          ab_full_barriers[slot].wait()
          tcgen05.mma(
              acc,
              mgpu.memref_slice(a_smem, slot),
              mgpu.memref_transpose(mgpu.memref_slice(b_smem, slot), (1, 0, 3, 2)),
              a_swizzle=swizzle,
              b_swizzle=swizzle,
              accumulate=accumulate,
          )
          accumulate = arith.constant(i1, 1)
          tcgen05.commit_arrive(ab_empty_barriers[slot].get_ptr(), ctx=ctx)
          return accumulate
        tcgen05.commit_arrive(mma_done_barrier.get_ptr(), ctx=ctx)

      gpu.barrier()
      mma_done_barrier.wait(for_tensor_core=True)

      acc[:].astype(ir.F16Type.get()).store_tiled(d_smem, swizzle=128)
      mgpu.commit_shared()
      ctx.async_copy(
          src_ref=d_smem,
          dst_ref=d,
          gmem_slice=(ds(m_start, tile_m), ds(n_start, tile_n)),
          gmem_transform=mgpu.TileTransform((128, swizzle_elems)),
          swizzle=swizzle,
      )
      ctx.await_async_copy(0)

  compute_buffers = (
    jax.ShapeDtypeStruct(
        mgpu.tile_shape((max_concurrent_steps, tile_m, tile_k), tiling),
        jnp.float16),
    jax.ShapeDtypeStruct(
         mgpu.tile_shape((max_concurrent_steps, tile_n, tile_k), tiling),
         jnp.float16),
  )
  epilogue_buffer = jax.ShapeDtypeStruct(
      mgpu.tile_shape((tile_m, tile_n), (128, swizzle_elems)),
      jnp.float16)
  smem_buffers = mgpu.Union([compute_buffers, epilogue_buffer])
  smem = (
      smem_buffers,
      [mgpu.Barrier(arrival_count=1, num_barriers=max_concurrent_steps)] * 2,
      mgpu.Barrier(arrival_count=1),
      mgpu.TMEM((128, tile_n), jnp.float32),
  )

  f = mgpu.as_gpu_kernel(
      kernel,
      (cta_count, 1, 1),  # persistent kernel
      (128, 1, 1),
      (
          jax.ShapeDtypeStruct((m, k), jnp.float16),
          jax.ShapeDtypeStruct((n, k), jnp.float16),
          jax.ShapeDtypeStruct((group_chunk_count+1,), jnp.int32),  # group_offsets
          jax.ShapeDtypeStruct((group_chunk_count+0,), jnp.int32),  # expert_ids
      ),
      jax.ShapeDtypeStruct((m, expert_n), jnp.float16),
      smem,
  )

  return f


def generate_group_sizes(expert_count, token_count, key1, key2):
  v = jr.truncated_normal(key1, -2., 2., expert_count) + 2.
  expert_probs = v / jnp.sum(v)
  expert_assignment = jr.choice(key2, expert_count, (token_count,), p=expert_probs)
  group_sizes = jnp.bincount(expert_assignment)
  return group_sizes


def get_schedule(group_sizes, n, tile_m, tile_n):
  assert n % tile_n == 0
  group_sizes = group_sizes.tolist()
  chunks = []
  expert_ids = []
  for i, g in enumerate(group_sizes):
    while g > tile_m:
      expert_ids.append(i)
      chunks.append(tile_m)
      g -= tile_m
    expert_ids.append(i)
    chunks.append(g)

  chunks = jnp.array(chunks)
  group_offsets = jnp.cumulative_sum(chunks, include_initial=True)
  expert_ids = jnp.array(expert_ids)

  return group_offsets, expert_ids


def ref(activations, weights, group_sizes, expert_n):
  group_offsets = np.cumulative_sum(group_sizes, include_initial=True).tolist()
  results = []
  for i, (a, b) in enumerate(zip(group_offsets, group_offsets[1:])):
    group = activations[a:b, :]
    expert = weights[i*expert_n:(i+1)*expert_n, :]
    results.append(group @ expert.T)
  return jnp.concatenate(results)


def main(unused_argv):
  m = 1024  # seqlen
  k = 512
  expert_count = 8  # expert count
  expert_n = 512
  n = expert_count * expert_n
  in_dtype = jnp.float16
  # TODO(aportnoy@nvidia.com) this leads to low occupancy, so unless
  # each CTA is using tensor cores at all times, we are leaving perf
  # on the table. A solution would be to separate the epilogue into a
  # warpgroup of its own, and double buffer the accumulator, so that
  # the next work item can run while we are storing the previous one
  # out.
  cta_count = get_sm_count()
  ka, kb, ke1, ke2 = jr.split(jr.key(0), 4)
  activations = jr.normal(key=ka, shape=(m, k), dtype=in_dtype)
  weights     = jr.normal(key=kb, shape=(n, k), dtype=in_dtype)
  group_sizes = generate_group_sizes(expert_count, m, ke1, ke2)
  # TODO(aportnoy@nvidia.com) test different tile sizes
  tile_m = 128
  tile_n = 128  # 256, 512
  # TODO(aportnoy@nvidia.com) move this computation into the kernel
  group_offsets, expert_ids = get_schedule(group_sizes, n, tile_m, tile_n)
  group_chunk_count = len(expert_ids)

  # TODO(aportnoy@nvidia.com) test different stage counts
  max_concurrent_steps = 2  # 4, 5, 6
  with mlir.make_ir_context(), ir.Location.unknown():
    f = build_kernel(
      m, n, k,
      cta_count=cta_count,
      expert_count=expert_count,
      group_chunk_count=group_chunk_count,
      expert_n=expert_n,
      tile_m=tile_m,
      tile_n=tile_n,
      max_concurrent_steps=max_concurrent_steps,
      in_dtype=in_dtype,
    )
  d, runtime = profiler.measure(f, mode='cupti')(activations, weights, group_offsets, expert_ids)
  tflops = float(2 * k * m * n) / (runtime / 1e3) / 1e12
  print(f"Kernel:    {runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")


  # TODO(aportnoy@nvidia.com) compute reference and check correctness
  d_ref = ref(activations, weights, group_sizes, expert_n)

  #ref_tflops = float(2 * k * m * n) / (ref_runtime / 1e3) / 1e12
  np.testing.assert_allclose(d, d_ref, atol=1e-3, rtol=1e-3)
  #print(f"Reference: {ref_runtime * 1000:.1f} us = {ref_tflops:.1f} TFLOPS")


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
