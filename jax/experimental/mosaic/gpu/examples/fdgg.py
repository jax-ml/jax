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
"""Grouped GEMM kernel for Blackwell."""

from dataclasses import dataclass
import contextlib
import itertools
import math

import jax
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import c, ds
from jax.experimental.mosaic.gpu import tcgen05
from jax.experimental.mosaic.gpu import mma_utils
from jax.experimental.mosaic.gpu import profiler
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from cuda.bindings.runtime import cudaDeviceGetAttribute, cudaDeviceAttr, cudaGetDeviceCount, cudaError_t

WARPSIZE = 32
WARPGROUPSIZE = 4 * WARPSIZE
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
# Maximum x- or y-dimensionality of a block
MAX_WARPS_PER_BLOCK_DIM = 32


TMA_WARP = 1
MMA_WARP = 0


# TODO(andportnoy) move into tcgen05.py
def tmem_dealloc(tmem: ir.Value, ncols: int, collective: bool = False):
  num_cta = 2 if collective else 1
  return llvm.inline_asm(
    ir.Type.parse("!llvm.void"),
    [tmem],
    f"tcgen05.dealloc.cta_group::{num_cta}.sync.aligned.b32 $0, {ncols};",
    "r",
    has_side_effects=True,
  )


def get_sm_count():
  err, value = cudaGetDeviceCount()
  if err != cudaError_t.cudaSuccess:
    raise RuntimeError(err)
  assert value > 0
  err, value = cudaDeviceGetAttribute(cudaDeviceAttr.cudaDevAttrMultiProcessorCount, 0)
  if err != cudaError_t.cudaSuccess:
    raise RuntimeError(err)
  return value


def subgroup_for(work_id, work_id_group_end, worker_count):
  # Maybe use scf.while instead, this is a bit of a kludge to make
  # work_id both the induction variable and a loop-carried variable.
  # Here we simulate (using scf.for):
  #
  #  while work_id < work_id_group_end:
  #    ...
  #    work_id += worker_count
  def wrapper(f):
    step = worker_count
    for_op = scf.ForOp(
      work_id,  # lowerBound
      work_id_group_end,  # upperBound
      step,  # step
      [work_id],  # initArgs
    )
    with ir.InsertionPoint(for_op.body):
      f(for_op.induction_variable)
      scf.yield_([arith.addi(for_op.induction_variable, step)])
    final_work_id = for_op.results[0]
    return final_work_id
  return wrapper


def bytecount(shape, dtype):
  return int(np.prod(shape) * dtype.dtype.itemsize)
@dataclass
class Warp:
  x: int  # id of warp
  def __repr__(self):
    x = self.x
    return f"Warp({x=}, range={x*WARPSIZE}-{(x+1)*WARPSIZE-1})"


@dataclass
class Warpgroup:
  x: int  # id of first warp
  def __repr__(self):
    x = self.x
    return f"Warpgroup({x=}, range={x*WARPSIZE}-{(x+4)*WARPSIZE-1})"


@dataclass
class WarpAllocator:
  def __init__(self):
    self.warp_free = MAX_WARPS_PER_BLOCK_DIM * [True]

  def alloc_warp(self):
    i = self.warp_free.index(True)
    result = Warp(i)
    self.warp_free[i] = False
    return result

  def alloc_warpgroup(self):
    i = None
    for j in range(0, len(self.warp_free), 4):
      if all(self.warp_free[j:j+4]):
        i = j
        break
    assert i is not None
    result = Warpgroup(i)
    for j in range(4):
      self.warp_free[i+j] = False
    return result

  def block_size(self):
    return WARPSIZE * (len(self.warp_free) - self.warp_free[::-1].index(False))


@contextlib.contextmanager
def _only(w, warpid):
  index = ir.IndexType.get()
  if isinstance(w, Warp):
    is_warp_x = arith.cmpi(arith.CmpIPredicate.eq, warpid, c(w.x,index))
    with ir.InsertionPoint(scf.IfOp(is_warp_x).then_block):
      yield
      scf.yield_([])
  elif isinstance(w, Warpgroup):
    a = w.x
    b = a+4
    gea = arith.cmpi(arith.CmpIPredicate.uge, warpid, c(a,index))
    ltb = arith.cmpi(arith.CmpIPredicate.ult, warpid, c(b,index))
    predicate = arith.andi(gea, ltb)
    with ir.InsertionPoint(scf.IfOp(predicate).then_block):
      yield
      scf.yield_([])
  else:
    raise ValueError


@contextlib.contextmanager
def single_warp_thread():
  i1 = ir.IntegerType.get_signless(1)
  elected = nvvm.elect_sync(i1)
  with ir.InsertionPoint(scf.IfOp(elected).then_block):
    yield
    scf.yield_([])


def build_kernel(
    m, n, k,
    cta_count: int,
    expert_count: int,
    tile_m: int = 128,
    tile_n: int = 128,
    max_concurrent_steps: int = 2,
    in_dtype = jnp.float16,
    profiler_spec: profiler.ProfilerSpec | None = None,
):
  i1 = ir.IntegerType.get_signless(1)
  i32 = ir.IntegerType.get_signless(32)
  index = ir.IndexType.get()
  f32 = ir.F32Type.get()

  cx = lambda v: c(v, index)

  swizzle = 128
  swizzle_elems = tile_k = swizzle // 2
  tiling = (8, swizzle_elems)
  tmem_slot_count = 2

  k_loop_iter = k // tile_k
  max_concurrent_steps = min(max_concurrent_steps, k_loop_iter)

  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  if n % expert_count != 0:
    raise ValueError(f"{n=} must be divisible by {expert_count=}")

  expert_n = n // expert_count

  if expert_n % tile_n != 0:
    raise ValueError(f"{expert_n=} must be divisible by {tile_n=}")

  n_tile_count = expert_n // tile_n

  def ceildiv(x, y):
    return arith.divui(arith.subi(arith.addi(x, y),
                                  cx(1)),
                       y)

  allocator = WarpAllocator()
  warp = allocator.alloc_warp
  warpgroup = allocator.alloc_warpgroup

  tma_warp = warp()
  mma_warp = warp()


  def fdgg(ctx, a, w, group_sizes, d, smem):
    ((a_smem, b_smem), d_smem), barriers, mma_done_barrier, acc = smem
    (ab_full_barriers, ab_empty_barriers) = barriers

    warp_idx = arith.index_cast(index, mgpu.warp_idx(sync=True))

    @contextlib.contextmanager
    def only(w):
      with _only(w, warp_idx):
        yield

    worker_id = gpu.block_id(gpu.Dimension.x)
    worker_count = cx(cta_count)
    work_id_group_start = cx(0)
    initial_work_id = worker_id
    group_offset = cx(0)
    @mgpu.fori(cx(expert_count), [initial_work_id, work_id_group_start, group_offset])
    def group_for_body(expert_id, carrys):
      work_id, work_id_group_start, group_offset = carrys
      group_size = arith.index_cast(index,
                                    memref.load(group_sizes, [expert_id]))
      group_work_item_count = arith.muli(ceildiv(group_size, cx(tile_m)),
                                         cx(n_tile_count))
      work_id_group_end = arith.addi(work_id_group_start, group_work_item_count)

      @subgroup_for(work_id, work_id_group_end, worker_count)
      def subgroup_for_body(work_id):
        tile_n_start = arith.muli(arith.remui(work_id, cx(n_tile_count)),
                                  cx(tile_n))
        work_id_within_group = arith.subi(work_id, work_id_group_start)
        subgroup = arith.divui(work_id_within_group, cx(n_tile_count))
        # TODO(andportnoy) try to pick better names for all these variables
        #                  and streamline layout calculations in general
        tile_m_start_within_group = arith.muli(subgroup, cx(tile_m))
        tile_m_start = arith.addi(group_offset, tile_m_start_within_group)
        tile_m_end_within_group = arith.minui(arith.addi(tile_m_start_within_group, cx(tile_m)),
                                              group_size)
        tile_m_end = arith.addi(group_offset, tile_m_end_within_group)
        tile_m_count = arith.subi(tile_m_end, tile_m_start)

        # output n_start
        o_n_start = tile_n_start

        # weight n_start
        w_n_start = arith.addi(arith.muli(expert_id, cx(expert_n)),
                               tile_n_start)

        local_work_id = arith.divui(work_id, worker_count)
        tmem_slot = arith.remui(local_work_id, cx(tmem_slot_count))

        tmem_address = arith.addi(acc.address,
                                  arith.index_cast(i32,
                                                   arith.muli(tmem_slot, cx(tile_n))))
        acc_slot = tcgen05.TMEMRef(
            address=tmem_address,
            shape=(tile_m, tile_n),
            layout=acc.layout,
            dtype=acc.dtype,
        )
        get_persistent_ki = lambda ki: arith.addi(ki, arith.muli(local_work_id, cx(k_loop_iter)))
        with only(tma_warp), single_warp_thread():
          @mgpu.fori(cx(k_loop_iter), None)
          def _tma_body(ki, _):
            persistent_ki = get_persistent_ki(ki)
            slot = arith.remui(persistent_ki, c(max_concurrent_steps, index))
            with ctx.named_region("TMA wait for MMA"):
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

            a_smem_slot = mgpu.memref_slice(a_smem, slot)
            a_smem_slot_2d_shape, _ = mma_utils.tiled_memref_shape(a_smem_slot)
            a_smem_slot_2d = mgpu.memref_reshape(a_smem_slot, a_smem_slot_2d_shape)

            ctx.async_copy(
                src_ref=a,
                dst_ref=a_smem_slot_2d,
                # We load a fixed tile_m, even though we only need
                # group_chunk_m. With tensormap.replace we'll be able to
                # update the window dynamically.
                gmem_slice=(ds(tile_m_start, tile_m), ds(k_start, tile_k)),
                **common_args,
            )
            ctx.async_copy(
                src_ref=w,
                dst_ref=mgpu.memref_slice(b_smem, slot),
                gmem_slice=(ds(w_n_start, tile_n), ds(k_start, tile_k)),
                gmem_transform=mgpu.TileTransform(tiling),
                **common_args,
            )

        with only(mma_warp), single_warp_thread():
          @mgpu.fori(c(k_loop_iter, index), arith.constant(i1, 0))
          def _mma_body(ki, accumulate):
            persistent_ki = get_persistent_ki(ki)
            slot = arith.remui(persistent_ki, c(max_concurrent_steps, index))
            ab_full_barriers[slot].wait()
            tcgen05.mma(
                acc_slot,
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

        with ctx.named_region("GMEM store"):
          gpu.barrier()
          mma_done_barrier.wait(for_tensor_core=True)

          # TODO make optimized an autotuned variable
          acc_slot.load().astype(ir.F16Type.get()).store_untiled(d_smem, swizzle=128, optimized=False)
          mgpu.commit_shared()

          store_row = cx(0)
          for i in range(math.ceil(math.log2(tile_m)) + 1):
            num_rows = 1 << i
            do_store = arith.cmpi(
                arith.CmpIPredicate.ne, arith.andi(tile_m_count, cx(num_rows)), cx(0),
            )
            with mgpu.when(do_store):
              gmem_off = arith.addi(tile_m_start, store_row)
              out_smem_slice = mgpu.memref_slice(d_smem, ds(store_row, num_rows))
              out_smem_slice = mgpu.memref_reshape(
                  out_smem_slice,
                  (num_rows, tile_n // swizzle_elems, swizzle_elems)
              )
              ctx.async_copy(
                  src_ref=out_smem_slice,
                  dst_ref=d,
                  gmem_slice=(ds(gmem_off, num_rows), ds(o_n_start, tile_n)),
                  swizzle=swizzle,
                  gmem_transform=mgpu.TileTransform((swizzle_elems,)),
              )
            store_row = arith.select(
                do_store, arith.addi(store_row, cx(num_rows)), store_row
            )
          ctx.await_async_copy(0)
        # subgroup_for
      work_id = subgroup_for_body
      work_id_group_start = work_id_group_end
      group_offset = arith.addi(group_offset, group_size)
      carrys = [work_id, work_id_group_start, group_offset]
      return carrys
      # group_for

  compute_buffers = (
    jax.ShapeDtypeStruct(
        mgpu.tile_shape((max_concurrent_steps, tile_m, tile_k), tiling),
        jnp.float16),
    jax.ShapeDtypeStruct(
        mgpu.tile_shape((max_concurrent_steps, tile_n, tile_k), tiling),
        jnp.float16),
  )
  epilogue_buffer = jax.ShapeDtypeStruct(
      (tile_m, tile_n),
      jnp.float16)
  smem_buffers = mgpu.Union([compute_buffers, epilogue_buffer])
  tmem_cols = tmem_slot_count * tile_n
  assert tmem_cols.bit_count() == 1
  smem = (
      smem_buffers,
      [mgpu.Barrier(arrival_count=1, num_barriers=max_concurrent_steps)] * 2,
      mgpu.Barrier(arrival_count=1),
      mgpu.TMEM((128, tmem_cols), jnp.float32),
  )

  f = mgpu.as_gpu_kernel(
      fdgg,
      (cta_count, 1, 1),  # persistent kernel
      (128, 1, 1),
      (
          jax.ShapeDtypeStruct((m, k), jnp.float16),
          jax.ShapeDtypeStruct((n, k), jnp.float16),
          jax.ShapeDtypeStruct((expert_count,), jnp.int32),  # group_sizes
      ),
      jax.ShapeDtypeStruct((m, expert_n), jnp.float16),
      smem,
      prof_spec=profiler_spec,
  )

  return f


def generate_group_sizes(expert_count, token_count, key1, key2):
  v = jr.truncated_normal(key1, -2., 2., expert_count) + 2.
  expert_probs = v / jnp.sum(v)
  expert_assignment = jr.choice(key2, expert_count, (token_count,), p=expert_probs)
  group_sizes = jnp.bincount(expert_assignment)
  return group_sizes


def ref(activations, weights, group_sizes, expert_n):
  group_offsets = np.cumulative_sum(group_sizes, include_initial=True).tolist()
  results = []
  for i, (a, b) in enumerate(zip(group_offsets, group_offsets[1:])):
    group = activations[a:b, :]
    expert = weights[i*expert_n:(i+1)*expert_n, :]
    results.append(group @ expert.T)

  return jnp.concatenate(results)


def main(unused_argv):
  profile = False
  m = 4096  # seqlen
  k = 4096
  expert_count = 32
  expert_n = 4096
  n = expert_count * expert_n
  in_dtype = jnp.float16
  # TODO(andportnoy) this leads to low occupancy, so unless each CTA
  # is using tensor cores at all times, we are leaving perf on the
  # table. A solution would be to separate the epilogue into a
  # warpgroup of its own, and double buffer the accumulator, so that
  # the next work item can run while we are storing the previous one
  # out.
  cta_count = get_sm_count()
  ka, kb, ke1, ke2 = jr.split(jr.key(0), 4)
  activations = jr.normal(key=ka, shape=(m, k), dtype=in_dtype)
  weights     = jr.normal(key=kb, shape=(n, k), dtype=in_dtype)
  group_sizes = generate_group_sizes(expert_count, m, ke1, ke2)
  #group_sizes = jnp.ones(expert_count, dtype=jnp.int32) * (m // expert_count)
  print(f"{group_sizes=}")
  assert sum(group_sizes) == m
  # TODO(andportnoy) test different tile sizes
  tile_m = 128
  tile_n = 256  # 256, 512

  # TODO(andportnoy) test different stage counts
  max_concurrent_steps = 4  # 4, 5, 6
  profiler_spec = profiler.ProfilerSpec(4096) if profile else None
  with mlir.make_ir_context(), ir.Location.unknown():
    f = build_kernel(
      m, n, k,
      cta_count=cta_count,
      expert_count=expert_count,
      tile_m=tile_m,
      tile_n=tile_n,
      max_concurrent_steps=max_concurrent_steps,
      in_dtype=in_dtype,
      profiler_spec=profiler_spec,
    )

  if profile:
    d = f(activations, weights, group_sizes)
    exit()

  d, runtime = profiler.measure(f, mode="cupti")(activations, weights, group_sizes)
  print(f"{runtime=}")
  tflops = (2 * m * expert_n * k / 1e12) / (runtime / 1e3)
  print(f"{tflops=}")
  group_chunk_count = jnp.sum(jnp.ceil(group_sizes / tile_m))
  print(f"{group_chunk_count=}")
  physical_tflops = (2 * group_chunk_count * tile_m * expert_n * k / 1e12) / (runtime / 1e3)
  print(f"{physical_tflops=}")
  d_ref = ref(activations, weights, group_sizes, expert_n)
  np.testing.assert_allclose(d, d_ref, atol=1e-3, rtol=1e-3)
  # TODO(andportnoy) measure FLOPS


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
