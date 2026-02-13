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
import enum
import itertools
import math
import os

import jax
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
from jax.experimental import mesh_utils, shard_map
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import c, ds
from jax.experimental.mosaic.gpu import tcgen05
from jax.experimental.mosaic.gpu import mma_utils
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.mosaic.gpu import utils
import jax.numpy as jnp
import jax.random as jr
import numpy as np


WARPSIZE = 32
WARPGROUPSIZE = 4 * WARPSIZE
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
# Maximum x- or y-dimensionality of a block
MAX_WARPS_PER_BLOCK_DIM = 32


TMA_WARP = 1
MMA_WARP = 0

P = jax.sharding.PartitionSpec

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


def global_membar():
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [],
      "membar.gl;",
      "",
      has_side_effects=True,
  )


def signal_with_red(mc_ptr, is_relaxed=False):
  mode = "relaxed" if is_relaxed else "release"
  asm_instr = f"""
  {{
      multimem.red.{mode}.sys.global.add.u32 [$0], 1;
      fence.proxy.alias;
  }}
  """
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [mc_ptr],
      asm_instr,
      "l",
      has_side_effects=True,
      asm_dialect=0,
  )


def wait_loop(uc_ptr, num_gpus=8, is_relaxed=False):
  mode = "relaxed" if is_relaxed else "acquire"
  asm_instr = f"""
  {{
      .reg .u32   %tmp32_<1>;
      .reg .pred  %p<1>;
      wait_signal:
          atom.global.sys.{mode}.cas.b32 %tmp32_0, [$0], {num_gpus}, 0;
          setp.eq.u32 %p0, %tmp32_0, {num_gpus};
          @!%p0 bra wait_signal;
  }}"""
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [uc_ptr],
      asm_instr,
      "l",
      has_side_effects=True,
      asm_dialect=0,
  )


def get_sm_count():
  return jax.local_devices()[0].core_count


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


def warpgroup_stride_loop(upper_bound, items_per_iteration):
  def wrapper(f):
    index = ir.IndexType.get()
    thread_id = gpu.thread_id(gpu.Dimension.x)
    warpgroup_thread_id = arith.andi(thread_id,
                                     arith.constant(index, (1 << 7) - 1))
    lower_bound = arith.muli(warpgroup_thread_id,
                             arith.constant(index, items_per_iteration))
    step = arith.constant(index, items_per_iteration*WARPGROUPSIZE)
    for_op = scf.ForOp(lower_bound, upper_bound, step, [])
    with ir.InsertionPoint(for_op.body):
      i = for_op.induction_variable
      in_bounds = arith.cmpi(arith.CmpIPredicate.ult, i, upper_bound)
      with mgpu.when(in_bounds):
        f(i)
      scf.yield_([])
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


def multimem_ld_reduce_bf16(mc_ptr, n=8):
  i32 = ir.IntegerType.get_signless(32)

  assert n in (2, 4, 8)
  v = n // 2
  operands = "{" + ", ".join(f"${i}" for i in range(v)) + f"}}, [${v}]"
  constraints = ",".join("=r" for _ in range(v)) + ",l"

  return_type = ir.Type.parse("!llvm.struct<(" + ",".join(["i32"]*v) + ")>")
  insn = f"multimem.ld_reduce.relaxed.sys.global.add.v{v}.bf16x2"
  return_value = llvm.inline_asm(
      return_type,
      [mc_ptr],
      f"{insn} {operands};",
      constraints,
      has_side_effects=True,
      asm_dialect=0,
  )
  return [llvm.extractvalue(i32, return_value, [i]) for i in range(v)]


def multimem_st(mc_ptr, values):
  v = len(values)
  n = 2 * v
  assert n in (2, 4, 8)
  constraints = "l," + ",".join(["r"]*v)
  operands = "[$0], {" + ", ".join(f"${1+i}" for i in range(v)) + "}"
  insn = f"multimem.st.relaxed.sys.global.v{v}.f32"
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [mc_ptr] + values,

      f"{insn} {operands};",
      constraints,
      has_side_effects=True,
      asm_dialect=0,
  )


def local_st(ptr, values):
  v = len(values)
  n = 2 * v
  assert n in (2, 4, 8)
  constraints = "l," + ",".join(["r"]*v)
  operands = "[$0], {" + ", ".join(f"${1+i}" for i in range(v)) + "}"
  insn = f"st.global.v{v}.b32"
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [ptr] + values,
      f"{insn} {operands};",
      constraints,
      has_side_effects=True,
      asm_dialect=0,
  )


class Reduction(enum.Enum):
  AllReduce = enum.auto()
  ReduceScatter = enum.auto()


def build_kernel(
    m, n, k,
    cta_count: int,
    expert_count: int,
    in_dtype,
    acc_dtype,
    out_dtype,
    num_gpus: int,
    tile_m: int = 128,
    tile_n: int = 128,
    max_concurrent_steps: int = 2,
    profiler_spec: profiler.ProfilerSpec | None = None,
    reduction = None,
):
  assert (in_dtype == jnp.float16 and acc_dtype == jnp.float16 or
          in_dtype == jnp.float16 and acc_dtype == jnp.float32 or
          in_dtype == jnp.bfloat16 and acc_dtype == jnp.float32)

  i1 = ir.IntegerType.get_signless(1)
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  index = ir.IndexType.get()
  f16 = ir.F16Type.get()
  bf16 = ir.BF16Type.get()
  f32 = ir.F32Type.get()

  mlir_dtype = {
    jnp.float16: f16,
    jnp.bfloat16: bf16,
    jnp.float32: f32,
  }

  cx = lambda v: c(v, index)

  swizzle = 128
  out_swizzle_elems = swizzle // out_dtype.dtype.itemsize
  in_swizzle_elems = tile_k = swizzle // in_dtype.dtype.itemsize
  in_tiling = (8, in_swizzle_elems)
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

  if reduction == Reduction.ReduceScatter and ((expert_n // num_gpus) % tile_n) != 0:
    raise ValueError(f"{expert_n // num_gpus=} must be divisible by {tile_n=}")

  n_tile_count = expert_n // tile_n

  def ceildiv(x, y):
    return arith.divui(arith.subi(arith.addi(x, y),
                                  cx(1)),
                       y)

  allocator = WarpAllocator()
  warp = allocator.alloc_warp
  warpgroup = allocator.alloc_warpgroup

  producer_warpgroup = warpgroup()
  tma_warp = Warp(producer_warpgroup.x + 0)
  mma_warp = Warp(producer_warpgroup.x + 1)
  consumer_warpgroup = warpgroup()

  def fdgg(ctx, a, w, group_sizes, start_sem, done_sem, d, smem):
    ((a_smem, b_smem), d_smem), barriers, acc = smem
    (ab_full_barriers, ab_empty_barriers, acc_full_barrier, acc_empty_barrier) = barriers

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

    # In the outer loop, we iterate over experts.
    @mgpu.fori(cx(expert_count), [initial_work_id, work_id_group_start, group_offset])
    def group_for_body(expert_id, carrys):
      work_id, work_id_group_start, group_offset = carrys
      # group_size = how many tokens/rows are routed to the expert
      # identified by expert_id?
      group_size = arith.index_cast(index,
                                    memref.load(group_sizes, [expert_id]))
      # We tile each "group" along the m dimension (via tile_m) and
      # along the n dimension (via tile_n).
      # group_work_item_count is the number of output tiles we need to
      # compute for this group.
      group_work_item_count = arith.muli(ceildiv(group_size, cx(tile_m)),
                                         cx(n_tile_count))
      # What is the starting work/tile id for this group?
      work_id_group_end = arith.addi(work_id_group_start, group_work_item_count)

      # In the inner loop, we iterate over tiles/work items in the group.
      @subgroup_for(work_id, work_id_group_end, worker_count)
      def subgroup_for_body(work_id):
        # The index of the tile along the n dimension.
        tile_n_id = arith.remui(work_id, cx(n_tile_count))
        # The offset of the tile along the n dimension.
        tile_n_start = arith.muli(tile_n_id, cx(tile_n))
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
                predicate=None,
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
                gmem_slice=(expert_id, ds(k_start, tile_k), ds(tile_n_start, tile_n)),
                gmem_transform=mgpu.TileTransform(in_tiling),
                **common_args,
            )

        with only(mma_warp), single_warp_thread():
          # Make sure the accumulator is available (a consumer is not reading it).
          not_first_local_work_item = arith.cmpi(
            arith.CmpIPredicate.ne, local_work_id, cx(0),
          )
          with mgpu.when(not_first_local_work_item):
            acc_empty_barrier.wait()
          @mgpu.fori(c(k_loop_iter, index), arith.constant(i1, 0))
          def _mma_body(ki, accumulate):
            persistent_ki = get_persistent_ki(ki)
            slot = arith.remui(persistent_ki, c(max_concurrent_steps, index))
            ab_full_barriers[slot].wait()
            tcgen05.mma(
                acc,
                mgpu.memref_slice(a_smem, slot),
                mgpu.memref_slice(b_smem, slot),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                accumulate=accumulate,
            )
            accumulate = arith.constant(i1, 1)
            tcgen05.commit_arrive(ab_empty_barriers[slot].get_ptr(), ctx=ctx)
            return accumulate
          tcgen05.commit_arrive(acc_full_barrier.get_ptr(), ctx=ctx)

        with only(consumer_warpgroup):
          with ctx.named_region("GMEM store"):
            utils.warpgroup_barrier()
            acc_full_barrier.wait(orders_tensor_core=True)

            # TODO make optimized an autotuned variable
            acc_data = acc.load().astype(mlir_dtype[out_dtype]).store_untiled(d_smem, swizzle=swizzle, optimized=False)
            acc_empty_barrier.arrive()
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
                    (num_rows, tile_n // out_swizzle_elems, out_swizzle_elems)
                )
                ctx.async_copy(
                    src_ref=out_smem_slice,
                    dst_ref=d,
                    gmem_slice=(ds(gmem_off, num_rows), ds(o_n_start, tile_n)),
                    swizzle=swizzle,
                    gmem_transform=mgpu.TileTransform((out_swizzle_elems,)),
                )
              store_row = arith.select(
                  do_store, arith.addi(store_row, cx(num_rows)), store_row
              )
            ctx.await_async_copy(0)

          with ctx.named_region("Reduction"):
            if reduction is not None:
              with ctx.named_region("Global membar + start semaphore"):
                global_membar()

                # We use unicast references to semaphores to read/wait on them.
                # Start semaphores are double buffered. Need to clarify why
                # this is needed, but this prevents deadlocks in some cases.
                # I think roughly this helps in cases when the corresponding
                # SM on a peer GPU runs ahead of the SM on this GPU.
                start_semaphore_index = arith.addi(worker_id,
                                                   arith.muli(worker_count,
                                                              arith.remui(local_work_id, cx(2))))
                start_semaphore_memref = utils.memref_slice(start_sem, start_semaphore_index)

                start_semaphore_unicast_ptr = utils.memref_ptr(start_semaphore_memref)

                start_semaphore_multicast_ptr = utils.memref_ptr(
                  ctx.to_remote_multicast(start_semaphore_memref).ref)

                # Whichever GPU is going to reduce this tile, we need to
                # synchronize to allow the reduction to happen.
                with mgpu.single_thread(scope=mgpu.ThreadSubset.BLOCK):
                  signal_with_red(start_semaphore_multicast_ptr, is_relaxed=True)
                  wait_loop(start_semaphore_unicast_ptr, num_gpus, is_relaxed=True)

            if reduction == Reduction.AllReduce:
              utils.warpgroup_barrier()
              device_id = arith.index_cast(index, ctx.device_id())

              # multimem.ld_reduce does 128 bits at a time
              multimem_bitwidth = 128
              multimem_bytes = multimem_bitwidth // 8
              num_red_elements = multimem_bytes//out_dtype.dtype.itemsize
              world_size = cx(num_gpus)
              num_rows_per_gpu = arith.ceildivui(tile_m_count, world_size)
              num_threads_per_gpu = arith.divui(cx(tile_n), cx(num_red_elements))
              thread_idx = arith.index_cast(index, utils.thread_idx())
              in_bound_for_n = arith.cmpi(arith.CmpIPredicate.ult, thread_idx, num_threads_per_gpu)
              with mgpu.when(in_bound_for_n):
                @mgpu.fori(num_rows_per_gpu, None)
                def _(i, _):
                  m_offset = arith.addi(arith.muli(i, world_size), device_id)
                  in_bound_for_m = arith.cmpi(arith.CmpIPredicate.ult, m_offset, tile_m_count)
                  with mgpu.when(in_bound_for_m):
                    n_offset = arith.muli(thread_idx, cx(num_red_elements))
                    m_idx = arith.addi(tile_m_start, m_offset)
                    n_idx = arith.addi(o_n_start, n_offset)
                    unicast_memref = utils.memref_slice(d, (ds(m_idx, 1), ds(n_idx, 1)))
                    multicast_ptr = utils.memref_ptr(ctx.to_remote_multicast(unicast_memref).ref)
                    values = multimem_ld_reduce_bf16(multicast_ptr, n=num_red_elements)
                    multimem_st(multicast_ptr, values)

            elif reduction == Reduction.ReduceScatter:
              # decide whether the tile needs to be reduced (is it my tile?)
              device_id = arith.index_cast(index, ctx.device_id())
              thread_id = arith.index_cast(index, utils.thread_idx())
              n_tiles = arith.divui(cx(expert_n), cx(tile_n))
              n_tiles_per_gpu = arith.divui(n_tiles, cx(num_gpus))
              my_tiles_n_id_start = arith.muli(device_id, n_tiles_per_gpu)
              my_tiles_n_id_end = arith.addi(my_tiles_n_id_start, n_tiles_per_gpu)
              should_reduce_tile = arith.andi(
                arith.cmpi(arith.CmpIPredicate.uge, tile_n_id, my_tiles_n_id_start),
                arith.cmpi(arith.CmpIPredicate.ult, tile_n_id, my_tiles_n_id_end))
              with mgpu.when(should_reduce_tile):
                # bits processed per multimem instruction
                multimem_bitwidth = 128
                # bytes processed per multimem instruction
                multimem_bytes = multimem_bitwidth // 8
                # bytes per element
                element_bytesize = out_dtype.dtype.itemsize
                # elements processed per multimem instruction
                multimem_elements = multimem_bytes // element_bytesize
                # give each thread a row of the output tile
                thread_id = gpu.thread_id(gpu.Dimension.x)
                warpgroup_thread_id = arith.andi(thread_id,
                                                 arith.constant(index, (1 << 7) - 1))
                # Note this potentially extends out of bounds of the
                # actual tile of interest, our true tile row count size is
                # tile_m_count, not tile_m.
                tile_memref = utils.memref_slice(d,
                    (
                      ds(tile_m_start, tile_m),
                      ds(o_n_start, tile_n),
                    ))
                with ctx.named_region("ReduceScatter body"):
                  # Each warp will process a row of the output tile.
                  # TODO(andportnoy) Generalize over tile_n so later we
                  #                  could do more flexible autotuning
                  #                  sweeps.
                  assert WARPSIZE * multimem_elements == tile_n
                  # Here we are building a basic warpgroup stride loop.

                  linear_element_index = arith.muli(warpgroup_thread_id, cx(multimem_elements))
                  row_id = arith.divui(linear_element_index, cx(tile_n))
                  col_id = arith.remui(linear_element_index, cx(tile_n))
                  start_element = utils.memref_slice(tile_memref, (row_id, col_id))
                  start_local_ptr = utils.memref_ptr(start_element)
                  start_multicast_ptr = utils.memref_ptr(ctx.to_remote_multicast(start_element).ref)

                  # what is the stride?
                  tile_strides, _ = tile_memref.type.get_strides_and_offset()
                  rows_per_unrolled_iter, rem = divmod(WARPGROUPSIZE * multimem_elements, tile_n)
                  assert rem == 0
                  # A warpgroup will process 4 rows at a time.
                  bytestride = element_bytesize * rows_per_unrolled_iter * tile_strides[0]
                  tile_extent_ptr = utils.memref_ptr(
                    utils.memref_slice(tile_memref, arith.addi(tile_m_count, cx(1))))

                  def idx2ptr(idx):
                    return llvm.inttoptr(ir.Type.parse("!llvm.ptr"), arith.index_cast(i64, idx))

                  def ptr2idx(ptr):
                    return arith.index_cast(index, llvm.ptrtoint(i64, ptr))

                  unroll_factor = 8
                  rows_per_outer_iter = unroll_factor * rows_per_unrolled_iter
                  unrolled_outer_iterations = arith.divui(tile_m_count, cx(rows_per_outer_iter))
                  unrolled_rows = arith.muli(unrolled_outer_iterations, cx(rows_per_outer_iter))

                  lower_bound = row_id
                  upper_bound = unrolled_rows
                  step = cx(rows_per_outer_iter)
                  def ptr_increment(ptr, increment):
                    return idx2ptr(arith.addi(ptr2idx(ptr), increment))

                  utils.warpgroup_barrier()
                  with ctx.named_region("Unrolled ReduceScatter"):
                    for_op = scf.ForOp(lower_bound, upper_bound, step, [start_local_ptr, start_multicast_ptr])
                    with ir.InsertionPoint(for_op.body):
                      local_ptr, multicast_ptr = for_op.inner_iter_args
                      vals = [
                        multimem_ld_reduce_bf16(
                          ptr_increment(multicast_ptr, cx(u*bytestride)),
                          n=multimem_elements,
                        )
                        for u in range(unroll_factor)
                      ]
                      for u, values in enumerate(vals):
                        local_st(ptr_increment(local_ptr, cx(u*bytestride)), values)

                      new_local_ptr = ptr_increment(local_ptr, cx(unroll_factor*bytestride))
                      new_multicast_ptr = ptr_increment(multicast_ptr, cx(unroll_factor*bytestride))
                      scf.yield_([new_local_ptr, new_multicast_ptr])

                  with ctx.named_region("Tail of ReduceScatter"):
                    # process remaining rows
                    start_local_ptr, start_multicast_ptr = for_op.results
                    lower_bound = arith.addi(row_id, unrolled_rows)
                    upper_bound = tile_m_count
                    step = cx(rows_per_unrolled_iter)
                    for_op = scf.ForOp(lower_bound, upper_bound, step, [start_local_ptr, start_multicast_ptr])
                    with ir.InsertionPoint(for_op.body):
                      local_ptr, multicast_ptr = for_op.inner_iter_args
                      values = multimem_ld_reduce_bf16(multicast_ptr, n=multimem_elements)
                      local_st(local_ptr, values)
                      new_local_ptr = idx2ptr(arith.addi(ptr2idx(local_ptr), cx(bytestride)))
                      new_multicast_ptr = idx2ptr(arith.addi(ptr2idx(multicast_ptr), cx(bytestride)))
                      scf.yield_([new_local_ptr, new_multicast_ptr])

      work_id = subgroup_for_body
      work_id_group_start = work_id_group_end
      group_offset = arith.addi(group_offset, group_size)
      carrys = [work_id, work_id_group_start, group_offset]
      return carrys
      # group_for

    with only(consumer_warpgroup):
      if reduction is not None:
        # sync to end the kernel
        done_semaphore_memref = utils.memref_slice(done_sem, worker_id)
        done_semaphore_unicast_ptr = utils.memref_ptr(done_semaphore_memref)
        done_semaphore_multicast_ptr = utils.memref_ptr(ctx.to_remote_multicast(
          done_semaphore_memref).ref)

        utils.warpgroup_barrier()
        with mgpu.single_thread(scope=mgpu.ThreadSubset.BLOCK):
          signal_with_red(done_semaphore_multicast_ptr)
          wait_loop(done_semaphore_unicast_ptr, num_gpus)

  compute_buffers = (
    jax.ShapeDtypeStruct(
        mgpu.tile_shape((max_concurrent_steps, tile_m, tile_k), in_tiling),
        in_dtype),
    jax.ShapeDtypeStruct(
        mgpu.tile_shape((max_concurrent_steps, tile_k, tile_n), in_tiling),
        in_dtype),
  )
  epilogue_buffer = jax.ShapeDtypeStruct(
      (tile_m, tile_n),
      out_dtype)
  smem_buffers = [compute_buffers, epilogue_buffer]
  tmem_cols = tile_n
  assert tmem_cols.bit_count() == 1
  barriers = [
    mgpu.Barrier(arrival_count=1, num_barriers=max_concurrent_steps),
    mgpu.Barrier(arrival_count=1, num_barriers=max_concurrent_steps),
    mgpu.Barrier(arrival_count=1),
    mgpu.Barrier(arrival_count=128),
  ]
  accumulator = mgpu.TMEM((128, tmem_cols), acc_dtype)
  smem = (
      smem_buffers,
      barriers,
      accumulator,
  )

  start_sem_dtype_shape = jax.ShapeDtypeStruct((2*cta_count,), jnp.int32)
  done_sem_dtype_shape = jax.ShapeDtypeStruct((cta_count,), jnp.int32)

  f = mgpu.as_gpu_kernel(
      fdgg,
      (cta_count, 1, 1),  # persistent kernel
      (allocator.block_size(), 1, 1),
      (
          jax.ShapeDtypeStruct((m, k), in_dtype),
          jax.ShapeDtypeStruct((expert_count, k, expert_n), in_dtype),
          jax.ShapeDtypeStruct((expert_count,), jnp.int32),  # group_sizes
      ),
      (
          jax.ShapeDtypeStruct((m, expert_n), out_dtype),
      ),
      smem,
      prof_spec=profiler_spec,
      inout_shape=(start_sem_dtype_shape, done_sem_dtype_shape),
      kernel_name="fdgg",
  )

  return f


def generate_group_sizes(expert_count, token_count, key1, key2):
  v = jr.truncated_normal(key1, -2., 2., expert_count) + 2.
  expert_probs = v / jnp.sum(v)
  expert_assignment = jr.choice(key2, expert_count, (token_count,), p=expert_probs)
  group_sizes = jnp.bincount(expert_assignment)
  return group_sizes


def make_ref(reduction, axis_name):
  def ref(activations, weights, group_sizes):
    output = jax.lax.ragged_dot(
      lhs=activations,
      rhs=weights,
      group_sizes=group_sizes,
    )

    match reduction:
      case Reduction.AllReduce:
        output = jax.lax.psum(output, axis_name=axis_name)
      case Reduction.ReduceScatter:
        output = jax.lax.psum_scatter(output, axis_name=axis_name, scatter_dimension=1, tiled=True)

    return output
  return ref


def make_kernel(m, n, k, num_gpus, expert_count, in_dtype, acc_dtype, out_dtype, tile_m, tile_n, max_concurrent_steps, profiler_spec, reduction, cta_count, tensor_parallelism_axis_name):
  with mlir.make_ir_context(), ir.Location.unknown():
    f = build_kernel(
      m, n, k,
      cta_count=cta_count,
      expert_count=expert_count,
      in_dtype=in_dtype,
      acc_dtype=acc_dtype,
      out_dtype=out_dtype,
      num_gpus=num_gpus,
      tile_m=tile_m,
      tile_n=tile_n,
      max_concurrent_steps=max_concurrent_steps,
      profiler_spec=profiler_spec,
      reduction=reduction,
    )

  def distributed_grouped_gemm(activations, weights, group_sizes):
    start_sem = jnp.zeros(2*cta_count,dtype=jnp.int32).reshape(-1)
    done_sem  = jnp.zeros(cta_count,dtype=jnp.int32).reshape(-1)
    output, _, _ = f(activations, weights, group_sizes, start_sem, done_sem)
    if reduction == Reduction.ReduceScatter:
      axis_size = jax.lax.axis_size(tensor_parallelism_axis_name)
      axis_idx = jax.lax.axis_index(tensor_parallelism_axis_name)
      slice_size, rem = divmod(output.shape[1], axis_size)
      assert rem == 0
      slice_start = slice_size * axis_idx
      output = jax.lax.dynamic_slice_in_dim(
        output, slice_start, slice_size, axis=1)
    return output

  return distributed_grouped_gemm


def main(unused_argv):
  jax.distributed.initialize()
  num_gpus = jax.device_count()
  print(f"{num_gpus=}")

  profile = False
  m = 8192  # seqlen
  # Mixtral 8x22b shapes
  k = 16 * 1024
  expert_count = 8
  expert_n = 6144
  n = expert_count * expert_n
  in_dtype = jnp.bfloat16
  acc_dtype = jnp.float32
  out_dtype = jnp.bfloat16
  # TODO(andportnoy) this leads to low occupancy, so unless each CTA
  # is using tensor cores at all times, we are leaving perf on the
  # table. A solution would be to separate the epilogue into a
  # warpgroup of its own, and double buffer the accumulator, so that
  # the next work item can run while we are storing the previous one
  # out.
  cta_count = get_sm_count()
  # TODO(andportnoy) test different tile sizes
  tile_m = 128
  tile_n = 256  # 256, 512

  # TODO(andportnoy) test different stage counts
  max_concurrent_steps = 3  # 4, 5, 6
  profiler_spec = None
  if profile:
    profiler_spec = profiler.ProfilerSpec(4096)
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = "."

  assert k % num_gpus == 0

  devices = mesh_utils.create_device_mesh((num_gpus,))
  mesh = jax.sharding.Mesh(devices, ("x",))
  activation_sharding = jax.sharding.NamedSharding(mesh, P(None, "x"))
  weight_sharding = jax.sharding.NamedSharding(mesh, P(None, "x", None))

  reduction = None
  if num_gpus > 1:
    reduction = Reduction.ReduceScatter

  f = make_kernel(
    m=m,
    n=n,
    k=k//num_gpus,
    num_gpus=num_gpus,
    expert_count=expert_count,
    in_dtype=in_dtype,
    acc_dtype=acc_dtype,
    out_dtype=out_dtype,
    tile_m=tile_m,
    tile_n=tile_n,
    max_concurrent_steps=max_concurrent_steps,
    profiler_spec=profiler_spec,
    reduction=reduction,
    cta_count=cta_count,
  )

  shard_map_common_args = dict(
    mesh=mesh,
    in_specs=(
      P(None, "x"),  # activations
      P(None, "x", None),  # weights
      P(None,),  # group_sizes
    ),
    out_specs=P(None, None),
    check_rep=False,
  )

  sharded_f = shard_map.shard_map(f, **shard_map_common_args)

  ka, kb, ke1, ke2 = jr.split(jr.key(0), 4)
  activations = jr.uniform(key=ka, shape=(m, k), dtype=in_dtype)
  activations = jax.device_put(activations, activation_sharding)
  weights     = jr.uniform(key=kb, shape=(expert_count, k, expert_n), dtype=in_dtype)
  weights     = jax.device_put(weights, weight_sharding)
  group_sizes = generate_group_sizes(expert_count, m, ke1, ke2)
  #group_sizes = jnp.ones(expert_count, dtype=jnp.int32) * (m // expert_count)
  print(f"{group_sizes=}")
  assert sum(group_sizes) == m

  profiler_aggregate = False

  for _ in range(5):
    d, runtime = profiler.measure(sharded_f, aggregate=profiler_aggregate)(activations, weights, group_sizes)
    print("Runtime:")
    if profiler_aggregate:
      print(f"\t{runtime:10.3f} ms")
    else:
      for measurement in runtime:
        kernel, t = measurement
        print(f"\t{t:10.3f} ms: {kernel}")
      assert 1 == sum(1 for x in runtime if x[0] == "fdgg_mosaic_gpu_kernel")
      runtime = dict(runtime)["fdgg_mosaic_gpu_kernel"]
  print(f"{d.shape=}")
  tflops = (2 * m * expert_n * k / 1e12) / (runtime / 1e3) / num_gpus
  print(f"{tflops=}")
  physical_tflops_correction = tile_m * np.sum(np.ceil(group_sizes / tile_m)).item() / m
  physical_tflops = tflops * physical_tflops_correction
  print(f"{physical_tflops=}")
  ref = make_ref(reduction, axis_name="x")
  sharded_ref = shard_map.shard_map(ref, **shard_map_common_args)

  d_ref, ref_runtime = profiler.measure(sharded_ref, aggregate=profiler_aggregate)(activations.astype(jnp.float32), weights.astype(jnp.float32), group_sizes)
  print(f"{d_ref.shape=}")
  print("Reference runtime:")
  if profiler_aggregate:
    print(f"\t{ref_runtime:10.3f} ms")
  else:
    for measurement in ref_runtime:
      kernel, t = measurement
      print(f"\t{t:10.3f} ms: {kernel}")
  np.testing.assert_allclose(d, d_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
