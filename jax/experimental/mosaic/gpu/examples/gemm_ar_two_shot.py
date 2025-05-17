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
"""simple two-shot distributed persistent GEMM overlapped with all reduce for H100."""

import dataclasses
import itertools
from typing import Any
from functools import partial

import jax
from jax import random
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax._src.interpreters import mlir
import jax._src.lib.mosaic_gpu  as mosaic_gpu_lib
from jax.experimental import mesh_utils, shard_map
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.mosaic.gpu import *  # noqa: F403
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.mosaic.gpu import core as mgpu_core
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
import numpy as np

# mypy: ignore-errors
# ruff: noqa: F405
# pylint: disable=line-too-long, wildcard-import, missing-function-docstring, bad-continuation, g-bad-todo, protected-access

SmemRef = ir.Value

P = jax.sharding.PartitionSpec


@dataclasses.dataclass(frozen=True)
class Tiling:
  m: int
  n: int
  k: int

  # Allow access by .mk, .kn, .mn, etc.
  def __getattr__(self, name):
    if len(name) == 1:
      return super().__getattribute__(name)
    return tuple(getattr(self, d) for d in name)


class WGMMADefaultImpl:
  """Default WGMMA implementation.

  The kernel can accept any class that satisfies the same interface as this
  class.
  """

  @staticmethod
  def zero_accs(tile_m: int, tile_n: int) -> WGMMAAccumulator:
    return WGMMAAccumulator.zero(tile_m, tile_n)

  @staticmethod
  def smem_shape_extra(
      block_tiling: Tiling,
      tma_tiling: Tiling,
      lhs_dtype: jnp.dtype, rhs_dtype: jnp.dtype,
      rhs_transpose: bool,
  ) -> dict[str, jax.ShapeDtypeStruct]:
    del block_tiling, tma_tiling, lhs_dtype, rhs_dtype, rhs_transpose  # Unused.
    return ()

  @staticmethod
  def get_result(acc: WGMMAAccumulator) -> FragmentedArray:
    return acc.value

  @staticmethod
  def wgmma(
      smem_scratch: Any,  # pylint: disable=unused-argument
      acc: WGMMAAccumulator,
      a_slice: SmemRef,
      b_slice: SmemRef,
      swizzle: int,
  ) -> dict[str, WGMMAAccumulator]:
    """Perform a matrix multiplication.

    This function must guarantee that all WGMMA operations queued before it was
    called have completed before returning.
    """
    acc = wgmma(acc, a_slice, b_slice, swizzle=swizzle)
    nvvm.wgmma_commit_group_sync_aligned()
    nvvm.wgmma_wait_group_sync_aligned(1)
    return acc


def mlir_context(f):
  def wrap(*args, **kw):
    with mlir.make_ir_context(), ir.Location.unknown():
      return f(*args, **kw)

  return wrap


def worker_for(worker_id, worker_count, work_count, init_state):
  def wrapper(f):
    for_op = scf.ForOp(worker_id, work_count, worker_count, [init_state])
    with ir.InsertionPoint(for_op.body):
      new_state = f(for_op.induction_variable, for_op.inner_iter_args[0])
      scf.yield_([new_state])
  return wrapper


@mlir_context
def build_kernel(
    m, n, k,
    lhs_dtype, rhs_dtype, out_dtype,
    cta_count: int = 132, # 132 SMs for Hopper
    num_gpus: int = 8,
    stages: int = 2,
    tile_m: int = 128,
    tile_n: int = 128,
    swizzle: int = 128,
    rhs_transpose: bool = False,
    wgmma_impl=WGMMADefaultImpl,
    profiler_spec: profiler.ProfilerSpec | None = None,
):
  if tile_m % 64 != 0:
    raise ValueError(f"{tile_m=} must be divisible by 64")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if stages < 2:
    raise ValueError(f"Need at least 2 stages, but got {stages=}")
  if not rhs_transpose and jnp.dtype(rhs_dtype).itemsize != 2:
    raise ValueError(f"Transpose only supported for 16bit types (got: {rhs_transpose=}, {rhs_dtype=})")
  if swizzle not in {32, 64, 128}:
    raise ValueError(f"swizzle must be 32, 64, or 128, but got {swizzle=}")

  out_mlir_dtype = mlir.dtype_to_ir_type(out_dtype)
  out_swizzle = swizzle
  if bytewidth(out_mlir_dtype) == 4:
    if tile_n % 32 == 0:
      out_swizzle = 128
    elif tile_n % 16 == 0:
      out_swizzle = 64
    else:
      raise NotImplementedError(
          f"{tile_n=} must by divisible by 16 for 32-bit output"
      )
  out_swizzle_elems = out_swizzle // bytewidth(out_mlir_dtype)
  out_tiling = (64, out_swizzle_elems)
  out_tile = jax.ShapeDtypeStruct(tile_shape((tile_m, tile_n), out_tiling), out_dtype)

  lhs_elem_bytes = bytewidth(mlir.dtype_to_ir_type(lhs_dtype))
  rhs_elem_bytes = bytewidth(mlir.dtype_to_ir_type(rhs_dtype))
  lhs_swizzle_elems = swizzle // lhs_elem_bytes
  rhs_swizzle_elems = swizzle // rhs_elem_bytes
  tile_k = max(lhs_swizzle_elems, rhs_swizzle_elems)

  if tile_n % rhs_swizzle_elems != 0:
    raise ValueError(
        f"{tile_n=} must be divisible by {swizzle} bytes ="
        f" {((lhs_swizzle_elems, lhs_dtype), (rhs_swizzle_elems, rhs_dtype))}"
    )

  if k % tile_k != 0:
    raise ValueError(f"k must be divisible by {tile_k=}, but got {k=}")

  block_tiling = Tiling(m=tile_m, n=tile_n, k=tile_k)
  tma_tiling = Tiling(m=64, n=rhs_swizzle_elems, k=lhs_swizzle_elems)
  k_steps = k // block_tiling.k
  stages = min(stages, k_steps)

  def safe_div(x, y):
    assert x % y == 0, (x, y)
    return x // y

  def partial_tile_div(x, y):
    if x % y == 0:
      return x // y
    else:
      return (x // y) + 1

  m_tile_count = partial_tile_div(m, block_tiling.m)
  n_tile_count = n // tile_n
  tile_count = m_tile_count * n_tile_count

  block = (128, 1, 1)

  c = arith.ConstantOp.create_index

  compute_scratch_shape = (
      jax.ShapeDtypeStruct((stages, *tile_shape(block_tiling.mk, tma_tiling.mk)), lhs_dtype),
      jax.ShapeDtypeStruct((stages, *tile_shape(block_tiling.kn, tma_tiling.kn)), rhs_dtype),
      wgmma_impl.smem_shape_extra(block_tiling, tma_tiling, lhs_dtype, rhs_dtype, rhs_transpose),
  )
  epilogue_scratch_shape = jax.ShapeDtypeStruct(out_tile.shape, out_tile.dtype)
  smem_shape = Union([compute_scratch_shape, epilogue_scratch_shape])

  def _main(ctx, a_device, b_device, c_device, start_sem, done_sem, c_dev_alias, start_sem_alias, done_sem_alias, smem):
    ((lhs_smem, rhs_smem, impl_smem), epilogue_smem), tma_barriers = smem

    i64_ty = ir.IntegerType.get_signless(64)
    i32_ty = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()

    cta = gpu.block_id(gpu.Dimension.x)
    cta_idx = arith.index_cast(i64_ty,gpu.block_id(gpu.Dimension.x))
    sync_flip_flop = mgpu_core.c(0, ir.IndexType.get())

    @worker_for(worker_id=cta, worker_count=mgpu_core.c(cta_count, index), work_count=mgpu_core.c(tile_count, index), init_state=sync_flip_flop)
    def body(work_id, sync_flip_flop):
      m_idx = arith.divui(work_id, mgpu_core.c(n_tile_count, index))
      n_idx = arith.remui(work_id, mgpu_core.c(n_tile_count, index))
      m_start = arith.muli(m_idx, mgpu_core.c(tile_m, index))
      n_start = arith.muli(n_idx, mgpu_core.c(tile_n, index))

      def fetch(slot, ki):
        barrier = tma_barriers[slot]
        k_start = arith.muli(c(block_tiling.k), ki)
        lhs_tma_tile_bytes = int(np.prod(block_tiling.mk) * lhs_elem_bytes)
        rhs_tma_tile_bytes = int(np.prod(block_tiling.kn) * rhs_elem_bytes)
        txcount = lhs_tma_tile_bytes + rhs_tma_tile_bytes
        common_copy_args = dict(
            swizzle=swizzle, barrier=barrier, arrive=False, uniform=False,
        )
        with single_thread():
          barrier.arrive_expect_tx(txcount)
          ctx.async_copy(
              src_ref=a_device,
              dst_ref=memref_slice(lhs_smem, slot),
              gmem_slice=(ds(m_start, block_tiling.m), ds(k_start, block_tiling.k)),
              gmem_transform=TileTransform(tma_tiling.mk),
              collective=(gpu.Dimension.x, gpu.Dimension.z),
              **common_copy_args,
          )
          rhs_slice = (ds(k_start, block_tiling.k), ds(n_start, block_tiling.n))
          rhs_transform = (TileTransform(tma_tiling.kn),)
          if rhs_transpose:
            rhs_slice = rhs_slice[::-1]
            rhs_transform += (TransposeTransform((1, 0, 2, 3)),)
            assert tma_tiling.n == tma_tiling.k, block_tiling  # No need to flip the tiling.
          ctx.async_copy(
              src_ref=b_device,
              dst_ref=memref_slice(rhs_smem, slot),
              gmem_slice=rhs_slice,
              gmem_transform=rhs_transform,
              collective=gpu.Dimension.y,
              **common_copy_args,
          )

      accs = wgmma_impl.zero_accs(block_tiling.m, block_tiling.n)

      with ctx.named_region("TMA warmup"):
        for i in range(stages):
          fetch(c(i), c(i))

      @fori(c(k_steps), accs)
      def stage_loop_body(ki, accs):
        si = arith.remui(ki, c(stages))

        with ctx.named_region("TMA wait"):
          tma_barriers[si].wait()

        with ctx.named_region("WGMMA"):
          a_slice = memref_slice(lhs_smem, si)
          b_slice = memref_slice(rhs_smem, si)
          if rhs_transpose:
            b_slice = memref_transpose(b_slice, (0, 1, 3, 2))
          accs = wgmma_impl.wgmma(
              impl_smem, accs, a_slice, b_slice, swizzle=swizzle
          )

        with ctx.named_region("TMA start"):
          tma_ki = arith.addi(ki, c(stages - 1))
          tma_si = arith.remui(tma_ki, c(stages))
          not_first_step = arith.cmpi(arith.CmpIPredicate.ne, ki, c(0))
          tma_ki_in_bounds = arith.cmpi(
              arith.CmpIPredicate.slt, tma_ki, c(k_steps)
          )
          do_tma = arith.andi(not_first_step, tma_ki_in_bounds)
          with ir.InsertionPoint(scf.IfOp(do_tma).then_block):
            fetch(tma_si, tma_ki)
            scf.yield_([])

        return accs

      # Wait until WGMMA is complete and we can safely read the accumulator.
      with ctx.named_region("WGMMA drain"):
        nvvm.wgmma_wait_group_sync_aligned(0)

      with ctx.named_region("SMEM store"):
        acc_val = wgmma_impl.get_result(stage_loop_body.result)
        acc_val.astype(out_mlir_dtype).store_tiled(epilogue_smem, swizzle=out_swizzle)
        commit_shared()  # Make sure the stores are visible to TMA.

      with ctx.named_region("GMEM store"):
        ctx.async_copy(
            src_ref=epilogue_smem,
            dst_ref=c_device,
            gmem_slice=(ds(m_start, tile_m), ds(n_start, tile_n)),
            gmem_transform=TileTransform(out_tiling),
            swizzle=out_swizzle,
        )
      # ensure all write are done
      ctx.await_async_copy(0)
      utils.global_membar()

      # sync all blocks
      bsem_uc_memref = memref_slice(
        start_sem, arith.addi(
          arith.index_cast(ir.IndexType.get(), cta_idx),
          arith.muli(sync_flip_flop, mgpu_core.c(cta_count, ir.IndexType.get()))
        )
      )
      bsem_mc_ptr = ctx.to_remote_mc_ptr(
        bsem_uc_memref,
        team=mgpu.utils.c(0, i32_ty),
      )
      bsem_uc_ptr = mgpu.utils.memref_ptr(
        memref_slice(start_sem, arith.index_cast(ir.IndexType.get(), cta_idx))
      )
      with ctx.named_region("sync all blocks before reduction"):
        mgpu_utils.warpgroup_barrier()
        with single_thread(scope=ThreadSubset.BLOCK):
          mgpu_utils.signal_with_red(bsem_mc_ptr, is_relaxed=True)
          mgpu_utils.wait_loop(bsem_uc_ptr, num_gpus, is_relaxed=True)
      sync_flip_flop = arith.xori(sync_flip_flop, mgpu_core.c(1, ir.IndexType.get()))
      # multimem load reduce and multimem store
      with ctx.named_region("ld red and st"):
        mgpu_utils.warpgroup_barrier()
        device_idx = arith.index_cast(index, ctx.device_id())
        num_red_elements = mgpu.utils.c(8, index)
        world_size = mgpu.utils.c(8, index)
        num_m = arith.minui(mgpu.utils.c(m, index), mgpu.utils.c(tile_m, index))
        num_rows_per_gpu = arith.ceildivui(num_m, world_size)
        num_threads_per_gpu = arith.divui(mgpu_core.c(tile_n, index), num_red_elements)
        thread_idx = arith.index_cast(index, utils.thread_idx())
        if_in_bound = scf.IfOp(arith.cmpi(arith.CmpIPredicate.ult, thread_idx, num_threads_per_gpu),hasElse=False)
        with ir.InsertionPoint(if_in_bound.then_block):
          for_op = scf.ForOp(mgpu.utils.c(0, index),num_rows_per_gpu,mgpu.utils.c(1, index))
          with ir.InsertionPoint(for_op.body):
            m_offset = arith.addi(arith.muli(for_op.induction_variable,world_size), device_idx)
            if_in_bound_m = scf.IfOp(arith.cmpi(arith.CmpIPredicate.ult, m_offset, num_m),hasElse=False)
            with ir.InsertionPoint(if_in_bound_m.then_block):
              n_offset = arith.muli(thread_idx, num_red_elements)
              m_idx = arith.addi(m_start, m_offset)
              n_idx = arith.addi(n_start, n_offset)
              uc_memref = memref_slice(c_device, (ds(m_idx, 1), ds(n_idx, 1)))
              mc_ptr = ctx.to_remote_mc_ptr(uc_memref,team=mgpu.utils.c(0, i32_ty))
              x, y, z, w = utils.multimem_ld_reduce_128(mc_ptr)
              utils.multimem_st_128(mc_ptr,x, y, z, w)
              scf.yield_([])
            scf.yield_([])
          scf.yield_([])
      return sync_flip_flop

    # sync to end the kernel
    sem_uc_memref = memref_slice(
      done_sem, arith.index_cast(ir.IndexType.get(), cta_idx)
    )
    sem_mc_ptr = ctx.to_remote_mc_ptr(
      sem_uc_memref,
      team=mgpu.utils.c(0, i32_ty),
    )
    sem_uc_ptr = mgpu.utils.memref_ptr(
      memref_slice(done_sem, arith.index_cast(ir.IndexType.get(), cta_idx))
    )

    with ctx.named_region("sync to end kernel"):
      mgpu_utils.warpgroup_barrier()
      with single_thread(scope=ThreadSubset.BLOCK):
        mgpu_utils.signal_with_red(sem_mc_ptr)
        mgpu_utils.wait_loop(sem_uc_ptr, num_gpus)

  return as_gpu_kernel(
      _main,
      (cta_count, 1, 1),
      block,
      (
          jax.ShapeDtypeStruct((m, k), lhs_dtype),
          jax.ShapeDtypeStruct((n, k) if rhs_transpose else (k, n), rhs_dtype),
          jax.ShapeDtypeStruct((m, n), out_dtype),
          jax.ShapeDtypeStruct((2*cta_count,), jnp.int32),
          jax.ShapeDtypeStruct((cta_count,), jnp.int32),
      ),
      (
          jax.ShapeDtypeStruct((m, n), out_dtype),
          jax.ShapeDtypeStruct((cta_count,), jnp.int32),
          jax.ShapeDtypeStruct((cta_count,), jnp.int32),
      ),
      (
          smem_shape,
          TMABarrier(num_barriers=stages),
      ),
      profiler_spec,
      input_output_aliases=((2,0),(3,1),(4,2),),
  )


def verify(x, y, actual_output):
  dimension_numbers = (((1,), (0,)), ((), ()))
  def lax_dot_general_psum(x, y):
    matmul_result = jax.lax.dot_general(
        x,
        y,
        dimension_numbers=dimension_numbers,
        preferred_element_type=dtype,
    )
    #Sum the result from all devices along the "x" axis
    ar_result = jax.lax.psum(matmul_result, axis_name='x')
    return ar_result.astype(dtype)

  jitted_ref_f = jax.jit(
      shard_map.shard_map(
          lax_dot_general_psum,
          mesh=mesh,
          in_specs=(P(None, 'x'), P('x', None)),
          out_specs=P(None, None),
      )
  )

  desired_output = jitted_ref_f(x, y)
  np.testing.assert_allclose(
      actual_output.astype(jnp.float32), desired_output.astype(jnp.float32), atol=1e-3, rtol=1e-3
  )


if __name__ == "__main__":
  jax.distributed.initialize()
  num_gpus = jax.device_count()
  assert num_gpus == 8, f"Expected 8 devices, but got {num_gpus}."
  devices = mesh_utils.create_device_mesh((num_gpus,))
  mesh = jax.sharding.Mesh(devices, ("x",))
  sharding_x = jax.sharding.NamedSharding(mesh, P(None, 'x'))
  sharding_y = jax.sharding.NamedSharding(mesh, P('x', None))

  dtype = jnp.dtype(jnp.float16)
  m, k, n = 64, 32768, 8192
  kx, ky = random.split(random.key(1234))
  x = random.uniform(kx, (m, k), dtype=dtype) * 0.001
  x = jax.device_put(x, sharding_x)
  y = random.uniform(ky, (k, n), dtype=dtype) * 0.001
  y = jax.device_put(y, sharding_y)
  assert k % 8 == 0, f"Expected k to be divisible by {num_gpus} got {k}."
  local_k = k//8

  tile_m = tile_n = (64, 128)
  swizzle = (128,)
  stages = (2, 4, 5, 6)
  cta_count = mosaic_gpu_lib._mosaic_gpu_ext._get_gpu_sm_count()
  configs = itertools.product(tile_m, tile_n, stages, swizzle)
  names = ("tile_m", "tile_n", "stages", "swizzle")
  best_runtime = float("inf")
  best_kwargs = {}

  for config in configs:
    kwargs = dict(zip(names, config))
    if n < kwargs["tile_n"]:
      continue
    try:
      @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
      def gemm_ar(inp_a, inp_b, dtype, m, n, k, cta_count, num_gpus, stages, tile_m, tile_n, swizzle):

        gemm_ar_kernel = build_kernel(
            m, n, k, dtype, dtype, dtype, cta_count = cta_count, num_gpus=num_gpus, stages=stages,
            tile_m=tile_m, tile_n=tile_n, swizzle=swizzle, wgmma_impl=WGMMADefaultImpl,
          )

        sharded_gemm_ar_kernel = shard_map.shard_map(
          gemm_ar_kernel,
          mesh=mesh,
          in_specs=(P(None, 'x'), P('x', None), P(None, None), P(None,), P(None,)),
          out_specs=P(None, None),
          check_rep=False,
        )

        def kernel_call(i, init_val):
          z = init_val
          z = jnp.zeros(m * n,dtype=dtype).reshape(m, n)
          sem_in = jnp.zeros(cta_count,dtype=jnp.int32).reshape(cta_count,)
          sem_out = jnp.zeros(cta_count,dtype=jnp.int32).reshape(cta_count,)
          z, sem_in, sem_out = sharded_gemm_ar_kernel(inp_a,inp_b, z, sem_in, sem_out)
          return z

        z = jnp.zeros(m * n,dtype=dtype).reshape(m, n)
        z = lax.fori_loop(0, 5, kernel_call, (z))
        return z

      # warm up call
      jax.experimental.multihost_utils.sync_global_devices('barrier')
      z = jax.block_until_ready(gemm_ar(
          x, y, dtype, m, n, local_k, cta_count, num_gpus, kwargs["stages"],
          kwargs["tile_m"], kwargs["tile_n"], kwargs["swizzle"],
          )
      )
      jax.experimental.multihost_utils.sync_global_devices('barrier')

      # profile gemm+ar kernel
      jax.experimental.multihost_utils.sync_global_devices('barrier')
      (z), mgpu_runtime_ms = profiler.measure(
        gemm_ar,mode='cupti',aggregate=False)(
        x, y, dtype, m, n, local_k, cta_count, num_gpus, kwargs["stages"],
        kwargs["tile_m"],kwargs["tile_n"], kwargs["swizzle"]
      )
      jax.experimental.multihost_utils.sync_global_devices('barrier')
      last_mosaic_gpu_runtime = None
      for det_runtime in mgpu_runtime_ms:
        if 'mosaic_gpu__main_kernel' in det_runtime[0]:
          last_mosaic_gpu_runtime = det_runtime[1]
      if last_mosaic_gpu_runtime < best_runtime:
        best_runtime = last_mosaic_gpu_runtime
        best_kwargs = kwargs
      verify(x,y,z)
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" not in str(e):
        raise
  print("Best parameters for GEMM+AR: ", " ".join(f"{k}={v}" for k, v in best_kwargs.items()))
  print(f"GEMM+AR mosaic-gpu kernel time={best_runtime:.4f}")
