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
"""Matmul kernels for H100."""

import contextlib
import enum
import jax
from jax import random
from jax._src.interpreters import mlir
from jax._src import test_util as jtu  # DO NOT REMOVE! This defines flags.
from jax.experimental.mosaic import gpu as mosaic_gpu
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.mosaic.gpu.dsl import *  # noqa: F403
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvgpu
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector
import numpy as np

# mypy: ignore-errors
# ruff: noqa: F405
# pylint: disable=line-too-long, wildcard-import, missing-function-docstring, bad-continuation, g-bad-todo, protected-access

SmemRef = ir.Value


class F32Precision(enum.Enum):
  DEFAULT = enum.auto()
  TF32_X3 = enum.auto()


class WGMMADefaultImpl:
  """Default WGMMA implementation."""

  @staticmethod
  def zero_accs(tile_m: int, tile_n: int) -> WGMMAAccumulator:
    return WGMMAAccumulator.zero(tile_m, tile_n)

  @staticmethod
  def smem_shape_extra(
      lhs_tile: jax.ShapeDtypeStruct,
      rhs_tile: jax.ShapeDtypeStruct,
  ) -> dict[str, jax.ShapeDtypeStruct]:
    del lhs_tile, rhs_tile
    return {}

  @staticmethod
  def get_result_tile(acc: WGMMAAccumulator) -> FragmentedArray:
    return acc.value

  @staticmethod
  def wgmma(
      smem_scratch: dict[str, SmemRef],  # pylint: disable=unused-argument
      acc: WGMMAAccumulator,
      b_order: WGMMALayout,
      a_slice: SmemRef,
      b_slice: SmemRef,
  ) -> dict[str, WGMMAAccumulator]:
    acc = wgmma(acc, a_slice, b_slice, b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    nvvm.wgmma_wait_group_sync_aligned(1)
    return acc


class WGMMATF32x3Impl:
  """WGMMA implementation for 3xTF32 precision."""

  @staticmethod
  def zero_accs(tile_m, tile_n) -> dict[str, WGMMAAccumulator]:
    zero_acc = WGMMADefaultImpl.zero_accs(tile_m, tile_n)
    return {"main": zero_acc, "errs": zero_acc}

  @staticmethod
  def smem_shape_extra(
      lhs_tile: jax.ShapeDtypeStruct,
      rhs_tile: jax.ShapeDtypeStruct,
  ) -> dict[str, jax.ShapeDtypeStruct]:
    return {"lhs_err": lhs_tile, "rhs_err": rhs_tile}

  @staticmethod
  def get_result_tile(accs) -> FragmentedArray:
    return accs["main"].value + accs["errs"].value

  @staticmethod
  def rounding_error(x_ref, err_ref):
    """Store the TF32 rounding error of x_ref in err_ref."""
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    t = FragmentedArray.load_strided(x_ref)
    tf32_mask = FragmentedArray.splat(c(0xFFFFE000, i32), t.shape, t.layout)
    t_tf32 = (t.bitcast(i32) & tf32_mask).bitcast(f32)
    (t - t_tf32).store_untiled(err_ref)

  @staticmethod
  def wgmma(
      smem_scratch: dict[str, SmemRef],
      accs: dict[str, WGMMAAccumulator],
      b_order: WGMMALayout,
      a_slice: SmemRef,
      b_slice: SmemRef,
  ) -> dict[str, WGMMAAccumulator]:
    acc = wgmma(accs["main"], a_slice, b_slice, b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    # Note: we assert that only the slice_ab and err_b mmas are still running
    # which are unaffected by writing to the err_a shared memory.
    # After nvvm.wgmma_wait_group_sync_aligned(2) there are no wgmmas
    # accessing err_a so we can safely write to it.
    nvvm.wgmma_wait_group_sync_aligned(2)
    WGMMATF32x3Impl.rounding_error(a_slice, smem_scratch["lhs_err"])
    commit_shared()
    acc_err = wgmma(accs["errs"], smem_scratch["lhs_err"], b_slice, b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    # Note: similar to the above we wait for the last wgmma access to
    # err_b which was 2 wgmmas ago.
    nvvm.wgmma_wait_group_sync_aligned(2)
    WGMMATF32x3Impl.rounding_error(b_slice, smem_scratch["rhs_err"])
    commit_shared()
    acc_err = wgmma(acc_err, a_slice, smem_scratch["rhs_err"], b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    nvvm.wgmma_wait_group_sync_aligned(2)
    return {"main": acc, "errs": acc_err}


def mlir_context(f):
  def wrap(*args, **kw):
    with mlir.make_ir_context(), ir.Location.unknown():
      return f(*args, **kw)

  return wrap

@mlir_context
def build_kernel(
    m: int,
    k: int,
    n: int,
    in_dtype: jnp.dtype,
    stages: int = 2,
    tile_m: int = 128,
    tile_n: int = 128,
    rhs_transpose: bool = False,
    wgmma_impl=WGMMADefaultImpl,
    profiler_spec: profiler.ProfilerSpec | None = None,
):
  in_dtype = jnp.dtype(in_dtype)
  if in_dtype == jnp.float16:
    in_mlir_dtype = ir.F16Type.get()
  elif in_dtype == jnp.bfloat16:
    in_mlir_dtype = ir.BF16Type.get()
  elif in_dtype == jnp.float32:
    in_mlir_dtype = ir.F32Type.get()
  else:
    raise ValueError(f"Unsupported input dtype: {in_dtype}")

  in_bytewidth = bytewidth(in_mlir_dtype)
  in_128b_elems = 128 // in_bytewidth
  out_128b_elems = 128 // bytewidth(ir.F32Type.get())
  tile_k = in_128b_elems
  if tile_m % 64 != 0:
    raise ValueError(f"{tile_m=} must be divisible by 64")
  if tile_n % in_128b_elems != 0:
    raise ValueError(
        f"{tile_n=} must be divisible by 128 bytes ="
        f" {in_128b_elems} {in_mlir_dtype} elements"
    )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if k % (stages * tile_k) != 0:
    raise ValueError(
        f"k must be divisible by {stages=} * {tile_k=} (={stages * tile_k}),"
        f" but got {k=}"
    )
  if n % 64 != 0:
    raise ValueError(f"n must be divisible by 64, but got {n=}")
  if stages < 2:
    raise ValueError(f"Need at least 2 stages, but got {stages=}")

  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")

  assert tile_k == in_128b_elems
  lhs_tiling = (64, in_128b_elems)
  lhs_tile = jax.ShapeDtypeStruct(tile_shape((tile_m, tile_k), lhs_tiling), in_dtype)
  rhs_tiling = (in_128b_elems, in_128b_elems)
  rhs_tile = jax.ShapeDtypeStruct(tile_shape((tile_k, tile_n), rhs_tiling), in_dtype)
  out_tiling = (64, out_128b_elems)

  f32 = ir.F32Type.get()
  index = ir.IndexType.get()
  i8 = ir.IntegerType.get_signless(8)

  def safe_div(x, y):
    assert x % y == 0
    return x // y

  grid = (safe_div(m, tile_m), safe_div(n, tile_n), 1)
  block = (128, 1, 1)

  def c(value, ty=index):
    return arith.ConstantOp(ty, ir.IntegerAttr.get(ty, value))

  smem_shape = {
      "lhs": jax.ShapeDtypeStruct((stages, *lhs_tile.shape), lhs_tile.dtype),
      "rhs": jax.ShapeDtypeStruct((stages, *rhs_tile.shape), rhs_tile.dtype),
  }
  smem_shape |= wgmma_impl.smem_shape_extra(lhs_tile, rhs_tile)

  def _main(ctx, a_device, b_device, c_device, smem_scratch):
    memref.assume_alignment(c_device, 16)

    barrier_group = BarrierArray(stages)
    m_start = arith.muli(c(tile_m), gpu.block_id(gpu.Dimension.x))
    n_start = arith.muli(c(tile_n), gpu.block_id(gpu.Dimension.y))

    def fetch(slot, ki):
      barrier = barrier_group[slot]
      k_start = arith.muli(c(tile_k), ki)
      lhs_tile_bytes = np.prod(lhs_tile.shape) * in_bytewidth
      rhs_tile_bytes = np.prod(rhs_tile.shape) * in_bytewidth
      txcount = c(lhs_tile_bytes + rhs_tile_bytes)
      common_copy_args = dict(
          swizzle=128, barrier=barrier, arrive=False, uniform=False,
      )
      with once():
        nvgpu.mbarrier_arrive_expect_tx(barrier_group.value, txcount, slot)
        ctx.async_copy(
            src_ref=a_device,
            dst_ref=memref_slice(smem_scratch["lhs"], slot),
            gmem_slice=(ds(m_start, tile_m), ds(k_start, tile_k)),
            gmem_transform=mosaic_gpu.TileTransform(lhs_tiling),
            **common_copy_args,
        )
        rhs_slice = (ds(k_start, tile_k), ds(n_start, tile_n))
        rhs_transform = (mosaic_gpu.TileTransform(rhs_tiling),)
        if rhs_transpose:
          rhs_slice = rhs_slice[::-1]
          rhs_transform += (mosaic_gpu.TransposeTransform((1, 0, 2, 3)),)
          assert rhs_tiling[0] == rhs_tiling[1]  # No need to flip the tiling.
        ctx.async_copy(
            src_ref=b_device,
            dst_ref=memref_slice(smem_scratch["rhs"], slot),
            gmem_slice=rhs_slice,
            gmem_transform=rhs_transform,
            **common_copy_args,
        )

    k_steps = k // tile_k
    accs = wgmma_impl.zero_accs(tile_m, tile_n)


    with ctx.named_region("TMA warmup"):
      for i in range(stages):
        fetch(c(i), c(i))

    @fori(c(k_steps), accs)
    def stage_loop_body(ki, accs):
      si = arith.remui(ki, c(stages))

      with ctx.named_region("TMA wait"):
        barrier_group[si].wait()

      with ctx.named_region("WGMMA"):
        a_slice = memref_slice(smem_scratch["lhs"], si)
        b_slice = memref_slice(smem_scratch["rhs"], si)
        b_order = (
            WGMMALayout.COL_MAJOR if rhs_transpose else WGMMALayout.ROW_MAJOR
        )
        accs = wgmma_impl.wgmma(smem_scratch, accs, b_order, a_slice, b_slice)

      with ctx.named_region("TMA start"):
        tma_ki = arith.addi(ki, c(stages - 1))
        do_tma = arith.cmpi(arith.CmpIPredicate.slt, tma_ki, c(k_steps))
        not_first_step = arith.cmpi(arith.CmpIPredicate.ne, ki, c(0))
        if_op = scf.IfOp(arith.andi(not_first_step, do_tma))
        with ir.InsertionPoint(if_op.then_block):
          tma_si = arith.remui(tma_ki, c(stages))
          fetch(tma_si, tma_ki)
          scf.yield_([])

      return accs

    # Wait until everyone is done with their WMMA
    with ctx.named_region("WGMMA drain"):
      nvvm.wgmma_wait_group_sync_aligned(0)

    # We can repurpose the tile SMEM for the epilogue now
    # TODO(apaszke): Add support for aliased SMEM allocations to the DSL
    dynamic_smem = gpu.dynamic_shared_memory(
        ir.MemRefType.get(
            (ir.ShapedType.get_dynamic_size(),), i8, memory_space=smem
        )
    )
    with ctx.named_region("SMEM store"):
      acc_val = wgmma_impl.get_result_tile(stage_loop_body.result)
      acc_smem = memref.view(
          ir.MemRefType.get(
              tile_shape((tile_m, tile_n), out_tiling), f32,
              memory_space=smem
          ),
          dynamic_smem, c(0), [],
      )
      acc_val.store_tiled(acc_smem, swizzle=128)
      gpu.barrier()

    with ctx.named_region("GMEM store"):
      # Vectorized epilogue to move results from SMEM to GMEM
      # TODO(apaszke): Make this into a proper copy function.
      warps_per_warpgroup = 4
      lanes_per_warp = 32
      n_out_tiling = out_tiling[-1]
      tidx = gpu.thread_id(gpu.Dimension.x)
      warp_id = arith.divui(tidx, c(lanes_per_warp))
      lane_id = arith.remui(tidx, c(lanes_per_warp))
      # We store 4 f32 numbers for a block of 16B.
      vector_len = 4
      num_vectors = safe_div(tile_n, vector_len)
      for_op = scf.ForOp(warp_id, c(tile_m), c(warps_per_warpgroup))
      with ir.InsertionPoint(for_op.body):
        nested_for_op = scf.ForOp(lane_id, c(num_vectors), c(lanes_per_warp))
        with ir.InsertionPoint(nested_for_op.body):
          vector_idx = nested_for_op.induction_variable
          n_store = arith.muli(vector_idx, c(vector_len))
          col_group = arith.divui(n_store, c(n_out_tiling))
          n_load = arith.remui(n_store, c(n_out_tiling))

          m_smem = for_op.induction_variable
          m_within_tile = arith.remui(m_smem, c(64))
          m_tile = arith.divui(m_smem, c(64))
          swizzle_source = arith.shli(arith.remui(m_smem, c(8)), c(2))
          n_acc = arith.xori(n_load, swizzle_source)
          acc_part = vector.load(
              ir.VectorType.get((vector_len,), f32),
              acc_smem,
              [m_tile, col_group, m_within_tile, n_acc],
          )
          vector.store(
              acc_part,
              c_device,
              [arith.addi(m_start, m_smem), arith.addi(n_start, n_store)],
          )
          scf.yield_([])
        scf.yield_([])

  rhs_shape = (n, k) if rhs_transpose else (k, n)
  return mosaic_gpu.as_gpu_kernel(
      _main,
      grid,
      block,
      (
          jax.ShapeDtypeStruct((m, k), in_dtype),
          jax.ShapeDtypeStruct(rhs_shape, in_dtype),
      ),
      jax.ShapeDtypeStruct((m, n), jnp.float32),
      smem_shape,
      profiler_spec,
  )


def verify(
    m=(33 * 128),
    k=2048,
    n=(4 * 128),
    stages=4,
    tile_m=128,
    tile_n=128,
    profile=False,
    in_dtype=jnp.float16,
    rhs_transpose=False,
    precision: F32Precision = F32Precision.DEFAULT,
):
  in_dtype = jnp.dtype(in_dtype)
  kx, ky = random.split(random.key(1234))
  x = random.uniform(kx, (m, k), dtype=in_dtype)
  y = random.uniform(ky, (n, k) if rhs_transpose else (k, n), dtype=in_dtype)

  match precision:
    case F32Precision.DEFAULT:
      impl = WGMMADefaultImpl
    case F32Precision.TF32_X3:
      impl = WGMMATF32x3Impl

  prof_spec = profiler.ProfilerSpec(132 * 4096) if profile else None
  f = build_kernel(
      m,
      k,
      n,
      stages=stages,
      tile_m=tile_m,
      tile_n=tile_n,
      in_dtype=in_dtype,
      rhs_transpose=rhs_transpose,
      wgmma_impl=impl,
      profiler_spec=prof_spec,
  )
  z, runtime = profiler.measure(f, x, y)

  if rhs_transpose:
    dimension_numbers = ((1,), (1,)), ((), ())
  else:
    dimension_numbers = ((1,), (0,)), ((), ())
  if in_dtype == jnp.dtype(jnp.float32):  # Account for the tf32 precision
    exponent_bits, mantissa_bits = 8, 10
    x, y = (
        jax.lax.reduce_precision(v, exponent_bits, mantissa_bits)
        for v in (x, y)
    )
  ref = jax.lax.dot_general(
      x, y, dimension_numbers,
      preferred_element_type=jnp.float32,
  )
  np.testing.assert_allclose(z, ref, atol=1e-3, rtol=1e-3)
  return runtime


if __name__ == "__main__":
  m, k, n = 33 * 128, 2048, 4 * 128
  runtime = verify(m=m, k=k, n=n)
  tflops = float(2 * k * m * n) / (runtime / 1e3) / 1e12
  print(f"{runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
