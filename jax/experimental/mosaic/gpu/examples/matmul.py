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

import dataclasses
import functools
from typing import Any
import math

import jax
from jax import random
from jax._src.interpreters import mlir
from jax.experimental.mosaic import gpu as mosaic_gpu
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.mosaic.gpu.dsl import *  # noqa: F403
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector
import numpy as np

# mypy: ignore-errors
# ruff: noqa: F405
# pylint: disable=line-too-long, wildcard-import, missing-function-docstring, bad-continuation, g-bad-todo, protected-access

SmemRef = ir.Value


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
      rhs_transpose: WGMMALayout,
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
      b_order: WGMMALayout,
      a_slice: SmemRef,
      b_slice: SmemRef,
      swizzle: int,
  ) -> dict[str, WGMMAAccumulator]:
    """Perform a matrix multiplication.

    This function must guarantee that all WGMMA operations queued before it was
    called have completed before returning.
    """
    acc = wgmma(acc, a_slice, b_slice, b_order=b_order, swizzle=swizzle)
    nvvm.wgmma_commit_group_sync_aligned()
    nvvm.wgmma_wait_group_sync_aligned(1)
    return acc


def mlir_context(f):
  def wrap(*args, **kw):
    with mlir.make_ir_context(), ir.Location.unknown():
      return f(*args, **kw)

  return wrap

@mlir_context
def build_kernel(
    m, n, k,
    lhs_dtype, rhs_dtype, out_dtype,
    stages: int = 2,
    tile_m: int = 128,
    tile_n: int = 128,
    swizzle: int = 128,
    cluster: tuple[int, int] = (1, 1),
    rhs_transpose: bool = False,
    wgmma_impl=WGMMADefaultImpl,
    profiler_spec: profiler.ProfilerSpec | None = None,
):
  f32 = ir.F32Type.get()
  if tile_m % 64 != 0:
    raise ValueError(f"{tile_m=} must be divisible by 64")
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % 64 != 0:
    raise ValueError(f"n must be divisible by 64, but got {n=}")
  if stages < 2:
    raise ValueError(f"Need at least 2 stages, but got {stages=}")
  if not rhs_transpose and jnp.dtype(rhs_dtype).itemsize != 2:
    raise ValueError("Transpose only supported for only happen for 16bit types")
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

  grid = (safe_div(m, block_tiling.m), safe_div(n, block_tiling.n), 1)
  block = (128, 1, 1)

  c = arith.ConstantOp.create_index

  compute_scratch_shape = (
      jax.ShapeDtypeStruct((stages, *tile_shape(block_tiling.mk, tma_tiling.mk)), lhs_dtype),
      jax.ShapeDtypeStruct((stages, *tile_shape(block_tiling.kn, tma_tiling.kn)), rhs_dtype),
      wgmma_impl.smem_shape_extra(block_tiling, tma_tiling, lhs_dtype, rhs_dtype, rhs_transpose),
  )
  epilogue_scratch_shape = jax.ShapeDtypeStruct(out_tile.shape, out_tile.dtype)
  smem_shape = mosaic_gpu.Union([compute_scratch_shape, epilogue_scratch_shape])

  def _main(ctx, a_device, b_device, c_device, smem):
    ((lhs_smem, rhs_smem, impl_smem), epilogue_smem), *barriers = smem
    tma_barriers, cluster_barrier = barriers

    memref.assume_alignment(c_device, 16)

    m_start = arith.muli(c(block_tiling.m), gpu.block_id(gpu.Dimension.x))
    n_start = arith.muli(c(block_tiling.n), gpu.block_id(gpu.Dimension.y))

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
            gmem_transform=mosaic_gpu.TileTransform(tma_tiling.mk),
            collective=gpu.Dimension.y,
            **common_copy_args,
        )
        rhs_slice = (ds(k_start, block_tiling.k), ds(n_start, block_tiling.n))
        rhs_transform = (mosaic_gpu.TileTransform(tma_tiling.kn),)
        if rhs_transpose:
          rhs_slice = rhs_slice[::-1]
          rhs_transform += (mosaic_gpu.TransposeTransform((1, 0, 2, 3)),)
          assert tma_tiling.n == tma_tiling.k, block_tiling  # No need to flip the tiling.
        ctx.async_copy(
            src_ref=b_device,
            dst_ref=memref_slice(rhs_smem, slot),
            gmem_slice=rhs_slice,
            gmem_transform=rhs_transform,
            collective=gpu.Dimension.x,
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
        rhs_smem_order = (
            WGMMALayout.COL_MAJOR if rhs_transpose else WGMMALayout.ROW_MAJOR
        )
        accs = wgmma_impl.wgmma(
            impl_smem, accs, rhs_smem_order, a_slice, b_slice, swizzle=swizzle
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
          if cluster_barrier is not None:
            with ctx.named_region("Cluster barrier"):
              cluster_barrier[tma_si].arrive()
              cluster_barrier[tma_si].wait()  # Make sure everyone is done.
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
          gmem_transform=mosaic_gpu.TileTransform(out_tiling),
          swizzle=out_swizzle,
      )
      ctx.await_async_copy(0)

  return mosaic_gpu.as_gpu_kernel(
      _main,
      grid,
      block,
      (
          jax.ShapeDtypeStruct((m, k), lhs_dtype),
          jax.ShapeDtypeStruct((n, k) if rhs_transpose else (k, n), rhs_dtype),
      ),
      jax.ShapeDtypeStruct((m, n), out_dtype),
      (
          smem_shape,
          TMABarrier(num_barriers=stages),
          ClusterBarrier(
              collective_dims=(gpu.Dimension.x, gpu.Dimension.y),
              num_barriers=stages,
          ) if math.prod(cluster) > 1 else None,
      ),
      profiler_spec,
      cluster=(*cluster, 1),
  )


def verify(
    m=(33 * 128),
    k=2048,
    n=(4 * 128),
    stages=4,
    tile_m=128,
    tile_n=128,
    cluster_m=1,
    cluster_n=1,
    swizzle=128,
    profile=False,
    in_dtype=jnp.float16,
    out_dtype=jnp.float32,
    rhs_transpose=False,
):
  lhs_dtype, rhs_dtype = in_dtype, in_dtype

  kx, ky = random.split(random.key(1234))
  x = random.uniform(kx, (m, k), dtype=lhs_dtype)
  y = random.uniform(ky, (n, k) if rhs_transpose else (k, n), dtype=rhs_dtype)

  prof_spec = profiler.ProfilerSpec(4096) if profile else None
  f = build_kernel(
      m, n, k,
      jnp.dtype(lhs_dtype), jnp.dtype(rhs_dtype), jnp.dtype(out_dtype),
      stages=stages,
      tile_m=tile_m,
      tile_n=tile_n,
      cluster=(cluster_m, cluster_n),
      rhs_transpose=rhs_transpose,
      swizzle=swizzle,
      wgmma_impl=WGMMADefaultImpl,
      profiler_spec=prof_spec,
  )
  z, runtime = profiler.measure(f, x, y)

  if rhs_transpose:
    dimension_numbers = ((1,), (1,)), ((), ())
  else:
    dimension_numbers = ((1,), (0,)), ((), ())
  if lhs_dtype == jnp.dtype(jnp.float32):  # Account for the tf32 precision
    exponent_bits, mantissa_bits = 8, 10
    x, y = (
        jax.lax.reduce_precision(v, exponent_bits, mantissa_bits)
        for v in (x, y)
    )

  @jax.jit
  def ref_f(x, y):
    return jax.lax.dot_general(
        x,
        y,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    ).astype(out_dtype)

  ref, ref_runtime = profiler.measure(ref_f, x, y)
  np.testing.assert_allclose(
      z.astype(jnp.float32), ref.astype(jnp.float32), atol=1e-3, rtol=1e-3
  )
  return runtime, ref_runtime


if __name__ == "__main__":
  m, k, n = 4 * 33 * 128, 2048, 4 * 128
  runtime, ref_runtime = verify(m=m, k=k, n=n, cluster_m=1, cluster_n=4)
  tflops = float(2 * k * m * n) / (runtime / 1e3) / 1e12
  ref_tflops = float(2 * k * m * n) / (ref_runtime / 1e3) / 1e12
  print(f"Kernel:    {runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
  print(f"Reference: {ref_runtime * 1000:.1f} us = {ref_tflops:.1f} TFLOPS")
