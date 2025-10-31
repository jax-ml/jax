# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Matrix Multiplication kernel for Blackwell GPUs."""
import dataclasses
import enum
import functools
import itertools
import statistics

import jax
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

class MatmulDimension(enum.IntEnum):
  M = 0
  N = 1


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
  collective: bool
  epilogue_tile_n: int = 64
  grid_minor_dim: MatmulDimension = MatmulDimension.N
  grid_tile_width: int = 1


def matmul_kernel(a, b, config: TuningConfig):
  dtype = a.dtype
  if a.dtype != b.dtype:
    raise ValueError(
        f"Matmul LHS and RHS have incompatible dtypes {a.dtype} vs {b.dtype}"
    )
  m, k = a.shape
  k2, n = b.shape
  if k != k2:
    raise ValueError(
        f"Matmul LHS and RHS have incompatible shapes {a.shape} vs {b.shape}"
    )
  collective = config.collective
  tile_m, tile_n, tile_k = (config.tile_m, config.tile_n, config.tile_k)
  epilogue_tile_n = config.epilogue_tile_n
  if tile_n % epilogue_tile_n != 0:
    raise ValueError(
        f"{tile_n=} must be divisible by {epilogue_tile_n=}"
    )
  block_tile_m = tile_m
  block_tile_n = tile_n
  if collective:
    tile_m *= 2
    tile_n *= 2
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  out_swizzle = plgpu.find_swizzle(epilogue_tile_n * jnp.dtype(dtype).itemsize * 8)
  out_swizzle_elems = out_swizzle // jnp.dtype(dtype).itemsize
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  TMA_WARP = 0
  MMA_WARP = 1
  COMPUTE_WG = 0
  STORE_WG = 1

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             ab_tma_barrier, store_done_barrier, mma_done_barrier,
             consumed_barrier):
    wg_idx = lax.axis_index("wg")
    cluster_idx = lax.axis_index("x")
    is_lead_block = cluster_idx == 0

    @plgpu.dynamic_scheduling_loop(grid_names=("mn_linear",), thread_axis="wg")
    def mn_loop(loop_info: plgpu.NDLoopInfo):  # pylint: disable=unused-variable
      (lin_idx,) = loop_info.index
      local_index = loop_info.local_index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      block_m_index = m_index * 2 + cluster_idx if collective else m_index

      block_slice_m = pl.ds(block_m_index * block_tile_m, block_tile_m)
      slice_m = pl.ds(m_index * tile_m, tile_m)
      slice_n = pl.ds(n_index * tile_n, tile_n)
      acc_slot = lax.rem(local_index, jnp.int32(2))

      @pl.when(wg_idx == COMPUTE_WG)
      def _():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")
          @pl.when(warp_id == TMA_WARP)
          def _memory():
            def _loop_body(ki, _):
              slice_k = pl.ds(ki * tile_k, tile_k)
              slot = lax.rem(ki, max_concurrent_steps)
              @pl.when(jnp.logical_or(ki >= max_concurrent_steps,
                                      local_index > 0))
              def _():
                plgpu.barrier_wait(consumed_barrier.at[slot])
              plgpu.copy_gmem_to_smem(
                  a_gmem.at[slice_m, slice_k],
                  a_smem.at[slot],
                  ab_tma_barrier.at[slot],
                  partitioned_axis=0 if collective else None,
                  collective_axes="x" if collective else None,
              )
              plgpu.copy_gmem_to_smem(
                  b_gmem.at[slice_k, slice_n],
                  b_smem.at[slot],
                  ab_tma_barrier.at[slot],
                  partitioned_axis=1 if collective else None,
                  collective_axes="x" if collective else None,
              )
            lax.fori_loop(0, k_iters, _loop_body, None)

          @pl.when(jnp.logical_and(warp_id == MMA_WARP, local_index > 1))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])
          @pl.when(jnp.logical_and(warp_id == MMA_WARP, is_lead_block))
          def _compute():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              plgpu.barrier_wait(ab_tma_barrier.at[slot])

              is_last_iter = ki >= k_iters - 1
              acc_tmem_slice = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
              plgpu.tcgen05_mma(
                  acc_tmem_slice,
                  a_smem.at[slot],
                  b_smem.at[slot],
                  consumed_barrier.at[slot],
                  accumulate=(ki > 0),
                  collective_axis="x" if collective else None,
              )
              @pl.when(is_last_iter)
              def _():
                plgpu.tcgen05_commit_arrive(
                    mma_done_barrier.at[acc_slot],
                    collective_axis="x" if collective else None,
                )

            lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(wg_idx == STORE_WG)
      def _():
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        acc_tmem_slot = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
        step_out_gmem = out_gmem.at[block_slice_m, slice_n]
        for ni in range(tile_n // epilogue_tile_n):
          acc_smem_ni = acc_smem.at[ni % 2]
          ni_col_slice = pl.ds(ni * epilogue_tile_n, epilogue_tile_n)
          acc_smem_ni[...] = plgpu.async_load_tmem(
              acc_tmem_slot.at[:, ni_col_slice]
          ).astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(acc_smem_ni, step_out_gmem.at[:, ni_col_slice])
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
        plgpu.wait_load_tmem()  # Load must complete before we continue.
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])

  if collective:
    store_done_barrier = plgpu.ClusterBarrier(
        collective_axes=("x",),
        num_arrivals=1,
        num_barriers=2,
        orders_tensor_core=True,
    )
  else:
    store_done_barrier = plgpu.Barrier(  # type: ignore
        num_arrivals=1, num_barriers=2, orders_tensor_core=True
    )
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters * n_iters,),
      grid_names=("mn_linear",),
      num_threads=2,
      thread_name="wg",
      cluster_names=("x",),
      cluster=(1 + collective,),
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, block_tile_m, tile_k),
              dtype,
              transforms=transforms,
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, block_tile_n),
              dtype,
              transforms=transforms,
          ),
          acc_tmem=plgpu.TMEM(
              (block_tile_m, tile_n * 2), jnp.float32, collective=collective
          ),
          acc_smem=plgpu.SMEM(
              (2, block_tile_m, epilogue_tile_n),
              dtype,
              transforms=out_transforms,
          ),
          ab_tma_barrier=plgpu.Barrier(
              num_arrivals=2, num_barriers=max_concurrent_steps
          ),
          store_done_barrier=store_done_barrier,
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
          consumed_barrier=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
      ),
  )
  return f(a, b)


def main(_) -> None:
  problem_it = [(4096, 8192, 4096)]
  for M, N, K in problem_it:
    print(f"==== {M=} {N=} {K=} ====")
    matmul_flops = 2 * M * N * K
    peak_flops = 2.25e15  # f16 TensorCore peak = 2250 TFLOPS
    a = jax.random.uniform(jax.random.key(1), (M, K), jnp.float16, -1, 1)
    b = jax.random.uniform(jax.random.key(2), (K, N), jnp.float16, -1, 1)
    tuning_it = itertools.product(
        (128,),  # tile_m
        (128, 256),  # tile_n
        (64,),  # tile_k
        MatmulDimension,  # grid_minor_dim
        (1, 4, 8, 12, 16),  # grid_tile_width
        (2, 4, 6),  # max_concurrent_steps
        (False, True),  # collective
        (32,),  # epilogue_tile_n
    )
    best_util = -float("inf")
    expected = jnp.dot(a, b, precision=jax.lax.DotAlgorithmPreset.F16_F16_F32)
    for (tile_m, tile_n, tile_k, grid_minor_dim, grid_tile_width,
         max_concurrent_steps, collective, epilogue_tile_n) in tuning_it:
      # Only N <= 128 are supported for collective MMAs
      if collective and tile_n > 128:
        continue
      config = TuningConfig(
          tile_m=tile_m,
          tile_n=tile_n,
          tile_k=tile_k,
          max_concurrent_steps=max_concurrent_steps,
          collective=collective,
          epilogue_tile_n=epilogue_tile_n,
          grid_minor_dim=grid_minor_dim,
          grid_tile_width=grid_tile_width,
      )
      if collective:
        tile_m *= 2
        tile_n *= 2
      try:
        out, runtimes_ms = profiler.measure(
            functools.partial(matmul_kernel, config=config), iterations=10
        )(a, b)
        assert runtimes_ms is not None
        runtime_ms = statistics.median(runtimes_ms)
      except ValueError as e:
        if ("exceeds available shared memory" in e.args[0] or
            "Accumulator layout mismatch:" in e.args[0]):
          # Accumulator layout mismatch triggers for tile_n=256 on some configs.
          continue
        raise
      runtime_us = runtime_ms * 1e3   # type: ignore
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      if achieved_tc_util > best_util:
        np.testing.assert_allclose(out, expected)
        best_util = achieved_tc_util
      print(
          f"{tile_m=} {tile_n=} {tile_k=} {max_concurrent_steps=} "
          f"{grid_minor_dim=} {grid_tile_width=} "
          f"{epilogue_tile_n=} "
          f"{collective=} : "
          f"{runtime_us:<7.1f}us"
          f" = {achieved_tc_util:4.1f}% TC utilization"
      )
    print(f"\tBest utilization: {best_util:4.1f}%")
    _, runtimes_ms = profiler.measure(
        functools.partial(
            jnp.dot, precision=jax.lax.DotAlgorithmPreset.F16_F16_F32
        ),
        iterations=10,
    )(a, b)
    assert runtimes_ms is not None
    runtime_ms = statistics.median(runtimes_ms)
    runtime_us = runtime_ms * 1e3   # type: ignore
    optimal_time = matmul_flops / peak_flops * 1e6  # us
    achieved_tc_util = optimal_time / runtime_us * 100
    print(f"\tReference: {achieved_tc_util:4.1f}%")


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
