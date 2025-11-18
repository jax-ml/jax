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
"""Ragged/Grouped Matrix Multiplication kernel for Blackwell GPUs."""
import dataclasses
import functools
import itertools
import math
import jax
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu
import jax.numpy as jnp
import numpy as np
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
  collective: bool
  grid_tile_width: int
  grid_minor_dim: blackwell_matmul_mgpu.MatmulDimension
  epilogue_tile_n: int = 64

  def __str__(self):
    return "_".join(f"{k}={v}" for k, v in dataclasses.asdict(self).items())


# TODO(justinfu): Merge with blackwell_matmul_mgpu.py
def do_matmul(a_gmem,
              b_gmem,
              out_gmem,
              grid_indices: Sequence[jax.Array],
              wg_axis: str,
              collective_axes: tuple[str, ...],
              local_index: jax.Array,
              config: TuningConfig,
              group_info: ragged_dot_mgpu.GroupInfo,
              a_smem, b_smem, acc_tmem, acc_smem,
              a_tma_barrier, b_tma_barrier, store_done_barrier, mma_done_barrier,
              consumed_barrier
              ):
  """Compute a non-ragged matmul for a single output block."""
  dtype = out_gmem.dtype
  m, k = a_gmem.shape
  collective = config.collective
  tile_m, tile_n, tile_k = (config.tile_m, config.tile_n, config.tile_k)
  epilogue_tile_n = config.epilogue_tile_n
  max_concurrent_steps = config.max_concurrent_steps
  block_tile_m = tile_m
  if collective:
    tile_m *= 2
    tile_n *= 2
  k_iters = k // tile_k

  if collective:
    m_index, n_index, cluster_idx = grid_indices
    block_m_index = m_index * 2 + cluster_idx
    is_lead_block = cluster_idx == 0
  else:
    m_index, n_index = grid_indices
    cluster_idx = 0  # type: ignore
    block_m_index = m_index
    is_lead_block = True  # type: ignore
  wg_idx = lax.axis_index(wg_axis)
  collective_axis = collective_axes[0] if collective else None

  TMA_WARP = 0
  MMA_WARP = 1
  COMPUTE_WG = 0
  STORE_WG = 1

  block_slice_m = pl.ds(block_m_index * block_tile_m, block_tile_m)
  slice_m = pl.ds(m_index * tile_m, tile_m)
  slice_n = pl.ds(n_index * tile_n, tile_n)
  acc_slot = lax.rem(local_index, jnp.int32(2))
  regs_layout = plgpu.Layout.TCGEN05

  @pl.when(wg_idx == COMPUTE_WG)
  @jax.named_scope("compute_wg")
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
              a_tma_barrier.at[slot],
              partitioned_axis=0 if collective else None,
              collective_axes=collective_axis,
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[slice_k, slice_n],
              b_smem.at[slot],
              b_tma_barrier.at[slot],
              partitioned_axis=1 if collective else None,
              collective_axes=collective_axis,
          )
        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(jnp.logical_and(warp_id == MMA_WARP, local_index > 1))
      def _wait_store():
        plgpu.barrier_wait(store_done_barrier.at[acc_slot])
      @pl.when(jnp.logical_and(warp_id == MMA_WARP, is_lead_block))
      def _compute():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(a_tma_barrier.at[slot])
          plgpu.barrier_wait(b_tma_barrier.at[slot])

          is_last_iter = ki >= k_iters - 1
          acc_tmem_slice = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
          plgpu.tcgen05_mma(
              acc_tmem_slice,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barrier.at[slot],
              accumulate=(ki > 0),
              collective_axis=collective_axis,
          )
          @pl.when(is_last_iter)
          def _():
            plgpu.tcgen05_commit_arrive(
                mma_done_barrier.at[acc_slot],
                collective_axis=collective_axis,
            )

        lax.fori_loop(0, k_iters, _loop_body, None)

  @pl.when(wg_idx == STORE_WG)
  @jax.named_scope("store_wg")
  def _():
    plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
    acc_tmem_slot = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
    step_out_gmem = out_gmem.at[block_slice_m, slice_n]
    # group_info contains start/size info relative to the logical
    # tiling (tile_m) but because for collective matmuls we use 2 CTAs per
    # logical block, but we need to compute the start/size relative to the
    # current block.
    # For example, for the following parameters:
    #     block_tile_m=64 (tile_m=128)
    #     group_info.start_within_block=60
    #     group_info.actual_size=37
    # The requested copy will be split across both blocks
    # Memory:         | Block 0  |  Block 1 |
    #                 |--- 64 ---|--- 64 ---|
    # Copy:                    |-- 37 --|
    # Where block 0 copies rows 60-64 (4 rows total) and block 1 copies
    # the remaining rows 64-97 (33 rows total).
    smem_start = group_info.start_within_block - cluster_idx * block_tile_m
    smem_start = lax.max(smem_start, jnp.int32(0))
    def _clamp(min, x, max):
      return lax.max(lax.min(x, max), min)
    block0_copy_size = _clamp(
        jnp.int32(0),
        block_tile_m - group_info.start_within_block,
        group_info.actual_size)
    block_local_size = lax.select(is_lead_block,
      # block 0 copies up to end of the first block or actual_size,
      # whichever comes first.
      block0_copy_size,
      # block 1 copies the remaining rows that block 0 did not copy.
      group_info.actual_size - block0_copy_size
    )
    for ni in range(tile_n // epilogue_tile_n):
      acc_smem[...] = plgpu.async_load_tmem(
          acc_tmem_slot.at[:, pl.ds(ni * epilogue_tile_n, epilogue_tile_n)],
          layout=regs_layout).astype(dtype)
      plgpu.commit_smem()
      cur_smem_idx = smem_start
      remaining_rows = min(block_tile_m, m)
      while remaining_rows > 0:
        const_rows_len = 1 << int(math.log2(remaining_rows))
        remaining_rows //= 2
        @pl.when(block_local_size & const_rows_len != 0)
        def _():
          o_smem_slice = acc_smem.at[pl.ds(cur_smem_idx, const_rows_len)]
          o_gref_slice = step_out_gmem.at[
              pl.ds(cur_smem_idx, const_rows_len),
              pl.ds(ni * epilogue_tile_n, epilogue_tile_n),
          ]
          plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)
        cur_smem_idx += block_local_size & const_rows_len
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)
    plgpu.wait_load_tmem()  # Load must complete before we continue.
    plgpu.barrier_arrive(store_done_barrier.at[acc_slot])


def ragged_dot_kernel(a, b, group_sizes, config: TuningConfig):
  dtype = a.dtype
  if a.dtype != b.dtype:
    raise ValueError(
        f"Matmul LHS and RHS have incompatible dtypes {a.dtype} vs {b.dtype}"
    )
  m, k = a.shape
  num_groups, k2, n = b.shape
  if num_groups != group_sizes.shape[0]:
    raise ValueError("RHS and group_sizes have incompatible shapes.")
  if k != k2:
    raise ValueError(
        "Matmul LHS and RHS have incompatible shapes "
        f"{a.shape} vs {b.shape[1:]}"
    )
  collective = config.collective
  tile_m, tile_n, tile_k = (config.tile_m, config.tile_n, config.tile_k)
  block_tile_m = tile_m
  block_tile_n = tile_n
  if collective:
    tile_m *= 2
    tile_n *= 2
  m_iters = m // tile_m
  n_iters = n // tile_n

  max_concurrent_steps = config.max_concurrent_steps
  epilogue_tile_n = config.epilogue_tile_n
  if tile_n % epilogue_tile_n != 0:
    raise ValueError(
        f"{tile_n=} must be divisible by {epilogue_tile_n=}"
    )

  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )

  def kernel(a_gmem, b_gmem, group_sizes_gmem, out_gmem):
    linear_grid = (m_iters + num_groups - 1) * n_iters
    group_sizes_regs = [group_sizes_gmem[i] for i in range(num_groups)]
    cluster_idx = lax.axis_index("x")

    @functools.partial(pl.run_scoped,
        a_smem=plgpu.SMEM(
            (max_concurrent_steps, block_tile_m, tile_k),
            dtype, transforms=transforms
        ),
        b_smem=plgpu.SMEM(
            (max_concurrent_steps, tile_k, block_tile_n),
            dtype, transforms=transforms
        ),
        # Temporary SMEM used for storing accumulator output to GMEM.
        acc_smem=plgpu.SMEM(
            (block_tile_m, epilogue_tile_n), dtype),
        # a/b_tma_barrier
        a_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
        b_tma_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
        # store_done_barrier, double-buffered
        store_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2,
                      orders_tensor_core=True),
        # mma_done_barrier, double-buffered
        mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2,
                      orders_tensor_core=True),
        # consumed_barrier
        consumed_barrier=plgpu.Barrier(
            num_arrivals=1,
            num_barriers=max_concurrent_steps,
            orders_tensor_core=True,
        ),
        # Accumulator TMEM (double-buffered)
        acc_tmem=plgpu.TMEM(
            (block_tile_m, tile_n * 2), jnp.float32, collective=collective),
        collective_axes=("wg",)
    )
    def _scoped(**ref_kwargs):
      @plgpu.nd_loop(grid=(linear_grid,),
                     collective_axes="sm")
      def mn_loop(loop_info: plgpu.NDLoopInfo):  # pylint: disable=unused-variable
        linear_idx, = loop_info.index
        local_index = loop_info.local_index  # type: ignore
        m_index, n_index = plgpu.planar_snake(
          linear_idx,
          (m_iters + num_groups - 1, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
        )
        with jax.named_scope("create_group_info"):
          group_info = ragged_dot_mgpu.GroupInfo.create(
              group_sizes_regs, tile_m, m_index
          )
        do_matmul(
            a_gmem,
            b_gmem.at[group_info.group_id],
            out_gmem,
            grid_indices=(group_info.block, n_index, cluster_idx),
            wg_axis="wg",
            collective_axes=("x",) if collective else (),
            local_index=local_index,  # type: ignore
            config=config,
            group_info=group_info,
            **ref_kwargs
        )

  num_sms = jax.local_devices()[0].core_count
  compiler_params = None
  f = plgpu.kernel(
      kernel,
      compiler_params=compiler_params,
      kernel_name=f"ragged_dot_kernel_{str(config)}",
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms//2,) if collective else (num_sms,),
      grid_names=("sm",),
      num_threads=2,
      thread_name="wg",
      cluster_names=("x",) if collective else (),
      cluster=(2,) if collective else (),
  )
  return f(a, b, group_sizes)


def ragged_dot_reference(a, b, g):
  return lax.ragged_dot(a, b, g, preferred_element_type=jnp.float16)


def sample_group_sizes(key: jax.Array,
                       num_groups: int,
                       num_elements: int,
                       alpha: float = 10.0,
                       ):
  """Sample group sizes.

  Args:
    key: PRNG key.
    num_groups: Number of groups to sample.
    num_elements: Total number of elements to sample.
    alpha: Shape parameter. The lower the alpha, the more imbalanced the
      group sizes will be. As alpha approaches infinity, the group sizes
      approach a uniform distribution.

  Returns:
    A jax.Array of shape (num_groups,) that sums to num_elements.
  """
  probs_key, sample_key = jax.random.split(key)
  probs = jax.random.dirichlet(probs_key, jnp.ones((num_groups,)) * alpha)
  return jax.random.multinomial(
      sample_key, num_elements, probs).astype(jnp.int32)


def main(_) -> None:
  M = 16 * 1024
  K = 2048
  N = 16 * 1024
  num_groups = 16
  group_sizes = sample_group_sizes(jax.random.key(0), num_groups, M, alpha=10.0)

  print(f"==== {M=} {N=} {K=} {num_groups=}====")
  matmul_flops = 2 * M * N * K
  peak_flops = 2.25e15  # f16 TensorCore peak = 2250 TFLOPS
  a = jax.random.uniform(jax.random.key(1), (M, K), jnp.float16)
  b = jax.random.uniform(jax.random.key(2), (num_groups, K, N), jnp.float16)

  tuning_it = itertools.product(
      (128,),  # tile_m
      (128,),  # tile_n
      (64,),  # tile_k
      (1, 8, 12, 16),  # grid_tile_width
      blackwell_matmul_mgpu.MatmulDimension,  # grid_minor_dim
      (4, 6)  # max_concurrent_steps
  )
  best_util = -float("inf")
  for (tile_m, tile_n, tile_k, grid_tile_width, grid_minor_dim,
        max_concurrent_steps,) in tuning_it:
    config = TuningConfig(
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      grid_tile_width=grid_tile_width,
      grid_minor_dim=grid_minor_dim,
      max_concurrent_steps=max_concurrent_steps,
      collective=True,
    )
    try:
      out, runtime_ms = profiler.measure(
          functools.partial(ragged_dot_kernel, config=config),
          iterations=10
      )(a, b, group_sizes)
      runtime_ms = np.median(runtime_ms if runtime_ms else [])  # type: ignore
    except ValueError as e:
      if ("exceeds available shared memory" in e.args[0] or
          "Accumulator layout mismatch:" in e.args[0]):
        print(e.args[0])
        continue
      raise
    expected = ragged_dot_reference(a, b, group_sizes)
    np.testing.assert_allclose(out, expected)

    runtime_us = runtime_ms * 1e3   # type: ignore
    optimal_time = matmul_flops / peak_flops * 1e6  # us
    achieved_tc_util = optimal_time / runtime_us * 100
    if achieved_tc_util > best_util:
      best_util = achieved_tc_util
    print(
        f"{tile_m=} {tile_n=} {tile_k=} {grid_tile_width=} {grid_minor_dim=} {max_concurrent_steps=} "
        f"{runtime_us:<7.1f}us"
        f" = {achieved_tc_util:4.1f}% TC utilization"
    )
  print(f"\tBest utilization: {best_util:4.1f}%")


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
