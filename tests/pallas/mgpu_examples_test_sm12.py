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
"""Tests for examples from Pallas:MGPU documentation."""

import dataclasses
import functools
import itertools
import statistics

from absl.testing import absltest
from absl.testing import parameterized
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
import jax.experimental.mosaic.gpu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.extend import backend
import jax
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
  epilogue_tile_n: int | None = None
  grid_minor_dim: int = 0
  grid_tile_width: int = 1
  gs_num_sms_factor: int = 4


def matmul0(a, b, config: TuningConfig):
  # baseline: emit_pipeline

  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

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

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=('m', 'n'),
      scratch_types=[
          plgpu.SMEM((tile_m, tile_n), dtype),  # output
      ],
  )
  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      out_smem,
  ):
    mi = lax.axis_index("m")
    ni = lax.axis_index("n")
    m_slice = pl.ds(mi * tile_m, tile_m)
    n_slice = pl.ds(ni * tile_n, tile_n)

    acc = plgpu.layout_cast(
        jnp.zeros((tile_m, tile_n), jnp.float32), plgpu.Layout.MMA_ACC(dtype)
    )

    def do_mma(_, a_smem, b_smem, carry):
      with jax.named_scope("load"):
        a = plgpu.load(
            a_smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=True
        )
        b = plgpu.load(
            b_smem, layout=plgpu.Layout.MMA_RHS(dtype), optimized=True
        )
      with jax.named_scope("mma"):
        return plgpu.mma(carry, a, b)

    acc = plgpu.emit_pipeline(
      do_mma,
      grid=(k_iters,),
      in_specs=[
          plgpu.BlockSpec(
              (tile_m, tile_k), lambda ki: (mi, ki), delay_release=1
          ),
          plgpu.BlockSpec(
              (tile_k, tile_n), lambda ki: (ki, ni), delay_release=1
          ),
      ],
      max_concurrent_steps=max_concurrent_steps,
      init_carry=acc,
    )(a_gmem, b_gmem)

    with jax.named_scope("epilogue"):
      out_smem[...] = acc.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[m_slice, n_slice])
      plgpu.wait_smem_to_gmem(0)  # Wait for all copies to finish.

  return kernel(a, b)


def matmul1(a, b, config: TuningConfig):
  # use persistent kernel
  num_sms = backend.get_default_device().core_count

  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

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

  grid_size = min(num_sms * 4, m_iters * n_iters)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(grid_size,),
      grid_names=("sm",),
      scratch_types=dict(
          a_smem=plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), dtype),
          b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), dtype),
          out_smem=plgpu.SMEM((tile_m, tile_n), dtype),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),  # gmem to smem copy barriers
      ),
  )
  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      out_smem,
      load_barriers,
  ):
    @plgpu.nd_loop((m_iters, n_iters), collective_axes="sm")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      mi, ni = loop_info.index
      m_slice = pl.ds(mi * tile_m, tile_m)
      n_slice = pl.ds(ni * tile_n, tile_n)

      acc = plgpu.layout_cast(
          jnp.zeros((tile_m, tile_n), jnp.float32), plgpu.Layout.MMA_ACC(dtype)
      )

      def _data_copy(ki, slot):
        k_slice = pl.ds(ki * tile_k, tile_k)
        plgpu.copy_gmem_to_smem(
            a_gmem.at[m_slice, k_slice],
            a_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )
        plgpu.copy_gmem_to_smem(
            b_gmem.at[k_slice, n_slice],
            b_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )

      def _compute(slot, carry):
        plgpu.barrier_wait(load_barriers.at[slot])
        with jax.named_scope("compute:load"):
          a = plgpu.load(
              a_smem.at[slot], layout=plgpu.Layout.MMA_LHS(dtype), optimized=True
          )
          b = plgpu.load(
              b_smem.at[slot], layout=plgpu.Layout.MMA_RHS(dtype), optimized=True
          )
        with jax.named_scope("compute:mma"):
          output = plgpu.mma(carry, a, b)
        return output

      prefetch = max_concurrent_steps - 1
      for j in range(min(prefetch, k_iters)):
        _data_copy(j, j % max_concurrent_steps)

      def body(ki, acc):
        _data_copy(ki + prefetch, (ki + prefetch) % max_concurrent_steps)
        return _compute(ki % max_concurrent_steps, acc)

      steady_end = max(0, k_iters - prefetch)
      acc = lax.fori_loop(0, steady_end, body, acc)

      for ki in range(steady_end, k_iters):
        acc = _compute(ki % max_concurrent_steps, acc)

      with jax.named_scope("epilogue"):
        out_smem[...] = acc.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[m_slice, n_slice])
        plgpu.wait_smem_to_gmem(0)  # Wait for all copies to finish.

  return kernel(a, b)


def matmul2(a, b, config: TuningConfig):
  # use persistent kernel and tiled epilogue
  num_sms = backend.get_default_device().core_count

  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

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

  grid_size = min(num_sms * 4, m_iters * n_iters)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(grid_size,),
      grid_names=("sm",),
      scratch_types=dict(
          a_smem=plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), dtype),
          b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), dtype),
          out_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),  # gmem to smem copy barriers
      ),
  )
  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      out_smem,
      load_barriers,
  ):
    @plgpu.nd_loop((m_iters, n_iters), collective_axes="sm")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      mi, ni = loop_info.index
      m_slice = pl.ds(mi * tile_m, tile_m)
      n_slice = pl.ds(ni * tile_n, tile_n)

      acc = plgpu.layout_cast(
          jnp.zeros((tile_m, tile_n), jnp.float32), plgpu.Layout.MMA_ACC(dtype)
      )

      def _data_copy(ki, slot):
        k_slice = pl.ds(ki * tile_k, tile_k)
        plgpu.copy_gmem_to_smem(
            a_gmem.at[m_slice, k_slice],
            a_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )
        plgpu.copy_gmem_to_smem(
            b_gmem.at[k_slice, n_slice],
            b_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )

      def _compute(slot, carry):
        plgpu.barrier_wait(load_barriers.at[slot])
        with jax.named_scope("compute:load"):
          a = plgpu.load(
              a_smem.at[slot], layout=plgpu.Layout.MMA_LHS(dtype), optimized=True
          )
          b = plgpu.load(
              b_smem.at[slot], layout=plgpu.Layout.MMA_RHS(dtype), optimized=True
          )
        with jax.named_scope("compute:mma"):
          output = plgpu.mma(carry, a, b)
        return output

      prefetch = max_concurrent_steps - 1
      for j in range(min(prefetch, k_iters)):
        _data_copy(j, j % max_concurrent_steps)

      def body(ki, acc):
        _data_copy(ki + prefetch, (ki + prefetch) % max_concurrent_steps)
        return _compute(ki % max_concurrent_steps, acc)

      steady_end = max(0, k_iters - prefetch)
      acc = lax.fori_loop(0, steady_end, body, acc)

      for ki in range(steady_end, k_iters):
        acc = _compute(ki % max_concurrent_steps, acc)

      with jax.named_scope("epilogue"):
        out_gmem_window = out_gmem.at[m_slice, n_slice]
        for ni in range(tile_n // config.epilogue_tile_n):
          ni_slice = slice(
              ni * config.epilogue_tile_n,
              (ni + 1) * config.epilogue_tile_n
          )
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          out_smem_ni = out_smem.at[ni % 2]
          out_smem_ni[...] = acc[:, ni_slice].astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(out_smem_ni, out_gmem_window.at[:, ni_slice])
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  return kernel(a, b)


def matmul3(a, b, config: TuningConfig):
  # use persistent kernel and tiled epilogue and grid tiling
  num_sms = backend.get_default_device().core_count

  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

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

  grid_size = min(num_sms * 4, m_iters * n_iters)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(grid_size,),
      grid_names=("sm",),
      scratch_types=dict(
          a_smem=plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), dtype),
          b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), dtype),
          out_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),  # gmem to smem copy barriers
      ),
  )
  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      out_smem,
      load_barriers,
  ):
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          lin_idx,  # Linear index.
          (m_iters, n_iters),  # The 2D iteration space.
          config.grid_minor_dim,  # 0 or 1, indicates the fastest changing dim.
          config.grid_tile_width,  # The width of tiles along the fastest changing dim.
      )
      m_slice = pl.ds(mi * tile_m, tile_m)
      n_slice = pl.ds(ni * tile_n, tile_n)

      acc = plgpu.layout_cast(
          jnp.zeros((tile_m, tile_n), jnp.float32), plgpu.Layout.MMA_ACC(dtype)
      )

      def _data_copy(ki, slot):
        k_slice = pl.ds(ki * tile_k, tile_k)
        plgpu.copy_gmem_to_smem(
            a_gmem.at[m_slice, k_slice],
            a_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )
        plgpu.copy_gmem_to_smem(
            b_gmem.at[k_slice, n_slice],
            b_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )

      def _compute(slot, carry):
        plgpu.barrier_wait(load_barriers.at[slot])
        with jax.named_scope("compute:load"):
          a = plgpu.load(
              a_smem.at[slot], layout=plgpu.Layout.MMA_LHS(dtype), optimized=True
          )
          b = plgpu.load(
              b_smem.at[slot], layout=plgpu.Layout.MMA_RHS(dtype), optimized=True
          )
        with jax.named_scope("compute:mma"):
          output = plgpu.mma(carry, a, b)
        return output

      prefetch = max_concurrent_steps - 1
      for j in range(min(prefetch, k_iters)):
        _data_copy(j, j % max_concurrent_steps)

      def body(ki, acc):
        _data_copy(ki + prefetch, (ki + prefetch) % max_concurrent_steps)
        return _compute(ki % max_concurrent_steps, acc)

      steady_end = max(0, k_iters - prefetch)
      acc = lax.fori_loop(0, steady_end, body, acc)

      for ki in range(steady_end, k_iters):
        acc = _compute(ki % max_concurrent_steps, acc)

      with jax.named_scope("epilogue"):
        out_gmem_window = out_gmem.at[m_slice, n_slice]
        for ni in range(tile_n // config.epilogue_tile_n):
          ni_slice = slice(
              ni * config.epilogue_tile_n,
              (ni + 1) * config.epilogue_tile_n
          )
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          out_smem_ni = out_smem.at[ni % 2]
          out_smem_ni[...] = acc[:, ni_slice].astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(out_smem_ni, out_gmem_window.at[:, ni_slice])
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  return kernel(a, b)


def matmul4(a, b, config: TuningConfig):
  # use persistent kernel and tiled epilogue and grid tiling
  # + reduce registers spill
  num_sms = backend.get_default_device().core_count

  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

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

  epilogue_n_iters = tile_n // config.epilogue_tile_n

  grid_size = min(num_sms * config.gs_num_sms_factor, m_iters * n_iters)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(grid_size,),
      grid_names=("sm",),
      scratch_types=dict(
          a_smem=plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), dtype),
          b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), dtype),
          out_smem=plgpu.SMEM((tile_m, config.epilogue_tile_n), dtype),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),  # gmem to smem copy barriers
      ),
  )
  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      out_smem,
      load_barriers,
  ):
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          lin_idx,  # Linear index.
          (m_iters, n_iters),  # The 2D iteration space.
          config.grid_minor_dim,  # 0 or 1, indicates the fastest changing dim.
          config.grid_tile_width,  # The width of tiles along the fastest changing dim.
      )
      m_slice = pl.ds(mi * tile_m, tile_m)
      n_slice = pl.ds(ni * tile_n, tile_n)

      def _data_copy(ki, slot):
        k_slice = pl.ds(ki * tile_k, tile_k)
        plgpu.copy_gmem_to_smem(
            a_gmem.at[m_slice, k_slice],
            a_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )
        plgpu.copy_gmem_to_smem(
            b_gmem.at[k_slice, n_slice],
            b_smem.at[slot],
            load_barriers.at[slot],
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )

      def _compute(slot, carry):
        plgpu.barrier_wait(load_barriers.at[slot])
        with jax.named_scope("compute:load+mma"):
          a = plgpu.load(
              a_smem.at[slot], layout=plgpu.Layout.MMA_LHS(dtype), optimized=True
          )
          for ni in range(epilogue_n_iters):
            ni_slice = slice(
                ni * config.epilogue_tile_n,
                (ni + 1) * config.epilogue_tile_n
            )
            b = plgpu.load(
                b_smem.at[slot, :, ni_slice], layout=plgpu.Layout.MMA_RHS(dtype), optimized=True
            )
            carry[ni] = plgpu.mma(carry[ni], a, b)
        return carry

      prefetch = max_concurrent_steps - 1
      for j in range(min(prefetch, k_iters)):
        _data_copy(j, j % max_concurrent_steps)

      def body(ki, acc):
        _data_copy(ki + prefetch, (ki + prefetch) % max_concurrent_steps)
        return _compute(ki % max_concurrent_steps, acc)

      acc = [
        plgpu.layout_cast(
            jnp.zeros((tile_m, config.epilogue_tile_n), jnp.float32), plgpu.Layout.MMA_ACC(dtype)
        )
        for _ in range(epilogue_n_iters)
      ]
      steady_end = max(0, k_iters - prefetch)
      acc = lax.fori_loop(0, steady_end, body, acc)

      for ki in range(steady_end, k_iters):
        acc = _compute(ki % max_concurrent_steps, acc)

      with jax.named_scope("epilogue"):
        out_gmem_window = out_gmem.at[m_slice, n_slice]
        for ni in range(tile_n // config.epilogue_tile_n):
          ni_slice = slice(
              ni * config.epilogue_tile_n,
              (ni + 1) * config.epilogue_tile_n
          )
          plgpu.wait_smem_to_gmem(0, wait_read_only=True)
          out_smem[...] = acc[ni].astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(out_smem, out_gmem_window.at[:, ni_slice])
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  return kernel(a, b)


def matmul5(a, b, config: TuningConfig):
  # use persistent kernel, grid tiling
  # and warpgroup specialization
  num_sms = backend.get_default_device().core_count

  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

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

  grid_size = min(num_sms * config.gs_num_sms_factor, m_iters * n_iters)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(grid_size,),
      grid_names=("sm",),
      scratch_types=dict(
          a_smem=plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), dtype),
          b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), dtype),
          out_smem=plgpu.SMEM((tile_m, tile_n), dtype),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),  # gmem to smem copy barriers
          consumed_barriers=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
      ),
      num_threads=2, # Two warpgroups (256 threads) per block
      thread_name="wg",
  )
  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      out_smem,
      load_barriers,
      consumed_barriers,
  ):
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          lin_idx,  # Linear index.
          (m_iters, n_iters),  # The 2D iteration space.
          config.grid_minor_dim,  # 0 or 1, indicates the fastest changing dim.
          config.grid_tile_width,  # The width of tiles along the fastest changing dim.
      )
      m_slice = pl.ds(mi * tile_m, tile_m)
      n_slice = pl.ds(ni * tile_n, tile_n)

      wg_idx = lax.axis_index("wg")

      # Warpgroup 0: Dedicated Memory Prefetcher
      @pl.when(wg_idx == 0)
      def _memory_wg():
        def _loop_body_mem(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)

          # Wait for the Compute Warpgroup to complete its read cycle before overwriting
          @pl.when(ki >= max_concurrent_steps)
          def _await_consumed():
            plgpu.barrier_wait(consumed_barriers.at[slot])

          @pl.when(ki < k_iters)
          def _produce():
            k_slice = pl.ds(ki * tile_k, tile_k)
            plgpu.copy_gmem_to_smem(
                a_gmem.at[m_slice, k_slice],
                a_smem.at[slot],
                load_barriers.at[slot],
                oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
            )
            plgpu.copy_gmem_to_smem(
                b_gmem.at[k_slice, n_slice],
                b_smem.at[slot],
                load_barriers.at[slot],
                oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
            )

        lax.fori_loop(0, k_iters + max_concurrent_steps, _loop_body_mem, None)

      # Warpgroup 1: Dedicated Compute Engine
      @pl.when(wg_idx == 1)
      def _compute_wg():
        def _loop_body_comp(ki, acc):
          slot = lax.rem(ki, max_concurrent_steps)

          # Wait for the Memory Warpgroup to populate the SMEM buffers
          plgpu.barrier_wait(load_barriers.at[slot])

          with jax.named_scope("compute:load"):
            a = plgpu.load(
                a_smem.at[slot], layout=plgpu.Layout.MMA_LHS(dtype), optimized=True
            )
            b = plgpu.load(
                b_smem.at[slot], layout=plgpu.Layout.MMA_RHS(dtype), optimized=True
            )
          with jax.named_scope("compute:mma"):
            acc = plgpu.mma(acc, a, b)

          # Signal to Warpgroup 0 that the SMEM read is complete
          plgpu.barrier_arrive(consumed_barriers.at[slot])
          return acc

        # Initialize local register tiles
        acc = plgpu.layout_cast(
            jnp.zeros((tile_m, tile_n), jnp.float32), plgpu.Layout.MMA_ACC(dtype)
        )

        acc = lax.fori_loop(0, k_iters, _loop_body_comp, acc)

        # Write out epilogue from Warpgroup 1 (where registers are active)
        with jax.named_scope("epilogue"):
          out_smem[...] = acc.astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[m_slice, n_slice])
          plgpu.wait_smem_to_gmem(0)  # Wait for all copies to finish.

  return kernel(a, b)


def matmul6(a, b, config: TuningConfig):
  # use persistent kernel, tiled epilogue, grid tiling
  # and warpgroup specialization
  num_sms = backend.get_default_device().core_count

  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k

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

  grid_size = min(num_sms * config.gs_num_sms_factor, m_iters * n_iters)

  @plgpu.kernel(
      out_type=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(grid_size,),
      grid_names=("sm",),
      scratch_types=dict(
          a_smem=plgpu.SMEM((max_concurrent_steps, tile_m, tile_k), dtype),
          b_smem=plgpu.SMEM((max_concurrent_steps, tile_k, tile_n), dtype),
          out_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),  # gmem to smem copy barriers
          consumed_barriers=plgpu.Barrier(num_arrivals=1, num_barriers=max_concurrent_steps),
      ),
      num_threads=2, # Two warpgroups (256 threads) per block
      thread_name="wg",
  )
  def kernel(
      a_gmem,
      b_gmem,
      out_gmem,
      a_smem,
      b_smem,
      out_smem,
      load_barriers,
      consumed_barriers,
  ):
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          lin_idx,  # Linear index.
          (m_iters, n_iters),  # The 2D iteration space.
          config.grid_minor_dim,  # 0 or 1, indicates the fastest changing dim.
          config.grid_tile_width,  # The width of tiles along the fastest changing dim.
      )
      m_slice = pl.ds(mi * tile_m, tile_m)
      n_slice = pl.ds(ni * tile_n, tile_n)

      wg_idx = lax.axis_index("wg")

      # Warpgroup 0: Dedicated Memory Prefetcher
      @pl.when(wg_idx == 0)
      def _memory_wg():
        def _loop_body_mem(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)

          # Wait for the Compute Warpgroup to complete its read cycle before overwriting
          @pl.when(ki >= max_concurrent_steps)
          def _await_consumed():
            plgpu.barrier_wait(consumed_barriers.at[slot])

          @pl.when(ki < k_iters)
          def _produce():
            k_slice = pl.ds(ki * tile_k, tile_k)
            plgpu.copy_gmem_to_smem(
                a_gmem.at[m_slice, k_slice],
                a_smem.at[slot],
                load_barriers.at[slot],
                oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
            )
            plgpu.copy_gmem_to_smem(
                b_gmem.at[k_slice, n_slice],
                b_smem.at[slot],
                load_barriers.at[slot],
                oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
            )

        lax.fori_loop(0, k_iters + max_concurrent_steps, _loop_body_mem, None)

      # Warpgroup 1: Dedicated Compute Engine
      @pl.when(wg_idx == 1)
      def _compute_wg():
        def _loop_body_comp(ki, acc):
          slot = lax.rem(ki, max_concurrent_steps)

          # Wait for the Memory Warpgroup to populate the SMEM buffers
          plgpu.barrier_wait(load_barriers.at[slot])

          with jax.named_scope("compute:load"):
            a = plgpu.load(
                a_smem.at[slot], layout=plgpu.Layout.MMA_LHS(dtype), optimized=True
            )
            b = plgpu.load(
                b_smem.at[slot], layout=plgpu.Layout.MMA_RHS(dtype), optimized=True
            )
          with jax.named_scope("compute:mma"):
            acc = plgpu.mma(acc, a, b)

          # Signal to Warpgroup 0 that the SMEM read is complete
          plgpu.barrier_arrive(consumed_barriers.at[slot])
          return acc

        # Initialize local register tiles
        acc = plgpu.layout_cast(
            jnp.zeros((tile_m, tile_n), jnp.float32), plgpu.Layout.MMA_ACC(dtype)
        )

        acc = lax.fori_loop(0, k_iters, _loop_body_comp, acc)

        # Write out epilogue from Warpgroup 1 (where registers are active)
        with jax.named_scope("epilogue"):
          out_gmem_window = out_gmem.at[m_slice, n_slice]
          for ni in range(tile_n // config.epilogue_tile_n):
            ni_slice = slice(
                ni * config.epilogue_tile_n,
                (ni + 1) * config.epilogue_tile_n
            )
            out_smem_ni = out_smem.at[ni % 2]
            out_smem_ni[...] = acc[:, ni_slice].astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(out_smem_ni, out_gmem_window.at[:, ni_slice])
            plgpu.wait_smem_to_gmem(1, wait_read_only=True)

  return kernel(a, b)


@jtu.with_config(jax_traceback_filtering="off")
class MatmulTutorialSM12XTest(jtu.JaxTestCase, jtu.CudaArchSpecificTest):
  BENCHMARK = False

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["rocm"]):
      self.skipTest("Mosaic GPU is not supported on ROCm.")
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Test requires an NVIDIA GPU")
    self.skip_unless_sm120_sm121()

  def benchmark(self, matmul_impl, a, b, config_search_space):
    if not self.BENCHMARK:
      return
    config_names = config_search_space.keys()
    config_all_values = config_search_space.values()

    peak_flops_dict = {
      "NVIDIA GB10": 120e12,
      "NVIDIA RTX PRO 6000 Blackwell": 250e12,
      "NVIDIA RTX PRO 5000 Blackwell": 267e12,
      "NVIDIA RTX PRO 4500 Blackwell": 200e12,
    }
    device_kind = jax.devices()[0].device_kind
    peak_flops = peak_flops_dict.get(device_kind, 200e12)
    matmul_flops = 2 * a.shape[0] * b.shape[0] * b.shape[1]
    optimal_time_us = matmul_flops / peak_flops * 1e6  # us
    best_util = 0.0
    best_runtime_us = None
    ref = jnp.dot(a, b, precision=jax.lax.DotAlgorithmPreset.F16_F16_F32)
    for config_values in itertools.product(*config_all_values):
      config = TuningConfig(**dict(zip(config_names, config_values)))
      try:
        out, runtimes_ms = profiler.measure(
            functools.partial(matmul_impl, config=config), iterations=100
        )(a, b)
      except ValueError as e:
        if "exceeds available shared memory" in e.args[0]:  # Ignore SMEM OOMs.
          continue
        raise
      assert runtimes_ms is not None
      runtime_ms = statistics.median(runtimes_ms)
      runtime_us = runtime_ms * 1e3
      achieved_tc_util = optimal_time_us / runtime_us * 100
      print(f"{config} {achieved_tc_util:.2f}% TC utilization")
      np.testing.assert_allclose(out, ref)
      if achieved_tc_util > best_util:
        best_util = achieved_tc_util
        best_runtime_us = runtime_us
    print(
        f"Best result for {matmul_impl.__name__}: {best_util:.2f}% TC utilization\n"
        f"- best runtime us: {best_runtime_us}\n"
    )
    _, runtimes_ms = profiler.measure(
        functools.partial(
            jnp.dot, precision=jax.lax.DotAlgorithmPreset.F16_F16_F32
        ),
        iterations=100,
    )(a, b)
    runtime_ms = statistics.median(runtimes_ms)
    runtime_us = runtime_ms * 1e3
    achieved_tc_util = optimal_time_us / runtime_us * 100
    print(
        f"Reference: {achieved_tc_util:.2f}% TC utilization\n"
        f"- runtime us: {runtime_us}"
    )

  def _test_matmul(self, matmul_impl, example_config, config_search_space):
    dtype = jnp.float16
    m = 4096
    n = 8192
    k = 4096
    k1, k2, = jax.random.split(jax.random.key(42), 2)
    a = jax.random.normal(k1, (m, k), dtype)
    b = jax.random.normal(k2, (k, n), dtype)

    out = matmul_impl(a, b, example_config)
    out_ref = jnp.dot(a, b, precision=jax.lax.DotAlgorithmPreset.F16_F16_F32)
    np.testing.assert_allclose(out, out_ref)
    self.benchmark(matmul_impl, a, b, config_search_space)

  @parameterized.parameters(matmul0, matmul1, matmul2)
  def test_matmul(self, matmul_impl):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=64,
        tile_k=64,
        max_concurrent_steps=2,
        epilogue_tile_n=32,
    )
    max_concurrent_steps = (2, 4, 6)
    if matmul_impl in (matmul1, matmul2):
      max_concurrent_steps = (1,) + max_concurrent_steps

    config_search_space = {
        "tile_m": (128,),
        "tile_n": (64, 128,),
        "tile_k": (32, 64),
        "epilogue_tile_n": (32, 64),
        "max_concurrent_steps": max_concurrent_steps,
    }
    self._test_matmul(matmul_impl, example_config, config_search_space)

  def test_matmul3(self):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        max_concurrent_steps=2,
        epilogue_tile_n=32,
        grid_minor_dim=0,
        grid_tile_width=8,
    )
    config_search_space = {
        "tile_m": (128,),
        "tile_n": (128,),
        "tile_k": (32, 64),
        "epilogue_tile_n": (32, 64, 128),
        "max_concurrent_steps": (2, 3, 4),
        "grid_minor_dim": (0, 1),
        "grid_tile_width": (4, 6, 8, 12, 16),
    }
    self._test_matmul(matmul3, example_config, config_search_space)

  def test_matmul4(self):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        max_concurrent_steps=2,
        epilogue_tile_n=32,
        grid_minor_dim=0,
        grid_tile_width=8,
        gs_num_sms_factor=2,
    )
    config_search_space = {
        "tile_m": (128,),
        "tile_n": (128,),
        "tile_k": (32, 64),
        "epilogue_tile_n": (32, 64),
        "max_concurrent_steps": (2, 3, 4),
        "grid_minor_dim": (0, 1),
        "grid_tile_width": (4, 6, 8, 12, 16),
        "gs_num_sms_factor": (4, 2, 1),
    }
    self._test_matmul(matmul4, example_config, config_search_space)

  def test_matmul5(self):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        max_concurrent_steps=2,
        grid_minor_dim=0,
        grid_tile_width=8,
        gs_num_sms_factor=1,
    )
    config_search_space = {
        "tile_m": (128, 256),
        "tile_n": (128,),
        "tile_k": (32, 64),
        "max_concurrent_steps": (2, 3, 4),
        "grid_minor_dim": (0, 1),
        "grid_tile_width": (4, 6, 8, 12, 16),
        "gs_num_sms_factor": (4, 2, 1),
    }
    self._test_matmul(matmul5, example_config, config_search_space)

  def test_matmul6(self):
    example_config = TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        epilogue_tile_n=32,
        max_concurrent_steps=2,
        grid_minor_dim=0,
        grid_tile_width=8,
        gs_num_sms_factor=1,
    )
    config_search_space = {
        "tile_m": (128, 256),
        "tile_n": (128,),
        "tile_k": (64, 128),
        "epilogue_tile_n": (32, 64),
        "max_concurrent_steps": (2, 3, 4),
        "grid_minor_dim": (0, 1),
        "grid_tile_width": (4, 6, 8, 12, 16),
        "gs_num_sms_factor": (4, 2, 1),
    }
    self._test_matmul(matmul6, example_config, config_search_space)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
