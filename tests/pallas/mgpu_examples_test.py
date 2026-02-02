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
from jax.extend import backend
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
import jax.experimental.mosaic.gpu  # noqa: F401
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import profiler
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
  epilogue_tile_n: int = 64
  grid_minor_dim: int = 0
  grid_tile_width: int = 1


def matmul0(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
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

  def kernel(a_gmem, b_gmem, out_gmem, acc_tmem, acc_smem, consumed_barriers):
    mi = lax.axis_index("m")
    ni = lax.axis_index("n")
    m_slice = pl.ds(mi * tile_m, tile_m)
    n_slice = pl.ds(ni * tile_n, tile_n)

    def do_mma(idxs, a_smem, b_smem):
      (ki,) = idxs
      arrive_barrier_slot = ki % 2
      wait_barrier_slot = 1 - arrive_barrier_slot
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier=consumed_barriers.at[arrive_barrier_slot],
          accumulate=(ki > 0),
      )
      plgpu.barrier_wait(consumed_barriers.at[wait_barrier_slot])

    # Make sure the wait succeeds in the first iteration.
    plgpu.barrier_arrive(consumed_barriers.at[1])
    block_kwargs = dict(transforms=transforms, delay_release=1)
    plgpu.emit_pipeline(
      do_mma,
      in_specs=[
          plgpu.BlockSpec((tile_m, tile_k), lambda ki: (mi, ki), **block_kwargs),
          plgpu.BlockSpec((tile_k, tile_n), lambda ki: (ki, ni), **block_kwargs),
      ],
      grid=(k_iters,),
      max_concurrent_steps=max_concurrent_steps,
    )(a_gmem, b_gmem)

    final_barrier = 1 - (k_iters % 2)
    plgpu.barrier_wait(consumed_barriers.at[final_barrier])
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      scratch_shapes=dict(
          acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
      ),
  )
  return f(a, b)


def matmul1(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
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

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier):
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    m_slice = pl.ds(m_index * tile_m, tile_m)
    n_slice = pl.ds(n_index * tile_n, tile_n)

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          @pl.when(ki >= max_concurrent_steps)
          def _():  # Make sure the data has been consumed before overwriting.
            plgpu.barrier_wait(consumed_barriers.at[slot])
          k_slice = pl.ds(ki * tile_k, tile_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot]
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot]
          )
        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(warp_id == 1)
      def _compute():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barriers.at[slot],
              accumulate=(ki > 0),
          )
        lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier)

    plgpu.barrier_wait(mma_done_barrier)
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k),
              dtype,
              transforms=transforms,
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n),
              dtype,
              transforms=transforms,
          ),
          acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
          load_barriers=plgpu.Barrier(
              num_arrivals=2, num_barriers=max_concurrent_steps
          ),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=1, orders_tensor_core=True
          ),
      ),
  )
  return f(a, b)


def matmul2(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
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

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier):
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    m_slice = pl.ds(m_index * tile_m, tile_m)
    n_slice = pl.ds(n_index * tile_n, tile_n)

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          @pl.when(ki >= max_concurrent_steps)
          def _():  # Make sure the data has been consumed before overwriting.
            plgpu.barrier_wait(consumed_barriers.at[slot])
          k_slice = pl.ds(ki * tile_k, tile_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot]
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot]
          )
        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(warp_id == 1)
      def _compute():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barriers.at[slot],
              accumulate=(ki > 0),
          )
        lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier)

    plgpu.barrier_wait(mma_done_barrier)
    out_gmem_window = out_gmem.at[m_slice, n_slice]
    for ni in range(tile_n // config.epilogue_tile_n):
      acc_smem_ni = acc_smem.at[ni % 2]
      ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
      # Make sure that previous copy is done before we overwrite.
      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      acc_smem_ni[...] = plgpu.async_load_tmem(acc_tmem.at[:, ni_slice]).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms
          ),
          acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
      )
  )
  return f(a, b)


def matmul3(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier):
    is_lead_block = lax.axis_index("cluster") == 0
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
    n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          @pl.when(ki >= max_concurrent_steps)
          def _():  # Make sure the data has been consumed before overwriting.
            plgpu.barrier_wait(consumed_barriers.at[slot])
          k_slice = pl.ds(ki * tile_k, tile_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot],
              collective_axes="cluster", partitioned_axis=0
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot],
              collective_axes="cluster", partitioned_axis=1
          )
        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
      def _compute():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barriers.at[slot],
              accumulate=(ki > 0),
              collective_axis="cluster",
          )
        lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier, collective_axis="cluster")

    plgpu.barrier_wait(mma_done_barrier)
    out_m_index = m_index * 2 + lax.axis_index("cluster")
    out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
    out_gmem_window = out_gmem.at[out_m_slice, n_slice]
    for ni in range(cluster_tile_n // config.epilogue_tile_n):
      acc_smem_ni = acc_smem.at[ni % 2]
      ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
      # Make sure that previous copy is done before we overwrite.
      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      acc_smem_ni[...] = plgpu.async_load_tmem(acc_tmem.at[:, ni_slice]).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      cluster=(2,),
      cluster_names=("cluster",),
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms,
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms,
          ),
          acc_tmem=plgpu.TMEM((tile_m, cluster_tile_n), jnp.float32, collective=True),
          acc_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
      )
  )
  return f(a, b)


def matmul4(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier):
    is_lead_block = lax.axis_index("cluster") == 0

    @plgpu.nd_loop((m_iters, n_iters), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      m_index, n_index = loop_info.index
      m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
      n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)

      @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
      def _per_warp():
        warp_id = lax.axis_index("warp")

        @pl.when(warp_id == 0)
        def _memory():
          def _loop_body(ki, _):
            slot = lax.rem(ki, max_concurrent_steps)
            @pl.when(jnp.logical_or(ki >= max_concurrent_steps, loop_info.local_index > 0))
            def _():  # Make sure the data has been consumed before overwriting.
              plgpu.barrier_wait(consumed_barriers.at[slot])
            k_slice = pl.ds(ki * tile_k, tile_k)
            plgpu.copy_gmem_to_smem(
                a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot],
                collective_axes="cluster", partitioned_axis=0
            )
            plgpu.copy_gmem_to_smem(
                b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot],
                collective_axes="cluster", partitioned_axis=1
            )

          lax.fori_loop(0, k_iters, _loop_body, None)

        @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
        def _compute():
          def _loop_body(ki, _):
            slot = lax.rem(ki, max_concurrent_steps)
            plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
            plgpu.tcgen05_mma(
                acc_tmem,
                a_smem.at[slot],
                b_smem.at[slot],
                consumed_barriers.at[slot],
                accumulate=(ki > 0),
                collective_axis="cluster",
            )
          lax.fori_loop(0, k_iters, _loop_body, None)
          plgpu.tcgen05_commit_arrive(
              mma_done_barrier,
              collective_axis="cluster",
          )

      plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # Make sure that previous store is done.
      plgpu.barrier_wait(mma_done_barrier)
      out_m_index = m_index * 2 + lax.axis_index("cluster")
      out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
      out_gmem_window = out_gmem.at[out_m_slice, n_slice]
      for ni in range(cluster_tile_n // config.epilogue_tile_n):
        acc_smem_ni = acc_smem.at[ni % 2]
        ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
        # Make sure that previous copy is done before we overwrite.
        plgpu.wait_smem_to_gmem(1, wait_read_only=True)
        acc_smem_ni[...] = plgpu.async_load_tmem(acc_tmem.at[:, ni_slice]).astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
      plgpu.wait_load_tmem()  # Load must complete before MMA can overwrite TMEM.
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  num_sms = backend.get_default_device().core_count
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms // 2,),
      grid_names=("cluster_grid",),
      cluster=(2,),
      cluster_names=("cluster",),
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms,
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms,
          ),
          acc_tmem=plgpu.TMEM((tile_m, cluster_tile_n), jnp.float32, collective=True),
          acc_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
      ),
  )
  return f(a, b)


def matmul5(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier, store_done_barrier):
    wg_idx = lax.axis_index("wg")
    is_lead_block = lax.axis_index("cluster") == 0

    @plgpu.nd_loop((m_iters, n_iters), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      m_index, n_index = loop_info.index
      m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
      n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)
      acc_slot = lax.rem(loop_info.local_index, jnp.int32(2))
      mn_acc_tmem = acc_tmem.at[:, pl.ds(acc_slot * cluster_tile_n, cluster_tile_n)]

      @pl.when(wg_idx == 0)
      def _compute_wg():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")

          @pl.when(warp_id == 0)
          def _memory():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              @pl.when(jnp.logical_or(ki >= max_concurrent_steps, loop_info.local_index > 0))
              def _():  # Make sure the data has been consumed before overwriting.
                plgpu.barrier_wait(consumed_barriers.at[slot])
              k_slice = pl.ds(ki * tile_k, tile_k)
              plgpu.copy_gmem_to_smem(
                  a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", partitioned_axis=0
              )
              plgpu.copy_gmem_to_smem(
                  b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", partitioned_axis=1
              )

            lax.fori_loop(0, k_iters, _loop_body, None)

          # Wait for store to complete (except for the first two steps).
          @pl.when(jnp.logical_and(warp_id == 1, loop_info.local_index >= 2))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])
          @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
          def _compute():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
              plgpu.tcgen05_mma(
                  mn_acc_tmem,
                  a_smem.at[slot],
                  b_smem.at[slot],
                  consumed_barriers.at[slot],
                  accumulate=(ki > 0),
                  collective_axis="cluster",
              )
            lax.fori_loop(0, k_iters, _loop_body, None)
            plgpu.tcgen05_commit_arrive(
                mma_done_barrier.at[acc_slot],
                collective_axis="cluster",
            )

      @pl.when(wg_idx == 1)
      def _store_wg():
        # Ensure that copies from the previous mn step have completed.
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        out_m_index = m_index * 2 + lax.axis_index("cluster")
        out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
        out_gmem_window = out_gmem.at[out_m_slice, n_slice]
        for ni in range(cluster_tile_n // config.epilogue_tile_n):
          acc_smem_ni = acc_smem.at[ni % 2]
          ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
          # Make sure that previous copy is done before we overwrite.
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          acc_smem_ni[...] = plgpu.async_load_tmem(mn_acc_tmem.at[:, ni_slice]).astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
        plgpu.wait_load_tmem()  # Load must complete before we signal.
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  num_sms = backend.get_default_device().core_count
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms // 2,),
      grid_names=("cluster_grid",),
      cluster=(2,),
      cluster_names=("cluster",),
      num_threads=2,
      thread_name="wg",
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms,
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms,
          ),
          acc_tmem=plgpu.TMEM((tile_m, 2 * cluster_tile_n), jnp.float32, collective=True),
          acc_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
          store_done_barrier=plgpu.ClusterBarrier(
              collective_axes=("cluster",),
              num_arrivals=1,
              num_barriers=2,
              orders_tensor_core=True,
          ),
      ),
  )
  return f(a, b)


def matmul6(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier, store_done_barrier):
    wg_idx = lax.axis_index("wg")
    is_lead_block = lax.axis_index("cluster") == 0

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
      n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)
      acc_slot = lax.rem(loop_info.local_index, jnp.int32(2))
      mn_acc_tmem = acc_tmem.at[:, pl.ds(acc_slot * cluster_tile_n, cluster_tile_n)]

      @pl.when(wg_idx == 0)
      def _compute_wg():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")

          @pl.when(warp_id == 0)
          def _memory():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              @pl.when(jnp.logical_or(ki >= max_concurrent_steps, loop_info.local_index > 0))
              def _():  # Make sure the data has been consumed before overwriting.
                plgpu.barrier_wait(consumed_barriers.at[slot])
              k_slice = pl.ds(ki * tile_k, tile_k)
              plgpu.copy_gmem_to_smem(
                  a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", partitioned_axis=0
              )
              plgpu.copy_gmem_to_smem(
                  b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", partitioned_axis=1
              )

            lax.fori_loop(0, k_iters, _loop_body, None)

          # Wait for store to complete (except for the first two steps).
          @pl.when(jnp.logical_and(warp_id == 1, loop_info.local_index >= 2))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])
          @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
          def _compute():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              plgpu.barrier_wait(load_barriers.at[slot])  # Wait for data to arrive.
              plgpu.tcgen05_mma(
                  mn_acc_tmem,
                  a_smem.at[slot],
                  b_smem.at[slot],
                  consumed_barriers.at[slot],
                  accumulate=(ki > 0),
                  collective_axis="cluster",
              )
            lax.fori_loop(0, k_iters, _loop_body, None)
            plgpu.tcgen05_commit_arrive(
                mma_done_barrier.at[acc_slot],
                collective_axis="cluster",
            )

      @pl.when(wg_idx == 1)
      def _store_wg():
        # Ensure that copies from the previous mn step have completed.
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        out_m_index = m_index * 2 + lax.axis_index("cluster")
        out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
        out_gmem_window = out_gmem.at[out_m_slice, n_slice]
        for ni in range(cluster_tile_n // config.epilogue_tile_n):
          acc_smem_ni = acc_smem.at[ni % 2]
          ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
          # Make sure that previous copy is done before we overwrite.
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          acc_smem_ni[...] = plgpu.async_load_tmem(mn_acc_tmem.at[:, ni_slice]).astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
        plgpu.wait_load_tmem()  # Load must complete before we signal.
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  num_sms = backend.get_default_device().core_count
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms // 2,),
      grid_names=("cluster_grid",),
      cluster=(2,),
      cluster_names=("cluster",),
      num_threads=2,
      thread_name="wg",
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms
          ),
          acc_tmem=plgpu.TMEM((tile_m, 2 * cluster_tile_n), jnp.float32, collective=True),
          acc_smem=plgpu.SMEM((2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms),
          load_barriers=plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(num_arrivals=1, num_barriers=2, orders_tensor_core=True),
          store_done_barrier=plgpu.ClusterBarrier(
              collective_axes=("cluster",),
              num_arrivals=1,
              num_barriers=2,
              orders_tensor_core=True,
          ),
      )
  )
  return f(a, b)


@jtu.with_config(jax_traceback_filtering="off")
class MatmulTutorialTCGen05Test(jtu.JaxTestCase, jtu.CudaArchSpecificTest):
  BENCHMARK = False

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Test requires an NVIDIA GPU")
    self.skip_unless_tcgen05()
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

  def benchmark(self, matmul_impl, a, b, config_search_space):
    if not self.BENCHMARK:
      return
    config_names = config_search_space.keys()
    config_all_values = config_search_space.values()
    peak_flops = 2250e12  # f16 TensorCore peak = 2250 TFLOPS
    matmul_flops = 2 * a.shape[0] * b.shape[0] * b.shape[1]
    optimal_time_us = matmul_flops / peak_flops * 1e6  # us
    best_util = 0.0
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
      runtime_us = runtime_ms * 1e3   # type: ignore
      achieved_tc_util = optimal_time_us / runtime_us * 100
      print(f"{config} {achieved_tc_util:.2f}% TC utilization")
      if achieved_tc_util > best_util:
        best_util = achieved_tc_util
        np.testing.assert_allclose(out, ref)
    print(f"Best result for {matmul_impl.__name__}: {best_util:.2f}% TC utilization")
    _, runtimes_ms = profiler.measure(
        functools.partial(
            jnp.dot, precision=jax.lax.DotAlgorithmPreset.F16_F16_F32
        ),
        iterations=100,
    )(a, b)
    runtime_ms = statistics.median(runtimes_ms)
    runtime_us = runtime_ms * 1e3   # type: ignore
    achieved_tc_util = optimal_time_us / runtime_us * 100
    print(f"Reference: {achieved_tc_util:.2f}% TC utilization")

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
        tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4,
    )
    config_search_space = {
        "tile_m": (128,),
        "tile_n": (128, 256, 512),
        "tile_k": (64,),
        "max_concurrent_steps": (4, 6),
    }
    self._test_matmul(matmul_impl, example_config, config_search_space)

  def test_matmul3(self):
    example_config = TuningConfig(
        tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4,
    )
    config_search_space = {
        "tile_m": (128,),
        "tile_n": (128,),
        "tile_k": (64,),
        "max_concurrent_steps": (6,),
    }
    self._test_matmul(matmul3, example_config, config_search_space)

  def test_matmul4(self):
    example_config = TuningConfig(
        tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4,
    )
    config_search_space = {
        "tile_m": (128,),
        "tile_n": (128,),
        "tile_k": (64,),
        "max_concurrent_steps": (6,),
    }
    self._test_matmul(matmul4, example_config, config_search_space)

  def test_matmul5(self):
    example_config = TuningConfig(
        tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4,
    )
    config_search_space = {
        "tile_m": (128,),
        "tile_n": (128,),
        "tile_k": (64,),
        "max_concurrent_steps": (6,),
    }
    self._test_matmul(matmul5, example_config, config_search_space)

  def test_matmul6(self):
    example_config = TuningConfig(
        tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4,
        grid_minor_dim=0, grid_tile_width=6,
    )
    config_search_space = {
        "tile_m": (128,),
        "tile_n": (128,),
        "tile_k": (64,),
        "max_concurrent_steps": (6,),
        "grid_minor_dim": (0, 1),
        "grid_tile_width": (1, 4, 12, 16),
    }
    self._test_matmul(matmul6, example_config, config_search_space)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
