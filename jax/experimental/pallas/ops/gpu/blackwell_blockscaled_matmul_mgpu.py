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
from jax._src import dtypes
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


class MatmulDimension(enum.IntEnum):
  M = 0
  N = 1


def matmul_kernel(a, b, a_scale, b_scale, config):
  m, k = a.shape
  n, k2 = b.shape
  if k != k2:
    raise ValueError(
        f"Matmul LHS and RHS have incompatible shapes {a.shape} vs {b.shape}"
    )
  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  max_concurrent_steps = config.max_concurrent_steps
  collective = config.collective
  collect_profile = config.collect_profile

  cluster_block_m = block_m * 2 if collective else block_m
  cluster_block_n = block_n * 2 if collective else block_n
  m_iters = m // cluster_block_m
  n_iters = n // cluster_block_n
  k_iters = k // block_k
  dtype = jnp.float4_e2m1fn

  TMA_WARP = 1
  MMA_WARP = 0
  COMPUTE_WG = 0
  STORE_WG = 1

  def kernel(
      a_gmem,
      b_gmem,
      a_scale_gmem,
      b_scale_gmem,
      out_gmem,
      a_smem,
      b_smem,
      a_scale_smem,
      b_scale_smem,
      acc_smem,
      a_scale_tmem,
      b_scale_tmem,
      acc_tmem,
      tma_barrier,
      consumed_barrier,
      mma_done_barrier,
      store_done_barrier,
  ):
    wg_idx = lax.axis_index("wg")
    cluster_idx = lax.axis_index("x")
    is_lead_block = cluster_idx == 0

    @plgpu.nd_loop(grid=(m_iters * n_iters,), collective_axes="sm")
    def mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      local_index = loop_info.local_index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      block_m_index = m_index * 2 + cluster_idx if collective else m_index
      block_slice_m = pl.ds(block_m_index * block_m, block_m)
      cluster_slice_m = pl.ds(m_index * cluster_block_m, cluster_block_m)
      cluster_slice_n = pl.ds(n_index * cluster_block_n, cluster_block_n)
      cluster_scale_block_m = cluster_block_m // 128
      cluster_scale_block_n = (
          cluster_block_n // 128
      )  # B scales are replicated with 2 CTA MMA
      cluster_scale_slice_m = pl.ds(
          m_index * cluster_scale_block_m, cluster_scale_block_m
      )
      cluster_scale_slice_n = pl.ds(
          n_index * cluster_scale_block_n, cluster_scale_block_n
      )

      @pl.when(wg_idx == COMPUTE_WG)
      def _():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")

          @pl.when(warp_id == TMA_WARP)
          def _memory():
            def _loop_body(ki, _):
              slice_k = pl.ds(ki * block_k, block_k)
              scale_block_k = block_k // block_size // 4
              scale_slice_k = pl.ds(ki * scale_block_k, scale_block_k)
              slot = lax.rem(ki, max_concurrent_steps)

              @pl.when(
                  jnp.logical_or(ki >= max_concurrent_steps, local_index > 0)
              )
              def _():
                plgpu.barrier_wait(consumed_barrier.at[slot])

              plgpu.copy_gmem_to_smem(
                  a_gmem.at[cluster_slice_m, slice_k],
                  a_smem.at[slot],
                  tma_barrier.at[slot],
                  partitioned_axis=0 if collective else None,
                  collective_axes="x" if collective else None,
              )
              plgpu.copy_gmem_to_smem(
                  b_gmem.at[cluster_slice_n, slice_k],
                  b_smem.at[slot],
                  tma_barrier.at[slot],
                  partitioned_axis=0 if collective else None,
                  collective_axes="x" if collective else None,
              )
              plgpu.copy_gmem_to_smem(
                  a_scale_gmem.at[cluster_scale_slice_m, scale_slice_k],
                  a_scale_smem.at[slot],
                  tma_barrier.at[slot],
                  partitioned_axis=0 if collective else None,
                  collective_axes="x" if collective else None,
              )
              # B scales are replicated! Note that this does not use 2CTA TMA
              # and will need to be awaited in the non-leader CTA or else we
              # will double arrive.
              plgpu.copy_gmem_to_smem(
                  b_scale_gmem.at[cluster_scale_slice_n, scale_slice_k],
                  b_scale_smem.at[slot],
                  tma_barrier.at[slot],
                  collective_axes="x" if collective else None,
              )

            lax.fori_loop(0, k_iters, _loop_body, None)

          @pl.when(jnp.logical_and(warp_id == MMA_WARP, local_index > 1))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier)

          @pl.when(jnp.logical_and(warp_id == MMA_WARP, is_lead_block))
          def _compute():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              with jax.named_scope("wait_tma"):
                plgpu.barrier_wait(tma_barrier.at[slot])
              with jax.named_scope("copy_scales_to_tmem"):
                plgpu.async_copy_scales_to_tmem(
                    a_scale_smem.at[slot], a_scale_tmem, collective_axis="x" if collective else None,
                )
                plgpu.async_copy_scales_to_tmem(
                    b_scale_smem.at[slot], b_scale_tmem, collective_axis="x" if collective else None,
                )
              with jax.named_scope("mma"):
                plgpu.tcgen05_mma(
                    acc_tmem,
                    a_smem.at[slot],
                    plgpu.transpose_ref(b_smem.at[slot], (1, 0)),
                    a_scale=a_scale_tmem,
                    b_scale=b_scale_tmem,
                    accumulate=(ki > 0),
                    barrier=consumed_barrier.at[slot],
                    collective_axis="x" if collective else None,
                )

              is_last_iter = ki >= k_iters - 1

              @pl.when(is_last_iter)
              def _():
                plgpu.tcgen05_commit_arrive(
                    mma_done_barrier,
                    collective_axis="x" if collective else None,
                )

            lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(wg_idx == STORE_WG)
      def _():
        plgpu.barrier_wait(mma_done_barrier)
        with jax.named_scope("store"):
          acc_smem[...] = plgpu.async_load_tmem(acc_tmem)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(
              acc_smem, out_gmem.at[block_slice_m, cluster_slice_n]
          )
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          plgpu.wait_load_tmem()  # Load must complete before we continue.
        plgpu.barrier_arrive(store_done_barrier)

  swizzle = plgpu.find_swizzle(block_k * jnp.finfo(dtype).bits)
  swizzle_elems = 8 * swizzle // dtypes.itemsize_bits(dtype)
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  out_swizzle = plgpu.find_swizzle(
      block_n * dtypes.itemsize_bits(jnp.float32),
  )
  out_swizzle_elems = 8 * out_swizzle // dtypes.itemsize_bits(jnp.float32)
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )
  block_size = 32
  if collective:
    store_done_barrier = plgpu.ClusterBarrier(
        num_arrivals=1,
        num_barriers=1,
        orders_tensor_core=True,
        collective_axes=("x",),
    )
  else:
    store_done_barrier = plgpu.Barrier(
        num_arrivals=1, num_barriers=1, orders_tensor_core=True
    )
  scratch_shapes = dict(
      a_smem=plgpu.SMEM(
          (max_concurrent_steps, block_m, block_k), dtype, transforms=transforms
      ),
      b_smem=plgpu.SMEM(
          (max_concurrent_steps, block_n, block_k), dtype, transforms=transforms
      ),
      a_scale_smem=plgpu.SMEM(
          (
              max_concurrent_steps,
              block_m // 128,
              block_k // block_size // 4,
              32,
              16,
          ),
          jnp.float8_e8m0fnu,
      ),
      b_scale_smem=plgpu.SMEM(
          (
              max_concurrent_steps,
              cluster_block_n // 128,
              block_k // block_size // 4,
              32,
              16,
          ),
          jnp.float8_e8m0fnu,
      ),
      acc_smem=plgpu.SMEM(
          (block_m, cluster_block_n), jnp.float32, transforms=out_transforms
      ),
      a_scale_tmem=plgpu.TMEM(
          (block_m, block_k // block_size),
          jnp.float8_e8m0fnu,
          layout=plgpu.TMEMLayout.SCALES_LAYOUT,
          collective=collective,
      ),
      b_scale_tmem=plgpu.TMEM(
          (cluster_block_n, block_k // block_size),
          jnp.float8_e8m0fnu,
          layout=plgpu.TMEMLayout.SCALES_LAYOUT,
          collective=collective,
      ),
      acc_tmem=plgpu.TMEM(
          (block_m, cluster_block_n), jnp.float32, collective=collective
      ),
      tma_barrier=plgpu.Barrier(
          num_arrivals=4, num_barriers=max_concurrent_steps
      ),
      consumed_barrier=plgpu.Barrier(
          num_arrivals=1,
          num_barriers=max_concurrent_steps,
          orders_tensor_core=True,
      ),
      mma_done_barrier=plgpu.Barrier(orders_tensor_core=True),
      store_done_barrier=store_done_barrier,
  )
  compiler_params = plgpu.CompilerParams(
      profile_space=100 if collect_profile else 0,
      profile_dir="sponge" if collect_profile else "",
  )
  num_sms = jax.local_devices()[0].core_count
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      grid=(num_sms // 2,) if collective else (num_sms,),
      grid_names=("sm",),
      scratch_shapes=scratch_shapes,
      num_threads=2,
      thread_name="wg",
      cluster_names=("x",),
      cluster=(1 + collective,),
      compiler_params=compiler_params,
  )
  return f(a, b, a_scale, b_scale)


@dataclasses.dataclass(frozen=True)
class Config:
  block_m: int = 128
  block_n: int = 128
  block_k: int = 128
  max_concurrent_steps: int = 1
  grid_minor_dim: MatmulDimension = MatmulDimension.N
  grid_tile_width: int = 1
  collective: bool = False
  collect_profile: bool = False


def format_scales(scales):
  mn, k = scales.shape
  assert mn % 128 == 0 and k % 4 == 0
  return (
      scales.reshape(mn // 128, 4, 32, k // 4, 4)
      .transpose(0, 3, 2, 1, 4)
      .reshape(mn // 128, k // 4, 32, 16)
  )


def main(_) -> None:
  dtype = jnp.float4_e2m1fn
  problem_shapes = [
      (4096, 4096, 8192),
  ]
  results = list()
  for m, n, k in problem_shapes:
    block_size = 32
    x = jax.random.uniform(
        jax.random.key(1), shape=(m, k), dtype=jnp.float32
    ).astype(dtype)
    y = jax.random.uniform(
        jax.random.key(2), shape=(n, k), dtype=jnp.float32
    ).astype(dtype)
    ksx, ksy = jax.random.split(jax.random.key(1234), 2)
    x_scale = jax.lax.bitcast_convert_type(
        jax.random.randint(
            ksx, (m, k // block_size), 122, 132, dtype=jnp.uint8
        ),
        jnp.float8_e8m0fnu,
    )
    y_scale = jax.lax.bitcast_convert_type(
        jax.random.randint(
            ksy, (n, k // block_size), 122, 132, dtype=jnp.uint8
        ),
        jnp.float8_e8m0fnu,
    )

    x_logical_scale = jnp.repeat(x_scale, 32, axis=1).astype(jnp.float32)
    y_logical_scale = jnp.repeat(y_scale, 32, axis=1).astype(jnp.float32)

    expected = jnp.dot(
        x.astype(jnp.float32) * x_logical_scale,
        (y.astype(jnp.float32) * y_logical_scale).T,
    )

    # (m//128, 4, 32, k//4, 4) -> (m//128, k//4, 32, 4, 4) -> (m//128, k//4, 32, 16)
    x_scale = format_scales(x_scale)
    # (n//128, 4, 32, k//4, 4) -> (n//128, k//4, 32, 4, 4) -> (n//128, k//4, 32, 16)
    y_scale = format_scales(y_scale)
    tuning_it = itertools.product(
        (128,),  # block_m
        (
            64,
            128,
            256,
        ),  # block_n
        (128, 256),  # block_k
        (2, 4, 6, 8),  # max_concurrent_steps
        MatmulDimension,  # grid_minor_dim
        (1, 4, 8, 12, 16),  # grid_tile_width
        (True, False),  # collective
    )
    best_util = -float("inf")
    best_config = None
    print(f"==== {m=} {n=} {k=} ====")
    for (
        block_m,
        block_n,
        block_k,
        max_concurrent_steps,
        grid_minor_dim,
        grid_tile_width,
        collective,
    ) in tuning_it:
      config = Config(
          block_m=block_m,
          block_n=block_n,
          block_k=block_k,
          max_concurrent_steps=max_concurrent_steps,
          grid_minor_dim=grid_minor_dim,
          grid_tile_width=grid_tile_width,
          collective=collective,
      )
      assert config.block_m == 128
      if config.block_n == 64 and not config.collective:
        continue
      print(f"Evaluating config: {config}")
      try:
        out, runtimes_ms = profiler.measure(
            functools.partial(matmul_kernel, config=config), iterations=10
        )(x, y, x_scale, y_scale)
        runtime_ms = statistics.median(runtimes_ms)
      except ValueError as e:
        if (
            "exceeds available shared memory" in e.args[0]
            or "Accumulator layout mismatch:" in e.args[0]
        ):
          print("SMEM OOM", config)
          # Accumulator layout mismatch triggers for tile_n=256 on some configs.
          continue
        raise
      np.testing.assert_allclose(out, expected, rtol=1e-3)
      runtime_us = runtime_ms * 1e3  # type: ignore
      matmul_flops = 2 * m * n * k
      peak_flops = 9e15  # fp4 TensorCore peak = 9000 TFLOPs
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      print(
          f"{block_m=} {block_n=} {block_k=} {max_concurrent_steps=} "
          f"{grid_minor_dim=} {grid_tile_width=} "
          f"{runtime_us:<4.1f}us "
          f"{achieved_tc_util:4.1f}% TC utilization"
      )
      if achieved_tc_util > best_util:
        best_util = achieved_tc_util
        best_config = config
    results.append(
        {"problem_shape": (m, n, k), "config": best_config, "util": best_util}
    )
  print("==== Results ====")
  for result in results:
    m, n, k = result["problem_shape"]
    config = result["config"]
    util = result["util"]
    print(f"==== {m=} {n=} {k=} ====")
    print(f"Best utilization: {util:4.1f}%, {config}")


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
o
