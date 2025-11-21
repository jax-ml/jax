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
"""Transposed ragged dot Pallas-Mosaic-GPU implementation."""

import functools
import itertools
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as plgpu
import numpy as np


def transposed_ragged_dot(
    lhs,  # (K, M)
    rhs,  # (K, N)
    *,
    group_sizes,  # (G,)
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
) -> jax.Array:
  if lhs.dtype != rhs.dtype:
    raise NotImplementedError(
        f"lhs and rhs must have the same dtype, got {lhs.dtype} and {rhs.dtype}"
    )
  k, m = lhs.shape
  k2, n = rhs.shape
  g = group_sizes.shape[0]

  if k != k2:
    raise ValueError(f"lhs.shape={k} must match rhs.shape={k2}")

  if m % block_m != 0:
    raise ValueError(f"m={m} must be a multiple of block_m={block_m}")
  if n % block_n != 0:
    raise ValueError(f"n={n} must be a multiple of block_n={block_n}")

  group_sizes = group_sizes.astype(int)
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=int), jnp.cumsum(group_sizes)[:-1]]
  ).astype(int)
  group_ends = jnp.cumsum(group_sizes)
  group_block_starts = group_starts // block_k * block_k
  group_block_ends = -(group_ends // -block_k) * block_k
  group_num_blocks = (group_block_ends - group_block_starts) // block_k

  swizzle = plgpu.find_swizzle(block_k * jnp.dtype(lhs.dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(lhs.dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )

  def body(
      group_sizes_gmem,
      group_starts_gmem,
      group_ends_gmem,
      group_num_blocks_gmem,
      group_block_starts_gmem,
      lhs_gmem,
      rhs_gmem,
      o_gmem,
  ):

    grid_m = pl.cdiv(m, block_m)
    grid_n = pl.cdiv(n, block_n)

    @plgpu.nd_loop((g, grid_m * grid_n), collective_axes="sm")
    def mn_loop(loop_info: plgpu.NDLoopInfo):
      g_i = loop_info.index[0]
      m_i, n_i = plgpu.planar_snake(
          loop_info.index[1],
          (grid_m, grid_n),
          1,
          grid_block_n,
      )

      # This slice is potentially out of bounds, but we never access the
      # out of bound part in emit_pipeline.
      gmem_slice = pl.ds(group_block_starts_gmem[g_i], k)

      def acc_scope(acc_ref):
        def block_matmul(block_idx, lhs_smem, rhs_smem):
          block_idx = block_idx[0]

          @pl.when(block_idx == 0)
          def _():
            # Handles the first block of the group, where there might be
            # data from the previous group in the beginning of the block.
            lhs_reg = lhs_smem[...]
            start_index = lax.rem(group_starts_gmem[g_i], block_k)
            indices = plgpu.layout_cast(
                jax.lax.broadcasted_iota(jnp.int32, (block_k, block_m), 0),
                plgpu.Layout.WGMMA
            )
            lhs_mask = (indices >= start_index).astype(lhs_smem.dtype)

            lhs_reg = lhs_reg * lhs_mask
            lhs_smem[...] = lhs_reg
            plgpu.commit_smem()

          @pl.when(block_idx == group_num_blocks_gmem[g_i] - 1)
          def _():
            # Handles the last block of the group, where there might be
            # data from the next group in the end of the block.
            lhs_reg = lhs_smem[...]
            last_index = lax.rem(group_ends_gmem[g_i] - 1, block_k)
            indices = plgpu.layout_cast(
                jax.lax.broadcasted_iota(jnp.int32, (block_k, block_m), 0),
                plgpu.Layout.WGMMA
            )
            lhs_mask = (indices <= last_index).astype(lhs_smem.dtype)

            lhs_reg = lhs_reg * lhs_mask
            lhs_smem[...] = lhs_reg
            plgpu.commit_smem()

          plgpu.wgmma(acc_ref, plgpu.transpose_ref(lhs_smem, (1, 0)), rhs_smem)
          if max_concurrent_steps == 1:
            # Without delayed release, we won't have at least two separate
            # smem blocks in flight. Therefore, we cannot rely on the implicit
            # wait of wgmma to gaurantee that the data in smem is ready to be
            # overwritten by the next pipeline iteration.
            plgpu.wgmma_wait(0)

        @pl.when(group_sizes_gmem[g_i] > 0) # Skip the group if it is empty.
        def _():
          plgpu.emit_pipeline(
              block_matmul,
              grid=(group_num_blocks_gmem[g_i],),
              in_specs=[
                  plgpu.BlockSpec(
                      (block_k, block_m),
                      lambda k_i: (k_i, m_i),
                      delay_release=1 if max_concurrent_steps > 1 else 0,
                      transforms=transforms,
                  ),
                  plgpu.BlockSpec(
                      (block_k, block_n),
                      lambda k_i: (k_i, n_i),
                      delay_release=1 if max_concurrent_steps > 1 else 0,
                      transforms=transforms,
                  ),
              ],
              max_concurrent_steps=max_concurrent_steps,
          )(lhs_gmem.at[gmem_slice, :], rhs_gmem.at[gmem_slice, :])

        return acc_ref[...]

      acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

      @functools.partial(
          pl.run_scoped,
          o_smem=plgpu.SMEM(
              (block_m, block_n),
              dtype=o_gmem.dtype,
              transforms=transforms,
          )
      )
      def store_scope(o_smem):
        o_smem[...] = acc.astype(o_smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(
            o_smem, o_gmem.at[
                g_i,
                pl.ds(m_i * block_m, block_m),
                pl.ds(n_i * block_n, block_n)
            ]
        )
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  # There are 132 SMs on a H100 SXM GPU.
  num_sms = jax.devices()[0].core_count
  kernel = plgpu.kernel(
      body,
      out_shape=jax.ShapeDtypeStruct((g, m, n), lhs.dtype),
      grid=(num_sms,),
      grid_names=("sm",),
  )
  return kernel(
      group_sizes,
      group_starts,
      group_ends,
      group_num_blocks,
      group_block_starts,
      lhs,
      rhs,
  )


def ref_transposed_ragged_dot(lhs, rhs, group_sizes):
  return jax.lax.ragged_dot_general(
      lhs, rhs, group_sizes,
      ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
          dot_dimension_numbers=(((0,), (0,)), ((), ())),
          lhs_ragged_dimensions=[0],
          rhs_group_dimensions=[],
      )
  )


def main(unused_argv):
  k, m, n, num_groups = 16 * 1024, 2048, 2048, 16
  kx, ky, kz = random.split(random.key(1234), num=3)

  lhs = jax.random.normal(kx, (k, m), jnp.float16)
  rhs = jax.random.normal(ky, (k, n), jnp.float16)
  group_boundaries = jax.lax.sort(
      jax.random.randint(kz, (num_groups - 1,), 0, k, jnp.int32)
  )
  group_starts = lax.concatenate(
      [jnp.array([0], dtype=jnp.int32), group_boundaries], 0
  )
  group_ends = lax.concatenate(
      [group_boundaries, jnp.array([k], dtype=jnp.int32)], 0
  )
  group_sizes = group_ends - group_starts
  assert group_sizes.shape == (num_groups,)

  block_m = block_n = [64, 128]
  block_k = [64, 128]
  max_concurrent_steps = [1, 2, 4, 5, 6]
  grid_block_n = [1, 2, 4, 8, 16]

  configs = itertools.product(
      block_m, block_n, block_k, max_concurrent_steps, grid_block_n
  )
  names = (
      "block_m", "block_n", "block_k", "max_concurrent_steps", "grid_block_n",
  )
  best_runtime = float("inf")
  best_kwargs = {}
  for config in configs:
    kwargs = dict(zip(names, config))
    if n %  kwargs["block_n"]:
      continue
    try:
      f = functools.partial(
          transposed_ragged_dot, group_sizes=group_sizes,
          **kwargs
      )
      _, runtime = profiler.measure(f)(lhs, rhs)
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" not in str(e):
        raise
      runtime = float("inf")
    # Enable this to get more detailed information.
    else:
      print(
          " ".join(f"{k}={v}" for k, v in kwargs.items()),
          f"{int(runtime * 1000):.1f} us",
      )
    if runtime < best_runtime:  # pytype: disable=unsupported-operands
      best_runtime = runtime
      best_kwargs = kwargs
  if not best_kwargs:
    raise ValueError("No valid configuration found")

  ref, ref_runtime = profiler.measure(ref_transposed_ragged_dot)(
      lhs, rhs, group_sizes=group_sizes
  )
  result = transposed_ragged_dot(
      lhs, rhs, group_sizes=group_sizes, **best_kwargs
  )

  tflops = float(2 * k * m * n) / (best_runtime / 1e3) / 1e12
  ref_tflops = float(2 * k * m * n) / (ref_runtime / 1e3) / 1e12
  print(
      "Best parameters: ", " ".join(f"{k}={v}" for k, v in best_kwargs.items())
  )
  print(f"Kernel:    {best_runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
  print(f"Reference: {ref_runtime * 1000:.1f} us = {ref_tflops:.1f} TFLOPS")
  np.testing.assert_allclose(result, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
