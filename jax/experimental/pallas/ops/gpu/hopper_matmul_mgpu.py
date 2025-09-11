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
"""Matrix Multiplication kernel for Hopper GPUs."""
import statistics
import dataclasses
import enum
import functools
import itertools
import jax
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.extend import backend
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
  epi_tile_n: int | None = 64  # This needs to be lowered for for small N.
  epi_tile_m: int | None = 64
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
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  max_concurrent_steps = config.max_concurrent_steps
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % (2 * tile_m) != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  epi_tile_n = config.epi_tile_n or tile_n
  epi_tile_m = config.epi_tile_m or tile_m
  if tile_n % epi_tile_n != 0:
    raise ValueError(f"{tile_n=} must be divisible by {epi_tile_n=}")
  if tile_m % epi_tile_m != 0:
    raise ValueError(f"{tile_m=} must be divisible by {epi_tile_m=}")
  m_iters = m // (2 * tile_m)
  n_iters = n // tile_n
  k_iters = k // tile_k

  def kernel(a_gmem, b_gmem, out_gmem, out_smem):
    # TODO(apaszke): Avoid memory pipeline bubbles between MN blocks
    wg_idx = lax.axis_index("wg")
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
    def _mn_loop(idxs):
      (lin_idx,) = idxs
      m_idx, n_idx = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      m_slice = pl.ds(m_idx * 2 * tile_m, 2 * tile_m)
      wg_m_slice = pl.ds(lax.axis_index("wg") * tile_m, tile_m)
      n_slice = pl.ds(n_idx * tile_n, tile_n)

      def prologue_epilogue(eval_pipeline):
        @functools.partial(
            pl.run_scoped, acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32)
        )
        def _acc_scope(acc_ref):
          eval_pipeline(acc_ref)
          acc = acc_ref[...].astype(dtype)
          plgpu.wait_smem_to_gmem(0, wait_read_only=True)
          for epi_mi in range(tile_m // epi_tile_m):
            for epi_ni in range(tile_n // epi_tile_n):
              epi_m_slice = slice(epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m)
              epi_n_slice = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
              slot = (epi_mi * (tile_n // epi_tile_n) + epi_ni) % 2
              plgpu.wait_smem_to_gmem(1, wait_read_only=True)
              out_smem[wg_idx, slot] = acc[epi_m_slice, epi_n_slice]
              plgpu.commit_smem()
              plgpu.copy_smem_to_gmem(
                  out_smem.at[wg_idx, slot],
                  out_gmem.at[m_slice, n_slice].at[wg_m_slice].at[epi_m_slice, epi_n_slice],
              )

      def mma_body(_, a_smem, b_smem, acc_ref):
        plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem)
        return acc_ref

      plgpu.emit_pipeline_warp_specialized(
          mma_body,
          grid=(k_iters,),
          memory_registers=40,
          in_specs=[
              plgpu.BlockSpec(
                  (2 * tile_m, tile_k),
                  lambda k: (0, k),
                  transforms=transforms,
                  memory_space=plgpu.SMEM,
                  delay_release=1,
              ),
              plgpu.BlockSpec(
                  (tile_k, tile_n),
                  lambda k: (k, 0),
                  transforms=transforms,
                  memory_space=plgpu.SMEM,
                  delay_release=1,
              ),
          ],
          wg_axis="wg",
          num_compute_wgs=2,
          max_concurrent_steps=max_concurrent_steps,
          compute_context=prologue_epilogue,
      )(a_gmem.at[m_slice, :], b_gmem.at[:, n_slice])
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  # We don't need multiple slots if there's only one epilogue tile.
  num_out_slots = min(2, (tile_m * tile_n) // (epi_tile_m * epi_tile_n))
  out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(dtype).itemsize * 8)
  out_swizzle_elems = out_swizzle // jnp.dtype(dtype).itemsize
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )
  scratch_shapes = [
      plgpu.SMEM(
          (2, num_out_slots, epi_tile_m, epi_tile_n),
          dtype,
          transforms=out_transforms,
      ),
  ]
  num_sms = backend.get_default_device().core_count
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms,),
      grid_names=("sm",),
      num_threads=3,
      thread_name="wg",
      scratch_shapes=scratch_shapes,
  )
  return f(a, b)


def main(_) -> None:
  problem_it = [(4096, 8192, 4096)]
  for M, N, K in problem_it:
    print(f"==== {M=} {N=} {K=} ====")
    matmul_flops = 2 * M * N * K
    peak_flops = 990e12  # f16 TensorCore peak = 990 TFLOPS
    a = jax.random.uniform(jax.random.key(0), (M, K), jnp.float16)
    b = jax.random.uniform(jax.random.key(1), (K, N), jnp.float16)
    tuning_it = itertools.product(
        (64, 128,),  # tile_m
        (64, 128,),  # tile_n
        (64,),  # tile_k
        (4,),  # max_concurrent_steps
        (True,),  # Tiled epilogue
        (MatmulDimension.M, MatmulDimension.N),  # grid_minor_dim
        (1, 4, 8, 16),  # grid_tile_width
    )
    best_util = 0.0
    best_runtime = float("inf")
    for tile_m, tile_n, tile_k, max_concurrent_steps, tiled_epilogue, grid_minor_dim, grid_tile_width in tuning_it:
      config = TuningConfig(
          tile_m=tile_m,
          tile_n=tile_n,
          tile_k=tile_k,
          max_concurrent_steps=max_concurrent_steps,
          epi_tile_n=64 if tiled_epilogue else None,
          epi_tile_m=64 if tiled_epilogue else None,
          grid_minor_dim=grid_minor_dim,
          grid_tile_width=grid_tile_width,
      )
      try:
        out, runtimes_ms = profiler.measure(
            functools.partial(matmul_kernel, config=config), iterations=10,
        )(a, b)
        assert runtimes_ms is not None
        runtime_ms = statistics.median(runtimes_ms)
      except ValueError as e:
        if "exceeds available shared memory" in e.args[0]:  # Ignore SMEM OOMs.
          continue
        raise
      np.testing.assert_allclose(out, a @ b)
      runtime_us = runtime_ms * 1e3   # type: ignore
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      if achieved_tc_util > best_util:
        best_runtime = runtime_us
        best_util = achieved_tc_util
      print(
          f"{tile_m=} {tile_n=} {tile_k=} {max_concurrent_steps=} {tiled_epilogue=} {grid_minor_dim=} {grid_tile_width=}:"
          f" {runtime_us:<7.1f}us = {achieved_tc_util:4.1f}% TC utilization"
      )
    print(f"\tBest: {best_runtime:<7.1f}us = {best_util:4.1f}% TC utilization")


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
