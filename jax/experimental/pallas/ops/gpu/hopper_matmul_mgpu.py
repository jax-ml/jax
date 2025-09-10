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


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
  tma_epilogue: bool


def _find_swizzle(dim_size_bits: int):
  """Finds the largest swizzle that fits the dimension size."""
  for swizzle_bytes in (128, 64, 32, 16):
    if dim_size_bits % (swizzle_bytes * 8) == 0:
      return swizzle_bytes
  raise ValueError(
      f"Dimension size has {dim_size_bits} bits, which is not a multiple of 128"
  )


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
  swizzle = _find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
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
  m_iters = m // (2 * tile_m)
  n_iters = n // tile_n
  k_iters = k // tile_k

  def kernel(a_gmem, b_gmem, out_gmem, out_smem, epilogue_entry_barrier):
    # TODO(apaszke): Grid tiling
    # TODO(apaszke): Avoid memory pipeline bubbles between MN blocks
    wg_idx = lax.axis_index("wg")
    if not config.tma_epilogue:
      @pl.when(wg_idx == 0)
      def _init_epilogue_done_barrier():
        plgpu.barrier_arrive(epilogue_entry_barrier.at[0])
    @plgpu.nd_loop((m_iters, n_iters), collective_axes="sm")
    def _mn_loop(idx):
      m_idx, n_idx = idx
      m_slice = pl.ds(m_idx * 2 * tile_m, 2 * tile_m)
      wg_m_slice = pl.ds(wg_idx * tile_m, tile_m)
      n_slice = pl.ds(n_idx * tile_n, tile_n)

      def prologue_epilogue(eval_pipeline):
        @functools.partial(
            pl.run_scoped, acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32)
        )
        def _acc_scope(acc_ref):
          eval_pipeline(acc_ref)
          if config.tma_epilogue:
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            out_smem[wg_m_slice] = acc_ref[...].astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                out_smem.at[wg_m_slice],
                out_gmem.at[m_slice, n_slice].at[wg_m_slice],
            )
          else:
            # Invariant: at entry and exit from _acc_scope,
            # epilogue_entry_barrier is complete for WG 0 but not for WG 1.
            plgpu.barrier_wait(epilogue_entry_barrier.at[wg_idx])
            out_smem[...] = acc_ref[...].astype(dtype)
            acc_to_store = plgpu.load(
                out_smem,
                (),
                layout=plgpu.Layout.SMEM_GMEM_COPY(
                    out_smem.shape, out_smem.dtype, swizzle=128
                ),
            )
            plgpu.barrier_arrive(epilogue_entry_barrier.at[1 - wg_idx])
            out_gmem.at[m_slice, n_slice][wg_m_slice] = acc_to_store

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

  num_sms = backend.get_default_device().core_count
  if config.tma_epilogue:
    scratch_shapes = [
        plgpu.SMEM((2 * tile_m, tile_n), dtype, transforms=transforms),
        None,
    ]
  else:
    scratch_shapes = [
        plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
        plgpu.Barrier(num_arrivals=1, num_barriers=2),
    ]
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
  problem_it = [(4096, 4096, 4096)]
  for M, N, K in problem_it:
    print(f"==== {M=} {N=} {K=} ====")
    matmul_flops = 2 * M * N * K
    peak_flops = 990e12  # f16 TensorCore peak = 990 TFLOPS
    a = jax.random.uniform(jax.random.key(0), (M, K), jnp.float16)
    b = jax.random.uniform(jax.random.key(1), (K, N), jnp.float16)
    ref = a @ b
    tuning_it = itertools.product(
        (64, 128),  # tile_m
        (64, 128, 256),  # tile_n
        (64,),  # tile_k
        (2, 4),  # max_concurrent_steps
        (False, True),  # tma_epilogue
    )
    best_util = 0.0
    best_runtime = float("inf")
    for tile_m, tile_n, tile_k, max_concurrent_steps, tma_epilogue in tuning_it:
      config = TuningConfig(
          tile_m=tile_m,
          tile_n=tile_n,
          tile_k=tile_k,
          max_concurrent_steps=max_concurrent_steps,
          tma_epilogue=tma_epilogue,
      )
      try:
        out, runtimes_ms = profiler.measure(
            functools.partial(matmul_kernel, config=config), mode="cupti", iterations=10,
        )(a, b)
        runtime_ms = statistics.median(runtimes_ms)
      except ValueError as e:
        if "exceeds available shared memory" in e.args[0]:  # Ignore SMEM OOMs.
          print(
              f"{tile_m=} {tile_n=} {tile_k=} {max_concurrent_steps=} {tma_epilogue=}: OOM"
          )
          continue
        raise
      np.testing.assert_allclose(out, ref)
      runtime_us = runtime_ms * 1e3   # type: ignore
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      if achieved_tc_util > best_util:
        best_runtime = runtime_us
        best_util = achieved_tc_util
      print(
          f"{tile_m=} {tile_n=} {tile_k=} {max_concurrent_steps=} {tma_epilogue=}:"
          f" {runtime_us:<7.1f}us = {achieved_tc_util:4.1f}% TC utilization"
      )
    print(f"\tBest: {best_runtime:<7.1f}us = {best_util:4.1f}% TC utilization")


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
