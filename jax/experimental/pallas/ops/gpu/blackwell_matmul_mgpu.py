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
import functools
import itertools
import math
import jax
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np
from functools import partial


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  blk_m: int
  blk_n: int
  blk_k: int
  max_concurrent_steps: int


def matmul_kernel(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  k2, n = b.shape
  if k != k2:
    raise ValueError(
        f"Matmul LHS and RHS have incompatible shapes {a.shape} vs {b.shape}")
  out_shape = (m, n)
  blk_m, blk_n, blk_k = (config.blk_m, config.blk_n, config.blk_k)
  blk_lhs = (blk_m, blk_k)
  blk_rhs = (blk_k, blk_n)
  blk_out = (blk_m, blk_n)
  if m % blk_m != 0:
    raise ValueError(f"{m=} must be divisible by {blk_m=}")
  if n % blk_n != 0:
    raise ValueError(f"{n=} must be divisible by {blk_n=}")
  if k % blk_k != 0:
    raise ValueError(f"{k=} must be divisible by {blk_k=}")
  m_iters = m // blk_m
  n_iters = n // blk_n
  k_iters = k // blk_k
  max_concurrent_steps = config.max_concurrent_steps
  swizzle = 128
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  warp_mesh = plgpu.WarpMesh(axis_name="warp")

  def kernel(a_gmem, b_gmem, out_gmem):
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    slice_m = pl.ds(m_index * blk_m, blk_m)
    slice_n = pl.ds(n_index * blk_n, blk_n)

    @functools.partial(pl.run_scoped,
      a_smem=plgpu.SMEM((max_concurrent_steps, *blk_lhs), dtype, transforms=transforms),
      b_smem=plgpu.SMEM((max_concurrent_steps, *blk_rhs), dtype, transforms=transforms),
      acc_tmem=plgpu.TMEM(blk_out, jnp.float32, collective=False),
      scratch_smem=plgpu.SMEM(blk_out, dtype, transforms=transforms),
      a_tma_barrier=plgpu.Barrier(
          num_arrivals=1, num_barriers=max_concurrent_steps,
          thread_scope=plgpu.ThreadScope.WARP),
      b_tma_barrier=plgpu.Barrier(
          num_arrivals=1, num_barriers=max_concurrent_steps,
          thread_scope=plgpu.ThreadScope.WARP),
      consumed_barrier=plgpu.Barrier(
          num_arrivals=1, num_barriers=max_concurrent_steps,
          thread_scope=plgpu.ThreadScope.WARP),
      mma_done_barrier=plgpu.Barrier(num_arrivals=1, for_tensor_core=True),
    )
    def _scoped(a_smem, b_smem,
                acc_tmem, scratch_smem,
                a_tma_barrier, b_tma_barrier,
                consumed_barrier,
                mma_done_barrier,
                ):
      @pl.core_map(warp_mesh)
      def _per_warp():
        warp_id = lax.axis_index("warp")
        @pl.when(warp_id == 0)
        def _memory():
          def _loop_body(ki, _):
            slot = lax.rem(ki, max_concurrent_steps)
            @pl.when(ki >= max_concurrent_steps)
            def _():
              plgpu.barrier_wait(consumed_barrier.at[slot])
            slice_k = pl.ds(ki * blk_k, blk_k)
            plgpu.copy_gmem_to_smem(
                a_gmem.at[slice_m, slice_k], a_smem.at[slot],
                a_tma_barrier.at[slot])
            plgpu.copy_gmem_to_smem(
                b_gmem.at[slice_k, slice_n], b_smem.at[slot],
                b_tma_barrier.at[slot])
          lax.fori_loop(0, k_iters, _loop_body, None)

        @pl.when(warp_id == 1)
        def _compute():
          def _loop_body(ki, _):
            slot = lax.rem(ki, max_concurrent_steps)
            plgpu.barrier_wait(a_tma_barrier.at[slot])
            plgpu.barrier_wait(b_tma_barrier.at[slot])
            is_last_iter = ki >= k_iters - 1
            # TODO(justinfu): Implement select on barriers instead of using
            # a conditional.
            @pl.when(~is_last_iter)
            def _():
              plgpu.tcgen05_mma(acc_tmem,
                                a_smem.at[slot],
                                b_smem.at[slot],
                                consumed_barrier.at[slot],
                                accumulate=(ki != 0))
            @pl.when(is_last_iter)
            def _():
              plgpu.tcgen05_mma(acc_tmem,
                                a_smem.at[slot],
                                b_smem.at[slot],
                                mma_done_barrier,
                                accumulate=True)
          lax.fori_loop(0, k_iters, _loop_body, None)

      plgpu.barrier_wait(mma_done_barrier)
      scratch_smem[...] = acc_tmem[...].astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_smem, out_gmem.at[slice_m, slice_n])
      plgpu.wait_smem_to_gmem(0)

  f = plgpu.kernel(
    kernel,
    out_shape=jax.ShapeDtypeStruct(out_shape, dtype),
    grid=(m_iters, n_iters),
    grid_names=("m", "n"),
  )
  return f(a, b)

def main(_) -> None:
  problem_it = itertools.product(
      (1024, 4096, 8192), (1024, 4096, 8192), (1024, 8192))
  for M, N, K in problem_it:
    print(f"==== {M=} {N=} {K=} ====")
    a = jax.random.uniform(jax.random.key(0), (M, K), jnp.bfloat16)
    b = jax.random.uniform(jax.random.key(1), (K, N), jnp.bfloat16)
    tuning_it = itertools.product((128,), (128,), (128,), (2, 3))
    for blk_m, blk_n, blk_k, max_concurrent_steps in tuning_it:
      config = TuningConfig(blk_m=blk_m, blk_n=blk_n, blk_k=blk_k,
                                  max_concurrent_steps=max_concurrent_steps)
      try:
        out, runtime_ms = profiler.measure(functools.partial(matmul_kernel, config=config))(a, b)
      except ValueError as e:
        if "exceeds available shared memory" in e.args[0]:
          continue
        raise
      if M*N*K < 1024*1024*1024:
        expected = a @ b
        np.testing.assert_allclose(out, expected)
      runtime_us = runtime_ms * 1e3
      matmul_flops = 2 * M * N * K
      peak_flops = 2.25e15  # f16 TensorCore peak = 2250 TFLOPS
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      print(
          f"{blk_m=} {blk_n=} {blk_k=} {max_concurrent_steps=}:  "
          f"{runtime_us:<7.1f}us"
          f" = {achieved_tc_util:4.1f}% TC utilization"
      )


if __name__ == "__main__":
  from absl import app
  jax.config.config_with_absl()
  app.run(main)
