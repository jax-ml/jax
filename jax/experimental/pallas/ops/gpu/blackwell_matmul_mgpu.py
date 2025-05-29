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
import jax
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  block_m: int
  block_n: int
  block_k: int
  max_concurrent_steps: int
  collective: bool


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
  collective = config.collective
  if collective:
    raise ValueError("Collective matmul is not supported yet.")
  block_m, block_n, block_k = (config.block_m, config.block_n, config.block_k)
  swizzle = _find_swizzle(block_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )
  block_lhs = (block_m, block_k)
  block_rhs = (block_k, block_n)
  block_out = (block_m, block_n)
  if m % block_m != 0:
    raise ValueError(f"{m=} must be divisible by {block_m=}")
  if n % block_n != 0:
    raise ValueError(f"{n=} must be divisible by {block_n=}")
  if k % block_k != 0:
    raise ValueError(f"{k=} must be divisible by {block_k=}")
  m_iters = m // block_m
  n_iters = n // block_n
  k_iters = k // block_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             a_tma_barrier, b_tma_barrier, consumed_barrier):
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    slice_m = pl.ds(m_index * block_m, block_m)
    slice_n = pl.ds(n_index * block_n, block_n)
    acc_slice_m = pl.ds(m_index * block_m, block_m)
    acc_slice_n = pl.ds(n_index * block_n, block_n)

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)

          @pl.when(ki >= max_concurrent_steps)
          def _():
            plgpu.barrier_wait(consumed_barrier.at[slot])

          slice_k = pl.ds(ki * block_k, block_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[slice_m, slice_k],
              a_smem.at[slot],
              a_tma_barrier.at[slot],
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[slice_k, slice_n],
              b_smem.at[slot],
              b_tma_barrier.at[slot],
          )

        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(warp_id == 1)
      def _compute():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(a_tma_barrier.at[slot])
          plgpu.barrier_wait(b_tma_barrier.at[slot])
          is_last_iter = ki >= k_iters - 1
          barrier_slot = lax.select_n(is_last_iter,
                                      slot, max_concurrent_steps)
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barrier.at[barrier_slot],
              accumulate=(ki > 0),
          )
        lax.fori_loop(0, k_iters, _loop_body, None)

    plgpu.barrier_wait(consumed_barrier.at[max_concurrent_steps])
    acc_smem[...] = acc_tmem[...].astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(
        acc_smem, out_gmem.at[acc_slice_m, acc_slice_n]
    )
    plgpu.wait_smem_to_gmem(0)

  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      # TODO(justinfu): Add collective support.
      cluster_names=(),
      cluster=(),
      scratch_shapes=(   # type: ignore
        plgpu.SMEM(
            (max_concurrent_steps, *block_lhs), dtype, transforms=transforms
        ),
        plgpu.SMEM(
            (max_concurrent_steps, *block_rhs), dtype, transforms=transforms
        ),
        plgpu.TMEM(block_out, jnp.float32, collective=collective),
        plgpu.SMEM(block_out, dtype, transforms=transforms),
        plgpu.Barrier(
            num_arrivals=1, num_barriers=max_concurrent_steps
        ),
        plgpu.Barrier(
            num_arrivals=1, num_barriers=max_concurrent_steps
        ),
        plgpu.Barrier(
            num_arrivals=1,
            num_barriers=max_concurrent_steps + 1,
            for_tensor_core=True,
        ),
      )
  )
  return f(a, b)


def main(_) -> None:
  problem_it = itertools.product(
      (1024, 4096, 8192), (1024, 4096, 8192), (1024, 8192)
  )
  for M, N, K in problem_it:
    print(f"==== {M=} {N=} {K=} ====")
    matmul_flops = 2 * M * N * K
    peak_flops = 2.25e15  # f16 TensorCore peak = 2250 TFLOPS
    a = jax.random.uniform(jax.random.key(0), (M, K), jnp.bfloat16)
    b = jax.random.uniform(jax.random.key(1), (K, N), jnp.bfloat16)
    tuning_it = itertools.product(
        (128,), (128, 256), (64, 128), (2, 3, 4), (False,)
    )
    best_util = -float("inf")
    for (block_m, block_n, block_k,
         max_concurrent_steps, collective) in tuning_it:
      config = TuningConfig(
          block_m=block_m,
          block_n=block_n,
          block_k=block_k,
          max_concurrent_steps=max_concurrent_steps,
          collective=collective,
      )
      try:
        out, runtime_ms = profiler.measure(
            functools.partial(matmul_kernel, config=config)
        )(a, b)
      except ValueError as e:
        if "exceeds available shared memory" in e.args[0]:
          continue
        raise
      if M * N * K <= 1024 * 1024 * 1024:
        expected = a @ b
        np.testing.assert_allclose(out, expected)
      runtime_us = runtime_ms * 1e3   # type: ignore
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      if achieved_tc_util > best_util:
        best_util = achieved_tc_util
      print(
          f"{block_m=} {block_n=} {block_k=} {max_concurrent_steps=}:  "
          f"{runtime_us:<7.1f}us"
          f" = {achieved_tc_util:4.1f}% TC utilization"
      )
    print(f"\tBest utilization: {best_util:4.1f}%")


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
