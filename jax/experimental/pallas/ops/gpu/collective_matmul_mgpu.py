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

"""A collective matmul kernel implemented using Mosaic GPU."""

import functools
import itertools

import jax
import os
from jax import lax
from jax.experimental import multihost_utils
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu
import jax.numpy as jnp


MatmulDimension = hopper_matmul_mgpu.MatmulDimension
TuningConfig = hopper_matmul_mgpu.TuningConfig


def is_nvshmem_used() -> bool:
  return (
      "XLA_FLAGS" in os.environ
      and "--xla_gpu_experimental_enable_nvshmem" in os.environ["XLA_FLAGS"]
  )

def all_gather_lhs_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    axis_name,
    *,
    config: hopper_matmul_mgpu.TuningConfig,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
  if (
      num_devices := jax.device_count()
  ) != jax.process_count() and num_devices != jax.local_device_count():
    raise ValueError(
        "Kernel requires either 1 process per single GPU or 1 process per all"
        " GPUs."
    )
  if (axis_size := lax.axis_size(axis_name)) != num_devices:
    raise ValueError("The kernel can only work over all devices in a Mesh.")
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(f"Only f16 and bf16 are supported, got {dtype=}")
  if config.cluster_dimension is not None:
    raise NotImplementedError("Cluster dimension must be None for all-gather matmuls.")

  m_shard, k = lhs.shape
  k2, n_shard = rhs.shape
  if k != k2:
    raise ValueError(
        f"lhs and rhs must have the same contraction size, got {k} and {k2}."
    )
  if (element_type := lhs.dtype) != rhs.dtype:
    raise ValueError(
        f"lhs and rhs must have the same element type, got {element_type} and"
        f" {rhs.dtype}."
    )
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  max_concurrent_steps = config.max_concurrent_steps
  if max_concurrent_steps < 2:
    raise ValueError("max_concurrent_steps must be >= 2")
  cta_tile_m = tile_m * (1 + (config.wg_dimension == MatmulDimension.M))

  epi_tile_n = config.epi_tile_n or tile_n
  epi_tile_m = config.epi_tile_m or tile_m
  if tile_n % epi_tile_n != 0:
    raise ValueError(f"{tile_n=} must be divisible by {epi_tile_n=}")
  if tile_m % epi_tile_m != 0:
    raise ValueError(f"{tile_m=} must be divisible by {epi_tile_m=}")

  num_sms = jax.devices()[0].core_count  # 132 for H100 SXM GPUs.

  def kernel_body(lhs_local_ref, rhs_ref, out_ref, scratch_ref):
    received_sem = pl.get_global(plgpu.SemaphoreType.REGULAR)
    wg_idx = lax.axis_index("wg")
    dev_id = lax.axis_index(axis_name)
    send_dev_id = lax.rem(dev_id + axis_size - 1, jnp.int32(axis_size))
    send_scratch_ref = plgpu.remote_ref(scratch_ref, send_dev_id)

    def send_lhs(m_idx, n_idx, k_idx, a_smem, b_smem, send_ref, should_send):
      del b_smem  # Unused.
      # We only send when n_idx == 0 to avoid sending the same data
      # multiple times when revisiting lhs.
      @pl.when(should_send & jnp.bool(n_idx == 0))
      def _():
        k_slice = pl.ds(k_idx * tile_k, tile_k)
        m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
        plgpu.copy_smem_to_gmem(a_smem, send_ref.at[m_slice, k_slice])
        # We only delay release by 1 step, so we need to wait for the
        # previous copies.
        plgpu.wait_smem_to_gmem(1, wait_read_only=True)

    def device_step(lhs_source_ref, device_offset):
      # Invariant: lhs_source_ref is ready to be used
      next_scratch_slot = device_offset
      out_device_m_slice = pl.ds(
          lax.rem(device_offset + dev_id, jnp.int32(num_devices)) * m_shard,
          m_shard,
      )
      is_send_wg = wg_idx == 0
      has_send_space = next_scratch_slot < num_devices - 1
      should_send = is_send_wg & has_send_space

      # This reuses the regular matmul kernel, only with the exception of
      # inserting send_lhs into the pipeline.
      # TODO(apaszke): This contains run_scoped inside, meaning that it will
      # synchronize all threads at each device step. If we optimize the barrier
      # below, then it might be better to move it out to make bubbles smaller.
      hopper_matmul_mgpu.kernel(
          lhs_source_ref,  # Use the lhs from previous step.
          rhs_ref,  # Use the same rhs for all steps.
          out_ref.at[out_device_m_slice],  # Use a slice of the output.
          config=config,
          pipeline_callback=functools.partial(
              send_lhs,
              send_ref=send_scratch_ref.at[next_scratch_slot],
              should_send=should_send,
          ),
          delay_release=1,
      )

      # Wait for the next scratch to arrive --- see the device loop invariant.
      @pl.when(should_send)
      def _signal():
        # TODO(apaszke): We could do this signal a lot earlier if we better
        # control the order of sends. If we tile the grid along N, then we can
        # signal as soon as everyone moves on from the first column tile.
        # Make sure the copy is done and signal the receiving device.
        plgpu.wait_smem_to_gmem(0, wait_read_only=False)
        pl.semaphore_signal(received_sem, device_id=send_dev_id)
      @pl.when(next_scratch_slot < num_devices - 1)
      def _wait():
        pl.semaphore_wait(received_sem, value=(device_offset + 1) * num_sms, decrement=False)

    # We peel the first step to copy data directly form lhs_local_ref.
    device_step(lhs_local_ref, 0)
    @pl.loop(1, num_devices)
    def _device_loop(device_offset):
      device_step(scratch_ref.at[device_offset - 1], device_offset)
    # Make sure all copies are fully done.
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  result, _ = plgpu.kernel(
      kernel_body,
      out_shape=[
          # The output, with its M dimension all-gathered.
          jax.ShapeDtypeStruct((axis_size * m_shard, n_shard), dtype),
          # The scratch buffer used for the all-gather.
          jax.ShapeDtypeStruct((num_devices - 1, m_shard, k), dtype),
      ],
      grid=(num_sms,),
      grid_names=("cluster_grid",),
      num_threads=3,
      thread_name="wg",
      cluster=(1,),
      cluster_names=("cluster",),
  )(lhs, rhs)
  return result


def _min_results_across_devices(kernels_ms : list[tuple[str, float]]) -> float:
  # We choose the minimum across processes to choose the runtime that didn't
  # include devices waiting for other devices.
  if is_nvshmem_used():
    time_us = sum(t * 1e3 for _, t in kernels_ms)
    return min(multihost_utils.process_allgather(time_us).tolist())

  # profiler.measures measures all devices visible to the process, so we
  # need to select the mimimum result of each kernel across devices.
  # This code relies on the fact that with collective metadata a single kernel
  # with unique name is launched on each device.
  min_values : dict[str, float] = {}
  for kernel_name, t in kernels_ms:
    if kernel_name not in min_values or t < min_values[kernel_name]:
      min_values[kernel_name] = t
  return sum(time_ms * 1e3 for time_ms in min_values.values())


def _run_example():
  P = jax.sharding.PartitionSpec
  m_shard = 1024
  n_shard = 4096
  k = 4096
  dtype = jnp.bfloat16
  shards = jax.device_count()
  mesh = jax.make_mesh(
      (shards,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
  )
  jax.set_mesh(mesh)

  # We measure time per-shard and so we only need FLOPs per shard.
  matmul_flops = 2 * (shards * m_shard) * n_shard * k
  peak_flops = 990e12  # f16 TensorCore peak = 990 TFLOPS
  optimal_time = matmul_flops / peak_flops * 1e6  # us
  a = jax.random.normal(jax.random.key(1), (shards * m_shard, k), dtype)
  b = jax.random.normal(jax.random.key(2), (k, shards * n_shard), dtype)
  a = jax.sharding.reshard(a, P("x", None))
  b = jax.sharding.reshard(b, P(None, "x"))
  _, ref_kernels_ms = profiler.measure(jax.jit(
      jax.shard_map(
          lambda x, y: lax.all_gather(x, "x", axis=0, tiled=True) @ y,
          out_specs=P(None, "x"),
          check_vma=False,
      )
  ), aggregate=False)(a, b)

  ref_time_us = _min_results_across_devices(ref_kernels_ms)
  ref_util = optimal_time / ref_time_us * 100

  tuning_it = itertools.product(
      (128, 256,),  # tile_m
      (64, 128),  # tile_n
      (64,),  # tile_k
      (4,),  # max_concurrent_steps
      (MatmulDimension.M, MatmulDimension.N),  # grid_minor_dim
      (4, 8, 16),  # grid_tile_width
      MatmulDimension,  # wg_dimension
  )
  best_util = 0.0
  best_runtime = float("inf")
  def build_kernel(**kwargs):
    return jax.jit(
        jax.shard_map(
            functools.partial(all_gather_lhs_matmul, **kwargs),
            out_specs=P(None, "x"),
            check_vma=False,
        )
    )

  for tile_m, tile_n, tile_k, max_concurrent_steps, grid_minor_dim, grid_tile_width, wg_dimension in tuning_it:
    try:
      config = TuningConfig(
          tile_m=tile_m,
          tile_n=tile_n,
          tile_k=tile_k,
          max_concurrent_steps=max_concurrent_steps,
          grid_minor_dim=grid_minor_dim,
          grid_tile_width=grid_tile_width,
          wg_dimension=wg_dimension,
      )
      _, kernels_ms = profiler.measure(
        build_kernel(axis_name="x", config=config, dtype=dtype),
        aggregate=False,
      )(a, b)
    except ValueError as e:
      if "exceeds available shared memory" in e.args[0]:  # Ignore SMEM OOMs.
        continue
      raise
    runtime_us = _min_results_across_devices(kernels_ms)
    achieved_tc_util = optimal_time / runtime_us * 100
    if achieved_tc_util > best_util:
      best_runtime = runtime_us
      best_util = achieved_tc_util
    print(
        f"{tile_m=} {tile_n=} {tile_k=} {max_concurrent_steps=} {grid_minor_dim=} {grid_tile_width=} {wg_dimension=}: "
        f"{runtime_us:<7.1f}us"
        f" = {achieved_tc_util:4.1f}% TC utilization"
    )
  print(f"\tBest: {best_runtime:<7.1f}us = {best_util:4.1f}% TC utilization")
  print(f"\tRef: {ref_time_us:<7.1f}us = {ref_util:4.1f}% TC utilization")


if __name__ == "__main__":
  if is_nvshmem_used():
    from jax._src import test_multiprocess as jt_multiprocess  # pytype: disable=import-error
    jt_multiprocess.main(shard_main=_run_example)
  else:
    from jax._src.config import config as jax_config
    from absl import app
    jax_config.config_with_absl()
    app.run(lambda _: _run_example())
