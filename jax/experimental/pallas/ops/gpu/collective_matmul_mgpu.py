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
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from . import hopper_matmul_mgpu


MatmulDimension = hopper_matmul_mgpu.MatmulDimension
TuningConfig = hopper_matmul_mgpu.TuningConfig


def all_gather_lhs_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    axis_name,
    *,
    config: hopper_matmul_mgpu.TuningConfig,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
  if (num_devices := jax.device_count()) != jax.process_count():
    raise ValueError("The kernel only supports one device per process")
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

  def kernel_body(lhs_local_ref, rhs_ref, out_ref, scratch_ref, out_smem, received_sem):
    wg_idx = lax.axis_index("wg")
    dev_id = lax.axis_index(axis_name)
    send_dev_id = lax.rem(dev_id + axis_size - 1, axis_size)
    send_scratch_ref = plgpu.remote_ref(
        scratch_ref, send_dev_id, device_id_type=pl.DeviceIdType.LOGICAL
    )

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
          lax.rem(device_offset + dev_id, num_devices) * m_shard, m_shard
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
          out_smem,
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

  def kernel_entry(*args):
    return pl.run_scoped(
        functools.partial(kernel_body, *args),
        received_sem=plgpu.SemaphoreType.REGULAR,
        collective_axes=("cluster_grid", "cluster", "wg"),
    )

  num_out_slots = min(2, (tile_m * tile_n) // (epi_tile_m * epi_tile_n))
  out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(dtype).itemsize * 8)
  out_swizzle_elems = out_swizzle // jnp.dtype(dtype).itemsize
  out_transforms = (
      plgpu.TilingTransform((8, out_swizzle_elems)),
      plgpu.SwizzleTransform(out_swizzle),
  )
  result, _ = plgpu.kernel(
      kernel_entry,
      out_shape=[
          # The output, with its M dimension all-gathered.
          jax.ShapeDtypeStruct((axis_size * m_shard, n_shard), dtype),
          # The scratch buffer used for the all-gather.
          jax.ShapeDtypeStruct((num_devices - 1, m_shard, k), dtype),
      ],
      scratch_shapes=[
          plgpu.SMEM(
              (2, num_out_slots, epi_tile_m, epi_tile_n),
              dtype,
              transforms=out_transforms,
          ),
      ],
      grid=(num_sms,),
      grid_names=("cluster_grid",),
      num_threads=3,
      thread_name="wg",
      cluster=(1,),
      cluster_names=("cluster",),
  )(lhs, rhs)
  return result
