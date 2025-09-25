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

import dataclasses
import enum
import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp


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
  wg_dimension: MatmulDimension = MatmulDimension.N


def all_gather_lhs_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    axis_name,
    *,
    config: TuningConfig,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
  if (num_devices := jax.device_count()) != jax.process_count():
    raise ValueError("The kernel only supports one device per process")
  if (axis_size := lax.axis_size(axis_name)) != num_devices:
    raise ValueError("The kernel can only work over all devices in a Mesh.")
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(f"Only f16 and bf16 are supported, got {dtype=}")

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
  cta_tile_n = tile_n * (1 + (config.wg_dimension == MatmulDimension.N))
  if k % tile_k != 0:
    raise NotImplementedError(f"{k=} must be a multiple of {tile_k=}")
  if m_shard % cta_tile_m != 0:
    raise NotImplementedError(f"{m_shard=} must be a multiple of {cta_tile_m=}")
  if n_shard % cta_tile_n != 0:
    raise NotImplementedError(f"{n_shard=} must be a multiple of {cta_tile_n=}")
  m_iters = m_shard // cta_tile_m
  n_iters = n_shard // cta_tile_n
  k_iters = k // tile_k

  epi_tile_n = config.epi_tile_n or tile_n
  epi_tile_m = config.epi_tile_m or tile_m
  if tile_n % epi_tile_n != 0:
    raise ValueError(f"{tile_n=} must be divisible by {epi_tile_n=}")
  if tile_m % epi_tile_m != 0:
    raise ValueError(f"{tile_m=} must be divisible by {epi_tile_m=}")

  swizzle = plgpu.find_swizzle(tile_k * jnp.finfo(element_type).bits, "lhs")
  transforms = (
      plgpu.TilingTransform((8, swizzle // jnp.dtype(element_type).itemsize)),
      plgpu.SwizzleTransform(swizzle),
  )

  num_sms = jax.devices()[0].core_count  # 132 for H100 SXM GPUs.

  def do_matmul(lhs_ref, rhs_ref, out_gmem, out_smem, send_ref, should_send):
    """The inner loop of the matmul kernel.

    This is a very slightly modified version of the plain MGPU matmul kernel.
    The only difference is the async copy in the pipeline step.
    """
    wg_idx = lax.axis_index("wg")

    def get_pipeline(pipeline_body, compute_context):
      return plgpu.emit_pipeline_warp_specialized(
          pipeline_body,
          grid=(k_iters,),
          memory_registers=40,
          in_specs=[
              plgpu.BlockSpec(
                  (cta_tile_m, tile_k),
                  lambda k: (0, k),
                  transforms=transforms,
                  memory_space=plgpu.SMEM,
                  delay_release=1,
              ),
              plgpu.BlockSpec(
                  (tile_k, cta_tile_n),
                  lambda k: (k, 0),
                  transforms=transforms,
                  memory_space=plgpu.SMEM,
                  delay_release=1,
              ),
          ],
          wg_axis="wg",
          num_compute_wgs=2,
          max_concurrent_steps=max_concurrent_steps,
          compute_context=compute_context,
      )

    # Functions don't influence the allocations necessary to run the pipeline.
    ignore = lambda *_, **__: None
    @functools.partial(
        pl.run_scoped,
        pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(lhs_ref, rhs_ref),
        collective_axes="wg",
    )
    def _pipeline_scope(pipeline_allocs):
      @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
      def _mn_loop(loop_info: plgpu.NDLoopInfo):
        (lin_idx,) = loop_info.index
        m_idx, n_idx = plgpu.planar_snake(
            lin_idx,
            (m_iters, n_iters),
            config.grid_minor_dim,
            config.grid_tile_width,
        )
        cta_m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
        cta_n_slice = pl.ds(n_idx * cta_tile_n, cta_tile_n)
        if config.wg_dimension == MatmulDimension.M:
          wg_m_slice = pl.ds(wg_idx * tile_m, tile_m)
          wg_n_slice = slice(None)
        else:
          wg_m_slice = slice(None)
          wg_n_slice = pl.ds(wg_idx * tile_n, tile_n)  # type: ignore

        def compute_context(eval_pipeline):
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
                    out_gmem.at[cta_m_slice, cta_n_slice]
                    .at[wg_m_slice, wg_n_slice]
                    .at[epi_m_slice, epi_n_slice],
                )

        def mma_body(mma_idxs, a_smem, b_smem, acc_ref):
          (ki,) = mma_idxs
          plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem.at[:, wg_n_slice])
          # We only send when n_idx == 0 to avoid sending the same data
          # multiple times when revisiting lhs.
          @pl.when(should_send & jnp.bool(n_idx == 0))
          def _():
            k_slice = pl.ds(ki * tile_k, tile_k)
            m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
            plgpu.copy_smem_to_gmem(a_smem, send_ref.at[m_slice, k_slice])
            # We only delay release by 1 step, so we need to wait for the
            # previous copies.
            plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          plgpu.wgmma_wait(1)
          return acc_ref

        get_pipeline(mma_body, compute_context)(
            lhs_ref.at[cta_m_slice, :],
            rhs_ref.at[:, cta_n_slice],
            allocations=pipeline_allocs,
        )

  def kernel_body(lhs_local_ref, rhs_ref, out_ref, scratch_ref, out_smem, received_sem):
    wg_idx = lax.axis_index("wg")
    dev_id = lax.axis_index(axis_name)
    send_dev_id = lax.rem(dev_id + axis_size - 1, axis_size)
    send_scratch_ref = plgpu.remote_ref(
        scratch_ref, send_dev_id, device_id_type=pl.DeviceIdType.LOGICAL
    )

    def device_step(lhs_source_ref, device_offset):
      # Invariant: lhs_source_ref is ready to be used
      next_scratch_slot = device_offset
      out_device_m_slice = pl.ds(
          lax.rem(device_offset + dev_id, num_devices) * m_shard, m_shard
      )
      is_send_wg = wg_idx == 0
      has_send_space = next_scratch_slot < num_devices - 1
      should_send = is_send_wg & has_send_space

      do_matmul(
          lhs_source_ref,  # Use the lhs from previous step.
          rhs_ref,  # Use the same rhs for all steps.
          out_ref.at[out_device_m_slice],  # Use a slice of the output.
          out_smem,
          send_scratch_ref.at[next_scratch_slot],
          should_send,
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
        collective_axes=("sm", "wg"),
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
      grid_names=("sm",),
      num_threads=3,
      thread_name="wg",
  )(lhs, rhs)
  return result
