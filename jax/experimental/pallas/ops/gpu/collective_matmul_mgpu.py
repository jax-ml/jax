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


# TODO(apaszke): Add grid tiling
def all_gather_lhs_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    axis_name,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    sm_n_tile: int,
    max_concurrent_steps: int,
    dtype: jnp.dtype = jnp.float16,
) -> jax.Array:
  if (num_devices := jax.device_count()) != jax.process_count():
    raise ValueError("The kernel only supports one device per process")
  if (axis_size := lax.axis_size(axis_name)) != num_devices:
    raise ValueError("The kernel can only work over all devices in a Mesh.")
  if max_concurrent_steps < 2:
    raise ValueError("max_concurrent_steps must be >= 2")
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
  if k % block_k != 0:
    raise NotImplementedError(f"{k=} must be a multiple of {block_k=}")
  if m_shard % block_m != 0:
    raise NotImplementedError(f"{m_shard=} must be a multiple of {block_m=}")
  if n_shard % block_n != 0:
    raise NotImplementedError(f"{n_shard=} must be a multiple of {block_n=}")

  # max_num_sms is 132 for H100 SXM GPUs.
  if (max_num_sms := jax.devices()[0].core_count) % sm_n_tile != 0:
    raise ValueError(f"{max_num_sms=} must be divisible by {sm_n_tile=}.")
  if n_shard % sm_n_tile != 0:
    raise NotImplementedError(f"{n_shard=} must be divisible by {sm_n_tile=}")
  if (n_shard_per_sm_n := n_shard // sm_n_tile) % block_n != 0:
    raise NotImplementedError(
        f"{n_shard_per_sm_n=} must be divisible by {block_n=}"
    )
  num_sms_m = max_num_sms // sm_n_tile

  swizzle = min(
      plgpu.find_swizzle(block_k * jnp.finfo(element_type).bits, "lhs"),
      plgpu.find_swizzle(block_n * jnp.finfo(element_type).bits, "rhs"),
  )
  transforms = (
      plgpu.TilingTransform((8, swizzle // jnp.dtype(element_type).itemsize)),
      plgpu.SwizzleTransform(swizzle),
  )

  def kernel_body(lhs_ref, rhs_ref, out_ref, scratch_ref, out_smem, received_sem):
    sm_m = lax.axis_index('sm_m')
    sm_n = lax.axis_index('sm_n')
    n_start = sm_n * n_shard_per_sm_n
    scratch_ref = scratch_ref.at[sm_m]

    dev_id = lax.axis_index(axis_name)
    send_dev_id = lax.rem(dev_id + axis_size - 1, axis_size)
    send_scratch_ref = plgpu.remote_ref(
        scratch_ref, send_dev_id, device_id_type=pl.DeviceIdType.LOGICAL
    )

    @plgpu.nd_loop((m_shard // block_m,), collective_axes="sm_m")
    def _m_loop(idx):
      (mi,) = idx
      m_tile_slice = pl.ds(mi * block_m, block_m)

      # For some reason ptxas spills if we unroll the loop over k
      copy_block = 32
      @pl.loop(0, k, step=copy_block)
      def _k_copy_loop(ki):
        k_slice = pl.ds(ki, copy_block)
        scratch_ref[0, :, k_slice] = lhs_ref[m_tile_slice, k_slice]

      @pl.loop(0, num_devices)
      def _device_loop(device_offset):
        device_m_slice = pl.ds(
            lax.rem(device_offset + dev_id, num_devices) * m_shard, block_m
        )

        scratch_slot = device_offset
        next_scratch_slot = scratch_slot + 1

        def compute(n_tile_slice, send: bool):
          @functools.partial(
              pl.run_scoped, acc_ref=plgpu.ACC((block_m, block_n))
          )
          def _(acc_ref):
            @functools.partial(
                plgpu.emit_pipeline,
                grid=(k // block_k,),
                in_specs=[
                    plgpu.BlockSpec(
                        (block_m, block_k),
                        lambda k: (0, k),
                        transforms=transforms,
                        delay_release=1,
                    ),
                    plgpu.BlockSpec(
                        (block_k, block_n),
                        lambda k: (k, 0),
                        transforms=transforms,
                        delay_release=1,
                    ),
                ],
                max_concurrent_steps=max_concurrent_steps,
            )
            def k_loop(idxs, lhs_smem, rhs_smem):
              plgpu.wgmma(acc_ref, lhs_smem, rhs_smem)
              if send:
                # TODO(giorgioa): Send only for first sm_n.
                @pl.when(next_scratch_slot <= num_devices - 1)
                def _():
                  (ki,) = idxs
                  k_slice = pl.ds(ki * block_k, block_k)
                  plgpu.copy_smem_to_gmem(
                      lhs_smem, send_scratch_ref.at[next_scratch_slot, :, k_slice]
                  )
                  # We only delay release by 1 step, so we need to wait for the
                  # previous copies.
                  plgpu.wait_smem_to_gmem(1, wait_read_only=True)
            k_loop(scratch_ref.at[scratch_slot], rhs_ref.at[..., n_tile_slice])
            if send:
              # Make sure the copy is done and signal the receiving device.
              plgpu.wait_smem_to_gmem(0, wait_read_only=False)
              pl.semaphore_signal(received_sem, device_id=send_dev_id)
            # Make sure all TMAs have read SMEM before we overwrite it.
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            out_smem[...] = acc_ref[...].astype(out_smem.dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(
                out_smem,
                out_ref.at[device_m_slice, n_tile_slice].at[m_tile_slice],
            )

        compute(pl.ds(n_start, block_n), send=True)

        @pl.loop(1, n_shard_per_sm_n // block_n)
        def _n_loop(ni):
          compute(pl.ds(n_start + ni * block_n, block_n), send=False)

        # Wait for the next scratch to arrive --- see the device loop invariant.
        pl.semaphore_wait(received_sem)

    # Make sure all copies are fully done.
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  result, _ = plgpu.kernel(
      kernel_body,
      out_shape=[
          # The output, with its M dimension all-gathered.
          jax.ShapeDtypeStruct((axis_size * m_shard, n_shard), dtype),
          # The scratch buffer used for the all-gather.
          jax.ShapeDtypeStruct((num_sms_m, num_devices, block_m, k), dtype),
      ],
      scratch_shapes=[
          plgpu.SMEM((block_m, block_n), dtype, transforms=transforms),
          plgpu.SemaphoreType.REGULAR,  # Received semaphore
      ],
      grid=(num_sms_m, sm_n_tile),
      grid_names=('sm_m', 'sm_n'),
  )(lhs, rhs)
  return result
