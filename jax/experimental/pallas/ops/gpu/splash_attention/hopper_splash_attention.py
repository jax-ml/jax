# ruff: noqa: E731, E741
# Copyright 2026 The JAX Authors.
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
"""Splash attention implementation (using Mosaic GPU as the backend, based on FA3)."""

import dataclasses
import math
from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu.splash_attention import attention_mask
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax._src.lib import cuda_versions  # noqa: F401


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  block_q: int
  block_kv: int
  max_concurrent_steps: int
  use_schedule_barrier: bool = True

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_q_dq: int | None = None
  block_kv_dq: int | None = None

  def __post_init__(self):
    if self.block_q % 64:
      raise ValueError(f"{self.block_q=} must be a multiple of 64")
    if self.block_kv % 64:
      raise ValueError(f"{self.block_kv=} must be a multiple of 64")
    if self.max_concurrent_steps < 2:
      raise ValueError(f"{self.max_concurrent_steps=} must be at least 2")

    backward_blocks = [self.block_q_dkv,
                       self.block_kv_dkv, self.block_q_dq, self.block_kv_dq]
    block_is_set = [blk is not None for blk in backward_blocks]
    if any(block_is_set) and not all(block_is_set):
      raise ValueError(
          "Backward block sizes (block_q_dkv, block_kv_dkv, block_q_dq, "
          "block_kv_dq) must either all be specified or all be None."
      )

  @property
  def has_backward_blocks(self) -> bool:
    return self.block_q_dkv is not None


def _get_large_negative(dtype):
  dtype_max = dtypes.finfo(dtype).max
  return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def maybe_apply_mask(qk, num_nonzeros, block_mask_value, mask_smem_ref, barriers=None, wg_key="wg"):
  """Conditionally applies a mask to QK given the block mask value.

  Args:
      qk: The QK (logits) matrix to mask.
      num_nonzeros: The number of non-zero blocks encountered along the reduction dimension.
          Has shape of ``qk.shape[0]``.
      block_mask_value: The value indicating the type of block. If 0, the block is
          all masked. If 1, the block is partially masked (and ``mask_smem_ref`` will be applied).
          If 2, the block is unmasked.
      mask_smem_ref: The shared memory reference to the mask to apply for partially masked blocks.
      barriers: Optional list of barriers to wait on before loading the mask from shared memory.
      wg_key: The axis name for the warpgroup index.

  Returns:
      Masked QK matrix and updated ``num_nonzeros`` tensor.
  """
  large_negative = _get_large_negative(qk.dtype)

  if barriers:
    # TODO(justinjfu): Only issue and wait the mask TMA if `block_mask_value=1`
    for wg in range(len(barriers)):

      @pl.when(lax.axis_index(wg_key) == wg)
      def _() -> None:
        plgpu.barrier_wait(barriers[wg])

  def apply_smem_mask(qk, num_nonzeros):
    mask = plgpu.load(mask_smem_ref, (), layout=plgpu.Layout.WGMMA)
    assert mask.shape == qk.shape, f"{mask.shape=} != {qk.shape=}"
    return jnp.where(mask, qk, large_negative), num_nonzeros + mask.sum(axis=1)

  def _nonzero_mask(qk, num_nonzeros):
    return lax.cond(
        block_mask_value == attention_mask.MaskType.PARTIAL.value,
        lambda qk, num_nonzeros: apply_smem_mask(
            qk, num_nonzeros),  # Masked block
        lambda qk, num_nonzeros: (qk, num_nonzeros + 1),  # Full block
        qk,
        num_nonzeros,
    )

  qk, num_nonzeros = lax.cond(
      block_mask_value == attention_mask.MaskType.ZEROS.value,
      # Note: jnp.full is slower than jnp.minimum here
      lambda qk, num_nonzeros: (jnp.minimum(
          qk, large_negative), num_nonzeros),  # Zero block
      _nonzero_mask,
      qk,
      num_nonzeros,
  )
  return qk, num_nonzeros


def _attention_forward(
    q,
    k,
    v,
    mask_info: attention_mask.MaskInfo,
    config: TuningConfig,
    scale: float | None,
    save_residuals: bool = False,
):
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    raise ValueError(
        f"q, k, and v should all be 4D, got: {q.ndim=}, {k.ndim=}, {v.ndim=}")
  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape
  kv_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
  if k.shape != kv_shape:
    raise ValueError(f"Expected {k.shape=} to be {kv_shape} (inferred from q)")
  if v.shape != kv_shape:
    raise ValueError(f"Expected {v.shape=} to be {kv_shape} (inferred from q)")
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(
        f"q, k, and v should all have the same dtype, got: {q.dtype}, {k.dtype}, {v.dtype}")
  if num_q_heads % num_kv_heads:
    raise ValueError(
        f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if head_dim % 64:
    raise ValueError(f"{head_dim=} must be divisible by 64")
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(
        f"Only f16 and bf16 are supported, got dtype: {dtype}")
  if scale is None:
    scale = 1.0

  compute_wgs = 2
  max_concurrent_steps = min(
      config.max_concurrent_steps, kv_seq_len // config.block_kv)
  if max_concurrent_steps < 2:
    raise NotImplementedError(
        "Splash attention requires max_concurrent_steps > 2.")
  block_q, block_kv = config.block_q, config.block_kv
  if kv_seq_len % block_kv:
    raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")
  if mask_info and mask_info.q_block_size != config.block_q:
    raise ValueError(f"{mask_info.q_block_size=} must match {config.block_q=}")
  if mask_info and mask_info.kv_block_size != config.block_kv:
    raise ValueError(
        f"{mask_info.kv_block_size=} must match {config.block_kv=}")

  def kernel(
      q_ref,
      k_ref,
      v_ref,
      out_ref,
      num_nonzero_blocks_ref,
      block_mask_ref,
      mask_next_ref,
      data_next_ref,
      partial_mask_blocks_ref,
      lse_ref,
      scoped,
  ):
    batch = lax.axis_index("batch")
    q_head = lax.axis_index("heads")
    scratch_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
    wg_idx = lax.axis_index("wg")
    q_seq = lax.axis_index("q_seq")
    qo_smem2, k_smem, v_smem, lse_smem2, mask_smem = scratch_buffers
    k_barriers, v_barriers, q_barriers, mask_barriers = buffer_barriers
    k_consumed_barriers, v_consumed_barriers, mask_consumed_barriers = consumed_barriers

    def perform_schedule_barrier():
      plgpu.barrier_arrive(schedule_barrier)
      plgpu.barrier_wait(schedule_barrier)

    num_nonzero_blocks = num_nonzero_blocks_ref[batch, 0, q_seq]
    # Need at least max_concurrent_steps, or else we issue OOB TMAs in the memory wg prologue.
    block_max_kv_steps = jnp.maximum(num_nonzero_blocks, max_concurrent_steps)

    @pl.when(wg_idx < 2)
    def _compute_wg():
      plgpu.set_max_registers(232, action="increase")
      qo_smem = qo_smem2.at[wg_idx]
      lse_smem = lse_smem2.at[wg_idx] if lse_smem2 is not None else None
      q_seq_base = q_seq * (2 * block_q) + wg_idx * block_q
      kv_steps = block_max_kv_steps

      plgpu.copy_gmem_to_smem(
          q_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
          qo_smem,
          q_barriers.at[wg_idx],
      )
      plgpu.barrier_wait(q_barriers.at[wg_idx])

      @pl.when(kv_steps > 0)
      def _():
        plgpu.barrier_wait(k_barriers.at[0])

      pl.when(wg_idx == 1)(perform_schedule_barrier)

      def kv_loop(kv_step, carry):
        acc, m_i, l_i, num_nonzeros = carry
        slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))

        # QK
        def compute_qk(acc_ref):
          plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(
              k_smem.at[slot], (1, 0)))
          perform_schedule_barrier()
          return acc_ref[...]

        qk = pl.run_scoped(compute_qk, plgpu.ACC(
            (block_q, block_kv), jnp.float32))
        plgpu.barrier_arrive(k_consumed_barriers.at[slot])
        qk = qk * scale

        block_mask_value = block_mask_ref[batch,
                                          0, q_seq * 2 + wg_idx, kv_step]

        qk, num_nonzeros = maybe_apply_mask(
            qk,
            num_nonzeros,
            block_mask_value,
            mask_smem.at[slot, wg_idx],
            barriers=[barrier.at[slot] for barrier in mask_barriers],
        )

        for wg in range(compute_wgs):

          @pl.when(wg_idx == wg)
          def _():
            plgpu.barrier_arrive(mask_consumed_barriers[wg].at[slot])

        # Softmax
        # We keep m scaled by log2e to use FMA instructions when computing p.
        log2e = math.log2(math.e)
        m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)
        alpha = jnp.exp2(m_i - m_ij)
        m_i = m_ij
        p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
        acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
        l_i *= alpha
        p16 = p.astype(dtype)

        def end_softmax_barriers():
          plgpu.barrier_arrive(schedule_barrier)  # Done with softmax!
          plgpu.barrier_wait(v_barriers.at[slot])
          # Wait until TensorCore is free.
          plgpu.barrier_wait(schedule_barrier)

        # Can't fully explain why, but empirically the ordering here influences
        # the performance of the final kernel quite significantly.
        if head_dim <= 128:
          l_i += p.sum(axis=1)
          acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
          end_softmax_barriers()
        else:
          end_softmax_barriers()
          l_i += p.sum(axis=1)

        # PV
        def compute_pv(acc_ref):
          plgpu.wgmma(acc_ref, p16, v_smem.at[slot])

          wait_step = kv_step + 1
          wait_slot = lax.rem(wait_step, jnp.array(
              max_concurrent_steps, kv_step.dtype))

          @pl.when(wait_step < kv_steps)
          def _wait():
            plgpu.barrier_wait(k_barriers.at[wait_slot])

        acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
        plgpu.barrier_arrive(v_consumed_barriers.at[slot])
        return acc, m_i, l_i, num_nonzeros

      num_nonzeros = plgpu.layout_cast(
          jnp.full((block_q,), 0, dtype=jnp.int32),
          plgpu.Layout.WGMMA_ROW,
      )
      l_i = plgpu.layout_cast(
          jnp.full((block_q,), 0, dtype=jnp.float32),
          plgpu.Layout.WGMMA_ROW,
      )
      m_i = plgpu.layout_cast(
          jnp.full((block_q,), _get_large_negative(
              jnp.float32), dtype=jnp.float32),
          plgpu.Layout.WGMMA_ROW,
      )
      acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
          plgpu.Layout.WGMMA,
      )
      acc, m_i, l_i, num_nonzeros = lax.fori_loop(
          0, kv_steps, kv_loop, (acc, m_i, l_i, num_nonzeros))
      pl.when(wg_idx == 0)(perform_schedule_barrier)

      # TODO(apaszke): Invert and multiply to avoid expensive divisions.
      acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
      acc = jnp.where(lax.broadcast_in_dim(
          num_nonzeros > 0, acc.shape, [0]), acc, 0)
      qo_smem[...] = acc.astype(dtype)
      if lse_smem is not None:
        RCP_LN2 = 1.4426950408889634
        def log2(x): return jnp.log(x) * RCP_LN2
        lse_smem[...] = jnp.where(
            num_nonzeros > 0, m_i + log2(l_i), _get_large_negative(l_i.dtype))

      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          qo_smem,
          out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
      )
      if lse_smem is not None:
        plgpu.copy_smem_to_gmem(
            lse_smem,
            lse_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
        )
      plgpu.wait_smem_to_gmem(0)

    @pl.when(wg_idx == 2)
    def _memory_wg():
      plgpu.set_max_registers(40, action="decrease")
      kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
      for i in range(max_concurrent_steps):
        block_idx = data_next_ref[batch, 0, q_seq, i]
        block_idx = jnp.maximum(block_idx, 0)
        s = (batch, pl.ds(block_idx * block_kv, block_kv), kv_head)
        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[i], k_barriers.at[i])
        plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[i], v_barriers.at[i])
        for wg in range(compute_wgs):
          block_idx = mask_next_ref[batch, 0, q_seq * 2 + wg, i]
          # TODO(justinjfu): Only issue tma when block_idx >= 0
          block_idx = jnp.maximum(block_idx, 0)
          s = (batch, 0, block_idx, pl.ds(0, block_q), pl.ds(0, block_kv))
          plgpu.copy_gmem_to_smem(
              partial_mask_blocks_ref.at[s], mask_smem.at[i,
                                                          wg], mask_barriers[wg].at[i]
          )

      @pl.loop(0, block_max_kv_steps - max_concurrent_steps)
      def _kv_loop(kv_step):
        tma_step = kv_step + max_concurrent_steps
        tma_slot = lax.rem(kv_step, jnp.array(
            max_concurrent_steps, kv_step.dtype))
        block_idx = data_next_ref[batch, 0, q_seq, tma_step]
        block_idx = jnp.maximum(block_idx, 0)
        s = (batch, pl.ds(block_idx * block_kv, block_kv), kv_head)
        plgpu.barrier_wait(k_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(
            k_ref.at[s], k_smem.at[tma_slot], k_barriers.at[tma_slot])
        plgpu.barrier_wait(v_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(
            v_ref.at[s], v_smem.at[tma_slot], v_barriers.at[tma_slot])

        for wg in range(compute_wgs):
          block_idx = mask_next_ref[batch, 0, q_seq * 2 + wg, tma_step]
          plgpu.barrier_wait(mask_consumed_barriers[wg].at[tma_slot])
          # TODO(justinjfu): Only issue tma when block_idx >= 0
          block_idx = jnp.maximum(block_idx, 0)
          s = (batch, 0, block_idx, pl.ds(0, block_q), pl.ds(0, block_kv))
          plgpu.copy_gmem_to_smem(
              partial_mask_blocks_ref.at[s],
              mask_smem.at[tma_slot, wg],
              mask_barriers[wg].at[tma_slot],
          )

  def entry(
      q_ref,
      k_ref,
      v_ref,
      num_nozero_blocks_ref,
      block_mask_ref,
      mask_next_ref,
      data_next_ref,
      partial_mask_blocks_ref,
      out_ref,
      lse_ref,
  ):
    tiling = plgpu.TilingTransform((8, 64))
    swizzle = plgpu.SwizzleTransform(128)
    qo_scratch = plgpu.SMEM(
        (compute_wgs, block_q, head_dim),
        jnp.bfloat16,
        transforms=(tiling, swizzle),
    )
    k_scratch = plgpu.SMEM(
        (max_concurrent_steps, block_kv, head_dim),
        jnp.bfloat16,
        transforms=(tiling, swizzle),
    )
    v_scratch = plgpu.SMEM(
        (max_concurrent_steps, block_kv, head_dim),
        jnp.bfloat16,
        transforms=(tiling, swizzle),
    )
    scratch = [qo_scratch, k_scratch, v_scratch, None, None]
    if save_residuals:
      scratch[3] = plgpu.SMEM((compute_wgs, block_q), jnp.float32)
    scratch[4] = plgpu.SMEM(
        (max_concurrent_steps, compute_wgs, block_q, block_kv),
        jnp.int16,
        transforms=(tiling, swizzle),
    )

    barriers = [
        plgpu.Barrier(num_barriers=max_concurrent_steps),  # K barrier
        plgpu.Barrier(num_barriers=max_concurrent_steps),  # V barrier
        plgpu.Barrier(num_barriers=compute_wgs),  # Q barrier
        [plgpu.Barrier(num_barriers=max_concurrent_steps)] *
        compute_wgs  # Mask barriers
    ]
    consumed_barriers = [
        plgpu.Barrier(num_arrivals=compute_wgs,
                      num_barriers=max_concurrent_steps),  # K consumed
        plgpu.Barrier(num_arrivals=compute_wgs,
                      num_barriers=max_concurrent_steps),  # V consumed
        # Mask consumed barriers
        [plgpu.Barrier(
            num_arrivals=1, num_barriers=max_concurrent_steps)] * compute_wgs
    ]

    pl.run_scoped(
        lambda *args: kernel(
            q_ref,
            k_ref,
            v_ref,
            out_ref,
            num_nozero_blocks_ref,
            block_mask_ref,
            mask_next_ref,
            data_next_ref,
            partial_mask_blocks_ref,
            lse_ref,
            args,
        ),
        scratch,
        barriers,
        consumed_barriers,
        plgpu.Barrier(num_arrivals=compute_wgs),
        collective_axes="wg",
    )

  num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {block_q * 2=}")

  out_shape = [q, None]
  if save_residuals:
    # Note that we keep seq_len in the minor-most dimension so that we can do
    # 1D TMAs on chunks of `block_q`.
    out_shape[1] = jax.ShapeDtypeStruct(
        (batch_size, num_q_heads, q_seq_len), jnp.float32)

  out, lse = plgpu.kernel(
      entry,
      out_shape=out_shape,
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_seq", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(approx_math=True),
      kernel_name="mgpu_splash_attn_fwd",
  )(
      q,
      k,
      v,
      mask_info.num_nonzero_blocks,
      mask_info.block_mask,
      mask_info.mask_next.astype(jnp.int16),
      mask_info.data_next.astype(jnp.int16),
      mask_info.partial_mask_blocks.astype(jnp.int16),
  )

  if save_residuals:
    assert lse is not None
    return out, (lse,)

  return out


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
@partial(jax.jit, static_argnames=["config", "scale", "save_residuals"])
def attention(
    q,
    k,
    v,
    mask_info: attention_mask.MaskInfo,
    mask_info_dkv: attention_mask.MaskInfo | None = None,
    config: TuningConfig | None = None,
    scale: float | None = None,
    save_residuals: bool = False,
):
  del mask_info_dkv
  if config is None:
    config = TuningConfig(
        block_q=64,
        block_kv=128,
        max_concurrent_steps=2,
        block_q_dkv=64,
        block_kv_dkv=64,
        block_q_dq=64,
        block_kv_dq=64,
    )
  return _attention_forward(q, k, v, mask_info, config, scale, save_residuals)


def _attention_fwd(
    q,
    k,
    v,
    mask_info: attention_mask.MaskInfo,
    mask_info_dkv: attention_mask.MaskInfo | None,
    config: TuningConfig,
    scale: float | None,
    save_residuals: bool,
):
  del save_residuals
  out, (lse,) = _attention_forward(q, k, v, mask_info, config, scale, True)
  return out, (q, k, v, mask_info, mask_info_dkv, out, lse)


def _attention_bwd(config: TuningConfig, scale: float | None, save_residuals: bool, res, do):
  del save_residuals
  q, k, v, mask_info, mask_info_dkv, out, lse = res
  if mask_info_dkv is None:
    raise ValueError("Need to specify mask_info_dkv for backwards pass.")

  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")

  assert config.block_q_dq is not None
  assert config.block_kv_dq is not None
  assert config.block_q_dkv is not None
  assert config.block_kv_dkv is not None
  if scale is None:
    scale = 1.0

  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  dtype = q.dtype
  compute_wgs = 2
  if compute_wgs != 2:
    raise NotImplementedError(
        f"Only 2 compute wgs supported in backwards pass. Got {compute_wgs=}")

  num_q_tiles, rem = divmod(q_seq_len, config.block_q_dq * compute_wgs)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {config.block_q_dq=} * {compute_wgs=}")

  num_kv_tiles, rem = divmod(kv_seq_len, config.block_kv_dkv * compute_wgs)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len=} must be a multiple of {config.block_kv_dkv=} * {compute_wgs=}")

  num_q_tiles_in_dkv, rem = divmod(q_seq_len, config.block_q_dkv)
  if rem:
    raise NotImplementedError(
        f"{q_seq_len=} must be a multiple of {config.block_q_dkv=}")

  num_kv_tiles_in_dq, rem = divmod(kv_seq_len, config.block_kv_dq)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len=} must be a multiple of {config.block_kv_dq=}")

  tiling = plgpu.TilingTransform((8, 64))
  swizzle = plgpu.SwizzleTransform(128)

  delta = jnp.einsum(
      "bqhd,bqhd->bhq", out.astype(jnp.float32), do.astype(jnp.float32))
  del out  # Not needed anymore.
  dq_max_concurrent_steps = min([config.max_concurrent_steps, num_q_tiles])
  dkv_max_concurrent_steps = min([config.max_concurrent_steps, num_kv_tiles])
  if dq_max_concurrent_steps < 2:
    raise ValueError("Need at least 2 concurrent steps for dq kernel.")
  if dkv_max_concurrent_steps < 2:
    raise ValueError("Need at least 2 concurrent steps for dkv kernel.")
  if mask_info.q_block_size != config.block_q_dq:
    raise ValueError(
        f"{mask_info.q_block_size=} must match {config.block_q_dq=}")
  if mask_info.kv_block_size != config.block_kv_dq:
    raise ValueError(
        f"{mask_info.kv_block_size=} must match {config.block_kv_dq=}")
  if mask_info_dkv.q_block_size != config.block_q_dkv:
    raise ValueError(
        f"{mask_info_dkv.q_block_size=} must match {config.block_q_dkv=}")
  if mask_info_dkv.kv_block_size != config.block_kv_dkv:
    raise ValueError(
        f"{mask_info_dkv.kv_block_size=} must match {config.block_kv_dkv=}")

  def kernel_dq(
      q_ref,
      k_ref,
      v_ref,
      num_nonzero_blocks_ref,
      block_mask_ref,
      mask_next_ref,
      data_next_ref,
      partial_mask_blocks_ref,
      do_ref,
      lse_ref,
      delta_ref,
      dq_ref,
      smem_buffers,
      buffer_barriers,
      block_q,
      block_kv,
  ):
    batch = lax.axis_index("batch")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    q_seq = lax.axis_index("q_seq")
    kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
    q_smem2, do_smem2, lse_smem2, delta_smem2 = smem_buffers
    q_barriers, do_barriers, lse_barriers, delta_barriers = buffer_barriers

    num_nonzero_blocks = num_nonzero_blocks_ref[batch, 0, q_seq]
    block_max_kv_steps = jnp.maximum(
        num_nonzero_blocks, dq_max_concurrent_steps)

    def _compute_thread(pipeline_callback):
      q_smem, do_smem, lse_smem, delta_smem = (
          q_smem2.at[wg_idx],
          do_smem2.at[wg_idx],
          lse_smem2.at[wg_idx],
          delta_smem2.at[wg_idx],
      )
      q_seq_base = q_seq * (compute_wgs * block_q) + wg_idx * block_q
      q_slice = (batch, pl.ds(q_seq_base, block_q), q_head)
      plgpu.copy_gmem_to_smem(q_ref.at[q_slice], q_smem, q_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(
          do_ref.at[q_slice], do_smem, do_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(
          delta_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
          delta_smem,
          delta_barriers.at[wg_idx],
      )
      plgpu.copy_gmem_to_smem(
          lse_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
          lse_smem,
          lse_barriers.at[wg_idx],
      )
      for buffer in buffer_barriers:
        plgpu.barrier_wait(buffer.at[wg_idx])

      delta = plgpu.load(delta_smem, (), layout=plgpu.Layout.WGMMA_ROW)
      lse = plgpu.load(lse_smem, (), layout=plgpu.Layout.WGMMA_ROW)
      num_nonzeros = plgpu.layout_cast(
          jnp.full((block_q,), 0, dtype=jnp.int32),
          plgpu.Layout.WGMMA_ROW,
      )
      dq_acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim), 0, dtype=jnp.float32),
          plgpu.Layout.WGMMA,
      )
      dq, num_nonzeros, _, _ = pipeline_callback(
          (dq_acc, num_nonzeros, lse, delta))
      dq = jnp.where(lax.broadcast_in_dim(
          num_nonzeros > 0, dq.shape, [0]), dq, 0)
      q_smem[...] = dq.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[q_slice])
      plgpu.wait_smem_to_gmem(0)

    def kv_pipeline(
        indices, k_smem, v_smem, masks_smem, k_consumed_barrier, v_consumed_barrier, masks_consumed, carry
    ):
      (kv_step,) = indices
      q_smem, do_smem = q_smem2.at[wg_idx], do_smem2.at[wg_idx]
      (dq_acc, num_nonzeros, lse, delta) = carry

      def compute_qk(acc_ref):
        plgpu.wgmma(acc_ref, q_smem, plgpu.transpose_ref(k_smem, (1, 0)))
        return acc_ref[...]

      qk = pl.run_scoped(compute_qk, plgpu.ACC(
          (block_q, block_kv), jnp.float32))
      qk = qk * scale

      qk, num_nonzeros = lax.cond(
          wg_idx == 0,
          lambda qk, num_nonzeros: maybe_apply_mask(  # first compute wg
              qk, num_nonzeros, block_mask_ref[batch, 0,
                                               q_seq * compute_wgs, kv_step], masks_smem[0]
          ),
          lambda qk, num_nonzeros: maybe_apply_mask(  # second compute wg
              qk, num_nonzeros, block_mask_ref[batch, 0,
                                               q_seq * compute_wgs + 1, kv_step], masks_smem[1]
          ),
          qk,
          num_nonzeros,
      )
      for wg in range(compute_wgs):
        plgpu.barrier_arrive(masks_consumed[wg])
      p = jnp.exp2(qk * math.log2(math.e) -
                   lax.broadcast_in_dim(lse, (block_q, block_kv), [0]))

      # dP
      def compute_dp(acc_ref):
        plgpu.wgmma(acc_ref, do_smem, plgpu.transpose_ref(v_smem, (1, 0)))
        return acc_ref[...]

      dp = pl.run_scoped(compute_dp, plgpu.ACC(
          (block_q, block_kv), jnp.float32))
      plgpu.barrier_arrive(v_consumed_barrier)

      # dS
      ds = p * (dp - lax.broadcast_in_dim(delta, (block_q, block_kv), [0]))
      ds *= scale

      # dQ
      def compute_dq(acc_ref):
        plgpu.wgmma(acc_ref, ds.astype(k_ref.dtype), k_smem)

      dq_acc = pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))
      plgpu.barrier_arrive(k_consumed_barrier)
      return (dq_acc, num_nonzeros, lse, delta)

    k_spec = plgpu.BlockSpec(
        block_shape=(block_kv, head_dim),
        index_map=lambda i: (jnp.maximum(
            data_next_ref[batch, 0, q_seq, i], 0), 0),
        transforms=[tiling, swizzle],
    )
    v_spec = plgpu.BlockSpec(
        block_shape=(block_kv, head_dim),
        index_map=lambda i: (jnp.maximum(
            data_next_ref[batch, 0, q_seq, i], 0), 0),
        transforms=[tiling, swizzle],
    )
    mask_spec = [
        plgpu.BlockSpec(
            block_shape=(pl.Squeezed(), block_q, block_kv),
            index_map=lambda i: (jnp.maximum(
                mask_next_ref[batch, 0, q_seq * compute_wgs + 0, i], 0), 0, 0),
            transforms=[tiling, swizzle],
        ),
        plgpu.BlockSpec(
            block_shape=(pl.Squeezed(), block_q, block_kv),
            index_map=lambda i: (jnp.maximum(
                mask_next_ref[batch, 0, q_seq * compute_wgs + 1, i], 0), 0, 0),
            transforms=[tiling, swizzle],
        ),
    ]
    pipeline = plgpu.emit_pipeline_warp_specialized(
        kv_pipeline,
        grid=(block_max_kv_steps,),
        max_concurrent_steps=dq_max_concurrent_steps,
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=_compute_thread,
        in_specs=[k_spec, v_spec, mask_spec],
    )
    k_ref = k_ref.at[batch, :, kv_head, :]
    v_ref = v_ref.at[batch, :, kv_head, :]
    partial_mask_blocks_ref = partial_mask_blocks_ref.at[batch, 0, :, :, :]
    pipeline(k_ref, v_ref, [partial_mask_blocks_ref] * compute_wgs)

  def kernel_dkv(
      q_ref,
      k_ref,
      v_ref,
      num_nonzero_blocks_ref,
      block_mask_ref,
      mask_next_ref,
      data_next_ref,
      partial_mask_blocks_ref,
      do_ref,
      lse_ref,
      delta_ref,
      dk_ref,
      dv_ref,
      smem_buffers,
      buffer_barriers,
      block_q: int,
      block_kv: int,
  ):
    batch = lax.axis_index("batch")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    kv_seq = lax.axis_index("kv_seq")
    (k_smem2, v_smem2) = smem_buffers
    (k_barriers, v_barriers) = buffer_barriers

    block_max_q_steps = num_nonzero_blocks_ref[batch, 0, kv_seq]
    block_max_q_steps = jnp.maximum(
        block_max_q_steps, dkv_max_concurrent_steps)

    def _compute_thread(pipeline_callback):
      k_smem, v_smem = k_smem2.at[wg_idx], v_smem2.at[wg_idx]
      kv_seq_base = kv_seq * (compute_wgs * block_kv) + wg_idx * block_kv
      kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
      plgpu.copy_gmem_to_smem(
          k_ref.at[(batch, pl.ds(kv_seq_base, block_kv), kv_head)
                   ], k_smem, k_barriers.at[wg_idx]
      )
      plgpu.copy_gmem_to_smem(
          v_ref.at[(batch, pl.ds(kv_seq_base, block_kv), kv_head)
                   ], v_smem, v_barriers.at[wg_idx]
      )
      plgpu.barrier_wait(k_barriers.at[wg_idx])
      plgpu.barrier_wait(v_barriers.at[wg_idx])
      dk_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim), 0, dtype=jnp.float32),
          plgpu.Layout.WGMMA,
      )
      dv_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim), 0, dtype=jnp.float32),
          plgpu.Layout.WGMMA,
      )

      num_nonzeros = plgpu.layout_cast(
          jnp.full((block_kv,), 0, dtype=jnp.int32),
          plgpu.Layout.WGMMA_ROW,
      )
      (dk, dv, num_nonzeros) = pipeline_callback((dv_acc, dk_acc, num_nonzeros))
      dk = jnp.where(lax.broadcast_in_dim(
          num_nonzeros > 0, dk.shape, [0]), dk, 0)
      dv = jnp.where(lax.broadcast_in_dim(
          num_nonzeros > 0, dv.shape, [0]), dv, 0)
      k_smem[...] = dk.astype(dtype)
      v_smem[...] = dv.astype(dtype)

      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          k_smem, dk_ref.at[(batch, pl.ds(
              kv_seq_base, block_kv), q_head)], commit_group=False
      )
      plgpu.copy_smem_to_gmem(
          v_smem, dv_ref.at[(batch, pl.ds(
              kv_seq_base, block_kv), q_head)], commit_group=False
      )
      plgpu.commit_smem_to_gmem_group()
      plgpu.wait_smem_to_gmem(0)

    def q_pipeline(
        indices,
        q_smem,
        masks_smem,
        do_smem,
        lse_smem,
        delta_smem,
        q_consumed_barrier,
        masks_consumed,
        do_consumed_barrier,
        lse_consumed_barrier,
        delta_consumed_barrier,
        carry,
    ):
      (q_step,) = indices
      k_smem, v_smem = k_smem2.at[wg_idx], v_smem2.at[wg_idx]
      dk_acc, dv_acc, num_nonzeros = carry

      def _compute_qkT(acc_ref):
        plgpu.wgmma(acc_ref, k_smem, plgpu.transpose_ref(q_smem, (1, 0)))
        return acc_ref[...]

      qkT = pl.run_scoped(_compute_qkT, plgpu.ACC(
          (block_kv, block_q), jnp.float32))
      qkT, num_nonzeros = lax.cond(
          wg_idx == 0,
          lambda qk, num_nonzeros: maybe_apply_mask(  # first compute wg
              qk,
              num_nonzeros,
              block_mask_ref[batch, 0, num_q_tiles_in_dkv -
                             1 - q_step, kv_seq * compute_wgs],
              masks_smem[0],
          ),
          lambda qk, num_nonzeros: maybe_apply_mask(  # second compute wg
              qk,
              num_nonzeros,
              block_mask_ref[batch, 0, num_q_tiles_in_dkv -
                             1 - q_step, kv_seq * compute_wgs + 1],
              masks_smem[1],
          ),
          qkT,
          num_nonzeros,
      )
      for wg in range(compute_wgs):
        plgpu.barrier_arrive(masks_consumed[wg])

      lse = plgpu.load(lse_smem, (), layout=plgpu.Layout.WGMMA_COL)
      plgpu.barrier_arrive(lse_consumed_barrier)
      pT = jnp.exp2(qkT * math.log2(math.e) * scale -
                    lax.broadcast_in_dim(lse, (block_kv, block_q), [1]))

      def _compute(refs):
        # Combining two WGMMA calls in one block to avoid the unnecessary
        # synchronization from two `wgmma.wait_group` calls.
        dv_acc_ref, dpT_acc_ref = refs
        plgpu.wgmma(dv_acc_ref, pT.astype(dtype), do_smem)  # dV
        plgpu.wgmma(dpT_acc_ref, v_smem,
                    plgpu.transpose_ref(do_smem, (1, 0)))  # dpT

      zeros = plgpu.layout_cast(
          jnp.full((block_kv, block_q), 0, dtype=jnp.float32),
          plgpu.Layout.WGMMA,
      )
      dv_acc, dpT = pl.run_state(_compute)(
          (plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
      plgpu.barrier_arrive(do_consumed_barrier)

      delta = plgpu.load(delta_smem, (), layout=plgpu.Layout.WGMMA_COL)
      plgpu.barrier_arrive(delta_consumed_barrier)

      dqkT = pT * (
          dpT - lax.broadcast_in_dim(delta, (block_kv, block_q), [1])
      )  # pytype: disable=wrong-arg-types  # jax-operator-types
      dqkT *= scale

      def compute_dk(acc_ref):
        plgpu.wgmma(acc_ref, dqkT.astype(dtype), q_smem)

      dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
      plgpu.barrier_arrive(q_consumed_barrier)

      return (dk_acc, dv_acc, num_nonzeros)

    q_spec = plgpu.BlockSpec(
        block_shape=(block_kv, head_dim),
        index_map=lambda i: (jnp.maximum(
            data_next_ref[batch, 0, num_q_tiles_in_dkv - 1 - i, kv_seq], 0), 0),
        transforms=[tiling, swizzle],
    )
    do_spec = plgpu.BlockSpec(
        block_shape=(block_kv, head_dim),
        index_map=lambda i: (jnp.maximum(
            data_next_ref[batch, 0, num_q_tiles_in_dkv - 1 - i, kv_seq], 0), 0),
        transforms=[tiling, swizzle],
    )
    lse_spec = plgpu.BlockSpec(
        block_shape=(block_kv,),
        index_map=lambda i: (jnp.maximum(
            data_next_ref[batch, 0, num_q_tiles_in_dkv - 1 - i, kv_seq], 0),),
    )
    delta_spec = plgpu.BlockSpec(
        block_shape=(block_kv,),
        index_map=lambda i: (jnp.maximum(
            data_next_ref[batch, 0, num_q_tiles_in_dkv - 1 - i, kv_seq], 0),),
    )
    mask_spec = [
        plgpu.BlockSpec(
            block_shape=(pl.Squeezed(), block_kv, block_q),
            index_map=lambda i: (
                jnp.maximum(
                    mask_next_ref[batch, 0, num_q_tiles_in_dkv - 1 - i, kv_seq * compute_wgs + 0], 0),
                0,
                0,
            ),
            transforms=[tiling, swizzle],
        ),
        plgpu.BlockSpec(
            block_shape=(pl.Squeezed(), block_kv, block_q),
            index_map=lambda i: (
                jnp.maximum(
                    mask_next_ref[batch, 0, num_q_tiles_in_dkv - 1 - i, kv_seq * compute_wgs + 1], 0),
                0,
                0,
            ),
            transforms=[tiling, swizzle],
        ),
    ]
    pipeline = plgpu.emit_pipeline_warp_specialized(
        q_pipeline,
        grid=(block_max_q_steps,),
        max_concurrent_steps=dkv_max_concurrent_steps,
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=_compute_thread,
        in_specs=[
            q_spec,
            mask_spec,
            do_spec,
            lse_spec,
            delta_spec,
        ],
    )
    q_ref = q_ref.at[batch, :, q_head, :]
    do_ref = do_ref.at[batch, :, q_head, :]
    lse_ref = lse_ref.at[batch, q_head, :]
    delta_ref = delta_ref.at[batch, q_head, :]
    partial_mask_blocks_ref = partial_mask_blocks_ref.at[batch, 0, :, :, :]
    pipeline(q_ref, [partial_mask_blocks_ref] *
             compute_wgs, do_ref, lse_ref, delta_ref)

  q_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq, head_dim),
      jnp.bfloat16,
      transforms=(tiling, swizzle),
  )
  do_scratch = q_scratch
  lse_scratch = plgpu.SMEM((compute_wgs, config.block_q_dq), jnp.float32)
  delta_scratch = plgpu.SMEM((compute_wgs, config.block_q_dq), jnp.float32)
  dq = plgpu.kernel(
      partial(kernel_dq, block_q=config.block_q_dq,
              block_kv=config.block_kv_dq),
      out_shape=q,
      scratch_shapes=[
          (q_scratch, do_scratch, lse_scratch, delta_scratch),  # type: ignore
          (plgpu.Barrier(num_barriers=compute_wgs),) * 4,  # type: ignore
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_seq", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
      kernel_name="mgpu_splash_attn_bwd_dq",
  )(
      q,
      k,
      v,
      mask_info.num_nonzero_blocks,
      mask_info.block_mask,
      mask_info.mask_next.astype(jnp.int16),
      mask_info.data_next.astype(jnp.int16),
      mask_info.partial_mask_blocks.astype(jnp.int16),
      do,
      lse,
      delta,
  )

  k_scratch = plgpu.SMEM(
      (compute_wgs, config.block_kv_dkv, head_dim),
      jnp.bfloat16,
      transforms=(tiling, swizzle),
  )
  v_scratch = k_scratch
  out_shape_kv = jax.ShapeDtypeStruct(
      (batch_size, kv_seq_len, num_q_heads, head_dim), dtype=jnp.bfloat16)
  dk, dv = plgpu.kernel(
      partial(kernel_dkv, block_q=config.block_q_dkv,
              block_kv=config.block_kv_dkv),
      out_shape=[out_shape_kv, out_shape_kv],
      scratch_shapes=[
          (k_scratch, v_scratch),  # type: ignore
          (plgpu.Barrier(num_barriers=compute_wgs),) * 2,  # type: ignore
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, num_kv_tiles, batch_size),
      grid_names=("heads", "kv_seq", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
      kernel_name="mgpu_splash_attn_bwd_dkv",
  )(
      q,
      k,
      v,
      mask_info_dkv.num_nonzero_blocks,
      mask_info_dkv.block_mask,
      mask_info_dkv.mask_next.astype(jnp.int16),
      mask_info_dkv.data_next.astype(jnp.int16),
      mask_info_dkv.partial_mask_blocks.astype(jnp.int16),
      do,
      lse,
      delta,
  )

  if q_heads_per_kv_head > 1:
    sum_shape = (*k.shape[:-1], q_heads_per_kv_head, head_dim)
    dk = dk.reshape(sum_shape).astype(
        jnp.float32).sum(axis=-2).astype(dk.dtype)
    dv = dv.reshape(sum_shape).astype(
        jnp.float32).sum(axis=-2).astype(dv.dtype)

  return dq, dk, dv, None, None


attention.defvjp(_attention_fwd, _attention_bwd)
