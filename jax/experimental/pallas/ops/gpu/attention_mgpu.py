# Copyright 2024 The JAX Authors.
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
"""FlashAttention3 implementation (using Mosaic GPU as the backend)."""

import dataclasses
import functools
import itertools
import math
import jax
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax._src.lib import cuda_versions  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np
from functools import partial

@dataclasses.dataclass(frozen=True)
class TuningConfig:
  block_q: int
  block_kv: int
  max_concurrent_steps: int
  use_schedule_barrier: bool = True
  causal: bool = False
  compute_wgs_bwd: int = 1

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

    backward_blocks = [self.block_q_dkv, self.block_kv_dkv, self.block_q_dq, self.block_kv_dq]
    block_is_set = [blk is not None for blk in backward_blocks]
    if any(block_is_set) and not all(block_is_set):
      raise ValueError(
          "Backward block sizes (block_q_dkv, block_kv_dkv, block_q_dq, "
          "block_kv_dq) must either all be specified or all be None."
      )

  @property
  def has_backward_blocks(self) -> bool:
    return self.block_q_dkv is not None

def _attention_forward(q, k, v, config: TuningConfig, save_residuals: bool = False):
  assert cuda_versions is not None
  cuda_runtime_version = cuda_versions.cuda_runtime_get_version()
  # TODO(pobudzey): Undo when we upgrade to cuda 12.9.1.
  if config.causal and cuda_runtime_version >= 12080 and cuda_runtime_version < 12091:
    raise ValueError(
        "Causal masking not supported with cuda versions between 12.8.0 and"
        " 12.9.1 due to a ptxas miscompilation."
    )
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    raise ValueError(f"q, k, and v should all be 4D, got: {q.ndim=}, {k.ndim=}, {v.ndim=}")
  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape
  kv_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
  if k.shape != kv_shape:
    raise ValueError(f"Expected {k.shape=} to be {kv_shape} (inferred from q)")
  if k.shape != kv_shape:
    raise ValueError(f"Expected {v.shape=} to be {kv_shape} (inferred from q)")
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(f"q, k, and v should all have the same dtype, got: {q.dtype}, {k.dtype}, {v.dtype}")
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if head_dim % 64:
    raise ValueError(f"{head_dim=} must be divisible by 64")
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(f"Only f16 and bf16 are supported, got dtype: {dtype}")

  max_concurrent_steps = min(
      config.max_concurrent_steps, kv_seq_len // config.block_kv
  )
  block_q, block_kv = config.block_q, config.block_kv
  if kv_seq_len % block_kv:
    raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")

  def kernel(q_ref, k_ref, v_ref, out_ref, lse_ref, scoped):
    batch = lax.axis_index("batch")
    q_head = lax.axis_index("heads")
    smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
    wg_idx = lax.axis_index("wg")
    qo_smem2, k_smem, v_smem, lse_smem2 = smem_buffers
    k_barriers, v_barriers, q_barriers = buffer_barriers
    k_consumed_barriers, v_consumed_barriers = consumed_barriers
    def perform_schedule_barrier():
      plgpu.barrier_arrive(schedule_barrier)
      plgpu.barrier_wait(schedule_barrier)

    if config.causal:
      block_q_end = (lax.axis_index("q_seq") + 1) * (2 * block_q)
      block_max_kv_steps = pl.cdiv(block_q_end, jnp.array(block_kv, jnp.int32))
    else:
      block_max_kv_steps = kv_seq_len // block_kv

    @pl.when(wg_idx < 2)
    def _compute_wg():
      plgpu.set_max_registers(232, action="increase")
      qo_smem = qo_smem2.at[wg_idx]
      lse_smem = lse_smem2.at[wg_idx] if lse_smem2 is not None else None
      q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q

      if config.causal:
        kv_steps = pl.cdiv(q_seq_base + block_q, jnp.array(block_kv, jnp.int32))
      else:
        kv_steps = block_max_kv_steps

      plgpu.copy_gmem_to_smem(
          q_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
          qo_smem,
          q_barriers.at[wg_idx],
      )
      plgpu.barrier_wait(q_barriers.at[wg_idx])

      m_i = plgpu.layout_cast(
          jnp.full((block_q,), -jnp.inf, dtype=jnp.float32), plgpu.Layout.WGMMA_ROW,
      )
      l_i = plgpu.layout_cast(
          jnp.full((block_q,), 0, dtype=jnp.float32), plgpu.Layout.WGMMA_ROW,
      )
      acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim), 0, dtype=jnp.float32), plgpu.Layout.WGMMA,
      )

      @pl.when(kv_steps > 0)
      def _():
        plgpu.barrier_wait(k_barriers.at[0])

      pl.when(wg_idx == 1)(perform_schedule_barrier)
      def kv_loop(kv_step, carry, causal: bool = False):
        acc, m_i, l_i = carry
        slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))

        # QK
        def compute_qk(acc_ref):
          plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
          perform_schedule_barrier()
          return acc_ref[...]
        qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
        plgpu.barrier_arrive(k_consumed_barriers.at[slot])

        if causal:
          q_ids = plgpu.broadcasted_iota(jnp.int32, (block_q, block_kv), 0, layout=plgpu.Layout.WGMMA)
          kv_ids = plgpu.broadcasted_iota(jnp.int32, (block_q, block_kv), 1, layout=plgpu.Layout.WGMMA)
          mask = (q_ids + q_seq_base) >= (kv_ids + kv_step * block_kv)
          qk = jnp.where(mask, qk, -jnp.inf)

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
          plgpu.barrier_wait(schedule_barrier)  # Wait until TensorCore is free.
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
          wait_slot = lax.rem(wait_step, jnp.array(max_concurrent_steps, kv_step.dtype))
          @pl.when(wait_step < kv_steps)
          def _wait():
            plgpu.barrier_wait(k_barriers.at[wait_slot])
        acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
        plgpu.barrier_arrive(v_consumed_barriers.at[slot])
        return acc, m_i, l_i

      if not config.causal:
        acc, m_i, l_i = lax.fori_loop(0, block_max_kv_steps, kv_loop, (acc, m_i, l_i))
      else:
        def epilogue_kv_loop(kv_step, _):
          # This loop makes sure that all the pipelined KV data is processed, even
          # if one compute wg finishes early like with causal masking.
          slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
          plgpu.barrier_arrive(k_consumed_barriers.at[slot])
          plgpu.barrier_arrive(v_consumed_barriers.at[slot])
          perform_schedule_barrier()
          perform_schedule_barrier()

        causal_kv_loop = functools.partial(kv_loop, causal=True)
        full_kv_steps = lax.div(q_seq_base, jnp.array(block_kv, jnp.int32))
        # With causal masking, the KV loop unrolling is split in 3 sections:
        # 1. A fast path where no causal mask is needed.
        acc, m_i, l_i = lax.fori_loop(0, full_kv_steps, kv_loop, (acc, m_i, l_i))
        # 2. Causal masking.
        acc, m_i, l_i = lax.fori_loop(full_kv_steps, kv_steps, causal_kv_loop, (acc, m_i, l_i))
        # 3. Epilogue to flush the data pipeline.
        lax.fori_loop(kv_steps, block_max_kv_steps, epilogue_kv_loop, None)
      pl.when(wg_idx == 0)(perform_schedule_barrier)

      # TODO(apaszke): Invert and multiply to avoid expensive divisions.
      acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
      qo_smem[...] = acc.astype(dtype)
      if lse_smem is not None:
        RCP_LN2 = 1.4426950408889634
        log2 = lambda x: jnp.log(x) * RCP_LN2
        lse_smem[...] = m_i + log2(l_i)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          qo_smem, out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
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
        s = (batch, pl.ds(i * block_kv, block_kv), kv_head)
        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[i], k_barriers.at[i])
        plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[i], v_barriers.at[i])

      @pl.loop(0, block_max_kv_steps - max_concurrent_steps)
      def _kv_loop(kv_step):
        tma_step = kv_step + max_concurrent_steps
        tma_slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
        s = (batch, pl.ds(tma_step * block_kv, block_kv), kv_head)
        plgpu.barrier_wait(k_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[tma_slot], k_barriers.at[tma_slot])
        plgpu.barrier_wait(v_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[tma_slot], v_barriers.at[tma_slot])

  def entry(q_ref, k_ref, v_ref, out_ref, lse_ref):
    compute_wgs = 2
    tiling = plgpu.TilingTransform((8, 64))
    swizzle = plgpu.SwizzleTransform(128)
    qo_scratch = plgpu.SMEM(
        (compute_wgs, block_q, head_dim), jnp.float16,
        transforms=(tiling, swizzle),
    )
    k_scratch = plgpu.SMEM(
        (max_concurrent_steps, block_kv, head_dim), jnp.float16,
        transforms=(tiling, swizzle),
    )
    v_scratch = plgpu.SMEM(
        (max_concurrent_steps, block_kv, head_dim), jnp.float16,
        transforms=(tiling, swizzle),
    )
    scratch = [qo_scratch, k_scratch, v_scratch, None]
    if save_residuals:
      scratch[3] = plgpu.SMEM((compute_wgs, block_q), jnp.float32)
    pl.run_scoped(
        lambda *args: kernel(q_ref, k_ref, v_ref, out_ref, lse_ref, args),
        scratch,
        (
            plgpu.Barrier(num_barriers=max_concurrent_steps),
            plgpu.Barrier(num_barriers=max_concurrent_steps),
            plgpu.Barrier(num_barriers=compute_wgs),
        ),
        (plgpu.Barrier(num_arrivals=compute_wgs, num_barriers=max_concurrent_steps),) * 2,
        plgpu.Barrier(num_arrivals=compute_wgs),
        collective_axes="wg",
    )

  num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
  if rem:
    raise NotImplementedError(f"{q_seq_len=} must be a multiple of {block_q * 2=}")

  out_shape = [q, None]
  if save_residuals:
    # Note that we keep seq_len in the minor-most dimension so that we can do
    # 1D TMAs on chunks of `block_q`.
    out_shape[1] = jax.ShapeDtypeStruct(
        (batch_size, num_q_heads, q_seq_len), jnp.float32
    )

  out, lse = plgpu.kernel(
      entry,
      out_shape=out_shape,
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_seq", "batch"),
      num_threads=3,
      thread_name="wg",
      compiler_params=plgpu.CompilerParams(approx_math=True),
  )(q, k, v)

  if save_residuals:
    assert lse is not None
    return out, (lse,)

  return out

@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
@partial(jax.jit, static_argnames=["config", "save_residuals"])
def attention(q, k, v, config: TuningConfig, save_residuals: bool = False):
  return _attention_forward(q, k, v, config, save_residuals)

def _attention_fwd(q, k, v, config: TuningConfig, save_residuals: bool):
  del save_residuals

  out, (lse,) = _attention_forward(q, k, v, config, save_residuals=True)
  return out, (q, k, v, out, lse)

def _attention_bwd(config: TuningConfig, save_residuals: bool, res, do):
  del save_residuals
  q, k, v, out, lse = res

  if config.causal:
    raise NotImplementedError("Causal attention not supported in the backwards pass yet.")

  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")

  assert config.block_q_dq is not None
  assert config.block_kv_dq is not None
  assert config.block_q_dkv is not None
  assert config.block_kv_dkv is not None

  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  dtype = q.dtype
  compute_wgs = config.compute_wgs_bwd

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
    raise NotImplementedError(f"{q_seq_len=} must be a multiple of {config.block_q_dkv=}")

  num_kv_tiles_in_dq, rem = divmod(kv_seq_len, config.block_kv_dq)
  if rem:
    raise NotImplementedError(f"{kv_seq_len=} must be a multiple of {config.block_kv_dq=}")

  tiling = plgpu.TilingTransform((8, 64))
  swizzle = plgpu.SwizzleTransform(128)

  delta = jnp.einsum('bqhd,bqhd->bhq', out.astype(jnp.float32), do.astype(jnp.float32))
  del out  # Not needed anymore.

  def kernel_dq(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dq_ref,
                smem_buffers, buffer_barriers, block_q, block_kv):
    batch = lax.axis_index("batch")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
    q_smem2, do_smem2, lse_smem2, delta_smem2 = smem_buffers
    q_barriers, do_barriers, lse_barriers, delta_barriers = buffer_barriers
    def _compute_thread(pipeline_callback):
      q_smem, do_smem, lse_smem, delta_smem = q_smem2.at[wg_idx], do_smem2.at[wg_idx], lse_smem2.at[wg_idx], delta_smem2.at[wg_idx]
      q_seq_base = lax.axis_index("q_seq") * (compute_wgs * block_q) + wg_idx * block_q
      q_slice = (batch, pl.ds(q_seq_base, block_q), q_head)
      plgpu.copy_gmem_to_smem(q_ref.at[q_slice], q_smem, q_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(do_ref.at[q_slice], do_smem, do_barriers.at[wg_idx])
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
      dq_acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim), 0, dtype=jnp.float32), plgpu.Layout.WGMMA,
      )
      dq, _, _ = pipeline_callback((dq_acc, lse, delta))
      q_smem[...] = dq.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[q_slice])
      plgpu.wait_smem_to_gmem(0)

    def kv_pipeline(_, k_smem, v_smem, k_consumed_barrier, v_consumed_barrier, carry):
      q_smem, do_smem = q_smem2.at[wg_idx], do_smem2.at[wg_idx]
      (dq_acc, lse, delta) = carry

      def compute_s(acc_ref):
        plgpu.wgmma(acc_ref, q_smem, plgpu.transpose_ref(k_smem, (1, 0)))
        return acc_ref[...]

      s = pl.run_scoped(compute_s, plgpu.ACC((block_q, block_kv), jnp.float32))
      s *= math.log2(math.e)
      p = jnp.exp2(s - lax.broadcast_in_dim(lse, (block_q, block_kv), [0]))

      # dP
      def compute_dp(acc_ref):
        plgpu.wgmma(acc_ref, do_smem, plgpu.transpose_ref(v_smem, (1, 0)))
        return acc_ref[...]

      dp = pl.run_scoped(compute_dp, plgpu.ACC((block_q, block_kv), jnp.float32))
      plgpu.barrier_arrive(v_consumed_barrier)

      # dS
      ds = p * (dp - lax.broadcast_in_dim(delta, (block_q, block_kv), [0]))

      # dQ
      def compute_dq(acc_ref):
        plgpu.wgmma(acc_ref, ds.astype(k_ref.dtype), k_smem)

      dq_acc = pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))
      plgpu.barrier_arrive(k_consumed_barrier)

      return (dq_acc, lse, delta)

    pipeline = plgpu.emit_pipeline_warp_specialized(
        kv_pipeline,
        grid=(num_kv_tiles_in_dq,),
        max_concurrent_steps=min([config.max_concurrent_steps, num_q_tiles]),
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=_compute_thread,
        in_specs=[
            plgpu.BlockSpec(  # k
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0),
                transforms=[tiling, swizzle]),
            plgpu.BlockSpec(  # v
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0),
                transforms=[tiling, swizzle]),
        ])
    k_ref = k_ref.at[batch, :, kv_head, :]
    v_ref = v_ref.at[batch, :, kv_head, :]
    pipeline(k_ref, v_ref)

  def kernel_dkv(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref,
                 dk_ref, dv_ref, smem_buffers, buffer_barriers, block_q: int, block_kv: int):
    batch = lax.axis_index("batch")
    q_head = lax.axis_index("heads")
    wg_idx = lax.axis_index("wg")
    (k_smem2, v_smem2) = smem_buffers
    (k_barriers, v_barriers) = buffer_barriers

    def _compute_thread(pipeline_callback):
      k_smem, v_smem = k_smem2.at[wg_idx], v_smem2.at[wg_idx]
      kv_seq_base = lax.axis_index("kv_seq") * (compute_wgs * block_kv) + wg_idx * block_kv
      kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))
      plgpu.copy_gmem_to_smem(
          k_ref.at[(batch, pl.ds(kv_seq_base, block_kv), kv_head)],
          k_smem,
          k_barriers.at[wg_idx])
      plgpu.copy_gmem_to_smem(
          v_ref.at[(batch, pl.ds(kv_seq_base, block_kv), kv_head)],
          v_smem,
          v_barriers.at[wg_idx])
      plgpu.barrier_wait(k_barriers.at[wg_idx])
      plgpu.barrier_wait(v_barriers.at[wg_idx])
      dk_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim), 0, dtype=jnp.float32), plgpu.Layout.WGMMA,
      )
      dv_acc = plgpu.layout_cast(
          jnp.full((block_kv, head_dim), 0, dtype=jnp.float32), plgpu.Layout.WGMMA,
      )
      (dk, dv) = pipeline_callback((dv_acc, dk_acc))
      k_smem[...] = dk.astype(dtype)
      v_smem[...] = dv.astype(dtype)

      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          k_smem,
          dk_ref.at[(batch, pl.ds(kv_seq_base, block_kv), q_head)],
          commit_group=False)
      plgpu.copy_smem_to_gmem(
          v_smem,
          dv_ref.at[(batch, pl.ds(kv_seq_base, block_kv), q_head)],
          commit_group=False)
      plgpu.commit_smem_to_gmem_group()
      plgpu.wait_smem_to_gmem(0)

    def q_pipeline(_, q_smem, do_smem, lse_smem, delta_smem, q_consumed_barrier, do_consumed_barrier, lse_consumed_barrier, delta_consumed_barrier, carry):
      k_smem, v_smem = k_smem2.at[wg_idx], v_smem2.at[wg_idx]
      dk_acc, dv_acc = carry

      def _compute_sT(acc_ref):
        plgpu.wgmma(acc_ref, k_smem, plgpu.transpose_ref(q_smem, (1, 0)))
        return acc_ref[...]
      sT = pl.run_scoped(_compute_sT, plgpu.ACC((block_kv, block_q), jnp.float32))
      sT *= math.log2(math.e)

      lse = plgpu.load(lse_smem, (), layout=plgpu.Layout.WGMMA_COL)
      plgpu.barrier_arrive(lse_consumed_barrier)
      pT = jnp.exp2(sT - lax.broadcast_in_dim(lse, (block_kv, block_q), [1]))

      def _compute(refs):
        # Combining two WGMMA calls in one block to avoid the unnecessary
        # synchronization from two `wgmma.wait_group` calls.
        dv_acc_ref, dpT_acc_ref = refs
        plgpu.wgmma(dv_acc_ref, pT.astype(dtype), do_smem)  # dV
        plgpu.wgmma(dpT_acc_ref, v_smem, plgpu.transpose_ref(do_smem, (1, 0)))  # dpT

      zeros = plgpu.layout_cast(
          jnp.full((block_kv, block_q), 0, dtype=jnp.float32), plgpu.Layout.WGMMA,
      )
      dv_acc, dpT = pl.run_state(_compute)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
      plgpu.barrier_arrive(do_consumed_barrier)

      delta = plgpu.load(delta_smem, (), layout=plgpu.Layout.WGMMA_COL)
      plgpu.barrier_arrive(delta_consumed_barrier)

      dsT = pT * (dpT - lax.broadcast_in_dim(delta, (block_kv, block_q), [1]))

      def compute_dk(acc_ref):
        plgpu.wgmma(acc_ref, dsT.astype(dtype), q_smem)

      dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
      plgpu.barrier_arrive(q_consumed_barrier)

      return (dk_acc, dv_acc)

    pipeline = plgpu.emit_pipeline_warp_specialized(
      q_pipeline,
      grid=(num_q_tiles_in_dkv,),
      max_concurrent_steps=min([config.max_concurrent_steps, num_kv_tiles]),
      num_compute_wgs=compute_wgs,
      memory_registers=40,
      wg_axis="wg",
      manual_consumed_barriers=True,
      compute_context=_compute_thread,
      in_specs=[
          plgpu.BlockSpec(  # q
              block_shape=(block_q, head_dim),
              index_map=lambda i: (i, 0),
              transforms=[tiling, swizzle]),
          plgpu.BlockSpec(  # do
              block_shape=(block_q, head_dim),
              index_map=lambda i: (i, 0),
              transforms=[tiling, swizzle]),
          plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,)),
          plgpu.BlockSpec(block_shape=(block_q,), index_map=lambda i: (i,))
      ])
    q_ref = q_ref.at[batch, :, q_head, :]
    do_ref = do_ref.at[batch, :, q_head, :]
    lse_ref = lse_ref.at[batch, q_head, :]
    delta_ref = delta_ref.at[batch, q_head, :]
    pipeline(q_ref, do_ref, lse_ref, delta_ref)

  q_scratch = plgpu.SMEM(
      (compute_wgs, config.block_q_dq, head_dim), jnp.float16,
      transforms=(tiling, swizzle),
  )
  do_scratch = q_scratch
  lse_scratch = plgpu.SMEM((compute_wgs, config.block_q_dq), jnp.float32)
  delta_scratch = plgpu.SMEM((compute_wgs, config.block_q_dq), jnp.float32)
  dq = plgpu.kernel(
      partial(kernel_dq, block_q=config.block_q_dq, block_kv=config.block_kv_dq),
      out_shape=q,
      scratch_shapes=[
          (q_scratch, do_scratch, lse_scratch, delta_scratch),  # type: ignore
          (plgpu.Barrier(num_barriers=compute_wgs),) * 4  # type: ignore
      ],
      compiler_params=plgpu.CompilerParams(approx_math=True),
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_seq", "batch"),
      num_threads=compute_wgs + 1,
      thread_name="wg",
  )(q, k, v, do, lse, delta)

  k_scratch = plgpu.SMEM(
          (compute_wgs, config.block_kv_dkv, head_dim), jnp.float16,
          transforms=(tiling, swizzle),
      )
  v_scratch = k_scratch
  out_shape_kv = jax.ShapeDtypeStruct(
      (batch_size, kv_seq_len, num_q_heads, head_dim), dtype=jnp.float16)
  dk, dv = plgpu.kernel(
    partial(kernel_dkv, block_q=config.block_q_dkv, block_kv=config.block_kv_dkv),
    out_shape=[out_shape_kv, out_shape_kv],
    scratch_shapes=[
        (k_scratch, v_scratch),  # type: ignore
        (plgpu.Barrier(num_barriers=compute_wgs),) * 2  # type: ignore
  ],
    compiler_params=plgpu.CompilerParams(approx_math=True),
    grid=(num_q_heads, num_kv_tiles, batch_size),
    grid_names=("heads", "kv_seq", "batch"),
    num_threads=compute_wgs + 1,
    thread_name="wg"
  )(q, k, v, do, lse, delta)

  if q_heads_per_kv_head > 1:
    sum_shape = (*k.shape[:-1], q_heads_per_kv_head, head_dim)
    dk = dk.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dk.dtype)
    dv = dv.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dv.dtype)

  return dq, dk, dv

attention.defvjp(_attention_fwd, _attention_bwd)

@functools.partial(jax.jit, static_argnames=["config", "save_residuals"])
def attention_with_pipeline_emitter(q, k, v, config: TuningConfig, save_residuals=False):
  if config.causal:
    raise NotImplementedError("Causal attention is not supported with the pipeline emitter yet.")
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    raise ValueError(f"q, k, and v should all be 4D, got: {q.ndim=}, {k.ndim=}, {v.ndim=}")
  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape
  kv_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
  if k.shape != kv_shape:
    raise ValueError(f"Expected {k.shape=} to be {kv_shape} (inferred from q)")
  if k.shape != kv_shape:
    raise ValueError(f"Expected {v.shape=} to be {kv_shape} (inferred from q)")
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(f"q, k, and v should all have the same dtype, got: {q.dtype}, {k.dtype}, {v.dtype}")
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if head_dim % 64:
    raise ValueError(f"{head_dim=} must be divisible by 64")
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(f"Only f16 and bf16 are supported, got dtype: {dtype}")

  max_concurrent_steps = min(
      config.max_concurrent_steps, kv_seq_len // config.block_kv
  )
  compute_wgs = 2
  block_q, block_kv = config.block_q, config.block_kv
  num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
  if rem:
    raise NotImplementedError(f"{q_seq_len=} must be a multiple of {block_q * 2=}")

  def fa3_kernel(q_ref, k_ref, v_ref, out_ref, lse_ref, smem_buffers, q_barriers, schedule_barrier):
    batch = lax.axis_index("batch")
    wg_idx = lax.axis_index("wg")
    qo_smem2, lse_smem2 = smem_buffers
    q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q
    q_head = lax.axis_index("heads")
    kv_head = lax.div(q_head, jnp.array(q_heads_per_kv_head, q_head.dtype))

    def perform_schedule_barrier():
      if config.use_schedule_barrier:
        plgpu.barrier_arrive(schedule_barrier)
        plgpu.barrier_wait(schedule_barrier)

    def _compute_thread(pipeline_callback):
      qo_smem = qo_smem2.at[wg_idx]
      lse_smem = lse_smem2.at[wg_idx] if lse_smem2 is not None else None
      m_i = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
      l_i = jnp.full((block_q,), 0, dtype=jnp.float32)
      acc = jnp.full((block_q, head_dim), 0, dtype=jnp.float32)
      # Q is not pipelined, so we load in with a manual DMA.
      plgpu.copy_gmem_to_smem(
          q_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
          qo_smem,
          q_barriers.at[wg_idx],
      )
      plgpu.barrier_wait(q_barriers.at[wg_idx])
      pl.when(wg_idx == 1)(perform_schedule_barrier)
      final_carry = pipeline_callback((acc, m_i, l_i))
      pl.when(wg_idx == 0)(perform_schedule_barrier)
      acc, m_i, l_i = final_carry
      acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
      qo_smem[...] = acc.astype(dtype)
      if lse_smem is not None:
        RCP_LN2 = 1.4426950408889634
        log2 = lambda x: jnp.log(x) * RCP_LN2
        lse_smem[...] = m_i + log2(l_i)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          qo_smem, out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
      )
      if lse_smem is not None:
        plgpu.copy_smem_to_gmem(
            lse_smem,
            lse_ref.at[batch, q_head, pl.ds(q_seq_base, block_q)],
        )
      plgpu.wait_smem_to_gmem(0)

    def kv_pipeline(_, k_smem, v_smem,
                    k_consumed_barrier, v_consumed_barrier,
                    carry):
      acc, m_i, l_i = carry
      qo_smem = qo_smem2.at[wg_idx]
      def compute_qk(acc_ref):
        plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem, (1, 0)))
        perform_schedule_barrier()
        return acc_ref[...]
      qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
      plgpu.barrier_arrive(k_consumed_barrier)

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
      perform_schedule_barrier()
      l_i += p.sum(axis=1)
      # PV
      def compute_pv(acc_ref):
        plgpu.wgmma(acc_ref, p16, v_smem)
      acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
      plgpu.barrier_arrive(v_consumed_barrier)
      return acc, m_i, l_i
    pipeline = plgpu.emit_pipeline_warp_specialized(
        kv_pipeline,
        grid=(kv_seq_len // block_kv,),
        max_concurrent_steps=max_concurrent_steps,
        num_compute_wgs=compute_wgs,
        memory_registers=40,
        wg_axis="wg",
        manual_consumed_barriers=True,
        compute_context=_compute_thread,
        in_specs=[
            plgpu.BlockSpec(  # k
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0)),
            plgpu.BlockSpec(  # v
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0)),
        ],
        out_specs=[],
    )
    k_ref = k_ref.at[batch, :, kv_head, :]
    v_ref = v_ref.at[batch, :, kv_head, :]
    pipeline(k_ref, v_ref)

  out_shape = [q, None]
  if save_residuals:
    out_shape[1] = jax.ShapeDtypeStruct((batch_size, num_q_heads, q_seq_len), jnp.float32)

  qo_scratch = plgpu.SMEM((compute_wgs, block_q, head_dim), jnp.float16)
  smem_scratch = [qo_scratch, None]
  if save_residuals:
    smem_scratch[1] = plgpu.SMEM((compute_wgs, block_q), jnp.float32)

  out, lse = plgpu.kernel(
      fa3_kernel,
      grid=(num_q_heads, num_q_tiles, batch_size),
      grid_names=("heads", "q_seq", "batch"),
      num_threads=3,
      thread_name="wg",
            out_shape=out_shape,
      scratch_shapes=(
          tuple(smem_scratch),  # type: ignore
          plgpu.Barrier(num_barriers=compute_wgs),  # type: ignore
          plgpu.Barrier(num_arrivals=compute_wgs),),  # type: ignore
      compiler_params=plgpu.CompilerParams(
          approx_math=True, lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
      ),
  )(q, k, v)

  if save_residuals:
    assert lse is not None
    return out, (lse,)

  return out


@functools.partial(jax.jit, static_argnames=["causal", "save_residuals"])
def attention_reference(q, k, v, causal=False, save_residuals=False):
  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  kv_seq_len, num_kv_heads = k.shape[1], k.shape[2]
  q, k, v = map(lambda x: x.astype(jnp.float32), (q, k, v))
  q_reshaped = q.reshape(
      batch_size, q_seq_len, num_kv_heads, num_q_heads // num_kv_heads, head_dim
  )
  logits = jnp.einsum("bqHhc,bkHc->bqHhk", q_reshaped, k)

  if causal:
    mask = jnp.arange(q_seq_len)[:, None] >= jnp.arange(kv_seq_len)[None, :]
    mask = jnp.broadcast_to(mask[:, None, None, :], logits.shape)
    logits = jnp.where(mask, logits, -jnp.inf)

  m = logits.max(axis=-1, keepdims=True)
  unnormalized = jnp.exp(logits - m)
  l = unnormalized.sum(axis=-1, keepdims=True)
  weights = unnormalized / l
  out = jnp.einsum("bqHhk,bkHc->bqHhc", weights, v).reshape(*q.shape)

  if save_residuals:
    log2e = math.log2(math.e)
    l = l.reshape(*q.shape[:-1])
    m = m.reshape(*q.shape[:-1])
    lse = m * log2e + jnp.log2(l)
    return out, (lse.swapaxes(-1, -2),)
  else:
    return out

def main(unused_argv):
  num_q_heads = 16
  num_kv_heads = 16
  use_pipeline_emitter = False
  if use_pipeline_emitter:
    attention_impl = attention_with_pipeline_emitter
    schedule_barrier_opts = (True, False)
  else:
    attention_impl = attention
    schedule_barrier_opts = (True,)

  problem_it = itertools.product(
      (1,), (4096, 32768,), (64, 128, 256,), schedule_barrier_opts, (False, True))
  for batch_size, seq_len, head_dim, use_schedule_barrier, causal in problem_it:
    assert cuda_versions is not None
    cuda_runtime_version = cuda_versions.cuda_runtime_get_version()
    # TODO(pobudzey): Undo when we upgrade to cuda 12.9.1.
    if causal and cuda_runtime_version >= 12080 and cuda_runtime_version < 12091:
      continue

    if causal and use_pipeline_emitter:
      continue
    q_seq_len = kv_seq_len = seq_len
    print(f"==== {batch_size=:<6} {kv_seq_len=:<6} {q_seq_len=:<6}"
          f"{num_q_heads=:<4} {head_dim=:<6} {use_schedule_barrier=:} {causal=:} ====")
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    q = jax.random.normal(k1, (batch_size, q_seq_len, num_q_heads, head_dim), jnp.float16)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    block_q = 64
    best = None
    for block_kv in (256, 128, 64):
      config = TuningConfig(block_q=block_q, block_kv=block_kv, max_concurrent_steps=2, use_schedule_barrier=use_schedule_barrier, causal=causal)
      try:
        out, runtime_ms = profiler.measure(functools.partial(attention_impl, config=config))(q, k, v)
        if seq_len < 32768:
          out_ref = attention_reference(q, k, v, causal=causal)
          np.testing.assert_allclose(out, out_ref, atol=2e-3, rtol=1e-3)
      except ValueError as e:
        if "exceeds available shared memory" in e.args[0]:
          continue
        raise
      runtime_us = runtime_ms * 1e3
      matmul_flops = (
          4 * q_seq_len * kv_seq_len * head_dim * num_q_heads * batch_size
      )
      if causal:
        matmul_flops //= 2
      peak_flops = 1e15  # f16 TensorCore peak = 1000TFLOPS
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      print(
          f"block_q={block_q:<4}block_kv={block_kv:<4}:  {runtime_us:<7.1f}us"
          f" = {achieved_tc_util:4.1f}% TC utilization"
      )
      if best is None or runtime_us < best[0]:
        best = (runtime_us, achieved_tc_util)
      break  # Remove this for full autotuning.
    if best is not None:
      print(f"Best: {best[0]:<7.1f}us = {best[1]:4.1f}% TC utilization")


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
