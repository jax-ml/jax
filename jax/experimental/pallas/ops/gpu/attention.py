# Copyright 2023 The JAX Authors.
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

"""Module containing fused attention forward and backward pass."""
from __future__ import annotations

import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import numpy as np
import dataclasses

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
  """
  Tile sizes parameterizing the attention kernel. These block sizes
  should be tuned for the model and hardware for optimal performance.

  Attributes:
    block_q: Block size along Q sequence length for forward kernel.
    block_k: Block size along KV sequence length for forward kernel.
    block_kv: Block size along KV sequence length for forward kernel.
    block_q_dkv: Block size along Q sequence length for dKV backward kernel.
    block_kv_dkv: Block size along KV sequence length for dKV backward kernel.
    block_q_dq: Block size along Q sequence length for dQ backward kernel.
    block_kv_dq: Block size along KV sequence length for dQ backward kernel.
  """
  block_q: int
  block_k: int

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_q_dq: int | None = None
  block_kv_dq: int | None = None

  @classmethod
  def get_default(cls):
    return BlockSizes(
        block_q=128,
        block_k=128,
        block_q_dkv=32,
        block_kv_dkv=32,
        block_q_dq=32,
        block_kv_dq=32,
    )

  @property
  def has_backward_blocks(self) -> bool:
    """Returns True if all backward blocks are specified for the fused

    dq and dk/dv backwards pass.
    """
    backward_blocks = [
        self.block_q_dkv,
        self.block_kv_dkv,
        self.block_q_dq,
        self.block_kv_dq,
    ]

    return all(b is not None for b in backward_blocks)


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,  # Input arrays
    segment_ids_ref: jax.Array | None,  # segment_id arrays
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    head_dim: int,
):
  seq_len = k_ref.shape[0]
  start_q = pl.program_id(0)
  head_dim_padded = q_ref.shape[-1]

  # o is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  m_i = jnp.zeros(block_q, dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros(block_q, dtype=jnp.float32)
  # acc is the buffer where we accumulate the output on sram.
  o = jnp.zeros((block_q, head_dim_padded), dtype=jnp.float32)

  # Load q: it will stay in L1 throughout. Indices form a matrix because we
  # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
  # q tile has shape [block_q, head_dim_padded], head_dim_padded >= head_dim.
  curr_q_slice = pl.dslice(start_q * block_q, block_q)
  head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
  q = plgpu.load(q_ref, mask=head_mask, other=0.0)
  q_segment_ids = (
      None if segment_ids_ref is None else segment_ids_ref[curr_q_slice]
  )
  # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
  # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
  # Here we only loop over blocks of kv to process entire seq_len, the loop over
  # blocks of q is carried out by the grid.
  def body(start_k, carry):
    o_prev, m_prev, l_prev = carry
    curr_k_slice = pl.dslice(start_k * block_k, block_k)

    k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
    qk = pl.dot(q, k.T)   # [block_q, block_k]

    # Scale logits to convert from base-2 to the natural log domain.
    # This is based on the identity: e^x = 2^(x * log2(e)).
    qk_scale = math.log2(math.e)
    if sm_scale != 1.:
      qk_scale *= sm_scale
    qk *= qk_scale

    # Avoids Triton crash.
    # if num_heads > 2:
    #   qk = qk.astype(q_ref.dtype)
    #   qk = qk.astype(jnp.float32)

    if causal or segment_ids_ref is not None:
      mask = None
      if segment_ids_ref is not None:
        kv_segment_ids = segment_ids_ref[curr_k_slice]
        mask = segment_mask(q_segment_ids, kv_segment_ids)
      if causal:
        span_q = start_q * block_q + jnp.arange(block_q)
        span_k = start_k * block_k + jnp.arange(block_k)
        causal_mask = span_q[:, None] >= span_k[None, :]
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )
      # Apply mask to qk.
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    m_curr = jnp.max(qk, axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp2(m_prev - m_next)
    l_prev_corr = correction * l_prev
    s_curr = jnp.exp2(
        qk - m_next[:, None]
    )  # Use m_next instead of m_curr to avoid a correction on l_curr
    l_curr = s_curr.sum(axis=-1)
    l_next = l_prev_corr + l_curr
    o_prev_corr = correction[:, None] * o_prev
    v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask)
    o_curr = pl.dot(s_curr.astype(v.dtype), v)

    o_next = o_prev_corr + o_curr
    return o_next, m_next, l_next
  if causal:
    # Ceildiv (`pl.cdiv` and `//` do not work due to type of start_q)
    upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
  else:
    upper_bound = pl.cdiv(seq_len, block_k)
  o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

  # We keep an unscaled version of o during the scan over seq_len. Scaling it
  # by the last l_i gives us the correct final output. See section 3.1.1 in the
  # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
  o /= l_i[:, None]

  if residual_refs:
    lse_ref = residual_refs[0]
    lse_ref[...] = m_i + jnp.log2(l_i)
  # Write output to dram.
  plgpu.store(o_ref.at[:, : o.shape[-1]], o.astype(o_ref.dtype), mask=head_mask)

def segment_mask(
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
):
  # [B, T, 1] or [T, 1]
  q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
  # [B, 1, S] or [1, S]
  if kv_segment_ids.ndim == 1:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
  else:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
  return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


@functools.partial(
    jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "block_sizes",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "return_residuals",
    ],
)
def mha(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False,
):
  del backward_pass_impl
  batch_size, q_seq_len, num_heads, head_dim = q.shape
  kv_seq_len = k.shape[1]
  block_q = min(block_sizes.block_q, q_seq_len)
  block_k = min(block_sizes.block_k, kv_seq_len)
  head_dim_padded = pl.next_power_of_2(head_dim)
  if (q.shape[-1] != k.shape[-1]) or (q.shape[-1] != v.shape[-1]):
    raise ValueError(
        f"This kernel expects q, k, and v to have the same head dimension, but"
        f" found {q.shape=}, {k.shape=}, {v.shape=}."
    )
  if q_seq_len % block_q != 0:
    raise ValueError(f"{q_seq_len=} must be a multiple of {block_q=}")
  if kv_seq_len % block_k != 0:
    raise ValueError(f"{kv_seq_len=} must be a multiple of {block_k=}")

  # Heuristics.
  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_forward_kernel, sm_scale=sm_scale,
                             block_q=block_q, block_k=block_k,
                             head_dim=head_dim, causal=causal)

  in_specs = [
      pl.BlockSpec((None, block_q, None, head_dim_padded),
                   lambda i, j, k: (j, i, k, 0)),
      pl.BlockSpec((None, kv_seq_len, None, head_dim_padded),
                   lambda _, j, k: (j, 0, k, 0)),
      pl.BlockSpec((None, kv_seq_len, None, head_dim_padded),
                   lambda _, j, k: (j, 0, k, 0)),
  ]
  in_specs.append(
      None  # type: ignore[arg-type]
      if segment_ids is None
      else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
  )
  out_shape = [q]
  out_specs = [pl.BlockSpec((None, block_q, None, head_dim_padded),
                            lambda i, j, k: (j, i, k, 0))]
  if return_residuals:
    out_shape.append(jax.ShapeDtypeStruct(
        shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32))  # lse
    out_specs.append(
        pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)))  # lse
  out = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=in_specs,
      out_specs=out_specs,
      compiler_params=plgpu.CompilerParams(
          num_warps=num_warps_, num_stages=num_stages),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward",
  )(q, k, v, segment_ids)
  return out if return_residuals else out[0]


def _mha_forward(
    q,
    k,
    v,
    segment_ids: jax.Array | None,
    sm_scale: float,
    causal: bool,
    block_sizes: BlockSizes,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    return_residuals: bool,
):
  out, lse = mha(q, k, v, segment_ids=segment_ids, sm_scale=sm_scale,
                 causal=causal, block_sizes=block_sizes,
                 backward_pass_impl=backward_pass_impl,
                 num_warps=num_warps, num_stages=num_stages,
                 grid=grid, interpret=interpret, debug=debug,
                 return_residuals=True)
  residuals = (q, k, v, segment_ids, out, lse)
  ret = (out, lse) if return_residuals else out
  return ret, residuals


def _preprocess_backward_kernel(out_ref, dout_ref, delta_ref, head_dim: int):
  # load
  head_mask = (jnp.arange(out_ref.shape[-1]) < head_dim)[None, :]
  o = plgpu.load(out_ref, mask=head_mask, other=0.0)
  do = plgpu.load(dout_ref, mask=head_mask, other=0.0)
  # compute
  delta = jnp.sum(o * do, axis=1)
  # write-back
  delta_ref[...] = delta.astype(delta_ref.dtype)

@jax.named_scope("preprocess_backward")
def _preprocess_backward(out, do, lse, block_q: int,
                         debug: bool, interpret: bool):
  batch_size, seq_len, num_heads, head_dim = out.shape
  head_dim_padded = pl.next_power_of_2(head_dim)
  out_shape = jax.ShapeDtypeStruct(lse.shape, lse.dtype)
  delta = pl.pallas_call(
      functools.partial(_preprocess_backward_kernel, head_dim=head_dim),
      grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
      in_specs=[
          pl.BlockSpec((None, block_q, None, head_dim_padded),
                       lambda i, j, k: (j, i, k, 0)),
          pl.BlockSpec((None, block_q, None, head_dim_padded),
                       lambda i, j, k: (j, i, k, 0)),
      ],
      out_specs=pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_preprocess_backward",
  )(out, do)
  return delta


# This kernel computes dK_i, dV_i and dQ_i in parallel across the sequence
# length.
# Inspired by the triton tutorial: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
def mha_backward_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    head_dim: int,
):
  del out_ref  # Not needed
  q_seq_len = q_ref.shape[0]
  kv_seq_len = k_ref.shape[0]

  # Scan #1: dK and dV
  #   1. Load a block of K and V of size (block_kv_dkv, head_dim) in SMEM.
  #   2. Iterate through Q in chunks of (block_q_dkv, head_dim) to accumulate
  #      dK and dV.
  start_k = pl.program_id(2)
  curr_k_slice = pl.dslice(start_k * block_kv_dkv, block_kv_dkv)

  head_dim_padded = q_ref.shape[-1]
  dv = jnp.zeros([block_kv_dkv, head_dim_padded], dtype=jnp.float32)
  dk = jnp.zeros([block_kv_dkv, head_dim_padded], dtype=jnp.float32)

  head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
  v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
  k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
  span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)
  kv_segment_ids = (
      None if segment_ids_ref is None else segment_ids_ref[curr_k_slice]
  )

  def inner_loop_dkdv(start_q, carry):
    dv, dk = carry
    curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)

    q = plgpu.load(q_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
    qk = pl.dot(q, k.T)
    qk_scale = math.log2(math.e)
    if sm_scale != 1.:
      qk_scale *= sm_scale
    qk *= qk_scale

    if causal or segment_ids_ref is not None:
      mask = None
      if segment_ids_ref is not None:
        q_segment_ids = segment_ids_ref[curr_q_slice]
        mask = segment_mask(q_segment_ids, kv_segment_ids)

      if causal:
        span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
        causal_mask = span_q[:, None] >= span_k[None, :]
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    lse = lse_ref[curr_q_slice]
    di = delta_ref[curr_q_slice]
    do = plgpu.load(
        do_scaled_ref.at[curr_q_slice, :], mask=head_mask, other=0.0
    )

    p = jnp.exp2(qk - lse[:, None])
    dv = dv + pl.dot(p.astype(do.dtype).T, do)
    dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
    dp = dp + pl.dot(do, v.T)
    ds = p * dp
    if sm_scale != 1.0:
      ds = ds * sm_scale
    dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)

    return dv, dk

  lower_bound = lax.div(start_k * block_kv_dkv, block_q_dkv) if causal else 0
  dv, dk = lax.fori_loop(
      lower_bound, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk)
  )
  plgpu.store(
      dv_ref.at[:, : dv.shape[-1]], dv.astype(dv_ref.dtype), mask=head_mask
  )
  plgpu.store(
      dk_ref.at[:, : dk.shape[-1]], dk.astype(dk_ref.dtype), mask=head_mask
  )

  # Scan #2: dQ
  #   1. Load a block of Q of size (block_q_dq, head_dim) in SMEM.
  #   2. Iterate through K and V in chunks of (block_kv_dq, head_dim) to
  #     accumulate dQ.
  start_q = pl.program_id(2)
  curr_q_slice = pl.ds(start_q * block_q_dq, block_q_dq)
  span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
  dq = jnp.zeros([block_q_dq, head_dim_padded], dtype=jnp.float32)

  q = plgpu.load(q_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
  q_segment_ids = (
      None if segment_ids_ref is None else segment_ids_ref[curr_q_slice]
  )
  lse = lse_ref[curr_q_slice]
  do = plgpu.load(do_scaled_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
  di = delta_ref[curr_q_slice]

  def inner_loop_dq(start_k, dq):
    curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
    k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
    v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)

    qk = pl.dot(q, k.T)
    qk_scale = math.log2(math.e)
    if sm_scale != 1.:
      qk_scale *= sm_scale
    qk *= qk_scale

    if causal or segment_ids_ref is not None:
      mask = None
      if segment_ids_ref is not None:
        kv_segment_ids = segment_ids_ref[curr_k_slice]
        mask = segment_mask(q_segment_ids, kv_segment_ids)

      if causal:
        span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
        causal_mask = span_q[:, None] >= span_k[None, :]
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    p = jnp.exp2(qk - lse[:, None])
    dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
    dp = dp + pl.dot(do, v.T)
    ds = p * dp
    if sm_scale != 1.0:
      ds = ds * sm_scale

    dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)

    return dq

  if causal:
    upper_bound = pl.cdiv((start_q + 1) * block_q_dq, block_kv_dq)
  else:
    upper_bound = pl.cdiv(kv_seq_len, block_kv_dq)

  dq = lax.fori_loop(0, upper_bound, inner_loop_dq, (dq))
  plgpu.store(
      dq_ref.at[:, : dq.shape[-1]], dq.astype(dq_ref.dtype), mask=head_mask
  )


def _mha_backward(sm_scale: float, causal: bool, block_sizes: BlockSizes,
                  backward_pass_impl: str, num_warps: int | None,
                  num_stages: int, grid: Any, interpret: bool,
                  debug: bool, return_residuals: bool, res, do):
  if return_residuals:
    raise ValueError(
        "Kernel differentiation is not supported if return_residuals is True.")
  q, k, v, segment_ids, out, lse = res
  del num_stages, grid, return_residuals

  if backward_pass_impl == "xla":
    return jax.vjp(
        functools.partial(mha_reference, sm_scale=sm_scale, causal=causal),
        q,
        k,
        v,
        segment_ids,
    )[1](do)
  elif backward_pass_impl == "triton":
    if not block_sizes.has_backward_blocks:
      raise ValueError("Backward block sizes must all be set.")

    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
    block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
    block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
    block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)
    head_dim_padded = pl.next_power_of_2(head_dim)

    if q_seq_len // block_q_dq != kv_seq_len // block_kv_dkv:
      raise ValueError(
          "q_seq_len and kv_seq_len must be divided into the same "
          "number of blocks for the fused backward pass."
      )

    delta = _preprocess_backward(out, do, lse, block_q, debug, interpret)
    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
    ]

    in_specs = [
        pl.BlockSpec((None, q_seq_len, None, head_dim_padded),
                     lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim_padded),
                     lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim_padded),
                     lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, q_seq_len, None, head_dim_padded),
                     lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, q_seq_len, None, head_dim_padded),
                     lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
    ]
    if segment_ids is None:
      in_specs.insert(3, None)  # type: ignore[arg-type]
    else:
      in_specs.insert(3, pl.BlockSpec((None, kv_seq_len),
                                      lambda i, j, _: (i, 0)))

    grid = (batch_size, num_heads, pl.cdiv(kv_seq_len, block_kv_dkv))
    num_warps_ = num_warps
    if num_warps_ is None:
      if (
          block_q_dkv * block_kv_dkv < 128 * 128
          or block_q_dq * block_kv_dq < 128 * 128
      ):
        num_warps_ = 4
      else:
        num_warps_ = 8


    dq, dk, dv = pl.pallas_call(
        functools.partial(
            mha_backward_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_q_dkv=block_q_dkv,
            block_kv_dkv=block_kv_dkv,
            block_q_dq=block_q_dq,
            block_kv_dq=block_kv_dq,
            head_dim=head_dim,
        ),
        out_shape=out_shapes,
        in_specs=in_specs,
        grid=grid,
        out_specs=[
            pl.BlockSpec(
                (None, block_q_dq, None, head_dim_padded),
                lambda i, j, k: (i, k, j, 0),  # dq
            ),
            pl.BlockSpec(
                (None, block_kv_dkv, None, head_dim_padded),
                lambda i, j, k: (i, k, j, 0),  # dk
            ),
            pl.BlockSpec(
                (None, block_kv_dkv, None, head_dim_padded),
                lambda i, j, k: (i, k, j, 0),  # dv
            ),
        ],
        name="mha_backward",
        debug=debug,
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps_, num_stages=2
        ),
    )(q, k, v, segment_ids, out, do, lse, delta)
  else:
    raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
  return dq.astype(q.dtype), dk, dv, None
mha.defvjp(_mha_forward, _mha_backward)


@functools.partial(jax.jit, static_argnames=['sm_scale', 'causal'])
def mha_reference(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale=1.0,
    causal: bool = False,
):
  q_seq_len = q.shape[1]
  kv_seq_len = k.shape[1]
  logits = jnp.einsum(
      'bqhc,bkhc->bhqk', q, k, preferred_element_type=jnp.float32
  )
  mask = None
  if segment_ids is not None:
    mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
    mask = jnp.broadcast_to(mask, logits.shape)
  if causal:
    causal_mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
    causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
  weights = jax.nn.softmax(logits * sm_scale)
  return jnp.einsum(
      'bhqk,bkhc->bqhc', weights, v, preferred_element_type=jnp.float32
  )
