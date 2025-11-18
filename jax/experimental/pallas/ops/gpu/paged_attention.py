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

"""Module containing decode attention."""
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

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def paged_attention_kernel(
    # inputs
    q_ref,  # [block_h, head_dim]
    k_pages_ref,  # [total_num_pages, page_size, head_dim]
    k_scales_pages_ref,  # [total_num_pages, page_size]
    v_pages_ref,  # [total_num_pages, page_size, head_dim]
    v_scales_pages_ref,  # [total_num_pages, page_size]
    block_tables_ref,  # [pages_per_partition]
    lengths_ref,  # [1]
    # outputs
    o_ref: Any,  # [block_h, head_dim]
    *residual_refs: Any,  # Residual outputs: [block_h,], [block_h,]
    num_heads: int,
    pages_per_compute_block: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
):
  partition_idx = pl.program_id(2)
  block_h, head_dim = q_ref.shape
  page_size = k_pages_ref.shape[-2]
  pages_per_partition = block_tables_ref.shape[0]
  block_k = pages_per_compute_block * page_size

  def _compute(start_page_idx, end_page_idx, o, m_i, l_i):
    q_slice = pl.ds(0, block_h)
    q = q_ref[q_slice, :]

    # Loop over blocks of pages to process a entire page sequence partition.
    # Grid loops over q blocks over num_heads.
    def body(start_k, carry):
      o_prev, m_prev, l_prev = carry

      block_tables_slice = pl.ds(
          start_k * pages_per_compute_block, pages_per_compute_block
      )
      block_tables = block_tables_ref[block_tables_slice]
      k = k_pages_ref[block_tables].reshape(block_k, head_dim)
      v = v_pages_ref[block_tables].reshape(block_k, head_dim)
      if k_scales_pages_ref is not None:
        # dynamic lhs quantized dot is not currently implemented
        # so we cast rhs to the lhs dtype
        k = k.astype(q.dtype)
      uncapped_logits = pl.dot(q, k.T)  # [block_h, block_k]
      if k_scales_pages_ref is not None:
        # k_scales_pages_ref are one per head
        # they're laid out across the output dimension, so scale output
        k_scale = k_scales_pages_ref[block_tables].reshape((1, block_k))
        uncapped_logits *= k_scale.astype(uncapped_logits.dtype)
      if attn_logits_soft_cap is not None:
        logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
        logits = logits * attn_logits_soft_cap
      else:
        logits = uncapped_logits

      if lengths_ref is not None:
        curr_start_page_idx = (
            partition_idx * pages_per_partition
            + start_k * pages_per_compute_block
        )
        curr_start_token_idx = curr_start_page_idx * page_size

        mask = jnp.arange(block_k) + curr_start_token_idx < lengths_ref[0]
        mask = lax.broadcast_in_dim(mask, (block_h, block_k), (1,))
        logits = jnp.where(mask, logits, mask_value)

      log2e = math.log2(math.e)
      m_curr = logits.max(axis=-1)
      m_next = jnp.maximum(m_prev, m_curr)
      correction = jnp.exp2((m_prev - m_next) * log2e)
      l_prev_corr = correction * l_prev
      s_curr = jnp.exp2((logits - m_next[:, None]) * log2e)
      l_curr = s_curr.sum(axis=-1)
      l_next = l_prev_corr + l_curr
      o_prev_corr = correction[:, None] * o_prev
      if v_scales_pages_ref is not None:
        # v_scales are 1 per head
        # they're laid out across the reduction dimension, so scale lhs
        v_scale = v_scales_pages_ref[block_tables].reshape((1, block_k))
        s_curr *= v_scale.astype(s_curr.dtype)
        # dynamic lhs quantized dot is not currently implemented
        # so we cast rhs to the lhs dtype
        v = v.astype(s_curr.dtype)
      o_curr = pl.dot(s_curr.astype(v.dtype), v)

      o_next = o_prev_corr + o_curr
      return o_next, m_next, l_next

    max_it = pl.cdiv(end_page_idx - start_page_idx, pages_per_compute_block)
    (o, m_i, l_i) = lax.fori_loop(0, max_it, body, (o, m_i, l_i))

    return o, m_i, l_i

  m_i = jnp.zeros(block_h, dtype=jnp.float32) + jnp.finfo(jnp.float32).min
  l_i = jnp.zeros(block_h, dtype=jnp.float32)
  o = jnp.zeros((block_h, head_dim), dtype=jnp.float32)

  start_page_idx = partition_idx * pages_per_partition
  end_page_idx = start_page_idx + pages_per_partition

  if lengths_ref is None:
    o, m_i, l_i = _compute(start_page_idx, end_page_idx, o, m_i, l_i)
  else:
    end_page_idx = jnp.minimum(pl.cdiv(lengths_ref[0], page_size), end_page_idx)

    o, m_i, l_i = jax.lax.cond(
        start_page_idx >= end_page_idx,
        lambda: (o, m_i, l_i),
        lambda: _compute(start_page_idx, end_page_idx, o, m_i, l_i),
    )

  o_ref[...] = o.astype(o_ref.dtype)

  if residual_refs is not None:
    l_ref, m_ref = residual_refs
    l_ref[...] = l_i
    m_ref[...] = m_i


def paged_attention_unbatched(
    q: jax.Array,  #  [num_q_heads, head_dim]
    k_pages: jax.Array,  #  [num_kv_heads, total_num_pages, page_size, head_dim]
    v_pages: jax.Array,  #  [num_kv_heads, total_num_pages, page_size, head_dim]
    block_tables: jax.Array,  #  [pages_per_sequence]
    lengths: jax.Array | None,  #  [1]
    k_scales_pages: jax.Array | None = None,  # [num_kv_heads, total_num_pages, page_size]
    v_scales_pages: jax.Array | None = None,  # [num_kv_heads, total_num_pages, page_size]
    *,
    block_h: int,
    pages_per_compute_block: int,
    k_splits: int,
    num_warps: int,
    num_stages: int,
    interpret: bool,
    debug: bool,
    mask_value: float,
    attn_logits_soft_cap: float | None,
) -> jax.Array:
  num_q_heads, head_dim = q.shape
  num_kv_heads, total_num_pages, page_size, _ = k_pages.shape
  pages_per_sequence = block_tables.shape[0]

  assert (
      pages_per_sequence % k_splits == 0
  ), f"{pages_per_sequence=} must be divisible by {k_splits=}."

  pages_per_partition = pages_per_sequence // k_splits
  pages_per_compute_block = min(pages_per_partition, pages_per_compute_block)

  assert (
      pages_per_partition % pages_per_compute_block == 0
  ), f"{pages_per_partition=} must de divisible by {pages_per_compute_block=}."

  block_tables = block_tables.reshape(k_splits, pages_per_sequence // k_splits)

  q_heads_per_kv_head = num_q_heads // num_kv_heads
  q_reshaped = q.reshape(num_kv_heads, q_heads_per_kv_head, head_dim)

  if q_heads_per_kv_head % block_h:
    q_reshaped = jnp.pad(
        q_reshaped, ((0, 0), (0, -q_heads_per_kv_head % block_h), (0, 0))
    )

  head_splits = pl.cdiv(q_heads_per_kv_head, block_h)
  grid = (num_kv_heads, head_splits, k_splits)
  kernel = functools.partial(
      paged_attention_kernel,
      num_heads=q_heads_per_kv_head,
      pages_per_compute_block=pages_per_compute_block,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )
  # set up quantization scales
  if k_scales_pages is not None:
    assert k_scales_pages.shape == (num_kv_heads, total_num_pages, page_size)
    k_scales_spec = pl.BlockSpec((None, total_num_pages, page_size),
                                 lambda h, i, k: (h, 0, 0))
  else:
    k_scales_spec = None
  if v_scales_pages is not None:
    assert v_scales_pages.shape == (num_kv_heads, total_num_pages, page_size)
    v_scales_spec = pl.BlockSpec((None, total_num_pages, page_size),
                                 lambda h, i, k: (h, 0, 0))
  else:
    v_scales_spec = None

  o, l, m = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[
          pl.BlockSpec(
              (None, block_h, head_dim), lambda h, i, k: (h, i, 0)
          ),  # q
          pl.BlockSpec(
              (None, total_num_pages, page_size, head_dim),
              lambda h, i, k: (h, 0, 0, 0),
          ),  # k_pages
          k_scales_spec,  # k_pages_scale
          pl.BlockSpec(
              (None, total_num_pages, page_size, head_dim),
              lambda h, i, k: (h, 0, 0, 0),
          ),  # v_pages
          v_scales_spec,  # v_pages_scale
          pl.BlockSpec(
              (None, pages_per_partition), lambda h, i, k: (k, 0)
          ),  # block_tables
      ]
      + [
          None if lengths is None else pl.BlockSpec((1,), lambda h, i, k: (0,))
      ],  # lengths
      out_specs=[
          pl.BlockSpec(
              (None, None, block_h, head_dim), lambda h, i, k: (k, h, i, 0)
          ),  # q
          pl.BlockSpec((None, None, block_h), lambda h, i, k: (k, h, i)),  # l
          pl.BlockSpec((None, None, block_h), lambda h, i, k: (k, h, i)),  # m
      ],
      out_shape=[
          jax.ShapeDtypeStruct(
              (k_splits, *q_reshaped.shape), dtype=q.dtype
          ),  # o
          jax.ShapeDtypeStruct(
              (k_splits, *q_reshaped.shape[:-1]), dtype=jnp.float32
          ),  # l
          jax.ShapeDtypeStruct(
              (k_splits, *q_reshaped.shape[:-1]), dtype=jnp.float32
          ),  # m
      ],
      debug=debug,
      interpret=interpret,
      compiler_params=plgpu.CompilerParams(
          num_warps=num_warps, num_stages=num_stages
      ),
      name=f"paged_attention_{block_h=}_{pages_per_compute_block=}",
  )(q_reshaped, k_pages, k_scales_pages, v_pages, v_scales_pages, block_tables, lengths)

  if q_heads_per_kv_head % block_h:
    o = o[..., :q_heads_per_kv_head, :]
    l = l[..., :q_heads_per_kv_head]
    m = m[..., :q_heads_per_kv_head]

  # final round of flash
  m_next = m.max(axis=0)
  correction = jnp.exp(m - m_next[None])
  o = o * correction[..., None].astype(o.dtype)
  l_next = (l * correction).sum(axis=0)
  eps = jnp.finfo(l_next.dtype).eps
  o = o.sum(axis=0) / ((l_next[..., None] + eps).astype(o.dtype))

  o = o.reshape(q.shape).astype(q.dtype)
  return o


@functools.partial(
    jax.jit,
    static_argnames=[
        "block_h",
        "pages_per_compute_block",
        "k_splits",
        "num_warps",
        "num_stages",
        "interpret",
        "debug",
        "mask_value",
        "attn_logits_soft_cap",
    ],
)
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    block_tables: jax.Array,
    lengths: jax.Array | None,
    k_scales_pages: jax.Array | None = None,
    v_scales_pages: jax.Array | None = None,
    *,
    block_h: int = 16,
    pages_per_compute_block: int = 8,
    k_splits: int = 16,
    num_warps: int = 8,
    num_stages: int = 2,
    interpret: bool = False,
    debug: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array:
  """Paged grouped query attention.

  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    block_tables: A i32[batch_size, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`.
    lengths: A i32[batch_size] jax.Array the length of each example.
    k_scales_pages: A [num_kv_heads, total_num_pages, page_size] jax.Array.
    v_scales_pages: A [num_kv_heads, total_num_pages, page_size] jax.Array.
    block_h: int The block size that partitions the number of head groups.
    pages_per_compute_block: int The maximum number of blocks per compute block.
    k_splits: int Number of partitions used to parallelize key-value sequence
      pages processing.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    attn_logits_soft_cap: The value used for soft capping the attention logits.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]).
  """
  batch_size, num_heads, head_dim = q.shape
  num_kv_heads, _, _, head_dim_k = k_pages.shape
  batch_size_paged_indices, _ = block_tables.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"  # pytype: disable=attribute-error
    )
  if num_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_heads} and {num_kv_heads}."
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if batch_size_paged_indices != batch_size:
    raise ValueError("`block_tables` and `q` must have the same batch size")
  if lengths is not None:
    if lengths.shape != (batch_size,):
      raise ValueError("`lengths` and `q` must have the same batch size")
    if lengths.dtype != jnp.int32:
      raise ValueError(
          "The dtype of `lengths` must be int32. Got {lengths.dtype}"
      )

  if block_h % 16:
    raise ValueError(f"block_h must divisible by 16, but is {block_h}.")

  impl = functools.partial(
      paged_attention_unbatched,
      block_h=block_h,
      pages_per_compute_block=pages_per_compute_block,
      k_splits=k_splits,
      num_warps=num_warps,
      num_stages=num_stages,
      interpret=interpret,
      debug=debug,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )

  o = jax.vmap(impl, (0, None, None, 0, 0, None, None), 0)(
      q,
      k_pages,
      v_pages,
      block_tables,
      lengths[..., None] if lengths is not None else None,
      k_scales_pages,
      v_scales_pages,
  )

  return o


@functools.partial(
    jax.jit, static_argnames=["mask_value", "attn_logits_soft_cap"]
)
def paged_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array:
  """Grouped query attention reference implementation.

  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k: A [batch_size, kv_seq_len, num_kv_heads, head_dim] jax.Array.
    v: A [batch_size, kv_seq_len, num_kv_heads, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array the length of each example.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    attn_logits_soft_cap: The value used for soft capping the attention logits.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]).
  """
  batch_size, num_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape

  q_heads_per_kv_head = num_heads // num_kv_heads
  q_reshaped = q.reshape(
      batch_size, num_kv_heads, q_heads_per_kv_head, head_dim
  )
  k_transposed = jnp.swapaxes(
      k, 1, 2
  )  # [batch_size, num_kv_heads, kv_seq_len, head_dim]
  v_transposed = jnp.swapaxes(
      v, 1, 2
  )  # [batch_size, num_kv_heads, kv_seq_len, head_dim]

  uncapped_logits = jnp.einsum(
      "bkgd,bksd->bkgs", q_reshaped, k_transposed,
      preferred_element_type=jnp.float32
  ).astype(jnp.float32)

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap
  else:
    logits = uncapped_logits

  if lengths is not None:
    mask = jnp.arange(kv_seq_len)[None, :] < lengths[:, None]
    mask = jnp.broadcast_to(mask[:, None, None, :], logits.shape)
    logits = jnp.where(mask, logits, mask_value)

  weights = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum(
      "bkgs,bksd->bkgd", weights, v_transposed.astype(jnp.float32),
      preferred_element_type=jnp.float32
  ).astype(q.dtype)
  o = o.reshape(q.shape)

  return o
