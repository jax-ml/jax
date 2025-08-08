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

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import get_tuned_block_sizes
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.util import (
    align_to,
    cdiv,
    get_dtype_packing,
)
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    actual_num_kv_heads: int,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE

  _, actual_num_q_heads, actual_head_dim = queries.shape
  _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
      kv_cache.shape
  )
  num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
  assert num_kv_heads_x2 % 2 == 0
  assert actual_num_q_heads % actual_num_kv_heads == 0
  assert head_dim % 128 == 0
  assert get_dtype_packing(kv_cache.dtype) == kv_packing
  assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  outputs = []

  for i in range(distribution[-1]):
    q_start = cu_q_lens[i]
    q_end = cu_q_lens[i + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[i]
    indices_start = i * pages_per_seq
    indices_end = indices_start + cdiv(kv_len, page_size)
    indices = page_indices[indices_start:indices_end]
    q = queries[q_start:q_end, :, :actual_head_dim]
    kv = (
        kv_cache[indices]
        .reshape(-1, num_kv_heads_x2, head_dim)[:, : actual_num_kv_heads * 2, :]
        .reshape(-1, actual_num_kv_heads, head_dim * 2)
    )
    k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
    v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
    if k_scale is not None:
      k = (k * k_scale).astype(q.dtype)
    if v_scale is not None:
      v = (v * v_scale).astype(q.dtype)
    k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
    v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale
    q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
    mask = q_span < kv_span
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
    if soft_cap is not None:
      attn = soft_cap * jnp.tanh(attn / soft_cap)
    attn += jnp.where(mask, mask_value, 0.0)
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)


def get_smem_estimate_bytes(max_num_seqs, pages_per_seq):
  total_bits = (
      # kv_lens_ref: i32[max_num_seqs]
      align_to(max_num_seqs, 128) * 32
      +
      # page_indices_ref: i32[max_num_seqs * pages_per_seq]
      align_to(max_num_seqs * pages_per_seq, 128) * 32
      +
      # cu_q_lens_ref: i32[max_num_seqs + 1]
      align_to(max_num_seqs + 1, 128) * 32
      +
      # distribution_ref: i32[3]
      128 * 32
      +
      # sem_ids_ref: i32[3]
      128 * 32
      +
      # bo_ids_ref: i32[4]
      128 * 32
  )
  return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
):
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)
  num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
  num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
  head_dim = align_to(actual_head_dim, 128)

  total_bits = (
      # bkv_x2_ref
      (2 * bkv_sz * num_kv_heads_x2 * head_dim) * (32 // kv_packing)
      +
      # bq_x2_ref + bo_x2_ref
      2
      * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim)
      * (32 // q_packing)
      +
      # l_ref + m_ref
      2 * (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32
      +
      # acc_ref
      (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * 32
  )
  return cdiv(total_bits, 8)


def get_q_kv_shape(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    *,
    max_num_tokens=128,
    total_num_pages=1000,
    page_size=16,
):
  assert actual_num_q_heads % actual_num_kv_heads == 0
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
  q_packing = get_dtype_packing(q_dtype)
  kv_packing = get_dtype_packing(kv_dtype)
  head_dim = align_to(head_dim, 128)

  num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
  num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)

  q_shape = (
      actual_num_kv_heads,
      max_num_tokens,
      num_q_heads_per_kv_head // q_packing,
      q_packing,
      head_dim,
  )

  kv_cache_shape = (
      total_num_pages,
      page_size,
      num_kv_heads_x2 // kv_packing,
      kv_packing,
      head_dim,
  )

  return q_shape, kv_cache_shape


def _ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    # TODO(jevinjiang): merge these into one so we can save SMEM.
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    # Output
    o_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    # Scratch
    bkv_x2_ref,  # [2, bkv_sz, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    sems,  # [3, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim],
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
):
  assert q_hbm_ref.shape == o_hbm_ref.shape
  assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]
  (
      actual_num_kv_heads,
      max_num_tokens,
      num_q_heads_per_kv_head_per_packing,
      q_packing,
      head_dim,
  ) = q_hbm_ref.shape
  (
      total_num_pages,
      page_size,
      num_kv_heads_x2_per_kv_packing,
      kv_packing,
      _,
  ) = kv_cache_hbm_ref.shape
  max_num_seqs = kv_lens_ref.shape[0]
  num_page_indices = page_indices_ref.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
  num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
  q_dtype = q_hbm_ref.dtype
  kv_dtype = kv_cache_hbm_ref.dtype
  assert o_hbm_ref.dtype == q_dtype
  assert get_dtype_packing(q_dtype) == q_packing
  assert get_dtype_packing(kv_dtype) == kv_packing
  assert head_dim % 128 == 0
  bkv_sz = bkv_p * page_size
  seq_idx = pl.program_id(0)
  num_seqs = pl.num_programs(0)
  decode_end = distribution_ref[0]
  prefill_end = distribution_ref[1]
  mixed_end = distribution_ref[2]

  q_start = cu_q_lens_ref[seq_idx]
  q_end = cu_q_lens_ref[seq_idx + 1]
  q_len = q_end - q_start
  kv_len = kv_lens_ref[seq_idx]

  def flash_attention(
      q,  # [actual_bq_sz * num_q_heads_per_kv_head, head_dim]
      k,  # [bkv_sz, head_dim]
      v,  # [bkv_sz, head_dim]
      *,
      bq_idx,
      bkv_idx,
      kv_head_idx,
  ):
    assert len(q.shape) == 2
    assert q.shape[0] % num_q_heads_per_kv_head == 0
    assert q.shape[1] == head_dim
    assert k.shape == v.shape == (bkv_sz, head_dim)
    assert k.dtype == v.dtype
    head_l_ref = l_ref.at[kv_head_idx, : q.shape[0]]
    head_m_ref = m_ref.at[kv_head_idx, : q.shape[0]]
    head_acc_ref = acc_ref.at[kv_head_idx, : q.shape[0]]

    def load_with_init(ref, init_val):
      return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val), ref[...])

    # Follow FlashAttention-2 forward pass.
    s = (
        jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
        * sm_scale
    )
    q_span = (
        kv_len
        - q_len
        + bq_idx * bq_sz
        + lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
    )
    k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape, 1)
    mask = q_span < k_span
    # TODO(jevinjiang, xiowei): reduce pages_per_seq based on sliding_window.
    if sliding_window is not None:
      mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

    if soft_cap is not None:
      s = soft_cap * jnp.tanh(s / soft_cap)
    s += jnp.where(mask, mask_value, 0.0)
    s_rowmax = jnp.max(s, axis=1, keepdims=True)
    m_prev = load_with_init(head_m_ref, -jnp.inf)
    m_curr = jnp.maximum(m_prev, s_rowmax)
    head_m_ref[...] = m_curr
    p = jnp.exp(s - broadcast_minor(m_curr, s.shape))
    pv = jnp.einsum("nm,md->nd", p, v, preferred_element_type=jnp.float32)
    p_rowsum = jnp.sum(p, axis=1, keepdims=True)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    l_prev = load_with_init(head_l_ref, 0.0)
    l_curr = exp_m_diff * l_prev + p_rowsum
    head_l_ref[...] = l_curr
    o_prev = load_with_init(head_acc_ref, 0.0)
    o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
    head_acc_ref[...] = o_curr

  def async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
    sem = sems.at[0, bkv_sem_idx]
    vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

    hbm_shape = kv_cache_hbm_ref.shape
    hbm_ref = kv_cache_hbm_ref.reshape(
        hbm_shape[0] * hbm_shape[1], *hbm_shape[2:]
    )
    kv_len = kv_lens_ref[seq_idx]
    kv_len_start = bkv_idx * bkv_sz
    kv_p_start = bkv_idx * bkv_p
    kv_left = kv_len - kv_len_start

    def _fetch_bkv_effective_sz(i, _):
      sz = jnp.minimum(page_size, kv_left - i * page_size)
      async_copy(
          hbm_ref.at[
              pl.ds(
                  page_indices_ref[seq_idx * pages_per_seq + kv_p_start + i]
                  * page_size,
                  sz,
              )
          ],
          vmem_ref.at[pl.ds(i * page_size, sz)],
          sem,
          wait,
      )

    lax.fori_loop(
        0,
        jnp.minimum(cdiv(kv_left, page_size), bkv_p),
        _fetch_bkv_effective_sz,
        None,
        unroll=False,
    )

  def fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
    sem = sems.at[1, bq_sem_idx]
    vmem_ref = bq_x2_ref.at[bq_sem_idx]
    q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
    q_end = cu_q_lens_ref[seq_idx + 1]
    sz = jnp.minimum(bq_sz, q_end - q_len_start)
    async_copy(
        q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
        vmem_ref.at[:, pl.ds(0, sz)],
        sem,
        wait,
    )

  def send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
    sem = sems.at[2, bo_sem_idx]
    vmem_ref = bo_x2_ref.at[bo_sem_idx]
    q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
    q_end = cu_q_lens_ref[seq_idx + 1]
    sz = jnp.minimum(bq_sz, q_end - q_len_start)
    async_copy(
        vmem_ref.at[:, pl.ds(0, sz)],
        o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
        sem,
        wait,
    )

  def wait_bkv(seq_idx, bkv_idx, bkv_sem_idx):
    fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

  def wait_bq(seq_idx, bq_idx, bq_sem_idx):
    fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

  def wait_bo(seq_idx, bo_idx, bo_sem_idx):
    send_bo(seq_idx, bo_idx, bo_sem_idx, wait=True)

  def wait_bo_by_sem(bo_sem_idx):
    wip_seq_idx = bo_ids_ref[bo_sem_idx]
    wip_bo_idx = bo_ids_ref[bo_sem_idx + 2]

    @pl.when(jnp.logical_and(0 <= wip_seq_idx, wip_seq_idx <= seq_idx))
    def wait_wip_bo():
      wait_bo(wip_seq_idx, wip_bo_idx, bo_sem_idx)

  def load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz):
    q_ref = (
        bq_x2_ref.bitcast(jnp.uint32)
        .at[bq_sem_idx, kv_head_idx]
        .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
    )
    return pltpu.bitcast(
        q_ref[: actual_bq_sz * num_q_heads_per_kv_head_per_packing], q_dtype
    )

  def strided_load(ref, start, step, *, dtype=None):
    assert get_dtype_packing(ref.dtype) == 1
    assert len(ref.shape) == 2
    r, l = ref.shape
    assert l % 128 == 0
    folds = l // 128
    ref = ref.reshape(r * folds, 128)
    start *= folds
    step *= folds
    vec = jnp.concat([ref[start + i :: step] for i in range(folds)], axis=1)
    if dtype is not None:
      vec = pltpu.bitcast(vec, dtype)
    return vec

  def strided_load_bkv(bkv_sem_idx, start, step, *, bkv_bitmask):
    assert start % kv_packing == 0
    assert step % kv_packing == 0
    start //= kv_packing
    step //= kv_packing
    kv_ref = (
        bkv_x2_ref.bitcast(jnp.uint32)
        .at[bkv_sem_idx]
        .reshape(bkv_sz * step, head_dim)
    )

    def _mask_kv(k, v):
      k = pltpu.bitcast(k, jnp.uint32)
      v = pltpu.bitcast(v, jnp.uint32)
      k = k & bkv_bitmask
      v = v & bkv_bitmask
      k = pltpu.bitcast(k, kv_dtype)
      v = pltpu.bitcast(v, kv_dtype)
      return (k, v)

    def _dequantize_kv(k, v):
      if k_scale is not None:
        k = (k.astype(jnp.float32) * k_scale).astype(q_dtype)
      if v_scale is not None:
        v = (v.astype(jnp.float32) * v_scale).astype(q_dtype)
      return (k, v)

    if kv_packing == 1:
      k = strided_load(kv_ref, start, step, dtype=kv_dtype)
      v = strided_load(kv_ref, start + 1, step, dtype=kv_dtype)
      return [_dequantize_kv(*_mask_kv(k, v))]

    kv = strided_load(kv_ref, start, step)
    bitwidth = 32 // kv_packing
    repack_ty = jnp.dtype(f"uint{bitwidth}")
    lst = []
    for i in range(0, kv_packing, 2):
      k = (kv >> (i * bitwidth)).astype(repack_ty)
      v = (kv >> ((i + 1) * bitwidth)).astype(repack_ty)
      lst.append(_dequantize_kv(*_mask_kv(k, v)))
    return lst

  def broadcast_minor(src, shape):
    if src.shape == shape:
      return src
    assert src.shape[:-1] == shape[:-1]
    if shape[-1] < src.shape[-1]:
      return src[..., : shape[-1]]
    assert shape[-1] % src.shape[-1] == 0, f"{shape=} % {src.shape=}"
    # no-op concatenation.
    return jnp.concatenate(
        [src for _ in range(shape[-1] // src.shape[-1])], axis=-1
    )

  def process(static_q_len=None):
    num_bkv = cdiv(kv_len, bkv_sz)
    if static_q_len is None:
      num_bq = cdiv(q_len, bq_sz)
      actual_bq_sz = bq_sz
      is_static_q_loop = False
    else:
      num_bq = cdiv(static_q_len, bq_sz)
      actual_bq_sz = min(bq_sz, static_q_len)
      is_static_q_loop = True

    def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
      next_bq_idx = bq_idx + 1
      is_last_bq = next_bq_idx == num_bq
      next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
      next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
      next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
      return next_seq_idx, next_bq_idx, next_bq_sem_idx

    def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
      next_bkv_idx = bkv_idx + 1
      is_last_bkv = next_bkv_idx == num_bkv
      next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
      next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
      is_last_bq = next_bq_idx == num_bq
      next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
      next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
      next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
      return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

    def compute_with_bq(bq_idx, _):
      bq_sem_idx = sem_ids_ref[0]
      next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
          seq_idx, bq_idx, bq_sem_idx
      )

      # Prefetch next bq
      @pl.when(next_seq_idx < num_seqs)
      def prefetch_next_bq():
        sem_ids_ref[0] = next_bq_sem_idx
        fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

      def compute_with_bkv(bkv_idx, _):
        # Create bitmask for KV.
        assert bkv_sz % kv_packing == 0
        actual_bkv_sz = jnp.minimum(bkv_sz, kv_len - bkv_idx * bkv_sz)
        bkv_shape = (bkv_sz, head_dim)
        bkv_mask = lax.broadcasted_iota(jnp.int32, bkv_shape, 0) < actual_bkv_sz
        bkv_bitmask = pltpu.bitcast(
            lax.select(
                bkv_mask,
                jnp.full(bkv_shape, 0xFFFFFFFF, dtype=jnp.uint32),
                jnp.full(bkv_shape, 0, dtype=jnp.uint32),
            ).astype(jnp.dtype(f"uint{32 // kv_packing}")),
            jnp.uint32,
        )

        # Get next bkv ids.
        bkv_sem_idx = sem_ids_ref[1]
        next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
            seq_idx, bq_idx, bkv_idx, bkv_sem_idx
        )

        # Prefetch next bkv
        @pl.when(next_seq_idx < num_seqs)
        def prefetch_next_bkv():
          sem_ids_ref[1] = next_bkv_sem_idx
          fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

        # Wait for cur bq if not ready yet
        @pl.when(bkv_idx == 0)
        def wait_cur_bq():
          wait_bq(seq_idx, bq_idx, bq_sem_idx)

        # Wait for cur bkv
        wait_bkv(seq_idx, bkv_idx, bkv_sem_idx)

        # Flash attention with cur bkv and bq
        # NOTE: kv_packing is divided by 2 because k and v are packed together.
        heads_per_load = max(1, kv_packing // 2)
        for kv_head_start in range(0, actual_num_kv_heads, heads_per_load):
          bkv_lst = strided_load_bkv(
              bkv_sem_idx,
              kv_head_start * 2,
              num_kv_heads_x2,
              bkv_bitmask=bkv_bitmask,
          )
          assert len(bkv_lst) == heads_per_load
          for i in range(heads_per_load):
            kv_head_idx = kv_head_start + i
            if kv_head_idx >= actual_num_kv_heads:
              break
            bq = load_bq(bq_sem_idx, kv_head_idx, actual_bq_sz=actual_bq_sz)
            bk, bv = bkv_lst[i]
            flash_attention(
                bq,
                bk,
                bv,
                bq_idx=bq_idx,
                bkv_idx=bkv_idx,
                kv_head_idx=kv_head_idx,
            )

      lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

      # Load acc and calculate final output.
      acc = acc_ref[...]
      l = broadcast_minor(l_ref[...], acc.shape)
      out = (
          lax.div(acc, l)
          if q_dtype == jnp.float32
          else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)
      )

      # Wait for previous bo to be fully sent before storing new bo.
      bo_sem_idx = sem_ids_ref[2]
      wait_bo_by_sem(bo_sem_idx)

      # Store output from acc to bo.
      bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
          actual_num_kv_heads,
          bq_sz * num_q_heads_per_kv_head_per_packing,
          head_dim,
      )[...] = pltpu.bitcast(out, jnp.int32)

      # Send cur bo
      bo_ids_ref[bo_sem_idx] = seq_idx
      bo_ids_ref[bo_sem_idx + 2] = bq_idx
      sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
      send_bo(seq_idx, bq_idx, bo_sem_idx)

    lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=is_static_q_loop)

  ### ------- Kernel start ------- ###

  @pl.when(seq_idx == 0)
  def prefetch_first_blks():
    fetch_bq(0, 0, 0)
    fetch_bkv(0, 0, 0)

  @pl.when(seq_idx < decode_end)
  def process_decode():
    process(static_q_len=1)

  @pl.when(jnp.logical_and(decode_end <= seq_idx, seq_idx < prefill_end))
  def process_prefill():
    process(static_q_len=chunk_prefill_size)

  @pl.when(jnp.logical_and(prefill_end <= seq_idx, seq_idx < mixed_end))
  def process_mixed():
    process()

  @pl.when(seq_idx == num_seqs - 1)
  def wait_all_bo():
    for i in range(2):
      wait_bo_by_sem(i)

  ### ------- Kernel end ------- ###


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
    actual_num_kv_heads: int,
):
  """Transform inputs to the desired shape for the RPA kernel.

  From
    q: [max_num_tokens, actual_num_q_heads, actual_head_dim]

  Return
    transformed_q: [max_num_tokens, actual_num_kv_heads,
    num_q_heads_per_kv_head // q_packing, q_packing, head_dim],
    actual_num_q_heads_per_kv_head,
    actual_head_dim
  """
  max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
  assert actual_num_q_heads % actual_num_kv_heads == 0
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
  q_packing = get_dtype_packing(q.dtype)
  num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
  head_dim = align_to(actual_head_dim, 128)
  q = (
      jnp.pad(
          q.reshape(
              max_num_tokens,
              actual_num_kv_heads,
              actual_num_q_heads_per_kv_head,
              actual_head_dim,
          ),
          (
              (0, 0),
              (0, 0),
              (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
              (0, head_dim - actual_head_dim),
          ),
          constant_values=0,
      )
      .reshape(
          max_num_tokens,
          actual_num_kv_heads,
          num_q_heads_per_kv_head // q_packing,
          q_packing,
          head_dim,
      )
      .swapaxes(0, 1)
  )
  return q, actual_num_q_heads_per_kv_head, actual_head_dim


def prepare_outputs(
    out,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
):
  """Transform the output of the RPA kernel to the desired shape.

  Input
    out: [actual_num_kv_heads, max_num_tokens,
    num_q_heads_per_kv_head // q_packing,
    q_packing, head_dim]

  Return
    transformed_out: [max_num_tokens, actual_num_q_heads, actual_head_dim]
  """
  (
      actual_num_kv_heads,
      max_num_tokens,
      num_q_heads_per_kv_head_per_q_packing,
      q_packing,
      head_dim,
  ) = out.shape
  actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
  return (
      out.swapaxes(0, 1)
      .reshape(
          max_num_tokens,
          actual_num_kv_heads,
          num_q_heads_per_kv_head_per_q_packing * q_packing,
          head_dim,
      )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim]
      .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
  )


# Expect to run this validation during runtime.
def dynamic_validate_inputs(
    q: jax.Array,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  static_validate_inputs(
      q,
      kv_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      chunk_prefill_size=chunk_prefill_size,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  max_num_tokens = q.shape[1]
  total_num_pages = kv_cache.shape[0]
  page_size = kv_cache.shape[1]
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs

  i, j, k = distribution
  if not (i <= j <= k):
    raise ValueError(f"Invalid distribution: {distribution=}")

  if k > max_num_seqs:
    raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")

  if cu_q_lens[k] > max_num_tokens:
    raise ValueError(
        f"Total q tokens {cu_q_lens[k]} must be <= {max_num_tokens=}."
    )
  for i in range(k):
    q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    kv_len = kv_lens[i]
    if not (0 < q_len <= kv_len):
      raise ValueError(f"Require 0 < {q_len=} <= {kv_len=} at sequence {i}.")
    page_cnt = cdiv(kv_len, page_size)
    if page_cnt > pages_per_seq:
      raise ValueError(
          f"Require {page_cnt=} <= {pages_per_seq=} at sequence {i} where"
          f" {kv_len=} and {page_size=}."
      )
    for p in range(page_cnt):
      page_idx = page_indices[i * pages_per_seq + p]
      if not (0 <= page_idx < total_num_pages):
        raise ValueError(
            f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence"
            f" {i} where {kv_len=} and {page_size=}."
        )


# Expect to run this validation during compile time.
def static_validate_inputs(
    q: jax.Array,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Validate inputs to the RPA kernel statically."""
  (
      actual_num_kv_heads,
      _,
      _,
      q_packing,
      head_dim,
  ) = q.shape
  (
      _,
      page_size,
      num_kv_heads_x2_per_kv_packing,
      kv_packing,
      _,
  ) = kv_cache.shape

  if kv_cache.shape[-1] != head_dim:
    raise ValueError(f"{kv_cache.shape[-1]=} must be equal to {head_dim=}")
  if head_dim % 128 != 0:
    raise ValueError(f"{head_dim=} must be divisible by 128.")
  if q_packing != get_dtype_packing(q.dtype):
    raise ValueError(f"{q_packing=} does not match with {q.dtype=}")
  if kv_packing != get_dtype_packing(kv_cache.dtype):
    raise ValueError(f"{kv_packing=} does not match with {kv_cache.dtype=}")

  num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
  if num_kv_heads_x2 % 2 != 0:
    raise ValueError(
        f"Combined KV heads must be divisible by 2, but got {num_kv_heads_x2}"
    )
  if align_to(actual_num_kv_heads * 2, kv_packing) != num_kv_heads_x2:
    raise ValueError(
        f"Invalid {num_kv_heads_x2=}, {actual_num_kv_heads=}, {kv_packing=}"
    )

  if not (
      jnp.int32
      == kv_lens.dtype
      == page_indices.dtype
      == cu_q_lens.dtype
      == distribution.dtype
  ):
    raise ValueError(
        f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}, {distribution.dtype=}"
    )

  if not (
      len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1
  ):
    raise ValueError(
        f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
        f" {cu_q_lens.shape=}"
    )

  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  if num_page_indices % max_num_seqs != 0:
    raise ValueError(
        f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
    )
  if distribution.shape != (3,):
    raise ValueError(f"Expected {distribution.shape=} to be (3,).")

  if page_size % kv_packing != 0:
    raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
  if sliding_window is not None and sliding_window <= 0:
    raise ValueError(f"{sliding_window=} must be positive.")
  if soft_cap is not None and soft_cap == 0.0:
    raise ValueError(f"{soft_cap=} must not be 0.0.")
  if chunk_prefill_size is not None and chunk_prefill_size <= 0:
    raise ValueError(f"{chunk_prefill_size=} must be positive.")
  if num_kv_pages_per_block is not None:
    if num_kv_pages_per_block <= 0:
      raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
  if num_queries_per_block is not None:
    if num_queries_per_block <= 0:
      raise ValueError(f"{num_queries_per_block=} must be positive.")
  if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
    raise ValueError(f"{vmem_limit_bytes=} must be positive.")

  # No constraints for the following inputs.
  del sm_scale
  del mask_value
  del k_scale
  del v_scale


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
)
def ragged_paged_attention(
    q: jax.Array,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    # TODO(jevinjiang, chengjiyao): fuse ragged kv scatter to RPA kernel!
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences' queries.
    kv_cache: paged KV cache with TPU-friendly shape.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  static_validate_inputs(
      q,
      kv_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      k_scale=k_scale,
      v_scale=v_scale,
      chunk_prefill_size=chunk_prefill_size,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  (
      actual_num_kv_heads,
      max_num_tokens,
      num_q_heads_per_kv_head_per_q_packing,
      q_packing,
      head_dim,
  ) = q.shape
  page_size = kv_cache.shape[1]
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing
  actual_num_q_heads = num_q_heads_per_kv_head * actual_num_kv_heads

  bkv_p = num_kv_pages_per_block
  bq_sz = num_queries_per_block
  if bq_sz is None or bkv_p is None:
    bkv_p, bq_sz = get_tuned_block_sizes(
        q.dtype,
        kv_cache.dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size,
        max_num_tokens,
        pages_per_seq,
    )

  bkv_sz = bkv_p * page_size
  grid = (distribution[2],)

  in_specs = [
      pl.BlockSpec(memory_space=pltpu.HBM),
      pl.BlockSpec(memory_space=pltpu.HBM),
  ]

  out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

  bkv_double_buf = pltpu.VMEM(
      (2, bkv_sz, *kv_cache.shape[2:]),
      kv_cache.dtype,
  )

  bq_double_buf = pltpu.VMEM(
      (2, actual_num_kv_heads, bq_sz, *q.shape[2:]),
      q.dtype,
  )

  bo_double_buf = bq_double_buf

  l_scratch = pltpu.VMEM(
      (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )
  m_scratch = l_scratch

  acc_scratch = pltpu.VMEM(
      (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
      jnp.float32,
  )

  scratch_shapes = [
      bkv_double_buf,  # Double buffering for kv block.
      bq_double_buf,  # Double buffering for q block.
      bo_double_buf,  # Double buffering for output block.
      # Semaphores for double buffering of bkv, bq, bo.
      pltpu.SemaphoreType.DMA((3, 2)),
      # Intermediate buffers per kv head for flash attention.
      l_scratch,
      m_scratch,
      acc_scratch,
  ]

  scalar_prefetches = (
      kv_lens,
      # TODO(jevinjiang): can we use ragged page_indices to save some smem?
      page_indices,
      cu_q_lens,
      distribution,
      # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
      jnp.zeros((3,), jnp.int32),
      # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
      jnp.full((4,), -1, jnp.int32),
  )

  scope_name = f"RPA-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
  kernel = jax.named_scope(scope_name)(
      pl.pallas_call(
          functools.partial(
              _ragged_paged_attention_kernel,
              sm_scale=sm_scale,
              sliding_window=sliding_window,
              soft_cap=soft_cap,
              mask_value=mask_value,
              k_scale=k_scale,
              v_scale=v_scale,
              chunk_prefill_size=chunk_prefill_size,
              bq_sz=bq_sz,
              bkv_p=bkv_p,
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=len(scalar_prefetches),
              in_specs=in_specs,
              out_specs=out_specs,
              grid=grid,
              scratch_shapes=scratch_shapes,
          ),
          compiler_params=pltpu.CompilerParams(
              # TODO(jevinjiang): since each sequence depends on the previous
              # one, we need some extra work to support Megacore mode.
              dimension_semantics=("arbitrary",),
              vmem_limit_bytes=vmem_limit_bytes,
              # disable_bounds_checks=True,
          ),
          out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
          name=scope_name,
      )
  )

  output = kernel(*scalar_prefetches, q, kv_cache)
  return output
