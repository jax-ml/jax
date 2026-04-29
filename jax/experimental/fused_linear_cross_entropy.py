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

"""Memory-efficient fused linear cross-entropy loss.

Implements the algorithm from "Cut Your Losses in Large-Vocabulary Language
Models" (https://arxiv.org/abs/2411.09009), which avoids materializing the
full [N, V] logits matrix by processing the vocabulary in chunks via
``lax.scan``.

Typical LLM training:

  logits = hidden @ weight.T   # shape [N, V] — multi-GB for 128K+ vocab
  loss   = cross_entropy(logits, labels)

This module replaces those two steps with a single call that never allocates
the [N, V] tensor.  Memory complexity is O(N·D + V·D) instead of O(N·V).
"""
from __future__ import annotations

import functools

import jax
from jax import lax
import jax.numpy as jnp


def fused_linear_cross_entropy_loss(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array,
    *,
    chunk_size: int = 4096,
    shift: int = 0,
    vocab_sort_indices: jax.Array | None = None,
    filter_eps: float | None = None,
) -> jax.Array:
  """Memory-efficient fused linear + cross-entropy loss.

  Computes ``mean(cross_entropy(x @ w.T, labels))`` without materializing the
  full ``[N, V]`` logits matrix.  The vocabulary dimension is processed in
  chunks of size *chunk_size* using ``lax.scan``, so peak memory usage is
  proportional to ``chunk_size`` rather than to ``V``.

  The backward pass is implemented via ``jax.custom_vjp``.  Only ``x``, ``w``,
  the per-sample log-sum-exp vector (shape ``[N]``), and the per-sample
  global-max vector (shape ``[N]``) are saved as residuals; the full logits
  are recomputed chunk by chunk during the backward pass.

  Compatible with ``jax.jit``, ``jax.grad``, and ``jax.value_and_grad``.
  Both ``float32`` and ``bfloat16`` inputs are supported; internal
  accumulation is always performed in ``float32``.

  Args:
    x: Input activations.  Shape ``[N, D]``, or a batched sequence
      ``[..., T, D]`` when *shift* > 0.
    w: Vocabulary weight matrix, shape ``[V, D]``.
    labels: Integer class labels in ``[0, V)``.  Shape ``[N]``, or
      ``[..., T]`` when *shift* > 0.
    chunk_size: Number of vocabulary entries processed per scan step.
      Must be a positive Python ``int``.  Larger values trade memory for
      fewer kernel launches; defaults to ``4096``.
    shift: When > 0, apply a causal sequence shift before computing the
      loss.  Specifically, ``x[..., :-shift, :]`` is used to predict
      ``labels[..., shift:]``, both reshaped to ``[-1, D]`` / ``[-1]``.
      ``shift=1`` implements standard next-token prediction.  Defaults to
      ``0`` (no shift).
    vocab_sort_indices: Optional integer array of shape ``[V]``.  When
      provided, weight rows are reordered as ``w[vocab_sort_indices]``
      before chunking, grouping related tokens for better cache locality.
      Labels are remapped from original to sorted vocabulary space
      internally so the loss value is identical to the unsorted case.
      JAX's gather VJP automatically scatters ``dw`` back to the original
      (unsorted) row order.  Defaults to ``None`` (no reordering).
    filter_eps: Optional float threshold for gradient filtering.  When
      set, chunks whose per-sample maximum logit falls more than
      *filter_eps* below the global per-sample maximum are zeroed out in
      the backward pass (their ``softmax`` contribution is negligible by
      construction).  The forward loss is always exact regardless of this
      setting.  Use ``float('inf')`` to request the filter code path with
      no actual filtering (useful for benchmarking).  Defaults to
      ``None`` (no filtering).

  Returns:
    Scalar mean cross-entropy loss in ``float32``.

  References:
    "Cut Your Losses in Large-Vocabulary Language Models",
    arxiv.org/abs/2411.09009

    Apple CCE reference implementation:
    github.com/apple/ml-cross-entropy
  """
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}")
  if shift < 0:
    raise ValueError(f"shift must be non-negative, got {shift}")

  # --- Feature 1: causal sequence shift -----------------------------------
  # x[t] predicts labels[t + shift].  Slice off the unmatched prefix/suffix
  # then flatten the leading dimensions so the core function sees [N, D].
  if shift > 0:
    x = x[..., :-shift, :].reshape(-1, x.shape[-1])
    labels = labels[..., shift:].reshape(-1)

  # --- Feature 2: vocab sorting -------------------------------------------
  # Reorder weight rows so that frequently-used tokens cluster together,
  # improving cache locality during chunk matmuls and enabling filter_eps
  # to skip more chunks in the backward pass.
  #
  # Labels are remapped from original→sorted vocab space so that the loss
  # value is bit-for-bit identical to the unsorted computation.
  #
  # Gradient unsort is free: JAX's lax.gather VJP uses lax.scatter_add to
  # map dw_sorted[i] → dw_original[vocab_sort_indices[i]], which is the
  # correct inverse permutation.
  if vocab_sort_indices is not None:
    w = w[vocab_sort_indices]
    rank = jnp.argsort(vocab_sort_indices)   # rank[k] = sorted pos of token k
    labels = rank[labels]

  return _fused_linear_cross_entropy(x, w, labels, chunk_size, filter_eps)


# ---------------------------------------------------------------------------
# Memory note: buffer donation / in-place gradient accumulation
#
# The CCE Triton kernel (github.com/apple/ml-cross-entropy) overwrites the
# forward logit buffer with gradient values in-place, so each chunk needs
# only one O(N·C) allocation that is reused for both passes.
#
# JAX's functional model does not permit in-place mutation inside
# custom_vjp.  However, XLA's buffer-assignment pass performs liveness
# analysis on the compiled HLO and will automatically reuse the memory of
# dead buffers — including consumed residuals — when they fall out of scope,
# so peak allocation is often comparable without any user intervention.
#
# At the JIT boundary, callers may additionally pass donate_argnums to
# jax.jit to donate the input arrays' backing buffers to outputs:
#
#   train_step = jax.jit(train_step, donate_argnums=(0,))  # donate params
#
# This lets XLA overwrite e.g. the old parameter buffer with the updated
# parameters in a gradient-descent step, cutting one full-model allocation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_w(w: jax.Array, chunk_size: int) -> tuple[jax.Array, int, int]:
  """Return (w_chunks, num_chunks, padded_V).

  w_chunks has shape [num_chunks, chunk_size, D].
  Padding rows are zero so they contribute nothing to dot products.
  """
  V, D = w.shape
  num_chunks = (V + chunk_size - 1) // chunk_size
  padded_V = num_chunks * chunk_size
  if padded_V > V:
    w = jnp.concatenate(
        [w, jnp.zeros((padded_V - V, D), dtype=w.dtype)], axis=0
    )
  return w.reshape(num_chunks, chunk_size, D), num_chunks, padded_V


def _valid_mask(V: int, num_chunks: int, chunk_size: int) -> jax.Array:
  """Boolean mask [num_chunks, chunk_size]: True for real vocab entries."""
  return jnp.arange(num_chunks * chunk_size).reshape(num_chunks, chunk_size) < V


def _compute_lse(
    x: jax.Array, w: jax.Array, chunk_size: int
) -> tuple[jax.Array, jax.Array]:
  """Compute log-sum-exp of ``x @ w.T`` via online softmax.

  Returns ``(lse, global_max)`` both of shape ``[N]`` float32, without ever
  allocating ``[N, V]``.
  """
  N = x.shape[0]
  V = w.shape[0]
  w_chunks, num_chunks, _ = _pad_w(w, chunk_size)
  mask = _valid_mask(V, num_chunks, chunk_size)  # [num_chunks, chunk_size]

  x_f32 = x.astype(jnp.float32)
  w_chunks_f32 = w_chunks.astype(jnp.float32)

  def scan_fn(
      carry: tuple[jax.Array, jax.Array],
      inputs: tuple[jax.Array, jax.Array],
  ) -> tuple[tuple[jax.Array, jax.Array], None]:
    running_max, running_sum_exp = carry  # both [N]
    chunk_w, chunk_mask = inputs          # [C, D], [C]

    logits_chunk = x_f32 @ chunk_w.T     # [N, C]
    # Mask padding to -inf so padded positions don't affect max or sum.
    logits_chunk = jnp.where(chunk_mask[None, :], logits_chunk, -jnp.inf)

    chunk_max = jnp.max(logits_chunk, axis=-1)  # [N]
    new_max = jnp.maximum(running_max, chunk_max)  # [N]

    # Rescale accumulated sum to the new max, then add current chunk.
    rescale = jnp.exp(running_max - new_max)       # [N]  (0 when running_max=-inf)
    exp_chunk = jnp.where(
        chunk_mask[None, :],
        jnp.exp(logits_chunk - new_max[:, None]),
        0.0,
    )  # [N, C]
    new_sum_exp = running_sum_exp * rescale + jnp.sum(exp_chunk, axis=-1)

    return (new_max, new_sum_exp), None

  init = (
      jnp.full((N,), -jnp.inf, dtype=jnp.float32),
      jnp.zeros((N,), dtype=jnp.float32),
  )
  (global_max, final_sum_exp), _ = lax.scan(
      scan_fn, init, (w_chunks_f32, mask)
  )
  lse = global_max + jnp.log(final_sum_exp)  # [N]
  return lse, global_max


# ---------------------------------------------------------------------------
# custom_vjp implementation
# nondiff_argnums=(3, 4): chunk_size (int) and filter_eps (float | None)
# are static Python values; JAX retraces when they change.
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _fused_linear_cross_entropy(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array,
    chunk_size: int,
    filter_eps: float | None,
) -> jax.Array:
  """Primal: used when gradients are not required."""
  lse, _ = _compute_lse(x, w, chunk_size)
  target_logit = jnp.sum(
      x.astype(jnp.float32) * w[labels].astype(jnp.float32), axis=-1
  )  # [N]
  return jnp.mean(-target_logit + lse)


def _fused_linear_cross_entropy_fwd(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array,
    chunk_size: int,
    filter_eps: float | None,
) -> tuple[jax.Array, tuple]:
  """Forward pass: same output as primal, plus residuals for backward."""
  lse, global_max = _compute_lse(x, w, chunk_size)
  target_logit = jnp.sum(
      x.astype(jnp.float32) * w[labels].astype(jnp.float32), axis=-1
  )
  loss = jnp.mean(-target_logit + lse)
  # Save only x, w, labels, lse, global_max — never the full [N, V] logits.
  return loss, (x, w, labels, lse, global_max)


def _fused_linear_cross_entropy_bwd(
    chunk_size: int,
    filter_eps: float | None,
    residuals: tuple,
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  """Backward pass: chunked recomputation of logits, no [N, V] allocation.

  --- Feature 3: gradient filtering (filter_eps) ---
  When filter_eps is set, chunks whose per-sample maximum logit falls more
  than filter_eps below the global per-sample maximum are zeroed out.
  Their softmax contribution is exp(chunk_max - global_max) < exp(-filter_eps),
  which is negligible for large filter_eps values.

  Because lax.scan requires uniform computation per step, we multiply by 0
  (masking) rather than truly branching.  The matmul still executes, but its
  results are discarded for filtered samples/chunks.  The one-hot correction
  term is preserved even for filtered chunks so that labels in low-probability
  chunks still receive the correct −1/N gradient signal.
  """
  x, w, labels, lse, global_max = residuals
  N, D = x.shape
  V = w.shape[0]

  w_chunks, num_chunks, padded_V = _pad_w(w, chunk_size)
  mask = _valid_mask(V, num_chunks, chunk_size)         # [num_chunks, chunk_size]
  chunk_starts = jnp.arange(num_chunks, dtype=labels.dtype) * chunk_size

  x_f32 = x.astype(jnp.float32)
  w_chunks_f32 = w_chunks.astype(jnp.float32)
  lse_f32 = lse.astype(jnp.float32)
  global_max_f32 = global_max.astype(jnp.float32)      # [N], always cheap to keep
  # Scale by upstream gradient and 1/N (gradient through jnp.mean).
  g_scale = g.astype(jnp.float32) / N  # scalar

  def bwd_chunk(
      dx_acc: jax.Array,
      inputs: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    chunk_w, chunk_start, chunk_mask = inputs  # [C, D], [], [C]

    # Recompute softmax for this vocab chunk — O(N*C) not O(N*V).
    logits_chunk = x_f32 @ chunk_w.T            # [N, C]
    # Use masked logits so padded slots don't pollute chunk_max.
    logits_masked = jnp.where(chunk_mask[None, :], logits_chunk, -jnp.inf)
    softmax_chunk = jnp.exp(logits_masked - lse_f32[:, None])  # [N, C]
    # Zero out padded positions (exp(0 - lse) is nonzero for zero-padded w).
    softmax_chunk = jnp.where(chunk_mask[None, :], softmax_chunk, 0.0)

    # --- gradient filtering (filter_eps) ---
    # Evaluated at Python level (filter_eps is a nondiff arg), so the
    # filtering ops are only included in the compiled graph when needed.
    if filter_eps is not None:
      chunk_max = jnp.max(logits_masked, axis=-1)          # [N]
      active = chunk_max >= global_max_f32 - filter_eps    # [N] bool
      softmax_chunk = jnp.where(active[:, None], softmax_chunk, 0.0)
      # Note: we do NOT filter the one_hot below.  Even if the softmax
      # contribution is negligible, the label correction term (−1/N) must
      # still be applied so that samples with labels in low-logit chunks
      # receive the correct gradient signal.

    # Build sparse one-hot for labels that fall within [chunk_start, chunk_start+C).
    local_idx = labels - chunk_start            # [N]  (may be negative/out-of-range)
    in_chunk = (local_idx >= 0) & (local_idx < chunk_size)  # [N]
    safe_idx = jnp.where(in_chunk, local_idx, 0)            # [N]  safe for gather

    one_hot = (
        (jnp.arange(chunk_size, dtype=labels.dtype)[None, :] == safe_idx[:, None])
        & in_chunk[:, None]
    ).astype(jnp.float32)  # [N, C]

    # dL/d(logits[i,k]) = (softmax[i,k] - 1[k==labels[i]]) * g / N
    dlogits = (softmax_chunk - one_hot) * g_scale  # [N, C]

    dx_acc = dx_acc + dlogits @ chunk_w          # [N, D]
    dw_chunk = dlogits.T @ x_f32                 # [C, D]
    # Ensure no gradient flows into padding rows.
    dw_chunk = jnp.where(chunk_mask[:, None], dw_chunk, 0.0)

    return dx_acc, dw_chunk

  dx_init = jnp.zeros((N, D), dtype=jnp.float32)
  dx, dw_chunks = lax.scan(
      bwd_chunk, dx_init, (w_chunks_f32, chunk_starts, mask)
  )
  # dw_chunks: [num_chunks, chunk_size, D] → [padded_V, D] → [V, D]
  dw = dw_chunks.reshape(padded_V, D)[:V]

  return dx.astype(x.dtype), dw.astype(w.dtype), None  # None: no grad for labels


_fused_linear_cross_entropy.defvjp(
    _fused_linear_cross_entropy_fwd,
    _fused_linear_cross_entropy_bwd,
)
