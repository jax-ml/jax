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
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp


def fused_linear_cross_entropy_loss(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array,
    *,
    chunk_size: int = 4096,
) -> jax.Array:
  """Memory-efficient fused linear + cross-entropy loss.

  Computes ``mean(cross_entropy(x @ w.T, labels))`` without materializing the
  full ``[N, V]`` logits matrix.  The vocabulary dimension is processed in
  chunks of size *chunk_size* using ``lax.scan``, so peak memory usage is
  proportional to ``chunk_size`` rather than to ``V``.

  The backward pass is implemented via ``jax.custom_vjp``.  Only ``x``, ``w``
  and the per-sample log-sum-exp vector (shape ``[N]``) are saved as
  residuals; the full logits are recomputed chunk by chunk during the backward
  pass.

  Compatible with ``jax.jit``, ``jax.grad``, and ``jax.value_and_grad``.
  Both ``float32`` and ``bfloat16`` inputs are supported; internal
  accumulation is always performed in ``float32``.

  Args:
    x: Input activations, shape ``[N, D]``.
    w: Vocabulary weight matrix, shape ``[V, D]``.
    labels: Integer class labels in ``[0, V)``, shape ``[N]``.
    chunk_size: Number of vocabulary entries processed per scan step.
      Must be a positive Python ``int``.  Larger values trade memory for
      fewer kernel launches; defaults to ``4096``.

  Returns:
    Scalar mean cross-entropy loss.  The dtype matches ``float32``
    regardless of the input dtype (consistent with numerically-stable loss
    computations in mixed-precision training).

  References:
    "Cut Your Losses in Large-Vocabulary Language Models",
    arxiv.org/abs/2411.09009
  """
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}")
  return _fused_linear_cross_entropy(x, w, labels, chunk_size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_w(w: jax.Array, chunk_size: int) -> tuple[jax.Array, int, int]:
  """Return (w_padded, num_chunks, padded_V).

  w_padded is reshaped to [num_chunks, chunk_size, D].
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
) -> jax.Array:
  """Compute log-sum-exp of ``x @ w.T`` via online softmax.

  Returns shape ``[N]`` float32, without ever allocating ``[N, V]``.
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
    # Mask padding to -inf so padded positions don't affect the max or sum.
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
  (final_max, final_sum_exp), _ = lax.scan(
      scan_fn, init, (w_chunks_f32, mask)
  )
  return final_max + jnp.log(final_sum_exp)  # [N]


# ---------------------------------------------------------------------------
# custom_vjp implementation
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fused_linear_cross_entropy(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array,
    chunk_size: int,
) -> jax.Array:
  """Primal: used when gradients are not required."""
  lse = _compute_lse(x, w, chunk_size)
  target_logit = jnp.sum(
      x.astype(jnp.float32) * w[labels].astype(jnp.float32), axis=-1
  )  # [N]
  return jnp.mean(-target_logit + lse)


def _fused_linear_cross_entropy_fwd(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, tuple]:
  """Forward pass: same output as primal, plus residuals for backward."""
  lse = _compute_lse(x, w, chunk_size)
  target_logit = jnp.sum(
      x.astype(jnp.float32) * w[labels].astype(jnp.float32), axis=-1
  )
  loss = jnp.mean(-target_logit + lse)
  # Save only x, w, labels, lse — never the full [N, V] logits.
  return loss, (x, w, labels, lse)


def _fused_linear_cross_entropy_bwd(
    chunk_size: int,
    residuals: tuple,
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  """Backward pass: chunked recomputation of logits, no [N, V] allocation."""
  x, w, labels, lse = residuals
  N, D = x.shape
  V = w.shape[0]

  w_chunks, num_chunks, padded_V = _pad_w(w, chunk_size)
  mask = _valid_mask(V, num_chunks, chunk_size)         # [num_chunks, chunk_size]
  chunk_starts = jnp.arange(num_chunks, dtype=labels.dtype) * chunk_size

  x_f32 = x.astype(jnp.float32)
  w_chunks_f32 = w_chunks.astype(jnp.float32)
  lse_f32 = lse.astype(jnp.float32)
  # Scale by upstream gradient and 1/N (gradient through jnp.mean).
  g_scale = g.astype(jnp.float32) / N  # scalar

  def bwd_chunk(
      dx_acc: jax.Array,
      inputs: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    chunk_w, chunk_start, chunk_mask = inputs  # [C, D], [], [C]

    # Recompute softmax for this vocab chunk — O(N*C) not O(N*V).
    logits_chunk = x_f32 @ chunk_w.T            # [N, C]
    softmax_chunk = jnp.exp(logits_chunk - lse_f32[:, None])  # [N, C]
    # Zero out padded positions (their w rows are zero, giving logit=0, but
    # exp(0 - lse) is nonzero so we must mask explicitly).
    softmax_chunk = jnp.where(chunk_mask[None, :], softmax_chunk, 0.0)

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
