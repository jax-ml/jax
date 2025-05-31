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

"""JAX reference implementation of grouped query attention."""

import jax
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp

MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def grouped_query_attention_reference(
    queries: jax.Array,  # [batch_size, num_q_heads, head_dim]
    k_pages: jax.Array,  # [batch_size, num_kv_heads, max_seq_len, head_dim]
    v_pages: jax.Array,  # [batch_size, num_kv_heads, max_seq_len, head_dim]
    seq_lens: jax.Array,  # i32[batch_size]
    soft_cap: float | None = None,
    debug: bool = False,
) -> jax.Array:  # [batch_size, num_q_heads, head_dim]
  """Grouped query attention with a single query per request."""
  # Check input shapes
  assert k_pages.shape == v_pages.shape
  batch_size, num_q_heads, head_dim = queries.shape
  batch_size2, num_kv_heads, max_seq_len, head_dim2 = k_pages.shape
  assert batch_size2 == batch_size
  assert head_dim2 == head_dim

  # Unquantize kv pages if necessary
  if isinstance(k_pages, quantization_utils.QuantizedTensor):
    k_pages = quantization_utils.unquantize_from_int8(
        k_pages, dtype=jnp.float32
    )
  if isinstance(v_pages, quantization_utils.QuantizedTensor):
    v_pages = quantization_utils.unquantize_from_int8(
        v_pages, dtype=jnp.float32
    )

  # Reshape for num_groups queries per k head
  assert num_q_heads % num_kv_heads == 0
  num_groups = num_q_heads // num_kv_heads
  queries = queries.reshape(batch_size, num_kv_heads, num_groups, head_dim)

  # Compute the dot product q*k and apply soft cap if necessary
  qk = jnp.einsum(
      "bhgd,bhtd->bhgt",
      queries.astype(jnp.float32),
      k_pages.astype(jnp.float32),
  )
  if soft_cap is not None and soft_cap != 0.0:
    qk = jnp.tanh(qk / soft_cap) * soft_cap
  assert qk.shape == (batch_size, num_kv_heads, num_groups, max_seq_len)
  if debug:
    jax.debug.print("qk: {qk}", qk=qk)

  # Enforce causal mask (adding dimensions when necessary)
  mask = jnp.arange(max_seq_len)[None] < seq_lens[:, None]
  qk += jnp.where(mask, 0.0, MASK_VALUE)[:, None, None, :]
  if debug:
    jax.debug.print("masked: {qk}", qk=qk)

  # Generate probability distribution using softmax
  probs = jax.nn.softmax(qk, axis=-1).astype(v_pages.dtype)
  assert probs.shape == (batch_size, num_kv_heads, num_groups, max_seq_len)
  if debug:
    jax.debug.print("softmax: {probs}", probs=probs)

  # Attention is probability-weighted sum of v heads
  attention = jnp.einsum("bhgt,bhtd->bhgd", probs, v_pages)
  assert attention.shape == (batch_size, num_kv_heads, num_groups, head_dim)
  return attention.reshape(batch_size, num_q_heads, head_dim)
