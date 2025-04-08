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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import paged_attention
import jax.numpy as jnp
import numpy as np


def _generate_qkv_simplest(
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Generates queries with one query head, kv pages, and attention."""
  max_seq_len = 4
  seq_lens = jnp.asarray([max_seq_len // 2])
  assert seq_lens.shape == (1,)

  # q_shape = (batch_size=1, num_q_heads=1, head_dim=1)
  queries = jnp.asarray([[[1.2]]], dtype)
  assert queries.shape == (1, 1, 1)

  # kv_shape = (batch_size=1, num_kv_heads=1, max_seq_len=4, head_dim=1)
  k_pages = jnp.asarray([[[[0.1], [0.2], [0.3], [0.4]]]], dtype)
  v_pages = jnp.asarray([[[[4.0], [3.0], [2.0], [1.0]]]], dtype)
  assert k_pages.shape == (1, 1, 4, 1)
  assert v_pages.shape == k_pages.shape

  # q*k:      [[[ [.12,  .24,   .36,   .48] ]]]
  # masked:   [[[ [.12,  .24,  -inf,  -inf] ]]]
  # softmax:  [[[ [.47,  .53,     0,     0] ]]]
  # softmax(q*k) * v: .47*4 + .53*3 + 0*... = 3.47
  attention = jnp.asarray([[[3.47]]], dtype)
  assert attention.shape == queries.shape
  return seq_lens, queries, k_pages, v_pages, attention


def _generate_qkv_with_one_q_head(
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Generates queries with one query head, kv pages, and attention."""
  max_seq_len = 4
  seq_lens = jnp.asarray([max_seq_len - 1])
  assert seq_lens.shape == (1,)

  # q_shape = (batch_size=1, num_q_heads=1, head_dim=1)
  queries = jnp.asarray([[[1.7]]], dtype)
  assert queries.shape == (1, 1, 1)

  # kv_shape = (batch_size=1, num_kv_heads=1, max_seq_len=4, head_dim=1)
  k_pages = jnp.asarray([[[[0.12], [0.23], [0.34], [0.45]]]], dtype)
  v_pages = jnp.asarray([[[[4.32], [3.21], [2.10], [1.09]]]], dtype)
  assert k_pages.shape == (1, 1, 4, 1)
  assert v_pages.shape == k_pages.shape

  # q*k:      [[[ [.204,  .391,  .578,  .765] ]]]
  # masked:   [[[ [.204,  .391,  .578,  -inf] ]]]
  # softmax:  [[[ [.273,  .330,  .397,     0] ]]]
  # softmax(q*k) * v: .273*4.32 + .330*3.21 + .397*2.10 + 0*... = 3.0723
  attention = jnp.asarray([[[3.0723]]], dtype)
  assert attention.shape == queries.shape
  return seq_lens, queries, k_pages, v_pages, attention


def _generate_qkv_with_two_q_heads(
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Generates queries with two query heads, kv pages, and attention."""
  max_seq_len = 4
  seq_lens = jnp.asarray([max_seq_len])
  assert seq_lens.shape == (1,)

  # q_shape = (batch_size=1, num_q_heads=2, head_dim=1)
  queries = jnp.asarray([[[1.3], [9.7]]], dtype)
  assert queries.shape == (1, 2, 1)

  # kv_shape = (batch_size=1, num_kv_heads=1, max_seq_len=4, head_dim=1)
  k_pages = jnp.asarray([[[[0.12], [0.23], [0.34], [0.45]]]], dtype)
  v_pages = jnp.asarray([[[[4.32], [3.21], [2.10], [1.09]]]], dtype)
  assert k_pages.shape == (1, 1, 4, 1)
  assert v_pages.shape == k_pages.shape

  # q*k:      [[[ [ .156,  .299,  .442,  .585],
  #               [1.164, 2.231, 3.298, 4.365] ]]]
  # softmax:  [[[ [ .199,  .230,  .265,  .306],
  #               [ .027,  .079,  .229,  .665] ]]]
  # softmax(q*k) * v: .199*4.32 + .230*3.21 + .265*2.10 + .306*1.09 = 2.488
  # softmax(q*k) * v: .027*4.32 + .079*3.21 + .229*2.10 + .665*1.09 = 1.576
  attention = jnp.asarray([[[2.488], [1.576]]], dtype)
  assert attention.shape == queries.shape
  return seq_lens, queries, k_pages, v_pages, attention


def _generate_qkv_with_head_dim_two(
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Generates queries, kv pages, and attention with head_dim=2."""
  max_seq_len = 4
  seq_lens = jnp.asarray([max_seq_len // 2])
  assert seq_lens.shape == (1,)

  # q_shape = (batch_size=1, num_q_heads=1, head_dim=2)
  queries = jnp.asarray([[[1.2, 9.0]]], dtype)
  assert queries.shape == (1, 1, 2)

  # kv_shape = (batch_size=1, num_kv_heads=1, max_seq_len=4, head_dim=2)
  k_pages = jnp.asarray(
      [[[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]]], dtype
  )
  v_pages = jnp.asarray(
      [[[[4.0, 5.0], [3.0, 6.0], [2.0, 7.0], [1.0, 8.0]]]], dtype
  )
  assert k_pages.shape == (1, 1, 4, 2)
  assert v_pages.shape == k_pages.shape

  # q*k:      [[[ [ 1.92,  2.94,  3.96,  4.98] ]]]
  # masked:   [[[ [ 1.92,  2.94,  -inf,  -inf] ]]]
  # softmax:  [[[ [ .265,  .735,     0,     0] ]]]
  # softmax(q*k) * v: .265*4 + 0.735*3 + 0*... = 3.265
  # softmax(q*k) * v: .265*5 + 0.735*6 + 0*... = 5.735
  attention = jnp.asarray([[[3.265, 5.735]]], dtype)
  assert attention.shape == queries.shape
  return seq_lens, queries, k_pages, v_pages, attention


def _generate_qkv(
    dtype: jnp.dtype,
    case: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  match case:
    case 0:
      return _generate_qkv_simplest(dtype)
    case 1:
      return _generate_qkv_with_one_q_head(dtype)
    case 2:
      return _generate_qkv_with_two_q_heads(dtype)
    case 3:
      return _generate_qkv_with_head_dim_two(dtype)
    case _:
      raise ValueError(f"Unsupported case: {case}")


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class StandardAttentionTest(jtu.JaxTestCase):

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      case=(0, 1, 2, 3),
  )
  def test_grouped_query_attention(self, dtype: jnp.dtype, case: int):
    # generate queries, kv pages, and seq_lens
    seq_lens, queries, k_pages, v_pages, expected = _generate_qkv(dtype, case)
    jax.debug.print("seq_lens: {seq_lens}", seq_lens=seq_lens)
    jax.debug.print("queries: {queries}", queries=queries)
    jax.debug.print("k_pages: {k_pages}", k_pages=k_pages)
    jax.debug.print("v_pages: {v_pages}", v_pages=v_pages)
    jax.debug.print("expected: {expected}", expected=expected)

    # calculate grouped query attention
    attention = paged_attention.grouped_query_attention(
        queries, k_pages, v_pages, seq_lens
    )
    jax.debug.print("attention: {attention}", attention=attention)

    # compare the results
    atol, rtol = (3e-3, 5e-3) if dtype == jnp.bfloat16 else (2e-4, 2e-4)
    np.testing.assert_allclose(attention, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
