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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import paged_attention
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
from jax.experimental.pallas.ops.tpu.paged_attention import util
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
class JaxGroupedQueryAttentionReferenceTest(jtu.JaxTestCase):

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
    attention = util.grouped_query_attention_reference(
        queries, k_pages, v_pages, seq_lens
    )
    jax.debug.print("attention: {attention}", attention=attention)

    # compare the results
    atol, rtol = (3e-3, 5e-3) if dtype == jnp.bfloat16 else (2e-4, 2e-4)
    self.assertAllClose(attention, expected, atol=atol, rtol=rtol)


def _generate_random_qkv(
    seq_lens,
    page_size,
    max_seq_len,
    num_kv_heads,
    num_q_heads,
    head_dim,
    prng_key,
    dtype=jnp.float32,
    are_kv_quantized=False,
):
  assert max_seq_len % page_size == 0
  pages_per_sequence = max_seq_len // page_size
  batch_size = len(seq_lens)
  total_pages = batch_size * pages_per_sequence
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )

  if are_kv_quantized:
    k_pages = quantization_utils.quantize_to_int8(k_pages)
    v_pages = quantization_utils.quantize_to_int8(v_pages)

  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(k4, (batch_size, num_q_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_indices


def _reconstruct_kv(page_indices, pages):
  if isinstance(pages, quantization_utils.QuantizedTensor):
    pages = quantization_utils.unquantize_from_int8(pages, dtype=jnp.float32)

  batch_size = page_indices.shape[0]
  num_kv_heads, _, _, head_dim = pages.shape

  def per_sequence_page_gather(pages, page_indices):
    return jnp.take(pages, page_indices, 1)

  gathered = jax.vmap(per_sequence_page_gather, in_axes=(None, 0))(
      pages, page_indices
  )
  return gathered.reshape(batch_size, num_kv_heads, -1, head_dim)


def _megacore_enabled():
  return jax.devices()[0].device_kind == "TPU v4" or jtu.is_device_tpu(
      version=5, variant="p"
  )


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PagedAttentionKernelTest(jtu.JaxTestCase):

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      page_size=(16, 32, 64),
      num_kv_heads=(1, 8),
      q_kv_head_ratio=(1, 4, 8),
      head_dim=(128, 256),
      megacore_mode=("batch", "kv_head", None),
      attn_logits_soft_cap=(1.0, None),
      are_kv_quantized=(
          False,
          True,
      ),
  )
  def test_paged_attention(
      self,
      dtype,
      page_size,
      num_kv_heads,
      q_kv_head_ratio,
      head_dim,
      megacore_mode,
      attn_logits_soft_cap,
      are_kv_quantized,
  ):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supports TPU generation 4 or above")
    if jtu.is_device_tpu(version=4) and are_kv_quantized:
      # TPU v4 has only 16MiB of VMEM which is not sufficient to store both the
      # weight and scale tensors for quantized tensors. When enabled on TPUv4,
      # the tests sometimes failed with resource exhausted error.
      self.skipTest("Quantization is not supported on TPU v4")
    if jtu.is_device_tpu_at_least(6) and are_kv_quantized:
      self.skipTest("Quantization is not supported on TPU v6")
    if megacore_mode and not _megacore_enabled():
      self.skipTest("Megacore is only available on TPU v4 or TPU v5p")
    if num_kv_heads % 2 != 0 and megacore_mode == "kv_head":
      self.skipTest("Skip kv_head megacore mode when num_kv_heads is odd")
    max_kv_len = 2048
    block_size = 512
    seq_lens = np.asarray([0, 3, 256, 513, 1023, 2048])
    q, k_pages, v_pages, page_indices = _generate_random_qkv(
        seq_lens,
        page_size,
        max_kv_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        jax.random.key(0),
        dtype,
        are_kv_quantized=are_kv_quantized,
    )
    o = paged_attention.paged_attention(
        q,
        k_pages,
        v_pages,
        seq_lens,
        page_indices,
        pages_per_compute_block=block_size // page_size,
        megacore_mode=megacore_mode,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    k = _reconstruct_kv(page_indices, k_pages)
    v = _reconstruct_kv(page_indices, v_pages)
    o_ref = util.grouped_query_attention_reference(
        q, k, v, seq_lens, attn_logits_soft_cap
    )

    if q_kv_head_ratio > 1:
      atol, rtol = 1e-2, 2e-2
    else:
      atol, rtol = 2e-1, 1e-1
    np.testing.assert_allclose(
        o[np.where(seq_lens > 0)].astype(jnp.float32),
        o_ref[np.where(seq_lens > 0)].astype(jnp.float32),
        atol=atol,
        rtol=rtol,
    )


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
