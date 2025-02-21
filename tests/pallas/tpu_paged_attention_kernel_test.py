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
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import paged_attention
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


def _generate_qkv(
    dtype,
    batch_size,
    page_size,
    max_seq_len,
    num_kv_heads,
    num_heads,
    head_dim,
):
  assert max_seq_len % page_size == 0
  pages_per_seq = max_seq_len // page_size
  total_pages = batch_size * pages_per_seq
  k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  page_table = jnp.arange(total_pages, dtype=jnp.int32)
  page_table = jax.random.permutation(k3, page_table, independent=True)
  page_table = page_table.reshape(batch_size, pages_per_seq)
  q = jax.random.normal(k4, (batch_size, num_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_table


def _reconstruct_kv(page_table, pages):
  batch_size = page_table.shape[0]
  num_heads, _, _, head_dim = pages.shape

  def per_seq_page_gather(pages, page_table):
    return jnp.take(pages, page_table, 1)

  gathered = jax.vmap(per_seq_page_gather, in_axes=(None, 0))(pages, page_table)
  return gathered.reshape(batch_size, num_heads, -1, head_dim)


def _grouped_query_attention_reference(q, k, v, lengths, attn_logits_soft_cap):
  batch_size, num_heads, head_dim = q.shape
  _, num_kv_heads, max_seq_len, _ = k.shape
  assert k.shape == v.shape
  assert num_heads % num_kv_heads == 0
  q = q.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)
  logits = jnp.einsum(
      "bhgd,bhtd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap
  mask = jnp.arange(max_seq_len)[None] < lengths[:, None]
  mask_value = -0.7 * float(np.finfo(np.dtype("float32")).max)
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None, None, :]
  weights = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum("bhgt,bhtd->bhgd", weights.astype(v.dtype), v)
  return o.reshape(batch_size, num_heads, head_dim)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PagedAttentionKernelTest(jtu.JaxTestCase):

  def test_paged_attention(
      self,
      dtype=jnp.bfloat16,
      page_size=16,
      num_kv_heads=8,
      num_heads=8,
      head_dim=128,
  ):
    if not (jtu.get_tpu_version() >= 6):
      self.skipTest("Only test TPU v6e")

    assert num_heads % num_kv_heads == 0
    max_kv_len, block_size = 2048, 512
    seq_lens = np.asarray([0, 3, 256, 513, 1023, 2048])
    q, k_pages, v_pages, page_table = _generate_qkv(
        dtype,
        len(seq_lens),
        page_size,
        max_kv_len,
        num_kv_heads,
        num_heads,
        head_dim,
    )
    o = paged_attention.paged_attention(
        q,
        k_pages,
        v_pages,
        seq_lens,
        page_table,
        pages_per_compute_block=block_size // page_size,
        megacore_mode=None,
        attn_logits_soft_cap=None,
    )
    k = _reconstruct_kv(page_table, k_pages)
    v = _reconstruct_kv(page_table, v_pages)
    o_ref = _grouped_query_attention_reference(
        q, k, v, seq_lens, attn_logits_soft_cap=None
    )

    if num_heads > num_kv_heads:
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
  absltest.main(testLoader=jtu.JaxTestLoader())
