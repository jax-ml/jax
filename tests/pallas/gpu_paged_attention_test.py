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

import sys
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

if sys.platform != "win32":
  from jax.experimental.pallas.ops.gpu import paged_attention
else:
  paged_attention = None

jax.config.parse_flags_with_absl()


def _generate_qkv(
    batch_size,
    page_size,
    max_seq_len,
    num_kv_heads,
    num_heads,
    head_dim,
    prng_key,
    dtype=jnp.float32,
):
  assert max_seq_len % page_size == 0
  max_num_blocks_per_seq = max_seq_len // page_size
  total_pages = batch_size * max_num_blocks_per_seq
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  k_pages = k_pages / jnp.linalg.norm(k_pages, axis=-1)[..., None]
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  v_pages = v_pages / jnp.linalg.norm(v_pages, axis=-1)[..., None]

  block_tables = jnp.arange(
      batch_size * max_num_blocks_per_seq, dtype=jnp.int32
  )
  block_tables = jax.random.permutation(k3, block_tables, independent=True)
  block_tables = block_tables.reshape(batch_size, max_num_blocks_per_seq)
  q = jax.random.normal(k4, (batch_size, num_heads, head_dim), dtype=dtype)
  q = q / jnp.linalg.norm(q, axis=-1)[..., None]
  return q, k_pages, v_pages, block_tables


def _reconstruct_kv(block_tables: jax.Array, pages: jax.Array) -> jax.Array:
  def fn(_block_tables, _pages):
    head_dim = _pages.shape[-1]
    out = _pages[_block_tables]  # [max_num_blocks_per_seq, page_size, head_dim]

    return out.reshape(-1, head_dim)

  with_batch = jax.vmap(fn, (0, None), 0)
  attn_fn = jax.vmap(with_batch, (None, 0), 1)

  out = attn_fn(block_tables, pages)
  out = jnp.swapaxes(out, 1, 2)

  return out

def _quantize(x: jax.Array, dtype=jnp.int8):
  if isinstance(dtype, jnp.floating):
    max_val = jnp.astype(jnp.finfo(dtype).max, x.dtype)
  else:
    max_val = 127
  x_scale = jnp.max(jnp.abs(x), axis=-1) / (0.95 * max_val)
  x_quant = (x / x_scale[..., None])
  if isinstance(dtype, jnp.floating):
    x_quant = jnp.rint(x_quant)
  return x_quant.astype(dtype), x_scale.astype(x.dtype)


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("Must only run on GPUs, or CPUs")
    if jtu.test_device_matches(["cpu"]) and not self.INTERPRET:
      self.skipTest("On CPU, the test works only in interpret mode")
    if jax.config.x64_enabled:
      self.skipTest("The test works only in 32-bit")
    if jtu.test_device_matches(
        ["cuda"]
    ) and not jtu.is_cuda_compute_capability_at_least("8.0"):
      self.skipTest("Only works on GPU with capability >= sm80")
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()

class PagedAttentionKernelTest(PallasBaseTest):

  def setUp(self):
    super().setUp()

  @jtu.sample_product(
      dtype=(jnp.float16,),
      page_size=(8, 16, 32),
      num_kv_heads=(1, 2),
      q_kv_head_ratio=(2, 16, 20),
      head_dim=(32, 64),
      block_h=(16, 32),
      pages_per_compute_block=(4, 8),
      k_splits=(4, 16),
      attn_logits_soft_cap=(None,),
  )
  def test_paged_attention(
      self,
      dtype,
      page_size,
      num_kv_heads,
      q_kv_head_ratio,
      head_dim,
      block_h,
      pages_per_compute_block,
      k_splits,
      attn_logits_soft_cap,
  ):
    max_kv_len = 2048
    seq_lens = np.asarray([3, 256, 513, 1023, 2048], dtype=jnp.int32)
    q, k_pages, v_pages, block_tables = _generate_qkv(
        seq_lens.shape[0],
        page_size,
        max_kv_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        jax.random.key(0),
        dtype,
    )
    k = _reconstruct_kv(block_tables, k_pages)
    v = _reconstruct_kv(block_tables, v_pages)

    o = paged_attention.paged_attention(
        q,
        k_pages,
        v_pages,
        block_tables,
        seq_lens,
        block_h=block_h,
        pages_per_compute_block=pages_per_compute_block,
        k_splits=k_splits,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
    )

    o_ref = paged_attention.paged_attention_reference(q, k, v, lengths=seq_lens)

    self.assertArraysAllClose(o, o_ref, rtol=5e-2, atol=5e-2)

  @jtu.sample_product(
      dtype=(jnp.float16,),
      page_size=(8, 16, 32),
      num_kv_heads=(1, 2),
      q_kv_head_ratio=(2, 16, 20),
      head_dim=(32, 64),
      block_h=(16, 32),
      pages_per_compute_block=(4, 8),
      k_splits=(4, 16),
      attn_logits_soft_cap=(None,),
      quantize_k=(True, False),
      quantize_v=(True, False),
      quant_dtype=(jnp.float8_e5m2, jnp.float8_e4m3fn, jnp.int8),
  )
  def test_quantized_paged_attention(
      self,
      dtype,
      page_size,
      num_kv_heads,
      q_kv_head_ratio,
      head_dim,
      block_h,
      pages_per_compute_block,
      k_splits,
      attn_logits_soft_cap,
      quantize_k,
      quantize_v,
      quant_dtype,
  ):
    if not quantize_k and not quantize_v:
      self.skipTest("Skipping since neither (k, v) quantization requested.")
    if (quant_dtype == jnp.float8_e4m3fn
        and not jtu.is_cuda_compute_capability_at_least("8.9")):
      self.skipTest("Skipping since float8_e4m3fn is not supported on < sm89")
    max_kv_len = 2048
    seq_lens = np.asarray([3, 256, 513, 1023, 2048], dtype=jnp.int32)
    q, k_pages, v_pages, block_tables = _generate_qkv(
        seq_lens.shape[0],
        page_size,
        max_kv_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        jax.random.key(0),
        dtype,
    )
    k = _reconstruct_kv(block_tables, k_pages)
    v = _reconstruct_kv(block_tables, v_pages)

    k_, k_scales = (_quantize(k_pages, quant_dtype)
                    if quantize_k else (k_pages, None))
    v_, v_scales = (_quantize(k_pages, quant_dtype)
                    if quantize_v else (v_pages, None))

    o = paged_attention.paged_attention(
        q,
        k_,
        v_,
        block_tables,
        seq_lens,
        k_scales_pages=k_scales,
        v_scales_pages=v_scales,
        block_h=block_h,
        pages_per_compute_block=pages_per_compute_block,
        k_splits=k_splits,
        attn_logits_soft_cap=attn_logits_soft_cap,
        interpret=self.INTERPRET,
    )

    o_ref = paged_attention.paged_attention_reference(q, k, v, lengths=seq_lens)

    error = (jnp.linalg.norm((o - o_ref).astype(jnp.float32), axis=-1)
              / jnp.linalg.norm(o_ref.astype(jnp.float32)))

    admissible_error = 3e-1
    self.assertLessEqual(jnp.mean(error), admissible_error)


class PagedAttentionInterpretTest(PagedAttentionKernelTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
