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

import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
    ragged_paged_attention,
    ref_ragged_paged_attention,
    validate_dynamic_inputs,
)
import jax.numpy as jnp


jax.config.parse_flags_with_absl()


def ceil_div(x, a):
  assert a != 0
  return (x + a - 1) // a


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PagedAttentionKernelTest(jtu.JaxTestCase):

  def _test_ragged_paged_attention(
      self,
      seq_lens,  # List[(q_len, kv_len)]
      num_heads,  # [num_q_heads, num_kv_heads]
      head_dim,
      page_size,
      dtype,
      num_pages,
      *,
      num_kv_pages_per_block=8,
      num_queries_per_block=64,
      vmem_limit_bytes=32 * 1024 * 1024,
      max_num_batched_tokens=512,
      max_num_seq=8,
      sliding_window: int | None = None,
      soft_cap: float | None = None,
  ):
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Expect TPUv4+")
    cu_q_lens = [0]
    kv_lens = []
    for q_len, kv_len in seq_lens:
      assert q_len <= kv_len
      cu_q_lens.append(cu_q_lens[-1] + q_len)
      kv_lens.append(kv_len)

    max_num_batched_tokens = max(cu_q_lens[-1], max_num_batched_tokens)
    max_num_seq = max(len(seq_lens), max_num_seq)
    max_kv_len = max(kv_lens)
    pages_per_seq = ceil_div(max_kv_len, page_size)
    num_q_heads, num_kv_heads = num_heads

    cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
    kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seq + 1 - cu_q_lens.shape[0]))
    kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
    prng_key = jax.random.key(1234)
    k0, k1, k2 = jax.random.split(prng_key, 3)
    q = jax.random.normal(
        k0,
        (max_num_batched_tokens, num_q_heads, head_dim),
        dtype=dtype,
    )
    kv_pages = jax.random.normal(
        k1,
        (num_pages, page_size, num_kv_heads * 2, head_dim),
        dtype=dtype,
    )
    page_indices = jax.random.randint(
        k2, (max_num_seq, pages_per_seq), 0, num_pages, dtype=jnp.int32
    )

    num_seqs = jnp.array([len(seq_lens)], dtype=jnp.int32)

    validate_dynamic_inputs(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sliding_window,
        soft_cap,
    )

    actual_num_q_tokens = cu_q_lens[num_seqs[0]]
    output = ragged_paged_attention(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs=num_seqs,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )[: actual_num_q_tokens]

    expected = ref_ragged_paged_attention(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs=num_seqs,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    tols = {
        "float32": 0.15,
        "bfloat16": 0.2,
    }
    tol = tols[jnp.dtype(dtype).name]
    self.assertAllClose(output, expected, atol=tol, rtol=tol)

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_ragged_paged_attention_basic(self, dtype):
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_ragged_paged_attention_decode_only(self, dtype):
    seq_lens = [
        (1, 18),
        (1, 129),
        (1, 597),
        (1, 122),
        (1, 64),
        (1, 322),
        (1, 463),
        (1, 181),
        (1, 1107),
        (1, 123),
        (1, 31),
        (1, 18),
        (1, 1229),
        (1, 229),
        (1, 87),
        (1, 1328),
    ]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_ragged_paged_attention_prefill_only(self, dtype):
    seq_lens = [
        (5, 18),
        (15, 129),
        (120, 597),
        (100, 122),
        (21, 64),
        (32, 322),
        (251, 463),
        (40, 181),
        (64, 1107),
        (99, 123),
        (10, 31),
        (5, 18),
        (3, 1229),
        (120, 229),
        (9, 87),
        (2, 1328),
    ]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_ragged_paged_attention_mixed(self, dtype):
    seq_lens = [
        (5, 18),
        (1, 129),
        (120, 597),
        (1, 122),
        (1, 64),
        (32, 322),
        (251, 463),
        (1, 181),
        (1, 1107),
        (99, 123),
        (1, 31),
        (5, 18),
        (3, 1229),
        (117, 229),
        (1, 87),
        (1, 1328),
    ]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  @parameterized.product(
      num_seqs=[1, 5, 16],
      # TODO(jevinjiang): Support more num_heads!
      num_heads=[(32, 8), (32, 16), (12, 2), (4, 4), (8, 1)],
      dtype=[jnp.float32, jnp.bfloat16],
      num_kv_pages_per_block=[4, 8],
      num_queries_per_block=[32, 64],
  )
  def test_ragged_paged_attention_complex(
      self,
      num_seqs,
      num_heads,
      dtype,
      num_kv_pages_per_block,
      num_queries_per_block,
  ):
    seq_lens = []
    for _ in range(num_seqs):
      q_len = random.randint(1, 100)
      kv_len = q_len + random.randint(0, 50)
      seq_lens.append((q_len, kv_len))
    # TODO(jevinjiang): Support non-128 head_dim!
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )

  @parameterized.product(
      num_kv_pages_per_block=[4, 8],
      num_queries_per_block=[32, 64],
      sliding_window=[None, 5, 128],
  )
  def test_ragged_paged_attention_sliding_window(
      self,
      num_kv_pages_per_block,
      num_queries_per_block,
      sliding_window: int | None,
  ):
    num_seqs = 5
    num_heads = (4, 4)
    dtype = jnp.float32
    seq_lens = []
    for _ in range(num_seqs):
      q_len = random.randint(1, 100)
      kv_len = q_len + random.randint(0, 50)
      seq_lens.append((q_len, kv_len))
    # TODO(jevinjiang): Support non-128 head_dim!
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        sliding_window=sliding_window,
    )

  @parameterized.product(
      num_kv_pages_per_block=[4, 8],
      num_queries_per_block=[32, 64],
      soft_cap=[None, 50.0],
  )
  def test_ragged_paged_attention_logit_soft_capping(
      self,
      num_kv_pages_per_block,
      num_queries_per_block,
      soft_cap: float | None,
  ):
    num_heads = (12, 2)
    num_seqs = 2
    dtype = jnp.float32
    seq_lens = []
    for _ in range(num_seqs):
      q_len = random.randint(1, 100)
      kv_len = q_len + random.randint(0, 50)
      seq_lens.append((q_len, kv_len))
    head_dim = 128
    page_size = 16
    num_pages = 1000

    self._test_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        soft_cap=soft_cap,
    )

  def test_ragged_paged_attention_sliding_window_should_be_positive(self):
    dtype = jnp.float32
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    with self.assertRaisesRegex(ValueError, "must be positive"):
      self._test_ragged_paged_attention(
          seq_lens,
          num_heads,
          head_dim,
          page_size,
          dtype,
          num_pages,
          sliding_window=0,
      )

    with self.assertRaisesRegex(ValueError, "must be positive"):
      self._test_ragged_paged_attention(
          seq_lens,
          num_heads,
          head_dim,
          page_size,
          dtype,
          num_pages,
          sliding_window=-1,
      )

  def test_ragged_paged_attention_soft_cap_cannot_be_zero(self):
    dtype = jnp.float32
    seq_lens = [(192, 328), (128, 180), (64, 255)]
    num_heads = (32, 8)
    head_dim = 128
    page_size = 16
    num_pages = 1000

    with self.assertRaisesRegex(ValueError, "must not be 0.0"):
      self._test_ragged_paged_attention(
          seq_lens,
          num_heads,
          head_dim,
          page_size,
          dtype,
          num_pages,
          soft_cap=0.0,
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
