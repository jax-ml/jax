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

"""Benchmarks for Jax tracing."""

import google_benchmark
import jax
from jax import random
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import numpy as np


def make_mqa_splash_attention_fn_and_args():
  seed = 0
  key = random.key(seed)
  k1, k2, k3 = random.split(key, 3)

  q_seq_len = 1024
  kv_seq_len = 1024
  num_q_heads = 2
  head_dim_qk = 128
  head_dim_v = 128
  dtype = np.dtype("float32")

  q = random.uniform(k1, (num_q_heads, q_seq_len, head_dim_qk), dtype=dtype)
  k = random.uniform(k2, (kv_seq_len, head_dim_qk), dtype=dtype)
  v = random.uniform(k3, (kv_seq_len, head_dim_v), dtype=dtype)

  mask = mask_lib.NumpyMask(
      mask_lib.make_random_mask((q_seq_len, kv_seq_len), sparsity=0.5, seed=0)
  )
  mask = mask_lib.MultiHeadMask(tuple(mask for _ in range(num_q_heads)))
  block_sizes = splash.BlockSizes.get_default()

  return (
      jax.jit(
          splash.make_splash_mqa_single_device(mask, block_sizes=block_sizes)
      )
  ), (q, k, v)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_trace(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()

  while state:
    _ = attn.trace(q, k, v)
    jax.clear_caches()


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_lower(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()
  traced = attn.trace(q, k, v)

  while state:
    _ = traced.lower(lowering_platforms=("tpu",))
    jax.clear_caches()


if __name__ == "__main__":
  google_benchmark.main()
