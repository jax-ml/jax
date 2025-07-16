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
import functools

import google_benchmark
import jax
from jax import random
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import jax.numpy as jnp
import numpy as np


def clear_caches(state):
  state.pause_timing()
  jax.clear_caches()
  state.resume_timing()


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
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_trace_no_cache_clear(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()

  while state:
    _ = attn.trace(q, k, v)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_lower(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()
  traced = attn.trace(q, k, v)

  while state:
    _ = traced.lower(lowering_platforms=("tpu",))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_pallas_mqa_splash_attention_lower_no_cache_clear(state):
  attn, (q, k, v) = make_mqa_splash_attention_fn_and_args()
  traced = attn.trace(q, k, v)

  while state:
    _ = traced.lower(lowering_platforms=("tpu",))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_dot_trace(state):
  fn = jax.jit(jnp.dot)
  while state:
    _ = fn.trace(jnp.arange(1024), jnp.arange(1024))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_dot_trace_no_cache_clear(state):
  fn = jax.jit(jnp.dot)
  while state:
    _ = fn.trace(jnp.arange(1024), jnp.arange(1024))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_concat_trace(state):
  fn = jax.jit(functools.partial(jnp.concat, axis=0))
  while state:
    _ = fn.trace((jnp.ones((1024, 1)), jnp.ones((1024, 1))))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_jnp_concat_trace_no_cache_clear(state):
  fn = jax.jit(functools.partial(jnp.concat, axis=0))
  while state:
    _ = fn.trace((jnp.ones((1024, 1)), jnp.ones((1024, 1))))


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
# NOTE(dsuo): Linear spacing so it's easier to eyeball historical plots.
@google_benchmark.option.arg(1)
@google_benchmark.option.dense_range(128, 896, 128)
def test_num_multiply_eqns_trace(state):
  fns = [lambda x: x * x for _ in range(state.range(0))]
  fn = jax.jit(functools.reduce(lambda a, b: (lambda x: a(b(x))), fns))
  while state:
    _ = fn.trace(jnp.ones((1024,)))
    clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
# NOTE(dsuo): Linear spacing so it's easier to eyeball historical plots.
@google_benchmark.option.arg(1)
@google_benchmark.option.dense_range(128, 896, 128)
def test_num_multiply_eqns_trace_no_cache_clear(state):
  fns = [lambda x: x * x for _ in range(state.range(0))]
  fn = jax.jit(functools.reduce(lambda a, b: (lambda x: a(b(x))), fns))
  while state:
    _ = fn.trace(jnp.ones((1024,)))


if __name__ == "__main__":
  google_benchmark.main()
