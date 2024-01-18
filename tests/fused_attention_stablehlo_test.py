# Copyright 2023 The JAX Authors.
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

from functools import partial
from absl.testing import absltest
from typing import Any, Optional
import os
os.environ['XLA_FLAGS'] = '--xla_dump_disable_metadata --xla_gpu_enable_triton_gemm=false --xla_dump_hlo_as_text --xla_dump_to=./scratch/hlo --xla_dump_hlo_module_re=.*pjit__unnamed_function.* --xla_dump_hlo_pass_re=.* --xla_gpu_enable_cudnn_fmha=true --xla_gpu_fused_attention_use_cudnn_rng=true'

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec, NamedSharding
from jax._src import config
from jax._src import test_util as jtu
from jax._src.cudnn.fused_attention_stablehlo import dot_product_attention

config.parse_flags_with_absl()
Array = jnp.ndarray

def f(query: Array,
      key: Array,
      value: Array,
      bias: Optional[Array] = None,
      mask: Optional[Array] = None,
      causal_mask: bool = False,
      scale: float = 0.5,
      dropout_rate: float = 0.1) -> Array:

  output = dot_product_attention(
    query,
    key,
    value,
    scale=scale,
    bias=bias,
    mask=mask,
    is_causal_mask=causal_mask,
    dropout_rate=dropout_rate)
  return output

def f_train(query: Array,
            key: Array,
            value: Array,
            grad: Array,
            bias: Optional[Array] = None,
            mask: Optional[Array] = None,
            causal_mask: bool = False,
            scale: float = 0.5,
            dropout_rate: float = 0.1) -> Array:

  out, f_vjp = jax.vjp(
    partial(f, scale=scale, causal_mask=causal_mask, dropout_rate=dropout_rate),
    query, key, value, bias, None)
  query_grad, key_grad, value_grad, _, _ = f_vjp(grad)
  return out, (query_grad, key_grad, value_grad)

def g(query: Array,
      key: Array,
      value: Array,
      bias: Optional[Array] = None,
      mask: Optional[Array] = None,
      causal_mask: bool = False,
      scale: float = 0.5,
      dropout_rate: float = 0.1) -> Array:

  def get_large_negative_number(dtype):
    if jnp.issubdtype(dtype, jnp.inexact):
        dtype_max = jnp.finfo(dtype).max
    elif jnp.issubdtype(dtype, jnp.integer):
        dtype_max = jnp.iinfo(dtype).max
    else:
        raise ValueError('Unsupported dtype for inputs.')
    return jnp.asarray(-0.7 * dtype_max, dtype=dtype)

  def get_causal_mask(input_t):
    large_negative_number = get_large_negative_number(input_t.dtype)
    t = input_t.shape[2]
    col_idx = jnp.tile(jnp.arange(t)[jnp.newaxis, :], [t, 1])
    row_idx = jnp.tile(jnp.arange(t)[:, jnp.newaxis], [1, t])
    mask = (row_idx < col_idx).astype(input_t.dtype) * large_negative_number
    return mask[jnp.newaxis, jnp.newaxis, :, :]

  if scale != 1.0:
    query = query * scale
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
  if causal_mask:
    bias = get_causal_mask(attn_weights)
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)
  attn_weights = jax.nn.softmax(attn_weights)
  if dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    dropout_rng = jax.random.key(0)
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    attn_weights = jax.lax.select(keep, attn_weights / keep_prob, jnp.zeros_like(attn_weights))

  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)

def g_train(query: Array,
            key: Array,
            value: Array,
            grad: Array,
            bias: Optional[Array] = None,
            mask: Optional[Array] = None,
            causal_mask: bool = False,
            scale: float = 0.5,
            dropout_rate: float = 0.1) -> Array:
  out_ref, g_vjp = jax.vjp(
    partial(g, scale=scale, causal_mask=causal_mask, dropout_rate=dropout_rate),
    query, key, value, bias, None)
  query_grad_ref, key_grad_ref, value_grad_ref, _, _ = g_vjp(grad)
  return out_ref, (query_grad_ref, key_grad_ref, value_grad_ref)

class DotProductAttentionTest(jtu.JaxTestCase):
  @jtu.sample_product(
      batch_size=[4],
      seq_len=[256, 1024],
      num_heads=[8],
      head_dim=[64, 128],
      use_bias=[True],
      is_causal_mask=[False],
      dropout_rate=[0, 0.5],
      scale=[0.5],
      dtype=[jnp.float16, jnp.bfloat16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa(self, batch_size: int, seq_len: int, num_heads: int,
                head_dim: int, use_bias: bool, is_causal_mask: bool,
                dropout_rate: float, scale: float, dtype: jnp.dtype):
    if seq_len == 256 and is_causal_mask:
      self.skipTest("Fused attention does not support mask generation.")
    if seq_len == 256 and head_dim == 128:
      self.skipTest("Fused attention does not support head dim = 128.")
    if len(jax.local_devices()) <= 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")

    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    query = jax.random.normal(
        k1, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    key = jax.random.normal(
        k2, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    value = jax.random.normal(
        k3, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    grad = jax.random.normal(
        k4, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    if use_bias:
      bias = jax.random.normal(
        k5, (batch_size, num_heads, seq_len, seq_len), dtype=dtype)
    else:
      bias = None

    jitted_f_train = jax.jit(partial(f_train, causal_mask=is_causal_mask, scale=scale, dropout_rate=dropout_rate))
    jitted_g_train = jax.jit(partial(g_train, causal_mask=is_causal_mask, scale=scale, dropout_rate=dropout_rate))
    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ('dp', 'tp')) as mesh:
      qkv_spec = PartitionSpec('dp', None, 'tp', None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      if bias is not None:
        bias_spec = PartitionSpec('dp', 'tp', None, None)
      else:
        bias_spec = PartitionSpec()
      bias_sharding = NamedSharding(mesh, bias_spec)
      replicated = NamedSharding(mesh, PartitionSpec())
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      if bias is not None:
        bias = jax.device_put(bias, bias_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding, qkv_sharding, bias_sharding, replicated)
      out_shardings = (replicated, (qkv_sharding, qkv_sharding, qkv_sharding))
      pjitted_f_train = jax.jit(jitted_f_train,
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      pjitted_g_train = jax.jit(jitted_g_train,
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out, (query_grad, key_grad, value_grad) = pjitted_f_train(query, key, value, grad, bias, None)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = pjitted_g_train(query, key, value, grad, bias, None)
      self.assertArraysAllClose(out_ref, out, rtol=1e-5, atol=1e-5)
      if seq_len > 512:
        # query_grad in flash attention is not deterministic
        self.assertArraysAllClose(query_grad_ref, query_grad, rtol=1e-2, atol=1e-2)
      else:
        self.assertArraysAllClose(query_grad_ref, query_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(key_grad_ref, key_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(value_grad_ref, value_grad, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
