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

from functools import partial
from absl.testing import absltest
from typing import Optional
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_cudnn_fmha=true --xla_gpu_fused_attention_use_cudnn_rng=true'

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

def sdpa_train(query: Array,
            key: Array,
            value: Array,
            grad: Array,
            bias: Optional[Array] = None,
            mask: Optional[Array] = None,
            scale: float = 0.5,
            is_causal_mask: bool = False,
            dropout_rate: float = 0.1) -> Array:
  if mask is not None:
    # convert bool mask to dtype mask
    mask = mask.astype(query.dtype)
  out, sdpa_vjp = jax.vjp(
    partial(dot_product_attention, scale=scale, is_causal_mask=is_causal_mask, dropout_rate=dropout_rate),
    query, key, value, bias, mask)
  query_grad, key_grad, value_grad, _, _ = sdpa_vjp(grad)
  return out, (query_grad, key_grad, value_grad)

def sdpa_ref(query: Array,
      key: Array,
      value: Array,
      bias: Optional[Array] = None,
      mask: Optional[Array] = None,
      scale: float = 0.5,
      is_causal_mask: bool = False,
      dropout_rate: float = 0.1) -> Array:

  def get_large_negative_number(input_t):
    dtype = input_t.dtype
    if jnp.issubdtype(dtype, jnp.inexact):
      dtype_max = jnp.finfo(dtype).max
    elif jnp.issubdtype(dtype, jnp.integer):
      dtype_max = jnp.iinfo(dtype).max
    else:
      raise ValueError('Unsupported dtype for inputs.')
    large_negative_number = jnp.asarray(-0.7 * dtype_max, dtype=dtype)
    return large_negative_number

  def get_causal_mask(input_t):
    large_negative_number = get_large_negative_number(input_t)
    t = input_t.shape[2]
    col_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 1)
    row_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 0)
    mask = (row_idx < col_idx).astype(input_t.dtype) * large_negative_number
    return mask[jnp.newaxis, jnp.newaxis, :, :]

  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
  if scale != 1.0:
    attn_weights = attn_weights * scale
  if is_causal_mask:
    bias = get_causal_mask(attn_weights)
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)
  if mask is not None:
    large_negative_number = get_large_negative_number(attn_weights)
    attn_weights = jax.lax.select(mask, attn_weights, jax.lax.broadcast(large_negative_number, attn_weights.shape))
  attn_weights = jax.nn.softmax(attn_weights)
  if dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    dropout_rng = jax.random.key(0)
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    attn_weights = jax.lax.select(keep, attn_weights / keep_prob, jnp.zeros_like(attn_weights))

  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)

def sdpa_train_ref(query: Array,
            key: Array,
            value: Array,
            grad: Array,
            bias: Optional[Array] = None,
            mask: Optional[Array] = None,
            scale: float = 0.5,
            is_causal_mask: bool = False,
            dropout_rate: float = 0.1) -> Array:
  out_ref, sdpa_vjp_ref = jax.vjp(
    partial(sdpa_ref, scale=scale, is_causal_mask=is_causal_mask, dropout_rate=dropout_rate),
    query, key, value, bias, mask)
  query_grad_ref, key_grad_ref, value_grad_ref, _, _ = sdpa_vjp_ref(grad)
  return out_ref, (query_grad_ref, key_grad_ref, value_grad_ref)

class DotProductAttentionTest(jtu.JaxTestCase):
  @jtu.sample_product(
      batch_size=[4],
      seq_len=[256, 1024],
      num_heads=[8],
      head_dim=[64, 128],
      use_bias=[False, True],
      use_mask=[False, True],
      is_causal_mask=[False],
      dropout_rate=[0, 0.5],
      scale=[0.5],
      dtype=[jnp.float16, jnp.bfloat16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa(self, batch_size: int, seq_len: int, num_heads: int,
                head_dim: int, use_bias: bool, use_mask: bool, is_causal_mask: bool,
                dropout_rate: float, scale: float, dtype: jnp.dtype):
    if seq_len == 256 and is_causal_mask:
      self.skipTest("Fused attention does not support mask generation.")
    if seq_len == 256 and head_dim == 128:
      self.skipTest("Fused attention does not support head dim = 128.")
    if len(jax.local_devices()) <= 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")

    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(0), 6)
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
    if use_mask:
      mask = jax.random.bernoulli(
        k5, 0.5, (batch_size, num_heads, seq_len, seq_len))
    else:
      mask = None
    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ('dp', 'tp')) as mesh:
      qkv_spec = PartitionSpec('dp', None, 'tp', None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      if bias is not None:
        bias_spec = PartitionSpec('dp', 'tp', None, None)
      else:
        bias_spec = PartitionSpec()
      if mask is not None:
        mask_spec = PartitionSpec('dp', 'tp', None, None)
      else:
        mask_spec = PartitionSpec()
      bias_sharding = NamedSharding(mesh, bias_spec)
      mask_sharding = NamedSharding(mesh, mask_spec)
      replicated = NamedSharding(mesh, PartitionSpec())
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      if bias is not None:
        bias = jax.device_put(bias, bias_sharding)
      if mask is not None:
        mask = jax.device_put(mask, mask_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding, qkv_sharding, bias_sharding, mask_sharding)
      out_shardings = (replicated, (qkv_sharding, qkv_sharding, qkv_sharding))
      jitted_sdpa_train = jax.jit(
        partial(sdpa_train, scale=scale, is_causal_mask=is_causal_mask, dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_train_ref = jax.jit(
        partial(sdpa_train_ref, scale=scale, is_causal_mask=is_causal_mask, dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out, (query_grad, key_grad, value_grad) = jitted_sdpa_train(query, key, value, grad, bias, mask)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = jitted_sdpa_train_ref(query, key, value, grad, bias, mask)
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
