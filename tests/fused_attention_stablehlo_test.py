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

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec, NamedSharding
from jax._src import config
from jax._src import test_util as jtu
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention,
    check_is_flash_attention,
    check_cudnn_version,
    MaskType,
    AttentionLayout,
)

config.parse_flags_with_absl()
Array = jnp.ndarray

fp8_meta_names = [
    "amax_dQ",
    "amax_dK",
    "amax_dV",
    "amax_dP",
    "descale_q",
    "descale_k",
    "descale_v",
    "descale_s",
    "scale_s",
    "scale_o",
    "descale_o",
    "descale_dO",
    "descale_dP",
    "scale_dQ",
    "scale_dK",
    "scale_dV",
    "scale_dP",
]


def quantize_to_fp8(x, q_dtype, compute_dtype, scale):
  # Explicitly cast the max values to the compute dtype to avoid unnecessary
  # casting to FP32 during the subsequent math operations."
  assert q_dtype in (
      jnp.float8_e4m3fn,
      jnp.float8_e5m2,
      jnp.float8_e4m3fnuz,
      jnp.float8_e5m2fnuz,
  )
  dtype_max = jnp.finfo(q_dtype).max.astype(compute_dtype)
  scaled_x = x / jnp.broadcast_to(
      jnp.asarray(scale, dtype=compute_dtype), x.shape
  )
  clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)
  return clipped_x.astype(q_dtype)


def quantize_dequantize_fp8(x, q_dtype, scale, compute_dtype):
  qx = quantize_to_fp8(x, q_dtype, compute_dtype, scale)
  out = qx.astype(x.dtype) * jnp.broadcast_to(
      jnp.asarray(scale, dtype=x.dtype), qx.shape
  )
  return out


cast_to_representable = partial(
    quantize_dequantize_fp8, scale=1, compute_dtype=jnp.bfloat16
)

quantize = partial(quantize_to_fp8, scale=1)

def get_large_negative_number(dtype):
    return 0.7 * jnp.finfo(dtype).min

def sdpa_train(query: Array,
               key: Array,
               value: Array,
               grad: Array,
               bias: Array | None = None,
               mask: Array | None = None,
               q_seqlen: Array | None = None,
               kv_seqlen: Array | None = None,
               q_offsets: Array | None = None,
               kv_offsets: Array | None = None,
               scale: float = 0.5,
               mask_type: MaskType = MaskType.NO_MASK,
               is_bnth: bool = False,
               dropout_rate: float = 0.1,
               sliding_window_length: int | None = None) -> Array:
  if mask_type == MaskType.PADDING:
    if is_bnth:
      B, _, S, _ = query.shape
    else:
      B, S, _, _ = query.shape
    q_seqlen = kv_seqlen = jnp.full((B,), S // 2, jnp.int32)
  out, sdpa_vjp = jax.vjp(
      partial(dot_product_attention, scale=scale, mask_type=mask_type,
              dropout_rate=dropout_rate,
              qkv_layout="BNTH" if is_bnth else "BTNH",
              sliding_window_length=sliding_window_length),
      query, key, value, bias, mask, q_seqlen, kv_seqlen, q_offsets, kv_offsets)
  query_grad, key_grad, value_grad, bias_grad = sdpa_vjp(grad)[:4]
  if bias is not None and len(bias.shape) == 3:
    # has dbias
    return out, (query_grad, key_grad, value_grad, bias_grad)
  return out, (query_grad, key_grad, value_grad)

def sdpa_ref(query: Array,
      key: Array,
      value: Array,
      bias: Array | None = None,
      mask: Array | None = None,
      scale: float = 0.5,
      mask_type: MaskType = MaskType.NO_MASK,
      dropout_rate: float = 0.1,
      sliding_window_length: int | None = None) -> Array:

  def get_causal_mask(logits):
    large_negative_number = get_large_negative_number(logits.dtype)
    t = logits.shape[-2]
    col_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 1)
    row_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 0)
    mask = (row_idx < col_idx).astype(logits.dtype) * large_negative_number
    return mask[(*([jnp.newaxis]*(len(logits.shape) - 2)), ...)]

  def get_padding_mask(logits):
    S, T = logits.shape[-2:]
    large_negative_number = get_large_negative_number(logits.dtype)
    q_padding = (jax.lax.iota(np.int32, S) >= S // 2).reshape((S, 1))
    kv_padding = (jax.lax.iota(np.int32, T) >= T // 2).reshape((1, T))
    combined_padding = \
      (q_padding + kv_padding).astype(logits.dtype) * large_negative_number
    return jax.lax.broadcast(combined_padding, logits.shape[:-2])

  def get_encoded_padding_mask(encoded):
    S = encoded.shape[1]
    encoded_padding = (jax.lax.iota(np.int32, S) < S // 2).astype(encoded.dtype)
    return jax.lax.broadcast_in_dim(
      encoded_padding, encoded.shape, broadcast_dimensions=[1])

  def get_sliding_window_mask(logits, window_length):
    large_negative_number = get_large_negative_number(logits.dtype)
    T = logits.shape[-2]
    col_idx = jax.lax.broadcasted_iota(np.int32, (T, T), 1)
    row_idx = jax.lax.broadcasted_iota(np.int32, (T, T), 0)
    mask = jnp.logical_or(
      row_idx < col_idx,
      col_idx <= row_idx - window_length).astype(logits.dtype) * large_negative_number
    return mask[(*([jnp.newaxis]*(len(logits.shape) - 2)), ...)]

  B, T, qN, H = query.shape
  _, _, kN, _ = key.shape
  logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32)
  if scale != 1.0:
    logits = logits * scale
  if mask_type == MaskType.CAUSAL:
    bias = get_causal_mask(logits)
  elif mask_type == MaskType.PADDING:
    bias = get_padding_mask(logits)
  elif sliding_window_length is not None:
    if sliding_window_length <= 0:
      raise ValueError(
        f"Expect sliding_window_length > 0, got {sliding_window_length}.")
    bias = get_sliding_window_mask(logits, sliding_window_length)
  if mask is not None:
    large_negative_number = get_large_negative_number(logits.dtype)
    mask = jnp.where(mask, 0, large_negative_number)
  # combine bias and mask
  if bias is None:
    bias = mask
  elif mask is not None:
    bias = bias.astype(logits.dtype)
    bias += mask
  # apply bias to logits
  if bias is not None:
    if bias.shape != logits.shape:
      bias = jnp.broadcast_to(bias, logits.shape)
    logits = logits + bias.astype(logits.dtype)
  probs = jax.nn.softmax(logits, axis=-1).astype(query.dtype)
  if dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    dropout_rng = jax.random.key(0)
    keep = jax.random.bernoulli(dropout_rng, keep_prob, probs.shape)
    probs = jax.lax.select(keep, probs / keep_prob, jnp.zeros_like(probs))
  encoded = jnp.einsum("bhqk,bkhd->bqhd", probs, value, preferred_element_type=jnp.float32)
  if mask_type == MaskType.PADDING:
    # cuDNN padding mask generation will mask out output accordingly
    # make sure the behavior is the same
    encoded_mask = get_encoded_padding_mask(encoded)
    encoded = encoded * encoded_mask
  return encoded.astype(query.dtype)

def sdpa_train_ref(query: Array,
            key: Array,
            value: Array,
            grad: Array,
            bias: Array | None = None,
            mask: Array | None = None,
            scale: float = 0.5,
            mask_type: MaskType = MaskType.NO_MASK,
            dropout_rate: float = 0.1,
            sliding_window_length: int | None = None) -> Array:
  out_ref, sdpa_vjp_ref = jax.vjp(
    partial(
      sdpa_ref, scale=scale, mask_type=mask_type, dropout_rate=dropout_rate,
      sliding_window_length=sliding_window_length),
    query, key, value, bias, mask)
  query_grad_ref, key_grad_ref, value_grad_ref, bias_grad_ref, _ = sdpa_vjp_ref(grad)
  if bias is not None and len(bias.shape) == 3:
    return out_ref, (query_grad_ref, key_grad_ref, value_grad_ref, bias_grad_ref)
  return out_ref, (query_grad_ref, key_grad_ref, value_grad_ref)


def sdpa_train_fp8(
    query: Array,
    key: Array,
    value: Array,
    grad: Array,
    fp8_metas: dict[Array],
    scale: float = 0.5,
    mask_type: MaskType = MaskType.NO_MASK,
):
  def dot_product_attention_fp8(query, key, value, fp8_metas):
    f_p = partial(
        dot_product_attention, scale=scale, mask_type=mask_type, use_fp8=True
    )
    return f_p(query, key, value, fp8_params=fp8_metas)

  out, sdpa_vjp = jax.vjp(
      dot_product_attention_fp8, query, key, value, fp8_metas
  )

  grad_amax_s = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
  grad_amax_o = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
  query_grad, key_grad, value_grad, *_ = sdpa_vjp(
      (grad, grad_amax_s, grad_amax_o)
  )
  return out[0], (query_grad, key_grad, value_grad)


class DotProductAttentionTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 8904:
      self.skipTest("Requires >= cuDNN 8.9.4")
    if not jtu.is_cuda_compute_capability_at_least("8.0"):
      self.skipTest("Requires at least Ampere arch")

  @jtu.sample_product(
      batch_size=[4],
      seq_len=[1024],
      num_heads=[8],
      head_dim=[64, 128],
      use_mask=[False, True],
      use_bias=[False, True],
      mask_type=[MaskType.NO_MASK],
      dropout_rate=[0],
      scale=[0.5],
      dtype=[jnp.float16, jnp.bfloat16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa(self, batch_size: int, seq_len: int, num_heads: int,
                head_dim: int, use_mask: bool, use_bias: bool, mask_type: MaskType,
                dropout_rate: float, scale: float, dtype: jnp.dtype):
    if len(jax.local_devices()) < 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")
    if use_mask and mask_type != MaskType.NO_MASK:
      self.skipTest("Either pass in mask or generate mask directly in cuDNN.")
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
        k6, 0.5, (batch_size, num_heads, seq_len, seq_len))
    else:
      mask = None
    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", None, "tp", None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      if bias is not None:
        bias_spec = PartitionSpec("dp", "tp", None, None)
      else:
        bias_spec = PartitionSpec()
      if mask is not None:
        mask_spec = PartitionSpec("dp", "tp", None, None)
      else:
        mask_spec = PartitionSpec()
      bias_sharding = NamedSharding(mesh, bias_spec)
      mask_sharding = NamedSharding(mesh, mask_spec)
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      if bias is not None:
        bias = jax.device_put(bias, bias_sharding)
      if mask is not None:
        mask = jax.device_put(mask, mask_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding,
                      qkv_sharding, bias_sharding, mask_sharding)
      out_shardings = (qkv_sharding, (qkv_sharding, qkv_sharding, qkv_sharding))
      jitted_sdpa_train = jax.jit(
        partial(
          sdpa_train, scale=scale, mask_type=mask_type,
          dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_train_ref = jax.jit(
        partial(
          sdpa_train_ref, scale=scale, mask_type=mask_type,
          dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out, (query_grad, key_grad, value_grad) = \
          jitted_sdpa_train(query, key, value, grad, bias, mask)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = \
          jitted_sdpa_train_ref(query, key, value, grad, bias, mask)
      self.assertArraysAllClose(out_ref, out, rtol=2e-2, atol=2e-2)
      self.assertArraysAllClose(
        query_grad_ref, query_grad, rtol=2e-1, atol=2e-1)
      self.assertArraysAllClose(
        key_grad_ref, key_grad, rtol=2e-1, atol=2e-1)
      self.assertArraysAllClose(
        value_grad_ref, value_grad, rtol=2e-1, atol=2e-1)

  @jtu.run_on_devices("cuda")
  def test_sdpa_inference(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(
        k1, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 1024, 4, 64), dtype=jnp.bfloat16)

    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", None, "tp", None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      in_shardings = (
        qkv_sharding, qkv_sharding, qkv_sharding)
      out_shardings = qkv_sharding
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      jitted_sdpa_inference = jax.jit(
        partial(
          dot_product_attention, scale=1.0, mask_type=MaskType.NO_MASK,
          dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_inference_ref = jax.jit(
        partial(
          sdpa_ref, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out = jitted_sdpa_inference(query, key, value)
      out_ref = jitted_sdpa_inference_ref(query, key, value)
      self.assertArraysAllClose(out_ref, out, rtol=2e-2, atol=2e-2)

  @jtu.run_on_devices("cuda")
  def test_sdpa_var_seq(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(
        k1, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    grad = jax.random.normal(
        k4, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    jitted_sdpa_train = jax.jit(
      partial(
        sdpa_train, scale=1.0, mask_type=MaskType.PADDING, dropout_rate=0),
    )

    jitted_sdpa_train_ref = jax.jit(
      partial(
        sdpa_train_ref, scale=1.0, mask_type=MaskType.PADDING, dropout_rate=0),
    )

    out, (query_grad, key_grad, value_grad) = \
      jitted_sdpa_train(query, key, value, grad)
    out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = \
      jitted_sdpa_train_ref(query, key, value, grad)
    self.assertArraysAllClose(out_ref, out, rtol=2e-2, atol=2e-2)
    self.assertArraysAllClose(query_grad_ref, query_grad, rtol=2e-1, atol=2e-1)
    self.assertArraysAllClose(key_grad_ref, key_grad, rtol=2e-1, atol=2e-1)
    self.assertArraysAllClose(value_grad_ref, value_grad, rtol=2e-1, atol=2e-1)

  @jtu.run_on_devices("cuda")
  def test_sdpa_broadcast_bias_and_dbias(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 8906:
      self.skipTest("Requires >= cuDNN 8.9.6")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Requires at least Hopper arch")

    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    query = jax.random.normal(
        k1, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    grad = jax.random.normal(
        k4, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    bias = jax.random.normal(
        k5, (4, 1024, 1024), dtype=jnp.bfloat16)
    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", None, "tp", None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      bias_spec = PartitionSpec("tp", None, None)
      bias_sharding = NamedSharding(mesh, bias_spec)
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding,
                      qkv_sharding, bias_sharding)
      out_shardings = (qkv_sharding, (qkv_sharding, qkv_sharding, qkv_sharding, bias_sharding))
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      bias = jax.device_put(bias, bias_sharding)
      jitted_sdpa_train = jax.jit(
        partial(
          sdpa_train, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_train_ref = jax.jit(
        partial(
          sdpa_train_ref, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out, (query_grad, key_grad, value_grad, bias_grad) = \
        jitted_sdpa_train(query, key, value, grad, bias)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref, bias_grad_ref) = \
        jitted_sdpa_train_ref(query, key, value, grad, bias)
      self.assertArraysAllClose(out_ref, out, rtol=2e-2, atol=2e-2)
      self.assertArraysAllClose(query_grad_ref, query_grad, rtol=2e-1, atol=2e-1)
      self.assertArraysAllClose(key_grad_ref, key_grad, rtol=2e-1, atol=2e-1)
      self.assertArraysAllClose(value_grad_ref, value_grad, rtol=2e-1, atol=2e-1)
      self.assertArraysAllClose(bias_grad_ref, bias_grad, rtol=2e-1, atol=2e-1)

  @jtu.sample_product(
      batch_size=[1, 16],
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa_dbias(self, batch_size: int):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    # cuDNN only supports dbias when batch size is 1. If the batch size is
    # greater, dbias is silently set to all zeros. This test verifies this
    # behavior for both vmap and regular use cases.
    # TODO: Remove this test once cuDNN adds broader dbias support.
    dtype = jnp.bfloat16
    x_shape = (batch_size, 512, 16, 48)
    bias_shape = (batch_size, 16, 512, 512)
    mask_shape = (1, 1, 512)

    keys = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(keys[0], x_shape, dtype=dtype)
    bias = jax.random.normal(keys[1], bias_shape, dtype=dtype)
    mask = jnp.ones(mask_shape, dtype=jnp.bool_)

    def attn(x, bias, mask):
      return dot_product_attention(x, x, x, bias, mask)

    def attn_vjp(x, bias, mask, target_fn):
      _, f_vjp = jax.vjp(target_fn, x, bias, mask)
      return f_vjp(x)

    attn_vmap = jax.vmap(attn, in_axes=(0, 0, None))
    attn_ref = jax.jit(partial(attn_vjp, target_fn=attn))
    attn_ans = jax.jit(partial(attn_vjp, target_fn=attn_vmap))

    _, dbias_ref, _ = attn_ref(x, bias, mask)
    x = jnp.expand_dims(x, axis=1)
    bias = jnp.expand_dims(bias, axis=1)
    _, dbias_ans, _ = attn_ans(x, bias, mask)
    dbias_ans = jnp.squeeze(dbias_ans, axis=1)
    self.assertArraysAllClose(dbias_ans, dbias_ref)
    if batch_size != 1:
      self.assertTrue(not jnp.any(dbias_ans))

  @jtu.run_on_devices("cuda")
  def test_sdpa_sliding_window_length(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(
        k1, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    grad = jax.random.normal(
        k4, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    jitted_sdpa_train = jax.jit(
      partial(
        sdpa_train, scale=1.0, mask_type=MaskType.CAUSAL, dropout_rate=0,
        sliding_window_length=64),
    )
    # for reference implementation
    # sliding_window_length option itself will setup correct mask
    jitted_sdpa_train_ref = jax.jit(
      partial(
        sdpa_train_ref, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0,
        sliding_window_length=64),
    )

    out, (query_grad, key_grad, value_grad) = \
      jitted_sdpa_train(query, key, value, grad)
    out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = \
      jitted_sdpa_train_ref(query, key, value, grad)
    self.assertArraysAllClose(out_ref, out, rtol=2e-2, atol=2e-2)
    self.assertArraysAllClose(query_grad_ref, query_grad, rtol=2e-1, atol=2e-1)
    self.assertArraysAllClose(key_grad_ref, key_grad, rtol=2e-1, atol=2e-1)
    self.assertArraysAllClose(value_grad_ref, value_grad, rtol=2e-1, atol=2e-1)

  @jtu.run_on_devices("cuda")
  def test_sdpa_large_head_size(self):
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 90500:
      self.skipTest("Requires >= cuDNN 9.5.0")
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Requires Hopper arch")

    B, T, N, H = 2, 64, 2, 256
    bf16 = jnp.bfloat16
    keys = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(keys[0], (B, T, N, H), dtype=bf16)
    key = jax.random.normal(keys[1], (B, T, N, H), dtype=bf16)
    value = jax.random.normal(keys[2], (B, T, N, H), dtype=bf16)
    grad = jax.random.normal(keys[3], (B, T, N, H), dtype=bf16)
    sdpa_train_ans = jax.jit(partial(
        sdpa_train, scale=1.0, mask_type=MaskType.CAUSAL, dropout_rate=0)
    )
    sdpa_train_rfc = jax.jit(partial(
        sdpa_train_ref, scale=1.0, mask_type=MaskType.CAUSAL, dropout_rate=0)
    )

    out_ans, grads_ans = sdpa_train_ans(query, key, value, grad)
    out_ref, grads_ref = sdpa_train_rfc(query, key, value, grad)
    self.assertArraysAllClose(out_ref, out_ans)
    self.assertArraysAllClose(grads_ref[0], grads_ans[0], rtol=2e-1, atol=2e-1)
    self.assertArraysAllClose(grads_ref[1], grads_ans[1], rtol=2e-1, atol=2e-1)
    self.assertArraysAllClose(grads_ref[2], grads_ans[2], rtol=2e-1, atol=2e-1)

  @jtu.run_on_devices("cuda")
  def test_sdpa_packed_layout(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 90600:
      self.skipTest("Requires >= cuDNN 9.6.0")
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(
        k1, (4, 512, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 512, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 512, 4, 64), dtype=jnp.bfloat16)
    grad = jax.random.normal(
        k4, (4, 512, 4, 64), dtype=jnp.bfloat16)

    def generate_padding_mask(segment_ids, padding_id, shape, dtype):
      # segment_ids [B, T]
      encoded_padding = jnp.where(segment_ids >= padding_id, 0, 1).astype(dtype)
      return jax.lax.broadcast_in_dim(
        encoded_padding, shape, broadcast_dimensions=[0, 1])

    def generate_segment_mask(segment_ids, dtype):
      segment_ids_1 = jnp.expand_dims(segment_ids, axis=-1)
      # segment_ids_1 = jnp.where(segment_ids_1 == 3, 4, segment_ids_1)
      segment_ids_2 = jnp.expand_dims(segment_ids, axis=1)
      mask = jnp.not_equal(segment_ids_1, segment_ids_2).astype(dtype)
      # broadcast to [B, N, T, T]
      mask = jnp.expand_dims(mask, 1)
      mask *= get_large_negative_number(dtype)
      return mask

    # starting pos of each segment
    q_offsets = jnp.asarray([
      [0, 170, 340, -1], # 3 segments
      [0, 150, 340, -1], # 3 segments
      [0, 190, -1, -1],  # 2 segments
      [0, -1, -1, -1]    # 1 segment
    ], dtype=np.int32)

    # actual seqlen of each segment without padding
    q_seqlen = jnp.asarray([
      [170, 170, 172], # No padding inside each segment
      [150, 187, 172], # 3 padding tokens inside second segment
      [190, 190, -1],  # 132 padding tokens inside last segment
      [400, -1, -1],   # 112 padding tokens inside last segment
    ], dtype=np.int32)

    # maximum number of segments is id for padding token
    segment_ids = jnp.asarray([
      [0]*170 + [1]*170 + [2]*172,
      [0]*150 + [1]*187 + [3]*3 + [2]*172,
      [0]*190 + [1]*190 + [3]*132,
      [0]*400 + [3]*112,
    ], dtype=np.int32)

    kv_offsets = q_offsets.copy()
    kv_seqlen = q_seqlen.copy()

    mask = generate_padding_mask(segment_ids, q_seqlen.shape[1], query.shape, query.dtype)
    bias = generate_segment_mask(segment_ids, jnp.float32)

    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", None, "tp", None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      bias_spec = PartitionSpec("dp", None, None, None)
      bias_sharding = NamedSharding(mesh, bias_spec)
      offsets_specs = PartitionSpec("dp", None)
      offsets_sharding = NamedSharding(mesh, offsets_specs)

      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      bias = jax.device_put(bias, bias_sharding)
      q_offsets = jax.device_put(q_offsets, offsets_sharding)
      kv_offsets = jax.device_put(kv_offsets, offsets_sharding)
      q_seqlen = jax.device_put(q_seqlen, offsets_sharding)
      kv_seqlen = jax.device_put(kv_seqlen, offsets_sharding)

      jitted_sdpa_train = jax.jit(
        partial(
          sdpa_train, scale=0.1, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=(qkv_sharding, qkv_sharding, qkv_sharding, qkv_sharding,
                      None, None, offsets_sharding, offsets_sharding, offsets_sharding, offsets_sharding),
        out_shardings=(qkv_sharding, (qkv_sharding, qkv_sharding, qkv_sharding))
      )

      jitted_sdpa_train_ref = jax.jit(
        partial(
          sdpa_train_ref, scale=0.1, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=(qkv_sharding, qkv_sharding, qkv_sharding, qkv_sharding,
                      bias_sharding),
        out_shardings=(qkv_sharding, (qkv_sharding, qkv_sharding, qkv_sharding))
      )

      query = query * mask
      key = key * mask
      value = value * mask
      grad = grad * mask

      out, (query_grad, key_grad, value_grad) = \
        jitted_sdpa_train(query, key, value, grad, None, None, q_seqlen, kv_seqlen, q_offsets, kv_offsets)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = \
        jitted_sdpa_train_ref(query, key, value, grad, bias)

      out = out * mask
      out_ref = out_ref * mask

      query_grad = query_grad * mask
      query_grad_ref = query_grad_ref * mask

      key_grad = key_grad * mask
      key_grad_ref = key_grad_ref * mask

      value_grad = value_grad * mask
      value_grad_ref = value_grad_ref * mask

      self.assertArraysAllClose(out_ref, out, rtol=1e-2, atol=1e-2)
      self.assertArraysAllClose(query_grad_ref, query_grad, rtol=1e-2, atol=1e-2)
      self.assertArraysAllClose(key_grad_ref, key_grad, rtol=1e-2, atol=1e-2)
      self.assertArraysAllClose(value_grad_ref, value_grad, rtol=1e-2, atol=1e-2)

  @jtu.run_on_devices("cuda")
  def test_layouts(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    dtype = "bfloat16"
    B, T, N, H = 4, 1024, 8, 128
    S = T
    k0, k1, k2, k3 = jax.random.split(jax.random.key(123), 4)
    query = jax.random.normal(k0, (B, T, N, H), dtype=dtype)
    key = jax.random.normal(k1, (B, S, N, H), dtype=dtype)
    value = jax.random.normal(k2, (B, S, N, H), dtype=dtype)
    grad = jax.random.normal(k3, (B, T, N, H), dtype=dtype)

    btnh_fn = jax.jit(partial(sdpa_train, scale=.5,
      mask_type=MaskType.CAUSAL, is_bnth=False, dropout_rate=0.0))
    out_ref, (dq_ref, dk_ref, dv_ref) = btnh_fn(query, key, value, grad)

    def _cvt(x):
      return jnp.einsum("BTNH->BNTH", x)
    def _cvt_back(x):
      return jnp.einsum("BNTH->BTNH", x)
    bnth_fn = jax.jit(partial(sdpa_train, scale=.5, mask_type=MaskType.CAUSAL,
                              is_bnth=True, dropout_rate=0.0))
    out, (dq, dk, dv) = bnth_fn(_cvt(query), _cvt(key), _cvt(value), _cvt(grad))

    self.assertArraysAllClose(out_ref, _cvt_back(out))
    self.assertArraysAllClose(dq_ref, _cvt_back(dq))
    self.assertArraysAllClose(dk_ref, _cvt_back(dk))
    self.assertArraysAllClose(dv_ref, _cvt_back(dv))

  def test_sdpa_utils(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    test_cases = [
      (1, 257, 64, 8905, False, True, True),
      (1, 1024, 64, 8905, False, False, True),
      (1024, 1024, 64, 8905, False, False, True),
      (1024, 1024, 128, 8905, False, False, True),
      (1024, 1024, 127, 8905, False, False, False),
    ]

    for k in test_cases:
      sql_q, sql_v, head_dim, cudnn_version, has_bias, is_training, \
        expected_pass = k
      query = jnp.empty((4, sql_q, 4, head_dim))
      key = jnp.empty((4, sql_v, 4, head_dim))
      if expected_pass:
        check_is_flash_attention(
          query, key, AttentionLayout.BNTH.value, cudnn_version, has_bias,
          is_training)
      else:
        with self.assertRaises(NotImplementedError):
          check_is_flash_attention(
            query, key, AttentionLayout.BNTH.value, cudnn_version, has_bias,
            is_training)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class DotProductAttentionF8Test(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 90100:
      self.skipTest("Requires >= cuDNN 9.1.0")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Requires at least Hopper arch")

  @jtu.sample_product(
      batch_size=[2, 4],
      seq_len=[128, 256],
      num_heads=[4, 8],
      head_dim=[128],
      mask_type=[MaskType.NO_MASK],
      scale=[1.0, 0.75],
      dtype=[jnp.bfloat16, jnp.float16],
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa_fp8(
      self,
      batch_size: int,
      seq_len: int,
      num_heads: int,
      head_dim: int,
      mask_type: MaskType,
      scale: float,
      dtype: jnp.dtype,
  ):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    input_shape = (
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )  # only test the default BTNH
    query_h = jax.random.normal(k1, input_shape, dtype=dtype)
    key_h = jax.random.normal(k2, input_shape, dtype=dtype)
    value_h = jax.random.normal(k3, input_shape, dtype=dtype)
    grad_h = jax.random.normal(k4, input_shape, dtype=dtype)
    query = cast_to_representable(query_h, jnp.float8_e4m3fn)
    key = cast_to_representable(key_h, jnp.float8_e4m3fn)
    value = cast_to_representable(value_h, jnp.float8_e4m3fn)
    grad = cast_to_representable(grad_h, jnp.float8_e4m3fn)

    query_quantized = quantize(query, jnp.float8_e4m3fn, jnp.float32)
    key_quantized = quantize(key, jnp.float8_e4m3fn, jnp.float32)
    value_quantized = quantize(value, jnp.float8_e4m3fn, jnp.float32)
    grad_quantized = quantize(grad, jnp.float8_e4m3fn, jnp.float32)

    sdpa_train_fp8_p = partial(sdpa_train_fp8, scale=scale, mask_type=mask_type)
    jitted_sdpa_train_fp8 = jax.jit(sdpa_train_fp8_p)
    jitted_sdpa_train_ref = jax.jit(
        partial(
            sdpa_train_ref, scale=scale, mask_type=mask_type, dropout_rate=0.0
        ),
    )

    fp8_metas = {
        name: jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
        for name in fp8_meta_names
    }
    out, (query_grad, key_grad, value_grad) = jitted_sdpa_train_fp8(
        query_quantized,
        key_quantized,
        value_quantized,
        grad_quantized,
        fp8_metas,
    )
    out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = (
        jitted_sdpa_train_ref(query, key, value, grad)
    )

    self.assertArraysAllClose(out_ref, out.astype(dtype), rtol=5e-1, atol=5e-1)
    self.assertArraysAllClose(
        query_grad_ref, query_grad.astype(dtype), rtol=5e-1, atol=3e0
    )
    self.assertArraysAllClose(
        key_grad_ref, key_grad.astype(dtype), rtol=5e-1, atol=3e0
    )
    self.assertArraysAllClose(
        value_grad_ref, value_grad.astype(dtype), rtol=5e-1, atol=5e-1
    )

  @jtu.sample_product(
      batch_size=[4, 2],
      seq_len=[4, 16],
      num_heads=[4, 16],
      head_dim=[16, 32],
      mask_type=[MaskType.NO_MASK],
      qkv_layout=["BNTH", "BTNH"],
      scale=[1.0, 0.75],
      dtype=[jnp.bfloat16, jnp.float16],
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa_fp8_inference(
      self,
      batch_size: int,
      seq_len: int,
      num_heads: int,
      head_dim: int,
      mask_type: MaskType,
      qkv_layout: str,
      scale: float,
      dtype: jnp.dtype,
  ):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    if qkv_layout == "BNTH":
      input_shape = (batch_size, num_heads, seq_len, head_dim)
    else:
      input_shape = (batch_size, seq_len, num_heads, head_dim)
    query_h = jax.random.normal(k1, input_shape, dtype=dtype)
    key_h = jax.random.normal(k2, input_shape, dtype=dtype)
    value_h = jax.random.normal(k3, input_shape, dtype=dtype)

    query = cast_to_representable(query_h, jnp.float8_e4m3fn)
    key = cast_to_representable(key_h, jnp.float8_e4m3fn)
    value = cast_to_representable(value_h, jnp.float8_e4m3fn)

    query_quantized = quantize(query, jnp.float8_e4m3fn, jnp.float32)
    key_quantized = quantize(key, jnp.float8_e4m3fn, jnp.float32)
    value_quantized = quantize(value, jnp.float8_e4m3fn, jnp.float32)

    def dot_product_attention_fp8(query, key, value, fp8_metas):
      f_p = partial(
          dot_product_attention,
          scale=scale,
          mask_type=mask_type,
          qkv_layout=qkv_layout,
          use_fp8=True,
      )
      return f_p(query, key, value, fp8_params=fp8_metas)

    jitted_sdpa_inference = jax.jit(
        dot_product_attention_fp8,
    )

    jitted_sdpa_inference_ref = jax.jit(
        partial(
            dot_product_attention,
            scale=scale,
            mask_type=mask_type,
            qkv_layout=qkv_layout,
        ),
    )
    fp8_metas = {
        name: jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
        for name in fp8_meta_names
    }
    out, _, _ = jitted_sdpa_inference(
        query_quantized, key_quantized, value_quantized, fp8_metas
    )
    out_ref = jitted_sdpa_inference_ref(query, key, value)
    self.assertArraysAllClose(out_ref, out.astype(dtype), rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
