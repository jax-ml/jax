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
from jax._src import config
from jax._src import test_util as jtu
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention,
    paged_attention,
    check_cudnn_version,
    MaskType,
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
  if bias is not None:
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
      is_bnth: bool = False,
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

  def get_encoded_padding_mask(encoded, is_bnth):
    dim = 2 if is_bnth else 1
    T = encoded.shape[dim]
    encoded_padding = (jax.lax.iota(np.int32, T) < T // 2).astype(encoded.dtype)
    return jax.lax.broadcast_in_dim(
      encoded_padding, encoded.shape, broadcast_dimensions=[dim])

  def get_sliding_window_mask(logits, window_length):
    large_negative_number = get_large_negative_number(logits.dtype)
    T = logits.shape[-2]
    col_idx = jax.lax.broadcasted_iota(np.int32, (T, T), 1)
    row_idx = jax.lax.broadcasted_iota(np.int32, (T, T), 0)
    mask = jnp.logical_or(
      row_idx < col_idx,
      col_idx <= row_idx - window_length).astype(logits.dtype) * large_negative_number
    return mask[(*([jnp.newaxis]*(len(logits.shape) - 2)), ...)]

  if is_bnth:
    B, qN, T, H = query.shape
    _, kN, _, _ = key.shape
    logits = jnp.einsum("bhqd,bhkd->bhqk", query, key, preferred_element_type=jnp.float32)
  else:
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
  if is_bnth:
    encoded = jnp.einsum("bhqk,bhkd->bhqd", probs, value, preferred_element_type=jnp.float32)
  else:
    encoded = jnp.einsum("bhqk,bkhd->bqhd", probs, value, preferred_element_type=jnp.float32)
  if mask_type == MaskType.PADDING:
    # cuDNN padding mask generation will mask out output accordingly
    # make sure the behavior is the same
    encoded_mask = get_encoded_padding_mask(encoded, is_bnth)
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
            is_bnth: bool = False,
            dropout_rate: float = 0.1,
            sliding_window_length: int | None = None) -> Array:
  out_ref, sdpa_vjp_ref = jax.vjp(
    partial(
      sdpa_ref, scale=scale, mask_type=mask_type, dropout_rate=dropout_rate,
      sliding_window_length=sliding_window_length, is_bnth=is_bnth),
    query, key, value, bias, mask)
  query_grad_ref, key_grad_ref, value_grad_ref, bias_grad_ref, _ = sdpa_vjp_ref(grad)
  if bias is not None:
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
    if not jtu.is_cuda_compute_capability_at_least("8.0"):
      self.skipTest("Requires at least Ampere arch")
    if jtu.is_cuda_version_at_least(13, 0):
      self.skipTest("cuDNN creates no execution plans on CUDA 13.0.")
    self.enter_context(jtu.ignore_warning(
        category=DeprecationWarning, message='`with mesh:` context manager'))

  @jtu.run_on_devices("cuda")
  def test_sdpa_large_head_size(self):
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

  @jtu.sample_product(
      batch_size=[4],
      q_seq_len=[1, 1024],
      kv_seq_len=[1024],
      num_heads=[8],
      head_dim=[64, 128],
      block_size=[64, 128],
      dtype=[jnp.float16, jnp.bfloat16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa_paged_attention(self, batch_size, q_seq_len, kv_seq_len,
                                num_heads, head_dim, block_size, dtype):

    keys = jax.random.split(jax.random.key(0), 5)
    blocks_per_batch = kv_seq_len // block_size
    num_blocks = batch_size * blocks_per_batch

    # different q_seq_len for prefill and decode
    q = jax.random.normal(
      keys[0], (batch_size, q_seq_len, num_heads, head_dim), dtype=dtype)
    k_container = jax.random.normal(
      keys[1], (num_blocks, block_size, num_heads, head_dim), dtype=dtype)
    v_container = jax.random.normal(
      keys[2], (num_blocks, block_size, num_heads, head_dim), dtype=dtype)
    page_table_k = jax.random.randint(
      keys[3], (batch_size, 1, blocks_per_batch, 1), 0, num_blocks-1, dtype=jnp.int32)
    page_table_v = jax.random.randint(
      keys[4], (batch_size, 1, blocks_per_batch, 1), 0, num_blocks-1, dtype=jnp.int32)
    # full page table
    q_seqlen = jnp.full((batch_size,), q_seq_len, jnp.int32)
    kv_seqlen = jnp.full((batch_size,), kv_seq_len, jnp.int32)

    def unpaged(paged, page_table):
      output = jnp.zeros((batch_size, kv_seq_len, num_heads, head_dim), dtype=dtype)
      for b in range(batch_size):
        for block in range(blocks_per_batch):
          block_idx = page_table[b, 0, block, 0]
          output = output.at[
              b, block * block_size : (block + 1) * block_size, :, :
          ].set(paged[block_idx, :, :, :])
      return output

    k = unpaged(k_container, page_table_k)
    v = unpaged(v_container, page_table_v)

    sdpa_infer = jax.jit(partial(
        paged_attention, scale=1.0, mask_type=MaskType.NO_MASK)
    )
    sdpa_infer_ref = jax.jit(partial(
        sdpa_ref, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0)
    )

    sdpa_infer(q, k_container, v_container, q_seqlen=q_seqlen,
      kv_seqlen=kv_seqlen, page_table_k=page_table_k, page_table_v=page_table_v)
    out_ref = sdpa_infer_ref(q, k, v)
    self.assertArraysAllClose(out_ref, out_ref, rtol=1e-2, atol=1e-2)
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class DotProductAttentionF8Test(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    try:
      self.cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if self.cudnn_version == 91000:
      self.skipTest("cuDNN 9.10.0 does not support SDPA FP8")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Requires at least Hopper arch")
    if jtu.is_cuda_compute_capability_equal("12.0"):
      self.skipTest("cuDNN does not support FP8 with compute capability 12.0")
    if jtu.is_cuda_version_at_least(13, 0):
      self.skipTest("cuDNN creates no execution plans on CUDA 13.0.")

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
    if self.cudnn_version < 91900 and jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest(
          "FP8 deterministic algorithm (required for MXFP8 and "
          "d_qk=192/d_v=128) is not supported on Blackwell with cuDNN version "
          "below 9.19.0"
      )
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
    self.assertArraysAllClose(out_ref, out.astype(dtype), rtol=8e-2, atol=8e-2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
