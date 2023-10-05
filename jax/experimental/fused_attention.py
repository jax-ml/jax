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

from typing import Any, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray

def get_large_negative_number(dtype):
    """Returns a large negative value for the given dtype."""
    if jnp.issubdtype(dtype, jnp.inexact):
        dtype_max = jnp.finfo(dtype).max
    elif jnp.issubdtype(dtype, jnp.integer):
        dtype_max = jnp.iinfo(dtype).max
    else:
        raise ValueError('Unsupported dtype for inputs.')
    return jnp.asarray(-0.7 * dtype_max, dtype=dtype)

def causal_mask(input_t):
    """Computes and returns causal mask."""
    assert (input_t.dtype == jnp.float32 or
            input_t.dtype == jnp.bfloat16 or input_t.dtype == jnp.float16), input_t.dtype
    large_negative_number = get_large_negative_number(input_t.dtype)
    t = input_t.shape[2]
    col_idx = jnp.tile(jnp.arange(t)[jnp.newaxis, :], [t, 1])
    row_idx = jnp.tile(jnp.arange(t)[:, jnp.newaxis], [1, t])
    mask = (row_idx < col_idx).astype(input_t.dtype) * large_negative_number
    return mask[jnp.newaxis, jnp.newaxis, :, :]

def combine_biases(*masks: Optional[jnp.ndarray]):
    """Combine attention biases."""
    masks = [m for m in masks if m is not None]
    if not masks: return None
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask

def apply_mask_and_bias(
    out: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    mask_application_type: Optional[str] = None,
) -> jnp.ndarray:
    
    # Infer dimensions from mask or bias, whichever is available.
    if mask is not None:
        batch_size, num_attn_heads, q_seq_len, kv_seq_len = mask.shape
    elif bias is not None:
        batch_size, num_attn_heads, q_seq_len, kv_seq_len = bias.shape
    else:  # If both mask and bias are None, return out as is.
        return out
    
    dtype = out.dtype
    print(f"{out.shape} is incoming out shape in apply_mask_and_bias before applying bias")
    if bias is not None:
        print(f"{bias.shape} is bias shape in apply_mask_and_bias before applying bias")
        out += bias.astype(dtype)
    print(f"{out.shape} is out shape in apply_mask_and_bias after applying bias")
    if mask is not None:
        print(f"{mask.shape} is mask shape in apply_mask_and_bias before applying mask")
        if mask_application_type == 'additive':
            out += mask.astype(dtype)
            print(f"{out.shape} is out shape in apply_mask_and_bias after applying additive mask")
        elif mask_application_type == 'selective':
            #ToDo check that the mask is of same shape as out and is comprised of 0 and -ve infinity.
            out = lax.select(
                mask,
                out,
                jnp.zeros((batch_size, num_attn_heads, q_seq_len, kv_seq_len), dtype=dtype)
            )
            print(f"{out.shape} is out shape in apply_mask_and_bias after applying selective mask")
        else:
            raise ValueError(f"Invalid mask_application_type: {mask_application_type}")
    print(f"{out.shape} is out shape in apply_mask_and_bias after applying bias and mask")
    #print(out)
    return out


def validate_shapes(
    query: jnp.ndarray, 
    key: jnp.ndarray, 
    value: jnp.ndarray, 
    mask: Optional[jnp.ndarray] = None, 
    bias: Optional[jnp.ndarray] = None
) -> None:
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    
    batch_dim = 0
    assert query.shape[batch_dim] == key.shape[batch_dim] == value.shape[batch_dim], (
        'q, k, v batch dims must match.')
    
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        'q, k, v num_heads must match.')
    
    seq_len_dim = 1
    assert key.shape[seq_len_dim] == value.shape[seq_len_dim], 'k, v lengths must match.'
    
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'
    
    batch_size_q, q_seq_len, num_attn_heads_q, head_dim_q = query.shape
    batch_size_k, kv_seq_len, num_attn_heads_k, head_dim_k = key.shape
    
    if bias is not None:
        batch_size_b, num_attn_heads_b, q_seq_len_b, kv_seq_len_b = bias.shape
        # assert batch_size_q == batch_size_b, "Batch sizes of query and bias must match."
        # assert num_attn_heads_q == num_attn_heads_b, "Number of attention heads must match between query and bias."
        assert q_seq_len == q_seq_len_b, "Query sequence lengths must match between query and bias."
        assert kv_seq_len == kv_seq_len_b, "Key/Value sequence lengths must match between key and bias."
    
    if mask is not None:
        batch_size_m, num_attn_heads_m, q_seq_len_m, kv_seq_len_m = mask.shape
        # assert batch_size_q == batch_size_m, "Batch sizes of query and mask must match."
        # assert num_attn_heads_q == num_attn_heads_m, "Number of attention heads must match between query and mask."
        assert q_seq_len == q_seq_len_m, "Query sequence lengths must match between query and mask."
        assert kv_seq_len == kv_seq_len_m, "Key/Value sequence lengths must match between key and mask."


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          scale: float = 1.0,
                          bias: Optional[Array] = None,
                          mask: Optional[Array] = None,
                          mask_application_type: Optional[str] = None,
                          mask_type: Optional[str] = None,
                          seed: int = 42,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.
  batch seq num_heads, head_dim // but all assume Q, K and V will have same
  b q_seq num_heads head_dim  -> Q
  b kv_seq num_heads head_dim -> K
  b kv_seq num_heads head_dim -> V
  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """

  validate_shapes(query, key, value, mask, bias)
  
  batch_size, q_seq_len, num_attn_heads, head_dim =  query.shape
  batch_size, kv_seq_len, num_attn_heads, head_dim = key.shape

  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
  
  if scale != 1.0:
    attn_weights = attn_weights * scale  
  
  if mask is not None and mask_type is not None:
      raise ValueError("mask_type input will not be honored when mask is provided.")
  
  if mask is None and mask_type is not None:
    if mask_type == 'causal':
        mask = causal_mask(jnp.zeros((batch_size, num_attn_heads, q_seq_len, kv_seq_len), dtype=dtype))
        if mask_application_type is None:
            mask_application_type = 'additive'
        elif mask_application_type != 'additive':
            raise NotImplementedError(f"{mask_application_type} mask application type for " f"{mask_type} mask is not implemented")
    else:
        raise NotImplementedError(f"{mask_type} not implemented") 
  
  attn_weights = apply_mask_and_bias(attn_weights, mask, bias, mask_application_type)
    
  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
     
#   # Apply attention dropout.    
#   attn_weights = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(attn_weights)
  
  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    key = jax.random.PRNGKey(seed)
    dropout_rng, _ = jax.random.split(key)
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)