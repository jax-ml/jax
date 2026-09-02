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
from typing import Any
import functools
import json
import logging
import numpy as np
import os
import platform
import sys

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.typing import Array
from jax._src.interpreters import mlir
from jax.interpreters import xla
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo

logger = logging.getLogger(__name__)

def _is_dtype_supported(dtype: jnp.dtype, cpuinfo: str):
   if dtype == jnp.float32:
      return True
   elif dtype == jnp.bfloat16:
      has_bf16 = any(flag in cpuinfo for flag in ["avx512f", "avx_ne_convert", "amx_bf16"])
      return True
   elif dtype == jnp.float16:
      has_f16 = any(flag in cpuinfo for flag in ["avx_ne_convert", "amx_fp16"])
      has_f16_combined_flags = "avx512_fp16" in cpuinfo and "avx512bw" in cpuinfo
      return has_f16 or has_f16_combined_flags
   else:
      return False

def can_use_cpu_fused_attention(dtype):
    xla_flags = os.environ.get("XLA_FLAGS", "")
    flags = xla_flags.split()
    # XLA sets xla_cpu_use_xnnpack flag to true by default
    # Currently fused attention op is only supported with oneDNN path
    # so we need to ensure that xnnpack path is disabled
    # This can be scaled to support xnnpack path in future if needed
    if ("--xla_cpu_use_xnnpack=false" not in flags):
      logger.info("Not using CPU Fused Attention, to use set XLA_FLAGS=--xla_cpu_use_xnnpack=false")
      return False

    if ("--xla_cpu_experimental_onednn_custom_call=true" not in flags):
      logger.info("Not using CPU Fused Attention, to use add --xla_cpu_experimental_onednn_custom_call=true to XLA_FLAGS")
      return False
    # Only supported for linux
    if platform.system().lower() != "linux":
      logger.info("CPU Fused Attention is supported only for linux OS")
      return False
    # Currentl supported for CPU backend only
    if not any("cpu" in device.platform.lower() for device in jax.devices()):
      logger.info("CPU Fused Attention is supported only for CPU backend")
      return False
    if dtype not in [jnp.float32, jnp.bfloat16, jnp.float16]:
      logger.info(f"CPU Fused Attention only supports float32, bfloat16 and float16, got {dtype}")
      return False
    # TODO (intel-jax): Add more checks for compute thresholds if needed
    try:
        # Check for CPU info for Intel platforms
        # Currently the benchmarks for fused attention op is only
        # run on Intel CPUs, this can be scaled once it provides
        # enhancements to other CPU platforms.
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()

        is_CPU_intel = "intel" in cpuinfo
    except Exception as e:
        logger.info(f"CPU Fused Attention Failed to detect CPU info: {e}", file=sys.stderr)
        return False

    is_supported_dtype = _is_dtype_supported(dtype, cpuinfo)

    return is_CPU_intel and is_supported_dtype

def _check_layout(query, key, value):
  def check_eq(a, b, c, msg):
    if not (a == b == c):
      raise ValueError(f"{msg} must be same, got {a}, {b}, {c}")

  q_rank, k_rank, v_rank = len(query.shape), len(key.shape), len(value.shape)
  if q_rank != 4:
    raise ValueError(f"Q must have a rank of 4, got {q_rank}")
  check_eq(q_rank, k_rank, v_rank, "QKV rank")

  q_dtype, k_dtype, v_dtype = query.dtype, key.dtype, value.dtype
  if q_dtype not in [jnp.float32, jnp.bfloat16, jnp.float16]:
    raise NotImplementedError(f"Q must be fp16/bf16/fp32, got {q_dtype}")
  check_eq(q_dtype, k_dtype, v_dtype, "QKV dtype")


  qB, qT, qN, qH = query.shape
  kB, kS, kN, kH = key.shape
  vB, vS, vN, vH = value.shape

  check_eq(qB, kB, vB, "QKV batch")
  check_eq(qH, kH, vH, "QKV dim_per_head")
  if kN != vN:
    raise ValueError(f"KV must have same number of heads, got {kN} vs {vN}")
  if kS != vS:
    raise ValueError(f"KV must have same seq length, got {kS} vs {vS}")
  if qN % kN != 0:
     raise ValueError(f"Number of Query heads {qN} must be multiple of KV heads {kN}")

def _get_causal_mask(T, S):
  mask = jnp.tril(jnp.ones((T, S), dtype=jnp.bool_))
  return mask[None, None, :, :]

def _reshape_to_grouped(t, num_query_heads, group_size, num_key_heads):

  tB, tN, tT, tS = t.shape
  if tN == 1:
    t = jnp.broadcast_to(t[:, :, None, :, :], (tB, tN, group_size, tT, tS))
  else:
    assert tN == num_query_heads
    t = jnp.reshape(t, (tB, num_key_heads, group_size, tT, tS))
  return t

_custom_call_name_map = {
    (False, False) : "QK_SOFTMAX_AV",
    (True, False) : "QK_SELECT_SOFTMAX_AV",
    (True, True): "QK_CAPPED_SELECT_SOFTMAX_AV",
}

def _get_custom_call_name(is_masked, is_attn_soft_cap):
  return _custom_call_name_map[(is_masked, is_attn_soft_cap)]

def _get_masks(mask, is_causal, batch_size, kv_num_heads, group_size, query_len, kv_len, is_gqa=False):

  if mask is None and not is_causal:
     return None

  combined_mask = jnp.ones((batch_size, kv_num_heads, query_len, kv_len), dtype=jnp.bool_)
  if is_gqa:
     combined_mask = jnp.ones((batch_size, kv_num_heads, group_size, query_len, kv_len), dtype=jnp.bool_)

  if mask is not None:
    assert mask.dtype == jnp.bool_
    combined_mask = jnp.logical_and(combined_mask, mask)

  T, S = query_len, kv_len

  if is_causal:
    mask = _get_causal_mask(T, S)
    if is_gqa:
      mask = mask[None, None, None, T, S]
    combined_mask = jnp.logical_and(combined_mask, mask)
  # Select operation inside SDPA pattern of OneDNN expects logits
  # as 3rd input; that is, OneDnn graph select op keeps logits
  # when input condition is False.
  # Hence we add logical NOT to combined mask
  return jnp.logical_not(combined_mask)

def default_layouts(*shapes):
  return [range(len(shape) - 1, -1, -1) for shape in shapes]

def _create_dot_product_attention_config(scale, attn_logits_soft_cap, mha_type_str):

  scale_bits = np.float32(scale).view(np.int32)
  attn_soft_cap_bits = np.float32(attn_logits_soft_cap).view(np.int32)

  onednn_fmha_config = {
     "scale": int(scale_bits),
     "attn_soft_cap_scale": int(attn_soft_cap_bits),
     "mha_kind": mha_type_str,
  }
  backend_config = {
    "onednn_fmha_config": onednn_fmha_config
  }
  return json.dumps(backend_config)


def _dot_product_attention_fwd_onednn_lowering(ctx, query, key, value, combined_mask_not, scale, attn_logits_soft_cap, mha_type_str, is_gqa):
    query_type = ir.RankedTensorType(query.type)
    query_shape = query_type.shape
    key_type = ir.RankedTensorType(key.type)
    key_shape = key_type.shape
    B, T, N, H = query_shape
    K = key_shape[2]
    S = key_shape[1]
    G = 1

    # Query [B, T, N, H] -> [B, N, T, H]
    query_tr = hlo.transpose(query, mlir.dense_int_array((0, 2, 1, 3)))
    # Key [B, S, K, H] -> [B, K , H, S]
    key_tr = hlo.transpose(key, mlir.dense_int_array((0, 2, 3, 1)))
    # Value [B, S, K, H] -> [B, K , S, H]
    value_tr = hlo.transpose(value, mlir.dense_int_array((0, 2, 1, 3)))
    # Operands for custom call
    operands = [query_tr, key_tr, value_tr]
    if is_gqa:
        G = N // K
        # Query_TR [B, N, T, H] -> [B, K, G, T, H]
        query_grouped = hlo.reshape(ir.RankedTensorType.get([B, K, G, T, H], query_type.element_type), query_tr)
        # Key_TR [B, K , H, S] -> [B, K, 1, H, S]
        key_grouped = hlo.reshape(ir.RankedTensorType.get([B, K, 1, H, S], query_type.element_type), key_tr)
        # Value_TR [B, K , H, S] -> [B, K, 1, S, H]
        value_grouped = hlo.reshape(ir.RankedTensorType.get([B, K, 1, S, H], query_type.element_type), value_tr)
        operands = [query_grouped, key_grouped, value_grouped]


    result_types = [ir.RankedTensorType(operands[0].type)]
    operand_layouts = default_layouts(*[ir.RankedTensorType(operand.type).shape for operand in operands])
    # result layout same as query layout
    result_layouts = [operand_layouts[0]]
    operands.append(combined_mask_not)
    combined_mask_not_layouts = default_layouts(ir.RankedTensorType(combined_mask_not.type).shape)[0]
    operand_layouts.append(combined_mask_not_layouts)
    backend_config = _create_dot_product_attention_config(scale, attn_logits_soft_cap, mha_type_str)

    # create custom call here
    out = mlir.custom_call(
      "__onednn$fMHA",
      result_types=result_types,
      operands=operands,
      operand_layouts=operand_layouts,
      backend_config=backend_config,
      result_layouts=result_layouts,
    )

    output_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))

    if is_gqa:
       # custom_call_out [B, K, G, T, H] -> [B, N, T, H]
       out_reshaped = hlo.reshape(ir.RankedTensorType.get([B, K*G, T, H], query_type.element_type), out.results[0])
       # out [B, N, T, H] -> [B, T, N, H]
       attn_out = hlo.transpose(out_reshaped, output_transpose_perm)
       return (attn_out,)

    # custom_call_out [B, N, T, H] -> [B, T, N, H]
    attn_out = hlo.transpose(out.results[0], output_transpose_perm)

    return (attn_out,)

def _dot_product_attention_fwd_p_abstract_eval(query_aval, *_, **__):
    return (query_aval)

# Create dot_product_attention_fwd_p for forward operation.
_dot_product_attention_onednn_p = core.Primitive("dot_product_attention_fwd")
_dot_product_attention_onednn_p.def_impl(
    functools.partial(xla.apply_primitive, _dot_product_attention_onednn_p)
)
_dot_product_attention_onednn_p.def_abstract_eval(
    _dot_product_attention_fwd_p_abstract_eval
)

mlir.register_lowering(
  _dot_product_attention_onednn_p,
  _dot_product_attention_fwd_onednn_lowering,
  platform="cpu",
)

def _dot_product_attention_onednn_fwd(
    query: Array,
    key: Array,
    value: Array,
    combined_mask_not: Array,
    scale: float,
    attn_logits_soft_cap: float,
    mha_type_str: str,
    is_gqa: bool
    ):

  outputs = _dot_product_attention_onednn_p.bind(query, key, value, combined_mask_not, scale=scale, attn_logits_soft_cap=attn_logits_soft_cap, mha_type_str=mha_type_str, is_gqa=is_gqa)

  return outputs

def _dot_product_attention_onednn_bwd(
    query: Array,
    key: Array,
    value: Array,
    combined_mask_not: Array,
    scale: float,
    attn_logits_soft_cap: float,
    mha_type_str: str,
    is_gqa: bool
    ):

   raise NotImplementedError("Backward pass is not supported for dot_product_attention_onednn.")

@functools.partial(jax.custom_vjp, nondiff_argnums=(4,5,6,7))
def _dot_product_attention_custom_call(
    query: Array,
    key: Array,
    value: Array,
    combined_mask_not: Array,
    scale: float,
    attn_logits_soft_cap: float,
    mha_type_str: str,
    is_gqa: bool
    ):

    out = _dot_product_attention_onednn_fwd(
      query, key, value, combined_mask_not, scale=scale, attn_logits_soft_cap=attn_logits_soft_cap, mha_type_str=mha_type_str, is_gqa=is_gqa
    )

    return out

_dot_product_attention_custom_call.defvjp(_dot_product_attention_onednn_fwd, _dot_product_attention_onednn_bwd)

def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None,
    is_causal: bool,
    scale: float,
    attn_logits_soft_cap: Any | None = None):
    r"""CPU implementation of Fused Scaled dot product attention function.

    Computes the attention function on Query, Key, and Value tensors:

    Throughout this function, we utilize the following uppercase letters to
    represent the shape of array::

      B = batch size
      S = length of the key/value
      T = length of the query
      N = number of attention heads
      H = dimensions of each attention head
      K = number of key/value heads
      G = number of groups, which equals to N // K

    Args:
      query: query array; shape :code:`BTNH`
      key: key array: shape :code:`BSKH`. When `K` equals `N`, multi-headed
        attention (MHA https://arxiv.org/abs/1706.03762) is performed. Otherwise,
        grouped query attention (GQA https://arxiv.org/abs/2305.13245) is
        performed if `N` is a multiple of `K`, and multi-query attention (MQA
        https://arxiv.org/abs/1911.02150) is performed if `K == 1` (a special case
        of GQA).
      value: value array, should have the same shape as the `key` array.
      mask: optional, mask array used to filter out logits. It is a boolean mask
        where `True` indicates the element should take part in attention. For an
        additive mask, users should pass it to `bias`. The shape must be 4D and be
        broadcastable to :code:`BNTS`.
      scale: scale for the logits.
      is_causal: If true, causal attention will be applied.

    Returns:
      An array of the attention output with the same shape as :code:`query`.
    """

    # query shape: B, T, N, H
    # key/value shape: B, S, K, H
    is_gqa = False
    if query.shape[2] != key.shape[2]:
        is_gqa = True
    S = key.shape[1]
    K = key.shape[2]
    B, T, N, H = query.shape
    G = 1

    if is_gqa:
        G = N // K
        if mask is not None:
          mask = _reshape_to_grouped(mask, N, G, K)

    is_masked = True
    combined_mask_not = _get_masks(mask, is_causal, B, K, G, T, S, is_gqa)
    if combined_mask_not is None:
       is_masked = False
       # Dummy mask array, as jax jit doesn't like NoneType object
       combined_mask_not = jnp.zeros(0,dtype=query.dtype)

    is_attn_soft_cap = True
    if attn_logits_soft_cap is None:
       is_attn_soft_cap = False
       attn_logits_soft_cap = 1.0

    mha_type_str = _get_custom_call_name(is_masked, is_attn_soft_cap)
    out = _dot_product_attention_custom_call(
      query, key, value, combined_mask_not, scale=scale, attn_logits_soft_cap=attn_logits_soft_cap, mha_type_str=mha_type_str, is_gqa=is_gqa
    )

    return out
