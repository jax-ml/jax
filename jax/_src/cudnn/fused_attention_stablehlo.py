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

import enum
import functools
import json
import math
from typing import TypedDict

from jax._src import core
from jax._src import custom_derivatives
from jax._src import dispatch
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src import xla_bridge
from jax._src.custom_partitioning import custom_partitioning
from jax._src.custom_partitioning_sharding_rule import BATCHING, ArrayMapping, CompoundFactor, SdyShardingRule
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import parallel as lax_parallel
from jax._src.lib import cuda_versions
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.sharding_impls import NamedSharding, PartitionSpec
from jax._src.typing import Array

import numpy as np


class FP8Params(TypedDict):
  amax_dQ: float  # Amax of gradient of query
  amax_dK: float  # Amax of gradient of key
  amax_dV: float  # Amax of gradient of value
  amax_dP: float  # Amax of gradient of state
  descale_q: float  # Descaling factor of query
  descale_k: float  # Descaling factor of key
  descale_v: float  # Descaling factor of value
  descale_s: float  # Descaling factor of attention score
  scale_s: float  # Scale factor for S tensor
  scale_o: float  # Scale factor for output
  descale_o: float  # Descale factor for output (bwd)
  descale_dO: float  # Descale factor for output gradient (bwd)
  descale_dP: float  # Descale factor for P gradient tensor (bwd)
  scale_dQ: float  # Scale factor for query gradient (bwd)
  scale_dK: float  # Scale factor for key gradient (bwd)
  scale_dV: float  # Scale factor for value gradient (bwd)
  scale_dP: float  # Scale factor for state gradient (bwd)


class AttentionLayout(enum.Enum):
  BTNH = 0
  BNTH = 1


class MaskType(enum.Enum):
  NO_MASK = 0
  PADDING = 1
  CAUSAL = 2
  PADDING_CAUSAL = 3
  ALIBI = 4


def convert_mask_type_to_string(mask_type: MaskType) -> str:
  if mask_type == MaskType.NO_MASK:
    return "NO_MASK"
  elif mask_type == MaskType.PADDING:
    return "PADDING"
  elif mask_type == MaskType.CAUSAL:
    return "CAUSAL"
  elif mask_type == MaskType.PADDING_CAUSAL:
    return "PADDING_CAUSAL"
  elif mask_type == MaskType.ALIBI:
    return "ALIBI"
  else:
    raise ValueError(f"Unexpected mask type: {mask_type}")

def has_padding(mask_type: MaskType) -> bool:
  return mask_type == MaskType.PADDING or mask_type == MaskType.PADDING_CAUSAL

def should_export_dbias(bias_shape, query_shape, layout) -> bool:
  b_B, b_N, _, _ = bias_shape
  if layout == AttentionLayout.BNTH.value:
    _, q_N, _, _ = query_shape
  else:
    _, _, q_N, _ = query_shape
  return b_B == 1 and b_N == q_N

def get_large_negative_number(dtype):
  # temp WAR as cuDNN has a bug for subtraction between two large negative value
  if dtype == np.dtype('bfloat16'):
    return jnp.asarray(-2 << 40, dtype=dtype)
  elif dtype == np.dtype('float16'):
    return jnp.asarray(-2 << 14, dtype=dtype)
  else:
    raise ValueError("Unsupported dtype for inputs.")

def _normalize_layout(layout: str) -> AttentionLayout:
  layout_upper = layout.upper()
  if layout_upper in ["BSNH", "BNSH", "BTNH", "BNTH"]:
    return AttentionLayout[layout_upper.replace("S", "T")]
  else:
    raise ValueError(f"Unsupported qkv_layout: {layout}")

def element_type_to_backend_config_type_mapping(dtype):
  _element_type_to_backend_config_type_mapping = {
    ir.BF16Type.get(): "BF16",
    ir.F16Type.get(): "F16",
  }
  return _element_type_to_backend_config_type_mapping[dtype]

def default_layouts(*shapes):
  return [range(len(shape) - 1, -1, -1) for shape in shapes]

def get_max_seg_per_batch(q_offsets):
  return q_offsets.shape[1] - 1 if len(q_offsets.shape) == 2 else 1

def check_is_paged_attention(page_table_k):
  return len(page_table_k.shape) == 4

def create_dot_product_attention_backend_config_base(
    batch, num_heads, seq_q, seq_kv, dtype, fmha_scale, mask_type, layout, is_bwd
):
  # Q, K, V: query, key, value in shape of BT(S)NH or BNT(S)H
  # P: BMM1 output in shape of BNTS
  # O: BMM2 output in the same shape with Q
  # BMM1: Q @ K -> P
  # BMM2: P @ V -> O
  # BMM1Grad1: dP @ Q -> dK
  # BMM1Grad2: dP @ K -> dQ
  # BMM2Grad1: P @ dO -> dV
  # BMM2Grad2: dO @ V -> dP
  cudnn_fmha_backend_config = {
    "algorithm": {
      "algo_id": "0",
      "math_type": "TENSOR_OP_MATH",
      "tuning_knobs": {"17": "1", "24": "0"},
      "is_cudnn_frontend": True,
      "workspace_size": "0",
    },
    "fmha_scale": fmha_scale,
    "intermediate_tensor_shape": {
      "element_type": element_type_to_backend_config_type_mapping(dtype),
      "dimensions": [str(batch), str(num_heads), str(seq_q), str(seq_kv)],
      "tuple_shapes": [],
      "layout": {
        "dim_level_types": [],
        "dim_unique": [],
        "dim_ordered": [],
        "minor_to_major": ["3", "2", "1", "0"],
        "tiles": [],
        "element_size_in_bits": "0",
        "memory_space": "0",
        "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
        "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
        "dynamic_shape_metadata_prefix_bytes": "0",
      },
      "is_dynamic_dimension": [False, False, False, False],
    },
    "is_flash_attention": True,
    "mask_type": convert_mask_type_to_string(mask_type),
  }

  # We define the contracting and batch dims in the format of
  # ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
  # rhs_batch_dims)).
  if layout == AttentionLayout.BNTH.value:
    dims = [
        ((3, 3), ((0, 1), (0, 1))), # BMM1: BNTH,BNSH->BNTS
        ((3, 2), ((0, 1), (0, 1))), # BMM2: BNTS,BNSH->BNTH
        ((2, 2), ((0, 1), (0, 1))), # BMM1_grad_1: BNTS,BNTH->BNSH
        ((3, 2), ((0, 1), (0, 1))), # BMM1_grad_2: BNTS,BNSH->BNTH
        ((2, 2), ((0, 1), (0, 1))), # BMM2_grad_1: BNTS,BNTH->BNSH
        ((3, 3), ((0, 1), (0, 1))), # BMM2_grad_2: BNTH,BNSH->BNTS
    ]
  else:
    dims = [
        ((3, 3), ((0, 2), (0, 2))), # BMM1: BTNH,BSNH->BNTS
        ((3, 1), ((0, 1), (0, 2))), # BMM2: BNTS,BSNH->BTNH
        ((2, 1), ((0, 1), (0, 2))), # BMM1_grad_1: BNTS,BTNH->BSNH
        ((3, 1), ((0, 1), (0, 2))), # BMM1_grad_2: BNTS,BSNH->BTNH
        ((2, 1), ((0, 1), (0, 2))), # BMM2_grad_1: BNTS,BTNH->BSNH
        ((3, 3), ((0, 2), (0, 2))), # BMM2_grad_2: BTNH,BSNH->BNTS
    ]
  keys = [
      "bmm1_dot_dimension_numbers",
      "bmm2_dot_dimension_numbers",
      "bmm1_grad_gemm1_dot_dimension_numbers",
      "bmm1_grad_gemm2_dot_dimension_numbers",
      "bmm2_grad_gemm1_dot_dimension_numbers",
      "bmm2_grad_gemm2_dot_dimension_numbers",
  ]
  fwd_dot_number = {}
  bwd_dot_number = {}
  for idx, (key, ((lc, rc), (lb, rb))) in enumerate(zip(keys, dims)):
    dims_to_write = fwd_dot_number if idx < 2 else bwd_dot_number
    dims_to_write[key] = {
        "lhs_contracting_dimensions": [str(lc)],
        "rhs_contracting_dimensions": [str(rc)],
        "lhs_batch_dimensions": [str(i) for i in lb],
        "rhs_batch_dimensions": [str(i) for i in rb],
    }

  if is_bwd:
    cudnn_fmha_backend_config = {**cudnn_fmha_backend_config, **bwd_dot_number}
  else:
    cudnn_fmha_backend_config = {**cudnn_fmha_backend_config, **fwd_dot_number}
  backend_config = {
    "operation_queue_id":"0",
    "wait_on_operation_queues":[],
    "cudnn_fmha_backend_config": cudnn_fmha_backend_config
  }
  return backend_config

def create_dot_product_attention_backend_config(
    batch,
    num_heads,
    seq_q,
    seq_kv,
    dtype,
    fmha_scale,
    seed,
    dropout_rate,
    mask_type,
    layout,
    sliding_window_length,
    max_seg_per_batch,
    is_paged_attention,
    is_bwd
):
  backend_config = create_dot_product_attention_backend_config_base(
      batch, num_heads, seq_q, seq_kv, dtype,
      fmha_scale, mask_type, layout, is_bwd
  )
  if sliding_window_length is None:
    sliding_window_length = 0
  backend_config['cudnn_fmha_backend_config']["dropout_rate"] = dropout_rate
  backend_config['cudnn_fmha_backend_config']["seed"] = seed
  backend_config['cudnn_fmha_backend_config']["sliding_window_length"] = sliding_window_length
  backend_config['cudnn_fmha_backend_config']["max_seg_per_batch"] = max_seg_per_batch
  backend_config['cudnn_fmha_backend_config']["is_paged_attention"] = is_paged_attention
  return json.dumps(backend_config)

def create_dot_product_attention_fp8_backend_config(
    batch, num_heads, seq_q, seq_kv, dtype, fmha_scale, mask_type, layout, is_bwd):
  backend_config = create_dot_product_attention_backend_config_base(
      batch, num_heads, seq_q, seq_kv, dtype, fmha_scale, mask_type, layout, is_bwd)
  return json.dumps(backend_config)

# mapping from (is_bwd, has_dropout, has_bias) to custom call name
_custom_name_maps = {
  # fMHA forward call targets.
  (False, False, False, False): "__cudnn$fmhaSoftmax",
  (False, False, True, False): "__cudnn$fmhaScaleBiasSoftmax",
  (False, True, False, False): "__cudnn$fmhaSoftmaxDropout",
  (False, True, True, False): "__cudnn$fmhaScaleBiasSoftmaxDropout",
  (False, False, False, True): "__cudnn$fmhaSoftmaxF8",
  # fMHA backward call targets.
  (True, False, False, False): "__cudnn$fmhaSoftmaxBackward",
  (True, False, True, False): "__cudnn$fmhaScaleBiasSoftmaxBackward",
  (True, True, False, False): "__cudnn$fmhaSoftmaxDropoutBackward",
  (True, True, True, False): "__cudnn$fmhaScaleBiasSoftmaxDropoutBackward",
  (True, False, False, True): "__cudnn$fmhaSoftmaxBackwardF8",
}

def get_custom_call_name(has_bias, has_dropout, is_bwd, is_fp8=False):
  return _custom_name_maps[(is_bwd, has_dropout, has_bias, is_fp8)]

get_fp8_custom_call_name = functools.partial(
    get_custom_call_name, has_bias=False, has_dropout=False, is_fp8=True
)

def check_layout(query, key, value, bias, q_seqlen, kv_seqlen,
  q_offsets, kv_offsets, page_table_k, page_table_v, layout):
  def check_eq(a, b, c, msg):
    if not (a == b == c):
      raise ValueError(f"{msg} must be same, got {a}, {b}, {b}")

  q_rank, k_rank, v_rank = len(query.shape), len(key.shape), len(value.shape)
  if q_rank != 4:
    raise ValueError(f"Q must have a rank of 4, got {q_rank}")
  check_eq(q_rank, k_rank, v_rank, "QKV rank")

  q_dtype, k_dtype, v_dtype = query.dtype, key.dtype, value.dtype
  if q_dtype not in [np.float16, dtypes.bfloat16, dtypes.float8_e4m3fn, dtypes.float8_e5m2]:
    raise NotImplementedError(f"Q must be fp16/bf16/fp8_e4m3fn/fp8_e5m2, got {q_dtype}")
  check_eq(q_dtype, k_dtype, v_dtype, "QKV dtype")

  if layout == AttentionLayout.BNTH:
    qB, qN, qT, qH = query.shape
    kB, kN, kS, kH = key.shape
    vB, vN, vS, vH = value.shape
  else:
    assert layout == AttentionLayout.BTNH
    qB, qT, qN, qH = query.shape
    kB, kS, kN, kH = key.shape
    vB, vS, vN, vH = value.shape

  if page_table_k is not None and page_table_v is not None:
    k_blocks, k_block_size = kB, kS
    v_blocks, v_block_size = vB, vS
    kB, _, k_blocks_per_batch, _ = page_table_k.shape
    vB, _, v_blocks_per_batch, _ = page_table_v.shape
    kS = k_blocks_per_batch * k_block_size
    vS = v_blocks_per_batch * v_block_size
    if kB * k_blocks_per_batch != k_blocks:
      raise ValueError(
        f"Key and page_table_k must have same number of blocks, "
        f"got {k_blocks} vs {kB * k_blocks_per_batch}")
    if vB * v_blocks_per_batch != v_blocks:
      raise ValueError(
        f"Value and page_table_v must have same number of blocks, "
        f"got {v_blocks} vs {vB * v_blocks_per_batch}")

  check_eq(qB, kB, vB, "QKV batch")
  if qH != kH:
    raise ValueError(f"QK must have same head dim, got {qH} vs {kH}")
  if kN != vN:
    raise ValueError(f"KV must have same number of heads, got {kN} vs {vN}")
  if kS != vS:
    raise ValueError(f"KV must have same seq length, got {kS} vs {vS}")

  # check bias
  if bias is not None:
    _, _, bT, bS = bias.shape
    if bT != qT or bS != vS:
      raise ValueError(
        f"Bias must have same seq length as QKV, got {bT} and {bS}")

  # check q_seqlen/kv_seqlen/q_offsets/kv_offsets
  expected_rank = 2 if q_offsets is not None else 1
  def check_seqlen_offsets(tensor, name):
    if tensor is not None:
      dtype = tensor.dtype
      rank = len(tensor.shape)
      if dtype != np.dtype('int32'):
        raise ValueError(f"{name} must have int32 datatype, got {dtype}")
      if rank != expected_rank:
        raise ValueError(f"{name} must have a rank of {expected_rank}, got {rank}")
      b = tensor.shape[0]
      if b != qB:
        raise ValueError(f"{name} must have same batch as Q, got {b}")

  check_seqlen_offsets(q_seqlen, "q_seqlen")
  check_seqlen_offsets(kv_seqlen, "kv_seqlen")
  check_seqlen_offsets(q_offsets, "q_offsets")
  check_seqlen_offsets(kv_offsets, "kv_offsets")


def check_is_flash_attention(
    query, key, value, layout: int, cudnn_version, has_bias, is_training,
    is_packed=False, is_paged_attention=False, is_fp8=False):
    # Extract sequence length (T) and head dim (H) based on layout
    if layout == AttentionLayout.BNTH.value:
        _, _, T, qH = query.shape
        _, _, S, vH = value.shape
    else:
        _, T, _, qH = query.shape
        _, S, _, vH = value.shape

    if is_cuda_compute_capability_equal("10.3") and cudnn_version < 91100:
      # cudnn support compute_cap 10.3 on cudnn 9.11+
      raise NotImplementedError(
        "Compute capability 10.3 requires cuDNN version >= 9.11.")

    # Flash attention conditions
    if is_fp8:
        # FP8 specific conditions
        if not ((is_training and qH == 128 and T % 128 == 0 and S % 128 == 0) or
                (not is_training and qH <= 256 and qH % 16 == 0)):
            raise NotImplementedError(
                f"Unsupported sequence length Q {T}, KV {S} and head dim {qH} for FP8."
            )
    else:
        # bf16/fp16 attention conditions
        # Check the head dim.
        is_on_hopper = is_cuda_compute_capability_equal("9.0")
        H_max = 256 if cudnn_version >= 90500 and is_on_hopper else 128
        # check if multi-head latent attention is needed
        is_mla = qH != vH
        if not (qH <= H_max and qH % 8 == 0):
          raise NotImplementedError(
              f"The head dim must be <= {H_max} and a multiple of 8, "
              f"but got {qH}."
          )

        # Check patterns with bias, seqlen should be divisible by 2
        if (is_training and has_bias and (T % 2 != 0 or S % 2 != 0)):
          raise NotImplementedError(
              f"Unsupported sequence length Q {T}, KV {S}."
          )

        if is_packed and (cudnn_version < 90600 or not check_compute_capability("9.0")):
          raise NotImplementedError(
            "Packed layout requires cudnn version >= 9.6 and at least hopper arch.")
        if is_paged_attention and cudnn_version < 90500:
          raise NotImplementedError("Page attention requires cudnn version >= 9.5.")
        if is_mla and (cudnn_version < 91000 or not check_compute_capability("9.0")):
          raise NotImplementedError(
            "mla requires cudnn version >= 9.10 and at least hopper arch.")

def check_cudnn_version():
  # check if cuDNN is installed
  if cuda_versions is None:
    raise RuntimeError("cuDNN is not detected.")
  return cuda_versions.cudnn_get_version()

def check_compute_capability(capability):
  if not 'cuda' in xla_bridge.get_backend().platform_version:
    return False
  d, *_ = xla_bridge.local_devices(backend="gpu")
  target = tuple(int(x) for x in capability.split("."))
  current = tuple(int(x) for x in d.compute_capability.split("."))
  return current >= target

def is_cuda_compute_capability_equal(capability):
  if not 'cuda' in xla_bridge.get_backend().platform_version:
    return False
  d, *_ = xla_bridge.local_devices(backend="gpu")
  target = tuple(int(x) for x in capability.split("."))
  current = tuple(int(x) for x in d.compute_capability.split("."))
  return current == target

def _dot_product_attention_fwd(
    query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
    page_table_k, page_table_v,
    scale, seed, dropout_rate, variadic_args, mask_type, layout,
    sliding_window_length, cudnn_version, return_residual):
  # check if flash attention is supported for this attention pattern
  check_is_flash_attention(
      query, key, value, layout, cudnn_version, bias is not None, False,
      get_max_seg_per_batch(q_offsets) > 1, check_is_paged_attention(page_table_k))
  outputs = _dot_product_attention_fwd_p_wrapper.bind(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, scale=scale, seed=seed, dropout_rate=dropout_rate,
      variadic_args=variadic_args, mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length, is_training=False or return_residual)
  if return_residual:
    return tuple(outputs)
  else:
    return outputs[0]

def _dot_product_attention_fwd_rule(
    query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
    page_table_k, page_table_v, scale, seed, dropout_rate, variadic_args,
    mask_type, layout, sliding_window_length, cudnn_version,
    return_residual):
  # check if flash attention is supported for this attention pattern
  check_is_flash_attention(
      query, key, value, layout, cudnn_version, bias is not None, True,
      get_max_seg_per_batch(q_offsets) > 1)
  outputs = _dot_product_attention_fwd_p_wrapper.bind(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, scale=scale, seed=seed, dropout_rate=dropout_rate,
      variadic_args=variadic_args, mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length, is_training=True)
  res = (query, key, value, bias, q_seqlen, kv_seqlen, q_offsets,
         kv_offsets, page_table_k, page_table_v, outputs[1], outputs[0])
  if return_residual:
    return tuple(outputs), res
  else:
    return outputs[0], res

def _dot_product_attention_bwd_rule(
    scale, seed, dropout_rate, variadic_args, mask_type, layout,
    sliding_window_length, is_training, return_residual, res, grad_output):
  (query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
   page_table_k, page_table_v, activation, fwd_output) = res
  if return_residual:
    grad_output = grad_output[0]
  grads = _dot_product_attention_bwd_p_wrapper.bind(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, activation, fwd_output, grad_output,
      scale=scale, seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length
  )
  grads = (*grads,) + (None,) * (10 - len(grads))
  return grads

def _fix_seqlen_offsets(q_seqlen, kv_seqlen, q_offsets, kv_offsets, query, key):
  # fix seqlen and offsets to what cuDNN expects in sequence packing.
  # cuDNN expects seqlen to have shape [S] where S is the total number of segments
  # while the SDPA API accetps seqlen with shape [B, M] where B is the batch and M
  # is the maximum number of segments of one batch. B x M is larger than S and seqlen
  # is filled with -1 for padded regions. Therefore, we need to shift all non negative
  # values to left side to form a correct seqlen. Similar layout is required for
  # offsets tensors.
  # cuDNN expects offsets to have offset for each segment starting from first segment
  # while SDPA API accetps offsets to have offset for each segment starting from
  # current batch, therefore we need to calculate accumulative offset of each segment
  # starting from first segment.
  def _shift_to_left(x, fill_value):
    # shift any non-negative value to left
    # [[1, 3, -1, -1], [2, 3, 4, -1]]
    # -> [[1, 3, 2, 3], [4, -1, -1, -1]]
    x_shape = x.shape
    x = x.flatten()
    size = x.size
    indices = jnp.nonzero(x >= 0, size=size, fill_value=size)[0]
    y = jnp.take(x, indices, fill_value=fill_value)
    return jnp.reshape(y, x_shape)

  def _cu_offset(offsets, max_seq):
    # calculate accumulative offset by batch
    # [[1, 3, 5, 7], [4, 5, -1, -1]], max_seq = 8
    # -> [[1, 3, 5, 7], [12, 13, -1, -1]]
    batch = offsets.shape[0]
    offsets = jnp.where(
        offsets >= 0,
        offsets + (jnp.arange(batch, dtype=offsets.dtype) * max_seq)[..., np.newaxis],
        offsets,
    )
    return offsets

  if get_max_seg_per_batch(q_offsets) > 1:
    B, T, N, H = query.shape
    _, S, _, _ = key.shape

    q_seqlen = _shift_to_left(q_seqlen, 0)
    kv_seqlen = _shift_to_left(kv_seqlen, 0)

    q_offsets = _cu_offset(q_offsets, T)
    kv_offsets = _cu_offset(kv_offsets, S)
    q_offsets = _shift_to_left(q_offsets, B * T)
    kv_offsets = _shift_to_left(kv_offsets, B * S)

    # multiply by stride_per_token to get correct offsets
    # do it here because real stride changes after sharding
    q_offsets = q_offsets * N * H
    kv_offsets = kv_offsets * N * H

  return q_seqlen, kv_seqlen, q_offsets, kv_offsets

def _dot_product_attention_fwd_impl(
    query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
    page_table_k, page_table_v, scale, seed, dropout_rate, variadic_args,
    mask_type, layout, sliding_window_length, is_training):
  # args: {Q, K, V, mask*, bias*}
  q_seqlen, kv_seqlen, q_offsets, kv_offsets = \
      _fix_seqlen_offsets(q_seqlen, kv_seqlen, q_offsets, kv_offsets, query, key)
  outputs = _dot_product_attention_fwd_p.bind(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, scale=scale, seed=seed, dropout_rate=dropout_rate,
      variadic_args=variadic_args, mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length, is_training=is_training)
  return outputs

def _dot_product_attention_bwd_impl(
    query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
    page_table_k, page_table_v, activation, fwd_output, grad_output, scale,
    seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length):
  q_seqlen, kv_seqlen, q_offsets, kv_offsets = \
      _fix_seqlen_offsets(q_seqlen, kv_seqlen, q_offsets, kv_offsets, query, key)
  grads = _dot_product_attention_bwd_p.bind(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, activation, fwd_output, grad_output,
      scale=scale, seed=seed,
      dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length)
  return grads

def _dot_product_attention_fwd_abstract(
    query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
    page_table_k, page_table_v, *, scale, seed, dropout_rate, variadic_args,
    mask_type, layout, sliding_window_length, is_training):
  query_dtype = dtypes.canonicalize_dtype(query.dtype)
  if layout == AttentionLayout.BNTH.value:
    B, N, T, _ = query.shape
    _, _, S, H = value.shape
    output_shape = (B, N, T, H)
  else:
    B, T, N, _ = query.shape
    _, S, _, H = value.shape
    output_shape = (B, T, N, H)

  max_seg_per_batch = get_max_seg_per_batch(q_offsets)
  softmax_stat_shape = (B * max_seg_per_batch, N, T)

  if is_training:
    return (
      core.ShapedArray(output_shape, query_dtype),  # output
      core.ShapedArray(softmax_stat_shape, np.float32),  # softmax_stat
    )
  else:
    return (
      core.ShapedArray(output_shape, query_dtype),  # output
    )

def _dot_product_attention_bwd_abstract(
    query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
    page_table_k, page_table_v, activation, fwd_output, grad_output, *,
    scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length):
  query_dtype = dtypes.canonicalize_dtype(query.dtype)
  key_dtype = dtypes.canonicalize_dtype(key.dtype)
  value_dtype = dtypes.canonicalize_dtype(value.dtype)

  _, has_dbias = variadic_args
  if has_dbias:
    # cuDNN supports bias for this case
    bias_dtype = dtypes.canonicalize_dtype(bias.dtype)
    return (
      core.ShapedArray(
          query.shape, query_dtype
      ),  # grad query
      core.ShapedArray(
          key.shape, key_dtype
      ),  # grad key
      core.ShapedArray(
          value.shape, value_dtype
      ),  # grad value
      core.ShapedArray(
          bias.shape, bias_dtype
      ),  # grad bias
    )
  else:
    return (
      core.ShapedArray(
          query.shape, query_dtype
      ),  # grad query
      core.ShapedArray(
          key.shape, key_dtype
      ),  # grad key
      core.ShapedArray(
          value.shape, value_dtype
      ),  # grad value
    )

def _dot_product_attention_fwd_cuda_lowering(
    ctx, query, key, value, bias, q_seqlen, kv_seqlen, q_offsets,
    kv_offsets, page_table_k, page_table_v, scale, seed, dropout_rate,
    variadic_args, mask_type, layout, sliding_window_length, is_training):
  query_type = ir.RankedTensorType(query.type)
  query_shape = query_type.shape
  value_type = ir.RankedTensorType(value.type)
  value_shape = value_type.shape

  if layout == AttentionLayout.BNTH.value:
    B, N, T, qk_H = query_shape
    _, _, S, v_H = value_shape
    output_layout = (3, 2, 1, 0)
    output_transpose_perm = mlir.dense_int_array((0, 1, 2, 3))
  else:
    B, T, N, qk_H = query_shape
    _, S, _, v_H = value_shape
    output_layout = (3, 1, 2, 0)
    output_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))

  max_seg_per_batch = get_max_seg_per_batch(ir.RankedTensorType(q_offsets.type))
  is_paged_attention = check_is_paged_attention(ir.RankedTensorType(page_table_k.type))

  output_shape = (B, N, T, v_H)
  softmax_stat_shape = (B * max_seg_per_batch, N, T)
  workspace_shape = (0,)
  workspace_type = ir.IntegerType.get_unsigned(8)

  has_bias, _ = variadic_args
  backend_config = create_dot_product_attention_backend_config(
      B, N, T, S, query_type.element_type, scale, seed, dropout_rate,
      mask_type, layout, sliding_window_length, max_seg_per_batch,
      is_paged_attention, is_bwd=False)
  # {Q, K, V, bias*, q_seqlen*, kv_seqlen*,  q_offsets*, kv_offsets*}}
  # {output, activation*, workspace}
  has_dropout = dropout_rate > 0
  operands = [query, key, value]
  if has_bias:
    operands.append(bias)
  if has_padding(mask_type) or max_seg_per_batch > 1 or is_paged_attention:
    operands.append(q_seqlen)
    operands.append(kv_seqlen)
  if max_seg_per_batch > 1:
    operands.append(q_offsets)
    operands.append(kv_offsets)
  if is_paged_attention:
    operands.append(page_table_k)
    operands.append(page_table_v)

  custom_call_name = get_custom_call_name(has_bias, has_dropout, False)

  if is_training:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get(softmax_stat_shape, ir.F32Type.get()),
      ir.RankedTensorType.get(workspace_shape, workspace_type),
    ]
    result_layouts = [output_layout] + default_layouts(softmax_stat_shape, workspace_shape)
  else:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get(workspace_shape, workspace_type)
    ]
    result_layouts = [output_layout] + default_layouts(workspace_shape)
  # create custom call here
  out = mlir.custom_call(
    custom_call_name,
    result_types=result_types,
    operands=operands,
    backend_config=backend_config,
    operand_layouts=default_layouts(
      *[ir.RankedTensorType(operand.type).shape for operand in operands]),
    result_layouts=result_layouts,
  )
  # drop workspace memory
  # output should be (B, T, N, H) instead of (B, N, T, H)
  if is_training:
    return [hlo.transpose(out.results[0], output_transpose_perm), out.results[1]]
  else:
    return [hlo.transpose(out.results[0], output_transpose_perm)]

def _dot_product_attention_bwd_cuda_lowering(
    ctx, query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
    page_table_k, page_table_v, activation, fwd_output, grad_output,
    scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length):
  query_type = ir.RankedTensorType(query.type)
  query_shape = query_type.shape
  key_type = ir.RankedTensorType(key.type)
  value_type = ir.RankedTensorType(value.type)
  value_shape = value_type.shape

  if layout == AttentionLayout.BNTH.value:
    B, q_N, T, qk_H = query_shape
    _, v_N, S, v_H = value_shape
    grad_layout = (3, 2, 1, 0)
    grad_transpose_perm = mlir.dense_int_array((0, 1, 2, 3))
  else:
    B, T, q_N, qk_H = query_shape
    _, S, v_N, v_H = value_shape
    grad_layout = (3, 1, 2, 0)
    grad_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))

  workspace_shape = (0,)
  workspace_type = ir.IntegerType.get_unsigned(8)

  grad_query_shape = (B, q_N, T, qk_H)
  grad_key_shape = (B, v_N, S, qk_H)
  grad_value_shape = (B, v_N, S, v_H)

  has_bias, has_dbias = variadic_args
  max_seg_per_batch = get_max_seg_per_batch(ir.RankedTensorType(q_offsets.type))
  backend_config = create_dot_product_attention_backend_config(
      B, q_N, T, S, query_type.element_type, scale, seed, dropout_rate,
      mask_type, layout, sliding_window_length, max_seg_per_batch,
      False, is_bwd=True)
  # {Q, K, V, activation, dO, bias*, O, q_seqlen*, kv_seqlen*,
  #  q_offsets*, kv_offsets*}
  # {dQ, dK, dV, dbias*, workspace}
  has_dropout = dropout_rate > 0
  # create operands
  operands = [query, key, value, activation, grad_output]
  if has_bias:
    # flash attention requires bias in the bwd for remat
    operands.append(bias)
  operands.append(fwd_output)
  if has_padding(mask_type) or max_seg_per_batch > 1:
    operands.append(q_seqlen)
    operands.append(kv_seqlen)
  if max_seg_per_batch > 1:
    operands.append(q_offsets)
    operands.append(kv_offsets)
  # get custom call name
  custom_call_name = get_custom_call_name(has_bias, has_dropout, True)

  # create output types and layouts
  # grad_query, grad_key, grad_value
  result_types = [
    ir.RankedTensorType.get(grad_query_shape, query_type.element_type),
    ir.RankedTensorType.get(grad_key_shape, key_type.element_type),
    ir.RankedTensorType.get(grad_value_shape, value_type.element_type),
  ]
  result_layouts = [grad_layout, grad_layout, grad_layout]
  bias_type = ir.RankedTensorType(bias.type)
  bias_shape = bias_type.shape
  if has_dbias:
    # cuDNN supports bias for this case
    result_types.append(
      ir.RankedTensorType.get(bias_shape, bias_type.element_type))
    result_layouts = result_layouts + default_layouts(bias_shape)
  # workspace
  result_types.append(ir.RankedTensorType.get(workspace_shape, workspace_type))
  result_layouts = result_layouts + default_layouts(workspace_shape)
  out = mlir.custom_call(
    custom_call_name,
    result_types=result_types,
    operands=operands,
    backend_config=backend_config,
    operand_layouts=default_layouts(
      *[ir.RankedTensorType(operand.type).shape for operand in operands]),
    result_layouts=result_layouts,
  )
  dqkv = (hlo.transpose(out.results[0], grad_transpose_perm),
          hlo.transpose(out.results[1], grad_transpose_perm),
          hlo.transpose(out.results[2], grad_transpose_perm))
  # Only keep dQ, dK, dV and dBias here
  if has_dbias:
    return dqkv + (out.results[3],)
  else:
    return dqkv

# batcher
def _check_valid_batch_dims(bdims):
  for dim in bdims:
    if dim not in [0, None]:
      raise NotImplementedError(
        f"Currently only support batch_dim in [0, None], but got {dim=}")

def _dot_product_attention_fwd_batcher(
    batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args,
    mask_type, layout, sliding_window_length, is_training):
  _check_valid_batch_dims(batch_dims)
  query, key, value, bias, q_seqlen, kv_seqlen, \
    q_offsets, kv_offsets, page_table_k, page_table_v = batched_args
  query_bdim = batch_dims[0]
  if is_training:
    out_bdims = query_bdim, query_bdim
  else:
    out_bdims = (query_bdim,)

  if layout == AttentionLayout.BNTH.value:
    *Bs, N, T, _ = query.shape
    *_, _, S, _ = key.shape
  else:
    *Bs, T, N, _ = query.shape
    *_, S, _, _ = key.shape
  B = math.prod(Bs)
  has_bias, _ = variadic_args
  original_shape = query.shape
  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])
  if has_bias and batch_dims[3] is not None:
    bias = jnp.reshape(bias, (B, N, T, S))
  if has_padding(mask_type):
    q_seqlen = jnp.reshape(q_seqlen, (B, ))
    kv_seqlen = jnp.reshape(kv_seqlen, (B, ))

  outputs = _dot_product_attention_fwd_p_wrapper.bind(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, scale=scale, seed=seed, dropout_rate=dropout_rate,
      variadic_args=variadic_args, mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length, is_training=is_training)

  # reshape to original shape
  output = outputs[0]
  output = jnp.reshape(output, original_shape)
  if is_training:
    activation = outputs[1]
    activation = jnp.reshape(activation, (*Bs, N, T))
    return (output, activation), out_bdims
  else:
    return (output,), out_bdims

def _dot_product_attention_bwd_batcher(
     batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args,
     mask_type, layout, sliding_window_length):
  _check_valid_batch_dims(batch_dims)
  query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets, \
    page_table_k, page_table_v, activation, fwd_output, grad_output = batched_args
  query_bdim = batch_dims[0]
  out_bdims = query_bdim, query_bdim, query_bdim

  if layout == AttentionLayout.BNTH.value:
    *Bs, N, T, _ = query.shape
    *_, _, S, _ = key.shape
  else:
    *Bs, T, N, _ = query.shape
    *_, S, _, _ = key.shape
  B = math.prod(Bs)
  has_bias, has_dbias = variadic_args
  # Reset the has_dbias if the combined batch size is not 1, because cuDNN only
  # supports dbias with a single batch. In this case, an all-zero dbias will be
  # appended instead.
  if B > 1:
    variadic_args = (has_bias, False)
  original_query_shape = query.shape
  original_key_shape = key.shape
  original_value_shape = value.shape
  original_bias_shape = bias.shape if has_bias else None
  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])
  if has_bias and batch_dims[3] is not None:
    bias = jnp.reshape(bias, (B, N, T, S))
  if has_padding(mask_type):
    q_seqlen = jnp.reshape(q_seqlen, (B, ))
    kv_seqlen = jnp.reshape(kv_seqlen, (B, ))

  activation = jnp.reshape(activation, (B, N, T))
  fwd_output = jnp.reshape(fwd_output, (B,) + query.shape[-3:])
  grad_output = jnp.reshape(grad_output, (B,) + query.shape[-3:])

  grads = _dot_product_attention_bwd_p_wrapper.bind(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, activation, fwd_output, grad_output,
      scale=scale, seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length,
  )

  # reshape to original shape
  grads[0] = jnp.reshape(grads[0], original_query_shape)
  grads[1] = jnp.reshape(grads[1], original_key_shape)
  grads[2] = jnp.reshape(grads[2], original_value_shape)
  if has_dbias:
    assert has_bias
    if variadic_args[1]:
      grads[3] = jnp.reshape(grads[3], original_bias_shape)
    else:
      grads.append(jnp.zeros(original_bias_shape, bias.dtype))
    out_bdims += (batch_dims[3],)
  return grads, out_bdims

# custom partitioning
def _get_padded_spec(arg_info):
  spec = None if arg_info.sharding is None else arg_info.sharding.spec
  ndim = arg_info.ndim
  if spec is None:
    return (None,) * ndim
  assert len(spec) <= ndim
  return spec + (None,) * (ndim - len(spec))

def _check_qkv_bias_mask_spec(
    query_spec, key_spec, value_spec, bias_spec, layout):
  # check qkv spec
  if not query_spec == key_spec == value_spec:
    raise ValueError("Query, key and value should have same sharding.")
  if layout == AttentionLayout.BNTH.value:
    *batch_spec, num_head_spec, q_seq_spec, head_spec = query_spec
  else:
    *batch_spec, q_seq_spec, num_head_spec, head_spec = query_spec
  if q_seq_spec is not None:
    raise ValueError("Sharding on sequence dim is not allowed.")
  if head_spec is not None:
    raise ValueError("Sharding on head dim is not allowed.")
  # check bias spec
  if bias_spec:
    *bias_batch_spec, bias_num_head_spec, bias_q_seq_spec, bias_kv_seq_spec = bias_spec
    if any(bias_batch_spec) and bias_batch_spec != batch_spec or \
      bias_num_head_spec is not None and bias_num_head_spec != num_head_spec:
      raise ValueError(
        "Query and bias should have same sharding on batch and num_head dim.")
    if bias_q_seq_spec is not None or bias_kv_seq_spec is not None:
      raise ValueError("Sharding on bias sequence dim is not allowed.")


# fwd custom partition
def _infer_fwd_output_sharding(mesh, arg_shapes, variadic_args, is_training, layout):
  # only sharding on batch and num_head dim is allowed
  # (*batch, q_seq, num_head, head)
  query_spec = _get_padded_spec(arg_shapes[0])
  # (*batch, kv_seq, num_head, head)
  key_spec = _get_padded_spec(arg_shapes[1])
  value_spec = _get_padded_spec(arg_shapes[2])
  has_bias, _ = variadic_args
  bias_spec = _get_padded_spec(arg_shapes[3]) if has_bias else None

  _check_qkv_bias_mask_spec(
    query_spec, key_spec, value_spec, bias_spec, layout)
  # keep out sharding same as query sharding since they have same shape
  out_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
  if is_training:
    # activation sharding
    if layout == AttentionLayout.BNTH.value:
      *batch_spec, num_head_spec, q_seq_spec, _ = query_spec
    else:
      *batch_spec, q_seq_spec, num_head_spec, _ = query_spec
    activation_sharding = NamedSharding(
      mesh, PartitionSpec(*batch_spec, num_head_spec, q_seq_spec, None))
    return [out_sharding, activation_sharding]
  return [out_sharding]

def _fwd_shardy_rule(value_types, result_types, layout, is_training, is_fp8):
  num_args = len(value_types)
  # We only need the query and value sharding, so use placeholders for the remaining args.
  input_sharding = [ArrayMapping(f'{BATCHING}{n}') for n in range(num_args)]
  if layout == AttentionLayout.BNTH.value:
    input_sharding[0] = ArrayMapping('batch', 'nhead', 'qseq', 'head')
  else:
    input_sharding[0] = ArrayMapping('batch', 'qseq', 'nhead', 'head')
  input_sharding[2] += ('v',)

  # The major dimensions are sharded like the query, the minor like the value.
  output_sharding = (ArrayMapping(*input_sharding[0][:-1], 'v'),)
  if is_fp8:
    # `amax` is a scalar.
    amax = ArrayMapping(f'{BATCHING}{num_args}')
    output_sharding += (amax, amax)
  factor_sizes = {}
  if is_training:
    # Activation sharding.
    if result_types[-1].shape[0] == value_types[0].shape[0]:
      output_sharding += (ArrayMapping('batch', 'nhead', 'qseq'),)
    else:
      factor_sizes['n'] = result_types[-1].shape[0] // value_types[0].shape[0]
      output_sharding += (ArrayMapping(CompoundFactor('batch', 'n'), 'nhead', 'qseq'),)
  return SdyShardingRule(tuple(input_sharding), output_sharding, **factor_sizes)

_dot_product_attention_fwd_lower = custom_partitioning(
    _dot_product_attention_fwd_impl, static_argnums=(10, 11, 12, 13, 14, 15, 16, 17))

def _dot_product_attention_fwd_infer_sharding_from_operands(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length,
    is_training, mesh, arg_shapes, result_shape):
  return _infer_fwd_output_sharding(mesh, arg_shapes, variadic_args, is_training, layout)

def _dot_product_attention_fwd_shardy_rule(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length,
    is_training, mesh, value_types, result_types):
  return _fwd_shardy_rule(value_types, result_types, layout, is_training, is_fp8=False)

def _dot_product_attention_fwd_partition(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, sliding_window_length,
    is_training, mesh, arg_shapes, result_shape):
  # args sharding
  arg_shardings = tuple(arg_i.sharding for arg_i in arg_shapes)
  out_shardings = _infer_fwd_output_sharding(
    mesh, arg_shapes, variadic_args, is_training, layout)
  impl = functools.partial(
      _dot_product_attention_fwd_impl,
      scale=scale,
      seed=seed,
      dropout_rate=dropout_rate,
      variadic_args=variadic_args,
      mask_type=mask_type,
      layout=layout,
      sliding_window_length=sliding_window_length,
      is_training=is_training,
  )
  return mesh, impl, out_shardings, arg_shardings

# bwd custom partition
def _infer_bwd_output_sharding(mesh, arg_shapes, layout, variadic_args):
  # (*batch, q_seq, num_head, head)
  query_spec = _get_padded_spec(arg_shapes[0])
  # (*batch, kv_seq, num_head, head)
  key_spec = _get_padded_spec(arg_shapes[1])
  value_spec = _get_padded_spec(arg_shapes[2])
  has_bias, has_dbias = variadic_args
  bias_spec = _get_padded_spec(arg_shapes[3]) if has_bias else None
  _check_qkv_bias_mask_spec(
    query_spec, key_spec, value_spec, bias_spec, layout)
  # keep grad query sharding same as query sharding
  grad_query_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
  grad_key_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
  grad_value_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
  out_shardings = [grad_query_sharding, grad_key_sharding, grad_value_sharding]
  if has_dbias:
    grad_bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))
    out_shardings = out_shardings + [grad_bias_sharding]
  return out_shardings

def _bwd_shardy_rule(num_args, has_dbias, is_fp8):
  input_sharding = tuple(ArrayMapping(f'{BATCHING}{n}') for n in range(num_args))

  if has_dbias:
    output_sharding = input_sharding[0:4]
  else:
    output_sharding = input_sharding[0:3]
  if is_fp8:
    amax = ArrayMapping(f'{BATCHING}{num_args}')
    output_sharding += (amax, amax, amax, amax)
  return SdyShardingRule(input_sharding, output_sharding)

_dot_product_attention_bwd_lower = custom_partitioning(
    _dot_product_attention_bwd_impl, static_argnums=(13, 14, 15, 16, 17, 18, 19)
)

def _dot_product_attention_bwd_infer_sharding_from_operands(
    scale, seed, dropout_rate, variadic_args, mask_type, layout,
    sliding_window_length, mesh, arg_shapes, result_shape):
  return _infer_bwd_output_sharding(mesh, arg_shapes, layout, variadic_args)

def _dot_product_attention_bwd_shardy_rule(
    scale, seed, dropout_rate, variadic_args,
    mask_type, layout, sliding_window_length, mesh, value_types, result_types):
  _, has_dbias = variadic_args
  return _bwd_shardy_rule(len(value_types), has_dbias, is_fp8=False)

def _dot_product_attention_bwd_partition(
    scale, seed, dropout_rate, variadic_args, mask_type, layout,
    sliding_window_length, mesh, arg_shapes, result_shape):
  out_shardings = _infer_bwd_output_sharding(mesh, arg_shapes, layout, variadic_args)
  # args sharding
  arg_shardings = tuple(arg_i.sharding for arg_i in arg_shapes)
  def sharded_impl(*args):
    impl = functools.partial(
      _dot_product_attention_bwd_impl,
      scale=scale,
      seed=seed,
      dropout_rate=dropout_rate,
      variadic_args=variadic_args,
      mask_type=mask_type,
      layout=layout,
      sliding_window_length=sliding_window_length,
    )
    grads = impl(*args)
    _, has_dbias = variadic_args
    if has_dbias:
      query_spec = arg_shardings[0].spec
      batch_spec = query_spec[0]
      local_dbias = grads[3]
      global_dbias = lax_parallel.psum(local_dbias, batch_spec)
      grads = grads[:3] + [global_dbias]
    return grads
  return mesh, sharded_impl, out_shardings, arg_shardings

# Create dot_product_attention_fwd_p for forward operation.
_dot_product_attention_fwd_p = core.Primitive("dot_product_attention_fwd")
_dot_product_attention_fwd_p.multiple_results = True
_dot_product_attention_fwd_p.def_impl(
    functools.partial(dispatch.apply_primitive, _dot_product_attention_fwd_p)
)
_dot_product_attention_fwd_p.def_abstract_eval(
    _dot_product_attention_fwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_fwd_p,
  _dot_product_attention_fwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_fwd_p_wrapper = core.Primitive(
    "dot_product_attention_fwd_wrapper"
)
_dot_product_attention_fwd_p_wrapper.multiple_results = True
_dot_product_attention_fwd_p_wrapper.def_impl(_dot_product_attention_fwd_impl)
_dot_product_attention_fwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_fwd_abstract
)

# Create dot_product_attention_bwd_p for backward operation.
_dot_product_attention_bwd_p = core.Primitive("dot_product_attention_bwd")
_dot_product_attention_bwd_p.multiple_results = True
_dot_product_attention_bwd_p.def_impl(
    functools.partial(dispatch.apply_primitive, _dot_product_attention_bwd_p)
)
_dot_product_attention_bwd_p.def_abstract_eval(
    _dot_product_attention_bwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_bwd_p,
  _dot_product_attention_bwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_bwd_p_wrapper = core.Primitive(
    "dot_product_attention_bwd_wrapper"
)
_dot_product_attention_bwd_p_wrapper.multiple_results = True
_dot_product_attention_bwd_p_wrapper.def_impl(_dot_product_attention_bwd_impl)
_dot_product_attention_bwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_bwd_abstract
)

batching.primitive_batchers[
    _dot_product_attention_fwd_p_wrapper
] = _dot_product_attention_fwd_batcher
batching.primitive_batchers[
    _dot_product_attention_bwd_p_wrapper
] = _dot_product_attention_bwd_batcher

_dot_product_attention_fwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_fwd_infer_sharding_from_operands,
  partition=_dot_product_attention_fwd_partition,
  sharding_rule=_dot_product_attention_fwd_shardy_rule)

mlir.register_lowering(_dot_product_attention_fwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_fwd_lower, multiple_results=True))

_dot_product_attention_bwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_bwd_infer_sharding_from_operands,
  partition=_dot_product_attention_bwd_partition,
  sharding_rule=_dot_product_attention_bwd_shardy_rule)

mlir.register_lowering(_dot_product_attention_bwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_bwd_lower, multiple_results=True))

dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fwd_p_wrapper
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_bwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_bwd_p_wrapper
)

@functools.partial(custom_derivatives.custom_vjp, nondiff_argnums=(10, 11, 12, 13, 14, 15, 16, 17, 18))
def _dot_product_attention(query: Array,
                           key: Array,
                           value: Array,
                           bias: Array,
                           q_seqlen: Array,
                           kv_seqlen: Array,
                           q_offsets: Array,
                           kv_offsets: Array,
                           page_table_k: Array,
                           page_table_v: Array,
                           scale: float,
                           seed: int,
                           dropout_rate: float,
                           variadic_args: tuple[bool, ...],
                           mask_type: bool,
                           layout: int,
                           sliding_window_length: int | None,
                           cudnn_version: int,
                           return_residual: bool):
  output = _dot_product_attention_fwd(
      query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      page_table_k, page_table_v, scale=scale, seed=seed, dropout_rate=dropout_rate,
      variadic_args=variadic_args, mask_type=mask_type, layout=layout,
      sliding_window_length=sliding_window_length,
      cudnn_version=cudnn_version, return_residual=return_residual)
  return output

_dot_product_attention.defvjp(
    _dot_product_attention_fwd_rule, _dot_product_attention_bwd_rule
)

fp8_params_keys = [
    'amax_dQ', 'amax_dK', 'amax_dV', 'amax_dP', # place holder for bwd output
    'descale_q', 'descale_k', 'descale_v', 'descale_s',
    'scale_s', 'scale_o', 'descale_o', 'descale_dO',
    'descale_dP', 'scale_dQ', 'scale_dK', 'scale_dV',
    'scale_dP'
]

fp8_params_keys_fwd = [
    'descale_q', 'descale_k', 'descale_v', 'descale_s', 'scale_s', 'scale_o'
]
fp8_params_keys_bwd = [
    'descale_q', 'descale_k', 'descale_v', 'descale_o', 'descale_dO', 'descale_s',
    'descale_dP', 'scale_s', 'scale_dQ', 'scale_dK', 'scale_dV', 'scale_dP',
]
params_from_keys = lambda params, keys: [params[key] for key in keys]

def check_fp8_params(params):
  # Check if all required keys are present
  missing_keys = set(fp8_params_keys) - set(params)
  if missing_keys:
    raise ValueError(f"The following keys are missing from fp8_params: {', '.join(missing_keys)}")

check_is_flash_attention_fp8 = functools.partial(
    check_is_flash_attention,
    has_bias=False,
    is_fp8=True
)

def _dot_product_attention_fp8_fwd(
    query, key, value,
    fp8_params_fwd,
    scale, use_causal_mask, layout, cudnn_version):
  check_is_flash_attention_fp8(
      query, key, value, layout, cudnn_version, is_training=False)
  descale_q, descale_k, descale_v, descale_s, scale_s, scale_o = fp8_params_fwd
  outputs = _dot_product_attention_fp8_fwd_p_wrapper.bind(
      query, key, value,
      descale_q, descale_k, descale_v, descale_s,
      scale_s, scale_o,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout, is_training=False)
  return outputs

def _dot_product_attention_fp8_fwd_rule(
    query, key, value,
    fp8_params,
    scale, use_causal_mask, layout, cudnn_version):
  check_is_flash_attention_fp8(
      query, key, value, layout, cudnn_version, is_training=True)

  outputs = _dot_product_attention_fp8_fwd_p_wrapper.bind(
      query, key, value, *params_from_keys(fp8_params, fp8_params_keys_fwd),
      scale=scale, use_causal_mask=use_causal_mask, layout=layout, is_training=True)
  res = (query, key, value, outputs[3], outputs[0], params_from_keys(fp8_params, fp8_params_keys_bwd))
  return (outputs[0], outputs[1], outputs[2]), res

def _dot_product_attention_fp8_bwd_rule(
    scale, use_causal_mask, layout, cudnn_version, res, g):
  (query, key, value, activation, fwd_output, aux_params) = res
  grad_output = g[0]
  grads = _dot_product_attention_fp8_bwd_p_wrapper.bind(
    query,
    key,
    value,
    fwd_output,
    grad_output,
    activation,
    *aux_params,
    scale=scale,
    use_causal_mask=use_causal_mask,
    layout=layout,
    )

  fp8_params_grads = dict.fromkeys(fp8_params_keys)
  keys_to_grad_indices = ['amax_dQ', 'amax_dK', 'amax_dV', 'amax_dP']
  # grads structure: (dQ, dK, dV, amax_dq, amax_dk, amax_dv, amax_dp)
  for i, key in enumerate(keys_to_grad_indices, start=3):
    fp8_params_grads[key] = grads[i]

  return (grads[0], grads[1], grads[2], fp8_params_grads)

def _dot_product_attention_fp8_fwd_impl(
    query, key, value,
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    scale, use_causal_mask, layout, is_training):
  outputs = _dot_product_attention_fp8_fwd_p.bind(
      query,
      key,
      value,
      descale_q,
      descale_k,
      descale_v,
      descale_s,
      scale_s,
      scale_o,
      scale=scale,
      use_causal_mask=use_causal_mask,
      layout=layout,
      is_training=is_training,
  )
  return outputs

def _dot_product_attention_fp8_bwd_impl(
    query, key, value, fwd_output, grad_output, activation,
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
    descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
    scale, use_causal_mask, layout):
  grads = _dot_product_attention_fp8_bwd_p.bind(
      query, key, value, fwd_output, grad_output, activation,
      descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
      descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout)
  return grads


def _dot_product_attention_fp8_fwd_abstract(
    query, key, value,
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    scale, use_causal_mask, layout, is_training):
  query_dtype = dtypes.canonicalize_dtype(query.dtype)
  if layout == AttentionLayout.BNTH.value:
    B, N, T, _ = query.shape
    _, _, S, _ = key.shape
  else:
    B, T, N, _ = query.shape
    _, S, _, _ = key.shape
  output_shape = query.shape
  softmax_stat_shape = (B, N, T)

  # output, amax_s, amax_o[, softmax_stat]
  if is_training:
    return (
      core.ShapedArray(output_shape, query_dtype),
      core.ShapedArray((1,1,1,1), np.float32),
      core.ShapedArray((1,1,1,1), np.float32),
      core.ShapedArray(softmax_stat_shape, np.float32),
    )
  else:
    return (
      core.ShapedArray(output_shape, query_dtype),
      core.ShapedArray((1,1,1,1), np.float32),
      core.ShapedArray((1,1,1,1), np.float32),
    )

def _dot_product_attention_fp8_bwd_abstract(
    query, key, value, fwd_output, grad_output, activation,
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
    descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
    scale, use_causal_mask, layout):
  query_dtype = dtypes.canonicalize_dtype(query.dtype)
  key_dtype = dtypes.canonicalize_dtype(key.dtype)
  value_dtype = dtypes.canonicalize_dtype(value.dtype)

  amax_shape = (1,1,1,1)

  return (
    core.ShapedArray(query.shape, query_dtype),
    core.ShapedArray(key.shape, key_dtype),
    core.ShapedArray(value.shape, value_dtype),
    core.ShapedArray(amax_shape, np.float32),
    core.ShapedArray(amax_shape, np.float32),
    core.ShapedArray(amax_shape, np.float32),
    core.ShapedArray(amax_shape, np.float32),
  )

def _dot_product_attention_fp8_fwd_cuda_lowering(
    ctx, query, key, value,
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    scale, use_causal_mask, layout, is_training):
  query_type = ir.RankedTensorType(query.type)
  query_shape = query_type.shape
  key_type = ir.RankedTensorType(key.type)
  key_shape = key_type.shape

  if layout == AttentionLayout.BNTH.value:
    B, N, T, H = query_shape
    _, _, S, _ = key_shape
    output_layout = (3, 2, 1, 0)
    output_transpose_perm = mlir.dense_int_array((0, 1, 2, 3))
  else:
    B, T, N, H = query_shape
    _, S, _, _ = key_shape
    output_layout = (3, 1, 2, 0)
    output_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))

  output_shape = (B, N, T, H)
  softmax_stat_shape = (B, N, T)
  workspace_shape = (0,)
  amax_shape = (1,1,1,1)
  workspace_type = ir.IntegerType.get_unsigned(8)
  mask_type = MaskType.CAUSAL if use_causal_mask else MaskType.NO_MASK
  backend_config = create_dot_product_attention_fp8_backend_config(
      B, N, T, S, ir.BF16Type.get(),  # query_type.element_type,
      scale, mask_type, layout, is_bwd=False,
  )

  operands = [query, key, value, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o]
  custom_call_name = get_fp8_custom_call_name(is_bwd=False)

  if is_training:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get(softmax_stat_shape, ir.F32Type.get()),
      ir.RankedTensorType.get(workspace_shape, workspace_type),
    ]
    result_layouts = [output_layout] + default_layouts(amax_shape, amax_shape, softmax_stat_shape, workspace_shape)
  else:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get(workspace_shape, workspace_type)
    ]
    result_layouts = [output_layout] + default_layouts(amax_shape, amax_shape, workspace_shape)

  operand_shapes = [ir.RankedTensorType(operand.type).shape for operand in operands[:3]]
  operand_shapes += [[1, 1, 1, 1]] * 6
  operand_layouts = default_layouts(*operand_shapes)
  out = mlir.custom_call(
    custom_call_name,
    result_types=result_types,
    operands=operands,
    backend_config=backend_config,
    operand_layouts=operand_layouts,
    result_layouts=result_layouts,
  )

  if is_training:
    return [hlo.transpose(out.results[0], output_transpose_perm), out.results[1], out.results[2], out.results[3]]
  else:
    return [hlo.transpose(out.results[0], output_transpose_perm), out.results[1], out.results[2]]



def _dot_product_attention_fp8_bwd_cuda_lowering(
    ctx, query, key, value, fwd_output, grad_output, activation,
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
    descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, scale,
    use_causal_mask, layout):
  query_type = ir.RankedTensorType(query.type)
  query_shape = query_type.shape
  key_type = ir.RankedTensorType(key.type)
  key_shape = key_type.shape
  value_type = ir.RankedTensorType(value.type)

  if layout == AttentionLayout.BNTH.value:
    B, q_N, T, H = query_shape
    _, k_N, S, _ = key_shape
    grad_layout = (3, 2, 1, 0)
    grad_transpose_perm = mlir.dense_int_array((0, 1, 2, 3))
  else:
    B, T, q_N, H = query_shape
    _, S, k_N, _ = key_shape
    grad_layout = (3, 1, 2, 0)
    grad_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))

  workspace_shape = (0,)
  workspace_type = ir.IntegerType.get_unsigned(8)
  amax_shape = (1,1,1,1)

  grad_query_shape = (B, q_N, T, H)
  grad_key_shape = (B, k_N, S, H)
  grad_value_shape = (B, k_N, S, H)
  mask_type = MaskType.CAUSAL if use_causal_mask else MaskType.NO_MASK

  backend_config = create_dot_product_attention_fp8_backend_config(
      B, q_N, T, S, ir.BF16Type.get(),
      scale, mask_type, layout, is_bwd=True,
  )

  operands = [
    query,
    key,
    value,
    fwd_output,
    grad_output,
    activation,
    descale_q,
    descale_k,
    descale_v,
    descale_o,
    descale_dO,
    descale_s,
    descale_dP,
    scale_s,
    scale_dQ,
    scale_dK,
    scale_dV,
    scale_dP,
  ]

  custom_call_name = get_fp8_custom_call_name(is_bwd=True)

  result_types = [
    ir.RankedTensorType.get(grad_query_shape, query_type.element_type),
    ir.RankedTensorType.get(grad_key_shape, key_type.element_type),
    ir.RankedTensorType.get(grad_value_shape, value_type.element_type),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
  ]
  result_layouts = [grad_layout, grad_layout, grad_layout] + default_layouts(amax_shape, amax_shape, amax_shape, amax_shape)

  result_types.append(ir.RankedTensorType.get(workspace_shape, workspace_type))
  result_layouts = result_layouts + default_layouts(workspace_shape)
  out = mlir.custom_call(
    custom_call_name,
    result_types=result_types,
    operands=operands,
    backend_config=backend_config,
    operand_layouts=default_layouts(
      *[ir.RankedTensorType(operand.type).shape for operand in operands]),
    result_layouts=result_layouts,
  )
  dqkv_amaxs = (hlo.transpose(out.results[0], grad_transpose_perm),
          hlo.transpose(out.results[1], grad_transpose_perm),
          hlo.transpose(out.results[2], grad_transpose_perm),
          out.results[3], out.results[4], out.results[5], out.results[6])
  # Only keep dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP here
  return dqkv_amaxs

def _dot_product_attention_fp8_fwd_batcher(
    batched_args, batch_dims, *, scale, use_causal_mask, layout, is_training):
  _check_valid_batch_dims(batch_dims)
  query, key, value,\
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, = batched_args
  query_bdim = batch_dims[0]
  if is_training:
    out_bdims = query_bdim, query_bdim
  else:
    out_bdims = (query_bdim,)

  if layout == AttentionLayout.BNTH.value:
    *Bs, N, T, _ = query.shape
    *_, _, S, _ = key.shape
  else:
    *Bs, T, N, _ = query.shape
    *_, S, _, _ = key.shape
  B = math.prod(Bs)

  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])

  outputs = _dot_product_attention_fp8_fwd_p_wrapper.bind(
      query, key, value, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout, is_training=is_training)

  # reshape to original shape
  output, amax_s, amax_o = outputs[0], outputs[1], outputs[2]
  output = jnp.reshape(output, query.shape)
  if is_training:
    activation = outputs[3]
    activation = jnp.reshape(activation, (*Bs, N, T))
    return (output, amax_s, amax_o, activation), out_bdims
  else:
    return (output, amax_s, amax_o), out_bdims

def _dot_product_attention_fp8_bwd_batcher(
    batched_args, batch_dims, *, scale, use_causal_mask, layout):
  _check_valid_batch_dims(batch_dims)
  query, key, value, fwd_output, grad_output, activation,\
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP,\
    scale_s, scale_dQ, scale_dK, scale_dV, scale_dP = batched_args
  query_bdim = batch_dims[0]
  out_bdims = query_bdim, query_bdim, query_bdim

  if layout == AttentionLayout.BNTH.value:
    *Bs, N, T, _ = query.shape
    *_, _, S, _ = key.shape
  else:
    *Bs, T, N, _ = query.shape
    *_, S, _, _ = key.shape
  B = math.prod(Bs)

  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])

  activation = jnp.reshape(activation, (B, N, T))
  fwd_output = jnp.reshape(fwd_output, (B,) + query.shape[-3:])
  grad_output = jnp.reshape(grad_output, (B,) + query.shape[-3:])

  grads = _dot_product_attention_fp8_bwd_p_wrapper.bind(
      query, key, value, fwd_output, grad_output, activation,
      descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout,
  )

  grad_query, grad_key, grad_value = grads[:3]
  # reshape to original shape
  grad_query = jnp.reshape(grad_query, query.shape)
  grad_key = jnp.reshape(grad_key, key.shape)
  grad_value = jnp.reshape(grad_value, value.shape)

  return grads, out_bdims

def _infer_fp8_fwd_output_sharding(mesh, arg_shapes, is_training, layout):
  # Prepare variadic_args for the original function
  has_bias = False  # Adjust as needed
  variadic_args = (has_bias, None)  # Dummy value, adjust as necessary

  # Call the original function with the required parameters
  output_sharding = _infer_fwd_output_sharding(mesh, arg_shapes, variadic_args, is_training, layout)
  amax_sharding = NamedSharding(mesh, PartitionSpec())
  if is_training:
    out_sharding, activation_sharding = output_sharding[0], output_sharding[1]
    return [out_sharding, amax_sharding, amax_sharding, activation_sharding]
  return output_sharding + [amax_sharding, amax_sharding]

_dot_product_attention_fp8_fwd_lower = custom_partitioning(
    _dot_product_attention_fp8_fwd_impl, static_argnums=(9, 10, 11, 12))

def _dot_product_attention_fp8_fwd_infer_sharding_from_operands(
    scale, use_causal_mask, layout, is_training,
    mesh, arg_shapes, result_shape):
  return _infer_fp8_fwd_output_sharding(mesh, arg_shapes, is_training, layout)

def _dot_product_attention_fp8_fwd_partition(
    scale, use_causal_mask, layout, is_training,
    mesh, arg_shapes, result_shape):
  # args sharding
  arg_shardings = tuple(arg_i.sharding for arg_i in arg_shapes)
  out_shardings = _infer_fp8_fwd_output_sharding(
    mesh, arg_shapes, is_training, layout)
  impl = functools.partial(
      _dot_product_attention_fp8_fwd_impl, scale=scale, use_causal_mask=use_causal_mask,
      layout=layout, is_training=is_training)
  return mesh, impl, out_shardings, arg_shardings

def _dot_product_attention_fp8_fwd_shardy_rule(
    scale, use_causal_mask, layout, is_training,
    mesh, value_types, result_types):
  return _fwd_shardy_rule(value_types, result_types, layout, is_training, is_fp8=True)

def _infer_fp8_bwd_output_sharding(mesh, arg_shapes, layout):
  # Prepare variadic_args for the original function
  has_bias = False  # Adjust as needed
  has_dbias = False  # Adjust as needed
  variadic_args = (has_bias, has_dbias)  # Dummy value, adjust as necessary

  # Call the original function with the required parameters
  output_shardings = _infer_bwd_output_sharding(mesh, arg_shapes, layout, variadic_args)

  # Prepare amax_sharding
  amax_sharding = NamedSharding(mesh, PartitionSpec())  # Use a default spec or adjust as needed

  # Append amax_sharding for each output sharding
  out_shardings_with_amax = output_shardings + [amax_sharding] * 4

  return out_shardings_with_amax

_dot_product_attention_fp8_bwd_lower = custom_partitioning(
    _dot_product_attention_fp8_bwd_impl, static_argnums=(18,19,20)
)

def _dot_product_attention_fp8_bwd_infer_sharding_from_operands(
    scale, use_causal_mask, layout, mesh,
    arg_shapes, result_shape):
  return _infer_fp8_bwd_output_sharding(mesh, arg_shapes, layout)

def _dot_product_attention_fp8_bwd_shardy_rule(
    scale, use_causal_mask, layout, mesh, value_types, result_types):
  return _bwd_shardy_rule(len(value_types), has_dbias=False, is_fp8=True)

def _dot_product_attention_fp8_bwd_partition(
    scale, use_causal_mask, layout, mesh,
    arg_shapes, result_shape):
  out_shardings = _infer_fp8_bwd_output_sharding(mesh, arg_shapes, layout)
  # args sharding
  arg_shardings = tuple(arg_i.sharding for arg_i in arg_shapes)
  impl = functools.partial(
      _dot_product_attention_fp8_bwd_impl, scale=scale,
      use_causal_mask=use_causal_mask, layout=layout
  )
  return mesh, impl, out_shardings, arg_shardings

# Create dot_product_attention_fp8_fwd_p for forward operation.
_dot_product_attention_fp8_fwd_p = core.Primitive("dot_product_attention_fp8_fwd")
_dot_product_attention_fp8_fwd_p.multiple_results = True
_dot_product_attention_fp8_fwd_p.def_impl(
    functools.partial(dispatch.apply_primitive, _dot_product_attention_fp8_fwd_p)
)
_dot_product_attention_fp8_fwd_p.def_abstract_eval(
    _dot_product_attention_fp8_fwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_fp8_fwd_p,
  _dot_product_attention_fp8_fwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_fp8_fwd_p_wrapper = core.Primitive(
    "dot_product_attention_fp8_fwd_wrapper"
)
_dot_product_attention_fp8_fwd_p_wrapper.multiple_results = True
_dot_product_attention_fp8_fwd_p_wrapper.def_impl(_dot_product_attention_fp8_fwd_impl)
_dot_product_attention_fp8_fwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_fp8_fwd_abstract
)

# Create dot_product_attention_bwd_p for backward operation.
_dot_product_attention_fp8_bwd_p = core.Primitive("dot_product_attention_fp8_bwd")
_dot_product_attention_fp8_bwd_p.multiple_results = True
_dot_product_attention_fp8_bwd_p.def_impl(
    functools.partial(dispatch.apply_primitive, _dot_product_attention_fp8_bwd_p)
)
_dot_product_attention_fp8_bwd_p.def_abstract_eval(
    _dot_product_attention_fp8_bwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_fp8_bwd_p,
  _dot_product_attention_fp8_bwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_fp8_bwd_p_wrapper = core.Primitive(
    "dot_product_attention_fp8_bwd_wrapper"
)
_dot_product_attention_fp8_bwd_p_wrapper.multiple_results = True
_dot_product_attention_fp8_bwd_p_wrapper.def_impl(_dot_product_attention_fp8_bwd_impl)
_dot_product_attention_fp8_bwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_fp8_bwd_abstract
)

batching.primitive_batchers[
    _dot_product_attention_fp8_fwd_p_wrapper
] = _dot_product_attention_fp8_fwd_batcher
batching.primitive_batchers[
    _dot_product_attention_fp8_bwd_p_wrapper
] = _dot_product_attention_fp8_bwd_batcher

_dot_product_attention_fp8_fwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_fp8_fwd_infer_sharding_from_operands,
  partition=_dot_product_attention_fp8_fwd_partition,
  sharding_rule=_dot_product_attention_fp8_fwd_shardy_rule)

mlir.register_lowering(_dot_product_attention_fp8_fwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_fp8_fwd_lower, multiple_results=True))

_dot_product_attention_fp8_bwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_fp8_bwd_infer_sharding_from_operands,
  partition=_dot_product_attention_fp8_bwd_partition,
  sharding_rule=_dot_product_attention_fp8_bwd_shardy_rule)

mlir.register_lowering(_dot_product_attention_fp8_bwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_fp8_bwd_lower, multiple_results=True))

dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_fwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_fwd_p_wrapper
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_bwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_bwd_p_wrapper
)

@functools.partial(custom_derivatives.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _dot_product_attention_fp8(query: Array,
                               key: Array,
                               value: Array,
                               fp8_params: dict[str, Array],
                               scale: float,
                               use_causal_mask: bool,
                               layout: int,
                               cudnn_version: int):
  output, amax_s, amax_o = _dot_product_attention_fp8_fwd(
      query, key, value, params_from_keys(fp8_params, fp8_params_keys_fwd),
      scale, use_causal_mask, layout, cudnn_version
  )
  return output, amax_s, amax_o

_dot_product_attention_fp8.defvjp(_dot_product_attention_fp8_fwd_rule, _dot_product_attention_fp8_bwd_rule)

def combine_bias_and_mask(bias, mask, dtype):
  if bias is not None:
    # reshape bias to have 4D shape
    bias = bias.reshape((1,) * (4 - len(bias.shape)) + bias.shape)

  if mask is not None:
    if mask.dtype == np.dtype('bool'):
      large_negative_number = get_large_negative_number(dtype)
      mask = jnp.where(mask, jnp.asarray(0, dtype), large_negative_number)
    # reshape mask to have 4D shape
    mask = mask.reshape((1,) * (4 - len(mask.shape)) + mask.shape)  # type: ignore[union-attr]

  # combine bias and mask
  if bias is None:
    bias = mask
  else:
    if mask is not None:
      # should be broadcast to same shape
      bias = bias + mask
  return bias

# User interface
def paged_attention(
    query: Array,
    key: Array,
    value: Array,
    q_seqlen: Array,
    kv_seqlen: Array,
    page_table_k: Array,
    page_table_v: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    fp8_params: FP8Params | None = None,
    *,
    scale: float = 1.0,
    mask_type: MaskType = MaskType.NO_MASK,
    seed: int = 42,
    dropout_rate: float = 0.,
    qkv_layout: str = "BTNH",
    sliding_window_length: int | None = None,
    use_fp8: bool = False,
    return_residual: bool = False
):
  """Computes paged attention described in https://arxiv.org/pdf/2309.06180.

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    N = number of attention heads
    H = dimensions of each attention head.

  Args:
    query: Queries for attention calculation with a shape of BTNH or BNTH.
    key: Keys for attention calculation with a shape of
      [num_blocks, block_size, N, H] or [num_blocks, N, block_size, H] where
      num_blocks = B * Ceil(S / block_size).
    value: Values to be used in attention with a shape of
      [num_blocks, block_size, N, H] or [num_blocks, N, block_size, H] where
      num_blocks = B * Ceil(S / block_size).
    q_seqlen: Non padded sequence length of query with a shape of B.
    kv_seqlen: Non padded sequence length of key and value with a shape of B.
    page_table_k: page table for key of shape [B, 1, num_blocks_per_batch, 1]
      where num_blocks_per_batch = Ceil(S / block_size).
    page_table_v: page table for value of shape [B, 1, num_blocks_per_batch, 1]
      where num_blocks_per_batch = Ceil(S / block_size).
    bias: Bias to be added to logits with a shape of BNTS.
    mask: Mask used to filter out logits with a shape of BNTS.
    scale: Scale for the query.
    qkv_layout: Layout string, with supported formats being BTNH, BNTH, BSNH,
      BNSH.
    sliding_window_length: Window size to make attention only attend to each
      token's left local window (pos - sliding_window_length, pos] where `pos`
      is the index of each token. E.g., if sliding_window_length == 3 and the
      sequence is [0, 1, 2, 3, c, 4, 5], token `c` can attend to [4, 5, c].
    use_fp8: Whether to use FP8 attention mechanism.
    return_residual: Whether to return the logsumexp tensor of shape BTN
      or BNT to users. See section 3.1.1 in the FlashAttention-2 paper:
      https://arxiv.org/pdf/2307.08691 to find the definition of logsumexp.
  Returns:
    output: the same shape as the query.
    residual: the logsumexp tensor if return_residual=True. (non fp8)
  """
  cudnn_version = check_cudnn_version()
  layout = _normalize_layout(qkv_layout)
  if use_fp8:
    raise ValueError("Paged attention doesn't support fp8 for now.")
  if has_padding(mask_type) and (q_seqlen is None or kv_seqlen is None):
    raise ValueError("Require q_seqlen and kv_seqlen to generate padding mask.")
  if sliding_window_length is not None and sliding_window_length <= 0:
    raise ValueError(
      f"Require sliding_window_length > 0, got {sliding_window_length}.")

  bias = combine_bias_and_mask(bias, mask, query.dtype)
  # check if input shape and data type is compatiable
  check_layout(query, key, value, bias, q_seqlen, kv_seqlen, None, None,
    page_table_k, page_table_v, layout)
  has_bias = bias is not None
  has_dbias = has_bias and \
    should_export_dbias(bias.shape, query.shape, layout)  # type: ignore[union-attr]
  variadic_args = (has_bias, has_dbias)

  _not_used = jnp.zeros(0, dtype=query.dtype)
  if bias is None:
    bias = _not_used

  output = _dot_product_attention(
      query, key, value, bias, q_seqlen, kv_seqlen, _not_used, _not_used,
      page_table_k, page_table_v, scale, seed, dropout_rate, variadic_args,
      mask_type, layout.value, sliding_window_length, cudnn_version,
      return_residual)
  return output


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    q_seqlen: Array | None = None,
    kv_seqlen: Array | None = None,
    q_offsets: Array | None = None,
    kv_offsets: Array | None = None,
    fp8_params: FP8Params | None = None,
    *,
    scale: float = 1.0,
    mask_type: MaskType = MaskType.NO_MASK,
    seed: int = 42,
    dropout_rate: float = 0.,
    qkv_layout: str = "BTNH",
    sliding_window_length: int | None = None,
    use_fp8: bool = False,
    return_residual: bool = False
):
  """Computes dot-product attention given query (Q), key (K), and value (V).

  This function serves as the core operation for applying attention
  mechanisms as described in the paper [https://arxiv.org/abs/1706.03762].
  Initially, it determines the attention weights by processing Q and K,
  subsequently combining the outcomes using K. Throughout this function, we
  utilize the following uppercase letters to represent specific parameters of
  array:

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    N = number of attention heads
    H = dimensions of each attention head.

  The supported layouts for Q, K, V are either BT(S)NH or BNT(S)H, and they must
  adhere to the same layout. The output layout remains consistent with Q,
  defaulting to BT(S)NH.

  Args:
    query: Queries for attention calculation with a shape of BTNH or BNTH.
    key: Keys for attention calculation with a shape of BSNH or BNSH.
    value: Values to be used in attention with a shape of BSNH or BNSH.
    bias: Bias to be added to logits with a shape of BNTS.
    mask: Mask used to filter out logits with a shape of BNTS.
    q_seqlen: Non padded sequence length of query with a shape of B.
      If q_offsets is set, q_seqlen should have shape [B,M] where M is the
      maximum number of segments per batch. For batch that has less segments
      than maximum segments, fill the padded entries with -1.
    kv_seqlen: Non padded sequence length of key and value with a shape of B.
      If kv_offsets is set, kv_seqlen should have shape [B,M] where M is the
      maximum number of segments per batch. For batch that has less segments
      than maximum segments, fill the padded entries with -1.
    q_offsets: offset of each segment packed in query with a shape of [B,M+1]
      where M is the maximum number of segments per batch. For batch that has
      less segments than maximum segments, fill the padded entries with -1.
      E.g, if 2 batches has 3 and 2 segments respectively, each segment has
      size 1, q_offsets = [[0,1,2,-1], [0,1,-1,-1]]. q_seqlen should be set
      to indicate the size of each segment.
    kv_offsets: offset of each segment packed in key with a shape of [B,M+1]
      where M is the maximum number of segments per batch. For batch that has
      less segments than maximum segments, fill the padded entries with -1.
      E.g, if 2 batches has 3 and 2 segments respectively, each segment has
      size 1, kv_offsets = [[0,1,2,-1], [0,1,-1,-1]]. kv_seqlen should be set
      to indicate the size of each segment.
    scale: Scale for the query.
    dropout_rate: Dropout rate.
    qkv_layout: Layout string, with supported formats being BTNH, BNTH, BSNH,
      BNSH.
    sliding_window_length: Window size to make attention only attend to each
      token's left local window (pos - sliding_window_length, pos] where `pos`
      is the index of each token. E.g., if sliding_window_length == 3 and the
      sequence is [0, 1, 2, 3, c, 4, 5], token `c` can attend to [4, 5, c].
    use_fp8: Whether to use FP8 attention mechanism.
    return_residual: Whether to return the logsumexp tensor of shape BTN
      or BNT to users. See section 3.1.1 in the FlashAttention-2 paper:
      https://arxiv.org/pdf/2307.08691 to find the definition of logsumexp.
  Returns:
    output: the same shape as the query.
    residual: the logsumexp tensor if return_residual=True. (non fp8)
    amax_s: amax of state. (fp8 only)
    amax_o: amax of output. (fp8 only)
  """
  # TODO(b/380898464): Check the compute capability, e.g., require GPU device,
  # in the kernel implementation (c++) code.
  cudnn_version = check_cudnn_version()
  layout = _normalize_layout(qkv_layout)

  if use_fp8:
    if fp8_params is None:
      raise ValueError("fp8_params should not be None.")
    if  mask_type not in (MaskType.NO_MASK, MaskType.CAUSAL):
      raise ValueError("Only NO_MASK or CAUSAL masks are supported for fp8.")
    if not all(x is None for x in [bias, mask, q_seqlen, kv_seqlen]):
      raise ValueError(
          f"Expected 'None' for bias, mask, q_seqlen, and kv_seqlen, "
          f"but got: bias={bias}, mask={mask}, q_seqlen={q_seqlen}, kv_seqlen={kv_seqlen}"
      )
    check_fp8_params(fp8_params)
    check_layout(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      None, None, layout)
    output, amax_s, amax_o = _dot_product_attention_fp8(
        query, key, value, fp8_params,
        scale, mask_type == MaskType.CAUSAL, layout.value, cudnn_version
    )
    return output, amax_s, amax_o
  else:
    if has_padding(mask_type) and (q_seqlen is None or kv_seqlen is None):
        raise ValueError("Require q_seqlen and kv_seqlen to generate padding mask")
    if sliding_window_length is not None and sliding_window_length <= 0:
      raise ValueError(
        f"Require sliding_window_length > 0, got {sliding_window_length}")
    if q_offsets is not None and (q_seqlen is None or kv_seqlen is None):
      raise ValueError("Require q_seqlen and kv_seqlen to use packed layout")

    bias = combine_bias_and_mask(bias, mask, query.dtype)
    # check if input shape and data type is compatiable
    check_layout(query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
      None, None, layout)
    has_bias = bias is not None
    has_dbias = has_bias and \
      should_export_dbias(bias.shape, query.shape, layout)  # type: ignore[union-attr]
    variadic_args = (has_bias, has_dbias)

    _not_used = jnp.zeros(0, dtype=query.dtype)
    if bias is None:
      bias = _not_used
    if q_seqlen is None:
      q_seqlen = _not_used
    if kv_seqlen is None:
      kv_seqlen = _not_used
    if q_offsets is None:
      q_offsets = _not_used
    if kv_offsets is None:
      kv_offsets = _not_used

    output = _dot_product_attention(
        query, key, value, bias, q_seqlen, kv_seqlen, q_offsets, kv_offsets,
        _not_used, _not_used, scale, seed, dropout_rate, variadic_args,
        mask_type, layout.value, sliding_window_length, cudnn_version,
        return_residual)
    return output
