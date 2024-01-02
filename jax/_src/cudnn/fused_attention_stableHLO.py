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

from functools import partial, reduce
import operator
from typing import Any, Optional
import json

import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from jax._src.lib.mlir.dialects import hlo
from jax._src.core import ShapedArray

from jax.experimental.custom_partitioning import custom_partitioning
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from jax._src.interpreters import batching
from jax._src import dispatch
from jax._src.lib import cuda_versions

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray

def element_type_to_backend_config_type_mapping(dtype):
    _element_type_to_backend_config_type_mapping = {
        ir.BF16Type.get(): "BF16",
        ir.F16Type.get(): "F16",
    }
    return _element_type_to_backend_config_type_mapping.get(dtype)

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def create_dot_product_attention_backend_config(batch,
                                                num_heads,
                                                seq_q,
                                                seq_kv,
                                                dtype,
                                                fmha_scale,
                                                seed,
                                                dropout_rate,
                                                is_flash_attention,
                                                is_causal_mask,
                                                is_bwd):
    # b q_seq num_heads head_dim  -> Q
    # b kv_seq num_heads head_dim -> K
    # b kv_seq num_heads head_dim -> V
    # b num_heads q_seq kv_seq -> P
    # b q_seq num_heads head_dim -> O
    # bmm1: Q @ K -> P
    # bmm2: P @ V -> O
    # bmm2Grad1: P @ dO -> dV
    # bmm2Grad2: dO @ V -> dP
    # bmm1Grad1: dP @ Q -> dK
    # bmm1Grad2: dP @ K -> dQ
    backend_config = {
        "algorithm":{"algo_id":"0","math_type":"TENSOR_OP_MATH","tuning_knobs":{"17":"1","24":"0"},"is_cudnn_frontend":True,"workspace_size":"0"},
        "fmha_scale":fmha_scale,
        "dropout_rate":dropout_rate,
        "intermediate_tensor_shape":{"element_type":element_type_to_backend_config_type_mapping(dtype),"dimensions":[str(batch),str(num_heads),str(seq_q),str(seq_kv)],"tuple_shapes":[],"layout":{"dim_level_types":[],"dim_unique":[],"dim_ordered":[],"minor_to_major":["3","2","1","0"],"tiles":[],"element_size_in_bits":"0","memory_space":"0","index_primitive_type":"PRIMITIVE_TYPE_INVALID","pointer_primitive_type":"PRIMITIVE_TYPE_INVALID","dynamic_shape_metadata_prefix_bytes":"0"},"is_dynamic_dimension":[False,False,False,False]},
        "seed":seed,
        "is_flash_attention":is_flash_attention,
        "is_causal_mask":is_causal_mask
        }
    fwd_dot_number = {
        "bmm1_dot_dimension_numbers":{"lhs_contracting_dimensions":["3"],"rhs_contracting_dimensions":["3"],"lhs_batch_dimensions":["0","2"],"rhs_batch_dimensions":["0","2"]},
        "bmm2_dot_dimension_numbers":{"lhs_contracting_dimensions":["3"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0","1"],"rhs_batch_dimensions":["0","2"]},
    }
    bwd_dot_number = {
        "bmm1_grad_gemm1_dot_dimension_numbers":{"lhs_contracting_dimensions":["2"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0","1"],"rhs_batch_dimensions":["0","2"]},
        "bmm1_grad_gemm2_dot_dimension_numbers":{"lhs_contracting_dimensions":["3"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0","1"],"rhs_batch_dimensions":["0","2"]},
        "bmm2_grad_gemm1_dot_dimension_numbers":{"lhs_contracting_dimensions":["2"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":["0","1"],"rhs_batch_dimensions":["0","2"]},
        "bmm2_grad_gemm2_dot_dimension_numbers":{"lhs_contracting_dimensions":["3"],"rhs_contracting_dimensions":["3"],"lhs_batch_dimensions":["0","2"],"rhs_batch_dimensions":["0","2"]},
    }
    if is_bwd:
        backend_config = {**backend_config, **bwd_dot_number}
    else:
        backend_config = {**backend_config, **fwd_dot_number}

    backend_config = json.dumps(backend_config)
    return backend_config

def get_custom_call_name(has_bias, has_mask, has_dropout, is_bwd):
    index = is_bwd << 3 | has_dropout << 2 | has_mask << 1 | has_bias
    _custom_name_maps = [
        # fMHA forward call targets.
        "__cudnn$fhmaSoftmax",
        "__cudnn$fhmaScaleBiasSoftmax",
        "__cudnn$fhmaScaleMaskSoftmax",
        "__cudnn$fhmaScaleBiasMaskSoftmax",
        "__cudnn$fhmaSoftmaxDropout",
        "__cudnn$fhmaScaleBiasSoftmaxDropout",
        "__cudnn$fhmaScaleMaskSoftmaxDropout",
        "__cudnn$fhmaScaleBiasMaskSoftmaxDropout",
        # fMHA backward call targets.
        "__cudnn$fhmaSoftmaxBackward",
        "__cudnn$fhmaScaleBiasSoftmaxBackward",
        "__cudnn$fhmaScaleMaskSoftmaxBackward",
        "__cudnn$fhmaScaleBiasMaskSoftmaxBackward",
        "__cudnn$fhmaSoftmaxDropoutBackward",
        "__cudnn$fhmaScaleBiasSoftmaxDropoutBackward",
        "__cudnn$fhmaScaleMaskSoftmaxDropoutBackward",
        "__cudnn$fhmaScaleBiasMaskSoftmaxDropoutBackward"
    ]
    return _custom_name_maps[index]

def check_qkv_layout(query, key, value):
    assert len(query.shape) == len(key.shape) == len(value.shape) == 4, \
        "query, key and value should have rank 4."
    
    # Only support fp16 and bf16 here
    query_dtype = query.dtype
    key_dtype = key.dtype
    value_dtype = value.dtype
    assert query_dtype == key_dtype == value_dtype and query_dtype in [jnp.float16, jnp.bfloat16], \
        "query, key and value should have same dtype and should be float16 or bfloat16"

    q_batch, q_seq_len, q_num_heads, q_head_dim = query.shape
    k_batch, k_seq_len, k_num_heads, k_head_dim = key.shape
    v_batch, v_seq_len, v_num_heads, v_head_dim = value.shape
    assert (q_batch == k_batch == v_batch) \
        and (k_seq_len == v_seq_len) \
        and (q_num_heads == k_num_heads == v_num_heads) \
        and (q_head_dim == k_head_dim == v_head_dim), \
        "query should have layout [batch, q_seq, num_heads, head_dim], " \
        "key and value should have layout [batch, kv_seq, num_heads, head_dim]."

def check_is_flash_attention(query, key):
    batch, q_seq_len, num_heads, head_dim = query.shape
    _, kv_sqe_len, _, _ = key.shape
    # check if attention pattern is supported by flash attention or fused attention
    if q_seq_len > 512 and q_seq_len == kv_sqe_len and head_dim in [64, 128]:
        # check if flash attention is supported
        is_flash_attention = True
    elif q_seq_len <= 512 and kv_sqe_len <= 512 and head_dim == 64:
        # check if regular fused attention is supported
        is_flash_attention = False
    else:
        raise NotImplementedError("Unsupported sequence length and head dim.")
    return is_flash_attention

def check_cuDNN_version(is_flash_attention):
    # check if cuDNN is installed and if cuDNN version contraint is satisfied
    if cuda_versions is None:
        raise RuntimeError("cuDNN is not detected.")
    elif is_flash_attention and cuda_versions.cudnn_get_version() < 8903:
        raise RuntimeError("Require cuDNN at lease 8.9.3 to run flash attention.")
    elif not is_flash_attention and cuda_versions.cudnn_get_version() < 8901:
        raise RuntimeError("Require cuDNN at lease 8.9.1 to run fused attention.")

def _dot_product_attention_fwd(query, key, value, bias, mask,
    scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    output, _ = _dot_product_attention_fwd_p_wrapper.bind(
        query, key, value, bias, mask, scale=scale, seed=seed, dropout_rate=dropout_rate,
        variadic_args=variadic_args, is_flash_attention=is_flash_attention,
        is_causal_mask=is_causal_mask)
    return output

def _dot_product_attention_fwd_rule(query, key, value, bias, mask, 
    scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    output, activation = _dot_product_attention_fwd_p_wrapper.bind(
        query, key, value, bias, mask, scale=scale, seed=seed, dropout_rate=dropout_rate,
        variadic_args=variadic_args, is_flash_attention=is_flash_attention,
        is_causal_mask=is_causal_mask)
    res = (query, key, value, bias, mask, activation, output)
    return output, res

def _dot_product_attention_bwd_rule(scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask, res, grad_output):
    # {Q, K, V, bias, mask, activation, fwd_output, dO}
    query, key, value, bias, mask, activation, fwd_output = res
    grad_query, grad_key, grad_value = _dot_product_attention_bwd_p_wrapper.bind(
        query, key, value, bias, mask, activation, fwd_output, grad_output,
        scale=scale, seed=seed, dropout_rate=dropout_rate,
        variadic_args=variadic_args, is_flash_attention=is_flash_attention,
        is_causal_mask=is_causal_mask)
    grads = (grad_query, grad_key, grad_value, None, None)
    return grads

def _dot_product_attention_fwd_impl(query, key, value, bias, mask,
    scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    # args: {Q, K, V, mask*, bias*}
    output, activation = _dot_product_attention_fwd_p.bind(
        query, key, value, bias, mask, scale=scale, seed=seed, dropout_rate=dropout_rate,
        variadic_args=variadic_args, is_flash_attention=is_flash_attention,
        is_causal_mask=is_causal_mask)
    return output, activation

def _dot_product_attention_bwd_impl(query, key, value, bias, mask, activation, fwd_output, grad_output,
    scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    grad_query, grad_key, grad_value = _dot_product_attention_bwd_p.bind(
        query, key, value, bias, mask, activation, fwd_output, grad_output,
        scale=scale, seed=seed, dropout_rate=dropout_rate,
        variadic_args=variadic_args, is_flash_attention=is_flash_attention,
        is_causal_mask=is_causal_mask)
    grads = (grad_query, grad_key, grad_value)
    return grads

def _dot_product_attention_fwd_abstract(query, key, value, bias, mask,
    *, scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    query_dtype = dtypes.canonicalize_dtype(query.dtype)
    batch, q_seq_len, num_heads, head_dim = query.shape
    _, kv_seq_len, _, _ = key.shape
    output_shape = (batch, q_seq_len, num_heads, head_dim)
    activation_shape = (batch, num_heads, q_seq_len, kv_seq_len)
    softmax_stat_shape = (batch, num_heads, q_seq_len)
    if q_seq_len > 512:
        # is flash attention
        return (
            ShapedArray(output_shape, query_dtype),  # output
            ShapedArray(softmax_stat_shape, jnp.float32),  # softmax_stat
        )
    else:
        return (
            ShapedArray(output_shape, query_dtype),  # output
            ShapedArray(activation_shape, query_dtype),  # activation
        )

def _dot_product_attention_bwd_abstract(query, key, value, bias, mask, activation, fwd_output, grad_output,
    *, scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    query_dtype = dtypes.canonicalize_dtype(query.dtype)
    key_dtype = dtypes.canonicalize_dtype(key.dtype)
    value_dtype = dtypes.canonicalize_dtype(value.dtype)

    return (
        ShapedArray(
            query.shape, query_dtype
        ),  # grad query
        ShapedArray(
            key.shape, key_dtype
        ),  # grad key
        ShapedArray(
            value.shape, value_dtype
        ),  # part value
    )

def _dot_product_attention_fwd_cuda_lowering(ctx, query, key, value, bias, mask,
    scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    query_type = ir.RankedTensorType(query.type)
    query_shape = query_type.shape
    key_type = ir.RankedTensorType(key.type)
    key_shape = key_type.shape
    value_type = ir.RankedTensorType(value.type)
    value_shape = value_type.shape
    
    batch, q_seq_len, num_heads, head_dim = query_shape
    _, kv_seq_len, _, _ = key_shape

    output_shape = (batch, num_heads, q_seq_len, head_dim)
    output_layout = (3, 1, 2, 0)
    output_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))
    activation_shape = (batch, num_heads, q_seq_len, kv_seq_len)
    softmax_stat_shape = (batch, num_heads, q_seq_len)
    scratch_shape = (0,)
    scratch_type = ir.IntegerType.get_unsigned(8)
    # get backend config
    backend_config = create_dot_product_attention_backend_config(batch, num_heads, q_seq_len, kv_seq_len, query_type.element_type, scale, seed, dropout_rate, is_flash_attention, is_causal_mask, False)
    # {Q, K, V, mask*, bias*}
    # {output, scratch, activation*}
    has_dropout = dropout_rate > 0
    has_bias, has_mask = variadic_args
    operands = [query, key, value]
    if has_mask:
        operands.append(mask)
    if has_bias:
        operands.append(bias)
    # get custom call name
    custom_call_name = get_custom_call_name(has_bias, has_mask, has_dropout, False)
    # create output types and layouts
    if is_flash_attention:
        result_types = [
            ir.RankedTensorType.get(output_shape, query_type.element_type),
            ir.RankedTensorType.get(scratch_shape, scratch_type),
            ir.RankedTensorType.get(softmax_stat_shape, ir.F32Type.get()),
        ]
        result_layouts = [output_layout] + default_layouts(scratch_shape, softmax_stat_shape)
    else:
        result_types = [
            ir.RankedTensorType.get(output_shape, query_type.element_type),
            ir.RankedTensorType.get(scratch_shape, scratch_type),
            ir.RankedTensorType.get(activation_shape, query_type.element_type),
        ]
        result_layouts = [output_layout] + default_layouts(scratch_shape, activation_shape)
    # create custom call here
    out = custom_call(
        custom_call_name,
        result_types=result_types,
        operands=operands,
        backend_config=backend_config,
        operand_layouts=default_layouts(*[ir.RankedTensorType(operand.type).shape for operand in operands]),
        result_layouts=result_layouts,
    )
    # dropout scratch memory
    # output should be (batch, q_seq_len, num_heads, head_dim) instead of (batch, num_heads, q_seq_len, head_dim)
    return [hlo.transpose(out.results[0], output_transpose_perm), out.results[2]]

def _dot_product_attention_bwd_cuda_lowering(ctx, query, key, value, bias, mask, activation, fwd_output, grad_output,
    scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    query_type = ir.RankedTensorType(query.type)
    query_shape = query_type.shape
    key_type = ir.RankedTensorType(key.type)
    key_shape = key_type.shape
    value_type = ir.RankedTensorType(value.type)
    value_shape = value_type.shape
    activation_type = ir.RankedTensorType(activation.type)
    activation_shape = activation_type.shape
    grad_output_type = ir.RankedTensorType(grad_output.type)
    grad_output_shape = grad_output_type.shape

    batch, q_seq_len, num_heads, head_dim = query_shape
    _, kv_seq_len, _, _ = key_shape
    scratch_shape = (0,)
    scratch_type = ir.IntegerType.get_unsigned(8)

    grad_query_shape = (batch, num_heads, q_seq_len, head_dim)
    grad_key_shape = (batch, num_heads, kv_seq_len, head_dim)
    grad_value_shape = (batch, num_heads, kv_seq_len, head_dim)
    softmax_sum_shape = (batch, num_heads, q_seq_len)
    grad_layout = (3, 1, 2, 0)
    grad_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))
    backend_config = create_dot_product_attention_backend_config(batch, num_heads, q_seq_len, kv_seq_len, query_type.element_type, scale, seed, dropout_rate, is_flash_attention, is_causal_mask, True)
    # {Q, K, V, activation, dO, mask*, bias*, O*}
    # {dQ, dK, dV, d_S*, softmax_sum*, d_Q_accum*, scratch, dbias*}
    has_dropout = dropout_rate > 0
    has_bias, has_mask = variadic_args
    # create operands
    operands = [query, key, value, activation, grad_output]
    if has_mask:
        operands.append(mask)
    if has_bias and is_flash_attention:
        # flash attention requires bias in the bwd for remat
        operands.append(bias)
    if is_flash_attention:
        operands.append(fwd_output)
    # get custom call name
    custom_call_name = get_custom_call_name(has_bias, has_mask, has_dropout, True)

    # create output types and layouts
    if is_flash_attention:
        result_types = [
            ir.RankedTensorType.get(grad_query_shape, query_type.element_type), # grad query
            ir.RankedTensorType.get(grad_key_shape, key_type.element_type), # grad key
            ir.RankedTensorType.get(grad_value_shape, value_type.element_type), # grad value
            ir.RankedTensorType.get(softmax_sum_shape, ir.F32Type.get()), # softmax_sum
            ir.RankedTensorType.get(grad_query_shape, ir.F32Type.get()), # d_Q_accum
            ir.RankedTensorType.get(scratch_shape, scratch_type), # scratch
        ]
        result_layouts = [grad_layout, grad_layout, grad_layout] + default_layouts(softmax_sum_shape, grad_query_shape, scratch_shape)
    else:
        result_types = [
            ir.RankedTensorType.get(grad_query_shape, query_type.element_type), # grad query
            ir.RankedTensorType.get(grad_key_shape, key_type.element_type), # grad key
            ir.RankedTensorType.get(grad_value_shape, value_type.element_type), # grad value
            ir.RankedTensorType.get(activation_shape, activation_type.element_type), # dS
            ir.RankedTensorType.get(scratch_shape, scratch_type), # scratch
        ]
        result_layouts = [grad_layout, grad_layout, grad_layout] + default_layouts(activation_shape, scratch_shape)
    out = custom_call(
        custom_call_name,
        result_types=result_types,
        operands=operands,
        backend_config=backend_config,
        operand_layouts=default_layouts(*[ir.RankedTensorType(operand.type).shape for operand in operands]),
        result_layouts=result_layouts,
    )
    # Only keep dQ, dK and dV here
    return [hlo.transpose(out.results[0], grad_transpose_perm),
            hlo.transpose(out.results[1], grad_transpose_perm),
            hlo.transpose(out.results[2], grad_transpose_perm)]

# batcher
def _check_valid_batch_dims(bdims):
    for dim in bdims:
        assert dim in [0, None], \
            "Currently only support batch_dim in [0, None], " \
            f"but got {dim=}"

def _dot_product_attention_fwd_batcher(batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    _check_valid_batch_dims(batch_dims)
    query, key, value, bias, mask = batched_args
    query_bdim = batch_dims[0]
    out_bdims = query_bdim, query_bdim

    *batch_tuple, q_seq_len, num_heads, head_dim = query.shape
    *_, kv_seq_len, _, _ = key.shape
    batch = reduce(operator.mul, batch_tuple)
    has_bias, has_mask = variadic_args
    # reshape to 4D shape
    query = jnp.reshape(query, (batch, q_seq_len, num_heads, head_dim))
    key = jnp.reshape(key, (batch, kv_seq_len, num_heads, head_dim))
    value = jnp.reshape(value, (batch, kv_seq_len, num_heads, head_dim))
    if has_bias:
        bias = jnp.reshape(bias, (batch, num_heads, q_seq_len, kv_seq_len))
    if has_mask:
        mask = jnp.reshape(mask, (batch, num_heads, q_seq_len, kv_seq_len))

    output, activation = _dot_product_attention_fwd_p_wrapper.bind(
        query, key, value, bias, mask,
        scale=scale, seed=seed, dropout_rate=dropout_rate,
        variadic_args=variadic_args, is_flash_attention=is_flash_attention,
        is_causal_mask=is_causal_mask)
    
    # reshape to original shape
    output = jnp.reshape(output, (*batch_tuple, q_seq_len, num_heads, head_dim))
    if is_flash_attention:
        activation = jnp.reshape(activation, (*batch_tuple, num_heads, q_seq_len))
    else:
        activation = jnp.reshape(activation, (*batch_tuple, num_heads, q_seq_len, kv_seq_len))
    return (output, activation), out_bdims

def _dot_product_attention_bwd_batcher(batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask):
    _check_valid_batch_dims(batch_dims)
    query, key, value, bias, mask, activation, fwd_output, grad_output = batched_args
    query_bdim = batch_dims[0]
    out_bdims = query_bdim, query_bdim, query_bdim

    *batch_tuple, q_seq_len, num_heads, head_dim = query.shape
    *_, kv_seq_len, _, _ = key.shape
    batch = reduce(operator.mul, batch_tuple)
    has_bias, has_mask = variadic_args
    # reshape to 4D shape
    query = jnp.reshape(query, (batch, q_seq_len, num_heads, head_dim))
    key = jnp.reshape(key, (batch, kv_seq_len, num_heads, head_dim))
    value = jnp.reshape(value, (batch, kv_seq_len, num_heads, head_dim))
    if has_bias:
        bias = jnp.reshape(bias, (batch, num_heads, q_seq_len, kv_seq_len))
    if has_mask:
        mask = jnp.reshape(mask, (batch, num_heads, q_seq_len, kv_seq_len))
    if is_flash_attention:
        activation = jnp.reshape(activation, (batch, num_heads, q_seq_len))
    else:
        activation = jnp.reshape(activation, (batch, num_heads, q_seq_len, kv_seq_len))
    fwd_output = jnp.reshape(fwd_output, (batch, q_seq_len, num_heads, head_dim))
    grad_output = jnp.reshape(grad_output, (batch, q_seq_len, num_heads, head_dim))

    grad_query, grad_key, grad_value = _dot_product_attention_bwd_p_wrapper.bind(
        query, key, value, bias,
        mask, activation, fwd_output, grad_output,
        scale=scale, seed=seed, dropout_rate=dropout_rate,
        variadic_args=variadic_args, is_flash_attention=is_flash_attention,
        is_causal_mask=is_causal_mask)
    
    # reshape to original shape
    grad_query = jnp.reshape(grad_query, (*batch_tuple, q_seq_len, num_heads, head_dim))
    grad_key = jnp.reshape(grad_key, (*batch_tuple, kv_seq_len, num_heads, head_dim))
    grad_value = jnp.reshape(grad_value, (*batch_tuple, kv_seq_len, num_heads, head_dim))
    grads = (grad_query, grad_key, grad_value)
    return grads, out_bdims

# custom partitioning
def _get_padded_spec(arg_info):
    spec = None if arg_info.sharding is None else arg_info.sharding.spec
    ndim = arg_info.ndim
    if spec is None:
        return (None,) * ndim
    assert len(spec) <= ndim
    return spec + (None,) * (ndim - len(spec))
    
# fwd custom partition
_dot_product_attention_fwd_lower = custom_partitioning(_dot_product_attention_fwd_impl, static_argnums=(5,6,7,8,9,10))
def _dot_product_attention_fwd_infer_sharding_from_operands(scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask, mesh, arg_shapes, result_shape):
    # (*batch, q_seq, num_head, head)
    query_spec = _get_padded_spec(arg_shapes[0])
    # (*batch, kv_seq, num_head, head)
    key_spec = _get_padded_spec(arg_shapes[1])
    # keep out sharding same as query sharding since they have same shape
    out_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
    # activation sharding
    if query_spec[-3] == key_spec[-3]:
        # self attention
        activation_sharding = NamedSharding(mesh, PartitionSpec(*query_spec[:-3], query_spec[-2], query_spec[-3], None))
    else:
        # cross attention
        activation_sharding = NamedSharding(mesh, PartitionSpec(*query_spec[:-3], query_spec[-2], query_spec[-3], key_spec[-3]))
    return (out_sharding, activation_sharding)

def _dot_product_attention_fwd_partition(scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask, mesh, arg_shapes, result_shape):
    # (*batch, q_seq, num_head, head)
    query_spec = _get_padded_spec(arg_shapes[0])
    # (*batch, kv_seq, num_head, head)
    key_spec = _get_padded_spec(arg_shapes[1])
    # keep out sharding same as query sharding since they have same shape
    out_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
    # activation sharding
    if query_spec[-3] == key_spec[-3]:
        # self attention
        activation_sharding = NamedSharding(mesh, PartitionSpec(*query_spec[:-3], query_spec[-2], query_spec[-3], None))
    else:
        # cross attention
        activation_sharding = NamedSharding(mesh, PartitionSpec(*query_spec[:-3], query_spec[-2], query_spec[-3], key_spec[-3]))
    # args sharding
    arg_shardings = tuple([arg_i.sharding for arg_i in arg_shapes])
    out_shardings = (out_sharding, activation_sharding)
    impl = partial(_dot_product_attention_fwd_impl, scale=scale, seed=seed, dropout_rate=dropout_rate,
                    variadic_args=variadic_args, is_flash_attention=is_flash_attention, is_causal_mask=is_causal_mask)
    return mesh, impl, out_shardings, arg_shardings

# bwd custom partition
_dot_product_attention_bwd_lower = custom_partitioning(_dot_product_attention_bwd_impl, static_argnums=(8,9,10,11,12,13))
def _dot_product_attention_bwd_infer_sharding_from_operands(scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask, mesh, arg_shapes, result_shape):
    # (*batch, q_seq, num_head, head)
    query_spec = _get_padded_spec(arg_shapes[0])
    # (*batch, kv_seq, num_head, head)
    key_spec = _get_padded_spec(arg_shapes[1])
    # keep grad query sharding same as query sharding
    grad_query_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
    grad_key_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
    grad_value_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
    out_shardings = (grad_query_sharding, grad_key_sharding, grad_value_sharding)
    return out_shardings

def _dot_product_attention_bwd_partition(scale, seed, dropout_rate, variadic_args, is_flash_attention, is_causal_mask, mesh, arg_shapes, result_shape):
    # (*batch, q_seq, num_head, head)
    query_spec = _get_padded_spec(arg_shapes[0])
    # (*batch, kv_seq, num_head, head)
    key_spec = _get_padded_spec(arg_shapes[1])
    # keep grad query sharding same as query sharding
    grad_query_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
    grad_key_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
    grad_value_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
    out_shardings = (grad_query_sharding, grad_key_sharding, grad_value_sharding)
    # args sharding
    arg_shardings = tuple([arg_i.sharding for arg_i in arg_shapes])
    impl = partial(_dot_product_attention_bwd_impl, scale=scale, seed=seed, dropout_rate=dropout_rate,
                    variadic_args=variadic_args, is_flash_attention=is_flash_attention, is_causal_mask=is_causal_mask)
    return mesh, impl, out_shardings, arg_shardings

# Create dot_product_attention_fwd_p for forward operation.
_dot_product_attention_fwd_p = core.Primitive("dot_product_attention_fwd")
_dot_product_attention_fwd_p.multiple_results = True
_dot_product_attention_fwd_p.def_impl(partial(xla.apply_primitive, _dot_product_attention_fwd_p))
_dot_product_attention_fwd_p.def_abstract_eval(_dot_product_attention_fwd_abstract)

mlir.register_lowering(
    _dot_product_attention_fwd_p,
    _dot_product_attention_fwd_cuda_lowering,
    platform="gpu",
)

_dot_product_attention_fwd_p_wrapper = core.Primitive("dot_product_attention_fwd_wrapper")
_dot_product_attention_fwd_p_wrapper.multiple_results = True
_dot_product_attention_fwd_p_wrapper.def_impl(_dot_product_attention_fwd_impl)
_dot_product_attention_fwd_p_wrapper.def_abstract_eval(_dot_product_attention_fwd_abstract)

# Create dot_product_attention_bwd_p for backward operation.
_dot_product_attention_bwd_p = core.Primitive("dot_product_attention_bwd")
_dot_product_attention_bwd_p.multiple_results = True
_dot_product_attention_bwd_p.def_impl(partial(xla.apply_primitive, _dot_product_attention_bwd_p))
_dot_product_attention_bwd_p.def_abstract_eval(_dot_product_attention_bwd_abstract)

mlir.register_lowering(
    _dot_product_attention_bwd_p,
    _dot_product_attention_bwd_cuda_lowering,
    platform="gpu",
)

_dot_product_attention_bwd_p_wrapper = core.Primitive("dot_product_attention_bwd_wrapper")
_dot_product_attention_bwd_p_wrapper.multiple_results = True
_dot_product_attention_bwd_p_wrapper.def_impl(_dot_product_attention_bwd_impl)
_dot_product_attention_bwd_p_wrapper.def_abstract_eval(_dot_product_attention_bwd_abstract)


batching.primitive_batchers[_dot_product_attention_fwd_p_wrapper] = _dot_product_attention_fwd_batcher
batching.primitive_batchers[_dot_product_attention_bwd_p_wrapper] = _dot_product_attention_bwd_batcher

_dot_product_attention_fwd_lower.def_partition(
    infer_sharding_from_operands=_dot_product_attention_fwd_infer_sharding_from_operands,
    partition=_dot_product_attention_fwd_partition)

mlir.register_lowering(_dot_product_attention_fwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_fwd_lower, multiple_results=True))
                        
_dot_product_attention_bwd_lower.def_partition(
    infer_sharding_from_operands=_dot_product_attention_bwd_infer_sharding_from_operands,
    partition=_dot_product_attention_bwd_partition)

mlir.register_lowering(_dot_product_attention_bwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_bwd_lower, multiple_results=True))

dispatch.prim_requires_devices_during_lowering.add(_dot_product_attention_fwd_p)
dispatch.prim_requires_devices_during_lowering.add(_dot_product_attention_fwd_p_wrapper)
dispatch.prim_requires_devices_during_lowering.add(_dot_product_attention_bwd_p)
dispatch.prim_requires_devices_during_lowering.add(_dot_product_attention_bwd_p_wrapper)

@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10))
def _dot_product_attention(query: Array,
                            key: Array,
                            value: Array,
                            bias: Array,
                            mask: Array,
                            scale: float,
                            seed: int,
                            dropout_rate: float,
                            variadic_args: tuple[bool],
                            is_flash_attention: bool,
                            is_causal_mask: bool):
    output = _dot_product_attention_fwd( 
        query, key, value, bias, mask,
        scale=scale, seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
        is_flash_attention=is_flash_attention, is_causal_mask=is_causal_mask)
    return output

# _dot_product_attention_fwd must have the same func signature as _dot_product_attention
_dot_product_attention.defvjp(_dot_product_attention_fwd_rule, _dot_product_attention_bwd_rule)
    
# User interface
def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          scale: float = 1.0,
                          bias: Optional[Array] = None,
                          mask: Optional[Array] = None,
                          is_causal_mask: bool = False,
                          seed: int = 42,
                          dropout_rate: float = 0.):
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
        scale: scale for the query.
        dropout_rate: dropout rate
    Returns:
        Output of shape `[batch, length, num_heads, v_depth_per_head]`.
    """
    # check if query, key and value layout meets cuDNN layout requirement
    check_qkv_layout(query, key, value)
    # check if flash attention is supported for this attention pattern
    is_flash_attention = check_is_flash_attention(query, key)
    # check if cuDNN is installed and if cuDNN version is sufficient 
    check_cuDNN_version(is_flash_attention)

    variadic_args = (bias is not None, mask is not None)
    if bias is None:
        bias = jnp.zeros(0, dtype=query.dtype)
    if mask is None:
        mask = jnp.zeros(0, dtype=query.dtype)
    # TODO: remove this once scale behavior is fixed
    if scale != 1.0:
        query = query * scale
        scale = 1.0
    output = _dot_product_attention(
        query, key, value, bias, mask, 
        scale, seed, dropout_rate, variadic_args, 
        is_flash_attention, is_causal_mask)
    return output
