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

from dataclasses import dataclass
import json
import operator
from functools import partial, reduce
from typing import List

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_vjp, lax
from jax._src import core, dispatch, dtypes
from jax._src.custom_partitioning import custom_partitioning
from jax._src.interpreters import batching
from jax._src.lax.lax import ranges_like, remaining
from jax._src.typing import DTypeLike
from jax._src.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


Array = jnp.ndarray
block_scaled_dot_name = "__op$block_scaled_dot"

@dataclass
class BlockScaleConfig:
  mode: str
  block_size: int
  data_type: DTypeLike
  scale_type: DTypeLike
  global_scale: Array | None
  infer_only: bool

def default_layouts(*shapes):
  return [range(len(shape) - 1, -1, -1) for shape in shapes]

def element_type_to_backend_config_type(dtype):
  _element_type_to_backend_config_type_mapping = {
      ir.BF16Type.get(): "BF16",
      ir.F16Type.get(): "F16",
      ir.F32Type.get(): "F32",
  }
  return _element_type_to_backend_config_type_mapping[dtype]


def _scaled_matmul_impl(a, b, a_scale, b_scale, preferred_element_type):
  return _scaled_matmul_p.bind(
      a, b, a_scale, b_scale, preferred_element_type=preferred_element_type
  )


def _scaled_matmul_cuda_lowering(
    ctx, a, b, a_scales, b_scales, preferred_element_type
  ):
  lhs_type = ir.RankedTensorType(a.type)
  lhs_shape = lhs_type.shape
  rhs_type = ir.RankedTensorType(b.type)
  rhs_shape = rhs_type.shape

  batch, non_contracting_lhs, contracting = lhs_shape
  _, non_contracting_rhs, _ = rhs_shape
  result_shape = (batch, non_contracting_lhs, non_contracting_rhs)

  out_type = mlir.dtype_to_ir_type(preferred_element_type)
  result_types = [ir.RankedTensorType.get(result_shape, out_type)]

  operands = [a, b, a_scales, b_scales]
  backend_config = {
      "scaled_dot_backend_config": {
          "lhs_batch_dimensions": [0],
          "rhs_batch_dimensions": [0],
          "dequantize_type": element_type_to_backend_config_type(out_type),
      }
  }

  backend_config = json.dumps(backend_config)
  out = mlir.custom_call(
      block_scaled_dot_name,
      result_types=result_types,
      operands=operands,
      backend_config=backend_config,
      operand_layouts=default_layouts(
          *[ir.RankedTensorType(operand.type).shape for operand in operands]
      ),
      result_layouts=default_layouts(result_shape),
  )
  return [out.result]


def _scaled_matmul_abstract(a, b, a_scale, b_scale, *, preferred_element_type):
  a_dtype = dtypes.canonicalize_dtype(a.dtype)
  batch, non_contracting_lhs, contracting_lhs = a.shape
  _, non_contracting_rhs, _ = b.shape
  output_shape = (batch, non_contracting_lhs, non_contracting_rhs)
  return (core.ShapedArray(output_shape, preferred_element_type),)


_scaled_matmul_p = core.Primitive("scaled_matmul")
_scaled_matmul_p.multiple_results = True
dispatch.simple_impl(_scaled_matmul_p)
_scaled_matmul_p.def_abstract_eval(_scaled_matmul_abstract)


mlir.register_lowering(
    _scaled_matmul_p,
    _scaled_matmul_cuda_lowering,
    platform="cuda",
)

_scaled_matmul_p_wrapper = core.Primitive("scaled_matmul_wrapper")
_scaled_matmul_p_wrapper.multiple_results = True
_scaled_matmul_p_wrapper.def_impl(_scaled_matmul_impl)
_scaled_matmul_p_wrapper.def_abstract_eval(_scaled_matmul_abstract)

# Given the inputs already sharded as
#   ([B], M, K1), ([B], N, K2)
# We define the following rule to apply necessary AllGather based on
# "Input specs", and to define the "Output spec".
# 1. If K1 == K2 != None and N == None:
#   - Input spec : ([B], M, K1), ([B], None, K2)
#   - Output spec: ([B], M, None) -> AllReduce -> ([B], M, None)
# 2. If K1 == K2 != None and M == N != None:
#   - Input spec : ([B], M, K1), ([B], None, K2)
#   - Output spec: ([B], M, None) -> ReduceScatter -> ([B], M, N)
# 3. If N == M:
#   - Input specs : ([B], M, None), ([B], None, None)
#   - Output specs: ([B], M, None)
# 4. If N != M:
#   - Input spec : ([B], M, None), ([B], N, None)
#   - Output spec: ([B], M, N)
def _check_shardings(shardings):
  if len(shardings) != 4:
    msg = f"shardings should container 4 inputs, but got {len(shardings)}"
    raise TypeError(msg)
  lhs, rhs, _, _ = shardings
  if len(lhs.spec) != 3 or len(rhs.spec) != 3:
    msg = (f'shardings specs rank should be 3, but got lhs: {len(lhs.spec)} '
            'and rhs: {len(rhs.spec)}')
    raise TypeError(msg)
  if lhs.spec[0] != rhs.spec[0]:
    msg = ('shardings spec for batch dim should be same, but got lhs: '
            '{lhs.spec[0]} and rhs: {rhs.spec[0]}')
    raise TypeError(msg)


def _enable_reduce_scatter(lhs, rhs):
  batch_spec, m_spec, lhs_k_spec = lhs.spec
  _, n_spec, rhs_k_spec = rhs.spec
  return (
      lhs_k_spec != None
      and lhs_k_spec == rhs_k_spec
      and m_spec != None
      and m_spec == n_spec
  )


def _enable_all_reduce(lhs, rhs):
  batch_spec, m_spec, lhs_k_spec = lhs.spec
  _, n_spec, rhs_k_spec = rhs.spec
  return lhs_k_spec != None and lhs_k_spec == rhs_k_spec and n_spec == None


def _get_output_sharding(mesh, shardings):
  lhs, rhs = shardings[0], shardings[1]
  batch_spec, m_spec, _ = lhs.spec
  _, n_spec, _ = rhs.spec

  if _enable_reduce_scatter(lhs, rhs):
    return [NamedSharding(lhs.mesh, P(*lhs.spec))]

  output_specs = (batch_spec, m_spec)
  output_specs += (n_spec,) if m_spec != n_spec else (None,)
  return [NamedSharding(lhs.mesh, P(*output_specs))]


def _scaled_matmul_infer_sharding_from_operands(
    preferred_element_type, mesh, shapes, output_shape
  ):
  shardings = jax.tree.map(lambda x: x.sharding, shapes)
  _check_shardings(shardings)

  return _get_output_sharding(mesh, shardings)


def supported_in_sharding(mesh, shardings):
  lhs_sharding, rhs_sharding = shardings[0], shardings[1]
  use_reduce_scatter = _enable_reduce_scatter(lhs_sharding, rhs_sharding)
  use_all_reduce = _enable_all_reduce(lhs_sharding, rhs_sharding)
  assert not (use_all_reduce and use_reduce_scatter)

  lhs_specs, rhs_specs = list(lhs_sharding.spec), list(rhs_sharding.spec)

  def named_sharding(lhs, rhs, lhs_specs, rhs_specs):
    lhs_sharding = NamedSharding(lhs.mesh, P(*lhs_specs))
    rhs_sharding = NamedSharding(rhs.mesh, P(*rhs_specs))
    return (lhs_sharding, rhs_sharding, lhs_sharding, rhs_sharding)

  if use_all_reduce:
    return named_sharding(lhs_sharding, rhs_sharding, lhs_specs, rhs_specs)

  if use_reduce_scatter:
    rhs_specs[1] = None
    return named_sharding(lhs_sharding, rhs_sharding, lhs_specs, rhs_specs)

  lhs_specs[2] = None
  rhs_specs[2] = None
  m_spec, n_spec = lhs_specs[1], rhs_specs[1]
  if m_spec == n_spec:
    rhs_specs[1] = None

  return named_sharding(lhs_sharding, rhs_sharding, lhs_specs, rhs_specs)


def _scaled_matmul_partition(
    preferred_element_type, mesh, shapes, output_shape
  ):
  shardings = jax.tree.map(lambda x: x.sharding, shapes)
  _check_shardings(shardings)

  lhs, rhs = shardings[0], shardings[1]
  use_all_reduce = _enable_all_reduce(lhs, rhs)
  use_reduce_scatter = _enable_reduce_scatter(lhs, rhs)
  lhs_k_spec = lhs.spec[2]

  def _scaled_matmul_impl_partition(a, b, a_scale, b_scale):
    z = _scaled_matmul_impl(a, b, a_scale, b_scale, preferred_element_type)
    if use_reduce_scatter:
        z = jax.lax.psum_scatter(
            z, lhs_k_spec, scatter_dimension=2, tiled=True
        )
    if use_all_reduce:
        z = jax.lax.psum(z, lhs_k_spec)
    return z

  out_shardings = _get_output_sharding(mesh, shardings)
  arg_shardings = supported_in_sharding(mesh, shardings)
  return mesh, _scaled_matmul_impl_partition, out_shardings, arg_shardings


_scaled_matmul_lower = custom_partitioning(
    _scaled_matmul_impl, static_argnums=(4,)
)

_scaled_matmul_lower.def_partition(
    infer_sharding_from_operands=_scaled_matmul_infer_sharding_from_operands,
    partition=_scaled_matmul_partition,
)


def _scaled_matmul_batcher(batched_args, batch_dims, *, preferred_element_type):
  assert len(batch_dims) == 4
  assert (
      batch_dims[0] == batch_dims[1]
      and batch_dims[0] == batch_dims[2]
      and batch_dims[0] == batch_dims[3]
  )
  lhs_bdims = batch_dims[0]
  out_bdims = (batch_dims[0],)
  lhs, rhs, lhs_scales, rhs_scales = batched_args
  *batch, lhs_non_contracting, contracting = lhs.shape
  *_, _, scales_contracting = lhs_scales.shape
  *_, rhs_non_contracting, _ = rhs.shape

  new_batch = reduce(operator.mul, batch)
  # reshape to 3D shape
  lhs = jnp.reshape(lhs, (new_batch, lhs_non_contracting, contracting))
  lhs_scales = jnp.reshape(
      lhs_scales, (new_batch, lhs_non_contracting, scales_contracting)
  )
  rhs = jnp.reshape(rhs, (new_batch, rhs_non_contracting, contracting))
  rhs_scales = jnp.reshape(
      rhs_scales, (new_batch, rhs_non_contracting, scales_contracting)
  )
  output = jnp.reshape(
      _scaled_matmul_p_wrapper.bind(
          lhs,
          rhs,
          lhs_scales,
          rhs_scales,
          preferred_element_type=preferred_element_type,
      )[0],
      (*batch, lhs_non_contracting, rhs_non_contracting),
  )
  return (output,), out_bdims


mlir.register_lowering(
    _scaled_matmul_p_wrapper,
    mlir.lower_fun(_scaled_matmul_lower, multiple_results=True),
)

dispatch.prim_requires_devices_during_lowering.add(_scaled_matmul_p)
dispatch.prim_requires_devices_during_lowering.add(_scaled_matmul_p_wrapper)

batching.primitive_batchers[_scaled_matmul_p_wrapper] = _scaled_matmul_batcher
batching.primitive_batchers[_scaled_matmul_p] = _scaled_matmul_batcher


@partial(jax.jit, static_argnames=("preferred_element_type",))
def _scaled_matmul(
    lhs: Array,
    rhs: Array,
    lhs_scales: Array,
    rhs_scales: Array,
    preferred_element_type: DTypeLike = jnp.float32,
  ) -> Array:
  output = _scaled_matmul_p_wrapper.bind(
      lhs, rhs, lhs_scales, rhs_scales,
      preferred_element_type=preferred_element_type
  )
  return output[0]

def scaled_matmul_wrapper(
    lhs: Array,
    rhs: Array,
    lhs_scales: Array,
    rhs_scales: Array,
    preferred_element_type: DTypeLike = jnp.float32,
) -> Array:
    """
    Performs scaled matrix multiplication between two 3D arrays, with scaling
    factors applied to the matrices.

    Args:
        lhs (Array): A 3D array of shape (B, M, K).
        rhs (Array): A 3D array of shape (B, N, K).
        lhs_scales (Array): A 3D array of shape (B, M, K_block).
        rhs_scales (Array): A 3D array of shape (B, N, K_block).
        preferred_element_type (DTypeLike, optional): The preferred data type
          for the computation. Defaults to `jnp.float32`.

    Returns:
        Array: A 3D array of shape (B, M, N) representing the scaled matrix
          multiplication result.

    Raises:
        AssertionError: If the number of columns in `lhs` (`lhs_K`) does not
          match the number of columns in `rhs` (`rhs_K`).

    Notes:
        - The function ensures that the `preferred_element_type` is
          danonicalized before passing it to the underlying computation.
        - Scaling is applied to the matrices based on the `lhs_scales` and
          `rhs_scales` arrays, enabling efficient computations in blocks.

    """
    B, M, lhs_K = lhs.shape
    _, N, rhs_K = rhs.shape
    assert lhs_K == rhs_K
    _, _, K_block = lhs_scales.shape

    preferred_element_type = dtypes.canonicalize_dtype(
        np.dtype(preferred_element_type)
    )

    out = _scaled_matmul(
        lhs,
        rhs,
        lhs_scales,
        rhs_scales,
        preferred_element_type=preferred_element_type,
    )

    return out

def shape_normalization(x, dimension_numbers):
  """
  Normalizes the shape of the input tensor `x` to `(B, M, K)`.

  This function rearranges and reshapes the input tensor `x` such that:
  - `B` represents the batch dimensions.
  - `M` represents the non-contracting dimensions.
  - `K` represents the contracting dimensions.

  The dimensions are reordered and reshaped based on the provided
  `dimension_numbers`.

  Parameters:
      x: The input tensor to normalize.
      dimension_numbers: A tuple containing two elements:
        - `batch_dims` (tuple): The dimensions of `x` to be treated as batch
          dimensions.
        - `contracting_dims` (tuple): The dimensions of `x` to be treated as
          contracting dimensions.

  Returns:
      jax.numpy.ndarray: The reshaped tensor with shape `(B, M, K)`
  """

  orig_order = list(range(x.ndim))
  contracting_dims, batch_dims = dimension_numbers
  contracting_order = [d for d in orig_order if d in contracting_dims]
  batch_order = [d for d in orig_order if d in batch_dims]
  non_contracting_order = [
      d
      for d in orig_order
      if d not in contracting_dims and d not in batch_dims
  ]
  batch_shape = [x.shape[d] for d in batch_order]
  rows_shape = [x.shape[d] for d in non_contracting_order]
  cols_shape = [x.shape[d] for d in contracting_order]
  new_order = batch_order + non_contracting_order + contracting_order
  rows, cols, batches = (
      np.prod(rows_shape),
      np.prod(cols_shape),
      np.prod(batch_shape, dtype=int),
  )
  t = jnp.transpose(x, new_order)
  return jnp.reshape(t, (batches, rows, cols))


def compute_dot_output_shape(
    lhs_shape, rhs_shape, lhs_dimension_numbers, rhs_dimension_numbers
  ):
  """
  Computes the output shape for a `lax.dot_general`-like operation.
  """
  lhs_contract, lhs_batch = lhs_dimension_numbers[0], lhs_dimension_numbers[1]
  rhs_contract, rhs_batch = rhs_dimension_numbers[0], rhs_dimension_numbers[1]

  output_shape = []
  # Add dimensions for batch (assuming the batch dims of LHS and RHS
  # should be same)
  for i, dim in enumerate(lhs_shape):
    if i in lhs_batch:
      output_shape.append(dim)
  # Add dimensions from the LHS that are non contracting
  for i, dim in enumerate(lhs_shape):
    if i not in lhs_contract and i not in lhs_batch:
      output_shape.append(dim)
  # Add dimensions from the RHS that are non contracting
  for i, dim in enumerate(rhs_shape):
    if i not in rhs_contract and i not in rhs_batch:
      output_shape.append(dim)
  return tuple(output_shape)


def cast_to_e8m0_with_rounding_up(x):
  temp = x.astype(jnp.float32).view(jnp.uint32)
  exp = temp >> 23
  mant = temp & 0x7FFFFF
  is_ru = jnp.logical_and(
      jnp.logical_and((mant > 0), (exp != 0xFE)),
      ~jnp.logical_and((exp == 0), (mant <= 0x400000))
  )
  exp = jnp.where(is_ru, exp + 1, exp)
  new_x = exp.astype(jnp.uint8)
  return new_x


def e8m0_to_dtype(x, dtype):
  temp = x.astype(jnp.uint32)
  exp = temp << 23
  new_x = exp.view(jnp.float32)
  near_zero_value = 2**-15 if dtype == jnp.float16 else 2**-127
  new_x = jnp.where(
      new_x == 0, jnp.array(near_zero_value, jnp.float32), new_x
  )
  return new_x.astype(dtype)

def quantize(x, config):
  x_shape = x.shape
  contract_dim = x_shape[-1]
  block_size = config.block_size
  assert contract_dim >= block_size and contract_dim % block_size == 0
  x_new_shape = x_shape[:-1] + (x_shape[-1] // block_size, block_size)
  x = x.reshape(x_new_shape)  # shape = (B, M, K / block_size, block_size)

  amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
  MAX = jnp.finfo(config.data_type).max.astype(x.dtype)
  scales = amax / MAX  # shape = (B, M, K / block_size, 1)

  if config.mode == "mxfp8":
    assert config.scale_type == jnp.float8_e8m0fnu
    scales_q = cast_to_e8m0_with_rounding_up(scales)
    scaled_x = x / e8m0_to_dtype(scales_q, scales.dtype)
  elif config.mode == "nvfp4":
    assert config.scale_type == jnp.float8_e4m3fn
    assert config.global_scale.dtype == jnp.float32
    SCALE_MAX = jnp.finfo(config.scale_type).max.astype(x.dtype)

    scales_q = jnp.clip(scales / config.global_scale, 0, SCALE_MAX)
    scales_q = jax.lax.optimization_barrier(scales_q.astype(config.scale_type))
    scaled_x = x / scales_q.astype(jnp.float32)
  else:
    raise ValueError(f"Unrecognized mode: {config.mode}.")

  clipped_x = jnp.clip(scaled_x, -MAX, MAX)
  x_q = clipped_x.astype(config.data_type)

  x_q = x_q.reshape(x_shape)  # shape = (B, M, K)
  scales_q = jnp.reshape(scales_q, scales_q.shape[:-1]).view(
      config.scale_type
  )
  return x_q, scales_q

def scaled_dot_impl(lhs, rhs, dimension_numbers, preferred_element_type,
                    configs):
  if preferred_element_type is None:
    preferred_element_type = dtypes.result_type(
        lhs, rhs, return_weak_type_flag=False
    )
  else:
    preferred_element_type = dtypes.canonicalize_dtype(
        np.dtype(preferred_element_type)
    )

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_dn = (lhs_contract, lhs_batch)
  rhs_dn = (rhs_contract, rhs_batch)

  lhs_3d = shape_normalization(lhs, lhs_dn)
  rhs_3d = shape_normalization(rhs, rhs_dn)
  lhs_config, rhs_config = configs[0], configs[1]
  lhs_q, lhs_scales = quantize(lhs_3d, lhs_config)
  rhs_q, rhs_scales = quantize(rhs_3d, rhs_config)

  out_dtype = preferred_element_type
  if configs[0].mode == 'nvfp4':
    out_dtype = jnp.float32

  out = scaled_matmul_wrapper(
      lhs_q, rhs_q, lhs_scales, rhs_scales, preferred_element_type=out_dtype
  )

  if configs[0].mode == 'nvfp4':
    out *= (configs[0].global_scale * configs[1].global_scale)
    out = out.astype(preferred_element_type)

  expanded_out_shape = compute_dot_output_shape(
      lhs.shape, rhs.shape, lhs_dn, rhs_dn
  )
  expanded_out = jnp.reshape(out, expanded_out_shape)
  return expanded_out


def scaled_dot_general_transpose_lhs(
    g, x, y, *, dimension_numbers, preferred_element_type, configs,
    swap_ans=False
  ):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim = x.aval.ndim
  x_kept = remaining(range(x_ndim), x_contract, x_batch)
  y_kept = remaining(range(np.ndim(y)), y_contract, y_batch)
  if swap_ans:
    ans_batch, ans_y, _ = ranges_like(x_batch, y_kept, x_kept)
  else:
    ans_batch, _, ans_y = ranges_like(x_batch, x_kept, y_kept)

  dims = ((ans_y, y_kept), (ans_batch, y_batch))
  x_contract_sorted_by_y = list(np.take(x_contract, np.argsort(y_contract)))
  out_axes = np.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)

  y_dn = (y_kept, y_batch)
  g_dn = (ans_y, ans_batch)

  y_3d = shape_normalization(y, y_dn)
  g_3d = shape_normalization(g, g_dn)

  g_config, y_config = configs[0], configs[1]
  if configs[0].mode != 'nvfp4':
    g_q, g_scales = quantize(g_3d, g_config)
    y_q, y_scales = quantize(y_3d, y_config)

    out = scaled_matmul_wrapper(
        g_q, y_q, g_scales, y_scales, preferred_element_type
    )
  else:
    out = jnp.matmul(g_3d, jnp.permute_dims(y_3d, (0, 2, 1)), preferred_element_type=preferred_element_type)

  expanded_out_shape = compute_dot_output_shape(g.shape, y.shape, g_dn, y_dn)
  expanded_out = jnp.reshape(out, expanded_out_shape)
  x_bar = lax.transpose(expanded_out, tuple(out_axes))
  return x_bar


def scaled_dot_general_transpose_rhs(
    g, x, y, *, dimension_numbers, preferred_element_type: DTypeLike,
    configs: List[BlockScaleConfig]
  ):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
  y_bar = scaled_dot_general_transpose_lhs(
      g,
      y,
      x,
      dimension_numbers=swapped_dimension_numbers,
      preferred_element_type=preferred_element_type,
      configs=configs,
      swap_ans=True,
  )
  return y_bar


@partial(custom_vjp, nondiff_argnums=(2, 3, 4))
def scaled_dot_general_fn(lhs, rhs, dimension_numbers, preferred_element_type,
                          configs):
  return scaled_dot_impl(lhs, rhs, dimension_numbers, preferred_element_type,
                         configs)


def scaled_dot_fwd(lhs, rhs, dimension_numbers, preferred_element_type,
                   configs):
  out = scaled_dot_impl(lhs, rhs, dimension_numbers, preferred_element_type,
                        configs)
  res = (lhs, rhs)
  return out, res


def scaled_dot_bwd(dimension_numbers, preferred_element_type, configs, res, g):
  (lhs, rhs) = res

  args = [g, lhs, rhs]
  kw_args = {
      "dimension_numbers": dimension_numbers,
      "preferred_element_type": preferred_element_type,
  }
  lhs_kw_args = {
      **kw_args,
      "configs": [configs[2], configs[1]]
  }
  rhs_kw_args = {
      **kw_args,
      "configs": [configs[2], configs[0]]
  }
  grad_lhs = scaled_dot_general_transpose_lhs(*args, **lhs_kw_args)
  grad_rhs = scaled_dot_general_transpose_rhs(*args, **rhs_kw_args)

  # We apply a Straight-Through Estimator (STE) with zero-out behavior: if
  # inputs are clipped during quantization in fprop, their corresponding gradients
  # are zeroed out; otherwise, they pass through unchanged.
  if configs[2].mode == "nvfp4":
    assert rhs.dtype == lhs.dtype
    MAX = jnp.finfo(configs[0].data_type).max.astype(lhs.dtype)
    SCALE_MAX = jnp.finfo(configs[0].scale_type).max.astype(lhs.dtype)
    grad_lhs = jnp.where(jnp.abs(lhs) <= configs[0].global_scale * MAX * SCALE_MAX, grad_lhs, 0)
    grad_rhs = jnp.where(jnp.abs(rhs) <= configs[1].global_scale * MAX * SCALE_MAX, grad_rhs, 0)

  return (grad_lhs, grad_rhs)


scaled_dot_general_fn.defvjp(scaled_dot_fwd, scaled_dot_bwd)


def ensure_tuple(dimension_numbers):
  _to_tuple = lambda x: x if isinstance(x, tuple) else tuple(x)

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_contract = _to_tuple(lhs_contract)
  rhs_contract = _to_tuple(rhs_contract)
  lhs_batch = _to_tuple(lhs_batch)
  rhs_batch = _to_tuple(rhs_batch)
  return (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)


def _ensure_batch_dim(lhs, rhs, dimension_numbers):
  contracting_dims, (lhs_batch, rhs_batch) = dimension_numbers
  lhs_batched = lhs
  rhs_batched = rhs

  if lhs_batch == ():  # expand the last dim
    lhs_batched = jnp.expand_dims(lhs, axis=lhs.aval.ndim)
    lhs_batch = (lhs.aval.ndim,)
  if rhs_batch == ():
    rhs_batched = jnp.expand_dims(rhs, axis=rhs.aval.ndim)
    rhs_batch = (rhs.aval.ndim,)
  dn_batched = contracting_dims, (lhs_batch, rhs_batch)
  return lhs_batched, rhs_batched, dn_batched


def scaled_dot_general_wrapper(
    lhs, rhs, dimension_numbers,
    preferred_element_type=jnp.float32,
    configs: List[BlockScaleConfig] | None=None,
  ):
  if preferred_element_type not in (jnp.float32, jnp.bfloat16, jnp.float16):
    msg = ('Only support preferred_element_type in (f32, bf16, f16), but got '
            '{preferred_element_type}')
    raise TypeError(msg)
  if configs is None:
    mxfp8_config = BlockScaleConfig(
        mode='mxfp8',
        block_size=32,
        data_type=jnp.float8_e4m3fn,
        scale_type=jnp.float8_e8m0fnu,
        global_scale=None,
        infer_only=False
    )
    configs = [mxfp8_config, mxfp8_config, mxfp8_config]

  dimension_numbers = ensure_tuple(dimension_numbers)
  lhs_batched, rhs_batched, dn_batched = _ensure_batch_dim(
      lhs, rhs, dimension_numbers
  )
  out = scaled_dot_general_fn(
      lhs_batched, rhs_batched, dn_batched, preferred_element_type, configs,
  )

  # Expanding batch dims for operands adds a singleton batch dim at axis 0 in
  # the output, which we need to squeeze.
  if dn_batched != dimension_numbers:
    return jnp.squeeze(out, axis=0)
  return out
