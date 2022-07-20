# Copyright 2021 The JAX Authors.
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

"""ANN (Approximate Nearest Neighbor) computes top-k with a configurable recall rate.

This package only optimizes the TPU backend. For other device types it fallbacks
to sort and slice.

Usage::

  import functools
  import jax

  # MIPS := maximal inner product search
  # Inputs:
  #   qy: f32[qy_size, feature_dim]
  #   db: f32[db_size, feature_dim]
  #
  # Returns:
  #   (f32[qy_size, k], i32[qy_size, k])
  @functools.partial(jax.jit, static_argnames=["k", "recall_target"])
  def mips(qy, db, k=10, recall_target=0.95):
    dists = jax.lax.dot(qy, db.transpose())
    # Computes max_k along the last dimension
    # returns (f32[qy_size, k], i32[qy_size, k])
    return jax.lax.approx_max_k(dists, k=k, recall_target=recall_target)

  # Multi-core example
  # Inputs:
  #   qy: f32[num_devices, qy_size, feature_dim]
  #   db: f32[num_devices, per_device_db_size, feature_dim]
  #   db_offset: i32[num_devices]
  #   db_size = num_devices * per_device_db_size
  #
  # Returns:
  #   (f32[qy_size, num_devices, k], i32[qy_size, num_devices, k])
  @functools.partial(
      jax.pmap,
      # static args: db_size, k, recall_target
      static_broadcasted_argnums=[3, 4, 5],
      out_axes=(1, 1))
  def pmap_mips(qy, db, db_offset, db_size, k, recall_target):
    dists = jax.lax.dot(qy, db.transpose())
    dists, neighbors = jax.lax.approx_max_k(
        dists, k=k, recall_target=recall_target,
        reduction_input_size_override=db_size)
    return (dists, neighbors + db_offset)

  # i32[qy_size, num_devices, k]
  pmap_neighbors = pmap_mips(qy, db, db_offset, db_size, 10, 0.95)[1]
  # i32[qy_size, num_devices * k]
  neighbors = jax.lax.collapse(pmap_neighbors, start_dimension=1, stop_dimension=3)

Todos::

  * On host top-k aggregation
  * Inaccurate but fast differentiation

"""

from functools import partial
from typing import (Any, Tuple)

import json
import numpy as np
from jax import core
from jax import linear_util
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax._src import ad_util, dtypes, util

from jax.interpreters import ad
from jax.interpreters import xla
from jax.interpreters import batching
from jax.interpreters import partial_eval
from jax.interpreters import mlir

Array = Any


def approx_max_k(operand: Array,
                 k: int,
                 reduction_dimension: int = -1,
                 recall_target: float = 0.95,
                 reduction_input_size_override: int = -1,
                 aggregate_to_topk: bool = True) -> Tuple[Array, Array]:
  """Returns max ``k`` values and their indices of the ``operand`` in an approximate manner.

  See https://arxiv.org/abs/2206.14286 for the algorithm details.

  Args:
    operand : Array to search for max-k. Must be a floating number type.
    k : Specifies the number of max-k.
    reduction_dimension : Integer dimension along which to search. Default: -1.
    recall_target : Recall target for the approximation.
    reduction_input_size_override : When set to a positive value, it overrides
      the size determined by ``operand[reduction_dim]`` for evaluating the
      recall. This option is useful when the given ``operand`` is only a subset
      of the overall computation in SPMD or distributed pipelines, where the
      true input size cannot be deferred by the operand shape.
    aggregate_to_topk : When true, aggregates approximate results to the top-k
      in sorted order. When false, returns the approximate results unsorted. In
      this case, the number of the approximate results is implementation defined
      and is greater or equal to the specified ``k``.

  Returns:
    Tuple of two arrays. The arrays are the max ``k`` values and the
    corresponding indices along the ``reduction_dimension`` of the input
    ``operand``. The arrays' dimensions are the same as the input ``operand``
    except for the ``reduction_dimension``: when ``aggregate_to_topk`` is true,
    the reduction dimension is ``k``; otherwise, it is greater equals to ``k``
    where the size is implementation-defined.

  We encourage users to wrap ``approx_max_k`` with jit. See the following
  example for maximal inner production search (MIPS):

  >>> import functools
  >>> import jax
  >>> import numpy as np
  >>> @functools.partial(jax.jit, static_argnames=["k", "recall_target"])
  ... def mips(qy, db, k=10, recall_target=0.95):
  ...   dists = jax.lax.dot(qy, db.transpose())
  ...   # returns (f32[qy_size, k], i32[qy_size, k])
  ...   return jax.lax.approx_max_k(dists, k=k, recall_target=recall_target)
  >>>
  >>> qy = jax.numpy.array(np.random.rand(50, 64))
  >>> db = jax.numpy.array(np.random.rand(1024, 64))
  >>> dot_products, neighbors = mips(qy, db, k=10)
  """
  comparator_jaxpr = _build_comparator_jaxpr(operand, is_max_k=True)
  return approx_top_k_p.bind(
      operand,
      comparator_jaxpr=comparator_jaxpr,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=True,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk)


def approx_min_k(operand: Array,
                 k: int,
                 reduction_dimension: int = -1,
                 recall_target: float = 0.95,
                 reduction_input_size_override: int = -1,
                 aggregate_to_topk: bool = True) -> Tuple[Array, Array]:
  """Returns min ``k`` values and their indices of the ``operand`` in an approximate manner.

  See https://arxiv.org/abs/2206.14286 for the algorithm details.

  Args:
    operand : Array to search for min-k. Must be a floating number type.
    k : Specifies the number of min-k.
    reduction_dimension: Integer dimension along which to search. Default: -1.
    recall_target: Recall target for the approximation.
    reduction_input_size_override : When set to a positive value, it overrides
      the size determined by ``operand[reduction_dim]`` for evaluating the
      recall. This option is useful when the given operand is only a subset of
      the overall computation in SPMD or distributed pipelines, where the true
      input size cannot be deferred by the ``operand`` shape.
    aggregate_to_topk : When true, aggregates approximate results to the top-k
      in sorted order. When false, returns the approximate results unsorted. In
      this case, the number of the approximate results is implementation defined
      and is greater or equal to the specified ``k``.

  Returns:
    Tuple of two arrays. The arrays are the least ``k`` values and the
    corresponding indices along the ``reduction_dimension`` of the input
    ``operand``.  The arrays' dimensions are the same as the input ``operand``
    except for the ``reduction_dimension``: when ``aggregate_to_topk`` is true,
    the reduction dimension is ``k``; otherwise, it is greater equals to ``k``
    where the size is implementation-defined.

  We encourage users to wrap ``approx_min_k`` with jit. See the following example
  for nearest neighbor search over the squared l2 distance:

  >>> import functools
  >>> import jax
  >>> import numpy as np
  >>> @functools.partial(jax.jit, static_argnames=["k", "recall_target"])
  ... def l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
  ...   dists = half_db_norms - jax.lax.dot(qy, db.transpose())
  ...   return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)
  >>>
  >>> qy = jax.numpy.array(np.random.rand(50, 64))
  >>> db = jax.numpy.array(np.random.rand(1024, 64))
  >>> half_db_norms = jax.numpy.linalg.norm(db, axis=1) / 2
  >>> dists, neighbors = l2_ann(qy, db, half_db_norms, k=10)

  In the example above, we compute ``db_norms/2 - dot(qy, db^T)`` instead of
  ``qy^2 - 2 dot(qy, db^T) + db^2`` for performance reason. The former uses less
  arithmetics and produces the same set of neighbors.
  """
  comparator_jaxpr = _build_comparator_jaxpr(operand, is_max_k=False)
  return approx_top_k_p.bind(
      operand,
      comparator_jaxpr=comparator_jaxpr,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=False,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk)


def _approx_top_k_abstract_eval(operand, *, comparator_jaxpr, k,
                                reduction_dimension, recall_target, is_max_k,
                                reduction_input_size_override,
                                aggregate_to_topk):
  if k <= 0:
    raise ValueError(f'k must be positive, got {k}')
  if len(operand.shape) == 0:
    raise TypeError('approx_top_k operand must have >= 1 dimension, got {}'.format(
        operand.shape))
  dims = list(operand.shape)
  if dims[reduction_dimension] < k:
    raise ValueError(
        'k must be smaller than the size of reduction_dim {}, got {}'.format(
            dims[reduction_dimension], k))
  if not dtypes.issubdtype(operand.dtype, np.floating):
    raise ValueError('operand must be a floating type')
  reduction_input_size = dims[reduction_dimension]
  dims[reduction_dimension] = xc.ops.ApproxTopKReductionOutputSize(
      reduction_input_size, len(dims), k, recall_target, aggregate_to_topk,
      reduction_input_size_override)[0]
  return (operand.update(
      shape=dims, dtype=operand.dtype, weak_type=operand.weak_type),
          operand.update(shape=dims, dtype=np.dtype(np.int32)))


def _build_comparator_jaxpr(operand, is_max_k):
  op_aval = core.ShapedArray((), dtypes.dtype(operand))
  idx_aval = core.ShapedArray((), dtype=np.int32)

  @linear_util.wrap_init
  def fast_comparator(val_x, val_y, idx_x, idx_y):
    if is_max_k:
      return (lax.gt(val_x, val_y),)
    return (lax.lt(val_x, val_y),)

  return partial_eval.trace_to_jaxpr_dynamic(
      fast_comparator, (op_aval, op_aval, idx_aval, idx_aval))[0]


def _build_aggregate_to_topk(k, reduction_dimension, is_max_k):

  def _aggregate_to_topk_jaximpl(val, idx, init_val, init_idx):
    if k == 1:
      out_shape = list(val.shape)
      out_shape[reduction_dimension] = 1

      def reducer(x, y):
        val_x, idx_x = x
        val_y, idx_y = y
        if is_max_k:
          select_x = val_x > val_y
        else:
          select_x = val_x < val_y
        out_val = lax.select(select_x, val_x, val_y)
        out_idx = lax.select(select_x, idx_x, idx_y)
        return out_val, out_idx

      val, idx = lax.reduce([val, idx],
                            init_values=[init_val, init_idx],
                            computation=reducer,
                            dimensions=[reduction_dimension])
      return lax.reshape(val, out_shape), lax.reshape(idx, out_shape)
    else:
      if is_max_k:
        val, idx = lax.sort_key_val(-val, idx, reduction_dimension)
        val = slicing.slice_in_dim(-val, 0, k, axis=reduction_dimension)
        idx = slicing.slice_in_dim(idx, 0, k, axis=reduction_dimension)
        return val, idx
      else:
        val, idx = lax.sort_key_val(val, idx, reduction_dimension)
        val = slicing.slice_in_dim(val, 0, k, axis=reduction_dimension)
        idx = slicing.slice_in_dim(idx, 0, k, axis=reduction_dimension)
        return val, idx

  return _aggregate_to_topk_jaximpl


def _approx_topk_falllback_lower(ctx, operand, *, comparator_jaxpr, k,
                                 reduction_dimension, recall_target, is_max_k,
                                 reduction_input_size_override,
                                 aggregate_to_topk):
  op_aval = ctx.avals_in[0]
  op_dims = list(op_aval.shape)
  op_type = op_aval.dtype
  if not op_dims:
    raise ValueError(f'operand must be an array, but was {op_dims}')
  if reduction_dimension < 0:
    reduction_dimension = len(op_dims) + reduction_dimension

  iota = mhlo.IotaOp(
      mlir.aval_to_ir_type(core.ShapedArray(op_dims, np.int32)),
      mlir.i64_attr(reduction_dimension)).results
  if is_max_k:
    init_val = mlir.ir_constant(
        np.array(lax._get_max_identity(op_type), dtype=op_type))
  else:
    init_val = mlir.ir_constant(
        np.array(lax._get_min_identity(op_type), dtype=op_type))
  init_idx = mlir.ir_constant(np.array(-1, np.int32))

  n = op_dims[reduction_dimension]
  output_size, _ = xc.ops.ApproxTopKReductionOutputSize(
      n, len(op_dims), k, recall_target, aggregate_to_topk,
      reduction_input_size_override)

  out_dims = list(op_dims)
  out_dims[reduction_dimension] = output_size

  fallback_jaximpl = _build_aggregate_to_topk(output_size, reduction_dimension,
                                              is_max_k)
  fallback_lowering = mlir.lower_fun(fallback_jaximpl, True)
  fallback_ctx = ctx.replace(
      avals_in=[
          op_aval,
          core.ShapedArray(op_dims, np.int32),
          core.ShapedArray((), op_type),
          core.ShapedArray((), np.int32)
      ],
      avals_out=[
          core.ShapedArray(out_dims, op_type),
          core.ShapedArray(out_dims, np.int32)
      ])
  return fallback_lowering(fallback_ctx, operand, iota, init_val, init_idx)


def _approx_top_k_tpu_lower(ctx, operand, *, comparator_jaxpr, k,
                            reduction_dimension, recall_target, is_max_k,
                            reduction_input_size_override, aggregate_to_topk):
  # we should only have one avals_in, which is the operand.
  op_aval = ctx.avals_in[0]
  op_dims = op_aval.shape
  op_type = op_aval.dtype

  if not op_dims:
    raise ValueError(f'operand must be an array, but was {op_dims}')
  if reduction_dimension < 0:
    reduction_dimension = len(op_dims) + reduction_dimension
  op_rank = len(op_dims)
  tpu_tiling = 1024 if op_rank == 1 else 128

  iota = mhlo.IotaOp(
      mlir.aval_to_ir_type(core.ShapedArray(op_dims, np.int32)),
      mlir.i64_attr(reduction_dimension)).results
  if is_max_k:
    init_val = mlir.ir_constant(
        np.array(lax._get_max_identity(op_type), dtype=op_type))
  else:
    init_val = mlir.ir_constant(
        np.array(lax._get_min_identity(op_type), dtype=op_type))
  init_idx = mlir.ir_constant(np.array(-1, np.int32))

  aggregate_to_topk_jaximpl = _build_aggregate_to_topk(k, reduction_dimension,
                                                       is_max_k)
  aggregate_to_topk_lowered = mlir.lower_fun(aggregate_to_topk_jaximpl, True)

  n = op_dims[reduction_dimension]
  init_val_aval = core.ShapedArray((), op_type)
  init_idx_aval = core.ShapedArray((), np.int32)

  if n <= tpu_tiling:
    if aggregate_to_topk:
      out_dims = list(op_dims)
      out_dims[reduction_dimension] = k
      agg_ctx = ctx.replace(
          primitive=None,
          avals_in=[
              op_aval,
              core.ShapedArray(op_dims, np.int32), init_val_aval, init_idx_aval
          ],
          avals_out=[
              core.ShapedArray(out_dims, op_type),
              core.ShapedArray(out_dims, np.int32)
          ])
      return aggregate_to_topk_lowered(agg_ctx, operand, iota, init_val,
                                       init_idx)
    return operand, iota

  approx_output_size, log2_reduction = xc.ops.ApproxTopKReductionOutputSize(
      n, op_rank, k, recall_target, False, reduction_input_size_override)

  # When there is no reduction, fallback to naive aggregation
  if log2_reduction == 0:
    if aggregate_to_topk:
      out_dims = list(op_dims)
      out_dims[reduction_dimension] = k
      agg_ctx = ctx.replace(
          primitive=None,
          avals_in=[
              op_aval,
              core.ShapedArray(op_dims, np.int32), init_val_aval, init_idx_aval
          ],
          avals_out=[
              core.ShapedArray(out_dims, op_type),
              core.ShapedArray(out_dims, np.int32)
          ])
      return aggregate_to_topk_lowered(agg_ctx, operand, iota, init_val,
                                       init_idx)
    return operand, iota

  partial_reduce_config = {
      'log2_reduction': log2_reduction,
      'reduction_dim': reduction_dimension,
      'to_apply_type': 'comparator',
      'top_k': k,
      'recall_target': recall_target,
  }

  closed_pr_comparator_jaxpr = core.ClosedJaxpr(comparator_jaxpr, [])

  comparator_name = f'pr_comparator_{op_type}_is_max_k_{is_max_k}'
  pr_comparator_func_op = mlir.lower_jaxpr_to_fun(ctx.module_context,
                                                  comparator_name,
                                                  closed_pr_comparator_jaxpr,
                                                  [])
  pr_comparator_symbol = pr_comparator_func_op.name.value

  approx_out_dims = list(op_dims)
  approx_out_dims[reduction_dimension] = approx_output_size
  pr_out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(approx_out_dims,
                              mlir.dtype_to_ir_type(np.dtype(dtype)))
      for dtype in [op_type, np.int32]
  ])
  pr_op = mhlo.CustomCallOp(
      [pr_out_type],
      [operand, iota, init_val, init_idx],
      call_target_name=ir.StringAttr.get('PartialReduce'),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(json.dumps(partial_reduce_config)),
      api_version=mlir.i32_attr(1),
      called_computations=ir.ArrayAttr.get(
          [ir.FlatSymbolRefAttr.get(pr_comparator_symbol)]),
      operand_layouts=None,
      result_layouts=None,
  )
  init_val_ir_type = mlir.aval_to_ir_type(init_val_aval)
  init_idx_ir_type = mlir.aval_to_ir_type(init_idx_aval)

  approx_val = mhlo.GetTupleElementOp(pr_op.result, mlir.i32_attr(0)).result
  approx_idx = mhlo.GetTupleElementOp(pr_op.result, mlir.i32_attr(1)).result

  if aggregate_to_topk:
    out_dims = list(op_dims)
    out_dims[reduction_dimension] = k
    agg_ctx = ctx.replace(
        primitive=None,
        avals_in=[
            core.ShapedArray(approx_out_dims, op_type),
            core.ShapedArray(approx_out_dims, np.int32), init_val_aval,
            init_idx_aval
        ],
        avals_out=[
            core.ShapedArray(out_dims, op_type),
            core.ShapedArray(out_dims, np.int32)
        ])
    return aggregate_to_topk_lowered(agg_ctx, approx_val, approx_idx, init_val,
                                     init_idx)

  return approx_val, approx_idx


def _approx_top_k_batch_rule(batch_operands, batch_axes, *, comparator_jaxpr, k,
                             reduction_dimension, recall_target, is_max_k,
                             reduction_input_size_override, aggregate_to_topk):
  assert len(batch_operands) == 1
  assert len(batch_axes) == 1
  operand, = batch_operands
  batch_axis, = batch_axes
  dim_map = [d for d in range(operand.ndim) if d is not batch_axis]
  reduction_dimension = dim_map[reduction_dimension]
  return approx_top_k_p.bind(
      operand,
      comparator_jaxpr=comparator_jaxpr,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=is_max_k,
      reduction_input_size_override=reduction_input_size_override,
      aggregate_to_topk=aggregate_to_topk), (batch_axis, batch_axis)


# Slow jvp implementation using gather.
#
# TODO(fchern): Some optimization ideas
# 1. ApproxTopK is internally a variadic reduce, so we can simply call
#    ApproxTopK(operand, tangent, iota) for jvp.
# 2. vjp cannot benefit from the algorithm above. We must run scatter to
#    distribute the output cotangent to input cotangent. A reasonable way to do
#    this is to run it on CPU.
def _approx_top_k_jvp(primals, tangents, *, comparator_jaxpr, k,
                      reduction_dimension, recall_target, is_max_k,
                      reduction_input_size_override, aggregate_to_topk):
  operand, = primals
  tangent, = tangents
  if is_max_k:
    val_out, arg_out = approx_max_k(operand, k, reduction_dimension,
                                    recall_target,
                                    reduction_input_size_override,
                                    aggregate_to_topk)
  else:
    val_out, arg_out = approx_min_k(operand, k, reduction_dimension,
                                    recall_target,
                                    reduction_input_size_override,
                                    aggregate_to_topk)
  if type(tangent) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    arg_shape = arg_out.shape
    rank = len(arg_shape)
    if reduction_dimension < 0:
      reduction_dimension += rank
    iotas = [
        lax.broadcasted_iota(arg_out.dtype, arg_shape, i) for i in range(rank)
    ]
    idx = tuple(
        arg_out if i == reduction_dimension else iotas[i] for i in range(rank))
    tangent_out = tangent[idx]
  return (val_out, arg_out), (tangent_out, ad_util.Zero.from_value(arg_out))


approx_top_k_p = core.Primitive('approx_top_k')
approx_top_k_p.multiple_results = True
approx_top_k_p.def_impl(partial(xla.apply_primitive, approx_top_k_p))
approx_top_k_p.def_abstract_eval(_approx_top_k_abstract_eval)
batching.primitive_batchers[approx_top_k_p] = _approx_top_k_batch_rule
ad.primitive_jvps[approx_top_k_p] = _approx_top_k_jvp

mlir.register_lowering(approx_top_k_p, _approx_topk_falllback_lower)
mlir.register_lowering(approx_top_k_p, _approx_top_k_tpu_lower, platform='tpu')
