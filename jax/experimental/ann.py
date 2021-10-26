# Copyright 2021 Google LLC
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

"""ANN (Approximate Nearest Neighbor) is an **experimental** module for fast top-k with a configurable recall rate on TPU.

TPUs are highly efficient on matrix multiplication, which scales up by TPU
generation. Nevertheless, TPUs are surprisingly inefficient in the other
generalized instructions used in standard algorithm designs.

We can illustrate the inefficiency of writing general algorithms on TPU by
comparing general instructions and matrix multiplication instruction throughput.
We could only use no more than ten vector instructions per matrix
multiplication; otherwise, we would see a significant performance regression.
This constraint prevents us from almost all standard exact top-k algorithms,
including parallel binary-heap, median-of-medians, and bitonic sort.

Our approach to this problem is to reshape the inputs into equally sized
windows,
and return the top-1s in each window containing the true top-k. Suppose we have
the actual top-k elements randomly spreaded in the windows, and we select the
top-1 from each window. The collision rate of the top-k affects the accuracy,
which we can model by the number of people with shared birthdays in the Birthday
problem.

The recall of this approximation depends on the output size :math:`M` and
desired :math`K` elements in top-k. The recall is approximately
:math:`\\exp\\left(\\frac{1-K}{M}\\right)`.  A quick estimate is the output
would roughly be :math:`M=10\\cdot K` for 90% target recall, and
:math:`M=100\\cdot K` for 99% target recall. The smaller the output, the smaller
memory bandwidth it consumes.

Usage::

  from jax.experimental import ann
  # Maximum inner product search
  scores, docids = ann.approx_max_k(lax.dot(qy, db), k=10, recall_target=0.95)

The current JAX implementation sorts and slice the approximate results M into
the final top-k on TPU. We'll also provide a on-host final top-k aggregation
for JAX in the future.

Todos::

  * On host top-k aggregation
  * Accurate but slow differentiation
  * Inaccurate but fast differentiation

"""

from functools import partial
from typing import (Any, Tuple)

import numpy as np
from jax import lax, core
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src import ad_util, dtypes

from jax.interpreters import ad, xla, batching

Array = Any


def approx_max_k(operand: Array,
                 k: int,
                 reduction_dimension: int = -1,
                 recall_target: float = 0.95) -> Tuple[Array, Array]:
  """Returns max ``k`` values and their indices of the ``operand``.

  Args:
    operand : Array to search for max-k.
    k : Specifies the number of max-k.
    reduction_dimension: Integer dimension along which to search. Default: -1.
    recall_target: Recall target for the approximation.

  Returns:
    Tuple[Array, Array] : Max k values and their indices of the inputs.
  """
  return approx_top_k_p.bind(
      operand,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=True)


# Build comparator like lax.sort
def approx_min_k(operand: Array,
                 k: int,
                 reduction_dimension: int = -1,
                 recall_target: float = 0.95) -> Tuple[Array, Array]:
  """Returns min ``k`` values and their indices of the ``operand``.

  Args:
    operand : Array to search for min-k.
    k : Specifies the number of min-k.
    reduction_dimension: Integer dimension along which to search. Default: -1.
    recall_target: Recall target for the approximation.

  Returns:
    Tuple[Array, Array] : Least k values and their indices of the inputs.
  """
  return approx_top_k_p.bind(
      operand,
      k=k,
      reduction_dimension=reduction_dimension,
      recall_target=recall_target,
      is_max_k=False)


def _approx_top_k_abstract_eval(operand, *, k, reduction_dimension,
                                recall_target, is_max_k):
  if k <= 0:
    raise ValueError('k must be positive, got {}'.format(k))
  if len(operand.shape) == 0:
    raise TypeError('approx_top_k operand must have >= 1 dimension, got {}'.format(
        operand.shape))
  dims = list(operand.shape)
  if dims[reduction_dimension] < k:
    raise ValueError(
        'k must be smaller than the size of reduction_dim {}, got {}'.format(
            dims[reduction_dimension], k))
  dims[reduction_dimension] = k
  return (operand.update(
      shape=dims, dtype=operand.dtype, weak_type=operand.weak_type),
          operand.update(shape=dims, dtype=np.dtype(np.int32)))


def _comparator_builder(operand, op_type, is_max_k):
  c = xc.XlaBuilder(
      'top_k_{}_comparator'.format('gt' if is_max_k else 'lt'))
  p0 = xb.parameter(c, 0, xc.Shape.scalar_shape(op_type))
  p1 = xb.parameter(c, 1, xc.Shape.scalar_shape(op_type))
  xb.parameter(c, 2, xc.Shape.scalar_shape(np.dtype(np.int32)))
  xb.parameter(c, 3, xc.Shape.scalar_shape(np.dtype(np.int32)))
  if is_max_k:
    cmp_result = xc.ops.Gt(p0, p1)
  else:
    cmp_result = xc.ops.Lt(p0, p1)
  return c.build(cmp_result)


def _approx_top_k_tpu_translation(c, operand, k, reduction_dimension,
                                  recall_target, is_max_k):
  op_shape = c.get_shape(operand)
  if not op_shape.is_array():
    raise ValueError('operand must be an array, but was {}'.format(op_shape))
  op_dims = op_shape.dimensions()
  op_type = op_shape.element_type()
  if reduction_dimension < 0:
    reduction_dimension = len(op_dims) + reduction_dimension
  comparator = _comparator_builder(operand, op_type, is_max_k)
  if is_max_k:
    if dtypes.issubdtype(op_type, np.floating):
      init_literal = np.array(np.NINF, dtype=op_type)
    else:
      init_literal = np.iinfo(op_type).min()
  else:
    if dtypes.issubdtype(op_type, np.floating):
      init_literal = np.array(np.Inf, dtype=op_type)
    else:
      init_literal = np.iinfo(op_type).max()
  iota = xc.ops.Iota(c, xc.Shape.array_shape(np.dtype(np.int32), op_dims),
                     reduction_dimension)
  init_val = xc.ops.Constant(c, init_literal)
  init_arg = xc.ops.Constant(c, np.int32(-1))
  return xc.ops.ApproxTopK(c, [operand, iota], [init_val, init_arg], k,
                           reduction_dimension, comparator, recall_target, True)


def _approx_top_k_fallback_translation(c, operand, k, reduction_dimension,
                                       recall_target, is_max_k):
  op_shape = c.get_shape(operand)
  if not op_shape.is_array():
    raise ValueError('operand must be an array, but was {}'.format(op_shape))
  op_dims = op_shape.dimensions()
  op_type = op_shape.element_type()
  if reduction_dimension < 0:
    reduction_dimension = len(op_dims) + reduction_dimension
  comparator = _comparator_builder(operand, op_type, is_max_k)
  iota = xc.ops.Iota(c, xc.Shape.array_shape(np.dtype(np.int32), op_dims),
                     reduction_dimension)
  val_arg = xc.ops.Sort(c, [operand, iota], comparator, reduction_dimension)
  vals = xc.ops.GetTupleElement(val_arg, 0)
  args = xc.ops.GetTupleElement(val_arg, 1)
  sliced_vals = xc.ops.SliceInDim(vals, 0, k, 1, reduction_dimension)
  sliced_args = xc.ops.SliceInDim(args, 0, k, 1, reduction_dimension)
  return xc.ops.Tuple(c, [sliced_vals, sliced_args])


def _approx_top_k_batch_rule(batched_args, batch_dims, *, k,
                             reduction_dimension, recall_target, is_max_k):
  prototype_arg, new_bdim = next(
      (a, b) for a, b in zip(batched_args, batch_dims) if b is not None)
  new_args = []
  for arg, bdim in zip(batched_args, batch_dims):
    if bdim is None:
      dims = np.delete(np.arange(prototype_arg.ndim), new_bdim)
      new_args.append(lax.broadcast_in_dim(arg, prototype_arg.shape, dims))
    else:
      new_args.append(batching.moveaxis(arg, bdim, new_bdim))
  new_reduction_dim = reduction_dimension + (new_bdim <= reduction_dimension)
  bdims = (new_bdim,) * len(new_args)
  return (approx_top_k_p.bind(
      *new_args,
      k=k,
      reduction_dimension=new_reduction_dim,
      recall_target=recall_target,
      is_max_k=False), bdims)


# Slow jvp implementation using gather.
#
# TODO(fchern): Some optimization ideas
# 1. ApproxTopK is internally a variadic reduce, so we can simply call
#    ApproxTopK(operand, tangent, iota) for jvp.
# 2. vjp cannot benefit from the algorithm above. We must run scatter to
#    distribute the output cotangent to input cotangent. A reasonable way to do
#    this is to run it on CPU.
def _approx_top_k_jvp(primals, tangents, *, k, reduction_dimension,
                      recall_target, is_max_k):
  operand, = primals
  tangent, = tangents
  if is_max_k:
    val_out, arg_out = approx_max_k(operand, k, reduction_dimension,
                                    recall_target)
  else:
    val_out, arg_out = approx_min_k(operand, k, reduction_dimension,
                                    recall_target)
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
xla.backend_specific_translations['tpu'][
    approx_top_k_p] = _approx_top_k_tpu_translation
xla.backend_specific_translations['cpu'][
    approx_top_k_p] = _approx_top_k_fallback_translation
xla.backend_specific_translations['gpu'][
    approx_top_k_p] = _approx_top_k_fallback_translation
batching.primitive_batchers[approx_top_k_p] = _approx_top_k_batch_rule
ad.primitive_jvps[approx_top_k_p] = _approx_top_k_jvp
