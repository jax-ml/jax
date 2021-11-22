# Copyright 2018 Google LLC
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
from typing import (Any, Callable, Optional, Sequence, Union, Tuple)
import warnings

import numpy as np

from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import xla

from jax import core
from jax.core import (ShapedArray, ConcreteArray)

from jax._src import ad_util
from jax._src import dtypes
import jax._src.lax.lax as lax
from jax._src.lax.lax import (
  _dilate_shape,
  _get_max_identity,
  _get_min_identity,
  _input_dtype,
)
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client


xb = xla_bridge
xc = xla_client
xops = xla_client.ops

Array = Any


def reduce_window(operand: Array, init_value: Array, computation: Callable,
                  window_dimensions: core.Shape, window_strides: Sequence[int],
                  padding: Union[str, Sequence[Tuple[int, int]]],
                  base_dilation: Optional[Sequence[int]] = None,
                  window_dilation: Optional[Sequence[int]] = None) -> Array:
  """Wraps XLA's `ReduceWindowWithGeneralPadding
  <https://www.tensorflow.org/xla/operation_semantics#reducewindow>`_
  operator.
  """
  if isinstance(padding, str):
    dilated_window_dims = (window_dimensions if window_dilation is None else
                           _dilate_shape(window_dimensions, window_dilation))
    padding = tuple(lax.padtype_to_pads(operand.shape, dilated_window_dims,
                                        window_strides, padding))
  else:
    padding = tuple(padding)
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  monoid_reducer = _get_monoid_window_reducer(computation, init_value)
  if monoid_reducer:
    return monoid_reducer(operand, window_dimensions, window_strides, padding,
                          base_dilation, window_dilation)
  else:
    jaxpr, consts = lax._reduction_jaxpr(computation, lax._abstractify(init_value))
    return reduce_window_p.bind(
        operand, init_value, jaxpr=jaxpr, consts=consts,
        window_dimensions=tuple(window_dimensions),
        window_strides=tuple(window_strides), padding=padding,
        base_dilation=tuple(base_dilation),
        window_dilation=tuple(window_dilation))

def _get_monoid_window_reducer(monoid_op: Callable, x: Array) -> Optional[Callable]:
  aval = core.get_aval(x)
  if (type(aval) is ConcreteArray) and aval.shape == ():
    if monoid_op is lax.add:
      return aval.val == 0 and _reduce_window_sum
    elif monoid_op is lax.max:
      return aval.val == _get_max_identity(aval.dtype) and _reduce_window_max
    elif monoid_op is lax.min:
      return aval.val == _get_min_identity(aval.dtype) and _reduce_window_min
  return None

def _reduce_window_sum(operand: Array, window_dimensions: core.Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[Tuple[int, int]],
                       base_dilation: Optional[Sequence[int]] = None,
                       window_dilation: Optional[Sequence[int]] = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_sum_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_prod(operand: Array, window_dimensions: core.Shape,
                        window_strides: Sequence[int],
                        padding: Sequence[Tuple[int, int]],
                        base_dilation: Optional[Sequence[int]] = None,
                        window_dilation: Optional[Sequence[int]] = None) -> Array:
  init_value = lax._const(operand, 1)
  jaxpr, consts = lax._reduction_jaxpr(lax.mul, lax._abstractify(init_value))
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_p.bind(
      operand, init_value, jaxpr=jaxpr, consts=consts,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_max(operand: Array, window_dimensions: core.Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[Tuple[int, int]],
                       base_dilation: Optional[Sequence[int]] = None,
                       window_dilation: Optional[Sequence[int]] = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_max_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_min(operand: Array, window_dimensions: core.Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[Tuple[int, int]],
                       base_dilation: Optional[Sequence[int]] = None,
                       window_dilation: Optional[Sequence[int]] = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_min_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _select_and_scatter(operand: Array, select: Callable,
                        window_dimensions: core.Shape,
                        window_strides: Sequence[int],
                        padding: Sequence[Tuple[int, int]], source: Array,
                        init_value: Array, scatter: Callable) -> Array:
  select_jaxpr, select_consts = lax._reduction_jaxpr(
    select, lax._abstractify(init_value))
  scatter_jaxpr, scatter_consts = lax._reduction_jaxpr(
    scatter, lax._abstractify(init_value))
  return select_and_scatter_p.bind(
      operand, source, init_value, select_jaxpr=select_jaxpr,
      select_consts=select_consts, scatter_jaxpr=scatter_jaxpr,
      scatter_consts=scatter_consts, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding))

def _select_and_scatter_add(source: Array, operand: Array,
                            select_prim: core.Primitive,
                            window_dimensions: core.Shape,
                            window_strides: Sequence[int],
                            padding: Sequence[Tuple[int, int]]) -> Array:
  return select_and_scatter_add_p.bind(
      source, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding))

def _select_and_gather_add(tangents: Array, operand: Array,
                           select_prim: core.Primitive,
                           window_dimensions: core.Shape,
                           window_strides: Sequence[int],
                           padding: Sequence[Tuple[int, int]],
                           base_dilation: Sequence[int],
                           window_dilation: Sequence[int]) -> Array:
  """Extracts the tangent corresponding to the minimum or maximum element in each
  window of the `operand` array.

  Wraps XLA's `ReduceWindow
  <https://www.tensorflow.org/xla/operation_semantics#reducewindow>`_
  operator, which applies a reduction function to all elements in each window of the
  input multi-dimensional array. In this case, the input multi-dimensional array is
  built by packing each element in the `operand` array with its corresponding
  element in the `tangents` array.

  Args:
    tangents: an array
    operand: an array with the same shape as `tangents`
    select_prim: a reduction function (restricted to `ge_p` and `le_p`)
    window_dimensions: an array of integers for window dimension values
    window_strides: an array of integers for window stride values
    base_dilation: an array of integers for base dilation values
    window_dilation: an array of integers for window dilation values

  Returns:
    An array containing the elements in `tangents` corresponding to the output of the
    reduction of `operand` fin each window.
  """
  return select_and_gather_add_p.bind(
      tangents, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))


def _reduce_window_shape_rule(operand, init_value, *, jaxpr, consts,
                              window_dimensions, window_strides, padding,
                              base_dilation, window_dilation):
  if operand.dtype != init_value.dtype:
    msg = ("reduce_window got inconsistent dtypes for operand and init_value: "
           " got operand dtype {} and init_value dtype {}.")
    raise TypeError(msg.format(operand.dtype, init_value.dtype))
  if init_value.shape != ():
    msg = ("reduce_window expected init_value to be a scalar but init_value "
           "has shape {}.")
    raise TypeError(msg.format(init_value.shape))
  return _common_reduce_window_shape_rule(
    operand, window_dimensions, window_strides, padding, base_dilation,
    window_dilation)

def _reduce_window_translation_rule(ctx, avals_in, avals_out, operand,
                                    init_value, *, jaxpr, consts,
                                    window_dimensions, window_strides, padding,
                                    base_dilation, window_dilation):
  xla_computation = lax._reduction_computation(ctx, jaxpr, consts, init_value)
  return [xops.ReduceWindowWithGeneralPadding(
    operand, init_value, xla_computation, window_dimensions,
    window_strides, base_dilation, window_dilation, padding)]

def _generic_reduce_window_batch_rule(
    batched_args, batch_dims, *, jaxpr, consts, window_dimensions,
    window_strides, padding, base_dilation, window_dilation):
  operand, init = batched_args
  bdim, init_bdim = batch_dims
  if init_bdim is not None:
    raise NotImplementedError("reduce_window batching is not implemented for "
                              "initial values")

  def reduce_window(x, window_dimensions, window_strides, padding, base_dilation,
                    window_dilation):
    return reduce_window_p.bind(
      x, init, jaxpr=jaxpr, consts=consts, window_dimensions=window_dimensions,
      window_strides=window_strides, padding=padding, base_dilation=base_dilation,
      window_dilation=window_dilation)
  return _reduce_window_batch_rule(
    reduce_window, (operand,), (bdim,), window_dimensions=window_dimensions,
    window_strides=window_strides, padding=padding, base_dilation=base_dilation,
    window_dilation=window_dilation)


reduce_window_p = lax.standard_primitive(
    _reduce_window_shape_rule, _input_dtype, 'reduce_window',
    _reduce_window_translation_rule)
batching.primitive_batchers[reduce_window_p] = _generic_reduce_window_batch_rule


def _reduce_window_sum_shape_rule(operand, *, window_dimensions, window_strides,
                                  padding, base_dilation, window_dilation):
  if not dtypes.issubdtype(operand.dtype, np.number):
    msg = "operand to reduce_window_sum must have a number dtype, got {}"
    raise TypeError(msg.format(np.dtype(operand.dtype).name))
  return _common_reduce_window_shape_rule(operand, window_dimensions,
                                          window_strides, padding, base_dilation,
                                          window_dilation)

def _reduce_window_sum_translation_rule(ctx, avals_in, avals_out, operand, *,
                                        window_dimensions, window_strides,
                                        padding, base_dilation,
                                        window_dilation):
  operand_aval, = avals_in
  scalar = ShapedArray((), operand_aval.dtype)
  return [xops.ReduceWindowWithGeneralPadding(
    operand,
    xla.pyval_to_ir_constant(ctx.builder, np.array(0, operand_aval.dtype)),
    xla.primitive_subcomputation(ctx.platform, lax.add_p, scalar, scalar),
      window_dimensions,
    window_strides, base_dilation, window_dilation, padding)]

def _reduce_window_sum_transpose_rule(cotangent, operand, *, window_dimensions,
                                      window_strides, padding, base_dilation,
                                      window_dilation):
  assert ad.is_undefined_primal(operand)
  input_shape = operand.aval.shape
  pads = lax._conv_general_vjp_lhs_padding(
      input_shape, window_dimensions, window_strides, cotangent.shape, padding,
      base_dilation, window_dilation)
  ones = [1] * len(input_shape)
  padding_config = [(lo, hi, stride - 1)
                    for (lo, hi), stride in zip(pads, window_strides)]
  pad_cotangent = lax.pad(cotangent, lax._zero(cotangent), padding_config)
  result = _reduce_window_sum(pad_cotangent, window_dimensions, base_dilation,
                              [(0, 0)] * len(input_shape),
                              base_dilation=ones,
                              window_dilation=window_dilation)
  assert result.shape == input_shape, (result.shape, input_shape)
  return [result]

def _reduce_window_batch_rule(reduce_window, batched_args, bdims, *,
                              window_dimensions, window_strides, padding,
                              base_dilation, window_dilation):
  operand, = batched_args
  bdim, = bdims

  if bdim is not None:
    window_dimensions = \
        window_dimensions[:bdim] + (1,) + window_dimensions[bdim:]
    window_strides = window_strides[:bdim] + (1,) + window_strides[bdim:]
    padding = padding[:bdim] + ((0, 0),) + padding[bdim:]
    base_dilation = base_dilation[:bdim] + (1,) + base_dilation[bdim:]
    window_dilation = window_dilation[:bdim] + (1,) + window_dilation[bdim:]

  operand = reduce_window(operand, window_dimensions, window_strides, padding,
                          base_dilation, window_dilation)
  return operand, bdim

reduce_window_sum_p = lax.standard_primitive(
    _reduce_window_sum_shape_rule, _input_dtype, 'reduce_window_sum',
    _reduce_window_sum_translation_rule)
ad.deflinear2(reduce_window_sum_p, _reduce_window_sum_transpose_rule)
batching.primitive_batchers[reduce_window_sum_p] = partial(
  _reduce_window_batch_rule, _reduce_window_sum)

def _reduce_window_chooser_translation_rule(
    prim, identity, ctx, avals_in, avals_out, operand, *, window_dimensions,
    window_strides, padding, base_dilation, window_dilation):
  operand_aval, = avals_in
  scalar = ShapedArray((), operand_aval.dtype)
  return [xops.ReduceWindowWithGeneralPadding(
    operand,
    xla.pyval_to_ir_constant(ctx.builder, identity(operand_aval.dtype)),
    xla.primitive_subcomputation(ctx.platform, prim, scalar, scalar),
      window_dimensions,
    window_strides, base_dilation, window_dilation, padding)]

def _reduce_window_chooser_jvp_rule(prim, g, operand, *, window_dimensions,
                                    window_strides, padding, base_dilation,
                                    window_dilation):
  assert prim is lax.max_p or prim is lax.min_p
  select_prim = lax.ge_p if prim is lax.max_p else lax.le_p
  return _select_and_gather_add(g, operand, select_prim, window_dimensions,
                                window_strides, padding, base_dilation,
                                window_dilation)


def _common_reduce_window_shape_rule(operand, window_dimensions,
                                     window_strides, padding, base_dilation,
                                     window_dilation):
  lax._check_shapelike("reduce_window", "window_dimensions", window_dimensions,
                       non_zero_shape=True)
  lax._check_shapelike("reduce_window", "window_strides", window_strides,
                       non_zero_shape=True)
  lax._check_shapelike("reduce_window", "base_dilation", base_dilation)
  lax._check_shapelike("reduce_window", "window_dilation", window_dilation)
  if operand.ndim != len(window_dimensions):
    msg = ("reduce_window got the wrong number of window_dimensions for "
           "operand: got operand shape {} with window_dimensions {}.")
    raise TypeError(msg.format(operand.shape, window_dimensions))
  if len(window_strides) != len(window_dimensions):
    msg = ("reduce_window got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))
  if len(base_dilation) != len(window_dimensions):
    msg = ("reduce_window got inconsistent base_dilation and "
           "window_dimensions: got base_dilation {} and window_dimensions {}.")
    raise TypeError(msg.format(base_dilation, window_dimensions))
  if len(window_dilation) != len(window_dimensions):
    msg = ("reduce_window got inconsistent window_dilation and "
           "window_dimensions: got window_dilation {} and window_dimensions "
           "{}.")
    raise TypeError(msg.format(window_dilation, window_dimensions))

  return reduce_window_shape_tuple(operand.shape, window_dimensions,
                                   window_strides, padding, base_dilation,
                                   window_dilation)

def reduce_window_shape_tuple(operand_shape, window_dimensions, window_strides,
                              padding, base_dilation=None,
                              window_dilation=None):
  if base_dilation is not None:
    operand_shape = _dilate_shape(operand_shape, base_dilation)
  if window_dilation is not None:
    window_dimensions = _dilate_shape(window_dimensions, window_dilation)
  pads_lo, pads_hi = zip(*padding)
  operand_padded = core.sum_shapes(operand_shape, pads_lo, pads_hi)
  return core.stride_shape(operand_padded, window_dimensions, window_strides)

_reduce_window_max_translation_rule = partial(
    _reduce_window_chooser_translation_rule, lax.max_p, _get_max_identity)
reduce_window_max_p = lax.standard_primitive(
    _common_reduce_window_shape_rule, _input_dtype, 'reduce_window_max',
    _reduce_window_max_translation_rule)
ad.defjvp(reduce_window_max_p, partial(_reduce_window_chooser_jvp_rule,
                                       lax.max_p))
batching.primitive_batchers[reduce_window_max_p] = partial(
  _reduce_window_batch_rule, _reduce_window_max)

_reduce_window_min_translation_rule = partial(
    _reduce_window_chooser_translation_rule, lax.min_p, _get_min_identity)
reduce_window_min_p = lax.standard_primitive(
    _common_reduce_window_shape_rule, _input_dtype, 'reduce_window_min',
    _reduce_window_min_translation_rule)
ad.defjvp(reduce_window_min_p, partial(_reduce_window_chooser_jvp_rule,
                                       lax.min_p))

_reduce_window_min_batch_rule = partial(_reduce_window_batch_rule,
                                        _reduce_window_min)
batching.primitive_batchers[reduce_window_min_p] = partial(
  _reduce_window_batch_rule, _reduce_window_min)


def _select_and_scatter_shape_rule(
    operand, source, init_value, *, select_jaxpr, select_consts, scatter_jaxpr,
    scatter_consts, window_dimensions, window_strides, padding):
  lax._check_shapelike("select_and_scatter", "window_dimensions",
                       window_dimensions)
  lax._check_shapelike("select_and_scatter", "window_strides", window_strides)
  if len(window_dimensions) != len(window_strides):
    msg = ("select_and_scatter got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))
  return operand.shape

def _select_and_scatter_translation(
    ctx, avals_in, avals_out, operand, source, init_value, *, select_jaxpr,
    select_consts, scatter_jaxpr, scatter_consts, window_dimensions,
    window_strides, padding):
  select = lax._reduction_computation(ctx, select_jaxpr, select_consts, init_value)
  scatter = lax._reduction_computation(ctx, scatter_jaxpr, scatter_consts, init_value)
  return [xops.SelectAndScatterWithGeneralPadding(
    operand, select, window_dimensions, window_strides, padding, source,
    init_value, scatter)]

select_and_scatter_p = lax.standard_primitive(
    _select_and_scatter_shape_rule, _input_dtype, 'select_and_scatter',
    _select_and_scatter_translation)


def _select_and_scatter_add_shape_rule(
    source, operand, *, select_prim, window_dimensions, window_strides,
    padding):
  return operand.shape

def _select_and_scatter_add_translation(
    ctx, avals_in, avals_out, source, operand, *, select_prim,
    window_dimensions, window_strides, padding, expand_padding):
  source_aval, operand_aval = avals_in
  c = ctx.builder
  dtype = operand_aval.dtype
  scalar = ShapedArray((), dtype)
  select = xla.primitive_subcomputation(
      ctx.platform, select_prim, scalar, scalar)
  scatter = xla.primitive_subcomputation(
      ctx.platform, lax.or_p if dtype == np.bool_ else lax.add_p, scalar, scalar)
  zero = xla.pyval_to_ir_constant(c, np.array(0, dtype))
  # TODO(b/161704903): remove this workaround when XLA:CPU bug is fixed.
  expand_padding = (expand_padding and
                    not all(lo == 0 and hi == 0 for (lo, hi) in padding))
  if expand_padding:
    original_padding = padding
    identity = (_get_max_identity if select_prim is lax.ge_p
                else _get_min_identity)
    pads = [(lo, hi, 0) for (lo, hi) in padding]
    operand = xops.Pad(operand, xla.pyval_to_ir_constant(c, identity(dtype)),
                       xc.make_padding_config(pads))
    padding = [(0, 0) for _ in padding]
  output = xops.SelectAndScatterWithGeneralPadding(
    operand, select, window_dimensions, window_strides, padding, source, zero,
    scatter)
  if expand_padding:
    start_indices = [lo for (lo, hi) in original_padding]
    stop_indices = [lo + d for ((lo, hi), d) in zip(original_padding,
                                                    operand_aval.shape)]
    output = xops.Slice(output, start_indices, stop_indices,
                        [1] * len(start_indices))
  return [output]

def _select_and_scatter_add_jvp(
    primals, tangents, *, select_prim, window_dimensions, window_strides,
    padding):
  source, operand = primals
  g_source, g_operand = tangents
  val_out = _select_and_scatter_add(
      source, operand, select_prim, window_dimensions, window_strides,
      padding)
  del g_operand
  if type(g_source) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    tangent_out = _select_and_scatter_add(
        g_source, operand, select_prim, window_dimensions,
        window_strides, padding)
  return val_out, tangent_out

def _select_and_scatter_add_transpose(
    t, source, operand, *, select_prim, window_dimensions, window_strides,
    padding):
  assert ad.is_undefined_primal(source) and not ad.is_undefined_primal(operand)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(source.aval), None]
  ones = (1,) * len(window_dimensions)
  source_t = _select_and_gather_add(t, operand, select_prim, window_dimensions,
                                    window_strides, padding, ones, ones)
  return [source_t, None]

def _select_and_scatter_add_batch_rule(
    batched_args, batch_dims, *, select_prim, window_dimensions, window_strides,
    padding):
  source, operand = batched_args
  s_bdim, o_bdim = batch_dims
  size = next(a.shape[bdim] for a, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  source = batching.bdim_at_front(source, s_bdim, size)
  operand = batching.bdim_at_front(operand, o_bdim, size)

  window_dimensions = (1,) + window_dimensions
  window_strides = (1,) + window_strides
  padding = ((0, 0),) + padding
  out = _select_and_scatter_add(source, operand, select_prim, window_dimensions,
                                window_strides, padding)
  return out, 0

select_and_scatter_add_p = lax.standard_primitive(
    _select_and_scatter_add_shape_rule, _input_dtype, 'select_and_scatter_add',
    partial(_select_and_scatter_add_translation, expand_padding=False))

ad.primitive_transposes[select_and_scatter_add_p] = \
    _select_and_scatter_add_transpose
ad.primitive_jvps[select_and_scatter_add_p] = _select_and_scatter_add_jvp
batching.primitive_batchers[select_and_scatter_add_p] = \
    _select_and_scatter_add_batch_rule

# TODO(b/161704903): workaround for XLA/CPU crash.
xla.register_translation(
    select_and_scatter_add_p,
    partial(_select_and_scatter_add_translation, expand_padding=True),
    platform='cpu')
# TODO(b/182390722): workaround for XLA/GPU crash.
xla.register_translation(
    select_and_scatter_add_p,
    partial(_select_and_scatter_add_translation, expand_padding=True),
    platform='gpu')

def _select_and_gather_add_shape_rule(
    tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  if tangents.shape != operand.shape:
    msg = ("select_and_gather_add tangents and operand shapes must match, "
           "got {} and {}.")
    raise TypeError(msg.format(tangents.shape, operand.shape))
  return _common_reduce_window_shape_rule(
    operand, window_dimensions, window_strides, padding, base_dilation,
    window_dilation)

def _select_and_gather_add_translation(
    ctx, avals_in, avals_out, tangents, operand, *, select_prim,
    window_dimensions, window_strides, padding, base_dilation, window_dilation,
    max_bits=64):
  c = ctx.builder
  tangents_aval, operand_aval, = avals_in
  dtype = operand_aval.dtype
  etype = xla.dtype_to_primitive_type(dtype)
  nbits = dtypes.finfo(dtype).bits

  assert nbits <= max_bits
  double_word_reduction = nbits * 2 <= max_bits

  const = lambda c, dtype, x: xops.Constant(c, np.array(x, dtype=dtype))

  if double_word_reduction:
    # TODO(b/73062247): XLA doesn't yet implement ReduceWindow on tuples, so
    # we implement a pair-wise ReduceWindow by packing two k-bit values into
    # 2k-bit unsigned integer using bit tricks.
    word_dtype = lax._UINT_DTYPES[nbits]
    double_word_dtype = lax._UINT_DTYPES[nbits * 2]
    word_type = xla.dtype_to_primitive_type(word_dtype)
    double_word_type = xla.dtype_to_primitive_type(double_word_dtype)

    # Packs two values into a tuple.
    def pack(a, b):
      a = xops.BitcastConvertType(a, word_type)
      b = xops.BitcastConvertType(b, word_type)
      a = xops.ConvertElementType(a, double_word_type)
      b = xops.ConvertElementType(b, double_word_type)
      a = xops.ShiftLeft(a, const(c, double_word_dtype, nbits))
      return xops.Or(a, b)

    # Unpacks the first element of a tuple.
    def fst(c, t):
      st = xops.ShiftRightLogical(t, const(c, double_word_dtype, nbits))
      return xops.BitcastConvertType(xops.ConvertElementType(st, word_type), etype)

    # Unpacks the second element of a tuple.
    def snd(t):
      return xops.BitcastConvertType(xops.ConvertElementType(t, word_type), etype)

  else:
    # The double-word trick above only works if we have a sufficiently large
    # type. As an alternative, we can pack two half words into a single word,
    # at the cost of precision.
    # TODO(b/73062247): add support for tuple reductions and remove this case.
    warnings.warn("Using reduced precision for gradient of reduce-window "
                  "min/max operator to work around missing XLA support for "
                  "pair-reductions. This is likely from a second or "
                  "higher derivative of a max-pooling operation.")
    r_nbits = nbits // 2
    # Drop/round the bottom mantissa bits.
    nexp = dtypes.finfo(dtype).nexp
    nmant = r_nbits - nexp - 1

    double_word_dtype = word_dtype = lax._UINT_DTYPES[nbits]
    word_type = xla.dtype_to_primitive_type(word_dtype)

    # Packs two values into a tuple.
    def pack(a, b):
      a = xops.ReducePrecision(a, exponent_bits=nexp, mantissa_bits=nmant)
      b = xops.ReducePrecision(b, exponent_bits=nexp, mantissa_bits=nmant)
      a = xops.BitcastConvertType(a, word_type)
      b = xops.BitcastConvertType(b, word_type)
      b = xops.ShiftRightLogical(b, const(c, word_dtype, r_nbits))
      return xops.Or(a, b)

    # Unpacks the first element of a tuple.
    def fst(c, t):
      st = xops.And(t, const(c, word_dtype, ((1 << r_nbits) - 1) << r_nbits))
      return xops.BitcastConvertType(st, etype)

    # Unpacks the second element of a tuple.
    def snd(t):
      return xops.BitcastConvertType(xops.ShiftLeft(t, const(c, word_dtype, r_nbits)),
                                  etype)

  def reducer():
    c = xc.XlaBuilder("select_and_gather_pair_reducer")
    x = xb.parameter(c, 0,
      xla_client.Shape.array_shape(np.dtype(double_word_dtype), ()))
    y = xb.parameter(c, 1,
      xla_client.Shape.array_shape(np.dtype(double_word_dtype), ()))
    assert select_prim is lax.ge_p or select_prim is lax.le_p
    which = xops.Ge if select_prim is lax.ge_p else xops.Le
    xops.Select(which(fst(c, x), fst(c, y)), x, y)
    return c.build()


  assert select_prim is lax.ge_p or select_prim is lax.le_p, select_prim
  init = -np.inf if select_prim is lax.ge_p else np.inf
  out = xops.ReduceWindowWithGeneralPadding(
    pack(operand, tangents), pack(const(c, dtype, init), const(c, dtype, 0)),
    reducer(), window_dimensions, window_strides, base_dilation,
    window_dilation, padding)
  return [snd(out)]

# TODO(phawkins): use this translation rule on all platforms.
def _select_and_gather_add_translation_using_variadic_reducewindow(
    ctx, avals_in, avals_out, tangents, operand, *, select_prim,
    window_dimensions, window_strides, padding, base_dilation, window_dilation):
  c = ctx.builder
  tangents_aval, operand_aval = avals_in
  dtype = operand_aval.dtype

  const = lambda c, dtype, x: xops.Constant(c, np.array(x, dtype=dtype))

  def reducer():
    c = xc.XlaBuilder("select_and_gather_pair_reducer")
    shape = xla_client.Shape.array_shape(np.dtype(dtype), ())
    kx, vx, ky, vy = (xb.parameter(c, i, shape) for i in range(4))
    which = (xops.Ge if select_prim is lax.ge_p else xops.Le)(kx, ky)
    xops.Tuple(c, [xops.Select(which, kx, ky), xops.Select(which, vx, vy)])
    return c.build()

  assert select_prim is lax.ge_p or select_prim is lax.le_p, select_prim
  init = -np.inf if select_prim is lax.ge_p else np.inf
  out = xops.ReduceWindowWithGeneralPadding(
    [operand, tangents], [const(c, dtype, init), const(c, dtype, 0)],
    reducer(), window_dimensions, window_strides, base_dilation,
    window_dilation, padding)
  return [xops.GetTupleElement(out, 1)]

def _select_and_gather_add_jvp(
    primals, tangents, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  source, operand = primals
  g_source, g_operand = tangents
  val_out = _select_and_gather_add(
      source, operand, select_prim, window_dimensions, window_strides,
      padding, base_dilation, window_dilation)
  del g_operand
  if type(g_source) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    tangent_out = _select_and_gather_add(
        g_source, operand, select_prim, window_dimensions,
        window_strides, padding, base_dilation, window_dilation)
  return val_out, tangent_out

def _select_and_gather_add_transpose(
    t, tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  assert select_prim in (lax.le_p, lax.ge_p)
  assert ad.is_undefined_primal(tangents) and not ad.is_undefined_primal(operand)
  if any(d != 1 for d in window_dilation):
    msg = ("VJP not implemented for select_and_gather (MaxPool) with window "
           "dilation, got window_dilation={}.")
    raise NotImplementedError(msg.format(window_dilation))
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(tangents.aval), None]
  has_base_dilation = any(d != 1 for d in base_dilation)
  if has_base_dilation:
    select_identity = (_get_max_identity if select_prim is lax.ge_p
                       else _get_min_identity)
    operand = lax.pad(operand, select_identity(operand.dtype),
                      tuple((0, 0, d - 1) for d in base_dilation))
  result = _select_and_scatter_add(t, operand, select_prim, window_dimensions,
                                   window_strides, padding)
  if has_base_dilation:
    result = lax.slice(result, (0,) * len(result.shape), result.shape,
                       base_dilation)
  return [result, None]

def _select_and_gather_add_batching_rule(
    batched_args, batch_dims, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  t, x = batched_args
  t_bdim, x_bdim = batch_dims
  size = next(a.shape[bdim] for a, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  t = batching.bdim_at_front(t, t_bdim, size)
  x = batching.bdim_at_front(x, x_bdim, size)
  window_dimensions = (1,) + window_dimensions
  window_strides = (1,) + window_strides
  padding = ((0, 0),) + padding
  base_dilation = (1,) + base_dilation
  window_dilation = (1,) + window_dilation
  out = _select_and_gather_add(t, x, select_prim, window_dimensions,
                               window_strides, padding, base_dilation,
                               window_dilation)
  return (out, 0)


select_and_gather_add_p = lax.standard_primitive(
    _select_and_gather_add_shape_rule, _input_dtype, 'select_and_gather_add',
    _select_and_gather_add_translation_using_variadic_reducewindow)
ad.primitive_jvps[select_and_gather_add_p] = _select_and_gather_add_jvp
ad.primitive_transposes[select_and_gather_add_p] = \
  _select_and_gather_add_transpose
batching.primitive_batchers[select_and_gather_add_p] = \
  _select_and_gather_add_batching_rule
# TODO(b/183233858): use variadic reducewindow on GPU, when implemented.
xla.register_translation(
    select_and_gather_add_p,
    _select_and_gather_add_translation,
    platform='gpu')
