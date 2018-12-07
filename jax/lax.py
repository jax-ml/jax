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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from .util import partial
import itertools
import operator
import six
from six.moves import builtins, xrange
import string

import numpy as onp

from . import core
from . import ad_util
from . import linear_util as lu
from .core import Primitive
from .abstract_arrays import (UnshapedArray, ShapedArray, ConcreteArray,
                              array_types, make_shaped_array)
from .api_util import flatten_fun, tree_to_jaxtuples
from .interpreters import partial_eval as pe
from .interpreters import xla
from .interpreters import ad
from .interpreters import batching
from .util import curry, safe_zip, unzip2, prod
from .tree_util import build_tree
from .lib import xla_bridge

_max = builtins.max
_min = builtins.max

if six.PY3:
  def maketrans(s1, s2):
    return s1.maketrans(s1, s2)
else:
  maketrans = string.maketrans

### traceables

def neg(x): return neg_p.bind(x)
def sign(x): return sign_p.bind(x)
def floor(x): return floor_p.bind(x)
def ceil(x): return ceil_p.bind(x)
def round(x): return round_p.bind(x)

def is_finite(x): return is_finite_p.bind(x)

def exp(x): return exp_p.bind(x)
def expm1(x): return expm1_p.bind(x)
def log(x): return log_p.bind(x)
def log1p(x): return log1p_p.bind(x)
def tanh(x): return tanh_p.bind(x)
def sin(x): return sin_p.bind(x)
def cos(x): return cos_p.bind(x)
def atan2(x, y): return atan2_p.bind(x, y)

def lgamma(x): return lgamma_p.bind(x)
def digamma(x): return digamma_p.bind(x)
def erf(x): return erf_p.bind(x)
def erfc(x): return erfc_p.bind(x)
def erf_inv(x): return erf_inv_p.bind(x)

def real(x): return real_p.bind(x)
def imag(x): return imag_p.bind(x)
def complex(x, y): return complex_p.bind(_brcast(x, y), _brcast(y, x))
def conj(x): return conj_p.bind(x)
def abs(x): return abs_p.bind(x)
def pow(x, y): return pow_p.bind(x, y)

def bitwise_not(x): return not_p.bind(x)
def bitwise_and(x, y): return and_p.bind(x, y)
def bitwise_or(x, y): return or_p.bind(x, y)
def bitwise_xor(x, y): return xor_p.bind(x, y)

def add(x, y): return add_p.bind(x, y)
def sub(x, y): return sub_p.bind(x, y)
def mul(x, y): return mul_p.bind(x, y)
def div(x, y): return div_p.bind(x, y)
def rem(x, y): return rem_p.bind(x, y)

def max(x, y): return max_p.bind(x, y)
def min(x, y): return min_p.bind(x, y)

def shift_left(x, y): return shift_left_p.bind(x, y)
def shift_right_arithmetic(x, y): return shift_right_arithmetic_p.bind(x, y)
def shift_right_logical(x, y): return shift_right_logical_p.bind(x, y)

def eq(x, y): return eq_p.bind(x, y)
def ne(x, y): return ne_p.bind(x, y)
def ge(x, y): return ge_p.bind(x, y)
def gt(x, y): return gt_p.bind(x, y)
def le(x, y): return le_p.bind(x, y)
def lt(x, y): return lt_p.bind(x, y)

def convert_element_type(operand, new_dtype):
  new_dtype = xla_bridge.canonicalize_dtype(new_dtype)
  old_dtype = _dtype(operand)
  if old_dtype != new_dtype:
    return convert_element_type_p.bind(
        operand, new_dtype=new_dtype, old_dtype=old_dtype)
  else:
    return operand

def bitcast_convert_type(operand, new_dtype):
  return bitcast_convert_type_p.bind(operand, new_dtype=new_dtype)

def clamp(min, operand, max):
  return clamp_p.bind(min, operand, max)

def concatenate(operands, dimension):
  return concatenate_p.bind(*operands, dimension=dimension,
                            operand_shapes=tuple(o.shape for o in operands))

def conv(lhs, rhs, window_strides, padding):
  pads = padtype_to_pads(lhs.shape[2:], rhs.shape[2:], window_strides, padding)
  return conv_general_dilated_p.bind(
      lhs, rhs, window_strides=tuple(window_strides), padding=tuple(pads),
      lhs_dilation=(), rhs_dilation=(), dimension_numbers=None,
      lhs_shape=lhs.shape, rhs_shape=rhs.shape)

def conv_with_general_padding(lhs, rhs, window_strides, padding,
                              lhs_dilation, rhs_dilation):
  return conv_general_dilated_p.bind(
      lhs, rhs, window_strides=tuple(window_strides), padding=tuple(padding),
      lhs_dilation=(), rhs_dilation=(), dimension_numbers=None,
      lhs_shape=lhs.shape, rhs_shape=rhs.shape)

def conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation,
                         rhs_dilation, dimension_numbers):
  if isinstance(padding, str):
    perms = conv_general_permutations(dimension_numbers)
    lhs_perm, rhs_perm, _ = perms
    padding = padtype_to_pads(onp.take(lhs.shape, lhs_perm)[2:],
                              onp.take(rhs.shape, rhs_perm)[2:],
                              window_strides, padding)
  return conv_general_dilated_p.bind(
      lhs, rhs, window_strides=tuple(window_strides), padding=tuple(padding),
      lhs_dilation=tuple(lhs_dilation), rhs_dilation=tuple(rhs_dilation),
      dimension_numbers=dimension_numbers, lhs_shape=lhs.shape,
      rhs_shape=rhs.shape)

def dot(lhs, rhs): return dot_p.bind(lhs, rhs)

def dot_general(lhs, rhs, dimension_numbers):
  lhs_dims, rhs_dims = dimension_numbers
  dimension_numbers = (tuple(map(tuple, lhs_dims)), tuple(map(tuple, rhs_dims)))
  return dot_general_p.bind(lhs, rhs, dimension_numbers=dimension_numbers)

def broadcast(operand, sizes):
  return broadcast_p.bind(operand, sizes=tuple(sizes))

def broadcast_in_dim(operand, shape, broadcast_dimensions):
  if operand.ndim == len(shape) and not len(broadcast_dimensions):
    return operand
  else:
    return broadcast_in_dim_p.bind(
        operand, shape=tuple(shape),
        broadcast_dimensions=tuple(broadcast_dimensions))

def reshape(operand, new_sizes, dimensions=None):
  same_shape = onp.shape(operand) == tuple(new_sizes)
  same_dims = dimensions is None or tuple(dimensions) == tuple(range(onp.ndim(operand)))
  if same_shape and same_dims:
    return operand
  else:
    return reshape_p.bind(
        operand, new_sizes=tuple(new_sizes),
        dimensions=None if dimensions is None else tuple(dimensions),
        old_sizes=onp.shape(operand))

def pad(operand, padding_value, padding_config):
  return pad_p.bind(operand, padding_value, padding_config=tuple(padding_config))

def rev(operand, dimensions):
  return rev_p.bind(operand, dimensions=tuple(dimensions))

def select(pred, on_true, on_false):
  return select_p.bind(pred, on_true, on_false)

def slice(operand, start_indices, limit_indices, strides=None):
  return slice_p.bind(operand, start_indices=tuple(start_indices),
                      limit_indices=tuple(limit_indices),
                      strides=None if strides is None else tuple(strides),
                      operand_shape=operand.shape)

def dynamic_slice(operand, start_indices, slice_sizes):
  start_indices = _dynamic_slice_indices(operand, start_indices)
  return dynamic_slice_p.bind(
      operand, start_indices, slice_sizes=tuple(slice_sizes),
      operand_shape=operand.shape)

def dynamic_update_slice(operand, update, start_indices):
  start_indices = _dynamic_slice_indices(operand, start_indices)
  return dynamic_update_slice_p.bind(operand, update, start_indices,
                                     update_shape=update.shape)

def index_take(src, idxs, axes):
  pvals = [_abstractify(arg) for arg in (src,) + idxs]
  jaxpr, _, consts = pe.trace_unwrapped_to_jaxpr(partial(_index_take, axes), pvals)
  return index_take_p.bind(src, *idxs, axes=tuple(axes),
                           input_shape=src.shape, jaxpr=jaxpr, consts=consts)

def _index_take(axes, src, *idxs):
  n = idxs[0].shape[0]
  slice_sizes = subvals(src.shape, zip(axes, [1] * len(axes)))

  def body_fun(i, state):
    src, idxs, out = state
    src_ind = (dynamic_index_in_dim(x, i, 0, False) for x in idxs)
    start_indices = subvals([0] * src.ndim, zip(axes, src_ind))
    update = dynamic_slice(src, start_indices, slice_sizes)
    update = reshape(update, (1,) + out.shape[1:])
    out = dynamic_update_slice(out, update, [i] + [0] * (out.ndim - 1))
    return src, idxs, out

  out = full_like(src, 0, shape=(n,) + tuple(onp.delete(src.shape, axes)))
  init_val = src, idxs, out
  _, _, out = fori_loop(0, n, body_fun, init_val)
  return out

def index_untake(src, dst, idxs, axes):
  pvals = [_abstractify(arg) for arg in (src, dst) + idxs]
  jaxpr, _, consts = pe.trace_unwrapped_to_jaxpr(partial(_index_untake, axes), pvals)
  return index_untake_p.bind(src, dst, *idxs, axes=tuple(axes),
                             jaxpr=jaxpr, consts=consts)

def _index_untake(axes, src, dst, *idxs):
  n = idxs[0].shape[0]
  slice_sizes = subvals(dst.shape, zip(axes, [1] * len(axes)))

  def body_fun(i, state):
    src, dst, idxs = state
    vals = dynamic_slice(src, [i] + [0] * (src.ndim - 1), (1,) + src.shape[1:])
    vals = reshape(vals, subvals(dst.shape, zip(axes, [1] * len(axes))))
    dst_ind = (dynamic_index_in_dim(x, i, 0, False) for x in idxs)
    start_indices = subvals([0] * dst.ndim, zip(axes, dst_ind))
    update = add(vals, dynamic_slice(dst, start_indices, slice_sizes))
    dst = dynamic_update_slice(dst, update, start_indices)
    return src, dst, idxs

  init_val = src, dst, idxs
  _, dst, _ = fori_loop(0, n, body_fun, init_val)
  return dst

def transpose(operand, permutation):
  return transpose_p.bind(operand, permutation=tuple(permutation))

def reduce(operand, init_value, computation, dimensions):
  monoid_reducer = _get_monoid_reducer(computation, init_value)
  if monoid_reducer:
    return monoid_reducer(operand, dimensions)
  else:
    jaxpr, consts = _reduction_jaxpr(computation, init_value)
    return reduce_p.bind(operand, init_value, jaxpr=jaxpr, consts=consts,
                         dimensions=tuple(dimensions))

def _reduction_jaxpr(computation, init_value):
  pval = _abstractify(init_value)
  jaxpr, _, consts = pe.trace_unwrapped_to_jaxpr(computation, (pval, pval))
  return jaxpr, consts

def _get_monoid_reducer(monoid_op, x):
  aval = core.get_aval(x)
  if (type(aval) is ConcreteArray) and aval.shape == ():
    if monoid_op is add:
      return aval.val == 0 and _reduce_sum
    elif monoid_op is max:
      return aval.val == _get_max_identity(aval.dtype) and _reduce_max
    elif monoid_op is min:
      return aval.val == _get_min_identity(aval.dtype) and _reduce_min

def _get_max_identity(dtype):
  if onp.issubdtype(dtype, onp.floating):
    return onp.array(-onp.inf, dtype)
  elif onp.issubdtype(dtype, onp.integer):
    return onp.array(onp.iinfo(dtype).min, dtype)

def _get_min_identity(dtype):
  if onp.issubdtype(dtype, onp.floating):
    return onp.array(onp.inf, dtype)
  elif onp.issubdtype(dtype, onp.integer):
    return onp.array(onp.iinfo(dtype).max, dtype)

def _reduce_sum(operand, axes):
  return reduce_sum_p.bind(operand, axes=tuple(axes), input_shape=operand.shape)

def _reduce_max(operand, axes):
  return reduce_max_p.bind(operand, axes=tuple(axes))

def _reduce_min(operand, axes):
  return reduce_min_p.bind(operand, axes=tuple(axes))

def reduce_window(operand, init_value, computation, window_dimensions,
                  window_strides, padding):
  monoid_reducer = _get_monoid_window_reducer(computation, init_value)
  if monoid_reducer:
    return monoid_reducer(operand, window_dimensions, window_strides, padding)
  else:
    jaxpr, consts = _reduction_jaxpr(computation, init_value)
    return reduce_window_p.bind(
        operand, init_value, jaxpr=jaxpr, consts=consts,
        window_dimensions=tuple(window_dimensions),
        window_strides=tuple(window_strides), padding=padding)

def _get_monoid_window_reducer(monoid_op, x):
  aval = core.get_aval(x)
  if (type(aval) is ConcreteArray) and aval.shape == ():
    if monoid_op is add:
      return aval.val == 0 and _reduce_window_sum
    elif monoid_op is max:
      return aval.val == _get_max_identity(aval.dtype) and _reduce_window_max
    elif monoid_op is min:
      return aval.val == _get_min_identity(aval.dtype) and _reduce_window_min

def _reduce_window_sum(operand, window_dimensions, window_strides, padding):
  return reduce_window_sum_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=padding,
      input_shape=operand.shape)

def _reduce_window_max(operand, window_dimensions, window_strides, padding):
  return reduce_window_max_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=padding)

def _reduce_window_min(operand, window_dimensions, window_strides, padding):
  return reduce_window_min_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=padding)

def _select_and_scatter(operand, select, window_dimensions, window_strides,
                        padding, source, init_value, scatter):
  select_jaxpr, select_consts = _reduction_jaxpr(select)
  scatter_jaxpr, scatter_consts = _reduction_jaxpr(scatter)
  return select_and_scatter_p.bind(
      operand, source, init_value, select_jaxpr=select_jaxpr,
      select_consts=select_consts, scatter_jaxpr=scatter_jaxpr,
      scatter_consts=scatter_consts, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=padding)

def _select_and_scatter_add(source, operand, select_prim, window_dimensions,
                            window_strides, padding):
  return select_and_scatter_add_p.bind(
      source, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=padding)

def _select_and_gather_add(tangents, operand, select_prim, window_dimensions,
                           window_strides, padding):
  return select_and_gather_add_p.bind(
      tangents, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=padding)

def sort(operand, dimension=-1):
  return sort_p.bind(operand, dimension=-1)

def sort_key_val(keys, values, dimension=-1):
  # TODO new sort_key_val is variadic
  result = sort_key_val_p.bind(keys, values, dimension=dimension)
  sorted_keys, sorted_values = result
  return sorted_keys, sorted_values

def _while_loop(cond_fun, body_fun, init_val):
  init_val_flat, in_tree = tree_to_jaxtuples(init_val)
  flat_body_fun, out_tree = flatten_fun(lu.wrap_init(body_fun), (in_tree,))
  flat_cond_fun, _ = flatten_fun(lu.wrap_init(cond_fun), (in_tree,))

  pval_flat = _abstractify(init_val_flat)
  cond_jaxpr, _, cond_consts = pe.trace_to_jaxpr(flat_cond_fun, (pval_flat,))
  body_jaxpr, pvout, body_consts = pe.trace_to_jaxpr(flat_body_fun, (pval_flat,))
  abs_out, _ = pvout

  params = OpaqueParam((abs_out, cond_jaxpr, cond_consts, body_jaxpr, body_consts))
  out_flat = while_p.bind(init_val_flat, opaque_params=params)
  if out_tree() != in_tree:
    raise TypeError("body_fun input and output must have identical structure")
  return build_tree(out_tree(), out_flat)

class OpaqueParam(object):
  __slots__ = ["val", "id"]
  def __init__(self, val):
    self.val = val
    self.id = next(opaque_param_ids)
  def __hash__(self):
    return self.id
opaque_param_ids = itertools.count()


### convenience wrappers around traceables


def full_like(x, fill_value, dtype=None, shape=None):
  """Create a full array like np.full based on the example array `x`.

  Args:
    x: example array-like, used for shape and dtype information.
    fill_value: a scalar value to fill the entries of the output array.
    dtype: optional, a dtype parameter for the output ndarray.
    shape: optional, a shape parameter for the output ndarray.

  Returns:
    An ndarray with the same shape as `x` with its entries set equal to
    `fill_value`, similar to the output of np.full.
  """
  shape = onp.shape(x) if shape is None else shape
  return broadcast(onp.array(fill_value, dtype or _dtype(x)), shape)


def collapse(operand, start_dimension, stop_dimension):
  lo, hi = start_dimension, stop_dimension
  size = prod(operand.shape[lo:hi])
  new_shape = operand.shape[:lo] + (size,) + operand.shape[hi:]
  return reshape(operand, new_shape)


def slice_in_dim(operand, start_index, limit_index, stride=1, axis=0):
  """Convenience wrapper around slice applying to only one dimension."""
  start_indices = [0] * operand.ndim
  limit_indices = list(operand.shape)
  strides = [1] * operand.ndim

  start_indices[axis] = start_index
  limit_indices[axis] = limit_index
  strides[axis] = stride

  return slice(operand, start_indices, limit_indices, strides)


def index_in_dim(operand, index, axis=0, keepdims=True):
  """Convenience wrapper around slice to perform int indexing."""
  axis_size = operand.shape[axis]
  wrapped_index = index + axis_size if index < 0 else index
  if not 0 <= wrapped_index < axis_size:
    msg = 'index {} is out of bounds for axis {} with size {}'
    raise IndexError(msg.format(index, axis, axis_size))
  result = slice_in_dim(operand, wrapped_index, wrapped_index + 1, 1, axis)
  if keepdims:
    return result
  else:
    return reshape(result, onp.delete(operand.shape, axis))


def dynamic_slice_in_dim(operand, start_index, slice_size, axis=0):
  """Convenience wrapper around dynamic_slice applying to one dimension."""
  start_indices = [onp.array([0])] * operand.ndim
  slice_sizes = list(operand.shape)

  start_indices[axis] = reshape(rem(start_index, operand.shape[axis]), [1])
  slice_sizes[axis] = slice_size

  start_indices = concatenate(start_indices, 0)
  return dynamic_slice(operand, start_indices, slice_sizes)


def dynamic_index_in_dim(operand, index, axis=0, keepdims=True):
  """Convenience wrapper around dynamic_slice to perform int indexing."""
  result = dynamic_slice_in_dim(operand, index, 1, axis)
  if keepdims:
    return result
  else:
    return reshape(result, onp.delete(operand.shape, axis))


def dynamic_update_slice_in_dim(operand, update, start_index, axis):
  start_indices = [0] * _ndim(operand)
  start_indices[axis] = start_index % operand.shape[axis]
  return dynamic_update_slice(operand, update, start_indices)


def dynamic_update_index_in_dim(operand, update, index, axis):
  if _ndim(update) != _ndim(operand):
    assert _ndim(update) + 1 == _ndim(operand)
    ax = axis % _ndim(operand)
    update = reshape(update, operand.shape[:ax] + (1,) + operand.shape[ax:])
  return dynamic_update_slice_in_dim(operand, update, index, axis)


def fori_loop(lower, upper, body_fun, init_val):
  """Loop from `lower` to `upper` by reduction to `while_loop`.

  Arguments:
    lower: loop index lower bound (inclusive)
    upper: loop index upper bound (exclusive)
    body_fun: function of type (int, T) -> T, where T is the type of `init_val`
    init_val: initial loop value, of type T

  Returns:
    Loop value from the final iteration, of type T.
  """
  # state: (upper limit, index, loop value)
  # The `lt` and `add` functions are added to the namespace programmatically.
  _, _, result = _while_loop(
      lambda upper_i_x: lt(upper_i_x[1], upper_i_x[0]),
      lambda upper_i_x: (upper_i_x[0], add(upper_i_x[1], 1),
                         body_fun(upper_i_x[1], upper_i_x[2])),
      (upper, lower, init_val))
  return result


def foreach_loop(sequence, body_fun, init_val):
  """Loop over `sequence` by reduction to `while_loop`.

  Arguments:
    sequence: tuple of loop items, each of type U
    body_fun: function of type (U, T) -> T, where T is the type of `init_val`
    init_val: initial loop value, of type T

  Returns:
    Loop value from the final iteration, of type T.
  """
  _, result = fori_loop(
      0, len(sequence),
      lambda i, seq_val: body_fun(seq_val[0][i], seq_val[1]),
      (sequence, init_val))
  return result


def batch_matmul(lhs, rhs):
  """Batch matrix multiplication."""
  if _min(lhs.ndim, rhs.ndim) < 2:
    raise ValueError('Arguments to batch_matmul must be at least 2D, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  if lhs.ndim != rhs.ndim:
    raise ValueError('Arguments to batch_matmul must have same ndim, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  lhs_contract = (lhs.ndim - 1,)
  rhs_contract = (rhs.ndim - 2,)
  batch = tuple(range(lhs.ndim - 2))
  return dot_general(lhs, rhs, [(lhs_contract, rhs_contract), (batch, batch)])


# These trig functions also exist in the XLA client library, but we treat them
# as non-primitive to maintain a smaller set of autodiff primitives.

def sqrt(x):
  return pow(x, _const(x, 0.5))

def rsqrt(x):
  return pow(x, _const(x, -0.5))

def square(x):
  return mul(x, x)

def reciprocal(x):
  return div(_const(x, 1.), x)

def tan(x):
  return div(sin(x), cos(x))

def asin(x):
  # asin(x) = 2 * atan(x / (1 + sqrt(1 - x**2)))
  return mul(_const(x, 2.),
             atan2(x, add(_const(x, 1.), sqrt(add(_const(x, 1.), square(x))))))

def acos(x):
  # acos(x) = 2 * atan(sqrt(1 - x**2) / (1 + x))
  return mul(_const(x, 2.),
             atan2(sqrt(sub(_const(x, 1.), square(x))), add(_const(x, 1.), x)))

def atan(x):
  return atan2(x, _const(x, 1.))

def sinh(x):
  return mul(_const(x, 0.5), sub(exp(x), exp(neg(x))))

def cosh(x):
  return mul(_const(x, 0.5), add(exp(x), exp(neg(x))))

def asinh(x):
  # asinh(x) = log(x + sqrt(x**2 + 1))
  return log(add(x, sqrt(add(mul(x, x), _const(x, 1.)))))

def acosh(x):
  # acosh(x) = log(x + sqrt((x + 1) * (x - 1)))
  return log(add(x, mul(sqrt(add(x, _const(x, 1.))),
                        sqrt(sub(x, _const(x, 1.))))))


# Add some methods to ShapedArray that rely on lax primitives

ShapedArray.broadcast = core.aval_method(broadcast)
ShapedArray.transpose = core.aval_method(transpose)  # clobbered by lax_numpy
ShapedArray.reshape = core.aval_method(reshape)      # clobbered by lax_numpy

def _iter(tracer):
  if tracer.ndim == 0:
    raise TypeError("iteration over a 0-d array")  # same as numpy error
  else:
    n = tracer.shape[0]
    return (index_in_dim(tracer, i, keepdims=False) for i in xrange(n))
ShapedArray._iter = staticmethod(_iter)

# Add some ad handlers that use (or could use) lax primitives

def zeros_like_array(x):
  dtype = xla_bridge.canonicalize_dtype(_dtype(x))
  return onp.broadcast_to(onp.zeros((), dtype), onp.shape(x))

for t in itertools.chain(array_types, [xla.DeviceArray]):
  ad_util.jaxval_adders[t] = add
  ad_util.jaxval_zeros_likers[t] = zeros_like_array

batching.pytype_aval_mappings[xla.DeviceArray] = make_shaped_array


### primitives


_input_dtype = lambda *args, **_: xla_bridge.canonicalize_dtype(args[0].dtype)
_fixed_dtype = lambda dtype: lambda *args, **kwargs: xla_bridge.canonicalize_dtype(dtype)
_complex_basetype = lambda dtype: onp.abs(onp.zeros((), dtype)).dtype

def identity(x): return x


def standard_primitive(shape_rule, dtype_rule, name, translation_rule=None):
  prim = Primitive(name)
  prim.def_impl(partial(xla.apply_primitive, prim))
  prim.def_abstract_eval(partial(standard_abstract_eval, shape_rule, dtype_rule))
  xla.translations[prim] = translation_rule or partial(standard_translate, name)
  return prim

def standard_abstract_eval(shape_rule, dtype_rule, *args, **kwargs):
  assert all(isinstance(arg, UnshapedArray) for arg in args), args
  least_specialized = _max(
      map(type, args), key=operator.attrgetter('array_abstraction_level'))
  if least_specialized is ConcreteArray:
    return ShapedArray(shape_rule(*args, **kwargs), dtype_rule(*args, **kwargs))
  elif least_specialized is ShapedArray:
    return ShapedArray(shape_rule(*args, **kwargs), dtype_rule(*args, **kwargs))
  elif least_specialized is UnshapedArray:
    return UnshapedArray(dtype_rule(*args, **kwargs))
  else:
    raise TypeError(args, least_specialized)


def standard_translate(name, c, *args, **kwargs):
  xla_opname = ''.join(term.capitalize() for term in name.split('_'))
  return getattr(c, xla_opname)(*args, **kwargs)


def unop_dtype_rule(result_dtype, accepted_dtypes, name, aval):
  if not any(onp.issubdtype(aval.dtype, t) for t in accepted_dtypes):
    msg = '{} does not accept dtype {}. Accepted dtypes are subtypes of {}.'
    typename = str(onp.dtype(aval.dtype).name)
    accepted_typenames = (str(onp.dtype(t).name) for t in accepted_dtypes)
    raise TypeError(msg.format(name, typename, ', '.join(accepted_typenames)))
  return result_dtype(aval.dtype)


def unop(result_dtype, accepted_dtypes, name):
  dtype_rule = partial(unop_dtype_rule, result_dtype, accepted_dtypes, name)
  prim = standard_primitive(operator.attrgetter('shape'), dtype_rule, name)
  batching.defvectorized(prim)
  return prim
standard_unop = partial(unop, identity)


def binop_dtype_rule(result_dtype, accepted_dtypes, name, *avals):
  aval_dtypes = [aval.dtype for aval in avals]
  for i, (aval_dtype, types) in enumerate(zip(aval_dtypes, accepted_dtypes)):
    if not any(onp.issubdtype(aval_dtype, t) for t in types):
      msg = ('{} does not accept dtype {} at position {}. '
             'Accepted dtypes at position {} are subtypes of {}.')
      typename = str(onp.dtype(aval_dtype).name)
      typenames = ', '.join(str(onp.dtype(t).name) for t in types)
      raise TypeError(msg.format(name, typename, i, i, typenames))
  _check_same_dtypes(name, False, *aval_dtypes)
  return result_dtype(*avals)


def broadcasting_shape_rule(name, *avals):
  shapes = onp.array([aval.shape for aval in avals if aval.shape])
  if not shapes.size:
    return ()
  if len({len(shape) for shape in shapes}) != 1:
    msg = '{} got arrays of different rank: {}.'
    raise TypeError(msg.format(name, ', '.join(map(str, map(tuple, shapes)))))
  result_shape = onp.max(shapes, axis=0)
  if not onp.all((shapes == result_shape) | (shapes == 1)):
    msg = '{} got incompatible shapes for broadcasting: {}.'
    raise TypeError(msg.format(name, ', '.join(map(str, map(tuple, shapes)))))
  return tuple(result_shape)


def binop(result_dtype, accepted_dtypes, name):
  dtype_rule = partial(binop_dtype_rule, result_dtype, accepted_dtypes, name)
  shape_rule = partial(broadcasting_shape_rule, name)
  prim = standard_primitive(shape_rule, dtype_rule, name)
  batching.defbroadcasting(prim)
  return prim
standard_binop = partial(binop, _input_dtype)


# NOTE(mattjj): this isn't great for orchestrate fwd mode because it means JVPs
# get two extra ops in them: a reshape and a broadcast_in_dim (or sometimes just
# a broadcast). but saving the shape info with the primitives isn't great either
# because then we can't trace these ops without shape data.
def _brcast(x, *others):
  # used in jvprules to make binop broadcasting explicit for transposability.
  # requires shape info during jvp tracing, which isn't strictly necessary.
  shapes = list(filter(None, map(onp.shape, (x,) + others)))
  shape = tuple(shapes and onp.max(shapes, axis=0))
  if onp.shape(x) != shape:
    return _brcast_to(x, shape)
  else:
    return x


def _brcast_to(x, shape):
  x_shape = onp.shape(x)
  assert x_shape != shape
  if x_shape:
    assert len(x_shape) == len(shape)
    broadcast_dimensions, = onp.where(onp.equal(x_shape, shape))
    squeezed_dimensions, = onp.where(onp.not_equal(x_shape, shape))
    inshape = onp.delete(x_shape, squeezed_dimensions)
    return broadcast_in_dim(reshape(x, inshape), shape, broadcast_dimensions)
  else:
    return broadcast(x, shape)


_f32 = {onp.float32}
_float = {onp.floating}
_complex = {onp.complex64}
_int = {onp.integer}
_bool = {onp.bool_}

_num = _int | _float | _complex
_any = _int | _float | _complex | _bool


neg_p = standard_unop(_num, 'neg')
ad.deflinear(neg_p, lambda t: [neg(t)])
batching.defvectorized(neg_p)

sign_p = standard_unop(_num, 'sign')
ad.defjvp_zero(sign_p)

floor_p = standard_unop(_float, 'floor')
ad.defjvp_zero(floor_p)

ceil_p = standard_unop(_float, 'ceil')
ad.defjvp_zero(ceil_p)

round_p = standard_unop(_float, 'round')
ad.defjvp_zero(round_p)

is_finite_p = unop(_fixed_dtype(onp.bool_), _float, 'is_finite')
ad.defjvp_zero(is_finite_p)

exp_p = standard_unop(_float | _complex, 'exp')
ad.defjvp2(exp_p, lambda g, ans, x: mul(g, ans))

log_p = standard_unop(_float | _complex, 'log')
ad.defjvp(log_p, lambda g, x: div(g, x))

expm1_p = standard_unop(_float | _complex, 'expm1')
ad.defjvp2(expm1_p, lambda g, ans, x: mul(g, add(ans, _one(ans))))

log1p_p = standard_unop(_float | _complex, 'log1p')
ad.defjvp(log1p_p, lambda g, x: div(g, add(x, _one(x))))

tanh_p = standard_unop(_float | _complex, 'tanh')
ad.defjvp(tanh_p, lambda g, x: div(g, pow(cosh(x), _two(x))))

sin_p = standard_unop(_float | _complex, 'sin')
ad.defjvp(sin_p, lambda g, x: mul(g, cos(x)))

cos_p = standard_unop(_float | _complex, 'cos')
ad.defjvp(cos_p, lambda g, x: neg(mul(g, sin(x))))

atan2_p = standard_binop([_float, _float], 'atan2')

lgamma_p = standard_unop(_float, 'lgamma')
ad.defjvp(lgamma_p, lambda g, x: mul(g, digamma(x)))

digamma_p = standard_unop(_float, 'digamma')

erf_p = standard_unop(_float, 'erf')
ad.defjvp(erf_p, lambda g, x: mul(_const(x, 2. / onp.sqrt(onp.pi)),
                                  mul(g, exp(neg(square(x))))))

erfc_p = standard_unop(_float, 'erfc')
ad.defjvp(erfc_p, lambda g, x: mul(_const(x, 2. / onp.sqrt(onp.pi)),
                                   mul(neg(g), exp(neg(square(x))))))

erf_inv_p = standard_unop(_float, 'erf_inv')
ad.defjvp2(erf_inv_p, lambda g, ans, x: mul(_const(x, onp.sqrt(onp.pi) / 2.),
                                            mul(g, exp(square(ans)))))

real_p = unop(_fixed_dtype(onp.float32), _complex, 'real')
ad.deflinear(real_p, lambda t: [complex(t, onp.zeros((), onp.float32))])

imag_p = unop(_fixed_dtype(onp.float32), _complex, 'imag')
ad.deflinear(imag_p, lambda t: [complex(onp.zeros((), onp.float32), neg(t))])

complex_p = standard_binop([_f32, _f32], 'complex')
ad.deflinear(complex_p, lambda t: [real(t), imag(t)])

# TODO promotes dtypes, need to remember whether we came from float or not
conj_p = unop(_fixed_dtype(onp.complex64), _float | _complex, 'conj')
ad.deflinear(conj_p, lambda t: [conj(t)])

abs_p = unop(_complex_basetype, _num, 'abs')
ad.defjvp2(abs_p,
           lambda g, ans, x: div(_maybe_real(mul(g, _maybe_conj(x))),
                                 _replace_zero(ans)))
_maybe_conj = lambda x: conj(x) if _iscomplex(x) else x
_maybe_real = lambda x: real(x) if _iscomplex(x) else x

# TODO handle broadcasting
pow_p = standard_binop([_float | _complex, _float | _complex], 'pow')
ad.defjvp(pow_p,
          lambda g, x, y: mul(_brcast(g, y), mul(y, pow(x, select(
              eq(y, _zeros(y)), _ones(y), sub(y, _ones(y)))))),
          lambda g, x, y: mul(_brcast(g, x),
                              mul(log(_replace_zero(x)), pow(x, y))))
_replace_zero = lambda x: select(eq(x, _const(x, 0)), _ones(x), x)

not_p = standard_unop(_int | _bool, 'not')

and_p = standard_binop([_any, _any], 'and')
ad.defjvp_zero(and_p)

or_p = standard_binop([_any, _any], 'or')
ad.defjvp_zero(or_p)

xor_p = standard_binop([_any, _any], 'xor')
ad.defjvp_zero(xor_p)

add_p = standard_binop([_num, _num], 'add')
ad.defjvp(add_p, lambda g, x, y: _brcast(g, y), lambda g, x, y: _brcast(g, x))

sub_p = standard_binop([_num, _num], 'sub')
ad.defjvp(sub_p,
          lambda g, x, y: _brcast(g, y),
          lambda g, x, y: _brcast(neg(g), x))

mul_p = standard_binop([_num, _num], 'mul')
ad.defbilinear_broadcasting(_brcast, mul_p, mul, mul)  # TODO


def div_transpose_rule(cotangent, x, y):
  assert x is None
  res = ad_util.zero if cotangent is ad_util.zero else div(cotangent, y)
  return res, None
div_p = standard_binop([_num, _num], 'div')
ad.defjvp(div_p,
          lambda g, x, y: div(_brcast(g, y), y),
          lambda g, x, y: div(mul(neg(_brcast(g, x)), x), pow(y, _two(y))))
ad.primitive_transposes[div_p] = div_transpose_rule

rem_p = standard_binop([_num, _num], 'rem')
ad.defjvp(rem_p,
          lambda g, x, y: _brcast(g, y),
          lambda g, x, y: mul(neg(g), floor(div(x, y))))


max_p = standard_binop([_any, _any], 'max')
ad.defjvp2(max_p,
           lambda g, ans, x, y: mul(_brcast(g, y), _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(_brcast(g, x), _balanced_eq(y, ans, x)))

min_p = standard_binop([_any, _any], 'min')
ad.defjvp2(min_p,
           lambda g, ans, x, y: mul(_brcast(g, y), _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(_brcast(g, x), _balanced_eq(y, ans, x)))


shift_left_p = standard_binop([_int, _int], 'shift_left')
ad.defjvp_zero(shift_left_p)

shift_right_arithmetic_p = standard_binop([_int, _int], 'shift_right_arithmetic')
ad.defjvp_zero(shift_right_arithmetic_p)

shift_right_logical_p = standard_binop([_int, _int], 'shift_right_logical')
ad.defjvp_zero(shift_right_logical_p)

eq_p = binop(_fixed_dtype(onp.bool_), [_any, _any], 'eq')
ad.defjvp_zero(eq_p)

ne_p = binop(_fixed_dtype(onp.bool_), [_any, _any], 'ne')
ad.defjvp_zero(ne_p)

ge_p = binop(_fixed_dtype(onp.bool_), [_any, _any], 'ge')
ad.defjvp_zero(ge_p)

gt_p = binop(_fixed_dtype(onp.bool_), [_any, _any], 'gt')
ad.defjvp_zero(gt_p)

le_p = binop(_fixed_dtype(onp.bool_), [_any, _any], 'le')
ad.defjvp_zero(le_p)

lt_p = binop(_fixed_dtype(onp.bool_), [_any, _any], 'lt')
ad.defjvp_zero(lt_p)


def convert_element_type_shape_rule(operand, new_dtype, old_dtype):
  return operand.shape

def convert_element_type_dtype_rule(operand, new_dtype, old_dtype):
  return new_dtype

def convert_element_type_translation_rule(c, operand, new_dtype, old_dtype):
  new_etype = xla_bridge.dtype_to_etype(new_dtype)
  return c.ConvertElementType(operand, new_element_type=new_etype)

convert_element_type_p = standard_primitive(
    convert_element_type_shape_rule, convert_element_type_dtype_rule,
    'convert_element_type', convert_element_type_translation_rule)
ad.deflinear(
    convert_element_type_p,
    lambda t, new_dtype, old_dtype: [convert_element_type(t, old_dtype)])
batching.defvectorized(convert_element_type_p)


def bitcast_convert_type_shape_rule(operand, new_dtype):
  return operand.shape

def bitcast_convert_type_dtype_rule(operand, new_dtype):
  return new_dtype

def bitcast_convert_type_translation_rule(c, operand, new_dtype):
  new_etype = xla_bridge.dtype_to_etype(new_dtype)
  return c.BitcastConvertType(operand, new_element_type=new_etype)

bitcast_convert_type_p = standard_primitive(
    bitcast_convert_type_shape_rule, bitcast_convert_type_dtype_rule,
    'bitcast_convert_type', bitcast_convert_type_translation_rule)
ad.defjvp_zero(bitcast_convert_type_p)
batching.defvectorized(bitcast_convert_type_p)


def conv_general_dilated_shape_rule(
    lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers=None, **unused_kwargs):
  if dimension_numbers is None:
    lhs_dilated = _dilate_shape(lhs.shape, lhs_dilation)
    rhs_dilated = _dilate_shape(rhs.shape, rhs_dilation)
    _check_conv_shapes('conv_general_dilated', lhs_dilated, rhs_dilated,
                       window_strides)
    return conv_shape_tuple(lhs_dilated, rhs_dilated, window_strides, padding)
  else:
    if not isinstance(dimension_numbers, (tuple, list)):
      msg = "conv_general_dilated dimension_numbers must be tuple/list, got {}."
      raise TypeError(msg.format(type(dimension_numbers)))
    if len(dimension_numbers) != 3:
      msg = "conv_general_dilated dimension_numbers must be length 3, got {}."
      raise TypeError(msg.format(len(dimension_numbers)))
    if not all(isinstance(elt, str) for elt in dimension_numbers):
      msg = ("conv_general_dilated dimension_numbers elements must be strings, "
            "got {}.")
      raise TypeError(msg.format(tuple(map(type, dimension_numbers))))
    msg = ("conv_general_dilated dimension_numbers[{}] must have len equal to "
           "the ndim of lhs and rhs, got {} for lhs and rhs shapes {} and {}.")
    for i, elt in enumerate(dimension_numbers):
      if len(elt) != lhs.ndim:
        raise TypeError(msg.format(i, len(elt), lhs.shape, rhs.shape))

    lhs_perm, rhs_perm, out_perm = conv_general_permutations(dimension_numbers)
    lhs_trans = _dilate_shape(onp.take(lhs.shape, lhs_perm), lhs_dilation)
    rhs_trans = _dilate_shape(onp.take(rhs.shape, rhs_perm), rhs_dilation)
    out_trans = conv_shape_tuple(lhs_trans, rhs_trans, window_strides, padding)
    return tuple(onp.take(out_trans, onp.argsort(out_perm)))

def conv_general_dilated_dtype_rule(
    lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, **unused_kwargs):
  return binop_dtype_rule(_input_dtype, [_f32, _f32], 'conv_general_dilated',
                          lhs, rhs)

def conv_general_dilated_transpose_lhs(
    g, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, lhs_shape, rhs_shape):
  if dimension_numbers is None:
    nd = len(lhs_shape)
    lhs_sdims = rhs_sdims = out_sdims = list(range(2, nd))
    trans_dimension_numbers = ConvolutionDimensionNumbers(
        tuple(range(nd)), (1, 0) + tuple(range(2, nd)), tuple(range(nd)))
  else:
    lhs_sdims, rhs_sdims, out_sdims = _get_sdims(dimension_numbers)
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    trans_dimension_numbers = out_spec, _charswap("I", "O", rhs_spec), lhs_spec

  padding = _conv_general_vjp_lhs_padding(
      onp.take(lhs_shape, lhs_sdims), onp.take(rhs_shape, rhs_sdims),
      window_strides, onp.take(g.shape, out_sdims), padding, lhs_dilation,
      rhs_dilation)
  revd_weights = rev(rhs, rhs_sdims)
  return conv_general_dilated(
      g, revd_weights, window_strides=lhs_dilation, padding=padding,
      lhs_dilation=window_strides, rhs_dilation=rhs_dilation,
      dimension_numbers=trans_dimension_numbers)


def conv_general_dilated_transpose_rhs(
    g, lhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, lhs_shape, rhs_shape):
  if dimension_numbers is None:
    nd = len(lhs_shape)
    lhs_sdims = rhs_sdims = out_sdims = list(range(2, nd))
    trans_dimension_numbers = ConvolutionDimensionNumbers(
        (1, 0) + tuple(range(2, nd)),
        (1, 0) + tuple(range(2, nd)),
        (1, 0) + tuple(range(2, nd)))
  else:
    lhs_sdims, rhs_sdims, out_sdims = _get_sdims(dimension_numbers)
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    trans_dimension_numbers = (_charswap("C", "N", lhs_spec),
                               out_spec.translate(maketrans("NC", "IO")),
                               rhs_spec.translate(maketrans("IO", "NC")))

  padding = _conv_general_vjp_rhs_padding(
      onp.take(lhs_shape, lhs_sdims), onp.take(rhs_shape, rhs_sdims),
      window_strides, onp.take(g.shape, out_sdims), padding, lhs_dilation,
      rhs_dilation)
  return conv_general_dilated(
      lhs, g, window_strides=rhs_dilation, padding=padding,
      lhs_dilation=lhs_dilation, rhs_dilation=window_strides,
      dimension_numbers=trans_dimension_numbers)


def conv_general_dilated_translation_rule(
    c, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, **unused_kwargs):
  if isinstance(dimension_numbers, ConvolutionDimensionNumbers):
    dimension_numbers = _conv_general_proto(dimension_numbers)
  return c.ConvGeneralDilated(lhs, rhs, window_strides, padding, lhs_dilation,
                              rhs_dilation, dimension_numbers)

conv_general_dilated_p = standard_primitive(
    conv_general_dilated_shape_rule, conv_general_dilated_dtype_rule,
    'conv_general_dilated', conv_general_dilated_translation_rule)
ad.defbilinear(conv_general_dilated_p,
               conv_general_dilated_transpose_lhs,
               conv_general_dilated_transpose_rhs)


def dot_shape_rule(lhs, rhs):
  if lhs.ndim == 0 or rhs.ndim == 0:
    msg = "Dot only supports rank 1 or above, got shapes {} and {}."
    raise TypeError(msg.format(lhs.shape, rhs.shape))
  if lhs.ndim > 2 or rhs.ndim > 2:
    msg = "Dot only supports rank 2 or less, got shapes {} and {}."
    raise TypeError(msg.format(lhs.shape, rhs.shape))

  def require(shape_cond):
    if not shape_cond:
      msg = "Incompatible shapes for dot: got {} and {}."
      raise TypeError(msg.format(lhs.shape, rhs.shape))

  if lhs.ndim == rhs.ndim == 1:
    require(lhs.shape == rhs.shape)
    return ()
  elif lhs.ndim == rhs.ndim == 2:
    require(lhs.shape[1] == rhs.shape[0])
    return (lhs.shape[0], rhs.shape[1])
  elif rhs.ndim == 1:
    require(lhs.shape[-1] == rhs.shape[0])
    return lhs.shape[:-1]
  else:
    require(lhs.shape[-1] == rhs.shape[-2])
    return lhs.shape[:-1] + rhs.shape[:-2] + rhs.shape[-1:]

def dot_transpose_lhs(t, rhs):
  if onp.ndim(t) == onp.ndim(rhs) == 2:
    return dot(t, transpose(rhs, (1, 0)))
  elif onp.ndim(t) == 1 and onp.ndim(rhs) == 2:
    return dot(rhs, t)
  elif onp.ndim(t) == onp.ndim(rhs) == 1:
    return _outer(t, rhs)
  elif onp.ndim(t) == 0 or onp.ndim(rhs) == 0:
    return mul(t, rhs)
  else:
    raise TypeError

def dot_transpose_rhs(t, lhs):
  if onp.ndim(lhs) == onp.ndim(t) == 2:
    return dot(transpose(lhs, (1, 0)), t)
  elif onp.ndim(lhs) == 2 and onp.ndim(t) == 1:
    return dot(t, lhs)
  elif onp.ndim(t) == onp.ndim(lhs) == 1:
    return _outer(lhs, t)
  elif onp.ndim(t) == 0 or onp.ndim(lhs) == 0:
    return mul(t, lhs)
  else:
    raise TypeError

def _outer(x, y):
  assert onp.ndim(x) == onp.ndim(y) == 1
  return mul(reshape(x, (x.shape[0], 1)), reshape(y, (1, y.shape[0])))

def dot_batch_rule(batched_args, batch_dims):
  lhs, rhs = batched_args
  lbd, rbd = batch_dims
  T = lambda x: transpose(x, onp.arange(onp.ndim(x))[::-1])

  if max(onp.ndim(lhs), onp.ndim(rhs)) <= 2:
    if rbd is None:
      assert lbd in (0, 1)
      if lbd == 0:
        return dot(lhs, rhs), 0
      else:
        return dot(T(rhs), lhs), 1

    if lbd is None:
      assert rbd in (0, 1)
      if rbd == onp.ndim(rhs) - 1:
        return dot(lhs, rhs), 1
      else:
        return dot(rhs, T(lhs)), 0

    assert False  # unreachable

  if lbd is None:
    assert rbd is not None
    lhs = broadcast(lhs, (rhs.shape[rbd],))
  else:
    lhs = batching.move_dim_to_front(lhs, lbd)
  lhs_batch = (0,)
  lhs_contracting = (onp.ndim(lhs) - 1,)

  if rbd is None:
    assert lbd is not None
    rhs = broadcast(rhs, (lhs.shape[lbd],))
  else:
    rhs = batching.move_dim_to_front(rhs, rbd)
  rhs_batch = (0,)
  rhs_contracting = (onp.arange(1, onp.ndim(rhs))[-2:][0],)

  dim_nums = [(lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch)]
  return dot_general(lhs, rhs, dim_nums), 0

dot_dtype_rule = partial(binop_dtype_rule, _input_dtype, [_num, _num], 'dot')
dot_p = standard_primitive(dot_shape_rule, dot_dtype_rule, 'dot')
ad.defbilinear(dot_p, dot_transpose_lhs, dot_transpose_rhs)
batching.primitive_batchers[dot_p] = dot_batch_rule


def dot_general_shape_rule(lhs, rhs, dimension_numbers):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  if len(lhs_batch) != len(rhs_batch):
    msg = ("dot_general requires equal numbers of lhs_batch and rhs_batch "
           "dimensions, got lhs_batch {} and rhs_batch {}.")
    raise TypeError(msg.format(lhs_batch, rhs_batch))
  if not onp.all(onp.equal(lhs_batch, rhs_batch)):
    msg = ("dot_general requires same lhs and rhs batch dimension numbers, "
           "got {} and {}.")
    raise TypeError(msg.format(lhs_batch, rhs_batch))
  lhs_batch_shape = onp.take(lhs.shape, lhs_batch)
  rhs_batch_shape = onp.take(rhs.shape, rhs_batch)
  if not onp.all(onp.equal(lhs_batch_shape, rhs_batch_shape)):
    msg = ("dot_general requires lhs batch dimensions and rhs batch dimensions "
           "to have the same shape, got {} and {}.")
    raise TypeError(msg.format(lhs_batch_shape, rhs_batch_shape))
  if tuple(sorted(lhs_batch)) != tuple(range(len(lhs_batch))):
    msg = ("dot_general requires lhs batch dimensions to precede contracting "
           "and non-contracting dimensions, got lhs_batch {}.")
    raise TypeError(msg.format(lhs_batch))
  if tuple(sorted(rhs_batch)) != tuple(range(len(rhs_batch))):
    msg = ("dot_general requires rhs batch dimensions to precede contracting "
           "and non-contracting dimensions, got rhs_batch {}.")
    raise TypeError(msg.format(rhs_batch))
  if not len(lhs_contracting) == len(rhs_contracting) == 1:
    msg = ("dot_general accepts exactly one lhs_contracting and "
           "rhs_contracting dimension, got {} and {}.")
    raise TypeError(msg.format(lhs_contracting, rhs_contracting))
  lhs_contracting_shape = onp.take(lhs.shape, lhs_contracting)
  rhs_contracting_shape = onp.take(rhs.shape, rhs_contracting)
  if not onp.all(onp.equal(lhs_contracting_shape, rhs_contracting_shape)):
    msg = ("dot_general requires contracting dimensions to have the same "
           "shape, got {} and {}.")
    raise TypeError(msg.format(lhs_contracting_shape, rhs_contracting_shape))
  if lhs.ndim > len(lhs_batch) + len(lhs_contracting) + 1:
    msg = ("dot_general requires either one or zero non-batch non-contracting "
           "lhs dimension, got {}.")
    diff = lhs.ndim - len(lhs_batch) - len(lhs_contracting)
    raise TypeError(msg.format(diff))
  if rhs.ndim > len(rhs_batch) + len(rhs_contracting) + 1:
    msg = ("dot_general requires either one or zero non-batch non-contracting "
           "rhs dimension, got {}.")
    diff = rhs.ndim - len(rhs_batch) - len(rhs_contracting)
    raise TypeError(msg.format(diff))

  batch_shape = tuple(onp.take(lhs.shape, lhs_batch))
  lhs_contract_or_batch = tuple(lhs_contracting) + tuple(lhs_batch)
  lhs_tensored_shape = tuple(onp.delete(lhs.shape, lhs_contract_or_batch))
  rhs_contract_or_batch = tuple(rhs_contracting) + tuple(rhs_batch)
  rhs_tensored_shape = tuple(onp.delete(rhs.shape, rhs_contract_or_batch))
  return batch_shape + lhs_tensored_shape + rhs_tensored_shape


def dot_general_dtype_rule(lhs, rhs, dimension_numbers):
  return binop_dtype_rule(_input_dtype, [_num, _num], 'dot_general', lhs, rhs)


def dot_general_transpose_lhs(g, y, dimension_numbers, swap_ans=False):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim = g.ndim - y.ndim + len(x_batch) + 2 * len(x_contract)
  x_kept = remaining(range(x_ndim), x_contract, x_batch)
  y_kept = remaining(range(y.ndim), y_contract, y_batch)
  if swap_ans:
    ans_batch, ans_y, _ = ranges_like(x_batch, y_kept, x_kept)
  else:
    ans_batch, _, ans_y = ranges_like(x_batch, x_kept, y_kept)
  dims = ((ans_y, y_kept), (ans_batch, y_batch))
  x_contract_sorted_by_y = list(onp.take(x_contract, onp.argsort(y_contract)))
  out_axes = onp.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)
  return transpose(dot_general(g, y, dims), tuple(out_axes))

def dot_general_transpose_rhs(g, x, dimension_numbers):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
  return dot_general_transpose_lhs(g, x, swapped_dimension_numbers, True)


# def dot_general_batch_rule(batched_args, batch_dims, dimension_numbers):
#   assert False  # TODO

dot_general_p = standard_primitive(dot_general_shape_rule,
                                   dot_general_dtype_rule, 'dot_general')
ad.defbilinear(dot_general_p,
               dot_general_transpose_lhs, dot_general_transpose_rhs)
# batching.primitive_batchers[dot_general_p] = dot_general_batch_rule


def broadcast_shape_rule(operand, sizes):
  _check_shapelike('broadcast', 'sizes', sizes)
  return tuple(sizes) + operand.shape

def broadcast_batch_rule(batched_args, batch_dims, sizes):
  operand, = batched_args
  bdim, = batch_dims
  new_bdim = None if bdim is None else bdim + len(sizes)
  return broadcast(operand, sizes), new_bdim

broadcast_p = standard_primitive(
    broadcast_shape_rule, _input_dtype, 'broadcast')
ad.deflinear(broadcast_p, lambda t, sizes: [_reduce_sum(t, range(len(sizes)))])
batching.primitive_batchers[broadcast_p] = broadcast_batch_rule


def broadcast_in_dim_shape_rule(operand, shape, broadcast_dimensions):
  _check_shapelike('broadcast_in_dim', 'shape', shape)
  _check_shapelike('broadcast_in_dim', 'broadcast_dimensions',
                   broadcast_dimensions)
  if operand.ndim != len(broadcast_dimensions):
    msg = ('broadcast_in_dim broadcast_dimensions must have length equal to '
           'operand ndim, got broadcast_dimensions for operand ndim {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand.ndim))
  if not set(broadcast_dimensions).issubset(set(range(len(shape)))):
    msg = ('broadcast_in_dim broadcast_dimensions must be a subset of output '
           'dimensions, got {} for operand ndim {} and shape {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand.ndim, shape))
  return shape

def broadcast_in_dim_transpose_rule(t, shape, broadcast_dimensions):
  axes = tuple(onp.delete(range(len(shape)), broadcast_dimensions))
  return [_reduce_sum(t, axes)]

def broadcast_in_dim_batch_rule(batched_args, batch_dims, shape,
                                broadcast_dimensions):
  operand, = batched_args
  bdim, = batch_dims
  new_shape = list(shape)
  new_shape.insert(bdim, operand.shape[bdim])
  new_broadcast_dimensions = [d if d < bdim else d + 1 for d in broadcast_dimensions]
  new_broadcast_dimensions.insert(bdim, bdim)
  return broadcast_in_dim(operand, new_shape, new_broadcast_dimensions), bdim


broadcast_in_dim_p = standard_primitive(
    broadcast_in_dim_shape_rule, _input_dtype, 'broadcast_in_dim')
ad.deflinear(broadcast_in_dim_p, broadcast_in_dim_transpose_rule)
batching.primitive_batchers[broadcast_in_dim_p] = broadcast_in_dim_batch_rule


def clamp_shape_rule(min, operand, max):
  if min.shape and min.shape != operand.shape:
    m = "clamp requires min.shape == operand.shape or min.shape == (), got {}."
    raise TypeError(m.format(min.shape))
  if max.shape and max.shape != operand.shape:
    m = "clamp requires max.shape == operand.shape or max.shape == (), got {}."
    raise TypeError(m.format(max.shape))
  return operand.shape

clamp_dtype_rule = partial(binop_dtype_rule, _input_dtype, [_any, _any, _any],
                           'clamp')

clamp_p = standard_primitive(clamp_shape_rule, clamp_dtype_rule, 'clamp')
ad.defjvp(clamp_p,
          lambda g, min, operand, max:
          select(bitwise_and(gt(min, operand), lt(min, max)),
                 _brcast(g, operand), _zeros(operand)),
          lambda g, min, operand, max:
          select(bitwise_and(gt(operand, min), lt(operand, max)),
                 g, _zeros(operand)),
          lambda g, min, operand, max:
          select(lt(max, operand), _brcast(g, operand), _zeros(operand)))


def concatenate_shape_rule(*operands, **kwargs):
  dimension = kwargs.pop('dimension')
  if not operands:
    msg = "concatenate expects at least one operand, got 0."
    raise TypeError(msg)
  if not all(isinstance(operand, UnshapedArray) for operand in operands):
    msg = "All objects to concatenate must be arrays, got {}."
    op = next(op for op in operands if not isinstance(op, UnshapedArray))
    raise TypeError(msg.format(type(op)))
  if len(set(operand.ndim for operand in operands)) != 1:
    msg = "Cannot concatenate arrays with different ranks, got {}."
    raise TypeError(msg.format(", ".join(str(o.ndim) for o in operands)))
  shapes = onp.array([operand.shape for operand in operands])
  if not 0 <= dimension < shapes.shape[1]:
    msg = "concatenate dimension out of bounds: dimension {} for shapes {}."
    raise TypeError(msg.format(dimension, ", ".join(map(str, shapes))))
  if not onp.all(onp.delete(shapes[0] == shapes, dimension, axis=1)):
    msg = ("Cannot concatenate arrays with shapes that differ in dimensions "
           "other than the one being concatenated: dimension {} for shapes {}.")
    raise TypeError(msg.format(dimension, ", ".join(map(str, shapes))))

  concat_size = sum(o.shape[dimension] for o in operands)
  ex_shape = operands[0].shape
  return ex_shape[:dimension] + (concat_size,) + ex_shape[dimension+1:]

def concatenate_dtype_rule(*operands, **kwargs):
  _check_same_dtypes('concatenate', False, *(o.dtype for o in operands))
  return operands[0].dtype

def concatenate_translation_rule(c, *operands, **kwargs):
  dimension = kwargs.pop('dimension')
  return c.Concatenate(operands, dimension=dimension)

def concatenate_transpose_rule(t, *operands, **kwargs):
  dimension = kwargs.pop('dimension')
  operand_shapes = kwargs.pop('operand_shapes')
  limit_points = onp.cumsum([shape[dimension] for shape in operand_shapes])

  starts = onp.zeros((len(operands), t.ndim), dtype=int)
  starts[1:, dimension] = limit_points[:-1]
  limits = onp.tile(t.shape, (len(operands), 1))
  limits[:, dimension] = limit_points

  return [slice(t, start, limit) if o is None else None
          for o, start, limit in zip(operands, starts, limits)]

concatenate_p = standard_primitive(
    concatenate_shape_rule, concatenate_dtype_rule, 'concatenate',
    concatenate_translation_rule)
ad.deflinear(concatenate_p, concatenate_transpose_rule)
ad.primitive_transposes[concatenate_p] = concatenate_transpose_rule


def pad_shape_rule(operand, padding_value, padding_config):
  if operand.dtype != padding_value.dtype:
    msg = "pad operand and padding_value must be same dtype: got {} and {}."
    raise TypeError(msg.format(operand.dtype, padding_value.dtype))

  lo, hi, interior = zip(*padding_config)
  out_shape = onp.add(onp.add(onp.add(lo, hi), operand.shape),
                      onp.multiply(interior, onp.subtract(operand.shape, 1)))
  return tuple(out_shape)

def pad_transpose(t, operand, padding_value, padding_config):
  lo, hi, interior = zip(*padding_config)
  if onp.any(onp.less(lo, 0)) or onp.any(onp.less(hi, 0)):
    msg = "pad transpose not implemented for negative padding, got {}."
    raise NotImplementedError(msg.format(padding_config))

  total = lambda x: _reduce_sum(x, list(range(t.ndim)))

  t_op = lambda: slice(t, lo, onp.subtract(t.shape, hi), onp.add(interior, 1))
  t_operand = t_op() if operand is None else None

  if padding_value is None:
    t_operand = t_op() if t_operand is None else t_operand
    t_padv = sub(total(t), total(t_operand))
  else:
    t_padv = None

  return [t_operand, t_padv]

pad_p = standard_primitive(pad_shape_rule, _input_dtype, 'pad')
ad.deflinear(pad_p, pad_transpose)
ad.primitive_transposes[pad_p] = pad_transpose


def reshape_shape_rule(operand, new_sizes, dimensions, **unused_kwargs):
  if not onp.all(onp.greater_equal(new_sizes, 0)):
    msg = 'reshape new_sizes must all be positive, got {}.'
    raise TypeError(msg.format(new_sizes))
  if prod(onp.shape(operand)) != prod(new_sizes):
    msg = 'reshape total size must be unchanged, got new_sizes {} for shape {}.'
    raise TypeError(msg.format(new_sizes, onp.shape(operand)))
  if dimensions is not None:
    if set(dimensions) != set(range(onp.ndim(operand))):
      msg = ('reshape dimensions must be a permutation of operand dimensions, '
             'got dimensions {} for shape {}.')
      raise TypeError(msg.format(dimensions, onp.shape(operand)))
  return tuple(new_sizes)

def reshape_dtype_rule(operand, new_sizes, dimensions, **unused_kwargs):
  return operand.dtype

def reshape_translation_rule(c, operand, new_sizes, dimensions, old_sizes):
  del old_sizes  # Unused.
  return c.Reshape(operand, new_sizes=new_sizes, dimensions=dimensions)

def reshape_transpose_rule(t, new_sizes, dimensions, old_sizes):
  out = reshape(t, old_sizes)
  if dimensions is None:
    return [out]
  else:
    return [transpose(out, onp.argsort(dimensions))]

def reshape_batch_rule(batched_args, batch_dims, new_sizes, dimensions, **unused):
  operand, = batched_args
  bdim, = batch_dims
  operand = batching.move_dim_to_front(operand, bdim)
  if dimensions is not None:
    raise NotImplementedError  # TODO(mattjj): handle reshape w/ dimensions
    dimensions = (0,) + tuple(onp.add(1, dimensions))
  return reshape(operand, operand.shape[:1] + new_sizes, dimensions), 0

reshape_p = standard_primitive(reshape_shape_rule, reshape_dtype_rule,
                               'reshape', reshape_translation_rule)
ad.deflinear(reshape_p, reshape_transpose_rule)
batching.primitive_batchers[reshape_p] = reshape_batch_rule


def rev_shape_rule(operand, dimensions):
  _check_shapelike('rev', 'dimensions', dimensions)
  if len(set(dimensions)) != len(dimensions):
    msg = 'rev dimensions must be unique, got {}.'
    raise TypeError(msg.format(dimensions))
  if not _max(dimensions) < operand.ndim:
    msg = ('rev dimensions must all be less than operand ndim, got dimensions '
           '{} for operand ndim {}.')
    raise TypeError(msg.format(dimensions, operand.ndim))
  return operand.shape

rev_p = standard_primitive(rev_shape_rule, _input_dtype, 'rev')
ad.deflinear(rev_p, lambda t, dimensions: [rev(t, dimensions)])


def transpose_shape_rule(operand, permutation):
  if not isinstance(permutation, (tuple, list, onp.ndarray)):
    msg = "transpose permutation must be a tuple/list/ndarray, got {}."
    raise TypeError(msg.format(type(permutation)))
  if tuple(sorted(permutation)) != tuple(range(operand.ndim)):
    msg = ("transpose permutation isn't a permutation of operand dimensions, "
           "got permutation {} for operand shape {}.")
    raise TypeError(msg.format(permutation, operand.shape))
  return tuple(onp.take(operand.shape, permutation))

def transpose_batch_rule(batched_args, batch_dims, permutation):
  operand, = batched_args
  bdim, = batch_dims
  perm = tuple(onp.insert(onp.add(permutation, 1), bdim, 0))
  return transpose(operand, perm), 0

transpose_p = standard_primitive(transpose_shape_rule, _input_dtype,
                                 'transpose')
ad.deflinear(transpose_p,
             lambda t, permutation: [transpose(t, onp.argsort(permutation))])
batching.primitive_batchers[transpose_p] = transpose_batch_rule


def select_shape_rule(pred, on_true, on_false):
  if on_true.shape != on_false.shape:
    msg = "select on_true and on_false must have the same shape, got {} and {}."
    raise TypeError(msg.format(on_true.shape, on_false.shape))
  if pred.shape and pred.shape != on_true.shape:
    msg = ("select pred must be scalar or have the same shape as on_true and "
           "on_false, got pred shape {} for on_true and on_false of shape {}.")
    raise TypeError(msg.format(pred.shape, on_true.shape))
  return on_true.shape

def select_dtype_rule(pred, on_true, on_false):
  _check_same_dtypes("select", False, on_true.dtype, on_false.dtype)
  if not onp.issubdtype(pred.dtype, onp.bool_):
    msg = "select pred must be boolean type, got {}."
    raise TypeError(msg.format(pred.dtype))
  return on_true.dtype

def select_transpose_rule(t, pred, on_true, on_false):
  return [None,
          select(pred, t, _zeros(on_false)) if on_true is None else None,
          select(pred, _zeros(on_true), t) if on_false is None else None]

def select_batch_rule(batched_args, batch_dims, **unused_kwargs):
  oprand, on_true, on_false, = batched_args
  pred_bdim, ot_bdim, of_bdim = batch_dims

  if (ot_bdim not in {None, pred_bdim}) or (of_bdim not in {None, pred_bdim}):
    raise NotImplementedError  # TODO(schsam, mattjj): Handle more cases.

  # TODO(schsam, mattjj): Switch to using broadcast_in_dim.
  ot = _ones(oprand) * on_true
  of = _ones(oprand) * on_false

  return select(oprand, ot, of), pred_bdim

select_p = standard_primitive(select_shape_rule, select_dtype_rule, 'select')
ad.defjvp(select_p,
          None,
          lambda g, b, x, y: select(b, g, _zeros(g)),
          lambda g, b, x, y: select(b, _zeros(g), g))
ad.primitive_transposes[select_p] = select_transpose_rule
batching.primitive_batchers[select_p] = select_batch_rule

def slice_shape_rule(operand, start_indices, limit_indices, strides,
                     operand_shape):
  _check_shapelike("slice", "start_indices", start_indices)
  _check_shapelike("slice", "limit_indices", limit_indices)
  if operand.ndim != len(start_indices):
    msg = ("slice start_indices must have length equal to the number of "
           "dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(limit_indices):
    msg = ("slice limit_indices must have the same length as start_indices, "
           "got start_inidices {} and limit_indices {}.")
    raise TypeError(msg.format(start_indices, limit_indices))
  if not onp.all(onp.less_equal(limit_indices, operand.shape)):
    msg = ("slice limit_indices must be less than or equal to operand shape, "
           "got limit_indices {} for operand shape {}.")
    raise TypeError(msg.format(limit_indices, operand.shape))
  if not onp.all(onp.greater_equal(start_indices, 0)):
    msg = ("slice start_indices must be greater than or equal to zero, "
           "got start_indices of {}.")
    raise TypeError(msg.format(start_indices))
  if not onp.all(onp.greater_equal(limit_indices, start_indices)):
    msg = ("slice limit_indices must be greater than or equal to start_indices,"
           " got start_indices {} and limit_indices {}.")
    raise TypeError(msg.format(start_indices, limit_indices))
  if strides is None:
    strides = onp.ones(operand.ndim, onp.int32)
  else:
    _check_shapelike("slice", "strides", strides)
    if len(strides) != operand.ndim:
      msg = ("slice strides must have length equal to the number of dimensions "
             "of the operand, got strides {} for operand shape {}.")
      raise TypeError(msg.format(strides, operand.shape))
    if not onp.all(onp.greater(strides, 0)):
      msg = "slice strides must be positive, got {}"
      raise TypeError(msg.format(strides))

  result_shape = onp.floor_divide(
      onp.add(onp.subtract(limit_indices, start_indices), strides) - 1, strides)
  return tuple(result_shape)

def slice_translation_rule(c, operand, start_indices, limit_indices, strides,
                           operand_shape):
  return c.Slice(operand, start_indices, limit_indices, strides)

def slice_transpose_rule(t, start_indices, limit_indices, strides,
                         operand_shape):
  if strides is None or onp.all(onp.equal(strides, 1)):
    pads = zip(start_indices, onp.subtract(operand_shape, limit_indices),
               (0,) * len(start_indices))
  else:
    real_limits = onp.add(onp.add(start_indices, 1),
                          onp.multiply(onp.subtract(t.shape, 1), strides))
    pads = zip(start_indices, onp.subtract(operand_shape, real_limits),
               onp.subtract(strides, 1))
  result = pad(t, _const(t, 0), pads)
  assert result.shape == operand_shape
  return [result]

def slice_batching_rule(batched_args, batch_dims, start_indices, limit_indices,
                        strides, **unused_kwargs):
  operand, = batched_args
  bdim, = batch_dims

  new_start_indices = list(start_indices)
  new_start_indices.insert(bdim, 0)

  new_limit_indices = list(limit_indices)
  new_limit_indices.insert(bdim, operand.shape[bdim])

  if strides is None:
    new_strides = None
  else:
    new_strides = list(strides)
    new_strides.insert(bdim, 1)

  out = slice(operand, new_start_indices, new_limit_indices, new_strides)
  return out, bdim

slice_p = standard_primitive(slice_shape_rule, _input_dtype, 'slice',
                             slice_translation_rule)
ad.deflinear(slice_p, slice_transpose_rule)
batching.primitive_batchers[slice_p] = slice_batching_rule


def dynamic_slice_shape_rule(operand, start_indices, slice_sizes,
                             operand_shape):
  if operand.ndim != len(start_indices):
    msg = ("dynamic_slice start_indices must have length equal to the number "
           "of dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(slice_sizes):
    msg = ("dynamic_slice slice_sizes must have the same length as "
           "start_indices, got start_inidices length {} and slice_sizes {}.")
    raise TypeError(msg.format(len(start_indices), slice_sizes))
  if not onp.all(onp.less_equal(slice_sizes, operand.shape)):
    msg = ("slice slice_sizes must be less than or equal to operand shape, "
           "got slice_sizes {} for operand shape {}.")
    raise TypeError(msg.format(slice_sizes, operand.shape))
  if not onp.all(onp.greater_equal(slice_sizes, 0)):
    msg = ("slice slice_sizes must be greater than or equal to zero, "
           "got slice_sizes of {}.")
    raise TypeError(msg.format(slice_sizes))
  return tuple(slice_sizes)

def dynamic_slice_translation_rule(c, operand, start_indices, slice_sizes,
                                   operand_shape):
  return c.DynamicSlice(operand, start_indices, slice_sizes)

def dynamic_slice_jvp_rule(g, operand, start_indices, slice_sizes,
                           operand_shape):
  return dynamic_slice(g, start_indices, slice_sizes)

def dynamic_slice_transpose_rule(t, operand, start_indices, slice_sizes,
                                 operand_shape):
  assert operand is None
  zeros = broadcast(_const(t, 0), operand_shape)
  return [dynamic_update_slice(zeros, t, start_indices), ad_util.zero]

dynamic_slice_p = standard_primitive(
    dynamic_slice_shape_rule, _input_dtype, 'dynamic_slice',
    dynamic_slice_translation_rule)
ad.defjvp(dynamic_slice_p, dynamic_slice_jvp_rule, None)
ad.primitive_transposes[dynamic_slice_p] = dynamic_slice_transpose_rule


def dynamic_update_slice_shape_rule(operand, update, start_indices,
                                    update_shape):
  if operand.ndim != update.ndim:
    msg = ("dynamic_update_slice update must have the same rank as operand, "
           "got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  if operand.ndim != len(start_indices):
    msg = ("dynamic_update_slice start_indices must have length equal to the "
           "rank of operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if not onp.all(onp.less_equal(update.shape, operand.shape)):
    msg = ("dynamic_update_slice update shape must be smaller than operand "
           "shape, got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  return operand.shape

def dynamic_update_slice_dtype_rule(operand, update, start_indices,
                                    update_shape):
  _check_same_dtypes("dynamic_update_slice", False, operand.dtype, update.dtype)
  return operand.dtype

def dynamic_update_slice_jvp(primals, tangents, update_shape):
  operand, update, start_indices = primals
  g_operand, g_update, g_start_indices = tangents
  assert g_start_indices is ad_util.zero
  val_out = dynamic_update_slice(operand, update, start_indices)
  if g_operand is ad_util.zero and g_update is ad_util.zero:
    tangent_out = ad_util.zero
  else:
    g_operand = ad.instantiate_zeros(operand, g_operand)
    g_update = ad.instantiate_zeros(update, g_update)
    tangent_out = dynamic_update_slice(g_operand, g_update, start_indices)
  return val_out, tangent_out

def dynamic_update_slice_transpose_rule(t, operand, update, start_indices,
                                        update_shape):
  assert start_indices is not None
  dus = dynamic_update_slice
  ds = dynamic_slice
  zeros = _zeros(t, shape=update_shape)
  operand_t = dus(t, zeros, start_indices) if operand is None else None
  update_t = ds(t, start_indices, update_shape) if update is None else None
  return [operand_t, update_t, None]

def dynamic_update_slice_translation_rule(c, operand, update, start_indices,
                                          update_shape):
  return c.DynamicUpdateSlice(operand, update, start_indices)

dynamic_update_slice_p = standard_primitive(
    dynamic_update_slice_shape_rule, dynamic_update_slice_dtype_rule,
    'dynamic_update_slice', dynamic_update_slice_translation_rule)
ad.primitive_jvps[dynamic_update_slice_p] = dynamic_update_slice_jvp
ad.primitive_transposes[dynamic_update_slice_p] = \
    dynamic_update_slice_transpose_rule


def index_take_shape_rule(src, *idxs, **kwargs):
  axes = kwargs['axes']
  return (idxs[0].shape[0],) + tuple(onp.delete(src.shape, axes))

def index_take_translation_rule(c, src, *idxs, **kwargs):
  jaxpr = kwargs['jaxpr']
  consts = kwargs['consts']
  shapes = map(c.GetShape, (src,) + idxs)
  xla_computation = xla.jaxpr_computation(jaxpr, consts, (), *shapes)
  return c.Call(xla_computation, (src,) + idxs)

def index_take_jvp(primals, tangents, axes, input_shape, jaxpr, consts):
  src = primals[0]
  idxs = tuple(primals[1:])
  g = ad.instantiate_zeros(src, tangents[0])
  return index_take(src, idxs, axes), index_take(g, idxs, axes)

def index_take_transpose_rule(t, src, *idxs, **kwargs):
  assert src is None
  axes = kwargs['axes']
  input_shape = kwargs['input_shape']
  t_src = index_untake(t, _zeros(t, shape=input_shape), idxs, axes)
  return [t_src] + [None] * len(idxs)

index_take_p = standard_primitive(index_take_shape_rule, _input_dtype,
                                  'index_take', index_take_translation_rule)
ad.primitive_jvps[index_take_p] = index_take_jvp
ad.primitive_transposes[index_take_p] = index_take_transpose_rule


def index_untake_shape_rule(src, dst, *idxs, **kwargs):
  return dst.shape

def index_untake_translation_rule(c, src, dst, *idxs, **kwargs):
  jaxpr = kwargs['jaxpr']
  consts = kwargs['consts']
  shapes = map(c.GetShape, (src, dst) + idxs)
  xla_computation = xla.jaxpr_computation(jaxpr, consts, (), *shapes)
  return c.Call(xla_computation, (src, dst) + idxs)

def index_untake_jvp(primals, tangents, axes, jaxpr, consts):
  src, dst = primals[0], primals[1]
  idxs = tuple(primals[2:])
  g_src, g_dst = tangents[0], tangents[1]
  g_src = ad.instantiate_zeros(src, g_src)
  g_dst = ad.instantiate_zeros(dst, g_dst)
  val_out = index_untake(src, dst, idxs, axes)
  tangent_out = index_untake(g_src, g_dst, idxs, axes)
  return val_out, tangent_out

def index_untake_transpose_rule(t, src, dst, *idxs, **kwargs):
  axes = kwargs['axes']
  if src is None:
    t_src = index_take(t, idxs, axes)
  if dst is None:
    t_dst = t
  return [t_src, t_dst] + [None] * len(idxs)

index_untake_p = standard_primitive(
    index_untake_shape_rule, _input_dtype, 'index_untake',
    index_untake_translation_rule)
ad.primitive_jvps[index_untake_p] = index_untake_jvp
ad.primitive_transposes[index_untake_p] = index_untake_transpose_rule


def reduce_shape_rule(operand, init_value, jaxpr, consts, dimensions):
  return tuple(onp.delete(operand.shape, dimensions))

def reduce_translation_rule(c, operand, init_value, jaxpr, consts, dimensions):
  xla_computation = _reduction_computation(c, jaxpr, consts, init_value)
  return c.Reduce(operand, init_value, xla_computation, dimensions)

def _reduction_computation(c, jaxpr, consts, init_value):
  shape = c.GetShape(init_value)
  return xla.jaxpr_computation(jaxpr, consts, (), shape, shape)

reduce_p = standard_primitive(reduce_shape_rule, _input_dtype, 'reduce',
                              reduce_translation_rule)
batching.defreducer(reduce_p)


def reduce_sum_shape_rule(operand, axes, input_shape):
  assert operand.shape == input_shape, ('{} != {}'
                                        .format(operand.shape, input_shape))
  return tuple(onp.delete(operand.shape, axes))

def reduce_sum_translation_rule(c, operand, axes, input_shape):
  dtype = c.GetShape(operand).numpy_dtype()
  scalar = xla_bridge.Shape.array_shape(dtype, ())
  return c.Reduce(operand, c.Constant(onp.array(0, dtype)),
                  xla.primitive_computation(add_p, scalar, scalar),
                  axes)

def reduce_sum_transpose_rule(cotangent, input_shape, axes):
  broadcast_dimensions = tuple(onp.delete(onp.arange(len(input_shape)), axes))
  result = broadcast_in_dim(cotangent, input_shape, broadcast_dimensions)
  assert result.shape == input_shape
  return [result]

reduce_sum_p = standard_primitive(reduce_sum_shape_rule, _input_dtype,
                                  'reduce_sum', reduce_sum_translation_rule)
ad.deflinear(reduce_sum_p, reduce_sum_transpose_rule)
batching.defreducer(reduce_sum_p)


def reduce_chooser_shape_rule(operand, axes):
  return tuple(onp.delete(operand.shape, axes))

def reduce_chooser_translation_rule(prim, identity, c, operand, axes):
  dtype = c.GetShape(operand).numpy_dtype()
  scalar = xla_bridge.Shape.array_shape(dtype, ())
  return c.Reduce(operand, c.Constant(identity(dtype)),
                  xla.primitive_computation(prim, scalar, scalar), axes)

def reduce_chooser_jvp_rule(g, ans, operand, axes):
  # TODO(mattjj): an alternative is to use variadic reduce to compute the chosen
  # locations in a single pass (rather than comparing equality) and use a
  # gather, and/or even push along the chosen elements of g (b/112040122)
  shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
  location_indicators = convert_element_type(
      _eq_meet(operand, reshape(ans, shape)), g.dtype)
  counts = _reduce_sum(location_indicators, axes)
  return div(_reduce_sum(mul(g, location_indicators), axes), counts)

reduce_max_translation_rule = partial(reduce_chooser_translation_rule, max_p,
                                      _get_max_identity)
reduce_max_p = standard_primitive(reduce_chooser_shape_rule, _input_dtype,
                                  'reduce_max', reduce_max_translation_rule)
ad.defjvp2(reduce_max_p, reduce_chooser_jvp_rule)
batching.defreducer(reduce_max_p)


reduce_min_translation_rule = partial(
    reduce_chooser_translation_rule, min_p, _get_min_identity)
reduce_min_p = standard_primitive(reduce_chooser_shape_rule, _input_dtype,
                                  'reduce_min', reduce_min_translation_rule)
ad.defjvp2(reduce_min_p, reduce_chooser_jvp_rule)
batching.defreducer(reduce_min_p)


def reduce_window_shape_rule(operand, init_value, jaxpr, consts,
                             window_dimensions, window_strides, padding):
  if operand.dtype != init_value.dtype:
    msg = ("reduce_window got inconsistent dtypes for operand and init_value: "
           " got operand dtype {} and init_value dtype {}.")
    raise TypeError(msg.format(operand.dtype, init_value.dtype))
  return common_reduce_window_shape_rule(operand, window_dimensions,
                                         window_strides, padding)

def reduce_window_translation_rule(c, operand, init_value, jaxpr, consts,
                                   window_dimensions, window_strides, padding):
  xla_computation = _reduction_computation(c, jaxpr, consts, init_value)
  return c.ReduceWindow(operand, init_value, xla_computation, window_dimensions,
                        window_strides, padding)

reduce_window_p = standard_primitive(
    reduce_window_shape_rule, _input_dtype, 'reduce_window',
    reduce_window_translation_rule)


def reduce_window_sum_shape_rule(operand, window_dimensions, window_strides,
                                 padding, input_shape):
  return common_reduce_window_shape_rule(operand, window_dimensions,
                                         window_strides, padding)

def reduce_window_sum_translation_rule(c, operand, window_dimensions,
                                       window_strides, padding, input_shape):
  dtype = c.GetShape(operand).numpy_dtype()
  scalar = xla_bridge.Shape.array_shape(dtype, ())
  return c.ReduceWindow(operand, c.Constant(onp.array(0, dtype)),
                        xla.primitive_computation(add_p, scalar, scalar),
                        window_dimensions, window_strides, padding)

def reduce_window_sum_transpose_rule(cotangent, window_dimensions,
                                     window_strides, padding, input_shape):
  in_pads = padtype_to_pads(input_shape, window_dimensions, window_strides,
                            padding)
  ones = [1] * len(input_shape)
  pads = _conv_general_vjp_lhs_padding(
      input_shape, window_dimensions, window_strides, cotangent.shape, in_pads,
      ones, ones)
  padding_config = [(lo, hi, stride - 1)
                    for (lo, hi), stride in zip(pads, window_strides)]
  pad_cotangent = pad(cotangent, _zero(cotangent), padding_config)
  result = _reduce_window_sum(pad_cotangent, window_dimensions, ones,
                              xla_bridge.get_xla_client().PaddingType.VALID)
  assert result.shape == input_shape
  return [result]

reduce_window_sum_p = standard_primitive(
    reduce_window_sum_shape_rule, _input_dtype, 'reduce_window_sum',
    reduce_window_sum_translation_rule)
ad.deflinear(reduce_window_sum_p, reduce_window_sum_transpose_rule)


def reduce_window_chooser_translation_rule(
    prim, identity, c, operand, window_dimensions, window_strides, padding):
  dtype = c.GetShape(operand).numpy_dtype()
  scalar = xla_bridge.Shape.array_shape(dtype, ())
  return c.ReduceWindow(operand, c.Constant(identity(dtype)),
                        xla.primitive_computation(prim, scalar, scalar),
                        window_dimensions, window_strides, padding)

def reduce_window_chooser_jvp_rule(prim, g, operand, window_dimensions,
                                   window_strides, padding):
  assert prim is max_p or prim is min_p
  select_prim = ge_p if prim is max_p else le_p
  return _select_and_gather_add(g, operand, select_prim, window_dimensions,
                                window_strides, padding)


def common_reduce_window_shape_rule(operand, window_dimensions, window_strides,
                                    padding):
  _check_shapelike("reduce_window", "window_dimensions", window_dimensions)
  _check_shapelike("reduce_window", "window_strides", window_strides)
  if operand.ndim != len(window_dimensions):
    msg = ("reduce_window got the wrong number of window_dimensions for "
           "operand: got operand shape {} with window_dimensions {}.")
    raise TypeError(msg.format(operand.shape, window_dimensions))
  if len(window_strides) != len(window_dimensions):
    msg = ("reduce_window got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))

  return reduce_window_shape_tuple(operand.shape, window_dimensions,
                                   window_strides, padding)

def reduce_window_shape_tuple(operand_shape, window_dimensions, window_strides,
                              padding):
  pads = padtype_to_pads(operand_shape, window_dimensions, window_strides, padding)
  operand_padded = onp.add(operand_shape, onp.add(*zip(*pads)))
  t = onp.floor_divide(
      onp.subtract(operand_padded, window_dimensions), window_strides) + 1
  return tuple(t)


reduce_window_max_translation_rule = partial(
    reduce_window_chooser_translation_rule, max_p, _get_max_identity)
reduce_window_max_p = standard_primitive(
    common_reduce_window_shape_rule, _input_dtype, 'reduce_window_max',
    reduce_window_max_translation_rule)
ad.defjvp(reduce_window_max_p, partial(reduce_window_chooser_jvp_rule, max_p))


reduce_window_min_translation_rule = partial(
    reduce_window_chooser_translation_rule, min_p, _get_min_identity)
reduce_window_min_p = standard_primitive(
    common_reduce_window_shape_rule, _input_dtype, 'reduce_window_min',
    reduce_window_min_translation_rule)
ad.defjvp(reduce_window_min_p, partial(reduce_window_chooser_jvp_rule, min_p))


def select_and_scatter_shape_rule(
    operand, source, init_value, select_jaxpr, select_consts, scatter_jaxpr,
    scatter_consts, window_dimensions, window_strides, padding):
  _check_shapelike("select_and_scatter", "window_dimensions", window_dimensions)
  _check_shapelike("select_and_scatter", "window_strides", window_strides)
  if len(window_dimensions) != len(window_strides):
    msg = ("select_and_scatter got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))
  return operand.shape

def select_and_scatter_translation(operand, source, init_value, select_jaxpr,
                                   select_consts, scatter_jaxpr, scatter_consts,
                                   window_dimensions, window_strides, padding):
  select = _reduction_computation(c, select_jaxpr, select_consts, init_value)
  scatter = _reduction_computation(c, scatter_jaxpr, scatter_consts, init_value)
  return c.SelectAndScatter(operand, select, window_dimensions, window_strides,
                            padding, source, init_value, scatter)

select_and_scatter_p = standard_primitive(
    select_and_scatter_shape_rule, _input_dtype, 'select_and_scatter',
    select_and_scatter_translation)


def select_and_scatter_add_shape_rule(
    source, operand, select_prim, window_dimensions, window_strides, padding):
  return operand.shape

def select_and_scatter_add_translation(
    c, source, operand, select_prim, window_dimensions, window_strides,
    padding):
  dtype = c.GetShape(operand).numpy_dtype()
  scalar = xla_bridge.Shape.array_shape(dtype, ())
  select = xla.primitive_computation(select_prim, scalar, scalar)
  scatter = xla.primitive_computation(add_p, scalar, scalar)
  zero = c.Constant(onp.array(0, dtype))
  return c.SelectAndScatter(operand, select, window_dimensions, window_strides,
                            padding, source, zero, scatter)

def select_and_scatter_add_transpose(
    t, source, operand, select_prim, window_dimensions, window_strides,
    padding):
  assert source is None and operand is not None
  result = _select_and_gather_add(t, operand, select_prim, window_dimensions,
                                  window_strides, padding)
  return [result, None]

select_and_scatter_add_p = standard_primitive(
    select_and_scatter_add_shape_rule, _input_dtype, 'select_and_scatter_add',
    select_and_scatter_add_translation)
ad.primitive_transposes[select_and_scatter_add_p] = \
    select_and_scatter_add_transpose


def select_and_gather_add_shape_rule(
    tangents, operand, select_prim, window_dimensions, window_strides, padding):
  if tangents.shape != operand.shape:
    msg = ("select_and_gather_add tangents and operand shapes must match, "
           "got {} and {}.")
    raise TypeError(msg.format(tangents.shape, operand.shape))
  return common_reduce_window_shape_rule(operand, window_dimensions,
                                         window_strides, padding)

def select_and_gather_add_translation(
    c, tangents, operand, select_prim, window_dimensions, window_strides,
    padding):
  raise NotImplementedError("No efficient translation.")

def select_and_gather_add_transpose(
    t, tangents, operand, select_prim, window_dimensions, window_strides,
    padding):
  assert tangents is None and operand is not None
  result = _select_and_scatter_add(t, operand, select_prim, window_dimensions,
                                   window_strides, padding)
  return [result, None]

select_and_gather_add_p = standard_primitive(
    select_and_gather_add_shape_rule, _input_dtype, 'select_and_gather_add',
    select_and_gather_add_translation)
ad.primitive_transposes[select_and_gather_add_p] = \
    select_and_gather_add_transpose


sort_shape = lambda operand, dimension: operand.shape

def sort_jvp_rule(g, operand, dimension):
  _, g_out = sort_key_val(operand, g, dimension)
  return g_out

sort_p = standard_primitive(sort_shape, _input_dtype, 'sort')
ad.defjvp(sort_p, sort_jvp_rule)


def sort_key_val_abstract_eval(keys, values, dimension):
  return core.AbstractTuple((keys, values))

def sort_key_val_impl(keys, values, dimension):
  out = xla.apply_primitive(sort_key_val_p, keys, values, dimension=dimension)
  sorted_keys, sorted_values = out
  return core.pack((sorted_keys, sorted_values))

def sort_key_val_jvp(primals, tangents, dimension):
  # NOTE(mattjj): this re-sorts three times, but if we had a variadic
  # sort_key_val, or if we could apply a fixed permutation efficiently, we could
  # implement this jvp rule with a single sort. The apply_permutation primitive
  # would make the jvp (and corresponding transpose rule) faster and easier.
  # This would also be cleaner if we didn't get the sorted keys out.
  # TODO(mattjj): make sort_key_val variadic, no sorted keys out by default
  keys, values = primals
  keys_tangents, values_tangents = tangents

  val_out = sort_key_val(keys, values, dimension)

  if keys_tangents is ad_util.zero:
    keys_tangents_out = ad_util.zero
  else:
    keys_tangents_out = sort_jvp_rule(keys_tangents, keys, dimension)

  if values_tangents is ad_util.zero:
    values_tangents_out = ad_util.zero
  else:
    values_tangents_out = sort_jvp_rule(values_tangents, keys, dimension)

  tangents_out = keys_tangents_out, values_tangents_out
  return core.pack(val_out), core.pack(tangents_out)

def sort_key_val_transpose_rule(t, keys, values, dimension):
  t_keys, t_values = t
  assert t_keys is ad_util.zero
  broadcasted_iota = broadcast_in_dim(
      onp.arange(keys.shape[dimension]), keys.shape, [dimension % keys.ndim])
  _, perm = sort_key_val(keys, broadcasted_iota)
  keys_result = ad_util.zero if keys is None else None
  values_result = sort_key_val(perm, t_values)[1] if values is None else None
  return [keys_result, values_result]

sort_key_val_p = Primitive('sort_key_val')
sort_key_val_p.def_impl(sort_key_val_impl)
sort_key_val_p.def_abstract_eval(sort_key_val_abstract_eval)
xla.translations[sort_key_val_p] = partial(standard_translate, 'sort_key_val')
ad.primitive_jvps[sort_key_val_p] = sort_key_val_jvp
ad.primitive_transposes[sort_key_val_p] = sort_key_val_transpose_rule


def while_loop_abstract_eval(init_val, opaque_params):
  abs_out = opaque_params.val[0]
  return maybe_tracer_tuple_to_abstract_tuple(abs_out)

def while_loop_translation_rule(c, init_val, opaque_params):
  shape = c.GetShape(init_val)
  abs_out, cond_jaxpr, cond_consts, body_jaxpr, body_consts = opaque_params.val
  cond_computation = xla.jaxpr_computation(cond_jaxpr, cond_consts, (), shape)
  body_computation = xla.jaxpr_computation(body_jaxpr, body_consts, (), shape)
  return c.While(cond_computation, body_computation, init_val)

while_p = Primitive('while')
while_p.def_impl(partial(xla.apply_primitive, while_p))
while_p.def_abstract_eval(while_loop_abstract_eval)
xla.translations[while_p] = while_loop_translation_rule


### util


def _dilate_shape(shape, dilation):
  """Utility function for computing the shape resulting from a dilation."""
  if not onp.all(onp.greater(dilation, 0)):
    msg = "All dilations must be positive, got {}."
    raise TypeError(msg.format(dilation))
  dilation = (1,) * (len(shape) - len(dilation)) + tuple(dilation)
  return onp.multiply(dilation, onp.subtract(shape, 1)) + 1



def padtype_to_pads(in_shape, window_shape, window_strides, padding):
  """Convert padding string to list of pairs of pad values."""
  PaddingType = xla_bridge.get_xla_client().PaddingType

  if isinstance(padding, str):
    mapping = {'VALID': PaddingType.VALID, 'SAME': PaddingType.SAME}
    try:
      padding = mapping[padding.upper()]
    except KeyError:
      msg = "Unrecognized padding type: expected 'VALID' or 'SAME', got {}."
      raise RuntimeError(msg.format(padding))

  if padding == PaddingType.SAME:
    out_shape = onp.ceil(onp.true_divide(in_shape, window_strides)).astype(int)
    pad_sizes = [_max((out_size - 1) * stride + window_shape - in_size, 0)
                 for out_size, stride, window_shape, in_size
                 in zip(out_shape, window_strides, window_shape, in_shape)]
    return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
  elif padding == PaddingType.VALID:
    return [(0, 0)] * len(in_shape)
  else:
    msg = "Unknown padding type: {}."
    raise TypeError(msg.format(padding))


def _check_same_dtypes(name, ignore_fp_precision, *dtypes):
  """Check that dtypes agree, possibly ignoring float precision."""
  # the `ignore_fp_precision` flag exists because the XLA shape inference logic
  # allows mixed floating point precision, but the HLO verifier often rejects it
  dtypes = list(map(onp.dtype, dtypes))  # canonicalize
  if ignore_fp_precision:
    dtypes = [
        onp.floating if onp.issubdtype(dtype, onp.floating)
        else onp.complexfloating if onp.issubdtype(dtype, onp.complexfloating)
        else dtype for dtype in dtypes]
  if len({xla_bridge.canonicalize_dtype(t) for t in dtypes}) != 1:
    if ignore_fp_precision:
      msg = ("{} requires arguments to have same dtypes up to floating point "
             "precision, got {}.")
    else:
      msg = "{} requires arguments to have the same dtypes, got {}."
    raise TypeError(msg.format(name, ", ".join(map(str, dtypes))))


def _check_conv_shapes(name, lhs_shape, rhs_shape, window_strides):
  """Check that conv shapes are valid and are consistent with window_strides."""
  if len(lhs_shape) != len(rhs_shape):
    msg = "Arguments to {} must have same rank, got {} and {}."
    raise TypeError(msg.format(name, len(lhs_shape), len(rhs_shape)))
  if len(lhs_shape) < 2:
    msg = "Arguments to {} must have rank at least 2, got {} and {}."
    raise TypeError(msg.format(name, len(lhs_shape), len(rhs_shape)))
  if lhs_shape[1] != rhs_shape[1]:
    msg = "Arguments to {} must agree on input feature size, got {} and {}."
    raise TypeError(msg.format(name, lhs_shape[1], rhs_shape[1]))
  _check_shapelike(name, "window_strides", window_strides)
  if not onp.all(onp.greater(window_strides, 0)):
    msg = "All elements of window_strides must be positive, got {}."
    raise TypeError(msg.format(window_strides))
  if len(window_strides) != len(lhs_shape) - 2:
    msg = "{} window_strides has wrong length: expected {}, got {}."
    expected_length = len(lhs_shape) - 2
    raise TypeError(msg.format(name, expected_length, len(window_strides)))


def conv_shape_tuple(lhs_shape, rhs_shape, strides, pads):
  """Compute the shape tuple of a conv given input shapes in canonical order."""
  if isinstance(pads, str):
    pads = padtype_to_pads(lhs_shape[2:], rhs_shape[2:], strides, pads)
  if len(pads) != len(lhs_shape) - 2:
    msg = "Wrong number of explicit pads for convolution: expected {}, got {}."
    raise TypeError(msg.format(len(lhs_shape) - 2, len(pads)))

  lhs_padded = onp.add(lhs_shape[2:], onp.add(*zip(*pads)))
  out_space = onp.floor_divide(
      onp.subtract(lhs_padded, rhs_shape[2:]), strides) + 1
  out_space = onp.maximum(0, out_space)
  out_shape = (lhs_shape[0], rhs_shape[0]) + tuple(out_space)
  return tuple(out_shape)


def conv_general_shape_tuple(lhs_shape, rhs_shape, window_strides, padding,
                             dimension_numbers):
  lhs_perm, rhs_perm, out_perm = conv_general_permutations(dimension_numbers)
  lhs_trans = onp.take(lhs_shape, lhs_perm)
  rhs_trans = onp.take(rhs_shape, rhs_perm)
  out_trans = conv_shape_tuple(lhs_trans, rhs_trans, window_strides, padding)
  return tuple(onp.take(out_trans, onp.argsort(out_perm)))


def _check_shapelike(fun_name, arg_name, obj):
  """Check that `obj` is a shape-like value (e.g. tuple of nonnegative ints)."""
  if not isinstance(obj, (tuple, list, onp.ndarray)):
    msg = "{} {} must be of type tuple/list/ndarray, got {}."
    raise TypeError(msg.format(fun_name, arg_name, type(obj)))
  # bool(obj) for an ndarray raises an error, so we check len
  if not len(obj):  # pylint: disable=g-explicit-length-test
    return
  obj_arr = onp.array(obj)
  if obj_arr.ndim != 1:
    msg = "{} {} must be rank 1, got {}."
    raise TypeError(msg.format(obj_arr.ndim))
  if not onp.issubdtype(obj_arr.dtype, onp.integer):
    msg = "{} {} must have every element be an integer type, got {}."
    raise TypeError(msg.format(fun_name, arg_name, tuple(map(type, obj))))
  if not (obj_arr >= 0).all():
    msg = "{} {} must have every element be nonnegative, got {}."
    raise TypeError(msg.format(fun_name, arg_name, obj))


def conv_general_permutations(dimension_numbers):
  """Utility for convolution dimension permutations relative to Conv HLO."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  lhs_char, rhs_char, out_char = charpairs = ("N", "C"), ("O", "I"), ("N", "C")
  for i, (a, b) in enumerate(charpairs):
    if not dimension_numbers[i].count(a) == dimension_numbers[i].count(b) == 1:
      msg = ("convolution dimension_numbers[{}] must contain the characters "
             "'{}' and '{}' exatly once, got {}.")
      raise TypeError(msg.format(i, a, b, dimension_numbers[i]))
    if len(dimension_numbers[i]) != len(set(dimension_numbers[i])):
      msg = ("convolution dimension_numbers[{}] cannot have duplicate "
             "characters, got {}.")
      raise TypeError(msg.format(i, dimension_numbers[i]))
  if not (set(lhs_spec) - set(lhs_char) == set(rhs_spec) - set(rhs_char) ==
          set(out_spec) - set(out_char)):
    msg = ("convolution dimension_numbers elements must each have the same "
           "set of spatial characters, got {}.")
    raise TypeError(msg.format(dimension_numbers))

  def getperm(spec, charpair):
    spatial = (i for i, c in enumerate(spec) if c not in charpair)
    if spec is not rhs_spec:
      spatial = sorted(spatial, key=lambda i: rhs_spec.index(spec[i]))
    return (spec.index(charpair[0]), spec.index(charpair[1])) + tuple(spatial)

  lhs_perm, rhs_perm, out_perm = map(getperm, dimension_numbers, charpairs)
  return lhs_perm, rhs_perm, out_perm


def _dynamic_slice_indices(operand, start_indices):
  if isinstance(start_indices, (tuple, list)):
    start_indices = concatenate([reshape(i, [1]) for i in start_indices], 0)
  return rem(start_indices, onp.array(operand.shape, start_indices.dtype))


_const = lambda example, val: onp.array(val, _dtype(example))
_zeros = partial(full_like, fill_value=0)
_zero = partial(full_like, shape=(), fill_value=0)
_ones = partial(full_like, fill_value=1)
_one = partial(full_like, shape=(), fill_value=1)
_twos = partial(full_like, fill_value=2)
_two = partial(full_like, shape=(), fill_value=2)

_dtype = onp.result_type
_iscomplex = lambda x: onp.issubdtype(_dtype(x), onp.complexfloating)


def ranges_like(*xs):
  start = 0
  for x in xs:
    x_len = len(x)
    yield range(start, start + x_len)
    start += x_len


def remaining(original, *removed_lists):
  blacklist = set(itertools.chain(*removed_lists))
  return [i for i in original if i not in blacklist]


def _charswap(a, b, s):
  return s.translate(maketrans(a + b, b + a))


def _get_sdims(dimension_numbers):
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  rhs_sdims = [i for i, c in enumerate(rhs_spec) if c not in {"I", "O"}]
  lhs_sdims = sorted((i for i, c in enumerate(lhs_spec) if c not in {"N", "C"}),
                     key=lambda i: rhs_spec.index(lhs_spec[i]))
  out_sdims = sorted((i for i, c in enumerate(out_spec) if c not in {"N", "C"}),
                     key=lambda i: rhs_spec.index(out_spec[i]))
  return lhs_sdims, rhs_sdims, out_sdims


ConvolutionDimensionNumbers = collections.namedtuple(
    "ConvolutionDimensionNumbers", ["lhs_spec", "rhs_spec", "out_spec"])

def _conv_general_proto(dimension_numbers):
  assert type(dimension_numbers) is ConvolutionDimensionNumbers
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  proto = xla_bridge.xla_data_pb2.ConvolutionDimensionNumbers()
  proto.input_batch_dimension = lhs_spec[0]
  proto.input_feature_dimension = lhs_spec[1]
  proto.output_batch_dimension = out_spec[0]
  proto.output_feature_dimension = out_spec[1]
  proto.kernel_output_feature_dimension = rhs_spec[0]
  proto.kernel_input_feature_dimension = rhs_spec[1]
  proto.input_spatial_dimensions.extend(lhs_spec[2:])
  proto.kernel_spatial_dimensions.extend(rhs_spec[2:])
  proto.output_spatial_dimensions.extend(out_spec[2:])
  return proto


def _conv_general_vjp_lhs_padding(
    in_shape, window_dimensions, window_strides, out_shape, padding,
    lhs_dilation, rhs_dilation):
  lhs_dilated_shape = _dilate_shape(in_shape, lhs_dilation)
  out_dilated_shape = _dilate_shape(out_shape, window_strides)
  pad_before = onp.subtract(window_dimensions, [lo for lo, _ in padding]) - 1
  pad_after = (onp.add(lhs_dilated_shape, window_dimensions) - 1
               - out_dilated_shape - pad_before)
  return zip(pad_before, pad_after)


def _conv_general_vjp_rhs_padding(
    in_shape, window_dimensions, window_strides, out_shape, padding,
    lhs_dilation, rhs_dilation):
  lhs_dilated_shape = _dilate_shape(in_shape, lhs_dilation)
  rhs_dilated_shape = _dilate_shape(window_dimensions, rhs_dilation)
  out_dilated_shape = _dilate_shape(out_shape, window_strides)
  total_in_pad = out_dilated_shape + rhs_dilated_shape - lhs_dilated_shape - 1
  return [(pad[0], tot - pad[0]) for pad, tot in zip(padding, total_in_pad)]


def _balanced_eq(x, z, y):
  return div(select(_eq_meet(x, z), _ones(z), _zeros(z)),
             select(_eq_meet(y, z), _twos(z), _ones(z)))


def _eq_meet(a, b):
  a_dtype, b_dtype = _dtype(a), _dtype(b)
  if a_dtype != b_dtype:
    higher_dtype = onp.promote_types(a_dtype, b_dtype)
    if higher_dtype == a_dtype:
      a = convert_element_type(a, b_dtype)
    else:
      b = convert_element_type(b, a_dtype)
  return eq(a, b)


def maybe_tracer_tuple_to_abstract_tuple(tup):
  if isinstance(tup, pe.JaxprTracerTuple):
    return core.AbstractTuple(list(map(maybe_tracer_tuple_to_abstract_tuple, tup)))
  elif isinstance(tup, core.AbstractValue):
    return tup
  elif tup is None:
    return core.AbstractTuple(())  # TODO(dougalm): check this
  else:
    raise TypeError(tup)


def subvals(lst, replace):
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)


def _abstractify(x):
  # abstractify wrapper used internally for primitives like _while_loop
  if isinstance(x, core.Tracer):
    # TODO(mattjj,dougalm): check that it's at least ShapedArray
    return pe.PartialVal((x.aval, core.unit))
  else:
    return pe.PartialVal((xla.abstractify(x), core.unit))
