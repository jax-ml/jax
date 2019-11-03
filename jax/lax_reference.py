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
import itertools

import numpy as onp
import opt_einsum
import scipy.special

from six.moves import builtins

_slice = builtins.slice
_max = builtins.max
_min = builtins.min
_map = builtins.map

neg = onp.negative
sign = onp.sign
floor = onp.floor
ceil = onp.ceil
round = onp.round

is_finite = onp.isfinite

exp = onp.exp
expm1 = onp.expm1
log = onp.log
log1p = onp.log1p
tanh = onp.tanh
sin = onp.sin
cos = onp.cos
atan2 = onp.arctan2

sqrt = onp.sqrt
rsqrt = lambda x: 1. / onp.sqrt(x)
square = onp.square
reciprocal = onp.reciprocal
tan = onp.tan
asin = onp.arcsin
acos = onp.arccos
atan = onp.arctan
sinh = onp.sinh
cosh = onp.cosh

lgamma = scipy.special.gammaln
digamma = scipy.special.digamma
erf = scipy.special.erf
erfc = scipy.special.erfc
erf_inv = scipy.special.erfinv
bessel_i0e = scipy.special.i0e
bessel_i1e = scipy.special.i1e

real = onp.real
imag = onp.imag

def conj(x):
  return onp.conj(x) + onp.complex64(0)

def complex(x, y):
  return x + onp.complex64(1j) * y

abs = onp.absolute
pow = onp.power

bitwise_not = onp.bitwise_not
bitwise_and = onp.bitwise_and
bitwise_or = onp.bitwise_or
bitwise_xor = onp.bitwise_xor

add = onp.add
sub = onp.subtract
mul = onp.multiply

def div(lhs, rhs):
  if onp.issubdtype(onp.result_type(lhs), onp.integer):
    quotient = onp.floor_divide(lhs, rhs)
    select = onp.logical_and(onp.sign(lhs) != onp.sign(rhs), onp.remainder(lhs, rhs) != 0)
    return onp.where(select, quotient + 1, quotient)
  else:
    return onp.divide(lhs, rhs)

def rem(lhs, rhs):
  return onp.sign(lhs) * onp.remainder(onp.abs(lhs), onp.abs(rhs))

max = onp.maximum
min = onp.minimum

shift_left = onp.left_shift
shift_right_arithmetic = onp.right_shift
# TODO shift_right_logical

eq = onp.equal
ne = onp.not_equal
ge = onp.greater_equal
gt = onp.greater
le = onp.less_equal
lt = onp.less

def convert_element_type(operand, dtype):
  return onp.asarray(operand, dtype=dtype)

def bitcast_convert_type(operand, dtype):
  return onp.asarray(operand).view(dtype)

def clamp(min, operand, max):
  return onp.clip(operand, onp.clip(min, None, max), max)

def concatenate(operands, dimension):
  return onp.concatenate(operands, axis=dimension)

def conv(lhs, rhs, window_strides, padding):
  pads = padtype_to_pads(lhs.shape[2:], rhs.shape[2:], window_strides, padding)
  return _conv(lhs, rhs, window_strides, pads)

def conv_with_general_padding(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation):
  return _conv(_dilate(lhs, lhs_dilation), _dilate(rhs, rhs_dilation), window_strides, padding)

def conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
                         dimension_numbers):
  lhs_perm, rhs_perm, out_perm = _conv_general_permutations(dimension_numbers)
  if isinstance(padding, str):
    padding = padtype_to_pads(
        onp.take(lhs.shape, lhs_perm)[2:],
        onp.take(rhs.shape, rhs_perm)[2:], window_strides, padding)
  trans_lhs = transpose(lhs, lhs_perm)
  trans_rhs = transpose(rhs, rhs_perm)
  out = conv_with_general_padding(trans_lhs, trans_rhs, window_strides, padding, lhs_dilation,
                                  rhs_dilation)
  return transpose(out, onp.argsort(out_perm))

dot = onp.dot

def dot_general(lhs, rhs, dimension_numbers):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  new_id = itertools.count()
  lhs_axis_ids = [next(new_id) for _ in lhs.shape]
  rhs_axis_ids = [next(new_id) for _ in rhs.shape]
  lhs_out_axis_ids = lhs_axis_ids[:]
  rhs_out_axis_ids = rhs_axis_ids[:]

  for lhs_axis, rhs_axis in zip(lhs_contracting, rhs_contracting):
    shared_id = next(new_id)
    lhs_axis_ids[lhs_axis] = shared_id
    rhs_axis_ids[rhs_axis] = shared_id
    lhs_out_axis_ids[lhs_axis] = None
    rhs_out_axis_ids[rhs_axis] = None

  batch_ids = []
  for lhs_axis, rhs_axis in zip(lhs_batch, rhs_batch):
    shared_id = next(new_id)
    lhs_axis_ids[lhs_axis] = shared_id
    rhs_axis_ids[rhs_axis] = shared_id
    lhs_out_axis_ids[lhs_axis] = None
    rhs_out_axis_ids[rhs_axis] = None
    batch_ids.append(shared_id)

  not_none = lambda x: x is not None
  out_axis_ids = filter(not_none, batch_ids + lhs_out_axis_ids + rhs_out_axis_ids)
  return onp.einsum(lhs, lhs_axis_ids, rhs, rhs_axis_ids, out_axis_ids)

def broadcast(operand, sizes):
  return onp.broadcast_to(operand, sizes + onp.shape(operand))

def broadcast_in_dim(operand, shape, broadcast_dimensions):
  inshape = tuple(1 if i not in broadcast_dimensions else d for i, d in enumerate(shape))
  return onp.broadcast_to(onp.reshape(operand, inshape), shape)

sum = onp.sum

def reshape(operand, new_sizes, dimensions=None):
  if dimensions is None:
    dimensions = range(len(onp.shape(operand)))
  return onp.reshape(onp.transpose(operand, dimensions), new_sizes)

def pad(operand, padding_value, padding_config):
  lo, hi, interior = zip(*padding_config)
  outshape = onp.add(
      onp.add(onp.add(lo, hi), operand.shape), onp.multiply(interior,
                                                            onp.subtract(operand.shape, 1)))
  out = onp.full(outshape, padding_value, operand.dtype)
  lhs_slices = tuple(
      _slice(l if l > 0 else 0, -h if h > 0 else None, step)
      for l, h, step in zip(lo, hi, onp.add(1, interior)))
  rhs_slices = tuple(_slice(l if l < 0 else 0, -h if h < 0 else None) for l, h in zip(lo, hi))
  out[lhs_slices] = operand[rhs_slices]
  return out

def rev(operand, dimensions):
  dimensions = frozenset(dimensions)
  indexer = (
      _slice(None, None, -1) if d in dimensions else _slice(None) for d in range(onp.ndim(operand)))
  return operand[tuple(indexer)]

select = onp.where

def slice(operand, start_indices, limit_indices, strides=None):  # pylint: disable=redefined-builtin
  if strides is None:
    strides = onp.ones(len(start_indices)).astype(int)
  slices = tuple(_map(_slice, start_indices, limit_indices, strides))
  return operand[slices]

def dynamic_slice(operand, start_indices, slice_sizes):
  out = onp.zeros(slice_sizes, dtype=operand.dtype)
  idx = tuple(_slice(start, start + size) for start, size in zip(start_indices, slice_sizes))
  section = operand[idx]
  out[tuple(_slice(None, stop) for stop in section.shape)] = section
  return out

def dynamic_update_slice(operand, update, start_indices):
  slices = tuple(_map(_slice, start_indices, onp.add(start_indices, update.shape)))
  updated_operand = onp.copy(operand)
  updated_operand[slices] = update
  return updated_operand

transpose = onp.transpose

def reduce(operand, init_value, computation, dimensions):  # pylint: disable=redefined-builtin
  reducer = _make_reducer(computation, init_value)
  return reducer(operand, tuple(dimensions)).astype(onp.asarray(operand).dtype)

def reduce_window(operand, init_value, computation, window_dimensions, window_strides, padding):
  op, dims, strides = operand, window_dimensions, window_strides
  pads = padtype_to_pads(op.shape, dims, strides, padding)
  view = _conv_view(
      op.reshape((1, 1) + op.shape), (1, 1) + dims, strides, pads, pad_value=init_value)[0]
  view = view.reshape(view.shape[1:1 + len(dims)] + (-1,))
  reducer = _make_reducer(computation, init_value)
  return reducer(view, axis=-1)

# TODO(mattjj): select_and_scatter

sort = onp.sort

def sort_key_val(keys, values, dimension=-1):
  idxs = list(onp.ix_(*[onp.arange(d) for d in keys.shape]))
  idxs[dimension] = onp.argsort(keys, axis=dimension)
  return keys[idxs], values[idxs]

# TODO untake

### conv util

def _conv(lhs, rhs, window_strides, pads):
  view, view_axes, rhs_axes, out_axes = _conv_view(lhs, rhs.shape, window_strides, pads, 0.)
  return opt_einsum.contract(view, view_axes, rhs, rhs_axes, out_axes, use_blas=True)

def padtype_to_pads(in_shape, filter_shape, window_strides, padding):
  if padding.upper() == 'SAME':
    out_shape = onp.ceil(onp.true_divide(in_shape, window_strides)).astype(int)
    pad_sizes = [
        _max((out_size - 1) * stride + filter_size - in_size, 0) for out_size, stride, filter_size,
        in_size in zip(out_shape, window_strides, filter_shape, in_shape)
    ]
    return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
  else:
    return [(0, 0)] * len(in_shape)

def _conv_view(lhs, rhs_shape, window_strides, pads, pad_value):
  """Compute the view (and its axes) of a convolution or window reduction."""
  if (_min(lhs.ndim, len(rhs_shape)) < 2 or lhs.ndim != len(rhs_shape)
      or lhs.shape[1] != rhs_shape[1]):
    raise ValueError('Dimension mismatch')
  if len(window_strides) != len(rhs_shape) - 2:
    raise ValueError('Wrong number of strides for spatial dimensions')
  if len(pads) != len(rhs_shape) - 2:
    raise ValueError('Wrong number of pads for spatial dimensions')

  lhs = _pad(lhs, [(0, 0)] * 2 + list(pads), pad_value)
  in_shape = lhs.shape[2:]
  filter_shape = rhs_shape[2:]
  dim = len(filter_shape)  # number of 'spatial' dimensions in convolution

  out_strides = onp.multiply(window_strides, lhs.strides[2:])
  view_strides = lhs.strides[:1] + tuple(out_strides) + lhs.strides[1:]

  out_shape = onp.floor_divide(onp.subtract(in_shape, filter_shape), window_strides) + 1
  view_shape = lhs.shape[:1] + tuple(out_shape) + rhs_shape[1:]

  view = onp.lib.stride_tricks.as_strided(lhs, view_shape, view_strides)

  view_axes = list(range(view.ndim))
  sum_axes = view_axes[-dim - 1:]
  rhs_axes = [view.ndim] + sum_axes
  out_axes = [0, view.ndim] + list(range(1, dim + 1))

  return view, view_axes, rhs_axes, out_axes

def _pad(arr, pads, pad_value):
  out = onp.pad(arr, onp.maximum(0, pads), mode='constant',
                constant_values=pad_value).astype(arr.dtype)
  slices = tuple(
      _slice(abs(lo) if lo < 0 else 0, hi % dim if hi < 0 else None)
      for (lo, hi), dim in zip(pads, onp.shape(arr)))
  return out[slices]

def _dilate(operand, factors):
  # this logic is like lax.pad, but with two leading dimensions, no edge
  # padding, and factors are at least 1 (interior padding is at least 0)
  outspace = onp.add(operand.shape[2:],
                     onp.multiply(onp.subtract(factors, 1), onp.subtract(operand.shape[2:], 1)))
  out = onp.zeros(operand.shape[:2] + tuple(outspace), operand.dtype)
  lhs_slices = tuple(_slice(None, None, step) for step in factors)
  out[(_slice(None),) * 2 + lhs_slices] = operand
  return out

def _conv_general_permutations(dimension_numbers):
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  rhs_perm = ((rhs_spec.index('O'), rhs_spec.index('I')) +
              tuple(i for i, c in enumerate(rhs_spec) if c not in {'O', 'I'}))
  lhs_perm = ((lhs_spec.index('N'), lhs_spec.index('C')) + tuple(
      sorted((i for i, c in enumerate(lhs_spec) if c not in {'N', 'C'}),
             key=lambda i: rhs_spec.index(lhs_spec[i]))))
  out_perm = ((out_spec.index('N'), out_spec.index('C')) + tuple(
      sorted((i for i, c in enumerate(out_spec) if c not in {'N', 'C'}),
             key=lambda i: rhs_spec.index(out_spec[i]))))
  return lhs_perm, rhs_perm, out_perm

### reduce util

def _make_reducer(py_binop, init_val):
  """Make a reducer function given a Python binop and an initial value."""
  # It's tempting to use onp.ufunc.reduce (even with a ufunc generated by
  # onp.frompyfunc(py_binop)), but this may not agree with custom init_val.
  # We make an attempt to uncover an underlying numpy ufunc (which might be
  # wrapped by autograd or lax) and check its identity against init_val.
  monoid_record = _monoids.get(getattr(py_binop, '__name__'))
  if monoid_record:
    reducer, monoid_identity = monoid_record
    if init_val == monoid_identity(onp.result_type(init_val)):
      return reducer
  return _reducer_from_pyfunc(py_binop, init_val)

def _get_max_identity(dt):
  return -onp.inf if onp.issubdtype(dt, onp.floating) else onp.iinfo(dt).min

def _get_min_identity(dt):
  return onp.inf if onp.issubdtype(dt, onp.floating) else onp.iinfo(dt).max

def _identity_getter(op):
  return lambda dtype: onp.asarray(op.identity, dtype=dtype)

MonoidRecord = collections.namedtuple('MonoidRecord', ['reducer', 'identity'])
_monoids = {
    'max': MonoidRecord(onp.maximum.reduce, _get_max_identity),
    'min': MonoidRecord(onp.minimum.reduce, _get_min_identity),
    'add': MonoidRecord(onp.add.reduce, _identity_getter(onp.add)),
    'mul': MonoidRecord(onp.multiply.reduce, _identity_getter(onp.multiply)),
    'multiply': MonoidRecord(onp.multiply.reduce, _identity_getter(onp.multiply)),
    'logical_and': MonoidRecord(onp.logical_and.reduce, _identity_getter(onp.logical_and)),
    'logical_or': MonoidRecord(onp.logical_or.reduce, _identity_getter(onp.logical_or)),
}

def _reducer_from_pyfunc(py_binop, init_val):
  def reducer(operand, axis=0):
    axis = range(onp.ndim(operand)) if axis is None else axis
    result = onp.full(
        onp.delete(onp.shape(operand), axis), init_val, dtype=onp.asarray(operand).dtype)
    for idx, _ in onp.ndenumerate(operand):
      out_idx = tuple(onp.delete(idx, axis))
      result[out_idx] = py_binop(result[out_idx], operand[idx])
    return result

  return reducer
