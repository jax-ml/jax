# Copyright 2020 Google LLC
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
from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Dict, Callable, Union, Sequence, Tuple, List
import builtins
import itertools
import numpy as np
from numpy import ndarray
import opt_einsum
import scipy.special

from .. import core, config, dtypes, xla, util, ad_util, random, partial, curry
from ..core import Primitive, Trace, Jaxpr, TypedJaxpr
from ..lax import lax

_slice = builtins.slice
_min = builtins.min
_map = builtins.map

map = util.safe_map
zip = util.safe_zip

Array = Union[ndarray, xla.DeviceArray]

np_impl: Dict[Primitive, Callable[..., Any]] = {}

class NumpyEvalTrace(Trace):
  def pure(self, x): return x
  lift = sublift = pure

  def process_primitive(self, primitive, tracers, params):
    impl = np_impl.get(primitive)
    if impl is None:
      raise NotImplementedError("NumPy backend does not yet support the "
                                f"'{primitive}' primitive.")
    return impl(*tracers, **params)

  def process_call(self, primitive, f, tracers, params):
    return f.call_wrapped(*tracers)
  process_map = process_call

@contextmanager
def numpy_eval():
  """
  Makes JAX code evaluate with NumPy instead of XLA.

  >>> @numpy_eval()
  ... def some_fun(...):
  ...  # jax code

  Inline:

  >>> from jax.interpreters.numpy_eval import numpy_eval
  ... with numpy_eval():
  ...   # jax code

  ``jit(numpy_eval()(some_fun))`` will collapse constant calculations using NumPy,
  so that no operations are performed on the XLA device during compilation.

  `numpy_eval` is thread-safe.
  """
  assert config.omnistaging_enabled, "The NumPy backend requires omnistaging."
  with core.new_base_main(NumpyEvalTrace):
    yield

def _ensure_numpy(a: Array) -> ndarray:
  return np.asarray(a) if isinstance(a, xla.DeviceArray) else a

neg = np.negative
np_impl[lax.neg_p] = neg

sign = np.sign
np_impl[lax.sign_p] = sign

floor = np.floor
np_impl[lax.floor_p] = floor

ceil = np.ceil
np_impl[lax.ceil_p] = ceil

def round(x):
  return np.trunc(
    x + np.copysign(np.nextafter(np.array(.5, dtype=x.dtype),
                                 np.array(0., dtype=x.dtype),
                                 dtype=x.dtype), x)).astype(x.dtype)
np_impl[lax.round_p] = round

nextafter = np.nextafter
np_impl[lax.nextafter_p] = nextafter

is_finite = np.isfinite
np_impl[lax.is_finite_p] = is_finite

exp = np.exp
np_impl[lax.exp_p] = exp

expm1 = np.expm1
np_impl[lax.expm1_p] = expm1

log = np.log
np_impl[lax.log_p] = log

log1p = np.log1p
np_impl[lax.log1p_p] = log1p

tanh = np.tanh
np_impl[lax.tanh_p] = tanh

sin = np.sin
np_impl[lax.sin_p] = sin

cos = np.cos
np_impl[lax.cos_p] = cos

def atan2(x, x2): return np.arctan2(x, x2).astype(x.dtype)
np_impl[lax.atan2_p] = atan2

sqrt = np.sqrt
np_impl[lax.sqrt_p] = sqrt

def rsqrt(x): return np.ones_like(x) / np.sqrt(x)
np_impl[lax.rsqrt_p] = rsqrt

sinh = np.sinh
np_impl[lax.sinh_p] = sinh

cosh = np.cosh
np_impl[lax.cosh_p] = cosh

asinh = np.arcsinh
np_impl[lax.asinh_p] = asinh

acosh = np.arccosh
np_impl[lax.acosh_p] = acosh

atanh = np.arctanh
np_impl[lax.atanh_p] = atanh

def betainc(a, b, x): return scipy.special.betainc(a, b, x).astype(x.dtype)
np_impl[lax.regularized_incomplete_beta_p] = betainc

def lgamma(x): return scipy.special.gammaln(x).astype(x.dtype)
np_impl[lax.lgamma_p] = lgamma

def digamma(x): return scipy.special.digamma(x).astype(x.dtype)
np_impl[lax.digamma_p] = digamma

igamma = scipy.special.gammainc
np_impl[lax.igamma_p] = igamma

igammac = scipy.special.gammaincc
np_impl[lax.igammac_p] = igammac

# TODO lax.igamma_grad_a_p

def erf(x): return scipy.special.erf(x).astype(x.dtype)
np_impl[lax.erf_p] = erf

def erfc(x): return scipy.special.erfc(x).astype(x.dtype)
np_impl[lax.erfc_p] = erfc

def erf_inv(x): return scipy.special.erfinv(x).astype(x.dtype)
np_impl[lax.erf_inv_p] = erf_inv

def bessel_i0e(x): return scipy.special.i0e(x).astype(x.dtype)
np_impl[lax.bessel_i0e_p] = bessel_i0e

def bessel_i1e(x): return scipy.special.i1e(x).astype(x.dtype)
np_impl[lax.bessel_i1e_p] = bessel_i1e

real = np.real
np_impl[lax.real_p] = real

imag = np.imag
np_impl[lax.imag_p] = imag

def conj(x, input_dtype=None): return np.conj(x) + np.complex64(0)
np_impl[lax.conj_p] = conj

def complex(x, y): return x + np.complex64(1j) * y
np_impl[lax.complex_p] = complex

abs = np.absolute
np_impl[lax.abs_p] = abs

pow = np.power
np_impl[lax.pow_p] = pow

def integer_pow(x, y): return np.asarray(pow(x, y)).astype(x.dtype)
np_impl[lax.integer_pow_p] = integer_pow

bitwise_not = np.bitwise_not
np_impl[lax.not_p] = bitwise_not

bitwise_and = np.bitwise_and
np_impl[lax.and_p] = bitwise_and

bitwise_or = np.bitwise_or
np_impl[lax.or_p] = bitwise_or

bitwise_xor = np.bitwise_xor
np_impl[lax.xor_p] = bitwise_xor

add = np.add
np_impl[lax.add_p] = add

sub = np.subtract
np_impl[lax.sub_p] = sub

mul = np.multiply
np_impl[lax.mul_p] = mul

def div(lhs, rhs):
  if dtypes.issubdtype(dtypes.dtype(lhs), np.integer):
    quotient = np.floor_divide(lhs, rhs)
    select = np.logical_and(np.sign(lhs) != np.sign(rhs),
                            np.remainder(lhs, rhs) != 0)
    return np.where(select, quotient + 1, quotient)
  else:
    return np.divide(lhs, rhs)
np_impl[lax.div_p] = div

def rem(lhs, rhs): return np.sign(lhs) * np.remainder(np.abs(lhs), np.abs(rhs))
np_impl[lax.rem_p] = rem

max = np.maximum
np_impl[lax.max_p] = max

min = np.minimum
np_impl[lax.min_p] = min

shift_left = np.left_shift
np_impl[lax.shift_left_p] = shift_left

@curry
def _shift_right(shift_types: Dict, x1, x2):
  shift_type = shift_types[x1.dtype]
  shifted = np.right_shift(x1.view(shift_type), x2.astype(shift_type))
  return shifted.astype(shift_type).view(x1.dtype)

_arithmetic_shift_types = {
  np.dtype('int8'): np.int8,
  np.dtype('int16'): np.int16,
  np.dtype('int32'): np.int32,
  np.dtype('int64'): np.int64,
  # lax does arithmetic (signed) shift irrespective of the type:
  np.dtype('uint8'): np.int8,
  np.dtype('uint16'): np.int16,
  np.dtype('uint32'): np.int32,
  np.dtype('uint64'): np.int64,
}

shift_right_arithmetic = _shift_right(_arithmetic_shift_types)
np_impl[lax.shift_right_arithmetic_p] = shift_right_arithmetic

_logical_shift_types = {
  np.dtype('int8'): np.uint8,
  np.dtype('int16'): np.uint16,
  np.dtype('int32'): np.uint32,
  np.dtype('int64'): np.uint64,
  np.dtype('uint8'): np.uint8,
  np.dtype('uint16'): np.uint16,
  np.dtype('uint32'): np.uint32,
  np.dtype('uint64'): np.uint64,
}

shift_right_logical = _shift_right(_logical_shift_types)
np_impl[lax.shift_right_logical_p] = shift_right_logical

def population_count(x):
  assert np.issubdtype(x.dtype, np.integer)
  dtype = x.dtype
  iinfo = np.iinfo(x.dtype)
  if np.iinfo(x.dtype).bits < 32:
    assert iinfo.kind in ('i', 'u')
    x = x.astype(np.uint32 if iinfo.kind == 'u' else np.int32)
  if iinfo.kind == 'i':
    x = x.view(f"uint{np.iinfo(x.dtype).bits}")
  assert x.dtype in (np.uint32, np.uint64)
  m = [
    0x5555555555555555,  # binary: 0101...
    0x3333333333333333,  # binary: 00110011..
    0x0f0f0f0f0f0f0f0f,  # binary:  4 zeros,  4 ones ...
    0x00ff00ff00ff00ff,  # binary:  8 zeros,  8 ones ...
    0x0000ffff0000ffff,  # binary: 16 zeros, 16 ones ...
    0x00000000ffffffff,  # binary: 32 zeros, 32 ones
  ]

  if x.dtype == np.uint32:
    m = list(_map(np.uint32, m[:-1]))
  else:
    m = list(_map(np.uint64, m))

  x = (x & m[0]) + ((x >>  1) & m[0])  # put count of each  2 bits into those  2 bits
  x = (x & m[1]) + ((x >>  2) & m[1])  # put count of each  4 bits into those  4 bits
  x = (x & m[2]) + ((x >>  4) & m[2])  # put count of each  8 bits into those  8 bits
  x = (x & m[3]) + ((x >>  8) & m[3])  # put count of each 16 bits into those 16 bits
  x = (x & m[4]) + ((x >> 16) & m[4])  # put count of each 32 bits into those 32 bits
  if x.dtype == np.uint64:
    x = (x & m[5]) + ((x >> 32) & m[5])  # put count of each 64 bits into those 64 bits
  return x.astype(dtype)
np_impl[lax.population_count_p] = population_count

eq = np.equal
np_impl[lax.eq_p] = eq

ne = np.not_equal
np_impl[lax.ne_p] = ne

ge = np.greater_equal
np_impl[lax.ge_p] = ge

gt = np.greater
np_impl[lax.gt_p] = gt

le = np.less_equal
np_impl[lax.le_p] = le

lt = np.less
np_impl[lax.lt_p] = lt

def convert_element_type(operand, new_dtype, old_dtype=None):
  return np.asarray(operand, dtype=new_dtype)
np_impl[lax.convert_element_type_p] = convert_element_type

def bitcast_convert_type(operand, new_dtype):
  return np.asarray(operand).view(new_dtype)
np_impl[lax.bitcast_convert_type_p] = bitcast_convert_type

def clamp(min, operand, max):
  return np.clip(operand, np.clip(min, None, max), max).astype(operand.dtype)
np_impl[lax.clamp_p] = clamp

def concatenate(*operands, dimension):
  return np.concatenate(operands, dimension)
np_impl[lax.concatenate_p] = concatenate

def conv_general_dilated(lhs, rhs, window_strides, padding,
                         lhs_dilation, rhs_dilation, dimension_numbers,
                         feature_group_count, batch_group_count, **_):
  lhs_perm, rhs_perm, out_perm = dimension_numbers
  lhs = np.transpose(lhs, lhs_perm)
  rhs = np.transpose(rhs, rhs_perm)
  batch_size, feature_size, *spatial_shape = lhs.shape
  lhs = np.reshape(
    lhs, (batch_group_count, batch_size // batch_group_count,
          feature_group_count, feature_size // feature_group_count) +
         tuple(spatial_shape)).swapaxes(1, 2)
  feature_size, *sh = rhs.shape
  rhs = np.reshape(
    rhs, (batch_group_count, feature_group_count,
          feature_size // batch_group_count // feature_group_count) + tuple(sh))
  outs = [conv_with_general_padding(lhs, rhs, window_strides, padding,
                                    lhs_dilation, rhs_dilation)
          for lhs, rhs in zip(lhs, rhs) for lhs, rhs in zip(lhs, rhs)]
  return np.transpose(np.concatenate(outs, axis=1), np.argsort(out_perm))
np_impl[lax.conv_general_dilated_p] = conv_general_dilated

def conv_with_general_padding(lhs, rhs, window_strides, padding, lhs_dilation,
                              rhs_dilation, precision=None):
  return _conv(_dilate(lhs, lhs_dilation), _dilate(rhs, rhs_dilation),
               window_strides, padding)

def _dilate(operand, factors, fill_value=0):
  # this logic is like lax.pad, but with two leading dimensions, no edge
  # padding, and factors are at least 1 (interior padding is at least 0)
  outspace = lax._dilate_shape(operand.shape[2:], dilation=factors)
  out = np.full(operand.shape[:2] + tuple(outspace), fill_value, operand.dtype)
  lhs_slices = tuple(_slice(None, None, step) for step in factors)
  out[(_slice(None),) * 2 + lhs_slices] = operand
  return out

def _conv(lhs, rhs, window_strides, pads):
  view, view_axes, rhs_axes, out_axes = _conv_view(
    lhs, rhs.shape, window_strides, pads, 0.)
  return opt_einsum.contract(
    view, view_axes, rhs, rhs_axes, out_axes, use_blas=True)

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

  out_strides = np.multiply(window_strides, lhs.strides[2:])
  view_strides = lhs.strides[:1] + tuple(out_strides) + lhs.strides[1:]

  out_shape = np.maximum(
    0, np.floor_divide(np.subtract(in_shape, filter_shape), window_strides) + 1)
  view_shape = lhs.shape[:1] + tuple(out_shape) + rhs_shape[1:]

  view = np.lib.stride_tricks.as_strided(lhs, view_shape, view_strides)

  view_axes = list(range(view.ndim))
  sum_axes = view_axes[-dim-1:]
  rhs_axes = [view.ndim] + sum_axes
  out_axes = [0, view.ndim] + list(range(1, dim+1))

  return view, view_axes, rhs_axes, out_axes

def _pad(arr, pads, pad_value):
  out = np.pad(arr, np.maximum(0, pads), mode='constant',
               constant_values=pad_value).astype(arr.dtype)
  slices = tuple(_slice(abs(lo) if lo < 0 else 0, hi % dim if hi < 0 else None)
                 for (lo, hi), dim in zip(pads, np.shape(arr)))
  return out[slices]

is_np_version_below_1_19 = np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('1.19.0')
Ufunc = namedtuple('ufunc', ['reduce', 'identity'])
ufunc = Union[np.ufunc, Ufunc]

@curry
def _reduce(reducer: ufunc, operand: Array, axes: Tuple[int, ...]) -> ndarray:
  operand = np.asarray(operand)
  dtype = operand.dtype
  if dtype == dtypes.bfloat16:
    operand = operand.astype(np.float32)
  elif dtype == np.uint64 and not config.FLAGS.jax_enable_x64:
    operand = operand.astype(np.uint32)
  out = reducer.reduce(operand, axis=axes)
  if dtype == dtypes.bfloat16 and out.dtype == np.object:
    out = out.astype(np.float32)
  return out.astype(dtype)
np_impl[lax.reduce_sum_p] = _reduce(np.add)
np_impl[lax.reduce_prod_p] = _reduce(np.multiply)
np_impl[lax.reduce_max_p] = _reduce(np.maximum)
np_impl[lax.reduce_min_p] = _reduce(np.minimum)
np_impl[lax.reduce_or_p] = _reduce(np.logical_or)
np_impl[lax.reduce_and_p] = _reduce(np.logical_and)

def reduce(operand: Array, init_value: Array, computation: Callable,
           jaxpr: Jaxpr, consts: Tuple[Any, ...],
           dimensions: Tuple[int, ...]) -> ndarray:
  reducer = _reducer(jaxpr, consts, init_value)
  return _reduce(reducer)(operand, dimensions)
np_impl[lax.reduce_p] = reduce

def _reducer(jaxpr: Jaxpr, consts: Tuple[Any, ...], init_value: Array) -> ufunc:
  aval = lax._abstractify(init_value)
  fun = core.jaxpr_as_fun(TypedJaxpr(jaxpr, consts, (aval, aval), (aval,)))
  @numpy_eval()
  def binop(x, y):
    out, = fun(x, y)
    return out

  binop_ = (lambda x, y: binop(np.array(x, np.uint64), np.array(y, np.uint64))
            if dtypes.dtype(init_value) == np.uint64 else binop)

  if dtypes.dtype(init_value) == dtypes.bfloat16:
    init_value = np.asarray(init_value, dtype=np.float32)

  if is_np_version_below_1_19:
    # np.frompyfunc does not have identity parameter before NumPy version 1.19:
    # https://numpy.org/doc/1.18/reference/generated/numpy.frompyfunc.html
    def reduce(operand, axis=0):
      axis = range(np.ndim(operand)) if axis is None else axis
      result = np.full(np.delete(np.shape(operand), axis), init_value,
                       dtype=dtypes.dtype(operand))
      for idx, _ in np.ndenumerate(operand):
        out_idx = tuple(np.delete(idx, axis))
        result[out_idx] = binop_(result[out_idx], operand[idx])
      return result
    return Ufunc(reduce=reduce, identity=init_value)

  return np.frompyfunc(binop_, nin=2, nout=1, identity=init_value)

@curry
def _reduce_window(reducer: ufunc, operand: Array,
                   window_dimensions: Tuple[int, ...],
                   window_strides: Tuple[int, ...],
                   padding: Tuple[Tuple[int, int], ...],
                   base_dilation: Tuple[int, ...],
                   window_dilation: Tuple[int, ...]) -> ndarray:
  operand = np.reshape(operand, (1, 1) + operand.shape)
  dilated_window_shape = lax._dilate_shape(window_dimensions, window_dilation)
  identity = _identity(reducer, operand.dtype)
  if any(d != 1 for d in base_dilation):
    operand = _dilate(operand, base_dilation, identity)
  view, *_ = _conv_view(operand, (1, 1) + tuple(dilated_window_shape),
                        window_strides, padding, pad_value=identity)
  el: List[Any] = [...]
  view = view[tuple(el + [_slice(None, None, d) for d in window_dilation])]
  view = view.reshape(view.shape[1:1+len(window_dimensions)] + (-1,))
  return _reduce(reducer)(view, axes=-1)
np_impl[lax.reduce_window_max_p] = _reduce_window(np.maximum)
np_impl[lax.reduce_window_min_p] = _reduce_window(np.minimum)
np_impl[lax.reduce_window_sum_p] = _reduce_window(np.add)

def _identity(reducer: ufunc, dtype: Any) -> ndarray:
  if reducer is np.maximum or reducer is np.minimum:
    ismin = reducer is np.minimum
    if dtype == np.bool: return ismin
    if dtypes.issubdtype(dtype, np.inexact): return np.inf if ismin else -np.inf
    return np.iinfo(dtype).max if ismin else np.iinfo(dtype).min
  return np.asarray(reducer.identity, dtype=dtype)

def reduce_window(operand: Array, init_value: Array,
                  jaxpr: Jaxpr, consts: Tuple[Any, ...],
                  window_dimensions: Tuple[int, ...],
                  window_strides: Tuple[int, ...],
                  padding: Tuple[Tuple[int, int], ...],
                  base_dilation: Tuple[int, ...],
                  window_dilation: Tuple[int, ...]) -> ndarray:
  reducer = _reducer(jaxpr, consts, init_value)
  return _reduce_window(reducer)(operand, window_dimensions,
                                 window_strides, padding,
                                 base_dilation, window_dilation)
np_impl[lax.reduce_window_p] = reduce_window

def dot_general(lhs, rhs, dimension_numbers, precision=None):
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
  out_axis_ids = filter(not_none,
                        batch_ids + lhs_out_axis_ids + rhs_out_axis_ids)
  assert lhs.dtype == rhs.dtype
  dtype = np.float32 if lhs.dtype == dtypes.bfloat16 else None
  out = np.einsum(lhs, lhs_axis_ids, rhs, rhs_axis_ids, out_axis_ids,
                  dtype=dtype)
  return out.astype(dtypes.bfloat16) if lhs.dtype == dtypes.bfloat16 else out
np_impl[lax.dot_general_p] = dot_general

def broadcast_in_dim(operand, shape, broadcast_dimensions):
  in_reshape = np.ones(len(shape), dtype=np.int32)
  for i, bd in enumerate(broadcast_dimensions):
    in_reshape[bd] = operand.shape[i]
  return np.broadcast_to(np.reshape(operand, in_reshape), shape)
np_impl[lax.broadcast_in_dim_p] = broadcast_in_dim

def squeeze(array, dimensions): return np.squeeze(array, dimensions)
np_impl[lax.squeeze_p] = squeeze

def reshape(operand, new_sizes, dimensions=None):
  if dimensions is not None:
    operand = np.transpose(operand, dimensions)
  return np.reshape(_ensure_numpy(operand), new_sizes)
np_impl[lax.reshape_p] = reshape

def pad(operand, padding_value, padding_config):
  # https://www.tensorflow.org/xla/operation_semantics#pad
  lo, hi, interior = zip(*padding_config)
  # Handle first the positive edge padding and interior
  lo_pos, hi_pos = np.clip(lo, 0, None), np.clip(hi, 0, None)
  outshape = np.add(
    np.add(np.add(lo_pos, hi_pos), operand.shape),
    np.maximum(0, np.multiply(interior, np.subtract(operand.shape, 1))))
  out = np.full(outshape, padding_value, operand.dtype)
  lhs_slices = tuple(_slice(l if l > 0 else 0, -h if h > 0 else None, step)
                     for l, h, step in zip(lo_pos, hi_pos, np.add(1, interior)))
  out[lhs_slices] = operand
  trim_slices = tuple(_slice(-l if l < 0 else 0, h if h < 0 else None)
                      for l, h in zip(lo, hi))
  return out[trim_slices]
np_impl[lax.pad_p] = pad

def rev(operand, dimensions):
  dimensions = frozenset(dimensions)
  indexer = (_slice(None, None, -1) if d in dimensions else _slice(None)
             for d in range(np.ndim(operand)))
  return operand[tuple(indexer)]
np_impl[lax.rev_p] = rev

select = np.where
np_impl[lax.select_p] = select

def slice(operand, start_indices, limit_indices, strides=None):  # pylint: disable=redefined-builtin
  if strides is None:
    strides = np.ones(len(start_indices)).astype(int)
  idx = tuple(map(_slice, start_indices, limit_indices, strides))
  return _ensure_numpy(operand)[idx]
np_impl[lax.slice_p] = slice

def dynamic_slice(operand, *start_indices, slice_sizes):
  out = np.zeros(slice_sizes, dtype=operand.dtype)
  idx = tuple(_slice(start, start+size)
              for start, size in zip(start_indices, slice_sizes))
  section = _ensure_numpy(operand)[idx]
  out[tuple(_slice(None, stop) for stop in section.shape)] = section
  return out
np_impl[lax.dynamic_slice_p] = dynamic_slice

def dynamic_update_slice(operand, update, *start_indices):
  slices = tuple(map(_slice, start_indices, np.add(start_indices, update.shape)))
  updated_operand = np.copy(operand)
  updated_operand[slices] = update
  return updated_operand
np_impl[lax.dynamic_update_slice_p] = dynamic_update_slice

def transpose(operand: Array, permutation: Sequence[int]) -> ndarray:
  return np.transpose(_ensure_numpy(operand), permutation)
np_impl[lax.transpose_p] = transpose

def sort(*operands, dimension, is_stable, num_keys):
  if len(operands) == 1:
    operand, = operands
    return np.sort(operand, dimension, 'stable' if is_stable else 'quicksort'),
  keys = operands[:num_keys][::-1]
  indices = np.lexsort(keys, axis=dimension)
  return tuple(np.take_along_axis(o, indices, axis=dimension) for o in operands)
np_impl[lax.sort_p] = sort

def sort_key_val(keys, values, dimension=-1, is_stable=True):
  return sort(keys, values,
              dimension=dimension, is_stable=is_stable, num_keys=1)

def top_k(x, k):
  bcast_idxs = np.broadcast_to(np.arange(x.shape[-1], dtype=np.int32), x.shape)
  sorted_vals, sorted_idxs = sort_key_val(x, bcast_idxs)
  return sorted_vals[..., :-k-1:-1], sorted_idxs[..., :-k-1:-1]
np_impl[lax.top_k_p] = top_k

def argmax(operand, axes, index_dtype):
  axis, = axes
  return np.argmax(operand, axis).astype(index_dtype)
np_impl[lax.argmax_p] = argmax

def argmin(operand, axes, index_dtype):
  axis, = axes
  return np.argmin(operand, axis).astype(index_dtype)
np_impl[lax.argmin_p] = argmin

cummin = np.minimum.accumulate
np_impl[lax.cummin_p] = cummin

cummax = np.maximum.accumulate
np_impl[lax.cummax_p] = cummax

def cumsum(x, axis): return np.cumsum(x, axis).astype(x.dtype)
np_impl[lax.cumsum_p] = cumsum

def cumprod(x, axis): return np.cumprod(x, axis).astype(x.dtype)
np_impl[lax.cumprod_p] = cumprod

def gather(operand: Array, start_indices: Array,
           dimension_numbers: lax.GatherDimensionNumbers,
           slice_sizes: lax.Shape) -> ndarray:
  offset_dims, collapsed_slice_dims, axes = dimension_numbers
  indices_shape = start_indices.shape[:-1]
  single = np.prod(indices_shape) == 1
  uncollapsed_slice_dims = [d for d in range(len(slice_sizes))
                            if d in axes and d not in collapsed_slice_dims]
  start_indices = _ensure_numpy(start_indices)
  if not single:
    start_indices = start_indices.reshape([1] * len(uncollapsed_slice_dims) +
                                          list(start_indices.shape))
  def index(dim: int, slice_size: int):
    if dim not in axes:
      return _slice(0, slice_size)
    (start_dim_index,), = np.argwhere(np.equal(axes, dim))
    starts = start_indices[..., start_dim_index]
    # lax.gather_p rectifies start indices to avoid out-of-bound slice elements:
    starts = np.minimum(starts, operand.shape[dim] - slice_size)
    if dim in collapsed_slice_dims:
      return starts
    if single:
      start = starts.flat[0]
      return _slice(start, start + slice_size)

    (udim_index,), = np.argwhere(np.equal(uncollapsed_slice_dims, dim))
    starts = np.squeeze(starts, udim_index)
    return np.linspace(starts, stop=starts + slice_size, num=slice_size,
                       dtype=int, axis=udim_index, endpoint=False)
  idx = tuple(map(index, range(len(operand.shape)), slice_sizes))
  out = _ensure_numpy(operand)[idx]
  if single:
    return out.reshape(list(indices_shape) + list(out.shape))
  uncollapsed_slice_output_dims = [
    udim + len(indices_shape) - np.sum(np.less(collapsed_slice_dims, udim))
    for udim in uncollapsed_slice_dims]
  for udim_index, uoutdim in enumerate(reversed(uncollapsed_slice_output_dims)):
    out = np.moveaxis(out, udim_index, uoutdim)
  for rev_odim_index, odim in enumerate(reversed(offset_dims)):
    out = np.moveaxis(out, -(rev_odim_index + 1), odim)
  return out
np_impl[lax.gather_p] = gather

# TODO scatter[_add/mul/min/max]_p, select_and_scatter[_add]_p

np_impl[random.threefry2x32_p] = numpy_eval()(partial(
  random._threefry2x32_lowering, use_rolled_loops=False))
np_impl[xla.device_put_p] = lambda x, device: x
np_impl[ad_util.stop_gradient_p] = lambda x: x
np_impl[ad_util.zeros_like_p] = lambda x: np.zeros(np.shape(x), dtypes.dtype(x))
