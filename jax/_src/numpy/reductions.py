# Copyright 2022 Google LLC
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

import builtins
from functools import partial
import operator
from typing import Optional, Tuple, Union
import warnings

import numpy as np

import jax
from jax import core
from jax import lax
from jax._src import api
from jax._src import dtypes
from jax._src.numpy.ndarray import ndarray
from jax._src.numpy.util import _broadcast_to, _check_arraylike, _complex_elem_type, _promote_dtypes_inexact, _where, _wraps
from jax._src.lax import lax as lax_internal
from jax._src.util import canonicalize_axis as _canonicalize_axis, maybe_named_axis


_all = builtins.all
_lax_const = lax_internal._const


def _asarray(a):
  # simplified version of jnp.asarray() for local use.
  return a if isinstance(a, ndarray) else api.device_put(a)

def _isscalar(element):
  if hasattr(element, '__jax_array__'):
    element = element.__jax_array__()
  return dtypes.is_python_scalar(element) or np.isscalar(element)

def _moveaxis(a, source: int, destination: int):
  # simplified version of jnp.moveaxis() for local use.
  _check_arraylike("moveaxis", a)
  a = _asarray(a)
  source = _canonicalize_axis(source, np.ndim(a))
  destination = _canonicalize_axis(destination, np.ndim(a))
  perm = [i for i in range(np.ndim(a)) if i != source]
  perm.insert(destination, source)
  return lax.transpose(a, perm)

def _upcast_f16(dtype):
  if dtype in [np.float16, dtypes.bfloat16]:
    return np.dtype('float32')
  return dtype

def _reduction(a, name, np_fun, op, init_val, has_identity=True,
               preproc=None, bool_op=None, upcast_f16_for_computation=False,
               axis=None, dtype=None, out=None, keepdims=False, initial=None,
               where_=None, parallel_reduce=None):
  bool_op = bool_op or op
  # Note: we must accept out=None as an argument, because numpy reductions delegate to
  # object methods. For example `np.sum(x)` will call `x.sum()` if the `sum()` method
  # exists, passing along all its arguments.
  if out is not None:
    raise NotImplementedError(f"The 'out' argument to jnp.{name} is not supported.")
  _check_arraylike(name, a)
  lax_internal._check_user_dtype_supported(dtype, name)
  axis = core.concrete_or_error(None, axis, f"axis argument to jnp.{name}().")

  if initial is None and not has_identity and where_ is not None:
    raise ValueError(f"reduction operation {name} does not have an identity, so to use a "
                     f"where mask one has to specify 'initial'")

  a = a if isinstance(a, ndarray) else _asarray(a)
  a = preproc(a) if preproc else a
  pos_dims, dims = _reduction_dims(a, axis)

  if initial is None and not has_identity:
    shape = np.shape(a)
    if not _all(core.greater_equal_dim(shape[d], 1) for d in pos_dims):
      raise ValueError(f"zero-size array to reduction operation {name} which has no identity")

  result_dtype = dtypes.canonicalize_dtype(dtype or dtypes.dtype(np_fun(np.ones((), dtype=dtypes.dtype(a)))))
  if upcast_f16_for_computation and dtypes.issubdtype(result_dtype, np.inexact):
    computation_dtype = _upcast_f16(result_dtype)
  else:
    computation_dtype = result_dtype
  a = lax.convert_element_type(a, computation_dtype)
  op = op if computation_dtype != np.bool_ else bool_op
  # NB: in XLA, init_val must be an identity for the op, so the user-specified
  # initial value must be applied afterward.
  init_val = _reduction_init_val(a, init_val)
  if where_ is not None:
    a = _where(where_, a, init_val)
  if pos_dims is not dims:
    if parallel_reduce is None:
      raise NotImplementedError(f"Named reductions not implemented for jnp.{name}()")
    result = parallel_reduce(a, dims)
  else:
    result = lax.reduce(a, init_val, op, dims)
  if initial is not None:
    result = op(lax.convert_element_type(initial, a.dtype), result)
  if keepdims:
    result = lax.expand_dims(result, pos_dims)
  return lax.convert_element_type(result, dtype or result_dtype)

def _canonicalize_axis_allow_named(x, rank):
  return maybe_named_axis(x, lambda i: _canonicalize_axis(i, rank), lambda name: name)

def _reduction_dims(a, axis):
  if axis is None:
    return (tuple(range(np.ndim(a))),) * 2
  elif not isinstance(axis, (np.ndarray, tuple, list)):
    axis = (axis,)
  canon_axis = tuple(_canonicalize_axis_allow_named(x, np.ndim(a))
                     for x in axis)
  if len(canon_axis) != len(set(canon_axis)):
    raise ValueError(f"duplicate value in 'axis': {axis}")
  canon_pos_axis = tuple(x for x in canon_axis if isinstance(x, int))
  if len(canon_pos_axis) != len(canon_axis):
    return canon_pos_axis, canon_axis
  else:
    return canon_axis, canon_axis

def _reduction_init_val(a, init_val):
  # This function uses np.* functions because lax pattern matches against the
  # specific concrete values of the reduction inputs.
  a_dtype = dtypes.canonicalize_dtype(dtypes.dtype(a))
  if a_dtype == 'bool':
    return np.array(init_val > 0, dtype=a_dtype)
  try:
    return np.array(init_val, dtype=a_dtype)
  except OverflowError:
    assert dtypes.issubdtype(a_dtype, np.integer)
    sign, info = np.sign(init_val), dtypes.iinfo(a_dtype)
    return np.array(info.min if sign < 0 else info.max, dtype=a_dtype)

def _cast_to_bool(operand):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=np.ComplexWarning)
    return lax.convert_element_type(operand, np.bool_)


def _ensure_optional_axes(x):
  def force(x):
    if x is None:
      return None
    try:
      return operator.index(x)
    except TypeError:
      return tuple(i if isinstance(i, str) else operator.index(i) for i in x)
  return core.concrete_or_error(
    force, x, "The axis argument must be known statically.")


@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'), inline=True)
def _reduce_sum(a, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                dtype=None, out=None, keepdims=None, initial=None, where=None):
  return _reduction(a, "sum", np.sum, lax.add, 0,
                    bool_op=lax.bitwise_or, upcast_f16_for_computation=True,
                    axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                    initial=initial, where_=where, parallel_reduce=lax.psum)

@_wraps(np.sum, skip_params=['out'])
def sum(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
        out=None, keepdims=None, initial=None, where=None):
  return _reduce_sum(a, axis=_ensure_optional_axes(axis), dtype=dtype, out=out,
                     keepdims=keepdims, initial=initial, where=where)


@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'), inline=True)
def _reduce_prod(a, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                 dtype=None, out=None, keepdims=None, initial=None, where=None):
  return _reduction(a, "prod", np.prod, lax.mul, 1,
                    bool_op=lax.bitwise_and, upcast_f16_for_computation=True,
                    axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                    initial=initial, where_=where)

@_wraps(np.prod, skip_params=['out'])
def prod(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
         out=None, keepdims=None, initial=None, where=None):
  return _reduce_prod(a, axis=_ensure_optional_axes(axis), dtype=dtype,
                      out=out, keepdims=keepdims, initial=initial, where=where)


@partial(api.jit, static_argnames=('axis', 'keepdims'), inline=True)
def _reduce_max(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
                keepdims=None, initial=None, where=None):
  return _reduction(a, "max", np.max, lax.max, -np.inf, has_identity=False,
                    axis=axis, out=out, keepdims=keepdims,
                    initial=initial, where_=where, parallel_reduce=lax.pmax)

@_wraps(np.max, skip_params=['out'])
def max(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
        keepdims=None, initial=None, where=None):
  return _reduce_max(a, axis=_ensure_optional_axes(axis), out=out,
                     keepdims=keepdims, initial=initial, where=where)

@partial(api.jit, static_argnames=('axis', 'keepdims'), inline=True)
def _reduce_min(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
                keepdims=None, initial=None, where=None):
  return _reduction(a, "min", np.min, lax.min, np.inf, has_identity=False,
                    axis=axis, out=out, keepdims=keepdims,
                    initial=initial, where_=where, parallel_reduce=lax.pmin)

@_wraps(np.min, skip_params=['out'])
def min(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
        keepdims=None, initial=None, where=None):
  return _reduce_min(a, axis=_ensure_optional_axes(axis), out=out,
                     keepdims=keepdims, initial=initial, where=where)

@partial(api.jit, static_argnames=('axis', 'keepdims'), inline=True)
def _reduce_all(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
                keepdims=None, *, where=None):
  return _reduction(a, "all", np.all, lax.bitwise_and, True, preproc=_cast_to_bool,
                    axis=axis, out=out, keepdims=keepdims, where_=where)

@_wraps(np.all, skip_params=['out'])
def all(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
        keepdims=None, *, where=None):
  return _reduce_all(a, axis=_ensure_optional_axes(axis), out=out,
                     keepdims=keepdims, where=where)

@partial(api.jit, static_argnames=('axis', 'keepdims'), inline=True)
def _reduce_any(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
                keepdims=None, *, where=None):
  return _reduction(a, "any", np.any, lax.bitwise_or, False, preproc=_cast_to_bool,
                    axis=axis, out=out, keepdims=keepdims, where_=where)

@_wraps(np.any, skip_params=['out'])
def any(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
        keepdims=None, *, where=None):
  return _reduce_any(a, axis=_ensure_optional_axes(axis), out=out,
                     keepdims=keepdims, where=where)

product = prod
amin = min
amax = max
alltrue = all
sometrue = any

def _axis_size(a, axis):
  if not isinstance(axis, (tuple, list)):
    axis = (axis,)
  size = 1
  a_shape = np.shape(a)
  for a in axis:
    size *= maybe_named_axis(a, lambda i: a_shape[i], lambda name: lax.psum(1, name))
  return size

@_wraps(np.mean, skip_params=['out'])
def mean(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
         out=None, keepdims=False, *, where=None):
  return _mean(a, _ensure_optional_axes(axis), dtype, out, keepdims,
               where=where)

@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'), inline=True)
def _mean(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
         out=None, keepdims=False, *, where=None):
  _check_arraylike("mean", a)
  lax_internal._check_user_dtype_supported(dtype, "mean")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.mean is not supported.")

  if where is None:
    if axis is None:
      normalizer = core.dimension_as_value(np.size(a))
    else:
      normalizer = core.dimension_as_value(_axis_size(a, axis))
  else:
    normalizer = sum(_broadcast_to(where, np.shape(a)), axis, dtype=dtype, keepdims=keepdims)

  if dtype is None:
    dtype = dtypes._to_inexact_dtype(dtypes.dtype(a))
  dtype = dtypes.canonicalize_dtype(dtype)

  return lax.div(
      sum(a, axis, dtype=dtype, keepdims=keepdims, where=where),
      lax.convert_element_type(normalizer, dtype))

@_wraps(np.average)
def average(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, weights=None,
            returned=False, keepdims=False):
  return _average(a, _ensure_optional_axes(axis), weights, returned, keepdims)

@partial(api.jit, static_argnames=('axis', 'returned', 'keepdims'), inline=True)
def _average(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, weights=None,
             returned=False, keepdims=False):
  if weights is None: # Treat all weights as 1
    _check_arraylike("average", a)
    a, = _promote_dtypes_inexact(a)
    avg = mean(a, axis=axis, keepdims=keepdims)
    if axis is None:
      weights_sum = lax.full((), core.dimension_as_value(a.size), dtype=avg.dtype)
    else:
      weights_sum = lax.full_like(avg, core.dimension_as_value(a.shape[axis]), dtype=avg.dtype)
  else:
    _check_arraylike("average", a, weights)
    a, weights = _promote_dtypes_inexact(a, weights)

    a_shape = np.shape(a)
    a_ndim = len(a_shape)
    weights_shape = np.shape(weights)
    axis = None if axis is None else _canonicalize_axis(axis, a_ndim)

    if a_shape != weights_shape:
      # Make sure the dimensions work out
      if axis is None:
        raise ValueError("Axis must be specified when shapes of a and "
                         "weights differ.")
      if len(weights_shape) != 1:
        raise ValueError("1D weights expected when shapes of a and "
                         "weights differ.")
      if not core.symbolic_equal_dim(weights_shape[0], a_shape[axis]):
        raise ValueError("Length of weights not "
                         "compatible with specified axis.")

      weights = _broadcast_to(weights, (a_ndim - 1) * (1,) + weights_shape)
      weights = _moveaxis(weights, -1, axis)

    weights_sum = sum(weights, axis=axis, keepdims=keepdims)
    avg = sum(a * weights, axis=axis, keepdims=keepdims) / weights_sum

  if returned:
    if avg.shape != weights_sum.shape:
      weights_sum = _broadcast_to(weights_sum, avg.shape)
    return avg, weights_sum
  return avg


@_wraps(np.var, skip_params=['out'])
def var(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
        out=None, ddof=0, keepdims=False, *, where=None):
  return _var(a, _ensure_optional_axes(axis), dtype, out, ddof, keepdims,
              where=where)

@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'))
def _var(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
        out=None, ddof=0, keepdims=False, *, where=None):
  _check_arraylike("var", a)
  lax_internal._check_user_dtype_supported(dtype, "var")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.var is not supported.")

  computation_dtype, dtype = _var_promote_types(dtypes.dtype(a), dtype)
  a = a.astype(computation_dtype)
  a_mean = mean(a, axis, dtype=computation_dtype, keepdims=True, where=where)
  centered = lax.sub(a, a_mean)
  if dtypes.issubdtype(centered.dtype, np.complexfloating):
    centered = lax.real(lax.mul(centered, lax.conj(centered)))
  else:
    centered = lax.square(centered)

  if where is None:
    if axis is None:
      normalizer = core.dimension_as_value(np.size(a))
    else:
      normalizer = core.dimension_as_value(_axis_size(a, axis))
  else:
    normalizer = sum(_broadcast_to(where, np.shape(a)), axis, dtype=dtype, keepdims=keepdims)
  normalizer = normalizer - ddof

  result = sum(centered, axis, keepdims=keepdims, where=where)
  out = lax.div(result, lax.convert_element_type(normalizer, result.dtype))
  return lax.convert_element_type(out, dtype)


def _var_promote_types(a_dtype, dtype):
  if dtype:
    if (not dtypes.issubdtype(dtype, np.complexfloating) and
        dtypes.issubdtype(a_dtype, np.complexfloating)):
      msg = ("jax.numpy.var does not yet support real dtype parameters when "
             "computing the variance of an array of complex values. The "
             "semantics of numpy.var seem unclear in this case. Please comment "
             "on https://github.com/google/jax/issues/2283 if this behavior is "
             "important to you.")
      raise ValueError(msg)
    computation_dtype = dtype
  else:
    if not dtypes.issubdtype(a_dtype, np.inexact):
      dtype = dtypes._to_inexact_dtype(a_dtype)
      computation_dtype = dtype
    else:
      dtype = _complex_elem_type(a_dtype)
      computation_dtype = a_dtype
  return _upcast_f16(computation_dtype), dtype


@_wraps(np.std, skip_params=['out'])
def std(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
        out=None, ddof=0, keepdims=False, *, where=None):
  return _std(a, _ensure_optional_axes(axis), dtype, out, ddof, keepdims,
              where=where)

@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'))
def _std(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
        out=None, ddof=0, keepdims=False, *, where=None):
  _check_arraylike("std", a)
  lax_internal._check_user_dtype_supported(dtype, "std")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.std is not supported.")
  return lax.sqrt(var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where))


@_wraps(np.ptp, skip_params=['out'])
def ptp(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
        keepdims=False):
  return _ptp(a, _ensure_optional_axes(axis), out, keepdims)

@partial(api.jit, static_argnames=('axis', 'keepdims'))
def _ptp(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
        keepdims=False):
  _check_arraylike("ptp", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.ptp is not supported.")
  x = amax(a, axis=axis, keepdims=keepdims)
  y = amin(a, axis=axis, keepdims=keepdims)
  return lax.sub(x, y)


@_wraps(np.count_nonzero)
@partial(api.jit, static_argnames=('axis', 'keepdims'))
def count_nonzero(a, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                  keepdims=False):
  _check_arraylike("count_nonzero", a)
  return sum(lax.ne(a, _lax_const(a, 0)), axis=axis,
             dtype=dtypes.canonicalize_dtype(np.int_), keepdims=keepdims)


def _nan_reduction(a, name, jnp_reduction, init_val, nan_if_all_nan,
                   axis=None, keepdims=None, **kwargs):
  _check_arraylike(name, a)
  if not dtypes.issubdtype(dtypes.dtype(a), np.inexact):
    return jnp_reduction(a, axis=axis, keepdims=keepdims, **kwargs)

  out = jnp_reduction(_where(lax_internal._isnan(a), _reduction_init_val(a, init_val), a),
                      axis=axis, keepdims=keepdims, **kwargs)
  if nan_if_all_nan:
    return _where(all(lax_internal._isnan(a), axis=axis, keepdims=keepdims),
                  _lax_const(a, np.nan), out)
  else:
    return out

@_wraps(np.nanmin, skip_params=['out'])
@partial(api.jit, static_argnames=('axis', 'keepdims'))
def nanmin(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
           keepdims=None, initial=None, where=None):
  return _nan_reduction(a, 'nanmin', min, np.inf, nan_if_all_nan=initial is None,
                        axis=axis, out=out, keepdims=keepdims,
                        initial=initial, where=where)

@_wraps(np.nanmax, skip_params=['out'])
@partial(api.jit, static_argnames=('axis', 'keepdims'))
def nanmax(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
           keepdims=None, initial=None, where=None):
  return _nan_reduction(a, 'nanmax', max, -np.inf, nan_if_all_nan=initial is None,
                        axis=axis, out=out, keepdims=keepdims,
                        initial=initial, where=where)

@_wraps(np.nansum, skip_params=['out'])
@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'))
def nansum(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
           out=None, keepdims=None, initial=None, where=None):
  lax_internal._check_user_dtype_supported(dtype, "nanprod")
  return _nan_reduction(a, 'nansum', sum, 0, nan_if_all_nan=False,
                        axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                        initial=initial, where=where)

# Work around a sphinx documentation warning in NumPy 1.22.
if nansum.__doc__ is not None:
  nansum.__doc__ = nansum.__doc__.replace("\n\n\n", "\n\n")

@_wraps(np.nanprod, skip_params=['out'])
@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'))
def nanprod(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
            out=None, keepdims=None, initial=None, where=None):
  lax_internal._check_user_dtype_supported(dtype, "nanprod")
  return _nan_reduction(a, 'nanprod', prod, 1, nan_if_all_nan=False,
                        axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                        initial=initial, where=where)

@_wraps(np.nanmean, skip_params=['out'])
@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'))
def nanmean(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
            out=None, keepdims=False, where=None):
  _check_arraylike("nanmean", a)
  lax_internal._check_user_dtype_supported(dtype, "nanmean")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.nanmean is not supported.")
  if dtypes.issubdtype(dtypes.dtype(a), np.bool_) or dtypes.issubdtype(dtypes.dtype(a), np.integer):
    return mean(a, axis, dtype, out, keepdims, where=where)
  if dtype is None:
    dtype = dtypes.dtype(a)
  nan_mask = lax_internal.bitwise_not(lax_internal._isnan(a))
  normalizer = sum(nan_mask, axis=axis, dtype=np.int32, keepdims=keepdims, where=where)
  normalizer = lax.convert_element_type(normalizer, dtype)
  td = lax.div(nansum(a, axis, dtype=dtype, keepdims=keepdims, where=where), normalizer)
  return td


@_wraps(np.nanvar, skip_params=['out'])
@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'))
def nanvar(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
           out=None, ddof=0, keepdims=False, where=None):
  _check_arraylike("nanvar", a)
  lax_internal._check_user_dtype_supported(dtype, "nanvar")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.nanvar is not supported.")

  computation_dtype, dtype = _var_promote_types(dtypes.dtype(a), dtype)
  a = a.astype(computation_dtype)
  a_mean = nanmean(a, axis, dtype=computation_dtype, keepdims=True, where=where)

  centered = _where(lax_internal._isnan(a), 0, lax.sub(a, a_mean))  # double-where trick for gradients.
  if dtypes.issubdtype(centered.dtype, np.complexfloating):
    centered = lax.real(lax.mul(centered, lax.conj(centered)))
  else:
    centered = lax.square(centered)

  normalizer = sum(lax_internal.bitwise_not(lax_internal._isnan(a)),
                   axis=axis, keepdims=keepdims, where=where)
  normalizer = normalizer - ddof
  normalizer_mask = lax.le(normalizer, 0)
  result = sum(centered, axis, keepdims=keepdims, where=where)
  result = _where(normalizer_mask, np.nan, result)
  divisor = _where(normalizer_mask, 1, normalizer)
  out = lax.div(result, lax.convert_element_type(divisor, result.dtype))
  return lax.convert_element_type(out, dtype)


@_wraps(np.nanstd, skip_params=['out'])
@partial(api.jit, static_argnames=('axis', 'dtype', 'keepdims'))
def nanstd(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
           out=None, ddof=0, keepdims=False, where=None):
  _check_arraylike("nanstd", a)
  lax_internal._check_user_dtype_supported(dtype, "nanstd")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.nanstd is not supported.")
  return lax.sqrt(nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where))


def _make_cumulative_reduction(np_reduction, reduction, fill_nan=False, fill_value=0):
  @_wraps(np_reduction, skip_params=['out'])
  def cumulative_reduction(a,
                           axis: Optional[Union[int, Tuple[int, ...]]] = None,
                           dtype=None, out=None):
    return _cumulative_reduction(a, _ensure_optional_axes(axis), dtype, out)

  @partial(api.jit, static_argnames=('axis', 'dtype'))
  def _cumulative_reduction(a,
                           axis: Optional[Union[int, Tuple[int, ...]]] = None,
                           dtype=None, out=None):
    _check_arraylike(np_reduction.__name__, a)
    if out is not None:
      raise NotImplementedError(f"The 'out' argument to jnp.{np_reduction.__name__} "
                                f"is not supported.")
    lax_internal._check_user_dtype_supported(dtype, np_reduction.__name__)

    if axis is None or _isscalar(a):
      a = lax.reshape(a, (np.size(a),))
      axis = 0

    a_shape = list(np.shape(a))
    num_dims = len(a_shape)
    axis = _canonicalize_axis(axis, num_dims)

    if fill_nan:
      a = _where(lax_internal._isnan(a), _lax_const(a, fill_value), a)

    if not dtype and dtypes.dtype(a) == np.bool_:
      dtype = dtypes.canonicalize_dtype(dtypes.int_)
    if dtype:
      a = lax.convert_element_type(a, dtype)

    return reduction(a, axis)

  return cumulative_reduction


cumsum = _make_cumulative_reduction(np.cumsum, lax.cumsum, fill_nan=False)
cumprod = _make_cumulative_reduction(np.cumprod, lax.cumprod, fill_nan=False)
cumproduct = cumprod
nancumsum = _make_cumulative_reduction(np.nancumsum, lax.cumsum,
                                       fill_nan=True, fill_value=0)
nancumprod = _make_cumulative_reduction(np.nancumprod, lax.cumprod,
                                        fill_nan=True, fill_value=1)
