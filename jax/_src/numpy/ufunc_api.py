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

"""Tools to create numpy-style ufuncs."""

_AT_INPLACE_WARNING = """\
Because JAX arrays are immutable, jnp.ufunc.at() cannot operate inplace like
np.ufunc.at(). Instead, you can pass inplace=False and capture the result; e.g.
>>> arr = jnp.add.at(arr, ind, val, inplace=False)
"""
from functools import partial
import operator

import jax
from jax._src.lax import lax as lax_internal
from jax._src.numpy.lax_numpy import _eliminate_deprecated_list_indexing, append, take
from jax._src.numpy.reductions import _moveaxis
from jax._src.numpy.util import _wraps, check_arraylike, _broadcast_to, _where
from jax._src.numpy.vectorize import vectorize
from jax._src.util import canonicalize_axis
import numpy as np


class ufunc:
  """Functions that operate element-by-element on whole arrays.

  This is a class for LAX-backed implementations of numpy ufuncs.
  """
  def __init__(self, func, /, nin, nout, *, name=None, nargs=None, identity=None):
    # TODO(jakevdp): validate the signature of func via eval_shape.
    self.__name__ = name or func.__name__
    self._call = vectorize(func)
    self.nin = operator.index(nin)
    self.nout = operator.index(nout)
    self.nargs = nargs or self.nin
    self.identity = identity

  def __repr__(self):
    return f"<jnp.ufunc '{self.__name__}'>"

  def __call__(self, *args, out=None, where=None, **kwargs):
    if out is not None:
      raise NotImplementedError(f"out argument of {self}")
    if where is not None:
      raise NotImplementedError(f"where argument of {self}")
    return self._call(*args, **kwargs)

  @_wraps(np.ufunc.reduce, module="numpy.ufunc")
  def reduce(self, a, axis=0, dtype=None, out=None, keepdims=False, initial=None, where=None):
    if self.nin != 2:
      raise ValueError("reduce only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduce only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduce()")
    # TODO(jakevdp): implement where.
    if where is not None:
      raise NotImplementedError(f"where argument of {self.__name__}.reduce()")
    return self._reduce_via_scan(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial)

  def _reduce_via_scan(self, arr, axis=0, dtype=None, keepdims=False, initial=None):
    assert self.nin == 2 and self.nout == 1
    check_arraylike(f"{self.__name__}.reduce", arr)
    arr = lax_internal.asarray(arr)
    if initial is None:
      initial = self.identity
    if dtype is None:
      dtype = jax.eval_shape(self, lax_internal._one(arr), lax_internal._one(arr)).dtype

    if isinstance(axis, tuple):
      axis = tuple(canonicalize_axis(a, arr.ndim) for a in axis)
      raise NotImplementedError("tuple of axes")
    elif axis is None:
      if keepdims:
        final_shape = (1,) * arr.ndim
      else:
        final_shape = ()
      arr = arr.ravel()
      axis = 0
    else:
      axis = canonicalize_axis(axis, arr.ndim)
      if keepdims:
        final_shape = (*arr.shape[:axis], 1, *arr.shape[axis + 1:])
      else:
        final_shape = (*arr.shape[:axis], *arr.shape[axis + 1:])

    # TODO: handle without transpose?
    if axis != 0:
      arr = _moveaxis(arr, axis, 0)

    if initial is None and arr.shape[0] == 0:
      raise ValueError("zero-size array to reduction operation {self.__name__} which has no ideneity")

    def body_fun(i, val):
      return self._call(val, arr[i].astype(dtype))

    if initial is None:
      start = 1
      initial = arr[0]
    else:
      check_arraylike(f"{self.__name__}.reduce", arr)
      start = 0

    initial = _broadcast_to(lax_internal.asarray(initial).astype(dtype), arr.shape[1:])

    result = jax.lax.fori_loop(start, arr.shape[0], body_fun, initial)
    if keepdims:
      result = result.reshape(final_shape)
    return result

  @_wraps(np.ufunc.accumulate, module="numpy.ufunc")
  def accumulate(self, a, axis=0, dtype=None, out=None):
    if self.nin != 2:
      raise ValueError("accumulate only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("accumulate only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.accumulate()")
    return self._accumulate_via_scan(a, axis=axis, dtype=dtype)

  def _accumulate_via_scan(self, arr, axis=0, dtype=None):
    assert self.nin == 2 and self.nout == 1
    check_arraylike(f"{self.__name__}.accumulate", arr)
    arr = lax_internal.asarray(arr)

    if dtype is None:
      dtype = jax.eval_shape(self, lax_internal._one(arr), lax_internal._one(arr)).dtype

    if axis is None or isinstance(axis, tuple):
      raise ValueError("accumulate does not allow multiple axes")
    axis = canonicalize_axis(axis, np.ndim(arr))

    arr = _moveaxis(arr, axis, 0)
    def scan_fun(carry, _):
      i, x = carry
      y = _where(i == 0, arr[0].astype(dtype), self._call(x.astype(dtype), arr[i].astype(dtype)))
      return (i + 1, y), y
    _, result = jax.lax.scan(scan_fun, (0, arr[0].astype(dtype)), None, length=arr.shape[0])
    return _moveaxis(result, 0, axis)

  @_wraps(np.ufunc.accumulate, module="numpy.ufunc")
  def at(self, a, indices, b=None, /, *, inplace=True):
    if inplace:
      raise NotImplementedError(_AT_INPLACE_WARNING)
    if b is None:
      return self._at_via_scan(a, indices)
    else:
      return self._at_via_scan(a, indices, b)

  def _at_via_scan(self, a, indices, *args):
    check_arraylike(f"{self.__name__}.at", a, *args)
    dtype = jax.eval_shape(self, lax_internal._one(a), *(lax_internal._one(arg) for arg in args)).dtype
    a = lax_internal.asarray(a).astype(dtype)
    args = tuple(lax_internal.asarray(arg).astype(dtype) for arg in args)
    indices = _eliminate_deprecated_list_indexing(indices)
    if not indices:
      return a

    shapes = [np.shape(i) for i in indices if not isinstance(i, slice)]
    shape = shapes and jax.lax.broadcast_shapes(*shapes)
    if not shape:
      return a.at[indices].set(self._call(a.at[indices].get(), *args))

    args = tuple(_broadcast_to(arg, shape).ravel() for arg in args)
    indices = [idx if isinstance(idx, slice) else _broadcast_to(idx, shape).ravel() for idx in indices]

    def scan_fun(carry, x):
      i, a = carry
      idx = tuple(ind if isinstance(ind, slice) else ind[i] for ind in indices)
      a = a.at[idx].set(self._call(a.at[idx].get(), *(arg[i] for arg in args)))
      return (i + 1, a), x
    carry, _ = jax.lax.scan(scan_fun, (0, a), None, len(indices[0]))
    return carry[1]

  @_wraps(np.ufunc.reduceat, module="numpy.ufunc")
  def reduceat(self, a, indices, axis=0, dtype=None, out=None):
    if self.nin != 2:
      raise ValueError("reduceat only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduceat only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduceat()")
    return self._reduceat_via_scan(a, indices, axis=axis, dtype=dtype)

  def _reduceat_via_scan(self, a, indices, axis=0, dtype=None):
    check_arraylike(f"{self.__name__}.reduceat", a, indices)
    a = lax_internal.asarray(a)
    idx_tuple = _eliminate_deprecated_list_indexing(indices)
    assert len(idx_tuple) == 1
    indices = idx_tuple[0]
    if a.ndim == 0:
      raise ValueError(f"reduceat: a must have 1 or more dimension, got {a.shape=}")
    if indices.ndim != 1:
      raise ValueError(f"reduceat: indices must be one-dimensional, got {indices.shape=}")
    if dtype is None:
      dtype = a.dtype
    if axis is None or isinstance(axis, (tuple, list)):
      raise ValueError("reduceat requires a single integer axis.")
    axis = canonicalize_axis(axis, a.ndim)
    out = take(a, indices, axis=axis)
    ind = jax.lax.expand_dims(append(indices, a.shape[axis]),
                              np.delete(np.arange(out.ndim), axis))
    ind_start = jax.lax.slice_in_dim(ind, 0, ind.shape[axis] - 1, axis=axis)
    ind_end = jax.lax.slice_in_dim(ind, 1, ind.shape[axis], axis=axis)
    def loop_body(i, out):
      return _where((i > ind_start) & (i < ind_end),
                    self._call(out, take(a, i.reshape(1), axis=axis)),
                    out)
    return jax.lax.fori_loop(0, a.shape[axis], loop_body, out)

  @_wraps(np.ufunc.outer, module="numpy.ufunc")
  def outer(self, A, B, /, **kwargs):
    if self.nin != 2:
      raise ValueError("outer only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("outer only supported for functions returning a single value")
    check_arraylike(f"{self.__name__}.outer", A, B)
    _ravel = lambda A: jax.lax.reshape(A, (np.size(A),))
    result = jax.vmap(jax.vmap(partial(self._call, **kwargs), (None, 0)), (0, None))(_ravel(A), _ravel(B))
    return result.reshape(*np.shape(A), *np.shape(B))


def frompyfunc(func, /, nin, nout, *, identity=None):
  """Create a JAX ufunc from an arbitrary JAX-compatible scalar function.

  Args:
    func : a callable that takes `nin` scalar arguments and return `nout` outputs.
    nin: integer specifying the number of scalar inputs
    nout: integer specifying the number of scalar outputs
    identity: (optional) a scalar specifying the identity of the operation, if any.

  Returns:
    wrapped : jax.numpy.ufunc wrapper of func.
  """
  # TODO(jakevdp): use functools.wraps or similar to wrap the docstring?
  return ufunc(func, nin, nout, identity=identity)
