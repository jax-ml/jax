# Copyright 2022 The JAX Authors.
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

from __future__ import annotations

from functools import partial
import math
import operator
from textwrap import dedent as _dedent
from typing import cast

import numpy as np

from jax import jit
from jax import lax

from jax._src import core
from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax._src.numpy.lax_numpy import (
    append, arange, array, asarray, concatenate, diff,
    empty, full_like, lexsort, moveaxis, nonzero, ones, ravel,
    sort, where, zeros)
from jax._src.numpy.reductions import any, cumsum
from jax._src.numpy.ufuncs import isnan
from jax._src.numpy.util import check_arraylike, _wraps
from jax._src.typing import Array, ArrayLike


_lax_const = lax_internal._const


@_wraps(np.in1d, lax_description="""
In the JAX version, the `assume_unique` argument is not referenced.
""")
def in1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False, invert: bool = False) -> Array:
  del assume_unique  # unused
  return _in1d(ar1, ar2, invert)

@partial(jit, static_argnames=('invert',))
def _in1d(ar1: ArrayLike, ar2: ArrayLike, invert: bool) -> Array:
  check_arraylike("in1d", ar1, ar2)
  ar1_flat = ravel(ar1)
  ar2_flat = ravel(ar2)
  # Note: an algorithm based on searchsorted has better scaling, but in practice
  # is very slow on accelerators because it relies on lax control flow. If XLA
  # ever supports binary search natively, we should switch to this:
  #   ar2_flat = jnp.sort(ar2_flat)
  #   ind = jnp.searchsorted(ar2_flat, ar1_flat)
  #   if invert:
  #     return ar1_flat != ar2_flat[ind]
  #   else:
  #     return ar1_flat == ar2_flat[ind]
  if invert:
    return (ar1_flat[:, None] != ar2_flat[None, :]).all(-1)
  else:
    return (ar1_flat[:, None] == ar2_flat[None, :]).any(-1)

@_wraps(np.setdiff1d,
  lax_description=_dedent("""
    Because the size of the output of ``setdiff1d`` is data-dependent, the function is not
    typically compatible with JIT. The JAX version adds the optional ``size`` argument which
    must be specified statically for ``jnp.setdiff1d`` to be used within some of JAX's
    transformations."""),
  extra_params=_dedent("""
    size : int, optional
        If specified, the first ``size`` elements of the result will be returned. If there are
        fewer elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value : array_like, optional
        When ``size`` is specified and there are fewer than the indicated number of elements, the
        remaining elements will be filled with ``fill_value``, which defaults to zero."""))
def setdiff1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False,
              *, size: int | None = None, fill_value: ArrayLike | None = None) -> Array:
  check_arraylike("setdiff1d", ar1, ar2)
  if size is None:
    ar1 = core.concrete_or_error(None, ar1, "The error arose in setdiff1d()")
  else:
    size = core.concrete_or_error(operator.index, size, "The error arose in setdiff1d()")
  arr1 = asarray(ar1)
  fill_value = asarray(0 if fill_value is None else fill_value, dtype=arr1.dtype)
  if arr1.size == 0:
    return full_like(arr1, fill_value, shape=size or 0)
  if not assume_unique:
    arr1 = cast(Array, unique(arr1, size=size and arr1.size))
  mask = in1d(arr1, ar2, invert=True)
  if size is None:
    return arr1[mask]
  else:
    if not (assume_unique or size is None):
      # Set mask to zero at locations corresponding to unique() padding.
      n_unique = arr1.size + 1 - (arr1 == arr1[0]).sum()
      mask = where(arange(arr1.size) < n_unique, mask, False)
    return where(arange(size) < mask.sum(), arr1[where(mask, size=size)], fill_value)


@_wraps(np.union1d,
  lax_description=_dedent("""
    Because the size of the output of ``union1d`` is data-dependent, the function is not
    typically compatible with JIT. The JAX version adds the optional ``size`` argument which
    must be specified statically for ``jnp.union1d`` to be used within some of JAX's
    transformations."""),
  extra_params=_dedent("""
    size : int, optional
        If specified, the first ``size`` elements of the result will be returned. If there are
        fewer elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value : array_like, optional
        When ``size`` is specified and there are fewer than the indicated number of elements, the
        remaining elements will be filled with ``fill_value``, which defaults to the minimum
        value of the union."""))
def union1d(ar1: ArrayLike, ar2: ArrayLike,
            *, size: int | None = None, fill_value: ArrayLike | None = None) -> Array:
  check_arraylike("union1d", ar1, ar2)
  if size is None:
    ar1 = core.concrete_or_error(None, ar1, "The error arose in union1d()")
    ar2 = core.concrete_or_error(None, ar2, "The error arose in union1d()")
  else:
    size = core.concrete_or_error(operator.index, size, "The error arose in union1d()")
  out = unique(concatenate((ar1, ar2), axis=None), size=size,
               fill_value=fill_value)
  return cast(Array, out)


@_wraps(np.setxor1d, lax_description="""
In the JAX version, the input arrays are explicitly flattened regardless
of assume_unique value.
""")
def setxor1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> Array:
  check_arraylike("setxor1d", ar1, ar2)
  ar1 = core.concrete_or_error(None, ar1, "The error arose in setxor1d()")
  ar2 = core.concrete_or_error(None, ar2, "The error arose in setxor1d()")

  ar1 = ravel(ar1)
  ar2 = ravel(ar2)

  if not assume_unique:
    ar1 = unique(ar1)
    ar2 = unique(ar2)

  aux = concatenate((ar1, ar2))
  if aux.size == 0:
    return aux

  aux = sort(aux)
  flag = concatenate((array([True]), aux[1:] != aux[:-1], array([True])))
  return aux[flag[1:] & flag[:-1]]


@partial(jit, static_argnames=['return_indices'])
def _intersect1d_sorted_mask(ar1: ArrayLike, ar2: ArrayLike, return_indices: bool = False) -> tuple[Array, ...]:
  """
    Helper function for intersect1d which is jit-able
    """
  ar = concatenate((ar1, ar2))
  if return_indices:
    iota = lax.broadcasted_iota(np.int64, np.shape(ar), dimension=0)
    aux, indices = lax.sort_key_val(ar, iota)
  else:
    aux = sort(ar)

  mask = aux[1:] == aux[:-1]
  if return_indices:
    return aux, mask, indices
  else:
    return aux, mask


@_wraps(np.intersect1d)
def intersect1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False,
                return_indices: bool = False) -> Array | tuple[Array, Array, Array]:
  check_arraylike("intersect1d", ar1, ar2)
  ar1 = core.concrete_or_error(None, ar1, "The error arose in intersect1d()")
  ar2 = core.concrete_or_error(None, ar2, "The error arose in intersect1d()")

  if not assume_unique:
    if return_indices:
      ar1, ind1 = unique(ar1, return_index=True)
      ar2, ind2 = unique(ar2, return_index=True)
    else:
      ar1 = unique(ar1)
      ar2 = unique(ar2)
  else:
    ar1 = ravel(ar1)
    ar2 = ravel(ar2)

  if return_indices:
    aux, mask, aux_sort_indices = _intersect1d_sorted_mask(ar1, ar2, return_indices)
  else:
    aux, mask = _intersect1d_sorted_mask(ar1, ar2, return_indices)

  int1d = aux[:-1][mask]

  if return_indices:
    ar1_indices = aux_sort_indices[:-1][mask]
    ar2_indices = aux_sort_indices[1:][mask] - np.size(ar1)
    if not assume_unique:
      ar1_indices = ind1[ar1_indices]
      ar2_indices = ind2[ar2_indices]

    return int1d, ar1_indices, ar2_indices
  else:
    return int1d


@_wraps(np.isin, lax_description="""
In the JAX version, the `assume_unique` argument is not referenced.
""")
def isin(element: ArrayLike, test_elements: ArrayLike,
         assume_unique: bool = False, invert: bool = False) -> Array:
  result = in1d(element, test_elements, assume_unique=assume_unique, invert=invert)
  return result.reshape(np.shape(element))


### SetOps

UNIQUE_SIZE_HINT = (
  "To make jnp.unique() compatible with JIT and other transforms, you can specify "
  "a concrete value for the size argument, which will determine the output size.")

@partial(jit, static_argnums=1)
def _unique_sorted_mask(ar: Array, axis: int) -> tuple[Array, Array, Array]:
  aux = moveaxis(ar, axis, 0)
  if np.issubdtype(aux.dtype, np.complexfloating):
    # Work around issue in sorting of complex numbers with Nan only in the
    # imaginary component. This can be removed if sorting in this situation
    # is fixed to match numpy.
    aux = where(isnan(aux), _lax_const(aux, np.nan), aux)
  size, *out_shape = aux.shape
  if math.prod(out_shape) == 0:
    size = 1
    perm = zeros(1, dtype=int)
  else:
    perm = lexsort(aux.reshape(size, math.prod(out_shape)).T[::-1])
  aux = aux[perm]
  if aux.size:
    if dtypes.issubdtype(aux.dtype, np.inexact):
      # This is appropriate for both float and complex due to the documented behavior of np.unique:
      # See https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/arraysetops.py#L212-L220
      neq = lambda x, y: lax.ne(x, y) & ~(isnan(x) & isnan(y))
    else:
      neq = lax.ne
    mask = ones(size, dtype=bool).at[1:].set(any(neq(aux[1:], aux[:-1]), tuple(range(1, aux.ndim))))
  else:
    mask = zeros(size, dtype=bool)
  return aux, mask, perm

def _unique(ar: Array, axis: int, return_index: bool = False, return_inverse: bool = False,
            return_counts: bool = False, size: int | None = None,
            fill_value: ArrayLike | None = None, return_true_size: bool = False
            ) -> Array | tuple[Array, ...]:
  """
  Find the unique elements of an array along a particular axis.
  """
  if ar.shape[axis] == 0 and size and fill_value is None:
    raise ValueError(
      "jnp.unique: for zero-sized input with nonzero size argument, fill_value must be specified")

  aux, mask, perm = _unique_sorted_mask(ar, axis)
  if size is None:
    ind = core.concrete_or_error(None, mask,
        "The error arose in jnp.unique(). " + UNIQUE_SIZE_HINT)
  else:
    ind = nonzero(mask, size=size)[0]
  result = aux[ind] if aux.size else aux
  if size is not None and fill_value is not None:
    fill_value = asarray(fill_value, dtype=result.dtype)
    if result.shape[0]:
      valid = lax.expand_dims(arange(size) < mask.sum(), tuple(range(1, result.ndim)))
      result = where(valid, result, fill_value)
    else:
      result = full_like(result, fill_value, shape=(size, *result.shape[1:]))
  result = moveaxis(result, 0, axis)

  ret: tuple[Array, ...] = (result,)
  if return_index:
    if aux.size:
      ret += (perm[ind],)
    else:
      ret += (perm,)
  if return_inverse:
    if aux.size:
      imask = cumsum(mask) - 1
      inv_idx = zeros(mask.shape, dtype=dtypes.canonicalize_dtype(dtypes.int_))
      inv_idx = inv_idx.at[perm].set(imask)
    else:
      inv_idx = zeros(ar.shape[axis], dtype=int)
    ret += (inv_idx,)
  if return_counts:
    if aux.size:
      if size is None:
        idx = append(nonzero(mask)[0], mask.size)
      else:
        idx = nonzero(mask, size=size + 1)[0]
        idx = idx.at[1:].set(where(idx[1:], idx[1:], mask.size))
      ret += (diff(idx),)
    elif ar.shape[axis]:
      ret += (array([ar.shape[axis]], dtype=dtypes.canonicalize_dtype(dtypes.int_)),)
    else:
      ret += (empty(0, dtype=int),)
  if return_true_size:
    # Useful for internal uses of unique().
    ret += (mask.sum(),)
  return ret[0] if len(ret) == 1 else ret

@_wraps(np.unique, skip_params=['axis'],
  lax_description=_dedent("""
    Because the size of the output of ``unique`` is data-dependent, the function is not
    typically compatible with JIT. The JAX version adds the optional ``size`` argument which
    must be specified statically for ``jnp.unique`` to be used within some of JAX's
    transformations."""),
  extra_params=_dedent("""
    size : int, optional
        If specified, the first ``size`` unique elements will be returned. If there are fewer unique
        elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value : array_like, optional
        When ``size`` is specified and there are fewer than the indicated number of elements, the
        remaining elements will be filled with ``fill_value``. The default is the minimum value
        along the specified axis of the input."""))
def unique(ar: ArrayLike, return_index: bool = False, return_inverse: bool = False,
           return_counts: bool = False, axis: int | None = None,
           *, size: int | None = None, fill_value: ArrayLike | None = None):
  check_arraylike("unique", ar)
  if size is None:
    ar = core.concrete_or_error(None, ar,
        "The error arose for the first argument of jnp.unique(). " + UNIQUE_SIZE_HINT)
  else:
    size = core.concrete_or_error(operator.index, size,
         "The error arose for the size argument of jnp.unique(). " + UNIQUE_SIZE_HINT)
  arr = asarray(ar)
  if axis is None:
    axis = 0
    arr = arr.flatten()
  axis_int: int = core.concrete_or_error(operator.index, axis, "axis argument of jnp.unique()")
  return _unique(arr, axis_int, return_index, return_inverse,
                 return_counts, size=size, fill_value=fill_value)
