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

from functools import partial
from typing import Sequence

import numpy as np

import jax
from jax._src import api
from jax._src import core
from jax._src import dtypes
from jax._src.numpy import util
from jax._src.util import canonicalize_axis, set_module
from jax._src.typing import Array, ArrayLike
from jax import lax

export = set_module('jax.numpy')

@export
@partial(api.jit, static_argnames=('axis', 'kind', 'order', 'stable', 'descending'))
def sort(
    a: ArrayLike,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array:
  """Return a sorted copy of an array.

  JAX implementation of :func:`numpy.sort`.

  Args:
    a: array to sort
    axis: integer axis along which to sort. Defaults to ``-1``, i.e. the last
      axis. If ``None``, then ``a`` is flattened before being sorted.
    stable: boolean specifying whether a stable sort should be used. Default=True.
    descending: boolean specifying whether to sort in descending order. Default=False.
    kind: deprecated; instead specify sort algorithm using stable=True or stable=False.
    order: not supported by JAX

  Returns:
    Sorted array of shape ``a.shape`` (if ``axis`` is an integer) or of shape
    ``(a.size,)`` (if ``axis`` is None).

  Examples:
    Simple 1-dimensional sort

    >>> x = jnp.array([1, 3, 5, 4, 2, 1])
    >>> jnp.sort(x)
    Array([1, 1, 2, 3, 4, 5], dtype=int32)

    Sort along the last axis of an array:

    >>> x = jnp.array([[2, 1, 3],
    ...                [4, 3, 6]])
    >>> jnp.sort(x, axis=1)
    Array([[1, 2, 3],
           [3, 4, 6]], dtype=int32)

  See also:
    - :func:`jax.numpy.argsort`: return indices of sorted values.
    - :func:`jax.numpy.lexsort`: lexicographical sort of multiple arrays.
    - :func:`jax.lax.sort`: lower-level function wrapping XLA's Sort operator.
  """
  arr = util.ensure_arraylike("sort", a)
  if kind is not None:
    raise TypeError("'kind' argument to sort is not supported. Use"
                    " stable=True or stable=False to specify sort stability.")
  if order is not None:
    raise TypeError("'order' argument to sort is not supported.")
  if axis is None:
    arr = arr.ravel()
    axis = 0
  dimension = canonicalize_axis(axis, arr.ndim)
  result = lax.sort(arr, dimension=dimension, is_stable=stable)
  return lax.rev(result, dimensions=[dimension]) if descending else result

@export
@partial(api.jit, static_argnames=('axis', 'kind', 'order', 'stable', 'descending'))
def argsort(
    a: ArrayLike,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array:
  """Return indices that sort an array.

  JAX implementation of :func:`numpy.argsort`.

  Args:
    a: array to sort
    axis: integer axis along which to sort. Defaults to ``-1``, i.e. the last
      axis. If ``None``, then ``a`` is flattened before being sorted.
    stable: boolean specifying whether a stable sort should be used. Default=True.
    descending: boolean specifying whether to sort in descending order. Default=False.
    kind: deprecated; instead specify sort algorithm using stable=True or stable=False.
    order: not supported by JAX

  Returns:
    Array of indices that sort an array. Returned array will be of shape ``a.shape``
    (if ``axis`` is an integer) or of shape ``(a.size,)`` (if ``axis`` is None).

  Examples:
    Simple 1-dimensional sort

    >>> x = jnp.array([1, 3, 5, 4, 2, 1])
    >>> indices = jnp.argsort(x)
    >>> indices
    Array([0, 5, 4, 1, 3, 2], dtype=int32)
    >>> x[indices]
    Array([1, 1, 2, 3, 4, 5], dtype=int32)

    Sort along the last axis of an array:

    >>> x = jnp.array([[2, 1, 3],
    ...                [6, 4, 3]])
    >>> indices = jnp.argsort(x, axis=1)
    >>> indices
    Array([[1, 0, 2],
           [2, 1, 0]], dtype=int32)
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[1, 2, 3],
           [3, 4, 6]], dtype=int32)


  See also:
    - :func:`jax.numpy.sort`: return sorted values directly.
    - :func:`jax.numpy.lexsort`: lexicographical sort of multiple arrays.
    - :func:`jax.lax.sort`: lower-level function wrapping XLA's Sort operator.
  """
  arr = util.ensure_arraylike("argsort", a)
  if kind is not None:
    raise TypeError("'kind' argument to argsort is not supported. Use"
                    " stable=True or stable=False to specify sort stability.")
  if order is not None:
    raise TypeError("'order' argument to argsort is not supported.")
  if axis is None:
    arr = arr.ravel()
    axis = 0
  dimension = canonicalize_axis(axis, arr.ndim)
  use_64bit_index = not core.is_constant_dim(arr.shape[dimension]) or arr.shape[dimension] >= (1 << 31)
  iota = lax.broadcasted_iota(np.dtype('int64') if use_64bit_index else dtypes.int_, arr.shape, dimension)
  # For stable descending sort, we reverse the array and indices to ensure that
  # duplicates remain in their original order when the final indices are reversed.
  # For non-stable descending sort, we can avoid these extra operations.
  if descending and stable:
    arr = lax.rev(arr, dimensions=[dimension])
    iota = lax.rev(iota, dimensions=[dimension])
  _, indices = lax.sort_key_val(arr, iota, dimension=dimension, is_stable=stable)
  return lax.rev(indices, dimensions=[dimension]) if descending else indices


@export
@partial(api.jit, static_argnames=['kth', 'axis'])
def partition(a: ArrayLike, kth: int, axis: int = -1) -> Array:
  """Returns a partially-sorted copy of an array.

  JAX implementation of :func:`numpy.partition`. The JAX version differs from
  NumPy in the treatment of NaN entries: NaNs which have the negative bit set
  are sorted to the beginning of the array.

  Args:
    a: array to be partitioned.
    kth: static integer index about which to partition the array.
    axis: static integer axis along which to partition the array; default is -1.

  Returns:
    A copy of ``a`` partitioned at the ``kth`` value along ``axis``. The entries
    before ``kth`` are values smaller than ``take(a, kth, axis)``, and entries
    after ``kth`` are indices of values larger than ``take(a, kth, axis)``

  Note:
    The JAX version requires the ``kth`` argument to be a static integer rather than
    a general array. This is implemented via two calls to :func:`jax.lax.top_k`. If
    you're only accessing the top or bottom k values of the output, it may be more
    efficient to call :func:`jax.lax.top_k` directly.

  See Also:
    - :func:`jax.numpy.sort`: full sort
    - :func:`jax.numpy.argpartition`: indirect partial sort
    - :func:`jax.lax.top_k`: directly find the top k entries
    - :func:`jax.lax.approx_max_k`: compute the approximate top k entries
    - :func:`jax.lax.approx_min_k`: compute the approximate bottom k entries

  Examples:
    >>> x = jnp.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3])
    >>> kth = 4
    >>> x_partitioned = jnp.partition(x, kth)
    >>> x_partitioned
    Array([1, 2, 3, 3, 4, 9, 8, 7, 6, 5], dtype=int32)

    The result is a partially-sorted copy of the input. All values before ``kth``
    are of smaller than the pivot value, and all values after ``kth`` are larger
    than the pivot value:

    >>> smallest_values = x_partitioned[:kth]
    >>> pivot_value = x_partitioned[kth]
    >>> largest_values = x_partitioned[kth + 1:]
    >>> print(smallest_values, pivot_value, largest_values)
    [1 2 3 3] 4 [9 8 7 6 5]

    Notice that among ``smallest_values`` and ``largest_values``, the returned
    order is arbitrary and implementation-dependent.
  """
  # TODO(jakevdp): handle NaN values like numpy.
  arr = util.ensure_arraylike("partition", a)
  if dtypes.issubdtype(arr.dtype, np.complexfloating):
    raise NotImplementedError("jnp.partition for complex dtype is not implemented.")
  axis = canonicalize_axis(axis, arr.ndim)
  kth = canonicalize_axis(kth, arr.shape[axis])

  arr = jax.numpy.swapaxes(arr, axis, -1)
  if dtypes.isdtype(arr.dtype, "unsigned integer"):
    # Here, we apply a trick to handle correctly 0 values for unsigned integers
    bottom = -lax.top_k(-(arr + 1), kth + 1)[0] - 1
  else:
    bottom = -lax.top_k(-arr, kth + 1)[0]
  top = lax.top_k(arr, arr.shape[-1] - kth - 1)[0]
  out = lax.concatenate([bottom, top], dimension=arr.ndim - 1)
  return jax.numpy.swapaxes(out, -1, axis)


@export
@partial(api.jit, static_argnames=['kth', 'axis'])
def argpartition(a: ArrayLike, kth: int, axis: int = -1) -> Array:
  """Returns indices that partially sort an array.

  JAX implementation of :func:`numpy.argpartition`. The JAX version differs from
  NumPy in the treatment of NaN entries: NaNs which have the negative bit set are
  sorted to the beginning of the array.

  Args:
    a: array to be partitioned.
    kth: static integer index about which to partition the array.
    axis: static integer axis along which to partition the array; default is -1.

  Returns:
    Indices which partition ``a`` at the ``kth`` value along ``axis``. The entries
    before ``kth`` are indices of values smaller than ``take(a, kth, axis)``, and
    entries after ``kth`` are indices of values larger than ``take(a, kth, axis)``

  Note:
    The JAX version requires the ``kth`` argument to be a static integer rather than
    a general array. This is implemented via two calls to :func:`jax.lax.top_k`. If
    you're only accessing the top or bottom k values of the output, it may be more
    efficient to call :func:`jax.lax.top_k` directly.

  See Also:
    - :func:`jax.numpy.partition`: direct partial sort
    - :func:`jax.numpy.argsort`: full indirect sort
    - :func:`jax.lax.top_k`: directly find the top k entries
    - :func:`jax.lax.approx_max_k`: compute the approximate top k entries
    - :func:`jax.lax.approx_min_k`: compute the approximate bottom k entries

  Examples:
    >>> x = jnp.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3])
    >>> kth = 4
    >>> idx = jnp.argpartition(x, kth)
    >>> idx
    Array([4, 8, 3, 9, 2, 0, 1, 5, 6, 7], dtype=int32)

    The result is a sequence of indices that partially sort the input. All indices
    before ``kth`` are of values smaller than the pivot value, and all indices
    after ``kth`` are of values larger than the pivot value:

    >>> x_partitioned = x[idx]
    >>> smallest_values = x_partitioned[:kth]
    >>> pivot_value = x_partitioned[kth]
    >>> largest_values = x_partitioned[kth + 1:]
    >>> print(smallest_values, pivot_value, largest_values)
    [1 2 3 3] 4 [6 8 9 7 5]

    Notice that among ``smallest_values`` and ``largest_values``, the returned
    order is arbitrary and implementation-dependent.
  """
  # TODO(jakevdp): handle NaN values like numpy.
  arr = util.ensure_arraylike("partition", a)
  if dtypes.issubdtype(arr.dtype, np.complexfloating):
    raise NotImplementedError("jnp.argpartition for complex dtype is not implemented.")
  axis = canonicalize_axis(axis, arr.ndim)
  kth = canonicalize_axis(kth, arr.shape[axis])

  arr = jax.numpy.swapaxes(arr, axis, -1)
  if dtypes.isdtype(arr.dtype, "unsigned integer"):
    # Here, we apply a trick to handle correctly 0 values for unsigned integers
    bottom_ind = lax.top_k(-(arr + 1), kth + 1)[1]
  else:
    bottom_ind = lax.top_k(-arr, kth + 1)[1]

  # To avoid issues with duplicate values, we compute the top indices via a proxy
  set_to_zero = lambda a, i: a.at[i].set(0)
  for _ in range(arr.ndim - 1):
    set_to_zero = jax.vmap(set_to_zero)
  proxy = set_to_zero(jax.numpy.ones(arr.shape), bottom_ind)
  top_ind = lax.top_k(proxy, arr.shape[-1] - kth - 1)[1]
  out = lax.concatenate([bottom_ind, top_ind], dimension=arr.ndim - 1)
  return jax.numpy.swapaxes(out, -1, axis)


@export
@api.jit
def sort_complex(a: ArrayLike) -> Array:
  """Return a sorted copy of complex array.

  JAX implementation of :func:`numpy.sort_complex`.

  Complex numbers are sorted lexicographically, meaning by their real part
  first, and then by their imaginary part if real parts are equal.

  Args:
    a: input array. If dtype is not complex, the array will be upcast to complex.

  Returns:
    A sorted array of the same shape and complex dtype as the input. If ``a``
    is multi-dimensional, it is sorted along the last axis.

  See also:
    - :func:`jax.numpy.sort`: Return a sorted copy of an array.

  Examples:
    >>> a = jnp.array([1+2j, 2+4j, 3-1j, 2+3j])
    >>> jnp.sort_complex(a)
    Array([1.+2.j, 2.+3.j, 2.+4.j, 3.-1.j], dtype=complex64)

    Multi-dimensional arrays are sorted along the last axis:

    >>> a = jnp.array([[5, 3, 4],
    ...                [6, 9, 2]])
    >>> jnp.sort_complex(a)
    Array([[3.+0.j, 4.+0.j, 5.+0.j],
           [2.+0.j, 6.+0.j, 9.+0.j]], dtype=complex64)
  """
  a = util.ensure_arraylike("sort_complex", a)
  a = lax.sort(a)
  return lax.convert_element_type(a, dtypes.to_complex_dtype(a.dtype))


@export
@partial(api.jit, static_argnames=('axis',))
def lexsort(keys: Array | np.ndarray | Sequence[ArrayLike], axis: int = -1) -> Array:
  """Sort a sequence of keys in lexicographic order.

  JAX implementation of :func:`numpy.lexsort`.

  Args:
    keys: a sequence of arrays to sort; all arrays must have the same shape.
      The last key in the sequence is used as the primary key.
    axis: the axis along which to sort (default: -1).

  Returns:
    An array of integers of shape ``keys[0].shape`` giving the indices of the
    entries in lexicographically-sorted order.

  See also:
    - :func:`jax.numpy.argsort`: sort a single entry by index.
    - :func:`jax.lax.sort`: direct XLA sorting API.

  Examples:
    :func:`lexsort` with a single key is equivalent to :func:`argsort`:

    >>> key1 = jnp.array([4, 2, 3, 2, 5])
    >>> jnp.lexsort([key1])
    Array([1, 3, 2, 0, 4], dtype=int32)
    >>> jnp.argsort(key1)
    Array([1, 3, 2, 0, 4], dtype=int32)

    With multiple keys, :func:`lexsort` uses the last key as the primary key:

    >>> key2 = jnp.array([2, 1, 1, 2, 2])
    >>> jnp.lexsort([key1, key2])
    Array([1, 2, 3, 0, 4], dtype=int32)

    The meaning of the indices become more clear when printing the sorted keys:

    >>> indices = jnp.lexsort([key1, key2])
    >>> print(f"{key1[indices]}\\n{key2[indices]}")
    [2 3 2 4 5]
    [1 1 2 2 2]

    Notice that the elements of ``key2`` appear in order, and within the sequences
    of duplicated values the corresponding elements of ```key1`` appear in order.

    For multi-dimensional inputs, :func:`lexsort` defaults to sorting along the
    last axis:

    >>> key1 = jnp.array([[2, 4, 2, 3],
    ...                   [3, 1, 2, 2]])
    >>> key2 = jnp.array([[1, 2, 1, 3],
    ...                   [2, 1, 2, 1]])
    >>> jnp.lexsort([key1, key2])
    Array([[0, 2, 1, 3],
           [1, 3, 2, 0]], dtype=int32)

    A different sort axis can be chosen using the ``axis`` keyword; here we sort
    along the leading axis:

    >>> jnp.lexsort([key1, key2], axis=0)
    Array([[0, 1, 0, 1],
           [1, 0, 1, 0]], dtype=int32)
  """
  key_arrays = util.ensure_arraylike_tuple("lexsort", tuple(keys))
  if len(key_arrays) == 0:
    raise TypeError("need sequence of keys with len > 0 in lexsort")
  if len({np.shape(key) for key in key_arrays}) > 1:
    raise ValueError("all keys need to be the same shape")
  if np.ndim(key_arrays[0]) == 0:
    return jax.numpy.array(0, dtype=dtypes.canonicalize_dtype(dtypes.int_))
  axis = canonicalize_axis(axis, np.ndim(key_arrays[0]))
  use_64bit_index = key_arrays[0].shape[axis] >= (1 << 31)
  iota = lax.broadcasted_iota(np.dtype('int64') if use_64bit_index else dtypes.int_,
                              np.shape(key_arrays[0]), axis)
  return lax.sort((*key_arrays[::-1], iota), dimension=axis, num_keys=len(key_arrays))[-1]
