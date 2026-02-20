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

# pytype: skip-file
"""Indexing code for jax.numpy."""

from __future__ import annotations


import enum
from functools import partial
import operator
import string

import numpy as np

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import literals
from jax._src.indexing import NDIndexer, StaticSliceIndexer, DynamicSliceIndexer
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import utils as lax_utils
from jax._src.numpy import einsum
from jax._src.numpy import error as jnp_error
from jax._src.numpy import lax_numpy
from jax._src.numpy import ufuncs
from jax._src.numpy import util
from jax._src.partition_spec import PartitionSpec
from jax._src.pjit import auto_axes
from jax._src.sharding_impls import canonicalize_sharding, NamedSharding
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.typing import Array, ArrayLike, Index, StaticScalar
from jax._src.util import canonicalize_axis, set_module, tuple_update

export = set_module('jax.numpy')


# Internal utilities for parsing and validating NumPy-style indices.









@export
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: StaticScalar | None = None,
) -> Array:
  """Take elements from an array.

  JAX implementation of :func:`numpy.take`, implemented in terms of
  :func:`jax.lax.gather`. JAX's behavior differs from NumPy in the case
  of out-of-bound indices; see the ``mode`` parameter below.

  Args:
    a: array from which to take values.
    indices: N-dimensional array of integer indices of values to take from the array.
    axis: the axis along which to take values. If not specified, the array will
      be flattened before indexing is applied.
    mode: Out-of-bounds indexing mode, either ``"fill"`` or ``"clip"``. The default
      ``mode="fill"`` returns invalid values (e.g. NaN) for out-of bounds indices;
      the ``fill_value`` argument gives control over this value. For more discussion
      of ``mode`` options, see :attr:`jax.numpy.ndarray.at`.
    fill_value: The fill value to return for out-of-bounds slices when mode is 'fill'.
      Ignored otherwise. Defaults to NaN for inexact types, the largest negative value for
      signed types, the largest positive value for unsigned types, and True for booleans.
    unique_indices: If True, the implementation will assume that the indices are unique
      after normalization of negative indices, which lets the compiler emit more efficient
      code during the backward pass. If set to True and normalized indices are not unique,
      the result is implementation-defined and may be non-deterministic.
    indices_are_sorted : If True, the implementation will assume that the indices are
      sorted in ascending order after normalization of negative indices, which can lead
      to more efficient execution on some backends. If set to True and normalized indices
      are not sorted, the output is implementation-defined.

  Returns:
    Array of values extracted from ``a``.

  See also:
    - :attr:`jax.numpy.ndarray.at`: take values via indexing syntax.
    - :func:`jax.numpy.take_along_axis`: take values along an axis

  Examples:
    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 6.]])
    >>> indices = jnp.array([2, 0])

    Passing no axis results in indexing into the flattened array:

    >>> jnp.take(x, indices)
    Array([3., 1.], dtype=float32)
    >>> x.ravel()[indices]  # equivalent indexing syntax
    Array([3., 1.], dtype=float32)

    Passing an axis results ind applying the index to every subarray along the axis:

    >>> jnp.take(x, indices, axis=1)
    Array([[3., 1.],
           [6., 4.]], dtype=float32)
    >>> x[:, indices]  # equivalent indexing syntax
    Array([[3., 1.],
           [6., 4.]], dtype=float32)

    Out-of-bound indices fill with invalid values. For float inputs, this is `NaN`:

    >>> jnp.take(x, indices, axis=0)
    Array([[nan, nan, nan],
           [ 1.,  2.,  3.]], dtype=float32)
    >>> x.at[indices].get(mode='fill', fill_value=jnp.nan)  # equivalent indexing syntax
    Array([[nan, nan, nan],
           [ 1.,  2.,  3.]], dtype=float32)

    This default out-of-bound behavior can be adjusted using the ``mode`` parameter, for
    example, we can instead clip to the last valid value:

    >>> jnp.take(x, indices, axis=0, mode='clip')
    Array([[4., 5., 6.],
           [1., 2., 3.]], dtype=float32)
    >>> x.at[indices].get(mode='clip')  # equivalent indexing syntax
    Array([[4., 5., 6.],
           [1., 2., 3.]], dtype=float32)
  """
  return _take(a, indices, None if axis is None else operator.index(axis), out,
               mode, unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
               fill_value=fill_value)


@api.jit(static_argnames=('axis', 'mode', 'unique_indices', 'indices_are_sorted', 'fill_value'))
def _take(a, indices, axis: int | None = None, out=None, mode=None,
          unique_indices=False, indices_are_sorted=False, fill_value=None):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.take is not supported.")
  a, indices = util.ensure_arraylike("take", a, indices)

  if axis is None:
    a = a.ravel()
    axis_idx = 0
  else:
    axis_idx = canonicalize_axis(axis, np.ndim(a))

  if mode is None or mode == "fill":
    gather_mode = slicing.GatherScatterMode.FILL_OR_DROP
    # lax.gather() does not support negative indices, so we wrap them here
    indices = util._where(indices < 0, indices + a.shape[axis_idx], indices)
  elif mode == "raise":
    # TODO(phawkins): we have no way to report out of bounds errors yet.
    raise NotImplementedError("The 'raise' mode to jnp.take is not supported.")
  elif mode == "wrap":
    indices = ufuncs.mod(indices, lax._const(indices, a.shape[axis_idx]))
    gather_mode = slicing.GatherScatterMode.PROMISE_IN_BOUNDS
  elif mode == "clip":
    gather_mode = slicing.GatherScatterMode.CLIP
  else:
    raise ValueError(f"Invalid mode '{mode}' for np.take")

  index_dims = len(np.shape(indices))
  slice_sizes = list(np.shape(a))
  if slice_sizes[axis_idx] == 0:
    if indices.size != 0:
      raise IndexError("Cannot do a non-empty jnp.take() from an empty axis.")
    return a

  if indices.size == 0:
    out_shape = (slice_sizes[:axis_idx] + list(indices.shape) +
                 slice_sizes[axis_idx + 1:])
    return lax.full_like(a, 0, shape=out_shape)

  slice_sizes[axis_idx] = 1
  dnums = slicing.GatherDimensionNumbers(
    offset_dims=tuple(
      list(range(axis_idx)) +
      list(range(axis_idx + index_dims, len(a.shape) + index_dims - 1))),
    collapsed_slice_dims=(axis_idx,),
    start_index_map=(axis_idx,))
  return slicing.gather(a, indices[..., None], dimension_numbers=dnums,
                        slice_sizes=tuple(slice_sizes),
                        mode=gather_mode, unique_indices=unique_indices,
                        indices_are_sorted=indices_are_sorted, fill_value=fill_value)


def _normalize_index(index, axis_size):
  """Normalizes an index value in the range [-N, N) to the range [0, N)."""
  if dtypes.issubdtype(dtypes.dtype(index), np.unsignedinteger):
    return index
  if core.is_constant_dim(axis_size):
    axis_size_val = lax._const(index, axis_size)
  else:
    axis_size_val = lax.convert_element_type(core.dimension_as_value(axis_size),
                                             dtypes.dtype(index))
  if isinstance(index, (int, np.integer)):
    return lax.add(index, axis_size_val) if index < 0 else index
  else:
    return lax.select(index < 0, lax.add(index, axis_size_val), index)


@export
@api.jit(static_argnames=('axis', 'mode', 'fill_value'))
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None = -1,
    mode: str | slicing.GatherScatterMode | None = None,
    fill_value: StaticScalar | None = None,
) -> Array:
  """Take elements from an array.

  JAX implementation of :func:`numpy.take_along_axis`, implemented in
  terms of :func:`jax.lax.gather`. JAX's behavior differs from NumPy
  in the case of out-of-bound indices; see the ``mode`` parameter below.

  Args:
    a: array from which to take values.
    indices: array of integer indices. If ``axis`` is ``None``, must be one-dimensional.
      If ``axis`` is not None, must have ``a.ndim == indices.ndim``, and ``a`` must be
      broadcast-compatible with ``indices`` along dimensions other than ``axis``.
    axis: the axis along which to take values. If not specified, the array will
      be flattened before indexing is applied.
    mode: Out-of-bounds indexing mode, either ``"fill"`` or ``"clip"``. The default
      ``mode="fill"`` returns invalid values (e.g. NaN) for out-of bounds indices.
      For more discussion of ``mode`` options, see :attr:`jax.numpy.ndarray.at`.

  Returns:
    Array of values extracted from ``a``.

  See also:
    - :attr:`jax.numpy.ndarray.at`: take values via indexing syntax.
    - :func:`jax.numpy.take`: take the same indices along every axis slice.

  Examples:
    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 6.]])
    >>> indices = jnp.array([[0, 2],
    ...                      [1, 0]])
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[1., 3.],
           [5., 4.]], dtype=float32)
    >>> x[jnp.arange(2)[:, None], indices]  # equivalent via indexing syntax
    Array([[1., 3.],
           [5., 4.]], dtype=float32)

    Out-of-bound indices fill with invalid values. For float inputs, this is `NaN`:

    >>> indices = jnp.array([[1, 0, 2]])
    >>> jnp.take_along_axis(x, indices, axis=0)
    Array([[ 4.,  2., nan]], dtype=float32)
    >>> x.at[indices, jnp.arange(3)].get(
    ...     mode='fill', fill_value=jnp.nan)  # equivalent via indexing syntax
    Array([[ 4.,  2., nan]], dtype=float32)

    ``take_along_axis`` is helpful for extracting values from multi-dimensional
    argsorts and arg reductions. For, here we compute :func:`~jax.numpy.argsort`
    indices along an axis, and use ``take_along_axis`` to construct the sorted
    array:

    >>> x = jnp.array([[5, 3, 4],
    ...                [2, 7, 6]])
    >>> indices = jnp.argsort(x, axis=1)
    >>> indices
    Array([[1, 2, 0],
           [0, 2, 1]], dtype=int32)
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[3, 4, 5],
           [2, 6, 7]], dtype=int32)

    Similarly, we can use :func:`~jax.numpy.argmin` with ``keepdims=True`` and
    use ``take_along_axis`` to extract the minimum value:

    >>> idx = jnp.argmin(x, axis=1, keepdims=True)
    >>> idx
    Array([[1],
           [0]], dtype=int32)
    >>> jnp.take_along_axis(x, idx, axis=1)
    Array([[3],
           [2]], dtype=int32)
  """
  a, indices = util.ensure_arraylike("take_along_axis", arr, indices)
  index_dtype = indices.dtype
  idx_shape = np.shape(indices)
  if not dtypes.issubdtype(index_dtype, np.integer):
    raise TypeError("take_along_axis indices must be of integer type, got "
                    f"{index_dtype}")
  if axis is None:
    if np.ndim(indices) != 1:
      msg = "take_along_axis indices must be 1D if axis=None, got shape {}"
      raise ValueError(msg.format(idx_shape))
    a = a.ravel()
    axis = 0
  rank = a.ndim
  if rank != np.ndim(indices):
    msg = "indices and arr must have the same number of dimensions; {} vs. {}"
    raise ValueError(msg.format(np.ndim(indices), a.ndim))
  axis_int = canonicalize_axis(axis, rank)

  def replace(tup, val):
    lst = list(tup)
    lst[axis_int] = val
    return tuple(lst)

  index_dtype = lax_utils.int_dtype_for_dim(a.shape, signed=True)
  indices = lax.convert_element_type(indices, index_dtype)

  axis_size = a.shape[axis_int]
  arr_shape = replace(a.shape, 1)
  out_shape = lax.broadcast_shapes(idx_shape, arr_shape)
  if axis_size == 0:
    return lax.full(out_shape, 0, a.dtype)

  if mode == "one_hot":
    from jax import nn  # pytype: disable=import-error

    indices = _normalize_index(indices, axis_size)
    hot = nn.one_hot(indices, axis_size, dtype=np.bool_)
    if a.ndim == 1:
      return einsum.einsum("...b,b->...", hot, a, preferred_element_type=a.dtype)
    if axis_int > len(string.ascii_letters) - 2:
      raise ValueError(
          "One Hot indexing is only supported for up to 50 leading dimensions."
      )
    labels = "".join([string.ascii_letters[i] for i in range(axis_int)])
    eq = labels + "y...z," + labels + "z...->" + labels + "y..."
    return einsum.einsum(
        eq,
        hot,
        a,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=a.dtype,
    )

  index_dims = [i for i, idx in enumerate(idx_shape) if i == axis_int or not core.definitely_equal(idx, 1)]

  gather_index_shape = tuple(np.array(out_shape)[index_dims]) + (1,)
  gather_indices = []
  slice_sizes = []
  offset_dims = []
  start_index_map = []
  collapsed_slice_dims = []
  operand_batching_dims = []
  start_indices_batching_dims = []

  # We will squeeze the array. i is the index of the unsqueezed shape, while
  # new_i is the index of the squeezed shape. j is the index of the gather
  # indices.
  dims_to_squeeze = []
  new_i = 0
  j = 0
  for i in range(rank):
    if i == axis_int:
      if mode != 'promise_in_bounds':
        indices = _normalize_index(indices, axis_size)
      gather_indices.append(lax.reshape(indices, gather_index_shape))
      slice_sizes.append(1)
      start_index_map.append(new_i)
      collapsed_slice_dims.append(new_i)
      new_i += 1
      j += 1
    elif core.definitely_equal(idx_shape[i], 1):
      # If idx_shape[i] == 1, we can just take the entirety of the arr's axis
      # and avoid forming an iota index.
      offset_dims.append(i)
      slice_sizes.append(arr_shape[i])
      new_i += 1
    elif core.definitely_equal(arr_shape[i], 1):
      # If the array dimension is 1 but the index dimension is not, we will
      # squeeze this dimension.
      dims_to_squeeze.append(i)
      j += 1
    else:
      # Otherwise, idx_shape[i] == arr_shape[i]. Mark the dimensions in both
      # array and index as batching so corresponding elements are gathered.
      if core.definitely_equal(arr_shape[i], 0):
        slice_sizes.append(0)
      else:
        slice_sizes.append(1)
      operand_batching_dims.append(new_i)
      start_indices_batching_dims.append(j)
      new_i += 1
      j += 1

  # Squeeze a to remove singleton dimensions.
  a = lax.squeeze(a, dims_to_squeeze)
  gather_indices_arr = lax.concatenate(gather_indices, dimension=j)
  dnums = slicing.GatherDimensionNumbers(
    offset_dims=tuple(offset_dims),
    collapsed_slice_dims=tuple(collapsed_slice_dims),
    start_index_map=tuple(start_index_map),
    operand_batching_dims=tuple(operand_batching_dims),
    start_indices_batching_dims=tuple(start_indices_batching_dims))
  return slicing.gather(a, gather_indices_arr, dnums, tuple(slice_sizes),
                        mode="fill" if mode is None else mode, fill_value=fill_value)


def _make_along_axis_idx(shape, indices, axis):
  if axis < 0:
    axis += len(shape)
  return tuple_update(lax_numpy.indices(shape, sparse=True), axis, indices)


@export
@api.jit(static_argnames=('axis', 'inplace', 'mode'))
def put_along_axis(
  arr: ArrayLike,
  indices: ArrayLike,
  values: ArrayLike,
  axis: int | None,
  inplace: bool = True,
  *,
  mode: str | None = None,
) -> Array:
  """Put values into the destination array by matching 1d index and data slices.

  JAX implementation of :func:`numpy.put_along_axis`.

  The semantics of :func:`numpy.put_along_axis` are to modify arrays in-place, which
  is not possible for JAX's immutable arrays. The JAX version returns a modified
  copy of the input, and adds the ``inplace`` parameter which must be set to
  `False`` by the user as a reminder of this API difference.

  Args:
    arr: array into which values will be put.
    indices: array of indices at which to put values.
    values: array of values to put into the array.
    axis: the axis along which to put values. If not specified, the array will
      be flattened before indexing is applied.
    inplace: must be set to False to indicate that the input is not modified
      in-place, but rather a modified copy is returned.
    mode: Out-of-bounds indexing mode. For more discussion of ``mode`` options,
      see :attr:`jax.numpy.ndarray.at`.

  Returns:
    A copy of ``a`` with specified entries updated.

  See Also:
    - :func:`jax.numpy.put`: put elements into an array at given indices.
    - :func:`jax.numpy.place`: place elements into an array via boolean mask.
    - :func:`jax.numpy.ndarray.at`: array updates using NumPy-style indexing.
    - :func:`jax.numpy.take`: extract values from an array at given indices.
    - :func:`jax.numpy.take_along_axis`: extract values from an array along an axis.

  Examples:
    >>> from jax import numpy as jnp
    >>> a = jnp.array([[10, 30, 20], [60, 40, 50]])
    >>> i = jnp.argmax(a, axis=1, keepdims=True)
    >>> print(i)
    [[1]
     [0]]
    >>> b = jnp.put_along_axis(a, i, 99, axis=1, inplace=False)
    >>> print(b)
    [[10 99 20]
     [99 40 50]]
  """
  if inplace:
    raise ValueError(
      "jax.numpy.put_along_axis cannot modify arrays in-place, because JAX arrays"
      "are immutable. Pass inplace=False to instead return an updated array.")

  arr, indices, values = util.ensure_arraylike("put_along_axis", arr, indices, values)

  original_axis = axis
  original_arr_shape = arr.shape

  if axis is None:
    arr = arr.ravel()
    axis = 0

  if not arr.ndim == indices.ndim:
    raise ValueError(
      "put_along_axis arguments 'arr' and 'indices' must have same ndim. Got "
      f"{arr.ndim=} and {indices.ndim=}."
    )

  try:
    values = util._broadcast_to(values, indices.shape)
  except ValueError:
    raise ValueError(
      "put_along_axis argument 'values' must be broadcastable to 'indices'. Got "
      f"{values.shape=} and {indices.shape=}."
    )

  idx = _make_along_axis_idx(arr.shape, indices, axis)
  result = arr.at[idx].set(values, mode=mode)

  if original_axis is None:
    result = result.reshape(original_arr_shape)

  return result


### Indexing



class IndexingStrategy(enum.Enum):
  AUTO = 'auto'
  GATHER = 'gather'
  SCATTER = 'scatter'
  STATIC_SLICE = 'static_slice'
  DYNAMIC_SLICE = 'dynamic_slice'


def rewriting_take(
    arr: Array,
    idx: Index | tuple[Index, ...], *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | slicing.GatherScatterMode | None = None,
    fill_value: ArrayLike | None = None,
    normalize_indices: bool = True,
    out_sharding: NamedSharding | PartitionSpec | None = None,
    strategy: IndexingStrategy = IndexingStrategy.AUTO,
) -> Array:
  # Computes arr[idx].
  # All supported cases of indexing can be implemented as an XLA gather,
  # followed by an optional reverse and broadcast_in_dim.
  indexer = NDIndexer.from_raw_indices(idx, arr.shape)

  if not isinstance(strategy, IndexingStrategy):
    raise TypeError(f"Expected strategy to be IndexingStrategy; got {strategy}")

  if config.check_static_indices.value and (mode is None or slicing.GatherScatterMode.from_any(mode) == slicing.GatherScatterMode.PROMISE_IN_BOUNDS):
    indexer.validate_static_indices(normalize_indices=normalize_indices)

  if strategy == IndexingStrategy.STATIC_SLICE:
    static_slice_indexer = indexer.to_static_slice(
      arr_is_sharded=indexer.is_sharded(arr),
      normalize_indices=normalize_indices,
      mode=mode)
    return _static_slice(arr, static_slice_indexer)

  if strategy == IndexingStrategy.DYNAMIC_SLICE:
    dynamic_slice_indexer = indexer.to_dynamic_slice(
      arr_is_sharded=indexer.is_sharded(arr),
      normalize_indices=normalize_indices,
      mode=mode)
    return _dynamic_slice(arr, dynamic_slice_indexer)

  if strategy == IndexingStrategy.AUTO:
    # Attempt static slice first
    try:
      static_slice_indexer = indexer.to_static_slice(
        arr_is_sharded=indexer.is_sharded(arr),
        normalize_indices=normalize_indices,
        mode=mode)
    except (TypeError, ValueError, IndexError):
      pass
    else:
      return _static_slice(arr, static_slice_indexer)

    # Attempt dynamic slice next
    try:
      dynamic_slice_indexer = indexer.to_dynamic_slice(
        arr_is_sharded=indexer.is_sharded(arr),
        normalize_indices=normalize_indices,
        mode=mode)
    except (TypeError, ValueError, IndexError):
      pass
    else:
      return _dynamic_slice(arr, dynamic_slice_indexer)

  # In remaining cases, compute via gather.
  indexer = indexer.expand_bool_indices()
  dynamic_idx, treedef = tree_flatten(indexer)
  internal_gather = partial(
      _gather, treedef=treedef,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode, fill_value=fill_value, normalize_indices=normalize_indices)
  if out_sharding is not None:
    out_sharding = canonicalize_sharding(out_sharding, 'take')
    return auto_axes(internal_gather, out_sharding=out_sharding,
                     axes=out_sharding.mesh.explicit_axes,  # type: ignore
                     )(arr, dynamic_idx)
  return internal_gather(arr, dynamic_idx)


def _static_slice(arr: Array, indexer: StaticSliceIndexer) -> Array:
  """Equivalent of arr[idx] implemented in terms of static :func:`lax.slice` operations.

  This supports only INTEGER, ELLIPSIS, NONE, and SLICE indices, and will raise a
  TypeError if other indices are present.
  """
  if indexer.is_trivial_slice(arr.shape):
    result = arr
  else:
    result = slicing.slice(arr, indexer.start_indices,
                            indexer.limit_indices, indexer.strides)
  if indexer.rev_axes:
    result = lax.rev(result, indexer.rev_axes)
  if indexer.squeeze_axes:
    result = lax.squeeze(result, indexer.squeeze_axes)
  if indexer.newaxis_dims:
    result = lax.expand_dims(result, indexer.newaxis_dims)
  return result


def _dynamic_slice(arr: Array, indexer: DynamicSliceIndexer) -> Array:
  """Equivalent of arr[idx] implemented in terms of static :func:`lax.dynamic_slice`.

  This supports only INTEGER, ELLIPSIS, NONE, SLICE, and scalar ARRAY indices,
  and will raise a TypeError if other indices are present.
  """
  if indexer.trivial_slicing:
    result = arr
  else:
    result = slicing.dynamic_slice(
      arr,
      start_indices=indexer.start_indices,
      slice_sizes=indexer.slice_sizes,
      allow_negative_indices=indexer.normalize_indices)
  if indexer.rev_axes:
    result = lax.rev(result, indexer.rev_axes)
  if indexer.squeeze_axes:
    result = lax.squeeze(result, indexer.squeeze_axes)
  if indexer.newaxis_dims:
    result = lax.expand_dims(result, indexer.newaxis_dims)
  return result


# TODO(phawkins): re-enable jit after fixing excessive recompilation for
# slice indexes (e.g., slice(0, 5, None), slice(10, 15, None), etc.).
# @api.jit(static_argnums=(1, 2))
def _gather(arr, dynamic_idx, *, treedef, indices_are_sorted,
            unique_indices, mode, fill_value, normalize_indices):
  parsed_idx = tree_unflatten(treedef, dynamic_idx)
  indexer = parsed_idx.to_gather(core.typeof(arr).sharding,
                                 normalize_indices=normalize_indices)
  jnp_error._check_precondition_oob_gather(arr.shape, indexer.gather_indices)
  y = arr

  if fill_value is not None:
    core.concrete_or_error(None, fill_value,
                           "fill_value argument to indexed get()")
    if np.ndim(fill_value) != 0:
      raise ValueError("fill_value argument to indexed get() must be a scalar")
    if isinstance(fill_value, (np.ndarray, literals.TypedNdArray)):
      fill_value = fill_value.item()

  if indexer.scalar_bool_dims:
    y = lax.expand_dims(y, indexer.scalar_bool_dims)

  # Avoid calling gather if the slice shape is empty, both as a fast path and to
  # handle cases like zeros(0)[array([], int32)].
  if core.is_empty_shape(indexer.slice_shape):
    return lax.full_like(y, 0, shape=indexer.slice_shape)

  # We avoid generating a gather when indexer.gather_indices.size is empty.
  if not core.is_empty_shape(indexer.gather_indices.shape):
    y = slicing.gather(
        y, indexer.gather_indices, indexer.dnums, indexer.gather_slice_shape,
        unique_indices=unique_indices or indexer.unique_indices,
        indices_are_sorted=indices_are_sorted or indexer.indices_are_sorted,
        mode=mode, fill_value=fill_value)

  # Reverses axes with negative strides.
  if indexer.reversed_y_dims:
    y = lax.rev(y, indexer.reversed_y_dims)
  # This adds np.newaxis/None dimensions.
  return lax.expand_dims(y, indexer.newaxis_dims)


@export
def place(arr: ArrayLike, mask: ArrayLike, vals: ArrayLike, *,
          inplace: bool = True) -> Array:
  """Update array elements based on a mask.

  JAX implementation of :func:`numpy.place`.

  The semantics of :func:`numpy.place` are to modify arrays in-place, which
  is not possible for JAX's immutable arrays. The JAX version returns a modified
  copy of the input, and adds the ``inplace`` parameter which must be set to
  `False`` by the user as a reminder of this API difference.

  Args:
    arr: array into which values will be placed.
    mask: boolean mask with the same size as ``arr``.
    vals: values to be inserted into ``arr`` at the locations indicated
      by mask. If too many values are supplied, they will be truncated.
      If not enough values are supplied, they will be repeated.
    inplace: must be set to False to indicate that the input is not modified
      in-place, but rather a modified copy is returned.

  Returns:
    A copy of ``arr`` with masked values set to entries from `vals`.

  See Also:
    - :func:`jax.numpy.put`: put elements into an array at numerical indices.
    - :func:`jax.numpy.ndarray.at`: array updates using NumPy-style indexing

  Examples:
    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> mask = (jnp.arange(x.size) % 3 == 0).reshape(x.shape)
    >>> mask
    Array([[ True, False, False,  True, False],
           [False,  True, False, False,  True],
           [False, False,  True, False, False]], dtype=bool)

    Placing a scalar value:

    >>> jnp.place(x, mask, 1, inplace=False)
    Array([[1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0]], dtype=int32)

    In this case, ``jnp.place`` is similar to the masked array update syntax:

    >>> x.at[mask].set(1)
    Array([[1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0]], dtype=int32)

    ``place`` differs when placing values from an array. The array is repeated
    to fill the masked entries:

    >>> vals = jnp.array([1, 3, 5])
    >>> jnp.place(x, mask, vals, inplace=False)
    Array([[1, 0, 0, 3, 0],
           [0, 5, 0, 0, 1],
           [0, 0, 3, 0, 0]], dtype=int32)
  """
  data, mask_arr, vals_arr = util.ensure_arraylike("place", arr, mask, vals)
  vals_arr = vals_arr.ravel()
  if inplace:
    raise ValueError(
      "jax.numpy.place cannot modify arrays in-place, because JAX arrays are immutable. "
      "Pass inplace=False to instead return an updated array.")
  if data.size != mask_arr.size:
    raise ValueError("place: arr and mask must be the same size")
  if not vals_arr.size:
    raise ValueError("Cannot place values from an empty array")
  if not data.size:
    return data
  indices = lax_numpy.where(mask_arr.ravel(), size=mask_arr.size, fill_value=mask_arr.size)[0]
  vals_arr = lax_numpy._tile_to_size(vals_arr, len(indices))
  return data.ravel().at[indices].set(vals_arr, mode='drop').reshape(data.shape)


@export
def put(a: ArrayLike, ind: ArrayLike, v: ArrayLike,
        mode: str | None = None, *, inplace: bool = True) -> Array:
  """Put elements into an array at given indices.

  JAX implementation of :func:`numpy.put`.

  The semantics of :func:`numpy.put` are to modify arrays in-place, which
  is not possible for JAX's immutable arrays. The JAX version returns a modified
  copy of the input, and adds the ``inplace`` parameter which must be set to
  `False`` by the user as a reminder of this API difference.

  Args:
    a: array into which values will be placed.
    ind: array of indices over the flattened array at which to put values.
    v: array of values to put into the array.
    mode: string specifying how to handle out-of-bound indices. Supported values:

      - ``"clip"`` (default): clip out-of-bound indices to the final index.
      - ``"wrap"``: wrap out-of-bound indices to the beginning of the array.

    inplace: must be set to False to indicate that the input is not modified
      in-place, but rather a modified copy is returned.

  Returns:
    A copy of ``a`` with specified entries updated.

  See Also:
    - :func:`jax.numpy.place`: place elements into an array via boolean mask.
    - :func:`jax.numpy.ndarray.at`: array updates using NumPy-style indexing.
    - :func:`jax.numpy.take`: extract values from an array at given indices.

  Examples:
    >>> x = jnp.zeros(5, dtype=int)
    >>> indices = jnp.array([0, 2, 4])
    >>> values = jnp.array([10, 20, 30])
    >>> jnp.put(x, indices, values, inplace=False)
    Array([10,  0, 20,  0, 30], dtype=int32)

    This is equivalent to the following :attr:`jax.numpy.ndarray.at` indexing syntax:

    >>> x.at[indices].set(values)
    Array([10,  0, 20,  0, 30], dtype=int32)

    There are two modes for handling out-of-bound indices. By default they are
    clipped:

    >>> indices = jnp.array([0, 2, 6])
    >>> jnp.put(x, indices, values, inplace=False, mode='clip')
    Array([10,  0, 20,  0, 30], dtype=int32)

    Alternatively, they can be wrapped to the beginning of the array:

    >>> jnp.put(x, indices, values, inplace=False, mode='wrap')
    Array([10,  30, 20,  0, 0], dtype=int32)

    For N-dimensional inputs, the indices refer to the flattened array:

    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> indices = jnp.array([0, 7, 14])
    >>> jnp.put(x, indices, values, inplace=False)
    Array([[10,  0,  0,  0,  0],
           [ 0,  0, 20,  0,  0],
           [ 0,  0,  0,  0, 30]], dtype=int32)
  """
  if inplace:
    raise ValueError(
      "jax.numpy.put cannot modify arrays in-place, because JAX arrays are immutable. "
      "Pass inplace=False to instead return an updated array.")
  arr, ind_arr, _ = util.ensure_arraylike("put", a, ind, v)
  ind_arr = ind_arr.ravel()
  v_arr = lax_numpy.ravel(v)
  if not arr.size or not ind_arr.size or not v_arr.size:
    return arr
  v_arr = lax_numpy._tile_to_size(v_arr, len(ind_arr))
  if mode is None:
    scatter_mode = "drop"
  elif mode == "clip":
    ind_arr = lax_numpy.clip(ind_arr, 0, arr.size - 1)
    scatter_mode = "promise_in_bounds"
  elif mode == "wrap":
    ind_arr = ind_arr % arr.size
    scatter_mode = "promise_in_bounds"
  elif mode == "raise":
    raise NotImplementedError("The 'raise' mode to jnp.put is not supported.")
  else:
    raise ValueError(f"mode should be one of 'wrap' or 'clip'; got {mode=}")
  return arr.at[lax_numpy.unravel_index(ind_arr, arr.shape)].set(v_arr, mode=scatter_mode)
