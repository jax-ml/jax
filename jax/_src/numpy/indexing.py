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

from functools import partial
import operator
import string
from typing import Any, NamedTuple, Sequence

import numpy as np

from jax import lax
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import errors
from jax._src.api import jit
from jax._src.lax import lax as lax_internal
from jax._src.numpy import einsum
from jax._src.numpy import error as jnp_error
from jax._src.numpy import lax_numpy
from jax._src.numpy import ufuncs
from jax._src.numpy import util
from jax._src.pjit import auto_axes
from jax._src.tree_util import tree_flatten
from jax._src.typing import Array, ArrayLike, StaticScalar
from jax._src.util import canonicalize_axis, safe_zip, set_module, tuple_update

export = set_module('jax.numpy')


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
    unique_indices: If True, the implementation will assume that the indices are unique,
      which can result in more efficient execution on some backends. If set to True and
      indices are not unique, the output is undefined.
    indices_are_sorted : If True, the implementation will assume that the indices are
      sorted in ascending order, which can lead to more efficient execution on some
      backends. If set to True and indices are not sorted, the output is undefined.

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


@partial(jit, static_argnames=('axis', 'mode', 'unique_indices', 'indices_are_sorted', 'fill_value'))
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
    gather_mode = lax.GatherScatterMode.FILL_OR_DROP
    # lax.gather() does not support negative indices, so we wrap them here
    indices = util._where(indices < 0, indices + a.shape[axis_idx], indices)
  elif mode == "raise":
    # TODO(phawkins): we have no way to report out of bounds errors yet.
    raise NotImplementedError("The 'raise' mode to jnp.take is not supported.")
  elif mode == "wrap":
    indices = ufuncs.mod(indices, lax_internal._const(indices, a.shape[axis_idx]))
    gather_mode = lax.GatherScatterMode.PROMISE_IN_BOUNDS
  elif mode == "clip":
    gather_mode = lax.GatherScatterMode.CLIP
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
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(
      list(range(axis_idx)) +
      list(range(axis_idx + index_dims, len(a.shape) + index_dims - 1))),
    collapsed_slice_dims=(axis_idx,),
    start_index_map=(axis_idx,))
  return lax.gather(a, indices[..., None], dimension_numbers=dnums,
                    slice_sizes=tuple(slice_sizes),
                    mode=gather_mode, unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted, fill_value=fill_value)


def _normalize_index(index, axis_size):
  """Normalizes an index value in the range [-N, N) to the range [0, N)."""
  if dtypes.issubdtype(dtypes.dtype(index, canonicalize=True), np.unsignedinteger):
    return index
  if core.is_constant_dim(axis_size):
    axis_size_val = lax_internal._const(index, axis_size)
  else:
    axis_size_val = lax.convert_element_type(core.dimension_as_value(axis_size),
                                             dtypes.dtype(index, canonicalize=True))
  if isinstance(index, (int, np.integer)):
    return lax.add(index, axis_size_val) if index < 0 else index
  else:
    return lax.select(index < 0, lax.add(index, axis_size_val), index)


@export
@partial(jit, static_argnames=('axis', 'mode', 'fill_value'))
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None,
    mode: str | lax.GatherScatterMode | None = None,
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
  index_dtype = dtypes.dtype(indices)
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

  use_64bit_index = any(not core.is_constant_dim(d) or d >= (1 << 31) for d in a.shape)
  index_dtype = np.dtype('int64' if use_64bit_index else 'int32')
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
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(offset_dims),
    collapsed_slice_dims=tuple(collapsed_slice_dims),
    start_index_map=tuple(start_index_map),
    operand_batching_dims=tuple(operand_batching_dims),
    start_indices_batching_dims=tuple(start_indices_batching_dims))
  return lax.gather(a, gather_indices_arr, dnums, tuple(slice_sizes),
                    mode="fill" if mode is None else mode, fill_value=fill_value)


def _make_along_axis_idx(shape, indices, axis):
  if axis < 0:
    axis += len(shape)
  return tuple_update(lax_numpy.indices(shape, sparse=True), axis, indices)


@export
@partial(jit, static_argnames=('axis', 'inplace', 'mode'))
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

def _is_integer_index(idx: Any) -> bool:
  return isinstance(idx, (int, np.integer)) and not isinstance(idx, (bool, np.bool_))

def _is_simple_reverse_slice(idx: Any) -> bool:
  return (isinstance(idx, slice) and
          idx.start is idx.stop is None and
          isinstance(idx.step, int) and idx.step == -1)

def _is_valid_integer_index_for_slice(idx, size, mode):
  if size == 0:
    return False
  if _is_integer_index(idx):
    return -size <= idx < size
  try:
    shape, dtype = np.shape(idx), dtypes.dtype(idx, canonicalize=True)
  except:
    return False
  if shape == () and np.issubdtype(dtype, np.integer):
    # For dynamic integer indices, semantics require promise_inbounds.
    return mode in [None, 'promise_inbounds']
  return False

def _is_contiguous_slice(idx):
  return (isinstance(idx, slice) and
          (idx.start is None or _is_integer_index(idx.start)) and
          (idx.stop is None or _is_integer_index(idx.stop)) and
          (idx.step is None or (_is_integer_index(idx.step) and idx.step == 1)))

def _attempt_rewriting_take_via_slice(arr: Array, idx: Any, mode: str | None,
                                      out_sharding=None) -> Array | None:
  # attempt to compute _rewriting_take via lax.slice(); return None if not possible.
  idx = idx if isinstance(idx, tuple) else (idx,)

  if not all(isinstance(i, int) for i in arr.shape):
    return None
  if any(i is None for i in idx):
    return None  # TODO(jakevdp): handle newaxis case
  # For symbolic dimensions fallback to gather
  if any(core.is_symbolic_dim(elt)
         for i in idx if isinstance(i, slice)
         for elt in (i.start, i.stop, i.step)):
    return None
  if any(i is Ellipsis for i in idx):
    # Remove ellipses and pad with trailing `slice(None)` if necessary.
    # Do this before checking against rank of `arr` so that `...` can
    # count as no dimensions at all (e.g. `my_1d_array[:, ...]` succeeds)
    idx = _canonicalize_tuple_index(arr.ndim, idx=idx)
  if len(idx) > arr.ndim:
    return None

  simple_revs = {i for i, ind in enumerate(idx) if _is_simple_reverse_slice(ind)}
  int_indices = {i for i, (ind, size) in enumerate(zip(idx, arr.shape))
                 if _is_valid_integer_index_for_slice(ind, size, mode)}
  contiguous_slices = {i for i, ind in enumerate(idx) if _is_contiguous_slice(ind)}

  # For sharded inputs, indexing (like x[0]) and partial slices (like x[:2] as
  # opposed to x[:]) lead to incorrect sharding semantics when computed via
  # dynamic_slice, so we fall back to gather.
  # TODO(yashkatariya): fix dynamic_slice with sharding
  is_sharded = (isinstance(arr, array.ArrayImpl) and
                not dispatch.is_single_device_sharding(arr.sharding))
  has_partial_slices = any(idx[i].indices(arr.shape[i]) != (0, arr.shape[i], 1)
                           for i in contiguous_slices)
  if is_sharded and (int_indices or has_partial_slices):
    return None

  if len(simple_revs) + len(int_indices) + len(contiguous_slices) != len(idx):
    return None

  if simple_revs:
    arr = lax.rev(arr, tuple(simple_revs))
    idx = tuple(slice(None) if i in simple_revs else ind
                for i, ind in enumerate(idx))
    contiguous_slices |= simple_revs

  if not (int_indices or has_partial_slices):
    return arr

  idx += (arr.ndim - len(idx)) * (slice(None),)
  start_indices: Sequence[ArrayLike] = []
  slice_sizes: list[int] = []
  allow_negative_indices: list[bool] = []

  for ind, size in safe_zip(idx, arr.shape):
    if isinstance(ind, slice):
      start, stop, step = ind.indices(size)
      assert step == 1  # checked above
      start_indices.append(start)
      slice_sizes.append(max(0, stop - start))
      allow_negative_indices.append(start < 0 or stop < 0)
    else:
      assert np.issubdtype(dtypes.dtype(ind), np.integer)  # checked above
      assert np.shape(ind) == ()  # checked above
      start_indices.append(ind)
      slice_sizes.append(1)
      allow_negative_indices.append(
          not isinstance(ind, (int, np.integer)) or bool(ind < 0))

  # Try to use static slicing when possible.
  if all(isinstance(i, (int, np.integer)) and i >= 0 for i in start_indices):
    int_start_indices = [int(i) for i in start_indices]  # type: ignore
    int_limit_indices = [i + s for i, s in zip(int_start_indices, slice_sizes)]
    arr = lax.slice(
        arr, start_indices=int_start_indices, limit_indices=int_limit_indices)
  else:
    # We must be careful with dtypes because dynamic_slice requires all
    # start indices to have matching types.
    if len(start_indices) > 1:
      start_indices = util.promote_dtypes(*start_indices)
    jnp_error._check_precondition_oob_dynamic_slice(
        arr.shape, start_indices, slice_sizes, allow_negative_indices
    )
    internal_ds = partial(lax.dynamic_slice, slice_sizes=slice_sizes,
                          allow_negative_indices=allow_negative_indices)
    if out_sharding is not None:
      arr = auto_axes(internal_ds, out_sharding=out_sharding)(arr, start_indices)
    else:
      arr = internal_ds(arr, start_indices)
  if int_indices:
    arr = lax.squeeze(arr, tuple(int_indices))
  return arr


def rewriting_take(arr, idx, indices_are_sorted=False, unique_indices=False,
                   mode=None, fill_value=None, out_sharding=None):
  # Computes arr[idx].
  # All supported cases of indexing can be implemented as an XLA gather,
  # followed by an optional reverse and broadcast_in_dim.

  # For simplicity of generated primitives, we call lax.dynamic_slice in the
  # simplest cases: i.e. non-dynamic arrays indexed with integers and slices.

  result = _attempt_rewriting_take_via_slice(arr, idx, mode, out_sharding)
  if result is not None:
    return result

  # TODO(mattjj,dougalm): expand dynamic shape indexing support
  if config.dynamic_shapes.value and arr.ndim > 0:
    try: aval = core.get_aval(idx)
    except: pass
    else:
      if (isinstance(aval, core.DShapedArray) and aval.shape == () and
          dtypes.issubdtype(aval.dtype, np.integer) and
          not dtypes.issubdtype(aval.dtype, dtypes.bool_) and
          isinstance(arr.shape[0], int)):
        return lax.dynamic_index_in_dim(arr, idx, keepdims=False)

  treedef, static_idx, dynamic_idx = split_index_for_jit(idx, arr.shape)
  internal_gather = partial(
      _gather, treedef=treedef, static_idx=static_idx,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode, fill_value=fill_value)
  if out_sharding is not None:
    return auto_axes(internal_gather, out_sharding=out_sharding
                     )(arr, dynamic_idx)
  return internal_gather(arr, dynamic_idx)


# TODO(phawkins): re-enable jit after fixing excessive recompilation for
# slice indexes (e.g., slice(0, 5, None), slice(10, 15, None), etc.).
# @partial(jit, static_argnums=(1, 2))
def _gather(arr, dynamic_idx, *, treedef, static_idx, indices_are_sorted,
            unique_indices, mode, fill_value):
  idx = merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx)
  indexer = index_to_gather(np.shape(arr), idx)  # shared with _scatter_update
  jnp_error._check_precondition_oob_gather(arr.shape, indexer.gather_indices)
  y = arr

  if fill_value is not None:
    core.concrete_or_error(None, fill_value,
                           "fill_value argument to indexed get()")
    if np.ndim(fill_value) != 0:
      raise ValueError("fill_value argument to indexed get() must be a scalar")
    if isinstance(fill_value, np.ndarray):
      fill_value = fill_value.item()

  if indexer.scalar_bool_dims:
    y = lax.expand_dims(y, indexer.scalar_bool_dims)

  # Avoid calling gather if the slice shape is empty, both as a fast path and to
  # handle cases like zeros(0)[array([], int32)].
  if core.is_empty_shape(indexer.slice_shape):
    return lax.full_like(y, 0, shape=indexer.slice_shape)

  # We avoid generating a gather when indexer.gather_indices.size is empty.
  if not core.is_empty_shape(indexer.gather_indices.shape):
    y = lax.gather(
        y, indexer.gather_indices, indexer.dnums, indexer.gather_slice_shape,
        unique_indices=unique_indices or indexer.unique_indices,
        indices_are_sorted=indices_are_sorted or indexer.indices_are_sorted,
        mode=mode, fill_value=fill_value)

  # Reverses axes with negative strides.
  if indexer.reversed_y_dims:
    y = lax.rev(y, indexer.reversed_y_dims)
  # This adds np.newaxis/None dimensions.
  return lax.expand_dims(y, indexer.newaxis_dims)


class _Indexer(NamedTuple):
  # The expected shape of the slice output.
  slice_shape: Sequence[int]
  # The slice shape to pass to lax.gather().
  gather_slice_shape: Sequence[int]
  # The gather indices to use.
  gather_indices: ArrayLike
  # A GatherDimensionNumbers object describing the gather to perform.
  dnums: lax.GatherDimensionNumbers

  # Are the gather_indices known to be non-overlapping and/or sorted?
  # (In practice, these translate to "there no advanced indices", because
  # only advanced indices could lead to index repetition.)
  unique_indices: bool
  indices_are_sorted: bool

  # Slice dimensions that have negative strides, and so must be reversed after
  # the gather.
  reversed_y_dims: Sequence[int]

  # Keep track of any axes created by `newaxis`. These must be inserted for
  # gathers and eliminated for scatters.
  newaxis_dims: Sequence[int]

  # Keep track of dimensions with scalar bool indices. These must be inserted
  # for gathers before performing other index operations.
  scalar_bool_dims: Sequence[int]


def split_index_for_jit(idx, shape):
  """Splits indices into necessarily-static and dynamic parts.

  Used to pass indices into `jit`-ted function.
  """
  # Convert list indices to tuples in cases (deprecated by NumPy.)
  idx = eliminate_deprecated_list_indexing(idx)
  if any(isinstance(i, str) for i in idx):
    raise TypeError(f"JAX does not support string indexing; got {idx=}")

  # Expand any (concrete) boolean indices. We can then use advanced integer
  # indexing logic to handle them.
  idx = _expand_bool_indices(idx, shape)

  leaves, treedef = tree_flatten(idx)
  dynamic = [None] * len(leaves)
  static = [None] * len(leaves)
  for i, x in enumerate(leaves):
    if x is Ellipsis:
      static[i] = x
    elif isinstance(x, slice):
      # slice objects aren't hashable.
      static[i] = (x.start, x.stop, x.step)
    else:
      dynamic[i] = x
  return treedef, tuple(static), dynamic

def merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx):
  """Recombines indices that were split by split_index_for_jit."""
  idx = []
  for s, d in zip(static_idx, dynamic_idx):
    if d is not None:
      idx.append(d)
    elif isinstance(s, tuple):
      idx.append(slice(s[0], s[1], s[2]))
    else:
      idx.append(s)
  return treedef.unflatten(idx)

def _int(aval):
  return not aval.shape and dtypes.issubdtype(aval.dtype, np.integer)

def _aval_or_none(x):
  try:
    return core.get_aval(x)
  except:
    return None

def index_to_gather(x_shape: Sequence[int], idx: Sequence[Any],
                    normalize_indices: bool = True) -> _Indexer:
  # Convert sequences to arrays
  idx = tuple(lax_numpy.asarray(i, dtype=None if i else int)
              if isinstance(i, Sequence) else i for i in idx)
  abstract_idx = [_aval_or_none(i) for i in idx]
  float_indices = [(i, val, aval) for i, (val, aval) in enumerate(zip(idx, abstract_idx))
                   if aval is not None and dtypes.issubdtype(aval, np.inexact)]

  # Check for float or complex indices:
  if float_indices:
    i, val, aval = float_indices[0]
    msg = ("Indexer must have integer or boolean type, got indexer "
           "with type {} at position {}, indexer value {}")
    raise TypeError(msg.format(aval.dtype.name, i, val))

  # Check whether advanced indices are contiguous. We must do this before
  # removing ellipses (https://github.com/jax-ml/jax/issues/25109)
  # If advanced idexing axes do not appear contiguously, NumPy semantics
  # move the advanced axes to the front.
  is_advanced, = np.nonzero([isinstance(e, (int, np.integer, Array, np.ndarray))
                             or lax_numpy.isscalar(e) for e in idx])
  advanced_axes_are_contiguous = np.all(np.diff(is_advanced) == 1)

  # Remove ellipses and add trailing slice(None)s.
  idx = _canonicalize_tuple_index(len(x_shape), idx)

  # Check for scalar boolean indexing: this requires inserting extra dimensions
  # before performing the rest of the logic.
  scalar_bool_dims: Sequence[int] = [n for n, i in enumerate(idx) if isinstance(i, bool)]
  if scalar_bool_dims:
    idx = tuple(np.arange(int(i)) if isinstance(i, bool) else i for i in idx)
    x_shape = list(x_shape)
    for i in sorted(scalar_bool_dims):
      x_shape.insert(i, 1)
    x_shape = tuple(x_shape)

  # Check for advanced indexing:
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

  advanced_indexes: Sequence[Array | np.ndarray] | None = None

  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes: Sequence[int] = []

  # The positions of the advanced indexes in x's shape.
  # collapsed, after None axes have been removed. See below.
  x_advanced_axes: Sequence[int] | None = None

  if _is_advanced_int_indexer(idx):
    idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
    advanced_pairs = (
      (lax_numpy.asarray(e), i, j) for j, (i, e) in enumerate(idx_no_nones)
      if lax_numpy.isscalar(e) or isinstance(e, (Sequence, Array, np.ndarray)))
    if normalize_indices:
      advanced_pairs = ((_normalize_index(e, x_shape[j]), i, j)
                        for e, i, j in advanced_pairs)
    advanced_indexes, idx_advanced_axes, x_advanced_axes = zip(*advanced_pairs)

  x_axis = 0  # Current axis in x.
  y_axis = 0  # Current axis in y, before collapsing. See below.
  collapsed_y_axis = 0  # Current axis in y, after collapsing.

  # Scatter dimension numbers.
  offset_dims: list[int] = []
  collapsed_slice_dims: list[int] = []
  start_index_map: list[int] = []

  use_64bit_index = (
    any(not core.is_constant_dim(d) or d >= (1 << 31) for d in x_shape) and
    config.enable_x64.value)
  index_dtype = np.dtype('int64') if use_64bit_index else np.dtype('int32')

  # Gather indices.
  # Pairs of (array, start_dim) values. These will be broadcast into
  # gather_indices_shape, with the array dimensions aligned to start_dim, and
  # then concatenated.
  gather_indices: list[tuple[Array, int]] = []
  gather_indices_shape: list[int] = []

  # We perform three transformations to y before the scatter op, in order:
  # First, y is broadcast to slice_shape. In general `y` only need broadcast to
  # the right shape.
  slice_shape: list[int] = []

  # Next, y is squeezed to remove newaxis_dims. This removes np.newaxis/`None`
  # indices, which the scatter cannot remove itself.
  newaxis_dims: list[int] = []

  # Finally, we reverse reversed_y_dims to handle slices with negative strides.
  reversed_y_dims: list[int] = []

  gather_slice_shape: list[int] = []

  for idx_pos, i in enumerate(idx):
    # Handle the advanced indices here if:
    # * the advanced indices were not contiguous and we are the start.
    # * we are at the position of the first advanced index.
    if (advanced_indexes is not None and
        (advanced_axes_are_contiguous and idx_pos == idx_advanced_axes[0] or
         not advanced_axes_are_contiguous and idx_pos == 0)):
      advanced_index_arrs = util._broadcast_arrays(*advanced_indexes)
      shape = advanced_index_arrs[0].shape
      ndim = len(shape)

      start_dim = len(gather_indices_shape)
      gather_indices.extend(
          (lax.convert_element_type(a, index_dtype), start_dim)
          for a in advanced_index_arrs
      )
      gather_indices_shape += shape

      assert x_advanced_axes is not None
      start_index_map.extend(x_advanced_axes)
      collapsed_slice_dims.extend(x_advanced_axes)
      slice_shape.extend(shape)
      y_axis += ndim
      collapsed_y_axis += ndim

    # Per-index bookkeeping for advanced indexes.
    if idx_pos in idx_advanced_axes:
      x_axis += 1
      gather_slice_shape.append(1)
      continue

    # Handle basic int indexes.
    abstract_i = _aval_or_none(i)
    if isinstance(abstract_i, core.ShapedArray) and _int(abstract_i):
      if core.definitely_equal(x_shape[x_axis], 0):
        # XLA gives error when indexing into an axis of size 0
        raise IndexError(f"index is out of bounds for axis {x_axis} with size 0")
      i = _normalize_index(i, x_shape[x_axis]) if normalize_indices else i
      i_converted = lax.convert_element_type(i, index_dtype)
      gather_indices.append((i_converted, len(gather_indices_shape)))
      collapsed_slice_dims.append(x_axis)
      gather_slice_shape.append(1)
      start_index_map.append(x_axis)
      x_axis += 1
    # Handle np.newaxis (None)
    elif i is None:
      slice_shape.append(1)
      newaxis_dims.append(y_axis)
      y_axis += 1

    elif isinstance(i, slice):
      # Handle slice index (only static, otherwise an error is raised)
      if not all(_is_slice_element_none_or_constant_or_symbolic(elt)
                 for elt in (i.start, i.stop, i.step)):
        msg = ("Array slice indices must have static start/stop/step to be used "
               "with NumPy indexing syntax. "
               f"Found slice({i.start}, {i.stop}, {i.step}). "
               "To index a statically sized "
               "array at a dynamic position, try lax.dynamic_slice/"
               "dynamic_update_slice (JAX does not support dynamically sized "
               "arrays within JIT compiled functions).")
        raise IndexError(msg)

      start, step, slice_size = core.canonicalize_slice(i, x_shape[x_axis])
      slice_shape.append(slice_size)

      if core.definitely_equal(step, 1):
        # Avoid generating trivial gather (an optimization)
        if not core.definitely_equal(slice_size, x_shape[x_axis]):
          gather_indices.append((lax.convert_element_type(start, index_dtype),
                                len(gather_indices_shape)))
          start_index_map.append(x_axis)
        gather_slice_shape.append(slice_size)
        offset_dims.append(collapsed_y_axis)
      else:
        indices = (lax_numpy.array(start, dtype=index_dtype) +
                   lax_numpy.array(step, dtype=index_dtype) * lax.iota(index_dtype, slice_size))
        if step < 0:
          reversed_y_dims.append(collapsed_y_axis)
          indices = lax.rev(indices, dimensions=(0,))

        gather_slice_shape.append(1)
        gather_indices.append((indices, len(gather_indices_shape)))
        start_index_map.append(x_axis)
        gather_indices_shape.append(slice_size)
        collapsed_slice_dims.append(x_axis)

      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    else:
      if (abstract_i is not None and
          not (dtypes.issubdtype(abstract_i.dtype, np.integer) or dtypes.issubdtype(abstract_i.dtype, np.bool_))):
        msg = ("Indexer must have integer or boolean type, got indexer "
               "with type {} at position {}, indexer value {}")
        raise TypeError(msg.format(abstract_i.dtype.name, idx_pos, i))

      raise IndexError("Indexing mode not yet supported. Got unsupported indexer "
                      f"at position {idx_pos}: {i!r}")

  if len(gather_indices) == 0:
    gather_indices_array: ArrayLike = np.zeros((0,), dtype=index_dtype)
  elif len(gather_indices) == 1:
    g, _ = gather_indices[0]
    gather_indices_array = lax.expand_dims(g, (g.ndim,))
  else:
    last_dim = len(gather_indices_shape)
    gather_indices_shape.append(1)
    gather_indices_array = lax.concatenate([
      lax.broadcast_in_dim(g, gather_indices_shape, tuple(range(i, i + g.ndim)))
      for g, i in gather_indices],
      last_dim)

  dnums = lax.GatherDimensionNumbers(
    offset_dims = tuple(offset_dims),
    collapsed_slice_dims = tuple(sorted(collapsed_slice_dims)),
    start_index_map = tuple(start_index_map)
  )
  return _Indexer(
    slice_shape=slice_shape,
    newaxis_dims=tuple(newaxis_dims),
    gather_slice_shape=gather_slice_shape,
    reversed_y_dims=reversed_y_dims,
    dnums=dnums,
    gather_indices=gather_indices_array,
    unique_indices=advanced_indexes is None,
    indices_are_sorted=advanced_indexes is None,
    scalar_bool_dims=scalar_bool_dims)

def _should_unpack_list_index(x):
  """Helper for eliminate_deprecated_list_indexing."""
  return (isinstance(x, (np.ndarray, Array)) and np.ndim(x) != 0
          or isinstance(x, (Sequence, slice))
          or x is Ellipsis or x is None)

def eliminate_deprecated_list_indexing(idx):
  # "Basic slicing is initiated if the selection object is a non-array,
  # non-tuple sequence containing slice objects, [Ellipses, or newaxis
  # objects]". Detects this and raises a TypeError.
  if not isinstance(idx, tuple):
    if isinstance(idx, Sequence) and not isinstance(idx, (Array, np.ndarray, str)):
      # As of numpy 1.16, some non-tuple sequences of indices result in a warning, while
      # others are converted to arrays, based on a set of somewhat convoluted heuristics
      # (See https://github.com/numpy/numpy/blob/v1.19.2/numpy/core/src/multiarray/mapping.c#L179-L343)
      # In JAX, we raise an informative TypeError for *all* non-tuple sequences.
      if any(_should_unpack_list_index(i) for i in idx):
        msg = ("Using a non-tuple sequence for multidimensional indexing is not allowed; "
               "use `arr[tuple(seq)]` instead of `arr[seq]`. "
               "See https://github.com/jax-ml/jax/issues/4564 for more information.")
      else:
        msg = ("Using a non-tuple sequence for multidimensional indexing is not allowed; "
               "use `arr[array(seq)]` instead of `arr[seq]`. "
               "See https://github.com/jax-ml/jax/issues/4564 for more information.")
      raise TypeError(msg)
    else:
      idx = (idx,)
  return idx

def _is_boolean_index(i):
  try:
    abstract_i = core.get_aval(i)
  except TypeError:
    abstract_i = None
  return (isinstance(abstract_i, core.ShapedArray) and dtypes.issubdtype(abstract_i.dtype, np.bool_)
          or isinstance(i, list) and i and all(_is_scalar(e)
          and dtypes.issubdtype(dtypes.dtype(e), np.bool_) for e in i))

def _expand_bool_indices(idx, shape):
  """Converts concrete bool indexes into advanced integer indexes."""
  out = []
  total_dims = len(shape)
  num_ellipsis = sum(e is Ellipsis for e in idx)
  if num_ellipsis > 1:
    raise IndexError("an index can only have a single ellipsis ('...')")
  elif num_ellipsis == 1:
    total_dims = sum(np.ndim(e) if _is_boolean_index(e) else 1 for e in idx
                     if e is not None and e is not Ellipsis)
  ellipsis_offset = 0
  newaxis_offset = 0
  for dim_number, i in enumerate(idx):
    try:
      abstract_i = core.get_aval(i)
    except TypeError:
      abstract_i = None
    if _is_boolean_index(i):
      if isinstance(i, list):
        i = lax_numpy.array(i)
        abstract_i = core.get_aval(i)

      if not core.is_concrete(i):
        # TODO(mattjj): improve this error by tracking _why_ the indices are not concrete
        raise errors.NonConcreteBooleanIndexError(abstract_i)
      elif np.ndim(i) == 0:
        out.append(bool(i))
      else:
        i_shape = np.shape(i)
        start = len(out) + ellipsis_offset - newaxis_offset
        expected_shape = shape[start: start + np.ndim(i)]
        if len(i_shape) != len(expected_shape):
          raise IndexError(f"too many boolean indices at index {dim_number}: got mask of shape "
                           f"{i_shape}, but only {len(expected_shape)} dimensions remain.")
        if not all(s1 in (0, s2) for s1, s2 in zip(i_shape, expected_shape)):
          raise IndexError("boolean index did not match shape of indexed array in index "
                           f"{dim_number}: got {i_shape}, expected {expected_shape}")
        out.extend(np.where(i))
    else:
      out.append(i)
    if i is Ellipsis:
      ellipsis_offset = len(shape) - total_dims - 1
    if i is None:
      newaxis_offset += 1
  return tuple(out)


def _is_slice_element_none_or_constant_or_symbolic(elt):
  """Return True if elt is a constant or None."""
  if elt is None: return True
  if core.is_symbolic_dim(elt): return True
  try:
    return core.is_concrete(elt)
  except TypeError:
    return False

# TODO(mattjj): clean up this logic
def _is_advanced_int_indexer(idx):
  """Returns True if idx should trigger int array indexing, False otherwise."""
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
  assert isinstance(idx, tuple)
  if all(e is None or e is Ellipsis or isinstance(e, slice)
         or _is_scalar(e) and dtypes.issubdtype(dtypes.dtype(e), np.integer) for e in idx):
    return False
  return all(e is None or e is Ellipsis or isinstance(e, slice)
             or _is_int_arraylike(e) for e in idx)

def _is_int_arraylike(x):
  """Returns True if x is array-like with integer dtype, False otherwise."""
  return (isinstance(x, int) and not isinstance(x, bool)
          or dtypes.issubdtype(getattr(x, "dtype", None), np.integer)
          or isinstance(x, (list, tuple)) and all(_is_int_arraylike(e) for e in x))

def _is_scalar(x):
  """Checks if a Python or NumPy scalar."""
  return  np.isscalar(x) or (isinstance(x, (np.ndarray, Array))
                             and np.ndim(x) == 0)

def _canonicalize_tuple_index(arr_ndim, idx):
  """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
  num_dimensions_consumed = sum(not (e is None or e is Ellipsis or isinstance(e, bool)) for e in idx)
  if num_dimensions_consumed > arr_ndim:
    index_or_indices = "index" if num_dimensions_consumed == 1 else "indices"
    raise IndexError(
        f"Too many indices: {arr_ndim}-dimensional array indexed "
        f"with {num_dimensions_consumed} regular {index_or_indices}.")
  ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
  ellipsis_index = next(ellipses, None)
  if ellipsis_index is not None:
    if next(ellipses, None) is not None:
      raise IndexError(
          f"Multiple ellipses (...) not supported: {list(map(type, idx))}.")
    colons = (slice(None),) * (arr_ndim - num_dimensions_consumed)
    idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1:]
  elif num_dimensions_consumed < arr_ndim:
    colons = (slice(None),) * (arr_ndim - num_dimensions_consumed)
    idx = tuple(idx) + colons
  return idx


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
