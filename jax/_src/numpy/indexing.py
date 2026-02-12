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

import dataclasses
import enum
from functools import partial
import operator
import string
from typing import Any, NamedTuple
from collections.abc import Sequence

import numpy as np

from jax._src import api
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import errors
from jax._src import indexing
from jax._src import literals
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import utils as lax_utils
from jax._src.numpy import array_constructors
from jax._src.numpy import einsum
from jax._src.numpy import error as jnp_error
from jax._src.numpy import lax_numpy
from jax._src.numpy import ufuncs
from jax._src.numpy import util
from jax._src.partition_spec import PartitionSpec
from jax._src.pjit import auto_axes
from jax._src.sharding_impls import canonicalize_sharding, NamedSharding
from jax._src.tree_util import tree_flatten, tree_unflatten, register_pytree_node_class
from jax._src.typing import Array, ArrayLike, Index, StaticScalar
from jax._src.util import canonicalize_axis, safe_zip, set_module, tuple_update, unzip3

export = set_module('jax.numpy')


# Internal utilities for parsing and validating NumPy-style indices.

class IndexType(enum.Enum):
  """Enum for tracking the type of an index."""
  NONE = "none"
  SLICE = "slice"
  ELLIPSIS = "ellipsis"
  INTEGER = "integer"
  BOOLEAN = "boolean"
  ARRAY = "array"
  DYNAMIC_SLICE = "dynamic_slice"

  @classmethod
  def from_index(cls, idx: Index) -> IndexType:
    """Create an IndexType enum from a supported JAX array index."""
    if idx is None:
      return cls.NONE
    elif idx is Ellipsis:
      return cls.ELLIPSIS
    elif isinstance(idx, slice):
      return cls.SLICE
    elif isinstance(idx, indexing.Slice):
      return cls.DYNAMIC_SLICE
    elif _is_integer_index(idx):
      return cls.INTEGER
    elif _is_boolean_index(idx):
      return cls.BOOLEAN
    elif isinstance(idx, (Array, np.ndarray, literals.TypedNdArray)):
      if dtypes.issubdtype(idx.dtype, np.integer):
        return cls.ARRAY
      else:
        raise TypeError(
          f"Indexer must have integer or boolean type, got indexer with type {idx.dtype}")
    elif isinstance(idx, str):
      # TODO(jakevdp): this TypeError is for backward compatibility.
      # We should switch to IndexError for consistency.
      raise TypeError(f"JAX does not support string indexing; got {idx=}")
    elif isinstance(idx, Sequence):
      if not idx:  # empty indices default to float, so special-case this.
        return cls.ARRAY
      idx_aval = api.eval_shape(array_constructors.asarray, idx)
      if idx_aval.dtype == bool:
        return cls.BOOLEAN
      elif dtypes.issubdtype(idx_aval.dtype, np.integer):
        return cls.ARRAY
      else:
        raise TypeError(
          f"Indexer must have integer or boolean type, got indexer with type {idx_aval.dtype}")
    elif isinstance(idx, (float, complex, np.generic)):
      raise TypeError(
        f"Indexer must have integer or boolean type, got indexer with type {np.dtype(type(idx))}")
    else:
      raise IndexError("only integers, slices (`:`), ellipsis (`...`), newaxis (`None`)"
                       f" and integer or boolean arrays are valid indices. Got {idx}")


class ParsedIndex(NamedTuple):
  """Structure for tracking an indexer parsed within the context of an array shape."""
  index: Index  # type: ignore[assignment]  # seems to be a strange misfire by mypy.
  typ: IndexType
  consumed_axes: tuple[int, ...]


def _parse_indices(
    indices: tuple[Index, ...],
    shape: tuple[int, ...],
) -> list[ParsedIndex]:
  """Parse indices in the context of an array shape.

  Args:
    indices: a tuple of user-supplied indices to be parsed.
    shape: the shape of the array being indexed.

  Returns:
    The list of parsed indices stored in :class:`ParsedIndex` objects.
    This list will have the same length as ``indices``.

  Raises:
    IndexError: if any unrecognized index types are present or if there
      are too many indices, or too many ellipses.
  """
  # 1. go through indices to count the number of consumed dimensions.
  # This is required to determine the effect of any ellipses.
  dimensions_consumed: list[int] = []
  ellipses_indices: list[int] = []
  index_types: list[IndexType] = []
  for i, idx in enumerate(indices):
    typ = IndexType.from_index(idx)
    index_types.append(typ)

    if typ == IndexType.NONE:
      dimensions_consumed.append(0)
    elif typ == IndexType.ELLIPSIS:
      # We don't yet know how many dimensions are consumed, so set to zero
      # for now and update later.
      dimensions_consumed.append(0)
      ellipses_indices.append(i)
    elif typ == IndexType.BOOLEAN:
      dimensions_consumed.append(np.ndim(idx))  # type: ignore[arg-type]
    elif typ in [IndexType.INTEGER, IndexType.ARRAY, IndexType.SLICE, IndexType.DYNAMIC_SLICE]:
      dimensions_consumed.append(1)
    else:
      raise IndexError(f"Unrecognized index type: {typ}")

  # 2. Validate the consumed dimensions and ellipses.
  if len(ellipses_indices) > 1:
    raise IndexError("an index can only have a single ellipsis ('...')")
  total_consumed = sum(dimensions_consumed)
  if total_consumed > len(shape):
    raise IndexError(f"Too many indices: array is {len(shape)}-dimensional,"
                     f" but {total_consumed} were indexed")
  if ellipses_indices:
    dimensions_consumed[ellipses_indices[0]] = len(shape) - total_consumed

  # 3. Generate the final sequence of parsed indices.
  result: list[ParsedIndex] = []
  current_dim = 0
  for index, typ, n_consumed in safe_zip(indices, index_types, dimensions_consumed):
    consumed_axes = tuple(range(current_dim, current_dim + n_consumed))
    current_dim += len(consumed_axes)
    result.append(ParsedIndex(index=index, typ=typ, consumed_axes=consumed_axes))
  return result


@register_pytree_node_class
@dataclasses.dataclass(frozen=True, kw_only=True)
class NDIndexer:
  """Object that implements NumPy-style indexing operations on top of JAX.

  Generally this will be constructed via the :meth:`NDIndexer.from_raw_indices`
  method.

  Attributes:
    shape: the shape of the array being indexed.
    indices: a list of :class:`ParsedIndex` objects.
  """
  shape: tuple[int, ...]
  indices: list[ParsedIndex]

  @classmethod
  def from_raw_indices(cls, indices: Index | tuple[Index, ...], shape: tuple[int, ...]) -> NDIndexer:
    """Create an NDIndexer object from raw user-supplied indices."""
    indices = eliminate_deprecated_list_indexing(indices)
    parsed = _parse_indices(indices, shape)
    return cls(shape=shape, indices=parsed)

  def validate_static_indices(self, normalize_indices: bool = True) -> None:
    """Check that all static integer indices are in-bounds.

    Raises an IndexError in case of out-of-bound indices
    """
    for idx in self.indices:
      if idx.typ == IndexType.INTEGER:
        assert isinstance(idx.index, (int, np.integer))
        i = operator.index(idx.index)
        axis, = idx.consumed_axes
        size = self.shape[axis]
        normed_idx = i + size if normalize_indices and i < 0 else i
        if not 0 <= normed_idx < size:
          raise IndexError(f"index {i} out of bounds for axis {axis} with size {size}"
                           f" ({normalize_indices=})")

  def validate_slices(self) -> None:
    """Check that all slices have static start/stop/step values.

    Raises an IndexError in case of non-static entries.
    """
    for position, idx in enumerate(self.indices):
      if idx.typ == IndexType.SLICE:
        assert isinstance(idx.index, slice)
        if not all(_is_slice_element_none_or_constant_or_symbolic(val)
                   for val in [idx.index.start, idx.index.stop, idx.index.step]):
          raise IndexError("Slice entries must be static integers."
                          f" Got {idx.index} at position {position}")

  @staticmethod
  def is_sharded(arr) -> bool:
    """Check whether the array is sharded."""
    return isinstance(arr, array.ArrayImpl) and not dispatch.is_single_device_sharding(arr.sharding)

  def has_partial_slices(self) -> bool:
    """Check whether the indexer contains partial slices.

    For sharded arrays, partial slices cannot automatically propagate
    sharding.
    """
    for idx in self.indices:
      if idx.typ in [IndexType.INTEGER, IndexType.DYNAMIC_SLICE]:
        return True
      if idx.typ == IndexType.SLICE:
        slc = idx.index
        assert isinstance(slc, slice)
        axis, = idx.consumed_axes
        size = self.shape[axis]
        start, stop, step = slc.indices(self.shape[axis])
        if abs(step) != 1 or abs(stop - start) != size:
          return True
    return False

  def expand_bool_indices(self) -> NDIndexer:
    """Returns a new NDIndexer with boolean indices replaced by array indices.

    The only exception are scalar boolean indices, which are left in-place.
    """
    expanded_indices: list[ParsedIndex] = []

    for position, idx in enumerate(self.indices):
      if idx.typ != IndexType.BOOLEAN:
        expanded_indices.append(idx)
        continue
      if not core.is_concrete(idx.index):
        # TODO(mattjj): improve this error by tracking _why_ the indices are not concrete
        raise errors.NonConcreteBooleanIndexError(core.get_aval(idx.index))
      assert isinstance(idx.index, (bool, np.ndarray, Array, literals.TypedNdArray, list))
      if np.ndim(idx.index) == 0:  # pyrefly: ignore[bad-argument-type]
        # Scalar booleans
        assert idx.consumed_axes == ()
        expanded_indices.append(ParsedIndex(index=bool(idx.index), typ=idx.typ, consumed_axes=()))
        continue
      idx_shape = np.shape(idx.index)  # pyrefly: ignore[no-matching-overload]
      expected_shape = [self.shape[i] for i in idx.consumed_axes]
      if not all(s1 in (0, s2) for s1, s2 in zip(idx_shape, expected_shape)):
        raise IndexError("boolean index did not match shape of indexed array in index"
                        f" {position}: got {idx_shape}, expected {expected_shape}")
      expanded_indices_raw = np.where(np.asarray(idx.index))
      expanded_indices.extend(ParsedIndex(index=i, typ=IndexType.ARRAY, consumed_axes=(axis,))
                              for i, axis in safe_zip(expanded_indices_raw, idx.consumed_axes))
    return NDIndexer(shape=self.shape, indices=expanded_indices)

  def expand_scalar_bool_indices(self, sharding_spec: Any = None) -> tuple[NDIndexer, Any]:
    new_shape = list(self.shape)
    new_sharding_spec = list((None for _ in self.shape) if sharding_spec is None else sharding_spec)
    new_indices = list(self.indices)
    current_dim = 0
    for i, idx in enumerate(self.indices):
      if idx.typ == IndexType.BOOLEAN and np.ndim(idx.index) == 0:  # type: ignore[arg-type]
        new_shape.insert(i, 1)
        new_sharding_spec.insert(i, None)
        new_indices[i] = ParsedIndex(
          np.arange(int(idx.index)), typ=IndexType.ARRAY, consumed_axes=(current_dim,))  # type: ignore[arg-type]
        current_dim += 1
      else:
        n_consumed = len(idx.consumed_axes)
        new_indices[i] = ParsedIndex(
          index=idx.index,
          typ=idx.typ,
          consumed_axes = tuple(range(current_dim, current_dim + n_consumed))
        )
        current_dim += n_consumed
    new_sharding_spec = None if sharding_spec is None else tuple(new_sharding_spec)
    return NDIndexer(indices=new_indices, shape=tuple(new_shape)), new_sharding_spec

  def convert_sequences_to_arrays(self) -> NDIndexer:
    new_indices = [ParsedIndex(lax_numpy.asarray(idx.index), typ=idx.typ, consumed_axes=idx.consumed_axes)
                   if isinstance(idx.index, Sequence) else idx for idx in self.indices]
    return NDIndexer(indices=new_indices, shape=self.shape)

  def expand_ellipses(self) -> NDIndexer:
    """
    Returns a new indexer with ellipsis and implicit trailing slices
    replaced by explicit empty slices.
    """
    expanded: list[ParsedIndex] = []
    consumed = 0
    for idx in self.indices:
      consumed += len(idx.consumed_axes)
      if idx.typ == IndexType.ELLIPSIS:
        for axis in idx.consumed_axes:
          expanded.append(ParsedIndex(index=slice(None), typ=IndexType.SLICE, consumed_axes=(axis,)))
      else:
        expanded.append(idx)
    for axis in range(consumed, len(self.shape)):
      expanded.append(ParsedIndex(index=slice(None), typ=IndexType.SLICE, consumed_axes=(axis,)))
    return NDIndexer(shape=self.shape, indices=expanded)

  def normalize_indices(self) -> NDIndexer:
    new_indices: list[ParsedIndex] = []
    for idx in self.indices:
      if idx.typ == IndexType.INTEGER:
        axis, = idx.consumed_axes
        size: ArrayLike = self.shape[axis]
        if isinstance(idx.index, np.unsignedinteger):
          normed_index: Index = idx.index
        else:
          normed_index = idx.index + size if idx.index < 0 else idx.index  # type: ignore[assignment,operator]
        new_indices.append(ParsedIndex(normed_index, typ=idx.typ, consumed_axes=idx.consumed_axes))
      elif idx.typ == IndexType.ARRAY:
        assert isinstance(idx.index, (Array, np.ndarray, literals.TypedNdArray))
        axis, = idx.consumed_axes
        if dtypes.issubdtype(idx.index.dtype, np.unsignedinteger):
          normed_index = idx.index
        else:
          size = self.shape[axis]
          if core.is_constant_dim(size):
            size = lax._const(idx.index, size)
          else:
            size = lax.convert_element_type(core.dimension_as_value(size),
                                            idx.index.dtype)
          normed_index = lax.select(idx.index < 0, lax.add(idx.index, size), idx.index)
        new_indices.append(ParsedIndex(normed_index, typ=idx.typ, consumed_axes=idx.consumed_axes))
      else:
        new_indices.append(idx)
    return NDIndexer(indices=new_indices, shape=self.shape)

  def to_static_slice(
      self, *,
      arr_is_sharded: bool = False,
      normalize_indices: bool = True,
      mode: str | slicing.GatherScatterMode | None) -> _StaticSliceIndexer:
    """Convert to StaticSliceIndexer data structure.

    If this is not possible, raise a ValueError, TypeError, or IndexError.
    """
    if mode is None:
      parsed_mode = slicing.GatherScatterMode.PROMISE_IN_BOUNDS
    else:
      parsed_mode = slicing.GatherScatterMode.from_any(mode)
    if any(core.is_symbolic_dim(s) for s in self.shape):
      raise ValueError("mode='slice' is not valid for polymorphic shapes.")

    if parsed_mode not in [
        slicing.GatherScatterMode.PROMISE_IN_BOUNDS, slicing.GatherScatterMode.CLIP]:
      raise ValueError("static_slice requires mode='promise_in_bounds' or mode='clip'")

    # Validation of the unmodified user indices.
    if parsed_mode == slicing.GatherScatterMode.PROMISE_IN_BOUNDS:
      self.validate_static_indices(normalize_indices=normalize_indices)
    self.validate_slices()

    # For sharded inputs, indexing (like x[0]) and partial slices (like x[:2] as
    # opposed to x[:]) lead to incorrect sharding semantics when computed via slice.
    # TODO(yashkatariya): fix slice with sharding
    if arr_is_sharded and self.has_partial_slices():
      raise ValueError("static_slice with partial slices does not support nontrivial array sharding.")

    for position, pidx in enumerate(self.indices):
      if pidx.typ in [IndexType.INTEGER, IndexType.ELLIPSIS, IndexType.SLICE, IndexType.NONE]:
        pass
      elif pidx.typ in [IndexType.ARRAY, IndexType.BOOLEAN, IndexType.DYNAMIC_SLICE]:
        raise TypeError("static_slice: indices must be static scalars or slices."
                        f" Got index of type {type(pidx.index)} at position {position}")
      else:
        raise TypeError(f"static_slice: unrecognized index {pidx.index} at position {position}.")

    # Now re-iterate to generate static slices.
    start_indices: list[int] = []
    limit_indices: list[int] = []
    strides: list[int] = []
    rev_axes: list[int] = []
    squeeze_axes: list[int] = []
    newaxis_dims: list[int] = []

    expanded = self.expand_ellipses()
    for pidx in expanded.indices:
      if pidx.typ in [IndexType.ARRAY, IndexType.BOOLEAN, IndexType.ELLIPSIS]:
        raise RuntimeError(f"Internal: unexpected index encountered: {pidx}")
      elif pidx.typ == IndexType.NONE:
        # Expanded axes indices are based on the rank of the array after slicing
        # (tracked by start_indices) and squeezing (tracked by squeeze_axes), and
        # expand_dims inserts dimensions in order, so we must also account for
        # previous expanded dimensions.
        newaxis_dims.append(len(start_indices) - len(squeeze_axes) + len(newaxis_dims) )
      elif pidx.typ == IndexType.INTEGER:
        assert isinstance(pidx.index, (int, np.integer))
        axis, = pidx.consumed_axes
        if core.definitely_equal(self.shape[axis], 0):
          # XLA gives error when indexing into an axis of size 0
          raise IndexError(f"index is out of bounds for axis {axis} with size 0")
        start_index = int(pidx.index)
        if normalize_indices and start_index < 0:
          start_index += self.shape[axis]
        # Normalization & validation have already been handled, so clip start_index
        # to valid range
        start_index = min(max(start_index, 0), self.shape[axis] - 1)
        start_indices.append(start_index)
        limit_indices.append(start_index + 1)
        strides.append(1)
        squeeze_axes.append(axis)
      elif pidx.typ == IndexType.SLICE:
        assert isinstance(pidx.index, slice)
        axis, = pidx.consumed_axes
        size = self.shape[axis]
        start, stop, stride = pidx.index.indices(size)
        if stride < 0:
          new_start = min(size, stop + 1 + abs(start - stop - 1) % abs(stride))
          start_indices.append(new_start)
          limit_indices.append(max(new_start, start + 1))
          strides.append(abs(stride))
          rev_axes.append(axis)
        else:
          start_indices.append(start)
          limit_indices.append(max(start, stop))
          strides.append(stride)
      else:
        raise TypeError(f"static_slice: unrecognized index {pidx.index}")
    return _StaticSliceIndexer(
      start_indices=start_indices,
      limit_indices=limit_indices,
      strides=None if all(s == 1 for s in strides) else strides,
      rev_axes=rev_axes,
      squeeze_axes=squeeze_axes,
      newaxis_dims=newaxis_dims,
    )

  def to_dynamic_slice(
      self, *,
      arr_is_sharded: bool = False,
      normalize_indices: bool = True,
      mode: str | slicing.GatherScatterMode | None) -> _DynamicSliceIndexer:
    """Convert to DynamicSliceIndexer data structure.

    If this is not possible, raise a ValueError, TypeError, or IndexError.
    """
    if mode is not None:
      parsed_mode = slicing.GatherScatterMode.from_any(mode)
      if parsed_mode not in [
          slicing.GatherScatterMode.PROMISE_IN_BOUNDS, slicing.GatherScatterMode.CLIP]:
        raise ValueError("dynamic_slice requires mode='promise_in_bounds' or mode='clip'")

    # For sharded inputs, indexing (like x[0]) and partial slices (like x[:2] as
    # opposed to x[:]) lead to incorrect sharding semantics when computed via slice.
    # TODO(yashkatariya): fix slice with sharding
    if arr_is_sharded and self.has_partial_slices():
      raise ValueError("dynamic_slice with partial slices does not support nontrivial array sharding.")

    for position, pidx in enumerate(self.indices):
      if pidx.typ in [IndexType.INTEGER, IndexType.ELLIPSIS, IndexType.NONE]:
        pass
      elif pidx.typ == IndexType.DYNAMIC_SLICE:
        assert isinstance(pidx.index, indexing.Slice)
        if pidx.index.stride != 1:
          raise TypeError("dynamic_slice: only unit steps supported in slice."
                          f" Got {pidx.index} at position {position}")
      elif pidx.typ == IndexType.SLICE:
        assert isinstance(pidx.index, slice)
        if pidx.index.step is not None and pidx.index.step not in [-1, 1]:
          raise TypeError("dynamic_slice: only unit steps supported in slice."
                          f" Got {pidx.index} at position {position}")
      elif pidx.typ == IndexType.ARRAY:
        if isinstance(pidx.index, Sequence) or np.shape(pidx.index) != ():  # type: ignore[arg-type]
          raise TypeError("dynamic_slice: only scalar indices allowed."
                          f" Got index of type {type(pidx.index)} at position {position}")
      elif pidx.typ == IndexType.BOOLEAN:
        raise TypeError("dynamic_slice: indices must be scalars or slices."
                        f" Got index of type {type(pidx.index)} at position {position}")
      else:
        raise TypeError(f"dynamic_slice: unrecognized index {pidx.index} at position {position}.")

    start_indices: list[ArrayLike] = []
    slice_sizes: list[int] = []
    rev_axes: list[int] = []
    squeeze_axes: list[int] = []
    newaxis_dims: list[int] = []

    expanded = self.expand_ellipses()
    trivial_slicing = True
    for pidx in expanded.indices:
      if pidx.typ in [IndexType.BOOLEAN, IndexType.ELLIPSIS]:
        raise RuntimeError(f"Internal: unexpected index encountered: {pidx}")
      elif pidx.typ == IndexType.NONE:
        # Expanded axes indices are based on the rank of the array after slicing
        # (tracked by start_indices) and squeezing (tracked by squeeze_axes), and
        # expand_dims inserts dimensions in order, so we must also account for
        # previous expanded dimensions.
        newaxis_dims.append(len(start_indices) - len(squeeze_axes) + len(newaxis_dims))
      elif pidx.typ in [IndexType.INTEGER, IndexType.ARRAY]:
        trivial_slicing = False
        index = lax_numpy.asarray(pidx.index)
        assert index.shape == ()  # Validated above.
        axis, = pidx.consumed_axes
        if core.definitely_equal(self.shape[axis], 0):
          # XLA gives error when indexing into an axis of size 0
          raise IndexError(f"index is out of bounds for axis {axis} with size 0")
        start_indices.append(index)
        slice_sizes.append(1)
        squeeze_axes.append(axis)
      elif pidx.typ == IndexType.SLICE:
        assert isinstance(pidx.index, slice)
        if pidx.index != slice(None):
          trivial_slicing = False
        axis, = pidx.consumed_axes
        size = self.shape[axis]
        start, stop, stride = pidx.index.indices(size)
        assert stride in [-1, 1]  # validated above
        if stride < 0:
          new_start = stop + 1 + abs(start - stop - 1) % abs(stride)
          start_indices.append(new_start)
          slice_sizes.append(max(0, start + 1 - new_start))
          rev_axes.append(axis)
        else:
          start_indices.append(start)
          slice_sizes.append(max(0, stop - start))
      elif pidx.typ == IndexType.DYNAMIC_SLICE:
        assert isinstance(pidx.index, indexing.Slice)
        start_indices.append(pidx.index.start)
        slice_sizes.append(pidx.index.size)
        trivial_slicing = False
      else:
        raise TypeError(f"dynamic_slice: unrecognized index {pidx.index}")

    if len(start_indices) > 1:
      # We must be careful with dtypes because dynamic_slice requires all
      # start indices to have matching types.
      dt = lax_utils.int_dtype_for_shape(self.shape, signed=True)
      start_indices = [lax.convert_element_type(i, dt) for i in start_indices]

    return _DynamicSliceIndexer(
      start_indices=start_indices,
      slice_sizes=slice_sizes,
      rev_axes=rev_axes,
      squeeze_axes=squeeze_axes,
      newaxis_dims=newaxis_dims,
      normalize_indices=normalize_indices,
      trivial_slicing=trivial_slicing,
    )

  def is_advanced_int_indexer(self):
    """Returns True if idx should trigger int array indexing, False otherwise."""
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    return any(idx.typ in [IndexType.ARRAY, IndexType.BOOLEAN] and np.ndim(idx.index) > 0  # type: ignore[arg-type]
               for idx in self.indices)

  def to_gather(self, x_sharding: NamedSharding | Any,
                normalize_indices: bool = True) -> _GatherIndexer:
    return _index_to_gather(self, x_sharding=x_sharding, normalize_indices=normalize_indices)

  def tree_flatten(self):
    # split dynamic and static indices
    def is_dynamic(i: ParsedIndex):
      return i.typ in [IndexType.INTEGER, IndexType.ARRAY, IndexType.BOOLEAN]
    raw_dynamic_indices = [i.index if is_dynamic(i) else None for i in self.indices]
    static_metadata = [
      ParsedIndex(index=None, typ=i.typ, consumed_axes=i.consumed_axes) if is_dynamic(i) else i
      for i in self.indices]
    return raw_dynamic_indices, (self.shape, static_metadata)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    shape, static_metadata = aux_data
    indices = [idx if dyn_index is None else ParsedIndex(dyn_index, idx.typ, idx.consumed_axes)
               for dyn_index, idx in safe_zip(children, static_metadata)]
    return cls(indices=indices, shape=shape)


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

def _is_integer_index(idx: Any) -> bool:
  return isinstance(idx, (int, np.integer)) and not isinstance(idx, (bool, np.bool_))


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


def _static_slice(arr: Array, indexer: _StaticSliceIndexer) -> Array:
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


def _dynamic_slice(arr: Array, indexer: _DynamicSliceIndexer) -> Array:
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


class _StaticSliceIndexer(NamedTuple):
  start_indices: Sequence[int]
  limit_indices: Sequence[int]
  strides: Sequence[int] | None
  rev_axes: Sequence[int]
  squeeze_axes: Sequence[int]
  newaxis_dims: Sequence[int]

  def is_trivial_slice(self, arr_shape: Sequence[int]):
    if self.strides is not None or len(arr_shape) != len(self.start_indices):
      return False
    return all(
      (start, stop) == (0, size)
      for start, stop, size in zip(self.start_indices, self.limit_indices, arr_shape)
    )


class _DynamicSliceIndexer(NamedTuple):
    start_indices: Sequence[ArrayLike]
    slice_sizes: Sequence[int]
    rev_axes: Sequence[int]
    squeeze_axes: Sequence[int]
    newaxis_dims: Sequence[int]
    trivial_slicing: bool
    normalize_indices: bool


class _GatherIndexer(NamedTuple):
  # The expected shape of the slice output.
  slice_shape: Sequence[int]
  # The slice shape to pass to lax.gather().
  gather_slice_shape: Sequence[int]
  # The gather indices to use.
  gather_indices: ArrayLike
  # A GatherDimensionNumbers object describing the gather to perform.
  dnums: slicing.GatherDimensionNumbers

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

  # The expected sharding of the slice output.
  slice_sharding: NamedSharding | None = None


def _index_to_gather(indexer: NDIndexer, *, x_sharding: NamedSharding | Any,
                     normalize_indices: bool = True) -> _GatherIndexer:
  indexer.validate_slices()
  indexer = indexer.convert_sequences_to_arrays()

  is_advanced = np.nonzero(
    np.array([idx.typ in {IndexType.ARRAY, IndexType.INTEGER} for idx in indexer.indices]))
  advanced_axes_are_contiguous = np.all(np.diff(is_advanced) == 1)

  indexer = indexer.expand_ellipses()

  scalar_bool_dims: Sequence[int] = [n for n, i in enumerate(indexer.indices) if i.typ == IndexType.BOOLEAN]
  indexer, x_spec = indexer.expand_scalar_bool_indices(x_sharding.spec)

  if normalize_indices:
    indexer = indexer.normalize_indices()

  # Check for advanced indexing:
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

  # The advanced indices.
  advanced_indexes: Sequence[Array] = []

  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes: Sequence[int] = []

  # The positions of the advanced indexes in x's shape.
  # collapsed, after None axes have been removed. See below.
  x_advanced_axes: Sequence[int] = []

  if indexer.is_advanced_int_indexer():
    idx_without_none = [(i, d) for i, d in enumerate(indexer.indices) if d.typ != IndexType.NONE]
    advanced_pairs = (
      (lax_numpy.asarray(e.index), i, j)
      for j, (i, e) in enumerate(idx_without_none)
      if e.typ in [IndexType.ARRAY, IndexType.INTEGER]
    )
    advanced_indexes, idx_advanced_axes, x_advanced_axes = unzip3(advanced_pairs)

  x_axis = 0  # Current axis in x.
  y_axis = 0  # Current axis in y, before collapsing. See below.
  collapsed_y_axis = 0  # Current axis in y, after collapsing.

  # Scatter dimension numbers.
  offset_dims: list[int] = []
  collapsed_slice_dims: list[int] = []
  start_index_map: list[int] = []

  index_dtype = lax_utils.int_dtype_for_shape(indexer.shape, signed=True)

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
  slice_spec = []

  for idx_pos, index in enumerate(indexer.indices):
    # Handle the advanced indices here if:
    # * the advanced indices were not contiguous and we are the start.
    # * we are at the position of the first advanced index.
    if (advanced_indexes and
        (advanced_axes_are_contiguous and idx_pos == idx_advanced_axes[0] or
         not advanced_axes_are_contiguous and idx_pos == 0)):
      advanced_index_arrs = util._broadcast_arrays(*advanced_indexes)
      shape = advanced_index_arrs[0].shape
      aia_spec = core.typeof(advanced_index_arrs[0]).sharding.spec
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
      slice_spec.extend(aia_spec)
      y_axis += ndim
      collapsed_y_axis += ndim

    # Per-index bookkeeping for advanced indexes.
    if idx_pos in idx_advanced_axes:
      x_axis += 1
      gather_slice_shape.append(1)
      continue

    if index.typ in [IndexType.INTEGER, IndexType.ARRAY] and np.ndim(index.index) == 0:  # type: ignore[arg-type]
      # Basic scalar int indices
      if core.definitely_equal(indexer.shape[x_axis], 0):
        # XLA gives error when indexing into an axis of size 0
        raise IndexError(f"index is out of bounds for axis {x_axis} with size 0")
      i_converted = lax.convert_element_type(index.index, index_dtype)  # type: ignore[arg-type]
      gather_indices.append((i_converted, len(gather_indices_shape)))
      collapsed_slice_dims.append(x_axis)
      gather_slice_shape.append(1)
      start_index_map.append(x_axis)
      x_axis += 1

    elif index.typ == IndexType.NONE:
      # None indexing: add a dimension.
      slice_shape.append(1)
      slice_spec.append(None)
      newaxis_dims.append(y_axis)
      y_axis += 1

    elif index.typ in [IndexType.SLICE, IndexType.DYNAMIC_SLICE]:
      # Handle static slice index.
      if isinstance(index.index, indexing.Slice):
        start, step, slice_size = index.index.start, index.index.stride, index.index.size
      elif isinstance(index.index, slice):
        start, step, slice_size = core.canonicalize_slice(index.index, indexer.shape[x_axis])
      else:
        raise RuntimeError(f"Internal: expected slice or Slice, got {type(index.index)}")
      slice_shape.append(slice_size)
      slice_spec.append(x_spec[x_axis])

      if core.definitely_equal(step, 1):
        # Optimization: avoid generating trivial gather.
        if not core.definitely_equal(slice_size, indexer.shape[x_axis]):
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
      raise IndexError(f"Got unsupported indexer at position {idx_pos}: {index!r}")

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

  dnums = slicing.GatherDimensionNumbers(
    offset_dims = tuple(offset_dims),
    collapsed_slice_dims = tuple(sorted(collapsed_slice_dims)),
    start_index_map = tuple(start_index_map)
  )
  slice_sharding = x_sharding.update(spec=slice_spec)
  return _GatherIndexer(
    slice_shape=slice_shape,
    newaxis_dims=tuple(newaxis_dims),
    gather_slice_shape=gather_slice_shape,
    reversed_y_dims=reversed_y_dims,
    dnums=dnums,
    gather_indices=gather_indices_array,
    unique_indices=not advanced_indexes,
    indices_are_sorted=not advanced_indexes,
    scalar_bool_dims=scalar_bool_dims,
    slice_sharding=slice_sharding)

def _should_unpack_list_index(x):
  """Helper for eliminate_deprecated_list_indexing."""
  return (isinstance(x, (np.ndarray, Array, literals.TypedNdArray))
          and np.ndim(x) != 0
          or isinstance(x, (Sequence, slice))
          or x is Ellipsis or x is None)

def eliminate_deprecated_list_indexing(idx):
  # "Basic slicing is initiated if the selection object is a non-array,
  # non-tuple sequence containing slice objects, [Ellipses, or newaxis
  # objects]". Detects this and raises a TypeError.
  if not isinstance(idx, tuple):
    if isinstance(idx, Sequence) and not isinstance(
        idx, (Array, np.ndarray, literals.TypedNdArray, str)
    ):
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


def _is_slice_element_none_or_constant_or_symbolic(elt):
  """Return True if elt is a constant or None."""
  if elt is None: return True
  if core.is_symbolic_dim(elt): return True
  try:
    return core.is_concrete(elt)
  except TypeError:
    return False

def _is_scalar(x):
  """Checks if a Python or NumPy scalar."""
  return np.isscalar(x) or (
      isinstance(x, (np.ndarray, literals.TypedNdArray, Array))
      and np.ndim(x) == 0
  )


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
