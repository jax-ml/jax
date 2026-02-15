# Copyright 2026 The JAX Authors.
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

from collections.abc import Sequence
import dataclasses
import enum
import operator
from typing import Any, NamedTuple
import numpy as np

from jax._src import api
from jax._src import array
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import errors
from jax._src import literals
from jax._src import tree_util
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import utils as lax_utils
from jax._src.sharding_impls import NamedSharding
from jax._src.typing import Array, ArrayLike, Index
from jax._src.util import safe_zip, unzip3


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Slice:
  """A slice with a start index and a size.

  Both start index and size can either be static, i.e. known at tracing
  and compilation time, or dynamic.
  """

  start: int | Array
  size: int | Array
  stride: int = 1

  def __post_init__(self):
    if self.stride < 0:
      raise ValueError("`stride` must be >= 0.")

  @property
  def is_dynamic_start(self):
    return not core.is_dim(self.start)

  @property
  def is_dynamic_size(self):
    return not core.is_dim(self.size)

  def tree_flatten(self):
    # If `start` is statically known, we treat it as static information
    xs = ()
    data = ()
    xs += (self.start,) if self.is_dynamic_start else (None,)
    data += (None,) if self.is_dynamic_start else (self.start,)
    xs += (self.size,) if self.is_dynamic_size else (None,)
    data += (None,) if self.is_dynamic_size else (self.size,)
    data += (self.stride,)
    return xs, data

  @classmethod
  def tree_unflatten(cls, aux_data, children) -> Slice:
    start, size = (
        a if a is not None else b for a, b in zip(children, aux_data[:2])
    )
    return cls(start, size, aux_data[2])

  @classmethod
  def from_slice(cls, slc: slice, size: int) -> Slice:
    start, step, size = core.canonicalize_slice(slc, size)
    if step < 1:
      raise ValueError(f"slice must have a step >= 1 (found: {step})")
    return cls(start, size, step)


def dslice(
    start: int | Array | None,
    size: int | Array | None = None,
    stride: int | None = None,
) -> slice | Slice:
  """Constructs a ``Slice`` from a start index and a size.

  The semantics of ``dslice`` mirror those of the builtin ``slice`` type:

  * ``dslice(None)`` is ``:``
  * ``dslice(j)`` is ``:j``
  * ``dslice(i, j)`` is ``i:i+j``
  * ``dslice(i, j, stride)`` is ``i:i+j:stride``

  Examples:

    >>> x = jax.numpy.arange(10)
    >>> i = 4
    >>> x[i: i + 2]  # standard indexing requires i to be static
    Array([4, 5], dtype=int32)
    >>> x[jax.ds(i, 2)]  # equivalent which allows i to be dynamic
    Array([4, 5], dtype=int32)

    Here is an explicit example of slicing with a dynamic start index:

    >>> @jax.jit(static_argnames='size')
    ... def f(x, i, size):  # example of when `
    ...   return x[jax.ds(i, size)]
    ...
    >>> f(x, i, 2)
    Array([4, 5], dtype=int32)
  """
  if start is None:
    return slice(None)
  if stride is None:
    stride = 1
  if not isinstance(stride, int):
    raise ValueError("Non-static stride in `dslice`")
  if size is None:
    if not isinstance(start, int):
      raise ValueError("Non-static `dslice`")
    return Slice(0, start, stride)
  return Slice(start, size, stride)


ds = dslice  # Handy alias.


class StaticSliceIndexer(NamedTuple):
  """Data structure encoding general indexing via static slice."""
  # start_indices, limit_indices, and strides passed to lax.slice()
  start_indices: Sequence[int]
  limit_indices: Sequence[int]
  strides: Sequence[int] | None

  # axes of the sliced array to pass to lax.rev()
  rev_axes: Sequence[int]

  # axes of the sliced array to pass to lax.squeeze()
  squeeze_axes: Sequence[int]

  # axes of the sliced array to pass to lax.expand_dims()
  newaxis_dims: Sequence[int]

  def is_trivial_slice(self, arr_shape: Sequence[int]) -> bool:
    """Return true if slices are trivial."""
    if self.strides is not None or len(arr_shape) != len(self.start_indices):
      return False
    return all(
      (start, stop) == (0, size)
      for start, stop, size in zip(self.start_indices, self.limit_indices, arr_shape)
    )


class DynamicSliceIndexer(NamedTuple):
  """Data structure encoding general indexing via a dynamic slice."""
  # start_indices and slice_sizes passed to lax.dynamic_slice()
  start_indices: Sequence[ArrayLike]
  slice_sizes: Sequence[int]

  # axes of the sliced array to pass to lax.rev()
  rev_axes: Sequence[int]

  # axes of the sliced array to pass to lax.squeeze()
  squeeze_axes: Sequence[int]

  # axes of the sliced array to pass to lax.expand_dims()
  newaxis_dims: Sequence[int]

  # flag indicating whether dynamic slice is trivial and can be skipped.
  trivial_slicing: bool

  # flag indicating whether negative indices should be normalized.
  normalize_indices: bool


class GatherIndexer(NamedTuple):
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
    from jax._src.numpy import array_constructors
    """Create an IndexType enum from a supported JAX array index."""
    if idx is None:
      return cls.NONE
    elif idx is Ellipsis:
      return cls.ELLIPSIS
    elif isinstance(idx, slice):
      return cls.SLICE
    elif isinstance(idx, Slice):
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


@tree_util.register_pytree_node_class
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
    from jax._src.numpy import lax_numpy
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
      mode: str | slicing.GatherScatterMode | None) -> StaticSliceIndexer:
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
    return StaticSliceIndexer(
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
      mode: str | slicing.GatherScatterMode | None) -> DynamicSliceIndexer:
    from jax._src.numpy import lax_numpy
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
        assert isinstance(pidx.index, Slice)
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
        assert isinstance(pidx.index, Slice)
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

    return DynamicSliceIndexer(
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
                normalize_indices: bool = True) -> GatherIndexer:
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


def _is_integer_index(idx: Any) -> bool:
  return isinstance(idx, (int, np.integer)) and not isinstance(idx, (bool, np.bool_))


def _index_to_gather(indexer: NDIndexer, *, x_sharding: NamedSharding | Any,
                     normalize_indices: bool = True) -> GatherIndexer:
  from jax._src.numpy import lax_numpy
  from jax._src.numpy import util
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
      if isinstance(index.index, Slice):
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
  return GatherIndexer(
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
