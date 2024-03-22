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

"""Contains shared logic and abstractions for Pallas indexing ops."""

from __future__ import annotations

import dataclasses
from typing import Any, Union

from jax._src import core
from jax._src import tree_util
from jax._src.typing import Array
from jax._src.util import merge_lists
from jax._src.util import partition_list
import numpy as np


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Slice:
  """Represents a slice with a dynamic start index and a fixed size."""
  start: int | Array
  size: int | Array
  stride: int = 1

  def __post_init__(self):
    if self.stride < 1:
      raise ValueError("`stride` must be >= 1.")

  @property
  def is_dynamic_start(self):
    return not isinstance(self.start, int)

  @property
  def is_dynamic_size(self):
    return not isinstance(self.size, int)

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
    start, size = [
        a if a is not None else b for a, b in zip(children, aux_data[:2])
    ]
    return cls(start, size, aux_data[2])

  @classmethod
  def from_slice(cls, slc: slice, size: int) -> Slice:
    start, stop, step = slc.indices(size)
    if step < 1:
      raise ValueError(f"slice must have a step >= 1 (found: {step})")
    return cls(start, max((stop - start + step - 1) // step, 0), step)


def dslice(
    start: int | Array | None,
    size: int | Array | None = None,
    stride: int | None = None,
) -> slice | Slice:
  """Constructs a `Slice` from a start and a size."""
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


ds = dslice  # Handy alias


IntIndexer = Union[int, Array]
DimIndexer = Union[IntIndexer, Slice]

def unpack_ndindexer(indexer: NDIndexer) -> tuple[tuple[bool, ...],
                                                  tuple[Slice, ...],
                                                  tuple[IntIndexer, ...]]:
  is_int_indexing = [not isinstance(i, Slice) for i in indexer.indices]
  slice_indexers, int_indexers = partition_list(
      is_int_indexing, indexer.indices)
  return tuple(is_int_indexing), tuple(slice_indexers), tuple(int_indexers)  # type: ignore

def _maybe_concretize(x: Any):
  # This is roughly the same logic as core.concrete_or_error, but we avoid
  # calling that because constructing the ConcretizationTypeError can be
  # expensive as the size of the tracing context (i.e. the jaxpr) grows.
  if isinstance(x, core.Tracer):
    if isinstance(x.aval, core.ConcreteArray):
      return x.aval.val
    else:
      return None
  return x

@tree_util.register_pytree_node_class
@dataclasses.dataclass
class NDIndexer:
  indices: tuple[DimIndexer, ...]
  shape: tuple[int, ...]
  int_indexer_shape: tuple[int, ...]
  validate: bool = False

  def __post_init__(self):
    if not self.validate:
      return
    if len(self.indices) != len(self.shape):
      raise ValueError(
          f"`indices` must be the same length as `Ref` shape.: {self}."
      )
    # We validate integer indexing shapes here
    for idx, s in zip(self.indices, self.shape):
      if isinstance(idx, Slice):
        start = idx.start
        if value := _maybe_concretize(start):
          if value >= s:
            raise ValueError(f"Out of bound slice: start={value}, dim={s}.")
          if value + (idx.size - 1) * idx.stride >= s:
            raise ValueError(
                f"Out of bound slice: start={value}, size={idx.size},"
                f" stride={idx.stride}, dim={s}."
            )
        continue
      # The shape of indexer integers should be broadcastable up to the
      # int_indexer_shape of the whole NDIndexer
      if not np.shape(idx):
        if (value := _maybe_concretize(idx)) and value >= s:
          raise ValueError(f"Out of bound indexer: idx={value}, dim={s}.")
        # For ()-shaped indexers, we can broadcast no problm.
        continue
      # If we don't have a ()-shaped indexer, the rank must match
      # int_indexer_shape
      if np.ndim(idx) != len(self.int_indexer_shape):
        raise ValueError(
            f"Indexer must have rank {np.ndim(idx)}: {idx=} vs."
            f" {self.int_indexer_shape=}"
        )
      # Here we check that the shapes broadcast.
      try:
        np.broadcast_shapes(np.shape(idx), self.int_indexer_shape)
      except ValueError as e:
        raise ValueError(
            f"Could not broadcast integer indexer: {idx=} vs."
            f" {self.int_indexer_shape=}"
        ) from e

  @property
  def is_dynamic_size(self):
    return any(isinstance(i, Slice) and i.is_dynamic_size for i in self.indices)

  def tree_flatten(self):
    flat_idx, idx_tree = tree_util.tree_flatten(self.indices)
    return flat_idx, (idx_tree, self.shape, self.int_indexer_shape)

  @classmethod
  def tree_unflatten(cls, data, flat_idx):
    idx_tree, shape, int_indexer_shape = data
    indices = tree_util.tree_unflatten(idx_tree, flat_idx)
    return NDIndexer(tuple(indices), shape, int_indexer_shape)

  @classmethod
  def from_indices_shape(cls, indices, shape) -> NDIndexer:
    if not isinstance(indices, tuple):
      indices = (indices,)
    if len(indices) == 1 and indices[0] is ...:
      indices = (slice(None),) * len(shape)
    if any(idx is ... for idx in indices):
      # TODO(sharadmv,mattjj): support patterns that include ellipsis in them
      #                        e.g. x[0, ..., 1].
      raise NotImplementedError("Ellipsis in indexer not supported yet.")
    if len(indices) > len(shape):
      raise ValueError("`indices` must not be longer than `shape`: "
                       f"{indices=}, {shape=}")
    # Pad out indices with slice(None)
    indices = [*indices, *[slice(None)] * (len(shape) - len(indices))]
    # Convert all `slice`s to `Slice`s
    indices = tuple(Slice.from_slice(i, s) if isinstance(i, slice)
                    else i for i, s in zip(indices, shape))
    is_int_indexing = [not isinstance(i, Slice) for i in indices]
    other_indexers, int_indexers = partition_list(is_int_indexing, indices)
    indexer_shapes = [core.get_aval(i).shape for i in int_indexers]
    if indexer_shapes:
      try:
        bcast_shape = np.broadcast_shapes(*indexer_shapes)
      except ValueError as e:
        # Raise a nicer error than the NumPy one.
        raise ValueError("Cannot broadcast shapes for indexing: "
                         f"{tuple(a for a in indexer_shapes)}") from e
    else:
      bcast_shape = ()
    # Here we use the `broadcast_to` primitive instead of composing lax
    # primitives together because it is easier to lower in targets like
    # Triton/Mosaic.
    from jax._src.state import primitives as sp  # pytype: disable=import-error
    int_indexers = [sp.broadcast_to(i, bcast_shape) for i in int_indexers]
    indices = merge_lists(is_int_indexing, other_indexers, int_indexers)
    return NDIndexer(tuple(indices), shape, bcast_shape, validate=True)

  def get_indexer_shape(self) -> tuple[int | Array, ...]:
    _, slice_indexers, _ = unpack_ndindexer(self)
    slice_shape = [s.size for s in slice_indexers]
    # In NDIndexers, the int_indexer_shape is *always* at the front of the
    # result.
    return (*self.int_indexer_shape, *slice_shape)
