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
from typing import Any

import jax
from jax import core as jax_core
from jax import tree_util
from jax._src.interpreters import mlir
from jax._src.util import merge_lists
from jax._src.util import partition_list
import jax.numpy as jnp
import numpy as np


# Currently, JAX doesn't have a primitive that does an equal-rank broadcast.
# We could use `jnp.broadcast_to` but that lowers to squeezing,
# then broadcast_in_dim. Triton has an equal-rank broadcast (`tl.broadcast_to`)
# so in the lowering, we have to expand out those squeezed dimensions again.
# Having a simple `broadcast_to` primitive allows us to lower directly
# to `tl.broadcast_to`.
broadcast_to_p = jax_core.Primitive('broadcast_to')

def broadcast_to(a: jax.Array, shape: tuple[int, ...]) -> jax.Array:
  if a.shape == shape:
    return a
  return broadcast_to_p.bind(a, shape=shape)

@broadcast_to_p.def_impl
def _broadcast_to_impl(a, *, shape):
  return jnp.broadcast_to(a, shape)

@broadcast_to_p.def_abstract_eval
def _broadcast_to_abstract_eval(aval, *, shape):
  return jax_core.ShapedArray(shape, aval.dtype)

mlir.register_lowering(
    broadcast_to_p, mlir.lower_fun(_broadcast_to_impl, False)
)


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Slice:
  """Represents a slice with a dynamic start index and a fixed size."""
  start: Any
  size: int

  def __post_init__(self):
    if self.size < 0:
      raise ValueError("`size` must not be negative.")

  def tree_flatten(self):
    # If `start` is statically known, we treat it as static information
    if isinstance(self.start, int):
      return (), (self.start, self.size)
    return (self.start,), (self.size,)

  @classmethod
  def tree_unflatten(cls, aux_data, children) -> Slice:
    return cls(*children, *aux_data)

  @classmethod
  def from_slice(cls, slc: slice, size: int) -> Slice:
    start, stop, step = slc.indices(size)
    if step != 1:
      raise ValueError(f"slice must have a step of 1 (found: {step})")
    return cls(start, stop - start)


def dslice(start: int |  jax.Array | None, size: int | None = None
           ) -> slice | Slice:
  """Constructs a `Slice` from a start and a size."""
  if start is None:
    return slice(None)
  if size is None:
    if not isinstance(start, int):
      raise ValueError("Non-static `dslice`")
    return Slice(0, start)
  return Slice(start, size)
ds = dslice  # Handy alias

@tree_util.register_pytree_node_class
@dataclasses.dataclass
class NDIndexer:
  indices: tuple[int | Slice | jax.Array, ...]
  shape: tuple[int, ...]
  int_indexer_shape: tuple[int, ...]

  def __post_init__(self):
    if len(self.indices) != len(self.shape):
      raise ValueError("`indices` must be the same length as `Ref` shape.")

  def tree_flatten(self):
    indexed_dims = [not isinstance(idx, slice) for idx in self.indices]
    slice_idx, non_slice_idx = partition_list(indexed_dims, self.indices)
    flat_idx, idx_tree = tree_util.tree_flatten(non_slice_idx)
    return flat_idx, (slice_idx, idx_tree, indexed_dims, self.shape,
                      self.int_indexer_shape)

  @classmethod
  def tree_unflatten(cls, data, flat_idx):
    slice_idx, idx_tree, indexed_dims, shape, int_indexer_shape = data
    non_slice_idx = tree_util.tree_unflatten(idx_tree, flat_idx)
    indices = merge_lists(indexed_dims, slice_idx, non_slice_idx)
    return NDIndexer(tuple(indices), shape, int_indexer_shape)

  @classmethod
  def from_indices_shape(cls, indices, shape) -> NDIndexer:
    if len(indices) > len(shape):
      raise ValueError("`indices` must not be longer than `shape`.")
    # Pad out indices with slice(None)
    indices = [*indices, *[slice(None)] * (len(shape) - len(indices))]
    # Convert all `slice`s to `Slice`s
    indices = tuple(Slice.from_slice(i, s) if isinstance(i, slice)
                    else i for i, s in zip(indices, shape))
    is_int_indexing = [not isinstance(i, Slice) for i in indices]
    other_indexers, int_indexers = partition_list(is_int_indexing, indices)
    int_indexers = [np.array(i, np.int32) if isinstance(i, int) else i for i in
                    int_indexers]
    indexer_shapes = [i.shape for i in int_indexers]
    if indexer_shapes:
      try:
        bcast_shape = np.broadcast_shapes(*indexer_shapes)
      except ValueError as e:
        # Raise a nicer error than the NumPy one.
        raise ValueError("Cannot broadcast shapes for indexing: "
                         f"{tuple(a for a in indexer_shapes)}") from e
    else:
      bcast_shape = ()
    int_indexers = [broadcast_to(i, bcast_shape) for i in int_indexers]
    indices = merge_lists(is_int_indexing, other_indexers, int_indexers)
    return NDIndexer(tuple(indices), shape, bcast_shape)

  def get_indexer_shape(self) -> tuple[int, ...]:
    is_int_indexing = [not isinstance(i, Slice) for i in self.indices]
    other_indexers, _ = partition_list(is_int_indexing, self.indices)
    other_shape = [s.size for s in other_indexers]  # type: ignore
    return (*self.int_indexer_shape, *other_shape)
