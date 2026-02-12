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
import math
import operator
from typing import Any, ClassVar, Union

from jax._src import core
from jax._src import pretty_printer as pp
from jax._src import tree_util
from jax._src.indexing import Slice, dslice, ds  # noqa: F401
from jax._src.state import types as state_types  # pytype: disable=import-error
from jax._src.typing import Array
from jax._src.util import merge_lists
from jax._src.util import partition_list
import numpy as np


def _pp_slice(context: core.JaxprPpContext, dim, slc: Slice) -> str:
  start, size = slc.start, slc.size
  if isinstance(start, core.Var):
    start_str = core.pp_var(start, context)
    size_str = (
        core.pp_var(size, context) if isinstance(size, core.Var) else str(size)
    )
    return f"{start_str}:{start_str}+{size_str}"
  else:
    start_str = str(start)
    if start == 0:
      start_str = ""
    if isinstance(size, core.Var):
      size_str = core.pp_var(size, context)
      if start_str:
        return f"{start_str}:{start_str}+{size_str}"
      else:
        return f":{size_str}"
    else:
      end = start + size
      end_str = "" if end == dim else str(end)
      return f"{start_str}:{end_str}"


IntIndexer = Union[int, Array]
DimIndexer = Union[IntIndexer, Slice]

def unpack_ndindexer(indexer: NDIndexer) -> tuple[tuple[bool, ...],
                                                  tuple[Slice, ...],
                                                  tuple[IntIndexer, ...]]:
  # TODO(slebedev): Flip this to be ``is_slice_indexing`` and update callers.
  is_int_indexing = [not isinstance(i, Slice) for i in indexer.indices]
  slice_indexers, int_indexers = partition_list(
      is_int_indexing, indexer.indices)
  return tuple(is_int_indexing), tuple(slice_indexers), tuple(int_indexers)  # type: ignore

def _maybe_concretize(x: Any):
  # This is roughly the same logic as core.concrete_or_error, but we avoid
  # calling that because constructing the ConcretizationTypeError can be
  # expensive as the size of the tracing context (i.e. the jaxpr) grows.
  return core.to_concrete_value(x)

# This registry is used to allow hitypes that are being indexed to register
# type transformation rules.
indexer_transform_type_registry: set[type] = set()

@tree_util.register_pytree_node_class
@dataclasses.dataclass
class NDIndexer(state_types.Transform):
  indices: tuple[DimIndexer, ...]
  shape: tuple[int, ...]
  int_indexer_shape: tuple[int | Array, ...]
  # Off by default to avoid doing validation during pytree operations.
  validate: bool = False

  def __post_init__(self):
    if len(self.indices) != len(self.shape):
      raise ValueError(
          f"`indices` must be the same length as `Ref` shape.: {self}."
      )
    if not self.validate:
      return
    # We validate integer indexing shapes here
    for idx, s in zip(self.indices, self.shape):
      if isinstance(idx, Slice):
        start = idx.start
        if value := _maybe_concretize(start):
          if value >= s:
            raise ValueError(f"Out of bound slice: start={value}, dim={s}.")
          if size := _maybe_concretize(idx.size):
            if value + (size - 1) * idx.stride >= s:
              raise ValueError(
                  f"Out of bound slice: start={value}, size={size},"
                  f" stride={idx.stride}, dim={s}."
              )
        continue
      # The shape of indexer integers should be broadcastable up to the
      # int_indexer_shape of the whole NDIndexer
      idx_shape = (
          idx.shape
          if isinstance(idx, state_types.TransformedRef)
          else core.get_aval(idx).shape
      )
      if not idx_shape:
        if (value := _maybe_concretize(idx)) and value >= s:
          raise ValueError(f"Out of bound indexer: idx={value}, dim={s}.")
        # For ()-shaped indexers, we can broadcast no problm.
        continue
      # If we don't have a ()-shaped indexer, the rank must match
      # int_indexer_shape
      if len(idx_shape) != len(self.int_indexer_shape):
        raise ValueError(
            f"Indexer must have rank {len(idx_shape)}: {idx=} vs."
            f" {self.int_indexer_shape=}"
        )
      # Here we check that the shapes broadcast.
      try:
        np.broadcast_shapes(idx_shape, self.int_indexer_shape)
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
    if not all(isinstance(i, int) for i in self.int_indexer_shape):
      return (*flat_idx, self.int_indexer_shape), (idx_tree, self.shape)
    else:
      return flat_idx, (idx_tree, self.shape, self.int_indexer_shape)

  @classmethod
  def tree_unflatten(cls, data, flat_idx):
    if len(data) == 3:
      idx_tree, shape, int_indexer_shape = data
    else:
      # The ``int_indexer_shape`` is dynamic.
      idx_tree, shape = data
      *flat_idx, int_indexer_shape = flat_idx
    indices = tree_util.tree_unflatten(idx_tree, flat_idx)
    return cls(tuple(indices), shape, int_indexer_shape)

  @classmethod
  def from_indices_shape(cls, indices, shape) -> NDIndexer:
    if not isinstance(indices, tuple):
      # TODO(slebedev): Consider requiring `indices` to be a Sequence.
      indices = (indices,)

    if num_ellipsis := sum(idx is ... for idx in indices):
      if num_ellipsis > 1:
        raise ValueError("Only one ellipsis is supported.")
      # Expand ... so that `indices` has the same length as `shape`.
      ip = next(i for i, idx in enumerate(indices) if idx is ...)
      indices = list(indices)
      indices[ip:ip+1] = [slice(None)] * (len(shape) - len(indices) + 1)
      indices = tuple(indices)
    if len(indices) > len(shape):
      raise ValueError("`indices` must not be longer than `shape`: "
                       f"{indices=}, {shape=}")
    elif len(indices) < len(shape):
      # Pad `indices` to have the same length as `shape`.
      indices = (*indices, *[slice(None)] * (len(shape) - len(indices)))

    # Promote all builtin `slice`s to `Slice`.
    indices = tuple(
        Slice.from_slice(i, s) if isinstance(i, slice) else i
        for i, s in zip(indices, shape))

    is_slice_indexing = [isinstance(i, Slice) for i in indices]
    if all(is_slice_indexing):
      return cls(indices, shape, (), validate=True)

    other_indexers, slice_indexers = partition_list(is_slice_indexing, indices)
    validate = True

    # We treat refs differently from scalars and arrays, because refs can have
    # a dynamic shape, making it impossible to statically determine the
    # broadcasted shape in the presence of other non-slice indexers.
    from jax._src.state import types as state_types  # pytype: disable=import-error
    if ref_indexers := [
        i
        for i in other_indexers
        if isinstance(i, state_types.TransformedRef)
        or isinstance(core.get_aval(i), state_types.AbstractRef)
    ]:
      # TODO(slebedev): Consider pushing these checks to lowering time.
      if len(ref_indexers) > 1:
        raise NotImplementedError("Multiple Ref indexers are not supported")
      if len(ref_indexers) != len(other_indexers):
        raise NotImplementedError(
            "Ref cannot be mixed with other non-slice indexers"
        )
      [ref_indexer] = ref_indexers
      indexer_shape = ref_indexer.shape  # type: ignore
      try:
        core.canonicalize_shape(indexer_shape)
      except TypeError:
        validate = False  # The shape is dynamic.
    else:
      indexer_shapes = [core.get_aval(i).shape for i in other_indexers]
      try:
        indexer_shape = np.broadcast_shapes(*indexer_shapes)
      except ValueError as e:
        # Raise a nicer error than the NumPy one.
        raise ValueError(
            "Cannot broadcast shapes for indexing: {indexer_shapes}"
        ) from e

      # Here we use the `broadcast_to` primitive instead of composing lax
      # primitives together because it is easier to lower in targets like
      # Triton/Mosaic.
      #
      # The local import avoids a circular dependency between primitives
      # and this module.
      from jax._src.state import primitives as sp  # pytype: disable=import-error
      other_indexers = [
          sp.broadcast_to(i, indexer_shape) for i in other_indexers  # type: ignore[arg-type]
      ]
      indices = tuple(
          merge_lists(is_slice_indexing, other_indexers, slice_indexers)
       )
    return cls(indices, shape, indexer_shape, validate)

  @classmethod
  def make_trivial_indexer(cls, shape: tuple[int, ...]) -> NDIndexer:
    return NDIndexer.from_indices_shape(
        tuple(slice(0, e) for e in shape),
        shape,
    )

  def get_indexer_shape(self) -> tuple[int | Array, ...]:
    is_int_indexing, slice_indexers, _ = unpack_ndindexer(self)

    slice_shape = tuple(s.size for s in slice_indexers)
    int_indexers_contiguous = bool(
        np.all(np.diff(np.where(is_int_indexing)[0]) == 1)
    )
    if not int_indexers_contiguous:
      return self.int_indexer_shape + slice_shape

    has_int_indexers = any(is_int_indexing)
    if has_int_indexers:
      pos = is_int_indexing.index(True)
      return slice_shape[:pos] + self.int_indexer_shape + slice_shape[pos:]

    return slice_shape

  def transform_type(self, x: core.AbstractValue):
    match x:
      case state_types.AbstractRef():
        return x.update(inner_aval=self.transform_type(x.inner_aval))
      case core.ShapedArray():
        self._validate_sharding(x.sharding)
        if self.is_dynamic_size:
          return DShapedArray(self.get_indexer_shape(), x.dtype,
                              weak_type=x.weak_type,
                              sharding=x.sharding,
                              vma=x.vma,
                              memory_space=x.memory_space)
        return x.update(shape=self.get_indexer_shape())
      case _:
        if type(x) in indexer_transform_type_registry:
          assert hasattr(x, "transform_ndindexer")
          return x.transform_ndindexer(self)
        raise TypeError(f"Cannot transform type: {x}")

  def undo(self, x: core.AbstractValue):
    raise NotImplementedError

  def _validate_sharding(self, sharding):
    if all(p is None for p in sharding.spec):
      return
    # If there are explicit axes, we don't support changing the shape, so
    # we don't support int indexers and instead require all slices.
    if self.int_indexer_shape or not all(
        isinstance(idx, Slice) for idx in self.indices
    ):
      raise TypeError(
          "sharded ref (array reference) can only be indexed by "
          "slices, not integers"
      )
    #  Moreover, only allow trivial slice(None) slices on explicitly sharded
    #  axes. Then the sharding stays the same.
    _, slice_indexers, _ = unpack_ndindexer(self)
    for i, (d, sl, s) in enumerate(
        zip(self.shape, slice_indexers, sharding.spec)
    ):
      if s is None:
        continue
      if not (
          type(sl.start) is int
          and sl.start == 0
          and type(sl.size) is int
          and sl.size == d
          and type(sl.stride) is int
          and sl.stride == 1
      ):
        raise ValueError(
            "sharded ref (array reference) can only be sliced "
            f"along unsharded axes, but ref of shape {self.shape} "
            f"was sliced on axis {i}, which is sharded like {s}"
        )

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    indices = []
    for idx, dim in zip(self.indices, self.shape):
      if isinstance(idx, Slice):
        indices.append(_pp_slice(context, dim, idx))
      else:
        indices.append(core.pp_var(idx, context, print_literal_dtype=False))  # type: ignore
    return pp.concat([pp.text("["), pp.text(",".join(indices)), pp.text("]")])


class DShapedArray:
  def __init__(self, shape, dtype, weak_type=False, *, sharding=None,
               vma: frozenset[core.AxisName] = frozenset(),
               memory_space: core.MemorySpace = core.MemorySpace.Device):
    self.shape = shape
    self.dtype = core._dtype_object(dtype)
    self.weak_type = weak_type
    self.sharding = sharding
    self.vma = core.get_vma(vma, self.sharding)
    self.memory_space = core.get_memory_space(memory_space)

  def lower_val(self, val): return [val]
  def raise_val(self, val): return val
  def lo_ty(self): return [self]

  def update(self, shape=None, dtype=None, weak_type=None, **kwargs):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    if 'sharding' not in kwargs:
      kwargs['sharding'] = self.sharding
    if 'vma' not in kwargs:
      kwargs['vma'] = self.vma
    if 'memory_space' not in kwargs:
      kwargs['memory_space'] = self.memory_space
    return DShapedArray(shape, dtype, weak_type, **kwargs)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self:
                  0 if any(type(d) is int and d == 0 for d in self.shape)
                  else math.prod(self.shape))

  broadcast: ClassVar[core.aval_method | None] = None
  transpose: ClassVar[core.aval_method | None] = None
  reshape: ClassVar[core.aval_method | None] = None
  _iter: ClassVar[staticmethod | None] = None

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type
            and self.sharding == other.sharding
            and self.vma == other.vma
            and self.memory_space == other.memory_space)

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.shape, self.dtype, self.weak_type, self.sharding,
                 self.vma, self.memory_space))

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    wt_str = ", weak_type=True" if self.weak_type else ""
    return f'DShapedArray({self.str_short()}{wt_str})'

  def __str__(self):
    wt_str = "~" if self.weak_type else ""
    return f'{wt_str}{self.str_short()}'

  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    return core.str_short_aval(
        self.shape, self.dtype, self.sharding.mesh, self.sharding.spec,  # pyrefly: ignore[missing-attribute]
        self.vma, self.memory_space, short_dtypes, mesh_axis_types)

  def _len(self, ignored_tracer):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error

  def update_vma(self, vma):
    return self.update(vma=vma)

  def update_weak_type(self, weak_type):
    return self.update(weak_type=weak_type)

  _bool    = core.concretization_function_error(bool)
  _int     = core.concretization_function_error(int, True)
  _float   = core.concretization_function_error(float, True)
  _complex = core.concretization_function_error(complex, True)
  _hex     = core.concretization_function_error(hex)
  _oct     = core.concretization_function_error(oct)
  _index   = core.concretization_function_error(operator.index)
