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
"""Module for state types."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import math
from typing import Any, Protocol, Union

from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src import pretty_printer as pp
from jax._src import traceback_util
from jax._src import tree_util
from jax._src.typing import Array
from jax._src.util import safe_map, safe_zip
import numpy as np

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
traceback_util.register_exclusion(__file__)

_ref_effect_color = pp.Color.GREEN

class RefEffect(effects.JaxprInputEffect):
  name: str

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False
    return self.input_index == other.input_index

  def __hash__(self):
    return hash((self.__class__, self.input_index))

  def _pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    if isinstance(self.input_index, core.Var):
      index_text = pp.text(core.pp_var(self.input_index, context))
    else:
      index_text = pp.text(self.input_index)
    return pp.concat([
      pp.color(pp.text(self.name), foreground=_ref_effect_color),
      pp.text("<"),
      index_text,
      pp.text(">")])

  def __str__(self):
    return f"{self.name}<{self.input_index}>"

class ReadEffect(RefEffect):
  name: str = "Read"

class WriteEffect(RefEffect):
  name: str = "Write"

class AccumEffect(RefEffect):
  name: str = "Accum"

effects.control_flow_allowed_effects.add_type(RefEffect)
effects.custom_derivatives_allowed_effects.add_type(RefEffect)
effects.custom_derivatives_allowed_effects.add_type(core.InternalMutableArrayEffect)
effects.partial_eval_kept_effects.add_type(RefEffect)
effects.remat_allowed_effects.add_type(RefEffect)

StateEffect = Union[ReadEffect, WriteEffect, AccumEffect]


# ## Transforms


class Transform(Protocol):

  def transform_type(self, x: core.AbstractValue) -> core.AbstractValue:
    raise NotImplementedError(type(self))

  def undo(self, x: core.AbstractValue) -> Transform:
    raise NotImplementedError(type(self))

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    return pp.text(f"{{{self}}}")


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BitcastTransform(Transform):
  dtype: dtypes.DType = dataclasses.field(metadata=dict(static=True))

  def transform_type(self, x):
    match x:
      case AbstractRef():
        return x.update(inner_aval=self.transform_type(x.inner_aval))
      case core.ShapedArray():
        from jax._src.state.utils import eval_bitcast_shape  # pytype: disable=import-error

        new_shape = eval_bitcast_shape(x, self.dtype)
        if not all(p is None for p in x.sharding.spec):
          raise NotImplementedError
        return x.update(shape=new_shape, dtype=self.dtype)
      case _:
        raise TypeError(f"Cannot bitcast {x} to {self.dtype}")

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    del context  # Unused.
    return pp.text(f"{{bitcast({self.dtype})}}")


def _canonicalize_reshape(
    input_shape: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[int, ...]:
  num_negative_ones = sum(s == -1 for s in shape)
  if num_negative_ones == 0:
    if np.prod(shape) != np.prod(input_shape):
      raise ValueError(
          f"cannot reshape shape {input_shape} into shape {shape}"
      )
    return shape
  num_elements = math.prod(input_shape)
  defined_dims = [d for d in shape if d != -1]
  if len(defined_dims) != len(shape) - 1:
    raise ValueError(f"At most one dimension can be -1, but got {shape}")
  if num_elements % math.prod(defined_dims):
    raise ValueError(
        f"Specified dims {shape} do not evenly divide the size of the "
        f"ref ({num_elements})."
    )
  remaining_dim = num_elements // math.prod(defined_dims)
  return tuple(d if d != -1 else remaining_dim for d in shape)


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ReshapeTransform(Transform):
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  def _validate_shape(self, input_shape: tuple[int, ...]):
    if np.prod(self.shape) != np.prod(input_shape):
      raise ValueError(
          f"cannot reshape ref of shape {input_shape} into shape {self.shape}"
      )

  def transform_type(self, x):
    match x:
      case AbstractRef():
        return x.update(inner_aval=self.transform_type(x.inner_aval))
      case core.ShapedArray():
        self._validate_shape(x.shape)
        # If there are no explicit axes, do nothing.
        if not all(p is None for p in x.sharding.spec):
          raise NotImplementedError
        return x.update(shape=self.shape)
      case _:
        raise TypeError(f"Cannot reshape {x} to {self.shape}")

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    del context  # Unused.
    return pp.text(f"{{reshape({list(self.shape)})}}")


def _perm_inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
  inverse = [-1] * len(permutation)
  for i, p in enumerate(permutation):
    inverse[p] = i
  return tuple(inverse)


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TransposeTransform(Transform):
  permutation: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  def undo(self, x: core.AbstractValue) -> Transform:
    return TransposeTransform(_perm_inverse(self.permutation))

  def transform_type(self, x):
    match x:
      case AbstractRef():
        return x.update(inner_aval=self.transform_type(x.inner_aval))
      case core.ShapedArray():
        if len(self.permutation) != x.ndim:
          raise ValueError(
              f"Permutation {self.permutation} does not match the rank of the "
              f"type ({x.ndim})"
          )
        # If there are no explicit axes, do nothing.
        if not all(p is None for p in x.sharding.spec):
          raise NotImplementedError
        new_shape = tuple(x.shape[i] for i in self.permutation)
        return x.update(shape=new_shape)
      case _:
        raise TypeError(f"Cannot transpose {x} to {self.permutation}")

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    del context  # Unused.
    return pp.text(f"{{transpose({list(self.permutation)})}}")


@dataclasses.dataclass
class RefIndexer:
  """An object temporarily generated when doing ``ref.at``."""
  ref_or_view: Any

  def __getitem__(self, slc) -> TransformedRef:
    if not isinstance(slc, tuple):
      slc = (slc,)
    from jax._src.state import indexing  # pytype: disable=import-error
    indexer = indexing.NDIndexer.from_indices_shape(slc, self.ref_or_view.shape)
    if isinstance(self.ref_or_view, TransformedRef):
      view = self.ref_or_view
      return TransformedRef(view.ref, (*view.transforms, indexer))
    return TransformedRef(self.ref_or_view, (indexer,))


@dataclasses.dataclass(frozen=True)
class TransformedRef:
  ref: Any
  transforms: tuple[Transform, ...]

  @property
  def is_dynamic_size(self):
    return any(not isinstance(i, int) for i in self.shape)

  @functools.cached_property
  def type(self) -> core.AbstractValue:
    if type(self.ref) in core.pytype_aval_mappings:
      ref_ty = core.typeof(self.ref)
    else:
      ref_ty = self.ref
    for t in self.transforms:
      ref_ty = t.transform_type(ref_ty)
    return ref_ty

  @property
  def shape(self) -> tuple[int | Array, ...]:
    if not hasattr(self.type, "shape"):
      raise AttributeError(f"{self!r} has no `shape`.") from None
    return self.type.shape

  @property
  def dtype(self):
    if not hasattr(self.type, "dtype"):
      raise AttributeError(f"{self!r} has no `dtype`.") from None
    return self.type.dtype

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: math.prod(self.shape))
  T = property(lambda self: self.transpose(tuple(reversed(range(self.ndim)))))

  @property
  def at(self) -> RefIndexer:
    return RefIndexer(self)

  def bitcast(self, dtype):
    if self.is_dynamic_size:
      raise NotImplementedError(
          "Bitcast ref with dynamic size is not supported."
      )
    dtype = dtypes.dtype(dtype)
    return TransformedRef(self.ref, (*self.transforms, BitcastTransform(dtype)))

  def reshape(self, *shape):
    if self.is_dynamic_size:
      raise NotImplementedError(
          "Reshape ref with dynamic size is not supported."
      )
    if len(shape) == 1 and isinstance(shape[0], tuple):
      shape = shape[0]
    shape = _canonicalize_reshape(self.shape, shape)
    return TransformedRef(self.ref, (*self.transforms, ReshapeTransform(shape)))

  def transpose(self, permutation: Sequence[int]):
    transposer = TransposeTransform(tuple(permutation))
    return TransformedRef(self.ref, (*self.transforms, transposer))

  def set(self, value, idx=()):
    from jax._src.state.primitives import ref_set  # pytype: disable=import-error
    return ref_set(self, idx, value)

  def swap(self, value, idx=()):
    from jax._src.state.primitives import ref_swap  # pytype: disable=import-error
    return ref_swap(self, idx, value)

  def get(self, idx=()):
    from jax._src.state.primitives import ref_get  # pytype: disable=import-error
    return ref_get(self, idx)

  def __getattr__(self, name):
    return getattr(self.ref, name)

  def __getitem__(self, slc):
    from jax._src.state.primitives import ref_get  # pytype: disable=import-error
    return ref_get(self, slc)

  def __setitem__(self, slc, value):
    from jax._src.state.primitives import ref_set # pytype: disable=import-error
    return ref_set(self, slc, value)


def transform_type(
    ts: Sequence[Transform], ty: core.AbstractValue
) -> core.AbstractValue:
  for t in ts:
    ty = t.transform_type(ty)
  return ty


# We need an aval for `Ref`s so we can represent `get` and `swap` in Jaxprs.
class AbstractRef(core.AbstractValue):
  """Abstract mutable array reference.

  Refer to the `Ref guide`_ for more information.

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  __slots__ = ["inner_aval", "memory_space", "kind"]

  def __init__(self, inner_aval: core.AbstractValue, memory_space: Any = None,
               kind: Any = None):
    self.inner_aval = inner_aval
    self.memory_space = memory_space
    self.kind = kind

  @property
  def is_high(self):
    return self.inner_aval.is_high

  def lo_ty(self):
    return [
        AbstractRef(x, memory_space=self.memory_space)
        for x in self.inner_aval.lo_ty()
    ]

  def lower_val(self, ref):
    if not self.is_high:
      return [ref]
    return self.inner_aval.lower_val(ref._refs)  # type: ignore

  def raise_val(self, *vals):
    if not self.is_high:
      ref, = vals
      return ref
    return core.Ref(self, self.inner_aval.raise_val(*vals))  # type: ignore

  @property
  def weak_type(self) -> bool:
    if not hasattr(self.inner_aval, "weak_type"):
      raise AttributeError
    return self.inner_aval.weak_type

  def update_weak_type(self, weak_type):
    return self.update(inner_aval=self.inner_aval.update_weak_type(weak_type))

  def update(self, inner_aval=None, memory_space=None, kind=None):
    inner_aval = self.inner_aval if inner_aval is None else inner_aval
    memory_space = self.memory_space if memory_space is None else memory_space
    kind = self.kind if kind is None else kind
    return AbstractRef(inner_aval, memory_space, kind)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: math.prod(self.shape))

  def _len(self, ignored_tracer) -> int:
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error

  @property
  def shape(self):
    try:
      return self.inner_aval.shape  # pytype: disable=attribute-error
    except AttributeError:
      raise AttributeError(
          f"{self!r} has no `shape`."
      ) from None

  @property
  def dtype(self):
    try:
      return self.inner_aval.dtype  # pytype: disable=attribute-error
    except AttributeError:
      raise AttributeError(
          f"{self!r} has no `dtype`."
      ) from None

  @property
  def sharding(self):
    try:
      return self.inner_aval.sharding  # pytype: disable=attribute-error
    except AttributeError:
      raise AttributeError(
          f"{self!r} has no `sharding`."
      ) from None

  @property
  def vma(self):
    try:
      return self.inner_aval.vma  # pytype: disable=attribute-error
    except AttributeError:
      raise AttributeError(
          f"{self!r} has no `vma`."
      ) from None

  @core.aval_property
  def at(self):
    return RefIndexer(self)

  @core.aval_method
  def bitcast(self, dtype):
    return TransformedRef(self, ()).bitcast(dtype)

  @core.aval_method
  def reshape(self, *shape):
    return TransformedRef(self, ()).reshape(*shape)

  @core.aval_method
  def transpose(self, *permutation):
    return TransformedRef(self, ()).transpose(*permutation)

  @core.aval_property
  def T(self):
    return TransformedRef(self, ()).T

  @core.aval_method
  @staticmethod
  def get(tracer, idx=()):
    from jax._src.state.primitives import ref_get  # pytype: disable=import-error
    return ref_get(tracer, idx)

  @core.aval_method
  @staticmethod
  def swap(tracer, value, idx=()):
    from jax._src.state.primitives import ref_swap  # pytype: disable=import-error
    return ref_swap(tracer, idx, value)

  @core.aval_method
  @staticmethod
  def set(tracer, value, idx=()):
    from jax._src.state.primitives import ref_set  # pytype: disable=import-error
    return ref_set(tracer, idx, value)

  @core.aval_method
  @staticmethod
  def addupdate(tracer, value, idx=()):
    from jax._src.state.primitives import ref_addupdate  # pytype: disable=import-error
    ref_addupdate(tracer, idx, value)

  def _getitem(self, tracer, idx) -> Array:
    from jax._src.state.primitives import ref_get  # pytype: disable=import-error
    return ref_get(tracer, idx)

  def _setitem(self, tracer, idx, value) -> None:
    from jax._src.state.primitives import ref_set  # pytype: disable=import-error
    return ref_set(tracer, idx, value)

  def _addupdate(self, tracer, idx, value):
    from jax._src.state.primitives import ref_addupdate  # pytype: disable=import-error
    ref_addupdate(tracer, idx, value)

  def str_short(self, short_dtypes=False, mesh_axis_types=False) -> str:
    inner_aval_str = self.inner_aval.str_short(
        short_dtypes=short_dtypes,
        mesh_axis_types=mesh_axis_types,
    )
    if self.memory_space is not None:
      return f'Ref<{self.memory_space}>{{{inner_aval_str}}}'
    return f'Ref{{{inner_aval_str}}}'

  def __repr__(self) -> str:
    return self.str_short()
  __str__ = __repr__

  def to_tangent_aval(self):
    return AbstractRef(self.inner_aval.to_tangent_aval(), self.memory_space,
                       kind=self.kind)

  def to_cotangent_aval(self):
    return AbstractRef(self.inner_aval.to_cotangent_aval(), self.memory_space,
                       kind=self.kind)

  def __eq__(self, other):
    return (type(self) is type(other) and self.inner_aval == other.inner_aval
            and self.memory_space == other.memory_space)

  def __hash__(self):
    return hash((self.__class__, self.inner_aval, self.memory_space))

def _map_ref(size, axis, ref_aval):
  return AbstractRef(core.mapped_aval(size, axis, ref_aval.inner_aval),
                     ref_aval.memory_space, ref_aval.kind)

def _unmap_ref(size, axis, explicit_mesh_axis, ref_aval):
  return AbstractRef(core.unmapped_aval(
      size, axis, ref_aval.inner_aval, explicit_mesh_axis),
                     ref_aval.memory_space, ref_aval.kind)

core.aval_mapping_handlers[AbstractRef] = (_map_ref, _unmap_ref)

def get_ref_state_effects(
    avals: Sequence[core.AbstractValue],
    effects: core.Effects) -> list[set[StateEffect]]:
  return [{eff for eff in effects
           if isinstance(eff, (ReadEffect, WriteEffect, AccumEffect))
           and eff.input_index == i} for i, _ in enumerate(avals)]

def shaped_array_ref(
    shape: tuple[int, ...], dtype, weak_type: bool = False) -> AbstractRef:
  return AbstractRef(core.ShapedArray(shape, dtype, weak_type=weak_type))

def _shard_ref(mesh, auto, check_rep, names, ref_aval: AbstractRef):
  aval = core.shard_aval(mesh, auto, check_rep, names, ref_aval.inner_aval)
  return AbstractRef(aval)
core.shard_aval_handlers[AbstractRef] = _shard_ref

def _unshard_ref(mesh, check_rep, names, ref_aval: AbstractRef):
  raise TypeError("can't unshard a ref")
core.unshard_aval_handlers[AbstractRef] = _unshard_ref


# Sentinel type for indicating an uninitialized value.
class Uninitialized:
  pass
uninitialized = Uninitialized()


_ref_type_aval_mappings: dict[
    type[Any], Callable[[Any], tuple[AbstractRef, Array | Uninitialized]],
] = {}


def _default_value_to_ref_aval(x: Any) -> tuple[AbstractRef, Array]:
  # Default type mapping just creates an AbstractRef from the array's aval.
  aval = core.get_aval(x)
  return AbstractRef(aval), x


def get_ref_aval_from_value(x: Any):
  if type(x) in _ref_type_aval_mappings:
    return _ref_type_aval_mappings[type(x)](x)
  return _default_value_to_ref_aval(x)

# === pinned, chained LinearVals ===

@dataclasses.dataclass(frozen=True)
class AbstractLinVal(core.AbstractValue):
  inner_aval: core.AbstractValue
  memory_space: Any = None

  shape = property(lambda self: self.inner_aval.shape)  # type: ignore
  dtype = property(lambda self: self.inner_aval.dtype)  # type: ignore
  ndim = property(lambda self: self.inner_aval.ndim)  # type: ignore
