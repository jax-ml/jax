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

from collections.abc import Sequence
import dataclasses
import math
from typing import Any, Protocol, Union
from collections.abc import Callable

from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src import pretty_printer as pp
from jax._src import traceback_util
from jax._src import tree_util
from jax._src.state import indexing
from jax._src.typing import Array
from jax._src.typing import DTypeLike
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

StateEffect = Union[ReadEffect, WriteEffect, AccumEffect]


# ## `Ref`s
@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class RefBitcaster:
  dtype: dtypes.DType
  shape: tuple[int, ...]

  @classmethod
  def from_ref_new_dtype(cls, ref_or_view: Any, dtype) -> RefBitcaster:
    if isinstance(ref_or_view, TransformedRef):
      if ref_or_view.is_dynamic_size:
        raise NotImplementedError(
            "Bitcast ref with dynamic size is not supported."
        )
    from jax._src.state.utils import eval_bitcast_shape  # pytype: disable=import-error
    dtype = dtypes.dtype(dtype)
    return cls(dtype, eval_bitcast_shape(ref_or_view, dtype))

  @property
  def is_dynamic_size(self):
    return False

  def tree_flatten(self):
    return (), (self.dtype, self.shape)

  @classmethod
  def tree_unflatten(cls, metadata, arrays):
    assert not arrays
    return cls(*metadata)

  def transform_shape(
      self, shape: tuple[int | Array, ...] | None
  ) -> tuple[int | Array, ...] | None:
    del shape  # Unused
    return self.shape

  def transform_dtype(self, dtype):
    del dtype  # Unused
    return self.dtype

  def transform_sharding(self, sharding):
    # If there are no explicit axes, do nothing.
    if all(p is None for p in sharding.spec):
      return sharding
    raise NotImplementedError

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    del context  # Unused.
    return pp.text(f"{{bitcast({self.dtype}{list(self.shape)}])}}")


@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class RefReshaper:
  dtype: dtypes.DType
  shape: tuple[int, ...]

  @classmethod
  def from_ref_new_shape(cls, ref_or_view: Any, *shape: Any) -> RefReshaper:
    if len(shape) == 1 and isinstance(shape[0], tuple):
      shape = shape[0]
    if not shape:
      raise ValueError("Cannot reshape ref to empty shape")
    if any(s == -1 for s in shape):
      num_elements = math.prod(ref_or_view.shape)
      defined_dims = [d for d in shape if d != -1]
      if len(defined_dims) != len(shape) - 1:
        raise ValueError(f"At most one dimension can be -1, but got {shape}")
      if num_elements % math.prod(defined_dims):
        raise ValueError(
            f"Specified dims {shape} do not evenly divide the size of the "
            f"ref ({num_elements})."
        )
      remaining_dim = num_elements // math.prod(defined_dims)
      shape = tuple(d if d != -1 else remaining_dim for d in shape)
    if np.prod(shape) != np.prod(ref_or_view.shape):
      raise TypeError(
          f"cannot reshape ref of shape {ref_or_view.shape} into shape {shape}"
      )
    if isinstance(ref_or_view, TransformedRef):
      if ref_or_view.is_dynamic_size:
        raise NotImplementedError(
            "Reshape ref with dynamic size is not supported."
        )
    dtype = dtypes.dtype(ref_or_view.dtype)
    return cls(dtype, shape)

  @property
  def is_dynamic_size(self):
    return False

  def tree_flatten(self):
    return (), (self.dtype, self.shape)

  @classmethod
  def tree_unflatten(cls, metadata, arrays):
    assert not arrays
    return cls(*metadata)

  def transform_shape(
      self, shape: tuple[int | Array, ...] | None
  ) -> tuple[int | Array, ...] | None:
    del shape  # Unused
    return self.shape

  def transform_dtype(self, dtype: DTypeLike | None) -> DTypeLike | None:
    del dtype  # Unused
    return self.dtype

  def transform_sharding(self, sharding):
    # If there are no explicit axes, do nothing.
    if all(p is None for p in sharding.spec):
      return sharding
    raise NotImplementedError

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    del context  # Unused.
    return pp.text(f"{{reshape({self.dtype}{list(self.shape)})}}")


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RefTransposer:
  permutation: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def from_ref_new_permutation(
      cls, ref_or_view: Any, *perm: int
  ) -> RefTransposer:
    if len(perm) == 1 and isinstance(perm[0], tuple):
      perm = perm[0]
    if len(perm) != ref_or_view.ndim:
      raise ValueError(
          f"Permutation {perm} does not match the rank of the ref"
          f" ({ref_or_view.ndim})"
      )
    return cls(perm)

  def transform_shape(
      self, shape: tuple[int | Array, ...] | None
  ) -> tuple[int | Array, ...] | None:
    if shape is None:
      return None
    return tuple(shape[i] for i in self.permutation)

  def transform_dtype(self, dtype):
    return dtype

  def transform_sharding(self, sharding):
    # If there are no explicit axes, do nothing.
    if all(p is None for p in sharding.spec):
      return sharding
    raise NotImplementedError

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    del context  # Unused.
    return pp.text(f"{{transpose({list(self.permutation)})}}")


class Transform(Protocol):

  def transform_shape(
      self, shape: tuple[int | Array, ...] | None
  ) -> tuple[int | Array, ...] | None:
    """Transform the shape.

    Can return None if the input shape is not known, but must return a concrete
    result when the input shape is known.
    """
    return shape

  def transform_dtype(self, dtype: DTypeLike | None) -> DTypeLike | None:
    """Transform the dtype.

    Can return None if the input dtype is not known, but must return a concrete
    result when the input dtype is known.
    """
    return dtype

  def transform_sharding(self, sharding):
    if all(p is None for p in sharding.spec): return sharding # no explicit axes
    raise NotImplementedError

  def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc:
    return pp.text(f"{{{self}}}")


@dataclasses.dataclass
class RefIndexer:
  ref_or_view: Any

  def __getitem__(self, slc):
    if not isinstance(slc, tuple):
      slc = (slc,)
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

  @property
  def shape(self) -> tuple[int | Array, ...]:
    unprocessed, shape = 0, None
    # We first go backwards to find the first transform that knows its output
    # shape. It's possible none of them do!
    for unprocessed, t in enumerate(reversed(self.transforms), 1):
      if (shape := t.transform_shape(None)) is not None:
        unprocessed -= 1
        break
    if shape is None:
      shape = self.ref.shape
    if not unprocessed:
      return shape
    # If there are any unprocessed transforms left, we apply them to the shape
    # we've found previously.
    for t in self.transforms[-unprocessed:]:
      shape = t.transform_shape(shape)
    assert shape is not None
    return shape

  @property
  def dtype(self):
    # The structure of this method is analogous to `shape`. See comments there.
    unprocessed, dtype = 0, None
    for unprocessed, t in enumerate(reversed(self.transforms), 1):
      if (dtype := t.transform_dtype(None)) is not None:
        unprocessed -= 1
        break
    if dtype is None:
      dtype = self.ref.dtype
    if not unprocessed:
      return dtype
    for t in self.transforms[-unprocessed:]:
      dtype = t.transform_dtype(dtype)
    assert dtype is not None
    return dtype

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: math.prod(self.shape))
  T = property(lambda self: self.transpose(*reversed(range(self.ndim))))

  @property
  def at(self) -> RefIndexer:
    return RefIndexer(self)

  def bitcast(self, dtype):
    return TransformedRef(
        self.ref,
        (*self.transforms, RefBitcaster.from_ref_new_dtype(self, dtype)),
    )

  def reshape(self, *shape):
    return TransformedRef(
        self.ref,
        (*self.transforms, RefReshaper.from_ref_new_shape(self, *shape)),
    )

  def transpose(self, *permutation):
    transposer = RefTransposer.from_ref_new_permutation(self, *permutation)
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


def get_transforms_shape(
    ts: Sequence[Transform], shape: tuple[int | Array, ...]
) -> tuple[int | Array, ...]:
  for t in ts:
    shape = t.transform_shape(shape)  # type: ignore
  assert shape is not None
  return shape


# We need an aval for `Ref`s so we can represent `get` and `swap` in Jaxprs.
class AbstractRef(core.AbstractValue):
  """Abstract mutable array reference.

  Refer to the `Ref guide`_ for more information.

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  __slots__ = ["inner_aval", "memory_space"]

  def __init__(self, inner_aval: core.AbstractValue, memory_space: Any = None):
    self.inner_aval = inner_aval
    self.memory_space = memory_space

  @property
  def is_high(self):
    return self.inner_aval.is_high

  def lo_ty(self):
    return map(AbstractRef, self.inner_aval.lo_ty())

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

  def update(self, inner_aval=None, memory_space=None):
    inner_aval = self.inner_aval if inner_aval is None else inner_aval
    memory_space = self.memory_space if memory_space is None else memory_space
    return AbstractRef(inner_aval, memory_space)

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
    return AbstractRef(self.inner_aval.to_tangent_aval(), self.memory_space)

  def __eq__(self, other):
    return (type(self) is type(other) and self.inner_aval == other.inner_aval
            and self.memory_space == other.memory_space)

  def __hash__(self):
    return hash((self.__class__, self.inner_aval, self.memory_space))

def _map_ref(size, axis, ref_aval):
  return AbstractRef(core.mapped_aval(size, axis, ref_aval.inner_aval),
                     ref_aval.memory_space)

def _unmap_ref(size, axis, explicit_mesh_axis, ref_aval):
  return AbstractRef(core.unmapped_aval(
      size, axis, ref_aval.inner_aval, explicit_mesh_axis),
                     ref_aval.memory_space)

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
