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
import math
from typing import Any, Generic, TypeVar, Union

from jax._src import core
from jax._src import effects
from jax._src import pretty_printer as pp
from jax._src.util import safe_map, safe_zip

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Array = Any

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

StateEffect = Union[ReadEffect, WriteEffect, AccumEffect]

# ## `Ref`s

Aval = TypeVar("Aval", bound=core.AbstractValue)

# We need an aval for `Ref`s so we can represent `get` and `swap` in Jaxprs.
class AbstractRef(core.AbstractValue, Generic[Aval]):
  __slots__ = ["inner_aval"]

  def __init__(self, inner_aval: core.AbstractValue):
    self.inner_aval = inner_aval

  def join(self, other):
    assert isinstance(other, AbstractRef)
    return AbstractRef(self.inner_aval.join(other.inner_aval))

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: math.prod(self.shape))

  @property
  def shape(self):
    if not isinstance(self.inner_aval, core.ShapedArray):
      raise ValueError(f"`Ref{{{self.inner_aval.str_short()}}} has no `shape`.")
    return self.inner_aval.shape

  @property
  def dtype(self):
    if not isinstance(self.inner_aval, core.UnshapedArray):
      raise ValueError(f"`Ref{{{self.inner_aval.str_short()}}} has no `dtype`.")
    return self.inner_aval.dtype

  @core.aval_method
  @staticmethod
  def get(tracer, idx=()):
    from jax._src.state.primitives import ref_get  # pytype: disable=import-error
    return ref_get(tracer, idx)

  @core.aval_method
  @staticmethod
  def set(tracer, value, idx=()):
    from jax._src.state.primitives import ref_set  # pytype: disable=import-error
    return ref_set(tracer, idx, value)

  def _getitem(self, tracer, idx) -> Array:
    if not isinstance(idx, tuple):
      idx = idx,
    from jax._src.state.primitives import ref_get  # pytype: disable=import-error
    return ref_get(tracer, idx)

  def _setitem(self, tracer, idx, value) -> None:
    if not isinstance(idx, tuple):
      idx = idx,
    from jax._src.state.primitives import ref_set  # pytype: disable=import-error
    return ref_set(tracer, idx, value)

  def __repr__(self) -> str:
    return f'Ref{{{self.inner_aval.str_short()}}}'

  def at_least_vspace(self):
    return AbstractRef(self.inner_aval.at_least_vspace())

  def __eq__(self, other):
    return (type(self) is type(other) and self.inner_aval == other.inner_aval)

  def __hash__(self):
    return hash((self.__class__, self.inner_aval))

def _ref_raise_to_shaped(ref_aval: AbstractRef, weak_type):
  return AbstractRef(core.raise_to_shaped(ref_aval.inner_aval, weak_type))
core.raise_to_shaped_mappings[AbstractRef] = _ref_raise_to_shaped

def _map_ref(size, axis, ref_aval):
  return AbstractRef(core.mapped_aval(size, axis, ref_aval.inner_aval))

def _unmap_ref(size, axis_name, axis, ref_aval):
  return AbstractRef(core.unmapped_aval(size, axis_name, axis,
                                        ref_aval.inner_aval))

core.aval_mapping_handlers[AbstractRef] = (_map_ref, _unmap_ref)

def get_ref_state_effects(
    avals: Sequence[core.AbstractValue],
    effects: core.Effects) -> list[set[StateEffect]]:
  return [{eff for eff in effects
           if isinstance(eff, (ReadEffect, WriteEffect, AccumEffect))
           and eff.input_index == i} for i, _ in enumerate(avals)]

def shaped_array_ref(shape: tuple[int, ...], dtype,
                     weak_type: bool = False,
                     named_shape = None) -> AbstractRef[core.AbstractValue]:
  return AbstractRef(core.ShapedArray(shape, dtype, weak_type=weak_type,
                                      named_shape=named_shape))
