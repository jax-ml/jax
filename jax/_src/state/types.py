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

from typing import Any, List, Sequence, Set, Union

from jax._src import core
from jax._src import effects
from jax._src import pretty_printer as pp
from jax._src.lib import xla_bridge, xla_client
from jax._src.util import safe_map, safe_zip, tuple_insert, tuple_delete, prod

xc = xla_client
xb = xla_bridge

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

# We need an aval for `Ref`s so we can represent `get` and `swap` in Jaxprs.
# A `ShapedArrayRef` is a abstract value for mutable containers of array types
class ShapedArrayRef(core.AbstractValue):
  __slots__ = ["shape", "dtype"]

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def join(self, other):
    assert core.symbolic_equal_shape(self.shape, other.shape)
    assert self.dtype == other.dtype
    return self

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: prod(self.shape))

  @core.aval_method
  @staticmethod
  def get(tracer, idx=()):
    from jax._src.state.primitives import ref_get
    return ref_get(tracer, idx)

  @core.aval_method
  @staticmethod
  def set(tracer, value, idx=()):
    from jax._src.state.primitives import ref_set
    return ref_set(tracer, idx, value)

  def _getitem(self, tracer, idx) -> Array:
    if not isinstance(idx, tuple):
      idx = idx,
    from jax._src.state.primitives import ref_get
    return ref_get(tracer, idx)

  def _setitem(self, tracer, idx, value) -> None:
    if not isinstance(idx, tuple):
      idx = idx,
    from jax._src.state.primitives import ref_set
    return ref_set(tracer, idx, value)

  def __repr__(self) -> str:
    a = core.ShapedArray(self.shape, self.dtype)
    return f'Ref{{{a.str_short()}}}'

  def at_least_vspace(self):
    return self

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape)

  def __hash__(self):
    return hash((self.shape, self.dtype))


core.raise_to_shaped_mappings[ShapedArrayRef] = lambda aval, _: aval

def _map_ref(size, axis, aval):
  if axis is None: return aval
  return ShapedArrayRef(tuple_delete(aval.shape, axis), aval.dtype)

def _unmap_ref(size, axis_name, axis, aval):
  if axis is None: return aval
  return ShapedArrayRef(tuple_insert(aval.shape, axis, size), aval.dtype)

core.aval_mapping_handlers[ShapedArrayRef] = (_map_ref, _unmap_ref)

def get_ref_state_effects(
    avals: Sequence[core.AbstractValue],
    effects: core.Effects) -> List[Set[StateEffect]]:
  return [{eff for eff in effects
           if isinstance(eff, (ReadEffect, WriteEffect, AccumEffect))
           and eff.input_index == i} for i, aval in enumerate(avals)]
