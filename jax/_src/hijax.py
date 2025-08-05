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

from __future__ import annotations

from dataclasses import dataclass
import itertools as it
from typing import Any

from jax._src import core
from jax._src.effects import Effect
from jax._src.interpreters import ad
from jax._src.interpreters import partial_eval as pe
from jax._src import ad_util
from jax._src.util import safe_zip, safe_map
from jax._src.tree_util import tree_flatten, tree_unflatten, tree_leaves

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

PyTreeDef = Any
LoVal = Any
HiVal = Any


# Hijax extension API

LoType = core.AbstractValue
QDD = core.QuasiDynamicData
ShapedArray = core.ShapedArray

class HiPrimitive(core.Primitive):
  def __init__(self, name):
    self.name = name
    ad.primitive_jvps[self] = self.jvp
    ad.primitive_transposes[self] = self.transpose
    pe.custom_staging_rules[self] = self.staging

  def staging(self, trace, source_info, *args, **kwargs):
    trace.frame.is_high = True
    return trace.default_process_primitive(
        self, args, kwargs, source_info=source_info)

  def is_high(self, **params) -> bool:
    return True

  def is_effectful(self, params) -> bool:  # type: ignore
    return False  # default immutable

  # type checking and forward type propagation
  def abstract_eval(self, *arg_avals, **params):
    assert False, "must override"

  # lowering implements the primitive in terms of lojax inputs/outputs/ops
  def to_lojax(self, *lotypes_wrapped_in_hitypes, **params):
    assert False, "must override"

  # autodiff interface
  def jvp(self, primals, tangents, **params):
    assert False, "must override"
  # transposition is only required if the primitive is linear in some inputs
  def transpose(self, *args, **params):
    assert False, "must override"


class HiType(core.AbstractValue):
  has_qdd = False  # immutable

  # type equality
  def __hash__(self): assert False, "must override"
  def __eq__(self, other): assert False, "must override"

  # lowering from hijax type to lojax types
  def lo_ty(self) -> list[core.AbstractValue]:
    assert False, "must override"

  # define lowering from hijax value to lojax values and back (like pytrees)
  def lower_val(self, hi_val: HiVal) -> list[LoVal]:
    assert False, "must override"
  def raise_val(self, *lo_vals: LoVal) -> HiVal:
    assert False, "must override"

  # autodiff interface
  def to_tangent_aval(self) -> HiType:
    assert False, "must override"
  # the next two are required if this type is itself a tangent type
  def vspace_zero(self) -> HiVal:
    assert False, "must override"
  def vspace_add(self, x: HiVal, y: HiVal) -> HiVal:
    assert False, "must override"

class MutableHiType(core.AbstractValue):
  has_qdd = True  # mutable and potentially type-changing
  type_state = core.aval_method(core.cur_qdd)

  # type equality
  def __hash__(self): assert False, "must override"
  def __eq__(self, other): assert False, "must override"

  # define lowering from (mutable) hijax type to (immutable) lojax types
  def lo_ty_qdd(self, state: QDD) -> list[core.AbstractValue]:
    assert False, "must override"

  # define lowering from hijax value to lojax values and back, depending on qdd
  def new_from_loval(self, state: QDD, *vals: LoVal) -> HiVal:
    assert False, "must override"
  def read_loval(self, state: QDD, val: HiVal) -> list[LoVal]:
    assert False, "must override"

  # define how to mutate/set the mutable hijax value given immutable lojax vals
  def update_from_loval(self, state: QDD, val: HiVal, *lo_vals: LoVal) -> None:
    assert False, "must override"

  # autodiff interface
  def to_tangent_aval(self) -> HiType:
    assert False, "must override"

def register_hitype(val_cls, typeof_fn) -> None:
  core.pytype_aval_mappings[val_cls] = typeof_fn

def hijax_method(f):
  return core.aval_method(f)


# Boxes

## Box API

def new_box():
  (), treedef = tree_flatten(None)
  return new_box_p.bind(treedef=treedef)

def box_get(box):
  tys = core.cur_qdd(box)
  leaf_vals = box_get_p.bind(box, avals=tuple(tys.leaf_avals))
  return tree_unflatten(tys.treedef, leaf_vals)

def box_set(box, val):
  leaves, treedef = tree_flatten(val)
  box_set_p.bind(box, *leaves, treedef=treedef)

## Box implementation

@dataclass(frozen=True)
class BoxTypeState(QDD):
  leaf_avals: tuple[core.AbstractValue, ...]
  treedef: PyTreeDef

  def to_tangent_qdd(self):
    leaf_avals = tuple(a.to_tangent_aval() for a in self.leaf_avals)
    return BoxTypeState(leaf_avals, self.treedef)

  def normalize(self):
    leaf_types = tuple(a.normalize() for a in self.leaf_avals)
    return BoxTypeState(leaf_types, self.treedef)

class BoxTy(MutableHiType):
  has_qdd = True

  # forwarded to value
  get = core.aval_method(box_get)
  set = core.aval_method(box_set)

  # aval interface: hashability and str_short
  def __hash__(self): return hash(BoxTy)
  def __eq__(self, other): return isinstance(other, BoxTy)

  def str_short(self, short_dtypes=False, **_) -> str:  # type: ignore
    return 'BoxTy'

  # mutable interface
  def lo_ty_qdd(self, box_state):
    return [lo_ty for t in box_state.leaf_avals for lo_ty in t.lo_ty()]

  def new_from_loval(self, box_state: BoxTypeState, *lo_vals) -> Box:  # type: ignore
    lo_vals_ = iter(lo_vals)
    hi_vals = [hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))  # type: ignore
               for hi_ty in box_state.leaf_avals]
    assert next(lo_vals_, None) is None
    return Box(tree_unflatten(box_state.treedef, hi_vals))  # will be mutated

  def read_loval(self, box_state: BoxTypeState, box) -> list:  # type: ignore
    leaf_vals, treedef = tree_flatten(box_get(box))
    assert treedef == box_state.treedef
    return [lo_val for hi_ty, hi_val in zip(box_state.leaf_avals, leaf_vals)
            for lo_val in hi_ty.lower_val(hi_val)]  # type: ignore

  def update_from_loval(self, box_state: BoxTypeState, box, *lo_vals) -> None:  # type: ignore
    lo_vals_ = iter(lo_vals)
    hi_vals = [hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))  # type: ignore
               for hi_ty in box_state.leaf_avals]
    assert next(lo_vals_, None) is None
    box_set(box, tree_unflatten(box_state.treedef, hi_vals))

  def to_tangent_aval(self):
    return BoxTy()

class Box:  # noqa: F811
  def __init__(self, val):
    self._val = val

  def get(self):
    return box_get(self)

  def set(self, val):
    box_set(self, val)

  def cur_qdd(self):
    return self.type_state()

  @property
  def ty(self):
    return BoxTy()

  def type_state(self):
    leaves, treedef = tree_flatten(self._val)
    leaf_avals = tuple(map(core.typeof, leaves))
    return BoxTypeState(leaf_avals, treedef)

register_hitype(Box, lambda b: b.ty)

box_effect = Effect()

class NewBox(HiPrimitive):
  def is_high(self, *, treedef) -> bool: return True  # type: ignore

  def abstract_eval(self, *, treedef):
    leaves, treedef = tree_flatten(None)
    qdd = BoxTypeState(tuple(leaves), treedef)
    return core.AvalQDD(BoxTy(), qdd), {box_effect}

  def to_lojax(_, *, treedef):
    return Box(None)

  def jvp(_, primals, tangents, *, treedef):
    assert False  # TODO

  def transpose(_, *args, treedef):
    assert False  # TODO
new_box_p = NewBox('new_box')

class BoxSet(HiPrimitive):
  multiple_results = True

  def is_high(self, *, treedef) -> bool: return True  # type: ignore

  def abstract_eval(self, box_ty, *leaf_avals, treedef):
    box_ty.mutable_qdd.update(BoxTypeState(leaf_avals, treedef))
    return [], {box_effect}  # TODO better typechecking...

  def to_lojax(_, box, *leaves, treedef):
    box._val = tree_unflatten(treedef, leaves)
    return []

  def jvp(_, primals, tangents, *, treedef):
    box, *vals = primals
    box_dot, *val_dots = tangents
    if type(box_dot) is ad_util.Zero:
      raise Exception("can't differentiate Box.set operation, "
                      "did you forget jax.lax.stop_gradient?")
    box_set_p.bind(box, *vals, treedef=treedef)
    box_set_p.bind(box_dot, *val_dots, treedef=treedef)
    return [], []

  def transpose(_, *args, treedef):
    assert False  # TODO
box_set_p = BoxSet('box_set')


class BoxGet(HiPrimitive):
  multiple_results = True

  def abstract_eval(self, box_ty, *, avals):
    return avals, {box_effect}

  def to_lojax(_, box, *, avals):
    return tree_leaves(box._val)

  def jvp(_, primals, tangents, *, avals):
    (box,), (box_dot,) = primals, tangents
    return (
      box_get_p.bind(box, avals=avals),
      box_get_p.bind(box_dot, avals=tuple(a.to_tangent_aval() for a in avals))
    )

  def transpose(_, *args):
    assert False  # TODO
box_get_p = BoxGet('box_get')
