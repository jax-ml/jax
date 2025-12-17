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
from functools import partial
import itertools as it
from typing import Any, Hashable

from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src import ad_util
from jax._src.util import safe_zip, safe_map, split_list
from jax._src.tree_util import tree_flatten, tree_unflatten, tree_leaves, tree_map
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

PyTreeOfAvals = Any
PyTreeDef = Any
LoVal = Any
HiVal = Any


# Hijax extension API

Ty = core.AbstractValue
LoType = core.AbstractValue
QDD = core.QuasiDynamicData
ShapedArray = core.ShapedArray

class HiPrimitive(core.Primitive):
  def __init__(self, name):
    self.name = name
    ad.primitive_jvps[self] = self.jvp
    ad.primitive_transposes[self] = self.transpose

  def is_high(self, *avals, **params) -> bool:
    return True

  def is_effectful(self, params) -> bool:  # type: ignore
    return False  # default immutable

  # type checking and forward type propagation
  def abstract_eval(self, *arg_avals, **params):
    assert False, "must override"

  # lowering implements the primitive in terms of lojax inputs/outputs/ops
  def to_lojax(self, *lotypes_wrapped_in_hitypes, **params):
    assert False, f"must override for {self}"

  # autodiff interface
  def jvp(self, primals, tangents, **params):
    assert False, "must override"
  # transposition is only required if the primitive is linear in some inputs
  def transpose(self, *args, **params):
    assert False, "must override"


class HiType(core.AbstractValue):
  is_high = True
  has_qdd = False  # immutable

  # type equality
  def __hash__(self): assert False, "must override"
  def __eq__(self, other): assert False, "must override"

  # lowering from hijax type to lojax types
  def lo_ty(self) -> list[core.AbstractValue]:
    assert False, "must override"

  # define lowering from hijax value to lojax values and back (like pytrees)
  def lower_val(self, hi_val: HiVal) -> list[LoVal]:  # TODO(mattjj); not lovals
    assert False, "must override"
  def raise_val(self, *lo_vals: LoVal) -> HiVal:
    assert False, "must override"

  # autodiff interface
  def to_tangent_aval(self) -> HiType:
    assert False, "must override"

  # Subclasses should override if the cotangent type is a function of primal
  # type. For example, CT unreduced = reduced and vice-versa.
  def to_cotangent_aval(self) -> HiType:
    return self.to_tangent_aval()

  # the next two are required if this type is itself a tangent type
  def vspace_zero(self) -> HiVal:
    assert False, "must override"

  def vspace_add(self, x: HiVal, y: HiVal) -> HiVal:
    assert False, "must override"

class MutableHiType(core.AbstractValue):
  is_high = True
  has_qdd = True  # mutable and potentially type-changing
  type_state = core.aval_method(core.cur_qdd)

  # type equality
  def __hash__(self): assert False, "must override"
  def __eq__(self, other): assert False, "must override"

  # define lowering from (mutable) hijax type to (immutable) lojax types
  def lo_ty_qdd(self, state: QDD) -> list[core.AbstractValue]:
    assert False, "must override"
  def lo_ty(self):
    assert False, "mutable hitypes should use lo_ty_qdd instead"

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

  # Subclasses should override if the cotangent type is a function of primal
  # type. For example, CT unreduced = reduced and vice-versa.
  def to_cotangent_aval(self) -> HiType:
    return self.to_tangent_aval()

def register_hitype(val_cls, typeof_fn) -> None:
  core.pytype_aval_mappings[val_cls] = typeof_fn
  dtypes.canonicalize_value_handlers[val_cls] = lambda x: x

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
    return Box._new(tree_unflatten(box_state.treedef, hi_vals))  # will be mutated

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

# Override isinstance checks under tracing
class _BoxMeta(type):
  def __instancecheck__(self, instance):
    return (super().__instancecheck__(instance) or
            isinstance(instance, core.Tracer) and
            isinstance(core.typeof(instance), BoxTy))

class Box(metaclass=_BoxMeta):  # noqa: F811
  _val = None  # always clobbered by __new__, but pytype likes this

  # We want `Box(x)` to bind a primitive, so we override __new__ and provide a
  # raw `_new` method below.
  def __new__(cls, init_val=None):
    (), treedef = tree_flatten(None)
    box = new_box_p.bind(treedef=treedef)
    box.set(init_val)
    return box

  @classmethod
  def _new(cls, init_val):
    new = super().__new__(cls)
    new._val = init_val
    return new

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

class BoxEffect(effects.Effect): ...
box_effect = BoxEffect()
effects.control_flow_allowed_effects.add_type(BoxEffect)

class NewBox(HiPrimitive):
  def is_high(self, *, treedef) -> bool: return True  # type: ignore

  def abstract_eval(self, *, treedef):
    leaves, treedef = tree_flatten(None)
    qdd = BoxTypeState(tuple(leaves), treedef)
    return core.AvalQDD(BoxTy(), qdd), {box_effect}

  def to_lojax(_, *, treedef):
    return Box._new(None)

  def jvp(_, primals, tangents, *, treedef):
    assert False  # TODO

  def transpose(_, *args, treedef):
    assert False  # TODO
new_box_p = NewBox('new_box')

class BoxSet(HiPrimitive):
  multiple_results = True

  def is_high(self, *leaf_avals, treedef) -> bool: return True  # type: ignore

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


# === new-style hijax primitive implementation ===

class VJPHiPrimitive:
  in_avals: tuple[PyTreeOfAvals, ...]
  out_aval: PyTreeOfAvals
  params: dict[str, Hashable]

  def __init__(self):
    if not hasattr(self, 'in_avals'):
      raise AttributeError("subclass __init__ should set `self.in_avals`")
    if not hasattr(self, 'out_aval'):
      raise AttributeError("subclass __init__ should set `self.out_aval`")
    if not hasattr(self, 'params'):
      raise AttributeError("subclass __init__ should set `self.params`")
    if (type(self).vjp_bwd is not VJPHiPrimitive.vjp_bwd and
        type(self).vjp_bwd_retval is not VJPHiPrimitive.vjp_bwd_retval):
      raise AttributeError(f"subclass {type(self)} should not override both "
                           "`vjp_bwd` and `vjp_bwd_retval`")
    self.in_avals_flat, self.in_tree = tree_flatten(self.in_avals)
    self.out_avals_flat, self.out_tree = tree_flatten(self.out_aval)
    self.__dict__.update(self.params)

  # Operation implementation in terms of lojax primitives
  def expand(self, *args):
    raise NotImplementedError(f"subclass {type(self)} must implement `expand`")

  def vjp_fwd(self, *args):
    raise NotImplementedError(f"for grad support, subclass {type(self)} must "
                              "implement `vjp_fwd`")

  def vjp_bwd(self, res, outgrad, *arg_accums):
    args_grad = self.vjp_bwd_retval(res, outgrad)
    tree_map(lambda acc, leaf_grad: acc.accum(leaf_grad), arg_accums, args_grad)

  def vjp_bwd_retval(self, res, outgrad):
    # Classic API: returns values instead of using accumulators
    raise NotImplementedError(f"for grad support, subclass {type(self)} must "
                              "implement `vjp_bwd` or `vjp_bwd_retval`")

  def batch(self, axis_data, args, dims):
    raise NotImplementedError(f"for vmap support, subclass {type(self)} must "
                              "implement `batch`")

  def jvp(self, primals, tangents):
    raise NotImplementedError(f"for jvp support, subclass {type(self)} must "
                              "implement `jvp`")

  def __call__(self, *args):
    args_flat = tree_leaves_checked(self.in_tree, args)
    ans_flat = call_hi_primitive_p.bind(*args_flat, prim=self)
    return tree_unflatten(self.out_tree, ans_flat)

  def check(self, *arg_tys):
    # subclass can optionally override this to add checking logic
    return

  def __repr__(self):
    return f"{self.__class__.__name__}[{self.params}]"

  def __hash__(self):
    return hash((self.__class__.__name__, tuple(self.params.items())))

  def __eq__(self, other):
    return type(self) is type(other) and self.params == other.params

def tree_leaves_checked(treedef_expected, tree):
  flat_vals, treedef_actual = tree_flatten(tree)
  assert treedef_actual == treedef_expected
  return flat_vals

call_hi_primitive_p = core.Primitive("call_hi_primitive")
call_hi_primitive_p.multiple_results = True
call_hi_primitive_p.is_high = lambda *args, prim: True  # type: ignore
@call_hi_primitive_p.def_abstract_eval
def _call_hi_primitive_abstract_eval(*_args, prim):
  return prim.out_avals_flat

def _call_hi_primitive_to_lojax(*args_flat, prim):
  args = tree_unflatten(prim.in_tree, args_flat)
  return tree_leaves_checked(prim.out_tree, prim.expand(*args))
call_hi_primitive_p.to_lojax = _call_hi_primitive_to_lojax

def _call_hi_primitive_batcher(axis_data, args_flat, dims_flat, prim):
  args = tree_unflatten(prim.in_tree, args_flat)
  dims = tree_unflatten(prim.in_tree, dims_flat)
  ans, dims = prim.batch(axis_data, args, dims)
  ans_flat = tree_leaves_checked(prim.out_tree, ans)
  dims_flat = prim.out_tree.flatten_up_to(dims)
  return ans_flat, dims_flat
batching.fancy_primitive_batchers[call_hi_primitive_p] = _call_hi_primitive_batcher

def _call_hi_primitive_linearize(nz_in_flat, *args_flat, prim):
  args = tree_unflatten(prim.in_tree, args_flat)
  ans, residuals = prim.vjp_fwd(*args)
  # TODO(dougalm): does the fwd/bwd API force us to assume the nzs_out are all False
  # (except in the case that all the nzs_in are True, which is handled in
  # LinearizeTrace.ProcessPrimitive)?
  ans_flat = tree_leaves_checked(prim.out_tree, ans)
  nzs_out = [True for _ in ans_flat]
  return (ans_flat, nzs_out, residuals, partial(fake_linear_op, prim, nz_in_flat))

def fake_linear_op(prim, nz_in_flat, rs, *tangents):
  residuals_flat, residuals_tree = tree_flatten(rs)
  tangents_flat, _ = tree_flatten(tangents)  # prune symbolic zeros
  return call_hi_primitive_linearized_p.bind(
      *residuals_flat, *tangents_flat,
      residuals_tree=residuals_tree, nz_in_flat=tuple(nz_in_flat), prim=prim)

ad.primitive_linearizations[call_hi_primitive_p] = _call_hi_primitive_linearize

call_hi_primitive_linearized_p = core.Primitive("call_hi_primitive_linearized")
call_hi_primitive_linearized_p.multiple_results = True
call_hi_primitive_linearized_p.is_high = lambda *args, prim, **_: True  # type: ignore
@call_hi_primitive_linearized_p.def_abstract_eval
def _call_hi_primitive_linearized_abstract_eval(*_args, prim, residuals_tree, nz_in_flat):
  return [t.to_tangent_aval() for t in prim.out_avals_flat]  # TODO(dougalm): handle nonzeros

def _call_hi_primitive_linearized_transpose(cts_flat, *args, prim, residuals_tree, nz_in_flat):
  residuals_flat, accums_flat = split_list(args, [residuals_tree.num_leaves])
  residuals = tree_unflatten(residuals_tree, residuals_flat)
  accums_flat_ = iter(accums_flat)
  accums_flat = [next(accums_flat_) if nz else ad.NullAccum() for nz in nz_in_flat]
  assert next(accums_flat_, None) is None
  accums = tree_unflatten(prim.in_tree, accums_flat)
  cts = tree_unflatten(prim.out_tree, cts_flat)
  none = prim.vjp_bwd(residuals, cts, *accums)
  assert none is None
ad.fancy_transposes[call_hi_primitive_linearized_p] = _call_hi_primitive_linearized_transpose

def _call_hi_primitive_jvp(primals, tangents, *, prim):
  primals = tree_unflatten(prim.in_tree, primals)
  tangents = tree_unflatten(prim.in_tree, tangents)
  out_primals, out_tangents =  prim.jvp(primals, tangents)
  out_primals_flat = tree_leaves_checked(prim.out_tree, out_primals)
  out_tangents_flat = prim.out_tree.flatten_up_to(out_tangents)
  return out_primals_flat, out_tangents_flat
ad.primitive_jvps[call_hi_primitive_p] = _call_hi_primitive_jvp
