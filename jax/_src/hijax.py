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
import inspect
import itertools as it
from typing import Any, Hashable, Callable

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src.api_util import resolve_kwargs, infer_argnums_and_argnames
from jax._src.core import typeof
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.custom_derivatives import CustomVJPPrimal
from jax._src.errors import UnexpectedTracerError
from jax._src.state.types import AbstractRef
from jax._src import ad_util
from jax._src.util import safe_zip, safe_map, split_list, unzip2
from jax._src.tree_util import (
    tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_leaves_checked,
    broadcast_prefix, register_static, tree_structure, tree_map_with_path,
    keystr)
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
effects.custom_derivatives_allowed_effects.add_type(BoxEffect)

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

  def vjp_fwd(self, nzs_in, *args):
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
    out_dim = self.batch_dim_rule(axis_data, dims)
    return VmapOf(self, axis_data, dims, out_dim)(*args), out_dim

  def batch_dim_rule(self, axis_data, dims):
    raise NotImplementedError(f"for vmap support, subclass {type(self)} must "
                              "implement `batch` or `batch_dim_rule`")

  def jvp(self, primals, tangents):
    raise NotImplementedError(f"for jvp support, subclass {type(self)} must "
                              "implement `jvp`")

  def __call__(self, *args):
    args_flat = tree_leaves_checked(self.in_tree, args)
    ans_flat = call_hi_primitive_p.bind(*args_flat, prim=self)
    return tree_unflatten(self.out_tree, ans_flat)

  def check(self, *arg_tys):
    return  # subclass can optionally override this to add checking logic

  def staging(self, trace, source_info, *args):
    args_flat = tree_leaves_checked(self.in_tree, args)
    ans_flat = trace.default_process_primitive(
        call_hi_primitive_p, args_flat, dict(prim=self), source_info)
    return tree_unflatten(self.out_tree, ans_flat)

  def __repr__(self):
    return f"{self.__class__.__name__}[{self.params}]"

  def __hash__(self):
    return hash((self.__class__.__name__, tuple(self.params.items())))

  def __eq__(self, other):
    return type(self) is type(other) and self.params == other.params

class VmapOf(VJPHiPrimitive):
  def __init__(self, prim, axis_data, in_dims, out_dim):
    unmap = lambda a, d: core.unmapped_aval(axis_data.size, d, a,
                                            axis_data.explicit_mesh_axis)
    self.in_avals = tree_map(unmap, prim.in_avals, in_dims)
    self.out_aval = tree_map(unmap, prim.out_aval, out_dim)
    self.params = dict(prim=prim, axis_data=axis_data, in_dims=in_dims,
                       out_dim=out_dim)
    super().__init__()

  @property
  def _vmap_params(self):
    return dict(axis_size=self.axis_data.size, axis_name=self.axis_data.name,  # type: ignore
                spmd_axis_name=self.axis_data.spmd_name or self.axis_data.explicit_mesh_axis)  # type: ignore

  def expand(self, *args):
    return api.vmap(self.prim.expand, in_axes=self.in_dims, out_axes=self.out_dim,  # type: ignore
                    **self._vmap_params)(*args)

  def jvp(self, primals, tangents):
    # TODO probably gonna get non-pytree-prefix errors because of sym zeros...
    return api.vmap(self.prim.jvp, in_axes=(self.in_dims, self.in_dims),  # type: ignore
                    out_axes=(self.out_dim, self.out_dim),  # type: ignore
                    **self._vmap_params)(primals, tangents)  # type: ignore

  def vjp_fwd(self, in_nzs, *args):
    store = lambda: None
    def fwd(*args):
      primal_out, res, *maybe_out_nzs = self.prim.vjp_fwd(in_nzs, *args)  # type: ignore
      store.out_nzs = maybe_out_nzs
      return primal_out, res
    (primal_out, res), (_, res_axes) = api.vmap(
        fwd, in_axes=self.in_dims, out_axes=(self.out_dim, batching.infer),  # type: ignore
        **self._vmap_params)(*args)
    return primal_out, (res, Static(res_axes)), *store.out_nzs  # type: ignore

  def vjp_bwd_retval(self, res_, g):
    # TODO probably gonna get non-pytree-prefix errors because of sym zeros...
    res, res_axes = res_[0], res_[1].val
    in_dims = tree_map(lambda x: batching.sum_axis if x is None else x, self.in_dims,  # type: ignore
                       is_leaf=lambda x: x is None)
    g = tree_map(partial(map_zero, self.axis_data), self.out_dim, g, is_leaf=lambda x: x is None)  # type: ignore
    out = api.vmap(self.prim.vjp_bwd_retval, in_axes=(res_axes, self.out_dim),  # type: ignore
                   out_axes=in_dims, **self._vmap_params, sum_match=True)(res, g)
    return tree_map(partial(unmap_zero, self.axis_data), self.in_dims, out, is_leaf=lambda x: x is None)  # type: ignore

  def batch_dim_rule(self, axis_data, in_dims):
    in_dims_ = tree_map(lambda d, d_: d - (d_ < d), in_dims, self.in_dims)  # type: ignore
    out_dim = self.prim.batch_dim_rule(axis_data, in_dims_)  # type: ignore
    return tree_map(lambda d, d_: d + (d_ < d), out_dim, self.out_dim)  # type: ignore

def map_zero(axis_data, d, ct):
  if isinstance(ct, ad_util.Zero):
    return ad_util.Zero(core.mapped_aval(axis_data.size, d, ct.aval))
  return ct

def unmap_zero(axis_data, d, ct):
  if isinstance(ct, ad_util.Zero):
    return ad_util.Zero(core.unmapped_aval(axis_data.size, d, ct.aval,
                                           axis_data.explicit_mesh_axis))
  return ct


call_hi_primitive_p = core.Primitive("call_hi_primitive")
call_hi_primitive_p.multiple_results = True
call_hi_primitive_p.is_high = lambda *args, prim: True  # type: ignore
@call_hi_primitive_p.def_abstract_eval
def _call_hi_primitive_abstract_eval(*_args, prim):
  return prim.out_avals_flat

def _call_hi_primitive_staging(trace, source_info, *args_flat, prim):
  trace.frame.is_high = True
  args = tree_unflatten(prim.in_tree, args_flat)
  ans = prim.staging(trace, source_info, *args)
  return tree_leaves_checked(prim.out_tree, ans)
pe.custom_staging_rules[call_hi_primitive_p] = _call_hi_primitive_staging

def _call_hi_primitive_to_lojax(*args_flat, prim):
  args = tree_unflatten(prim.in_tree, args_flat)
  ans = prim.expand(*args)
  return tree_leaves_checked(prim.out_tree, ans)
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
  nzs_in = tree_unflatten(prim.in_tree, nz_in_flat)
  ans, residuals, *maybe_nzs_out = prim.vjp_fwd(nzs_in, *args)
  ans_flat = tree_leaves_checked(prim.out_tree, ans)
  nzs_out = True if maybe_nzs_out == [] else maybe_nzs_out[0]
  nzs_out_flat = broadcast_prefix(nzs_out, ans)
  linearized = partial(fake_linear_op, prim, nz_in_flat)
  return (ans_flat, nzs_out_flat, residuals, linearized)

def fake_linear_op(prim, nz_in_flat, rs, *tangents):
  residuals_flat, residuals_tree = tree_flatten(rs)
  assert nz_in_flat == [not isinstance(t, ad_util.Zero) for t in tangents]
  nz_tangents = tree_leaves(tangents)
  return call_hi_primitive_linearized_p.bind(
      *residuals_flat, *nz_tangents, residuals_tree=residuals_tree, prim=prim,
      nz_in_flat=tuple(nz_in_flat))

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
  out_primals, out_tangents = prim.jvp(primals, tangents)
  out_primals_flat = tree_leaves_checked(prim.out_tree, out_primals)
  out_tangents_flat = prim.out_tree.flatten_up_to(out_tangents)
  return out_primals_flat, out_tangents_flat
ad.primitive_jvps[call_hi_primitive_p] = _call_hi_primitive_jvp

def _call_hi_primitive_dce(used_outs_flat, eqn):
  if hasattr(prim := eqn.params['prim'], 'dce'):
    return prim.dce(used_outs_flat, eqn)
  else:
    return pe._default_dce_rule(used_outs_flat, eqn)
pe.dce_rules[call_hi_primitive_p] = _call_hi_primitive_dce

call_hi_primitive_linearized_p.to_lojax = ad.raise_custom_vjp_error_on_jvp
batching.fancy_primitive_batchers[call_hi_primitive_linearized_p] = ad.raise_custom_vjp_error_on_jvp


class CustomVJPTraced(VJPHiPrimitive):
  def __init__(self, traced, fwd, bwd, in_avals, sym_zeros, static_argnums, opt_remat):
    self.in_avals = in_avals
    self.out_aval = traced.out_avals
    self.params = dict(traced=traced, fwd=fwd, bwd=bwd, symbolic_zeros=sym_zeros,
                       static_argnums=static_argnums, opt_remat=opt_remat)
    super().__init__()

  def expand(self, *args):
    args = [x for x in args if not isinstance(x, Static)]
    return self.traced(*args)  # type: ignore

  def vjp_fwd(self, in_nzs, *args):
    in_nzs = tuple(x.val if isinstance(x, Static) else x for x in in_nzs)
    args = tuple(x.val if isinstance(x, Static) else x for x in args)
    if self.symbolic_zeros:  # type: ignore
      args = tree_map(CustomVJPPrimal, args, in_nzs)
    out, res = self.fwd(*args)  # type: ignore
    if ((tree := tree_structure(out)) != self.out_tree):
      raise TypeError(_vjp_primal_fwd_tree_mismatch_err(self, tree))
    tree_map_with_path(_vjp_fwd_aval_mismatch_err, self.out_aval, out)
    if self.symbolic_zeros:  # type: ignore
      out_pairs_flat = tree_leaves_checked(self.out_tree, out)
      out_flat, out_nzs_flat = unzip2(
          (x.value, x.perturbed) if isinstance(x, CustomVJPPrimal) else
          (x, True) for x in out_pairs_flat)
      out_nzs = tree_unflatten(self.out_tree, out_nzs_flat)
      out = tree_unflatten(self.out_tree, out_flat)
      return out, res, out_nzs
    else:
      return out, res

  def vjp_bwd_retval(self, res, out_ct):
    static_args = tuple(x.val for x in self.in_avals if isinstance(x, Static))
    in_avals_ = tuple(x for x in self.in_avals if not isinstance(x, Static))
    leaf = lambda x: isinstance(x, ad_util.Zero)
    if self.symbolic_zeros:  # type: ignore
      out_ct = tree_map(ad_util.replace_internal_symbolic_zeros, out_ct, is_leaf=leaf)
    else:
      out_ct = tree_map(ad_util.instantiate, out_ct, is_leaf=leaf)
    in_cts = self.bwd(*static_args, res, out_ct)  # type: ignore
    if isinstance(in_cts, list):
      in_cts = tuple(in_cts)
    if not isinstance(in_cts, tuple):
      raise TypeError(f"Custom VJP bwd rule {self.bwd} must produce a tuple "  # type: ignore
                      f"but got {type(in_cts)}.")  # type: ignore
    if len(in_cts) != len(self.in_tree.children()) - len(self.static_argnums):  # type: ignore
      raise ValueError(f"Custom VJP bwd rule {self.bwd} must produce a tuple "  # type: ignore
                       "of length equal to the primal args tuple, but got "
                       f"length {len(in_cts)}")  # type: ignore
    in_cts = broadcast_prefix(in_cts, in_avals_, is_leaf=lambda x: x is None)
    in_cts = tree_unflatten(self.in_tree, map(_replace_none, self.in_avals_flat, in_cts))
    tree_map_with_path(_vjp_bwd_aval_mismatch_err, self.in_avals, in_cts)
    if self.symbolic_zeros:  # type: ignore
      in_cts = tree_map(ad_util.replace_rule_output_symbolic_zeros, in_cts)
    return in_cts

  def jvp(self, primals, tangents):
    if self.symbolic_zeros: raise NotImplementedError  # type: ignore
    zero = lambda x: isinstance(x, ad_util.Zero)
    tangents = tree_map(ad_util.instantiate, tangents, is_leaf=zero)
    if self.opt_remat:  # type: ignore
      fwd_traced = api.jit(partial(self.vjp_fwd, (True,) * len(primals))).trace(*primals)
      primals_out, residuals = OptRemat(self.traced, fwd_traced)(*primals)  # type: ignore
    else:
      primals_out, residuals, *_ = self.vjp_fwd((True,) * len(primals), *primals)
    tangents_out_flat = fake_linear_op(self, [True] * len(tangents), residuals, *tangents)
    tangents_out = tree_unflatten(self.out_tree, tangents_out_flat)
    return primals_out, tangents_out

  def batch_dim_rule(self, axis_data, in_dims):
    in_dims_flat = self.in_tree.flatten_up_to(in_dims)
    _, out_dims = batching.batch_jaxpr2(self.traced.jaxpr, axis_data, tuple(in_dims_flat))  # type: ignore
    return tree_unflatten(self.out_tree, out_dims)

def _vjp_primal_fwd_tree_mismatch_err(self, tree):
  return (f"Custom VJP fwd rule {self.fwd.__name__} for function {self.traced.fun_name} "  # type: ignore
          "must produce a pair (list or tuple of length two) where the first "
          "element represents the primal output "
          "(equal to the output of the custom_vjp-decorated function "
          f"{self.traced.fun_name}) and the "  # type: ignore
          "second element represents residuals (i.e. values stored from the "
          "forward pass for use on the backward pass), but "
          f"instead the fwd rule output's first element had container/pytree "
          "structure:\n"
          f"""    {str(tree ).replace("'", "")}\n"""  # type: ignore
          f"while the custom_vjp-decorated function {self.traced.fun_name} had output "  # type: ignore
          "container/pytree structure:\n"
          f"""    {str(self.out_tree).replace("'", "")}.""")  # type: ignore

def _vjp_fwd_aval_mismatch_err(path, primal_aval, fwd_val):
  if not core.typematch(ty := typeof(fwd_val), primal_aval):
    raise TypeError(f"at {keystr(path)}, got fwd output type {ty.str_short()} "
                    f"which doesn't match primal output type {primal_aval.str_short()}")

def _vjp_bwd_aval_mismatch_err(path, primal_aval, ct_val):
  if config.disable_bwd_checks.value: return
  if isinstance(ct_val, ad_util.Zero): return
  if isinstance(primal_aval, AbstractRef): primal_aval = primal_aval.inner_aval
  expected = primal_aval.to_cotangent_aval()
  ty = ct_val.aval if isinstance(ct_val, ad_util.SymbolicZero) else typeof(ct_val)
  if not core.typematch(ty, expected) and getattr(expected, 'dtype', None) is not dtypes.float0:
    result = f"at output{keystr(path)} " if path else ""
    raise ValueError(f"{result}the bwd rule produced an output of type {ty.str_short()} "
                     f"which doesn't match expected type {expected.str_short()}")

def _replace_none(primal_in_aval, maybe_ct):
  if maybe_ct is None:
    return ad_util.Zero(primal_in_aval.to_cotangent_aval())
  else:
    return maybe_ct

class custom_vjp3:
  fwd: Callable | None = None
  bwd: Callable | None = None

  def __init__(self, f, *, nondiff_argnums=(), nondiff_argnames=()):
    self.f = f
    self.static_argnums = _set_up_nondiff(f, nondiff_argnums, nondiff_argnames)

  def defvjp(self, fwd, bwd, *, symbolic_zeros=False, optimize_remat=False):
    self.fwd = fwd
    self.bwd = bwd
    self.symz = symbolic_zeros
    self.opt_remat = optimize_remat
    return self

  def __call__(self, *args, **kwargs):
    if not self.fwd or not self.bwd:
      msg = f"No VJP defined for custom_vjp function {self.f.__name__} using defvjp."
      raise AttributeError(msg)

    args = resolve_kwargs(self.f, args, kwargs)
    if any(isinstance(args[i], core.Tracer) for i in self.static_argnums):
      raise UnexpectedTracerError("custom_vjp inputs marked with nondiff_argnums "
                                  "must be static, not Tracers")
    traced = api.jit(self.f, static_argnums=(*self.static_argnums,)).trace(*args)
    if any(isinstance(x, core.Tracer) for x in traced._consts):
      raise Exception  # TODO(mattjj):error tracer type, value type, primal name
    args = tuple(Static(x) if i in self.static_argnums else x for i, x in enumerate(args))
    in_avals = tree_map(typeof, args)
    prim = CustomVJPTraced(traced, self.fwd, self.bwd, in_avals, self.symz,  # type: ignore
                           self.static_argnums, self.opt_remat)  # type: ignore
    return prim(*args)

class OptRemat(VJPHiPrimitive):
  traced_fwd: Any
  traced_primal: Any

  def __init__(self, traced_primal, traced_fwd):
    self.in_avals, _ = traced_primal.in_avals
    self.out_aval = traced_fwd.out_avals
    self.params = dict(traced_primal=traced_primal, traced_fwd=traced_fwd)
    super().__init__()

  def expand(self, *primals):
    return self.traced_fwd(*primals)

  def dce(self, used_outs, eqn):
    num_primals_in = len(self.traced_primal.jaxpr.in_avals)
    num_primals_out = len(self.traced_primal.jaxpr.out_avals)
    _, used_res = split_list(used_outs, [num_primals_out])
    if any(used_res):
      return [True] * num_primals_in, eqn
    else:
      outvars = [v for used, v in zip(used_outs, eqn.outvars) if used]
      primal_eqn = pe.new_jaxpr_eqn(
          eqn.invars, outvars, core.closed_call_p, dict(call_jaxpr=self.traced_primal.jaxpr),
          self.traced_primal.jaxpr.effects, eqn.source_info, eqn.ctx)
      return [True] * num_primals_in, primal_eqn

  # TODO(mattjj): jvp and transpose? does anyone rely on them?


def _set_up_nondiff(f, argnums_, argnames) -> frozenset[int]:
  argnums = set(argnums_)
  if argnames:
    sig = inspect.signature(f)  # needed for static_argnames
    argnums |= set(infer_argnums_and_argnames(sig, None, argnames)[0])
  return frozenset(argnums)

@register_static
@dataclass(frozen=True)
class Static:
  val: Any
