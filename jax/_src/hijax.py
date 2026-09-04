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
from contextlib import contextmanager
from functools import partial, reduce, update_wrapper
import inspect
from typing import Any, NoReturn, NamedTuple
from collections.abc import Hashable, Callable

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src.api_util import (
    resolve_kwargs, infer_argnums_and_argnames, debug_info, is_hashable,
    dyn_args_fun, WrapHashably)
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src.core import typeof
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import remat
from jax._src.partition_spec import PartitionSpec
from jax._src.custom_derivatives import (
    CustomVJPPrimal, _temporary_dtype_exception, _check_for_returned_refs)
from jax._src.errors import UnexpectedTracerError
from jax._src.state.types import AbstractRef
from jax._src import ad_util
from jax._src.util import (
    safe_zip, safe_map, split_list, unzip2, partition_list, merge_lists)
from jax._src.tree_util import (
    tree_map, tree_flatten, tree_unflatten, tree_leaves, tree_leaves_checked,
    broadcast_prefix, register_static, register_pytree_node, tree_map_with_path,
    keystr, tracing_registry)
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

PyTreeOfAvals = Any
PyTreeDef = Any
LoVal = Any
HiVal = Any

traceback_util.register_exclusion(__file__)


# Hijax extension API

Ty = core.AbstractValue
LoType = core.AbstractValue
ShapedArray = core.ShapedArray

AxisName = Any

def _must_override(ty, method: str, needed_for: str) -> NoReturn:
  raise NotImplementedError(
      f"{needed_for} requires {type(ty).__name__} to implement the "
      f"`{method}` method")

class HiType(core.AbstractValue):
  is_high = True

  # type equality
  def __hash__(self):
    _must_override(self, "__hash__", "type equality")
  def __eq__(self, other):
    _must_override(self, "__eq__", "type equality")

  # lowering from hijax type to lojax types
  def lo_ty(self) -> list[core.AbstractValue]:
    _must_override(self, "lo_ty", "lowering (e.g. under jit)")

  # define lowering from hijax value to lojax values and back (like pytrees)
  def lower_val(self, hi_val: HiVal, /) -> list[LoVal]:  # TODO(mattjj): not lovals
    _must_override(self, "lower_val", "lowering values (e.g. under jit)")
  def raise_val(self, *lo_vals: LoVal) -> HiVal:
    _must_override(self, "raise_val", "raising lowered values (e.g. under jit)")

  # autodiff interface
  def to_tangent_aval(self) -> HiType:
    _must_override(self, "to_tangent_aval", "autodiff")
  def to_ct_aval(self) -> HiType:
    return self.to_tangent_aval()
  # the next two are required if this type is itself a tangent type
  def vspace_zero(self) -> HiVal:
    _must_override(self, "vspace_zero", "use as a tangent/cotangent type")
  def vspace_add(self, x: HiVal, y: HiVal) -> HiVal:
    _must_override(self, "vspace_add", "use as a tangent/cotangent type")

  # vmap interface (also needed for scan)
  def dec_rank(self, size: int | None, spec: MappingSpec) -> HiType:
    _must_override(self, "dec_rank", "vmap")
  def inc_rank(self, size: int | None, spec: MappingSpec) -> HiType:
    _must_override(self, "inc_rank", "vmap")

  # scan interface
  def leading_axis_spec(self) -> MappingSpec:
    _must_override(self, "leading_axis_spec", "scan")

  # shard_map interface
  def shard(self, mesh, manual_axes: frozenset, check_vma: bool, spec: HiPspec
            ) -> HiType:
    _must_override(self, "shard", "shard_map")
  def unshard(self, mesh, check_vma: bool, spec: HiPspec) -> HiType:
    _must_override(self, "unshard", "shard_map")
  def nospec(self, mesh, check_vma: bool, all_names: tuple[AxisName, ...]
             ) -> HiPspec:
    _must_override(self, "nospec", "autodiff through shard_map")


def register_hitype(val_cls, typeof_fn) -> None:
  core.pytype_aval_mappings[val_cls] = typeof_fn
  dtypes.register_canonicalize_value_handler(val_cls, None)

def hijax_method(f):
  return core.aval_method(f)


# === new-style hijax primitive implementation ===

class HiPrim:
  in_avals: tuple[PyTreeOfAvals, ...]
  out_aval: PyTreeOfAvals
  params: dict[str, Hashable]
  effects: frozenset[effects.Effect] = frozenset()

  def __init__(self):
    if not hasattr(self, 'in_avals'):
      raise AttributeError("subclass __init__ should set `self.in_avals`")
    if not hasattr(self, 'out_aval'):
      raise AttributeError("subclass __init__ should set `self.out_aval`")
    if not hasattr(self, 'params'):
      raise AttributeError("subclass __init__ should set `self.params`")
    self.in_avals_flat, self.in_tree = tracing_registry.flatten(self.in_avals)
    self.out_avals_flat, self.out_tree = tracing_registry.flatten(self.out_aval)
    self.__dict__.update(self.params)
    self.check(*self.in_avals)

  # Operation implementation in terms of lojax primitives
  def expand(self, *args):
    raise NotImplementedError(f"subclass {type(self)} must implement `expand`")

  # reverse-mode AD interface
  def vjp_fwd(self, nzs_in, /, *args):
    raise NotImplementedError(
        f"for grad support, subclass {type(self)} must implement `vjp_fwd`, "
        "or derive its reverse-mode rules from its jvp or lin rules by "
        "setting `vjp_fwd, vjp_bwd_retval = vjp_from_jvp` (or `= "
        "vjp_from_lin`)")

  skip_linearization_on_zero_tangents: bool = False

  def vjp_bwd(self, res, outgrad, /, *arg_accums):
    args_grad, logs = self.vjp_bwd_retval_logs(res, outgrad)
    maybe_accum = lambda acc, v: isinstance(acc, ad.GradAccum) and acc.accum(v)
    tree_map(maybe_accum, arg_accums, args_grad)
    return logs

  def vjp_bwd_retval_logs(self, res, outgrad, /):
    return self.vjp_bwd_retval(res, outgrad), None

  def vjp_bwd_retval(self, res, outgrad, /):
    # Classic API: returns values instead of using accumulators
    raise NotImplementedError(
        f"for grad support, subclass {type(self)} must implement `vjp_bwd` or "
        "`vjp_bwd_retval`, or derive its reverse-mode rules by setting "
        "`vjp_fwd, vjp_bwd_retval = vjp_from_jvp` (or `= vjp_from_lin`)")

  # optional forward-mode AD interfaces
  def jvp(self, primals, tangents):
    raise NotImplementedError(f"for jvp support, subclass {type(self)} must "
                              "implement `jvp`")

  def lin(self, nzs_in, *primals):
    raise NotImplementedError(
        f"for linearize support, subclass {type(self)} must implement `lin` "
        "and `linearized`, or derive them from its `jvp` rule by setting "
        "`lin, linearized = linearize_from_jvp`")

  def linearized(self, residuals, *tangents):
    raise NotImplementedError(
        f"for linearize support, subclass {type(self)} must implement `lin` "
        "and `linearized`, or derive them from its `jvp` rule by setting "
        "`lin, linearized = linearize_from_jvp`")

  # optional transpose rule, for primitives that are linear in some inputs
  def transpose(self, out_ct, *maybe_accums):
    raise NotImplementedError(f"for transpose support, subclass {type(self)} "
                              "must implement `transpose`")

  # vmap interface
  def batch(self, axis_data, args, dims):
    out_dim = self.batch_dim_rule(axis_data, dims)
    return VmapOf(self, axis_data, dims, out_dim)(*args), out_dim

  def batch_dim_rule(self, axis_data, dims, /):
    raise NotImplementedError(f"for vmap support, subclass {type(self)} must "
                              "implement `batch` or `batch_dim_rule`")

  # optional dce control
  def dce(self, used_outs):
    used_outs_flat = tree_leaves_checked(self.out_tree, used_outs)
    if not any(used_outs_flat):
      return False, False, None
    else:
      return True, True, self

  # optional remat control
  def remat(self, _trace, *args):
    return self(*args), self  # full remat by default

  def __call__(self, *args):
    args_flat = tree_leaves_checked(self.in_tree, args)
    ans_flat = call_hi_primitive_p.bind(*args_flat, _prim=self)
    return tree_unflatten(self.out_tree, ans_flat)

  def check(self, *arg_tys):
    return  # subclass can optionally override this to add checking logic

  def staging(self, trace, source_info, *args):
    args_flat = tree_leaves_checked(self.in_tree, args)
    ans_flat = trace.default_process_primitive(
        call_hi_primitive_p, args_flat, dict(_prim=self), source_info)
    return tree_unflatten(self.out_tree, ans_flat)

  def __repr__(self):
    return f"{self.__class__.__name__}[{self.params}]"

  def __hash__(self):
    return hash((self.__class__.__name__, tuple(self.params.items()), self.effects))

  def __eq__(self, other):
    return (type(self) is type(other) and self.params == other.params
            and self.effects == other.effects)


class VmapOf(HiPrim):
  prim: core.Primitive
  axis_data: batching.AxisData
  in_dims: Any
  out_dim: Any

  def __init__(self, prim, axis_data, in_dims, out_dim):
    self.skip_linearization_on_zero_tangents = prim.skip_linearization_on_zero_tangents
    unmap = lambda a, d: core.unmapped_aval(axis_data.size, d, a,
                                            axis_data.explicit_mesh_axis)
    self.in_avals = tree_map(unmap, prim.in_avals, in_dims)
    self.out_aval = tree_map(unmap, prim.out_aval, out_dim)
    self.params = dict(prim=prim, axis_data=axis_data, in_dims=in_dims,
                       out_dim=out_dim)
    super().__init__()

  @property
  def _vmap_params(self):
    return dict(axis_size=self.axis_data.size, axis_name=self.axis_data.name,
                spmd_axis_name=self.axis_data.spmd_name or self.axis_data.explicit_mesh_axis)

  def expand(self, *args):
    return api.vmap(self.prim.expand, in_axes=self.in_dims, out_axes=self.out_dim,  # pyrefly: ignore[missing-attribute]
                    **self._vmap_params)(*args)

  def jvp(self, primals, tangents):
    tangents = tree_map(partial(map_zero, self.axis_data), self.in_dims,
                        tangents, is_leaf=lambda x: x is None)
    with _explain_overbatched_member(self.prim, 'jvp rule'):
      primals_out, tangents_out = api.vmap(
          self.prim.jvp, in_axes=(self.in_dims, self.in_dims),  # pyrefly: ignore[missing-attribute]
          out_axes=(self.out_dim, self.out_dim),
          **self._vmap_params)(primals, tangents)
    tangents_out = tree_map(partial(unmap_zero, self.axis_data), self.out_dim,
                            tangents_out, is_leaf=lambda x: x is None)
    return primals_out, tangents_out

  def vjp_fwd(self, in_nzs, *args):
    store = lambda: None
    def fwd(*args):
      primal_out, res, *maybe_out_nzs = self.prim.vjp_fwd(in_nzs, *args)  # pyrefly: ignore[missing-attribute]
      store.out_nzs = maybe_out_nzs  # pyrefly: ignore[missing-attribute]
      return primal_out, res
    with _explain_overbatched_member(self.prim, 'fwd rule'):
      (primal_out, res), (_, res_axes) = api.vmap(
          fwd, in_axes=self.in_dims, out_axes=(self.out_dim, batching.infer),
          **self._vmap_params)(*args)
    return primal_out, (res, Static(res_axes)), *store.out_nzs  # pyrefly: ignore[missing-attribute]

  def vjp_bwd_retval_logs(self, res_, g):
    # TODO probably gonna get non-pytree-prefix errors because of sym zeros...
    res, res_axes = res_[0], res_[1].val
    in_dims = tree_map(lambda x: batching.sum_axis if x is None else x, self.in_dims,
                       is_leaf=lambda x: x is None)
    g = tree_map(partial(map_zero, self.axis_data), self.out_dim, g, is_leaf=lambda x: x is None)
    out, logs = api.vmap(self.prim.vjp_bwd_retval_logs, in_axes=(res_axes, self.out_dim),  # pyrefly: ignore[missing-attribute]
                         out_axes=(in_dims, 0), **self._vmap_params, sum_match=True)(res, g)
    out = tree_map(partial(unmap_zero, self.axis_data), self.in_dims, out, is_leaf=lambda x: x is None)
    return out, logs

  def batch_dim_rule(self, axis_data, in_dims):
    fix = lambda d, d_: d if (d is None or d_ is None) else d - (d_ < d)
    in_dims_ = tree_map(fix, in_dims, self.in_dims, is_leaf=lambda x: x is None)
    out_dim = self.prim.batch_dim_rule(axis_data, in_dims_)  # pyrefly: ignore[missing-attribute]
    unfix = lambda d, d_: d if (d is None or d_ is None) else d + (d_ < d)
    return tree_map(unfix, out_dim, self.out_dim, is_leaf=lambda x: x is None)

@contextmanager
def _explain_overbatched_member(prim, member_name):
  try:
    yield
  except ValueError as e:
    if ('but output was batched' not in str(e) and
        'vmap has mapped output' not in str(e)):
      raise
    name = getattr(getattr(prim, 'traced', None), 'fun_name',
                   type(prim).__name__)
    raise ValueError(
        f"under vmap, the {member_name} of {name} produced an output batched "
        "along the mapped axis where the application itself was inferred to "
        "be unbatched. The batchedness of a custom_jvp/custom_vjp "
        "application under vmap is inferred from its primal function alone, "
        "but a rule may produce more-batched outputs (e.g. if a tangent "
        "depends on a batched input that the primal output does not use). "
        "To support that, define the operation as a "
        "jax.experimental.hijax.HiPrim and override its "
        "`batch_dim_rule` (or `batch`) method to declare the batched "
        "outputs.") from e

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
call_hi_primitive_p.skip_canonicalization = True
call_hi_primitive_p.is_high = lambda *args, _prim: True
call_hi_primitive_p.is_effectful = lambda params: bool(params['_prim'].effects)
@call_hi_primitive_p.def_effectful_abstract_eval
def _call_hi_primitive_abstract_eval(*_args, _prim):
  return _prim.out_avals_flat, _prim.effects

def _call_hi_primitive_typecheck(_ctx_factory, *in_atoms_flat, _prim):
  in_avals = [x.aval for x in in_atoms_flat]
  if not all(map(core.typematch, in_avals, _prim.in_avals_flat)):
    raise TypeError(f"input type mismatch for {_prim}")
  _prim.check()
  return _prim.out_avals_flat, _prim.effects
core.custom_typechecks[call_hi_primitive_p] = _call_hi_primitive_typecheck

def _call_hi_primitive_staging(trace, source_info, *args_flat, _prim):
  trace.frame.is_high = True
  args = tree_unflatten(_prim.in_tree, args_flat)
  ans = _prim.staging(trace, source_info, *args)
  return tree_leaves_checked(_prim.out_tree, ans)
pe.custom_staging_rules[call_hi_primitive_p] = _call_hi_primitive_staging

def _call_hi_primitive_to_lojax(*args_flat, _prim):
  args = tree_unflatten(_prim.in_tree, args_flat)
  ans = _prim.expand(*args)
  return tree_leaves_checked(_prim.out_tree, ans)
call_hi_primitive_p.to_lojax = _call_hi_primitive_to_lojax

def _call_hi_primitive_prettyprint(eqn, context, settings):
  # print CustomVJPTraced/CustomJVPTraced tersely since their params reprs are
  # noise (Traced objects, functions), but let prims like RematTraced print in
  # full since their repr shows the inner jaxpr
  if isinstance(eqn.params['_prim'], (CustomVJPTraced, CustomJVPTraced)):
    params = dict(eqn.params, _prim=eqn.params['_prim'].__class__.__name__)
    eqn = eqn.replace(params=params)
  return core._pp_eqn(eqn, context, settings)
core.pp_eqn_rules[call_hi_primitive_p] = _call_hi_primitive_prettyprint

def _call_hi_primitive_batcher(axis_data, args_flat, dims_flat, _prim):
  args = tree_unflatten(_prim.in_tree, args_flat)
  dims = tree_unflatten(_prim.in_tree, dims_flat)
  ans, dims = _prim.batch(axis_data, args, dims)
  ans_flat = tree_leaves_checked(_prim.out_tree, ans)
  dims_flat = _prim.out_tree.flatten_up_to(dims)
  return ans_flat, dims_flat
batching.fancy_primitive_batchers[call_hi_primitive_p] = _call_hi_primitive_batcher

# A `lin` or `vjp_fwd` rule may return (ans, res), (ans, res, nzs_out), or
# (ans, res, nzs_out, sres). When it returns structured residuals, the paired
# backward rule receives them explicitly: `linearized(res, sres, *tangents)`,
# and `vjp_bwd(res, sres, outgrad, *arg_accums)` (which must be overridden).
def _call_hi_primitive_linearize(is_vjp, nz_in_flat, *args_flat, _prim):
  zero_in = (not any(nz_in_flat) and
             not any(isinstance(typeof(x), AbstractRef) for x in args_flat))
  if zero_in and _prim.skip_linearization_on_zero_tangents:
    ans_flat = call_hi_primitive_p.bind(*args_flat, _prim=_prim)
    linearized = lambda _, __, *ts: [ad_util.p2tz(x) for x in ans_flat]
    return ans_flat, [False] * len(ans_flat), None, None, linearized
  args = tree_unflatten(_prim.in_tree, args_flat)
  nzs_in = tree_unflatten(_prim.in_tree, nz_in_flat)
  if is_vjp:
    ans, residuals, *rest = _prim.vjp_fwd(nzs_in, *args)
    linearized = partial(fake_linear_op, _prim, nz_in_flat)
  else:
    ans, residuals, *rest = _prim.lin(nzs_in, *args)
    linearized = partial(flatten_user_linearized, _prim)
  ans_flat = tree_leaves_checked(_prim.out_tree, ans)
  nzs_out = rest[0] if rest else not zero_in
  sres = rest[1] if len(rest) > 1 else None
  if (sres is not None and is_vjp and
      type(_prim).vjp_bwd is HiPrim.vjp_bwd):
    raise TypeError(
        f"{type(_prim).__name__} returned structured residuals from `vjp_fwd`, "
        "which requires overriding `vjp_bwd(res, sres, outgrad, *arg_accums)`")
  nzs_out_flat = broadcast_prefix(nzs_out, ans)
  linearized = partial(linearized, nzs_out_flat) if is_vjp else linearized
  return ans_flat, nzs_out_flat, residuals, sres, linearized
ad.primitive_linearizations[call_hi_primitive_p] = _call_hi_primitive_linearize
ad.linearize_on_zero_tangents.add(call_hi_primitive_p)

def fake_linear_op(prim, nz_in_flat, nz_out_flat, rs, sres, *tangents):
  if not any(nz_out_flat):
    return [ad_util.Zero(a.to_tangent_aval()) for a in prim.out_avals_flat]
  rs = rs if sres is None else (rs, sres)  # unpacked in the transpose rule
  residuals_flat, residuals_tree = tree_flatten(rs)
  assert nz_in_flat == [not isinstance(t, ad_util.Zero) for t in tangents]
  nz_tangents = tree_leaves(tangents)
  out_nz = call_hi_primitive_linearized_p.bind(
      *residuals_flat, *nz_tangents, residuals_tree=residuals_tree, _prim=prim,
      nz_in_flat=tuple(nz_in_flat), nz_out_flat=tuple(nz_out_flat),
      has_sres=sres is not None)
  out_nz_iter = iter(out_nz)
  out = [next(out_nz_iter) if nz else ad_util.Zero(a.to_tangent_aval())
         for a, nz in zip(prim.out_avals_flat, nz_out_flat)]
  assert next(out_nz_iter, sentinel := object()) is sentinel
  return out

def flatten_user_linearized(prim, residuals, sres, *tangents_flat):
  tangents = tree_unflatten(prim.in_tree, tangents_flat)
  tangents_out = (prim.linearized(residuals, *tangents) if sres is None else
                  prim.linearized(residuals, sres, *tangents))
  flat_vals, treedef_actual = tracing_registry.flatten(
      tangents_out, lambda x: isinstance(x, ad_util.Zero))
  if treedef_actual != prim.out_tree:
    raise RuntimeError(
        f"tree mismatch during linearization of {prim=}."
        f" Expected: {prim.out_tree} got: {treedef_actual}"
    )
  return flat_vals

call_hi_primitive_linearized_p = core.Primitive("call_hi_primitive_linearized")
call_hi_primitive_linearized_p.multiple_results = True
call_hi_primitive_linearized_p.is_high = lambda *args, _prim, **_: True
@call_hi_primitive_linearized_p.def_abstract_eval
def _call_hi_primitive_linearized_abstract_eval(
    *_args, _prim, residuals_tree, nz_in_flat, nz_out_flat, has_sres):
  return [t.to_tangent_aval() for t, nz in zip(_prim.out_avals_flat, nz_out_flat) if nz]

def _call_hi_primitive_linearized_transpose(
    cts_flat_, *args, _prim, residuals_tree, nz_in_flat, nz_out_flat, has_sres):
  residuals_flat, accums_flat = split_list(args, [residuals_tree.num_leaves])
  residuals = tree_unflatten(residuals_tree, residuals_flat)
  accums_flat_ = iter(accums_flat)
  accums_flat = [next(accums_flat_) if nz else ad.NullAccum(aval.to_ct_aval())
                 for aval, nz in zip(_prim.in_avals_flat, nz_in_flat)]
  assert next(accums_flat_, None) is None
  accums = tree_unflatten(_prim.in_tree, accums_flat)
  cts_flat_iter = iter(cts_flat_)
  cts_flat = [next(cts_flat_iter) if nz else ad_util.Zero(a.to_ct_aval())
              for a, nz in zip(_prim.out_avals_flat, nz_out_flat)]
  assert next(cts_flat_iter, sentinel := object()) is sentinel
  cts = tree_unflatten(_prim.out_tree, cts_flat)
  # A vjp_bwd rule may return a dict of pytrees to log out of the backward
  # pass (see VJP.with_logs), or None (the usual case) to log nothing.
  if has_sres:
    residuals, sres = residuals
    log = _prim.vjp_bwd(residuals, sres, cts, *accums)
  else:
    log = _prim.vjp_bwd(residuals, cts, *accums)
  if log is not None and type(log) is not dict:
    raise TypeError(
        f"{type(_prim).__name__}.vjp_bwd should return None or a dict of "
        f"backward-pass log entries, got {type(log).__name__}")
  return log
ad.fancy_transposes[call_hi_primitive_linearized_p] = _call_hi_primitive_linearized_transpose

def _call_hi_primitive_linearized_prettyprint(eqn, context, settings):
  params = dict(eqn.params, _prim=eqn.params['_prim'].__class__.__name__,
                residuals_tree='...')
  if not params['has_sres']:
    del params['has_sres']
  return core._pp_eqn(eqn.replace(params=params), context, settings)
core.pp_eqn_rules[call_hi_primitive_linearized_p] = _call_hi_primitive_linearized_prettyprint

def _call_hi_primitive_jvp(primals, tangents, *, _prim):
  primals = tree_unflatten(_prim.in_tree, primals)
  tangents = tree_unflatten(_prim.in_tree, tangents)
  out_primals, out_tangents = _prim.jvp(primals, tangents)
  out_primals_flat = tree_leaves_checked(_prim.out_tree, out_primals)
  out_tangents_flat = _prim.out_tree.flatten_up_to(out_tangents)
  return out_primals_flat, out_tangents_flat
ad.primitive_jvps[call_hi_primitive_p] = _call_hi_primitive_jvp

def _call_hi_primitive_transpose(cts_flat, *primals_flat, _prim):
  cts = tree_unflatten(_prim.out_tree, cts_flat)
  primals = tree_unflatten(_prim.in_tree, primals_flat)
  log = _prim.transpose(cts, *primals)  # a returned dict logs entries
  if log is not None and type(log) is not dict:
    raise TypeError(
        f"{type(_prim).__name__}.transpose should return None or a dict of "
        f"backward-pass log entries, got {type(log).__name__}")
  return log
ad.fancy_transposes[call_hi_primitive_p] = _call_hi_primitive_transpose

def _call_hi_primitive_dce(used_outs_flat, eqn):
  _prim = eqn.params['_prim']
  used_out = tree_unflatten(_prim.out_tree, used_outs_flat)
  used_ins, produced_outs, new_prim = _prim.dce(used_out)
  if new_prim is None:
    return [False] * len(eqn.invars), None
  name = f'{type(_prim).__name__}.dce'
  used_ins_flat = api.tuptree_flags(
      used_ins, _prim.in_tree, 'used_ins',
      f'the first (used inputs) return value of {name}')
  produced_outs_flat = api.tuptree_flags(
      produced_outs, _prim.out_tree, 'produced_outs',
      f'the second (produced outputs) return value of {name}')
  new_invars = [x for x, u in zip(eqn.invars, used_ins_flat) if u]
  new_outvars = [v for v, u in zip(eqn.outvars, produced_outs_flat) if u]
  new_eqn = eqn.replace(invars=new_invars, outvars=new_outvars,
                        params=dict(_prim=new_prim))
  return used_ins_flat, new_eqn
pe.dce_rules[call_hi_primitive_p] = _call_hi_primitive_dce

call_hi_primitive_linearized_p.to_lojax = ad.raise_custom_vjp_error_on_jvp
batching.fancy_primitive_batchers[call_hi_primitive_linearized_p] = ad.raise_custom_vjp_error_on_jvp

def _call_hi_primitive_remat(trace, *args_flat, _prim):
  args = tree_unflatten(_prim.in_tree, args_flat)
  out, rem_ = _prim.remat(trace, *args)
  def rem(*args_flat):
    args = tree_unflatten(_prim.in_tree, args_flat)
    out = rem_(*args)
    return tree_leaves_checked(_prim.out_tree, out)
  return tree_leaves_checked(_prim.out_tree, out), rem
remat.rules[call_hi_primitive_p] = _call_hi_primitive_remat


# === deriving lin and vjp rules from jvp and lin rules ===

class DerivedLinearization:
  """Residuals of `linearize_from_jvp`, closing over the linear map."""
  __slots__ = ['consts', 'apply']

  def __init__(self, consts, apply):
    self.consts = consts
    self.apply = apply

register_pytree_node(DerivedLinearization,
                     lambda res: ((res.consts,), res.apply),
                     lambda apply, children: DerivedLinearization(children[0], apply))

def _lin_from_jvp(self, nzs_in, *primals):
  """The `lin` half of the `linearize_from_jvp` pair."""
  primals_flat = tree_leaves_checked(self.in_tree, primals)
  nzs_in_flat = tree_leaves_checked(self.in_tree, nzs_in)

  def jvp_flat(primals_flat, tangents_flat):
    primals = tree_unflatten(self.in_tree, primals_flat)
    tangents = tree_unflatten(self.in_tree, tangents_flat)
    out_primals, out_tangents = self.jvp(primals, tangents)
    out_primals_flat = tree_leaves_checked(self.out_tree, out_primals)
    out_tangents_flat = self.out_tree.flatten_up_to(out_tangents)
    return out_primals_flat, out_tangents_flat

  dbg = debug_info('linearize_from_jvp', self.jvp, (primals, primals), {})
  out_primals_flat, nzs_out_flat, consts, _, linearized = ad.linearize_from_jvp(
      lu.wrap_init(jvp_flat, debug_info=dbg), True, nzs_in_flat,
      False, False, primals_flat, {})
  out_primals = tree_unflatten(self.out_tree, out_primals_flat)
  nzs_out = tree_unflatten(self.out_tree, list(nzs_out_flat))
  return out_primals, DerivedLinearization(consts, linearized), nzs_out

def _linearized_from_jvp(self, residuals, *tangents):
  """The `linearized` half of the `linearize_from_jvp` pair."""
  tangents_flat = self.in_tree.flatten_up_to(tangents)
  out_tangents_flat = residuals.apply(residuals.consts, None, *tangents_flat)
  return tree_unflatten(self.out_tree, out_tangents_flat)

def jvp_from_lin(self, primals, tangents):
  if type(self).lin is _lin_from_jvp:
    raise TypeError(
        f"subclass {type(self)} can't set both `jvp = jvp_from_lin` and "
        "`lin, linearized = linearize_from_jvp`, since each would be defined "
        "in terms of the other")
  tangents_flat = self.in_tree.flatten_up_to(tangents)
  nzs_in = tree_unflatten(
      self.in_tree, [not isinstance(t, ad_util.Zero) for t in tangents_flat])
  out_primals, residuals, *rest = self.lin(nzs_in, *primals)
  out_tangents = (self.linearized(residuals, *tangents) if len(rest) < 2 else
                  self.linearized(residuals, rest[1], *tangents))
  return out_primals, out_tangents

def _vjp_fwd_from_jvp(self, nzs_in, *primals):
  """The `vjp_fwd` half of the `vjp_from_jvp` pair."""
  return self(*primals), (primals, nzs_in)

def _transpose_jvp(self, res, out_ct):
  """The `vjp_bwd_retval` half of the `vjp_from_jvp` pair."""
  primals, nzs_in = res
  nzs_flat = tree_leaves_checked(self.in_tree, nzs_in)
  zero = lambda x: isinstance(x, (ad_util.Zero, ad_util.SymbolicZero))
  inst = lambda x: ad_util.zeros_like_aval(x.aval) if zero(x) else x

  def tangent_map(*nz_tangents_flat):
    nz_ = iter(nz_tangents_flat)
    tangents_flat = [next(nz_) if nz else ad_util.Zero(a.to_tangent_aval())
                     for a, nz in zip(self.in_avals_flat, nzs_flat)]
    assert next(nz_, None) is None
    tangents = tree_unflatten(self.in_tree, tangents_flat)
    _, out_tangents = self.jvp(primals, tangents)
    return tree_map(inst, out_tangents, is_leaf=zero)

  out_ct = tree_map(ad_util.instantiate, out_ct, is_leaf=zero)
  dummies = [ad_util.zeros_like_aval(a.to_tangent_aval())
             for a, nz in zip(self.in_avals_flat, nzs_flat) if nz]
  in_cts_nz = api.linear_transpose(tangent_map, *dummies)(out_ct)
  in_cts_nz_ = iter(in_cts_nz)
  in_cts_flat = [next(in_cts_nz_) if nz else ad_util.Zero(a.to_tangent_aval())
                 for a, nz in zip(self.in_avals_flat, nzs_flat)]
  assert next(in_cts_nz_, None) is None
  return tree_unflatten(self.in_tree, in_cts_flat)

def _vjp_fwd_from_lin(self, nzs_in, *primals):
  """The `vjp_fwd` half of the `vjp_from_lin` pair."""
  return self.lin(nzs_in, *primals)

def _transpose_linearized(self, residuals, out_ct):
  """The `vjp_bwd_retval` half of the `vjp_from_lin` pair."""
  def tangent_map(*tangents):
    return self.linearized(residuals, *tangents)
  zero = lambda x: isinstance(x, ad_util.Zero)
  out_ct = tree_map(ad_util.instantiate, out_ct, is_leaf=zero)
  dummies = tree_map(lambda a: ad_util.zeros_like_aval(a.to_tangent_aval()),
                     self.in_avals)
  return api.linear_transpose(tangent_map, *dummies)(out_ct)

class _LinearizeFromJVP(NamedTuple):
  lin: Callable
  linearized: Callable
  def __call__(self, *args, **kwargs):
    raise TypeError(
        "`linearize_from_jvp` is a pair of rules, not a single rule; unpack "
        "it in the class body: `lin, linearized = linearize_from_jvp`")

class _VJPFromJVP(NamedTuple):
  vjp_fwd: Callable
  vjp_bwd_retval: Callable
  def __call__(self, *args, **kwargs):
    raise TypeError(
        "`vjp_from_jvp` is a pair of rules, not a single rule; unpack it in "
        "the class body: `vjp_fwd, vjp_bwd_retval = vjp_from_jvp`")

class _VJPFromLin(NamedTuple):
  vjp_fwd: Callable
  vjp_bwd_retval: Callable
  def __call__(self, *args, **kwargs):
    raise TypeError(
        "`vjp_from_lin` is a pair of rules, not a single rule; unpack it in "
        "the class body: `vjp_fwd, vjp_bwd_retval = vjp_from_lin`")

linearize_from_jvp = _LinearizeFromJVP(_lin_from_jvp, _linearized_from_jvp)
vjp_from_jvp = _VJPFromJVP(_vjp_fwd_from_jvp, _transpose_jvp)
vjp_from_lin = _VJPFromLin(_vjp_fwd_from_lin, _transpose_linearized)


class CustomVJPTraced(HiPrim):
  skip_linearization_on_zero_tangents = True  # run the primal, not the fwd rule
  """Applications take ``(consts, fwd_consts, *args)``.

  The two leading arguments are synthetic, and both get zero cotangents:
  ``consts`` is the primal function's closure environment promoted to an
  argument (see ``Traced.with_consts_as_arg``), and ``fwd_consts`` is extra
  inputs consumed only by the fwd rule and ignored by the primal (ordinarily
  ``()``; the ``remat`` rule uses it to pass replay residuals to the helper
  primitive it builds). Any value a rule needs beyond the primal arguments
  must arrive through these slots as an explicit input, never by closure: the
  rules are re-invoked by transformations after the trace of application time
  is gone, so closed-over tracers go stale. The primal ``traced`` takes
  ``(consts, *args)``; ``drop_fwd_consts`` maps the application signature
  onto it.
  """
  traced: Any
  fwd: Any
  bwd: Any
  symbolic_zeros: Any
  static_argnums: Any
  opt_remat: bool
  with_logs: bool

  @staticmethod
  def drop_fwd_consts(consts, fwd_consts, *args):
    del fwd_consts
    return (consts, *args)

  def __init__(self, traced, fwd, bwd, in_avals, sym_zeros, static_argnums,
               opt_remat, with_logs=False):
    self.in_avals = in_avals
    self.out_aval = traced.out_avals
    self.effects = traced.effects
    self.params = dict(traced=traced, fwd=fwd, bwd=bwd, symbolic_zeros=sym_zeros,
                       static_argnums=static_argnums, opt_remat=opt_remat,
                       with_logs=with_logs)
    super().__init__()

  def expand(self, *args):
    args = self.drop_fwd_consts(*args)
    return self.traced(*[x for x in args if not isinstance(x, Static)])

  def lin(self, nzs_in, *primals):
    out, res, *rest = self.vjp_fwd(nzs_in, *primals)
    nzs_out = rest[0] if rest else True
    nzs_out_flat = broadcast_prefix(nzs_out, out)
    return out, (res, tuple(nzs_out_flat)), nzs_out

  def linearized(self, residuals, *tangents):  # pyrefly: ignore[bad-param-name-override]
    res, nz_out_flat = residuals
    tangents_flat, in_tree = tracing_registry.flatten(
        tangents, lambda x: isinstance(x, ad_util.Zero))
    assert in_tree == self.in_tree
    nz_in_flat = [not isinstance(t, ad_util.Zero) for t in tangents_flat]
    outs_flat = fake_linear_op(self, nz_in_flat, list(nz_out_flat), res, None,
                               *tangents_flat)
    return tree_unflatten(self.out_tree, outs_flat)

  def vjp_fwd(self, in_nzs, *args):
    if any(tree_leaves(in_nzs[0])):
      raise ad.CustomVJPException()
    if self.symbolic_zeros:
      args = tree_map(CustomVJPPrimal, args, in_nzs)  # tree_map skips Statics
    args_ = tuple(x.val if isinstance(x, Static) else x for x in args)
    out, res = self.fwd(*args_)
    if config.mutable_array_checks.value:
      _check_for_returned_refs(self.fwd, (out, res), "fwd", tree_leaves(args),
                               self.out_tree.num_leaves)
    if ((tree := tracing_registry.flatten(out)[1]) != self.out_tree):
      raise TypeError(_vjp_primal_fwd_tree_mismatch_err(self, tree))
    tree_map_with_path(_vjp_fwd_aval_mismatch_err, self.out_aval, out)
    if self.symbolic_zeros:
      out_pairs_flat = tree_leaves_checked(self.out_tree, out)
      out_flat, out_nzs_flat = unzip2(
          (x.value, x.perturbed) if isinstance(x, CustomVJPPrimal) else
          (x, True) for x in out_pairs_flat)
      out_nzs = tree_unflatten(self.out_tree, out_nzs_flat)
      out = tree_unflatten(self.out_tree, out_flat)
      return out, res, out_nzs
    else:
      return out, res

  def vjp_bwd_retval_logs(self, res, out_ct):
    static_args = tuple(x.val for x in self.in_avals if isinstance(x, Static))
    in_avals_ = tuple(x for x in self.in_avals if not isinstance(x, Static))
    leaf = lambda x: isinstance(x, ad_util.Zero)
    if self.symbolic_zeros:
      out_ct = tree_map(ad_util.replace_internal_symbolic_zeros, out_ct, is_leaf=leaf)
    else:
      out_ct = tree_map(ad_util.instantiate, out_ct, is_leaf=leaf)
    in_cts = self.bwd(*static_args, res, out_ct)
    logs = None
    if self.with_logs:
      if not (isinstance(in_cts, (list, tuple)) and len(in_cts) == 2):
        raise TypeError(
            f"Custom VJP bwd rule {self.bwd} was registered with "
            "defvjp_with_logs and so must produce a pair (in_cts, logs), "
            f"but got {in_cts}.")
      in_cts, logs = in_cts
      if logs is not None and type(logs) is not dict:
        raise TypeError(
            f"Custom VJP bwd rule {self.bwd} was registered with "
            "defvjp_with_logs, and so the second element of the pair it "
            "returns must be None or a dict of backward-pass log entries, "
            f"but got {type(logs).__name__}.")
    if isinstance(in_cts, list):
      in_cts = tuple(in_cts)
    if not isinstance(in_cts, tuple):
      raise TypeError(f"Custom VJP bwd rule {self.bwd} must produce a tuple "
                      f"but got {type(in_cts)}.")
    in_cts = (None, None, *in_cts)  # zero cts for the consts and fwd_consts args
    if len(in_cts) != len(self.in_tree.children()) - len(self.static_argnums):
      raise ValueError(f"Custom VJP bwd rule {self.bwd} must produce a tuple "
                       "of length equal to the primal args tuple, but got "
                       f"length {len(in_cts)}")
    in_cts = broadcast_prefix(in_cts, in_avals_, is_leaf=lambda x: x is None)
    in_cts = tree_unflatten(self.in_tree, map(_replace_none, self.in_avals_flat, in_cts))
    tree_map_with_path(partial(_vjp_bwd_aval_mismatch_err, self.traced._fun_sourceinfo),
                               self.in_avals[2:], in_cts[2:])
    if self.symbolic_zeros:
      in_cts = tree_map(ad_util.replace_rule_output_symbolic_zeros, in_cts)
    return in_cts, logs

  def jvp(self, primals, tangents):
    if self.symbolic_zeros: ad.raise_custom_vjp_error_on_jvp()
    zero = lambda x: isinstance(x, ad_util.Zero)
    nzs_in = tuple(tree_map(lambda t: not isinstance(t, ad_util.Zero), t,
                            is_leaf=zero) for t in tangents)
    tangents = tree_map(ad_util.instantiate, tangents, is_leaf=zero)
    if self.opt_remat:
      fwd_traced = api.jit(partial(self.vjp_fwd, nzs_in)).trace(*primals)
      primals_out, residuals = OptRemat(self, fwd_traced)(*primals)
    else:
      primals_out, residuals, *_ = self.vjp_fwd(nzs_in, *primals)
    nzs_in_flat = [True] * len(self.in_avals_flat)
    nzs_out_flat = [True] * len(self.out_avals_flat)
    tangents_flat = tree_leaves_checked(self.in_tree, tangents)
    tangents_out_flat = fake_linear_op(self, nzs_in_flat, nzs_out_flat, residuals,
                                       None, *tangents_flat)
    tangents_out = tree_unflatten(self.out_tree, tangents_out_flat)
    return primals_out, tangents_out

  def batch_dim_rule(self, axis_data, in_dims):
    _, primal_in_tree = tracing_registry.flatten(self.drop_fwd_consts(*self.in_avals))
    in_dims_flat = primal_in_tree.flatten_up_to(self.drop_fwd_consts(*in_dims))
    _, out_dims = batching.batch_jaxpr2(self.traced.jaxpr, axis_data, tuple(in_dims_flat))
    return tree_unflatten(self.out_tree, out_dims)

  def check(self, *_):
    effs = self.traced.jaxpr.effects
    disallowed = effects.custom_derivatives_allowed_effects.filter_not_in(effs)
    if disallowed:
      raise NotImplementedError(f'Effects not supported in `custom_jvp`: {disallowed}')

  def remat(self, trace, *args):  # type: ignore
    if self.opt_remat:
      return self(*args), self
    if not trace.custom_vjp_rules:
      return self(*args), self  # see https://github.com/jax-ml/jax/pull/38914
    if not self.static_argnums:
      fwd, dyn_args = self.fwd, args
    else:
      which_static = [i in self.static_argnums for i in range(len(args))]
      dyn_args, static_args = partition_list(which_static, args)
      static_args = [x.val for x in static_args]
      fwd = lambda *dyn_args: self.fwd(*merge_lists(which_static, list(dyn_args), static_args))
    # custom_vjp_rules=False so that custom_vjp applications inside fwd hit
    # the early return above rather than recursively tracing their fwds.
    (out, _), rem_ = remat.remat_transform(trace.policy, fwd, *dyn_args,
                                           custom_vjp_rules=False)
    res = tuple(rem_.args[0])
    replay, statics = rem_.func, self.static_argnums
    def fwd2(consts, fc_res, *rest):
      fc, res = fc_res
      args_ = (consts, fc, *rest)
      return replay(res, *[x for i, x in enumerate(args_) if i not in statics])
    in_avals = (self.in_avals[0],
                (self.in_avals[1], tuple(map(typeof, res))),
                *self.in_avals[2:])
    helper = CustomVJPTraced(self.traced, fwd2, self.bwd, in_avals,
                             False, self.static_argnums, False, self.with_logs)
    return out, lambda consts, fc, *rest: helper(consts, (fc, res), *rest)


def _vjp_primal_fwd_tree_mismatch_err(self, tree):
  return (f"Custom VJP fwd rule {self.fwd.__name__} for function {self.traced.fun_name} "
          "must produce a pair (list or tuple of length two) where the first "
          "element represents the primal output "
          "(equal to the output of the custom_vjp-decorated function "
          f"{self.traced.fun_name}) and the "
          "second element represents residuals (i.e. values stored from the "
          "forward pass for use on the backward pass), but "
          f"instead the fwd rule output's first element had container/pytree "
          "structure:\n"
          f"""    {str(tree ).replace("'", "")}\n"""
          f"while the custom_vjp-decorated function {self.traced.fun_name} had output "
          "container/pytree structure:\n"
          f"""    {str(self.out_tree).replace("'", "")}.""")

def _vjp_fwd_aval_mismatch_err(path, primal_aval, fwd_val):
  if not core.typematch(ty := typeof(fwd_val), primal_aval):
    raise TypeError(f"at {keystr(path)}, got fwd output type {ty.str_short()} "
                    f"which doesn't match primal output type {primal_aval.str_short()}")

def _vjp_bwd_aval_mismatch_err(primal_sourceinfo, path, primal_aval, ct):
  if config.disable_bwd_checks.value:
    return
  if isinstance(ct, ad_util.Zero):
    return
  if isinstance(primal_aval, AbstractRef):
    primal_aval = primal_aval.inner_aval
  expected = primal_aval.to_ct_aval()
  ct_aval = ct.aval if isinstance(ct, ad_util.SymbolicZero) else typeof(ct)
  if (not core.typematch(expected, ct_aval) and
      not _temporary_dtype_exception(expected, ct_aval) and
      getattr(expected, 'dtype', None) is not dtypes.float0):
    result = f"at output{keystr(path)} " if path else ""
    raise ValueError(
        f"{result}the bwd rule attached to {primal_sourceinfo} produced an"
        f" output of type {ct_aval.str_short()} which doesn't match expected"
        f" type {expected.str_short()}")

def _replace_none(primal_in_aval, maybe_ct):
  if maybe_ct is None:
    return ad_util.Zero(primal_in_aval.to_ct_aval())
  else:
    return maybe_ct

class custom_vjp3:
  fwd: Callable | None = None
  bwd: Callable | None = None
  symz: bool = False
  opt_remat: bool = False
  with_logs: bool = False

  def __init__(self, f, nondiff_argnums=(), nondiff_argnames=()):
    self.static_argnums = _set_up_nondiff(f, nondiff_argnums, nondiff_argnames)
    update_wrapper(self, f)
    self.f = f

  def defvjp(self, fwd, bwd, *, symbolic_zeros=False, optimize_remat=False):
    self.fwd = fwd
    self.bwd = bwd
    self.symz = symbolic_zeros
    self.opt_remat = optimize_remat

  def defvjp_with_logs(self, fwd, bwd, *, symbolic_zeros=False,
                       optimize_remat=False):
    self.defvjp(fwd, bwd, symbolic_zeros=symbolic_zeros,
                optimize_remat=optimize_remat)
    self.with_logs = True

  def __call__(self, *args, **kwargs):
    if not self.fwd or not self.bwd:
      msg = f"No VJP defined for custom_vjp function {self.f.__name__} using defvjp."
      raise AttributeError(msg)

    args = resolve_kwargs(self.f, args, kwargs)
    if any(isinstance(l, core.Tracer) for i in self.static_argnums
           for l in tree_leaves(args[i])):
      raise UnexpectedTracerError("custom_vjp inputs marked with nondiff_argnums "
                                  "must be static, not Tracers")
    if core.trace_state_clean() and not any(isinstance(l, core.Tracer) for l in tree_leaves(args)):
      return self.f(*args)
    if all(is_hashable(args[i]) for i in self.static_argnums):
      traced = api.jit(self.f, static_argnums=(*self.static_argnums,)).trace(*args)
    else:
      # jit requires hashable static_argnums values, but classic custom_vjp
      # accepted unhashable nondiff_argnums values, so close over them instead
      which_static = [i in self.static_argnums for i in range(len(args))]
      dyn_args, static_args = partition_list(which_static, args)
      f = dyn_args_fun(self.f, self.static_argnums,
                       tuple(map(WrapHashably, static_args)), len(args))
      traced = api.jit(f).trace(*dyn_args)
    if any(isinstance(eff, effects.ErrorEffect) for eff in traced.effects):
      raise NotImplementedError(
          "checkify effects (e.g. checkify.check) are not supported in"
          " custom_vjp-decorated primal functions under jax_custom_vjp3. Place"
          " the check outside the custom_vjp decorator, or in the fwd/bwd"
          " rules."
      )
    args = tuple(Static(x) if i in self.static_argnums else x for i, x in enumerate(args))
    consts, traced = traced.with_consts_as_arg()
    fwd_ = update_wrapper(lambda _, __, *args: self.fwd(*args), self.fwd)
    static_argnums = frozenset(i + 2 for i in self.static_argnums)
    in_avals = tree_map(typeof, (consts, (), *args))
    prim = CustomVJPTraced(traced, fwd_, self.bwd, in_avals, self.symz,
                           static_argnums, self.opt_remat, self.with_logs)
    return prim(consts, (), *args)

  def def_vmap(self, rule, /): return self.f.def_vmap(rule)
  def def_transpose(self, rule, /): return self.f.def_transpose(rule)

class OptRemat(HiPrim):
  orig: CustomVJPTraced
  traced_fwd: Any

  def __init__(self, orig, traced_fwd):
    self.in_avals = orig.in_avals
    self.out_aval = traced_fwd.out_avals
    self.effects = traced_fwd.effects
    self.params = dict(orig=orig, traced_fwd=traced_fwd)
    super().__init__()

  def expand(self, *primals):
    return self.traced_fwd(*primals)

  def dce(self, used_outs):
    used_primals, used_res = used_outs
    if any(tree_leaves(used_res)):
      return True, (True, True), self  # if any res used, no dce at all
    elif any(tree_leaves(used_primals)):
      return True, (True, False), self.orig  # if only primals used, undo AD
    else:
      return False, (False, False), None

  # TODO(mattjj): jvp and transpose? does anyone rely on them?


def _set_up_nondiff(f, argnums_, argnames) -> frozenset[int]:
  argnums = set(argnums_)
  if argnames:
    sig = inspect.signature(f)  # needed for static_argnames
    argnums |= set(infer_argnums_and_argnames(sig, None, argnames)[0])
  return frozenset(argnums)

@register_static
@dataclass(frozen=True, slots=True)
class Static:
  val: Any


class CustomJVPTraced(HiPrim):
  traced: Any
  jvp_fun: Any  # named to avoid shadowing the jvp method via params
  symbolic_zeros: Any
  static_argnums: Any

  def __init__(self, traced, jvp_fun, in_avals, sym_zeros, static_argnums):
    self.in_avals = in_avals
    self.out_aval = traced.out_avals
    self.effects = traced.effects
    self.params = dict(traced=traced, jvp_fun=jvp_fun, symbolic_zeros=sym_zeros,
                       static_argnums=static_argnums)
    super().__init__()

  def expand(self, *args):
    args = [x for x in args if not isinstance(x, Static)]
    return self.traced(*args)

  def jvp(self, primals, tangents):
    static_args = tuple(x.val for x in primals if isinstance(x, Static))
    primals_ = tuple(x for x in primals if not isinstance(x, Static))
    tangents_ = tuple(t for x, t in zip(primals, tangents)
                      if not isinstance(x, Static))
    zero = lambda x: isinstance(x, ad_util.Zero)
    if self.symbolic_zeros:
      tangents_ = tree_map(ad_util.replace_internal_symbolic_zeros, tangents_,
                           is_leaf=zero)
    else:
      tangents_ = tree_map(ad_util.instantiate, tangents_, is_leaf=zero)
    pair_out = self.jvp_fun(*static_args, primals_, tangents_)
    jvp_name = getattr(self.jvp_fun, '__name__', str(self.jvp_fun))
    if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
      raise TypeError(
          f"Custom JVP rule {jvp_name} for function {self.traced.fun_name} "
          "must produce a pair (list or tuple of length two) representing "
          f"primal and tangent outputs, but got {pair_out}.")
    out, out_tangent = pair_out
    if (tree := tracing_registry.flatten(out)[1]) != self.out_tree:
      raise TypeError(_jvp_primal_tree_mismatch_err(self, jvp_name, out))
    _jvp_check_primal_avals(self, jvp_name, out)
    zero_ = lambda x: isinstance(x, (ad_util.Zero, ad_util.SymbolicZero))
    if (tree := tracing_registry.flatten(out_tangent, zero_)[1]) != self.out_tree:
      raise TypeError(
          f"Custom JVP rule {jvp_name} for function {self.traced.fun_name} "
          "must produce primal and tangent outputs with equal container "
          f"(pytree) structures, but got {self.out_tree} and {tree} "
          "respectively.")
    _jvp_check_tangent_avals(self, out, out_tangent)
    out_tangent = tree_map(ad_util.replace_rule_output_symbolic_zeros,
                           out_tangent, is_leaf=zero_)
    return out, out_tangent

  lin, linearized = linearize_from_jvp
  vjp_fwd, vjp_bwd_retval = vjp_from_jvp

  def transpose(self, out_ct, *args):
    # The application must be linear in the accumulated args
    args_flat = tree_leaves_checked(self.in_tree, args)
    is_lin = [isinstance(x, ad.GradAccum) for x in args_flat]
    vals = [x for x, l in zip(args_flat, is_lin) if not l]

    def lin_map(*lin_flat):
      full = merge_lists(is_lin, vals, list(lin_flat))
      return self.expand(*tree_unflatten(self.in_tree, full))

    zero = lambda x: isinstance(x, (ad_util.Zero, ad_util.SymbolicZero))
    out_ct = tree_map(ad_util.instantiate, out_ct, is_leaf=zero)
    dummies = [ad_util.zeros_like_aval(x.aval)
               for x, l in zip(args_flat, is_lin) if l]
    cts = iter(api.linear_transpose(lin_map, *dummies)(out_ct))
    for x in args_flat:
      if isinstance(x, ad.GradAccum): x.accum(next(cts))
    assert next(cts, None) is None

  def batch_dim_rule(self, axis_data, in_dims):
    in_dims_flat = self.in_tree.flatten_up_to(in_dims)
    _, out_dims = batching.batch_jaxpr2(self.traced.jaxpr, axis_data, tuple(in_dims_flat))
    return tree_unflatten(self.out_tree, out_dims)

  def check(self, *_):
    effs = self.traced.jaxpr.effects
    disallowed = effects.custom_derivatives_allowed_effects.filter_not_in(effs)
    if disallowed:
      raise NotImplementedError(f'Effects not supported in `custom_jvp`: {disallowed}')

def _jvp_primal_tree_mismatch_err(self, jvp_name, out):
  flat, tree = tracing_registry.flatten(out)
  ty_tree = tree_unflatten(tree, [typeof(x).str_short() for x in flat])
  ty_tree_ = tree_unflatten(self.out_tree,
                            [a.str_short() for a in self.out_avals_flat])
  return (f"Custom JVP rule {jvp_name} for function {self.traced.fun_name} "
          "must produce a pair (list or tuple of length two) "
          "where the first element represents the primal output "
          "(equal in value to the output of the custom_jvp-decorated function "
          f"{self.traced.fun_name}, "
          "and in particular of the same container/pytree structure), but "
          "instead the JVP rule output's first element had container/pytree "
          "structure:\n"
          f"""    {str(ty_tree ).replace("'", "")}\n"""
          f"while the custom_jvp-decorated function {self.traced.fun_name} "
          "had output container/pytree structure:\n"
          f"""    {str(ty_tree_).replace("'", "")}.""")

def _jvp_check_primal_avals(self, jvp_name, out):
  out_flat = tree_leaves_checked(self.out_tree, out)
  avals = [typeof(x) for x in out_flat]
  if not all(map(core.typematch, avals, self.out_avals_flat)):
    ty_tree = tree_unflatten(self.out_tree, [a.str_short() for a in avals])
    ty_tree_ = tree_unflatten(self.out_tree,
                              [a.str_short() for a in self.out_avals_flat])
    raise TypeError(
        f"Custom JVP rule {jvp_name} for function {self.traced.fun_name} "
        "must produce a pair (list or tuple of length two) "
        "where the first element represents the primal output "
        "(equal in value to the output of the custom_jvp-decorated function "
        f"{self.traced.fun_name}, "
        "and in particular with leaves of the same shape/dtype), but "
        "instead the JVP rule output's first element had shapes/dtypes of:\n"
        f"""    {str(ty_tree ).replace("'", "")}\n"""
        f"while the custom_jvp-decorated function {self.traced.fun_name} "
        "had output shapes/dtypes of:\n"
        f"""    {str(ty_tree_).replace("'", "")}""")

def _jvp_check_tangent_avals(self, out, out_tangent):
  strip = lambda a: a.strip_weak_type() if hasattr(a, 'strip_weak_type') else a
  out_flat = tree_leaves_checked(self.out_tree, out)
  tangents_flat = self.out_tree.flatten_up_to(out_tangent)
  primal_avals_out = [strip(typeof(x)) for x in out_flat]
  expected_tangent_avals_out = [a.to_tangent_aval() for a in primal_avals_out]
  tangent_avals_out = [
      strip(t.aval) if isinstance(t, (ad_util.Zero, ad_util.SymbolicZero))
      else strip(typeof(t)) for t in tangents_flat]
  if not all(map(core.typematch, expected_tangent_avals_out, tangent_avals_out)):
    if len(expected_tangent_avals_out) == 1:
      (av_p,), (av_et,), (av_t,) = (primal_avals_out,
                                    expected_tangent_avals_out,
                                    tangent_avals_out)
      msg = ("Custom JVP rule must produce primal and tangent outputs with "
             "corresponding shapes and dtypes. "
             "Expected {} (tangent type of {}) but got {}.")
      raise TypeError(msg.format(av_et.str_short(), av_p.str_short(),
                                 av_t.str_short()))
    else:
      disagreements = "\n".join(
          f"  primal {av_p.str_short()} with tangent {av_t.str_short()}, "
          f"expecting tangent {av_et}"
          for av_p, av_et, av_t in zip(primal_avals_out,
                                       expected_tangent_avals_out,
                                       tangent_avals_out)
          if not core.typematch(av_et, av_t))
      raise TypeError(
          "Custom JVP rule must produce primal and tangent outputs with "
          f"corresponding shapes and dtypes, but got:\n{disagreements}")


class custom_jvp3:
  jvp_fun: Callable | None = None
  symz: bool = False

  def __init__(self, f, nondiff_argnums=(), nondiff_argnames=()):
    self.static_argnums = _set_up_nondiff(f, nondiff_argnums, nondiff_argnames)
    update_wrapper(self, f)
    self.f = f

  def defjvp(self, jvp, symbolic_zeros=False):
    self.jvp_fun = jvp
    self.symz = symbolic_zeros
    return jvp

  def defjvps(self, *jvps):
    if self.static_argnums:
      raise TypeError("Can't use ``defjvps`` with ``nondiff_argnums``.")
    def jvp(primals, tangents):
      primal_out = self(*primals)
      zeros = tree_map(ad_util.p2tz, primal_out)
      all_tangents_out = [j(t, primal_out, *primals) if j else zeros
                          for t, j in zip(tangents, jvps)]
      sum_tangents = lambda _, x, *xs: reduce(ad.add_tangents, xs, x)
      tangent_out = tree_map(sum_tangents, primal_out, *all_tangents_out)
      return primal_out, tangent_out
    self.defjvp(jvp)

  def __call__(self, *args, **kwargs):
    if not self.jvp_fun:
      msg = (f"No JVP defined for custom_jvp function {self.f.__name__} "
             "using defjvp.")
      raise AttributeError(msg)

    try:
      args = resolve_kwargs(self.f, args, kwargs)
    except TypeError as e:
      raise TypeError(
          "The input arguments to the custom_jvp-decorated function "
          f"{self.f.__name__} could not be resolved to positional-only "
          f"arguments. Binding failed with the error:\n{e}") from e
    if any(isinstance(args[i], core.Tracer) for i in self.static_argnums):
      raise UnexpectedTracerError("custom_jvp inputs marked with nondiff_argnums "
                                  "must be static, not Tracers")
    if all(is_hashable(args[i]) for i in self.static_argnums):
      traced = api.jit(self.f, static_argnums=(*self.static_argnums,)).trace(*args)
    else:
      # jit requires hashable static_argnums values, but classic custom_jvp
      # accepted unhashable nondiff_argnums values, so close over them instead
      which_static = [i in self.static_argnums for i in range(len(args))]
      dyn_args, static_args = partition_list(which_static, args)
      f = dyn_args_fun(self.f, self.static_argnums,
                       tuple(map(WrapHashably, static_args)), len(args))
      traced = api.jit(f).trace(*dyn_args)
    if any(isinstance(x, core.Tracer) for x in traced._consts):
      t = next(x for x in traced._consts if isinstance(x, core.Tracer))
      raise UnexpectedTracerError(
          f"custom_jvp-decorated function {self.f} closed over a {type(t).__name__} "
          f"of type {t.aval.str_short()}, but custom_jvp functions can't close "
          f"over Tracers. Rewrite {self.f} to take it as an explicit input.")
    args = tuple(Static(x) if i in self.static_argnums else x for i, x in enumerate(args))
    in_avals = tree_map(typeof, args)
    prim = CustomJVPTraced(traced, self.jvp_fun, in_avals, self.symz,
                           self.static_argnums)
    return prim(*args)


class MappingSpec: pass
class HiPspec:
  def to_lo(self) -> tuple[PartitionSpec, ...]:
    _must_override(self, "to_lo", "shard_map")
  def to_tangent_spec(self) -> HiPspec:
    _must_override(self, "to_tangent_spec", "autodiff through shard_map")
  def to_ct_spec(self) -> HiPspec:
    _must_override(self, "to_ct_spec", "autodiff through shard_map")
