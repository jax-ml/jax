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

"""Custom fusible dtypes."""

import abc
import dataclasses
import functools
import itertools as it
from typing import Any, TypeVar
from collections.abc import Sequence

import jax
from jax._src import api_util
from jax._src import core
from jax._src import custom_derivatives
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow import conditionals
from jax._src.pallas import core as pallas_core
from jax._src.pallas import pallas_call
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.fuser import block_spec
from jax._src.pallas.fuser.fusible import fusible_p
from jax._src.state import discharge as state_discharge
from jax._src.state import primitives as state_primitives
from jax._src.util import foreach

# TODO(sharadmv): Enable type checking.
# mypy: ignore-errors

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

T = TypeVar("T")

_physicalize_rules = {}

pack_dtype_p = core.Primitive("pack_dtype")


@pack_dtype_p.def_abstract_eval
def pack_dtype_abstract_eval(*xs, dtype):
  if dtypes.issubdtype(dtype, FusibleElementDType):
    return dtype.abstract_pack(*xs)
  raise ValueError("Attempted to pack non-fusion dtype: {dtype}")


def pack(*xs, dtype):
  return pack_dtype_p.bind(*xs, dtype=dtype)


unpack_dtype_p = core.Primitive("unpack_dtype")
unpack_dtype_p.multiple_results = True


@unpack_dtype_p.def_abstract_eval
def unpack_dtype_abstract_eval(x):
  if dtypes.issubdtype(x.dtype, FusibleElementDType):
    return x.dtype.abstract_unpack(x)
  elif isinstance(x.dtype, state.AbstractRef):
    raise NotImplementedError()
  raise ValueError("Attempted to unpack non-fusion dtype: {dtype}")


def unpack(x):
  return unpack_dtype_p.bind(x)


class FusibleElementDType(dtypes.extended):
  """Scalar dtype for fusible dtypes."""


class FusibleTyRules:
  allow_conversion: bool = False


class FusionDType(dtypes.ExtendedDType, metaclass=abc.ABCMeta):
  """Base class for fusible extended dtypes."""

  _op_registry = {}
  _rules = FusibleTyRules
  type = FusibleElementDType

  @abc.abstractmethod
  def abstract_unpack(self, x) -> Sequence[Any]:
    raise NotImplementedError()

  @abc.abstractmethod
  def abstract_pack(self, *xs):
    raise NotImplementedError()

  @classmethod
  def register_op(cls, primitive):
    def _register_fn(fn):
      cls._op_registry[primitive] = fn

    return _register_fn

  @classmethod
  def get_op_rule(cls, primitive):
    return cls._op_registry.get(primitive)

  @property
  def name(self):
    return str(self)

  @abc.abstractmethod
  def pull_block_spec_one_step(self, aval_out, *args, **kwargs):
    raise NotImplementedError()


def physicalize(f):
  """Runs a function that contains fusible extended dtypes."""

  def wrapper(*args, **kwargs):
    if kwargs:
      raise NotImplementedError()
    flattened_args, treedef = jax.tree.flatten(args)
    debug_info = api_util.debug_info("physicalize", f, args, kwargs)
    wrapped_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
        lu.wrap_init(f, debug_info=debug_info), treedef
    )
    avals = [core.ShapedArray(a.shape, a.dtype) for a in flattened_args]
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals)
    new_jaxpr = physicalize_closed_jaxpr(core.ClosedJaxpr(jaxpr, consts))
    out_flat = core.eval_jaxpr(
        new_jaxpr.jaxpr, new_jaxpr.consts, *flattened_args
    )
    return tree_util.tree_unflatten(out_tree_thunk(), out_flat)

  return wrapper


@util.weakref_lru_cache
def physicalize_closed_jaxpr(jaxpr: core.ClosedJaxpr) -> core.ClosedJaxpr:
  """Replaces all extended dtypes with physical types in a jaxpr."""
  fun = functools.partial(physicalize_interp, jaxpr.jaxpr, jaxpr.consts)
  in_avals = [_physical_aval(aval) for aval in jaxpr.in_avals]
  flat_avals, treedef = tree_util.tree_flatten(in_avals)
  debug_info = api_util.debug_info("physicalize_closed_jaxpr", fun, (), {})
  wrapped_fun, _ = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun, debug_info=debug_info), treedef
  )
  new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, flat_avals)
  assert len(new_jaxpr.constvars) == len(consts), "Mismatched consts"
  return core.ClosedJaxpr(new_jaxpr, consts)


def _physical_aval(aval):
  if isinstance(aval, core.ShapedArray):
    if isinstance(aval.dtype, FusionDType):
      return aval.dtype.abstract_unpack(aval)
    return core.ShapedArray(aval.shape, aval.dtype)
  if isinstance(aval, state.AbstractRef):
    if isinstance(aval.dtype, FusionDType):
      unpacked = aval.dtype.abstract_unpack(aval.inner_aval)
      return tuple(aval.update(inner_aval=u) for u in unpacked)
    return aval
  return aval


def physicalize_jaxpr(jaxpr: core.Jaxpr) -> core.Jaxpr:
  """Replaces all extended dtypes with physical types in a jaxpr."""

  def _flat_jaxpr_eval(consts, args):
    return physicalize_interp(jaxpr, consts, *args)

  in_avals = [_physical_aval(v.aval) for v in jaxpr.invars]
  const_avals = [_physical_aval(v.aval) for v in jaxpr.constvars]
  flat_avals, treedef = jax.tree.flatten((const_avals, in_avals))
  debug_info = api_util.debug_info(
      "physicalize_jaxpr", _flat_jaxpr_eval, (), {}
  )
  wrapped_fun, _ = api_util.flatten_fun_nokwargs(
      lu.wrap_init(_flat_jaxpr_eval, debug_info=debug_info), treedef
  )
  new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, flat_avals)
  assert not consts
  new_jaxpr = pe.convert_invars_to_constvars(
      new_jaxpr, len(tree_util.tree_leaves(const_avals))
  )
  return new_jaxpr


@dataclasses.dataclass
class Context:
  avals_in: Sequence[Any]
  avals_out: Sequence[Any]


def physicalize_interp(
    jaxpr: core.Jaxpr, consts: Sequence[core.Value], *args: core.Value
):
  """Physicalizes a jaxpr by replacing fusible dtypes with physical types."""
  # TODO: Merge into JAX core.
  env: dict[core.Var, Any] = {}

  def read_env(var: core.Atom):
    if isinstance(var, core.Literal):
      return var.val
    return env[var]

  def write_env(var: core.Var, val: Any):
    env[var] = val

  foreach(write_env, jaxpr.constvars, consts)
  assert len(jaxpr.invars) == len(
      args
  ), f"Length mismatch: {jaxpr.invars} != {args}"
  foreach(write_env, jaxpr.invars, args)

  for eqn in jaxpr.eqns:
    invals = list(map(read_env, eqn.invars))
    avals_in = tuple(x.aval for x in eqn.invars)
    name_stack = (
        source_info_util.current_name_stack() + eqn.source_info.name_stack
    )
    with (
        source_info_util.user_context(
            eqn.source_info.traceback, name_stack=name_stack
        ),
        eqn.ctx.manager,
    ):
      # need to check types and then invoke the correct rule.
      in_types = [aval.dtype for aval in avals_in]  # pytype: disable=attribute-error
      ctx = Context(
          avals_in=avals_in, avals_out=[var.aval for var in eqn.outvars]
      )
      custom_rule = _phys_find_rule(eqn.primitive, in_types)
      if custom_rule:
        outvals = custom_rule(ctx, *invals, **eqn.params)
      else:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)

    if eqn.primitive.multiple_results:
      assert len(outvals) == len(eqn.outvars), eqn
      foreach(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)

  return map(read_env, jaxpr.outvars)


def _phys_find_rule(primitive, types: Sequence[dtypes.DType]):
  """Finds the physicalization rule for a primitive."""
  if primitive in _physicalize_rules:
    return _physicalize_rules[primitive]
  fusion_types = {type_ for type_ in types if isinstance(type_, FusionDType)}
  if len(fusion_types) == 0:
    return None
  elif len(fusion_types) > 1:
    raise ValueError(f"Multiple fusion types for primitive: {fusion_types}")
  fusion_type = fusion_types.pop()
  if primitive not in fusion_type._op_registry:
    raise ValueError(
        f"No implementation found for primitive {primitive} "
        f"for custom type {fusion_type}"
    )
  return fusion_type.get_op_rule(primitive)


def _assert_no_fusion_types(avals: Sequence[core.AbstractValue]):
  for aval in avals:
    if isinstance(aval, (core.ShapedArray, state.AbstractRef)):
      if isinstance(aval.dtype, FusionDType):
        raise NotImplementedError(f"Fusion type found in avals: {avals}")


def _pallas_call_physicalize_rule(
    ctx: Context, *args, jaxpr, grid_mapping: pallas_core.GridMapping, **kwargs
):
  _assert_no_fusion_types(ctx.avals_in)
  _assert_no_fusion_types(ctx.avals_out)
  with grid_mapping.trace_env():
    new_jaxpr = physicalize_closed_jaxpr(core.ClosedJaxpr(jaxpr, ()))
  num_new_vals = len(new_jaxpr.jaxpr.invars) - len(jaxpr.invars)
  grid_mapping = grid_mapping.replace(
      num_scratch_operands=grid_mapping.num_scratch_operands + num_new_vals
  )
  return pallas_call.pallas_call_p.bind(
      *args, jaxpr=new_jaxpr.jaxpr, grid_mapping=grid_mapping, **kwargs
  )


_physicalize_rules[pallas_call.pallas_call_p] = _pallas_call_physicalize_rule


def _cond_physicalize_rule(ctx: Context, *args, branches, **kwargs):
  _assert_no_fusion_types(ctx.avals_out)
  physicalized_branches = tuple(
      physicalize_closed_jaxpr(branch) for branch in branches
  )
  flat_args = jax.tree.leaves(args)
  return conditionals.cond_p.bind(
      *flat_args, branches=physicalized_branches, **kwargs
  )


_physicalize_rules[conditionals.cond_p] = _cond_physicalize_rule


@lu.transformation2
def _physicalize_transform(f, *args):
  vals, zeros = args[::2], args[1::2]
  assert len(vals) == len(zeros)
  wrapper = lambda *inner_vals: f(
      *it.chain.from_iterable(zip(inner_vals, zeros))
  )
  return physicalize(wrapper)(*vals)


@lu.transformation2
def _physicalize_transform_bwd(f, const_avals, *args):
  return [custom_derivatives.Zero(a) for a in const_avals] + list(
      physicalize(f)(*args)
  )


def _custom_vjp_call_physicalize_rule(
    ctx: Context, *args, call_jaxpr, num_consts, fwd_jaxpr_thunk, bwd, **kwargs
):
  _assert_no_fusion_types(ctx.avals_out)
  new_jaxpr = physicalize_closed_jaxpr(call_jaxpr)
  fun = lu.wrap_init(core.jaxpr_as_fun(new_jaxpr),
                     debug_info=call_jaxpr.jaxpr.debug_info)
  fwd = custom_derivatives.lift_fwd(num_consts, fwd_jaxpr_thunk)
  fwd_physicalized = _physicalize_transform(fwd)
  const_avals, _ = util.split_list(new_jaxpr.in_avals, [num_consts])
  bwd_physicalized = _physicalize_transform_bwd(bwd, const_avals)
  return custom_derivatives.custom_vjp_call_p.bind(
      fun, fwd_physicalized, bwd_physicalized, *args, **kwargs
  )

_physicalize_rules[custom_derivatives.custom_vjp_call_p] = _custom_vjp_call_physicalize_rule


def _run_state_rule(ctx: Context, *args, jaxpr, which_linear, is_initialized):
  _assert_no_fusion_types(ctx.avals_in)
  _assert_no_fusion_types(ctx.avals_out)
  jaxpr = physicalize_jaxpr(jaxpr)
  return state_discharge.run_state_p.bind(
      *args,
      jaxpr=jaxpr,
      which_linear=which_linear,
      is_initialized=is_initialized,
  )


_physicalize_rules[state_discharge.run_state_p] = _run_state_rule


def _core_map_rule(ctx: Context, *args, jaxpr, **params):
  _assert_no_fusion_types(ctx.avals_in)
  _assert_no_fusion_types(ctx.avals_out)
  assert not jaxpr.invars
  with core.extend_axis_env_nd(params["mesh"].shape.items()):
    jaxpr = physicalize_jaxpr(jaxpr)
  return pallas_core.core_map_p.bind(*args, jaxpr=jaxpr, **params)


_physicalize_rules[pallas_core.core_map_p] = _core_map_rule


def _run_scoped_rule(ctx: Context, *args, jaxpr, **params):
  _assert_no_fusion_types(ctx.avals_out)
  jaxpr = physicalize_jaxpr(jaxpr)
  flat_args = tree_util.tree_leaves(args)
  assert len(flat_args) == len(
      jaxpr.constvars
  ), f"Length mismatch: {len(flat_args)=} != {len(jaxpr.constvars)=}"
  return pallas_primitives.run_scoped_p.bind(*flat_args, jaxpr=jaxpr, **params)


_physicalize_rules[pallas_primitives.run_scoped_p] = _run_scoped_rule


def _scan_rule(ctx: Context, *args, jaxpr, **params):
  _assert_no_fusion_types(ctx.avals_in)
  _assert_no_fusion_types(ctx.avals_out)
  jaxpr = physicalize_closed_jaxpr(jaxpr)
  return jax.lax.scan_p.bind(*args, jaxpr=jaxpr, **params)


_physicalize_rules[jax.lax.scan_p] = _scan_rule


def _while_rule(
    ctx: Context, *args, body_jaxpr, cond_jaxpr, body_nconsts,
    cond_nconsts, **params
):
  _assert_no_fusion_types(ctx.avals_out)
  cond_avals = [v.aval for v in cond_jaxpr.jaxpr.invars]
  _, cond_in_avals = util.split_list(cond_avals, [cond_nconsts])
  _assert_no_fusion_types(cond_in_avals)
  new_cond_jaxpr = physicalize_closed_jaxpr(cond_jaxpr)
  new_num_cond_consts = (
      cond_nconsts
      + len(new_cond_jaxpr.jaxpr.invars)
      - len(cond_jaxpr.jaxpr.invars)
  )

  body_avals = [v.aval for v in body_jaxpr.jaxpr.invars]
  _, body_in_avals = util.split_list(body_avals, [body_nconsts])
  _assert_no_fusion_types(body_in_avals)
  new_body_jaxpr = physicalize_closed_jaxpr(body_jaxpr)
  new_num_body_consts = (
      body_nconsts
      + len(new_body_jaxpr.jaxpr.invars)
      - len(body_jaxpr.jaxpr.invars)
  )
  flat_args = tree_util.tree_leaves(args)
  cond_consts, body_consts, flat_args = \
        util.split_list(flat_args, [cond_nconsts, body_nconsts])
  assert len(flat_args) + len(body_consts) == len(
      new_body_jaxpr.jaxpr.invars), (
      f"Length mismatch: {len(flat_args) + len(body_consts)} !="
      f" {len(new_body_jaxpr.jaxpr.invars)=}"
  )
  assert len(flat_args) + len(cond_consts) == len(
      new_cond_jaxpr.jaxpr.invars), (
      f"Length mismatch: {len(flat_args) + len(cond_consts)} !="
      f" {len(new_cond_jaxpr.jaxpr.invars)=}"
  )
  return jax.lax.while_p.bind(
      *(cond_consts + body_consts + flat_args),
      body_jaxpr=new_body_jaxpr,
      cond_jaxpr=new_cond_jaxpr,
      body_nconsts=new_num_body_consts,
      cond_nconsts=new_num_cond_consts,
      **params,
  )


_physicalize_rules[jax.lax.while_p] = _while_rule


def _pack_rule(_, *args, dtype):
  del dtype
  return args


_physicalize_rules[pack_dtype_p] = _pack_rule


def _unpack_rule(_, arg):
  return arg


_physicalize_rules[unpack_dtype_p] = _unpack_rule


def _swap_rule(ctx: Context, ref, val, *args, tree):
  ref_aval, *_ = ctx.avals_in
  if not isinstance(ref_aval.dtype, FusionDType):
    return state_primitives.swap_p.bind(ref, val, *args, tree=tree)
  return ref_aval.dtype.swap(ref, val, *args, tree=tree)


_physicalize_rules[state_primitives.swap_p] = _swap_rule


def _get_rule(ctx: Context, ref, *args, tree):
  ref_aval, *_ = ctx.avals_in
  if not isinstance(ref_aval.dtype, FusionDType):
    return state_primitives.get_p.bind(ref, *args, tree=tree)
  return ref_aval.dtype.get(ref, *args, tree=tree)


_physicalize_rules[state_primitives.get_p] = _get_rule


@block_spec.register_eval_rule(pack_dtype_p)
def _pack_dtype_eval_rule(eval_ctx: block_spec.KernelEvalContext, *args, dtype):
  return dtype.eval_rule(eval_ctx, *args)


@block_spec.register_pull_block_spec_rule(pack_dtype_p)
def _pack_dtype_pull_rule(
    ctx: block_spec.PullRuleContext,
    block_spec: pallas_core.BlockSpec,
    *,
    dtype: FusionDType,
):
  aval_out = ctx.avals_out[0]
  return dtype.pull_block_spec_one_step(aval_out, block_spec)  # pytype: disable=attribute-error


def _fusible_physicalize_rule(
    _, *consts_and_args, jaxpr, num_consts, in_tree, out_tree, func
):
  consts, _ = util.split_list(consts_and_args, [num_consts])
  new_jaxpr = physicalize_closed_jaxpr(core.ClosedJaxpr(jaxpr, consts))
  return fusible_p.bind(
      *consts_and_args,
      jaxpr=new_jaxpr.jaxpr,
      num_consts=num_consts,
      in_tree=in_tree,
      out_tree=out_tree,
      func=func,
  )


_physicalize_rules[fusible_p] = _fusible_physicalize_rule
