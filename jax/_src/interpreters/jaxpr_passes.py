# Copyright 2024 The JAX Authors.
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
from collections.abc import Callable, Sequence
import dataclasses
import functools
from functools import partial
import numpy as np
from typing import Any

from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src import linear_util as lu

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

resolve_edtypes_rules = {}

def register_edtype_rule(primitive, rule, always_invoke=False):
  def _edtype_rule(ctx, *args, **params):
    has_extended_input = any(is_extended(aval) for aval in ctx.avals_in)
    has_extended_output = any(is_extended(aval) for aval in ctx.avals_out)
    if always_invoke or has_extended_input or has_extended_output:
      return rule(ctx, *args, **params)
    else:
      return default_edtypes_rule(primitive, ctx, *args, **params)
  resolve_edtypes_rules[primitive] = _edtype_rule

def default_edtypes_rule(primitive, ctx, *args, **params):
  del ctx
  subfuns, bind_params = primitive.get_bind_params(params)
  return primitive.bind(*subfuns, *args, **bind_params)


def resolve_edtypes_fun(fun: Callable, multiple_results: bool = True) -> Callable:
  """Converts a traceable JAX function `fun` into a translation rule."""
  def _rule(ctx: ResolveEdtypesContext, *args, **params):
    f = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)
    if config.dynamic_shapes.value:
      # We might be applying this function to arguments with dynamic shapes,
      # i.e. there might be Vars in the shape tuples of ctx.avals_in. In that
      # case, we need to form a jaxpr with leading binders for those axis size
      # arguments (by computing an InputType and using trace_to_jaxpr_dynamic2),
      # and we need to call jaxpr_subcomp with these arguments made explicit.
      assert ctx.axis_size_env is not None
      args = (*ctx.axis_size_env.values(), *args)
      idx = {d: core.DBIdx(i) for i, d in enumerate(ctx.axis_size_env)}
      i32_aval = core.ShapedArray((), np.dtype('int32'))
      implicit_args = [(i32_aval, False)] * len(ctx.axis_size_env)
      explicit_args = [(a.update(shape=tuple(idx.get(d, d) for d in a.shape))
                        if type(a) is core.DShapedArray else a, True)
                      for a in ctx.avals_in]
      wrapped_fun = lu.annotate(wrapped_fun, (*implicit_args, *explicit_args))
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic2(wrapped_fun)
    else:
      jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    phys_jaxpr = resolve_edtypes_jaxpr(core.ClosedJaxpr(jaxpr, consts))
    result = core.eval_jaxpr(phys_jaxpr.jaxpr, phys_jaxpr.consts, *args)
    if multiple_results:
      return result
    else:
      return result[0]
  return _rule


@util.weakref_lru_cache
def resolve_edtypes_jaxpr(jaxpr: core.ClosedJaxpr) -> core.ClosedJaxpr:
  """Replaces all extended dtypes with physical types in a jaxpr."""
  def _to_physical_const(const):
    if is_extended(const):
      return const.dtype._rules.physical_const(const)
    return const
  physical_consts = map(_to_physical_const, jaxpr.consts)
  fun = partial(resolve_edtypes_interp, jaxpr.jaxpr, physical_consts)
  physical_in_avals = [core.physical_aval(aval) for aval in jaxpr.in_avals]
  wrapped_fn = lu.wrap_init(fun)
  new_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      wrapped_fn, physical_in_avals)
  assert len(new_jaxpr.constvars) == len(consts)
  return core.ClosedJaxpr(new_jaxpr, consts)


@dataclasses.dataclass(frozen=True)
class ResolveEdtypesContext:
  avals_in: Sequence[Any]
  avals_out: Sequence[Any]

def is_extended(aval: Any):
  if isinstance(aval, core.AbstractToken):
    return False
  if hasattr(aval, 'dtype'):
    return dtypes.issubdtype(aval.dtype, dtypes.extended)
  return False

def resolve_edtypes_interp(jaxpr: core.Jaxpr,
                           physical_consts: Sequence[core.Value],
                           *physical_in_args: core.Value) -> list[Any]:
  env: dict[core.Var, Any] = {}

  last_used = core.last_used(jaxpr)

  def read_env(var: core.Atom):
    if isinstance(var, core.Literal):
      return var.val
    return env[var]

  def write_env(var: core.Var, val: Any):
    env[var] = val

  map(write_env, jaxpr.constvars, physical_consts)
  assert len(jaxpr.invars) == len(physical_in_args), (
      f"Length mismatch: {jaxpr.invars} != {physical_in_args}")
  map(write_env, jaxpr.invars, physical_in_args)

  for eqn in jaxpr.eqns:
    physical_invals = map(read_env, eqn.invars)
    edtype_rule = resolve_edtypes_rules.get(
        eqn.primitive, functools.partial(default_edtypes_rule, eqn.primitive)
    )
    name_stack = (
        source_info_util.current_name_stack() + eqn.source_info.name_stack
    )
    with source_info_util.user_context(
        eqn.source_info.traceback, name_stack=name_stack
    ), eqn.ctx.manager:
      ctx = ResolveEdtypesContext(
        avals_in = tuple(x.aval for x in eqn.invars),
        avals_out = tuple(x.aval for x in eqn.outvars),
      )
      physical_outvals = edtype_rule(
          ctx, *physical_invals, **eqn.params
      )

    if eqn.primitive.multiple_results:
      assert len(physical_outvals) == len(eqn.outvars)
      map(write_env, eqn.outvars, physical_outvals)
    else:
      write_env(eqn.outvars[0], physical_outvals)
    core.clean_up_dead_vars(eqn, env, last_used)

  return map(read_env, jaxpr.outvars)
