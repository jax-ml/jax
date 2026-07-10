# Copyright 2026 The JAX Authors.
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

from functools import partial
from collections.abc import Callable

from jax._src import core
from jax._src import api_util
from jax._src.util import (safe_map, safe_zip, unzip2, weakref_lru_cache,
                           partition_list)
from jax._src.interpreters import partial_eval as pe
from jax._src import flattree as ft
from jax._src.tree_util import (
    Partial, tree_unflatten, tree_leaves_checked)
from jax._src import source_info_util
from jax._src.core import typeof

map = safe_map
zip = safe_zip

# TODO
#  [ ] static_argnums and static_argnames (via FlatTree)
#  [ ] allow NotAvailable sentinels
#  [ ] primal-output-to-residual forwarding

def remat_transform(policy, f, *args, custom_vjp_rules):
  dbg = api_util.debug_info("remat", f, args, {})
  with core.take_current_trace() as parent_trace:
    jaxpr_trace = pe.DynamicJaxprTrace(None)
    jaxpr_trace.tag = core.TraceTag()
    trace = RematTrace(parent_trace, jaxpr_trace, policy, custom_vjp_rules)
    args_ft, _ = ft.flatten_args(*args).unpack()
    in_tracers = args_ft.map(
        lambda x: RematTracer(trace, x, jaxpr_trace.new_arg(typeof(x), None)))  # type: ignore # noqa F821
    with core.set_current_trace(trace):
      args = in_tracers.unflatten()
      ans_pytree = f(*args)
      dbg = dbg.set_result_paths(ans_pytree)
      ans_ft = ft.flatten(ans_pytree)
      del ans_pytree, args
    out_ft, out_tracer_ft = ans_ft.map(trace.to_val_tracer_pair).unzip2()
    src = source_info_util.current()
    out_tracer_ft = out_tracer_ft.map(partial(jaxpr_trace.to_jaxpr_tracer, source_info=src))
    jaxpr = jaxpr_trace.to_jaxpr(list(out_tracer_ft), dbg, src)
    jaxpr, res = jaxpr.separate_consts()
    in_tree, out_tree = args_ft.tree, out_ft.tree
    del trace, in_tracers, out_tracer_ft
  def f_rem(res, *args):
    args_flat = tree_leaves_checked(in_tree, args)
    out_flat = core.eval_jaxpr(jaxpr, (), *res, *args_flat)
    return tree_unflatten(out_tree, out_flat)
  return out_ft.unflatten(), Partial(f_rem, map(reduce_precision, res))

class RematTracer(core.Tracer['RematTrace']):
  _trace: RematTrace

  def __init__(self, trace, x, jaxpr_tracer):
    super().__init__(trace, core.typeof(x))
    self.val = x
    self.tracer = jaxpr_tracer

class RematTrace(core.Trace):
  def __init__(self, parent_trace, jaxpr_trace, policy, custom_vjp_rules):
    super().__init__()
    self.parent_trace = parent_trace
    self.jaxpr_trace = jaxpr_trace
    self.policy = policy
    self.requires_low = False
    # flag to handle checkpoint_name calls in custom_vjp fwd w/o inf recursion
    self.custom_vjp_rules = custom_vjp_rules

  tag = property(lambda self: self.jaxpr_trace.tag)

  def to_val_tracer_pair(self, x):
    if isinstance(x, RematTracer) and x._trace.tag is self.tag:
      return x.val, x.tracer
    else:
      return x, x

  def stage_value(self, val):
    new_val = self.parent_trace.stage_value(val)
    new_tracer = self.jaxpr_trace.stage_value(val)
    return RematTracer(self, new_val, new_tracer)

  def process_primitive(self, prim, tracers, params, /):
    in_vals, in_vals2 = unzip2(map(self.to_val_tracer_pair, tracers))
    if prim in rules:
      with core.set_current_trace(self.parent_trace):
        out_primal, rem = rules[prim](self, *in_vals, **params)
      with core.set_current_trace(self.jaxpr_trace):
        out_primal2 = rem(*in_vals2)
    else:  # default: full remat
      with core.set_current_trace(self.parent_trace):
        out_primal = prim.bind(*in_vals, **params)
      with core.set_current_trace(self.jaxpr_trace):
        out_primal2 = prim.bind(*in_vals2, **params)
    if prim.multiple_results:
      return map(partial(RematTracer, self), out_primal, out_primal2)
    else:
      return RematTracer(self, out_primal, out_primal2)

  def process_call(self, call_primitive, f, tracers, params):
    in_vals, in_vals2 = unzip2(map(self.to_val_tracer_pair, tracers))
    raise NotImplementedError  # TODO remat_subtrace...

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, /, *, symbolic_zeros):
    in_vals, in_vals2 = unzip2(map(self.to_val_tracer_pair, tracers))
    with core.set_current_trace(self.parent_trace):
      out_primal = prim.bind(*in_vals, subfuns=(fun, jvp),
                             symbolic_zeros=symbolic_zeros)
    with core.set_current_trace(self.jaxpr_trace):
      out_primal2 = prim.bind(*in_vals2, subfuns=(fun, jvp),
                              symbolic_zeros=symbolic_zeros)
    return map(partial(RematTracer, self), out_primal, out_primal2)

  def process_custom_vjp_call(self, prim, f, fwd, bwd, tracers, /, *, out_trees, symbolic_zeros):
    in_vals, in_vals2 = unzip2(map(self.to_val_tracer_pair, tracers))
    with core.set_current_trace(self.parent_trace):
      out_primal = prim.bind(*in_vals, subfuns=(f, fwd, bwd),
                             out_trees=out_trees, symbolic_zeros=symbolic_zeros)
    with core.set_current_trace(self.jaxpr_trace):
      out_primal2 = prim.bind(*in_vals2, subfuns=(f, fwd, bwd),
                              out_trees=out_trees, symbolic_zeros=symbolic_zeros)
    return map(partial(RematTracer, self), out_primal, out_primal2)

def remat_subtrace(f: Callable, tag: core.TraceTag, policy,
                   debug_info: core.DebugInfo, args, custom_vjp_rules):
  source_info = source_info_util.current()
  with core.take_current_trace() as parent_trace:
    rem_trace = pe.DynamicJaxprTrace(debug_info, auto_dce=True)
    rem_trace.tag = tag
    trace = RematTrace(parent_trace, rem_trace, policy, custom_vjp_rules)
    tracers = [RematTracer(trace, x, rem_trace.new_arg(typeof(x), source_info))
               for x in args]
    with core.set_current_trace(trace, check_leaks=True):
      ans = f(*tracers)
      out_primals, out_rem = ans.map(trace.to_val_tracer_pair).unzip2()
      del trace, ans, tracers
  out_rem = map(partial(rem_trace.to_jaxpr_tracer, source_info=source_info), out_rem)
  jaxpr, consts = rem_trace.to_jaxpr(out_rem, debug_info.with_unknown_names(),
                                     source_info)
  which_env = [(isinstance(c, pe.DynamicJaxprTracer) and
                getattr(c._trace, 'tag', None) is tag) for c in consts]
  jaxpr = pe.move_envvars(jaxpr, tuple(which_env))
  res, env = partition_list(which_env, consts)
  residual_avals = map(typeof, res)
  id_map = {id(p): i for i, p in enumerate(args)}
  in_fwd: list[int | None] = [id_map.get(id(r)) for r in res]
  id_map = {id(p): i for i, p in enumerate(out_primals)}
  out_fwd: list[int | None] = [id_map.get(id(r)) for r in res]
  res = [p for p, f1, f2 in zip(res, in_fwd, out_fwd) if f1 is None and f2 is None]
  aux = (residual_avals, jaxpr, env, in_fwd, out_fwd)
  return res, out_primals, aux


def reduce_precision(x):
  if (h := reduce_precision_handlers.get(type(t := core.typeof(x)))):
    return h(t, x)
  return x

rules: dict[core.Primitive, Callable] = {}
reduce_precision_handlers: dict[type, Callable] = {}


def remat_jaxpr(
    jaxpr, policy, custom_vjp_rules, allow_fwds
) -> tuple[core.Jaxpr, core.Jaxpr, list[int | None]]:
  n = len(jaxpr.outvars)
  if type(allow_fwds) is bool:
    allow_fwds = (allow_fwds,) * n
  assert len(allow_fwds) == n
  return _remat_jaxpr(jaxpr, policy, custom_vjp_rules, tuple(allow_fwds))

@weakref_lru_cache
def _remat_jaxpr(jaxpr, policy, custom_vjp_rules, allow_fwds):
  dbg = jaxpr.debug_info
  fwd_trace = pe.DynamicJaxprTrace(dbg)
  rem_trace = pe.DynamicJaxprTrace(dbg, auto_dce=True)
  rem_trace.tag = core.TraceTag()
  trace = RematTrace(fwd_trace, rem_trace, policy, custom_vjp_rules)
  src = source_info_util.current()

  def new_arg(a):
    return RematTracer(trace, fwd_trace.new_arg(a, src), rem_trace.new_arg(a, src))  # noqa: F821

  tracers = map(new_arg, jaxpr.in_aval_qdds)
  with core.set_current_trace(trace, check_leaks=True):
    ans = core.eval_jaxpr(jaxpr, jaxpr.consts, *tracers)
    out_primals, out_rem = unzip2(map(trace.to_val_tracer_pair, ans))
    del trace, ans, new_arg, tracers

  out_rem = [rem_trace.to_jaxpr_tracer(x, source_info=src) for x in out_rem]
  rem_jaxpr = rem_trace.to_jaxpr(out_rem, dbg.with_unknown_names(), src)
  rem_jaxpr, rem_consts = rem_jaxpr.separate_consts()
  rem_trace.invalidate()

  # Residuals that are just primal outputs needn't be returned again by the
  # fwd jaxpr; the caller passes its own copies of the primal outputs for
  # them, using fwds as the wiring.
  id_map = {id(x): i for i, (x, a) in enumerate(zip(out_primals, allow_fwds)) if a}
  fwds = [id_map.get(id(c)) for c in rem_consts]
  rem_consts = [c for c, f in zip(rem_consts, fwds) if f is None]

  rem_consts = map(partial(fwd_trace.to_jaxpr_tracer, source_info=src), rem_consts)
  out_primals = [fwd_trace.to_jaxpr_tracer(x, source_info=src) for x in out_primals]
  fwd_jaxpr_ = fwd_trace.to_jaxpr(
      [*out_primals, *rem_consts], dbg.with_unknown_names(), src)
  fwd_trace.invalidate()

  return fwd_jaxpr_, rem_jaxpr, fwds
