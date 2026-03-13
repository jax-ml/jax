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
from typing import Callable

from jax._src import core
from jax._src import api_util
from jax._src.util import safe_map, safe_zip, unzip2, weakref_lru_cache
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (
    FlatTree, Partial, tree_unflatten, tree_leaves_checked)
from jax._src import source_info_util

map = safe_map
zip = safe_zip

# TODO
#  [ ] static_argnums and static_argnames (via FlatTree)
#  [ ] allow NotAvailable sentinels
#  [ ] DCE pass
#  [ ] primal-output-to-residual forwarding

def remat_transform(policy, f, *args):
  dbg = api_util.debug_info("remat", f, args, {})
  with core.take_current_trace() as parent_trace:
    jaxpr_trace = pe.DynamicJaxprTrace(None)
    trace = RematTrace(parent_trace, jaxpr_trace, core.TraceTag(), policy)
    args_ft = FlatTree.flatten_static_argnums_argnames(args, {}, (), ())
    new_arg = lambda x: RematTracer(trace, x, jaxpr_trace.new_arg(core.typeof(x), None))  # noqa F821  # type: ignore
    in_tracers = args_ft.map(new_arg)
    with core.set_current_trace(trace):
      args, kwargs = in_tracers.unflatten()
      ans_pytree = f(*args, **kwargs)
      dbg = dbg.set_result_paths(ans_pytree)
      ans_ft = FlatTree.flatten(ans_pytree)
      del ans_pytree, args, kwargs
    out_ft, out_tracer_ft = ans_ft.map(trace.to_val_tracer_pair).unzip2()
    jaxpr, rs = jaxpr_trace.to_jaxpr(list(out_tracer_ft), dbg, source_info_util.current())
    in_tree, out_tree = args_ft.tree, out_ft.tree
    del trace, in_tracers, out_tracer_ft
  def f_rem(rs, *args):
    args_flat = tree_leaves_checked(in_tree, (args, {}))
    out_flat = core.eval_jaxpr(jaxpr, rs, *args_flat)
    return tree_unflatten(out_tree, out_flat)
  return out_ft.unflatten(), Partial(f_rem, map(reduce_precision, rs))

class RematTracer(core.Tracer['RematTrace']):
  _trace: RematTrace

  def __init__(self, trace, x, jaxpr_tracer):
    self._trace = trace  # pytype: disable=name-error
    self.val = x
    self.tracer = jaxpr_tracer

  @property
  def aval(self):
    return core.typeof(self.val)

class RematTrace(core.Trace):
  def __init__(self, parent_trace, jaxpr_trace, tag, policy):
    super().__init__()
    self.parent_trace = parent_trace
    self.jaxpr_trace = jaxpr_trace
    self.tag = tag
    self.policy = policy
    self.requires_low = False

  def to_val_tracer_pair(self, x):
    if isinstance(x, RematTracer) and x._trace.tag is self.tag:
      return (x.val, x.tracer)
    else:
      raise NotImplementedError  # TODO(mattjj)

  def process_primitive(self, prim, tracers, params, /):
    in_vals, in_vals2 = unzip2(map(self.to_val_tracer_pair, tracers))
    if prim in rules:
      with core.set_current_trace(self.parent_trace):
        out_primal, rem = rules[prim](self.policy, *in_vals, **params)
      with core.set_current_trace(self.jaxpr_trace):
        out_primal2 = rem(*in_vals2)
    else:
      with core.set_current_trace(self.parent_trace):
        out_primal = prim.bind(*in_vals, **params)
      with core.set_current_trace(self.jaxpr_trace):
        out_primal2 = prim.bind(*in_vals2, **params)
    if prim.multiple_results:
      return map(partial(RematTracer, self), out_primal, out_primal2)
    else:
      return RematTracer(self, out_primal, out_primal2)

def reduce_precision(x):
  if (h := reduce_precision_handlers.get(type(t := core.typeof(x)))):
    return h(t, x)
  return x

rules: dict[core.Primitive, Callable] = {}
reduce_precision_handlers: dict[type, Callable] = {}


def remat_jaxpr(jaxpr, policy):
  return _remat_jaxpr(jaxpr, frozenset(policy))

@weakref_lru_cache
def _remat_jaxpr(jaxpr, policy):
  dbg = jaxpr.jaxpr.debug_info
  fwd_trace = pe.DynamicJaxprTrace(dbg)
  rem_trace = pe.DynamicJaxprTrace(dbg, auto_dce=True)
  tag = core.TraceTag()
  trace = RematTrace(fwd_trace, rem_trace, tag, policy)
  rem_trace.tag = tag
  src = source_info_util.current()

  def new_arg(a):
    return RematTracer(trace, fwd_trace.new_arg(a, src), rem_trace.new_arg(a, src))  # noqa: F821  # pytype: disable=name-error

  tracers = map(new_arg, jaxpr.in_aval_qdds)
  with core.set_current_trace(trace, check_leaks=True):
    ans = core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *tracers)
    out_primals, out_rem = unzip2(map(trace.to_val_tracer_pair, ans))
    del trace, ans, new_arg, tracers

  rem_jaxpr_, rem_consts = rem_trace.to_jaxpr(out_rem, dbg.with_unknown_names(), src)
  rem_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(rem_jaxpr_))
  rem_trace.invalidate()
  rem_consts = map(partial(fwd_trace.to_jaxpr_tracer, source_info=src), rem_consts)
  fwd_jaxpr_, fwd_consts = fwd_trace.to_jaxpr(
      [*out_primals, *rem_consts], dbg.with_unknown_names(), src)
  fwd_trace.invalidate()
  fwd_jaxpr = core.ClosedJaxpr(fwd_jaxpr_, fwd_consts)
  return fwd_jaxpr, rem_jaxpr, len(rem_consts)
