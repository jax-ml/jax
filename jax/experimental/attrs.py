from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax._src.lax.control_flow.loops import scan_p
from jax.tree_util import (tree_flatten, tree_unflatten, PyTreeDef, tree_map)

from jax._src.core import (Jaxpr, AbstractValue, ClosedJaxpr, raise_to_shaped)
from jax._src.util import (unzip2, safe_map, split_list)
from jax._src.interpreters.ad import jvp_jaxpr
from jax._src.interpreters.partial_eval import (
  DynamicJaxprTrace, DynamicJaxprTracer, extend_jaxpr_stack,
  JaxprStackFrame, source_info_util, _input_type_to_tracers, make_jaxpr_effects)
from jax._src import core
from jax._src.api_util import shaped_abstractify as get_aval

# === utils ===

AttrsTracked = list[tuple[int, str]]
AttrStates = list

def set_states(attrs_tracked : AttrsTracked, vals : AttrStates):
  for ((obj, attr), val) in zip(attrs_tracked, vals):
    jax_setattr(obj, attr, val)

def get_states(attrs_tracked : AttrsTracked):
  return [jax_getattr(obj, attr) for (obj, attr) in attrs_tracked]

# === JAX internals ===

def trace_to_jaxpr_dynamic3(
    fun: Callable,
    in_avals: Sequence[AbstractValue],
    in_tree: PyTreeDef
) -> tuple[ClosedJaxpr, PyTreeDef, AttrsTracked]:
  with core.new_main(DynamicJaxprTrace, dynamic=True) as main:  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    frame = JaxprStackFrame()
    with extend_jaxpr_stack(main, frame), source_info_util.reset_name_stack():
      trace = DynamicJaxprTrace(main, core.cur_sublevel())
      in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
      out_tracers = fun(*tree_unflatten(in_tree, in_tracers))
      (out_tracers_flat, out_tree) = tree_flatten(out_tracers)
      out_tracers_flat = map(trace.full_raise, out_tracers_flat)
      jaxpr, attrs_tracked = to_jaxpr_with_state(frame, out_tracers_flat)
      return jaxpr, out_tree, attrs_tracked

def to_jaxpr_with_state(
    self: JaxprStackFrame, out_tracers
) -> tuple[ClosedJaxpr, AttrsTracked]:
  invars = self.attrs_vars + self.invars
  explicit_outvars = [self.tracer_to_var[id(t)] for t in out_tracers]
  state_outvars    = [self.tracer_to_var[id(t)] for t in get_states(self.attrs_tracked)]
  outvars = state_outvars + explicit_outvars
  constvars, constvals = unzip2(self.constvar_to_val.items())
  jaxpr_effects = make_jaxpr_effects(constvars, self.invars, outvars, self.eqns)
  jaxpr = Jaxpr(constvars, invars, outvars, self.eqns, jaxpr_effects)
  set_states(self.attrs_tracked, self.attrs_inits)
  return ClosedJaxpr(jaxpr, list(constvals)), self.attrs_tracked

def get_current_jaxpr_stack_frame() -> (None | JaxprStackFrame):
  dynamic = core.thread_local_state.trace_state.trace_stack.dynamic
  if dynamic.trace_type is core.EvalTrace:
    return None
  else:
    return dynamic

def ensure_tracked(obj, attr):
  main = get_current_jaxpr_stack_frame()
  if main is None: return
  frame = main.jaxpr_stack[-1]
  if (obj, attr) not in frame.attrs_tracked:
    init_val = getattr(obj, attr)
    aval = raise_to_shaped(get_aval(init_val))
    tracer = DynamicJaxprTracer(main.with_cur_sublevel(), aval, source_info_util.current())
    frame.tracer_to_var[id(tracer)] = var = frame.newvar(aval)
    setattr(obj, attr, tracer)
    frame.attrs_tracked.append((obj, attr))
    frame.attrs_inits.append(init_val)
    frame.attrs_vars.append(var)

# === API to object-attr system ===

@dataclass
class StagedFunction:
  jaxpr         : ClosedJaxpr
  attrs_tracked : AttrsTracked
  pytree_in     : PyTreeDef
  pytree_out    : PyTreeDef

  def __call__(self, *args):
    (args_flat, _) = jax.tree_util.tree_flatten(args)
    init_states = get_states(self.attrs_tracked)
    results = core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *init_states, *args_flat)
    n_states = len(init_states)
    final_states, results = results[:n_states], results[n_states:]
    set_states(self.attrs_tracked, final_states)
    return jax.tree_util.tree_unflatten(self.pytree_out, results)

def stage_out(f, avals) -> StagedFunction:
  (avals_flat, in_tree) = tree_flatten(avals)
  (jaxpr, out_tree, attrs_tracked) = trace_to_jaxpr_dynamic3(f, avals_flat, in_tree)
  return StagedFunction(jaxpr, attrs_tracked, in_tree, out_tree)

def for_loop(n:int, body:Callable[[], None]) -> None:
  staged = stage_out(body, [])
  states = get_states(staged.attrs_tracked)
  num_states = len(states)
  outs = scan_p.bind(*states, length=n, num_consts=0, linear=(False,) * num_states,
                     num_carry=num_states, jaxpr=staged.jaxpr, unroll=1, reverse=False)
  set_states(staged.attrs_tracked, outs)

def jax_setattr(self, attr, val):
  ensure_tracked(self, attr)
  setattr(self, attr, val)

def jax_getattr(self, attr):
  ensure_tracked(self, attr)
  return getattr(self, attr)

# === library code ===

@dataclass
class Thing:
  x : float
  def __hash__(self): return id(self)
  def __eq__(self, other): return self is other

# === user code ===

jax.config.update('jax_traceback_filtering', 'off')

thing = Thing(1.0)

def double_it() -> None:
  cur_x = jax_getattr(thing, "x")
  jax_setattr(thing, "x", cur_x * 2)

double_it_staged = stage_out(double_it, ())

print(thing)  # Thing(x=1.0)
double_it_staged()
print(thing)  # Thing(x=2.0)
double_it()
print(thing)  # Thing(x=4.0)
double_it_staged()
print(thing)  # Thing(x=8.0)
double_it_staged()
print(thing)  # Thing(x=16.0)

thing = Thing(1.0)

def double_it_10() -> None:
  def for_body():
    cur_x = jax_getattr(thing, "x")
    jax_setattr(thing, "x", cur_x * 2.0)
  for_loop(10, for_body)

double_it_staged_10 = stage_out(double_it_10, ())
double_it_staged_10()
print(thing)

# =========================================

@dataclass
class MutableCell:
  value : Any # actually a parameter

TangentVals = Any
PrimalVals = Any
TangentAttrDict = Any

# TODO: jvp version that goes from StagedJaxpr -> StagedJaxpr

# for each tangent ref, we could be (1) reading (2) writing (3) both
def jvp2(fun, tangent_attr_dict, primal_vals, tangent_vals) -> tuple[PrimalVals, TangentVals, TangentAttrDict] :
  primal_avals = tree_map(get_aval, primal_vals)
  # args are (*primal_referents, *primal_vals)
  staged = stage_out(fun, primal_avals)
  # args of fun_jvp are (*primal_referents, *primal_vals, *tangent_referents, *tangent_vals)
  fun_jvp, _ = jvp_jaxpr(staged.jaxpr, (True,)*len(staged.jaxpr.in_avals), True)
  primal_init_states = get_states(staged.attrs_tracked)
  def get_tangent_val(primal_ref):
    primal_obj, attr_name = primal_ref
    maybe_tangent_val = tangent_attr_dict.get(primal_ref)
    if maybe_tangent_val is None:
      # make zeros. Need to get the type from the primal ref val
      assert False
    else:
      return maybe_tangent_val

  tangent_init_states = map(get_tangent_val, staged.attrs_tracked)

  # results are fun_jvp are (*primal_referents, *primal_vals, *tangent_referents, *tangent_vals)
  results = core.eval_jaxpr(fun_jvp.jaxpr, fun_jvp.consts, *primal_init_states,
                            *primal_vals, *tangent_init_states, *tangent_vals)
  num_refs = len(staged.attrs_tracked)
  num_vals = len(primal_vals)
  primal_final_states, primal_vals_out, tangent_final_states, tangent_vals_out = split_list(
    results, [num_refs, num_vals, num_refs])
  set_states(staged.attrs_tracked, primal_final_states)
  tangent_attr_dict_out = dict(zip(staged.attrs_tracked, tangent_final_states))
  return (primal_vals_out, tangent_vals_out, tangent_attr_dict_out)

thing_primal = Thing(2.0)

thing_tangent = MutableCell(0.0)

def thing_do_ad() -> None:
  jax_setattr(thing_primal, "x", jnp.sin(jax_getattr(thing_primal, "x")))

print(thing_do_ad())
print(thing_primal)

# result: ((),())
print(thing_primal)
print(jvp2(thing_do_ad, {(thing_primal,"x") : 2.0}, (), ()))
print(thing_primal)
