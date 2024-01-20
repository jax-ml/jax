from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import jax
from jax.tree_util import (tree_flatten, tree_unflatten, PyTreeDef)

from jax._src.core import (Jaxpr, AbstractValue, ClosedJaxpr, raise_to_shaped)
from jax._src.util import (unzip2, safe_map)
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
    setattr(obj, attr, val)

def get_states(attrs_tracked : AttrsTracked):
  return [getattr(obj, attr) for (obj, attr) in attrs_tracked]

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

# === user code ===

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
