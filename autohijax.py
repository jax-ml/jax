from __future__ import annotations

from functools import partial
from dataclasses import dataclass
import numpy as np

import jax
from jax._src.tree_util import FlatTree
from jax._src import core as jax_core
from jax._src import pjit as pjit
from jax._src.interpreters import partial_eval as pe
from jax._src.util import safe_zip, safe_map

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

# === lojax interface ===

lojax_traces = [jax_core.eval_trace]

def start_djt(*tys) -> pe.DynamicJaxprTracer:
  trace = pe.DynamicJaxprTrace(None)
  jax_core.trace_ctx.set_trace(trace)
  lojax_traces.append(trace)
  return [trace.new_arg(ty, None) for ty in tys]

def end_djt(result) -> jax_core.Jaxpr:
  trace = lojax_traces.pop()
  jax_core.trace_ctx.set_trace(lojax_traces[-1])
  jaxpr, consts = trace.frame.to_jaxpr(trace, [result], None, None)
  return jax_core.ClosedJaxpr(jaxpr, consts)

def emit_jit_call(jaxpr, args):
  # TODO: figure out how to generate jit params so we can do this without roundtripping
  ans, = jax.jit(partial(jax_core.eval_jaxpr, jaxpr.jaxpr, jaxpr.consts))(*args)
  return ans

# === core ===

@dataclass
class TraceTimeContext:
  cur_trace : Trace

class Trace:
  bind_method_name : str

def bind(op, *args):
  trace = ctx.cur_trace
  return getattr(op, trace.bind_method_name)(trace, *args)

@dataclass(frozen=True)
class Ty:
  pass

@dataclass(frozen=True)
class Val:
  val : Any  # tuple tree of lojax "vals" (tracers or concrete vals)
  ty : Ty

literal_handlers = {}
def lift(val:Any) -> Val:
  if isinstance(val, Val):
    return val
  elif type(val) in literal_handlers:
    return literal_handlers[type(val)](val)
  else:
    raise Exception(f"Don't recognize type {type(val)}")

class Op:
  def lower(self, trace, *args):
    assert False, "subclass should implement this"    

class BaseTraceCtx: pass

class BaseTrace:
  lojaxpr_trace : core.Trace
  ctx : list[BaseTraceCtx]
  def __init__(self, lojaxpr_trace):
    self.ctx = []
    self.lojaxpr_trace = lojaxpr_trace
    self.bind_method_name = "lower"

ctx = TraceTimeContext(BaseTrace(jax_core.trace_ctx.trace))

@dataclass
class EnterJitCtx(BaseTraceCtx):
  lo_args : tuple

class EnterJit(Op):
  def lower(self, trace, *args):
    # TODO: two levels of multiplicity: args can be a tree. And also
    # each arg can be a tree of lojax values.
    trace.ctx.append(EnterJitCtx(tuple(arg.val for arg in args)))
    arg_tys = [arg.ty for arg in args]
    arg_lo_tys = [t.lo_ty() for t in arg_tys]
    local_args_lo = start_djt(*arg_lo_tys)
    return tuple(Val(x, t) for x, t in zip(local_args_lo, arg_tys))

class LeaveJit(Op):
  def lower(self, trace, result):
    enter_ctx = trace.ctx.pop()
    assert isinstance(enter_ctx, EnterJitCtx)
    jaxpr = end_djt(result.val)
    lo_ans = emit_jit_call(jaxpr, enter_ctx.lo_args)
    return Val(lo_ans, result.ty)

# TODO: cache and curry
def jit_call(f, *args):
  args = [lift(arg) for arg in args]
  ans = f(*bind(EnterJit(), *args))
  ans =  lift(ans)
  return bind(LeaveJit(), ans)
jit = partial(partial, jit_call)

# === library level ===

literal_handlers[int] = lambda x: Val(x, ArrayTy((), np.int32))
literal_handlers[float] = lambda x: Val(x, ArrayTy((), np.float32))

@dataclass(frozen=True)
class ArrayTy(Ty):
  shape : tuple[int]
  dtype : Any

  def lo_ty(self):
    return jax_core.ShapedArray(self.shape, self.dtype)

class Add(Op):
  def lower(self, _, x, y):
    assert x.ty == y.ty
    return Val(x.val + y.val, x.ty)

def add(x, y):
  x_ = lift(x)
  y_ = lift(y)
  return bind(Add(), x_, y_)

@dataclass(frozen=True)
class Mul(Op):
  def lower(self, _, x, y):
    assert x.ty == y.ty
    return Val(x.val * y.val, x.ty)

def mul(x, y):
  x_ = lift(x)
  y_ = lift(y)
  return bind(Mul(), x_, y_)

# === user level ===

@jit
def foo(x, y):
  return mul(add(x, y), 2)

@jit
def bar(x, y):
  return add(foo(x, y), y)

print(foo(1, 2))
print(bar(1, 2))

# TODO
#   * multiple outputs
#   * cond
#   * vjp
#   * custom_vjp
#   * scan

