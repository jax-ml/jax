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

def start_djt(tys) -> pe.DynamicJaxprTracer:
  trace = pe.DynamicJaxprTrace(None)
  jax_core.trace_ctx.set_trace(trace)
  lojax_traces.append(trace)
  return tys.map(lambda t: trace.new_arg(t, None))

def end_djt(result) -> jax_core.Jaxpr:
  trace = lojax_traces.pop()
  jax_core.trace_ctx.set_trace(lojax_traces[-1])
  jaxpr, consts = trace.frame.to_jaxpr(trace, list(result), None, None)
  return jax_core.ClosedJaxpr(jaxpr, consts)

def emit_jit_call(jaxpr, args):
  # TODO: figure out how to generate jit params so we can do this without roundtripping
  return jax.jit(partial(jax_core.eval_jaxpr, jaxpr.jaxpr, jaxpr.consts))(*args)

# === core ===

@dataclass
class TraceTimeContext:
  cur_trace : Trace

class Trace:
  bind_method_name : str

def bind(op, *args: FlatTree) -> FlatTree:
  assert all(isinstance(arg, FlatTree) for arg in args)
  trace = ctx.cur_trace
  ans = getattr(op, trace.bind_method_name)(trace, *args)
  assert isinstance(ans, FlatTree)
  return ans

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

# === built-in HOPs ===

@dataclass
class EnterJitCtx(BaseTraceCtx):
  lo_args : tuple

class EnterJit(Op):
  def lower(self, trace, args):
    # TODO: two levels of multiplicity: args can be a tree. And also
    # each arg can be a tree of lojax values.
    arg_vals = args.map(lambda x: x.val)
    arg_tys  = args.map(lambda x: x.ty)
    trace.ctx.append(EnterJitCtx(arg_vals))
    arg_lo_tys = arg_tys.map(lambda t: t.lo_ty())
    local_args_lo = start_djt(arg_lo_tys)
    return local_args_lo.map2(Val, arg_tys)

class LeaveJit(Op):
  def lower(self, trace, result):
    enter_ctx = trace.ctx.pop()
    assert isinstance(enter_ctx, EnterJitCtx)
    jaxpr = end_djt(result.map(lambda x: x.val))
    lo_ans = emit_jit_call(jaxpr, enter_ctx.lo_args)
    return result.map2(lambda x, lo: Val(lo, x.ty), lo_ans)

# TODO: cache and curry
def jit_call(f, *args):
  args = FlatTree.flatten(args).map(lift)
  local_args = bind(EnterJit(), args)
  ans = f(*local_args.unflatten())
  ans =  FlatTree.flatten(ans).map(lift)
  return bind(LeaveJit(), ans).unflatten()
jit = partial(partial, jit_call)

# === helper for library-level ops ===

class SimpleOp(Op):
  def lower(self, _, *args):
    args = tuple(arg.from_leaf() for arg in args)
    arg_vals = tuple(arg.val for arg in args)
    arg_tys  = tuple(arg.ty  for arg in args)
    ans_ty  = self.simple_type_rule(*arg_tys)
    ans_val = self.simple_lower(*arg_vals)
    return FlatTree.leaf(Val(ans_val, ans_ty))

def simple_bind(op, *args):
  args = tuple(FlatTree.leaf(lift(arg)) for arg in args)
  return bind(op, *args).from_leaf()

# === library level ===


literal_handlers[int] = lambda x: Val(x, ArrayTy((), np.int32))
literal_handlers[float] = lambda x: Val(x, ArrayTy((), np.float32))

@dataclass(frozen=True)
class ArrayTy(Ty):
  shape : tuple[int]
  dtype : Any

  def lo_ty(self):
    return jax_core.ShapedArray(self.shape, self.dtype)

class Add(SimpleOp):
  def simple_type_rule(self, x_ty, y_ty):
    assert x_ty == y_ty
    return x_ty

  def simple_lower(self, x, y):
    return x + y

add = partial(simple_bind, Add())

@dataclass(frozen=True)
class Mul(SimpleOp):
  def simple_type_rule(self, x_ty, y_ty):
    assert x_ty == y_ty
    return x_ty

  def simple_lower(self, x, y): 
    return x * y

mul = partial(simple_bind, Mul())

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

