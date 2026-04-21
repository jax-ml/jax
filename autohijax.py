from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from dataclasses import dataclass
import numpy as np

import jax
import jax.numpy as jnp
from jax._src.array import ArrayImpl
from jax._src.tree_util import FlatTree
from jax._src import core as jax_core
from jax._src import pjit as pjit
from jax._src import dtypes
from jax._src.interpreters import partial_eval as pe
from jax._src.util import safe_zip, safe_map, split_list, unzip2
from jax._src import source_info_util

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

jax.config.update('jax_check_tracer_leaks', True)

# TODO
#   [x] multiple outputs / pytrees
#   [x] vjp (basic)
#   [x] scan
#   [x] grad-of-jit
#   [x] higher-order AD
#   [ ] AD zeros, polymorhpism of +
#   [ ] zero or many lo types from lo_ty()
#   [ ] cond
#   [ ] system for importing existing primitives
#   [ ] custom_vjp
#   [ ] caching/ool functions


# TODO eventually
#  [ ] eager DCE under AD (weakref tape etc)

# === lojax interface ===

lojax_traces = [jax_core.eval_trace]

def start_djt(tys) -> pe.DynamicJaxprTracer:
  trace = pe.DynamicJaxprTrace(None)
  jax_core.trace_ctx.set_trace(trace)
  lojax_traces.append(trace)
  return tys.map(lambda t: trace.new_arg(t, None))

def end_djt(result) -> tuple[jax_core.ClosedJaxpr, list]:
  trace = lojax_traces.pop()
  jax_core.trace_ctx.set_trace(lojax_traces[-1])
  djt_lift = partial(trace.to_jaxpr_tracer, source_info=source_info_util.current())
  result = result.map(pe._canonicalize_dtype).map(djt_lift)
  jaxpr, consts = trace.frame.to_jaxpr(trace, list(result), None, None)
  return pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr)), consts

def emit_jit_call(jaxpr, consts, args):
  # TODO: figure out how to generate jit params so we can do this without roundtripping
  try: return jax.jit(partial(jax_core.eval_jaxpr, jaxpr.jaxpr, jaxpr.consts))(*consts, *args)
  except: breakpoint()

# === core ===

@dataclass
class TraceTimeContext:
  # TODO: maybe keep a stack, not for "parent" but to make it easier to restore
  cur_trace : Trace

  @contextmanager
  def set_current_trace(self, trace):
    prev = self.cur_trace
    try:
      self.cur_trace = trace
      yield
    finally:
      self.cur_trace = prev

class Trace:
  bind_method_name : str
  # Should return the Tracer subclass corresponding to this trace
  def lift(self, val: Tracer) -> Tracer:
    assert False, "subclass should implement"

# Different traces subclass this with their own payloads
class Tracer:
  ty : Ty

  # non-tree args and single result
  def simple_bind(self, *args):
    assert  False, "subclass should implement"

  # args and result are all FlatTrees
  def bind(self, *args):
    assert  False, "subclass should implement"

@dataclass(frozen=True)
class Ty:
  pass

# There are several sorts of values we need to lift
#  * python scalar literals and numpy arrays
#  * tracers from one trace into another, but the eval trace

literal_handlers = {}
# lifting should only need to be done with user-supplied values
def lift(val:Any) -> Val:
  trace_lift = ctx.cur_trace.lift
  if isinstance(val, Tracer):
    return trace_lift(val)
  elif type(val) in literal_handlers:
    return trace_lift(literal_handlers[type(val)](val))
  else:
    breakpoint()
    raise Exception(f"Don't recognize type {type(val)}")

# We flatten user-supplied data. So lifting is usually necessary.
def lift_ft(val:Any) -> FlatTree:
  return FlatTree.flatten(val).map(lift)

# General ops whose methods expose full control at the cost of more boilerplate
class Op:
  def lower(self, trace, *args):
    assert False, f"subclass {type(self)} should implement this"    

# Simple ops that take (non-tree) *args and return a single result
class SimpleOp:
  def simple_lower(self, trace, *args):
    assert False, f"subclass {type(self)} should implement this"    

class BaseTraceCtx: pass

class BaseTrace(Trace):
  ctx : list[BaseTraceCtx]
  def __init__(self, lojaxpr_trace):
    self.ctx = []
    self.lojaxpr_trace = lojaxpr_trace

  def bind(self, op, *args):
    assert isinstance(op, Op)
    assert all(isinstance(arg, FlatTree) for arg in args)
    return op.lower(self, *args)

  def simple_bind(self, op, *args):
    assert isinstance(op, SimpleOp)
    arg_vals = tuple(arg.val for arg in args)
    arg_tys  = tuple(arg.ty  for arg in args)
    ans_ty  = op.simple_type_rule(*arg_tys)
    ans_val = op.simple_lower(*arg_vals)
    return Val(ans_val, ans_ty)

  def lift(self, val):
    assert isinstance(val, Val), breakpoint()
    return val

base_trace = BaseTrace(jax_core.trace_ctx.trace)
ctx = TraceTimeContext(base_trace)

# Tracer for the BaseTrace
class Val(Tracer):
  trace = base_trace
  val : Any  # tuple tree of lojax "vals" (DJP tracers or concrete vals)
  ty : Ty
  def __init__(self, val, ty):
    self.val = val
    self.ty = ty
  def __repr__(self):
    return format(f"{self.val}:{self.ty}")

# === VJP ===

@dataclass
class VjpTraceCtx:
  tape: list[Callable[[], None]]

class VJPTrace(Trace):
  ctx : list[VjpTraceCtx]
  def __init__(self, parent):
    self.ctx = []
    self.parent = parent

  @property
  def tape(self):
    return self.ctx[-1].tape

  def new_arg(self, val):
    return VJPTracer(self, val, Accumulator(val.ty.tangent_ty()))

  def lift(self, val):
    if val.trace is self:
      return val
    else:
      with ctx.set_current_trace(self.parent):
        val = lift(val)
      return VJPTracer(self, val, Accumulator(val.ty.tangent_ty()))

  def bind(self, op, *args):
    return op.vjp(self, *args)

  def simple_bind(self, op, *args):
    primals = tuple(arg.primal for arg in args)
    left_accums = tuple(arg.accum for arg in args)
    with ctx.set_current_trace(self.parent):
      ans = ctx.cur_trace.simple_bind(op, *primals)
      res, pullbacks = unzip2(op.simple_vjp(ans, *primals))
    right_accum = Accumulator(ans.ty.tangent_ty())
    def bwd(res):
      ct = right_accum.finalize()
      for r, pullback, left_accum in zip(res, pullbacks, left_accums):
        left_accum.accum(pullback(r, ct))
        assert left_accum.val is not None
    self.tape.append(Pullback(res, bwd))
    return VJPTracer(self, ans, right_accum)

class VJPTracer(Tracer):
  def __init__(self, trace, primal, accum : Accumulator | None):
    self.trace = trace
    self.primal = primal
    self.accum = accum
    self.ty = primal.ty

never_set = object()
already_finalized = object()

class Accumulator:
  def __init__(self, ty):
    self.ty = ty
    self.val = never_set

  def accum(self, val):
    if self.val is never_set:
      self.val = val
    else: 
      self.val = add(self.val, val)  # TODO: dispatch to ty

  def finalize(self):
    val = self.val 
    del self.val
    assert val is not already_finalized
    return self.ty.zero() if val is never_set else val

def vjp(f, args, ct_right):
  # TODO: zeros on fwd and bwd
  # TODO: think about how to handle refs
  args_ft = lift_ft(args)
  trace = VJPTrace(ctx.cur_trace)
  trace.ctx.append(VjpTraceCtx(tape := []))
  tracers = args_ft.map(trace.new_arg)
  with ctx.set_current_trace(trace):
    ans_ft = lift_ft(f(*tracers.unflatten()))
  ct_right_ft = lift_ft(ct_right)
  ans_ft.map2(lambda tracer, ct: tracer.accum.accum(ct), ct_right_ft)
  while tape: tape.pop()()  # backward pass
  ct_left = tracers.map(lambda tracer: tracer.accum.finalize()) 
  return ans_ft.map(lambda x: x.primal).unflatten(), ct_left.unflatten()

# === built-in HOPs ===

@dataclass
class EnterJitCtx(BaseTraceCtx):
  lo_args : tuple

@dataclass
class EnterJitVjpCtx(VjpTraceCtx):
  left_accum: FlatTree
  local_left_accum: FlatTree

@dataclass
class Pullback:
  res: Any
  pull: Callable
  def __call__(self, *args):
    return self.pull(self.res, *args)
  def __iter__(self):
    return iter((self.res, self.pull))

class EnterJit(Op):
  def lower(self, trace: BaseTrace, args):
    # TODO: two levels of multiplicity: args can be a tree. And also
    # each arg can be a tree of lojax values.
    arg_vals, arg_tys = args.map(lambda x: (x.val, x.ty)).unzip2()
    trace.ctx.append(EnterJitCtx(arg_vals))
    arg_lo_tys = arg_tys.map(lambda t: t.lo_ty())
    local_args_lo = start_djt(arg_lo_tys)
    return local_args_lo.map2(Val, arg_tys)

  def vjp(self, trace: VJPTrace, args):
    primals, arg_accums = args.map(lambda x: (x.primal, x.accum)).unzip2()
    with ctx.set_current_trace(trace.parent):
      local_primals = ctx.cur_trace.bind(EnterJit(), primals)
    local_args = local_primals.map(trace.new_arg)
    local_left_accum = local_args.map(lambda x: x.accum)
    trace.ctx.append(EnterJitVjpCtx([], arg_accums, local_left_accum))
    return local_args

class LeaveJit(Op):
  def lower(self, trace, result):
    enter_ctx = trace.ctx.pop()
    assert isinstance(enter_ctx, EnterJitCtx)
    jaxpr, consts = end_djt(result.map(lambda x: x.val))
    lo_ans = emit_jit_call(jaxpr, consts, enter_ctx.lo_args)
    return result.map2(lambda x, lo: Val(lo, x.ty), lo_ans)

  def vjp(self, trace: VJPTrace, tracers_out):
    enter_ctx = trace.ctx.pop()
    primals_out, local_right_accum = tracers_out.map(lambda x: (x.primal, x.accum)).unzip2()
    res, bwds = unzip2(enter_ctx.tape)
    outs = FlatTree.pack((primals_out, FlatTree.flatten(res)))
    # TODO forwarding
    # TODO env
    with ctx.set_current_trace(trace.parent):
      outs = ctx.cur_trace.bind(LeaveJit(), outs)
    primals_out, res = outs.unpack()
    right_accum = primals_out.map(lambda x: Accumulator(x.ty.tangent_ty()))
    def bwd(res):
      res = FlatTree.flatten(res)
      right_ct = right_accum.map(lambda x: x.finalize())
      res, right_ct = ctx.cur_trace.bind(EnterJit(), FlatTree.pack((res, right_ct))).unpack()
      local_right_accum.map2(lambda a, ct: a.accum(ct), right_ct)
      tape = map(Pullback, res.unflatten(), bwds)
      while tape: tape.pop()()
      left_ct = enter_ctx.local_left_accum.map(lambda a: lift(a.finalize()))
      left_ct = ctx.cur_trace.bind(LeaveJit(), left_ct)
      enter_ctx.left_accum.map2(lambda a, ct: a.accum(ct), left_ct)
    trace.tape.append(Pullback(res.unflatten(), bwd))
    return primals_out.map2(partial(VJPTracer, trace), right_accum)

# TODO: cache and curry
def jit_call(f, *args):
  args = lift_ft(args)
  local_args = ctx.cur_trace.bind(EnterJit(), args)
  ans = lift_ft(f(*local_args.unflatten()))
  return ctx.cur_trace.bind(LeaveJit(), ans).unflatten()
jit = partial(partial, jit_call)

# === library level ===

# TODO: dtype canonicalization etc
# TODO: allow more context for handlers to handle binops
literal_handlers[int]   = lambda x: Val(x, ArrayTy((), np.int32))
literal_handlers[float] = lambda x: Val(x, ArrayTy((), np.float32))
literal_handlers[ArrayImpl] = lambda x: Val(x, ArrayTy(x.shape, x.dtype))
literal_handlers[jax_core.Tracer] = lambda x: Val(x, ArrayTy(x.shape, x.dtype))
literal_handlers[pe.DynamicJaxprTracer] = lambda x: Val(x, ArrayTy(x.shape, x.dtype))

@dataclass(frozen=True)
class ArrayTy(Ty):
  shape : tuple[int]
  dtype : Any

  def lo_ty(self):
    return jax_core.ShapedArray(self.shape, self.dtype)

  def tangent_ty(self):
    # TODO: integer types
    return self

  def zero(self):
    return Val(jnp.zeros(self.shape, self.dtype), ArrayTy(self.shape, self.dtype))

  def __repr__(self):
    return f"{self.dtype}{list(self.shape)}"

class Add(SimpleOp):
  def simple_type_rule(self, x_ty, y_ty):
    assert x_ty == y_ty
    return x_ty

  def simple_vjp(self, _ans, x, y):
    return Pullback((), lambda _, g: g), Pullback((), lambda _, g: g)

  def simple_lower(self, x, y):
    return x + y

def add(x, y):
  x_ = lift(x)
  y_ = lift(y)
  return ctx.cur_trace.simple_bind(Add(), x_, y_)

@dataclass(frozen=True)
class Mul(SimpleOp):
  def simple_type_rule(self, x_ty, y_ty):
    assert x_ty == y_ty, breakpoint()
    return x_ty

  def simple_vjp(self, _ans, x, y):
    return Pullback(y, mul), Pullback(x, mul)

  def simple_lower(self, x, y): 
    return x * y

def mul(x, y):
  x_ = lift(x)
  y_ = lift(y)
  return ctx.cur_trace.simple_bind(Mul(), x_, y_)

# === calling lojax ===

def to_lojax(x:FlatTree):
  return x.map(lambda x: x.val).unflatten()

class CallLojax(Op):
  def __init__(self, f):
    self.f = f

  def lower(self, trace, args_ft):
    return lift_ft(self.f(*to_lojax(args_ft)))

  def vjp(self, trace, args_ft):
    primals = args_ft.map(lambda x: x.primal)
    with ctx.set_current_trace(trace.parent):
      ans_ft, f_vjp_ft = ctx.cur_trace.bind(CallLojax(partial(jax.vjp, self.f)), primals).unpack()
    left_accums = args_ft.map(lambda x: x.accum)
    right_accum = ans_ft.map(lambda x: Accumulator(x.ty.tangent_ty()))
    def bwd(f_vjp_):
      res, treedef = jax.tree.flatten(f_vjp_)
      res = FlatTree.flatten(res)
      def f_vjp(res, right_ct):
        return jax.tree.unflatten(treedef, res)(right_ct)
      right_ct = right_accum.map(lambda x: x.finalize())
      left_cts = ctx.cur_trace.bind(CallLojax(f_vjp), FlatTree.pack((res, right_ct)))
      return left_accums.map2(lambda acc, ct: acc.accum(ct), left_cts)
    trace.tape.append(Pullback(f_vjp_ft.unflatten(), bwd))
    return ans_ft.map2(partial(VJPTracer, trace), right_accum)

def call_lojax(f, *args):
  args_ft = lift_ft(args)
  result = ctx.cur_trace.bind(CallLojax(f), args_ft)
  return result.unflatten()

# === final-style open-face ===

@dataclass
class EnterScanCtx(BaseTraceCtx):
  lo_carry : FlatTree
  lo_xs : FlatTree
  length : int

class EnterScan(Op):
  def __init__(self, length):
    self.length = length

  def lower(self, trace, carry, xs):
    carry_vals, carry_tys = carry.map(lambda x: (x.val, x.ty)).unzip2()
    xs_vals, xs_tys = xs.map(lambda x: (x.val, x.ty)).unzip2()
    x_tys = xs_tys.map(lambda t: ArrayTy(t.shape[1:], t.dtype))  # TODO: generalize
    trace.ctx.append(EnterScanCtx(carry_vals, xs_vals, self.length))
    local_tys = FlatTree.pack((carry_tys, x_tys))
    arg_lo_tys = local_tys.map(lambda t: t.lo_ty())
    local_args_lo = start_djt(arg_lo_tys)
    return local_args_lo.map2(Val, local_tys)

class LeaveScan(Op):
  def __init__(self):
    pass

  def lower(self, trace, c, y):
    ec = trace.ctx.pop()
    assert isinstance(ec, EnterScanCtx)
    jaxpr, consts = end_djt(FlatTree.pack((c, y)).map(lambda x: x.val))
    c_ys = jax.lax.scan_p.bind(*consts, *ec.lo_carry, *ec.lo_xs, jaxpr=jaxpr,
                               length=ec.length, reverse=False, unroll=1,
                               num_consts=len(consts), num_carry=len(ec.lo_carry))
    c, ys = split_list(c_ys, [len(ec.lo_carry)])
    return lift_ft((c, ys))

def scan(body, c, xs, length):
  # TODO: rev, axis name
  c, x = ctx.cur_trace.bind(EnterScan(length), lift_ft(c), lift_ft(xs)).unflatten()
  c, y = body(c, x)
  c, ys = ctx.cur_trace.bind(LeaveScan(), lift_ft(c), lift_ft(y)).unpack()
  return c.unflatten(), ys.unflatten()

# === user level ===

def scan_body(c, x):
  return add(c, 1), add(x, 1)

print(scan(scan_body, 0, jnp.arange(4), length=4))


@jit
def foo(x, y):
  return mul(add(x, y), 2.)

print(vjp(foo, (1., 2.), 1.0))

@jit
def foo(x, y):
  return mul(add(x, y), 2)

@jit
def bar(x, y):
  return add(foo(x, y), y)

print(foo(1, 2))
print(bar(1, 2))

print(vjp(lambda x: mul(mul(x, x), x), (2.0, ), 1.0))

@jit
def baz(x):
  return mul(call_lojax(jnp.sin, x),  x)


print(baz(1.0))
print(vjp(baz, (1.0,), 1.0))

@jit
def closed_over(x, y):
  @jit
  def f(x):
    return add(x, y)
  return f(x)

print(closed_over(1, 2))



def foo(x, y):
  return mul(add(x, y), 2.)

def grad(f):
  def gradfun(*args):
    _, (g, *_) = vjp(f, args, 1.0)
    return g
  return gradfun

print(grad(foo)(1., 1.))
print(grad(grad(lambda x: foo(x, x)))(2.0))

@jit
def baz(x):
  return mul(x, mul(x, x))

print(grad(grad(baz))(1.))

# def fun_with_nested_calls_2(x):
#   def bar(y):
#     def baz(w):
#       return jit_call(lambda _: w, y)
#     _, (t,) = vjp(baz, (1.0,), 3.0)
#     return t
#   return jit_call(bar, x)

# fun_with_nested_calls_2(2.0)
# # grad(fun_with_nested_calls_2)(2.0)
