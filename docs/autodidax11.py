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

import io
import numpy as np
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Any, Callable, Union

from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func
from jax.extend.mlir.dialects import stablehlo as hlo
from jax._src import xla_bridge as xb

# === utils ===

def map(f, *xs):
  return tuple(builtins.map(f, *xs))

# === printing utils ===

class Printer:
  lines: list[(int, str)]
  cur_indent: int

  def __init__(self):
    self.lines = []
    self.cur_indent = 0

  def emit_line(self, s:str):
    self.lines.append((self.cur_indent, s))

  def to_str(self):
    return '\n'.join(['  ' * indent + s for indent, s in self.lines]) + '\n'

  @contextmanager
  def indent(self):
    prev = self.cur_indent
    self.cur_indent = prev + 1
    yield
    self.cur_indent = prev

def str_from_print_lines(self):
  p = Printer()
  self.print_lines(p)
  return p.to_str()

def print_args(args:list[str]):
  return "(" + ", ".join(args) + ")"

# === syntax ===

class Ty:
  def to_ir_type(self):
    raise NotImplementedError(f"{type(self).__name__}.to_ir_type")

  def tangent_ty(self):
    raise NotImplementedError(f"{type(self).__name__}.tangent_ty")

  def strip_leading_axis(self):
    raise NotImplementedError(f"{type(self).__name__}.strip_leading_axis")

  def broadcast_leading_axis(self, val, size):
    raise NotImplementedError(f"{type(self).__name__}.broadcast_leading_axis")

class Val:
  ty : Ty

  def to_runtime_buf(self):
    raise NotImplementedError(f"{type(self).__name__}.to_runtime_buf")

  def to_hlo_const(self, c):
    raise NotImplementedError(f"{type(self).__name__}.to_hlo_const")

  def __add__(self, other): return self.ty._add(self, other)
  def __mul__(self, other): return self.ty._mul(self, other)
  def __radd__(self, other): return self.ty._add(other, self)
  def __rmul__(self, other): return self.ty._mul(other, self)
  def __repr__(self): return self.__str__()

class Primitive:
  arg_tys : tuple[Ty]
  result_ty : Ty
  is_linear : bool = False
  linear_args : tuple[int] = ()

  # type checking goes here
  def __init__(self, *arg_tys, **params):
    raise NotImplementedError(f"{type(self).__name__}.__init__")

  # user-facing call with raw args
  def call(self, *args, **kwargs):
    raise NotImplementedError(f"{type(self).__name__}.call")

  def to_hlo(self, c, *hlo_args):
    raise NotImplementedError(f"{type(self).__name__}.to_hlo")

  def linearize(self, *args):
    raise NotImplementedError(f"{type(self).__name__}.linearize")

  def transpose(self, *args):
    raise NotImplementedError(f"{type(self).__name__}.transpose")

  def __str__(self):
    s = type(self).__name__
    if self.is_linear:
      s = "lin_" + s
    return s

class Var:
  ty   : Ty
  name : str
  def __init__(self, ty, name):
    self.ty = ty
    self.name = name

  def __str__(self):
    return f"%{self.name}: {self.ty}"

ScalarVal = Val
Atom = Union[Var, ScalarVal]

class Eqn:
  binder : Var
  prim   : Primitive
  args   : tuple[Atom]
  def __init__(self, binder, prim, args):
    self.binder = binder
    self.prim = prim
    self.args = args

  def print_lines(self, p):
    if hasattr(self.prim, "print_prim_app"):
      p.emit_line(f"{self.binder} =")
      with p.indent():
        self.prim.print_prim_app(p, self.args)
    else:
      p.emit_line(f"{self.binder} = {self.prim}{print_args(map(str, self.args))}")

  __str__ = str_from_print_lines

class Jaxpr:
  binders : tuple[Var]
  eqns : tuple[Eqn]
  result : Atom
  def __init__(self, binders, eqns, result):
    self.binders = binders
    self.eqns = eqns
    self.result = result

  def print_lines(self, p):
    p.emit_line(f"{print_args(map(str, self.binders))} ->")
    with p.indent():
      for eqn in self.eqns:
        eqn.print_lines(p)
      p.emit_line(f"{self.result}")

  __str__ = str_from_print_lines

# === tracing ===

class Tracer:
  ty : Ty

  def __add__(self, other): return self.ty._add(self, other)
  def __mul__(self, other): return self.ty._mul(self, other)
  def __radd__(self, other): return self.ty._add(other, self)
  def __rmul__(self, other): return self.ty._mul(other, self)
  def __getitem__(self, i): return self.ty._getitem(self, i)

TraceVal = Union[Tracer, Val]

class Trace:
  def process_primitive(self, prim:Primitive, args):
    raise NotImplementedError(f"{type(self).__name__}.process_primitive")

class EvalTrace(Trace):
  def process_primitive(self, prim, args):
    return compile_prim(prim).execute(*args)

class TraceTag: pass

@dataclass
class TraceCtx:
  trace : Trace

def emit(prim:Primitive, args):
  trace = trace_ctx.trace
  return trace.process_primitive(prim, args)

def canonicalize_arg(arg: Any) -> TraceVal:
  if isinstance(arg, (Val, Tracer)):
    return arg
  elif isinstance(arg, np.ndarray):
    return ArrayVal(arg)
  elif isinstance(arg, (float, np.float64, np.float32)):
    return ArrayVal(np.array(arg))
  elif type(arg) is tuple:
    elts = map(canonicalize_arg, arg)
    if all(isinstance(elt, Val) for elt in elts):
      return TupleVal(elts)
    else:
      elt_tys = tuple(elt.ty for elt in elts)
      return emit(TupleCon(elt_tys), elts)
  else:
    raise TypeError(f"Unrecognized type: {arg}")

eval_trace = EvalTrace()
trace_ctx = TraceCtx(eval_trace)

@contextmanager
def set_current_trace(trace, check_leaks=False):
  prev = trace_ctx.trace
  try:
    trace_ctx.trace = trace
    yield
  finally:
    trace_ctx.trace= prev

# === jaxpr tracing ===

class GenSym:
  def __init__(self):
    self.cur = 0

  def new(self):
    self.cur += 1
    return self.cur
gensym = GenSym()

class JaxprTrace(Trace):
  binders : list[Var]
  eqns : list[Eqn]
  def __init__(self):
    self.binders = []
    self.eqns = []

  def to_atom(self, x:TraceVal):
    if isinstance(x, JaxprTracer):
      return x.var
    elif isinstance(x, Val):
      return x
    else:
      raise TypeError(f"Unexpected TraceVal: {x}")

  def process_primitive(self, prim, args):
    arg_atoms = map(self.to_atom, args)
    binder = Var(prim.result_ty, gensym.new())
    self.eqns.append(Eqn(binder, prim, arg_atoms))
    return JaxprTracer(binder)

  def new_arg(self, ty:Ty) -> Tracer:
    v = Var(ty, gensym.new())
    self.binders.append(v)
    return JaxprTracer(v)

  def finalize_jaxpr(self, result:TraceVal):
    result_atom = self.to_atom(result)
    return Jaxpr(tuple(self.binders), tuple(self.eqns), result_atom)

class JaxprTracer(Tracer):
  var: Var
  def __init__(self, v:Var):
    self.var = v
    self.ty = v.ty

def trace_to_jaxpr(f:Callable, arg_tys:tuple[Ty]) -> Jaxpr:
  trace = JaxprTrace()
  args = tuple(trace.new_arg(t) for t in arg_tys)
  with set_current_trace(trace):
    result = canonicalize_arg(f(*args))
  return trace.finalize_jaxpr(result)

# === types ===

class ArrayTy(Ty):
  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def to_ir_type(self):
    return ir.RankedTensorType.get(self.shape, _mlir_dtype(self.dtype))

  def tangent_ty(self):
    # TODO: ints
    return ArrayTy(self.shape, self.dtype)

  def strip_leading_axis(self):
     if len(self.shape) == 0:
       raise TypeError("Can't batch a rank-0 array")
     else:
       return self.shape[0], ArrayTy(self.shape[1:], self.dtype)

  def broadcast_leading_axis(self, val, size):
    raise NotImplementedError("need to implement broadcast")

  def from_runtime_buf(self, buf):
    val = ArrayVal(np.asarray(buf))
    assert val.ty == self
    return val

  def __eq__(self, other):
    return self.shape == other.shape and self.dtype == other.dtype

  def __str__(self):
    return f"{self.dtype}{list(self.shape)}"

  def _mul(self, x, y): return mul(x, y)
  def _add(self, x, y): return add(x, y)

class ArrayVal(Val):
  def __init__(self, x):
    assert isinstance(x, np.ndarray)
    self.ty = ArrayTy(x.shape, x.dtype)
    self.val = x

  def to_runtime_buf(self):
    return xb.get_backend(None).buffer_from_pyval(self.val)


  def to_hlo_const(c, self):
    a = np.asarray(x)
    if a.dtype == np.bool_:
      return hlo.constant(ir.DenseElementsAttr.get(
        np.packbits(a, bitorder='little'), type=ir.IntegerType.get_signless(1),
        shape=a.shape))
    else:
      return hlo.constant(ir.DenseElementsAttr.get(a))

  def __str__(self):
    return str(self.val)

# === tuples ===

class TupleTy(Ty):
  def __init__(self, elt_tys):
    self.elt_tys = elt_tys

  def __str__(self):
    return print_args(map(str, self.elt_tys))

  def _getitem(self, xs, i):
    return tuple_proj(xs, i)

class TupleVal(Val):
  def __init__(self, elts):
    self.elts = elts
    self.ty = TupleTy(tuple(elt.ty for elt in elts))

  def to_hlo_const(self, c):
    return tuple(x.to_hlo_const(c) for x in self.elts)

  def __str__(self):
    return print_args(map(str, self.elts))

# === linearize ===

class LinearizeTrace(Trace):

  def __init__(self, parent_trace, tangent_trace, tag=None):
    self.tag = TraceTag() if tag is None else tag
    self.parent_trace = parent_trace
    self.tangent_trace = tangent_trace

  def process_primitive(self, prim: Primitive, args):
    args = map(self.lift, args)
    primals = tuple(arg.primal for arg in args)
    tangents = tuple(arg.tangent for arg in args)
    if all(t is None for t in tangents):
      return self.parent_trace.process_primitive(prim, primals)
    assert all(t is not None for t in tangents)  # TODO: instantiate zeros
    with set_current_trace(self.parent_trace):
      primal_out, linearized = prim.linearize(*primals)
    with set_current_trace(self.tangent_trace):
      tangent_out = linearized(*tangents)
    return LinearizeTracer(self, primal_out, tangent_out)

  def lift(self, val):
    if isinstance(val, LinearizeTracer) and val.trace.tag is self.tag:
      return LinearizeTracer(self, val.primal, val.tangent)
    else:
      tangent_zero = Zero.from_primal_value(val)
      return LinearizeTracer(self, val, tangent_zero)

class Zero:
  ty : Ty
  def __init__(self, ty):
    self.ty = ty

class LinearizeTracer(Tracer):
  trace   : Trace
  primal  : TraceVal
  tangent : TraceVal | Zero
  def __init__(self, trace, primal, tangent):
    self.trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def ty(self):
    return self.primal.ty

def linearize(f, primals):
  primals = map(canonicalize_arg, primals)
  parent_trace = trace_ctx.trace
  tangent_trace = JaxprTrace()
  tangent_ty = primals[0].ty.tangent_ty()
  tangent = tangent_trace.new_arg(tangent_ty)
  lin_trace = LinearizeTrace(parent_trace, tangent_trace)
  arg0_tracer = LinearizeTracer(lin_trace, primals[0], tangent)
  rest_tracers = [LinearizeTracer(lin_trace, p, Zero(p.ty.tangent_ty())) for p in primals[1:]]
  with set_current_trace(lin_trace):
    ans = f(arg0_tracer, *rest_tracers)

  ans = lin_trace.lift(canonicalize_arg(ans))
  tangent_jaxpr = tangent_trace.finalize_jaxpr(ans.tangent)
  return ans.primal, tangent_jaxpr

# === transpose ===

def backward_pass(jaxpr, initial_cotangent):
  initial_cotangent = canonicalize_arg(initial_cotangent)
  # forward pass on non-linear eqns
  primal_env: dict[Var, TraceVal] = {}
  def eval_arg(x) -> TraceVal:
    return primal_env[x] if isinstance(x, Var) else x

  for eqn in jaxpr.eqns:
    if eqn.prim.is_linear: continue
    primal_env[eqn.binder] = emit(eqn.prim, map(eval_arg, eqn.args))

  # actual backward pass
  ct_env: dict[Var, TraceVal] = {}
  def read_cotangent(v: Var) -> Any:
    return ct_env.pop(v, Zero(v.ty))

  def write_cotangent(x: Atom, val: TraceVal):
    assert isinstance(x, Var)
    assert x.ty == val.ty
    ct_env[x] = add_lin(ct_env[x], val) if x in ct_env else val

  write_cotangent(jaxpr.result, initial_cotangent)
  for eqn in jaxpr.eqns[::-1]:
    if not eqn.prim.is_linear: continue
    primals = [eval_arg(arg) for i, arg in enumerate(eqn.args) if i not in eqn.prim.linear_args]
    ct_in = read_cotangent(eqn.binder)
    cts_out = eqn.prim.transpose(ct_in, *primals)
    for ct_out, arg_index in zip(cts_out, eqn.prim.linear_args):
      write_cotangent(eqn.args[arg_index], ct_out)

  binder, = jaxpr.binders
  return read_cotangent(binder)

# === vmap ===

class BatchTrace:
  def __init__(self, parent_trace, size, tag=None):
    self.parent_trace = parent_trace
    self.size = size
    self.tag = TraceTag() if tag is None else tag

  def process_primitive(self, prim: Primitive, args: tuple[TraceVal]):
    args = map(self.lift, args)
    with set_current_trace(self.parent_trace):
      if all(not arg.batched for arg in args):
        args = [arg.val for arg in args]
        return emit(prim, args)
      else:
        args = [arg.force_batched().val for arg in args]
        result = prim.batch(*args)
        return BatchTracer(self, result, True)

  def lift(self, val):
    if isinstance(val, BatchTracer) and val.trace.tag is self.tag:
      return BatchTracer(self, val.val, val.batched)
    else:
      return BatchTracer(self, val, False)

class BatchTracer(Tracer):
  def __init__(self, trace:Trace, val:TraceVal, batched:bool):
    self.trace = trace
    self.val = val
    self.batched = bool

  @property
  def ty(self):
    if self.batched:
      _, ty = self.val.ty.strip_leading_axis()
      return ty
    else:
      return self.val.ty

  def force_batched(self):
    if self.batched:
      return self
    else:
      return self.ty.broadcast_leading_axis(self.val, self.trace.size)

def vmap(f, *args):
  args = map(canonicalize_arg, args)
  sizes = [arg.ty.strip_leading_axis()[0] for arg in args]
  size = sizes[0]
  for other_size in sizes:
    assert size == other_size

  parent_trace = trace_ctx.trace
  trace = BatchTrace(parent_trace, size)
  args = [BatchTracer(trace, arg, True) for arg in args]
  with set_current_trace(trace):
    ans = f(*args)

  ans = trace.lift(ans)
  return ans.force_batched().val

# === XLA stuff ===

@dataclass
class CompiledObject:
  mlir_module : Any
  result_ty : Ty

  def execute(self, *args: Val):
    input_bufs = tuple(x.to_runtime_buf() for x in args)
    out_buf, = self.mlir_module.execute(input_bufs)
    return self.result_ty.from_runtime_buf(out_buf)

class MlirContext(NamedTuple):
  module: ir.Module
  symbol_table: ir.SymbolTable

def compile_prim(prim: Primitive) -> CompiledObject:
  with ir.Context() as ctx, ir.Location.unknown(ctx):
    hlo.register_dialect(ctx)
    m = ir.Module.create()
    c = MlirContext(m, ir.SymbolTable(m.operation))
    with ir.InsertionPoint(c.module.body):
      @func.func(*(ty.to_ir_type() for ty in prim.arg_tys))
      def main(*args):
        return prim.to_hlo(c, *args)

  output = io.StringIO()
  c.module.operation.print(file=output)
  compiled = xb.get_backend(None).compile(output.getvalue())
  return CompiledObject(compiled, prim.result_ty)

def _mlir_dtype(dtype: np.dtype) -> ir.Type:
  if np.issubdtype(dtype, np.signedinteger):
    return ir.IntegerType.get_signless(np.iinfo(dtype).bits)
  elif dtype == np.float32:
    return ir.F32Type.get()
  elif dtype == np.float64:
    return ir.F64Type.get()
  else:
    raise NotImplementedError("MLIR conversion not implemented for ", dtype)

# === ops ===

class Sin(Primitive):
  def __init__(self, ty):
    assert isinstance(ty, ArrayTy)
    self.arg_tys = (ty,)
    self.result_ty = ty

  def to_hlo(self, _, x):
    return hlo.sine(x)

  def linearize(self, x):
    cos_x = cos(x)
    return sin(x), lambda t: mul_lin(t, cos_x)

  def batch(self, x):
    return sin(x)

def sin(x):
  x = canonicalize_arg(x)
  return emit(Sin(x.ty), (x,))

class Cos(Primitive):
  def __init__(self, ty):
    assert isinstance(ty, ArrayTy)
    self.arg_tys = (ty,)
    self.result_ty = ty

  def to_hlo(self, _, x):
    return hlo.cosine(x)

  def linearize(self, x):
    sin_x = sin(x)
    return cos(x), lambda t: neg_lin(mul_lin(t, sin_x))

  def batch(self, x):
    return cos(x, y)

def cos(x):
  x = canonicalize_arg(x)
  return emit(Cos(x.ty), (x,))

class Mul(Primitive):
  linear_args = (0,)

  def __init__(self, ty, is_linear=False):
    self.arg_tys = (ty, ty)
    self.result_ty = ty
    # Mul is linear in its first argument only
    self.is_linear = is_linear

  def to_hlo(self, _, x, y):
    return hlo.multiply(x, y)

  def linearize(self, x):
    def lin(xt, yt):
      return add_lin(mul_lin(xt, y),
                     mul_lin(yt, x))
    return mul(x, y),

  def transpose(self, ct, x):
    return (mul_lin(ct, x),)

  def batch(self, x, y):
    return mul(x, y)

def mul(x, y):
  x = canonicalize_arg(x)
  y = canonicalize_arg(y)
  assert x.ty == y.ty
  return emit(Mul(x.ty), (x, y))

# linear in first argument
def mul_lin(x, y):
  if isinstance(x, Zero):
    return x
  else:
    x = canonicalize_arg(x)
    y = canonicalize_arg(y)
    assert x.ty == y.ty
    return emit(Mul(x.ty, is_linear=[0]), (x, y))

class Add(Primitive):
  linear_args = (0, 1)

  def __init__(self, ty, is_linear=False):
    self.arg_tys = (ty, ty)
    self.result_ty = ty
    self.is_linear = is_linear

  def to_hlo(self, _, x, y):
    return hlo.add(x, y)

  def linearize(self, x, y):
    return add(x, y), lambda xt, yt: add_lin(xt, yt)

  def batch(self, x, y):
    return add(x, y)

def add(x, y):
  x = canonicalize_arg(x)
  y = canonicalize_arg(y)
  assert x.ty == y.ty
  return emit(Add(x.ty), (x, y))

def add_lin(x, y):
  if isinstance(x, Zero):
    return y
  elif isinstance(y, Zero):
    return x
  else:
    x = canonicalize_arg(x)
    y = canonicalize_arg(y)
    assert x.ty == y.ty
    return emit(Add(x.ty, is_linear=True), (x, y))


class Neg(Primitive):
  linear_args = (0,)

  def __init__(self, ty, is_linear=False):
    self.arg_tys = (ty,)
    self.result_ty = ty
    self.is_linear = is_linear

  def to_hlo(self, _, x):
    return hlo.negate(x)

  def linearize(self, x):
    return neg(x), lambda xt: neg_lin(xt)

  def batch(self, x):
    return neg(x)

def neg(x):
  x = canonicalize_arg(x)
  return emit(Neg(x.ty), (x,))

def neg_lin(x):
  if isinstance(x, Zero):
    return x
  else:
    x = canonicalize_arg(x)
    return emit(Neg(x.ty, is_linear=True), (x,))

class TupleProj(Primitive):
  def __init__(self, elt_tys, i):
    self.arg_tys = TupleTy(elt_tys)
    self.result_ty = elt_tys[i]

def tuple_proj(x, i):
  x = canonicalize_arg(x)
  elt_tys = x.ty.elt_tys
  return emit(TupleProj(elt_tys, i), (x,))

class TupleCon(Primitive):
  def __init__(self, elt_tys):
    self.arg_tys = elt_tys
    self.result_ty = TupleTy(elt_tys)

  def to_hlo(self, _, *elts):
    return tuple(elts)

class Call(Primitive):
  def __init__(self, jaxpr):
    self.jaxpr = jaxpr
    self.arg_tys = tuple(b.ty for b in jaxpr.binders)
    self.result_ty = jaxpr.result.ty

  def to_hlo(self, c, *args):
    with ir.InsertionPoint(c.module.body):
      @func.func(*(ty.to_ir_type() for ty in self.arg_tys))
      def inner_xla_call(*params):
        return jaxpr_subcomp(c, self.jaxpr, params)
      name = c.symbol_table.insert(inner_xla_call.func_op)
    result, = func.CallOp(inner_xla_call.func_op, list(args)).results
    return result

  def print_prim_app(self, p, args):
    p.emit_line(f"Call{print_args(map(str, args))}")
    with p.indent():
      self.jaxpr.print_lines(p)

XLAVal = Any
XLAVal = ir.Value | tuple[XLAVal]

def jaxpr_subcomp(c: MlirContext, jaxpr: Jaxpr, args: list[ir.Value]) -> list[ir.Value]:
  env: dict[Var, ir.Value] = {}

  def read(x) -> ir.Value:
    if type(x) is Var:
      return env[x]
    else:
      return x.to_hlo_const(c)

  def write(v: Var, val: ir.Value) -> None:
    env[v] = val

  map(write, jaxpr.binders, args)
  for eqn in jaxpr.eqns:
    in_tys = [x.ty for x in eqn.args]
    in_vals = map(read, eqn.args)
    assert all(isinstance(v, ir.Value) for v in in_vals), in_vals
    out_val = eqn.prim.to_hlo(c, *in_vals)
    assert isinstance(out_val, ir.Value)
    write(eqn.binder, out_val)
  return read(jaxpr.result)

def jit(f):
  def wrapped(*args):
    args = map(canonicalize_arg, args)
    jaxpr = trace_to_jaxpr(f, tuple(arg.ty for arg in args))
    return emit(Call(jaxpr), args)
  return wrapped

# === tests ===

@jit
def foo(x):
  return sin(sin(x))

print(sin(1.0))
float_ty = canonicalize_arg(1.0).ty
print(trace_to_jaxpr(lambda x: sin(foo(x)), (float_ty,)))
print(foo(1.0))

def bar(x, y):
  return sin(x + y)

ans, jaxpr = linearize(bar, (1.0, 2.0))
print(ans)
print(jaxpr)

print(backward_pass(jaxpr, 1.0))

print(vmap(bar, np.arange(4.0), np.arange(4.0)))

@jit
def unpack_and_add(xy):
  x, y = xy
  return x + y

print(unpack_and_add((1., 2.)))

print(trace_to_jaxpr(foo, (float_ty,)))

print(foo(1.0))
