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
from typing import NamedTuple, Any, Callable

from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func
from jax.extend.mlir.dialects import stablehlo as hlo
from jax._src import xla_bridge as xb

# === utils ===

def map(f, *xs):
  return list(builtins.map(f, *xs))

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

class Val:
  ty : Ty

class Primitive:
  arg_tys : tuple[Ty]
  result_ty : Ty

  # type checking goes here
  def __init__(self, *arg_tys, **params):
    raise NotImplementedError(f"{type(self).__name__}.__init__")

  # user-facing call with raw args
  def call(self, *args, **kwargs):
    raise NotImplementedError(f"{type(self).__name__}.call")

  def to_hlo(self, c, *hlo_args):
    raise NotImplementedError(f"{type(self).__name__}.to_hlo")

  def __str__(self):
    return type(self).__name__

class Var:
  ty : Ty
  def __init__(self, ty):
    self.ty = ty

  def __str__(self):
    return f"%{id(self) % 1000}: {self.ty}"

class Eqn:
  binder : Var
  prim   : Primitive
  args   : tuple[Val|Var]
  def __init__(self, binder, prim, args):
    self.binder = binder
    self.prim = prim
    self.args = args

  def print_lines(self, p):
    # TODO: handle multiline primitives by checking if they implement multiline print
    p.emit_line(f"{self.binder} = {self.prim}{print_args(map(str, self.args))}")

  __str__ = str_from_print_lines

class Jaxpr:
  binders : tuple[Var]
  eqns : tuple[Eqn]
  result : Val|Var
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

class Trace:
  def process_primitive(self, prim:Primitive, args):
    raise NotImplementedError(f"{type(self).__name__}.process_primitive")

class EvalTrace(Trace):
  def process_primitive(self, prim, args):
    return compile_prim(prim).execute(*args)

@dataclass
class TraceCtx:
  trace : Trace

def emit(prim:Primitive, args):
  trace = trace_ctx.trace
  return trace.process_primitive(prim, args)

def canonicalize_arg(arg: Any) -> Val | Tracer:
  if isinstance(arg, (Val, Tracer)):
    return arg
  elif isinstance(arg, np.ndarray):
    return ArrayVal(arg)
  elif isinstance(arg, (float, np.float64, np.float32)):
    return ArrayVal(np.array(arg))
  else:
    raise TypeError("Unrecognized type: {arg}")

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

class JaxprTrace(Trace):
  binders : list[Var]
  eqns : list[Eqn]
  def __init__(self):
    self.binders = []
    self.eqns = []

  def process_primitive(self, prim, args):
    # TODO: handle literals and other traces' tracers too
    arg_atoms = [arg.var for arg in args]
    binder = Var(prim.result_ty)
    self.eqns.append(Eqn(binder, prim, arg_atoms))
    return JaxprTracer(binder)

  def new_arg(self, ty:Ty) -> Tracer:
    v = Var(ty)
    self.binders.append(v)
    return JaxprTracer(v)

  def finalize_jaxpr(self, result:Val|Tracer):
    return Jaxpr(tuple(self.binders), tuple(self.eqns), result.var)

class JaxprTracer(Tracer):
  var: Var
  def __init__(self, v:Var):
    self.var = v
    self.ty = v.ty

def trace_to_jaxpr(f:Callable, arg_tys:tuple[Ty]) -> Jaxpr:
  trace = JaxprTrace()
  args = tuple(trace.new_arg(t) for t in arg_tys)
  with set_current_trace(trace):
    result = f(*args)
  # TODO: lift result if needed
  return trace.finalize_jaxpr(result)


# === types ===

class ArrayTy(Ty):
  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def to_ir_type(self):
    return ir.RankedTensorType.get(self.shape, _mlir_dtype(self.dtype))

  def from_runtime_buf(self, buf):
    val = ArrayVal(np.asarray(buf))
    assert val.ty == self
    return val

  def __eq__(self, other):
    return self.shape == other.shape and self.dtype == other.dtype

  def __str__(self):
    return f"{self.dtype}{list(self.shape)}"

class ArrayVal(Val):
  def __init__(self, x):
    assert isinstance(x, np.ndarray)
    self.ty = ArrayTy(x.shape, x.dtype)
    self.val = x

  def to_runtime_buf(self):
    return xb.get_backend(None).buffer_from_pyval(self.val)

  def __str__(self):
    return str(self.val)

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

  @classmethod
  def call(cls, x):
    x = canonicalize_arg(x)
    return emit(cls(x.ty), (x,))

  def to_hlo(self, _, x):
    return hlo.sine(x)

sin = Sin.call

# === tests ===

print(sin(1.0))
float_ty = canonicalize_arg(1.0).ty
print(trace_to_jaxpr(lambda x: sin(sin(x)), (float_ty,)))
