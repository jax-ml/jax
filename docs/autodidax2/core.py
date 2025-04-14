# ---
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
#
# ---

from __future__ import annotations

from typing import Any
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np

from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func
from jax.extend.mlir.dialects import stablehlo as hlo
from jax._src import xla_bridge as xb

# === IR data types ===

# Base class for JAX-level types
class Ty:
  def __eq__(self, other): raise NotImplementedError(type(self))
  def __str__(self):     raise NotImplementedError(type(self))

  def tangent_ty(self): raise NotImplementedError(type(self))

  # dispatch path for "dunder" methods on values
  def _add_(self, x: Any, y: Any): raise NotImplementedError(type(self))
  def _mul_(self, x: Any, y: Any): raise NotImplementedError(type(self))

  def __repr__(self): return str(self)

# Base class for JAX-level op.
# See note [op_representation]
class Op:
  def eval(self, ty: Ty, *args: RuntimeVal): raise NotImplementedError(type(self))
  def check(self, ty: Ty, *arg_tys, Ty):     raise NotImplementedError(type(self))

class TraceTag: pass

class Tracer:
  ty: Ty

  def __str__(self): raise(NotImplementedError(type(self)))

  # The "dunder" methods dispatch to methods defined by the type
  def __add__(self, other): return self.ty._add_(self, other)
  def __mul__(self, other): return self.ty._mul_(self, other)
  def __radd__(self, other): return self.ty._add_(other, self)
  def __rmul__(self, other): return self.ty._mul_(other, self)
  def __getitem__(self, idx): return self.ty._getitem_(self, idx)
  def __setitem__(self, idx, val): return self.ty._setitem_(self, idx, val)

# Valid runtime value (passes is_valid_runtime_val(x) check).
# Does *not* include Python scalars.
RuntimeVal = Any  # concrete values *except* python scalars
PythonLiteral = Any  # a float, int, or bool
InterpreterVal = Tracer | RuntimeVal
Val = Tracer | RuntimeVal | PythonLiteral
# The type of data this interpreter works with. e.g. JVPTracer for JVPInterpreter.
ThisInterpreterVal = Any

class Interpreter:
  def lift(self, val: InterpreterVal) -> ThisInterpreterVal:
    raise NotImplementedError(type(self))

  def interpret_op(self, op:Op, result_ty: Ty | Tuple[Ty], args: tuple(ThisInterpreterVal)) -> ThisInterpreterVal:
    raise NotImplementedError(type(self))

# === pretty printer ===

# Pretty printer object for keeping track of indentation.
class Printer:
  lines: list[(int, str)] # Lines emitted so far, along with their indentation level
  cur_indent: int         # The current indentation level of the printer

  def __init__(self):
    self.lines = []
    self.cur_indent = 0

  def emit_line(self, s:str):
    self.lines.append((self.cur_indent, s))

  def render(self) -> str:
    return '\n'.join(['  ' * indent + s for indent, s in self.lines]) + '\n'

  @contextmanager
  def indent(self):
    prev = self.cur_indent
    self.cur_indent = prev + 1
    yield
    self.cur_indent = prev

# === "jaxpr" IR ===

class Atom:
  ty: Ty

class Literal(Atom):
  def __init__(self, val):
    assert isinstance(val, np.ndarray) and val.shape == ()
    self.val = val
    self.ty = typeof(val)

  def __repr__(self): return self.__str__()
  def __str__(self): return f"{self.val}:{self.ty}"

class Var(Atom):
  def __init__(self, name: str , ty: Ty):
    self.name = name
    self.ty = ty

  def __repr__(self): return self.__str__()
  def __str__(self): return f"{self.name}:{self.ty}"

class Eqn:
  def __init__(self, binder: Var | tuple[Var], prim: Op, args: tuple[Atom]):
    assert isinstance(binder, (Var, tuple))
    assert isinstance(prim, Op)
    assert isinstance(args, tuple)
    assert all(isinstance(x, Atom) for x in args)

    self.binder = binder
    self.prim   = prim
    self.args   = args

  def pretty_print(self, p:Printer):
    if hasattr(self.prim, "pretty_print_prim_app"):
      # Handle multi-line primitives specially
      p.emit_line(f"{self.binder} =")
      with p.indent():
        self.prim.pretty_print_prim_app(p, self.args)
    else:
      p.emit_line(f"{self.binder} = {self.prim}{tuple(self.args)}")

  def __repr__(self): return self.__str__()
  def __str__(self):
    p = Printer()
    self.pretty_print(p)
    return p.render()

# We call an IR function a "Jaxpr", for "JAX expression"
@dataclass
class Jaxpr:
  binders : list[Var]  # The function's formal parameters (arguments)
  eqns: list[Eqn]      # The body of the function, a list of instructions/equations
  result: Atom         # The function's return value

  def __init__(self, binders, eqns, result):
    assert all(isinstance(b, Var) for b in binders)
    assert all(isinstance(eqn, Eqn) for eqn in eqns)
    assert isinstance(result, Atom)
    self.binders = binders
    self.eqns = eqns
    self.result = result

  def pretty_print(self, p:Printer):
    p.emit_line(f"{str(tuple(b for b in self.binders))} =>")
    with p.indent():
      for eqn in self.eqns:
        eqn.pretty_print(p)
      p.emit_line(f"return {self.result}")

  def __str__(self):
    p = Printer()
    self.pretty_print(p)
    return p.render()


@dataclass(frozen=True)
class JaxprName:
  name_stem : str
  name_int  : int

  def __repr__(self): return self.__str__()
  def __str__(self): return f"{self.name_stem}_{self.name_int}"

# === runtime values ===

class RuntimeValueHandler:
  def typeof(self, val: Any) -> Ty | None:
    raise NotImplementedError(type(self))

  def to_runtime_val(self, val:Any) -> RuntimeVal:
    raise NotImplementedError(type(self))

  def is_valid_runtime_val(self, val:Any) -> bool:
    raise NotImplementedError(type(self))

class PythonScalarRuntimeValueHandler:
  def typeof(self, x): return typeof(self.to_runtime_val(x))
  def to_runtime_val(self, x): return np.array(x)
  def is_valid_runtime_val(self, _): return False

class ArrayRuntimeValueHandler:
  def typeof(self, x): return ArrayTy(x.shape, to_jax_dtype(x.dtype))
  def to_runtime_val(self, x): return x
  def is_valid_runtime_val(self, _): return True

runtime_types : dict[type, RuntimeValueHandler] = {
  float : PythonScalarRuntimeValueHandler(),
  int   : PythonScalarRuntimeValueHandler(),
  bool  : PythonScalarRuntimeValueHandler(),
  np.float32 : ArrayRuntimeValueHandler(),
  np.float64 : ArrayRuntimeValueHandler(),
  np.ndarray : ArrayRuntimeValueHandler()
}

def register_runtime_type(python_type: type, handler: RuntimeValueHandler):
  assert isinstance(python_type, type)
  assert isinstance(handler, RuntimeValueHandler)
  runtime_times[python_type] = handler

def get_runtime_value_handler(x: Any) -> RuntimeValueHandler:
  handler = runtime_types.get(type(x))
  if handler is None:
    raise ValueError(f"Not a valid JAX type: {type(x)}")
  else:
     return handler

def to_runtime_val(x: Any) -> InterpreterVal:
  if isinstance(x, Tracer):
    return x
  else:
    return get_runtime_value_handler(x).to_runtime_val(x)

def typeof(x: Any) -> Ty:
  if isinstance(x, Tracer):
    return x.ty
  else:
    return get_runtime_value_handler(x).typeof(x)

def is_valid_runtime_val(x: Any) -> bool:
  if isinstance(x, Tracer):
    return True
  else:
    return get_runtime_value_handler(x).is_valid_runtime_val(x)

def is_python_scalar(x:Any) -> bool:
  return isinstance(x, (int, float, bool))

def cast_python_scalar(x, ty):
  return np.array(x, dtype=ty.dtype.to_numpy_dtype())

def cast_if_scalar(x, ty):
  if is_python_scalar(x):
    return cast_python_scalar(x, ty)

# === dtypes ===

class DType:
  def __getitem__(self, shape): return ArrayTy(shape, self)
  def to_numpy_dtype(self): raise NotImplementedError(type(self))

class Float32(DType):
  def to_numpy_dtype(self): return np.dtype(np.float32)
  def __str__(self): return "f32"

class Float64(DType):
  def to_numpy_dtype(self): return np.dtype(np.float64)
  def __str__(self): return "f64"

class Int32(DType):
  def to_numpy_dtype(self): return np.dtype(np.int32)
  def __str__(self): return "i32"

class Int64(DType):
  def to_numpy_dtype(self): return np.dtype(np.int64)
  def __str__(self): return "i64"

f32 = Float32()
f64 = Float64()
i32 = Int32()
i64 = Int64()

numpy_to_jax_dtype_mapping = {
  np.dtype(np.float32) : f32,
  np.dtype(np.float64) : f64,
  np.dtype(np.int32)   : i32,
  np.dtype(np.int64)   : i64}

def to_jax_dtype(numpy_dtype):
  jax_dtype = numpy_to_jax_dtype_mapping.get(np.dtype(numpy_dtype))
  if jax_dtype is None:
    raise ValueError(f"Unrecognized numpy dtype: {numpy_dtype}")
  else:
    return jax_dtype

# === array type ===

@dataclass(frozen=True)
class ArrayTy(Ty):
  shape : tuple[int]
  dtype: DType

  def tangent_ty(self):
    return ArrayTy(self.shape, self.dtype)  # TODO: tangent types of ints

  def zeros_like(self):
    return np.zeros(self.shape, self.dtype.to_numpy_dtype())

  def __str__(self):
    return f"{self.dtype}[{",".join(self.shape)}]"

  # dispatch path for "dunder" methods on values
  def _add_(self, x, y): return add(x, y)
  def _mul_(self, x, y): return mul(x, y)

# === jaxpr interpreter ===

def interpret_jaxpr(jaxpr:Jaxpr, args):
  assert tuple(b.ty for b in jaxpr.binders) == tuple(map(typeof, args))

  def interpret_atom(atom):
    if isinstance(atom, Var):
      return env[atom]
    elif isinstance(atom, Literal):
      return atom.val
    else:
      breakpoint()
      raise Exception(f"Not an atom: {atom}")

  env = dict(zip(jaxpr.binders, args))
  for eqn in jaxpr.eqns:
    eqn_args = tuple(interpret_atom(arg) for arg in eqn.args)
    ans = emit_op(eqn.prim, eqn.binder.ty, eqn_args)
    env[eqn.binder] = ans

  return interpret_atom(jaxpr.result)

# === evaluating interpreter ===

class EvalInterpreter(Interpreter):
  def lift(self, x):
    return x

  def interpret_op(self, op, ty, args):
    return op.eval(ty, *args)

  def interpret_call(self, fun, args):
    return interpret_app(fun.jaxpr, args)

# === staging interpreter ===

class StagingTracer(Tracer):
  def __init__(self, val, interpreter):
    assert isinstance(val, Atom)
    self.val = val
    self.ty = val.ty
    self.interpreter = interpreter

  def __str__(self):
    raise NotImplementedError(type(self))

class StagingInterpreter(Interpreter):
  def __init__(self):
    self.equations = []
    self.name_counter = 0

  def fresh_var(self, ty):
    self.name_counter += 1
    return Var(f"v{self.name_counter}", ty)

  def lift(self, x):
    if isinstance(x, StagingTracer) and x.interpreter is self:
      return x
    elif isinstance(x, np.ndarray) and x.shape == ():
      return StagingTracer(Literal(x), self)
    else:
      assert is_valid_runtime_val(x), x
      assert False

  def interpret_op(self, op, ty, args):
    args = tuple(x.val for x in args)
    if isinstance(ty, tuple):
      binders = tuple(map(self.fresh_var, ty))
      self.equations.append(Eqn(binders, op, args))
      return tuple(StagingTracer(b, self) for b in binders)
    else:
      binder = self.fresh_var(ty)
      self.equations.append(Eqn(binder, op, args))
      return StagingTracer(binder, self)

def make_jaxpr(f, arg_tys):
  interpreter = StagingInterpreter()
  arg_vars = tuple(interpreter.fresh_var(ty) for ty in arg_tys)
  with set_interpreter(interpreter):
    result = f(*(StagingTracer(v, interpreter) for v in arg_vars))
  result = interpreter.lift(result)
  return Jaxpr(arg_vars, interpreter.equations, result.val)

# === tracing context ===

# The current interpreter is initially the evaluating interpreter.
current_interpreter = EvalInterpreter()

# A context manager for temporarily changing the current interpreter
@contextmanager
def set_interpreter(new_interpreter):
  global current_interpreter
  prev_interpreter = current_interpreter
  try:
    current_interpreter = new_interpreter
    yield
  finally:
    current_interpreter = prev_interpreter

def emit_op(op: Op, result_ty: Ty, args: list[InterpreterVal]):
  args = tuple(current_interpreter.lift(arg) for arg in args)
  return current_interpreter.interpret_op(op, result_ty, args)

# === jvp ===

class JVPInterpreter:
  def __init__(self, parent, tag):
    self.parent = parent
    self.tag = tag

  def interpret_op(self, op, result_ty, args):
    primals = tuple(arg.primal for arg in args)
    tangents = tuple(arg.tangent for arg in args)
    with set_interpreter(self.parent):
      p, t = op.jvp(result_ty, primals, tangents)
    return self.new_tracer(p, t)

  def lift(self, val):
    if isinstance(val, JVPTracer) and val.tag is self.tag:
      return val
    else:
      ty = typeof(val)
      return JVPTracer(self.tag, val, ty.tangent_ty().zeros_like())

  def new_tracer(self, primal, tangent):
    return JVPTracer(self.tag, primal, tangent)

class JVPTracer(Tracer):
  def __init__(self, tag, primal, tangent):
    self.tag = tag
    self.primal = primal
    self.tangent = tangent
    self.ty = typeof(primal)

def jvp(f, primals, tangents):
  assert tuple(map(typeof, primals)) == tuple(map(typeof, tangents))
  interpreter = JVPInterpreter(current_interpreter, TraceTag())
  arg_tracers = tuple(interpreter.new_tracer(p, t) for p, t in zip(primals, tangents))
  with set_interpreter(interpreter):
    result = f(*arg_tracers)
  result = interpreter.lift(result)
  return result.primal, result.tangent

# === jit ===

@dataclass(frozen=True)
class NamedJaxpr:
  jaxpr : Jaxpr
  name  : JaxprName

jaxpr_names = {}

def new_named_jaxpr(jaxpr, name_stem):
  int_part = jaxpr_names.get(name_stem, 0)
  jaxpr_names[name_stem] = int_part + 1
  jaxpr_name = JaxprName(name_stem, int_part)
  return NamedJaxpr(jaxpr, int_part)

class Call(Op):
  def __init__(self, fun: NamedJaxpr):
    self.fun = fun

  def eval(self, _, *args: RuntimeVal):
    return interpret_jaxpr(self.fun.jaxpr, args)

  def check(self, ty: Ty, *arg_tys, Ty):
    assert False

class JittedFunction:
  f : Callable
  cache : Dict[JitCacheKey, Jaxpr]

  def __init__(self, f):
    self.f = f
    self.cache = {}

  def __call__(self, *args):
    arg_tys = tuple(map(typeof, args))
    if arg_tys in self.cache:
      named_jaxpr = self.cache[arg_tys]
    else:
      jaxpr = make_jaxpr(self.f, arg_tys)
      named_jaxpr = new_named_jaxpr(jaxpr, self.f.__name__)
      self.cache[arg_tys] = named_jaxpr

    result_ty = named_jaxpr.jaxpr.result.ty
    return emit_op(Call(named_jaxpr), result_ty, args)

jit = JittedFunction


# === Core array ops needed for __add__ and friends ===

def infer_binop(x:InterpreterVal, y:InterpreterVal):
  # TODO: broadcasting
  x_val = to_runtime_val(x)
  y_val = to_runtime_val(y)
  if is_python_scalar(x):
    x_val = cast_python_scalar(x_val, typeof(y_val))
  elif is_python_scalar(y):
    y_val = cast_python_scalar(y_val, typeof(x_val))
  return typeof(x_val), x_val, y_val

class Add(Op):
  def eval(self, _, x, y): return x + y
  def jvp(self, _, primals, tangents):
    x, y = primals
    xt, yt = tangents
    return add(x, y), add(xt, yt)

  def __str__(self): return "add"

def add(x, y):
  result_ty, x, y = infer_binop(x, y)
  return emit_op(Add(), result_ty, (x, y))

class Mul(Op):
  def eval(self, _, x, y): return x * y
  def __str__(self): return "mul"

def mul(x, y):
  result_ty, x, y = infer_binop(x, y)
  return emit_op(Mul(), result_ty, (x, y))
