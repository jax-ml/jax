# ---
# Copyright 2024 The JAX Authors.
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

from typing import TypeAlias, Union, Sequence, Optional, Any, Callable, TypeVar
from contextlib import contextmanager
from dataclasses import dataclass

from util import *

# === pre-declared aliases to avoid toposorting issues ===

LazyJaxpr : TypeAlias = Any
Jaxpr     : TypeAlias = Any
JaxprType : TypeAlias = Any

# === data ===

class JaxType:
  def __eq__(self, other): raise NotImplementedError(type(self))
  def __str__(self): raise NotImplementedError(type(self))

# Subclass this for new values. These should always be wholly concrete values.
# No tracers or variables inside.
class JaxVal:
  ty : JaxType
  def __str__(self):  raise NotImplementedError(type(self))

@dataclass
class Var:
  ty : JaxType
  def __str__(self):
    s = id(self)%1000 # hack!
    return f'v{s}:{str(self.ty)}'
  def __hash__(self): return id(self)
  def __eq__(self, other): return self is other

Atom : TypeAlias = Var | JaxVal

# === primitives ops ===

class Op:
  @property
  def ir_level(self):          raise NotImplementedError(type(self))
  def __str__(self):           raise NotImplementedError(type(self))

  def result_type(self, *args):  raise NotImplementedError(type(self))

  # MdJax ops only
  def jvp(self, primals:list[Atom], tangents:list[Atom]): raise NotImplementedError(type(self))
  # LoJax ops only
  def impl(self, *args):       raise NotImplementedError(type(self))

class JaxprHof:
  @property
  def ir_level(self):          raise NotImplementedError(type(self))
  def __str__(self):           raise NotImplementedError(type(self))

  # MdJax ops only
  def jvp(self, funargs, primals, tangents): raise NotImplementedError(type(self))
  # LoJax ops only
  def impl(self, *args_and_funargs): raise NotImplementedError(type(self))

class CallableHof:
  @property
  def ir_level(self):          raise NotImplementedError(type(self))
  def __str__(self):           raise NotImplementedError(type(self))


  # MdJax ops only
  def jvp(self, funargs, primals, tangents): raise NotImplementedError(type(self))
  # LoJax ops only
  def impl(self, *args_and_funargs): raise NotImplementedError(type(self))

Primitive : TypeAlias = Op | JaxprHof | CallableHof

@dataclass
class JaxprEqn:
  binder: Var
  op: Op
  args: list[Atom]
  funargs: list[Jaxpr]
  def __str__(self): return PrettyPrinter.to_str(self)
  def pretty_print(self, p):
    p.print_line(f'{self.binder} = {self.op}{arglist_str(self.args)}')
    with p.indent():
      for jaxpr in self.funargs:
        jaxpr.pretty_print(p)

@dataclass
class Jaxpr:
  binders: list[Var]
  eqns: list[JaxprEqn]
  result: Atom

  @property
  def ty(self) -> JaxprType:
    return JaxprType([b.ty for b in self.binders], self.result.ty)

  def __str__(self): return PrettyPrinter.to_str(self)
  def pretty_print(self, p):
    p.print_line(f'{arglist_str(self.binders)} =>')
    with p.indent():
      for eqn in self.eqns:
        eqn.pretty_print(p)
      p.print_line(f'return {self.result}')

@dataclass
class JaxprType:
  arg_types:  list[JaxType]
  result_type : JaxType
  def __str__(self): return f'{arglist_str(self.arg_types)} -> {self.result_type}'

# An `OpStream` is a function that takes an Emitter object explicitly
OpStream : TypeAlias = Callable # [[Emitter, ...], ...]

# === program transformations ===

TraceVal : TypeAlias = Any
FunArg : TypeAlias = Jaxpr | Callable

class Emitter:
  def emit_primitive(self, op:Op, result_type: JaxType, args:list[TraceVal], funargs:list[FunArg]) -> TraceVal:
    raise NotImplementedError(type(self))

class Tracer:
  ty : JaxType
  # these methods defer to the type
  def __getitem__(self, ix):
    return self.ty._getitem(self, ix)

TraceVal : TypeAlias = JaxVal | Tracer

# === Evaluation with concrete values ===

class EvalEmitter(Emitter):
  def emit_primitive(self, p:Primitive, result_ty, args:tuple[JaxVal], fun_args):
    return p.impl(result_ty, *(tuple(args) + tuple(fun_args)))

eval_emitter = EvalEmitter()

# No need for an EvalTracer because the only data we track is the value and `JaxVal` itself can do that just fine.

# === Builder ===

def new_var(ty):
  # keeping some indirection to make room for an explicit counter or something
  return Var(ty)

class BuilderEmitter(Emitter):
  def __init__(self):
    self.eqns = []

  def new_tracer(self, ty):
    return BuilderTracer(new_var(ty), ty)

  def traceval_to_atom(self, arg:TraceVal) -> TraceVal:
    if isinstance(arg, BuilderTracer):
      return arg.var
    elif isinstance(arg, JaxVal):
      return arg
    else:
      raise Exception

  def add_eqn(self, result_ty, p, args, funargs):
    v = new_var(result_ty)
    arg_atoms = [self.traceval_to_atom(arg) for arg in args]
    result = self.new_tracer(result_ty)
    self.eqns.append(JaxprEqn(result.var, p, arg_atoms, funargs))
    return result

  def emit_primitive(self, p:Primitive, result_ty, args, funargs):
    arg_tys = tuple(arg.ty for arg in args)
    return self.add_eqn(result_ty, p, args, funargs)

@dataclass
class BuilderTracer(Tracer):
  var : Var
  ty  : JaxType

def materialize_jaxpr(stream:OpStream, arg_types:list[JaxType]) -> Jaxpr:
  builder = BuilderEmitter()
  args = [builder.new_tracer(ty) for ty in arg_types]
  binders = [arg.var for arg in args]
  stream_result = stream(builder, *args)
  atom_result = builder.traceval_to_atom(canonicalize_pyval(stream_result))
  return Jaxpr(binders, builder.eqns, atom_result)

def trace_to_jaxpr(f, arg_types:list[JaxType]) -> Jaxpr:
  return materialize_jaxpr(WithExplicitEmitter(f), arg_types)

def apply_jaxpr(emitter, jaxpr, vals):
  env = {b : arg for b in jaxpr.binders}
  def interpret_atom(x):
    if isinstance(x, Var):
      return env[x]
    elif isinstance(x, JaxVal):
      return x
    else:
      raise Exception(x, type(x))
  for eqn in jaxpr.eqns:
    args = [interpret_atom(x) for x in eqn.args]
    env[eqn.binder] = emitter.emit_primitive(eqn.op, eqn.binder.ty, args, eqn.funargs)
  return interpret_atom(jaxpr.result)

# === IR levels ===

# Higher-level IRs are a strict superset of the IRs beneath them

FrontendIR = 2  # `jvp`, pytrees
BackendIR  = 1

FrontendAtom : TypeAlias = Atom
BackendAtom  : TypeAlias = Atom

FrontendVar : TypeAlias = Var
BackendVar  : TypeAlias = Var

# === Lowering ===

@dataclass
class FrontendLoweringEmitter(Emitter):
  parent_emitter : Emitter

  def __init__(self, parent):
    self.parent_emitter = parent

  def emit_primitive(self, p:Primitive, result_ty, args, funargs):
    if hasattr(p, "lower_frontend"):
      assert False
      # return p.lower_frontend(self, result_ty, *args)
    else:
      args_low = [self.lower_to_rep(arg) for arg in args]
      if isinstance(p, CallableHof):
        assert False
      else:
        funargs_low = [self.lower_jaxpr(f) for f in funargs]
      result_ty_low = self.lower_type(result_ty)
      result_rep = self.parent_emitter.emit_primitive(p, result_ty_low, args_low, funargs_low)
      return self.lift_rep(result_ty, result_rep)

  def lower_jaxpr(self, jaxpr_high:Jaxpr) -> Jaxpr:
    def f_lowered(emitter, *arg_reps):
      lowering_emitter = FrontendLoweringEmitter(emitter)
      args_high = [self.lift_rep(ty, rep) for ty, rep in zip(jaxpr_high.ty.arg_types, arg_reps)]
      result_high = apply_jaxpr(lowering_emitter, jaxpr_high, args_high)
      return self.lower_to_rep(result_high)

    lowered_arg_types = [self.lower_type(arg_ty) for arg_ty in jaxpr_high.ty.arg_types]
    lowered_jaxpr = materialize_jaxpr(f_lowered, lowered_arg_types)
    return lowered_jaxpr

  def lower_to_rep(self, x:TraceVal) -> TraceVal:
    # Lowering transforms an entire programs with no free variables. So
    # all FrontentLoweringTracers should be ours
    if isinstance(x, FrontendLoweringTracer):
      return x.rep
    elif isinstance(x, JaxVal):
      if hasattr(x, "lower_frontend"):
        return x.lower_frontend()
      else:
        return x
    else:
      raise Exception(x, type(x))

  def lift_rep(self, ty, rep:TraceVal) -> TraceVal:
    if isinstance(rep, Tracer):
      return FrontendLoweringTracer(ty, rep)
    elif isinstance(rep, JaxVal):
      if hasattr(rep, "lift_frontend"):
        return rep.lift_frontend(ty)
      else:
        return rep
    else:
      raise Exception

  def lower_type(self, ty:JaxType) -> JaxType:
    if hasattr(ty, "lower_frontend"):
      return ty.lower_frontend()
    else:
      return ty

@dataclass
class FrontendLoweringTracer(Tracer):
  ty : JaxVal
  rep : TraceVal

# === embedding ===

# We keep a "current emitter" as a globally-readable context. This is purely to
# reduce clutter in user-facing code. Internally we pass around emitters
# explicitly so it's easier to follow the flow of data.
@dataclass
class CurrentEmitter:
  emitter : Emitter

@contextmanager
def set_current_emitter(emitter):
  prev = current_emitter.emitter
  current_emitter.emitter = emitter
  try:
    yield
  finally:
    current_emitter.emitter = prev

def top_level_emitter():
  return FrontendLoweringEmitter(eval_emitter)

current_emitter = CurrentEmitter(top_level_emitter())

def emit_primitive(p, args, funargs=()):
  emitter = current_emitter.emitter
  args_canonical = [canonicalize_pyval(arg) for arg in args]
  arg_tys = [arg.ty for arg in args_canonical]
  if isinstance(p, CallableHof):
    result_ty = None
  else:
    fun_tys = [f.ty for f in funargs]
    result_ty = p.result_type(*(tuple(arg_tys) + tuple(fun_tys)))
  return emitter.emit_primitive(p, result_ty, args_canonical, funargs)

# This turns a function that reads the implicit "current_emitter" context into
# one that takes the emitter explicitly, conforming to the `OpStream` API
@dataclass
class WithExplicitEmitter:
  f : Callable
  def __call__(self, emitter, *args):
    with set_current_emitter(emitter):
      return self.f(*args)

# `emit` requires each argument to be a `TraceVal`, which is either a `JaxVal`
# or a `Tracer`. A PyVal could be a tuples of tracers, or a Python float
# representing a rank-0 array. We canonicalize these to a `TraceVal` before
# calling `emit`.

PyVal : TypeAlias = Any
pyval_canonicalizers = {}

def register_canonicalizer(t, f):
  pyval_canonicalizers[t] = f

# may use current emitter, for example to build a FancyTuple from a python
# tuple.
def canonicalize_pyval(x: PyVal) -> TraceVal:
  if isinstance(x, JaxVal):
    return x
  elif isinstance(x, Tracer):
    return x
  elif type(x) in pyval_canonicalizers:
    return pyval_canonicalizers[type(x)](x)
  else:
    raise TypeError(f'Unrecognized JAX type: {type(x)}')

# === jvp ===

class JVPEmitter(Emitter):
  def __init__(self, parent):
    self.parent = parent

  def emit(self, p:Primitive, args, funargs):
    with set_emitter(self.parent):
      return p.jvp(primals, tangents, funargs)

class JVP(CallableHof):
  pass

