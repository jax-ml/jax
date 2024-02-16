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

# === core AST types ===

# Subclass these for new types and operations. See Note [defining_new_types_and_ops]
class JaxType:
  def __eq__(self, other): raise NotImplementedError(type(self))
  def __str__(self):       raise NotImplementedError(type(self))
  @staticmethod
  def var_to_tracer():     raise NotImplementedError(type(self))
  def tangent_type(self):  raise NotImplementedError(type(self))
  # add_tangents and instantiate_zeros only need to be defined for vector spaces
  def add_tangents(self, x, y):   raise NotImplementedError(type(self))
  def instantiate_zeros(self, x): raise NotImplementedError(type(self))
  # to_repval, from_repval, rep_type describe how to convert to/from a flat list of arrays
  def to_repval(self, x):   raise NotImplementedError(type(self))
  def from_repval(self, x): raise NotImplementedError(type(self))
  @property
  def rep_types(self):      raise NotImplementedError(type(self))

class JaxVal:
  @property
  def ty(self) -> JaxType:     raise NotImplementedError(type(self))
  def to_atom(self):           raise NotImplementedError(type(self))
  def eval_atom(self, _):      raise NotImplementedError(type(self))
  def eval_tangent(self, _):   raise NotImplementedError(type(self))
  def push_cotangent(self, _): raise NotImplementedError(type(self))
  def free_vars(self):         raise NotImplementedError(type(self))
  def __str__(self):           raise NotImplementedError(type(self))

class Primitive:
  def impl(self, env, *args):                      raise NotImplementedError(type(self))
  def eval_type(self, *args):                      raise NotImplementedError(type(self))
  def linearize_rule(self, ty, primals, tangents): raise NotImplementedError(type(self))
  def transpose_rule(self, ct, *args):             raise NotImplementedError(type(self))
  def __str__(self):                               raise NotImplementedError(type(self))
@dataclass
class Var:
  ty: JaxType
  s: str # for pretty-printing only
  def free_vars(self): return {self}
  def pretty_print(self, p):
    p.emit(f'{self.binder} = {self.primitive}{arglist_str(self.args)}')
  def as_tracer(self):
    return self.ty.var_to_tracer(self)

  def __str__(self):
    return f'{self.s}:{self.ty}'

  def __hash__(self): return id(self)
  def __eq__(self, other): return self is other

  def eval_atom(self, env):
    try:
      return env[self]
    except KeyError:
      # The variable should be defined in the trace stack. We could check.
      return self.as_tracer()

  def eval_tangent(self, env):
    try:
      return env[self]
    except KeyError:
      return SymbolicZero(self.ty.tangent_type())

  def push_cotangent(self, cotangents, ct):
    accum_cotangent(cotangents, self, ct)

@dataclass
class Tracer:
  var : Var
  @property
  def ty(self): return self.var.ty
  def to_atom(self): return self.var
  def __str__(self): return f'Tracer({self.ty})'

Atom        : TypeAlias = Union[JaxVal, Var]       # used in jaxprs. not tracers
# TODO: should we drop this distinction and just use LoosePyVal everywhere?
StrictPyVal : TypeAlias = Union[JaxVal]            # jax-produced pyvals
LoosePyVal  : TypeAlias = Union[StrictPyVal, Any]  # user-supplied pyvals

@dataclass
class JaxprEqn:
  binder: Var
  primitive: Primitive
  args: list[Atom]

  def __str__(self): return PrettyPrinter.to_str(self)
  def pretty_print(self, p):
    p.emit(f'{self.binder} = {self.primitive}{arglist_str(self.args)}')
    if isinstance(self.primitive, SecondOrderPrimitive):
      with p.indent():
        for label, jaxpr in self.primitive.jaxprs.items():
          p.emit(f"{label} = ")
          with p.indent():
            jaxpr.pretty_print(p)


@dataclass
class Jaxpr:
  binders: list[Var]
  eqns: list[JaxprEqn]
  result: Atom

  @property
  def ty(self):
    return JaxprType([v.ty for v in self.binders], self.result.ty)

  def __str__(self): return PrettyPrinter.to_str(self)
  def pretty_print(self, p):
    p.emit(f'{arglist_str(self.binders)} =>')
    with p.indent():
      for eqn in self.eqns:
        eqn.pretty_print(p)
      p.emit(f'return {self.result}')

@dataclass
class JaxprType:
  arg_types:  list[JaxType]
  result_type : JaxType
  def __str__(self): return f'{arglist_str(self.arg_types)} -> {self.result_type}'

class SecondOrderPrimitive(Primitive):
  @property
  def jaxprs(self): raise NotImplementedError(type(self))

# === tracing ===

Trace : TypeAlias = list[JaxprEqn]
@dataclass
class TraceCtx:
  stack : list[Trace]
  var_count : int  # TODO: make some effort to recycle names

  @contextmanager
  def new_trace(self):
    self.stack.append(Trace([]))
    try:
      yield self.stack[-1]
    finally:
      self.stack.pop()

  @property
  def cur_trace(self) -> Optional[Trace]:
    if self.stack:
      return self.stack[-1]
    else:
      return None

  def new_var(self, ty:JaxType):
    v = Var(ty, f'v{self.var_count}')
    self.var_count += 1
    return v

  def __str__(self): return PrettyPrinter.to_str(self)
  def pretty_print(self, p):
    original_indent = p.cur_indent
    p.emit("Tracing Context:")
    for trace in self.traces:
      for eqn in traces: eqn.pretty_print(p)
      p.cur_indent += '  '
    p.cur_indent = original_indent

trace_ctx = TraceCtx([], 0)

def apply_primitive(prim: Primitive, args_raw:list[LoosePyVal]) -> StrictPyVal:
  args = map(canonicalize_pyval, args_raw)
  return apply_primitive_strict(prim, args)

def apply_primitive_strict(prim: Primitive, args:list[StrictPyVal]) -> StrictPyVal:
  trace = trace_ctx.cur_trace
  result_ty = prim.eval_type(*[arg.ty for arg in args])
  if trace is None:
    return prim.impl({}, *args)
  else:
    arg_atoms = [arg.to_atom() for arg in args]
    v = trace_ctx.new_var(result_ty)
    trace.append(JaxprEqn(v, prim, arg_atoms))
    return v.as_tracer()

def trace_to_jaxpr(f: Callable[[StrictPyVal], LoosePyVal], in_tys: list[JaxType]) -> Jaxpr:
  with trace_ctx.new_trace() as trace:
    binders = map(trace_ctx.new_var, in_tys)
    arg_pyvals = [v.as_tracer() for v in binders]
    result_pyval = f(*arg_pyvals)
    result_atom = canonicalize_pyval(result_pyval).to_atom()
    return Jaxpr(binders, trace, result_atom)

# === loose-to-strict conversion ===

pyval_canonicalizers = {}

def register_canonicalizer(t, f):
  pyval_canonicalizers[t] = f

def canonicalize_pyval(x: LoosePyVal) -> StrictPyVal:
  if isinstance(x, Tracer):
    return x
  elif isinstance(x, JaxVal):
    return x
  elif type(x) in pyval_canonicalizers:
    return pyval_canonicalizers[type(x)](x)
  else:
    raise TypeError(f'Unrecognized type: {type(x)}')

# === evaluation ===

Env : TypeAlias = dict[Var, StrictPyVal]

def eval_jaxpr(outer_env: Env, jaxpr: Jaxpr, top_args_loose: list[LoosePyVal]):
  assert trace_ctx.cur_trace is None
  env = outer_env.copy()  # could avoid this with a list of dicts
  top_args = map(canonicalize_pyval, top_args_loose)
  assert jaxpr.ty.arg_types == [arg.ty for arg in top_args]
  for v, val in zip(jaxpr.binders, top_args):
    env[v] = val
  for eqn in jaxpr.eqns:
    args = [arg.eval_atom(env) for arg in eqn.args]
    env[eqn.binder] = eqn.primitive.impl(env, *args)
  return jaxpr.result.eval_atom(env)

# === type checking ===

def typecheck_jaxpr(jaxpr: Jaxpr):
  for eqn in jaxpr.eqns:
    result_type = eqn.primitive.eval_type(*[x.ty for x in eqn.args])
    if not result_type == eqn.binder.ty: raise TypeError

# === linearize ===

@dataclass
class SymbolicZero:
  ty : JaxType
  # this is for the "with symbolic zeros" version,
  # which would be written `to_atom: LinearPyVal -> LinearAtom`
  def to_atom(self): return self

LinearPyVal = Union[StrictPyVal, SymbolicZero]  # TODO: tuple recursion
LinearAtom  = Union[Atom       , SymbolicZero]  # TODO: tuple recursion
IsLinear: TypeAlias = bool

class MaybeLinear:
  def __init__(self, is_linear, val):
    self.is_linear = is_linear
    self.val = val

  def __str__(self):
    if self.is_linear:
      return f'lin {self.val}'
    else:
      return str(self.val)

MaybeLinearAtom: TypeAlias = MaybeLinear # [LinearAtom, Atom]

def linear(val): return MaybeLinear(True, val)
def nonlinear(val): return MaybeLinear(False, val)

@dataclass
class LinearJaxprEqn:
  is_linear: IsLinear
  binder: Var
  primitive: Primitive
  args: list[MaybeLinearAtom]

  def __str__(self): return PrettyPrinter.to_str(self)
  def pretty_print(self, p):
    if self.is_linear:
      p.emit(f'lin {self.binder} = {self.primitive}{arglist_str(self.args)}')
    else:
      p.emit(f'{self.binder} = {self.primitive}{arglist_str(self.args)}')

@dataclass
class LinearJaxpr:
  binder : Var
  eqns: list[LinearJaxprEqn]
  result: LinearAtom

  @property
  def ty(self):
    return LinearJaxprType(self.binder.ty, self.result.ty)

  def __str__(self): return PrettyPrinter.to_str(self)
  def pretty_print(self, p):
    p.emit(f'({self.binder}) =>')
    with p.indent():
      for eqn in self.eqns:
        eqn.pretty_print(p)
      p.emit(f'return {self.result}')

@dataclass
class LinearJaxprType:
  arg_type:  JaxType
  result_type : JaxType

def linearize_jaxpr(jaxpr:Jaxpr, top_args_loose:LoosePyVal) -> tuple[StrictPyVal, LinearJaxpr]:
  argnum = 0  # todo: make it an argument
  assert argnum < len(top_args_loose)
  primal_env = {}
  top_args = map(canonicalize_pyval, top_args_loose)
  assert jaxpr.ty.arg_types == [arg.ty for arg in top_args]
  for v, val in zip(jaxpr.binders, top_args):
    primal_env[v] = val
  tangent_binder_ty = jaxpr.ty.arg_types[argnum].tangent_type()
  with trace_ctx.new_trace():
    tangent_binder = trace_ctx.new_var(tangent_binder_ty)
  tangent_env = {jaxpr.binders[argnum] : tangent_binder.as_tracer()}
  tangent_eqns = []
  for eqn in jaxpr.eqns:
    primal_args = [arg.eval_atom(primal_env) for arg in eqn.args]
    tangent_args = [arg.eval_tangent(tangent_env) for arg in eqn.args]
    result_tangent_type = eqn.binder.ty.tangent_type()
    primal_result, eqns, tangent_result = linearize_primitive(
      result_tangent_type, eqn.primitive, primal_args, tangent_args)
    tangent_eqns.extend(eqns)
    primal_env[eqn.binder] = primal_result
    tangent_env[eqn.binder] = tangent_result
  final_primal = jaxpr.result.eval_atom(primal_env)
  final_tangent = jaxpr.result.eval_tangent(tangent_env)
  tangent_jaxpr = make_linear_jaxpr(tangent_binder, tangent_eqns, final_tangent)
  return (final_primal, tangent_jaxpr)

def linearize_primitive(
    ty:JaxType, prim:Primitive, primals:list[StrictPyVal], tangents:list[LinearPyVal]
    ) -> tuple[StrictPyVal, list[JaxprEqn], LinearPyVal]:
  if all(isinstance(t, SymbolicZero) for t in tangents):
    return SymbolicZero(ty)
  else:
    primal_result, tangent_thunk = prim.linearize_rule(ty, primals, tangents)
    with trace_ctx.new_trace() as eqns:
      tangent_result = tangent_thunk()
      return primal_result, eqns, tangent_result

def make_linear_jaxpr(
    top_binder:Var, eqns:list[JaxprEqn], result_pyval:LinearPyVal) -> LinearJaxpr:
  linear_eqns = []
  linear_vars = {top_binder}
  for eqn in eqns:
    linearities = [bool(linear_vars.intersection(arg.free_vars())) for arg in eqn.args]
    is_linear = any(linearities)
    linear_args = [MaybeLinear(lin, arg) for lin, arg in zip(linearities, eqn.args)]
    linear_eqns.append(LinearJaxprEqn(is_linear, eqn.binder, eqn.primitive, linear_args))
    if is_linear: linear_vars.add(eqn.binder)
  result = result_pyval.to_atom()
  return LinearJaxpr(top_binder, linear_eqns, result)

# === transposition ===

CotangentMap : TypeAlias = dict[Var, LinearPyVal]

def transpose_linear_jaxpr(jaxpr:LinearJaxpr, top_ct:LinearPyVal) -> LinearPyVal:
  assert jaxpr.ty.result_type == top_ct.ty
  primal_env = {}
  top_binder = jaxpr.binder
  # forward pass for nonlinear stuff
  for eqn in jaxpr.eqns:
    if not eqn.is_linear:
      args = [arg.val.eval_atom(primal_env) for arg in eqn.args]
      primal_env[eqn.binder] = apply_primitive(eqn.primitive, args)
  # backward pass for the linear stuff
  cotangents : CotangentMap = {}
  jaxpr.result.push_cotangent(cotangents, top_ct)
  for eqn in jaxpr.eqns[::-1]:
    if eqn.is_linear:
      args = [eval_maybe_linear_atom(primal_env, arg) for arg in eqn.args]
      ct = get_cotangent(cotangents, eqn.binder)
      if not isinstance(ct, SymbolicZero):
        updates = eqn.primitive.transpose_rule(ct, *args)
        for (lin_atom, arg_ct) in updates:
          lin_atom.push_cotangent(cotangents, arg_ct)
  return get_cotangent(cotangents, top_binder)

MaybeEvaluatedLinearAtom = MaybeLinear # [LinearAtom, StrictPyVal]
def eval_maybe_linear_atom(env: Env, x:MaybeLinearAtom) -> MaybeEvaluatedLinearAtom:
  if x.is_linear:
    return linear(x.val)
  else:
    return nonlinear(x.val.eval_atom(env))

def get_cotangent(cts:CotangentMap, v:Var):
  try:
    return cts[v]
  except KeyError:
    return SymbolicZero(v.ty.tangent_type())

def accum_cotangent(cts:CotangentMap, v:Var, ct:LinearPyVal):
  cts[v] = v.ty.add_tangents(get_cotangent(cts, v), ct)
