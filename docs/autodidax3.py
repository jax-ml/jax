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

import builtins
import itertools as it
import operator as op
from typing import NamedTuple, Union, Optional, Any, Callable, TypeVar
from collections.abc import Sequence
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import string

import numpy as np

# diffs
#  * types instead of avals. No ConcreteArray
#  * linearize instead of jvp+partial-eval
#  * JaxTupleType
#  * Jaxprs with lexical scope. Free vars may be bound by enclosing eqn builder.
#  * no explicit constvars. just use let bindings and literals
#  * Only one type of tracer. It's just a pointer to a list of eqns.
#  * No trace lifting
#  * No eager

# === core AST types ===

class JaxType: pass

@dataclass
class JaxArrayType(JaxType):
  shape: tuple[int, ...]
  dtype: np.dtype

  @property
  def ndim(self): return len(self.shape)
  def str_short(self):
    return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

@dataclass
class JaxTupleType(JaxType):
  elt_types: tuple[JaxType]
  def str_short(self):
    return f'({", ".join(str(elt) for elt in self.elt_types)})'

@dataclass
class Var:
  ty: JaxType
  def __hash__(self): return id(self)

@dataclass
class ArrayLit:
  val: Any
  @property
  def ty(self):
    return JaxArrayType(self.val.shape, self.val.dtype)
  def __str__(self): return str(self.val)

TupleCon_ = Any  # TODO: figure out how to spell recursive types
Atom = Union[Var, ArrayLit, TupleCon_]

@dataclass
class TupleCon:
  elts: tuple[Atom]

@dataclass
class SymbolicZero:
  ty : JaxType

pylit_types = { bool, int, float, np.bool_, np.int32
              , np.int64, np.float32, np.float64, np.ndarray}
PyLit = Any # any of the types in pylit_types
PyVal = Any # What the user sees. PyLit | Tracer | tuple of PyVals
TangentPyVal = Union[Atom, SymbolicZero]  # TODO: tangent tuple

@dataclass
class Primitive:
  def bind(self, *args:PyVal) -> PyVal:
    if cur_trace() is None:
      return self.impl(*args)
    else:
      arg_atoms = map(pyval_to_atom, args)
      result_ty = self.eval_type(*[atom.ty for atom in arg_atoms])
      v = Var(result_ty)
      cur_trace().eqns.append(JaxprEqn(v, self, arg_atoms))
      return Tracer(v)

  def impl(self, *args):
    raise NotImplementedError("No impl rule for " + type(self).__name__)

  def eval_type(self, *arg_types) -> JaxType :
    raise NotImplementedError("No type rule for " + type(self).__name__)

  # The implementation can assume that at least one of the tangent args is not a symbolic zero.
  def linearize_rule(self, primals:list[PyVal], tangents:list[TangentPyVal]
                     ) -> tuple[PyVal, Callable[[], TangentPyVal]]:
    raise NotImplementedError("No linearization rule for " + type(self).__name__)

@dataclass
class JaxprEqn:
  binder: Var
  primitive: Primitive
  args: list[Atom]

@dataclass
class Jaxpr:
  arg_binders: list[Var]
  eqns: list[JaxprEqn]
  result: Atom

  def __repr__(self): return str(pp_jaxpr(self))

@dataclass
class JaxprType:
  arg_types:  list[JaxType]
  result_type : JaxType

# === tracing ===

@dataclass
class Trace:
  eqns: list[JaxprEqn]

# Tracer just wraps a Var with user-exposed methods. NB: the identity of the
# tracer itself doesn't matter. Only the identity of the Var matters.
@dataclass
class Tracer:
  var: Var  # only array-typed variables should be tracers
  __array_priority__ = 1000

  def __neg__(self): return neg(self)
  def __add__(self, other): return add(self, other)
  def __radd__(self, other): return add(other, self)
  def __mul__(self, other): return mul(self, other)
  def __rmul__(self, other): return mul(other, self)
  def __gt__(self, other): return greater(self, other)
  def __lt__(self, other): return less(self, other)
  def __bool__(self): raise Exception("Tracer can't be converted to bool")
  def __nonzero__(self): raise Exception("Traver can't be converted to bool")

trace_stack = []
def cur_trace():
  if trace_stack:
    return trace_stack[-1]
  else:
    return None

@contextmanager
def new_trace():
  trace_stack.append(Trace([]))
  try:
    yield trace_stack[-1]
  finally:
    trace_stack.pop()

def make_arg_pyval(v: Var, ty: JaxType) -> PyVal:
  if isinstance(ty, JaxArrayType):
    return Tracer(v)
  elif isinstance(ty, JaxTupleType):
    raise NotImplementedError
  else:
    raise Exception("unexpected type: " + str(type(v)))

def eval_atom(env: dict[Var, PyVal], x:Atom) -> PyVal:
  if isinstance(x, Var):
    return env.get(x, Tracer(x))
  elif isinstance(x, ArrayLit):
    return x.val
  elif isisntance(x, TupleCon):
    raise NotImplementedError
  else:
    raise Exception ("unexpected type: " + str(type(x)))

def pyval_to_atom(x: PyVal) -> Atom:
  if isinstance(x, Tracer):
    return x.var
  elif type(x) in pylit_types:
    return ArrayLit(np.asarray(x))
  elif isinstance(x, tuple):
    return TupleCon(tuple(map(pyval_to_atom, x)))
  else:
    raise Exception("unexpected type: " + str(type(x)))

def get_pyval_type(x: PyVal) -> JaxType:
  if isinstance(x, Tracer):
    return x.var.ty
  elif type(x) in pylit_types:
    x_arr = np.asarray(x)
    return JaxArrayType(x_arr.shape, x_arr.dtype)
  elif isinstance(x, tuple):
    return JaxTupleType(tuple(map(get_pyval_type, x)))
  else:
    raise Exception

def stage_out(f: Callable, in_tys: JaxType,) -> Jaxpr:
  with new_trace() as trace:
    arg_binders = map(Var, in_tys)
    arg_pyvals = map(make_arg_pyval, arg_binders, in_tys)
    result_pyval = f(*arg_pyvals)
    return Jaxpr(arg_binders, trace.eqns, pyval_to_atom(result_pyval))

T = TypeVar("T")
def collect_eqns(f: Callable[[], T]) -> tuple[list[JaxprEqn], T]:
  with new_trace() as trace:
    result = f()
    return result, trace.eqns

# === type checking ===

def typecheck_jaxpr(jaxpr: Jaxpr) -> None:
  env: set[Var] = set()
  [check_binder(env, b) for b in jaxpr.arg_binders]
  for eqn in jaxpr.eqns:
    [check_atom(env, arg) for arg in eqn.args]
    result_type = eqn.primitive.eval_type(*[x.ty for x in eqn.args])
    if not result_type == eqn.binder.ty: raise TypeError
    check_binder(env, eqn.binder)
  check_atom(env, jaxpr.result)

def check_binder(env, b):
  if b in env: raise TypeError
  env.add(b)

def check_atom(env: set[Var], x: Atom) -> None:
  if isinstance(x, Var):
    if x not in env: raise TypeError("unbound variable")
  elif isinstance(x, ArrayLit):
    pass
  elif isinstance(x, TupleCon):
    [check_atom(env, elt) for elt in x.elts]
  else:
    assert False

# === linearize ===

def linearize_jaxpr(jaxpr:Jaxpr, args:PyVal) -> tuple[PyVal, Jaxpr]:
  primal_env = {b : arg for (b, arg) in zip(jaxpr.arg_binders, args)}
  tangent_binders = [Var(get_tangent_type(b.ty)) for b in jaxpr.arg_binders]
  tangent_eqns = []
  tangent_env = {b : Tracer(arg) for (b, arg) in zip(jaxpr.arg_binders, tangent_binders)}
  for eqn in jaxpr.eqns:
    primal_args = [eval_atom(primal_env, arg) for arg in eqn.args]
    tangent_args = [get_tangent_arg(tangent_env, arg) for arg in eqn.args]
    (primal_result, eqns, tangent_result) = linearize_primitive(eqn.primitive, primal_args, tangent_args)
    tangent_eqns.extend(eqns)
    primal_env[eqn.binder] = primal_result
    tangent_env[eqn.binder] = tangent_result
  final_primal = eval_atom(primal_env, jaxpr.result)
  final_tangent = instantiate_zeros(get_tangent_arg(tangent_env, jaxpr.result))
  tangent_jaxpr = Jaxpr(tangent_binders, tangent_eqns, final_tangent)
  return (final_primal, tangent_jaxpr)

def get_tangent_type(ty:JaxType) -> JaxType:
  return ty # todo

def instantiate_zeros(x:TangentPyVal) -> Atom:
  if isinstance(x, SymbolicZero):
    raise NotImplementedError
  elif isinstance(x, Tracer):
    return x.var
  elif type(x) in pylit_types:
    return ArrayLit(np.asarray(x))
  elif isinstance(x, tuple):
    return TupleCon(tuple(map(instantiate_zeros, x)))
  else:
    raise Exception("unexpected type: " + str(type(x)))

def linearize_primitive(prim:Primitive, primals:list[PyVal], tangents:list[TangentPyVal]
                        ) -> tuple[PyVal, list[JaxprEqn], TangentPyVal]:
  if all(isinstance(t, SymbolicZero) for t in tangents):
    raise NotImplementedError
  else:
    (primal_result, tangent_thunk) = prim.linearize_rule(primals, tangents)
    tangent_result, eqns = collect_eqns(tangent_thunk)
    return primal_result, eqns, tangent_result

def get_tangent_arg(tangent_env: dict[Var, TangentPyVal], primal:Atom) -> TangentPyVal:
  zero = SymbolicZero(get_tangent_type(primal.ty))
  if isinstance(primal, Var):
    return tangent_env.get(primal, zero)
  elif isinstance(primal, ArrayLit):
    return zero
  elif isinstance(primal, TupleCon):
    raise NotImplementedError
  else:
    raise Exception

# === transpose ===

def transpose_jaxpr(jaxpr:Jaxpr) -> Jaxpr:
  assert False

# === vmap ===

def vmap_jaxpr(jaxpr:Jaxpr) -> Jaxpr:
  assert False

# === compilation ===

@dataclass
class CompiledJaxpr: pass

def compile_jaxpr(japxr:Jaxpr) -> CompiledJaxpr:
  assert False

# === primitives ===

def id_op(x): return IdP().bind(x)
class IdP(Primitive):
  def impl(self, x) : return x
  def eval_type(self, x): return x

def add(x, y): return AddP().bind(x, y)
class AddP(Primitive):
  def impl(self, x, y): return np.add(x, y)
  def eval_type(self, x, y): return binop_eval_type(x, y)
  def linearize_rule(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    primal_result = x + y
    result_ty = get_tangent_type(get_pyval_type(primal_result))
    return primal_result, lambda: maybe_add(result_ty, x_dot, y_dot)

def mul(x, y): return MulP().bind(x, y)
class MulP(Primitive):
  def impl(self, x, y): return np.multiply(x, y)
  def eval_type(self, x, y): return binop_eval_type(x, y)
  def linearize_rule(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    primal_result = x * y
    def tangent_thunk():
      result_ty = get_tangent_type(get_pyval_type(primal_result))
      tx = maybe_mul(result_ty, y, x_dot)
      ty = maybe_mul(result_ty, x, y_dot)
      return maybe_add(result_ty, tx, ty)
    return x * y, tangent_thunk

def neg(x): return NegP().bind(x)
class NegP(Primitive):
  def impl(self, x): return np.negative(x)
  def eval_type(self, t): return t
  def linearize_rule(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return neg(x), lambda: neg(x_dot)

def sin(x): return SinP().bind(x)
class SinP(Primitive):
  def impl(self, x): return np.sin(x)
  def eval_type(self, t): return t
  def linearize_rule(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return sin(x), lambda: x_dot * cos(x)

def cos(x): return CosP().bind(x)
class CosP(Primitive):
  def impl(self, x): return np.cos(x)
  def eval_type(self, t): return t

def binop_eval_type(x: JaxArrayType, y: JaxArrayType) -> list[JaxArrayType]:
  if not isinstance(x, JaxArrayType) or not isinstance(y, JaxArrayType):
    raise TypeError
  if x != y: raise TypeError
  return JaxArrayType(x.shape, x.dtype)

def maybe_mul(result_ty:JaxType, x:PyVal, t:TangentPyVal) -> TangentPyVal:
  if isinstance(t, SymbolicZero):
    return SymbolicZero(result_ty)
  else:
    return x * t

def maybe_add(result_ty:JaxType, t1:TangentPyVal, t2:TangentPyVal) -> TangentPyVal:
  if isinstance(t1, SymbolicZero):
    return t2
  elif isinstance(t2, SymbolicZero):
    return t1
  else:
    return t1 + t2

# === higher-order primitives ===

class CondP(Primitive):
  def impl(self, when_true, when_false) : assert False
  def eval_type(self, when_true, when_false): assert False

class ScanP(Primitive): pass

# === pretty printer ===

class PPrint:
  lines: list[tuple[int, str]]

  def __init__(self, lines):
    self.lines = lines

  def indent(self, indent: int) -> 'PPrint':
    return PPrint([(indent + orig_indent, s) for orig_indent, s in self.lines])

  def __add__(self, rhs: 'PPrint') -> 'PPrint':
    return PPrint(self.lines + rhs.lines)

  def __rshift__(self, rhs: 'PPrint') -> 'PPrint':
    if not rhs.lines: return self
    if not self.lines: return rhs
    indent, s = self.lines[-1]
    indented_block = rhs.indent(indent + len(s))
    common_line = s + ' ' * rhs.lines[0][0] + rhs.lines[0][1]
    return PPrint(self.lines[:-1]
                  + [(indent, common_line)]
                  + indented_block.lines[1:])

  def __str__(self) -> str:
    return '\n'.join(' ' * indent + s for indent, s in self.lines)

def pp(s: Any) -> PPrint:
  return PPrint([(0, line) for line in str(s).splitlines()])

def vcat(ps: list[PPrint]) -> PPrint:
  return sum(ps, pp(''))

def pp_jaxpr(jaxpr: Jaxpr) -> PPrint:
  namegen = (''.join(s) for r in it.count(1)
             for s in it.permutations(string.ascii_lowercase, r))
  names = defaultdict(lambda: next(namegen))
  arg_binders = ', '.join(binder_str(names, x) for x in jaxpr.arg_binders)
  eqns = vcat([pp_eqn(names, e) for e in jaxpr.eqns])
  out = atom_str(names, jaxpr.result)
  return (pp(f'{{ lambda {arg_binders} .') +
          ((pp('let ') >> eqns) + pp(f'in {out}}}')).indent(2))

def pp_eqn(names: defaultdict[Var, str], eqn: JaxprEqn) -> PPrint:
  lhs = binder_str(names, eqn.binder)
  rhs = (pp(str(eqn.primitive)) >>
         pp(' ' + ' '.join(atom_str(names, arg) for arg in eqn.args)))
  return pp(lhs) >> pp(' = ') >> rhs

def binder_str(names: defaultdict[Var, str], v: Var) -> str:
  return f'{names[v]}:{v.ty.str_short()}'

def atom_str(names: defaultdict[Var, str], atom:Atom) -> str:
  if isinstance(atom, Var):
    return f'{names[atom]}'
  else:
    return str(atom)

# === user-facing wrappers ===

def jit(f): assert False

def grad(f): assert False

# === tests ===

def f(x):
  y = sin(x) * 2.
  z = - y + x
  return z

print(f(3.0))

jaxpr = stage_out(f, (get_pyval_type(3.),))

print(linearize_jaxpr(jaxpr, (3.0,)))

print(jaxpr)
typecheck_jaxpr(jaxpr)
