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
#  * explicit type for linear jaxprs with linearity checker


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
  def __eq__(self, other): return self is other
  def free_vars(self): return {self}

@dataclass
class ArrayLit:
  val: Any
  @property
  def ty(self):
    return JaxArrayType(self.val.shape, self.val.dtype)
  def __str__(self): return str(self.val)
  def free_vars(self): return set()

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
LinearPyVal = Union[Atom, SymbolicZero]  # TODO: tangent tuple
LinearAtom_ = Any # TODO: toposort

@dataclass
class Primitive:
  def apply(self, *args:PyVal) -> PyVal:
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
  def linearize_rule(self, primals:list[PyVal], tangents:list[LinearPyVal]
                     ) -> tuple[PyVal, Callable[[], LinearPyVal]]:
    raise NotImplementedError("No linearization rule for " + type(self).__name__)

  def transpose_rule(self, cotangent: PyVal, *args:list[LinearPyVal]
                     ) -> list[tuple[LinearAtom_, PyVal]]:
    raise NotImplementedError("No transposition rule for " + type(self).__name__)

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

  @property
  def ty(self):
    return JaxprType([v.ty for v in self.arg_binders], self.result.ty)


  def __repr__(self): return str(pp_jaxpr(self))

@dataclass
class JaxprType:
  arg_types:  list[JaxType]
  result_type : JaxType

# === linear jaxprs ===

@dataclass
class LinearAtom:
  value: Union[SymbolicZero, Var]
  @property
  def ty(self): return self.value.ty
MaybeLinearAtom = Union[Atom, LinearAtom]

@dataclass
class LinearJaxprEqn:
  is_linear: bool
  binder: Var
  primitive: Primitive
  args: list[MaybeLinearAtom]

@dataclass
class LinearJaxpr:
  binder : Var
  eqns: list[LinearJaxprEqn]
  result: LinearAtom

  @property
  def ty(self):
    return LinearJaxprType(self.binder.ty, self.result.ty)

@dataclass
class LinearJaxprType:
  arg_type:  JaxType
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

EvalEnv = dict[Var, PyVal]

def eval_atom(env: EvalEnv, x:Atom) -> PyVal:
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

def pyval_type(x: PyVal) -> JaxType:
  if isinstance(x, Tracer):
    return x.var.ty
  elif type(x) in pylit_types:
    x_arr = np.asarray(x)
    return JaxArrayType(x_arr.shape, x_arr.dtype)
  elif isinstance(x, tuple):
    return JaxTupleType(tuple(map(pyval_type, x)))
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

def linearize_jaxpr(jaxpr:Jaxpr, args:PyVal) -> tuple[PyVal, LinearJaxpr]:
  argnum = 0
  assert argnum < len(args)
  assert jaxpr.ty.arg_types == [pyval_type(arg) for arg in args]
  primal_env = {b : arg for (b, arg) in zip(jaxpr.arg_binders, args)}
  tangent_binder = Var(tangent_type(jaxpr.ty.arg_types[argnum]))
  tangent_env = {jaxpr.arg_binders[argnum] : Tracer(tangent_binder)}
  tangent_eqns = []
  for eqn in jaxpr.eqns:
    primal_args = [eval_atom(primal_env, arg) for arg in eqn.args]
    tangent_args = [get_tangent_arg(tangent_env, arg) for arg in eqn.args]
    (primal_result, eqns, tangent_result) = linearize_primitive(eqn.primitive, primal_args, tangent_args)
    tangent_eqns.extend(eqns)
    primal_env[eqn.binder] = primal_result
    tangent_env[eqn.binder] = tangent_result
  final_primal = eval_atom(primal_env, jaxpr.result)
  final_tangent = get_tangent_arg(tangent_env, jaxpr.result)
  tangent_jaxpr = make_linear_jaxpr(tangent_binder, tangent_eqns, final_tangent)
  return (final_primal, tangent_jaxpr)

def tangent_type(ty:JaxType) -> JaxType:
  return ty # todo

def instantiate_zeros(x:LinearPyVal) -> PyVal:
  if isinstance(x, SymbolicZero):
    if isinstance(x.ty, JaxArrayType):
      return np.zeros(x.ty.shape, x.ty.dtype)
    else:
      raise NotImplementedError
  elif isinstance(x, Tracer):
    return x
  elif type(x) in pylit_types:
    return x
  elif isinstance(x, tuple):
    return tuple(map(instantiate_zeros, x))
  else:
    raise Exception("unexpected type: " + str(type(x)))

def linearize_primitive(prim:Primitive, primals:list[PyVal], tangents:list[LinearPyVal]
                        ) -> tuple[PyVal, list[JaxprEqn], LinearPyVal]:
  if all(isinstance(t, SymbolicZero) for t in tangents):
    print(prim)
    raise NotImplementedError
  else:
    (primal_result, tangent_thunk) = prim.linearize_rule(primals, tangents)
    tangent_result, eqns = collect_eqns(tangent_thunk)
    return primal_result, eqns, tangent_result

def get_tangent_arg(tangent_env: dict[Var, LinearPyVal], primal:Atom) -> LinearPyVal:
  zero = SymbolicZero(tangent_type(primal.ty))
  if isinstance(primal, Var):
    return tangent_env.get(primal, zero)
  elif isinstance(primal, ArrayLit):
    return zero
  elif isinstance(primal, TupleCon):
    raise NotImplementedError
  else:
    raise Exception

def make_linear_jaxpr(
    top_binder:Var, eqns:list[JaxprEqn], result_pyval:LinearPyVal) -> LinearJaxpr:
  linear_eqns = []
  linear_vars = {top_binder}
  for eqn in eqns:
    maybe_linear_args = [as_maybe_linear_arg(linear_vars, arg) for arg in eqn.args]
    is_linear = any(isinstance(arg, LinearAtom) for arg in maybe_linear_args)
    linear_eqns.append(LinearJaxprEqn(is_linear, eqn.binder, eqn.primitive, maybe_linear_args))
    if is_linear: linear_vars.add(eqn.binder)
  if isinstance(result_pyval, SymbolicZero):
    result = LinearAtom(result_pyval)
  else:
    result = LinearAtom(pyval_to_atom(result_pyval))
  return LinearJaxpr(top_binder, linear_eqns, result)

def as_maybe_linear_arg(linear_vars:set[Var], x:Union[Atom]) -> MaybeLinearAtom:
  if linear_vars.intersection(x.free_vars()):
    return LinearAtom(x)
  else:
    return x

# === transpose ===

CotangentMap = dict[Var, PyVal]

def transpose_linear_jaxpr(ljaxpr:LinearJaxpr, top_ct:LinearPyVal) -> LinearPyVal:
  assert ljaxpr.ty.result_type == pyval_type(top_ct)
  primal_env = {}
  top_binder = ljaxpr.binder
  # forward pass for nonlinear stuff
  for eqn in ljaxpr.eqns:
    if not eqn.is_linear:
      args = [eval_atom(primal_env, arg) for arg in eqn.args]
      primal_env[eqn.binder] = eqn.primitive.apply(*args)
  # backward pass for the linear stuff
  cotangents : CotangentMap = {}
  if isinstance(ljaxpr.result.value, Var):
    cotangents[ljaxpr.result.value] = top_ct
  else:
    raise NotImplementedError # todo: symbolic zero case
  for eqn in ljaxpr.eqns[::-1]:
    if eqn.is_linear:
      args = [eval_maybe_linear_atom(primal_env, arg) for arg in eqn.args]
      ct = get_cotangent(cotangents, eqn.binder)
      if not isinstance(ct, SymbolicZero):
        updates = eqn.primitive.transpose_rule(ct, *args)
        for (lin_atom, arg_ct) in updates:
          if isinstance(lin_atom.value, Var):
            accum_cotangent(cotangents, lin_atom.value, arg_ct)
  return get_cotangent(cotangents, top_binder)

def eval_maybe_linear_atom(env: EvalEnv, x:Union[Atom, LinearAtom]) -> Union[PyVal, LinearAtom]:
  if isinstance(x, LinearAtom):
    return x
  else:
    return eval_atom(env, x)

def get_cotangent(cts:CotangentMap, v:Var):
  if v in cts:
    return cts[v]
  else:
    return SymbolicZero(tangent_type(v.ty))

def accum_cotangent(cts:dict[Var, PyVal], v:Var, ct:LinearPyVal):
  ct_new = maybe_add(v.ty, get_cotangent(cts, v), ct)
  if not isinstance(ct_new, SymbolicZero):
    cts[v] = ct_new

# === vmap ===

def vmap_jaxpr(jaxpr:Jaxpr) -> Jaxpr:
  assert False

# === compilation ===

@dataclass
class CompiledJaxpr: pass

def compile_jaxpr(japxr:Jaxpr) -> CompiledJaxpr:
  assert False

# === primitives ===

def id_op(x): return IdP().apply(x)
class IdP(Primitive):
  def impl(self, x) : return x
  def eval_type(self, x): return x

def add(x, y): return AddP().apply(x, y)
class AddP(Primitive):
  def impl(self, x, y): return np.add(x, y)
  def eval_type(self, x, y): return binop_eval_type(x, y)
  def linearize_rule(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    primal_result = x + y
    result_ty = tangent_type(pyval_type(primal_result))
    return primal_result, lambda: maybe_add(result_ty, x_dot, y_dot)
  def transpose_rule(self, ct, x, y):
    assert isinstance(x, LinearAtom) and isinstance(y, LinearAtom)
    return [(x, ct), (y, ct)]

def mul(x, y): return MulP().apply(x, y)
class MulP(Primitive):
  def impl(self, x, y): return np.multiply(x, y)
  def eval_type(self, x, y): return binop_eval_type(x, y)
  def linearize_rule(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    primal_result = x * y
    def tangent_thunk():
      result_ty = tangent_type(pyval_type(primal_result))
      tx = maybe_mul(result_ty, y, x_dot)
      ty = maybe_mul(result_ty, x, y_dot)
      return maybe_add(result_ty, tx, ty)
    return x * y, tangent_thunk
  def transpose_rule(self, ct, x, y):
    updates = []
    if isinstance(x, LinearAtom):
      updates.append((x, ct * y))
    if isinstance(y, LinearAtom):
      updates.append((y, x * ct))
    return updates

def neg(x): return NegP().apply(x)
class NegP(Primitive):
  def impl(self, x): return np.negative(x)
  def eval_type(self, t): return t
  def linearize_rule(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return neg(x), lambda: neg(x_dot)
  def transpose_rule(self, ct, x):
    assert isinstance(x, LinearAtom)
    return [(x, -ct)]

def sin(x): return SinP().apply(x)
class SinP(Primitive):
  def impl(self, x): return np.sin(x)
  def eval_type(self, t): return t
  def linearize_rule(self, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return sin(x), lambda: x_dot * cos(x)

def cos(x): return CosP().apply(x)
class CosP(Primitive):
  def impl(self, x): return np.cos(x)
  def eval_type(self, t): return t

def binop_eval_type(x: JaxArrayType, y: JaxArrayType) -> list[JaxArrayType]:
  if not isinstance(x, JaxArrayType) or not isinstance(y, JaxArrayType):
    raise TypeError
  if x != y: raise TypeError
  return JaxArrayType(x.shape, x.dtype)

def maybe_mul(result_ty:JaxType, x:PyVal, t:LinearPyVal) -> LinearPyVal:
  if isinstance(t, SymbolicZero):
    return SymbolicZero(result_ty)
  else:
    return x * t

def maybe_add(result_ty:JaxType, t1:LinearPyVal, t2:LinearPyVal) -> LinearPyVal:
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

# === utils ===

def map(f, *xs):
  return list(builtins.map(f, *xs))

# === user-facing wrappers ===

def jit(f): assert False

def value_and_grad(f, *args):
  jaxpr = stage_out(f, map(pyval_type, args))
  assert jaxpr.ty.result_type == JaxArrayType((), np.float64)
  (value, linearized) = linearize_jaxpr(jaxpr, args)
  gradient = instantiate_zeros(transpose_linear_jaxpr(linearized, 1.0))
  return value, gradient

def value_and_deriv(f, *args):
  jaxpr = stage_out(f, map(pyval_type, args))
  assert jaxpr.ty.result_type == JaxArrayType((), np.float64)
  (value, linearized) = linearize_jaxpr(jaxpr, args)
  gradient = instantiate_zeros(transpose_linear_jaxpr(linearized, 1.0))
  return value, gradient

# === tests ===

def nd(f, x):
  deriv = (f(x + 0.001) - f(x - 0.001)) / 0.002
  return f(x), deriv

def f(x):
  y = sin(x) * 2.
  z = - y + x
  return z

print(f(3.0))

jaxpr = stage_out(f, (pyval_type(3.),))
print(linearize_jaxpr(jaxpr, (3.0,)))

print(value_and_grad(f, 3.0))
print(nd(f, 3.0))


