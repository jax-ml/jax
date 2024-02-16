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

import numpy as np

from util import *
from core import *

# === array type ===

@dataclass
class JaxArrayType(JaxType):
  shape: tuple[int, ...]
  dtype: np.dtype

  @property
  def ndim(self): return len(self.shape)

  def __str__(self):
    return f'{dtype_str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

  @staticmethod
  def var_to_tracer(v):
    return JaxArrayTracer(v)

  def tangent_type(self):
    # TODO: handle ints and bools. Do we make a unit dtype or just use JaxTuple?
    return JaxArrayType(self.shape, self.dtype)

  def add_tangents(self, x:LinearPyVal, y:LinearPyVal) -> LinearPyVal:
    return maybe_add(self, x, y)

  def instantiate_zeros(self, x:LinearPyVal) -> StrictPyVal:
    if isisntance(x, SymbolicZero):
      return JaxArrayType(np.zeros(self.shape, self.dtype))
    else:
      return x

  def to_repval(self, val):
    return [val]

  def from_repval(self, repvals):
    val, = repvals
    return val

  @property
  def rep_types(self):
    return [self]

jax_bool = JaxArrayType((), np.bool_)
jax_int  = JaxArrayType((), np.int64)

class JaxArray(JaxVal):
  __array_priority__ = 1000

  @property
  def shape(self): return self.ty.shape

  def __getitem__(self, ix): return array_getitem(self, ix)
  def __neg__(self): return neg(self)
  def __add__(self, other): return add(self, other)
  def __radd__(self, other): return add(other, self)
  def __sub__(self, other): return sub(self, other)
  def __rsub__(self, other): return sub(other, self)
  def __mul__(self, other): return mul(self, other)
  def __rmul__(self, other): return mul(other, self)
  def __gt__(self, other): return greater(self, other)
  def __lt__(self, other): return less(self, other)

@dataclass
class JaxArrayLit(JaxArray):
  val: np.ndarray

  def to_atom(self): return self
  def free_vars(self): return set()
  def eval_atom(self, env:Env): return self
  def eval_tangent(self, _):
    return SymbolicZero(self.ty.tangent_type())

  @property
  def ty(self):
    return JaxArrayType(self.val.shape, self.val.dtype)

  def __str__(self):
    return str(self.val)

  def __bool__(self): return bool(self.val)
  def __nonzero__(self): return bool(self.val)

class JaxArrayTracer(Tracer, JaxArray):
  def __bool__(self): raise Exception("Tracer can't be converted to bool")
  def __nonzero__(self): raise Exception("Traver can't be converted to bool")

def dtype_str(dtype):
  match dtype:
   case np.bool_: return "bool"
   case np.float64: return "f64"
   case np.int64: return "i64"
   case _ : raise Exception(f"unrecognized dtype: {dtype}")

register_canonicalizer(float, lambda x: JaxArrayLit(np.asarray(x, dtype=np.float64)))
register_canonicalizer(bool, lambda x: JaxArrayLit(np.asarray(x, dtype=np.bool_)))
register_canonicalizer(int, lambda x: JaxArrayLit(np.asarray(x, dtype=np.int64)))
register_canonicalizer(np.ndarray, lambda x: JaxArrayLit(x))

# === list type ===

@dataclass
class JaxListType(JaxType):
  size : int
  elt_ty : JaxType

  # def to_repval(self, x):

class JaxList(JaxVal):
  def __getitem__(self, ix): return list_getitem(self, ix)

@dataclass
class JaxListLit(JaxList):
  list_ty : JaxListType
  list_repval  : JaxVal

  @property
  def ty(self): return self.list_ty

  def __str__(self):
    return f'[{", ".join(str(elt) for elt in self)}]'

class JaxListTracer(Tracer, JaxList): pass

def make_jax_list(elt_ty:JaxType, elts: list[JaxVal]) -> JaxListLit:
  n = len(elts)
  list_ty = JaxListType(n, elt_ty)
  repval_list = [elt_ty.to_repval(elt) for elt in elts]
  repval_size = len(elt_ty.rep_types)
  repval = [JaxArrayLit(np.stack(map(lambda x: x[i], repval_list)))
            for i in range(repval_size)]
  return JaxListLit(list_ty, repval)

def list_getitem(xs, i): return apply_primitive(ListGetitemP(), (xs, i))
class ListGetitemP(Primitive):

  def eval_type(self, xs_ty, i_ty):
    assert i_ty == jax_int
    return xs_ty.elt_ty

  def impl(self, _, xs, i):
    elt_ty = xs.ty.elt_ty
    return elt_ty.from_repval([arr[i] for arr in xs.list_repval])

# === higher-order ops ===

def cond(predicate_raw, when_true_callable, when_false_callable):
  predicate = canonicalize_pyval(predicate_raw)
  assert predicate.ty == jax_bool
  when_true  = trace_to_jaxpr(when_true_callable , ())
  when_false = trace_to_jaxpr(when_false_callable, ())
  return apply_primitive_strict(CondP(when_true, when_false), (predicate,))

@dataclass
class CondP(SecondOrderPrimitive):
  when_true: Jaxpr
  when_false: Jaxpr
  @property
  def jaxprs(self):
    return {"when_true": self.when_true, "when_false": self.when_false}

  def eval_type(self, predicate):
    assert self.when_true.ty == self.when_false.ty
    assert predicate == jax_bool
    return self.when_true.ty.result_type

  def impl(self, env, predicate):
    if predicate:
      return eval_jaxpr(env, self.when_true, ())
    else:
      print(self.when_false)
      return eval_jaxpr(env, self.when_false, ())

  def __str__(self): return "cond"

def fori(n:int, body_callable:Callable):
  assert isinstance(n, int)
  body = trace_to_jaxpr(body_callable, (jax_int,))
  return apply_primitive_strict(ForP(n, body), ())

@dataclass
class ForP(SecondOrderPrimitive):
  n : int
  body: Jaxpr

  @property
  def jaxprs(self):
     return {"body": self.body}

  def eval_type(self):
    return JaxListType(self.n, self.body.ty.result_type)

  def impl(self, env):
    elts = [eval_jaxpr(env, self.body, (canonicalize_pyval(i),)) for i in range(self.n)]
    return make_jax_list(self.body.ty.result_type, elts)

  def __str__(self): return "for"

# === ops ===

def add(x, y): return apply_primitive(AddP(), (x, y))
class AddP(Primitive):
  def impl(self, _, x, y): return JaxArrayLit(np.add(x.val, y.val))
  def eval_type(self, x, y): return binop_eval_type(x, y)
  def __str__(self): return "add"
  def linearize_rule(self, ty, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    primal_result = x + y
    return primal_result, lambda: ty.add_tangents(x_dot, y_dot)
  def transpose_rule(self, ct, x, y):
    assert x.is_linear and y.is_linear
    return [(x.val, ct), (y.val, ct)]

def sub(x, y): return apply_primitive(SubP(), (x, y))
class SubP(Primitive):
  def impl(self, _, x, y): return JaxArrayLit(np.subtract(x.val, y.val))
  def eval_type(self, x, y): return binop_eval_type(x, y)
  def __str__(self): return "sub"
  def linearize_rule(self, ty, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    primal_result = x - y
    return primal_result, lambda: ty.maybe_sub(x_dot, y_dot)
  def transpose_rule(self, ct, x, y):
    assert x.is_linear and y.is_linear
    return [(x.val, ct), (y.val, maybe_neg(ct))]

def mul(x, y): return apply_primitive(MulP(), (x, y))
class MulP(Primitive):
  def impl(self, _, x, y): return JaxArrayLit(np.multiply(x.val, y.val))
  def eval_type(self, x, y): return binop_eval_type(x, y)
  def __str__(self): return "mul"
  def linearize_rule(self, result_ty, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    primal_result = x * y
    def tangent_thunk():
      tx = maybe_mul(result_ty, y, x_dot)
      ty = maybe_mul(result_ty, x, y_dot)
      return result_ty.add_tangents(tx, ty)
    return x * y, tangent_thunk
  def transpose_rule(self, ct, x, y):
    updates = []
    if x.is_linear:
      updates.append((x.val, ct * y.val))
    if y.is_linear:
      updates.append((y.val, x.val * ct))
    return updates

def neg(x): return apply_primitive(NegP(), (x,))
class NegP(Primitive):
  def impl(self, _, x): return JaxArrayLit(np.negative(x.val))
  def eval_type(self, t): return t
  def __str__(self): return "neg"
  def linearize_rule(self, ty, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return neg(x), lambda: neg(x_dot)
  def transpose_rule(self, ct, x):
    assert x.is_linear
    return [(x.val, -ct)]

def sin(x): return apply_primitive(SinP(), (x,))
class SinP(Primitive):
  def impl(self, _, x): return JaxArrayLit(np.sin(x.val))
  def eval_type(self, t): return t
  def __str__(self): return "sin"
  def linearize_rule(self, _, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return sin(x), lambda: x_dot * cos(x)

def cos(x): return apply_primitive(CosP(), (x,))
class CosP(Primitive):
  def impl(self, _, x): return JaxArrayLit(np.cos(x.val))
  def eval_type(self, t): return t
  def __str__(self): return "cos"

def array_getitem(x, ixs): return apply_primitive(ArrayGetItemP(), (x, ixs))
class ArrayGetItemP(Primitive):
  def impl(self, _, x, ix): return JaxArrayLit(x.val.__getitem__(ix.val))
  def eval_type(self, x_ty, ixs_ty):
    if isinstance(ixs_ty, JaxArrayType) and ixs_ty.shape == ():
      assert len(x_ty.shape) > 0
      return JaxArrayType(x_ty.shape[1:], x_ty.dtype)
    else:
      return False
  def __str__(self): assert Fales
  def linearize_rule(self, _, primals, tangents): assert False

def greater(x, y): return apply_primitive(GreaterP(), (x, y))
class GreaterP(Primitive):
  def impl(self, _, x, y): return JaxArrayLit(np.greater(x.val, y.val))
  def eval_type(self, x, y): return cmp_binop_eval_type(x, y)
  def __str__(self): return "greater"

def binop_eval_type(x: JaxArrayType, y: JaxArrayType) -> JaxArrayType:
  if not isinstance(x, JaxArrayType) or not isinstance(y, JaxArrayType):
    raise TypeError
  if x != y: raise TypeError
  return JaxArrayType(x.shape, x.dtype)

def cmp_binop_eval_type(x: JaxArrayType, y: JaxArrayType) -> JaxArrayType:
  if not isinstance(x, JaxArrayType) or not isinstance(y, JaxArrayType):
    raise TypeError
  if x != y: raise TypeError
  return JaxArrayType(x.shape, np.bool_)

# === linear operations handling symbolic zeros ===

def maybe_add(ty:JaxType, x:LinearPyVal, y:LinearPyVal) -> LinearPyVal:
  if isinstance(x, SymbolicZero):
    return y
  elif isinstance(y, SymbolicZero):
    return x
  else:
    return x + y

def maybe_neg(ty:JaxType, x:StrictPyVal) -> LinearPyVal:
  if isinstance(x, SymbolicZero):
    return SymbolicZero(ty)
  else:
    return neg(x)

def maybe_mul(ty:JaxType, x:StrictPyVal, t:LinearPyVal) -> LinearPyVal:
  if isinstance(t, SymbolicZero):
    return SymbolicZero(ty)
  else:
    return x * t

def maybe_sub(ty:JaxType, x:LinearPyVal, y:LinearPyVal) -> LinearPyVal:
  if isinstance(x, SymbolicZero):
    return neg(y)
  elif isinstance(y, SymbolicZero):
    return x
  else:
    return x - y
