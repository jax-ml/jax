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
    return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

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

class JaxArray(JaxVal):
  __array_priority__ = 1000

  def __neg__(self): return neg(self)
  def __add__(self, other): return add(self, other)
  def __radd__(self, other): return add(other, self)
  def __sub__(self, other): return sub(self, other)
  def __rsub__(self, other): return sub(other, self)
  def __mul__(self, other): return mul(self, other)
  def __rmul__(self, other): return mul(other, self)
  def __gt__(self, other): return greater(self, other)
  def __lt__(self, other): return less(self, other)
  def __bool__(self): raise Exception("Tracer can't be converted to bool")
  def __nonzero__(self): raise Exception("Traver can't be converted to bool")

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

class JaxArrayTracer(Tracer, JaxArray): pass

register_canonicalizer(float, lambda x: JaxArrayLit(np.asarray(x, dtype=np.float64)))

# === ops ===

def add(x, y): return apply_primitive(AddP(), (x, y))
class AddP(Primitive):
  def impl(self, x, y): return JaxArrayLit(np.add(x.val, y.val))
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
  def impl(self, x, y): return JaxArrayLit(np.subtract(x.val, y.val))
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
  def impl(self, x, y): return JaxArrayLit(np.multiply(x.val, y.val))
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
  def impl(self, x): return JaxArrayLit(np.negative(x.val))
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
  def impl(self, x): return JaxArrayLit(np.sin(x.val))
  def eval_type(self, t): return t
  def __str__(self): return "sin"
  def linearize_rule(self, _, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return sin(x), lambda: x_dot * cos(x)

def cos(x): return apply_primitive(CosP(), (x,))
class CosP(Primitive):
  def impl(self, x): return JaxArrayLit(np.cos(x.val))
  def eval_type(self, t): return t
  def __str__(self): return "cos"

def binop_eval_type(x: JaxArrayType, y: JaxArrayType) -> list[JaxArrayType]:
  if not isinstance(x, JaxArrayType) or not isinstance(y, JaxArrayType):
    raise TypeError
  if x != y: raise TypeError
  return JaxArrayType(x.shape, x.dtype)

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
