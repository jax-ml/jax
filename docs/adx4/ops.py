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

import numpy as np

from util import *
from core import *
from embedding import *

# === array type ===

@dataclass
class ArrayType(JaxType):
  shape: tuple[int, ...]
  dtype: np.dtype

  @property
  def ndim(self): return len(self.shape)

  def __str__(self):
    return f'{dtype_str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

jax_bool  = ArrayType((), np.bool_)
jax_int   = ArrayType((), np.int32)
jax_float = ArrayType((), np.float32)

@dataclass
class Array(JaxVal):
  __array_priority__ = 1000
  val : np.ndarray

  @property
  def shape(self): return self.ty.shape

  @property
  def ty(self):
    return ArrayType(self.val.shape, self.val.dtype)

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
  def __str__(self): return str(self.val)

def dtype_str(dtype):
  match dtype:
   case np.bool_: return "bool"
   case np.float32: return "f32"
   case np.int32: return "i32"
   case _ : raise Exception(f"unrecognized dtype: {dtype}")

register_canonicalizer(float, lambda x: Array(np.asarray(x, dtype=np.float32)))
register_canonicalizer(bool, lambda x: Array(np.asarray(x, dtype=np.bool_)))
register_canonicalizer(int, lambda x: Array(np.asarray(x, dtype=np.int32)))
register_canonicalizer(np.ndarray, lambda x: Array(x))
register_canonicalizer(tuple, lambda xs: JaxTupleCon([canonicalize_pyval(x) for x in xs]))

# === tuple type ===

@dataclass
class TupleType(JaxType):
  elt_types : list[JaxType]

  def __str__(self):
    return f'({",".join(str(t) for t in self.elt_types)})'

  def _getitem(self, tup, idx):
    if isinstance(idx, int):
      return emit_primitive(ProjectTuple(idx), (tup,))

class ProjectTuple(Op):
  i : int
  def __init__(self, i):
    assert isinstance(i, int)  # must be concrete
    self.i = i

  def result_type(self, tup_ty):
    return tup_ty.elt_types[self.i]

  def __str__(self):
    return f'proj{self.i}'

class ConstructTuple(Op):
  pass

# === ops ===

class Sin(Op):
  def impl(self, _, x): return Array(np.sin(x.val))
  def result_type(self, x_ty): return x_ty
  def __str__(self): return "sin"

def sin(x):
  return emit_primitive(Sin(), (x,))

class Add(Op):
  def impl(self, _, x, y): return Array(x.val + y.val)
  def result_type(self, x_ty, y_ty):
    assert x_ty == y_ty, (x_ty, y_ty)
    return x_ty
  def __str__(self): return "add"

def add(x, y):
  return emit_primitive(Add(), (x, y))

@dataclass
class Cond(JaxprHof):
  def impl(self, _, p:JaxVal, then_fun:Jaxpr, else_fun:Jaxpr):
    if p:
      return apply_jaxpr(eval_emitter, then_fun, ())
    else:
      return apply_jaxpr(eval_emitter, else_fun, ())

  def result_type(self, p_ty:JaxType, then_fun_ty:JaxprType, else_fun_ty:JaxprType):
    assert p_ty == jax_bool
    assert then_fun_ty.result_type == else_fun_ty.result_type
    return then_fun_ty.result_type

  def __str__(self): return "cond"


def cond(p, then_fun, else_fun):
  then_fun = trace_to_jaxpr(then_fun, ())
  else_fun = trace_to_jaxpr(else_fun, ())
  return emit_primitive(Cond(), (p,), (then_fun, else_fun))
