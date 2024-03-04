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

from typing import Any, Callable
from jax._src import tree_util
from jax.tree_util import register_pytree_node
import dataclasses
from dataclasses import dataclass
import numpy as np
import jax

JaxType = Any
JaxVal = Any
FancyTupleKind = Any
PRNGKey = Any

# === core types and functions ===

@dataclass
class ArrayType:
  shape : tuple[int]
  dtype : np.dtype
  def __str__(self): return self.__repr__()
  def __repr__(self):
    return f'{self.dtype}[{",".join(str(d) for d in self.shape)}]'

@dataclass
class FancyTupleKind:
  pytype : type
  meta   : Any

def is_tuple(val:Any) -> bool:
  # TODO: canonical way to do this in pytree land?
  return type(val) in tree_util._registry

def to_plain_tuple(val:Any) -> tuple:
  handler = tree_util._registry[type(val)]
  children, _ = handler.to_iter(val)
  return tuple(children)

def to_fancy_tuple(kind:FancyTupleKind, val:tuple):
  handler = tree_util._registry[kind.pytype]
  return handler.from_iter(kind.meta, val)

def tuple_kind(val) -> FancyTupleKind:
  handler = tree_util._registry[type(val)]
  _, meta = handler.to_iter(val)
  return FancyTupleKind(type(val), meta)

def pytree_dataclass(cls):
  dcls = dataclass(cls)
  fields = [f.name for f in dataclasses.fields(dcls)]
  def flatten(x): return tuple(getattr(x, f) for f in fields), ()
  def unflatten(_, xs): return dcls(**dict(zip(fields, xs)))
  register_pytree_node(dcls, flatten, unflatten)
  return dcls

# === common traversals ===

def tuple_map(f, xs):
  kind = tuple_kind(xs)
  xs_plain = to_plain_tuple(xs)
  ys_plain = tuple(f(x) for x in xs_plain)
  return to_fancy_tuple(kind, ys_plain)

def add_rank(ty:JaxType, size):
  if isinstance(ty, ArrayType):
    return ArrayType((size,) + ty.shape, ty.dtype)
  elif is_tuple(ty):
    return tuple_map(lambda t: add_rank(t, size), ty)
  else:
    raise TypeError(f"Not a valid JaxType: {ty}")

def tangent_type(ty:JaxType):
  if isinstance(ty, ArrayType):
    tangent_dtype = ty.dtype # TODO: handle ints specially
    return ArrayType(ty.shape, tangent_dtype)
  elif is_tuple(ty):
    return tuple_map(tangent_type, ty)
  else:
    raise TypeError(f"Not a valid JaxType: {ty}")

def is_type(val:Any):
  if is_tuple(val):
    return all(map(is_type, to_plain_tuple(val)))
  elif isinstance(val, ArrayType):
    return True
  else:
    return False

def prng_map(f, key, val):
  if is_tuple(val):
    kind = tuple_kind(val)
    xs = to_plain_tuple(val)
    keys = jax.random.split(key, len(xs))
    ys = tuple(prng_map(f, k, x) for k, x in zip(keys, xs))
    return to_fancy_tuple(kind, ys)
  else:
    return f(key, val)

# === shorthand syntax ===

# DTypeShorthand lets you write types with shorthand syntax. Instead of this:
#    ProductType(TupleProduct(), (ArrayType((10,), np.float32), ArrayType((128,), np.int32)))
# you write this:
#    (f32[10], i32[128])

@dataclass
class DTypeShorthand:
  dtype : np.dtype
  def __getitem__(self, shape):
    if not isinstance(shape, tuple):  # what's the idiomatic thing to do here?
      shape = (shape,)
    return ArrayType(shape, self.dtype)

# === type checking utils ===

prng_dtype = jax.random.key(0).dtype

f32  = DTypeShorthand(np.float32)
i32  = DTypeShorthand(np.int32)
b    = DTypeShorthand(np.bool_)
prng = DTypeShorthand(prng_dtype)

def check_function_type(f:Callable, arg_tys:tuple[JaxType], result_ty:JaxType):
  result_pytree_type = jax.eval_shape(f, *arg_tys)
  result_type = tree_util.tree_map(lambda arr: ArrayType(arr.shape, arr.dtype), result_pytree_type)
  check_types_equal(result_ty, result_type)

def check_types_equal(t1:JaxType, t2:JaxType):
  assert t1 == t2, f"type mismatch:\n  {t1}\n  {t2}"

def check_has_type(x:Any, ty:JaxType) -> bool:
  check_types_equal(ty, type_of(x))

def type_of(x:Any) -> JaxType:
  assert False
