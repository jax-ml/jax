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

import jax
from jax.tree_util import tree_map
from typing import Any, Generic, TypeVar
from dataclasses import dataclass

PRNGKey = Any
Values = Any
JaxVal = Any
Activations = Any
State = Any
DType = Any
T = TypeVar("T")
V = TypeVar("V")
JaxType = Any # for now, pytree of ArrayType

@dataclass
class ArrayType:
  shape : tuple[int]
  dtype : DType

  def __str__(self):
    return f'{str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

  def __repr__(self): return self.__str__()

def tree_map_with_index(tree, f):
  cell = lambda: None
  cell.val = 0
  def body(x):
    i = cell.val
    cell.val = i + 1
    return f(i, x)
  return tree_map(body, tree)

def jax_type(x):
  return tree_map(lambda x: ArrayType(x.shape, x.dtype), x)

def scalar_jaxtype(dtype:DType) -> JaxType:
  return ArrayType((), dtype)

def hastype(x:Any, ty:JaxType) -> bool:
  return False

def eval_type(f, *arg_types:JaxType) -> JaxType:
  assert False

def check_types_equal(expected:JaxType, actual:JaxType):
  assert False

def add_leading_rank(x: JaxType, n:int) -> JaxType:
  assert False

prngkey_dtype = jax.random.key(0).dtype
float_dtype = jax.numpy.array(0.0).dtype

prngkey_type = scalar_jaxtype(prngkey_dtype)
float_type = scalar_jaxtype(float_dtype)

