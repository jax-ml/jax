# Copyright 2018 Google LLC
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

import numpy as onp

from .core import lattice_join, Primitive, Unit, unit, AbstractUnit, array_types, ShapedArray
from .dtypes import canonicalize_dtype, result_type, python_scalar_dtypes
from .tree_util import register_pytree_node
from .util import safe_map

map = safe_map

jaxval_adders = {}
jaxval_adders[Unit] = lambda _, __: unit

def add_jaxvals(x, y):
  return add_jaxvals_p.bind(x, y)

add_jaxvals_p = Primitive('add_any')

@add_jaxvals_p.def_impl
def add_impl(xs, ys):
  # assert type(xs) == type(ys), (xs, ys)
  return jaxval_adders[type(xs)](xs, ys)

@add_jaxvals_p.def_abstract_eval
def add_abstract(xs, ys):
  return lattice_join(xs, ys)

def _zeros_like_array(x):
  dtype = canonicalize_dtype(result_type(x))
  return onp.broadcast_to(onp.array(0, dtype), onp.shape(x))

def _zeros_like_python_scalar(x):
  return onp.array(0, python_scalar_dtypes[type(x)])

jaxval_zeros_likers = {}
for t in array_types:
  jaxval_zeros_likers[t] = _zeros_like_array

for t in python_scalar_dtypes.keys():
  jaxval_zeros_likers[t] = _zeros_like_python_scalar

def zeros_like_aval(aval):
  return aval_zeros_likers[type(aval)](aval)

def _zeros_like_shaped_array(aval):
  assert isinstance(aval, ShapedArray)
  return onp.zeros(aval.shape, dtype=aval.dtype)

aval_zeros_likers = {}
aval_zeros_likers[AbstractUnit] = lambda _: unit
aval_zeros_likers[ShapedArray] = _zeros_like_shaped_array

def zeros_like_jaxval(val):
  return zeros_like_p.bind(val)

zeros_like_p = Primitive('zeros_like')

@zeros_like_p.def_impl
def zeros_like_impl(example):
  return jaxval_zeros_likers[type(example)](example)

zeros_like_p.def_abstract_eval(lambda x: x)

class Zero(object):
  def __repr__(self):
    return "Zero"

zero = Zero()

register_pytree_node(Zero, lambda z: ((), None), lambda _, xs: zero)
