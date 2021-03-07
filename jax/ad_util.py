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


from jax import core
from .core import (lattice_join, Primitive, Unit, unit, AbstractUnit,
                   valid_jaxtype, raise_to_shaped, get_aval)
from .tree_util import register_pytree_node
from typing import Any, Callable, Dict, Type
from ._src.util import safe_map

from ._src import traceback_util
traceback_util.register_exclusion(__file__)

Array = Any

map = safe_map

jaxval_adders: Dict[type, Callable] = {}
jaxval_adders[Unit] = lambda _, __: unit

def add_jaxvals(x, y):
  if core.get_aval(x) is core.abstract_unit is core.get_aval(y):
    return core.unit
  else:
    return add_jaxvals_p.bind(x, y)

add_jaxvals_p: Primitive = Primitive('add_any')
add_any_p = add_jaxvals_p

@add_jaxvals_p.def_impl
def add_impl(xs, ys):
  return jaxval_adders[type(xs)](xs, ys)

@add_jaxvals_p.def_abstract_eval
def add_abstract(xs, ys):
  return lattice_join(xs, ys)

jaxval_zeros_likers: Dict[type, Array] = {}

def zeros_like_aval(aval):
  return aval_zeros_likers[type(aval)](aval)

aval_zeros_likers: Dict[Type[core.AbstractValue], Array] = {}
aval_zeros_likers[AbstractUnit] = lambda _: unit

def zeros_like_jaxval(val):
  return zeros_like_p.bind(val)

zeros_like_p: Primitive = Primitive('zeros_like')

@zeros_like_p.def_impl
def zeros_like_impl(example):
  return jaxval_zeros_likers[type(example)](example)

zeros_like_p.def_abstract_eval(lambda x: x)

class Zero:
  __slots__ = ['aval']
  def __init__(self, aval):
    self.aval = aval
  def __repr__(self):
    return 'Zero({})'.format(self.aval)
  @staticmethod
  def from_value(val):
    return Zero(raise_to_shaped(get_aval(val)))

register_pytree_node(Zero, lambda z: ((), z.aval), lambda aval, _: Zero(aval))


def _stop_gradient_impl(x):
  if not valid_jaxtype(x):
    raise TypeError("stop_gradient only works on valid JAX arrays, but "
                    f"input argument is: {x}")
  return x

stop_gradient_p : Primitive = Primitive('stop_gradient')
stop_gradient_p.def_impl(_stop_gradient_impl)
stop_gradient_p.def_abstract_eval(lambda x: x)
