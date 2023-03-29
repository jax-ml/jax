# Copyright 2018 The JAX Authors.
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
from __future__ import annotations

import types
from typing import Any, Callable, Dict, TypeVar, Union, cast

from jax._src import core
from jax._src import traceback_util
from jax._src.core import (lattice_join, Primitive, valid_jaxtype,
                           raise_to_shaped, get_aval)
from jax._src.tree_util import register_pytree_node
from jax._src.typing import Array, ArrayLike
from jax._src.util import safe_map

traceback_util.register_exclusion(__file__)

T = TypeVar('T')

map = safe_map

jaxval_adders: Dict[type, Callable[[ArrayLike, ArrayLike], Array]] = {}

def add_jaxvals(x: ArrayLike, y: ArrayLike) -> Array:
  return add_jaxvals_p.bind(x, y)

add_jaxvals_p: Primitive = Primitive('add_any')
add_any_p = add_jaxvals_p

@add_jaxvals_p.def_impl
def add_impl(xs, ys):
  return jaxval_adders[type(xs)](xs, ys)

@add_jaxvals_p.def_abstract_eval
def add_abstract(xs, ys):
  return lattice_join(xs, ys)

jaxval_zeros_likers: Dict[type, Callable[[Any], Array]] = {}

def instantiate(z: Union[Zero, Array]) -> Array:
  if type(z) is Zero:
    return zeros_like_aval(z.aval)
  return cast(Array, z)

def zeros_like_aval(aval: core.AbstractValue) -> Array:
  return aval_zeros_likers[type(aval)](aval)

aval_zeros_likers: Dict[type, Callable[[Any], Array]] = {}

def zeros_like_jaxval(val: ArrayLike) -> Array:
  return zeros_like_p.bind(val)

zeros_like_p: Primitive = Primitive('zeros_like')

@zeros_like_p.def_impl
def zeros_like_impl(example):
  return jaxval_zeros_likers[type(example)](example)

zeros_like_p.def_abstract_eval(lambda x: x)

class Zero:
  __slots__ = ['aval']
  def __init__(self, aval: core.AbstractValue):
    self.aval = aval
  def __repr__(self) -> str:
    return f'Zero({self.aval})'
  @staticmethod
  def from_value(val: Any) -> Zero:
    return Zero(raise_to_shaped(get_aval(val)))

register_pytree_node(Zero, lambda z: ((), z.aval), lambda aval, _: Zero(aval))


def _stop_gradient_impl(x: T) -> T:
  if not valid_jaxtype(x):
    raise TypeError("stop_gradient only works on valid JAX arrays, but "
                    f"input argument is: {x}")
  return x

stop_gradient_p : Primitive = Primitive('stop_gradient')
stop_gradient_p.def_impl(_stop_gradient_impl)
stop_gradient_p.def_abstract_eval(lambda x: x)


class SymbolicZero:
  def __init__(self, aval: core.AbstractValue) -> None:
    self.aval = aval

  def __repr__(self) -> str:
    return self.__class__.__name__

  # TODO(mattjj,frostig): this forwards attr lookup to self.aval delegate;
  # should dedup with core.Tracer.__getattr__ which does the same thing
  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    try:
      attr = getattr(self.aval, name)
    except KeyError as err:
      raise AttributeError(
          f"{self.__class__.__name__} has no attribute {name}"
      ) from err
    else:
      t = type(attr)
      if t is core.aval_property:
        return attr.fget(self)
      elif t is core.aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

JaxTypeOrTracer = Any

def replace_internal_symbolic_zeros(
    x: Union[JaxTypeOrTracer, Zero]) -> Union[JaxTypeOrTracer, SymbolicZero]:
  return SymbolicZero(x.aval) if type(x) is Zero else x

def replace_rule_output_symbolic_zeros(
    x: Union[JaxTypeOrTracer, SymbolicZero]) -> Union[JaxTypeOrTracer, Zero]:
  return Zero(x.aval) if type(x) is SymbolicZero else x
