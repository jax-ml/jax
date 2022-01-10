from dataclasses import dataclass
from functools import partial
import operator
from typing import (Any, Type, Callable, List, Tuple, Dict, Union, Optional,
                    Hashable)

import jax
import jax.numpy as jnp
import jaxlib
from jax import core
from jax import tree_util
from jax.util import safe_map, safe_zip
from jax.interpreters import batching
from jax.interpreters import xla

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Elt = Any
EltTy = Any
Array = Union[jnp.ndarray, core.Tracer]  # "JaxprType"?

# General-element-array (Garray) data type

class Garray:
  data: List[Array]
  etype: EltTy
  shape: Tuple[int]

  def __init__(self, data, etype, shape):
    self.data = data
    self.etype = etype
    self.shape = shape

  ndim = property(lambda self: len(self.shape))

  def __repr__(self) -> str:
    return f'Garray[{self.shape} ; {self.etype}]'

  def __getitem__(self, idx):
    if isinstance(idx, int):
      idx = (idx,)
    if not (isinstance(idx, tuple) and all(type(i) is int for i in idx)):
      raise NotImplementedError
    new_data = [x[idx] for x in self.data]
    new_shape = self.shape[len(idx):]
    return Garray(new_data, self.etype, new_shape)

  def __neg__(self): return _garray_unop(operator.neg, self)
  def __abs__(self): return _garray_unop(operator.abs, self)

  def __add__ (self, other): return _garray_binop(operator.add, self, other)
  def __sub__ (self, other): return _garray_binop(operator.sub, self, other)
  def __mul__ (self, other): return _garray_binop(operator.mul, self, other)
  def __truediv__ (self, other): return _garray_binop(operator.truediv, self, other)


# Register Garray as jittable

def _flatten_for_jit(x: Garray) -> Tuple[List[Array], Hashable]:
  return x.data, (x.etype, x.shape)
def _unflatten_for_jit(data: List[Array], meta: Hashable) -> Garray:
  assert isinstance(meta, tuple)
  etype, shape = meta
  return Garray(data, etype, shape)
xla.register_jittable(Garray, _flatten_for_jit, _unflatten_for_jit)

# Register Garray as vmappable

def _index(cont, i, x, axis):
  new_data = [cont(i, a, axis) for a in x.data]
  new_shape = _tuple_pop(x.shape, axis)
  return Garray(new_data, etype=x.etype, shape=new_shape)

def _build(cont, axis_size, x, axis):
  new_data = [cont(axis_size, a, axis) for a in x.data]
  new_shape = _tuple_insert(x.shape, axis_size, axis)
  return Garray(new_data, etype=x.etype, shape=new_shape)

def _tuple_pop(tup: Tuple, idx: Optional[int]) -> Tuple:
  if idx is None:
    return tup
  return tuple(x for i, x in enumerate(tup) if i != idx)

def _tuple_insert(tup: Tuple, val: Any, idx: Optional[int]) -> Tuple:
  if idx is None:
    return tup
  lst = list(tup)
  lst.insert(idx, val)
  return tuple(lst)

batching.register_vmappable(Garray, int, int, _index, _build, None)

# The API functions, other than vmap, convert between elements and rank0 arrays

def elt_to_rank0_garray(e: Elt) -> Garray:
  data, etype = flatten_elt(e)
  return Garray(data, etype, ())

def rank0_garray_to_elt(x: Garray) -> Elt:
  gelt_spec = unflatteners[type(x.etype)]
  return gelt_spec.unflatten(x.data, x.etype)  # type: ignore

# Garrays are a way to present an array-of-structures API while maintaining an
# efficient structure-of-arrays representation. So to support a type as an
# element type, we need to know how to flatten it. (This is like the PyTree
# typeclass, but we keep them distinct because e.g. we may not want tree_util
# functions to see these element types as containers, and some pytrees may not
# be appropriate as element types.)

def is_jaxpr_type(x):
  try: core.get_aval(x)
  except TypeError: return False
  else: return True

def flatten_elt(e: Elt) -> Tuple[List[Array], EltTy]:
  gelt_spec = gelts.get(type(e))
  if gelt_spec:
    arrays, etype = gelt_spec.flatten(e)  # type: ignore
    assert isinstance(arrays, list) and all(map(is_jaxpr_type, arrays))
    return arrays, etype
  else:
    raise TypeError(f'type {type(e)} is not registered as a general element '
                    f'type, and it is not a jaxpr type.')

Flattener = Callable[[Elt], Tuple[List[Array], EltTy]]
Unflattener = Callable[[List[Array], EltTy], Elt]

def register_garray_elt(
    ty: Type, ety: Type, flatten: Flattener, unflatten: Unflattener) -> None:
  gelts[ty] = unflatteners[ety] = GeltSpec(ty, ety, flatten, unflatten)

@dataclass(frozen=True)
class GeltSpec:
  ty: Type
  etype: Type
  flatten: Flattener
  unflatten: Unflattener
gelts: Dict[Type, GeltSpec] = {}
unflatteners: Dict[Type, GeltSpec] = {}

# All jaxpr types can be element types.

@dataclass(frozen=True)
class JaxprTy:
  aval: core.AbstractValue
  def __repr__(self) -> str:
    if isinstance(self.aval, core.UnshapedArray):
      return self.aval.str_short()
    else:
      return str(self.aval)
def _flatten_jaxpr_type(x: Array) -> Tuple[List[Array], JaxprTy]:
  assert is_jaxpr_type(x)
  return [x], JaxprTy(core.raise_to_shaped(core.get_aval(x)))
def _unflatten_jaxpr_type(xs: List[Array], ty: JaxprTy) -> Array:
  del ty
  x, = xs
  return x
register_garray_elt(jaxlib.xla_extension.DeviceArray,
                    JaxprTy, _flatten_jaxpr_type, _unflatten_jaxpr_type)
register_garray_elt(core.Tracer,
                    JaxprTy, _flatten_jaxpr_type, _unflatten_jaxpr_type)

# Garrays themselves can be element types.

@dataclass(frozen=True)
class GarrayTy:
  shape: Tuple[int]
  etype: EltTy
  def __repr__(self) -> str:
    return f'GarrayTy[{self.shape} ; {self.etype}]'
def _flatten_garray(x: Garray) -> Tuple[List[Array], GarrayTy]:
  return x.data, GarrayTy(x.shape, x.etype)
def _unflatten_garray(xs: List[Array], ty: GarrayTy) -> Garray:
  return Garray(xs, ty.etype, ty.shape)
register_garray_elt(Garray, GarrayTy,
                    _flatten_garray, _unflatten_garray)

# Standard pytrees can be element types.

PyTree = Any
@dataclass(frozen=True)
class PyTreeEtype:
  name: str
  treedef: tree_util.PyTreeDef
  def __repr__(self) -> str:
    return f'{self.name}'
def _pytree_flattener(x: PyTree) -> Tuple[List[Array], PyTreeEtype]:
  leaves, treedef = tree_util.tree_flatten(x)
  name = getattr(type(x), '__name__', str(type(x)))
  return leaves, PyTreeEtype(name, treedef)
def _pytree_unflattener(xs: List[Array], ty: PyTreeEtype) -> PyTree:
  return tree_util.tree_unflatten(ty.treedef, xs)
def register_garray_elt_from_pytree(ty: Type) -> None:
  register_garray_elt(ty, PyTreeEtype, _pytree_flattener, _pytree_unflattener)
register_garray_elt_from_pytree(list)
register_garray_elt_from_pytree(tuple)
register_garray_elt_from_pytree(dict)

# We can lift operations supported by element types to operations on Garrays.

Unop = Callable[[Elt], Elt]
def _rank0_garray_unop(op: Unop, x: Garray) -> Garray:
  x_ = rank0_garray_to_elt(x)
  y_ = op(x_)
  return elt_to_rank0_garray(y_)

def _garray_unop(op: Unop, x: Garray) -> Garray:
  f = partial(_rank0_garray_unop, op)
  for _ in range(x.ndim):
    f = jax.vmap(f)
  return f(x)

Binop = Callable[[Elt, Elt], Elt]
def _rank0_garray_binop(op: Binop, x: Garray, y: Garray) -> Garray:
  x_ = rank0_garray_to_elt(x)
  y_ = rank0_garray_to_elt(y)
  z_ = op(x_, y_)
  return elt_to_rank0_garray(z_)

def _garray_binop(op: Binop, x: Garray, y: Any) -> Garray:
  if not isinstance(y, Garray): raise TypeError
  if x.shape != y.shape: raise NotImplementedError
  f = partial(_rank0_garray_binop, op)
  for _ in range(x.ndim):
    f = jax.vmap(f)
  return f(x, y)
