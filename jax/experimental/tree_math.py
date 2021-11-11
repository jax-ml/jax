# Copyright 2021 Google LLC
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

import functools
import operator
import typing
from typing import Tuple

from jax import tree_util
from jax._src import api
from jax._src.tree_util import tree_unflatten
from jax._src.util import prod
import jax.numpy as jnp


def _flatten_together(*args):
  """Flatten a collection of pytrees with matching structure/shapes together."""
  all_values, all_treedefs = zip(*map(tree_util.tree_flatten, args))
  all_treedefs = typing.cast(Tuple[tree_util.PyTreeDef, ...], all_treedefs)

  if not all(treedef == all_treedefs[0] for treedef in all_treedefs[1:]):
    treedefs_str = ' vs '.join(map(str, all_treedefs))
    raise ValueError(
        f"arguments have different tree structures: {treedefs_str}"
    )

  all_shapes = [list(map(jnp.shape, values)) for values in all_values]
  if not all(shapes == all_shapes[0] for shapes in all_shapes[1:]):
    shapes_str = ' vs '.join(map(str, all_shapes))
    raise ValueError(f"tree leaves have different array shapes: {shapes_str}")

  return all_values, all_treedefs[0]


def _argnums_partial(f, args, static_argnums):
  def g(*args3):
    args3 = list(args3)
    for i in static_argnums:
      args3.insert(i, args[i])
    return f(*args3)
  args2 = tuple(x for i, x in enumerate(args) if i not in static_argnums)
  return g, args2


def _broadcasting_map(func, *args):
  """Like tree_map, but scalar arguments are broadcast to all leaves."""
  static_argnums = [i for i, x in enumerate(args) if not isinstance(x, Vector)]
  func2, vector_args = _argnums_partial(func, args, static_argnums)
  for arg in args:
    if not isinstance(arg, Vector):
      shape = jnp.shape(arg)
      if shape != ():
        raise TypeError(
            f"non-tree_math.Vector argument is not a scalar: {arg!r}"
        )
  if not vector_args:
    return func2()  # result is a scalar
  _flatten_together(*[arg.tree for arg in vector_args])  # check shapes
  return tree_util.tree_map(func2, *vector_args)


def _binary_method(func, name):
  """Implement a forward binary method, e.g., __add__."""
  def wrapper(self, other):
    return _broadcasting_map(func, self, other)
  wrapper.__name__ = f'__{name}__'
  return wrapper


def _reflected_binary_method(func, name):
  """Implement a reflected binary method, e.g., __radd__."""
  def wrapper(self, other):
    return _broadcasting_map(func, other, self)
  wrapper.__name__ = f'__r{name}__'
  return wrapper


def _numeric_methods(func, name):
  """Implement forward and reflected methods."""
  return (_binary_method(func, name), _reflected_binary_method(func, name))


def _unary_method(func, name):
  def wrapper(self):
    return tree_util.tree_map(func, self)
  wrapper.__name__ = f'__{name}__'
  return wrapper


def matmul(left, right, *, precision=None):
  if not isinstance(left, Vector) or not isinstance(right, Vector):
    raise TypeError("matmul arguments must both be tree_math.Vector objects")

  def _vector_dot(a, b):
    return jnp.dot(jnp.ravel(a), jnp.ravel(b), precision=precision)

  (left_values, right_values), _ = _flatten_together(left.tree, right.tree)
  parts = map(_vector_dot, left_values, right_values)
  return functools.reduce(operator.add, parts)


@tree_util.register_pytree_node_class
class Vector:
  """A wrapper for treating an arbitrary pytree as a 1D vector."""
  def __init__(self, tree):
    self._tree = tree

  @property
  def tree(self):
    return self._tree

  # TODO(shoyer): consider casting to a common dtype?

  def __repr__(self):
    return f'tree_math.Vector({self._tree!r})'

  def tree_flatten(self):
    return (self.tree,), None

  @classmethod
  def tree_unflatten(cls, _, args):
    return cls(*args)

  def __len__(self):
    values = tree_util.tree_leaves(self.tree)
    return sum(prod(jnp.shape(value)) for value in values)

  @property
  def shape(self):
    return (len(self),)

  @property
  def ndim(self):
    return 1

  @property
  def dtype(self):
    values = tree_util.tree_leaves(self.tree)
    return jnp.result_type(*values)

  # comparison
  __lt__ = _binary_method(operator.lt, 'lt')
  __le__ = _binary_method(operator.le, 'le')
  __eq__ = _binary_method(operator.eq, 'eq')
  __ne__ = _binary_method(operator.ne, 'ne')
  __ge__ = _binary_method(operator.ge, 'ge')
  __gt__ = _binary_method(operator.gt, 'gt')

  # arithmetic
  __add__, __radd__ = _numeric_methods(operator.add, 'add')
  __sub__, __rsub__ = _numeric_methods(operator.sub, 'sub')
  __mul__, __rmul__ = _numeric_methods(operator.mul, 'mul')
  __truediv__, __rtruediv__ = _numeric_methods(operator.truediv, 'truediv')
  __floordiv__, __rfloordiv__ = _numeric_methods(operator.floordiv, 'floordiv')
  __mod__, __rmod__ = _numeric_methods(operator.mod, 'mod')
  __pow__, __rpow__ = _numeric_methods(operator.pow, 'pow')
  __matmul__ = __rmatmul__ = matmul

  # TODO(shoyer): implement this via divmod() on the leaves
  def __divmod__(self, other):
    return self // other, self % other
  def __rdivmod__(self, other):
    return other // self, other % self

  # bitwise
  __lshift__, __rlshift__ = _numeric_methods(operator.lshift, 'lshift')
  __rshift__, __rrshift__ = _numeric_methods(operator.rshift, 'rshift')
  __and__, __rand__ = _numeric_methods(operator.and_, 'and')
  __xor__, __rxor__ = _numeric_methods(operator.xor, 'xor')
  __or__, __ror__ = _numeric_methods(operator.or_, 'or')

  # unary methods
  __neg__ = _unary_method(operator.neg, 'neg')
  __pos__ = _unary_method(operator.pos, 'pos')
  __abs__ = _unary_method(abs, 'abs')
  __invert__ = _unary_method(operator.invert, 'invert')

  # numpy methods
  conj = _unary_method(jnp.conj, 'conj')

  def sum(self):
    parts = map(jnp.sum, tree_util.tree_leaves(self))
    return functools.reduce(operator.add, parts)

  def mean(self):
    return self.sum() / len(self)

  def min(self):
    parts = map(jnp.min, tree_util.tree_leaves(self))
    return jnp.asarray(list(parts)).min()

  def max(self):
    parts = map(jnp.max, tree_util.tree_leaves(self))
    return jnp.asarray(list(parts)).max()


def where(condition, x, y):
  """Tree math compatible version of jnp.where."""
  return _broadcasting_map(jnp.where, condition, x, y)


zeros_like = functools.partial(tree_util.tree_map, jnp.zeros_like)
ones_like = functools.partial(tree_util.tree_map, jnp.ones_like)

def full_like(x, *args, **kwargs):
  return tree_util.tree_map(lambda y: jnp.full_like(y, *args, **kwargs), x)

maximum = functools.partial(_broadcasting_map, jnp.maximum)
minimum = functools.partial(_broadcasting_map, jnp.minimum)
square = functools.partial(_broadcasting_map, jnp.square)


def _infer_argnums_and_argnames(fun, argnums, argnames):
  if argnums is None and argnames is None:
    return None, None  # wrap all arguments
  return api._infer_argnums_and_argnames(fun, argnums, argnames)


def _apply_argnums(wrapper, args, argnums):
  return tuple(wrapper(arg) if argnums is None or i in argnums else arg
               for i, arg in enumerate(args))


def _apply_argnames(wrapper, kwargs, argnames):
  return {k: wrapper(arg) if argnames is None or k in argnames else arg
          for k, arg in kwargs.items()}



def _maybe_get_tree(arg):
  return arg.tree if isinstance(arg, Vector) else arg


def _is_vector(arg):
  return isinstance(arg, Vector)


def wrap(fun, vector_argnums=None, vector_argnames=None):
  """Convert a vector -> vector function to a pytree -> pytree function."""
  vector_argnums, vector_argnames = _infer_argnums_and_argnames(
      fun, vector_argnums, vector_argnames)
  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    args = _apply_argnums(Vector, args, vector_argnums)
    kwargs = _apply_argnames(Vector, kwargs, vector_argnames)
    result = fun(*args, **kwargs)
    return tree_util.tree_map(_maybe_get_tree, result, is_leaf=_is_vector)
  return wrapper


def _get_tree(vector):
  return vector.tree


def _maybe_vector(condition, arg):
  return Vector(arg) if condition else arg


def unwrap(fun, vector_argnums=None, vector_argnames=None, out_vectors=True):
  """Convert a pytree -> pytree function to a vector -> vector function."""
  vector_argnums, vector_argnames = _infer_argnums_and_argnames(
      fun, vector_argnums, vector_argnames)
  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    args = _apply_argnums(_get_tree, args, vector_argnums)
    kwargs = _apply_argnames(_get_tree, kwargs, vector_argnames)
    result = fun(*args, **kwargs)
    return tree_util.tree_map(_maybe_vector, out_vectors, result)
  return wrapper
