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

import operator
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np

from . import core
from ._src import dtypes
from .tree_util import (tree_flatten, tree_unflatten, tree_multimap,
                        tree_structure, treedef_children, treedef_is_leaf)
from ._src.tree_util import _replace_nones
from . import linear_util as lu
from ._src.util import safe_map, WrapHashably, WrapKwArgs, Hashable
from .core import unit

from ._src import traceback_util
traceback_util.register_exclusion(__file__)

map = safe_map

def _ensure_index(x: Any) -> Union[int, Tuple[int, ...]]:
  """Ensure x is either an index or a tuple of indices."""
  try:
    return operator.index(x)
  except TypeError:
    return tuple(map(operator.index, x))

def _ensure_index_tuple(x: Any) -> Tuple[int, ...]:
  """Convert x to a tuple of indices."""
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(map(operator.index, x))

def _ensure_str(x: str) -> str:
  if not isinstance(x, str):
    raise TypeError(f"argument is not a string: {x}")
  return x

def _ensure_str_tuple(x: Union[str, Iterable[str]]) -> Tuple[str, ...]:
  """Convert x to a tuple of strings."""
  if isinstance(x, str):
    return (x,)
  else:
    return tuple(map(_ensure_str, x))

@lu.transformation_with_aux
def flatten_fun(in_tree, *args_flat):
  py_args, py_kwargs = tree_unflatten(in_tree, args_flat)
  ans = yield py_args, py_kwargs
  yield tree_flatten(ans)

def apply_flat_fun(fun, io_tree, *py_args):
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten((py_args, {}))
  if in_tree != in_tree_expected:
    raise TypeError("Expected {}, got {}".format(in_tree_expected, in_tree))
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)

@lu.transformation_with_aux
def flatten_fun_nokwargs(in_tree, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  ans = yield py_args, {}
  yield tree_flatten(ans)

def apply_flat_fun_nokwargs(fun, io_tree, py_args):
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten(py_args)
  if in_tree != in_tree_expected:
    raise TypeError("Expected {}, got {}".format(in_tree_expected, in_tree))
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)

@lu.transformation_with_aux
def flatten_fun_nokwargs2(in_tree, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  pair = yield py_args, {}
  if not isinstance(pair, (list, tuple)) or len(pair) != 2:
    raise TypeError("expected function with aux output to return a two-element "
                    f"tuple, but got type {type(pair)} with value {repr(pair)}")
  ans, aux = pair
  ans_flat, ans_tree = tree_flatten(ans)
  aux_flat, aux_tree = tree_flatten(aux)
  yield (ans_flat, aux_flat), (ans_tree, aux_tree)


def argnums_partial(f, dyn_argnums, args):
  dyn_argnums = _ensure_index_tuple(dyn_argnums)
  fixed_args = tuple(unit if i in dyn_argnums else wrap_hashably(arg)
                     for i, arg in enumerate(args))
  dyn_args = tuple(args[i] for i in dyn_argnums)
  return _argnums_partial(f, dyn_argnums, fixed_args), dyn_args


def argnums_partial_except(f: lu.WrappedFun, static_argnums: Tuple[int, ...],
                           args: Tuple[Any], *, allow_invalid: bool):
  """Version of ``argnums_partial`` that checks hashability of static_argnums."""
  if not static_argnums:
    return f, args
  dyn_argnums = tuple(i for i in range(len(args)) if i not in static_argnums)
  dyn_args = tuple(args[i] for i in dyn_argnums)

  fixed_args = [unit] * len(args)  # type: ignore
  for i in static_argnums:
    # TODO(shoyer): set allow_invalid=True permanently after enabling
    # static_argnames.
    if allow_invalid and i >= len(args):
      continue
    static_arg = args[i]
    try:
      hash(static_arg)
    except TypeError:
      raise ValueError(
          "Non-hashable static arguments are not supported, as this can lead "
          f"to unexpected cache-misses. Static argument (index {i}) of type "
          f"{type(static_arg)} for function {f.__name__} is non-hashable.")
    else:
      fixed_args[i] = Hashable(static_arg)  # type: ignore

  return _argnums_partial(f, dyn_argnums, tuple(fixed_args)), dyn_args


@lu.transformation
def _argnums_partial(dyn_argnums, fixed_args, *dyn_args, **kwargs):
  args = [None if arg is unit else arg.val for arg in fixed_args]
  for i, arg in zip(dyn_argnums, dyn_args):
    args[i] = arg
  ans = yield args, kwargs
  yield ans


def argnames_partial(f, dyn_argnames, kwargs):
  dyn_argnames = _ensure_str_tuple(dyn_argnames)
  fixed_kwargs = tuple((k, unit if k in dyn_argnames else wrap_hashably(v))
                       for k, v in kwargs.items())
  dyn_kwargs = {k: kwargs[k] for k in dyn_argnames}
  return _argnames_partial(f, WrapKwArgs(fixed_kwargs)), dyn_kwargs


def argnames_partial_except(f: lu.WrappedFun, static_argnames: Tuple[str, ...],
                            kwargs: Dict[str, Any]):
  if not static_argnames:
    return f, kwargs
  dyn_kwargs = {k: v for k, v in kwargs.items() if k not in static_argnames}

  fixed_kwargs: Dict[str, Any] = {}
  for k, arg in kwargs.items():
    if k in dyn_kwargs:
      fixed_kwargs[k] = unit
    else:
      try:
        hash(arg)
      except TypeError:
        raise ValueError(
            "Non-hashable static arguments are not supported, as this can lead "
            f"to unexpected cache-misses. Static argument (name {k}) of type "
            f"{type(arg)} for function {f.__name__} is non-hashable.")
      else:
        fixed_kwargs[k] = Hashable(arg)  # type: ignore

  return _argnames_partial(f, WrapKwArgs(fixed_kwargs)), dyn_kwargs


@lu.transformation
def _argnames_partial(fixed_kwargs: WrapKwArgs, *args, **dyn_kwargs):
  kwargs = {k: None if arg is unit else arg.val
            for k, arg in fixed_kwargs.val.items()}
  kwargs.update(dyn_kwargs)
  ans = yield args, kwargs
  yield ans


def donation_vector(donate_argnums, args, kwargs) -> Tuple[bool, ...]:
  """Returns a tuple with a boolean value for each leaf in args."""
  res = []
  for i, arg in enumerate(args):
    donate = bool(i in donate_argnums)
    res.extend((donate,) * tree_structure(arg).num_leaves)
  res.extend((False,) * tree_structure(kwargs).num_leaves)
  return tuple(res)

def rebase_donate_argnums(donate_argnums, static_argnums) -> Tuple[int, ...]:
  """Shifts donate to account for static.

  >>> rebase_donate_argnums((3, 4), (0, 1))
  (1, 2)

  Args:
    donate_argnums: An iterable of ints.
    static_argnums: An iterable of ints.

  Returns:
    A tuple of unique, sorted integer values based on donate_argnums with each
    element offset to account for static_argnums.
  """
  if not (static_argnums or donate_argnums):
    return tuple(sorted(donate_argnums))

  static_argnums = sorted(set(static_argnums))
  donate_argnums = sorted(set(donate_argnums))
  i = j = o = 0
  out = []
  while j < len(donate_argnums):
    if i < len(static_argnums) and static_argnums[i] == donate_argnums[j]:
      raise ValueError(f"`static_argnums` {static_argnums} and "
                       f"`donate_argnums` {donate_argnums} cannot intersect.")

    if i < len(static_argnums) and static_argnums[i] < donate_argnums[j]:
      o += 1
      i += 1
    else:
      out.append(donate_argnums[j] - o)
      j += 1
  return tuple(out)

def wrap_hashably(arg):
  try:
    hash(arg)
  except TypeError:
    return WrapHashably(arg)  # e.g. ndarrays, DeviceArrays
  else:
    return Hashable(arg)

def flatten_axes(name, treedef, axis_tree, *, kws=False):
  # given an axis spec tree axis_tree (a pytree with integers and Nones at the
  # leaves, i.e. the Nones are to be considered leaves) that is a tree prefix of
  # the given treedef, build a complete axis spec tree with the same structure
  # and return the flattened result
  # TODO(mattjj,phawkins): improve this implementation
  proxy = object()
  dummy = tree_unflatten(treedef, [object()] * treedef.num_leaves)
  axes = []
  add_leaves = lambda i, x: axes.extend([i] * len(tree_flatten(x)[0]))
  try:
    tree_multimap(add_leaves, _replace_nones(proxy, axis_tree), dummy)
  except ValueError:
    if kws:
      # if keyword arguments are included in the tree, we make adapt the error
      # message only to be about the positional arguments
      treedef, leaf = treedef_children(treedef)
      assert treedef_is_leaf(leaf)
      axis_tree, _ = axis_tree
    raise ValueError(f"{name} specification must be a tree prefix of the "
                     f"corresponding value, got specification {axis_tree} "
                     f"for value tree {treedef}.") from None
  axes = [None if a is proxy else a for a in axes]
  assert len(axes) == treedef.num_leaves
  return axes

def _dtype(x):
  try:
    return dtypes.result_type(x)
  except ValueError:
    return dtypes.result_type(getattr(x, 'dtype'))

def shaped_abstractify(x):
  try:
    return core.raise_to_shaped(core.get_aval(x))
  except TypeError:
    pass

  weak_type = getattr(x, 'weak_type', False)
  named_shape = getattr(x, 'named_shape', {})
  return core.ShapedArray(np.shape(x), _dtype(x), weak_type=weak_type,
                          named_shape=named_shape)

# This decorator exists to make it easier to monkey-patch APIs in JAX.
# By default it does nothing, but it can be monkey-patched to do other things.
def api_hook(fun, tag: str):
  return fun
