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

import inspect
import operator
from functools import partial
from typing import Any, Dict, Iterable, Sequence, Set, Tuple, Union, Optional
import warnings

import numpy as np

from jax import core
from jax._src import dtypes
from jax._src.tree_util import (
    PyTreeDef, tree_flatten, tree_unflatten, tree_map, tree_structure,
    treedef_children, treedef_is_leaf)
from jax._src.tree_util import _replace_nones
from jax import linear_util as lu
from jax._src.util import safe_map, WrapKwArgs, Hashable, Unhashable

from jax._src import traceback_util
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
    raise TypeError(f"Expected {in_tree_expected}, got {in_tree}")
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
    raise TypeError(f"Expected {in_tree_expected}, got {in_tree}")
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)

def flattened_fun_in_tree(fn: lu.WrappedFun) -> Optional[Tuple[PyTreeDef, bool]]:
  # This implementation relies on internal details of linear_util.py's
  # WrappedFun, but it's for the worthy cause of better user error messages.
  # It can fail (i.e. return None) if its WrappedFun argument is not transformed
  # with flatten_fun or flatten_fun_nokwargs, which could happen e.g. when
  # core.eval_jaxpr encounters a call primitive (though at that point we're just
  # round-tripping jaxprs and the user errors in question are impossible).
  assert isinstance(flatten_fun, partial) and len(flatten_fun.args) == 1
  assert (isinstance(flatten_fun_nokwargs, partial) and
          len(flatten_fun_nokwargs.args) == 1)
  flat_xforms = {flatten_fun.args[0], flatten_fun_nokwargs.args[0]}
  try:
    (in_tree, has_kwargs), = ((args[0], f is flatten_fun.args[0])
                              for f, args in fn.transforms if f in flat_xforms)
  except ValueError:
    return None
  else:
    return in_tree, has_kwargs

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

class _HashableWithStrictTypeEquality:
  """Box object used when comparing static arguments as a jit key.

  Requires exact type equality using `is` and value equality."""
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return type(self.val) is type(other.val) and self.val == other.val

_POSITIONAL_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_ONLY,
  inspect.Parameter.POSITIONAL_OR_KEYWORD
)

def validate_argnums(sig: inspect.Signature, argnums: Tuple[int, ...], argnums_name: str) -> None:
  """
  Validate that the argnums are sensible for a given function.

  For functions that accept a variable number of positions arguments
  (`f(..., *args)`) all positive argnums are considered valid.
  """
  n_pos_args = 0
  for param in sig.parameters.values():
    if param.kind in _POSITIONAL_ARGUMENTS:
      n_pos_args += 1

    elif param.kind is inspect.Parameter.VAR_POSITIONAL:
      # We can have any number of positional arguments
      return

  if argnums and (-min(argnums) > n_pos_args or max(argnums) >= n_pos_args):
    # raise ValueError(f"Jitted function has {argnums_name}={argnums}, "
    #                  f"but only accepts {n_pos_args} positional arguments.")
    # TODO: 2022-08-20 or later: replace with error
    warnings.warn(f"Jitted function has {argnums_name}={argnums}, "
                  f"but only accepts {n_pos_args} positional arguments. "
                  "This warning will be replaced by an error after 2022-08-20 "
                  "at the earliest.", SyntaxWarning)

_INVALID_KEYWORD_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_ONLY,
  inspect.Parameter.VAR_POSITIONAL
)

_KEYWORD_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_OR_KEYWORD,
  inspect.Parameter.KEYWORD_ONLY,
)
def validate_argnames(sig: inspect.Signature, argnames: Tuple[str, ...], argnames_name: str) -> None:
  """
  Validate that the argnames are sensible for a given function.

  For functions that accept a variable keyword arguments
  (`f(..., **kwargs)`) all argnames are considered valid except those
  marked as position-only (`f(pos_only, /, ...)`).
  """
  var_kwargs = False
  valid_kwargs: Set[str] = set()
  invalid_kwargs: Set[str] = set()
  for param_name, param in sig.parameters.items():
    if param.kind in _KEYWORD_ARGUMENTS:
      valid_kwargs.add(param_name)

    elif param.kind is inspect.Parameter.VAR_KEYWORD:
      var_kwargs = True

    elif param.kind in _INVALID_KEYWORD_ARGUMENTS:
      invalid_kwargs.add(param_name)


  # Check whether any kwargs are invalid due to position only
  invalid_argnames = invalid_kwargs & set(argnames)
  if invalid_argnames:
    # raise ValueError(f"Jitted function has invalid argnames {invalid_argnames} "
    #                  f"in {argnames_name}. These are positional-only")
    # TODO: 2022-08-20 or later: replace with error
    warnings.warn(f"Jitted function has invalid argnames {invalid_argnames} "
                  f"in {argnames_name}. These are positional-only. "
                  "This warning will be replaced by an error after 2022-08-20 "
                  "at the earliest.", SyntaxWarning)

  # Takes any kwargs
  if var_kwargs:
    return

  # Check that all argnames exist on function
  invalid_argnames = set(argnames) - valid_kwargs
  if invalid_argnames:
    # TODO: 2022-08-20 or later: replace with error
    # raise ValueError(f"Jitted function has invalid argnames {invalid_argnames} "
    #                  f"in {argnames_name}. Function does not take these args.")
    warnings.warn(f"Jitted function has invalid argnames {invalid_argnames} "
                  f"in {argnames_name}. Function does not take these args."
                  "This warning will be replaced by an error after 2022-08-20 "
                  "at the earliest.", SyntaxWarning)



def argnums_partial(f, dyn_argnums, args, require_static_args_hashable=True):
  dyn_argnums = _ensure_index_tuple(dyn_argnums)
  dyn_argnums = _ensure_inbounds(False, len(args), dyn_argnums)
  if require_static_args_hashable:
    fixed_args = []
    for i, arg in enumerate(args):
      if i in dyn_argnums: continue
      if not is_hashable(arg):
        raise ValueError(
            "Non-hashable static arguments are not supported, as this can lead "
            f"to unexpected cache-misses. Static argument (index {i}) of type "
            f"{type(arg)} for function {f.__name__} is non-hashable.")
      fixed_args.append(_HashableWithStrictTypeEquality(arg))
  else:
    fixed_args = [Unhashable(arg) for i, arg in enumerate(args)
                  if i not in dyn_argnums]
  dyn_args = tuple(args[i] for i in dyn_argnums)
  return _argnums_partial(f, dyn_argnums, tuple(fixed_args)), dyn_args

def _ensure_inbounds(allow_invalid: bool, num_args: int, argnums: Sequence[int]
                     ) -> Tuple[int, ...]:
  """
  Ensure argnum is within bounds.

  Also resolves negative argnums
  """
  result = []
  for i in argnums:
    if i >= num_args and allow_invalid: continue
    if not -num_args <= i < num_args:
      raise ValueError(
          "Positional argument indices, e.g. for `static_argnums`, must have "
          "value greater than or equal to -len(args) and less than len(args), "
          f"but got value {i} for len(args) == {num_args}.")
    result.append(i % num_args)  # Resolve negative
  return tuple(result)


def argnums_partial_except(f: lu.WrappedFun, static_argnums: Tuple[int, ...],
                           args: Tuple[Any], *, allow_invalid: bool):
  """Version of ``argnums_partial`` that checks hashability of static_argnums."""
  if not static_argnums:
    return f, args
  static_argnums = _ensure_inbounds(allow_invalid, len(args), static_argnums)
  dyn_argnums = tuple(i for i in range(len(args)) if i not in static_argnums)
  dyn_args = tuple(args[i] for i in dyn_argnums)

  fixed_args = []
  for i in static_argnums:
    # TODO(shoyer): set allow_invalid=True permanently after static_argnames.
    if allow_invalid and i >= len(args):
      continue
    static_arg = args[i]
    if not is_hashable(static_arg):
      raise ValueError(
          "Non-hashable static arguments are not supported, as this can lead "
          f"to unexpected cache-misses. Static argument (index {i}) of type "
          f"{type(static_arg)} for function {f.__name__} is non-hashable.")
    else:
      fixed_args.append(_HashableWithStrictTypeEquality(static_arg))  # type: ignore

  return _argnums_partial(f, dyn_argnums, tuple(fixed_args)), dyn_args

@lu.transformation
def _argnums_partial(dyn_argnums, fixed_args, *dyn_args, **kwargs):
  sentinel = object()
  args = [sentinel] * (len(fixed_args) + len(dyn_args))
  for i, arg in zip(dyn_argnums, dyn_args):
    args[i] = arg
  fixed_args_ = iter(fixed_args)
  args = [next(fixed_args_).val if x is sentinel else x for x in args]
  assert next(fixed_args_, sentinel) is sentinel
  ans = yield args, kwargs
  yield ans


def argnames_partial_except(f: lu.WrappedFun, static_argnames: Tuple[str, ...],
                            kwargs: Dict[str, Any]):
  if not static_argnames:
    return f, kwargs
  dyn_kwargs = {k: v for k, v in kwargs.items() if k not in static_argnames}

  fixed_kwargs: Dict[str, Any] = {}
  for k, arg in kwargs.items():
    if k not in dyn_kwargs:
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
  kwargs = dict({k: v.val for k, v in fixed_kwargs.val.items()}, **dyn_kwargs)
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


def is_hashable(arg):
  try:
    hash(arg)
    return True
  except TypeError:
    return False


def flatten_axes(name, treedef, axis_tree, *, kws=False, tupled_args=False):
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
    tree_map(add_leaves, _replace_nones(proxy, axis_tree), dummy)
  except ValueError:
    if kws:
      # if keyword arguments are included in the tree, we make adapt the error
      # message only to be about the positional arguments
      treedef, leaf = treedef_children(treedef)
      assert treedef_is_leaf(leaf)
      axis_tree, _ = axis_tree
    hint = ""
    if tupled_args:
      hint += (f" Note that {name} that are non-trivial pytrees should always be "
               f"wrapped in a tuple representing the argument list.")
      if len(treedef.children()) == 1:
        try:
          flatten_axes(name, treedef, (axis_tree,))
        except ValueError:
          pass  # That's not the issue.
        else:
          hint += (f" In particular, you're passing in a single argument which "
                   f"means that {name} might need to be wrapped in "
                   f"a singleton tuple.")
    raise ValueError(f"{name} specification must be a tree prefix of the "
                     f"corresponding value, got specification {axis_tree} "
                     f"for value tree {treedef}.{hint}") from None
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
    return core.raise_to_shaped(
      x if isinstance(x, core.AbstractValue) else core.get_aval(x))
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
