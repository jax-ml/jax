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


from .tree_util import (build_tree, tree_flatten, tree_unflatten,
                        treedef_is_leaf)
from . import linear_util as lu
from .util import safe_map, unzip2, partial, curry, WrapHashably, Hashable
from .core import unit

map = safe_map


@curry
def wraps(wrapped, fun, namestr="{fun}", docstr="{doc}", **kwargs):
  try:
    fun.__name__ = namestr.format(fun=get_name(wrapped))
    fun.__module__ = get_module(wrapped)
    fun.__doc__ = docstr.format(fun=get_name(wrapped), doc=get_doc(wrapped), **kwargs)
    fun.__wrapped__ = wrapped
  finally:
    return fun

def get_name(fun): return getattr(fun, "__name__", "<unnamed function>")
def get_module(fun): return getattr(fun, "__module__", "<unknown module>")
def get_doc(fun): return getattr(fun, "__doc__", "")

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
  ans, aux = yield py_args, {}
  ans_flat, ans_tree = tree_flatten(ans)
  aux_flat, aux_tree = tree_flatten(aux)
  yield (ans_flat, aux_flat), (ans_tree, aux_tree)

def argnums_partial(f, dyn_argnums, args):
  if isinstance(dyn_argnums, int):
    dyn_argnums = (dyn_argnums,)
  else:
    dyn_argnums = tuple(dyn_argnums)
  fixed_args = tuple([unit if i in dyn_argnums else wrap_hashably(arg)
                      for i, arg in enumerate(args)])
  dyn_args = tuple(args[i] for i in dyn_argnums)
  return _argnums_partial(f, dyn_argnums, fixed_args), dyn_args

def wrap_hashably(arg):
  try:
    hash(arg)
  except TypeError:
    return WrapHashably(arg)  # e.g. ndarrays, DeviceArrays
  else:
    return Hashable(arg)

@lu.transformation
def _argnums_partial(dyn_argnums, fixed_args, *dyn_args, **kwargs):
  args = [None if arg is unit else arg.val for arg in fixed_args]
  for i, arg in zip(dyn_argnums, dyn_args):
    args[i] = arg
  ans = yield args, kwargs
  yield ans
