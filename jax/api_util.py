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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .core import pack
from .tree_util import (build_tree, process_pytree, tree_flatten,
                        tree_unflatten, leaf)
from .linear_util import transformation_with_aux
from .util import safe_map, unzip2, partial, curry

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

@transformation_with_aux
def pytree_fun_to_jaxtupletree_fun(args_trees, *args):
  py_args = map(build_tree, args_trees, args)
  ans = yield py_args, {}
  yield pytree_to_jaxtupletree(ans)

@transformation_with_aux
def pytree_fun_to_jaxtupletree_fun2(kwargs_tree, args_trees, kwargs, *args):
  py_args = map(build_tree, args_trees, args)
  py_kwargs = build_tree(kwargs_tree, kwargs)
  ans = yield py_args, py_kwargs
  yield pytree_to_jaxtupletree(ans)

def apply_jaxtree_fun(fun, io_tree, *py_args):
  in_trees_expected, out_tree = io_tree
  args, in_trees = unzip2(map(pytree_to_jaxtupletree, py_args))
  for i, (in_tree, expected) in enumerate(zip(in_trees, in_trees_expected)):
    if in_tree != expected:
      raise TypeError("Expected {}, got {}".format(expected, in_tree))

  ans = fun(*args)
  return build_tree(out_tree, ans)

pytree_to_jaxtupletree = partial(process_pytree, pack)


@transformation_with_aux
def pytree_fun_to_flatjaxtuple_fun(in_trees, *args):
  py_args = map(tree_unflatten, in_trees, args)
  ans = yield py_args, {}
  yield pytree_to_flatjaxtuple(ans)

@transformation_with_aux
def flatten_fun(in_tree, *args_flat):
  py_args, py_kwargs = tree_unflatten(in_tree, args_flat)
  ans = yield py_args, py_kwargs
  yield pytree_to_flatjaxtuple(ans)

def pytree_to_flatjaxtuple(pytree):
  flat, out_tree = tree_flatten(pytree)
  return pack(flat), out_tree


@transformation_with_aux
def flatten_fun_leafout(in_tree, *args_flat):
  # like flatten_fun but doesn't pack output leaves
  py_args, py_kwargs = tree_unflatten(in_tree, args_flat)
  ans = yield py_args, py_kwargs
  flat_ans, out_tree = tree_flatten(ans)
  if out_tree is leaf:
    yield ans, out_tree
  else:
    yield pack(flat_ans), out_tree
