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
from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, TypeVar, overload

from jax._src import tree_util

T = TypeVar("T")


def _add_doc(docstr):
  def wrapper(fun):
    doc = fun.__doc__
    firstline, rest = doc.split('\n', 1)
    fun.__doc__ = f'{firstline}\n\n  {docstr}\n{rest}'
    return fun
  return wrapper


@_add_doc("Alias of :func:`jax.tree_util.tree_all`.")
@functools.wraps(tree_util.tree_all)
def all(tree: Any) -> bool:
  return tree_util.tree_all(tree)


@_add_doc("Alias of :func:`jax.tree_util.tree_flatten`.")
@functools.wraps(tree_util.tree_flatten)
def flatten(tree: Any,
            is_leaf: Callable[[Any], bool] | None = None
            ) -> tuple[list[tree_util.Leaf], tree_util.PyTreeDef]:
  return tree_util.tree_flatten(tree, is_leaf)


@_add_doc("Alias of :func:`jax.tree_util.tree_leaves`.")
@functools.wraps(tree_util.tree_leaves)
def leaves(tree: Any,
           is_leaf: Callable[[Any], bool] | None = None
           ) -> list[tree_util.Leaf]:
  return tree_util.tree_leaves(tree, is_leaf)


@_add_doc("Alias of :func:`jax.tree_util.tree_map`.")
@functools.wraps(tree_util.tree_map)
def map(f: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None) -> Any:
  return tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)


@overload
def reduce(function: Callable[[T, Any], T],
           tree: Any,
           *,
           is_leaf: Callable[[Any], bool] | None = None) -> T:
    ...
@overload
def reduce(function: Callable[[T, Any], T],
           tree: Any,
           initializer: T,
           is_leaf: Callable[[Any], bool] | None = None) -> T:
    ...
@_add_doc("Alias of :func:`jax.tree_util.tree_reduce`.")
@functools.wraps(tree_util.tree_reduce)
def reduce(function: Callable[[T, Any], T],
           tree: Any,
           initializer: Any = tree_util.no_initializer,
           is_leaf: Callable[[Any], bool] | None = None) -> T:
  return tree_util.tree_reduce(function, tree, initializer, is_leaf=is_leaf)


@_add_doc("Alias of :func:`jax.tree_util.tree_structure`.")
@functools.wraps(tree_util.tree_structure)
def structure(tree: Any,
              is_leaf: None | (Callable[[Any], bool]) = None) -> tree_util.PyTreeDef:
  return tree_util.tree_structure(tree, is_leaf)


@_add_doc("Alias of :func:`jax.tree_util.tree_transpose`.")
@functools.wraps(tree_util.tree_transpose)
def transpose(outer_treedef: tree_util.PyTreeDef,
              inner_treedef: tree_util.PyTreeDef,
              pytree_to_transpose: Any) -> Any:
  return tree_util.tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose)


@_add_doc("Alias of :func:`jax.tree_util.tree_unflatten`.")
@functools.wraps(tree_util.tree_unflatten)
def unflatten(treedef: tree_util.PyTreeDef,
              leaves: Iterable[tree_util.Leaf]) -> Any:
  return tree_util.tree_unflatten(treedef, leaves)
