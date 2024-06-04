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

from typing import Any, Callable, Iterable, TypeVar, overload

from jax._src import tree_util

T = TypeVar("T")


def all(tree: Any) -> bool:
  """Alias of :func:`jax.tree_util.tree_all`."""
  return tree_util.tree_all(tree)


def flatten(tree: Any,
            is_leaf: Callable[[Any], bool] | None = None
            ) -> tuple[list[tree_util.Leaf], tree_util.PyTreeDef]:
  """Alias of :func:`jax.tree_util.tree_flatten`."""
  return tree_util.tree_flatten(tree, is_leaf)


def leaves(tree: Any,
           is_leaf: Callable[[Any], bool] | None = None
           ) -> list[tree_util.Leaf]:
  """Alias of :func:`jax.tree_util.tree_leaves`."""
  return tree_util.tree_leaves(tree, is_leaf)


def map(f: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None) -> Any:
  """Alias of :func:`jax.tree_util.tree_map`."""
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
def reduce(function: Callable[[T, Any], T],
           tree: Any,
           initializer: Any = tree_util.no_initializer,
           is_leaf: Callable[[Any], bool] | None = None) -> T:
  """Alias of :func:`jax.tree_util.tree_reduce`."""
  return tree_util.tree_reduce(function, tree, initializer, is_leaf=is_leaf)


def structure(tree: Any,
              is_leaf: None | (Callable[[Any], bool]) = None) -> tree_util.PyTreeDef:
  """Alias of :func:`jax.tree_util.tree_structure`."""
  return tree_util.tree_structure(tree, is_leaf)


def transpose(outer_treedef: tree_util.PyTreeDef,
              inner_treedef: tree_util.PyTreeDef,
              pytree_to_transpose: Any) -> Any:
  """Alias of :func:`jax.tree_util.tree_transpose`."""
  return tree_util.tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose)


def unflatten(treedef: tree_util.PyTreeDef,
              leaves: Iterable[tree_util.Leaf]) -> Any:
  """Alias of :func:`jax.tree_util.tree_unflatten`."""
  return tree_util.tree_unflatten(treedef, leaves)
