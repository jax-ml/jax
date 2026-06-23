# Copyright 2026 The JAX Authors.
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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
import itertools as it
from typing import Any

from jax._src.lib import pytree
from jax._src.tree_util import (
    PyTree, PyTreeDef, tracing_registry, register_static, treedef_children)
from jax._src.util import unzip2, partition_list, merge_lists

class FlatTree:
  """A FlatTree stores a treedef and a flat list of values. It's meant to be
  isomorphic to the corresponding pytree but we can map over it more easily.
  Compared to `tree_map`, FlatTree.map has these benefits:
    1. It doesn't touch user flatten/unflatten code (which shouldn't have side
       effects but sometimes does in practice).
    2. It can be faster, because it skips the recursive traversal.
    3. It actually obeys the functor rules. For example,
       `flat_tree.map(lambda x: (f(x), g(x))).unzip2()[0]` will give
       the same result as `flat_tree.map(f)`, whereas in the `tree_map` version
       the tuple-returning function would change the tree structure and `unzip`
       wouldn't be able to recover it.
  """
  # `FlatTree` constructor is private. Use `flatten` instead
  def __init__(self, vals, treedef: PyTreeDef, statics,
               registry=tracing_registry):
    if not isinstance(vals, tuple):
      vals = tuple(vals)
    self.vals = vals
    assert isinstance(treedef, pytree.PyTreeDef)
    self.tree = treedef
    self.statics = statics  # tree-prefix tuple-dict-tree of bools
    self.registry = registry

  def __eq__(self, other):
    return (isinstance(other, FlatTree) and self.vals == other.vals
            and self.tree == other.tree and self.statics == other.statics
            and self.registry is other.registry)

  def __hash__(self):
    return hash((self.vals, self.tree))

  def __repr__(self):
    return f"FlatTree({self.vals})"

  def map(self, f: Callable) -> FlatTree:
    return self.update(f(x) for x in self.vals)

  def map2(self: FlatTree, f: Callable, t2: Sequence[Any]) -> FlatTree:
    n = len(self)
    assert len(t2) == n
    return self.update(f(x1, x2) for x1, x2 in zip(self.vals, list(t2)))

  def map3(
      self: FlatTree, f: Callable, t2: Sequence[Any], t3: Sequence[Any]) -> FlatTree:
    n = len(self)
    assert len(t2) == n and len(t3) == n
    return self.update(f(x1, x2, x3)
                       for x1, x2, x3 in zip(self.vals, list(t2), list(t3)))

  def unzip2(self: FlatTree) -> tuple[FlatTree, FlatTree]:
    ys = []
    zs = []
    for y, z in self.vals:
      ys.append(y)
      zs.append(z)
    return self.update(ys), self.update(zs)

  # TODO: add other helpers like map3, zip, unzip3 etc. as needed

  def unpack(self: FlatTree) -> tuple[FlatTree, ...]:
    # TODO: this is O(N) not O(1) (with N as the number of leaves). If it
    # becomes a problem we can fix it with a fancier data structure.
    # TODO(dougalm): assert that we're dealing with a tuple
    trees = treedef_children(self.tree)
    children = []
    offset = 0
    for i, tree in enumerate(trees):
      statics = False if isinstance(self.statics, bool) else self.statics[i]
      new_offset = offset + tree.num_leaves
      children.append(FlatTree(self.vals[offset:new_offset], tree, statics,
                      registry=self.registry))
      offset = new_offset
    return tuple(children)

  def with_aux(self:FlatTree, aux:Any) -> FlatTree:
    return pack((self, flatten(Static(aux))))

  def unpack_aux(self:FlatTree) -> tuple[FlatTree, Any]:
    x, aux = self.unpack()
    return x, aux.unflatten().val

  def unflatten(self) -> PyTree:
    pytree = self.tree.unflatten(self.vals)
    return unwrap_statics(pytree, self.statics)

  @property
  def tree_without_statics(self):
    return filter_statics_from_treedef(self.registry, self.tree, self.statics)

  def update(self, new_vals) -> FlatTree:
    # `new_vals` can be a generator because `FlatTree` forces it to a tuple
    new = FlatTree(new_vals, self.tree, self.statics, registry=self.registry)
    assert len(self.vals) == len(new.vals)
    return new

  @cached_property
  def paths(self) -> FlatTree:
    # TODO(dougalm): find a way to do this without roundtripping
    try:
      paths, _ = unzip2(self.registry.flatten_with_path(self.unflatten())[0])
      assert len(paths) == len(self.vals)
      return self.update(paths)
    except:
      return self.update([()] * len(self.vals))  # not our fault

  def __len__(self):
    return self.len

  @cached_property
  def len(self):
    return self.tree.num_leaves

  def __iter__(self):
    return iter(self.vals)

  def __getitem__(self, i):
    return self.vals[i]

  def filter(self, f):
    # a FlatTree version of list.filter. Unlike the latter, it keeps
    # the filtered-out data in the pytree structure, so that it can
    # be reinstantiated with `unfilter`.
    return self.filter_with_mask(map(f, self))

  def filter_with_mask(self, mask):
    xs = list(self)
    none_ft = self.map(lambda _: None)
    keep_mask = list(mask)
    rejected, kept = partition_list(keep_mask, xs)
    return flatten_list(kept).with_aux((none_ft, keep_mask, rejected))

  def unfilter(self):
    kept_ft, (none_ft, keep_mask, rejected) = self.unpack_aux()
    kept_list = kept_ft.unflatten()
    return none_ft.update(merge_lists(keep_mask, rejected, kept_list))

  def enumerate(self):
    idxs = it.count()
    return self.map(lambda x: (next(idxs), x))

def pack(tree, registry=tracing_registry):
  # We could generalize this to arbitrary pytrees of FlatTree but tuples/dicts
  # are sufficient for now.
  if isinstance(tree, FlatTree):
    return tree
  elif isinstance(tree, tuple):
    vals = []
    trees = []
    staticss = []
    for child_tree in tree:
      child = pack(child_tree, registry=registry)
      vals.extend(child.vals)
      trees.append(child.tree)
      staticss.append(child.statics)
    return FlatTree(vals, pytree.treedef_tuple(registry, trees),
                    tuple(staticss), registry=registry)
  elif isinstance(tree, dict):
    # only empty case handled for now
    if tree == {}:
      return flatten({}, registry=registry)
    else:
      assert False
  else:
    assert False, type(tree)

def pack_args(*args, **kwargs):
  # TODO: check elements of args and kwargs are all flat trees
  return pack((args, kwargs))

def flatten(tree: PyTree, is_leaf=None, registry=tracing_registry) -> FlatTree:
  vals, tree = registry.flatten(tree, is_leaf)
  return FlatTree(vals, tree, False, registry=registry)

def flatten_args(*arg_trees: PyTree, registry=tracing_registry) -> FlatTree:
  return flatten((arg_trees, {}), registry=registry)

def flatten_static_argnums(args, static_argnums, registry=tracing_registry):
  if not static_argnums:
    return flatten(args, registry=registry)
  else:
    assert isinstance(args, tuple)
    num_args = len(args)
    static_argnums = [i % num_args if i < 0 else i for i in static_argnums]
    statics = tuple(i in static_argnums for i, _ in enumerate(args))
    tree_with_statics = tuple(
        Static(x) if static else x for static, x in zip(statics, args))
    vals, treedef = registry.flatten(tree_with_statics)
    return FlatTree(vals, treedef, statics=statics, registry=registry)

def flatten_static_argnames(kwargs, static_argnames, registry=tracing_registry):
  if not static_argnames:
    return flatten(kwargs, registry=registry)
  else:
    assert isinstance(kwargs, dict)
    statics = {k : k in static_argnames for k, _ in kwargs.items()}
    tree_with_statics = {k : Static(v) if statics[k] else v
                         for k, v in kwargs.items()}
    vals, treedef = registry.flatten(tree_with_statics)
    return FlatTree(vals, treedef, statics=statics, registry=registry)

def flatten_static_argnums_argnames(
    args, kwargs, static_argnums, static_argnames, registry=tracing_registry):
  return pack((
      flatten_static_argnums(args, static_argnums, registry=registry),
      flatten_static_argnames(kwargs, static_argnames, registry=registry)),
      registry=registry)

def flatten_list(xs):
  # [a] -> FlatTree[a] . Treats list elements as leaves.
  return pack(tuple(singleton(x) for x in xs))

def singleton(x):
  # a -> FlatTree[a]
  _, tree = tracing_registry.flatten(0)
  return FlatTree([x], tree, False)

def unwrap_statics(pytree, statics):
  if statics is False:
    return pytree
  elif statics is True:
    return pytree.val  # pytree should be a `Static` object
  elif isinstance(pytree, tuple):
    return tuple(unwrap_statics(p, s) for p, s in zip(pytree, statics))
  elif isinstance(pytree, dict):
    return {k : unwrap_statics(p, statics[k]) for k, p in pytree.items()}
  else:
    assert False, "unreachable"

def filter_statics_from_treedef(registry, treedef, statics):
  if statics is False:
    return treedef
  elif statics is True:
    assert False, "unreachable"
  elif isinstance(statics, tuple):
    filtered = tuple(
        filter_statics_from_treedef(registry, td, s)
        for td, s in zip(treedef.children(), statics) if s is not True)
    return treedef.from_node_data_and_children(registry, treedef.node_data(), filtered)
  elif isinstance(statics, dict):
    ty, keys = treedef.node_data()
    filtered_keys, filtered_subtrees = unzip2(
        (k, filter_statics_from_treedef(registry, td, statics[k]))
        for td, k in zip(treedef.children(), keys) if statics[k] is not True)
    return treedef.from_node_data_and_children(registry, (ty, filtered_keys), filtered_subtrees)
  else:
    assert False, "unreachable"

@register_static
@dataclass(frozen=True, slots=True)
class Static:
  val: Any

  def __eq__(self, other):
    return (type(other) is Static and type(self.val) is type(other.val) and
            self.val == other.val)
