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

"""Utilities for working with tree-like container data structures.

This module provides a small set of utility functions for working with tree-like
data structures, such as nested tuples, lists, and dicts. We call these
structures pytrees. They are trees in that they are defined recursively (any
non-pytree is a pytree, i.e. a leaf, and any pytree of pytrees is a pytree) and
can be operated on recursively (object identity equivalence is not preserved by
mapping operations, and the structures cannot contain reference cycles).

The set of Python types that are considered pytree nodes (e.g. that can be
mapped over, rather than treated as leaves) is extensible. There is a single
module-level registry of types, and class hierarchy is ignored. By registering a
new pytree node type, that type in effect becomes transparent to the utility
functions in this file.

The primary purpose of this module is to enable the interoperability between
user defined data structures and JAX transformations (e.g. `jit`). This is not
meant to be a general purpose tree-like data structure handling library.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple
import itertools as it
from six.moves import reduce

from .lib import pytree

from .util import unzip2, partial, safe_map


# TODO(phawkins): use the first case unconditionally when the minimum Jaxlib
# version has been increased to 0.1.23.
if pytree:

  def tree_map(f, tree):
    """Map a function over a pytree to produce a new pytree.

    Args:
      f: function to be applied at each leaf.
      tree: a pytree to be mapped over.

    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x)` where `x` is the value at the corresponding leaf in
      `tree`.
    """
    leaves, treedef = pytree.flatten(tree)
    return treedef.unflatten(map(f, leaves))

  def tree_multimap(f, tree, *rest):
    """Map a multi-input function over pytree args to produce a new pytree.

    Args:
      f: function that takes `1 + len(rest)` arguments, to be applied at the
        corresponding leaves of the pytrees.
      tree: a pytree to be mapped over, with each leaf providing the first
        positional argument to `f`.
      *rest: a tuple of pytrees, each of which has the same structure as tree or
        or has tree as a prefix.
    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x, *xs)` where `x` is the value at the corresponding leaf
      in `tree` and `xs` is the tuple of values at corresponding nodes in
      `rest`.
    """
    leaves, treedef = pytree.flatten(tree)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))

  def tree_leaves(tree):
    return pytree.flatten(tree)[0]

  def process_pytree(process_node, tree):
    leaves, treedef = pytree.flatten(tree)
    return treedef.walk(process_node, None, leaves), treedef

  tree_flatten = pytree.flatten

  def build_tree(treedef, xs):
    return treedef.from_iterable_tree(xs)

  def treedef_is_leaf(treedef):
    return treedef.num_nodes == 1

  def tree_unflatten(treedef, xs):
    return treedef.unflatten(xs)

  def tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose):
    flat, treedef = tree_flatten(pytree_to_transpose)
    expected_treedef = outer_treedef.compose(inner_treedef)
    if treedef != expected_treedef:
      raise TypeError("Mismatch\n{}\n != \n{}".format(treedef, expected_treedef))

    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    flat = iter(flat)
    lol = [[next(flat) for _ in range(inner_size)] for __ in range(outer_size)]
    transposed_lol = zip(*lol)
    subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
    return tree_unflatten(inner_treedef, subtrees)

  def tree_structure(tree):
    _, treedef = pytree.flatten(tree)
    return treedef

  def treedef_tuple(trees):
    return pytree.tuple(list(trees))

  def treedef_children(treedef):
      return treedef.children()

  register_pytree_node = pytree.register_node

else:
  def tree_map(f, tree):
    """Map a function over a pytree to produce a new pytree.

    Args:
      f: function to be applied at each leaf.
      tree: a pytree to be mapped over.

    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x)` where `x` is the value at the corresponding leaf in
      `tree`.
    """
    node_type = _get_node_type(tree)
    if node_type:
      children, node_spec = node_type.to_iterable(tree)
      new_children = [tree_map(f, child) for child in children]
      return node_type.from_iterable(node_spec, new_children)
    else:
      return f(tree)

  def tree_multimap(f, tree, *rest):
    """Map a multi-input function over pytree args to produce a new pytree.

    Args:
      f: function that takes `1 + len(rest)` arguments, to be applied at the
        corresponding leaves of the pytrees.
      tree: a pytree to be mapped over, with each leaf providing the first
        positional argument to `f`.
      *rest: a tuple of pytrees, each with the same structure as `tree`.

    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x, *xs)` where `x` is the value at the corresponding leaf
      in `tree` and `xs` is the tuple of values at corresponding leaves in `rest`.
    """
    node_type = _get_node_type(tree)
    if node_type:
      children, aux_data = node_type.to_iterable(tree)
      all_children = [children]
      for other_tree in rest:
        other_node_type = _get_node_type(other_tree)
        if node_type != other_node_type:
          raise TypeError('Mismatch: {} != {}'.format(other_node_type, node_type))
        other_children, other_aux_data = node_type.to_iterable(other_tree)
        if other_aux_data != aux_data:
          raise TypeError('Mismatch: {} != {}'.format(other_aux_data, aux_data))
        all_children.append(other_children)

      new_children = [tree_multimap(f, *xs) for xs in zip(*all_children)]
      return node_type.from_iterable(aux_data, new_children)
    else:
      return f(tree, *rest)

  def _walk_pytree(f_node, f_leaf, tree):
    node_type = _get_node_type(tree)
    if node_type:
      children, node_spec = node_type.to_iterable(tree)
      proc_children, child_specs = unzip2([_walk_pytree(f_node, f_leaf, child)
                                           for child in children])
      tree_def = _PyTreeDef(node_type, node_spec, child_specs)
      return f_node(proc_children), tree_def
    else:
      return f_leaf(tree), leaf

  def process_pytree(process_node, tree):
    return _walk_pytree(process_node, lambda x: x, tree)

  def build_tree(treedef, xs):
    if treedef is leaf:
      return xs
    else:
      # We use 'iter' for clearer error messages
      children = safe_map(build_tree, iter(treedef.children), iter(xs))
      return treedef.node_type.from_iterable(treedef.node_data, children)

  def tree_leaves(tree):
    """Generator that iterates over all leaves of a pytree."""
    node_type = _get_node_type(tree)
    if node_type:
      children, _ = node_type.to_iterable(tree)
      for child in children:
        # TODO(mattjj,phawkins): use 'yield from' when PY2 is dropped
        for leaf in tree_leaves(child):
          yield leaf
    else:
      yield tree

  def tree_flatten(tree):
    itr, treedef = _walk_pytree(it.chain.from_iterable, lambda x: (x,), tree)
    return list(itr), treedef

  def _tree_unflatten(xs, treedef):
    if treedef is leaf:
      return next(xs)
    else:
      children = tuple(map(partial(_tree_unflatten, xs), treedef.children))
      return treedef.node_type.from_iterable(treedef.node_data, children)

  def tree_unflatten(treedef, xs):
    return _tree_unflatten(iter(xs), treedef)

  def tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose):
    flat, treedef = tree_flatten(pytree_to_transpose)
    expected_treedef = _nested_treedef(inner_treedef, outer_treedef)
    if treedef != expected_treedef:
      raise TypeError("Mismatch\n{}\n != \n{}".format(treedef, expected_treedef))

    inner_size = _num_leaves(inner_treedef)
    outer_size = _num_leaves(outer_treedef)
    flat = iter(flat)
    lol = [[next(flat) for _ in range(inner_size)] for __ in range(outer_size)]
    transposed_lol = zip(*lol)
    subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
    return tree_unflatten(inner_treedef, subtrees)

  def _num_leaves(treedef):
    return 1 if treedef is leaf else sum(map(_num_leaves, treedef.children))

  def _nested_treedef(inner, outer):
    # just used in tree_transpose error checking
    if outer is leaf:
      return inner
    else:
      children = map(partial(_nested_treedef, inner), outer.children)
      return _PyTreeDef(outer.node_type, outer.node_data, tuple(children))

  def tree_structure(tree):
    _, spec = process_pytree(lambda _: None, tree)
    return spec


  class _PyTreeDef(object):
    __slots__ = ("node_type", "node_data", "children")

    def __init__(self, node_type, node_data, children):
      self.node_type = node_type
      self.node_data = node_data
      self.children = children

    def __repr__(self):
      if self.node_data is None:
        data_repr = ""
      else:
        data_repr = "[{}]".format(self.node_data)

      return "PyTree({}{}, [{}])".format(self.node_type.name, data_repr,
                                       ','.join(map(repr, self.children)))

    def __hash__(self):
      return hash((self.node_type, self.node_data, tuple(self.children)))

    def __eq__(self, other):
      if other is leaf:
        return False
      else:
        return (self.node_type == other.node_type and
                self.node_data == other.node_data and
                self.children == other.children)

    def __ne__(self, other):
      return not self == other


  class _PyLeaf(object):
    __slots__ = ()

    def __repr__(self):
      return '*'

  leaf = _PyLeaf()

  def treedef_is_leaf(treedef):
    return treedef is leaf

  def treedef_tuple(treedefs):
    return _PyTreeDef(node_types[tuple], None, tuple(treedefs))

  def treedef_children(treedef):
    return treedef.children

  def dict_to_iterable(xs):
    keys = tuple(sorted(xs.keys()))
    return tuple(map(xs.get, keys)), keys

  class NodeType(object):
    def __init__(self, name, to_iterable, from_iterable):
      self.name = name
      self.to_iterable = to_iterable
      self.from_iterable = from_iterable

    def __repr__(self):
      return self.name

  node_types = {}

  def register_pytree_node(py_type, to_iterable, from_iterable):
    assert py_type not in node_types
    node_types[py_type] = NodeType(str(py_type), to_iterable, from_iterable)

  register_pytree_node(tuple, lambda xs: (xs, None), lambda _, xs: tuple(xs))
  register_pytree_node(list, lambda xs: (tuple(xs), None), lambda _, xs: list(xs))
  register_pytree_node(dict, dict_to_iterable, lambda keys, xs: dict(zip(keys, xs)))

  # To handle namedtuples, we can't just use the standard table of node_types
  # because every namedtuple creates its own type and thus would require its own
  # entry in the table. Instead we use a heuristic check on the type itself to
  # decide whether it's a namedtuple type, and if so treat it as a pytree node.
  def _get_node_type(maybe_tree):
    t = type(maybe_tree)
    return node_types.get(t) or _namedtuple_node(t)

  def _namedtuple_node(t):
    if issubclass(t, tuple) and hasattr(t, '_fields'):
      return NamedtupleNode

  NamedtupleNode = NodeType('namedtuple',
                            lambda xs: (tuple(xs), type(xs)),
                            lambda t, xs: t(*xs))


def tree_reduce(f, tree):
  return reduce(f, tree_leaves(tree))


def tree_all(tree):
  return all(tree_leaves(tree))


class Partial(functools.partial):
  """A version of functools.partial that works in pytrees.

  Use it for partial function evaluation in a way that is compatible with JAX's
  transformations, e.g., ``Partial(func, *args, **kwargs)``.

  (You need to explicitly opt-in to this behavior because we didn't want to give
  functools.partial different semantics than normal function closures.)
  """

register_pytree_node(
    Partial,
    lambda partial_: ((partial_.args, partial_.keywords), partial_.func),
    lambda func, xs: Partial(func, *xs[0], **xs[1]),
)
