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

See the `JAX pytrees notebook <https://jax.readthedocs.io/en/latest/notebooks/JAX_pytrees.html>`_
for examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import collections
from six.moves import reduce

from .lib import pytree

from .util import partial, safe_zip, unzip2

def tree_flatten(tree):
  """Flattens a pytree.

  Args:
    tree: a pytree to flatten.
  Returns:
    a pair with a list of leaves and the corresponding treedef.
  """
  return pytree.flatten(tree)

def tree_unflatten(treedef, leaves):
  """Reconstructs a pytree from the treedef and the leaves.

  The inverse of `tree_flatten`.

  Args:
    treedef: the treedef to reconstruct
    leaves: the list of leaves to use for reconstruction. The list must
      match the leaves of the treedef.
  Returns:
    The reconstructed pytree, containing the `leaves` placed in the
    structure described by `treedef`.
  """
  return treedef.unflatten(leaves)

def tree_leaves(tree):
  """Gets the leaves of a pytree."""
  return pytree.flatten(tree)[0]

def tree_structure(tree):
  """Gets the treedef for a pytree."""
  return pytree.flatten(tree)[1]

def treedef_tuple(treedefs):
  """Makes a tuple treedef from a list of child treedefs."""
  return pytree.tuple(list(treedefs))

def treedef_children(treedef):
  return treedef.children()

def treedef_is_leaf(treedef):
  return treedef.num_nodes == 1

def register_pytree_node(nodetype, flatten_func, unflatten_func):
  """Extends the set of types that are considered internal nodes in pytrees.

  See `example usage <https://jax.readthedocs.io/en/latest/notebooks/JAX_pytrees.html#Pytrees-are-extensible>`_.

  Args:
    nodetype: a Python type to treat as an internal pytree node.
    flatten_func: a function to be used during flattening, taking a value
      of type `nodetype` and returning a pair, with (1) an iterable for
      the children to be flattened recursively, and (2) some auxiliary data
      to be stored in the treedef and to be passed to the `unflatten_func`.
    unflatten_func: a function taking two arguments: the auxiliary data that
      was returned by `flatten_func` and stored in the treedef, and the
      unflattened children. The function should return an instance of
      `nodetype`.
  """
  pytree.register_node(nodetype, flatten_func, unflatten_func)
  _registry[nodetype] = _RegistryEntry(flatten_func, unflatten_func)

def tree_map(f, tree):
  """Maps a function over a pytree to produce a new pytree.

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
  """Maps a multi-input function over pytree args to produce a new pytree.

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

# TODO(mattjj,phawkins): consider removing this function
def _process_pytree(process_node, tree):
  leaves, treedef = pytree.flatten(tree)
  return treedef.walk(process_node, None, leaves), treedef

def build_tree(treedef, xs):
  return treedef.from_iterable_tree(xs)

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

# TODO(mattjj): remove the Python-side registry when the C++-side registry is
# sufficiently queryable that we can express _replace_nones. That may mean once
# we have a flatten_one function.
_RegistryEntry = collections.namedtuple("RegistryEntry", ["to_iter", "from_iter"])
_registry = {
    tuple: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: tuple(xs)),
    list: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: list(xs)),
    dict: _RegistryEntry(lambda xs: unzip2(sorted(xs.items()))[::-1],
                         lambda keys, xs: dict(zip(keys, xs))),
    type(None): _RegistryEntry(lambda z: ((), None), lambda _, xs: None),
}
def _replace_nones(sentinel, tree):
  if tree is None:
    return sentinel
  else:
    handler = _registry.get(type(tree))
    if handler:
      children, metadata = handler.to_iter(tree)
      proc_children = [_replace_nones(sentinel, child) for child in children]
      return handler.from_iter(metadata, proc_children)
    elif isinstance(tree, tuple) and hasattr(tree, '_fields'):
      # handle namedtuple as a special case, based on heuristic
      children = iter(tree)
      proc_children = [_replace_nones(sentinel, child) for child in children]
      return type(tree)(*proc_children)
    else:
      return tree

def tree_reduce(f, tree):
  return reduce(f, tree_leaves(tree))

def tree_all(tree):
  return all(tree_leaves(tree))

register_pytree_node(
  collections.OrderedDict,
  lambda x: (list(x.values()), list(x.keys())),
  lambda keys, values: collections.OrderedDict(safe_zip(keys, values)))

register_pytree_node(
  collections.defaultdict,
  lambda x: (tuple(x.values()), (x.default_factory, tuple(x.keys()))),
  lambda s, values: collections.defaultdict(s[0], safe_zip(s[1], values)))


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
