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

See the `JAX pytrees note <pytrees.html>`_
for examples.
"""

# flake8: noqa: F401
from jax._src.tree_util import (
  Partial,
  all_leaves,
  build_tree,
  register_pytree_node,
  register_pytree_node_class,
  tree_all,
  tree_flatten,
  tree_leaves,
  tree_map,
  tree_multimap,
  tree_reduce,
  tree_structure,
  tree_transpose,
  tree_unflatten,
  treedef_children,
  treedef_is_leaf,
  treedef_tuple,
)
