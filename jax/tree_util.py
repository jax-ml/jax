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

from collections import namedtuple
import itertools as it
from six.moves import reduce

from .util import unzip2, concatenate, partial, safe_map

map = safe_map


def tree_map(f, tree):
  node_type = node_types.get(type(tree))
  if node_type:
    children, node_spec = node_type.to_iterable(tree)
    new_children = [tree_map(f, child) for child in children]
    return node_type.from_iterable(node_spec, new_children)
  else:
    return f(tree)


def tree_multimap(f, tree, *rest):
  tree_type = type(tree)
  node_type = node_types.get(tree_type)
  if node_type:
    children, node_spec = node_type.to_iterable(tree)
    all_children = [children]
    for other_tree in rest:
      other_children, other_node_spec = node_type.to_iterable(other_tree)
      if other_node_spec != node_spec:
        raise TypeError('Mismatch: {} != {}'.format(other_node_spec, node_spec))
      all_children.append(other_children)

    new_children = [tree_multimap(f, *xs) for xs in zip(*all_children)]
    return node_type.from_iterable(node_spec, new_children)
  else:
    return f(tree, *rest)


def tree_reduce(f, tree):
  flat, _ = tree_flatten(tree)
  return reduce(f, flat)


def tree_all(tree):
  flat, _ = tree_flatten(tree)
  return all(flat)


def process_pytree(process_node, tree):
  return walk_pytree(process_node, lambda x: x, tree)


def walk_pytree(f_node, f_leaf, tree):
  node_type = node_types.get(type(tree))
  if node_type:
    children, node_spec = node_type.to_iterable(tree)
    proc_children, child_specs = unzip2([walk_pytree(f_node, f_leaf, child)
                                         for child in children])
    tree_def = PyTreeDef(node_type, node_spec, child_specs)
    return f_node(proc_children), tree_def
  else:
    return f_leaf(tree), leaf


def build_tree(treedef, xs):
  if treedef is leaf:
    return xs
  else:
    # We use 'iter' for clearer error messages
    children = map(build_tree, iter(treedef.children), iter(xs))
    return treedef.node_type.from_iterable(treedef.node_data, children)


tree_flatten = partial(walk_pytree, concatenate, lambda x: [x])

def tree_unflatten(xs, treedef):
  xs = iter(xs)
  if treedef is leaf:
    return next(xs)
  else:
    children = map(partial(tree_unflatten, xs), treedef.children)
    return treedef.node_type.from_iterable(treedef.node_data, children)


def tree_structure(tree):
  spec, _ = process_pytree(tree, lambda _: None)
  return spec


class PyTreeDef(object):
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
    return (self.node_type == other.node_type and
            self.node_data == other.node_data and
            self.children == other.children)

  def __ne__(self, other):
    return not self == other


class PyLeaf(object):
  def __repr__(self):
    return '*'

leaf = PyLeaf()

def dict_to_iterable(xs):
  keys = tuple(sorted(xs.keys()))
  return tuple(map(xs.get, keys)), keys

class NodeType(object):
  def __init__(self, name, to_iterable, from_iterable):
    self.name = name
    self.to_iterable = to_iterable
    self.from_iterable = from_iterable

node_types = {}

def register_pytree_node(py_type, to_iterable, from_iterable):
  assert py_type not in node_types
  node_types[py_type] = NodeType(str(py_type), to_iterable, from_iterable)

register_pytree_node(tuple, lambda xs: (xs, None), lambda _, xs: tuple(xs))
register_pytree_node(list, lambda xs: (tuple(xs), None), lambda _, xs: list(xs))
register_pytree_node(dict, dict_to_iterable, lambda keys, xs: dict(zip(keys, xs)))
register_pytree_node(type(None), lambda z: ((), None), lambda _, xs: None)
