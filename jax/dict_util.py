# Copyright 2019 Google LLC
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

"""Utilities for working with nested dictionaries as path-indexed containers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it

def maybe_set_dict(tree, path, val, accum_fn):
  if len(path) == 0:
    raise ValueError("cannot index nested dict with an empty path")
  subtree = tree
  for path_element in path[:-1]:
    if path_element not in subtree:
      if accum_fn is None:
        subtree[path_element] = dict()
      else:
        return val
    subtree = subtree[path_element]
  if accum_fn is None: # storing, i.e. collect
    if path[-1] in subtree:
      # Some users might want `val = subtree[path[-1]]` here?
      raise RuntimeError("cannot store multiple values to the same nested dict path")
    else:
      subtree[path[-1]] = val
  elif path[-1] in subtree:
    val = accum_fn(val, subtree[path[-1]])
  return val

def iterpaths(tree, prefix=()):
  for key, val in tree.items():
    path = prefix + (key,)
    if isinstance(val, dict):
      # TODO(jekbradbury): replace with `yield from` when Python 2 is dropped
      for path2, val2 in iterpaths(val, path):
        yield path2, val2
    else:
      yield path, val

def as_dict(paths_and_values):
  tree = dict()
  for path, val in paths_and_values:
    maybe_set_dict(tree, path, val, None)
  return tree

def dict_join(*trees):
  """Union the contents of nested dictionaries.
  
  For example:

  >>> dict_join({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
  {"a": 1, "b": {"c": 2, "d": 3}}
  """
  return as_dict(it.chain(*(iterpaths(tree) for tree in trees)))
