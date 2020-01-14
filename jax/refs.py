# Copyright 2020 Google LLC
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

"""
A system for maintaining mutable state in Python, and lifting references
to this state (loads and stores) into explicit function arguments and
return values. This lets JAX handle a limited set of Python side effects,
allowing more ergonomic user code, while retaining a functional core.
"""

from . import core
from . import linear_util as lu
from .tree_util import tree_map, tree_multimap
from .util import partial

import threading

no_value = core.unit

class Ref(threading.local):
  """A container for managing state in Python."""
  __slots__ = ["value"]

  def __init__(self, value=no_value):
    super().__init__()
    self.value = value

  def __repr__(self):
    return self.__class__.__name__ + '(' + repr(self.value) + ')'

  def load(self, allow_no_value=False):
    if not allow_no_value and self.value is no_value:
      raise RuntimeError("Cannot load from empty ref.")
    return self.value

  def store(self, value):
    self.value = value

  # def swap(self, value):
  #   if self.value is no_value:
  #     raise RuntimeError("Cannot swap into empty ref.")
  #   value, self.value = self.value, value
  #   return value

  # def initialize(self, value)
  #   if self.value is no_value:
  #     self.value = value
  #   return self.value

# Three choices for Ref's pytree behavior:
# 1. register Ref as a pytree with its object identity as metadata
# 2. register Ref as a pytree that always throws an error
# 3. (chosen) don't register Ref as a pytree (treat it as a leaf)
#    --> enables using Refs in tree_util but not jit

def tree_load(ref_tree, typ=Ref, allow_no_value=False):
  loaded = set()
  def load(ref):
    if isinstance(ref, typ) and ref not in loaded:
      loaded.add(ref)
      return ref.load(allow_no_value=allow_no_value)
  return tree_map(load, ref_tree)

def tree_store(ref_tree, val_tree, typ=Ref):
  stored = set()
  def store(ref, val):
    if isinstance(ref, typ) and ref not in stored:
      stored.add(ref)
      ref.store(val)
  tree_multimap(store, ref_tree, val_tree)

def collect(fun, ref_tree):
  def inner(*args, **kwargs):
    out = fun(*args, **kwargs)
    val_tree = tree_load(ref_tree)
    return out, val_tree
  return inner

def inject(fun, ref_tree):
  def inner(val_tree, *args, **kwargs):
    tree_store(ref_tree, val_tree)
    return fun(*args, **kwargs)
  return inner
