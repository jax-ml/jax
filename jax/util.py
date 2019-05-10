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

import collections
import functools
import itertools as it
from operator import mul
import types
import numpy as onp

import six

allow_memoize_hash_failures = False


def safe_zip(*args):
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
  return list(zip(*args))


def safe_map(f, *args):
  args = list(map(list, args))
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
  return list(map(f, *args))


def unzip2(xys):
  xs = []
  ys = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)


def unzip3(xyzs):
  xs = []
  ys = []
  zs = []
  for x, y, z in xyzs:
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return tuple(xs), tuple(ys), tuple(zs)


def concatenate(xs):
  return list(it.chain.from_iterable(xs))


def partial(fun, *args, **kwargs):
  wrapped = functools.partial(fun, *args, **kwargs)
  functools.update_wrapper(wrapped, fun)
  wrapped._bound_args = args
  return wrapped

class partialmethod(functools.partial):
  def __get__(self, instance, owner):
    if instance is None:
      return self
    else:
      return partial(self.func, instance,
                     *(self.args or ()), **(self.keywords or {}))

def curry(f):
  """Curries arguments of f, returning a function on any remaining arguments.

  For example:
  >>> f = lambda x, y, z, w: x * y + z * w
  >>> f(2,3,4,5)
  26
  >>> curry(f)(2)(3, 4, 5)
  26
  >>> curry(f)(2, 3)(4, 5)
  26
  >>> curry(f)(2, 3, 4, 5)()
  26
  """
  return partial(partial, f)

def toposort(end_node):
  child_counts = {}
  stack = [end_node]
  while stack:
    node = stack.pop()
    if id(node) in child_counts:
      child_counts[id(node)] += 1
    else:
      child_counts[id(node)] = 1
      stack.extend(node.parents)

  sorted_nodes = []
  childless_nodes = [end_node]
  while childless_nodes:
    node = childless_nodes.pop()
    sorted_nodes.append(node)
    for parent in node.parents:
      if child_counts[id(parent)] == 1:
        childless_nodes.append(parent)
      else:
        child_counts[id(parent)] -= 1

  return sorted_nodes[::-1]


def split_merge(predicate, xs):
  sides = list(map(predicate, xs))
  lhs = [x for x, s in zip(xs, sides) if s]
  rhs = [x for x, s in zip(xs, sides) if not s]
  def merge(new_lhs, new_rhs):
    out = []
    for s in sides:
      if s:
        out.append(new_lhs[0])
        new_lhs = new_lhs[1:]
      else:
        out.append(new_rhs[0])
        new_rhs = new_rhs[1:]
    assert not new_rhs
    assert not new_lhs
    return out

  return lhs, rhs, merge


if six.PY3:
  OrderedDict = collections.OrderedDict
else:
  # Retrofits a move_to_end method to OrderedDict in Python 2 mode.
  class OrderedDict(collections.OrderedDict):
    def move_to_end(self, key):
      value = self[key]
      del self[key]
      self[key] = value


_NO_MEMO_ENTRY = object()

def memoize(fun, max_size=4096):
  cache = OrderedDict()
  def memoized_fun(*args, **kwargs):
    key = (args, tuple(kwargs and sorted(kwargs.items())))
    try:
      ans = cache.get(key, _NO_MEMO_ENTRY)
      if ans != _NO_MEMO_ENTRY:
        cache.move_to_end(key)
        return ans
    except TypeError:
      if not allow_memoize_hash_failures:
        raise

    if len(cache) > max_size:
      cache.popitem(last=False)

    ans = cache[key] = fun(*args, **kwargs)
    return ans
  return memoized_fun


def memoize_unary(func):
  class memodict(dict):
    def __missing__(self, key):
      val = self[key] = func(key)
      return val
  return memodict().__getitem__


def prod(xs):
  return functools.reduce(mul, xs, 1)


class WrapHashably(object):
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return id(self.val)

  def __eq__(self, other):
    return self.val is other.val

class Hashable(object):
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return self.val == other.val



def get_module_functions(module):
  """Finds functions in module.
  Args:
    module: A Python module.
  Returns:
    module_fns: A set of functions, builtins or ufuncs in `module`.
  """

  module_fns = set()
  for key in dir(module):
    attr = getattr(module, key)
    if isinstance(
        attr, (types.BuiltinFunctionType, types.FunctionType, onp.ufunc)):
      module_fns.add(attr)
  return module_fns
