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


import functools
import itertools as it
import types

import numpy as np


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

def unzip4(wxyzs):
  ws = []
  xs = []
  ys = []
  zs = []
  for w, x, y, z in wxyzs:
    ws.append(w)
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return tuple(ws), tuple(xs), tuple(ys), tuple(zs)

def subvals(lst, replace):
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)

def split_list(args, ns):
  assert type(ns) is list
  args = list(args)
  lists = []
  for n in ns:
    lists.append(args[:n])
    args = args[n:]
  lists.append(args)
  return lists

def split_dict(dct, names):
  dct = dict(dct)
  lst = [dct.pop(name) for name in names]
  assert not dct
  return lst

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

def toposort(end_nodes):
  if not end_nodes: return []
  end_nodes = _remove_duplicates(end_nodes)

  child_counts = {}
  stack = list(end_nodes)
  while stack:
    node = stack.pop()
    if id(node) in child_counts:
      child_counts[id(node)] += 1
    else:
      child_counts[id(node)] = 1
      stack.extend(node.parents)
  for node in end_nodes:
    child_counts[id(node)] -= 1

  sorted_nodes = []
  childless_nodes = [node for node in end_nodes if child_counts[id(node)] == 0]
  assert childless_nodes
  while childless_nodes:
    node = childless_nodes.pop()
    sorted_nodes.append(node)
    for parent in node.parents:
      if child_counts[id(parent)] == 1:
        childless_nodes.append(parent)
      else:
        child_counts[id(parent)] -= 1

  check_toposort(sorted_nodes[::-1])
  return sorted_nodes[::-1]

def check_toposort(nodes):
  visited = set()
  for node in nodes:
    assert all(id(parent) in visited for parent in node.parents)
    visited.add(id(node))

def _remove_duplicates(node_list):
  seen = set()
  out = []
  for n in node_list:
    if id(n) not in seen:
      seen.add(id(n))
      out.append(n)
  return out

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

def cache(max_size=4096):
  return functools.lru_cache(maxsize=max_size)

memoize = functools.lru_cache(maxsize=None)

def prod(xs):
  out = 1
  for x in xs:
    out *= x
  return out

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
    module_fns: A dict of names mapped to functions, builtins or ufuncs in `module`.
  """
  module_fns = {}
  for key in dir(module):
    # Omitting module level __getattr__, __dir__ which was added in Python 3.7
    # https://www.python.org/dev/peps/pep-0562/
    if key in ('__getattr__', '__dir__'):
      continue
    attr = getattr(module, key)
    if isinstance(
        attr, (types.BuiltinFunctionType, types.FunctionType, np.ufunc)):
      module_fns[key] = attr
  return module_fns

def wrap_name(name, transform_name):
  return transform_name + '(' + name + ')'

def extend_name_stack(stack, name=''):
  return stack + name + '/'
