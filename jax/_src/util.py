# Copyright 2018 The JAX Authors.
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

from collections.abc import Iterable, Iterator, Sequence
import functools
from functools import partial
import itertools as it
import logging
import operator
from typing import (Any, Callable, Generic, Optional, TypeVar, overload, TYPE_CHECKING, cast)
import weakref

import numpy as np

from jax._src.lib import xla_client as xc
from jax._src.lib import utils as jaxlib_utils
from jax._src.config import config

logger = logging.getLogger(__name__)

Seq = Sequence

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


if TYPE_CHECKING:
  # safe_zip cannot yet be fully annotated, so we use a strategy similar
  # to that used for builtins.zip in python/typeshed. This supports
  # return types matching input types for up to three arguments.
  @overload
  def safe_zip(__arg1: Iterable[T1]) -> list[tuple[T1]]: ...
  @overload
  def safe_zip(__arg1: Iterable[T1], __arg2: Iterable[T2]) -> list[tuple[T1, T2]]: ...
  @overload
  def safe_zip(__arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3]) -> list[tuple[T1, T2, T3]]: ...
  @overload
  def safe_zip(__arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], *args) -> list[tuple[Any, ...]]: ...

  def safe_zip(*args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
      assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(zip(*args))

else:
  safe_zip = jaxlib_utils.safe_zip


if TYPE_CHECKING:
  # safe_map cannot yet be fully annotated, so we use a strategy similar
  # to that used for builtins.map in python/typeshed. This supports
  # checking input types for the callable with up to three arguments.
  @overload
  def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> list[T]: ...

  @overload
  def safe_map(f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]) -> list[T]: ...

  @overload
  def safe_map(f: Callable[[T1, T2, T3], T], __arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3]) -> list[T]: ...

  @overload
  def safe_map(f: Callable[..., T], __arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], *args) -> list[T]: ...

  def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
      assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))

else:
  safe_map = jaxlib_utils.safe_map

def unzip2(xys: Iterable[tuple[T1, T2]]
    ) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
  """Unzip sequence of length-2 tuples into two tuples."""
  # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
  # is too permissive about inputs, and does not guarantee a length-2 output.
  xs: list[T1] = []
  ys: list[T2] = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)

def unzip3(xyzs: Iterable[tuple[T1, T2, T3]]
    ) -> tuple[tuple[T1, ...], tuple[T2, ...], tuple[T3, ...]]:
  """Unzip sequence of length-3 tuples into three tuples."""
  # Note: we deliberately don't use zip(*xyzs) because it is lazily evaluated,
  # is too permissive about inputs, and does not guarantee a length-3 output.
  xs: list[T1] = []
  ys: list[T2] = []
  zs: list[T3] = []
  for x, y, z in xyzs:
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return tuple(xs), tuple(ys), tuple(zs)

def subvals(lst, replace):
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)

def split_list(args: Sequence[T], ns: Sequence[int]) -> list[list[T]]:
  args = list(args)
  lists = []
  for n in ns:
    lists.append(args[:n])
    args = args[n:]
  lists.append(args)
  return lists

def partition_list(bs: Sequence[bool], l: Sequence[T]) -> tuple[list[T], list[T]]:
  assert len(bs) == len(l)
  lists = [], []  # type: ignore
  for b, x in zip(bs, l):
    lists[b].append(x)
  return lists

def merge_lists(bs: Sequence[bool], l0: Sequence[T], l1: Sequence[T]) -> list[T]:
  assert sum(bs) == len(l1) and len(bs) - sum(bs) == len(l0)
  i0, i1 = iter(l0), iter(l1)
  out = [next(i1) if b else next(i0) for b in bs]
  sentinel = object()
  assert next(i0, sentinel) is next(i1, sentinel) is sentinel
  return out

def split_dict(dct, names):
  dct = dict(dct)
  lst = [dct.pop(name) for name in names]
  assert not dct
  return lst

def concatenate(xs: Iterable[Sequence[T]]) -> list[T]:
  """Concatenates/flattens a list of lists."""
  return list(it.chain.from_iterable(xs))

flatten = concatenate

_unflatten_done = object()

def unflatten(xs: Iterable[T], ns: Sequence[int]) -> list[list[T]]:
  """Splits `xs` into subsequences of lengths `ns`.

  Unlike `split_list`, the `sum(ns)` must be equal to `len(xs)`."""
  xs_iter = iter(xs)
  unflattened = [[next(xs_iter) for _ in range(n)] for n in ns]
  assert next(xs_iter, _unflatten_done) is _unflatten_done
  return unflattened


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
  sorted_nodes = sorted_nodes[::-1]

  check_toposort(sorted_nodes)
  return sorted_nodes

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
  def wrap(f):
    @functools.lru_cache(max_size)
    def cached(_, *args, **kwargs):
      return f(*args, **kwargs)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      if config.jax_check_tracer_leaks:
        return f(*args, **kwargs)
      else:
        return cached(config._trace_context(), *args, **kwargs)

    wrapper.cache_clear = cached.cache_clear
    wrapper.cache_info = cached.cache_info
    return wrapper
  return wrap

memoize = cache(max_size=None)

def weakref_lru_cache(call: Callable, maxsize=2048):
  """
  Least recently used cache decorator with weakref support.

  The cache will take a weakref to the first argument of the wrapped function
  and strong refs to all subsequent operations. In all other respects it should
  behave similar to `functools.lru_cache`.
  """
  global _weakref_lru_caches
  cached_call = xc.weakref_lru_cache(config._trace_context, call, maxsize)
  _weakref_lru_caches.add(cached_call)
  return cached_call

_weakref_lru_caches = weakref.WeakSet()  # type: ignore

def clear_all_weakref_lru_caches():
  for cached_call in _weakref_lru_caches:
    cached_call.cache_clear()

class Unhashable:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __eq__(self, other):
    return self.val == other.val

class Hashable:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return self.val == other.val

class WrapKwArgs:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(tuple((k, v) for k, v in sorted(self.val.items())))

  def __eq__(self, other):
    return self.val == other.val

def wrap_name(name, transform_name):
  return transform_name + '(' + name + ')'

def canonicalize_axis(axis, num_dims) -> int:
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(f"axis {axis} is out of bounds for array of dimension {num_dims}")
  if axis < 0:
    axis = axis + num_dims
  return axis

def moveaxis(x, src, dst):
  if src == dst:
    return x
  if isinstance(src, int):
    src = (src,)
  if isinstance(dst, int):
    dst = (dst,)
  src = [canonicalize_axis(a, x.ndim) for a in src]
  dst = [canonicalize_axis(a, x.ndim) for a in dst]
  perm = [i for i in range(np.ndim(x)) if i not in src]
  for d, s in sorted(zip(dst, src)):
    perm.insert(d, s)
  return x.transpose(perm)

def ceil_of_ratio(x, y):
  return -(-x // y)


def wraps(
    wrapped: Callable,
    namestr: Optional[str] = None,
    docstr: Optional[str] = None,
    **kwargs,
) -> Callable[[T], T]:
  """
  Like functools.wraps, but with finer-grained control over the name and docstring
  of the resulting function.
  """
  def wrapper(fun: T) -> T:
    try:
      name = getattr(wrapped, "__name__", "<unnamed function>")
      doc = getattr(wrapped, "__doc__", "") or ""
      fun.__dict__.update(getattr(wrapped, "__dict__", {}))
      fun.__annotations__ = getattr(wrapped, "__annotations__", {})
      fun.__name__ = name if namestr is None else namestr.format(fun=name)
      fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
      fun.__doc__ = (doc if docstr is None
                     else docstr.format(fun=name, doc=doc, **kwargs))
      fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
      fun.__wrapped__ = wrapped
    finally:
      return fun
  return wrapper


# NOTE: Ideally we would annotate both the argument and return type as NoReturn
#       but it seems like pytype doesn't support that...
def assert_unreachable(x):
  raise AssertionError(f"Unhandled case: {type(x).__name__}")

def tuple_insert(t, idx, val):
  assert 0 <= idx <= len(t), (idx, len(t))
  return t[:idx] + (val,) + t[idx:]

def tuple_delete(t, idx):
  assert 0 <= idx < len(t), (idx, len(t))
  return t[:idx] + t[idx + 1:]

class HashableFunction:
  """Decouples function equality and hash from its identity.

  Local lambdas and function defs are reallocated on each function call, making
  the functions created on different calls compare as unequal. This breaks our
  caching logic, which should really only care about comparing the semantics and
  not actual identity.

  This class makes it possible to compare different functions based on their
  semantics. The parts that are taken into account are: the bytecode of the
  wrapped function (which is cached by the CPython interpreter and is stable
  across the invocations of the surrounding function), and `closure` which
  should contain all values in scope that affect the function semantics. In
  particular `closure` should contain all elements of the function closure, or
  it should be possible to derive the relevant elements of the true function
  closure based solely on the contents of the `closure` argument (e.g. in case
  some closed-over values are not hashable, but are entirely determined by
  hashable locals).
  """

  def __init__(self, f, closure):
    self.f = f
    self.closure = closure

  def __eq__(self, other):
    return (type(other) is HashableFunction and
            self.f.__code__ == other.f.__code__ and
            self.closure == other.closure)

  def __hash__(self):
    return hash((self.f.__code__, self.closure))

  def __call__(self, *args, **kwargs):
    return self.f(*args, **kwargs)

  def __repr__(self):
    return f'<hashable {self.f.__name__} with closure={self.closure}>'

def as_hashable_function(closure):
  return lambda f: HashableFunction(f, closure)

class HashablePartial:
  def __init__(self, f, *args, **kwargs):
    self.f = f
    self.args = args
    self.kwargs = kwargs

  def __eq__(self, other):
    return (type(other) is HashablePartial and
            self.f.__code__ == other.f.__code__ and
            self.args == other.args and self.kwargs == other.kwargs)

  def __hash__(self):
    return hash((self.f.__code__, self.args, tuple(self.kwargs.items())))

  def __call__(self, *args, **kwargs):
    return self.f(*self.args, *args, **self.kwargs, **kwargs)

def maybe_named_axis(axis, if_pos, if_named):
  try:
    pos = operator.index(axis)
    named = False
  except TypeError:
    named = True
  return if_named(axis) if named else if_pos(pos)

def distributed_debug_log(*pairs):
  """Format and log `pairs` if config.jax_distributed_debug is enabled.

  Args:
    pairs: A sequence of label/value pairs to log. The first pair is treated as
    a heading for subsequent pairs.
  """
  if config.jax_distributed_debug:
    lines = ["\nDISTRIBUTED_DEBUG_BEGIN"]
    try:
      lines.append(f"{pairs[0][0]}: {pairs[0][1]}")
      for label, value in pairs[1:]:
        lines.append(f"  {label}: {value}")
    except Exception as e:
      lines.append("DISTRIBUTED_DEBUG logging failed!")
      lines.append(f"{e}")
    lines.append("DISTRIBUTED_DEBUG_END")
    logger.warning("\n".join(lines))


class OrderedSet(Generic[T]):
  elts_set: set[T]
  elts_list: list[T]

  def __init__(self):
    self.elts_set = set()
    self.elts_list = []

  def add(self, elt: T) -> None:
    if elt not in self.elts_set:
      self.elts_set.add(elt)
      self.elts_list.append(elt)

  def update(self, elts: Seq[T]) -> None:
    for e in elts:
      self.add(e)

  def __iter__(self) -> Iterator[T]:
    return iter(self.elts_list)

  def __len__(self) -> int:
    return len(self.elts_list)

  def __contains__(self, elt: T) -> bool:
    return elt in self.elts_set


class HashableWrapper:
  x: Any
  hash: Optional[int]
  def __init__(self, x):
    self.x = x
    try: self.hash = hash(x)
    except: self.hash = None
  def __hash__(self):
    return self.hash if self.hash is not None else id(self.x)
  def __eq__(self, other):
    if not isinstance(other, HashableWrapper):
      return False
    return self.x == other.x if self.hash is not None else self.x is other.x


def _original_func(f):
  if isinstance(f, property):
    return cast(property, f).fget
  elif isinstance(f, functools.cached_property):
    return f.func
  return f


def set_module(module):
  def wrapper(func):
    if module is not None:
      func.__module__ = module
    return func
  return wrapper


if TYPE_CHECKING:
  def use_cpp_class(cpp_cls: Any) -> Callable[[T], T]:
    def wrapper(cls: T) -> T:
      return cls
    return wrapper

  def use_cpp_method(is_enabled: bool = True) -> Callable[[T], T]:
    def wrapper(cls: T) -> T:
      return cls
    return wrapper

else:
  def use_cpp_class(cpp_cls):
    """A helper decorator to replace a python class with its C++ version"""

    def wrapper(cls):
      if cpp_cls is None:
        return cls

      exclude_methods = {'__module__', '__dict__', '__doc__'}

      originals = {}
      for attr_name, attr in cls.__dict__.items():
        if attr_name not in exclude_methods:
          if hasattr(_original_func(attr), "_use_cpp"):
            originals[attr_name] = attr
          else:
            setattr(cpp_cls, attr_name, attr)

      cpp_cls.__doc__ = cls.__doc__
      # TODO(pschuh): Remove once fastpath is gone.
      cpp_cls._original_py_fns = originals
      return cpp_cls

    return wrapper

  def use_cpp_method(is_enabled=True):
    """A helper decorator to exclude methods from the set that are forwarded to C++ class"""
    def decorator(f):
      if is_enabled:
        original_func = _original_func(f)
        original_func._use_cpp = True
      return f

    if not isinstance(is_enabled, bool):
      raise TypeError(
          "Decorator got wrong type: @use_cpp_method(is_enabled: bool=True)"
      )
    return decorator
