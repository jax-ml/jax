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

from __future__ import annotations

from collections.abc import Callable, Iterator
import dataclasses
import functools
from typing import Any, Type, TYPE_CHECKING
import weakref

from jax._src import config
from jax._src.lib import pytree
from jax._src.lib import version as jaxlib_version
from jax._src.lib import weakref_lru_cache
from jax._src import tree_util
from jax._src import util


def _ignore(): return None


@dataclasses.dataclass(frozen=True, slots=True, weakref_slot=True)
class MultiWeakRefCacheKey:
  """Used for multiple weakrefs in jaxlib < (0, 8)."""
  weakrefs: tuple[weakref.ref, ...]


# Stands for an arg/kwarg that was replaced with a weakref
class MultiWeakRefPlaceholder:
  pass

_multi_weakref_placeholder = MultiWeakRefPlaceholder()

# The types of arguments for which `multi_weakref_lru_cache` should keep
# weak references.
weakref_cache_key_types: set[Type] = set()
def is_weakref_cache_key_type(v):
  return callable(v) or (type(v) in weakref_cache_key_types)

# TODO(phawkins, necula): We're using our own pytree registry to mimic the
# behavior of the original code that only looked for weakrefs as children of
# tuples and dicts. We also want to preserve the object identity of namedtuples
# at the moment because otherwise we break at least on `is` test in the PRNG
# library.
tuple_dict_registry = pytree.PyTreeRegistry(
    enable_none=False, enable_tuple=True, enable_namedtuple=False,
    enable_list=False, enable_dict=True)


def multi_weakref_lru_cache(
      call: Callable, *,
      maxsize=2048,
      trace_context_in_key: bool = True):
  """
  Least recently used cache decorator with weakref support.

  Similar to `weakref_lru_cache`, except that it keeps weak references
  to all positional and keyword arguments for which
  `is_weakref_cache_key_type()` is true, and strong references to
  other arguments. The cache entry is removed if any of the weakref
  arguments dies.
  """
  if not TYPE_CHECKING and jaxlib_version >= (0, 8, 1):
    def cache_miss(weakref_key: tuple,
                   treedef: tree_util.PyTreeDef,
                   non_weak_leaves: tuple[Any, ...]):
      key_iter = iter(weakref_key)
      value_leaves = []
      for leaf in non_weak_leaves:
        if leaf is _multi_weakref_placeholder:
          value_leaves.append(next(key_iter))
        else:
          value_leaves.append(leaf)
      orig_args, orig_kwargs = treedef.unflatten(value_leaves)
      return call(*orig_args, **orig_kwargs)

    cached_call = weakref_lru_cache.weakref_lru_cache(
        config.trace_context if trace_context_in_key else _ignore,
        cache_miss, maxsize
    )
    util.register_cache(cached_call, str(call))

    @functools.wraps(call)
    def wrapper(*orig_args, **orig_kwargs):
      leaves, treedef = tuple_dict_registry.flatten((orig_args, orig_kwargs))
      weakref_objs = []
      for i, leaf in enumerate(leaves):
        if is_weakref_cache_key_type(leaf):
          weakref_objs.append(leaf)
          leaves[i] = _multi_weakref_placeholder
      return cached_call(tuple(weakref_objs), treedef, tuple(leaves))

    wrapper.cache_info = cached_call.cache_info
    wrapper.cache_clear = cached_call.cache_clear
    wrapper.cache_keys = cached_call.cache_keys
    return wrapper

  else:
    # Keep strong references to the MultiWeakRefCacheKeys that resulted in
    # cache misses, and are cache keys. Indexed by id. Only keys with all
    # included weakrefs live are present.
    id_to_key: dict[int, MultiWeakRefCacheKey] = {}
    # For each `wr: weakref.ref` present in `key: MultiWeakRefCacheKey` we have
    # `id(key) in weakref_to_key_ids[wr]`.
    weakref_to_key_ids: dict[weakref.ref, set[int]] = {}

    def remove_weakref(wr: weakref.ref):
      key_ids = weakref_to_key_ids.get(wr, set())
      for key_id in key_ids:
        try:
          del id_to_key[key_id]
        except KeyError:
          pass
      try:
        del weakref_to_key_ids[wr]
      except KeyError:
        pass

    def weakrefs_to_sentinel(v, acc: list[Any]):
      if type(v) is tuple:
        return tuple(weakrefs_to_sentinel(v1, acc) for v1 in v)
      elif type(v) is dict:
        return {k: weakrefs_to_sentinel(v1, acc) for k, v1 in v.items()}
      elif is_weakref_cache_key_type(v):
        acc.append(v)
        return _multi_weakref_placeholder
      else:
        return v

    def sentinel_to_referrents(v,
                               it: Iterator[weakref.ref],
                               key_id: int | None):
      # key_id is not None iff we use a MultiWeakRefCacheKey (>= 2 weakrefs)
      if type(v) is tuple:
        return tuple(sentinel_to_referrents(v1, it, key_id) for v1 in v)
      elif type(v) is dict:
        return {k: sentinel_to_referrents(v1, it, key_id)
                for k, v1 in v.items()}
      elif v is _multi_weakref_placeholder:
        wr = next(it)
        if key_id is not None:
          weakref_to_key_ids.setdefault(wr, set()).add(key_id)
        return wr()
      else:
        return v

    def cache_miss(key: MultiWeakRefCacheKey | MultiWeakRefPlaceholder | Any,
                   *args, **kwargs):
      if isinstance(key, MultiWeakRefCacheKey):  # had at least 2 weakrefs
        # We know `key` is in `cached_call` cache, so store strong references
        key_id = id(key)
        id_to_key[key_id] = key
        orig_args, orig_kwargs = sentinel_to_referrents(
            (args, kwargs), iter(key.weakrefs), key_id)
      elif key is _multi_weakref_placeholder:  # had 0 weakrefs
        orig_args = args
        orig_kwargs = kwargs
      else:  # had 1 weakref, we had put it first as the `key`
        orig_args, orig_kwargs = sentinel_to_referrents(
            (args, kwargs), iter([weakref.ref(key)]), None)
      return call(*orig_args, **orig_kwargs)

    cached_call = weakref_lru_cache.weakref_lru_cache(
        config.trace_context if trace_context_in_key else _ignore,
        cache_miss, maxsize
    )
    util.register_cache(cached_call, str(call))

    @functools.wraps(call)
    def wrapper(*orig_args, **orig_kwargs):
      acc_weakrefs: list[Any] = []
      args, kwargs = weakrefs_to_sentinel((orig_args, orig_kwargs),
                                          acc_weakrefs)
      nr_weakrefs = len(acc_weakrefs)
      if nr_weakrefs == 0:
        return cached_call(_multi_weakref_placeholder,
                           *orig_args, **orig_kwargs)
      elif nr_weakrefs == 1:
        return cached_call(acc_weakrefs[0],
                           *args, **kwargs)
      else:
        value_to_weakref = {v: weakref.ref(v, remove_weakref)
                            for v in set(acc_weakrefs)}
        key = MultiWeakRefCacheKey(weakrefs=tuple(value_to_weakref[v]
                                                  for v in acc_weakrefs))
        return cached_call(key, *args, **kwargs)

    wrapper.cache_info = cached_call.cache_info
    wrapper.cache_clear = cached_call.cache_clear
    wrapper.cache_keys = cached_call.cache_keys
    wrapper._multi_weakref_id_to_key = id_to_key  # stays alive as long as wrapper
    wrapper._multi_weakref_to_key_ids = weakref_to_key_ids
    return wrapper
