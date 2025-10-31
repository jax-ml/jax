# Copyright 2025 The JAX Authors.
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
"""JAX AOT API utilities."""

from collections.abc import Hashable
import functools
import pickle
from typing import Any, Callable, NamedTuple, Self, Sequence

from absl import logging
from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import mesh as mesh_lib
from jax._src import pjit
from jax._src import stages
from jax._src import tree_util
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib import jax_jit
from jax._src.lib import xla_client as xc


# For now, we don't worry about serialization.
SerializedType = bytes | Any


class ComponentKey:
  def __init__(self, user_key: Hashable):
    self.user_key = user_key

  def __hash__(self):
    return hash(self.user_key)

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __str__(self):
    return self.user_key

  def __repr__(self):
    return self.__str__


def _validate_component_cache(val):
  assert val is None or isinstance(val, Cache)


component_cache = config.string_or_object_state(
  name="jax_component_cache",
  default=None,
  help="Cache dir for components. Components won't be cached if None.",
  validator=_validate_component_cache,
)


class TracedCacheEntry:
  def __init__(self, traced: stages.Traced, hits: int = 0):
    self.traced = traced
    self.hits = hits


_traced_cache: dict[Hashable, TracedCacheEntry] = {}


def get_traced(key: Hashable, fun: Callable[..., Any], *args):
  entry = _traced_cache.get(key, None)
  if entry:
    entry.hits += 1
  else:
    entry = _traced_cache[key] = TracedCacheEntry(api.trace(fun, *args))
  return entry.traced


class CacheEntry:
  def __init__(
    self,
    avals_out: Sequence[core.AbstractValue] | None,
    module: ir.Module | None = None,
    hits: int = 0,
  ):
    self.avals_out = avals_out
    self.module = module
    self.hits = hits

  def serialize(self) -> SerializedType:
    module_bytecode = mlir.module_to_bytecode(self.module)
    return pickle.dumps((self.avals_out, module_bytecode))

  @classmethod
  def deserialize(cls, blob: SerializedType) -> Self:
    avals_out, module = pickle.loads(blob)
    return cls(avals_out, module)


class Cache(NamedTuple):
  get: Callable[[ComponentKey], bytes | None]
  put: Callable[[ComponentKey, bytes], None]
  keys: Callable[[], list[ComponentKey]]
  clear: Callable[[], None]


_in_memory_cache: dict[ComponentKey, SerializedType] = {}
_in_memory_cache_hits: dict[ComponentKey, int] = {}


def make_in_memory_cache():
  def get(key: ComponentKey) -> SerializedType | None:
    hits = _in_memory_cache_hits.setdefault(key, 0)
    _in_memory_cache_hits[key] += 1
    return _in_memory_cache.get(key, None)

  def put(key: ComponentKey, data: SerializedType):
    _in_memory_cache[key] = data

  def keys() -> list[ComponentKey]:
    return list(_in_memory_cache.keys())

  def clear():
    _in_memory_cache.clear()

  return Cache(get, put, keys, clear)


def get_entry(
  key: ComponentKey, make: Callable[[], CacheEntry]
) -> CacheEntry:
  cache: Cache = component_cache.value
  if cache is None:
    return make()
  if blob := cache.get(key):
    return CacheEntry.deserialize(blob)
  entry = make()
  blob = entry.serialize()
  cache.put(key, blob)
  return entry
