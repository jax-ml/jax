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
import pickle
import traceback
from typing import Any, Callable, NamedTuple, Self, Sequence

from absl import logging
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import stages
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect


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
  if entry is None:
    entry = _traced_cache[key] = TracedCacheEntry(api.trace(fun, *args))
  else:
    entry.hits += 1
  return entry.traced


class CacheEntry:
  def __init__(
    self,
    avals_out: Sequence[core.AbstractValue] | None,
    module: ir.Module | None = None,
  ):
    self.avals_out = avals_out
    self.module = module

  def serialize(self) -> SerializedType:
    module_bytecode = None
    if self.module is not None:
      module_bytecode = mlir.module_to_bytecode(self.module)
    return pickle.dumps((self.avals_out, module_bytecode))

  @classmethod
  def deserialize(
    cls, blob: SerializedType, ctx: ir.Context | None = None
  ) -> Self:
    avals_out, module_bytecode = pickle.loads(blob)
    if module_bytecode is None or ctx is None:
      module = None
    else:
      with ctx:
        module = ir.Module.parse(module_bytecode)
    return cls(avals_out, module)


class Cache(NamedTuple):
  get: Callable[[ComponentKey], bytes | None]
  put: Callable[[ComponentKey, bytes], None]
  keys: Callable[[], list[ComponentKey]]
  clear: Callable[[], None]
  info: Callable[[ComponentKey], dict[str, Any]]


_in_memory_cache: dict[ComponentKey, SerializedType] = {}
_in_memory_cache_info: dict[ComponentKey, dict[str, Any]] = {}


def make_in_memory_cache():
  def get(key: ComponentKey) -> SerializedType | None:
    entry = _in_memory_cache.get(key, None)
    if entry is None:
      _in_memory_cache_info[key] = dict(hits=0)
    else:
      _in_memory_cache_info[key] = dict(
        hits=_in_memory_cache_info[key]["hits"] + 1
      )
    return entry

  def put(key: ComponentKey, data: SerializedType, update: bool):
    _in_memory_cache[key] = data
    if not update:
      _in_memory_cache_info[key] = dict(hits=0)

  def keys() -> list[ComponentKey]:
    return list(_in_memory_cache.keys())

  def clear() -> None:
    _in_memory_cache.clear()
    _in_memory_cache_info.clear()

  def info(key: ComponentKey) -> dict[str, Any]:
    if key not in _in_memory_cache_info:
      raise ValueError(f"`{key}` not found in _in_memory_cache_info")
    return _in_memory_cache_info[key]

  return Cache(get, put, keys, clear, info)


def get_cache() -> Cache | None:
  return component_cache.value


def get_entry(
  key: ComponentKey, ctx: ir.Context | None = None
) -> CacheEntry | None:
  if (cache := get_cache()) is not None:
    if (blob := cache.get(key)) is not None:
      return CacheEntry.deserialize(blob, ctx)
  return None  # sigh pytype


def put_entry(
  key: ComponentKey, entry: CacheEntry, update: bool = False
) -> None:
  if (cache := get_cache()) is not None:
    cache.put(key, entry.serialize(), update)
