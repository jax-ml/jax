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
import dataclasses
import functools
import pickle
from typing import Any, Callable, NamedTuple

from absl import logging
from jax._src import api
from jax._src import config
from jax._src import stages
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib import xla_client as xc


# For now, we don't worry about serialization.
SerializedType = bytes | Any


def _validate_component_cache(val):
  assert val is None or isinstance(val, Cache)


component_cache = config.string_or_object_state(
  name="jax_component_cache",
  default=None,
  help="Cache dir for components. Components won't be cached if None.",
  validator=_validate_component_cache,
)


traced_cache: dict[Hashable, stages.Traced] = {}


def trace(f: Callable[..., Any], *args, **kwargs):
  if type(f) is xc._xla.PjitFunction:
    return f.trace(*args, **kwargs)
  try:
    hash(f)
  except TypeError:
    fun = functools.partial(f)
  return api.jit(f).trace(*args, **kwargs)


class CacheEntry:
  def __init__(self, blob: SerializedType, hits: int = 0):
    self.blob = blob
    self.hits = hits

class Cache(NamedTuple):
  get: Callable[[Hashable, bool], bytes | None]
  put: Callable[[Hashable, bytes], None]
  keys: Callable[[], list[Hashable]]
  clear: Callable[[], None]


_in_memory_cache: dict[Hashable, CacheEntry] = {}


def make_in_memory_cache():
  def get(key: Hashable, update_hits: bool = True) -> SerializedType | None:
    entry = _in_memory_cache.get(key, None)
    if entry is not None and update_hits:
      _in_memory_cache[key].hits += 1
      return entry.blob
    return entry

  def put(key: Hashable, data: SerializedType):
    _in_memory_cache[key] = CacheEntry(data)

  def keys() -> list[Hashable]:
    return list(_in_memory_cache.keys())

  def clear():
    _in_memory_cache.clear()

  return Cache(get, put, keys, clear)


KeyFn = Callable[[Hashable], Hashable]
SerFn = Callable[[Any], SerializedType]
DesFn = Callable[[SerializedType], Any]

make_abstract_eval_key: KeyFn = lambda k: f"{k}.abstract_eval"
serialize_abstract_eval: SerFn = lambda obj: pickle.dumps(obj)
deserialize_abstract_eval: DesFn = lambda blob: pickle.loads(blob)
make_lowering_key: KeyFn = lambda k: f"{k}.lowering"

# TODO(dsuo): When is serialize/deserialize portable artifact needed?
serialize_lowering: SerFn = lambda obj: mlir.module_to_bytecode(obj)
deserialize_lowering: DesFn = lambda blob: ir.Module.parse(blob)


def get_cached_or_put(key, make, serialize, deserialize):
  if (cache := component_cache.value) is None:
    logging.info("Component cache is not set.")
    return make()

  if blob := cache.get(key):  # pytype: disable=attribute-error
    logging.info("Key %s found with blob %s.", key, blob)
    return deserialize(blob)

  logging.info("Key %s missing.", key)
  obj = make()
  blob = serialize(obj)
  logging.info("Putting key %s with blob %s.", key, blob)
  cache.put(key, blob)  # pytype: disable=attribute-error
  logging.info("Cache keys: %s", cache.keys())
  return obj
