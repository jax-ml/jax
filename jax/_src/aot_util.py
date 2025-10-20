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
import hashlib
from typing import Any, Callable, NamedTuple

from absl import logging
import dill as pickle
from jax._src import config
from jax._src import path as pathlib


def _validate_component_cache(val):
  assert val is None or isinstance(val, Cache)


component_cache = config.string_or_object_state(
    name="jax_component_cache",
    default=None,
    help="""
    Cache directory for JAX components. If not set, components will not be
    cached.
    """,
    validator=_validate_component_cache,
)


def hashable_to_sha256_string(item: Hashable) -> str:
  """Converts a hashable Python object into a deterministic SHA256 string.

  This function is designed to create a canonical string for any hashable
  type, which is especially useful for serialization or creating consistent
  identifiers from complex keys.

  For frozensets, it ensures the output is ordered, so two equal frozensets
  will always produce the same string. For all other types, it uses repr()
  for an unambiguous representation.

  Args:
      item: A hashable object (e.g., int, str, tuple, frozenset).

  Returns:
      A SHA256 hash of the item.
  """
  if isinstance(item, frozenset):
    # To create a canonical representation for a frozenset, we recursively
    # convert each element to a string, sort the results, and then join them.
    # This guarantees that frozenset({1, 'a'}) and frozenset({'a', 1})
    # produce the exact same output string.
    sorted_elements = sorted([hashable_to_sha256_string(x) for x in item])
    msg = f"frozenset({', '.join(sorted_elements)})"

  elif isinstance(item, tuple):
    # Recursively process elements of a tuple.
    elements = [hashable_to_sha256_string(x) for x in item]
    # Add a trailing comma for single-element tuples to be unambiguous
    trailing_comma = "," if len(elements) == 1 else ""
    msg = f"({', '.join(elements)}{trailing_comma})"

  # For all other hashable types (int, float, str, bool, custom objects),
  # repr() provides an unambiguous string representation.
  # For example, repr("hello") is "'hello'", which distinguishes it from
  # the bare word hello.
  else:
    msg = repr(item)

  return hashlib.sha256(msg.encode("utf-8")).hexdigest()


def get_cached_or_put(key, make, serialize, deserialize):
  if (cache := component_cache.value) is None:
    logging.info("Component cache is not set.")
    return make()

  if blob := cache.get(key):  # pytype: disable=attribute-error
    logging.info("Key %s found.", key)
    return deserialize(blob)

  logging.info("Key %s missing.", key)
  obj = make()
  logging.info("Putting key %s.", key)
  cache.put(key, serialize(obj))  # pytype: disable=attribute-error
  logging.info("Cache keys: %s", cache.keys())
  return obj


class Cache(NamedTuple):

  get: Callable[[Hashable], bytes | None]
  put: Callable[[Hashable, bytes], None]
  keys: Callable[[], list[Hashable]]
  # TODO(dsuo): Maybe other functions like clear, size, etc.


_in_memory_cache = {}


def make_in_memory_cache():
  """In-memory cache. Allow data to be `Any` to not worry about serialize."""

  def get(key: Hashable) -> bytes | Any | None:
    return _in_memory_cache.get(key, None)

  def put(key: Hashable, data: bytes | Any):
    _in_memory_cache[key] = data

  def keys() -> list[Hashable]:
    return list(_in_memory_cache.keys())

  return Cache(get, put, keys)


def make_file_system_cache(cache_dir: str):

  def get(key: Hashable) -> bytes | None:
    key = pathlib.Path(f"{cache_dir}/{hashable_to_sha256_string(key)}")
    if not key.exists():
      return None
    with key.open("rb") as f:
      return f.read()

  def put(key: Hashable, data: bytes):
    key = pathlib.Path(f"{cache_dir}/{hashable_to_sha256_string(key)}")
    key.parent.mkdir(parents=True, exist_ok=True)
    with key.open("wb") as f:
      f.write(data)

  def keys() -> list[Hashable]:
    raise NotImplementedError

  return Cache(get, put, keys)


def serialize_abstract_eval(out: Any) -> bytes:
  return pickle.dumps(out)


def deserialize_abstract_eval(blob: bytes) -> Any:
  return pickle.loads(blob)


# TODO(dsuo): For now don't serialize.
def serialize_lowering(out: Any) -> bytes | Any:
  return out


def deserialize_lowering(blob: bytes | Any) -> Any:
  return blob
