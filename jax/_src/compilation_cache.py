# Copyright 2021 The JAX Authors.
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

import hashlib
import logging
from typing import Optional
import zlib

import numpy as np

# If zstandard is installed, we use zstd compression, otherwise we use zlib.
try:
  import zstandard
except ImportError:
  zstandard = None

from jax._src import path as pathlib
from jax._src.cache_key import CacheKey
from jax._src.compilation_cache_interface import CacheInterface
from jax._src.gfile_cache import GFileCache
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir


logger = logging.getLogger(__name__)

_cache: Optional[CacheInterface] = None


def initialize_cache(path):
  """Creates a global cache object.

  Should only be called once per process.

  Will throw an assertion error if called a second time with a different path.

  Args:
    path: path for the cache directory.
  """
  global _cache
  if _cache is not None and _cache._path == pathlib.Path(path):
    logger.warning("Cache already previously initialized at %s", _cache._path)
    return

  assert (
      _cache is None
  ), f"The cache path has already been initialized to {_cache._path}"
  _cache = GFileCache(path)
  logger.warning("Initialized persistent compilation cache at %s", path)


def get_executable(
    cache_key: str, compile_options, backend
) -> Optional[xla_client.LoadedExecutable]:
  """Returns the cached executable if present, or None otherwise."""
  assert (
      _cache is not None
  ), "initialize_cache must be called before you can call get_executable()"
  serialized_executable = _cache.get(cache_key)
  if not serialized_executable:
    return None
  if zstandard:
    decompressor = zstandard.ZstdDecompressor()
    serialized_executable = decompressor.decompress(serialized_executable)
  else:
    serialized_executable = zlib.decompress(serialized_executable)
  xla_executable_deserialized = backend.deserialize_executable(
      serialized_executable, compile_options
  )
  return xla_executable_deserialized


def put_executable(
    cache_key: str,
    module_name: str,
    executable: xla_client.LoadedExecutable,
    backend,
) -> None:
  """Adds 'executable' to the cache, possibly evicting older entries."""
  assert (
      _cache is not None
  ), "initialize_cache must be called before you can call put_executable()"
  logger.info(
      "Writing %s to persistent compilation cache with key %s.",
      module_name,
      cache_key,
  )
  serialized_executable = backend.serialize_executable(executable)
  if zstandard:
    compressor = zstandard.ZstdCompressor()
    serialized_executable = compressor.compress(serialized_executable)
  else:
    serialized_executable = zlib.compress(serialized_executable)
  _cache.put(cache_key, serialized_executable)


def _log_cache_key_hash(hash_obj, last_serialized: str, hashfn):
  if logger.isEnabledFor(logging.DEBUG):
    # Log the hash of just this entry
    fresh_hash_obj = hashlib.sha256()
    hashfn(fresh_hash_obj)
    logger.debug(
        "get_cache_key hash of serialized %s: %s",
        last_serialized,
        fresh_hash_obj.digest().hex(),
    )
    # Log the cumulative hash
    logger.debug(
        "get_cache_key hash after serializing %s: %s",
        last_serialized,
        hash_obj.digest().hex(),
    )


def get_cache_key(module: ir.Module, devices: np.ndarray, compile_options,
                  backend) -> str:
  return CacheKey().get(module, devices, compile_options, backend)


def is_initialized():
  return _cache is not None


def reset_cache():
  global _cache
  assert is_initialized()
  logger.info("Resetting cache at %s.", _cache._path)
  _cache = None
