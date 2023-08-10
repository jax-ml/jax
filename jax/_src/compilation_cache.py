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
from jax._src import cache_key
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


def get_executable_and_time(
    cache_key: str, compile_options, backend
) -> tuple[Optional[xla_client.LoadedExecutable], Optional[int]]:
  """Returns the cached executable and its compilation time if present, or None
  otherwise.
  """
  assert _cache is not None, (
      "initialize_cache must be called before you can call"
      " get_executable_and_time()"
  )
  executable_and_time = _cache.get(cache_key)
  if not executable_and_time:
    return None, None
  if zstandard:
    decompressor = zstandard.ZstdDecompressor()
    executable_and_time = decompressor.decompress(executable_and_time)
  else:
    executable_and_time = zlib.decompress(executable_and_time)
  serialized_executable, compile_time = extract_executable_and_time(
      executable_and_time)
  xla_executable_deserialized = backend.deserialize_executable(
      serialized_executable, compile_options)
  return xla_executable_deserialized, compile_time


def put_executable_and_time(
    cache_key: str,
    module_name: str,
    executable: xla_client.LoadedExecutable,
    backend,
    compile_time: int
) -> None:
  """Adds the 'executable' and its compilation time to the cache repository,
  possibly evicting older entries.
  """
  assert _cache is not None, (
      "initialize_cache must be called before you can call"
      "put_executable_and_time()"
  )
  logger.info(
      "Writing %s to persistent compilation cache with key %s.",
      module_name,
      cache_key,
  )
  serialized_executable = backend.serialize_executable(executable)
  executable_and_time = combine_executable_and_time(
      serialized_executable, compile_time)
  if zstandard:
    compressor = zstandard.ZstdCompressor()
    executable_and_time = compressor.compress(executable_and_time)
  else:
    executable_and_time = zlib.compress(executable_and_time)
  _cache.put(cache_key, executable_and_time)


def get_cache_key(module: ir.Module, devices: np.ndarray, compile_options,
                  backend, produce_original_cache_key: bool = True) -> str:
  return cache_key.get(module, devices, compile_options, backend,
                       "zstandard" if zstandard is not None else "zlib",
                       produce_original_cache_key)


def is_initialized():
  return _cache is not None


def reset_cache():
  global _cache
  assert is_initialized()
  logger.info("Resetting cache at %s.", _cache._path)
  _cache = None


def combine_executable_and_time(
    serialized_executable: bytes, compile_time: int
) -> bytes:
  """Given the serialized executable and the compilation time, produce a cache
  entry in the format shown below.

  The cache entry is of the form:
  Byte:     0    1    2    3    4 ...
  Content:  compilation time    serialized executable
            (big-endian int)
  """
  return int(compile_time).to_bytes(4, byteorder='big') + serialized_executable


def extract_executable_and_time(
    exectuable_and_time: bytes
) -> tuple[bytes, int]:
  """Given the cache entry in the format shown below, extract the serialized
  executable and the compilation time.

  The cache entry 'executable_and_time' is of the form:
  Byte:     0    1    2    3    4 ...
  Content:  compilation time    serialized executable
            (big-endian int)
  """
  return exectuable_and_time[4:], int.from_bytes(
      exectuable_and_time[:4], byteorder='big')
