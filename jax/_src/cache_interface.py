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

from __future__ import annotations

import abc
import logging
import pathlib
from typing import Callable
import zlib

# If zstandard is installed, we use zstd compression, otherwise we use zlib.
try:
  # compression.zstd should be present in Python 3.14+
  from compression import zstd  # pyrefly: ignore[missing-import]
except ImportError:
  zstd = None

if zstd is None:
  # TODO(phawkins): remove this case when we drop support for Python 3.13.
  try:
    import zstandard  # pyrefly: ignore[missing-import]
  except ImportError:
    zstandard = None
else:
  zstandard = None

from jax._src import util


class CacheInterface(util.StrictABC):
  _path: pathlib.Path

  @abc.abstractmethod
  def get(self, key: str):
    pass

  @abc.abstractmethod
  def put(self, key: str, value: bytes):
    pass


def compress_bytes(data: bytes) -> bytes:
  """Compresses bytes using zstd (if available) or zlib."""
  if zstd:
    return zstd.compress(data)
  elif zstandard:
    compressor = zstandard.ZstdCompressor()
    return compressor.compress(data)
  else:
    return zlib.compress(data)


def decompress_bytes(data: bytes) -> bytes:
  """Decompresses bytes using zstd (if available) or zlib."""
  if zstd:
    return zstd.decompress(data)
  elif zstandard:
    decompressor = zstandard.ZstdDecompressor()
    return decompressor.decompress(data)
  else:
    return zlib.decompress(data)


def compression_algorithm() -> str:
  """Returns the active compression algorithm name for cache key hashing."""
  return "zstandard" if (zstd is not None or zstandard is not None) else "zlib"


def initialize_cache_once(
    *,
    cache_name: str,
    path: str | None,
    is_enabled: bool,
    get_file_cache_fn: Callable[[str], tuple[CacheInterface, str] | None],
    logger: logging.Logger,
    on_before_create: Callable[[], None] | None = None,
) -> tuple[bool, CacheInterface | None]:
  """Helper to initialize a persistent file cache once when enabled and path is set.

  Returns:
    (initialized_attempted, cache_instance): `initialized_attempted` is True if
    both `path` and `is_enabled` were truthy and initialization ran.
  """
  if not path:
    return False, None

  if not is_enabled:
    logger.debug("_initialize_cache: %s cache is disabled!", cache_name)
    return False, None

  if on_before_create is not None:
    on_before_create()

  cache_and_path = get_file_cache_fn(path)
  if cache_and_path is None:
    logger.debug("_initialize_cache: %s cache initialization failed!", cache_name)
    return True, None
  cache, resolved_path = cache_and_path
  logger.debug("Initialized persistent %s cache at %s", cache_name, resolved_path)
  return True, cache
