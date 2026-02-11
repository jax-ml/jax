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

import logging
import threading
import warnings
import zlib

import numpy as np

# If zstandard is installed, we use zstd compression, otherwise we use zlib.
try:
  # compression.zstd should be present in Python 3.14+
  from compression import zstd  # pytype: disable=import-error
except ImportError:
  zstd = None

if zstd is None:
  # TODO(phawkins): remove this case when we drop support for Python 3.13.
  try:
    import zstandard  # pytype: disable=import-error
  except ImportError:
    zstandard = None
else:
  zstandard = None

from jax._src import cache_key
from jax._src import config
from jax._src import monitoring
from jax._src.compilation_cache_interface import CacheInterface
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lru_cache import LRUCache


logger = logging.getLogger(__name__)

_cache: CacheInterface | None = None

_cache_initialized: bool = False

_cache_checked: bool = False

_cache_used: bool = False

# Mutex to protect _cache_initialized, _cache_checked and _cache_used.
_cache_initialized_mutex = threading.Lock()

_UNSUPPORTED_RUNTIMES: set[str] = set()

_TIME_BYTES = 4

def is_cache_used(backend: xla_client.Client) -> bool:
  """Check if cache is used and report adoption metrics one-time per task.
  The cache may be initialized during the first call to this function.
  """
  # Return _cache_used directly if _cache_checked is True. If _cache_checked is
  # False, set it to True, report metrics and return if cache is used. This
  # provides a mechanism to report the metrics once per task. Note that
  # reset_cache() will reset _cache_checked and _cache_used also.
  global _cache_checked, _cache_used
  with _cache_initialized_mutex:
    if _cache_checked:
      return _cache_used

  with _cache_initialized_mutex:
    if not _cache_checked:
      _cache_checked = True

      # Persistent compilation cache only implemented on TPU and GPU and the
      # backend that supports serialization of executables.
      # TODO(skye): add warning when initializing cache on unsupported default
      # platform
      supported_platforms = ["tpu", "gpu", "cpu", "neuron"]

      if not _is_cache_enabled():
        monitoring.record_event('/jax/compilation_cache/task_disabled_cache')
      elif (
          backend.platform in supported_platforms
          and getattr(backend, "supports_executable_serialization", True)
      ):
        monitoring.record_event('/jax/compilation_cache/tasks_using_cache')
        _cache_used = True
      return _cache_used

  return False


def get_file_cache(path: str) -> tuple[CacheInterface, str] | None:
  """Returns the file cache and the path to the cache."""
  max_size = config.compilation_cache_max_size.value
  cache = LRUCache(path, max_size=max_size)
  if config.compilation_cache_check_contents.value:
    return VerificationCache(cache), path
  return cache, path


class VerificationCache(CacheInterface):
  """A cache that wraps another cache and verifies its contents.

  If jax_compilation_cache_check_contents is True, then the first time
  we encounter a new key in the disk cache in this process, even if the
  disk cache already contains such an entry, we return None from get(),
  forcing a recompilation. Then, when put() is called with the compiled
  executable, we verify that it matches what's on disk.
  """

  def __init__(self, base_cache: CacheInterface):
    self._base_cache = base_cache
    self._verified_keys: set[str] = set()

  @property
  def _path(self):
    return self._base_cache._path

  def get(self, key: str) -> bytes | None:
    if key not in self._verified_keys:
      # Force a recompile the first time we see a key.
      return None

    return self._base_cache.get(key)

  def put(self, key: str, value: bytes) -> None:
    if key not in self._verified_keys:
      on_disk = self._base_cache.get(key)
      if on_disk is not None:
        # The cache content is [timestamp] + [executable].
        # We decompress both and compare skip the timestamp which will
        # differ for fresh compilations.
        decompressed_on_disk = decompress_executable(on_disk)
        decompressed_new = decompress_executable(value)
        executable_on_disk, _ = extract_executable_and_time(decompressed_on_disk)
        executable_new, _ = extract_executable_and_time(decompressed_new)
        if executable_on_disk != executable_new:
          raise RuntimeError(
              f"Persistent compilation cache inconsistency for key {key}. "
              "Executable found in the disk cache does not match the "
              "freshly compiled executable."
          )
      self._verified_keys.add(key)

    self._base_cache.put(key, value)

  def clear(self):
    self._verified_keys.clear()


def set_cache_dir(path) -> None:
  """
  Sets the persistent compilation cache directory.

  After calling this, jit-compiled functions are saved to `path`, so they
  do not need be recompiled if the process is restarted or otherwise run again.
  This also tells Jax where to look for compiled functions before compiling.
  """
  config.config.update("jax_compilation_cache_dir", path)


def initialize_cache(path) -> None:
  """
  This API is deprecated; use set_cache_dir instead.

  Set the path. To take effect, should be called prior to any calls to
  get_executable_and_time() and put_executable_and_time().
  """
  config.config.update("jax_compilation_cache_dir", path)


def default_min_cache_entry_size() -> int:
  """Returns the minimum size below which the entry should not be cached."""
  return 0


def _is_cache_enabled() -> bool:
  return config.enable_compilation_cache.value


def _initialize_cache() -> None:
  # Attempt to initialize the cache at most once.
  global _cache_initialized
  with _cache_initialized_mutex:
    if _cache_initialized:
      return

    path: str | None = config.compilation_cache_dir.value
    # If the path is not set, the cache will not be built.
    if not path:
      return

    # Nothing to do if the cache is disabled.
    if not _is_cache_enabled():
      logger.debug("_initialize_cache: cache is disabled!")
      return

    _cache_initialized = True

    # Set the minimum cache size entry only if the flag
    # --jax_persistent_cache_min_entry_size_bytes has not been set.
    if config.persistent_cache_min_entry_size_bytes.value == 0:
      config.config.update("jax_persistent_cache_min_entry_size_bytes",
                           default_min_cache_entry_size())

    global _cache
    assert _cache is None, "The cache has already been initialized!"

    cache_and_path = get_file_cache(path)
    if cache_and_path is None:
      logger.debug("_initialize_cache: cache initialization failed!")
    else:
      _cache, path = cache_and_path
      logger.debug("Initialized persistent compilation cache at %s", path)

def is_persistent_cache_enabled() -> bool:
  return (config.compilation_cache_dir.value is not None
          and config.enable_compilation_cache.value)


def _get_cache(backend) -> CacheInterface | None:
  # TODO(b/289098047): consider making this an API and changing the callers of
  # get_executable_and_time() and put_executable_and_time() to call get_cache()
  # and passing the result to them.
  if backend.runtime_type in _UNSUPPORTED_RUNTIMES:
    log_priority = (logging.WARNING if is_persistent_cache_enabled()
                    else logging.DEBUG)
    logger.log(log_priority, "_get_cache: Unsupported runtime: %s",
               backend.runtime_type)
    return None
  if _cache is None:
    _initialize_cache()  # initialization is done at most once; see above
  return _cache


def compress_executable(executable: bytes) -> bytes:
  if zstd:
    return zstd.compress(executable)
  elif zstandard:
    compressor = zstandard.ZstdCompressor()
    return compressor.compress(executable)
  else:
    return zlib.compress(executable)

def decompress_executable(executable: bytes) -> bytes:
  if zstd:
    return zstd.decompress(executable)
  elif zstandard:
    decompressor = zstandard.ZstdDecompressor()
    return decompressor.decompress(executable)
  else:
    return zlib.decompress(executable)


def is_executable_in_cache(backend, cache_key: str) -> bool:
  """Checks if the executable is in the cache."""
  cache = _get_cache(backend)
  if cache is None:
    return False

  # TODO(patrios): add check cache key method to cache interface.
  executable_and_time = cache.get(cache_key)
  return executable_and_time is not None


def get_executable_and_time(
    cache_key: str, compile_options, backend, executable_devices
) -> tuple[xla_client.LoadedExecutable | None, int | None]:
  """Returns the cached executable and its compilation time if present, or None
  otherwise.
  """
  cache = _get_cache(backend)
  if cache is None:
    logger.debug("get_executable_and_time: cache is disabled/not initialized")
    return None, None
  executable_and_time = cache.get(cache_key)
  if executable_and_time is None:
    return None, None

  executable_and_time = decompress_executable(executable_and_time)
  serialized_executable, compile_time = extract_executable_and_time(
      executable_and_time)
  xla_executable_deserialized = backend.deserialize_executable(
      serialized_executable, executable_devices, compile_options)
  return xla_executable_deserialized, compile_time


def put_executable_and_time(
    cache_key: str,
    module_name: str,
    executable: xla_client.LoadedExecutable,
    backend,
    compile_time: int
) -> None:
  """Adds the 'executable' and its compilation time to the cache, possibly
  evicting older entries.
  """
  log_priority = (logging.WARNING
                  if config.explain_cache_misses.value
                  and is_persistent_cache_enabled()
                  else logging.DEBUG)
  cache = _get_cache(backend)
  if cache is None:
    logger.log(log_priority,
               "Not writing persistent cache entry with key %r"
               " since cache is disabled/not initialized", cache_key)
    return

  if hasattr(executable, "serialize") or xla_client._version >= 389:
    serialized_executable = executable.serialize()
  else:
    serialized_executable = backend.serialize_executable(executable)
  executable_and_time = combine_executable_and_time(
      serialized_executable, compile_time)
  executable_and_time = compress_executable(executable_and_time)

  min_entry_size = config.persistent_cache_min_entry_size_bytes.value
  entry_size = len(executable_and_time)
  if entry_size < min_entry_size:
    logger.log(log_priority,
        "Not writing persistent cache entry with key %r since its size"
        " (%d bytes) is less than threshold (%d bytes)", cache_key, entry_size,
        min_entry_size)
  else:
    logger.log(log_priority,
               "Writing %s to persistent compilation cache with key %r",
               module_name, cache_key)
    monitoring.record_event('/jax/compilation_cache/cache_misses')
    if config.compilation_cache_expect_pgle.value:
      # User asserted that the compilation cache would already contain PGLE-optimized
      # executables. Because of the size/compile-time thresholds, it is expected that
      # some compilation of small modules will still happen, but that should not lead
      # to compilation cache writes.
      warnings.warn(
          f"PERSISTENT CACHE WRITE with key {cache_key}, this is unexpected because "
          "JAX_COMPILATION_CACHE_EXPECT_PGLE is set. The execution that populated the "
          "cache may lack coverage, "
          "https://docs.jax.dev/en/latest/persistent_compilation_cache.html may "
          "help debug why this has happened")

    cache.put(cache_key, executable_and_time)


def get_cache_key(
    module: ir.Module,
    devices: np.ndarray,
    compile_options,
    backend,
    ignore_callbacks: cache_key.IgnoreCallbacks = cache_key.IgnoreCallbacks.NO,
) -> str:
  return cache_key.get(
      module,
      devices,
      compile_options,
      backend,
      "zstandard" if zstandard is not None else "zlib",
      ignore_callbacks,
  )


def is_initialized() -> bool:
  """
  Deprecated.

  Return whether the cache is enabled. Initialization can be deferred, so
  initialized status is not checked. The name is retained for backwards
  compatibility.
  """
  return _is_cache_enabled()


def reset_cache() -> None:
  """Get back to pristine, uninitialized state."""
  global _cache
  global _cache_initialized
  global _cache_checked
  global _cache_used
  logger.info("Resetting cache at %s.",
               _cache._path if _cache is not None else "<empty>")
  _cache = None
  with _cache_initialized_mutex:
    _cache_initialized = False
    _cache_checked = False
    _cache_used = False


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
  return (
      int(compile_time).to_bytes(_TIME_BYTES, byteorder="big")
      + serialized_executable
  )


def extract_executable_and_time(
    executable_and_time: bytes
) -> tuple[bytes, int]:
  """Given the cache entry in the format shown below, extract the serialized
  executable and the compilation time.

  The cache entry 'executable_and_time' is of the form:
  Byte:     0    1    2    3    4 ...
  Content:  compilation time    serialized executable
            (big-endian int)
  """
  return executable_and_time[_TIME_BYTES:], int.from_bytes(
      executable_and_time[:_TIME_BYTES], byteorder='big')
