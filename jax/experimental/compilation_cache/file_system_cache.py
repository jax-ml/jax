# Copyright 2021 Google LLC
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

import os
from jax.experimental.compilation_cache.cache_interface import CacheInterface
import tempfile
from typing import Optional
import warnings

class FileSystemCache(CacheInterface):

  def __init__(self, path: str, max_cache_size_bytes=32 * 2**30):
    """Sets up a cache at 'path'. Cached values may already be present."""
    os.makedirs(path, exist_ok=True)
    self._path = path
    self._max_cache_size_bytes = max_cache_size_bytes

  def get(self, key: str) -> Optional[bytes]:
    """Returns None if 'key' isn't present."""
    if not key:
      raise ValueError("key cannot be empty")
    path_to_key = os.path.join(self._path, key)
    if os.path.exists(path_to_key):
      with open(path_to_key, "rb") as file:
        return file.read()
    else:
      return None

  def put(self, key: str, value: bytes):
    """Adds new cache entry, possibly evicting older entries."""
    if not key:
      raise ValueError("key cannot be empty")
    if self._evict_entries_if_necessary(key, value):
      path_to_new_file = os.path.join(self._path, key)
      # Create the path for the file in a temporary directory so we can use the
      # atomic move function to ensure that the file is properly stored and read
      # in the case of concurrent access across multiple threads or processes
      with tempfile.TemporaryDirectory() as tmpdir:
          temp_path_to_file = os.path.join(tmpdir, key)
          with open(temp_path_to_file, "wb") as file:
              file.write(value)
              file.flush()
              os.fsync(file.fileno())
          os.rename(temp_path_to_file, path_to_new_file)
    else:
      warnings.warn(f"Cache value of size {len(value)} is larger than"
                    f" the max cache size of {self._max_cache_size_bytes}")

  def _evict_entries_if_necessary(self, key: str, value: bytes) -> bool:
    """Returns True if there's enough space to add 'value', False otherwise."""
    new_file_size = len(value)

    if new_file_size >= self._max_cache_size_bytes:
      return False

    while new_file_size + self._get_cache_directory_size() > self._max_cache_size_bytes:
      last_time = float('inf')
      file_to_delete = None
      for file_name in os.listdir(self._path):
        file_to_inspect = os.path.join(self._path, file_name)
        atime = os.stat(file_to_inspect).st_atime
        if atime < last_time:
          last_time = atime
          file_to_delete = file_to_inspect
      assert file_to_delete
      os.remove(file_to_delete)
    return True

  def _get_cache_directory_size(self):
    """Retrieves the current size of the directory, self.path"""
    return sum(os.path.getsize(f) for f in os.scandir(self._path) if f.is_file())
