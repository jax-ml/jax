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

from typing import Optional
import os

class FileSystemCache:

  def __init__(self, path: str):
    """Sets up a cache at 'path'. Cached values may already be present."""
    os.makedirs(path, exist_ok=True)
    self._path  = path

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
      #TODO(colemanliyah):implement eviction
    with open(os.path.join(self._path, key), "wb") as file:
      file.write(value)
