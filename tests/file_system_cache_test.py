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

from absl.testing import absltest
from jax.experimental.compilation_cache.file_system_cache import FileSystemCache
import jax.test_util as jtu
import tempfile
import time

class FileSystemCacheTest(jtu.JaxTestCase):

  def test_get_nonexistent_key(self):
   with tempfile.TemporaryDirectory() as tmpdir:
     cache = FileSystemCache(tmpdir)
     self.assertEqual(cache.get("nonExistentKey"), None)

  def test_put_and_get_key(self):
   with tempfile.TemporaryDirectory() as tmpdir:
     cache = FileSystemCache(tmpdir)
     cache.put("foo", b"bar")
     self.assertEqual(cache.get("foo"), b"bar")

  def test_existing_cache_path(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache1 = FileSystemCache(tmpdir)
      cache1.put("foo", b"bar")
      del cache1
      cache2 = FileSystemCache(tmpdir)
      self.assertEqual(cache2.get("foo"), b"bar")

  def test_empty_value_put(self):
    with tempfile.TemporaryDirectory() as tmpdir:
     cache = FileSystemCache(tmpdir)
     cache.put("foo", b"")
     self.assertEqual(cache.get("foo"), b"")

  def test_empty_key_put(self):
    with tempfile.TemporaryDirectory() as tmpdir:
     cache = FileSystemCache(tmpdir)
     with self.assertRaisesRegex(ValueError , r"key cannot be empty"):
       cache.put("", b"bar")

  def test_empty_key_get(self):
    with tempfile.TemporaryDirectory() as tmpdir:
     cache = FileSystemCache(tmpdir)
     with self.assertRaisesRegex(ValueError , r"key cannot be empty"):
       cache.get("")

  def test_size_of_directory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = FileSystemCache(tmpdir)
      cache.put("foo", b"bar")
      self.assertEqual(cache._get_cache_directory_size(), 3)

  def test_size_of_empty_directory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = FileSystemCache(tmpdir)
      self.assertEqual(cache._get_cache_directory_size(), 0)

  def test_size_of_existing_directory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache1 = FileSystemCache(tmpdir)
      cache1.put("foo", b"bar")
      del cache1
      cache2 = FileSystemCache(tmpdir)
      self.assertEqual(cache2._get_cache_directory_size(), 3)

  def test_cache_is_full(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = FileSystemCache(tmpdir, max_cache_size_bytes=6)
      cache.put("first", b"one")
      # Sleep because otherwise these operations execute too fast and
      # the access time isn't captured properly.
      time.sleep(1)
      cache.put("second", b"two")
      cache.put("third", b"the")
      self.assertEqual(cache.get("first"), None)
      self.assertEqual(cache.get("second"), b"two")
      self.assertEqual(cache.get("third"), b"the")

  def test_delete_multiple_files(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = FileSystemCache(tmpdir, max_cache_size_bytes=6)
      cache.put("first", b"one")
      # Sleep because otherwise these operations execute too fast and
      # the access time isn't captured properly.
      time.sleep(1)
      cache.put("second", b"two")
      cache.put("third", b"three")
      self.assertEqual(cache.get("first"), None)
      self.assertEqual(cache.get("second"), None)
      self.assertEqual(cache.get("third"), b"three")

  def test_least_recently_accessed_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = FileSystemCache(tmpdir, max_cache_size_bytes=6)
      cache.put("first", b"one")
      cache.put("second", b"two")
      # Sleep because otherwise these operations execute too fast and
      # the access time isn't captured properly.
      time.sleep(1)
      cache.get("first")
      cache.put("third", b"the")
      self.assertEqual(cache.get("first"), b"one")
      self.assertEqual(cache.get("second"), None)

  @jtu.ignore_warning(message=("Cache value of size 3 is larger than the max cache size of 2"))
  def test_file_bigger_than_cache(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = FileSystemCache(tmpdir, max_cache_size_bytes=2)
      cache.put("foo", b"bar")
      self.assertEqual(cache.get("foo"), None)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
