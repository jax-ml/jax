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

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
