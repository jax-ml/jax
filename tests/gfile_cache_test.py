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

import tempfile
import threading

from absl.testing import absltest

from jax._src.gfile_cache import GFileCache
import jax._src.test_util as jtu


class FileSystemCacheTest(jtu.JaxTestCase):

  def test_get_nonexistent_key(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = GFileCache(tmpdir)
      self.assertEqual(cache.get("nonExistentKey"), None)

  def test_put_and_get_key(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = GFileCache(tmpdir)
      cache.put("foo", b"bar")
      self.assertEqual(cache.get("foo"), b"bar")

  def test_existing_cache_path(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache1 = GFileCache(tmpdir)
      cache1.put("foo", b"bar")
      del cache1
      cache2 = GFileCache(tmpdir)
      self.assertEqual(cache2.get("foo"), b"bar")

  def test_empty_value_put(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = GFileCache(tmpdir)
      cache.put("foo", b"")
      self.assertEqual(cache.get("foo"), b"")

  def test_empty_key_put(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = GFileCache(tmpdir)
      with self.assertRaisesRegex(ValueError, r"key cannot be empty"):
        cache.put("", b"bar")

  def test_empty_key_get(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache = GFileCache(tmpdir)
      with self.assertRaisesRegex(ValueError, r"key cannot be empty"):
        cache.get("")

  def test_threads(self):
    file_contents1 = "1" * (65536 + 1)
    file_contents2 = "2" * (65536 + 1)

    def call_multiple_puts_and_gets(cache):
      for _ in range(50):
        cache.put("foo", file_contents1.encode("utf-8").strip())
        cache.put("foo", file_contents2.encode("utf-8").strip())
        cache.get("foo")
        self.assertEqual(
            cache.get("foo"), file_contents2.encode("utf-8").strip()
        )

    with tempfile.TemporaryDirectory() as tmpdir:
      cache = GFileCache(tmpdir)
      threads = []
      for _ in range(50):
        t = threading.Thread(target=call_multiple_puts_and_gets(cache))
        t.start()
        threads.append(t)
      for t in threads:
        t.join()

      self.assertEqual(cache.get("foo"), file_contents2.encode("utf-8").strip())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
