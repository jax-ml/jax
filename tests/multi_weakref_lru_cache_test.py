# Copyright 2020 The JAX Authors.
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
import dataclasses
import random
from functools import partial
import gc
import threading

from absl.testing import absltest, parameterized
import jax
from jax._src import multi_weakref_lru_cache
from jax._src import test_util as jtu
from jax._src import util

jax.config.parse_flags_with_absl()


class MultiWeakrefLruCacheTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      dict(weakref_count=weakref_count,
           testcase_name=f"_{weakref_count=}")
      for weakref_count in [0, 1, 3])
  def test_multi_weak_ref_cache(self, *, weakref_count=1):

    class Key:  # hashed by id
      def __init__(self, x):
        self.x = x

    if weakref_count > 0:
      multi_weakref_lru_cache.weakref_cache_key_types.add(Key)

    @partial(multi_weakref_lru_cache.multi_weakref_lru_cache, trace_context_in_key=False)
    def myfun(a, k1, *, k2, k3):
      return f"{a=}, {k1=}, {k2=}, {k3=}"

    k1 = Key(1)
    k3 = (k1, k1) if weakref_count > 1 else 4
    util.clear_all_caches()
    r1 = myfun(2, k1, k2=3, k3=k3)  # miss
    c1 = myfun.cache_info()
    self.assertEqual((0, 1, 1), (c1.hits, c1.misses, c1.currsize))

    for i in range(10):
      r2 = myfun(2, k1, k2=3, k3=k3)  # all hits
      self.assertIs(r1, r2)
      c2 = myfun.cache_info()
      self.assertEqual((1 + i, 1, 1), (c2.hits, c2.misses, c2.currsize))

    del k1, k3  # expect that the cache entries are removed (if weakref_count > 0)
    gc.collect()
    c3 = myfun.cache_info()
    self.assertEqual(c3.currsize, 0 if weakref_count > 0 else 1)

    k1_2 = Key(2)
    k3_2 = (Key(3), Key(3)) if weakref_count > 1 else (3, 3)
    r4 = myfun(2, k1_2, k2=3, k3=k3_2)  # miss
    c4 = myfun.cache_info()
    self.assertEqual((10, 2, (1 if weakref_count > 0 else 2)), (c4.hits, c4.misses, c4.currsize))

    if weakref_count > 1:
      del k3_2  # clear the cache entry
      gc.collect()
      c5 = myfun.cache_info()
      self.assertEqual((10, 2, 0), (c5.hits, c5.misses, c5.currsize))

      k3_3 = (Key(3), Key(3))
      r6 = myfun(2, k1_2, k2=3, k3=k3_3)  # miss because Key hashed by it
      self.assertIsNot(r4, r6)
      c6 = myfun.cache_info()
      self.assertEqual((10, 3, 1), (c6.hits, c6.misses, c6.currsize))

    del k1_2
    gc.collect()
    c7 = myfun.cache_info()
    self.assertEqual(0 if weakref_count > 0 else 2, c7.currsize )

  def test_multi_weak_ref_cache_custom_tuple(self):
    class MyTuple(tuple):
      pass

    class Key:  # hashed by id
      def __init__(self, x):
        self.x = x

    multi_weakref_lru_cache.weakref_cache_key_types.add(Key)

    @partial(multi_weakref_lru_cache.multi_weakref_lru_cache, trace_context_in_key=False)
    def my_fun(a):
      self.assertIsInstance(a, MyTuple)
      return str(a)

    key = Key(1)
    my_fun(MyTuple([key, key]))

    del key
    self.assertEqual(1, my_fun.cache_info().currsize)  # cache is not cleaned

  def test_multi_weakref_lru_cache_threads(self):
    num_workers = 5
    num_live_keys_per_worker = 16
    size_key_space = 32
    @dataclasses.dataclass(frozen=True)
    class WRKey:
      f: int

    multi_weakref_lru_cache.weakref_cache_key_types.add(WRKey)

    @partial(multi_weakref_lru_cache.multi_weakref_lru_cache, maxsize=size_key_space // 2)
    def myfun(k: WRKey):
      return None

    def Worker():
      keys = [None] * num_live_keys_per_worker  # These are the live keys for this worker
      for i in range(1000):
        key_idx = random.randint(0, num_live_keys_per_worker - 1)
        key = WRKey(random.randint(0, size_key_space))
        myfun(key)
        keys[key_idx] = key  # Kill some previous key and keep this live

    workers = [threading.Thread(target=Worker()) for _ in range(num_workers)]
    for t in workers:
      t.start()
    for t in workers:
      t.join()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
