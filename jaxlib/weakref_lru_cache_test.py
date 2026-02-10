# Copyright 2023 The JAX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gc
import random
import threading
import time
import weakref

from absl.testing import absltest
from jax.jaxlib import weakref_lru_cache


class WeakrefLRUCacheTest(absltest.TestCase):

  def testMultiThreaded(self):
    insert_evs = [threading.Event() for _ in range(2)]
    insert_evs_i = 0

    class WRKey:
      pass

    class ClashingKey:

      def __eq__(self, other):
        return False

      def __hash__(self):
        return 333  # induce maximal caching problems.

    class GilReleasingCacheKey:

      def __eq__(self, other):
        nonlocal insert_evs_i
        if isinstance(other, GilReleasingCacheKey) and insert_evs_i < len(
            insert_evs
        ):
          insert_evs[insert_evs_i].set()
          insert_evs_i += 1
          time.sleep(0.01)
        return False

      def __hash__(self):
        return 333  # induce maximal caching problems.

    def CacheFn(obj, gil_releasing_cache_key):
      del obj
      del gil_releasing_cache_key
      return None

    cache = weakref_lru_cache.weakref_lru_cache(lambda: None, CacheFn, 2048)

    wrkey = WRKey()

    def Body():
      for insert_ev in insert_evs:
        insert_ev.wait()
        for _ in range(20):
          cache(wrkey, ClashingKey())

    t = threading.Thread(target=Body)
    t.start()
    for _ in range(3):
      cache(wrkey, GilReleasingCacheKey())
    t.join()

  def testAnotherMultiThreaded(self):
    num_workers = 5
    barrier = threading.Barrier(num_workers)

    def f(x, y):
      time.sleep(0.01)
      return y

    cache = weakref_lru_cache.weakref_lru_cache(lambda: None, f, 2048)

    class WRKey:

      def __init__(self, x):
        self.x = x

      def __eq__(self, other):
        return self.x == other.x

      def __hash__(self):
        return self.x

    def WorkerAddToCache():
      barrier.wait()
      for i in range(10000):
        cache(WRKey(i), i)

    def WorkerCleanCache():
      barrier.wait()
      for _ in range(10000):
        cache.cache_info()

    workers = [
        threading.Thread(target=WorkerAddToCache)
        for _ in range(num_workers - 1)
    ] + [threading.Thread(target=WorkerCleanCache)]

    for t in workers:
      t.start()

    for t in workers:
      t.join()

  def testKwargsDictOrder(self):
    miss_id = 0

    class WRKey:
      pass

    def CacheFn(obj, kwkey1, kwkey2):
      del obj, kwkey1, kwkey2
      nonlocal miss_id
      miss_id += 1
      return miss_id

    cache = weakref_lru_cache.weakref_lru_cache(lambda: None, CacheFn, 4)

    wrkey = WRKey()

    self.assertEqual(cache(wrkey, kwkey1="a", kwkey2="b"), 1)
    self.assertEqual(cache(wrkey, kwkey1="b", kwkey2="a"), 2)
    self.assertEqual(cache(wrkey, kwkey2="b", kwkey1="a"), 1)

  def testGetKeys(self):
    def CacheFn(obj, arg):
      del obj
      return arg + "extra"

    cache = weakref_lru_cache.weakref_lru_cache(lambda: None, CacheFn, 4)

    class WRKey:
      pass

    wrkey = WRKey()

    self.assertEmpty(cache.cache_keys())
    cache(wrkey, "arg1")
    cache(wrkey, "arg2")
    self.assertLen(cache.cache_keys(), 2)

  def testNonWeakreferenceableKey(self):
    class NonWRKey:
      __slots__ = ()

    non_wr_key = NonWRKey()
    with self.assertRaises(TypeError):
      weakref.ref(non_wr_key)

    cache = weakref_lru_cache.weakref_lru_cache(lambda: None, lambda x: 2048)
    for _ in range(100):
      with self.assertRaises(TypeError):
        cache(non_wr_key)

  def testCrashingKey(self):
    class WRKey:
      pass

    class CrashingKey:
      # A key that raises exceptions if eq or hash is called.

      def __eq__(self, other):
        raise ValueError("eq")

      def __hash__(self):
        raise ValueError("hash")

    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, lambda x, y: y, 2048
    )
    wrkey = WRKey()
    with self.assertRaises(ValueError):
      for _ in range(100):
        cache(wrkey, CrashingKey())

  def testPrintingStats(self):
    class WRKey:
      pass

    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, lambda x, y: y, 2048
    )
    wrkey = WRKey()
    for i in range(10):
      cache(wrkey, i)
    for i in range(5):
      cache(wrkey, i)

    self.assertEqual(
        repr(cache.cache_info()),
        "WeakrefLRUCache(hits=5, misses=10, maxsize=2048, currsize=10)",
    )

  def testGCKeys(self):
    class WRKey:

      def __init__(self, x):
        self.x = x

      def __eq__(self, other):
        return self.x == other.x

      def __hash__(self):
        return hash(self.x)

    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, lambda x, y: y, 2048
    )
    keys = [WRKey(i) for i in range(10)]
    for i in range(10):
      cache(keys[i], i)

    # Delete some keys, to exercise the weakref callback behavior.
    del keys[::2]

    for key in keys:
      cache(key, 7)

  def testTpTraverse(self):
    class WRKey:
      pass

    def CacheContextFn():
      return None

    def CallFn(x, y, *args, **kwargs):
      del x, args, kwargs
      return y

    cache = weakref_lru_cache.weakref_lru_cache(CacheContextFn, CallFn, 2048)

    keys = [WRKey() for _ in range(10)]
    values = [str(i) for i in range(10)]
    args = [str(i) for i in range(10)]
    kwargs = {"a": "b"}

    for key, value in zip(keys, values):
      cache(key, value, *args, **kwargs)

    expected_refs = (
        [
            CacheContextFn,
            CallFn,
            weakref_lru_cache.WeakrefLRUCache,
        ]
        + list(kwargs.keys())
        + list(kwargs.values())
        + [weakref.getweakrefs(key)[0] for key in keys]
        + values
        + args
    )

    # Can't use assertContainsSubset because it doesn't support kwargs since
    # dicts aren't hashable.
    for ref in expected_refs:
      self.assertIn(ref, gc.get_referents(cache))

  def testReentrantKey(self):
    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, lambda x, y: y, 2048
    )

    class WRKey:
      pass

    class ReentrantKey:

      def __eq__(self, other):
        cache(WRKey(), None)
        return False

      def __hash__(self):
        return 42

    wrkey = WRKey()
    for _ in range(100):
      cache(wrkey, ReentrantKey())

  def testRecursiveFunction(self):
    class WRKey:
      pass

    wrkey = WRKey()

    def recursive_fn(x, y):
      return cache(x, y)

    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, recursive_fn, 2048
    )

    with self.assertRaisesRegex(RecursionError, "Recursively calling"):
      cache(wrkey, 1)

  def testStressMultiThreaded(self):
    class WeakKey:

      def __init__(self, x):
        self.x = x

      def __eq__(self, other):
        time.sleep(1e-4)  # Encourage a GIL release
        return isinstance(other, WeakKey) and self.x == other.x

      def __hash__(self):
        return 42

    class StrongKey:

      def __init__(self, y):
        self.y = y

      def __eq__(self, other):
        time.sleep(1e-4)  # Encourage a GIL release
        return isinstance(other, StrongKey) and self.y == other.y

      def __hash__(self):
        return 43

    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, lambda x, y: (x, y), 2048
    )

    num_threads = 10
    actions_per_thread = 1000

    weak_keys = [WeakKey(i) for i in range(10)]
    strong_keys = [StrongKey(i) for i in range(10)]

    def worker():
      for _ in range(actions_per_thread):
        r = random.random()
        if r < 0.9:
          # Lookup
          wk = random.choice(weak_keys)
          sk = random.choice(strong_keys)
          # Occasionally use a new object that compares equal
          if random.random() < 0.1:
            wk = WeakKey(wk.x)
          if random.random() < 0.1:
            sk = StrongKey(sk.y)
          cache(wk, sk)
        elif r < 0.95:
          cache.cache_info()
        else:
          cache.cache_clear()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

  def testEvictWeakref(self):
    dtor_list = []

    class NoisyDestructor:

      def __init__(self, v):
        self.v = v

      def __del__(self):
        dtor_list.append(self.v)

    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, lambda x, y: NoisyDestructor(y)
    )

    class WRKey:
      pass

    N = 100
    expected_deletes = []
    plan = list(range(N)) * 2
    random.shuffle(plan)
    keys = [None] * N
    for i in plan:
      if keys[i] is None:
        keys[i] = WRKey()
        cache(keys[i], i)
      else:
        cache.evict_weakref(keys[i])
        expected_deletes.append(i)
        self.assertEqual(dtor_list, expected_deletes)

  def testExplain(self):

    def explain(keys, x):
      self.assertLen(keys, num_keys_should_be)

    cache = weakref_lru_cache.weakref_lru_cache(
        lambda: None, lambda x: None, explain=lambda: explain
    )

    class A:
      ...

    a = A()

    num_keys_should_be = 0
    cache(a)

    num_keys_should_be = 1
    b = A()
    cache(b)


if __name__ == "__main__":
  absltest.main()
