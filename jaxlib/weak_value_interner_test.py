# Copyright 2026 The JAX Authors.
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

import gc
import random
import threading
import time
from absl.testing import absltest
from jax._src.util import weak_value_interner


# We use a custom class because instances of the built-in `object` class
# cannot be weakly referenced.
class Object:
  pass


class WeakValueInternerTest(absltest.TestCase):

  def test_basic(self):
    calls = 0

    def f(x, y):
      nonlocal calls
      calls += 1
      obj = Object()
      obj.id = calls
      return obj

    cache = weak_value_interner(f)

    x = 1
    y = 2

    o1 = cache(x, y)
    self.assertEqual(o1.id, 1)
    o2 = cache(x, y)
    self.assertEqual(o2.id, 1)

    self.assertIs(o1, o2)
    self.assertEqual(calls, 1)

    o3 = cache(x, y=1)
    o4 = cache(x, y=1)
    self.assertIs(o3, o4)
    self.assertIsNot(o1, o3)
    self.assertEqual(calls, 2)

  def test_eviction(self):
    calls = 0

    def f(x):
      nonlocal calls
      calls += 1
      obj = Object()
      obj.id = calls
      return obj

    cache = weak_value_interner(f)
    x = 1

    o1 = cache(x)
    self.assertEqual(o1.id, 1)
    self.assertEqual(calls, 1)

    del o1
    gc.collect()

    o2 = cache(x)
    self.assertEqual(o2.id, 2)
    self.assertEqual(calls, 2)

  def test_multithreaded_stress(self):
    class SleepyKey:

      def __init__(self, val):
        self.val = val

      def __hash__(self):
        return hash(self.val)

      def __eq__(self, other):
        time.sleep(0.001)  # Encourage a GIL release.
        return self.val == other.val

    def f(x):
      return Object()

    cache = weak_value_interner(f)

    num_threads = 5
    actions_per_thread = 1000

    num_objects = 50
    results = [None] * num_objects
    results_lock = threading.Lock()

    def worker():
      for _ in range(actions_per_thread):
        r = random.random()
        if r < 0.7:
          # Lookup
          obj = SleepyKey(random.randint(0, num_objects - 1))
          res = cache(obj)
          with results_lock:
            results[obj.val] = res
        elif r < 0.9:
          # Drop result
          obj_val = random.randint(0, num_objects - 1)
          with results_lock:
            results[obj_val] = None
        elif r < 0.95:
          # GC
          gc.collect()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

  def test_reentrant(self):
    calls = 0

    def f(x):
      nonlocal calls
      calls += 1
      obj = Object()
      obj.id = x
      if x > 0:
        obj.prev = cache(x - 1)
      return obj

    cache = weak_value_interner(f)

    o = cache(3)
    self.assertEqual(o.id, 3)
    self.assertEqual(calls, 4)

    self.assertEqual(cache(2).id, 2)
    self.assertEqual(cache(1).id, 1)
    self.assertEqual(cache(0).id, 0)


if __name__ == '__main__':
  absltest.main()
