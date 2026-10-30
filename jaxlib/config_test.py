# Copyright 2024 The JAX Authors
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
import threading
import weakref

from absl.testing import absltest

from jax.jaxlib import _jax
from jax.jaxlib import xla_client

config = _jax.config


test_race_config = config.Config("test_race", 1)
test_tc_config = config.Config(
    "test_tc", "global_default", include_in_trace_context=True
)
test_non_tc_config = config.Config(
    "test_non_tc", "non_tc_default", include_in_trace_context=False
)
test_tc_race_config = config.Config(
    "test_tc_race", 1, include_in_trace_context=True
)


class ConfigTest(absltest.TestCase):

  def testBasic(self):
    c = config.Config("test", 1)
    self.assertEqual(c.name, "test")
    self.assertEqual(c.value, 1)
    self.assertEqual(c.get_global(), 1)
    self.assertEqual(c.get_local(), config.unset)

    c.set_global(2)
    self.assertEqual(c.value, 2)
    self.assertEqual(c.get_global(), 2)
    self.assertEqual(c.get_local(), config.unset)

    c.set_local(3)
    self.assertEqual(c.value, 3)
    self.assertEqual(c.get_global(), 2)
    self.assertEqual(c.get_local(), 3)

    c.set_global(4)
    self.assertEqual(c.value, 3)
    self.assertEqual(c.get_global(), 4)
    self.assertEqual(c.get_local(), 3)

    c.set_local(config.unset)
    self.assertEqual(c.value, 4)
    self.assertEqual(c.get_global(), 4)
    self.assertEqual(c.get_local(), config.unset)

  def testThreading(self):
    c = config.Config("test", 1)

    def Body():
      for i in range(100):
        c.set_local(i)
        self.assertEqual(c.get_local(), i)
        self.assertEqual(c.get_global(), 1)
        self.assertEqual(c.value, i)

    threads = [threading.Thread(target=Body) for _ in range(4)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

  def testConcurrentGlobalStateRace(self):
    def writer():
      for _ in range(2000):
        test_race_config.set_global(2)
        test_race_config.set_global(1)

    def reader():
      for _ in range(2000):
        _ = test_race_config.value

    def collector():
      for _ in range(50):
        gc.collect()

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
        threading.Thread(target=collector),
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

  def testUninitializedClear(self):
    if xla_client._version < 517:
      self.skipTest("Requires jaxlib_extension_version >= 517")
    obj = config.Config.__new__(config.Config)
    cycle = [obj]
    cycle.append(cycle)
    del obj, cycle
    gc.collect()
    self.assertEqual(test_race_config.value, 1)

  def testTraceContextCaching(self):
    tc_idx = config.trace_context_names().index("test_tc")
    try:
      t1 = config.trace_context()
      t2 = config.trace_context()
      self.assertIs(t1, t2)
      self.assertEqual(t1[tc_idx], "global_default")

      # Updating a global trace-context config invalidates the cached tuple.
      test_tc_config.set_global("global_updated")
      t3 = config.trace_context()
      self.assertIsNot(t3, t1)
      self.assertEqual(t3[tc_idx], "global_updated")
      self.assertIs(config.trace_context(), t3)

      # Thread-local overrides on non-trace-context configs do not bypass cache.
      test_non_tc_config.set_local("local_non_tc")
      try:
        self.assertIs(config.trace_context(), t3)
      finally:
        test_non_tc_config.set_local(config.unset)

      # Nested thread-local overrides on trace-context configs bypass the global
      # cache on this thread and restore cleanly when unset.
      prev = test_tc_config.swap_local("local_1")
      self.assertIs(prev, config.unset)
      try:
        self.assertEqual(config.trace_context()[tc_idx], "local_1")
        test_tc_config.set_local(None)
        self.assertIsNone(config.trace_context()[tc_idx])
      finally:
        test_tc_config.set_local(config.unset)
      self.assertIs(config.trace_context(), t3)

      # Registering a new trace-context config invalidates the cached tuple.
      new_tc = config.Config(
          "test_tc_dynamic", "dyn_val", include_in_trace_context=True
      )
      t4 = config.trace_context()
      self.assertIsNot(t4, t3)
      self.assertEqual(t4[-1], "dyn_val")
      self.assertIs(config.trace_context(), t4)
      del new_tc
    finally:
      test_tc_config.set_local(config.unset)
      test_tc_config.set_global("global_default")

  def testTraceContextPerThreadIsolation(self):
    tc_idx = config.trace_context_names().index("test_tc")
    barrier = threading.Barrier(2)
    errors = []

    def thread_with_override():
      test_tc_config.set_local("thread_local_override")
      try:
        barrier.wait()
        if config.trace_context()[tc_idx] != "thread_local_override":
          errors.append("override thread saw wrong value")
        barrier.wait()
      finally:
        test_tc_config.set_local(config.unset)

    def thread_without_override():
      barrier.wait()
      tb1 = config.trace_context()
      tb2 = config.trace_context()
      if tb1[tc_idx] != "global_default":
        errors.append("global thread saw local override")
      if tb1 is not tb2:
        errors.append("global thread bypassed cached tuple")
      barrier.wait()

    t_local = threading.Thread(target=thread_with_override)
    t_global = threading.Thread(target=thread_without_override)
    t_local.start()
    t_global.start()
    t_local.join()
    t_global.join()
    self.assertEmpty(errors)

  def testTraceContextGarbageCollection(self):
    class Node:
      pass

    # A cyclic non-trace-context Config is collected even while trace_context()
    # is cached.
    config.trace_context()
    node = Node()
    cyclic_cfg = config.Config("test_cyclic_non_tc", node)
    node.cfg = cyclic_cfg
    ref = weakref.ref(node)
    del node, cyclic_cfg
    gc.collect()
    self.assertIsNone(ref())

    # A cyclic trace-context Config is collected via tp_traverse/tp_clear once
    # the cached tuple is invalidated, and subsequent trace_context() calls
    # (both global and thread-local) remain valid after tp_clear.
    node_tc = Node()
    cyclic_tc = config.Config(
        "test_cyclic_tc", node_tc, include_in_trace_context=True
    )
    node_tc.cfg = cyclic_tc
    config.trace_context()
    test_tc_config.set_global("global_default")
    ref_tc = weakref.ref(node_tc)
    del node_tc, cyclic_tc
    gc.collect()
    self.assertIsNone(ref_tc())
    self.assertIsInstance(config.trace_context(), tuple)
    test_tc_config.set_local("local_after_gc")
    try:
      self.assertIsInstance(config.trace_context(), tuple)
    finally:
      test_tc_config.set_local(config.unset)

  def testConcurrentTraceContextRace(self):
    tc_idx = config.trace_context_names().index("test_tc_race")
    errors = []

    def global_writer():
      for _ in range(1000):
        test_tc_race_config.set_global(2)
        test_tc_race_config.set_global(1)

    def local_override_worker():
      for _ in range(1000):
        test_tc_race_config.set_local(3)
        if config.trace_context()[tc_idx] != 3:
          errors.append("expected local override 3")
          break
        test_tc_race_config.set_local(config.unset)

    def global_reader():
      for _ in range(2000):
        val = config.trace_context()[tc_idx]
        if val not in (1, 2):
          errors.append(f"unexpected global value {val}")
          break

    def collector():
      for _ in range(50):
        gc.collect()

    try:
      threads = [
          threading.Thread(target=global_writer),
          threading.Thread(target=local_override_worker),
          threading.Thread(target=global_reader),
          threading.Thread(target=global_reader),
          threading.Thread(target=collector),
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()
    finally:
      test_tc_race_config.set_local(config.unset)
      test_tc_race_config.set_global(1)
    self.assertEmpty(errors)


if __name__ == "__main__":
  absltest.main()
