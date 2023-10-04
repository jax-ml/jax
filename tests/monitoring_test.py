# Copyright 2022 The JAX Authors.
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
"""Tests for jax.monitoring and jax._src.monitoring.

Verify that callbacks are registered/uregistered and invoked correctly to record
events.
"""
from absl.testing import absltest
from jax import monitoring
from jax._src import monitoring as jax_src_monitoring


class MonitoringTest(absltest.TestCase):

  def tearDown(self):
    monitoring.clear_event_listeners()
    super().tearDown()

  def test_record_event(self):
    events = []
    counters  = {}  # Map event names to frequency.
    def increment_event_counter(event):
      if event not in counters:
        counters[event] = 0
      counters[event] += 1
    # Test that we can register multiple callbacks.
    monitoring.register_event_listener(events.append)
    monitoring.register_event_listener(increment_event_counter)

    monitoring.record_event("test_unique_event")
    monitoring.record_event("test_common_event")
    monitoring.record_event("test_common_event")

    self.assertListEqual(events, ["test_unique_event",
                                  "test_common_event", "test_common_event"])
    self.assertDictEqual(counters, {"test_unique_event": 1,
                                    "test_common_event": 2})

  def test_record_event_durations(self):
    durations  = {}  # Map event names to frequency.
    def increment_event_duration(event, duration):
      if event not in durations:
        durations[event] = 0.
      durations[event] += duration
    monitoring.register_event_duration_secs_listener(increment_event_duration)

    monitoring.record_event_duration_secs("test_short_event", 1)
    monitoring.record_event_duration_secs("test_short_event", 2)
    monitoring.record_event_duration_secs("test_long_event", 10)

    self.assertDictEqual(durations, {"test_short_event": 3,
                                     "test_long_event": 10})

  def test_unregister_exist_callback_success(self):
    original_duration_listeners = jax_src_monitoring.get_event_duration_listeners()
    callback = lambda event, durations: None
    self.assertNotIn(callback, original_duration_listeners)
    monitoring.register_event_duration_secs_listener(callback)
    self.assertIn(callback, jax_src_monitoring.get_event_duration_listeners())
    # Verify that original listeners list is not modified by register function.
    self.assertNotEqual(original_duration_listeners,
                        jax_src_monitoring.get_event_duration_listeners())

    jax_src_monitoring._unregister_event_duration_listener_by_callback(callback)

    self.assertEqual(original_duration_listeners,
                     jax_src_monitoring.get_event_duration_listeners())

  def test_unregister_not_exist_callback_fail(self):
    callback = lambda event, durations: None
    self.assertNotIn(callback,
                     jax_src_monitoring.get_event_duration_listeners())

    with self.assertRaises(AssertionError):
      jax_src_monitoring._unregister_event_duration_listener_by_callback(
          callback)

  def test_unregister_callback_index_in_range_success(self):
    original_duration_listeners = jax_src_monitoring.get_event_duration_listeners()
    callback = lambda event, durations: None
    self.assertNotIn(callback, original_duration_listeners)
    monitoring.register_event_duration_secs_listener(callback)
    self.assertIn(callback, jax_src_monitoring.get_event_duration_listeners())
    # Verify that original listeners list is not modified by register function.
    self.assertNotEqual(original_duration_listeners,
                        jax_src_monitoring.get_event_duration_listeners())

    jax_src_monitoring._unregister_event_duration_listener_by_index(-1)

    self.assertEqual(original_duration_listeners,
                     jax_src_monitoring.get_event_duration_listeners())

  def test_unregister_callback_index_out_of_range_fail(self):
    size = len(jax_src_monitoring.get_event_duration_listeners())

    # Verify index >= size raises AssertionError.
    with self.assertRaises(AssertionError):
      jax_src_monitoring._unregister_event_duration_listener_by_index(size)

    # Verify index < -size raises AssertionError.
    with self.assertRaises(AssertionError):
      jax_src_monitoring._unregister_event_duration_listener_by_index(-size - 1)

  def test_get_event_duration_listeners_returns_a_copy(self):
    original_duration_listeners = jax_src_monitoring.get_event_duration_listeners()
    callback = lambda event, durations: None

    original_duration_listeners.append(callback)

    self.assertNotIn(callback, jax_src_monitoring.get_event_duration_listeners())
    self.assertNotEqual(original_duration_listeners,
                        jax_src_monitoring.get_event_duration_listeners())

  def test_unregister_exist_event_callback_success(self):
    original_event_listeners = jax_src_monitoring.get_event_listeners()
    callback = lambda event: None
    self.assertNotIn(callback, original_event_listeners)
    monitoring.register_event_listener(callback)
    self.assertIn(callback, jax_src_monitoring.get_event_listeners())
    # Verify that original listeners list is not modified by register function.
    self.assertNotEqual(original_event_listeners,
                        jax_src_monitoring.get_event_listeners())

    jax_src_monitoring._unregister_event_listener_by_callback(callback)

    self.assertEqual(original_event_listeners,
                     jax_src_monitoring.get_event_listeners())

  def test_unregister_not_exist_event_callback_fail(self):
    callback = lambda event: None
    self.assertNotIn(callback, jax_src_monitoring.get_event_listeners())

    with self.assertRaises(AssertionError):
      jax_src_monitoring._unregister_event_listener_by_callback(callback)

if __name__ == "__main__":
  absltest.main()
