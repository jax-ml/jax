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
"""Tests for jax.monitoring.

Verify that callbacks are registered and invoked correctly to record events.
"""
from absl.testing import absltest
from jax import monitoring

class MonitoringTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
