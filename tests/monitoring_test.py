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
import jax
from jax import monitoring
from jax._src import dispatch
from jax._src import monitoring as jax_src_monitoring
from jax._src import test_util as jtu


class MonitoringTest(absltest.TestCase):

  def tearDown(self):
    monitoring.clear_event_listeners()
    super().tearDown()

  def test_record_event(self):
    events = []
    counters = {}  # Map event names to frequency.
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
    durations = {}  # Map event names to frequency.
    def increment_event_duration(event, duration, **kwargs):
      del kwargs
      if event not in durations:
        durations[event] = 0.
      durations[event] += duration
    monitoring.register_event_duration_secs_listener(increment_event_duration)

    monitoring.record_event_duration_secs("test_short_event", 1)
    monitoring.record_event_duration_secs("test_short_event", 2)
    monitoring.record_event_duration_secs("test_long_event", 10)

    self.assertDictEqual(durations, {"test_short_event": 3,
                                     "test_long_event": 10})

  def test_record_scalar(self):
    observed_keys = []
    observed_values = []

    monitoring.register_scalar_listener(
        lambda key, _, **kwargs: observed_keys.append(key),
    )
    monitoring.register_scalar_listener(
        lambda _, value, **kwargs: observed_values.append(value),
    )

    monitoring.record_scalar("test_unique_event", 1)
    monitoring.record_scalar("test_common_event", 2.5)
    monitoring.record_scalar("test_common_event", 5e5)

    self.assertListEqual(
        observed_keys,
        ["test_unique_event", "test_common_event", "test_common_event"],
    )
    self.assertListEqual(
        observed_values,
        [1, 2.5, 5e5],
    )

  def test_unregister_exist_callback_success(self):
    original_duration_listeners = jax_src_monitoring.get_event_duration_listeners()
    callback = lambda event, durations, **kwargs: None
    self.assertNotIn(callback, original_duration_listeners)
    monitoring.register_event_duration_secs_listener(callback)
    self.assertIn(callback, jax_src_monitoring.get_event_duration_listeners())
    # Verify that original listeners list is not modified by register function.
    self.assertNotEqual(original_duration_listeners,
                        jax_src_monitoring.get_event_duration_listeners())

    jax_src_monitoring.unregister_event_duration_listener(callback)

    self.assertEqual(original_duration_listeners,
                     jax_src_monitoring.get_event_duration_listeners())

  def test_unregister_not_exist_callback_fail(self):
    callback = lambda event, durations, **kwargs: None
    self.assertNotIn(callback,
                     jax_src_monitoring.get_event_duration_listeners())

    with self.assertRaises(AssertionError):
      jax_src_monitoring.unregister_event_duration_listener(callback)

  def test_get_event_duration_listeners_returns_a_copy(self):
    original_duration_listeners = jax_src_monitoring.get_event_duration_listeners()
    callback = lambda event, durations, **kwargs: None

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

    jax_src_monitoring.unregister_event_listener(callback)

    self.assertEqual(original_event_listeners,
                     jax_src_monitoring.get_event_listeners())

  def test_unregister_not_exist_event_callback_fail(self):
    callback = lambda event: None
    self.assertNotIn(callback, jax_src_monitoring.get_event_listeners())

    with self.assertRaises(AssertionError):
      jax_src_monitoring.unregister_event_listener(callback)

  def test_jit_compilation_time_span_events_share_compilation_uuid(self):
    spans = []

    def listener(event, start_time, end_time, **kwargs):
      del start_time, end_time
      spans.append((event, kwargs))

    monitoring.register_event_time_span_listener(listener)

    def make_fn(scale):
      def same_name(x):
        return x * scale

      return jax.jit(same_name)

    fn1 = make_fn(2)
    fn2 = make_fn(3)
    fn1(1)
    fn2(1)

    compile_events = {
        dispatch.JAXPR_TRACE_EVENT,
        dispatch.JAXPR_TO_MLIR_MODULE_EVENT,
        dispatch.BACKEND_COMPILE_EVENT,
    }
    recorded = [(ev, kw) for ev, kw in spans if ev in compile_events]
    self.assertLen(recorded, 6)
    for _, kw in recorded:
      self.assertIsInstance(kw.get("compilation_uuid"), str)
      self.assertNotEmpty(kw["compilation_uuid"])
    fn1_uuids = {kw["compilation_uuid"] for _, kw in recorded[:3]}
    fn2_uuids = {kw["compilation_uuid"] for _, kw in recorded[3:]}
    self.assertLen(fn1_uuids, 1)
    self.assertLen(fn2_uuids, 1)
    self.assertNotEqual(fn1_uuids, fn2_uuids)

  def test_aot_staged_and_transformed_compilations_share_compilation_uuid(self):
    spans = []

    def listener(event, start_time, end_time, **kwargs):
      del start_time, end_time
      spans.append((event, kwargs))

    monitoring.register_event_time_span_listener(listener)
    compile_events = {
        dispatch.JAXPR_TRACE_EVENT,
        dispatch.JAXPR_TO_MLIR_MODULE_EVENT,
        dispatch.BACKEND_COMPILE_EVENT,
    }

    def take_compile_spans():
      res = [(ev, kw) for ev, kw in spans if ev in compile_events]
      spans.clear()
      for _, kw in res:
        self.assertIsInstance(kw.get("compilation_uuid"), str)
        self.assertNotEmpty(kw["compilation_uuid"])
      return res

    # 1. 2-step staged AOT: jit(f).lower(x) -> lowered.compile()
    lowered = jax.jit(lambda x: x + 1).lower(1)
    lowered.compile()
    rec_2step = take_compile_spans()
    self.assertLen(rec_2step, 3)
    uuids_2step = {kw["compilation_uuid"] for _, kw in rec_2step}
    self.assertLen(uuids_2step, 1)

    # 2. 3-step staged AOT: jit(f).trace(x) -> traced.lower() -> lowered.compile()
    traced = jax.jit(lambda x: x + 2).trace(1)
    lowered_3step = traced.lower()
    lowered_3step.compile()
    rec_3step = take_compile_spans()
    self.assertLen(rec_3step, 3)
    uuids_3step = {kw["compilation_uuid"] for _, kw in rec_3step}
    self.assertLen(uuids_3step, 1)
    self.assertNotEqual(uuids_2step, uuids_3step)

    # 3. Interleaved staged AOT of two jit wrappers around the SAME function
    # (hits pe.trace_to_jaxpr cache on the second trace).
    def shared_fn(x):
      return x * 5

    t1 = jax.jit(shared_fn, donate_argnums=0).trace(1)
    t2 = jax.jit(shared_fn).trace(1)
    l1 = t1.lower()
    l2 = t2.lower()
    l1.compile()
    l2.compile()
    rec_interleaved = take_compile_spans()
    self.assertLen(rec_interleaved, 6)
    t1_uuids = {
        rec_interleaved[0][1]["compilation_uuid"],
        rec_interleaved[2][1]["compilation_uuid"],
        rec_interleaved[4][1]["compilation_uuid"],
    }
    t2_uuids = {
        rec_interleaved[1][1]["compilation_uuid"],
        rec_interleaved[3][1]["compilation_uuid"],
        rec_interleaved[5][1]["compilation_uuid"],
    }
    self.assertLen(t1_uuids, 1)
    self.assertLen(t2_uuids, 1)
    self.assertNotEqual(t1_uuids, t2_uuids)

    # 4. Traced.with_consts_as_arg() transformation preserves compilation_uuid.
    y = jax.numpy.asarray(3)
    spans.clear()
    traced_const = jax.jit(lambda x: x + y).trace(1)
    _, traced_with_consts = traced_const.with_consts_as_arg()
    traced_with_consts.lower().compile()
    rec_consts = take_compile_spans()
    self.assertGreaterEqual(len(rec_consts), 3)
    self.assertLen({kw["compilation_uuid"] for _, kw in rec_consts}, 1)

    # 5. Eager jit_p.bind via _pjit_call_impl (grad of jit).
    jax.grad(jax.jit(lambda x: x * 2.0))(1.0)
    rec_grad = take_compile_spans()
    self.assertNotEmpty(rec_grad)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
