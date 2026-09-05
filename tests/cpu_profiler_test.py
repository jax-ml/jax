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

import typing
from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import _profiler
from jax.experimental import profiler
import jax.numpy as jnp

config.parse_flags_with_absl()


@jtu.thread_unsafe_test_class()
class CpuProfilerTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(jtu.global_config_context(jax_platform_name="cpu"))
    self.enter_context(
        jtu.ignore_warning(
            category=DeprecationWarning,
            message="builtin type event_stats has no __module__ attribute",
        )
    )

  def assertValidThunkTimings(self, timings: typing.Any) -> None:
    self.assertIsInstance(timings, list)
    self.assertNotEmpty(timings)
    for i, item in enumerate(timings):
      self.assertIsInstance(item, tuple, msg=f"Item {i} is not a tuple")
      self.assertLen(item, 2, msg=f"Item {i} does not have length 2")
      name, duration_ms = item
      self.assertIsInstance(name, str, msg=f"Name in item {i} is not a string")
      self.assertIsInstance(
          duration_ms, float, msg=f"Duration in item {i} is not a float"
      )
      self.assertGreater(
          duration_ms, 0.0, msg=f"Duration in item {i} is not greater than 0.0"
      )

  def assertValidAggregatedTimings(self, timings: typing.Any) -> None:
    self.assertIsInstance(timings, list)
    for i, t in enumerate(timings):
      self.assertIsInstance(t, float, msg=f"Item {i} is not a float")
      self.assertGreaterEqual(t, 0.0, msg=f"Item {i} is negative")

  def test_measure_basic(self):
    # Must JIT to ensure XLA CPU thunks are compiled and executed.
    f = jax.jit(lambda x: x + 1.0)
    x = jnp.array([1.0, 2.0, 3.0])

    res, duration_ms = profiler.measure(f)(x)
    self.assertIsInstance(duration_ms, float)
    self.assertGreater(duration_ms, 0.0)

  def test_measure_per_thunk(self):
    f = jax.jit(lambda x: jnp.sin(x + 1.0))
    x = jnp.array([1.0, 2.0, 3.0])

    res, timings = profiler.measure(f, aggregate=False)(x)
    self.assertValidThunkTimings(timings)
    self.assertEqual(res.shape, x.shape)

  def test_measure_multiple_iterations_aggregated(self):
    f = jax.jit(lambda x: x * 2.0)
    x = jnp.array([1.0, 2.0, 3.0])

    _, timings_agg = profiler.measure(f, aggregate=True, iterations=3)(x)
    self.assertLen(timings_agg, 3)
    self.assertValidAggregatedTimings(timings_agg)

  def test_measure_multiple_iterations_per_thunk(self):
    f = jax.jit(lambda x: x * 2.0)
    x = jnp.array([1.0, 2.0, 3.0])

    # Aggregate=False, iterations=2 -> should return list of 2 lists of tuples.
    _, timings_per = profiler.measure(f, aggregate=False, iterations=2)(x)
    self.assertLen(timings_per, 2)
    self.assertNotEmpty(timings_per[0])

  def test_measure_collision(self):
    f = jax.jit(lambda x: x)
    x = jnp.array([1.0])

    # Create an active session manually.
    options = _profiler.ProfileOptions()
    active_session = _profiler.ProfilerSession(options)
    self.addCleanup(active_session.stop)

    # Measure should fail because another session is active.
    with self.assertRaisesRegex(
        RuntimeError, "another profiler session is already active"
    ):
      profiler.measure(f)(x)

  def test_measure_empty_trace_warning(self):
    # Run pure Python without JAX: won't produce XLA thunks, leading to empty trace.
    f = lambda x: [v + 1.0 for v in x]
    x = [1.0, 2.0, 3.0]

    with self.assertWarnsRegex(
        RuntimeWarning, "No CPU thunk events were captured"
    ):
      res, timings = profiler.measure(f)(x)
    self.assertIsNone(timings)
    self.assertEqual(res, [2.0, 3.0, 4.0])

  def test_measure_warmup_exclusion(self):
    # Use a function that takes some time to compile (e.g., nested maps)
    # so that compilation overhead is obvious if included.
    f = jax.jit(lambda x: jnp.sin(jnp.cos(x) + jnp.exp(x)))
    x = jnp.ones((100, 100))

    _, duration_ms = profiler.measure(f)(x)
    self.assertLess(
        duration_ms,
        20.0,
        msg=(
            "The measurement should be small (<20ms) and should NOT include the"
            " compilation time."
        ),
    )

  def test_measure_no_warmup(self):
    # Use a function that takes some time to compile.
    def complex_but_fast(x):
      for _ in range(200):  # Unroll many ops to slow down compilation.
        x = x + 1.0
      return x

    f = jax.jit(complex_but_fast)
    x = jnp.array([1.0])

    # First measurement: compilation is included in the profiler session, but the
    # returned timings only measure thunk execution. `duration_first` is expected
    # to be larger than `duration_second` due to first-run/cold-start overhead
    # (e.g., initialization of buffers/executables) rather than the compilation
    # time itself (which happens on host and is not captured as thunk execution).
    _, duration_first = profiler.measure(f, warmup=False)(x)

    # Second measurement: warm execution where cold-start overhead is avoided.
    _, duration_second = profiler.measure(f, warmup=False)(x)

    msg = (
        "duration_first is expected to be larger than duration_second due to"
        " first-run/cold-start overhead (e.g., initialization of"
        " buffers/executables) not captured in subsequent runs."
    )
    self.assertGreater(duration_first, duration_second * 2.0, msg=msg)

  def test_measure_invalid_iterations(self):
    f = jax.jit(lambda x: x)
    with self.assertRaisesRegex(ValueError, "must be positive"):
      profiler.measure(f, iterations=0)

  def test_measure_failure_cleanup(self):
    calls = 0

    def failing_f(x):
      nonlocal calls
      calls += 1
      if calls > 1:
        raise ValueError("Intentional failure")
      return x + 1.0

    x = jnp.array([1.0, 2.0, 3.0])

    with self.assertRaisesRegex(ValueError, "Intentional failure"):
      profiler.measure(failing_f, iterations=1)(x)

    # Subsequent call should succeed (not block on active session).
    f_ok = jax.jit(lambda x: x + 1.0)
    res, duration_ms = profiler.measure(f_ok)(x)
    self.assertIsInstance(duration_ms, float)
    self.assertGreater(duration_ms, 0.0)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
