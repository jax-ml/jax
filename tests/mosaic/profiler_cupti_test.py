# Copyright 2024 The JAX Authors. All Rights Reserved.
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

import pathlib
import tempfile

from absl.testing import absltest, parameterized
import jax
import jax.profiler
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
try:
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib
  HAS_MOSAIC_GPU = True
except ImportError:
  HAS_MOSAIC_GPU = False
  mosaic_gpu_lib = None
else:
  from jax.experimental.mosaic.gpu import profiler

# ruff: noqa: F405
config.parse_flags_with_absl()


def _assert_trace_has_device_events(testcase, trace_dir):
  profile_paths = list(pathlib.Path(trace_dir).glob("**/*.xplane.pb"))
  testcase.assertTrue(profile_paths, "No XPlane profile files written")
  event_count = 0
  for profile_path in profile_paths:
    profile = jax.profiler.ProfileData.from_serialized_xspace(
        profile_path.read_bytes())
    event_count += sum(
        len(list(line.events))
        for plane in profile.planes
        if plane.name.startswith("/device:")
        for line in plane.lines)
  testcase.assertGreater(event_count, 0, "No device events found in XPlane")


def _cupti_v2_available():
  return bool(
      mosaic_gpu_lib is not None and
      mosaic_gpu_lib._mosaic_gpu_ext._cupti_v2_available())


@jtu.thread_unsafe_test_class()
class ProfilerCuptiTest(parameterized.TestCase):

  def setUp(self):
    if jtu.test_device_matches(["rocm"]):
      self.skipTest("Mosaic GPU is not supported on ROCm.")
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    if (not jtu.test_device_matches(["cuda"])):
      self.skipTest("Only works on NVIDIA GPUs")
    super().setUp()
    self.x = jnp.arange(1024 * 1024)
    self.f = lambda x: 2*x

  def test_measure_cupti_explicit(self):
    _, runtime_ms = profiler.measure(self.f)(self.x)
    self.assertIsInstance(runtime_ms, float)

  def test_measure_per_kernel(self):
    _, runtimes_ms = profiler.measure(self.f, aggregate=False)(self.x)
    for item in runtimes_ms:
      self.assertIsInstance(item, tuple)
      self.assertEqual(len(item), 2)
      name, runtime_ms = item
      self.assertIsInstance(name, str)
      self.assertIsInstance(runtime_ms, float)

  def test_measure_cupti_repeated(self):
    f_profiled = profiler.measure(self.f)
    n = 3
    timings = [f_profiled(self.x)[1] for _ in range(n)]
    for item in timings:
      self.assertIsInstance(item, float)

  def test_tokamax_cupti_xprof_ordering(self):
    """Regression test for TokaMax's CUPTI/XProf ordering.

    This checks that the XLA-backed JAX
    profiler and Mosaic can both use CUPTI V2 in the TokaMax ordering.
    """
    if not _cupti_v2_available():
      self.skipTest("CUPTI V2 multi-subscriber APIs are unavailable")

    f = jax.jit(lambda x: jnp.sin(x) ** 2 + 10.0)
    x = jnp.ones((512, 512), dtype=jnp.float32)
    timer = profiler.Cupti(finalize=False).measure(f)

    def run_mosaic():
      result, runtime_ms = timer(x)
      jax.block_until_ready(result)
      self.assertIsInstance(runtime_ms, float)
      self.assertGreater(runtime_ms, 0.0)

    run_mosaic()
    run_mosaic()
    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        jax.block_until_ready(f(x))
      _assert_trace_has_device_events(self, trace_dir)
    run_mosaic()
    run_mosaic()

  def test_default_mosaic_cupti_then_jax_trace_uses_v2(self):
    if not _cupti_v2_available():
      self.skipTest("CUPTI V2 multi-subscriber APIs are unavailable")

    f = jax.jit(lambda x: jnp.sin(x) ** 2 + 10.0)
    x = jnp.ones((512, 512), dtype=jnp.float32)
    result, runtime_ms = profiler.Cupti(finalize=True).measure(f)(x)
    jax.block_until_ready(result)
    self.assertIsInstance(runtime_ms, float)
    self.assertGreater(runtime_ms, 0.0)

    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        jax.block_until_ready(f(x))
      _assert_trace_has_device_events(self, trace_dir)

  def test_mosaic_cupti_inside_jax_trace_uses_v2(self):
    if not _cupti_v2_available():
      self.skipTest("CUPTI V2 multi-subscriber APIs are unavailable")

    f = jax.jit(lambda x: x @ x)
    x = jnp.ones((512, 512), dtype=jnp.float32)
    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        result, runtime_ms = profiler.Cupti(finalize=False).measure(f)(x)
        jax.block_until_ready(result)
      _assert_trace_has_device_events(self, trace_dir)
    self.assertIsInstance(runtime_ms, float)
    self.assertGreater(runtime_ms, 0.0)

  def test_measure_repeated_interleaved(self):
    # test that kernels run outside of measure() are not captured
    _, timings = profiler.measure(self.f, aggregate=False)(self.x)
    self.assertEqual(len(timings), 1)
    self.f(self.x)
    _, timings = profiler.measure(self.f, aggregate=False)(self.x)
    self.assertEqual(len(timings), 1)

  def test_iterations(self):
    _, timings = profiler.measure(
        self.f, aggregate=False, iterations=10
    )(self.x)
    self.assertEqual(len(timings), 10)
    self.assertTrue(
        all(
            isinstance(n, str) and isinstance(t, float)
            for iter_timings in timings
            for n, t in iter_timings
        )
    )
    _, timings = profiler.measure(
        self.f, aggregate=True, iterations=5
    )(self.x)
    self.assertEqual(len(timings), 5)
    self.assertTrue(all(isinstance(t, float) for t in timings))

  def test_measure_double_subscription(self):
    # This needs to run in a separate process, otherwise it affects the
    # outcomes of other tests since CUPTI state is global.
    self.skipTest("Must run in a separate process from other profiler tests")
    # Initialize profiler manually, which subscribes to CUPTI. There can only
    # be one CUPTI subscriber at a time.
    jax._src.lib.mosaic_gpu._mosaic_gpu_ext._cupti_init()
    with self.assertRaisesRegex(RuntimeError,
      "Attempted to subscribe to CUPTI while another subscriber, "
      "such as Nsight Systems or Nsight Compute, is active."):
      profiler.measure(self.f, aggregate=False)(self.x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
