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

import os
import tempfile

from absl.testing import absltest, parameterized
import jax
import jax.profiler
from jax._src import config
from jax._src import profiler as jax_profiler_src
from jax._src import test_util as jtu
from jax._src.lib import cuda_versions
import jax.numpy as jnp
try:
  import jax._src.lib.mosaic_gpu  # noqa: F401
  HAS_MOSAIC_GPU = True
except ImportError:
  HAS_MOSAIC_GPU = False
else:
  from jax.experimental.compilation_cache import compilation_cache as cc
  from jax.experimental.mosaic.gpu import profiler

# ruff: noqa: F405
config.parse_flags_with_absl()

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


class _CuptiV2TestBase(jtu.JaxTestCase):
  """Base for V2 multi-subscriber tests."""

  def setUp(self):
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Only runs on NVIDIA GPUs")
    super().setUp()
    self.x = jnp.ones((512, 512), dtype=jnp.float32)

  def _skip_unless_cupti_v2(self):
    if cuda_versions is None:
      self.skipTest("cuda_versions is unavailable")
    cupti_version = cuda_versions.cupti_get_version()
    if cupti_version < 130200:
      self.skipTest(
          f"Requires CUPTI >= 13.2 for multi-subscriber V2 APIs; "
          f"found {cupti_version}")

  def _run_mosaic_profile(self):
    result, runtime_ms = profiler.measure(lambda x: x @ x)(self.x)
    jax.block_until_ready(result)
    self.assertIsInstance(runtime_ms, float)
    self.assertGreater(runtime_ms, 0.0)
    return runtime_ms

  def _run_jax_trace(self):
    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        jax.block_until_ready(jax.jit(lambda x: x @ x)(self.x))
      trace_files = list(os.scandir(trace_dir))
      self.assertTrue(trace_files, "No trace files written")

  def _run_pgle_trace(self):
    runner = jax_profiler_src.PGLEProfiler(retries=1, percentile=90)
    with jax_profiler_src.PGLEProfiler.trace(runner):
      jax.block_until_ready(jax.jit(lambda x: x @ x)(self.x))
    self.assertEqual(runner.called_times, 1)


class ProfilerCuptiMultiSubscriberTest(_CuptiV2TestBase):
  """jax.profiler + mosaic_gpu.profiler coexistence via CUPTI V2."""

  def setUp(self):
    super().setUp()
    self._skip_unless_cupti_v2()

  def test_mosaic_profiler_inside_jax_trace(self):
    """mosaic profiler inside a jax.profiler trace (both V2 subscribers)."""
    with tempfile.TemporaryDirectory() as trace_dir:
      jax.profiler.start_trace(trace_dir)
      try:
        runtime_ms = self._run_mosaic_profile()
      finally:
        jax.profiler.stop_trace()
    self.assertIsInstance(runtime_ms, float)
    self.assertGreater(runtime_ms, 0.0)

  def test_jax_trace_then_mosaic_profiler(self):
    """jax.profiler trace then mosaic profiler."""
    self._run_jax_trace()
    self._run_mosaic_profile()

  def test_mosaic_profiler_then_jax_trace(self):
    """mosaic profiler then jax trace."""
    self._run_mosaic_profile()
    self._run_jax_trace()

  def test_finalize_flag_ignored_on_cupti_v2(self):
    """finalize has no effect on the CUPTI V2 multi-subscriber path."""
    f = lambda x: x @ x

    result_true, runtime_ms1 = profiler.Cupti(finalize=True).measure(f)(self.x)
    jax.block_until_ready(result_true)

    result_false, runtime_ms2 = profiler.Cupti(finalize=False).measure(f)(self.x)
    jax.block_until_ready(result_false)

    self.assertIsInstance(runtime_ms1, float)
    self.assertGreater(runtime_ms1, 0.0)
    self.assertIsInstance(runtime_ms2, float)
    self.assertGreater(runtime_ms2, 0.0)

  def test_mosaic_trace_mosaic_reuse(self):
    """Mosaic profiling works before and after a jax.profiler trace session."""
    f = jax.jit(lambda x: x @ x)

    result1, runtime_ms1 = profiler.measure(f)(self.x)
    jax.block_until_ready(result1)

    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        jax.block_until_ready(f(self.x))
      trace_files = list(os.scandir(trace_dir))
      self.assertTrue(trace_files, "No trace files written")

    result2, runtime_ms2 = profiler.measure(f)(self.x)
    jax.block_until_ready(result2)

    self.assertIsInstance(runtime_ms1, float)
    self.assertGreater(runtime_ms1, 0.0)
    self.assertIsInstance(runtime_ms2, float)
    self.assertGreater(runtime_ms2, 0.0)


class ProfilerCuptiPgleTest(_CuptiV2TestBase):
  """PGLE + mosaic_gpu.profiler coexistence via CUPTI V2."""

  def setUp(self):
    super().setUp()
    self._skip_unless_cupti_v2()

  def test_mosaic_profiler_inside_pgle_trace(self):
    """mosaic profiler inside an active PGLE trace (both as V2 subscribers)."""
    runner = jax_profiler_src.PGLEProfiler(retries=1, percentile=90)
    with jax_profiler_src.PGLEProfiler.trace(runner):
      result, runtime_ms = profiler.measure(lambda x: x @ x)(self.x)
      jax.block_until_ready(result)
    self.assertIsInstance(runtime_ms, float)
    self.assertGreater(runtime_ms, 0.0)
    # PGLE session completed; collected profile data may be empty if no HLO ops
    # were captured.
    self.assertEqual(runner.called_times, 1)

  def test_pgle_trace_then_mosaic_profiler(self):
    """PGLE trace then mosaic profiler."""
    self._run_pgle_trace()
    self._run_mosaic_profile()

  def test_mosaic_profiler_then_pgle_trace(self):
    """mosaic profiler then PGLE trace."""
    self._run_mosaic_profile()
    self._run_pgle_trace()

  def test_pgle_workflow_with_mosaic_profiler(self):
    """jax_enable_pgle=True: mosaic profiler concurrent with PGLE (V2 mode)."""
    with (tempfile.TemporaryDirectory(prefix="pgle_test_") as cache_dir,
          config.enable_pgle(True),
          config.pgle_profiling_runs(2),
          config.enable_compilation_cache(True),
          config.raise_persistent_cache_errors(True),
          config.persistent_cache_min_entry_size_bytes(0),
          config.persistent_cache_min_compile_time_secs(0)):
      jax.clear_caches()
      cc.reset_cache()
      cc.set_cache_dir(cache_dir)
      self.addCleanup(cc.set_cache_dir, None)
      self.addCleanup(cc.reset_cache)
      self.addCleanup(jax.clear_caches)

      @jax.jit
      def step(x):
        return x @ x

      x = jnp.ones((256, 256), dtype=jnp.float32)

      for i in range(4):
        x = step(x)
        x.block_until_ready()
        _, ms = profiler.measure(lambda v: v @ v)(jnp.ones((128, 128)))
        self.assertIsInstance(ms, float, f"iter {i}: bad mosaic timing {ms}")
        self.assertGreater(ms, 0.0, f"iter {i}: bad mosaic timing {ms}")


class ProfilerCuptiXplaneTest(_CuptiV2TestBase):
  """XPlane (jax.profiler.trace) + mosaic profiler coexistence via CUPTI V2."""

  def setUp(self):
    super().setUp()
    self._skip_unless_cupti_v2()

  def test_xplane_trace_written_while_mosaic_active(self):
    """jax.profiler.trace writes trace files with mosaic profiler active."""
    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        jax.block_until_ready(jax.jit(lambda x: x @ x)(self.x))
        runtime_ms = self._run_mosaic_profile()
      # Check trace files were written
      trace_files = list(os.scandir(trace_dir))
      self.assertTrue(trace_files, "No trace files written by jax.profiler.trace()")
    self.assertIsInstance(runtime_ms, float)
    self.assertGreater(runtime_ms, 0.0)

  def test_xplane_trace_with_mosaic_profiler(self):
    """XPlane trace and mosaic profiler concurrently (two V2 subscribers)."""
    with tempfile.TemporaryDirectory() as trace_dir:
      with jax.profiler.trace(trace_dir):
        # mosaic profiler as second V2 subscriber
        result1, ms1 = profiler.measure(lambda x: x @ x)(self.x)
        jax.block_until_ready(result1)
        # regular jit computation captured by XPlane
        jax.block_until_ready(jax.jit(lambda x: x @ x)(self.x))
      trace_files = list(os.scandir(trace_dir))
      self.assertTrue(trace_files, "No trace files written")
    self.assertIsInstance(ms1, float)
    self.assertGreater(ms1, 0.0)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
