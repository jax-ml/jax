# Copyright 2021 The JAX Authors.
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

from functools import partial
import math
import os
import tempfile
from collections import Counter
from unittest import mock
from unittest import SkipTest
import warnings

from absl.testing import absltest
import jax
from jax import config
from jax import jit
from jax import lax
from jax import pmap
from jax._src import compilation_cache as cc
from jax._src import compiler
from jax._src import monitoring
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.config import persistent_cache_min_compile_time_secs
from jax._src.config import raise_persistent_cache_errors
from jax._src.config import use_original_compilation_cache_key_generation
from jax._src.lib import xla_client
from jax.experimental.maps import xmap
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
import numpy as np


config.parse_flags_with_absl()
FLAGS = config.FLAGS

FAKE_COMPILE_TIME = 10
_counts = Counter()  # Map event name to count


def setUpModule():
  monitoring.register_event_listener(increment_event_count)


def tearDownModule():
  monitoring._unregister_event_listener_by_callback(increment_event_count)


def increment_event_count(event):
  _counts[event] += 1


@jtu.with_config(
    jax_raise_persistent_cache_errors=True,
    jax_persistent_cache_min_compile_time_secs=0,
)
class CompilationCacheTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    supported_platforms = ["tpu", "gpu"]

    if "--xla_cpu_use_xla_runtime=true" in os.environ.get("XLA_FLAGS", ""):
      supported_platforms.append("cpu")

    if jtu.device_under_test() not in supported_platforms:
      raise SkipTest(
          "serialize executable only works on " + ",".join(supported_platforms)
      )

    # Reset cache if already initialized by JaxTestCase
    if cc.is_initialized():
      cc.reset_cache()

  def tearDown(self):
    if cc.is_initialized():
      cc.reset_cache()
    super().tearDown()

  def test_get_no_executable(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)
      computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
      devices = np.array([[jax.local_devices()[0]]])
      compile_options = compiler.get_compile_options(
          num_replicas=1, num_partitions=1
      )
      backend = xla_bridge.get_backend()
      key = cc.get_cache_key(computation, devices, compile_options, backend)
      executable, compile_time = cc.get_executable_and_time(
          key, compile_options, backend)
      self.assertIsNone(executable)
      self.assertIsNone(compile_time)

  def test_diff_executables(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)
      computation1 = str(jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir())
      computation2 = str(jax.jit(lambda x, y: x * y).lower(2, 2).compiler_ir())
      compile_options = compiler.get_compile_options(
          num_replicas=1, num_partitions=1
      )
      backend = xla_bridge.get_backend()
      executable1 = backend.compile(computation1, compile_options)
      executable2 = backend.compile(computation2, compile_options)
      cc.put_executable_and_time(
          "key1", "computation1", executable1, backend, FAKE_COMPILE_TIME)
      cc.put_executable_and_time(
          "key2", "computation2", executable2, backend, FAKE_COMPILE_TIME)
      self.assertNotEqual(
          cc.get_executable_and_time("key1", compile_options, backend)[0],
          cc.get_executable_and_time("key2", compile_options, backend)[0]
      )

  def test_put_executable(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)
      computation = (
          jax.jit(lambda x, y: x + y)
          .lower(np.int32(1), np.int32(1))
          .compiler_ir()
      )
      devices = np.array([[jax.local_devices()[0]]])
      compile_options = compiler.get_compile_options(
          num_replicas=1, num_partitions=1
      )
      backend = xla_bridge.get_backend()
      executable = backend.compile(str(computation), compile_options)
      key = cc.get_cache_key(computation, devices, compile_options, backend)
      cc.put_executable_and_time(
          key, "alambda", executable, backend, FAKE_COMPILE_TIME)
      executable_retrieved, compile_time_retrieved = cc.get_executable_and_time(
          key, compile_options, backend)
      inputs_to_executable = (
          np.array(1, dtype=np.int32),
          np.array(2, dtype=np.int32),
      )
      expected = xla_client.execute_with_python_values(
          executable, inputs_to_executable, backend
      )
      actual = xla_client.execute_with_python_values(
          executable_retrieved, inputs_to_executable, backend
      )
      self.assertEqual(expected, actual)
      self.assertEqual(FAKE_COMPILE_TIME, compile_time_retrieved)

  def test_pmap(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)
      f = pmap(lambda x: x - lax.psum(x, "i"), axis_name="i")
      x = np.arange(jax.device_count(), dtype=np.int64)
      f(x)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 1)
      x = np.arange(jax.device_count(), dtype=np.float32)
      f(x)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 2)
      # TODO: create a test for calling pmap with the same input more than once

  def test_jit(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)
      f = jit(lambda x: x * x)
      f(1)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 1)
      f(1.0)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 2)

  @jtu.with_mesh([("x", 2)])
  def test_pjit(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)

      @partial(pjit, in_shardings=(P("x"), P("x")), out_shardings=None)
      def f(x, y):
        return x + y

      shape = (8, 8)
      x = np.arange(math.prod(shape), dtype=np.int64).reshape(shape)
      f(x, x + 1)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 1)
      x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
      f(x, x + 1)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 2)

  @jtu.with_mesh([("x", 2)])
  def test_xmap(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)

      def f(x):
        return x * 2

      devices = np.array(jax.local_devices()[:2])
      if devices.size < 2:
        raise SkipTest("Test requires 2 devices")
      x = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
      xmap(
          f, in_axes=["a", ...], out_axes=["a", ...], axis_resources={"a": "x"}
      )(x)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 1)
      x = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
      xmap(
          f, in_axes=["a", ...], out_axes=["a", ...], axis_resources={"a": "x"}
      )(x)
      files_in_directory = len(os.listdir(tmpdir))
      self.assertEqual(files_in_directory, 2)

  def test_cache_write_warning(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)
      f = jit(lambda x: x * x)

      with raise_persistent_cache_errors(False), mock.patch.object(
          cc._cache.__class__, "put"
      ) as mock_put, warnings.catch_warnings(record=True) as w:
        mock_put.side_effect = RuntimeError("test error")
        self.assertEqual(f(2), 4)
        self.assertLen(w, 1)
        self.assertIn(
            (
                "Error writing persistent compilation cache entry "
                "for 'jit__lambda_': RuntimeError: test error"
            ),
            str(w[0].message),
        )

  def test_cache_read_warning(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)
      f = jit(lambda x: x * x)

      with raise_persistent_cache_errors(False), mock.patch.object(
          cc._cache.__class__, "get"
      ) as mock_get, warnings.catch_warnings(record=True) as w:
        mock_get.side_effect = RuntimeError("test error")
        self.assertEqual(f(2), 4)
        if len(w) > 1:
          print("Warnings:", [str(w_) for w_ in w], flush=True)
        self.assertLen(w, 1)
        self.assertIn(
            (
                "Error reading persistent compilation cache entry "
                "for 'jit__lambda_': RuntimeError: test error"
            ),
            str(w[0].message),
        )

  def test_min_compile_time(self):
    with tempfile.TemporaryDirectory() as tmpdir, persistent_cache_min_compile_time_secs(
        2
    ):
      cc.initialize_cache(tmpdir)

      # Mock time to progress in small intervals so compilation time is small.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 10, 0.1)):
        jit(lambda x: x + 1)(1)
        files_in_cache = len(os.listdir(tmpdir))
        self.assertEqual(files_in_cache, 0)

      # Mock time to progress in large intervals so compilation time is large.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
        jit(lambda x: x + 2)(1)
        files_in_cache = len(os.listdir(tmpdir))
        self.assertEqual(files_in_cache, 1)

  def test_cache_saving_metric(self):
    with tempfile.TemporaryDirectory() as tmpdir, persistent_cache_min_compile_time_secs(
        2):
      cc.initialize_cache(tmpdir)

      durations = Counter()  # Map metric name to time duration.
      def append_metric_duration(metric, duration):
        durations[metric] += duration

      with jtu.register_event_duration_listener(append_metric_duration):

        # Mock time to create a short compilation time, no cache saved, no cache
        # hit, no metric recorded.
        with mock.patch("time.monotonic", side_effect=np.arange(0, 1, 0.1)):
          jit(lambda x: x + 1)(1)

        jit(lambda x: x + 1)(1)
        self.assertNotIn(
            "/jax/compilation_cache/cache_retrieval_time_sec", durations)
        self.assertNotIn(
            "/jax/compilation_cache/original_compile_time_saved_sec", durations)

        # Mock time to create a long compilation time, metrics incremented with
        # a cache hit.
        with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
          jit(lambda x: x + 2)(1)

        jit(lambda x: x + 2)(1)
        self.assertGreater(
            durations["/jax/compilation_cache/cache_retrieval_time_sec"], 0)
        self.assertGreater(
            durations["/jax/compilation_cache/original_compile_time_saved_sec"],
            0)

  def test_task_using_original_cache_metric(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)

      jit(lambda x: x + 1)(1)
      self.assertEqual(
          _counts['/jax/compilation_cache/tasks_using_cache'], 1)

      # Verify that the count is incremented only once per task.
      cc.reset_cache()
      cc.initialize_cache(tmpdir)
      jit(lambda x: x + 3)(3)
      self.assertEqual(
          _counts['/jax/compilation_cache/tasks_using_cache'], 1)

  def test_compile_requests_use_cache_metric(self):
    previous_counts = Counter(_counts)
    with tempfile.TemporaryDirectory() as tmpdir:
      cc.initialize_cache(tmpdir)

      jit(lambda x: x + 1)(1)
      jit(lambda x: x + 2)(1)
      jit(lambda x: x + 1)(1)

    self.assertEqual(
        _counts["/jax/compilation_cache/compile_requests_use_cache"]
        - previous_counts["/jax/compilation_cache/compile_requests_use_cache"],
        3)

  def test_cache_misses_metric(self):
    previous_counts = Counter(_counts)
    with tempfile.TemporaryDirectory() as tmpdir, persistent_cache_min_compile_time_secs(
        2):
      cc.initialize_cache(tmpdir)

      # Mock time to create a long compilation time and make cache misses.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
        jit(lambda x: x + 1)(1)
        jit(lambda x: x + 2)(1)

    self.assertEqual(
        _counts["/jax/compilation_cache/cache_misses"]
        - previous_counts["/jax/compilation_cache/cache_misses"],
        2)

  def test_cache_hits_original_metric(self):
    previous_counts = Counter(_counts)
    with tempfile.TemporaryDirectory() as tmpdir, persistent_cache_min_compile_time_secs(
        2), use_original_compilation_cache_key_generation(True):
      cc.initialize_cache(tmpdir)

      # Mock time to create a long compilation time, cache saved.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
        jit(lambda x: x + 1)(1)
      jit(lambda x: x + 1)(1)

    self.assertEqual(
        _counts["/jax/compilation_cache/cache_hits_original"]
        - previous_counts["/jax/compilation_cache/cache_hits_original"],
        1)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
