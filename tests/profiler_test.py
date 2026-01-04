# Copyright 2020 The JAX Authors.
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

import concurrent.futures
from functools import partial
import glob
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from absl.testing import absltest
import pathlib

import jax
import jax.numpy as jnp
import jax.profiler
import jax._src.test_util as jtu

from jax._src import profiler
from jax import jit


try:
  import portpicker
except ImportError:
  portpicker = None

try:
  from xprof.convert import _pywrap_profiler_plugin
  import jax.collect_profile
except ImportError:
  _pywrap_profiler_plugin = None

jax.config.parse_flags_with_absl()


# We do not allow multiple concurrent profiler sessions.
@jtu.thread_unsafe_test_class()
class ProfilerTest(unittest.TestCase):
  # These tests simply test that the profiler API does not crash; they do not
  # check functional correctness.

  def setUp(self):
    if (
        sys.version_info < (3, 14)
        and hasattr(sys, "_is_gil_enabled")
        and not sys._is_gil_enabled()
    ):
      self.skipTest(
          "Profiler tests are not thread-safe under Python 3.13 free threading"
      )

    super().setUp()
    self.worker_start = threading.Event()
    self.profile_done = False

  @unittest.skipIf(not portpicker, "Test requires portpicker")
  def testStartStopServer(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port=port)
    del port
    jax.profiler.stop_server()

  @unittest.skipIf(not portpicker, "Test requires portpicker")
  def testCantStartMultipleServers(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port=port)
    port = portpicker.pick_unused_port()
    with self.assertRaisesRegex(
        ValueError, "Only one profiler server can be active at a time."):
      jax.profiler.start_server(port=port)
    jax.profiler.stop_server()

  def testCantStopServerBeforeStartingServer(self):
    with self.assertRaisesRegex(ValueError, "No active profiler server."):
      jax.profiler.stop_server()

  def testProgrammaticProfiling(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        jax.profiler.start_trace(tmpdir)
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count()))
      finally:
        jax.profiler.stop_trace()

      proto_path = glob.glob(os.path.join(tmpdir, "**/*.xplane.pb"),
                             recursive=True)
      self.assertEqual(len(proto_path), 1)
      with open(proto_path[0], "rb") as f:
        proto = f.read()
      # Sanity check that serialized proto contains host, device, and
      # Python traces without deserializing.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)
      self.assertIn(b"pxla.py", proto)

  def testProgrammaticProfilingConcurrency(self):
    def work():
      x = jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
          jnp.ones(jax.local_device_count()))
      jax.block_until_ready(x)
    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        jax.profiler.start_trace(tmpdir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
          for _ in range(10):
            executor.submit(work)
      finally:
        jax.profiler.stop_trace()

      proto_path = glob.glob(os.path.join(tmpdir, "**/*.xplane.pb"),
                             recursive=True)
      self.assertEqual(len(proto_path), 1)
      with open(proto_path[0], "rb") as f:
        proto = f.read()
      # Sanity check that serialized proto contains host, device, and
      # Python traces without deserializing.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)
      self.assertIn(b"pxla.py", proto)

  def testProgrammaticProfilingWithOptions(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        jax.profiler.start_trace(tmpdir, profiler_options=options)
        jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
            jnp.ones(jax.local_device_count())
        )
      finally:
        jax.profiler.stop_trace()

      proto_path = glob.glob(
          os.path.join(tmpdir, "**/*.xplane.pb"), recursive=True
      )
      self.assertEqual(len(proto_path), 1)
      with open(proto_path[0], "rb") as f:
        proto = f.read()
      # Verify that the serialized proto contains host and device traces, and
      # does not contain Python traces.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)
      self.assertNotIn(b"pxla.py", proto)

  def testProgrammaticProfilingPathlib(self):
    with tempfile.TemporaryDirectory() as tmpdir_string:
      tmpdir = pathlib.Path(tmpdir_string)
      try:
        jax.profiler.start_trace(tmpdir)
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count()))
      finally:
        jax.profiler.stop_trace()

      proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
      self.assertEqual(len(proto_path), 1)
      proto = proto_path[0].read_bytes()
      # Sanity check that serialized proto contains host, device, and
      # Python traces without deserializing.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)
      self.assertIn(b"pxla.py", proto)

  def testProgrammaticProfilingWithOptionsPathlib(self):
    with tempfile.TemporaryDirectory() as tmpdir_string:
      tmpdir = pathlib.Path(tmpdir_string)
      try:
        options = jax.profiler.ProfileOptions()
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_ONLY_HOST"}
        jax.profiler.start_trace(tmpdir, profiler_options=options)
        jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
            jnp.ones(jax.local_device_count())
        )
      finally:
        jax.profiler.stop_trace()

      proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
      self.assertEqual(len(proto_path), 1)
      proto = proto_path[0].read_bytes()
      # Verify that the serialized proto contains host traces and does not
      # contain TPU device traces.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertNotIn(b"/device:TPU", proto)
      self.assertIn(b"pxla.py", proto)

  def testProfilerGetFDOProfile(self):
    # Tests stop_and_get_fod_profile could run.
    try:
      jax.profiler.start_trace("test")
      jax.pmap(lambda x: jax.lax.psum(x + 1, "i"), axis_name="i")(
          jnp.ones(jax.local_device_count())
      )
    finally:
      fdo_profile = profiler.stop_and_get_fdo_profile()
    if jtu.test_device_matches(["gpu"]) and jtu.is_device_cuda():
      self.assertIn(b"copy", fdo_profile)

  def testProgrammaticProfilingErrors(self):
    with self.assertRaisesRegex(RuntimeError, "No profile started"):
      jax.profiler.stop_trace()

    try:
      with tempfile.TemporaryDirectory() as tmpdir:
        jax.profiler.start_trace(tmpdir)
        with self.assertRaisesRegex(
          RuntimeError,
          "Profile has already been started. Only one profile may be run at a "
          "time."):
          jax.profiler.start_trace(tmpdir)
    finally:
      jax.profiler.stop_trace()

  def testProgrammaticProfilingContextManager(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with jax.profiler.trace(tmpdir):
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count()))

      proto_path = glob.glob(os.path.join(tmpdir, "**/*.xplane.pb"),
                             recursive=True)
      self.assertEqual(len(proto_path), 1)
      with open(proto_path[0], "rb") as f:
        proto = f.read()
      # Sanity check that serialized proto contains host and device traces
      # without deserializing.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)

  @jtu.run_on_devices("gpu")
  @jtu.thread_unsafe_test()
  def testProgrammaticGpuCuptiTracing(self):
    @jit
    def xy_plus_z(x, y, z):
      return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z
    k = jax.random.key(0)
    s = 1, 16, 16
    jax.devices()
    x = jnp.int8(jax.random.normal(k, shape=s))
    y = jnp.bfloat16(jax.random.normal(k, shape=s))
    z = jnp.float32(jax.random.normal(k, shape=s))
    with tempfile.TemporaryDirectory() as tmpdir_string:
      tmpdir = pathlib.Path(tmpdir_string)
      with jax.profiler.trace(tmpdir):
        print(xy_plus_z(x, y, z))

      proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
      proto_bytes = proto_path[0].read_bytes()
      self.assertIn(b"/device:GPU", proto_bytes)

  @jtu.run_on_devices("gpu")
  @jtu.thread_unsafe_test()
  def testProgrammaticGpuCuptiTracingWithOptions(self):
    @jit
    def xy_plus_z(x, y, z):
      return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z

    k = jax.random.key(0)
    s = 1, 16, 16
    jax.devices()
    x = jnp.int8(jax.random.normal(k, shape=s))
    y = jnp.bfloat16(jax.random.normal(k, shape=s))
    z = jnp.float32(jax.random.normal(k, shape=s))
    with tempfile.TemporaryDirectory() as tmpdir_string:
      tmpdir = pathlib.Path(tmpdir_string)
      options = jax.profiler.ProfileOptions()
      options.advanced_configuration = {
          "gpu_max_callback_api_events": 1000000,
          "gpu_enable_nvtx_tracking": True,
      }
      with jax.profiler.trace(tmpdir):
        xy_plus_z(x, y, z).block_until_ready()

      proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
      proto_bytes = proto_path[0].read_bytes()
      self.assertIn(b"/device:GPU", proto_bytes)

  # TODO: b/443121646 - Enable PM sampling test on JAX OSS once the Github CI
  # host machine has privileged access.
  # @jtu.run_on_devices("gpu")
  # @jtu.thread_unsafe_test()
  # def testProgrammaticGpuCuptiTracingWithPmSampling(self):
  #   if not (jtu.is_cuda_compute_capability_equal("9.0")):
  #     self.skipTest("Only works on GPU with capability sm90")

  #   @jit
  #   def xy_plus_z(x, y, z):
  #     return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z

  #   k = jax.random.key(0)
  #   s = 1, 16, 16
  #   jax.devices()
  #   x = jnp.int8(jax.random.normal(k, shape=s))
  #   y = jnp.bfloat16(jax.random.normal(k, shape=s))
  #   z = jnp.float32(jax.random.normal(k, shape=s))
  #   with tempfile.TemporaryDirectory() as tmpdir_string:
  #     tmpdir = pathlib.Path(tmpdir_string)
  #     options = jax.profiler.ProfileOptions()
  #     options.advanced_configuration = {
  #         "gpu_pm_sample_counters": (
  #             "sm__cycles_active.sum"
  #         ),
  #         "gpu_pm_sample_interval_us": 500,
  #     }
  #     with jax.profiler.trace(tmpdir, profiler_options=options):
  #       xy_plus_z(x, y, z).block_until_ready()

  #     proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
  #     proto_bytes = proto_path[0].read_bytes()
  #     self.assertIn(b"/device:GPU", proto_bytes)
  #     self.assertIn(
  #         b"sm__cycles_active.sum", proto_bytes
  #     )

  def testProgrammaticProfilingContextManagerPathlib(self):
    with tempfile.TemporaryDirectory() as tmpdir_string:
      tmpdir = pathlib.Path(tmpdir_string)
      with jax.profiler.trace(tmpdir):
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count()))

      proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
      self.assertEqual(len(proto_path), 1)
      proto = proto_path[0].read_bytes()
      # Sanity check that serialized proto contains host and device traces
      # without deserializing.
      self.assertIn(b"/host:CPU", proto)
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"/device:TPU", proto)

  def testTraceAnnotation(self):
    x = 3
    with jax.profiler.TraceAnnotation("mycontext"):
      x = x + 2

  def testTraceFunction(self):
    @jax.profiler.annotate_function
    def f(x, *, y):
      return x + 2 * y
    self.assertEqual(f(7, y=3), 13)

    @jax.profiler.annotate_function
    def f(x, *, name):
      return x + 2 * len(name)
    self.assertEqual(f(7, name="abc"), 13)

    @partial(jax.profiler.annotate_function, name="aname")
    def g(x):
      return x + 2
    self.assertEqual(g(7), 9)

    @partial(jax.profiler.annotate_function, name="aname", akwarg="hello")
    def h(x):
      return x + 2
    self.assertEqual(h(7), 9)

  def testDeviceMemoryProfile(self):
    x = jnp.ones((20,)) + 7.
    self.assertIsInstance(jax.profiler.device_memory_profile(), bytes)
    del x

  def _check_xspace_pb_exist(self, logdir):
    path = os.path.join(logdir, 'plugins', 'profile', '*', '*.xplane.pb')
    self.assertEqual(1, len(glob.glob(path)),
                     'Expected one path match: ' + path)

  @unittest.skip("Test causes OOMs")
  @unittest.skipIf(not (portpicker and _pywrap_profiler_plugin),
    "Test requires xprof and portpicker")
  def testSingleWorkerSamplingMode(self, delay_ms=None):
    def on_worker(port, worker_start):
      jax.profiler.start_server(port)
      worker_start.set()
      x = jnp.ones((1000, 1000))
      while True:
        with jax.profiler.TraceAnnotation("atraceannotation"):
          jnp.dot(x, x.T).block_until_ready()
          if self.profile_done:
            jax.profiler.stop_server()
            break

    def on_profile(port, logdir, worker_start):
      worker_start.wait()
      options = {
          "host_tracer_level": 2,
          "python_tracer_level": 2,
          "device_tracer_level": 1,
          "delay_ms": delay_ms,
      }

      # Request for 1000 milliseconds of profile.
      duration_ms = 1000
      _pywrap_profiler_plugin.trace(
          f'localhost:{port}',
          logdir,
          '',
          True,
          duration_ms,
          3,
          options
      )
      self.profile_done = True

    logdir = absltest.get_default_test_tmpdir()
    # Remove any existing log files.
    shutil.rmtree(logdir, ignore_errors=True)
    port = portpicker.pick_unused_port()
    thread_profiler = threading.Thread(
        target=on_profile, args=(port, logdir, self.worker_start))
    thread_worker = threading.Thread(
        target=on_worker, args=(port, self.worker_start))
    thread_worker.start()
    thread_profiler.start()
    thread_profiler.join()
    thread_worker.join(120)
    self._check_xspace_pb_exist(logdir)

  @unittest.skipIf(
      not (portpicker and _pywrap_profiler_plugin),
    "Test requires xprof and portpicker")
  def test_remote_profiler(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port)

    profile_done = threading.Event()
    logdir = absltest.get_default_test_tmpdir()
    # Remove any existing log files.
    shutil.rmtree(logdir, ignore_errors=True)
    def on_profile():
      os.system(
          f"{sys.executable} -m jax.collect_profile {port} 500 "
          f"--log_dir {logdir} --no_perfetto_link")
      profile_done.set()

    thread_profiler = threading.Thread(
        target=on_profile, args=())
    thread_profiler.start()
    start_time = time.time()
    y = jnp.zeros((5, 5))
    while not profile_done.is_set():
      # The timeout here must be relatively high. The profiler takes a while to
      # start up on Cloud TPUs.
      if time.time() - start_time > 30:
        raise RuntimeError("Profile did not complete in 30s")
      y = jnp.dot(y, y)
    jax.profiler.stop_server()
    thread_profiler.join()
    self._check_xspace_pb_exist(logdir)

  def testDeviceVersionSavedToMetadata(self):
    with tempfile.TemporaryDirectory() as tmpdir_string:
      tmpdir = pathlib.Path(tmpdir_string)
      with jax.profiler.trace(tmpdir):
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count()))

      proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
      self.assertEqual(len(proto_path), 1)
      (proto_file,) = proto_path
      proto = proto_file.read_bytes()
      if jtu.test_device_matches(["tpu"]):
        self.assertIn(b"libtpu_version", proto)
      if jtu.test_device_matches(["gpu"]):
        self.assertIn(b"cuda_version", proto)
        self.assertIn(b"cuda_runtime_version", proto)
        self.assertIn(b"cuda_driver_version", proto)

  @unittest.skip("Profiler takes >30s on Cloud TPUs")
  @unittest.skipIf(
      not (portpicker and _pywrap_profiler_plugin),
    "Test requires xprof and portpicker")
  def test_remote_profiler_gcs_path(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port)

    profile_done = threading.Event()
    logdir = "gs://mock-test-bucket/test-dir"
    # Mock XProf call in collect_profile.
    _pywrap_profiler_plugin.trace = unittest.mock.MagicMock()
    def on_profile():
      jax.collect_profile(port, 500, logdir, no_perfetto_link=True)
      profile_done.set()

    thread_profiler = threading.Thread(
        target=on_profile, args=())
    thread_profiler.start()
    start_time = time.time()
    y = jnp.zeros((5, 5))
    while not profile_done.is_set():
      # The timeout here must be relatively high. The profiler takes a while to
      # start up on Cloud TPUs.
      if time.time() - start_time > 30:
        raise RuntimeError("Profile did not complete in 30s")
      y = jnp.dot(y, y)
    jax.profiler.stop_server()
    thread_profiler.join()
    _pywrap_profiler_plugin.trace.assert_called_once_with(
        unittest.mock.ANY,
        logdir,
        unittest.mock.ANY,
        unittest.mock.ANY,
        unittest.mock.ANY,
        unittest.mock.ANY,
        unittest.mock.ANY,
    )

  def test_advanced_configuration_getter(self):
    options = jax.profiler.ProfileOptions()
    advanced_config = {
        "tpu_trace_mode": "TRACE_COMPUTE",
        "tpu_num_sparse_cores_to_trace": 1,
        "enableFwThrottleEvent": True,
    }
    options.advanced_configuration = advanced_config
    returned_config = options.advanced_configuration
    self.assertDictEqual(returned_config, advanced_config)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
