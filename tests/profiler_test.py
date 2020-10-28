# Copyright 2020 Google LLC
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
import glob
import os
import shutil
import threading
import unittest
from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax.profiler
from jax.config import config
import jax.test_util as jtu

try:
  import portpicker
except ImportError:
  portpicker = None

try:
  from tensorflow.python.profiler import profiler_client
  from tensorflow.python.profiler import profiler_v2 as profiler
except ImportError:
  profiler_client = None

config.parse_flags_with_absl()


class ProfilerTest(unittest.TestCase):
  # These tests simply test that the profiler API does not crash; they do not
  # check functional correctness.

  def setUp(self):
    super().setUp()
    self.worker_start = threading.Event()
    self.profile_done = False

  @unittest.skipIf(not portpicker, "Test requires portpicker")
  def testStartServer(self):
    port = portpicker.pick_unused_port()
    jax.profiler.start_server(port=port)
    del port

  def testTraceContext(self):
    x = 3
    with jax.profiler.TraceContext("mycontext"):
      x = x + 2

  def testTraceFunction(self):
    @jax.profiler.trace_function
    def f(x):
      return x + 2
    self.assertEqual(f(7), 9)

    @partial(jax.profiler.trace_function, name="aname")
    def g(x):
      return x + 2
    self.assertEqual(g(7), 9)

    @partial(jax.profiler.trace_function, name="aname", akwarg="hello")
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

  @unittest.skipIf(not (portpicker or profiler_client),
    "Test requires tensorflow.profiler and portpicker")
  def testSingleWorkerSamplingMode(self, delay_ms=None):
    def on_worker(port, worker_start):
      profiler.start_server(port)
      worker_start.set()
      x = jnp.ones((1000, 1000))
      while True:
        with jax.profiler.TraceContext("atracecontext"):
          jnp.dot(x, x.T).block_until_ready()
          if self.profile_done:
            break

    def on_profile(port, logdir, worker_start):
      worker_start.wait()
      options = profiler.ProfilerOptions(
          host_tracer_level=2,
          python_tracer_level=2,
          device_tracer_level=1,
          delay_ms=delay_ms,
      )

      # Request for 1000 milliseconds of profile.
      duration_ms = 1000
      profiler_client.trace('localhost:{}'.format(port), logdir, duration_ms,
                            '', 1000, options)
      self.profile_done = True

    logdir = absltest.get_default_test_tmpdir()
    # Remove any existing log files.
    shutil.rmtree(logdir)
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

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
