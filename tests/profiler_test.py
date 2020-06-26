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
import unittest

from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax.profiler
from jax.config import config
import jax.test_util

try:
  import portpicker
except ImportError:
  portpicker = None

config.parse_flags_with_absl()


class ProfilerTest(unittest.TestCase):
  # These tests simply test that the profiler API does not crash; they do not
  # check functional correctness.

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

  def testHeapProfile(self):
    x = jnp.ones((20,)) + 7.
    self.assertTrue(isinstance(jax.profiler.heap_profile(), bytes))

if __name__ == "__main__":
  absltest.main()
