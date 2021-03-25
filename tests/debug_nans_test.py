# Copyright 2019 Google LLC
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

"""Tests for --debug_nans."""

from absl.testing import absltest

import jax
import numpy as np
from unittest import SkipTest

from jax import test_util as jtu
from jax import numpy as jnp
from jax.experimental import pjit

from jax.config import config
config.parse_flags_with_absl()

class DebugNaNsTest(jtu.JaxTestCase):

  def setUp(self):
    self.cfg = config._read("jax_debug_nans")
    config.update("jax_debug_nans", True)

  def tearDown(self):
    config.update("jax_debug_nans", self.cfg)

  def testSingleResultPrimitiveNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jnp.tanh(A)
    ans.block_until_ready()

  def testMultipleResultPrimitiveNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans, _ = jnp.linalg.eigh(A)
    ans.block_until_ready()

  def testJitComputationNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jax.jit(jnp.tanh)(A)
    ans.block_until_ready()

  def testJitComputationNaN(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = jax.jit(lambda x: 0. / x)(A)
      ans.block_until_ready()

  def testJitComputationNaNContextManager(self):
    config.update("jax_debug_nans", False)
    A = jnp.array(0.)
    f = jax.jit(lambda x: 0. / x)
    ans = f(A)
    ans = f(A)
    with self.assertRaises(FloatingPointError):
      with jax.debug_nans(True):
        ans = f(A)
      ans.block_until_ready()

  def testSingleResultPrimitiveNaN(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = 0. / A
      ans.block_until_ready()

  def testCallDeoptimized(self):
    for jit in [jax.api._python_jit, jax.api._cpp_jit]:

      @jit
      def f(x):
        return jax.lax.cond(
            x == 1, lambda _: np.nan, lambda _: 2., operand=None)

      # This makes sure, when using the C++ jit, that the Python code has been
      # run to compile, and the next call won't go through `cache_miss`.
      f(2)
      # 'cond' not 'xla_call'
      msg = r"invalid value \(nan\) encountered in cond"
      with self.assertRaisesRegex(FloatingPointError, msg):
        f(1)

  def testPmap(self):
    f = jax.pmap(lambda x: 0. / x)

    with self.assertRaisesRegex(
        FloatingPointError,
        r"invalid value \(nan\) encountered in parallel computation"):
      ans = f(jnp.array([0.]))
      ans.block_until_ready()

    if jax.device_count() >= 2:
      with self.assertRaisesRegex(
          FloatingPointError,
          r"invalid value \(nan\) encountered in parallel computation"):
        ans = f(jnp.array([1., 0.]))
        ans.block_until_ready()

  def testPmapNoNaN(self):
    ans = jax.pmap(lambda x: 0. / x)(jnp.array([1.]))
    ans.block_until_ready()

  @jtu.ignore_warning(message=".*is an experimental.*")
  def testXmap(self):
    if not config.omnistaging_enabled:
      raise SkipTest("xmap requires omnistaging")

    f = jax.experimental.maps.xmap(
        lambda x: 0. / x,
        in_axes=['i'],
        out_axes=['i'],
        axis_resources={'i': 'x'})

    with jax.experimental.maps.mesh(np.array(jax.local_devices()[:1]), ('x',)):
      with self.assertRaisesRegex(
          FloatingPointError,
          r"invalid value \(nan\) encountered in parallel computation"):
        ans = f(jnp.array([0.]))
        ans.block_until_ready()

    if jax.device_count() >= 2:
      with jax.experimental.maps.mesh(np.array(jax.local_devices()[:2]), ('x',)):
        with self.assertRaises(FloatingPointError):
          ans = f(jnp.array([1., 0.]))
          ans.block_until_ready()

  @jtu.ignore_warning(message=".*is an experimental.*")
  @jtu.skip_on_devices("cpu", "gpu")
  def testPjit(self):
    if jax.device_count() < 2:
      raise SkipTest("test requires >=2 devices")

    p = jax.experimental.PartitionSpec('x')
    f = pjit.pjit(lambda x: 0. / x,
                  in_axis_resources=p,
                  out_axis_resources=p)

    with jax.experimental.maps.mesh(np.array(jax.local_devices()[:2]), ('x',)):
      with self.assertRaises(FloatingPointError):
        ans = f(jnp.array([0., 1.]))
        ans.block_until_ready()

  # TODO(skye): add parallel inf tests, ideally by factoring out test logic

class DebugInfsTest(jtu.JaxTestCase):

  def setUp(self):
    self.cfg = config._read("jax_debug_infs")
    config.update("jax_debug_infs", True)

  def tearDown(self):
    config.update("jax_debug_infs", self.cfg)

  def testSingleResultPrimitiveNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jnp.tanh(A)
    ans.block_until_ready()

  def testMultipleResultPrimitiveNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans, _ = jnp.linalg.eigh(A)
    ans.block_until_ready()

  def testJitComputationNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jax.jit(jnp.tanh)(A)
    ans.block_until_ready()

  def testSingleResultPrimitiveInf(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = 1. / A
      ans.block_until_ready()

  def testCallDeoptimized(self):
    for jit in [jax.api._python_jit, jax.api._cpp_jit]:

      @jit
      def f(x):
        return jax.lax.cond(
            x == 1, lambda _: np.inf, lambda _: 2., operand=None)

      # This makes sure, when using the C++ jit, that the Python code has been
      # run to compile, and the next call won't go through `cache_miss`.
      f(2)
      # 'cond' not 'xla_call'
      msg = r"invalid value \(inf\) encountered in cond"
      with self.assertRaisesRegex(FloatingPointError, msg):
        f(1)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
