# Copyright 2021 Google LLC
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
"""Tests for interoperability between JAX and pickling libraries."""

import pickle
import unittest

from absl.testing import absltest

try:
  import cloudpickle
except ImportError:
  cloudpickle = None

import jax
from jax import core
from jax import numpy as jnp
from jax.config import config
from jax._src import test_util as jtu
import jax._src.lib

config.parse_flags_with_absl()


class CloudpickleTest(jtu.JaxTestCase):

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def testPickleOfJittedFunctions(self):

    @jax.jit
    def f(x, y):
      return x * y

    @jax.jit
    def g(z):
      return f(z, z + 77)  # noqa: F821

    expected = g(32)
    s = cloudpickle.dumps(g)
    del f, g

    g_unpickled = pickle.loads(s)
    actual = g_unpickled(32)
    self.assertEqual(expected, actual)

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def testPickleOfPmappedFunctions(self):

    @jax.pmap
    def f(x, y):
      return x * y

    @jax.pmap
    def g(z):
      return f(z, z + 77)  # noqa: F821

    expected = g(jnp.asarray([[32]]))
    s = cloudpickle.dumps(g)
    del f, g

    g_unpickled = pickle.loads(s)
    actual = g_unpickled(jnp.asarray([[32]]))
    self.assertEqual(expected, actual)


class PickleTest(jtu.JaxTestCase):

  def testPickleOfDeviceArray(self):
    x = jnp.arange(10.0)
    s = pickle.dumps(x)
    y = pickle.loads(s)
    self.assertArraysEqual(x, y)
    self.assertIsInstance(y, type(x))
    self.assertEqual(x.aval, y.aval)

  def testPickleOfDeviceArrayWeakType(self):
    x = jnp.array(4.0)
    self.assertEqual(x.aval.weak_type, True)
    s = pickle.dumps(x)
    y = pickle.loads(s)
    self.assertArraysEqual(x, y)
    self.assertIsInstance(y, type(x))
    self.assertEqual(x.aval, y.aval)

  def testPickleX64(self):
    with jax.experimental.enable_x64():
      x = jnp.array(4.0, dtype='float64')
      s = pickle.dumps(x)

    with jax.experimental.disable_x64():
      y = pickle.loads(s)

    self.assertEqual(x.dtype, jnp.float64)
    self.assertArraysEqual(x, y, check_dtypes=False)
    self.assertEqual(y.dtype, jnp.float32)
    self.assertEqual(y.aval.dtype, jnp.float32)
    self.assertIsInstance(y, type(x))

  def testPickleTracerError(self):
    with self.assertRaises(core.ConcretizationTypeError):
      jax.jit(pickle.dumps)(0)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
