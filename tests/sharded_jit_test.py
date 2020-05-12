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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from unittest import SkipTest

import numpy as np
from absl.testing import absltest

import jax
from jax import test_util as jtu
from jax.interpreters import pxla
from jax.interpreters.sharded_jit import sharded_jit, set_sharding
from jax.interpreters.sharded_jit import PartitionSpec as P

from jax.config import config
config.parse_flags_with_absl()


class ShardedJitTest(jtu.JaxTestCase):

  def setUp(self):
    if jtu.device_under_test() != "tpu":
      raise SkipTest

  def testBasic(self):
    if jax.device_count() < 2:
      raise SkipTest

    @partial(sharded_jit, in_parts=(P(2, 1), P(2, 1)), out_parts=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(actual.device_buffers[0].to_py(), expected,
                        check_dtypes=False)

  def testPyTreeArgs(self):
    if jax.device_count() < 2:
      raise SkipTest

    def f(a, b, c):
      a1, a2 = a
      c1, (c2, c3) = c
      return a1 + a2 + b + c1 + c2 + c3

    def _make_arg(*shape):
      return np.arange(np.prod(shape)).reshape(shape)

    a = (_make_arg(4, 4), 1)
    b = _make_arg(4, 4)
    c = [2, (_make_arg(4, 4), _make_arg(4, 4))]

    in_parts = (None, P(2, 1), [None, P(2, 1)])
    out_parts = P(2, 1)

    result = sharded_jit(f, in_parts, out_parts)(a, b, c)
    expected = f(a, b, c)

    self.assertAllClose(result, expected, check_dtypes=False)
    self.assertIsInstance(result, pxla.ShardedDeviceArray)
    self.assertLen(result.device_buffers, 2)

    in_parts = None
    result = sharded_jit(f, in_parts, out_parts)(a, b, c)
    self.assertAllClose(result, expected, check_dtypes=False)
    self.assertIsInstance(result, pxla.ShardedDeviceArray)
    self.assertLen(result.device_buffers, 2)

  def testPyTreeOutputs(self):
    if jax.device_count() < 2:
      raise SkipTest

    def f(x):
      return x + 1, ((x + 2, x + 3), x + 4)

    shape = (4, 4)
    x = np.arange(np.prod(shape)).reshape(shape)
    in_parts = (P(2, 1),)
    out_parts = (P(2, 1), ((P(1, 2), None), P(2, 1)))

    result = sharded_jit(f, in_parts, out_parts)(x)
    expected = f(x)
    self.assertAllClose(result, expected, check_dtypes=False)

    out_parts = None
    result = sharded_jit(f, in_parts, out_parts)(x)
    self.assertAllClose(result, expected, check_dtypes=False)

  def testAllArgsOutputsReplicated(self):
    @partial(sharded_jit, in_parts=None, out_parts=None)
    def f(x):
      return x + 1

    result = f(1.)
    self.assertEqual(result, 2.)
    self.assertIsInstance(result, pxla.ShardedDeviceArray)
    self.assertLen(result.device_buffers, 1)

  def testSetSharding(self):
    def f(x):
      y = x + 1
      y = set_sharding(y, P(1,2))
      return y * 2

    shape = (8, 8)
    x = np.arange(np.prod(shape)).reshape(shape)
    expected = (x + 1) * 2

    # Matching sharded_jit partitions
    actual = sharded_jit(f, in_parts=P(2,1), out_parts=P(2,1))(x)
    self.assertAllClose(actual, expected)
    self.assertLen(actual.device_buffers, 2)
    self.assertEqual(actual.device_buffers[0].shape(), (4,8))
    self.assertEqual(actual.device_buffers[1].shape(), (4,8))

    # Mismatched sharded_jit partitions
    with self.assertRaisesRegex(
        ValueError, "fooooo"):
      sharded_jit(f, in_parts=P(2,2), out_parts=P(2,2))(x)

    # Replicated sharded_jit
    actual = sharded_jit(f, in_parts=None, out_parts=None)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(actual.device_buffers[0].to_py(),
                        actual.device_buffers[1].to_py(),
                        check_dtypes=False)

  def testGradOfSetSharding(self):
    @partial(sharded_jit, partitions=(P(4,1), None))
    def f(x):
      y = x + 1
      p, vjp_f = vjp(lambda z: np.sin(set_sharding(z, P(2,2))), y)
      return vjp_f(p)

    def expected_f(x):
      y = x + 1
      p, vjp_f = vjp(lambda z: np.sin(z), y)
      return vjp_f(p)

    shape = (4, 4)
    x = onp.arange(onp.prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x)
    expected = expected_f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)

# TODO(skye): add error tests

# Tests that don't need a TPU to run.
class ShardedJitTestNoTpu(jtu.JaxTestCase):

  def testTranslationRule(self):
    @partial(sharded_jit, in_parts=(P(2, 1), P(2, 1)), out_parts=None)
    def f(x, y):
      return x + y

    # Test that the translation rule runs without error and produces the
    # OpShardings we expect somewhere.
    shape = (8, 8)
    hlo = jax.xla_computation(f)(np.ones(shape), np.ones(shape))
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  def testSetShardingAnnotation(self):
    @partial(sharded_jit, in_parts=None, out_parts=None)
    def f(x):
      y = x + 1
      y = set_sharding(y, P(2,1))
      return y * 2

    shape = (8, 8)
    hlo = jax.xla_computation(f)(np.ones(shape))
    # Annotation from set_sharding
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    # Annotation from sharded_jit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())


if __name__ == "__main__":
  absltest.main()
