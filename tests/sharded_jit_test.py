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
from absl.testing import parameterized

import jax
from jax import jit, pmap, vjp
from jax import lax
from jax import test_util as jtu
from jax import tree_util
from jax.interpreters import pxla
from jax.interpreters.sharded_jit import sharded_jit, with_sharding_constraint
from jax.interpreters.sharded_jit import PartitionSpec as P
import jax.numpy as jnp

from jax.config import config
config.parse_flags_with_absl()


class ShardedJitTest(jtu.JaxTestCase):

  def setUp(self):
    super(ShardedJitTest, self).setUp()
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

  def testShardingConstraint(self):
    if jax.local_device_count() < 2:
      raise SkipTest("requires 2 devices")

    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, P(1,2))
      return y * 2

    shape = (8, 8)
    x = np.arange(np.prod(shape)).reshape(shape)
    expected = (x + 1) * 2

    # Matching sharded_jit partitions
    actual = sharded_jit(f, in_parts=P(2,1), out_parts=P(2,1))(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual.device_buffers, 2)
    self.assertEqual(actual.device_buffers[0].shape().dimensions(), (4,8))
    self.assertEqual(actual.device_buffers[1].shape().dimensions(), (4,8))

    # Mismatched sharded_jit partitions
    with self.assertRaisesRegex(
        ValueError,
        r"with_sharding_constraint with partitions=PartitionSpec\(1, 2\) "
        r"\(total partitions: 2\) doesn't match expected number of partitions: "
        r"4. If these partitions look right, check outer sharded_jit and/or "
        r"other with_sharding_constraint calls."):
      sharded_jit(f, in_parts=P(2,2), out_parts=P(2,2))(x)

    # Replicated sharded_jit
    actual = sharded_jit(f, in_parts=None, out_parts=None)(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(actual.device_buffers[0].to_py(),
                        actual.device_buffers[1].to_py(),
                        check_dtypes=False)

  def testNestedShardingConstraint(self):
    if jax.local_device_count() < 2:
      raise SkipTest("requires 2 devices")

    shape = (8, 8)

    @jit
    def f(x):
      return lax.while_loop(lambda i: i[0,0] < 10.,
                            lambda i: with_sharding_constraint(i + 1., P(2, 1)),
                            x)

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = x + 10.
    actual = sharded_jit(f, in_parts=None, out_parts=None)(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual.device_buffers, 2)

  def testGradOfShardingConstraint(self):
    if jax.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    @partial(sharded_jit, in_parts=P(4,1), out_parts=None)
    def f(x):
      y = x + 1
      p, vjp_f = vjp(lambda z: jnp.sin(with_sharding_constraint(z, P(2,2))), y)
      return vjp_f(p)

    def expected_f(x):
      y = x + 1
      p, vjp_f = vjp(lambda z: jnp.sin(z), y)
      return vjp_f(p)

    shape = (4, 4)
    x = jnp.arange(jnp.prod(shape), dtype=jnp.float32).reshape(shape)
    actual = f(x)
    expected = expected_f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)

  @parameterized.named_parameters({
      "testcase_name": f"_partition_input={partition_input}",
      "partition_input": partition_input
  } for partition_input in [True, False])
  def testInfeed(self, partition_input):
    if jax.local_device_count() % 2 != 0:
      raise SkipTest

    shape = (jax.local_device_count() * 2, 4)
    # Run computation across all devices so we know which devices to feed.
    parts = P(jax.local_device_count(), 1)
    in_parts = parts if partition_input else None
    infeed_shapes = (jax.ShapedArray(shape, np.float32),
                     jax.ShapedArray((1,), np.float32))
    infeed_parts = (parts, None)

    @partial(sharded_jit, in_parts=in_parts, out_parts=None)
    def f(x):
      token = lax.create_token(x)
      (y, z), token = lax.infeed(token, infeed_shapes, partitions=infeed_parts)
      return x @ y.T + z

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = x + 1
    shard_size = shape[0] // jax.local_device_count()
    y_shards = [y[i:i+shard_size] for i in range(0, shape[0], shard_size)]
    z = jnp.array([3.], dtype=np.float32)

    result = f(x)
    assert len(jax.local_devices()) == len(y_shards)
    for device, y_shard in zip(jax.local_devices(), y_shards):
      device.transfer_to_infeed((y_shard, z))

    expected = x @ y.T + z
    self.assertAllClose(result, expected, check_dtypes=False)


# TODO(skye): add more error tests
class ShardedJitErrorsTest(jtu.JaxTestCase):

  def setUp(self):
    super(ShardedJitErrorsTest, self).setUp()
    if jtu.device_under_test() != "tpu":
      raise SkipTest

  def testNotEnoughDevices(self):
    ndevices = jax.local_device_count()

    @partial(sharded_jit, in_parts=P(ndevices + 1), out_parts=None)
    def f(x):
      return x + x

    with self.assertRaisesRegex(
        ValueError,
        f"sharded_jit computation requires {ndevices + 1} devices, "
        f"but only {ndevices} devices are available."):
      f(np.ones(ndevices + 1))


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

  def testShardingConstraintAnnotation(self):
    @partial(sharded_jit, in_parts=None, out_parts=None)
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, P(2,1))
      return y * 2

    shape = (8, 8)
    hlo = jax.xla_computation(f)(np.ones(shape))
    # Annotation from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    # Annotation from sharded_jit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

class PmapOfShardedJitTest(jtu.JaxTestCase):

  def setUp(self):
    super(PmapOfShardedJitTest, self).setUp()
    if jtu.device_under_test() != "tpu":
      raise SkipTest

  # TODO(skye): make a similar version for ShardedJitTest and run the same tests
  def _runTest(self, f, in_partitions, out_partitions, dtype=np.float32):
    """Compares pmap(sharded_jit(f, ...)) to pmap(f)"""
    shape = (2, 4, 4)
    num_shards = shape[0] * np.prod(in_partitions[0])
    if num_shards > jax.local_device_count():
      raise SkipTest("requires %d devices" % num_shards)

    x = np.arange(np.prod(shape, dtype=dtype)).reshape(shape)
    y = x + 1
    result = pmap(
        sharded_jit(f, in_parts=in_partitions, out_parts=out_partitions))(x, y)
    expected = pmap(f)(x, y)
    self.assertAllClose(result, expected, check_dtypes=False)

    flat_result = tree_util.tree_flatten(result)[0]
    for r in flat_result:
      self.assertTrue(isinstance(r, pxla.ShardedDeviceArray))
      self.assertEqual(len(r.device_buffers), num_shards)


  @parameterized.named_parameters({
      "testcase_name":
          "_in_parts={}_out_parts={}".format(in_partitions,
                                             out_partitions).replace(" ", ""),
      "in_partitions":
          in_partitions,
      "out_partitions":
          out_partitions
  } for in_partitions in [
      (P(2, 1), P(2, 1)),
      (P(2, 1), P(1, 2)),
      (P(2, 2), P(2, 2)),
      (P(4, 1), P(2, 2)),
  ] for out_partitions in [in_partitions[0], None])
  def testBasic(self, in_partitions, out_partitions):

    def f(x, y):
      return lax.dot(x, y)

    self._runTest(f, in_partitions, out_partitions)

  @parameterized.named_parameters({
      "testcase_name":
          "_in_parts={}_out_parts={}".format(in_partitions,
                                             out_partitions).replace(" ", ""),
      "in_partitions":
          in_partitions,
      "out_partitions":
          out_partitions
  } for in_partitions in [
      (P(2, 1), P(2, 1)),
      (P(2, 1), P(1, 2)),
      (P(4, 1), P(2, 2))
  ] for out_partitions in [(in_partitions[1], in_partitions[0], None),
                           (None, None, None)])
  def testMultipleOutputs(self, in_partitions, out_partitions):

    def f(x, y):
      a = lax.dot(x, y)
      # TODO(skye): use these more interesting outputs once returning constants
      # works
      # return a, a + 1, 3
      return a, a + x, x + y

    self._runTest(f, in_partitions, out_partitions)

  @parameterized.named_parameters({
      "testcase_name":
          "_in_parts={}_out_parts={}".format(in_partitions,
                                             out_partitions).replace(" ", ""),
      "in_partitions":
          in_partitions,
      "out_partitions":
          out_partitions
  } for in_partitions in [
      (P(2, 1), P(2, 1)),
      (P(2, 1), P(1, 2)),
      (P(4, 1), P(2, 2))
  ] for out_partitions in [in_partitions[0], None])
  def testArrayConstants(self, in_partitions, out_partitions):

    def f(x, y):
      a = lax.dot(x, y)
      b = a + jnp.ones(a.shape)
      c = b + jnp.ones(a.shape[0])
      return c

    self._runTest(f, in_partitions, out_partitions)

  def testPyTreeArgs(self):
    if jax.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    def f(a, b, c):
      a1, a2 = a
      c1, (c2, c3) = c
      return a1 + a2 + b + c1 + c2 + c3

    def _make_arg(*shape):
      return np.arange(np.prod(shape)).reshape(shape)

    a = (_make_arg(2, 4, 4), _make_arg(2))
    b = _make_arg(2, 4, 4)
    c = (_make_arg(2), (_make_arg(2, 4, 4), _make_arg(2, 4, 4)))

    in_parts = (None, P(2, 1), (None, P(2, 1)))
    out_parts = P(2, 1)

    result = pmap(sharded_jit(f, in_parts=in_parts, out_parts=out_parts))(a, b, c)
    expected = pmap(f)(a, b, c)

    self.assertAllClose(result, expected, check_dtypes=False)
    self.assertTrue(isinstance(result, pxla.ShardedDeviceArray))
    self.assertEqual(len(result.device_buffers), 4)

  def testPyTreeOutputs(self):
    if jax.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    def f(x):
      return x + 1, ((x + 2, x + 3), x + 4)

    shape = (2, 4, 4)
    x = np.arange(np.prod(shape)).reshape(shape)
    in_parts = (P(2, 1),)
    out_parts = (P(2, 1), ((P(1, 2), None), P(2, 1)))

    result = pmap(sharded_jit(f, in_parts=in_parts, out_parts=out_parts))(x)
    expected = pmap(f)(x)

    self.assertAllClose(result, expected, check_dtypes=False)

  def testManyArgs(self):
    if jax.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    num_args = 200

    def f(*args):
      return jnp.sum(args)

    shape = (2, 4, 4)
    args = [np.arange(np.prod(shape)).reshape(shape)] * num_args
    in_partitions = (P(2, 1),) * num_args
    out_partitions = None
    result = pmap(sharded_jit(
        f, in_parts=in_partitions, out_parts=out_partitions))(*args)
    expected = pmap(f)(*args)

    self.assertAllClose(result, expected, check_dtypes=False)
    self.assertTrue(isinstance(result, pxla.ShardedDeviceArray))
    self.assertEqual(len(result.device_buffers), 4)

  def testShardingConstraint(self):
    if jax.local_device_count() < 4:
      raise SkipTest("requires 4 devices")

    @partial(sharded_jit, in_parts=None, out_parts=None)
    def f(x):
      y = jnp.dot(x, x)
      y = with_sharding_constraint(y, P(2,1))
      return y * 2

    def expected_f(x):
      return jnp.dot(x, x) * 2

    shape = (2, 8, 8)
    x = np.arange(np.prod(shape)).reshape(shape)
    result = pmap(f)(x)
    expected = pmap(expected_f)(x)

    self.assertAllClose(result, expected, check_dtypes=False)
    self.assertIsInstance(result, pxla.ShardedDeviceArray)
    self.assertLen(result.device_buffers, 4)

  def testInAxesNone(self):
    shape = (4, 4)
    replicas = 2
    in_partitions = (P(2, 1), None, None)
    out_partitions = P(2, 1)
    in_axes = (None, None, 0)
    x = y = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    dummy = np.arange(replicas, dtype=np.float32) + 1
    num_shards = replicas * np.prod(in_partitions[0])
    if num_shards > jax.local_device_count():
      raise SkipTest("requires %d devices" % num_shards)

    def f(x, y, _):
      return x @ y

    result = pmap(
        sharded_jit(f, in_parts=in_partitions, out_parts=out_partitions),
        in_axes=in_axes)(x, y, dummy)
    expected = pmap(f, in_axes=in_axes)(x, y, dummy)
    self.assertAllClose(result, expected, check_dtypes=True)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
