# Copyright 2018 Google LLC
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

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax import lax
from jax.api import pmap, vmap, jvp, grad, make_jaxpr, linearize, device_put
from jax.lax import psum
from jax.lib import xla_bridge
from jax.util import prod
from jax.interpreters import pxla

from jax.config import config
config.parse_flags_with_absl()


class PmapTest(jtu.JaxTestCase):

  def testBasic(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = x - onp.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNestedBasic(self):
    f = lambda x: lax.psum(lax.psum(x, 'i'), 'j')
    f = pmap(pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return onp.repeat(onp.sum(x, axis, keepdims=True), x.shape[axis], axis)

    shape = (xla_bridge.device_count(), 1, 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = f(x)
    expected = sum_and_broadcast(sum_and_broadcast(x, 0), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testNestedShardingAndStacking(self, device_mesh_shape):
    device_count = xla_bridge.device_count()
    try:
      mesh_shape = onp.arange(device_count).reshape(device_mesh_shape).shape
    except ValueError:
      raise SkipTest("incompatible device count for test: {}")

    f = lambda x: x
    f = pmap(pmap(f, 'i'), 'j')

    shape = mesh_shape + (4,)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = f(x)
    expected = x
    self.assertEqual(ans.shape, expected.shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testJvpAndPartialEval(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return np.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(np.ones_like(x))

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    expected = onp.cos(x)

    ans = splitjvp(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    make_jaxpr(splitjvp)(x)  # doesn't crash

  def testGradBasic(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return np.sin(x)

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = grad(lambda x: np.sum(np.sin(x)))(x)
    expected = grad(lambda x: np.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGradOfJvp(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return np.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(np.ones_like(x))

    fun = lambda x: np.sum(jvp(np.sin, (x,), (np.ones_like(x),))[1])

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = grad(lambda x: np.sum(splitjvp(x)))(x)
    expected = grad(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testTwoArgsGrad(self):
    def f(x, y):
      return lax.psum(5. * np.cos(x) * np.sin(y), 'i')
    f = pmap(f, 'i')

    def g(x, y):
      tot = np.sum(5. * np.cos(x) * np.sin(y))
      return tot * np.ones_like(x)  # broadcast to map like pjit does

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)
    y = 4 + x
    ans = grad(lambda x, y: np.sum(g(x, y)))(x, y)
    expected = grad(lambda x, y: np.sum(g(x, y)))(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testNestedWithClosure(self, device_mesh_shape):
    device_count = xla_bridge.device_count()
    try:
      mesh_shape = onp.arange(device_count).reshape(device_mesh_shape).shape
    except ValueError:
      raise SkipTest("incompatible device count for test: {}")

    @partial(pmap, axis_name='i')
    def test_fun(x):
      y = np.sum(np.sin(x))

      @partial(pmap, axis_name='j')
      def g(z):
        return 3. * np.exp(np.sin(x).sum() * np.cos(y) * np.tan(z))

      return grad(lambda w: np.sum(g(w)))(x)

    @vmap
    def baseline_fun(x):
      y = np.sum(np.sin(x))

      @vmap
      def g(z):
        return 3. * np.exp(np.sin(x).sum() * np.cos(y) * np.tan(z))

      return grad(lambda w: np.sum(g(w)))(x)

    shape = mesh_shape + (4,)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    ans = grad(lambda x: np.sum(test_fun(x)))(x)
    expected = grad(lambda x: np.sum(baseline_fun(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testShardedDeviceValues(self):
    f = lambda x: 2 * x
    f = pmap(f, axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = onp.arange(prod(shape), dtype=onp.float32).reshape(shape)

    # test that we can pass in and out ShardedDeviceArrays
    y = f(x)
    assert type(y) is pxla.ShardedDeviceArray  # pylint: disable=unidiomatic-typecheck
    self.assertAllClose(y, 2 * x, check_dtypes=False)
    z = f(y)
    assert type(z) is pxla.ShardedDeviceArray  # pylint: disable=unidiomatic-typecheck
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can pass in a regular DeviceArray
    y = f(device_put(x))
    assert type(y) is pxla.ShardedDeviceArray  # pylint: disable=unidiomatic-typecheck
    self.assertAllClose(y, 2 * x, check_dtypes=False)

    # test that we can pass a ShardedDeviceArray to a regular jit computation
    z = y + y
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can handle device movement on dispatch
    y.device_buffers = y.device_buffers[::-1]
    z = f(y)
    self.assertAllClose(z, 2 * 2 * x[::-1], check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
