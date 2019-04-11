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

from unittest import SkipTest

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax import lax
from jax import lax_parallel
from jax.api import _serial_pmap, _papply, jit, make_jaxpr
from jax.linear_util import wrap_init

from jax.config import config
config.parse_flags_with_absl()


class SerialPmapTest(jtu.JaxTestCase):

  def testConstantFunction(self):
    f = lambda x: 3
    ans = _serial_pmap(f, axis_name='i')(onp.ones(4))
    expected = 3 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceSum(self):
    f = lambda x: lax_parallel.psum(x, 'i')
    ans = _serial_pmap(f, axis_name='i')(onp.ones(4))
    expected = 4 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceMax(self):
    f = lambda x: lax_parallel.pmax(x, 'i')
    ans = _serial_pmap(f, axis_name='i')(onp.arange(4))
    expected = 3 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPsplit(self):
    f = lambda x: lax_parallel.psplit(x, 'i', 2)
    arg = onp.arange(3 * 2 * 3 * 5).reshape(3, 2, 3, 5)
    ans = _serial_pmap(f, axis_name='i', out_axes=2)(arg)
    expected = arg
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPsplitLike(self):
    f = lambda x, y: lax_parallel.psplit_like(x, y, 'i')
    arg = onp.arange(3 * 2 * 3 * 5).reshape(3, 2, 3, 5)
    ans = _serial_pmap(f, axis_name='i', in_axes=(None, 2), out_axes=2)(arg, arg)
    expected = arg
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testLogSoftmax(self):
    f = lambda x: x - np.log(lax_parallel.psum(np.exp(x), 'i'))
    x = onp.log(onp.arange(1., 10., dtype=onp.float32))
    ans = _serial_pmap(f, axis_name='i')(x)
    expected = x - onp.log(onp.sum(onp.exp(x)))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNested(self):
    f = lambda x: lax_parallel.psum(lax_parallel.psum(x, 'i'), 'j')
    x = onp.ones((2, 2))
    ans1 = _serial_pmap(_serial_pmap(f, 'i'), 'j')(x)
    ans2 = _serial_pmap(_serial_pmap(f, 'j'), 'i')(x)
    expected = 4 * onp.ones((2, 2))
    self.assertAllClose(ans1, expected, check_dtypes=False)
    self.assertAllClose(ans2, expected, check_dtypes=False)


class PapplyTest(jtu.JaxTestCase):

  def testIdentity(self):
    pfun, axis_name = _papply(lambda x: x, 3)
    ans = pfun(onp.arange(3))
    expected = onp.arange(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMap(self):
    pfun, axis_name = _papply(np.sin, 3)
    ans = pfun(onp.arange(3.))
    expected = onp.sin(onp.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSum(self):
    pfun, axis_name = _papply(lambda x: np.sum(x, axis=0), 5)

    jaxpr = make_jaxpr(pfun)(onp.ones(3))
    expected_jaxpr = make_jaxpr(
        lambda x: lax_parallel.psum(x, axis_name))(onp.zeros((5, 3)))
    assert repr(jaxpr) == repr(expected_jaxpr)

    arg = onp.arange(15.).reshape((5, 3))
    ans = _serial_pmap(pfun, axis_name)(arg)[0]
    expected = onp.sum(arg, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMax(self):
    pfun, axis_name = _papply(lambda x: np.max(x, axis=0), 5)

    jaxpr = make_jaxpr(pfun)(onp.ones(3))
    expected_jaxpr = make_jaxpr(
        lambda x: lax_parallel.pmax(x, axis_name))(onp.zeros((5, 3)))
    assert repr(jaxpr) == repr(expected_jaxpr)

    arg = onp.arange(15.).reshape((5, 3))
    ans = _serial_pmap(pfun, axis_name)(arg)[0]
    expected = onp.max(arg, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSelect(self):
    pfun, axis_name = _papply(lax.select, 5,
                             in_axes=(None, 0, None))

    p = onp.arange(15).reshape((5, 3)) % 4 == 1
    t = onp.ones((5, 3))
    f = onp.zeros((5, 3))
    jaxpr = make_jaxpr(pfun)(p, t[0], f)

    def expected_spmd(p, t, f):
      return lax.select(
          lax_parallel.psplit_like(p, t, axis_name),
          t,
          lax_parallel.psplit_like(f, t, axis_name))

    expected_jaxpr = make_jaxpr(expected_spmd)(p, t[0], f)
    assert repr(jaxpr) == repr(expected_jaxpr)

    ans = _serial_pmap(pfun, axis_name, in_axes=(None, 0, None))(p, t, f)
    expected = lax.select(p, t, f)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testLogSoftmax(self):
    return SkipTest("test doesn't pass yet")  # TODO(frostig)

    def fun(x):
      return x - np.log(np.sum(np.exp(x)))

    pfun, axis_name = _papply(fun, 5)

    jaxpr = make_jaxpr(pfun)(onp.zeros(5))
    expected_jaxpr = make_jaxpr(
        lambda x: x - np.log(lax_parallel.psum(np.exp(x), axis_name)))(onp.zeros(5))
    assert repr(jaxpr) == repr(expected_jaxpr)

    ans = _serial_pmap(pfun, axis_name)(onp.arange(1., 5.))
    expected = fun(onp.arange(1., 5.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAdd(self):
    x = onp.array([[1, 2, 3], [4, 5, 6]])
    expected = x + x

    pfun, axis_name = _papply(np.add, 2)
    ans = _serial_pmap(pfun, axis_name)(x, x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testAddBroadcasting(self):
    return SkipTest("test doesn't pass yet")  # TODO(frostig)

    def fun(x):
      return x + 3

    x = onp.array([[1, 2], [3, 4]])
    expected = x + 3

    pfun, axis_name = _papply(fun, 2)
    ans = _serial_pmap(pfun, axis_name)(x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testTranspose(self):

    def fun(x):
      return x.T

    xs = [
        onp.reshape(onp.arange(4., dtype=onp.float32), (2, 2)),
        onp.reshape(onp.arange(9., dtype=onp.float32), (3, 3)),
    ]
    for x in xs:
      expected = x.T
      pfun, axis_name = _papply(fun, x.shape[0])
      ans = _serial_pmap(pfun, axis_name)(x)
      self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeWithOddPermutation(self):

    def fun(x):
      return np.transpose(x, (2, 0, 1))

    xs = [
        onp.reshape(onp.arange(8., dtype=onp.float32), (2, 2, 2)),
        onp.reshape(onp.arange(27., dtype=onp.float32), (3, 3, 3)),
    ]
    for x in xs:
      expected = np.transpose(x, (2, 0, 1))
      pfun, axis_name = _papply(fun, x.shape[0])
      ans = _serial_pmap(pfun, axis_name)(x)
      self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeAndAddRank2(self):

    def fun(x):
      return x + x.T

    x = onp.reshape(onp.arange(4., dtype=onp.float32), (2, 2))
    expected = x + x.T

    pfun, axis_name = _papply(fun, 2)
    ans = _serial_pmap(pfun, axis_name)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeAndAddRank3(self):

    def fun(x):
      return x + x.T

    x = onp.reshape(onp.arange(8., dtype=onp.float32), (2, 2, 2))
    expected = x + x.T

    pfun, axis_name = _papply(fun, 2)
    ans = _serial_pmap(pfun, axis_name)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot(self):
    return SkipTest("test doesn't pass yet")  # TODO(frostig)

    def fun(x, y):
      return lax.dot(x, y)
    xs = [
        onp.reshape(onp.arange(4., dtype=onp.float32), (2, 2)),
        onp.reshape(onp.arange(9., dtype=onp.float32), (3, 3)),
    ]
    in_axes_combos = [(0, 0), (0, 1)] # [(1, 0)]
    for in_axes in in_axes_combos:
      for x in xs:
        expected = fun(x, x)
        pfun, axis_name = _papply(fun, x.shape[0], in_axes=in_axes)
        ans = _serial_pmap(pfun, axis_name)(x, x)
        self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
