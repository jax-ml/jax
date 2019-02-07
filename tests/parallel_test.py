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

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax import lax
from jax.api import pmap, papply, jit, make_jaxpr, axisvar_split
from jax.linear_util import wrap_init

from jax.config import config
config.parse_flags_with_absl()


class PmapTest(jtu.JaxTestCase):

  def testConstantFunction(self):
    f = lambda x: 3
    ans = pmap(f, axis_name='i')(onp.ones(4))
    expected = 3 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceSum(self):
    f = lambda x: lax.parallel.psum(x, 'i')
    ans = pmap(f, axis_name='i')(onp.ones(4))
    expected = 4 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testLogSoftmax(self):
    f = lambda x: x - np.log(lax.parallel.psum(np.exp(x), 'i'))
    x = onp.log(onp.arange(1., 10., dtype=onp.float32))
    ans = pmap(f, axis_name='i')(x)
    expected = x - onp.log(onp.sum(onp.exp(x)))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNested(self):
    f = lambda x: lax.parallel.psum(lax.parallel.psum(x, 'i'), 'j')
    x = onp.ones((2, 2))
    ans1 = pmap(pmap(f, 'i'), 'j')(x)
    ans2 = pmap(pmap(f, 'j'), 'i')(x)
    expected = 4 * onp.ones((2, 2))
    self.assertAllClose(ans1, expected, check_dtypes=False)
    self.assertAllClose(ans2, expected, check_dtypes=False)


class PapplyTest(jtu.JaxTestCase):

  def testIdentity(self):
    pfun, axis_name = papply(lambda x: x, 3)
    ans = pfun(onp.arange(3))
    expected = onp.arange(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMap(self):
    pfun, axis_name = papply(np.sin, 3)
    ans = pfun(onp.arange(3.))
    expected = onp.sin(onp.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def DISABLED_testSum(self):
    pfun, axis_name = papply(np.sum, 5)

    jaxpr = make_jaxpr(pfun)(onp.zeros(5))
    expected_jaxpr = make_jaxpr(
        lambda x: lax.parallel.psum(x, axis_name))(onp.zeros(5))
    assert repr(jaxpr) == repr(expected_jaxpr)

    ans = pmap(pfun, axis_name)(onp.arange(3.))
    expected = onp.sum(onp.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def DISABLED_testLogSoftmax(self):

    def fun(x):
      return x - np.log(np.sum(np.exp(x)))

    pfun, axis_name = papply(fun, 5)

    jaxpr = make_jaxpr(pfun)(onp.zeros(5))
    expected_jaxpr = make_jaxpr(
        lambda x: x - np.log(lax.parallel.psum(np.exp(x), axis_name)))(onp.zeros(5))
    assert repr(jaxpr) == repr(expected_jaxpr)

    ans = pmap(pfun, axis_name)(onp.arange(1., 5.))
    expected = fun(onp.arange(1., 5.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAdd(self):
    x = onp.array([[1, 2, 3], [4, 5, 6]])
    expected = x + x

    pfun, axis_name = papply(np.add, 2)
    ans = pmap(pfun, axis_name)(x, x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def DISABLED_testAddBroadcasting(self):

    def fun(x):
      return x + 3

    x = onp.array([[1, 2], [3, 4]])
    expected = x + 3

    pfun, axis_name = papply(fun, 2)
    ans = pmap(pfun, axis_name)(x)
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
      pfun, axis_name = papply(fun, x.shape[0])
      ans = pmap(pfun, axis_name)(x)
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
      pfun, axis_name = papply(fun, x.shape[0])
      ans = pmap(pfun, axis_name)(x)
      self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeAndAddRank2(self):

    def fun(x):
      return x + x.T

    x = onp.reshape(onp.arange(4., dtype=onp.float32), (2, 2))
    expected = x + x.T

    pfun, axis_name = papply(fun, 2)
    ans = pmap(pfun, axis_name)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def DISABLED_testTransposeAndAddRank3(self):

    def fun(x):
      return x + x.T

    x = onp.reshape(onp.arange(8., dtype=onp.float32), (2, 2, 2))
    expected = x + x.T

    pfun, axis_name = papply(fun, 2)
    ans = pmap(pfun, axis_name)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)


class SplitTest(jtu.JaxTestCase):

  def testSplitBasic(self):
    f = lambda x: lax.parallel.psum(np.sin(x), 'i')
    x = onp.ones((2, 2))
    fsplit = axisvar_split(f, 'i', ('j', 'k'))
    ans = pmap(pmap(fsplit, 'j'), 'k')(x)
    expected = onp.sum(onp.sin(x))
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
