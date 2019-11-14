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

import itertools
import unittest
from unittest import SkipTest

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax import lax
from jax.api import _papply, _parallelize, soft_pmap, jit, make_jaxpr
from jax.linear_util import wrap_init
from jax.util import prod

from jax.config import config
config.parse_flags_with_absl()


class PapplyTest(jtu.JaxTestCase):

  def testIdentity(self):
    pfun, axis_name = _papply(lambda x: x)
    ans = pfun(onp.arange(3))
    expected = onp.arange(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMap(self):
    pfun, axis_name = _papply(np.sin)
    ans = pfun(onp.arange(3.))
    expected = onp.sin(onp.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSum(self):
    pfun, axis_name = _papply(lambda x: np.sum(x, axis=0))

    jaxpr = make_jaxpr(pfun)(onp.ones(3))
    expected_jaxpr = make_jaxpr(
        lambda x: lax.psum(x, axis_name))(onp.zeros((5, 3)))
    assert repr(jaxpr) == repr(expected_jaxpr)

    arg = onp.arange(15.).reshape((5, 3))
    ans = soft_pmap(pfun, axis_name)(arg)[0]
    expected = onp.sum(arg, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMax(self):
    pfun, axis_name = _papply(lambda x: np.max(x, axis=0))

    jaxpr = make_jaxpr(pfun)(onp.ones(3))
    expected_jaxpr = make_jaxpr(
        lambda x: lax.pmax(x, axis_name))(onp.zeros((5, 3)))
    assert repr(jaxpr) == repr(expected_jaxpr)

    arg = onp.arange(15.).reshape((5, 3))
    ans = soft_pmap(pfun, axis_name)(arg)[0]
    expected = onp.max(arg, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testSelect(self):
    raise SkipTest("buggy")  # TODO(mattjj): fix
    p = onp.arange(15).reshape((5, 3)) % 4 == 1
    f = onp.zeros((5, 3))

    def fun(t):
      return lax.select(p, t, f)

    t = onp.ones((5, 3))
    ans = soft_pmap(*_papply(fun))(t)
    expected = fun(t)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testLogSoftmax(self):
    raise SkipTest("test doesn't pass yet")  # TODO(frostig)

    def fun(x):
      return x - np.log(np.sum(np.exp(x)))

    pfun, axis_name = _papply(fun)

    jaxpr = make_jaxpr(pfun)(onp.zeros(5))
    expected_jaxpr = make_jaxpr(
        lambda x: x - np.log(lax.psum(np.exp(x), axis_name)))(onp.zeros(5))
    assert repr(jaxpr) == repr(expected_jaxpr)

    ans = soft_pmap(pfun, axis_name)(onp.arange(1., 5.))
    expected = fun(onp.arange(1., 5.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAdd(self):
    x = onp.array([[1, 2, 3], [4, 5, 6]])
    expected = x + x

    pfun, axis_name = _papply(np.add)
    ans = soft_pmap(pfun, axis_name)(x, x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testAddBroadcasting(self):
    raise SkipTest("test doesn't pass yet")  # TODO(frostig)

    def fun(x):
      return x + 3

    x = onp.array([[1, 2], [3, 4]])
    expected = x + 3

    pfun, axis_name = _papply(fun)
    ans = soft_pmap(pfun, axis_name)(x)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testMakeJaxprPapplyComposition(self):
    raise SkipTest(             # TODO(mattjj)
        "fails because select's papply rule calls an SPMD primitive")
    x = b = onp.ones(3)
    pfun, axis_name = _papply(lambda a: np.where(x, a, b))
    make_jaxpr(pfun)(onp.ones(3))  # doesn't crash


class ParallelizeTest(jtu.JaxTestCase):

  def dedup(self, arr, expected_rank):
    if arr.ndim == expected_rank + 1:
      for i in range(arr.shape[0] - 1):
        self.assertAllClose(arr[i], arr[i + 1], check_dtypes=True)
      return arr[0]
    else:
      assert arr.ndim == expected_rank
      return arr

  def testNormalize(self):

    def f(x):
      return x / x.sum(0)

    x = onp.arange(4.)
    expected = f(x)
    ans = _parallelize(f)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    jaxpr = make_jaxpr(_parallelize(f))(x)
    self.assertIn('psum', repr(jaxpr))

  def testAdd(self):
    raise SkipTest("buggy")  # TODO(mattjj): fix
    x = onp.arange(10)
    y = 2 * onp.arange(10)
    def f(x): return x + y
    expected = f(x)
    ans = _parallelize(f)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAdd2(self):
    raise SkipTest("buggy")  # TODO(mattjj): fix
    x = onp.arange(10)
    y = 2 * onp.arange(10)
    def f(y): return x + y
    expected = f(y)
    ans = _parallelize(f)(y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAdd3(self):
    x = onp.arange(10)
    y = 2 * onp.arange(10)
    def f(x, y):
      return x + y
    expected = f(x, y)
    ans = _parallelize(f)(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @unittest.skip("Missing cases in gather papply rule")
  def testOuter(self):
    x = onp.arange(10)
    y = 2 * onp.arange(10)
    def f(x): return x[:, None] * y
    expected = f(x)
    ans = _parallelize(f)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testOuter2(self):
    x = onp.arange(10)
    y = 2 * onp.arange(10)
    def f(y): return x[:, None] * y
    expected = f(y)
    ans = _parallelize(f)(y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @unittest.skip("Missing cases in gather papply rule")
  def testOuter3(self):
    x = onp.arange(10)
    y = 2 * onp.arange(10)
    def f(x, y): return x[:, None] * y
    expected = f(x, y)
    ans = _parallelize(f)(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "testTranspose_shape={}_perm={}"
       .format(shape, perm),
       "shape": shape, "perm": perm}
      for shape in [
          (2, 2),
          (3, 3),
          (2, 2, 2),
          (2, 3, 4),
          (2, 3, 2)
      ]
      for perm in itertools.permutations(list(range(len(shape))))
  ))
  def testTranspose(self, shape, perm):

    def fun(x):
      return lax.transpose(x, perm)

    x = onp.arange(prod(shape)).reshape(shape)
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeAndAddRank2(self):

    def fun(x):
      return x + x.T

    x = onp.reshape(onp.arange(4., dtype=onp.float32), (2, 2))
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeAndAddRank3(self):

    def fun(x):
      return x + x.T

    x = onp.reshape(onp.arange(8., dtype=onp.float32), (2, 2, 2))
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot(self):
    raise SkipTest("known failure")  # TODO(frostig)
    x = onp.reshape(onp.arange(4., dtype=onp.float32), (2, 2))

    def fun(x, y):
      return lax.dot(x, y)

    expected = fun(x, x)
    pfun, axis_name = _papply(fun)
    ans = soft_pmap(pfun, axis_name)(x, x)
    ans = self.dedup(ans, expected.ndim)
    self.assertAllClose(ans, expected, check_dtypes=False)

  # Test lax.dot_general on two rank-3 arguments, generating a test method call
  # for every matching of dimensions, and each matched pair of dimensions being
  # {batch, contracting, neither}. In combination with that, split the first
  # dimension of the LHS, that of the RHS, and that of both.
  @parameterized.named_parameters(
      {"testcase_name": "_dimMatch={}_matchTypes={}_split={}".format(
          matching, coloring, split),
       "matching": matching, "coloring": coloring, "split": split}
      for matching in itertools.permutations(range(3))
      for coloring in itertools.product(range(3), range(3), range(3))
      for split in range(3))
  def testDotGeneral(self, matching, coloring, split):
    BATCH, CONTRACT, _ = range(3)
    SPLIT_LHS, SPLIT_RHS, SPLIT_BOTH = range(3)

    x = onp.reshape(onp.arange(8.), (2, 2, 2))
    y = onp.reshape(onp.arange(8.), (2, 2, 2)) + 4.

    cdims = [(i, matching[i]) for i in range(3) if coloring[i] == CONTRACT]
    bdims = [(i, matching[i]) for i in range(3) if coloring[i] == BATCH]
    dimension_numbers = [
        list(zip(*cdims)) or [(), ()],
        list(zip(*bdims)) or [(), ()]
    ]

    def f(x, y):
      return lax.dot_general(x, y, dimension_numbers)

    if split == SPLIT_LHS:
      fun = lambda x: f(x, y)
    elif split == SPLIT_RHS:
      fun = lambda y: f(x, y)
    else:
      fun = f

    try:
      if split != SPLIT_BOTH:
        expected = fun(x)
        pfun, axis_name = _papply(fun)
        ans = soft_pmap(pfun, axis_name)(x)
      else:
        expected = fun(x, y)
        pfun, axis_name = _papply(fun)
        ans = soft_pmap(pfun, axis_name)(x, y)
    except (NotImplementedError, TypeError) as e:
      raise SkipTest(e)

    ans = self.dedup(ans, expected.ndim)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCall(self):
    @jit
    def fun(x):
      return x

    x = onp.reshape(onp.arange(8., dtype=onp.float32), (2, 2, 2))
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
