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

import collections
from functools import partial

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr

from jax import api
from jax import core
from jax import lax
from jax import test_util as jtu
import jax.numpy as np  # scan tests use numpy

def scan_reference(f, init, xs):
  carry = init
  ys = []
  for x in xs:
    (carry, y) = f(carry, x)
    ys.append(lax.reshape(y, (1,) + onp.shape(y)))
  ys = lax.concatenate(ys, 0)
  return carry, ys


class LaxControlFlowTest(jtu.JaxTestCase):

  def testWhileWithTuple(self):
    limit = 10

    def loop_cond(state):
      pos, _ = state
      return lax.lt(pos, limit)

    def loop_body(state):
      pos, count = state
      return (lax.add(pos, 1), lax.add(count, 1))

    def loop(init):
      result = lax.while_loop(loop_cond, loop_body, (init, 0))
      _, count = result
      return count

    cloop = api.jit(loop)

    self.assertEqual(loop(2), limit - 2)
    self.assertEqual(cloop(2), limit - 2)
    self.assertEqual(cloop(2), limit - 2)
    self.assertEqual(cloop(3), limit - 3)

  def testNestedWhile(self):

    def outer_loop(num):  # pylint: disable=missing-docstring
      def cond_fun(state):
        num, i, _ = state
        return lax.lt(i, num)

      def body_fun(state):
        num, i, count = state
        return (num, lax.add(i, 1), inner_loop(i, count))

      init_val = (num, 0, 0)
      _, i, count = lax.while_loop(cond_fun, body_fun, init_val)
      return (i, count)

    def inner_loop(i, count):  # pylint: disable=missing-docstring
      def cond_fun(state):
        i, j, _ = state
        return lax.le(j, i)

      def body_fun(state):
        i, j, count = state
        return (i, lax.add(j, 1), lax.add(count, 1))

      init_val = (i, 0, count)
      _, _, count = lax.while_loop(cond_fun, body_fun, init_val)
      return count

    cloop = api.jit(outer_loop)

    self.assertEqual(outer_loop(3), (3, 6))
    self.assertEqual(cloop(3), (3, 6))
    self.assertEqual(cloop(3), (3, 6))
    self.assertEqual(cloop(2), (2, 3))
    self.assertEqual(cloop(4), (4, 10))

  def testWhileWithClosure(self):

    def loop(init, local_limit, inc):

      def loop_cond(state):
        pos, _ = state
        return lax.lt(pos, local_limit)

      def loop_body(state):
        effect[0] = True
        pos, count = state
        return (lax.add(pos, 1), lax.add(count, inc))

      result = lax.while_loop(loop_cond, loop_body, (init, 0))
      _, count = result
      return count

    cloop = api.jit(loop)

    limit = 10
    effect = [False]
    self.assertEqual(loop(2, limit, 1), limit - 2)
    assert effect[0]
    effect[0] = False
    self.assertEqual(cloop(2, limit, 1), limit - 2)
    assert effect[0]
    effect[0] = False
    self.assertEqual(cloop(2, limit, 1), limit - 2)
    self.assertEqual(cloop(3, limit, 1), limit - 3)
    assert not effect[0]

  def testWhileWithClosureJit(self):

    def loop(init, local_limit, inc):

      def loop_cond(state):
        pos, _ = state
        return lax.lt(pos, local_limit)

      def loop_body(state):
        effect[0] = True
        pos, count = state
        f = lambda pos, inc: (lax.add(pos, 1), lax.add(count, inc))
        return api.jit(f)(pos, inc)

      result = lax.while_loop(loop_cond, loop_body, (init, 0))
      _, count = result
      return count

    cloop = api.jit(loop)

    limit = 10
    effect = [False]
    self.assertEqual(loop(2, limit, 1), limit - 2)
    assert effect[0]
    effect[0] = False
    self.assertEqual(cloop(2, limit, 1), limit - 2)
    assert effect[0]
    effect[0] = False
    self.assertEqual(cloop(2, limit, 1), limit - 2)
    self.assertEqual(cloop(3, limit, 1), limit - 3)
    assert not effect[0]

  def testNestedWhileWithDynamicUpdateSlice(self):
    num = 5

    def update_entry(arr, val, i, j):
      val = lax.reshape(val, [1, 1])
      return lax.dynamic_update_slice(arr, val, (i, j))

    def outer_loop(arr):  # pylint: disable=missing-docstring

      def cond_fun(state):
        i, num, _, _ = state
        return lax.lt(i, num)

      def body_fun(state):
        i, num, arr, out = state
        return (lax.add(i, 1), num, arr, inner_loop(i, arr, out))

      out = onp.zeros(arr.shape, dtype=arr.dtype)
      init_val = (0, num, arr, out)
      _, _, _, out = lax.while_loop(cond_fun, body_fun, init_val)
      return out

    def inner_loop(i, arr, out):  # pylint: disable=missing-docstring

      def cond_fun(state):
        i, j, _, _ = state
        return lax.le(j, i)

      def body_fun(state):
        i, j, arr, out = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        arr_i_j = lax.dynamic_index_in_dim(arr_i, j, 0, False)
        out = update_entry(out, arr_i_j, i, j)
        return (i, lax.add(j, 1), arr, out)

      init_val = (i, 0, arr, out)
      _, _, _, out = lax.while_loop(cond_fun, body_fun, init_val)
      return out

    cloop = api.jit(outer_loop)
    arr = npr.RandomState(0).randn(5, 5)
    self.assertAllClose(outer_loop(arr), onp.tril(arr), check_dtypes=False)
    self.assertAllClose(cloop(arr), onp.tril(arr), check_dtypes=False)
    self.assertAllClose(cloop(arr), onp.tril(arr), check_dtypes=False)

  def testLoopWithConjunctionCondition(self):
    def sum_first_n(arr, num):  # pylint: disable=missing-docstring
      def cond_fun(state):
        arr, num, i, _ = state
        return lax.bitwise_and(lax.lt(i, num), lax.lt(i, arr.shape[0]))

      def body_fun(state):
        arr, num, i, total = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return (arr, num, lax.add(i, 1), lax.add(total, arr_i))

      init_val = (arr, num, 0, 0.)
      _, _, _, total = lax.while_loop(cond_fun, body_fun, init_val)
      return total

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  def testForiLoopBasic(self):
    def count(num):
      def body_fun(i, tot):
        return lax.add(tot, i)
      return lax.fori_loop(0, num, body_fun, 0)

    cfun = api.jit(count)

    self.assertEqual(count(2), 1)
    self.assertEqual(count(2), cfun(2))
    self.assertEqual(count(3), 3)
    self.assertEqual(count(3), cfun(3))
    self.assertEqual(count(4), 6)
    self.assertEqual(count(4), cfun(4))

  def testForiLoopClosure(self):
    def count(num):
      def body_fun(i, tot):
        return lax.add(num, lax.add(tot, i))
      return lax.fori_loop(0, num, body_fun, 0)

    cfun = api.jit(count)

    self.assertEqual(count(2), 1 + 2**2)
    self.assertEqual(count(2), cfun(2))
    self.assertEqual(count(3), 3 + 3**2)
    self.assertEqual(count(3), cfun(3))
    self.assertEqual(count(4), 6 + 4**2)
    self.assertEqual(count(4), cfun(4))

  def testForiLoopTupleState(self):
    def sum_first_n(arr, num):
      def body_fun(i, state):
        arr, total = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return (arr, lax.add(total, arr_i))

      init_val = (arr, 0.)
      _, total = lax.fori_loop(0, lax.min(arr.shape[0], num), body_fun,
                               init_val)
      return total

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  def testForiLoopDictState(self):
    def sum_first_n(arr, num):
      def body_fun(i, state):
        arr, total = state['arr'], state['total']
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return {'arr': arr, 'total': lax.add(total, arr_i)}

      init_val = {'arr': arr, 'total': 0.}
      out_val = lax.fori_loop(0, lax.min(arr.shape[0], num), body_fun, init_val)
      return out_val['total']

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  def testForiLoopEmptyTupleInState(self):
    def sum_first_n(arr, num):
      def body_fun(i, state):
        arr, total, _ = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return (arr, lax.add(total, arr_i), ())

      init_val = (arr, 0., ())
      _, tot, _ = lax.fori_loop(0, lax.min(arr.shape[0], num), body_fun, init_val)
      return tot

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  def testCond(self):
    def fun(x):
      if x < 3:
        return (x, x)
      else:
        y = lax.mul(2, x)
        return y, lax.mul(2, y)

    @api.jit
    def cfun(x):
      def false_fun(x):
        y = lax.mul(2, x)
        return y, lax.mul(2, y)
      return lax.cond(lax.lt(x, 3), x, lambda x: (x, x), x, false_fun)

    self.assertEqual(fun(0), cfun(0))
    self.assertEqual(fun(0), (0, 0))
    self.assertEqual(fun(1), cfun(1))
    self.assertEqual(fun(1), (1, 1))
    self.assertEqual(fun(2), cfun(2))
    self.assertEqual(fun(2), (2, 2))
    self.assertEqual(fun(3), cfun(3))
    self.assertEqual(fun(3), (6, 12))
    self.assertEqual(fun(4), cfun(4))
    self.assertEqual(fun(4), (8, 16))

  def testNestedCond(self):
    def fun(x):
      if x < 2:
        return lax.mul(2, x)
      else:
        if x < 5:
          return lax.mul(3, x)
        else:
          return lax.mul(4, x)

    @api.jit
    def cfun(x):
      return lax.cond(
          lax.lt(x, 2),
          x, lambda x: lax.mul(2, x),
          x, lambda x: lax.cond(lax.lt(x, 5),
                                x, lambda x: lax.mul(3, x),
                                4, lambda y: lax.mul(y, x)))

    self.assertEqual(cfun(1), 2)
    self.assertEqual(cfun(3), 9)
    self.assertEqual(cfun(6), 24)
    self.assertEqual(cfun(1), fun(1))
    self.assertEqual(cfun(3), fun(3))
    self.assertEqual(cfun(6), fun(6))

  def testCondOneBranchConstant(self):
    def fun(x):
      if x < 3:
        return 5.
      else:
        return x

    @api.jit
    def cfun(x):
      return lax.cond(lax.lt(x, 3), x, lambda x: 5, x, lambda x: x)

    self.assertEqual(fun(0), cfun(0))
    self.assertEqual(cfun(0), 5)
    self.assertEqual(fun(4), cfun(4))
    self.assertEqual(cfun(4), 4)

  def testCondOneBranchConstantTuple(self):
    def fun(x):
      if x < 3:
        return (1., 2., 3.)
      else:
        return (x, 2., 4.)

    @api.jit
    def cfun(x):
      return lax.cond(lax.lt(x, 3),
                      x, lambda x: (1, 2., 3.),
                      x, lambda x: (x, 2., 4.))

    self.assertEqual(fun(0), cfun(0))
    self.assertEqual(cfun(0), (1, 2., 3.))
    self.assertEqual(fun(4), cfun(4))
    self.assertEqual(cfun(4), (4, 2., 4.))

  def testIssue514(self):
    # just check this doesn't crash
    lax.cond(True,
            (0, 0), lambda x: (x[0], 0),
            (1, 1), lambda x: x)

  def testIssue649(self):
    from jax import lax

    def body(x):
      a, b = x
      return (7, b + 1)

    def cond(x):
      a, b = x
      return b < 10

    out = lax.while_loop(cond, body, (33, 4))
    self.assertEqual(out, (7, 10))

  @parameterized.named_parameters(
      {"testcase_name": "jit_scan={}_jit_f={}".format(jit_scan, jit_f),
       "jit_scan": jit_scan, "jit_f": jit_f}
      for jit_scan in [False, True]
      for jit_f in [False, True])
  def testScanImpl(self, jit_scan, jit_f):
    d = np.zeros(2)
    def f(c, a):
      assert a.shape == (3,)
      assert c.shape == (4,)
      b = np.sum(np.sin(a)) + np.sum(np.sin(c)) + np.sum(np.sin(d))
      c = np.sin(c * b)
      assert b.shape == ()
      return c, b

    if jit_f:
      f = api.jit(f)
    if jit_scan:
      scan = api.jit(lax.scan, (0,))
    else:
      scan = lax.scan

    as_ = np.ones((5, 3))
    c = np.ones(4)

    ans =                scan(f, c, as_)
    expected = scan_reference(f, c, as_)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "jit_scan={}_jit_f={}".format(jit_scan, jit_f),
       "jit_scan": jit_scan, "jit_f": jit_f}
      for jit_scan in [False, True]
      for jit_f in [False, True])
  def testScanJVP(self, jit_scan, jit_f):
    d = np.zeros(2)
    def f(c, a):
      assert a.shape == (3,)
      assert c.shape == (4,)
      b = np.sum(np.sin(a)) + np.sum(np.sin(c)) + np.sum(np.sin(d))
      c = np.sin(c * b)
      assert b.shape == ()
      return c, b

    if jit_f:
      f = api.jit(f)
    if jit_scan:
      scan = api.jit(lax.scan, (0,))
    else:
      scan = lax.scan

    as_ = np.ones((5, 3))
    c = np.ones(4)

    ans = api.jvp(lambda c, as_:                scan(f, c, as_), (c, as_), (c, as_))[1]
    expected = api.jvp(lambda c, as_: scan_reference(f, c, as_), (c, as_), (c, as_))[1]
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "jit_scan={}_jit_f={}".format(jit_scan, jit_f),
       "jit_scan": jit_scan, "jit_f": jit_f}
      for jit_scan in [False, True]
      for jit_f in [False, True])
  def testScanLinearize(self, jit_scan, jit_f):
    d = np.zeros(2)
    def f(c, a):
      assert a.shape == (3,)
      assert c.shape == (4,)
      b = np.sum(np.sin(a)) + np.sum(np.sin(c)) + np.sum(np.sin(d))
      c = np.sin(c * b)
      assert b.shape == ()
      return c, b

    if jit_f:
      f = api.jit(f)
    if jit_scan:
      scan = api.jit(lax.scan, (0,))
    else:
      scan = lax.scan

    as_ = np.ones((5, 3))
    c = np.ones(4)

    ans = api.linearize(lambda c, as_:                scan(f, c, as_), c, as_)[1](c, as_)
    expected = api.linearize(lambda c, as_: scan_reference(f, c, as_), c, as_)[1](c, as_)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "jit_scan={}_jit_f={}".format(jit_scan, jit_f),
       "jit_scan": jit_scan, "jit_f": jit_f}
      for jit_scan in [False, True]
      for jit_f in [False, True])
  def testScanGrad(self, jit_scan, jit_f):
    d = np.zeros(2)
    def f(c, a):
      assert a.shape == (3,)
      assert c.shape == (4,)
      b = np.sum(np.sin(a)) + np.sum(np.sin(c)) + np.sum(np.sin(d))
      c = np.sin(c * b)
      assert b.shape == ()
      return c, b

    if jit_f:
      f = api.jit(f)
    if jit_scan:
      scan = api.jit(lax.scan, (0,))
    else:
      scan = lax.scan

    as_ = np.ones((5, 3))
    c = np.ones(4)

    ans = api.grad(lambda c, as_:      list(          scan(f, c, as_))[0].sum())(c, as_)
    expected = api.grad(lambda c, as_: list(scan_reference(f, c, as_))[0].sum())(c, as_)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testScanRnn(self):
    r = npr.RandomState(0)

    n_in = 4
    n_hid = 3
    n_out = 2
    length = 5

    W_trans = r.randn(n_hid, n_hid + n_in)
    W_out = r.randn(n_out, n_hid + n_in)
    params = W_trans, W_out

    inputs = r.randn(length, n_in)
    targets = r.randn(length, n_out)

    def step(params, state, input):
      W_trans, W_out = params
      stacked = np.concatenate([state, input])
      output = np.tanh(np.dot(W_out, stacked))
      next_state = np.tanh(np.dot(W_trans, stacked))
      return next_state, output

    def rnn(params, inputs):
      init_state = np.zeros(n_hid)
      _, outputs = lax.scan(partial(step, params), init_state, inputs)
      return outputs

    def loss(params, inputs, targets):
      predictions = rnn(params, inputs)
      return np.sum((predictions - targets)**2)

    # evaluation doesn't crash
    loss(params, inputs, targets)

    # jvp evaluation doesn't crash
    api.jvp(lambda params: loss(params, inputs, targets), (params,), (params,))

    # gradient evaluation doesn't crash
    api.grad(loss)(params, inputs, targets)

    # gradient is zero in the right place
    predictions = rnn(params, inputs)
    ans = api.grad(loss)(params, inputs, predictions)
    expected = (onp.zeros_like(W_trans), onp.zeros_like(W_out))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testIssue711(self):
    # Tests reverse-mode differentiation through a scan for which the scanned
    # function also involves reverse-mode differentiation.
    # See https://github.com/google/jax/issues/711
    def harmonic_bond(conf, params):
      return np.sum(conf * params)

    def minimize_structure(test_params):
      energy_fn = partial(harmonic_bond, params=test_params)
      grad_fn = api.grad(energy_fn)

      def apply_carry(carry, _):
        i, x = carry
        new_x = x - 0.1 * api.grad(energy_fn)(x)
        new_carry = (i+1, new_x)
        return new_carry, _

      x0 = np.array([1., 2., 3.])
      carry_final, _ = lax.scan(apply_carry, (0, x0), np.zeros((75, 0)))
      _, x_final = carry_final
      return x_final

    initial_params = 0.5
    minimize_structure(initial_params)  # doesn't crash

    def loss(test_params):
      x_final = minimize_structure(test_params)
      return np.sum(np.sin(1.0 - x_final))

    api.grad(loss)(0.25)  # doesn't crash

  def testIssue744(self):
    Point = collections.namedtuple('Point', ['x', 'y'])
    p0 = Point(x=np.array(1), y=np.array(2))

    def plus_one(p, iter_idx):
      return Point(p.x+1, p.y+1), iter_idx

    self.assertRaisesRegexp(
        ValueError,
        'scan got value with no leading axis to scan over.*',
        lambda: lax.scan(plus_one, p0, list(range(5))))

  def testScanHigherOrderDifferentiation(self):
    d = 0.75
    def f(c, a):
      b = np.sin(c * np.sum(np.cos(d * a)))
      c = 0.9 * np.cos(d * np.sum(np.sin(c * a)))
      return c, b

    as_ = np.arange(6.).reshape((3, 2))
    c = 1.

    jtu.check_grads(lambda c: lax.scan(f, c, as_), (c,), modes=["fwd", "rev"], order=2)


if __name__ == '__main__':
  absltest.main()
