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
from jax.abstract_arrays import ShapedArray
from jax import lax
from jax.api import jit, grad, jvp, vjp, trace_to_jaxpr
from jax.api import vmap
from jax.config import config
from jax.core import unit
from jax.interpreters import partial_eval as pe
from jax.util import partial

import functools as fn

class BatchingTest(jtu.JaxTestCase):

  def testConstantFunction(self):
    ans = vmap(lambda x: 3, onp.ones(4))
    expected = 3 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNestedBatchingMatMat(self):
    def matvec(A, b):
      return vmap(np.vdot, A, b, in_axes=(0, None))

    def matmat(A, B):
      return vmap(matvec, A, B, in_axes=(None, 1), out_axes=1)

    R = onp.random.RandomState(0).randn
    A = R(4, 3)
    B = R(3, 2)

    ans = matmat(A, B)
    expected = onp.dot(A, B)
    self.assertAllClose(ans, expected, check_dtypes=False)

    # this is a crude check that we only call a single dot
    def pv_like(x):
      aval = ShapedArray(onp.shape(x), onp.result_type(x))
      return pe.PartialVal((aval, unit))

    def make_jaxpr(fun, example_args):
      jaxpr, _, _, _ = trace_to_jaxpr(fun, map(pv_like, example_args))
      return jaxpr

    jaxpr = make_jaxpr(matmat, (A, B))
    self.assertEqual(len(jaxpr.eqns), 1)

  def testPerExampleGradients(self):
    def predict(params, inputs):
      for W, b in params:
        outputs = np.dot(W, inputs) + b
        inputs = np.tanh(outputs)
      return outputs

    def loss(params, data):
      inputs, targets = data
      predictions = predict(params, inputs)
      return np.sum((predictions - targets)**2)

    batch_size = 5
    layer_sizes = [3, 2, 4]

    R = onp.random.RandomState(0).randn
    params = [(R(m, n), R(m))
              for m, n in zip(layer_sizes[1:], layer_sizes[:-1])]

    input_vec = R(3)
    target_vec = R(4)
    datum = (input_vec, target_vec)

    input_batch = R(5, 3)
    target_batch = R(5, 4)
    batch = (input_batch, target_batch)

    ans = vmap(partial(grad(loss), params), batch)

    for ans_pair, param_pair in zip(ans, params):
      dW, db = ans_pair
      W, b = param_pair

      self.assertEqual(dW.shape, (batch_size,) + W.shape)
      self.assertEqual(db.shape, (batch_size,) + b.shape)

  def testJacobians(self):
    def jacbwd(f, x):
      y, pullback = vjp(f, x)
      std_basis = onp.eye(onp.size(y)).reshape((-1,) + onp.shape(y))
      jac_flat, = vmap(pullback, std_basis, out_axes=onp.ndim(y))
      return jac_flat.reshape(onp.shape(y) + onp.shape(x))

    def jacfwd(f, x):
      pushfwd = lambda v: jvp(f, (x,), (v,))
      std_basis = onp.eye(onp.size(x)).reshape((-1,) + onp.shape(x))
      y, jac_flat = vmap(pushfwd, std_basis, out_axes=(None, 0))
      return jac_flat.reshape(onp.shape(y) + onp.shape(x))

    R = onp.random.RandomState(0).randn

    A = R(4, 3)
    b = R(4)
    f = lambda x: np.tanh(np.dot(A, x) + b)

    x = R(3)
    self.assertAllClose(jacfwd(f, x), jacbwd(f, x), check_dtypes=False)

  def testBatchOfCompile(self):
    side = []

    @jit
    def f(x):
      side.append(None)
      return x + x

    g = jit(lambda x: vmap(f, x))
    self.assertAllClose(g(onp.ones(2)), 2 * onp.ones(2), check_dtypes=False)
    self.assertEqual(len(side), 1)
    self.assertAllClose(g(2 * onp.ones(2)), 4 * onp.ones(2),
                        check_dtypes=False)
    self.assertEqual(len(side), 1)

  def testSliceLax(self):
    fun = lambda x: lax.slice(x, (2,), (4,))
    R = onp.random.RandomState(0).randn
    x = R(5, 10)

    ans = vmap(fun, x)
    expected_ans = x[:, 2:4]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testSliceNumpy(self):
    fun = lambda x: x[:, 2]
    R = onp.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun, x)
    expected_ans = x[:, :, 2]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testNpMaximum(self):
    fun = lambda x: np.maximum(x, 0.0)
    R = onp.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun, x)
    expected_ans = onp.maximum(x, 0.0)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testNpGtrThan(self):
    R = onp.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(lambda x: x > 1.0, x)
    expected_ans = x > 1.0
    self.assertAllClose(ans, expected_ans, check_dtypes=True)

  def testNpMaximumPerExampleGrad(self):
    R = onp.random.RandomState(0).randn
    x = R(10, 5)
    W = R(5, 5)

    fun = lambda W, x: np.sum(np.maximum(np.dot(x, W), 0.0) ** 2)

    ans = vmap(fn.partial(grad(fun), W), x)

    W_t = np.transpose(W)
    for i in range(10):
      x_ex = x[i:i + 1]

      expected_ans = 2.0 * np.dot(
          np.maximum(np.dot(W_t, np.transpose(x_ex)), 0.0), x_ex)
      expected_ans = np.transpose(expected_ans)

      self.assertAllClose(ans[i], expected_ans, check_dtypes=False)

  def testDotGeneral(self):
    R = onp.random.RandomState(0).randn
 
    x = R(10, 3, 4, 5)
    y = R(10, 3, 5, 6)
    ans = vmap(lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))]), x, y)
    expected = lax.dot_general(x, y, [((3,), (2,)), ((0, 1), (0, 1))])
    self.assertAllClose(ans, expected, check_dtypes=True)
 
    x = R(3, 4, 10, 5)
    y = R(3, 10, 5, 6)
    ans = vmap(lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))]), x, y,
               in_axes=(2, 1))
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    expected = onp.stack([fun(x[..., i, :], y[:, i, ...]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)
 
    x = R(3, 4, 5, 10)
    y = R(3, 5, 6)
    ans = vmap(lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))]), x, y,
               in_axes=(3, None))
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    expected = onp.stack([fun(x[..., i], y) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)
 
    x = R(3, 4, 5)
    y = R(3, 5, 10, 6)
    ans = vmap(lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))]), x, y,
               in_axes=(None, 2))
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    expected = onp.stack([fun(x, y[..., i, :]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
