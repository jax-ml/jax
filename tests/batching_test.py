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


import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as np
from jax import test_util as jtu
from jax import lax
from jax import lax_linalg
from jax import random
from jax.api import jit, grad, jvp, vjp, make_jaxpr, jacfwd, jacrev, hessian
from jax.api import vmap
from jax.util import partial, curry
import jax.ops

from jax.config import config
config.parse_flags_with_absl()


# These are 'manual' tests for batching (vmap). The more exhaustive, more
# systematic tests are in lax_test.py's LaxVmapTest class.

class BatchingTest(jtu.JaxTestCase):

  def testConstantFunction(self):
    ans = vmap(lambda x: 3)(onp.ones(4))
    expected = 3 * onp.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNestedBatchingMatMat(self):
    matvec = vmap(np.vdot, in_axes=(0, None))
    matmat = vmap(matvec, in_axes=(None, 1), out_axes=1)

    R = onp.random.RandomState(0).randn
    A = R(4, 3)
    B = R(3, 2)

    ans = matmat(A, B)
    expected = onp.dot(A, B)
    self.assertAllClose(
        ans, expected, check_dtypes=False,
        rtol={onp.float32:1e-2} if jtu.device_under_test() == "tpu" else None)

    jaxpr = make_jaxpr(matmat)(A, B)
    self.assertEqual(len(jaxpr.jaxpr.eqns), 1)

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

    ans = vmap(partial(grad(loss), params))(batch)

    for ans_pair, param_pair in zip(ans, params):
      dW, db = ans_pair
      W, b = param_pair

      self.assertEqual(dW.shape, (batch_size,) + W.shape)
      self.assertEqual(db.shape, (batch_size,) + b.shape)

  def testJacobians(self):
    def jacbwd(f, x):
      y, pullback = vjp(f, x)
      std_basis = onp.eye(onp.size(y)).reshape((-1,) + onp.shape(y))
      jac_flat, = vmap(pullback, out_axes=onp.ndim(y))(std_basis)
      return jac_flat.reshape(onp.shape(y) + onp.shape(x))

    def jacfwd(f, x):
      pushfwd = lambda v: jvp(f, (x,), (v,))
      std_basis = onp.eye(onp.size(x)).reshape((-1,) + onp.shape(x))
      y, jac_flat = vmap(pushfwd, out_axes=(None, 0))(std_basis)
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

    g = jit(vmap(f))
    self.assertAllClose(g(onp.ones(2)), 2 * onp.ones(2), check_dtypes=False)
    self.assertEqual(len(side), 1)
    self.assertAllClose(g(2 * onp.ones(2)), 4 * onp.ones(2),
                        check_dtypes=False)
    self.assertEqual(len(side), 1)

  def testSliceLax(self):
    fun = lambda x: lax.slice(x, (2,), (4,))
    R = onp.random.RandomState(0).randn
    x = R(5, 10)

    ans = vmap(fun)(x)
    expected_ans = x[:, 2:4]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testSliceNumpy(self):
    fun = lambda x: x[:, 2]
    R = onp.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun)(x)
    expected_ans = x[:, :, 2]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testRevLax(self):
    fun = lambda x: lax.rev(x, [0])
    R = onp.random.RandomState(0).randn
    x = R(2, 3)

    ans = vmap(fun)(x)
    expected_ans = x[:, ::-1]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    ans = vmap(fun, (1,), 1)(x)
    expected_ans = x[::-1, :]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testRevNumpy(self):
    fun = lambda x: x[:, ::-1]
    R = onp.random.RandomState(0).randn
    x = R(3, 2, 4)

    ans = vmap(fun)(x)
    expected_ans = x[:, :, ::-1]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    ans = vmap(fun, (1,), 1)(x)
    expected_ans = x[:, :, ::-1]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    ans = vmap(fun, (2,), 2)(x)
    expected_ans = x[:, ::-1, :]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testNpMaximum(self):
    fun = lambda x: np.maximum(x, 0.0)
    R = onp.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun)(x)
    expected_ans = onp.maximum(x, 0.0)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testNpGtrThan(self):
    R = onp.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(lambda x: x > 1.0)(x)
    expected_ans = x > 1.0
    self.assertAllClose(ans, expected_ans, check_dtypes=True)

  def testNpMaximumPerExampleGrad(self):
    R = onp.random.RandomState(0).randn
    x = R(10, 5)
    W = R(5, 5)

    fun = lambda W, x: np.sum(np.maximum(np.dot(x, W), 0.0) ** 2)

    ans = vmap(partial(grad(fun), W))(x)

    W_t = np.transpose(W)
    for i in range(10):
      x_ex = x[i:i + 1]

      expected_ans = 2.0 * np.dot(
          np.maximum(np.dot(W_t, np.transpose(x_ex)), 0.0), x_ex)
      expected_ans = np.transpose(expected_ans)

      self.assertAllClose(
          ans[i], expected_ans, check_dtypes=False,
          atol={onp.float32:5e-2} if jtu.device_under_test() == "tpu" else None)

  def testDotGeneral(self):
    R = onp.random.RandomState(0).randn

    x = R(10, 3, 4, 5)
    y = R(10, 3, 5, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun)(x, y)
    expected = lax.dot_general(x, y, [((3,), (2,)), ((0, 1), (0, 1))])
    self.assertAllClose(ans, expected, check_dtypes=True)

    x = R(3, 4, 10, 5)
    y = R(3, 10, 5, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(2, 1))(x, y)
    expected = onp.stack([fun(x[..., i, :], y[:, i, ...]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    x = R(3, 4, 5, 10)
    y = R(3, 5, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(3, None))(x, y)
    expected = onp.stack([fun(x[..., i], y) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    x = R(3, 4, 5)
    y = R(3, 5, 10, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(None, 2))(x, y)
    expected = onp.stack([fun(x, y[..., i, :]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    x = R(4)
    y = R(4, 10)
    fun = lambda x, y: lax.dot_general(x, y, [((0,), (0,)), ((), ())])
    ans = vmap(fun, in_axes=(None, 1))(x, y)
    expected = onp.stack([fun(x, y[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testDot(self):
    # these tests are based on @shoyer's notebook studying gufuncs

    def vecvec(a, b):
      dot = np.dot
      for ndim in range(1, max(a.ndim, b.ndim)):
        a_ax = 0 if a.ndim > ndim else None
        b_ax = 0 if b.ndim > ndim else None
        dot = vmap(dot, in_axes=(a_ax, b_ax))
      return dot(a, b)

    assert vecvec(np.zeros((3,)), np.zeros((3,))).shape == ()
    assert vecvec(np.zeros((2, 3)), np.zeros((3,))).shape == (2,)
    assert vecvec(np.zeros((4, 2, 3)), np.zeros((3,))).shape == (4, 2)

  def testDot2(self):
    R = onp.random.RandomState(0).randn
    xs = R(10, 3)
    ys = R(10, 3)
    ans = vmap(np.dot)(xs, ys)
    expected = onp.einsum('ni,ni->n', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot3(self):
    R = onp.random.RandomState(0).randn
    xs = R(5, 8, 10)
    ys = R(10, 1)
    ans = vmap(np.dot, in_axes=(1, None))(xs, ys)
    expected = onp.einsum('inj,jk->nik', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot4(self):
    R = onp.random.RandomState(0).randn
    xs = R(3, 2)
    ys = R(3)
    ans = vmap(np.dot, in_axes=(1, None))(xs, ys)
    expected = onp.einsum('ij,i->j', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot5(self):
    f = vmap(partial(np.einsum, 'ij,j->i'), (None, 0))
    jaxpr = make_jaxpr(f)(np.zeros((1000, 1000)), np.zeros((1000, 1000)))
    assert "broadcast" not in str(jaxpr)

  def testPad(self):
    R = onp.random.RandomState(0).randn

    fun = lambda x: lax.pad(x, onp.float32(0), [(1, 2, 1)])
    x = R(5, 10).astype(onp.float32)
    ans = vmap(fun)(x)
    expected_ans = np.stack(list(map(fun, x)))
    self.assertAllClose(ans, expected_ans, check_dtypes=False)


    fun = lambda x: lax.pad(x, onp.float32(0), [(1, 2, 1), (0, 1, 0)])
    x = R(5, 10, 3).astype(onp.float32)
    ans = vmap(fun)(x)
    expected_ans = np.stack(list(map(fun, x)))
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testConcatenate(self):
    R = lambda *shape: onp.random.RandomState(0).randn(*shape).astype(onp.float32)

    fun = lambda *args: lax.concatenate(args, dimension=0)
    x, y, z = R(10, 2, 3), R(1, 10, 3), R(4, 3)
    ans = vmap(fun, in_axes=(0, 1, None))(x, y, z)
    expected_ans = onp.concatenate([x, onp.swapaxes(y, 0, 1),
                                    onp.broadcast_to(z, (10, 4, 3))], 1)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    fun = lambda *args: lax.concatenate(args, dimension=1)
    x, y, z = R(10, 2, 1), R(2, 3), R(2, 4, 10)
    ans = vmap(fun, in_axes=(0, None, 2))(x, y, z)
    expected_ans = onp.concatenate([x, onp.broadcast_to(y, (10, 2, 3)),
                                    onp.moveaxis(z, 2, 0)], 2)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testJacobianIssue54(self):
    # test modeling the code in https://github.com/google/jax/issues/54

    def func(xs):
      return np.array([x for x in xs])

    xs = np.ones((5, 1))
    jacrev(func)(xs)  # don't crash
    jacfwd(func)(xs)  # don't crash

  def testAny(self):
    # test modeling the code in https://github.com/google/jax/issues/108

    ans = vmap(np.any)(np.array([[True, False], [False, False]]))
    expected = np.array([True, False])
    self.assertAllClose(ans, expected, check_dtypes=True)

  @jtu.skip_on_devices("tpu")
  def testHessian(self):
    # test based on code from sindhwani@google
    def fun(x, t):
      return np.sum(np.power(np.maximum(x, 0.0), 2)) + t

    x = onp.array([-1., -0.5, 0., 0.5, 1.0])

    ans = hessian(lambda x: fun(x, 0.0))(x)
    expected = onp.array([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0.,0.5, 0., 0.],
                          [0., 0., 0., 2., 0.],
                          [0., 0., 0., 0., 2.]])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDynamicSlice(self):
    # test dynamic_slice via numpy indexing syntax
    # see https://github.com/google/jax/issues/1613 for an explanation of why we
    # need to use np rather than onp to create x and idx
    x = np.arange(30).reshape((10, 3))

    ans = vmap(lambda x, i: x[i], in_axes=(0, None))(x, 1)
    expected = x[:, 1]
    self.assertAllClose(ans, expected, check_dtypes=False)


    idx = np.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, i: x[i], in_axes=(0, 0))(x, idx)
    expected = x[onp.arange(10), idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = np.arange(3)
    idx = np.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, i: x[i], in_axes=(None, 0))(x, idx)
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDynamicUpdateSlice(self):
    x = onp.random.randn(10, 3)
    y = onp.random.randn(10)
    ans = vmap(lambda x, y, i: lax.dynamic_update_index_in_dim(x, y, i, axis=0),
               in_axes=(0, 0, None))(x, y, 1)
    expected = x.copy()
    expected[:, 1] = y
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = onp.random.randn(3)
    idx = onp.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, y, i: lax.dynamic_update_index_in_dim(x, y, i, axis=0),
               in_axes=(None, 0, 0))(x, y, idx)
    expected = onp.broadcast_to(x, (10, 3)).copy()
    expected[onp.arange(10), idx] = y
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testRandom(self):
    seeds = vmap(random.PRNGKey)(onp.arange(10))
    ans = vmap(partial(random.normal, shape=(3, 2)))(seeds)
    expected = onp.stack([random.normal(random.PRNGKey(seed), (3, 2))
                          for seed in onp.arange(10)])
    self.assertAllClose(ans, expected, check_dtypes=False)
    assert len(onp.unique(ans)) == 10 * 3 * 2

  def testSort(self):
    v = onp.arange(12)[::-1].reshape(3, 4)

    sv = vmap(partial(lax.sort, dimension=0), (0,))(v)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sv = vmap(partial(lax.sort, dimension=-1), (0,))(v)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sv = vmap(partial(lax.sort, dimension=0), (1,))(v)
    self.assertAllClose(sv, v[::-1, :].T, check_dtypes=True)

    sv = vmap(partial(lax.sort, dimension=0), (1,), 1)(v)
    self.assertAllClose(sv, v[::-1, :], check_dtypes=True)

  def testSortKeyVal(self):
    k = onp.arange(12)[::-1].reshape(3, 4)
    v = onp.random.RandomState(0).permutation(12).reshape(3, 4)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (0, 0))(k, v)
    self.assertAllClose(sk, k[:, ::-1], check_dtypes=True)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (1, 1), 1)(k, v)
    self.assertAllClose(sk, k[::-1, :], check_dtypes=True)
    self.assertAllClose(sv, v[::-1, :], check_dtypes=True)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (0, 1))(k, v.T)
    self.assertAllClose(sk, k[:, ::-1], check_dtypes=True)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (1, 0))(k.T, v)
    self.assertAllClose(sk, k[:, ::-1], check_dtypes=True)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (None, 0))(k[0], v)
    self.assertAllClose(sk, onp.broadcast_to(k[0, ::-1], (3, 4)),
                        check_dtypes=True)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (1, None))(k.T, v[0])
    self.assertAllClose(sk, k[:, ::-1], check_dtypes=True)
    self.assertAllClose(sv, onp.broadcast_to(v[0, ::-1], (3, 4)),
                        check_dtypes=True)

  def testConvGeneralDilated(self):
    W = np.array(onp.random.randn(3, 3, 1, 5), dtype=onp.float32)
    X = np.array(onp.random.randn(10, 5, 5, 1), dtype=onp.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      return y
    grad_loss = grad(lambda params, x: np.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(np.reshape(X, (10, 1, 5, 5, 1)))
    per_example = np.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(np.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, np.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          np.reshape(g, (1,) + g.shape)]
    per_example_direct = np.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True,
                        rtol=2e-2)

  def testConvGeneralDilatedBatchNotMajor(self):
    W = np.array(onp.random.randn(3, 3, 1, 4), dtype=onp.float32)
    x = np.array(onp.random.randn(3, 5, 7, 5, 1), dtype=onp.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('HNWC', 'HWIO', 'HWNC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      return y

    per_example = vmap(partial(f, W))(x)
    per_example = np.reshape(np.transpose(per_example, (1, 2, 0, 3, 4)),
                             (5, 5, 21, 4))
    per_example_direct = f(W, np.reshape(np.transpose(x, (1, 0, 2, 3, 4)),
                                         (5, 21, 5, 1)))
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

  @parameterized.named_parameters(
    {"testcase_name": "_op={}".format(name), "op": op, "unit": unit}
    for name, op, unit in [("max", lax.max, -np.inf), ("min", lax.min, np.inf)])
  def testMinMaxPool(self, op, unit):
    W = np.array(onp.random.randn(3, 3, 1, 5), dtype=onp.float32)
    X = np.array(onp.random.randn(10, 5, 5, 1), dtype=onp.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      y = lax.reduce_window(
          y, unit, op, (1, 2, 2, 1), (1, 1, 1, 1), 'SAME')
      return y
    grad_loss = grad(lambda params, x: np.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(np.reshape(X, (10, 1, 5, 5, 1)))
    per_example = np.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(np.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, np.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          np.reshape(g, (1,) + g.shape)]
    per_example_direct = np.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True,
                        rtol=5e-2)

  def testSumPool(self):
    W = np.array(onp.random.randn(3, 3, 1, 5), dtype=onp.float32)
    X = np.array(onp.random.randn(10, 5, 5, 1), dtype=onp.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      y = lax.reduce_window(
          y, 0.0, lax.add, (1, 2, 2, 1), (1, 1, 1, 1), 'SAME')
      return y
    grad_loss = grad(lambda params, x: np.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(np.reshape(X, (10, 1, 5, 5, 1)))
    per_example = np.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(np.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, np.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          np.reshape(g, (1,) + g.shape)]
    per_example_direct = np.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True,
                        rtol=3e-2)

  def testCumProd(self):
   x = np.arange(9).reshape(3, 3) + 1
   y = vmap(lambda x: np.cumprod(x, axis=-1))(x)
   self.assertAllClose(onp.cumprod(x, axis=1, dtype=np.int_), y,
                       check_dtypes=True)

  def testSelect(self):
    pred = onp.array([True, False])
    on_true = onp.array([0, 1])
    on_false = onp.array([2, 3])
    ans = vmap(lax.select)(pred, on_true, on_false)
    expected = onp.array([0, 3])
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = onp.array([False, True])
    on_true = onp.array([0, 1])
    on_false = onp.array([2, 3])
    ans = vmap(lax.select, (0, None, None))(pred, on_true, on_false)
    expected = onp.array([[2, 3],
                          [0, 1]])
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = True
    on_true = onp.array([0, 1], onp.float32)
    on_false = onp.array(3, onp.float32)
    ans = vmap(lax.select, (None, 0, None))(pred, on_true, on_false)
    expected = onp.array([0, 1], onp.float32)
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = onp.array([False, True])
    on_true = onp.array([0, 1], onp.float32)
    on_false = onp.array(3, onp.float32)
    ans = vmap(lax.select, (0, 0, None))(pred, on_true, on_false)
    expected = onp.array([3, 1], onp.float32)
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = onp.array([False, True])
    on_true = onp.array([2], onp.float32)
    on_false = onp.array([[3, 4]], onp.float32)
    ans = vmap(lax.select, (0, None, 1), 1)(pred, on_true, on_false)
    expected = onp.array([[3, 2]], onp.float32)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testLaxLinalgCholesky(self):
    a = onp.random.RandomState(0).randn(10, 5, 5).astype(onp.float32)
    a = onp.matmul(a, onp.conj(onp.swapaxes(a, -1, -2)))

    ans = vmap(lax_linalg.cholesky)(a)
    expected = onp.linalg.cholesky(a)
    self.assertAllClose(ans, expected, check_dtypes=False, rtol=1e-4)

    b = onp.random.RandomState(0).randn(10, 5, 5).astype(onp.float32)
    b = onp.matmul(b, onp.conj(onp.swapaxes(b, -1, -2)))
    b_trans = onp.swapaxes(b, 0, 1)  # shape is (5, 10, 5)

    ans = vmap(lax_linalg.cholesky, in_axes=1, out_axes=0)(b_trans)
    expected = onp.linalg.cholesky(b)
    self.assertAllClose(ans, expected, check_dtypes=False, rtol=1e-4)

  def testLaxLinalgTriangularSolve(self):
    a = onp.random.RandomState(0).randn(4, 10, 4).astype(onp.float32)
    a += onp.eye(4, dtype=np.float32)[:, None, :]
    b = onp.random.RandomState(0).randn(5, 4, 10).astype(onp.float32)

    ans = vmap(lax_linalg.triangular_solve, in_axes=(1, 2))(a, b)
    expected = onp.stack(
      [lax_linalg.triangular_solve(a[:, i], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    ans = vmap(lax_linalg.triangular_solve, in_axes=(None, 2))(a[:, 0], b)
    expected = onp.stack(
      [lax_linalg.triangular_solve(a[:, 0], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    ans = vmap(lax_linalg.triangular_solve, in_axes=(1, None))(a, b[..., 0])
    expected = onp.stack(
      [lax_linalg.triangular_solve(a[:, i], b[..., 0]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [onp.float32, onp.int32]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (3, 5), onp.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          (1, (10, 3), onp.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          (1, (10, 3, 5), onp.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          (2, (10, 5, 3), onp.array([[0, 2], [1, 0]]),
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,),
             start_index_map=(0, 1)),
            (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherBatchedOperand(self, axis, shape, dtype, idxs, dnums,
                               slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory()
    rng_idx = rng_idx_factory()
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    ans = vmap(fun, (axis, None))(operand, idxs)
    expected = onp.stack([fun(operand[(slice(None),) * axis + (i,)], idxs)
                          for i in range(operand.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [onp.float32, onp.float64]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (3, 5), onp.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          (1, (10, 3), onp.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          (1, (10, 3, 5), onp.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          (2, (10, 5, 3), onp.array([[0, 2], [1, 0]]),
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,),
             start_index_map=(0, 1)),
            (1, 3)),      ]
      for rng_idx_factory in [partial(jtu.rand_int, max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherGradBatchedOperand(self, axis, shape, dtype, idxs, dnums,
                                   slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory()
    rng_idx = rng_idx_factory()
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: np.sum(np.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    ans = vmap(gfun, (axis, None))(operand, idxs)
    expected = onp.stack([gfun(operand[(slice(None),) * axis + (i,)], idxs)
                          for i in range(operand.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [onp.float32, onp.int32]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (5,), onp.array([[[0], [2]], [[1], [3]]]), lax.GatherDimensionNumbers(
              offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)), (1,)),
          (1, (10,), onp.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)), (2,)),
          (1, (10, 5), onp.array([[0, 2, 1], [0, 3, 3]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)), (1, 3)),
          (0, (10, 5), onp.array([[[0, 1], [2, 0]],
                                  [[1, 0], [2, 3]]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)), (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherBatchedIndices(self, axis, shape, dtype, idxs, dnums,
                               slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory()
    rng_idx = rng_idx_factory()
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    ans = vmap(fun, (None, axis))(operand, idxs)
    expected = onp.stack([fun(operand, idxs[(slice(None),) * axis + (i,)])
                          for i in range(idxs.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [onp.float32, onp.float64]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (5,), onp.array([[[0], [2]], [[1], [3]]]), lax.GatherDimensionNumbers(
              offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)), (1,)),
          (1, (10,), onp.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)), (2,)),
          (1, (10, 5), onp.array([[0, 2, 1], [0, 3, 3]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)), (1, 3)),
          (0, (10, 5), onp.array([[[0, 1], [2, 0]],
                                  [[1, 0], [2, 3]]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)), (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherGradBatchedIndices(self, axis, shape, dtype, idxs, dnums,
                                   slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory()
    rng_idx = rng_idx_factory()
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: np.sum(np.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    ans = vmap(gfun, (None, axis))(operand, idxs)
    expected = onp.stack([gfun(operand, idxs[(slice(None),) * axis + (i,)])
                          for i in range(idxs.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_op_axis={}_idxs_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), op_axis, idxs_axis, idxs,
          dnums, slice_sizes),
       "op_axis": op_axis, "idxs_axis": idxs_axis, "shape": shape, "dtype":
       dtype, "idxs": idxs, "dnums": dnums, "slice_sizes": slice_sizes,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in [onp.float32, onp.int32]
      for op_axis, idxs_axis, shape, idxs, dnums, slice_sizes in [
          (0, 0, (2, 5), onp.array([[[0], [2]], [[1], [3]]]),
           lax.GatherDimensionNumbers(
             offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1,)),
          (1, 1, (10, 2), onp.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
           (2,)),
          (0, 1, (2, 10, 5,), onp.array([[[0, 2, 1], [0, 3, 3]]]).T,
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1, 3)),
          (2, 0, (10, 5, 2), onp.array([[[0, 2], [1, 0]],
                                        [[1, 0], [2, 0]]]),
          lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
           (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherBatchedBoth(self, op_axis, idxs_axis, shape, dtype, idxs, dnums,
                            slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory()
    rng_idx = rng_idx_factory()
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    assert operand.shape[op_axis] == idxs.shape[idxs_axis]
    ans = vmap(fun, (op_axis, idxs_axis))(operand, idxs)
    expected = onp.stack([fun(operand[(slice(None),) * op_axis + (i,)],
                              idxs[(slice(None),) * idxs_axis + (i,)])
                          for i in range(idxs.shape[idxs_axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_op_axis={}_idxs_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), op_axis, idxs_axis, idxs,
          dnums, slice_sizes),
       "op_axis": op_axis, "idxs_axis": idxs_axis, "shape": shape, "dtype":
       dtype, "idxs": idxs, "dnums": dnums, "slice_sizes": slice_sizes,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in [onp.float32]
      for op_axis, idxs_axis, shape, idxs, dnums, slice_sizes in [
          (0, 0, (2, 5), onp.array([[[0], [2]], [[1], [3]]]),
           lax.GatherDimensionNumbers(
             offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1,)),
          (1, 1, (10, 2), onp.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
           (2,)),
          (0, 1, (2, 10, 5,), onp.array([[[0, 2, 1], [0, 3, 3]]]).T,
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1, 3)),
          (2, 0, (10, 5, 2), onp.array([[[0, 2], [1, 0]],
                                        [[1, 0], [2, 0]]]),
          lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
           (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherGradBatchedBoth(self, op_axis, idxs_axis, shape, dtype, idxs, dnums,
                                slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory()
    rng_idx = rng_idx_factory()
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: np.sum(np.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    assert operand.shape[op_axis] == idxs.shape[idxs_axis]
    ans = vmap(gfun, (op_axis, idxs_axis))(operand, idxs)
    expected = onp.stack([gfun(operand[(slice(None),) * op_axis + (i,)],
                              idxs[(slice(None),) * idxs_axis + (i,)])
                          for i in range(idxs.shape[idxs_axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNumpyIndexing1(self):
    a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    ind = onp.array([[0, 1],
                    [2, 0]])
    def f(a, ind):
      return a[:, ind]
    expected = onp.stack([f(a, ind[i, :]) for i in range(ind.shape[0])])
    ans = vmap(f, (None, 0))(a, ind)
    assert onp.all(ans == expected)

  def testNumpyIndexing2(self):
    a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    def f(a):
      inds = np.array([0, 2])
      return a[:, inds]
    ans = vmap(f)(a)
    expected = onp.stack([f(a[:, i, :]) for i in range(a.shape[1])], axis=1)
    assert onp.all(ans == expected)

  def testTranspose(self):
    x = onp.arange(4 * 3 * 3).reshape((4, 3, 3))
    ans = vmap(lambda x: x + x.T)(x)
    expected = x + onp.swapaxes(x, -1, -2)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposePermutation(self):
    x = onp.arange(6 * 3 * 4 * 5).reshape((6, 3, 4, 5))
    ans = vmap(lambda x: np.transpose(x, (1, 0, 2)))(x)
    expected = onp.transpose(x, (0, 2, 1, 3))
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = onp.arange(6 * 3 * 4 * 5).reshape((6, 3, 4, 5))
    ans = vmap(lambda x: np.transpose(x, (1, 2, 0)))(x)
    expected = onp.transpose(x, (0, 2, 3, 1))
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = onp.arange(6 * 3 * 4 * 5).reshape((3, 4, 6, 5))
    ans = vmap(lambda x: np.transpose(x, (1, 2, 0)), in_axes=2)(x)
    expected = onp.transpose(x, (2, 1, 3, 0))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testIssue354(self):
    psd_mat = onp.random.randn(20, 10)
    psd_mat = psd_mat.T.dot(psd_mat)
    vec = onp.random.randn(10)

    def f(scale):
      scaled_mat = scale * psd_mat
      chol = np.linalg.cholesky(scaled_mat)
      return -0.5 * np.sum((np.einsum('ij,j->i', chol, vec))**2)
    vmapped_f = vmap(f)
    vmapped_f_grad = grad(lambda x: np.sum(vmapped_f(x)))

    scales = onp.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    ans = vmapped_f_grad(scales)  # don't crash!
    expected = onp.stack([grad(f)(scale) for scale in scales])
    self.assertAllClose(ans, expected, check_dtypes=False,
                        rtol=jtu.default_gradient_tolerance)

  def testIssue387(self):
    # https://github.com/google/jax/issues/387
    R = onp.random.RandomState(0).rand(100, 2)

    def dist_sq(R):
      dR = R[:, np.newaxis, :] - R[np.newaxis, :, :]
      zero = np.zeros_like(dR)
      dR = dR - np.where(np.abs(dR) < 0.5, zero, 0.5 * np.sign(dR))
      return np.sum(dR ** 2, axis=2)

    @jit
    def f(R):
      dr = dist_sq(R)
      return np.sum(R ** 2)

    H = hessian(f)(R)  # don't crash on UnshapedArray

  def testIssue489(self):
    def f(key):
      def body_fn(uk):
        key = uk[1]
        u = random.uniform(key, (), dtype=np.float64)
        key, _ = random.split(key)
        return u, key

      u, _ = lax.while_loop(lambda uk: uk[0] > 0.5, body_fn,
                            (np.float64(1.), key))
      return u

    print(vmap(f)(random.split(random.PRNGKey(0), 2)))  # no crash

  def testEmptyTuples(self):
    # Ensure there is no crash when a vectorized input contains empty tuples.
    result = vmap(lambda x, _: x + 1)(onp.array([0, 1]), ())
    self.assertAllClose(result, onp.array([1, 2]), check_dtypes=False)
    # Ensure there is no crash when a vectorized output contains empty tuples.
    result, empty_tuple = vmap(lambda x: (x + 1, ()))(onp.array([0, 1]))
    self.assertAllClose(result, onp.array([1, 2]), check_dtypes=False)
    self.assertEqual((), empty_tuple)

  def testIndexAddBatchedIndexesOnly(self):
    f = lambda x, idx, y: jax.ops.index_add(x, jax.ops.index[idx], y)
    result = vmap(f, (None, 0, None))(onp.zeros((10,)), onp.arange(10,), 1.)
    self.assertAllClose(result, onp.eye(10), check_dtypes=False)

  def testIssue1170(self):
    def f(index1, index2):
      return np.arange(36).reshape(6, 6)[index1, index2]
    g = jax.jit(jax.pmap(f))
    ans = g(index1=onp.asarray([1]), index2=onp.asarray([2]))
    expected = g(onp.asarray([1]), onp.asarray([2]))
    self.assertAllClose(ans, expected, check_dtypes=True)


if __name__ == '__main__':
  absltest.main()
