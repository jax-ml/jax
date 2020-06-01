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


import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import test_util as jtu
from jax import lax
from jax import lax_linalg
from jax import random
from jax.api import jit, grad, jvp, vjp, make_jaxpr, jacfwd, jacrev, hessian
from jax.api import vmap
from jax.util import partial
import jax.ops

from jax.config import config
config.parse_flags_with_absl()


# These are 'manual' tests for batching (vmap). The more exhaustive, more
# systematic tests are in lax_test.py's LaxVmapTest class.

class BatchingTest(jtu.JaxTestCase):

  def testConstantFunction(self):
    ans = vmap(lambda x: 3)(np.ones(4))
    expected = 3 * np.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNestedBatchingMatMat(self):
    matvec = vmap(jnp.vdot, in_axes=(0, None))
    matmat = vmap(matvec, in_axes=(None, 1), out_axes=1)

    R = np.random.RandomState(0).randn
    A = R(4, 3)
    B = R(3, 2)

    ans = matmat(A, B)
    expected = np.dot(A, B)
    self.assertAllClose(
        ans, expected, check_dtypes=False,
        rtol={np.float32:1e-2} if jtu.device_under_test() == "tpu" else None)

    jaxpr = make_jaxpr(matmat)(A, B)
    self.assertEqual(len(jaxpr.jaxpr.eqns), 1)

  def testPerExampleGradients(self):
    def predict(params, inputs):
      for W, b in params:
        outputs = jnp.dot(W, inputs) + b
        inputs = jnp.tanh(outputs)
      return outputs

    def loss(params, data):
      inputs, targets = data
      predictions = predict(params, inputs)
      return jnp.sum((predictions - targets)**2)

    batch_size = 5
    layer_sizes = [3, 2, 4]

    R = np.random.RandomState(0).randn
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
      std_basis = np.eye(np.size(y)).reshape((-1,) + np.shape(y))
      jac_flat, = vmap(pullback, out_axes=np.ndim(y))(std_basis)
      return jac_flat.reshape(np.shape(y) + np.shape(x))

    def jacfwd(f, x):
      pushfwd = lambda v: jvp(f, (x,), (v,))
      std_basis = np.eye(np.size(x)).reshape((-1,) + np.shape(x))
      y, jac_flat = vmap(pushfwd, out_axes=(None, 0))(std_basis)
      return jac_flat.reshape(np.shape(y) + np.shape(x))

    R = np.random.RandomState(0).randn

    A = R(4, 3)
    b = R(4)
    f = lambda x: jnp.tanh(jnp.dot(A, x) + b)

    x = R(3)
    self.assertAllClose(jacfwd(f, x), jacbwd(f, x), check_dtypes=False)

  def testBatchOfCompile(self):
    side = []

    @jit
    def f(x):
      side.append(None)
      return x + x

    g = jit(vmap(f))
    self.assertAllClose(g(np.ones(2)), 2 * np.ones(2), check_dtypes=False)
    self.assertEqual(len(side), 1)
    self.assertAllClose(g(2 * np.ones(2)), 4 * np.ones(2),
                        check_dtypes=False)
    self.assertEqual(len(side), 1)

  def testSliceLax(self):
    fun = lambda x: lax.slice(x, (2,), (4,))
    R = np.random.RandomState(0).randn
    x = R(5, 10)

    ans = vmap(fun)(x)
    expected_ans = x[:, 2:4]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testSliceNumpy(self):
    fun = lambda x: x[:, 2]
    R = np.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun)(x)
    expected_ans = x[:, :, 2]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testRevLax(self):
    fun = lambda x: lax.rev(x, [0])
    R = np.random.RandomState(0).randn
    x = R(2, 3)

    ans = vmap(fun)(x)
    expected_ans = x[:, ::-1]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    ans = vmap(fun, (1,), 1)(x)
    expected_ans = x[::-1, :]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testRevNumpy(self):
    fun = lambda x: x[:, ::-1]
    R = np.random.RandomState(0).randn
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
    fun = lambda x: jnp.maximum(x, 0.0)
    R = np.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun)(x)
    expected_ans = np.maximum(x, 0.0)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testNpGtrThan(self):
    R = np.random.RandomState(0).randn
    x = R(10, 5, 3, 7)

    ans = vmap(lambda x: x > 1.0)(x)
    expected_ans = x > 1.0
    self.assertAllClose(ans, expected_ans, check_dtypes=True)

  def testNpMaximumPerExampleGrad(self):
    R = np.random.RandomState(0).randn
    x = R(10, 5)
    W = R(5, 5)

    fun = lambda W, x: jnp.sum(jnp.maximum(jnp.dot(x, W), 0.0) ** 2)

    ans = vmap(partial(grad(fun), W))(x)

    W_t = jnp.transpose(W)
    for i in range(10):
      x_ex = x[i:i + 1]

      expected_ans = 2.0 * jnp.dot(
          jnp.maximum(jnp.dot(W_t, jnp.transpose(x_ex)), 0.0), x_ex)
      expected_ans = jnp.transpose(expected_ans)

      self.assertAllClose(
          ans[i], expected_ans, check_dtypes=False,
          atol={np.float32:5e-2} if jtu.device_under_test() == "tpu" else None)

  def testDotGeneral(self):
    R = np.random.RandomState(0).randn

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
    expected = np.stack([fun(x[..., i, :], y[:, i, ...]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    x = R(3, 4, 5, 10)
    y = R(3, 5, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(3, None))(x, y)
    expected = np.stack([fun(x[..., i], y) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    x = R(3, 4, 5)
    y = R(3, 5, 10, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(None, 2))(x, y)
    expected = np.stack([fun(x, y[..., i, :]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    x = R(4)
    y = R(4, 10)
    fun = lambda x, y: lax.dot_general(x, y, [((0,), (0,)), ((), ())])
    ans = vmap(fun, in_axes=(None, 1))(x, y)
    expected = np.stack([fun(x, y[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testDot(self):
    # these tests are based on @shoyer's notebook studying gufuncs

    def vecvec(a, b):
      dot = jnp.dot
      for ndim in range(1, max(a.ndim, b.ndim)):
        a_ax = 0 if a.ndim > ndim else None
        b_ax = 0 if b.ndim > ndim else None
        dot = vmap(dot, in_axes=(a_ax, b_ax))
      return dot(a, b)

    assert vecvec(jnp.zeros((3,)), jnp.zeros((3,))).shape == ()
    assert vecvec(jnp.zeros((2, 3)), jnp.zeros((3,))).shape == (2,)
    assert vecvec(jnp.zeros((4, 2, 3)), jnp.zeros((3,))).shape == (4, 2)

  def testDot2(self):
    R = np.random.RandomState(0).randn
    xs = R(10, 3)
    ys = R(10, 3)
    ans = vmap(jnp.dot)(xs, ys)
    expected = np.einsum('ni,ni->n', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot3(self):
    R = np.random.RandomState(0).randn
    xs = R(5, 8, 10)
    ys = R(10, 1)
    ans = vmap(jnp.dot, in_axes=(1, None))(xs, ys)
    expected = np.einsum('inj,jk->nik', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot4(self):
    R = np.random.RandomState(0).randn
    xs = R(3, 2)
    ys = R(3)
    ans = vmap(jnp.dot, in_axes=(1, None))(xs, ys)
    expected = np.einsum('ij,i->j', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot5(self):
    f = vmap(partial(jnp.einsum, 'ij,j->i'), (None, 0))
    jaxpr = make_jaxpr(f)(jnp.zeros((1000, 1000)), jnp.zeros((1000, 1000)))
    assert "broadcast" not in str(jaxpr)

  def testPad(self):
    R = np.random.RandomState(0).randn

    fun = lambda x: lax.pad(x, np.float32(0), [(1, 2, 1)])
    x = R(5, 10).astype(np.float32)
    ans = vmap(fun)(x)
    expected_ans = jnp.stack(list(map(fun, x)))
    self.assertAllClose(ans, expected_ans, check_dtypes=False)


    fun = lambda x: lax.pad(x, np.float32(0), [(1, 2, 1), (0, 1, 0)])
    x = R(5, 10, 3).astype(np.float32)
    ans = vmap(fun)(x)
    expected_ans = jnp.stack(list(map(fun, x)))
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testConcatenate(self):
    R = lambda *shape: np.random.RandomState(0).randn(*shape).astype(np.float32)

    fun = lambda *args: lax.concatenate(args, dimension=0)
    x, y, z = R(10, 2, 3), R(1, 10, 3), R(4, 3)
    ans = vmap(fun, in_axes=(0, 1, None))(x, y, z)
    expected_ans = np.concatenate([x, np.swapaxes(y, 0, 1),
                                    np.broadcast_to(z, (10, 4, 3))], 1)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    fun = lambda *args: lax.concatenate(args, dimension=1)
    x, y, z = R(10, 2, 1), R(2, 3), R(2, 4, 10)
    ans = vmap(fun, in_axes=(0, None, 2))(x, y, z)
    expected_ans = np.concatenate([x, np.broadcast_to(y, (10, 2, 3)),
                                    np.moveaxis(z, 2, 0)], 2)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testJacobianIssue54(self):
    # test modeling the code in https://github.com/google/jax/issues/54

    def func(xs):
      return jnp.array([x for x in xs])

    xs = jnp.ones((5, 1))
    jacrev(func)(xs)  # don't crash
    jacfwd(func)(xs)  # don't crash

  def testAny(self):
    # test modeling the code in https://github.com/google/jax/issues/108

    ans = vmap(jnp.any)(jnp.array([[True, False], [False, False]]))
    expected = jnp.array([True, False])
    self.assertAllClose(ans, expected, check_dtypes=True)

  @jtu.skip_on_devices("tpu")
  def testHessian(self):
    # test based on code from sindhwani@google
    def fun(x, t):
      return jnp.sum(jnp.power(jnp.maximum(x, 0.0), 2)) + t

    x = np.array([-1., -0.5, 0., 0.5, 1.0])

    ans = hessian(lambda x: fun(x, 0.0))(x)
    expected = np.array([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0.,0.5, 0., 0.],
                          [0., 0., 0., 2., 0.],
                          [0., 0., 0., 0., 2.]])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDynamicSlice(self):
    # test dynamic_slice via numpy indexing syntax
    # see https://github.com/google/jax/issues/1613 for an explanation of why we
    # need to use np rather than np to create x and idx
    x = jnp.arange(30).reshape((10, 3))

    ans = vmap(lambda x, i: x[i], in_axes=(0, None))(x, 1)
    expected = x[:, 1]
    self.assertAllClose(ans, expected, check_dtypes=False)


    idx = jnp.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, i: x[i], in_axes=(0, 0))(x, idx)
    expected = x[np.arange(10), idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = jnp.arange(3)
    idx = jnp.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, i: x[i], in_axes=(None, 0))(x, idx)
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDynamicUpdateSlice(self):
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    ans = vmap(lambda x, y, i: lax.dynamic_update_index_in_dim(x, y, i, axis=0),
               in_axes=(0, 0, None))(x, y, 1)
    expected = x.copy()
    expected[:, 1] = y
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = np.random.randn(3)
    idx = np.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, y, i: lax.dynamic_update_index_in_dim(x, y, i, axis=0),
               in_axes=(None, 0, 0))(x, y, idx)
    expected = np.broadcast_to(x, (10, 3)).copy()
    expected[np.arange(10), idx] = y
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testRandom(self):
    seeds = vmap(random.PRNGKey)(np.arange(10))
    ans = vmap(partial(random.normal, shape=(3, 2)))(seeds)
    expected = np.stack([random.normal(random.PRNGKey(seed), (3, 2))
                          for seed in np.arange(10)])
    self.assertAllClose(ans, expected, check_dtypes=False)
    assert len(np.unique(ans)) == 10 * 3 * 2

  def testSort(self):
    v = np.arange(12)[::-1].reshape(3, 4)

    sv = vmap(partial(lax.sort, dimension=0), (0,))(v)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sv = vmap(partial(lax.sort, dimension=-1), (0,))(v)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sv = vmap(partial(lax.sort, dimension=0), (1,))(v)
    self.assertAllClose(sv, v[::-1, :].T, check_dtypes=True)

    sv = vmap(partial(lax.sort, dimension=0), (1,), 1)(v)
    self.assertAllClose(sv, v[::-1, :], check_dtypes=True)

  def testSortKeyVal(self):
    k = np.arange(12)[::-1].reshape(3, 4)
    v = np.random.RandomState(0).permutation(12).reshape(3, 4)

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
    self.assertAllClose(sk, np.broadcast_to(k[0, ::-1], (3, 4)),
                        check_dtypes=True)
    self.assertAllClose(sv, v[:, ::-1], check_dtypes=True)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (1, None))(k.T, v[0])
    self.assertAllClose(sk, k[:, ::-1], check_dtypes=True)
    self.assertAllClose(sv, np.broadcast_to(v[0, ::-1], (3, 4)),
                        check_dtypes=True)

  def testConvGeneralDilated(self):
    W = jnp.array(np.random.randn(3, 3, 1, 5), dtype=np.float32)
    X = jnp.array(np.random.randn(10, 5, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      return y
    grad_loss = grad(lambda params, x: jnp.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example = jnp.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, jnp.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          jnp.reshape(g, (1,) + g.shape)]
    per_example_direct = jnp.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True,
                        rtol=2e-2)

  def testConvGeneralDilatedBatchNotMajor(self):
    W = jnp.array(np.random.randn(3, 3, 1, 4), dtype=np.float32)
    x = jnp.array(np.random.randn(3, 5, 7, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('HNWC', 'HWIO', 'HWNC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      return y

    per_example = vmap(partial(f, W))(x)
    per_example = jnp.reshape(jnp.transpose(per_example, (1, 2, 0, 3, 4)),
                             (5, 5, 21, 4))
    per_example_direct = f(W, jnp.reshape(jnp.transpose(x, (1, 0, 2, 3, 4)),
                                         (5, 21, 5, 1)))
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

  @parameterized.named_parameters(
    {"testcase_name": "_op={}".format(name), "op": op, "unit": unit}
    for name, op, unit in [("max", lax.max, -jnp.inf), ("min", lax.min, jnp.inf)])
  def testMinMaxPool(self, op, unit):
    W = jnp.array(np.random.randn(3, 3, 1, 5), dtype=np.float32)
    X = jnp.array(np.random.randn(10, 5, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      y = lax.reduce_window(
          y, unit, op, (1, 2, 2, 1), (1, 1, 1, 1), 'SAME')
      return y
    grad_loss = grad(lambda params, x: jnp.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example = jnp.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, jnp.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          jnp.reshape(g, (1,) + g.shape)]
    per_example_direct = jnp.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True,
                        rtol=5e-2)

  def testSumPool(self):
    W = jnp.array(np.random.randn(3, 3, 1, 5), dtype=np.float32)
    X = jnp.array(np.random.randn(10, 5, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      y = lax.reduce_window(
          y, 0.0, lax.add, (1, 2, 2, 1), (1, 1, 1, 1), 'SAME')
      return y
    grad_loss = grad(lambda params, x: jnp.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example = jnp.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, jnp.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          jnp.reshape(g, (1,) + g.shape)]
    per_example_direct = jnp.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct, check_dtypes=True,
                        rtol=3e-2)

  def testCumProd(self):
   x = jnp.arange(9).reshape(3, 3) + 1
   y = vmap(lambda x: jnp.cumprod(x, axis=-1))(x)
   self.assertAllClose(np.cumprod(x, axis=1, dtype=jnp.int_), y,
                       check_dtypes=True)

  def testSelect(self):
    pred = np.array([True, False])
    on_true = np.array([0, 1])
    on_false = np.array([2, 3])
    ans = vmap(lax.select)(pred, on_true, on_false)
    expected = np.array([0, 3])
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = np.array([False, True])
    on_true = np.array([0, 1])
    on_false = np.array([2, 3])
    ans = vmap(lax.select, (0, None, None))(pred, on_true, on_false)
    expected = np.array([[2, 3],
                          [0, 1]])
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = True
    on_true = np.array([0, 1], np.float32)
    on_false = np.array(3, np.float32)
    ans = vmap(lax.select, (None, 0, None))(pred, on_true, on_false)
    expected = np.array([0, 1], np.float32)
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = np.array([False, True])
    on_true = np.array([0, 1], np.float32)
    on_false = np.array(3, np.float32)
    ans = vmap(lax.select, (0, 0, None))(pred, on_true, on_false)
    expected = np.array([3, 1], np.float32)
    self.assertAllClose(ans, expected, check_dtypes=True)

    pred = np.array([False, True])
    on_true = np.array([2], np.float32)
    on_false = np.array([[3, 4]], np.float32)
    ans = vmap(lax.select, (0, None, 1), 1)(pred, on_true, on_false)
    expected = np.array([[3, 2]], np.float32)
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testLaxLinalgCholesky(self):
    a = np.random.RandomState(0).randn(10, 5, 5).astype(np.float32)
    a = np.matmul(a, np.conj(np.swapaxes(a, -1, -2)))

    ans = vmap(lax_linalg.cholesky)(a)
    expected = np.linalg.cholesky(a)
    self.assertAllClose(ans, expected, check_dtypes=False, rtol=1e-4)

    b = np.random.RandomState(0).randn(10, 5, 5).astype(np.float32)
    b = np.matmul(b, np.conj(np.swapaxes(b, -1, -2)))
    b_trans = np.swapaxes(b, 0, 1)  # shape is (5, 10, 5)

    ans = vmap(lax_linalg.cholesky, in_axes=1, out_axes=0)(b_trans)
    expected = np.linalg.cholesky(b)
    self.assertAllClose(ans, expected, check_dtypes=False, rtol=1e-4)

  def testLaxLinalgTriangularSolve(self):
    a = np.random.RandomState(0).randn(4, 10, 4).astype(np.float32)
    a += np.eye(4, dtype=jnp.float32)[:, None, :]
    b = np.random.RandomState(0).randn(5, 4, 10).astype(np.float32)

    ans = vmap(lax_linalg.triangular_solve, in_axes=(1, 2))(a, b)
    expected = np.stack(
      [lax_linalg.triangular_solve(a[:, i], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    ans = vmap(lax_linalg.triangular_solve, in_axes=(None, 2))(a[:, 0], b)
    expected = np.stack(
      [lax_linalg.triangular_solve(a[:, 0], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

    ans = vmap(lax_linalg.triangular_solve, in_axes=(1, None))(a, b[..., 0])
    expected = np.stack(
      [lax_linalg.triangular_solve(a[:, i], b[..., 0]) for i in range(10)])
    self.assertAllClose(ans, expected, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [np.float32, np.int32]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (3, 5), np.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          (1, (10, 3), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          (1, (10, 3, 5), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          (2, (10, 5, 3), np.array([[0, 2], [1, 0]]),
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,),
             start_index_map=(0, 1)),
            (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherBatchedOperand(self, axis, shape, dtype, idxs, dnums,
                               slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    ans = vmap(fun, (axis, None))(operand, idxs)
    expected = np.stack([fun(operand[(slice(None),) * axis + (i,)], idxs)
                          for i in range(operand.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [np.float32, np.float64]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (3, 5), np.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          (1, (10, 3), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          (1, (10, 3, 5), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          (2, (10, 5, 3), np.array([[0, 2], [1, 0]]),
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,),
             start_index_map=(0, 1)),
            (1, 3)),      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherGradBatchedOperand(self, axis, shape, dtype, idxs, dnums,
                                   slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: jnp.sum(jnp.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    ans = vmap(gfun, (axis, None))(operand, idxs)
    expected = np.stack([gfun(operand[(slice(None),) * axis + (i,)], idxs)
                          for i in range(operand.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [np.float32, np.int32]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (5,), np.array([[[0], [2]], [[1], [3]]]), lax.GatherDimensionNumbers(
              offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)), (1,)),
          (1, (10,), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)), (2,)),
          (1, (10, 5), np.array([[0, 2, 1], [0, 3, 3]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)), (1, 3)),
          (0, (10, 5), np.array([[[0, 1], [2, 0]],
                                  [[1, 0], [2, 3]]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)), (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherBatchedIndices(self, axis, shape, dtype, idxs, dnums,
                               slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    ans = vmap(fun, (None, axis))(operand, idxs)
    expected = np.stack([fun(operand, idxs[(slice(None),) * axis + (i,)])
                          for i in range(idxs.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in [np.float32, np.float64]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (5,), np.array([[[0], [2]], [[1], [3]]]), lax.GatherDimensionNumbers(
              offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)), (1,)),
          (1, (10,), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)), (2,)),
          (1, (10, 5), np.array([[0, 2, 1], [0, 3, 3]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)), (1, 3)),
          (0, (10, 5), np.array([[[0, 1], [2, 0]],
                                  [[1, 0], [2, 3]]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)), (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherGradBatchedIndices(self, axis, shape, dtype, idxs, dnums,
                                   slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: jnp.sum(jnp.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    ans = vmap(gfun, (None, axis))(operand, idxs)
    expected = np.stack([gfun(operand, idxs[(slice(None),) * axis + (i,)])
                          for i in range(idxs.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_op_axis={}_idxs_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), op_axis, idxs_axis, idxs,
          dnums, slice_sizes),
       "op_axis": op_axis, "idxs_axis": idxs_axis, "shape": shape, "dtype":
       dtype, "idxs": idxs, "dnums": dnums, "slice_sizes": slice_sizes,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in [np.float32, np.int32]
      for op_axis, idxs_axis, shape, idxs, dnums, slice_sizes in [
          (0, 0, (2, 5), np.array([[[0], [2]], [[1], [3]]]),
           lax.GatherDimensionNumbers(
             offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1,)),
          (1, 1, (10, 2), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
           (2,)),
          (0, 1, (2, 10, 5,), np.array([[[0, 2, 1], [0, 3, 3]]]).T,
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1, 3)),
          (2, 0, (10, 5, 2), np.array([[[0, 2], [1, 0]],
                                        [[1, 0], [2, 0]]]),
          lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
           (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherBatchedBoth(self, op_axis, idxs_axis, shape, dtype, idxs, dnums,
                            slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    assert operand.shape[op_axis] == idxs.shape[idxs_axis]
    ans = vmap(fun, (op_axis, idxs_axis))(operand, idxs)
    expected = np.stack([fun(operand[(slice(None),) * op_axis + (i,)],
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
      for dtype in [np.float32]
      for op_axis, idxs_axis, shape, idxs, dnums, slice_sizes in [
          (0, 0, (2, 5), np.array([[[0], [2]], [[1], [3]]]),
           lax.GatherDimensionNumbers(
             offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1,)),
          (1, 1, (10, 2), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
           (2,)),
          (0, 1, (2, 10, 5,), np.array([[[0, 2, 1], [0, 3, 3]]]).T,
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1, 3)),
          (2, 0, (10, 5, 2), np.array([[[0, 2], [1, 0]],
                                        [[1, 0], [2, 0]]]),
          lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
           (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(shape))]
      for rng_factory in [jtu.rand_default])
  def testGatherGradBatchedBoth(self, op_axis, idxs_axis, shape, dtype, idxs, dnums,
                                slice_sizes, rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: jnp.sum(jnp.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    assert operand.shape[op_axis] == idxs.shape[idxs_axis]
    ans = vmap(gfun, (op_axis, idxs_axis))(operand, idxs)
    expected = np.stack([gfun(operand[(slice(None),) * op_axis + (i,)],
                              idxs[(slice(None),) * idxs_axis + (i,)])
                          for i in range(idxs.shape[idxs_axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNumpyIndexing1(self):
    a = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    ind = np.array([[0, 1],
                    [2, 0]])
    def f(a, ind):
      return a[:, ind]
    expected = np.stack([f(a, ind[i, :]) for i in range(ind.shape[0])])
    ans = vmap(f, (None, 0))(a, ind)
    assert np.all(ans == expected)

  def testNumpyIndexing2(self):
    a = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    def f(a):
      inds = jnp.array([0, 2])
      return a[:, inds]
    ans = vmap(f)(a)
    expected = np.stack([f(a[:, i, :]) for i in range(a.shape[1])], axis=1)
    assert np.all(ans == expected)

  def testTranspose(self):
    x = np.arange(4 * 3 * 3).reshape((4, 3, 3))
    ans = vmap(lambda x: x + x.T)(x)
    expected = x + np.swapaxes(x, -1, -2)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposePermutation(self):
    x = np.arange(6 * 3 * 4 * 5).reshape((6, 3, 4, 5))
    ans = vmap(lambda x: jnp.transpose(x, (1, 0, 2)))(x)
    expected = np.transpose(x, (0, 2, 1, 3))
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = np.arange(6 * 3 * 4 * 5).reshape((6, 3, 4, 5))
    ans = vmap(lambda x: jnp.transpose(x, (1, 2, 0)))(x)
    expected = np.transpose(x, (0, 2, 3, 1))
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = np.arange(6 * 3 * 4 * 5).reshape((3, 4, 6, 5))
    ans = vmap(lambda x: jnp.transpose(x, (1, 2, 0)), in_axes=2)(x)
    expected = np.transpose(x, (2, 1, 3, 0))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testIssue354(self):
    psd_mat = np.random.randn(20, 10)
    psd_mat = psd_mat.T.dot(psd_mat)
    vec = np.random.randn(10)

    def f(scale):
      scaled_mat = scale * psd_mat
      chol = jnp.linalg.cholesky(scaled_mat)
      return -0.5 * jnp.sum((jnp.einsum('ij,j->i', chol, vec))**2)
    vmapped_f = vmap(f)
    vmapped_f_grad = grad(lambda x: jnp.sum(vmapped_f(x)))

    scales = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    ans = vmapped_f_grad(scales)  # don't crash!
    expected = np.stack([grad(f)(scale) for scale in scales])
    self.assertAllClose(ans, expected, check_dtypes=False,
                        rtol=jtu.default_gradient_tolerance)

  def testIssue387(self):
    # https://github.com/google/jax/issues/387
    R = np.random.RandomState(0).rand(100, 2)

    def dist_sq(R):
      dR = R[:, jnp.newaxis, :] - R[jnp.newaxis, :, :]
      zero = jnp.zeros_like(dR)
      dR = dR - jnp.where(jnp.abs(dR) < 0.5, zero, 0.5 * jnp.sign(dR))
      return jnp.sum(dR ** 2, axis=2)

    @jit
    def f(R):
      dr = dist_sq(R)
      return jnp.sum(R ** 2)

    H = hessian(f)(R)  # don't crash on UnshapedArray

  def testIssue489(self):
    def f(key):
      def body_fn(uk):
        key = uk[1]
        u = random.uniform(key, (), dtype=jnp.float64)
        key, _ = random.split(key)
        return u, key

      u, _ = lax.while_loop(lambda uk: uk[0] > 0.5, body_fn,
                            (jnp.float64(1.), key))
      return u

    print(vmap(f)(random.split(random.PRNGKey(0), 2)))  # no crash

  def testEmptyTuples(self):
    # Ensure there is no crash when a vectorized input contains empty tuples.
    result = vmap(lambda x, _: x + 1)(np.array([0, 1]), ())
    self.assertAllClose(result, np.array([1, 2]), check_dtypes=False)
    # Ensure there is no crash when a vectorized output contains empty tuples.
    result, empty_tuple = vmap(lambda x: (x + 1, ()))(np.array([0, 1]))
    self.assertAllClose(result, np.array([1, 2]), check_dtypes=False)
    self.assertEqual((), empty_tuple)

  def testIndexAddBatchedIndexesOnly(self):
    f = lambda x, idx, y: jax.ops.index_add(x, jax.ops.index[idx], y)
    result = vmap(f, (None, 0, None))(np.zeros((10,)), np.arange(10,), 1.)
    self.assertAllClose(result, np.eye(10), check_dtypes=False)

  def testIssue1170(self):
    def f(index1, index2):
      return jnp.arange(36).reshape(6, 6)[index1, index2]
    g = jax.jit(jax.pmap(f))
    ans = g(index1=np.asarray([1]), index2=np.asarray([2]))
    expected = g(np.asarray([1]), np.asarray([2]))
    self.assertAllClose(ans, expected, check_dtypes=True)


if __name__ == '__main__':
  absltest.main()
