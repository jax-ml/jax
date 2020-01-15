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

from jax import test_util as jtu, core as jc, api
from jax.interpreters.masking import ShapeError, shape_as_value, parse_spec, \
  constant_poly, Mon, Poly
from jax import mask, vmap, jit, grad, shapecheck
from jax import lax
import jax.numpy as np

from jax.config import config
config.parse_flags_with_absl()


# These are 'manual' tests for masking and shape checking. The more exhaustive,
# more systematic tests should live in lax_test.py.

class MaskingTest(jtu.JaxTestCase):

  @parameterized.parameters([
      ['(m, n)', 'ShapeSpec(m, n)'],
      ['(m * n)', 'ShapeSpec(m n)'],
      ['m * n', 'ShapeSpec(m n)'],
      ['(m * n,)', 'ShapeSpec(m n)'],
      ['(3, m)', 'ShapeSpec(3, m)'],
      ['(10, m)', 'ShapeSpec(10, m)'],
      ['(-10, m)', 'ShapeSpec(-10, m)'],
      ['(3 * m)', 'ShapeSpec(3 m)'],
      ['m', 'ShapeSpec(m)'],
      ['', 'ShapeSpec()'],
      ['m + n', 'ShapeSpec(m + n)'],
      ['m + n * k', 'ShapeSpec(m + k n)'],
      ['m + 3 * k', 'ShapeSpec(3 k + m)'],
      ['', 'ShapeSpec()'],
      ['_', 'ShapeSpec(_)'],
  ])
  def test_shape_parsing(self, spec, ans):
    self.assertEqual(str(parse_spec(spec)), ans)

  def test_poly_equal(self):
    assert constant_poly(3) == 3
    assert constant_poly(4) != 3
    assert 3 == constant_poly(3)
    assert 4 != constant_poly(3)
    assert constant_poly(4) == constant_poly(4)
    assert constant_poly(3) != constant_poly(4)
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) == Poly({Mon({'n': 1}): 4, Mon(): 3})
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 2}): 4})
    assert Poly({Mon(): 3, Mon({'m': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 1}): 4})

  # def test_shapecheck_add_broadcast(self):
  #   @shapecheck(['(m, n)', 'n'], '(m, n)')
  #   @shapecheck(['n', ''], 'n')
  #   def add(a, b):
  #     return a + b

  def test_shapecheck_sum(self):
    @shapecheck(['(m, n)'], '')
    def sum(x):
      return np.sum(x)

  def test_shapecheck_prod(self):
    @shapecheck(['(m, n)'], '')
    def prod(x):
      return np.prod(x)

  def test_shapecheck_max(self):
    @shapecheck(['(m, n)'], '')
    def prod(x):
      return np.max(x)

  def test_shapecheck_min(self):
    @shapecheck(['(m, n)'], '')
    def prod(x):
      return np.min(x)

  def test_shapecheck_dot(self):
    @shapecheck(['(m, n)', 'n'], 'm')
    def matvec(A, b):
      return np.dot(A, b)

    def thunk():
      @shapecheck(['(m, n)', 'n'], 'm')
      def matvec(A, b):
        return lax.dot_general(A, b, [((0,), (0,)), ((), ())])
    self.assertRaisesRegex(TypeError, "", thunk)

  def test_shapecheck_flatten(self):
    @shapecheck(['(m, n)'], 'm * n')
    def flatten(x):
      return lax.reshape(x, (x.shape[0] * x.shape[1],))

  def test_shapecheck_concatenate(self):
    @shapecheck(['m', 'n', 'm'], '3*m + n')
    def cat(x, y, z):
      return lax.concatenate([x, y, x, z], 0)

    def thunk():
      @shapecheck(['m', 'n', 'm'], '3*m + n')
      def cat(x, y, z):
        return lax.concatenate([x, y, x], 0)
    self.assertRaisesRegex(ShapeError, "", thunk)

  def test_shapecheck_device_put(self):
    @shapecheck(['n'], 'n')
    def d_put(x):
      return api.device_put(x)

  def test_shapecheck_broadcast_in_dim(self):
    x = np.zeros((7, 1))
    lax.broadcast_in_dim(x, shape=(3, x.shape[0], 4), broadcast_dimensions=(1, 2))
    @shapecheck(['(n, 1)'], '(3, n, 4)')
    def broadcast_in_dim(x):
      return lax.broadcast_in_dim(x, shape=(3, x.shape[0], 4), broadcast_dimensions=(1, 2))

  def test_shapecheck_jit(self):
    @shapecheck(['n'], '2*n')
    @jit
    def concat(x):
      return lax.concatenate([x, x], 0)

  def test_shapecheck_pad(self):
    @shapecheck(['n'], '2*n+1')
    def p(x):
      return lax.pad(x, 0, [(1, 1, 1)])

  def test_shapecheck_numpy_pad(self):
    @shapecheck(['n'], 'n+1')
    def p(x):
      return np.pad(x, (0, 1))

  @parameterized.named_parameters(jtu.cases_from_list(
    {
      'testcase_name': "strides={}_padding={}_lhs_dilation={}_dimension_numbers"
                       "={}_lhs_perm={}_rhs_perm={}_out_perm={}".format(
        strides, padding, lhs_dilation, dimension_numbers, lhs_perm, rhs_perm, out_perm),
      'strides': strides, 'padding': padding, 'lhs_dilation': lhs_dilation,
      'dimension_numbers': dimension_numbers, 'lhs_perm': lhs_perm,
      'rhs_perm': rhs_perm, 'out_perm': out_perm}
    for strides in [(1, 1), (2, 1)]
    for padding in ['SAME', 'VALID', ((1, 0), (2, 0))]
    for lhs_dilation in (None, (1, 2))
    for dimension_numbers, (lhs_perm, rhs_perm, out_perm) in (
            (("NCHW", "OIHW", "NCHW"), ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))),
            (("NHWC", "HWIO", "NHWC"), ((0, 2, 3, 1), (2, 3, 1, 0), (0, 2, 3, 1))),
            (("NCHW", "HWIO", "NHWC"), ((0, 1, 2, 3), (2, 3, 1, 0), (0, 2, 3, 1)))
    )
    # String padding is not implemented for transposed convolution, see conv_general_dilated implementation:
    if (lhs_dilation is None or not isinstance(padding, str)) and
    # only test strides with same padding:
    (strides[0] == 1 or padding == 'SAME')))
  def test_shapecheck_conv(self, strides, padding, lhs_dilation,
                           dimension_numbers, lhs_perm, rhs_perm, out_perm):
    valid = padding == 'VALID'
    is_strided = strides[0] != 1
    lhs_shape = '({}, {}, {}, {})'.format(*onp.take(['n', 'i', '2*h' if is_strided else 'h', 'w'], lhs_perm))
    rhs_shape = '({}, {}, {}, {})'.format(*onp.take(['o', 'i', '2', '3'], rhs_perm))
    out_shape = '({}, {}, {}, {})'.format(*onp.take([
      'n', 'o', 'h+-1' if valid and not is_strided else 'h',
      ('w+-2' if valid else 'w') if lhs_dilation is None else '2*w+-1'], out_perm))

    @shapecheck([lhs_shape, rhs_shape], out_shape)
    def conv(lhs, rhs):
      return lax.conv_general_dilated(
        lhs, rhs, strides, padding,
        lhs_dilation=lhs_dilation, dimension_numbers=dimension_numbers)

  def test_shapecheck_unsupported_op(self):
    p = jc.Primitive('unsupported_op')
    p.def_impl(lambda x: x)

    def thunk():
      @shapecheck(['n'], 'n')
      def identity(x):
        return p.bind(x)

    self.assertRaisesRegex(NotImplementedError, "Shape rule for unsupported_op not implemented yet.", thunk)

  def test_sum(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = 8
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=4))
    expected = 9
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_sum_vmap(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    ans = vmap(padded_sum)([np.ones((5, 10))], dict(n=np.arange(5)))
    expected = onp.array([0, 1, 2, 3, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_add(self):
    @partial(mask, in_shapes=['n', 'n'], out_shape='n')
    def addvecs(x, y):
      return x + y

    x = np.array([3, 1, 4, 1, 5, 9])
    y = np.array([2, 6, 5, 3, 5, 8])
    ans = addvecs([x, y], dict(n=3))
    expected = onp.array([5, 7, 9])
    self.assertAllClose(ans[:3], expected, check_dtypes=False)

    thunk = lambda: addvecs([np.arange(5), np.arange(6)], dict(n=3))
    self.assertRaisesRegex(ShapeError, "", thunk)

  def test_scan(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = cumsum([np.array([5, 2, 9, 1, 4])], dict(n=3))
    expected = 16
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_vmap(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = vmap(cumsum)([np.arange(6).reshape(2, 3)], dict(n=np.array([1, 2])))
    expected = onp.array([0, 7])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_jit(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    @jit
    def jit_cumsum(args, shape_env):
      assert python_should_be_executing
      return cumsum(args, shape_env)

    python_should_be_executing = True
    ans = jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=3))
    expected = 16
    self.assertAllClose(ans, expected, check_dtypes=False)

    python_should_be_executing = False
    ans = jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=4))
    expected = 17
    self.assertAllClose(ans, expected, check_dtypes=False)

    python_should_be_executing = False
    ans = jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=1))
    expected = 5
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_concatenate(self):
    @partial(mask, in_shapes=['n', 'm', 'n'], out_shape='m + 2 * n')
    def cat(x, y, z):
      return lax.concatenate([x, y, z], 0)

    ans = cat([np.array([1, 9]), np.array([2, 4, 9]), np.array([3, 9])],
              dict(n=1, m=2))
    expected = onp.array([1, 2, 4, 3])
    self.assertAllClose(ans[:4], expected, check_dtypes=False)

  def test_dot(self):
    @partial(mask, in_shapes=['(m, k)', '(k, n)'], out_shape='(m, n)')
    def dot(x, y):
      return lax.dot(x, y)

    x = onp.arange(6, dtype=onp.float32).reshape((2, 3))
    y = onp.arange(12, dtype=onp.float32).reshape((3, 4))
    ans = dot([x, y], dict(m=2, k=2, n=2))
    expected = onp.dot(x[:2, :2], y[:2, :2])
    self.assertAllClose(ans[:2, :2], expected, check_dtypes=False)

  def test_mean(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x) / shape_as_value(x.shape)[0]

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = 8 / 3
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    ans = padded_sum([np.array([[3, 4], [5, 6]])], dict(n=1))
    expected = 8
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic2(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='n')
    def padded_sum(x):
      return np.sum(x, axis=0)

    ans = padded_sum([np.array([[3, 4], [5, 6]])], dict(n=2))
    expected = np.array([8, 10])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic3(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='_')
    def padded_sum(x):
      return np.sum(x, axis=1)

    ans = padded_sum([np.array([[3, 4], [5, 6]])], dict(n=1))
    expected = np.array([3, 5])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_rnn(self):
    n = 3

    @partial(mask, in_shapes=['(_, _)', '(t, _)'], out_shape='_')
    def rnn(W, xs):
      def step(h, x):
        new_h = np.dot(W, h) + np.dot(W, x)
        return new_h, ()
      predicted, _ = lax.scan(step, np.zeros(n), xs)
      return predicted

    rng = onp.random.RandomState(0)
    W = np.eye(n)
    xs = rng.randn(10, n).astype(np.float_)
    ans = rnn([W, xs], dict(t=4))
    expected = xs[:4].sum(0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_rnn_grad(self):
    n = 3

    @partial(mask, in_shapes=['(_, _)', '(t, _)', '_'], out_shape='')
    def rnn(W, xs, target):
      def step(h, x):
        new_h = np.tanh(np.dot(W, h) + np.dot(W, x))
        return new_h, ()
      predicted, _ = lax.scan(step, np.zeros(n), xs)
      return np.sum((predicted - target)**2)

    rng = onp.random.RandomState(0)
    W = rng.randn(n, n).astype(np.float_)
    xs = rng.randn(10, n).astype(np.float_)
    y = rng.randn(n).astype(np.float_)

    ans = grad(lambda W: rnn([W, xs, y], dict(t=4)))(W)

    def rnn_reference(W, xs, target):
      h = np.zeros(n)
      for x in xs:
        h = np.tanh(np.dot(W, h) + np.dot(W, x))
      predicted = h
      return np.sum((predicted - target)**2)

    expected = grad(lambda W: rnn_reference(W, xs[:4], y))(W)

    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_ragged_batched_rnn(self):
    n = 3

    @partial(mask, in_shapes=('(_, _)', '(t, _)', '_'), out_shape='')
    def rnn(W, xs, target):
      def step(h, x):
        new_h = np.tanh(np.dot(W, h) + np.dot(W, x))
        return new_h, ()
      predicted, _ = lax.scan(step, np.zeros(n), xs)
      return np.sum((predicted - target)**2)

    rng = onp.random.RandomState(0)
    W = rng.randn(n, n).astype(np.float_)
    seqs = rng.randn(3, 10, n).astype(np.float_)
    ts = np.array([2, 5, 4])
    ys = rng.randn(3, n)

    ans = grad(lambda W: vmap(rnn, ((None, 0, 0), 0))((W, seqs, ys), dict(t=ts)).sum())(W)

    def rnn_reference(W, seqs, targets):
      total_loss = 0
      for xs, target in zip(seqs, targets):
        h = np.zeros(n)
        for x in xs:
          h = np.tanh(np.dot(W, h) + np.dot(W, x))
        predicted = h
        total_loss = total_loss + np.sum((predicted - target)**2)
      return total_loss

    seqs_ = [xs[:t] for xs, t in zip(seqs, ts)]
    expected = grad(lambda W: rnn_reference(W, seqs_, ys).sum())(W)

    self.assertAllClose(
        ans, expected, check_dtypes=False,
        rtol=2e-2 if jtu.device_under_test() == "tpu" else 1e-5)

  def test_nesting(self):
    raise SkipTest("not yet implemented")

    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return np.sum(x)

    batched_sum = vmap(padded_sum)

    @partial(mask, in_shapes=['(m, _)', 'm'], out_shape='')
    def fun(x, ns):
      return batched_sum([x], dict(n=ns)).sum()

    x = np.array([[3, 1, 4, 1],
                  [5, 9, 2, 6],
                  [5, 3, 5, 8]])
    ns = np.array([2, 3, 2])
    ans = fun([x, ns], dict(m=2))
    expected = 3+1 + 5+9+2
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_arange(self):
    raise SkipTest("not yet implemented")

    @partial(mask, in_shapes=['n'], out_shape='n')
    def padded_add(x):
      return x + lax.iota(x.shape[0])

    ans = padded_add([np.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = onp.array([3, 2, 6])
    self.assertAllClose(ans[:3], expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
