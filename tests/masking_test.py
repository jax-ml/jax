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

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

from jax import test_util as jtu
from jax.interpreters.masking import ShapeError
from jax import mask, vmap, jit, Shape, shapecheck, s_
from jax import lax
import jax.numpy as np

from jax.config import config
config.parse_flags_with_absl()


# These are 'manual' tests for masking and shape checking. The more exhaustive,
# more systematic tests should live in lax_test.py.

class MaskingTest(jtu.JaxTestCase):

  def test_shape_parsing(self):
    self.assertEqual(str(Shape('(m, n)')),    'ShapeExpr(m, n)')
    self.assertEqual(str(Shape('(m * n)')),   'ShapeExpr(m n)')
    self.assertEqual(str(Shape('m * n')),     'ShapeExpr(m n)')
    self.assertEqual(str(Shape('(m * n,)')),  'ShapeExpr(m n)')
    self.assertEqual(str(Shape('(3, m)')),    'ShapeExpr(3, m)')
    self.assertEqual(str(Shape('(3 * m)')),   'ShapeExpr(3 m)')
    self.assertEqual(str(Shape('m')),         'ShapeExpr(m)')
    self.assertEqual(str(Shape('')),          'ShapeExpr()')
    self.assertEqual(str(Shape('m + n')),     'ShapeExpr(m + n)')
    self.assertEqual(str(Shape('m + n * k')), 'ShapeExpr(m + k n)')
    self.assertEqual(str(Shape('m + 3 * k')), 'ShapeExpr(3 k + m)')

  def test_dot_shape_checking(self):
    @shapecheck((s_['m', 'n'], s_['n']), s_['m'])
    def matvec(A, b):
      return np.dot(A, b)

    def thunk():
      @shapecheck((s_['m', 'n'], s_['n']), s_['m'])
      def matvec(A, b):
        return np.dot(b, A)
    self.assertRaisesRegex(ShapeError, "", thunk)

  def test_flatten_shape_checking(self):
    @shapecheck((s_['m', 'n'],), s_['m * n'])
    def flatten(x):
      return lax.reshape(x, (x.shape[0] * x.shape[1],))

  def test_concatenate_shape_checking(self):
    @shapecheck((s_['m'], s_['n'], s_['m']), s_['3*m + n'])
    def cat(x, y, z):
      return lax.concatenate([x, y, x, z], 0)

    def thunk():
      @shapecheck((s_['m'], s_['n'], s_['m']), s_['3*m + n'])
      def cat(x, y, z):
        return lax.concatenate([x, y, x], 0)
    self.assertRaisesRegex(ShapeError, "", thunk)

  def test_sum(self):
    @partial(mask, in_shapes=[Shape('n')], out_shape=Shape())
    def padded_sum(x):
      return np.sum(x)

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = 8
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = padded_sum([np.array([3, 1, 4, 1, 5])], dict(n=4))
    expected = 9
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_sum_vmap(self):
    @partial(mask, in_shapes=[Shape('n')], out_shape=Shape())
    def padded_sum(x):
      return np.sum(x)

    ans = vmap(padded_sum)([np.ones((5, 10))], dict(n=np.arange(5)))
    expected = onp.array([0, 1, 2, 3, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_add(self):
    @partial(mask, in_shapes=[Shape('n'), Shape('n')], out_shape=Shape('n'))
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
    @partial(mask, in_shapes=[Shape('n')], out_shape=Shape())
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = cumsum([np.array([5, 2, 9, 1, 4])], dict(n=3))
    expected = 16
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_vmap(self):
    @partial(mask, in_shapes=[Shape('n')], out_shape=Shape())
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = vmap(cumsum)([np.arange(6).reshape(2, 3)], dict(n=np.array([1, 2])))
    expected = onp.array([0, 7])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_jit(self):
    @partial(mask, in_shapes=[Shape('n')], out_shape=Shape())
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
    @partial(mask, in_shapes=[Shape('n'), Shape('m'), Shape('n')],
            out_shape=Shape('m + 2 * n'))
    def cat(x, y, z):
      return lax.concatenate([x, y, z], 0)

    ans = cat([np.array([1, 9]), np.array([2, 4, 9]), np.array([3, 9])],
              dict(n=1, m=2))
    expected = onp.array([1, 2, 4, 3])
    self.assertAllClose(ans[:4], expected, check_dtypes=False)

  def test_dot(self):
    @partial(mask, in_shapes=[Shape('(m, k)'), Shape(('k, n'))],
            out_shape=[Shape('(m, n)')])
    def dot(x, y):
      return lax.dot(x, y)

    x = onp.arange(6, dtype=onp.float32).reshape((2, 3))
    y = onp.arange(12, dtype=onp.float32).reshape((3, 4))
    ans = dot([x, y], dict(m=2, k=2, n=2))
    expected = onp.dot(x[:2, :2], y[:2, :2])
    self.assertAllClose(ans[:2, :2], expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
