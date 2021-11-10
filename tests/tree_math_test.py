# Copyright 2020 Google LLC
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

import operator
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from jax import lax
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.experimental import tree_math as tm
from jax.tree_util import tree_map, tree_flatten, tree_leaves

from jax.config import config
config.parse_flags_with_absl()


class TreeMathTest(jtu.JaxTestCase):

  def assertTreeEqual(self, expected, actual, check_dtypes):
    expected_leaves, expected_treedef = tree_flatten(expected)
    actual_leaves, actual_treedef = tree_flatten(actual)
    self.assertEqual(actual_treedef, expected_treedef)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
      self.assertArraysEqual(actual_leaf, expected_leaf, check_dtypes=check_dtypes)

  def test_vector(self):
    tree = {'a': 0, 'b': jnp.array([1, 2], dtype=jnp.int32)}
    vector = tm.Vector(tree)
    self.assertEqual(len(vector), 3)
    self.assertEqual(vector.shape, (3,))
    self.assertEqual(vector.ndim, 1)
    self.assertEqual(vector.dtype, jnp.int32)
    self.assertEqual(repr(tm.Vector({'a': 1})), "tree_math.Vector({'a': 1})")
    self.assertTreeEqual(tree_leaves(tree), tree_leaves(vector), check_dtypes=True)
    vector2 = tree_map(lambda x: x, vector)
    self.assertTreeEqual(vector, vector2, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": op.__name__, "op": op}
      for op in [operator.pos, operator.neg, abs, operator.invert]
  ))
  def test_unary_math(self, op):
    tree = {'a': 1, 'b': -jnp.array([2, 3])}
    expected = tm.Vector(tree_map(op, tree))
    actual = op(tm.Vector(tree))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_arithmetic_with_scalar(self):
    vector = tm.Vector({'a': 0, 'b': jnp.array([1, 2])})
    expected = tm.Vector({'a': 1, 'b': jnp.array([2, 3])})
    self.assertTreeEqual(vector + 1, expected, check_dtypes=True)
    self.assertTreeEqual(1 + vector, expected, check_dtypes=True)
    with self.assertRaisesRegex(
        TypeError, "non-tree_math.Vector arguments must be scalars",
    ):
      vector + jnp.ones((3,))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": op.__name__, "op": op}
      for op in [
          operator.add,
          operator.sub,
          operator.mul,
          operator.truediv,
          operator.floordiv,
          operator.mod,
      ]
  ))
  def test_binary_arithmetic(self, op):
    rng = jtu.rand_default(self.rng())
    tree1 = {'a': rng((), jnp.float32), 'b': rng((2, 3), jnp.float32)}
    tree2 = {'a': rng((), jnp.float32), 'b': rng((2, 3), jnp.float32)}
    expected = tm.Vector(tree_map(op, tree1, tree2))
    actual = op(tm.Vector(tree1), tm.Vector(tree2))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_pow(self):
    expected = tm.Vector({'a': 2 ** 3})
    actual = tm.Vector({'a': 2}) ** tm.Vector({'a': 3})
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_divmod(self):
    raise unittest.SkipTest("not working yet")
    x, y = divmod(jnp.arange(5), 2)
    expected = tm.Vector({'a': x}), tm.Vector({'a': y})
    actual = divmod(tm.Vector({'a': jnp.arange(5)}), 2)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_matmul_scalars(self):
    actual = tm.Vector(1.0) @ tm.Vector(2.0)
    expected = 2.0
    self.assertAllClose(actual, expected, check_dtypes=True)

  def test_matmul(self):
    rng = jtu.rand_default(self.rng())
    tree1 = {'a': rng((), jnp.float32), 'b': rng((2, 3), jnp.float32)}
    tree2 = {'a': rng((), jnp.float32), 'b': rng((2, 3), jnp.float32)}
    expected = tree1['a'] * tree2['a'] + tree1['b'].ravel() @ tree2['b'].ravel()
    vector1 = tm.Vector(tree1)
    vector2 = tm.Vector(tree2)
    actual = vector1 @ vector2
    self.assertAllClose(actual, expected, check_dtypes=True)
    with self.assertRaisesRegex(
        TypeError, "matmul arguments must both be tree_math.Vector objects",
    ):
      vector1 @ jnp.ones((7,))

  # TODO(shoyer): test comparisons and bitwise ops

  def test_conj(self):
    vector = tm.Vector({"a": jnp.array([1, 1j])})
    actual = vector.conj()
    expected = tm.Vector({"a": jnp.array([1, -1j])})
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_sum_mean_min_max(self):
    vector = tm.Vector({"a": 1, "b": jnp.array([2, 3, 4])})
    self.assertTreeEqual(vector.sum(), 10, check_dtypes=True)
    self.assertTreeEqual(vector.min(), 1, check_dtypes=True)
    self.assertTreeEqual(vector.max(), 4, check_dtypes=True)

  def test_where(self):
    condition = tm.Vector({"a": jnp.array([True, False])})
    x = tm.Vector({"a": jnp.array([1, 2])})
    y = 3
    expected = tm.Vector({"a": jnp.array([1, 3])})
    actual = tm.where(condition, x, y)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_where_all_scalars(self):
    expected = 1
    actual = tm.where(True, 1, 2)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_zeros_like(self):
    x = jnp.array([1, 2])
    expected = tm.Vector({"a": jnp.zeros_like(x)})
    actual = tm.zeros_like(tm.Vector({"a": x}))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_ones_like(self):
    x = jnp.array([1, 2])
    expected = tm.Vector({"a": jnp.ones_like(x)})
    actual = tm.ones_like(tm.Vector({"a": x}))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_full_like(self):
    x = jnp.array([1, 2])
    expected = tm.Vector({"a": jnp.full_like(x, 3)})
    actual = tm.full_like(tm.Vector({"a": x}), 3)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": name, "func_name": name}
      for name in ["maximum", "minimum"]
  ))
  def test_binary_ufuncs(self, func_name):
    jnp_func = getattr(jnp, func_name)
    tree_func = getattr(tm, func_name)
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 3, 2])
    expected = tm.Vector({"a": jnp_func(x, y)})
    actual = tree_func(tm.Vector({"a": x}), tm.Vector({"a": y}))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_basic_wrap(self):
    @tm.wrap
    def f(x, y):
      return x - y

    x = {'a': 10, 'b': jnp.array([20, 30])}
    y = {'a': 1, 'b': jnp.array([2, 3])}
    expected = {'a': 9, 'b': jnp.array([18, 27])}
    actual = f(x, y)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_tree_out_wrap(self):
    @tm.wrap
    def f(x, y):
      return x + y, x - y

    x = {'a': 10, 'b': jnp.array([20, 30])}
    y = {'a': 1, 'b': jnp.array([2, 3])}
    expected = ({'a': 11, 'b': jnp.array([22, 33])},
                {'a': 9, 'b': jnp.array([18, 27])})
    actual = f(x, y)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_unwrap(self):

    @tm.unwrap
    def f(x):
      return {'b': x['a'] + 1}

    actual = f(tm.Vector({'a': 1}))
    expected = tm.Vector({'b': 2})
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_cg(self):
    # an integration test to verify non-trivial examples work

    def cg(b, x0, maxiter=5, tol=1e-5, atol=0.0):

      # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
      bs = b @ b
      atol2 = tm.maximum(tol**2 * bs, atol**2)

      # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

      def cond_fun(value):
        x, r, gamma, p, k = value
        rs = r @ r
        return (rs > atol2) & (k < maxiter)

      def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / (p.conj() @ Ap)
        x_ = x + alpha * p
        r_ = r - alpha * Ap
        z_ = M(r_)
        gamma_ = r_.conj() @ z_
        beta_ = gamma_ / gamma
        p_ = z_ + beta_ * p
        return x_, r_, gamma_, p_, k + 1

      r0 = b - A(x0)
      p0 = z0 = M(r0)
      gamma0 = r0 @ z0
      initial_value = (x0, r0, gamma0, p0, 0)

      x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

      return x_final

    A = tm.unwrap(
        lambda x: {'a': x['a'] + 0.5 * x['b'], 'b': 0.5 * x['a'] + x['b']}
    )
    b = {'a': 1.0, 'b': -1.0}
    x0 = {'a': 0.0, 'b': 0.0}
    M = lambda x: x

    actual = tm.wrap(cg)(b, x0)

    expected = {'a': 2.0, 'b': -2.0}
    self.assertAllClose(actual, expected, check_dtypes=True)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
