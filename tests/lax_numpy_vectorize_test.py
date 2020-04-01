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

from functools import partial
import itertools
import unittest

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax import numpy as np
from jax import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()


class VectorizeTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_leftshape={}_rightshape={}".format(left_shape, right_shape),
       "left_shape": left_shape, "right_shape": right_shape, "result_shape": result_shape}
      for left_shape, right_shape, result_shape in [
          ((2, 3), (3, 4), (2, 4)),
          ((2, 3), (1, 3, 4), (1, 2, 4)),
          ((5, 2, 3), (1, 3, 4), (5, 2, 4)),
          ((6, 5, 2, 3), (3, 4), (6, 5, 2, 4)),
      ]))
  def test_matmat(self, left_shape, right_shape, result_shape):
    matmat = np.vectorize(np.dot, signature='(n,m),(m,k)->(n,k)')
    self.assertEqual(matmat(np.zeros(left_shape),
                            np.zeros(right_shape)).shape, result_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_leftshape={}_rightshape={}".format(left_shape, right_shape),
       "left_shape": left_shape, "right_shape": right_shape, "result_shape": result_shape}
      for left_shape, right_shape, result_shape in [
          ((2, 3), (3,), (2,)),
          ((2, 3), (1, 3), (1, 2)),
          ((4, 2, 3), (1, 3), (4, 2)),
          ((5, 4, 2, 3), (1, 3), (5, 4, 2)),
      ]))
  def test_matvec(self, left_shape, right_shape, result_shape):
    matvec = np.vectorize(np.dot, signature='(n,m),(m)->(n)')
    self.assertEqual(matvec(np.zeros(left_shape),
                            np.zeros(right_shape)).shape, result_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_leftshape={}_rightshape={}".format(left_shape, right_shape),
       "left_shape": left_shape, "right_shape": right_shape, "result_shape": result_shape}
      for left_shape, right_shape, result_shape in [
          ((3,), (3,), ()),
          ((2, 3), (3,), (2,)),
          ((4, 2, 3), (3,), (4, 2)),
      ]))
  def test_vecmat(self, left_shape, right_shape, result_shape):
    vecvec = np.vectorize(np.dot, signature='(m),(m)->()')
    self.assertEqual(vecvec(np.zeros(left_shape),
                            np.zeros(right_shape)).shape, result_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(shape),
       "shape": shape, "result_shape": result_shape}
      for shape, result_shape in [
          ((3,), ()),
          ((2, 3,), (2,)),
          ((1, 2, 3,), (1, 2)),
      ]))
  def test_magnitude(self, shape, result_shape):
    size = 1
    for x in shape:
        size *= x
    inputs = np.arange(size).reshape(shape)

    @partial(np.vectorize, signature='(n)->()')
    def magnitude(x):
      return np.dot(x, x)

    self.assertEqual(magnitude(inputs).shape, result_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(shape),
       "shape": shape, "result_shape": result_shape}
      for shape, result_shape in [
          ((3,), ()),
          ((2, 3), (2,)),
          ((1, 2, 3, 4), (1, 2, 3)),
      ]))
  def test_mean(self, shape, result_shape):
    mean = np.vectorize(np.mean, signature='(n)->()')
    self.assertEqual(mean(np.zeros(shape)).shape, result_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(shape),
       "shape": shape, "result_shape": result_shape}
      for shape, result_shape in [
          ((), (2,)),
          ((3,), (3,2,)),
      ]))
  def test_stack_plus_minus(self, shape, result_shape):

    @partial(np.vectorize, signature='()->(n)')
    def stack_plus_minus(x):
      return np.stack([x, -x])

    self.assertEqual(stack_plus_minus(np.zeros(shape)).shape, result_shape)

  def test_center(self):

    @partial(np.vectorize, signature='(n)->(),(n)')
    def center(array):
      bias = np.mean(array)
      debiased = array - bias
      return bias, debiased

    b, a = center(np.arange(3))
    self.assertEqual(a.shape, (3,))
    self.assertEqual(b.shape, ())
    self.assertAllClose(1.0, b, check_dtypes=False)

    b, a = center(np.arange(6).reshape(2, 3))
    self.assertEqual(a.shape, (2, 3))
    self.assertEqual(b.shape, (2,))
    self.assertAllClose(np.array([1.0, 4.0]), b, check_dtypes=False)

  def test_exclude_first(self):

    @partial(np.vectorize, excluded={0})
    def f(x, y):
      assert x == 'foo'
      assert y.ndim == 0
      return y

    x = np.arange(3)
    self.assertAllClose(x, f('foo', x), check_dtypes=True)
    self.assertAllClose(x, jax.jit(f, 0)('foo', x), check_dtypes=True)

  def test_exclude_second(self):

    @partial(np.vectorize, excluded={1})
    def f(x, y):
      assert x.ndim == 0
      assert y == 'foo'
      return x

    x = np.arange(3)
    self.assertAllClose(x, f(x, 'foo'), check_dtypes=True)
    self.assertAllClose(x, jax.jit(f, 1)(x, 'foo'), check_dtypes=True)

  def test_exclude_errors(self):
    with self.assertRaisesRegex(
        TypeError, "jax.numpy.vectorize can only exclude"):
      np.vectorize(lambda x: x, excluded={'foo'})

    with self.assertRaisesRegex(
        ValueError, r"excluded=\{-1\} contains negative numbers"):
      np.vectorize(lambda x: x, excluded={-1})

    f = np.vectorize(lambda x: x, excluded={1})
    with self.assertRaisesRegex(
        ValueError, r"excluded=\{1\} is invalid for 1 argument\(s\)"):
      f(1.0)

  def test_bad_inputs(self):
    matmat = np.vectorize(np.dot, signature='(n,m),(m,k)->(n,k)')
    with self.assertRaisesRegex(
        TypeError, "wrong number of positional arguments"):
      matmat(np.zeros((3, 2)))
    with self.assertRaisesRegex(
        ValueError,
        r"input with shape \(2,\) does not have enough dimensions"):
      matmat(np.zeros((2,)), np.zeros((2, 2)))
    with self.assertRaisesRegex(
        ValueError, r"inconsistent size for core dimension 'm'"):
      matmat(np.zeros((2, 3)), np.zeros((4, 5)))

  def test_wrong_output_type(self):
    f = np.vectorize(np.dot, signature='(n,m),(m,k)->(n,k),()')
    with self.assertRaisesRegex(
        TypeError, "output must be a tuple"):
      f(np.zeros((2, 2)), np.zeros((2, 2)))

  def test_wrong_num_outputs(self):
    f = np.vectorize(lambda *args: args, signature='(),()->(),(),()')
    with self.assertRaisesRegex(
        TypeError, "wrong number of output arguments"):
      f(1, 2)

  def test_wrong_output_shape(self):
    f = np.vectorize(np.dot, signature='(n,m),(m,k)->(n)')
    with self.assertRaisesRegex(
        ValueError, r"output shape \(2, 2\) does not match"):
      f(np.zeros((2, 2)), np.zeros((2, 2)))

  def test_inconsistent_output_size(self):
    f = np.vectorize(np.dot, signature='(n,m),(m,k)->(n,n)')
    with self.assertRaisesRegex(
        ValueError, r"inconsistent size for core dimension 'n'"):
      f(np.zeros((2, 3)), np.zeros((3, 4)))


if __name__ == "__main__":
  absltest.main()
