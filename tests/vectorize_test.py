# Copyright 2019 Google LLC
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
"""Tests for Vectorize library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import numpy as np
from jax import test_util as jtu
from jax import random
from jax.experimental.vectorize import vectorize

from jax.config import config
config.parse_flags_with_absl()

matmat = vectorize('(n,m),(m,k)->(n,k)')(np.dot)
matvec = vectorize('(n,m),(m)->(n)')(np.dot)
vecmat = vectorize('(m),(m,k)->(k)')(np.dot)
vecvec = vectorize('(m),(m)->()')(np.dot)

@vectorize('(n)->()')
def magnitude(x):
  return np.dot(x, x)

mean = vectorize('(n)->()')(np.mean)

@vectorize('()->(n)')
def stack_plus_minus(x):
  return np.stack([x, -x])

@vectorize('(n)->(),(n)')
def center(array):
  bias = np.mean(array)
  debiased = array - bias
  return bias, debiased

class VectorizeTest(jtu.JaxTestCase):
  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_leftshape={}_rightshape={}".format(left_shape, right_shape),
          "left_shape": left_shape,
          "right_shape": right_shape,
          "result_shape": result_shape
      } for left_shape, right_shape, result_shape in [
          ((2, 3), (3, 4), (2, 4)),
          ((2, 3), (1, 3, 4), (1, 2, 4)),
          ((5, 2, 3), (1, 3, 4), (5, 2, 4)),
          ((6, 5, 2, 3), (3, 4), (6, 5, 2, 4)),
      ]))
  def test_matmat(self, left_shape, right_shape, result_shape):
    self.assertEqual(matmat(np.zeros(left_shape), np.zeros(right_shape)).shape, result_shape)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_leftshape={}_rightshape={}".format(left_shape, right_shape),
          "left_shape": left_shape,
          "right_shape": right_shape,
          "result_shape": result_shape
      } for left_shape, right_shape, result_shape in [
          ((2, 3), (3,), (2,)),
          ((2, 3), (1, 3), (1, 2)),
          ((4, 2, 3), (1, 3), (4, 2)),
          ((5, 4, 2, 3), (1, 3), (5, 4, 2)),
      ]))
  def test_matvec(self, left_shape, right_shape, result_shape):
    self.assertEqual(matvec(np.zeros(left_shape), np.zeros(right_shape)).shape, result_shape)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_leftshape={}_rightshape={}".format(left_shape, right_shape),
          "left_shape": left_shape,
          "right_shape": right_shape,
          "result_shape": result_shape
      } for left_shape, right_shape, result_shape in [
          ((3,), (3,), ()),
          ((2, 3), (3,), (2,)),
          ((4, 2, 3), (3,), (4, 2)),
      ]))
  def test_vecvec(self, left_shape, right_shape, result_shape):
    self.assertEqual(vecvec(np.zeros(left_shape), np.zeros(right_shape)).shape, result_shape)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_shape={}".format(shape),
          "shape": shape,
          "result_shape": result_shape
      } for shape, result_shape in [
          ((3,), ()),
          ((
              2,
              3,
          ), (2,)),
          ((
              1,
              2,
              3,
          ), (1, 2)),
      ]))
  def test_magnitude(self, shape, result_shape):
    size = 1
    for x in shape:
      size *= x
    self.assertEqual(magnitude(np.arange(size).reshape(shape)).shape, result_shape)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_shape={}".format(shape),
          "shape": shape,
          "result_shape": result_shape
      } for shape, result_shape in [
          ((3,), ()),
          ((2, 3), (2,)),
          ((1, 2, 3, 4), (1, 2, 3)),
      ]))
  def test_mean(self, shape, result_shape):
    self.assertEqual(mean(np.zeros(shape)).shape, result_shape)

  def test_mean_axis(self):
    self.assertEqual(mean(np.zeros((2, 3)), axis=0).shape, (3,))

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_shape={}".format(shape),
          "shape": shape,
          "result_shape": result_shape
      } for shape, result_shape in [
          ((), (2,)),
          ((3,), (
              3,
              2,
          )),
      ]))
  def test_stack_plus_minus(self, shape, result_shape):
    self.assertEqual(stack_plus_minus(np.zeros(shape)).shape, result_shape)

  def test_center(self):
    b, a = center(np.arange(3))
    self.assertEqual(a.shape, (3,))
    self.assertEqual(b.shape, ())
    self.assertAllClose(1.0, b, False)

    X = np.arange(12).reshape((3, 4))
    b, a = center(X, axis=1)
    self.assertEqual(a.shape, (3, 4))
    self.assertEqual(b.shape, (3,))
    self.assertAllClose(np.mean(X, axis=1), b, True)

    b, a = center(X, axis=0)
    self.assertEqual(a.shape, (3, 4))
    self.assertEqual(b.shape, (4,))
    self.assertAllClose(np.mean(X, axis=0), b, True)

if __name__ == "__main__":
  absltest.main()
