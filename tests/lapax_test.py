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
"""Tests for the LAPAX linear algebra module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

from jax import jit
from jax import test_util as jtu
from jax.experimental import lapax

from jax.config import config
config.parse_flags_with_absl()

# Primitive

float_types = [onp.float32, onp.float64]

class LapaxTest(jtu.JaxTestCase):
  def testSolveLowerTriangularVec(self):
    npr = onp.random.RandomState(1)
    lhs = onp.tril(npr.randn(3, 3))
    lhs2 = onp.tril(npr.randn(3, 3))
    rhs = npr.randn(3, 1)
    rhs2 = npr.randn(3, 1)

    def check(fun, lhs, rhs):
      a1 = onp.linalg.solve(lhs, rhs)
      a2 = fun(lhs, rhs)
      a3 = fun(lhs, rhs)
      self.assertArraysAllClose(a1, a2, check_dtypes=True)
      self.assertArraysAllClose(a2, a3, check_dtypes=True)

    solve_triangular = lambda a, b: lapax.solve_triangular(a, b, left_side=True, lower=True,
                                                           trans_a=False)

    fun = jit(solve_triangular)
    check(fun, lhs, rhs)
    check(fun, lhs2, rhs2)

  def testSolveLowerTriangularMat(self):
    npr = onp.random.RandomState(1)
    lhs = onp.tril(npr.randn(4, 4))
    lhs2 = onp.tril(npr.randn(4, 4))
    rhs = npr.randn(4, 3)
    rhs2 = npr.randn(4, 3)

    def check(fun, lhs, rhs):
      a1 = onp.linalg.solve(lhs, rhs)
      a2 = fun(lhs, rhs)
      a3 = fun(lhs, rhs)
      self.assertArraysAllClose(a1, a2, check_dtypes=True)
      self.assertArraysAllClose(a2, a3, check_dtypes=True)

    solve_triangular = lambda a, b: lapax.solve_triangular(a, b, left_side=True, lower=True,
                                                           trans_a=False)

    fun = jit(solve_triangular)
    check(fun, lhs, rhs)
    check(fun, lhs2, rhs2)

  def testSolveLowerTriangularBroadcasting(self):
    npr = onp.random.RandomState(1)
    lhs = onp.tril(npr.randn(3, 3, 3))
    lhs2 = onp.tril(npr.randn(3, 3, 3))
    rhs = npr.randn(3, 3, 2)
    rhs2 = npr.randn(3, 3, 2)

    def check(fun, lhs, rhs):
      a1 = onp.linalg.solve(lhs, rhs)
      a2 = fun(lhs, rhs)
      a3 = fun(lhs, rhs)
      self.assertArraysAllClose(a1, a2, check_dtypes=True)
      self.assertArraysAllClose(a2, a3, check_dtypes=True)

    solve_triangular = lambda a, b: lapax.solve_triangular(a, b, left_side=True, lower=True,
                                                           trans_a=False)

    fun = jit(solve_triangular)
    check(fun, lhs, rhs)
    check(fun, lhs2, rhs2)

  def testCholeskyMat(self):
    npr = onp.random.RandomState(0)
    square = lambda rhs: onp.dot(rhs, rhs.T)
    arr = square(npr.randn(4, 4))
    arr2 = square(npr.randn(4, 4))

    def check(fun, arr):
      a1 = onp.linalg.cholesky(arr)
      a2 = fun(arr)
      a3 = fun(arr)
      self.assertArraysAllClose(a1, a2, check_dtypes=True)
      self.assertArraysAllClose(a2, a3, check_dtypes=True)

    fun = jit(lapax.cholesky)
    check(fun, arr)
    check(fun, arr2)

  def testBlockedCholeskyMat(self):
    npr = onp.random.RandomState(0)
    square = lambda rhs: onp.dot(rhs, rhs.T)
    arr = square(npr.randn(11, 11))
    arr2 = square(npr.randn(11, 11))

    chol = lambda x: lapax.cholesky(x, block_size=3)

    def check(fun, arr):
      a1 = onp.linalg.cholesky(arr)
      a2 = fun(arr)
      a3 = fun(arr)
      self.assertArraysAllClose(a1, a2, check_dtypes=True)
      self.assertArraysAllClose(a2, a3, check_dtypes=True)

    fun = jit(chol)
    check(fun, arr)
    check(fun, arr2)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_lhs={}_rhs={}_lower={}_leftside={}_transposea={}".format(
                  jtu.format_shape_dtype_string(lhs_shape, dtype),
                  jtu.format_shape_dtype_string(rhs_shape, dtype), lower, left, transpose_a),
          "lower":
              lower,
          "left_side":
              left,
          "transpose_a":
              transpose_a,
          "lhs_shape":
              lhs_shape,
          "rhs_shape":
              rhs_shape,
          "dtype":
              dtype,
          "rng":
              rng
      } for lower, left, transpose_a in itertools.product([False, True], repeat=3)
                          for lhs_shape, rhs_shape in [
                              ((2, 4, 4), (2, 4, 6) if left else (2, 6, 4)),
                          ] for dtype in float_types for rng in [jtu.rand_default()]))
  def testSolveTriangular(self, lower, left_side, transpose_a, lhs_shape, rhs_shape, dtype, rng):
    # pylint: disable=invalid-name
    T = lambda X: onp.swapaxes(X, -1, -2)
    K = rng(lhs_shape, dtype)
    L = onp.linalg.cholesky(onp.matmul(K, T(K)) + lhs_shape[-1] * onp.eye(lhs_shape[-1]))
    L = L.astype(K.dtype)
    B = rng(rhs_shape, dtype)

    A = L if lower else T(L)
    inv = onp.linalg.inv(T(A) if transpose_a else A)
    np_ans = onp.matmul(inv, B) if left_side else onp.matmul(B, inv)

    lapax_ans = lapax.solve_triangular(L if lower else T(L), B, left_side, lower, transpose_a)

    self.assertAllClose(np_ans, lapax_ans, check_dtypes=False)
    # pylint: enable=invalid-name

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_lhs={}_rhs={}_lower={}_leftside={}_transposea={}".format(
                  jtu.format_shape_dtype_string(lhs_shape, dtype),
                  jtu.format_shape_dtype_string(rhs_shape, dtype), lower, left, transpose_a),
          "lower":
              lower,
          "left_side":
              left,
          "transpose_a":
              transpose_a,
          "lhs_shape":
              lhs_shape,
          "rhs_shape":
              rhs_shape,
          "dtype":
              dtype,
          "rng":
              rng
      } for lower, left, transpose_a in itertools.product([False, True], repeat=3)
                          for lhs_shape, rhs_shape in [
                              ((2, 8, 8), (2, 8, 10) if left else (2, 10, 8)),
                          ] for dtype in float_types for rng in [jtu.rand_default()]))
  def testSolveTriangularBlocked(self, lower, left_side, transpose_a, lhs_shape, rhs_shape, dtype,
                                 rng):
    # pylint: disable=invalid-name
    T = lambda X: onp.swapaxes(X, -1, -2)
    K = rng(lhs_shape, dtype)
    L = onp.linalg.cholesky(onp.matmul(K, T(K)) + lhs_shape[-1] * onp.eye(lhs_shape[-1]))
    L = L.astype(K.dtype)
    B = rng(rhs_shape, dtype)

    A = L if lower else T(L)
    inv = onp.linalg.inv(T(A) if transpose_a else A).astype(A.dtype)
    np_ans = onp.matmul(inv, B) if left_side else onp.matmul(B, inv)

    lapax_ans = lapax.solve_triangular(
        L if lower else T(L), B, left_side, lower, transpose_a, block_size=3)

    self.assertAllClose(np_ans, lapax_ans, check_dtypes=False)
    # pylint: enable=invalid-name

if __name__ == "__main__":
  absltest.main()
