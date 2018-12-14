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

from jax import numpy as np
from jax import scipy
from jax import test_util as jtu
from jax.lib import xla_bridge

from jax.config import config
config.parse_flags_with_absl()

T = lambda x: onp.swapaxes(x, -1, -2)


def float_types():
  return set(onp.dtype(xla_bridge.canonicalize_dtype(dtype))
             for dtype in [onp.float32, onp.float64])


class NumpyLinalgTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng": rng}
      for shape in [(1, 1), (4, 4), (2, 5, 5), (200, 200), (1000, 0, 0)]
      for dtype in float_types()
      for rng in [jtu.rand_default()]))
  def testCholesky(self, shape, dtype, rng):
    def args_maker():
      a = rng(shape, dtype)
      return [onp.matmul(a, T(a))]

    self._CheckAgainstNumpy(onp.linalg.cholesky, np.linalg.cholesky, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(np.linalg.cholesky, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_fullmatrices={}".format(
          jtu.format_shape_dtype_string(shape, dtype), full_matrices),
       "shape": shape, "dtype": dtype, "full_matrices": full_matrices,
       "rng": rng}
      for shape in [(1, 1), (3, 4), (2, 10, 5), (2, 200, 200)]
      for dtype in float_types()
      for full_matrices in [False, True]
      for rng in [jtu.rand_default()]))
  def testQr(self, shape, dtype, full_matrices, rng):
    m, n = shape[-2:]

    if full_matrices:
      mode, k = "complete", m
    else:
      mode, k = "reduced", min(m, n)

    a = rng(shape, dtype)
    lq, lr = np.linalg.qr(a, mode=mode)

    # onp.linalg.qr doesn't support broadcasting. But it seems like an
    # inevitable extension so we support it in our version.
    nq = onp.zeros(shape[:-2] + (m, k), dtype)
    nr = onp.zeros(shape[:-2] + (k, n), dtype)
    for index in onp.ndindex(*shape[:-2]):
      nq[index], nr[index] = onp.linalg.qr(a[index], mode=mode)

    max_rank = max(m, n)

    # Norm, adjusted for dimension and type.
    def norm(x):
      n = onp.linalg.norm(x, axis=(-2, -1))
      return n / (max_rank * onp.finfo(dtype).eps)

    def compare_orthogonal(q1, q2):
      # Q is unique up to sign, so normalize the sign first.
      sum_of_ratios = onp.sum(onp.divide(q1, q2), axis=-2, keepdims=True)
      phases = onp.divide(sum_of_ratios, onp.abs(sum_of_ratios))
      q1 *= phases
      self.assertTrue(onp.all(norm(q1 - q2) < 30))

    # Check a ~= qr
    self.assertTrue(onp.all(norm(a - onp.matmul(lq, lr)) < 30))

    # Compare the first 'k' vectors of Q; the remainder form an arbitrary
    # orthonormal basis for the null space.
    compare_orthogonal(nq[..., :k], lq[..., :k])

    # Check that q is close to unitary.
    self.assertTrue(onp.all(norm(onp.eye(k) - onp.matmul(T(lq), lq)) < 5))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng": rng}
      for shape in [(1, 1), (4, 4), (2, 5, 5), (200, 200), (5, 5, 5)]
      for dtype in float_types()
      for rng in [jtu.rand_default()]))
  def testInv(self, shape, dtype, rng):
    def args_maker():
      invertible = False
      while not invertible:
        a = rng(shape, dtype)
        try:
          onp.linalg.inv(a)
          invertible = True
        except onp.linalg.LinAlgError:
          pass
      return [a]

    self._CheckAgainstNumpy(onp.linalg.inv, np.linalg.inv, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(np.linalg.inv, args_maker, check_dtypes=True)


class ScipyLinalgTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs={}_rhs={}_lower={}_transposea={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           lower, transpose_a),
       "lower": lower, "transpose_a": transpose_a,
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng": rng}
      for lower, transpose_a in itertools.product([False, True], repeat=2)
      for lhs_shape, rhs_shape in [
          ((4, 4), (4,)),
          ((4, 4), (4, 3)),
          ((2, 8, 8), (2, 8, 10)),
      ]
      for dtype in float_types()
      for rng in [jtu.rand_default()]))
  def testSolveTriangularBlocked(self, lower, transpose_a, lhs_shape,
                                 rhs_shape, dtype, rng):
    k = rng(lhs_shape, dtype)
    l = onp.linalg.cholesky(onp.matmul(k, T(k))
                            + lhs_shape[-1] * onp.eye(lhs_shape[-1]))
    l = l.astype(k.dtype)
    b = rng(rhs_shape, dtype)

    a = l if lower else T(l)
    inv = onp.linalg.inv(T(a) if transpose_a else a).astype(a.dtype)
    if len(lhs_shape) == len(rhs_shape):
      onp_ans = onp.matmul(inv, b)
    else:
      onp_ans = onp.einsum("...ij,...j->...i", inv, b)

    # The standard scipy.linalg.solve_triangular doesn't support broadcasting.
    # But it seems like an inevitable extension so we support it.
    ans = scipy.linalg.solve_triangular(
        l if lower else T(l), b, trans=1 if transpose_a else 0, lower=lower)

    self.assertAllClose(onp_ans, ans, check_dtypes=True)


if __name__ == "__main__":
  absltest.main()
