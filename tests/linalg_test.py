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

from functools import partial
import itertools
import unittest
import sys

import numpy as np
import scipy as osp

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.lib
from jax import jit, grad, jvp, vmap
from jax import lax
from jax import lax_linalg
from jax import numpy as jnp
from jax import scipy as jsp
from jax import test_util as jtu
from jax.lib import xla_bridge
from jax.lib import lapack

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

T = lambda x: np.swapaxes(x, -1, -2)


float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]

def _skip_if_unsupported_type(dtype):
  dtype = np.dtype(dtype)
  if (not FLAGS.jax_enable_x64 and
      dtype in (np.dtype('float64'), np.dtype('complex128'))):
    raise unittest.SkipTest("--jax_enable_x64 is not set")


class NumpyLinalgTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (2, 5, 5), (200, 200), (1000, 0, 0)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_mac_linalg_bug()
  def testCholesky(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    def args_maker():
      factor_shape = shape[:-1] + (2 * shape[-1],)
      a = rng(factor_shape, dtype)
      return [np.matmul(a, jnp.conj(T(a)))]

    if (jnp.issubdtype(dtype, jnp.complexfloating) and
        jtu.device_under_test() == "tpu"):
      self.skipTest("Unimplemented case for complex Cholesky decomposition.")

    self._CheckAgainstNumpy(np.linalg.cholesky, jnp.linalg.cholesky, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.cholesky, args_maker, check_dtypes=True)

    if jnp.finfo(dtype).bits == 64:
      jtu.check_grads(jnp.linalg.cholesky, args_maker(), order=2)

  def testCholeskyGradPrecision(self):
    rng = jtu.rand_default(self.rng())
    a = rng((3, 3), np.float32)
    a = np.dot(a, a.T)
    jtu.assert_dot_precision(
        lax.Precision.HIGHEST, partial(jvp, jnp.linalg.cholesky), (a,), (a,))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_n={}".format(jtu.format_shape_dtype_string((n,n), dtype)),
       "n": n, "dtype": dtype, "rng_factory": rng_factory}
      for n in [0, 4, 5, 25]  # TODO(mattjj): complex64 unstable on large sizes?
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def testDet(self, n, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng((n, n), dtype)]

    self._CheckAgainstNumpy(np.linalg.det, jnp.linalg.det, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.det, args_maker, check_dtypes=True,
                          rtol={np.float64: 1e-13, np.complex128: 1e-13})

  def testDetOfSingularMatrix(self):
    x = jnp.array([[-1., 3./2], [2./3, -1.]], dtype=np.float32)
    self.assertAllClose(np.float32(0), jsp.linalg.det(x), check_dtypes=True)
    
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (3, 3), (2, 4, 4)]
      for dtype in float_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testDetGrad(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    a = rng(shape, dtype)
    jtu.check_grads(jnp.linalg.det, (a,), 2, atol=1e-1, rtol=1e-1)
    # make sure there are no NaNs when a matrix is zero
    if len(shape) == 2:
      pass
      jtu.check_grads(
        jnp.linalg.det, (jnp.zeros_like(a),), 1, atol=1e-1, rtol=1e-1)
    else:
      a[0] = 0
      jtu.check_grads(jnp.linalg.det, (a,), 1, atol=1e-1, rtol=1e-1)

  def testDetGradOfSingularMatrixCorank1(self):
    # Rank 2 matrix with nonzero gradient
    a = jnp.array([[ 50, -30,  45],
                  [-30,  90, -81],
                  [ 45, -81,  81]], dtype=jnp.float32)
    jtu.check_grads(jnp.linalg.det, (a,), 1, atol=1e-1, rtol=1e-1)

  @jtu.skip_on_devices("tpu")  # TODO(mattjj,pfau): nan on tpu, investigate
  def testDetGradOfSingularMatrixCorank2(self):
    # Rank 1 matrix with zero gradient
    b = jnp.array([[ 36, -42,  18],
                  [-42,  49, -21],
                  [ 18, -21,   9]], dtype=jnp.float32)
    jtu.check_grads(jnp.linalg.det, (b,), 1, atol=1e-1, rtol=1e-1)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_m={}_n={}_q={}".format(
            jtu.format_shape_dtype_string((m,), dtype),
            jtu.format_shape_dtype_string((nq[0],), dtype),
            jtu.format_shape_dtype_string(nq[1], dtype)),
       "m": m,
       "nq": nq, "dtype": dtype, "rng_factory": rng_factory}
      for m in [1, 5, 7, 23]
      for nq in zip([2, 4, 6, 36], [(1, 2), (2, 2), (1, 2, 3), (3, 3, 1, 4)])
      for dtype in float_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_mac_linalg_bug()
  def testTensorsolve(self, m, nq, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)

    # According to numpy docs the shapes are as follows:
    # Coefficient tensor (a), of shape b.shape + Q.
    # And prod(Q) == prod(b.shape)
    # Therefore, n = prod(q)
    n, q = nq
    b_shape = (n, m)
    # To accomplish prod(Q) == prod(b.shape) we append the m extra dim
    # to Q shape
    Q = q + (m,)
    args_maker = lambda: [
        rng(b_shape + Q, dtype), # = a
        rng(b_shape, dtype)]     # = b
    a, b = args_maker()
    result = jnp.linalg.tensorsolve(*args_maker())
    self.assertEqual(result.shape, Q)

    self._CheckAgainstNumpy(np.linalg.tensorsolve, 
                            jnp.linalg.tensorsolve, args_maker,
                            check_dtypes=True,
                            tol={np.float32: 1e-2, np.float64: 1e-3})
    self._CompileAndCheck(jnp.linalg.tensorsolve, 
                          args_maker, check_dtypes=True,
                          rtol={np.float64: 1e-13})

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(0, 0), (1, 1), (3, 3), (4, 4), (10, 10), (200, 200),
                    (2, 2, 2), (2, 3, 3), (3, 2, 2)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")
  @jtu.skip_on_mac_linalg_bug()
  def testSlogdet(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(np.linalg.slogdet, jnp.linalg.slogdet, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.slogdet, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (5, 5), (2, 7, 7)]
      for dtype in float_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testSlogdetGrad(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    a = rng(shape, dtype)
    jtu.check_grads(jnp.linalg.slogdet, (a,), 2, atol=1e-1, rtol=1e-1)

  def testIssue1213(self):
    for n in range(5):
      mat = jnp.array([np.diag(np.ones([5], dtype=np.float32))*(-.01)] * 2)
      args_maker = lambda: [mat]
      self._CheckAgainstNumpy(np.linalg.slogdet, jnp.linalg.slogdet, args_maker,
                              check_dtypes=True, tol=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
           jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(0, 0), (4, 4), (5, 5), (50, 50), (2, 6, 6)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  # TODO(phawkins): enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.skip_on_devices("gpu", "tpu")
  @jtu.skip_on_mac_linalg_bug()
  def testEig(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    n = shape[-1]
    args_maker = lambda: [rng(shape, dtype)]

    # Norm, adjusted for dimension and type.
    def norm(x):
      norm = np.linalg.norm(x, axis=(-2, -1))
      return norm / ((n + 1) * jnp.finfo(dtype).eps)

    a, = args_maker()
    w, v = jnp.linalg.eig(a)
    self.assertTrue(np.all(norm(np.matmul(a, v) - w[..., None, :] * v) < 100))

    self._CompileAndCheck(partial(jnp.linalg.eig), args_maker,
                          check_dtypes=True, rtol=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (5, 5), (50, 50), (2, 10, 10)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  # TODO(phawkins): enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.skip_on_devices("gpu", "tpu")
  @jtu.skip_on_mac_linalg_bug()
  def testEigGrad(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    a = rng(shape, dtype)

    da = rng(shape, dtype)
    (w, v), (dw, dv) = jvp(jnp.linalg.eig, (a,), (da,))
    zero = jnp.zeros(v.shape[:-2] + v.shape[-1:], dtype)
    self.assertAllClose((jnp.conj(v) * dv).sum(-2), zero, check_dtypes=False)

    jtu.check_grads(jnp.linalg.eig, (a,), 2, rtol=1e-1)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
           jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(4, 4), (5, 5), (50, 50)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  # TODO: enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.skip_on_devices("gpu", "tpu")
  @jtu.skip_on_mac_linalg_bug()
  def testEigvals(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    n = shape[-1]
    args_maker = lambda: [rng(shape, dtype)]
    a, = args_maker()
    w1, _ = jnp.linalg.eig(a)
    w2 = jnp.linalg.eigvals(a)
    self.assertAllClose(w1, w2, check_dtypes=True)

  @jtu.skip_on_devices("gpu", "tpu")
  def testEigvalsInf(self):
    # https://github.com/google/jax/issues/2661
    x = jnp.array([[jnp.inf]], jnp.float64)
    self.assertTrue(jnp.all(jnp.isnan(jnp.linalg.eigvals(x))))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (5, 5)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("gpu", "tpu")
  def testEigBatching(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    shape = (10,) + shape
    args = rng(shape, dtype)
    ws, vs = vmap(jnp.linalg.eig)(args)
    self.assertTrue(np.all(np.linalg.norm(
        np.matmul(args, vs) - ws[..., None, :] * vs) < 1e-3))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_n={}_lower={}".format(
           jtu.format_shape_dtype_string((n,n), dtype), lower),
       "n": n, "dtype": dtype, "lower": lower, "rng_factory": rng_factory}
      for n in [0, 4, 5, 50]
      for dtype in float_types + complex_types
      for lower in [False, True]
      for rng_factory in [jtu.rand_default]))
  def testEigh(self, n, dtype, lower, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    tol = 30
    if jtu.device_under_test() == "tpu":
      if jnp.issubdtype(dtype, np.complexfloating):
        raise unittest.SkipTest("No complex eigh on TPU")
      # TODO(phawkins): this tolerance is unpleasantly high.
      tol = 1500
    args_maker = lambda: [rng((n, n), dtype)]

    uplo = "L" if lower else "U"

    # Norm, adjusted for dimension and type.
    def norm(x):
      norm = np.linalg.norm(x, axis=(-2, -1))
      return norm / ((n + 1) * jnp.finfo(dtype).eps)

    a, = args_maker()
    a = (a + np.conj(a.T)) / 2
    w, v = jnp.linalg.eigh(np.tril(a) if lower else np.triu(a),
                          UPLO=uplo, symmetrize_input=False)
    self.assertTrue(norm(np.eye(n) - np.matmul(np.conj(T(v)), v)) < 5)
    self.assertTrue(norm(np.matmul(a, v) - w * v) < tol)

    self._CompileAndCheck(partial(jnp.linalg.eigh, UPLO=uplo), args_maker,
                          check_dtypes=True, rtol=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
           jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(4, 4), (5, 5), (50, 50)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def testEigvalsh(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    if jtu.device_under_test() == "tpu":
      if jnp.issubdtype(dtype, jnp.complexfloating):
        raise unittest.SkipTest("No complex eigh on TPU")
    n = shape[-1]
    def args_maker():
      a = rng((n, n), dtype)
      a = (a + np.conj(a.T)) / 2
      return [a]
    self._CheckAgainstNumpy(np.linalg.eigvalsh, jnp.linalg.eigvalsh, args_maker,
                            check_dtypes=True, tol=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_lower={}".format(jtu.format_shape_dtype_string(shape, dtype),
                                   lower),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "lower":lower}
      for shape in [(1, 1), (4, 4), (5, 5), (50, 50), (2, 10, 10)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]
      for lower in [True, False]))
  def testEighGrad(self, shape, dtype, rng_factory, lower):
    rng = rng_factory(self.rng())
    self.skipTest("Test fails with numeric errors.")
    uplo = "L" if lower else "U"
    a = rng(shape, dtype)
    a = (a + np.conj(T(a))) / 2
    ones = np.ones((a.shape[-1], a.shape[-1]), dtype=dtype)
    a *= np.tril(ones) if lower else np.triu(ones)
    # Gradient checks will fail without symmetrization as the eigh jvp rule
    # is only correct for tangents in the symmetric subspace, whereas the
    # checker checks against unconstrained (co)tangents.
    if dtype not in complex_types:
      f = partial(jnp.linalg.eigh, UPLO=uplo, symmetrize_input=True)
    else:  # only check eigenvalue grads for complex matrices
      f = lambda a: partial(jnp.linalg.eigh, UPLO=uplo, symmetrize_input=True)(a)[0]
    jtu.check_grads(f, (a,), 2, rtol=1e-1)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_lower={}".format(jtu.format_shape_dtype_string(shape, dtype),
                                   lower),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "lower":lower, "eps":eps}
      for shape in [(1, 1), (4, 4), (5, 5), (50, 50)]
      for dtype in complex_types
      for rng_factory in [jtu.rand_default]
      for lower in [True, False]
      for eps in [1e-4]))
  # TODO(phawkins): enable when there is a complex eigendecomposition
  # implementation for TPU.
  @jtu.skip_on_devices("tpu")
  def testEighGradVectorComplex(self, shape, dtype, rng_factory, lower, eps):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    # Special case to test for complex eigenvector grad correctness.
    # Exact eigenvector coordinate gradients are hard to test numerically for complex
    # eigensystem solvers given the extra degrees of per-eigenvector phase freedom.
    # Instead, we numerically verify the eigensystem properties on the perturbed
    # eigenvectors.  You only ever want to optimize eigenvector directions, not coordinates!
    uplo = "L" if lower else "U"
    a = rng(shape, dtype)
    a = (a + np.conj(a.T)) / 2
    a = np.tril(a) if lower else np.triu(a)
    a_dot = eps * rng(shape, dtype)
    a_dot = (a_dot + np.conj(a_dot.T)) / 2
    a_dot = np.tril(a_dot) if lower else np.triu(a_dot)
    # evaluate eigenvector gradient and groundtruth eigensystem for perturbed input matrix
    f = partial(jnp.linalg.eigh, UPLO=uplo)
    (w, v), (dw, dv) = jvp(f, primals=(a,), tangents=(a_dot,))
    new_a = a + a_dot
    new_w, new_v = f(new_a)
    new_a = (new_a + np.conj(new_a.T)) / 2
    # Assert rtol eigenvalue delta between perturbed eigenvectors vs new true eigenvalues.
    RTOL=1e-2
    assert np.max(
      np.abs((np.diag(np.dot(np.conj((v+dv).T), np.dot(new_a,(v+dv)))) - new_w) / new_w)) < RTOL
    # Redundant to above, but also assert rtol for eigenvector property with new true eigenvalues.
    assert np.max(
      np.linalg.norm(np.abs(new_w*(v+dv) - np.dot(new_a, (v+dv))), axis=0) /
      np.linalg.norm(np.abs(new_w*(v+dv)), axis=0)
    ) < RTOL

  def testEighGradPrecision(self):
    rng = jtu.rand_default(self.rng())
    a = rng((3, 3), np.float32)
    jtu.assert_dot_precision(
        lax.Precision.HIGHEST, partial(jvp, jnp.linalg.eigh), (a,), (a,))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (5, 5)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def testEighBatching(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    if (jtu.device_under_test() == "tpu" and
        jnp.issubdtype(dtype, np.complexfloating)):
      raise unittest.SkipTest("No complex eigh on TPU")
    shape = (10,) + shape
    args = rng(shape, dtype)
    args = (args + np.conj(T(args))) / 2
    ws, vs = vmap(jsp.linalg.eigh)(args)
    self.assertTrue(np.all(np.linalg.norm(
        np.matmul(args, vs) - ws[..., None, :] * vs) < 1e-3))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_ord={}_axis={}_keepdims={}".format(
         jtu.format_shape_dtype_string(shape, dtype), ord, axis, keepdims),
       "shape": shape, "dtype": dtype, "axis": axis, "keepdims": keepdims,
       "ord": ord, "rng_factory": rng_factory}
      for axis, shape in [
        (None, (1,)), (None, (7,)), (None, (5, 8)),
        (0, (9,)), (0, (4, 5)), ((1,), (10, 7, 3)), ((-2,), (4, 8)),
        (-1, (6, 3)), ((0, 2), (3, 4, 5)), ((2, 0), (7, 8, 9)),
        (None, (7, 8, 11))]
      for keepdims in [False, True]
      for ord in (
          [None] if axis is None and len(shape) > 2
          else [None, 0, 1, 2, 3, -1, -2, -3, jnp.inf, -jnp.inf]
          if (axis is None and len(shape) == 1) or
             isinstance(axis, int) or
             (isinstance(axis, tuple) and len(axis) == 1)
          else [None, 'fro', 1, 2, -1, -2, jnp.inf, -jnp.inf, 'nuc'])
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))  # type: ignore
  def testNorm(self, shape, dtype, ord, axis, keepdims, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    if (ord in ('nuc', 2, -2) and (
        jtu.device_under_test() != "cpu" or
        (isinstance(axis, tuple) and len(axis) == 2))):
      raise unittest.SkipTest("No adequate SVD implementation available")

    args_maker = lambda: [rng(shape, dtype)]
    np_fn = partial(np.linalg.norm, ord=ord, axis=axis, keepdims=keepdims)
    np_fn = partial(jnp.linalg.norm, ord=ord, axis=axis, keepdims=keepdims)
    self._CheckAgainstNumpy(np_fn, np_fn, args_maker,
                            check_dtypes=False, tol=1e-3)
    self._CompileAndCheck(np_fn, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_n={}_full_matrices={}_compute_uv={}".format(
          jtu.format_shape_dtype_string(b + (m, n), dtype), full_matrices,
          compute_uv),
       "b": b, "m": m, "n": n, "dtype": dtype, "full_matrices": full_matrices,
       "compute_uv": compute_uv, "rng_factory": rng_factory}
      for b in [(), (3,), (2, 3)]
      for m in [2, 7, 29, 53]
      for n in [2, 7, 29, 53]
      for dtype in float_types + complex_types
      for full_matrices in [False, True]
      for compute_uv in [False, True]
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")
  def testSVD(self, b, m, n, dtype, full_matrices, compute_uv, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng(b + (m, n), dtype)]

    # Norm, adjusted for dimension and type.
    def norm(x):
      norm = np.linalg.norm(x, axis=(-2, -1))
      return norm / (max(m, n) * jnp.finfo(dtype).eps)

    a, = args_maker()
    out = jnp.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    if compute_uv:
      # Check the reconstructed matrices
      if full_matrices:
        k = min(m, n)
        if m < n:
          self.assertTrue(np.all(
              norm(a - np.matmul(out[1][..., None, :] * out[0], out[2][..., :k, :])) < 50))
        else:
          self.assertTrue(np.all(
              norm(a - np.matmul(out[1][..., None, :] * out[0][..., :, :k], out[2])) < 350))
      else:
        self.assertTrue(np.all(
          norm(a - np.matmul(out[1][..., None, :] * out[0], out[2])) < 350))

      # Check the unitary properties of the singular vector matrices.
      self.assertTrue(np.all(norm(np.eye(out[0].shape[-1]) - np.matmul(np.conj(T(out[0])), out[0])) < 15))
      if m >= n:
        self.assertTrue(np.all(norm(np.eye(out[2].shape[-1]) - np.matmul(np.conj(T(out[2])), out[2])) < 10))
      else:
        self.assertTrue(np.all(norm(np.eye(out[2].shape[-2]) - np.matmul(out[2], np.conj(T(out[2])))) < 20))

    else:
      self.assertTrue(np.allclose(np.linalg.svd(a, compute_uv=False), np.asarray(out), atol=1e-4, rtol=1e-4))

    self._CompileAndCheck(partial(jnp.linalg.svd, full_matrices=full_matrices, compute_uv=compute_uv),
                          args_maker, check_dtypes=True)
    if not (compute_uv and full_matrices):
      svd = partial(jnp.linalg.svd, full_matrices=full_matrices,
                    compute_uv=compute_uv)
      # TODO(phawkins): these tolerances seem very loose.
      jtu.check_jvp(svd, partial(jvp, svd), (a,), rtol=5e-2, atol=2e-1)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_fullmatrices={}".format(
          jtu.format_shape_dtype_string(shape, dtype), full_matrices),
       "shape": shape, "dtype": dtype, "full_matrices": full_matrices,
       "rng_factory": rng_factory}
      for shape in [(1, 1), (3, 3), (3, 4), (2, 10, 5), (2, 200, 100)]
      for dtype in float_types + complex_types
      for full_matrices in [False, True]
      for rng_factory in [jtu.rand_default]))
  def testQr(self, shape, dtype, full_matrices, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    if (jnp.issubdtype(dtype, np.complexfloating) and
        jtu.device_under_test() == "tpu"):
      raise unittest.SkipTest("No complex QR implementation")
    m, n = shape[-2:]

    if full_matrices:
      mode, k = "complete", m
    else:
      mode, k = "reduced", min(m, n)

    a = rng(shape, dtype)
    lq, lr = jnp.linalg.qr(a, mode=mode)

    # np.linalg.qr doesn't support batch dimensions. But it seems like an
    # inevitable extension so we support it in our version.
    nq = np.zeros(shape[:-2] + (m, k), dtype)
    nr = np.zeros(shape[:-2] + (k, n), dtype)
    for index in np.ndindex(*shape[:-2]):
      nq[index], nr[index] = np.linalg.qr(a[index], mode=mode)

    max_rank = max(m, n)

    # Norm, adjusted for dimension and type.
    def norm(x):
      n = np.linalg.norm(x, axis=(-2, -1))
      return n / (max_rank * jnp.finfo(dtype).eps)

    def compare_orthogonal(q1, q2):
      # Q is unique up to sign, so normalize the sign first.
      sum_of_ratios = np.sum(np.divide(q1, q2), axis=-2, keepdims=True)
      phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
      q1 *= phases
      self.assertTrue(np.all(norm(q1 - q2) < 30))

    # Check a ~= qr
    self.assertTrue(np.all(norm(a - np.matmul(lq, lr)) < 30))

    # Compare the first 'k' vectors of Q; the remainder form an arbitrary
    # orthonormal basis for the null space.
    compare_orthogonal(nq[..., :k], lq[..., :k])

    # Check that q is close to unitary.
    self.assertTrue(np.all(
        norm(np.eye(k) -np.matmul(np.conj(T(lq)), lq)) < 5))

    if not full_matrices and m >= n:
        jtu.check_jvp(jnp.linalg.qr, partial(jvp, jnp.linalg.qr), (a,), atol=3e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for shape in [(10, 4, 5), (5, 3, 3), (7, 6, 4)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def testQrBatching(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    args = rng(shape, jnp.float32)
    qs, rs = vmap(jsp.linalg.qr)(args)
    self.assertTrue(np.all(np.linalg.norm(args - np.matmul(qs, rs)) < 1e-3))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_pnorm={}".format(jtu.format_shape_dtype_string(shape, dtype), pnorm),
       "shape": shape, "pnorm": pnorm, "dtype": dtype}
      for shape in [(1, 1), (4, 4), (2, 3, 5), (5, 5, 5), (20, 20), (5, 10)]
      for pnorm in [jnp.inf, -jnp.inf, 1, -1, 2, -2, 'fro']
      for dtype in float_types + complex_types))
  @jtu.skip_on_devices("tpu")  # SVD is not implemented on the TPU backend
  @jtu.skip_on_devices("gpu")  # TODO(#2203): numerical errors
  def testCond(self, shape, pnorm, dtype):
    _skip_if_unsupported_type(dtype)

    def gen_mat():
      # arr_gen = jtu.rand_some_nan(self.rng())
      arr_gen = jtu.rand_default(self.rng())
      res = arr_gen(shape, dtype)
      return res

    def args_gen(p):
      def _args_gen():
        return [gen_mat(), p]
      return _args_gen

    args_maker = args_gen(pnorm)
    if pnorm not in [2, -2] and len(set(shape[-2:])) != 1:
      with self.assertRaises(np.linalg.LinAlgError):
        jnp.linalg.cond(*args_maker())
    else:
      self._CheckAgainstNumpy(np.linalg.cond, jnp.linalg.cond, args_maker,
                              check_dtypes=False, tol=1e-3)
      partial_norm = partial(jnp.linalg.cond, p=pnorm)
      self._CompileAndCheck(partial_norm, lambda: [gen_mat()],
                            check_dtypes=False, rtol=1e-03, atol=1e-03)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (200, 200), (7, 7, 7, 7)]
      for dtype in float_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_mac_linalg_bug()
  def testTensorinv(self, shape, dtype, rng_factory):
    _skip_if_unsupported_type(dtype)
    rng = rng_factory(self.rng())

    def tensor_maker():
      invertible = False
      while not invertible:
        a = rng(shape, dtype)
        try:
          np.linalg.inv(a)
          invertible = True
        except np.linalg.LinAlgError:
          pass
      return a

    args_maker = lambda: [tensor_maker(), int(np.floor(len(shape) / 2))]
    self._CheckAgainstNumpy(np.linalg.tensorinv, jnp.linalg.tensorinv, args_maker,
                            check_dtypes=False, tol=1e-3)
    partial_inv = partial(jnp.linalg.tensorinv, ind=int(np.floor(len(shape) / 2)))
    self._CompileAndCheck(partial_inv, lambda: [tensor_maker()], check_dtypes=False, rtol=1e-03, atol=1e-03)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs={}_rhs={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [
          ((1, 1), (1, 1)),
          ((4, 4), (4,)),
          ((8, 8), (8, 4)),
          ((1, 2, 2), (3, 2)),
          ((2, 1, 3, 3), (2, 4, 3, 4)),
      ]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def testSolve(self, lhs_shape, rhs_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(np.linalg.solve, jnp.linalg.solve, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.solve, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (2, 5, 5), (200, 200), (5, 5, 5)]
      for dtype in float_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_mac_linalg_bug()
  def testInv(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    if jtu.device_under_test() == "gpu" and shape == (200, 200):
      raise unittest.SkipTest("Test is flaky on GPU")

    def args_maker():
      invertible = False
      while not invertible:
        a = rng(shape, dtype)
        try:
          np.linalg.inv(a)
          invertible = True
        except np.linalg.LinAlgError:
          pass
      return [a]

    self._CheckAgainstNumpy(np.linalg.inv, jnp.linalg.inv, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.inv, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 4), (2, 70, 7), (2000, 7), (7, 1000), (70, 7, 2)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")  # SVD is not implemented on the TPU backend
  @jtu.skip_on_mac_linalg_bug()
  def testPinv(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(np.linalg.pinv, jnp.linalg.pinv, args_maker,
                            check_dtypes=True, tol=1e-2)
    self._CompileAndCheck(jnp.linalg.pinv, args_maker, check_dtypes=True)
    # TODO(phawkins): 1e-1 seems like a very loose tolerance.
    jtu.check_grads(jnp.linalg.pinv, args_maker(), 2, rtol=1e-1, atol=2e-1)

  @jtu.skip_on_devices("tpu")  # SVD is not implemented on the TPU backend
  def testPinvGradIssue2792(self):
    def f(p):
      a = jnp.array([[0., 0.],[-p, 1.]], jnp.float32) * 1 / (1 + p**2)
      return jnp.linalg.pinv(a)
    j = jax.jacobian(f)(jnp.float32(2.))
    self.assertAllClose(jnp.array([[0., -1.], [ 0., 0.]], jnp.float32), j,
                        check_dtypes=True)

    expected = jnp.array([[[[-1., 0.], [ 0., 0.]], [[0., -1.], [0.,  0.]]],
                         [[[0.,  0.], [-1., 0.]], [[0.,  0.], [0., -1.]]]],
                         dtype=jnp.float32)
    self.assertAllClose(
      expected, jax.jacobian(jnp.linalg.pinv)(jnp.eye(2, dtype=jnp.float32)),
      check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}".format(
          jtu.format_shape_dtype_string(shape, dtype), n),
       "shape": shape, "dtype": dtype, "n": n, "rng_factory": rng_factory}
      for shape in [(1, 1), (2, 2), (4, 4), (5, 5),
                    (1, 2, 2), (2, 3, 3), (2, 5, 5)]
      for dtype in float_types + complex_types
      for n in [-5, -2, -1, 0, 1, 2, 3, 4, 5, 10]
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")  # TODO(b/149870255): Bug in XLA:TPU?.
  def testMatrixPower(self, shape, dtype, n, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng(shape, dtype)]
    tol = 1e-1 if jtu.device_under_test() == "tpu" else 1e-3
    self._CheckAgainstNumpy(partial(np.linalg.matrix_power, n=n),
                            partial(jnp.linalg.matrix_power, n=n),
                            args_maker, check_dtypes=True, tol=tol)
    self._CompileAndCheck(partial(jnp.linalg.matrix_power, n=n), args_maker,
                          check_dtypes=True, rtol=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
           jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(3, ), (1, 2), (8, 5), (4, 4), (5, 5), (50, 50)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")
  def testMatrixRank(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    n = shape[-1]
    args_maker = lambda: [rng(shape, dtype)]
    a, = args_maker()
    self._CheckAgainstNumpy(np.linalg.matrix_rank, jnp.linalg.matrix_rank,
                            args_maker, check_dtypes=False, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.matrix_rank, args_maker,
                          check_dtypes=False, rtol=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shapes={}".format(
           ','.join(jtu.format_shape_dtype_string(s, dtype) for s in shapes)),
       "shapes": shapes, "dtype": dtype, "rng_factory": rng_factory}
      for shapes in [
        [(3, ), (3, 1)],  # quick-out codepath
        [(1, 3), (3, 5), (5, 2)],  # multi_dot_three codepath
        [(1, 3), (3, 5), (5, 2), (2, 7), (7, )]  # dynamic programming codepath
      ]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def testMultiDot(self, shapes, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [[rng(shape, dtype) for shape in shapes]]

    np_fun = np.linalg.multi_dot
    jnp_fun = partial(jnp.linalg.multi_dot, precision=lax.Precision.HIGHEST)
    tol = {np.float32: 1e-4, np.float64: 1e-10,
           np.complex64: 1e-4, np.complex128: 1e-10}

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, 
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          atol=tol, rtol=tol)

  # Regression test for incorrect type for eigenvalues of a complex matrix.
  @jtu.skip_on_devices("tpu")  # TODO(phawkins): No complex eigh implementation on TPU.
  def testIssue669(self):
    def test(x):
      val, vec = jnp.linalg.eigh(x)
      return jnp.real(jnp.sum(val))

    grad_test_jc = jit(grad(jit(test)))
    xc = np.eye(3, dtype=np.complex)
    self.assertAllClose(xc, grad_test_jc(xc), check_dtypes=True)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testIssue1151(self):
    A = jnp.array(np.random.randn(100, 3, 3), dtype=jnp.float32)
    b = jnp.array(np.random.randn(100, 3), dtype=jnp.float32)
    x = jnp.linalg.solve(A, b)
    self.assertAllClose(vmap(jnp.dot)(A, x), b, atol=1e-3, rtol=1e-2,
                        check_dtypes=True)
    jac0 = jax.jacobian(jnp.linalg.solve, argnums=0)(A, b)
    jac1 = jax.jacobian(jnp.linalg.solve, argnums=1)(A, b)
    jac0 = jax.jacobian(jnp.linalg.solve, argnums=0)(A[0], b[0])
    jac1 = jax.jacobian(jnp.linalg.solve, argnums=1)(A[0], b[0])

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testIssue1383(self):
    seed = jax.random.PRNGKey(0)
    tmp = jax.random.uniform(seed, (2,2))
    a = jnp.dot(tmp, tmp.T)

    def f(inp):
      val, vec = jnp.linalg.eigh(inp)
      return jnp.dot(jnp.dot(vec, inp), vec.T)

    grad_func = jax.jacfwd(f)
    hess_func = jax.jacfwd(grad_func)
    cube_func = jax.jacfwd(hess_func)
    self.assertFalse(np.any(np.isnan(cube_func(a))))


class ScipyLinalgTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_i={}".format(i), "args": args}
      for i, args in enumerate([
        (),
        (1,),
        (7, -2),
        (3, 4, 5),
        (np.ones((3, 4), dtype=jnp.float_), 5,
         np.random.randn(5, 2).astype(jnp.float_)),
      ])))
  def testBlockDiag(self, args):
    args_maker = lambda: args
    self._CheckAgainstNumpy(osp.linalg.block_diag, jsp.linalg.block_diag,
                            args_maker, check_dtypes=True)
    self._CompileAndCheck(jsp.linalg.block_diag, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 5), (10, 5), (50, 50)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_mac_linalg_bug()
  def testLu(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng(shape, dtype)]
    x, = args_maker()
    p, l, u = jsp.linalg.lu(x)
    self.assertAllClose(x, np.matmul(p, np.matmul(l, u)), check_dtypes=True,
                        rtol={np.float32: 1e-3, np.float64: 1e-12,
                              np.complex64: 1e-3, np.complex128: 1e-12})
    self._CompileAndCheck(jsp.linalg.lu, args_maker, check_dtypes=True)

  def testLuOfSingularMatrix(self):
    x = jnp.array([[-1., 3./2], [2./3, -1.]], dtype=np.float32)
    p, l, u = jsp.linalg.lu(x)
    self.assertAllClose(x, np.matmul(p, np.matmul(l, u)), check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(1, 1), (4, 5), (10, 5), (10, 10), (6, 7, 7)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")  # TODO(phawkins): precision problems on TPU.
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testLuGrad(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    a = rng(shape, dtype)
    lu = vmap(jsp.linalg.lu) if len(shape) > 2 else jsp.linalg.lu
    jtu.check_grads(lu, (a,), 2, atol=5e-2, rtol=3e-1)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(4, 5), (6, 5)]
      for dtype in [jnp.float32]
      for rng_factory in [jtu.rand_default]))
  def testLuBatching(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args = [rng(shape, jnp.float32) for _ in range(10)]
    expected = list(osp.linalg.lu(x) for x in args)
    ps = np.stack([out[0] for out in expected])
    ls = np.stack([out[1] for out in expected])
    us = np.stack([out[2] for out in expected])

    actual_ps, actual_ls, actual_us = vmap(jsp.linalg.lu)(jnp.stack(args))
    self.assertAllClose(ps, actual_ps, check_dtypes=True)
    self.assertAllClose(ls, actual_ls, check_dtypes=True)
    self.assertAllClose(us, actual_us, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_n={}".format(jtu.format_shape_dtype_string((n,n), dtype)),
       "n": n, "dtype": dtype, "rng_factory": rng_factory}
      for n in [1, 4, 5, 200]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_mac_linalg_bug()
  def testLuFactor(self, n, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng((n, n), dtype)]

    x, = args_maker()
    lu, piv = jsp.linalg.lu_factor(x)
    l = np.tril(lu, -1) + np.eye(n, dtype=dtype)
    u = np.triu(lu)
    for i in range(n):
      x[[i, piv[i]],] = x[[piv[i], i],]
    self.assertAllClose(x, np.matmul(l, u), check_dtypes=True, rtol=1e-3,
                        atol=1e-3)
    self._CompileAndCheck(jsp.linalg.lu_factor, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs={}_rhs={}_trans={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           trans),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "trans": trans, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [
          ((1, 1), (1, 1)),
          ((4, 4), (4,)),
          ((8, 8), (8, 4)),
      ]
      for trans in [0, 1, 2]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("cpu")  # TODO(frostig): Test fails on CPU sometimes
  def testLuSolve(self, lhs_shape, rhs_shape, dtype, trans, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    osp_fun = lambda lu, piv, rhs: osp.linalg.lu_solve((lu, piv), rhs, trans=trans)
    jsp_fun = lambda lu, piv, rhs: jsp.linalg.lu_solve((lu, piv), rhs, trans=trans)

    def args_maker():
      a = rng(lhs_shape, dtype)
      lu, piv = osp.linalg.lu_factor(a)
      return [lu, piv, rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs={}_rhs={}_sym_pos={}_lower={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           sym_pos, lower),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "sym_pos": sym_pos, "lower": lower, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [
          ((1, 1), (1, 1)),
          ((4, 4), (4,)),
          ((8, 8), (8, 4)),
      ]
      for sym_pos, lower in [
        (False, False),
        (True, False),
        (True, True),
      ]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def testSolve(self, lhs_shape, rhs_shape, dtype, sym_pos, lower, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    if (sym_pos and jnp.issubdtype(dtype, np.complexfloating) and
        jtu.device_under_test() == "tpu"):
      raise unittest.SkipTest(
        "Complex Cholesky decomposition not implemented on TPU")
    osp_fun = lambda lhs, rhs: osp.linalg.solve(lhs, rhs, sym_pos=sym_pos, lower=lower)
    jsp_fun = lambda lhs, rhs: jsp.linalg.solve(lhs, rhs, sym_pos=sym_pos, lower=lower)

    def args_maker():
      a = rng(lhs_shape, dtype)
      if sym_pos:
        a = np.matmul(a, np.conj(T(a)))
        a = np.tril(a) if lower else np.triu(a)
      return [a, rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            check_dtypes=True, tol=1e-3)
    self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs={}_rhs={}_lower={}_transposea={}_unit_diagonal={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           lower, transpose_a, unit_diagonal),
       "lower": lower, "transpose_a": transpose_a,
       "unit_diagonal": unit_diagonal, "lhs_shape": lhs_shape,
       "rhs_shape": rhs_shape, "dtype": dtype, "rng_factory": rng_factory}
      for lower in [False, True]
      for transpose_a in [False, True]
      for unit_diagonal in [False, True]
      for lhs_shape, rhs_shape in [
          ((4, 4), (4,)),
          ((4, 4), (4, 3)),
          ((2, 8, 8), (2, 8, 10)),
      ]
      for dtype in float_types
      for rng_factory in [jtu.rand_default]))
  def testSolveTriangular(self, lower, transpose_a, unit_diagonal, lhs_shape,
                          rhs_shape, dtype, rng_factory):
    _skip_if_unsupported_type(dtype)
    rng = rng_factory(self.rng())
    k = rng(lhs_shape, dtype)
    l = np.linalg.cholesky(np.matmul(k, T(k))
                            + lhs_shape[-1] * np.eye(lhs_shape[-1]))
    l = l.astype(k.dtype)
    b = rng(rhs_shape, dtype)

    if unit_diagonal:
      a = np.tril(l, -1) + np.eye(lhs_shape[-1], dtype=dtype)
    else:
      a = l
    a = a if lower else T(a)

    inv = np.linalg.inv(T(a) if transpose_a else a).astype(a.dtype)
    if len(lhs_shape) == len(rhs_shape):
      np_ans = np.matmul(inv, b)
    else:
      np_ans = np.einsum("...ij,...j->...i", inv, b)

    # The standard scipy.linalg.solve_triangular doesn't support broadcasting.
    # But it seems like an inevitable extension so we support it.
    ans = jsp.linalg.solve_triangular(
        l if lower else T(l), b, trans=1 if transpose_a else 0, lower=lower,
        unit_diagonal=unit_diagonal)

    self.assertAllClose(np_ans, ans, check_dtypes=True,
                        rtol={np.float32: 1e-4, np.float64: 1e-11})

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_A={}_B={}_lower={}_transposea={}_conja={}_unitdiag={}_leftside={}".format(
           jtu.format_shape_dtype_string(a_shape, dtype),
           jtu.format_shape_dtype_string(b_shape, dtype),
           lower, transpose_a, conjugate_a, unit_diagonal, left_side),
       "lower": lower, "transpose_a": transpose_a, "conjugate_a": conjugate_a,
       "unit_diagonal": unit_diagonal, "left_side": left_side,
       "a_shape": a_shape, "b_shape": b_shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for lower in [False, True]
      for unit_diagonal in [False, True]
      for dtype in float_types + complex_types
      for transpose_a in [False, True]
      for conjugate_a in (
          [False] if jnp.issubdtype(dtype, jnp.floating) else [False, True])
      for left_side, a_shape, b_shape in [
          (False, (4, 4), (4,)),
          (False, (4, 4), (1, 4,)),
          (False, (3, 3), (4, 3)),
          (True, (4, 4), (4,)),
          (True, (4, 4), (4, 1)),
          (True, (4, 4), (4, 3)),
          (True, (2, 8, 8), (2, 8, 10)),
      ]
      for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("tpu")  # TODO(phawkins): Test fails on TPU.
  def testTriangularSolveGrad(
      self, lower, transpose_a, conjugate_a, unit_diagonal, left_side, a_shape,
      b_shape, dtype, rng_factory):
    _skip_if_unsupported_type(dtype)
    rng = rng_factory(self.rng())
    # Test lax_linalg.triangular_solve instead of scipy.linalg.solve_triangular
    # because it exposes more options.
    A = jnp.tril(rng(a_shape, dtype) + 5 * np.eye(a_shape[-1], dtype=dtype))
    A = A if lower else T(A)
    B = rng(b_shape, dtype)
    f = partial(lax_linalg.triangular_solve, lower=lower,
                transpose_a=transpose_a, conjugate_a=conjugate_a,
                unit_diagonal=unit_diagonal, left_side=left_side)
    jtu.check_grads(f, (A, B), 2, rtol=4e-2, eps=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_A={}_B={}_bdim={}_leftside={}".format(
           a_shape, b_shape, bdims, left_side),
       "left_side": left_side, "a_shape": a_shape, "b_shape": b_shape,
       "bdims": bdims}
      for left_side, a_shape, b_shape, bdims in [
          (False, (4, 4), (2, 3, 4,), (None, 0)),
          (False, (2, 4, 4), (2, 2, 3, 4,), (None, 0)),
          (False, (2, 4, 4), (3, 4,), (0, None)),
          (False, (2, 4, 4), (2, 3, 4,), (0, 0)),
          (True, (2, 4, 4), (2, 4, 3), (0, 0)),
          (True, (2, 4, 4), (2, 2, 4, 3), (None, 0)),
      ]))
  def testTriangularSolveBatching(self, left_side, a_shape, b_shape, bdims):
    rng = jtu.rand_default(self.rng())
    A = jnp.tril(rng(a_shape, np.float32)
                + 5 * np.eye(a_shape[-1], dtype=np.float32))
    B = rng(b_shape, np.float32)
    solve = partial(lax_linalg.triangular_solve, lower=True,
                    transpose_a=False, conjugate_a=False,
                    unit_diagonal=False, left_side=left_side)
    X = vmap(solve, bdims)(A, B)
    matmul = partial(jnp.matmul, precision=lax.Precision.HIGHEST)
    Y = matmul(A, X) if left_side else matmul(X, A)
    np.testing.assert_allclose(Y - B, 0, atol=1e-4)

  def testTriangularSolveGradPrecision(self):
    rng = jtu.rand_default(self.rng())
    a = jnp.tril(rng((3, 3), np.float32))
    b = rng((1, 3), np.float32)
    jtu.assert_dot_precision(
        lax.Precision.HIGHEST,
        partial(jvp, lax_linalg.triangular_solve),
        (a, b),
        (a, b))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_n={}".format(jtu.format_shape_dtype_string((n,n), dtype)),
       "n": n, "dtype": dtype, "rng_factory": rng_factory}
      for n in [1, 4, 5, 20, 50, 100]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_small]))
  @jtu.skip_on_mac_linalg_bug()
  def testExpm(self, n, dtype, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    args_maker = lambda: [rng((n, n), dtype)]

    osp_fun = lambda a: osp.linalg.expm(a)
    jsp_fun = lambda a: jsp.linalg.expm(a)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            check_dtypes=True)
    self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=True)

    args_maker_triu = lambda: [np.triu(rng((n, n), dtype))]
    jsp_fun_triu = lambda a: jsp.linalg.expm(a,upper_triangular=True)
    self._CheckAgainstNumpy(osp_fun, jsp_fun_triu, args_maker_triu,
                            check_dtypes=True)
    self._CompileAndCheck(jsp_fun_triu, args_maker_triu, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name":
     "_n={}".format(jtu.format_shape_dtype_string((n,n), dtype)),
     "n": n, "dtype": dtype}
    for n in [1, 4, 5, 20, 50, 100]
    for dtype in float_types + complex_types
  ))
  @jtu.skip_on_mac_linalg_bug()
  def testIssue2131(self, n, dtype):
    args_maker_zeros = lambda: [np.zeros((n, n), dtype)]
    osp_fun = lambda a: osp.linalg.expm(a)
    jsp_fun = lambda a: jsp.linalg.expm(a)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker_zeros,
                            check_dtypes=True)
    self._CompileAndCheck(jsp_fun, args_maker_zeros, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs={}_rhs={}_lower={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype),
          lower),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng_factory": rng_factory, "lower": lower}
      for lhs_shape, rhs_shape in [
          [(1, 1), (1,)],
          [(4, 4), (4,)],
          [(4, 4), (4, 4)],
      ]
      for dtype in float_types
      for lower in [True, False]
      for rng_factory in [jtu.rand_default]))
  def testChoSolve(self, lhs_shape, rhs_shape, dtype, lower, rng_factory):
    rng = rng_factory(self.rng())
    _skip_if_unsupported_type(dtype)
    def args_maker():
      b = rng(rhs_shape, dtype)
      if lower:
        L = np.tril(rng(lhs_shape, dtype))
        return [(L, lower), b]
      else:
        U = np.triu(rng(lhs_shape, dtype))
        return [(U, lower), b]
    self._CheckAgainstNumpy(osp.linalg.cho_solve, jsp.linalg.cho_solve,
                            args_maker, check_dtypes=True, tol=1e-3)


if __name__ == "__main__":
  absltest.main()
