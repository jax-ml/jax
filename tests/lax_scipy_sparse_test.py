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
from absl.testing import parameterized
from absl.testing import absltest

from jax import jit
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg
from jax import lax
from jax import test_util as jtu
import jax.scipy.sparse.linalg


float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]


def matmul_high_precision(a, b):
  return jnp.matmul(a, b, precision=lax.Precision.HIGHEST)


@jit
def posify(matrix):
  return matmul_high_precision(matrix, matrix.T.conj())


def lax_cg(A, b, M=None, tol=0.0, atol=0.0, **kwargs):
  A = partial(matmul_high_precision, A)
  if M is not None:
    M = partial(matmul_high_precision, M)
  x, _ = jax.scipy.sparse.linalg.cg(A, b, tol=tol, atol=atol, M=M, **kwargs)
  return x


def scipy_cg(A, b, tol=0.0, atol=0.0, **kwargs):
  x, _ = scipy.sparse.linalg.cg(A, b, tol=tol, atol=atol, **kwargs)
  return x


def rand_sym_pos_def(rng, shape, dtype):
  matrix = np.eye(N=shape[0], dtype=dtype) + rng(shape, dtype)
  return matrix @ matrix.T.conj()


class LaxBackedScipyTests(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_preconditioner={}".format(
            jtu.format_shape_dtype_string(shape, dtype),
            preconditioner),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
       "preconditioner": preconditioner}
      for shape in [(4, 4), (7, 7), (32, 32)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]
      for rng_factory in [jtu.rand_default]
      for preconditioner in [None, 'random', 'identity', 'exact']))
  def test_cg_against_scipy(self, shape, dtype, rng_factory, preconditioner):

    rng = rng_factory()
    A = rand_sym_pos_def(rng, shape, dtype)
    b = rng(shape[:1], dtype)

    if preconditioner == 'identity':
      M = np.eye(shape[0], dtype=dtype)
    elif preconditioner == 'random':
      M = np.linalg.inv(rand_sym_pos_def(rng, shape, dtype))
    elif preconditioner == 'exact':
      M = np.linalg.inv(A)
    else:
      M = None

    def args_maker():
      return A, b

    self._CheckAgainstNumpy(
        partial(scipy_cg, M=M, maxiter=1),
        partial(lax_cg, M=M, maxiter=1),
        args_maker,
        check_dtypes=True,
        tol=3e-5)

    self._CheckAgainstNumpy(
        partial(scipy_cg, M=M, maxiter=3),
        partial(lax_cg, M=M, maxiter=3),
        args_maker,
        check_dtypes=True,
        tol=1e-4)

    self._CheckAgainstNumpy(
        np.linalg.solve,
        partial(lax_cg, M=M, atol=1e-6),
        args_maker,
        check_dtypes=True,
        tol=2e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(2, 2)]
      for dtype in float_types
      for rng_factory in [jtu.rand_default]))
  def test_cg_as_solve(self, shape, dtype, rng_factory):

    rng = rng_factory()
    a = rng(shape, dtype)
    b = rng(shape[:1], dtype)

    expected = np.linalg.solve(posify(a), b)
    actual = lax_cg(posify(a), b)
    self.assertAllClose(expected, actual, check_dtypes=True)

    actual = jit(lax_cg)(posify(a), b)
    self.assertAllClose(expected, actual, check_dtypes=True)

    # numerical gradients are only well defined if ``a`` is guaranteed to be
    # positive definite.
    jtu.check_grads(
        lambda x, y: lax_cg(posify(x), y),
        (a, b), order=2, rtol=1e-2)

  def test_cg_ndarray(self):
    A = lambda x: 2 * x
    b = jnp.arange(9.0).reshape((3, 3))
    expected = b / 2
    actual, _ = jax.scipy.sparse.linalg.cg(A, b)
    self.assertAllClose(expected, actual, check_dtypes=True)

  def test_cg_pytree(self):
    A = lambda x: {"a": x["a"] + 0.5 * x["b"], "b": 0.5 * x["a"] + x["b"]}
    b = {"a": 1.0, "b": -4.0}
    expected = {"a": 4.0, "b": -6.0}
    actual, _ = jax.scipy.sparse.linalg.cg(A, b)
    self.assertEqual(expected.keys(), actual.keys())
    self.assertAlmostEqual(expected["a"], actual["a"])
    self.assertAlmostEqual(expected["b"], actual["b"])

  def test_cg_errors(self):
    A = lambda x: x
    b = jnp.zeros((2, 1))
    x0 = jnp.zeros((2,))
    with self.assertRaisesRegexp(
        ValueError, "x0 and b must have matching shape"):
      jax.scipy.sparse.linalg.cg(A, b, x0)


if __name__ == "__main__":
  absltest.main()
