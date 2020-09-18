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
import numpy as np
import scipy.sparse.linalg

from jax import jit
import jax.numpy as jnp
from jax import lax
from jax import test_util as jtu
from jax.tree_util import register_pytree_node_class
import jax.scipy.sparse.linalg

from jax.config import config
config.parse_flags_with_absl()
config.update('jax_enable_x64', True)

float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]


def matmul_high_precision(a, b):
  return jnp.matmul(a, b, precision=lax.Precision.HIGHEST)


@jit
def posify(matrix):
  return matmul_high_precision(matrix, matrix.T.conj())


def lax_cg(A, b, M=None, atol=0.0, **kwargs):
  A = partial(matmul_high_precision, A)
  if M is not None:
    M = partial(matmul_high_precision, M)
  x, _ = jax.scipy.sparse.linalg.cg(A, b, atol=atol, M=M, **kwargs)
  return x


def scipy_cg(A, b, atol=0.0, **kwargs):
  x, _ = scipy.sparse.linalg.cg(A, b, atol=atol, **kwargs)
  return x


def rand_sym_pos_def(rng, shape, dtype):
  matrix = np.eye(N=shape[0], dtype=dtype) + rng(shape, dtype)
  return matrix @ matrix.T.conj()






class LaxBackedScipyTests(jtu.JaxTestCase):
  def _fetch_preconditioner(self, preconditioner, A, rng=None,
                            return_function=False):
    """
    Returns one of various preconditioning matrices depending on the identifier
    `preconditioner' and the input matrix A whose inverse it supposedly
    approximates.
    """
    if preconditioner == 'identity':
      M = np.eye(A.shape[0], dtype=A.dtype)
    elif preconditioner == 'random':
      if rng is None:
        rng = jtu.rand_default(self.rng())
      M = np.linalg.inv(rand_sym_pos_def(rng, A.shape, A.dtype))
    elif preconditioner == 'exact':
      M = np.linalg.inv(A)
    else:
      M = None

    if M is None or not return_function:
      return M
    else:
      return lambda x: jnp.dot(M, x, precision=lax.Precision.HIGHEST)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_preconditioner={}".format(
            jtu.format_shape_dtype_string(shape, dtype),
            preconditioner),
       "shape": shape, "dtype": dtype, "preconditioner": preconditioner}
      for shape in [(4, 4), (7, 7), (32, 32)]
      for dtype in float_types + complex_types
      for preconditioner in [None, 'identity', 'exact']))
  # TODO(#2951): reenable 'random' preconditioner.
  def test_cg_against_scipy(self, shape, dtype, preconditioner):

    rng = jtu.rand_default(self.rng())
    A = rand_sym_pos_def(rng, shape, dtype)
    b = rng(shape[:1], dtype)
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)

    def args_maker():
      return A, b

    self._CheckAgainstNumpy(
        partial(scipy_cg, M=M, maxiter=1),
        partial(lax_cg, M=M, maxiter=1),
        args_maker,
        tol=1e-3)

    # TODO(shoyer,mattjj): I had to loosen the tolerance for complex64[7,7]
    # with preconditioner=random
    self._CheckAgainstNumpy(
        partial(scipy_cg, M=M, maxiter=3),
        partial(lax_cg, M=M, maxiter=3),
        args_maker,
        tol=3e-3)

    self._CheckAgainstNumpy(
        np.linalg.solve,
        partial(lax_cg, M=M, atol=1e-6),
        args_maker,
        tol=2e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(2, 2)]
      for dtype in float_types + complex_types))
  def test_cg_as_solve(self, shape, dtype):

    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    b = rng(shape[:1], dtype)

    expected = np.linalg.solve(posify(a), b)
    actual = lax_cg(posify(a), b)
    self.assertAllClose(expected, actual)

    actual = jit(lax_cg)(posify(a), b)
    self.assertAllClose(expected, actual)

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
    self.assertAllClose(expected, actual)

  def test_cg_pytree(self):
    A = lambda x: {"a": x["a"] + 0.5 * x["b"], "b": 0.5 * x["a"] + x["b"]}
    b = {"a": 1.0, "b": -4.0}
    expected = {"a": 4.0, "b": -6.0}
    actual, _ = jax.scipy.sparse.linalg.cg(A, b)
    self.assertEqual(expected.keys(), actual.keys())
    self.assertAlmostEqual(expected["a"], actual["a"], places=6)
    self.assertAlmostEqual(expected["b"], actual["b"], places=6)

  def test_cg_errors(self):
    A = lambda x: x
    b = jnp.zeros((2,))
    with self.assertRaisesRegex(
        ValueError, "x0 and b must have matching tree structure"):
      jax.scipy.sparse.linalg.cg(A, {'x': b}, {'y': b})
    with self.assertRaisesRegex(
        ValueError, "x0 and b must have matching shape"):
      jax.scipy.sparse.linalg.cg(A, b, b[:, np.newaxis])

  def test_cg_without_pytree_equality(self):

    @register_pytree_node_class
    class MinimalPytree:
      def __init__(self, value):
        self.value = value
      def tree_flatten(self):
        return [self.value], None
      @classmethod
      def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    A = lambda x: MinimalPytree(2 * x.value)
    b = MinimalPytree(jnp.arange(5.0))
    expected = b.value / 2
    actual, _ = jax.scipy.sparse.linalg.cg(A, b)
    self.assertAllClose(expected, actual.value)

  # GMRES
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(2, 2)]
      for dtype in float_types + complex_types))
  def test_gmres_on_small_fixed_problem(self, shape, dtype):
    """
    GMRES gives the right answer for a small fixed system.
    """
    A = jnp.array(([[1, 1], [3, -4]]), dtype=dtype)
    b = jnp.array([3, 2], dtype=dtype)
    x0 = jnp.ones(2, dtype=dtype)
    restart = 2
    maxiter = 1

    @jax.tree_util.Partial
    def A_mv(x):
      return matmul_high_precision(A, x)
    tol = A.size * jnp.finfo(dtype).eps
    x, _ = jax.scipy.sparse.linalg.gmres(A_mv, b, x0=x0, tol=tol, atol=tol,
                                         restart=restart, maxiter=maxiter)
    solution = jnp.array([2., 1.], dtype=dtype)
    self.assertAllClose(solution, x)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_preconditioner={}_qr_mode={}".format(
         jtu.format_shape_dtype_string(shape, dtype),
         preconditioner,
         qr_mode),
      "shape": shape, "dtype": dtype, "preconditioner": preconditioner,
      "qr_mode": qr_mode}
      for shape in [(2, 2), (7, 7)]
      for dtype in float_types + complex_types
      for preconditioner in [None, 'identity', 'exact']
      for qr_mode in [True, False]
      ))
  def test_gmres_on_identity_system(self, shape, dtype, preconditioner,
                                    qr_mode):
    A = jnp.eye(shape[1], dtype=dtype)

    solution = jnp.ones(shape[1], dtype=dtype)
    @jax.tree_util.Partial
    def A_mv(x):
      return matmul_high_precision(A, x)
    rng = jtu.rand_default(self.rng())
    M = self._fetch_preconditioner(preconditioner, A, rng=rng,
                                   return_function=True)
    b = A_mv(solution)
    restart = shape[-1]
    tol = shape[0] * jnp.finfo(dtype).eps
    x, info = jax.scipy.sparse.linalg.gmres(A_mv, b, tol=tol, atol=tol,
                                            restart=restart,
                                            M=M,
                                            qr_mode=qr_mode)
    err = jnp.linalg.norm(solution - x) / jnp.linalg.norm(b)
    rtol = tol*jnp.linalg.norm(b)
    true_tol = max(rtol, tol)
    self.assertLessEqual(err, true_tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_preconditioner={}_qr_mode={}".format(
         jtu.format_shape_dtype_string(shape, dtype),
         preconditioner,
         qr_mode),
      "shape": shape, "dtype": dtype, "preconditioner": preconditioner,
      "qr_mode": qr_mode}
      for shape in [(2, 2), (7, 7)]
      for dtype in float_types + complex_types
      for preconditioner in [None, 'identity', 'exact']
      for qr_mode in [True, False]
      ))
  def test_gmres_on_random_system(self, shape, dtype, preconditioner,
                                  qr_mode):
    rng = jtu.rand_default(self.rng())
    A = rng(shape, dtype)

    solution = rng(shape[1:], dtype)
    @jax.tree_util.Partial
    def A_mv(x):
      return matmul_high_precision(A, x)
    M = self._fetch_preconditioner(preconditioner, A, rng=rng,
                                   return_function=True)
    b = A_mv(solution)
    restart = shape[-1]
    tol = shape[0] * jnp.finfo(dtype).eps
    x, info = jax.scipy.sparse.linalg.gmres(A_mv, b, tol=tol, atol=tol,
                                            restart=restart,
                                            M=M,
                                            qr_mode=qr_mode)
    err = jnp.linalg.norm(solution - x) / jnp.linalg.norm(b)
    rtol = tol*jnp.linalg.norm(b)
    true_tol = max(rtol, tol)
    self.assertLessEqual(err, true_tol)

  def test_gmres_pytree(self):
    A = lambda x: {"a": x["a"] + 0.5 * x["b"], "b": 0.5 * x["a"] + x["b"]}
    b = {"a": 1.0, "b": -4.0}
    expected = {"a": 4.0, "b": -6.0}
    actual, _ = jax.scipy.sparse.linalg.gmres(A, b)
    self.assertEqual(expected.keys(), actual.keys())
    self.assertAlmostEqual(expected["a"], actual["a"], places=6)
    self.assertAlmostEqual(expected["b"], actual["b"], places=6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_preconditioner={}".format(
         jtu.format_shape_dtype_string(shape, dtype),
         preconditioner),
      "shape": shape, "dtype": dtype, "preconditioner": preconditioner}
      for shape in [(2, 2), (7, 7), (32, 32)]
      for dtype in float_types + complex_types
      for preconditioner in [None, 'identity']))
  def test_gmres_arnoldi_step(self, shape, dtype, preconditioner):
    """
    The Arnoldi decomposition within GMRES is correct.
    """
    rng = jtu.rand_default(self.rng())
    A = rng(shape, dtype)
    if preconditioner is None:
      M = lambda x: x
    else:
      M = self._fetch_preconditioner(preconditioner, A, rng=rng,
                                     return_function=True)

    n = shape[0]
    x0 = rng(shape[:1], dtype)
    Q = np.zeros((n, n + 1), dtype=dtype)
    Q[:, 0] = x0/jax.numpy.linalg.norm(x0)
    Q = jnp.array(Q)
    H = jax.numpy.eye(n, n + 1, dtype=dtype)
    tol = A.size*A.size*jax.numpy.finfo(dtype).eps

    @jax.tree_util.Partial
    def A_mv(x):
      return matmul_high_precision(A, x)
    for k in range(n):
      Q, H, _ = jax.scipy.sparse.linalg.kth_arnoldi_iteration(k, A_mv, M, Q, H,
                                                              tol)
    QAQ = matmul_high_precision(Q[:, :n].conj().T, A)
    QAQ = matmul_high_precision(QAQ, Q[:, :n])
    self.assertAllClose(QAQ, H.T[:n, :], rtol=tol, atol=tol)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
