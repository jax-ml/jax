# Copyright 2020 The JAX Authors.
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
import unittest

from absl.testing import absltest
import numpy as np
import scipy.sparse.linalg

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
from jax import jit
from jax import lax
from jax.tree_util import register_pytree_node_class

import jax._src.scipy.sparse.linalg as sp_linalg
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu

config.parse_flags_with_absl()


float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex


def matmul_high_precision(a, b):
  return jnp.matmul(a, b, precision=lax.Precision.HIGHEST)


@jit
def posify(matrix):
  return matmul_high_precision(matrix, matrix.T.conj())


def solver(func, A, b, M=None, atol=0.0, **kwargs):
  x, _ = func(A, b, atol=atol, M=M, **kwargs)
  return x


lax_cg = partial(solver, jax.scipy.sparse.linalg.cg)
lax_gmres = partial(solver, jax.scipy.sparse.linalg.gmres)
lax_bicgstab = partial(solver, jax.scipy.sparse.linalg.bicgstab)
scipy_cg = partial(solver, scipy.sparse.linalg.cg)
scipy_gmres = partial(solver, scipy.sparse.linalg.gmres)
scipy_bicgstab = partial(solver, scipy.sparse.linalg.bicgstab)


def rand_sym_pos_def(rng, shape, dtype):
  matrix = np.eye(N=shape[0], dtype=dtype) + rng(shape, dtype)
  return matrix @ matrix.T.conj()


class CustomOperator:
  def __init__(self, A):
    self.A = A
    self.shape = self.A.shape

  def __matmul__(self, x):
    return self.A @ x


class LaxBackedScipyTests(jtu.JaxTestCase):

  def _fetch_preconditioner(self, preconditioner, A, rng=None):
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
    return M

  @jtu.sample_product(
    shape=[(4, 4), (7, 7)],
    dtype=[np.float64, np.complex128],
    preconditioner=[None, 'identity', 'exact', 'random'],
  )
  def test_cg_against_scipy(self, shape, dtype, preconditioner):
    if not config.enable_x64.value:
      raise unittest.SkipTest("requires x64 mode")

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
        tol=1e-12)

    self._CheckAgainstNumpy(
        partial(scipy_cg, M=M, maxiter=3),
        partial(lax_cg, M=M, maxiter=3),
        args_maker,
        tol=1e-12)

    self._CheckAgainstNumpy(
        np.linalg.solve,
        partial(lax_cg, M=M, atol=1e-10),
        args_maker,
        tol=1e-6)

  @jtu.sample_product(
    shape=[(2, 2)],
    dtype=float_types + complex_types,
  )
  def test_cg_as_solve(self, shape, dtype):

    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    b = rng(shape[:1], dtype)

    expected = np.linalg.solve(posify(a), b)
    actual = lax_cg(posify(a), b)
    self.assertAllClose(expected, actual, atol=1e-5, rtol=1e-5)

    actual = jit(lax_cg)(posify(a), b)
    self.assertAllClose(expected, actual, atol=1e-5, rtol=1e-5)

    # numerical gradients are only well defined if ``a`` is guaranteed to be
    # positive definite.
    jtu.check_grads(
        lambda x, y: lax_cg(posify(x), y),
        (a, b), order=2, rtol=2e-1)

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

  @jtu.skip_on_devices('tpu')
  def test_cg_matmul(self):
    A = CustomOperator(2 * jnp.eye(3))
    b = jnp.arange(9.0).reshape(3, 3)
    expected = b / 2
    actual, _ = jax.scipy.sparse.linalg.cg(A, b)
    self.assertAllClose(expected, actual)

  def test_cg_errors(self):
    A = lambda x: x
    b = jnp.zeros((2,))
    with self.assertRaisesRegex(
        ValueError, "x0 and b must have matching tree structure"):
      jax.scipy.sparse.linalg.cg(A, {'x': b}, {'y': b})
    with self.assertRaisesRegex(
        ValueError, "x0 and b must have matching shape"):
      jax.scipy.sparse.linalg.cg(A, b, b[:, np.newaxis])
    with self.assertRaisesRegex(ValueError, "must be a square matrix"):
      jax.scipy.sparse.linalg.cg(jnp.zeros((3, 2)), jnp.zeros((2,)))
    with self.assertRaisesRegex(
        TypeError, "linear operator must be either a function or ndarray"):
      jax.scipy.sparse.linalg.cg([[1]], jnp.zeros((1,)))

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

  def test_cg_weak_types(self):
    x, _ = jax.scipy.sparse.linalg.bicgstab(lambda x: x, 1.0)
    self.assertTrue(dtypes.is_weakly_typed(x))

  # BICGSTAB
  @jtu.sample_product(
    shape=[(5, 5)],
    dtype=[np.float64, np.complex128],
    preconditioner=[None, 'identity', 'exact', 'random'],
  )
  def test_bicgstab_against_scipy(
      self, shape, dtype, preconditioner):
    if not config.enable_x64.value:
      raise unittest.SkipTest("requires x64 mode")

    rng = jtu.rand_default(self.rng())
    A = rng(shape, dtype)
    b = rng(shape[:1], dtype)
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)

    def args_maker():
      return A, b

    self._CheckAgainstNumpy(
        partial(scipy_bicgstab, M=M, maxiter=1),
        partial(lax_bicgstab, M=M, maxiter=1),
        args_maker,
        tol=1e-5)

    self._CheckAgainstNumpy(
        partial(scipy_bicgstab, M=M, maxiter=2),
        partial(lax_bicgstab, M=M, maxiter=2),
        args_maker,
        tol=1e-4)

    self._CheckAgainstNumpy(
        partial(scipy_bicgstab, M=M, maxiter=1),
        partial(lax_bicgstab, M=M, maxiter=1),
        args_maker,
        tol=1e-4)

    self._CheckAgainstNumpy(
        np.linalg.solve,
        partial(lax_bicgstab, M=M, atol=1e-6),
        args_maker,
        tol=1e-4)

  @jtu.sample_product(
    shape=[(2, 2), (7, 7)],
    dtype=float_types + complex_types,
    preconditioner=[None, 'identity', 'exact'],
  )
  @jtu.skip_on_devices("gpu")
  def test_bicgstab_on_identity_system(self, shape, dtype, preconditioner):
    A = jnp.eye(shape[1], dtype=dtype)
    solution = jnp.ones(shape[1], dtype=dtype)
    rng = jtu.rand_default(self.rng())
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)
    b = matmul_high_precision(A, solution)
    tol = shape[0] * float(jnp.finfo(dtype).eps)
    x, info = jax.scipy.sparse.linalg.bicgstab(A, b, tol=tol, atol=tol,
                                               M=M)
    using_x64 = solution.dtype.kind in {np.float64, np.complex128}
    solution_tol = 1e-8 if using_x64 else 1e-4
    self.assertAllClose(x, solution, atol=solution_tol, rtol=solution_tol)

  @jtu.sample_product(
    shape=[(2, 2), (4, 4)],
    dtype=float_types + complex_types,
    preconditioner=[None, 'identity', 'exact'],
  )
  @jtu.skip_on_devices("gpu")
  def test_bicgstab_on_random_system(self, shape, dtype, preconditioner):
    rng = jtu.rand_default(self.rng())
    A = rng(shape, dtype)
    solution = rng(shape[1:], dtype)
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)
    b = matmul_high_precision(A, solution)
    tol = shape[0] * float(jnp.finfo(A.dtype).eps)
    x, info = jax.scipy.sparse.linalg.bicgstab(A, b, tol=tol, atol=tol, M=M)
    using_x64 = solution.dtype.kind in {np.float64, np.complex128}
    solution_tol = 1e-8 if using_x64 else 1e-4
    self.assertAllClose(x, solution, atol=solution_tol, rtol=solution_tol)
    # solve = lambda A, b: jax.scipy.sparse.linalg.bicgstab(A, b)[0]
    # jtu.check_grads(solve, (A, b), order=1, rtol=3e-1)


  def test_bicgstab_pytree(self):
    A = lambda x: {"a": x["a"] + 0.5 * x["b"], "b": 0.5 * x["a"] + x["b"]}
    b = {"a": 1.0, "b": -4.0}
    expected = {"a": 4.0, "b": -6.0}
    actual, _ = jax.scipy.sparse.linalg.bicgstab(A, b)
    self.assertEqual(expected.keys(), actual.keys())
    self.assertAlmostEqual(expected["a"], actual["a"], places=5)
    self.assertAlmostEqual(expected["b"], actual["b"], places=5)

  def test_bicgstab_weak_types(self):
    x, _ = jax.scipy.sparse.linalg.bicgstab(lambda x: x, 1.0)
    self.assertTrue(dtypes.is_weakly_typed(x))

  @jtu.skip_on_devices('tpu')
  def test_bicgstab_matmul(self):
    A = CustomOperator(2 * jnp.eye(3))
    b = jnp.arange(9.0).reshape(3, 3)
    expected = b / 2
    actual, _ = jax.scipy.sparse.linalg.bicgstab(A, b)
    self.assertAllClose(expected, actual)

  @jax.default_matmul_precision("highest")
  def test_bicgstab_numerical_stability_regression(self):
    """Regression test for issue #32978.

    Tests BiCGStab numerical stability with matrix structures that can cause
    breakdown in the p_i update step. The original formula
    beta * (p - omega * q) could lead to catastrophic cancellation on GPU,
    while the reformulated version with intermediate gamma variable provides
    better numerical stability.

    This test uses a structure similar to finite element stiffness matrices
    where the issue was originally observed, with tight tolerances and
    multiple iterations to exercise the numerical properties of the algorithm.
    """
    if not config.enable_x64.value:
      raise unittest.SkipTest("requires x64 mode")

    # Use float64 precision as in the original issue report
    rng = jtu.rand_default(self.rng())
    n = 50

    # Create a symmetric positive definite matrix similar to FEM problems.
    # Use a structure that requires multiple BiCGStab iterations to converge.
    A_base = rng((n, n), jnp.float64)
    A = posify(A_base) + 0.1 * jnp.eye(n)

    # Create a non-trivial solution
    solution = jnp.arange(1, n + 1, dtype=np.float64)
    b = matmul_high_precision(A, solution)

    # Solve with tight tolerance to require multiple iterations
    x, _ = jax.scipy.sparse.linalg.bicgstab(
        A, b, tol=1e-10, atol=1e-10, maxiter=200
    )

    # Verify solution accuracy - if the solver experienced numerical breakdown,
    # this check would fail due to poor convergence
    self.assertAllClose(x, solution, rtol=1e-5, atol=1e-5)

  # GMRES
  @jtu.sample_product(
    shape=[(3, 3)],
    dtype=[np.float64, np.complex128],
    preconditioner=[None, 'identity', 'exact', 'random'],
    solve_method=['incremental', 'batched'],
  )
  def test_gmres_against_scipy(
      self, shape, dtype, preconditioner, solve_method):
    if not config.enable_x64.value:
      raise unittest.SkipTest("requires x64 mode")

    rng = jtu.rand_default(self.rng())
    A = rng(shape, dtype)
    b = rng(shape[:1], dtype)
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)

    def args_maker():
      return A, b

    self._CheckAgainstNumpy(
        partial(scipy_gmres, M=M, restart=1, maxiter=1),
        partial(lax_gmres, M=M, restart=1, maxiter=1, solve_method=solve_method),
        args_maker,
        tol=1e-10)

    self._CheckAgainstNumpy(
        partial(scipy_gmres, M=M, restart=1, maxiter=2),
        partial(lax_gmres, M=M, restart=1, maxiter=2, solve_method=solve_method),
        args_maker,
        tol=1e-10)

    self._CheckAgainstNumpy(
        partial(scipy_gmres, M=M, restart=2, maxiter=1),
        partial(lax_gmres, M=M, restart=2, maxiter=1, solve_method=solve_method),
        args_maker,
        tol=1e-9)

    self._CheckAgainstNumpy(
        np.linalg.solve,
        partial(lax_gmres, M=M, atol=1e-6, solve_method=solve_method),
        args_maker,
        tol=1e-8)

  @jtu.sample_product(
    shape=[(2, 2), (7, 7)],
    dtype=float_types + complex_types,
    preconditioner=[None, 'identity', 'exact'],
    solve_method=['batched', 'incremental'],
  )
  @jtu.skip_on_devices("gpu")
  def test_gmres_on_identity_system(self, shape, dtype, preconditioner,
                                    solve_method):
    A = jnp.eye(shape[1], dtype=dtype)

    solution = jnp.ones(shape[1], dtype=dtype)
    rng = jtu.rand_default(self.rng())
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)
    b = matmul_high_precision(A, solution)
    restart = shape[-1]
    tol = shape[0] * float(jnp.finfo(dtype).eps)
    x, info = jax.scipy.sparse.linalg.gmres(A, b, tol=tol, atol=tol,
                                            restart=restart,
                                            M=M, solve_method=solve_method)
    using_x64 = solution.dtype.kind in {np.float64, np.complex128}
    solution_tol = 1e-8 if using_x64 else 1e-4
    self.assertAllClose(x, solution, atol=solution_tol, rtol=solution_tol)

  @jtu.sample_product(
    shape=[(2, 2), (4, 4)],
    dtype=float_types + complex_types,
    preconditioner=[None, 'identity', 'exact'],
    solve_method=['incremental', 'batched'],
  )
  @jtu.skip_on_devices("gpu")
  def test_gmres_on_random_system(self, shape, dtype, preconditioner,
                                  solve_method):
    rng = jtu.rand_default(self.rng())
    A = rng(shape, dtype)

    solution = rng(shape[1:], dtype)
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)
    b = matmul_high_precision(A, solution)
    restart = shape[-1]
    tol = shape[0] * float(jnp.finfo(A.dtype).eps)
    x, info = jax.scipy.sparse.linalg.gmres(A, b, tol=tol, atol=tol,
                                            restart=restart,
                                            M=M, solve_method=solve_method)
    using_x64 = solution.dtype.kind in {np.float64, np.complex128}
    solution_tol = 1e-8 if using_x64 else 1e-4
    self.assertAllClose(x, solution, atol=solution_tol, rtol=solution_tol)
    # solve = lambda A, b: jax.scipy.sparse.linalg.gmres(A, b)[0]
    # jtu.check_grads(solve, (A, b), order=1, rtol=2e-1)

  def test_gmres_pytree(self):
    A = lambda x: {"a": x["a"] + 0.5 * x["b"], "b": 0.5 * x["a"] + x["b"]}
    b = {"a": 1.0, "b": -4.0}
    expected = {"a": 4.0, "b": -6.0}
    actual, _ = jax.scipy.sparse.linalg.gmres(A, b)
    self.assertEqual(expected.keys(), actual.keys())
    self.assertAlmostEqual(expected["a"], actual["a"], places=5)
    self.assertAlmostEqual(expected["b"], actual["b"], places=5)

  @jax.default_matmul_precision("float32")
  def test_gmres_matmul(self):
    A = CustomOperator(2 * jnp.eye(3))
    b = jnp.arange(9.0).reshape(3, 3)
    expected = b / 2
    actual, _ = jax.scipy.sparse.linalg.gmres(A, b)
    self.assertAllClose(expected, actual)

  @jtu.sample_product(
    shape=[(2, 2), (3, 3)],
    dtype=float_types + complex_types,
    preconditioner=[None, 'identity'],
  )
  def test_gmres_arnoldi_step(self, shape, dtype, preconditioner):
    """
    The Arnoldi decomposition within GMRES is correct.
    """
    if not config.enable_x64.value:
      raise unittest.SkipTest("requires x64 mode")

    rng = jtu.rand_default(self.rng())
    A = rng(shape, dtype)
    M = self._fetch_preconditioner(preconditioner, A, rng=rng)
    if preconditioner is None:
      M = lambda x: x
    else:
      M = partial(matmul_high_precision, M)
    n = shape[0]
    x0 = rng(shape[:1], dtype)
    Q = np.zeros((n, n + 1), dtype=dtype)
    Q[:, 0] = x0 / jnp.linalg.norm(x0).astype(dtype)
    Q = jnp.array(Q)
    H = jnp.eye(n, n + 1, dtype=dtype)

    @jax.tree_util.Partial
    def A_mv(x):
      return matmul_high_precision(A, x)
    for k in range(n):
      Q, H, _ = sp_linalg._kth_arnoldi_iteration(
          k, A_mv, M, Q, H)
    QA = matmul_high_precision(Q[:, :n].conj().T, A)
    QAQ = matmul_high_precision(QA, Q[:, :n])
    self.assertAllClose(QAQ, H.T[:n, :], rtol=2e-5, atol=2e-5)

  def test_gmres_weak_types(self):
    x, _ = jax.scipy.sparse.linalg.gmres(lambda x: x, 1.0)
    self.assertTrue(dtypes.is_weakly_typed(x))

  def test_linear_solve_batching_via_jacrev(self):
    # See https://github.com/jax-ml/jax/issues/14249
    rng = np.random.RandomState(0)
    M = rng.randn(5, 5)
    A = np.dot(M, M.T)
    matvec = lambda x: (jnp.dot(A, x[0]), jnp.dot(A, x[1]))

    def f(b):
      return jax.scipy.sparse.linalg.cg(matvec, (b, b))[0]

    b = rng.randn(5)
    jax.jacrev(f)(b)  # doesn't crash


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
