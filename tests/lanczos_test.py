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

from absl.testing import absltest

import jax
from jax import lax
import jax.test_util as jtu
from jax.experimental import lanczos
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg


from jax.config import config
config.parse_flags_with_absl()


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)


def basic_lanczos(A, m, q0):
  """Basic Lanczos iteration without reothogonalization."""

  def loop(carray, j):
    V, v, v_prev, beta = carray
    u = A(v) - beta * v_prev
    alpha = _dot(v.conj(), u)
    u = u - alpha * v
    beta = jnp.linalg.norm(u)
    v_prev = v
    v = u / beta
    V = V.at[:, j + 1].set(v)
    return (V, v, v_prev, beta), (alpha, beta)

  q0 = q0 / jnp.linalg.norm(q0)
  Q = jnp.zeros((q0.size, m + 1)).at[:, 0].set(q0)
  (Q, *_), (alpha, beta) = lax.scan(loop, (Q, q0, q0, 0.0), jnp.arange(m))
  return Q, alpha, beta


def basic_lanczos_alt(A, m, q0):
  """Basic Lanczos iteration implemented via Lanczos restart."""
  q0 = q0 / jnp.linalg.norm(q0)
  Q = jnp.zeros((q0.size, m + 1)).at[:, 0].set(q0)
  alpha = jnp.zeros((m,))
  beta = jnp.zeros((m,))
  return lanczos._lanczos_restart(A, 0, m, Q, alpha, beta)


class SolversTest(jtu.JaxTestCase):

  def setUp(self):
    x = np.random.RandomState(0).randn(10, 10)
    self.matrix = x + x.T
    self.matvec = jax.jit(partial(jnp.dot, self.matrix))
    self.q0 = jnp.array(np.random.RandomState(1).randn(10))
    self.w_expected, self.v_expected = jnp.linalg.eigh(self.matrix)
    super().setUp()

  def test_basic_lanczos(self):
    Q, alpha, beta = basic_lanczos(self.matvec, self.q0.size, self.q0)
    T = lanczos._build_arrowhead_matrix(alpha, beta, k=0)
    lambda_, Y = jnp.linalg.eigh(T)
    w_actual = lambda_
    v_actual = Q[:, :-1] @ Y
    # note: error tolerances here (and below) are set based on how well we do
    # in float32 precision
    np.testing.assert_allclose(w_actual, self.w_expected, atol=1e-5)
    np.testing.assert_allclose(abs(v_actual), abs(self.v_expected), atol=1e-4)

  def test_basic_lanczos_consistency(self):
    Q_expected, alpha_expected, beta_expected = basic_lanczos(
        self.matvec, self.q0.size, self.q0)
    Q_actual, alpha_actual, beta_actual = basic_lanczos_alt(
        self.matvec, self.q0.size, self.q0)
    # The last column of Q does not have a well-defined value, since we already
    # produced a basis set.
    np.testing.assert_allclose(Q_expected[:, :-1], Q_actual[:, :-1], atol=2e-4)
    np.testing.assert_allclose(alpha_expected, alpha_actual, atol=5e-6)
    np.testing.assert_allclose(beta_expected[:-1], beta_actual[:-1], atol=5e-6)
    np.testing.assert_allclose(beta_actual[-1], 0.0, atol=5e-6)

  def test_build_arrowhead_matrix(self):
    actual = lanczos._build_arrowhead_matrix(
        jnp.arange(1, 5), -jnp.arange(1, 5), k=2)
    expected = jnp.array([
        [1, 0, -1, 0],
        [0, 2, -2, 0],
        [-1, -2, 3, -3],
        [0, 0, -3, 4],
    ])
    np.testing.assert_array_equal(actual, expected)

  def test_eigsh_smallest_random(self):
    A = scipy.sparse.linalg.LinearOperator(
        shape=(self.q0.size,)*2, dtype=jnp.float64, matvec=self.matvec)
    w_expected, v_expected = scipy.sparse.linalg.eigsh(A, 5, which='SA')

    w_actual, v_actual, info = lanczos.eigsh_smallest(
        self.matvec, self.q0, 5, max_restarts=3, return_info=True)

    np.testing.assert_allclose(w_actual, w_expected, atol=1e-5)
    np.testing.assert_allclose(abs(v_actual), abs(v_expected), atol=1e-5)
    self.assertEqual(info['num_restarts'], 1)

  def test_eigh_smallest_jit(self):

    @jax.jit
    def eigsh_smallest(matrix, q0):
      matvec = partial(jnp.dot, matrix)
      return lanczos.eigsh_smallest(
          matvec, q0, 5, max_restarts=3, return_info=True)

    w_actual, v_actual, info = eigsh_smallest(self.matrix, self.q0)
    np.testing.assert_allclose(w_actual, self.w_expected[:5], atol=1e-5)
    np.testing.assert_allclose(
        abs(v_actual), abs(self.v_expected[:, :5]), atol=1e-5)
    self.assertEqual(info['num_restarts'], 1)

  def test_eigsh_smallest_many_iterations(self):
    tolerance = 1e-6
    w_actual, v_actual, info = lanczos.eigsh_smallest(
        self.matvec, self.q0, 2, inner_iterations=5, max_restarts=100,
        tolerance=tolerance, return_info=True)
    self.assertGreater(info['num_restarts'], 1)
    last_iteration = info['num_restarts'] - 1
    self.assertGreaterEqual(info['num_converged'][last_iteration], 2)
    self.assertTrue((info['saved'] >= info['converged']).all())
    np.testing.assert_array_less(
        info['residual_norm'][last_iteration, :2], tolerance)
    np.testing.assert_allclose(info['precondition'], 0.0, atol=2e-5)
    np.testing.assert_allclose(w_actual, self.w_expected[:2], atol=5e-6)
    np.testing.assert_allclose(
        abs(v_actual), abs(self.v_expected[:, :2]), atol=1e-4)

  def test_quantum_harmonic_oscillator(self):
    x = jnp.linspace(-5, 5, num=200)
    n = x.size
    dx = x[1] - x[0]
    kinetic = 1/dx**2 * (jnp.eye(n) - 1/2 * (jnp.eye(n, k=1) + jnp.eye(n, k=-1)))
    potential = jnp.diag(1/2 * x**2)
    hamiltonian = kinetic + potential

    w_expected, v_expected = jnp.linalg.eigh(hamiltonian)
    np.testing.assert_allclose(w_expected[:5], 0.5 + jnp.arange(5), atol=0.01)

    matvec = partial(jnp.dot, hamiltonian)
    v0 = np.random.RandomState(0).randn(n)
    w_actual, v_actual = lanczos.eigsh_smallest(matvec, v0, num_desired=5)
    np.testing.assert_allclose(w_actual, w_expected[:5], atol=3e-4)
    np.testing.assert_allclose(
        abs(v_actual), abs(v_expected[:, :5]), atol=3e-5)


if __name__ == "__main__":
  absltest.main()
