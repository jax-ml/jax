# Copyright 2021 Google LLC
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
# limitations under the License

"""Tests for the library of QDWH-based polar decomposition."""
import functools

from jax.config import config
import jax.numpy as jnp
import numpy as np
import scipy.linalg as osp_linalg
from jax._src.lax import qdwh
from jax._src import test_util as jtu

from absl.testing import absltest
from absl.testing import parameterized


config.parse_flags_with_absl()
_JAX_ENABLE_X64 = config.x64_enabled

# Input matrix data type for QdwhTest.
_QDWH_TEST_DTYPE = np.float64 if _JAX_ENABLE_X64 else np.float32

# Machine epsilon used by QdwhTest.
_QDWH_TEST_EPS = jnp.finfo(_QDWH_TEST_DTYPE).eps

# Largest log10 value of condition numbers used by QdwhTest.
_MAX_LOG_CONDITION_NUM = np.log10(int(1 / _QDWH_TEST_EPS))


def _check_symmetry(x: jnp.ndarray) -> bool:
  """Check if the array is symmetric."""
  m, n = x.shape
  eps = jnp.finfo(x.dtype).eps
  tol = 50.0 * eps
  is_symmetric = False
  if m == n:
    if np.linalg.norm(x - x.T.conj()) / np.linalg.norm(x) < tol:
      is_symmetric = True

  return is_symmetric

def _compute_relative_diff(actual, expected):
  """Computes relative difference between two matrices."""
  return np.linalg.norm(actual - expected) / np.linalg.norm(expected)

_dot = functools.partial(jnp.dot, precision="highest")


class QdwhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {    # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_by_n={}_log_cond={}'.format(m, n, log_cond),
          'm': m, 'n': n, 'log_cond': log_cond}
      for m, n in zip([8, 10, 20], [6, 10, 18])
      for log_cond in np.linspace(1, _MAX_LOG_CONDITION_NUM, 4)))
  def testQdwhUnconvergedAfterMaxNumberIterations(
      self, m, n, log_cond):
    """Tests unconvergence after maximum number of iterations."""
    a = jnp.triu(jnp.ones((m, n)))
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.expand_dims(jnp.linspace(cond, 1, min(m, n)), range(u.ndim - 1))
    a = (u * s) @ v
    is_symmetric = _check_symmetry(a)
    max_iterations = 2

    _, _, actual_num_iterations, is_converged = qdwh.qdwh(
        a, is_symmetric, max_iterations)

    with self.subTest('Number of iterations.'):
      self.assertEqual(max_iterations, actual_num_iterations)

    with self.subTest('Converged.'):
      self.assertFalse(is_converged)

  @parameterized.named_parameters(jtu.cases_from_list(
      {    # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_by_n={}_log_cond={}'.format(m, n, log_cond),
          'm': m, 'n': n, 'log_cond': log_cond}
      for m, n in zip([8, 10, 20], [6, 10, 18])
      for log_cond in np.linspace(1, _MAX_LOG_CONDITION_NUM, 4)))
  # TODO(tianjianlu): Fails on A100 GPU.
  @jtu.skip_on_devices("gpu")
  def testQdwhWithUpperTriangularInputAllOnes(self, m, n, log_cond):
    """Tests qdwh with upper triangular input of all ones."""
    a = jnp.triu(jnp.ones((m, n))).astype(_QDWH_TEST_DTYPE)
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.expand_dims(jnp.linspace(cond, 1, min(m, n)), range(u.ndim - 1))
    a = (u * s) @ v
    is_symmetric = _check_symmetry(a)
    max_iterations = 10

    actual_u, actual_h, _, _ = qdwh.qdwh(a, is_symmetric, max_iterations)
    expected_u, expected_h = osp_linalg.polar(a)

    # Sets the test tolerance.
    rtol = 1E6 * _QDWH_TEST_EPS

    with self.subTest('Test u.'):
      relative_diff_u = _compute_relative_diff(actual_u, expected_u)
      np.testing.assert_almost_equal(relative_diff_u, 1E-6, decimal=5)

    with self.subTest('Test h.'):
      relative_diff_h = _compute_relative_diff(actual_h, expected_h)
      np.testing.assert_almost_equal(relative_diff_h, 1E-6, decimal=5)

    with self.subTest('Test u.dot(h).'):
      a_round_trip = _dot(actual_u, actual_h)
      relative_diff_a = _compute_relative_diff(a_round_trip, a)
      np.testing.assert_almost_equal(relative_diff_a, 1E-6, decimal=5)

    with self.subTest('Test orthogonality.'):
      actual_results = _dot(actual_u.T, actual_u)
      expected_results = np.eye(n)
      self.assertAllClose(
          actual_results, expected_results, rtol=rtol, atol=1E-5)

  @parameterized.named_parameters(jtu.cases_from_list(
      {  # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_by_n={}_log_cond={}'.format(
              m, n, log_cond),
          'm': m, 'n': n, 'log_cond': log_cond}
      for m, n in zip([6, 8], [6, 4])
      for log_cond in np.linspace(1, 4, 4)))
  def testQdwhWithRandomMatrix(self, m, n, log_cond):
    """Tests qdwh with random input."""
    rng = jtu.rand_uniform(self.rng(), low=0.3, high=0.9)
    a = rng((m, n), _QDWH_TEST_DTYPE)
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.expand_dims(jnp.linspace(cond, 1, min(m, n)), range(u.ndim - 1))
    a = (u * s) @ v
    is_symmetric = _check_symmetry(a)
    max_iterations = 10

    def lsp_linalg_fn(a):
      u, h, _, _ = qdwh.qdwh(
          a, is_symmetric=is_symmetric, max_iterations=max_iterations)
      return u, h

    args_maker = lambda: [a]

    # Sets the test tolerance.
    rtol = 1E6 * _QDWH_TEST_EPS

    with self.subTest('Test JIT compatibility'):
      self._CompileAndCheck(lsp_linalg_fn, args_maker)

    with self.subTest('Test against numpy.'):
      self._CheckAgainstNumpy(osp_linalg.polar, lsp_linalg_fn, args_maker,
                              rtol=rtol, atol=1E-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {   # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_by_n={}_log_cond={}'.format(
              m, n, log_cond),
          'm': m, 'n': n, 'log_cond': log_cond}
      for m, n in zip([10, 12], [10, 12])
      for log_cond in np.linspace(1, 4, 4)))
  # TODO(tianjianlu): Fails on A100 GPU.
  @jtu.skip_on_devices("gpu")
  def testQdwhWithOnRankDeficientInput(self, m, n, log_cond):
    """Tests qdwh with rank-deficient input."""
    a = jnp.triu(jnp.ones((m, n))).astype(_QDWH_TEST_DTYPE)

    # Generates a rank-deficient input.
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.linspace(cond, 1, min(m, n))
    s = jnp.expand_dims(s.at[-1].set(0), range(u.ndim - 1))
    a = (u * s) @ v

    is_symmetric = _check_symmetry(a)
    max_iterations = 10
    actual_u, actual_h, _, _ = qdwh.qdwh(a, is_symmetric, max_iterations)
    _, expected_h = osp_linalg.polar(a)

    # Sets the test tolerance.
    rtol = 1E6 * _QDWH_TEST_EPS

    # For rank-deficient matrix, `u` is not unique.
    with self.subTest('Test h.'):
      relative_diff_h = _compute_relative_diff(actual_h, expected_h)
      np.testing.assert_almost_equal(relative_diff_h, 1E-6, decimal=5)

    with self.subTest('Test u.dot(h).'):
      a_round_trip = _dot(actual_u, actual_h)
      relative_diff_a = _compute_relative_diff(a_round_trip, a)
      np.testing.assert_almost_equal(relative_diff_a, 1E-6, decimal=5)

    with self.subTest('Test orthogonality.'):
      actual_results = _dot(actual_u.T, actual_u)
      expected_results = np.eye(n)
      self.assertAllClose(
          actual_results, expected_results, rtol=rtol, atol=1E-3)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
