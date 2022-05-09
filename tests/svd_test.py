# Copyright 2022 Google LLC
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

"""Tests for the library of QDWH-based singular value decomposition."""
import functools

import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np
import scipy.linalg as osp_linalg
from jax._src.lax import svd
from jax._src import test_util as jtu

from absl.testing import absltest
from absl.testing import parameterized


config.parse_flags_with_absl()
_JAX_ENABLE_X64 = config.x64_enabled

# Input matrix data type for SvdTest.
_SVD_TEST_DTYPE = np.float64 if _JAX_ENABLE_X64 else np.float32

# Machine epsilon used by SvdTest.
_SVD_TEST_EPS = jnp.finfo(_SVD_TEST_DTYPE).eps

# SvdTest relative tolerance.
_SVD_RTOL = 1E-6 if _JAX_ENABLE_X64 else 1E-2

_MAX_LOG_CONDITION_NUM = 9 if _JAX_ENABLE_X64 else 4


@jtu.with_config(jax_numpy_rank_promotion='allow')
class SvdTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {    # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_by_n={}_log_cond={}_full_matrices={}'.format(
              m, n, log_cond, full_matrices),
          'm': m, 'n': n, 'log_cond': log_cond, 'full_matrices': full_matrices}
      for m, n in zip([2, 8, 10, 20], [4, 6, 10, 18])
      for log_cond in np.linspace(1, _MAX_LOG_CONDITION_NUM, 4)
      for full_matrices in [True, False]))
  @jtu.skip_on_devices("rocm")  # will be fixed on rocm-5.1
  def testSvdWithRectangularInput(self, m, n, log_cond, full_matrices):
    """Tests SVD with rectangular input."""
    with jax.default_matmul_precision('float32'):
      a = np.random.uniform(
          low=0.3, high=0.9, size=(m, n)).astype(_SVD_TEST_DTYPE)
      u, s, v = osp_linalg.svd(a, full_matrices=False)
      cond = 10**log_cond
      s = jnp.linspace(cond, 1, min(m, n))
      a = (u * s) @ v
      a = a + 1j * a

      osp_linalg_fn = functools.partial(
          osp_linalg.svd, full_matrices=full_matrices)
      actual_u, actual_s, actual_v = svd.svd(a, full_matrices=full_matrices)

      k = min(m, n)
      if m > n:
        unitary_u = jnp.real(actual_u.T.conj() @ actual_u)
        unitary_v = jnp.real(actual_v.T.conj() @ actual_v)
        unitary_u_size = m if full_matrices else k
        unitary_v_size = k
      else:
        unitary_u = jnp.real(actual_u @ actual_u.T.conj())
        unitary_v = jnp.real(actual_v @ actual_v.T.conj())
        unitary_u_size = k
        unitary_v_size = n if full_matrices else k

      _, expected_s, _ = osp_linalg_fn(a)

      svd_fn = lambda a: svd.svd(a, full_matrices=full_matrices)
      args_maker = lambda: [a]

      with self.subTest('Test JIT compatibility'):
        self._CompileAndCheck(svd_fn, args_maker)

      with self.subTest('Test unitary u.'):
        self.assertAllClose(np.eye(unitary_u_size), unitary_u, rtol=_SVD_RTOL,
                            atol=2E-3)

      with self.subTest('Test unitary v.'):
        self.assertAllClose(np.eye(unitary_v_size), unitary_v, rtol=_SVD_RTOL,
                            atol=2E-3)

      with self.subTest('Test s.'):
        self.assertAllClose(
            expected_s, jnp.real(actual_s), rtol=_SVD_RTOL, atol=1E-6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name': '_m={}_by_n={}'.format(m, n), 'm': m, 'n': n}
      for m, n in zip([50, 6], [3, 60])))
  def testSvdWithSkinnyTallInput(self, m, n):
    """Tests SVD with skinny and tall input."""
    # Generates a skinny and tall input
    with jax.default_matmul_precision('float32'):
      np.random.seed(1235)
      a = np.random.randn(m, n).astype(_SVD_TEST_DTYPE)
      u, s, v = svd.svd(a, full_matrices=False, hermitian=False)

      relative_diff = np.linalg.norm(a - (u * s) @ v) / np.linalg.norm(a)

      np.testing.assert_almost_equal(relative_diff, 1E-6, decimal=6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {   # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_r={}_log_cond={}'.format(m, r, log_cond),
          'm': m, 'r': r, 'log_cond': log_cond}
      for m, r in zip([8, 8, 8, 10], [3, 5, 7, 9])
      for log_cond in np.linspace(1, 3, 3)))
  @jtu.skip_on_devices("rocm")  # will be fixed on rocm-5.1
  def testSvdWithOnRankDeficientInput(self, m, r, log_cond):
    """Tests SVD with rank-deficient input."""
    with jax.default_matmul_precision('float32'):
      a = jnp.triu(jnp.ones((m, m))).astype(_SVD_TEST_DTYPE)

      # Generates a rank-deficient input.
      u, s, v = jnp.linalg.svd(a, full_matrices=False)
      cond = 10**log_cond
      s = jnp.linspace(cond, 1, m)
      s = s.at[r:m].set(jnp.zeros((m-r,)))
      a = (u * s) @ v

      with jax.default_matmul_precision('float32'):
        u, s, v = svd.svd(a, full_matrices=False, hermitian=False)
      diff = np.linalg.norm(a - (u * s) @ v)

      np.testing.assert_almost_equal(diff, 1E-4, decimal=2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {    # pylint:disable=g-complex-comprehension
          'testcase_name': '_m={}_by_n={}_log_cond={}_full_matrices={}'.format(
              m, n, log_cond, full_matrices),
          'm': m, 'n': n, 'log_cond': log_cond, 'full_matrices': full_matrices}
      for m, n in zip([2, 8, 10, 20], [4, 6, 10, 18])
      for log_cond in np.linspace(1, _MAX_LOG_CONDITION_NUM, 4)
      for full_matrices in [True, False]))
  @jtu.skip_on_devices("rocm")  # will be fixed on rocm-5.1
  def testSingularValues(self, m, n, log_cond, full_matrices):
    """Tests singular values."""
    with jax.default_matmul_precision('float32'):
      a = np.random.uniform(
          low=0.3, high=0.9, size=(m, n)).astype(_SVD_TEST_DTYPE)
      u, s, v = osp_linalg.svd(a, full_matrices=False)
      cond = 10**log_cond
      s = np.linspace(cond, 1, min(m, n))
      a = (u * s) @ v
      a = a + 1j * a

      # Only computes singular values.
      compute_uv = False

      osp_linalg_fn = functools.partial(
          osp_linalg.svd, full_matrices=full_matrices, compute_uv=compute_uv)
      actual_s = svd.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

      expected_s = osp_linalg_fn(a)

      svd_fn = lambda a: svd.svd(a, full_matrices=full_matrices)
      args_maker = lambda: [a]

      with self.subTest('Test JIT compatibility'):
        self._CompileAndCheck(svd_fn, args_maker)

      with self.subTest('Test s.'):
        self.assertAllClose(expected_s, actual_s, rtol=_SVD_RTOL, atol=1E-6)

      with self.subTest('Test non-increasing order.'):
        # Computes `actual_diff[i] = s[i+1] - s[i]`.
        actual_diff = jnp.diff(actual_s, append=0)
        np.testing.assert_array_less(actual_diff, np.zeros_like(actual_diff))

  @parameterized.named_parameters([
      {'testcase_name': f'_m={m}_by_n={n}_full_matrices={full_matrices}_'  # pylint:disable=g-complex-comprehension
                        f'compute_uv={compute_uv}_dtype={dtype}',
       'm': m, 'n': n, 'full_matrices': full_matrices,  # pylint:disable=undefined-variable
       'compute_uv': compute_uv, 'dtype': dtype}  # pylint:disable=undefined-variable
      for m, n in zip([2, 4, 8], [4, 4, 6])
      for full_matrices in [True, False]
      for compute_uv in [True, False]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
  ])
  def testSvdOnZero(self, m, n, full_matrices, compute_uv, dtype):
    """Tests SVD on matrix of all zeros."""
    osp_fun = functools.partial(osp_linalg.svd, full_matrices=full_matrices,
                                compute_uv=compute_uv)
    lax_fun = functools.partial(svd.svd, full_matrices=full_matrices,
                                compute_uv=compute_uv)
    args_maker_svd = lambda: [jnp.zeros((m, n), dtype=dtype)]
    self._CheckAgainstNumpy(osp_fun, lax_fun, args_maker_svd)
    self._CompileAndCheck(lax_fun, args_maker_svd)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
