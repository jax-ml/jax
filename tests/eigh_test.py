# Copyright 2026 The JAX Authors.
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
import platform
import unittest

from absl.testing import absltest
import numpy as np
import scipy.linalg

import jax
from jax import jvp, vmap
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import version as jaxlib_version

config.parse_flags_with_absl()

T = lambda x: np.swapaxes(x, -1, -2)

float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex
int_types = jtu.dtypes.all_integer

# (complex) Eigenvectors are only unique up to an arbitrary phase. This makes the gradient
# tests based on finite differences unstable, since perturbing the input matri may cause an
# arbitrary sign flip of one or more of the eigenvectors. To remedy this, we normalize the
# vectors such that the first component has phase 0.
def _normalizing_eigh(H: np.ndarray, lower: bool, symmetrize_input: bool):
  uplo = "L" if lower else "U"
  e, v = jnp.linalg.eigh(H, UPLO=uplo, symmetrize_input=symmetrize_input)
  top_rows = v[..., 0:1, :]
  if np.issubdtype(H.dtype, np.complexfloating):
    angle = -jnp.angle(top_rows)
    phase = lax.complex(jnp.cos(angle), jnp.sin(angle))
  else:
    phase = jnp.sign(top_rows)
  v *= phase
  return e, v

class EighTest(jtu.JaxTestCase):

  @jtu.sample_product(
      n=[0, 4, 5, 50, 512],
      dtype=float_types + complex_types,
      lower=[True, False],
  )
  def testEigh(self, n, dtype, lower):
    if n == 512 and jtu.test_device_matches(["tpu"]):
      self.skipTest("n=512 is slow on TPU")
    rng = jtu.rand_default(self.rng())
    eps = np.finfo(dtype).eps
    args_maker = lambda: [rng((n, n), dtype)]

    uplo = "L" if lower else "U"

    a, = args_maker()
    a = (a + np.conj(a.T)) / 2
    w, v = jnp.linalg.eigh(np.tril(a) if lower else np.triu(a),
                           UPLO=uplo, symmetrize_input=False)
    w = w.astype(v.dtype)
    tol = 2 * n * eps
    self.assertAllClose(
        np.eye(n, dtype=v.dtype),
        np.matmul(np.conj(T(v)), v),
        atol=tol,
        rtol=tol,
    )

    with jax.numpy_rank_promotion('allow'):
      tol = 100 * eps
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v), tol * np.linalg.norm(a)
      )

    self._CompileAndCheck(
        partial(jnp.linalg.eigh, UPLO=uplo), args_maker, rtol=eps
    )

    # Compare eigenvalues against Numpy using double precision. We do not compare
    # eigenvectors because they are not uniquely defined, but the two checks above
    # guarantee that that they satisfy the conditions for being eigenvectors.
    double_type = dtype
    if dtype == np.float32:
      double_type = np.float64
    if dtype == np.complex64:
      double_type = np.complex128
    w_np = np.linalg.eigvalsh(a.astype(double_type))
    tol = 8 * eps
    self.assertAllClose(
        w_np.astype(w.dtype), w, atol=tol * np.linalg.norm(a), rtol=tol
    )

  @jax._src.config.explicit_x64_dtypes("allow")
  @jtu.run_on_devices("gpu")
  @unittest.skip("Needs a large amount of GPU memory, doesn't work in CI")
  def testEighLargeMatrix(self):
    # https://github.com/jax-ml/jax/issues/33062
    n = 16384
    A = jnp.eye(n, dtype=jnp.float64)
    jax.block_until_ready(jax.lax.linalg.eigh(A))

  @jtu.sample_product(
      start=[0, 1, 63, 64, 65, 255],
      end=[1, 63, 64, 65, 256],
  )
  @jtu.run_on_devices("tpu")  # TODO(rmlarsen: enable on other devices)
  def testEighSubsetByIndex(self, start, end):
    if start >= end:
      return
    dtype = np.float32
    n = 256
    rng = jtu.rand_default(self.rng())
    eps = np.finfo(dtype).eps
    args_maker = lambda: [rng((n, n), dtype)]
    subset_by_index = (start, end)
    k = end - start
    (a,) = args_maker()
    a = (a + np.conj(a.T)) / 2

    v, w = lax.linalg.eigh(
        a, symmetrize_input=False, subset_by_index=subset_by_index
    )
    w = w.astype(v.dtype)

    self.assertEqual(v.shape, (n, k))
    self.assertEqual(w.shape, (k,))
    with jax.numpy_rank_promotion("allow"):
      tol = 200 * eps
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v), tol * np.linalg.norm(a)
      )
    tol = 3 * n * eps
    self.assertAllClose(
        np.eye(k, dtype=v.dtype),
        np.matmul(np.conj(T(v)), v),
        atol=tol,
        rtol=tol,
    )

    self._CompileAndCheck(partial(jnp.linalg.eigh), args_maker, rtol=eps)

    # Compare eigenvalues against Numpy. We do not compare eigenvectors because
    # they are not uniquely defined, but the two checks above guarantee that
    # that they satisfy the conditions for being eigenvectors.
    double_type = dtype
    if dtype == np.float32:
      double_type = np.float64
    if dtype == np.complex64:
      double_type = np.complex128
    w_np = np.linalg.eigvalsh(a.astype(double_type))[
        subset_by_index[0] : subset_by_index[1]
    ]
    tol = 20 * eps
    self.assertAllClose(
        w_np.astype(w.dtype), w, atol=tol * np.linalg.norm(a), rtol=tol
    )

  def testEighZeroDiagonal(self):
    a = np.array([[0., -1., -1.,  1.],
                  [-1.,  0.,  1., -1.],
                  [-1.,  1.,  0., -1.],
                  [1., -1., -1.,  0.]], dtype=np.float32)
    w, v = jnp.linalg.eigh(a)
    w = w.astype(v.dtype)
    eps = jnp.finfo(a.dtype).eps
    with jax.numpy_rank_promotion('allow'):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v), 2.5 * eps * np.linalg.norm(a)
      )

  def testEighTinyNorm(self):
    # Skip test on ROCm due to numerical error. Issue #34711
    # TODO(GulsumGudukbay): Unskip once fixed.
    if jtu.is_device_rocm():
      self.skipTest("Skipped on ROCm due to numerical error.")
    rng = jtu.rand_default(self.rng())
    a = rng((300, 300), dtype=np.float32)
    eps = jnp.finfo(a.dtype).eps
    a = eps * (a + np.conj(a.T))
    w, v = jnp.linalg.eigh(a)
    w = w.astype(v.dtype)
    with jax.numpy_rank_promotion("allow"):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v), 80 * eps * np.linalg.norm(a)
      )

  @jtu.sample_product(
      rank=[1, 3, 299],
  )
  def testEighRankDeficient(self, rank):
    rng = jtu.rand_default(self.rng())
    eps = jnp.finfo(np.float32).eps
    a = rng((300, rank), dtype=np.float32)
    a = a @ np.conj(a.T)
    w, v = jnp.linalg.eigh(a)
    w = w.astype(v.dtype)
    with jax.numpy_rank_promotion("allow"):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v),
          85 * eps * np.linalg.norm(a),
      )

  @jtu.sample_product(
    n=[0, 4, 5, 50, 512],
    dtype=float_types + complex_types,
    lower=[True, False],
  )
  def testEighIdentity(self, n, dtype, lower):
    tol = np.finfo(dtype).eps
    uplo = "L" if lower else "U"

    a = jnp.eye(n, dtype=dtype)
    w, v = jnp.linalg.eigh(a, UPLO=uplo, symmetrize_input=False)
    w = w.astype(v.dtype)
    self.assertLessEqual(
        np.linalg.norm(np.eye(n) - np.matmul(np.conj(T(v)), v)), tol
    )
    with jax.numpy_rank_promotion('allow'):
      self.assertLessEqual(np.linalg.norm(np.matmul(a, v) - w * v),
                           tol * np.linalg.norm(a))

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (50, 50)],
    dtype=float_types + complex_types,
  )
  def testEigvalsh(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    n = shape[-1]
    def args_maker():
      a = rng((n, n), dtype)
      a = (a + np.conj(a.T)) / 2
      return [a]
    self._CheckAgainstNumpy(
        np.linalg.eigvalsh, jnp.linalg.eigvalsh, args_maker, tol=2e-5
    )

  @jtu.sample_product(
      shape=[(1, 1), (4, 4), (5, 5), (25, 25), (2, 10, 10)],
      dtype=float_types + complex_types,
      lower=[True, False],
  )
  def testEighGrad(self, shape, dtype, lower):
    if platform.system() == "Windows":
      self.skipTest("Skip on Windows due to tolerance issues.")
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    a = (a + np.conj(T(a))) / 2
    ones = np.ones((a.shape[-1], a.shape[-1]), dtype=dtype)
    a *= np.tril(ones) if lower else np.triu(ones)
    # Gradient checks will fail without symmetrization as the eigh jvp rule
    # is only correct for tangents in the symmetric subspace, whereas the
    # checker checks against unconstrained (co)tangents.
    f = partial(_normalizing_eigh, lower=lower, symmetrize_input=True)
    norm_a = jnp.linalg.norm(a)
    eps = 2e-5 * norm_a
    atol = 5e-3 * norm_a
    rtol = 0.025
    jtu.check_grads(f, (a,), 2, atol=atol, rtol=rtol, eps=eps)

  def testEighGradPrecision(self):
    rng = jtu.rand_default(self.rng())
    a = rng((3, 3), np.float32)
    jtu.assert_dot_precision(
        lax.Precision.HIGHEST, partial(jvp, jnp.linalg.eigh), (a,), (a,))

  def testEighGradRankPromotion(self):
    rng = jtu.rand_default(self.rng())
    a = rng((10, 3, 3), np.float32)
    jvp(jnp.linalg.eigh, (a,), (a,))  # doesn't crash

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (5, 5), (300, 300)],
    dtype=float_types + complex_types,
  )
  def testEighBatching(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    shape = (10,) + shape
    args = rng(shape, dtype)
    args = (args + np.conj(T(args))) / 2
    ws, vs = vmap(jsp.linalg.eigh)(args)
    ws = ws.astype(vs.dtype)
    norm = np.max(np.linalg.norm(np.matmul(args, vs) - ws[..., None, :] * vs))
    self.assertLess(norm, 1.4e-2)


class LaxLinalgEighTest(jtu.JaxTestCase):

  @jtu.sample_product(
    n=[0, 4, 5, 50],
    dtype=float_types + complex_types,
    lower=[True, False],
    sort_eigenvalues=[True, False],
  )
  def testEigh(self, n, dtype, lower, sort_eigenvalues):
    implementations = [
        None,
        lax.linalg.EighImplementation.QR,
        lax.linalg.EighImplementation.JACOBI,
        lax.linalg.EighImplementation.QDWH,
    ]

    for implementation in implementations:
      if (
          implementation == lax.linalg.EighImplementation.QR
          and jtu.test_device_matches(["tpu"])
      ):
        continue
      if (
          implementation == lax.linalg.EighImplementation.JACOBI
          and jtu.test_device_matches(["cpu"])
      ):
        continue
      if (
          implementation == lax.linalg.EighImplementation.QDWH
          and jtu.test_device_matches(["cpu", "gpu"])
      ):
        continue

      rng = jtu.rand_default(self.rng())
      tol = 1e-3
      args_maker = lambda: [rng((n, n), dtype)]

      a, = args_maker()
      a = (a + np.conj(a.T)) / 2
      v, w = lax.linalg.eigh(np.tril(a) if lower else np.triu(a),
                             lower=lower, symmetrize_input=False,
                             sort_eigenvalues=sort_eigenvalues,
                             implementation=implementation)
      w = np.asarray(w)
      v = np.asarray(v)
      self.assertLessEqual(
          np.linalg.norm(np.eye(n) - np.matmul(np.conj(T(v)), v)), 1e-3)
      self.assertLessEqual(np.linalg.norm(np.matmul(a, v) - w * v),
                           tol * np.linalg.norm(a))

      w_expected, v_expected = np.linalg.eigh(np.asarray(a))
      self.assertAllClose(w_expected, w if sort_eigenvalues else np.sort(w),
                          rtol=1e-4, atol=1e-4)

  def run_eigh_tridiagonal_test(self, alpha, beta, rtol=2e-3, multiplier=4, check_eigvecs=True):
    n = alpha.shape[-1]
    # scipy.linalg.eigh_tridiagonal doesn't support complex inputs, so for
    # this we call the slower numpy.linalg.eigh.
    if np.issubdtype(alpha.dtype, np.complexfloating):
      tridiagonal = np.diag(alpha) + np.diag(beta, 1) + np.diag(
          np.conj(beta), -1)
      eigvals_expected, _ = np.linalg.eigh(tridiagonal)
    else:
      eigvals_expected = scipy.linalg.eigh_tridiagonal(
          alpha, beta, eigvals_only=True)

    eigvals = jax.scipy.linalg.eigh_tridiagonal(
        alpha, beta, eigvals_only=True)
    finfo = np.finfo(alpha.dtype)

    atol = 4 * np.sqrt(n) * finfo.eps * np.amax(np.abs(eigvals_expected))
    self.assertAllClose(eigvals_expected, eigvals, atol=atol, rtol=1e-4)

    # TODO: The ROCm 0.10.0 plugin is not yet released. This will be
    # re-enabled for ROCm on the 0.10.0 ROCm plugin release.
    if (jaxlib_version >= (0, 10) and
        not jtu.test_device_matches(["tpu", "rocm"])):
      @jax.jit
      def solve(a, b):
        return jax.scipy.linalg.eigh_tridiagonal(a, b, eigvals_only=False)

      eigvals, eigvecs = solve(alpha, beta)
      self.assertAllClose(eigvals_expected, eigvals, atol=atol, rtol=1e-4)

      if check_eigvecs:
        atol_eigvecs = multiplier * np.sqrt(n) * finfo.eps * np.amax(np.abs(eigvals_expected))

        A = np.diag(np.real(alpha)) + np.diag(beta, 1) + np.diag(np.conj(beta), -1)
        self.assertAllClose(
            A @ eigvecs, eigvecs * eigvals[None, :].astype(eigvecs.dtype),
            atol=atol_eigvecs, rtol=rtol,
        )

  @jtu.sample_product(
    n=[1, 2, 3, 7, 8, 100],
    dtype=float_types + complex_types,
  )
  def testToeplitz(self, n, dtype):
    for a, b in [[2, -1], [1, 0], [0, 1], [-1e10, 1e10], [-1e-10, 1e-10]]:
      alpha = a * np.ones([n], dtype=dtype)
      beta = b * np.ones([n - 1], dtype=dtype)

      self.run_eigh_tridiagonal_test(alpha, beta, check_eigvecs=False)

  @jtu.sample_product(
    n=[1, 2, 3, 7, 8, 100],
    dtype=float_types + complex_types,
  )
  def testRandomUniform(self, n, dtype):
    alpha = jtu.rand_uniform(self.rng())((n,), dtype)
    beta = jtu.rand_uniform(self.rng())((n - 1,), dtype)

    multiplier = 4
    rtol = 2e-3
    if jtu.test_device_matches(["gpu"]):
      if dtype == np.complex64:
        multiplier = 600
        rtol = 5e-3 * np.sqrt(n)
      elif dtype == np.float32:
        multiplier = 600
        rtol = 2e-3 * np.sqrt(n)

    self.run_eigh_tridiagonal_test(alpha, beta, rtol=rtol, multiplier=multiplier)

  @jtu.sample_product(dtype=float_types + complex_types)
  def testSelect(self, dtype):
    n = 5
    alpha = jtu.rand_uniform(self.rng())((n,), dtype)
    beta = jtu.rand_uniform(self.rng())((n - 1,), dtype)
    eigvals_all = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, select="a",
                                                    eigvals_only=True)
    eps = np.finfo(alpha.dtype).eps
    atol = 2 * n * eps
    for first in range(n - 1):
      for last in range(first + 1, n - 1):
        # Check that we get the expected eigenvalues by selecting by
        # index range.
        eigvals_index = jax.scipy.linalg.eigh_tridiagonal(
            alpha, beta, select="i", select_range=(first, last),
            eigvals_only=True)
        self.assertAllClose(
            eigvals_all[first:(last + 1)], eigvals_index, atol=atol)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
