"""Test for polar.py."""
import numpy as np
import pytest

from jax import lax
import jax.numpy as jnp
from jax._src.scipy import qdwh
import scipy as sp


shapes = [(12, 12), (16, 12), (16, 16), (128, 128)]  # Input shapes
n_zero_svs = [0,]  # Number of zero singular values.
degeneracy = [0, 4]  # The middle singular value will be
                     #  repeated this many times.
geometric_spectrum = [False, True]  # False: linear spectrum; True: geometric.
max_sv = [0.1, 1.0, 10.]  # The largest singular value.
nonzero_condition_number = [10, 100, 1000, 10000] # The smallest nonzero
                                                  # singular value will differ
                                                  # from the largest by this
                                                  # factor.
precision = [lax.Precision.HIGHEST, ]  # TPU matmul precision.


def _initialize(shape, n_zero_svs, degeneracy, geometric_spectrum, max_sv,
                nonzero_condition_number):
  n_rows, n_cols = shape
  min_dim = min(shape)
  left_vecs = np.random.randn(n_rows, min_dim).astype(np.float64)
  left_vecs, _ = np.linalg.qr(left_vecs)
  right_vecs = np.random.randn(n_cols, min_dim).astype(np.float64)
  right_vecs, _ = np.linalg.qr(right_vecs)

  min_nonzero_sv = max_sv / nonzero_condition_number
  num_nonzero_svs = min_dim - n_zero_svs
  if geometric_spectrum:
    nonzero_svs = np.geomspace(min_nonzero_sv, max_sv, num=num_nonzero_svs,
                               dtype=np.float64)
  else:
    nonzero_svs = np.linspace(min_nonzero_sv, max_sv, num=num_nonzero_svs,
                              dtype=np.float64)
  half_point = n_zero_svs // 2
  for i in range(half_point, half_point + degeneracy):
    nonzero_svs[i] = nonzero_svs[half_point]
  svs = np.zeros(min(shape), dtype=np.float64)
  svs[n_zero_svs:] = nonzero_svs
  svs = svs[::-1]

  result = np.dot(left_vecs * svs, right_vecs.conj().T).astype(np.float32)
  result = jnp.array(result)
  spectrum = jnp.array(svs.astype(np.float32))
  return result, spectrum


@pytest.mark.parametrize("n_zero_svs", n_zero_svs)
@pytest.mark.parametrize("degeneracy", degeneracy)
@pytest.mark.parametrize("geometric_spectrum", geometric_spectrum)
@pytest.mark.parametrize("max_sv", max_sv)
@pytest.mark.parametrize("nonzero_condition_number", nonzero_condition_number)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("precision", precision)
def test_polar_qdwh(shape, precision, n_zero_svs, degeneracy,
                    geometric_spectrum, max_sv, nonzero_condition_number):
  np.random.seed(10)
  matrix, _ = _initialize(
    shape, n_zero_svs, degeneracy, geometric_spectrum, max_sv,
    nonzero_condition_number)
  _, _, l0 = qdwh._initialize_qdwh(matrix)
  unitary, posdef, j_qr, j_chol = qdwh.polar(matrix, precision=precision)

  if shape[0] >= shape[1]:
    should_be_eye = jnp.matmul(unitary.conj().T, unitary,
                               precision=lax.Precision.HIGHEST)
  else:
    should_be_eye = jnp.matmul(unitary, unitary.conj().T,
                               precision=lax.Precision.HIGHEST)
  #TODO: Account for precision
  tol = 10 * nonzero_condition_number * jnp.finfo(matrix.dtype).eps
  recon = jnp.matmul(unitary, posdef, precision=lax.Precision.HIGHEST)
  np.testing.assert_allclose(matrix, recon, atol=10 * tol)

  eye_mat = jnp.eye(should_be_eye.shape[0], dtype=should_be_eye.dtype)
  np.testing.assert_allclose(eye_mat, should_be_eye, atol=10 * tol)

  np.testing.assert_allclose(posdef, posdef.conj().T, atol=tol)

  ev, _ = jnp.linalg.eigh(posdef)
  ev = ev[jnp.abs(ev) > tol]
  negative_ev = jnp.sum(ev < 0.)
  assert negative_ev == 0.

  np.testing.assert_allclose(matrix, recon, atol=10 * tol)

  if n_zero_svs == 0:
    polar_factor, _ = sp.linalg.polar(matrix)
    np.testing.assert_allclose(polar_factor, unitary, atol=10 * tol)
