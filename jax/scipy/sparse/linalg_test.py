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
import numpy as np
import jax
from jax.tree_util import Partial
from jax import jit
from jax.scipy.sparse.linalg import SA_sort, lanczos_factorization, eigsh

import jax.numpy as jnp
import pytest
jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize("dtype",
                         [np.float32, np.float64, np.complex64, np.complex128])
def test_SA_sort(dtype):
  np.random.seed(10)
  x = np.random.rand(20).astype(dtype)
  p = 10
  actual_x, actual_inds = SA_sort(p, jnp.array(np.real(x)))
  exp_inds = np.argsort(x)
  exp_x = x[exp_inds][-p:]
  np.testing.assert_allclose(exp_x, actual_x)
  np.testing.assert_allclose(exp_inds, actual_inds)


@pytest.mark.parametrize("dtype",
                         [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("ncv", [10, 20, 30])
def test_lanczos_factorization(dtype, ncv):
  np.random.seed(10)
  D = 1000
  precision = jax.lax.Precision.HIGHEST
  mat = np.random.rand(D, D).astype(dtype)
  Ham = mat + mat.T.conj()
  x = np.random.rand(D).astype(dtype)

  def matvec(vector):
    return Ham @ vector

  Vm = jnp.zeros((ncv, D), dtype=dtype)
  alphas = jnp.zeros(ncv, dtype=dtype)
  betas = jnp.zeros(ncv - 1, dtype=dtype)
  start = 0
  tol = 1E-5
  Vm, alphas, betas, residual, norm, _, _ = lanczos_factorization(
      matvec, x, Vm, alphas, betas, start, ncv, tol, precision)
  Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(betas.conj(), 1)
  fm = residual * norm
  em = np.zeros((1, Vm.shape[0]))
  em[0, -1] = 1
  #test lanczos relation
  decimal = np.finfo(dtype).precision - 2
  np.testing.assert_almost_equal(
      Ham @ Vm.T - Vm.T @ Hm - fm[:, None] * em,
      np.zeros((D, ncv)).astype(dtype),
      decimal=decimal)


@pytest.mark.parametrize("dtype",
                         [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("ncv", [10, 20, 30])
def test_lanczos_factorization_jit(dtype, ncv):
  np.random.seed(10)
  D = 1000
  precision = jax.lax.Precision.HIGHEST
  mat = np.random.rand(D, D).astype(dtype)
  Ham = mat + mat.T.conj()
  x = np.random.rand(D).astype(dtype)

  def matvec(vector):
    return Ham @ vector

  Vm = jnp.zeros((ncv, D), dtype=dtype)
  alphas = jnp.zeros(ncv, dtype=dtype)
  betas = jnp.zeros(ncv - 1, dtype=dtype)
  start = 0
  tol = 1E-5
  lan_fact_jit = jit(lanczos_factorization, static_argnums=(5, 6, 7, 8))
  Vm, alphas, betas, residual, norm, _, _ = lan_fact_jit(
      Partial(matvec), x, Vm, alphas, betas, start, ncv, tol, precision)
  Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(betas.conj(), 1)
  fm = residual * norm
  em = np.zeros((1, Vm.shape[0]))
  em[0, -1] = 1
  #test lanczos relation
  decimal = np.finfo(dtype).precision - 2
  np.testing.assert_almost_equal(
      Ham @ Vm.T - Vm.T @ Hm - fm[:, None] * em,
      np.zeros((D, ncv)).astype(dtype),
      decimal=decimal)


def generate_data(dtype, D):
  H = np.random.randn(D, D).astype(dtype)
  init = np.random.randn(D).astype(dtype)
  if dtype in (np.complex64, np.complex128):
    H += 1j * np.random.randn(D, D).astype(dtype)
    init += 1j * np.random.randn(D).astype(dtype)
  return H + H.T.conj(), init


def compare_eigvals_and_eigvecs(U, eta, U_exact, eta_exact, thresh=1E-8):
  _, iy = np.nonzero(np.abs(eta[:, None] - eta_exact[None, :]) < thresh)
  U_exact_perm = U_exact[:, iy]
  U_exact_perm = U_exact_perm / np.expand_dims(np.sum(U_exact_perm, axis=0), 0)
  U = U / np.expand_dims(np.sum(U, axis=0), 0)
  prec = np.finfo(U.dtype).precision
  atol = 10**(-prec // 2)
  rtol = atol
  np.testing.assert_allclose(U_exact_perm, U, atol=atol, rtol=rtol)
  np.testing.assert_allclose(eta, eta_exact[iy], atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype",
                         [np.float64, np.complex128, np.float32, np.complex64])
@pytest.mark.parametrize("which", ['SA', 'LA'])
def test_eigsh_small_matrix(dtype, which):
  thresh = {
      np.complex64: 1E-3,
      np.float32: 1E-3,
      np.float64: 1E-4,
      np.complex128: 1E-4
  }
  D = 1000
  np.random.seed(10)
  H, init = generate_data(dtype, D)

  def mv(x):
    return jnp.matmul(H, x, precision=jax.lax.Precision.HIGHEST)

  eta, U, _ = eigsh(
      mv,
      init,
      num_krylov_vecs=60,
      numeig=4,
      which=which,
      tol=1E-10,
      maxiter=500,
      precision=jax.lax.Precision.HIGHEST)
  eta_exact, U_exact = jnp.linalg.eigh(H)
  compare_eigvals_and_eigvecs(
      np.stack(U, axis=1), eta, U_exact, eta_exact, thresh=thresh[dtype])


def get_hoppings(dtype, N, which):
  if which == 'uniform':
    hop = -jnp.ones(N - 1, dtype)
    if dtype in (np.complex128, np.complex64):
      hop -= 1j * jnp.ones(N - 1, dtype)
  elif which == 'randn':
    hop = -jnp.array(np.random.randn(N - 1).astype(dtype))
    if dtype in (np.complex128, np.complex64):
      hop -= 1j * jnp.array(np.random.randn(N - 1).astype(dtype))
  return hop


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("hop_type", ['uniform', 'randn'])
@pytest.mark.parametrize("N", [14, 18])
def test_eigsh_large_problem(N, dtype, hop_type):
  """
  Find the lowest eigenvalues and eigenvectors
  of a 1d free-fermion Hamiltonian on N sites.
  The dimension of the hermitian matrix is
  (2**N, 2**N).
  """
  hop = get_hoppings(dtype, N, hop_type)
  pot = jnp.ones(N, dtype)
  P = jnp.diag(np.array([0, -1])).astype(dtype)
  c = jnp.array([[0, 1], [0, 0]], dtype)
  n = c.T @ c
  eye = jnp.eye(2,dtype=dtype)
  neye = jnp.kron(n, eye)
  eyen = jnp.kron(eye, n)
  ccT = jnp.kron(c @ P, c.T)
  cTc = jnp.kron(c.T, c)

  @jax.jit
  def matvec(vec):
    x = vec.reshape((4, 2**(N - 2)))
    out = jnp.zeros(x.shape, x.dtype)
    t1 = neye * pot[0] + eyen * pot[1] / 2
    t2 = cTc * hop[0] - ccT * jnp.conj(hop[0])
    out += jnp.einsum('ij,ki -> kj', x, t1 + t2)
    x = x.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape((4, 2**(N - 2)))
    out = out.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape(
        (4, 2**(N - 2)))
    for site in range(1, N - 2):
      t1 = neye * pot[site] / 2 + eyen * pot[site + 1] / 2
      t2 = cTc * hop[site] - ccT * jnp.conj(hop[site])
      out += jnp.einsum('ij,ki -> kj', x, t1 + t2)
      x = x.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape((4, 2**(N - 2)))
      out = out.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape(
          (4, 2**(N - 2)))
    t1 = neye * pot[N - 2] / 2 + eyen * pot[N - 1]
    t2 = cTc * hop[N - 2] - ccT * jnp.conj(hop[N - 2])
    out += jnp.einsum('ij,ki -> kj', x, t1 + t2)
    x = x.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape((4, 2**(N - 2)))
    out = out.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape(
        (4, 2**(N - 2)))

    x = x.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape(2**N)
    out = out.reshape((2, 2**(N - 1))).transpose((1, 0)).reshape(2**N)
    return out.ravel()

  H = np.diag(pot) + np.diag(hop.conj(), 1) + np.diag(hop, -1)
  single_particle_energies = np.linalg.eigh(H)[0]

  many_body_energies = []
  for n in range(2**N):
    many_body_energies.append(
        np.sum(single_particle_energies[np.nonzero(
            np.array(list(bin(n)[2:]), dtype=int)[::-1])[0]]))
  many_body_energies = np.sort(many_body_energies)

  init = jnp.array(np.random.randn(2**N)).astype(dtype)
  init /= jnp.linalg.norm(init)

  ncv = 20
  numeig = 6
  which = 'SA'
  tol = 1E-8
  maxiter = 30
  eta, _, _ = eigsh(
      matvec=matvec,
      initial_state=init,
      num_krylov_vecs=ncv,
      numeig=numeig,
      which=which,
      tol=tol,
      maxiter=maxiter,
      precision=jax.lax.Precision.HIGHEST)
  np.testing.assert_allclose(
      eta, many_body_energies[:numeig], atol=1E-13, rtol=1E-13)
