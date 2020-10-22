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
from jax.tree_util import Partial
from jax import test_util as jtu
from jax.tree_util import register_pytree_node_class
import jax.scipy.sparse.linalg

from jax.config import config
config.update('jax_enable_x64', True)
config.parse_flags_with_absl()

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


  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_dtype={}".format(np.dtype(dtype).name),
          "dtype": dtype}
       for dtype in float_types + complex_types))
  def test_SA_sort(self, dtype):
    np.random.seed(10)
    x = np.random.rand(20).astype(dtype)
    p = 10
    actual_x, actual_inds = jax.scipy.sparse.linalg.SA_sort(
        p, jnp.array(np.real(x)))
    exp_inds = np.argsort(x)
    exp_x = x[exp_inds][-p:]
    self.assertAllClose(exp_x.astype(dtype), actual_x.astype(dtype))
    self.assertAllClose(exp_inds, actual_inds)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_dtype={}".format(np.dtype(dtype).name),
          "dtype": dtype}
       for dtype in float_types + complex_types))
  def test_LA_sort(self, dtype):
    np.random.seed(10)
    x = np.random.rand(20).astype(dtype)
    p = 10
    actual_x, actual_inds = jax.scipy.sparse.linalg.LA_sort(
        p, jnp.array(np.real(x)))
    exp_inds = np.argsort(x)
    exp_x = x[exp_inds][p-1::-1]
    self.assertAllClose(exp_x.astype(dtype), actual_x.astype(dtype))
    self.assertAllClose(exp_inds[::-1], actual_inds)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_dtype={}_ncv={}".format(np.dtype(dtype).name, ncv),
          "dtype": dtype, "ncv": ncv}
       for dtype in float_types + complex_types
       for ncv in [10,20,30]))
  def test_lanczos_factorization(self, dtype, ncv):
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
    Vm, alphas, betas, residual, norm, _, _ = jax.scipy.sparse.linalg.lanczos_factorization(
        matvec, x, Vm, alphas, betas, start, ncv, tol, precision)
    Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(betas.conj(), 1)
    fm = residual * norm
    em = np.zeros((1, Vm.shape[0]))
    em[0, -1] = 1
    #test lanczos relation
    decimal = np.finfo(dtype).precision - 2
    np.testing.assert_almost_equal(
        Ham @ Vm.T - Vm.T @ Hm - fm[:, None] * em,
        np.zeros((D, ncv)).astype(dtype), decimal=decimal)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_dtype={}_ncv={}_jit".format(np.dtype(dtype).name, ncv),
          "dtype": dtype, "ncv": ncv}
       for dtype in float_types + complex_types
       for ncv in [10,20,30]))
  def test_lanczos_factorization_jit(self, dtype, ncv):
    np.random.seed(10)
    D = 1000
    precision = jax.lax.Precision.HIGHEST
    mat = np.random.rand(D, D).astype(dtype)
    Ham = mat + mat.T.conj()
    x = np.random.rand(D).astype(dtype)

    @jit
    def matvec(vector):
      return Ham @ vector

    Vm = jnp.zeros((ncv, D), dtype=dtype)
    alphas = jnp.zeros(ncv, dtype=dtype)
    betas = jnp.zeros(ncv - 1, dtype=dtype)
    start = 0
    tol = 1E-5
    lan_fact_jit = jit(jax.scipy.sparse.linalg.lanczos_factorization,
                       static_argnums=(5, 6, 7, 8))
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
        np.zeros((D, ncv)).astype(dtype), decimal=decimal)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_dtype={}_which={}".format(np.dtype(dtype).name, which),
          "dtype": dtype, "which": which}
       for dtype in float_types + complex_types
       for which in ['SA', 'LA']))
  def test_eigsh_small_matrix(self, dtype, which):

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

    eta, U, _ = jax.scipy.sparse.linalg.eigsh(
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



  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_dtype={}_N={}_hop_type={}".format(
                  np.dtype(dtype).name, N, hop_type_atol),
          "dtype": dtype,
          "N": N,
          "hop_type_atol": hop_type_atol
      } for dtype in [np.float64, np.complex128]
        for N in [14, 18]
        for hop_type_atol in [('uniform', 1E-12), ('randn', 1E-8)]))

  def test_eigsh_large_problem(self, N, dtype, hop_type_atol):
    """
    Find the lowest eigenvalues and eigenvectors
    of a 1d free-fermion Hamiltonian on N sites.
    The dimension of the hermitian matrix is
    (2**N, 2**N).
    """

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
    hop_type, atol = hop_type_atol
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

    np.random.seed(10)
    init = jnp.array(np.random.randn(2**N)).astype(dtype)
    init /= jnp.linalg.norm(init)

    ncv = 20
    numeig = 4
    which = 'SA'
    tol = 1E-8
    maxiter = 30
    eta, _, _ = jax.scipy.sparse.linalg.eigsh(
        matvec=matvec,
        initial_state=init,
        num_krylov_vecs=ncv,
        numeig=numeig,
        which=which,
        tol=tol,
        maxiter=maxiter,
        precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(
        eta, many_body_energies[:numeig], atol=atol, rtol=atol)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
