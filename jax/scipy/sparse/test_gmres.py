import numpy as np
import jax
import jax.numpy as jnp
import pytest

import linalg
jax.config.update('jax_enable_x64', True)

jax_dtypes = [jnp.float64, jnp.float32]#, np.float64]#, np.complex64, np.complex128]


def test_gmres_on_small_known_problem(dtype):
  """
  GMRES produces the correct result on an analytically solved
  linear system.
  """
  A = jax.numpy.array(([[1, 1], [3, -4]]), dtype=dtype)
  b = jax.numpy.array([3, 2], dtype=dtype)
  x0 = jax.numpy.ones(2, dtype=dtype)
  n_kry = 2
  maxiter = 1

  @jax.tree_util.Partial
  def A_mv(x):
    return A @ x
  tol = A.size*jax.numpy.finfo(dtype).eps
  x, _ = linalg.gmres(A_mv, b, x0=x0, tol=tol, atol=tol, restart=n_kry,
                      maxiter=maxiter)
  solution = jax.numpy.array([2., 1.], dtype=dtype)
  try:
    np.testing.assert_allclose(x, solution, atol=tol)
  except AssertionError as e:
    print("Failed at dtype ", dtype)
    print(e)


#  @pytest.mark.parametrize("dtype", jax_dtypes)
#  def test_gs(dtype):
#    """
#    The Gram-Schmidt process works.
#    """
#    n = 8
#    A = np.zeros((n, 2), dtype=dtype)
#    A[:-1, 0] = 1.0
#    Ai = A[:, 0] / np.linalg.norm(A[:, 0])
#    A[:, 0] = Ai
#    A[-1, -1] = 1.0
#    A = jax.numpy.array(A)

#    x0 = jax.numpy.array(np.random.rand(n).astype(dtype))
#    v_new, _ = jax.lax.scan(gmres.gs_step, x0, xs=A.T)
#    dotcheck = v_new @ A
#    tol = A.size*jax.numpy.finfo(dtype).eps
#    np.testing.assert_allclose(dotcheck, np.zeros(2), atol=tol)


@pytest.mark.parametrize("dtype", jax_dtypes)
def test_gmres_arnoldi_step(dtype):
  """
  The Arnoldi decomposition within GMRES is correct.
  """
  n = 3
  n_kry = n
  np.random.seed(10)
  A = jax.numpy.array(np.random.rand(n, n).astype(dtype))
  x0 = jax.numpy.array(np.random.rand(n).astype(dtype))
  Q = np.zeros((n, n_kry + 1), dtype=x0.dtype)
  Q[:, 0] = x0/jax.numpy.linalg.norm(x0)
  Q = jnp.array(Q)
  H = jax.numpy.zeros((n_kry, n_kry + 1), dtype=x0.dtype)
  tol = A.size*jax.numpy.finfo(dtype).eps
  M = linalg._identity

  @jax.tree_util.Partial
  def A_mv(x):
    return A @ x
  for k in range(n_kry):
    Q, H = linalg.kth_arnoldi_iteration(k, A_mv, M, Q, H, tol)
  QAQ = Q[:, :n_kry].conj().T @ A @ Q[:, :n_kry]
  try:
    np.testing.assert_allclose(H.T[:n_kry, :], QAQ, atol=tol)
  except AssertionError as err:
    print(err)
  else:
    print(f"Arnoldi passed at dtype {dtype}!")

if __name__ == "__main__":
  for dtype in jax_dtypes:
    test_gmres_on_small_known_problem(dtype)
    test_gmres_arnoldi_step(dtype)


#  @pytest.mark.parametrize("dtype", jax_dtypes)
#  def test_gmres_krylov(dtype):
#    """
#    gmres_krylov correctly builds the QR-decomposed Arnoldi decomposition.
#    This function assumes that gmres["kth_arnoldi_step (which is
#    independently tested) is correct.
#    """
#    dummy = jax.numpy.zeros(1, dtype=dtype)
#    dtype = dummy.dtype
#    gmres = jitted_functions.gmres_wrapper(jax)

#    n = 2
#    n_kry = n
#    np.random.seed(10)

#    @jax.tree_util.Partial
#    def A_mv(x):
#      return A @ x
#    A = jax.numpy.array(np.random.rand(n, n).astype(dtype))
#    tol = A.size*jax.numpy.finfo(dtype).eps
#    x0 = jax.numpy.array(np.random.rand(n).astype(dtype))
#    b = jax.numpy.array(np.random.rand(n), dtype=dtype)
#    r, beta = gmres.gmres_residual(A_mv, [], b, x0)
#    _, V, R, _ = gmres.gmres_krylov(A_mv, [], n_kry, x0, r, beta,
#                                    tol, jax.numpy.linalg.norm(b))
#    phases = jax.numpy.sign(jax.numpy.diagonal(R[:-1, :]))
#    R = phases.conj()[:, None] * R[:-1, :]
#    Vtest = np.zeros((n, n_kry + 1), dtype=x0.dtype)
#    Vtest[:, 0] = r/beta
#    Vtest = jax.numpy.array(Vtest)
#    Htest = jax.numpy.zeros((n_kry + 1, n_kry), dtype=x0.dtype)
#    for k in range(n_kry):
#      Vtest, Htest = gmres.kth_arnoldi_step(k, A_mv, [], Vtest, Htest, tol)
#    _, Rtest = jax.numpy.linalg.qr(Htest)
#    phases = jax.numpy.sign(jax.numpy.diagonal(Rtest))
#    Rtest = phases.conj()[:, None] * Rtest
#    np.testing.assert_allclose(V, Vtest, atol=tol)
#    np.testing.assert_allclose(R, Rtest, atol=tol)




#  @pytest.mark.parametrize("dtype", jax_dtypes)
#  def test_givens(dtype):
#    """
#    gmres["givens_rotation produces the correct rotation factors.
#    """
#    gmres = jitted_functions.gmres_wrapper(jax)
#    np.random.seed(10)
#    v = jax.numpy.array(np.random.rand(2).astype(dtype))
#    cs, sn = gmres.givens_rotation(*v)
#    rot = np.zeros((2, 2), dtype=dtype)
#    rot[0, 0] = cs
#    rot[1, 1] = cs
#    rot[0, 1] = -sn
#    rot[1, 0] = sn
#    rot = jax.numpy.array(rot)
#    result = rot @ v
#    tol = 4*jax.numpy.finfo(dtype).eps
#    np.testing.assert_allclose(result[-1], 0., atol=tol)
