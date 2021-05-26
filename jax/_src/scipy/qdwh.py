"""
Functions to compute the polar decomposition of the m x n matrix A, A = U @ H
where U is unitary (an m x n isometry in the m > n case) and H is n x n and
positive semidefinite (or positive definite if A is nonsingular). The method
is described in the docstring to `polarU`. This file covers the serial
case.
"""
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp


def _dot(a, b):
  return jnp.dot(a, b, precision=lax.Precision.HIGHEST)


# TODO: Handle thin case with a preliminary QR factorization.
# TODO: Handle fat case with a transpose
# TODO: Handle the identity stacking
# TODO: Allow singular avlue estimates to be manually specified
# TODO: Lower precision

def polar(matrix, eps=None, maxiter=6, compute_posdef=True,
          precision=lax.Precision.HIGHEST):
  """
  Computes the polar decomposition of the m x n matrix A, A = U @ H where U is
  unitary (an m x n isometry in the m > n case) and H is n x n and positive
  semidefinite (or positive definite if A is nonsingular) using the
  QDWH method.

  Args:
    matrix: The m x n input matrix. Currently n > m is unsupported.
    eps: The final result will satisfy |X_k - X_k-1| < |X_k| * (4*eps)**(1/3) .
    maxiter: Iterations will terminate after this many steps even if the
             above is unsatisfied.
    compute_posdef: Whether to return the positive-definite factor.
  Returns:
    unitary: The unitary factor (m x n).
    posdef: The positive-semidefinite factor (n x n) (None if compute_posdef
            is False).
    j_qr: Number of QR iterations.
    j_chol: Number of Cholesky iterations.
  """
  scaled_matrix, q_factor, l0 = _initialize_qdwh(matrix)
  unitary, j_qr, j_chol = _qdwh(scaled_matrix, l0, eps=eps, maxiter=maxiter)
  unitary = _dot(q_factor, unitary)
  posdef = None
  if compute_posdef:
    posdef = _dot(unitary.conj().T, matrix)
    posdef = 0.5 * (posdef + posdef.conj().T)

  return unitary, posdef, j_qr, j_chol


def _qdwh(matrix, l0, eps=None, maxiter=6):
  """
  Computes the unitary factor in the polar decomposition of A using
  the QDWH method. QDWH implements a 3rd order Pade approximation to the
  matrix sign function,

  X' = X * (aI + b X^H X)(I + c X^H X)^-1, X0 = A / ||A||_2.          (1)

  The coefficients a, b, and c are chosen dynamically based on an evolving
  estimate of the matrix condition number. Specifically,

  a = h(l), b = g(a), c = a + b - 1, h(x) = x g(x^2), g(x) = a + bx / (1 + cx)

  where l is initially a lower bound on the smallest singular value of X0,
  and subsequently evolves according to l' = l (a + bl^2) / (1 + c l^2).

  For poorly conditioned matrices
  (c > 100) the iteration (1) is rewritten in QR form,

  X' = (b / c) X + (1 / c)(a - b/c) Q1 Q2^H,   [Q1] R = [sqrt(c) X]   (2)
                                               [Q2]     [I        ].

  For well-conditioned matrices it is instead formulated using cheaper
  Cholesky iterations,

  X' = (b / c) X + (a - b/c) (X W^-1) W^-H,   W = chol(I + c X^H X).  (3)

  The QR iterations rapidly improve the condition number, and typically
  only 1 or 2 are required. A maximum of 6 iterations total are required
  for backwards stability to double precision.

  Args:
    matrix: The m x n input matrix.
    eps: The final result will satisfy |X_k - X_k-1| < |X_k| * (4*eps)**(1/3) .
    maxiter: Iterations will terminate after this many steps even if the
             above is unsatisfied.
  Returns:
    matrix: The unitary factor (m x n).
    jq: The number of QR iterations (1).
    jc: The number of Cholesky iterations (2).
  """
  if eps is None:
    eps = jnp.finfo(matrix.dtype).eps  # TODO: account for precision
  eps = (4 * eps)**(1 / 3)
  coefs = _qdwh_coefs(l0)
  matrix, j_qr, coefs, err = _qdwh_qr(matrix, coefs, 2 * eps, eps, maxiter)
  matrix, j_chol, _, _ = _qdwh_cholesky(matrix, coefs, err, eps, maxiter - j_qr)
  return matrix, j_qr, j_chol


@jax.jit
def _initialize_qdwh(matrix):
  """
  Does preparatory computations for QDWH:
    1. Computes an initial QR factorization of the input A. The iterations
       will be on the triangular factor R, whose condition is more easily
       estimated, and which is square even when A is rectangular.
    2. Computes R -> R / ||R||_F. Now 1 is used to upper-bound ||R||_2.
    3. Computes R^-1 by solving R R^-1 = I.
    4. Uses sqrt(N) * ||R^-1||_1 as a lower bound for ||R^-2||.
  1 / sqrt(N) * ||R^-1||_1 is then used as the initial l_0. It should be clear
  there is room for improvement here.

  Returns:
    X = R / ||R||_F;
    Q from A -> Q @ R;
    l0, the initial estimate for the QDWH coefficients.
  """
  # alpha = jnp.linalg.norm(matrix)
  # scaled_matrix = matrix / alpha
  # one_norm = jnp.linalg.norm(scaled_matrix, ord=1)

  q_factor, r_factor = jnp.linalg.qr(matrix, mode="reduced")
  alpha = jnp.linalg.norm(r_factor)
  r_factor /= alpha
  r_inv = jsp.linalg.solve_triangular(
    r_factor, jnp.eye(*(r_factor.shape), dtype=r_factor.dtype),
    overwrite_b=True)
  one_norm_inv = jnp.linalg.norm(r_inv, ord=1)

  #l0 = alpha / (1.1 * one_norm * one_norm_inv)
  l0 = 1 / (jnp.sqrt(matrix.shape[1]) * one_norm_inv)
  l0 = jnp.array(l0, dtype=r_factor.real.dtype)
  return r_factor, q_factor, l0


@jax.jit
def _qdwh_coefs(lk):
  """
  Computes a, b, c, l for the QDWH iterations.
  """
  d = (4. * (1. - lk**2) / (lk**4))**(1 / 3)
  f = 8. * (2. - lk**2) / (lk**2 * (1. + d)**(1 / 2))
  a = (1. + d)**(1 / 2) + 0.5 * (8. - 4. * d + f)**0.5
  b = (a - 1.)**2 / 4
  c = a + b - 1.
  lk = lk * (a + b * lk**2) / (1 + c * lk**2)
  return a, b, c, lk


@jax.jit
def _qdwh_qr(matrix, coefs, err0, eps, maxiter):
  """
  Applies the QDWH iteration formulated as

  X' = (b / c) X + (1 / c)(a - b/c) Q1 Q2^H,   [Q1] R = [sqrt(c) X]
                                               [Q2]     [I        ]

  to X until either c < 100, ||X' - X|| < eps||X'||,
  or the iteration count exceeds maxiter.
  """
  m, n = matrix.shape
  eye = jnp.eye(n, dtype=matrix.dtype)

  def _do_qr(args):
    matrix, j, coefs, err = args
    c = coefs[2]
    ill_conditioned = c >= 100.
    unconverged = err > (eps * jnp.linalg.norm(matrix))
    iterating = j < maxiter
    keep_going = jnp.logical_and(ill_conditioned, unconverged)
    return jnp.logical_and(keep_going, iterating)[0]

  def _qr_work(args):
    matrix, j, coefs, err0 = args
    a, b, c, lk = coefs
    csqrt = jnp.sqrt(c)
    matrixI = jnp.vstack((csqrt * matrix, eye))
    Q, _ = jnp.linalg.qr(matrixI, mode="reduced")
    Q1 = Q[:m, :]
    Q2 = Q[m:, :]
    coef = (1 / csqrt) * (a - (b / c))
    matrix *= (b / c)
    matrix += coef * _dot(Q1, Q2.T.conj())

    err = jnp.linalg.norm(matrix - matrixI[:m, :] / csqrt).astype(err0.dtype)
    coefs = _qdwh_coefs(lk)
    return matrix, j + 1, coefs, err

  j = jnp.zeros(1, dtype=jnp.float32)
  return jax.lax.while_loop(_do_qr, _qr_work, (matrix, j, coefs, err0))


@jax.jit
def _qdwh_cholesky(matrix, coefs, err0, eps, maxiter):
  """
  Applies the QDWH iteration formulated as

  matrix' = (b / c) matrix + (a - b/c) B,
    B = (matrix W^-1) W^-H,  W = chol(I + c matrix^H matrix).

  to matrix until either ||matrix' - matrix|| < eps * ||matrix'||,
  or the iteration count exceeds maxiter.
  """
  m, n = matrix.shape
  eye = jnp.eye(n, dtype=matrix.dtype)

  def _do_cholesky(args):
    matrix, j, coefs, err = args
    unconverged = err > (eps * jnp.linalg.norm(matrix))
    iterating = j < maxiter
    return jnp.logical_and(unconverged, iterating)[0]

  def _cholesky_work(args):
    matrix, j, coefs, err0 = args
    matrix0 = matrix
    a, b, c, lk = coefs
    Z = eye + c * _dot(matrix.T.conj(), matrix)
    W = jsp.linalg.cholesky(Z)
    B = jsp.linalg.solve_triangular(W.T, matrix.T, lower=True).conj()
    B = jsp.linalg.solve_triangular(W, B).conj().T
    matrix = (b / c) * matrix + (a - b / c) * B
    err = jnp.linalg.norm(matrix - matrix0).astype(err0.dtype)
    coefs = _qdwh_coefs(lk)
    return matrix, j + 1, coefs, err

  j = jnp.zeros(1, dtype=jnp.float32)
  carry = (matrix, j, coefs, err0)
  carry = jax.lax.while_loop(_do_cholesky, _cholesky_work, carry)
  return carry
