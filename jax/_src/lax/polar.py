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


"""
Functions to compute the polar decomposition of the m x n matrix A, A = U @ H
where U is unitary (an m x n isometry in the m > n case) and H is n x n and
positive semidefinite (or positive definite if A is nonsingular). The method
is described in the docstring to `polarU`. This file covers the serial
case.
"""
import functools
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp


# TODO: Allow singular value estimates to be manually specified
@jax.jit
def _add_to_diagonal(X, val):
  new_diagonal = X.diagonal() + val
  diag_indices = jnp.diag_indices(X.shape[0])
  return X.at[diag_indices].set(new_diagonal)


@jax.jit
def _dot(a, b):
  return jnp.dot(a, b, precision=lax.Precision.HIGHEST)


def polar(a, side='right', method='qdwh', eps=None, maxiter=50):
  """ Computes the polar decomposition.

  Given the (m x n) matrix `a`, returns the factors of the polar decomposition
  `u` (m x n) and `p` such that `a = up` (if side is "right"; p is (n x n)) or
  `a = pu` (if side is "left"; p is (m x m)), where `p` is positive
  semidefinite.  If `a` is nonsingular, `p` is positive definite and the
  decomposition is unique. `u` has orthonormal columns unless n > m, in which
  case it has orthonormal rows.

  Writing an SVD of `a` as `a = u_svd @ s_svd @ v^h_svd`, we have
  `u = u_svd @ v^h_svd`. Thus the unitary factor `u` can be construed as
  the application of the signum function to the singular values of `a`;
  or, if `a` is Hermitian, the eigenvalues.

  Several methods exist to compute the polar decomposition. Currently two
  are supported:
    `method`="svd": Computes the SVD of `a` and then forms
                    `u = u_svd @ v^h_svd`. This fails on the TPU, since
                    no SVD algorithm independent of the polar decomposition
                    exists there.
    `method`="qdwh": Applies a certain iterative expansion of the matrix
                     signum function to `a` based on QR and Cholesky
                     decompositions.

  Args:
    a: The m x n input matrix.
    side: Determines whether a right or left polar decomposition is computed.
      If side is "right" then `a = up`. If side is "left" then `a = pu`. The
      default is "right".
    method: Determines the algorithm used, as described above.
    precision: :class:`~jax.lax.Precision` object specifying the matmul precision.

    The remaining arguments are only meaningful if method is "qdwh".
    eps: The final result will satisfy |X_k - X_k-1| < |X_k| * (4*eps)**(1/3) .
    maxiter: Iterations will terminate after this many steps even if the
             above is unsatisfied.
  Returns:
    unitary: The unitary factor (m x n).
    posdef: The positive-semidefinite factor. Either (n, n) or (m, m)
      depending on whether side is "right" or "left", respectively.
    info: Stores convergence information.
      if method is "svd": None
      if method is "qdwh": j_qr: Number of QR iterations.
                           j_chol: Number of Cholesky iterations.
                           errs: Convergence history.
  """
  return _polar(a, side, method, eps, maxiter)


@functools.partial(jax.jit, static_argnums=(1, 2, 4))
def _polar(a, side, method, eps, maxiter):
  if side not in ("left", "right"):
    raise ValueError(f"side={side} was invalid.")

  unitary, info = _polar_unitary(a, method, eps, maxiter)
  if side == "right":
    posdef = _dot(unitary.conj().T, a)
  else:
    posdef = _dot(a, unitary.conj().T)
  posdef = 0.5 * (posdef + posdef.conj().T)
  return unitary, posdef, info


def polar_unitary(a, method="qdwh", eps=None, maxiter=50):
  """ Computes the unitary factor u in the polar decomposition `a = u p`
  (or `a = p u`).
  """
  return _polar_unitary(a, method, eps, maxiter)


@functools.partial(jax.jit, static_argnums=(1, 3))
def _polar_unitary(a, method, eps, maxiter):
  if method not in ("svd", "qdwh"):
    raise ValueError(f"method={method} is unsupported.")

  if method == "svd":
    u_svd, _, vh_svd = jnp.linalg.svd(a, full_matrices=False)
    unitary = _dot(u_svd, vh_svd)
    info = None
  elif method == "qdwh":
    unitary, j_qr, j_chol, errs = _qdwh(a, eps, maxiter)
    info = (j_qr, j_chol, errs)
  else:
    raise ValueError("How did we get here?")
  return unitary, info


@functools.partial(jax.jit, static_argnums=(2,))
def _qdwh(matrix, eps, maxiter):
  """ Computes the unitary factor in the polar decomposition of A using
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
    errs: Convergence history.
  """
  n_rows, n_cols = matrix.shape
  fat = n_cols > n_rows
  if fat:
    matrix = matrix.T
  matrix, q_factor, l0 = _initialize_qdwh(matrix)

  if eps is None:
    eps = jnp.finfo(matrix.dtype).eps
  tol_lk = 5 * eps  # stop when lk differs from 1 by less
  tol_delta = jnp.cbrt(tol_lk)  # stop when the iterates change by less
  coefs = _qdwh_coefs(l0)
  errs = jnp.zeros(maxiter, dtype=matrix.real.dtype)
  matrix, j_qr, coefs, errs = _qdwh_qr(
    matrix, coefs, errs, tol_lk, tol_delta, maxiter)
  matrix, j_chol, errs = _qdwh_cholesky(
    matrix, coefs, errs, tol_lk, tol_delta, j_qr, maxiter)
  matrix = _dot(q_factor, matrix)

  if fat:
    matrix = matrix.T
  return matrix, j_qr, j_chol, errs


@jax.jit
def _initialize_qdwh(matrix):
  """ Does preparatory computations for QDWH:
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
  q_factor, r_factor = jnp.linalg.qr(matrix, mode="reduced")
  alpha = jnp.linalg.norm(r_factor)
  r_factor /= alpha
  eye = jnp.eye(*r_factor.shape, dtype=r_factor.dtype)
  r_inv = jsp.linalg.solve_triangular(r_factor, eye, overwrite_b=True)
  one_norm_inv = jnp.linalg.norm(r_inv, ord=1)
  l0 = 1 / (jnp.sqrt(matrix.shape[1]) * one_norm_inv)
  eps = jnp.finfo(r_factor.dtype).eps
  l0 = jnp.array(l0, dtype=r_factor.real.dtype)
  l0 = jnp.where(l0 < eps, x=eps, y=l0)
  l0 = jnp.where(l0 > 1.0, x=1.0, y=l0)
  return r_factor, q_factor, l0


@jax.jit
def _qdwh_coefs(lk):
  """ Computes a, b, c, l for the QDWH iterations.
  The input lk must be in (0, 1]; lk=1 is a fixed point.
  Some facts about the coefficients:
    -for lk = 1 we have a=3, b=1, c=3, lk_new = 1.
    -The float64 vs float32 computation of each coef appears to differ
     only by noise on the order of 1E-9 to 1E-7 for all values of lk.
     There is no apparent secular change in the (relative) error.
    -All coefs change roughly as power laws; over e.g. [1E-14, 1]:
      - a decreases from 5.43E9 to 3.
      - b decreases from 7.37E18 to 1.
      - c decreases from 7.37E18 to 3, only diverging from b near lk=1.
      - lk increases from 5.45E-5 to 1.

  lk is an estimate of the scaled matrix's smallest singular value
  """
  lk = jnp.where(lk > 1.0, x=1.0, y=lk)
  d = (4. * (1. - lk**2) / (lk**4))**(1 / 3)
  f = 8. * (2. - lk**2) / (lk**2 * (1. + d)**(1 / 2))
  a = (1. + d)**(1 / 2) + 0.5 * (8. - 4. * d + f)**0.5
  b = (a - 1.)**2 / 4
  c = a + b - 1.
  lk = lk * (a + b * lk**2) / (1 + c * lk**2)
  return a, b, c, lk


@jax.jit
def _unconverged(lk, j, maxiter, err, tol_delta, tol_lk):
  changing = err > tol_delta
  far_from_end = jnp.abs(1 - lk) > tol_lk
  unconverged = jnp.logical_or(changing, far_from_end)
  iterating = j < maxiter
  return jnp.logical_and(iterating, unconverged)[0]


@jax.jit
def _qdwh_qr(matrix, coefs, errs, tol_lk, tol_delta, maxiter):
  """ Applies the QDWH iteration formulated as

  X' = (b / c) X + (1 / c)(a - b/c) Q1 Q2^H,   [Q1] R = [sqrt(c) X]
                                               [Q2]     [I        ]

  to X until either c < 100, ||X' - X|| < eps||X'||,
  or the iteration count exceeds maxiter.
  """
  n_rows, n_cols = matrix.shape
  eye = jnp.eye(n_cols, dtype=matrix.dtype)

  def _do_qr(args):
    _, j, coefs, _, err = args
    c = coefs[2]
    lk = coefs[-1]
    unconverged = _unconverged(lk, j, maxiter, err, tol_delta, tol_lk)
    ill_conditioned = c > 100.
    return jnp.logical_and(ill_conditioned, unconverged)

  def _qr_work(args):
    matrix, j, coefs, errs, _ = args
    a, b, c, lk = coefs
    csqrt = jnp.sqrt(c)
    matrixI = jnp.vstack((csqrt * matrix, eye))
    # Note: it should be possible to compute the QR of csqrt * matrix
    # and build the concatenation with I at O(N).
    Q, _ = jnp.linalg.qr(matrixI, mode="reduced")
    Q1 = Q[:n_rows, :]
    Q2 = Q[n_rows:, :]
    coef = (1 / csqrt) * (a - (b / c))
    new_matrix = (b / c) * matrix + coef * _dot(Q1, Q2.T.conj())
    err = jnp.linalg.norm(matrix - new_matrix)
    err = jnp.full(1, err).astype(errs[0].dtype)
    errs = errs.at[j].set(err)
    coefs = _qdwh_coefs(lk)
    return new_matrix, j + 1, coefs, errs, err

  j = jnp.zeros(1, dtype=jnp.int32)
  err = jnp.full(1, 2 * tol_delta).astype(matrix.real.dtype)
  matrix, j, coefs, errs, _ = jax.lax.while_loop(
    _do_qr, _qr_work, (matrix, j, coefs, errs, err))
  return matrix, j, coefs, errs


@jax.jit
def _qdwh_cholesky(matrix, coefs, errs, tol_delta, tol_lk, j0, maxiter):
  """ Applies the QDWH iteration formulated as

  matrix' = (b / c) matrix + (a - b/c) B,
    B = (matrix W^-1) W^-H,  W = chol(I + c matrix^H matrix).

  to matrix until either ||matrix' - matrix|| < eps * ||matrix'||,
  or the iteration count exceeds maxiter.
  """

  def _do_cholesky(args):
    _, j, coefs, errs = args
    lk = coefs[-1]
    return _unconverged(lk, j, maxiter, errs[j - 1], tol_delta, tol_lk)

  def _cholesky_work(args):
    matrix, j, coefs, errs = args
    a, b, c, lk = coefs
    Z = c * _dot(matrix.T.conj(), matrix)
    Z = _add_to_diagonal(Z, 1.)
    W = jsp.linalg.cholesky(Z)
    B = jsp.linalg.solve_triangular(W.T, matrix.T, lower=True).conj()
    B = jsp.linalg.solve_triangular(W, B).conj().T
    new_matrix = (b / c) * matrix + (a - b / c) * B
    # possible instability if a ~ b / c
    err = jnp.linalg.norm(new_matrix - matrix).astype(errs[0].dtype)
    errs = errs.at[j].set(err)
    coefs = _qdwh_coefs(lk)
    return new_matrix, j + 1, coefs, errs

  carry = (matrix, j0, coefs, errs)
  matrix, j_total, coefs, errs = jax.lax.while_loop(
    _do_cholesky, _cholesky_work, carry)
  return matrix, j_total - j0, errs
