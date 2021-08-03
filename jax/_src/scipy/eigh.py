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


"""Serial algorithm for eigh."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax


def _fill_diagonal(X, vals):
  return jax.ops.index_update(X, jnp.diag_indices(X.shape[0]), vals)


def _shift_diagonal(X, val):
  return _fill_diagonal(X, X.diagonal() + val)


def _gershgorin(H):
  """ Computes bounds on the smalles and largest magnitude eigenvalues
  of H using the "Gershgorin circle theorem".
  """
  H_diag = jnp.diag(H)
  diag_elements = jnp.diag_indices_from(H)
  abs_H_diag0 = jnp.abs(H.at[diag_elements].set(0.))
  col_sums = jnp.sum(abs_H_diag0, axis=0)
  row_sums = jnp.sum(abs_H_diag0, axis=1)
  row_min = jnp.min(H_diag - row_sums)
  row_max = jnp.max(H_diag + row_sums)
  col_min = jnp.min(H_diag - col_sums)
  col_max = jnp.max(H_diag + col_sums)
  min_est = jnp.max(jnp.array([row_min, col_min]))
  max_est = jnp.min(jnp.array([row_max, col_max]))
  return min_est, max_est


def _similarity_transform(
    matrix_in, matrix_out, precision=lax.Precision.HIGHEST):
  """ Returns matrix_out.conj().T @ matrix_in @ matrix_out, done in
  order from left to right.
  """
  out = jnp.dot(matrix_out.conj().T, matrix_in, precision=precision)
  return jnp.dot(out, matrix_out, precision=precision)


def _canonically_purify_initial_guess(H, rank):
  lambda_min, lambda_max = _gershgorin(H)
  N = H.shape[0]
  mu = jnp.trace(H) / N
  theta = rank / N
  beta = theta / (lambda_max - mu)
  beta_bar = (1 - theta) / (mu - lambda_min)
  beta_2 = jnp.where(beta <= beta_bar, x=beta, y=beta_bar)
  beta_2_bar = jnp.where(beta >= beta_bar, x=beta, y=beta_bar)

  P0 = _shift_diagonal(-beta_2 * H, theta + beta_2 * mu)
  P0_bar = _shift_diagonal(-beta_2_bar * H, theta + beta_2_bar * mu)
  guess = 0.5 * (P0 + P0_bar)
  perturbation = (rank - jnp.trace(guess)) / N
  return _shift_diagonal(guess, perturbation)


def _canonically_purify(H, rank, maxiter=200):
  """ Computes an orthogonal projector into the eigenspace of `H`'s
  `rank` most negative eigenpairs.

  Args:
    H: The Hermitian matrix to be purified.
    rank: Rank of the projector to compute.
    maxiter: Terminate iteration here even if unconverged.
  Returns:
    P: The orthogonal projector.
  """
  N = H.shape[0]
  tol = jnp.finfo(H.dtype).eps * N / 2

  def _keep_purifying(carry):
    _, j, err = carry
    still_going = j < maxiter
    unconverged = err > tol
    return jnp.logical_and(still_going, unconverged)[0]

  def _purify(carry):
    P, j, _ = carry
    proj_2bar = -1.0 * _shift_diagonal(P, -1.)
    proj_2bar = jnp.dot(P, proj_2bar, precision=lax.Precision.HIGHEST)
    proj_3bar = jnp.dot(P, proj_2bar, precision=lax.Precision.HIGHEST)
    tr_proj_2bar = jnp.abs(jnp.trace(proj_2bar))
    tr_proj_2bar = jnp.where(tol > tr_proj_2bar, x=tol, y=tr_proj_2bar)
    tr_proj_3bar = jnp.trace(proj_3bar)
    coef = tr_proj_3bar / tr_proj_2bar
    P += 2 * (proj_3bar - coef * proj_2bar)
    return P, j + 1, err

  P = _canonically_purify_initial_guess(H, rank)
  j = jnp.zeros(1, dtype=jnp.int32)
  err = 2.0 * tol
  P, _, _ = lax.while_loop(_keep_purifying, _purify, (P, j, err))
  return P


def _projector_subspace(P, H, rank, maxiter=2):
  """ Decomposes the `n x n` rank `rank` Hermitian projector `P` into
  an `n x rank` isometry `Vm` such that `P = Vm @ Vm.conj().T` and
  an `n x (n - rank)` isometry `Vm` such that -(I - P) = Vp @ Vp.conj().T`.

  The subspaces are computed using the naiive QR eigendecomposition
  algorithm, which converges very quickly due to the sharp separation
  between the relevant eigenvalues of the projector.

  Args:
    P: A rank-`rank` Hermitian projector into the space of `H`'s
       first `rank` eigenpairs.
    H: The aforementioned Hermitian matrix, which is used to track
       convergence.
    rank: Rank of `P`.
    maxiter: Maximum number of iterations.
  Returns:
    Vm, Vp: Isometries into the eigenspaces described in the docstring.
  """
  # Choose an initial guess: the `rank` largest-norm columns of P.
  column_norms = jnp.linalg.norm(P, axis=1)
  sort_idxs = jnp.argsort(column_norms)
  X = P[:, sort_idxs]
  X = X[:, :rank]

  H_norm = jnp.linalg.norm(H)
  thresh = 10 * jnp.finfo(X.dtype).eps * H_norm

  # First iteration skips the matmul.
  def body_f_after_matmul(X):
    Q, _ = jnp.linalg.qr(X, mode="complete")
    V1 = Q[:, :rank]
    V2 = Q[:, rank:]
    # TODO: might be able to get away with lower precision here
    error_matrix = jnp.dot(V2.conj().T, H, precision=lax.Precision.HIGHEST)
    error_matrix = jnp.dot(error_matrix, V1, precision=lax.Precision.HIGHEST)
    error = jnp.linalg.norm(error_matrix) / H_norm
    return V1, V2, error

  def cond_f(args):
    _, _, j, error = args
    still_counting = j < maxiter
    unconverged = error > thresh
    return jnp.logical_and(still_counting, unconverged)[0]

  def body_f(args):
    V1, _, j, _ = args
    X = jnp.dot(P, V1, precision=lax.Precision.HIGHEST)
    V1, V2, error = body_f_after_matmul(X)
    return V1, V2, j + 1, error

  V1, V2, error = body_f_after_matmul(X)
  one = jnp.ones(1, dtype=jnp.int32)
  V1, V2, _, error = lax.while_loop(cond_f, body_f, (V1, V2, one, error))
  return V1, V2


def _split_spectrum(H, V0, precision):
  """ The `N x N` Hermitian matrix `H` is split into two matrices `Hm`
  `Hp`, respectively sharing its eigenspaces beneath and above
  its `N // 2`th eigenvalue.

  Returns, in addition, `Vm` and `Vp`, isometries such that
  `Hi = Vi.conj().T @ H @ Vi`. If `V0` is not None, `V0 @ Vi` are
  returned instead; this allows the overall isometries mapping from
  an initial input matrix to progressively smaller blocks to be formed.

  Args:
    H: The Hermitian matrix to split.
    V0: Matrix of isometries to be updated.
    precision: TPU matmul precision.
  Returns:
    Hm: A Hermitian matrix sharing the eigenvalues of `H` beneath
      the `N // 2`th.
    Vm: An isometry from the input space of `V0` to `Hm`.
    Hp: A Hermitian matrix sharing the eigenvalues of `H` above
      the `N // 2`th.
    Vp: An isometry from the input space of `V0` to `Hp`.
  """
  rank = H.shape[1] // 2
  P = _canonically_purify(H, rank)
  Vm, Vp = _projector_subspace(P, H, rank)
  Hm = _similarity_transform(H, Vm, precision)
  Hp = _similarity_transform(H, Vp, precision)
  if V0 is not None:
    Vm = jnp.dot(V0, Vm, precision=precision)
    Vp = jnp.dot(V0, Vp, precision=precision)
  return Hm, Vm, Hp, Vp


def _eigh_work(H, V, precision, termination_size):
  """ The main work loop performing the symmetric eigendecomposition of H.
  Each step recursively computes a projector into the space of eigenvalues
  above jnp.mean(jnp.diag(H)). The result of the projections into and out of
  that space, along with the isometries accomplishing these, are then computed.
  This is performed recursively until the projections have size 1, and thus
  store an eigenvalue of the original input; the corresponding isometry is
  the related eigenvector. The results are then composed.

  Args:
    H: The Hermitian input.
    V: Stores the isometries projecting H into its subspaces.
    precision: The matmul precision.

  Returns:
    H, V: The result of the projection.
  """
  if H.shape[0] <= termination_size:
    evals, evecs = jnp.linalg.eigh(H)
    if V is not None:
      evecs = jnp.dot(V, evecs, precision=precision)
    return evals, evecs

  Hm, Vm, Hp, Vp = _split_spectrum(H, V, precision)
  Hm, Vm = _eigh_work(Hm, Vm, precision, termination_size)
  Hp, Vp = _eigh_work(Hp, Vp, precision, termination_size)

  if Hm.ndim != 1 or Hp.ndim != 1:
    raise ValueError(f"One of Hm.ndim={Hm.ndim} or Hp.ndim={Hp.ndim} != 1 ",
                     "indicating recursion terminated unexpectedly.")

  evals = jnp.hstack((Hm, Hp))
  evecs = jnp.hstack((Vm, Vp))
  return evals, evecs


<<<<<<< HEAD
@jax.partial(jax.jit, static_argnums=(1, 2, 3))
=======
#@jax.partial(jax.jit, static_argnums=(1, 2, 3))
>>>>>>> 417b54f9ee95f519d427bc6cb65524afe3608f49
def _eigh(H, precision, symmetrize, termination_size):
  nrows, ncols = H.shape
  if nrows != ncols:
    raise TypeError(f"Input H of shape {H.shape} must be square.")
  if symmetrize:
    H = 0.5 * (H + H.conj().T)

  if ncols <= termination_size:
    return jnp.linalg.eigh(H)

  evals, evecs = _eigh_work(H, None, precision, termination_size)
  sort_idxs = jnp.argsort(evals)
  evals = evals[sort_idxs]
  evecs = evecs[:, sort_idxs]
  return evals, evecs


def eigh(
    H, precision=lax.Precision.HIGHEST, symmetrize=True, termination_size=128):
  """ Computes the eigendecomposition of the symmetric/Hermitian matrix H.

  Args:
    H: The `n x n` Hermitian input.
    precision: The matmul precision.
    symmetrize: If True, `0.5 * (H + H.conj().T)` rather than `H` is used.
    termination_size: Recursion ends once the blocks reach this linear size.
  Returns:
    vals: The `n` eigenvalues of `H`, sorted from lowest to higest.
    vecs: A unitary matrix such that `vecs[:, i]` is a normalized eigenvector
      of `H` corresponding to `vals[i]`. We have `H @ vecs = vals * vecs` up
      to numerical error.
  """
  return _eigh(H, precision, symmetrize, termination_size)


<<<<<<< HEAD
@jax.partial(jax.jit, static_argnums=(1, 2))
=======
#@jax.partial(jax.jit, static_argnums=(1, 2))
>>>>>>> 417b54f9ee95f519d427bc6cb65524afe3608f49
def _svd(A, precision, termination_size):
  Up, H = jsp.linalg.polar(A)
  S, V = _eigh(H, precision, False, termination_size)
  U = jnp.dot(Up, V, precision=precision)
  return U, S, V.conj().T


def svd(A, precision=lax.Precision.HIGHEST, termination_size=128):
  """ Computes an SVD of `A`.

  Args:
    A: The `m` by `n` input matrix.
    precision: TPU matmul precision.
    termination_size: Recursion ends once the blocks reach this linear size.
  Returns:
    U: An `m` by `m` unitary matrix of `A`'s left singular vectors.
    S: A length-`min(m, n)` vector of `A`'s singular values.
    V_dag: An `n` by `n` unitary matrix of `A`'s conjugate transposed
      right singular vectors.
  """
  return _svd(A, precision, termination_size)
