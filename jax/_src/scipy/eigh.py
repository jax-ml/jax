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


def _similarity_transform(
    matrix_in, matrix_out, precision=lax.Precision.HIGHEST):
  """ Returns matrix_out.conj().T @ matrix_in @ matrix_out, done in
  order from left to right.
  """
  out = jnp.dot(matrix_out.conj().T, matrix_in, precision=precision)
  return jnp.dot(out, matrix_out, precision=precision)


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


@jax.partial(jax.jit, static_argnums=(3, 4))
def _split_spectrum_jittable(P, H, V0, rank, precision):
  """ The jittable portion of `split_spectrum`. At this point the sizes of the
  relevant matrix blocks have been concretized.

  Args:
    P: Projection matrix.
    H: Matrix to be projected.
    V0: Accumulates the isometries into the projected subspaces.
    rank: Rank of P.
    precision: The matmul precision.
  Returns:
    H1, V1: Projection of H into the column space of P, and the accumulated
            isometry performing that projection.
    H2, V2: Projection of H into the null space of P, and the accumulated
            isometry performing that projection.
  """
  Vm, Vp = _projector_subspace(P, H, rank)
  Hm = _similarity_transform(H, Vm, precision)
  Hp = _similarity_transform(H, Vp, precision)
  if V0 is not None:
    Vm = jnp.dot(V0, Vm, precision=precision)
    Vp = jnp.dot(V0, Vp, precision=precision)
  return Hm, Vm, Hp, Vp


def split_spectrum(H, split_point, V0=None, precision=lax.Precision.HIGHEST):
  """ The Hermitian matrix `H` is split into two matrices `Hm`
  `Hp`, respectively sharing its eigenspaces beneath and above
  its `split_point`th eigenvalue.

  Returns, in addition, `Vm` and `Vp`, isometries such that
  `Hi = Vi.conj().T @ H @ Vi`. If `V0` is not None, `V0 @ Vi` are
  returned instead; this allows the overall isometries mapping from
  an initial input matrix to progressively smaller blocks to be formed.

  Args:
    H: The Hermitian matrix to split.
    split_point: The eigenvalue to split along.
    V0: Matrix of isometries to be updated.
    precision: TPU matmul precision.
  Returns:
    Hm: A Hermitian matrix sharing the eigenvalues of `H` beneath
      `split_point`.
    Vm: An isometry from the input space of `V0` to `Hm`.
    Hp: A Hermitian matrix sharing the eigenvalues of `H` above
      `split_point`.
    Vp: An isometry from the input space of `V0` to `Hp`.
  """
  def _fill_diagonal(X, vals):
    return jax.ops.index_update(X, jnp.diag_indices(X.shape[0]), vals)

  H_shift = _fill_diagonal(H, H.diagonal() - split_point)
  U, _ = jsp.linalg.polar_unitary(H_shift)
  P = -0.5 * _fill_diagonal(U, U.diagonal() - 1.)
  rank = jnp.round(jnp.trace(P)).astype(jnp.int32)
  rank = int(rank)
  return _split_spectrum_jittable(P, H, V0, rank, precision)


def _eigh_work(
    H, V=None, precision=lax.Precision.HIGHEST, termination_size=128):
  """ The main work loop performing the symmetric eigendecomposition of H.
  Each step recursively computes a projector into the space of eigenvalues
  above jnp.mean(jnp.diag(H)). The result of the projections into and out of
  that space, along with the isometries accomplishing these, are then computed.
  This is performed recursively until the projections have size 1, and thus
  store an eigenvalue of the original input; the corresponding isometry is
  the related eigenvector. The results are then composed.

  This function cannot be Jitted because the internal split_spectrum cannot
  be.

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

  split_point = jnp.median(jnp.diag(H))  # TODO: Improve this?
  Hm, Vm, Hp, Vp = split_spectrum(H, split_point, V0=V, precision=precision)
  Hm, Vm = _eigh_work(
    Hm, V=Vm, precision=precision, termination_size=termination_size)
  Hp, Vp = _eigh_work(
    Hp, V=Vp, precision=precision, termination_size=termination_size)

  if Hm.ndim != 1 or Hp.ndim != 1:
    raise ValueError(f"One of Hm.ndim={Hm.ndim} or Hp.ndim={Hp.ndim} != 1 ",
                     "indicating recursion terminated unexpectedly.")

  evals = jnp.hstack((Hm, Hp))
  evecs = jnp.hstack((Vm, Vp))
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
    vals: The `n` eigenvalues of `H`, sorted from lowest to highest.
    vecs: A unitary matrix such that `vecs[:, i]` is a normalized eigenvector
      of `H` corresponding to `vals[i]`. We have `H @ vecs = vals * vecs` up
      to numerical error.
  """
  nrows, ncols = H.shape
  if nrows != ncols:
    raise TypeError(f"Input H of shape {H.shape} must be square.")

  if ncols <= termination_size:
    return jnp.linalg.eigh(H)

  evals, evecs = _eigh_work(H, precision=precision)
  sort_idxs = jnp.argsort(evals)
  evals = evals[sort_idxs]
  evecs = evecs[:, sort_idxs]
  return evals, evecs


def svd(A, precision=lax.Precision.HIGHEST):
  """ Computes an SVD of `A`.

  Args:
    A: The `m` by `n` input matrix.
    precision: TPU matmul precision.
  Returns:
    U: An `m` by `m` unitary matrix of `A`'s left singular vectors.
    S: A length-`min(m, n)` vector of `A`'s singular values.
    V_dag: An `n` by `n` unitary matrix of `A`'s conjugate transposed
      right singular vectors.
  """
  Up, H = jsp.linalg.polar(A)
  S, V = eigh(H, precision=precision)
  U = jnp.dot(Up, V, precision=precision)
  return U, S, V.conj().T
