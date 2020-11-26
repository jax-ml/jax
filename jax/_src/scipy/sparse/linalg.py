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
import operator

import numpy as np
import jax.numpy as jnp
from jax import scipy as jsp
from jax import lax, device_put, ops
from jax.tree_util import (tree_leaves, tree_map, tree_multimap, tree_structure,
                           tree_reduce, Partial)
from jax.util import safe_map as map
from typing import List, Tuple, Callable, Text

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


# aliases for working with pytrees
def _vdot_real_part(x, y):
  """Vector dot-product guaranteed to have a real valued result despite
     possibly complex input. Thus neglects the real-imaginary cross-terms.
     The result is a real float.
  """
  # all our uses of vdot() in CG are for computing an operator of the form
  #  z^H M z
  #  where M is positive definite and Hermitian, so the result is
  # real valued:
  # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
  vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)
  result = vdot(x.real, y.real)
  if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
    result += vdot(x.imag, y.imag)
  return result


def _vdot_real_tree(x, y):
  return sum(tree_leaves(tree_multimap(_vdot_real_part, x, y)))


def _norm_tree(x):
  return jnp.sqrt(_vdot_real_tree(x, x))


def _vdot_tree(x, y):
  return sum(tree_leaves(tree_multimap(_vdot, x, y)))


def _mul(scalar, tree):
  return tree_map(partial(operator.mul, scalar), tree)


def _div(tree, scalar):
  return tree_map(partial(lambda v: v / scalar), tree)


_add = partial(tree_multimap, operator.add)
_sub = partial(tree_multimap, operator.sub)
_dot_tree = partial(tree_multimap, _dot)


@Partial
def _identity(x):
  return x


def _cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):

  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
  bs = _vdot_real_tree(b, b)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

  def cond_fun(value):
    _, r, gamma, _, k = value
    rs = gamma if M is _identity else _vdot_real_tree(r, r)
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, gamma, p, k = value
    Ap = A(p)
    alpha = gamma / _vdot_real_tree(p, Ap)
    x_ = _add(x, _mul(alpha, p))
    r_ = _sub(r, _mul(alpha, Ap))
    z_ = M(r_)
    gamma_ = _vdot_real_tree(r_, z_)
    beta_ = gamma_ / gamma
    p_ = _add(z_, _mul(beta_, p))
    return x_, r_, gamma_, p_, k + 1

  r0 = _sub(b, A(x0))
  p0 = z0 = M(r0)
  gamma0 = _vdot_real_tree(r0, z0)
  initial_value = (x0, r0, gamma0, p0, 0)

  x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

  return x_final


def _shapes(pytree):
  return map(jnp.shape, tree_leaves(pytree))


def cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
  """Use Conjugate Gradient iteration to solve ``Ax = b``.

  The numerics of JAX's ``cg`` should exact match SciPy's ``cg`` (up to
  numerical precision), but note that the interface is slightly different: you
  need to supply the linear operator ``A`` as a function instead of a sparse
  matrix or ``LinearOperator``.

  Derivatives of ``cg`` are implemented via implicit differentiation with
  another ``cg`` solve, rather than by differentiating *through* the solver.
  They will be accurate only if both solves converge.

  Parameters
  ----------
  A : function
      Function that calculates the matrix-vector product ``Ax`` when called
      like ``A(x)``. ``A`` must represent a hermitian, positive definite
      matrix, and must return array(s) with the same structure and shape as its
      argument.
  b : array or tree of arrays
      Right hand side of the linear system representing a single vector. Can be
      stored as an array or Python container of array(s) with any shape.

  Returns
  -------
  x : array or tree of arrays
      The converged solution. Has the same structure as ``b``.
  info : None
      Placeholder for convergence information. In the future, JAX will report
      the number of iterations when convergence is not achieved, like SciPy.

  Other Parameters
  ----------------
  x0 : array
      Starting guess for the solution. Must have the same structure as ``b``.
  tol, atol : float, optional
      Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
      We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
      differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
  maxiter : integer
      Maximum number of iterations.  Iteration will stop after maxiter
      steps even if the specified tolerance has not been achieved.
  M : function
      Preconditioner for A.  The preconditioner should approximate the
      inverse of A.  Effective preconditioning dramatically improves the
      rate of convergence, which implies that fewer iterations are needed
      to reach a given error tolerance.

  See also
  --------
  scipy.sparse.linalg.cg
  jax.lax.custom_linear_solve
  """
  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)

  b, x0 = device_put((b, x0))

  if maxiter is None:
    size = sum(bi.size for bi in tree_leaves(b))
    maxiter = 10 * size  # copied from scipy

  if M is None:
    M = _identity

  if tree_structure(x0) != tree_structure(b):
    raise ValueError(
        'x0 and b must have matching tree structure: '
        f'{tree_structure(x0)} vs {tree_structure(b)}')

  if _shapes(x0) != _shapes(b):
    raise ValueError(
        'arrays in x0 and b must have matching shapes: '
        f'{_shapes(x0)} vs {_shapes(b)}')

  cg_solve = partial(
      _cg_solve, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)

  # real-valued positive-definite linear operators are symmetric
  def real_valued(x):
    return not issubclass(x.dtype.type, np.complexfloating)
  symmetric = all(map(real_valued, tree_leaves(b)))
  x = lax.custom_linear_solve(
      A, b, solve=cg_solve, transpose_solve=cg_solve, symmetric=symmetric)
  info = None  # TODO(shoyer): return the real iteration count here
  return x, info


def _safe_normalize(x, thresh=None):
  """
  Returns the L2-normalized vector (which can be a pytree) x, and optionally
  the computed norm. If the computed norm is less than the threshold `thresh`,
  which by default is the machine precision of x's dtype, it will be
  taken to be 0, and the normalized x to be the zero vector.
  """
  norm = _norm_tree(x)
  dtype = jnp.result_type(*tree_leaves(x))
  if thresh is None:
    thresh = jnp.finfo(norm.dtype).eps
  thresh = thresh.astype(dtype).real

  use_norm = norm > thresh
  normalized_x = tree_map(lambda y: jnp.where(use_norm, y / norm, 0.0), x)
  norm = jnp.where(use_norm, norm, 0.0)
  return normalized_x, norm


def _project_on_columns(A, v):
  """
  Returns A.T.conj() @ v.
  """
  v_proj = tree_multimap(
      lambda X, y: _einsum("...n,...->n", X.conj(), y), A, v,
  )
  return tree_reduce(operator.add, v_proj)


def _iterative_classical_gram_schmidt(Q, x, max_iterations=2):
  """
  Orthogonalize x against the columns of Q. The process is repeated
  up to `max_iterations` times, or fewer if the condition
  ||r|| < (1/sqrt(2)) ||x|| is met earlier (see below for the meaning
  of r and x).

  Parameters
  ----------
  Q : array or tree of arrays
      A matrix of orthonormal columns.
  x : array or tree of arrays
      A vector. It will be replaced with a new vector q which is orthonormal
      to the columns of Q, such that x in span(col(Q), q).

  Returns
  -------
  q : array or tree of arrays
      A unit vector, orthonormal to each column of Q, such that
      x in span(col(Q), q).
  r : array
      Stores the overlaps of x with each vector in Q.
  """
  # "twice is enough"
  # http://slepc.upv.es/documentation/reports/str1.pdf

  # This assumes that Q's leaves all have the same dimension in the last
  # axis.
  r = jnp.zeros((tree_leaves(Q)[0].shape[-1]))
  q = x
  _, xnorm = _safe_normalize(x)
  xnorm_scaled = xnorm / jnp.sqrt(2)

  def body_function(carry):
    k, q, r, qnorm_scaled = carry
    h = _project_on_columns(Q, q)
    Qh = tree_map(lambda X: _dot_tree(X, h), Q)
    q = _sub(q, Qh)
    r = _add(r, h)

    def qnorm_cond(carry):
      k, not_done, _, _ = carry
      return jnp.logical_and(not_done, k < (max_iterations - 1))

    def qnorm(carry):
      k, _, q, qnorm_scaled = carry
      _, qnorm = _safe_normalize(q)
      qnorm_scaled = qnorm / jnp.sqrt(2)
      return (k, False, q, qnorm_scaled)

    init = (k, True, q, qnorm_scaled)
    _, _, q, qnorm_scaled = lax.while_loop(qnorm_cond, qnorm, init)
    return (k + 1, q, r, qnorm_scaled)

  def cond_function(carry):
    k, _, r, qnorm_scaled = carry
    _, rnorm = _safe_normalize(r)
    return jnp.logical_and(k < (max_iterations - 1), rnorm < qnorm_scaled)

  k, q, r, qnorm_scaled = body_function((0, q, r, xnorm_scaled))
  k, q, r, _ = lax.while_loop(cond_function, body_function,
                              (k, q, r, qnorm_scaled))
  return q, r


def _kth_arnoldi_iteration(k, A, M, V, H, tol):
  """
  Performs a single (the k'th) step of the Arnoldi process. Thus,
  adds a new orthonormalized Krylov vector A(M(V[:, k])) to V[:, k+1],
  and that vectors overlaps with the existing Krylov vectors to
  H[k, :]. The tolerance 'tol' sets the threshold at which an invariant
  subspace is declared to have been found, in which case in which case the new
  vector is taken to be the zero vector.
  """

  v = tree_map(lambda x: x[..., k], V)  # Gets V[:, k]
  v = A(M(v))
  v, h = _iterative_classical_gram_schmidt(V, v, max_iterations=2)
  unit_v, v_norm = _safe_normalize(v, thresh=tol)
  V = tree_multimap(lambda X, y: X.at[..., k + 1].set(y), V, unit_v)

  h = h.at[k + 1].set(v_norm)
  H = H.at[k, :].set(h)
  breakdown = v_norm == 0.
  return V, H, breakdown


def _apply_givens_rotations(H_row, givens, k):
  """
  Applies the Givens rotations stored in the vectors cs and sn to the vector
  H_row. Then constructs and applies a new Givens rotation that eliminates
  H_row's k'th element.
  """
  # This call successively applies each of the
  # Givens rotations stored in givens[:, :k] to H_col.

  def apply_ith_rotation(i, H_row):
    cs, sn = givens[i, :]
    H_i = cs * H_row[i] - sn * H_row[i + 1]
    H_ip1 = sn * H_row[i] + cs * H_row[i + 1]
    H_row = H_row.at[i].set(H_i)
    H_row = H_row.at[i + 1].set(H_ip1)
    return H_row

  R_row = lax.fori_loop(0, k, apply_ith_rotation, H_row)

  def givens_rotation(v1, v2):
    t = jnp.sqrt(v1**2 + v2**2)
    cs = v1 / t
    sn = -v2 / t
    return cs, sn
  givens_factors = givens_rotation(R_row[k], R_row[k + 1])
  givens = givens.at[k, :].set(givens_factors)
  cs_k, sn_k = givens_factors

  R_row = R_row.at[k].set(cs_k * R_row[k] - sn_k * R_row[k + 1])
  R_row = R_row.at[k + 1].set(0.)
  return R_row, givens


def _gmres_qr(A, b, x0, unit_residual, residual_norm, inner_tol, restart, M):
  """
  Implements a single restart of GMRES. The restart-dimensional Krylov subspace
  K(A, x0) = span(A(x0), A@x0, A@A@x0, ..., A^restart @ x0) is built, and the
  projection of the true solution into this subspace is returned.

  This implementation builds the QR factorization during the Arnoldi process.
  """
  # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
  #  residual = _sub(b, A(x0))
  #  unit_residual, beta = _safe_normalize(residual)

  V = tree_map(
      lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
      unit_residual,
  )
  dtype = jnp.result_type(*tree_leaves(b))
  R = jnp.eye(restart, restart + 1, dtype=dtype) # eye to avoid constructing
                                                 # a singular matrix in case
                                                 # of early termination.
  b_norm = _norm_tree(b)

  givens = jnp.zeros((restart, 2), dtype=dtype)
  beta_vec = jnp.zeros((restart + 1), dtype=dtype)
  beta_vec = beta_vec.at[0].set(residual_norm)

  def loop_cond(carry):
    k, err, _, _, _, _ = carry
    return jnp.logical_and(k < restart, err > inner_tol)

  def arnoldi_qr_step(carry):
    k, _, V, R, beta_vec, givens = carry
    V, H, _ = _kth_arnoldi_iteration(k, A, M, V, R, inner_tol)
    R_row, givens = _apply_givens_rotations(H[k, :], givens, k)
    R = R.at[k, :].set(R_row[:])
    cs, sn = givens[k, :] * beta_vec[k]
    beta_vec = beta_vec.at[k].set(cs)
    beta_vec = beta_vec.at[k + 1].set(sn)
    err = jnp.abs(sn) / b_norm
    return k + 1, err, V, R, beta_vec, givens

  carry = (0, residual_norm, V, R, beta_vec, givens)
  carry = lax.while_loop(loop_cond, arnoldi_qr_step, carry)
  k, residual_norm, V, R, beta_vec, _ = carry
  del k  # Until we figure out how to pass this to the user.

  y = jsp.linalg.solve_triangular(R[:, :-1].T, beta_vec[:-1])
  Vy = tree_map(lambda X: _dot(X[..., :-1], y), V)
  dx = M(Vy)

  x = _add(x0, dx)
  residual = _sub(b, A(x))
  unit_residual, residual_norm = _safe_normalize(residual)
  return x, unit_residual, residual_norm


def _gmres_plain(A, b, x0, unit_residual, residual_norm, inner_tol, restart, M):
  """
  Implements a single restart of GMRES. The ``restart``-dimensional Krylov
  subspace
  K(A, x0) = span(A(x0), A@x0, A@A@x0, ..., A^restart @ x0) is built, and the
  projection of the true solution into this subspace is returned.

  This implementation solves a dense linear problem instead of building
  a QR factorization during the Arnoldi process.
  """
  # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
  V = tree_map(
      lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
      unit_residual,
  )
  dtype = jnp.result_type(*tree_leaves(b))
  H = jnp.eye(restart, restart + 1, dtype=dtype)

  def loop_cond(carry):
    _, _, breakdown, k = carry
    return jnp.logical_and(k < restart, jnp.logical_not(breakdown))

  def arnoldi_process(carry):
    V, H, _, k = carry
    V, H, breakdown = _kth_arnoldi_iteration(k, A, M, V, H, inner_tol)
    return V, H, breakdown, k + 1

  carry = (V, H, False, 0)
  V, H, _, _ = lax.while_loop(loop_cond, arnoldi_process, carry)

  # The following is equivalent to:
  beta_vec = jnp.zeros((restart,), dtype=dtype)
  beta_vec = beta_vec.at[0].set(residual_norm) # it really is the original value
  y = jsp.linalg.solve(H[:, :-1].T, beta_vec)
  Vy = tree_map(lambda X: _dot(X[..., :-1], y), V)

  dx = M(Vy)
  x = _add(x0, dx)

  residual = _sub(b, A(x))
  unit_residual, residual_norm = _safe_normalize(residual)
  return x, unit_residual, residual_norm


def _gmres_solve(A, b, x0, outer_tol, inner_tol, restart, maxiter, M,
                 gmres_func):
  """
  The main function call wrapped by custom_linear_solve. Repeatedly calls GMRES
  to find the projected solution within the order-``restart``
  Krylov space K(A, x0, restart), using the result of the previous projection
  in place of x0 each time. Parameters are the same as in ``gmres`` except:

  outer_tol: Tolerance to be used between restarts.
  inner_tol: Tolerance used within a restart.
  gmres_func: A function performing a single GMRES restart.

  Returns: The solution.
  """
  residual = _sub(b, A(x0))
  unit_residual, residual_norm = _safe_normalize(residual)

  def cond_fun(value):
    _, k, _, residual_norm = value
    return jnp.logical_and(k < maxiter, residual_norm > outer_tol)

  def body_fun(value):
    x, k, unit_residual, residual_norm = value
    x, unit_residual, residual_norm = gmres_func(A, b, x, unit_residual,
                                                 residual_norm, inner_tol,
                                                 restart, M)
    return x, k + 1, unit_residual, residual_norm

  initialization = (x0, 0, unit_residual, residual_norm)
  x_final, k, _, err = lax.while_loop(cond_fun, body_fun, initialization)
  _ = k # Until we can pass this out
  _ = err
  return x_final  # , info


def _gmres(A, b, x0=None, *, tol=1e-5, atol=0.0, restart=20, maxiter=None,
           M=None, qr_mode=False):
  """
  GMRES solves the linear system A x = b for x, given A and b.

  A is specified as a function performing A(vi) -> vf = A @ vi, and in principle
  need not have any particular special properties, such as symmetry. However,
  convergence is often slow for nearly symmetric operators.

  Parameters
  ----------
  A: function
     Function that calculates the linear map (matrix-vector product)
     ``Ax`` when called like ``A(x)``. ``A`` must return array(s) with the same
     structure and shape as its argument.
  b : array or tree of arrays
      Right hand side of the linear system representing a single vector. Can be
      stored as an array or Python container of array(s) with any shape.

  Returns
  -------
  x : array or tree of arrays
      The converged solution. Has the same structure as ``b``.
  info : None
      Placeholder for convergence information. In the future, JAX will report
      the number of iterations when convergence is not achieved, like SciPy.

  Other Parameters
  ----------------
  x0 : array, optional
       Starting guess for the solution. Must have the same structure as ``b``.
       If this is unspecified, zeroes are used.
  tol, atol : float, optional
      Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
      We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
      differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``gmres``.
  restart : integer, optional
      Size of the Krylov subspace ("number of iterations") built between
      restarts. GMRES works by approximating the true solution x as its
      projection into a Krylov space of this dimension - this parameter
      therefore bounds the maximum accuracy achievable from any guess
      solution. Larger values increase both number of iterations and iteration
      cost, but may be necessary for convergence. The algorithm terminates
      early if convergence is achieved before the full subspace is built.
      Default is 20.
  maxiter : integer
      Maximum number of times to rebuild the size-``restart`` Krylov space
      starting from the solution found at the last iteration. If GMRES
      halts or is very slow, decreasing this parameter may help.
      Default is infinite.
  M : function
      Preconditioner for A.  The preconditioner should approximate the
      inverse of A.  Effective preconditioning dramatically improves the
      rate of convergence, which implies that fewer iterations are needed
      to reach a given error tolerance.
  qr_mode : bool
      If True, the algorithm builds an internal Krylov subspace using a QR
      based algorithm, which reduces overhead and improved stability. However,
      it may degrade performance significantly on GPUs or TPUs, in which case
      this flag should be set False.

  See also
  --------
  scipy.sparse.linalg.gmres
  jax.lax.custom_linear_solve
  """

  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)
  if M is None:
    M = _identity

  b, x0 = device_put((b, x0))
  size = sum(bi.size for bi in tree_leaves(b))

  if maxiter is None:
    maxiter = 10 * size  # copied from scipy
  restart = min(restart, size)

  if tree_structure(x0) != tree_structure(b):
    raise ValueError(
        'x0 and b must have matching tree structure: '
        f'{tree_structure(x0)} vs {tree_structure(b)}')

  b_norm = _norm_tree(b)
  if b_norm == 0:
    return b, 0
  outer_tol = jnp.maximum(tol * b_norm, atol)

  Mb = M(b)
  Mb_norm = _norm_tree(Mb)
  inner_tol = Mb_norm * min(1.0, outer_tol / b_norm)

  if qr_mode:
    def _solve(A, b):
      return _gmres_solve(A, b, x0, outer_tol, inner_tol, restart, maxiter, M,
                          _gmres_plain)
  else:
    def _solve(A, b):
      return _gmres_solve(A, b, x0, outer_tol, inner_tol, restart, maxiter, M,
                          _gmres_qr)

  x = lax.custom_linear_solve(A, b, solve=_solve, transpose_solve=_solve)

  failed = jnp.isnan(_norm_tree(x))
  info = jnp.where(failed, x=-1, y=0)
  return x, info


def iterative_classical_gram_schmidt(vector, krylov_vectors,
                                     precision=lax.Precision.HIGHEST,
                                     iterations=2):
  """
  Orthogonalize `vector`  to all rows of `krylov_vectors`, using
  an iterated classical gram schmidt orthogonalization.
  Args:
    vector: Initial vector.
    krylov_vectors: Matrix of krylov vectors, each row is treated as a
      vector.
    iterations: Number of iterations.
  Returns:
    jax.numpy.ndarray: The orthogonalized vector.
    jax.numpy.ndarray: The overlaps of `vector` with all previous
      krylov vectors
  """
  vec = vector
  overlaps = 0
  for _ in range(iterations):
    ov = jnp.dot(krylov_vectors.conj(), vec, precision=precision)
    vec = vec - jnp.dot(ov, krylov_vectors, precision=precision)
    overlaps = overlaps + ov
  return vec, overlaps

def _lanczos_factorization(matvec: Callable, v0: jnp.ndarray,
    Vm: jnp.ndarray, alphas: jnp.ndarray, betas: jnp.ndarray,
    start: int, num_krylov_vecs: int, tol: float, precision):
  """
  Compute an m-step lanczos factorization of `matvec`, with
  m <=`num_krylov_vecs`. The factorization will
  do at most `num_krylov_vecs` steps, and terminate early
  if an invariant subspace is encountered. The returned arrays
  `alphas`, `betas` and `Vm` will satisfy the Lanczos recurrence relation
  ```
  matrix @ Vm - Vm @ Hm - fm * em = 0
  ```
  with `matrix` the matrix representation of `matvec`,
  `Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(betas.conj(), 1)`
  `fm=residual * norm`, and `em` a cartesian basis vector of shape
  `(1, kv.shape[1])` with `em[0, -1] == 1` and 0 elsewhere.

  Note that the caller is responsible for dtype consistency between
  the inputs, i.e. dtypes between all input arrays have to match.

  Args:
    matvec: The matrix vector product.
    v0: Initial state to `matvec`.
    Vm: An array for storing the krylov vectors. The individual
      vectors are stored as rows.
      The shape of `krylov_vecs` has to be
      (num_krylov_vecs, np.ravel(v0).shape[0]).
    alphas: An array for storing the diagonal elements of the reduced
      operator.
    betas: An array for storing the lower diagonal elements of the
      reduced operator.
    start: Integer denoting the start position where the first
      produced krylov_vector should be inserted into `Vm`
    num_krylov_vecs: Number of krylov iterations, should be identical to
      `Vm.shape[0]`
    tol: Convergence parameter. Iteration is terminated if the norm of a
      krylov-vector falls below `tol`.

  Returns:
    jax.numpy.ndarray: An array of shape
      `(num_krylov_vecs, np.prod(initial_state.shape))` of krylov vectors.
    jax.numpy.ndarray: The diagonal elements of the tridiagonal reduced
      operator ("alphas")
    jax.numpy.ndarray: The lower-diagonal elements of the tridiagonal reduced
      operator ("betas")
    jax.numpy.ndarray: The unnormalized residual of the Lanczos process.
    float: The norm of the residual.
    int: The number of performed iterations.
    bool: if `True`: iteration hit an invariant subspace.
          if `False`: iteration terminated without encountering
          an invariant subspace.
  """

  shape = v0.shape
  Z = jnp.linalg.norm(v0)
  #only normalize if norm > tol, else return zero vector
  v = lax.cond(Z > tol, lambda x: v0 / Z, lambda x: v0 * 0.0, None)
  Vm = Vm.at[start, :].set(jnp.ravel(v))
  betas = lax.cond(
      start > 0,
      lambda x: betas.at[start - 1].set(Z),
      lambda x: betas, start)

  # body of the lanczos iteration
  def body(vals):
    Vm, alphas, betas, previous_vector, _, i = vals
    Av = matvec(previous_vector)
    Av, overlaps = iterative_classical_gram_schmidt(
        Av.ravel(),
        (i >= jnp.arange(Vm.shape[0]))[:, None] * Vm, precision)
    alphas = alphas.at[i].set(overlaps[i])
    norm = jnp.linalg.norm(Av)
    Av = jnp.reshape(Av, shape)
    # only normalize if norm is larger than threshold,
    # otherwise return zero vector
    Av = lax.cond(norm > tol, lambda x: Av/norm, lambda x: Av * 0.0, None)
    Vm, betas = lax.cond(
        i < num_krylov_vecs - 1,
        lambda x: (Vm.at[i + 1, :].set(Av.ravel()), betas.at[i].set(norm)),
        lambda x: (Vm, betas),
        None)

    return [Vm, alphas, betas, Av, norm, i + 1]

  def cond_fun(vals):
    # Continue loop while iteration < num_krylov_vecs and norm > tol
    norm, iteration = vals[4], vals[5]
    counter_done = (iteration >= num_krylov_vecs)
    norm_not_too_small = norm > tol
    continue_iteration = lax.cond(counter_done, lambda x: False,
                                      lambda x: norm_not_too_small, None)
    return continue_iteration

  initial_values = [Vm, alphas, betas, v, Z, start]
  final_values = lax.while_loop(cond_fun, body, initial_values)
  Vm, alphas, betas, residual, norm, it = final_values
  return Vm, alphas, betas, residual, norm, it, norm < tol

def SA_sort(
    p: int,
    evals: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  inds = jnp.argsort(evals)
  shifts = evals[inds][-p:]
  return shifts, inds

def LA_sort(p: int,
            evals: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  inds = jnp.argsort(evals)[::-1]
  shifts = evals[inds][-p:]
  return shifts, inds

def shifted_QR(
    Vm: jnp.ndarray, Hm: jnp.ndarray, fm: jnp.ndarray,
    shifts: jnp.ndarray,
    numeig: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  # compress factorization
  q = jnp.zeros(Hm.shape[0], dtype=Hm.dtype)
  q = q.at[-1].set(1.0)

  def body(i, vals):
    Vm, Hm, q = vals
    shift = shifts[i] * jnp.eye(Hm.shape[0], dtype=Hm.dtype)
    Qj, R = jnp.linalg.qr(Hm - shift)
    Hm = jnp.matmul(R, Qj, precision=lax.Precision.HIGHEST) + shift
    Vm = jnp.matmul(Qj.T, Vm, precision=lax.Precision.HIGHEST)
    q = jnp.matmul(q, Qj, precision=lax.Precision.HIGHEST)
    return Vm, Hm, q

  Vm, Hm, q = lax.fori_loop(0, shifts.shape[0], body,
                                (Vm, Hm, q))
  fk = Vm[numeig, :] * Hm[numeig, numeig - 1] + fm * q[numeig - 1]
  return Vm, Hm, fk

def get_vectors(Vm: jnp.ndarray, unitary: jnp.ndarray,
                inds: jnp.ndarray, numeig: int) -> jnp.ndarray:

  def body_vector(i, states):
    dim = unitary.shape[1]
    n, m = jnp.divmod(i, dim)
    states = ops.index_add(states, ops.index[n, :],
                           Vm[m, :] * unitary[m, inds[n]])
    return states

  state_vectors = jnp.zeros([numeig, Vm.shape[1]], dtype=Vm.dtype)
  state_vectors = lax.fori_loop(0, numeig * Vm.shape[0], body_vector,
                                    state_vectors)
  state_norms = jnp.linalg.norm(state_vectors, axis=1)
  state_vectors = state_vectors / state_norms[:, None]
  return state_vectors


def check_eigvals_convergence(beta_m: float, Hm: jnp.ndarray, Hm_norm: float,
                              tol: float) -> bool:
  eigvals, eigvecs = jnp.linalg.eigh(Hm)
  # TODO (mganahl): confirm that this is a valid matrix norm
  thresh = jnp.maximum(
      jnp.finfo(eigvals.dtype).eps * Hm_norm,
      jnp.abs(eigvals) * tol)
  vals = jnp.abs(eigvecs[-1, :])
  return jnp.all(beta_m * vals < thresh)


def eigsh(matvec: Callable,#pylint: disable=too-many-statements
          initial_state: jnp.ndarray,
          num_krylov_vecs: int,
          numeig: int, which: Text, tol: float, maxiter: int,
          precision) -> Tuple[jnp.ndarray, List[jnp.ndarray], int]:
  """
  Implicitly restarted Lanczos factorization of `matvec`. The routine
  finds the lowest `numeig` eigenvector-eigenvalue pairs of `matvec`
  by alternating between compression and re-expansion of an initial
  `num_krylov_vecs`-step Lanczos factorization.

  Note: The caller has to ensure that the dtype of the return value
  of `matvec` matches the dtype of the initial state. Otherwise jax
  will raise a TypeError.

  NOTE: Under certain circumstances, the routine can return spurious
  eigenvalues 0.0: if the Lanczos iteration terminated early
  (after numits < num_krylov_vecs iterations)
  and numeig > numits, then spurious 0.0 eigenvalues will be returned.

  References:
  http://emis.impa.br/EMIS/journals/ETNA/vol.2.1994/pp1-21.dir/pp1-21.pdf
  http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

  Args:
    matvec: A callable representing the linear operator.
    initial_state: An starting vector for the iteration.
    num_krylov_vecs: Number of krylov vectors of the Lanczos factorization.
      numeig: The number of desired eigenvector-eigenvalue pairs.
    which: Which eigenvalues to target.
      Currently supported: `which = 'LR'` (largest real part).
    tol: Desired precision of computed eigenvalues.
    maxiter: Maximum number of (outer) iteration steps.
    precision: jax.lax.Precision used within lax operations.

  Returns:
    jax.numpy.ndarray: Eigenvalues, sorted in most-desired order
      (most desired first).
    List: Eigenvectors.
    int: Number of inner Krylov iterations of the last Lanczos
      factorization.
  """

  shape = initial_state.shape
  dtype = initial_state.dtype

  dim = np.prod(shape).astype(np.int32)
  num_expand = num_krylov_vecs - numeig
  #note: the second part of the cond is for testing purposes
  if num_krylov_vecs <= numeig < dim:
    raise ValueError(f"num_krylov_vecs must be between numeig <"
                     f" num_krylov_vecs <= dim = {dim},"
                     f" num_krylov_vecs = {num_krylov_vecs}")
  if numeig > dim:
    raise ValueError(f"number of requested eigenvalues numeig = {numeig} "
                     f"is larger than the dimension of the operator "
                     f"dim = {dim}")

  # initialize arrays
  Vm = jnp.zeros(
      (num_krylov_vecs, jnp.ravel(initial_state).shape[0]), dtype=dtype)
  alphas = jnp.zeros(num_krylov_vecs, dtype=dtype)
  betas = jnp.zeros(num_krylov_vecs - 1, dtype=dtype)

  # perform initial lanczos factorization
  Vm, alphas, betas, residual, norm, numits, ar_converged = _lanczos_factorization(#pylint: disable=line-too-long
      matvec, initial_state, Vm, alphas, betas, 0, num_krylov_vecs, tol,
      precision)
  fm = residual.ravel() * norm

  # sort_fun returns `num_expand` least relevant eigenvalues
  # (those to be removed by shifted QR)
  if which == 'LA':
    sort_fun = Partial(LA_sort, num_expand)
  elif which == 'SA':
    sort_fun = Partial(SA_sort, num_expand)
  else:
    raise ValueError(f"which = {which} not implemented")

  it = 1  # we already did one lanczos factorization
  def outer_loop(carry):
    alphas, betas, Vm, fm, it, numits, ar_converged, _, _ = carry
    # pack alphas and betas into tridiagonal matrix
    Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(
        betas.conj(), 1)
    evals, _ = jnp.linalg.eigh(Hm)
    shifts, _ = sort_fun(evals)
    # perform shifted QR iterations to compress lanczos factorization
    # Note that ||fk|| typically decreases as one iterates the outer loop
    # indicating that irlm converges.
    # ||fk|| = \beta_m in references above

    Vk, Hk, fk = shifted_QR(Vm, Hm, fm, shifts, numeig)
    # extract new alphas and betas
    alphas = jnp.diag(Hk)
    betas = jnp.diag(Hk, -1)
    alphas = alphas.at[numeig:].set(0.0)
    betas = betas.at[numeig-1:].set(0.0)
    beta_k = jnp.linalg.norm(fk)
    Hktest = Hk[:numeig, :numeig]
    matnorm = jnp.linalg.norm(Hktest)
    converged = check_eigvals_convergence(beta_k, Hktest, matnorm, tol)
    #####################################################
    # fake conditional statement using while control flow
    # only perform a lanczos factorization if `not converged`
    def do_lanczos(vals):
      Vk, alphas, betas, fk, _, _, _, _ = vals
      # restart
      Vm, alphas, betas, residual, norm, numits, ar_converged = lanczos_factorization(#pylint: disable=line-too-long
          matvec, jnp.reshape(fk, shape), Vk, alphas, betas,
          numeig, num_krylov_vecs, tol, precision)
      fm = residual.ravel() * norm
      return [Vm, alphas, betas, fm, norm, numits, ar_converged, False]

    def cond_lanczos(vals):
      return vals[7]

    res = lax.while_loop(cond_lanczos, do_lanczos, [
        Vk, alphas, betas, fk,
        jnp.linalg.norm(fk), numeig, False,
        jnp.logical_not(converged)
    ])
    Vm, alphas, betas, fm, norm, numits, ar_converged = res[0:7]
    #####################################################

    out_vars = [
        alphas, betas, Vm, fm, it + 1, numits, ar_converged, converged, norm]
    return out_vars

  def cond_fun(carry):
    it, ar_converged, converged = carry[4], carry[6], carry[7]
    return lax.cond(
        it < maxiter, lambda x: x, lambda x: False,
        jnp.logical_not(jnp.logical_or(converged, ar_converged)))

  converged = False
  carry = [alphas, betas, Vm, fm, it, numits, ar_converged, converged, norm]
  res = lax.while_loop(cond_fun, outer_loop, carry)
  alphas, betas, Vm = res[0], res[1], res[2]
  numits, ar_converged, converged = res[5], res[6], res[7]
  Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(
      betas.conj(), 1)
  # FIXME (mganahl): under certain circumstances, the routine can still
  # return spurious 0 eigenvalues: if lanczos terminated early
  # (after numits < num_krylov_vecs iterations)
  # and numeig > numits, then spurious 0.0 eigenvalues will be returned
  Hm = (numits > jnp.arange(num_krylov_vecs))[:, None] * Hm * (
      numits > jnp.arange(num_krylov_vecs))[None, :]

  eigvals, U = jnp.linalg.eigh(Hm)
  inds = sort_fun(eigvals)[1][:numeig]
  vectors = get_vectors(Vm, U, inds, numeig)
  return eigvals[inds], [
      jnp.reshape(vectors[n, :], shape) for n in range(numeig)
  ], numits
