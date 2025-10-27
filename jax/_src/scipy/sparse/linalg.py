# Copyright 2020 The JAX Authors.
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

from jax._src import api
from jax._src import dtypes
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.lax import lax as lax_internal
from jax._src.numpy import einsum as jnp_einsum
from jax._src.scipy import linalg as jsp_linalg
from jax._src.tree_util import (tree_leaves, tree_map, tree_structure,
                                tree_reduce, Partial)
from jax._src.typing import Array
from jax._src.util import safe_map as map


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp_einsum.einsum, precision=lax.Precision.HIGHEST)


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
  result = _vdot(x.real, y.real)
  if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
    result += _vdot(x.imag, y.imag)
  return result


def _vdot_real_tree(x, y):
  return sum(tree_leaves(tree_map(_vdot_real_part, x, y)))


def _vdot_tree(x, y):
  return sum(tree_leaves(tree_map(partial(
    jnp.vdot, precision=lax.Precision.HIGHEST), x, y)))


def _norm(x):
  xs = tree_leaves(x)
  return jnp.sqrt(sum(map(_vdot_real_part, xs, xs)))


def _mul(scalar, tree):
  return tree_map(partial(operator.mul, scalar), tree)


_add = partial(tree_map, operator.add)
_sub = partial(tree_map, operator.sub)
_dot_tree = partial(tree_map, _dot)


@Partial
def _identity(x):
  return x


def _normalize_matvec(f):
  """Normalize an argument for computing matrix-vector products."""
  if callable(f):
    return f
  elif isinstance(f, (np.ndarray, Array)):
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
      raise ValueError(
          f'linear operator must be a square matrix, but has shape: {f.shape}')
    return partial(_dot, f)
  elif hasattr(f, '__matmul__'):
    if hasattr(f, 'shape') and len(f.shape) != 2 or f.shape[0] != f.shape[1]:
      raise ValueError(
          f'linear operator must be a square matrix, but has shape: {f.shape}')
    return partial(operator.matmul, f)
  else:
    raise TypeError(
        f'linear operator must be either a function or ndarray: {f}')


def _cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):

  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
  bs = _vdot_real_tree(b, b)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

  def cond_fun(value):
    _, r, gamma, _, k = value
    rs = gamma.real if M is _identity else _vdot_real_tree(r, r)
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, gamma, p, k = value
    Ap = A(p)
    alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
    x_ = _add(x, _mul(alpha, p))
    r_ = _sub(r, _mul(alpha, Ap))
    z_ = M(r_)
    gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
    beta_ = gamma_ / gamma
    p_ = _add(z_, _mul(beta_, p))
    return x_, r_, gamma_, p_, k + 1

  r0 = _sub(b, A(x0))
  p0 = z0 = M(r0)
  dtype = dtypes.result_type(*tree_leaves(p0))
  gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
  initial_value = (x0, r0, gamma0, p0, 0)

  x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

  return x_final


# aliases for working with pytrees

def _bicgstab_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):

  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.bicgstab
  bs = _vdot_real_tree(b, b)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  # https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB

  def cond_fun(value):
    x, r, *_, k = value
    rs = _vdot_real_tree(r, r)
    # the last condition checks breakdown
    return (rs > atol2) & (k < maxiter) & (k >= 0)

  def body_fun(value):
    x, r, rhat, alpha, omega, rho, p, q, k = value
    rho_ = _vdot_tree(rhat, r)
    beta = rho_ / rho * alpha / omega
    p_ = _add(r, _mul(beta, _sub(p, _mul(omega, q))))
    phat = M(p_)
    q_ = A(phat)
    alpha_ = rho_ / _vdot_tree(rhat, q_)
    s = _sub(r, _mul(alpha_, q_))
    exit_early = _vdot_real_tree(s, s) < atol2
    shat = M(s)
    t = A(shat)
    omega_ = _vdot_tree(t, s) / _vdot_tree(t, t)  # make cases?
    x_ = tree_map(partial(jnp.where, exit_early),
                  _add(x, _mul(alpha_, phat)),
                  _add(x, _add(_mul(alpha_, phat), _mul(omega_, shat)))
                  )
    r_ = tree_map(partial(jnp.where, exit_early),
                  s, _sub(s, _mul(omega_, t)))
    k_ = jnp.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
    k_ = jnp.where((rho_ == 0), -10, k_)
    return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_

  r0 = _sub(b, A(x0))
  rho0 = alpha0 = omega0 = lax_internal._convert_element_type(
      1, *dtypes.lattice_result_type(*tree_leaves(b)))
  initial_value = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0)

  x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

  return x_final


def _shapes(pytree):
  return map(np.shape, tree_leaves(pytree))


def _isolve(_isolve_solve, A, b, x0=None, *, tol=1e-5, atol=0.0,
            maxiter=None, M=None, check_symmetric=False):
  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)

  b, x0 = api.device_put((b, x0))

  if maxiter is None:
    size = sum(bi.size for bi in tree_leaves(b))
    maxiter = 10 * size  # copied from scipy

  if M is None:
    M = _identity
  A = _normalize_matvec(A)
  M = _normalize_matvec(M)

  if tree_structure(x0) != tree_structure(b):
    raise ValueError(
        'x0 and b must have matching tree structure: '
        f'{tree_structure(x0)} vs {tree_structure(b)}')

  if _shapes(x0) != _shapes(b):
    raise ValueError(
        'arrays in x0 and b must have matching shapes: '
        f'{_shapes(x0)} vs {_shapes(b)}')

  isolve_solve = partial(
      _isolve_solve, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)

  # real-valued positive-definite linear operators are symmetric
  def real_valued(x):
    return not issubclass(x.dtype.type, np.complexfloating)
  symmetric = all(map(real_valued, tree_leaves(b))) \
    if check_symmetric else False
  x = lax.custom_linear_solve(
      A, b, solve=isolve_solve, transpose_solve=isolve_solve,
      symmetric=symmetric)
  info = None
  return x, info


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
  A: ndarray, function, or matmul-compatible object
      2D array or function that calculates the linear map (matrix-vector
      product) ``Ax`` when called like ``A(x)`` or ``A @ x``. ``A`` must represent
      a hermitian, positive definite matrix, and must return array(s) with the
      same structure and shape as its argument.
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
  x0 : array or tree of arrays
      Starting guess for the solution. Must have the same structure as ``b``.
  tol, atol : float, optional
      Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
      We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
      differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
  maxiter : integer
      Maximum number of iterations.  Iteration will stop after maxiter
      steps even if the specified tolerance has not been achieved.
  M : ndarray, function, or matmul-compatible object
      Preconditioner for A.  The preconditioner should approximate the
      inverse of A.  Effective preconditioning dramatically improves the
      rate of convergence, which implies that fewer iterations are needed
      to reach a given error tolerance.

  See also
  --------
  scipy.sparse.linalg.cg
  jax.lax.custom_linear_solve
  """
  return _isolve(_cg_solve,
                 A=A, b=b, x0=x0, tol=tol, atol=atol,
                 maxiter=maxiter, M=M, check_symmetric=True)


def _safe_normalize(x, thresh=None):
  """
  Returns the L2-normalized vector (which can be a pytree) x, and optionally
  the computed norm. If the computed norm is less than the threshold `thresh`,
  which by default is the machine precision of x's dtype, it will be
  taken to be 0, and the normalized x to be the zero vector.
  """
  norm = _norm(x)
  dtype, weak_type = dtypes.lattice_result_type(*tree_leaves(x))
  if thresh is None:
    thresh = dtypes.finfo(norm.dtype).eps
  thresh = thresh.astype(dtype).real

  use_norm = norm > thresh

  norm_cast = lax_internal._convert_element_type(norm, dtype, weak_type)
  normalized_x = tree_map(lambda y: jnp.where(use_norm, y / norm_cast, 0.0), x)
  norm = jnp.where(use_norm, norm, 0.0)
  return normalized_x, norm


def _project_on_columns(A, v):
  """
  Returns A.T.conj() @ v.
  """
  v_proj = tree_map(
      lambda X, y: _einsum("...n,...->n", X.conj(), y), A, v,
  )
  return tree_reduce(operator.add, v_proj)


def _iterative_classical_gram_schmidt(Q, x, xnorm, max_iterations=2):
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
  xnorm : float
      Norm of x.

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

  # TODO(shoyer): consider switching to only one iteration, like SciPy?

  # This assumes that Q's leaves all have the same dimension in the last
  # axis.
  Q0 = tree_leaves(Q)[0]
  r = jnp.zeros(Q0.shape[-1], dtype=Q0.dtype)
  q = x

  def body_function(carry):
    k, q, r, _, qnorm_scaled_old = carry
    h = _project_on_columns(Q, q)
    Qh = tree_map(lambda X: _dot(X, h), Q)
    q = _sub(q, Qh)
    r = _add(r, h)

    qnorm_scaled_old = qnorm_scaled_old / 2
    _, qnorm = _safe_normalize(q)

    return (k + 1, q, r, qnorm_scaled_old, qnorm)

  def cond_function(carry):
    k, _, r, qnorm_scaled_old, qnorm = carry
    return jnp.logical_and(k < max_iterations, qnorm_scaled_old > qnorm)

  k, q, r, qnorm_scaled_old, qnorm = body_function((0, q, r, xnorm + 1, xnorm))
  k, q, r, _, _ = lax.while_loop(cond_function, body_function,
                                 (k, q, r, qnorm_scaled_old, qnorm))
  return q, r


def _kth_arnoldi_iteration(k, A, M, V, H):
  """
  Performs a single (the k'th) step of the Arnoldi process. Thus,
  adds a new orthonormalized Krylov vector A(M(V[:, k])) to V[:, k+1],
  and that vectors overlaps with the existing Krylov vectors to
  H[k, :]. The tolerance 'tol' sets the threshold at which an invariant
  subspace is declared to have been found, in which case in which case the new
  vector is taken to be the zero vector.
  """
  dtype, _ = dtypes.lattice_result_type(*tree_leaves(V))
  eps = dtypes.finfo(dtype).eps

  v = tree_map(lambda x: x[..., k], V)  # Gets V[:, k]
  v = M(A(v))
  _, v_norm_0 = _safe_normalize(v)
  v, h = _iterative_classical_gram_schmidt(V, v, v_norm_0, max_iterations=2)

  tol = eps * v_norm_0
  unit_v, v_norm_1 = _safe_normalize(v, thresh=tol)
  V = tree_map(lambda X, y: X.at[..., k + 1].set(y), V, unit_v)

  h = h.at[k + 1].set(v_norm_1.astype(dtype))
  H = H.at[k, :].set(h)
  breakdown = v_norm_1 == 0.
  return V, H, breakdown


def _rotate_vectors(H, i, cs, sn):
  x1 = H[i]
  y1 = H[i + 1]
  x2 = cs.conj() * x1 - sn.conj() * y1
  y2 = sn * x1 + cs * y1
  H = H.at[i].set(x2)
  H = H.at[i + 1].set(y2)
  return H


def _givens_rotation(a, b):
  b_zero = abs(b) == 0
  a_lt_b = abs(a) < abs(b)
  t = -jnp.where(a_lt_b, a, b) / jnp.where(a_lt_b, b, a)
  r = lax.rsqrt(1 + abs(t) ** 2).astype(t.dtype)
  cs = jnp.where(b_zero, 1, jnp.where(a_lt_b, r * t, r))
  sn = jnp.where(b_zero, 0, jnp.where(a_lt_b, r, r * t))
  return cs, sn


def _apply_givens_rotations(H_row, givens, k):
  """
  Applies the Givens rotations stored in the vectors cs and sn to the vector
  H_row. Then constructs and applies a new Givens rotation that eliminates
  H_row's k'th element.
  """
  # This call successively applies each of the
  # Givens rotations stored in givens[:, :k] to H_col.

  def apply_ith_rotation(i, H_row):
    return _rotate_vectors(H_row, i, *givens[i, :])
  R_row = lax.fori_loop(0, k, apply_ith_rotation, H_row)

  givens_factors = _givens_rotation(R_row[k], R_row[k + 1])
  givens = givens.at[k, :].set(givens_factors)
  R_row = _rotate_vectors(R_row, k, *givens_factors)
  return R_row, givens


def _gmres_incremental(A, b, x0, unit_residual, residual_norm, ptol, restart, M):
  """
  Implements a single restart of GMRES. The restart-dimensional Krylov subspace
  K(A, x0) = span(A(x0), A@x0, A@A@x0, ..., A^restart @ x0) is built, and the
  projection of the true solution into this subspace is returned.

  This implementation builds the QR factorization during the Arnoldi process.
  """
  # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf

  V = tree_map(
      lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
      unit_residual,
  )
  dtype = dtypes.result_type(*tree_leaves(b))
  # use eye() to avoid constructing a singular matrix in case of early
  # termination
  R = jnp.eye(restart, restart + 1, dtype=dtype)

  givens = jnp.zeros((restart, 2), dtype=dtype)
  beta_vec = jnp.zeros((restart + 1), dtype=dtype)
  beta_vec = beta_vec.at[0].set(residual_norm.astype(dtype))

  def loop_cond(carry):
    k, err, _, _, _, _ = carry
    return jnp.logical_and(k < restart, err > ptol)

  def arnoldi_qr_step(carry):
    k, _, V, R, beta_vec, givens = carry
    V, H, _ = _kth_arnoldi_iteration(k, A, M, V, R)
    R_row, givens = _apply_givens_rotations(H[k, :], givens, k)
    R = R.at[k, :].set(R_row)
    beta_vec = _rotate_vectors(beta_vec, k, *givens[k, :])
    err = abs(beta_vec[k + 1])
    return k + 1, err, V, R, beta_vec, givens

  carry = (0, residual_norm, V, R, beta_vec, givens)
  carry = lax.while_loop(loop_cond, arnoldi_qr_step, carry)
  k, residual_norm, V, R, beta_vec, _ = carry
  del k  # Until we figure out how to pass this to the user.

  y = jsp_linalg.solve_triangular(R[:, :-1].T, beta_vec[:-1])
  dx = tree_map(lambda X: _dot(X[..., :-1], y), V)

  x = _add(x0, dx)
  residual = M(_sub(b, A(x)))
  unit_residual, residual_norm = _safe_normalize(residual)
  # TODO(shoyer): "Inner loop tolerance control" on ptol, like SciPy
  return x, unit_residual, residual_norm


def _lstsq(a, b):
  # faster than jsp_linalg.lstsq
  a2 = _dot(a.T.conj(), a)
  b2 = _dot(a.T.conj(), b)
  return jsp_linalg.solve(a2, b2, assume_a='pos')


def _gmres_batched(A, b, x0, unit_residual, residual_norm, ptol, restart, M):
  """
  Implements a single restart of GMRES. The ``restart``-dimensional Krylov
  subspace
  K(A, x0) = span(A(x0), A@x0, A@A@x0, ..., A^restart @ x0) is built, and the
  projection of the true solution into this subspace is returned.

  This implementation solves a dense linear problem instead of building
  a QR factorization during the Arnoldi process.
  """
  del ptol  # unused
  # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
  V = tree_map(
      lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
      unit_residual,
  )
  dtype, weak_type = dtypes.lattice_result_type(*tree_leaves(b))
  H = lax_internal._convert_element_type(
      jnp.eye(restart, restart + 1, dtype=dtype), weak_type=weak_type)

  def loop_cond(carry):
    _, _, breakdown, k = carry
    return jnp.logical_and(k < restart, jnp.logical_not(breakdown))

  def arnoldi_process(carry):
    V, H, _, k = carry
    V, H, breakdown = _kth_arnoldi_iteration(k, A, M, V, H)
    return V, H, breakdown, k + 1

  carry = (V, H, False, 0)
  V, H, _, _ = lax.while_loop(loop_cond, arnoldi_process, carry)

  beta_vec = jnp.zeros_like(H, shape=(restart + 1,)).at[0].set(residual_norm.astype(dtype))
  y = _lstsq(H.T, beta_vec)
  dx = tree_map(lambda X: _dot(X[..., :-1], y), V)

  x = _add(x0, dx)

  residual = M(_sub(b, A(x)))
  unit_residual, residual_norm = _safe_normalize(residual)
  return x, unit_residual, residual_norm


def _gmres_solve(A, b, x0, atol, ptol, restart, maxiter, M, gmres_func):
  """
  The main function call wrapped by custom_linear_solve. Repeatedly calls GMRES
  to find the projected solution within the order-``restart``
  Krylov space K(A, x0, restart), using the result of the previous projection
  in place of x0 each time. Parameters are the same as in ``gmres`` except:

  atol: Tolerance for norm(A(x) - b), used between restarts.
  ptol: Tolerance for norm(M(A(x) - b)), used within a restart.
  gmres_func: A function performing a single GMRES restart.

  Returns: The solution.
  """
  residual = M(_sub(b, A(x0)))
  unit_residual, residual_norm = _safe_normalize(residual)

  def cond_fun(value):
    _, k, _, residual_norm = value
    return jnp.logical_and(k < maxiter, residual_norm > atol)

  def body_fun(value):
    x, k, unit_residual, residual_norm = value
    x, unit_residual, residual_norm = gmres_func(
        A, b, x, unit_residual, residual_norm, ptol, restart, M)
    return x, k + 1, unit_residual, residual_norm

  initialization = (x0, 0, unit_residual, residual_norm)
  x_final, k, _, err = lax.while_loop(cond_fun, body_fun, initialization)
  _ = k  # Until we can pass this out
  _ = err
  return x_final  # , info


def gmres(A, b, x0=None, *, tol=1e-5, atol=0.0, restart=20, maxiter=None,
          M=None, solve_method='batched'):
  """
  GMRES solves the linear system A x = b for x, given A and b.

  A is specified as a function performing A(vi) -> vf = A @ vi, and in principle
  need not have any particular special properties, such as symmetry. However,
  convergence is often slow for nearly symmetric operators.

  Parameters
  ----------
  A: ndarray, function, or matmul-compatible object
      2D array or function that calculates the linear map (matrix-vector
      product) ``Ax`` when called like ``A(x)`` or ``A @ x``. ``A``
      must return array(s) with the same structure and shape as its argument.
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
  x0 : array or tree of arrays, optional
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
  M : ndarray, function, or matmul-compatible object
      Preconditioner for A.  The preconditioner should approximate the
      inverse of A.  Effective preconditioning dramatically improves the
      rate of convergence, which implies that fewer iterations are needed
      to reach a given error tolerance.
  solve_method : 'incremental' or 'batched'
      The 'incremental' solve method builds a QR decomposition for the Krylov
      subspace incrementally during the GMRES process using Givens rotations.
      This improves numerical stability and gives a free estimate of the
      residual norm that allows for early termination within a single "restart".
      In contrast, the 'batched' solve method solves the least squares problem
      from scratch at the end of each GMRES iteration. It does not allow for
      early termination, but has much less overhead on GPUs.

  See also
  --------
  scipy.sparse.linalg.gmres
  jax.lax.custom_linear_solve
  """

  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)
  if M is None:
    M = _identity
  A = _normalize_matvec(A)
  M = _normalize_matvec(M)

  b, x0 = api.device_put((b, x0))
  size = sum(bi.size for bi in tree_leaves(b))

  if maxiter is None:
    maxiter = 10 * size  # copied from scipy
  restart = min(restart, size)

  if tree_structure(x0) != tree_structure(b):
    raise ValueError(
        'x0 and b must have matching tree structure: '
        f'{tree_structure(x0)} vs {tree_structure(b)}')

  b_norm = _norm(b)
  atol = jnp.maximum(tol * b_norm, atol)

  Mb = M(b)
  Mb_norm = _norm(Mb)
  ptol = Mb_norm * jnp.minimum(1.0, atol / b_norm)

  if solve_method == 'incremental':
    gmres_func = _gmres_incremental
  elif solve_method == 'batched':
    gmres_func = _gmres_batched
  else:
    raise ValueError(f"invalid solve_method {solve_method}, must be either "
                     "'incremental' or 'batched'")

  def _solve(A, b):
    return _gmres_solve(A, b, x0, atol, ptol, restart, maxiter, M, gmres_func)
  x = lax.custom_linear_solve(A, b, solve=_solve, transpose_solve=_solve)

  failed = jnp.isnan(_norm(x))
  info = jnp.where(failed, -1, 0)
  return x, info


def bicgstab(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
  """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``.

  The numerics of JAX's ``bicgstab`` should exact match SciPy's
  ``bicgstab`` (up to numerical precision), but note that the interface
  is slightly different: you need to supply the linear operator ``A`` as
  a function instead of a sparse matrix or ``LinearOperator``.

  As with ``cg``, derivatives of ``bicgstab`` are implemented via implicit
  differentiation with another ``bicgstab`` solve, rather than by
  differentiating *through* the solver. They will be accurate only if
  both solves converge.

  Parameters
  ----------
  A: ndarray, function, or matmul-compatible object
      2D array or function that calculates the linear map (matrix-vector
      product) ``Ax`` when called like ``A(x)`` or ``A @ x``. ``A`` can represent
      any general (nonsymmetric) linear operator, and function must return array(s)
      with the same structure and shape as its argument.
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
  x0 : array or tree of arrays
      Starting guess for the solution. Must have the same structure as ``b``.
  tol, atol : float, optional
      Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
      We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
      differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
  maxiter : integer
      Maximum number of iterations.  Iteration will stop after maxiter
      steps even if the specified tolerance has not been achieved.
  M : ndarray, function, or matmul-compatible object
      Preconditioner for A.  The preconditioner should approximate the
      inverse of A.  Effective preconditioning dramatically improves the
      rate of convergence, which implies that fewer iterations are needed
      to reach a given error tolerance.

  See also
  --------
  scipy.sparse.linalg.bicgstab
  jax.lax.custom_linear_solve
  """

  return _isolve(_bicgstab_solve,
                 A=A, b=b, x0=x0, tol=tol, atol=atol,
                 maxiter=maxiter, M=M)

def _eigs_krylov_schur(A, H, eigvals, eigvecs, residual_norms, howmany,
                       restart, rdtype, cdtype, real, kouter, which, rtol):
  """
  This function implements the inner part of the Krylov-Schur algorithm
  where a Krylov decomposition using the Arnoldi iteration is constructed.
  This decomposition is transformed into a Schur form which is used to
  select the wanted eigenvalues.

  Returns the eigenvalues and eigenvectors as well the convergence and the
  Schur form.
  """

  def prepare_restart(H, eigvals, eigvecs, residual_norms):
    converged = howmany - jnp.sum(jnp.where(
      residual_norms[:howmany] > rtol * jnp.abs(eigvals[:howmany]), 0, 1
    ))
    keep = (3 * restart + 2 * converged) // 5  # Copied from KrylovKit.jl

    keep = lax.cond(H[keep, keep-1] != 0,
                    lambda k: lax.cond(k > 1, lambda x: (x-1), lambda x: (x+1), k),
                    lambda k: k,
                    keep)

    H = H.at[keep, :].set(H[restart, :])
    H = H.at[:, keep].set(0)
    H = H.at[restart, :].set(0)
    H = H.at[keep, keep].set(1)
    eigvecs = tree_map(lambda X: X.at[..., keep].set(X[..., restart]), eigvecs)
    eigvecs = tree_map(lambda X: X.at[..., restart].set(0), eigvecs)

    def update_tensors(i, carry):
      H, v = carry
      H = H.at[i, :].set(0)
      H = H.at[:, i].set(0)
      H = H.at[i, i].set(1)
      v = tree_map(lambda X: X.at[..., i].set(0), v)
      return H, v

    H, eigvecs = lax.fori_loop(keep+1, restart, update_tensors, (H, eigvecs))

    return H, eigvecs, keep

  H, eigvecs, start = lax.cond(kouter > 0,
                               prepare_restart,
                               lambda H, _, eigvecs, __: (H, eigvecs, 0),
                               H, eigvals, eigvecs, residual_norms)

  H = H.T

  def loop_cond(carry):
    _, _, breakdown, k = carry
    return jnp.logical_and(k < restart, jnp.logical_not(breakdown))

  def arnoldi_process(carry):
    eigvecs, H, _, k = carry
    eigvecs, H, breakdown = _kth_arnoldi_iteration(k, A, _identity, eigvecs, H)
    return eigvecs, H, breakdown, k + 1

  carry = (eigvecs, H, False, start)
  eigvecs, H, breakdown, _ = lax.while_loop(loop_cond, arnoldi_process, carry)

  H = H.T

  T, U, sigma = lax.linalg.schur(
    H[:restart, :],
    compute_schur_vectors=True,
    compute_eig_vals=True,
  )

  if which == "LM":
    eigval_sort = jnp.argsort(jnp.abs(sigma), descending=True, stable=True)
  elif which == "SM":
    eigval_sort = jnp.argsort(jnp.abs(sigma), descending=False, stable=True)
  elif which == "LR":
    eigval_sort = jnp.argsort(jnp.real(sigma), descending=True, stable=True)
  elif which == "SR":
    eigval_sort = jnp.argsort(jnp.real(sigma), descending=False, stable=True)
  elif real and (which == "LI" or which == "SI"):
    eigval_sort = jnp.argsort(jnp.abs(jnp.imag(sigma)), descending=True, stable=True)
  elif which == "LI":
    eigval_sort = jnp.argsort(jnp.imag(sigma), descending=True, stable=True)
  elif which == "SI":
    eigval_sort = jnp.argsort(jnp.imag(sigma), descending=False, stable=True)
  else:
    raise ValueError(f"Unknown parameter which, got '{which}'.")

  sigma = sigma[eigval_sort]

  T, U = lax.linalg.schur_reorder(T, U, eigval_sort)

  H = H.at[:restart, :].set(T)
  H = H.at[restart, :].set(H[restart, :] @ U)

  residual_norms = jnp.abs(H[restart, :])

  eigvecs = tree_map(
    lambda X: X.at[..., :restart].set(X[..., :restart] @ U), eigvecs
  )

  return sigma, eigvecs, residual_norms, H

def _eigs_krylov_schur_outer(A, howmany, v0, restart, maxiter, which, rtol):
  """
  In this function the outer loop of the Krylov-Schur algorithm is implemented,
  which checks the convergence of the eigenvalues.
  After convergence or reaching the maximal number of restarts of the algorithm
  the eigenvectors are calculated.

  Implementation follows the description in
  https://slepc.upv.es/release/_downloads/5229480744b7c2533563dee75c16dfde/str7.pdf .

  Returns the solution for the eigenvalues and eigenvectors.
  """
  dtype, weak_type = dtypes.lattice_result_type(*tree_leaves(v0))
  if dtype == np.float32:
    rdtype = np.float32
    cdtype = np.complex64
    real = True
  elif dtype == np.float64:
    rdtype = np.float64
    cdtype = np.complex128
    real = True
  elif dtype == np.complex64:
    rdtype = np.float32
    cdtype = np.complex64
    real = False
  elif dtype == np.complex128:
    rdtype = np.float64
    cdtype = np.complex128
    real = False
  else:
    raise ValueError("Unsupported dtype.")

  eigvals = jnp.ones(restart, dtype=cdtype)
  eigvecs = tree_map(
    lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
    v0,
  )
  residual_norms = jnp.ones(restart, dtype=rdtype) * rtol + 1

  H = jnp.eye(restart + 1, restart, dtype=dtype)

  def cond_fun(value):
    eigvals, _, k, residual_norms, _ = value

    if real and (which == "LI" or which == "SI"):
      def imag_converged_cond(carry):
        _, _, converged, k = carry
        return jnp.logical_and(k < restart, converged < howmany)

      def imag_converged_body(carry):
        residual_norms, eigvals, converged, k = carry
        val = jnp.where(
          jnp.abs(eigvals[k]) != 0,
          residual_norms[k] / jnp.abs(eigvals[k]),
          0
        )
        converged = lax.cond(
          val <= rtol,
          lambda x: x+1,
          lambda x: x,
          converged)
        k = lax.cond(
          jnp.imag(eigvals[k]) != 0,
          lambda x: x+2,
          lambda x: x+1,
          k
        )
        return residual_norms, eigvals, converged, k

      _, _, converged, _ = lax.while_loop(
        imag_converged_cond,
        imag_converged_body,
        (residual_norms, eigvals, 0, 0)
      )

      return jnp.logical_and(k < maxiter, converged < howmany)

    eigval_compare_part = jnp.abs(eigvals[:howmany])
    return jnp.logical_and(
      k < maxiter,
      jnp.any(
        jnp.where(
          jnp.abs(eigval_compare_part) != 0,
          residual_norms[:howmany] / eigval_compare_part,
          0
        ) > rtol
      )
    )

  def body_fun(value):
    eigvals, eigvecs, k, residual_norms, H = value
    eigvals, eigvecs, residual_norms, H = _eigs_krylov_schur(
        A, H, eigvals, eigvecs, residual_norms, howmany, restart,
        rdtype, cdtype, real, k, which, rtol
    )
    return eigvals, eigvecs, k+1, residual_norms, H

  initialization = (eigvals, eigvecs, 0, residual_norms, H)
  eigvals, eigvecs, _, residual_norms, H = lax.while_loop(cond_fun, body_fun,
                                                          initialization)

  V = lax.linalg.schur_eigenvectors(H[:restart, :], eigvals)

  if real and (which == "LI" or which == "SI"):
    if which == "LI":
      eigval_sort = jnp.argsort(jnp.imag(eigvals), descending=True, stable=True)
    elif which == "SI":
      eigval_sort = jnp.argsort(jnp.imag(eigvals), descending=False, stable=True)
    eigvals = eigvals[eigval_sort[:howmany]]
    eigvecs = tree_map(
      lambda X: X[..., :restart] @ V[:, eigval_sort[:howmany]], eigvecs
    )
  else:
    eigvals = eigvals[:howmany]
    eigvecs = tree_map(
      lambda X: X[..., :restart] @ V[:, :howmany], eigvecs
    )

  return eigvals, eigvecs


def eigs(A, k, v0, *, M=None, sigma=None, which="LM", ncv=None, maxiter=None,
         tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None):
  """
  Calculate a subset of the eigenvalues/(right) eigenvectors of a linear operator.

  In difference to SciPy's ``eigs`` method, this function implements the
  Krylov-Schur method which is another variant of the family of Arnoldi-based
  algorithms.

  Parameters
  ----------
  A: ndarray, function, or matmul-compatible object
      2D array or function that calculates the linear map (matrix-vector
      product) ``Ax`` when called like ``A(x)`` or ``A @ x``. ``A`` can represent
      any general (nonsymmetric) linear operator, and function must return array(s)
      with the same structure and shape as its argument.
  k : int
      Number of eigenvalues and eigenvectors which should be calculated. Which
      eigenvalues are calculated is selected by the ``which`` parameter.
  v0 : array or tree of arrays
      Initial guess for the starting vector of algorithm. Can be
      stored as an array or Python container of array(s) with any shape.

  Returns
  -------
  eigvals : array
      The ``k`` eigenvalues sorted in the order defined by ``which``.
  eigvecs : array or tree of arrays
      The ``k`` eigenvectors. Has similar structure to ``v0`` with additional
      new axis indexing the ``i``th eigenvectors corresponding to the ``i``th
      eigenvalue. Not returned if ``return_eigenvectors = False``.

  Other Parameters
  ----------------
  which : str
      Select which kind of eigenvalues should be calculated:
      - ``LM``: Eigenvalues with largest magnitude (euclidean norm)
      - ``SM``: Eigenvalues with smallest magnitude (euclidean norm)
      - ``LR``: Eigenvalues with largest real part
      - ``SR``: Eigenvalues with smallest real part
      - ``LI``: Eigenvalues with largest imaginary part
      - ``SI``: Eigenvalues with smallest imaginary part
  ncv : int
      Number of Lanczos vectors generated. It is recommended to be at least
      :math:`ncv > 2*k`. Default: `min(n, max(2*k + 1, 20))`.
  maxiter : int
      Number of maximal restarts of the algorithm. Default: `10 * n`.
  tol : float
      Relative accuracy required before stopping the algorithm. If set
      to ``None`` or ``0``, machine precision is implied.
  return_eigenvectors : bool
      Flag if the eigenvectors should be returned.
  M : None
      Not implemented.
  sigma : None
      Not implemented.
  Minv : None
      Not implemented.
  OPinv : None
      Not implemented.
  OPpart : None
      Not implemented.

  See also
  --------
  scipy.sparse.linalg.eigs
  """

  if (M is not None
      or sigma is not None
      or Minv is not None
      or OPinv is not None
      or OPpart is not None):
    raise NotImplementedError

  A = _normalize_matvec(A)

  v0 = api.device_put(v0)
  size = sum(bi.size for bi in tree_leaves(v0))
  dtype = dtypes.result_type(*tree_leaves(v0))

  if not (dtypes.issubdtype(dtype, np.floating)
          or dtypes.issubdtype(dtype, np.complexfloating)):
    raise ValueError("Only (complex) floating point dtype supported.")

  A_dtype = api.eval_shape(A, v0)
  if dtypes.result_type(*tree_leaves(A_dtype)) != dtype:
    raise ValueError("The dtype of v0 and A @ v0 has to be the same.")

  if maxiter is None:
    maxiter = 10 * size  # copied from scipy

  if ncv is None:
    ncv = min(size, max(2*k + 1, 20)) # copied from scipy

  if tol is None or tol == 0:
    tol = dtypes.finfo(dtype).eps

  v0, _ = _safe_normalize(v0)

  eigvals, eigvecs = _eigs_krylov_schur_outer(A, k, v0, ncv, maxiter, which, tol)

  if return_eigenvectors:
    return eigvals, eigvecs

  return eigvals
