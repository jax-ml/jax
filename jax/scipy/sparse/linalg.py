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
from jax import lax, device_put, jit, ops
from jax.tree_util import (tree_leaves, tree_map, tree_multimap, tree_structure,
                           Partial)
from jax.util import safe_map as map


def _vdot_real_part(x, y):
  """Vector dot-product guaranteed to have a real valued result."""
  # all our uses of vdot() in CG are for computing an operator of the form
  # `z^T M z` where `M` is positive definite and Hermitian, so the result is
  # real valued:
  # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
  vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)
  result = vdot(x.real, y.real)
  if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
    result += vdot(x.imag, y.imag)
  return result


# aliases for working with pytrees

def _vdot_tree(x, y):
  return sum(tree_leaves(tree_multimap(_vdot_real_part, x, y)))

def _mul(scalar, tree):
  return tree_map(partial(operator.mul, scalar), tree)

_add = partial(tree_multimap, operator.add)
_sub = partial(tree_multimap, operator.sub)


def _identity(x):
  return x


def _cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):

  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
  bs = _vdot_tree(b, b)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

  def cond_fun(value):
    x, r, gamma, p, k = value
    rs = gamma if M is _identity else _vdot_tree(r, r)
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, gamma, p, k = value
    Ap = A(p)
    alpha = gamma / _vdot_tree(p, Ap)
    x_ = _add(x, _mul(alpha, p))
    r_ = _sub(r, _mul(alpha, Ap))
    z_ = M(r_)
    gamma_ = _vdot_tree(r_, z_)
    beta_ = gamma_ / gamma
    p_ = _add(z_, _mul(beta_, p))
    return x_, r_, gamma_, p_, k + 1

  r0 = _sub(b, A(x0))
  p0 = z0 = M(r0)
  gamma0 = _vdot_tree(r0, z0)
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
  real_valued = lambda x: not issubclass(x.dtype.type, np.complexfloating)
  symmetric = all(map(real_valued, tree_leaves(b)))
  x = lax.custom_linear_solve(
      A, b, solve=cg_solve, transpose_solve=cg_solve, symmetric=symmetric)
  info = None  # TODO(shoyer): return the real iteration count here
  return x, info


@partial(jit, static_argnums=(4, 5, 6, 7))
def arnoldi_fact(matvec, v0, Vm, Hm, start, num_krylov_vecs,
                 tol, precision):
  """
  Compute an m-step arnoldi factorization of `matvec`, with
  m <= `num_krylov_vecs`. The factorization will
  do at most `num_krylov_vecs` steps, and stop early if
  an invariant subspace is encountered. Krylov vectors,
  overlaps between krylov vectors and norms are written to
  `Vm` and `Hm`, respectively. Upon termination,
  `Vm` and `Hm` will satisfy the Arnoldi recurrence relation
  ```
  matrix @ Vm.T - Vm.T @ Hm - fm * em = 0
  ```
  with `matrix` the matrix representation of `matvec`,
  `fm = residual * norm` (`residual` and `norm` returned
  by `arnoldi_fact`), and `em` a cartesian basis vector of
  shape `(1, kv.shape[1])` with `em[0, -1] == 1` and 0 elsewhere.
  NOTE: Krylov vectors are currently stored in row-major
  format in `Vm`, i.e. each row of `Vm` is a krylov vector.
  NOTE: The caller is responsible for dtype consistency between
  the inputs, i.e. dtypes between all input arrays have to match.

  Args:
    matvec: The matrix vector product.
    v0: Initial state to `matvec`.
    Vm: An array for storing the krylov vectors. The individual
      vectors are stored as rows in the array.
      The shape of `krylov_vecs` has to be
      (num_krylov_vecs + 1, np.ravel(v0).shape[0]).
    Hm: Matrix of overlaps. The shape has to be
      (num_krylov_vecs + 1,num_krylov_vecs + 1).
    start: Integer denoting the start position where the first
      produced krylov_vector should be inserted into `Vm`
    num_krylov_vecs: Number of krylov iterations, should be identical to
      `Vm.shape[0] + 1`
    tol: Convergence parameter. Iteration is terminated if the norm of a
      krylov-vector falls below `tol`.
  Returns:
    jax.ShapedArray: An array of shape
      `(num_krylov_vecs, np.prod(initial_state.shape))` of krylov vectors.
    jax.ShapedArray: Upper Hessenberg matrix of shape
      `(num_krylov_vecs, num_krylov_vecs`) of the Arnoldi processs.
    jax.ShapedArray: The unnormalized residual of the Arnoldi process.
    int: The norm of the residual.
    int: The number of performed iterations.
    bool: if `True`: iteration hit an invariant subspace.
          if `False`: iteration terminated without encountering

  """

  def iterative_classical_gram_schmidt(vector, krylov_vectors, iterations=2):
    """
    Orthogonalize `vector`  to all rows of `krylov_vectors`, using
    an iterated classical gram schmidt orthogonalization.
    Args:
      vector: Initial vector.
      krylov_vectors: Matrix of krylov vectors, each row is treated as a
        vector.
      iterations: Number of iterations.
    Returns:
      jax.ShapedArray: The orthogonalized vector.
      jax.ShapedArray: The overlaps of `vector` with all previous
        krylov vectors
    """
    vec = vector
    overlaps = 0
    for _ in range(iterations):
      ov = jnp.dot(krylov_vectors.conj(), vec, precision=precision)
      vec = vec - jnp.dot(ov, krylov_vectors, precision=precision)
      overlaps = overlaps + ov
    return vec, overlaps

  Z = jnp.linalg.norm(v0)
  #only normalize if norm > tol, else return zero vector
  v = lax.cond(Z > tol, lambda x: v0 / Z, lambda x: v0 * 0.0, None)
  Vm = Vm.at[start, :].set(v)
  Hm = lax.cond(start > 0, lambda x: Hm.at[x, x - 1].set(Z), lambda x: Hm,
                start)

  # body of the arnoldi iteration
  def body(vals):
    Vm, Hm, previous_vector, _, i = vals
    Av = matvec(previous_vector)

    Av, overlaps = iterative_classical_gram_schmidt(
        Av,
        (i >= jnp.arange(Vm.shape[0]))[:, None] * Vm)
    Hm = Hm.at[:, i].set(overlaps)
    norm = jnp.linalg.norm(Av)

    # only normalize if norm is larger than threshold,
    # otherwise return zero vector
    Av = lax.cond(norm > tol, lambda x: Av / norm, lambda x: Av * 0.0, None)
    Vm, Hm = lax.cond(
        i < num_krylov_vecs - 1,
        lambda x: (Vm.at[i + 1, :].set(Av), Hm.at[i + 1, i].set(norm)),  #pylint: disable=line-too-long
        lambda x: (x[0], x[1]),
        (Vm, Hm, Av, i, norm))

    return [Vm, Hm, Av, norm, i + 1]

  def cond_fun(vals):
    # Continue loop while iteration < num_krylov_vecs and norm > tol
    norm, iteration = vals[3], vals[4]
    counter_done = (iteration >= num_krylov_vecs)
    norm_not_too_small = norm > tol
    continue_iteration = lax.cond(counter_done, lambda x: False,
                                  lambda x: norm_not_too_small, None)
    return continue_iteration

  initial_values = [Vm, Hm, v, Z, start]
  final_values = lax.while_loop(cond_fun, body, initial_values)
  krylov_vectors, H, residual, norm, it = final_values
  return krylov_vectors, H, residual, norm, it, norm < tol


# ######################################################
# #######  NEW SORTING FUCTIONS INSERTED HERE  #########
# ######################################################
@partial(jit, static_argnums=(0,))
def LR_sort(p, evals):
  inds = jnp.argsort(jnp.real(evals), kind='stable')[::-1]
  shifts = evals[inds][-p:]
  return shifts, inds
# #######################################################
# #######################################################


@partial(jit, static_argnums=(3, 4))
def shifted_QR(Vm, Hm, fm, numeig, sort_fun):
  evals, _ = jnp.linalg.eig(Hm)
  shifts, _ = sort_fun(evals)
  # compress arnoldi factorization
  q = jnp.zeros(Hm.shape[0], dtype=Hm.dtype)
  q = q.at[-1].set(1.0)

  def body(i, vals):
    Vm, Hm, q = vals
    Qj, _ = jnp.linalg.qr(Hm - shifts[i] * jnp.eye(Hm.shape[0], dtype=Hm.dtype))
    Hm = Qj.T.conj() @ Hm @ Qj
    Vm = Qj.T @ Vm
    q = q @ Qj
    return Vm, Hm, q

  Vm, Hm, q = lax.fori_loop(0, shifts.shape[0], body, (Vm, Hm, q))
  fk = Vm[numeig, :] * Hm[numeig, numeig - 1] + fm * q[numeig - 1]
  return Vm, Hm, fk


@partial(jit, static_argnums=(3,))
def get_vectors(Vm, unitary, inds, numeig):

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


@partial(jit, static_argnums=(2, 3))
def check_eigvals_convergence_iram(beta_m, Hm, tol, numeig):
  #ARPACK convergence criterion
  eigvals, eigvecs = jnp.linalg.eig(Hm)
  # TODO (mganahl) confirm that this is a valid matrix norm)
  Hm_norm = jnp.linalg.norm(Hm)
  thresh = jnp.maximum(
      jnp.finfo(eigvals.dtype).eps * Hm_norm,
      jnp.abs(eigvals[:numeig]) * tol)
  vals = jnp.abs(eigvecs[numeig - 1, :numeig])
  return jnp.all(beta_m * vals < thresh)


#@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7))
def eigs(matvec, x0,
         restart=20, numeig=4, which="LR", tol=1E-8,
         maxiter=1000, precision=lax.Precision.HIGHEST):
  """
  Implicitly restarted arnoldi factorization of `matvec`. The routine
  finds the lowest `numeig` eigenvector-eigenvalue pairs of `matvec`
  by alternating between compression and re-expansion of an initial
  `restart`-step Arnoldi factorization.

  Note: The caller has to ensure that the dtype of the return value
  of `matvec` matches the dtype of the initial state. Otherwise jax
  will raise a TypeError.

  NOTE: Under certain circumstances, the routine can return spurious
  eigenvalues 0.0: if the Arnoldi iteration terminated early
  (after numits < restart iterations)
  and numeig > numits, then spurious 0.0 eigenvalues will be returned.

  Args:
    matvec: A callable representing the linear operator.
    x0: An starting vector for the iteration.
    restart: Number of krylov vectors of the arnoldi factorization.
      numeig: The number of desired eigenvector-eigenvalue pairs.
    which: Which eigenvalues to target.
      Currently supported: `which = 'LR'` (largest real part).
    tol: Convergence flag. If the norm of a krylov vector drops below `tol`
      the iteration is terminated.
    maxiter: Maximum number of (outer) iteration steps.
    precision: lax.Precision used within lax operations.
  Returns:
    jax.ShapedArray: Eigenvalues
    List: Eigenvectors
    int: Number of inner krylov iterations of the last arnoldi
      factorization.
  """
  dtype = x0.dtype
  dim = x0.shape[0]
  num_expand = restart - numeig

  if (num_expand <= 1) and (restart < dim):
    raise ValueError(f"restart must be between numeig + 1 <"
                     f" restart <= dim = {dim},"
                     f" restart = {restart}")
  if numeig > dim:
    raise ValueError(f"number of requested eigenvalues numeig = {numeig} "
                     f"is larger than the dimension of the operator "
                     f"dim = {dim}")

  # initialize arrays
  Vm = jnp.zeros((restart, dim),
                 dtype=dtype)
  Hm = jnp.zeros((restart, restart), dtype=dtype)
  # perform initial arnoldi factorization
  Vm, Hm, residual, norm, numits, ar_converged = arnoldi_fact(
      matvec, x0, Vm, Hm, 0, restart, tol, precision)
  fm = residual * norm

  ######################################################
  #######  NEW SORTING FUCTIONS INSERTED HERE  #########
  ######################################################
  # sort_fun returns `num_expand` least relevant eigenvalues
  # (those to be projected out)
  if which == 'LR':
    sort_fun = Partial(LR_sort, num_expand)
  else:
    raise ValueError(f"which = {which} not implemented")
  ######################################################
  ######################################################

  it = 1  # we already did one arnoldi factorization
  if maxiter > 1:
    # cast arrays to correct complex dtype
    if Vm.dtype == np.float64:
      dtype = np.complex128
    elif Vm.dtype == np.float32:
      dtype = np.complex64
    elif Vm.dtype == np.complex128:
      dtype = Vm.dtype
    elif Vm.dtype == np.complex64:
      dtype = Vm.dtype
    else:
      raise TypeError(f'dtype {Vm.dtype} not supported')

    Vm = Vm.astype(dtype)
    Hm = Hm.astype(dtype)
    fm = fm.astype(dtype)

  def outer_loop(carry):
    Hm, Vm, fm, it, numits, ar_converged, _, _, = carry
    # perform shifted QR iterations to compress arnoldi factorization
    # Note that ||fk|| typically decreases as one iterates the outer loop
    # indicating that iram converges.
    # ||fk|| = \beta_m in reference above
    Vk, Hk, fk = shifted_QR(Vm, Hm, fm, numeig, sort_fun)
    # reset matrices
    Vk = Vk.at[numeig:, :].set(0.0)
    Hk = Hk.at[numeig:, :].set(0.0)
    Hk = Hk.at[:, numeig:].set(0.0)
    beta_k = jnp.linalg.norm(fk)
    converged = check_eigvals_convergence_iram(beta_k, Hk, tol, numeig)

    def do_arnoldi(vals):
      Vk, Hk, fk = vals
      # restart
      Vm, Hm, residual, norm, numits, ar_converged = arnoldi_fact(
          matvec, fk, Vk, Hk, numeig, restart,
          tol, precision)
      fm = residual * norm
      return Vm, Hm, fm, norm, numits, ar_converged

    res = lax.cond(converged, lambda x:
                   (Vk, Hk, fk, jnp.linalg.norm(fk), numeig, False),
                   lambda x: do_arnoldi((Vk, Hk, fk)), None)

    Vm, Hm, fm, norm, numits, ar_converged = res
    out_vars = [Hm, Vm, fm, it + 1, numits, ar_converged, converged, norm]
    return out_vars

  def cond_fun(carry):
    it, ar_converged, converged = carry[3], carry[5], carry[6]
    return lax.cond(it < maxiter, lambda x: x, lambda x: False,
                    jnp.logical_not(jnp.logical_or(converged, ar_converged)))

  converged = False
  carry = [Hm, Vm, fm, it, numits, ar_converged, converged, norm]
  res = lax.while_loop(cond_fun, outer_loop, carry)
  Hm, Vm = res[0], res[1]
  numits, converged = res[4], res[6]
  # if `ar_converged` then `norm`is below convergence threshold
  # set it to 0.0 in this case to prevent `jnp.linalg.eig` from finding a
  # spurious eigenvalue of order `norm`.
  Hm = Hm.at[numits, numits - 1].set(
      lax.cond(converged, lambda x: Hm.dtype.type(0.0), lambda x: x,
               Hm[numits, numits - 1]))

  # if the Arnoldi-factorization stopped early (after `numit` iterations)
  # before exhausting the allowed size of the Krylov subspace,
  # (i.e. `numit` < 'restart'), set elements
  # at positions m, n with m, n >= `numit` to 0.0.

  # FIXME (mganahl): under certain circumstances, the routine can still
  # return spurious 0 eigenvalues: if arnoldi terminated early
  # (after numits < restart iterations)
  # and numeig > numits, then spurious 0.0 eigenvalues will be returned

  Hm = (numits > jnp.arange(restart))[:, None] * Hm * (
      numits > jnp.arange(restart))[None, :]
  eigvals, U = jnp.linalg.eig(Hm)
  inds = jnp.argsort(jnp.real(eigvals[0:numeig]), kind='stable')[::-1]
  vectors = get_vectors(Vm, U, inds, numeig)
  return eigvals[inds], [
      vectors[n, :] for n in range(numeig)
  ], numits
