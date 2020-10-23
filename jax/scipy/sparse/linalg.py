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
from typing import List, Tuple, Callable, Text
import numpy as np
import jax.numpy as jnp
from jax import lax, device_put, ops
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
    _, r, gamma, _, k = value
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

def lanczos_factorization(matvec: Callable, v0: jnp.ndarray,
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
  # TODO (mganahl) confirm that this is a valid matrix norm)
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
    tol: Convergence flag. If the norm of a krylov vector drops below `tol`
      the iteration is terminated.
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
  Vm, alphas, betas, residual, norm, numits, ar_converged = lanczos_factorization(#pylint: disable=line-too-long
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
