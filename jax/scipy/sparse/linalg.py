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
from jax import lax, device_put, jit
from jax.tree_util import tree_leaves, tree_map, tree_multimap, tree_structure, tree_reduce
from jax.util import safe_map as map


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)

def _vdot_real_part(x, y):
  """Vector dot-product guaranteed to have a real valued result."""
  # all our uses of vdot() in CG are for computing an operator of the form
  # `z^T M z` where `M` is positive definite and Hermitian, so the result is
  # real valued:
  # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
  result = _vdot(x.real, y.real)
  if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
    result += _vdot(x.imag, y.imag)
  return result


# aliases for working with pytrees

def _vdot_tree(x, y, assume_real=True):
  if assume_real:
    return sum(tree_leaves(tree_multimap(_vdot_real_part, x, y)))
  else:
    return sum(tree_leaves(tree_multimap(_vdot, x, y)))

def _mul(scalar, tree):
  return tree_map(partial(operator.mul, scalar), tree)

_add = partial(tree_multimap, operator.add)
_sub = partial(tree_multimap, operator.sub)
_dot_tree = partial(tree_multimap, _dot)


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


def _project_on_columns(A, v):
  v_proj = tree_multimap(
    lambda X, y: jnp.einsum("...n,...->n", X.conj(), y),
    A,
    v,
  )
  return tree_reduce(operator.add, v_proj)


def _safe_normalize(x, return_norm=False):
  norm = jnp.sqrt(_vdot_tree(x, x, assume_real=False))

  normalized_x, norm = lax.cond(
    norm > 1e-12,
    lambda y: (tree_map(lambda v: v / norm, y), norm),
    lambda y: (y, 0.),
    x,
  )
  if return_norm:
    return normalized_x, norm
  else:
    return normalized_x


def _iterative_classical_gram_schmidt(Q, x, iterations=2):
  """Orthogonalize x against the columns of Q."""
  # "twice is enough"
  # http://slepc.upv.es/documentation/reports/str1.pdf

  # This assumes that Q's leaves all have the same dimension in the last
  # axis.
  r = jnp.zeros((tree_leaves(Q)[0].shape[-1]))
  q = x

  for _ in range(iterations):
    h = _project_on_columns(Q, q)
    q = _sub(q, tree_map(lambda X: jnp.dot(X, h), Q))
    r = _add(r, h)
  return q, r


def arnoldi_iteration(A, b, n, M=None):
  # https://en.wikipedia.org/wiki/Arnoldi_iteration#The_Arnoldi_iteration
  if M is None:
    M = _identity
  q = _safe_normalize(b)
  Q = tree_map(
    lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, n),)),
    q,
  )
  H = jnp.zeros((n, n + 1), jnp.result_type(*tree_leaves(b)))

  def step(carry, k):
    Q, H = carry
    q = tree_map(lambda x: x[..., k], Q)
    v = A(M(q))
    v, h = _iterative_classical_gram_schmidt(Q, v, iterations=1)
    v, v_norm = _safe_normalize(v, return_norm=True)
    Q = tree_multimap(lambda X, y: X.at[..., k + 1].set(y), Q, v)
    h = h.at[k + 1].set(v_norm)
    H = H.at[k, :].set(h)
    return (Q, H), None

  (Q, H), _ = lax.scan(step, (Q, H), jnp.arange(n))
  return Q, H


@jit
def _lstsq(a, b):
  return jnp.linalg.lstsq(a, b)[0]


def _gmres(A, b, x0, n, M, residual=None):
  # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
  Q, H = arnoldi_iteration(A, b, n, M)
  if residual is None:
    residual = _sub(b, A(x0))

  beta = jnp.sqrt(_vdot_tree(residual, residual, assume_real=False))
  dtype = beta.dtype
  e1 = jnp.concatenate([jnp.ones((1,), dtype), jnp.zeros((n,), dtype)])
  y = _lstsq(H.T, beta * e1)

  dx = M(tree_map(lambda X: jnp.dot(X[..., :-1], y), Q))
  x = _add(x0, dx)
  return x


def _gmres_solve(A, b, x0, *, tol, atol, restart, maxiter, M):
  bs = _vdot_tree(b, b, assume_real=False)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))
  num_restarts = maxiter // restart

  def cond_fun(value):
    x, residual, k = value
    sqr_error = _vdot_tree(residual, residual, assume_real=False)
    return (sqr_error > atol2) & (k < num_restarts) & ~jnp.isnan(sqr_error)

  def body_fun(value):
    x, residual, k = value
    x = _gmres(A, b, x, restart, M, residual)
    residual = _sub(b, A(x))
    return x, residual, k + 1

  residual = _sub(b, A(x0))
  if num_restarts:
    x, residual, _ = lax.while_loop(
      cond_fun, body_fun, (x0, residual, 0))
  else:
    x = x0

  iters = maxiter % restart
  sqr_error = _vdot_tree(residual, residual)
  if iters > 0:
    x_final = lax.cond(
      sqr_error > atol2,
      true_fun=lambda values: _gmres(A, b, values[0], iters, M, values[1]),
      false_fun=lambda values: values[0],
      operand=(x, residual),
    )
  else:
    x_final = x
  return x_final


def gmres(A, b, x0=None, *, tol=1e-5, atol=0.0, restart=20, maxiter=None,
          M=None):
  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)
  if M is None:
    M = _identity

  size = sum(bi.size for bi in tree_leaves(b))
  if maxiter is None:
    maxiter = 10 * size  # copied from scipy
  if restart > size:
    restart = size

  if tree_structure(x0) != tree_structure(b):
    raise ValueError(
      'x0 and b must have matching tree structure: '
      f'{tree_structure(x0)} vs {tree_structure(b)}')

  b, x0 = device_put((b, x0))

  def _solve(A, b):
    return _gmres_solve(A, b, x0, tol=tol, atol=atol, maxiter=maxiter,
                        restart=restart, M=M)

  x = lax.custom_linear_solve(A, b, solve=_solve, transpose_solve=_solve)
  info = None
  return x, info
