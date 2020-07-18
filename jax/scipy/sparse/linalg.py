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
from jax import lax, device_put
from jax.tree_util import tree_leaves, tree_map, tree_multimap, tree_structure
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

def _vdot_tree_real(x, y):
  return sum(tree_leaves(tree_multimap(_vdot_real_part, x, y)))


def _vdot_tree(x, y):
  return sum(tree_leaves(tree_multimap(partial(
    jnp.vdot, precision=lax.Precision.HIGHEST), x, y)))


def _mul(scalar, tree):
  return tree_map(partial(operator.mul, scalar), tree)


_add = partial(tree_multimap, operator.add)
_sub = partial(tree_multimap, operator.sub)


def _identity(x):
  return x


def _cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):

  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
  bs = _vdot_tree_real(b, b)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

  def cond_fun(value):
    x, r, gamma, p, k = value
    rs = gamma if M is _identity else _vdot_tree_real(r, r)
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, gamma, p, k = value
    Ap = A(p)
    alpha = gamma / _vdot_tree_real(p, Ap)
    x_ = _add(x, _mul(alpha, p))
    r_ = _sub(r, _mul(alpha, Ap))
    z_ = M(r_)
    gamma_ = _vdot_tree_real(r_, z_)
    beta_ = gamma_ / gamma
    p_ = _add(z_, _mul(beta_, p))
    return x_, r_, gamma_, p_, k + 1

  r0 = _sub(b, A(x0))
  p0 = z0 = M(r0)
  gamma0 = _vdot_tree_real(r0, z0)
  initial_value = (x0, r0, gamma0, p0, 0)

  x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

  return x_final


def _bicgstab_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):

  # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.bicgstab
  bs = _vdot_tree_real(b, b)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  # https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB

  def cond_fun(value):
    x, r, *_, k = value
    rs = _vdot_tree_real(r, r)
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, rhat, alpha, omega, rho, p, q, k = value
    rho_ = _vdot_tree(jnp.conj(rhat), r)
    beta = rho_ / rho * alpha / omega
    p_ = _add(r, _mul(beta, _sub(p, _mul(omega, q))))
    phat = M(p_)
    q_ = A(p_)
    alpha_ = rho_ / _vdot_tree(jnp.conj(rhat), q_)
    s = _sub(r, _mul(alpha_, q_))
    # TODO(sunilkpai): stop early?
    #  It requires accessing cond_fun like this
    #  if cond_fun((x, s, r0, alpha_, omega, rho_, p_, q_, k)):
    #    x_ = _add(x, _mul(alpha_, phat))
    #    return x_, s, rhat, alpha_, omega, rho_, p_, q, k
    shat = M(s)
    t = A(shat)
    omega_ = _vdot_tree(jnp.conj(s), t) / _vdot_tree_real(t, t)
    x_ = _add(x, _add(_mul(alpha_, phat), _mul(omega_, shat)))
    r_ = _sub(s, _mul(omega_, t))
    return x_, r_, rhat, alpha_, omega_, rho_, p_, q, k + 1

  r0 = _sub(b, A(x0))
  rho0 = alpha0 = omega0 = 1.0
  initial_value = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0)

  x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

  return x_final


def _shapes(pytree):
  return map(jnp.shape, tree_leaves(pytree))


def isolve(_isolve, A, b, x0=None, *, tol=1e-5, atol=0.0,
           maxiter=None, M=None):
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

  isolve = partial(_isolve, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)

  real_valued = lambda x: not issubclass(x.dtype.type, np.complexfloating)
  symmetric = all(map(real_valued, tree_leaves(b)))
  x = lax.custom_linear_solve(
    A, b, solve=isolve, transpose_solve=isolve, symmetric=symmetric)
  info = None  # TODO(shoyer): return the real iteration count here
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

  return isolve(_cg_solve,
                A=A, b=b, x0=x0, tol=tol, atol=atol,
                maxiter=maxiter, M=M)


def bicgstab(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
  """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``.

  The numerics of JAX's ``bicgstab`` should exact match SciPy's
  ``bicgstab`` (up to numerical precision), but note that the interface
  is slightly different: you need to supply the linear operator ``A`` as
  a function instead of a sparse matrix or ``LinearOperator``.

  Derivatives of ``bicgstab`` are implemented via implicit
  differentiation with another ``bicgstab`` solve, rather than by
  differentiating *through* the solver. They will be accurate only if
  both solves converge.

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

  return isolve(_bicgstab_solve,
                A=A, b=b, x0=x0, tol=tol, atol=atol,
                maxiter=maxiter, M=M)
