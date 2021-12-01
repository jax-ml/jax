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

"""A JIT-compatible library for QDWH-based polar decomposition.

QDWH is short for QR-based dynamically weighted Halley iteration. The Halley
iteration implemented through QR decmopositions does not require matrix
inversion. This is desirable for multicore and heterogeneous computing systems.

Reference: Nakatsukasa, Yuji, Zhaojun Bai, and François Gygi.
"Optimizing Halley's iteration for computing the matrix polar decomposition."
SIAM Journal on Matrix Analysis and Applications 31, no. 5 (2010): 2700-2720.
https://epubs.siam.org/doi/abs/10.1137/090774999
"""

import functools

import jax
from jax import core
import jax.numpy as jnp
from jax._src.lax import linalg as lax_linalg


def _use_qr(u, params):
  """Uses QR decomposition."""
  a, b, c = params
  m, n = u.shape
  y = jnp.concatenate([jnp.sqrt(c) * u, jnp.eye(n)])
  q, _ = lax_linalg.qr(y, full_matrices=False)
  q1 = q[:m, :]
  q2 = (q[m:, :]).T.conj()
  e = b / c
  u = (e * u + (a - e) / jnp.sqrt(c) * jnp.einsum('ij,jk->ik', q1, q2))
  return u


def _use_cholesky(u, params):
  """Uses Cholesky decomposition."""
  a, b, c = params
  _, n = u.shape
  x = c * u.T.conj() @ u + jnp.eye(n)

  # `y` is lower triangular.
  y = lax_linalg.cholesky(x, symmetrize_input=False)

  z = lax_linalg.triangular_solve(
      y, u.T, left_side=True, lower=True, conjugate_a=True).conj()

  z = lax_linalg.triangular_solve(y, z, left_side=True, lower=True,
                                  transpose_a=True, conjugate_a=True).T.conj()

  e = b / c
  u = e * u + (a - e) * z
  return u


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _qdwh(x, is_symmetric, max_iterations):
  """QR-based dynamically weighted Halley iteration for polar decomposition."""

  # Estimates `alpha` and `beta = alpha * l`, where `alpha` is an estimate of
  # norm(x, 2) such that `alpha >= norm(x, 2)` and `beta` is a lower bound for
  # the smallest singular value of x.
  eps = jnp.finfo(x.dtype).eps
  alpha = jnp.sqrt(jnp.linalg.norm(x, ord=1) * jnp.linalg.norm(x, ord=jnp.inf))
  l = eps

  u = x / alpha

  # Iteration tolerances.
  tol_l = 10.0 * eps / 2.0
  tol_norm = jnp.cbrt(tol_l)

  def cond_fun(state):
    _, _, _, is_unconverged, is_not_max_iteration = state
    return jnp.logical_and(is_unconverged, is_not_max_iteration)

  def body_fun(state):
    u, l, iter_idx, _, _ = state

    u_prev = u

    # Computes parameters.
    l2 = l**2
    dd = jnp.cbrt(4.0 * (1.0 / l2 - 1.0) / l2)
    sqd = jnp.sqrt(1.0 + dd)
    a = (sqd + jnp.sqrt(8.0 - 4.0 * dd + 8.0 * (2.0 - l2) / (l2 * sqd)) / 2)
    a = jnp.real(a)
    b = (a - 1.0)**2 / 4.0
    c = a + b - 1.0

    # Updates l.
    l = l * (a + b * l2) / (1.0 + c * l2)

    # Uses QR or Cholesky decomposition.
    def true_fn(u):
      return _use_qr(u, params=(a, b, c))

    def false_fn(u):
      return _use_cholesky(u, params=(a, b, c))

    u = jax.lax.cond(c > 100, true_fn, false_fn, operand=(u))

    if is_symmetric:
      u = (u + u.T.conj()) / 2.0

    # Checks convergence.
    iterating_l = jnp.abs(1.0 - l) > tol_l
    iterating_u = jnp.linalg.norm((u-u_prev)) > tol_norm
    is_unconverged = jnp.logical_or(iterating_l, iterating_u)

    is_not_max_iteration = iter_idx < max_iterations

    return u, l, iter_idx + 1, is_unconverged, is_not_max_iteration

  iter_idx = 1
  is_unconverged = True
  is_not_max_iteration = True
  u, _, num_iters, is_unconverged, _ = jax.lax.while_loop(
      cond_fun=cond_fun, body_fun=body_fun,
      init_val=(u, l, iter_idx, is_unconverged, is_not_max_iteration))

  # Applies Newton-Schulz refinement for better accuracy.
  u = 1.5 * u - 0.5 * u @ (u.T.conj() @ u)

  h = u.T.conj() @ x
  h = (h + h.T.conj()) / 2.0

  # Converged within the maximum number of iterations.
  is_converged = jnp.logical_not(is_unconverged)

  return u, h, num_iters - 1, is_converged


# TODO: Add pivoting.
def qdwh(x, is_symmetric, max_iterations=10):
  """QR-based dynamically weighted Halley iteration for polar decomposition.

  Args:
    x: A full-rank matrix of shape `m x n` with `m  >= n`.
    is_symmetric: True if `x` is symmetric.
    max_iterations: The predefined maximum number of iterations.

  Returns:
    A four-tuple of (u, h, num_iters, is_converged) containing the
    polar decomposition of `x = u * h`, the number of iterations to compute `u`,
    and `is_converged`, whose value is `True` when the convergence is achieved
    within the maximum number of iterations.
  """
  m, n = x.shape

  if m < n:
    raise ValueError('The input matrix of shape m x n must have m >= n.')

  max_iterations = core.concrete_or_error(
      int, max_iterations, 'The `max_iterations` argument must be statically '
      'specified to use `qdwh` within JAX transformations.')

  is_symmetric = core.concrete_or_error(
      bool, is_symmetric, 'The `is_symmetric` argument must be statically '
      'specified to use `qdwh` within JAX transformations.')

  if is_symmetric:
    eps = jnp.finfo(x.dtype).eps
    tol = 50.0 * eps
    relative_diff = jnp.linalg.norm(x - x.T.conj()) / jnp.linalg.norm(x)
    if relative_diff > tol:
      raise ValueError('The input `x` is NOT symmetric because '
                       '`norm(x-x.H) / norm(x)` is {}, which is greater than '
                       'the tolerance {}.'.format(relative_diff, tol))

  with jax.default_matmul_precision('float32'):
     u, h, num_iters, is_converged = _qdwh(x, is_symmetric, max_iterations)

  return u, h, num_iters, is_converged
