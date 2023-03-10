# Copyright 2021 The JAX Authors.
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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import core
from jax._src.lax import linalg as lax_linalg


# Helpers for working with padded shapes
def _mask(x, dims, alternative=0):
  """Masks `x` up to the dynamic shape `dims`.

  Replaces values outside those dimensions with `alternative`. `alternative` is
  broadcast with `x`.
  """
  assert jnp.ndim(x) == len(dims)
  mask = None
  for i, d in enumerate(dims):
    if d is not None:
      mask_dim_i = lax.broadcasted_iota(jnp.int32, x.shape, i) < d
      mask = mask_dim_i if mask is None else (mask & mask_dim_i)
  return x if mask is None else jnp.where(mask, x, alternative)

def _pad_in_dim(x, low=0, high=0, interior=0, fill_value=0, axis=0):
  pads = [(0, 0, 0)] * x.ndim
  pads[axis] = (low, high, interior)
  return lax.pad(x, jnp.array(fill_value, x.dtype), pads)

def _dynamic_concat(a, b, m, axis=0):
  "Concatenates padded arrays `a` and `b` where the true size of `a` is `m`."
  if m is None:
    return jnp.concatenate([a, b], axis=axis)
  return lax.dynamic_update_slice_in_dim(
      _pad_in_dim(a, high=b.shape[axis], axis=axis), b, m, axis)


def _use_qr(u, m, n, params):
  """QDWH iteration using QR decomposition.

  Args:
  u: a matrix, with static (padded) shape M x N.
  m, n: the dynamic shape of the matrix, where m <= M and n <= N.
  params: the QDWH parameters.
  """
  a, b, c = params
  M, N = u.shape

  y = _dynamic_concat(jnp.sqrt(c) * u, jnp.eye(N, dtype=jnp.dtype(u)), m)
  q, _ = lax_linalg.qr(y, full_matrices=False)
  # q1 = q[:m, :]
  q1 = _mask(lax.slice(q, (0, 0), (M, N)), (m, n))
  # q2 = (q[m:, :]).T.conj()
  q2 = lax.dynamic_slice_in_dim(q, m, N, axis=0)
  q2 = _mask(q2, (n, n)).T.conj()
  e = b / c
  u = (e * u + (a - e) / jnp.sqrt(c) * jnp.einsum('ij,jk->ik', q1, q2))
  return u


def _use_cholesky(u, m, n, params):
  """QDWH iteration using Cholesky decomposition.

  Args:
  u: a matrix, with static (padded) shape M x N
  m, n: the dynamic shape of the matrix, where m <= M and n <= N.
  params: the QDWH parameters.
  """
  a, b, c = params
  _, N = u.shape
  x = c * (u.T.conj() @ u) + jnp.eye(N, dtype=jnp.dtype(u))
  # Pads the lower-right corner with the identity matrix to prevent the Cholesky
  # decomposition from failing due to the matrix not being PSD if padded with
  # zeros.
  x = _mask(x, (n, n), jnp.eye(N, dtype=x.dtype))

  # `y` is lower triangular.
  y = lax_linalg.cholesky(x, symmetrize_input=False)

  z = lax_linalg.triangular_solve(
      y, u.T, left_side=True, lower=True, conjugate_a=True).conj()

  z = lax_linalg.triangular_solve(y, z, left_side=True, lower=True,
                                  transpose_a=True, conjugate_a=True).T.conj()

  e = b / c
  u = e * u + (a - e) * z
  return u

def _qdwh(x, m, n, is_hermitian, max_iterations, eps):
  """QR-based dynamically weighted Halley iteration for polar decomposition."""

  # Estimates `alpha` and `beta = alpha * l`, where `alpha` is an estimate of
  # norm(x, 2) such that `alpha >= norm(x, 2)` and `beta` is a lower bound for
  # the smallest singular value of x.
  if eps is None:
    eps = float(jnp.finfo(x.dtype).eps)
  alpha = (jnp.sqrt(jnp.linalg.norm(x, ord=1)) *
           jnp.sqrt(jnp.linalg.norm(x, ord=jnp.inf))).astype(x.dtype)
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
      return _use_qr(u, m, n, params=(a, b, c))

    def false_fn(u):
      return _use_cholesky(u, m, n, params=(a, b, c))

    u = jax.lax.cond(c > 100, true_fn, false_fn, operand=(u))

    if is_hermitian:
      u = (u + u.T.conj()) / 2.0

    # Checks convergence.
    iterating_l = jnp.abs(1.0 - l) > tol_l
    iterating_u = jnp.linalg.norm(u-u_prev) > tol_norm
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
@functools.partial(jax.jit, static_argnames=('is_hermitian',))
def qdwh(x, *, is_hermitian=False, max_iterations=None, eps=None,
         dynamic_shape: Optional[Tuple[int, int]] = None):
  """QR-based dynamically weighted Halley iteration for polar decomposition.

  Args:
    x: A full-rank matrix, with shape `M x N`. The matrix may be
      padded up to that size from a smaller true shape (``dynamic_shape``).
    is_hermitian: True if `x` is Hermitian. Default to `False`.
    eps: The final result will satisfy
      ``|x_k - x_k-1| < |x_k| * (4*eps)**(1/3)`` where `x_k` is the iterate.
    max_iterations: Iterations will terminate after this many steps even if the
      above is unsatisfied.
    dynamic_shape: the unpadded shape as an ``(m, n)`` tuple; optional.

  Returns:
    A four-tuple of (u, h, num_iters, is_converged) containing the
    polar decomposition of `x = u * h`, the number of iterations to compute `u`,
    and `is_converged`, whose value is `True` when the convergence is achieved
    within the maximum number of iterations.
  """
  is_hermitian = core.concrete_or_error(
      bool, is_hermitian, 'The `is_hermitian` argument must be statically '
      'specified to use `qdwh` within JAX transformations.')

  if max_iterations is None:
    max_iterations = 10

  M, N = x.shape
  if M < N:
    raise ValueError('The input matrix of shape M x N must have M >= N.')
  if dynamic_shape is not None:
    m, n = dynamic_shape
    x = _mask(x, (m, n))
  else:
    m, n = M, N

  with jax.default_matmul_precision('float32'):
    u, h, num_iters, is_converged = _qdwh(x, m, n, is_hermitian, max_iterations,
                                          eps)


  return u, h, num_iters, is_converged
