# Copyright 2022 Google LLC
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

"""A JIT-compatible library for QDWH-based SVD decomposition.

QDWH is short for QR-based dynamically weighted Halley iteration. The Halley
iteration implemented through QR decmopositions is numerically stable and does
not require solving a linear system involving the iteration matrix or
computing its inversion. This is desirable for multicore and heterogeneous
computing systems.

References:
Nakatsukasa, Yuji, and Nicholas J. Higham.
"Stable and efficient spectral divide and conquer algorithms for the symmetric
eigenvalue decomposition and the SVD." SIAM Journal on Scientific Computing 35,
no. 3 (2013): A1325-A1349.
https://epubs.siam.org/doi/abs/10.1137/120876605

Nakatsukasa, Yuji, Zhaojun Bai, and François Gygi.
"Optimizing Halley's iteration for computing the matrix polar decomposition."
SIAM Journal on Matrix Analysis and Applications 31, no. 5 (2010): 2700-2720.
https://epubs.siam.org/doi/abs/10.1137/090774999
"""

import functools

from typing import Sequence

import jax
from jax import core
from jax import lax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(1, 2))
def _svd(a: jnp.ndarray,
         is_hermitian: bool,
         max_iterations: int) -> Sequence[jnp.ndarray]:
  """Singular value decomposition for m x n matrix and m >= n.

  Args:
    a: A matrix of shape `m x n` with `m >= n`.
    is_hermitian: True if `a` is Hermitian.
    max_iterations: The predefined maximum number of iterations of QDWH.

  Returns:
    A 3-tuple (`u`, `s`, `v`), where `u` is a unitary matrix of shape `m x n`,
    `s` is vector of length `n` containing the singular values in the descending
    order, `v` is a unitary matrix of shape `n x n`, and
    `a = (u * s) @ v.T.conj()`.
  """

  u, h, _, _ = lax.linalg.qdwh(a, is_hermitian, max_iterations)

  v, s = lax.linalg.eigh(h)

  # Flips the singular values in descending order.
  s_out = jnp.flip(s)

  # Reorders eigenvectors.
  v_out = jnp.fliplr(v)

  u_out = u @ v_out

  # Makes correction if computed `u` from qdwh is not unitary.
  # Section 5.5 of Nakatsukasa, Yuji, and Nicholas J. Higham. "Stable and
  # efficient spectral divide and conquer algorithms for the symmetric
  # eigenvalue decomposition and the SVD." SIAM Journal on Scientific Computing
  # 35, no. 3 (2013): A1325-A1349.
  def correct_rank_deficiency(u_out):
    u_out, r = lax.linalg.qr(u_out, full_matrices=False)
    u_out = u_out @ jnp.diag(lax.sign(jnp.diag(r)))
    return u_out

  eps = jnp.finfo(a.dtype).eps
  u_out = lax.cond(s[0] < a.shape[1] * eps * s_out[0],
                   correct_rank_deficiency,
                   lambda u_out: u_out,
                   operand=(u_out))

  return (u_out, s_out, v_out)


@functools.partial(jax.jit, static_argnums=(1, 2))
def svd(a: jnp.ndarray,
        is_hermitian: bool = False,
        max_iterations: int = 10) -> Sequence[jnp.ndarray]:
  """Singular value decomposition.

  Args:
    a: A matrix of shape `m x n`.
    is_hermitian: True if `a` is Hermitian.
    max_iterations: The predefined maximum number of iterations of QDWH.

  Returns:
    A 3-tuple (`u`, `s`, `vh`), where `u` is a unitary matrix of shape `m x k`,
    `s` is vector of length `k` containing the singular values in the descending
    order, `vh` is a unitary matrix of shape `k x n`, `k = min(m, n)`, and
    `a = (u * s) @ vh`.
  """

  is_hermitian = core.concrete_or_error(
      bool, is_hermitian, 'The `is_hermitian` argument must be statically '
      'specified to use `qdwh` within JAX transformations.')

  max_iterations = core.concrete_or_error(
      int, max_iterations, 'The `max_iterations` argument must be statically '
      'specified to use `qdwh` within JAX transformations.')

  m, n = a.shape

  is_flip = False
  if m < n:
    a = a.T.conj()
    m, n = a.shape
    is_flip = True

  reduce_to_square = False
  if m > 1.15 * n:
    m = n
    q, a = lax.linalg.qr(a, full_matrices=False)
    reduce_to_square = True

  u_out, s_out, v_out = _svd(a, is_hermitian, max_iterations)

  if reduce_to_square:
    u_out = q @ u_out

  if is_flip:
    return(v_out, s_out, u_out.T.conj())

  return (u_out, s_out, v_out.T.conj())
