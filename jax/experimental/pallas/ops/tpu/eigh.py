# Copyright 2024 The JAX Authors.
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

"""Eighendecomposition of a Hermitian matrix TPU kernel."""

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


def sym_schur2x2(w_tl_ref, w_tr_ref, w_br_ref):
  w_tl = pallas_diag(jnp.real(w_tl_ref))
  w_tr = pallas_diag(w_tr_ref)
  w_br = pallas_diag(jnp.real(w_br_ref))
  is_complex = (
      jnp.iscomplexobj(w_tl) or jnp.iscomplexobj(w_tr) or jnp.iscomplexobj(w_br)
  )
  if is_complex:
    raise NotImplementedError("Complex numbers in eigh.")

  tau = (w_br - w_tl) / (2 * w_tr)
  t = jnp.sqrt(1 + jnp.square(tau))
  t = 1 / (tau + jnp.where(tau >= 0, t, -t))

  # TODO(mvoz): Fudge factor documentation
  kFudgeFactor = 0.1
  tiny = kFudgeFactor * jnp.finfo(float).eps * jnp.ones_like(w_tr)
  off_diag_is_tiny = jnp.abs(w_tr) <= tiny * jnp.minimum(
      jnp.abs(w_tl), jnp.abs(w_br)
  )
  t = jnp.where(off_diag_is_tiny, jnp.zeros_like(t), t)
  c = 1 / jnp.sqrt(1 + jnp.square(t))
  s = t * c

  rt1 = w_tl - t * w_tr
  rt2 = w_br + t * w_tr

  return rt1, rt2, c, s


def permute_cols_in_row(left, right):
  left_columns = jnp.split(left, left.shape[1], axis=1)
  right_columns = jnp.split(right, right.shape[1], axis=1)

  left_out = jnp.concatenate(
      [left_columns[0], right_columns[0]] + left_columns[1:-1], axis=1
  )

  right_out = jnp.concatenate(right_columns[1:] + [left_columns[-1]], axis=1)

  return left_out, right_out


def square_norm(x):
  return x * x


def permute_rows_in_col(top, bottom):
  temp_top_last = top[-1]
  bottom[:-1] = bottom[1:]
  bottom[-1] = temp_top_last
  top[2:] = top[1:-1]
  top[1] = bottom[0]


def permute_rows_in_col_out(top, bottom):
  top_out = jnp.concatenate([top[0:1], bottom[0:1], top[1:-1]])
  bottom_out = jnp.concatenate([bottom[1:], top[-1:]])

  return top_out, bottom_out


def pallas_diag(x):
  return jnp.sum(jax.lax.mul(jnp.eye(x.shape[0]), x), axis=0)


def auto_off_diag(x):
  n = x.shape[0]
  diag_mask = jnp.eye(n, dtype=bool)
  return x[~diag_mask].reshape(n, n - 1)


def fill_diagonal(a, value):
  n, m = a.shape
  mask = jnp.eye(n, m, dtype=bool)
  return jnp.where(mask, value, a)


def set_matrix_diagonal(matrix, diag, k=0):
  matrix = fill_diagonal(matrix, diag)
  return matrix


def sweep(w_tl, w_tr, w_bl, w_br, v_tl, v_tr, v_bl, v_br):
  eigh_2x2 = sym_schur2x2(w_tl, w_tr, w_br)
  rt1, rt2, c, s = eigh_2x2
  # Rotation over Rows
  w_tl, w_tr, w_bl, w_br = (
      w_tl * c[:, None] - w_bl * s[:, None],
      w_tr * c[:, None] - w_br * s[:, None],
      w_tl * s[:, None] + w_bl * c[:, None],
      w_tr * s[:, None] + w_br * c[:, None],
  )
  # Rotation over cols
  w_tl, w_tr, w_bl, w_br = (
      w_tl * c[None, :] - w_tr * s[None, :],
      w_tl * s[None, :] + w_tr * c[None, :],
      w_bl * c[None, :] - w_br * s[None, :],
      w_bl * s[None, :] + w_br * c[None, :],
  )

  w_tl = set_matrix_diagonal(w_tl, rt1)
  w_tr = set_matrix_diagonal(w_tr, 0)
  w_bl = set_matrix_diagonal(w_bl, 0)
  w_br = set_matrix_diagonal(w_br, rt2)

  w_tl, w_tr = permute_cols_in_row(w_tl, w_tr)
  w_bl, w_br = permute_cols_in_row(w_bl, w_br)
  w_tl, w_bl = permute_rows_in_col_out(w_tl, w_bl)
  w_tr, w_br = permute_rows_in_col_out(w_tr, w_br)

  v_tl, v_tr, v_bl, v_br = (
      v_tl * c[:, None] - v_bl * s[:, None],
      v_tr * c[:, None] - v_br * s[:, None],
      v_tl * s[:, None] + v_bl * c[:, None],
      v_tr * s[:, None] + v_br * c[:, None],
  )
  v_tl, v_bl = permute_rows_in_col_out(v_tl, v_bl)
  v_tr, v_br = permute_rows_in_col_out(v_tr, v_br)
  return w_tl, w_tr, w_bl, w_br, v_tl, v_tr, v_bl, v_br


def _jacobi_pallas(A_ref, w_ref, v_ref):
  shape = A_ref.shape
  m, n = shape[-2], shape[-1]

  tl = A_ref[: n // 2, : n // 2]
  bl = A_ref[n // 2 :, : n // 2]
  tr = A_ref[: n // 2, n // 2 :]
  br = A_ref[n // 2 :, n // 2 :]
  v_tl = jnp.eye(n // 2, dtype=A_ref.dtype)

  v_br = jnp.eye(n // 2, dtype=A_ref.dtype)
  v_tr = jnp.zeros(v_tl.shape, A_ref.dtype)
  v_bl = jnp.zeros(v_tl.shape, A_ref.dtype)

  if n % 2:
    raise NotImplementedError("Odd dimensions not yet implemented")

  def condition(*operands):
    tl, tr, bl, br, _, _, _, _ = operands[0]

    def compute_frobenius_norms(w_tl, w_tr, w_bl, w_br):
      square_norm = lambda x: np.real(x * x.conj())
      off_diag = lambda x: jnp.where(jnp.eye(x.shape[-1], dtype=bool), 0, x)

      frobenius_sq_norm = (
          square_norm(w_tl).sum()
          + square_norm(w_tr).sum()
          + square_norm(w_bl).sum()
          + square_norm(w_br).sum()
      )
      off_diagonal_sq_norm = (
          square_norm(off_diag(w_tl)).sum()
          + square_norm(w_tr).sum()
          + square_norm(w_bl).sum()
          + square_norm(off_diag(w_br)).sum()
      )

      return frobenius_sq_norm, off_diagonal_sq_norm

    frobenius_sq_norm, off_diagonal_sq_norm = compute_frobenius_norms(
        tl, tr, bl, br
    )
    tol = frobenius_sq_norm * 1e-6**2
    return tol < off_diagonal_sq_norm

  def sweep_loop_step(*operands):
    tl, tr, bl, br, v_tl, v_tr, v_bl, v_br = operands[0]
    for _ in range(n - 1):
      tl, tr, bl, br, v_tl, v_tr, v_bl, v_br = sweep(
          tl, tr, bl, br, v_tl, v_tr, v_bl, v_br
      )
    return tl, tr, bl, br, v_tl, v_tr, v_bl, v_br

  tl, _, _, br, v_tl, v_tr, v_bl, v_br = jax.lax.while_loop(
      condition,
      sweep_loop_step,
      init_val=(tl, tr, bl, br, v_tl, v_tr, v_bl, v_br),
  )

  tl_diag = pallas_diag(tl).reshape(1, -1)
  br_diag = pallas_diag(br).reshape(1, -1)
  cat = jnp.concatenate((tl_diag, br_diag), axis=1)

  v_top = jnp.concatenate((v_tl, v_tr), axis=A_ref.ndim - 1)
  v_bottom = jnp.concatenate((v_bl, v_br), axis=A_ref.ndim - 1)
  v = jnp.concatenate((v_top, v_bottom), axis=A_ref.ndim - 2)

  _w = cat
  _v = jnp.swapaxes(v, -1, -2)

  w_ref[...] = _w
  v_ref[...] = _v


# TODO(mvoz): Complex is not yet implemented.
# TODO(mvoz): Sort api
# TODO(mvoz): Row major batching
# TODO(mvoz): n % 2 != 0
def eigh(A: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Eigendecomposition of a Hermitian matrix.

  Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate
  symmetric) or a real symmetric matrix.

  Args:
    A: The matrix to decompose.

  Returns:
    A tuple, (v, w), where v is the left eigenvectors and w are the eigenvalues.
    The elements of v and w are not sorted, and have the same dtype as A.

  Raises:
    NotImplementedError: If A.ndim > 2 or if A is not square.
    RuntimeError: If the eigenvalues are not sorted.
  """
  grid = (1,)
  if A.ndim != 2:
    raise NotImplementedError(f"A.ndim > 2 : {A.ndim}")

  m, n = A.shape
  if m != n:
    raise RuntimeError(f"Eigendecomp input arg must be a square, got {m} x {n}")

  v, w = pl.pallas_call(
      _jacobi_pallas,
      grid=grid,
      out_shape=[
          jax.ShapeDtypeStruct(
              (
                  A.shape[0],
              ),
              A.dtype,
          ),
          jax.ShapeDtypeStruct(A.shape, A.dtype),
      ],
      in_specs=[
          pl.BlockSpec(
              A.shape,
              lambda i: (0, 0),
          ),
      ],
      out_specs=[
          pl.BlockSpec(
              (A.shape[0],),
              lambda i: (0,),
          ),
          pl.BlockSpec(
              A.shape,
              lambda i: (0, 0),
          ),
      ],
      interpret=False,
  )(A)
  return (v, w)
