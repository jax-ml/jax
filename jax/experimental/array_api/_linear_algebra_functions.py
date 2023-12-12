# Copyright 2023 The JAX Authors.
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
from typing import NamedTuple

import jax
from jax.experimental.array_api._data_type_functions import (
    _promote_to_default_dtype,
)

class EighResult(NamedTuple):
  eigenvalues: jax.Array
  eigenvectors: jax.Array

class QRResult(NamedTuple):
  Q: jax.Array
  R: jax.Array

class SlogdetResult(NamedTuple):
  sign: jax.Array
  logabsdet: jax.Array

class SVDResult(NamedTuple):
  U: jax.Array
  S: jax.Array
  Vh: jax.Array

def cholesky(x, /, *, upper=False):
  """
  Returns the lower (upper) Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix x.
  """
  return jax.numpy.linalg.cholesky(jax.numpy.matrix_transpose(x) if upper else x)

def cross(x1, x2, /, *, axis=-1):
  """
  Returns the cross product of 3-element vectors.
  """
  return jax.numpy.linalg.cross(x1, x2, axis=axis)

def det(x, /):
  """
  Returns the determinant of a square matrix (or a stack of square matrices) x.
  """
  return jax.numpy.linalg.det(x)

def diagonal(x, /, *, offset=0):
  """
  Returns the specified diagonals of a matrix (or a stack of matrices) x.
  """
  f = partial(jax.numpy.diagonal, offset=offset)
  for _ in range(x.ndim - 2):
    f = jax.vmap(f)
  return f(x)

def eigh(x, /):
  """
  Returns an eigenvalue decomposition of a complex Hermitian or real symmetric matrix (or a stack of matrices) x.
  """
  eigenvalues, eigenvectors = jax.numpy.linalg.eigh(x)
  return EighResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

def eigvalsh(x, /):
  """
  Returns the eigenvalues of a complex Hermitian or real symmetric matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.eigvalsh(x)

def inv(x, /):
  """
  Returns the multiplicative inverse of a square matrix (or a stack of square matrices) x.
  """
  return jax.numpy.linalg.inv(x)

def matmul(x1, x2, /):
  """Computes the matrix product."""
  return jax.numpy.matmul(x1, x2)

def matrix_norm(x, /, *, keepdims=False, ord='fro'):
  """
  Computes the matrix norm of a matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.norm(x, ord=ord, keepdims=keepdims, axis=(-1, -2))

def matrix_power(x, n, /):
  """
  Raises a square matrix (or a stack of square matrices) x to an integer power n.
  """
  return jax.numpy.linalg.matrix_power(x, n)

def matrix_rank(x, /, *, rtol=None):
  """
  Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices).
  """
  return jax.numpy.linalg.matrix_rank(x, tol=rtol)

def matrix_transpose(x, /):
  """Transposes a matrix (or a stack of matrices) x."""
  if x.ndim < 2:
    raise ValueError(f"matrix_transpose requres at least 2 dimensions; got {x.ndim=}")
  return jax.lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))

def outer(x1, x2, /):
  """
  Returns the outer product of two vectors x1 and x2.
  """
  return jax.numpy.linalg.outer(x1, x2)

def pinv(x, /, *, rtol=None):
  """
  Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.pinv(x, rcond=rtol)

def qr(x, /, *, mode='reduced'):
  """
  Returns the QR decomposition of a full column rank matrix (or a stack of matrices).
  """
  Q, R = jax.numpy.linalg.qr(x, mode=mode)
  return QRResult(Q=Q, R=R)

def slogdet(x, /):
  """
  Returns the sign and the natural logarithm of the absolute value of the determinant of a square matrix (or a stack of square matrices) x.
  """
  sign, logabsdet = jax.numpy.linalg.slogdet(x)
  return SlogdetResult(sign, logabsdet)

def solve(x1, x2, /):
  """
  Returns the solution of a square system of linear equations with a unique solution.
  """
  if x2.ndim == 1:
    x2 = x2.reshape(*x1.shape[:-2], *x2.shape, 1)
    return jax.numpy.linalg.solve(x1, x2)[..., 0]
  if x2.ndim > x1.ndim:
    x1 = x1.reshape(*x2.shape[:-2], *x1.shape)
  elif x1.ndim > x2.ndim:
    x2 = x2.reshape(*x1.shape[:-2], *x2.shape)
  return jax.numpy.linalg.solve(x1, x2)


def svd(x, /, *, full_matrices=True):
  """
  Returns a singular value decomposition (SVD) of a matrix (or a stack of matrices) x.
  """
  U, S, Vh = jax.numpy.linalg.svd(x, full_matrices=full_matrices)
  return SVDResult(U=U, S=S, Vh=Vh)

def svdvals(x, /):
  """
  Returns the singular values of a matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.svd(x, compute_uv=False)

def tensordot(x1, x2, /, *, axes=2):
  """Returns a tensor contraction of x1 and x2 over specific axes."""
  return jax.numpy.tensordot(x1, x2, axes=axes)

def trace(x, /, *, offset=0, dtype=None):
  """
  Returns the sum along the specified diagonals of a matrix (or a stack of matrices) x.
  """
  x = _promote_to_default_dtype(x)
  return jax.numpy.trace(x, offset=offset, dtype=dtype, axis1=-2, axis2=-1)

def vecdot(x1, x2, /, *, axis=-1):
  """Computes the (vector) dot product of two arrays."""
  rank = max(x1.ndim, x2.ndim)
  x1 = jax.lax.broadcast_to_rank(x1, rank)
  x2 = jax.lax.broadcast_to_rank(x2, rank)
  if x1.shape[axis] != x2.shape[axis]:
    raise ValueError("x1 and x2 must have the same size along specified axis.")
  x1, x2 = jax.numpy.broadcast_arrays(x1, x2)
  x1 = jax.numpy.moveaxis(x1, axis, -1)
  x2 = jax.numpy.moveaxis(x2, axis, -1)
  return jax.numpy.matmul(x1[..., None, :], x2[..., None])[..., 0, 0]

def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
  """Computes the vector norm of a vector (or batch of vectors) x."""
  return jax.numpy.linalg.norm(x, axis=axis, keepdims=keepdims, ord=ord)
