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

import jax
from jax.experimental.array_api._data_type_functions import (
    _promote_to_default_dtype,
)

def cholesky(x, /, *, upper=False):
  """
  Returns the lower (upper) Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix x.
  """
  return jax.numpy.linalg.cholesky(x, upper=upper)

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
  return jax.numpy.linalg.diagonal(x, offset=offset)

def eigh(x, /):
  """
  Returns an eigenvalue decomposition of a complex Hermitian or real symmetric matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.eigh(x)

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
  return jax.numpy.linalg.matmul(x1, x2)

def matrix_norm(x, /, *, keepdims=False, ord='fro'):
  """
  Computes the matrix norm of a matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.matrix_norm(x, ord=ord, keepdims=keepdims)

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
  return jax.numpy.linalg.matrix_transpose(x)

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
  return jax.numpy.linalg.qr(x, mode=mode)

def slogdet(x, /):
  """
  Returns the sign and the natural logarithm of the absolute value of the determinant of a square matrix (or a stack of square matrices) x.
  """
  return jax.numpy.linalg.slogdet(x)

def solve(x1, x2, /):
  """
  Returns the solution of a square system of linear equations with a unique solution.
  """
  if x2.ndim == 1:
    signature = "(m,m),(m)->(m)"
  else:
    signature = "(m,m),(m,n)->(m,n)"
  return jax.numpy.vectorize(jax.numpy.linalg.solve, signature=signature)(x1, x2)


def svd(x, /, *, full_matrices=True):
  """
  Returns a singular value decomposition (SVD) of a matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.svd(x, full_matrices=full_matrices)

def svdvals(x, /):
  """
  Returns the singular values of a matrix (or a stack of matrices) x.
  """
  return jax.numpy.linalg.svdvals(x)

def tensordot(x1, x2, /, *, axes=2):
  """Returns a tensor contraction of x1 and x2 over specific axes."""
  return jax.numpy.linalg.tensordot(x1, x2, axes=axes)

def trace(x, /, *, offset=0, dtype=None):
  """
  Returns the sum along the specified diagonals of a matrix (or a stack of matrices) x.
  """
  x = _promote_to_default_dtype(x)
  return jax.numpy.trace(x, offset=offset, dtype=dtype, axis1=-2, axis2=-1)

def vecdot(x1, x2, /, *, axis=-1):
  """Computes the (vector) dot product of two arrays."""
  return jax.numpy.linalg.vecdot(x1, x2, axis=axis)

def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
  """Computes the vector norm of a vector (or batch of vectors) x."""
  return jax.numpy.linalg.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
