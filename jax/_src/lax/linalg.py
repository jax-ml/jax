# Copyright 2018 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Callable
import enum
from functools import partial
import math
import string
from typing import Any, Literal, overload

import numpy as np

from jax import lax

from jax._src import ad_util
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src.core import ShapedArray, is_constant_dim, is_constant_shape
from jax._src.custom_partitioning_sharding_rule import (
    sdy_sharding_rule_to_mlir, str_to_sdy_sharding_rule)
from jax._src import ffi
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import control_flow
from jax._src.lax import eigh as lax_eigh
from jax._src.lax import lax as lax_internal
from jax._src.lax import svd as lax_svd
from jax._src.lax import utils as lax_utils
from jax._src.lax.lax import _float, _complex, _int
from jax._src.lib import gpu_linalg
from jax._src.lib import gpu_solver
from jax._src.lib import gpu_sparse
from jax._src.lib import lapack
from jax._src.lib import version as jaxlib_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import hlo
from jax._src.partition_spec import PartitionSpec as P
from jax._src.typing import Array, ArrayLike


def register_module_custom_calls(module):
  if hasattr(module, "registrations"):
    for platform, targets in module.registrations().items():
      for name, value, api_version in targets:
        ffi.register_ffi_target(
            name, value, platform=platform, api_version=api_version
        )
  if hasattr(module, "batch_partitionable_targets"):
    for name in module.batch_partitionable_targets():
      ffi.register_ffi_target_as_batch_partitionable(name)


register_module_custom_calls(gpu_linalg)
register_module_custom_calls(gpu_solver)
register_module_custom_calls(gpu_sparse)
register_module_custom_calls(lapack)


# Top-level functions in alphabetical order.

def cholesky(x: Array, *, symmetrize_input: bool = True) -> Array:
  r"""Cholesky decomposition.

  Computes the Cholesky decomposition

  .. math::
    A = L . L^H

  of square matrices, :math:`A`, such that :math:`L`
  is lower triangular. The matrices of :math:`A` must be positive-definite and
  either Hermitian, if complex, or symmetric, if real.

  Args:
    x: A batch of square Hermitian (symmetric if real) positive-definite
      matrices with shape ``[..., n, n]``.
    symmetrize_input: If ``True``, the matrix is symmetrized before Cholesky
      decomposition by computing :math:`\frac{1}{2}(x + x^H)`. If ``False``,
      only the lower triangle of ``x`` is used; the upper triangle is ignored
      and not accessed.

  Returns:
    The Cholesky decomposition as a matrix with the same dtype as ``x`` and
    shape ``[..., n, n]``. If Cholesky decomposition fails, returns a matrix
    full of NaNs. The behavior on failure may change in the future.
  """
  if symmetrize_input:
    x = symmetrize(x)
  return _tril(cholesky_p.bind(x))


def cholesky_update(r_matrix: ArrayLike, w_vector: ArrayLike) -> Array:
  r"""Cholesky rank-1 update.

  Given a Cholesky decomposition :math:`A = R.T \, R` and a vector :math:`w`,
  computes the Cholesky decomposition of :math:`A + w \, w.T` in :math:`O(N^2)`
  time.

  Args:
    r_matrix: An upper-triangular matrix (R) such that :math:`A = R^T \, R`.
    w_vector: A vector :math:`w` for rank-1 update.

  Returns:
    A new upper-triangular matrix :math:`R` defining the Cholesky decomposition
    of :math:`A + w \, w^T`.
  """
  r_matrix, w_vector = core.standard_insert_pvary(r_matrix, w_vector)
  return cholesky_update_p.bind(r_matrix, w_vector)


def eig(
    x: ArrayLike,
    *,
    compute_left_eigenvectors: bool = True,
    compute_right_eigenvectors: bool = True,
    use_magma: bool | None = None,
) -> list[Array]:
  """Eigendecomposition of a general matrix.

  Nonsymmetric eigendecomposition is only implemented on CPU and GPU. On GPU,
  the default implementation calls LAPACK directly on the host CPU, but an
  experimental GPU implementation using `MAGMA <https://icl.utk.edu/magma/>`_
  is also available. The MAGMA implementation is typically slower than the
  equivalent LAPACK implementation for small matrices (less than about 2048),
  but it may perform better for larger matrices.

  To enable the MAGMA implementation, you must install MAGMA yourself (there
  are Debian and conda-forge packages, or you can build from source). Then set
  the ``use_magma`` argument to ``True``, or set the ``jax_use_magma``
  configuration variable to ``"on"`` or ``"auto"``:

  .. code-block:: python

      jax.config.update('jax_use_magma', 'on')

  JAX will try to ``dlopen`` the installed MAGMA shared library, raising an
  error if it is not found. To explicitly specify the path to the MAGMA
  library, set the environment variable `JAX_GPU_MAGMA_PATH` to the full
  installation path.

  If ``jax_use_magma`` is set to ``"auto"``, the MAGMA implementation will
  be used if the library can be found, and the input matrix is sufficiently
  large (>= 2048x2048).

  Args:
    x: A batch of square matrices with shape ``[..., n, n]``.
    compute_left_eigenvectors: If true, the left eigenvectors will be computed.
    compute_right_eigenvectors: If true, the right eigenvectors will be
      computed.
    use_magma: Locally override the ``jax_use_magma`` flag. If ``True``, the
      eigendecomposition is computed using MAGMA. If ``False``, the computation
      is done using LAPACK on to the host CPU. If ``None`` (default), the
      behavior is controlled by the ``jax_use_magma`` flag. This argument
      is only used on GPU.

  Returns:
    The eigendecomposition of ``x``, which is a tuple of the form
    ``(w, vl, vr)`` where ``w`` are the eigenvalues, ``vl`` are the left
    eigenvectors, and ``vr`` are the right eigenvectors. ``vl`` and ``vr`` are
    optional and will only be included if ``compute_left_eigenvectors`` or
    ``compute_right_eigenvectors`` respectively are ``True``.

    If the eigendecomposition fails, then arrays full of NaNs will be returned
    for that batch element.
  """
  return eig_p.bind(x, compute_left_eigenvectors=compute_left_eigenvectors,
                    compute_right_eigenvectors=compute_right_eigenvectors,
                    use_magma=use_magma)


def eigh(
    x: Array,
    *,
    lower: bool = True,
    symmetrize_input: bool = True,
    sort_eigenvalues: bool = True,
    subset_by_index: tuple[int, int] | None = None,
) -> tuple[Array, Array]:
  r"""Eigendecomposition of a Hermitian matrix.

  Computes the eigenvectors and eigenvalues of a complex Hermitian or real
  symmetric square matrix.

  Args:
    x: A batch of square complex Hermitian or real symmetric matrices with shape
      ``[..., n, n]``.
    lower: If ``symmetrize_input`` is ``False``, describes which triangle of the
      input matrix to use. If ``symmetrize_input`` is ``False``, only the
      triangle given by ``lower`` is accessed; the other triangle is ignored and
      not accessed.
    symmetrize_input: If ``True``, the matrix is symmetrized before the
      eigendecomposition by computing :math:`\frac{1}{2}(x + x^H)`.
    sort_eigenvalues: If ``True``, the eigenvalues will be sorted in ascending
      order. If ``False`` the eigenvalues are returned in an
      implementation-defined order.
    subset_by_index: Optional 2-tuple [start, end] indicating the range of
      indices of eigenvalues to compute. For example, is ``range_select`` =
      [n-2,n], then ``eigh`` computes the two largest eigenvalues and their
      eigenvectors.

  Returns:
    A tuple ``(v, w)``.

    ``v`` is an array with the same dtype as ``x`` such that ``v[..., :, i]`` is
    the normalized eigenvector corresponding to eigenvalue ``w[..., i]``.

    ``w`` is an array with the same dtype as ``x`` (or its real counterpart if
    complex) with shape ``[..., d]`` containing the eigenvalues of ``x`` in
    ascending order(each repeated according to its multiplicity).
    If ``subset_by_index`` is ``None`` then ``d`` is equal to ``n``. Otherwise
    ``d`` is equal to ``subset_by_index[1] - subset_by_index[0]``.
  """
  if symmetrize_input:
    x = symmetrize(x)
  v, w = eigh_p.bind(
      x,
      lower=lower,
      sort_eigenvalues=sort_eigenvalues,
      subset_by_index=subset_by_index,
  )
  return v, w


def hessenberg(a: ArrayLike) -> tuple[Array, Array]:
  """Reduces a square matrix to upper Hessenberg form.

  Currently implemented on CPU only.

  Args:
    a: A floating point or complex square matrix or batch of matrices.

  Returns:
    A ``(a, taus)`` pair, where the upper triangle and first subdiagonal of
    ``a`` contain the upper Hessenberg matrix, and the elements below the first
    subdiagonal contain the Householder reflectors. For each Householder
    reflector ``taus`` contains the scalar factors of the elementary Householder
    reflectors.
  """
  return hessenberg_p.bind(a)


def householder_product(a: ArrayLike, taus: ArrayLike) -> Array:
  """Product of elementary Householder reflectors.

  Args:
    a: A matrix with shape ``[..., m, n]``, whose lower triangle contains
      elementary Householder reflectors.
    taus: A vector with shape ``[..., k]``, where ``k < min(m, n)``, containing
      the scalar factors of the elementary Householder reflectors.

  Returns:
    A batch of orthogonal (unitary) matrices with the same shape as ``a``,
    containing the products of the elementary Householder reflectors.
  """
  a, taus = core.standard_insert_pvary(a, taus)
  return householder_product_p.bind(a, taus)


def lu(x: ArrayLike) -> tuple[Array, Array, Array]:
  r"""LU decomposition with partial pivoting.

  Computes the matrix decomposition:

  .. math::
    P \, A = L \, U

  where :math:`P` is a permutation of the rows of :math:`A`, :math:`L` is a
  lower-triangular matrix with unit-diagonal elements, and :math:`U` is an
  upper-triangular matrix.

  Args:
    x: A batch of matrices with shape ``[..., m, n]``.

  Returns:
    A tuple ``(lu, pivots, permutation)``.

    ``lu`` is a batch of matrices with the same shape and dtype as ``x``
    containing the :math:`L` matrix in its lower triangle and the :math:`U`
    matrix in its upper triangle. The (unit) diagonal elements of :math:`L` are
    not represented explicitly.

    ``pivots`` is an int32 array with shape ``[..., min(m, n)]`` representing a
    sequence of row swaps that should be performed on :math:`A`.

    ``permutation`` is an alternative representation of the sequence of row
    swaps as a permutation, represented as an int32 array with shape
    ``[..., m]``.
  """
  return lu_p.bind(x)


def lu_pivots_to_permutation(pivots: ArrayLike, permutation_size: int) -> Array:
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `pivots` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    pivots: an int32 array of shape (..., k) of row swaps to perform
    permutation_size: the size of the output permutation. Has to be >= k.

  Returns:
    An int32 array of shape (..., permutation_size).
  """
  return lu_pivots_to_permutation_p.bind(
      pivots, permutation_size=permutation_size)


@overload
def qr(x: ArrayLike, *, pivoting: Literal[False], full_matrices: bool = True,
      use_magma: bool | None = None) -> tuple[Array, Array]:
  ...

@overload
def qr(x: ArrayLike, *, pivoting: Literal[True], full_matrices: bool = True,
      use_magma: bool | None = None) -> tuple[Array, Array, Array]:
  ...

@overload
def qr(x: ArrayLike, *, pivoting: bool = False, full_matrices: bool = True,
      use_magma: bool | None = None
      ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  ...

def qr(x: ArrayLike, *, pivoting: bool = False, full_matrices: bool = True,
       use_magma: bool | None = None
      ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  r"""QR decomposition.

  Computes the QR decomposition

  .. math::
    A = Q \, R

  of matrices :math:`A`, such that :math:`Q` is a unitary (orthogonal) matrix,
  and :math:`R` is an upper-triangular matrix.

  Args:
    x: A batch of matrices with shape ``[..., m, n]``.
    pivoting: Allows the QR decomposition to be rank-revealing. If ``True``,
      compute the column pivoted decomposition ``A[:, P] = Q @ R``, where ``P``
      is chosen such that the diagonal of ``R`` is non-increasing. Currently
      supported on CPU and GPU backends only.
    full_matrices: Determines if full or reduced matrices are returned; see
      below.
    use_magma: Locally override the ``jax_use_magma`` flag. If ``True``, the
      pivoted `qr` factorization is computed using MAGMA. If ``False``, the
      computation is done using LAPACK on the host CPU. If ``None`` (default),
      the behavior is controlled by the ``jax_use_magma`` flag. This argument is
      only used on GPU.

  Returns:
    A pair of arrays ``(q, r)``, if ``pivoting=False``, otherwise ``(q, r, p)``.

    Array ``q`` is a unitary (orthogonal) matrix,
    with shape ``[..., m, m]`` if ``full_matrices=True``, or
    ``[..., m, min(m, n)]`` if ``full_matrices=False``.

    Array ``r`` is an upper-triangular matrix with shape ``[..., m, n]`` if
    ``full_matrices=True``, or ``[..., min(m, n), n]`` if
    ``full_matrices=False``.

    Array ``p`` is an index vector with shape [..., n]

  Notes:
    - `MAGMA <https://icl.utk.edu/magma/>`_ support is experimental - see
      :func:`jax.lax.linalg.eig` for further assumptions and limitations.
    - If ``jax_use_magma`` is set to ``"auto"``, the MAGMA implementation will
      be used if the library can be found, and the input matrix is sufficiently
      large (has at least 2048 columns).
  """
  q, r, *p = qr_p.bind(x, pivoting=pivoting, full_matrices=full_matrices,
                       use_magma=use_magma)
  if pivoting:
    return q, r, p[0]
  return q, r


def schur(
    x: ArrayLike,
    *,
    compute_schur_vectors: bool = True,
    sort_eig_vals: bool = False,
    select_callable: Callable[..., Any] | None = None,
) -> tuple[Array, Array]:
  r"""Schur decomposition.

  Only implemented on CPU.

  Computes the Schur decomposition:

  .. math::
    A = Q \, U \, Q^{-H}

  for a square matrix :math:`A`.

  Args:
    x: A batch of square matrices with shape ``[..., m, m]``.
    compute_schur_vectors: If ``True``, compute the Schur vectors ::math:`Q`,
      otherwise only :math:`U` is computed.
    sort_eig_vals: Unused.
    select_callable: Unused.

  Returns:
    A pair of arrays ``U, Q``, if ``compute_schur_vectors=True``, otherwise
    only ``U`` is returned.
  """
  return schur_p.bind(
      x,
      compute_schur_vectors=compute_schur_vectors,
      sort_eig_vals=sort_eig_vals,
      select_callable=select_callable)


class SvdAlgorithm(enum.Enum):
  """Enum for SVD algorithm."""
  DEFAULT = "default"
  QR = "QR"
  JACOBI = "Jacobi"


@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: Literal[True],
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> tuple[Array, Array, Array]:
  ...


@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: Literal[False],
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array:
  ...


@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array | tuple[Array, Array, Array]:
  ...


# TODO: Add `max_qdwh_iterations` to the function signature for TPU SVD.
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array | tuple[Array, Array, Array]:
  """Singular value decomposition.

  Computes the singular value decomposition of an input matrix.

  Args:
    x: A batch of matrices with shape ``[..., m, n]``.
    full_matrices: Determines if full or reduced matrices are returned.
    compute_uv: If ``True``, returns the left singular vectors, the singular
      values and the adjoint of the right singular vectors. Otherwise, only
      the singular values are returned.
    subset_by_index: If ``None``, the entire matrix is returned. Otherwise,
      returns the singular values and vectors for the given range of indices.
    algorithm: The SVD algorithm to use. Must be ``None`` or a value from
      :class:`~jax.lax.linalg.SvdAlgorithm`.

  Returns:
    The singular values if ``compute_uv`` is ``False``, otherwise returns a
    triple containing the left singular vectors, the singular values, and the
    adjoint of the right singular vectors.
  """
  result = svd_p.bind(
      x,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
      algorithm=algorithm,
  )
  if compute_uv:
    s, u, v = result
    return u, s, v
  else:
    s, = result
    return s


def symmetric_product(
    a_matrix: ArrayLike,
    c_matrix: ArrayLike,
    *,
    alpha: float = 1.,
    beta: float = 0.,
    symmetrize_output: bool = False
):
  r"""Symmetric product.

  Computes the symmetric product

  .. math::
    \alpha \, A \, A^T + \beta \, C

  where :math:`A` is a rectangular matrix and :math:`C` is a symmetric matrix.

  Args:
    a_matrix: A batch of matrices with shape ``[..., m, n]``.
    c_matrix: A batch of matrices with shape ``[..., m, m]``.
    alpha: A scalar.
    beta: A scalar.
    symmetrize_output: If ``True``, the upper triangle of the output is
      replaced with its transpose.

  Returns:
    A batch of matrices with shape ``[..., m, m]`` where only the lower
    triangle is guaranteed to include the correct values on all platforms. If
    ``symmetrize_output`` is ``True``, the upper triangle is filled with the
    transpose of the lower triangle, and the whole matrix is valid.
  """
  a_matrix, c_matrix = core.standard_insert_pvary(a_matrix, c_matrix)
  result = symmetric_product_p.bind(a_matrix, c_matrix, alpha=alpha, beta=beta)
  if symmetrize_output:
    upper_half = lax.transpose(
        _tril(result, k=-1),
        (*range(result.ndim - 2), result.ndim - 1, result.ndim - 2))
    result = _tril(result, k=0) + upper_half
  return result


def triangular_solve(
    a: ArrayLike,
    b: ArrayLike,
    *,
    left_side: bool = False,
    lower: bool = False,
    transpose_a: bool = False,
    conjugate_a: bool = False,
    unit_diagonal: bool = False,
) -> Array:
  r"""Triangular solve.

  Solves either the matrix equation

  .. math::
    \mathit{op}(A) . X = B

  if ``left_side`` is ``True`` or

  .. math::
    X . \mathit{op}(A) = B

  if ``left_side`` is ``False``.

  ``A`` must be a lower or upper triangular square matrix, and where
  :math:`\mathit{op}(A)` may either transpose :math:`A` if ``transpose_a``
  is ``True`` and/or take its complex conjugate if ``conjugate_a`` is ``True``.

  Args:
    a: A batch of matrices with shape ``[..., m, m]``.
    b: A batch of matrices with shape ``[..., m, n]`` if ``left_side`` is
      ``True`` or shape ``[..., n, m]`` otherwise.
    left_side: describes which of the two matrix equations to solve; see above.
    lower: describes which triangle of ``a`` should be used. The other triangle
      is ignored.
    transpose_a: if ``True``, the value of ``a`` is transposed.
    conjugate_a: if ``True``, the complex conjugate of ``a`` is used in the
      solve. Has no effect if ``a`` is real.
    unit_diagonal: if ``True``, the diagonal of ``a`` is assumed to be unit
      (all 1s) and not accessed.

  Returns:
    A batch of matrices the same shape and dtype as ``b``.
  """
  conjugate_a = conjugate_a and dtypes.issubdtype(lax.dtype(a), np.complexfloating)
  singleton = np.ndim(b) == np.ndim(a) - 1
  if singleton:
    b = lax.expand_dims(b, (-1 if left_side else -2,))
  a, b = core.standard_insert_pvary(a, b)
  out = triangular_solve_p.bind(
      a, b, left_side=left_side, lower=lower, transpose_a=transpose_a,
      conjugate_a=conjugate_a, unit_diagonal=unit_diagonal)
  if singleton:
    out = out[..., 0] if left_side else out[..., 0, :]
  return out


def tridiagonal(
    a: ArrayLike, *, lower: bool=True
) -> tuple[Array, Array, Array, Array]:
  """Reduces a symmetric/Hermitian matrix to tridiagonal form.

  Currently implemented on CPU and GPU only.

  Args:
    a: A floating point or complex matrix or batch of matrices.
    lower: Describes which triangle of the input matrices to use.
      The other triangle is ignored and not accessed.

  Returns:
    A ``(a, d, e, taus)`` tuple. If ``lower=True``, the diagonal and first
    subdiagonal of matrix (or batch of matrices) ``a`` contain the tridiagonal
    representation, and elements below the first subdiagonal contain the
    elementary Householder reflectors, where additionally ``d`` contains the
    diagonal of the matrix and ``e`` contains the first subdiagonal. If
    ``lower=False`` the diagonal and first superdiagonal of the matrix contains
    the tridiagonal representation, and elements above the first superdiagonal
    contain the elementary Householder reflectors, where additionally ``d``
    contains the diagonal of the matrix and ``e`` contains the first
    superdiagonal. ``taus`` contains the scalar factors of the elementary
    Householder reflectors.
  """
  return tridiagonal_p.bind(lax_internal.asarray(a), lower=lower)


def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
  r"""Computes the solution of a tridiagonal linear system.

  This function computes the solution of a tridiagonal linear system:

  .. math::
    A \, X = B

  Args:

    dl: A batch of vectors with shape ``[..., m]``.
      The lower diagonal of A: ``dl[i] := A[i, i-1]`` for i in ``[0,m)``.
      Note that ``dl[0] = 0``.
    d: A batch of vectors with shape ``[..., m]``.
      The middle diagonal of A: ``d[i]  := A[i, i]`` for i in ``[0,m)``.
    du: A batch of vectors with shape ``[..., m]``.
      The upper diagonal of A: ``du[i] := A[i, i+1]`` for i in ``[0,m)``.
      Note that ``dl[m - 1] = 0``.
    b: Right hand side matrix.

  Returns:
    Solution ``X`` of tridiagonal system.
  """
  dl, d, du, b = core.standard_insert_pvary(dl, d, du, b)
  return tridiagonal_solve_p.bind(dl, d, du, b)


# Primitive registration helper functions

_platform_prefix_map = {"cpu": "cpu", "cuda": "cu", "rocm": "hip"}

def register_cpu_gpu_lowering(
    prim, lowering_rule, supported_platforms=("cpu", "cuda", "rocm")
):
  for platform in supported_platforms:
    prefix = _platform_prefix_map[platform]
    mlir.register_lowering(
        prim,
        partial(lowering_rule, target_name_prefix=prefix),
        platform=platform)

def linalg_shape_rule(multiple_results, supports_batching, ranks, result_shape,
                      name, *avals, **kwargs):
  batch_dims, dims = [], []
  for i, (rank, aval) in enumerate(zip(ranks, avals)):
    shape = aval.shape
    if len(shape) < rank:
      raise TypeError(
          f"Input {i} to {name} must have rank at least {rank}, but got "
          f"shape={shape}"
      )
    if not supports_batching and len(shape) != rank:
      raise TypeError(
          f"Input {i} to {name} must have a rank of exactly {rank}, but got "
          f"shape={shape}"
      )
    batch_dims.append(shape[:len(shape) - rank])
    dims.append(shape[len(shape) - rank:])
  if not all(len(batch_dims[0]) == len(b) for b in batch_dims):
    raise TypeError(
        f"All inputs to {name} must have the same number of batch dimensions, "
        f"but got {[len(b) for b in batch_dims]} batch dimensions for the "
        "inputs."
    )
  batch_dims = tuple(batch_dims[0])
  out = result_shape(*dims, **kwargs)
  if multiple_results:
    return tuple(batch_dims + tuple(d) for d in out)
  else:
    return batch_dims + tuple(out)

def linalg_sharding_rule(
    multiple_results, shape_rule, ranks, name, *avals, **kwargs
):
  output_shapes = shape_rule(*avals, **kwargs)
  batch_specs = []
  for i, (rank, aval) in enumerate(zip(ranks, avals)):
    spec = aval.sharding.spec
    batch_spec, rest_spec = spec[:len(spec) - rank], spec[len(spec) - rank:]
    if not all(s is None for s in rest_spec):
      raise core.ShardingTypeError(
          f"Input {i} to {name} must be unsharded on non-batch dimensions, "
          f"but got {spec}."
      )
    batch_specs.append(batch_spec)
  batch_spec = batch_specs[0]
  if any(b != batch_spec for b in batch_specs[1:]):
    raise core.ShardingTypeError(
        f"All inputs to {name} must have the same batch sharding, but got "
        f"{batch_specs}."
    )
  sharding = avals[0].sharding
  if multiple_results:
    return [
        sharding.with_spec(
            P(*(tuple(batch_spec) + (None,) * (len(s) - len(batch_spec))))
        )
        for s in output_shapes
    ]
  else:
    ndim = len(output_shapes) - len(batch_spec)
    return sharding.with_spec(P(*(tuple(batch_spec) + (None,) * ndim)))

def linalg_vma_rule(multiple_results, shape_rule, name, *avals, **kwargs):
  output_shapes = shape_rule(*avals, **kwargs)
  out_vma = core.standard_vma_rule(name, *avals)
  if multiple_results:
    return [out_vma] * len(output_shapes)
  else:
    return out_vma

def linalg_primitive(result_dtype, accepted_dtypes, ranks, result_shape, name,
                     multiple_results=False, supports_batching=True,
                     require_same=True):
  dtype_rule = partial(
      lax_internal.naryop_dtype_rule, result_dtype, accepted_dtypes, name,
      require_same=require_same)
  shape_rule = partial(
      linalg_shape_rule, multiple_results, supports_batching, ranks,
      result_shape, name)
  if supports_batching:
    sharding_rule = partial(
        linalg_sharding_rule, multiple_results, shape_rule, ranks, name)
  else:
    sharding_rule = None
  vma_rule = partial(linalg_vma_rule, multiple_results, shape_rule, name)
  prim = core.Primitive(name)
  prim.multiple_results = multiple_results
  prim.def_impl(partial(dispatch.apply_primitive, prim))
  if multiple_results:
    prim.def_abstract_eval(
        partial(lax_utils.standard_multi_result_abstract_eval, prim,
                shape_rule, dtype_rule, lax_utils._standard_weak_type_rule,
                sharding_rule, vma_rule))
  else:
    prim.def_abstract_eval(
      partial(lax_utils.standard_abstract_eval, prim, shape_rule, dtype_rule,
              lax_utils._standard_weak_type_rule, sharding_rule,
              partial(core.standard_vma_rule, name),
              None))
  if supports_batching:
    batching.primitive_batchers[prim] = partial(
        batching.expand_dims_batcher, prim)
  return prim

standard_linalg_primitive = partial(linalg_primitive, lax_internal._input_dtype)


# Primitive implementations

# Cholesky decomposition

def _cholesky_shape_rule(shape):
  if shape[0] != shape[1]:
    raise ValueError(
        f"The input to cholesky must be a square matrix. Got shape {shape}.")
  return shape


def _cholesky_jvp_rule(primals, tangents):
  x, = primals
  sigma_dot, = tangents
  L = _tril(cholesky_p.bind(x))

  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  def phi(X):
    l = _tril(X)
    return l / lax.expand_dims(
        lax_internal._const(X, 1) + lax_internal._eye(X.dtype, (X.shape[-1], X.shape[-1])),
        range(l.ndim - 2))

  tmp = triangular_solve(L, sigma_dot, left_side=False, transpose_a=True,
                         conjugate_a=True, lower=True)
  L_dot = lax.batch_matmul(L, phi(triangular_solve(
      L, tmp, left_side=True, transpose_a=False, lower=True)),
      precision=lax.Precision.HIGHEST)
  return L, L_dot


def _cholesky_lowering(ctx, x):
  del ctx  # unused
  return [hlo.cholesky(x, lower=ir.BoolAttr.get(True))]


def _cholesky_cpu_lowering(ctx, operand):
  operand_aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  batch_dims = operand_aval.shape[:-2]
  target_name = lapack.prepare_lapack_call("potrf_ffi", operand_aval.dtype)
  info_aval = ShapedArray(batch_dims, np.int32)
  rule = _linalg_ffi_lowering(target_name, avals_out=[operand_aval, info_aval],
                              operand_output_aliases={0: 0})
  result, info = rule(ctx, operand, uplo=_matrix_uplo_attr(True))
  ok = mlir.compare_hlo(info, mlir.full_like_aval(ctx, 0, info_aval), "EQ",
                        "SIGNED")
  return [_replace_not_ok_with_nan(ctx, batch_dims, ok, result, out_aval)]


cholesky_p = standard_linalg_primitive(
    (_float | _complex,), (2,), _cholesky_shape_rule, "cholesky")
ad.primitive_jvps[cholesky_p] = _cholesky_jvp_rule
mlir.register_lowering(cholesky_p, _cholesky_lowering)
mlir.register_lowering(cholesky_p, _cholesky_cpu_lowering, platform="cpu")


# Cholesky update

def _cholesky_update_shape_rule(r_shape, w_shape):
  if r_shape[0] != r_shape[1] or w_shape[0] != r_shape[1]:
    raise ValueError(
        "Rank-1 update to Cholesky decomposition takes a square matrix "
        f"and a vector of the same size as input. Got shapes {r_shape} and "
        f"{w_shape} instead")
  return r_shape


def _cholesky_update_jax_fn(R, z):
  def _drotg(x, y):
    """Get coefs for Givens rotation in a numerically stable way."""
    def _drotg_nonzero(x, y):
      abs_x = abs(x)
      abs_y = abs(y)
      denominator = lax.select(abs_x > abs_y, abs_x, abs_y)
      x /= denominator
      y /= denominator
      rh = 1 / lax.sqrt(x ** 2 + y ** 2)
      return x * rh, -y * rh
    one_and_zero = (
        np.array(1., dtype=x.dtype),
        np.array(0., dtype=x.dtype),
    )
    return lax.cond(y == 0, lambda x, y: one_and_zero, _drotg_nonzero, x, y)

  def _drot(
      first_vector: Array, second_vector: Array,
      c_coef: float, s_coef: float) -> tuple[Array, Array]:
    return (
        c_coef * first_vector - s_coef * second_vector,
        c_coef * second_vector + s_coef * first_vector)
  n = z.shape[0]
  for k in range(n):
    c, s = _drotg(R[k, k], z[k])
    row_k, z = _drot(R[k, :], z, c, s)
    R = R.at[k, :].set(row_k)
  return R


def _cholesky_update_gpu_lowering_rule(target_name_prefix, ctx, r_matrix,
                                       w_vector):
  rule = ffi.ffi_lowering(f"{target_name_prefix}_cholesky_update_ffi",
                          operand_output_aliases={0: 0, 1: 1})
  sub_ctx = ctx.replace(avals_out=ctx.avals_in)
  return rule(sub_ctx, r_matrix, w_vector)[:1]


cholesky_update_p = standard_linalg_primitive(
    (_float, _float), (2, 1), _cholesky_update_shape_rule, "cholesky_update",
    supports_batching=False)
mlir.register_lowering(
    cholesky_update_p, partial(_cholesky_update_gpu_lowering_rule, "cu"),
    platform="cuda")
mlir.register_lowering(
    cholesky_update_p,
    mlir.lower_fun(_cholesky_update_jax_fn, multiple_results=False))

# General eigendecomposition

def _eig_dtype_rule(
    a_dtype, *, compute_left_eigenvectors, compute_right_eigenvectors, **_
):
  dtype = dtypes.to_complex_dtype(dtypes.canonicalize_dtype(a_dtype))
  return (dtype,) * (1 + compute_left_eigenvectors + compute_right_eigenvectors)

def _eig_shape_rule(
    shape, *, compute_left_eigenvectors, compute_right_eigenvectors, **_
):
  if shape[0] != shape[1]:
    raise ValueError(
        f"The input to eig must be a square matrix. Got shape {shape}.")
  count = compute_left_eigenvectors + compute_right_eigenvectors
  return (shape[:-1],) + (shape,) * count

def _eig_compute_attr(compute):
  return _enum_attr(
      lapack.eig.ComputationMode.kComputeEigenvectors if compute
      else lapack.eig.ComputationMode.kNoEigenvectors
  )

def _eig_cpu_lowering(ctx, operand, *, compute_left_eigenvectors,
                      compute_right_eigenvectors, use_magma):
  del use_magma  # unused
  operand_aval, = ctx.avals_in
  out_aval = ctx.avals_out[0]
  batch_dims = operand_aval.shape[:-2]
  real = operand_aval.dtype == np.float32 or operand_aval.dtype == np.float64
  eigvals_aval = ShapedArray(operand_aval.shape[:-1], operand_aval.dtype)
  eigvecs_aval = ShapedArray(operand_aval.shape,
                              dtypes.to_complex_dtype(operand_aval.dtype))
  info_aval = ShapedArray(batch_dims, np.int32)
  avals_out = [eigvals_aval, eigvecs_aval, eigvecs_aval, info_aval]
  if real:
    avals_out = [eigvals_aval, *avals_out]
  target_name = lapack.prepare_lapack_call("geev_ffi", operand_aval.dtype)
  rule = _linalg_ffi_lowering(target_name, avals_out=avals_out)
  *w, vl, vr, info = rule(ctx, operand,
                          compute_left=_eig_compute_attr(compute_left_eigenvectors),
                          compute_right=_eig_compute_attr(compute_right_eigenvectors))
  w = hlo.complex(w[0], w[1]) if real else w[0]

  ok = mlir.compare_hlo(
      info, mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")
  w = _replace_not_ok_with_nan(ctx, batch_dims, ok, w, out_aval)
  output = [w]
  if compute_left_eigenvectors:
    aval = ctx.avals_out[len(output)]
    vl = _replace_not_ok_with_nan(ctx, batch_dims, ok, vl, aval)
    output.append(vl)
  if compute_right_eigenvectors:
    aval = ctx.avals_out[len(output)]
    vr = _replace_not_ok_with_nan(ctx, batch_dims, ok, vr, aval)
    output.append(vr)
  return output

def _eig_gpu_lowering(ctx, operand, *,
                      compute_left_eigenvectors, compute_right_eigenvectors,
                      use_magma, target_name_prefix):
  operand_aval, = ctx.avals_in
  batch_dims = operand_aval.shape[:-2]
  n, m = operand_aval.shape[-2:]
  assert n == m

  gpu_solver.initialize_hybrid_kernels()
  dtype = operand_aval.dtype
  is_real = dtype == np.float32 or dtype == np.float64
  if is_real:
    target_name = f"{target_name_prefix}hybrid_eig_real"
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
  else:
    target_name = f"{target_name_prefix}hybrid_eig_comp"
    assert dtype == np.complex64 or dtype == np.complex128
    complex_dtype = dtype

  avals_out = [
      ShapedArray(batch_dims + (n,), dtype),
      ShapedArray(batch_dims + (n, n), complex_dtype),
      ShapedArray(batch_dims + (n, n), complex_dtype),
      ShapedArray(batch_dims, np.int32),
  ]
  if is_real:
    avals_out = [ShapedArray(batch_dims + (n,), dtype)] + avals_out

  magma = config.gpu_use_magma.value
  if use_magma is not None:
    magma = "on" if use_magma else "off"

  rule = _linalg_ffi_lowering(target_name, avals_out=avals_out)
  *w, vl, vr, info = rule(ctx, operand, magma=magma,
                          left=compute_left_eigenvectors,
                          right=compute_right_eigenvectors)
  if is_real:
    assert len(w) == 2
    w = hlo.complex(*w)
  else:
    assert len(w) == 1
    w = w[0]
  zeros = mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.int32))
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  w_aval = ShapedArray(batch_dims + (n,), complex_dtype)
  w = _replace_not_ok_with_nan(ctx, batch_dims, ok, w, w_aval)
  output = [w]
  if compute_left_eigenvectors:
    vl_aval = ShapedArray(batch_dims + (n, n), complex_dtype)
    vl = _replace_not_ok_with_nan(ctx, batch_dims, ok, vl, vl_aval)
    output.append(vl)
  if compute_right_eigenvectors:
    vr_aval = ShapedArray(batch_dims + (n, n), complex_dtype)
    vr = _replace_not_ok_with_nan(ctx, batch_dims, ok, vr, vr_aval)
    output.append(vr)
  return output

def eig_jvp_rule(primals, tangents, *, compute_left_eigenvectors,
                 compute_right_eigenvectors, use_magma):
  del use_magma  # unused
  if compute_left_eigenvectors or compute_right_eigenvectors:
    raise NotImplementedError(
        'The derivatives of eigenvectors are not implemented, only '
        'eigenvalues. See '
        'https://github.com/jax-ml/jax/issues/2748 for discussion.')
  # Formula for derivative of eigenvalues w.r.t. a is eqn 4.60 in
  # https://arxiv.org/abs/1701.00392
  a, = primals
  da, = tangents
  l, v = eig(a, compute_left_eigenvectors=False)
  return [l], [(_solve(v, da.astype(v.dtype)) * _T(v)).sum(-1)]

eig_p = linalg_primitive(
    _eig_dtype_rule, (_float | _complex,), (2,), _eig_shape_rule, "eig",
    multiple_results=True)
ad.primitive_jvps[eig_p] = eig_jvp_rule
mlir.register_lowering(eig_p, _eig_cpu_lowering, platform="cpu")
register_cpu_gpu_lowering(eig_p, _eig_gpu_lowering, ("cuda", "rocm"))


# Symmetric/Hermitian eigendecomposition

def eigh_jacobi(x: ArrayLike, *, lower: bool = True,
                sort_eigenvalues: bool = True) -> tuple[Array, Array]:
  """Helper Jacobi eigendecomposition implemented by XLA.

  Used as a subroutine of QDWH-eig on TPU.
  """
  return eigh_jacobi_p.bind(x, lower=lower, sort_eigenvalues=sort_eigenvalues)

def _eigh_jacobi_shape_rule(shape, **_):
  if shape[0] != shape[-1]:
    raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n], "
        f"got shape {shape}"
    )
  n = shape[0]
  return (n,), (n, n)

def _eigh_jacobi_dtype_rule(dtype, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  return lax_internal._complex_basetype(dtype), dtype

def _eigh_jacobi_lowering_rule(ctx, operand, lower, sort_eigenvalues):
  operand_aval, = ctx.avals_in
  if operand_aval.shape[-1] == 0:
    reshape_aval = operand_aval.update(shape=operand_aval.shape[:-1])
    return [
        hlo.real(mlir.reshape(ctx, operand, reshape_aval)),
        operand,
    ]

  eigvals_type = mlir.aval_to_ir_type(ctx.avals_out[0])
  eigvecs_type = mlir.aval_to_ir_type(ctx.avals_out[1])
  result_types = [eigvecs_type, eigvals_type]

  backend_config = f"{int(lower)},{int(sort_eigenvalues)},100,1e-6"

  if any(not is_constant_shape(aval_out.shape)
         for aval_out in ctx.avals_out):
    result_shapes = [
        mlir.eval_dynamic_shape_as_tensor(ctx, aval_out.shape)
        # The custom call returns the results swapped
        for aval_out in list(reversed(ctx.avals_out))
    ]
  else:
    result_shapes = None
  op = mlir.custom_call(
      "Eigh",
      result_types=result_types,
      operands=[operand],
      backend_config=backend_config,
      api_version=1,
      result_shapes=result_shapes,
  )
  return op.results[1], op.results[0]

eigh_jacobi_p = linalg_primitive(
    _eigh_jacobi_dtype_rule, (_float | _complex,), (2,),
    _eigh_jacobi_shape_rule, "eigh_jacobi", multiple_results=True)
mlir.register_lowering(eigh_jacobi_p, _eigh_jacobi_lowering_rule)


def _eigh_shape_rule(shape, *, subset_by_index, **_):
  if shape[0] != shape[-1]:
    raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n], "
        f"got shape {shape}"
    )
  n = shape[0]
  d = (n if subset_by_index is None else
       subset_by_index[1] - subset_by_index[0])
  return (n, d), (d,)

def _eigh_dtype_rule(dtype, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  return dtype, lax_internal._complex_basetype(dtype)

def _eigh_cpu_gpu_lowering(
    ctx, operand, *, lower, sort_eigenvalues, subset_by_index,
    target_name_prefix: str
):
  del sort_eigenvalues  # The CPU/GPU implementations always sort.
  operand_aval, = ctx.avals_in
  v_aval, w_aval = ctx.avals_out
  n = operand_aval.shape[-1]
  if not (subset_by_index is None or subset_by_index == (0, n)):
    raise NotImplementedError("subset_by_index not supported on CPU and GPU")
  batch_dims = operand_aval.shape[:-2]
  if target_name_prefix == "cpu":
    dtype = operand_aval.dtype
    prefix = "he" if dtypes.issubdtype(dtype, np.complexfloating) else "sy"
    target_name = lapack.prepare_lapack_call(f"{prefix}evd_ffi",
                                             operand_aval.dtype)
    kwargs = {
      "mode": np.uint8(ord("V")),
      "uplo": np.uint8(ord("L" if lower else "U")),
    }
  else:
    target_name = f"{target_name_prefix}solver_syevd_ffi"
    kwargs = {"lower": lower, "algorithm": np.uint8(0)}

  info_aval = ShapedArray(batch_dims, np.int32)
  avals_out = [v_aval, w_aval, info_aval]
  rule = _linalg_ffi_lowering(target_name, avals_out=avals_out,
                              operand_output_aliases={0: 0})
  v, w, info = rule(ctx, operand, **kwargs)

  zeros = mlir.full_like_aval(ctx, 0, info_aval)
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  v = _replace_not_ok_with_nan(ctx, batch_dims, ok, v, v_aval)
  w = _replace_not_ok_with_nan(ctx, batch_dims, ok, w, w_aval)
  return [v, w]


def _eigh_tpu_impl(x, *, lower, sort_eigenvalues, subset_by_index):
  *_, m, n = x.shape
  assert m == n, (m, n)

  termination_size = 256
  if not is_constant_dim(m):
    # TODO: maybe we can relax the check below for shape polymorphism?
    raise NotImplementedError(
        "Shape polymorphism for native lowering for eigh is implemented "
        f"only for the batch dimensions: {x.shape}")
  if m <= termination_size and (
      subset_by_index is None or subset_by_index == (0, n)
  ):
    eig_vals, eig_vecs = eigh_jacobi(x, lower=lower,
                                     sort_eigenvalues=sort_eigenvalues)
    return eig_vecs, eig_vals

  def eigh_qdwh(x):
    if len(x.shape) > 2:
      return control_flow.map(eigh_qdwh, x)

    # We should only look at elements from the lower/upper triangle. Reflects
    # that triangle into the other triangle to form a Hermitian matrix.
    if lower:
      mask = lax_internal._tri(bool, (n, n), 0)
    else:
      mask = lax.bitwise_not(lax_internal._tri(bool, (n, n), -1))
    if dtypes.issubdtype(x.dtype, np.complexfloating):
      re = lax.select(mask, lax.real(x), _T(lax.real(x)))
      if lower:
        im_mask = lax_internal._tri(bool, (n, n), -1)
      else:
        im_mask = lax.bitwise_not(lax_internal._tri(bool, (n, n), 0))
      im = lax.imag(x)
      im = lax.select(im_mask, im, lax.full_like(im, 0))
      im = lax.select(mask, im, -_T(im))
      x = lax.complex(re, im)
    else:
      x = lax.select(mask, x, _T(x))

    return lax_eigh.eigh(
        x,
        sort_eigenvalues=sort_eigenvalues,
        termination_size=termination_size,
        subset_by_index=subset_by_index,
    )

  eig_vals, eig_vecs = eigh_qdwh(x)
  return eig_vecs, eig_vals


def _eigh_jvp_rule(
    primals, tangents, *, lower, sort_eigenvalues, subset_by_index
):
  (a,) = primals
  n = a.shape[-1]
  if not (subset_by_index is None or subset_by_index == (0, n)):
    raise NotImplementedError(
        "Derivatives not defined for partial eigen decomposition."
    )
  # Derivative for eigh in the simplest case of distinct eigenvalues.
  # This is classic nondegenerate perurbation theory, but also see
  # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  # The general solution treating the case of degenerate eigenvalues is
  # considerably more complicated. Ambitious readers may refer to the general
  # methods below or refer to degenerate perturbation theory in physics.
  # https://www.win.tue.nl/analysis/reports/rana06-33.pdf and
  # https://people.orie.cornell.edu/aslewis/publications/99-clarke.pdf
  a_dot, = tangents

  v, w_real = eigh_p.bind(
      symmetrize(a),
      lower=lower,
      sort_eigenvalues=sort_eigenvalues,
      subset_by_index=subset_by_index,
  )

  # for complex numbers we need eigenvalues to be full dtype of v, a:
  w = w_real.astype(a.dtype)
  eye_n = lax_internal._eye(a.dtype, (n, n))
  # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
  with config.numpy_rank_promotion("allow"):
    Fmat = lax.integer_pow(eye_n + w[..., np.newaxis, :] - w[..., np.newaxis], -1) - eye_n
  # eigh impl doesn't support batch dims, but future-proof the grad.
  dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)
  vdag_adot_v = dot(dot(_H(v), a_dot), v)
  dv = dot(v, Fmat * vdag_adot_v)
  dw = _extract_diagonal(vdag_adot_v.real)
  return (v, w_real), (dv, dw)


eigh_p = linalg_primitive(
    _eigh_dtype_rule, (_float | _complex,), (2,), _eigh_shape_rule, "eigh",
    multiple_results=True)
ad.primitive_jvps[eigh_p] = _eigh_jvp_rule
mlir.register_lowering(
    eigh_p, mlir.lower_fun(_eigh_tpu_impl, multiple_results=True),
    platform='tpu')
register_cpu_gpu_lowering(eigh_p, _eigh_cpu_gpu_lowering)


# Hessenberg reduction

def _hessenberg_shape_rule(shape, **_):
  if shape[0] != shape[-1]:
    raise ValueError(
        "Argument to Hessenberg reduction must have shape [..., n, n], "
        f"got shape {shape}"
    )
  return shape, shape[:-2] + (shape[-1] - 1,)


def _hessenberg_dtype_rule(dtype, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  return dtype, dtype


def _hessenberg_cpu_lowering(ctx, a):
  a_aval, = ctx.avals_in
  batch_dims = a_aval.shape[:-2]
  n = a_aval.shape[-1]
  if not core.is_constant_dim(n):
    raise ValueError("hessenberg requires the last dimension of a to be "
                     f"constant, got a.shape of {a.shape}.")
  target_name = lapack.prepare_lapack_call("gehrd_ffi", a_aval.dtype)
  avals_out = [*ctx.avals_out, ShapedArray(batch_dims, np.int32)]
  rule = _linalg_ffi_lowering(target_name, avals_out=avals_out,
                              operand_output_aliases={0: 0})
  a, taus, info = rule(ctx, a, low=np.int32(1), high=np.int32(n))
  ok = mlir.compare_hlo(
      info, mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")
  return [
      _replace_not_ok_with_nan(ctx, batch_dims, ok, a, ctx.avals_out[0]),
      _replace_not_ok_with_nan(ctx, batch_dims, ok, taus, ctx.avals_out[1]),
  ]


hessenberg_p = linalg_primitive(
    _hessenberg_dtype_rule, (_float | _complex,), (2,), _hessenberg_shape_rule,
    "hessenberg", multiple_results=True)
mlir.register_lowering(hessenberg_p, _hessenberg_cpu_lowering, platform="cpu")


# Householder product

def _householder_product_shape_rule(a_shape, taus_shape, **_):
  m, n = a_shape
  if m < n:
    raise ValueError(
        "The first argument to householder_product must have at least as many "
        f"rows as columns, got shape {a_shape}")
  k = taus_shape[0]
  if k > core.min_dim(m, n):
    raise ValueError(
        "The second argument to householder_product must not have more rows "
        "than the minimum of the first argument's rows and columns.")
  return a_shape


def _householder_product_lowering(ctx, a, taus):
  aval_out, = ctx.avals_out
  if not is_constant_shape(aval_out.shape):
    result_shapes = [
        mlir.eval_dynamic_shape_as_tensor(ctx, aval_out.shape)]
  else:
    result_shapes = None
  op = mlir.custom_call(
      "ProductOfElementaryHouseholderReflectors",
      result_types=[mlir.aval_to_ir_type(aval_out)],
      operands=[a, taus],
      api_version=1,
      result_shapes=result_shapes)
  return [op.result]


def _householder_product_cpu_gpu_lowering(ctx, a, taus, *,
                                          target_name_prefix: str):
  a_aval, _ = ctx.avals_in
  if target_name_prefix == "cpu":
    dtype = a_aval.dtype
    prefix = "un" if dtypes.issubdtype(dtype, np.complexfloating) else "or"
    target_name = lapack.prepare_lapack_call(f"{prefix}gqr_ffi", dtype)
  else:
    target_name = f"{target_name_prefix}solver_orgqr_ffi"
  rule = _linalg_ffi_lowering(target_name, operand_output_aliases={0: 0})
  return rule(ctx, a, taus)

householder_product_p = standard_linalg_primitive(
    (_float | _complex, _float | _complex), (2, 1),
    _householder_product_shape_rule, "householder_product")
mlir.register_lowering(householder_product_p, _householder_product_lowering)
register_cpu_gpu_lowering(
    householder_product_p, _householder_product_cpu_gpu_lowering)


# LU decomposition

# Computes a pivoted LU decomposition such that
# PA = LU
# In the style of LAPACK, LU are stored in the same matrix.

def _lu_unblocked(a):
  """Unblocked LU decomposition, as a rolled loop."""
  m, n = a.shape
  def body(k, state):
    pivot, perm, a = state
    m_idx = lax.iota('int32', m)
    n_idx = lax.iota('int32', n)

    if dtypes.issubdtype(a.dtype, np.complexfloating):
      t = a[:, k]
      magnitude = abs(t.real) + abs(t.imag)
    else:
      magnitude = abs(a[:, k])
    i = lax.argmax(lax.select(m_idx >= k, magnitude, lax.full_like(magnitude, -np.inf)),
                   axis=0, index_dtype=pivot.dtype)
    pivot = pivot.at[k].set(i)
    a = a.at[[k, i],].set(a[[i, k],])
    perm = perm.at[[i, k],].set(perm[[k, i],])

    # a[k+1:, k] /= a[k, k], adapted for loop-invariant shapes
    x = a[k, k]
    a = a.at[:, k].set(lax.select((m_idx > k) & (x != 0), a[:, k] / x, a[:, k]))

    # a[k+1:, k+1:] -= jnp.outer(a[k+1:, k], a[k, k+1:])
    a_outer = a[:, k, None] * a[k, None]
    a = a - lax.select((m_idx[:, None] > k) & (n_idx[None, :] > k),
                       a_outer, lax_internal._zeros(a_outer))
    return pivot, perm, a

  pivot = lax.full((min(m, n),), 0, dtype=np.int32)
  perm = lax.iota('int32', m)
  if m == 0 and n == 0:
    # If the array is empty, the loop body never executes but tracing it to a
    # jaxpr fails because the indexing cannot succeed.
    return (pivot, perm, a)
  return lax.fori_loop(0, min(m, n), body, (pivot, perm, a))


def _lu_blocked(a, block_size=128):
  """Blocked LU decomposition, as an unrolled loop."""
  m, n = a.shape
  r = min(m, n)
  pivot = lax.full((r,), 0, dtype=np.int32)
  perm = lax.iota('int32', m)
  for k in range(0, r, block_size):
    b = min(r - k, block_size)
    block_pivot, block_perm, lu_block = _lu_unblocked(a[k:, k:k+b])

    pivot = pivot.at[k:k+b].set(block_pivot + k)
    perm = perm.at[k:].set(perm[block_perm + k])
    a = a.at[k:, :].set(a[block_perm + k, :])
    a = a.at[k:, k:k+b].set(lu_block)

    if k + b < n:
      a = a.at[k:k+b, k+b:].set(
        triangular_solve(a[k:k+b, k:k+b], a[k:k+b, k+b:], left_side=True,
                         lower=True, unit_diagonal=True))
      a = a.at[k+b:, k+b:].add(-lax.dot(a[k+b:, k:k+b], a[k:k+b, k+b:],
                                        precision=lax.Precision.HIGHEST))
  return a, pivot, perm

def _lu_python(x):
  """Default LU decomposition in Python, where no better version exists."""
  batch_dims = x.shape[:-2]
  fn = _lu_blocked
  for _ in range(len(batch_dims)):
    fn = api.vmap(fn)

  return fn(x)


def _lu_shape_rule(shape):
  m, n = shape
  return shape, (core.min_dim(m, n),), (m,)


def _lu_dtype_rule(dtype, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  return dtype, dtypes.dtype(np.int32), dtypes.dtype(np.int32)


def _lu_jvp_inner(lu, a_dot, permutation):
  # Differentiation of Matrix Functionals Using Triangular Factorization
  # F. R. De Hoog, R. S. Anderssen, and M. A. Lukas
  #
  #     LU = A
  # ==> L'U + LU' = A'
  # ==> inv(L) . L' + U' . inv(U) = inv(L) A' inv(U)
  # ==> L' = L . tril(inv(L) . A' . inv(U), -1)
  #     U' = triu(inv(L) . A' . inv(U)) . U

  a_shape = np.shape(a_dot)
  assert len(a_shape) == 2
  m, n = a_shape
  dtype = lax.dtype(a_dot)
  k = min(m, n)

  l_padding = [(0, 0, 0)] * 2
  l_padding[-1] = (0, m - k, 0)
  zero = lax_internal._const(lu, 0)
  l = lax.pad(_tril(lu[:, :k], -1), zero, l_padding)
  l = l + lax_internal._eye(dtype, (m, m))
  u_eye = lax.pad(lax_internal._eye(dtype, (n - k, n - k)), zero,
                  ((k, 0, 0), (k, 0, 0)))
  u_padding = [(0, 0, 0)] * 2
  u_padding[-2] = (0, n - k, 0)
  u = lax.pad(_triu(lu[:k, :]), zero, u_padding) + u_eye

  la = triangular_solve(l, a_dot[permutation], left_side=True,
                        transpose_a=False, lower=True, unit_diagonal=True)
  lau = triangular_solve(u, la, left_side=False, transpose_a=False,
                         lower=False)
  with config.default_matmul_precision("highest"):
    l_dot = l @ _tril(lau, -1)
    u_dot = _triu(lau) @ u
  return l_dot + u_dot


def _lu_jvp_rule(primals, tangents):
  a, = primals
  a_dot, = tangents
  lu, pivots, permutation = lu_p.bind(a)

  lu_dot_fun = _lu_jvp_inner
  for _ in np.shape(a)[:-2]:
    lu_dot_fun = api.vmap(lu_dot_fun)
  lu_dot = lu_dot_fun(lu, a_dot, permutation)

  return (lu, pivots, permutation), (lu_dot, ad_util.Zero.from_primal_value(pivots),
                                     ad_util.Zero.from_primal_value(permutation))


def _lu_cpu_gpu_lowering(ctx, operand, *, target_name_prefix: str):
  operand_aval, = ctx.avals_in
  out_aval, pivot_aval, perm_aval = ctx.avals_out
  batch_dims = operand_aval.shape[:-2]
  info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
  m = operand_aval.shape[-2]

  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("getrf_ffi", operand_aval.dtype)
  else:
    target_name = f"{target_name_prefix}solver_getrf_ffi"
  rule = _linalg_ffi_lowering(target_name,
                              avals_out=[out_aval, pivot_aval, info_aval],
                              operand_output_aliases={0: 0})
  lu, pivot, info = rule(ctx, operand)

  # Subtract 1 from the pivot to get 0-based indices.
  pivot = hlo.subtract(pivot, mlir.full_like_aval(ctx, 1, pivot_aval))
  ok = mlir.compare_hlo(info, mlir.full_like_aval(ctx, 0, info_aval),
      "GE", "SIGNED")
  lu = _replace_not_ok_with_nan(ctx, batch_dims, ok, lu, out_aval)
  sub_ctx = ctx.replace(primitive=None, avals_in=[pivot_aval],
                        avals_out=[perm_aval])
  perm_fn = mlir.lower_fun(lambda x: lu_pivots_to_permutation(x, m),
                           multiple_results=False)
  perm, = perm_fn(sub_ctx, pivot)
  return [lu, pivot, perm]


def _lu_tpu_lowering_rule(ctx, operand):
  result_types = [
    mlir.aval_to_ir_type(ctx.avals_out[0]),
    mlir.aval_to_ir_type(ctx.avals_out[1]),
    mlir.aval_to_ir_type(ctx.avals_out[2])]
  if any(not is_constant_shape(a.shape) for a in ctx.avals_out):
    result_shapes = [
      mlir.eval_dynamic_shape_as_tensor(ctx, a.shape)
      for a in ctx.avals_out]
  else:
    result_shapes = None
  op = mlir.custom_call(
    "LuDecomposition",
    result_types=result_types,
    operands=[operand],
    result_shapes=result_shapes)
  return op.results


lu_p = linalg_primitive(
    _lu_dtype_rule, (_float | _complex,), (2,), _lu_shape_rule, "lu",
    multiple_results=True)
ad.primitive_jvps[lu_p] = _lu_jvp_rule
mlir.register_lowering(lu_p, mlir.lower_fun(_lu_python, multiple_results=True))
mlir.register_lowering(lu_p, _lu_tpu_lowering_rule, platform='tpu')
register_cpu_gpu_lowering(lu_p, _lu_cpu_gpu_lowering)


def lu_solve(lu: ArrayLike, permutation: ArrayLike, b: ArrayLike,
             trans: int = 0) -> Array:
  """LU solve with broadcasting."""
  return _lu_solve(lu, permutation, b, trans)


def _lu_solve_core(lu: Array, permutation: Array, b: Array, trans: int) -> Array:
  m = lu.shape[0]
  x = lax.reshape(b, (m, math.prod(b.shape[1:])))
  if trans == 0:
    x = x[permutation, :]
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True)
    x = triangular_solve(lu, x, left_side=True, lower=False)
  elif trans == 1 or trans == 2:
    conj = trans == 2
    x = triangular_solve(lu, x, left_side=True, lower=False, transpose_a=True,
                         conjugate_a=conj)
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True,
                         transpose_a=True, conjugate_a=conj)
    _, ind = lax.sort_key_val(permutation, lax.iota('int32', permutation.shape[0]))
    x = x[ind, :]
  else:
    raise ValueError(f"'trans' value must be 0, 1, or 2, got {trans}")
  return lax.reshape(x, b.shape)


@partial(api.jit, static_argnums=(3,))
def _lu_solve(lu: Array, permutation: Array, b: Array, trans: int) -> Array:
  if len(lu.shape) < 2 or lu.shape[-1] != lu.shape[-2]:
    raise ValueError("last two dimensions of LU decomposition must be equal, "
                     "got shape {}".format(lu.shape))
  if len(b.shape) < 1:
    raise ValueError("b matrix must have rank >= 1, got shape {}"
                     .format(b.shape))
  # Broadcasting follows NumPy's convention for linalg.solve: the RHS is
  # treated as a (batched) vector if the number of dimensions differ by 1.
  # Otherwise, broadcasting rules apply.
  rhs_vector = lu.ndim == b.ndim + 1
  if rhs_vector:
    if b.shape[-1] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b have the same "
                       "number of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and b array (shape {}) must match"
                       .format(lu.shape, b.shape))
    b = b[..., np.newaxis]
  else:
    if b.shape[-2] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b different "
                       "numbers of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and second to last axis of b array "
                       "(shape {}) must match"
                       .format(lu.shape, b.shape))

  batch_shape = lax.broadcast_shapes(lu.shape[:-2], permutation.shape[:-1], b.shape[:-2])
  lu = _broadcast_to(lu, (*batch_shape, *lu.shape[-2:]))
  permutation = _broadcast_to(permutation, (*batch_shape, permutation.shape[-1]))
  b = _broadcast_to(b, (*batch_shape, *b.shape[-2:]))
  fn = _lu_solve_core
  for _ in batch_shape:
    fn = api.vmap(fn, in_axes=(0, 0, 0, None))
  x = fn(lu, permutation, b, trans)
  return x[..., 0] if rhs_vector else x

# Support operation for LU decomposition: Transformation of the pivots returned
# by LU decomposition into permutations.

# Define this outside lu_pivots_to_permutation to ensure fori_loop cache hits
def _lu_pivots_body_fn_inner(i, permutation, swaps):
  j = swaps[i]
  x = permutation[i]
  y = permutation[j]
  permutation = permutation.at[i].set(y)
  return permutation.at[j].set(x)


def _lu_pivots_body_fn(i, permutation_and_swaps):
  permutation, swaps = permutation_and_swaps
  batch_dims = swaps.shape[:-1]
  fn = _lu_pivots_body_fn_inner
  for _ in range(len(batch_dims)):
    fn = api.vmap(fn, in_axes=(None, 0, 0), out_axes=0)
  return fn(i, permutation, swaps), swaps


def _generic_lu_pivots_to_permutation(swaps, permutation_size):
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `swaps` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    swaps: an array of shape (..., k) of row swaps to perform
    permutation_size: the size of the output permutation. Should be >= k.
  Returns:
    An int32 array of shape (..., m).
  """
  assert len(swaps.shape) >= 1
  batch_dims = swaps.shape[:-1]
  k = swaps.shape[-1]
  m = permutation_size

  permutation = lax.broadcasted_iota(np.int32, batch_dims + (m,),
                                     len(batch_dims))
  if m == 0 or k == 0:
    return permutation
  upper = np.array(k, np.int32) if is_constant_dim(k) else k
  permutation, swaps = core.standard_insert_pvary(permutation, swaps)
  result, _ = lax.fori_loop(np.array(0, np.int32), upper, _lu_pivots_body_fn,
                            (permutation, swaps))
  return result


def _lu_pivots_to_permutation_shape_rule(shape, *, permutation_size):
  pivots_size, = shape
  if not permutation_size >= pivots_size:
    raise ValueError(
        f"Output permutation size {permutation_size} has to exceed the "
        f"trailing dimension of the pivots. Got pivots size {pivots_size}")
  return (permutation_size,)


def _lu_pivots_to_permutation_gpu_lowering(ctx, pivots, *,
                                           permutation_size,
                                           target_name_prefix):
  del permutation_size  # unused
  rule = _linalg_ffi_lowering(f"{target_name_prefix}_lu_pivots_to_permutation",
                              num_non_batch_dims=1, column_major=False)
  return rule(ctx, pivots)


lu_pivots_to_permutation_p = standard_linalg_primitive(
    ({np.int32},), (1,), _lu_pivots_to_permutation_shape_rule,
    "lu_pivots_to_permutation")
mlir.register_lowering(
    lu_pivots_to_permutation_p,
    mlir.lower_fun(_generic_lu_pivots_to_permutation, multiple_results=False))
register_cpu_gpu_lowering(
    lu_pivots_to_permutation_p, _lu_pivots_to_permutation_gpu_lowering,
    ("cuda", "rocm"))


# QR decomposition

# QR decomposition is implemented as a composition of two lower-level primitives
# geqrf and orgqr. The names, while cryptic Fortran alphabet soup, are LAPACK's
# names for the primitives, and we stick with them for consistency.

def geqrf(a: ArrayLike) -> tuple[Array, Array]:
  """Computes the QR decomposition of a matrix.

  Args:
    a: an ``[..., m, n]`` batch of matrices, with floating-point or complex type.
  Returns:
    An ``(a, taus)`` pair where ``r`` is in the upper triangle of ``a``,
    ``q`` is represented in the lower triangle of ``a`` and in ``taus`` as
    elementary Householder reflectors.
  """
  a_out, taus = geqrf_p.bind(a)
  return a_out, taus

def _geqrf_shape_rule(shape):
  m, n = shape
  return shape, (core.min_dim(m, n),)

def _geqrf_dtype_rule(dtype):
  dtype = dtypes.canonicalize_dtype(dtype)
  return dtype, dtype

def _geqrf_lowering_rule(ctx, operand):
  ts_type = mlir.aval_to_ir_type(ctx.avals_out[0])
  r_type = mlir.aval_to_ir_type(ctx.avals_out[1])
  result_types = [ts_type, r_type]
  if any(not is_constant_shape(aval_out.shape)
         for aval_out in ctx.avals_out):
    result_shapes = [
        mlir.eval_dynamic_shape_as_tensor(ctx, aval_out.shape)
        for aval_out in ctx.avals_out
    ]
  else:
    result_shapes = None
  op = mlir.custom_call(
      "Qr",
      result_types=result_types,
      operands=[operand],
      api_version=1,
      result_shapes=result_shapes
  )
  return op.results

def _geqrf_cpu_gpu_lowering(ctx, a, *, target_name_prefix: str):
  operand_aval, = ctx.avals_in
  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("geqrf_ffi", operand_aval.dtype)
  else:
    target_name = f"{target_name_prefix}solver_geqrf_ffi"
  rule = _linalg_ffi_lowering(target_name, operand_output_aliases={0: 0})
  return rule(ctx, a)

geqrf_p = linalg_primitive(
    _geqrf_dtype_rule, (_float | _complex,), (2,), _geqrf_shape_rule, "geqrf",
    multiple_results=True)
mlir.register_lowering(geqrf_p, _geqrf_lowering_rule)
register_cpu_gpu_lowering(geqrf_p, _geqrf_cpu_gpu_lowering)


def geqp3(a: ArrayLike, jpvt: ArrayLike, *,
          use_magma: bool | None = None) -> tuple[Array, Array, Array]:
  """Computes the column-pivoted QR decomposition of a matrix.

  Args:
    a: a ``[..., m, n]`` batch of matrices, with floating-point or complex type.
    jpvt: a ``[..., n]`` batch of column-pivot index vectors with integer type,
    use_magma: Locally override the ``jax_use_magma`` flag. If ``True``, the
      `geqp3` is computed using MAGMA. If ``False``, the computation is done using
      LAPACK on to the host CPU. If ``None`` (default), the behavior is controlled
      by the ``jax_use_magma`` flag. This argument is only used on GPU.
  Returns:
    A ``(a, jpvt, taus)`` triple, where ``r`` is in the upper triangle of ``a``,
    ``q`` is represented in the lower triangle of ``a`` and in ``taus`` as
    elementary Householder reflectors, and ``jpvt`` is the column-pivot indices
    such that ``a[:, jpvt] = q @ r``.
  """
  a, jpvt = core.standard_insert_pvary(a, jpvt)
  a_out, jpvt_out, taus = geqp3_p.bind(a, jpvt, use_magma=use_magma)
  return a_out, jpvt_out, taus

def _geqp3_shape_rule(a_shape, jpvt_shape, **_):
  m, n = a_shape
  return a_shape, jpvt_shape, (core.min_dim(m, n),)

def _geqp3_dtype_rule(dtype, jpvt_dtype, *_, **__):
  dtype = dtypes.canonicalize_dtype(dtype)
  jpvt_dtype = dtypes.canonicalize_dtype(jpvt_dtype)
  return dtype, jpvt_dtype, dtype

def _geqp3_cpu_gpu_lowering(ctx, a, jpvt, *, use_magma, target_name_prefix):
  a_aval, _ = ctx.avals_in
  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("geqp3_ffi", a_aval.dtype)
    params = {}
  else:
    gpu_solver.initialize_hybrid_kernels()
    magma = config.gpu_use_magma.value
    target_name = f"{target_name_prefix}hybrid_geqp3"
    if use_magma is not None:
      magma = "on" if use_magma else "off"
    params = {"magma": magma}
  rule = _linalg_ffi_lowering(target_name, operand_output_aliases={0: 0, 1: 1})
  return rule(ctx, a, jpvt, **params)

geqp3_p = linalg_primitive(
    _geqp3_dtype_rule, (_float | _complex, _int), (2, 1),
    _geqp3_shape_rule, "geqp3", multiple_results=True, require_same=False)
register_cpu_gpu_lowering(geqp3_p, _geqp3_cpu_gpu_lowering)


def _qr_shape_rule(shape, *, pivoting, full_matrices, **_):
  m, n = shape
  k = m if full_matrices else core.min_dim(m, n)
  return ((m, k), (k, n), (n,)) if pivoting else ((m, k), (k, n))

def _qr_dtype_rule(dtype, *, pivoting, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  return (dtype, dtype, dtypes.dtype(np.int32)) if pivoting else (dtype, dtype)

def qr_jvp_rule(primals, tangents, *, pivoting, full_matrices, use_magma):
  # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
  x, = primals
  dx, = tangents
  q, r, *p = qr_p.bind(x, pivoting=pivoting, full_matrices=False, use_magma=use_magma)
  *_, m, n = x.shape
  if m < n or (full_matrices and m != n):
    raise NotImplementedError(
      "Unimplemented case of QR decomposition derivative")
  if pivoting:
    dx = dx[..., p[0]]
  dx_rinv = triangular_solve(r, dx)  # Right side solve by default
  qt_dx_rinv = _H(q) @ dx_rinv
  qt_dx_rinv_lower = _tril(qt_dx_rinv, -1)
  do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric
  # The following correction is necessary for complex inputs
  I = lax.expand_dims(lax_internal._eye(do.dtype, (n, n)), range(qt_dx_rinv.ndim - 2))
  do = do + I * (qt_dx_rinv - qt_dx_rinv.real.astype(qt_dx_rinv.dtype))
  dq = q @ (do - qt_dx_rinv) + dx_rinv
  dr = (qt_dx_rinv - do) @ r
  if pivoting:
    dp = ad_util.Zero.from_primal_value(p[0])
    return (q, r, p[0]), (dq, dr, dp)
  return (q, r), (dq, dr)

def _qr_lowering(a, *, pivoting, full_matrices, use_magma):
  *batch_dims, m, n = a.shape
  if m == 0 or n == 0:
    k = m if full_matrices else core.min_dim(m, n)
    q = lax.broadcast_in_dim(lax_internal._eye(a.dtype, (m, k)),
                             (*batch_dims, m, k),
                             (len(batch_dims), len(batch_dims) + 1))
    r = lax.full((*batch_dims, k, n), 0, dtype=a.dtype)
    if pivoting:
      p = lax.full((*batch_dims, n), 0, dtype=np.dtype(np.int32))
      return q, r, p
    return q, r

  if pivoting:
    jpvt = lax.full((*batch_dims, n), 0, dtype=np.dtype(np.int32))
    r, p, taus = geqp3(a, jpvt, use_magma=use_magma)
    p -= 1  # Convert geqp3's 1-based indices to 0-based indices by subtracting 1.
  else:
    r, taus = geqrf(a)

  if m < n:
    q = householder_product(r[..., :m, :m], taus)
  elif full_matrices:
    pads = [(0, 0, 0)] * (len(batch_dims) + 1) + [(0, m - n, 0)]
    q = lax.pad(r, lax_internal._zero(r), pads)
    q = householder_product(q, taus)
  else:
    q = householder_product(r, taus)
    r = r[..., :n, :n]
  r = _triu(r)
  if pivoting:
    return q, r, p
  return q, r

qr_p = linalg_primitive(
    _qr_dtype_rule, (_float | _complex,), (2,), _qr_shape_rule, "qr",
    multiple_results=True)
ad.primitive_jvps[qr_p] = qr_jvp_rule
mlir.register_lowering(qr_p, mlir.lower_fun(_qr_lowering))


# Schur Decomposition

def _schur_shape_rule(shape, *, compute_schur_vectors, **_):
  if shape[0] != shape[1]:
    raise ValueError(
        f"The input to schur must be a square matrix. Got shape {shape}.")
  return (shape, shape) if compute_schur_vectors else (shape,)

def _schur_dtype_rule(dtype, *, compute_schur_vectors, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  return (dtype, dtype) if compute_schur_vectors else (dtype,)

def _schur_cpu_lowering(ctx, operand, *, compute_schur_vectors, sort_eig_vals,
                        select_callable):
  del select_callable  # unused
  if sort_eig_vals:
    raise NotImplementedError(
        "The sort feature of LAPACK's gees routine is not implemented.")

  operand_aval, = ctx.avals_in
  batch_dims = operand_aval.shape[:-2]
  real = operand_aval.dtype == np.float32 or operand_aval.dtype == np.float64
  target_name = lapack.prepare_lapack_call("gees_ffi", operand_aval.dtype)

  info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
  eigvals_aval = ShapedArray(operand_aval.shape[:-1], operand_aval.dtype)
  if real:
    avals_out = [operand_aval, operand_aval, eigvals_aval, eigvals_aval,
                 info_aval, info_aval]
  else:
    avals_out = [operand_aval, operand_aval, eigvals_aval, info_aval, info_aval]

  mode = (
      lapack.schur.ComputationMode.kComputeSchurVectors
      if compute_schur_vectors
      else lapack.schur.ComputationMode.kNoComputeSchurVectors
  )
  rule = _linalg_ffi_lowering(target_name, avals_out=avals_out,
                              operand_output_aliases={0: 0})
  schur_form, schur_vectors, *_, info = rule(
      ctx, operand, mode=_enum_attr(mode),
      sort=_enum_attr(lapack.schur.Sort.kNoSortEigenvalues))

  ok = mlir.compare_hlo(
      info, mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")

  schur_form = _replace_not_ok_with_nan(ctx, batch_dims, ok, schur_form,
                                        ctx.avals_out[0])
  output = [schur_form]
  if compute_schur_vectors:
    schur_vectors = _replace_not_ok_with_nan(ctx, batch_dims, ok, schur_vectors,
                                             ctx.avals_out[1])
    output.append(schur_vectors)

  return output

schur_p = linalg_primitive(
    _schur_dtype_rule, (_float | _complex,), (2,), _schur_shape_rule, "schur",
    multiple_results=True)
mlir.register_lowering(schur_p, _schur_cpu_lowering, platform="cpu")


# Singular value decomposition

def _svd_shape_rule(shape, *, full_matrices, compute_uv, subset_by_index, **_):
  m, n = shape
  rank = core.min_dim(m, n)
  if subset_by_index is not None:
    if full_matrices and subset_by_index != (0, rank):
      raise ValueError("full_matrices and subset_by_index cannot both be set")
    rank = core.min_dim(rank, subset_by_index[1] - subset_by_index[0])
  if compute_uv:
    return (
        (rank,),
        (m, m if full_matrices else rank),
        (n if full_matrices else rank, n),
    )
  else:
    return (rank,),

def _svd_dtype_rule(dtype, *, compute_uv, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  real_dtype = lax_internal._complex_basetype(dtype)
  if compute_uv:
    return real_dtype, dtype, dtype
  else:
    return real_dtype,

@config.default_matmul_precision("float32")
def _svd_jvp_rule(
    primals, tangents, *, full_matrices, compute_uv, subset_by_index,
    algorithm=None,
):
  A, = primals
  dA, = tangents
  s, U, Vt = svd_p.bind(
      A, full_matrices=False, compute_uv=True, subset_by_index=subset_by_index,
      algorithm=algorithm,
  )

  if compute_uv and full_matrices:
    # TODO: implement full matrices case, documented here: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    raise NotImplementedError(
      "Singular value decomposition JVP not implemented for full matrices")

  Ut, V = _H(U), _H(Vt)
  s_dim = s[..., None, :]
  dS = Ut @ dA @ V
  ds = _extract_diagonal(dS.real)

  if not compute_uv:
    return (s,), (ds,)

  s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
  s_diffs_zeros = lax_internal._eye(s.dtype, (s.shape[-1], s.shape[-1]))  # jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
  s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
  F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
  dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
  SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

  s_zeros = (s == 0).astype(s.dtype)
  s_inv = 1 / (s + s_zeros) - s_zeros
  s_inv_mat = _construct_diagonal(s_inv)
  dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
  dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
  dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))

  m, n = A.shape[-2:]
  if m > n:
    dAV = dA @ V
    dU = dU + (dAV - U @ (Ut @ dAV)) / s_dim.astype(A.dtype)
  if n > m:
    dAHU = _H(dA) @ U
    dV = dV + (dAHU - V @ (Vt @ dAHU)) / s_dim.astype(A.dtype)

  return (s, U, Vt), (ds, dU, _H(dV))

def _empty_svd(a, *, full_matrices, compute_uv):
  batch_shape = a.shape[:-2]
  m, n = a.shape[-2:]
  s = lax.full(batch_shape + (0,), 0, dtype=lax_internal._complex_basetype(a.dtype))
  if not compute_uv:
    return (s,)
  if full_matrices:
    size = max(m, n)
    u = lax.broadcast_in_dim(lax_internal._eye(a.dtype, (size, size)),
                             (*batch_shape, size, size),
                             (len(batch_shape), len(batch_shape) + 1))
  else:
    u = lax.full(batch_shape + (m, n), 0, dtype=a.dtype)
  v = lax.full(batch_shape + (0, 0), 0, dtype=a.dtype)
  if m < n:
    u, v = v, u
  return s, u, v

def _svd_computation_attr(compute_uv, full_matrices):
  mode = "A"
  if full_matrices is None:
    full_matrices = True
  if not compute_uv:
    mode = "N"
  elif not full_matrices:
    mode = "S"
  return _char_attr(mode)

def _svd_cpu_gpu_lowering(
    ctx,
    operand,
    *,
    full_matrices,
    compute_uv,
    subset_by_index,
    target_name_prefix: str,
    algorithm=None,
):
  operand_aval, = ctx.avals_in
  s_aval = ctx.avals_out[0]
  m, n = operand_aval.shape[-2:]
  batch_dims = operand_aval.shape[:-2]

  if not (subset_by_index is None or subset_by_index == (0, min(m, n))):
    raise NotImplementedError("subset_by_index not implemented for CPU and GPU")

  if m == 0 or n == 0:
    return mlir.lower_fun(_empty_svd, multiple_results=True)(
        ctx,
        operand,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )
  if target_name_prefix == "cpu":
    if algorithm is None or algorithm == SvdAlgorithm.DEFAULT:
      target_name = lapack.prepare_lapack_call("gesdd_ffi", operand_aval.dtype)
    elif algorithm == SvdAlgorithm.QR:
      target_name = lapack.prepare_lapack_call("gesvd_ffi", operand_aval.dtype)
    else:
      raise NotImplementedError(
          "The SVD Jacobi algorithm is not implemented on CPU.")
    mode = _svd_computation_attr(compute_uv, full_matrices)
    info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
    if compute_uv:
      s_aval, u_aval, vt_aval = ctx.avals_out
    else:
      s_aval, = ctx.avals_out
      # TODO(danfm): It should be possible to skip instantiating these arrays
      # when they are not used.
      u_aval = ShapedArray((*batch_dims, m,
                            m if full_matrices else core.min_dim(m, n)),
                           operand_aval.dtype)
      vt_aval = ShapedArray((*batch_dims,
                             n if full_matrices else core.min_dim(m, n), n),
                            operand_aval.dtype)
    avals_out = [operand_aval, s_aval, u_aval, vt_aval, info_aval]
    rule = _linalg_ffi_lowering(target_name, avals_out=avals_out,
                                operand_output_aliases={0: 0})
    _, s, u, vt, info = rule(ctx, operand, mode=mode)
  else:
    s, u, vt, info = _svd_gpu_sub_lowering(ctx, operand,
                                           full_matrices=full_matrices,
                                           compute_uv=compute_uv,
                                           target_name_prefix=target_name_prefix,
                                           algorithm=algorithm)

  zeros = mlir.full_like_aval(ctx, 0, ShapedArray(batch_dims, np.dtype(np.int32)))
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  s = _replace_not_ok_with_nan(ctx, batch_dims, ok, s, s_aval)
  result = [s]
  if compute_uv:
    u_aval, vt_aval = ctx.avals_out[1:]
    u = _replace_not_ok_with_nan(ctx, batch_dims, ok, u, u_aval)
    vt = _replace_not_ok_with_nan(ctx, batch_dims, ok, vt, vt_aval)
    result += [u, vt]

  return result

def _svd_gpu_sub_lowering(ctx, operand, *, full_matrices, compute_uv,
                          target_name_prefix, algorithm):
  operand_aval, = ctx.avals_in
  if compute_uv:
    s_aval, u_aval, vt_aval = ctx.avals_out
  else:
    s_aval, = ctx.avals_out
    u_aval = vt_aval = ShapedArray((), operand_aval.dtype)
  batch_dims = operand_aval.shape[:-2]
  info_aval = ShapedArray(batch_dims, np.dtype(np.int32))
  nb = len(batch_dims)
  m, n = operand_aval.shape[-2:]
  k = core.min_dim(m, n)

  transposed = False
  kwargs = {}

  # The Jacobi algorithm appears to outperform the default QR algorithm for
  # small to medium sized matrices. See:
  # https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9226-fast-singular-value-decomposition-on-gpus-v2.pdf
  # slide 5. With this in mind, we default to using the Jacobi algorithm for
  # matrices smaller than 1024x1024.
  #
  # Note that the Jacobi algorithm is only used by default for matrices with
  # concrete matrix dimensions. When using dynamic shapes, we always use the
  # default QR algorithm, but users can (in principle) override this behavior
  # by passing `use_jacobi=True`.
  #
  # TODO(danfm): Since this was originally implemented, hipSolver appears to
  # have added support for the Jacobi algorithm, so we should investigate
  # removing this condition.
  if algorithm is None or algorithm == SvdAlgorithm.DEFAULT:
    try:
      use_jacobi = target_name_prefix == "cu" and m <= 1024 and n <= 1024
    except core.InconclusiveDimensionOperation:
      use_jacobi = False
  else:
    use_jacobi = algorithm == SvdAlgorithm.JACOBI
  column_major = True
  if use_jacobi:
    target_name = f"{target_name_prefix}solver_gesvdj_ffi"
    # The gesvdjbatched kernel doesn't support "econ" mode, but it also only
    # supports matrices up to 32x32, so it's always worth using the batched
    # version and then slicing afterwards when the matrix is small enough.
    try:
      econ = not full_matrices and m > 32 and n > 32
    except core.InconclusiveDimensionOperation:
      econ = False
  else:
    target_name = f"{target_name_prefix}solver_gesvd_ffi"
    econ = not full_matrices
    # Because the base gesvd kernel only supports matrices where m >= n, we.
    transposed = m < n
    kwargs = {"transposed": transposed}
    if transposed:
      column_major = False

  if use_jacobi:
    # When using the Jacobi algorithm, the U and V matrices must always be
    # allocated even if compute_uv is False.
    u_aval = ShapedArray((*batch_dims, m, k if econ else m), u_aval.dtype)
    v_aval = ShapedArray((*batch_dims, n, k if econ else n), vt_aval.dtype)
    avals_out = [operand_aval, s_aval, u_aval, v_aval, info_aval]
  elif transposed:
    avals_out = [operand_aval, s_aval, vt_aval, u_aval, info_aval]
  else:
    avals_out = [operand_aval, s_aval, u_aval, vt_aval, info_aval]
  rule = _linalg_ffi_lowering(target_name, avals_out=avals_out,
                              operand_output_aliases={0: 0},
                              column_major=column_major)
  _, s, u, vt, info = rule(ctx, operand, full_matrices=not econ,
                           compute_uv=compute_uv, **kwargs)
  if use_jacobi and compute_uv:
    vt = hlo.transpose(
        vt,
        mlir.dense_int_array(np.array(tuple(range(nb)) + (nb + 1, nb))))
    if np.issubdtype(operand_aval.dtype, np.complexfloating):
      vt = hlo.complex(hlo.real(vt), hlo.negate(hlo.imag(vt)))
    if not full_matrices and not econ:
      nd = len(operand_aval.shape)
      u = mlir.slice_op(ctx, u, ctx.avals_out[1],
                        start_indices=np.zeros([nd], np.int64),
                        limit_indices=batch_dims + (m, k),
                        strides=np.ones([nd], np.int64))
      vt = mlir.slice_op(ctx, vt, ctx.avals_out[2],
                         start_indices=np.zeros([nd], np.int64),
                         limit_indices=batch_dims + (k, n),
                         strides=np.ones([nd], np.int64))
  if transposed:
    return s, vt, u, info
  else:
    return s, u, vt, info

def _svd_tpu(a, *, full_matrices, compute_uv, subset_by_index, algorithm=None):
  if algorithm is not None and algorithm != SvdAlgorithm.DEFAULT:
    raise NotImplementedError(
        "The SVD algorithm parameter is not implemented on TPU.")

  batch_dims = a.shape[:-2]
  fn = partial(
      lax_svd.svd,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
  )
  for _ in range(len(batch_dims)):
    fn = api.vmap(fn)

  if compute_uv:
    u, s, vh = fn(a)
    return [s, u, vh]
  else:
    s = fn(a)
    return [s]

def _svd_tpu_lowering_rule(
    ctx, operand, *, full_matrices, compute_uv, subset_by_index, algorithm=None
):
  del algorithm  # unused
  operand_aval, = ctx.avals_in
  m, n = operand_aval.shape[-2:]

  if m == 0 or n == 0:
    return mlir.lower_fun(_empty_svd, multiple_results=True)(
        ctx,
        operand,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
    )

  return mlir.lower_fun(_svd_tpu, multiple_results=True)(
      ctx,
      operand,
      full_matrices=full_matrices,
      compute_uv=compute_uv,
      subset_by_index=subset_by_index,
  )

svd_p = linalg_primitive(
    _svd_dtype_rule, (_float | _complex,), (2,), _svd_shape_rule, "svd",
    multiple_results=True)
ad.primitive_jvps[svd_p] = _svd_jvp_rule
register_cpu_gpu_lowering(svd_p, _svd_cpu_gpu_lowering)
mlir.register_lowering(svd_p, _svd_tpu_lowering_rule)


# Symmetric product

def _symmetric_product_shape_rule(a_shape, c_shape, **_):
  if a_shape[0] != c_shape[1] or c_shape[0] != c_shape[1]:
    raise ValueError(
        "symmetric_update expects a rectangular matrix of shape (m, n) and a "
        f"square matrix of shape (n, n). Got shapes {a_shape} and {c_shape}.")
  return c_shape

def _symmetric_product_jax_fn(a, c, *, alpha, beta):
  a_T = lax.transpose(a, (*range(a.ndim - 2), a.ndim - 1, a.ndim - 2))
  return alpha * lax.batch_matmul(
      a, a_T, precision=lax.Precision.HIGHEST) + beta * c

def _symmetric_product_gpu_lowering(
    platform, ctx, a_tensor, c_tensor, alpha, beta):
  a_aval, c_aval = ctx.avals_in[:2]
  dtype = a_aval.dtype
  alpha_aval = beta_aval = ShapedArray((), dtype)

  alpha_array = mlir.full_like_aval(ctx, alpha, alpha_aval)
  beta_array = mlir.full_like_aval(ctx, beta, beta_aval)

  rule = ffi.ffi_lowering(f"{platform}solver_syrk_ffi",
                          operand_output_aliases={1: 0})
  ctx = ctx.replace(avals_in=[a_aval, c_aval, alpha_aval, beta_aval])
  return rule(ctx, a_tensor, c_tensor, alpha_array, beta_array, transpose=False)

symmetric_product_p = standard_linalg_primitive(
    (_float, _float), (2, 2), _symmetric_product_shape_rule,
    "symmetric_product")
mlir.register_lowering(
    symmetric_product_p,
    partial(_symmetric_product_gpu_lowering, "cu"), platform="cuda")
mlir.register_lowering(
    symmetric_product_p,
    mlir.lower_fun(_symmetric_product_jax_fn, multiple_results=False))


# Triangular solve

def _triangular_solve_shape_rule(a_shape, b_shape, *, left_side=False, **_):
  if a_shape[0] != a_shape[1]:
    raise ValueError(
        "The first input to triangular_solve must be a square matrix. Got "
        f"shape {a_shape}.")
  common_dim = -2 if left_side else -1
  if a_shape[-1] != b_shape[common_dim]:
    raise ValueError(
        f"Incompatible shapes for arguments to triangular_solve: {a_shape} and "
        f"{b_shape}.")
  return b_shape

def _triangular_solve_dtype_rule(dtype, *_, **__):
  return dtypes.canonicalize_dtype(dtype)

def _triangular_solve_jvp_rule_a(
    g_a, ans, a, b, *, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  m, n = b.shape[-2:]
  k = 1 if unit_diagonal else 0
  g_a = _tril(g_a, k=-k) if lower else _triu(g_a, k=k)
  g_a = lax.neg(g_a)
  g_a = _T(g_a) if transpose_a else g_a
  g_a = g_a.conj() if conjugate_a else g_a
  dot = partial(lax.dot if g_a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)

  def a_inverse(rhs):
    return triangular_solve(a, rhs, left_side=left_side, lower=lower,
                            transpose_a=transpose_a, conjugate_a=conjugate_a,
                            unit_diagonal=unit_diagonal)

  # triangular_solve is about the same cost as matrix multiplication (~n^2 FLOPs
  # for matrix/vector inputs). Order these operations in whichever order is
  # cheaper.
  if left_side:
    assert g_a.shape[-2:] == a.shape[-2:] == (m, m) and ans.shape[-2:] == (m, n)
    if m > n:
      return a_inverse(dot(g_a, ans))  # A^{-1} (∂A X)
    else:
      return dot(a_inverse(g_a), ans)  # (A^{-1} ∂A) X
  else:
    assert g_a.shape[-2:] == a.shape[-2:] == (n, n) and ans.shape[-2:] == (m, n)
    if m < n:
      return a_inverse(dot(ans, g_a))  # (X ∂A) A^{-1}
    else:
      return dot(ans, a_inverse(g_a))  # X (∂A A^{-1})

def _triangular_solve_transpose_rule(
    cotangent, a, b, *, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  # Triangular solve is nonlinear in its first argument and linear in its second
  # argument, analogous to `div` but swapped.
  assert not ad.is_undefined_primal(a) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cotangent_b = ad_util.Zero(b.aval)
  else:
    cotangent_b = triangular_solve(a, cotangent, left_side=left_side,
                                   lower=lower, transpose_a=not transpose_a,
                                   conjugate_a=conjugate_a,
                                   unit_diagonal=unit_diagonal)
  return [None, cotangent_b]

def _triangular_solve_batching_rule(batched_args, batch_dims, *, left_side,
                                   lower, transpose_a, conjugate_a,
                                   unit_diagonal):
  x, y = batched_args
  bx, by = batch_dims
  if bx is batching.not_mapped:
    if left_side:
      y = batching.moveaxis(y, by, -1)
      y_flat = y.reshape(y.shape[:-2] + (y.shape[-2] * y.shape[-1],))
      bdim_out = y.ndim - 1
    else:
      y = batching.moveaxis(y, by, -2)
      y_flat = y.reshape(y.shape[:-3]  + (y.shape[-3] * y.shape[-2], y.shape[-1]))
      bdim_out = y.ndim - 2
    out_flat = triangular_solve(
        x, y_flat, left_side=left_side, lower=lower,
        transpose_a=transpose_a, conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal)
    return out_flat.reshape(y.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    x = batching.bdim_at_front(x, bx, size)
    y = batching.bdim_at_front(y, by, size)
    return triangular_solve(x, y, left_side=left_side, lower=lower,
                            transpose_a=transpose_a, conjugate_a=conjugate_a,
                            unit_diagonal=unit_diagonal), 0

def _triangular_solve_lowering(
    ctx, a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  out_aval, = ctx.avals_out
  if conjugate_a and not transpose_a:
    a = chlo.ConjOp(a)
    conjugate_a = False
  if not transpose_a:
    transpose = "NO_TRANSPOSE"
  else:
    transpose = "ADJOINT" if conjugate_a else "TRANSPOSE"
  out = hlo.triangular_solve(a, b, ir.BoolAttr.get(left_side),
                             ir.BoolAttr.get(lower),
                             ir.BoolAttr.get(unit_diagonal),
                             hlo.TransposeAttr.get(transpose))
  return [mlir.lower_with_sharding_in_types(ctx, out, out_aval)]


_cpu_lapack_types = {np.dtype(np.float32), np.dtype(np.float64),
                     np.dtype(np.complex64), np.dtype(np.complex128)}

def _triangular_solve_cpu_lower(
    ctx, a, b, *, left_side, lower, transpose_a,
    conjugate_a, unit_diagonal):
  a_aval, b_aval = ctx.avals_in

  if conjugate_a and not transpose_a:
    a = chlo.conj(a)
    conjugate_a = False
  if np.dtype(a_aval.dtype) in _cpu_lapack_types:
    target_name = lapack.prepare_lapack_call("trsm_ffi", a_aval.dtype)
    # TODO(b/397715595): Remove forward_compat check no earlier than 2025-03-18.
    if ctx.is_forward_compat() or jaxlib_version <= (0, 5, 1):
      alpha = mlir.ir_constant(np.array(1, dtype=a_aval.dtype)),
      alpha_aval = ShapedArray((), a_aval.dtype),
      batch_partitionable = False
    else:
      alpha = ()
      alpha_aval = ()
      batch_partitionable = True
    rule = _linalg_ffi_lowering(target_name,
                                [a_aval, b_aval, *alpha_aval],
                                operand_output_aliases={1: 0},
                                batch_partitionable=batch_partitionable)
    return rule(ctx, a, b, *alpha,
                side=_matrix_side_attr(left_side),
                uplo=_matrix_uplo_attr(lower),
                trans_x=_matrix_transpose_attr(transpose_a, conjugate_a),
                diag=_matrix_diagonal_attr(unit_diagonal))
  else:
    # Fall back to the HLO implementation for unsupported types or batching.
    # TODO: Consider swapping XLA for LAPACK in batched case
    if transpose_a:
      transpose = "ADJOINT" if conjugate_a else "TRANSPOSE"
    else:
      transpose = "NO_TRANSPOSE"
    return [hlo.triangular_solve(a, b, ir.BoolAttr.get(left_side),
                                 ir.BoolAttr.get(lower),
                                 ir.BoolAttr.get(unit_diagonal),
                                 hlo.TransposeAttr.get(transpose))]

triangular_solve_p = linalg_primitive(
    _triangular_solve_dtype_rule, (_float | _complex, _float | _complex),
    (2, 2), _triangular_solve_shape_rule, "triangular_solve")
ad.defjvp2(triangular_solve_p,
           _triangular_solve_jvp_rule_a,
           lambda g_b, _, a, b, **kws: triangular_solve(a, g_b, **kws))
ad.primitive_transposes[triangular_solve_p] = _triangular_solve_transpose_rule
batching.primitive_batchers[triangular_solve_p] = _triangular_solve_batching_rule
mlir.register_lowering(triangular_solve_p, _triangular_solve_lowering)
mlir.register_lowering(triangular_solve_p, _triangular_solve_cpu_lower,
                       platform="cpu")


# tridiagonal: Upper Hessenberg reduction

def _tridiagonal_shape_rule(shape, **_):
  if shape[0] != shape[1] or shape[1] == 0:
    raise ValueError(
        f"The input to tridiagonal must be a square matrix. Got shape {shape}.")
  n, _ = shape
  return shape, (n,), (n - 1,), (n - 1,)

def _tridiagonal_dtype_rule(dtype, **_):
  dtype = dtypes.canonicalize_dtype(dtype)
  real_dtype = lax_internal._complex_basetype(dtype)
  return dtype, real_dtype, real_dtype, dtype

def _tridiagonal_cpu_gpu_lowering(ctx, a, *, lower, target_name_prefix):
  a_aval, = ctx.avals_in
  arr_aval, d_aval, e_aval, taus_aval = ctx.avals_out
  batch_dims = a_aval.shape[:-2]
  if target_name_prefix == "cpu":
    real = a_aval.dtype == np.float32 or a_aval.dtype == np.float64
    prefix = "sy" if real else "he"
    target_name = lapack.prepare_lapack_call(f"{prefix}trd_ffi", a_aval.dtype)
    params = {"uplo": _matrix_uplo_attr(lower)}
  else:
    target_name = f"{target_name_prefix}solver_sytrd_ffi"
    params = {"lower": lower}
  info_aval = ShapedArray(batch_dims, np.int32)
  rule = _linalg_ffi_lowering(
      target_name, avals_out=(*ctx.avals_out, info_aval),
      operand_output_aliases={0: 0})
  arr, d, e, taus, info = rule(ctx, a, **params)
  zeros = mlir.full_like_aval(ctx, 0, info_aval)
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  arr = _replace_not_ok_with_nan(ctx, batch_dims, ok, arr, arr_aval)
  d = _replace_not_ok_with_nan(ctx, batch_dims, ok, d, d_aval)
  e = _replace_not_ok_with_nan(ctx, batch_dims, ok, e, e_aval)
  taus = _replace_not_ok_with_nan(ctx, batch_dims, ok, taus, taus_aval)
  return arr, d, e, taus

tridiagonal_p = linalg_primitive(
    _tridiagonal_dtype_rule, (_float | _complex,), (2,),
    _tridiagonal_shape_rule, "tridiagonal", multiple_results=True)
register_cpu_gpu_lowering(tridiagonal_p, _tridiagonal_cpu_gpu_lowering)


# Tridiagonal solve

def _tridiagonal_solve_shape_rule(dl_shape, d_shape, du_shape, b_shape, **_):
  if dl_shape != d_shape or dl_shape != du_shape:
    raise TypeError(
        "tridiagonal_solve requires that all diagonal arguments have the same "
        "shape.")
  if dl_shape != b_shape[:-1]:
    raise TypeError(
        "tridiagonal_solve requires that the leading ndim-1 dimensions of b "
        "equal the dimensions of the diagonal arguments.")
  return b_shape

def _tridiagonal_solve_gpu_lowering(ctx, dl, d, du, b, *, target_name_prefix):
  target_name = f"{target_name_prefix}sparse_gtsv2_ffi"
  rule = _linalg_ffi_lowering(target_name, operand_output_aliases={3: 0})
  return rule(ctx, dl, d, du, b)

def _tridiagonal_solve_cpu_lowering(ctx, dl, d, du, b, **kwargs):
  del kwargs  # unused
  b_aval = ctx.avals_in[-1]
  batch_dims = b_aval.shape[:-2]
  target_name = lapack.prepare_lapack_call("gtsv_ffi", b_aval.dtype)
  info_aval = ShapedArray(batch_dims, np.int32)
  rule = _linalg_ffi_lowering(target_name,
                              avals_out=[*ctx.avals_in, info_aval],
                              operand_output_aliases={0: 0, 1: 1, 2: 2, 3: 3})
  *_, b_out, info = rule(ctx, dl, d, du, b)
  zeros = mlir.full_like_aval(ctx, 0, info_aval)
  ok = mlir.compare_hlo(info, zeros, "EQ", "SIGNED")
  return [_replace_not_ok_with_nan(ctx, batch_dims, ok, b_out, b_aval)]

def _tridiagonal_product(dl, d, du, b):
  y = lax.reshape(d, d.shape + (1,)) * b
  y = y.at[..., 1:, :].add(dl[..., 1:, None] * b[..., :-1, :])
  y = y.at[..., :-1, :].add(du[..., :-1, None] * b[..., 1:, :])
  return y

def _tridiagonal_solve_jvp_rule(primals, tangents):
  *diags, _ = primals
  *diags_dot, b_dot = tangents
  ans = tridiagonal_solve_p.bind(*primals)
  if all(type(p) is ad_util.Zero for p in diags_dot):
    rhs = b_dot
  else:
    matvec_dot = _tridiagonal_product(*map(ad.instantiate_zeros, diags_dot), ans)
    rhs = ad.add_tangents(b_dot, -matvec_dot)
  ans_dot = tridiagonal_solve_p.bind(*diags, rhs)
  return ans, ans_dot

def _tridiagonal_solve_transpose_rule(cotangent, dl, d, du, b):
  # Tridiagonal solve is nonlinear in the tridiagonal arguments and linear
  # otherwise.
  assert not (ad.is_undefined_primal(dl) or ad.is_undefined_primal(d) or
              ad.is_undefined_primal(du)) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cotangent_b = ad_util.Zero(b.aval)
  else:
    dl_trans = lax.concatenate((lax.zeros_like_array(du[..., -1:]), du[..., :-1]),
                               du.ndim-1)
    du_trans = lax.concatenate((dl[..., 1:], lax.zeros_like_array(dl[..., :1])),
                               dl.ndim-1)
    cotangent_b = tridiagonal_solve(dl_trans, d, du_trans, cotangent)
  return [None, None, None, cotangent_b]

def _tridiagonal_solve_batching_rule(batched_args, batch_dims):
  dl, d, du, b = batched_args
  bdl, bd, bdu, bb = batch_dims
  if (bdl is batching.not_mapped and
      bd is batching.not_mapped and
      bdu is batching.not_mapped):

    b = batching.moveaxis(b, bb, -2)
    b_flat = b.reshape(b.shape[:-3]  + (b.shape[-3], b.shape[-2] * b.shape[-1]))
    bdim_out = b.ndim - 2
    out_flat = tridiagonal_solve(dl, d, du, b_flat)
    return out_flat.reshape(b.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    dl = batching.bdim_at_front(dl, bdl, size)
    d = batching.bdim_at_front(d, bd, size)
    du = batching.bdim_at_front(du, bdu, size)
    b = batching.bdim_at_front(b, bb, size)
    return tridiagonal_solve(dl, d, du, b), 0

def _tridiagonal_solve_jax_impl(dl, d, du, b):
  def fwd(carry, args):
    cp, dp = carry
    a, b, c, d = args
    cp_next = c / (b - a * cp)
    dp_next = (d - a * dp) / (b - a * cp)
    return (cp_next, dp_next), (cp, dp)

  (_, final), (cp, dp) = lax.scan(
      fwd, (du[0] / d[0], b[0] / d[0]), (dl[1:], d[1:], du[1:], b[1:, :]),
      unroll=32)

  def bwd(xn, args):
    cp, dp = args
    x = dp - cp * xn
    return x, xn

  end, ans = lax.scan(bwd, final, (cp, dp), unroll=32, reverse=True)
  return lax.concatenate((end[None], ans), 0)

def _tridiagonal_solve_jax(dl, d, du, b, **_):
  impl = _tridiagonal_solve_jax_impl
  for _ in range(dl.ndim - 1):
    impl = api.vmap(impl)
  return impl(dl, d, du, b)

tridiagonal_solve_p = standard_linalg_primitive(
    (_float | _complex, _float | _complex, _float | _complex, _float | _complex),
    (1, 1, 1, 2), _tridiagonal_solve_shape_rule, "tridiagonal_solve")
ad.primitive_jvps[tridiagonal_solve_p] = _tridiagonal_solve_jvp_rule
ad.primitive_transposes[tridiagonal_solve_p] = _tridiagonal_solve_transpose_rule
batching.primitive_batchers[tridiagonal_solve_p] = _tridiagonal_solve_batching_rule
mlir.register_lowering(
    tridiagonal_solve_p,
    _tridiagonal_solve_cpu_lowering,
    platform='cpu')
mlir.register_lowering(
    tridiagonal_solve_p,
    partial(_tridiagonal_solve_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
mlir.register_lowering(
    tridiagonal_solve_p,
    partial(_tridiagonal_solve_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')
mlir.register_lowering(tridiagonal_solve_p, mlir.lower_fun(
    _tridiagonal_solve_jax, multiple_results=False))


# Utilities

def _broadcasted_matvec(a: Array, b: Array) -> Array:
  # This is a broadcasted dot_general with signature (...,n,m),(...,m)->(...,n)
  assert a.ndim >= 2
  assert b.ndim >= 1
  batch_shape = lax.broadcast_shapes(a.shape[:-2], b.shape[:-1])
  n_batch = len(batch_shape)
  a = _broadcast_to(a, (*batch_shape, *a.shape[-2:]))
  b = _broadcast_to(b, (*batch_shape, b.shape[-1]))

  dimension_numbers = (([a.ndim - 1], [b.ndim - 1]), (list(range(n_batch)), list(range(n_batch))))
  return lax.dot_general(a, b, dimension_numbers=dimension_numbers, precision=lax.Precision.HIGHEST)

def _check_solve_shapes(a: Array, b: Array):
  if not (a.ndim >= 2 and b.ndim in [a.ndim, a.ndim - 1] and
          a.shape[-1] == a.shape[-2] == b.shape[a.ndim - 2]):
    raise ValueError(
        "The arguments to solve must have shapes a=[..., m, m] and "
        f"b=[..., m, k] or b=[..., m]; got a={a.shape} and b={b.shape}")

def _solve(a: Array, b: Array) -> Array:
  _check_solve_shapes(a, b)

  # Broadcast leading dimensions of b to the shape of a, as is required by
  # custom_linear_solve.
  out_shape = tuple(d_a if d_b == 1 else d_b
                    for d_a, d_b in zip(a.shape[:-1] + (1,), b.shape))
  b = lax.broadcast_in_dim(b, out_shape, range(b.ndim))

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  lu_, _, permutation = lu(lax.stop_gradient(a))
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: _broadcasted_matvec(a, x),
      solve=lambda _, x: lu_solve(lu_, permutation, x, trans=0),
      transpose_solve=lambda _, x: lu_solve(lu_, permutation, x, trans=1))
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return api.vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)

def _T(x: Array) -> Array:
  return lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))

def _H(x: Array) -> Array:
  return _T(x).conj()

def symmetrize(x: Array) -> Array: return (x + _H(x)) / 2

def _tril(m: Array, k:int = 0) -> Array:
  *_, N, M = m.shape
  mask = lax_internal._tri(bool, (N, M), k)
  return lax.select(lax.broadcast(mask, m.shape[:-2]), m, lax.zeros_like_array(m))

def _triu(m: Array, k:int = 0) -> Array:
  *_, N, M = m.shape
  mask = lax_internal._tri(bool, (N, M), k - 1)
  return lax.select(lax.broadcast(mask, m.shape[:-2]), lax.zeros_like_array(m), m)

def _construct_diagonal(s: Array) -> Array:
  """Construct a (batched) diagonal matrix"""
  i = lax.iota('int32', s.shape[-1])
  return lax.full((*s.shape, s.shape[-1]), 0, s.dtype).at[..., i, i].set(s)

def _extract_diagonal(s: Array) -> Array:
  """Extract the diagonal from a batched matrix"""
  i = lax.iota('int32', min(s.shape[-2], s.shape[-1]))
  return s[..., i, i]

def _broadcast_to(x: Array, shape: tuple[int, ...]) -> Array:
  assert x.ndim <= len(shape)
  return lax.broadcast_in_dim(x, shape, range(len(shape) - x.ndim, len(shape)))

def _nan_like_hlo(ctx: mlir.LoweringRuleContext, aval) -> ir.Value:
  if dtypes.issubdtype(aval.dtype, np.complexfloating):
    return mlir.full_like_aval(ctx, np.nan + np.nan * 1j, aval)
  else:
    return mlir.full_like_aval(ctx, np.nan, aval)

def _broadcasting_select_hlo(ctx, which, which_aval, x, x_aval, y, y_aval) -> ir.Value:
  """Wrapper around XLA `Select` that broadcasts its arguments."""
  out_shapes = list(lax_internal.broadcast_shapes(
      tuple(which_aval.shape), tuple(x_aval.shape), tuple(y_aval.shape)))
  which, x, y = mlir.multi_broadcast_in_dim(ctx, (which, x, y),
                                            (which_aval, x_aval, y_aval),
                                            out_shapes)
  return hlo.select(which, x, y)

def _replace_not_ok_with_nan(ctx, batch_dims, ok, x, x_aval):
  num_bcast_dims = len(x_aval.shape) - len(batch_dims)
  select_aval = ShapedArray(batch_dims + (1,) * num_bcast_dims, np.bool_)
  return _broadcasting_select_hlo(
      ctx,
      mlir.broadcast_in_dim(ctx, ok, select_aval,
                            broadcast_dimensions=range(len(batch_dims))),
      select_aval,
      x, x_aval, _nan_like_hlo(ctx, x_aval), x_aval)

def _enum_attr(e):
  return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(8), e.value)

def _char_attr(c):
  return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(8), ord(c))

def _matrix_side_attr(left_side):
  return _char_attr("L" if left_side else "R")

def _matrix_uplo_attr(lower):
  return _char_attr("L" if lower else "U")

def _matrix_transpose_attr(transpose: bool, conjugate: bool):
  return _char_attr(("C" if conjugate else "T") if transpose else "N")

def _matrix_diagonal_attr(unit_diag: bool):
  return _char_attr("U" if unit_diag else "N")

def _column_major_matrix_layout(dim: int) -> tuple[int, ...]:
  # The layout for a batch of matrices with Fortran order.
  return (dim - 2, dim - 1) + tuple(range(dim - 3, -1, -1))

def _sdy_rule_for_aval(letters, num_batch_dims, aval):
  d = len(aval.shape) - num_batch_dims
  prefix = "... " if num_batch_dims and d >= 0 else ""
  return prefix + " ".join(next(letters) for _ in range(d))

def _build_sdy_sharding_rule(num_batch_dims, avals_in, avals_out):
  letters = iter(string.ascii_letters)
  lhs = ", ".join(
      _sdy_rule_for_aval(letters, num_batch_dims, a) for a in avals_in)
  rhs = ", ".join(
      _sdy_rule_for_aval(letters, num_batch_dims, a) for a in avals_out)
  sdy_sharding_rule = str_to_sdy_sharding_rule(f"{lhs} -> {rhs}")
  return sdy_sharding_rule_to_mlir(
      sdy_sharding_rule,
      [mlir.aval_to_ir_type(a) for a in avals_in],
      [mlir.aval_to_ir_type(a) for a in avals_out])

def _linalg_ffi_lowering(target_name, avals_in=None, avals_out=None,
                         operand_output_aliases=None, column_major=True,
                         num_non_batch_dims=2, batch_partitionable=True):
  # A lightweight wrapper around ffi.ffi_lowering that can automatically set
  # the layouts appropriately for column-major matrices, which most handlers
  # used here will expect.
  def rule(ctx, *args, **kwargs):
    avals_in_ = ctx.avals_in if avals_in is None else avals_in
    avals_out_ = ctx.avals_out if avals_out is None else avals_out

    # TODO(danfm): Add support for shape polymorphism and batch partitioning.
    has_dynamic_shape = any(
        not is_constant_shape(aval.shape) for aval in (*avals_in_, *avals_out_))
    batch_partitionable_ = batch_partitionable and not has_dynamic_shape

    max_num_dims = max(len(v.shape) for v in avals_in_)
    ctx = ctx.replace(avals_in=avals_in_, avals_out=avals_out_)
    operand_layouts = [
        _column_major_matrix_layout(len(aval.shape))
        if column_major and len(aval.shape) == max_num_dims else None
        for aval in avals_in_]
    result_layouts = [
        _column_major_matrix_layout(len(aval.shape))
        if column_major and len(aval.shape) == max_num_dims else None
        for aval in avals_out_]
    num_batch_dims = max_num_dims - num_non_batch_dims
    frontend_attrs = mlir.ir_attribute({"num_batch_dims": str(num_batch_dims)})
    if batch_partitionable_:
      extra_attributes = {"mhlo.frontend_attributes": frontend_attrs}
      if config.use_shardy_partitioner.value:
        extra_attributes["sdy.sharding_rule"] = _build_sdy_sharding_rule(
            num_batch_dims, avals_in_, avals_out_)
    else:
      extra_attributes = None
    rule = ffi.ffi_lowering(target_name, operand_layouts=operand_layouts,
                            result_layouts=result_layouts,
                            operand_output_aliases=operand_output_aliases,
                            extra_attributes=extra_attributes)
    return rule(ctx, *args, **kwargs)
  return rule
