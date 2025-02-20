# Copyright 2025 The JAX Authors.
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
"""
Tensor contraction operations for the jax.numpy namespace.
"""

from collections.abc import Sequence
from functools import partial

import numpy as np

import jax
from jax import lax
from jax._src import core
from jax._src import dtypes
from jax._src.api import jit
from jax._src.lax import lax as lax_internal
from jax._src.lax.lax import PrecisionLike
from jax._src.numpy import ufuncs
from jax._src.numpy import util
from jax._src.numpy.vectorize import vectorize
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.util import canonicalize_axis, set_module

export = set_module('jax.numpy')

@export
@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def dot(a: ArrayLike, b: ArrayLike, *,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None) -> Array:
  """Compute the dot product of two arrays.

  JAX implementation of :func:`numpy.dot`.

  This differs from :func:`jax.numpy.matmul` in two respects:

  - if either ``a`` or ``b`` is a scalar, the result of ``dot`` is equivalent to
    :func:`jax.numpy.multiply`, while the result of ``matmul`` is an error.
  - if ``a`` and ``b`` have more than 2 dimensions, the batch indices are
    stacked rather than broadcast.

  Args:
    a: first input array, of shape ``(..., N)``.
    b: second input array. Must have shape ``(N,)`` or ``(..., N, M)``.
      In the multi-dimensional case, leading dimensions must be broadcast-compatible
      with the leading dimensions of ``a``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the dot product of the inputs, with batch dimensions of
    ``a`` and ``b`` stacked rather than broadcast.

  See also:
    - :func:`jax.numpy.matmul`: broadcasted batched matmul.
    - :func:`jax.lax.dot_general`: general batched matrix multiplication.

  Examples:
    For scalar inputs, ``dot`` computes the element-wise product:

    >>> x = jnp.array([1, 2, 3])
    >>> jnp.dot(x, 2)
    Array([2, 4, 6], dtype=int32)

    For vector or matrix inputs, ``dot`` computes the vector or matrix product:

    >>> M = jnp.array([[2, 3, 4],
    ...                [5, 6, 7],
    ...                [8, 9, 0]])
    >>> jnp.dot(M, x)
    Array([20, 38, 26], dtype=int32)
    >>> jnp.dot(M, M)
    Array([[ 51,  60,  29],
           [ 96, 114,  62],
           [ 61,  78,  95]], dtype=int32)

    For higher-dimensional matrix products, batch dimensions are stacked, whereas
    in :func:`~jax.numpy.matmul` they are broadcast. For example:

    >>> a = jnp.zeros((3, 2, 4))
    >>> b = jnp.zeros((3, 4, 1))
    >>> jnp.dot(a, b).shape
    (3, 2, 3, 1)
    >>> jnp.matmul(a, b).shape
    (3, 2, 1)
  """
  a, b = util.ensure_arraylike("dot", a, b)
  dtypes.check_user_dtype_supported(preferred_element_type, "dot")
  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(a, b, return_weak_type_flag=True)
  else:
    output_weak_type = False

  batch_dims = ((), ())
  a_ndim, b_ndim = np.ndim(a), np.ndim(b)
  if a_ndim == 0 or b_ndim == 0:
    contract_dims: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
  else:
    if b_ndim == 1:
      contract_dims = ((a_ndim - 1,), (0,))
    else:
      contract_dims = ((a_ndim - 1,), (b_ndim - 2,))
  result = lax.dot_general(a, b, dimension_numbers=(contract_dims, batch_dims),
                           precision=precision,
                           preferred_element_type=preferred_element_type)
  return lax_internal._convert_element_type(result, preferred_element_type,
                                            output_weak_type)


@export
@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def matmul(a: ArrayLike, b: ArrayLike, *,
           precision: PrecisionLike = None,
           preferred_element_type: DTypeLike | None = None,
           ) -> Array:
  """Perform a matrix multiplication.

  JAX implementation of :func:`numpy.matmul`.

  Args:
    a: first input array, of shape ``(N,)`` or ``(..., K, N)``.
    b: second input array. Must have shape ``(N,)`` or ``(..., N, M)``.
      In the multi-dimensional case, leading dimensions must be broadcast-compatible
      with the leading dimensions of ``a``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the matrix product of the inputs. Shape is ``a.shape[:-1]``
    if ``b.ndim == 1``, otherwise the shape is ``(..., K, M)``, where leading
    dimensions of ``a`` and ``b`` are broadcast together.

  See Also:
    - :func:`jax.numpy.linalg.vecdot`: batched vector product.
    - :func:`jax.numpy.linalg.tensordot`: batched tensor product.
    - :func:`jax.lax.dot_general`: general N-dimensional batched dot product.

  Examples:
    Vector dot products:

    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.matmul(a, b)
    Array(32, dtype=int32)

    Matrix dot product:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[1, 2],
    ...                [3, 4],
    ...                [5, 6]])
    >>> jnp.matmul(a, b)
    Array([[22, 28],
           [49, 64]], dtype=int32)

    For convenience, in all cases you can do the same computation using
    the ``@`` operator:

    >>> a @ b
    Array([[22, 28],
           [49, 64]], dtype=int32)
  """
  a, b = util.ensure_arraylike("matmul", a, b)
  dtypes.check_user_dtype_supported(preferred_element_type, "matmul")
  for i, x in enumerate((a, b)):
    if np.ndim(x) < 1:
      msg = (f"matmul input operand {i} must have ndim at least 1, "
             f"but it has ndim {np.ndim(x)}")
      raise ValueError(msg)
  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(a, b, return_weak_type_flag=True)
  else:
    output_weak_type = False

  a_is_mat, b_is_mat = (np.ndim(a) > 1), (np.ndim(b) > 1)
  a_batch_dims: tuple[int | None, ...] = np.shape(a)[:-2] if a_is_mat else ()
  b_batch_dims: tuple[int | None, ...] = np.shape(b)[:-2] if b_is_mat else ()
  num_batch_dims = max(len(a_batch_dims), len(b_batch_dims))
  a_batch_dims = (None,) * (num_batch_dims - len(a_batch_dims)) + a_batch_dims
  b_batch_dims = (None,) * (num_batch_dims - len(b_batch_dims)) + b_batch_dims

  # Dimensions to squeeze from the inputs.
  a_squeeze: list[int] = []
  b_squeeze: list[int] = []

  # Positions of batch dimensions in squeezed inputs.
  a_batch = []
  b_batch = []

  # Desired index in final output of each kind of dimension, in the order that
  # lax.dot_general will emit them.
  idx_batch: list[int] = []
  idx_a_other: list[int] = []  # other = non-batch, non-contracting.
  idx_b_other: list[int] = []
  for i, (ba, bb) in enumerate(zip(a_batch_dims, b_batch_dims)):
    if ba is None:
      idx_b_other.append(i)
    elif bb is None:
      idx_a_other.append(i)
    elif core.definitely_equal(ba, 1):
      idx_b_other.append(i)
      a_squeeze.append(len(idx_batch) + len(idx_a_other) + len(a_squeeze))
    elif core.definitely_equal(bb, 1):
      idx_a_other.append(i)
      b_squeeze.append(len(idx_batch) + len(idx_b_other) + len(b_squeeze))
    elif core.definitely_equal(ba, bb):
      a_batch.append(len(idx_batch) + len(idx_a_other))
      b_batch.append(len(idx_batch) + len(idx_b_other))
      idx_batch.append(i)
    else:
      raise ValueError("Incompatible shapes for matmul arguments: {} and {}"
                       .format(np.shape(a), np.shape(b)))

  if a_is_mat: idx_a_other.append(num_batch_dims)
  if b_is_mat: idx_b_other.append(num_batch_dims + a_is_mat)
  perm = np.argsort(np.concatenate([idx_batch, idx_a_other, idx_b_other]))

  a = lax.squeeze(a, tuple(a_squeeze))
  b = lax.squeeze(b, tuple(b_squeeze))
  out = lax.dot_general(
    a, b, (((np.ndim(a) - 1,), (np.ndim(b) - 1 - b_is_mat,)), (a_batch, b_batch)),
    precision=precision, preferred_element_type=preferred_element_type)
  result = lax.transpose(out, perm)
  return lax_internal._convert_element_type(result, preferred_element_type, output_weak_type)


@export
@jit
def matvec(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Batched matrix-vector product.

  JAX implementation of :func:`numpy.matvec`.

  Args:
    x1: array of shape ``(..., M, N)``
    x2: array of shape ``(..., N)``. Leading dimensions must be broadcast-compatible
      with leading dimensions of ``x1``.

  Returns:
    An array of shape ``(..., M)`` containing the batched matrix-vector product.

  See also:
    - :func:`jax.numpy.linalg.vecdot`: batched vector product.
    - :func:`jax.numpy.vecmat`: vector-matrix product.
    - :func:`jax.numpy.matmul`: general matrix multiplication.

  Examples:
    Simple matrix-vector product:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> x2 = jnp.array([7, 8, 9])
    >>> jnp.matvec(x1, x2)
    Array([ 50, 122], dtype=int32)

    Batched matrix-vector product:

    >>> x2 = jnp.array([[7, 8, 9],
    ...                 [5, 6, 7]])
    >>> jnp.matvec(x1, x2)
    Array([[ 50, 122],
           [ 38,  92]], dtype=int32)
  """
  util.check_arraylike("matvec", x1, x2)
  return vectorize(matmul, signature="(n,m),(m)->(n)")(x1, x2)


@export
@jit
def vecmat(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Batched conjugate vector-matrix product.

  JAX implementation of :func:`numpy.vecmat`.

  Args:
    x1: array of shape ``(..., M)``.
    x2: array of shape ``(..., M, N)``. Leading dimensions must be broadcast-compatible
      with leading dimensions of ``x1``.

  Returns:
    An array of shape ``(..., N)`` containing the batched conjugate vector-matrix product.

  See also:
    - :func:`jax.numpy.linalg.vecdot`: batched vector product.
    - :func:`jax.numpy.matvec`: matrix-vector product.
    - :func:`jax.numpy.matmul`: general matrix multiplication.

  Examples:
    Simple vector-matrix product:

    >>> x1 = jnp.array([[1, 2, 3]])
    >>> x2 = jnp.array([[4, 5],
    ...                 [6, 7],
    ...                 [8, 9]])
    >>> jnp.vecmat(x1, x2)
    Array([[40, 46]], dtype=int32)

    Batched vector-matrix product:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> jnp.vecmat(x1, x2)
    Array([[ 40,  46],
           [ 94, 109]], dtype=int32)
  """
  util.check_arraylike("matvec", x1, x2)
  return vectorize(matmul, signature="(n),(n,m)->(m)")(ufuncs.conj(x1), x2)


@export
@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def vdot(
    a: ArrayLike, b: ArrayLike, *,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array:
  """Perform a conjugate multiplication of two 1D vectors.

  JAX implementation of :func:`numpy.vdot`.

  Args:
    a: first input array, if not 1D it will be flattened.
    b: second input array, if not 1D it will be flattened. Must have ``a.size == b.size``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    Scalar array (shape ``()``) containing the conjugate vector product of the inputs.

  See Also:
    - :func:`jax.numpy.vecdot`: batched vector product.
    - :func:`jax.numpy.matmul`: general matrix multiplication.
    - :func:`jax.lax.dot_general`: general N-dimensional batched dot product.

  Examples:
    >>> x = jnp.array([1j, 2j, 3j])
    >>> y = jnp.array([1., 2., 3.])
    >>> jnp.vdot(x, y)
    Array(0.-14.j, dtype=complex64)

    Note the difference between this and :func:`~jax.numpy.dot`, which does not
    conjugate the first input when complex:

    >>> jnp.dot(x, y)
    Array(0.+14.j, dtype=complex64)
  """
  util.check_arraylike("vdot", a, b)
  if dtypes.issubdtype(dtypes.dtype(a, canonicalize=True), np.complexfloating):
    a = ufuncs.conj(a)
  return dot(jax.numpy.ravel(a), jax.numpy.ravel(b), precision=precision,
             preferred_element_type=preferred_element_type)


@export
def vecdot(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1,
           precision: PrecisionLike = None,
           preferred_element_type: DTypeLike | None = None) -> Array:
  """Perform a conjugate multiplication of two batched vectors.

  JAX implementation of :func:`numpy.vecdot`.

  Args:
    a: left-hand side array.
    b: right-hand side array. Size of ``b[axis]`` must match size of ``a[axis]``,
      and remaining dimensions must be broadcast-compatible.
    axis: axis along which to compute the dot product (default: -1)
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the conjugate dot product of ``a`` and ``b`` along ``axis``.
    The non-contracted dimensions are broadcast together.

  See Also:
    - :func:`jax.numpy.vdot`: flattened vector product.
    - :func:`jax.numpy.vecmat`: vector-matrix product.
    - :func:`jax.numpy.matmul`: general matrix multiplication.
    - :func:`jax.lax.dot_general`: general N-dimensional batched dot product.

  Examples:
    Vector conjugate-dot product of two 1D arrays:

    >>> a = jnp.array([1j, 2j, 3j])
    >>> b = jnp.array([4., 5., 6.])
    >>> jnp.linalg.vecdot(a, b)
    Array(0.-32.j, dtype=complex64)

    Batched vector dot product of two 2D arrays:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[2, 3, 4]])
    >>> jnp.linalg.vecdot(a, b, axis=-1)
    Array([20, 47], dtype=int32)
  """
  x1_arr, x2_arr = util.ensure_arraylike("jnp.vecdot", x1, x2)
  if x1_arr.shape[axis] != x2_arr.shape[axis]:
    raise ValueError(f"axes must match; got shapes {x1_arr.shape} and {x2_arr.shape} with {axis=}")
  x1_arr = jax.numpy.moveaxis(x1_arr, axis, -1)
  x2_arr = jax.numpy.moveaxis(x2_arr, axis, -1)
  return vectorize(partial(vdot, precision=precision, preferred_element_type=preferred_element_type),
                   signature="(n),(n)->()")(x1_arr, x2_arr)


@export
def tensordot(a: ArrayLike, b: ArrayLike,
              axes: int | Sequence[int] | Sequence[Sequence[int]] = 2,
              *, precision: PrecisionLike = None,
              preferred_element_type: DTypeLike | None = None) -> Array:
  """Compute the tensor dot product of two N-dimensional arrays.

  JAX implementation of :func:`numpy.linalg.tensordot`.

  Args:
    a: N-dimensional array
    b: M-dimensional array
    axes: integer or tuple of sequences of integers. If an integer `k`, then
      sum over the last `k` axes of ``a`` and the first `k` axes of ``b``,
      in order. If a tuple, then ``axes[0]`` specifies the axes of ``a`` and
      ``axes[1]`` specifies the axes of ``b``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the tensor dot product of the inputs

  See also:
    - :func:`jax.numpy.einsum`: NumPy API for more general tensor contractions.
    - :func:`jax.lax.dot_general`: XLA API for more general tensor contractions.

  Examples:
    >>> x1 = jnp.arange(24.).reshape(2, 3, 4)
    >>> x2 = jnp.ones((3, 4, 5))
    >>> jnp.tensordot(x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Equivalent result when specifying the axes as explicit sequences:

    >>> jnp.tensordot(x1, x2, axes=([1, 2], [0, 1]))
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Equivalent result via :func:`~jax.numpy.einsum`:

    >>> jnp.einsum('ijk,jkm->im', x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Setting ``axes=1`` for two-dimensional inputs is equivalent to a matrix
    multiplication:

    >>> x1 = jnp.array([[1, 2],
    ...                 [3, 4]])
    >>> x2 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> jnp.linalg.tensordot(x1, x2, axes=1)
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)
    >>> x1 @ x2
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)

    Setting ``axes=0`` for one-dimensional inputs is equivalent to
    :func:`~jax.numpy.outer`:

    >>> x1 = jnp.array([1, 2])
    >>> x2 = jnp.array([1, 2, 3])
    >>> jnp.linalg.tensordot(x1, x2, axes=0)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
    >>> jnp.outer(x1, x2)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
  """
  a, b = util.ensure_arraylike("tensordot", a, b)
  dtypes.check_user_dtype_supported(preferred_element_type, "tensordot")
  a_ndim = np.ndim(a)
  b_ndim = np.ndim(b)

  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(a, b, return_weak_type_flag=True)
  else:
    output_weak_type = False

  if type(axes) is int:
    if axes > min(a_ndim, b_ndim):
      msg = "Number of tensordot axes (axes {}) exceeds input ranks ({} and {})"
      raise TypeError(msg.format(axes, a.shape, b.shape))
    contracting_dims = tuple(range(a_ndim - axes, a_ndim)), tuple(range(axes))
  elif isinstance(axes, (tuple, list)) and len(axes) == 2:
    ax1, ax2 = axes
    if type(ax1) == type(ax2) == int:
      contracting_dims = ((canonicalize_axis(ax1, a_ndim),),
                          (canonicalize_axis(ax2, b_ndim),))
    elif isinstance(ax1, (tuple, list)) and isinstance(ax2, (tuple, list)):
      if len(ax1) != len(ax2):
        msg = "tensordot requires axes lists to have equal length, got {} and {}."
        raise TypeError(msg.format(ax1, ax2))
      contracting_dims = (tuple(canonicalize_axis(i, a_ndim) for i in ax1),
                          tuple(canonicalize_axis(i, b_ndim) for i in ax2))
    else:
      msg = ("tensordot requires both axes lists to be either ints, tuples or "
             "lists, got {} and {}")
      raise TypeError(msg.format(ax1, ax2))
  else:
    msg = ("tensordot axes argument must be an int, a pair of ints, or a pair "
           "of lists/tuples of ints.")
    raise TypeError(msg)
  result = lax.dot_general(a, b, (contracting_dims, ((), ())),
                           precision=precision, preferred_element_type=preferred_element_type)
  return lax_internal._convert_element_type(result, preferred_element_type, output_weak_type)



@export
@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def inner(
    a: ArrayLike, b: ArrayLike, *, precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array:
  """Compute the inner product of two arrays.

  JAX implementation of :func:`numpy.inner`.

  Unlike :func:`jax.numpy.matmul` or :func:`jax.numpy.dot`, this always performs
  a contraction along the last dimension of each input.

  Args:
    a: array of shape ``(..., N)``
    b: array of shape ``(..., N)``
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array of shape ``(*a.shape[:-1], *b.shape[:-1])`` containing the batched vector
    product of the inputs.

  See also:
    - :func:`jax.numpy.vecdot`: conjugate multiplication along a specified axis.
    - :func:`jax.numpy.tensordot`: general tensor multiplication.
    - :func:`jax.numpy.matmul`: general batched matrix & vector multiplication.

  Examples:
    For 1D inputs, this implements standard (non-conjugate) vector multiplication:

    >>> a = jnp.array([1j, 3j, 4j])
    >>> b = jnp.array([4., 2., 5.])
    >>> jnp.inner(a, b)
    Array(0.+30.j, dtype=complex64)

    For multi-dimensional inputs, batch dimensions are stacked rather than broadcast:

    >>> a = jnp.ones((2, 3))
    >>> b = jnp.ones((5, 3))
    >>> jnp.inner(a, b).shape
    (2, 5)
  """
  a, b = util.ensure_arraylike("inner", a, b)
  if np.ndim(a) == 0 or np.ndim(b) == 0:
    a = jax.numpy.asarray(a, dtype=preferred_element_type)
    b = jax.numpy.asarray(b, dtype=preferred_element_type)
    return a * b
  return tensordot(a, b, (-1, -1), precision=precision,
                   preferred_element_type=preferred_element_type)


@export
@partial(jit, inline=True)
def outer(a: ArrayLike, b: ArrayLike, out: None = None) -> Array:
  """Compute the outer product of two arrays.

  JAX implementation of :func:`numpy.outer`.

  Args:
    a: first input array, if not 1D it will be flattened.
    b: second input array, if not 1D it will be flattened.
    out: unsupported by JAX.

  Returns:
    The outer product of the inputs ``a`` and ``b``. Returned array
    will be of shape ``(a.size, b.size)``.

  See also:
    - :func:`jax.numpy.inner`: compute the inner product of two arrays.
    - :func:`jax.numpy.einsum`: Einstein summation.

  Examples:
    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.outer(a, b)
    Array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]], dtype=int32)
  """
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.outer is not supported.")
  util.check_arraylike("outer", a, b)
  a, b = util.promote_dtypes(a, b)
  return jax.numpy.ravel(a)[:, None] * jax.numpy.ravel(b)[None, :]
