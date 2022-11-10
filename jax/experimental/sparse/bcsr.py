# Copyright 2022 The JAX Authors.
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

"""BCSR (Bached compressed row) matrix object and associated primitives."""

import operator

from typing import NamedTuple, Sequence, Tuple

import numpy as np

from jax import core
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse import bcoo
from jax.experimental.sparse.util import _broadcasting_vmap, _count_stored_elements, _csr_to_coo, _safe_asarray
import jax.numpy as jnp
from jax.util import split_list, safe_zip
from jax.interpreters import batching
from jax.interpreters import mlir

Shape = Tuple[int, ...]


class BCSRProperties(NamedTuple):
  n_batch: int
  n_dense: int
  nse: int


def _compatible(shape1, shape2):
  return all(s1 in (1, s2) for s1, s2 in safe_zip(shape1, shape2))


def _validate_bcsr_indices(indices: jnp.ndarray, indptr: jnp.ndarray,
                           shape: Sequence[int]) -> BCSRProperties:
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  assert jnp.issubdtype(indptr.dtype, jnp.integer)
  shape = tuple(shape)

  nse = indices.shape[-1]
  n_batch = indices.ndim - 1
  n_dense = len(shape) - n_batch - 2
  assert n_dense >= 0

  if not _compatible(indices.shape[:n_batch], shape[:n_batch]):
    raise ValueError("indices batch dimensions not compatible for "
                     f"indices.shape={indices.shape}, shape={shape}")
  if not _compatible(indptr.shape[:n_batch], shape[:n_batch]):
    raise ValueError("indptr batch dimensions not compatible for "
                     f"indptr.shape={indptr.shape}, shape={shape}")
  if indptr.shape[n_batch:] != (shape[n_batch] + 1,):
    raise ValueError("indptr shape must match the matrix shape plus 1.")

  return BCSRProperties(n_batch=n_batch, n_dense=n_dense, nse=nse)


def _validate_bcsr(data: jnp.ndarray, indices: jnp.ndarray,
                   indptr: jnp.ndarray, shape: Sequence[int]) -> BCSRProperties:
  props = _validate_bcsr_indices(indices, indptr, shape)
  shape = tuple(shape)
  n_batch, n_dense, nse = props.n_batch, props.n_dense, props.nse
  n_sparse = len(shape) - n_batch - n_dense
  if n_sparse != 2:
    raise ValueError("BCSR array must have 2 sparse dimensions; "
                     f"{n_sparse} is given.")
  if not _compatible(data.shape[:n_batch], shape[:n_batch]):
    raise ValueError("data batch dimensions not compatible for "
                    f"data.shape={data.shape}, shape={shape}")
  if data.shape[-(n_dense + 1):] != (nse,) + shape[n_batch + 2:]:
    raise ValueError(f"Invalid data.shape={data.shape} for "
                    f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  return props


def _bcsr_to_bcoo(indices: jnp.ndarray, indptr: jnp.ndarray, *,
                  shape: Sequence[int]) -> jnp.ndarray:
  """Given BCSR (indices, indptr), return BCOO (indices)."""
  n_batch, _, _ = _validate_bcsr_indices(indices, indptr, shape)
  csr_to_coo = _csr_to_coo
  for _ in range(n_batch):
    csr_to_coo = _broadcasting_vmap(csr_to_coo)
  return jnp.stack(csr_to_coo(indices, indptr), axis=indices.ndim)


#--------------------------------------------------------------------
# bcsr_fromdense
bcsr_fromdense_p = core.Primitive('bcsr_fromdense')
bcsr_fromdense_p.multiple_results = True


_TRACED_NSE_ERROR = """
The error arose for the nse argument of bcsr_fromdense. In order for
BCSR.fromdense() to be used in traced/compiled code, you must pass a concrete
value to the nse (number of stored elements) argument.
"""


def bcsr_fromdense(mat, *, nse=None, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create BCSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCOO.
    nse : number of stored elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    mat_bcsr: BCSR representation of the matrix.
  """
  mat = jnp.asarray(mat)
  if nse is None:
    nse = _count_stored_elements(mat, n_batch, n_dense)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return BCSR(_bcsr_fromdense(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                              index_dtype=index_dtype),
              shape=mat.shape)


def _bcsr_fromdense(mat, *, nse, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create BCSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCSR, with
      ``ndim = n_batch + n_sparse + n_dense``.
    nse : number of stored elements in each batch.
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    data : array of shape
    ``mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]``
      and dtype ``mat.dtype``
    indices : array of shape ``mat.shape[:n_batch] + (nse,)`` and dtype of
      ``index_type``.
    indptr: array of shape ``mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)``
      and dtype of ``index_type``.
  """
  mat = jnp.asarray(mat)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return bcsr_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)


@bcsr_fromdense_p.def_impl
def _bcsr_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  mat = jnp.asarray(mat)
  n_sparse = mat.ndim - n_dense - n_batch
  if n_sparse != 2:
    raise ValueError("bcsr_fromdense: must have 2 sparse dimensions.")
  bcoo_mat = bcoo.bcoo_fromdense(mat, nse=nse, index_dtype=index_dtype,
                                 n_dense=n_dense, n_batch=n_batch)
  indices, indptr = bcoo._bcoo_to_bcsr(bcoo_mat.indices, shape=mat.shape)
  return bcoo_mat.data, indices, indptr


@bcsr_fromdense_p.def_abstract_eval
def _bcoo_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
  n_sparse = mat.ndim - n_batch - n_dense
  if n_sparse != 2:
    raise ValueError("bcsr_fromdense: must have 2 sparse dimensions.")
  data_shape = mat.shape[:n_batch] + (nse,) + mat.shape[n_batch + n_sparse:]
  index_shape = mat.shape[:n_batch] + (nse,)
  indptr_shape = mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)
  return (core.ShapedArray(data_shape, mat.dtype),
          core.ShapedArray(index_shape, index_dtype),
          core.ShapedArray(indptr_shape, index_dtype))


def _bcsr_fromdense_batching_rule(batched_args, batch_dims, *, nse, n_batch,
                                  n_dense, index_dtype):
  M, = batched_args
  if batch_dims != (0,):
    raise NotImplementedError(f"batch_dims={batch_dims}")
  new_n_batch = n_batch + 1
  n_sparse = M.ndim - n_dense - new_n_batch
  if n_sparse != 2:
    raise ValueError("_bcsr_fromdense_batching_rule: must have 2 sparse "
                     f"dimensions but {n_sparse} is given.")
  return _bcsr_fromdense(M, nse=nse, n_batch=new_n_batch, n_dense=n_dense,
                         index_dtype=index_dtype), (0, 0, 0)


batching.primitive_batchers[bcsr_fromdense_p] = _bcsr_fromdense_batching_rule
mlir.register_lowering(bcsr_fromdense_p, mlir.lower_fun(
    _bcsr_fromdense_impl, multiple_results=True))


#----------------------------------------------------------------------
# bcsr_todense
bcsr_todense_p = core.Primitive('bcsr_todense')


def bcsr_todense(mat):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    mat: BCSR matrix.

  Returns:
    The dense version of ``mat``.
  """
  return _bcsr_todense(mat.data, mat.indices, mat.indptr,
                       shape=tuple(mat.shape))


def _bcsr_todense(data, indices, indptr, *, shape):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + dense_dims``.
    indices : array of shape ``batch_dims + (nse,)``.
    indptr : array of shape ``batch_dims + (shape[len(batch_dims)] + 1,).
    shape : tuple; the shape of the (batched) matrix. Equal to
      ``batch_dims + 2(sparse_dims) + dense_dims``
  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcsr_todense_p.bind(jnp.asarray(data), jnp.asarray(indices),
                             jnp.asarray(indptr), shape=shape)


@bcsr_todense_p.def_impl
def _bcsr_todense_impl(data, indices, indptr, *, shape):
  bcoo_indices = _bcsr_to_bcoo(indices, indptr, shape=shape)
  return (bcoo.BCOO((data, bcoo_indices), shape=shape)).todense()


@bcsr_todense_p.def_abstract_eval
def _bcsr_todense_abstract_eval(data, indices, indptr, *, shape):
  _validate_bcsr(data, indices, indptr, shape)
  return core.ShapedArray(shape, data.dtype)


def _bcsr_todense_batching_rule(batched_args, batch_dims, *, shape):
  data, indices, indptr = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. "
                              "Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  if batch_dims[2] is None:
    indptr = indptr[None, ...]
  return _bcsr_todense(data, indices, indptr, shape=shape), 0

batching.primitive_batchers[bcsr_todense_p] = _bcsr_todense_batching_rule
mlir.register_lowering(bcsr_todense_p, mlir.lower_fun(
    _bcsr_todense_impl, multiple_results=False))


#--------------------------------------------------------------------
# bcsr_extract
bcsr_extract_p = core.Primitive('bcsr_extract')


def bcsr_extract(indices, indptr, mat):
  """Extract values from a dense matrix at given BCSR (indices, indptr).

  Args:
    indices: An ndarray; see BCSR indices.
    indptr: An ndarray; see BCSR indptr.
    mat: A dense matrix.

  Returns:
    An ndarray; see BCSR data.
  """
  return bcsr_extract_p.bind(indices, indptr, mat)


@bcsr_extract_p.def_impl
def _bcsr_extract_impl(indices, indptr, mat):
  mat = jnp.asarray(mat)
  bcoo_indices = _bcsr_to_bcoo(indices, indptr, shape=mat.shape)
  return bcoo.bcoo_extract(bcoo_indices, mat)


@bcsr_extract_p.def_abstract_eval
def _bcsr_extract_abstract_eval(indices, indptr, mat):
  n_batch, n_dense, nse = _validate_bcsr_indices(indices, indptr, mat.shape)
  out_shape = mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]
  return core.ShapedArray(out_shape, mat.dtype)


mlir.register_lowering(bcsr_extract_p, mlir.lower_fun(
    _bcsr_extract_impl, multiple_results=False))


class BCSR(JAXSparse):
  """Experimental batched CSR matrix implemented in JAX."""

  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  shape: Shape
  nse = property(lambda self: self.indices.shape[-1])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 1)
  n_sparse = property(lambda _: 2)
  n_dense = property(lambda self: self.data.ndim - self.indices.ndim)

  @property
  def _sparse_shape(self):
    return tuple(self.shape[self.n_batch:self.n_batch + 2])

  def __init__(self, args, *, shape):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices, self.indptr = _safe_asarray(args)
    super().__init__(args, shape=shape)

  def __repr__(self):
    name = self.__class__.__name__
    try:
      nse = self.nse
      n_batch = self.n_batch
      n_dense = self.n_dense
      dtype = self.dtype
      shape = list(self.shape)
    except Exception:  # pylint: disable=broad-except
      repr_ = f"{name}(<invalid>)"
    else:
      extra = f", nse={nse}"
      if n_batch: extra += f", n_batch={n_batch}"
      if n_dense: extra += f", n_dense={n_dense}"
      repr_ = f"{name}({dtype}{shape}{extra})"
    if isinstance(self.data, core.Tracer):
      repr_ = f"{type(self.data).__name__}[{repr_}]"
    return repr_

  def transpose(self, *args, **kwargs):
    raise NotImplementedError("Tranpose is not implemented.")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {}

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32', n_dense=0,
             n_batch=0, nse=0):
    """Create an empty BCSR instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if n_dense < 0 or n_batch < 0 or nse < 0:
      raise ValueError(f"Invalid inputs: shape={shape}, n_dense={n_dense},"
                       f"n_batch={n_batch}, nse={nse}")
    n_sparse = len(shape) - n_dense - n_batch
    if n_sparse != 2:
      raise ValueError("BCSR sparse.empty: must have 2 sparse dimensions.")
    batch_shape, sparse_shape, dense_shape = split_list(shape,
                                                        [n_batch, n_sparse])
    data = jnp.zeros((*batch_shape, nse, *dense_shape), dtype)
    indices = jnp.full((*batch_shape, nse), jnp.array(sparse_shape[1]),
                       index_dtype)
    indptr = jnp.zeros((*batch_shape, sparse_shape[0] + 1), index_dtype)
    return cls((data, indices, indptr), shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32, n_dense=0,
                n_batch=0):
    """Create a BCSR array from a (dense) :class:`DeviceArray`."""
    return bcsr_fromdense(mat, nse=nse, index_dtype=index_dtype,
                          n_dense=n_dense, n_batch=n_batch)

  def todense(self):
    """Create a dense version of the array."""
    return bcsr_todense(self)

  @classmethod
  def from_scipy_sparse(cls, mat, *, index_dtype=None, n_dense=0, n_batch=0):
    """Create a BCSR array from a :mod:`scipy.sparse` array."""
    if n_dense != 0 or n_batch != 0:
      raise NotImplementedError("BCSR from_scipy_sparse with nonzero n_dense/n_batch.")

    if mat.ndim != 2:
      raise ValueError(f"BCSR from_scipy_sparse requires 2D array; {mat.ndim}D is given.")

    mat = mat.tocsr()
    data = jnp.asarray(mat.data)
    indices = jnp.asarray(mat.indices).astype(index_dtype or jnp.int32)
    indptr = jnp.asarray(mat.indptr).astype(index_dtype or jnp.int32)
    return cls((data, indices, indptr), shape=mat.shape)

#--------------------------------------------------------------------
# vmappable handlers
def _bcsr_to_elt(cont, _, val, axis):
  if axis is None:
    return val
  if axis >= val.n_batch:
    raise ValueError(f"Cannot map in_axis={axis} for BCSR array with n_batch="
                     f"{val.n_batch}. in_axes for batched BCSR operations must "
                     "correspond to a batched dimension.")
  return BCSR((cont(val.data, axis),
               cont(val.indices, axis),
               cont(val.indptr, axis)),
              shape=val.shape[:axis] + val.shape[axis + 1:])


def _bcsr_from_elt(cont, axis_size, elt, axis):
  if axis > elt.n_batch:
    raise ValueError(f"BCSR: cannot add out_axis={axis} for BCSR array with "
                     f"n_batch={elt.n_batch}. BCSR batch axes must be a "
                     "contiguous block of leading dimensions.")
  return BCSR((cont(axis_size, elt.data, axis),
               cont(axis_size, elt.indices, axis),
               cont(axis_size, elt.indptr, axis)),
              shape=elt.shape[:axis] + (axis_size,) + elt.shape[axis:])

batching.register_vmappable(BCSR, int, int, _bcsr_to_elt, _bcsr_from_elt, None)
