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
from __future__ import annotations

import operator

from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from jax import core
from jax import lax
from jax import tree_util
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse import bcoo
from jax.experimental.sparse.util import (
    nfold_vmap, _count_stored_elements,
    _csr_to_coo, _dot_general_validated_shape,
    SparseInfo, Shape)
import jax.numpy as jnp
from jax._src import api_util
from jax._src.lax.lax import DotDimensionNumbers
from jax.util import split_list, safe_zip
from jax.interpreters import batching
from jax.interpreters import mlir
from jax._src.typing import Array, ArrayLike, DTypeLike


class BCSRProperties(NamedTuple):
  n_batch: int
  n_dense: int
  nse: int


def _compatible(shape1: Sequence[int], shape2: Sequence[int]) -> bool:
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
    raise ValueError(f"indices batch dimensions not compatible for {indices.shape=}, {shape=}")
  if not _compatible(indptr.shape[:n_batch], shape[:n_batch]):
    raise ValueError(f"indptr batch dimensions not compatible for {indptr.shape=}, {shape=}")
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
    raise ValueError(f"data batch dimensions not compatible for {data.shape=}, {shape=}")
  if data.shape[-(n_dense + 1):] != (nse,) + shape[n_batch + 2:]:
    raise ValueError(f"Invalid {data.shape=} for {nse=}, {n_batch=}, {n_dense=}")
  return props


def _bcsr_to_bcoo(indices: jnp.ndarray, indptr: jnp.ndarray, *,
                  shape: Sequence[int]) -> jnp.ndarray:
  """Given BCSR (indices, indptr), return BCOO (indices)."""
  n_batch, _, _ = _validate_bcsr_indices(indices, indptr, shape)
  csr_to_coo = nfold_vmap(_csr_to_coo, n_batch)
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


def bcsr_fromdense(mat: ArrayLike, *, nse: Optional[int] = None, n_batch: int = 0,
                   n_dense:int = 0, index_dtype: DTypeLike = jnp.int32) -> BCSR:
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
  nse_int: int = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return BCSR(_bcsr_fromdense(mat, nse=nse_int, n_batch=n_batch, n_dense=n_dense,
                              index_dtype=index_dtype),
              shape=mat.shape)


def _bcsr_fromdense(mat: ArrayLike, *, nse: int, n_batch: int = 0, n_dense: int = 0,
                    index_dtype: DTypeLike = jnp.int32) -> Tuple[Array, Array, Array]:
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
def _bcsr_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
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
    raise NotImplementedError(f"{batch_dims=}")
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


def bcsr_todense(mat: BCSR) -> Array:
  """Convert batched sparse matrix to a dense matrix.

  Args:
    mat: BCSR matrix.

  Returns:
    The dense version of ``mat``.
  """
  return _bcsr_todense(mat.data, mat.indices, mat.indptr, spinfo=mat._info)


def _bcsr_todense(data: ArrayLike, indices: ArrayLike, indptr: ArrayLike, *,
                  spinfo: SparseInfo) -> Array:
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + dense_dims``.
    indices : array of shape ``batch_dims + (nse,)``.
    indptr : array of shape ``batch_dims + (shape[len(batch_dims)] + 1,).
    spinfo : SparseInfo. In particular, this includes the shape
      of the matrix, which is equal to
      ``batch_dims + 2(sparse_dims) + block_dims`` where
      ``len(sparse_dims) == 2``.
  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcsr_todense_p.bind(jnp.asarray(data), jnp.asarray(indices),
                             jnp.asarray(indptr), spinfo=spinfo)


@bcsr_todense_p.def_impl
def _bcsr_todense_impl(data, indices, indptr, *, spinfo):
  shape = spinfo.shape
  bcoo_indices = _bcsr_to_bcoo(indices, indptr, shape=shape)
  return (bcoo.BCOO((data, bcoo_indices), shape=shape)).todense()


@bcsr_todense_p.def_abstract_eval
def _bcsr_todense_abstract_eval(data, indices, indptr, *, spinfo):
  shape = spinfo.shape
  _validate_bcsr(data, indices, indptr, shape)
  return core.ShapedArray(shape, data.dtype)


def _bcsr_todense_batching_rule(batched_args, batch_dims, *, spinfo):
  data, indices, indptr = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"{batch_dims=}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  if batch_dims[2] is None:
    indptr = indptr[None, ...]
  return _bcsr_todense(data, indices, indptr, spinfo=spinfo), 0

batching.primitive_batchers[bcsr_todense_p] = _bcsr_todense_batching_rule
mlir.register_lowering(bcsr_todense_p, mlir.lower_fun(
    _bcsr_todense_impl, multiple_results=False))


#--------------------------------------------------------------------
# bcsr_extract
bcsr_extract_p = core.Primitive('bcsr_extract')


def bcsr_extract(indices: ArrayLike, indptr: ArrayLike, mat: ArrayLike) -> Array:
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
  return bcoo._bcoo_extract(bcoo_indices, mat)


@bcsr_extract_p.def_abstract_eval
def _bcsr_extract_abstract_eval(indices, indptr, mat):
  n_batch, n_dense, nse = _validate_bcsr_indices(indices, indptr, mat.shape)
  out_shape = mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]
  return core.ShapedArray(out_shape, mat.dtype)


mlir.register_lowering(bcsr_extract_p, mlir.lower_fun(
    _bcsr_extract_impl, multiple_results=False))


#----------------------------------------------------------------------
# bcsr_dot_general


bcsr_dot_general_p = core.Primitive('bcsr_dot_general')


def bcsr_dot_general(lhs: Union[BCSR, Array], rhs: Array, *,
                     dimension_numbers: DotDimensionNumbers,
                     precision: None = None,
                     preferred_element_type: None = None) -> Array:
  """A general contraction operation.

  Args:
    lhs: An ndarray or BCSR-format sparse array.
    rhs: An ndarray or BCSR-format sparse array..
    dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
      (lhs_batch_dims, rhs_batch_dims))`.
    precision: unused
    preferred_element_type: unused

  Returns:
    An ndarray or BCSR-format sparse array containing the result. If both inputs
    are sparse, the result will be sparse, of type BCSR. If either input is
    dense, the result will be dense, of type ndarray.
  """
  del precision, preferred_element_type  # unused
  if isinstance(rhs, (np.ndarray, jnp.ndarray)):
    if isinstance(lhs, (np.ndarray, jnp.ndarray)):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    if isinstance(lhs, BCSR):
      lhs_data, lhs_indices, lhs_indptr = lhs._bufs
      return _bcsr_dot_general(lhs_data, lhs_indices, lhs_indptr, rhs,
                               dimension_numbers=dimension_numbers,
                               lhs_spinfo=lhs._info)

  raise NotImplementedError("bcsr_dot_general currently implemented for BCSR "
                            "lhs and ndarray rhs.")


def _bcsr_dot_general(lhs_data: jnp.ndarray, lhs_indices: jnp.ndarray,
                      lhs_indptr: jnp.ndarray, rhs: Array, *,
                      dimension_numbers: DotDimensionNumbers,
                      lhs_spinfo: SparseInfo) -> Array:
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcsr_dot_general_p.bind(jnp.asarray(lhs_data),
                                 jnp.asarray(lhs_indices),
                                 jnp.asarray(lhs_indptr), jnp.asarray(rhs),
                                 dimension_numbers=(cdims, bdims),
                                 lhs_spinfo=lhs_spinfo)


@bcsr_dot_general_p.def_impl
def _bcsr_dot_general_impl(lhs_data, lhs_indices, lhs_indptr, rhs, *,
                           dimension_numbers, lhs_spinfo):
  lhs_data = jnp.asarray(lhs_data)
  lhs_bcsr_indices = jnp.asarray(lhs_indices)
  lhs_bcsr_indptr = jnp.asarray(lhs_indptr)
  rhs = jnp.asarray(rhs)
  lhs_bcoo_indices = _bcsr_to_bcoo(lhs_bcsr_indices, lhs_bcsr_indptr,
                                   shape=lhs_spinfo.shape)
  return bcoo._bcoo_dot_general_impl(lhs_data, lhs_bcoo_indices, rhs,
                                     dimension_numbers=dimension_numbers,
                                     lhs_spinfo=lhs_spinfo)


@bcsr_dot_general_p.def_abstract_eval
def _bcsr_dot_general_abstract_eval(lhs_data, lhs_indices, lhs_indptr, rhs, *,
                                    dimension_numbers, lhs_spinfo):
  if lhs_data.dtype != rhs.dtype:
    raise ValueError("bcsr_dot_general requires arguments to have matching "
                     f"dtypes; got lhs.dtype={lhs_data.dtype}, "
                     f"rhs.dtype={rhs.dtype}")

  (lhs_contracting, _), (lhs_batch, _) = dimension_numbers
  props = _validate_bcsr_indices(lhs_indices, lhs_indptr, lhs_spinfo.shape)
  out_shape = _dot_general_validated_shape(lhs_spinfo.shape, rhs.shape,
                                           dimension_numbers)

  if lhs_batch and max(lhs_batch) >= props.n_batch:
    raise NotImplementedError(
      "bcsr_dot_general batch dimensions must be among the batch dimensions in the sparse representtaion.\n"
      f"got {lhs_batch=}, {props.n_batch=}")

  # TODO: support contraction of dense dimensions?
  if any(d >= props.n_batch + 2 for d in lhs_contracting):
    raise NotImplementedError("bcsr_dot_general: contracting over dense dimensions.")

  return core.ShapedArray(out_shape, lhs_data.dtype)


# def _bcsr_dot_general_jvp_lhs(lhs_data_dot, lhs_data, lhs_indices, lhs_indptr,
#                               rhs, *, dimension_numbers, lhs_spinfo):
#   del lhs_data
#   return _bcsr_dot_general(lhs_data_dot, lhs_indices, lhs_indptr, rhs,
#                            dimension_numbers=dimension_numbers,
#                            lhs_spinfo=lhs_spinfo)


# def _bcsr_dot_general_jvp_rhs(rhs_dot, lhs_data, lhs_indices, lhs_indptr, rhs,
#                               *, dimension_numbers, lhs_spinfo):
#   del rhs
#   return _bcsr_dot_general(lhs_data, lhs_indices, lhs_indptr, rhs_dot,
#                            dimension_numbers=dimension_numbers,
#                            lhs_spinfo=lhs_spinfo)


# def _bcsr_dot_general_transpose(ct, lhs_data, lhs_indices, lhs_inptr, rhs, *,
#                                  dimension_numbers, lhs_spinfo):
#   lhs_bcoo_indices = _bcsr_to_bcoo(
#     lhs_indices, lhs_inptr, shape=lhs_spinfo.shape)
#   return bcoo._bcoo_dot_general_transpose(
#       ct, lhs_data, lhs_bcoo_indices, rhs, dimension_numbers=dimension_numbers,
#       lhs_spinfo=lhs_spinfo)


# def _bcsr_dot_general_batch_rule(batched_args, batch_dims, *,
#                                  dimension_numbers, lhs_spinfo):
#   lhs_data, lhs_indices, lhs_indptr, rhs = batched_args
#   lhs_bcoo_indices = _bcsr_to_bcoo(
#     lhs_indices, lhs_indptr, shape=lhs_spinfo.shape)
#   return bcoo._bcoo_dot_general_batch_rule(
#       (lhs_data, lhs_bcoo_indices, rhs), batch_dims,
#       dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)


# ad.defjvp(bcsr_dot_general_p, _bcsr_dot_general_jvp_lhs, None,
#           _bcsr_dot_general_jvp_rhs)
# ad.primitive_transposes[bcsr_dot_general_p] = _bcsr_dot_general_transpose
# batching.primitive_batchers[bcsr_dot_general_p] = _bcsr_dot_general_batch_rule


_bcsr_dot_general_default_lowering = mlir.lower_fun(
    _bcsr_dot_general_impl, multiple_results=False)
mlir.register_lowering(
    bcsr_dot_general_p, _bcsr_dot_general_default_lowering)


@tree_util.register_pytree_node_class
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
  _bufs = property(lambda self: (self.data, self.indices, self.indptr))
  _info = property(lambda self: SparseInfo(self.shape))

  @property
  def _sparse_shape(self):
    return tuple(self.shape[self.n_batch:self.n_batch + 2])

  def __init__(self, args, *, shape):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices, self.indptr = map(jnp.asarray, args)
    super().__init__(args, shape=shape)
    _validate_bcsr(self.data, self.indices, self.indptr, self.shape)

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
      extra = f", {nse=}"
      if n_batch: extra += f", {n_batch=}"
      if n_dense: extra += f", {n_dense=}"
      repr_ = f"{name}({dtype}{shape}{extra})"
    if isinstance(self.data, core.Tracer):
      repr_ = f"{type(self.data).__name__}[{repr_}]"
    return repr_

  def transpose(self, *args, **kwargs):
    raise NotImplementedError("Tranpose is not implemented.")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), self._info._asdict()

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = object.__new__(cls)
    obj.data, obj.indices, obj.indptr = children
    if aux_data.keys() != {'shape', 'indices_sorted', 'unique_indices'}:
      raise ValueError(f"BCSR.tree_unflatten: invalid {aux_data=}")
    obj.__dict__.update(**aux_data)
    return obj

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32', n_dense=0,
             n_batch=0, nse=0):
    """Create an empty BCSR instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if n_dense < 0 or n_batch < 0 or nse < 0:
      raise ValueError(f"Invalid inputs: {shape=}, {n_dense=}, {n_batch=}, {nse=}")
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
  if axis is None:
    return elt
  if axis > elt.n_batch:
    raise ValueError(f"BCSR: cannot add out_axis={axis} for BCSR array with "
                     f"n_batch={elt.n_batch}. BCSR batch axes must be a "
                     "contiguous block of leading dimensions.")
  return BCSR((cont(axis_size, elt.data, axis),
               cont(axis_size, elt.indices, axis),
               cont(axis_size, elt.indptr, axis)),
              shape=elt.shape[:axis] + (axis_size,) + elt.shape[axis:])

batching.register_vmappable(BCSR, int, int, _bcsr_to_elt, _bcsr_from_elt, None)
