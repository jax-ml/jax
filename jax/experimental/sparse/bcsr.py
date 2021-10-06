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
# limitations under the License.

"""BCOO (Bached coordinate format) matrix object and associated primitives."""
import functools
import operator
from typing import Any, Tuple

import numpy as np

from jax import core
from jax import tree_util
from jax import vmap
from jax.interpreters import xla
import jax.numpy as jnp
from jax.util import safe_zip
from . import bcoo, ops

Dtype = Any
Shape = Tuple[int, ...]

#----------------------------------------------------------------------
# BCSR primitives

def _validate_bcsr(data, indices, indptr, shape):
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  assert jnp.issubdtype(indptr.dtype, jnp.integer)
  shape = tuple(shape)

  nse = indices.shape[-1]
  n_batch = indices.ndim - 1
  n_dense = len(shape) - n_batch - 2
  assert n_dense >= 0

  def _compatible(shape1, shape2):
    return all(s1 in (1, s2) for s1, s2 in safe_zip(shape1, shape2))

  if data is not None:
    if not _compatible(data.shape[:n_batch], shape[:n_batch]):
      raise ValueError("data batch dimensions not compatible for "
                      f"data.shape={data.shape}, shape={shape}")
    if data.shape[-(n_dense + 1):] != (nse,) + shape[n_batch + 2:]:
      raise ValueError(f"Invalid data.shape={data.shape} for "
                      f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  if not _compatible(indices.shape[:n_batch], shape[:n_batch]):
    raise ValueError("indices batch dimensions not compatible for "
                     f"indices.shape={indices.shape}, shape={shape}")
  if not _compatible(indptr.shape[:n_batch], shape[:n_batch]):
    raise ValueError("indptr batch dimensions not compatible for "
                     f"indptr.shape={indptr.shape}, shape={shape}")
  if indptr.shape[n_batch:] != (shape[n_batch] + 1,):
    raise ValueError("indptr shape must match the matrix shape plus 1.")

  return n_batch, n_dense


def _bcsr_to_bcoo(indices, indptr):
  assert indices.ndim == indptr.ndim
  n_batch = indices.ndim - 1
  def csr_to_coo(indptr, nse=indices.shape[-1]):
    return jnp.cumsum(jnp.zeros_like(indptr, shape=nse).at[indptr].add(1)) - 1
  for i in range(n_batch):
    csr_to_coo = vmap(csr_to_coo)
  return jnp.stack([csr_to_coo(indptr), indices], axis=indices.ndim)

# Note: no general _bcoo_to_bcsr here because BSCR requires sorted indices.

#----------------------------------------------------------------------
# bcsr_fromdense

bcsr_fromdense_p = core.Primitive('bcsr_fromdense')
bcsr_fromdense_p.multiple_results = True

def bcsr_fromdense(mat, *, nse=None, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create CSR-format sparse matrix from a dense matrix.

  Arguments must satisfy ``n_batch + n_dense + 2 == mat.ndim``

  Args:
    mat : array to be converted to COO, with ``ndim = n_batch + n_sparse + n_dense``.
    nse : number of specified elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of block_dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    data : array of shape ``mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]``
      and dtype ``mat.dtype``
    indices : array of shape ``mat.shape[:n_batch] + (nse,)``
      and dtype ``index_dtype``
    indptr : array of shape ``mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)``
      and dtype ``index_dtype``
  """
  mat = jnp.asarray(mat)
  if mat.ndim != n_batch + n_dense + 2:
    raise ValueError(f"bcsr_fromdense: expected 2 sparse dimensions, got {mat.ndim - n_batch - n_dense}")
  if nse is None:
    nse = bcoo._bcoo_nse(mat, n_batch, n_dense)
  nse = core.concrete_or_error(operator.index, nse, "nse argument of bcsr_fromdense")
  return bcsr_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)

@bcsr_fromdense_p.def_impl
def _bcsr_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  data, bcoo_indices, true_nonzeros = bcoo._bcoo_fromdense_helper(mat,
      nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)

  row_indices = bcoo_indices[..., 0]
  indices = bcoo_indices[..., 1]
  m = mat.shape[n_batch]

  get_ptr = lambda i: jnp.cumsum(jnp.bincount(i, length=m))
  for i in range(n_batch):
    get_ptr = vmap(get_ptr)
  indptr = jnp.zeros((*mat.shape[:n_batch], m + 1), index_dtype)
  indptr = indptr.at[..., 1:].set(get_ptr(row_indices) - nse + true_nonzeros.sum(-1, keepdims=True))
  return data, indices, indptr


@bcsr_fromdense_p.def_abstract_eval
def _bcsr_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
  n_sparse = mat.ndim - n_batch - n_dense
  if n_sparse != 2:
    raise ValueError("bcsr_fromdense: must have 2 sparse dimensions.")
  data_shape = mat.shape[:n_batch] + (nse,) + mat.shape[n_batch + n_sparse:]
  index_shape = mat.shape[:n_batch] + (nse,)
  indptr_shape = mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)
  return core.ShapedArray(data_shape, mat.dtype), core.ShapedArray(index_shape, index_dtype), core.ShapedArray(indptr_shape, index_dtype)

# def _bcsr_fromdense_jvp(primals, tangents, *, nse, n_batch, n_dense, index_dtype):
#   M, = primals
#   Mdot, = tangents

#   primals_out = bcsr_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)
#   data, indices, indptr = primals_out

#   if type(Mdot) is ad.Zero:
#     data_dot = ad.Zero.from_value(data)
#   else:
#     data_dot = bcsr_extract(indices, Mdot)

#   tangents_out = (data_dot, ad.Zero.from_value(indices))

#   return primals_out, tangents_out

# def _bcsr_fromdense_transpose(ct, M, *, nse, n_batch, n_dense, index_dtype):
#   data, indices = ct
#   n_sparse = M.ndim = n_batch - n_dense
#   assert data.shape == M.shape[:n_batch] + (nse,) + M.shape[n_batch + n_sparse:]
#   assert indices.shape == M.shape[:n_batch] + (n_sparse, nse)
#   assert indices.dtype == index_dtype
#   if isinstance(indices, ad.Zero):
#     raise ValueError("Cannot transpose with respect to sparse indices")
#   assert ad.is_undefined_primal(M)
#   return bcsr_todense(data, indices, shape=M.aval.shape)

# def _bcsr_fromdense_batching_rule(batched_args, batch_dims, *, nse, n_batch, n_dense, index_dtype):
#   M, = batched_args
#   if batch_dims != (0,):
#     raise NotImplementedError(f"batch_dims={batch_dims}")
#   return bcsr_fromdense(M, nse=nse, n_batch=n_batch + 1, n_dense=n_dense, index_dtype=index_dtype), (0, 0)

# ad.primitive_jvps[bcsr_fromdense_p] = _bcsr_fromdense_jvp
# ad.primitive_transposes[bcsr_fromdense_p] = _bcsr_fromdense_transpose
# batching.primitive_batchers[bcsr_fromdense_p] = _bcsr_fromdense_batching_rule
xla.translations[bcsr_fromdense_p] = xla.lower_fun(
    _bcsr_fromdense_impl, multiple_results=True)


#----------------------------------------------------------------------
# bcsr_todense

bcsr_todense_p = core.Primitive('bcsr_todense_p')

def bcsr_todense(data, indices, indptr, *, shape):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + block_dims``.
    indices : array of shape ``batch_dims + (nse,)``
    indptr : array of shape ``batch_dims + (shape[len(batch_dims)] + 1,)
    shape : tuple; the shape of the (batched) matrix. Equal to
      ``batch_dims + 2 + block_dims``

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcsr_todense_p.bind(jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr), shape=tuple(shape))

@bcsr_todense_p.def_impl
def _bcsr_todense_impl(data, indices, indptr, *, shape):
  n_batch, _ = _validate_bcsr(data, indices, indptr, shape)
  csr_to_coo = functools.partial(ops._csr_to_coo, nse=indices.shape[-1])
  for i in range(n_batch):
    csr_to_coo = vmap(csr_to_coo)
  bcoo_indices = jnp.stack([csr_to_coo(indptr), indices], axis=indices.ndim)
  return bcoo.bcoo_todense(data, bcoo_indices, shape=shape)

@bcsr_todense_p.def_abstract_eval
def _bcsr_todense_abstract_eval(data, indices, indptr, *, shape):
  _validate_bcsr(data, indices, indptr, shape)
  return core.ShapedArray(shape, data.dtype)

# def _bcsr_todense_jvp(data_dot, data, indices, *, shape):
#   return bcsr_todense(data_dot, indices, shape=shape)

# def _bcsr_todense_transpose(ct, data, indices, *, shape):
#   assert ad.is_undefined_primal(data)
#   if ad.is_undefined_primal(indices):
#     raise ValueError("Cannot transpose with respect to sparse indices")
#   assert ct.shape == shape
#   assert ct.dtype == data.aval.dtype
#   return bcsr_extract(indices, ct), indices

# def _bcsr_todense_batching_rule(batched_args, batch_dims, *, shape):
#   data, indices = batched_args
#   if any(b not in [0, None] for b in batch_dims):
#     raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
#   if batch_dims[0] is None:
#     data = data[None, ...]
#   if batch_dims[1] is None:
#     indices = indices[None, ...]
#   return bcsr_todense(data, indices, shape=(max(data.shape[0], indices.shape[0]), *shape)), 0

# ad.defjvp(bcsr_todense_p, _bcsr_todense_jvp, None)
# ad.primitive_transposes[bcsr_todense_p] = _bcsr_todense_transpose
# batching.primitive_batchers[bcsr_todense_p] = _bcsr_todense_batching_rule
xla.translations[bcsr_todense_p] = xla.lower_fun(
    _bcsr_todense_impl, multiple_results=False)


#----------------------------------------------------------------------
# bcsr_extract

bcsr_extract_p = core.Primitive('bcsr_extract')

def bcsr_extract(indices, indptr, mat):
  """Extract BCOO values from dense matrix `mat` at given BCOO indices."""
  return bcsr_extract_p.bind(indices, indptr, mat)

@bcsr_extract_p.def_impl
def _bcsr_extract_impl(indices, indptr, mat):
  _validate_bcsr(None, indices, indptr, mat.shape)
  coo_indices = _bcsr_to_bcoo(indices, indptr)
  return bcoo.bcoo_extract(coo_indices, mat)

@bcsr_extract_p.def_abstract_eval
def _bcsr_extract_abstract_eval(indices, indptr, mat):
  n_batch, n_dense = _validate_bcsr(None, indices, indptr,  mat.shape)
  nse = indices.shape[-1]
  out_shape = mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]
  return core.ShapedArray(out_shape, mat.dtype)

# def _bcsr_extract_jvp(mat_dot, indices, mat):
#   assert mat_dot.shape == mat.shape
#   return bcsr_extract(indices, mat_dot)

# def _bcsr_extract_transpose(ct, indices, mat):
#   assert ad.is_undefined_primal(mat)
#   if ad.is_undefined_primal(indices):
#     raise ValueError("Cannot transpose with respect to sparse indices")
#   assert ct.dtype == mat.aval.dtype
#   return indices, bcsr_todense(ct, indices, shape=mat.aval.shape)

# def _bcsr_extract_batching_rule(batched_args, batch_dims):
#   indices, mat = batched_args
#   assert any(b is not None for b in batch_dims)
#   if batch_dims[0] is None:
#     bdim = batch_dims[1]
#     indices = lax.expand_dims(indices, (bdim,))
#   elif batch_dims[1] is None:
#     bdim = batch_dims[0]
#     mat = lax.expand_dims(mat, (bdim,))
#   else:
#     assert batch_dims[0] == batch_dims[1]
#     bdim = batch_dims[0]
#   n_batch = indices.ndim - 2
#   if bdim >= n_batch:
#     raise ValueError(f"batch_dims={batch_dims} out of range for indices with n_batch={n_batch}")
#   return bcsr_extract(indices, mat), bdim

# ad.defjvp(bcsr_extract_p, None, _bcsr_extract_jvp)
# ad.primitive_transposes[bcsr_extract_p] = _bcsr_extract_transpose
# batching.primitive_batchers[bcsr_extract_p] = _bcsr_extract_batching_rule
xla.translations[bcsr_extract_p] = xla.lower_fun(
    _bcsr_extract_impl, multiple_results=False)

@tree_util.register_pytree_node_class
class BCSR(ops.JAXSparse):
  """Experimental batched CSR matrix"""
  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  nse = property(lambda self: self.indices.shape[-1])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 1)
  n_sparse = property(lambda _: 2)
  n_dense = property(lambda self: self.data.ndim - self.indices.ndim)
  shape = Tuple[int, ...]

  @property
  def _sparse_shape(self):
    return tuple(self.shape[self.n_batch:self.n_batch + 2])

  def __init__(self, args, *, shape):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices, self.indptr = args if bcoo._is_placeholder(*args) else map(bcoo._asarray_or_float0, args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32, n_dense=0, n_batch=0):
    """Create a BCSR array from a (dense) :class:`DeviceArray`."""
    return cls(bcsr_fromdense(mat, nse=nse, index_dtype=index_dtype, n_dense=n_dense, n_batch=n_batch), shape=mat.shape)

  # @jax.jit
  def todense(self):
    """Create a dense version of the array."""
    return bcsr_todense(self.data, self.indices, self.indptr, shape=self.shape)

  def tree_flatten(self):
    children = (self.data, self.indices, self.indptr)
    # pytree sometimes creates placeholder objects & we need to handle that.
    sparse_shape = self.shape if bcoo._is_placeholder(*children) else self._sparse_shape
    # We serialize the sparse shape only to support batching.
    return children, {"sparse_shape": sparse_shape}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    data, indices, indptr = children
    sparse_shape = aux_data["sparse_shape"]
    # pytree sometimes creates placeholder objects & we need to handle that.
    if bcoo._is_placeholder(data, indices):
      shape = sparse_shape
    else:
      if np.ndim(indices) < 1 or len(sparse_shape) != 2:
        raise ValueError(f"Invalid sparse representation: got indices.shape={np.shape(indices)}, "
                         f"data.shape={np.shape(data)}, sparse_shape={sparse_shape}")
      n_batch = indices.ndim - 1
      shape = (
          tuple(np.maximum(data.shape[:n_batch], indices.shape[:n_batch]))
          + tuple(sparse_shape)
          + tuple(data.shape[n_batch + 1:]))
    return cls(children, shape=shape)
