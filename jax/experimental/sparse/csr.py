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

"""CSR (compressed sparse row) matrix object and associated primitives."""

import operator
from typing import Tuple
import warnings

import numpy as np

from jax import core
from jax.interpreters import ad
from jax.interpreters import xla
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.coo import _coo_matmat_impl, _coo_matvec_impl, _coo_todense_impl
from jax.experimental.sparse.util import _csr_to_coo, _csr_extract, _safe_asarray, CuSparseEfficiencyWarning
from jax import tree_util
from jax._src.lib import cusparse
from jax._src.numpy.lax_numpy import _promote_dtypes
import jax.numpy as jnp


@tree_util.register_pytree_node_class
class CSR(JAXSparse):
  """Experimental CSR matrix implemented in JAX; API subject to change."""
  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  shape: Tuple[int, int]
  nse = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)

  def __init__(self, args, *, shape):
    self.data, self.indices, self.indptr = _safe_asarray(args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
    if nse is None:
      nse = (mat != 0).sum()
    return cls(csr_fromdense(mat, nse=nse, index_dtype=index_dtype), shape=mat.shape)

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
    """Create an empty CSR instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if len(shape) != 2:
      raise ValueError(f"CSR must have ndim=2; got shape={shape}")
    data = jnp.empty(0, dtype)
    indices = jnp.empty(0, index_dtype)
    indptr = jnp.zeros(shape[0] + 1, index_dtype)
    return cls((data, indices, indptr), shape=shape)

  def todense(self):
    return csr_todense(self.data, self.indices, self.indptr, shape=self.shape)

  def transpose(self, axes=None):
    assert axes is None
    return CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])

  def __matmul__(self, other):
    if isinstance(other, JAXSparse):
      raise NotImplementedError("matmul between two sparse objects.")
    other = jnp.asarray(other)
    data, other = _promote_dtypes(self.data, other)
    if other.ndim == 1:
      return csr_matvec(data, self.indices, self.indptr, other, shape=self.shape)
    elif other.ndim == 2:
      return csr_matmat(data, self.indices, self.indptr, other, shape=self.shape)
    else:
      raise NotImplementedError(f"matmul with object of shape {other.shape}")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {"shape": self.shape}


@tree_util.register_pytree_node_class
class CSC(JAXSparse):
  """Experimental CSC matrix implemented in JAX; API subject to change."""
  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  shape: Tuple[int, int]
  nse = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)

  def __init__(self, args, *, shape):
    self.data, self.indices, self.indptr = _safe_asarray(args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
    if nse is None:
      nse = (mat != 0).sum()
    return cls(csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype), shape=mat.shape)

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
    """Create an empty CSC instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if len(shape) != 2:
      raise ValueError(f"CSC must have ndim=2; got shape={shape}")
    data = jnp.empty(0, dtype)
    indices = jnp.empty(0, index_dtype)
    indptr = jnp.zeros(shape[1] + 1, index_dtype)
    return cls((data, indices, indptr), shape=shape)

  def todense(self):
    return csr_todense(self.data, self.indices, self.indptr, shape=self.shape[::-1]).T

  def transpose(self, axes=None):
    assert axes is None
    return CSR((self.data, self.indices, self.indptr), shape=self.shape[::-1])

  def __matmul__(self, other):
    if isinstance(other, JAXSparse):
      raise NotImplementedError("matmul between two sparse objects.")
    other = jnp.asarray(other)
    data, other = _promote_dtypes(self.data, other)
    if other.ndim == 1:
      return csr_matvec(data, self.indices, self.indptr, other, shape=self.shape[::-1], transpose=True)
    elif other.ndim == 2:
      return csr_matmat(data, self.indices, self.indptr, other, shape=self.shape[::-1], transpose=True)
    else:
      raise NotImplementedError(f"matmul with object of shape {other.shape}")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {"shape": self.shape}


#--------------------------------------------------------------------
# csr_todense

csr_todense_p = core.Primitive('csr_todense')

def csr_todense(data, indices, indptr, *, shape):
  """Convert CSR-format sparse matrix to a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    shape : length-2 tuple representing the matrix shape

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return csr_todense_p.bind(data, indices, indptr, shape=shape)

@csr_todense_p.def_impl
def _csr_todense_impl(data, indices, indptr, *, shape):
  return _coo_todense_impl(data, *_csr_to_coo(indices, indptr), shape=shape)

@csr_todense_p.def_abstract_eval
def _csr_todense_abstract_eval(data, indices, indptr, *, shape):
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert indices.dtype == indptr.dtype
  assert data.shape == indices.shape
  assert indptr.shape[0] == shape[0] + 1
  return core.ShapedArray(shape, data.dtype)

_csr_todense_translation_rule = xla.lower_fun(
    _csr_todense_impl, multiple_results=False, new_style=True)

def _csr_todense_gpu_translation_rule(ctx, avals_in, avals_out, data, indices,
                                      indptr, *, shape):
  dtype = avals_in[0].dtype
  if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
    warnings.warn(f"csr_todense cusparse lowering not available for dtype={dtype}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_todense_translation_rule(ctx, avals_in, avals_out, data, indices,
                                         indptr, shape=shape)
  return [cusparse.csr_todense(ctx.builder, data, indices, indptr, shape=shape)]

def _csr_todense_jvp(data_dot, data, indices, indptr, *, shape):
  return csr_todense(data_dot, indices, indptr, shape=shape)

def _csr_todense_transpose(ct, data, indices, indptr, *, shape):
  # Note: we assume that transpose has the same sparsity pattern.
  # Can we check this?
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert indices.aval.dtype == indptr.aval.dtype
  assert ct.dtype == data.aval.dtype
  return _csr_extract(indices, indptr, ct), indices, indptr

ad.defjvp(csr_todense_p, _csr_todense_jvp, None, None)
ad.primitive_transposes[csr_todense_p] = _csr_todense_transpose
xla.register_translation(csr_todense_p, _csr_todense_translation_rule)
if cusparse and cusparse.is_supported:
  xla.register_translation(csr_todense_p, _csr_todense_gpu_translation_rule,
                           platform='gpu')

#--------------------------------------------------------------------
# csr_fromdense

csr_fromdense_p = core.Primitive('csr_fromdense')
csr_fromdense_p.multiple_results = True

def csr_fromdense(mat, *, nse, index_dtype=np.int32):
  """Create CSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to CSR.
    nse : number of specified entries in ``mat``
    index_dtype : dtype of sparse indices

  Returns:
    data : array of shape ``(nse,)`` and dtype ``mat.dtype``.
    indices : array of shape ``(nse,)`` and dtype ``index_dtype``
    indptr : array of shape ``(mat.shape[0] + 1,)`` and dtype ``index_dtype``
  """
  mat = jnp.asarray(mat)
  nse = core.concrete_or_error(operator.index, nse, "nse argument of csr_fromdense()")
  return csr_fromdense_p.bind(mat, nse=nse, index_dtype=np.dtype(index_dtype))

@csr_fromdense_p.def_impl
def _csr_fromdense_impl(mat, *, nse, index_dtype):
  mat = jnp.asarray(mat)
  assert mat.ndim == 2
  m = mat.shape[0]

  row, col = jnp.nonzero(mat, size=nse)
  data = mat[row, col]

  true_nonzeros = jnp.arange(nse) < (mat != 0).sum()
  data = jnp.where(true_nonzeros, data, 0)
  row = jnp.where(true_nonzeros, row, m)
  indices = col.astype(index_dtype)
  indptr = jnp.zeros(m + 1, dtype=index_dtype).at[1:].set(
      jnp.cumsum(jnp.bincount(row, length=m)))
  return data, indices, indptr

@csr_fromdense_p.def_abstract_eval
def _csr_fromdense_abstract_eval(mat, *, nse, index_dtype):
  data = core.ShapedArray((nse,), mat.dtype)
  indices = core.ShapedArray((nse,), index_dtype)
  indptr = core.ShapedArray((mat.shape[0] + 1,), index_dtype)
  return data, indices, indptr

_csr_fromdense_translation_rule = xla.lower_fun(
    _csr_fromdense_impl, multiple_results=True, new_style=True)

def _csr_fromdense_gpu_translation_rule(ctx, avals_in, avals_out, mat, *, nse,
                                        index_dtype):
  dtype = avals_in[0].dtype
  if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
    warnings.warn(f"csr_fromdense cusparse lowering not available for dtype={dtype}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_fromdense_translation_rule(ctx, avals_in, avals_out, mat,
                                           nse=nse, index_dtype=index_dtype)
  data, indices, indptr = cusparse.csr_fromdense(
      ctx.builder, mat, nnz=nse, index_dtype=np.dtype(index_dtype))
  return [data, indices, indptr]

def _csr_fromdense_jvp(primals, tangents, *, nse, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = csr_fromdense(M, nse=nse, index_dtype=index_dtype)
  data, indices, indptr = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_value(data)
  else:
    data_dot = _csr_extract(indices, indptr, Mdot)

  tangents_out = (data_dot, ad.Zero.from_value(indices), ad.Zero.from_value(indptr))

  return primals_out, tangents_out

def _csr_fromdense_transpose(ct, M, *, nse, index_dtype):
  data, indices, indptr = ct
  assert len(data) == nse
  assert indices.dtype == indptr.dtype == index_dtype
  if isinstance(indices, ad.Zero) or isinstance(indptr, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return csr_todense(data, indices, indptr, shape=M.aval.shape)

ad.primitive_jvps[csr_fromdense_p] = _csr_fromdense_jvp
ad.primitive_transposes[csr_fromdense_p] = _csr_fromdense_transpose
xla.register_translation(csr_fromdense_p, _csr_fromdense_translation_rule)
if cusparse and cusparse.is_supported:
  xla.register_translation(csr_fromdense_p,
                           _csr_fromdense_gpu_translation_rule,
                           platform='gpu')

#--------------------------------------------------------------------
# csr_matvec

csr_matvec_p = core.Primitive('csr_matvec')

def csr_matvec(data, indices, indptr, v, *, shape, transpose=False):
  """Product of CSR sparse matrix and a dense vector.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    v : array of shape ``(shape[0] if transpose else shape[1],)``
      and dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
  """
  return csr_matvec_p.bind(data, indices, indptr, v, shape=shape, transpose=transpose)

@csr_matvec_p.def_impl
def _csr_matvec_impl(data, indices, indptr, v, *, shape, transpose):
  return _coo_matvec_impl(data, *_csr_to_coo(indices, indptr), v, shape=shape, transpose=transpose)

@csr_matvec_p.def_abstract_eval
def _csr_matvec_abstract_eval(data, indices, indptr, v, *, shape, transpose):
  assert len(shape) == 2
  assert v.ndim == data.ndim == indices.ndim == indptr.ndim == 1
  assert data.shape == indices.shape
  assert data.dtype == v.dtype
  assert indices.dtype == indptr.dtype
  assert indptr.shape[0] == shape[0] + 1
  out_shape = shape[1] if transpose else shape[0]
  assert v.shape[0] == (shape[0] if transpose else shape[1])
  return core.ShapedArray((out_shape,), data.dtype)

_csr_matvec_translation_rule = xla.lower_fun(
    _csr_matvec_impl, multiple_results=False, new_style=True)

def _csr_matvec_gpu_translation_rule(ctx, avals_in, avals_out, data, indices,
                                     indptr, v, *, shape, transpose):
  dtype = avals_in[0].dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"csr_matvec cusparse lowering not available for dtype={dtype}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_matvec_translation_rule(ctx, avals_in, avals_out, data, indices, indptr, v,
                                        shape=shape, transpose=transpose)
  return [cusparse.csr_matvec(ctx.builder, data, indices, indptr, v,
                              shape=shape, transpose=transpose)]

def _csr_matvec_jvp_mat(data_dot, data, indices, indptr, v, *, shape, transpose):
  return csr_matvec(data_dot, indices, indptr, v, shape=shape, transpose=transpose)

def _csr_matvec_jvp_vec(v_dot, data, indices, indptr, v, *, shape, transpose):
  return csr_matvec(data, indices, indptr, v_dot, shape=shape, transpose=transpose)

def _csr_matvec_transpose(ct, data, indices, indptr, v, *, shape, transpose):
  assert not ad.is_undefined_primal(indices)
  assert not ad.is_undefined_primal(indptr)

  if ad.is_undefined_primal(v):
    return data, indices, indptr, csr_matvec(data, indices, indptr, ct, shape=shape, transpose=not transpose)
  else:
    v = jnp.asarray(v)
    # The following lines do this, but more efficiently.
    # return _csr_extract(indices, indptr, jnp.outer(ct, v)), indices, indptr, v
    row, col = _csr_to_coo(indices, indptr)
    return ct[row] * v[col], indices, indptr, v

ad.defjvp(csr_matvec_p, _csr_matvec_jvp_mat, None, None, _csr_matvec_jvp_vec)
ad.primitive_transposes[csr_matvec_p] = _csr_matvec_transpose
xla.register_translation(csr_matvec_p, _csr_matvec_translation_rule)
if cusparse and cusparse.is_supported:
  xla.register_translation(csr_matvec_p, _csr_matvec_gpu_translation_rule,
                           platform='gpu')


#--------------------------------------------------------------------
# csr_matmat

csr_matmat_p = core.Primitive('csr_matmat')

def csr_matmat(data, indices, indptr, B, *, shape, transpose=False):
  """Product of CSR sparse matrix and a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(shape[1] if transpose else shape[0], cols)``
      representing the matrix-matrix product product.
  """
  return csr_matmat_p.bind(data, indices, indptr, B, shape=shape, transpose=transpose)

@csr_matmat_p.def_impl
def _csr_matmat_impl(data, indices, indptr, B, *, shape, transpose):
  return _coo_matmat_impl(data, *_csr_to_coo(indices, indptr), B, shape=shape, transpose=transpose)

@csr_matmat_p.def_abstract_eval
def _csr_matmat_abstract_eval(data, indices, indptr, B, *, shape, transpose):
  assert len(shape) == 2
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert B.ndim == 2
  assert data.shape == indices.shape
  assert data.dtype == B.dtype
  assert indices.dtype == indptr.dtype
  assert indptr.shape[0] == shape[0] + 1
  out_shape = shape[1] if transpose else shape[0]
  assert B.shape[0] == (shape[0] if transpose else shape[1])
  return core.ShapedArray((out_shape, B.shape[1]), data.dtype)

_csr_matmat_translation_rule = xla.lower_fun(
    _csr_matmat_impl, multiple_results=False, new_style=True)

def _csr_matmat_gpu_translation_rule(ctx, avals_in, avals_out, data, indices,
                                     indptr, B, *, shape, transpose):
  dtype = avals_in[0].dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"csr_matmat cusparse lowering not available for dtype={dtype}. "
                  "Falling back to default implementation.", CuSparseEfficiencyWarning)
    return _csr_matmat_translation_rule(ctx, avals_in, avals_out, data, indices, indptr, B,
                                        shape=shape, transpose=transpose)
  return [cusparse.csr_matmat(ctx.builder, data, indices, indptr, B,
                              shape=shape, transpose=transpose)]

def _csr_matmat_jvp_left(data_dot, data, indices, indptr, B, *, shape, transpose):
  return csr_matmat(data_dot, indices, indptr, B, shape=shape, transpose=transpose)

def _csr_matmat_jvp_right(B_dot, data, indices, indptr, B, *, shape, transpose):
  return csr_matmat(data, indices, indptr, B_dot, shape=shape, transpose=transpose)

def _csr_matmat_transpose(ct, data, indices, indptr, B, *, shape, transpose):
  assert not ad.is_undefined_primal(indices)
  assert not ad.is_undefined_primal(indptr)

  if ad.is_undefined_primal(B):
    return data, indices, indptr, csr_matmat(data, indices, indptr, ct, shape=shape, transpose=not transpose)
  else:
    B = jnp.asarray(B)
    row, col = _csr_to_coo(indices, indptr)
    return (ct[row] * B[col]).sum(1), indices, indptr, B

ad.defjvp(csr_matmat_p, _csr_matmat_jvp_left, None, None, _csr_matmat_jvp_right)
ad.primitive_transposes[csr_matmat_p] = _csr_matmat_transpose
xla.register_translation(csr_matmat_p, _csr_matmat_translation_rule)
if cusparse and cusparse.is_supported:
  xla.register_translation(csr_matmat_p, _csr_matmat_gpu_translation_rule,
                           platform='gpu')
