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

"""JAX primitives related to sparse operations.

This is experimental work to explore sparse support in JAX.

The primitives defined here are deliberately low-level: i.e. for now there is
no JAX CSR or COO matrix class. Each primitive implements a common sparse
operation (sparse to dense, dense to sparse, sparse matrix/vector product,
sparse matrix/matrix product) for two common sparse representations
(CSR and COO).

These routines have reference implementations defined via XLA scatter/gather
operations that will work on any backend, although they are not particularly
performant. On GPU runtimes with jaxlib 0.1.66 or newer built against CUDA 11.0
or newer, each operation is computed efficiently via cusparse.
"""

from jax import core
from jax.interpreters import xla
from jax.lib import cusparse
from jax.lib import xla_bridge
from jax.lib import xla_client
import jax.numpy as jnp
import numpy as np

xb = xla_bridge
xops = xla_client.ops

#--------------------------------------------------------------------
# csr_todense

csr_todense_p = core.Primitive('csr_todense')

def csr_todense(data, indices, indptr, *, shape):
  """Convert CSR-format sparse matrix to a dense matrix.

  Args:
    data : array of shape ``(nnz,)``.
    indices : array of shape ``(nnz,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    shape : length-2 tuple representing the matrix shape

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return csr_todense_p.bind(data, indices, indptr, shape=shape)

@csr_todense_p.def_impl
def _csr_todense_impl(data, indices, indptr, *, shape):
  row = jnp.zeros_like(indices).at[indptr].add(1).cumsum() - 1
  return _coo_todense_impl(data, row, indices, shape=shape)

@csr_todense_p.def_abstract_eval
def _csr_todense_abstract_eval(data, indices, indptr, *, shape):
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert indices.dtype == indptr.dtype
  assert data.shape == indices.shape
  assert indptr.shape[0] == shape[0] + 1
  return core.ShapedArray(shape, data.dtype)

def _csr_todense_gpu_translation_rule(c, data, indices, indptr, *, shape):
  return cusparse.csr_todense(c, data, indices, indptr, shape=shape)

xla.translations[csr_todense_p] = xla.lower_fun(
    _csr_todense_impl, multiple_results=False)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      csr_todense_p] = _csr_todense_gpu_translation_rule

#--------------------------------------------------------------------
# csr_fromdense

csr_fromdense_p = core.Primitive('csr_fromdense')
csr_fromdense_p.multiple_results = True

def csr_fromdense(mat, *, nnz, index_dtype=np.int32):
  """Create CSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to CSR.
    nnz : number of nonzero entries in ``mat``
    index_dtype : dtype of sparse indices

  Returns:
    data : array of shape ``(nnz,)`` and dtype ``mat.dtype``.
    indices : array of shape ``(nnz,)`` and dtype ``index_dtype``
    indptr : array of shape ``(mat.shape[0] + 1,)`` and dtype ``index_dtype``
  """
  return csr_fromdense_p.bind(
      mat,
      nnz=nnz,
      index_dtype=np.dtype(index_dtype))

@csr_fromdense_p.def_impl
def _csr_fromdense_impl(mat, *, nnz, index_dtype):
  m = mat.shape[0]
  data, row, col = _coo_fromdense_impl(mat, nnz=nnz, index_dtype=index_dtype)
  indptr = jnp.zeros(m + 1, dtype=index_dtype).at[1:].set(jnp.cumsum(jnp.bincount(row, length=m)))
  return data, col, indptr

@csr_fromdense_p.def_abstract_eval
def _csr_fromdense_abstract_eval(mat, *, nnz, index_dtype):
  data = core.ShapedArray((nnz,), mat.dtype)
  indices = core.ShapedArray((nnz,), index_dtype)
  indptr = core.ShapedArray((mat.shape[0] + 1,), index_dtype)
  return data, indices, indptr

def _csr_fromdense_gpu_translation_rule(c, mat, *, nnz, index_dtype):
  data, indices, indptr = cusparse.csr_fromdense(
      c, mat, nnz=nnz, index_dtype=np.dtype(index_dtype))
  return xops.Tuple(c, [data, indices, indptr])

xla.translations[csr_fromdense_p] = xla.lower_fun(
    _csr_fromdense_impl, multiple_results=True)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      csr_fromdense_p] = _csr_fromdense_gpu_translation_rule

#--------------------------------------------------------------------
# csr_matvec

csr_matvec_p = core.Primitive('csr_matvec')

def csr_matvec(data, indices, indptr, v, *, shape, transpose=False):
  """Product of CSR sparse matrix and a dense vector.

  Args:
    data : array of shape ``(nnz,)``.
    indices : array of shape ``(nnz,)``
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
  row = jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1
  return _coo_matvec_impl(data, row, indices, v, shape=shape, transpose=transpose)

@csr_matvec_p.def_abstract_eval
def _csr_matvec_abstract_eval(data, indices, indptr, v, *, shape, transpose):
  assert len(shape) == 2
  assert v.ndim == data.ndim == indices.ndim == indptr.ndim == 1
  assert data.shape == indices.shape
  assert data.dtype == v.dtype
  assert indices.dtype == indptr.dtype
  assert len(indptr) == shape[0] + 1
  out_shape = shape[1] if transpose else shape[0]
  assert v.shape == (shape[0],) if transpose else (shape[1],)
  return core.ShapedArray((out_shape,), data.dtype)

def _csr_matvec_gpu_translation_rule(c, data, indices, indptr, v, *, shape, transpose):
  return cusparse.csr_matvec(c, data, indices, indptr, v, shape=shape, transpose=transpose)

xla.translations[csr_matvec_p] = xla.lower_fun(
    _csr_matvec_impl, multiple_results=False)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      csr_matvec_p] = _csr_matvec_gpu_translation_rule


#--------------------------------------------------------------------
# csr_matmat

csr_matmat_p = core.Primitive('csr_matmat')

def csr_matmat(data, indices, indptr, B, *, shape, transpose=False):
  """Product of CSR sparse matrix and a dense matrix.

  Args:
    data : array of shape ``(nnz,)``.
    indices : array of shape ``(nnz,)``
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
  row = jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1
  return _coo_matmat_impl(data, row, indices, B, shape=shape, transpose=transpose)

@csr_matmat_p.def_abstract_eval
def _csr_matmat_abstract_eval(data, indices, indptr, B, *, shape, transpose):
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert B.ndim == 2
  assert data.shape == indices.shape
  assert data.dtype == B.dtype
  assert indices.dtype == indptr.dtype
  assert len(indptr) == shape[0] + 1
  out_shape = shape[1] if transpose else shape[0]
  assert B.shape[0] == shape[0] if transpose else shape[1]
  return core.ShapedArray((out_shape, B.shape[1]), data.dtype)

def _csr_matmat_gpu_translation_rule(c, data, indices, indptr, B, *, shape, transpose):
  return cusparse.csr_matmat(c, data, indices, indptr, B, shape=shape, transpose=transpose)

xla.translations[csr_matmat_p] = xla.lower_fun(
    _csr_matmat_impl, multiple_results=False)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      csr_matmat_p] = _csr_matmat_gpu_translation_rule


#--------------------------------------------------------------------
# coo_todense

coo_todense_p = core.Primitive('coo_todense')

def coo_todense(data, row, col, *, shape):
  """Convert CSR-format sparse matrix to a dense matrix.

  Args:
    data : array of shape ``(nnz,)``.
    row : array of shape ``(nnz,)``
    col : array of shape ``(nnz,)`` and dtype ``row.dtype``
    shape : length-2 tuple representing the matrix shape

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return coo_todense_p.bind(data, row, col, shape=shape)

@coo_todense_p.def_impl
def _coo_todense_impl(data, row, col, *, shape):
  return jnp.zeros(shape, data.dtype).at[row, col].set(data)

@coo_todense_p.def_abstract_eval
def _coo_todense_abstract_eval(data, row, col, *, shape):
  return core.ShapedArray(shape, data.dtype)

def _coo_todense_gpu_translation_rule(c, data, row, col, *, shape):
  return cusparse.coo_todense(c, data, row, col, shape=shape)

xla.translations[coo_todense_p] = xla.lower_fun(
    _coo_todense_impl, multiple_results=False)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      coo_todense_p] = _coo_todense_gpu_translation_rule

#--------------------------------------------------------------------
# coo_fromdense

coo_fromdense_p = core.Primitive('coo_fromdense')
coo_fromdense_p.multiple_results = True

def coo_fromdense(mat, *, nnz, index_dtype=jnp.int32):
  """Create COO-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to COO.
    nnz : number of nonzero entries in ``mat``
    index_dtype : dtype of sparse indices

  Returns:
    data : array of shape ``(nnz,)`` and dtype ``mat.dtype``
    row : array of shape ``(nnz,)`` and dtype ``index_dtype``
    col : array of shape ``(nnz,)`` and dtype ``index_dtype``
  """
  return coo_fromdense_p.bind(mat, nnz=nnz, index_dtype=index_dtype)

@coo_fromdense_p.def_impl
def _coo_fromdense_impl(mat, *, nnz, index_dtype):
  mat = jnp.asarray(mat)
  m, n = mat.shape
  mat_flat = jnp.ravel(mat)
  ind = jnp.nonzero(mat_flat, size=nnz)[0].astype(index_dtype)
  return mat_flat[ind], ind // n, ind % n

@coo_fromdense_p.def_abstract_eval
def _coo_fromdense_abstract_eval(mat, *, nnz, index_dtype):
  data = core.ShapedArray((nnz,), mat.dtype)
  row = col = core.ShapedArray((nnz,), index_dtype)
  return data, row, col

def _coo_fromdense_gpu_translation_rule(c, mat, *, nnz, index_dtype):
  data, row, col = cusparse.coo_fromdense(
      c, mat, nnz=nnz, index_dtype=np.dtype(index_dtype))
  return xops.Tuple(c, [data, row, col])

xla.translations[coo_fromdense_p] = xla.lower_fun(
    _coo_fromdense_impl, multiple_results=True)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      coo_fromdense_p] = _coo_fromdense_gpu_translation_rule

#--------------------------------------------------------------------
# coo_matvec

coo_matvec_p = core.Primitive('coo_matvec')

def coo_matvec(data, row, col, v, *, shape, transpose=False):
  """Product of COO sparse matrix and a dense vector.

  Args:
    data : array of shape ``(nnz,)``.
    row : array of shape ``(nnz,)``
    col : array of shape ``(nnz,)`` and dtype ``row.dtype``
    v : array of shape ``(shape[0] if transpose else shape[1],)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
  """
  return coo_matvec_p.bind(data, row, col, v, shape=shape, transpose=transpose)

@coo_matvec_p.def_impl
def _coo_matvec_impl(data, row, col, v, *, shape, transpose):
  v = jnp.asarray(v)
  if transpose:
    row, col = col, row
  out_shape = shape[1] if transpose else shape[0]
  dv = data * v[col]
  return jnp.zeros(out_shape, dv.dtype).at[row].add(dv)

@coo_matvec_p.def_abstract_eval
def _coo_matvec_abstract_eval(data, row, col, v, *, shape, transpose):
  assert data.shape == row.shape == col.shape
  assert data.dtype == v.dtype
  assert row.dtype == col.dtype
  assert len(shape) == 2
  assert v.shape == (shape[0],) if transpose else (shape[1],)
  out_shape = shape[1] if transpose else shape[0]
  return core.ShapedArray((out_shape,), data.dtype)

def _coo_matvec_gpu_translation_rule(c, data, row, col, v, *, shape, transpose):
  return cusparse.coo_matvec(c, data, row, col, v, shape=shape, transpose=transpose)

xla.translations[coo_matvec_p] = xla.lower_fun(
    _coo_matvec_impl, multiple_results=False)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      coo_matvec_p] = _coo_matvec_gpu_translation_rule

#--------------------------------------------------------------------
# coo_matmat

coo_matmat_p = core.Primitive('coo_matmat')

def coo_matmat(data, row, col, B, *, shape, transpose=False):
  """Product of COO sparse matrix and a dense matrix.

  Args:
    data : array of shape ``(nnz,)``.
    row : array of shape ``(nnz,)``
    col : array of shape ``(nnz,)`` and dtype ``row.dtype``
    B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(shape[1] if transpose else shape[0], cols)``
      representing the matrix vector product.
  """
  return coo_matmat_p.bind(data, row, col, B, shape=shape, transpose=transpose)

@coo_matmat_p.def_impl
def _coo_matmat_impl(data, row, col, B, *, shape, transpose):
  B = jnp.asarray(B)
  if transpose:
    row, col = col, row
  out_shape = shape[1] if transpose else shape[0]
  dB = data[:, None] * B[col]
  return jnp.zeros((out_shape, B.shape[1]), dB.dtype).at[row].add(dB)

@coo_matmat_p.def_abstract_eval
def _coo_matmat_abstract_eval(data, row, col, B, *, shape, transpose):
  assert data.shape == row.shape == col.shape
  assert data.dtype == B.dtype
  assert len(shape) == 2
  assert B.shape[0] == shape[0] if transpose else shape[1]
  out_shape = shape[1] if transpose else shape[0]
  return core.ShapedArray((out_shape, B.shape[1]), data.dtype)

def _coo_matmat_gpu_translation_rule(c, data, row, col, B, *, shape, transpose):
  return cusparse.coo_matmat(c, data, row, col, B, shape=shape, transpose=transpose)

xla.translations[coo_matmat_p] = xla.lower_fun(
    _coo_matmat_impl, multiple_results=False)
if cusparse and cusparse.is_supported:
  xla.backend_specific_translations['gpu'][
      coo_matmat_p] = _coo_matmat_gpu_translation_rule
