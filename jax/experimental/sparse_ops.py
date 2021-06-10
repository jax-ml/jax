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

Further down are some examples of potential high-level wrappers for sparse objects.
(API should be considered unstable and subject to change).
"""
import functools
import operator

from typing import Any, Tuple

from jax import api
from jax import core
from jax import jit
from jax import lax
from jax import tree_util
from jax import vmap
from jax.interpreters import batching
from jax.interpreters import xla
from jax.lib import cusparse
from jax.lib import xla_bridge
from jax.lib import xla_client
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad
from jax.util import safe_zip
from jax._src.lax.lax import ranges_like, remaining, _dot_general_batch_dim_nums

xb = xla_bridge
xops = xla_client.ops
Dtype = Any

#--------------------------------------------------------------------
# utilities
# TODO: possibly make these utilities into primitives, targeting
#       csr2coo/coo2csr/SPDDMM
@functools.partial(jit, static_argnums=1)
def _csr_to_coo(indptr, nnz):
  return jnp.cumsum(jnp.zeros_like(indptr, shape=nnz).at[indptr].add(1)) - 1

@functools.partial(jit, static_argnums=1)
def _coo_to_csr(row, nrows):
  indptr = jnp.zeros(nrows + 1, row.dtype)
  return indptr.at[1:].set(jnp.cumsum(jnp.bincount(row, length=nrows)))

@jit
def _csr_extract(indices, indptr, mat):
  """Extract values of dense matrix mat at given CSR indices."""
  return _coo_extract(_csr_to_coo(indptr, len(indices)), indices, mat)

@jit
def _coo_extract(row, col, mat):
  """Extract values of dense matrix mat at given COO indices."""
  return mat[row, col]

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
  return _coo_todense_impl(data, _csr_to_coo(indptr, len(indices)), indices, shape=shape)

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
  mat = jnp.asarray(mat)
  nnz = core.concrete_or_error(operator.index, nnz, "nnz argument of csr_fromdense()")
  return csr_fromdense_p.bind(mat, nnz=nnz, index_dtype=np.dtype(index_dtype))

@csr_fromdense_p.def_impl
def _csr_fromdense_impl(mat, *, nnz, index_dtype):
  mat = jnp.asarray(mat)
  assert mat.ndim == 2
  m = mat.shape[0]

  row, col = jnp.nonzero(mat, size=nnz)
  data = mat[row, col]

  true_nonzeros = jnp.arange(nnz) < (mat != 0).sum()
  data = jnp.where(true_nonzeros, data, 0)
  row = jnp.where(true_nonzeros, row, m)
  indices = col.astype(index_dtype)
  indptr = jnp.zeros(m + 1, dtype=index_dtype).at[1:].set(
      jnp.cumsum(jnp.bincount(row, length=m)))
  return data, indices, indptr

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
  row = _csr_to_coo(indptr, len(indices))
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
  row = _csr_to_coo(indptr, len(indices))
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
  return jnp.zeros(shape, data.dtype).at[row, col].add(data)

@coo_todense_p.def_abstract_eval
def _coo_todense_abstract_eval(data, row, col, *, shape):
  return core.ShapedArray(shape, data.dtype)

def _coo_todense_gpu_translation_rule(c, data, row, col, *, shape):
  return cusparse.coo_todense(c, data, row, col, shape=shape)

def _coo_todense_jvp(data_dot, data, row, col, *, shape):
  return coo_todense(data_dot, row, col, shape=shape)

def _coo_todense_transpose(ct, data, row, col, *, shape):
  # Note: we assume that transpose has the same sparsity pattern.
  # Can we check this?
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(row) or ad.is_undefined_primal(col):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert row.aval.dtype == col.aval.dtype
  assert ct.dtype == data.aval.dtype
  return _coo_extract(row, col, ct), row, col

ad.defjvp(coo_todense_p, _coo_todense_jvp, None, None)
ad.primitive_transposes[coo_todense_p] = _coo_todense_transpose
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
  mat = jnp.asarray(mat)
  nnz = core.concrete_or_error(operator.index, nnz, "nnz argument of coo_fromdense()")
  return coo_fromdense_p.bind(mat, nnz=nnz, index_dtype=index_dtype)

@coo_fromdense_p.def_impl
def _coo_fromdense_impl(mat, *, nnz, index_dtype):
  mat = jnp.asarray(mat)
  assert mat.ndim == 2

  row, col = jnp.nonzero(mat, size=nnz)
  data = mat[row, col]

  true_nonzeros = jnp.arange(nnz) < (mat != 0).sum()
  data = jnp.where(true_nonzeros, data, 0)

  return data, row.astype(index_dtype), col.astype(index_dtype)

@coo_fromdense_p.def_abstract_eval
def _coo_fromdense_abstract_eval(mat, *, nnz, index_dtype):
  data = core.ShapedArray((nnz,), mat.dtype)
  row = col = core.ShapedArray((nnz,), index_dtype)
  return data, row, col

def _coo_fromdense_gpu_translation_rule(c, mat, *, nnz, index_dtype):
  data, row, col = cusparse.coo_fromdense(
      c, mat, nnz=nnz, index_dtype=np.dtype(index_dtype))
  return xops.Tuple(c, [data, row, col])

def _coo_fromdense_jvp(primals, tangents, *, nnz, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = coo_fromdense(M, nnz=nnz, index_dtype=index_dtype)
  data, row, col = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_value(data)
  else:
    data_dot = _coo_extract(row, col, Mdot)

  tangents_out = (data_dot, ad.Zero.from_value(row), ad.Zero.from_value(col))

  return primals_out, tangents_out

def _coo_fromdense_transpose(ct, M, *, nnz, index_dtype):
  data, row, col = ct
  assert len(data) == nnz
  assert row.dtype == col.dtype == index_dtype
  if isinstance(row, ad.Zero) or isinstance(col, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return coo_todense(data, row, col, shape=M.aval.shape)

ad.primitive_jvps[coo_fromdense_p] = _coo_fromdense_jvp
ad.primitive_transposes[coo_fromdense_p] = _coo_fromdense_transpose

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

def _coo_matvec_jvp_mat(data_dot, data, row, col, v, *, shape, transpose):
  return coo_matvec(data_dot, row, col, v, shape=shape, transpose=transpose)

def _coo_matvec_jvp_vec(v_dot, data, row, col, v, *, shape, transpose):
  return coo_matvec(data, row, col, v_dot, shape=shape, transpose=transpose)

def _coo_matvec_transpose(ct, data, row, col, v, *, shape, transpose):
  assert not ad.is_undefined_primal(row)
  assert not ad.is_undefined_primal(col)

  if ad.is_undefined_primal(v):
    return data, row, col, coo_matvec(data, row, col, ct, shape=shape, transpose=not transpose)
  else:
    v = jnp.asarray(v)
    # return _coo_extract(row, col, jnp.outer(ct, v)), row, col, v
    return ct[row] * v[col], row, col, v

ad.defjvp(coo_matvec_p, _coo_matvec_jvp_mat, None, None, _coo_matvec_jvp_vec)
ad.primitive_transposes[coo_matvec_p] = _coo_matvec_transpose
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

def _coo_matmat_jvp_rule(primals_in, tangents_in, **params):
  vals, rows, cols, mat = primals_in
  sparse_mat_dot, rows_dot, cols_dot, mat_dot = tangents_in
  assert type(rows_dot) is ad.Zero
  assert type(cols_dot) is ad.Zero

  primals_out = coo_matmat(vals, rows, cols, mat, **params)
  _zero = lambda p, t: lax.zeros_like_array(p) if isinstance(t, ad.Zero) else t
  _sparse_mat_dot = _zero(vals, sparse_mat_dot)
  _mat_dot = _zero(mat, mat_dot)

  tangents_out = coo_matmat(_sparse_mat_dot, rows, cols, mat, **params) + coo_matmat(vals, rows, cols, _mat_dot, **params)
  return primals_out, tangents_out
ad.primitive_jvps[coo_matmat_p] = _coo_matmat_jvp_rule


#----------------------------------------------------------------------
# BCOO primitives: batched extension of COO.

def _bcoo_nse(mat, n_batch=0, n_dense=0):
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  mask = mask.sum(list(range(n_batch, mask.ndim)))
  return mask.max()

def _dedupe_bcoo(data, indices):
  f = _dedupe_bcoo_one
  n_batch = indices.ndim - 2
  for s1, s2 in safe_zip(indices.shape[:n_batch], data.shape[:n_batch]):
    if s1 != s2:
      # TODO: handle broadcasted dimensions.
      raise NotImplementedError("dedupe_bcoo for broadcasted dimensions.")
    f = vmap(f)
  return f(data, indices)

def _dedupe_bcoo_one(data, indices):
  assert indices.ndim == 2
  assert data.shape[:1] == indices.shape[1:]

  if indices.shape[0] == 0:
    return data, indices

  # This is a fixed-size version of jnp.unique() with return_indices=True
  # unique values are zero-filled at the end.
  perm = jnp.lexsort(indices[::-1])
  aux = indices[:, perm]
  mask = jnp.ones(indices.shape[1], dtype=bool)
  mask = mask.at[1:].set(jnp.any(aux[:, 1:] != aux[:, :-1], 0))
  imask = jnp.cumsum(mask) - 1
  indices_unique = jnp.where(mask, aux, 0)[:, jnp.argsort(~mask)]
  inv_idx = jnp.zeros_like(imask).at[perm].set(imask)

  # With the above, de-duping is easy.
  data_unique = jnp.zeros_like(data).at[inv_idx].add(data)
  return data_unique, indices_unique


def _validate_bcoo(data, indices, shape):
  assert jnp.issubdtype(indices.dtype, jnp.integer)

  n_sparse, nse = indices.shape[-2:]
  n_batch = indices.ndim - 2
  n_dense = len(shape) - n_batch - n_sparse
  assert n_dense >= 0

  def _compatible(shape1, shape2):
    return all(s1 in (1, s2) for s1, s2 in safe_zip(shape1, shape2))

  assert _compatible(data.shape[:n_batch], shape[:n_batch])
  assert data.shape[-(n_dense + 1):] == (nse,) + shape[n_batch + n_sparse:]

  assert _compatible(indices.shape[:n_batch], shape[:n_batch])
  assert indices.shape[n_batch:] == (n_sparse, nse)


#----------------------------------------------------------------------
# bcoo_todense

bcoo_todense_p = core.Primitive('bcoo_todense_p')

def bcoo_todense(data, indices, *, shape):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + block_dims``.
    indices : array of shape ``batch_dims + (n_sparse, nse)``
    shape : tuple; the shape of the (batched) matrix. Equal to
      ``batch_dims + sparse_dims + block_dims``
      where ``len(sparse_dims) == n_sparse``

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcoo_todense_p.bind(jnp.asarray(data), jnp.asarray(indices), shape=tuple(shape))

@bcoo_todense_p.def_impl
def _bcoo_todense_impl(data, indices, *, shape):
  _validate_bcoo(data, indices, shape)
  n_sparse = indices.shape[-2]
  n_batch = indices.ndim - 2
  batch_slices = tuple(slice(s) for s in shape[:n_batch])
  sparse_ind = tuple(indices[tuple(np.mgrid[batch_slices]) + (i,)] for i in range(n_sparse))
  batch_ind = tuple(np.mgrid[batch_slices + (slice(1),)])[:-1]
  if not sparse_ind:
    data = data.sum(n_batch, keepdims=bool(batch_ind))
  return jnp.zeros(shape, data.dtype).at[batch_ind + sparse_ind].add(data)

@bcoo_todense_p.def_abstract_eval
def _bcoo_todense_abstract_eval(data, indices, *, shape):
  _validate_bcoo(data, indices, shape)
  return core.ShapedArray(shape, data.dtype)

def _bcoo_todense_jvp(data_dot, data, indices, *, shape):
  return bcoo_todense(data_dot, indices, shape=shape)

def _bcoo_todense_transpose(ct, data, indices, *, shape):
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert ct.dtype == data.aval.dtype
  return bcoo_extract(indices, ct), indices

def _bcoo_todense_batching_rule(batched_args, batch_dims, *, shape):
  data, indices = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  return bcoo_todense(data, indices, shape=(max(data.shape[0], indices.shape[0]), *shape)), 0

ad.defjvp(bcoo_todense_p, _bcoo_todense_jvp, None)
ad.primitive_transposes[bcoo_todense_p] = _bcoo_todense_transpose
batching.primitive_batchers[bcoo_todense_p] = _bcoo_todense_batching_rule
xla.translations[bcoo_todense_p] = xla.lower_fun(
    _bcoo_todense_impl, multiple_results=False)

#--------------------------------------------------------------------
# bcoo_fromdense

bcoo_fromdense_p = core.Primitive('bcoo_fromdense')
bcoo_fromdense_p.multiple_results = True

def bcoo_fromdense(mat, *, nse=None, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create COO-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to COO, with ``ndim = n_batch + n_sparse + n_dense``.
    nse : number of specified elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of block_dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    data : array of shape ``mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]``
      and dtype ``mat.dtype``
    indices : array of shape ``mat.shape[:n_batch] + (n_sparse, nse)``
  """
  mat = jnp.asarray(mat)
  if nse is None:
    nse = _bcoo_nse(mat, n_batch, n_dense)
  nse = core.concrete_or_error(operator.index, nse, "nse argument of bcoo_fromdense")
  return bcoo_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)

@bcoo_fromdense_p.def_impl
def _bcoo_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  nonzero = lambda a: jnp.nonzero(a, size=nse) if a.ndim else ()
  for _ in range(n_batch):
    nonzero = vmap(nonzero, 0)
  indices = nonzero(mask)
  if not indices:
    indices = jnp.zeros(mask.shape[:n_batch] + (0, nse), index_dtype)
  else:
    indices = jnp.moveaxis(jnp.array(indices, index_dtype), 0, n_batch)
  data = bcoo_extract(indices, mat)

  true_nonzeros = jnp.arange(nse) < mask.sum(list(range(n_batch, mask.ndim)))[..., None]
  true_nonzeros = true_nonzeros[(n_batch + 1) * (slice(None),) + n_dense * (None,)]
  data = jnp.where(true_nonzeros, data, 0)

  return data, indices

@bcoo_fromdense_p.def_abstract_eval
def _bcoo_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
  n_sparse = mat.ndim - n_batch - n_dense
  data_shape = mat.shape[:n_batch] + (nse,) + mat.shape[n_batch + n_sparse:]
  index_shape = mat.shape[:n_batch] + (n_sparse, nse)
  return core.ShapedArray(data_shape, mat.dtype), core.ShapedArray(index_shape, index_dtype)

def _bcoo_fromdense_jvp(primals, tangents, *, nse, n_batch, n_dense, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)
  data, indices = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_value(data)
  else:
    data_dot = bcoo_extract(indices, Mdot)

  tangents_out = (data_dot, ad.Zero.from_value(indices))

  return primals_out, tangents_out

def _bcoo_fromdense_transpose(ct, M, *, nse, n_batch, n_dense, index_dtype):
  data, indices = ct
  n_sparse = M.ndim = n_batch - n_dense
  assert data.shape == M.shape[:n_batch] + (nse,) + M.shape[n_batch + n_sparse:]
  assert indices.shape == M.shape[:n_batch] + (n_sparse, nse)
  assert indices.dtype == index_dtype
  if isinstance(indices, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return bcoo_todense(data, indices, shape=M.aval.shape)

def _bcoo_fromdense_batching_rule(batched_args, batch_dims, *, nse, n_batch, n_dense, index_dtype):
  M, = batched_args
  if batch_dims != (0,):
    raise NotImplementedError(f"batch_dims={batch_dims}")
  return bcoo_fromdense(M, nse=nse, n_batch=n_batch + 1, n_dense=n_dense, index_dtype=index_dtype), (0, 0)

ad.primitive_jvps[bcoo_fromdense_p] = _bcoo_fromdense_jvp
ad.primitive_transposes[bcoo_fromdense_p] = _bcoo_fromdense_transpose
batching.primitive_batchers[bcoo_fromdense_p] = _bcoo_fromdense_batching_rule
xla.translations[bcoo_fromdense_p] = xla.lower_fun(
    _bcoo_fromdense_impl, multiple_results=True)

#----------------------------------------------------------------------
# bcoo_extract

bcoo_extract_p = core.Primitive('bcoo_extract')

def bcoo_extract(indices, mat):
  """Extract BCOO values from dense matrix `mat` at given BCOO indices."""
  return bcoo_extract_p.bind(indices, mat)

@bcoo_extract_p.def_impl
def _bcoo_extract_impl(indices, mat):
  n_sparse, _ = indices.shape[-2:]
  n_batch = indices.ndim - 2
  batch_slices = tuple(slice(s) for s in mat.shape[:n_batch])
  sparse_ind = tuple(indices[tuple(np.mgrid[batch_slices]) + (i,)] for i in range(n_sparse))
  batch_ind = tuple(np.mgrid[batch_slices + (slice(1),)])[:-1]
  if not sparse_ind + batch_ind:
    return mat[None]
  return mat[batch_ind + sparse_ind]

@bcoo_extract_p.def_abstract_eval
def _bcoo_extract_abstract_eval(indices, mat):
  n_sparse, nse = indices.shape[-2:]
  n_batch = indices.ndim - 2
  n_dense = mat.ndim - n_sparse - n_batch
  assert mat.shape[:n_batch] == indices.shape[:n_batch]
  out_shape = mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]
  return core.ShapedArray(out_shape, mat.dtype)

def _bcoo_extract_jvp(mat_dot, indices, mat):
  assert mat_dot.shape == mat.shape
  return bcoo_extract(indices, mat_dot)

def _bcoo_extract_transpose(ct, indices, mat):
  assert ad.is_undefined_primal(mat)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.dtype == mat.aval.dtype
  return indices, bcoo_todense(ct, indices, shape=mat.aval.shape)

def _bcoo_extract_batching_rule(batched_args, batch_dims):
  indices, mat = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    indices = indices[None, ...]
  if batch_dims[1] is None:
    mat = mat[None, ...]
  return bcoo_extract(indices, mat), 0

ad.defjvp(bcoo_extract_p, None, _bcoo_extract_jvp)
ad.primitive_transposes[bcoo_extract_p] = _bcoo_extract_transpose
batching.primitive_batchers[bcoo_extract_p] = _bcoo_extract_batching_rule
xla.translations[bcoo_extract_p] = xla.lower_fun(
    _bcoo_extract_impl, multiple_results=False)

#----------------------------------------------------------------------
# bcoo_dot_general
# (batched) general dot product of a BCOO sparse ND array and a dense ND array,
# returning a dense ND array.

bcoo_dot_general_p = core.Primitive('bcoo_dot_general')

def bcoo_dot_general(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  return bcoo_dot_general_p.bind(jnp.asarray(lhs_data), jnp.asarray(lhs_indices), jnp.asarray(rhs),
                                 dimension_numbers=dimension_numbers, lhs_shape=tuple(lhs_shape))

@bcoo_dot_general_p.def_impl
def _bcoo_dot_general_impl(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  lhs_data = jnp.asarray(lhs_data)
  lhs_indices = jnp.asarray(lhs_indices)
  rhs = jnp.asarray(rhs)
  # Validate all inputs via abstract_eval
  out_aval = _bcoo_dot_general_abstract_eval(lhs_data.aval, lhs_indices.aval, rhs.aval,
                                             dimension_numbers=dimension_numbers,
                                             lhs_shape=lhs_shape)

  (lhs_contracting, rhs_contracting) , (lhs_batch, rhs_batch) = dimension_numbers
  n_sparse = lhs_indices.shape[-2]
  n_batch = lhs_indices.ndim - 2

  # Move lhs batch dimensions to the front
  if lhs_batch:
    perm = list(lhs_batch) + remaining(range(n_batch), lhs_batch)
    lhs_data = lhs_data.transpose(perm + list(range(n_batch, lhs_data.ndim)))
    lhs_indices = lhs_indices.transpose(perm + list(range(n_batch, lhs_indices.ndim)))

  # Move lhs contracting dimensions to the front of sparse dims, in order
  n_contracting = len(lhs_contracting)
  lhs_contracting = [d - n_batch for d in lhs_contracting]
  perm = list(lhs_contracting) + remaining(range(n_sparse), lhs_contracting)
  lhs_indices = lhs_indices[..., jnp.array(perm), :]

  # Move rhs batch dimensions then contracting dimensions to the front, in order
  perm = (list(rhs_batch) + list(rhs_contracting) +
          remaining(range(rhs.ndim), rhs_batch, rhs_contracting))
  rhs = rhs.transpose(perm)

  out_array = jnp.zeros(out_aval.shape, out_aval.dtype)
  def result(out_array, lhs_data, lhs_indices, rhs):
    idx = tuple(lhs_indices)
    idx_right, idx_out = idx[:n_contracting], idx[n_contracting:]
    ctc = [0] if n_contracting else []
    prod = lax.dot_general(lhs_data, rhs[idx_right], (([], []), (ctc, ctc)))
    return out_array.at[idx_out].add(prod) if idx_out else prod.sum(0)
  for i in range(n_batch)[::-1]:
    axes_in = [0, 0, 0, 0]
    if lhs_data.shape[i] == 1:
      lhs_data = lax.squeeze(lhs_data, (i,))
      axes_in[1] = None
    if lhs_indices.shape[i] == 1:
      lhs_indices = lax.squeeze(lhs_indices, (i,))
      axes_in[2] = None
    if i >= len(lhs_batch):
      axes_in[3] = None
    result = vmap(result, tuple(axes_in))
  return result(out_array, lhs_data, lhs_indices, rhs)

@bcoo_dot_general_p.def_abstract_eval
def _bcoo_dot_general_abstract_eval(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  n_sparse = lhs_indices.shape[-2]
  n_batch = lhs_indices.ndim - 2
  _validate_bcoo(lhs_data, lhs_indices, lhs_shape)

  # Check for proper dimension_numbers
  for dims in [lhs_contracting, rhs_contracting, lhs_batch, rhs_batch]:
    assert len(dims) == len(set(dims))
  assert not set(lhs_contracting).intersection(lhs_batch)
  assert not set(rhs_contracting).intersection(rhs_batch)
  assert [lhs_shape[d] for d in lhs_contracting] == [rhs.shape[d] for d in rhs_contracting]
  assert [lhs_shape[d] for d in lhs_batch] == [rhs.shape[d] for d in rhs_batch]

  if lhs_batch and max(lhs_batch) >= n_batch:
    raise NotImplementedError(
      "bcoo_dot_general batch dimensions must be among the batch dimensions in the sparse representtaion.\n"
      f"got lhs_batch={lhs_batch}, n_batch={n_batch}")

  # TODO: support constraction of batch dimensions?
  if any(d < n_batch for d in lhs_contracting):
    raise NotImplementedError("bcoo_dot_general: contracting over batch dimensions.")

  # TODO: support contraction of dense dimensions?
  if any(d >= n_batch + n_sparse for d in lhs_contracting):
    raise NotImplementedError("bcoo_dot_general: contracting over dense dimensions.")

  out_dtype = jnp.promote_types(lhs_data.dtype, rhs.dtype)
  out_shape = (tuple(lhs_shape[i] for i in lhs_batch) +
               tuple(s for i, s in enumerate(lhs_shape) if i not in lhs_contracting + lhs_batch) +
               tuple(s for i, s in enumerate(rhs.shape) if i not in rhs_contracting + rhs_batch))
  return core.ShapedArray(out_shape, out_dtype)

def _bcoo_dot_general_jvp_lhs(lhs_data_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  return bcoo_dot_general(lhs_data_dot, lhs_indices, rhs, dimension_numbers=dimension_numbers, lhs_shape=lhs_shape)

def _bcoo_dot_general_jvp_rhs(rhs_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  return bcoo_dot_general(lhs_data, lhs_indices, rhs_dot, dimension_numbers=dimension_numbers, lhs_shape=lhs_shape)

def _bcoo_dot_general_transpose(ct, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  assert not ad.is_undefined_primal(lhs_indices)
  if type(ct) is ad.Zero:
    return ad.Zero
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_ndim = len(lhs_shape)
  rhs_ndim = rhs.aval.ndim if ad.is_undefined_primal(rhs) else rhs.ndim
  lhs_kept = remaining(range(lhs_ndim), lhs_contract, lhs_batch)
  rhs_kept = remaining(range(rhs_ndim), rhs_contract, rhs_batch)
  ans_batch, ans_lhs, ans_rhs = ranges_like(lhs_batch, lhs_kept, rhs_kept)
  if ad.is_undefined_primal(lhs_data):
    dims = ((ans_rhs, rhs_kept), (ans_batch, rhs_batch))
    lhs_contract_sorted_by_rhs = list(np.take(lhs_contract, np.argsort(rhs_contract)))
    # TODO: extract these sparse indices without constructing the dense matrix.
    out_axes = np.argsort(list(lhs_batch) + lhs_kept + lhs_contract_sorted_by_rhs)
    out_dense = lax.transpose(lax.dot_general(ct, rhs, dimension_numbers=dims), out_axes)
    return bcoo_extract(lhs_indices, out_dense), lhs_indices, rhs
  else:
    dims = ((lhs_kept, ans_lhs), (lhs_batch, ans_batch))
    rhs_contract_sorted_by_lhs = list(np.take(rhs_contract, np.argsort(lhs_contract)))
    out_axes = np.argsort(list(rhs_batch) + rhs_contract_sorted_by_lhs + rhs_kept)
    result = bcoo_dot_general(lhs_data, lhs_indices, ct, lhs_shape=lhs_shape, dimension_numbers=dims)
    return lhs_data, lhs_indices, lax.transpose(result, out_axes)

def _bcoo_dot_general_batch_rule(batched_args, batch_dims, *, dimension_numbers, lhs_shape):
  lhs_data, lhs_indices, rhs = batched_args
  batch_dims = list(batch_dims)
  batch_size = max(0 if dim is None else arg.shape[dim]
                   for arg, dim in zip(batched_args, batch_dims))
  if batch_dims[0] is None:
    lhs_data = lhs_data[None]
    batch_dims[0] = 0
  if batch_dims[1] is None:
    lhs_indices = lhs_indices[None]
    batch_dims[1] = 0
  # TODO: handle different batchings between lhs_data and lhs_indices?
  assert batch_dims[0] == batch_dims[1] == 0
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (len(lhs_shape), rhs.ndim), (batch_dims[0], batch_dims[2]), dimension_numbers)
  new_shape = (batch_size, *lhs_shape)
  batched_out = bcoo_dot_general(lhs_data, lhs_indices, rhs, lhs_shape=new_shape,
                                 dimension_numbers=new_dimension_numbers)
  return batched_out, result_batch_dim

ad.defjvp(bcoo_dot_general_p, _bcoo_dot_general_jvp_lhs, None, _bcoo_dot_general_jvp_rhs)
ad.primitive_transposes[bcoo_dot_general_p] = _bcoo_dot_general_transpose
batching.primitive_batchers[bcoo_dot_general_p] = _bcoo_dot_general_batch_rule
xla.translations[bcoo_dot_general_p] = xla.lower_fun(
    _bcoo_dot_general_impl, multiple_results=False)

#----------------------------------------------------------------------
# Sparse objects (APIs subject to change)
class JAXSparse:
  """Base class for high-level JAX sparse objects."""
  shape: Tuple[int, int]
  nnz: property
  dtype: property

  def __init__(self, args, *, shape):
    self.shape = shape

  def __repr__(self):
    return f"{self.__class__.__name__}({self.dtype}{list(self.shape)}, nnz={self.nnz})"

  def tree_flatten(self):
    raise NotImplementedError("tree_flatten")

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(children, **aux_data)

  def matvec(self, v):
    raise NotImplementedError("matvec")

  def matmat(self, B):
    raise NotImplementedError("matmat")

  def transpose(self):
    raise NotImplementedError()

  @property
  def T(self):
    return self.transpose()

  def __matmul__(self, other):
    if isinstance(other, JAXSparse):
      raise NotImplementedError("matmul between two sparse objects.")
    other = jnp.asarray(other)
    if other.ndim == 1:
      return self.matvec(other)
    elif other.ndim == 2:
      return self.matmat(other)
    else:
      raise NotImplementedError(f"matmul with object of shape {other.shape}")


@tree_util.register_pytree_node_class
class CSR(JAXSparse):
  """Experimental CSR matrix implemented in JAX; API subject to change."""
  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  nnz = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)

  def __init__(self, args, *, shape):
    self.data, self.indices, self.indptr = map(jnp.asarray, args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nnz=None, index_dtype=np.int32):
    if nnz is None:
      nnz = (mat != 0).sum()
    return cls(csr_fromdense(mat, nnz=nnz, index_dtype=index_dtype), shape=mat.shape)

  @api.jit
  def todense(self):
    return csr_todense(self.data, self.indices, self.indptr, shape=self.shape)

  @api.jit
  def matvec(self, v):
    return csr_matvec(self.data, self.indices, self.indptr, v, shape=self.shape)

  @api.jit
  def matmat(self, B):
    return csr_matmat(self.data, self.indices, self.indptr, B, shape=self.shape)

  def transpose(self):
    return CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {"shape": self.shape}


@tree_util.register_pytree_node_class
class CSC(JAXSparse):
  """Experimental CSC matrix implemented in JAX; API subject to change."""
  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  nnz = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)

  def __init__(self, args, *, shape):
    self.data, self.indices, self.indptr = map(jnp.asarray, args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nnz=None, index_dtype=np.int32):
    if nnz is None:
      nnz = (mat != 0).sum()
    return cls(csr_fromdense(mat.T, nnz=nnz, index_dtype=index_dtype), shape=mat.shape)

  @api.jit
  def todense(self):
    return csr_todense(self.data, self.indices, self.indptr, shape=self.shape[::-1]).T

  @api.jit
  def matvec(self, v):
    return csr_matvec(self.data, self.indices, self.indptr, v, shape=self.shape[::-1], transpose=True)

  @api.jit
  def matmat(self, B):
    return csr_matmat(self.data, self.indices, self.indptr, B, shape=self.shape[::-1], transpose=True)

  def transpose(self):
    return CSR((self.data, self.indices, self.indptr), shape=self.shape[::-1])

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {"shape": self.shape}


@tree_util.register_pytree_node_class
class COO(JAXSparse):
  """Experimental COO matrix implemented in JAX; API subject to change."""
  data: jnp.ndarray
  row: jnp.ndarray
  col: jnp.ndarray
  nnz = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)

  def __init__(self, args, *, shape):
    self.data, self.row, self.col = map(jnp.asarray, args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nnz=None, index_dtype=np.int32):
    if nnz is None:
      nnz = (mat != 0).sum()
    return cls(coo_fromdense(mat, nnz=nnz, index_dtype=index_dtype), shape=mat.shape)

  @api.jit
  def todense(self):
    return coo_todense(self.data, self.row, self.col, shape=self.shape)

  @api.jit
  def matvec(self, v):
    return coo_matvec(self.data, self.row, self.col, v, shape=self.shape)

  @api.jit
  def matmat(self, B):
    return coo_matmat(self.data, self.row, self.col, B, shape=self.shape)

  def transpose(self):
    return COO((self.data, self.col, self.row), shape=self.shape[::-1])

  def tree_flatten(self):
    return (self.data, self.row, self.col), {"shape": self.shape}


@tree_util.register_pytree_node_class
class BCOO(JAXSparse):
  """Experimental BCOO matrix implemented in JAX; API subject to change."""
  data: jnp.ndarray
  indices: jnp.ndarray
  nnz = property(lambda self: self.data.size)
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 2)
  n_sparse = property(lambda self: self.indices.shape[-2])
  n_dense = property(lambda self: self.data.ndim - 1 - self.n_batch)

  def __init__(self, args, *, shape):
    self.data, self.indices = map(jnp.asarray, args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nnz=None, index_dtype=np.int32):
    return cls(bcoo_fromdense(mat, nse=nnz, index_dtype=index_dtype), shape=mat.shape)

  @api.jit
  def todense(self):
    return bcoo_todense(self.data, self.indices, shape=self.shape)

  @api.jit
  def matvec(self, v):
    return bcoo_dot_general(self.data, self.indices, v, lhs_shape=self.shape,
                            dimension_numbers=(([1], [0]), ([], [])))

  @api.jit
  def matmat(self, B):
    return bcoo_dot_general(self.data, self.indices, B, lhs_shape=self.shape,
                            dimension_numbers=(([1], [0]), ([], [])))

  def transpose(self):
    if self.n_batch or self.n_dense:
      raise NotImplementedError("BCOO transpose with batch or dense dimensions")
    return BCOO((self.data, self.indices[::-1]), shape=self.shape[::-1])

  def tree_flatten(self):
    return (self.data, self.indices), {"shape": self.shape}
