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
"""Primitives for calling out to cusparse.

In general, these primitives are not meant to be used directly, but rather
are used internally in GPU translation rules of higher-level primitives.
"""

from functools import partial
from typing import Any

from jax._src import core
from jax._src import dispatch
from jax._src import ffi
from jax._src.interpreters import mlir
from jax._src.lib import gpu_sparse
from jax._src.lib import has_cpu_sparse
import numpy as np

if hasattr(gpu_sparse, "registrations"):
  for platform, targets in gpu_sparse.registrations().items():
    for name, value, api_version in targets:
      ffi.register_ffi_target(
          name, value, platform=platform, api_version=api_version
      )

if has_cpu_sparse:
  from jax._src.lib import cpu_sparse

  if hasattr(cpu_sparse, "registrations"):
    for platform, targets in cpu_sparse.registrations().items():
      for name, value, api_version in targets:
        ffi.register_ffi_target(
            name, value, platform=platform, api_version=api_version
        )

def _get_module(target_name_prefix: str) -> Any:
  if target_name_prefix == "cu":
    return gpu_sparse._cusparse
  elif target_name_prefix == "hip":
    return gpu_sparse._hipsparse
  else:
    raise ValueError(f"Unsupported target_name_prefix: {target_name_prefix}")

SUPPORTED_DATA_DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
SUPPORTED_INDEX_DTYPES = [np.int32]

# coo_spmv_p
# This is an internal-only primitive that calls into cusparse coo SpMV.
# This is a raw lowering that does no validation of inputs; the indices are
# assumed to be lexicographically sorted, deduplicated, and in-bounds.
coo_spmv_p = core.Primitive("coo_spmv")

def _coo_spmv_abstract_eval(data, row, col, x, *, transpose, shape):
  # TODO(jakevdp) support for batched matvec.
  assert data.shape == row.shape == col.shape
  assert row.ndim == 1
  assert x.ndim == 1

  assert row.dtype == col.dtype
  assert row.dtype in SUPPORTED_INDEX_DTYPES

  assert data.dtype == x.dtype
  assert x.dtype in SUPPORTED_DATA_DTYPES

  assert len(shape) == 2
  assert x.shape[0] == (shape[0] if transpose else shape[1])

  return core.ShapedArray(
    shape=shape[1:] if transpose else shape[:1],
    dtype=x.dtype)

def _coo_spmv_gpu_lowering(ctx, data, row, col, x, *, transpose, shape,
                           target_name_prefix):
  rows, cols = shape
  data_aval, row_aval, _, x_aval = ctx.avals_in
  nnz, = data_aval.shape
  buffer_size, opaque = _get_module(target_name_prefix).build_coo_matvec_descriptor(
      data_aval.dtype, x_aval.dtype, data_aval.dtype, row_aval.dtype,
      rows, cols, nnz, transpose)
  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_matvec_ffi")
  return rule(sub_ctx, data, row, col, x, opaque=opaque)[:1]

coo_spmv_p.def_abstract_eval(_coo_spmv_abstract_eval)
dispatch.simple_impl(coo_spmv_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    coo_spmv_p,
    partial(_coo_spmv_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
    coo_spmv_p,
    partial(_coo_spmv_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')


# coo_spmm_p
# This is an internal-only primitive that calls into cusparse COO SpMM.
# This is a raw lowering that does no validation of inputs; the indices are
# assumed to be lexicographically sorted, deduplicated, and in-bounds.
coo_spmm_p = core.Primitive("coo_spmm")

def _coo_spmm_abstract_eval(data, row, col, x, *, transpose, shape):
  # TODO(jakevdp) support for batched matmat.
  assert data.shape == row.shape == col.shape
  assert row.ndim == 1
  assert x.ndim == 2

  assert row.dtype == col.dtype
  assert row.dtype in SUPPORTED_INDEX_DTYPES

  assert data.dtype == x.dtype
  assert x.dtype in SUPPORTED_DATA_DTYPES

  assert len(shape) == 2
  assert x.shape[0] == (shape[0] if transpose else shape[1])

  return core.ShapedArray(
    shape=(shape[1] if transpose else shape[0], x.shape[1]),
    dtype=x.dtype)

def _coo_spmm_gpu_lowering(ctx, data, row, col, x, *, transpose, shape,
                           target_name_prefix):
  data_aval, row_aval, _, x_aval = ctx.avals_in
  nnz, = data_aval.shape
  _, Ccols = x_aval.shape

  batch_count = 1
  if len(shape) == 2:
    rows, cols = shape
  elif len(shape) == 3:
    batch_count, rows, cols = shape
    nnz = nnz // batch_count
  else:
    raise NotImplementedError(f"Unsupported shape: {shape}")

  # TODO(tianjianlu): use batch stride to trigger different mode of batch
  # computation. Currently batch_stride = 0 is not allowed because of the issue
  # in cusparse https://github.com/NVIDIA/CUDALibrarySamples/issues/81#issuecomment-1205562643
  # Set batch stride to be the matrix size for now.
  lhs_batch_stride = nnz
  B_rows = rows if transpose else cols
  rhs_batch_stride =  B_rows * Ccols

  buffer_size, opaque = _get_module(target_name_prefix).build_coo_matmat_descriptor(
      data_aval.dtype, x_aval.dtype, data_aval.dtype, row_aval.dtype,
      rows, cols, Ccols, nnz, transpose, batch_count, lhs_batch_stride,
      rhs_batch_stride)

  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_matmat_ffi")
  return rule(sub_ctx, data, row, col, x, opaque=opaque)[:1]


coo_spmm_p.def_abstract_eval(_coo_spmm_abstract_eval)
dispatch.simple_impl(coo_spmm_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    coo_spmm_p,
    partial(_coo_spmm_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
    coo_spmm_p,
    partial(_coo_spmm_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')

# csr_spmv_p
# This is an internal-only primitive that calls into cusparse csr SpMV.
# This is a raw lowering that does no validation of inputs; the indices are
# assumed to be lexicographically sorted, deduplicated, and in-bounds.
csr_spmv_p = core.Primitive("csr_spmv")

def _csr_spmv_abstract_eval(data, indices, indptr, x, *, transpose, shape):
  # TODO(tianjianlu) support for batched matvec.
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert data.shape == indices.shape
  assert indptr.shape[0] == shape[0] + 1
  assert x.ndim == 1

  assert indices.dtype == indptr.dtype
  assert indices.dtype in SUPPORTED_INDEX_DTYPES
  assert data.dtype == x.dtype
  assert x.dtype in SUPPORTED_DATA_DTYPES

  assert len(shape) == 2
  assert x.shape[0] == (shape[0] if transpose else shape[1])

  return core.ShapedArray(
    shape=shape[1:] if transpose else shape[:1],
    dtype=x.dtype)

def _csr_spmv_gpu_lowering(ctx, data, indices, indptr, x, *, transpose, shape,
                           target_name_prefix):
  rows, cols = shape
  data_aval, indices_aval, _, x_aval = ctx.avals_in
  nnz, = data_aval.shape
  buffer_size, opaque = _get_module(target_name_prefix).build_csr_matvec_descriptor(
      data_aval.dtype, x_aval.dtype, data_aval.dtype, indices_aval.dtype,
      rows, cols, nnz, transpose)
  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_matvec_ffi")
  return rule(sub_ctx, data, indices, indptr, x, opaque=opaque)[:1]

csr_spmv_p.def_abstract_eval(_csr_spmv_abstract_eval)
dispatch.simple_impl(csr_spmv_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    csr_spmv_p,
    partial(_csr_spmv_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
    csr_spmv_p,
    partial(_csr_spmv_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')

# csr_spmm_p
# This is an internal-only primitive that calls into cusparse CSR SpMM.
# This is a raw lowering that does no validation of inputs; the indices are
# assumed to be lexicographically sorted, deduplicated, and in-bounds.
csr_spmm_p = core.Primitive("csr_spmm")

def _csr_spmm_abstract_eval(data, indices, indptr, x, *, transpose, shape):
  # TODO(tianjianlu) support for batched matmat.
  assert data.ndim == indices.ndim == indptr.ndim == 1
  assert data.shape == indices.shape
  assert indptr.shape[0] == shape[0] + 1
  assert x.ndim == 2

  assert indices.dtype == indptr.dtype
  assert indices.dtype in SUPPORTED_INDEX_DTYPES
  assert data.dtype == x.dtype
  assert x.dtype in SUPPORTED_DATA_DTYPES

  assert len(shape) == 2
  assert x.shape[0] == (shape[0] if transpose else shape[1])

  return core.ShapedArray(
    shape=(shape[1] if transpose else shape[0], x.shape[1]),
    dtype=x.dtype)

def _csr_spmm_gpu_lowering(ctx, data, indices, indptr, x, *, transpose, shape,
                           target_name_prefix):
  rows, cols = shape
  data_aval, indices_aval, _, x_aval = ctx.avals_in
  nnz, = data_aval.shape
  _, Ccols = x_aval.shape
  buffer_size, opaque = _get_module(target_name_prefix).build_csr_matmat_descriptor(
      data_aval.dtype, x_aval.dtype, data_aval.dtype, indices_aval.dtype,
      rows, cols, Ccols, nnz, transpose)
  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_matmat_ffi")
  return rule(sub_ctx, data, indices, indptr, x, opaque=opaque)[:1]

csr_spmm_p.def_abstract_eval(_csr_spmm_abstract_eval)
dispatch.simple_impl(csr_spmm_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    csr_spmm_p,
    partial(_csr_spmm_gpu_lowering, target_name_prefix='cu'),
    platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(
    csr_spmm_p,
    partial(_csr_spmm_gpu_lowering, target_name_prefix='hip'),
    platform='rocm')


if has_cpu_sparse:
  def _csr_spmm_cpu_lowering(ctx, data, outer_indices, inner_indices, rhs):
    rule = ffi.ffi_lowering("cpu_csr_sparse_dense_ffi")
    return rule(ctx, data, outer_indices, inner_indices, rhs)


  # _csr_spmm_cpu_lowering can handle both matrix-matrix and matrix-vector
  # multiplication.
  mlir.register_lowering(
      csr_spmv_p,
      _csr_spmm_cpu_lowering,
      platform="cpu",
  )
  mlir.register_lowering(
      csr_spmm_p,
      _csr_spmm_cpu_lowering,
      platform="cpu",
  )

def coo_todense_gpu_lowering(ctx, data, row, col, *, shape, target_name_prefix):
  data_aval, row_aval, _ = ctx.avals_in
  nnz, = data_aval.shape
  rows, cols = shape
  buffer_size, opaque = _get_module(target_name_prefix).build_coo_todense_descriptor(
      data_aval.dtype, row_aval.dtype, rows, cols, nnz)
  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_todense_ffi")
  return rule(sub_ctx, data, row, col, opaque=opaque)[0]

def coo_fromdense_gpu_lowering(ctx, mat, *, nnz, index_dtype, target_name_prefix):
  mat_aval, = ctx.avals_in
  rows, cols = mat_aval.shape
  buffer_size, opaque = _get_module(target_name_prefix).build_coo_fromdense_descriptor(
      mat_aval.dtype, np.dtype(index_dtype), rows, cols, nnz)
  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[*ctx.avals_out, buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_fromdense_ffi")
  return rule(sub_ctx, mat, opaque=opaque)[:3]

def csr_todense_gpu_lowering(ctx, data, indices, indptr, *, shape, target_name_prefix):
  data_aval, indices_aval, _, = ctx.avals_in
  nnz, = data_aval.shape
  rows, cols = shape
  buffer_size, opaque = _get_module(target_name_prefix).build_csr_todense_descriptor(
      data_aval.dtype, indices_aval.dtype, rows, cols, nnz)
  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[ctx.avals_out[0], buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_todense_ffi")
  return rule(sub_ctx, data, indices, indptr, opaque=opaque)[0]

def csr_fromdense_gpu_lowering(ctx, mat, *, nnz, index_dtype, target_name_prefix):
  mat_aval, = ctx.avals_in
  rows, cols = mat_aval.shape
  buffer_size, opaque = _get_module(target_name_prefix).build_csr_fromdense_descriptor(
      mat_aval.dtype, np.dtype(index_dtype), rows, cols, nnz)
  buffer_aval = core.ShapedArray(shape=(buffer_size,), dtype=np.int8)
  sub_ctx = ctx.replace(avals_out=[*ctx.avals_out, buffer_aval])
  rule = ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_fromdense_ffi")
  return rule(sub_ctx, mat, opaque=opaque)[:3]
