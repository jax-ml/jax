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

from jax._src import core
from jax._src import dispatch
from jax._src import ffi
from jax._src.interpreters import mlir
from jax._src.lib import cpu_sparse
from jax._src.lib import gpu_sparse
import numpy as np

if hasattr(gpu_sparse, "registrations"):
  for platform, targets in gpu_sparse.registrations().items():
    for name, value, api_version in targets:
      ffi.register_ffi_target(
          name, value, platform=platform, api_version=api_version
      )

for platform, targets in cpu_sparse.registrations().items():
  for name, value, api_version in targets:
    ffi.register_ffi_target(
        name, value, platform=platform, api_version=api_version
    )

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
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_matvec_ffi_v2")(
      ctx, data, row, col, x, transpose=transpose)

coo_spmv_p.def_abstract_eval(_coo_spmv_abstract_eval)
dispatch.simple_impl(coo_spmv_p)
mlir.register_lowering(
  coo_spmv_p,
  partial(_coo_spmv_gpu_lowering, target_name_prefix='cu'),
  platform='cuda')
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
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_matmat_ffi_v2")(
      ctx, data, row, col, x, transpose=transpose)


coo_spmm_p.def_abstract_eval(_coo_spmm_abstract_eval)
dispatch.simple_impl(coo_spmm_p)
mlir.register_lowering(
  coo_spmm_p,
  partial(_coo_spmm_gpu_lowering, target_name_prefix='cu'),
  platform='cuda')
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
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_matvec_ffi_v2")(
      ctx, data, indices, indptr, x, transpose=transpose)

csr_spmv_p.def_abstract_eval(_csr_spmv_abstract_eval)
dispatch.simple_impl(csr_spmv_p)
mlir.register_lowering(
  csr_spmv_p,
  partial(_csr_spmv_gpu_lowering, target_name_prefix='cu'),
  platform='cuda')
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
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_matmat_ffi_v2")(
      ctx, data, indices, indptr, x, transpose=transpose)

csr_spmm_p.def_abstract_eval(_csr_spmm_abstract_eval)
dispatch.simple_impl(csr_spmm_p)
mlir.register_lowering(
  csr_spmm_p,
  partial(_csr_spmm_gpu_lowering, target_name_prefix='cu'),
  platform='cuda')
mlir.register_lowering(
  csr_spmm_p,
  partial(_csr_spmm_gpu_lowering, target_name_prefix='hip'),
  platform='rocm')


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
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_todense_ffi_v2")(
      ctx, data, row, col)[0]

def coo_fromdense_gpu_lowering(ctx, mat, *, nnz, index_dtype, target_name_prefix):
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_coo_fromdense_ffi_v2")(
      ctx, mat)

def csr_todense_gpu_lowering(ctx, data, indices, indptr, *, shape, target_name_prefix):
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_todense_ffi_v2")(
      ctx, data, indices, indptr)[0]

def csr_fromdense_gpu_lowering(ctx, mat, *, nnz, index_dtype, target_name_prefix):
  return ffi.ffi_lowering(f"{target_name_prefix}sparse_csr_fromdense_ffi_v2")(
      ctx, mat)
