/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_GPU_SPARSE_KERNELS_H_
#define JAXLIB_GPU_SPARSE_KERNELS_H_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/handle_pool.h"
#include "xla/service/custom_call_status.h"

namespace jax {

using SparseHandlePool = HandlePool<gpusparseHandle_t, gpuStream_t>;

template <>
/*static*/ absl::StatusOr<SparseHandlePool::Handle> SparseHandlePool::Borrow(
    gpuStream_t stream);

namespace JAX_GPU_NAMESPACE {

union SparseConst {
  int8_t i8[2];
  int16_t i16[2];
  int32_t i32[2];
  int64_t i64[2];
  uint8_t u8[2];
  uint16_t u16[2];
  uint32_t u32[2];
  uint64_t u64[2];
  float f32[2];
  double f64[2];
};

SparseConst ConstZero(gpuDataType type);
SparseConst ConstOne(gpuDataType type);

struct SparseMatDescriptor {
  gpuDataType value_type;
  gpusparseIndexType_t index_type;
  int rows, cols, nnz;
  int batch_count = 1;
  int batch_stride = 0;
};

struct DenseMatDescriptor {
  gpuDataType type;
  int rows, cols;
  int batch_count = 1;
  int batch_stride = 0;
};

struct DenseVecDescriptor {
  gpuDataType type;
  int size;
};

#if JAX_GPU_HAVE_SPARSE
// CsrToDense: Convert CSR matrix to dense matrix

void CsrToDense(gpuStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status);

// CsrFromDense: Convert dense matrix to CSR matrix

void CsrFromDense(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

// CsrMatvec: Product of CSR matrix and dense vector.

struct CsrMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  gpusparseOperation_t op;
};

void CsrMatvec(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);

// CsrMatmat: Product of CSR matrix and dense matrix.

struct CsrMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  gpusparseOperation_t op_A;
};

void CsrMatmat(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);

// CooToDense: Convert COO matrix to dense matrix

void CooToDense(gpuStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status);

// CooFromDense: Convert dense matrix to COO matrix

void CooFromDense(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

// CooMatvec: Product of COO matrix and dense vector.

struct CooMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  gpusparseOperation_t op;
};

void CooMatvec(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);

// CooMatmat: Product of COO matrix and dense matrix.

struct CooMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  gpusparseOperation_t op_A;
};

void CooMatmat(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);
#endif  // JAX_GPU_HAVE_SPARSE

struct Gtsv2Descriptor {
  int batch, m, n, ldb;
};

void gtsv2_f32(gpuStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status);

void gtsv2_f64(gpuStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_SPARSE_KERNELS_H_
