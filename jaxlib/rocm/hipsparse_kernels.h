/* Copyright 2021 Google LLC

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

#ifndef JAXLIB_HIPSPARSE_KERNELS_H_
#define JAXLIB_HIPSPARSE_KERNELS_H_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "jaxlib/handle_pool.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/hipsparse.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

// Some functionality defined here is only available in CUSPARSE 11.3 or newer.
#define JAX_CUSPARSE_11030 (CUSPARSE_VERSION >= 11300)

namespace jax {

using SparseHandlePool = HandlePool<hipsparseHandle_t, hipStream_t>;

template <>
/*static*/ absl::StatusOr<SparseHandlePool::Handle>
SparseHandlePool::Borrow(hipStream_t stream);

union HipConst {
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

HipConst HipZero(hipDataType type);
HipConst HipOne(hipDataType type);

struct SparseMatDescriptor {
  hipDataType value_type;
  hipsparseIndexType_t index_type;
  int rows, cols, nnz;
};

struct DenseMatDescriptor {
  hipDataType type;
  int rows, cols;
};

struct DenseVecDescriptor {
  hipDataType type;
  int size;
};

// CsrToDense: Convert CSR matrix to dense matrix

void CsrToDense(hipStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status);

// CsrFromDense: Convert dense matrix to CSR matrix

void CsrFromDense(hipStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

// CsrMatvec: Product of CSR matrix and dense vector.

struct CsrMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  hipsparseOperation_t op;
};

void CsrMatvec(hipStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);

// CsrMatmat: Product of CSR matrix and dense matrix.

struct CsrMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  hipsparseOperation_t op_A;
};

void CsrMatmat(hipStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);

// CooToDense: Convert COO matrix to dense matrix

void CooToDense(hipStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status);

// CooFromDense: Convert dense matrix to COO matrix

void CooFromDense(hipStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

// CooMatvec: Product of COO matrix and dense vector.

struct CooMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  hipsparseOperation_t op;
};

void CooMatvec(hipStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);

// CooMatmat: Product of COO matrix and dense matrix.

struct CooMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  hipsparseOperation_t op_A;
};

void CooMatmat(hipStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status);

struct Gtsv2Descriptor {
  int m, n, ldb;
};

void gtsv2_f32(hipStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status);

void gtsv2_f64(hipStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace jax

#endif  // JAXLIB_HIPSPARSE_KERNELS_H_
