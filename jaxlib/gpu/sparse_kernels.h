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

#include <cstdint>

#include "absl/status/statusor.h"
#include "jaxlib/gpu/handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

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
absl::StatusOr<SparseConst> ConstOne(gpuDataType type);

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

struct CsrMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  gpusparseOperation_t op;
};

struct CsrMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  gpusparseOperation_t op_A;
};

struct CooMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  gpusparseOperation_t op;
};

struct CooMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  gpusparseOperation_t op_A;
};

XLA_FFI_DECLARE_HANDLER_SYMBOL(CsrToDenseFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CsrFromDenseFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CsrMatvecFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CsrMatmatFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CooToDenseFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CooFromDenseFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CooMatvecFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CooMatmatFfi);

#endif  // JAX_GPU_HAVE_SPARSE

XLA_FFI_DECLARE_HANDLER_SYMBOL(kGtsv2);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_SPARSE_KERNELS_H_
