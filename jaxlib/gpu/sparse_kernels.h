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

XLA_FFI_DECLARE_HANDLER_SYMBOL(kCsrToDense);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kCsrFromDense);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kCsrMatvec);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kCsrMatmat);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kCooToDense);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kCooFromDense);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kCooMatvec);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kCooMatmat);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kGtsv2);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_SPARSE_KERNELS_H_
