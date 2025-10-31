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

#include "jaxlib/gpu/linalg_kernels.h"

#include <string>
#include <string_view>

#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;

namespace {
ffi::Error CholeskyUpdateFfiImpl(gpuStream_t stream, ffi::AnyBuffer matrix_in,
                                 ffi::AnyBuffer vector_in,
                                 ffi::Result<ffi::AnyBuffer> matrix_out,
                                 ffi::Result<ffi::AnyBuffer> vector_out) {
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(matrix_in.dimensions()));
  if (rows != cols) {
    return ffi::Error::InvalidArgument(
        "The matrix input to Cholesky update must be square.");
  }
  FFI_RETURN_IF_ERROR(CheckShape(vector_in.dimensions(), {batch, cols},
                                 "vector", "cholesky_update"));
  FFI_RETURN_IF_ERROR(CheckShape(matrix_out->dimensions(), {batch, rows, cols},
                                 "matrix_out", "cholesky_update"));
  FFI_RETURN_IF_ERROR(CheckShape(vector_out->dimensions(), {batch, cols},
                                 "vector_out", "cholesky_update"));
  FFI_ASSIGN_OR_RETURN(auto size, MaybeCastNoOverflow<int>(cols));
  auto dtype = matrix_in.element_type();
  if (dtype != ffi::F32 && dtype != ffi::F64) {
    return ffi::Error::InvalidArgument(
        "Invalid input type for Cholesky update; must be float32 or float64.");
  }
  if (vector_in.element_type() != dtype ||
      matrix_out->element_type() != dtype ||
      vector_out->element_type() != dtype) {
    return ffi::Error::InvalidArgument(
        "All input and output types for Cholesky update must match.");
  }
  bool is_single_precision = dtype == ffi::F32;
  auto matrix = matrix_out->untyped_data();
  if (matrix_in.untyped_data() != matrix) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(
        gpuMemcpyAsync(matrix, matrix_in.untyped_data(), matrix_in.size_bytes(),
                       gpuMemcpyDeviceToDevice, stream)));
  }
  auto vector = vector_out->untyped_data();
  if (vector_in.untyped_data() != vector) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(
        gpuMemcpyAsync(vector, vector_in.untyped_data(), vector_in.size_bytes(),
                       gpuMemcpyDeviceToDevice, stream)));
  }
  for (auto n = 0; n < batch; ++n) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(LaunchCholeskyUpdateFfiKernel(
        stream, matrix, vector, size, is_single_precision)));
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));
  }
  return ffi::Error::Success();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholeskyUpdateFfi, CholeskyUpdateFfiImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());

namespace {
ffi::Error LuPivotsToPermutationImpl(
    gpuStream_t stream, ffi::Buffer<ffi::DataType::S32> pivots,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> permutation) {
  FFI_ASSIGN_OR_RETURN((auto [batch_size, pivot_size]),
                       SplitBatch1D(pivots.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [permutation_batch, permutation_size]),
                       SplitBatch1D(permutation->dimensions()));
  if (permutation_batch != batch_size) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "pivots and permutation must have the same batch size.");
  }
  if (permutation_size < pivot_size) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        absl::StrFormat("Output permutation size %d must match or exceed the "
                        "trailing dimension of the input pivots %d.",
                        permutation_size, pivot_size));
  }
  LaunchLuPivotsToPermutationKernel(stream, batch_size, pivot_size,
                                    permutation_size, pivots.typed_data(),
                                    permutation->typed_data());
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));
  return ffi::Error::Success();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(LuPivotsToPermutation, LuPivotsToPermutationImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::S32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::S32>>());

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
