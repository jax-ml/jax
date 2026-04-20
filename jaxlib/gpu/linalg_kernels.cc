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

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>

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

namespace {
ffi::Error TridiagonalSolvePerturbedImpl(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer subdiag,
    ffi::AnyBuffer diag, ffi::AnyBuffer superdiag, ffi::AnyBuffer rhs,
    ffi::Result<ffi::AnyBuffer> x) {
  FFI_ASSIGN_OR_RETURN((auto [batch_size, n]), SplitBatch1D(diag.dimensions()));

  FFI_RETURN_IF_ERROR(CheckShape(subdiag.dimensions(),
                                 std::make_tuple(batch_size, n), "subdiag",
                                 "tridiagonal_solve_perturbed"));
  FFI_RETURN_IF_ERROR(CheckShape(superdiag.dimensions(),
                                 std::make_tuple(batch_size, n), "superdiag",
                                 "tridiagonal_solve_perturbed"));

  FFI_ASSIGN_OR_RETURN((auto [rhs_batch, rhs_n, k_rhs]),
                       SplitBatch2D(rhs.dimensions()));
  if (rhs_batch != batch_size || rhs_n != n) {
    return ffi::Error::InvalidArgument(
        "RHS batch size and length must match diagonals.");
  }

  FFI_RETURN_IF_ERROR(CheckShape(x->dimensions(),
                                 std::make_tuple(batch_size, n, k_rhs), "x",
                                 "tridiagonal_solve_perturbed"));

  auto dtype = diag.element_type();
  if (dtype != ffi::DataType::F32 && dtype != ffi::DataType::F64 &&
      dtype != ffi::DataType::C64 && dtype != ffi::DataType::C128) {
    return ffi::Error::InvalidArgument(
        "Invalid input type for tridiagonal solve; must be float32, float64, "
        "complex64, or complex128.");
  }
  if (subdiag.element_type() != dtype || superdiag.element_type() != dtype ||
      rhs.element_type() != dtype || x->element_type() != dtype) {
    return ffi::Error::InvalidArgument(
        "All input and output types for tridiagonal solve must match.");
  }

  size_t workspace_bytes =
      (batch_size * n * 3 + batch_size * k_rhs) * ffi::ByteWidth(dtype);
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<char>(scratch, workspace_bytes,
                                               "tridiagonal_solve_perturbed"));

  LaunchTridiagonalSolvePerturbedKernel(
      stream, batch_size, n, k_rhs, dtype, subdiag.untyped_data(),
      diag.untyped_data(), superdiag.untyped_data(), rhs.untyped_data(),
      x->untyped_data(), workspace);

  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));
  return ffi::Error::Success();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(TridiagonalSolvePerturbedFfi,
                              TridiagonalSolvePerturbedImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
