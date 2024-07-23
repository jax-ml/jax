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
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;

namespace {
absl::Status CholeskyUpdateImpl(gpuStream_t stream, void** buffers,
                                const char* opaque, std::size_t opaque_len) {
  auto s = UnpackDescriptor<CholeskyUpdateDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CholeskyUpdateDescriptor& d = **s;
  LaunchCholeskyUpdateKernel(stream, buffers, d);
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuGetLastError()));
  return absl::OkStatus();
}
}  // namespace

void CholeskyUpdate(gpuStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CholeskyUpdateImpl(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    std::string_view message = s.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

namespace {
ffi::Error LuPivotsToPermutationImpl(
    gpuStream_t stream, std::int32_t permutation_size,
    ffi::Buffer<ffi::DataType::S32> pivots,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> permutation) {
  auto dims = pivots.dimensions();

  if (dims.size() < 1) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "pivots must have at least one dimension");
  }
  FFI_ASSIGN_OR_RETURN(std::int32_t pivot_size,
                       MaybeCastNoOverflow<std::int32_t>(dims.back()));
  std::int64_t batch_size = 1;
  if (dims.size() >= 2) {
    batch_size =
        absl::c_accumulate(dims.first(dims.size() - 1), 1, std::multiplies<>());
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
                                  .Attr<std::int32_t>("permutation_size")
                                  .Arg<ffi::Buffer<ffi::DataType::S32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::S32>>());

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
