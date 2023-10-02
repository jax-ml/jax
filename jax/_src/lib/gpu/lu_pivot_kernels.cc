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

#include "jax/_src/lib/gpu/lu_pivot_kernels.h"

#include <string_view>

#include "jax/_src/lib/gpu/gpu_kernel_helpers.h"
#include "jax/_src/lib/gpu/vendor.h"
#include "jax/_src/lib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

absl::Status LuPivotsToPermutation_(gpuStream_t stream, void** buffers,
                                    const char* opaque,
                                    std::size_t opaque_len) {
  auto s =
      UnpackDescriptor<LuPivotsToPermutationDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  LaunchLuPivotsToPermutationKernel(stream, buffers, **s);
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuGetLastError()));
  return absl::OkStatus();
}

}  // namespace

void LuPivotsToPermutation(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len,
                           XlaCustomCallStatus* status) {
  auto s = LuPivotsToPermutation_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    std::string_view message = s.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
