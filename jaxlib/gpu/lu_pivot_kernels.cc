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

#include "jaxlib/gpu/lu_pivot_kernels.h"

#include <cstdint>
#include <string>

#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;

XLA_FFI_Error* LuPivotsToPermutation(XLA_FFI_CallFrame* call_frame) {
  static const auto* kImpl =
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<gpuStream_t>>()
          .Attr<std::int64_t>("batch_size")
          .Attr<std::int32_t>("pivot_size")
          .Attr<std::int32_t>("permutation_size")
          .Arg<ffi::Buffer<ffi::DataType::S32>>()
          .Ret<ffi::Buffer<ffi::DataType::S32>>()
          .To([](gpuStream_t stream, std::int64_t batch_size,
                 std::int32_t pivot_size, std::int32_t permutation_size,
                 auto pivots, auto permutation) -> ffi::Error {
            LaunchLuPivotsToPermutationKernel(stream, batch_size, pivot_size,
                                              permutation_size, pivots.data,
                                              permutation->data);
            if (auto status = JAX_AS_STATUS(gpuGetLastError()); !status.ok()) {
              return ffi::Error(static_cast<XLA_FFI_Error_Code>(status.code()),
                                std::string(status.message()));
            }
            return ffi::Error::Success();
          })
          .release();
  return kImpl->Call(call_frame);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
