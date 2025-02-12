/* Copyright 2019 The JAX Authors.

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

#include "jaxlib/gpu/prng_kernels.h"

#include <cstdint>
#include <functional>
#include <string_view>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;


namespace {
ffi::Error ThreeFry2x32Impl(gpuStream_t stream,
                            ffi::Buffer<ffi::DataType::U32> keys0,
                            ffi::Buffer<ffi::DataType::U32> keys1,
                            ffi::Buffer<ffi::DataType::U32> data0,
                            ffi::Buffer<ffi::DataType::U32> data1,
                            ffi::Result<ffi::Buffer<ffi::DataType::U32>> out0,
                            ffi::Result<ffi::Buffer<ffi::DataType::U32>> out1) {
  std::int64_t n =
      absl::c_accumulate(out0->dimensions(), 1, std::multiplies<int64_t>());
  LaunchThreeFry2x32KernelFfi(stream, n, keys0.typed_data(), keys1.typed_data(),
                              data0.typed_data(), data1.typed_data(),
                              out0->typed_data(), out1->typed_data());
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));
  return ffi::Error::Success();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(ThreeFry2x32Ffi, ThreeFry2x32Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U32>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U32>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U32>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::U32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::U32>>());

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
