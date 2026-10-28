/* Copyright 2026 The JAX Authors.

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

#include "jaxlib/gpu/digamma_kernels.h"

#include <cstdint>

#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/device_attributes.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace ffi = xla::ffi;

// Select `DigammaF32WithF64Kernel` when the GPU executes FP64 within 4x of FP32
// throughput (e.g., ratio == 2 on A100/H100/B200/MI250/MI300).
constexpr int kMaxF64PerfRatio = 4;

ffi::Error DigammaImpl(gpuStream_t stream, int32_t device_ordinal,
                       ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> out) {
  const ffi::DataType dtype = x.element_type();
  if (out->element_type() != dtype) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "digamma input and output element types must match.");
  }
  const int64_t n = static_cast<int64_t>(x.element_count());
  if (static_cast<int64_t>(out->element_count()) != n) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "digamma input and output element counts must match.");
  }
  if (n == 0) {
    return ffi::Error::Success();
  }

  switch (dtype) {
    case ffi::DataType::BF16:
    case ffi::DataType::F16:
    case ffi::DataType::F32: {
      const int ratio = GetSingleToDoublePrecisionPerfRatio(device_ordinal);
      if (ratio >= 1 && ratio <= kMaxF64PerfRatio) {
        LaunchDigammaF32WithF64Kernel(stream, n, dtype, x.untyped_data(),
                                      out->untyped_data());
      } else {
        LaunchDigammaF32Kernel(stream, n, dtype, x.untyped_data(),
                               out->untyped_data());
      }
      break;
    }
    case ffi::DataType::F64:
      LaunchDigammaF64Kernel(stream, n, x.reinterpret_data<double>(),
                             out->reinterpret_data<double>());
      break;
    default:
      return ffi::Error(
          ffi::ErrorCode::kInvalidArgument,
          "digamma FFI kernel only supports BF16, F16, F32, and F64.");
  }

  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(DigammaFfi, DigammaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>(),
                              {ffi::Traits::kCmdBufferCompatible});

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
