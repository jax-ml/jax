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

#include "jaxlib/gpu/sleep_kernel.h"

#include <cstdint>

#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;

namespace {
ffi::Error SleepImpl(gpuStream_t stream, ffi::AnyBuffer x,
                     ffi::Result<ffi::AnyBuffer> out, std::int64_t microseconds) {
  if (microseconds < 0) {
    return ffi::Error::InvalidArgument(
        "The sleep duration in microseconds must be non-negative");
  }
  // The result aliases the operand (see the callers), so nothing is copied:
  // it only lets later operations depend on the sleep.
  if (out->untyped_data() != x.untyped_data()) {
    return ffi::Error::InvalidArgument(
        "The result of the sleep must alias its operand");
  }
  LaunchSleepKernel(stream, microseconds * 1000);
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));
  return ffi::Error::Success();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(SleepFfi, SleepImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<std::int64_t>("microseconds"));

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
