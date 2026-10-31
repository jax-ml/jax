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

#ifndef JAXLIB_GPU_SLEEP_KERNEL_H_
#define JAXLIB_GPU_SLEEP_KERNEL_H_

#include <cstdint>

#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

// Launches a kernel that occupies the stream for approximately `nanoseconds`.
void LaunchSleepKernel(gpuStream_t stream, std::int64_t nanoseconds);

// FFI handler that sleeps on the device for approximately the `microseconds`
// attribute. Its operand is passed through as its result, which must alias the
// operand: the data is not touched, the operand and result only order the sleep
// between the operations that produce and consume them.
XLA_FFI_DECLARE_HANDLER_SYMBOL(SleepFfi);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_SLEEP_KERNEL_H_
