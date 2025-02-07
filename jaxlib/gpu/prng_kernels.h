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

#ifndef JAXLIB_GPU_PRNG_KERNELS_H_
#define JAXLIB_GPU_PRNG_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

void LaunchThreeFry2x32KernelFfi(gpuStream_t stream,
                                 std::int64_t n,
                                 std::uint32_t *keys0, std::uint32_t *keys1,
                                 std::uint32_t *data0, std::uint32_t *data1,
                                 std::uint32_t *out0, std::uint32_t *out1);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ThreeFry2x32Ffi);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_PRNG_KERNELS_H_
