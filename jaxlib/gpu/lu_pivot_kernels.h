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

#ifndef JAXLIB_GPU_LU_PIVOT_KERNELS_H_
#define JAXLIB_GPU_LU_PIVOT_KERNELS_H_

#include <cstdint>

#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

void LaunchLuPivotsToPermutationKernel(gpuStream_t stream,
                                       std::int64_t batch_size,
                                       std::int32_t pivot_size,
                                       std::int32_t permutation_size,
                                       const std::int32_t* pivots,
                                       std::int32_t* permutation);

XLA_FFI_Error* LuPivotsToPermutation(XLA_FFI_CallFrame* call_frame);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_LU_PIVOT_KERNELS_H_
