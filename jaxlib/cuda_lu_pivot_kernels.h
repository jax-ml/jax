/* Copyright 2021 Google LLC

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

#ifndef JAXLIB_CUDA_LU_PIVOT_KERNELS_H_
#define JAXLIB_CUDA_LU_PIVOT_KERNELS_H_

#include <cstddef>
#include <string>

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace jax {

struct LuPivotsToPermutationDescriptor {
  std::int64_t batch_size;
  std::int32_t pivot_size;
  std::int32_t permutation_size;
};

void LaunchLuPivotsToPermutationKernel(
    cudaStream_t stream, void** buffers,
    LuPivotsToPermutationDescriptor descriptor);

}  // namespace jax

#endif  // JAXLIB_CUDA_LU_PIVOT_KERNELS_H_
