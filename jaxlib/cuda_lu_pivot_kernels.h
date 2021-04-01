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

std::string BuildCudaLuPivotsToPermutationDescriptor(
    std::int64_t num_batches, std::int32_t num_pivots,
    std::int32_t num_permutation_rows);

void CudaLuPivotsToPermutation(cudaStream_t stream, void** buffers,
                               const char* opaque, std::size_t opaque_len);

}  // namespace jax

#endif  // JAXLIB_CUDA_LU_PIVOT_KERNELS_H_
