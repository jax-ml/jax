/* Copyright 2019 Google LLC

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

#include "jaxlib/cuda_lu_pivot_kernels.h"

#include <array>
#include <iostream>

#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"

namespace jax {
namespace {

__device__ void ComputePermutation(const std::int32_t* pivots,
                                   std::int32_t* permutation_out,
                                   const std::int32_t num_pivots,
                                   const std::int32_t num_permutation_rows) {
  for (int i = 0; i < num_permutation_rows; ++i) {
    permutation_out[i] = i;
  }

  // Compute the permutation from a sequence of transpositions encoded in the
  // pivot array by applying the transpositions in order on the identity
  // permutation.
  for (int i = 0; i < num_pivots; ++i) {
    if (pivots[i] < 0) {
      continue;
    }
    std::int32_t swap_temporary = permutation_out[i];
    permutation_out[i] = permutation_out[pivots[i]];
    permutation_out[pivots[i]] = swap_temporary;
  }
}

__global__ void LuPivotsToPermutationKernel(
    const std::int32_t* pivots, std::int32_t* permutation_out,
    const std::int64_t num_batches, const std::int32_t num_pivots,
    const std::int32_t num_permutation_rows) {
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < num_batches; idx += blockDim.x * gridDim.x) {
    // Fill in the output array with the identity permutation.
    ComputePermutation(pivots + idx * num_pivots,
                       permutation_out + idx * num_permutation_rows, num_pivots,
                       num_permutation_rows);
  }
}

}  // namespace

struct LuPivotsToPermutationDescriptor {
  std::int64_t num_batches;
  std::int32_t num_pivots;
  std::int32_t num_permutation_rows;
};

std::string BuildCudaLuPivotsToPermutationDescriptor(
    std::int64_t num_batches, std::int32_t num_pivots,
    std::int32_t num_permutation_rows) {
  return PackDescriptorAsString(LuPivotsToPermutationDescriptor{
      num_batches, num_pivots, num_permutation_rows});
}

void CudaLuPivotsToPermutation(cudaStream_t stream, void** buffers,
                               const char* opaque, std::size_t opaque_len) {
  const std::int32_t* pivots =
      reinterpret_cast<const std::int32_t*>(buffers[0]);
  std::int32_t* permutation_out = reinterpret_cast<std::int32_t*>(buffers[1]);
  const auto& descriptor =
      *UnpackDescriptor<LuPivotsToPermutationDescriptor>(opaque, opaque_len);

  const int block_dim = 128;
  const std::int64_t grid_dim = std::min<std::int64_t>(
      1024, (descriptor.num_batches + block_dim - 1) / block_dim);

  LuPivotsToPermutationKernel<<<grid_dim, block_dim,
                                /*dynamic_shared_mem_bytes=*/0, stream>>>(
      pivots, permutation_out, descriptor.num_batches, descriptor.num_pivots,
      descriptor.num_permutation_rows);
  ThrowIfError(cudaGetLastError());
}

}  // namespace jax
