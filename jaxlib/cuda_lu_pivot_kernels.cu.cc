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

#include "jaxlib/cuda_lu_pivot_kernels.h"

#include <array>
#include <iostream>

#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "third_party/tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {
namespace {

__device__ void ComputePermutation(const std::int32_t* pivots,
                                   std::int32_t* permutation_out,
                                   const std::int32_t pivot_size,
                                   const std::int32_t permutation_size) {
  for (int i = 0; i < permutation_size; ++i) {
    permutation_out[i] = i;
  }

  // Compute the permutation from a sequence of transpositions encoded in the
  // pivot array by applying the transpositions in order on the identity
  // permutation.
  for (int i = 0; i < pivot_size; ++i) {
    if ((pivots[i] < 0) || (pivots[i] >= permutation_size)) {
      continue;
    }
    std::int32_t swap_temporary = permutation_out[i];
    permutation_out[i] = permutation_out[pivots[i]];
    permutation_out[pivots[i]] = swap_temporary;
  }
}

__global__ void LuPivotsToPermutationKernel(
    const std::int32_t* pivots, std::int32_t* permutation_out,
    const std::int64_t batch_size, const std::int32_t pivot_size,
    const std::int32_t permutation_size) {
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < batch_size; idx += blockDim.x * gridDim.x) {
    // Fill in the output array with the identity permutation.
    ComputePermutation(pivots + idx * pivot_size,
                       permutation_out + idx * permutation_size, pivot_size,
                       permutation_size);
  }
}

}  // namespace

struct LuPivotsToPermutationDescriptor {
  std::int64_t batch_size;
  std::int32_t pivot_size;
  std::int32_t permutation_size;
};

std::string BuildCudaLuPivotsToPermutationDescriptor(
    std::int64_t batch_size, std::int32_t pivot_size,
    std::int32_t permutation_size) {
  return PackDescriptorAsString(LuPivotsToPermutationDescriptor{
      batch_size, pivot_size, permutation_size});
}

absl::Status CudaLuPivotsToPermutation_(cudaStream_t stream, void** buffers,
                                        const char* opaque,
                                        std::size_t opaque_len) {
  const std::int32_t* pivots =
      reinterpret_cast<const std::int32_t*>(buffers[0]);
  std::int32_t* permutation_out = reinterpret_cast<std::int32_t*>(buffers[1]);
  auto s =
      UnpackDescriptor<LuPivotsToPermutationDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const auto& descriptor = **s;

  const int block_dim = 128;
  const std::int64_t grid_dim = std::min<std::int64_t>(
      1024, (descriptor.batch_size + block_dim - 1) / block_dim);

  LuPivotsToPermutationKernel<<<grid_dim, block_dim,
                                /*dynamic_shared_mem_bytes=*/0, stream>>>(
      pivots, permutation_out, descriptor.batch_size, descriptor.pivot_size,
      descriptor.permutation_size);
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaGetLastError()));
  return absl::OkStatus();
}

void CudaLuPivotsToPermutation(cudaStream_t stream, void** buffers,
                               const char* opaque, size_t opaque_len,
                               XlaCustomCallStatus* status) {
  auto s = CudaLuPivotsToPermutation_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

}  // namespace jax
