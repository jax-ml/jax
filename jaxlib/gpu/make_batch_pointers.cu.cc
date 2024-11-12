/* Copyright 2024 The JAX Authors.

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

#include "jaxlib/gpu/make_batch_pointers.h"

#include <algorithm>
#include <cstdint>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace {
__global__ void MakeBatchPointersAsyncKernel(char* buffer_in, void** buffer_out,
                                             int64_t batch,
                                             int64_t batch_elem_size) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch;
       idx += blockDim.x * gridDim.x) {
    buffer_out[idx] = buffer_in + idx * batch_elem_size;
  }
}
}  // namespace

void MakeBatchPointersAsync(gpuStream_t stream, void* buffer_in,
                            void* buffer_out, int64_t batch,
                            int64_t batch_elem_size) {
  const std::size_t block_dim = 128;
  const std::size_t grid_dim =
      std::min<std::size_t>(1024, (batch + block_dim - 1) / block_dim);
  MakeBatchPointersAsyncKernel<<<grid_dim, block_dim, 0, stream>>>(
      static_cast<char*>(buffer_in), static_cast<void**>(buffer_out), batch,
      batch_elem_size);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
