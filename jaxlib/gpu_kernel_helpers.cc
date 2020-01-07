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

#include "jaxlib/gpu_kernel_helpers.h"

#include <stdexcept>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace jax {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(
        absl::StrCat("CUDA operation failed: ", cudaGetErrorString(error)));
  }
}

std::unique_ptr<void*[]> MakeBatchPointers(cudaStream_t stream, void* buffer,
                                           void** dev_ptrs, int batch,
                                           int batch_elem_size);
                                            {
  auto host_ptrs = absl::make_unique<void*[]>(batch);
  for (int i = 0; i < batch; ++i) {
    host_ptrs[i] = buffer;
    buffer += batch_elem_size;
  }
  ThrowIfError(cudaMemcpyAsync(dev_ptrs, host_ptrs.data(),
                               sizeof(void*) * batch, cudaMemcpyHostToDevice,
                               stream));
  return host_ptrs;
}
}  // namespace jax

