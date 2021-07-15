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

#ifndef JAXLIB_CUDA_GPU_KERNEL_HELPERS_H_
#define JAXLIB_CUDA_GPU_KERNEL_HELPERS_H_

#include <memory>

#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "third_party/gpus/cuda/include/cusparse.h"

#define JAX_THROW_IF_ERROR(expr) \
  jax::ThrowIfError(expr, __FILE__, __LINE__, #expr)

namespace jax {

// Used via JAX_THROW_IF_ERROR(expr) macro.
void ThrowIfError(cudaError_t error, const char* file, std::int64_t line,
                  const char* expr);
void ThrowIfError(cusolverStatus_t status, const char* file, std::int64_t line,
                  const char* expr);
void ThrowIfError(cusparseStatus_t status, const char* file, std::int64_t line,
                  const char* expr);
void ThrowIfError(cublasStatus_t status, const char* file, std::int64_t line,
                  const char* expr);

// Builds an array of pointers to each array in a batch, in device memory.
// Caution: the return value must be kept alive (e.g., via a stream
// synchronization) until the copy enqueued by MakeBatchPointers on `stream`
// completes.
std::unique_ptr<void*[]> MakeBatchPointers(cudaStream_t stream, void* buffer,
                                           void* dev_ptrs, int batch,
                                           int batch_elem_size);

}  // namespace jax

#endif  // JAXLIB_CUDA_GPU_KERNEL_HELPERS_H_
