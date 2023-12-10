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

#ifndef JAXLIB_GPU_GPU_KERNEL_HELPERS_H_
#define JAXLIB_GPU_GPU_KERNEL_HELPERS_H_

#include <memory>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/gpu/vendor.h"

#define JAX_AS_STATUS(expr) \
  jax::JAX_GPU_NAMESPACE::AsStatus(expr, __FILE__, __LINE__, #expr)

#define JAX_THROW_IF_ERROR(expr)                             \
  {                                                          \
    auto s___ = (expr);                                      \
    if (ABSL_PREDICT_FALSE(!s___.ok()))                      \
      throw std::runtime_error(std::string(s___.message())); \
  }

#define JAX_RETURN_IF_ERROR(expr)                    \
  {                                                  \
    auto s___ = (expr);                              \
    if (ABSL_PREDICT_FALSE(!s___.ok())) return s___; \
  }

#define JAX_ASSIGN_OR_RETURN(lhs, expr) \
  auto s___ = (expr);                   \
  if (ABSL_PREDICT_FALSE(!s___.ok())) { \
    return s___.status();               \
  }                                     \
  lhs = (*std::move(s___))

namespace jax {
namespace JAX_GPU_NAMESPACE {

// Used via JAX_AS_STATUS(expr) macro.
absl::Status AsStatus(gpuError_t error, const char* file, std::int64_t line,
                      const char* expr);
absl::Status AsStatus(gpusolverStatus_t status, const char* file,
                      std::int64_t line, const char* expr);
absl::Status AsStatus(gpusparseStatus_t status, const char* file,
                      std::int64_t line, const char* expr);
absl::Status AsStatus(gpublasStatus_t status, const char* file,
                      std::int64_t line, const char* expr);
#ifdef JAX_GPU_CUDA
absl::Status AsStatus(CUresult error, const char* file, std::int64_t line,
                      const char* expr);
absl::Status AsStatus(CUptiResult error, const char* file, std::int64_t line,
                      const char* expr);
absl::Status AsStatus(cufftResult error, const char* file, std::int64_t line,
                      const char* expr);
#endif

// Builds an array of pointers to each array in a batch, in device memory.
// Caution: the return value must be kept alive (e.g., via a stream
// synchronization) until the copy enqueued by MakeBatchPointers on `stream`
// completes.
absl::StatusOr<std::unique_ptr<void*[]>> MakeBatchPointers(gpuStream_t stream,
                                                           void* buffer,
                                                           void* dev_ptrs,
                                                           int batch,
                                                           int batch_elem_size);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_GPU_KERNEL_HELPERS_H_
