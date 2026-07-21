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

#include <cstdint>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
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

#if defined(JAX_GPU_CUDA)
std::string ErrorString(cudaError_t error);
std::string ErrorString(cusolverStatus_t status);
std::string ErrorString(cusparseStatus_t status);
std::string ErrorString(cublasStatus_t status);
std::string ErrorString(CUresult error);
std::string ErrorString(CUptiResult error);
std::string ErrorString(cufftResult error);
std::string ErrorString(cudnnStatus_t status);
#elif defined(JAX_GPU_HIP)
std::string ErrorString(hipError_t error);
std::string ErrorString(hipsolverStatus_t status);
std::string ErrorString(hipsparseStatus_t status);
std::string ErrorString(hipblasStatus_t status);
std::string ErrorString(miopenStatus_t status);
#endif

template <typename T>
absl::Status AsStatus(T error, const char* file, std::int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_FALSE(error != GpuErrorTraits<T>::kSuccess)) {
    return absl::InternalError(absl::StrFormat("%s:%d: operation %s failed: %s",
                                               file, line, expr,
                                               ErrorString(error)));
  }
  return absl::OkStatus();
}

inline absl::Status AsStatus(const absl::Status& status, const char* file,
                             std::int64_t line, const char* expr) {
  if (ABSL_PREDICT_FALSE(!status.ok())) {
    return absl::Status(status.code(),
                        absl::StrFormat("%s:%d: operation %s failed: %s", file,
                                        line, expr, status.message()));
  }
  return status;
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_GPU_KERNEL_HELPERS_H_
