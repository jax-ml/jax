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

#include "jaxlib/cuda_gpu_kernel_helpers.h"

#include <stdexcept>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace jax {
namespace {
std::string ErrorToString(cudaError_t error) {
  return cudaGetErrorString(error);
}

std::string ErrorToString(cusparseStatus_t status) {
  return cusparseGetErrorString(status);
}

std::string ErrorToString(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return "cuSolver success.";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "cuSolver has not been initialized";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "cuSolver allocation failed";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "cuSolver invalid value error";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "cuSolver architecture mismatch error";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "cuSolver mapping error";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "cuSolver execution failed";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "cuSolver internal error";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "cuSolver matrix type not supported error";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "cuSolver not supported error";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "cuSolver zero pivot error";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "cuSolver invalid license error";
    default:
      return absl::StrCat("Unknown cuSolver error: ", status);
  }
}

std::string ErrorToString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "cuBlas success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cuBlas has not been initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cuBlas allocation failure";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "cuBlas invalid value error";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "cuBlas architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "cuBlas mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "cuBlas execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "cuBlas internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "cuBlas not supported error";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "cuBlas license error";
    default:
      return "Unknown cuBlas error";
  }
}

template <typename T>
void ThrowError(T status, const char* file, std::int64_t line,
                const char* expr) {
  throw std::runtime_error(absl::StrFormat("%s:%d: operation %s failed: %s",
                                           file, line, expr,
                                           ErrorToString(status)));
}
}  // namespace

void ThrowIfError(cudaError_t error, const char* file, std::int64_t line,
                  const char* expr) {
  if (error != cudaSuccess) ThrowError(error, file, line, expr);
}

void ThrowIfError(cusolverStatus_t status, const char* file, std::int64_t line,
                  const char* expr) {
  if (status != CUSOLVER_STATUS_SUCCESS) ThrowError(status, file, line, expr);
}

void ThrowIfError(cusparseStatus_t status, const char* file, std::int64_t line,
                  const char* expr) {
  if (status != CUSPARSE_STATUS_SUCCESS) ThrowError(status, file, line, expr);
}

void ThrowIfError(cublasStatus_t status, const char* file, std::int64_t line,
                  const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) ThrowError(status, file, line, expr);
}

std::unique_ptr<void* []> MakeBatchPointers(cudaStream_t stream, void* buffer,
                                           void* dev_ptrs, int batch,
                                           int batch_elem_size) {
  char* ptr = static_cast<char*>(buffer);
  auto host_ptrs = absl::make_unique<void*[]>(batch);
  for (int i = 0; i < batch; ++i) {
    host_ptrs[i] = ptr;
    ptr += batch_elem_size;
  }
  JAX_THROW_IF_ERROR(cudaMemcpyAsync(dev_ptrs, host_ptrs.get(),
                                     sizeof(void*) * batch,
                                     cudaMemcpyHostToDevice, stream));
  return host_ptrs;
}
}  // namespace jax

