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

#include "jaxlib/gpu/rocm/potrf.h"

#include <map>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "rocm/include/rocsolver/rocsolver.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace solver {

namespace {

absl::Status RocblasStatusToStatus(rocblas_status status, const char* file,
                                   int line, const char* expr) {
  if (ABSL_PREDICT_FALSE(status != rocblas_status_success)) {
    return absl::InternalError(
        absl::StrFormat("%s:%d: %s failed: rocblas_status %d", file, line, expr,
                        static_cast<int>(status)));
  }
  return absl::OkStatus();
}



// Dedicated per-stream rocblas_handle pool for rocsolver potrf.
//
// We cannot reuse SolverHandlePool (hipsolverDnHandle_t) for direct rocsolver
// calls: after LU/QR/gesdd operations the handle's internal rocBLAS memory
// manager is in a state that conflicts with rocsolver's own allocations,
// causing rocblas_status_memory_error (5). A fresh rocblas_handle created via
// rocblas_create_handle has a clean memory manager state.
absl::Mutex g_potrf_handle_pool_mu(absl::kConstInit);
std::map<gpuStream_t, std::vector<rocblas_handle>> g_potrf_handle_pool
    ABSL_GUARDED_BY(g_potrf_handle_pool_mu);

}  // namespace

namespace {

absl::StatusOr<rocblas_handle> BorrowPotrfHandle(gpuStream_t stream) {
  rocblas_handle handle = nullptr;
  {
    absl::MutexLock lock(&g_potrf_handle_pool_mu);
    auto& pool = g_potrf_handle_pool[stream];
    if (!pool.empty()) {
      handle = pool.back();
      pool.pop_back();
    }
  }  // lock released before slow ROCm API calls
  if (handle == nullptr) {
    rocblas_status st = rocblas_create_handle(&handle);
    if (st != rocblas_status_success) {
      return RocblasStatusToStatus(st, __FILE__, __LINE__,
                                   "rocblas_create_handle");
    }
  }
  rocblas_status st = rocblas_set_stream(handle, stream);
  if (st != rocblas_status_success) {
    rocblas_destroy_handle(handle);
    return RocblasStatusToStatus(st, __FILE__, __LINE__, "rocblas_set_stream");
  }
  return handle;
}

void ReturnPotrfHandle(rocblas_handle handle, gpuStream_t stream) {
  absl::MutexLock lock(&g_potrf_handle_pool_mu);
  g_potrf_handle_pool[stream].push_back(handle);
}

struct PotrfHandleGuard {
  rocblas_handle h;
  gpuStream_t stream;
  ~PotrfHandleGuard() { ReturnPotrfHandle(h, stream); }
};

}  // namespace

#define JAX_GPU_DEFINE_ROC_POTRF(Type, CType, Name)                            \
  absl::Status RocPotrf(gpuStream_t stream, bool lower, int n,                 \
                        Type* a, int* info) {                                  \
    rocblas_fill uplo =                                                         \
        lower ? rocblas_fill_lower : rocblas_fill_upper;                        \
    auto maybe_h = BorrowPotrfHandle(stream);                                  \
    if (!maybe_h.ok()) return maybe_h.status();                                \
    rocblas_handle h_raw = maybe_h.value();                                    \
    PotrfHandleGuard guard{h_raw, stream};                                     \
    rocblas_status st = Name(h_raw, uplo, n,                                   \
                             reinterpret_cast<CType*>(a), n,                   \
                             reinterpret_cast<rocblas_int*>(info));             \
    return RocblasStatusToStatus(st, __FILE__, __LINE__, #Name);                \
  }

JAX_GPU_DEFINE_ROC_POTRF(float, float, rocsolver_spotrf);
JAX_GPU_DEFINE_ROC_POTRF(double, double, rocsolver_dpotrf);
JAX_GPU_DEFINE_ROC_POTRF(gpuComplex, rocblas_float_complex, rocsolver_cpotrf);
JAX_GPU_DEFINE_ROC_POTRF(gpuDoubleComplex, rocblas_double_complex,
                         rocsolver_zpotrf);
#undef JAX_GPU_DEFINE_ROC_POTRF

#define JAX_GPU_DEFINE_ROC_POTRF_BATCHED(Type, CType, Name)                    \
  absl::Status RocPotrfBatched(gpuStream_t stream, bool lower, int n,          \
                               Type** a, int* info, int batch) {               \
    rocblas_fill uplo =                                                         \
        lower ? rocblas_fill_lower : rocblas_fill_upper;                        \
    auto maybe_h = BorrowPotrfHandle(stream);                                  \
    if (!maybe_h.ok()) return maybe_h.status();                                \
    rocblas_handle h_raw = maybe_h.value();                                    \
    PotrfHandleGuard guard{h_raw, stream};                                     \
    rocblas_status st = Name(h_raw, uplo, n,                                   \
                             reinterpret_cast<CType* const*>(a), n,            \
                             reinterpret_cast<rocblas_int*>(info), batch);     \
    return RocblasStatusToStatus(st, __FILE__, __LINE__, #Name);                \
  }

JAX_GPU_DEFINE_ROC_POTRF_BATCHED(float, float, rocsolver_spotrf_batched);
JAX_GPU_DEFINE_ROC_POTRF_BATCHED(double, double, rocsolver_dpotrf_batched);
JAX_GPU_DEFINE_ROC_POTRF_BATCHED(gpuComplex, rocblas_float_complex,
                                 rocsolver_cpotrf_batched);
JAX_GPU_DEFINE_ROC_POTRF_BATCHED(gpuDoubleComplex, rocblas_double_complex,
                                 rocsolver_zpotrf_batched);
#undef JAX_GPU_DEFINE_ROC_POTRF_BATCHED

}  // namespace solver
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
