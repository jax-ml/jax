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

#ifndef JAXLIB_GPU_ROCM_POTRF_H_
#define JAXLIB_GPU_ROCM_POTRF_H_

#include "absl/status/status.h"
#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace solver {

// Cholesky decomposition via native rocsolver potrf (ROCm only).
//
// Bypasses hipSOLVER and calls rocsolver directly, which reaches the
// architecture-tuned TRSM+SYRK kernels on gfx942/gfx90a. Uses a dedicated
// per-stream rocblas_handle pool to avoid memory manager conflicts with
// handles that have been used for gesdd/getrf/geqrf.
//
// lower=true → rocblas_fill_lower; false → rocblas_fill_upper.
absl::Status RocPotrf(gpuStream_t stream, bool lower, int n,
                      float* a, int* info);
absl::Status RocPotrf(gpuStream_t stream, bool lower, int n,
                      double* a, int* info);
absl::Status RocPotrf(gpuStream_t stream, bool lower, int n,
                      gpuComplex* a, int* info);
absl::Status RocPotrf(gpuStream_t stream, bool lower, int n,
                      gpuDoubleComplex* a, int* info);

absl::Status RocPotrfBatched(gpuStream_t stream, bool lower, int n,
                             float** a, int* info, int batch);
absl::Status RocPotrfBatched(gpuStream_t stream, bool lower, int n,
                             double** a, int* info, int batch);
absl::Status RocPotrfBatched(gpuStream_t stream, bool lower, int n,
                             gpuComplex** a, int* info, int batch);
absl::Status RocPotrfBatched(gpuStream_t stream, bool lower, int n,
                             gpuDoubleComplex** a, int* info, int batch);

}  // namespace solver
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_ROCM_POTRF_H_
