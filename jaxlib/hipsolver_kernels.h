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

#ifndef JAXLIB_HIPSOLVER_KERNELS_H_
#define JAXLIB_HIPSOLVER_KERNELS_H_

#include "absl/status/statusor.h"
#include "jaxlib/handle_pool.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/hipblas.h"
#include "rocm/include/hipsolver.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {

using SolverHandlePool = HandlePool<hipsolverHandle_t, hipStream_t>;

template <>
absl::StatusOr<SolverHandlePool::Handle>
SolverHandlePool::Borrow(hipStream_t stream);

// Set of types known to Hipsolver.
enum class HipsolverType {
  F32,
  F64,
  C64,
  C128,
};

// potrf: Cholesky decomposition

struct PotrfDescriptor {
  HipsolverType type;
  hipsolverFillMode_t uplo;
  std::int64_t batch, n;
  int lwork;
};

void Potrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);
// getrf: LU decomposition

struct GetrfDescriptor {
  HipsolverType type;
  int batch, m, n, lwork;
};

void Getrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// geqrf: QR decomposition

struct GeqrfDescriptor {
  HipsolverType type;
  int batch, m, n, lwork;
};

void Geqrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// orgqr/ungqr: apply elementary Householder transformations

struct OrgqrDescriptor {
  HipsolverType type;
  int batch, m, n, k, lwork;
};

void Orgqr(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

struct SyevdDescriptor {
  HipsolverType type;
  hipsolverFillMode_t uplo;
  int batch, n;
  int lwork;
};

void Syevd(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Singular value decomposition using QR algorithm: gesvd

struct GesvdDescriptor {
  HipsolverType type;
  int batch, m, n;
  int lwork;
  signed char jobu, jobvt;
};

void Gesvd(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace jax

#endif  // JAXLIB_HIPSOLVER_KERNELS_H_
