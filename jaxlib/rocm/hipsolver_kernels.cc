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

#include "jaxlib/rocm/hipsolver_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/rocm/hip_gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/hipsolver.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {

template <>
/*static*/ absl::StatusOr<SolverHandlePool::Handle>
SolverHandlePool::Borrow(hipStream_t stream) {
  SolverHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  hipsolverHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

static int SizeOfHipsolverType(HipsolverType type) {
  switch (type) {
    case HipsolverType::F32:
      return sizeof(float);
    case HipsolverType::F64:
      return sizeof(double);
    case HipsolverType::C64:
      return sizeof(hipFloatComplex);
    case HipsolverType::C128:
      return sizeof(hipDoubleComplex);
  }
}

// potrf: Cholesky decomposition

static absl::Status Potrf_(hipStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<PotrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const PotrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
        hipMemcpyAsync(buffers[1], buffers[0],
                       SizeOfHipsolverType(d.type) * d.batch * d.n * d.n,
                       hipMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[2]);
  void* workspace = buffers[3];
  if (d.batch == 1) {
    switch (d.type) {
      case HipsolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverSpotrf(handle.get(), d.uplo, d.n, a, d.n,
                            static_cast<float*>(workspace), d.lwork, info)));
        break;
      }
      case HipsolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverDpotrf(handle.get(), d.uplo, d.n, a, d.n,
                            static_cast<double*>(workspace), d.lwork, info)));
        break;
      }
      case HipsolverType::C64: {
        hipFloatComplex* a = static_cast<hipFloatComplex*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverCpotrf(
            handle.get(), d.uplo, d.n, a, d.n,
            static_cast<hipFloatComplex*>(workspace), d.lwork, info)));
        break;
      }
      case HipsolverType::C128: {
        hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverZpotrf(
            handle.get(), d.uplo, d.n, a, d.n,
            static_cast<hipDoubleComplex*>(workspace), d.lwork, info)));
        break;
      }
    }
  } else {
    auto buffer_ptrs_host =
        MakeBatchPointers(stream, buffers[1], workspace, d.batch,
                          SizeOfHipsolverType(d.type) * d.n * d.n);
    JAX_RETURN_IF_ERROR(buffer_ptrs_host.status());
    // Make sure that accesses to buffer_ptrs_host complete before we delete it.
    // TODO(phawkins): avoid synchronization here.
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipStreamSynchronize(stream)));
    switch (d.type) {
      case HipsolverType::F32: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverSpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<float**>(workspace), d.n,
            static_cast<float*>(workspace + (d.batch * sizeof(float*))), d.lwork,
            info, d.batch)));
        break;
      }
      case HipsolverType::F64: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverDpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<double**>(workspace), d.n,
            static_cast<double*>(workspace + (d.batch * sizeof(double*))), d.lwork,
            info, d.batch)));
        break;
      }
      case HipsolverType::C64: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverCpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<hipFloatComplex**>(workspace), d.n,
            static_cast<hipFloatComplex*>(workspace + (d.batch * sizeof(hipFloatComplex*))),d.lwork,
            info, d.batch)));
        break;
      }
      case HipsolverType::C128: {
        hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverZpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<hipDoubleComplex**>(workspace), d.n,
            static_cast<hipDoubleComplex*>(workspace + (d.batch * sizeof(hipDoubleComplex*))), d.lwork,
            info, d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Potrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Potrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// getrf: LU decomposition

static absl::Status Getrf_(hipStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GetrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GetrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfHipsolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        hipMemcpyDeviceToDevice, stream)));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case HipsolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverSgetrf(handle.get(), d.m, d.n, a, d.m,
                            static_cast<float*>(workspace), d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case HipsolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverDgetrf(handle.get(), d.m, d.n, a, d.m,
                            static_cast<double*>(workspace), d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case HipsolverType::C64: {
      hipFloatComplex* a = static_cast<hipFloatComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverCgetrf(handle.get(), d.m, d.n, a, d.m,
                            static_cast<hipFloatComplex*>(workspace), d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case HipsolverType::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverZgetrf(
            handle.get(), d.m, d.n, a, d.m,
            static_cast<hipDoubleComplex*>(workspace), d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Getrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Getrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// geqrf: QR decomposition

static absl::Status Geqrf_(hipStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GeqrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GeqrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfHipsolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        hipMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  // TODO(rocm): workaround for unset devinfo. See SWDEV-317485
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(hipMemsetAsync(info, 0, sizeof(int) * d.batch, stream)));

  void* workspace = buffers[4];
  switch (d.type) {
    case HipsolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* tau = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverSgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                            static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case HipsolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* tau = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverDgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                            static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case HipsolverType::C64: {
      hipFloatComplex* a = static_cast<hipFloatComplex*>(buffers[1]);
      hipFloatComplex* tau = static_cast<hipFloatComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverCgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<hipFloatComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case HipsolverType::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      hipDoubleComplex* tau = static_cast<hipDoubleComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverZgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<hipDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Geqrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Geqrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// orgqr/ungqr: apply elementary Householder transformations

static absl::Status Orgqr_(hipStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<OrgqrDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const OrgqrDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[2] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
        buffers[2], buffers[0],
        SizeOfHipsolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        hipMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  // TODO(rocm): workaround for unset devinfo. See SWDEV-317485
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(hipMemsetAsync(info, 0, sizeof(int) * d.batch, stream)));

  void* workspace = buffers[4];
  switch (d.type) {
    case HipsolverType::F32: {
      float* a = static_cast<float*>(buffers[2]);
      float* tau = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverSorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                            static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case HipsolverType::F64: {
      double* a = static_cast<double*>(buffers[2]);
      double* tau = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverDorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                            static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case HipsolverType::C64: {
      hipFloatComplex* a = static_cast<hipFloatComplex*>(buffers[2]);
      hipFloatComplex* tau = static_cast<hipFloatComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverCungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<hipFloatComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case HipsolverType::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[2]);
      hipDoubleComplex* tau = static_cast<hipDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverZungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<hipDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Orgqr(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Orgqr_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

static absl::Status Syevd_(hipStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SyevdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SyevdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfHipsolverType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
      hipMemcpyDeviceToDevice, stream)));
  hipsolverEigMode_t jobz = HIPSOLVER_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  switch (d.type) {
    case HipsolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverSsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                            static_cast<float*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case HipsolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverDsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                            static_cast<double*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case HipsolverType::C64: {
      hipFloatComplex* a = static_cast<hipFloatComplex*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverCheevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                            static_cast<hipFloatComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case HipsolverType::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverZheevd(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<hipDoubleComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Syevd(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Syevd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// TODO(rocm): add Syevj_ apis when support from hipsolver is ready
// Singular value decomposition using QR algorithm: gesvd

static absl::Status Gesvd_(hipStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GesvdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GesvdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfHipsolverType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      hipMemcpyDeviceToDevice, stream)));
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  switch (d.type) {
    case HipsolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      float* u = static_cast<float*>(buffers[3]);
      float* vt = static_cast<float*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            hipsolverSgesvd(handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s,
                            u, d.m, vt, d.n, static_cast<float*>(work), d.lwork,
                            /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case HipsolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      double* u = static_cast<double*>(buffers[3]);
      double* vt = static_cast<double*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverDgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<double*>(work), d.lwork,
            /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case HipsolverType::C64: {
      hipFloatComplex* a = static_cast<hipFloatComplex*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      hipFloatComplex* u = static_cast<hipFloatComplex*>(buffers[3]);
      hipFloatComplex* vt = static_cast<hipFloatComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverCgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<hipFloatComplex*>(work), d.lwork, /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case HipsolverType::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      hipDoubleComplex* u = static_cast<hipDoubleComplex*>(buffers[3]);
      hipDoubleComplex* vt = static_cast<hipDoubleComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsolverZgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<hipDoubleComplex*>(work), d.lwork,
            /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Gesvd(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// TODO(rocm): add Gesvdj_ apis when support from hipsolver is ready
}  // namespace jax
