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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {
namespace {
namespace py = pybind11;

using SolverHandlePool = HandlePool<cusolverDnHandle_t, cudaStream_t>;

template <>
/*static*/ absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
    cudaStream_t stream) {
  SolverHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cusolverDnHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

// Set of types known to Cusolver.
enum class Type {
  F32,
  F64,
  C64,
  C128,
};

// Converts a NumPy dtype to a Type.
Type DtypeToType(const py::dtype& np_type) {
  static auto* types = new absl::flat_hash_map<std::pair<char, int>, Type>({
      {{'f', 4}, Type::F32},
      {{'f', 8}, Type::F64},
      {{'c', 8}, Type::C64},
      {{'c', 16}, Type::C128},
  });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", py::repr(np_type)));
  }
  return it->second;
}

int SizeOfType(Type type) {
  switch (type) {
    case Type::F32:
      return sizeof(float);
    case Type::F64:
      return sizeof(double);
    case Type::C64:
      return sizeof(cuComplex);
    case Type::C128:
      return sizeof(cuDoubleComplex);
  }
}

// potrf: Cholesky decomposition

struct PotrfDescriptor {
  Type type;
  cublasFillMode_t uplo;
  std::int64_t batch, n;
  int lwork;
};

// Returns the workspace size and a descriptor for a potrf operation.
std::pair<int, py::bytes> BuildPotrfDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  std::int64_t workspace_size;
  cublasFillMode_t uplo =
      lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  if (b == 1) {
    switch (type) {
      case Type::F32:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnSpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(float);
        break;
      case Type::F64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnDpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(double);
        break;
      case Type::C64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnCpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(cuComplex);
        break;
      case Type::C128:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnZpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(cuDoubleComplex);
        break;
    }
  } else {
    // We use the workspace buffer for our own scratch space.
    workspace_size = sizeof(void*) * b;
  }
  return {workspace_size,
          PackDescriptor(PotrfDescriptor{type, uplo, b, n, lwork})};
}

absl::Status Potrf_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<PotrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const PotrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0], SizeOfType(d.type) * d.batch * d.n * d.n,
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[2]);
  void* workspace = buffers[3];
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSpotrf(handle.get(), d.uplo, d.n, a, d.n,
                             static_cast<float*>(workspace), d.lwork, info)));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDpotrf(handle.get(), d.uplo, d.n, a, d.n,
                             static_cast<double*>(workspace), d.lwork, info)));
        break;
      }
      case Type::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCpotrf(
            handle.get(), d.uplo, d.n, a, d.n,
            static_cast<cuComplex*>(workspace), d.lwork, info)));
        break;
      }
      case Type::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZpotrf(
            handle.get(), d.uplo, d.n, a, d.n,
            static_cast<cuDoubleComplex*>(workspace), d.lwork, info)));
        break;
      }
    }
  } else {
    auto buffer_ptrs_host = MakeBatchPointers(
        stream, buffers[1], workspace, d.batch, SizeOfType(d.type) * d.n * d.n);
    JAX_RETURN_IF_ERROR(buffer_ptrs_host.status());
    // Make sure that accesses to buffer_ptrs_host complete before we delete it.
    // TODO(phawkins): avoid synchronization here.
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaStreamSynchronize(stream)));
    switch (d.type) {
      case Type::F32: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<float**>(workspace), d.n,

            info, d.batch)));
        break;
      }
      case Type::F64: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<double**>(workspace), d.n,
            info, d.batch)));
        break;
      }
      case Type::C64: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<cuComplex**>(workspace), d.n,
            info, d.batch)));
        break;
      }
      case Type::C128: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZpotrfBatched(
            handle.get(), d.uplo, d.n,
            static_cast<cuDoubleComplex**>(workspace), d.n, info, d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Potrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Potrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

// getrf: LU decomposition

struct GetrfDescriptor {
  Type type;
  int batch, m, n;
};

// Returns the workspace size and a descriptor for a getrf operation.
std::pair<int, py::bytes> BuildGetrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case Type::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnSgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case Type::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnDgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case Type::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnCgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case Type::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnZgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(GetrfDescriptor{type, b, m, n})};
}

absl::Status Getrf_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<GetrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GetrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSgetrf(handle.get(), d.m, d.n, a, d.m,
                             static_cast<float*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDgetrf(handle.get(), d.m, d.n, a, d.m,
                             static_cast<double*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnCgetrf(handle.get(), d.m, d.n, a, d.m,
                             static_cast<cuComplex*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgetrf(
            handle.get(), d.m, d.n, a, d.m,
            static_cast<cuDoubleComplex*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Getrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Getrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

// geqrf: QR decomposition

struct GeqrfDescriptor {
  Type type;
  int batch, m, n, lwork;
};

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildGeqrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case Type::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnSgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case Type::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnDgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case Type::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnCgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case Type::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnZgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(GeqrfDescriptor{type, b, m, n, lwork})};
}

absl::Status Geqrf_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<GeqrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GeqrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* tau = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                             static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* tau = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                             static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      cuComplex* tau = static_cast<cuComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<cuComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      cuDoubleComplex* tau = static_cast<cuDoubleComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<cuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Geqrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Geqrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

// orgqr/ungqr: apply elementary Householder transformations

struct OrgqrDescriptor {
  Type type;
  int batch, m, n, k, lwork;
};

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildOrgqrDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, int k) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case Type::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnSorgqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
    case Type::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnDorgqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
    case Type::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnCungqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
    case Type::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnZungqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(OrgqrDescriptor{type, b, m, n, k, lwork})};
}

absl::Status Orgqr_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<OrgqrDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const OrgqrDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[2] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[2], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[2]);
      float* tau = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                             static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[2]);
      double* tau = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                             static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case Type::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[2]);
      cuComplex* tau = static_cast<cuComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<cuComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case Type::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[2]);
      cuDoubleComplex* tau = static_cast<cuDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<cuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Orgqr(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Orgqr_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

struct SyevdDescriptor {
  Type type;
  cublasFillMode_t uplo;
  int batch, n;
  int lwork;
};

// Returns the workspace size and a descriptor for a syevd operation.
std::pair<int, py::bytes> BuildSyevdDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo =
      lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  switch (type) {
    case Type::F32:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
    case Type::F64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
    case Type::C64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
    case Type::C128:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
  }
  return {lwork, PackDescriptor(SyevdDescriptor{type, uplo, b, n, lwork})};
}

absl::Status Syevd_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<SyevdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SyevdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
      cudaMemcpyDeviceToDevice, stream)));
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                             static_cast<float*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                             static_cast<double*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case Type::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnCheevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                             static_cast<cuComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case Type::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevd(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuDoubleComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Syevd(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Syevd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// Supports batches of matrices up to size 32.

struct SyevjDescriptor {
  Type type;
  cublasFillMode_t uplo;
  int batch, n;
  int lwork;
};

// Returns the workspace size and a descriptor for a syevj_batched operation.
std::pair<int, py::bytes> BuildSyevjDescriptor(const py::dtype& dtype,
                                               bool lower, int batch, int n) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  syevjInfo_t params;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateSyevjInfo(&params)));
  std::unique_ptr<syevjInfo, void (*)(syevjInfo*)> params_cleanup(
      params, [](syevjInfo* p) { cusolverDnDestroySyevjInfo(p); });
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo =
      lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  if (batch == 1) {
    switch (type) {
      case Type::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
      case Type::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
      case Type::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
      case Type::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
    }
  } else {
    switch (type) {
      case Type::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
      case Type::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
      case Type::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
      case Type::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
    }
  }
  return {lwork, PackDescriptor(SyevjDescriptor{type, uplo, batch, n, lwork})};
}

absl::Status Syevj_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<SyevjDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SyevjDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }
  syevjInfo_t params;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateSyevjInfo(&params)));
  std::unique_ptr<syevjInfo, void (*)(syevjInfo*)> params_cleanup(
      params, [](syevjInfo* p) { cusolverDnDestroySyevjInfo(p); });

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<float*>(work), d.lwork, info, params)));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<double*>(work), d.lwork, info, params)));
        break;
      }
      case Type::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuComplex*>(work), d.lwork, info, params)));
        break;
      }
      case Type::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuDoubleComplex*>(work), d.lwork, info, params)));
        break;
      }
    }
  } else {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<float*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<double*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case Type::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuComplex*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case Type::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnZheevjBatched(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                                    static_cast<cuDoubleComplex*>(work),
                                    d.lwork, info, params, d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Syevj(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Syevj_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

// Singular value decomposition using QR algorithm: gesvd

struct GesvdDescriptor {
  Type type;
  int batch, m, n;
  int lwork;
  signed char jobu, jobvt;
};

// Returns the workspace size and a descriptor for a gesvd operation.
std::pair<int, py::bytes> BuildGesvdDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, bool compute_uv,
                                               bool full_matrices) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case Type::F32:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnSgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
    case Type::F64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnDgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
    case Type::C64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnCgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
    case Type::C128:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnZgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
  }
  signed char jobu, jobvt;
  if (compute_uv) {
    if (full_matrices) {
      jobu = jobvt = 'A';
    } else {
      jobu = jobvt = 'S';
    }
  } else {
    jobu = jobvt = 'N';
  }
  return {lwork,
          PackDescriptor(GesvdDescriptor{type, b, m, n, lwork, jobu, jobvt})};
}

absl::Status Gesvd_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<GesvdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GesvdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      cudaMemcpyDeviceToDevice, stream)));
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      float* u = static_cast<float*>(buffers[3]);
      float* vt = static_cast<float*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<float*>(work), d.lwork,
            /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      double* u = static_cast<double*>(buffers[3]);
      double* vt = static_cast<double*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvd(
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
    case Type::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      cuComplex* u = static_cast<cuComplex*>(buffers[3]);
      cuComplex* vt = static_cast<cuComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<cuComplex*>(work), d.lwork, /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case Type::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      cuDoubleComplex* u = static_cast<cuDoubleComplex*>(buffers[3]);
      cuDoubleComplex* vt = static_cast<cuDoubleComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<cuDoubleComplex*>(work), d.lwork,
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

void Gesvd(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

// Singular value decomposition using Jacobi algorithm: gesvdj

struct GesvdjDescriptor {
  Type type;
  int batch, m, n;
  int lwork;
  cusolverEigMode_t jobz;
};

// Returns the workspace size and a descriptor for a gesvdj operation.
std::pair<int, py::bytes> BuildGesvdjDescriptor(const py::dtype& dtype,
                                                int batch, int m, int n,
                                                bool compute_uv) {
  Type type = DtypeToType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  cusolverEigMode_t jobz =
      compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t params;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateGesvdjInfo(&params)));
  std::unique_ptr<gesvdjInfo, void (*)(gesvdjInfo*)> params_cleanup(
      params, [](gesvdjInfo* p) { cusolverDnDestroyGesvdjInfo(p); });
  if (batch == 1) {
    switch (type) {
      case Type::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
      case Type::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
      case Type::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
      case Type::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
    }
  } else {
    switch (type) {
      case Type::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
      case Type::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
      case Type::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
      case Type::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
    }
  }
  return {lwork,
          PackDescriptor(GesvdjDescriptor{type, batch, m, n, lwork, jobz})};
}

absl::Status Gesvdj_(cudaStream_t stream, void** buffers, const char* opaque,
                     size_t opaque_len) {
  auto s = UnpackDescriptor<GesvdjDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GesvdjDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      cudaMemcpyDeviceToDevice, stream)));
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  gesvdjInfo_t params;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateGesvdjInfo(&params)));
  std::unique_ptr<gesvdjInfo, void (*)(gesvdjInfo*)> params_cleanup(
      params, [](gesvdjInfo* p) { cusolverDnDestroyGesvdjInfo(p); });
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<float*>(work), d.lwork, info, params)));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<double*>(work), d.lwork, info, params)));
        break;
      }
      case Type::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        cuComplex* u = static_cast<cuComplex*>(buffers[3]);
        cuComplex* v = static_cast<cuComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<cuComplex*>(work), d.lwork, info, params)));
        break;
      }
      case Type::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        cuDoubleComplex* u = static_cast<cuDoubleComplex*>(buffers[3]);
        cuDoubleComplex* v = static_cast<cuDoubleComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<cuDoubleComplex*>(work), d.lwork, info, params)));
        break;
      }
    }
  } else {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<float*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<double*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case Type::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        cuComplex* u = static_cast<cuComplex*>(buffers[3]);
        cuComplex* v = static_cast<cuComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<cuComplex*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case Type::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        cuDoubleComplex* u = static_cast<cuDoubleComplex*>(buffers[3]);
        cuDoubleComplex* v = static_cast<cuDoubleComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<cuDoubleComplex*>(work), d.lwork, info, params,
            d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Gesvdj(cudaStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvdj_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().length());
  }
}

py::dict Registrations() {
  py::dict dict;
  dict["cusolver_potrf"] = EncapsulateFunction(Potrf);
  dict["cusolver_getrf"] = EncapsulateFunction(Getrf);
  dict["cusolver_geqrf"] = EncapsulateFunction(Geqrf);
  dict["cusolver_orgqr"] = EncapsulateFunction(Orgqr);
  dict["cusolver_syevd"] = EncapsulateFunction(Syevd);
  dict["cusolver_syevj"] = EncapsulateFunction(Syevj);
  dict["cusolver_gesvd"] = EncapsulateFunction(Gesvd);
  dict["cusolver_gesvdj"] = EncapsulateFunction(Gesvdj);
  return dict;
}

PYBIND11_MODULE(cusolver_kernels, m) {
  m.def("registrations", &Registrations);
  m.def("build_potrf_descriptor", &BuildPotrfDescriptor);
  m.def("build_getrf_descriptor", &BuildGetrfDescriptor);
  m.def("build_geqrf_descriptor", &BuildGeqrfDescriptor);
  m.def("build_orgqr_descriptor", &BuildOrgqrDescriptor);
  m.def("build_syevd_descriptor", &BuildSyevdDescriptor);
  m.def("build_syevj_descriptor", &BuildSyevjDescriptor);
  m.def("build_gesvd_descriptor", &BuildGesvdDescriptor);
  m.def("build_gesvdj_descriptor", &BuildGesvdjDescriptor);
}

}  // namespace
}  // namespace jax
