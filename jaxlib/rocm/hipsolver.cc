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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "jaxlib/rocm/hip_gpu_kernel_helpers.h"
#include "jaxlib/rocm/hipsolver_kernels.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/hipsolver.h"

namespace jax {
namespace {
namespace py = pybind11;

// Converts a NumPy dtype to a Type.
HipsolverType DtypeToHipsolverType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, HipsolverType>({
          {{'f', 4}, HipsolverType::F32},
          {{'f', 8}, HipsolverType::F64},
          {{'c', 8}, HipsolverType::C64},
          {{'c', 16}, HipsolverType::C128},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", py::repr(np_type)));
  }
  return it->second;
}

// potrf: Cholesky decomposition

// Returns the workspace size and a descriptor for a potrf operation.
std::pair<int, py::bytes> BuildPotrfDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  HipsolverType type = DtypeToHipsolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  std::int64_t workspace_size;
  hipsolverFillMode_t uplo =
      lower ? HIPSOLVER_FILL_MODE_LOWER : HIPSOLVER_FILL_MODE_UPPER;
  if (b == 1) {
    switch (type) {
      case HipsolverType::F32:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverSpotrf_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(float);
        break;
      case HipsolverType::F64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverDpotrf_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(double);
        break;
      case HipsolverType::C64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverCpotrf_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(hipComplex);
        break;
      case HipsolverType::C128:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverZpotrf_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(hipDoubleComplex);
        break;
    }
  } else {
    // TODO(rocm): when cuda and hip had same API for batched potrf, remove this
    // batched potrf has different API compared to CUDA. In hip we still need to create the workspace and additional space to copy the batch array pointers 
    switch (type) {
      case HipsolverType::F32:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverSpotrfBatched_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork, b)));
        workspace_size = (lwork * sizeof(float)) + (b * sizeof(float*));
        break;
      case HipsolverType::F64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverDpotrfBatched_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork, b)));
        workspace_size = (lwork * sizeof(double)) + (b * sizeof(double*));
        break;
      case HipsolverType::C64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverCpotrfBatched_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork, b)));
        workspace_size = (lwork * sizeof(hipComplex)) + (b * sizeof(hipComplex*));
        break;
      case HipsolverType::C128:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(hipsolverZpotrfBatched_bufferSize(handle.get(), uplo, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/n, &lwork, b)));
        workspace_size = (lwork * sizeof(hipDoubleComplex)) + (b * sizeof(hipDoubleComplex*));
        break;
    }

  }
  return {workspace_size,
          PackDescriptor(PotrfDescriptor{type, uplo, b, n, lwork})};
}

// getrf: LU decomposition

// Returns the workspace size and a descriptor for a getrf operation.
std::pair<int, py::bytes> BuildGetrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  HipsolverType type = DtypeToHipsolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case HipsolverType::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverSgetrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
    case HipsolverType::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverDgetrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
    case HipsolverType::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverCgetrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
    case HipsolverType::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverZgetrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(GetrfDescriptor{type, b, m, n, lwork})};
}

// geqrf: QR decomposition

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildGeqrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  HipsolverType type = DtypeToHipsolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case HipsolverType::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverSgeqrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
    case HipsolverType::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverDgeqrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
    case HipsolverType::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverCgeqrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
    case HipsolverType::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverZgeqrf_bufferSize(handle.get(), m, n,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(GeqrfDescriptor{type, b, m, n, lwork})};
}

// orgqr/ungqr: apply elementary Householder transformations

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildOrgqrDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, int k) {
  HipsolverType type = DtypeToHipsolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case HipsolverType::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverSorgqr_bufferSize(handle.get(), m, n, k,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m,
                                                   /*tau=*/nullptr, &lwork)));
      break;
    case HipsolverType::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverDorgqr_bufferSize(handle.get(), m, n, k,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m,
                                                   /*tau=*/nullptr, &lwork)));
      break;
    case HipsolverType::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverCungqr_bufferSize(handle.get(), m, n, k,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m,
                                                   /*tau=*/nullptr, &lwork)));
      break;
    case HipsolverType::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(hipsolverZungqr_bufferSize(handle.get(), m, n, k,
                                                   /*A=*/nullptr,
                                                   /*lda=*/m,
                                                   /*tau=*/nullptr, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(OrgqrDescriptor{type, b, m, n, k, lwork})};
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

// Returns the workspace size and a descriptor for a syevd operation.
std::pair<int, py::bytes> BuildSyevdDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  HipsolverType type = DtypeToHipsolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  hipsolverEigMode_t jobz = HIPSOLVER_EIG_MODE_VECTOR;
  hipsolverFillMode_t uplo =
      lower ? HIPSOLVER_FILL_MODE_LOWER : HIPSOLVER_FILL_MODE_UPPER;
  switch (type) {
    case HipsolverType::F32:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          hipsolverSsyevd_bufferSize(handle.get(), jobz, uplo, n, /*A=*/nullptr,
                                     /*lda=*/n, /*W=*/nullptr, &lwork)));
      break;
    case HipsolverType::F64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          hipsolverDsyevd_bufferSize(handle.get(), jobz, uplo, n, /*A=*/nullptr,
                                     /*lda=*/n, /*W=*/nullptr, &lwork)));
      break;
    case HipsolverType::C64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          hipsolverCheevd_bufferSize(handle.get(), jobz, uplo, n, /*A=*/nullptr,
                                     /*lda=*/n, /*W=*/nullptr, &lwork)));
      break;
    case HipsolverType::C128:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsolverZheevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
  }
  return {lwork, PackDescriptor(SyevdDescriptor{type, uplo, b, n, lwork})};
}

// Singular value decomposition using QR algorithm: gesvd

// Returns the workspace size and a descriptor for a gesvd operation.
std::pair<int, py::bytes> BuildGesvdDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, bool compute_uv,
                                               bool full_matrices) {
  HipsolverType type = DtypeToHipsolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
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
  switch (type) {
    case HipsolverType::F32:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          hipsolverSgesvd_bufferSize(handle.get(), jobu, jobvt, m, n, &lwork)));
      break;
    case HipsolverType::F64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          hipsolverDgesvd_bufferSize(handle.get(), jobu, jobvt, m, n, &lwork)));
      break;
    case HipsolverType::C64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          hipsolverCgesvd_bufferSize(handle.get(), jobu, jobvt, m, n, &lwork)));
      break;
    case HipsolverType::C128:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          hipsolverZgesvd_bufferSize(handle.get(), jobu, jobvt, m, n, &lwork)));
      break;
  }
  return {lwork,
          PackDescriptor(GesvdDescriptor{type, b, m, n, lwork, jobu, jobvt})};
}

py::dict Registrations() {
  py::dict dict;
  dict["hipsolver_potrf"] = EncapsulateFunction(Potrf);
  dict["hipsolver_getrf"] = EncapsulateFunction(Getrf);
  dict["hipsolver_geqrf"] = EncapsulateFunction(Geqrf);
  dict["hipsolver_orgqr"] = EncapsulateFunction(Orgqr);
  dict["hipsolver_syevd"] = EncapsulateFunction(Syevd);
  // dict["cusolver_syevj"] = EncapsulateFunction(Syevj); not supported by
  // ROCm yet
  dict["hipsolver_gesvd"] = EncapsulateFunction(Gesvd);
  // dict["cusolver_gesvdj"] = EncapsulateFunction(Gesvdj);  not supported by
  // ROCm yet
  return dict;
}

PYBIND11_MODULE(_hipsolver, m) {
  m.def("registrations", &Registrations);
  m.def("build_potrf_descriptor", &BuildPotrfDescriptor);
  m.def("build_getrf_descriptor", &BuildGetrfDescriptor);
  m.def("build_geqrf_descriptor", &BuildGeqrfDescriptor);
  m.def("build_orgqr_descriptor", &BuildOrgqrDescriptor);
  m.def("build_syevd_descriptor", &BuildSyevdDescriptor);
  // m.def("build_syevj_descriptor", &BuildSyevjDescriptor); not supported by
  // ROCm yet
  m.def("build_gesvd_descriptor", &BuildGesvdDescriptor);
  // m.def("build_gesvdj_descriptor", &BuildGesvdjDescriptor); not supported by
  // ROCm yet
}

}  // namespace
}  // namespace jax
