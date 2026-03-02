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

#include "jaxlib/gpu/solver_interface.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"

#ifdef JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace solver {

// LU decomposition: getrf

#define JAX_GPU_DEFINE_GETRF(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> GetrfBufferSize<Type>(gpusolverDnHandle_t handle, int m, \
                                            int n) {                           \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                         \
        Name##_bufferSize(handle, m, n, /*A=*/nullptr, m, &lwork)));           \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Getrf<Type>(gpusolverDnHandle_t handle, int m, int n, Type *a,  \
                           Type *workspace, int lwork, int *ipiv, int *info) { \
    return JAX_AS_STATUS(                                                      \
        Name(handle, m, n, a, m, workspace, lwork, ipiv, info));               \
  }

JAX_GPU_DEFINE_GETRF(float, gpusolverDnSgetrf);
JAX_GPU_DEFINE_GETRF(double, gpusolverDnDgetrf);
JAX_GPU_DEFINE_GETRF(gpuComplex, gpusolverDnCgetrf);
JAX_GPU_DEFINE_GETRF(gpuDoubleComplex, gpusolverDnZgetrf);
#undef JAX_GPU_DEFINE_GETRF

#define JAX_GPU_DEFINE_GETRF_BATCHED(Type, Name)                              \
  template <>                                                                 \
  absl::Status GetrfBatched<Type>(gpublasHandle_t handle, int n, Type **a,    \
                                  int lda, int *ipiv, int *info, int batch) { \
    return JAX_AS_STATUS(Name(handle, n, a, lda, ipiv, info, batch));         \
  }

JAX_GPU_DEFINE_GETRF_BATCHED(float, gpublasSgetrfBatched);
JAX_GPU_DEFINE_GETRF_BATCHED(double, gpublasDgetrfBatched);
JAX_GPU_DEFINE_GETRF_BATCHED(gpublasComplex, gpublasCgetrfBatched);
JAX_GPU_DEFINE_GETRF_BATCHED(gpublasDoubleComplex, gpublasZgetrfBatched);
#undef JAX_GPU_DEFINE_GETRF_BATCHED

// QR decomposition: geqrf

#define JAX_GPU_DEFINE_GEQRF(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> GeqrfBufferSize<Type>(gpusolverDnHandle_t handle, int m, \
                                            int n) {                           \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                         \
        Name##_bufferSize(handle, m, n, /*A=*/nullptr, m, &lwork)));           \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Geqrf<Type>(gpusolverDnHandle_t handle, int m, int n, Type *a,  \
                           Type *tau, Type *workspace, int lwork, int *info) { \
    return JAX_AS_STATUS(                                                      \
        Name(handle, m, n, a, m, tau, workspace, lwork, info));                \
  }

JAX_GPU_DEFINE_GEQRF(float, gpusolverDnSgeqrf);
JAX_GPU_DEFINE_GEQRF(double, gpusolverDnDgeqrf);
JAX_GPU_DEFINE_GEQRF(gpuComplex, gpusolverDnCgeqrf);
JAX_GPU_DEFINE_GEQRF(gpuDoubleComplex, gpusolverDnZgeqrf);
#undef JAX_GPU_DEFINE_GEQRF

#define JAX_GPU_DEFINE_GEQRF_BATCHED(Type, Name)                        \
  template <>                                                           \
  absl::Status GeqrfBatched<Type>(gpublasHandle_t handle, int m, int n, \
                                  Type **a, Type **tau, int *info,      \
                                  int batch) {                          \
    return JAX_AS_STATUS(Name(handle, m, n, a, m, tau, info, batch));   \
  }

JAX_GPU_DEFINE_GEQRF_BATCHED(float, gpublasSgeqrfBatched);
JAX_GPU_DEFINE_GEQRF_BATCHED(double, gpublasDgeqrfBatched);
JAX_GPU_DEFINE_GEQRF_BATCHED(gpublasComplex, gpublasCgeqrfBatched);
JAX_GPU_DEFINE_GEQRF_BATCHED(gpublasDoubleComplex, gpublasZgeqrfBatched);
#undef JAX_GPU_DEFINE_GEQRF_BATCHED

// Householder transformations: orgqr

#define JAX_GPU_DEFINE_ORGQR(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> OrgqrBufferSize<Type>(gpusolverDnHandle_t handle, int m, \
                                            int n, int k) {                    \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(Name##_bufferSize(                       \
        handle, m, n, k, /*A=*/nullptr, /*lda=*/m, /*tau=*/nullptr, &lwork))); \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Orgqr<Type>(gpusolverDnHandle_t handle, int m, int n, int k,    \
                           Type *a, Type *tau, Type *workspace, int lwork,     \
                           int *info) {                                        \
    return JAX_AS_STATUS(                                                      \
        Name(handle, m, n, k, a, m, tau, workspace, lwork, info));             \
  }

JAX_GPU_DEFINE_ORGQR(float, gpusolverDnSorgqr);
JAX_GPU_DEFINE_ORGQR(double, gpusolverDnDorgqr);
JAX_GPU_DEFINE_ORGQR(gpuComplex, gpusolverDnCungqr);
JAX_GPU_DEFINE_ORGQR(gpuDoubleComplex, gpusolverDnZungqr);
#undef JAX_GPU_DEFINE_ORGQR

// Cholesky decomposition: potrf

#define JAX_GPU_DEFINE_POTRF(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> PotrfBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            gpusolverFillMode_t uplo, int n) { \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                         \
        Name##_bufferSize(handle, uplo, n, /*A=*/nullptr, n, &lwork)));        \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Potrf<Type>(gpusolverDnHandle_t handle,                         \
                           gpusolverFillMode_t uplo, int n, Type *a,           \
                           Type *workspace, int lwork, int *info) {            \
    return JAX_AS_STATUS(                                                      \
        Name(handle, uplo, n, a, n, workspace, lwork, info));                  \
  }

JAX_GPU_DEFINE_POTRF(float, gpusolverDnSpotrf);
JAX_GPU_DEFINE_POTRF(double, gpusolverDnDpotrf);
JAX_GPU_DEFINE_POTRF(gpuComplex, gpusolverDnCpotrf);
JAX_GPU_DEFINE_POTRF(gpuDoubleComplex, gpusolverDnZpotrf);
#undef JAX_GPU_DEFINE_POTRF

#define JAX_GPU_DEFINE_POTRF_BATCHED(Type, Name)                               \
  template <>                                                                  \
  absl::Status PotrfBatched<Type>(gpusolverDnHandle_t handle,                  \
                                  gpusolverFillMode_t uplo, int n, Type **a,   \
                                  int lda, int *info, int batch) {             \
    return JAX_AS_STATUS(Name(handle, uplo, n, a, lda, info, batch));         \
  }

JAX_GPU_DEFINE_POTRF_BATCHED(float, gpusolverDnSpotrfBatched);
JAX_GPU_DEFINE_POTRF_BATCHED(double, gpusolverDnDpotrfBatched);
JAX_GPU_DEFINE_POTRF_BATCHED(gpuComplex, gpusolverDnCpotrfBatched);
JAX_GPU_DEFINE_POTRF_BATCHED(gpuDoubleComplex, gpusolverDnZpotrfBatched);
#undef JAX_GPU_DEFINE_POTRF_BATCHED

// Symmetric (Hermitian) eigendecomposition:
// * Jacobi algorithm: syevj/heevj (batches of matrices up to 32)
// * QR algorithm: syevd/heevd

#define JAX_GPU_DEFINE_SYEVJ(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> SyevjBufferSize<Type>(                                   \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                     \
      gpusolverFillMode_t uplo, int n, gpuSyevjInfo_t params) {                \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                         \
        Name##_bufferSize(handle, jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,     \
                          /*w=*/nullptr, &lwork, params)));                    \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Syevj<Type>(                                                    \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                     \
      gpusolverFillMode_t uplo, int n, Type *a, RealType<Type>::value *w,      \
      Type *workspace, int lwork, int *info, gpuSyevjInfo_t params) {          \
    return JAX_AS_STATUS(                                                      \
        Name(handle, jobz, uplo, n, a, n, w, workspace, lwork, info, params)); \
  }

JAX_GPU_DEFINE_SYEVJ(float, gpusolverDnSsyevj);
JAX_GPU_DEFINE_SYEVJ(double, gpusolverDnDsyevj);
JAX_GPU_DEFINE_SYEVJ(gpuComplex, gpusolverDnCheevj);
JAX_GPU_DEFINE_SYEVJ(gpuDoubleComplex, gpusolverDnZheevj);
#undef JAX_GPU_DEFINE_SYEVJ

#define JAX_GPU_DEFINE_SYEVJ_BATCHED(Type, Name)                           \
  template <>                                                              \
  absl::StatusOr<int> SyevjBatchedBufferSize<Type>(                        \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                 \
      gpusolverFillMode_t uplo, int n, gpuSyevjInfo_t params, int batch) { \
    int lwork;                                                             \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                     \
        Name##_bufferSize(handle, jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, \
                          /*w=*/nullptr, &lwork, params, batch)));         \
    return lwork;                                                          \
  }                                                                        \
                                                                           \
  template <>                                                              \
  absl::Status SyevjBatched<Type>(                                         \
      gpusolverDnHandle_t handle, gpusolverEigMode_t jobz,                 \
      gpusolverFillMode_t uplo, int n, Type *a, RealType<Type>::value *w,  \
      Type *workspace, int lwork, int *info, gpuSyevjInfo_t params,        \
      int batch) {                                                         \
    return JAX_AS_STATUS(Name(handle, jobz, uplo, n, a, n, w, workspace,   \
                              lwork, info, params, batch));                \
  }

JAX_GPU_DEFINE_SYEVJ_BATCHED(float, gpusolverDnSsyevjBatched);
JAX_GPU_DEFINE_SYEVJ_BATCHED(double, gpusolverDnDsyevjBatched);
JAX_GPU_DEFINE_SYEVJ_BATCHED(gpuComplex, gpusolverDnCheevjBatched);
JAX_GPU_DEFINE_SYEVJ_BATCHED(gpuDoubleComplex, gpusolverDnZheevjBatched);
#undef JAX_GPU_DEFINE_SYEVJ_BATCHED

#define JAX_GPU_DEFINE_SYEVD(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> SyevdBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            gpusolverEigMode_t jobz,           \
                                            gpusolverFillMode_t uplo, int n) { \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(                                                       \
        JAX_AS_STATUS(Name##_bufferSize(handle, jobz, uplo, n, /*A=*/nullptr,  \
                                        /*lda=*/n, /*w=*/nullptr, &lwork)));   \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Syevd<Type>(gpusolverDnHandle_t handle,                         \
                           gpusolverEigMode_t jobz, gpusolverFillMode_t uplo,  \
                           int n, Type *a, RealType<Type>::value *w,           \
                           Type *workspace, int lwork, int *info) {            \
    return JAX_AS_STATUS(                                                      \
        Name(handle, jobz, uplo, n, a, n, w, workspace, lwork, info));         \
  }

JAX_GPU_DEFINE_SYEVD(float, gpusolverDnSsyevd);
JAX_GPU_DEFINE_SYEVD(double, gpusolverDnDsyevd);
JAX_GPU_DEFINE_SYEVD(gpuComplex, gpusolverDnCheevd);
JAX_GPU_DEFINE_SYEVD(gpuDoubleComplex, gpusolverDnZheevd);
#undef JAX_GPU_DEFINE_SYEVD

// Symmetric rank-k update: syrk

#define JAX_GPU_DEFINE_SYRK(Type, Name)                                       \
  template <>                                                                 \
  absl::Status Syrk<Type>(gpublasHandle_t handle, gpublasFillMode_t uplo,     \
                          gpublasOperation_t trans, int n, int k,             \
                          const Type *alpha, const Type *a, const Type *beta, \
                          Type *c) {                                          \
    int lda = trans == GPUBLAS_OP_N ? n : k;                                  \
    return JAX_AS_STATUS(                                                     \
        Name(handle, uplo, trans, n, k, alpha, a, lda, beta, c, n));          \
  }

JAX_GPU_DEFINE_SYRK(float, gpublasSsyrk);
JAX_GPU_DEFINE_SYRK(double, gpublasDsyrk);
JAX_GPU_DEFINE_SYRK(gpublasComplex, gpublasCsyrk);
JAX_GPU_DEFINE_SYRK(gpublasDoubleComplex, gpublasZsyrk);
#undef JAX_GPU_DEFINE_SYRK

// Singular Value Decomposition: gesvd

#define JAX_GPU_DEFINE_GESVD(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> GesvdBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            signed char job, int m, int n) {   \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(                                                       \
        JAX_AS_STATUS(Name##_bufferSize(handle, job, job, m, n, &lwork)));     \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Gesvd<Type>(gpusolverDnHandle_t handle, signed char job, int m, \
                           int n, Type *a, RealType<Type>::value *s, Type *u,  \
                           Type *vt, Type *workspace, int lwork, int *info) {  \
    return JAX_AS_STATUS(Name(handle, job, job, m, n, a, m, s, u, m, vt, n,    \
                              workspace, lwork, /*rwork=*/nullptr, info));     \
  }

JAX_GPU_DEFINE_GESVD(float, gpusolverDnSgesvd);
JAX_GPU_DEFINE_GESVD(double, gpusolverDnDgesvd);
JAX_GPU_DEFINE_GESVD(gpuComplex, gpusolverDnCgesvd);
JAX_GPU_DEFINE_GESVD(gpuDoubleComplex, gpusolverDnZgesvd);
#undef JAX_GPU_DEFINE_GESVD

#define JAX_GPU_DEFINE_GESVDJ(Type, Name)                                      \
  template <>                                                                  \
  absl::StatusOr<int> GesvdjBufferSize<Type>(                                  \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int econ, int m,     \
      int n, gpuGesvdjInfo_t params) {                                         \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(Name##_bufferSize(                       \
        handle, job, econ, m, n, /*a=*/nullptr, /*lda=*/m, /*s=*/nullptr,      \
        /*u=*/nullptr, /*ldu=*/m, /*v=*/nullptr, /*ldv=*/n, &lwork, params))); \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Gesvdj<Type>(                                                   \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int econ, int m,     \
      int n, Type *a, RealType<Type>::value *s, Type *u, Type *v,              \
      Type *workspace, int lwork, int *info, gpuGesvdjInfo_t params) {         \
    return JAX_AS_STATUS(Name(handle, job, econ, m, n, a, m, s, u, m, v, n,    \
                              workspace, lwork, info, params));                \
  }

JAX_GPU_DEFINE_GESVDJ(float, gpusolverDnSgesvdj);
JAX_GPU_DEFINE_GESVDJ(double, gpusolverDnDgesvdj);
JAX_GPU_DEFINE_GESVDJ(gpuComplex, gpusolverDnCgesvdj);
JAX_GPU_DEFINE_GESVDJ(gpuDoubleComplex, gpusolverDnZgesvdj);
#undef JAX_GPU_DEFINE_GESVDJ

#define JAX_GPU_DEFINE_GESVDJ_BATCHED(Type, Name)                             \
  template <>                                                                 \
  absl::StatusOr<int> GesvdjBatchedBufferSize<Type>(                          \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int m, int n,       \
      gpuGesvdjInfo_t params, int batch) {                                    \
    int lwork;                                                                \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                        \
        Name##_bufferSize(handle, job, m, n, /*a=*/nullptr, /*lda=*/m,        \
                          /*s=*/nullptr, /*u=*/nullptr, /*ldu=*/m,            \
                          /*v=*/nullptr, /*ldv=*/n, &lwork, params, batch))); \
    return lwork;                                                             \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  absl::Status GesvdjBatched<Type>(                                           \
      gpusolverDnHandle_t handle, gpusolverEigMode_t job, int m, int n,       \
      Type *a, RealType<Type>::value *s, Type *u, Type *v, Type *workspace,   \
      int lwork, int *info, gpuGesvdjInfo_t params, int batch) {              \
    return JAX_AS_STATUS(Name(handle, job, m, n, a, m, s, u, m, v, n,         \
                              workspace, lwork, info, params, batch));        \
  }

JAX_GPU_DEFINE_GESVDJ_BATCHED(float, gpusolverDnSgesvdjBatched);
JAX_GPU_DEFINE_GESVDJ_BATCHED(double, gpusolverDnDgesvdjBatched);
JAX_GPU_DEFINE_GESVDJ_BATCHED(gpuComplex, gpusolverDnCgesvdjBatched);
JAX_GPU_DEFINE_GESVDJ_BATCHED(gpuDoubleComplex, gpusolverDnZgesvdjBatched);
#undef JAX_GPU_DEFINE_GESVDJ_BATCHED

#ifdef JAX_GPU_HIP
// GESDD (divide-and-conquer SVD) is provided by rocsolver; hipSOLVER does not
// expose it. rocsolver uses rocblas_handle; on ROCm the solver handle is
// compatible with rocblas_handle.
#include "rocm/include/rocsolver/rocsolver.h"

namespace {
rocblas_svect JobToRocblasSvect(signed char job) {
  switch (job) {
    case 'A':
      return rocblas_svect_all;
    case 'S':
      return rocblas_svect_singular;
    case 'N':
    default:
      return rocblas_svect_none;
  }
}

absl::Status RocblasStatusToStatus(rocblas_status status, const char* file,
                                  int line, const char* expr) {
  if (ABSL_PREDICT_FALSE(status != rocblas_status_success)) {
    return absl::InternalError(
        absl::StrFormat("%s:%d: %s failed: rocblas_status %d", file, line, expr,
                        static_cast<int>(status)));
  }
  return absl::OkStatus();
}
}  // namespace

#define JAX_GPU_DEFINE_GESDD_REAL(Type, Name)                                  \
  template <>                                                                  \
  absl::Status Gesdd<Type>(                                                    \
      gpusolverDnHandle_t handle, signed char jobu, signed char jobvt,        \
      int m, int n, Type *a, int lda, RealType<Type>::value *s, Type *u,      \
      int ldu, Type *v, int ldv, int *info) {                                  \
    auto h = reinterpret_cast<rocblas_handle>(handle);                        \
    rocblas_status st = Name(h, JobToRocblasSvect(jobu),                       \
                             JobToRocblasSvect(jobvt), m, n, a, lda, s, u,    \
                             ldu, v, ldv, info);                               \
    return RocblasStatusToStatus(st, __FILE__, __LINE__, #Name);               \
  }

JAX_GPU_DEFINE_GESDD_REAL(float, rocsolver_sgesdd);
JAX_GPU_DEFINE_GESDD_REAL(double, rocsolver_dgesdd);
#undef JAX_GPU_DEFINE_GESDD_REAL

template <>
absl::Status Gesdd<gpuComplex>(
    gpusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
    int n, gpuComplex *a, int lda, float *s, gpuComplex *u, int ldu,
    gpuComplex *v, int ldv, int *info) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  rocblas_status st = rocsolver_cgesdd(
      h, JobToRocblasSvect(jobu), JobToRocblasSvect(jobvt), m, n,
      reinterpret_cast<rocblas_float_complex*>(a), lda, s,
      reinterpret_cast<rocblas_float_complex*>(u), ldu,
      reinterpret_cast<rocblas_float_complex*>(v), ldv, info);
  return RocblasStatusToStatus(st, __FILE__, __LINE__, "rocsolver_cgesdd");
}

template <>
absl::Status Gesdd<gpuDoubleComplex>(
    gpusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m,
    int n, gpuDoubleComplex *a, int lda, double *s, gpuDoubleComplex *u,
    int ldu, gpuDoubleComplex *v, int ldv, int *info) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  rocblas_status st = rocsolver_zgesdd(
      h, JobToRocblasSvect(jobu), JobToRocblasSvect(jobvt), m, n,
      reinterpret_cast<rocblas_double_complex*>(a), lda, s,
      reinterpret_cast<rocblas_double_complex*>(u), ldu,
      reinterpret_cast<rocblas_double_complex*>(v), ldv, info);
  return RocblasStatusToStatus(st, __FILE__, __LINE__, "rocsolver_zgesdd");
}

// Workspace size query and set for rocsolver (two-phase memory model).
absl::Status SetWorkspace(gpusolverDnHandle_t handle, void* ptr, size_t size) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  rocblas_status st = rocblas_set_workspace(h, ptr, size);
  return RocblasStatusToStatus(st, __FILE__, __LINE__, "rocblas_set_workspace");
}

namespace {
template <typename T>
rocblas_status GesddQueryImpl(gpusolverDnHandle_t handle, signed char jobu,
                              signed char jobvt, int m, int n);
template <>
rocblas_status GesddQueryImpl<float>(gpusolverDnHandle_t handle, signed char jobu,
                                     signed char jobvt, int m, int n) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  return rocsolver_sgesdd(h, JobToRocblasSvect(jobu), JobToRocblasSvect(jobvt),
                          m, n, nullptr, m, nullptr, nullptr, m, nullptr, n,
                          nullptr);
}
template <>
rocblas_status GesddQueryImpl<double>(gpusolverDnHandle_t handle, signed char jobu,
                                      signed char jobvt, int m, int n) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  return rocsolver_dgesdd(h, JobToRocblasSvect(jobu), JobToRocblasSvect(jobvt),
                          m, n, nullptr, m, nullptr, nullptr, m, nullptr, n,
                          nullptr);
}
template <>
rocblas_status GesddQueryImpl<gpuComplex>(gpusolverDnHandle_t handle,
                                          signed char jobu, signed char jobvt,
                                          int m, int n) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  return rocsolver_cgesdd(h, JobToRocblasSvect(jobu), JobToRocblasSvect(jobvt),
                          m, n, nullptr, m, nullptr, nullptr, m, nullptr, n,
                          nullptr);
}
template <>
rocblas_status GesddQueryImpl<gpuDoubleComplex>(gpusolverDnHandle_t handle,
                                                 signed char jobu,
                                                 signed char jobvt, int m,
                                                 int n) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  return rocsolver_zgesdd(h, JobToRocblasSvect(jobu), JobToRocblasSvect(jobvt),
                          m, n, nullptr, m, nullptr, nullptr, m, nullptr, n,
                          nullptr);
}
}  // namespace

template <typename T>
absl::StatusOr<size_t> GesddWorkspaceSize(gpusolverDnHandle_t handle,
                                         signed char jobu, signed char jobvt,
                                         int m, int n) {
  auto h = reinterpret_cast<rocblas_handle>(handle);
  rocblas_status st = rocblas_start_device_memory_size_query(h);
  JAX_RETURN_IF_ERROR(RocblasStatusToStatus(st, __FILE__, __LINE__,
                                            "rocblas_start_device_memory_size_query"));
  st = GesddQueryImpl<T>(handle, jobu, jobvt, m, n);
  if (st != rocblas_status_success) {
    (void)rocblas_stop_device_memory_size_query(h, nullptr);
    return RocblasStatusToStatus(st, __FILE__, __LINE__, "rocsolver_*gesdd (query)");
  }
  size_t size = 0;
  st = rocblas_stop_device_memory_size_query(h, &size);
  JAX_RETURN_IF_ERROR(RocblasStatusToStatus(st, __FILE__, __LINE__,
                                            "rocblas_stop_device_memory_size_query"));
  return size;
}

template absl::StatusOr<size_t> GesddWorkspaceSize<float>(gpusolverDnHandle_t,
                                                          signed char, signed char,
                                                          int, int);
template absl::StatusOr<size_t> GesddWorkspaceSize<double>(gpusolverDnHandle_t,
                                                           signed char, signed char,
                                                           int, int);
template absl::StatusOr<size_t> GesddWorkspaceSize<gpuComplex>(gpusolverDnHandle_t,
                                                               signed char, signed char,
                                                               int, int);
template absl::StatusOr<size_t> GesddWorkspaceSize<gpuDoubleComplex>(
    gpusolverDnHandle_t, signed char, signed char, int, int);
#endif  // JAX_GPU_HIP

#ifdef JAX_GPU_CUDA

#define JAX_GPU_DEFINE_CSRLSVQR(Type, Scalar, Name)                          \
  template <>                                                                \
  absl::Status Csrlsvqr<Type>(                                               \
      cusolverSpHandle_t handle, int n, int nnz, cusparseMatDescr_t matdesc, \
      const Type *csrValA, const int *csrRowPtrA, const int *csrColIndA,     \
      const Type *b, double tol, int reorder, Type *x, int *singularity) {   \
    return JAX_AS_STATUS(Name(handle, n, nnz, matdesc, csrValA, csrRowPtrA,  \
                              csrColIndA, b, static_cast<Scalar>(tol),       \
                              reorder, x, singularity));                     \
  }

JAX_GPU_DEFINE_CSRLSVQR(float, float, cusolverSpScsrlsvqr);
JAX_GPU_DEFINE_CSRLSVQR(double, double, cusolverSpDcsrlsvqr);
JAX_GPU_DEFINE_CSRLSVQR(gpuComplex, float, cusolverSpCcsrlsvqr);
JAX_GPU_DEFINE_CSRLSVQR(gpuDoubleComplex, double, cusolverSpZcsrlsvqr);
#undef JAX_GPU_DEFINE_CSRLSVQR

#endif  // JAX_GPU_CUDA

// Symmetric tridiagonal reduction: sytrd

#define JAX_GPU_DEFINE_SYTRD(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> SytrdBufferSize<Type>(gpusolverDnHandle_t handle,        \
                                            gpusolverFillMode_t uplo, int n) { \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(Name##_bufferSize(                       \
        handle, uplo, n, /*A=*/nullptr, /*lda=*/n, /*D=*/nullptr,              \
        /*E=*/nullptr, /*tau=*/nullptr, &lwork)));                             \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Sytrd<Type>(gpusolverDnHandle_t handle,                         \
                           gpusolverFillMode_t uplo, int n, Type *a,           \
                           RealType<Type>::value *d, RealType<Type>::value *e, \
                           Type *tau, Type *workspace, int lwork, int *info) { \
    return JAX_AS_STATUS(                                                      \
        Name(handle, uplo, n, a, n, d, e, tau, workspace, lwork, info));       \
  }

JAX_GPU_DEFINE_SYTRD(float, gpusolverDnSsytrd);
JAX_GPU_DEFINE_SYTRD(double, gpusolverDnDsytrd);
JAX_GPU_DEFINE_SYTRD(gpuComplex, gpusolverDnChetrd);
JAX_GPU_DEFINE_SYTRD(gpuDoubleComplex, gpusolverDnZhetrd);
#undef JAX_GPU_DEFINE_SYTRD

}  // namespace solver
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
