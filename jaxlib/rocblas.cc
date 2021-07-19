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
#include <stdexcept>
#include <utility>
#include <vector>


#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "jaxlib/rocm_gpu_kernel_helpers.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/rocblas.h"
#include "rocm/include/rocsolver.h"


namespace jax {
namespace {

namespace py = pybind11;

void ThrowIfErrorStatus(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return;
    default:
      throw std::runtime_error(rocblas_status_to_string(status));
  }
}

using rocBlasHandlePool = HandlePool<rocblas_handle, hipStream_t>;

template <>
/*static*/ rocBlasHandlePool::Handle rocBlasHandlePool::Borrow(
    hipStream_t stream) {
  rocBlasHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  rocblas_handle handle;
  if (pool->handles_[stream].empty()) {
    ThrowIfErrorStatus(rocblas_create_handle(&handle));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    ThrowIfErrorStatus(rocblas_set_stream(handle, stream));
  }
  return rocBlasHandlePool::Handle(pool, handle, stream);
}

// Set of types known to Rocsolver.
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
      return sizeof(rocblas_float_complex);
    case Type::C128:
      return sizeof(rocblas_double_complex);
  }
}

// the buffers[] are all allocated in rocsolver.py
// where the number of buffers and their size is determined / hardcoded as
// expected here

//##########################
// rocblas
//##########################

// Batched triangular solve: Trsm

struct TrsmDescriptor {
  Type type;
  int batch, m, n;
  rocblas_side side;
  rocblas_fill uplo;
  rocblas_operation trans;
  rocblas_diagonal diag;
};

// Returns the descriptor for a Trsm operation.
std::pair<size_t, py::bytes> BuildTrsmDescriptor(const py::dtype& dtype,
                                                 int batch, int m, int n,
                                                 bool left_side, bool lower,
                                                 bool trans_a, bool conj_a,
                                                 bool unit_diagonal) {
  std::int64_t lwork =
      batch * sizeof(void*);  // number of bytes needed for the batch pointers
  TrsmDescriptor desc;
  desc.type = DtypeToType(dtype);
  desc.batch = batch;
  desc.m = m;
  desc.n = n;
  desc.side = left_side ? rocblas_side_left : rocblas_side_right;
  desc.uplo = lower ? rocblas_fill_lower : rocblas_fill_upper;
  desc.trans = trans_a ? (conj_a ? rocblas_operation_conjugate_transpose
                                 : rocblas_operation_transpose)
                       : rocblas_operation_none;
  desc.diag = unit_diagonal ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;
  return {lwork, PackDescriptor(desc)};
}

void Trsm(hipStream_t stream, void** buffers, const char* opaque,
          size_t opaque_len) {
  const TrsmDescriptor& d =
      *UnpackDescriptor<TrsmDescriptor>(opaque, opaque_len);
  auto handle = rocBlasHandlePool::Borrow(stream);

  // b is INOUT, so we copy the input to the output and use that if they are not
  // already the same
  if (buffers[2] != buffers[1]) {
    ThrowIfError(hipMemcpyAsync(buffers[2], buffers[1],
                                SizeOfType(d.type) * d.batch * d.m * d.n,
                                hipMemcpyDeviceToDevice, stream));
  }
  const int lda = d.side == rocblas_side_left ? d.m : d.n;
  const int ldb = d.m;

  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[0]);
        float* b = static_cast<float*>(buffers[2]);
        const float alpha = 1.0f;
        ThrowIfErrorStatus(rocblas_strsm(handle.get(), d.side, d.uplo, d.trans,
                                         d.diag, d.m, d.n, &alpha,
                                         const_cast<float*>(a), lda, b, ldb));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[0]);
        double* b = static_cast<double*>(buffers[2]);
        const double alpha = 1.0;
        ThrowIfErrorStatus(rocblas_dtrsm(handle.get(), d.side, d.uplo, d.trans,
                                         d.diag, d.m, d.n, &alpha,
                                         const_cast<double*>(a), lda, b, ldb));
        break;
      }
      case Type::C64: {
        rocblas_float_complex* a =
            static_cast<rocblas_float_complex*>(buffers[0]);
        rocblas_float_complex* b =
            static_cast<rocblas_float_complex*>(buffers[2]);
        const rocblas_float_complex alpha = {1.0f, 0.0f};
        ThrowIfErrorStatus(rocblas_ctrsm(
            handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
            const_cast<rocblas_float_complex*>(a), lda, b, ldb));
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a =
            static_cast<rocblas_double_complex*>(buffers[0]);
        rocblas_double_complex* b =
            static_cast<rocblas_double_complex*>(buffers[2]);
        const rocblas_double_complex alpha = {1.0d, 0.0d};
        ThrowIfErrorStatus(rocblas_ztrsm(
            handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
            const_cast<rocblas_double_complex*>(a), lda, b, ldb));
        break;
      }
    }
  } else {
    auto a_batch_host =
        MakeBatchPointers(stream, buffers[0], buffers[3], d.batch,
                          SizeOfType(d.type) * lda * lda);
    auto b_batch_host =
        MakeBatchPointers(stream, buffers[2], buffers[4], d.batch,
                          SizeOfType(d.type) * d.m * d.n);
    // TODO(phawkins): ideally we would not need to synchronize here, but to
    // avoid it we need a way to keep the host-side buffer alive until the copy
    // completes.
    ThrowIfError(hipStreamSynchronize(stream));

    switch (d.type) {
      case Type::F32: {
        float** a_batch_ptrs = static_cast<float**>(buffers[3]);
        float** b_batch_ptrs = static_cast<float**>(buffers[4]);
        const float alpha = 1.0f;
        ThrowIfErrorStatus(rocblas_strsm_batched(
            handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
            const_cast<float**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
            d.batch));
        break;
      }
      case Type::F64: {
        double** a_batch_ptrs = static_cast<double**>(buffers[3]);
        double** b_batch_ptrs = static_cast<double**>(buffers[4]);
        const double alpha = 1.0;
        ThrowIfErrorStatus(rocblas_dtrsm_batched(
            handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
            const_cast<double**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
            d.batch));
        break;
      }
      case Type::C64: {
        rocblas_float_complex** a_batch_ptrs =
            static_cast<rocblas_float_complex**>(buffers[3]);
        rocblas_float_complex** b_batch_ptrs =
            static_cast<rocblas_float_complex**>(buffers[4]);
        const rocblas_float_complex alpha = {1.0f, 0.0f};
        ThrowIfErrorStatus(rocblas_ctrsm_batched(
            handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
            const_cast<rocblas_float_complex**>(a_batch_ptrs), lda,
            b_batch_ptrs, ldb, d.batch));
        break;
      }
      case Type::C128: {
        rocblas_double_complex** a_batch_ptrs =
            static_cast<rocblas_double_complex**>(buffers[3]);
        rocblas_double_complex** b_batch_ptrs =
            static_cast<rocblas_double_complex**>(buffers[4]);
        const rocblas_double_complex alpha = {1.0d, 0.0d};
        ThrowIfErrorStatus(rocblas_ztrsm_batched(
            handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
            const_cast<rocblas_double_complex**>(a_batch_ptrs), lda,
            b_batch_ptrs, ldb, d.batch));
        break;
      }
    }
  }
}

//##########################
// rocsolver
//##########################

// potrf: Cholesky decomposition

struct PotrfDescriptor {
  Type type;
  rocblas_fill uplo;
  std::int64_t batch, n;
};

// Returns the descriptor for a potrf operation.
std::pair<int, py::bytes> BuildPotrfDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  Type type = DtypeToType(dtype);
  rocblas_fill uplo = lower ? rocblas_fill_lower : rocblas_fill_upper;
  std::int64_t lwork =
      b * sizeof(void*);  // number of bytes needed for the batch pointers
  return {lwork, PackDescriptor(PotrfDescriptor{type, uplo, b, n})};
}

void Potrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const PotrfDescriptor& d =
      *UnpackDescriptor<PotrfDescriptor>(opaque, opaque_len);
  auto handle = rocBlasHandlePool::Borrow(stream);
  // a is INOUT, so we copy the input to the output and use that if they are not
  // already the same
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0],
                                SizeOfType(d.type) * d.batch * d.n * d.n,
                                hipMemcpyDeviceToDevice, stream));
  }

  int* info = static_cast<int*>(buffers[2]);
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_spotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_dpotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
      case Type::C64: {
        rocblas_float_complex* a =
            static_cast<rocblas_float_complex*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_cpotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a =
            static_cast<rocblas_double_complex*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_zpotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
    }
  } else {
    auto a_ptrs_host =
        MakeBatchPointers(stream, buffers[1], buffers[3], d.batch,
                          SizeOfType(d.type) * d.n * d.n);
    // TODO(phawkins): ideally we would not need to synchronize here, but to
    // avoid it we need a way to keep the host-side buffer alive until the copy
    // completes.
    ThrowIfError(hipStreamSynchronize(stream));

    switch (d.type) {
      case Type::F32: {
        float** a_batch_ptrs = static_cast<float**>(buffers[3]);
        ThrowIfErrorStatus(rocsolver_spotrf_batched(
            handle.get(), d.uplo, d.n, a_batch_ptrs, d.n, info, d.batch));
        break;
      }
      case Type::F64: {
        double** a_batch_ptrs = static_cast<double**>(buffers[3]);
        ThrowIfErrorStatus(rocsolver_dpotrf_batched(
            handle.get(), d.uplo, d.n, a_batch_ptrs, d.n, info, d.batch));
        break;
      }
      case Type::C64: {
        rocblas_float_complex** a_batch_ptrs =
            static_cast<rocblas_float_complex**>(buffers[3]);
        ThrowIfErrorStatus(rocsolver_cpotrf_batched(
            handle.get(), d.uplo, d.n, a_batch_ptrs, d.n, info, d.batch));
        break;
      }
      case Type::C128: {
        rocblas_double_complex** a_batch_ptrs =
            static_cast<rocblas_double_complex**>(buffers[3]);
        ThrowIfErrorStatus(rocsolver_zpotrf_batched(
            handle.get(), d.uplo, d.n, a_batch_ptrs, d.n, info, d.batch));
        break;
      }
    }
  }
}

// getrf: LU decomposition

struct GetrfDescriptor {
  Type type;
  int batch, m, n;
};

// Returns the descriptor for a getrf operation.
std::pair<int, py::bytes> BuildGetrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  Type type = DtypeToType(dtype);
  std::int64_t lwork =
      b * sizeof(void*);  // number of bytes needed for the batch pointers
  return {lwork, PackDescriptor(GetrfDescriptor{type, b, m, n})};
}

void Getrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const GetrfDescriptor& d =
      *UnpackDescriptor<GetrfDescriptor>(opaque, opaque_len);
  auto handle = rocBlasHandlePool::Borrow(stream);

  // a is INOUT, so we copy the input to the output and use that if they are not
  // already the same
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0],
                                SizeOfType(d.type) * d.batch * d.m * d.n,
                                hipMemcpyDeviceToDevice, stream));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);

  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_sgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_dgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
      case Type::C64: {
        rocblas_float_complex* a =
            static_cast<rocblas_float_complex*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_cgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a =
            static_cast<rocblas_double_complex*>(buffers[1]);
        ThrowIfErrorStatus(
            rocsolver_zgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
    }
  } else {
    auto a_ptrs_host =
        MakeBatchPointers(stream, buffers[1], buffers[4], d.batch,
                          SizeOfType(d.type) * d.m * d.n);
    // TODO(phawkins): ideally we would not need to synchronize here, but to
    // avoid it we need a way to keep the host-side buffer alive until the copy
    // completes.
    ThrowIfError(hipStreamSynchronize(stream));

    switch (d.type) {
      case Type::F32: {
        float** batch_ptrs = static_cast<float**>(buffers[4]);
        ThrowIfErrorStatus(
            rocsolver_sgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     ipiv, std::min(d.m, d.n), info, d.batch));
        break;
      }
      case Type::F64: {
        double** batch_ptrs = static_cast<double**>(buffers[4]);
        ThrowIfErrorStatus(
            rocsolver_dgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     ipiv, std::min(d.m, d.n), info, d.batch));
        break;
      }
      case Type::C64: {
        rocblas_float_complex** batch_ptrs =
            static_cast<rocblas_float_complex**>(buffers[4]);
        ThrowIfErrorStatus(
            rocsolver_cgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     ipiv, std::min(d.m, d.n), info, d.batch));
        break;
      }
      case Type::C128: {
        rocblas_double_complex** batch_ptrs =
            static_cast<rocblas_double_complex**>(buffers[4]);
        ThrowIfErrorStatus(
            rocsolver_zgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     ipiv, std::min(d.m, d.n), info, d.batch));
        break;
      }
    }
  }
}

// geqrf: QR decomposition

struct GeqrfDescriptor {
  Type type;
  int batch, m, n;
};

std::pair<int, py::bytes> BuildGeqrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  Type type = DtypeToType(dtype);
  std::int64_t lwork =
      b * sizeof(void*);  // number of bytes needed for the batch pointers
  return {lwork, PackDescriptor(GeqrfDescriptor{type, b, m, n})};
}

void Geqrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const GeqrfDescriptor& d =
      *UnpackDescriptor<GeqrfDescriptor>(opaque, opaque_len);
  auto handle = rocBlasHandlePool::Borrow(stream);

  // a is INOUT, so we copy the input to the output and use that if they are not
  // already the same
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0],
                                SizeOfType(d.type) * d.batch * d.m * d.n,
                                hipMemcpyDeviceToDevice, stream));
  }

  // here tau is tau

  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* tau = static_cast<float*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_sgeqrf(handle.get(), d.m, d.n, a, d.m, tau));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* tau = static_cast<double*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_dgeqrf(handle.get(), d.m, d.n, a, d.m, tau));
        break;
      }
      case Type::C64: {
        rocblas_float_complex* a =
            static_cast<rocblas_float_complex*>(buffers[1]);
        rocblas_float_complex* tau =
            static_cast<rocblas_float_complex*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_cgeqrf(handle.get(), d.m, d.n, a, d.m, tau));
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a =
            static_cast<rocblas_double_complex*>(buffers[1]);
        rocblas_double_complex* tau =
            static_cast<rocblas_double_complex*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_zgeqrf(handle.get(), d.m, d.n, a, d.m, tau));
        break;
      }
    }
  } else {
    auto a_ptrs_host =
        MakeBatchPointers(stream, buffers[1], buffers[3], d.batch,
                          SizeOfType(d.type) * d.m * d.n);
    // TODO(phawkins): ideally we would not need to synchronize here, but to
    // avoid it we need a way to keep the host-side buffer alive until the copy
    // completes.
    ThrowIfError(hipStreamSynchronize(stream));

    switch (d.type) {
      case Type::F32: {
        float** batch_ptrs = static_cast<float**>(buffers[3]);
        float* tau = static_cast<float*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_sgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     tau, std::min(d.m, d.n), d.batch));
        break;
      }
      case Type::F64: {
        double** batch_ptrs = static_cast<double**>(buffers[3]);
        double* tau = static_cast<double*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_dgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     tau, std::min(d.m, d.n), d.batch));
        break;
      }
      case Type::C64: {
        rocblas_float_complex** batch_ptrs =
            static_cast<rocblas_float_complex**>(buffers[3]);
        rocblas_float_complex* tau =
            static_cast<rocblas_float_complex*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_cgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     tau, std::min(d.m, d.n), d.batch));
        break;
      }
      case Type::C128: {
        rocblas_double_complex** batch_ptrs =
            static_cast<rocblas_double_complex**>(buffers[3]);
        rocblas_double_complex* tau =
            static_cast<rocblas_double_complex*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolver_zgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m,
                                     tau, std::min(d.m, d.n), d.batch));
        break;
      }
    }
  }
}

// orgqr/ungqr: apply elementary Householder transformations
struct OrgqrDescriptor {
  Type type;
  int batch, m, n, k;
};

std::pair<int, py::bytes> BuildOrgqrDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, int k) {
  Type type = DtypeToType(dtype);
  std::int64_t lwork = 0;
  return {lwork, PackDescriptor(OrgqrDescriptor{type, b, m, n, k})};
}

void Orgqr(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const OrgqrDescriptor& d =
      *UnpackDescriptor<OrgqrDescriptor>(opaque, opaque_len);
  auto handle = rocBlasHandlePool::Borrow(stream);

  // a is INOUT, so we copy the input to the output and use that if they are not
  // already the same
  if (buffers[2] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[2], buffers[0],
                                SizeOfType(d.type) * d.batch * d.m * d.n,
                                hipMemcpyDeviceToDevice, stream));
  }

  switch (d.type) {
      // orgqr

    case Type::F32: {
      float* a = static_cast<float*>(buffers[2]);
      float* tau = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(
            rocsolver_sorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau));
        a += d.m * d.n;
        tau += d.k;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[2]);
      double* tau = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(
            rocsolver_dorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau));
        a += d.m * d.n;
        tau += d.k;
      }
      break;
    }

      // ungqr

    case Type::C64: {
      rocblas_float_complex* a =
          static_cast<rocblas_float_complex*>(buffers[2]);
      rocblas_float_complex* tau =
          static_cast<rocblas_float_complex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(
            rocsolver_cungqr(handle.get(), d.m, d.n, d.k, a, d.m, tau));
        a += d.m * d.n;
        tau += d.k;
      }
      break;
    }
    case Type::C128: {
      rocblas_double_complex* a =
          static_cast<rocblas_double_complex*>(buffers[2]);
      rocblas_double_complex* tau =
          static_cast<rocblas_double_complex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(
            rocsolver_zungqr(handle.get(), d.m, d.n, d.k, a, d.m, tau));
        a += d.m * d.n;
        tau += d.k;
      }
      break;
    }
  }
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd
// not implemented yet in rocsolver

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// not implemented yet in rocsolver

// Singular value decomposition using QR algorithm: gesvd

struct GesvdDescriptor {
  Type type;
  int batch, m, n;
  rocblas_svect jobu, jobvt;
};

std::pair<int, py::bytes> BuildGesvdDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, bool compute_uv,
                                               bool full_matrices) {
  Type type = DtypeToType(dtype);

  std::int64_t lwork =
      b * sizeof(void*);  // number of bytes needed for the batch pointers

  rocblas_svect jobu, jobvt;
  if (compute_uv) {
    if (full_matrices) {
      jobu = jobvt = rocblas_svect_all;
    } else {
      jobu = jobvt = rocblas_svect_singular;
    }
  } else {
    jobu = jobvt = rocblas_svect_none;
  }

  return {lwork, PackDescriptor(GesvdDescriptor{type, b, m, n, jobu, jobvt})};
}

void Gesvd(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const GesvdDescriptor& d =
      *UnpackDescriptor<GesvdDescriptor>(opaque, opaque_len);
  auto handle = rocBlasHandlePool::Borrow(stream);

  // a is INOUT, so we copy the input to the output and use that if they are not
  // already the same
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0],
                                SizeOfType(d.type) * d.batch * d.m * d.n,
                                hipMemcpyDeviceToDevice, stream));
  }

  int* info = static_cast<int*>(buffers[5]);

  const rocblas_int lda = d.m;
  const rocblas_int ldu = d.m;
  const rocblas_int ldv = d.n;

  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* vt = static_cast<float*>(buffers[4]);
        float* e = static_cast<float*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_sgesvd(handle.get(), d.jobu, d.jobvt, d.m,
                                            d.n, a, lda, s, u, ldu, vt, ldv, e,
                                            rocblas_inplace, info));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* vt = static_cast<double*>(buffers[4]);
        double* e = static_cast<double*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_dgesvd(handle.get(), d.jobu, d.jobvt, d.m,
                                            d.n, a, lda, s, u, ldu, vt, ldv, e,
                                            rocblas_inplace, info));
        break;
      }
      case Type::C64: {
        rocblas_float_complex* a =
            static_cast<rocblas_float_complex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        rocblas_float_complex* u =
            static_cast<rocblas_float_complex*>(buffers[3]);
        rocblas_float_complex* vt =
            static_cast<rocblas_float_complex*>(buffers[4]);
        float* e = static_cast<float*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_cgesvd(handle.get(), d.jobu, d.jobvt, d.m,
                                            d.n, a, lda, s, u, ldu, vt, ldv, e,
                                            rocblas_inplace, info));
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a =
            static_cast<rocblas_double_complex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        rocblas_double_complex* u =
            static_cast<rocblas_double_complex*>(buffers[3]);
        rocblas_double_complex* vt =
            static_cast<rocblas_double_complex*>(buffers[4]);
        double* e = static_cast<double*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_zgesvd(handle.get(), d.jobu, d.jobvt, d.m,
                                            d.n, a, lda, s, u, ldu, vt, ldv, e,
                                            rocblas_inplace, info));
        break;
      }
    }
  } else {
    const rocblas_stride stride_s = std::min(d.m, d.n);
    const rocblas_stride stride_u = ldu * d.m;
    const rocblas_stride stride_v = ldv * d.n;
    const rocblas_stride stride_e = std::min(d.m, d.n) - 1;

    auto a_ptrs_host =
        MakeBatchPointers(stream, buffers[1], buffers[7], d.batch,
                          SizeOfType(d.type) * d.m * d.n);
    // TODO(phawkins): ideally we would not need to synchronize here, but to
    // avoid it we need a way to keep the host-side buffer alive until the copy
    // completes.
    ThrowIfError(hipStreamSynchronize(stream));

    switch (d.type) {
      case Type::F32: {
        float** a_batch_ptrs = static_cast<float**>(buffers[7]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* vt = static_cast<float*>(buffers[4]);
        float* e = static_cast<float*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_sgesvd_batched(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a_batch_ptrs, lda, s,
            stride_s, u, ldu, stride_u, vt, ldv, stride_v, e, stride_e,
            rocblas_inplace, info, d.batch));
        break;
      }
      case Type::F64: {
        double** a_batch_ptrs = static_cast<double**>(buffers[7]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* vt = static_cast<double*>(buffers[4]);
        double* e = static_cast<double*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_dgesvd_batched(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a_batch_ptrs, lda, s,
            stride_s, u, ldu, stride_u, vt, ldv, stride_v, e, stride_e,
            rocblas_inplace, info, d.batch));
        break;
      }
      case Type::C64: {
        rocblas_float_complex** a_batch_ptrs =
            static_cast<rocblas_float_complex**>(buffers[7]);
        float* s = static_cast<float*>(buffers[2]);
        rocblas_float_complex* u =
            static_cast<rocblas_float_complex*>(buffers[3]);
        rocblas_float_complex* vt =
            static_cast<rocblas_float_complex*>(buffers[4]);
        float* e = static_cast<float*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_cgesvd_batched(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a_batch_ptrs, lda, s,
            stride_s, u, ldu, stride_u, vt, ldv, stride_v, e, stride_e,
            rocblas_inplace, info, d.batch));
        break;
      }
      case Type::C128: {
        rocblas_double_complex** a_batch_ptrs =
            static_cast<rocblas_double_complex**>(buffers[7]);
        double* s = static_cast<double*>(buffers[2]);
        rocblas_double_complex* u =
            static_cast<rocblas_double_complex*>(buffers[3]);
        rocblas_double_complex* vt =
            static_cast<rocblas_double_complex*>(buffers[4]);
        double* e = static_cast<double*>(buffers[6]);
        ThrowIfErrorStatus(rocsolver_zgesvd_batched(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a_batch_ptrs, lda, s,
            stride_s, u, ldu, stride_u, vt, ldv, stride_v, e, stride_e,
            rocblas_inplace, info, d.batch));
        break;
      }
    }
  }
}

// Singular value decomposition using Jacobi algorithm: gesvdj
// not implemented yet in rocsolver

py::dict Registrations() {
  py::dict dict;
  dict["rocblas_trsm"] = EncapsulateFunction(Trsm);
  // there are differnent versions of getrf in cublas and cusolver
  // however with rocm there is just one in rocsolver

  dict["rocsolver_potrf"] = EncapsulateFunction(Potrf);
  dict["rocsolver_getrf"] = EncapsulateFunction(Getrf);
  dict["rocsolver_geqrf"] = EncapsulateFunction(Geqrf);
  dict["rocsolver_orgqr"] = EncapsulateFunction(Orgqr);
  //  dict["rocsolver_syevd"] = EncapsulateFunction(Syevd);
  //  dict["rocsolver_syevj"] = EncapsulateFunction(Syevj);
  dict["rocsolver_gesvd"] = EncapsulateFunction(Gesvd);
  //  dict["rocsolver_gesvdj"] = EncapsulateFunction(Gesvdj);

  return dict;
}

PYBIND11_MODULE(rocblas_kernels, m) {
  m.def("registrations", &Registrations);

  m.def("build_trsm_descriptor", &BuildTrsmDescriptor);

  m.def("build_potrf_descriptor", &BuildPotrfDescriptor);
  m.def("build_getrf_descriptor", &BuildGetrfDescriptor);
  m.def("build_geqrf_descriptor", &BuildGeqrfDescriptor);
  m.def("build_orgqr_descriptor", &BuildOrgqrDescriptor);
  //  m.def("build_syevd_descriptor", &BuildSyevdDescriptor);
  //  m.def("build_syevj_descriptor", &BuildSyevjDescriptor);
  m.def("build_gesvd_descriptor", &BuildGesvdDescriptor);
  //  m.def("build_gesvdj_descriptor", &BuildGesvdjDescriptor);
}

}  // namespace
}  // namespace jax
