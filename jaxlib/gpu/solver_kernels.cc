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

#include "jaxlib/gpu/solver_kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/solver_handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"

#ifdef JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif  // JAX_GPU_CUDA

namespace jax {

namespace JAX_GPU_NAMESPACE {

static int SizeOfSolverType(SolverType type) {
  switch (type) {
    case SolverType::F32:
      return sizeof(float);
    case SolverType::F64:
      return sizeof(double);
    case SolverType::C64:
      return sizeof(gpuComplex);
    case SolverType::C128:
      return sizeof(gpuDoubleComplex);
  }
}

#ifdef JAX_GPU_CUDA

// csrlsvqr: Linear system solve via Sparse QR

static absl::Status Csrlsvqr_(gpuStream_t stream, void** buffers,
                              const char* opaque, size_t opaque_len,
                              int& singularity) {
  auto s = UnpackDescriptor<CsrlsvqrDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CsrlsvqrDescriptor& d = **s;

  // This is the handle to the CUDA session. Gets a cusolverSp handle.
  auto h = SpSolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  cusparseMatDescr_t matdesc = nullptr;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateMatDescr(&matdesc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseSetMatType(matdesc, CUSPARSE_MATRIX_TYPE_GENERAL)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseSetMatIndexBase(matdesc, CUSPARSE_INDEX_BASE_ZERO)));

  switch (d.type) {
    case SolverType::F32: {
      float* csrValA = static_cast<float*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      float* b = static_cast<float*>(buffers[3]);
      float* x = static_cast<float*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpScsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
    case SolverType::F64: {
      double* csrValA = static_cast<double*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      double* b = static_cast<double*>(buffers[3]);
      double* x = static_cast<double*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpDcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          d.tol, d.reorder, x, &singularity)));

      break;
    }
    case SolverType::C64: {
      gpuComplex* csrValA = static_cast<gpuComplex*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      gpuComplex* b = static_cast<gpuComplex*>(buffers[3]);
      gpuComplex* x = static_cast<gpuComplex*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpCcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* csrValA = static_cast<gpuDoubleComplex*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      gpuDoubleComplex* b = static_cast<gpuDoubleComplex*>(buffers[3]);
      gpuDoubleComplex* x = static_cast<gpuDoubleComplex*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpZcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
  }

  cusparseDestroyMatDescr(matdesc);
  return absl::OkStatus();
}

void Csrlsvqr(gpuStream_t stream, void** buffers, const char* opaque,
              size_t opaque_len, XlaCustomCallStatus* status) {
  // Is >= 0 if A is singular.
  int singularity = -1;

  auto s = Csrlsvqr_(stream, buffers, opaque, opaque_len, singularity);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }

  if (singularity >= 0) {
    auto s = std::string("Singular matrix in linear solve.");
    XlaCustomCallStatusSetFailure(status, s.c_str(), s.length());
  }
}

#endif  // JAX_GPU_CUDA

// sytrd/hetrd: symmetric (Hermitian) tridiagonal reduction

static absl::Status Sytrd_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SytrdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SytrdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.lda),
        gpuMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[5]);
  void* workspace = buffers[6];
  switch (d.type) {
    case SolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* d_out = static_cast<float*>(buffers[2]);
      float* e_out = static_cast<float*>(buffers[3]);
      float* tau = static_cast<float*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSsytrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<float*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
    case SolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* d_out = static_cast<double*>(buffers[2]);
      double* e_out = static_cast<double*>(buffers[3]);
      double* tau = static_cast<double*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnDsytrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<double*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
    case SolverType::C64: {
      gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
      float* d_out = static_cast<float*>(buffers[2]);
      float* e_out = static_cast<float*>(buffers[3]);
      gpuComplex* tau = static_cast<gpuComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnChetrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<gpuComplex*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
      double* d_out = static_cast<double*>(buffers[2]);
      double* e_out = static_cast<double*>(buffers[3]);
      gpuDoubleComplex* tau = static_cast<gpuDoubleComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZhetrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<gpuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Sytrd(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Sytrd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
