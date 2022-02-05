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

#include "jaxlib/cublas_kernels.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_helpers.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {
using BlasHandlePool = HandlePool<cublasHandle_t, cudaStream_t>;

template <>
/*static*/ absl::StatusOr<BlasHandlePool::Handle> BlasHandlePool::Borrow(
    cudaStream_t stream) {
  BlasHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cublasHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

namespace {

// Converts a NumPy dtype to a CublasType.

int SizeOfCublasType(CublasType type) {
  switch (type) {
    case CublasType::F32:
      return sizeof(float);
    case CublasType::F64:
      return sizeof(double);
    case CublasType::C64:
      return sizeof(cuComplex);
    case CublasType::C128:
      return sizeof(cuDoubleComplex);
  }
}

}  // namespace

// Batched triangular solve: trsmbatched

static absl::Status TrsmBatched_(cudaStream_t stream, void** buffers,
                                 const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<TrsmBatchedDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const TrsmBatchedDescriptor& d = **s;
  auto h = BlasHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[2] != buffers[1]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[2], buffers[1], SizeOfCublasType(d.type) * d.batch * d.m * d.n,
        cudaMemcpyDeviceToDevice, stream)));
  }
  const int lda = d.side == CUBLAS_SIDE_LEFT ? d.m : d.n;
  const int ldb = d.m;
  auto a_batch_host = MakeBatchPointers(stream, buffers[0], buffers[3], d.batch,
                                        SizeOfCublasType(d.type) * lda * lda);
  JAX_RETURN_IF_ERROR(a_batch_host.status());
  auto b_batch_host = MakeBatchPointers(stream, buffers[2], buffers[4], d.batch,
                                        SizeOfCublasType(d.type) * d.m * d.n);
  JAX_RETURN_IF_ERROR(b_batch_host.status());
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaStreamSynchronize(stream)));
  switch (d.type) {
    case CublasType::F32: {
      float** a_batch_ptrs = static_cast<float**>(buffers[3]);
      float** b_batch_ptrs = static_cast<float**>(buffers[4]);
      // NOTE(phawkins): if alpha is in GPU memory, cuBlas seems to segfault.
      const float alpha = 1.0f;
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasStrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const float**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch)));
      break;
    }
    case CublasType::F64: {
      double** a_batch_ptrs = static_cast<double**>(buffers[3]);
      double** b_batch_ptrs = static_cast<double**>(buffers[4]);
      const double alpha = 1.0;
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasDtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const double**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch)));
      break;
    }
    case CublasType::C64: {
      cuComplex** a_batch_ptrs = static_cast<cuComplex**>(buffers[3]);
      cuComplex** b_batch_ptrs = static_cast<cuComplex**>(buffers[4]);
      const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasCtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const cuComplex**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch)));
      break;
    }
    case CublasType::C128: {
      cuDoubleComplex** a_batch_ptrs =
          static_cast<cuDoubleComplex**>(buffers[3]);
      cuDoubleComplex** b_batch_ptrs =
          static_cast<cuDoubleComplex**>(buffers[4]);
      const cuDoubleComplex alpha = make_cuDoubleComplex(1.0f, 0.0f);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasZtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const cuDoubleComplex**>(a_batch_ptrs), lda, b_batch_ptrs,
          ldb, d.batch)));
      break;
    }
  }
  return absl::OkStatus();
}

void TrsmBatched(cudaStream_t stream, void** buffers, const char* opaque,
                 size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = TrsmBatched_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Batched LU decomposition: getrfbatched

static absl::Status GetrfBatched_(cudaStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GetrfBatchedDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GetrfBatchedDescriptor& d = **s;
  auto h = BlasHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[0] != buffers[1]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0], SizeOfCublasType(d.type) * d.batch * d.n * d.n,
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[4], d.batch,
                                       SizeOfCublasType(d.type) * d.n * d.n);
  JAX_RETURN_IF_ERROR(a_ptrs_host.status());
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaStreamSynchronize(stream)));
  switch (d.type) {
    case CublasType::F32: {
      float** batch_ptrs = static_cast<float**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasSgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case CublasType::F64: {
      double** batch_ptrs = static_cast<double**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasDgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case CublasType::C64: {
      cuComplex** batch_ptrs = static_cast<cuComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasCgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case CublasType::C128: {
      cuDoubleComplex** batch_ptrs = static_cast<cuDoubleComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cublasZgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
  }
  return absl::OkStatus();
}

void GetrfBatched(cudaStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = GetrfBatched_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace jax
