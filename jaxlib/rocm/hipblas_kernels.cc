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

#include "jaxlib/rocm/hipblas_kernels.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/hipblas.h"
#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_helpers.h"
#include "jaxlib/rocm/hip_gpu_kernel_helpers.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {

using BlasHandlePool = HandlePool<hipblasHandle_t, hipStream_t>;

template <>
/*static*/ absl::StatusOr<BlasHandlePool::Handle> BlasHandlePool::Borrow(
    hipStream_t stream) {
  BlasHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  hipblasHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

namespace {

// Converts a NumPy dtype to a CublasType.

int SizeOfHipblasType(HipblasType type) {
  switch (type) {
    case HipblasType::F32:
      return sizeof(float);
    case HipblasType::F64:
      return sizeof(double);
    case HipblasType::C64:
      return sizeof(hipComplex);
    case HipblasType::C128:
      return sizeof(hipDoubleComplex);
  }
}

}  // namespace

// Batched triangular solve: trsmbatched

static absl::Status TrsmBatched_(hipStream_t stream, void** buffers,
                                 const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<TrsmBatchedDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const TrsmBatchedDescriptor& d = **s;
  auto h = BlasHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[2] != buffers[1]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
        buffers[2], buffers[1], SizeOfHipblasType(d.type) * d.batch * d.m * d.n,
        hipMemcpyDeviceToDevice, stream)));
  }
  const int lda = d.side == HIPBLAS_SIDE_LEFT ? d.m : d.n;
  const int ldb = d.m;
  auto a_batch_host = MakeBatchPointers(stream, buffers[0], buffers[3], d.batch,
                                        SizeOfHipblasType(d.type) * lda * lda);
  JAX_RETURN_IF_ERROR(a_batch_host.status());
  auto b_batch_host = MakeBatchPointers(stream, buffers[2], buffers[4], d.batch,
                                        SizeOfHipblasType(d.type) * d.m * d.n);
  JAX_RETURN_IF_ERROR(b_batch_host.status());
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipStreamSynchronize(stream)));
  switch (d.type) {
    case HipblasType::F32: {
      float** a_batch_ptrs = static_cast<float**>(buffers[3]);
      float** b_batch_ptrs = static_cast<float**>(buffers[4]);
      // TODO(reza): is the following statement correct for rocm?
      // NOTE(phawkins): if alpha is in GPU memory, cuBlas seems to segfault.
      const float alpha = 1.0f;
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasStrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<float**>(a_batch_ptrs), lda, b_batch_ptrs, ldb, d.batch)));
      break;
    }
    case HipblasType::F64: {
      double** a_batch_ptrs = static_cast<double**>(buffers[3]);
      double** b_batch_ptrs = static_cast<double**>(buffers[4]);
      const double alpha = 1.0;
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasDtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<double**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch)));
      break;
    }
    case HipblasType::C64: {
      hipblasComplex** a_batch_ptrs = static_cast<hipblasComplex**>(buffers[3]);
      hipblasComplex** b_batch_ptrs = static_cast<hipblasComplex**>(buffers[4]);
      const hipblasComplex alpha = hipblasComplex(1.0f, 0.0f);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasCtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<hipblasComplex**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch)));
      break;
    }
    case HipblasType::C128: {
      hipblasDoubleComplex** a_batch_ptrs =
          static_cast<hipblasDoubleComplex**>(buffers[3]);
      hipblasDoubleComplex** b_batch_ptrs =
          static_cast<hipblasDoubleComplex**>(buffers[4]);
      const hipblasDoubleComplex alpha = hipblasDoubleComplex(1.0f, 0.0f);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasZtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<hipblasDoubleComplex**>(a_batch_ptrs), lda, b_batch_ptrs,
          ldb, d.batch)));
      break;
    }
  }
  return absl::OkStatus();
}

void TrsmBatched(hipStream_t stream, void** buffers, const char* opaque,
                 size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = TrsmBatched_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Batched LU decomposition: getrfbatched

static absl::Status GetrfBatched_(hipStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GetrfBatchedDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GetrfBatchedDescriptor& d = **s;
  auto h = BlasHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[0] != buffers[1]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
        buffers[1], buffers[0], SizeOfHipblasType(d.type) * d.batch * d.n * d.n,
        hipMemcpyDeviceToDevice, stream)));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[4], d.batch,
                                       SizeOfHipblasType(d.type) * d.n * d.n);
  JAX_RETURN_IF_ERROR(a_ptrs_host.status());
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipStreamSynchronize(stream)));
  switch (d.type) {
    case HipblasType::F32: {
      float** batch_ptrs = static_cast<float**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasSgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case HipblasType::F64: {
      double** batch_ptrs = static_cast<double**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasDgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case HipblasType::C64: {
      hipblasComplex** batch_ptrs = static_cast<hipblasComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasCgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case HipblasType::C128: {
      hipblasDoubleComplex** batch_ptrs =
          static_cast<hipblasDoubleComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipblasZgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
  }
  return absl::OkStatus();
}

void GetrfBatched(hipStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = GetrfBatched_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Batched QR decomposition: geqrfbatched

static absl::Status GeqrfBatched_(hipStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GeqrfBatchedDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GeqrfBatchedDescriptor& d = **s;
  auto h = BlasHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[0] != buffers[1]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipMemcpyAsync(
        buffers[1], buffers[0], SizeOfHipblasType(d.type) * d.batch * d.m * d.n,
        hipMemcpyDeviceToDevice, stream)));
  }

  std::vector<int> info(d.batch);
  auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[3], d.batch,
                                       SizeOfHipblasType(d.type) * d.m * d.n);
  JAX_RETURN_IF_ERROR(a_ptrs_host.status());
  auto tau_ptrs_host =
      MakeBatchPointers(stream, buffers[2], buffers[4], d.batch,
                        SizeOfHipblasType(d.type) * std::min(d.m, d.n));
  JAX_RETURN_IF_ERROR(tau_ptrs_host.status());
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipStreamSynchronize(stream)));
  switch (d.type) {
    case HipblasType::F32: {
      float** a_batch_ptrs = static_cast<float**>(buffers[3]);
      float** tau_batch_ptrs = static_cast<float**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          hipblasSgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
    case HipblasType::F64: {
      double** a_batch_ptrs = static_cast<double**>(buffers[3]);
      double** tau_batch_ptrs = static_cast<double**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          hipblasDgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
    case HipblasType::C64: {
      hipblasComplex** a_batch_ptrs = static_cast<hipblasComplex**>(buffers[3]);
      hipblasComplex** tau_batch_ptrs =
          static_cast<hipblasComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          hipblasCgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
    case HipblasType::C128: {
      hipblasDoubleComplex** a_batch_ptrs =
          static_cast<hipblasDoubleComplex**>(buffers[3]);
      hipblasDoubleComplex** tau_batch_ptrs =
          static_cast<hipblasDoubleComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          hipblasZgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
  }
  auto it =
      std::find_if(info.begin(), info.end(), [](int i) { return i != 0; });

  if (it != info.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("QR decomposition failed with status %d for batch "
                        "element %d",
                        *it, std::distance(info.begin(), it)));
  }

  return absl::OkStatus();
}

void GeqrfBatched(hipStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = GeqrfBatched_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace jax
