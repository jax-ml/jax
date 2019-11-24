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
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "jaxlib/kernel_helpers.h"

namespace jax {
namespace {

namespace py = pybind11;

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error("CUDA operation failed");
  }
}

void ThrowIfErrorStatus(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return;
    case CUBLAS_STATUS_NOT_INITIALIZED:
      throw std::runtime_error("cuBlas has not been initialized");
    case CUBLAS_STATUS_ALLOC_FAILED:
      throw std::runtime_error("cuBlas allocation failure");
    case CUBLAS_STATUS_INVALID_VALUE:
      throw std::runtime_error("cuBlas invalid value error");
    case CUBLAS_STATUS_ARCH_MISMATCH:
      throw std::runtime_error("cuBlas architecture mismatch");
    case CUBLAS_STATUS_MAPPING_ERROR:
      throw std::runtime_error("cuBlas mapping error");
    case CUBLAS_STATUS_EXECUTION_FAILED:
      throw std::runtime_error("cuBlas execution failed");
    case CUBLAS_STATUS_INTERNAL_ERROR:
      throw std::runtime_error("cuBlas internal error");
    case CUBLAS_STATUS_NOT_SUPPORTED:
      throw std::runtime_error("cuBlas not supported error");
    case CUBLAS_STATUS_LICENSE_ERROR:
      throw std::runtime_error("cuBlas license error");
    default:
      throw std::runtime_error("Unknown cuBlas error");
  }
}

// To avoid creating cublas contexts in the middle of execution, we maintain
// a pool of them.
class BlasHandlePool {
 public:
  BlasHandlePool() = default;

  // RAII class representing a cusolver handle borrowed from the pool. Returns
  // the handle to the pool on destruction.
  class Handle {
   public:
    Handle() = default;
    ~Handle() {
      if (pool_) {
        pool_->Return(handle_);
      }
    }

    Handle(Handle const&) = delete;
    Handle(Handle&& other) {
      pool_ = other.pool_;
      handle_ = other.handle_;
      other.pool_ = nullptr;
      other.handle_ = nullptr;
    }
    Handle& operator=(Handle const&) = delete;
    Handle& operator=(Handle&& other) {
      pool_ = other.pool_;
      handle_ = other.handle_;
      other.pool_ = nullptr;
      other.handle_ = nullptr;
      return *this;
    }

    cublasHandle_t get() { return handle_; }

   private:
    friend class BlasHandlePool;
    Handle(BlasHandlePool* pool, cublasHandle_t handle)
        : pool_(pool), handle_(handle) {}
    BlasHandlePool* pool_ = nullptr;
    cublasHandle_t handle_ = nullptr;
  };

  // Borrows a handle from the pool. If 'stream' is non-null, sets the stream
  // associated with the handle.
  static Handle Borrow(cudaStream_t stream = nullptr);

 private:
  static BlasHandlePool* Instance();

  void Return(cublasHandle_t handle);

  absl::Mutex mu_;
  std::vector<cublasHandle_t> handles_ GUARDED_BY(mu_);
};

/*static*/ BlasHandlePool* BlasHandlePool::Instance() {
  static auto* pool = new BlasHandlePool;
  return pool;
}

/*static*/ BlasHandlePool::Handle BlasHandlePool::Borrow(cudaStream_t stream) {
  BlasHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cublasHandle_t handle;
  if (pool->handles_.empty()) {
    ThrowIfErrorStatus(cublasCreate(&handle));
  } else {
    handle = pool->handles_.back();
    pool->handles_.pop_back();
  }
  if (stream) {
    ThrowIfErrorStatus(cublasSetStream(handle, stream));
  }
  return Handle(pool, handle);
}

void BlasHandlePool::Return(cublasHandle_t handle) {
  absl::MutexLock lock(&mu_);
  handles_.push_back(handle);
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

// Builds an array of pointers to each array in a batch, in device memory.
template <typename T>
cudaError_t MakeBatchPointers(T* buffer, T** dev_ptrs, int batch,
                              int batch_elem_size) {
  std::vector<T*> host_ptrs(batch);
  for (int i = 0; i < batch; ++i) {
    host_ptrs[i] = buffer;
    buffer += batch_elem_size;
  }
  // TODO(phawkins): ideally we would not use a synchronous copy here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  return cudaMemcpy(dev_ptrs, host_ptrs.data(), sizeof(T*) * batch,
                    cudaMemcpyHostToDevice);
}

// Batched triangular solve: trsmbatched

struct TrsmBatchedDescriptor {
  Type type;
  int batch, m, n;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasOperation_t trans;
  cublasDiagType_t diag;
};

// Returns the descriptor for a TrsmBatched operation.
std::pair<size_t, py::bytes> BuildTrsmBatchedDescriptor(
    const py::dtype& dtype, int batch, int m, int n, bool left_side, bool lower,
    bool trans_a, bool conj_a, bool unit_diagonal) {
  size_t size = batch * sizeof(void*);
  TrsmBatchedDescriptor desc;
  desc.type = DtypeToType(dtype);
  desc.batch = batch;
  desc.m = m;
  desc.n = n;
  desc.side = left_side ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  desc.uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  desc.trans = trans_a ? (conj_a ? CUBLAS_OP_C : CUBLAS_OP_T) : CUBLAS_OP_N;
  desc.diag = unit_diagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
  return {size, PackDescriptor(desc)};
}

void TrsmBatched(cudaStream_t stream, void** buffers, const char* opaque,
                 size_t opaque_len) {
  const TrsmBatchedDescriptor& d =
      *UnpackDescriptor<TrsmBatchedDescriptor>(opaque, opaque_len);
  auto handle = BlasHandlePool::Borrow(stream);
  if (buffers[2] != buffers[1]) {
    ThrowIfError(cudaMemcpyAsync(buffers[2], buffers[1],
                                 SizeOfType(d.type) * d.batch * d.m * d.n,
                                 cudaMemcpyDeviceToDevice, stream));
  }
  const int lda = d.side == CUBLAS_SIDE_LEFT ? d.m : d.n;
  const int ldb = d.m;
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[0]);
      float* b = static_cast<float*>(buffers[2]);
      float** a_batch_ptrs = static_cast<float**>(buffers[3]);
      float** b_batch_ptrs = static_cast<float**>(buffers[4]);
      ThrowIfError(MakeBatchPointers(a, a_batch_ptrs, d.batch, lda * lda));
      ThrowIfError(MakeBatchPointers(b, b_batch_ptrs, d.batch, d.m * d.n));
      // NOTE(phawkins): if alpha is in GPU memory, cuBlas seems to segfault.
      const float alpha = 1.0f;
      ThrowIfErrorStatus(cublasStrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const float**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch));
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[0]);
      double* b = static_cast<double*>(buffers[2]);
      double** a_batch_ptrs = static_cast<double**>(buffers[3]);
      double** b_batch_ptrs = static_cast<double**>(buffers[4]);
      const double alpha = 1.0;
      ThrowIfError(MakeBatchPointers(a, a_batch_ptrs, d.batch, lda * lda));
      ThrowIfError(MakeBatchPointers(b, b_batch_ptrs, d.batch, d.m * d.n));
      ThrowIfErrorStatus(cublasDtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const double**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch));
      break;
    }
    case Type::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[0]);
      cuComplex* b = static_cast<cuComplex*>(buffers[2]);
      cuComplex** a_batch_ptrs = static_cast<cuComplex**>(buffers[3]);
      cuComplex** b_batch_ptrs = static_cast<cuComplex**>(buffers[4]);
      const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
      ThrowIfError(MakeBatchPointers(a, a_batch_ptrs, d.batch, lda * lda));
      ThrowIfError(MakeBatchPointers(b, b_batch_ptrs, d.batch, d.m * d.n));
      ThrowIfErrorStatus(cublasCtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const cuComplex**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch));
      break;
    }
    case Type::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[0]);
      cuDoubleComplex* b = static_cast<cuDoubleComplex*>(buffers[2]);
      cuDoubleComplex** a_batch_ptrs =
          static_cast<cuDoubleComplex**>(buffers[3]);
      cuDoubleComplex** b_batch_ptrs =
          static_cast<cuDoubleComplex**>(buffers[4]);
      const cuDoubleComplex alpha = make_cuDoubleComplex(1.0f, 0.0f);
      ThrowIfError(MakeBatchPointers(a, a_batch_ptrs, d.batch, lda * lda));
      ThrowIfError(MakeBatchPointers(b, b_batch_ptrs, d.batch, d.m * d.n));
      ThrowIfErrorStatus(cublasZtrsmBatched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<const cuDoubleComplex**>(a_batch_ptrs), lda, b_batch_ptrs,
          ldb, d.batch));
      break;
    }
  }
}

// Batched LU decomposition: getrfbatched

struct GetrfBatchedDescriptor {
  Type type;
  int batch, n;
};

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, py::bytes> BuildGetrfBatchedDescriptor(const py::dtype& dtype,
                                                         int b, int n) {
  Type type = DtypeToType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GetrfBatchedDescriptor{type, b, n})};
}

void GetrfBatched(cudaStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len) {
  const GetrfBatchedDescriptor& d =
      *UnpackDescriptor<GetrfBatchedDescriptor>(opaque, opaque_len);
  auto handle = BlasHandlePool::Borrow(stream);
  if (buffers[0] != buffers[1]) {
    ThrowIfError(cudaMemcpyAsync(buffers[1], buffers[0],
                                 SizeOfType(d.type) * d.batch * d.n * d.n,
                                 cudaMemcpyDeviceToDevice, stream));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float** batch_ptrs = static_cast<float**>(buffers[4]);
      ThrowIfError(MakeBatchPointers(a, batch_ptrs, d.batch, d.n * d.n));
      ThrowIfErrorStatus(cublasSgetrfBatched(handle.get(), d.n, batch_ptrs, d.n,
                                             ipiv, info, d.batch));
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double** batch_ptrs = static_cast<double**>(buffers[4]);
      ThrowIfError(MakeBatchPointers(a, batch_ptrs, d.batch, d.n * d.n));
      ThrowIfErrorStatus(cublasDgetrfBatched(handle.get(), d.n, batch_ptrs, d.n,
                                             ipiv, info, d.batch));
      break;
    }
    case Type::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      cuComplex** batch_ptrs = static_cast<cuComplex**>(buffers[4]);
      ThrowIfError(MakeBatchPointers(a, batch_ptrs, d.batch, d.n * d.n));
      ThrowIfErrorStatus(cublasCgetrfBatched(handle.get(), d.n, batch_ptrs, d.n,
                                             ipiv, info, d.batch));
      break;
    }
    case Type::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      cuDoubleComplex** batch_ptrs = static_cast<cuDoubleComplex**>(buffers[4]);
      ThrowIfError(MakeBatchPointers(a, batch_ptrs, d.batch, d.n * d.n));
      ThrowIfErrorStatus(cublasZgetrfBatched(handle.get(), d.n, batch_ptrs, d.n,
                                             ipiv, info, d.batch));
      break;
    }
  }
}

py::dict Registrations() {
  py::dict dict;
  dict["cublas_trsm_batched"] = EncapsulateFunction(TrsmBatched);
  dict["cublas_getrf_batched"] = EncapsulateFunction(GetrfBatched);
  return dict;
}

PYBIND11_MODULE(cublas_kernels, m) {
  m.def("registrations", &Registrations);
  m.def("build_trsm_batched_descriptor", &BuildTrsmBatchedDescriptor);
  m.def("build_getrf_batched_descriptor", &BuildGetrfBatchedDescriptor);
}

}  // namespace
}  // namespace jax
