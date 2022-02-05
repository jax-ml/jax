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

#include "jaxlib/cusparse_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusparse.h"
#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_helpers.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

// cuSPARSE generic APIs are not supported on Windows until 11.0
// cusparseIndexType_t is used in very limited scope so manually define will
// workaround compiling issue without harm.
#if defined(_WIN32) && (CUSPARSE_VERSION < 11000)
typedef enum {
  CUSPARSE_INDEX_16U = 1,
  CUSPARSE_INDEX_32I = 2,
  CUSPARSE_INDEX_64I = 3
} cusparseIndexType_t;
#endif

namespace jax {

template <>
/*static*/ absl::StatusOr<SparseHandlePool::Handle> SparseHandlePool::Borrow(
    cudaStream_t stream) {
  SparseHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cusparseHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

CudaConst CudaZero(cudaDataType type) {
  CudaConst c;
  std::memset(&c, 0, sizeof(c));
  return c;
}

CudaConst CudaOne(cudaDataType type) {
  CudaConst c;
  std::memset(&c, 0, sizeof(c));
  switch (type) {
#if JAX_CUSPARSE_11030
    // TODO(jakevdp): 4I/4U here might break on big endian platforms.
    case CUDA_R_4I:
    case CUDA_C_4I:
#endif
    case CUDA_R_8I:
    case CUDA_C_8I:
      c.i8[0] = 1;
      break;
#if JAX_CUSPARSE_11030
    case CUDA_R_4U:
    case CUDA_C_4U:
#endif
    case CUDA_R_8U:
    case CUDA_C_8U:
      c.u8[0] = 1;
      break;
#if JAX_CUSPARSE_11030
    case CUDA_R_16I:
    case CUDA_C_16I:
      c.i16[0] = 1;
      break;
    case CUDA_R_16U:
    case CUDA_C_16U:
      c.u16[0] = 1;
      break;
#endif
    case CUDA_R_32I:
    case CUDA_C_32I:
      c.i32[0] = 1;
      break;
    case CUDA_R_32U:
    case CUDA_C_32U:
      c.u32[0] = 1;
      break;
#if JAX_CUSPARSE_11030
    case CUDA_R_64I:
    case CUDA_C_64I:
      c.i64[0] = 1;
      break;
    case CUDA_R_64U:
    case CUDA_C_64U:
      c.u64[0] = 1;
      break;
#endif
    // TODO(jakevdp): 16F/16BF here might break on big endian platforms.
    case CUDA_R_16F:
    case CUDA_C_16F:
      c.u16[0] = 0b11110000000000;  // 1.0 in little-endian float16
      break;
#if JAX_CUSPARSE_11030
    case CUDA_R_16BF:
    case CUDA_C_16BF:
      c.u16[0] = 0b11111110000000;  // 1.0 in little-endian bfloat16
      break;
#endif
    case CUDA_R_32F:
    case CUDA_C_32F:
      c.f32[0] = 1.0;
      break;
    case CUDA_R_64F:
    case CUDA_C_64F:
      c.f64[0] = 1.0;
      break;
  }
  return c;
}

#if JAX_CUSPARSE_11030
// CsrToDense: Convert CSR matrix to dense matrix

static absl::Status CsrToDense_(cudaStream_t stream, void** buffers,
                                const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseCreateCsr(&mat_a, d.rows, d.cols, d.nnz,
                        /*csrRowOffsets=*/buffers[2],
                        /*csrColInd=*/buffers[1],
                        /*csrValues=*/buffers[0], d.index_type, d.index_type,
                        CUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, buffers[3], d.value_type, CUSPARSE_ORDER_ROW)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseSparseToDense(handle.get(), mat_a, mat_b,
                            CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffers[4])));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_b)));
  return absl::OkStatus();
}

void CsrToDense(cudaStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrToDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrFromDense: Convert dense matrix to CSR matrix

static absl::Status CsrFromDense_(cudaStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  cusparseDnMatDescr_t mat_a = 0;
  cusparseSpMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, buffers[0], d.value_type, CUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseCreateCsr(&mat_b, d.rows, d.cols, d.nnz,
                        /*csrRowOffsets=*/buffers[3],
                        /*csrColInd=*/buffers[2],
                        /*csrValues=*/buffers[1], d.index_type, d.index_type,
                        CUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_b)));
  return absl::OkStatus();
}

void CsrFromDense(cudaStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrFromDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrMatvec: Product of CSR matrix and dense vector.

static absl::Status CsrMatvec_(cudaStream_t stream, void** buffers,
                               const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<CsrMatvecDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CsrMatvecDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  void* csr_values = buffers[0];
  void* csr_col_ind = buffers[1];
  void* csr_row_offsets = buffers[2];
  void* xbuf = buffers[3];
  void* ybuf = buffers[4];
  void* buf = buffers[5];

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  CudaConst alpha = CudaOne(d.y.type);
  CudaConst beta = CudaZero(d.y.type);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnVecDescr_t vec_x = 0;
  cusparseDnVecDescr_t vec_y = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseCreateCsr(&mat_a, d.A.rows, d.A.cols, d.A.nnz, csr_row_offsets,
                        csr_col_ind, csr_values, d.A.index_type, d.A.index_type,
                        CUSPARSE_INDEX_BASE_ZERO, d.A.value_type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseCreateDnVec(&vec_x, d.x.size, xbuf, d.x.type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseCreateDnVec(&vec_y, d.y.size, ybuf, d.y.type)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseSpMV(handle.get(), d.op, &alpha, mat_a, vec_x, &beta, vec_y,
                   d.y.type, CUSPARSE_MV_ALG_DEFAULT, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnVec(vec_x)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnVec(vec_y)));
  return absl::OkStatus();
}

void CsrMatvec(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrMatvec_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrMatmat: Product of CSR matrix and dense matrix.

static absl::Status CsrMatmat_(cudaStream_t stream, void** buffers,
                               const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<CsrMatmatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CsrMatmatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  void* csr_values = buffers[0];
  void* csr_col_ind = buffers[1];
  void* csr_row_offsets = buffers[2];
  void* Bbuf = buffers[3];
  void* Cbuf = buffers[4];
  void* buf = buffers[5];

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  CudaConst alpha = CudaOne(d.C.type);
  CudaConst beta = CudaZero(d.C.type);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  cusparseDnMatDescr_t mat_c = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseCreateCsr(&mat_a, d.A.rows, d.A.cols, d.A.nnz, csr_row_offsets,
                        csr_col_ind, csr_values, d.A.index_type, d.A.index_type,
                        CUSPARSE_INDEX_BASE_ZERO, d.A.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_b, d.B.rows, d.B.cols,
      /*ld=*/d.B.cols, Bbuf, d.B.type, CUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_c, d.C.rows, d.C.cols,
      /*ld=*/d.C.cols, Cbuf, d.C.type, CUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseSpMM(
      handle.get(), d.op_A, /*opB=*/CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, d.C.type, CUSPARSE_SPMM_ALG_DEFAULT, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_b)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_c)));
  return absl::OkStatus();
}

void CsrMatmat(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrMatmat_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooToDense: Convert COO matrix to dense matrix

static absl::Status CooToDense_(cudaStream_t stream, void** buffers,
                                const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseCreateCoo(&mat_a, d.rows, d.cols, d.nnz,
                                      /*cooRowInd=*/buffers[1],
                                      /*cooColInd=*/buffers[2],
                                      /*cooValues=*/buffers[0], d.index_type,
                                      CUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, buffers[3], d.value_type, CUSPARSE_ORDER_ROW)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseSparseToDense(handle.get(), mat_a, mat_b,
                            CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffers[4])));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_b)));
  return absl::OkStatus();
}

void CooToDense(cudaStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooToDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooFromDense: Convert dense matrix to COO matrix

static absl::Status CooFromDense_(cudaStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  cusparseDnMatDescr_t mat_a = 0;
  cusparseSpMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, buffers[0], d.value_type, CUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseCreateCoo(&mat_b, d.rows, d.cols, d.nnz,
                                      /*cooRowInd=*/buffers[2],
                                      /*cooColInd=*/buffers[3],
                                      /*cooValues=*/buffers[1], d.index_type,
                                      CUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_b)));
  return absl::OkStatus();
}

void CooFromDense(cudaStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooFromDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooMatvec: Product of COO matrix and dense vector.

static absl::Status CooMatvec_(cudaStream_t stream, void** buffers,
                               const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<CooMatvecDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CooMatvecDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  void* coo_values = buffers[0];
  void* coo_row_ind = buffers[1];
  void* coo_col_ind = buffers[2];
  void* xbuf = buffers[3];
  void* ybuf = buffers[4];
  void* buf = buffers[5];

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  CudaConst alpha = CudaOne(d.y.type);
  CudaConst beta = CudaZero(d.y.type);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnVecDescr_t vec_x = 0;
  cusparseDnVecDescr_t vec_y = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateCoo(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, coo_row_ind, coo_col_ind, coo_values,
      d.A.index_type, CUSPARSE_INDEX_BASE_ZERO, d.A.value_type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseCreateDnVec(&vec_x, d.x.size, xbuf, d.x.type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseCreateDnVec(&vec_y, d.y.size, ybuf, d.y.type)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseSpMV(handle.get(), d.op, &alpha, mat_a, vec_x, &beta, vec_y,
                   d.y.type, CUSPARSE_MV_ALG_DEFAULT, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnVec(vec_x)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnVec(vec_y)));
  return absl::OkStatus();
}

void CooMatvec(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooMatvec_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooMatmat: Product of COO matrix and dense matrix.

static absl::Status CooMatmat_(cudaStream_t stream, void** buffers,
                               const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<CooMatmatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CooMatmatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  void* coo_values = buffers[0];
  void* coo_row_ind = buffers[1];
  void* coo_col_ind = buffers[2];
  void* Bbuf = buffers[3];
  void* Cbuf = buffers[4];
  void* buf = buffers[5];

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  CudaConst alpha = CudaOne(d.C.type);
  CudaConst beta = CudaZero(d.C.type);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  cusparseDnMatDescr_t mat_c = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateCoo(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, coo_row_ind, coo_col_ind, coo_values,
      d.A.index_type, CUSPARSE_INDEX_BASE_ZERO, d.A.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_b, d.B.rows, d.B.cols,
      /*ld=*/d.B.cols, Bbuf, d.B.type, CUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateDnMat(
      &mat_c, d.C.rows, d.C.cols,
      /*ld=*/d.C.cols, Cbuf, d.C.type, CUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseSpMM(
      handle.get(), d.op_A, /*opB=*/CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, d.C.type, CUSPARSE_SPMM_ALG_DEFAULT, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_b)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseDestroyDnMat(mat_c)));
  return absl::OkStatus();
}

void CooMatmat(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooMatmat_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}
#endif  // if JAX_CUSPARSE_11030

template <typename T, typename F>
static absl::Status gtsv2(F computeGtsv2, cudaStream_t stream, void** buffers,
                          const char* opaque, std::size_t opaque_len) {
  auto h = SparseHandlePool::Borrow();
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  auto s = UnpackDescriptor<Gtsv2Descriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const Gtsv2Descriptor& descriptor = **s;
  int m = descriptor.m;
  int n = descriptor.n;
  int ldb = descriptor.ldb;

  const T* dl = (const T*)(buffers[0]);
  const T* d = (const T*)(buffers[1]);
  const T* du = (const T*)(buffers[2]);
  const T* B = (T*)(buffers[3]);
  T* X = (T*)(buffers[4]);
  void* buffer = buffers[5];

  // The solution X is written in place to B. We need to therefore copy the
  // contents of B into the output buffer X and pass that into the kernel as B.
  // Once copy insertion is supported for custom call aliasing, we could alias B
  // with X and avoid the copy, the code below is written defensively assuming B
  // and X might alias, but today we know they will not.
  // TODO(b/182906199): Update the comment here once copy insertion is WAI.
  if (X != B) {
    size_t B_bytes = ldb * n * sizeof(T);
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
        cudaMemcpyAsync(X, B, B_bytes, cudaMemcpyDeviceToDevice, stream)));
  }

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      computeGtsv2(handle.get(), m, n, dl, d, du, /*B=*/X, ldb, buffer)));
  return absl::OkStatus();
}

void gtsv2_f32(cudaStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = gtsv2<float>(cusparseSgtsv2, stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

void gtsv2_f64(cudaStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = gtsv2<double>(cusparseDgtsv2, stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace jax
