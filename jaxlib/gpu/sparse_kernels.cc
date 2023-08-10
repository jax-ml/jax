/* Copyright 2021 The JAX Authors.

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

#include "jaxlib/gpu/sparse_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax {

template <>
/*static*/ absl::StatusOr<SparseHandlePool::Handle> SparseHandlePool::Borrow(
    gpuStream_t stream) {
  SparseHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  gpusparseHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

namespace JAX_GPU_NAMESPACE {

SparseConst ConstZero(gpuDataType type) {
  SparseConst c;
  std::memset(&c, 0, sizeof(c));
  return c;
}

SparseConst ConstOne(gpuDataType type) {
  SparseConst c;
  std::memset(&c, 0, sizeof(c));
  switch (type) {
#ifdef JAX_GPU_CUDA
#if JAX_GPU_HAVE_SPARSE
    // TODO(jakevdp): 4I/4U here might break on big endian platforms.
    case CUDA_R_4I:
    case CUDA_C_4I:
#endif
    case CUDA_R_8I:
    case CUDA_C_8I:
      c.i8[0] = 1;
      break;
#if JAX_GPU_HAVE_SPARSE
    case CUDA_R_4U:
    case CUDA_C_4U:
#endif
    case CUDA_R_8U:
    case CUDA_C_8U:
      c.u8[0] = 1;
      break;
#if JAX_GPU_HAVE_SPARSE
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
#if JAX_GPU_HAVE_SPARSE
    case CUDA_R_64I:
    case CUDA_C_64I:
      c.i64[0] = 1;
      break;
    case CUDA_R_64U:
    case CUDA_C_64U:
      c.u64[0] = 1;
      break;
#endif
#if JAX_GPU_HAVE_FP8
    case CUDA_R_8F_E4M3:
      c.u8[0] = __nv_cvt_float_to_fp8(1.0f, __NV_NOSAT, __NV_E4M3);
      break;
    case CUDA_R_8F_E5M2:
      c.u8[0] = __nv_cvt_float_to_fp8(1.0f, __NV_NOSAT, __NV_E5M2);
      break;
#endif
#if JAX_GPU_HAVE_SPARSE
    case CUDA_R_16BF:
    case CUDA_C_16BF:
      c.u16[0] = 0b11111110000000;  // 1.0 in little-endian bfloat16
      break;
#endif
#endif  // JAX_GPU_CUDA
    // TODO(rocm): add more data types if new rocm supports them.

    // TODO(jakevdp): 16F/16BF here might break on big endian platforms.
    case GPU_R_16F:
    case GPU_C_16F:
      c.u16[0] = 0b11110000000000;  // 1.0 in little-endian float16
      break;
    case GPU_R_32F:
    case GPU_C_32F:
      c.f32[0] = 1.0;
      break;
    case GPU_R_64F:
    case GPU_C_64F:
      c.f64[0] = 1.0;
      break;
  }
  return c;
}

#if JAX_GPU_HAVE_SPARSE
// CsrToDense: Convert CSR matrix to dense matrix

static absl::Status CsrToDense_(gpuStream_t stream, void** buffers,
                                const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCreateCsr(&mat_a, d.rows, d.cols, d.nnz,
                         /*csrRowOffsets=*/buffers[2],
                         /*csrColInd=*/buffers[1],
                         /*csrValues=*/buffers[0], d.index_type, d.index_type,
                         GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, buffers[3], d.value_type, GPUSPARSE_ORDER_ROW)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseSparseToDense(handle.get(), mat_a, mat_b,
                             GPUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffers[4])));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));
  return absl::OkStatus();
}

void CsrToDense(gpuStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrToDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrFromDense: Convert dense matrix to CSR matrix

static absl::Status CsrFromDense_(gpuStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpusparseDnMatDescr_t mat_a = 0;
  gpusparseSpMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, buffers[0], d.value_type, GPUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCreateCsr(&mat_b, d.rows, d.cols, d.nnz,
                         /*csrRowOffsets=*/buffers[3],
                         /*csrColInd=*/buffers[2],
                         /*csrValues=*/buffers[1], d.index_type, d.index_type,
                         GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_b)));
  return absl::OkStatus();
}

void CsrFromDense(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrFromDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrMatvec: Product of CSR matrix and dense vector.

static absl::Status CsrMatvec_(gpuStream_t stream, void** buffers,
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
  SparseConst alpha = ConstOne(d.y.type);
  SparseConst beta = ConstZero(d.y.type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnVecDescr_t vec_x = 0;
  gpusparseDnVecDescr_t vec_y = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCsr(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, csr_row_offsets, csr_col_ind,
      csr_values, d.A.index_type, d.A.index_type, GPUSPARSE_INDEX_BASE_ZERO,
      d.A.value_type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_x, d.x.size, xbuf, d.x.type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_y, d.y.size, ybuf, d.y.type)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseSpMV(handle.get(), d.op, &alpha, mat_a, vec_x, &beta, vec_y,
                    d.y.type, GPUSPARSE_SPMV_CSR_ALG, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_x)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_y)));
  return absl::OkStatus();
}

void CsrMatvec(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrMatvec_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrMatmat: Product of CSR matrix and dense matrix.

static absl::Status CsrMatmat_(gpuStream_t stream, void** buffers,
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
  SparseConst alpha = ConstOne(d.C.type);
  SparseConst beta = ConstZero(d.C.type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  gpusparseDnMatDescr_t mat_c = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCsr(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, csr_row_offsets, csr_col_ind,
      csr_values, d.A.index_type, d.A.index_type, GPUSPARSE_INDEX_BASE_ZERO,
      d.A.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_b, d.B.rows, d.B.cols,
      /*ld=*/d.B.cols, Bbuf, d.B.type, GPUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_c, d.C.rows, d.C.cols,
      /*ld=*/d.C.cols, Cbuf, d.C.type, GPUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseSpMM(
      handle.get(), d.op_A, /*opB=*/GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, d.C.type, GPUSPARSE_SPMM_CSR_ALG, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_c)));
  return absl::OkStatus();
}

void CsrMatmat(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrMatmat_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooToDense: Convert COO matrix to dense matrix

static absl::Status CooToDense_(gpuStream_t stream, void** buffers,
                                const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCreateCoo(&mat_a, d.rows, d.cols, d.nnz,
                         /*cooRowInd=*/buffers[1],
                         /*cooColInd=*/buffers[2],
                         /*cooValues=*/buffers[0], d.index_type,
                         GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, buffers[3], d.value_type, GPUSPARSE_ORDER_ROW)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseSparseToDense(handle.get(), mat_a, mat_b,
                             GPUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffers[4])));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));
  return absl::OkStatus();
}

void CooToDense(gpuStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooToDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooFromDense: Convert dense matrix to COO matrix

static absl::Status CooFromDense_(gpuStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpusparseDnMatDescr_t mat_a = 0;
  gpusparseSpMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, buffers[0], d.value_type, GPUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCreateCoo(&mat_b, d.rows, d.cols, d.nnz,
                         /*cooRowInd=*/buffers[2],
                         /*cooColInd=*/buffers[3],
                         /*cooValues=*/buffers[1], d.index_type,
                         GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_b)));
  return absl::OkStatus();
}

void CooFromDense(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooFromDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooMatvec: Product of COO matrix and dense vector.

static absl::Status CooMatvec_(gpuStream_t stream, void** buffers,
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
  SparseConst alpha = ConstOne(d.y.type);
  SparseConst beta = ConstZero(d.y.type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnVecDescr_t vec_x = 0;
  gpusparseDnVecDescr_t vec_y = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCoo(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, coo_row_ind, coo_col_ind, coo_values,
      d.A.index_type, GPUSPARSE_INDEX_BASE_ZERO, d.A.value_type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_x, d.x.size, xbuf, d.x.type)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_y, d.y.size, ybuf, d.y.type)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseSpMV(handle.get(), d.op, &alpha, mat_a, vec_x, &beta, vec_y,
                    d.y.type, GPUSPARSE_SPMV_COO_ALG, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_x)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_y)));
  return absl::OkStatus();
}

void CooMatvec(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooMatvec_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CooMatmat: Product of COO matrix and dense matrix.

static absl::Status CooMatmat_(gpuStream_t stream, void** buffers,
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
  SparseConst alpha = ConstOne(d.C.type);
  SparseConst beta = ConstZero(d.C.type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  gpusparseDnMatDescr_t mat_c = 0;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCoo(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, coo_row_ind, coo_col_ind, coo_values,
      d.A.index_type, GPUSPARSE_INDEX_BASE_ZERO, d.A.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCooSetStridedBatch(mat_a, /*batchCount=*/d.A.batch_count,
                                  /*batchStride=*/d.A.batch_stride)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_b, d.B.rows, d.B.cols,
      /*ld=*/d.B.cols, Bbuf, d.B.type, GPUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseDnMatSetStridedBatch(mat_b, /*batchCount=*/d.B.batch_count,
                                    /*batchStride=*/d.B.batch_stride)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_c, d.C.rows, d.C.cols,
      /*ld=*/d.C.cols, Cbuf, d.C.type, GPUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseDnMatSetStridedBatch(mat_c, /*batchCount=*/d.C.batch_count,
                                    /*batchStride=*/d.C.batch_stride)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseSpMM(
      handle.get(), d.op_A, /*opB=*/GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, d.C.type, GPUSPARSE_SPMM_COO_ALG, buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_c)));
  return absl::OkStatus();
}

void CooMatmat(gpuStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CooMatmat_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}
#endif  // if JAX_GPU_HAVE_SPARSE

template <typename T, typename F>
static absl::Status gtsv2(F computeGtsv2, gpuStream_t stream, void** buffers,
                          const char* opaque, std::size_t opaque_len) {
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  auto s = UnpackDescriptor<Gtsv2Descriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const Gtsv2Descriptor& descriptor = **s;
  int batch = descriptor.batch;
  int m = descriptor.m;
  int n = descriptor.n;
  int ldb = descriptor.ldb;

  T* dl = static_cast<T*>(buffers[0]);
  T* d = static_cast<T*>(buffers[1]);
  T* du = static_cast<T*>(buffers[2]);
  T* B = static_cast<T*>(buffers[3]);
  T* X = static_cast<T*>(buffers[4]);
  void* buffer = static_cast<void *>(buffers[5]);

  // The solution X is written in place to B. We need to therefore copy the
  // contents of B into the output buffer X and pass that into the kernel as B.
  // Once copy insertion is supported for custom call aliasing, we could alias B
  // with X and avoid the copy, the code below is written defensively assuming B
  // and X might alias, but today we know they will not.
  // TODO(b/182906199): Update the comment here once copy insertion is WAI.
  if (X != B) {
    size_t B_bytes = ldb * n * sizeof(T) * batch;
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
        gpuMemcpyAsync(X, B, B_bytes, gpuMemcpyDeviceToDevice, stream)));
  }
  for (int i = 0; i < batch; ++i) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(computeGtsv2(
        handle.get(), m, n, dl, d, du, X, ldb, buffer)));
    dl += m;
    d += m;
    du += m;
    X += m * n;
  }
  return absl::OkStatus();
}

void gtsv2_f32(gpuStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = gtsv2<float>(gpusparseSgtsv2, stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

void gtsv2_f64(gpuStream_t stream, void** buffers, const char* opaque,
               std::size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = gtsv2<double>(gpusparseDgtsv2, stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
