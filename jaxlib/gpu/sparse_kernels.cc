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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/ffi_wrapper.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace ffi = ::xla::ffi;

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

absl::StatusOr<SparseConst> ConstOne(gpuDataType type) {
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
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported data type: ", type));
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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CsrToDenseFfi, CsrToDense_);

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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CsrFromDenseFfi, CsrFromDense_);

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
  JAX_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(d.y.type));
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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CsrMatvecFfi, CsrMatvec_);

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
  JAX_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(d.C.type));
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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CsrMatmatFfi, CsrMatmat_);

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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CooToDenseFfi, CooToDense_);

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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CooFromDenseFfi, CooFromDense_);

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
  JAX_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(d.y.type));
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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CooMatvecFfi, CooMatvec_);

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
  JAX_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(d.C.type));
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

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(CooMatmatFfi, CooMatmat_);
#endif  // if JAX_GPU_HAVE_SPARSE

template <typename T, typename BufferSizeF, typename KernelF>
ffi::Error Gtsv2Impl(BufferSizeF getBufferSize, KernelF kernel, int64_t batch,
                     int64_t rows, int64_t cols, gpuStream_t stream,
                     ffi::ScratchAllocator& scratch, ffi::AnyBuffer dl,
                     ffi::AnyBuffer d, ffi::AnyBuffer du, ffi::AnyBuffer b,
                     ffi::Result<ffi::AnyBuffer> out) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));
  size_t buffer_size_in_bytes;
  JAX_FFI_RETURN_IF_GPU_ERROR(getBufferSize(handle.get(), m, n, nullptr,
                                            nullptr, nullptr, nullptr, m,
                                            &buffer_size_in_bytes));
  auto maybe_workspace = scratch.Allocate(buffer_size_in_bytes);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for gtsv2");
  }
  void* workspace = maybe_workspace.value();

  auto dl_data = static_cast<T*>(dl.untyped_data());
  auto d_data = static_cast<T*>(d.untyped_data());
  auto du_data = static_cast<T*>(du.untyped_data());
  auto b_data = static_cast<T*>(b.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  if (b_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, b_data, b.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  for (int64_t i = 0; i < batch; ++i) {
    JAX_FFI_RETURN_IF_GPU_ERROR(kernel(handle.get(), m, n, dl_data, d_data,
                                       du_data, out_data, m, workspace));
    dl_data += m;
    d_data += m;
    du_data += m;
    out_data += m * n;
  }
  return ffi::Error::Success();
}

template <typename T, typename BufferSizeF, typename KernelF>
ffi::Error Gtsv2BatchedImpl(BufferSizeF getBufferSize, KernelF kernel,
                            int64_t batch, int64_t rows, gpuStream_t stream,
                            ffi::ScratchAllocator& scratch, ffi::AnyBuffer dl,
                            ffi::AnyBuffer d, ffi::AnyBuffer du,
                            ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out) {
  FFI_ASSIGN_OR_RETURN(auto batch_count, MaybeCastNoOverflow<int>(batch));
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));
  size_t buffer_size_in_bytes;
  JAX_FFI_RETURN_IF_GPU_ERROR(getBufferSize(handle.get(), m, nullptr, nullptr,
                                            nullptr, nullptr, batch_count, m,
                                            &buffer_size_in_bytes));
  auto maybe_workspace = scratch.Allocate(buffer_size_in_bytes);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for gtsv2");
  }
  void* workspace = maybe_workspace.value();

  auto dl_data = static_cast<T*>(dl.untyped_data());
  auto d_data = static_cast<T*>(d.untyped_data());
  auto du_data = static_cast<T*>(du.untyped_data());
  auto b_data = static_cast<T*>(b.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  if (b_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, b_data, b.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(kernel(handle.get(), m, dl_data, d_data, du_data,
                                     out_data, batch_count, m, workspace));
  return ffi::Error::Success();
}

ffi::Error Gtsv2(gpuStream_t stream, ffi::ScratchAllocator scratch,
                 ffi::AnyBuffer dl, ffi::AnyBuffer d, ffi::AnyBuffer du,
                 ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out) {
  auto dataType = dl.element_type();
  if (dataType != d.element_type() || dataType != du.element_type() ||
      dataType != b.element_type() || dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to gtsv2 must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(b.dimensions()));
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "gtsv2"));
  FFI_RETURN_IF_ERROR(
      CheckShape(dl.dimensions(), {batch, rows}, "dl", "gtsv2"));
  FFI_RETURN_IF_ERROR(CheckShape(d.dimensions(), {batch, rows}, "d", "gtsv2"));
  FFI_RETURN_IF_ERROR(
      CheckShape(du.dimensions(), {batch, rows}, "du", "gtsv2"));
  if (batch > 1 && cols == 1) {
    switch (dataType) {
      case ffi::F32:
        return Gtsv2BatchedImpl<float>(
            gpusparseSgtsv2StridedBatch_bufferSizeExt,
            gpusparseSgtsv2StridedBatch, batch, rows, stream, scratch, dl, d,
            du, b, out);
      case ffi::F64:
        return Gtsv2BatchedImpl<double>(
            gpusparseDgtsv2StridedBatch_bufferSizeExt,
            gpusparseDgtsv2StridedBatch, batch, rows, stream, scratch, dl, d,
            du, b, out);
      case ffi::C64:
        return Gtsv2BatchedImpl<gpuComplex>(
            gpusparseCgtsv2StridedBatch_bufferSizeExt,
            gpusparseCgtsv2StridedBatch, batch, rows, stream, scratch, dl, d,
            du, b, out);
      case ffi::C128:
        return Gtsv2BatchedImpl<gpuDoubleComplex>(
            gpusparseZgtsv2StridedBatch_bufferSizeExt,
            gpusparseZgtsv2StridedBatch, batch, rows, stream, scratch, dl, d,
            du, b, out);
      default:
        break;
    }

  } else {
    switch (dataType) {
      case ffi::F32:
        return Gtsv2Impl<float>(gpusparseSgtsv2_bufferSizeExt, gpusparseSgtsv2,
                                batch, rows, cols, stream, scratch, dl, d, du,
                                b, out);
      case ffi::F64:
        return Gtsv2Impl<double>(gpusparseDgtsv2_bufferSizeExt, gpusparseDgtsv2,
                                 batch, rows, cols, stream, scratch, dl, d, du,
                                 b, out);
      case ffi::C64:
        return Gtsv2Impl<gpuComplex>(gpusparseCgtsv2_bufferSizeExt,
                                     gpusparseCgtsv2, batch, rows, cols, stream,
                                     scratch, dl, d, du, b, out);
      case ffi::C128:
        return Gtsv2Impl<gpuDoubleComplex>(gpusparseZgtsv2_bufferSizeExt,
                                           gpusparseZgtsv2, batch, rows, cols,
                                           stream, scratch, dl, d, du, b, out);
      default:
        break;
    }
  }
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in gtsv2", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kGtsv2, Gtsv2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // dl
                                  .Arg<ffi::AnyBuffer>()  // d
                                  .Arg<ffi::AnyBuffer>()  // du
                                  .Arg<ffi::AnyBuffer>()  // b
                                  .Ret<ffi::AnyBuffer>()  // out
);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
