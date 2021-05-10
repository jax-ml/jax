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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusparse.h"
#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"

// Some functionality defined here is only available in CUSPARSE 11.3 or newer.
#define JAX_ENABLE_CUSPARSE (CUSPARSE_VERSION >= 11300)

namespace jax {
namespace {

namespace py = pybind11;

void ThrowIfErrorStatus(cusparseStatus_t status) {
  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      return;
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      throw std::runtime_error("cuSparse has not been initialized");
    case CUSPARSE_STATUS_ALLOC_FAILED:
      throw std::runtime_error("cuSparse allocation failure");
    case CUSPARSE_STATUS_INVALID_VALUE:
      throw std::runtime_error("cuSparse invalid value error");
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      throw std::runtime_error("cuSparse architecture mismatch");
    case CUSPARSE_STATUS_MAPPING_ERROR:
      throw std::runtime_error("cuSparse mapping error");
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      throw std::runtime_error("cuSparse execution failed");
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      throw std::runtime_error("cuSparse internal error");
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      throw std::runtime_error("cuSparse matrix type not supported error");
    case CUSPARSE_STATUS_ZERO_PIVOT:
      throw std::runtime_error("cuSparse zero pivot error");
    default:
      throw std::runtime_error("Unknown cuSparse error");
  }
}

union CudaConst {
  int8_t i8[2];
  int16_t i16[2];
  int32_t i32[2];
  int64_t i64[2];
  uint8_t u8[2];
  uint16_t u16[2];
  uint32_t u32[2];
  uint64_t u64[2];
  float f32[2];
  double f64[2];
};

CudaConst CudaZero(cudaDataType type) {
  CudaConst c;
  std::memset(&c, 0, sizeof(c));
  return c;
}

CudaConst CudaOne(cudaDataType type) {
  CudaConst c;
  std::memset(&c, 0, sizeof(c));
  switch (type) {
#if JAX_ENABLE_CUSPARSE
    // TODO(jakevdp): 4I/4U here might break on big endian platforms.
    case CUDA_R_4I:
    case CUDA_C_4I:
#endif
    case CUDA_R_8I:
    case CUDA_C_8I:
      c.i8[0] = 1;
      break;
#if JAX_ENABLE_CUSPARSE
    case CUDA_R_4U:
    case CUDA_C_4U:
#endif
    case CUDA_R_8U:
    case CUDA_C_8U:
      c.u8[0] = 1;
      break;
#if JAX_ENABLE_CUSPARSE
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
#if JAX_ENABLE_CUSPARSE
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
#if JAX_ENABLE_CUSPARSE
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

using SparseHandlePool = HandlePool<cusparseHandle_t, cudaStream_t>;

template <>
/*static*/ SparseHandlePool::Handle SparseHandlePool::Borrow(
    cudaStream_t stream) {
  SparseHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cusparseHandle_t handle;
  if (pool->handles_.empty()) {
    ThrowIfErrorStatus(cusparseCreate(&handle));
  } else {
    handle = pool->handles_.back();
    pool->handles_.pop_back();
  }
  if (stream) {
    ThrowIfErrorStatus(cusparseSetStream(handle, stream));
  }
  return Handle(pool, handle);
}

cusparseIndexType_t DtypeToCuSparseIndexType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, cusparseIndexType_t>({
          {{'u', 2}, CUSPARSE_INDEX_16U},
          {{'i', 4}, CUSPARSE_INDEX_32I},
          {{'i', 8}, CUSPARSE_INDEX_64I},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported index dtype: %s", py::repr(np_type)));
  }
  return it->second;
}

cudaDataType DtypeToCudaDataType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, cudaDataType>({
        {{'f', 2}, CUDA_R_16F}, {{'f', 4}, CUDA_R_32F}, {{'f', 4}, CUDA_R_32F},
            {{'c', 8}, CUDA_C_32F}, {{'f', 8}, CUDA_R_64F},
            {{'c', 16}, CUDA_C_64F}, {{'i', 1}, CUDA_R_8I},
            {{'u', 1}, CUDA_R_8U}, {{'i', 4}, CUDA_R_32I},
            {{'u', 4}, CUDA_R_32U},
#if JAX_ENABLE_CUSPARSE
            {{'V', 2}, CUDA_R_16BF},
#endif
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported data dtype: %s", py::repr(np_type)));
  }
  return it->second;
}

struct SparseMatDescriptor {
  cudaDataType value_type;
  cusparseIndexType_t index_type;
  int rows, cols, nnz;
};

struct DenseMatDescriptor {
  cudaDataType type;
  int rows, cols;
};

struct DenseVecDescriptor {
  cudaDataType type;
  int size;
};

// Returns the descriptor for a Sparse matrix.
SparseMatDescriptor BuildSparseMatDescriptor(const py::dtype& data_dtype,
                                             const py::dtype& index_dtype,
                                             int rows, int cols, int nnz) {
  cudaDataType value_type = DtypeToCudaDataType(data_dtype);
  cusparseIndexType_t index_type = DtypeToCuSparseIndexType(index_dtype);
  return SparseMatDescriptor{value_type, index_type, rows, cols, nnz};
}

// Returns the descriptor for a Dense matrix.
DenseMatDescriptor BuildDenseMatDescriptor(const py::dtype& data_dtype,
                                           int rows, int cols) {
  cudaDataType value_type = DtypeToCudaDataType(data_dtype);
  return DenseMatDescriptor{value_type, rows, cols};
}

// Returns the descriptor for a Dense vector.
DenseVecDescriptor BuildDenseVecDescriptor(const py::dtype& data_dtype,
                                           int size) {
  cudaDataType value_type = DtypeToCudaDataType(data_dtype);
  return DenseVecDescriptor{value_type, size};
}

#if JAX_ENABLE_CUSPARSE
// CsrToDense: Convert CSR matrix to dense matrix

// Returns the descriptor for a Sparse matrix.
std::pair<size_t, py::bytes> BuildCsrToDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;

  // buffer_size does not reference these pointers, but does error on NULL.
  // TODO(jakevdp): check whether this is documented.
  int val = 0;
  void* empty = &val;

  ThrowIfErrorStatus(cusparseCreateCsr(&mat_a, d.rows, d.cols, d.nnz, empty,
                                       empty, empty, d.index_type, d.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, d.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, d.rows, d.cols,
                                         /*ld=*/d.cols, empty, d.value_type,
                                         CUSPARSE_ORDER_ROW));
  size_t buffer_size;
  ThrowIfErrorStatus(cusparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));

  return {buffer_size, PackDescriptor(d)};
}

void CsrToDense(cudaStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len) {
  const SparseMatDescriptor& d =
      *UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  ThrowIfErrorStatus(cusparseCreateCsr(&mat_a, d.rows, d.cols, d.nnz,
                                       /*csrRowOffsets=*/buffers[2],
                                       /*csrColInd=*/buffers[1],
                                       /*csrValues=*/buffers[0], d.index_type,
                                       d.index_type, CUSPARSE_INDEX_BASE_ZERO,
                                       d.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, d.rows, d.cols,
                                         /*ld=*/d.cols, buffers[3],
                                         d.value_type, CUSPARSE_ORDER_ROW));

  ThrowIfErrorStatus(cusparseSparseToDense(handle.get(), mat_a, mat_b,
                                           CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                           buffers[4]));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));
}

// CsrFromDense: Convert dense matrix to CSR matrix

// Returns the descriptor for a CsrFromDense operation.
std::pair<size_t, py::bytes> BuildCsrFromDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  cusparseDnMatDescr_t mat_a = 0;
  cusparseSpMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_a, d.rows, d.cols,
                                         /*ld=*/d.cols, empty, d.value_type,
                                         CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateCsr(&mat_b, d.rows, d.cols, d.nnz, empty,
                                       empty, empty, d.index_type, d.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, d.value_type));
  size_t buffer_size;
  ThrowIfErrorStatus(cusparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size));

  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroySpMat(mat_b));

  return {buffer_size, PackDescriptor(d)};
}

void CsrFromDense(cudaStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len) {
  const SparseMatDescriptor& d =
      *UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

  cusparseDnMatDescr_t mat_a = 0;
  cusparseSpMatDescr_t mat_b = 0;
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_a, d.rows, d.cols,
                                         /*ld=*/d.cols, buffers[0],
                                         d.value_type, CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateCsr(&mat_b, d.rows, d.cols, d.nnz,
                                       /*csrRowOffsets=*/buffers[3],
                                       /*csrColInd=*/buffers[2],
                                       /*csrValues=*/buffers[1], d.index_type,
                                       d.index_type, CUSPARSE_INDEX_BASE_ZERO,
                                       d.value_type));
  ThrowIfErrorStatus(cusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4]));
  ThrowIfErrorStatus(cusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4]));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroySpMat(mat_b));
}

// CsrMatvec: Product of CSR matrix and dense vector.

struct CsrMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  cusparseOperation_t op;
};

// Returns the descriptor for a CsrMatvec operation.
std::pair<size_t, py::bytes> BuildCsrMatvecDescriptor(
    const py::dtype& data_dtype, const py::dtype& x_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz, bool transpose) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseVecDescriptor x =
      BuildDenseVecDescriptor(x_dtype, transpose ? rows : cols);
  DenseVecDescriptor y =
      BuildDenseVecDescriptor(compute_dtype, transpose ? cols : rows);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnVecDescr_t vec_x = 0;
  cusparseDnVecDescr_t vec_y = 0;
  cusparseOperation_t op = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                     : CUSPARSE_OPERATION_NON_TRANSPOSE;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  ThrowIfErrorStatus(cusparseCreateCsr(&mat_a, A.rows, A.cols, A.nnz, empty,
                                       empty, empty, A.index_type, A.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_x, x.size, empty, x.type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_y, y.size, empty, y.type));
  size_t buffer_size;
  CudaConst alpha = CudaOne(y.type);
  CudaConst beta = CudaZero(y.type);
  ThrowIfErrorStatus(cusparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y.type,
      CUSPARSE_MV_ALG_DEFAULT, &buffer_size));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_x));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_y));

  return {buffer_size, PackDescriptor(CsrMatvecDescriptor{A, x, y, op})};
}

void CsrMatvec(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len) {
  const CsrMatvecDescriptor& d =
      *UnpackDescriptor<CsrMatvecDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

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

  ThrowIfErrorStatus(
      cusparseCreateCsr(&mat_a, d.A.rows, d.A.cols, d.A.nnz, csr_row_offsets,
                        csr_col_ind, csr_values, d.A.index_type, d.A.index_type,
                        CUSPARSE_INDEX_BASE_ZERO, d.A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_x, d.x.size, xbuf, d.x.type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_y, d.y.size, ybuf, d.y.type));

  ThrowIfErrorStatus(cusparseSpMV(handle.get(), d.op, &alpha, mat_a, vec_x,
                                  &beta, vec_y, d.y.type,
                                  CUSPARSE_MV_ALG_DEFAULT, buf));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_x));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_y));
}

// CsrMatmat: Product of CSR matrix and dense matrix.

struct CsrMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  cusparseOperation_t op_A;
};

// Returns the descriptor for a CsrMatmat operation.
std::pair<size_t, py::bytes> BuildCsrMatmatDescriptor(
    const py::dtype& data_dtype, const py::dtype& b_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int BCcols, int nnz, bool transpose) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseMatDescriptor B =
      BuildDenseMatDescriptor(b_dtype, transpose ? rows : cols, BCcols);
  DenseMatDescriptor C =
      BuildDenseMatDescriptor(compute_dtype, transpose ? cols : rows, BCcols);
  cusparseOperation_t op_A = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                       : CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  cusparseDnMatDescr_t mat_c = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  ThrowIfErrorStatus(cusparseCreateCsr(&mat_a, A.rows, A.cols, A.nnz, empty,
                                       empty, empty, A.index_type, A.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, B.rows, B.cols, /*ld=*/B.cols,
                                         empty, B.type, CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_c, C.rows, C.cols, /*ld=*/C.cols,
                                         empty, C.type, CUSPARSE_ORDER_ROW));
  size_t buffer_size;
  CudaConst alpha = CudaOne(C.type);
  CudaConst beta = CudaZero(C.type);
  ThrowIfErrorStatus(cusparseSpMM_bufferSize(
      handle.get(), op_A, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a,
      mat_b, &beta, mat_c, C.type, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_c));

  return {buffer_size, PackDescriptor(CsrMatmatDescriptor{A, B, C, op_A})};
}

void CsrMatmat(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len) {
  const CsrMatmatDescriptor& d =
      *UnpackDescriptor<CsrMatmatDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

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

  ThrowIfErrorStatus(
      cusparseCreateCsr(&mat_a, d.A.rows, d.A.cols, d.A.nnz, csr_row_offsets,
                        csr_col_ind, csr_values, d.A.index_type, d.A.index_type,
                        CUSPARSE_INDEX_BASE_ZERO, d.A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, d.B.rows, d.B.cols,
                                         /*ld=*/d.B.cols, Bbuf, d.B.type,
                                         CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_c, d.C.rows, d.C.cols,
                                         /*ld=*/d.C.cols, Cbuf, d.C.type,
                                         CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseSpMM(
      handle.get(), d.op_A, /*opB=*/CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, d.C.type, CUSPARSE_SPMM_ALG_DEFAULT, buf));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_c));
}

// CooToDense: Convert COO matrix to dense matrix

// Returns the descriptor for a CooToDense operation.
std::pair<size_t, py::bytes> BuildCooToDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;

  ThrowIfErrorStatus(cusparseCreateCoo(&mat_a, d.rows, d.cols, d.nnz, empty,
                                       empty, empty, d.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, d.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, d.rows, d.cols,
                                         /*ld=*/d.cols, empty, d.value_type,
                                         CUSPARSE_ORDER_ROW));
  size_t buffer_size;
  ThrowIfErrorStatus(cusparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));

  return {buffer_size, PackDescriptor(d)};
}

void CooToDense(cudaStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len) {
  const SparseMatDescriptor& d =
      *UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  ThrowIfErrorStatus(cusparseCreateCoo(&mat_a, d.rows, d.cols, d.nnz,
                                       /*cooRowInd=*/buffers[1],
                                       /*cooColInd=*/buffers[2],
                                       /*cooValues=*/buffers[0], d.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, d.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, d.rows, d.cols,
                                         /*ld=*/d.cols, buffers[3],
                                         d.value_type, CUSPARSE_ORDER_ROW));

  ThrowIfErrorStatus(cusparseSparseToDense(handle.get(), mat_a, mat_b,
                                           CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                           buffers[4]));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));
}

// CooFromDense: Convert dense matrix to COO matrix

// Returns the descriptor for a CooFromDense operation.
std::pair<size_t, py::bytes> BuildCooFromDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  cusparseDnMatDescr_t mat_a = 0;
  cusparseSpMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_a, d.rows, d.cols,
                                         /*ld=*/d.cols, empty, d.value_type,
                                         CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateCoo(&mat_b, d.rows, d.cols, d.nnz, empty,
                                       empty, empty, d.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, d.value_type));
  size_t buffer_size;
  ThrowIfErrorStatus(cusparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size));

  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroySpMat(mat_b));

  return {buffer_size, PackDescriptor(d)};
}

void CooFromDense(cudaStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len) {
  const SparseMatDescriptor& d =
      *UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

  cusparseDnMatDescr_t mat_a = 0;
  cusparseSpMatDescr_t mat_b = 0;
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_a, d.rows, d.cols,
                                         /*ld=*/d.cols, buffers[0],
                                         d.value_type, CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateCoo(&mat_b, d.rows, d.cols, d.nnz,
                                       /*cooRowInd=*/buffers[2],
                                       /*cooColInd=*/buffers[3],
                                       /*cooValues=*/buffers[1], d.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, d.value_type));
  ThrowIfErrorStatus(cusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4]));
  ThrowIfErrorStatus(cusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4]));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroySpMat(mat_b));
}

// CooMatvec: Product of COO matrix and dense vector.

struct CooMatvecDescriptor {
  SparseMatDescriptor A;
  DenseVecDescriptor x, y;
  cusparseOperation_t op;
};

// Returns the descriptor for a CooMatvec operation.
std::pair<size_t, py::bytes> BuildCooMatvecDescriptor(
    const py::dtype& data_dtype, const py::dtype& x_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz, bool transpose) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseVecDescriptor x =
      BuildDenseVecDescriptor(x_dtype, transpose ? rows : cols);
  DenseVecDescriptor y =
      BuildDenseVecDescriptor(compute_dtype, transpose ? cols : rows);

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnVecDescr_t vec_x = 0;
  cusparseDnVecDescr_t vec_y = 0;
  cusparseOperation_t op = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                     : CUSPARSE_OPERATION_NON_TRANSPOSE;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  ThrowIfErrorStatus(cusparseCreateCoo(&mat_a, A.rows, A.cols, A.nnz, empty,
                                       empty, empty, A.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_x, x.size, empty, x.type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_y, y.size, empty, y.type));
  size_t buffer_size;
  CudaConst alpha = CudaOne(y.type);
  CudaConst beta = CudaZero(y.type);
  ThrowIfErrorStatus(cusparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y.type,
      CUSPARSE_MV_ALG_DEFAULT, &buffer_size));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_x));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_y));

  return {buffer_size, PackDescriptor(CooMatvecDescriptor{A, x, y, op})};
}

void CooMatvec(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len) {
  const CooMatvecDescriptor& d =
      *UnpackDescriptor<CooMatvecDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

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

  ThrowIfErrorStatus(cusparseCreateCoo(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, coo_row_ind, coo_col_ind, coo_values,
      d.A.index_type, CUSPARSE_INDEX_BASE_ZERO, d.A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_x, d.x.size, xbuf, d.x.type));
  ThrowIfErrorStatus(cusparseCreateDnVec(&vec_y, d.y.size, ybuf, d.y.type));

  ThrowIfErrorStatus(cusparseSpMV(handle.get(), d.op, &alpha, mat_a, vec_x,
                                  &beta, vec_y, d.y.type,
                                  CUSPARSE_MV_ALG_DEFAULT, buf));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_x));
  ThrowIfErrorStatus(cusparseDestroyDnVec(vec_y));
}

// CooMatmat: Product of COO matrix and dense matrix.

struct CooMatmatDescriptor {
  SparseMatDescriptor A;
  DenseMatDescriptor B, C;
  cusparseOperation_t op_A;
};

// Returns the descriptor for a CooMatmat operation.
std::pair<size_t, py::bytes> BuildCooMatmatDescriptor(
    const py::dtype& data_dtype, const py::dtype& b_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int BCcols, int nnz, bool transpose) {
  auto handle = SparseHandlePool::Borrow();
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseMatDescriptor B =
      BuildDenseMatDescriptor(b_dtype, transpose ? rows : cols, BCcols);
  DenseMatDescriptor C =
      BuildDenseMatDescriptor(compute_dtype, transpose ? cols : rows, BCcols);
  cusparseOperation_t op_A = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                       : CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusparseSpMatDescr_t mat_a = 0;
  cusparseDnMatDescr_t mat_b = 0;
  cusparseDnMatDescr_t mat_c = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  ThrowIfErrorStatus(cusparseCreateCoo(&mat_a, A.rows, A.cols, A.nnz, empty,
                                       empty, empty, A.index_type,
                                       CUSPARSE_INDEX_BASE_ZERO, A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, B.rows, B.cols, /*ld=*/B.cols,
                                         empty, B.type, CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_c, C.rows, C.cols, /*ld=*/C.cols,
                                         empty, C.type, CUSPARSE_ORDER_ROW));
  size_t buffer_size;
  CudaConst alpha = CudaOne(C.type);
  CudaConst beta = CudaZero(C.type);
  ThrowIfErrorStatus(cusparseSpMM_bufferSize(
      handle.get(), op_A, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a,
      mat_b, &beta, mat_c, C.type, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_c));

  return {buffer_size, PackDescriptor(CooMatmatDescriptor{A, B, C, op_A})};
}

void CooMatmat(cudaStream_t stream, void** buffers, const char* opaque,
               size_t opaque_len) {
  const CooMatmatDescriptor& d =
      *UnpackDescriptor<CooMatmatDescriptor>(opaque, opaque_len);
  auto handle = SparseHandlePool::Borrow(stream);

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

  ThrowIfErrorStatus(cusparseCreateCoo(
      &mat_a, d.A.rows, d.A.cols, d.A.nnz, coo_row_ind, coo_col_ind, coo_values,
      d.A.index_type, CUSPARSE_INDEX_BASE_ZERO, d.A.value_type));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_b, d.B.rows, d.B.cols,
                                         /*ld=*/d.B.cols, Bbuf, d.B.type,
                                         CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseCreateDnMat(&mat_c, d.C.rows, d.C.cols,
                                         /*ld=*/d.C.cols, Cbuf, d.C.type,
                                         CUSPARSE_ORDER_ROW));
  ThrowIfErrorStatus(cusparseSpMM(
      handle.get(), d.op_A, /*opB=*/CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, d.C.type, CUSPARSE_SPMM_ALG_DEFAULT, buf));

  ThrowIfErrorStatus(cusparseDestroySpMat(mat_a));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_b));
  ThrowIfErrorStatus(cusparseDestroyDnMat(mat_c));
}

#endif

py::dict Registrations() {
  py::dict dict;
#if JAX_ENABLE_CUSPARSE
  dict["cusparse_csr_todense"] = EncapsulateFunction(CsrToDense);
  dict["cusparse_csr_fromdense"] = EncapsulateFunction(CsrFromDense);
  dict["cusparse_csr_matvec"] = EncapsulateFunction(CsrMatvec);
  dict["cusparse_csr_matmat"] = EncapsulateFunction(CsrMatmat);
  dict["cusparse_coo_todense"] = EncapsulateFunction(CooToDense);
  dict["cusparse_coo_fromdense"] = EncapsulateFunction(CooFromDense);
  dict["cusparse_coo_matvec"] = EncapsulateFunction(CooMatvec);
  dict["cusparse_coo_matmat"] = EncapsulateFunction(CooMatmat);
#endif
  return dict;
}

PYBIND11_MODULE(cusparse_kernels, m) {
  m.attr("cusparse_supported") = py::bool_(JAX_ENABLE_CUSPARSE);
  m.def("registrations", &Registrations);
#if JAX_ENABLE_CUSPARSE
  m.def("build_csr_todense_descriptor", &BuildCsrToDenseDescriptor);
  m.def("build_csr_fromdense_descriptor", &BuildCsrFromDenseDescriptor);
  m.def("build_csr_matvec_descriptor", &BuildCsrMatvecDescriptor);
  m.def("build_csr_matmat_descriptor", &BuildCsrMatmatDescriptor);
  m.def("build_coo_todense_descriptor", &BuildCooToDenseDescriptor);
  m.def("build_coo_fromdense_descriptor", &BuildCooFromDenseDescriptor);
  m.def("build_coo_matvec_descriptor", &BuildCooMatvecDescriptor);
  m.def("build_coo_matmat_descriptor", &BuildCooMatmatDescriptor);
#endif
}

}  // namespace
}  // namespace jax
