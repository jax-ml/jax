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

#include "rocm/include/hipsparse.h"

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
#include "rocm/include/hip/hip_complex.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "jaxlib/rocm/hip_gpu_kernel_helpers.h"
#include "jaxlib/rocm/hipsparse_kernels.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"

namespace py = pybind11;

namespace jax {
namespace {

hipsparseIndexType_t DtypeToHipSparseIndexType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, hipsparseIndexType_t>({
          {{'u', 2}, HIPSPARSE_INDEX_16U},
          {{'i', 4}, HIPSPARSE_INDEX_32I},
          {{'i', 8}, HIPSPARSE_INDEX_64I},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported index dtype: %s", py::repr(np_type)));
  }
  return it->second;
}

// TODO(rocm): add more hip data types when supported
hipDataType DtypeToHipDataType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, hipDataType>(
          {{{'f', 2}, HIP_R_16F},
           {{'c', 4}, HIP_C_16F},
           {{'f', 4}, HIP_R_32F},
           {{'c', 8}, HIP_C_32F},
           {{'f', 8}, HIP_R_64F},
           {{'c', 16}, HIP_C_64F}});
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported data dtype: %s", py::repr(np_type)));
  }
  return it->second;
}
// Returns the descriptor for a Sparse matrix.
SparseMatDescriptor BuildSparseMatDescriptor(const py::dtype& data_dtype,
                                             const py::dtype& index_dtype,
                                             int rows, int cols, int nnz) {
  hipDataType value_type = DtypeToHipDataType(data_dtype);
  hipsparseIndexType_t index_type = DtypeToHipSparseIndexType(index_dtype);
  return SparseMatDescriptor{value_type, index_type, rows, cols, nnz};
}

// Returns the descriptor for a Dense matrix.
DenseMatDescriptor BuildDenseMatDescriptor(const py::dtype& data_dtype,
                                           int rows, int cols) {
  hipDataType value_type = DtypeToHipDataType(data_dtype);
  return DenseMatDescriptor{value_type, rows, cols};
}

// Returns the descriptor for a Dense vector.
DenseVecDescriptor BuildDenseVecDescriptor(const py::dtype& data_dtype,
                                           int size) {
  hipDataType value_type = DtypeToHipDataType(data_dtype);
  return DenseVecDescriptor{value_type, size};
}


// CsrToDense: Convert CSR matrix to dense matrix

// Returns the descriptor for a Sparse matrix.
std::pair<size_t, py::bytes> BuildCsrToDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  hipsparseSpMatDescr_t mat_a = 0;
  hipsparseDnMatDescr_t mat_b = 0;

  // buffer_size does not reference these pointers, but does error on NULL.
  // TODO(jakevdp): check whether this is documented.
  int val = 0;
  void* empty = &val;

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateCsr(
      &mat_a, d.rows, d.cols, d.nnz, empty, empty, empty, d.index_type,
      d.index_type, HIPSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, HIPSPARSE_ORDER_ROW)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

absl::Status CsrToDense_(hipStream_t stream, void** buffers,
                         const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  hipsparseSpMatDescr_t mat_a = 0;
  hipsparseDnMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      hipsparseCreateCsr(&mat_a, d.rows, d.cols, d.nnz,
                        /*csrRowOffsets=*/buffers[2],
                        /*csrColInd=*/buffers[1],
                        /*csrValues=*/buffers[0], d.index_type, d.index_type,
                        HIPSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, buffers[3], d.value_type, HIPSPARSE_ORDER_ROW)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      hipsparseSparseToDense(handle.get(), mat_a, mat_b,
                            HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, buffers[4])));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_b)));
  return absl::OkStatus();
}

void CsrToDense(hipStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrToDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrFromDense: Convert dense matrix to CSR matrix

// Returns the descriptor for a CsrFromDense operation.
std::pair<size_t, py::bytes> BuildCsrFromDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  hipsparseDnMatDescr_t mat_a = 0;
  hipsparseSpMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, HIPSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateCsr(
      &mat_b, d.rows, d.cols, d.nnz, empty, empty, empty, d.index_type,
      d.index_type, HIPSPARSE_INDEX_BASE_ZERO, d.value_type)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

absl::Status CsrFromDense_(hipStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  hipsparseDnMatDescr_t mat_a = 0;
  hipsparseSpMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, buffers[0], d.value_type, HIPSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      hipsparseCreateCsr(&mat_b, d.rows, d.cols, d.nnz,
                        /*csrRowOffsets=*/buffers[3],
                        /*csrColInd=*/buffers[2],
                        /*csrValues=*/buffers[1], d.index_type, d.index_type,
                        HIPSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_b)));
  return absl::OkStatus();
}

void CsrFromDense(hipStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrFromDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrMatvec: Product of CSR matrix and dense vector.

// Returns the descriptor for a CsrMatvec operation.
std::pair<size_t, py::bytes> BuildCsrMatvecDescriptor(
    const py::dtype& data_dtype, const py::dtype& x_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz, bool transpose) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseVecDescriptor x =
      BuildDenseVecDescriptor(x_dtype, transpose ? rows : cols);
  DenseVecDescriptor y =
      BuildDenseVecDescriptor(compute_dtype, transpose ? cols : rows);

  hipsparseSpMatDescr_t mat_a = 0;
  hipsparseDnVecDescr_t vec_x = 0;
  hipsparseDnVecDescr_t vec_y = 0;
  hipsparseOperation_t op = transpose ? HIPSPARSE_OPERATION_TRANSPOSE
                                     : HIPSPARSE_OPERATION_NON_TRANSPOSE;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateCsr(
      &mat_a, A.rows, A.cols, A.nnz, empty, empty, empty, A.index_type,
      A.index_type, HIPSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnVec(&vec_x, x.size, empty, x.type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnVec(&vec_y, y.size, empty, y.type)));
  size_t buffer_size;
  HipConst alpha = HipOne(y.type);
  HipConst beta = HipZero(y.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y.type,
      HIPSPARSE_MV_ALG_DEFAULT, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnVec(vec_x)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnVec(vec_y)));

  return {buffer_size, PackDescriptor(CsrMatvecDescriptor{A, x, y, op})};
}

// CsrMatmat: Product of CSR matrix and dense matrix.

// Returns the descriptor for a CsrMatmat operation.
std::pair<size_t, py::bytes> BuildCsrMatmatDescriptor(
    const py::dtype& data_dtype, const py::dtype& b_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int BCcols, int nnz, bool transpose) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseMatDescriptor B =
      BuildDenseMatDescriptor(b_dtype, transpose ? rows : cols, BCcols);
  DenseMatDescriptor C =
      BuildDenseMatDescriptor(compute_dtype, transpose ? cols : rows, BCcols);
  hipsparseOperation_t op_A = transpose ? HIPSPARSE_OPERATION_TRANSPOSE
                                       : HIPSPARSE_OPERATION_NON_TRANSPOSE;

  hipsparseSpMatDescr_t mat_a = 0;
  hipsparseDnMatDescr_t mat_b = 0;
  hipsparseDnMatDescr_t mat_c = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateCsr(
      &mat_a, A.rows, A.cols, A.nnz, empty, empty, empty, A.index_type,
      A.index_type, HIPSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnMat(&mat_b, B.rows, B.cols, /*ld=*/B.cols,
                                        empty, B.type, HIPSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnMat(&mat_c, C.rows, C.cols, /*ld=*/C.cols,
                                        empty, C.type, HIPSPARSE_ORDER_ROW)));
  size_t buffer_size;
  HipConst alpha = HipOne(C.type);
  HipConst beta = HipZero(C.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseSpMM_bufferSize(
      handle.get(), op_A, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a,
      mat_b, &beta, mat_c, C.type, HIPSPARSE_SPMM_ALG_DEFAULT, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_b)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_c)));

  return {buffer_size, PackDescriptor(CsrMatmatDescriptor{A, B, C, op_A})};
}

// CooToDense: Convert COO matrix to dense matrix

// Returns the descriptor for a CooToDense operation.
std::pair<size_t, py::bytes> BuildCooToDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  hipsparseSpMatDescr_t mat_a = 0;
  hipsparseDnMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(
      hipsparseCreateCoo(&mat_a, d.rows, d.cols, d.nnz, empty, empty, empty,
                        d.index_type, HIPSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, HIPSPARSE_ORDER_ROW)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

// CooFromDense: Convert dense matrix to COO matrix

// Returns the descriptor for a CooFromDense operation.
std::pair<size_t, py::bytes> BuildCooFromDenseDescriptor(
    const py::dtype& data_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);

  hipsparseDnMatDescr_t mat_a = 0;
  hipsparseSpMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, HIPSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(
      hipsparseCreateCoo(&mat_b, d.rows, d.cols, d.nnz, empty, empty, empty,
                        d.index_type, HIPSPARSE_INDEX_BASE_ZERO, d.value_type)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

// CooMatvec: Product of COO matrix and dense vector.

// Returns the descriptor for a CooMatvec operation.
std::pair<size_t, py::bytes> BuildCooMatvecDescriptor(
    const py::dtype& data_dtype, const py::dtype& x_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int nnz, bool transpose) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseVecDescriptor x =
      BuildDenseVecDescriptor(x_dtype, transpose ? rows : cols);
  DenseVecDescriptor y =
      BuildDenseVecDescriptor(compute_dtype, transpose ? cols : rows);

  hipsparseSpMatDescr_t mat_a = 0;
  hipsparseDnVecDescr_t vec_x = 0;
  hipsparseDnVecDescr_t vec_y = 0;
  hipsparseOperation_t op = transpose ? HIPSPARSE_OPERATION_TRANSPOSE
                                     : HIPSPARSE_OPERATION_NON_TRANSPOSE;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(
      hipsparseCreateCoo(&mat_a, A.rows, A.cols, A.nnz, empty, empty, empty,
                        A.index_type, HIPSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnVec(&vec_x, x.size, empty, x.type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnVec(&vec_y, y.size, empty, y.type)));
  size_t buffer_size;
  HipConst alpha = HipOne(y.type);
  HipConst beta = HipZero(y.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y.type,
      HIPSPARSE_MV_ALG_DEFAULT, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnVec(vec_x)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnVec(vec_y)));

  return {buffer_size, PackDescriptor(CooMatvecDescriptor{A, x, y, op})};
}

// CooMatmat: Product of COO matrix and dense matrix.

// Returns the descriptor for a CooMatmat operation.
std::pair<size_t, py::bytes> BuildCooMatmatDescriptor(
    const py::dtype& data_dtype, const py::dtype& b_dtype,
    const py::dtype& compute_dtype, const py::dtype& index_dtype, int rows,
    int cols, int BCcols, int nnz, bool transpose) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz);
  DenseMatDescriptor B =
      BuildDenseMatDescriptor(b_dtype, transpose ? rows : cols, BCcols);
  DenseMatDescriptor C =
      BuildDenseMatDescriptor(compute_dtype, transpose ? cols : rows, BCcols);
  hipsparseOperation_t op_A = transpose ? HIPSPARSE_OPERATION_TRANSPOSE
                                       : HIPSPARSE_OPERATION_NON_TRANSPOSE;

  hipsparseSpMatDescr_t mat_a = 0;
  hipsparseDnMatDescr_t mat_b = 0;
  hipsparseDnMatDescr_t mat_c = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(
      hipsparseCreateCoo(&mat_a, A.rows, A.cols, A.nnz, empty, empty, empty,
                        A.index_type, HIPSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnMat(&mat_b, B.rows, B.cols, /*ld=*/B.cols,
                                        empty, B.type, HIPSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(hipsparseCreateDnMat(&mat_c, C.rows, C.cols, /*ld=*/C.cols,
                                        empty, C.type, HIPSPARSE_ORDER_ROW)));
  size_t buffer_size;
  HipConst alpha = HipOne(C.type);
  HipConst beta = HipZero(C.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseSpMM_bufferSize(
      handle.get(), op_A, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a,
      mat_b, &beta, mat_c, C.type, HIPSPARSE_SPMM_ALG_DEFAULT, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_b)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipsparseDestroyDnMat(mat_c)));

  return {buffer_size, PackDescriptor(CooMatmatDescriptor{A, B, C, op_A})};
}


py::bytes BuildGtsv2Descriptor(int m, int n, int ldb) {
  return PackDescriptor(Gtsv2Descriptor{m, n, ldb});
}

template <typename F>
size_t Gtsv2BufferSize(F f, int m, int n, int ldb) {
  auto h = SparseHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  size_t size;
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(f(handle.get(), m, n, /*dl=*/nullptr, /*d=*/nullptr,
                      /*du=*/nullptr, /*B=*/nullptr, ldb, &size)));
  return size;
}

size_t Gtsv2BufferSizeF32(int m, int n, int ldb) {
  return Gtsv2BufferSize(hipsparseSgtsv2_bufferSizeExt, m, n, ldb);
}

size_t Gtsv2BufferSizeF64(int m, int n, int ldb) {
  return Gtsv2BufferSize(hipsparseDgtsv2_bufferSizeExt, m, n, ldb);
}

py::dict Registrations() {
  py::dict dict;
  dict["hipsparse_csr_todense"] = EncapsulateFunction(CsrToDense);
  dict["hipsparse_csr_fromdense"] = EncapsulateFunction(CsrFromDense);
  dict["hipsparse_csr_matvec"] = EncapsulateFunction(CsrMatvec);
  dict["hipsparse_csr_matmat"] = EncapsulateFunction(CsrMatmat);
  dict["hipsparse_coo_todense"] = EncapsulateFunction(CooToDense);
  dict["hipsparse_coo_fromdense"] = EncapsulateFunction(CooFromDense);
  dict["hipsparse_coo_matvec"] = EncapsulateFunction(CooMatvec);
  dict["hipsparse_coo_matmat"] = EncapsulateFunction(CooMatmat);
  dict["hipsparse_gtsv2_f32"] = EncapsulateFunction(gtsv2_f32);
  dict["hipsparse_gtsv2_f64"] = EncapsulateFunction(gtsv2_f64);
  // TODO(tomhennigan): Add support for gtsv2 complex 32/64.
  return dict;
}

PYBIND11_MODULE(_hipsparse, m) {
  m.attr("hipsparse_supported") = py::bool_(true);
  m.def("registrations", &Registrations);
  m.def("build_csr_todense_descriptor", &BuildCsrToDenseDescriptor);
  m.def("build_csr_fromdense_descriptor", &BuildCsrFromDenseDescriptor);
  m.def("build_csr_matvec_descriptor", &BuildCsrMatvecDescriptor);
  m.def("build_csr_matmat_descriptor", &BuildCsrMatmatDescriptor);
  m.def("build_coo_todense_descriptor", &BuildCooToDenseDescriptor);
  m.def("build_coo_fromdense_descriptor", &BuildCooFromDenseDescriptor);
  m.def("build_coo_matvec_descriptor", &BuildCooMatvecDescriptor);
  m.def("build_coo_matmat_descriptor", &BuildCooMatmatDescriptor);
  m.def("gtsv2_f32_buffer_size", &Gtsv2BufferSizeF32);
  m.def("gtsv2_f64_buffer_size", &Gtsv2BufferSizeF64);
  m.def("build_gtsv2_descriptor", &BuildGtsv2Descriptor);
}

}  // namespace
}  // namespace jax
