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
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace ffi = ::xla::ffi;

namespace jax {

template <>
/*static*/ absl::StatusOr<SparseHandlePool::Handle> SparseHandlePool::Borrow(
    gpuStream_t stream) {
  SparseHandlePool* pool = Instance();
  absl::MutexLock lock(pool->mu_);
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
    // TODO(jakevdp): 4I/4U here might break on big endian platforms.
    case CUDA_R_4I:
    case CUDA_C_4I:
    case CUDA_R_8I:
    case CUDA_C_8I:
      c.i8[0] = 1;
      break;
    case CUDA_R_4U:
    case CUDA_C_4U:
    case CUDA_R_8U:
    case CUDA_C_8U:
      c.u8[0] = 1;
      break;
    case CUDA_R_16I:
    case CUDA_C_16I:
      c.i16[0] = 1;
      break;
    case CUDA_R_16U:
    case CUDA_C_16U:
      c.u16[0] = 1;
      break;
    case CUDA_R_32I:
    case CUDA_C_32I:
      c.i32[0] = 1;
      break;
    case CUDA_R_32U:
    case CUDA_C_32U:
      c.u32[0] = 1;
      break;
    case CUDA_R_64I:
    case CUDA_C_64I:
      c.i64[0] = 1;
      break;
    case CUDA_R_64U:
    case CUDA_C_64U:
      c.u64[0] = 1;
      break;
#if JAX_GPU_HAVE_FP8
    case CUDA_R_8F_E4M3:
      c.u8[0] = __nv_cvt_float_to_fp8(1.0f, __NV_NOSAT, __NV_E4M3);
      break;
    case CUDA_R_8F_E5M2:
      c.u8[0] = __nv_cvt_float_to_fp8(1.0f, __NV_NOSAT, __NV_E5M2);
      break;
#endif
    case CUDA_R_16BF:
    case CUDA_C_16BF:
      c.u16[0] = 0b11111110000000;  // 1.0 in little-endian bfloat16
      break;
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

inline absl::StatusOr<gpuDataType> SparseDataType(ffi::DataType dataType) {
  switch (dataType) {
    case ffi::F16:
      return GPU_R_16F;
    case ffi::F32:
      return GPU_R_32F;
    case ffi::C64:
      return GPU_C_32F;
    case ffi::F64:
      return GPU_R_64F;
    case ffi::C128:
      return GPU_C_64F;
#ifdef JAX_GPU_CUDA
    case ffi::S8:
      return CUDA_R_8I;
    case ffi::U8:
      return CUDA_R_8U;
    case ffi::S32:
      return CUDA_R_32I;
    case ffi::U32:
      return CUDA_R_32U;
    case ffi::BF16:
      return CUDA_R_16BF;
#endif  // JAX_GPU_CUDA
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported sparse data dtype %s", absl::FormatStreamed(dataType)));
  }
}

inline absl::StatusOr<gpusparseIndexType_t> SparseIndexType(
    ffi::DataType dataType) {
  switch (dataType) {
    case ffi::U16:
      return GPUSPARSE_INDEX_16U;
    case ffi::S32:
      return GPUSPARSE_INDEX_32I;
    case ffi::S64:
      return GPUSPARSE_INDEX_64I;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported sparse index dtype %s", absl::FormatStreamed(dataType)));
  }
}

// CsrToDense: Convert CSR matrix to dense matrix

static ffi::Error CsrToDenseFfiV2(gpuStream_t stream,
                                  ffi::ScratchAllocator scratch,
                                  ffi::AnyBuffer csr_values,
                                  ffi::AnyBuffer csr_col_ind,
                                  ffi::AnyBuffer csr_row_offsets,
                                  ffi::Result<ffi::AnyBuffer> out) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(csr_values.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(csr_col_ind.element_type()));
  int64_t rows = out->dimensions()[0];
  int64_t cols = out->dimensions()[1];
  int64_t nnz = csr_values.dimensions()[0];

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateCsr(
      &mat_a, rows, cols, nnz, csr_row_offsets.untyped_data(),
      csr_col_ind.untyped_data(), csr_values.untyped_data(), index_type,
      index_type, GPUSPARSE_INDEX_BASE_ZERO, value_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_b, rows, cols,
      /*ld=*/cols, out->untyped_data(), value_type, GPUSPARSE_ORDER_ROW));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for csr_todense");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSparseToDense(
      handle.get(), mat_a, mat_b, GPUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      maybe_workspace.value()));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_b));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCsrToDense, CsrToDenseFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // csr_values
                                  .Arg<ffi::AnyBuffer>()  // csr_col_ind
                                  .Arg<ffi::AnyBuffer>()  // csr_row_offsets
                                  .Ret<ffi::AnyBuffer>()  // out
);

// CsrFromDense: Convert dense matrix to CSR matrix

static ffi::Error CsrFromDenseFfiV2(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer dense,
    ffi::Result<ffi::AnyBuffer> csr_values,
    ffi::Result<ffi::AnyBuffer> csr_col_ind,
    ffi::Result<ffi::AnyBuffer> csr_row_offsets) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(dense.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(csr_col_ind->element_type()));
  int64_t rows = dense.dimensions()[0];
  int64_t cols = dense.dimensions()[1];
  int64_t nnz = csr_values->dimensions()[0];

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  gpusparseDnMatDescr_t mat_a = 0;
  gpusparseSpMatDescr_t mat_b = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_a, rows, cols,
      /*ld=*/cols, dense.untyped_data(), value_type, GPUSPARSE_ORDER_ROW));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateCsr(
      &mat_b, rows, cols, nnz, csr_row_offsets->untyped_data(),
      csr_col_ind->untyped_data(), csr_values->untyped_data(), index_type,
      index_type, GPUSPARSE_INDEX_BASE_ZERO, value_type));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal(
        "Unable to allocate workspace for csr_fromdense");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      maybe_workspace.value()));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      maybe_workspace.value()));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_b));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCsrFromDense, CsrFromDenseFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // dense
                                  .Ret<ffi::AnyBuffer>()  // csr_values
                                  .Ret<ffi::AnyBuffer>()  // csr_col_ind
                                  .Ret<ffi::AnyBuffer>()  // csr_row_offsets
);

// CsrMatvec: Product of CSR matrix and dense vector.

static ffi::Error CsrMatvecFfiV2(
    gpuStream_t stream, ffi::ScratchAllocator scratch,
    ffi::AnyBuffer csr_values, ffi::AnyBuffer csr_col_ind,
    ffi::AnyBuffer csr_row_offsets, ffi::AnyBuffer x,
    ffi::Result<ffi::AnyBuffer> y, bool transpose) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(csr_values.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(csr_col_ind.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType x_type, SparseDataType(x.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType y_type, SparseDataType(y->element_type()));
  int64_t nnz = csr_values.dimensions()[0];
  int64_t x_size = x.dimensions()[0];
  int64_t y_size = y->dimensions()[0];
  int64_t rows = transpose ? x_size : y_size;
  int64_t cols = transpose ? y_size : x_size;
  gpusparseOperation_t op = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                      : GPUSPARSE_OPERATION_NON_TRANSPOSE;

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  FFI_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(y_type));
  SparseConst beta = ConstZero(y_type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnVecDescr_t vec_x = 0;
  gpusparseDnVecDescr_t vec_y = 0;

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateCsr(
      &mat_a, rows, cols, nnz, csr_row_offsets.untyped_data(),
      csr_col_ind.untyped_data(), csr_values.untyped_data(), index_type,
      index_type, GPUSPARSE_INDEX_BASE_ZERO, value_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseCreateDnVec(&vec_x, x_size, x.untyped_data(), x_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseCreateDnVec(&vec_y, y_size, y->untyped_data(), y_type));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y_type,
      GPUSPARSE_SPMV_CSR_ALG, &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for csr_matvec");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseSpMV(handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y,
                    y_type, GPUSPARSE_SPMV_CSR_ALG, maybe_workspace.value()));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnVec(vec_x));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnVec(vec_y));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCsrMatvec, CsrMatvecFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // csr_values
                                  .Arg<ffi::AnyBuffer>()  // csr_col_ind
                                  .Arg<ffi::AnyBuffer>()  // csr_row_offsets
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Ret<ffi::AnyBuffer>()  // y
                                  .Attr<bool>("transpose"));

// CsrMatmat: Product of CSR matrix and dense matrix.

static ffi::Error CsrMatmatFfiV2(
    gpuStream_t stream, ffi::ScratchAllocator scratch,
    ffi::AnyBuffer csr_values, ffi::AnyBuffer csr_col_ind,
    ffi::AnyBuffer csr_row_offsets, ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> C, bool transpose) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(csr_values.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(csr_col_ind.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType B_type, SparseDataType(B.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType C_type, SparseDataType(C->element_type()));
  int64_t nnz = csr_values.dimensions()[0];
  int64_t B_rows = B.dimensions()[0];
  int64_t B_cols = B.dimensions()[1];
  int64_t C_rows = C->dimensions()[0];
  int64_t C_cols = C->dimensions()[1];
  int64_t rows = transpose ? B_rows : C_rows;
  int64_t cols = transpose ? C_rows : B_rows;
  gpusparseOperation_t op_A = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                        : GPUSPARSE_OPERATION_NON_TRANSPOSE;

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  FFI_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(C_type));
  SparseConst beta = ConstZero(C_type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  gpusparseDnMatDescr_t mat_c = 0;

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateCsr(
      &mat_a, rows, cols, nnz, csr_row_offsets.untyped_data(),
      csr_col_ind.untyped_data(), csr_values.untyped_data(), index_type,
      index_type, GPUSPARSE_INDEX_BASE_ZERO, value_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_b, B_rows, B_cols,
      /*ld=*/B_cols, B.untyped_data(), B_type, GPUSPARSE_ORDER_ROW));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_c, C_rows, C_cols,
      /*ld=*/C_cols, C->untyped_data(), C_type, GPUSPARSE_ORDER_ROW));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSpMM_bufferSize(
      handle.get(), op_A, /*opB=*/GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, C_type, GPUSPARSE_SPMM_CSR_ALG,
      &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for csr_matmat");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSpMM(
      handle.get(), op_A, /*opB=*/GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, C_type, GPUSPARSE_SPMM_CSR_ALG,
      maybe_workspace.value()));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_b));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_c));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCsrMatmat, CsrMatmatFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // csr_values
                                  .Arg<ffi::AnyBuffer>()  // csr_col_ind
                                  .Arg<ffi::AnyBuffer>()  // csr_row_offsets
                                  .Arg<ffi::AnyBuffer>()  // B
                                  .Ret<ffi::AnyBuffer>()  // C
                                  .Attr<bool>("transpose"));

// CooToDense: Convert COO matrix to dense matrix

static ffi::Error CooToDenseFfiV2(gpuStream_t stream,
                                  ffi::ScratchAllocator scratch,
                                  ffi::AnyBuffer coo_values,
                                  ffi::AnyBuffer coo_row_ind,
                                  ffi::AnyBuffer coo_col_ind,
                                  ffi::Result<ffi::AnyBuffer> out) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(coo_values.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(coo_row_ind.element_type()));
  int64_t rows = out->dimensions()[0];
  int64_t cols = out->dimensions()[1];
  int64_t nnz = coo_values.dimensions()[0];

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseCreateCoo(&mat_a, rows, cols, nnz, coo_row_ind.untyped_data(),
                         coo_col_ind.untyped_data(), coo_values.untyped_data(),
                         index_type, GPUSPARSE_INDEX_BASE_ZERO, value_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_b, rows, cols,
      /*ld=*/cols, out->untyped_data(), value_type, GPUSPARSE_ORDER_ROW));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for coo_todense");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSparseToDense(
      handle.get(), mat_a, mat_b, GPUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      maybe_workspace.value()));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_b));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCooToDense, CooToDenseFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // coo_values
                                  .Arg<ffi::AnyBuffer>()  // coo_row_ind
                                  .Arg<ffi::AnyBuffer>()  // coo_col_ind
                                  .Ret<ffi::AnyBuffer>()  // out
);

// CooFromDense: Convert dense matrix to COO matrix

static ffi::Error CooFromDenseFfiV2(gpuStream_t stream,
                                    ffi::ScratchAllocator scratch,
                                    ffi::AnyBuffer dense,
                                    ffi::Result<ffi::AnyBuffer> coo_values,
                                    ffi::Result<ffi::AnyBuffer> coo_row_ind,
                                    ffi::Result<ffi::AnyBuffer> coo_col_ind) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(dense.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(coo_row_ind->element_type()));
  int64_t rows = dense.dimensions()[0];
  int64_t cols = dense.dimensions()[1];
  int64_t nnz = coo_values->dimensions()[0];

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  gpusparseDnMatDescr_t mat_a = 0;
  gpusparseSpMatDescr_t mat_b = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_a, rows, cols,
      /*ld=*/cols, dense.untyped_data(), value_type, GPUSPARSE_ORDER_ROW));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateCoo(
      &mat_b, rows, cols, nnz, coo_row_ind->untyped_data(),
      coo_col_ind->untyped_data(), coo_values->untyped_data(), index_type,
      GPUSPARSE_INDEX_BASE_ZERO, value_type));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal(
        "Unable to allocate workspace for coo_fromdense");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      maybe_workspace.value()));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      maybe_workspace.value()));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_b));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCooFromDense, CooFromDenseFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // dense
                                  .Ret<ffi::AnyBuffer>()  // coo_values
                                  .Ret<ffi::AnyBuffer>()  // coo_row_ind
                                  .Ret<ffi::AnyBuffer>()  // coo_col_ind
);

// CooMatvec: Product of COO matrix and dense vector.

static ffi::Error CooMatvecFfiV2(gpuStream_t stream,
                                 ffi::ScratchAllocator scratch,
                                 ffi::AnyBuffer coo_values,
                                 ffi::AnyBuffer coo_row_ind,
                                 ffi::AnyBuffer coo_col_ind, ffi::AnyBuffer x,
                                 ffi::Result<ffi::AnyBuffer> y,
                                 bool transpose) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(coo_values.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(coo_row_ind.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType x_type, SparseDataType(x.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType y_type, SparseDataType(y->element_type()));
  int64_t nnz = coo_values.dimensions()[0];
  int64_t x_size = x.dimensions()[0];
  int64_t y_size = y->dimensions()[0];
  int64_t rows = transpose ? x_size : y_size;
  int64_t cols = transpose ? y_size : x_size;
  gpusparseOperation_t op = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                      : GPUSPARSE_OPERATION_NON_TRANSPOSE;

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  FFI_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(y_type));
  SparseConst beta = ConstZero(y_type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnVecDescr_t vec_x = 0;
  gpusparseDnVecDescr_t vec_y = 0;

  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseCreateCoo(&mat_a, rows, cols, nnz, coo_row_ind.untyped_data(),
                         coo_col_ind.untyped_data(), coo_values.untyped_data(),
                         index_type, GPUSPARSE_INDEX_BASE_ZERO, value_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseCreateDnVec(&vec_x, x_size, x.untyped_data(), x_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseCreateDnVec(&vec_y, y_size, y->untyped_data(), y_type));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y_type,
      GPUSPARSE_SPMV_COO_ALG, &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for coo_matvec");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseSpMV(handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y,
                    y_type, GPUSPARSE_SPMV_COO_ALG, maybe_workspace.value()));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnVec(vec_x));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnVec(vec_y));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCooMatvec, CooMatvecFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // coo_values
                                  .Arg<ffi::AnyBuffer>()  // coo_row_ind
                                  .Arg<ffi::AnyBuffer>()  // coo_col_ind
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Ret<ffi::AnyBuffer>()  // y
                                  .Attr<bool>("transpose"));

// CooMatmat: Product of COO matrix and dense matrix.

static ffi::Error CooMatmatFfiV2(gpuStream_t stream,
                                 ffi::ScratchAllocator scratch,
                                 ffi::AnyBuffer coo_values,
                                 ffi::AnyBuffer coo_row_ind,
                                 ffi::AnyBuffer coo_col_ind, ffi::AnyBuffer B,
                                 ffi::Result<ffi::AnyBuffer> C,
                                 bool transpose) {
  FFI_ASSIGN_OR_RETURN(gpuDataType value_type,
                       SparseDataType(coo_values.element_type()));
  FFI_ASSIGN_OR_RETURN(gpusparseIndexType_t index_type,
                       SparseIndexType(coo_row_ind.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType B_type, SparseDataType(B.element_type()));
  FFI_ASSIGN_OR_RETURN(gpuDataType C_type, SparseDataType(C->element_type()));
  int64_t batch_count = 1;
  int64_t B_rows, B_cols, C_rows, C_cols;
  if (B.dimensions().size() == 2) {
    B_rows = B.dimensions()[0];
    B_cols = B.dimensions()[1];
    C_rows = C->dimensions()[0];
    C_cols = C->dimensions()[1];
  } else if (B.dimensions().size() == 3) {
    batch_count = B.dimensions()[0];
    B_rows = B.dimensions()[1];
    B_cols = B.dimensions()[2];
    C_rows = C->dimensions()[1];
    C_cols = C->dimensions()[2];
  } else {
    return ffi::Error::InvalidArgument(absl::StrFormat(
        "Unsupported rank %d for B in coo_matmat", B.dimensions().size()));
  }
  int64_t nnz = coo_values.element_count() / batch_count;
  int64_t rows = transpose ? B_rows : C_rows;
  int64_t cols = transpose ? C_rows : B_rows;
  gpusparseOperation_t op_A = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                        : GPUSPARSE_OPERATION_NON_TRANSPOSE;
  int64_t lhs_batch_stride = nnz;
  int64_t rhs_batch_stride = B_rows * B_cols;
  int64_t c_batch_stride = C_rows * C_cols;

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  // TODO(jakevdp): alpha and beta should be user-specifiable, but constants
  // are sufficient for basic matvec operations.
  // Note that, contrary to cusparse docs, alpha and beta must be host pointers
  // or else the operation will segfault.
  FFI_ASSIGN_OR_RETURN(SparseConst alpha, ConstOne(C_type));
  SparseConst beta = ConstZero(C_type);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  gpusparseDnMatDescr_t mat_c = 0;

  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpusparseCreateCoo(&mat_a, rows, cols, nnz, coo_row_ind.untyped_data(),
                         coo_col_ind.untyped_data(), coo_values.untyped_data(),
                         index_type, GPUSPARSE_INDEX_BASE_ZERO, value_type));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCooSetStridedBatch(
      mat_a, /*batchCount=*/batch_count, /*batchStride=*/lhs_batch_stride));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_b, B_rows, B_cols,
      /*ld=*/B_cols, B.untyped_data(), B_type, GPUSPARSE_ORDER_ROW));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDnMatSetStridedBatch(
      mat_b, /*batchCount=*/batch_count, /*batchStride=*/rhs_batch_stride));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseCreateDnMat(
      &mat_c, C_rows, C_cols,
      /*ld=*/C_cols, C->untyped_data(), C_type, GPUSPARSE_ORDER_ROW));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDnMatSetStridedBatch(
      mat_c, /*batchCount=*/batch_count, /*batchStride=*/c_batch_stride));

  size_t buffer_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSpMM_bufferSize(
      handle.get(), op_A, /*opB=*/GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, C_type, GPUSPARSE_SPMM_COO_ALG,
      &buffer_size));
  auto maybe_workspace = scratch.Allocate(buffer_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal("Unable to allocate workspace for coo_matmat");
  }

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseSpMM(
      handle.get(), op_A, /*opB=*/GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      mat_a, mat_b, &beta, mat_c, C_type, GPUSPARSE_SPMM_COO_ALG,
      maybe_workspace.value()));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroySpMat(mat_a));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_b));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusparseDestroyDnMat(mat_c));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kCooMatmat, CooMatmatFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // coo_values
                                  .Arg<ffi::AnyBuffer>()  // coo_row_ind
                                  .Arg<ffi::AnyBuffer>()  // coo_col_ind
                                  .Arg<ffi::AnyBuffer>()  // B
                                  .Ret<ffi::AnyBuffer>()  // C
                                  .Attr<bool>("transpose"));

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
