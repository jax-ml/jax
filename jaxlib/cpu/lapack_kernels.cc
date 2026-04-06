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

#include "jaxlib/cpu/lapack_kernels.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "jaxlib/ffi_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/future.h"

namespace ffi = xla::ffi;

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::Side);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::Transpose);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::Diag);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::MatrixParams::UpLo);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::svd::ComputationMode);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::eig::ComputationMode);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::schur::ComputationMode);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::schur::Sort);

namespace jax {

bool lapack_kernels_initialized = false;

template <typename T>
inline T CastNoOverflow(int64_t value, std::string_view source = __FILE__) {
  auto result = MaybeCastNoOverflow<T>(value, source);
  if (!result.ok()) {
    throw std::overflow_error{std::string(result.status().message())};
  }
  return result.value();
}

template <ffi::DataType dtype>
void CopyIfDiffBuffer(ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out) {
  if (x.typed_data() != x_out->typed_data()) {
    const auto x_size = x.element_count();
    std::copy_n(x.typed_data(), x_size, x_out->typed_data());
  }
}

//== Triangular System Solver ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error TriMatrixEquationSolver<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, ffi::Buffer<dtype> y, ffi::ResultBuffer<dtype> y_out,
    MatrixParams::Side side, MatrixParams::UpLo uplo,
    MatrixParams::Transpose trans_x, MatrixParams::Diag diag) {
  CopyIfDiffBuffer(y, y_out);
  FFI_ASSIGN_OR_RETURN((auto [batch_count, y_rows, y_cols]),
                       SplitBatch2D(y.dimensions()));
  auto* y_out_data = y_out->typed_data();
  IntType x_leading_dim_v = side == MatrixParams::Side::kLeft ? y_rows : y_cols;
  IntType y_leading_dim_v = y_rows;

  auto side_v = static_cast<char>(side);
  auto uplo_v = static_cast<char>(uplo);
  auto trans_x_v = static_cast<char>(trans_x);
  auto diag_v = static_cast<char>(diag);
  FFI_ASSIGN_OR_RETURN(auto y_rows_v, MaybeCastNoOverflow<IntType>(y_rows));
  FFI_ASSIGN_OR_RETURN(auto y_cols_v, MaybeCastNoOverflow<IntType>(y_cols));

  auto* x_data = x.typed_data();
  const int64_t y_out_step{y_rows * y_cols};
  const int64_t x_step{x_leading_dim_v * x_leading_dim_v};
  ffi::NativeType<dtype> alpha = static_cast<ffi::NativeType<dtype>>(1);
  for (int64_t i = 0; i < batch_count; ++i) {
    TriMatrixEquationSolver<dtype, IntType>::fn(
        &side_v, &uplo_v, &trans_x_v, &diag_v, &y_rows_v, &y_cols_v, &alpha,
        x_data, &x_leading_dim_v, y_out_data, &y_leading_dim_v);

    y_out_data += y_out_step;
    x_data += x_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error TriMatrixEquationSolverKernel(
    ffi::Buffer<dtype> x, ffi::Buffer<dtype> y, ffi::ResultBuffer<dtype> y_out,
    MatrixParams::Side side, MatrixParams::UpLo uplo,
    MatrixParams::Transpose trans_x, MatrixParams::Diag diag) {
  if (TriMatrixEquationSolver<dtype, int64_t>::fn != nullptr) {
    return TriMatrixEquationSolver<dtype, int64_t>::Kernel(x, y, y_out, side,
                                                           uplo, trans_x, diag);
  }
  return TriMatrixEquationSolver<dtype, int32_t>::Kernel(x, y, y_out, side,
                                                         uplo, trans_x, diag);
}

template struct TriMatrixEquationSolver<ffi::DataType::F32, int32_t>;
template struct TriMatrixEquationSolver<ffi::DataType::F32, int64_t>;
template struct TriMatrixEquationSolver<ffi::DataType::F64, int32_t>;
template struct TriMatrixEquationSolver<ffi::DataType::F64, int64_t>;
template struct TriMatrixEquationSolver<ffi::DataType::C64, int32_t>;
template struct TriMatrixEquationSolver<ffi::DataType::C64, int64_t>;
template struct TriMatrixEquationSolver<ffi::DataType::C128, int32_t>;
template struct TriMatrixEquationSolver<ffi::DataType::C128, int64_t>;

//== LU Decomposition ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error LuDecomposition<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<LapackIntDtype> ipiv,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* ipiv_data = ipiv->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<IntType>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t ipiv_step{std::min(x_rows, x_cols)};
  std::unique_ptr<IntType[]> ipiv_tmp;
  if constexpr (!std::is_same_v<IntType, int32_t>) {
    ipiv_tmp = std::make_unique<IntType[]>(ipiv_step);
  }
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    if constexpr (std::is_same_v<IntType, int32_t>) {
      LuDecomposition<dtype, IntType>::fn(&x_rows_v, &x_cols_v, x_out_data,
                                          &x_leading_dim_v, ipiv_data, &info_v);
    } else {
      LuDecomposition<dtype, IntType>::fn(&x_rows_v, &x_cols_v, x_out_data,
                                          &x_leading_dim_v, ipiv_tmp.get(),
                                          &info_v);
      for (int64_t j = 0; j < ipiv_step; ++j) {
        ipiv_data[j] = static_cast<int32_t>(ipiv_tmp[j]);
      }
    }
    // Suppress MSAN warnings when using a copy of LAPACK uninstrumented by
    // MSAN.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);

    using T = ffi::NativeType<dtype>;
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data,
                                        x_rows * x_cols * sizeof(T));
    if constexpr (std::is_same_v<IntType, int32_t>) {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(ipiv_data,
                                          ipiv_step * sizeof(int32_t));
    } else {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(ipiv_tmp.get(),
                                          ipiv_step * sizeof(IntType));
    }

    x_out_data += x_out_step;
    ipiv_data += ipiv_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error LuDecompositionKernel(ffi::Buffer<dtype> x,
                                 ffi::ResultBuffer<dtype> x_out,
                                 ffi::ResultBuffer<LapackIntDtype> ipiv,
                                 ffi::ResultBuffer<LapackIntDtype> info) {
  if (LuDecomposition<dtype, int64_t>::fn != nullptr) {
    return LuDecomposition<dtype, int64_t>::Kernel(x, x_out, ipiv, info);
  }
  return LuDecomposition<dtype, int32_t>::Kernel(x, x_out, ipiv, info);
}

template struct LuDecomposition<ffi::DataType::F32, int32_t>;
template struct LuDecomposition<ffi::DataType::F32, int64_t>;
template struct LuDecomposition<ffi::DataType::F64, int32_t>;
template struct LuDecomposition<ffi::DataType::F64, int64_t>;
template struct LuDecomposition<ffi::DataType::C64, int32_t>;
template struct LuDecomposition<ffi::DataType::C64, int64_t>;
template struct LuDecomposition<ffi::DataType::C128, int32_t>;
template struct LuDecomposition<ffi::DataType::C128, int64_t>;

//== QR Factorization ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error QrFactorization<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> tau) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* tau_data = tau->typed_data();
  IntType info;
  const int64_t work_size =
      QrFactorization<dtype, IntType>::GetWorkspaceSize(x_rows, x_cols);
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  CopyIfDiffBuffer(x, x_out);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<IntType>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    QrFactorization<dtype, IntType>::fn(
        &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, tau_data,
        work_data.get(), &workspace_dim_v, &info);
    x_out_data += x_out_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error QrFactorizationKernel(ffi::Buffer<dtype> x,
                                 ffi::ResultBuffer<dtype> x_out,
                                 ffi::ResultBuffer<dtype> tau) {
  if (QrFactorization<dtype, int64_t>::fn != nullptr) {
    return QrFactorization<dtype, int64_t>::Kernel(x, x_out, tau);
  }
  return QrFactorization<dtype, int32_t>::Kernel(x, x_out, tau);
}

template <ffi::DataType dtype, typename IntType>
int64_t QrFactorization<dtype, IntType>::GetWorkspaceSize(IntType x_rows,
                                                          IntType x_cols) {
  ValueType optimal_size{};
  IntType x_leading_dim_v = x_rows;
  IntType info = 0;
  IntType workspace_query = -1;
  fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct QrFactorization<ffi::DataType::F32, int32_t>;
template struct QrFactorization<ffi::DataType::F32, int64_t>;
template struct QrFactorization<ffi::DataType::F64, int32_t>;
template struct QrFactorization<ffi::DataType::F64, int64_t>;
template struct QrFactorization<ffi::DataType::C64, int32_t>;
template struct QrFactorization<ffi::DataType::C64, int64_t>;
template struct QrFactorization<ffi::DataType::C128, int32_t>;
template struct QrFactorization<ffi::DataType::C128, int64_t>;

//== Column Pivoting QR Factorization ==//

// lapack geqp3
template <ffi::DataType dtype, typename IntType>
ffi::Error PivotingQrFactorization<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, ffi::Buffer<LapackIntDtype> jpvt,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<LapackIntDtype> jpvt_out,
    ffi::ResultBuffer<dtype> tau) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* jpvt_out_data = jpvt_out->typed_data();
  auto* tau_data = tau->typed_data();

  const int64_t work_size =
      PivotingQrFactorization<dtype, IntType>::GetWorkspaceSize(x_rows, x_cols);
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  constexpr bool is_complex_dtype = ffi::IsComplexType<dtype>();
  std::unique_ptr<ffi::NativeType<ffi::ToReal(dtype)>[]> rwork_data;
  if constexpr (is_complex_dtype) {
    rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(2 * x_cols);
  }

  CopyIfDiffBuffer(x, x_out);
  CopyIfDiffBuffer(jpvt, jpvt_out);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<IntType>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t jpvt_step{x_cols};
  const int64_t tau_step{std::min(x_rows, x_cols)};
  std::unique_ptr<IntType[]> jpvt_tmp;
  if constexpr (!std::is_same_v<IntType, int32_t>) {
    jpvt_tmp = std::make_unique<IntType[]>(jpvt_step);
  }
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    if constexpr (std::is_same_v<IntType, int32_t>) {
      if constexpr (is_complex_dtype) {
        PivotingQrFactorization<dtype, IntType>::fn(
            &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, jpvt_out_data,
            tau_data, work_data.get(), &workspace_dim_v, rwork_data.get(),
            &info_v);
      } else {
        PivotingQrFactorization<dtype, IntType>::fn(
            &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, jpvt_out_data,
            tau_data, work_data.get(), &workspace_dim_v, &info_v);
      }
    } else {
      for (int64_t j = 0; j < jpvt_step; ++j) {
        jpvt_tmp[j] = static_cast<IntType>(jpvt_out_data[j]);
      }
      if constexpr (is_complex_dtype) {
        PivotingQrFactorization<dtype, IntType>::fn(
            &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, jpvt_tmp.get(),
            tau_data, work_data.get(), &workspace_dim_v, rwork_data.get(),
            &info_v);
      } else {
        PivotingQrFactorization<dtype, IntType>::fn(
            &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, jpvt_tmp.get(),
            tau_data, work_data.get(), &workspace_dim_v, &info_v);
      }
      for (int64_t j = 0; j < jpvt_step; ++j) {
        jpvt_out_data[j] = static_cast<int32_t>(jpvt_tmp[j]);
      }
    }
    x_out_data += x_out_step;
    jpvt_out_data += jpvt_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error PivotingQrFactorizationKernel(
    ffi::Buffer<dtype> x, ffi::Buffer<LapackIntDtype> jpvt,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<LapackIntDtype> jpvt_out,
    ffi::ResultBuffer<dtype> tau) {
  if (PivotingQrFactorization<dtype, int64_t>::fn != nullptr) {
    return PivotingQrFactorization<dtype, int64_t>::Kernel(x, jpvt, x_out,
                                                           jpvt_out, tau);
  }
  return PivotingQrFactorization<dtype, int32_t>::Kernel(x, jpvt, x_out,
                                                         jpvt_out, tau);
}

template <ffi::DataType dtype, typename IntType>
int64_t PivotingQrFactorization<dtype, IntType>::GetWorkspaceSize(
    IntType x_rows, IntType x_cols) {
  ValueType optimal_size{};
  IntType x_leading_dim_v = x_rows;
  IntType info = 0;
  IntType workspace_query = -1;
  if constexpr (ffi::IsComplexType<dtype>()) {
    fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
       &optimal_size, &workspace_query, nullptr, &info);
  } else {
    fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
       &optimal_size, &workspace_query, &info);
  }
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct PivotingQrFactorization<ffi::DataType::F32, int32_t>;
template struct PivotingQrFactorization<ffi::DataType::F32, int64_t>;
template struct PivotingQrFactorization<ffi::DataType::F64, int32_t>;
template struct PivotingQrFactorization<ffi::DataType::F64, int64_t>;
template struct PivotingQrFactorization<ffi::DataType::C64, int32_t>;
template struct PivotingQrFactorization<ffi::DataType::C64, int64_t>;
template struct PivotingQrFactorization<ffi::DataType::C128, int32_t>;
template struct PivotingQrFactorization<ffi::DataType::C128, int64_t>;

//== Orthogonal QR                                      ==//
//== Computes orthogonal matrix Q from QR Decomposition ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error OrthogonalQr<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, ffi::Buffer<dtype> tau,
    ffi::ResultBuffer<dtype> x_out) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* tau_data = tau.typed_data();
  auto* x_out_data = x_out->typed_data();
  IntType info;

  CopyIfDiffBuffer(x, x_out);

  // Prepare LAPACK workspaces.
  int64_t work_size = OrthogonalQr<dtype, IntType>::GetWorkspaceSize(
      x_rows, x_cols, tau.dimensions().back());
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<IntType>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto tau_size_v,
                       MaybeCastNoOverflow<IntType>(tau.dimensions().back()));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{tau_size_v};
  for (int64_t i = 0; i < batch_count; ++i) {
    OrthogonalQr<dtype, IntType>::fn(&x_rows_v, &x_cols_v, &tau_size_v,
                                     x_out_data, &x_leading_dim_v, tau_data,
                                     work_data.get(), &work_size_v, &info);
    x_out_data += x_out_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error OrthogonalQrKernel(ffi::Buffer<dtype> x, ffi::Buffer<dtype> tau,
                              ffi::ResultBuffer<dtype> x_out) {
  if (OrthogonalQr<dtype, int64_t>::fn != nullptr) {
    return OrthogonalQr<dtype, int64_t>::Kernel(x, tau, x_out);
  }
  return OrthogonalQr<dtype, int32_t>::Kernel(x, tau, x_out);
}

template <ffi::DataType dtype, typename IntType>
int64_t OrthogonalQr<dtype, IntType>::GetWorkspaceSize(IntType x_rows,
                                                       IntType x_cols,
                                                       IntType tau_size) {
  ValueType optimal_size = {};
  IntType x_leading_dim_v = x_rows;
  IntType info = 0;
  IntType workspace_query = -1;
  fn(&x_rows, &x_cols, &tau_size, nullptr, &x_leading_dim_v, nullptr,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct OrthogonalQr<ffi::DataType::F32, int32_t>;
template struct OrthogonalQr<ffi::DataType::F32, int64_t>;
template struct OrthogonalQr<ffi::DataType::F64, int32_t>;
template struct OrthogonalQr<ffi::DataType::F64, int64_t>;
template struct OrthogonalQr<ffi::DataType::C64, int32_t>;
template struct OrthogonalQr<ffi::DataType::C64, int64_t>;
template struct OrthogonalQr<ffi::DataType::C128, int32_t>;
template struct OrthogonalQr<ffi::DataType::C128, int64_t>;

//== Orthogonal QR Multiply ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error OrthogonalQrMultiply<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> a, ffi::Buffer<dtype> tau, ffi::Buffer<dtype> c,
    bool left, bool transpose, ffi::ResultBuffer<dtype> c_out) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, c_rows, c_cols]),
                       SplitBatch2D(c.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [a_batch, a_rows, a_cols]),
                       SplitBatch2D(a.dimensions()));

  auto* tau_data = tau.typed_data();
  auto* a_data = a.typed_data();
  auto* c_out_data = c_out->typed_data();

  CopyIfDiffBuffer(c, c_out);

  char side_v = left ? 'L' : 'R';
  char trans_v;
  if constexpr (ffi::IsComplexType<dtype>()) {
    trans_v = transpose ? 'C' : 'N';
  } else {
    trans_v = transpose ? 'T' : 'N';
  }

  FFI_ASSIGN_OR_RETURN(auto c_rows_v, MaybeCastNoOverflow<IntType>(c_rows));
  FFI_ASSIGN_OR_RETURN(auto c_cols_v, MaybeCastNoOverflow<IntType>(c_cols));
  FFI_ASSIGN_OR_RETURN(auto k_v,
                       MaybeCastNoOverflow<IntType>(tau.dimensions().back()));
  // LDA is the leading dimension of A: m (= c_rows) when left, n (= c_cols)
  // when right. The shape rule guarantees a_rows equals the correct value in
  // both cases, but we compute it explicitly to match the trsm convention.
  FFI_ASSIGN_OR_RETURN(auto lda_v,
                       MaybeCastNoOverflow<IntType>(left ? a_rows : c_cols));

  int64_t work_size =
      GetWorkspaceSize(side_v, trans_v, c_rows_v, c_cols_v, k_v, lda_v);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  auto c_leading_dim_v = c_rows_v;
  IntType info;

  const int64_t c_out_step{c_rows * c_cols};
  const int64_t a_step{a_rows * a_cols};
  const int64_t tau_step{tau.dimensions().back()};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&side_v, &trans_v, &c_rows_v, &c_cols_v, &k_v, a_data, &lda_v, tau_data,
       c_out_data, &c_leading_dim_v, work_data.get(), &work_size_v, &info);
    c_out_data += c_out_step;
    a_data += a_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype, typename IntType>
int64_t OrthogonalQrMultiply<dtype, IntType>::GetWorkspaceSize(
    char side, char trans, IntType m, IntType n, IntType k, IntType lda) {
  ValueType optimal_size = {};
  IntType c_leading_dim = m;
  IntType info = 0;
  IntType workspace_query = -1;
  fn(&side, &trans, &m, &n, &k, nullptr, &lda, nullptr, nullptr, &c_leading_dim,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct OrthogonalQrMultiply<ffi::DataType::F32, int32_t>;
template struct OrthogonalQrMultiply<ffi::DataType::F32, int64_t>;
template struct OrthogonalQrMultiply<ffi::DataType::F64, int32_t>;
template struct OrthogonalQrMultiply<ffi::DataType::F64, int64_t>;
template struct OrthogonalQrMultiply<ffi::DataType::C64, int32_t>;
template struct OrthogonalQrMultiply<ffi::DataType::C64, int64_t>;
template struct OrthogonalQrMultiply<ffi::DataType::C128, int32_t>;
template struct OrthogonalQrMultiply<ffi::DataType::C128, int64_t>;
template <ffi::DataType dtype>
ffi::Error OrthogonalQrMultiplyKernel(ffi::Buffer<dtype> a,
                                      ffi::Buffer<dtype> tau,
                                      ffi::Buffer<dtype> c, bool left,
                                      bool transpose,
                                      ffi::ResultBuffer<dtype> c_out) {
  if (OrthogonalQrMultiply<dtype, int64_t>::fn != nullptr) {
    return OrthogonalQrMultiply<dtype, int64_t>::Kernel(a, tau, c, left,
                                                        transpose, c_out);
  }
  return OrthogonalQrMultiply<dtype, int32_t>::Kernel(a, tau, c, left,
                                                      transpose, c_out);
}

//== Cholesky Factorization ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error CholeskyFactorization<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_order_v,
                       MaybeCastNoOverflow<IntType>(x.dimensions().back()));
  auto x_leading_dim_v = x_order_v;

  const int64_t x_out_step{x_rows * x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    CholeskyFactorization<dtype, IntType>::fn(&uplo_v, &x_order_v, x_out_data,
                                              &x_leading_dim_v, &info_v);
    // Suppress MSAN warnings when using a copy of LAPACK uninstrumented by
    // MSAN.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);

    using T = ffi::NativeType<dtype>;
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data,
                                        x_rows * x_cols * sizeof(T));

    x_out_data += x_out_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error CholeskyFactorizationKernel(ffi::Buffer<dtype> x,
                                       MatrixParams::UpLo uplo,
                                       ffi::ResultBuffer<dtype> x_out,
                                       ffi::ResultBuffer<LapackIntDtype> info) {
  if (CholeskyFactorization<dtype, int64_t>::fn != nullptr) {
    return CholeskyFactorization<dtype, int64_t>::Kernel(x, uplo, x_out, info);
  }
  return CholeskyFactorization<dtype, int32_t>::Kernel(x, uplo, x_out, info);
}

template struct CholeskyFactorization<ffi::DataType::F32, int32_t>;
template struct CholeskyFactorization<ffi::DataType::F32, int64_t>;
template struct CholeskyFactorization<ffi::DataType::F64, int32_t>;
template struct CholeskyFactorization<ffi::DataType::F64, int64_t>;
template struct CholeskyFactorization<ffi::DataType::C64, int32_t>;
template struct CholeskyFactorization<ffi::DataType::C64, int64_t>;
template struct CholeskyFactorization<ffi::DataType::C128, int32_t>;
template struct CholeskyFactorization<ffi::DataType::C128, int64_t>;

//== Singular Value Decomposition (SVD) ==//
//== using a divide and conquer method  ==//

namespace internal {

// TODO(basioli): It might be worth consider parallelizing on even smaller
// matrices if batch size is large enough.
// In this case we wouldn't queue a call per matrix, but a call where we'd
// handle a chunk of the batch (e.x. (10000, 2, 2) could be chunked into e.x.
// (250, 2, 2)).
bool ShouldParallelizeSVD(int64_t batch_size, int64_t rows, int64_t cols,
                          int64_t num_threads) {
  int64_t matrix_size = rows * cols;

  const int64_t kMinMatrixSizeForParallelization = 8 * 8;

  return matrix_size >= kMinMatrixSizeForParallelization && batch_size > 1;
}

template <ffi::DataType dtype, typename IntType>
static ffi::Error SvdKernel(
    ffi::ThreadPool thread_pool, ffi::Buffer<dtype> x,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  if (mode == svd::ComputationMode::kComputeVtOverwriteXPartialU) [[unlikely]] {
    return ffi::Error(
        XLA_FFI_Error_Code_UNIMPLEMENTED,
        "Current implementation does not support this computation mode");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* singular_values_data = singular_values->typed_data();
  auto* u_data = u->typed_data();
  auto* vt_data = vt->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<IntType>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows_v;
  auto u_leading_dim_v = x_rows_v;

  auto u_dims = u->dimensions().last(2);
  auto vt_dims = vt->dimensions().last(2);
  FFI_ASSIGN_OR_RETURN(auto vt_leading_dim_v,
                       MaybeCastNoOverflow<IntType>(vt_dims.front()));

  // Prepare LAPACK workspaces.
  auto work_size_or =
      svd::SVDType<dtype, IntType>::GetWorkspaceSize(x_rows_v, x_cols_v, mode);
  FFI_ASSIGN_OR_RETURN(const auto work_size, work_size_or);
  const auto iwork_size = svd::GetIntWorkspaceSize(x_rows, x_cols);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  int64_t rwork_size = 0;
  if constexpr (ffi::IsComplexType<dtype>()) {
    rwork_size = svd::GetRealWorkspaceSize(x_rows, x_cols, mode);
  }

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t singular_values_step{singular_values->dimensions().back()};
  const int64_t u_step{u_dims.front() * u_dims.back()};
  const int64_t vt_step{vt_leading_dim_v * vt_dims.back()};

  bool should_parallelize = ShouldParallelizeSVD(batch_count, x_rows, x_cols,
                                                 thread_pool.num_threads());

  auto thread_work = [&](auto* x_out_data, auto* singular_values_data,
                         auto* u_data, auto* vt_data, auto* info_data) {
    auto work_data = AllocateScratchMemory<dtype>(work_size);
    auto iwork_data =
        AllocateScratchMemory<LapackIntDtypeFor<IntType>>(iwork_size);
    std::unique_ptr<ffi::NativeType<ffi::ToReal(dtype)>[]> rwork;
    if constexpr (ffi::IsComplexType<dtype>()) {
      rwork = AllocateScratchMemory<ffi::ToReal(dtype)>(rwork_size);
    }

    IntType info_v;
    if constexpr (ffi::IsComplexType<dtype>()) {
      svd::SVDType<dtype, IntType>::fn(
          &mode_v, &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v,
          singular_values_data, u_data, &u_leading_dim_v, vt_data,
          &vt_leading_dim_v, work_data.get(), &workspace_dim_v, rwork.get(),
          iwork_data.get(), &info_v);
    } else {
      svd::SVDType<dtype, IntType>::fn(
          &mode_v, &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v,
          singular_values_data, u_data, &u_leading_dim_v, vt_data,
          &vt_leading_dim_v, work_data.get(), &workspace_dim_v,
          iwork_data.get(), &info_v);
    }

    // Suppress MSAN warnings when using a copy of LAPACK uninstrumented by
    // MSAN.
    using T [[maybe_unused]] = typename svd::SVDType<dtype, IntType>::ValueType;
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data,
                                        x_cols_v * x_leading_dim_v * sizeof(T));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        singular_values_data, std::min(x_rows_v, x_cols_v) *
                                  sizeof(ffi::NativeType<ffi::ToReal(dtype)>));
    if (mode_v == 'A') {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          u_data, u_leading_dim_v * x_rows_v * sizeof(T));
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          vt_data, vt_leading_dim_v * x_cols_v * sizeof(T));
    } else if (mode_v == 'O') {
      if (x_rows_v < x_cols_v) {
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
            u_data, u_leading_dim_v * x_rows_v * sizeof(T));
      } else {
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
            vt_data, vt_leading_dim_v * x_cols_v * sizeof(T));
      }
    } else if (mode_v == 'S') {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          u_data, u_leading_dim_v * std::min(x_rows_v, x_cols_v) * sizeof(T));
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
          vt_data, vt_leading_dim_v * x_cols_v * sizeof(T));
    }
  };

  std::vector<xla::Future<>> futures;
  futures.reserve(batch_count);
  for (int64_t i = 0; i < batch_count; ++i) {
    if (!should_parallelize) {
      thread_work(x_out_data, singular_values_data, u_data, vt_data, info_data);
    } else {
      auto [promise, future] = xla::MakePromise();
      futures.push_back(std::move(future));
      // We copy the thread_work parameters here because they get updated in the
      // loop.
      thread_pool.Schedule([&, promise = std::move(promise), x_out_data,
                            singular_values_data, u_data, vt_data,
                            info_data]() mutable {
        thread_work(x_out_data, singular_values_data, u_data, vt_data,
                    info_data);
        promise.Set();
      });
    }

    x_out_data += x_out_step;
    singular_values_data += singular_values_step;
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }

  if (should_parallelize) {
    absl::Status futures_status = xla::JoinFutures(std::move(futures)).Await();

    if (!futures_status.ok()) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string(futures_status.message()));
    }
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype, typename IntType>
static int64_t SvdGetWorkspaceSize(IntType x_rows, IntType x_cols,
                                   svd::ComputationMode mode) {
  ffi::NativeType<dtype> optimal_size = {};
  IntType info = 0;
  IntType workspace_query = -1;

  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows;
  auto u_leading_dim_v = x_rows;
  auto vt_leading_dim_v = mode == svd::ComputationMode::kComputeFullUVt
                              ? x_cols
                              : std::min(x_rows, x_cols);
  if constexpr (ffi::IsComplexType<dtype>()) {
    svd::SVDType<dtype, IntType>::fn(
        &mode_v, &x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
        &u_leading_dim_v, nullptr, &vt_leading_dim_v, &optimal_size,
        &workspace_query, nullptr, nullptr, &info);
  } else {
    svd::SVDType<dtype, IntType>::fn(
        &mode_v, &x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
        &u_leading_dim_v, nullptr, &vt_leading_dim_v, &optimal_size,
        &workspace_query, nullptr, &info);
  }
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template <ffi::DataType dtype, typename IntType>
static ffi::Error SvdQRKernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  if (mode == svd::ComputationMode::kComputeVtOverwriteXPartialU) [[unlikely]] {
    return ffi::Error(
        XLA_FFI_Error_Code_UNIMPLEMENTED,
        "SVD: Current implementation does not support this computation mode");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* singular_values_data = singular_values->typed_data();
  auto* u_data = u->typed_data();
  auto* vt_data = vt->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<IntType>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows_v;
  auto u_leading_dim_v = x_rows_v;

  // Prepare LAPACK workspaces.
  auto work_size_or = svd::SVDQRType<dtype, IntType>::GetWorkspaceSize(
      x_rows_v, x_cols_v, mode);
  FFI_ASSIGN_OR_RETURN(auto work_size, work_size_or);
  auto workspace_dim_v = work_size;
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  std::unique_ptr<ffi::NativeType<ffi::ToReal(dtype)>[]> rwork;
  if constexpr (ffi::IsComplexType<dtype>()) {
    const auto rwork_size = svd::GetRealWorkspaceSizeQR(x_rows, x_cols);
    rwork = AllocateScratchMemory<ffi::ToReal(dtype)>(rwork_size);
  }

  auto u_dims = u->dimensions().last(2);
  auto vt_dims = vt->dimensions().last(2);
  FFI_ASSIGN_OR_RETURN(auto vt_leading_dim_v,
                       MaybeCastNoOverflow<IntType>(vt_dims.front()));

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t singular_values_step{singular_values->dimensions().back()};
  const int64_t u_step{u_dims.front() * u_dims.back()};
  const int64_t vt_step{vt_leading_dim_v * vt_dims.back()};

  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    if constexpr (ffi::IsComplexType<dtype>()) {
      svd::SVDQRType<dtype, IntType>::fn(
          &mode_v, &mode_v, &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v,
          singular_values_data, u_data, &u_leading_dim_v, vt_data,
          &vt_leading_dim_v, work_data.get(), &workspace_dim_v, rwork.get(),
          &info_v);
    } else {
      svd::SVDQRType<dtype, IntType>::fn(
          &mode_v, &mode_v, &x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v,
          singular_values_data, u_data, &u_leading_dim_v, vt_data,
          &vt_leading_dim_v, work_data.get(), &workspace_dim_v, &info_v);
    }
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);
    x_out_data += x_out_step;
    singular_values_data += singular_values_step;
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype, typename IntType>
static absl::StatusOr<IntType> SvdQRGetWorkspaceSize(
    IntType x_rows, IntType x_cols, svd::ComputationMode mode) {
  ffi::NativeType<dtype> optimal_size = {};
  IntType info = 0;
  IntType workspace_query = -1;

  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows;
  auto u_leading_dim_v = x_rows;
  auto vt_leading_dim_v = mode == svd::ComputationMode::kComputeFullUVt
                              ? x_cols
                              : std::min(x_rows, x_cols);
  if constexpr (ffi::IsComplexType<dtype>()) {
    svd::SVDQRType<dtype, IntType>::fn(
        &mode_v, &mode_v, &x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr,
        nullptr, &u_leading_dim_v, nullptr, &vt_leading_dim_v, &optimal_size,
        &workspace_query, nullptr, &info);
  } else {
    svd::SVDQRType<dtype, IntType>::fn(
        &mode_v, &mode_v, &x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr,
        nullptr, &u_leading_dim_v, nullptr, &vt_leading_dim_v, &optimal_size,
        &workspace_query, &info);
  }
  return info == 0 ? MaybeCastNoOverflow<IntType>(std::real(optimal_size)) : -1;
}

}  // namespace internal

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionKernel(
    ::xla::ffi::ThreadPool thread_pool, ffi::Buffer<dtype> x,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  if (svd::SVDType<dtype, int64_t>::fn != nullptr) {
    return internal::SvdKernel<dtype, int64_t>(
        thread_pool, x, x_out, singular_values, u, vt, info, mode);
  }
  return internal::SvdKernel<dtype, int32_t>(
      thread_pool, x, x_out, singular_values, u, vt, info, mode);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionComplexKernel(
    ::xla::ffi::ThreadPool thread_pool, ffi::Buffer<dtype> x,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  if (svd::SVDType<dtype, int64_t>::fn != nullptr) {
    return internal::SvdKernel<dtype, int64_t>(
        thread_pool, x, x_out, singular_values, u, vt, info, mode);
  }
  return internal::SvdKernel<dtype, int32_t>(
      thread_pool, x, x_out, singular_values, u, vt, info, mode);
}

template <ffi::DataType dtype, typename IntType>
absl::StatusOr<int64_t>
SingularValueDecomposition<dtype, IntType>::GetWorkspaceSize(
    IntType x_rows, IntType x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype, IntType>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype, typename IntType>
absl::StatusOr<int64_t>
SingularValueDecompositionComplex<dtype, IntType>::GetWorkspaceSize(
    IntType x_rows, IntType x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype, IntType>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionQRKernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> singular_values, ffi::ResultBuffer<dtype> u,
    ffi::ResultBuffer<dtype> vt, ffi::ResultBuffer<LapackIntDtype> info,
    svd::ComputationMode mode) {
  if (svd::SVDQRType<dtype, int64_t>::fn != nullptr) {
    return internal::SvdQRKernel<dtype, int64_t>(x, x_out, singular_values, u,
                                                 vt, info, mode);
  }
  return internal::SvdQRKernel<dtype, int32_t>(x, x_out, singular_values, u, vt,
                                               info, mode);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionQRComplexKernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode) {
  if (svd::SVDQRType<dtype, int64_t>::fn != nullptr) {
    return internal::SvdQRKernel<dtype, int64_t>(x, x_out, singular_values, u,
                                                 vt, info, mode);
  }
  return internal::SvdQRKernel<dtype, int32_t>(x, x_out, singular_values, u, vt,
                                               info, mode);
}

template <ffi::DataType dtype, typename IntType>
absl::StatusOr<IntType>
SingularValueDecompositionQR<dtype, IntType>::GetWorkspaceSize(
    IntType x_rows, IntType x_cols, svd::ComputationMode mode) {
  return internal::SvdQRGetWorkspaceSize<dtype, IntType>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype, typename IntType>
absl::StatusOr<IntType>
SingularValueDecompositionQRComplex<dtype, IntType>::GetWorkspaceSize(
    IntType x_rows, IntType x_cols, svd::ComputationMode mode) {
  return internal::SvdQRGetWorkspaceSize<dtype, IntType>(x_rows, x_cols, mode);
}

int64_t svd::GetRealWorkspaceSize(int64_t x_rows, int64_t x_cols,
                                  svd::ComputationMode mode) {
  const auto min_dim = std::min(x_rows, x_cols);
  if (!ComputesUV(mode)) {
    return 7 * min_dim;
  }
  const auto max_dim = std::max(x_rows, x_cols);
  return std::max(5 * min_dim * min_dim + 5 * min_dim,
                  2 * max_dim * min_dim + 2 * min_dim * min_dim + min_dim);
}

int64_t svd::GetRealWorkspaceSizeQR(int64_t x_rows, int64_t x_cols) {
  return 5 * std::min(x_rows, x_cols);
}

int64_t svd::GetIntWorkspaceSize(int64_t x_rows, int64_t x_cols) {
  return 8 * std::min(x_rows, x_cols);
}

template struct SingularValueDecomposition<ffi::DataType::F32, int32_t>;
template struct SingularValueDecomposition<ffi::DataType::F32, int64_t>;
template struct SingularValueDecomposition<ffi::DataType::F64, int32_t>;
template struct SingularValueDecomposition<ffi::DataType::F64, int64_t>;
template struct SingularValueDecompositionComplex<ffi::DataType::C64, int32_t>;
template struct SingularValueDecompositionComplex<ffi::DataType::C64, int64_t>;
template struct SingularValueDecompositionComplex<ffi::DataType::C128, int32_t>;
template struct SingularValueDecompositionComplex<ffi::DataType::C128, int64_t>;

template struct SingularValueDecompositionQR<ffi::DataType::F32, int32_t>;
template struct SingularValueDecompositionQR<ffi::DataType::F32, int64_t>;
template struct SingularValueDecompositionQR<ffi::DataType::F64, int32_t>;
template struct SingularValueDecompositionQR<ffi::DataType::F64, int64_t>;
template struct SingularValueDecompositionQRComplex<ffi::DataType::C64,
                                                    int32_t>;
template struct SingularValueDecompositionQRComplex<ffi::DataType::C64,
                                                    int64_t>;
template struct SingularValueDecompositionQRComplex<ffi::DataType::C128,
                                                    int32_t>;
template struct SingularValueDecompositionQRComplex<ffi::DataType::C128,
                                                    int64_t>;

//== Eigenvalues and eigenvectors ==//

int64_t eig::GetWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return 2 * x_cols + 1;
    case ComputationMode::kComputeEigenvectors:
      return 1 + 6 * x_cols + 2 * x_cols * x_cols;
  }
}

int64_t eig::GetIntWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return 1;
    case ComputationMode::kComputeEigenvectors:
      return 3 + 5 * x_cols;
  }
}

template <ffi::DataType dtype, typename IntType>
ffi::Error EigenvalueDecompositionSymmetric<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* eigenvalues_data = eigenvalues->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  auto mode_v = static_cast<char>(mode);
  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<IntType>(x_cols));
  // Prepare LAPACK workspaces.
  FFI_ASSIGN_OR_RETURN(
      IntType work_size_v,
      MaybeCastNoOverflow<IntType>(eig::GetWorkspaceSize(x_cols, mode)));
  FFI_ASSIGN_OR_RETURN(
      IntType iwork_size_v,
      MaybeCastNoOverflow<IntType>(eig::GetIntWorkspaceSize(x_cols, mode)));
  auto work_data = AllocateScratchMemory<dtype>(work_size_v);
  auto iwork_data =
      AllocateScratchMemory<LapackIntDtypeFor<IntType>>(iwork_size_v);

  const int64_t x_out_step{x_cols * x_cols};
  const int64_t eigenvalues_step{x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    EigenvalueDecompositionSymmetric<dtype, IntType>::fn(
        &mode_v, &uplo_v, &x_cols_v, x_out_data, &x_leading_dim_v,
        eigenvalues_data, work_data.get(), &work_size_v, iwork_data.get(),
        &iwork_size_v, &info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data,
                                        sizeof(*x_out_data) * x_cols * x_cols);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigenvalues_data,
                                        sizeof(*eigenvalues_data) * x_cols);
    x_out_data += x_out_step;
    eigenvalues_data += eigenvalues_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionSymmetricKernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode) {
  if (EigenvalueDecompositionSymmetric<dtype, int64_t>::fn != nullptr) {
    return EigenvalueDecompositionSymmetric<dtype, int64_t>::Kernel(
        x, uplo, x_out, eigenvalues, info, mode);
  }
  return EigenvalueDecompositionSymmetric<dtype, int32_t>::Kernel(
      x, uplo, x_out, eigenvalues, info, mode);
}

namespace eig {

int64_t GetComplexWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return x_cols + 1;
    case ComputationMode::kComputeEigenvectors:
      return 2 * x_cols + x_cols * x_cols;
  }
}

int64_t GetRealWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return std::max(x_cols, int64_t{1});
    case ComputationMode::kComputeEigenvectors:
      return 1 + 5 * x_cols + 2 * x_cols * x_cols;
  }
}

}  // namespace eig

template <ffi::DataType dtype, typename IntType>
ffi::Error EigenvalueDecompositionHermitian<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  auto* x_out_data = x_out->typed_data();
  auto* eigenvalues_data = eigenvalues->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  auto mode_v = static_cast<char>(mode);
  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<IntType>(x_cols));
  // Prepare LAPACK workspaces.
  FFI_ASSIGN_OR_RETURN(
      IntType work_size_v,
      MaybeCastNoOverflow<IntType>(eig::GetComplexWorkspaceSize(x_cols, mode)));
  FFI_ASSIGN_OR_RETURN(
      IntType rwork_size_v,
      MaybeCastNoOverflow<IntType>(eig::GetRealWorkspaceSize(x_cols, mode)));
  FFI_ASSIGN_OR_RETURN(
      IntType iwork_size_v,
      MaybeCastNoOverflow<IntType>(eig::GetIntWorkspaceSize(x_cols, mode)));
  auto work_data = AllocateScratchMemory<dtype>(work_size_v);
  auto iwork_data =
      AllocateScratchMemory<LapackIntDtypeFor<IntType>>(iwork_size_v);
  auto rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(rwork_size_v);

  const int64_t x_out_step{x_cols * x_cols};
  const int64_t eigenvalues_step{x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    EigenvalueDecompositionHermitian<dtype, IntType>::fn(
        &mode_v, &uplo_v, &x_cols_v, x_out_data, &x_leading_dim_v,
        eigenvalues_data, work_data.get(), &work_size_v, rwork_data.get(),
        &rwork_size_v, iwork_data.get(), &iwork_size_v, &info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);
    x_out_data += x_out_step;
    eigenvalues_data += eigenvalues_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionHermitianKernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode) {
  if (EigenvalueDecompositionHermitian<dtype, int64_t>::fn != nullptr) {
    return EigenvalueDecompositionHermitian<dtype, int64_t>::Kernel(
        x, uplo, x_out, eigenvalues, info, mode);
  }
  return EigenvalueDecompositionHermitian<dtype, int32_t>::Kernel(
      x, uplo, x_out, eigenvalues, info, mode);
}

template struct EigenvalueDecompositionSymmetric<ffi::DataType::F32, int32_t>;
template struct EigenvalueDecompositionSymmetric<ffi::DataType::F32, int64_t>;
template struct EigenvalueDecompositionSymmetric<ffi::DataType::F64, int32_t>;
template struct EigenvalueDecompositionSymmetric<ffi::DataType::F64, int64_t>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C64, int32_t>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C64, int64_t>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C128, int32_t>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C128, int64_t>;

template <ffi::DataType dtype, typename IntType>
ffi::Error EigenvalueDecomposition<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_left,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));

  const auto* x_data = x.typed_data();
  auto* eigvecs_left_data = eigvecs_left->typed_data();
  auto* eigvecs_right_data = eigvecs_right->typed_data();
  auto* eigvals_real_data = eigvals_real->typed_data();
  auto* eigvals_imag_data = eigvals_imag->typed_data();
  auto* info_data = info->typed_data();

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  // Prepare LAPACK workspaces.
  int64_t work_size = EigenvalueDecomposition<dtype, IntType>::GetWorkspaceSize(
      x_cols_v, compute_left, compute_right);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  const int64_t x_size{x_cols * x_cols};
  auto x_copy = AllocateScratchMemory<dtype>(x_size);
  auto work_eigvecs_left = AllocateScratchMemory<dtype>(x_size);
  auto work_eigvecs_right = AllocateScratchMemory<dtype>(x_size);

  const auto is_finite = [](ffi::NativeType<dtype>* data, int64_t size) {
    return absl::c_all_of(
        absl::MakeSpan(data, size),
        [](ffi::NativeType<dtype> value) { return std::isfinite(value); });
  };

  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ffi::NativeType<dtype>);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ffi::NativeType<dtype>);
  for (int64_t i = 0; i < batch_count; ++i) {
    std::copy_n(x_data, x_size, x_copy.get());
    if (is_finite(x_copy.get(), x_size)) {
      IntType info_v;
      EigenvalueDecomposition<dtype, IntType>::fn(
          &compute_left_v, &compute_right_v, &x_cols_v, x_copy.get(), &x_cols_v,
          eigvals_real_data, eigvals_imag_data, work_eigvecs_left.get(),
          &x_cols_v, work_eigvecs_right.get(), &x_cols_v, work_data.get(),
          &work_size_v, &info_v);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
      *info_data = static_cast<int32_t>(info_v);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_copy.get(), x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_real_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_imag_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(work_eigvecs_left.get(),
                                          x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(work_eigvecs_right.get(),
                                          x_size_bytes);
      if (info_data[0] == 0) {
        UnpackEigenvectors(x_cols_v, eigvals_imag_data, work_eigvecs_left.get(),
                           eigvecs_left_data);
        UnpackEigenvectors(x_cols_v, eigvals_imag_data,
                           work_eigvecs_right.get(), eigvecs_right_data);
      }
    } else {
      info_data[0] = -4;
    }
    x_data += x_size;
    eigvals_real_data += x_cols;
    eigvals_imag_data += x_cols;
    eigvecs_left_data += x_size;
    eigvecs_right_data += x_size;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionKernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_left,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info) {
  if (EigenvalueDecomposition<dtype, int64_t>::fn != nullptr) {
    return EigenvalueDecomposition<dtype, int64_t>::Kernel(
        x, compute_left, compute_right, eigvals_real, eigvals_imag,
        eigvecs_left, eigvecs_right, info);
  }
  return EigenvalueDecomposition<dtype, int32_t>::Kernel(
      x, compute_left, compute_right, eigvals_real, eigvals_imag, eigvecs_left,
      eigvecs_right, info);
}

template <ffi::DataType dtype, typename IntType>
ffi::Error EigenvalueDecompositionComplex<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals,
    ffi::ResultBuffer<dtype> eigvecs_left,
    ffi::ResultBuffer<dtype> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  const auto* x_data = x.typed_data();
  auto* eigvecs_left_data = eigvecs_left->typed_data();
  auto* eigvecs_right_data = eigvecs_right->typed_data();
  auto* eigvals_data = eigvals->typed_data();
  auto* info_data = info->typed_data();

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  // Prepare LAPACK workspaces.
  int64_t work_size =
      EigenvalueDecompositionComplex<dtype, IntType>::GetWorkspaceSize(
          x_cols_v, compute_left, compute_right);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  const int64_t x_size{x_cols * x_cols};
  auto x_copy = AllocateScratchMemory<dtype>(x_size);
  auto rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(2 * x_cols);

  const auto is_finite = [](ffi::NativeType<dtype>* data, int64_t size) {
    return absl::c_all_of(absl::MakeSpan(data, size), [](const auto& z) {
      return std::isfinite(z.real()) && std::isfinite(z.imag());
    });
  };

  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ffi::NativeType<dtype>);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ffi::NativeType<dtype>);
  for (int64_t i = 0; i < batch_count; ++i) {
    std::copy_n(x_data, x_size, x_copy.get());
    if (is_finite(x_copy.get(), x_size)) {
      IntType info_v;
      EigenvalueDecompositionComplex<dtype, IntType>::fn(
          &compute_left_v, &compute_right_v, &x_cols_v, x_copy.get(), &x_cols_v,
          eigvals_data, eigvecs_left_data, &x_cols_v, eigvecs_right_data,
          &x_cols_v, work_data.get(), &work_size_v, rwork_data.get(), &info_v);
      *info_data = static_cast<int32_t>(info_v);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_copy.get(), x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvecs_left_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvecs_right_data, x_size_bytes);
    } else {
      info_data[0] = -4;
    }
    x_data += x_size;
    eigvals_data += x_cols;
    eigvecs_left_data += x_size;
    eigvecs_right_data += x_size;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionComplexKernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals,
    ffi::ResultBuffer<dtype> eigvecs_left,
    ffi::ResultBuffer<dtype> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info) {
  if (EigenvalueDecompositionComplex<dtype, int64_t>::fn != nullptr) {
    return EigenvalueDecompositionComplex<dtype, int64_t>::Kernel(
        x, compute_left, compute_right, eigvals, eigvecs_left, eigvecs_right,
        info);
  }
  return EigenvalueDecompositionComplex<dtype, int32_t>::Kernel(
      x, compute_left, compute_right, eigvals, eigvecs_left, eigvecs_right,
      info);
}

template <ffi::DataType dtype, typename IntType>
int64_t EigenvalueDecomposition<dtype, IntType>::GetWorkspaceSize(
    IntType x_cols, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right) {
  ValueType optimal_size = {};
  IntType workspace_query = -1;
  IntType info = 0;

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  fn(&compute_left_v, &compute_right_v, &x_cols, nullptr, &x_cols, nullptr,
     nullptr, nullptr, &x_cols, nullptr, &x_cols, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template <ffi::DataType dtype, typename IntType>
int64_t EigenvalueDecompositionComplex<dtype, IntType>::GetWorkspaceSize(
    IntType x_cols, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right) {
  ValueType optimal_size = {};
  IntType workspace_query = -1;
  IntType info = 0;
  // NULL rwork crashes, LAPACK unnecessarily writes x_cols into rwork
  RealType rwork[1];
  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  fn(&compute_left_v, &compute_right_v, &x_cols, nullptr, &x_cols, nullptr,
     nullptr, &x_cols, nullptr, &x_cols, &optimal_size, &workspace_query, rwork,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template struct EigenvalueDecomposition<ffi::DataType::F32, int32_t>;
template struct EigenvalueDecomposition<ffi::DataType::F32, int64_t>;
template struct EigenvalueDecomposition<ffi::DataType::F64, int32_t>;
template struct EigenvalueDecomposition<ffi::DataType::F64, int64_t>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C64, int32_t>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C64, int64_t>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C128, int32_t>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C128, int64_t>;

//== Schur Decomposition ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error SchurDecomposition<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> schur_vectors,
    ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    // TODO(paruzelp): Sort is not implemented because select function is not
    // supplied. For that reason, this parameter will always be zero!
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  if (sort != schur::Sort::kNoSortEigenvalues) {
    return ffi::Error(
        ffi::ErrorCode::kUnimplemented,
        "Ordering eigenvalues on the diagonal is not implemented");
  }

  CopyIfDiffBuffer(x, x_out);

  // TODO(paruzelp): `select` should be passed as an execution context
  bool (*select)(ffi::NativeType<dtype>, ffi::NativeType<dtype>) = nullptr;
  ffi::NativeType<dtype>* x_out_data = x_out->typed_data();
  ffi::NativeType<dtype>* eigvals_real_data = eigvals_real->typed_data();
  ffi::NativeType<dtype>* eigvals_imag_data = eigvals_imag->typed_data();
  ffi::NativeType<dtype>* schur_vectors_data = schur_vectors->typed_data();
  auto* selected_data = selected_eigvals->typed_data();
  auto* info_data = info->typed_data();

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));

  // Prepare LAPACK workspaces.
  std::unique_ptr<bool[]> bwork =
      sort != schur::Sort::kNoSortEigenvalues
          ? AllocateScratchMemory<ffi::DataType::PRED>(x_cols)
          : nullptr;
  auto work_size =
      SchurDecomposition<dtype, IntType>::GetWorkspaceSize(x_cols, mode, sort);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ffi::NativeType<dtype>);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ffi::NativeType<dtype>);
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType selected_v;
    IntType info_v;
    SchurDecomposition<dtype, IntType>::fn(
        &mode_v, &sort_v, select, &x_cols_v, x_out_data, &x_cols_v, &selected_v,
        eigvals_real_data, eigvals_imag_data, schur_vectors_data, &x_cols_v,
        work_data.get(), &work_size_v, bwork.get(), &info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&selected_v, sizeof(selected_v));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *selected_data = static_cast<int32_t>(selected_v);
    *info_data = static_cast<int32_t>(info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data, x_size_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_real_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_imag_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(schur_vectors_data, x_size_bytes);

    x_out_data += x_size;
    eigvals_real_data += x_cols;
    eigvals_imag_data += x_cols;
    schur_vectors_data += x_size;
    ++selected_data;
    ++info_data;
  }

  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error SchurDecompositionKernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> schur_vectors,
    ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    // TODO(paruzelp): Sort is not implemented because select function is not
    // supplied. For that reason, this parameter will always be zero!
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info) {
  if (SchurDecomposition<dtype, int64_t>::fn != nullptr) {
    return SchurDecomposition<dtype, int64_t>::Kernel(
        x, mode, sort, x_out, schur_vectors, eigvals_real, eigvals_imag,
        selected_eigvals, info);
  }
  return SchurDecomposition<dtype, int32_t>::Kernel(
      x, mode, sort, x_out, schur_vectors, eigvals_real, eigvals_imag,
      selected_eigvals, info);
}

template <ffi::DataType dtype, typename IntType>
ffi::Error SchurDecompositionComplex<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> schur_vectors,
    ffi::ResultBuffer<dtype> eigvals,
    // TODO(paruzelp): Sort is not implemented because select function is not
    // supplied. For that reason, this parameter will always be zero!
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));
  if (sort != schur::Sort::kNoSortEigenvalues) {
    return ffi::Error(
        ffi::ErrorCode::kUnimplemented,
        "Ordering eigenvalues on the diagonal is not implemented");
  }

  CopyIfDiffBuffer(x, x_out);

  // TODO(paruzelp): `select` should be passed as an execution context
  bool (*select)(ffi::NativeType<dtype>) = nullptr;
  ffi::NativeType<dtype>* x_out_data = x_out->typed_data();
  ffi::NativeType<dtype>* eigvals_data = eigvals->typed_data();
  ffi::NativeType<dtype>* schur_vectors_data = schur_vectors->typed_data();
  auto* selected_data = selected_eigvals->typed_data();
  auto* info_data = info->typed_data();

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));

  // Prepare LAPACK workspaces.
  std::unique_ptr<bool[]> bwork =
      sort != schur::Sort::kNoSortEigenvalues
          ? AllocateScratchMemory<ffi::DataType::PRED>(x_cols)
          : nullptr;
  auto work_size = SchurDecompositionComplex<dtype, IntType>::GetWorkspaceSize(
      x_cols, mode, sort);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);
  auto rwork_data = AllocateScratchMemory<ffi::ToReal(dtype)>(x_cols);

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ffi::NativeType<dtype>);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ffi::NativeType<dtype>);
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType selected_v;
    IntType info_v;
    SchurDecompositionComplex<dtype, IntType>::fn(
        &mode_v, &sort_v, select, &x_cols_v, x_out_data, &x_cols_v, &selected_v,
        eigvals_data, schur_vectors_data, &x_cols_v, work_data.get(),
        &work_size_v, rwork_data.get(), bwork.get(), &info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&selected_v, sizeof(selected_v));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *selected_data = static_cast<int32_t>(selected_v);
    *info_data = static_cast<int32_t>(info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(schur_vectors_data, x_size_bytes);

    x_out_data += x_size;
    eigvals_data += x_cols;
    schur_vectors_data += x_size;
    ++selected_data;
    ++info_data;
  }

  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error SchurDecompositionComplexKernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> schur_vectors,
    ffi::ResultBuffer<dtype> eigvals,
    // TODO(paruzelp): Sort is not implemented because select function is not
    // supplied. For that reason, this parameter will always be zero!
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info) {
  if (SchurDecompositionComplex<dtype, int64_t>::fn != nullptr) {
    return SchurDecompositionComplex<dtype, int64_t>::Kernel(
        x, mode, sort, x_out, schur_vectors, eigvals, selected_eigvals, info);
  }
  return SchurDecompositionComplex<dtype, int32_t>::Kernel(
      x, mode, sort, x_out, schur_vectors, eigvals, selected_eigvals, info);
}

template <ffi::DataType dtype, typename IntType>
int64_t SchurDecomposition<dtype, IntType>::GetWorkspaceSize(
    IntType x_cols, schur::ComputationMode mode, schur::Sort sort) {
  ValueType optimal_size = {};
  IntType workspace_query = -1;
  IntType info = 0;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  fn(&mode_v, &sort_v, nullptr, &x_cols, nullptr, &x_cols, nullptr, nullptr,
     nullptr, nullptr, &x_cols, &optimal_size, &workspace_query, nullptr,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template <ffi::DataType dtype, typename IntType>
int64_t SchurDecompositionComplex<dtype, IntType>::GetWorkspaceSize(
    IntType x_cols, schur::ComputationMode mode, schur::Sort sort) {
  ValueType optimal_size = {};
  IntType workspace_query = -1;
  IntType info = 0;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  fn(&mode_v, &sort_v, nullptr, &x_cols, nullptr, &x_cols, nullptr, nullptr,
     nullptr, &x_cols, &optimal_size, &workspace_query, nullptr, nullptr,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template struct SchurDecomposition<ffi::DataType::F32, int32_t>;
template struct SchurDecomposition<ffi::DataType::F32, int64_t>;
template struct SchurDecomposition<ffi::DataType::F64, int32_t>;
template struct SchurDecomposition<ffi::DataType::F64, int64_t>;
template struct SchurDecompositionComplex<ffi::DataType::C64, int32_t>;
template struct SchurDecompositionComplex<ffi::DataType::C64, int64_t>;
template struct SchurDecompositionComplex<ffi::DataType::C128, int32_t>;
template struct SchurDecompositionComplex<ffi::DataType::C128, int64_t>;

//== Hessenberg Decomposition ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error HessenbergDecomposition<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, int32_t low, int32_t high,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> tau,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));

  CopyIfDiffBuffer(x, x_out);

  ffi::NativeType<dtype>* x_out_data = x_out->typed_data();
  ffi::NativeType<dtype>* tau_data = tau->typed_data();
  auto* info_data = info->typed_data();
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<IntType>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<IntType>(x_rows));
  IntType low_v = static_cast<IntType>(low);
  IntType high_v = static_cast<IntType>(high);
  // Prepare LAPACK workspaces.
  int64_t work_size = HessenbergDecomposition<dtype, IntType>::GetWorkspaceSize(
      x_rows, x_cols, low_v, high_v);
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  int64_t x_size{x_rows * x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    HessenbergDecomposition<dtype, IntType>::fn(
        &x_cols_v, &low_v, &high_v, x_out_data, &x_leading_dim_v, tau_data,
        work_data.get(), &work_size_v, &info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);
    x_out_data += x_size;
    tau_data += x_cols - 1;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error HessenbergDecompositionKernel(
    ffi::Buffer<dtype> x, int32_t low, int32_t high,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> tau,
    ffi::ResultBuffer<LapackIntDtype> info) {
  if (HessenbergDecomposition<dtype, int64_t>::fn != nullptr) {
    return HessenbergDecomposition<dtype, int64_t>::Kernel(x, low, high, x_out,
                                                           tau, info);
  }
  return HessenbergDecomposition<dtype, int32_t>::Kernel(x, low, high, x_out,
                                                         tau, info);
}

template <ffi::DataType dtype, typename IntType>
int64_t HessenbergDecomposition<dtype, IntType>::GetWorkspaceSize(
    IntType x_rows, IntType x_cols, IntType low, IntType high) {
  ValueType optimal_size = {};
  IntType workspace_query = -1;
  IntType info = 0;
  fn(&x_cols, &low, &high, nullptr, &x_rows, nullptr, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct HessenbergDecomposition<ffi::DataType::F32, int32_t>;
template struct HessenbergDecomposition<ffi::DataType::F32, int64_t>;
template struct HessenbergDecomposition<ffi::DataType::F64, int32_t>;
template struct HessenbergDecomposition<ffi::DataType::F64, int64_t>;
template struct HessenbergDecomposition<ffi::DataType::C64, int32_t>;
template struct HessenbergDecomposition<ffi::DataType::C64, int64_t>;
template struct HessenbergDecomposition<ffi::DataType::C128, int32_t>;
template struct HessenbergDecomposition<ffi::DataType::C128, int64_t>;

//== Tridiagonal Reduction ==//

template <ffi::DataType dtype, typename IntType>
ffi::Error TridiagonalReduction<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> diagonal,
    ffi::ResultBuffer<ffi::ToReal(dtype)> off_diagonal,
    ffi::ResultBuffer<dtype> tau, ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, x_rows, x_cols]),
                       SplitBatch2D(x.dimensions()));

  CopyIfDiffBuffer(x, x_out);

  ffi::NativeType<dtype>* x_out_data = x_out->typed_data();
  ffi::NativeType<ffi::ToReal(dtype)>* diagonal_data = diagonal->typed_data();
  ffi::NativeType<ffi::ToReal(dtype)>* off_diagonal_data =
      off_diagonal->typed_data();
  ffi::NativeType<dtype>* tau_data = tau->typed_data();
  auto* info_data = info->typed_data();

  // Prepare LAPACK workspaces.
  const auto work_size =
      TridiagonalReduction<dtype, IntType>::GetWorkspaceSize(x_rows, x_cols);
  auto work_data = AllocateScratchMemory<dtype>(work_size);

  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_leading_dim_v,
                       MaybeCastNoOverflow<IntType>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto work_size_v,
                       MaybeCastNoOverflow<IntType>(work_size));
  FFI_ASSIGN_OR_RETURN(auto x_order_v, MaybeCastNoOverflow<IntType>(x_cols));

  int64_t x_size = x_rows * x_cols;
  int64_t tau_step = {tau->dimensions().back()};
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    TridiagonalReduction<dtype, IntType>::fn(
        &uplo_v, &x_order_v, x_out_data, &x_leading_dim_v, diagonal_data,
        off_diagonal_data, tau_data, work_data.get(), &work_size_v, &info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);
    x_out_data += x_size;
    diagonal_data += x_cols;
    off_diagonal_data += x_cols - 1;
    tau_data += tau_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error TridiagonalReductionKernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> diagonal,
    ffi::ResultBuffer<ffi::ToReal(dtype)> off_diagonal,
    ffi::ResultBuffer<dtype> tau, ffi::ResultBuffer<LapackIntDtype> info) {
  if (TridiagonalReduction<dtype, int64_t>::fn != nullptr) {
    return TridiagonalReduction<dtype, int64_t>::Kernel(
        x, uplo, x_out, diagonal, off_diagonal, tau, info);
  }
  return TridiagonalReduction<dtype, int32_t>::Kernel(x, uplo, x_out, diagonal,
                                                      off_diagonal, tau, info);
}

template <ffi::DataType dtype, typename IntType>
int64_t TridiagonalReduction<dtype, IntType>::GetWorkspaceSize(IntType x_rows,
                                                               IntType x_cols) {
  ValueType optimal_size = {};
  IntType workspace_query = -1;
  IntType info = 0;
  char uplo_v = 'L';
  fn(&uplo_v, &x_cols, nullptr, &x_rows, nullptr, nullptr, nullptr,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct TridiagonalReduction<ffi::DataType::F32, int32_t>;
template struct TridiagonalReduction<ffi::DataType::F32, int64_t>;
template struct TridiagonalReduction<ffi::DataType::F64, int32_t>;
template struct TridiagonalReduction<ffi::DataType::F64, int64_t>;
template struct TridiagonalReduction<ffi::DataType::C64, int32_t>;
template struct TridiagonalReduction<ffi::DataType::C64, int64_t>;
template struct TridiagonalReduction<ffi::DataType::C128, int32_t>;
template struct TridiagonalReduction<ffi::DataType::C128, int64_t>;

//== General Tridiagonal System Solver ==//

// lapack gtsv

template <ffi::DataType dtype, typename IntType>
ffi::Error TridiagonalSolver<dtype, IntType>::Kernel(
    ffi::Buffer<dtype> dl, ffi::Buffer<dtype> d, ffi::Buffer<dtype> du,
    ffi::Buffer<dtype> b, ffi::ResultBuffer<dtype> dl_out,
    ffi::ResultBuffer<dtype> d_out, ffi::ResultBuffer<dtype> du_out,
    ffi::ResultBuffer<dtype> b_out, ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_ASSIGN_OR_RETURN((auto [batch_count, b_rows, b_cols]),
                       SplitBatch2D(b.dimensions()));

  CopyIfDiffBuffer(dl, dl_out);
  CopyIfDiffBuffer(d, d_out);
  CopyIfDiffBuffer(du, du_out);
  CopyIfDiffBuffer(b, b_out);

  auto* dl_out_data = dl_out->typed_data();
  auto* d_out_data = d_out->typed_data();
  auto* du_out_data = du_out->typed_data();
  auto* b_out_data = b_out->typed_data();
  auto* info_data = info->typed_data();

  FFI_ASSIGN_OR_RETURN(auto b_rows_v, MaybeCastNoOverflow<IntType>(b_rows));
  FFI_ASSIGN_OR_RETURN(auto b_cols_v, MaybeCastNoOverflow<IntType>(b_cols));

  const int64_t b_out_step{b_rows * b_cols};
  const int64_t d_step{b_rows};
  for (int64_t i = 0; i < batch_count; ++i) {
    IntType info_v;
    TridiagonalSolver<dtype, IntType>::fn(&b_rows_v, &b_cols_v, dl_out_data + 1,
                                          d_out_data, du_out_data, b_out_data,
                                          &b_rows_v, &info_v);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&info_v, sizeof(info_v));
    *info_data = static_cast<int32_t>(info_v);
    b_out_data += b_out_step;
    dl_out_data += d_step;
    d_out_data += d_step;
    du_out_data += d_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error TridiagonalSolverKernel(ffi::Buffer<dtype> dl, ffi::Buffer<dtype> d,
                                   ffi::Buffer<dtype> du, ffi::Buffer<dtype> b,
                                   ffi::ResultBuffer<dtype> dl_out,
                                   ffi::ResultBuffer<dtype> d_out,
                                   ffi::ResultBuffer<dtype> du_out,
                                   ffi::ResultBuffer<dtype> b_out,
                                   ffi::ResultBuffer<LapackIntDtype> info) {
  if (TridiagonalSolver<dtype, int64_t>::fn != nullptr) {
    return TridiagonalSolver<dtype, int64_t>::Kernel(
        dl, d, du, b, dl_out, d_out, du_out, b_out, info);
  }
  return TridiagonalSolver<dtype, int32_t>::Kernel(dl, d, du, b, dl_out, d_out,
                                                   du_out, b_out, info);
}

template struct TridiagonalSolver<ffi::DataType::F32, int32_t>;
template struct TridiagonalSolver<ffi::DataType::F32, int64_t>;
template struct TridiagonalSolver<ffi::DataType::F64, int32_t>;
template struct TridiagonalSolver<ffi::DataType::F64, int64_t>;
template struct TridiagonalSolver<ffi::DataType::C64, int32_t>;
template struct TridiagonalSolver<ffi::DataType::C64, int64_t>;
template struct TridiagonalSolver<ffi::DataType::C128, int32_t>;
template struct TridiagonalSolver<ffi::DataType::C128, int64_t>;

// FFI Definition Macros (by DataType)

#define JAX_CPU_DEFINE_TRSM(name, data_type)             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, TriMatrixEquationSolverKernel<data_type>,    \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Arg<::xla::ffi::Buffer<data_type>>(/*y*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*y_out*/) \
          .Attr<MatrixParams::Side>("side")              \
          .Attr<MatrixParams::UpLo>("uplo")              \
          .Attr<MatrixParams::Transpose>("trans_x")      \
          .Attr<MatrixParams::Diag>("diag"))

#define JAX_CPU_DEFINE_GETRF(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, LuDecompositionKernel<data_type>,                \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*ipiv*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEQRF(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, QrFactorizationKernel<data_type>,            \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/))

#define JAX_CPU_DEFINE_GEQP3(name, data_type)                    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                 \
      name, PivotingQrFactorizationKernel<data_type>,            \
      ::xla::ffi::Ffi::Bind()                                    \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)             \
          .Arg<::xla::ffi::Buffer<LapackIntDtype>>(/*jpvt*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)         \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*jpvt_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/))

#define JAX_CPU_DEFINE_ORGQR(name, data_type)          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                       \
      name, OrthogonalQrKernel<data_type>,             \
      ::xla::ffi::Ffi::Bind()                          \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)   \
          .Arg<::xla::ffi::Buffer<data_type>>(/*tau*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/))

#define JAX_CPU_DEFINE_ORMQR(name, data_type)          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                       \
      name, OrthogonalQrMultiplyKernel<data_type>,     \
      ::xla::ffi::Ffi::Bind()                          \
          .Arg<::xla::ffi::Buffer<data_type>>(/*a*/)   \
          .Arg<::xla::ffi::Buffer<data_type>>(/*tau*/) \
          .Arg<::xla::ffi::Buffer<data_type>>(/*c*/)   \
          .Attr<bool>("left")                          \
          .Attr<bool>("transpose")                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*c_out*/))

#define JAX_CPU_DEFINE_POTRF(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, CholeskyFactorizationKernel<data_type>,      \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Attr<MatrixParams::UpLo>("uplo")              \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GESDD(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, SingularValueDecompositionKernel<data_type>,     \
      ::xla::ffi::Ffi::Bind()                                \
          .Ctx<::xla::ffi::ThreadPool>()                     \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*s*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)        \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/) \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESDD_COMPLEX(name, data_type)                    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SingularValueDecompositionComplexKernel<data_type>,          \
      ::xla::ffi::Ffi::Bind()                                            \
          .Ctx<::xla::ffi::ThreadPool>()                                 \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>(/*s*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)                    \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)             \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESVD(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, SingularValueDecompositionQRKernel<data_type>,   \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*s*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)        \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/) \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESVD_COMPLEX(name, data_type)                    \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SingularValueDecompositionQRComplexKernel<data_type>,        \
      ::xla::ffi::Ffi::Bind()                                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>(/*s*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)                    \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)             \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_SYEVD(name, data_type)                  \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                               \
      name, EigenvalueDecompositionSymmetricKernel<data_type>, \
      ::xla::ffi::Ffi::Bind()                                  \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)           \
          .Attr<MatrixParams::UpLo>("uplo")                    \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)       \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigenvalues*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)   \
          .Attr<eig::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_HEEVD(name, data_type)                      \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                   \
      name, EigenvalueDecompositionHermitianKernel<data_type>,     \
      ::xla::ffi::Ffi::Bind()                                      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)               \
          .Attr<MatrixParams::UpLo>("uplo")                        \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)           \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>( \
              /*eigenvalues*/)                                     \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)       \
          .Attr<eig::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GEEV(name, data_type)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                      \
      name, EigenvalueDecompositionKernel<data_type>,                 \
      ::xla::ffi::Ffi::Bind()                                         \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                  \
          .Attr<eig::ComputationMode>("compute_left")                 \
          .Attr<eig::ComputationMode>("compute_right")                \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_real*/)       \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_imag*/)       \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToComplex(data_type)>>( \
              /*eigvecs_left*/)                                       \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToComplex(data_type)>>( \
              /*eigvecs_right*/)                                      \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEEV_COMPLEX(name, data_type)             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                 \
      name, EigenvalueDecompositionComplexKernel<data_type>,     \
      ::xla::ffi::Ffi::Bind()                                    \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)             \
          .Attr<eig::ComputationMode>("compute_left")            \
          .Attr<eig::ComputationMode>("compute_right")           \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals*/)       \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvecs_left*/)  \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvecs_right*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEES(name, data_type)                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SchurDecompositionKernel<data_type>,                         \
      ::xla::ffi::Ffi::Bind()                                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Attr<schur::ComputationMode>("mode")                          \
          .Attr<schur::Sort>("sort")                                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<data_type>>(/*schur_vectors*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_real*/)          \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals_imag*/)          \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*selected_eigvals*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEES_COMPLEX(name, data_type)                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                         \
      name, SchurDecompositionComplexKernel<data_type>,                  \
      ::xla::ffi::Ffi::Bind()                                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                     \
          .Attr<schur::ComputationMode>("mode")                          \
          .Attr<schur::Sort>("sort")                                     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                 \
          .Ret<::xla::ffi::Buffer<data_type>>(/*schur_vectors*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigvals*/)               \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*selected_eigvals*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_SYTRD_HETRD(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                   \
      name, TridiagonalReductionKernel<data_type>,                 \
      ::xla::ffi::Ffi::Bind()                                      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)               \
          .Attr<MatrixParams::UpLo>("uplo")                        \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)           \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>( \
              /*diagonal*/)                                        \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>( \
              /*off_diagonal*/)                                    \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/)             \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEHRD(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, HessenbergDecompositionKernel<data_type>,    \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Attr<int32_t>("low")                          \
          .Attr<int32_t>("high")                         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/)   \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GTSV(name, data_type)              \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                          \
      name, TridiagonalSolverKernel<data_type>,           \
      ::xla::ffi::Ffi::Bind()                             \
          .Arg<::xla::ffi::Buffer<data_type>>(/*dl*/)     \
          .Arg<::xla::ffi::Buffer<data_type>>(/*d*/)      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*du*/)     \
          .Arg<::xla::ffi::Buffer<data_type>>(/*b*/)      \
          .Ret<::xla::ffi::Buffer<data_type>>(/*dl_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*d_out*/)  \
          .Ret<::xla::ffi::Buffer<data_type>>(/*du_out*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*b_out*/)  \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

// FFI Handlers

JAX_CPU_DEFINE_TRSM(lapack_strsm_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_TRSM(lapack_dtrsm_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_TRSM(lapack_ctrsm_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_TRSM(lapack_ztrsm_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GETRF(lapack_sgetrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GETRF(lapack_dgetrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GETRF(lapack_cgetrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GETRF(lapack_zgetrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEQRF(lapack_sgeqrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEQRF(lapack_dgeqrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEQRF(lapack_cgeqrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEQRF(lapack_zgeqrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEQP3(lapack_sgeqp3_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEQP3(lapack_dgeqp3_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEQP3(lapack_cgeqp3_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEQP3(lapack_zgeqp3_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_ORGQR(lapack_sorgqr_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_ORGQR(lapack_dorgqr_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_ORGQR(lapack_cungqr_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_ORGQR(lapack_zungqr_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_ORMQR(lapack_sormqr_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_ORMQR(lapack_dormqr_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_ORMQR(lapack_cunmqr_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_ORMQR(lapack_zunmqr_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_POTRF(lapack_spotrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_POTRF(lapack_dpotrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_POTRF(lapack_cpotrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_POTRF(lapack_zpotrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GESDD(lapack_sgesdd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GESDD(lapack_dgesdd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_cgesdd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_zgesdd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GESVD(lapack_sgesvd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GESVD(lapack_dgesvd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GESVD_COMPLEX(lapack_cgesvd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GESVD_COMPLEX(lapack_zgesvd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_SYEVD(lapack_ssyevd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_SYEVD(lapack_dsyevd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_HEEVD(lapack_cheevd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_HEEVD(lapack_zheevd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEEV(lapack_sgeev_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEEV(lapack_dgeev_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEEV_COMPLEX(lapack_cgeev_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEEV_COMPLEX(lapack_zgeev_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_SYTRD_HETRD(lapack_ssytrd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_dsytrd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_chetrd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_SYTRD_HETRD(lapack_zhetrd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEES(lapack_sgees_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEES(lapack_dgees_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEES_COMPLEX(lapack_cgees_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEES_COMPLEX(lapack_zgees_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEHRD(lapack_sgehrd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEHRD(lapack_dgehrd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEHRD(lapack_cgehrd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEHRD(lapack_zgehrd_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GTSV(lapack_sgtsv_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GTSV(lapack_dgtsv_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GTSV(lapack_cgtsv_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GTSV(lapack_zgtsv_ffi, ::xla::ffi::DataType::C128);

#undef JAX_CPU_DEFINE_TRSM
#undef JAX_CPU_DEFINE_GETRF
#undef JAX_CPU_DEFINE_GEQRF
#undef JAX_CPU_DEFINE_GEQP3
#undef JAX_CPU_DEFINE_ORGQR
#undef JAX_CPU_DEFINE_ORMQR
#undef JAX_CPU_DEFINE_POTRF
#undef JAX_CPU_DEFINE_GESDD
#undef JAX_CPU_DEFINE_GESDD_COMPLEX
#undef JAX_CPU_DEFINE_GESVD
#undef JAX_CPU_DEFINE_GESVD_COMPLEX
#undef JAX_CPU_DEFINE_SYEVD
#undef JAX_CPU_DEFINE_HEEVD
#undef JAX_CPU_DEFINE_GEEV
#undef JAX_CPU_DEFINE_GEEV_COMPLEX
#undef JAX_CPU_DEFINE_SYTRD_HETRD
#undef JAX_CPU_DEFINE_GEES
#undef JAX_CPU_DEFINE_GEES_COMPLEX
#undef JAX_CPU_DEFINE_GEHRD
#undef JAX_CPU_DEFINE_GTSV

}  // namespace jax
