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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/strings/str_format.h"

static_assert(sizeof(jax::lapack_int) == sizeof(int32_t),
              "Expected LAPACK integers to be 32-bit");

namespace ffi = xla::ffi;

namespace {

template <typename T>
inline T CastNoOverflow(int64_t value) {
  if constexpr (sizeof(T) == sizeof(int64_t)) {
    return value;
  } else {
    if (value > std::numeric_limits<T>::max()) [[unlikely]] {
      throw std::overflow_error{
          absl::StrFormat("%s: Value (=%d) exceeds the maximum representable "
                          "value of the desired type",
                          __FILE__, value)};
    }
    return static_cast<T>(value);
  }
}

template <typename T>
std::tuple<int64_t, int64_t, int64_t> SplitBatch2D(ffi::Span<T> dims) {
  if (dims.size() < 2) {
    throw std::invalid_argument("Matrix must have at least 2 dimensions");
  }
  auto matrix_dims = dims.last(2);
  return std::make_tuple(absl::c_accumulate(dims.first(dims.size() - 2), 1,
                                            std::multiplies<int64_t>()),
                         matrix_dims.front(), matrix_dims.back());
}

template <ffi::DataType dtype>
void CopyIfDiffBuffer(ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x.data != x_out->data) {
    const auto x_size = batch_count * x_rows * x_cols;
    std::copy_n(x.data, x_size, x_out->data);
  }
}

}  // namespace

#define REGISTER_CHAR_ENUM_ATTR_DECODING(type)                                \
  std::optional<type> xla::ffi::AttrDecoding<type>::Decode(                   \
      XLA_FFI_AttrType attr_type, void* attr, DiagnosticEngine& diagnostic) { \
    if (attr_type != XLA_FFI_AttrType_SCALAR) [[unlikely]] {                  \
      return diagnostic.Emit("Wrong attribute type: expected ")               \
             << XLA_FFI_AttrType_SCALAR << " but got" << attr_type;           \
    }                                                                         \
    auto* scalar = reinterpret_cast<XLA_FFI_Scalar*>(attr);                   \
    if (scalar->dtype != XLA_FFI_DataType_U8) [[unlikely]] {                  \
      return diagnostic.Emit("Wrong scalar data type: expected ")             \
             << XLA_FFI_DataType_U8 << " but got " << scalar->dtype;          \
    }                                                                         \
    auto underlying =                                                         \
        *reinterpret_cast<std::underlying_type_t<type>*>(scalar->value);      \
    return static_cast<type>(underlying);                                     \
  }

REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Side);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Transpose);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Diag);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::UpLo);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::svd::ComputationMode);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::eig::ComputationMode);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::schur::ComputationMode);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::schur::Sort);

#undef REGISTER_CHAR_ENUM_ATTR_DECODING

namespace jax {

// Triangular System Solver

template <ffi::DataType dtype>
ffi::Error TriMatrixEquationSolver<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::Buffer<dtype> y, ffi::BufferR0<dtype> alpha,
    ffi::ResultBuffer<dtype> y_out, MatrixParams::Side side,
    MatrixParams::UpLo uplo, MatrixParams::Transpose trans_x,
    MatrixParams::Diag diag) {
  CopyIfDiffBuffer(y, y_out);

  auto [batch_count, y_rows, y_cols] = SplitBatch2D(y.dimensions);
  auto* y_out_data = y_out->data;
  lapack_int x_leading_dim_v =
      side == MatrixParams::Side::kLeft ? y_rows : y_cols;
  lapack_int y_leading_dim_v = y_rows;

  auto side_v = static_cast<char>(side);
  auto uplo_v = static_cast<char>(uplo);
  auto trans_x_v = static_cast<char>(trans_x);
  auto diag_v = static_cast<char>(diag);
  auto y_rows_v = CastNoOverflow<lapack_int>(y_rows);
  auto y_cols_v = CastNoOverflow<lapack_int>(y_cols);

  auto* x_data = x.data;
  const int64_t y_out_step{y_rows * y_cols};
  const int64_t x_step{x_leading_dim_v * x_leading_dim_v};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&side_v, &uplo_v, &trans_x_v, &diag_v, &y_rows_v, &y_cols_v, alpha.data,
       x_data, &x_leading_dim_v, y_out_data, &y_leading_dim_v);

    y_out_data += y_out_step;
    x_data += x_step;
  }
  return ffi::Error::Success();
}

template struct TriMatrixEquationSolver<ffi::DataType::F32>;
template struct TriMatrixEquationSolver<ffi::DataType::F64>;
template struct TriMatrixEquationSolver<ffi::DataType::C64>;
template struct TriMatrixEquationSolver<ffi::DataType::C128>;

// LU Decomposition

template <ffi::DataType dtype>
ffi::Error LuDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<LapackIntDtype> ipiv,
    ffi::ResultBuffer<LapackIntDtype> info) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  auto* x_out_data = x_out->data;
  auto* ipiv_data = ipiv->data;
  auto* info_data = info->data;

  CopyIfDiffBuffer(x, x_out);

  auto x_rows_v = CastNoOverflow<lapack_int>(x_rows);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t ipiv_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_order_v, x_out_data, &x_leading_dim_v, ipiv_data,
       info_data);
    x_out_data += x_out_step;
    ipiv_data += ipiv_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct LuDecomposition<ffi::DataType::F32>;
template struct LuDecomposition<ffi::DataType::F64>;
template struct LuDecomposition<ffi::DataType::C64>;
template struct LuDecomposition<ffi::DataType::C128>;

// QR Factorization

template <ffi::DataType dtype>
ffi::Error QrFactorization<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> tau, ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<dtype> work) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  auto* x_out_data = x_out->data;
  auto* tau_data = tau->data;
  auto* info_data = info->data;
  auto* work_data = work->data;

  CopyIfDiffBuffer(x, x_out);

  auto workspace_dim_v = CastNoOverflow<lapack_int>(work->dimensions.back());
  auto x_rows_v = CastNoOverflow<lapack_int>(x_rows);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_order_v, x_out_data, &x_leading_dim_v, tau_data, work_data,
       &workspace_dim_v, info_data);
    x_out_data += x_out_step;
    tau_data += tau_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t QrFactorization<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                 lapack_int x_cols) {
  ValueType optimal_size{};
  lapack_int x_leading_dim_v = x_rows;
  lapack_int info = 0;
  lapack_int workspace_query = -1;
  fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct QrFactorization<ffi::DataType::F32>;
template struct QrFactorization<ffi::DataType::F64>;
template struct QrFactorization<ffi::DataType::C64>;
template struct QrFactorization<ffi::DataType::C128>;

// Orthogonal QR
// Computes orthogonal matrix Q from QR Decomposition

template <ffi::DataType dtype>
ffi::Error OrthogonalQr<dtype>::Kernel(ffi::Buffer<dtype> x,
                                       ffi::Buffer<dtype> tau,
                                       ffi::ResultBuffer<dtype> x_out,
                                       ffi::ResultBuffer<LapackIntDtype> info,
                                       ffi::ResultBuffer<dtype> work) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  auto* tau_data = tau.data;
  auto* x_out_data = x_out->data;
  auto* info_data = info->data;
  auto* work_data = work->data;

  CopyIfDiffBuffer(x, x_out);

  auto tau_size_v = CastNoOverflow<lapack_int>(tau.dimensions.back());
  auto x_rows_v = CastNoOverflow<lapack_int>(x_rows);
  auto x_cols_v = CastNoOverflow<lapack_int>(x_cols);
  auto workspace_dim_v = CastNoOverflow<lapack_int>(work->dimensions.back());
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{tau_size_v};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_cols_v, &tau_size_v, x_out_data, &x_leading_dim_v,
       tau_data, work_data, &workspace_dim_v, info_data);
    x_out_data += x_out_step;
    tau_data += tau_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t OrthogonalQr<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                              lapack_int x_cols,
                                              lapack_int tau_size) {
  ValueType optimal_size = {};
  lapack_int x_leading_dim_v = x_rows;
  lapack_int info = 0;
  lapack_int workspace_query = -1;
  fn(&x_rows, &x_cols, &tau_size, nullptr, &x_leading_dim_v, nullptr,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct OrthogonalQr<ffi::DataType::F32>;
template struct OrthogonalQr<ffi::DataType::F64>;
template struct OrthogonalQr<ffi::DataType::C64>;
template struct OrthogonalQr<ffi::DataType::C128>;

// Cholesky Factorization

template <ffi::DataType dtype>
ffi::Error CholeskyFactorization<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<LapackIntDtype> info) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  auto* x_out_data = x_out->data;
  auto* info_data = info->data;

  CopyIfDiffBuffer(x, x_out);

  auto uplo_v = static_cast<char>(uplo);
  auto x_order_v = CastNoOverflow<lapack_int>(x.dimensions.back());
  auto x_leading_dim_v = x_order_v;

  const int64_t x_out_step{x_rows * x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&uplo_v, &x_order_v, x_out_data, &x_leading_dim_v, info_data);
    x_out_data += x_out_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct CholeskyFactorization<ffi::DataType::F32>;
template struct CholeskyFactorization<ffi::DataType::F64>;
template struct CholeskyFactorization<ffi::DataType::C64>;
template struct CholeskyFactorization<ffi::DataType::C128>;

// Singular Value Decomposition (SVD)
// using divide and conquer method

namespace internal {

template <ffi::DataType dtype>
using RealBufferForComplexOrNull =
    std::conditional_t<ffi::IsComplexType<dtype>(),
                       ffi::ResultBuffer<ffi::ToReal(dtype)>, std::nullptr_t>;

template <ffi::DataType dtype>
static ffi::Error SvdKernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<LapackIntDtype> iwork, ffi::ResultBuffer<dtype> work,
    svd::ComputationMode mode, RealBufferForComplexOrNull<dtype> rwork) {
  if (mode == svd::ComputationMode::kComputeVtOverwriteXPartialU) [[unlikely]] {
    return ffi::Error(
        XLA_FFI_Error_Code_UNIMPLEMENTED,
        "Current implementation does not support this computation mode");
  }
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  auto* x_out_data = x_out->data;
  auto* singular_values_data = singular_values->data;
  auto* u_data = u->data;
  auto* vt_data = vt->data;
  auto* info_data = info->data;
  auto* iwork_data = iwork->data;
  auto* work_data = work->data;

  CopyIfDiffBuffer(x, x_out);

  auto x_rows_v = CastNoOverflow<lapack_int>(x_rows);
  auto x_cols_v = CastNoOverflow<lapack_int>(x_cols);
  auto mode_v = static_cast<char>(mode);
  auto workspace_dim_v = CastNoOverflow<lapack_int>(work->dimensions.back());
  auto x_leading_dim_v = x_rows_v;
  auto u_leading_dim_v = x_rows_v;

  auto u_dims = u->dimensions.last(2);
  auto vt_dims = vt->dimensions.last(2);
  auto vt_leading_dim_v = CastNoOverflow<lapack_int>(vt_dims.front());

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t singular_values_step{singular_values->dimensions.back()};
  const int64_t u_step{u_dims.front() * u_dims.back()};
  const int64_t vt_step{vt_leading_dim_v * vt_dims.back()};

  for (int64_t i = 0; i < batch_count; ++i) {
    if constexpr (ffi::IsComplexType<dtype>()) {
      svd::SVDType<dtype>::fn(&mode_v, &x_rows_v, &x_cols_v, x_out_data,
                              &x_leading_dim_v, singular_values_data, u_data,
                              &u_leading_dim_v, vt_data, &vt_leading_dim_v,
                              work_data, &workspace_dim_v, rwork->data,
                              iwork_data, info_data);
    } else {
      svd::SVDType<dtype>::fn(&mode_v, &x_rows_v, &x_cols_v, x_out_data,
                              &x_leading_dim_v, singular_values_data, u_data,
                              &u_leading_dim_v, vt_data, &vt_leading_dim_v,
                              work_data, &workspace_dim_v, iwork_data,
                              info_data);
    }
    x_out_data += x_out_step;
    singular_values_data += singular_values_step;
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
static int64_t SvdGetWorkspaceSize(lapack_int x_rows, lapack_int x_cols,
                                   svd::ComputationMode mode) {
  ffi::NativeType<dtype> optimal_size = {};
  lapack_int info = 0;
  lapack_int workspace_query = -1;

  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows;
  auto u_leading_dim_v = x_rows;
  auto vt_leading_dim_v = mode == svd::ComputationMode::kComputeFullUVt

                              ? x_cols
                              : std::min(x_rows, x_cols);
  if constexpr (ffi::IsComplexType<dtype>()) {
    svd::SVDType<dtype>::fn(
        &mode_v, &x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
        &u_leading_dim_v, nullptr, &vt_leading_dim_v, &optimal_size,
        &workspace_query, nullptr, nullptr, &info);
  } else {
    svd::SVDType<dtype>::fn(&mode_v, &x_rows, &x_cols, nullptr,
                            &x_leading_dim_v, nullptr, nullptr,
                            &u_leading_dim_v, nullptr, &vt_leading_dim_v,
                            &optimal_size, &workspace_query, nullptr, &info);
  }
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

}  // namespace internal

template <ffi::DataType dtype>
ffi::Error SingularValueDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> singular_values, ffi::ResultBuffer<dtype> u,
    ffi::ResultBuffer<dtype> vt, ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<LapackIntDtype> iwork, ffi::ResultBuffer<dtype> work,
    svd::ComputationMode mode) {
  return internal::SvdKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                    iwork, work, mode, nullptr);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<ffi::ToReal(dtype)> rwork,
    ffi::ResultBuffer<LapackIntDtype> iwork, ffi::ResultBuffer<dtype> work,
    svd::ComputationMode mode) {
  return internal::SvdKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                    iwork, work, mode, rwork);
}

template <ffi::DataType dtype>
int64_t SingularValueDecomposition<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype>
int64_t SingularValueDecompositionComplex<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

lapack_int svd::GetRealWorkspaceSize(int64_t x_rows, int64_t x_cols,
                                     svd::ComputationMode mode) {
  const auto min_dim = std::min(x_rows, x_cols);
  if (!ComputesUV(mode)) {
    return CastNoOverflow<lapack_int>(7 * min_dim);
  }
  const auto max_dim = std::max(x_rows, x_cols);
  return CastNoOverflow<lapack_int>(
      std::max(5 * min_dim * min_dim + 5 * min_dim,
               2 * max_dim * min_dim + 2 * min_dim * min_dim + min_dim));
}

lapack_int svd::GetIntWorkspaceSize(int64_t x_rows, int64_t x_cols) {
  return CastNoOverflow<lapack_int>(8 * std::min(x_rows, x_cols));
}

template struct SingularValueDecomposition<ffi::DataType::F32>;
template struct SingularValueDecomposition<ffi::DataType::F64>;
template struct SingularValueDecompositionComplex<ffi::DataType::C64>;
template struct SingularValueDecompositionComplex<ffi::DataType::C128>;

// Eigenvalues and eigenvectors

lapack_int eig::GetWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return CastNoOverflow<lapack_int>(2 * x_cols + 1);
    case ComputationMode::kComputeEigenvectors:
      return CastNoOverflow<lapack_int>(1 + 6 * x_cols + 2 * x_cols * x_cols);
  }
}

lapack_int eig::GetIntWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return 1;
    case ComputationMode::kComputeEigenvectors:
      return CastNoOverflow<lapack_int>(3 + 5 * x_cols);
  }
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionSymmetric<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, ffi::ResultBuffer<dtype> work,
    ffi::ResultBuffer<LapackIntDtype> iwork, eig::ComputationMode mode) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  auto* x_out_data = x_out->data;
  auto* eigenvalues_data = eigenvalues->data;
  auto* info_data = info->data;
  auto* work_data = work->data;
  auto* iwork_data = iwork->data;

  CopyIfDiffBuffer(x, x_out);

  auto mode_v = static_cast<char>(mode);
  auto uplo_v = static_cast<char>(uplo);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);
  auto workspace_dim_v = CastNoOverflow<lapack_int>(work->dimensions.back());
  auto iworkspace_dim_v = CastNoOverflow<lapack_int>(iwork->dimensions.back());
  auto x_leading_dim_v = CastNoOverflow<lapack_int>(x_cols);

  const int64_t x_out_step{x_cols * x_cols};
  const int64_t eigenvalues_step{x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &uplo_v, &x_order_v, x_out_data, &x_leading_dim_v,
       eigenvalues_data, work_data, &workspace_dim_v, iwork_data,
       &iworkspace_dim_v, info_data);
    x_out_data += x_out_step;
    eigenvalues_data += eigenvalues_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

namespace eig {

lapack_int GetComplexWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return CastNoOverflow<lapack_int>(x_cols + 1);
    case ComputationMode::kComputeEigenvectors:
      return CastNoOverflow<lapack_int>(2 * x_cols + x_cols * x_cols);
  }
}

lapack_int GetRealWorkspaceSize(int64_t x_cols, ComputationMode mode) {
  switch (mode) {
    case ComputationMode::kNoEigenvectors:
      return CastNoOverflow<lapack_int>(std::max(x_cols, int64_t{1}));
    case ComputationMode::kComputeEigenvectors:
      return CastNoOverflow<lapack_int>(1 + 5 * x_cols + 2 * x_cols * x_cols);
  }
}

}  // namespace eig

template <ffi::DataType dtype>
ffi::Error EigenvalueDecompositionHermitian<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info, ffi::ResultBuffer<dtype> work,
    ffi::ResultBuffer<ffi::ToReal(dtype)> rwork,
    ffi::ResultBuffer<LapackIntDtype> iwork, eig::ComputationMode mode) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  auto* x_out_data = x_out->data;
  auto* eigenvalues_data = eigenvalues->data;
  auto* info_data = info->data;
  auto* work_data = work->data;
  auto* iwork_data = iwork->data;

  CopyIfDiffBuffer(x, x_out);

  auto mode_v = static_cast<char>(mode);
  auto uplo_v = static_cast<char>(uplo);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);
  auto workspace_dim_v = CastNoOverflow<lapack_int>(work->dimensions.back());
  auto rworkspace_dim_v = CastNoOverflow<lapack_int>(rwork->dimensions.back());
  auto iworkspace_dim_v = CastNoOverflow<lapack_int>(iwork->dimensions.back());
  auto x_leading_dim_v = CastNoOverflow<lapack_int>(x_cols);

  const int64_t x_out_step{x_cols * x_cols};
  const int64_t eigenvalues_step{x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &uplo_v, &x_order_v, x_out_data, &x_leading_dim_v,
       eigenvalues_data, work_data, &workspace_dim_v, rwork->data,
       &rworkspace_dim_v, iwork_data, &iworkspace_dim_v, info_data);
    x_out_data += x_out_step;
    eigenvalues_data += eigenvalues_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct EigenvalueDecompositionSymmetric<ffi::DataType::F32>;
template struct EigenvalueDecompositionSymmetric<ffi::DataType::F64>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C64>;
template struct EigenvalueDecompositionHermitian<ffi::DataType::C128>;

// LAPACK uses a packed representation to represent a mixture of real
// eigenvectors and complex conjugate pairs. This helper unpacks the
// representation into regular complex matrices.
template <typename T>
static void UnpackEigenvectors(lapack_int n, const T* eigenvals_imag,
                               const T* packed, std::complex<T>* unpacked) {
  for (int j = 0; j < n;) {
    if (eigenvals_imag[j] == 0. || std::isnan(eigenvals_imag[j])) {
      // Real values in each row without imaginary part
      // Second row of the imaginary part is not provided
      for (int i = 0; i < n; ++i) {
        unpacked[j * n + i] = {packed[j * n + i], 0.};
      }
      ++j;
    } else {
      // Complex values where the real part is in the jth row
      // and the imaginary part is in the next row (j + 1)
      for (int i = 0; i < n; ++i) {
        const T real_part = packed[j * n + i];
        const T imag_part = packed[(j + 1) * n + i];
        unpacked[j * n + i] = {real_part, imag_part};
        unpacked[(j + 1) * n + i] = {real_part, -imag_part};
      }
      j += 2;
    }
  }
}

template <ffi::DataType dtype>
ffi::Error EigenvalueDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_left,
    ffi::ResultBuffer<ffi::ToComplex(dtype)> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info, ffi::ResultBuffer<dtype> x_work,
    ffi::ResultBuffer<ffi::ToReal(dtype)> work_eigvecs_left,
    ffi::ResultBuffer<ffi::ToReal(dtype)> work_eigvecs_right) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x_rows != x_cols) [[unlikely]] {
    throw std::invalid_argument(
        "Eigenvalue decomposition requires a square matrix");
  }

  const auto* x_data = x.data;
  auto* x_work_data = x_work->data;
  auto* work_eigvecs_left_data = work_eigvecs_left->data;
  auto* work_eigvecs_right_data = work_eigvecs_right->data;
  auto* eigvecs_left_data = eigvecs_left->data;
  auto* eigvecs_right_data = eigvecs_right->data;
  auto* eigvals_real_data = eigvals_real->data;
  auto* eigvals_imag_data = eigvals_imag->data;
  auto* info_data = info->data;

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);

  int64_t work_size = GetWorkspaceSize(x_order_v, compute_left, compute_right);
  auto work_size_v = CastNoOverflow<lapack_int>(work_size);
  // TODO(phawkins): preallocate workspace using XLA.
  auto work = std::make_unique<ValueType[]>(work_size);
  auto* work_data = work.get();

  const auto is_finite = [](ValueType* data, int64_t size) {
    return absl::c_all_of(absl::MakeSpan(data, size),
                          [](ValueType value) { return std::isfinite(value); });
  };

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    // TODO(paruzelp): copies the input buffer regardless - inconsistency
    //                 with the other kernels
    std::copy_n(x_data, x_size, x_work_data);
    if (is_finite(x_work_data, x_size)) {
      fn(&compute_left_v, &compute_right_v, &x_order_v, x_work_data, &x_order_v,
         eigvals_real_data, eigvals_imag_data, work_eigvecs_left_data,
         &x_order_v, work_eigvecs_right_data, &x_order_v, work_data,
         &work_size_v, info_data);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_work_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_real_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_imag_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(work_eigvecs_left_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(work_eigvecs_right_data,
                                          x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));
      if (info_data[0] == 0) {
        UnpackEigenvectors(x_order_v, eigvals_imag_data, work_eigvecs_left_data,
                           eigvecs_left_data);
        UnpackEigenvectors(x_order_v, eigvals_imag_data,
                           work_eigvecs_right_data, eigvecs_right_data);
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
ffi::Error EigenvalueDecompositionComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ffi::ResultBuffer<dtype> eigvals,
    ffi::ResultBuffer<dtype> eigvecs_left,
    ffi::ResultBuffer<dtype> eigvecs_right,
    ffi::ResultBuffer<LapackIntDtype> info, ffi::ResultBuffer<dtype> x_work,
    ffi::ResultBuffer<ffi::ToReal(dtype)> rwork) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x_rows != x_cols) [[unlikely]] {
    throw std::invalid_argument(
        "Eigenvalue decomposition requires a square matrix");
  }
  const auto* x_data = x.data;
  auto* x_work_data = x_work->data;
  auto* eigvecs_left_data = eigvecs_left->data;
  auto* eigvecs_right_data = eigvecs_right->data;
  auto* eigvals_data = eigvals->data;
  auto* info_data = info->data;

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);

  int64_t work_size = GetWorkspaceSize(x_order_v, compute_left, compute_right);
  auto work_size_v = CastNoOverflow<lapack_int>(work_size);
  // TODO(phawkins): preallocate workspace using XLA.
  auto work = std::make_unique<ValueType[]>(work_size);
  auto* work_data = work.get();

  const auto is_finite = [](ValueType* data, int64_t size) {
    return absl::c_all_of(absl::MakeSpan(data, size), [](const auto& z) {
      return std::isfinite(z.real()) && std::isfinite(z.imag());
    });
  };

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    std::copy_n(x_data, x_size, x_work_data);
    if (is_finite(x_work_data, x_size)) {
      fn(&compute_left_v, &compute_right_v, &x_order_v, x_work_data, &x_order_v,
         eigvals_data, eigvecs_left_data, &x_order_v, eigvecs_right_data,
         &x_order_v, work_data, &work_size_v, rwork->data, info_data);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_work_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_data, x_cols_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvecs_left_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvecs_right_data, x_size_bytes);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));
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
int64_t EigenvalueDecomposition<dtype>::GetWorkspaceSize(
    lapack_int x_cols, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;

  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  fn(&compute_left_v, &compute_right_v, &x_cols, nullptr, &x_cols, nullptr,
     nullptr, nullptr, &x_cols, nullptr, &x_cols, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template <ffi::DataType dtype>
int64_t EigenvalueDecompositionComplex<dtype>::GetWorkspaceSize(
    lapack_int x_cols, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;
  // NULL rwork crashes, LAPACK unnecessarily writes x_cols into rwork
  RealType rwork[1];
  auto compute_left_v = static_cast<char>(compute_left);
  auto compute_right_v = static_cast<char>(compute_right);
  fn(&compute_left_v, &compute_right_v, &x_cols, nullptr, &x_cols, nullptr,
     nullptr, &x_cols, nullptr, &x_cols, &optimal_size, &workspace_query, rwork,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template struct EigenvalueDecomposition<ffi::DataType::F32>;
template struct EigenvalueDecomposition<ffi::DataType::F64>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C64>;
template struct EigenvalueDecompositionComplex<ffi::DataType::C128>;

template <ffi::DataType dtype>
ffi::Error SchurDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> eigvals_real,
    ffi::ResultBuffer<dtype> eigvals_imag,
    ffi::ResultBuffer<dtype> schur_vectors,
    // TODO(paruzelp): Sort is not implemented because select function is not
    // supplied. For that reason, this parameter will always be zero!
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x_rows != x_cols) [[unlikely]] {
    throw std::invalid_argument("Schur decomposition requires a square matrix");
  }
  if (sort != schur::Sort::kNoSortEigenvalues) {
    throw std::runtime_error(
        "Ordering eigenvalues on the diagonal is not implemented");
  }

  CopyIfDiffBuffer(x, x_out);

  // TODO(paruzelp): `select` should be passed as an execution context
  bool (*select)(ValueType, ValueType) = nullptr;
  ValueType* x_out_data = x_out->data;
  ValueType* eigvals_real_data = eigvals_real->data;
  ValueType* eigvals_imag_data = eigvals_imag->data;
  ValueType* schur_vectors_data = schur_vectors->data;
  lapack_int* selected_data = selected_eigvals->data;
  lapack_int* info_data = info->data;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);

  std::unique_ptr<bool[]> bwork = sort != schur::Sort::kNoSortEigenvalues
                                      ? std::make_unique<bool[]>(x_cols)
                                      : nullptr;
  auto work_size = GetWorkspaceSize(x_cols, mode, sort);
  auto work = std::make_unique<ValueType[]>(work_size);
  auto work_size_v = CastNoOverflow<lapack_int>(work_size);

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &sort_v, select, &x_order_v, x_out_data, &x_order_v,
       selected_data, eigvals_real_data, eigvals_imag_data, schur_vectors_data,
       &x_order_v, work.get(), &work_size_v, bwork.get(), info_data);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(x_out_data, x_size_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(selected_data, sizeof(lapack_int));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_real_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_imag_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(schur_vectors_data, x_size_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));

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
ffi::Error SchurDecompositionComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> eigvals,
    ffi::ResultBuffer<dtype> schur_vectors,
    ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<ffi::ToReal(dtype)> rwork) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x_rows != x_cols) [[unlikely]] {
    throw std::invalid_argument("Schur decomposition requires a square matrix");
  }
  if (sort != schur::Sort::kNoSortEigenvalues) {
    throw std::runtime_error(
        "Ordering eigenvalues on the diagonal is not yet implemented."
        "It requires `select` function to be provided.");
  }

  CopyIfDiffBuffer(x, x_out);

  // TODO(paruzelp): `select` should be passed from the parameters
  bool (*select)(ValueType) = nullptr;
  ValueType* x_out_data = x_out->data;
  ValueType* eigvals_data = eigvals->data;
  ValueType* schur_vectors_data = schur_vectors->data;
  RealType* rwork_data = rwork->data;
  lapack_int* selected_data = selected_eigvals->data;
  lapack_int* info_data = info->data;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);

  std::unique_ptr<bool[]> bwork = sort != schur::Sort::kNoSortEigenvalues
                                      ? std::make_unique<bool[]>(x_cols)
                                      : nullptr;
  auto work_size = GetWorkspaceSize(x_cols, mode, sort);
  auto work = std::make_unique<ValueType[]>(work_size);
  auto work_size_v = CastNoOverflow<lapack_int>(work_size);

  const int64_t x_size{x_cols * x_cols};
  [[maybe_unused]] const auto x_size_bytes =
      static_cast<unsigned long>(x_size) * sizeof(ValueType);
  [[maybe_unused]] const auto x_cols_bytes =
      static_cast<unsigned long>(x_cols) * sizeof(ValueType);
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&mode_v, &sort_v, select, &x_order_v, x_out_data, &x_order_v,
       selected_data, eigvals_data, schur_vectors_data, &x_order_v, work.get(),
       &work_size_v, rwork_data, bwork.get(), info_data);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(eigvals_data, x_cols_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(schur_vectors_data, x_size_bytes);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_data, sizeof(lapack_int));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(selected_data, sizeof(lapack_int));

    x_out_data += x_size;
    eigvals_data += x_cols;
    schur_vectors_data += x_size;
    ++selected_data;
    ++info_data;
  }

  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t SchurDecomposition<dtype>::GetWorkspaceSize(lapack_int x_cols,
                                                    schur::ComputationMode mode,
                                                    schur::Sort sort) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  fn(&mode_v, &sort_v, nullptr, &x_cols, nullptr, &x_cols, nullptr, nullptr,
     nullptr, nullptr, &x_cols, &optimal_size, &workspace_query, nullptr,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template <ffi::DataType dtype>
int64_t SchurDecompositionComplex<dtype>::GetWorkspaceSize(
    lapack_int x_cols, schur::ComputationMode mode, schur::Sort sort) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;

  auto mode_v = static_cast<char>(mode);
  auto sort_v = static_cast<char>(sort);
  fn(&mode_v, &sort_v, nullptr, &x_cols, nullptr, &x_cols, nullptr, nullptr,
     nullptr, &x_cols, &optimal_size, &workspace_query, nullptr, nullptr,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
};

template struct SchurDecomposition<ffi::DataType::F32>;
template struct SchurDecomposition<ffi::DataType::F64>;
template struct SchurDecompositionComplex<ffi::DataType::C64>;
template struct SchurDecompositionComplex<ffi::DataType::C128>;

template <ffi::DataType dtype>
ffi::Error HessenbergDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, lapack_int low, lapack_int high,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> tau,
    ffi::ResultBuffer<LapackIntDtype> info, ffi::ResultBuffer<dtype> work) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x_rows != x_cols) [[unlikely]] {
    throw std::invalid_argument(
        "Hessenberg decomposition requires a square matrix");
  }

  CopyIfDiffBuffer(x, x_out);

  ValueType* x_out_data = x_out->data;
  ValueType* tau_data = tau->data;
  ValueType* work_data = work->data;
  lapack_int* info_data = info->data;

  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);
  auto x_leading_dim_v = CastNoOverflow<lapack_int>(x_rows);
  auto workspace_dim_v = CastNoOverflow<lapack_int>(work->dimensions.back());

  int64_t x_size{static_cast<int64_t>(x_leading_dim_v) * x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_order_v, &low, &high, x_out_data, &x_leading_dim_v, tau_data,
       work_data, &workspace_dim_v, info_data);
    x_out_data += x_size;
    tau_data += x_cols - 1;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t HessenbergDecomposition<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                         lapack_int x_cols,
                                                         lapack_int low,
                                                         lapack_int high) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;
  fn(&x_cols, &low, &high, nullptr, &x_rows, nullptr, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct HessenbergDecomposition<ffi::DataType::F32>;
template struct HessenbergDecomposition<ffi::DataType::F64>;
template struct HessenbergDecomposition<ffi::DataType::C64>;
template struct HessenbergDecomposition<ffi::DataType::C128>;

template <ffi::DataType dtype>
ffi::Error TridiagonalReduction<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<dtype> tau,
    ffi::ResultBuffer<ffi::ToReal(dtype)> diagonal,
    ffi::ResultBuffer<ffi::ToReal(dtype)> off_diagonal,
    ffi::ResultBuffer<LapackIntDtype> info, ffi::ResultBuffer<dtype> work) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions);
  if (x_rows != x_cols) [[unlikely]] {
    throw std::invalid_argument(
        "Tridiagonal reduction requires a square matrix");
  }

  CopyIfDiffBuffer(x, x_out);

  ValueType* x_out_data = x_out->data;
  RealType* diagonal_data = diagonal->data;
  RealType* off_diagonal_data = off_diagonal->data;
  ValueType* tau_data = tau->data;
  ValueType* work_data = work->data;
  lapack_int* info_data = info->data;

  auto uplo_v = static_cast<char>(uplo);
  auto x_leading_dim_v = CastNoOverflow<lapack_int>(x_rows);
  auto workspace_dim_v = CastNoOverflow<lapack_int>(work->dimensions.back());
  auto x_order_v = CastNoOverflow<lapack_int>(x_cols);

  int64_t x_size = {static_cast<int64_t>(x_leading_dim_v) * x_cols};
  int64_t tau_step = {tau->dimensions.back()};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&uplo_v, &x_order_v, x_out_data, &x_leading_dim_v, diagonal_data,
       off_diagonal_data, tau_data, work_data, &workspace_dim_v, info_data);
    x_out_data += x_size;
    diagonal_data += x_cols;
    off_diagonal_data += x_cols - 1;
    tau_data += tau_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t TridiagonalReduction<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                      lapack_int x_cols) {
  ValueType optimal_size = {};
  lapack_int workspace_query = -1;
  lapack_int info = 0;
  char uplo_v = 'L';
  fn(&uplo_v, &x_cols, nullptr, &x_rows, nullptr, nullptr, nullptr,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct TridiagonalReduction<ffi::DataType::F32>;
template struct TridiagonalReduction<ffi::DataType::F64>;
template struct TridiagonalReduction<ffi::DataType::C64>;
template struct TridiagonalReduction<ffi::DataType::C128>;

}  // namespace jax
