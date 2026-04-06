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

#ifndef JAXLIB_CPU_LAPACK_KERNELS_H_
#define JAXLIB_CPU_LAPACK_KERNELS_H_

#include <complex>
#include <cstdint>
#include <type_traits>

#include "absl/status/statusor.h"
#include "xla/ffi/api/ffi.h"

// Underlying function pointers (i.e., KERNEL_CLASS::Fn) are initialized either
// by the nanobind wrapper that links them to an existing SciPy lapack instance,
// or using the lapack_kernels_strong.cc static initialization to link them
// directly to lapack for use in a pure C++ context.

namespace jax {

extern bool lapack_kernels_initialized;

struct MatrixParams {
  enum class Side : char { kLeft = 'L', kRight = 'R' };
  enum class UpLo : char { kLower = 'L', kUpper = 'U' };
  enum class Diag : char { kNonUnit = 'N', kUnit = 'U' };
  enum class Transpose : char {
    kNoTrans = 'N',
    kTrans = 'T',
    kConjTrans = 'C'
  };
};

namespace svd {

enum class ComputationMode : char {
  kComputeFullUVt = 'A',  // Compute U and VT
  kComputeMinUVt = 'S',   // Compute min(M, N) columns of U and rows of VT
  kComputeVtOverwriteXPartialU = 'O',  // Compute VT, overwrite X
                                       // with partial U
  kNoComputeUVt = 'N',                 // Do not compute U or VT
};

inline bool ComputesUV(ComputationMode mode) {
  return mode == ComputationMode::kComputeFullUVt ||
         mode == ComputationMode::kComputeMinUVt;
}

}  // namespace svd

namespace eig {

enum class ComputationMode : char {
  kNoEigenvectors = 'N',
  kComputeEigenvectors = 'V',
};

}  // namespace eig

namespace schur {

enum class ComputationMode : char {
  kNoComputeSchurVectors = 'N',
  kComputeSchurVectors = 'V',
};

enum class Sort : char { kNoSortEigenvalues = 'N', kSortEigenvalues = 'S' };

}  // namespace schur

template <typename KernelType>
void AssignKernelFn(void* func) {
  KernelType::fn = reinterpret_cast<typename KernelType::FnType*>(func);
}

template <typename KernelType>
void AssignKernelFn(typename KernelType::FnType* func) {
  KernelType::fn = func;
}

}  // namespace jax

namespace jax {

inline constexpr auto LapackIntDtype = ::xla::ffi::DataType::S32;
inline constexpr auto LapackIntDtype64 = ::xla::ffi::DataType::S64;
static_assert(std::is_same_v<::xla::ffi::NativeType<LapackIntDtype>, int32_t>);

template <typename IntType>
inline constexpr auto LapackIntDtypeFor =
    std::is_same_v<IntType, int32_t> ? ::xla::ffi::DataType::S32
                                     : ::xla::ffi::DataType::S64;

//== Triangular System Solver ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct TriMatrixEquationSolver {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* side, char* uplo, char* transa, char* diag,
                      IntType* m, IntType* n, ValueType* alpha, ValueType* a,
                      IntType* lda, ValueType* b, IntType* ldb);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  ::xla::ffi::Buffer<dtype> y,
                                  ::xla::ffi::ResultBuffer<dtype> y_out,
                                  MatrixParams::Side side,
                                  MatrixParams::UpLo uplo,
                                  MatrixParams::Transpose trans_x,
                                  MatrixParams::Diag diag);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error TriMatrixEquationSolverKernel(
    ::xla::ffi::Buffer<dtype> x, ::xla::ffi::Buffer<dtype> y,
    ::xla::ffi::ResultBuffer<dtype> y_out, MatrixParams::Side side,
    MatrixParams::UpLo uplo, MatrixParams::Transpose trans_x,
    MatrixParams::Diag diag);

//== LU Decomposition ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct LuDecomposition {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(IntType* m, IntType* n, ValueType* a, IntType* lda,
                      IntType* ipiv, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> ipiv,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error LuDecompositionKernel(
    ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<LapackIntDtype> ipiv,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

//== QR Factorization ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct QrFactorization {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(IntType* m, IntType* n, ValueType* a, IntType* lda,
                      ValueType* tau, ValueType* work, IntType* lwork,
                      IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  ::xla::ffi::ResultBuffer<dtype> x_out,
                                  ::xla::ffi::ResultBuffer<dtype> tau);

  static int64_t GetWorkspaceSize(IntType x_rows, IntType x_cols);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error QrFactorizationKernel(::xla::ffi::Buffer<dtype> x,
                                        ::xla::ffi::ResultBuffer<dtype> x_out,
                                        ::xla::ffi::ResultBuffer<dtype> tau);

//== Column Pivoting QR Factorization ==//

// lapack geqp3
template <::xla::ffi::DataType dtype, typename IntType>
struct PivotingQrFactorization {
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = std::conditional_t<
      ::xla::ffi::IsComplexType<dtype>(),
      void(IntType* m, IntType* n, ValueType* a, IntType* lda, IntType* jpvt,
           ValueType* tau, ValueType* work, IntType* lwork, RealType* rwork,
           IntType* info),
      void(IntType* m, IntType* n, ValueType* a, IntType* lda, IntType* jpvt,
           ValueType* tau, ValueType* work, IntType* lwork, IntType* info)>;

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::Buffer<LapackIntDtype> jpvt,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> jpvt_out,
      ::xla::ffi::ResultBuffer<dtype> tau);

  static int64_t GetWorkspaceSize(IntType x_rows, IntType x_cols);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error PivotingQrFactorizationKernel(
    ::xla::ffi::Buffer<dtype> x, ::xla::ffi::Buffer<LapackIntDtype> jpvt,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<LapackIntDtype> jpvt_out,
    ::xla::ffi::ResultBuffer<dtype> tau);

//== Orthogonal QR ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct OrthogonalQr {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(IntType* m, IntType* n, IntType* k, ValueType* a,
                      IntType* lda, ValueType* tau, ValueType* work,
                      IntType* lwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  ::xla::ffi::Buffer<dtype> tau,
                                  ::xla::ffi::ResultBuffer<dtype> x_out);

  static int64_t GetWorkspaceSize(IntType x_rows, IntType x_cols,
                                  IntType tau_size);
};

//== Orthogonal QR Multiply ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct OrthogonalQrMultiply {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* side, char* trans, IntType* m, IntType* n,
                      IntType* k, ValueType* a, IntType* lda, ValueType* tau,
                      ValueType* c, IntType* ldc, ValueType* work,
                      IntType* lwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> a,
                                  ::xla::ffi::Buffer<dtype> tau,
                                  ::xla::ffi::Buffer<dtype> c, bool left,
                                  bool transpose,
                                  ::xla::ffi::ResultBuffer<dtype> c_out);

  static int64_t GetWorkspaceSize(char side, char trans, IntType m, IntType n,
                                  IntType k, IntType lda);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error OrthogonalQrKernel(::xla::ffi::Buffer<dtype> x,
                                     ::xla::ffi::Buffer<dtype> tau,
                                     ::xla::ffi::ResultBuffer<dtype> x_out);

//== Cholesky Factorization ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct CholeskyFactorization {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* uplo, IntType* n, ValueType* a, IntType* lda,
                      IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error CholeskyFactorizationKernel(
    ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

//== Singular Value Decomposition (SVD) ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct SingularValueDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ValueType;
  using FnType = void(char* jobz, IntType* m, IntType* n, ValueType* a,
                      IntType* lda, ValueType* s, ValueType* u, IntType* ldu,
                      ValueType* vt, IntType* ldvt, ValueType* work,
                      IntType* lwork, IntType* iwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::ThreadPool thread_pool, ::xla::ffi::Buffer<dtype> x,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<int64_t> GetWorkspaceSize(IntType x_rows,
                                                  IntType x_cols,
                                                  svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error SingularValueDecompositionKernel(
    ::xla::ffi::ThreadPool thread_pool, ::xla::ffi::Buffer<dtype> x,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<dtype> singular_values,
    ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

template <::xla::ffi::DataType dtype, typename IntType>
struct SingularValueDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobz, IntType* m, IntType* n, ValueType* a,
                      IntType* lda, RealType* s, ValueType* u, IntType* ldu,
                      ValueType* vt, IntType* ldvt, ValueType* work,
                      IntType* lwork, RealType* rwork, IntType* iwork,
                      IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::ThreadPool thread_pool, ::xla::ffi::Buffer<dtype> x,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<int64_t> GetWorkspaceSize(IntType x_rows,
                                                  IntType x_cols,
                                                  svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error SingularValueDecompositionComplexKernel(
    ::xla::ffi::ThreadPool thread_pool, ::xla::ffi::Buffer<dtype> x,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> singular_values,
    ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

template <::xla::ffi::DataType dtype, typename IntType>
struct SingularValueDecompositionQR {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ValueType;
  using FnType = void(char* jobu, char* jobvt, IntType* m, IntType* n,
                      ValueType* a, IntType* lda, ValueType* s, ValueType* u,
                      IntType* ldu, ValueType* vt, IntType* ldvt,
                      ValueType* work, IntType* lwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<IntType> GetWorkspaceSize(IntType x_rows,
                                                  IntType x_cols,
                                                  svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error SingularValueDecompositionQRKernel(
    ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<dtype> singular_values,
    ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

template <::xla::ffi::DataType dtype, typename IntType>
struct SingularValueDecompositionQRComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobu, char* jobvt, IntType* m, IntType* n,
                      ValueType* a, IntType* lda, RealType* s, ValueType* u,
                      IntType* ldu, ValueType* vt, IntType* ldvt,
                      ValueType* work, IntType* lwork, RealType* rwork,
                      IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<IntType> GetWorkspaceSize(IntType x_rows,
                                                  IntType x_cols,
                                                  svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error SingularValueDecompositionQRComplexKernel(
    ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> singular_values,
    ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

namespace svd {

template <::xla::ffi::DataType dtype, typename IntType = int32_t>
using SVDType =
    std::conditional_t<::xla::ffi::IsComplexType<dtype>(),
                       SingularValueDecompositionComplex<dtype, IntType>,
                       SingularValueDecomposition<dtype, IntType>>;

template <::xla::ffi::DataType dtype, typename IntType = int32_t>
using SVDQRType =
    std::conditional_t<::xla::ffi::IsComplexType<dtype>(),
                       SingularValueDecompositionQRComplex<dtype, IntType>,
                       SingularValueDecompositionQR<dtype, IntType>>;

int64_t GetIntWorkspaceSize(int64_t x_rows, int64_t x_cols);
int64_t GetRealWorkspaceSize(int64_t x_rows, int64_t x_cols,
                             ComputationMode mode);
int64_t GetRealWorkspaceSizeQR(int64_t x_rows, int64_t x_cols);

}  // namespace svd

//== Eigenvalues and eigenvectors ==//

namespace eig {

// Eigenvalue Decomposition
int64_t GetWorkspaceSize(int64_t x_cols, ComputationMode mode);
int64_t GetIntWorkspaceSize(int64_t x_cols, ComputationMode mode);

// Hermitian Eigenvalue Decomposition
int64_t GetComplexWorkspaceSize(int64_t x_cols, ComputationMode mode);
int64_t GetRealWorkspaceSize(int64_t x_cols, ComputationMode mode);

}  // namespace eig

template <::xla::ffi::DataType dtype, typename IntType>
struct EigenvalueDecompositionSymmetric {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobz, char* uplo, IntType* n, ValueType* a,
                      IntType* lda, ValueType* w, ValueType* work,
                      IntType* lwork, IntType* iwork, IntType* liwork,
                      IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  MatrixParams::UpLo uplo,
                                  ::xla::ffi::ResultBuffer<dtype> x_out,
                                  ::xla::ffi::ResultBuffer<dtype> eigenvalues,
                                  ::xla::ffi::ResultBuffer<LapackIntDtype> info,
                                  eig::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error EigenvalueDecompositionSymmetricKernel(
    ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<dtype> eigenvalues,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode);

template <::xla::ffi::DataType dtype, typename IntType>
struct EigenvalueDecompositionHermitian {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobz, char* uplo, IntType* n, ValueType* a,
                      IntType* lda, RealType* w, ValueType* work,
                      IntType* lwork, RealType* rwork, IntType* lrwork,
                      IntType* iwork, IntType* liwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> eigenvalues,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error EigenvalueDecompositionHermitianKernel(
    ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> eigenvalues,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info, eig::ComputationMode mode);

// LAPACK uses a packed representation to represent a mixture of real
// eigenvectors and complex conjugate pairs. This helper unpacks the
// representation into regular complex matrices.
template <typename T, typename Int = int32_t>
void UnpackEigenvectors(Int n, const T* eigenvals_imag, const T* packed,
                        std::complex<T>* unpacked) {
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

template <::xla::ffi::DataType dtype, typename IntType>
struct EigenvalueDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobvl, char* jobvr, IntType* n, ValueType* a,
                      IntType* lda, ValueType* wr, ValueType* wi, ValueType* vl,
                      IntType* ldvl, ValueType* vr, IntType* ldvr,
                      ValueType* work, IntType* lwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
      eig::ComputationMode compute_right,
      ::xla::ffi::ResultBuffer<dtype> eigvals_real,
      ::xla::ffi::ResultBuffer<dtype> eigvals_imag,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_left,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_right,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(IntType x_cols,
                                  eig::ComputationMode compute_left,
                                  eig::ComputationMode compute_right);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error EigenvalueDecompositionKernel(
    ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right,
    ::xla::ffi::ResultBuffer<dtype> eigvals_real,
    ::xla::ffi::ResultBuffer<dtype> eigvals_imag,
    ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_left,
    ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_right,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

template <::xla::ffi::DataType dtype, typename IntType>
struct EigenvalueDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobvl, char* jobvr, IntType* n, ValueType* a,
                      IntType* lda, ValueType* w, ValueType* vl, IntType* ldvl,
                      ValueType* vr, IntType* ldvr, ValueType* work,
                      IntType* lwork, RealType* rwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
      eig::ComputationMode compute_right,
      ::xla::ffi::ResultBuffer<dtype> eigvals,
      ::xla::ffi::ResultBuffer<dtype> eigvecs_left,
      ::xla::ffi::ResultBuffer<dtype> eigvecs_right,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(IntType x_cols,
                                  eig::ComputationMode compute_left,
                                  eig::ComputationMode compute_right);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error EigenvalueDecompositionComplexKernel(
    ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
    eig::ComputationMode compute_right, ::xla::ffi::ResultBuffer<dtype> eigvals,
    ::xla::ffi::ResultBuffer<dtype> eigvecs_left,
    ::xla::ffi::ResultBuffer<dtype> eigvecs_right,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

//== Schur Decomposition ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct SchurDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobvs, char* sort,
                      bool (*select)(ValueType, ValueType), IntType* n,
                      ValueType* a, IntType* lda, IntType* sdim, ValueType* wr,
                      ValueType* wi, ValueType* vs, IntType* ldvs,
                      ValueType* work, IntType* lwork, bool* bwork,
                      IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, schur::ComputationMode mode,
      schur::Sort sort, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> schur_vectors,
      ::xla::ffi::ResultBuffer<dtype> eigvals_real,
      ::xla::ffi::ResultBuffer<dtype> eigvals_imag,
      ::xla::ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(IntType x_cols, schur::ComputationMode mode,
                                  schur::Sort sort);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error SchurDecompositionKernel(
    ::xla::ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<dtype> schur_vectors,
    ::xla::ffi::ResultBuffer<dtype> eigvals_real,
    ::xla::ffi::ResultBuffer<dtype> eigvals_imag,
    ::xla::ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

template <::xla::ffi::DataType dtype, typename IntType>
struct SchurDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobvs, char* sort, bool (*select)(ValueType),
                      IntType* n, ValueType* a, IntType* lda, IntType* sdim,
                      ValueType* w, ValueType* vs, IntType* ldvs,
                      ValueType* work, IntType* lwork, RealType* rwork,
                      bool* bwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, schur::ComputationMode mode,
      schur::Sort sort, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> schur_vectors,
      ::xla::ffi::ResultBuffer<dtype> eigvals,
      ::xla::ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(IntType x_cols, schur::ComputationMode mode,
                                  schur::Sort sort);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error SchurDecompositionComplexKernel(
    ::xla::ffi::Buffer<dtype> x, schur::ComputationMode mode, schur::Sort sort,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<dtype> schur_vectors,
    ::xla::ffi::ResultBuffer<dtype> eigvals,
    ::xla::ffi::ResultBuffer<LapackIntDtype> selected_eigvals,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

//== Hessenberg Decomposition                                       ==//
//== Reduces a non-symmetric square matrix to upper Hessenberg form ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct HessenbergDecomposition {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(IntType* n, IntType* ilo, IntType* ihi, ValueType* a,
                      IntType* lda, ValueType* tau, ValueType* work,
                      IntType* lwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, int32_t low, int32_t high,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> tau,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(IntType x_rows, IntType x_cols, IntType low,
                                  IntType high);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error HessenbergDecompositionKernel(
    ::xla::ffi::Buffer<dtype> x, int32_t low, int32_t high,
    ::xla::ffi::ResultBuffer<dtype> x_out, ::xla::ffi::ResultBuffer<dtype> tau,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

//== Tridiagonal Reduction                                           ==//
//== Reduces a Symmetric/Hermitian square matrix to tridiagonal form ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct TridiagonalReduction {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* uplo, IntType* n, ValueType* a, IntType* lda,
                      RealType* d, RealType* e, ValueType* tau, ValueType* work,
                      IntType* lwork, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> diagonal,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> off_diagonal,
      ::xla::ffi::ResultBuffer<dtype> tau,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(IntType x_rows, IntType x_cols);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error TridiagonalReductionKernel(
    ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ::xla::ffi::ResultBuffer<dtype> x_out,
    ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> diagonal,
    ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> off_diagonal,
    ::xla::ffi::ResultBuffer<dtype> tau,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

//== General Tridiagonal System Solver ==//

template <::xla::ffi::DataType dtype, typename IntType>
struct TridiagonalSolver {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(IntType* n, IntType* nrhs, ValueType* dl, ValueType* d,
                      ValueType* du, ValueType* b, IntType* ldb, IntType* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> dl, ::xla::ffi::Buffer<dtype> d,
      ::xla::ffi::Buffer<dtype> du, ::xla::ffi::Buffer<dtype> b,
      ::xla::ffi::ResultBuffer<dtype> dl_out,
      ::xla::ffi::ResultBuffer<dtype> d_out,
      ::xla::ffi::ResultBuffer<dtype> du_out,
      ::xla::ffi::ResultBuffer<dtype> b_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

template <::xla::ffi::DataType dtype>
::xla::ffi::Error TridiagonalSolverKernel(
    ::xla::ffi::Buffer<dtype> dl, ::xla::ffi::Buffer<dtype> d,
    ::xla::ffi::Buffer<dtype> du, ::xla::ffi::Buffer<dtype> b,
    ::xla::ffi::ResultBuffer<dtype> dl_out,
    ::xla::ffi::ResultBuffer<dtype> d_out,
    ::xla::ffi::ResultBuffer<dtype> du_out,
    ::xla::ffi::ResultBuffer<dtype> b_out,
    ::xla::ffi::ResultBuffer<LapackIntDtype> info);

// Declare all the handler symbols
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_strsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dtrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ctrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ztrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeqp3_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sorgqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dorgqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cungqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zungqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_spotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgesvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ssyevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dsyevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cheevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zheevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ssytrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dsytrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_chetrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zhetrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgees_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgehrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgtsv_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgtsv_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgtsv_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgtsv_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sormqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dormqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cunmqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zunmqr_ffi);

}  // namespace jax

#endif  // JAXLIB_CPU_LAPACK_KERNELS_H_
