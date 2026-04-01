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

#include <cstdint>

#include "jaxlib/cpu/lapack_kernels.h"

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

namespace ffi = xla::ffi;

extern "C" {

jax::TriMatrixEquationSolver<ffi::DataType::F32, int32_t>::FnType strsm_;
jax::TriMatrixEquationSolver<ffi::DataType::F64, int32_t>::FnType dtrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C64, int32_t>::FnType ctrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C128, int32_t>::FnType ztrsm_;

jax::LuDecomposition<ffi::DataType::F32, int32_t>::FnType sgetrf_;
jax::LuDecomposition<ffi::DataType::F64, int32_t>::FnType dgetrf_;
jax::LuDecomposition<ffi::DataType::C64, int32_t>::FnType cgetrf_;
jax::LuDecomposition<ffi::DataType::C128, int32_t>::FnType zgetrf_;

jax::QrFactorization<ffi::DataType::F32, int32_t>::FnType sgeqrf_;
jax::QrFactorization<ffi::DataType::F64, int32_t>::FnType dgeqrf_;
jax::QrFactorization<ffi::DataType::C64, int32_t>::FnType cgeqrf_;
jax::QrFactorization<ffi::DataType::C128, int32_t>::FnType zgeqrf_;

jax::PivotingQrFactorization<ffi::DataType::F32, int32_t>::FnType sgeqp3_;
jax::PivotingQrFactorization<ffi::DataType::F64, int32_t>::FnType dgeqp3_;
jax::PivotingQrFactorization<ffi::DataType::C64, int32_t>::FnType cgeqp3_;
jax::PivotingQrFactorization<ffi::DataType::C128, int32_t>::FnType zgeqp3_;

jax::OrthogonalQr<ffi::DataType::F32, int32_t>::FnType sorgqr_;
jax::OrthogonalQr<ffi::DataType::F64, int32_t>::FnType dorgqr_;
jax::OrthogonalQr<ffi::DataType::C64, int32_t>::FnType cungqr_;
jax::OrthogonalQr<ffi::DataType::C128, int32_t>::FnType zungqr_;

jax::OrthogonalQrMultiply<ffi::DataType::F32, int32_t>::FnType sormqr_;
jax::OrthogonalQrMultiply<ffi::DataType::F64, int32_t>::FnType dormqr_;
jax::OrthogonalQrMultiply<ffi::DataType::C64, int32_t>::FnType cunmqr_;
jax::OrthogonalQrMultiply<ffi::DataType::C128, int32_t>::FnType zunmqr_;

jax::CholeskyFactorization<ffi::DataType::F32, int32_t>::FnType spotrf_;
jax::CholeskyFactorization<ffi::DataType::F64, int32_t>::FnType dpotrf_;
jax::CholeskyFactorization<ffi::DataType::C64, int32_t>::FnType cpotrf_;
jax::CholeskyFactorization<ffi::DataType::C128, int32_t>::FnType zpotrf_;

jax::SingularValueDecomposition<ffi::DataType::F32, int32_t>::FnType sgesdd_;
jax::SingularValueDecomposition<ffi::DataType::F64, int32_t>::FnType dgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C64, int32_t>::FnType
    cgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C128, int32_t>::FnType
    zgesdd_;

jax::SingularValueDecompositionQR<ffi::DataType::F32, int32_t>::FnType sgesvd_;
jax::SingularValueDecompositionQR<ffi::DataType::F64, int32_t>::FnType dgesvd_;
jax::SingularValueDecompositionQRComplex<ffi::DataType::C64, int32_t>::FnType
    cgesvd_;
jax::SingularValueDecompositionQRComplex<ffi::DataType::C128, int32_t>::FnType
    zgesvd_;

jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32, int32_t>::FnType
    ssyevd_;
jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64, int32_t>::FnType
    dsyevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C64, int32_t>::FnType
    cheevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C128, int32_t>::FnType
    zheevd_;

jax::EigenvalueDecomposition<ffi::DataType::F32, int32_t>::FnType sgeev_;
jax::EigenvalueDecomposition<ffi::DataType::F64, int32_t>::FnType dgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C64, int32_t>::FnType cgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C128, int32_t>::FnType
    zgeev_;

jax::SchurDecomposition<ffi::DataType::F32, int32_t>::FnType sgees_;
jax::SchurDecomposition<ffi::DataType::F64, int32_t>::FnType dgees_;
jax::SchurDecompositionComplex<ffi::DataType::C64, int32_t>::FnType cgees_;
jax::SchurDecompositionComplex<ffi::DataType::C128, int32_t>::FnType zgees_;

jax::HessenbergDecomposition<ffi::DataType::F32, int32_t>::FnType sgehrd_;
jax::HessenbergDecomposition<ffi::DataType::F64, int32_t>::FnType dgehrd_;
jax::HessenbergDecomposition<ffi::DataType::C64, int32_t>::FnType cgehrd_;
jax::HessenbergDecomposition<ffi::DataType::C128, int32_t>::FnType zgehrd_;

jax::TridiagonalReduction<ffi::DataType::F32, int32_t>::FnType ssytrd_;
jax::TridiagonalReduction<ffi::DataType::F64, int32_t>::FnType dsytrd_;
jax::TridiagonalReduction<ffi::DataType::C64, int32_t>::FnType chetrd_;
jax::TridiagonalReduction<ffi::DataType::C128, int32_t>::FnType zhetrd_;

jax::TridiagonalSolver<ffi::DataType::F32, int32_t>::FnType sgtsv_;
jax::TridiagonalSolver<ffi::DataType::F64, int32_t>::FnType dgtsv_;
jax::TridiagonalSolver<ffi::DataType::C64, int32_t>::FnType cgtsv_;
jax::TridiagonalSolver<ffi::DataType::C128, int32_t>::FnType zgtsv_;

}  // extern "C"

namespace jax {

static auto init = []() -> int {
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F32, int32_t>>(strsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F64, int32_t>>(dtrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C64, int32_t>>(ctrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C128, int32_t>>(ztrsm_);

  AssignKernelFn<LuDecomposition<ffi::DataType::F32, int32_t>>(sgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::F64, int32_t>>(dgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C64, int32_t>>(cgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C128, int32_t>>(zgetrf_);

  AssignKernelFn<QrFactorization<ffi::DataType::F32, int32_t>>(sgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::F64, int32_t>>(dgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C64, int32_t>>(cgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C128, int32_t>>(zgeqrf_);

  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F32, int32_t>>(sgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F64, int32_t>>(dgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C64, int32_t>>(cgeqp3_);
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C128, int32_t>>(
      zgeqp3_);

  AssignKernelFn<OrthogonalQr<ffi::DataType::F32, int32_t>>(sorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::F64, int32_t>>(dorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C64, int32_t>>(cungqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C128, int32_t>>(zungqr_);

  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F32, int32_t>>(sormqr_);
  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F64, int32_t>>(dormqr_);
  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C64, int32_t>>(cunmqr_);
  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C128, int32_t>>(zunmqr_);

  AssignKernelFn<CholeskyFactorization<ffi::DataType::F32, int32_t>>(spotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::F64, int32_t>>(dpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C64, int32_t>>(cpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C128, int32_t>>(zpotrf_);

  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F32, int32_t>>(
      sgesdd_);
  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F64, int32_t>>(
      dgesdd_);
  AssignKernelFn<
      SingularValueDecompositionComplex<ffi::DataType::C64, int32_t>>(cgesdd_);
  AssignKernelFn<
      SingularValueDecompositionComplex<ffi::DataType::C128, int32_t>>(zgesdd_);

  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F32, int32_t>>(
      sgesvd_);
  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F64, int32_t>>(
      dgesvd_);
  AssignKernelFn<
      SingularValueDecompositionQRComplex<ffi::DataType::C64, int32_t>>(
      cgesvd_);
  AssignKernelFn<
      SingularValueDecompositionQRComplex<ffi::DataType::C128, int32_t>>(
      zgesvd_);

  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F32, int32_t>>(
      ssyevd_);
  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F64, int32_t>>(
      dsyevd_);
  AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C64, int32_t>>(
      cheevd_);
  AssignKernelFn<
      EigenvalueDecompositionHermitian<ffi::DataType::C128, int32_t>>(zheevd_);

  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F32, int32_t>>(sgeev_);
  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F64, int32_t>>(dgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C64, int32_t>>(
      cgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C128, int32_t>>(
      zgeev_);

  AssignKernelFn<TridiagonalReduction<ffi::DataType::F32, int32_t>>(ssytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::F64, int32_t>>(dsytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C64, int32_t>>(chetrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C128, int32_t>>(zhetrd_);

  AssignKernelFn<SchurDecomposition<ffi::DataType::F32, int32_t>>(sgees_);
  AssignKernelFn<SchurDecomposition<ffi::DataType::F64, int32_t>>(dgees_);
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C64, int32_t>>(
      cgees_);
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C128, int32_t>>(
      zgees_);

  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F32, int32_t>>(sgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F64, int32_t>>(dgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C64, int32_t>>(cgehrd_);
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C128, int32_t>>(
      zgehrd_);

  AssignKernelFn<TridiagonalSolver<ffi::DataType::F32, int32_t>>(sgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::F64, int32_t>>(dgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C64, int32_t>>(cgtsv_);
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C128, int32_t>>(zgtsv_);

  lapack_kernels_initialized = true;
  return 0;
}();

}  // namespace jax
