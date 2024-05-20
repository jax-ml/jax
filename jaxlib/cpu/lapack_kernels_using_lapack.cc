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

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

namespace ffi = xla::ffi;

extern "C" {

jax::TriMatrixEquationSolver<ffi::DataType::F32>::FnType strsm_;
jax::TriMatrixEquationSolver<ffi::DataType::F64>::FnType dtrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C64>::FnType ctrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C128>::FnType ztrsm_;

jax::LuDecomposition<ffi::DataType::F32>::FnType sgetrf_;
jax::LuDecomposition<ffi::DataType::F64>::FnType dgetrf_;
jax::LuDecomposition<ffi::DataType::C64>::FnType cgetrf_;
jax::LuDecomposition<ffi::DataType::C128>::FnType zgetrf_;

jax::QrFactorization<ffi::DataType::F32>::FnType sgeqrf_;
jax::QrFactorization<ffi::DataType::F64>::FnType dgeqrf_;
jax::QrFactorization<ffi::DataType::C64>::FnType cgeqrf_;
jax::QrFactorization<ffi::DataType::C128>::FnType zgeqrf_;

jax::OrthogonalQr<ffi::DataType::F32>::FnType sorgqr_;
jax::OrthogonalQr<ffi::DataType::F64>::FnType dorgqr_;
jax::OrthogonalQr<ffi::DataType::C64>::FnType cungqr_;
jax::OrthogonalQr<ffi::DataType::C128>::FnType zungqr_;

jax::CholeskyFactorization<ffi::DataType::F32>::FnType spotrf_;
jax::CholeskyFactorization<ffi::DataType::F64>::FnType dpotrf_;
jax::CholeskyFactorization<ffi::DataType::C64>::FnType cpotrf_;
jax::CholeskyFactorization<ffi::DataType::C128>::FnType zpotrf_;

jax::SingularValueDecomposition<ffi::DataType::F32>::FnType sgesdd_;
jax::SingularValueDecomposition<ffi::DataType::F64>::FnType dgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C64>::FnType cgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C128>::FnType zgesdd_;

jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32>::FnType ssyevd_;
jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64>::FnType dsyevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C64>::FnType cheevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C128>::FnType zheevd_;

jax::EigenvalueDecomposition<ffi::DataType::F32>::FnType sgeev_;
jax::EigenvalueDecomposition<ffi::DataType::F64>::FnType dgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C64>::FnType cgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C128>::FnType zgeev_;

jax::SchurDecomposition<ffi::DataType::F32>::FnType sgees_;
jax::SchurDecomposition<ffi::DataType::F64>::FnType dgees_;
jax::SchurDecompositionComplex<ffi::DataType::C64>::FnType cgees_;
jax::SchurDecompositionComplex<ffi::DataType::C128>::FnType zgees_;

jax::HessenbergDecomposition<ffi::DataType::F32>::FnType sgehrd_;
jax::HessenbergDecomposition<ffi::DataType::F64>::FnType dgehrd_;
jax::HessenbergDecomposition<ffi::DataType::C64>::FnType cgehrd_;
jax::HessenbergDecomposition<ffi::DataType::C128>::FnType zgehrd_;

jax::TridiagonalReduction<ffi::DataType::F32>::FnType ssytrd_;
jax::TridiagonalReduction<ffi::DataType::F64>::FnType dsytrd_;
jax::TridiagonalReduction<ffi::DataType::C64>::FnType chetrd_;
jax::TridiagonalReduction<ffi::DataType::C128>::FnType zhetrd_;

}  // extern "C"

namespace jax {

static auto init = []() -> int {
  TriMatrixEquationSolver<ffi::DataType::F32>::fn = strsm_;
  TriMatrixEquationSolver<ffi::DataType::F64>::fn = dtrsm_;
  TriMatrixEquationSolver<ffi::DataType::C64>::fn = ctrsm_;
  TriMatrixEquationSolver<ffi::DataType::C128>::fn = ztrsm_;

  LuDecomposition<ffi::DataType::F32>::fn = sgetrf_;
  LuDecomposition<ffi::DataType::F64>::fn = dgetrf_;
  LuDecomposition<ffi::DataType::C64>::fn = cgetrf_;
  LuDecomposition<ffi::DataType::C128>::fn = zgetrf_;

  QrFactorization<ffi::DataType::F32>::fn = sgeqrf_;
  QrFactorization<ffi::DataType::F64>::fn = dgeqrf_;
  QrFactorization<ffi::DataType::C64>::fn = cgeqrf_;
  QrFactorization<ffi::DataType::C128>::fn = zgeqrf_;

  OrthogonalQr<ffi::DataType::F32>::fn = sorgqr_;
  OrthogonalQr<ffi::DataType::F64>::fn = dorgqr_;
  OrthogonalQr<ffi::DataType::C64>::fn = cungqr_;
  OrthogonalQr<ffi::DataType::C128>::fn = zungqr_;

  CholeskyFactorization<ffi::DataType::F32>::fn = spotrf_;
  CholeskyFactorization<ffi::DataType::F64>::fn = dpotrf_;
  CholeskyFactorization<ffi::DataType::C64>::fn = cpotrf_;
  CholeskyFactorization<ffi::DataType::C128>::fn = zpotrf_;

  SingularValueDecomposition<ffi::DataType::F32>::fn = sgesdd_;
  SingularValueDecomposition<ffi::DataType::F64>::fn = dgesdd_;
  SingularValueDecompositionComplex<ffi::DataType::C64>::fn = cgesdd_;
  SingularValueDecompositionComplex<ffi::DataType::C128>::fn = zgesdd_;

  EigenvalueDecompositionSymmetric<ffi::DataType::F32>::fn = ssyevd_;
  EigenvalueDecompositionSymmetric<ffi::DataType::F64>::fn = dsyevd_;
  EigenvalueDecompositionHermitian<ffi::DataType::C64>::fn = cheevd_;
  EigenvalueDecompositionHermitian<ffi::DataType::C128>::fn = zheevd_;

  EigenvalueDecomposition<ffi::DataType::F32>::fn = sgeev_;
  EigenvalueDecomposition<ffi::DataType::F64>::fn = dgeev_;
  EigenvalueDecompositionComplex<ffi::DataType::C64>::fn = cgeev_;
  EigenvalueDecompositionComplex<ffi::DataType::C128>::fn = zgeev_;

  SchurDecomposition<ffi::DataType::F32>::fn = sgees_;
  SchurDecomposition<ffi::DataType::F64>::fn = dgees_;
  SchurDecompositionComplex<ffi::DataType::C64>::fn = cgees_;
  SchurDecompositionComplex<ffi::DataType::C128>::fn = zgees_;

  HessenbergDecomposition<ffi::DataType::F32>::fn = sgehrd_;
  HessenbergDecomposition<ffi::DataType::F64>::fn = dgehrd_;
  HessenbergDecomposition<ffi::DataType::C64>::fn = cgehrd_;
  HessenbergDecomposition<ffi::DataType::C128>::fn = zgehrd_;

  TridiagonalReduction<ffi::DataType::F32>::fn = ssytrd_;
  TridiagonalReduction<ffi::DataType::F64>::fn = dsytrd_;
  TridiagonalReduction<ffi::DataType::C64>::fn = chetrd_;
  TridiagonalReduction<ffi::DataType::C128>::fn = zhetrd_;

  return 0;
}();

}  // namespace jax
