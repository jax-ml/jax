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

#ifdef JAX_LAPACK_ILP64
using LapackInt = int64_t;
#ifndef JAX_LAPACK_FUNC_SUFFIX
#define JAX_LAPACK_FUNC_SUFFIX _64_
#endif
#else
using LapackInt = int32_t;
#ifndef JAX_LAPACK_FUNC_SUFFIX
#define JAX_LAPACK_FUNC_SUFFIX _
#endif
#endif

#define JAX_LAPACK_CONCAT_IMPL(a, b) a##b
#define JAX_LAPACK_CONCAT(a, b) JAX_LAPACK_CONCAT_IMPL(a, b)
#define JAX_LAPACK_FN(name) JAX_LAPACK_CONCAT(name, JAX_LAPACK_FUNC_SUFFIX)

extern "C" {

jax::TriMatrixEquationSolver<ffi::DataType::F32, LapackInt>::FnType
    JAX_LAPACK_FN(strsm);
jax::TriMatrixEquationSolver<ffi::DataType::F64, LapackInt>::FnType
    JAX_LAPACK_FN(dtrsm);
jax::TriMatrixEquationSolver<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(ctrsm);
jax::TriMatrixEquationSolver<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(ztrsm);

jax::LuDecomposition<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(
    sgetrf);
jax::LuDecomposition<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(
    dgetrf);
jax::LuDecomposition<ffi::DataType::C64, LapackInt>::FnType JAX_LAPACK_FN(
    cgetrf);
jax::LuDecomposition<ffi::DataType::C128, LapackInt>::FnType JAX_LAPACK_FN(
    zgetrf);

jax::QrFactorization<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(
    sgeqrf);
jax::QrFactorization<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(
    dgeqrf);
jax::QrFactorization<ffi::DataType::C64, LapackInt>::FnType JAX_LAPACK_FN(
    cgeqrf);
jax::QrFactorization<ffi::DataType::C128, LapackInt>::FnType JAX_LAPACK_FN(
    zgeqrf);

jax::PivotingQrFactorization<ffi::DataType::F32, LapackInt>::FnType
    JAX_LAPACK_FN(sgeqp3);
jax::PivotingQrFactorization<ffi::DataType::F64, LapackInt>::FnType
    JAX_LAPACK_FN(dgeqp3);
jax::PivotingQrFactorization<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(cgeqp3);
jax::PivotingQrFactorization<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zgeqp3);

jax::OrthogonalQr<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(sorgqr);
jax::OrthogonalQr<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(dorgqr);
jax::OrthogonalQr<ffi::DataType::C64, LapackInt>::FnType JAX_LAPACK_FN(cungqr);
jax::OrthogonalQr<ffi::DataType::C128, LapackInt>::FnType JAX_LAPACK_FN(zungqr);

jax::OrthogonalQrMultiply<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(
    sormqr);
jax::OrthogonalQrMultiply<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(
    dormqr);
jax::OrthogonalQrMultiply<ffi::DataType::C64, LapackInt>::FnType JAX_LAPACK_FN(
    cunmqr);
jax::OrthogonalQrMultiply<ffi::DataType::C128, LapackInt>::FnType JAX_LAPACK_FN(
    zunmqr);

jax::CholeskyFactorization<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(
    spotrf);
jax::CholeskyFactorization<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(
    dpotrf);
jax::CholeskyFactorization<ffi::DataType::C64, LapackInt>::FnType JAX_LAPACK_FN(
    cpotrf);
jax::CholeskyFactorization<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zpotrf);

jax::SingularValueDecomposition<ffi::DataType::F32, LapackInt>::FnType
    JAX_LAPACK_FN(sgesdd);
jax::SingularValueDecomposition<ffi::DataType::F64, LapackInt>::FnType
    JAX_LAPACK_FN(dgesdd);
jax::SingularValueDecompositionComplex<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(cgesdd);
jax::SingularValueDecompositionComplex<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zgesdd);

jax::SingularValueDecompositionQR<ffi::DataType::F32, LapackInt>::FnType
    JAX_LAPACK_FN(sgesvd);
jax::SingularValueDecompositionQR<ffi::DataType::F64, LapackInt>::FnType
    JAX_LAPACK_FN(dgesvd);
jax::SingularValueDecompositionQRComplex<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(cgesvd);
jax::SingularValueDecompositionQRComplex<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zgesvd);

jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32, LapackInt>::FnType
    JAX_LAPACK_FN(ssyevd);
jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64, LapackInt>::FnType
    JAX_LAPACK_FN(dsyevd);
jax::EigenvalueDecompositionHermitian<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(cheevd);
jax::EigenvalueDecompositionHermitian<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zheevd);

jax::EigenvalueDecomposition<ffi::DataType::F32, LapackInt>::FnType
    JAX_LAPACK_FN(sgeev);
jax::EigenvalueDecomposition<ffi::DataType::F64, LapackInt>::FnType
    JAX_LAPACK_FN(dgeev);
jax::EigenvalueDecompositionComplex<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(cgeev);
jax::EigenvalueDecompositionComplex<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zgeev);

jax::SchurDecomposition<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(
    sgees);
jax::SchurDecomposition<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(
    dgees);
jax::SchurDecompositionComplex<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(cgees);
jax::SchurDecompositionComplex<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zgees);

jax::HessenbergDecomposition<ffi::DataType::F32, LapackInt>::FnType
    JAX_LAPACK_FN(sgehrd);
jax::HessenbergDecomposition<ffi::DataType::F64, LapackInt>::FnType
    JAX_LAPACK_FN(dgehrd);
jax::HessenbergDecomposition<ffi::DataType::C64, LapackInt>::FnType
    JAX_LAPACK_FN(cgehrd);
jax::HessenbergDecomposition<ffi::DataType::C128, LapackInt>::FnType
    JAX_LAPACK_FN(zgehrd);

jax::TridiagonalReduction<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(
    ssytrd);
jax::TridiagonalReduction<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(
    dsytrd);
jax::TridiagonalReduction<ffi::DataType::C64, LapackInt>::FnType JAX_LAPACK_FN(
    chetrd);
jax::TridiagonalReduction<ffi::DataType::C128, LapackInt>::FnType JAX_LAPACK_FN(
    zhetrd);

jax::TridiagonalSolver<ffi::DataType::F32, LapackInt>::FnType JAX_LAPACK_FN(
    sgtsv);
jax::TridiagonalSolver<ffi::DataType::F64, LapackInt>::FnType JAX_LAPACK_FN(
    dgtsv);
jax::TridiagonalSolver<ffi::DataType::C64, LapackInt>::FnType JAX_LAPACK_FN(
    cgtsv);
jax::TridiagonalSolver<ffi::DataType::C128, LapackInt>::FnType JAX_LAPACK_FN(
    zgtsv);

}  // extern "C"

namespace jax {

static auto init = []() -> int {
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(strsm));
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dtrsm));
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(ctrsm));
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(ztrsm));

  AssignKernelFn<LuDecomposition<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgetrf));
  AssignKernelFn<LuDecomposition<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgetrf));
  AssignKernelFn<LuDecomposition<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgetrf));
  AssignKernelFn<LuDecomposition<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgetrf));

  AssignKernelFn<QrFactorization<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgeqrf));
  AssignKernelFn<QrFactorization<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgeqrf));
  AssignKernelFn<QrFactorization<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgeqrf));
  AssignKernelFn<QrFactorization<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgeqrf));

  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgeqp3));
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgeqp3));
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgeqp3));
  AssignKernelFn<PivotingQrFactorization<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgeqp3));

  AssignKernelFn<OrthogonalQr<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sorgqr));
  AssignKernelFn<OrthogonalQr<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dorgqr));
  AssignKernelFn<OrthogonalQr<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cungqr));
  AssignKernelFn<OrthogonalQr<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zungqr));

  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sormqr));
  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dormqr));
  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cunmqr));
  AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zunmqr));

  AssignKernelFn<CholeskyFactorization<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(spotrf));
  AssignKernelFn<CholeskyFactorization<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dpotrf));
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cpotrf));
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zpotrf));

  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgesdd));
  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgesdd));
  AssignKernelFn<
      SingularValueDecompositionComplex<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgesdd));
  AssignKernelFn<
      SingularValueDecompositionComplex<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgesdd));

  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgesvd));
  AssignKernelFn<SingularValueDecompositionQR<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgesvd));
  AssignKernelFn<
      SingularValueDecompositionQRComplex<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgesvd));
  AssignKernelFn<
      SingularValueDecompositionQRComplex<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgesvd));

  AssignKernelFn<
      EigenvalueDecompositionSymmetric<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(ssyevd));
  AssignKernelFn<
      EigenvalueDecompositionSymmetric<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dsyevd));
  AssignKernelFn<
      EigenvalueDecompositionHermitian<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cheevd));
  AssignKernelFn<
      EigenvalueDecompositionHermitian<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zheevd));

  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgeev));
  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgeev));
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgeev));
  AssignKernelFn<
      EigenvalueDecompositionComplex<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgeev));

  AssignKernelFn<TridiagonalReduction<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(ssytrd));
  AssignKernelFn<TridiagonalReduction<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dsytrd));
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(chetrd));
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zhetrd));

  AssignKernelFn<SchurDecomposition<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgees));
  AssignKernelFn<SchurDecomposition<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgees));
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgees));
  AssignKernelFn<SchurDecompositionComplex<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgees));

  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgehrd));
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgehrd));
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgehrd));
  AssignKernelFn<HessenbergDecomposition<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgehrd));

  AssignKernelFn<TridiagonalSolver<ffi::DataType::F32, LapackInt>>(
      JAX_LAPACK_FN(sgtsv));
  AssignKernelFn<TridiagonalSolver<ffi::DataType::F64, LapackInt>>(
      JAX_LAPACK_FN(dgtsv));
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C64, LapackInt>>(
      JAX_LAPACK_FN(cgtsv));
  AssignKernelFn<TridiagonalSolver<ffi::DataType::C128, LapackInt>>(
      JAX_LAPACK_FN(zgtsv));

  lapack_kernels_initialized = true;
  return 0;
}();

#undef JAX_LAPACK_FN
#undef JAX_LAPACK_CONCAT
#undef JAX_LAPACK_CONCAT_IMPL

}  // namespace jax
