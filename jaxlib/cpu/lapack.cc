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

#include <complex>

#include "absl/base/call_once.h"
#include "nanobind/nanobind.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace {

namespace nb = nanobind;

using ::xla::ffi::DataType;

void GetLapackKernelsFromScipy() {
  static absl::once_flag initialized;
  // For reasons I'm not entirely sure of, if the import_ call is done inside
  // the call_once scope, we sometimes observe deadlocks in the test suite.
  // However it probably doesn't do much harm to just import them a second time,
  // since that costs little more than a dictionary lookup or two.
  nb::module_ cython_blas = nb::module_::import_("scipy.linalg.cython_blas");
  nb::module_ cython_lapack =
      nb::module_::import_("scipy.linalg.cython_lapack");
  absl::call_once(initialized, [&]() {
    // Technically this is a Cython-internal API. However, it seems highly
    // likely it will remain stable because Cython itself needs API stability
    // for cross-package imports to work in the first place.
    nb::dict blas_capi = cython_blas.attr("__pyx_capi__");
    auto blas_ptr = [&](const char* name) {
      return nb::cast<nb::capsule>(blas_capi[name]).data();
    };

    AssignKernelFn<Trsm<float>>(blas_ptr("strsm"));
    AssignKernelFn<Trsm<double>>(blas_ptr("dtrsm"));
    AssignKernelFn<Trsm<std::complex<float>>>(blas_ptr("ctrsm"));
    AssignKernelFn<Trsm<std::complex<double>>>(blas_ptr("ztrsm"));
    AssignKernelFn<TriMatrixEquationSolver<DataType::F32>>(blas_ptr("strsm"));
    AssignKernelFn<TriMatrixEquationSolver<DataType::F64>>(blas_ptr("dtrsm"));
    AssignKernelFn<TriMatrixEquationSolver<DataType::C64>>(blas_ptr("ctrsm"));
    AssignKernelFn<TriMatrixEquationSolver<DataType::C128>>(blas_ptr("ztrsm"));

    nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
    auto lapack_ptr = [&](const char* name) {
      return nb::cast<nb::capsule>(lapack_capi[name]).data();
    };
    AssignKernelFn<LuDecomposition<DataType::F32>>(lapack_ptr("sgetrf"));
    AssignKernelFn<LuDecomposition<DataType::F64>>(lapack_ptr("dgetrf"));
    AssignKernelFn<LuDecomposition<DataType::C64>>(lapack_ptr("cgetrf"));
    AssignKernelFn<LuDecomposition<DataType::C128>>(lapack_ptr("zgetrf"));

    AssignKernelFn<QrFactorization<DataType::F32>>(lapack_ptr("sgeqrf"));
    AssignKernelFn<QrFactorization<DataType::F64>>(lapack_ptr("dgeqrf"));
    AssignKernelFn<QrFactorization<DataType::C64>>(lapack_ptr("cgeqrf"));
    AssignKernelFn<QrFactorization<DataType::C128>>(lapack_ptr("zgeqrf"));

    AssignKernelFn<PivotingQrFactorization<DataType::F32>>(
        lapack_ptr("sgeqp3"));
    AssignKernelFn<PivotingQrFactorization<DataType::F64>>(
        lapack_ptr("dgeqp3"));
    AssignKernelFn<PivotingQrFactorization<DataType::C64>>(
        lapack_ptr("cgeqp3"));
    AssignKernelFn<PivotingQrFactorization<DataType::C128>>(
        lapack_ptr("zgeqp3"));

    AssignKernelFn<OrthogonalQr<DataType::F32>>(lapack_ptr("sorgqr"));
    AssignKernelFn<OrthogonalQr<DataType::F64>>(lapack_ptr("dorgqr"));
    AssignKernelFn<OrthogonalQr<DataType::C64>>(lapack_ptr("cungqr"));
    AssignKernelFn<OrthogonalQr<DataType::C128>>(lapack_ptr("zungqr"));

    AssignKernelFn<CholeskyFactorization<DataType::F32>>(lapack_ptr("spotrf"));
    AssignKernelFn<CholeskyFactorization<DataType::F64>>(lapack_ptr("dpotrf"));
    AssignKernelFn<CholeskyFactorization<DataType::C64>>(lapack_ptr("cpotrf"));
    AssignKernelFn<CholeskyFactorization<DataType::C128>>(lapack_ptr("zpotrf"));

    AssignKernelFn<svd::SVDType<DataType::F32>>(lapack_ptr("sgesdd"));
    AssignKernelFn<svd::SVDType<DataType::F64>>(lapack_ptr("dgesdd"));
    AssignKernelFn<svd::SVDType<DataType::C64>>(lapack_ptr("cgesdd"));
    AssignKernelFn<svd::SVDType<DataType::C128>>(lapack_ptr("zgesdd"));
    AssignKernelFn<svd::SVDQRType<DataType::F32>>(lapack_ptr("sgesvd"));
    AssignKernelFn<svd::SVDQRType<DataType::F64>>(lapack_ptr("dgesvd"));
    AssignKernelFn<svd::SVDQRType<DataType::C64>>(lapack_ptr("cgesvd"));
    AssignKernelFn<svd::SVDQRType<DataType::C128>>(lapack_ptr("zgesvd"));

    AssignKernelFn<EigenvalueDecompositionSymmetric<DataType::F32>>(
        lapack_ptr("ssyevd"));
    AssignKernelFn<EigenvalueDecompositionSymmetric<DataType::F64>>(
        lapack_ptr("dsyevd"));
    AssignKernelFn<EigenvalueDecompositionHermitian<DataType::C64>>(
        lapack_ptr("cheevd"));
    AssignKernelFn<EigenvalueDecompositionHermitian<DataType::C128>>(
        lapack_ptr("zheevd"));

    AssignKernelFn<EigenvalueDecomposition<DataType::F32>>(lapack_ptr("sgeev"));
    AssignKernelFn<EigenvalueDecomposition<DataType::F64>>(lapack_ptr("dgeev"));
    AssignKernelFn<EigenvalueDecompositionComplex<DataType::C64>>(
        lapack_ptr("cgeev"));
    AssignKernelFn<EigenvalueDecompositionComplex<DataType::C128>>(
        lapack_ptr("zgeev"));

    AssignKernelFn<RealGees<float>>(lapack_ptr("sgees"));
    AssignKernelFn<RealGees<double>>(lapack_ptr("dgees"));
    AssignKernelFn<ComplexGees<std::complex<float>>>(lapack_ptr("cgees"));
    AssignKernelFn<ComplexGees<std::complex<double>>>(lapack_ptr("zgees"));
    AssignKernelFn<SchurDecomposition<DataType::F32>>(lapack_ptr("sgees"));
    AssignKernelFn<SchurDecomposition<DataType::F64>>(lapack_ptr("dgees"));
    AssignKernelFn<SchurDecompositionComplex<DataType::C64>>(
        lapack_ptr("cgees"));
    AssignKernelFn<SchurDecompositionComplex<DataType::C128>>(
        lapack_ptr("zgees"));

    AssignKernelFn<HessenbergDecomposition<DataType::F32>>(
        lapack_ptr("sgehrd"));
    AssignKernelFn<HessenbergDecomposition<DataType::F64>>(
        lapack_ptr("dgehrd"));
    AssignKernelFn<HessenbergDecomposition<DataType::C64>>(
        lapack_ptr("cgehrd"));
    AssignKernelFn<HessenbergDecomposition<DataType::C128>>(
        lapack_ptr("zgehrd"));

    AssignKernelFn<Sytrd<float>>(lapack_ptr("ssytrd"));
    AssignKernelFn<Sytrd<double>>(lapack_ptr("dsytrd"));
    AssignKernelFn<Sytrd<std::complex<float>>>(lapack_ptr("chetrd"));
    AssignKernelFn<Sytrd<std::complex<double>>>(lapack_ptr("zhetrd"));
    AssignKernelFn<TridiagonalReduction<DataType::F32>>(lapack_ptr("ssytrd"));
    AssignKernelFn<TridiagonalReduction<DataType::F64>>(lapack_ptr("dsytrd"));
    AssignKernelFn<TridiagonalReduction<DataType::C64>>(lapack_ptr("chetrd"));
    AssignKernelFn<TridiagonalReduction<DataType::C128>>(lapack_ptr("zhetrd"));

    AssignKernelFn<TridiagonalSolver<DataType::F32>>(lapack_ptr("sgtsv"));
    AssignKernelFn<TridiagonalSolver<DataType::F64>>(lapack_ptr("dgtsv"));
    AssignKernelFn<TridiagonalSolver<DataType::C64>>(lapack_ptr("cgtsv"));
    AssignKernelFn<TridiagonalSolver<DataType::C128>>(lapack_ptr("zgtsv"));
  });
}

nb::dict Registrations() {
  nb::dict dict;
  dict["blas_strsm"] = EncapsulateFunction(Trsm<float>::Kernel);
  dict["blas_dtrsm"] = EncapsulateFunction(Trsm<double>::Kernel);
  dict["blas_ctrsm"] = EncapsulateFunction(Trsm<std::complex<float>>::Kernel);
  dict["blas_ztrsm"] = EncapsulateFunction(Trsm<std::complex<double>>::Kernel);
  dict["lapack_sgees"] = EncapsulateFunction(RealGees<float>::Kernel);
  dict["lapack_dgees"] = EncapsulateFunction(RealGees<double>::Kernel);
  dict["lapack_cgees"] =
      EncapsulateFunction(ComplexGees<std::complex<float>>::Kernel);
  dict["lapack_zgees"] =
      EncapsulateFunction(ComplexGees<std::complex<double>>::Kernel);
  dict["lapack_ssytrd"] = EncapsulateFunction(Sytrd<float>::Kernel);
  dict["lapack_dsytrd"] = EncapsulateFunction(Sytrd<double>::Kernel);
  dict["lapack_chetrd"] =
      EncapsulateFunction(Sytrd<std::complex<float>>::Kernel);
  dict["lapack_zhetrd"] =
      EncapsulateFunction(Sytrd<std::complex<double>>::Kernel);

  dict["lapack_strsm_ffi"] = EncapsulateFunction(lapack_strsm_ffi);
  dict["lapack_dtrsm_ffi"] = EncapsulateFunction(lapack_dtrsm_ffi);
  dict["lapack_ctrsm_ffi"] = EncapsulateFunction(lapack_ctrsm_ffi);
  dict["lapack_ztrsm_ffi"] = EncapsulateFunction(lapack_ztrsm_ffi);
  dict["lapack_sgetrf_ffi"] = EncapsulateFunction(lapack_sgetrf_ffi);
  dict["lapack_dgetrf_ffi"] = EncapsulateFunction(lapack_dgetrf_ffi);
  dict["lapack_cgetrf_ffi"] = EncapsulateFunction(lapack_cgetrf_ffi);
  dict["lapack_zgetrf_ffi"] = EncapsulateFunction(lapack_zgetrf_ffi);
  dict["lapack_sgeqrf_ffi"] = EncapsulateFunction(lapack_sgeqrf_ffi);
  dict["lapack_dgeqrf_ffi"] = EncapsulateFunction(lapack_dgeqrf_ffi);
  dict["lapack_cgeqrf_ffi"] = EncapsulateFunction(lapack_cgeqrf_ffi);
  dict["lapack_zgeqrf_ffi"] = EncapsulateFunction(lapack_zgeqrf_ffi);
  dict["lapack_sgeqp3_ffi"] = EncapsulateFunction(lapack_sgeqp3_ffi);
  dict["lapack_dgeqp3_ffi"] = EncapsulateFunction(lapack_dgeqp3_ffi);
  dict["lapack_cgeqp3_ffi"] = EncapsulateFunction(lapack_cgeqp3_ffi);
  dict["lapack_zgeqp3_ffi"] = EncapsulateFunction(lapack_zgeqp3_ffi);
  dict["lapack_sorgqr_ffi"] = EncapsulateFunction(lapack_sorgqr_ffi);
  dict["lapack_dorgqr_ffi"] = EncapsulateFunction(lapack_dorgqr_ffi);
  dict["lapack_cungqr_ffi"] = EncapsulateFunction(lapack_cungqr_ffi);
  dict["lapack_zungqr_ffi"] = EncapsulateFunction(lapack_zungqr_ffi);
  dict["lapack_spotrf_ffi"] = EncapsulateFunction(lapack_spotrf_ffi);
  dict["lapack_dpotrf_ffi"] = EncapsulateFunction(lapack_dpotrf_ffi);
  dict["lapack_cpotrf_ffi"] = EncapsulateFunction(lapack_cpotrf_ffi);
  dict["lapack_zpotrf_ffi"] = EncapsulateFunction(lapack_zpotrf_ffi);
  dict["lapack_sgesdd_ffi"] = EncapsulateFunction(lapack_sgesdd_ffi);
  dict["lapack_dgesdd_ffi"] = EncapsulateFunction(lapack_dgesdd_ffi);
  dict["lapack_cgesdd_ffi"] = EncapsulateFunction(lapack_cgesdd_ffi);
  dict["lapack_zgesdd_ffi"] = EncapsulateFunction(lapack_zgesdd_ffi);
  dict["lapack_sgesvd_ffi"] = EncapsulateFunction(lapack_sgesvd_ffi);
  dict["lapack_dgesvd_ffi"] = EncapsulateFunction(lapack_dgesvd_ffi);
  dict["lapack_cgesvd_ffi"] = EncapsulateFunction(lapack_cgesvd_ffi);
  dict["lapack_zgesvd_ffi"] = EncapsulateFunction(lapack_zgesvd_ffi);
  dict["lapack_ssyevd_ffi"] = EncapsulateFunction(lapack_ssyevd_ffi);
  dict["lapack_dsyevd_ffi"] = EncapsulateFunction(lapack_dsyevd_ffi);
  dict["lapack_cheevd_ffi"] = EncapsulateFunction(lapack_cheevd_ffi);
  dict["lapack_zheevd_ffi"] = EncapsulateFunction(lapack_zheevd_ffi);
  dict["lapack_sgeev_ffi"] = EncapsulateFunction(lapack_sgeev_ffi);
  dict["lapack_dgeev_ffi"] = EncapsulateFunction(lapack_dgeev_ffi);
  dict["lapack_cgeev_ffi"] = EncapsulateFunction(lapack_cgeev_ffi);
  dict["lapack_zgeev_ffi"] = EncapsulateFunction(lapack_zgeev_ffi);
  dict["lapack_ssytrd_ffi"] = EncapsulateFunction(lapack_ssytrd_ffi);
  dict["lapack_dsytrd_ffi"] = EncapsulateFunction(lapack_dsytrd_ffi);
  dict["lapack_chetrd_ffi"] = EncapsulateFunction(lapack_chetrd_ffi);
  dict["lapack_zhetrd_ffi"] = EncapsulateFunction(lapack_zhetrd_ffi);
  dict["lapack_sgees_ffi"] = EncapsulateFunction(lapack_sgees_ffi);
  dict["lapack_dgees_ffi"] = EncapsulateFunction(lapack_dgees_ffi);
  dict["lapack_cgees_ffi"] = EncapsulateFunction(lapack_cgees_ffi);
  dict["lapack_zgees_ffi"] = EncapsulateFunction(lapack_zgees_ffi);
  dict["lapack_sgehrd_ffi"] = EncapsulateFunction(lapack_sgehrd_ffi);
  dict["lapack_dgehrd_ffi"] = EncapsulateFunction(lapack_dgehrd_ffi);
  dict["lapack_cgehrd_ffi"] = EncapsulateFunction(lapack_cgehrd_ffi);
  dict["lapack_zgehrd_ffi"] = EncapsulateFunction(lapack_zgehrd_ffi);
  dict["lapack_sgtsv_ffi"] = EncapsulateFunction(lapack_sgtsv_ffi);
  dict["lapack_dgtsv_ffi"] = EncapsulateFunction(lapack_dgtsv_ffi);
  dict["lapack_cgtsv_ffi"] = EncapsulateFunction(lapack_cgtsv_ffi);
  dict["lapack_zgtsv_ffi"] = EncapsulateFunction(lapack_zgtsv_ffi);

  return dict;
}

NB_MODULE(_lapack, m) {
  // Populates the LAPACK kernels from scipy on first call.
  m.def("initialize", GetLapackKernelsFromScipy);
  m.def("registrations", &Registrations);
  // Submodules
  auto svd = m.def_submodule("svd");
  auto eig = m.def_submodule("eig");
  auto schur = m.def_submodule("schur");
  // Enums
  nb::enum_<svd::ComputationMode>(svd, "ComputationMode")
      // kComputeVtOverwriteXPartialU is not implemented
      .value("kComputeFullUVt", svd::ComputationMode::kComputeFullUVt)
      .value("kComputeMinUVt", svd::ComputationMode::kComputeMinUVt)
      .value("kNoComputeUVt", svd::ComputationMode::kNoComputeUVt);
  nb::enum_<eig::ComputationMode>(eig, "ComputationMode")
      .value("kComputeEigenvectors", eig::ComputationMode::kComputeEigenvectors)
      .value("kNoEigenvectors", eig::ComputationMode::kNoEigenvectors);
  nb::enum_<schur::ComputationMode>(schur, "ComputationMode")
      .value("kNoComputeSchurVectors",
             schur::ComputationMode::kNoComputeSchurVectors)
      .value("kComputeSchurVectors",
             schur::ComputationMode::kComputeSchurVectors);
  nb::enum_<schur::Sort>(schur, "Sort")
      .value("kNoSortEigenvalues", schur::Sort::kNoSortEigenvalues)
      .value("kSortEigenvalues", schur::Sort::kSortEigenvalues);
}

}  // namespace
}  // namespace jax
