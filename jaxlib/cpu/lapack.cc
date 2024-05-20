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

#include "nanobind/nanobind.h"
#include "jaxlib/cpu/cpu_kernels.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace {

namespace nb = nanobind;

using ::xla::ffi::DataType;

svd::ComputationMode GetSvdComputationMode(bool job_opt_compute_uv,
                                           bool job_opt_full_matrices) {
  if (!job_opt_compute_uv) {
    return svd::ComputationMode::kNoComputeUVt;
  } else if (!job_opt_full_matrices) {
    return svd::ComputationMode::kComputeMinUVt;
  }
  return svd::ComputationMode::kComputeFullUVt;
}

template <DataType dtype>
int64_t GesddGetWorkspaceSize(lapack_int m, lapack_int n,
                              bool job_opt_compute_uv,
                              bool job_opt_full_matrices) {
  svd::ComputationMode mode =
      GetSvdComputationMode(job_opt_compute_uv, job_opt_full_matrices);
  return svd::SVDType<dtype>::GetWorkspaceSize(m, n, mode);
};

lapack_int GesddGetRealWorkspaceSize(lapack_int m, lapack_int n,
                                     bool job_opt_compute_uv) {
  svd::ComputationMode mode = GetSvdComputationMode(job_opt_compute_uv, true);
  return svd::GetRealWorkspaceSize(m, n, mode);
}

// TODO(paruzelp): For some reason JAX prefers to assume a larger workspace
//                 Might need to investigate if that is necessary.
template <lapack_int (&f)(int64_t, eig::ComputationMode)>
inline constexpr auto BoundWithEigvecs = +[](lapack_int n) {
  return f(n, eig::ComputationMode::kComputeEigenvectors);
};

void GetLapackKernelsFromScipy() {
  static bool initialized = false;  // Protected by GIL
  if (initialized) return;
  nb::module_ cython_blas = nb::module_::import_("scipy.linalg.cython_blas");
  // Technically this is a Cython-internal API. However, it seems highly likely
  // it will remain stable because Cython itself needs API stability for
  // cross-package imports to work in the first place.
  nb::dict blas_capi = cython_blas.attr("__pyx_capi__");
  auto blas_ptr = [&](const char* name) {
    return nb::cast<nb::capsule>(blas_capi[name]).data();
  };

  AssignKernelFn<TriMatrixEquationSolver<DataType::F32>>(blas_ptr("strsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::F64>>(blas_ptr("dtrsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::C64>>(blas_ptr("ctrsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::C128>>(blas_ptr("ztrsm"));

  nb::module_ cython_lapack =
      nb::module_::import_("scipy.linalg.cython_lapack");
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

  AssignKernelFn<SchurDecomposition<DataType::F32>>(lapack_ptr("sgees"));
  AssignKernelFn<SchurDecomposition<DataType::F64>>(lapack_ptr("dgees"));
  AssignKernelFn<SchurDecompositionComplex<DataType::C64>>(lapack_ptr("cgees"));
  AssignKernelFn<SchurDecompositionComplex<DataType::C128>>(
      lapack_ptr("zgees"));

  AssignKernelFn<HessenbergDecomposition<DataType::F32>>(lapack_ptr("sgehrd"));
  AssignKernelFn<HessenbergDecomposition<DataType::F64>>(lapack_ptr("dgehrd"));
  AssignKernelFn<HessenbergDecomposition<DataType::C64>>(lapack_ptr("cgehrd"));
  AssignKernelFn<HessenbergDecomposition<DataType::C128>>(lapack_ptr("zgehrd"));

  AssignKernelFn<TridiagonalReduction<DataType::F32>>(lapack_ptr("ssytrd"));
  AssignKernelFn<TridiagonalReduction<DataType::F64>>(lapack_ptr("dsytrd"));
  AssignKernelFn<TridiagonalReduction<DataType::C64>>(lapack_ptr("chetrd"));
  AssignKernelFn<TridiagonalReduction<DataType::C128>>(lapack_ptr("zhetrd"));

  initialized = true;
}

nb::dict Registrations() {
  nb::dict dict;

  dict["blas_strsm"] = EncapsulateFunction(blas_strsm);
  dict["blas_dtrsm"] = EncapsulateFunction(blas_dtrsm);
  dict["blas_ctrsm"] = EncapsulateFunction(blas_ctrsm);
  dict["blas_ztrsm"] = EncapsulateFunction(blas_ztrsm);
  dict["lapack_sgetrf"] = EncapsulateFunction(lapack_sgetrf);
  dict["lapack_dgetrf"] = EncapsulateFunction(lapack_dgetrf);
  dict["lapack_cgetrf"] = EncapsulateFunction(lapack_cgetrf);
  dict["lapack_zgetrf"] = EncapsulateFunction(lapack_zgetrf);
  dict["lapack_sgeqrf"] = EncapsulateFunction(lapack_sgeqrf);
  dict["lapack_dgeqrf"] = EncapsulateFunction(lapack_dgeqrf);
  dict["lapack_cgeqrf"] = EncapsulateFunction(lapack_cgeqrf);
  dict["lapack_zgeqrf"] = EncapsulateFunction(lapack_zgeqrf);
  dict["lapack_sorgqr"] = EncapsulateFunction(lapack_sorgqr);
  dict["lapack_dorgqr"] = EncapsulateFunction(lapack_dorgqr);
  dict["lapack_cungqr"] = EncapsulateFunction(lapack_cungqr);
  dict["lapack_zungqr"] = EncapsulateFunction(lapack_zungqr);
  dict["lapack_spotrf"] = EncapsulateFunction(lapack_spotrf);
  dict["lapack_dpotrf"] = EncapsulateFunction(lapack_dpotrf);
  dict["lapack_cpotrf"] = EncapsulateFunction(lapack_cpotrf);
  dict["lapack_zpotrf"] = EncapsulateFunction(lapack_zpotrf);
  dict["lapack_sgesdd"] = EncapsulateFunction(lapack_sgesdd);
  dict["lapack_dgesdd"] = EncapsulateFunction(lapack_dgesdd);
  dict["lapack_cgesdd"] = EncapsulateFunction(lapack_cgesdd);
  dict["lapack_zgesdd"] = EncapsulateFunction(lapack_zgesdd);
  dict["lapack_ssyevd"] = EncapsulateFunction(lapack_ssyevd);
  dict["lapack_dsyevd"] = EncapsulateFunction(lapack_dsyevd);
  dict["lapack_cheevd"] = EncapsulateFunction(lapack_cheevd);
  dict["lapack_zheevd"] = EncapsulateFunction(lapack_zheevd);
  dict["lapack_sgeev"] = EncapsulateFunction(lapack_sgeev);
  dict["lapack_dgeev"] = EncapsulateFunction(lapack_dgeev);
  dict["lapack_cgeev"] = EncapsulateFunction(lapack_cgeev);
  dict["lapack_zgeev"] = EncapsulateFunction(lapack_zgeev);
  dict["lapack_sgees"] = EncapsulateFunction(lapack_sgees);
  dict["lapack_dgees"] = EncapsulateFunction(lapack_dgees);
  dict["lapack_cgees"] = EncapsulateFunction(lapack_cgees);
  dict["lapack_zgees"] = EncapsulateFunction(lapack_zgees);
  dict["lapack_sgehrd"] = EncapsulateFunction(lapack_sgehrd);
  dict["lapack_dgehrd"] = EncapsulateFunction(lapack_dgehrd);
  dict["lapack_cgehrd"] = EncapsulateFunction(lapack_cgehrd);
  dict["lapack_zgehrd"] = EncapsulateFunction(lapack_zgehrd);
  dict["lapack_ssytrd"] = EncapsulateFunction(lapack_ssytrd);
  dict["lapack_dsytrd"] = EncapsulateFunction(lapack_dsytrd);
  dict["lapack_chetrd"] = EncapsulateFunction(lapack_chetrd);
  dict["lapack_zhetrd"] = EncapsulateFunction(lapack_zhetrd);

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

  m.def("lapack_sgeqrf_workspace",
        &QrFactorization<DataType::F32>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_dgeqrf_workspace",
        &QrFactorization<DataType::F64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_cgeqrf_workspace",
        &QrFactorization<DataType::C64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_zgeqrf_workspace",
        &QrFactorization<DataType::C128>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_sorgqr_workspace",
        &OrthogonalQr<DataType::F32>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_dorgqr_workspace",
        &OrthogonalQr<DataType::F64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_cungqr_workspace",
        &OrthogonalQr<DataType::C64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_zungqr_workspace",
        &OrthogonalQr<DataType::C128>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("gesdd_iwork_size", &svd::GetIntWorkspaceSize, nb::arg("m"),
        nb::arg("n"));
  m.def("sgesdd_work_size", &svd::SVDType<DataType::F32>::GetWorkspaceSize,
        nb::arg("m"), nb::arg("n"), nb::arg("mode"));
  m.def("dgesdd_work_size", &svd::SVDType<DataType::F64>::GetWorkspaceSize,
        nb::arg("m"), nb::arg("n"), nb::arg("mode"));
  m.def("gesdd_rwork_size", &svd::GetRealWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("mode"));
  m.def("cgesdd_work_size", &svd::SVDType<DataType::C64>::GetWorkspaceSize,
        nb::arg("m"), nb::arg("n"), nb::arg("mode"));
  m.def("zgesdd_work_size", &svd::SVDType<DataType::C128>::GetWorkspaceSize,
        nb::arg("m"), nb::arg("n"), nb::arg("mode"));
  m.def("syevd_work_size", BoundWithEigvecs<eig::GetWorkspaceSize>,
        nb::arg("n"));
  m.def("syevd_iwork_size", BoundWithEigvecs<eig::GetIntWorkspaceSize>,
        nb::arg("n"));
  m.def("heevd_work_size", BoundWithEigvecs<eig::GetComplexWorkspaceSize>,
        nb::arg("n"));
  m.def("heevd_rwork_size", BoundWithEigvecs<eig::GetRealWorkspaceSize>,
        nb::arg("n"));

  m.def("lapack_sgehrd_workspace",
        &HessenbergDecomposition<DataType::F32>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_dgehrd_workspace",
        &HessenbergDecomposition<DataType::F64>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_cgehrd_workspace",
        &HessenbergDecomposition<DataType::C64>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_zgehrd_workspace",
        &HessenbergDecomposition<DataType::C128>::GetWorkspaceSize,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_ssytrd_workspace",
        &TridiagonalReduction<DataType::F32>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_dsytrd_workspace",
        &TridiagonalReduction<DataType::F64>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_chetrd_workspace",
        &TridiagonalReduction<DataType::C64>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_zhetrd_workspace",
        &TridiagonalReduction<DataType::C128>::GetWorkspaceSize, nb::arg("lda"),
        nb::arg("n"));
}

}  // namespace
}  // namespace jax
