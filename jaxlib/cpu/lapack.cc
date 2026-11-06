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
#include <memory>

#include "nanobind/nanobind.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/cpu/tridiagonal_solve_kernels.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace {

namespace nb = nanobind;

void Initialize() {}

nb::dict Registrations() {
  nb::dict dict;
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
  dict["lapack_sormqr_ffi"] = EncapsulateFunction(lapack_sormqr_ffi);
  dict["lapack_dormqr_ffi"] = EncapsulateFunction(lapack_dormqr_ffi);
  dict["lapack_cunmqr_ffi"] = EncapsulateFunction(lapack_cunmqr_ffi);
  dict["lapack_zunmqr_ffi"] = EncapsulateFunction(lapack_zunmqr_ffi);
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

  dict["tridiagonal_solve_perturbed_ffi"] =
      EncapsulateFunction(tridiagonal_solve_perturbed_ffi);

  return dict;
}

NB_MODULE(_lapack, m) {
  m.def("initialize", &Initialize);
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
