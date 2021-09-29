/* Copyright 2021 Google LLC

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

#include "jaxlib/kernel_pybind11_helpers.h"
#include "jaxlib/lapack_kernels.h"
#include "include/pybind11/pybind11.h"

namespace jax {
namespace {

namespace py = pybind11;

void GetLapackKernelsFromScipy() {
  py::module cython_blas = py::module::import("scipy.linalg.cython_blas");
  // Technically this is a Cython-internal API. However, it seems highly likely
  // it will remain stable because Cython itself needs API stability for
  // cross-package imports to work in the first place.
  py::dict blas_capi = cython_blas.attr("__pyx_capi__");
  auto blas_ptr = [&](const char* name) {
    return py::capsule(blas_capi[name]).get_pointer();
  };
  Trsm<float>::fn = reinterpret_cast<Trsm<float>::FnType*>(blas_ptr("strsm"));
  Trsm<double>::fn = reinterpret_cast<Trsm<double>::FnType*>(blas_ptr("dtrsm"));
  Trsm<std::complex<float>>::fn =
      reinterpret_cast<Trsm<std::complex<float>>::FnType*>(blas_ptr("ctrsm"));
  Trsm<std::complex<double>>::fn =
      reinterpret_cast<Trsm<std::complex<double>>::FnType*>(blas_ptr("ztrsm"));

  py::module cython_lapack = py::module::import("scipy.linalg.cython_lapack");
  py::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
  auto lapack_ptr = [&](const char* name) {
    return py::capsule(lapack_capi[name]).get_pointer();
  };
  Getrf<float>::fn =
      reinterpret_cast<Getrf<float>::FnType*>(lapack_ptr("sgetrf"));
  Getrf<double>::fn =
      reinterpret_cast<Getrf<double>::FnType*>(lapack_ptr("dgetrf"));
  Getrf<std::complex<float>>::fn =
      reinterpret_cast<Getrf<std::complex<float>>::FnType*>(
          lapack_ptr("cgetrf"));
  Getrf<std::complex<double>>::fn =
      reinterpret_cast<Getrf<std::complex<double>>::FnType*>(
          lapack_ptr("zgetrf"));
  Geqrf<float>::fn =
      reinterpret_cast<Geqrf<float>::FnType*>(lapack_ptr("sgeqrf"));
  Geqrf<double>::fn =
      reinterpret_cast<Geqrf<double>::FnType*>(lapack_ptr("dgeqrf"));
  Geqrf<std::complex<float>>::fn =
      reinterpret_cast<Geqrf<std::complex<float>>::FnType*>(
          lapack_ptr("cgeqrf"));
  Geqrf<std::complex<double>>::fn =
      reinterpret_cast<Geqrf<std::complex<double>>::FnType*>(
          lapack_ptr("zgeqrf"));
  Orgqr<float>::fn =
      reinterpret_cast<Orgqr<float>::FnType*>(lapack_ptr("sorgqr"));
  Orgqr<double>::fn =
      reinterpret_cast<Orgqr<double>::FnType*>(lapack_ptr("dorgqr"));
  Orgqr<std::complex<float>>::fn =
      reinterpret_cast<Orgqr<std::complex<float>>::FnType*>(
          lapack_ptr("cungqr"));
  Orgqr<std::complex<double>>::fn =
      reinterpret_cast<Orgqr<std::complex<double>>::FnType*>(
          lapack_ptr("zungqr"));
  Potrf<float>::fn =
      reinterpret_cast<Potrf<float>::FnType*>(lapack_ptr("spotrf"));
  Potrf<double>::fn =
      reinterpret_cast<Potrf<double>::FnType*>(lapack_ptr("dpotrf"));
  Potrf<std::complex<float>>::fn =
      reinterpret_cast<Potrf<std::complex<float>>::FnType*>(
          lapack_ptr("cpotrf"));
  Potrf<std::complex<double>>::fn =
      reinterpret_cast<Potrf<std::complex<double>>::FnType*>(
          lapack_ptr("zpotrf"));
  RealGesdd<float>::fn =
      reinterpret_cast<RealGesdd<float>::FnType*>(lapack_ptr("sgesdd"));
  RealGesdd<double>::fn =
      reinterpret_cast<RealGesdd<double>::FnType*>(lapack_ptr("dgesdd"));
  ComplexGesdd<std::complex<float>>::fn =
      reinterpret_cast<ComplexGesdd<std::complex<float>>::FnType*>(
          lapack_ptr("cgesdd"));
  ComplexGesdd<std::complex<double>>::fn =
      reinterpret_cast<ComplexGesdd<std::complex<double>>::FnType*>(
          lapack_ptr("zgesdd"));
  RealSyevd<float>::fn =
      reinterpret_cast<RealSyevd<float>::FnType*>(lapack_ptr("ssyevd"));
  RealSyevd<double>::fn =
      reinterpret_cast<RealSyevd<double>::FnType*>(lapack_ptr("dsyevd"));
  ComplexHeevd<std::complex<float>>::fn =
      reinterpret_cast<ComplexHeevd<std::complex<float>>::FnType*>(
          lapack_ptr("cheevd"));
  ComplexHeevd<std::complex<double>>::fn =
      reinterpret_cast<ComplexHeevd<std::complex<double>>::FnType*>(
          lapack_ptr("zheevd"));
  RealGeev<float>::fn =
      reinterpret_cast<RealGeev<float>::FnType*>(lapack_ptr("sgeev"));
  RealGeev<double>::fn =
      reinterpret_cast<RealGeev<double>::FnType*>(lapack_ptr("dgeev"));
  ComplexGeev<std::complex<float>>::fn =
      reinterpret_cast<ComplexGeev<std::complex<float>>::FnType*>(
          lapack_ptr("cgeev"));
  ComplexGeev<std::complex<double>>::fn =
      reinterpret_cast<ComplexGeev<std::complex<double>>::FnType*>(
          lapack_ptr("zgeev"));
  RealGees<float>::fn =
      reinterpret_cast<RealGees<float>::FnType*>(lapack_ptr("sgees"));
  RealGees<double>::fn =
      reinterpret_cast<RealGees<double>::FnType*>(lapack_ptr("dgees"));
  ComplexGees<std::complex<float>>::fn =
      reinterpret_cast<ComplexGees<std::complex<float>>::FnType*>(
          lapack_ptr("cgees"));
  ComplexGees<std::complex<double>>::fn =
      reinterpret_cast<ComplexGees<std::complex<double>>::FnType*>(
          lapack_ptr("zgees"));
}

py::dict Registrations() {
  py::dict dict;
  dict["blas_strsm"] = EncapsulateFunction(Trsm<float>::Kernel);
  dict["blas_dtrsm"] = EncapsulateFunction(Trsm<double>::Kernel);
  dict["blas_ctrsm"] = EncapsulateFunction(Trsm<std::complex<float>>::Kernel);
  dict["blas_ztrsm"] = EncapsulateFunction(Trsm<std::complex<double>>::Kernel);
  dict["lapack_sgetrf"] = EncapsulateFunction(Getrf<float>::Kernel);
  dict["lapack_dgetrf"] = EncapsulateFunction(Getrf<double>::Kernel);
  dict["lapack_cgetrf"] =
      EncapsulateFunction(Getrf<std::complex<float>>::Kernel);
  dict["lapack_zgetrf"] =
      EncapsulateFunction(Getrf<std::complex<double>>::Kernel);
  dict["lapack_sgeqrf"] = EncapsulateFunction(Geqrf<float>::Kernel);
  dict["lapack_dgeqrf"] = EncapsulateFunction(Geqrf<double>::Kernel);
  dict["lapack_cgeqrf"] =
      EncapsulateFunction(Geqrf<std::complex<float>>::Kernel);
  dict["lapack_zgeqrf"] =
      EncapsulateFunction(Geqrf<std::complex<double>>::Kernel);
  dict["lapack_sorgqr"] = EncapsulateFunction(Orgqr<float>::Kernel);
  dict["lapack_dorgqr"] = EncapsulateFunction(Orgqr<double>::Kernel);
  dict["lapack_cungqr"] =
      EncapsulateFunction(Orgqr<std::complex<float>>::Kernel);
  dict["lapack_zungqr"] =
      EncapsulateFunction(Orgqr<std::complex<double>>::Kernel);
  dict["lapack_spotrf"] = EncapsulateFunction(Potrf<float>::Kernel);
  dict["lapack_dpotrf"] = EncapsulateFunction(Potrf<double>::Kernel);
  dict["lapack_cpotrf"] =
      EncapsulateFunction(Potrf<std::complex<float>>::Kernel);
  dict["lapack_zpotrf"] =
      EncapsulateFunction(Potrf<std::complex<double>>::Kernel);
  dict["lapack_sgesdd"] = EncapsulateFunction(RealGesdd<float>::Kernel);
  dict["lapack_dgesdd"] = EncapsulateFunction(RealGesdd<double>::Kernel);
  dict["lapack_cgesdd"] =
      EncapsulateFunction(ComplexGesdd<std::complex<float>>::Kernel);
  dict["lapack_zgesdd"] =
      EncapsulateFunction(ComplexGesdd<std::complex<double>>::Kernel);
  dict["lapack_ssyevd"] = EncapsulateFunction(RealSyevd<float>::Kernel);
  dict["lapack_dsyevd"] = EncapsulateFunction(RealSyevd<double>::Kernel);
  dict["lapack_cheevd"] =
      EncapsulateFunction(ComplexHeevd<std::complex<float>>::Kernel);
  dict["lapack_zheevd"] =
      EncapsulateFunction(ComplexHeevd<std::complex<double>>::Kernel);
  dict["lapack_sgeev"] = EncapsulateFunction(RealGeev<float>::Kernel);
  dict["lapack_dgeev"] = EncapsulateFunction(RealGeev<double>::Kernel);
  dict["lapack_cgeev"] =
      EncapsulateFunction(ComplexGeev<std::complex<float>>::Kernel);
  dict["lapack_zgeev"] =
      EncapsulateFunction(ComplexGeev<std::complex<double>>::Kernel);

  dict["lapack_sgees"] = EncapsulateFunction(RealGees<float>::Kernel);
  dict["lapack_dgees"] = EncapsulateFunction(RealGees<double>::Kernel);
  dict["lapack_cgees"] =
      EncapsulateFunction(ComplexGees<std::complex<float>>::Kernel);
  dict["lapack_zgees"] =
      EncapsulateFunction(ComplexGees<std::complex<double>>::Kernel);
  return dict;
}

PYBIND11_MODULE(_lapack, m) {
  GetLapackKernelsFromScipy();

  m.def("registrations", &Registrations);
  m.def("lapack_sgeqrf_workspace", &Geqrf<float>::Workspace);
  m.def("lapack_dgeqrf_workspace", &Geqrf<double>::Workspace);
  m.def("lapack_cgeqrf_workspace", &Geqrf<std::complex<float>>::Workspace);
  m.def("lapack_zgeqrf_workspace", &Geqrf<std::complex<double>>::Workspace);
  m.def("lapack_sorgqr_workspace", &Orgqr<float>::Workspace);
  m.def("lapack_dorgqr_workspace", &Orgqr<double>::Workspace);
  m.def("lapack_cungqr_workspace", &Orgqr<std::complex<float>>::Workspace);
  m.def("lapack_zungqr_workspace", &Orgqr<std::complex<double>>::Workspace);
  m.def("gesdd_iwork_size", &GesddIworkSize);
  m.def("sgesdd_work_size", &RealGesdd<float>::Workspace);
  m.def("dgesdd_work_size", &RealGesdd<double>::Workspace);
  m.def("cgesdd_rwork_size", &ComplexGesddRworkSize);
  m.def("cgesdd_work_size", &ComplexGesdd<std::complex<float>>::Workspace);
  m.def("zgesdd_work_size", &ComplexGesdd<std::complex<double>>::Workspace);
  m.def("syevd_work_size", &SyevdWorkSize);
  m.def("syevd_iwork_size", &SyevdIworkSize);
  m.def("heevd_work_size", &HeevdWorkSize);
  m.def("heevd_rwork_size", &HeevdRworkSize);
}

}  // namespace
}  // namespace jax
