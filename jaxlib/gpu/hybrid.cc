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
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/gpu/hybrid_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {
namespace ffi = xla::ffi;
namespace nb = nanobind;

void GetLapackKernelsFromScipy() {
  static bool initialized = false;  // Protected by GIL
  if (initialized) return;
  nb::module_ cython_blas = nb::module_::import_("scipy.linalg.cython_blas");
  nb::module_ cython_lapack =
      nb::module_::import_("scipy.linalg.cython_lapack");
  nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
  auto lapack_ptr = [&](const char* name) {
    return nb::cast<nb::capsule>(lapack_capi[name]).data();
  };

  AssignKernelFn<EigenvalueDecomposition<ffi::F32>>(lapack_ptr("sgeev"));
  AssignKernelFn<EigenvalueDecomposition<ffi::F64>>(lapack_ptr("dgeev"));
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::C64>>(lapack_ptr("cgeev"));
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::C128>>(
      lapack_ptr("zgeev"));
}

NB_MODULE(_hybrid, m) {
  m.def("initialize", GetLapackKernelsFromScipy);
  m.def("has_magma", []() { return MagmaLookup().FindMagmaInit().ok(); });
  m.def("registrations", []() {
    nb::dict dict;
    dict[JAX_GPU_PREFIX "hybrid_eig_real"] = EncapsulateFfiHandler(kEigReal);
    dict[JAX_GPU_PREFIX "hybrid_eig_comp"] = EncapsulateFfiHandler(kEigComp);
    return dict;
  });
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
