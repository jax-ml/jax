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

#include <memory>

#include "nanobind/nanobind.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/gpu/hybrid_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/ffi/api/ffi.h"
#include "xla/python/safe_static_init.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {
namespace ffi = xla::ffi;
namespace nb = nanobind;

void GetLapackKernelsFromScipy() {
  static xla::SafeStatic<bool> initialized;
  initialized.Get([]() {
    if (lapack_kernels_initialized) {
      return std::make_unique<bool>(true);
    }
    // Technically these are Cython-internal APIs. However, it seems highly
    // likely they will remain stable because Cython itself needs API stability
    // for cross-package imports to work in the first place.
    nb::module_ cython_lapack =
        nb::module_::import_("scipy.linalg.cython_lapack");

    nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
    auto lapack_ptr = [&](const char* name) {
      return nb::cast<nb::capsule>(lapack_capi[name]).data();
    };

    AssignKernelFn<EigenvalueDecomposition<ffi::F32>>(lapack_ptr("sgeev"));
    AssignKernelFn<EigenvalueDecomposition<ffi::F64>>(lapack_ptr("dgeev"));
    AssignKernelFn<EigenvalueDecompositionComplex<ffi::C64>>(
        lapack_ptr("cgeev"));
    AssignKernelFn<EigenvalueDecompositionComplex<ffi::C128>>(
        lapack_ptr("zgeev"));
    AssignKernelFn<PivotingQrFactorization<ffi::F32>>(lapack_ptr("sgeqp3"));
    AssignKernelFn<PivotingQrFactorization<ffi::F64>>(lapack_ptr("dgeqp3"));
    AssignKernelFn<PivotingQrFactorization<ffi::C64>>(lapack_ptr("cgeqp3"));
    AssignKernelFn<PivotingQrFactorization<ffi::C128>>(lapack_ptr("zgeqp3"));
    lapack_kernels_initialized = true;
    return std::make_unique<bool>(true);
  });
}

NB_MODULE(_hybrid, m) {
  m.def("initialize", GetLapackKernelsFromScipy);
  m.def("has_magma", []() { return MagmaLookup().FindMagmaInit().ok(); });
  m.def("registrations", []() {
    nb::dict dict;
    dict[JAX_GPU_PREFIX "hybrid_eig_real"] = EncapsulateFfiHandler(kEigReal);
    dict[JAX_GPU_PREFIX "hybrid_eig_comp"] = EncapsulateFfiHandler(kEigComp);
    dict[JAX_GPU_PREFIX "hybrid_geqp3"] = EncapsulateFfiHandler(kGeqp3);
    return dict;
  });
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
