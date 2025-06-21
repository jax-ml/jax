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

#include "absl/base/call_once.h"
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
  static absl::once_flag initialized;
  // For reasons I'm not entirely sure of, if the import_ call is done inside
  // the call_once scope, we sometimes observe deadlocks in the test suite.
  // However it probably doesn't do much harm to just import them a second time,
  // since that costs little more than a dictionary lookup or two.
  nb::module_ cython_lapack =
      nb::module_::import_("scipy.linalg.cython_lapack");
  absl::call_once(initialized, [&]() {
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
