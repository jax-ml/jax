/* Copyright 2019 The JAX Authors.

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
#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;

nb::dict Registrations() {
  nb::dict dict;

  dict[JAX_GPU_PREFIX "solver_getrf_ffi"] = EncapsulateFfiHandler(GetrfFfi);
  dict[JAX_GPU_PREFIX "solver_geqrf_ffi"] = EncapsulateFfiHandler(GeqrfFfi);
  dict[JAX_GPU_PREFIX "solver_orgqr_ffi"] = EncapsulateFfiHandler(OrgqrFfi);
  dict[JAX_GPU_PREFIX "solver_syevd_ffi"] = EncapsulateFfiHandler(SyevdFfi);
  dict[JAX_GPU_PREFIX "solver_syrk_ffi"] = EncapsulateFfiHandler(SyrkFfi);
  dict[JAX_GPU_PREFIX "solver_gesvd_ffi"] = EncapsulateFfiHandler(GesvdFfi);
  dict[JAX_GPU_PREFIX "solver_sytrd_ffi"] = EncapsulateFfiHandler(SytrdFfi);

#ifdef JAX_GPU_CUDA
  dict[JAX_GPU_PREFIX "solver_gesvdj_ffi"] = EncapsulateFfiHandler(GesvdjFfi);
  dict[JAX_GPU_PREFIX "solver_csrlsvqr_ffi"] =
      EncapsulateFfiHandler(CsrlsvqrFfi);
#endif  // JAX_GPU_CUDA

  return dict;
}

NB_MODULE(_solver, m) {
  m.def("registrations", &Registrations);
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
