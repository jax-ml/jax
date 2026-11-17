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
#include "jaxlib/gpu/sparse_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace nb = nanobind;

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

nb::dict Registrations() {
  nb::dict dict;
  dict[JAX_GPU_PREFIX "sparse_csr_todense_ffi_v2"] =
      EncapsulateFfiHandler(kCsrToDense);
  dict[JAX_GPU_PREFIX "sparse_csr_fromdense_ffi_v2"] =
      EncapsulateFfiHandler(kCsrFromDense);
  dict[JAX_GPU_PREFIX "sparse_csr_matvec_ffi_v2"] =
      EncapsulateFfiHandler(kCsrMatvec);
  dict[JAX_GPU_PREFIX "sparse_csr_matmat_ffi_v2"] =
      EncapsulateFfiHandler(kCsrMatmat);
  dict[JAX_GPU_PREFIX "sparse_coo_todense_ffi_v2"] =
      EncapsulateFfiHandler(kCooToDense);
  dict[JAX_GPU_PREFIX "sparse_coo_fromdense_ffi_v2"] =
      EncapsulateFfiHandler(kCooFromDense);
  dict[JAX_GPU_PREFIX "sparse_coo_matvec_ffi_v2"] =
      EncapsulateFfiHandler(kCooMatvec);
  dict[JAX_GPU_PREFIX "sparse_coo_matmat_ffi_v2"] =
      EncapsulateFfiHandler(kCooMatmat);
  dict[JAX_GPU_PREFIX "sparse_gtsv2_ffi"] = EncapsulateFfiHandler(kGtsv2);

  // TODO(tomhennigan): Add support for gtsv2 complex 32/64.
  return dict;
}

NB_MODULE(_sparse, m) { m.def("registrations", &Registrations); }

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
