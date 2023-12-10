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
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/lu_pivot_kernels.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;

std::string BuildLuPivotsToPermutationDescriptor(
    std::int64_t batch_size, std::int32_t pivot_size,
    std::int32_t permutation_size) {
  return PackDescriptorAsString(LuPivotsToPermutationDescriptor{
      batch_size, pivot_size, permutation_size});
}

nb::dict Registrations() {
  nb::dict dict;
  dict[JAX_GPU_PREFIX "_lu_pivots_to_permutation"] =
      EncapsulateFunction(LuPivotsToPermutation);
  return dict;
}

NB_MODULE(_linalg, m) {
  m.def("registrations", &Registrations);
  m.def("lu_pivots_to_permutation_descriptor",
        [](std::int64_t batch_size, std::int32_t pivot_size,
           std::int32_t permutation_size) {
          std::string result = BuildLuPivotsToPermutationDescriptor(
              batch_size, pivot_size, permutation_size);
          return nb::bytes(result.data(), result.size());
        });
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
