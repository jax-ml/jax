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

#include "include/pybind11/pybind11.h"
#include "jaxlib/rocm/hip_gpu_kernel_helpers.h"
#include "jaxlib/rocm/hip_lu_pivot_kernels.h"
#include "jaxlib/kernel_pybind11_helpers.h"

namespace jax {
namespace {

std::string
BuildHipLuPivotsToPermutationDescriptor(std::int64_t batch_size,
                                        std::int32_t pivot_size,
                                        std::int32_t permutation_size) {
  return PackDescriptorAsString(LuPivotsToPermutationDescriptor{
      batch_size, pivot_size, permutation_size});
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["hip_lu_pivots_to_permutation"] =
      EncapsulateFunction(HipLuPivotsToPermutation);
  return dict;
}

PYBIND11_MODULE(_hip_linalg, m) {
  m.def("registrations", &Registrations);
  m.def("lu_pivots_to_permutation_descriptor",
        [](std::int64_t batch_size, std::int32_t pivot_size,
           std::int32_t permutation_size) {
          std::string result = BuildHipLuPivotsToPermutationDescriptor(
              batch_size, pivot_size, permutation_size);
          return pybind11::bytes(result);
        });
}

}  // namespace
}  // namespace jax
