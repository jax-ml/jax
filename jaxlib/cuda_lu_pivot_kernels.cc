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

#include "jaxlib/cuda_lu_pivot_kernels.h"

#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {
namespace {

std::string BuildCudaLuPivotsToPermutationDescriptor(
    std::int64_t batch_size, std::int32_t pivot_size,
    std::int32_t permutation_size) {
  return PackDescriptorAsString(LuPivotsToPermutationDescriptor{
      batch_size, pivot_size, permutation_size});
}

absl::Status CudaLuPivotsToPermutation_(cudaStream_t stream, void** buffers,
                                        const char* opaque,
                                        std::size_t opaque_len) {
  auto s =
      UnpackDescriptor<LuPivotsToPermutationDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  LaunchLuPivotsToPermutationKernel(stream, buffers, **s);
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaGetLastError()));
  return absl::OkStatus();
}

void CudaLuPivotsToPermutation(cudaStream_t stream, void** buffers,
                               const char* opaque, size_t opaque_len,
                               XlaCustomCallStatus* status) {
  auto s = CudaLuPivotsToPermutation_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    absl::string_view message = s.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cuda_lu_pivots_to_permutation"] =
      EncapsulateFunction(CudaLuPivotsToPermutation);
  return dict;
}

PYBIND11_MODULE(cuda_lu_pivot_kernels, m) {
  m.def("registrations", &Registrations);
  m.def("cuda_lu_pivots_to_permutation_descriptor",
        [](std::int64_t batch_size, std::int32_t pivot_size,
           std::int32_t permutation_size) {
          std::string result = BuildCudaLuPivotsToPermutationDescriptor(
              batch_size, pivot_size, permutation_size);
          return pybind11::bytes(result);
        });
}

}  // namespace
}  // namespace jax
