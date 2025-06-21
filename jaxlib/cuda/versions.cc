/* Copyright 2023 The JAX Authors.

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
#include "jaxlib/cuda/versions_helpers.h"
#include "jaxlib/gpu/vendor.h"

namespace jax::cuda {
namespace {

namespace nb = nanobind;

NB_MODULE(_versions, m) {
  // Nanobind's leak checking sometimes returns false positives for this file.
  // The problem appears related to forming a closure of a nanobind function.
  nb::set_leak_warnings(false);

  // Build versions, i.e., what version of the headers was JAX compiled against?
  m.def("cuda_runtime_build_version", []() { return CUDART_VERSION; });
  m.def("cudnn_build_version", []() { return CUDNN_VERSION; });
  m.def("cublas_build_version", []() { return CUBLAS_VERSION; });
  m.def("cupti_build_version", []() { return CUPTI_API_VERSION; });
  m.def("cufft_build_version", []() { return CUFFT_VERSION; });
  m.def("cusolver_build_version", []() { return CUSOLVER_VERSION; });
  m.def("cusparse_build_version", []() { return CUSPARSE_VERSION; });

  m.def("cuda_runtime_get_version", &CudaRuntimeGetVersion);
  m.def("cuda_driver_get_version", &CudaDriverGetVersion);
  m.def("cudnn_get_version", &CudnnGetVersion);
  m.def("cupti_get_version", &CuptiGetVersion);
  m.def("cufft_get_version", &CufftGetVersion);
  m.def("cusolver_get_version", &CusolverGetVersion);
  m.def("cublas_get_version", &CublasGetVersion);
  m.def("cusparse_get_version", &CusparseGetVersion);
  m.def("cuda_compute_capability", &CudaComputeCapability);
  m.def("cuda_device_count", &CudaDeviceCount);
}

}  // namespace
}  // namespace jax::cuda