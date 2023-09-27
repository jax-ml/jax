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
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"

namespace jax::cuda {
namespace {

int CudaRuntimeGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cudaRuntimeGetVersion(&version)));
  return version;
}

int CudaDriverGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cudaDriverGetVersion(&version)));
  return version;
}

uint32_t CuptiGetVersion() {
  uint32_t version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cuptiGetVersion(&version)));
  return version;
}

int CufftGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cufftGetVersion(&version)));
  return version;
}

int CusolverGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverGetVersion(&version)));
  return version;
}

NB_MODULE(_versions, m) {
  // Build versions, i.e., what version of the headers was JAX compiled against?
  m.def("cuda_runtime_build_version", []() { return CUDART_VERSION; });
  m.def("cudnn_build_version", []() { return CUDNN_VERSION; });
  m.def("cublas_build_version", []() { return CUBLAS_VERSION; });
  m.def("cupti_build_version", []() { return CUPTI_API_VERSION; });
  m.def("cufft_build_version", []() { return CUFFT_VERSION; });
  m.def("cusolver_build_version", []() { return CUSOLVER_VERSION; });
  m.def("cusparse_build_version", []() { return CUSPARSE_VERSION; });

  // TODO(phawkins): annoyingly cublas and cusparse have "get version" APIs that
  // require the library to be initialized.
  m.def("cuda_runtime_get_version", &CudaRuntimeGetVersion);
  m.def("cuda_driver_get_version", &CudaDriverGetVersion);
  m.def("cudnn_get_version", &cudnnGetVersion);
  m.def("cupti_get_version", &CuptiGetVersion);
  m.def("cufft_get_version", &CufftGetVersion);
  m.def("cusolver_get_version", &CusolverGetVersion);
}

}  // namespace
}  // namespace jax::cuda