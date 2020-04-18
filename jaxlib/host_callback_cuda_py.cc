/* Copyright 2020 Google LLC

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

// This exposes the host_callback_py Python module with the implementation
// of the CustomCalls for GPU.


#include <sstream>
#include <vector>

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/pytypes.h"
#include "jaxlib/gpu_kernel_helpers.h"
#include "jaxlib/host_callback.h"
#include "jaxlib/kernel_helpers.h"
#include "jaxlib/kernel_pybind11_helpers.h"

namespace jax {

namespace {

namespace py = pybind11;

// This is the entry point for the GPU CustomCall.
void PrintGPU(cudaStream_t stream, void **buffers, const char *opaque,
              std::size_t opaque_len) {
  const PrintMetadata meta =
      ParsePrintMetadata(std::string(opaque, opaque_len));
  int nr_args = meta.args_type_and_shape.size();
  // Start by writing the result, in case of errors.
  bool result = true;
<<<<<<< HEAD
  cudaError_t cuda_err = cudaMemcpy(buffers[nr_args], &result, sizeof(result),
                                    cudaMemcpyHostToDevice);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "cudaMemcpy returned error " << cudaGetErrorString(cuda_err);
    return;
  }

  void* host_memory = 0;  // Page-locked, device-accessible large-enough memory
  cuda_err = cudaMallocHost(&host_memory, meta.MaximumByteSize());
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "cudaMallocHost returned error " <<
        cudaGetErrorString(cuda_err);
    return;
  }
  std::ostringstream output_stream;

  for (int arg = 0; arg < nr_args && cuda_err == cudaSuccess; ++arg) {
    TypeAndShape arg_ts = meta.args_type_and_shape[arg];
    cuda_err = cudaMemcpy(host_memory, buffers[arg], arg_ts.ByteSize(),
                          cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
      LOG(ERROR) << "cudaMemcpy returned error "
          << cudaGetErrorString(cuda_err);
      return;
    }
=======
  ThrowIfError(cudaMemcpy(buffers[nr_args], &result, sizeof(result),
                          cudaMemcpyHostToDevice));

  void* host_memory = 0;  // Page-locked, device-accessible large-enough memory
  ThrowIfError(cudaMallocHost(&host_memory, meta.MaximumByteSize()));

  std::ostringstream output_stream;
  for (int arg = 0; arg < nr_args; ++arg) {
    TypeAndShape arg_ts = meta.args_type_and_shape[arg];
    ThrowIfError(cudaMemcpy(host_memory, buffers[arg], arg_ts.ByteSize(),
                            cudaMemcpyDeviceToHost));
>>>>>>> origin/changelist/306845374
    EmitOneArray(output_stream, meta, arg, host_memory);
  }
  std::cout << output_stream.str();

<<<<<<< HEAD
  cuda_err = cudaFreeHost(host_memory);
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "cudaFree returned error " << cudaGetErrorString(cuda_err);
    return;
  }
  ThrowIfError(cudaGetLastError());
=======
  ThrowIfError(cudaFreeHost(host_memory));
>>>>>>> origin/changelist/306845374
}

// Returns a dictionary with CustomCall functions to register.
py::dict CustomCallRegistrations() {
  py::dict dict;
  dict["jax_print"] = EncapsulateFunction(PrintGPU);
  return dict;
}

PYBIND11_MODULE(host_callback_cuda_py, m) {
  m.doc() = "Python bindings for the host_callback GPU runtime";
  m.def("customcall_registrations", &CustomCallRegistrations);
}

}  // namespace

}  // namespace jax
