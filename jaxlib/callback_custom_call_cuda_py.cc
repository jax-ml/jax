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

// Implementation of the callback_custom_call for GPU.
// See callback_custom_call.py module documentation for design comments.

#include <sys/types.h>

#include <vector>

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "jaxlib/callback_custom_call.h"
#include "jaxlib/callback_custom_call_generated.h"
#include "jaxlib/cuda_gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"
#include "third_party/tensorflow/compiler/xla/shape_util.h"
#include "third_party/tensorflow/core/platform/logging.h"

namespace jax {

namespace {

namespace py = pybind11;

// Calls the host_callback for GPU.
// opaque is the encoded descriptor
// buffers: token, *array_ops
//   the descriptor.operands describes: token, *array_ops
// results: token, *array_results
//   the descriptor.results describes: token, *array_results
void CallbackCustomCallGPU(cudaStream_t stream, void **buffers,
                           const char *opaque, std::size_t opaque_len) {
  callback_custom_call_fb::Descriptor descriptor =
      callback_custom_call_fb::DecodeDescriptor(opaque);

  int nr_array_ops = descriptor.operands.size() - 1;
  int nr_array_results = descriptor.results.size() - 1;
  // TODO(necula): make a DeviceArray backed by the device memory and pass
  // that to the callback.
  ssize_t ops_total_size = 0;
  for (auto const &op_shape : descriptor.operands) {
    ops_total_size += xla::ShapeUtil::ByteSizeOf(op_shape);
  }
  void *host_memory = 0;  // Page-locked, device-accessible, large-enough memory
  ThrowIfError(cudaMallocHost(&host_memory, ops_total_size));
  std::vector<const void *> host_ops;
  host_ops.reserve(nr_array_ops);

  uint8_t *p_host_memory = reinterpret_cast<uint8_t *>(host_memory);
  for (int i = 0; i < 1 + nr_array_ops; ++i) {
    VLOG(2) << (i == 0 ? "Token" : "Operand ")
            << (i == 0 ? "" : std::to_string(i - 1)) << " has shape "
            << descriptor.operands[i].ToString();
  }
  for (int i = 0; i < nr_array_ops; ++i) {
    size_t op_size = xla::ShapeUtil::ByteSizeOf(descriptor.operands[1 + i]);
    host_ops.push_back(static_cast<const void *>(p_host_memory));
    ThrowIfError(cudaMemcpy(p_host_memory, buffers[1 + i], op_size,
                            cudaMemcpyDeviceToHost));
    p_host_memory += op_size;
  }

  std::vector<const void *> results = RunHostCallback(descriptor, host_ops);
  CHECK_EQ(nr_array_results, results.size());
  // Copy the token from the input.
  ThrowIfError(cudaMemcpy(buffers[1 + nr_array_ops], buffers[0],
                          xla::ShapeUtil::ByteSizeOf(descriptor.results[0]),
                          cudaMemcpyDeviceToDevice));
  for (int res = 0; res < nr_array_results; ++res) {
    ThrowIfError(
        cudaMemcpy(buffers[1 + nr_array_ops + 1 + res], results[res],
                   xla::ShapeUtil::ByteSizeOf(descriptor.results[1 + res]),
                   cudaMemcpyHostToDevice));
  }

  ThrowIfError(cudaFreeHost(host_memory));
}

// Returns a dictionary with CustomCall functions to register.
py::dict CustomCallRegistrations() {
  py::dict dict;
  dict["callback_custom_call"] = EncapsulateFunction(CallbackCustomCallGPU);
  return dict;
}

PYBIND11_MODULE(callback_custom_call_cuda_py, m) {
  m.doc() = "Python bindings for the host_callback GPU runtime";
  m.def("custom_call_registrations", &CustomCallRegistrations);
}

}  // namespace

}  // namespace jax
