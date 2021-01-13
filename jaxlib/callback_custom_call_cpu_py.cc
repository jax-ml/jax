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

// Implementation of the callback_custom_call for CPU.
// See callback_custom_call.py module documentation for design comments.

#include "jaxlib/callback_custom_call.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"
#include "third_party/tensorflow/compiler/xla/service/hlo.proto.h"
#include "third_party/tensorflow/compiler/xla/shape_util.h"

namespace jax {

namespace py = pybind11;

namespace {

// Calls the host_callback.
//
// ins: descriptor, token, *array_ops
//   the descriptor.operands describes: token, *array_ops
// out: token, *array_results
//   the descriptor.results describes: token, *array_results
void CallbackCustomCallCPU(void *out, const void **ins) {
  callback_custom_call_fb::Descriptor descriptor =
      callback_custom_call_fb::DecodeDescriptor(ins[0]);
  int nr_array_ops = descriptor.operands.size() - 1;
  std::vector<const void *> array_ops;
  array_ops.reserve(nr_array_ops);
  for (int i = 0; i < nr_array_ops; ++i) {
    array_ops.push_back(ins[2 + i]);
  }
  std::vector<const void *> array_results =
      RunHostCallback(descriptor, array_ops);
  void **outs = reinterpret_cast<void **>(out);
  int nr_array_results = descriptor.results.size() - 1;
  CHECK_EQ(nr_array_results, array_results.size());
  memcpy(outs[0], ins[1],  // copy the input token
         xla::ShapeUtil::ByteSizeOf(descriptor.results[0]));
  for (int i = 0; i < nr_array_results; ++i) {
    memcpy(outs[i + 1], array_results[i],
           xla::ShapeUtil::ByteSizeOf(descriptor.results[i + 1]));
  }
}

// Returns a dictionary with CustomCall functions to register.
py::dict CustomCallRegistrations() {
  py::dict dict;
  dict["callback_custom_call"] = EncapsulateFunction(CallbackCustomCallCPU);
  return dict;
}

PYBIND11_MODULE(callback_custom_call_cpu_py, m) {
  m.doc() = "Python bindings for the host_callback runtime for CPU.";
  m.def("custom_call_registrations", &CustomCallRegistrations);
  m.def("set_callback_trampoline", &SetCallbackTrampoline,
        py::call_guard<py::gil_scoped_release>());
  m.def("encode_descriptor", &callback_custom_call_fb::EncodeDescriptor);
}

}  // namespace

}  // namespace jax
