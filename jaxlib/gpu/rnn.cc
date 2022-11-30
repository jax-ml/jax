/* Copyright 2022 The JAX Authors.

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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "jaxlib/gpu/rnn_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "pybind11_abseil/status_casters.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace py = pybind11;

py::bytes BuildRnnDescriptor(int input_size, int hidden_size, int num_layers,
                             int batch_size, int max_seq_length, float dropout,
                             bool bidirectional, int workspace_size,
                             int reserve_space_size) {
  return PackDescriptor(RnnDescriptor{
      input_size, hidden_size, num_layers, batch_size, max_seq_length, dropout,
      bidirectional, workspace_size, reserve_space_size});
}

py::dict Registrations() {
  py::dict dict;
  dict[JAX_GPU_PREFIX "dnn_rnn"] = EncapsulateFunction(RNNForward);
  dict[JAX_GPU_PREFIX "dnn_rnn_bwd"] = EncapsulateFunction(RNNBackward);
  return dict;
}

PYBIND11_MODULE(_rnn, m) {
  m.def("registrations", &Registrations);
  m.def("build_rnn_descriptor", &BuildRnnDescriptor);
  m.def("compute_rnn_workspace_reserve_space_sizes",
        &RnnComputeWorkspaceReserveSpaceSizes);
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
