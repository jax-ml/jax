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

#include <cstddef>

#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "jaxlib/absl_status_casters.h"
#include "jaxlib/gpu/rnn_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;

nb::bytes BuildRnnDescriptor(int input_size, int hidden_size, int num_layers,
                             int batch_size, int max_seq_length, float dropout,
                             bool bidirectional, bool cudnn_allow_tf32,
                             size_t workspace_size, size_t reserve_space_size) {
  return PackDescriptor(RnnDescriptor{
      input_size, hidden_size, num_layers, batch_size, max_seq_length, dropout,
      bidirectional, cudnn_allow_tf32, workspace_size, reserve_space_size});
}

nb::dict Registrations() {
  nb::dict dict;
  dict[JAX_GPU_PREFIX "dnn_rnn_ffi"] = EncapsulateFfiHandler(RNNForwardFfi);
  dict[JAX_GPU_PREFIX "dnn_rnn_bwd_ffi"] =
      EncapsulateFfiHandler(RNNBackwardFfi);
  return dict;
}

NB_MODULE(_rnn, m) {
  m.def("registrations", &Registrations);
  m.def("build_rnn_descriptor", &BuildRnnDescriptor);
  m.def("compute_rnn_workspace_reserve_space_sizes",
        ValueOrThrowWrapper(RnnComputeWorkspaceReserveSpaceSizes));
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
