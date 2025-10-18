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

#ifndef JAXLIB_GPU_RNN_KERNELS_H_
#define JAXLIB_GPU_RNN_KERNELS_H_

#include <cstddef>
#include <utility>

#include "absl/status/statusor.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

// Compile-time info passed as `opaque` to custom kernel.
struct RnnDescriptor {
  int input_size;
  int hidden_size;
  int num_layers;
  int batch_size;
  int max_seq_length;
  float dropout;
  int bidirectional;
  int cudnn_allow_tf32;
  size_t workspace_size;
  size_t reserve_space_size;
};

// Return (workspace size, reserve space size).
absl::StatusOr<std::pair<size_t, size_t>> RnnComputeWorkspaceReserveSpaceSizes(
    int input_size, int hidden_size, int num_layers, int batch_size,
    int max_seq_length, float dropout, bool bidirectional,
    bool cudnn_allow_tf32);

XLA_FFI_DECLARE_HANDLER_SYMBOL(RNNForwardFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(RNNBackwardFfi);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_RNN_KERNELS_H_
