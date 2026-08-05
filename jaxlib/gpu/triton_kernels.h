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

#ifndef JAXLIB_GPU_TRITON_H_
#define JAXLIB_GPU_TRITON_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "jaxlib/gpu/triton.pb.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/ffi.h"
#include "xla/service/custom_call_status.h"

namespace jax::JAX_GPU_NAMESPACE {

void TritonKernelCall(gpuStream_t stream, void** buffers, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status);

XLA_FFI_DECLARE_HANDLER_SYMBOL(kTritonKernelCallFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kTritonKernelCallFfiInitialize);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kTritonKernelCallFfiInstantiate);

class ModuleImage;

struct TritonCustomCallState {
  jax_triton::TritonCustomCallStateProto proto;

  TritonCustomCallState() { proto.set_version(1); }

  // Returns true if the kernel is fully compiled down to machine code (e.g.
  // CUBIN), and is ready to be launched.
  bool IsReadyForExecution() const {
    return proto.has_kernel_call() &&
           proto.kernel_call().kernel().has_module_image();
  }

  // Returns true if the state contains a set of compiled kernel candidates that
  // first need to be autotuned, before execution.
  bool NeedsAutotuning() const {
    return proto.has_autotuning_kernel_candidates();
  }

  static absl::StatusOr<std::string> Serialize(
      const TritonCustomCallState& state) {
    return state.proto.SerializeAsString();
  }

  static absl::StatusOr<std::unique_ptr<TritonCustomCallState>> Deserialize(
      absl::string_view data) {
    auto state = std::make_unique<TritonCustomCallState>();
    if (!state->proto.ParseFromString(data)) {
      return absl::InvalidArgumentError(
          "Failed to parse TritonCustomCallStateProto");
    }
    if (state->proto.version() != 1) {
      return absl::InvalidArgumentError(
          "Unsupported TritonCustomCallStateProto version");
    }
    return state;
  }
};

class Kernel {
 public:
  Kernel(std::string kernel_name, uint32_t num_warps, uint32_t num_ctas,
         uint32_t shared_mem_bytes, std::string ptx, std::string ttir,
         int compute_capability, ModuleImage* module_image = nullptr);

  absl::Status Launch(gpuStream_t stream, uint32_t grid[3], void** params);

  static absl::StatusOr<Kernel> FromProto(
      const jax_triton::TritonKernel& proto);
  jax_triton::TritonKernel ToProto() const;

  // Returns true if we can launch the kernel without crashing.
  bool CanLaunchOnDevice(gpuDevice_t) const;

  const std::string& kernel_name() const { return kernel_name_; }
  uint32_t shared_mem_bytes() const { return shared_mem_bytes_; }
  const std::string& ptx() const { return ptx_; }
  int compute_capability() const { return compute_capability_; }

 private:
  std::string kernel_name_;
  uint32_t block_dim_x_;
  uint32_t num_ctas_;
  uint32_t shared_mem_bytes_;
  std::string ptx_;
  std::string ttir_;
  int compute_capability_;

  ModuleImage* module_image_ = nullptr;
};

class KernelCall {
 public:
  struct Parameter {
    struct Array {
      size_t bytes_to_zero;
      size_t ptr_divisibility;
    };

    static absl::StatusOr<Parameter> FromProto(
        const jax_triton::TritonKernelCall_Parameter& proto);
    jax_triton::TritonKernelCall_Parameter ToProto() const;

    std::variant<Array, bool, int32_t, uint32_t, int64_t, uint64_t, float,
                 double>
        value;
  };

  KernelCall(Kernel kernel, uint32_t grid_0, uint32_t grid_1, uint32_t grid_2,
             std::vector<Parameter> parameters);

  absl::Status Launch(gpuStream_t stream, void** buffers);

  static absl::StatusOr<KernelCall> FromProto(
      const jax_triton::TritonKernelCall& proto);
  jax_triton::TritonKernelCall ToProto() const;

  // Returns true if we can launch the kernel without crashing.
  bool CanLaunchOnDevice(gpuDevice_t) const;

  const Kernel& kernel() const { return kernel_; }

 private:
  Kernel kernel_;
  uint32_t grid_[3];
  std::vector<Parameter> parameters_;
};

class AutotunedKernelCall {
 public:
  struct Config {
    KernelCall kernel_call;
    std::string description;
  };

  AutotunedKernelCall(
      std::string name, std::vector<Config> configs,
      std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases);

  static absl::StatusOr<KernelCall> Autotune(AutotunedKernelCall kernel_call,
                                             gpuStream_t stream,
                                             void** buffers);

  static absl::StatusOr<AutotunedKernelCall> FromProto(
      const jax_triton::TritonAutotunedKernelCall& proto);
  jax_triton::TritonAutotunedKernelCall ToProto() const;

 private:
  std::string name_;
  std::vector<Config> configs_;
  // (input buffer idx, output buffer idx, size)
  std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases_;
};

}  // namespace jax::JAX_GPU_NAMESPACE

#endif  // JAXLIB_GPU_TRITON_H_
