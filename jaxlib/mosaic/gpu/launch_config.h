/* Copyright 2026 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_LAUNCH_CONFIG_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_LAUNCH_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <variant>

#include "absl/container/inlined_vector.h"

namespace mosaic {
namespace gpu {

// CUDA-style kernel Dim3.
struct Dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// Describes a kernel launch.
struct MosaicKernelSpec {
  struct DeviceArg {
    void* ptr = nullptr;
  };

  // A byval argument (e.g. a TMA descriptor bundle) whose payload occupies
  // `host_bytes[offset, offset + size)`.
  struct HostArg {
    int32_t offset = 0;
    int32_t size = 0;
  };

  using Arg = std::variant<DeviceArg, HostArg>;

  Dim3 grid;
  Dim3 block;
  Dim3 cluster = {0, 0, 0};  // {0, 0, 0} => no cluster attribute.
  uint32_t smem_bytes = 0;
  bool uses_pdl = false;

  absl::InlinedVector<Arg, 8> args;
  // Owns copied host-argument bytes (e.g. TMA descriptor bundles).
  // `HostArg::offset` above indexes into this vector.
  absl::InlinedVector<std::byte, 256> host_bytes;

  bool IsHostArg(int32_t arg_index) const {
    return std::holds_alternative<HostArg>(args[arg_index]);
  }

  // Appends a device pointer argument.
  // Pre: `ptr` must be a valid device pointer.
  void AddDeviceArg(void* ptr) { args.push_back(DeviceArg{ptr}); }

  // Copies `size` bytes of byval payload into `host_bytes` and appends the
  // corresponding argument.
  // Pre: `size` must be positive.
  void AddHostArg(const void* data, int32_t size) {
    const size_t offset = host_bytes.size();
    host_bytes.resize(offset + size);
    std::memcpy(host_bytes.data() + offset, data, size);
    args.push_back(HostArg{static_cast<int32_t>(offset), size});
  }

  void Clear() {
    args.clear();
    host_bytes.clear();
  }

  // Pre-allocates space for arguments. Not mandatory to call this method.
  // However, it avoids reallocations when the arguments are added.
  void Reserve(int32_t num_args, int32_t host_bytes_size) {
    args.reserve(num_args);
    host_bytes.reserve(host_bytes_size);
  }

  // Returns a vector of kernel argument pointers, suitable for passing to
  // cuLaunchKernelEx.
  //
  // Borrows from `*this`: the result must not outlive this spec, and is
  // invalidated by any mutation of `args` or `host_bytes`.
  [[nodiscard]] absl::InlinedVector<void*, 8> kernel_params() {
    absl::InlinedVector<void*, 8> params;
    params.reserve(args.size());
    for (Arg& arg : args) {
      if (auto* host = std::get_if<HostArg>(&arg)) {
        // A byval argument is passed as the address of its payload.
        params.push_back(host_bytes.data() + host->offset);
      } else {
        // A device pointer is passed as the address of the pointer value.
        params.push_back(&std::get<DeviceArg>(arg).ptr);
      }
    }
    return params;
  }
};

}  // namespace gpu
}  // namespace mosaic

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_LAUNCH_CONFIG_H_
