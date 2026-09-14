/* Copyright 2024 The JAX Authors.

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
  // For a device pointer argument, `value` is the pointer value itself and
  // `size == 0`. For a host (byval) argument, `value` points
  // into `host_bytes` and `size` is the length in bytes.
  struct Arg {
    const void* value = nullptr;
    int32_t size = 0;
    bool is_host = false;
  };

  Dim3 grid;
  Dim3 block;
  Dim3 cluster = {0, 0, 0};  // {0, 0, 0} => no cluster attribute.
  uint32_t smem_bytes = 0;
  bool uses_pdl = false;

  absl::InlinedVector<Arg, 8> args;
  // Owns copied host-argument bytes (e.g. TMA descriptor bundles).
  // Arg::value above points into this vector.
  absl::InlinedVector<std::byte, 256> host_bytes;

  // Returns a vector of kernel argument pointers, suitable for passing to
  // cuLaunchKernelEx.
  absl::InlinedVector<void*, 8> kernel_params() {
    absl::InlinedVector<void*, 8> params(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      params[i] =
          args[i].is_host ? const_cast<void*>(args[i].value) : &args[i].value;
    }
    return params;
  }
};

}  // namespace gpu
}  // namespace mosaic

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_LAUNCH_CONFIG_H_
