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

#include <cstdint>

#include "jaxlib/gpu/sleep_kernel.h"
#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

// Approximate: cycles are counted at a nominal 2 GHz, like XLA's delay kernel.
constexpr std::int64_t kNominalCyclesPerNanosecond = 2;

__global__ void SleepKernel(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

}  // namespace

void LaunchSleepKernel(gpuStream_t stream, std::int64_t nanoseconds) {
  SleepKernel<<</*grid_dim=*/1, /*block_dim=*/1,
                /*dynamic_shared_mem_bytes=*/0, stream>>>(
      static_cast<long long>(nanoseconds * kNominalCyclesPerNanosecond));
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
