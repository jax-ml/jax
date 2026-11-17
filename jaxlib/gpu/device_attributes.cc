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

#include "jaxlib/gpu/device_attributes.h"

#include <atomic>
#include <cstdint>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

constexpr int32_t kMaxCachedDevices = 64;
std::atomic<int> g_perf_ratio_cache[kMaxCachedDevices];

int QuerySingleToDoublePrecisionPerfRatio(int32_t device_ordinal) {
#if defined(JAX_GPU_CUDA)
  int ratio = 0;
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, device_ordinal) == CUDA_SUCCESS &&
      cuDeviceGetAttribute(
          &ratio, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
          dev) == CUDA_SUCCESS &&
      ratio > 0) {
    return ratio;
  }
  return 32;
#else
  (void)device_ordinal;
  // TODO(phawkins): Query the ROCm device architecture once HIP provides an API
  // for the single-to-double precision performance ratio.
  return 32;
#endif
}

}  // namespace

int GetSingleToDoublePrecisionPerfRatio(int32_t device_ordinal) {
  if (device_ordinal >= 0 && device_ordinal < kMaxCachedDevices) {
    const int cached =
        g_perf_ratio_cache[device_ordinal].load(std::memory_order_relaxed);
    if (cached > 0) {
      return cached;
    }
    const int queried = QuerySingleToDoublePrecisionPerfRatio(device_ordinal);
    g_perf_ratio_cache[device_ordinal].store(queried,
                                             std::memory_order_relaxed);
    return queried;
  }
  return QuerySingleToDoublePrecisionPerfRatio(device_ordinal);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
