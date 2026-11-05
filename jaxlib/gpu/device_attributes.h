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

#ifndef JAXLIB_GPU_DEVICE_ATTRIBUTES_H_
#define JAXLIB_GPU_DEVICE_ATTRIBUTES_H_

#include <cstdint>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

// Returns the device's single-to-double precision throughput ratio (for
// example, `2` on A100/H100/B200 GPUs with high-throughput FP64 units, and
// `32` or `64` on GPUs with low-throughput FP64 units such as L4/T4/RTX).
// Queries `CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO` on CUDA
// and caches the result per `device_ordinal`.
int GetSingleToDoublePrecisionPerfRatio(int32_t device_ordinal);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_DEVICE_ATTRIBUTES_H_
