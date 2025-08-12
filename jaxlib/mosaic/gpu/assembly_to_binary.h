/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_GPU_ASSEMBLY_TO_BINARY_H_
#define JAXLIB_MOSAIC_GPU_ASSEMBLY_TO_BINARY_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace mosaic {
namespace gpu {

// Returns a compilation provider that can be used to compile PTX to a CUBIN.
//
// The choice of compilation provider mirrors the logic used in XLA:GPU.
absl::StatusOr<std::unique_ptr<stream_executor::cuda::CompilationProvider>>
GetAssemblyToBinaryCompilationProvider();

// Returns the compute capability of the current device.
absl::StatusOr<stream_executor::CudaComputeCapability>
GetCudaComputeCapability();

void registerAssemblyToBinaryPass();

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_ASSEMBLY_TO_BINARY_H_
