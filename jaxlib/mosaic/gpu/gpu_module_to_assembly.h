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

#ifndef JAXLIB_MOSAIC_GPU_MODULE_TO_ASSEMBLY_H_
#define JAXLIB_MOSAIC_GPU_MODULE_TO_ASSEMBLY_H_

#include <string>
#include <vector>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mosaic {
namespace gpu {

// Registers a pass that converts `gpu.module` ops into `gpu.binary` ops
// wrapping a PTX assembly.
void registerGpuModuleToAssemblyPass();

// Initializes the NVPTX target on first call.
void EnsureLLVMNVPTXTargetIsRegistered();

namespace internal {

// Implements the main logic of the pass. This is exposed for testing only.
llvm::LogicalResult LowerGpuModuleToAssembly(
    mlir::gpu::GPUModuleOp gpu_module,
    const std::vector<std::string>& libraries_to_link);
}  // namespace internal

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_MODULE_TO_ASSEMBLY_H_
