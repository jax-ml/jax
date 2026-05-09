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

// This file is not used by JAX itself, but exists to assist with running
// JAX-generated HLO code from outside of JAX.

#include "jaxlib/gpu/triton_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_target_registry.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("triton_kernel_call", TritonKernelCall,
                                         "CUDA");

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "triton_kernel_call_ffi", "CUDA",
                         {
                            /*instantiate=*/nullptr,
                            /*prepare=*/nullptr,
                            /*initialize=*/kTritonKernelCallFfiInitialize,
                            /*execute=*/kTritonKernelCallFfi,
                        });
}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
