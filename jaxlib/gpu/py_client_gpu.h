/* Copyright 2022 The JAX Authors

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

#ifndef JAX_JAXLIB_GPU_PY_CLIENT_GPU_H_
#define JAX_JAXLIB_GPU_PY_CLIENT_GPU_H_


#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
XLA_FFI_DECLARE_HANDLER_SYMBOL(kGpuTransposePlanCacheInstantiate);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kXlaFfiPythonGpuCallback);
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAX_JAXLIB_GPU_PY_CLIENT_GPU_H_
