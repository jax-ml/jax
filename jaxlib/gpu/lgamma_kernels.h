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

#ifndef JAXLIB_GPU_LGAMMA_KERNELS_H_
#define JAXLIB_GPU_LGAMMA_KERNELS_H_

#include <cstdint>

#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

void LaunchLgammaF32Kernel(gpuStream_t stream, int64_t n,
                           xla::ffi::DataType dtype, const void* in, void* out);

void LaunchLgammaF32WithF64Kernel(gpuStream_t stream, int64_t n,
                                  xla::ffi::DataType dtype, const void* in,
                                  void* out);

void LaunchLgammaF64Kernel(gpuStream_t stream, int64_t n, const double* in,
                           double* out);

XLA_FFI_DECLARE_HANDLER_SYMBOL(LgammaFfi);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_LGAMMA_KERNELS_H_
