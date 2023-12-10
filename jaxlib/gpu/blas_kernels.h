/* Copyright 2019 The JAX Authors.

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

#ifndef JAXLIB_GPU_BLAS_KERNELS_H_
#define JAXLIB_GPU_BLAS_KERNELS_H_

#include <cstddef>

#include "jaxlib/gpu/vendor.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

// Set of types known to Cusolver.
enum class BlasType {
  F32,
  F64,
  C64,
  C128,
};

// Batched LU decomposition: getrfbatched

struct GetrfBatchedDescriptor {
  BlasType type;
  int batch, n;
};

void GetrfBatched(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

// Batched QR decomposition: geqrfbatched

struct GeqrfBatchedDescriptor {
  BlasType type;
  int batch, m, n;
};

void GeqrfBatched(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_BLAS_KERNELS_H_
