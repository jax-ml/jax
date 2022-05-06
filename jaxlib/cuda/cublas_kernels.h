/* Copyright 2019 Google LLC

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

#ifndef JAXLIB_CUBLAS_KERNELS_H_
#define JAXLIB_CUBLAS_KERNELS_H_

#include <cstddef>

#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {

// Set of types known to Cusolver.
enum class CublasType {
  F32,
  F64,
  C64,
  C128,
};


// Batched triangular solve: trsmbatched

struct TrsmBatchedDescriptor {
  CublasType type;
  int batch, m, n;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasOperation_t trans;
  cublasDiagType_t diag;
};

void TrsmBatched(cudaStream_t stream, void** buffers, const char* opaque,
                 size_t opaque_len, XlaCustomCallStatus* status);

// Batched LU decomposition: getrfbatched

struct GetrfBatchedDescriptor {
  CublasType type;
  int batch, n;
};

void GetrfBatched(cudaStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace jax

#endif  // JAXLIB_CUBLAS_KERNELS_H_
