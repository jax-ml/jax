/* Copyright 2024 The JAX Authors.

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

#ifndef JAXLIB_TRITON_TRITON_DIALECT_CAPI_H_
#define JAXLIB_TRITON_TRITON_DIALECT_CAPI_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Triton, triton);

MLIR_CAPI_EXPORTED MlirType mlirTritonPointerTypeGet(MlirType pointeeType,
                                                     int addressSpace);
MLIR_CAPI_EXPORTED bool mlirTritonIsAPointer(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirTritonPointerTypeGetPointeeType(MlirType pointerType);
MLIR_CAPI_EXPORTED int
mlirTritonPointerTypeGetAddressSpace(MlirType pointerType);

MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonInferReduceOpEncoding(MlirAttribute operandEncoding, int axis);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JAXLIB_TRITON_TRITON_DIALECT_CAPI_H_
