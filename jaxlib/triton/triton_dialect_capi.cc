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

#include "jaxlib/triton/triton_dialect_capi.h"

#include "llvm/include/llvm/Support/Casting.h"
#include "mlir/include/mlir-c/IR.h"
#include "mlir/include/mlir/CAPI/IR.h"
#include "mlir/include/mlir/CAPI/Registration.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, triton,
                                      mlir::triton::TritonDialect);

MlirType mlirTritonPointerTypeGet(MlirType pointeeType, int addressSpace) {
  return wrap(
      mlir::triton::PointerType::get(unwrap(pointeeType), addressSpace));
}

bool mlirTritonIsAPointerType(MlirType type) {
  return llvm::isa<mlir::triton::PointerType>(unwrap(type));
}

MlirType mlirTritonPointerTypeGetPointeeType(MlirType pointerType) {
  return wrap(llvm::cast<mlir::triton::PointerType>(unwrap(pointerType))
                  .getPointeeType());
}

}  // extern "C"
