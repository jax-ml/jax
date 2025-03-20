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

#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, triton,
                                      mlir::triton::TritonDialect);

MlirType mlirTritonPointerTypeGet(MlirType pointeeType, int addressSpace) {
  return wrap(
      mlir::triton::PointerType::get(unwrap(pointeeType), addressSpace));
}

bool mlirTritonIsAPointer(MlirType type) {
  return llvm::isa<mlir::triton::PointerType>(unwrap(type));
}

MlirType mlirTritonPointerTypeGetPointeeType(MlirType pointerType) {
  return wrap(llvm::cast<mlir::triton::PointerType>(unwrap(pointerType))
                  .getPointeeType());
}

int mlirTritonPointerTypeGetAddressSpace(MlirType pointerType) {
  return llvm::cast<mlir::triton::PointerType>(unwrap(pointerType))
      .getAddressSpace();
}

MlirAttribute mlirTritonInferReduceOpEncoding(MlirAttribute operandEncoding,
                                              int axis) {
  auto opEncoding = unwrap(operandEncoding);
  mlir::Dialect &dialect = opEncoding.getDialect();
  auto inferLayoutInterface =
      llvm::dyn_cast<mlir::triton::DialectInferLayoutInterface>(&dialect);
  mlir::Attribute retEncoding;
  (void)inferLayoutInterface->inferReduceOpEncoding(opEncoding, axis,
                                                    retEncoding);
  return wrap(retEncoding);
}

}  // extern "C"
