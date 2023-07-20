/* Copyright 2023 The JAX Authors.

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

#include "jaxlib/mlir/_mlir_libs/jax_dialects.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/Passes.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arith, arith, mlir::arith::ArithDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Vector, vector, mlir::vector::VectorDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Math, math, mlir::math::MathDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MemRef, memref, mlir::memref::MemRefDialect)

MLIR_CAPI_EXPORTED void mlirJAXRegisterAllPasses() { mlir::registerTransformsPasses(); }

}