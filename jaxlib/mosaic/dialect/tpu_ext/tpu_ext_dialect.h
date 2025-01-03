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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_EXT_TPU_EXT_DIALECT_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_EXT_TPU_EXT_DIALECT_H_

#include "mlir/include/mlir/IR/BuiltinOps.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/Operation.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/PatternMatch.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep

namespace mlir::tpu::ext {
class TPUExtDialect;
}  // namespace mlir::tpu::ext

#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_attr_defs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_type_defs.h.inc"

#define GET_OP_CLASSES
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_dialect.h.inc"
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_ops.h.inc"

namespace mlir::tpu::ext {

#define GEN_PASS_REGISTRATION
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_pass_defs.h.inc"

}  // namespace mlir::tpu::ext

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_EXT_TPU_EXT_DIALECT_H_
