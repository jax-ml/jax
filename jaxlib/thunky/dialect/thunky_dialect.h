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

#ifndef JAXLIB_THUNKY_DIALECT_THUNKY_DIALECT_H_
#define JAXLIB_THUNKY_DIALECT_THUNKY_DIALECT_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "jaxlib/thunky/dialect/thunky_dialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "jaxlib/thunky/dialect/thunky_types.h.inc"
#define GET_OP_CLASSES
#include "jaxlib/thunky/dialect/thunky_ops.h.inc"
// clang-format on

#endif  // JAXLIB_THUNKY_DIALECT_THUNKY_DIALECT_H_
