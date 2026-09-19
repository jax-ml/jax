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

#include "jaxlib/thunky/dialect/thunky_dialect.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep

// clang-format off
#include "jaxlib/thunky/dialect/thunky_dialect.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "jaxlib/thunky/dialect/thunky_types.cc.inc"
#define GET_OP_CLASSES
#include "jaxlib/thunky/dialect/thunky_ops.cc.inc"
// clang-format on

namespace mlir::thunky {

void ThunkyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "jaxlib/thunky/dialect/thunky_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "jaxlib/thunky/dialect/thunky_types.cc.inc"
      >();
}

}  // namespace mlir::thunky
