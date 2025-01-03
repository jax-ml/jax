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

#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_dialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep.
#include "mlir/include/mlir/IR/DialectImplementation.h"  // IWYU pragma: keep.
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_dialect.cc.inc"
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_enums.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_attr_defs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_type_defs.cc.inc"

namespace mlir::tpu::ext {

void TPUExtDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_attr_defs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "jaxlib/mosaic/dialect/tpu_ext/tpu_ext_ops.cc.inc"
      >();
}

}  // namespace mlir::tpu::ext
