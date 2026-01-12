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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"  // IWYU pragma: keep
#include "jaxlib/mosaic/dialect/tpu/tpu_enums.h.inc"
#include "xla/layout.h"  // IWYU pragma: keep

namespace mlir::tpu {
class TPUDialect;
}  // namespace mlir::tpu

#define GET_ATTRDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_attr_defs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_type_defs.h.inc"

#define GET_OP_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h.inc"
#include "jaxlib/mosaic/dialect/tpu/tpu_ops.h.inc"

namespace mlir {
namespace tpu {

struct TpuTilingFlags {
  bool use_x16_large_second_minor = false;
  bool use_x8_large_second_minor = false;
  bool use_x4_large_second_minor = false;
};

std::pair<bool, bool> mightCommunicateBetweenChips(Operation *op);

// Creates a pass that infers the layout of memrefs in the given function.
//
// The `target_shape` must be 2D and corresponds to (sublane count, lane count)
// TensorCore tiling.
std::unique_ptr<OperationPass<func::FuncOp>> createInferMemRefLayoutPass(
    int hardware_generation, absl::Span<const int64_t> target_shape,
    const TpuTilingFlags& tpu_tiling_flags, bool align = true,
    bool infer_kernel_arguments = true);

#define GEN_PASS_DECL_MOSAICSERDEPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

// Determine the core type of the given op based on the `tpu.core_type`
// annotation of its parent function. If no such annotation is found, returns
// kTc.
FailureOr<CoreType> GetCoreTypeOfParentFunc(Operation &op);

// Changes the memory space of the value and propagates it through the program.
LogicalResult specializeMemorySpace(TypedValue<MemRefType> value,
                                    MemorySpace memory_space);

// In Mosaic, we often strip tiled layouts from memrefs, for compatibility with
// vector ops. This functions inverts the layout erasure applied to the value.
MemRefType getMemRefType(Value value);

bool isGuaranteedDivisible(Value value, int64_t divisor, int64_t fuel = 128);

DotDimensionNumbersAttr defaultDimensionNumbers(Builder &builder,
                                                bool transpose_lhs,
                                                bool transpose_rhs);

#define GEN_PASS_REGISTRATION
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

constexpr std::string_view kLeadingTileRows = "leading_tile_rows";

}  // namespace tpu
}  // namespace mlir

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_
