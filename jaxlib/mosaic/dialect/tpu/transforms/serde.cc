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

#include "jaxlib/mosaic/dialect/tpu/transforms/serde.h"

#include <cstdint>
#include <vector>

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/include/llvm/ADT/StringMap.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/serde.h"

namespace mlir::tpu {

namespace {

constexpr StringRef kMangledDialect = "stable_mosaic.";
constexpr StringRef kVersionAttrName = "stable_mosaic.version";
// When this is bumped, we should file a TODO to update the forward-compatible
// version in tpu_custom_call.py in a month!
constexpr int kVersion = 3;

using SerdeRuleType = jaxlib::mosaic::SerdeRuleType;

LogicalResult enqueue_dma_upgrade(Operation* op, int version) {
  // Added AttrSizedOperandSegments and core_id in version 2.
  if (version < 2) {
    if (op->getNumOperands() == 3) {  // Local DMA.
      op->setAttr(
          OpTrait::AttrSizedOperandSegments<
              EnqueueDMAOp>::getOperandSegmentSizeAttr(),
          mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 0, 1, 1, 0, 0}));
    } else if (op->getNumOperands() == 5) {  // Remote DMA.
      op->setAttr(
          OpTrait::AttrSizedOperandSegments<
              EnqueueDMAOp>::getOperandSegmentSizeAttr(),
          mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 1, 1, 0}));
    } else {
      return op->emitError("Unexpected operand count in tpu.enqueue_dma: ")
             << op->getNumOperands();
    }
  }
  return success();
}

LogicalResult enqueue_dma_downgrade(Operation* op, int version) {
  if (version < 2) {
    return op->emitError("Downgrade to version ") << version << " unsupported";
  }
  return success();
}

LogicalResult semaphore_signal_upgrade(Operation* op, int version) {
  // Added AttrSizedOperandSegments and core_id in version 2.
  if (version < 2) {
    if (op->getNumOperands() == 2) {  // Local signal.
      op->setAttr(OpTrait::AttrSizedOperandSegments<
                      EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                  mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 0, 0}));
    } else if (op->getNumOperands() == 3) {  // Remote signal.
      op->setAttr(OpTrait::AttrSizedOperandSegments<
                      EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                  mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 0}));
    } else {
      return op->emitError("Unexpected operand count in tpu.semaphore_signal");
    }
  }
  return success();
}

LogicalResult semaphore_signal_downgrade(Operation* op, int version) {
  if (version < 2) {
    auto operands = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!operands || operands.size() != 4) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    if (operands[3]) {
      return op->emitError("Downgrade to version ")
             << version << " impossible: core_id is set";
    }
    op->removeAttr(OpTrait::AttrSizedOperandSegments<
                   EnqueueDMAOp>::getOperandSegmentSizeAttr());
  }
  return success();
}

LogicalResult vector_multi_dim_reduce_upgrade(Operation* op, int version) {
  // Changed reductions_dims from ArrayAttr of IntegerAttrs to DenseI64ArrayAttr
  // in version 3.
  if (version < 3) {
    Attribute reduction_dims_attr = op->getAttr("reduction_dims");
    if (!reduction_dims_attr) {
      return op->emitError("Missing reduction_dims attribute");
    }
    ArrayAttr reduction_dims_array = dyn_cast<ArrayAttr>(reduction_dims_attr);
    if (!reduction_dims_array) {
      return op->emitOpError("reduction_dims attribute is not an ArrayAttr");
    }
    std::vector<int64_t> reduction_dims;
    reduction_dims.reserve(reduction_dims_array.size());
    for (Attribute reduction_dim : reduction_dims_array) {
      IntegerAttr reduction_dim_attr = dyn_cast<IntegerAttr>(reduction_dim);
      if (!reduction_dim_attr) {
        return op->emitOpError(
            "reduction_dims attribute contains a non-IntegerAttr");
      }
      reduction_dims.push_back(reduction_dim_attr.getInt());
    }
    op->setAttr("reduction_dims",
                DenseI64ArrayAttr::get(op->getContext(), reduction_dims));
  }
  return success();
}

LogicalResult vector_multi_dim_reduce_downgrade(Operation* op, int version) {
  if (version < 3) {
    return op->emitError("Downgrade to version ") << version << " unsupported";
  }
  return success();
}

const llvm::StringMap<SerdeRuleType>& upgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{
      {EnqueueDMAOp::getOperationName(), enqueue_dma_upgrade},
      {SemaphoreSignalOp::getOperationName(), semaphore_signal_upgrade},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_dim_reduce_upgrade}};
  return *rules;
}

const llvm::StringMap<SerdeRuleType>& downgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{
      {EnqueueDMAOp::getOperationName(), enqueue_dma_downgrade},
      {SemaphoreSignalOp::getOperationName(), semaphore_signal_downgrade},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_dim_reduce_downgrade}};
  return *rules;
}

}  // namespace

void MosaicSerdePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (!serialize.hasValue()) {
    module.emitError("serialize option must be specified");
    return signalPassFailure();
  }
  int serialize_version = -1;
  if (serialize) {
     serialize_version = target_version.hasValue() ? target_version : kVersion;
  }
  if (failed(jaxlib::mosaic::RunSerde(
          module, upgrade_rules(), downgrade_rules(), serialize,
          {.dialect_prefix = kMangledDialect,
           .highest_version = kVersion,
           .version_attr_name = kVersionAttrName,
           .serialize_version = serialize_version}))) {
    signalPassFailure();
  }
}

}  // namespace mlir::tpu
