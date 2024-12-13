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
#include <functional>
#include <optional>
#include <string>
#include <string_view>
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

namespace mlir::tpu {

namespace {

constexpr std::string_view kMangledDialect = "stable_mosaic.";
constexpr StringRef kVersionAttrName = "stable_mosaic.version";
// When this is bumped, we should file a TODO to update the forward-compatible
// version in tpu_custom_call.py in a month!
constexpr int kVersion = 3;

StringRef mangle(StringRef name, std::string* storage) {
  storage->clear();
  storage->reserve(kMangledDialect.size() + name.size());
  storage->insert(storage->end(), kMangledDialect.begin(),
                  kMangledDialect.end());
  storage->insert(storage->end(), name.begin(), name.end());
  return *storage;
}

std::optional<StringRef> demangle(StringRef name) {
  if (!name.starts_with(kMangledDialect)) {
    return std::nullopt;
  }
  return name.drop_front(kMangledDialect.size());
}

using rule_type = std::function<LogicalResult(Operation*, int)>;

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

const llvm::StringMap<rule_type>& upgrade_rules() {
  static auto rules = new llvm::StringMap<rule_type>{
      {EnqueueDMAOp::getOperationName(), enqueue_dma_upgrade},
      {SemaphoreSignalOp::getOperationName(), semaphore_signal_upgrade},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_dim_reduce_upgrade}
  };
  return *rules;
}

const llvm::StringMap<rule_type>& downgrade_rules() {
  static auto rules = new llvm::StringMap<rule_type>{
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
  int serialize_version = target_version.hasValue() ? target_version : kVersion;
  if (serialize && serialize_version > kVersion) {
    module.emitError("The highest supported version is ")
        << kVersion << " but requested serialization at version "
        << serialize_version;
    return signalPassFailure();
  }
  if (serialize && !module->getContext()->allowsUnregisteredDialects()) {
    module.emitError() << "Cannot serialize within a context that does not "
                          "allow unregistered dialects.";
    signalPassFailure();
    return;
  }
  int version = kVersion;
  if (serialize) {
    module->setAttr(kVersionAttrName,
                    IntegerAttr::get(IntegerType::get(module->getContext(), 64),
                                     serialize_version));
  } else {
    IntegerAttr version_attr =
        module->getAttrOfType<IntegerAttr>(kVersionAttrName);
    if (!version_attr) {
      module->emitError("Missing or invalid Mosaic version attribute");
      signalPassFailure();
      return;
    }
    if (version_attr.getInt() > kVersion) {
      module->emitError("Unsupported Mosaic version:  expected <= ")
          << kVersion << " but got " << version_attr.getInt();
      signalPassFailure();
      return;
    }
    version = version_attr.getInt();
    module->removeAttr(kVersionAttrName);
  }
  std::string name_storage;
  auto result = module.walk([&](Operation* op) {
    if (isa<ModuleOp>(op)) {  // Don't mangle the ModuleOp itself.
      return WalkResult::advance();
    }
    std::optional<OperationName> new_name;
    if (serialize) {
      auto new_name_str = mangle(op->getName().getStringRef(), &name_storage);
      new_name = OperationName(new_name_str, op->getContext());
    } else {
      if (auto demangled = demangle(op->getName().getStringRef())) {
        auto new_name_str = *demangled;
        if (auto registered = RegisteredOperationName::lookup(
                new_name_str, op->getContext())) {
          new_name = *registered;
        } else {
          new_name = OperationName(new_name_str, op->getContext());
        }
      } else {
        op->emitError("Operation not in a serialized form");
        return WalkResult::interrupt();
      }
      // Upgrade the op to the current version, if needed.
      if (const auto rule = upgrade_rules().find(new_name->getStringRef());
          rule != upgrade_rules().end()) {
        if (rule->second(op, version).failed()) {
          return WalkResult::interrupt();
        }
      }
    }
    auto new_op = Operation::create(
        op->getLoc(), *new_name, op->getResultTypes(), op->getOperands(),
        op->getAttrs(), nullptr, op->getSuccessors(), op->getRegions());
    // Downgrade the op to the target version, if needed.
    if (serialize && kVersion != serialize_version) {
      if (const auto rule =
              downgrade_rules().find(op->getName().getStringRef());
          rule != downgrade_rules().end()) {
        if (rule->second(new_op, serialize_version).failed()) {
          return WalkResult::interrupt();
        }
      }
    }
    op->getBlock()->getOperations().insertAfter(Block::iterator(op), new_op);
    op->replaceAllUsesWith(new_op->getResults());
    op->erase();
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

}  // namespace mlir::tpu
