/* Copyright 2025 The JAX Authors.

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

#include "jaxlib/mosaic/gpu/serde.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/serde.h"

namespace mosaic::gpu {

namespace {

using ::llvm::ArrayRef;
using ::llvm::LogicalResult;
using ::llvm::success;
using ::mlir::Operation;
using ::mlir::Value;

constexpr llvm::StringRef kMangledDialect = "stable_mosaic_gpu.";
constexpr llvm::StringRef kVersionAttrName = "stable_mosaic_gpu.version";
// When this is bumped, we should file a TODO to update the forward-compatible
// version in Mosaic GPU lowering in a month!
//
// TODO(apaszke): Update the forward-compatible version to 3 in Mosaic GPU
// lowering after 2025-10-08.
// TODO(apaszke): Update the forward-compatible version to 4 in Mosaic GPU
// lowering after 2025-11-13.
// TODO(apaszke): Update the forward-compatible version to 5 in Mosaic GPU
// lowering after 2025-12-07.
// TODO(apaszke): Update the forward-compatible version to 6 in Mosaic GPU
// lowering after 2025-12-18.
constexpr int kVersion = 6;

using SerdeRuleType = jaxlib::mosaic::SerdeRuleType;

LogicalResult vector_extractelement_upgrade(Operation* op, int version,
                                            bool& erased) {
  if (version < 2) {
    // vector.extractelement was removed in
    // https://github.com/llvm/llvm-project/commit/33465bb2bb75f26b7ad42ab87ccb2464c0245476.
    // We replace it with a vector.extract.
    mlir::OpBuilder b(op->getParentRegion());
    b.setInsertionPointAfter(op);
    Value vec = op->getOperand(0);
    Value position = op->getOperand(1);
    Value extracted_value = mlir::vector::ExtractOp::create(
        b, op->getLoc(), vec, ArrayRef<mlir::OpFoldResult>{position});

    op->replaceAllUsesWith(llvm::SmallVector<Value>{extracted_value});
    op->erase();
    erased = true;
  }
  return success();
}

LogicalResult vector_insertelement_upgrade(Operation* op, int version,
                                           bool& erased) {
  if (version < 2) {
    // vector.insertelement was removed in
    // https://github.com/llvm/llvm-project/commit/33465bb2bb75f26b7ad42ab87ccb2464c0245476.
    // We replace it with a vector.insert.
    mlir::OpBuilder b(op->getParentRegion());
    b.setInsertionPointAfter(op);
    Value source = op->getOperand(0);
    Value destination = op->getOperand(1);
    Value position = op->getOperand(2);

    Value inserted_value =
        mlir::vector::InsertOp::create(b, op->getLoc(), source, destination,
                                       ArrayRef<mlir::OpFoldResult>{position});
    op->replaceAllUsesWith(llvm::SmallVector<Value>{inserted_value});
    op->erase();
    erased = true;
  }
  return success();
}

LogicalResult nvvm_cp_async_bulk_tensor_global_shared_cta_upgrade(
    Operation* op, int version, bool& erased) {
  // A new operand was added in
  // https://github.com/llvm/llvm-project/pull/155435/commits/216550ca2169677dd6fc33bc47c3e1ba6d93fc20
  if (version < 3) {
    auto sizes_attr =
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes");
    if (!sizes_attr) {
      return op->emitOpError(
          "Missing or invalid operandSegmentSizes attribute");
    }
    if (sizes_attr.getSize() != 4) {
      return op->emitOpError("operandSegmentSizes attribute has wrong size");
    }
    auto new_sizes = sizes_attr.asArrayRef().vec();
    new_sizes.insert(new_sizes.end() - 1, 0);
    op->setAttr("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(op->getContext(), new_sizes));
  }
  return success();
}

LogicalResult nvvm_cp_async_bulk_tensor_global_shared_cta_downgrade(
    Operation* op, int version, bool& erased) {
  // A new operand was added in
  // https://github.com/llvm/llvm-project/pull/155435/commits/216550ca2169677dd6fc33bc47c3e1ba6d93fc20
  if (version < 3) {
    auto sizes_attr =
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes");
    if (!sizes_attr) {
      return op->emitOpError(
          "Missing or invalid operandSegmentSizes attribute");
    }
    if (sizes_attr.getSize() != 5) {
      return op->emitOpError("operandSegmentSizes attribute has wrong size");
    }
    auto new_sizes = sizes_attr.asArrayRef().vec();
    if (*(new_sizes.end() - 2) != 0) {
      return op->emitOpError("Can't downgrade: l2 hint operand is present");
    }
    new_sizes.erase(new_sizes.end() - 2);
    op->setAttr("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(op->getContext(), new_sizes));
  }
  return success();
}

LogicalResult vector_splat_upgrade(Operation* op, int version, bool& erased) {
  if (version < 4) {
    // vector.splat was removed in
    // https://github.com/llvm/llvm-project/commit/ea291d0e8c93d47d7953eff5ca1048891a5fcc55.
    // We replace it with a vector.broadcast.
    mlir::OpBuilder b(op->getParentRegion());
    b.setInsertionPointAfter(op);
    Value inserted_value = mlir::vector::BroadcastOp::create(
        b, op->getLoc(), op->getResult(0).getType(), op->getOperand(0));
    op->replaceAllUsesWith(llvm::SmallVector<Value>{inserted_value});
    op->erase();
    erased = true;
  }
  return success();
}

LogicalResult nvvm_mbarrier_init_shared_upgrade(Operation* op, int version,
                                                bool& erased) {
  // https://github.com/llvm/llvm-project/commit/523706f2cd6a06bd9557bf0dca9986d867eddd79
  if (version < 5) {
    mlir::OpBuilder b(op->getParentRegion());
    b.setInsertionPointAfter(op);
    mlir::NVVM::MBarrierInitOp::create(
        b, op->getLoc(), op->getOperand(0), op->getOperand(1),
        op->getNumOperands() < 3 ? Value{} : op->getOperand(2));
    op->erase();
    erased = true;
  }
  return success();
}

LogicalResult nvvm_mbarrier_try_wait_parity_shared_upgrade(Operation* op,
                                                           int version,
                                                           bool& erased) {
  // https://github.com/llvm/llvm-project/commit/7eeae8e41d7827d84de12df7b5ecfab3058900cb
  if (version < 6) {
    mlir::OpBuilder b(op->getParentRegion());
    b.setInsertionPointAfter(op);
    mlir::NVVM::MBarrierTryWaitParityOp::create(
        b, op->getLoc(), op->getOperand(0), op->getOperand(1),
        op->getOperand(2));
    op->erase();
    erased = true;
  }
  return success();
}

LogicalResult nvvm_mbarrier_arrive_expect_tx_shared_upgrade(Operation* op,
                                                            int version,
                                                            bool& erased) {
  // https://github.com/llvm/llvm-project/commit/7eeae8e41d7827d84de12df7b5ecfab3058900cb
  if (version < 6) {
    mlir::OpBuilder b(op->getParentRegion());
    b.setInsertionPointAfter(op);
    mlir::NVVM::MBarrierArriveExpectTxOp::create(
        b, op->getLoc(), op->getOperand(0), op->getOperand(1),
        op->getNumOperands() < 3 ? Value{} : op->getOperand(2));
    op->erase();
    erased = true;
  }
  return success();
}

const llvm::StringMap<SerdeRuleType>& upgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{
      {::llvm::StringLiteral("vector.extractelement"),
       vector_extractelement_upgrade},
      {::llvm::StringLiteral("vector.insertelement"),
       vector_insertelement_upgrade},
      {::llvm::StringLiteral("nvvm.cp.async.bulk.tensor.global.shared.cta"),
       nvvm_cp_async_bulk_tensor_global_shared_cta_upgrade},
      {::llvm::StringLiteral("vector.splat"), vector_splat_upgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.init.shared"),
       nvvm_mbarrier_init_shared_upgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.try_wait.parity.shared"),
       nvvm_mbarrier_try_wait_parity_shared_upgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.arrive.expect_tx.shared"),
       nvvm_mbarrier_arrive_expect_tx_shared_upgrade},
  };
  return *rules;
}

const llvm::StringMap<SerdeRuleType>& downgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{
      {::llvm::StringLiteral("nvvm.cp.async.bulk.tensor.global.shared.cta"),
       nvvm_cp_async_bulk_tensor_global_shared_cta_downgrade}};
  return *rules;
}

}  // namespace

void SerdePass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  if (!serialize.hasValue()) {
    module.emitError("serialize option must be specified");
    return signalPassFailure();
  }
  int serialize_version = -1;
  if (serialize) {
    serialize_version = target_version.hasValue() ? target_version : kVersion;
  }
  if (mlir::failed(jaxlib::mosaic::RunSerde(
          module, upgrade_rules(), downgrade_rules(), serialize,
          {.dialect_prefix = kMangledDialect,
           .highest_version = kVersion,
           .version_attr_name = kVersionAttrName,
           .serialize_version = serialize_version}))) {
    signalPassFailure();
  }
}

}  // namespace mosaic::gpu
