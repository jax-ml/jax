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

#include <cstdlib>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/serde.h"

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
// TODO(bchetioui): Update the forward-compatible version to 7 in Mosaic GPU
// lowering after 2026-08-31.
// TODO(bchetioui): Update the forward-compatible version to 8 in Mosaic GPU
// lowering after 2026-09-30.
constexpr int kVersion = 8;

using SerdeRuleType = jaxlib::mosaic::SerdeRuleType;

std::optional<mlir::Attribute> MakeNvvmProperty(llvm::StringRef tag,
                                                llvm::StringRef inner,
                                                mlir::MLIRContext* ctx) {
  if (tag == "mem_scope") {
    if (auto val = mlir::NVVM::symbolizeMemScopeKind(inner)) {
      return mlir::NVVM::MemScopeKindAttr::get(ctx, *val);
    }
  } else if (tag == "tma_load_mode") {
    if (auto val = mlir::NVVM::symbolizeTMALoadMode(inner)) {
      return mlir::NVVM::TMALoadModeAttr::get(ctx, *val);
    }
  } else if (tag == "tma_store_mode") {
    if (auto val = mlir::NVVM::symbolizeTMAStoreMode(inner)) {
      return mlir::NVVM::TMAStoreModeAttr::get(ctx, *val);
    }
  } else if (tag == "tcgen05_cp_shape") {
    if (auto val = mlir::NVVM::symbolizeTcgen05CpShape(inner)) {
      return mlir::NVVM::Tcgen05CpShapeAttr::get(ctx, *val);
    }
  } else if (tag == "tcgen05_cp_multicast") {
    if (auto val = mlir::NVVM::symbolizeTcgen05CpMulticast(inner)) {
      return mlir::NVVM::Tcgen05CpMulticastAttr::get(ctx, *val);
    }
  } else if (tag == "tcgen05_ldst_shape") {
    if (auto val = mlir::NVVM::symbolizeTcgen05LdStShape(inner)) {
      return mlir::NVVM::Tcgen05LdStShapeAttr::get(ctx, *val);
    }
  } else if (tag == "shfl_kind") {
    if (auto val = mlir::NVVM::symbolizeShflKind(inner)) {
      return mlir::NVVM::ShflKindAttr::get(ctx, *val);
    }
  } else if (tag == "proxy_kind") {
    if (auto val = mlir::NVVM::symbolizeProxyKind(inner)) {
      return mlir::NVVM::ProxyKindAttr::get(ctx, *val);
    }
  } else if (tag == "shared_space") {
    if (auto val = mlir::NVVM::symbolizeSharedSpace(inner)) {
      return mlir::NVVM::SharedSpaceAttr::get(ctx, *val);
    }
  } else if (tag == "action") {
    if (auto val = mlir::NVVM::symbolizeSetMaxRegisterAction(inner)) {
      return mlir::NVVM::SetMaxRegisterActionAttr::get(ctx, *val);
    }
  } else if (tag == "vote_sync_kind") {
    if (auto val = mlir::NVVM::symbolizeVoteSyncKind(inner)) {
      return mlir::NVVM::VoteSyncKindAttr::get(ctx, *val);
    }
  } else if (tag == "cta_group") {
    if (auto val = mlir::NVVM::symbolizeCTAGroupKind(inner)) {
      return mlir::NVVM::CTAGroupKindAttr::get(ctx, *val);
    }
  } else if (tag == "tcgen05_fence") {
    if (auto val = mlir::NVVM::symbolizeTcgen05FenceKind(inner)) {
      return mlir::NVVM::Tcgen05FenceKindAttr::get(ctx, *val);
    }
  } else if (tag == "tcgen05_wait") {
    if (auto val = mlir::NVVM::symbolizeTcgen05WaitKind(inner)) {
      return mlir::NVVM::Tcgen05WaitKindAttr::get(ctx, *val);
    }
  } else if (tag == "ld_st_matrix_elt_type") {
    if (auto val = mlir::NVVM::symbolizeLdStMatrixEltType(inner)) {
      return mlir::NVVM::LdStMatrixEltTypeAttr::get(ctx, *val);
    }
  } else if (tag == "mma_layout") {
    if (auto val = mlir::NVVM::symbolizeMMALayout(inner)) {
      return mlir::NVVM::MMALayoutAttr::get(ctx, *val);
    }
  } else if (tag == "ld_st_matrix_shape") {
    // The string is of the form "m = ?, n = ?".
    auto [m_str, n_str] = inner.split(',');
    int m = atoi(m_str.substr(3).data());
    int n = atoi(n_str.substr(4).data());
    return mlir::NVVM::LdStMatrixShapeAttr::get(ctx, m, n);
  } else if (tag == "load_cache_modifier") {
    if (auto val = mlir::NVVM::symbolizeLoadCacheModifierKind(inner)) {
      return mlir::NVVM::LoadCacheModifierKindAttr::get(ctx, *val);
    }
  } else if (tag == "reduction_kind") {
    if (auto val = mlir::NVVM::symbolizeReductionKind(inner)) {
      return mlir::NVVM::ReductionKindAttr::get(ctx, *val);
    }
  }

  return std::nullopt;
}

std::optional<mlir::Attribute> NvvmAttrToProperty(mlir::OpaqueAttr opaque,
                                                  mlir::MLIRContext* ctx) {
  CHECK_EQ(opaque.getDialectNamespace(), "nvvm");

  // Attribute format. Necessary to support upgrade rules.
  llvm::StringRef data = opaque.getAttrData();
  auto [tag, inner] = data.split('<');
  if (inner.consume_back(">")) {
    if (auto property = MakeNvvmProperty(tag.trim(), inner.trim(), ctx)) {
      return *property;
    }
  }

  // Property format. Necessary to support downgrade rules.
  auto [tag_sp, inner_sp] = data.split(' ');
  if (!inner_sp.empty()) {
    if (auto property = MakeNvvmProperty(tag_sp.trim(), inner_sp.trim(), ctx)) {
      return *property;
    }
  }
  return std::nullopt;
}

LogicalResult nvvm_attrs_upgrade(Operation* op, int version, bool& erased) {
  auto* ctx = op->getContext();
  llvm::SmallVector<mlir::NamedAttribute> new_attrs;
  bool changed = false;
  for (auto named_attr : op->getAttrs()) {
    auto opaque = mlir::dyn_cast<mlir::OpaqueAttr>(named_attr.getValue());
    if (!opaque || opaque.getDialectNamespace() != "nvvm") {
      new_attrs.push_back(named_attr);
      continue;
    }
    // This is currently version-independent, because we encode NVVM properties
    // as `OpaqueAttr`s in more recent versions of the code as well.
    //
    // TODO(bchetioui): make this version-dependent (version < 8) in a follow-up
    // change, once Pallas/Mosaic GPU stops generating NVVM ops.
    std::optional<mlir::Attribute> property = NvvmAttrToProperty(opaque, ctx);
    if (!property) {
      return op->emitOpError("Failed to upgrade NVVM attribute: ")
             << named_attr.getName() << " = " << opaque;
    }
    changed = true;
    new_attrs.push_back(mlir::NamedAttribute(named_attr.getName(), *property));
  }
  if (changed) {
    op->setAttrs(new_attrs);
  }
  return success();
}

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
  if (failed(nvvm_attrs_upgrade(op, version, erased))) {
    return mlir::failure();
  }
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

LogicalResult nvvm_attrs_downgrade(Operation* op, int version, bool& erased) {
  if (version < 8) {
    for (auto named_attr : op->getAttrs()) {
      auto attr = named_attr.getValue();
      if (auto opaque = mlir::dyn_cast<mlir::OpaqueAttr>(attr);
          !opaque || opaque.getDialectNamespace() != "nvvm") {
        continue;
      }

      std::string attr_str;
      llvm::raw_string_ostream os(attr_str);
      attr.print(os);
      // Ensure that the attribute can be represented as parsable attributes,
      // and error otherwise.
      if (!llvm::StringRef(attr_str).contains('<') ||
          !llvm::StringRef(attr_str).contains('>')) {
        return op->emitOpError(
                   "Can't downgrade: NVVM attribute cannot be serialized to "
                   "pre-v8 "
                   "format: ")
               << attr_str;
      }
    }
  }
  return success();
}

LogicalResult nvvm_cp_async_bulk_tensor_global_shared_cta_downgrade(
    Operation* op, int version, bool& erased) {
  if (failed(nvvm_attrs_downgrade(op, version, erased))) {
    return mlir::failure();
  }
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
  if (failed(nvvm_attrs_upgrade(op, version, erased))) {
    return mlir::failure();
  }
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
  if (failed(nvvm_attrs_upgrade(op, version, erased))) {
    return mlir::failure();
  }
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
  if (failed(nvvm_attrs_upgrade(op, version, erased))) {
    return mlir::failure();
  }
  // https://github.com/llvm/llvm-project/commit/fddf7b0510e5df7a08c512a177ea9c1ec4307718
  if (version < 6) {
    mlir::ImplicitLocOpBuilder b(op->getLoc(), op->getParentRegion());
    b.setInsertionPointAfter(op);
    auto new_op = mlir::NVVM::MBarrierArriveExpectTxOp::create(
        b, op->getResultTypes(), op->getOperand(0), op->getOperand(1),
        mlir::NVVM::MemScopeKind::CTA,
        /*relaxed=*/false,
        op->getNumOperands() < 3 ? mlir::Value{} : op->getOperand(2));
    op->replaceAllUsesWith(new_op);
    op->erase();
    erased = true;
  }
  return success();
}

LogicalResult gpu_launch_upgrade(Operation* op, int version, bool& erased) {
  // https://github.com/llvm/llvm-project/commit/056dae16f7d219f0dc943828dccd6cc3773d2674
  if (version < 7) {
    auto sizes_attr =
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes");
    if (!sizes_attr) {
      return op->emitOpError(
          "Missing or invalid operandSegmentSizes attribute");
    }
    if (sizes_attr.getSize() != 11) {
      return op->emitOpError("operandSegmentSizes attribute has wrong size");
    }

    auto new_sizes = sizes_attr.asArrayRef().vec();
    new_sizes.push_back(0);
    op->setAttr("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(op->getContext(), new_sizes));
  }
  return success();
}

LogicalResult gpu_launch_downgrade(Operation* op, int version, bool& erased) {
  // https://github.com/llvm/llvm-project/commit/056dae16f7d219f0dc943828dccd6cc3773d2674
  if (version < 7) {
    auto sizes_attr =
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes");
    if (!sizes_attr) {
      return op->emitOpError(
          "Missing or invalid operandSegmentSizes attribute");
    }

    if (sizes_attr.getSize() != 12) {
      return op->emitOpError("operandSegmentSizes attribute has wrong size");
    }
    auto new_sizes = sizes_attr.asArrayRef().vec();
    if (new_sizes.back() != 0) {
      return op->emitOpError("Can't downgrade: asyncObject operand is present");
    }
    new_sizes.pop_back();
    op->setAttr("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(op->getContext(), new_sizes));
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
      {::llvm::StringLiteral("gpu.launch"), gpu_launch_upgrade},
      {::llvm::StringLiteral("nvvm.shfl.sync"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.vote.sync"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.fence.proxy"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.cp.async.bulk.tensor.shared.cluster.global"),
       nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.init"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.arrive.expect_tx"),
       nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.test.wait"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.try_wait.parity"),
       nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.alloc"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.dealloc"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.relinquish_alloc_permit"),
       nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.commit"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.cp"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.ld"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.st"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.stmatrix"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.ldmatrix"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.setmaxregister"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.fence"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.wait"), nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.cp.async.shared.global"),
       nvvm_attrs_upgrade},
      {::llvm::StringLiteral("nvvm.redux.sync"), nvvm_attrs_upgrade},
  };
  return *rules;
}

const llvm::StringMap<SerdeRuleType>& downgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{
      {::llvm::StringLiteral("nvvm.cp.async.bulk.tensor.global.shared.cta"),
       nvvm_cp_async_bulk_tensor_global_shared_cta_downgrade},
      {::llvm::StringLiteral("gpu.launch"), gpu_launch_downgrade},
      // TODO(bchetioui): delete nvvm ops out of the Mosaic GPU codebase, and
      // get rid of the downgrade rules.
      {::llvm::StringLiteral("nvvm.shfl.sync"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.vote.sync"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.fence.proxy"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.cp.async.bulk.tensor.shared.cluster.global"),
       nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.init"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.arrive.expect_tx"),
       nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.test.wait"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.mbarrier.try_wait.parity"),
       nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.alloc"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.dealloc"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.relinquish_alloc_permit"),
       nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.commit"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.cp"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.ld"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.st"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.stmatrix"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.ldmatrix"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.setmaxregister"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.fence"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.tcgen05.wait"), nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.cp.async.shared.global"),
       nvvm_attrs_downgrade},
      {::llvm::StringLiteral("nvvm.redux.sync"), nvvm_attrs_downgrade},
  };
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
