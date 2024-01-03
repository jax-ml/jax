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

// We need to keep some extra headers for the code in tpu_passes.h.inc.

#include <memory>  // IWYU pragma: keep
#include <optional>
#include <string>
#include <string_view>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/include/mlir/IR/OperationSupport.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_MOSAICSERDEPASS
#define GEN_PASS_DEF_MOSAICSERDEPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

constexpr std::string_view kMangledDialect = "stable_mosaic.";
constexpr StringRef kVersionAttrName = "stable_mosaic.version";
constexpr int kVersion = 1;

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

struct MosaicSerdePass : public impl::MosaicSerdePassBase<MosaicSerdePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (serialize && !module->getContext()->allowsUnregisteredDialects()) {
      module.emitError() << "Cannot serialize within a context that does not "
                            "allow unregistered dialects.";
      signalPassFailure();
      return;
    }
    if (serialize) {
      module->setAttr(
          kVersionAttrName,
          IntegerAttr::get(IntegerType::get(module->getContext(), 64),
                           kVersion));
    } else {
      IntegerAttr version_attr =
          module->getAttrOfType<IntegerAttr>(kVersionAttrName);
      if (!version_attr) {
        module->emitError("Missing or invalid Mosaic version attribute");
        signalPassFailure();
        return;
      }
      if (version_attr.getValue() != kVersion) {
        module->emitError("Unsupported Mosaic version: ")
            << version_attr.getValue().getSExtValue();
        signalPassFailure();
        return;
      }
      module->removeAttr(kVersionAttrName);
    }
    std::string name_storage;
    auto result = module.walk([this, &name_storage](Operation* op) {
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
      }
      auto new_op = Operation::create(
          op->getLoc(), *new_name, op->getResultTypes(), op->getOperands(),
          op->getAttrs(), nullptr, op->getSuccessors(), op->getRegions());
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
};

}  // namespace

}  // namespace mlir::tpu
