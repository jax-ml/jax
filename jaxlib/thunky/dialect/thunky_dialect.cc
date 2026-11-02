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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

// clang-format off
#include "jaxlib/thunky/dialect/thunky_dialect.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "jaxlib/thunky/dialect/thunky_types.cc.inc"
#define GET_OP_CLASSES
#include "jaxlib/thunky/dialect/thunky_ops.cc.inc"
// clang-format on

namespace mlir::thunky {
namespace {

struct ThunkyInlinerInterface final : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const override {
    return true;
  }

  bool isLegalToInline(Operation* op, Region* dest, bool wouldBeCloned,
                       IRMapping& valueMapping) const override {
    return true;
  }

  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       IRMapping& valueMapping) const override {
    return true;
  }
};

}  // namespace

void CopyOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add(+[](CopyOp op, PatternRewriter& rewriter) -> LogicalResult {
    if (op.getSrc() == op.getDst()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  });
}

void ThunkyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "jaxlib/thunky/dialect/thunky_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "jaxlib/thunky/dialect/thunky_types.cc.inc"
      >();
  addInterfaces<ThunkyInlinerInterface>();
}

}  // namespace mlir::thunky
