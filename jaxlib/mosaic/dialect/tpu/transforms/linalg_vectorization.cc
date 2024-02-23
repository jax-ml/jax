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

#include <memory>
#include <utility>

#include "mlir/include/mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_LINALGVECTORIZATIONPASS
#define GEN_PASS_DEF_LINALGVECTORIZATIONPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {
struct VectorizationPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return vectorize(rewriter, op,
                     /*inputVectorSizes=*/{},
                     /*inputScalableVecDims=*/{},
                     /*vectorizeNDExtract=*/false);
  }
};

struct LinalgVectorizationPass
    : public impl::LinalgVectorizationPassBase<LinalgVectorizationPass> {
  LinalgVectorizationPass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<VectorizationPattern>(ctx);
    // Pull in patterns to shuffle broadcast/transpose ops around in order to
    // cancel them or embed into contract ops. Embedding in the flexible
    // contract ops will help to sustain the structure through various
    // transformations.
    vector::populateVectorReductionToContractPatterns(patterns);
    // Pull in patterns to canonicalize transfer ops.
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);

    // We do not want to apply the vector patterns above to the ops that are
    // unrelated to the original linalg op.
    SmallVector<Operation *> linalgOps;
    func.walk([&](linalg::LinalgOp op) { linalgOps.push_back(op); });
    if (failed(applyOpPatternsAndFold(linalgOps, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}

}  // namespace mlir::tpu
