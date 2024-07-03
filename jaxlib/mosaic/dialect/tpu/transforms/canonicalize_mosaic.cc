#include <functional>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// It requires these headers, but does not include them.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/include/mlir/IR/AffineExpr.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_CANONICALIZEMOSAICPASS
#define GEN_PASS_DEF_CANONICALIZEMOSAICPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

LogicalResult tpu_matmul_rule(tpu::MatmulOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();

  const VectorType lhs_ty = lhs.getType();
  const VectorType rhs_ty = rhs.getType();

  auto lhs_element_type = lhs_ty.getElementType();
  auto rhs_element_type = rhs_ty.getElementType();

  if (lhs_element_type != rhs_element_type) {
    if (lhs_element_type.isInteger() || rhs_element_type.isInteger()) {
      op->emitOpError("Mix int/float or different int/int - NYI");
      return failure();
    }
  }
  // TODO(voz): Add more invariants.
  // TODO(voz): Insert extf/ sitof/ etc ops to cast the operands to the
  // correct type for mixed matmul cases.
  return success();
};

LogicalResult canonicalize_matmul(Operation &op) {
  auto matmul_op = dyn_cast<tpu::MatmulOp>(op);
  if (!matmul_op) {
    op.emitOpError("Invariant violated: Not a matmul");
    return failure();
  }
  return tpu_matmul_rule(matmul_op);
};

LogicalResult canonicalize_contraction(Operation &op) {
  auto contraction_op = dyn_cast<vector::ContractionOp>(op);
  if (!contraction_op) {
    op.emitOpError("Invariant violated: Not a contraction");
    return failure();
  }
  // Rewrite the contraction as a matmul
  auto lhs = contraction_op.getLhs();
  auto rhs = contraction_op.getRhs();
  auto acc = contraction_op.getAcc();
  VectorType acc_ty;
  if (!(acc_ty = dyn_cast<VectorType>(acc.getType()))) {
    contraction_op->emitOpError("Not implemented: acc must be a vector");
    return failure();
  }

  if (contraction_op.getKind() != vector::CombiningKind::ADD) {
    contraction_op->emitOpError("Only ADD supported");
    return failure();
  }

  ImplicitLocOpBuilder builder(contraction_op->getLoc(),
                               contraction_op.getOperation());

  MLIRContext *const mlir_ctx = contraction_op->getContext();

  auto getMapAttr = [&](const unsigned first, const unsigned second) {
    return AffineMapAttr::get(AffineMap::get(
        3, 0,
        {getAffineDimExpr(first, mlir_ctx), getAffineDimExpr(second, mlir_ctx)},
        mlir_ctx));
  };

  const ArrayAttr matmul_indexing_maps = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(2, 1), getMapAttr(0, 1)});
  const ArrayAttr matmul_indexing_maps_transposed = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(1, 2), getMapAttr(0, 1)});
  const auto indexing_maps = contraction_op.getIndexingMaps();
  if (indexing_maps != matmul_indexing_maps &&
      indexing_maps != matmul_indexing_maps_transposed) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul or unsupported indexing_maps");
  }
  const bool transpose_rhs = indexing_maps == matmul_indexing_maps_transposed;

  const ArrayAttr matmul_iterator_types =
      builder.getArrayAttr({builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::reduction)});
  if (contraction_op->getAttr("iterator_types") != matmul_iterator_types) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
  }
  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      contraction_op->getAttrOfType<tpu::ContractPrecisionAttr>("precision");
  auto matmul_op = builder.create<tpu::MatmulOp>(
      contraction_op->getLoc(), acc_ty, lhs, rhs, acc,
      /*transpose_lhs=*/false, transpose_rhs, precision_attr);
  contraction_op.replaceAllUsesWith(matmul_op.getResult());
  contraction_op.erase();
  auto result = tpu_matmul_rule(matmul_op);
  return result;
}

using canonicalize_rule_type = std::function<LogicalResult(Operation &op)>;

const llvm::StringMap<canonicalize_rule_type> &rules() {
  static auto rules = new llvm::StringMap<canonicalize_rule_type>{
      {tpu::MatmulOp::getOperationName(), canonicalize_matmul},
      {vector::ContractionOp::getOperationName(), canonicalize_contraction}};
  return *rules;
}

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer() {}

  LogicalResult canonicalize(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return canonicalizeBlock(op.getBody().front());
  }

  LogicalResult canonicalizeBlock(Block &block) {
    // make_early_inc_range is utilized due to op mutation.
    for (Operation &any_op : make_early_inc_range(block)) {
      if (canonicalizeOp(any_op).failed()) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult canonicalizeOp(Operation &any_op) {
    // We must iterate over the op first, because canonicalization can cause
    // us to .erase() an op, and accessing getRegions on it after is not sound.
    // Invariant - top level ops with regions may never be invalidated.
    for (Region &region : any_op.getRegions()) {
      for (Block &block : region) {
        if (canonicalizeBlock(block).failed()) {
          return failure();
        }
      }
    }
    if (auto rule_it = rules().find(any_op.getName().getStringRef());
        rule_it != rules().end()) {
      const canonicalize_rule_type &rule = rule_it->getValue();
      return rule(any_op);
    }
    return success();
  }
};

struct CanonicalizeMosaicPass
    : public impl::CanonicalizeMosaicPassBase<CanonicalizeMosaicPass> {
  CanonicalizeMosaicPass() {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc;
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };
};

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass() {
  return std::make_unique<CanonicalizeMosaicPass>();
}

}  // namespace mlir::tpu
