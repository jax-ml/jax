#include "jaxlib/mosaic/dialect/tpu/transforms/relayout_insertion.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "absl/log/check.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/Support/MathExtras.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/Diagnostics.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_RELAYOUTINSERTIONPASS
#define GEN_PASS_DEF_RELAYOUTINSERTIONPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

FailureOr<TypedValue<VectorType>> relayout(OpBuilder &builder,
                                           TypedValue<VectorType> v,
                                           VectorLayout src, VectorLayout dst) {
  // change bitwidth
  if (v.getType().getElementType() == builder.getI1Type() &&
      // TODO(jevinjiang): for other relayout changes (tiling, offsets, implicit
      // dim), we currently rely on apply-vector-layout pass to do the relayout.
      src.bitwidth() != dst.bitwidth()) {
    CHECK(llvm::isPowerOf2_32(src.bitwidth()));
    CHECK(llvm::isPowerOf2_32(dst.bitwidth()));
    auto makeVectorType = [&](int bitwidth) {
      return VectorType::get(v.getType().getShape(),
                             builder.getIntegerType(bitwidth));
    };
    auto makeConstant = [&](int val, VectorLayout layout) {
      auto vty = makeVectorType(layout.bitwidth());
      auto constant_op = builder.create<arith::ConstantOp>(
          v.getLoc(),
          DenseElementsAttr::get(
              vty,
              builder.getIntegerAttr(vty.getElementType(), val)));
      setOutLayout(constant_op,
                   VectorLayout(layout.bitwidth(), {std::nullopt, std::nullopt},
                                layout.tiling(), layout.implicit_dim()));
      return constant_op;
    };
    auto makeSrcLayout = [&](int bitwidth) {
      return VectorLayout(bitwidth, src.offsets(), src.tiling(),
                          src.implicit_dim());
    };
    auto select_src_layout = makeSrcLayout(src.bitwidth());
    auto select_src_op = builder.create<arith::SelectOp>(
        v.getLoc(), v, makeConstant(1, select_src_layout),
        makeConstant(0, select_src_layout));
    setLayout(select_src_op,
              {select_src_layout, select_src_layout, select_src_layout},
              select_src_layout);
    Operation *cast_op = nullptr;
    auto cast_layout = makeSrcLayout(dst.bitwidth());
    // TODO(jevinjiang): some conversion might not be supported in HW.
    if (dst.bitwidth() > src.bitwidth()) {
      cast_op = builder.create<arith::ExtSIOp>(
          v.getLoc(), makeVectorType(dst.bitwidth()), select_src_op);
    } else {
      cast_op = builder.create<arith::TruncIOp>(
          v.getLoc(), makeVectorType(dst.bitwidth()), select_src_op);
    }
    setLayout(cast_op, select_src_layout, cast_layout);
    auto select_cast_layout = makeSrcLayout(dst.bitwidth());
    auto select_cast_op = builder.create<arith::CmpIOp>(
        v.getLoc(), v.getType(), arith::CmpIPredicate::eq,
        cast_op->getResult(0), makeConstant(1, select_cast_layout));
    setLayout(select_cast_op, {select_cast_layout, select_cast_layout},
              select_cast_layout);
    return cast<TypedValue<VectorType>>(select_cast_op.getResult());
  }
  return v;
}

// TODO(jevinjiang): Migrate relayout from apply-vector-layout pass to
// this pass. Unlike relayout in apply-vector-layout pass, we don't need to
// unroll vectors while inserting relayout ops, instead we directly insert the
// necessary ops with the correct vector layouts for relayout and rely on
// apply-vector-layout pass to do the unrolling. After migration, we should
// expect no relayout will be triggered in apply-vector-layout pass.
LogicalResult insertRelayout(Operation &op,
                             const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> in_layouts,
                             getInLayouts(op, target_shape));
  if (in_layouts.size() != op.getNumOperands()) {
    return op.emitError("Expected the same number of operands as in_layouts");
  }
  if (isa<tpu::AssumeLayoutOp>(op)) {
    return success();
  }
  // Relayout the operands, if their requested input layouts don't match the
  // layouts in which they were produced.
  for (auto [idx, tup] :
       llvm::enumerate(llvm::zip(op.getOperands(), in_layouts))) {
    auto [operand, li] = tup;
    auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand);
    TPU_ASSERT_EQ_OP(vector_operand != nullptr, li.has_value());
    if (vector_operand == nullptr) {
      continue;
    }
    // The operand should always be an Operation (and not a BlockArgument)
    // since we expect the FuncOp to have only memrefs and semaphores as
    // arguments.
    auto op_result = dyn_cast<OpResult>(vector_operand);
    if (op_result == nullptr) {
      return op.emitError("Expected operand to be an operation result");
    }
    Operation *const def_op = op_result.getOwner();
    TPU_ASSERT_OP(def_op);
    const unsigned res_idx = op_result.getResultNumber();
    FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                               getOutLayouts(*def_op, target_shape));
    const Layout lo = def_layouts[res_idx];
    TPU_ASSERT_OP(lo.has_value());
    if (*lo == *li) {
      continue;
    }
    OpBuilder builder(&op);
    FAILUREOR_ASSIGN_OR_RETURN(Value new_v,
                               relayout(builder, vector_operand, /*src=*/*lo,
                                        /*dst=*/*li));
    op.setOperand(idx, new_v);
  }
  return success();
}

struct RelayoutInsertionPass
    : public impl::RelayoutInsertionPassBase<RelayoutInsertionPass> {
  RelayoutInsertionPass(std::array<int64_t, 2> target_shape) {
    this->sublane_count = target_shape[0];
    this->lane_count = target_shape[1];
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto result = func.walk([&](Operation *op) {
      if (insertRelayout(*op, {sublane_count, lane_count}).failed()) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
  }
};


}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRelayoutInsertionPass(
    std::array<int64_t, 2> target_shape) {
  return std::make_unique<RelayoutInsertionPass>(target_shape);
}

}  // namespace mlir::tpu