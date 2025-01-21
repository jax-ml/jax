#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// It requires these headers, but does not include them.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/include/mlir/IR/AffineExpr.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/vreg_util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_CANONICALIZEMOSAICPASS
#define GEN_PASS_DEF_CANONICALIZEMOSAICPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

struct CanonicalizeContext {
  // see Note: Compatibility mode
  bool compatibility_mode;

  int hardware_generation;
};

LogicalResult tpu_matmul_rule(const CanonicalizeContext &ctx,
                              tpu::MatmulOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

  auto transpose_lhs = op.getTransposeLhs();
  auto transpose_rhs = op.getTransposeRhs();

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto acc = op.getAcc();

  const VectorType lhs_ty = lhs.getType();
  const VectorType rhs_ty = rhs.getType();
  const VectorType acc_ty = acc.getType();

  auto lhs_element_type = lhs_ty.getElementType();
  auto rhs_element_type = rhs_ty.getElementType();
  auto acc_element_type = acc_ty.getElementType();

  // there are a few primary paths for dimension_numbers in matmul
  // 1) No dimension numbers provided -> set to default
  // 2) defined and not default -> verify and apply
  // 3) defined and matching defaultDimensionNumbers -> no-op for
  // canonicalization of dims
  std::optional<int64_t> batch_size = std::nullopt;

  // MKN matmul - no dims or transpositions set
  if (!op.getDimensionNumbers().has_value()) {
    // Legacy API - convert it to dimension numbers
    op.setDimensionNumbersAttr(
        defaultDimensionNumbers(builder, transpose_lhs, transpose_rhs));
  } else if (
      // Dot dim API - dimensions are provided and are not default
      (op.getDimensionNumbers().value() !=
       defaultDimensionNumbers(builder, false, false))) {
    auto dimension_numbers = op.getDimensionNumbers();
    auto lhs_contracting_dims = dimension_numbers->getLhsContractingDims();
    auto rhs_contracting_dims = dimension_numbers->getRhsContractingDims();

    auto lhs_batch_dims = dimension_numbers->getLhsBatchDims();
    auto rhs_batch_dims = dimension_numbers->getRhsBatchDims();

    // Invariant in matmul verifier: <= 1 batch dim atm, and that lhs and rhs
    // are the same
    // Invariant in matmul verifier: Exactly one contracting and non contracting
    // dim in each of lhs and rhs for now.
    batch_size =
        lhs_batch_dims.empty()
            ? std::nullopt
            : std::optional<int64_t>(lhs_ty.getShape()[lhs_batch_dims[0]]);
    // Lower each dim in contracting dims by size(batch_dims)
    auto batch_adjusted_lhs_contracting_dim =
        lhs_contracting_dims[0] - lhs_batch_dims.size();
    auto batch_adjusted_rhs_contracting_dim =
        rhs_contracting_dims[0] - rhs_batch_dims.size();

    if (batch_adjusted_lhs_contracting_dim != 1) {
      transpose_lhs = true;
    }
    if (batch_adjusted_rhs_contracting_dim != 0) {
      transpose_rhs = true;
    }
  }

  auto extsi_sitofp = [&builder, &op](TypedValue<VectorType> element) {
    const VectorType ty = element.getType();
    auto shape = ty.getShape();
    CHECK(ty.getElementType().isInteger());
    TypedValue<VectorType> ext_ele;
    if (ty.getElementType().getIntOrFloatBitWidth() == 32) {
      ext_ele = element;
    } else {
      ext_ele = cast<TypedValue<VectorType>>(
          builder
              .create<arith::ExtSIOp>(
                  VectorType::get(shape, builder.getI32Type()), element)
              .getResult());
    }
    // TODO(mvoz): Go to bf16 when hardware supported, requires adding support
    // for 16 bitwidth in extsiop in infer/apply.
    auto ele_as_fp = builder.create<arith::SIToFPOp>(
        op.getLoc(), VectorType::get(shape, builder.getF32Type()), ext_ele);
    return ele_as_fp;
  };

  if (lhs_element_type != rhs_element_type) {
    if (!ctx.compatibility_mode) {
      return op->emitOpError(
          "Mosaic matmul invoked with mixed element types, but compatibility "
          "mode is disabled.");
    }
    if (lhs_element_type.isInteger() && rhs_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/int matmul.
      op->emitOpError("Mix int/int - NYI");
      return failure();
    }
    if (acc_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/float matmul with int acc.
      // Should be pretty straightforward.
      op->emitOpError("acc is int in mixed matmul. Expected float.");
      return failure();
    }
    if (lhs_element_type.isInteger()) {
      auto float_lhs = extsi_sitofp(lhs);
      op->setOperand(0, float_lhs);
      lhs = cast<TypedValue<VectorType>>(float_lhs.getResult());
    }
    if (rhs_element_type.isInteger()) {
      auto float_rhs = extsi_sitofp(rhs);
      op->setOperand(1, float_rhs);
      rhs = cast<TypedValue<VectorType>>(float_rhs.getResult());
    }
  }
  // TODO(mvoz): Add more invariants.
  if (acc_element_type.isInteger()) {
    if (!op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float lhs. Expected int lhs.");
      return failure();
    }
    if (!op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float rhs. Expected int rhs.");
      return failure();
    }
  } else {
    if (op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int lhs. Expected float lhs.");
      return failure();
    }
    if (op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int rhs. Expected float rhs.");
      return failure();
    }
  }

  auto dot_dim_matmul = [&](auto lhs, auto rhs, auto acc) {
    auto precision_attr = op.getPrecisionAttr();

    // If we are transposing the lhs, we need to transpose the lhs before
    // matmul here, as we don't have lhs fusion implemented in apply.
    if (transpose_lhs) {
      auto lhs_ty = cast<VectorType>(lhs.getType());
      auto rank = lhs_ty.getShape().size();

      // This transposition must run on vectors with rank >= 2
      CHECK_GE(rank, 2);

      std::vector<int64_t> perm(rank);
      std::iota(perm.begin(), perm.end(), 0);
      std::swap(perm[rank - 2], perm[rank - 1]);

      std::vector<int64_t> shape(lhs_ty.getShape());
      std::swap(shape[rank - 2], shape[rank - 1]);

      auto lhs_ty_transposed = VectorType::get(shape, lhs_ty.getElementType());

      const SmallVector<int64_t> perm_vec =
          SmallVector<int64_t>(perm.begin(), perm.end());
      lhs = builder.create<vector::TransposeOp>(
          lhs_ty_transposed, lhs,
          DenseI64ArrayAttr::get(builder.getContext(), perm_vec));
    }
    auto ddn = defaultDimensionNumbers(builder, /*transpose_lhs=*/false,
                                       transpose_rhs);
    // transpose flags are always false here, because ddn takes precedence
    // after this pass.
    auto matmul_res = builder.create<tpu::MatmulOp>(
        op.getLoc(), acc.getType(), lhs, rhs, acc,
        /*transpose_lhs=*/false,
        /*transpose_rhs=*/false, precision_attr, ddn);
    return matmul_res;
  };

  // If we have a batch_size, we want to slice rhs and lhs [:batch_size],
  // and then do O[i] = A[i] @ B[i]
  // Produce an output shape of [batch_size, m, n]
  if (batch_size.has_value()) {
    std::vector<Value> outputs;

    for (int64_t i = 0; i < batch_size; ++i) {
      auto sliced_lhs = builder.create<vector::ExtractOp>(op.getLoc(), lhs,
                                                          ArrayRef<int64_t>{i});
      auto sliced_rhs = builder.create<vector::ExtractOp>(op.getLoc(), rhs,
                                                          ArrayRef<int64_t>{i});

      auto sliced_acc = builder.create<vector::ExtractOp>(op.getLoc(), acc,
                                                          ArrayRef<int64_t>{i});

      auto matmul_res =
          dot_dim_matmul(sliced_lhs.getResult(), sliced_rhs.getResult(),
                         sliced_acc.getResult());
      auto res_ty = matmul_res.getType().cast<VectorType>();
      auto res_shape = res_ty.getShape();
      // reshape to 1x[prior_shape]
      auto reshape_shape = llvm::to_vector(res_shape);
      reshape_shape.insert(reshape_shape.begin(), 1);
      auto shape_cast = builder.create<vector::ShapeCastOp>(
          op.getLoc(), VectorType::get(reshape_shape, res_ty.getElementType()),
          matmul_res);
      outputs.push_back(shape_cast);
    }
    // Technically almost identical to the case where batch_size is 1, but
    // we want to avoid the spurious concat here.
    if (batch_size == 1) {
      op.replaceAllUsesWith(outputs[0]);
      op.erase();
      return success();
    }
    auto output = builder
                      .create<tpu::ConcatenateOp>(op.getLoc(), acc_ty, outputs,
                                                  /*dimension=*/0)
                      .getResult();
    op.replaceAllUsesWith(output);
    op.erase();
  } else {
    auto matmul_res = dot_dim_matmul(lhs, rhs, acc).getResult();
    op.replaceAllUsesWith(matmul_res);
    op.erase();
  }
  return success();
};

LogicalResult canonicalize_elementwise(const CanonicalizeContext &ctx,
                                       Operation &op) {
  OpBuilder builder(&op);
  auto operands = op.getOperands();
  auto res_ty = dyn_cast<VectorType>(op.getResult(0).getType());
  if (op.getNumResults() != 1) {
    op.emitOpError("Invariant violated: Unexpected number of results");
    return failure();
  }
  if (!res_ty) {
    // scalar
    // TODO(mvoz): Add canonicalization and invariants for scalar elementwise
    // ops.
    return success();
  }
  auto shape = res_ty.getShape();
  std::vector<Value> new_operands;
  new_operands.reserve(operands.size());

  bool should_rewrite_op = false;
  auto target_f32_ty = VectorType::get(shape, builder.getF32Type());
  for (int i = 0; i < operands.size(); ++i) {
    auto operand = operands[i];
    auto ty = dyn_cast<VectorType>(operand.getType());
    if (ty) {
      if (ty.getShape() != shape) {
        // Should already be checked my MLIR verification, but let's be safe.
        op.emitOpError("Mismatched shapes in elementwise op.");
        return failure();
      }
      auto element_type = ty.getElementType();
      // There's an annoying hodgepodge of elementwise ops that need to be
      // rewritten to f32 on later hardware.
      // TODO(mvoz): Look into (1) what it would take to support these ops
      // natively on later hardware, and (2) how to better organize this list.
      bool needs_cast = ctx.hardware_generation <= 5 || isa<math::PowFOp>(op) ||
                        isa<math::TanhOp>(op) || isa<math::ExpOp>(op) ||
                        isa<math::LogOp>(op);
      if (needs_cast && element_type.isBF16()) {
        if (ctx.compatibility_mode) {
          auto target_f32 =
              builder.create<arith::ExtFOp>(op.getLoc(), target_f32_ty, operand)
                  .getResult();
          should_rewrite_op = true;
          new_operands.push_back(target_f32);
        } else {
          op.emitOpError(
              "Compatibility mode disabled. Unsupported element type in "
              "elementwise op on hardware generation: ")
              << ctx.hardware_generation
              << ". Use hardware generation after 5 or cast to f32.";
          return failure();
        }
      } else {
        new_operands.push_back(operand);
      }
    } else {
      // Should already be checked my MLIR verification, but let's be safe.
      op.emitOpError("MLIR unsupported - mix scalar and vec elementwise ops");
      return failure();
    }
  }
  if (should_rewrite_op) {
    auto result_ty = dyn_cast<VectorType>(op.getResult(0).getType());
    if (!result_ty) {
      op.emitOpError("Not implemented: Unexpected result type");
      return failure();
    }
    auto result_element_type = result_ty.getElementType();
    if (!result_element_type.isF32() && !result_element_type.isBF16()) {
      op.emitOpError("Not implemented: Unexpected result element type");
      return failure();
    }
    // Do the new op in f32, then truncate to the original element type.
    auto new_op = builder.create(op.getLoc(), op.getName().getIdentifier(),
                                 new_operands, target_f32_ty);
    new_op = builder.create<arith::TruncFOp>(op.getLoc(), res_ty,
                                             new_op->getResult(0));
    op.replaceAllUsesWith(new_op);
    op.erase();
  }
  return success();
}

LogicalResult canonicalize_multi_dim_reduction(const CanonicalizeContext &ctx,
                                               Operation &operation) {
  ImplicitLocOpBuilder builder(operation.getLoc(), &operation);
  auto op = cast<vector::MultiDimReductionOp>(operation);
  auto source_ty = op.getSourceVectorType();
  auto result_ty = dyn_cast<VectorType>(op.getDestType());
  if (!result_ty) {
    return op->emitOpError() << "Only vector reductions supported";
  }

  auto element_type = source_ty.getElementType();
  if (element_type.isF32()) {
    return success();
  } else if (element_type.isBF16()) {
    bool reduces_sublanes = false;
    for (int64_t dim : op.getReductionDims()) {
      if (dim == source_ty.getRank() - 2) {
        reduces_sublanes = true;
      }
    }
    if (ctx.hardware_generation <= 5) {
      auto new_source = builder.create<arith::ExtFOp>(
          VectorType::get(source_ty.getShape(), builder.getF32Type()),
          op.getSource());

      auto result_ty_f32 =
          VectorType::get(result_ty.getShape(), builder.getF32Type());
      auto acc_ext = builder.create<arith::ExtFOp>(result_ty_f32, op.getAcc());
      Value new_acc = acc_ext.getResult();
      // Try to constant fold.
      if (auto const_acc = op.getAcc().getDefiningOp<arith::ConstantOp>()) {
        auto result =
            acc_ext.fold(arith::ExtFOp::FoldAdaptor(const_acc.getValue()));
        if (!result.isNull() && result.is<Attribute>()) {
          acc_ext->erase();
          new_acc = builder.create<arith::ConstantOp>(
              op.getLoc(), result_ty_f32,
              cast<TypedAttr>(result.get<Attribute>()));
        }
      }
      auto new_op = builder.create<vector::MultiDimReductionOp>(
          op.getLoc(), new_acc.getType(), op.getKindAttr(), new_source, new_acc,
          DenseI64ArrayAttr::get(builder.getContext(), op.getReductionDims()));
      auto new_result = builder.create<arith::TruncFOp>(op.getLoc(), result_ty,
                                                        new_op.getResult());
      op.replaceAllUsesWith(new_result.getResult());
      op.erase();
    }
    return success();
  } else if (element_type.isSignlessInteger(32) &&
             // TODO(b/384774084): Add support for u32 reductions.
             (op.getKind() == vector::CombiningKind::ADD ||
              op.getKind() == vector::CombiningKind::MAXSI ||
              op.getKind() == vector::CombiningKind::MINSI)) {
    return success();
  }
  op.emitOpError("Unsupported element type for the selected reduction");
  return failure();
}

LogicalResult canonicalize_matmul(const CanonicalizeContext &ctx,
                                  Operation &op) {
  auto matmul_op = dyn_cast<tpu::MatmulOp>(op);
  if (!matmul_op) {
    op.emitOpError("Invariant violated: Not a matmul");
    return failure();
  }
  return tpu_matmul_rule(ctx, matmul_op);
};

LogicalResult canonicalize_contraction(const CanonicalizeContext &ctx,
                                       Operation &op) {
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

  const auto dot_dimension_numbers_attr =
      defaultDimensionNumbers(builder, false, transpose_rhs);

  auto matmul_op = builder.create<tpu::MatmulOp>(
      contraction_op->getLoc(), acc_ty, lhs, rhs, acc,
      /*transpose_lhs=*/false,
      /*transpose_rhs=*/false, precision_attr, dot_dimension_numbers_attr);
  contraction_op.replaceAllUsesWith(matmul_op.getResult());
  contraction_op.erase();
  auto result = tpu_matmul_rule(ctx, matmul_op);
  return result;
}

LogicalResult canonicalize_extract(const CanonicalizeContext &ctx,
                                   Operation &raw_op) {
  auto op = dyn_cast<vector::ExtractOp>(raw_op);
  Type result_ty = op.getResult().getType();
  if (!isa<VectorType>(result_ty)) {
    bool is_supported = result_ty.isSignlessIntOrFloat() &&
                        result_ty.getIntOrFloatBitWidth() == 32;
    if (!is_supported) {
      return op.emitOpError(
          "Only 32-bit scalar vector.extracts supported. Cast your input to a "
          "32-bit type first.");
    }
  }
  return success();
}

LogicalResult canonicalize_select(const CanonicalizeContext &ctx,
                                  Operation &raw_op) {
  auto op = dyn_cast<arith::SelectOp>(raw_op);
  if (!isa<VectorType>(op.getType()) ||
      isa<VectorType>(op.getCondition().getType())) {
    return success();
  }
  // Canonicalize `i1 ? v1 : v2` -> `broadcast(i1) ? v1 : v2`.
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto cond_ty = VectorType::get(cast<VectorType>(op.getType()).getShape(),
                                 op.getCondition().getType());
  auto cond = builder.create<vector::BroadcastOp>(cond_ty, op.getCondition());
  auto new_op = builder.create<arith::SelectOp>(
      op.getLoc(), cond, op.getTrueValue(), op.getFalseValue());
  op.replaceAllUsesWith(new_op.getResult());
  op.erase();
  return success();
}

// All conversions that change bitwidth must be canonicalized to tpu.fptosi.
LogicalResult canonicalize_fptosi(const CanonicalizeContext &ctx,
                                  Operation &raw_op) {
  auto op = cast<arith::FPToSIOp>(raw_op);
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto src_vty = dyn_cast<VectorType>(op.getIn().getType());
  auto dst_vty = dyn_cast<VectorType>(op.getType());
  if (static_cast<bool>(src_vty) != static_cast<bool>(dst_vty)) {
    return op.emitOpError("Vector/scalar mismatch between input and output");
  }
  bool is_vector = static_cast<bool>(src_vty);
  unsigned src_bitwidth, dst_bitwidth;
  if (is_vector) {
    src_bitwidth = src_vty.getElementTypeBitWidth();
    dst_bitwidth = dst_vty.getElementTypeBitWidth();
  } else {
    src_bitwidth = op.getIn().getType().getIntOrFloatBitWidth();
    dst_bitwidth = op.getType().getIntOrFloatBitWidth();
  }
  if (dst_bitwidth > 32) {
    return op.emitOpError("Target bitwidth too large");
  }
  // We have low-level optimized code for bf16->s8 and bf16->s4 casts on v6.
  if (ctx.hardware_generation >= 6 && is_vector &&
      src_vty.getElementType().isBF16() &&
      (dst_vty.getElementType().isSignlessInteger(8) ||
       dst_vty.getElementType().isSignlessInteger(4))) {
    auto new_op = builder.create<tpu::FPToSIOp>(
        op.getType(), op.getIn(), tpu::RoundingMode::kTowardsZero);
    op.replaceAllUsesWith(new_op.getResult());
    op.erase();
    // We briefly trigger canonicalization here to potentially fuse the rounding
    // ops into the newly created tpu.fptosi.
    {
      PatternRewriter rewriter(new_op.getContext());
      rewriter.setInsertionPoint(new_op);
      // We don't care if the canonicalization pattern matched or not.
      (void)tpu::FPToSIOp::canonicalize(new_op, rewriter);
      new_op = nullptr;  // Canonicalization may have erased the op!
    }
    return success();
  }
  Value x = op.getIn();
  // Upcast the input to f32.
  if (src_bitwidth < 32) {
    if (is_vector) {
      x = builder.create<arith::ExtFOp>(
          VectorType::get(src_vty.getShape(), builder.getF32Type()), x);
    } else {
      x = builder.create<arith::ExtFOp>(builder.getF32Type(), x);
    }
  }
  if (dst_bitwidth < 32) {
    if (!ctx.compatibility_mode) {
      return op.emitOpError(
          "On this target only float-to-integer conversions can only happen on "
          "32-bit values. Enable compatibility mode or upcast to float32.");
    }
    // Need to clip values to match XLA
    auto clip = [&](Value x, Value low, Value high) {
      auto is_small =
          builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, x, low);
      x = builder.create<arith::SelectOp>(is_small, low, x);
      auto is_large =
          builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, x, high);
      x = builder.create<arith::SelectOp>(is_large, high, x);
      return x;
    };
    auto minval = builder.getF32FloatAttr(
        APInt::getSignedMinValue(dst_bitwidth).getSExtValue());
    auto maxval = builder.getF32FloatAttr(
        APInt::getSignedMaxValue(dst_bitwidth).getSExtValue());
    if (is_vector) {
      auto x_vty = cast<VectorType>(x.getType());
      x = clip(x, getFullVector(builder, x_vty, minval),
               getFullVector(builder, x_vty, maxval));
    } else {
      auto f32 = builder.getF32Type();
      x = clip(x, builder.create<arith::ConstantOp>(f32, minval),
               builder.create<arith::ConstantOp>(f32, maxval));
    }
  }
  if (is_vector) {
    x = builder.create<arith::FPToSIOp>(
        VectorType::get(src_vty.getShape(), builder.getI32Type()), x);
  } else {
    x = builder.create<arith::FPToSIOp>(builder.getI32Type(), x);
  }
  if (dst_bitwidth < 32) {
    if (!ctx.compatibility_mode) {
      return op.emitOpError(
          "On this target only float-to-integer conversions can only happen on "
          "32-bit values. Enable compatibility mode or cast to int32 and "
          "truncate later.");
    }
    x = builder.create<arith::TruncIOp>(op.getType(), x);
  }
  op.replaceAllUsesWith(x);
  op.erase();
  return success();
}

LogicalResult canonicalize_repeat(const CanonicalizeContext &ctx,
                                  Operation &raw_op) {
  auto op = dyn_cast<tpu::RepeatOp>(raw_op);
  if (!isa<VectorType>(op.getType())) {
    return op.emitOpError("Only vector types supported");
  }
  auto operand = op.getSource();
  auto times = op.getTimes();
  if (times == 1) {
    // A true no op - kind of an odd edge case, but this does come up in
    // flash_attention_backward tests.
    op.replaceAllUsesWith(operand);
    op.erase();
    return success();
  }
  auto operands = std::vector<Value>(times, operand);
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto concat = builder.create<tpu::ConcatenateOp>(op.getLoc(), op.getType(),
                                                   operands, op.getDimension());
  op.replaceAllUsesWith(concat.getResult());
  op.erase();
  return success();
}

using canonicalize_rule_type =
    std::function<LogicalResult(const CanonicalizeContext &ctx, Operation &op)>;

const llvm::StringMap<canonicalize_rule_type> &rules() {
  static auto rules = new llvm::StringMap<canonicalize_rule_type>{
      {tpu::MatmulOp::getOperationName(), canonicalize_matmul},
      {vector::ContractionOp::getOperationName(), canonicalize_contraction},
      {vector::ExtractOp::getOperationName(), canonicalize_extract},
      {vector::MultiDimReductionOp::getOperationName(),
       canonicalize_multi_dim_reduction},
      {arith::SelectOp::getOperationName(), canonicalize_select},
      {arith::FPToSIOp::getOperationName(), canonicalize_fptosi},
      {tpu::RepeatOp::getOperationName(), canonicalize_repeat}};
  return *rules;
}

const llvm::StringSet<> &elementwise_convertible_ops() {
  static auto ops = new llvm::StringSet<>{arith::MulFOp::getOperationName(),
                                          arith::DivFOp::getOperationName(),
                                          arith::AddFOp::getOperationName(),
                                          arith::SubFOp::getOperationName(),
                                          arith::MaximumFOp::getOperationName(),
                                          arith::MinimumFOp::getOperationName(),
                                          math::PowFOp::getOperationName(),
                                          math::TanhOp::getOperationName(),
                                          math::ExpOp::getOperationName(),
                                          math::LogOp::getOperationName()};
  return *ops;
}

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer(int hardware_generation, bool compatibility_mode)
      : hardware_generation_(hardware_generation),
        compatibility_mode_(compatibility_mode) {}

  int hardware_generation_;
  bool compatibility_mode_;

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
    CanonicalizeContext ctx({compatibility_mode_, hardware_generation_});
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
    if (elementwise_convertible_ops().contains(
            any_op.getName().getStringRef())) {
      return canonicalize_elementwise(ctx, any_op);
    }
    if (auto rule_it = rules().find(any_op.getName().getStringRef());
        rule_it != rules().end()) {
      const canonicalize_rule_type &rule = rule_it->getValue();
      return rule(ctx, any_op);
    }
    return success();
  }
};

struct CanonicalizeMosaicPass
    : public impl::CanonicalizeMosaicPassBase<CanonicalizeMosaicPass> {
  CanonicalizeMosaicPass(int hardware_generation_p, bool compatibility_mode_p)
      : compatibility_mode_(compatibility_mode_p) {
    this->hardware_generation = hardware_generation_p;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc(hardware_generation, compatibility_mode_);
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };

  bool compatibility_mode_;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass(
    int hardware_generation, bool compatibility_mode) {
  return std::make_unique<CanonicalizeMosaicPass>(hardware_generation,
                                                  compatibility_mode);
}

}  // namespace mlir::tpu
