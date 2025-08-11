/* Copyright 2024 The JAX Authors.

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

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"
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

  std::array<int64_t, 2> target_shape;
};

using canonicalize_rule_type = std::function<FailureOr<Value>(
    const CanonicalizeContext &ctx, Operation &op)>;
const llvm::StringMap<canonicalize_rule_type> &rules();

class CanonicalBuilder : public ImplicitLocOpBuilder {
 public:
  CanonicalBuilder(const CanonicalizeContext &ctx, Location loc, Operation *op)
      : ImplicitLocOpBuilder(loc, op), ctx_(ctx), op_(op) {}

  template <typename Op, typename... Args>
  Value create(Location loc, Args &&...args) {
    Op new_op = OpBuilder::create<Op>(loc, std::forward<Args>(args)...);
    // We perform a one-level check to avoid infinite recursion when recreating
    // the canonicalized operation. However, if there is an op that in its
    // canonicalization rule creates another op and vice versa, it will still
    // lead to infinite loops.
    if (auto rule_it = rules().find(Op::getOperationName());
        !isa<Op>(op_) && rule_it != rules().end()) {
      const canonicalize_rule_type &rule = rule_it->getValue();
      FailureOr<Value> result = rule(ctx_, *new_op);
      // We should not be creating uncanonicalizable ops inside this pass.
      CHECK(succeeded(result));
      return *result;
    }
    if constexpr (Op::template hasTrait<OpTrait::ZeroResults>()) {
      // For ops with no results (like stores), the function must still
      // return a Value. Return a null Value as a placeholder. The caller
      // for a result-less op should not be using the return value.
      return nullptr;
    } else {
      return new_op.getResult();
    }
  }

  template <typename Op, typename... Args>
  Value create(Args &&...args) {
    return create<Op>(getLoc(), std::forward<Args>(args)...);
  }

 private:
  const CanonicalizeContext &ctx_;
  Operation *op_;
};

bool need_elementwise_canonicalization(const CanonicalizeContext &ctx,
                                       Operation &op);

// Returns the collapsed lhs, rhs, acc and the new dimension numbers if the
// non-contracting dims can be collapsed, otherwise returns std::nullopt.
std::optional<std::tuple<TypedValue<VectorType>, TypedValue<VectorType>,
                         TypedValue<VectorType>, tpu::DotDimensionNumbersAttr>>
collapse_matmul_non_contracting_dims(
    CanonicalBuilder &builder, TypedValue<VectorType> lhs,
    TypedValue<VectorType> rhs, TypedValue<VectorType> acc,
    const tpu::DotDimensionNumbersAttr &dimension_numbers) {
  // Collapse
  //
  // 1. [batch_dims, non_contracting_dims, contracting_dims] into
  //   [batch_dims, prod(non_contracting_dims), contracting_dims] or
  // 2. [batch_dims, contracting_dims, non_contracting_dims] into
  //   [batch_dims, contracting_dims, prod(non_contracting_dims)].
  //
  // Returns a tuple of [new_operand, new_non_contracting_dims,
  // new_contracting_dims]. new_operand is nullptr if the operand does not need
  // to be collapsed.
  // TODO(b/413194126): Some shapes will trigger unsupported
  // vector::ShapeCastOp.
  auto maybe_collapse_non_contracting_dims =
      [&](TypedValue<VectorType> operand,
          ArrayRef<int64_t> non_contracting_dims,
          ArrayRef<int64_t> contracting_dims, ArrayRef<int64_t> batch_dims)
      -> std::tuple<TypedValue<VectorType>, SmallVector<int64_t, 2>,
                    SmallVector<int64_t, 2>> {
    VectorType vty = operand.getType();
    auto shape = vty.getShape();
    bool batch_dims_are_front =
        batch_dims == ArrayRef<int64_t>(llvm::to_vector(
                          llvm::seq<int64_t>(0, batch_dims.size())));
    // Case 1.
    bool trailing_contracting_dims =
        contracting_dims ==
            ArrayRef<int64_t>(llvm::to_vector(llvm::seq<int64_t>(
                shape.size() - contracting_dims.size(), shape.size()))) &&
        non_contracting_dims ==
            ArrayRef<int64_t>(llvm::to_vector(llvm::seq<int64_t>(
                batch_dims.size(),
                batch_dims.size() + non_contracting_dims.size())));
    // Case 2.
    bool trailing_non_contracting_dims =
        non_contracting_dims ==
            ArrayRef<int64_t>(llvm::to_vector(llvm::seq<int64_t>(
                shape.size() - non_contracting_dims.size(), shape.size()))) &&
        contracting_dims ==
            ArrayRef<int64_t>(llvm::to_vector(llvm::seq<int64_t>(
                batch_dims.size(),
                batch_dims.size() + contracting_dims.size())));
    bool should_collapse_non_contracting_dims =
        batch_dims_are_front &&
        (trailing_contracting_dims || trailing_non_contracting_dims) &&
        non_contracting_dims.size() > 1;
    if (!should_collapse_non_contracting_dims) {
      return {nullptr, llvm::to_vector(non_contracting_dims),
              llvm::to_vector(contracting_dims)};
    }
    SmallVector<int64_t, 2> new_shape;
    auto batch_shape = shape.take_front(batch_dims.size());
    new_shape.append(batch_shape.begin(), batch_shape.end());
    SmallVector<int64_t, 2> contracting_sizes;
    for (int64_t contracting_dim : contracting_dims) {
      contracting_sizes.push_back(shape[contracting_dim]);
    }
    int64_t collapsed_dim_size = 1;
    for (int64_t non_contracting_dim : non_contracting_dims) {
      collapsed_dim_size *= shape[non_contracting_dim];
    }
    if (trailing_contracting_dims) {
      new_shape.push_back(collapsed_dim_size);
      new_shape.append(contracting_sizes.begin(), contracting_sizes.end());
    } else {
      new_shape.append(contracting_sizes.begin(), contracting_sizes.end());
      new_shape.push_back(collapsed_dim_size);
    }
    auto new_operand =
        cast<TypedValue<VectorType>>(builder.create<vector::ShapeCastOp>(
            VectorType::get(new_shape, vty.getElementType()), operand));
    SmallVector<int64_t, 2> new_non_contracting_dims, new_contracting_dims;
    if (trailing_non_contracting_dims) {
      // Case 2 - contracting dims are not changed and non contracting dims are
      // changed to the last dim.
      new_contracting_dims = llvm::to_vector(contracting_dims);
      new_non_contracting_dims.push_back(new_shape.size() - 1);
    } else {
      // Case 1 - non contracting dims are collapsed in the middle so all
      // contracting dims are moved forward by (non_contracting_dims.size() -
      // 1).
      new_non_contracting_dims.push_back(batch_dims.size());
      for (int64_t contracting_dim : contracting_dims) {
        new_contracting_dims.push_back(contracting_dim -
                                       (non_contracting_dims.size() - 1));
      }
    }
    return {new_operand, new_non_contracting_dims, new_contracting_dims};
  };

  auto [new_lhs, new_lhs_non_contracting_dims, new_lhs_contracting_dims] =
      maybe_collapse_non_contracting_dims(
          lhs, dimension_numbers.getLhsNonContractingDims(),
          dimension_numbers.getLhsContractingDims(),
          dimension_numbers.getLhsBatchDims());

  auto [new_rhs, new_rhs_non_contracting_dims, new_rhs_contracting_dims] =
      maybe_collapse_non_contracting_dims(
          rhs, dimension_numbers.getRhsNonContractingDims(),
          dimension_numbers.getRhsContractingDims(),
          dimension_numbers.getRhsBatchDims());

  // Nothing to collapse.
  if (!new_lhs && !new_rhs) {
    return std::nullopt;
  }

  // Overwrite the operands if they were collapsed. We're going to access the
  // new shapes below.
  lhs = new_lhs ? new_lhs : lhs;
  rhs = new_rhs ? new_rhs : rhs;

  SmallVector<int64_t, 2> new_output_dim_order;
  SmallVector<int64_t, 2> new_acc_shape;
  for (int64_t batch_dim : dimension_numbers.getLhsBatchDims()) {
    new_output_dim_order.push_back(0);
    new_output_dim_order.push_back(batch_dim);
    new_acc_shape.push_back(lhs.getType().getDimSize(batch_dim));
  }
  for (int64_t non_contracting_dim : new_lhs_non_contracting_dims) {
    new_output_dim_order.push_back(0);
    new_output_dim_order.push_back(non_contracting_dim);
    new_acc_shape.push_back(lhs.getType().getDimSize(non_contracting_dim));
  }
  for (int64_t non_contracting_dim : new_rhs_non_contracting_dims) {
    new_output_dim_order.push_back(1);
    new_output_dim_order.push_back(non_contracting_dim);
    new_acc_shape.push_back(rhs.getType().getDimSize(non_contracting_dim));
  }

  // Batch dims are always at the front of the lhs and rhs.
  tpu::DotDimensionNumbersAttr new_dimension_numbers =
      tpu::DotDimensionNumbersAttr::get(
          builder.getContext(), new_lhs_contracting_dims,
          new_rhs_contracting_dims, new_lhs_non_contracting_dims,
          new_rhs_non_contracting_dims, new_output_dim_order,
          dimension_numbers.getLhsBatchDims(),
          dimension_numbers.getRhsBatchDims());

  // Reshape acc too.
  auto new_acc =
      cast<TypedValue<VectorType>>(builder.create<vector::ShapeCastOp>(
          VectorType::get(new_acc_shape, acc.getType().getElementType()), acc));

  return std::make_tuple(lhs, rhs, new_acc, new_dimension_numbers);
}

FailureOr<Value> canonicalize_matmul(const CanonicalizeContext &ctx,
                                     Operation &raw_op) {
  auto op = cast<tpu::MatmulOp>(raw_op);
  CanonicalBuilder builder(ctx, op.getLoc(), op.getOperation());

  auto transpose_lhs = op.getTransposeLhs();
  auto transpose_rhs = op.getTransposeRhs();

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto acc = op.getAcc();

  const VectorType old_lhs_ty = lhs.getType();
  const VectorType old_rhs_ty = rhs.getType();
  const VectorType old_acc_ty = acc.getType();

  auto lhs_element_type = old_lhs_ty.getElementType();
  auto rhs_element_type = old_rhs_ty.getElementType();
  auto acc_element_type = old_acc_ty.getElementType();

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
    if (auto collapsed_operands_and_ddn = collapse_matmul_non_contracting_dims(
            builder, lhs, rhs, acc, *op.getDimensionNumbers())) {
      tpu::DotDimensionNumbersAttr new_dimension_numbers;
      std::tie(lhs, rhs, acc, new_dimension_numbers) =
          *collapsed_operands_and_ddn;
      op.setDimensionNumbersAttr(new_dimension_numbers);
    }

    auto dimension_numbers = op.getDimensionNumbers();
    auto lhs_contracting_dims = dimension_numbers->getLhsContractingDims();
    auto rhs_contracting_dims = dimension_numbers->getRhsContractingDims();

    auto lhs_batch_dims = dimension_numbers->getLhsBatchDims();
    auto rhs_batch_dims = dimension_numbers->getRhsBatchDims();

    // Invariant in matmul verifier: <= 1 batch dim atm, and that lhs and rhs
    // are the same
    // Invariant in matmul verifier: Exactly one contracting and non contracting
    // dim in each of lhs and rhs at the moment.
    batch_size = lhs_batch_dims.empty()
                     ? std::nullopt
                     : std::optional<int64_t>(
                           lhs.getType().getShape()[lhs_batch_dims[0]]);
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

  auto dimension_numbers = op.getDimensionNumbers();
  // We can lower more efficiently if it's a f32 matrix-vector dot:
  // [B, M, K] @ [B, K] or [B, M, K] @ [B, 1, K].
  // Note that even though TPUv6+ have native bf16 ALU ops support, it doesn't
  // seem profitable in this case because we want to accumulate and return f32
  // in the end anyway.
  // TODO(twsung): Perhaps allow non-32 bits accumulation.
  bool is_rhs_vector_like =
      (dimension_numbers->getRhsNonContractingDims().empty() ||
       (dimension_numbers->getRhsNonContractingDims().size() == 1 &&
        rhs.getType().getDimSize(
            dimension_numbers->getRhsNonContractingDims()[0]) == 1)) &&
      dimension_numbers->getRhsContractingDims() ==
          ArrayRef<int64_t>{
              static_cast<int64_t>(rhs.getType().getShape().size() - 1)};
  bool is_matrix_vector_dot = dimension_numbers->getLhsContractingDims() ==
                                  ArrayRef<int64_t>{static_cast<int64_t>(
                                      lhs.getType().getShape().size() - 1)} &&
                              is_rhs_vector_like && lhs_element_type.isF32() &&
                              rhs_element_type.isF32();
  // Make sure there is only one or zero (for rhs) non-contracting dim in each
  // of lhs and rhs after collapsing.
  if (dimension_numbers->getLhsNonContractingDims().size() != 1) {
    return op->emitOpError(
        "Not implemented: lhs non contracting dims must be an infix/suffix of "
        "the shape.");
  }
  if (dimension_numbers->getRhsNonContractingDims().size() != 1 &&
      !is_matrix_vector_dot) {
    return op->emitOpError(
        "Not implemented: 1) rhs non contracting dims must be an infix/suffix "
        "of the shape or 2) the contracting dim of lhs/rhs must be the last "
        "dim and rhs must be vector-like [B, K] or [B, 1, K].");
  }

  auto extsi_sitofp = [&builder, &op](TypedValue<VectorType> element) {
    const VectorType ty = element.getType();
    auto shape = ty.getShape();
    CHECK(ty.getElementType().isInteger());
    TypedValue<VectorType> ext_ele;
    if (ty.getElementType().getIntOrFloatBitWidth() == 32) {
      ext_ele = element;
    } else {
      ext_ele = cast<TypedValue<VectorType>>(builder.create<arith::ExtSIOp>(
          VectorType::get(shape, builder.getI32Type()), element));
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
      lhs = cast<TypedValue<VectorType>>(float_lhs);
    }
    if (rhs_element_type.isInteger()) {
      auto float_rhs = extsi_sitofp(rhs);
      op->setOperand(1, float_rhs);
      rhs = cast<TypedValue<VectorType>>(float_rhs);
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

  // Attempt to canonicalize matmul(x, transpose(y)) to a matmul with the
  // dimension numbers changed which will later be lowered into a more efficient
  // operation that fuses the transpose into the matmul.
  auto transpose_op =
      dyn_cast_if_present<tpu::TransposeOp>(rhs.getDefiningOp());
  if (!is_matrix_vector_dot && transpose_op && transpose_op->hasOneUse() &&
      dimension_numbers->getRhsContractingDims().size() == 1 &&
      dimension_numbers->getRhsNonContractingDims().size() == 1) {
    auto rhs_non_contracting_dim =
        dimension_numbers->getRhsNonContractingDims()[0];
    auto rhs_contracting_dim = dimension_numbers->getRhsContractingDims()[0];
    auto permutation = transpose_op.getPermutation();
    if (permutation[rhs_contracting_dim] == rhs_non_contracting_dim &&
        permutation[rhs_non_contracting_dim] == rhs_contracting_dim &&
        std::all_of(dimension_numbers->getRhsBatchDims().begin(),
                    dimension_numbers->getRhsBatchDims().end(),
                    [&](int64_t batch_dim) {
                      return permutation[batch_dim] == batch_dim;
                    })) {
      if (auto transpose_op_vector_operand =
              dyn_cast<TypedValue<VectorType>>(transpose_op.getOperand())) {
        // The transpose is DCE'ed away at a later point.
        rhs = transpose_op_vector_operand;
        transpose_rhs = !transpose_rhs;
      } else {
        return op->emitOpError("Unexpected operand type for transpose op.");
      }
    }
  }

  auto dot_dim_matmul = [&](Value lhs, Value rhs, Value acc) {
    auto precision_attr = op.getPrecisionAttr();
    auto lhs_ty = cast<VectorType>(lhs.getType());
    auto rhs_ty = cast<VectorType>(rhs.getType());
    auto acc_ty = cast<VectorType>(acc.getType());

    // If we are transposing the lhs, we need to transpose the lhs before
    // matmul here, as we don't have lhs fusion implemented in apply.
    if (transpose_lhs) {
      auto rank = lhs_ty.getShape().size();

      // This transposition must run on vectors with rank >= 2
      CHECK_GE(rank, 2);

      std::vector<int64_t> perm(rank);
      std::iota(perm.begin(), perm.end(), 0);
      std::swap(perm[rank - 2], perm[rank - 1]);

      std::vector<int64_t> shape(lhs_ty.getShape());
      std::swap(shape[rank - 2], shape[rank - 1]);

      VectorType lhs_ty_transposed =
          VectorType::get(shape, lhs_ty.getElementType());

      const SmallVector<int64_t> perm_vec =
          SmallVector<int64_t>(perm.begin(), perm.end());
      lhs = builder.create<tpu::TransposeOp>(lhs_ty_transposed, lhs, perm_vec);
    }

    // Matrix-vector dot can be lowered to multiply > reduce over last dim.
    if (is_matrix_vector_dot) {
      // rhs is always broadcastable to lhs, from [K] or [1, K] to [M, K].
      rhs = builder.create<vector::BroadcastOp>(
          VectorType::get(lhs_ty.getShape(), rhs_ty.getElementType()), rhs);
      auto multiply = builder.create<arith::MulFOp>(lhs, rhs);
      acc = builder.create<vector::ShapeCastOp>(
          VectorType::get(lhs_ty.getShape().drop_back(),
                          acc_ty.getElementType()),
          acc);
      auto res = builder.create<vector::MultiDimReductionOp>(
          vector::CombiningKind::ADD, multiply, acc,
          /*reduction_dims=*/
          ArrayRef<int64_t>{
              static_cast<int64_t>(lhs_ty.getShape().size() - 1)});
      auto res_ty = cast<VectorType>(res.getType());
      if (res_ty.getShape() != acc_ty.getShape()) {
        res = builder.create<vector::ShapeCastOp>(acc_ty, res);
      }
      return res;
    }

    auto ddn = defaultDimensionNumbers(builder, /*transpose_lhs=*/false,
                                       transpose_rhs);
    // transpose flags are always false here, because ddn takes precedence
    // after this pass.
    return builder.create<tpu::MatmulOp>(
        op.getLoc(), acc.getType(), lhs, rhs, acc,
        /*transpose_lhs=*/false,
        /*transpose_rhs=*/false, precision_attr, ddn);
  };

  // If we have a batch_size, we want to slice rhs and lhs [:batch_size],
  // and then do O[i] = A[i] @ B[i]
  // Produce an output shape of [batch_size, m, n]
  Value res;
  if (batch_size.has_value()) {
    std::vector<Value> outputs;

    for (int64_t i = 0; i < batch_size; ++i) {
      auto sliced_lhs =
          builder.create<vector::ExtractOp>(lhs, ArrayRef<int64_t>{i});
      auto sliced_rhs =
          builder.create<vector::ExtractOp>(rhs, ArrayRef<int64_t>{i});
      auto sliced_acc =
          builder.create<vector::ExtractOp>(acc, ArrayRef<int64_t>{i});

      auto matmul_res = dot_dim_matmul(sliced_lhs, sliced_rhs, sliced_acc);
      auto res_ty = cast<VectorType>(matmul_res.getType());
      auto res_shape = res_ty.getShape();
      // reshape to 1x[prior_shape]
      auto reshape_shape = llvm::to_vector(res_shape);
      reshape_shape.insert(reshape_shape.begin(), 1);
      auto shape_cast = builder.create<vector::ShapeCastOp>(
          VectorType::get(reshape_shape, res_ty.getElementType()), matmul_res);
      outputs.push_back(shape_cast);
    }
    // Technically almost identical to the case where batch_size is 1, but
    // we want to avoid the spurious concat here.
    if (batch_size == 1) {
      res = outputs[0];
    } else {
      res = builder.create<tpu::ConcatenateOp>(acc.getType(), outputs,
                                               /*dimension=*/0);
    }
  } else {
    res = dot_dim_matmul(lhs, rhs, acc);
  }

  // Reshape the result to the old one as dims might have been collapsed.
  if (res.getType() != old_acc_ty) {
    res = builder.create<vector::ShapeCastOp>(old_acc_ty, res);
  }
  op.replaceAllUsesWith(res);
  op.erase();
  return res;
};

FailureOr<Value> canonicalize_elementwise(const CanonicalizeContext &ctx,
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
      if (element_type.isBF16()) {
        if (ctx.compatibility_mode) {
          auto target_f32 =
              builder.create<tpu::ExtFOp>(op.getLoc(), target_f32_ty, operand)
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
    if (!res_ty) {
      op.emitOpError("Not implemented: Unexpected result type");
      return failure();
    }
    // Do the new op in f32, then truncate to the original element type if
    // needed. For example, result of arith::CmpF is i1 and doesn't need to be
    // truncated.
    bool should_truncate = !isa<arith::CmpFOp>(op);
    auto new_res_ty =
        VectorType::get(shape, should_truncate ? builder.getF32Type()
                                               : res_ty.getElementType());
    // NOTE: We don't canonicalize recursively here
    Operation *new_op =
        builder.create(op.getLoc(), op.getName().getIdentifier(), new_operands,
                       new_res_ty, op.getAttrs());
    if (should_truncate) {
      new_op = builder.create<tpu::TruncFOp>(op.getLoc(), res_ty,
                                             new_op->getResult(0),
                                             tpu::RoundingMode::kToNearestEven);
    }
    op.replaceAllUsesWith(new_op);
    op.erase();
    return new_op->getResult(0);
  }
  return op.getResult(0);
}

FailureOr<Value> canonicalize_multi_dim_reduction(
    const CanonicalizeContext &ctx, Operation &operation) {
  CanonicalBuilder builder(ctx, operation.getLoc(), &operation);
  auto op = cast<vector::MultiDimReductionOp>(operation);
  auto source_ty = op.getSourceVectorType();
  auto result_ty = dyn_cast<VectorType>(op.getDestType());
  if (!result_ty) {
    return op->emitOpError() << "Only vector reductions supported";
  }

  auto element_type = source_ty.getElementType();
  if (element_type.isF32()) {
    return operation.getResult(0);
  } else if (element_type.isBF16()) {
    bool reduces_sublanes = false;
    for (int64_t dim : op.getReductionDims()) {
      if (dim == source_ty.getRank() - 2) {
        reduces_sublanes = true;
      }
    }
    if (ctx.hardware_generation <= 5) {
      auto new_source = builder.create<tpu::ExtFOp>(
          VectorType::get(source_ty.getShape(), builder.getF32Type()),
          op.getSource());

      auto result_ty_f32 =
          VectorType::get(result_ty.getShape(), builder.getF32Type());
      // createOrFold does not trigger recursive canonicalization, but
      // extensions to f32 are always supported.
      Value new_acc =
          builder.createOrFold<tpu::ExtFOp>(result_ty_f32, op.getAcc());
      auto new_op = builder.create<vector::MultiDimReductionOp>(
          op.getLoc(), new_acc.getType(), op.getKindAttr(), new_source, new_acc,
          DenseI64ArrayAttr::get(builder.getContext(), op.getReductionDims()));
      auto new_result = builder.create<tpu::TruncFOp>(
          op.getLoc(), result_ty, new_op, tpu::RoundingMode::kToNearestEven);
      op.replaceAllUsesWith(new_result);
      op.erase();
      return new_result;
    }
    return operation.getResult(0);
  } else if (element_type.isSignlessInteger(32) &&
             // TODO(b/384774084): Add support for u32 reductions.
             (op.getKind() == vector::CombiningKind::ADD ||
              op.getKind() == vector::CombiningKind::MAXSI ||
              op.getKind() == vector::CombiningKind::MINSI)) {
    return operation.getResult(0);
  }
  return op.emitOpError("Unsupported element type for the selected reduction");
}

FailureOr<Value> canonicalize_contraction(const CanonicalizeContext &ctx,
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

  CanonicalBuilder builder(ctx, contraction_op->getLoc(),
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

  Value matmul = builder.create<tpu::MatmulOp>(
      contraction_op->getLoc(), acc_ty, lhs, rhs, acc,
      /*transpose_lhs=*/false,
      /*transpose_rhs=*/false, precision_attr, dot_dimension_numbers_attr);
  contraction_op.replaceAllUsesWith(matmul);
  contraction_op.erase();
  return matmul;
}

FailureOr<Value> canonicalize_extract(const CanonicalizeContext &ctx,
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
  return raw_op.getResult(0);
}

FailureOr<Value> canonicalize_broadcast(const CanonicalizeContext &ctx,
                                        Operation &raw_op) {
  auto op = dyn_cast<vector::BroadcastOp>(raw_op);
  auto src_ty = op.getSource().getType();
  auto src_vty = dyn_cast<VectorType>(src_ty);
  if ((src_vty && src_vty.getElementType().isSignlessInteger(1)) ||
      op.getSource().getType().isSignlessInteger(1)) {
    // Canonicalize i1 broadcast.
    // i1 represents vmsk in Mosaic and TPU doesn't support vmsk replication
    // directly.
    // Instead, convert i1 to i32 vector, broadcast i32, and then convert it
    // back to i1.
    CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
    Value i32_src;
    if (src_vty) {
      i32_src = builder.create<arith::ExtUIOp>(
          VectorType::get(src_vty.getShape(), builder.getI32Type()),
          op.getSource());
    } else {
      i32_src =
          builder.create<arith::ExtUIOp>(builder.getI32Type(), op.getSource());
    }
    auto i32_res_vty =
        VectorType::get(op.getType().getShape(), builder.getI32Type());
    auto bcast = builder.create<vector::BroadcastOp>(i32_res_vty, i32_src);
    auto ones = builder.create<arith::ConstantOp>(
        i32_res_vty,
        SplatElementsAttr::get(i32_res_vty,
                               builder.getOneAttr(builder.getI32Type())));
    Value cmp =
        builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bcast, ones);
    op.replaceAllUsesWith(cmp);
    op.erase();
    return cmp;
  }
  return raw_op.getResult(0);
}

FailureOr<Value> canonicalize_select(const CanonicalizeContext &ctx,
                                     Operation &raw_op) {
  auto op = dyn_cast<arith::SelectOp>(raw_op);
  if (!isa<VectorType>(op.getType()) ||
      isa<VectorType>(op.getCondition().getType())) {
    return raw_op.getResult(0);
  }
  // Canonicalize `i1 ? v1 : v2` -> `broadcast(i1) ? v1 : v2`.
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  auto cond_ty = VectorType::get(cast<VectorType>(op.getType()).getShape(),
                                 op.getCondition().getType());
  auto cond = builder.create<vector::BroadcastOp>(cond_ty, op.getCondition());
  auto new_op = builder.create<arith::SelectOp>(
      op.getLoc(), cond, op.getTrueValue(), op.getFalseValue());
  op.replaceAllUsesWith(new_op);
  op.erase();
  return new_op;
}

// All conversions that change bitwidth must be canonicalized to tpu.fptosi.
FailureOr<Value> canonicalize_fptosi(const CanonicalizeContext &ctx,
                                     Operation &raw_op) {
  auto op = cast<arith::FPToSIOp>(raw_op);
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  auto src_vty = dyn_cast<VectorType>(op.getIn().getType());
  auto dst_vty = dyn_cast<VectorType>(op.getType());
  if (static_cast<bool>(src_vty) != static_cast<bool>(dst_vty)) {
    return op.emitOpError("Vector/scalar mismatch between input and output");
  }
  bool is_vector = static_cast<bool>(src_vty);
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned src_bitwidth,
                             getElementTypeBitwidth(op.getIn().getType()));
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned dst_bitwidth,
                             getElementTypeBitwidth(op.getType()));
  if (dst_bitwidth > 32) {
    return op.emitOpError("Target bitwidth too large");
  }
  // We have low-level optimized code for
  //
  // - bf16->s8 and bf16->s4 casts on TPUv6+.
  // - f32->s8 and f32->s4 casts on TPU7x.
  bool bf16_to_s8_or_s4 = is_vector && src_vty.getElementType().isBF16() &&
                          (dst_vty.getElementType().isSignlessInteger(8) ||
                           dst_vty.getElementType().isSignlessInteger(4));
  bool f32_to_s8_or_s4 = is_vector && src_vty.getElementType().isF32() &&
                         (dst_vty.getElementType().isSignlessInteger(8) ||
                          dst_vty.getElementType().isSignlessInteger(4));
  if ((ctx.hardware_generation >= 6 && bf16_to_s8_or_s4) ||
      (ctx.hardware_generation >= 7 && f32_to_s8_or_s4)) {
    auto new_op = builder.create<tpu::FPToSIOp>(
        op.getType(), op.getIn(), tpu::RoundingMode::kTowardsZero);
    op.replaceAllUsesWith(new_op);
    op.erase();
    return new_op;
  }

  if (src_bitwidth == 32 && dst_bitwidth == 32) {
    return raw_op.getResult(0);
  }
  if (!ctx.compatibility_mode) {
    return op.emitOpError(
        "On this target float-to-integer conversions can only happen on "
        "32-bit values. Enable compatibility mode or upcast to float32, cast "
        "to int32 and truncate to desired bitwidth.");
  }

  Value x = op.getIn();
  // Upcast the input to f32.
  if (src_bitwidth < 32) {
    if (is_vector) {
      x = builder.create<tpu::ExtFOp>(
          VectorType::get(src_vty.getShape(), builder.getF32Type()), x);
    } else {
      x = builder.create<tpu::ExtFOp>(builder.getF32Type(), x);
    }
  }
  if (dst_bitwidth < 32) {
    // Need to clip values to match XLA
    auto clip = [&](Value x, Value low, Value high) {
      x = builder.create<arith::MaximumFOp>(x, low);
      x = builder.create<arith::MinimumFOp>(x, high);
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
    x = builder.create<arith::TruncIOp>(op.getType(), x);
  }
  op.replaceAllUsesWith(x);
  op.erase();
  return x;
}

FailureOr<Value> canonicalize_sitofp(const CanonicalizeContext &ctx,
                                     Operation &raw_op) {
  auto op = cast<arith::SIToFPOp>(raw_op);
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  auto src_vty = dyn_cast<VectorType>(op.getIn().getType());
  auto dst_vty = dyn_cast<VectorType>(op.getType());
  if (static_cast<bool>(src_vty) != static_cast<bool>(dst_vty)) {
    return op.emitOpError("Vector/scalar mismatch between input and output");
  }
  bool is_vector = static_cast<bool>(src_vty);
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned src_bitwidth,
                             getElementTypeBitwidth(op.getIn().getType()));
  FAILUREOR_ASSIGN_OR_RETURN(const unsigned dst_bitwidth,
                             getElementTypeBitwidth(op.getType()));

  // We have low-level optimized code for s8->bf16 and s4->bf16 casts on v6.
  if (ctx.hardware_generation >= 6 && is_vector &&
      (src_vty.getElementType().isSignlessInteger(8) ||
       src_vty.getElementType().isSignlessInteger(4)) &&
      dst_vty.getElementType().isBF16()) {
    auto new_op = builder.create<tpu::SIToFPOp>(
        op.getType(), op.getIn(), tpu::RoundingMode::kToNearestEven);
    op.replaceAllUsesWith(new_op);
    op.erase();
    return new_op;
  }

  if (src_bitwidth == 32 && dst_bitwidth == 32) {
    return raw_op.getResult(0);
  }
  if (!ctx.compatibility_mode) {
    return op.emitOpError(
        "On this target integer-to-float conversions can only happen on "
        "32-bit values. Enable compatibility mode or upcast to int32, cast to "
        "float32 and truncate to desired bitwidth.");
  }

  // Canonicalize (intX -> floatY) to (intX -> int32 -> float32 -> floatY).
  Value x = op.getIn();
  if (src_bitwidth < 32) {
    if (is_vector) {
      x = builder.create<arith::ExtSIOp>(
          VectorType::get(src_vty.getShape(), builder.getI32Type()), x);
    } else {
      x = builder.create<arith::ExtSIOp>(builder.getI32Type(), x);
    }
  }
  if (is_vector) {
    x = builder.create<tpu::SIToFPOp>(
        VectorType::get(src_vty.getShape(), builder.getF32Type()), x,
        tpu::RoundingMode::kToNearestEven);
  } else {
    x = builder.create<tpu::SIToFPOp>(builder.getF32Type(), x,
                                      tpu::RoundingMode::kToNearestEven);
  }
  if (dst_bitwidth < 32) {
    x = builder.create<tpu::TruncFOp>(op.getType(), x,
                                      tpu::RoundingMode::kToNearestEven);
  }
  op.replaceAllUsesWith(x);
  op.erase();
  return x;
}

FailureOr<Value> canonicalize_repeat(const CanonicalizeContext &ctx,
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
    return operand;
  }
  auto operands = std::vector<Value>(times, operand);
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  auto concat = builder.create<tpu::ConcatenateOp>(op.getType(), operands,
                                                   op.getDimension());
  op.replaceAllUsesWith(concat);
  op.erase();
  return concat;
}

FailureOr<Value> canonicalize_vector_transpose(const CanonicalizeContext &ctx,
                                               Operation &raw_op) {
  auto op = cast<vector::TransposeOp>(raw_op);
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  auto new_op = builder.create<tpu::TransposeOp>(op.getType(), op.getVector(),
                                                 op.getPermutation());
  op.replaceAllUsesWith(new_op);
  op.erase();
  return new_op;
}

// Finds the split point for a reshape between a multi-dimensional shape and a
// shape where a suffix has been collapsed into a single dimension.
//
// This function checks if `src_shape` and `tgt_shape` follow the pattern:
//   src_shape: (P..., S_1, S_2, ..., S_N)
//   tgt_shape: (P..., T_collapsed)
// where `P` is a common prefix and `product(S_1..S_N) == T_collapsed`.
//
// It handles a differing number of leading 1s in the prefix by stripping them
// from both shapes before comparison.
//
// This utility is used for two inverse patterns:
// 1. Collapse (e.g., `load` -> `reshape`): The function is called directly,
//    where `src_shape` is the multi-dimensional pre-reshape vector shape.
// 2. Expand (e.g., `reshape` -> `store`): The function is called with swapped
//    arguments, where `src_shape` is the multi-dimensional *post-reshape*
//    vector shape.
//
// Returns:
//   - A pair containing:
//     1. The index in `src_shape` where the collapsing suffix begins.
//     2. The product of the collapsed dimensions excluding the innermost one
//        (i.e., product(S_1..S_{N-1})), used as the "sublane product".
//   - `std::nullopt` if the shapes do not match the pattern.
std::optional<std::pair<int, int>> findSplitPoint(ArrayRef<int64_t> src_shape,
                                                  ArrayRef<int64_t> tgt_shape) {
  int s = 0, t = 0;
  // drop leading 1s
  while (s < src_shape.size() && src_shape[s] == 1) {
    ++s;
  }
  while (t < tgt_shape.size() && tgt_shape[t] == 1) {
    ++t;
  }

  // Find the end of the common prefix between the shapes (ignoring leading 1s).
  int s_prefix_end = s, t_prefix_end = t;
  while (s_prefix_end < src_shape.size() && t_prefix_end < tgt_shape.size() &&
         src_shape[s_prefix_end] == tgt_shape[t_prefix_end]) {
    ++s_prefix_end;
    ++t_prefix_end;
  }

  // After the common prefix, the rest of the target shape must consist of just
  // one dimension (the collapsed one).
  if (t_prefix_end != tgt_shape.size() - 1) {
    return std::nullopt;
  }
  int64_t src_prod = 1;
  for (int i = s_prefix_end; i < src_shape.size(); ++i) {
    src_prod *= src_shape[i];
  }

  if (tgt_shape.back() != src_prod) {
    return std::nullopt;
  }
  src_prod /= src_shape.back();
  return std::make_pair(s_prefix_end, src_prod);
}

FailureOr<Value> canonicalize_reshape(const CanonicalizeContext &ctx,
                                      Operation &raw_op) {
  // def fused_load_reshape(memref, indices):
  //   # 1. Create a memref view for packed i32 loading.
  //   # Original shape: <Prefix_mem..., S, Lane, ElemTy>
  //   # New i32 view:   <Prefix_mem..., S/packing, Lane, i32>
  //   i32_view = memref.reshape_and_bitcast(...)
  //
  //   # 2. Load i32 rows, unpack elements from each, and collect the chunks.
  //   unpacked_chunks = []
  //   for i in range(S / packing):
  //     # Load a single row of packed data. This corresponds to a
  //     # `StridedLoad` + `ShapeCast`. Result shape: <Prefix_vec..., Lane, i32>
  //     i32_chunk = load_i32_row(i32_view, indices_prefix, i)
  //
  //     # Unpack `packing` smaller elements from each i32 in the chunk.
  //     for p in range(packing):
  //       unpacked_chunk = i32_chunk >> (p * bitwidth)
  //       unpacked_chunks.append(unpacked_chunk)
  //
  //   # 3. Concatenate all resulting chunks into a single large vector.
  //   # Result shape after concat: <Prefix_vec..., S*Lane, i32>
  //   concatenated_i32 = np.concatenate(unpacked_chunks, axis=-1)
  //
  //   # 4. Truncate/bitcast the i32 vector to the final target type.
  //   return concatenated_i32.trunc().bitcast(final_element_type)
  auto op = cast<vector::ShapeCastOp>(raw_op);
  Value src = op.getSource();
  auto tgt_ty = op.getResult().getType();

  auto load_op = dyn_cast_if_present<vector::LoadOp>(src.getDefiningOp());
  if (!load_op) {
    return raw_op.getResult(0);
  }
  // This rewrite is only safe if the load has one user (this shape_cast).
  if (!load_op.getResult().hasOneUse()) {
    return raw_op.getResult(0);
  }

  auto src_ty = cast<VectorType>(load_op.getResult().getType());
  auto ref = load_op.getBase();
  auto memref_ty = ref.getType();
  if (!isContiguousMemref(ref)) {
    return raw_op.getResult(0);
  }

  const int64_t sublane = ctx.target_shape[0];
  const int64_t lane = ctx.target_shape[1];
  // Check if we are collapsing the lanes.
  if (tgt_ty.getShape().back() == lane) {
    return raw_op.getResult(0);
  }
  // This pattern is for collapse only, not expand.
  if (src_ty.getRank() < tgt_ty.getRank()) {
    return raw_op.getResult(0);
  }

  auto split_opt = findSplitPoint(src_ty.getShape(), tgt_ty.getShape());
  if (!split_opt) {
    return raw_op.getResult(0);
  }
  auto [split_point, sublane_prod] = *split_opt;
  if (split_point == 0) {
    // This is a 1d case
    return raw_op.getResult(0);
  }

  auto memref_shape = cast<MemRefType>(memref_ty).getShape();
  auto src_ty_shape = src_ty.getShape();
  auto mem_rank = memref_shape.size();
  auto vec_rank = src_ty_shape.size();
  if (mem_rank < 2 || vec_rank < 2) {
    // This is a 1d case
    return raw_op.getResult(0);
  }

  if (memref_shape[mem_rank - 1] != src_ty_shape[vec_rank - 1] ||
      memref_shape[mem_rank - 2] != src_ty_shape[vec_rank - 2]) {
    // This indicates slicing in the 2nd minor or minor.
    return raw_op.getResult(0);
  }

  int bitwidth = src_ty.getElementTypeBitWidth();
  int packing = 32 / bitwidth;
  if (ctx.hardware_generation < 4 && packing > 1) {
    return raw_op.getResult(0);
  }
  if (sublane_prod == 0 || sublane_prod % packing != 0) {
    return raw_op.getResult(0);
  }
  // We only support cases where we fill a full vreg.
  auto tgt_sublane = *(tgt_ty.getShape().end() - 2);
  auto tgt_lane = *(tgt_ty.getShape().end() - 1);
  if (tgt_sublane % (sublane * packing) != 0) {
    if (tgt_lane % lane != 0) {
      return raw_op.getResult(0);
    }
  }

  auto indices = load_op.getIndices();

  // Distinguish between the memref's prefix and the (potentially sliced)
  // vector's prefix.
  SmallVector<int64_t> mem_shape(memref_ty.getShape().begin(),
                                 memref_ty.getShape().begin() + split_point);
  SmallVector<int64_t> vec_shape(src_ty.getShape().begin(),
                                 src_ty.getShape().begin() + split_point);

  CanonicalBuilder b(ctx, op->getLoc(), op.getOperation());
  auto loc = op.getLoc();
  auto i32_type = b.getI32Type();

  // Create a new memref view that matches the dimensions being collapsed.
  mem_shape.push_back(sublane_prod);
  mem_shape.push_back(lane);
  auto mem_shape_prod = 1;
  for (int i = 0; i < mem_shape.size(); ++i) {
    mem_shape_prod *= mem_shape[i];
  }
  auto ref_prod = 1;
  for (int i = 0; i < ref.getType().getShape().size(); ++i) {
    ref_prod *= ref.getType().getShape()[i];
  }
  if (mem_shape_prod != ref_prod) {
    // In certain cases, upstream padding may change the memref shape,
    // which makes the intermediary memref reshape we rely on not sound.
    // ex: 13 in sublanes will get padded to 16 in sublanes for bf16.
    return raw_op.getResult(0);
  }
  Value reshaped_ref = b.create<tpu::MemRefReshapeOp>(
      MemRefType::get(mem_shape, memref_ty.getElementType()), ref);

  // Bitcast this view to i32 for packed loading.
  int64_t num_i32_rows = sublane_prod / packing;
  *(mem_shape.end() - 2) = num_i32_rows;
  Value i32_view = b.create<tpu::MemRefBitcastOp>(
      MemRefType::get(mem_shape, i32_type), reshaped_ref);

  // Define the shape of the small i32 chunk we will load in each iteration.
  SmallVector<int64_t> chunk_shape = vec_shape;
  chunk_shape.push_back(1);
  chunk_shape.push_back(lane);
  auto chunk_ty = VectorType::get(chunk_shape, i32_type);

  // Set up strides for tpu.StridedLoadOp. We only stride along the dimension
  // we're iterating over.
  int stride_dim = split_point;
  SmallVector<int32_t> strides(mem_shape.size(), 1);
  strides[stride_dim] = num_i32_rows;

  // Loop to load, unpack, and collect all vector chunks.
  SmallVector<Value> unpacked_chunks;
  unpacked_chunks.reserve(sublane_prod);

  // Reuse indices from the original load for the prefix.
  SmallVector<Value> idxs(indices.begin(), indices.begin() + split_point);
  // Dummy
  idxs.push_back(nullptr);
  idxs.push_back(IdxConst(0, b, loc));

  // Collapse the '1' second minor dimension from the loaded chunk.
  SmallVector<int64_t> collapsed_shape = vec_shape;
  collapsed_shape.push_back(lane);
  for (int i = 0; i < num_i32_rows; ++i) {
    idxs[stride_dim] = IdxConst(i, b, loc);
    Value slice =
        b.create<tpu::StridedLoadOp>(chunk_ty, i32_view, idxs, strides);

    Value collapsed = b.create<vector::ShapeCastOp>(
        VectorType::get(collapsed_shape, i32_type), slice);

    // Unpack elements from i32 if necessary.
    for (int p = 0; p < packing; ++p) {
      unpacked_chunks.push_back(b.create<arith::ShRUIOp>(
          collapsed.getType(), collapsed,
          I32Const(p * bitwidth, collapsed_shape, b, loc)));
    }
  }

  Value i32_flat;
  if (unpacked_chunks.size() == 1) {
    i32_flat = unpacked_chunks.front();
  } else {
    SmallVector<int64_t> concat_shape = vec_shape;
    concat_shape.push_back(lane * sublane_prod);
    int concat_dim = concat_shape.size() - 1;
    i32_flat = b.create<tpu::ConcatenateOp>(
        VectorType::get(concat_shape, i32_type), unpacked_chunks, concat_dim);
  }

  // Convert the final i32 vector back to the target type.
  Value final_vec = i32_flat;
  if (packing > 1) {
    final_vec = b.create<arith::TruncIOp>(
        VectorType::get(cast<VectorType>(i32_flat.getType()).getShape(),
                        b.getIntegerType(bitwidth)),
        i32_flat);
  }
  final_vec = b.create<arith::BitcastOp>(
      VectorType::get(cast<VectorType>(final_vec.getType()).getShape(),
                      tgt_ty.getElementType()),
      final_vec);
  if (final_vec.getType() != tgt_ty) {
    final_vec = b.create<vector::ShapeCastOp>(tgt_ty, final_vec);
  }

  op.replaceAllUsesWith(final_vec);
  op.erase();
  return final_vec;
}

Value _canonicalize_store(const CanonicalizeContext &ctx, Operation &raw_op) {
  // Fuses a vector.shape_cast (that expands dimensions) into a subsequent
  // vector.store or dense tpu.vector_store. This is the inverse of the
  // canonicalize_reshape func.
  //
  // def fused_reshape_store(source_vector, target_memref, indices):
  //   # `source_vector` is large and flat, e.g., shape (P..., T_collapsed)
  //   # `target_memref` has an expanded shape, e.g., (P..., S1, S2, ..., Lane)
  //
  //   # 1. Create a memref view for packed i32 storing.
  //   # Original target shape: <Prefix_mem..., S1, S2, ..., Lane, ElemTy>
  //   # Let S_prod = S1 * S2 * ...
  //   # New i32 view shape:   <Prefix_mem..., S_prod/packing, Lane, i32>
  //   i32_view = target_memref.reshape_and_bitcast(...)
  //
  //   # 2. Loop over the rows of the i32 view, packing and storing data for
  //   each.
  //   for i in range(S_prod / packing):
  //     # a. Gather `packing` number of slices from the large source_vector.
  //     #    Each slice corresponds to one row of the original target memref.
  //     slices_to_pack = extract_slices(source_vector, base_offset=i*packing)
  //
  //     # b. Pack these smaller-typed slices into a single i32 vector chunk.
  //     #    This is the inverse of unpacking with right shifts. It involves
  //     #    extending, left-shifting, and OR-ing.
  //     i32_chunk = (slices_to_pack[0] << 0) | \
  //                 (slices_to_pack[1] << bitwidth) | \
  //                 ...
  //
  //     # c. Store the resulting packed i32_chunk into the i-th row of the
  //     #    `i32_view`. This corresponds to a `StridedStoreOp`.
  //     store_i32_row(i32_chunk, i32_view, indices_prefix, i)
  Value value_to_store;
  TypedValue<MemRefType> base;
  ValueRange indices;

  Operation *store_op;

  // Note(mvoz): We have code that handles both, and am not sure why
  // we don't just use the tpu::VectorStoreOp? Either way, they are
  // similar enough that we can handle them both here.
  if (auto store = dyn_cast<vector::StoreOp>(raw_op)) {
    store_op = store.getOperation();
    value_to_store = store.getValueToStore();
    base = store.getBase();
    indices = store.getIndices();
  } else if (auto store = dyn_cast<tpu::VectorStoreOp>(raw_op)) {
    store_op = store.getOperation();
    value_to_store = store.getValueToStore();
    base = store.getBase();
    indices = store.getIndices();
    if (!store.getStrides().empty() || store.getMask() || store.getAdd()) {
      // We don't support these cases.
      return value_to_store;
    }
  } else {
    return value_to_store;
  }

  auto shape_cast_op =
      dyn_cast_if_present<vector::ShapeCastOp>(value_to_store.getDefiningOp());
  if (!shape_cast_op || !shape_cast_op.getResult().hasOneUse()) {
    // Not a shape cast... (or has more than one use)
    // Note(mvoz): We could potentially support the case of > 1 users,
    // by just not eliding the reshape at the end, and only rewriting the
    // store.
    return value_to_store;
  }

  auto src_ty = shape_cast_op.getSource().getType();
  auto tgt_ty = shape_cast_op.getResult().getType();
  auto memref_ty = base.getType();
  // Consider src_shape=(1, 384) -> tgt_shape=(1, 1, 3, 128)
  // Upstream padding actually makes this:
  // Store shapes base: (1,1,4,128)
  // Store shapes src: (1,384)
  // Store shapes tgt: (1,1,3,128)
  // Store shapes memref: (1,1,4,128)
  if (tgt_ty.getShape() != memref_ty.getShape()) {
    return value_to_store;
  }
  if (!isContiguousMemref(base)) {
    return value_to_store;
  }
  if (src_ty.getRank() > tgt_ty.getRank()) {
    // Src is not a collapse of target!
    // TODO(mvoz): We can handle an arbitrary number of major
    // dimensions by adding a "free" major dimension reshape between
    // here and the strided loads!
    return value_to_store;
  }
  auto last_src_lanes = src_ty.getShape().back();
  if (last_src_lanes % ctx.target_shape[1] != 0) {
    // We don't support cases where the last dimension is not a multiple of
    // the lane size.
    return value_to_store;
  }
  std::optional<std::pair<int, int>> split_opt =
      findSplitPoint(tgt_ty.getShape(), src_ty.getShape());
  if (!split_opt) {
    // Not a collapse...
    return value_to_store;
  }
  auto [split_point, sublane_prod] = *split_opt;

  int bitwidth = src_ty.getElementTypeBitWidth();
  int packing = 32 / bitwidth;
  if (ctx.hardware_generation < 4 && packing > 1) {
    // Old hardware doesn't support strided store with packing > 1.
    return value_to_store;
  }
  if (sublane_prod % packing != 0) {
    // We don't support cases where we have offsets in the sublane dimension.
    return value_to_store;
  }

  // Note(mvoz): This is an inverse of the packing logic in
  // canonicalize_reshape.
  CanonicalBuilder b(ctx, store_op->getLoc(), store_op);
  auto loc = store_op->getLoc();
  auto i32_type = b.getI32Type();
  int64_t num_i32_rows = sublane_prod / packing;

  SmallVector<int64_t> mem_shape(memref_ty.getShape().begin(),
                                 memref_ty.getShape().begin() + split_point);
  mem_shape.push_back(sublane_prod);
  auto lane_dim = memref_ty.getShape().back();
  if (lane_dim != ctx.target_shape[1]) {
    // Note(mvoz): The math below *is* sound, but we don't support this case
    // because apply has a restriction on strided_store where
    // it expects that "The last dim size is not 128 in original base memref"
    return value_to_store;
  }
  mem_shape.push_back(lane_dim);
  Value reshaped_ref = b.create<tpu::MemRefReshapeOp>(
      MemRefType::get(mem_shape, memref_ty.getElementType()), base);

  *(mem_shape.end() - 2) = num_i32_rows;
  Value i32_view = b.create<tpu::MemRefBitcastOp>(
      MemRefType::get(mem_shape, i32_type), reshaped_ref);

  Value src_vec = shape_cast_op.getSource();
  SmallVector<int64_t> slice_sizes(src_ty.getShape());
  slice_sizes.back() = lane_dim;
  SmallVector<int64_t> unit_strides(src_ty.getRank(), 1);

  SmallVector<Value> store_indices(indices.begin(),
                                   indices.begin() + split_point);
  store_indices.push_back(nullptr);
  store_indices.push_back(IdxConst(0, b, loc));
  int stride_dim = split_point;

  for (int i = 0; i < num_i32_rows; ++i) {
    SmallVector<int64_t> offsets(src_ty.getRank(), 0);
    offsets.back() = i * packing * lane_dim;
    Value slice = b.create<vector::ExtractStridedSliceOp>(
        src_vec, offsets, slice_sizes, unit_strides);

    auto i_chunk_ty =
        VectorType::get(cast<VectorType>(slice.getType()).getShape(),
                        b.getIntegerType(bitwidth));
    auto i32_chunk_ty =
        VectorType::get(cast<VectorType>(slice.getType()).getShape(), i32_type);
    Value packed_chunk;
    if (packing > 1) {
      Value acc = b.create<arith::ExtUIOp>(
          i32_chunk_ty, b.create<arith::BitcastOp>(i_chunk_ty, slice));
      for (int p = 1; p < packing; ++p) {
        offsets.back() = (i * packing + p) * lane_dim;
        // Akin to updating offsets on the slice
        // Note(mvoz): Maybe find a better way to do this?
        slice = b.create<vector::ExtractStridedSliceOp>(
            src_vec, offsets, slice_sizes, unit_strides);
        Value sj_i32 = b.create<arith::ExtUIOp>(
            i32_chunk_ty, b.create<arith::BitcastOp>(i_chunk_ty, slice));
        Value sh = I32Const(p * bitwidth, i32_chunk_ty.getShape(), b, loc);
        acc = b.create<arith::OrIOp>(acc, b.create<arith::ShLIOp>(sj_i32, sh));
      }
      packed_chunk = acc;
    } else {
      packed_chunk = b.create<arith::BitcastOp>(i32_chunk_ty, slice);
    }

    auto i32_view_shape = cast<MemRefType>(i32_view.getType()).getShape();
    SmallVector<int64_t> target_vector_shape(i32_view_shape);
    target_vector_shape[stride_dim] = 1;
    auto target_vector_type = VectorType::get(target_vector_shape, i32_type);

    Value chunk_to_store =
        b.create<vector::ShapeCastOp>(target_vector_type, packed_chunk);

    store_indices[stride_dim] = IdxConst(i, b, loc);
    b.create<tpu::StridedStoreOp>(
        chunk_to_store, i32_view, store_indices,
        SmallVector<int32_t>(i32_view_shape.size(), 1));
  }

  store_op->erase();
  shape_cast_op->erase();

  return base;
}

FailureOr<Value> canonicalize_store(const CanonicalizeContext &ctx,
                                    Operation &raw_op) {
  return _canonicalize_store(ctx, raw_op);
}

FailureOr<Value> canonicalize_vector_store(const CanonicalizeContext &ctx,
                                           Operation &raw_op) {
  return _canonicalize_store(ctx, raw_op);
}

FailureOr<Value> canonicalize_transpose(const CanonicalizeContext &ctx,
                                        Operation &raw_op) {
  auto op = cast<tpu::TransposeOp>(raw_op);
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  VectorType input_vty = op.getVector().getType();
  VectorType output_vty = op.getType();
  Type element_type = op.getVector().getType().getElementType();
  // TODO(mvoz): Even gen 7 support is spotty on all test targets.
  if (element_type.getIntOrFloatBitWidth() == 8 && ctx.compatibility_mode &&
      ctx.hardware_generation > 3) {
    Value val_bf16;
    VectorType input_vty_bf16 =
        VectorType::get(input_vty.getShape(), builder.getBF16Type());
    if (isa<IntegerType>(element_type)) {
      val_bf16 =
          builder.create<arith::SIToFPOp>(input_vty_bf16, op.getOperand());
    } else {
      val_bf16 = builder.create<tpu::ExtFOp>(
          VectorType::get(input_vty.getShape(), builder.getBF16Type()),
          op.getOperand());
    }

    Value transposed_bf16 = builder.create<tpu::TransposeOp>(
        VectorType::get(output_vty.getShape(), builder.getBF16Type()), val_bf16,
        op.getPermutation());

    Value new_result;
    if (isa<IntegerType>(element_type)) {
      new_result =
          builder.create<arith::FPToSIOp>(op.getType(), transposed_bf16);
    } else {
      new_result = builder.create<tpu::TruncFOp>(
          output_vty, transposed_bf16, tpu::RoundingMode::kToNearestEven);
    }
    op.replaceAllUsesWith(new_result);
    op.erase();
    return new_result;
  }
  return raw_op.getResult(0);
}

FailureOr<Value> canonicalize_arith_extf(const CanonicalizeContext &ctx,
                                         Operation &raw_op) {
  auto op = cast<arith::ExtFOp>(raw_op);
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  // Canonicalize arith::ExtFOp to tpu::ExtFOp.
  auto new_result = builder.create<tpu::ExtFOp>(op.getType(), op.getOperand());
  op.replaceAllUsesWith(new_result);
  op.erase();
  return new_result;
}

FailureOr<Value> canonicalize_tpu_extf(const CanonicalizeContext &ctx,
                                       Operation &raw_op) {
  auto op = cast<tpu::ExtFOp>(raw_op);
  auto dst_ty = dyn_cast<VectorType>(op.getType());
  if (!dst_ty) {
    return raw_op.getResult(0);
  }

  auto src_ty = cast<VectorType>(op.getOperand().getType());
  auto dst_elem_ty = dst_ty.getElementType();
  auto src_elem_ty = src_ty.getElementType();
  if (dst_elem_ty.isF32()) {
    // Cast to f32 is always supported.
    return raw_op.getResult(0);
  }

  if (dst_elem_ty.isBF16() &&
      isa<Float8E5M2Type, Float8E4M3FNType>(src_elem_ty) &&
      ctx.hardware_generation >= 7) {
    // We have low-level optimized code for f8e4m3fn and f8e5m2 -> bf16 casts on
    // TPU7x+.
    return raw_op.getResult(0);
  }

  if (!ctx.compatibility_mode) {
    return op.emitOpError(
        "Enable compatibility mode to support extension to non-f32.");
  }

  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  // Otherwise, cast to f32 and then truncate.
  VectorType f32_ty = VectorType::get(src_ty.getShape(), builder.getF32Type());
  Value val_f32 = builder.create<tpu::ExtFOp>(f32_ty, op.getOperand());
  auto new_result = builder.create<tpu::TruncFOp>(
      dst_ty, val_f32, tpu::RoundingMode::kToNearestEven);
  op.replaceAllUsesWith(new_result);
  op.erase();
  return new_result;
}

FailureOr<Value> canonicalize_arith_truncf(const CanonicalizeContext &ctx,
                                           Operation &raw_op) {
  auto op = cast<arith::TruncFOp>(raw_op);
  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  // Canonicalize arith::TruncFOp to tpu::TruncFOp.
  auto new_result = builder.create<tpu::TruncFOp>(
      op.getType(), op.getOperand(), tpu::RoundingMode::kToNearestEven);
  op.replaceAllUsesWith(new_result);
  op.erase();
  return new_result;
}

FailureOr<Value> canonicalize_tpu_truncf(const CanonicalizeContext &ctx,
                                         Operation &raw_op) {
  auto op = cast<tpu::TruncFOp>(raw_op);
  auto src_ty = dyn_cast<VectorType>(op.getOperand().getType());
  if (!src_ty) {
    return raw_op.getResult(0);
  }

  auto dst_ty = cast<VectorType>(op.getType());
  auto src_elem_ty = src_ty.getElementType();
  auto dst_elem_ty = dst_ty.getElementType();
  if (src_elem_ty.isF32()) {
    // Truncate from f32 is always supported.
    return raw_op.getResult(0);
  }

  if (src_elem_ty.isBF16() &&
      isa<Float8E5M2Type, Float8E4M3FNType>(dst_elem_ty) &&
      ctx.hardware_generation >= 7) {
    // We have low-level optimized code for bf16 -> f8e4m3fn and f8e5m2 casts on
    // TPU7x+.
    return raw_op.getResult(0);
  }

  if (!ctx.compatibility_mode) {
    return op.emitOpError(
        "Enable compatibility mode to support truncation from non-f32.");
  }

  CanonicalBuilder builder(ctx, op->getLoc(), op.getOperation());
  // Otherwise, cast to f32 and then truncate.
  VectorType f32_ty = VectorType::get(src_ty.getShape(), builder.getF32Type());
  Value val_f32 = builder.create<tpu::ExtFOp>(f32_ty, op.getOperand());
  auto new_result = builder.create<tpu::TruncFOp>(
      dst_ty, val_f32, tpu::RoundingMode::kToNearestEven);
  op.replaceAllUsesWith(new_result);
  op.erase();
  return new_result;
}

const llvm::StringMap<canonicalize_rule_type> &rules() {
  static auto rules = new llvm::StringMap<canonicalize_rule_type>{
      {tpu::MatmulOp::getOperationName(), canonicalize_matmul},
      {vector::ContractionOp::getOperationName(), canonicalize_contraction},
      {vector::ExtractOp::getOperationName(), canonicalize_extract},
      {vector::MultiDimReductionOp::getOperationName(),
       canonicalize_multi_dim_reduction},
      {vector::TransposeOp::getOperationName(), canonicalize_vector_transpose},
      {vector::ShapeCastOp::getOperationName(), canonicalize_reshape},
      {vector::BroadcastOp::getOperationName(), canonicalize_broadcast},
      {arith::SelectOp::getOperationName(), canonicalize_select},
      {arith::FPToSIOp::getOperationName(), canonicalize_fptosi},
      {arith::SIToFPOp::getOperationName(), canonicalize_sitofp},
      {arith::TruncFOp::getOperationName(), canonicalize_arith_truncf},
      {arith::ExtFOp::getOperationName(), canonicalize_arith_extf},
      {tpu::TruncFOp::getOperationName(), canonicalize_tpu_truncf},
      {tpu::ExtFOp::getOperationName(), canonicalize_tpu_extf},
      {tpu::TransposeOp::getOperationName(), canonicalize_transpose},
      {tpu::RepeatOp::getOperationName(), canonicalize_repeat},
      {vector::StoreOp::getOperationName(), canonicalize_store},
      {tpu::VectorStoreOp::getOperationName(), canonicalize_vector_store}};
  return *rules;
}

const llvm::StringMap<int> &bf16_ops_min_supported_versions() {
  static const auto m = new llvm::StringMap<int>{
      {arith::DivFOp::getOperationName(), 4},
      {arith::SelectOp::getOperationName(), 5},
      {arith::CmpFOp::getOperationName(), 5},
      {arith::MulFOp::getOperationName(), 6},
      {arith::AddFOp::getOperationName(), 6},
      {arith::SubFOp::getOperationName(), 6},
      {arith::MaximumFOp::getOperationName(), 6},
      {arith::MinimumFOp::getOperationName(), 6},
      {math::PowFOp::getOperationName(), 6},
      {math::TanhOp::getOperationName(), 6},
      {math::ExpOp::getOperationName(), 6},
      {math::Exp2Op::getOperationName(), 6},
      {math::LogOp::getOperationName(), 6},
  };
  return *m;
}

bool need_elementwise_canonicalization(const CanonicalizeContext &ctx,
                                       Operation &op) {
  // Only rewrite when the hardware generation is below the minimum supported
  // version.
  auto it = bf16_ops_min_supported_versions().find(op.getName().getStringRef());
  if (it == bf16_ops_min_supported_versions().end() ||
      ctx.hardware_generation >= it->second) {
    return false;
  }
  return llvm::any_of(op.getOperands(), [](Value operand) {
    auto vty = dyn_cast<VectorType>(operand.getType());
    return vty && vty.getElementType().isBF16();
  });
}

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer(int hardware_generation, bool compatibility_mode,
                      std::array<int64_t, 2> target_shape)
      : hardware_generation_(hardware_generation),
        compatibility_mode_(compatibility_mode),
        target_shape_(target_shape) {}

  int hardware_generation_;
  bool compatibility_mode_;
  std::array<int64_t, 2> target_shape_;

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
    CanonicalizeContext ctx(
        {compatibility_mode_, hardware_generation_, target_shape_});
    // We must iterate over the op first, because canonicalization can cause
    // us to .erase() an op, and accessing getRegions on it after is not
    // sound. Invariant - top level ops with regions may never be invalidated.
    for (Region &region : any_op.getRegions()) {
      for (Block &block : region) {
        if (canonicalizeBlock(block).failed()) {
          return failure();
        }
      }
    }
    if (need_elementwise_canonicalization(ctx, any_op)) {
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
  CanonicalizeMosaicPass(int hardware_generation_p, bool compatibility_mode_p,
                         std::array<int64_t, 2> target_shape)
      : compatibility_mode_(compatibility_mode_p) {
    this->hardware_generation = hardware_generation_p;
    this->sublane_count = target_shape[0];
    this->lane_count = target_shape[1];
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc(hardware_generation, compatibility_mode_,
                            {sublane_count, lane_count});
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };

  bool compatibility_mode_;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass(
    int hardware_generation, bool compatibility_mode,
    std::array<int64_t, 2> target_shape) {
  return std::make_unique<CanonicalizeMosaicPass>(
      hardware_generation, compatibility_mode, target_shape);
}

}  // namespace mlir::tpu
