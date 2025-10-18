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
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_PRECANONICALIZATIONOPTIMIZATIONPASS
#define GEN_PASS_DEF_PRECANONICALIZATIONOPTIMIZATIONPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

struct RhsTraversalResult {
  tpu::TransposeOp transpose_op = nullptr;
  vector::ExtractStridedSliceOp slice_op = nullptr;
};

std::optional<RhsTraversalResult> walkRhsForFusibleTranspose(Value rhs) {
  RhsTraversalResult result;
  Value current_operand = rhs;

  // Walk backwards from matmul RHS: slice -> transpose
  while (Operation* defining_op = current_operand.getDefiningOp()) {
    if (auto slice_op = dyn_cast<vector::ExtractStridedSliceOp>(defining_op)) {
      if (slice_op->hasOneUse() && !result.slice_op) {
        result.slice_op = slice_op;
        current_operand = slice_op.getVector();
        continue;
      }
    } else if (auto transpose_op = dyn_cast<tpu::TransposeOp>(defining_op)) {
      if (transpose_op->hasOneUse()) {
        result.transpose_op = transpose_op;
        // The value *before* the transpose.
        current_operand = transpose_op.getVector();
      }
      break;
    }
    break;
  }

  if (!result.transpose_op) {
    return std::nullopt;
  }

  return result;
}

// Attempts to fuse a tpu.transpose on the RHS of a tpu.matmul.
std::optional<std::tuple<Value, tpu::DotDimensionNumbersAttr>>
tryFuseRhsTranspose(tpu::MatmulOp op, ImplicitLocOpBuilder& builder) {
  std::optional<RhsTraversalResult> trace_result =
      walkRhsForFusibleTranspose(op.getRhs());

  if (!trace_result.has_value()) {
    return std::nullopt;
  }

  auto& trace = *trace_result;
  auto dimension_numbers = op.getDimensionNumbers().value();

  // This fusion logic is for matmuls with one contracting and one
  // non-contracting dimension on the RHS
  if (dimension_numbers.getRhsContractingDims().size() != 1 ||
      dimension_numbers.getRhsNonContractingDims().size() != 1) {
    return std::nullopt;
  }

  auto rhs_non_contracting_dim =
      dimension_numbers.getRhsNonContractingDims()[0];
  auto rhs_contracting_dim = dimension_numbers.getRhsContractingDims()[0];
  auto permutation = trace.transpose_op.getPermutation();

  // The transpose is fusible if it swaps the contracting and non-contracting
  // dimensions and leaves all batch dimensions unchanged.
  bool is_fusible_perm =
      (permutation[rhs_contracting_dim] == rhs_non_contracting_dim &&
       permutation[rhs_non_contracting_dim] == rhs_contracting_dim &&
       std::all_of(dimension_numbers.getRhsBatchDims().begin(),
                   dimension_numbers.getRhsBatchDims().end(),
                   [&](int64_t batch_dim) {
                     return permutation[batch_dim] == batch_dim;
                   }));

  if (!is_fusible_perm) {
    return std::nullopt;
  }

  Value current_val = trace.transpose_op.getVector();

  Value new_rhs;
  if (trace.slice_op) {
    auto get_i64_values = [](ArrayAttr attr) {
      return llvm::map_to_vector(attr, [](Attribute a) {
        return cast<IntegerAttr>(a).getValue().getSExtValue();
      });
    };

    auto old_offsets = get_i64_values(trace.slice_op.getOffsets());
    auto old_sizes = get_i64_values(trace.slice_op.getSizes());
    auto old_strides = get_i64_values(trace.slice_op.getStrides());

    SmallVector<int64_t> permuted_offsets(old_offsets.size());
    SmallVector<int64_t> permuted_sizes(old_sizes.size());
    SmallVector<int64_t> permuted_strides(old_strides.size());

    // Permute the slice
    for (const auto& it : llvm::enumerate(permutation)) {
      permuted_offsets[it.index()] = old_offsets[it.value()];
      permuted_sizes[it.index()] = old_sizes[it.value()];
      permuted_strides[it.index()] = old_strides[it.value()];
    }

    new_rhs = builder.create<vector::ExtractStridedSliceOp>(
        current_val, permuted_offsets, permuted_sizes, permuted_strides);
  } else {
    // If there was no slice, the new RHS is simply the value.
    new_rhs = current_val;
  }

  SmallVector<int64_t> new_output_dim_order;
  ArrayRef<int64_t> old_output_dim_order =
      dimension_numbers.getOutputDimOrder();
  for (unsigned i = 0; i < old_output_dim_order.size(); i += 2) {
    int64_t operand_idx = old_output_dim_order[i];
    int64_t dim_idx = old_output_dim_order[i + 1];
    new_output_dim_order.push_back(operand_idx);
    if (operand_idx == 1 && dim_idx == rhs_non_contracting_dim) {
      // The non-contracting dim is now at the old contracting dim's index.
      new_output_dim_order.push_back(rhs_contracting_dim);
    } else {
      new_output_dim_order.push_back(dim_idx);
    }
  }

  // Create the new dimension numbers by swapping the RHS contracting and
  // non-contracting dimensions AND updating the output dimension order.
  auto new_dimension_numbers = tpu::DotDimensionNumbersAttr::get(
      builder.getContext(), dimension_numbers.getLhsContractingDims(),
      /*rhs_contracting_dims=*/dimension_numbers.getRhsNonContractingDims(),
      dimension_numbers.getLhsNonContractingDims(),
      /*rhs_non_contracting_dims=*/dimension_numbers.getRhsContractingDims(),
      new_output_dim_order, dimension_numbers.getLhsBatchDims(),
      dimension_numbers.getRhsBatchDims());

  // Return the new RHS value, the toggled transpose flag, and the new dnums.
  return std::make_tuple(new_rhs, new_dimension_numbers);
}

struct PreCanonicalizationOptimizationPass
    : impl::PreCanonicalizationOptimizationPassBase<
          PreCanonicalizationOptimizationPass> {
  void runOnOperation() override {
    getOperation().walk([&](tpu::MatmulOp op) {
      // We only attempt this fusion if dimension numbers are present.
      if (!op.getDimensionNumbers().has_value()) {
        return;
      }
      ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());
      if (auto fusion_result = tryFuseRhsTranspose(op, builder)) {
        auto [new_rhs_val, new_dnums] = *fusion_result;

        auto new_rhs = cast<TypedValue<VectorType>>(new_rhs_val);
        // Update the matmul op in-place.
        op.getRhsMutable().assign(new_rhs);
        op.setDimensionNumbersAttr(new_dnums);
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createPreCanonicalizationOptimizationPass() {
  return std::make_unique<PreCanonicalizationOptimizationPass>();
}

}  // namespace mlir::tpu