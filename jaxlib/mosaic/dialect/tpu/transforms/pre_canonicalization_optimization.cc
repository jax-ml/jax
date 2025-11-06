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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_PRECANONICALIZATIONOPTIMIZATIONPASS
#define GEN_PASS_DEF_PRECANONICALIZATIONOPTIMIZATIONPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

// Finds the split point between common prefix and expanded dimensions for
// shape canonicalization patterns.
std::optional<std::pair<int64_t, int64_t>> findSplitPoint(
    ArrayRef<int64_t> src_shape, ArrayRef<int64_t> tgt_shape) {
  int64_t s = 0, t = 0;
  while (s < src_shape.size() && src_shape[s] == 1) {
    ++s;
  }
  while (t < tgt_shape.size() && tgt_shape[t] == 1) {
    ++t;
  }

  int64_t s_prefix_end = s, t_prefix_end = t;
  while (s_prefix_end < src_shape.size() && t_prefix_end < tgt_shape.size() &&
         src_shape[s_prefix_end] == tgt_shape[t_prefix_end]) {
    ++s_prefix_end;
    ++t_prefix_end;
  }

  if (t_prefix_end != tgt_shape.size() - 1) {
    return std::nullopt;
  }
  int64_t src_prod = 1;
  for (int64_t i = s_prefix_end; i < src_shape.size(); ++i) {
    src_prod *= src_shape[i];
  }

  if (tgt_shape.back() != src_prod) {
    return std::nullopt;
  }
  src_prod /= src_shape.back();
  return std::make_pair(s_prefix_end, src_prod);
}

void optimizeStore(int hardware_generation, std::array<int64_t, 2> target_shape,
                   Operation& raw_op) {
  // Fuses a vector.shape_cast (that expands dimensions) into a subsequent
  // vector.store or dense tpu.vector_store. This is the inverse of the
  // canonicalize_reshape func.
  Value value_to_store;
  TypedValue<MemRefType> base;
  ValueRange indices;

  if (auto store = dyn_cast<vector::StoreOp>(raw_op)) {
    value_to_store = store.getValueToStore();
    base = store.getBase();
    indices = store.getIndices();
  } else if (auto store = dyn_cast<tpu::VectorStoreOp>(raw_op)) {
    value_to_store = store.getValueToStore();
    base = store.getBase();
    indices = store.getIndices();
    if (!store.getStrides().empty() || store.getMask() || store.getAdd()) {
      return;
    }
  } else {
    return;
  }

  // Look for vector::ShapeCastOp feeding the store
  auto shape_cast_op =
      dyn_cast_if_present<vector::ShapeCastOp>(value_to_store.getDefiningOp());
  if (!shape_cast_op || !shape_cast_op.getResult().hasOneUse()) {
    return;
  }

  auto src_ty = shape_cast_op.getSource().getType();
  auto tgt_ty = shape_cast_op.getResult().getType();
  auto memref_ty = base.getType();

  // TODO(mvoz,apaszke): Add slicing support for stores (analogous to loads).
  if (tgt_ty.getShape() != memref_ty.getShape()) {
    return;
  }
  if (!isContiguousMemref(base)) {
    return;
  }
  if (src_ty.getRank() > tgt_ty.getRank()) {
    return;
  }
  auto last_src_lanes = src_ty.getShape().back();
  if (last_src_lanes % target_shape[1] != 0) {
    return;
  }
  std::optional<std::pair<int64_t, int64_t>> split_opt =
      findSplitPoint(tgt_ty.getShape(), src_ty.getShape());
  if (!split_opt) {
    return;
  }
  auto [split_point, sublane_prod] = *split_opt;

  int64_t bitwidth = src_ty.getElementTypeBitWidth();
  int64_t packing = 32 / bitwidth;
  if (hardware_generation < 4 && packing > 1) {
    return;
  }
  if (sublane_prod % packing != 0) {
    return;
  }

  ImplicitLocOpBuilder b(raw_op.getLoc(), &raw_op);
  auto loc = raw_op.getLoc();
  auto i32_type = b.getI32Type();
  int64_t num_i32_rows = sublane_prod / packing;

  SmallVector<int64_t> mem_shape;
  if (split_point == 0) {
    // Expand 1D: no common prefix between shapes, create new dimension.
    mem_shape.push_back(sublane_prod);
  } else {
    mem_shape.assign(memref_ty.getShape().begin(),
                     memref_ty.getShape().begin() + split_point);
    int64_t prev_dim = mem_shape.back();
    int64_t new_dim = prev_dim * sublane_prod;
    // Check for overflow in multiplication.
    if (sublane_prod != 0 && new_dim / sublane_prod != prev_dim) {
      return;
    }
    mem_shape.back() = new_dim;
  }

  auto lane_dim = memref_ty.getShape().back();
  if (lane_dim != target_shape[1]) {
    return;
  }
  mem_shape.push_back(lane_dim);
  Value reshaped_ref = b.create<tpu::MemRefReshapeOp>(
      MemRefType::get(mem_shape, memref_ty.getElementType()), base);

  *(mem_shape.end() - 2) /= packing;
  Value i32_view = b.create<tpu::MemRefBitcastOp>(
      MemRefType::get(mem_shape, i32_type), reshaped_ref);

  Value src_vec = shape_cast_op.getSource();
  SmallVector<int64_t> slice_sizes(src_ty.getShape());
  slice_sizes.back() = lane_dim;
  SmallVector<int64_t> unit_strides(src_ty.getRank(), 1);

  auto i32_view_shape = cast<MemRefType>(i32_view.getType()).getShape();

  SmallVector<Value> store_indices;
  Value split_base_idx;
  int64_t stride_dim;

  if (split_point == 0) {
    // No common prefix - create indices for entire i32_view shape
    split_base_idx = IdxConst(0, b, loc);
    for (size_t i = 0; i < i32_view_shape.size(); ++i) {
      store_indices.push_back(IdxConst(0, b, loc));
    }
    stride_dim = 0;
  } else {
    // TODO(mvoz,apaszke): This common prefix handling could be simplified by
    // always reinterpreting an nd operation as a 3d operation with untiled,
    // second minor, and minor dimensions. Common prefix exists - use it
    // This happens when split_point == 0 (1D expansion case).
    // The i32_view has an extra leading dimension compared to the packed
    // vector, so we need to add a dimension via reshape.
    store_indices.assign(indices.begin(), indices.begin() + split_point);
    split_base_idx = store_indices.back();
    // Add remaining indices to match i32_view rank
    while (store_indices.size() < i32_view_shape.size()) {
      store_indices.push_back(IdxConst(0, b, loc));
    }
    stride_dim = split_point - 1;
  }
  SmallVector<int32_t> strides(i32_view_shape.size(), 1);
  strides[stride_dim] = num_i32_rows;
  for (int64_t i = 0; i < num_i32_rows; ++i) {
    SmallVector<int64_t> offsets(src_ty.getRank(), 0);
    offsets.back() = i * packing * lane_dim;
    Value slice_i = b.create<vector::ExtractStridedSliceOp>(
        src_vec, offsets, slice_sizes, unit_strides);

    auto i_chunk_ty =
        VectorType::get(cast<VectorType>(slice_i.getType()).getShape(),
                        b.getIntegerType(bitwidth));
    auto i32_chunk_ty = VectorType::get(
        cast<VectorType>(slice_i.getType()).getShape(), i32_type);
    Value packed_chunk;
    if (packing > 1) {
      // TODO(mvoz): This packing can be implemented more efficiently as an
      // interleaved pack. We don't have an op for this in the
      // pre-apply_vector_layout IR, but we'll soon have one.
      Value acc = b.create<arith::ExtUIOp>(
          i32_chunk_ty, b.create<arith::BitcastOp>(i_chunk_ty, slice_i));
      for (int64_t p = 1; p < packing; ++p) {
        offsets.back() = (i * packing + p) * lane_dim;
        Value slice_p = b.create<vector::ExtractStridedSliceOp>(
            src_vec, offsets, slice_sizes, unit_strides);
        Value slice_p_i32 = b.create<arith::ExtUIOp>(
            i32_chunk_ty, b.create<arith::BitcastOp>(i_chunk_ty, slice_p));
        Value sh = I32Const(p * bitwidth, i32_chunk_ty.getShape(), b, loc);
        acc = b.create<arith::OrIOp>(acc,
                                     b.create<arith::ShLIOp>(slice_p_i32, sh));
      }
      packed_chunk = acc;
    } else {
      packed_chunk = b.create<arith::BitcastOp>(i32_chunk_ty, slice_i);
    }

    auto packed_shape = cast<VectorType>(packed_chunk.getType()).getShape();
    Value chunk_to_store = packed_chunk;
    // Reshape to match i32_view rank when split_point == 0 adds extra dims.
    // We've already verified early that any needed reshape only adds size-1
    // dims.
    if (i32_view_shape.size() > packed_shape.size()) {
      SmallVector<int64_t> reshape_vec_shape(
          i32_view_shape.size() - packed_shape.size(), 1);
      reshape_vec_shape.append(packed_shape.begin(), packed_shape.end());
      auto reshape_type = VectorType::get(reshape_vec_shape, i32_type);
      chunk_to_store = b.create<tpu::ReshapeOp>(reshape_type, packed_chunk);
    }
    store_indices[stride_dim] =
        b.create<arith::AddIOp>(split_base_idx, IdxConst(i, b, loc));

    b.create<tpu::StridedStoreOp>(chunk_to_store, i32_view, store_indices,
                                  strides);
  }

  raw_op.erase();
  shape_cast_op->erase();
}

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

// Given a value computed in higher precision, checks if it can be computed
// losslessly in low precision. For now, only match graphs where intermediate
// nodes have only a single use, though this restriction can be removed by
// checking for truncf ops in the descendants of additional uses. We assume that
// the hardware supports low precision ops that will always make this a net win.
std::optional<Value> maybeComputeInLowPrecision(
    Value value, mlir::Type low_precision_type,
    SmallVector<Operation*>& old_ops, SmallVector<Operation*>& new_ops) {
  auto op = value.getDefiningOp();
  if (!op->hasOneUse()) {
    return std::nullopt;
  }
  old_ops.push_back(op);
  if (isa<arith::ExtFOp, arith::ExtSIOp>(op)) {
    auto vector_type =
        dyn_cast<mlir::VectorType>(op->getOperand(0).getType());
    if (!vector_type) {
      return std::nullopt;
    }
    auto type_in = vector_type.getElementType();
    int src_bitwidth = type_in.getIntOrFloatBitWidth();
    int dst_bitwidth = low_precision_type.getIntOrFloatBitWidth();
    if (type_in == low_precision_type) {
      return op->getOperand(0);
    } else if (type_in.isSignlessInteger() &&
               low_precision_type.isSignlessInteger() &&
               src_bitwidth >= dst_bitwidth) {
      ImplicitLocOpBuilder builder(op->getLoc(), op);
      Operation* new_op;
      new_op = builder.create<arith::ExtSIOp>(
          VectorType::get(vector_type.getShape(), low_precision_type),
          op->getOperand(0));
      new_ops.push_back(new_op);
      return new_op->getResult(0);
    } else {
      return std::nullopt;
    }
  } else if (auto broadcast_op = dyn_cast<vector::BroadcastOp>(op)) {
    auto newInput =
        maybeComputeInLowPrecision(broadcast_op.getSource(), low_precision_type,
                                   old_ops, new_ops);
    if (!newInput.has_value()) {
      return std::nullopt;
    }
    ImplicitLocOpBuilder builder(broadcast_op.getLoc(),
                                  broadcast_op.getOperation());
    vector::BroadcastOp newOp = builder.create<vector::BroadcastOp>(
        VectorType::get(broadcast_op.getType().getShape(),
                        low_precision_type),
        *newInput);
    new_ops.push_back(newOp.getOperation());
    return newOp.getResult();
  } else if (auto shape_cast_op = dyn_cast<vector::ShapeCastOp>(op)) {
    auto newInput = maybeComputeInLowPrecision(shape_cast_op.getSource(),
                                               low_precision_type, old_ops,
                                               new_ops);
    if (!newInput.has_value()) {
      return std::nullopt;
    }
    ImplicitLocOpBuilder builder(shape_cast_op.getLoc(),
                                  shape_cast_op.getOperation());
    vector::ShapeCastOp newOp = builder.create<vector::ShapeCastOp>(
        VectorType::get(shape_cast_op.getType().getShape(),
                        low_precision_type),
        *newInput);
    new_ops.push_back(newOp.getOperation());
    return newOp.getResult();
  } else if (auto transpose_op = dyn_cast<tpu::TransposeOp>(op)) {
    auto newInput = maybeComputeInLowPrecision(transpose_op.getVector(),
                                               low_precision_type, old_ops,
                                               new_ops);
    if (!newInput.has_value()) {
      return std::nullopt;
    }
    ImplicitLocOpBuilder builder(transpose_op.getLoc(),
                                  transpose_op.getOperation());
    tpu::TransposeOp newOp = builder.create<tpu::TransposeOp>(
        VectorType::get(transpose_op.getType().getShape(),
                        low_precision_type),
        *newInput, transpose_op.getPermutation());
    new_ops.push_back(newOp.getOperation());
    return newOp.getResult();
  } else if (auto select_op = dyn_cast<arith::SelectOp>(op)) {
    auto newTrue = maybeComputeInLowPrecision(select_op.getTrueValue(),
                                              low_precision_type, old_ops,
                                              new_ops);
    auto newFalse = maybeComputeInLowPrecision(select_op.getFalseValue(),
                                               low_precision_type, old_ops,
                                               new_ops);
    if (!newTrue.has_value() || !newFalse.has_value()) {
      return std::nullopt;
    }
    ImplicitLocOpBuilder builder(select_op.getLoc(),
                                  select_op.getOperation());
    arith::SelectOp newOp = builder.create<arith::SelectOp>(
        select_op.getCondition(), *newTrue, *newFalse);
    new_ops.push_back(newOp.getOperation());
    return newOp.getResult();
  } else {
    return std::nullopt;
  }
}

struct PreCanonicalizationOptimizationPass
    : impl::PreCanonicalizationOptimizationPassBase<
          PreCanonicalizationOptimizationPass> {
  PreCanonicalizationOptimizationPass(int hardware_generation_p,
                                      bool compatibility_mode_p,
                                      std::array<int64_t, 2> target_shape_p)
      : hardware_generation_(hardware_generation_p),
        compatibility_mode_(compatibility_mode_p),
        target_shape_(target_shape_p) {}

  void runOnOperation() override {
    // Single-pass traversal to apply all optimizations.
    getOperation().walk([&](Operation* op) {
      if (auto matmul = dyn_cast<tpu::MatmulOp>(op)) {
        // We only attempt this fusion if dimension numbers are present.
        if (!matmul.getDimensionNumbers().has_value()) {
          return;
        }
        ImplicitLocOpBuilder builder(matmul.getLoc(), matmul.getOperation());
        if (auto fusion_result = tryFuseRhsTranspose(matmul, builder)) {
          auto [new_rhs_val, new_dnums] = *fusion_result;

          auto new_rhs = cast<TypedValue<VectorType>>(new_rhs_val);
          // Update the matmul op in-place.
          matmul.getRhsMutable().assign(new_rhs);
          matmul.setDimensionNumbersAttr(new_dnums);
        }
      } else if (isa<vector::StoreOp, tpu::VectorStoreOp>(op)) {
        optimizeStore(hardware_generation_, target_shape_, *op);
      } else if (isa<arith::TruncFOp, arith::TruncIOp>(op)) {
        // Only apply when compatibility mode is enabled.
        if (!compatibility_mode_) {
          return;
        }

        auto out_type = dyn_cast<VectorType>(op->getResult(0).getType());
        // Only apply to vector types.
        if (!out_type) {
          return;
        }
        Value input = op->getOperand(0);
        auto low_precision_type = out_type.getElementType();

        unsigned bitwidth = low_precision_type.getIntOrFloatBitWidth();
        // Only apply to bit widths 16 and 8 on TPUs that support relevant low
        // precision ops for those bit widths.
        if (!(bitwidth == 16 && hardware_generation_ >= 5) &&
            !(bitwidth == 8 && hardware_generation_ >= 6)) {
          return;
        }

        // Construct a parallel graph of low precision ops. We won't know until
        // the end of the traversal if the new ops can be used, so store both
        // the old and new ops so we can erase the unused ones later.
        SmallVector<Operation*> old_ops;
        SmallVector<Operation*> new_ops;
        auto new_result = maybeComputeInLowPrecision(input, low_precision_type,
                                                     old_ops, new_ops);
        SmallVector<Operation*>& ops_to_erase =
            new_result.has_value() ? old_ops : new_ops;
        for (auto op : ops_to_erase) {
          op->dropAllUses();
          op->erase();
        }
        if (new_result.has_value()) {
          op->getResult(0).replaceAllUsesWith(*new_result);
          op->erase();
        }
      }
    });
  }

 private:
  int64_t hardware_generation_;
  bool compatibility_mode_;
  std::array<int64_t, 2> target_shape_;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createPreCanonicalizationOptimizationPass(int hardware_generation,
                                          bool compatibility_mode,
                                          std::array<int64_t, 2> target_shape) {
  return std::make_unique<PreCanonicalizationOptimizationPass>(
      hardware_generation, compatibility_mode, target_shape);
}

}  // namespace mlir::tpu
