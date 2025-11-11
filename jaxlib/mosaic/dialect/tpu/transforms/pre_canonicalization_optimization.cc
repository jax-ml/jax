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
#include <memory>
#include <optional>
#include <tuple>

#include "absl/log/check.h"
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

std::optional<int64_t> canOptimizeReshapeMemory(
    int hardware_generation, std::array<int64_t, 2> target_shape,
    TypedValue<MemRefType> ref, VectorType expanded_ty,
    VectorType collapsed_ty) {
  if (expanded_ty.getRank() < 2 || collapsed_ty.getRank() < 1) {
    return std::nullopt;
  }
  const int bitwidth = expanded_ty.getElementTypeBitWidth();
  const int packing = 32 / bitwidth;
  if (hardware_generation < 4 && packing > 1) {
    return std::nullopt;
  }

  // The reshape below might be invalid if the memref is not contiguous, but it
  // is an overly conservative check (we don't need all dims to be contiguous).
  if (!isContiguousMemref(ref)) {
    return std::nullopt;
  }

  const int64_t lane = target_shape[1];
  int64_t collapsed_minor = collapsed_ty.getShape().back();
  int64_t expanded_minor = expanded_ty.getShape().back();
  // Only handle the cases where the minor dim starts out as the number of lanes
  // and we fold at least the second minor dim into it, in a way that changes
  // its shape.
  if (expanded_minor != lane ||
      collapsed_minor % (packing * lane) != 0 ||
      collapsed_minor == expanded_minor ||
      collapsed_minor < llvm::product_of(expanded_ty.getShape().take_back(2))) {
    return std::nullopt;
  }

  // We don't handle memrefs with padding.
  MemRefType ref_ty = getMemRefType(ref);
  auto tiled_layout = dyn_cast<tpu::TiledLayoutAttr>(ref_ty.getLayout());
  if (!tiled_layout || tiled_layout.getTiles().empty()) {
    return std::nullopt;
  }
  ArrayRef<int64_t> front_tile = tiled_layout.getTiles().front().dimensions();
  ArrayRef<int64_t> ref_tiled_shape =
      ref_ty.getShape().take_back(front_tile.size());
  for (int i = 0; i < front_tile.size(); ++i) {
    if (ref_tiled_shape[i] % front_tile[i]) {
      return std::nullopt;
    }
  }

  // NOTE: We could generalize this to allow only flattening part of a dimension
  int folded_dims = 0;
  {
    int suffix_size = 1;
    auto sizes_it = expanded_ty.getShape().rbegin();
    while (suffix_size < collapsed_minor) {
      suffix_size *= *(sizes_it++);
    }
    // Make sure that the minor dim is folded only from entire major dims, not
    // from a part of some minor dim.
    if (suffix_size != collapsed_minor) {
      return std::nullopt;
    }
    folded_dims = sizes_it - expanded_ty.getShape().rbegin();
  }
  DCHECK_GE(folded_dims, 2);  // Should fold at least 2nd minor into minor.

  // We don't handle slicing in the folded dims at the moment.
  if (ref_ty.getShape().take_back(folded_dims) !=
      expanded_ty.getShape().take_back(folded_dims)) {
    return std::nullopt;
  }

  return folded_dims;
}

void optimizeLoadReshape(int hardware_generation,
                         std::array<int64_t, 2> target_shape,
                         Operation& raw_op) {
  // Below, we try to look for reshapes that flatten multiple dims into the
  // lane dimension. If the source of the reshape originates from a load of a
  // ref with 128 minor dimension (effectively untiled), we can replace the
  // load/reshape sequence with an efficient strided load. In essence, the
  // strided load creates vregs with a narrow slice along the target minor
  // dimension, but with the 2nd minor dim after the reshape already in
  // sublanes. The results of strided load can be concatenated to form the
  // final vector result.
  //
  // A little extra care needs to be applied to packed types, which we handle by
  // briefly extending to 32-bit and repacking them after concatenation.
  TypedValue<VectorType> src;
  VectorType tgt_ty;
  if (auto op = dyn_cast<tpu::ReshapeOp>(&raw_op)) {
    src = op.getSource();
    tgt_ty = op.getResult().getType();
  } else if (auto op = dyn_cast<vector::ShapeCastOp>(&raw_op)) {
    src = op.getSource();
    tgt_ty = op.getResult().getType();
  } else {
    return;
  }
  VectorType src_ty = src.getType();
  ArrayRef<int64_t> src_shape = src_ty.getShape();
  ArrayRef<int64_t> tgt_shape = tgt_ty.getShape();
  const int lane = target_shape[1];
  const int bitwidth = src_ty.getElementTypeBitWidth();
  const int packing = 32 / bitwidth;

  auto load_op = dyn_cast_if_present<vector::LoadOp>(src.getDefiningOp());
  // This rewrite might not be profitable if the load has other users.
  if (!load_op || !load_op.getBase().hasOneUse()) {
    return;
  }
  TypedValue<MemRefType> ref = load_op.getBase();
  MemRefType ref_ty = getMemRefType(ref);

  auto maybe_folded_dims = canOptimizeReshapeMemory(
      hardware_generation, target_shape, ref, src_ty, tgt_ty);
  if (!maybe_folded_dims.has_value()) {
    return;
  }
  int folded_dims = *maybe_folded_dims;

  Location loc = raw_op.getLoc();
  ImplicitLocOpBuilder b(loc, &raw_op);

  // Flatten the untiled dims into second minor and bitcast to i32.
  // NOTE: Source vector shape might be different from ref shape when slicing.
  SmallVector<int64_t> mem_shape(ref_ty.getShape().drop_back(folded_dims));
  if (mem_shape.empty()) {
    mem_shape.push_back(1);
  }
  mem_shape.back() *= tgt_shape.back() / lane;
  mem_shape.push_back(lane);
  Value reshaped_ref = b.create<tpu::MemRefReshapeOp>(
      MemRefType::get(mem_shape, ref_ty.getElementType()), ref);
  *(mem_shape.end() - 2) /= packing;
  Value i32_view = b.create<tpu::MemRefBitcastOp>(
      MemRefType::get(mem_shape, b.getI32Type()), reshaped_ref);

  // Define the shape of the small i32 chunk we will load in each iteration.
  // TODO(b/458291444): The loads we emit here might use suboptimal shapes and
  // we could do better by folding some dims (as much as slicing allows).
  SmallVector<int64_t> chunk_shape(src_shape.drop_back(folded_dims));
  if (chunk_shape.empty()) {
    chunk_shape.push_back(1);
  }
  chunk_shape.push_back(lane);
  VectorType chunk_ty = VectorType::get(chunk_shape, b.getI32Type());

  SmallVector<int32_t> strides(mem_shape.size(), 1);
  const int64_t sublane_prod = tgt_shape.back() / lane;
  const int64_t stride = sublane_prod / packing;
  *(strides.end() - 2) = stride;

  // Reuse indices from the original load for the prefix.
  auto indices = load_op.getIndices();
  SmallVector<Value> idxs(indices.drop_back(folded_dims));
  if (idxs.empty()) {
    idxs.push_back(IdxConst(0, b, loc));
  }
  Value split_base_idx =
      b.create<arith::MulIOp>(idxs.back(), IdxConst(stride, b, loc));
  idxs.push_back(IdxConst(0, b, loc));

  SmallVector<Value> unpacked_chunks;
  unpacked_chunks.reserve(stride * packing);
  for (int i = 0; i < stride; ++i) {
    *(idxs.end() - 2) =
        b.create<arith::AddIOp>(split_base_idx, IdxConst(i, b, loc));
    Value chunk =
        b.create<tpu::StridedLoadOp>(chunk_ty, i32_view, idxs, strides);
    // Unpack elements from i32 if necessary.
    for (int p = 0; p < packing; ++p) {
      unpacked_chunks.push_back(b.create<arith::ShRUIOp>(
          chunk.getType(), chunk, I32Const(p * bitwidth, chunk_shape, b, loc)));
    }
  }

  Value unpacked_flat;
  if (unpacked_chunks.size() == 1) {
    unpacked_flat = unpacked_chunks.front();
  } else {
    SmallVector<int64_t> concat_shape(src_shape.drop_back(folded_dims));
    if (concat_shape.empty()) {
      concat_shape.push_back(1);
    }
    concat_shape.push_back(tgt_shape.back());
    unpacked_flat = b.create<tpu::ConcatenateOp>(
        VectorType::get(concat_shape, b.getI32Type()), unpacked_chunks,
        concat_shape.size() - 1);
  }

  Value result = unpacked_flat;
  if (packing > 1) {  // Pack back, if needed.
    result = b.create<arith::TruncIOp>(
        VectorType::get(cast<VectorType>(result.getType()).getShape(),
                        b.getIntegerType(bitwidth)),
        result);
  }
  // Bitcast to the target type, if needed.
  if (cast<VectorType>(result.getType()) != tgt_ty.getElementType()) {
    result = b.create<arith::BitcastOp>(
        VectorType::get(cast<VectorType>(result.getType()).getShape(),
                        tgt_ty.getElementType()),
        result);
  }
  // Apply the reshape to major dims, if needed.
  if (cast<VectorType>(result.getType()).getShape() != tgt_ty.getShape()) {
    result = b.create<tpu::ReshapeOp>(tgt_ty, result);
  }
  DCHECK_EQ(result.getType(), tgt_ty);

  raw_op.replaceAllUsesWith(ValueRange{result});
  raw_op.erase();
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

  // This rewrite might not be profitable if the reshape has other users.
  auto shape_cast_op =
      dyn_cast_if_present<vector::ShapeCastOp>(value_to_store.getDefiningOp());
  if (!shape_cast_op || !shape_cast_op.getResult().hasOneUse()) {
    return;
  }

  MemRefType ref_ty = getMemRefType(base);
  VectorType src_ty = shape_cast_op.getSource().getType();
  VectorType tgt_ty = shape_cast_op.getResult().getType();
  auto src_shape = src_ty.getShape();
  auto tgt_shape = tgt_ty.getShape();
  const int64_t lane = target_shape[1];
  const int bitwidth = src_ty.getElementTypeBitWidth();
  const int packing = 32 / bitwidth;

  std::optional<int> maybe_expanded_dims = canOptimizeReshapeMemory(
      hardware_generation, target_shape, base, tgt_ty, src_ty);
  if (!maybe_expanded_dims.has_value()) {
    return;
  }
  int expanded_dims = *maybe_expanded_dims;

  ImplicitLocOpBuilder b(raw_op.getLoc(), &raw_op);
  auto loc = raw_op.getLoc();
  auto i32_type = b.getI32Type();

  // Normalize source to be 2D (since the target is at least 2D anyway).
  if (src_ty.getRank() == 1) {
    std::array<int64_t, 2> new_shape{1, src_ty.getShape().back()};
    Value source_2d = b.create<vector::ShapeCastOp>(
        VectorType::get(new_shape, src_ty.getElementType()),
        shape_cast_op.getSource());
    shape_cast_op->setOperand(0, source_2d);
    src_ty = cast<VectorType>(source_2d.getType());
    src_shape = src_ty.getShape();
  }

  SmallVector<int64_t> mem_shape(ref_ty.getShape().drop_back(expanded_dims));
  if (mem_shape.empty()) {
    mem_shape.push_back(1);
  }
  mem_shape.back() *= src_shape.back() / lane;
  mem_shape.push_back(lane);

  Value reshaped_ref = b.create<tpu::MemRefReshapeOp>(
      MemRefType::get(mem_shape, ref_ty.getElementType()), base);
  *(mem_shape.end() - 2) /= packing;
  Value i32_view = b.create<tpu::MemRefBitcastOp>(
      MemRefType::get(mem_shape, i32_type), reshaped_ref);


  Value src_vec = shape_cast_op.getSource();
  SmallVector<int64_t> slice_shape(src_shape);
  slice_shape.back() = lane;
  SmallVector<int64_t> slice_strides(slice_shape.size(), 1);
  SmallVector<int64_t> slice_offsets(slice_shape.size(), 0);

  // We don't support slicing so the program only didn't contain OOB stores
  // if all indices were 0.
  SmallVector<int32_t> store_strides(mem_shape.size(), 1);
  const int64_t sublane_prod = src_shape.back() / lane;
  const int64_t stride = sublane_prod / packing;
  *(store_strides.end() - 2) = stride;

  SmallVector<Value> store_indices(indices.drop_back(expanded_dims));
  if (store_indices.empty()) {
    store_indices.push_back(IdxConst(0, b, loc));
  }
  Value second_minor_base = b.create<arith::MulIOp>(
      store_indices.back(),
      b.create<arith::ConstantOp>(b.getIndexType(),
                                  b.getI32IntegerAttr(stride)));
  store_indices.back() = nullptr;
  store_indices.push_back(IdxConst(0, b, loc));
  SmallVector<int64_t> to_store_shape(tgt_shape.drop_back(expanded_dims));
  if (to_store_shape.empty()) {
    to_store_shape.push_back(1);
  }
  to_store_shape.push_back(lane);
  auto store_vty = VectorType::get(to_store_shape, b.getI32Type());

  for (int64_t i = 0; i < stride; ++i) {
    Value packed_chunk;
    if (packing > 1) {
      auto slice_int_vty =
          VectorType::get(slice_shape, b.getIntegerType(bitwidth));
      SmallVector<Value> packed_slices;
      packed_slices.reserve(packing);
      for (int64_t p = 0; p < packing; ++p) {
        slice_offsets.back() = (i * packing + p) * lane;
        Value slice = b.create<vector::ExtractStridedSliceOp>(
            src_vec, slice_offsets, slice_shape, slice_strides);
        Value slice_narrow_int =
            b.create<arith::BitcastOp>(slice_int_vty, slice);
        packed_slices.push_back(b.create<arith::ExtSIOp>(
            VectorType::get(slice_shape, b.getI32Type()), slice_narrow_int));
      }
      packed_chunk = b.create<tpu::PackElementwiseOp>(
          VectorType::get(slice_shape, b.getI32Type()), packed_slices,
          b.getIntegerType(bitwidth));
    } else {
      slice_offsets.back() = i * packing * lane;
      Value slice = b.create<vector::ExtractStridedSliceOp>(
          src_vec, slice_offsets, slice_shape, slice_strides);
      packed_chunk = b.create<arith::BitcastOp>(
          VectorType::get(slice_shape, i32_type), slice);
    }

    // TODO(b/458291444): This reshape might end up being non-trivial and might
    // produce a vector with an unnecessarily bad layout. Consider the where
    // src_shape is (24, 1024) and tgt_shape is (8, 3, 8, 128). In that case
    // slice_shape is (24, 128), which can be neatly packed into vregs, but here
    // we would reshape to (8, 3, 128), which of course is problematic and will
    // introduce lots of padding... We could work around this by flattening the
    // ref dimensions, but it is complicated by non-contiguous slices which
    // might prevent this. In case we find a non-contiguous slice we could still
    // try unrolling into multiple strided stores.
    Value chunk_to_store = b.create<tpu::ReshapeOp>(store_vty, packed_chunk);
    CHECK_GE(store_indices.size(), 2);
    *(store_indices.end() - 2) =
        b.create<arith::AddIOp>(second_minor_base, IdxConst(i, b, loc));
    b.create<tpu::StridedStoreOp>(chunk_to_store, i32_view, store_indices,
                                  store_strides);
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

struct PreCanonicalizationOptimizationPass
    : impl::PreCanonicalizationOptimizationPassBase<
          PreCanonicalizationOptimizationPass> {
  PreCanonicalizationOptimizationPass(int hardware_generation_p,
                                      std::array<int64_t, 2> target_shape_p)
      : hardware_generation_(hardware_generation_p),
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
      } else if (isa<vector::ShapeCastOp, tpu::ReshapeOp>(op)) {
        optimizeLoadReshape(hardware_generation_, target_shape_, *op);
      }
    });
  }

 private:
  int64_t hardware_generation_;
  std::array<int64_t, 2> target_shape_;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createPreCanonicalizationOptimizationPass(int hardware_generation,
                                          std::array<int64_t, 2> target_shape) {
  return std::make_unique<PreCanonicalizationOptimizationPass>(
      hardware_generation, target_shape);
}

}  // namespace mlir::tpu
