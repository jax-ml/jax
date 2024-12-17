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

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir {
namespace tpu {

LogicalResult UnrollVectorsOp::canonicalize(UnrollVectorsOp op,
                                            PatternRewriter &rewriter) {
  RollVectorsOp roll_op =
      dyn_cast_or_null<RollVectorsOp>(op.getOperand().getDefiningOp());
  if (!roll_op) {
     return failure();
  }
  if (roll_op.getNumOperands() != op.getNumResults()) {
     return failure();
  }
  for (auto [v1, v2] :
       llvm::zip(roll_op.getOperandTypes(), op.getResultTypes())) {
    if (v1 != v2) {
       return failure();
    }
  }
  rewriter.replaceOp(op, roll_op.getOperands());
  return success();
}

LogicalResult BitcastOp::verify() {
  auto in_ty = getInput().getType();
  auto out_ty = getOutput().getType();
  auto in_bitwidth = in_ty.getElementTypeBitWidth();
  auto out_bitwidth = out_ty.getElementTypeBitWidth();
  if (in_bitwidth != out_bitwidth) {
    if (in_ty.getRank() < 2 || out_ty.getRank() < 2) {
      return emitError(
          "Not implemented: bitcast between different bitwidths on a 1D "
          "vector.");
    }
    SmallVector<int64_t, 4> in_shape(in_ty.getShape());
    SmallVector<int64_t, 4> out_shape(out_ty.getShape());
    *(in_shape.end() - 2) *= in_bitwidth;
    *(out_shape.end() - 2) *= out_bitwidth;
    if (in_shape != out_shape) {
      return emitError(
          "Expected input and output shapes are the same after multiplying the "
          "second-minor dimension by the ratio of bitwidths.");
    }
  } else if (in_ty.getShape() != out_ty.getShape()) {
    return emitError(
        "Expected input and output shapes are the same when bitwidth does not "
        "change.");
  }
  return success();
}

LogicalResult MemRefSliceOp::verify() {
  auto source_type = getMemRefType(getMemRef());
  auto target_type = getType();
  auto target_layout = target_type.getLayout();
  auto target_memory_space = target_type.getMemorySpace();
  auto indices = getBaseIdx();
  auto slice_shape = getResult().getType().getShape();
  if (!source_type.hasStaticShape()) {
    return emitOpError(
        "Only slicing of memrefs with static shapes is supported.");
  }
  auto source_shape = source_type.getShape();
  bool is_semaphore =
      HasMemorySpace(source_type, tpu::MemorySpace::kSemaphoreMem);
  if (is_semaphore &&
      !isa<SemaphoreType, DMASemaphoreType>(source_type.getElementType())) {
    return emitOpError(
        "References to semaphore memory space must have a semaphore element "
        "type.");
  }
  if (indices.size() != slice_shape.size() ||
      indices.size() != source_shape.size()) {
    return emitOpError("Indices and slice shapes must match.");
  }
  // TODO(apaszke): Check that the result has a smaller shape.
  // TODO(apaszke): Check that strides are equivalent.
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type. Note that MemRefType does not allow a null layout so we treat
  // the default identity affine map as an "unset" value instead.
  return success(
      (target_memory_space == nullptr ||
       target_memory_space == source_type.getMemorySpace()) &&
      ((isa<AffineMapAttr>(target_layout) && target_layout.isIdentity()) ||
       target_type.getLayout() == source_type.getLayout()) &&
      getDynamicSizes().size() == target_type.getNumDynamicDims());
}

LogicalResult MemRefSliceOp::canonicalize(MemRefSliceOp op,
                                          PatternRewriter &rewriter) {
  auto erase_layout = op.getMemRef().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout) {
    return failure();
  }
  // Push layout erasure through slicing. It is important we see the layout
  // for lowering and don't make it hard for other ops to query it.
  auto layout_ref = erase_layout.getOperand();
  MemRefType layout_ty = layout_ref.getType();
  auto new_result_type = MemRefType::get(
      op.getResult().getType().getShape(), layout_ty.getElementType(),
      layout_ty.getLayout(), layout_ty.getMemorySpace());
  auto slice =
      rewriter.create<MemRefSliceOp>(op.getLoc(), new_result_type, layout_ref,
                                     op.getBaseIdx(), op.getDynamicSizes());
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), slice);
  return success();
}

LogicalResult MemRefSqueezeOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type.
  if (target_type.getMemorySpace() != nullptr &&
      target_type.getMemorySpace() != source_type.getMemorySpace()) {
    emitOpError("Memory spaces do not match.");
    return failure();
  }
  if (target_type.getElementType() != source_type.getElementType()) {
    this->emitOpError("Element types don't match.");
    return failure();
  }
  auto source_shape = source_type.getShape();
  auto target_shape = target_type.getShape();
  int source_index = source_shape.size() - 1;
  int target_index = target_shape.size() - 1;
  auto error_msg = llvm::formatv(
      "Target shape is not valid. "
      "Source type: {0}. Target type: {1}.",
      source_type, target_type);
  while (source_index >= 0 || target_index >= 0) {
    int target_dim = target_index < 0 ? -1 : target_shape[target_index];
    if (source_index < 0) {
       // We have run out of source shape but target shape still remains.
       emitOpError(error_msg);
       return failure();
    }
    int source_dim = source_shape[source_index];
    if (source_dim == target_dim) {
       source_index--;
       target_index--;
    } else {
       // Only the source dim can be 1 here.
       if (source_dim != 1) {
         this->emitOpError(error_msg);
         return failure();
       }
       source_index--;
    }
  }
  return success();
}

LogicalResult MemRefSqueezeOp::canonicalize(MemRefSqueezeOp op,
                                            PatternRewriter &rewriter) {
  auto source_type = getMemRefType(op.getInput());
  auto target_type = op.getType();
  auto erase_layout = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout) {
    return failure();
  }
  // Push layout erasure through squeezing. It is important we see the layout
  // for lowering and don't make it hard for other ops to query it.
  auto layout_ref = erase_layout.getOperand();
  MemRefType layout_ty = layout_ref.getType();
  auto source_shape = source_type.getShape();
  auto target_shape = target_type.getShape();
  int source_index = source_shape.size() - 1;
  int target_index = target_shape.size() - 1;
  auto old_layout = dyn_cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  auto target_strides = old_layout.getTileStrides();
  SmallVector<int64_t> tile_strides(target_strides.begin(),
                                    target_strides.end());
  // We want to remove all strides that correspond to squeezed dimensions and
  // update the corresponding output layout.
  while (source_index >= 0 || target_index >= 0) {
    int target_dim = target_index < 0 ? -1 : target_shape[target_index];
    int source_dim = source_shape[source_index];
    if (source_dim == target_dim) {
       source_index--;
       target_index--;
    } else {
       // Source index must be 1 here (otherwise verification will have failed).
       // We are safe to mutate the strides vector here because we are looping
       // backwards.
       tile_strides.erase(tile_strides.begin() + source_index);
       source_index--;
    }
  }
  auto new_layout = tpu::TiledLayoutAttr::get(
      source_type.getContext(), old_layout.getTiles(), tile_strides);
  auto new_result_type = MemRefType::get(op.getResult().getType().getShape(),
                                         layout_ty.getElementType(), new_layout,
                                         layout_ty.getMemorySpace());
  auto squeeze = rewriter.create<MemRefSqueezeOp>(op.getLoc(), new_result_type,
                                                  layout_ref);
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), squeeze);
  return success();
}

LogicalResult MemRefReshapeOp::verify() {
  auto src_ty = getMemRefType(getInput());
  auto tgt_ty = getType();
  if (tgt_ty.getMemorySpace() != nullptr &&
      tgt_ty.getMemorySpace() != src_ty.getMemorySpace()) {
    return emitOpError("Memory spaces do not match.");
  }
  if (src_ty.getShape().size() < 2 || tgt_ty.getShape().size() < 2) {
    return emitError("Not implemented: 1d memref reshape.");
  }
  if (tgt_ty.getElementType() != src_ty.getElementType()) {
    return emitOpError("Element types don't match.");
  }
  auto src_elements_num = ShapedType::getNumElements(src_ty.getShape());
  auto tgt_elements_num = ShapedType::getNumElements(tgt_ty.getShape());
  if (src_elements_num != tgt_elements_num) {
    return emitOpError(
        "Number of elements doesn't match between input and output memref "
        "type.");
  }
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type.
  auto tgt_layout = dyn_cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  if (!tgt_layout) {
    return success();
  }
  auto src_layout = dyn_cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  if (!src_layout || src_layout.getTiles().empty()) {
    return emitOpError("Expected a tiled layout for the input memref.");
  }
  if (src_layout.getTiles() != tgt_layout.getTiles()) {
    return emitOpError(
        "Expected the same tiling for the input and output memref.");
  }
  auto tile = src_layout.getTiles().front().dimensions();
  if (tile.size() != 2) {
    return emitOpError("Not implemented: memref reshape with 1D tiling.");
  }
  SmallVector<int64_t> src_tile_strides(src_layout.getTileStrides());
  if (ComputeTileStrides(src_ty, tile) != src_tile_strides) {
    return emitOpError("Not implemented: reshape on a non-contiguous memref.");
  }
  auto src_tiled_shape = src_ty.getShape().take_back(2);
  auto tgt_tiled_shape = tgt_ty.getShape().take_back(2);
  bool is_src_align_tile_2nd_minor = src_tiled_shape[0] % tile[0] == 0;
  bool is_src_align_tile_minor = src_tiled_shape[1] % tile[1] == 0;
  bool is_tgt_align_tile_2nd_minor = tgt_tiled_shape[0] % tile[0] == 0;
  bool is_tgt_align_tile_minor = tgt_tiled_shape[1] % tile[1] == 0;
  if (tile[0] == 1 && is_src_align_tile_minor && is_tgt_align_tile_minor) {
    // When the tiling is (1, ?) and the source and target shapes are aligned
    // to the tile, we support reshape on any dims.
  } else if (tgt_tiled_shape[1] != src_tiled_shape[1]) {
    return emitError("Expected the minormost dimension to be unchanged");
  } else if (tgt_tiled_shape[0] != src_tiled_shape[0]) {
    if (!is_src_align_tile_2nd_minor || !is_tgt_align_tile_2nd_minor) {
      return emitError(
          "Expected the 2nd minor dimension is aligned to the tile");
    }
  }
  return success();
}

LogicalResult MemRefReshapeOp::canonicalize(MemRefReshapeOp op,
                                            PatternRewriter &rewriter) {
  auto src_ty = op.getInput().getType();
  auto dst_ty = op.getType();
  auto erase_layout_op = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout_op) {
    return failure();
  }
  auto layout_ref = erase_layout_op.getOperand();
  auto layout_ty = layout_ref.getType();
  auto layout =
      dyn_cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  CHECK(!layout.getTiles().empty());
  auto tile = layout.getTiles().front().dimensions();
  auto new_tile_strides = ComputeTileStrides(dst_ty, tile);
  auto new_layout = tpu::TiledLayoutAttr::get(
      src_ty.getContext(), layout.getTiles(), new_tile_strides);
  auto new_result_ty =
      MemRefType::get(dst_ty.getShape(), dst_ty.getElementType(), new_layout,
                      layout_ty.getMemorySpace());
  auto reshape =
      rewriter.create<MemRefReshapeOp>(op.getLoc(), new_result_ty, layout_ref);
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), reshape);
  return success();
}

LogicalResult MemRefBitcastOp::verify() {
  auto src_ty = getMemRefType(getInput());
  auto tgt_ty = getType();
  if (tgt_ty.getMemorySpace() != nullptr &&
      tgt_ty.getMemorySpace() != src_ty.getMemorySpace()) {
    return emitOpError("Memory spaces do not match.");
  }
  if (src_ty.getRank() != tgt_ty.getRank()) {
    return emitOpError("Ranks do not match.");
  }
  if (src_ty.getRank() <= 1) {
    return emitOpError("Not implemented: 1d memref bitcast.");
  }
  auto src_bitwidth = src_ty.getElementTypeBitWidth();
  auto tgt_bitwidth = tgt_ty.getElementTypeBitWidth();
  for (int i = 0; i < src_ty.getRank(); ++i) {
    auto src_dim_size = src_ty.getDimSize(i);
    auto tgt_dim_size = tgt_ty.getDimSize(i);
    if (i == src_ty.getRank() - 2) {
      auto src_bits = src_dim_size * src_bitwidth;
      auto tgt_bits = tgt_dim_size * tgt_bitwidth;
      if (src_bits != tgt_bits) {
        return emitOpError(
                   "Expected the same number of bits on the 2nd minormost "
                   "dim: (")
               << src_dim_size << " * " << src_bitwidth << ") vs ("
               << tgt_dim_size << " * " << tgt_bitwidth << ")";
        ;
      }
    } else {
      if (src_dim_size != tgt_dim_size) {
        return emitOpError("Expected the same dim size on dim ")
               << i << ": " << src_dim_size << " vs " << tgt_dim_size;
      }
    }
  }
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type.
  auto tgt_layout = dyn_cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  if (!tgt_layout) {
    return success();
  }
  auto src_layout = dyn_cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  if (!src_layout) {
    return emitOpError("Expected a tiled layout for the input memref.");
  }
  // TODO(jevinjiang): verify memref tiling is valid. Here we just assume the
  // source and target tilings are valid.
  auto src_tile = src_layout.getTiles().front().dimensions();
  auto tgt_tile = tgt_layout.getTiles().front().dimensions();
  if (src_tile[0] * src_bitwidth != tgt_tile[0] * tgt_bitwidth) {
    return emitOpError("Invalid memref bitcast.");
  }
  return success();
}

LogicalResult MemRefBitcastOp::canonicalize(MemRefBitcastOp op,
                                            PatternRewriter &rewriter) {
  auto src_ty = op.getInput().getType();
  auto dst_ty = op.getType();
  if (src_ty == dst_ty) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  auto erase_layout_op = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout_op) {
    return failure();
  }
  auto src_bitwidth = src_ty.getElementTypeBitWidth();
  auto tgt_bitwidth = dst_ty.getElementTypeBitWidth();
  auto layout_ref = erase_layout_op.getOperand();
  auto layout_ty = layout_ref.getType();
  auto layout = cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  CHECK(!layout.getTiles().empty());
  auto tile = layout.getTiles().front().dimensions();
  if (tile[0] * src_bitwidth % tgt_bitwidth != 0) {
    return failure();
  }
  SmallVector<xla::Tile, 2> new_tiles =
      {xla::Tile({tile[0] * src_bitwidth / tgt_bitwidth, 128})};
  if (tgt_bitwidth < 32) {
    new_tiles.push_back(xla::Tile({32 / tgt_bitwidth, 1}));
  }
  auto new_layout = tpu::TiledLayoutAttr::get(src_ty.getContext(), new_tiles,
                                              layout.getTileStrides());
  auto new_result_ty =
      MemRefType::get(dst_ty.getShape(), dst_ty.getElementType(), new_layout,
                      layout_ty.getMemorySpace());
  auto bitcast =
      rewriter.create<MemRefBitcastOp>(op.getLoc(), new_result_ty, layout_ref);
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), bitcast);
  return success();
}

template <typename Op>
LogicalResult verifyStridedOp(Op op, MemRefType memref_ty,
                              VectorType vector_ty) {
  auto indices = op.getIndices();
  auto strides = op.getStrides();
  if (memref_ty.getRank() != indices.size()) {
    op.emitError("Base memref's rank and indices size do not match: ")
        << memref_ty.getRank() << " vs " << indices.size();
    return failure();
  }
  if (memref_ty.getRank() != strides.size()) {
    op.emitError("Base memref's rank and strides size do not match: ")
        << memref_ty.getRank() << " vs " << strides.size();
    return failure();
  }
  if (memref_ty.getRank() != vector_ty.getRank()) {
    op.emitError("Base memref's rank and result's rank do not match: ")
        << memref_ty.getRank() << " vs " << vector_ty.getRank();
    return failure();
  }
  for (int64_t i = 0; i < memref_ty.getRank(); ++i) {
    if (strides[i] < 1) {
      op.emitError("Strides[") << i << "]=" << strides[i] << " must be >= 1";
      return failure();
    }
  }
  return success();
}

LogicalResult StridedLoadOp::verify() {
  return verifyStridedOp<StridedLoadOp>(*this, getMemRefType(getBase()),
                                        getType());
}

LogicalResult StridedStoreOp::verify() {
  return verifyStridedOp<StridedStoreOp>(*this, getMemRefType(getBase()),
                                         getValueToStore().getType());
}

LogicalResult VectorStoreOp::verify() {
  if (!getStrides().empty()) {
    return emitError("Not implemented: general vector store with strides.");
  }
  VectorType value_ty = getValueToStore().getType();
  MemRefType ref_ty = getBase().getType();

  if (value_ty.getElementType() != ref_ty.getElementType()) {
    return emitOpError(
        "Expected base and valueToStore element type should match");
  }
  if (llvm::size(getIndices()) != ref_ty.getRank()) {
    return emitOpError("Expected ") << ref_ty.getRank() << " indices";
  }
  if (getMask()) {
    if (value_ty.getElementTypeBitWidth() != 32) {
      return emitError(
          "Not implemented: masked store with non-32-bit element type");
    }
    if (value_ty.getShape() != getMask().getType().getShape())
      return emitOpError("Expected valueToStore shape to match mask shape");
  }
  return success();
}

LogicalResult ReinterpretCastOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();
  return success(
      source_type.getMemorySpace() &&  // Require memory space annotations.
      source_type.getMemorySpace() == target_type.getMemorySpace());
}

template <typename Op>
LogicalResult verifyRotateOp(Op op) {
  auto vty = op.getResult().getType();
  if (vty.getRank() <= op.getDimension() || op.getDimension() < 0) {
    op.emitOpError("Invalid dimension: ") << op.getDimension();
    return failure();
  }
  if (op.getStride().has_value() && op.getStride().value() < 0) {
    op.emitOpError("Rotate stride must be >= 0 if it is specified");
    return failure();
  }
  if (op.getStrideDimension().has_value() &&
      (vty.getRank() <= op.getStrideDimension().value() ||
       op.getStrideDimension().value() < 0)) {
    op.emitOpError("Invalid stride dimension: ")
        << op.getStrideDimension().value();
    return failure();
  }
  if (op.getStride().has_value() != op.getStrideDimension().has_value()) {
    op.emitOpError(
        "Expected either none or both stride and stride dimension are "
        "present");
    return failure();
  }
  return success();
}

// TODO(b/347016737): deprecate static rotate
LogicalResult RotateOp::verify() { return verifyRotateOp<RotateOp>(*this); }

LogicalResult DynamicRotateOp::verify() {
  return verifyRotateOp<DynamicRotateOp>(*this);
}

// a + matmul(l, r, 0) == matmul(l, r, a)
template <typename AddOp>
class CanonicalizeAddOfMatmul : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
    auto try_canonicalize = [&](Value maybe_matmul, Value maybe_acc) {
      auto matmul = dyn_cast_if_present<MatmulOp>(maybe_matmul.getDefiningOp());
      if (!matmul || !matmul->hasOneUse()) {
        return failure();
      }
      if (auto const_acc = matmul.getAcc().getDefiningOp<arith::ConstantOp>();
          const_acc &&
          const_acc.getValue() == rewriter.getZeroAttr(const_acc.getType())) {
        IRMapping remap;
        remap.map(matmul.getAcc(), maybe_acc);
        Operation *new_matmul = rewriter.clone(*matmul, remap);
        rewriter.replaceOp(op, new_matmul->getResult(0));
        return success();
      }
      return failure();
    };
    // We tried try_canonicalize(op.getRhs(), op.getLhs()) and it caused
    // worrying numerical differences in some of kernels.
    return try_canonicalize(op.getLhs(), op.getRhs());
  }
};

LogicalResult MatmulOp::verify() {
  // Note - this is not yet an exhaustive verification of matmul. Many of the
  // invariants are spread across infer, apply, llo and below. This is,
  // however, a good start and the recommended place to add more invariants.
  const VectorType lhs_ty = getLhs().getType();
  const VectorType rhs_ty = getRhs().getType();
  const VectorType acc_ty = getAcc().getType();
  const VectorType res_ty = getResult().getType();
  if (acc_ty != res_ty) {
    return emitOpError(
        "Not implemented: matmul acc and result have different types");
  }
  if (acc_ty.getElementTypeBitWidth() != 32) {
    return emitOpError("Expected matmul acc to be 32-bit");
  }

  if (getTransposeLhs()) {
    emitOpError(
        "Lhs transpose not supported via this API - please use the "
        "dimension numbers API.");
    return failure();
  }

  if (getDimensionNumbers().has_value()) {
    auto dimension_numbers = getDimensionNumbers().value();
    auto lhs_contracting_dims = dimension_numbers.getLhsContractingDims();
    auto rhs_contracting_dims = dimension_numbers.getRhsContractingDims();
    if (lhs_contracting_dims.size() != 1) {
      emitOpError("Not implemented: lhs contracting dims must be of size 1");
      return failure();
    }
    if (rhs_contracting_dims.size() != 1) {
      emitOpError("Not implemented: rhs contracting dims must be of size 1");
      return failure();
    }

    auto lhs_contracting_dim = lhs_contracting_dims[0];
    auto rhs_contracting_dim = rhs_contracting_dims[0];

    auto lhs_batch_dims = dimension_numbers.getLhsBatchDims();
    auto rhs_batch_dims = dimension_numbers.getRhsBatchDims();

    auto lhs_non_contracting_dims =
        dimension_numbers.getLhsNonContractingDims();
    auto rhs_non_contracting_dims =
        dimension_numbers.getRhsNonContractingDims();

    if (lhs_contracting_dims.size() + lhs_non_contracting_dims.size() +
            lhs_batch_dims.size() !=
        lhs_ty.getShape().size()) {
      emitOpError(
          "Not implemented: lhs contracting + non contracting + batch dims "
          "must be of the same size as the lhs shape");
      return failure();
    }
    if (rhs_contracting_dims.size() + rhs_non_contracting_dims.size() +
            rhs_batch_dims.size() !=
        rhs_ty.getShape().size()) {
      emitOpError(
          "Not implemented: rhs contracting + non contracting + batch dims "
          "must be of the same size as the rhs shape");
      return failure();
    }

    if (lhs_ty.getShape()[lhs_contracting_dim] !=
        rhs_ty.getShape()[rhs_contracting_dim]) {
      emitOpError(
          "Not implemented: lhs and rhs contracting dims must be of the same "
          "size");
      return failure();
    }

    if (lhs_batch_dims.size() != rhs_batch_dims.size()) {
      emitOpError(
          "Not implemented: lhs and rhs should have the same number of batch "
          "dims");
      return failure();
    }
    if (lhs_batch_dims.size() > 1) {
      emitOpError("Not implemented: Up to 1 batch dim supported");
      return failure();
    }

    int64_t lhs_rank = lhs_ty.getShape().size();
    int64_t rhs_rank = rhs_ty.getShape().size();

    std::vector<bool> seen_dims_lhs(lhs_rank, false);
    std::vector<bool> seen_dims_rhs(rhs_rank, false);

    auto check_and_mark_dims = [&](const std::vector<int64_t> &dims,
                                   std::vector<bool> &seen_dims,
                                   const std::string_view operand) {
      for (int64_t dim : dims) {
        if (seen_dims[dim]) {
          emitOpError("Illegal: Dim ")
              << dim << " repeats in dimension numbers of " << operand;
          return failure();
        }
        seen_dims[dim] = true;
      }
      return success();
    };

    if (failed(
            check_and_mark_dims(lhs_contracting_dims, seen_dims_lhs, "lhs")) ||
        failed(check_and_mark_dims(lhs_non_contracting_dims, seen_dims_lhs,
                                   "lhs")) ||
        failed(check_and_mark_dims(lhs_batch_dims, seen_dims_lhs, "lhs"))) {
      return failure();
    }

    if (failed(
            check_and_mark_dims(rhs_contracting_dims, seen_dims_rhs, "rhs")) ||
        failed(check_and_mark_dims(rhs_non_contracting_dims, seen_dims_rhs,
                                   "rhs")) ||
        failed(check_and_mark_dims(rhs_batch_dims, seen_dims_rhs, "rhs"))) {
      return failure();
    }

    for (int64_t dim = 0; dim < lhs_rank; ++dim) {
      if (!seen_dims_lhs[dim]) {
        emitOpError("Illegal: Dim ")
            << dim << " is not seen in lhs dimension numbers";
        return failure();
      }
    }
    for (int64_t dim = 0; dim < rhs_rank; ++dim) {
      if (!seen_dims_rhs[dim]) {
        emitOpError("Illegal: Dim ")
            << dim << " is not seen in rhs dimension numbers";
      }
    }

    const std::optional<int64_t> batch_dim_lhs =
        lhs_batch_dims.empty() ? std::nullopt
                               : std::optional<int64_t>(lhs_batch_dims[0]);
    const std::optional<int64_t> batch_dim_rhs =
        rhs_batch_dims.empty() ? std::nullopt
                               : std::optional<int64_t>(rhs_batch_dims[0]);
    if (batch_dim_lhs != batch_dim_rhs) {
      emitOpError("Not Implemented: batch dims must be equal");
      return failure();
    }
    if (batch_dim_lhs.has_value() && (batch_dim_lhs.value() != 0)) {
      emitOpError("Not Implemented: batch dims pos must be 0");
      return failure();
    }
    // Invariant above enforces only 1 batch dim atm, and that both are eq
    std::optional<int64_t> batch_size = std::nullopt;
    if (batch_dim_lhs.has_value()) {
      batch_size = lhs_ty.getShape()[batch_dim_lhs.value()];
      auto rhs_batch_size = rhs_ty.getShape()[batch_dim_rhs.value()];
      if (batch_size != rhs_batch_size) {
        emitOpError("Not Implemented: batch dims must be equal");
        return failure();
      }
      if (batch_size == 0) {
        emitOpError("Illegal: batch size must be > 0");
        return failure();
      }
    }
    auto output_dim_order = dimension_numbers.getOutputDimOrder();
    if (output_dim_order.size() % 2 != 0) {
      emitOpError(
          "Illegal: output dim order must have an even number of elements.");
      return failure();
    }
    if (batch_size.has_value()) {
      if (output_dim_order[0] != 0 || output_dim_order[1] != 0) {
        emitOpError(
            "Not implemented: Output with batch size must be the lhs 0 idx for "
            "now.");
        return failure();
      }
    }

    // Invariants above enforce a single batch idx for now, and that it is in
    // position 0. Future extensions to this will be to:
    // 1. Support multiple batch dims
    // 2. Support batch dims in any position in the output dim order
    if (lhs_non_contracting_dims.size() != 1) {
      emitOpError(
          "Not implemented: lhs non contracting dims must be of size 1");
      return failure();
    }
    if (rhs_non_contracting_dims.size() != 1) {
      emitOpError(
          "Not implemented: rhs non contracting dims must be of size 1");
      return failure();
    }

    // A bit long winded, but the invariants we enforce below are:
    // 1. The output order idx is 0 (lhs) or 1 (rhs)
    // 2. The output dim order is in valid bounds
    // 3. We saw the rhs and lhs non contracting dims in the output dim order
    // 4. We never see the contracting dims in the output dim order
    // 5. We only see each of the non contracting dim once
    std::vector<bool> lhs_dims_seen_in_output(lhs_rank, false);
    std::vector<bool> rhs_dims_seen_in_output(rhs_rank, false);

    // Iterate over the output dimension order
    for (int dim_pos = 0; dim_pos < output_dim_order.size(); dim_pos += 2) {
      auto idx = output_dim_order[dim_pos];
      auto dim = output_dim_order[dim_pos + 1];

      if (idx != 0 && idx != 1) {
        emitOpError("Illegal: output dim order index must be 0 or 1");
        return failure();
      }
      auto is_lhs = (idx == 0);

      if (is_lhs) {
        if (dim < 0 || dim >= lhs_rank) {
          emitOpError("Illegal: lhs dimension index out of bounds");
          return failure();
        }
        if (lhs_dims_seen_in_output[dim]) {
          emitOpError("Illegal: lhs dimension ")
              << dim << " appears more than once in output dim order";
          return failure();
        }
        if (dim == lhs_contracting_dim) {
          emitOpError("Illegal: contracting dimension ")
              << dim << " appears in lhs output dim order";
          return failure();
        }
        // batch_dim_lhs is either 0 or nullopt
        if (dim == batch_dim_lhs) {
          // Upstream invariants enforce that batch dim is in position 0
          // of the output dim order.
          rhs_dims_seen_in_output[dim] = true;
        }
        lhs_dims_seen_in_output[dim] = true;
      } else {
        if (dim < 0 || dim >= rhs_rank) {
          emitOpError("Illegal: rhs dimension index out of bounds");
          return failure();
        }
        if (rhs_dims_seen_in_output[dim]) {
          emitOpError("Illegal: rhs dimension ")
              << dim << " appears more than once in output dim order";
          return failure();
        }
        if (dim == rhs_contracting_dim) {
          emitOpError("Illegal: contracting dimension ")
              << dim << " appears in rhs output dim order";
          return failure();
        }
        if (dim == batch_dim_rhs) {
          // Upstream invariants enforce that batch dim is in position 0
          // of the output dim order.
          lhs_dims_seen_in_output[dim] = true;
        }
        rhs_dims_seen_in_output[dim] = true;
      }
    }

    // Check that all dims have been seen (except contracting dims)
    for (int i = 0; i < lhs_rank; ++i) {
      if (i == lhs_contracting_dim) {
        continue;
      }
      if (!lhs_dims_seen_in_output[i]) {
        emitOpError("Illegal: lhs non-contracting dimension ")
            << i << " is not seen in output dim order";
        return failure();
      }
    }

    for (int i = 0; i < rhs_rank; ++i) {
      if (i == rhs_contracting_dim) {
        continue;
      }
      if (!rhs_dims_seen_in_output[i]) {
        emitOpError("Illegal: rhs non-contracting dimension ")
            << i << " is not seen in output dim order";
        return failure();
      }
    }
  }
  return success();
}

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<CanonicalizeAddOfMatmul<arith::AddFOp>,
               CanonicalizeAddOfMatmul<arith::AddIOp>>(context);
}

LogicalResult MaskCastOp::verify() {
  auto input_ty = getInput().getType();
  auto output_ty = getResult().getType();
  return success(input_ty.getElementType() == output_ty.getElementType() &&
                 output_ty.getRank() == 3 &&
                 (input_ty.getRank() == 2 ||
                  (input_ty.getRank() == 3 &&
                   input_ty.getDimSize(2) < output_ty.getDimSize(2))) &&
                 input_ty.getShape().take_front(2) ==
                     output_ty.getShape().take_front(2));
  return success();
}

LogicalResult GetBarrierSemaphoreOp::verify() {
  auto sem_type = getMemRefType(getResult());
  if (sem_type.getRank() != 0) {
    emitOpError("Barrier semaphore reference must be rank 0");
    return failure();
  }
  return success();
}

void SemaphoreSignalOp::build(OpBuilder &builder, OperationState &state,
                              Value semaphore, Value amount, Value device_id,
                              Value core_id) {
  build(builder, state, semaphore, amount, device_id, core_id,
        /*core_type=*/nullptr);
}

LogicalResult SemaphoreSignalOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    return emitOpError("Semaphore reference must be rank 0");
  }

  FailureOr<std::optional<CoreType>> issuing_core_type_maybe =
      GetCoreTypeOfParentFunc(**this);
  if (failed(issuing_core_type_maybe)) {
    return issuing_core_type_maybe;
  }
  CoreType issuing_core_type = issuing_core_type_maybe->value_or(CoreType::kTc);
  CoreType target_core_type = getCoreType().value_or(issuing_core_type);

  if (getCoreId() == nullptr && getDeviceId() == nullptr) {
    if (target_core_type != issuing_core_type) {
      return emitOpError(
          absl::StrFormat("Target core type (%s) must match source core type "
                          "(%s) when device_id and core_id are not specified",
                          stringifyCoreType(target_core_type),
                          stringifyCoreType(issuing_core_type)));
    }
  }
  if ((issuing_core_type == CoreType::kTc &&
       target_core_type == CoreType::kScScalarSubcore) ||
      (issuing_core_type == CoreType::kScScalarSubcore &&
       target_core_type == CoreType::kTc)) {
    return emitOpError("Signalling between TC and SC is not implemented");
  }
  return success();
}

LogicalResult SemaphoreWaitOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    return emitOpError("Semaphore reference must be rank 0");
  }
  return success();
}

LogicalResult EnqueueDMAOp::verify() {
  auto source_sem = getSourceSemaphore();
  if (source_sem) {
    auto source_sem_type = getMemRefType(getSourceSemaphore());
    if (source_sem_type.getRank() != 0) {
      return emitOpError("DMA source semaphore reference must be rank 0");
    }
  }
  auto target_sem_type = getMemRefType(getTargetSemaphore());
  if (target_sem_type.getRank() != 0) {
    return emitOpError("DMA target semaphore must be rank 0");
  }
  if (getDeviceId() || getCoreId()) {
    if (!getSourceSemaphore()) {
      return emitOpError(
          "DMA source semaphore must be specified when "
          "device_id or core_id is specified");
    }
  }
  return success();
}

LogicalResult WaitDMAOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    emitOpError("DMA wait semaphore must be rank 0");
    return failure();
  }
  return success();
}

LogicalResult RegionOp::verify() {
  for (auto result_type : getResultTypes()) {
    if (!isa<FloatType, IntegerType, VectorType, IndexType>(result_type)) {
      return emitOpError(
          "Region result must be float, int, index or a vector type.");
    }
  }
  return success();
}

LogicalResult ShuffledLoadOp::verify() {
  if (getBase().getType().getRank() != getIndices().size()) {
    return emitOpError("Base memref's rank and indices size do not match: ")
           << getBase().getType().getRank() << " vs " << getIndices().size();
  }
  if (getSublaneMask().size() != getType().getShape()[0]) {
    return emitOpError("Expected sublane mask size equals to ")
           << getType().getShape()[0] << " but got " << getSublaneMask().size();
  }
  if (getSublaneOffsets().size() != getType().getShape()[0]) {
    return emitOpError("Expected sublane offsets size equals to ")
           << getType().getShape()[0] << " but got "
           << getSublaneOffsets().size();
  }
  return success();
}

LogicalResult ShuffledLoadOp::canonicalize(ShuffledLoadOp op,
                                           PatternRewriter &rewriter) {
  bool can_convert_to_simple_load = true;
  for (int i = 0; i < op.getSublaneOffsets().size(); ++i) {
    if (op.getSublaneOffsets()[i] != i) {
      can_convert_to_simple_load = false;
      break;
    };
  }
  if (can_convert_to_simple_load) {
    rewriter.replaceOpWithNewOp<tpu::LoadOp>(
        op, op.getType(), op.getBase(), op.getIndices(), op.getSublaneMask(),
        /*sublane_stride=*/nullptr);
  }
  return success();
}

LogicalResult ShuffledStoreOp::verify() {
  if (getBase().getType().getRank() != getIndices().size()) {
    return emitOpError("Base memref's rank and indices size do not match: ")
           << getBase().getType().getRank() << " vs " << getIndices().size();
  }
  if (getValueToStore().getType().getRank() != getIndices().size()) {
    return emitOpError(
               "The rank of value to store and indices size do not match: ")
           << getBase().getType().getRank() << " vs " << getIndices().size();
  }
  if (getSublaneMask().size() != getValueToStore().getType().getShape()[0]) {
    return emitOpError("Expected sublane mask size equals to ")
           << getValueToStore().getType().getShape()[0] << " but got "
           << getSublaneMask().size();
  }
  if (getSublaneOffsets().size() != getValueToStore().getType().getShape()[0]) {
    return emitOpError("Expected sublane offsets size equals to ")
           << getValueToStore().getType().getShape()[0] << " but got "
           << getSublaneOffsets().size();
  }
  return success();
}

LogicalResult ShuffledStoreOp::canonicalize(ShuffledStoreOp op,
                                            PatternRewriter &rewriter) {
  bool can_convert_to_simple_store = true;
  for (int i = 0; i < op.getSublaneOffsets().size(); ++i) {
    if (op.getSublaneOffsets()[i] != i) {
      can_convert_to_simple_store = false;
      break;
    };
  }
  if (can_convert_to_simple_store) {
    rewriter.replaceOpWithNewOp<tpu::StoreOp>(op, op.getValueToStore(),
                                              op.getBase(), op.getIndices(),
                                              op.getSublaneMask(),
                                              /*mask=*/nullptr,
                                              /*sublane_stride=*/nullptr);
  }
  return success();
}

LogicalResult ConcatenateOp::verify() {
  auto dimension = getDimension();
  if (getOperands().size() < 2) {
    return emitOpError("Expected at least 2 operands for concatenate op.");
  }
  auto first_type = getOperand(0).getType().cast<VectorType>();
  auto first_shape = first_type.getShape();
  auto first_dtype = first_type.getElementType();
  for (auto operand : getOperands()) {
    auto vty = dyn_cast<VectorType>(operand.getType());
    if (!vty) {
      return emitOpError("Operand must be a vector type.");
    }
    auto shape = vty.getShape();
    auto dtype = vty.getElementType();
    if (dtype != first_dtype) {
      return emitOpError(
          "Not implemented:: Expected all operands to have the same element "
          "type.");
    }
    for (int dim = 0; dim < shape.size(); ++dim) {
      if (dim != dimension && shape[dim] != first_shape[dim]) {
        return emitOpError(
            "Not implemented: Expected all operands to have "
            "the same shape outside of the concat dim");
      }
    }
  }
  return success();
}

LogicalResult LogOp::verify() {
  FailureOr<std::optional<CoreType>> logging_core_type_maybe =
      GetCoreTypeOfParentFunc(**this);
  if (failed(logging_core_type_maybe)) {
    return failure();
  }
  CoreType logging_core_type = logging_core_type_maybe->value_or(CoreType::kTc);
  if ((logging_core_type == CoreType::kScScalarSubcore ||
       logging_core_type == CoreType::kScVectorSubcore) &&
      getFormattedAttr() != nullptr && getFormattedAttr().getValue()) {
    return emitOpError("Formatted logging is not supported on SC");
  }
  switch (logging_core_type) {
    case CoreType::kTc:
    case CoreType::kScScalarSubcore:
      return success();
    case CoreType::kScVectorSubcore:
      return emitOpError("Log op is not supported on the SC vector subcore");
  }
  return emitOpError(
      absl::StrFormat("Unexpected core type: %s",
                      stringifyCoreType(logging_core_type_maybe->value())));
}

LogicalResult WeirdOp::verify() {
  const mlir::Type in_type = getInput().getType();
  if (const auto in_vec_type = dyn_cast<VectorType>(in_type)) {  // Vector case.
    if (!in_vec_type.getElementType().isF32()) {
      return emitOpError("Input type must be F32");
    }
    const mlir::Type out_type = getResult().getType();
    const auto out_vec_type = dyn_cast<VectorType>(out_type);
    if (!out_vec_type) {
      return emitOpError("Output type must be a vector when input is a vector");
    }
    if (!out_vec_type.getElementType().isInteger(1)) {
      return emitOpError("Output type must be I1");
    }
  } else {  // Scalar case.
    if (!in_type.isF32()) {
      return emitOpError("Input type must be F32");
    }
    const mlir::Type out_type = getResult().getType();
    if (!out_type.isInteger(1)) {
      return emitOpError("Output type must be I1 scalar");
    }
  }
  return success();
}

}  // namespace tpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_ops.cc.inc"
