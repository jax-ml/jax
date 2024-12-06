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

#include "jaxlib/mosaic/dialect/tpu/util.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/Support/MathExtras.h"
#include "absl/types/span.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {
SmallVector<int64_t> ComputeTileStrides(MemRefType memref_ty,
                                        absl::Span<const int64_t> tiling) {
  SmallVector<int64_t> tile_strides(memref_ty.getRank());
  int64_t stride = 1;
  for (int64_t i = 0; i < memref_ty.getRank(); ++i) {
    int64_t idx = memref_ty.getRank() - 1 - i;
    int64_t tiling_idx = tiling.size() - 1 - i;
    tile_strides[idx] = stride;
    if (tiling_idx >= 0) {
      stride *= llvm::divideCeil(memref_ty.getShape()[idx], tiling[tiling_idx]);
    } else {
      stride *= memref_ty.getShape()[idx];
    }
  }
  return tile_strides;
}

std::optional<std::pair<bool, bool>> isTransposedMatmul(
    DotDimensionNumbersAttr dim_numbers) {
  auto lhs_contracting_dims = dim_numbers.getLhsContractingDims();
  auto rhs_contracting_dims = dim_numbers.getRhsContractingDims();
  auto lhs_non_contracting_dims = dim_numbers.getLhsNonContractingDims();
  auto rhs_non_contracting_dims = dim_numbers.getRhsNonContractingDims();

  if (lhs_contracting_dims.size() != 1 || rhs_contracting_dims.size() != 1 ||
      lhs_non_contracting_dims.size() != 1 ||
      rhs_non_contracting_dims.size() != 1) {
    return std::nullopt;
  }

  int64_t lhs_non_contracting_dim = lhs_non_contracting_dims[0];
  int64_t lhs_contracting_dim = lhs_contracting_dims[0];
  int64_t rhs_non_contracting_dim = rhs_non_contracting_dims[0];
  int64_t rhs_contracting_dim = rhs_contracting_dims[0];

  bool lhs_transposed = lhs_non_contracting_dim > lhs_contracting_dim;

  bool rhs_transposed = rhs_contracting_dim > rhs_non_contracting_dim;

  return std::pair<bool, bool>{lhs_transposed, rhs_transposed};
}

bool canReinterpretToUntiledMemref(TypedValue<MemRefType> tiled_memref,
                                   const std::array<int64_t, 2>& target_shape,
                                   bool allow_minormost_padding) {
  MemRefType tiled_memref_ty = tiled_memref.getType();
  auto tiled_layout =
      dyn_cast<tpu::TiledLayoutAttr>(tiled_memref_ty.getLayout());
  ValueRange dynamic_sizes = {};
  if (!tiled_layout) {
    if (auto erase_op = tiled_memref.getDefiningOp<tpu::EraseLayoutOp>()) {
      tiled_memref = erase_op.getOperand();
      tiled_memref_ty = tiled_memref.getType();
      tiled_layout =
          dyn_cast<tpu::TiledLayoutAttr>(tiled_memref_ty.getLayout());
      // TODO(b/375641258): Currently we rely on the pattern `slice ->
      // (squeeze)* -> eraseLayout` to get the dynamic sizes, but other patterns
      // may not work here: eg., slice -> eraseLayout -> reshape ->
      // eraseLayout`. We should fix this! For now, if we can not get the
      // expected dynamic sizes, we consider the memref cannot be reinterpreted
      // to untiled.
      Value ref = tiled_memref;
      while (auto squeeze_op = ref.getDefiningOp<tpu::MemRefSqueezeOp>()) {
        ref = squeeze_op.getInput();
      }
      if (auto slice_op = ref.getDefiningOp<tpu::MemRefSliceOp>()) {
        dynamic_sizes = slice_op.getDynamicSizes();
      }
    }
  }
  if (!tiled_layout) {
    // We expect the tiled memref to have a tiled layout.
    return false;
  }
  if (tiled_memref_ty.getNumDynamicDims() != dynamic_sizes.size()) {
    return false;
  }
  if (tiled_layout.getTiles().empty() ||
      tiled_layout.getTiles().front().dimensions().size() != 2 ||
      tiled_memref_ty.getRank() < 2) {
    // TODO(b/375642202): Currently we only support >= 2D memref, we might
    // need to handle 1D memref if we find a use case.
    return false;
  }
  auto rank = tiled_memref_ty.getRank();
  auto packing = 32 / tiled_memref_ty.getElementTypeBitWidth();
  if (tiled_memref_ty.isDynamicDim(rank - 1)) {
    // TODO(jevinjiang): we can still allow the minormost padding if we know the
    // max bound of the dynamic size is not larger than the target_shape[1].
    if (!isGuaranteedDivisible(dynamic_sizes.back(), target_shape[1])) {
      return false;
    }
    dynamic_sizes = dynamic_sizes.drop_back();
  } else {
    if (!allow_minormost_padding &&
        tiled_memref_ty.getShape()[rank - 1] != target_shape[1]) {
      return false;
    }
  }
  if (tiled_memref_ty.isDynamicDim(rank - 2)) {
    if (!isGuaranteedDivisible(dynamic_sizes.back(), packing)) {
      return false;
    }
  } else {
    if (tiled_memref_ty.getShape()[rank - 2] % packing != 0) {
      return false;
    }
  }
  // Check if the minormost dim has a single tile.
  return *(tiled_layout.getTileStrides().end() - 1) == 1 &&
         *(tiled_layout.getTileStrides().end() - 2) == 1;
}

bool HasMemorySpace(MemRefType ty, tpu::MemorySpace space) {
  auto memory_space =
      dyn_cast_or_null<tpu::MemorySpaceAttr>(ty.getMemorySpace());
  return memory_space && memory_space.getValue() == space;
}
}  // namespace mlir::tpu
