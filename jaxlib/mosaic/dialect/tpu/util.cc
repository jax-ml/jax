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

#include "llvm/Support/MathExtras.h"
#include "absl/types/span.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
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

SmallVector<Value, 2> getDynamicSizesFromSlicedMemref(Value value) {
  if (auto erase_op = value.getDefiningOp<tpu::EraseLayoutOp>()) {
    value = erase_op.getOperand();
    if (auto slice_op = value.getDefiningOp<tpu::MemRefSliceOp>()) {
      return slice_op.getDynamicSizes();
    }
  }
  return {};
}

bool canReinterpretToUntiledMemref(MemRefType tiled_memref_ty,
                                   const std::array<int64_t, 2>& target_shape,
                                   bool allow_minormost_padding,
                                   SmallVector<Value, 2> dynamic_sizes) {
  auto tiled_layout =
      dyn_cast<tpu::TiledLayoutAttr>(tiled_memref_ty.getLayout());
  if (!tiled_layout) {
    // We expect the tiled memref to have a tiled layout.
    return false;
  }
  if (tiled_layout.getTiles().empty() ||
      tiled_layout.getTiles().front().dimensions().size() != 2 ||
      tiled_memref_ty.getRank() < 2) {
    // TODO(jevinjiang): Currently we only support >= 2D memref, we might
    // need to handle 1D memref if we find a use case.
    return false;
  }
  CHECK_EQ(tiled_memref_ty.getNumDynamicDims(), dynamic_sizes.size());
  auto rank = tiled_memref_ty.getRank();
  auto dynamic_dim_cnt = 0;
  auto packing = 32 / tiled_memref_ty.getElementTypeBitWidth();
  if (tiled_memref_ty.isDynamicDim(rank - 1)) {
    dynamic_dim_cnt += 1;
    // TODO(jevinjiang): update to support padding when we support the max bound
    // for dynamic value.
    if (!isGuaranteedDivisible(*(dynamic_sizes.end() - dynamic_dim_cnt),
                               target_shape[1])) {
      return false;
    }
  } else {
    if (!allow_minormost_padding &&
        tiled_memref_ty.getShape()[rank - 1] != target_shape[1]) {
      return false;
    }
  }
  if (tiled_memref_ty.isDynamicDim(rank - 2)) {
    dynamic_dim_cnt += 1;
    if (!isGuaranteedDivisible(*(dynamic_sizes.end() - dynamic_dim_cnt),
                               packing)) {
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
}  // namespace mlir::tpu
