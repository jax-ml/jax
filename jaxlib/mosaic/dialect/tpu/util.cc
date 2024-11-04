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

bool canReinterpretToUntiledMemref(MemRefType tiled_memref_ty,
                                   const std::array<int64_t, 2>& target_shape,
                                   bool allow_minormost_padding) {
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
  if (!allow_minormost_padding &&
      *(tiled_memref_ty.getShape().end() - 1) != target_shape[1]) {
    return false;
  }
  auto packing = 32 / tiled_memref_ty.getElementTypeBitWidth();
  return (*(tiled_memref_ty.getShape().end() - 1) <= target_shape[1] &&
          *(tiled_memref_ty.getShape().end() - 2) % packing == 0 &&
          *(tiled_layout.getTileStrides().end() - 1) == 1 &&
          *(tiled_layout.getTileStrides().end() - 2) == 1);
}

}  // namespace mlir::tpu
