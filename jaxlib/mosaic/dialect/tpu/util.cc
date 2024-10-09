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

bool canReinterpretToUntiledMemref(MemRefType memref_ty,
                                   const std::array<int64_t, 2>& target_shape) {
  if (auto tiled_layout =
          dyn_cast<tpu::TiledLayoutAttr>(memref_ty.getLayout())) {
    auto packing = 32 / memref_ty.getElementTypeBitWidth();
    return (memref_ty.getRank() >= 2 &&
            *(memref_ty.getShape().end() - 1) == target_shape[1] &&
            *(memref_ty.getShape().end() - 2) % packing == 0 &&
            *(tiled_layout.getTileStrides().end() - 1) == 1 &&
            *(tiled_layout.getTileStrides().end() - 2) == 1);
  }
  return false;
}
}  // namespace mlir::tpu
