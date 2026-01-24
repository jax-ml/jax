/* Copyright 2026 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFORMS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFORMS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace jax::mosaic::gpu {

enum class Rounding { kUp, kDown };

struct TileTransform {
  std::vector<int64_t> tiling;
  std::optional<Rounding> rounding;

  // Tiles a suffix of dimensions of a MemRef.
  //
  // The last `tiling.size()` dimensions of the input MemRef are tiled.
  // The result has `rank + tiling.size()` dimensions. The layout is:
  //   - Outer dimensions (untiled)
  //   - Major dimensions (quotients) for each tiled dimension
  //   - Minor dimensions (remainders/block offsets) for each tiled dimension
  //
  // If `rounding` is kDown, the tiled dimensions are sliced to be a multiple
  // of the tile size. Otherwise, the dimension size must be divisible by the
  // tile size.
  //
  // Example:
  //   ref: memref<128xf32>
  //   tiling: {32}
  //   Result: memref<4x32xf32>
  //
  //   ref: memref<10x20xf32>
  //   tiling: {2, 5}
  //   Result: memref<5x4x2x5xf32>
  absl::StatusOr<mlir::Value> Apply(mlir::ImplicitLocOpBuilder& builder,
                                    mlir::Value ref) const;

  // Transforms indices from the original MemRef to the tiled MemRef.
  //
  // The input `idx` must have the same rank as the original MemRef.
  // The output indices correspond to the dimensions produced by `Apply`:
  //   - Outer indices are copied unchanged.
  //   - Tiled indices are split into quotient and remainder:
  //       q = idx / tile_size
  //       r = idx % tile_size
  //
  // Example:
  //   idx: {15} (for memref<128xf32>)
  //   tiling: {32}
  //   Result: {0, 15}
  //
  //   idx: {35} (for memref<128xf32>)
  //   tiling: {32}
  //   Result: {1, 3}
  absl::StatusOr<std::vector<mlir::Value>> TransformIndex(
      mlir::ImplicitLocOpBuilder& builder,
      const std::vector<mlir::Value>& idx) const;

  // Computes the shape of the tiled MemRef given the original shape.
  //
  // Example:
  //   shape: {100}
  //   tiling: {32}
  //   rounding: kDown
  //   Result: {3, 32}
  absl::StatusOr<std::vector<int64_t>> TransformShape(
      const std::vector<int64_t>& shape) const;

  // Computes the strides of the tiled MemRef given the original strides.
  //
  // The result strides follow the same order as the result dimensions:
  // outer strides, then major strides (original_stride * tile_size), then
  // minor strides (original_stride).
  std::vector<int64_t> TransformStrides(
      const std::vector<int64_t>& strides) const;
};

}  // namespace jax::mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFORMS_H_
