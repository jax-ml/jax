/* Copyright 2025 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_EXPERIMENTAL_MOSAIC_GPU_CC_TILED_LAYOUT_H_
#define THIRD_PARTY_PY_JAX_EXPERIMENTAL_MOSAIC_GPU_CC_TILED_LAYOUT_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"

namespace jax::mosaic::gpu {

// A tiling expression describing a permutation of elements of an nd-array.
//
// To apply one level of tiling to an array, each of the trailing dimensions (up
// to the rank of the tile) is unfolded into two dimensions: first equal to the
// ratio of the dimension size and the tile size, and second equal to the tile
// size. Then, all newly unfolded minor dimensions are transposed to appear at
// the end.
//
// This expression describes multi-level tiling, by applying each element of
// `tiles` in sequence to the array.
//
// See https://openxla.org/xla/tiled_layout for a more detailed explanation.
class Tiling {
 public:
  using Tile = std::vector<int64_t>;

  static absl::StatusOr<Tiling> Create(std::vector<Tile> tiles);

  bool operator==(const Tiling& other) const;
  bool operator!=(const Tiling& other) const { return !(*this == other); }

  const std::vector<Tile>& tiles() const { return tiles_; }

  // Compute the shape of an array after tiling.
  absl::StatusOr<std::vector<int64_t>> TileShape(
      const std::vector<int64_t>& shape) const;

  // Compute the shape of an array before tiling from its tiled shape.
  absl::StatusOr<std::vector<int64_t>> UntileShape(
      const std::vector<int64_t>& shape) const;

  // Compute the strides of an array after tiling.
  std::vector<int64_t> TileStrides(const std::vector<int64_t>& strides) const;

  // Compute the indices of an array after tiling.
  std::vector<int64_t> TileIndices(const std::vector<int64_t>& indices) const;

  // Compute the indices of an array before tiling from its tiled indices.
  std::vector<int64_t> UntileIndices(const std::vector<int64_t>& indices) const;

  // A fused version of `TileShape` and `TileStrides` for nested shapes.
  //
  // By nested shape we mean that each logical dimension (i.e. each element of
  // shape/strides) is actually composed out of multiple physical dimensions.
  // For example, a row-major array of logical shape (128, 128) that is tiled
  // into (64, 64) tiles would have a nested shape ((2, 64), (2, 64)) (i.e. each
  // dim is split into two sub-dims) and nested strides of
  // ((2 * 64 * 64, 64), (64 * 64, 1)).
  absl::StatusOr<std::pair<std::vector<Tile>, std::vector<Tile>>>
  TileNestedShapeStrides(
      const std::vector<std::vector<int64_t>>& shape,
      const std::vector<std::vector<int64_t>>& strides) const;

  // Returns true if the tiled dim originated from the given input dim.
  absl::StatusOr<std::vector<bool>> TileDimension(int dim) const;

  // Returns a tiling with the given dimension removed.
  absl::StatusOr<Tiling> RemoveDimension(int dim) const;

  // We define a tiling to be canonical if, at each step (except the first one,
  // which defines the base tile shape):

  // 1. The tiling partitions at least one dimension in more than 1 tile. For
  //    example, the tiling `(8, 8)(8, 8)` is not canonical, as applying it
  //    yields a shape `(1, 1, 8, 8)`. We canonicalize it to `(8, 8)`, which
  //    allows getting rid of the unnecessary `1` dimensions.
  // 2. The leading dimensions of each tile are not `1`. If canonicalizing a
  //    tile in this way leads to an empty tile, then the tile is given shape
  //    `(1,)`---which is still a meaningful (final) tile. For example, the
  //    tiling `(8, 8)(1, 4)` is not canonical, as applying it yields a shape
  //    `(8, 2, 1, 4)`. We canonicalize it to `(8, 8)(4,)`, which allows
  //    getting rid of the unnecessary `1` dimension, and yields a shape
  //    `(8, 2, 4)`.
  Tiling Canonicalize() const;

  std::string ToString() const;

  template <typename H>
  friend H AbslHashValue(H h, const Tiling& tiling) {
    return H::combine(std::move(h), tiling.tiles_);
  }

 private:
  explicit Tiling(std::vector<Tile> tiles);

  std::vector<Tile> tiles_;
};

}  // namespace jax::mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_EXPERIMENTAL_MOSAIC_GPU_CC_TILED_LAYOUT_H_
