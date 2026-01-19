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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

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

// Type wrapper for the number of times a dimension is replicated.
struct Replicated {
  int64_t times;

  std::string ToString() const;

  bool operator==(const Replicated& other) const {
    return times == other.times;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Replicated& rep) {
    return H::combine(std::move(h), rep.times);
  }
};

// A FragmentedArray layout derived from a tiling expression.

//  A logical array is transformed according to the tiling expression, and then
//  split across warps (within a warpgroup), lanes, and vectorized according to
//  the dimension indices. All dimension indices must be negative and should
//  refer to the dimensions after tiling is applied.
//
//  To better understand this layout, consider the example of WGMMA-related
//  tiling from
//  https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-d as
//  applied to a 128x128 array. The corresponding TiledLayout has a tiling of:
//
//      (64, 8)(16, 8)(8, 8)(1, 2)
//
//  and warp_dims=(-8,), lane_dims=(-4, -3), vector_dim=-1.
//
//  We begin by applying the tiling (note that it always applies to a suffix):
//
//          Tiled shape                       Remaining tiling actions
//  ===========================================================================
//  128 128                                  (64, 8)(16, 8)(8, 8)(1, 2)
//    2  16  64  8                           (16, 8)(8, 8)(1, 2)
//    2  16   4  1  16  8                    (8, 8)(1, 2)
//    2  16   4  1   2  1  8  8              (1, 2)
//    2  16   4  1   2  1  8  4  1  2
//
//  The last expression is our final shape. At this stage, we're ready to
//  partition the dimensions: warp_dims=(-8,) means that the 8-th dimension from
//  the end is partitioned over 4 warps in a warpgroup (and so it must be of
//  size 4). lane_dims=(-4, -3) indicate that those two dimensions are
//  partitioned over the lanes within a warp (their product must be equal to 32,
//  i.e. warp size). Finally, vector_dim=-1 indicates that each (logical)
//  register is a vector containing 2 elements (there are no shape restrictions
//  here).
//
//  Given the above, the shape of the (logical) register array used to represent
//  the array in each thread is: (2, 16, 1, 1, 2, 1, 1, 1, 1, 1). We have set
//  all the dimensions above to 1, since each thread is a member of a single
//  warp, a single lane, and the elements along the vectorized dimension are
//  represented by a single (logical) register.
//
class TiledLayout {
 public:
  using Dim = std::variant<int64_t, Replicated>;

  static absl::StatusOr<TiledLayout> Create(Tiling tiling,
                                            std::vector<Dim> warp_dims,
                                            std::vector<Dim> lane_dims,
                                            int64_t vector_dim,
                                            bool check_canonical = true);
  virtual ~TiledLayout() = default;

  bool operator==(const TiledLayout& other) const;
  bool operator!=(const TiledLayout& other) const { return !(*this == other); }

  const Tiling& tiling() const { return tiling_; }
  const std::vector<Dim>& warp_dims() const { return warp_dims_; }
  const std::vector<Dim>& lane_dims() const { return lane_dims_; }
  int64_t vector_dim() const { return vector_dim_; }

  // Returns the shape of the tiled tiling (without the base tile shape part).
  absl::StatusOr<std::vector<int64_t>> TiledTilingShape() const;

  // Canonicalizes the layout. E.g. If the tiling suffix is
  //   (4, 32, 1, 1, 1), vector_dim = -1, warp_dims = {-5}, lane_dims = {-4}
  // then the canonicalized layout is
  //   (4, 32, 1), vector_dim = -1, warp_dims = {-3}, lane_dims = {-2}
  absl::StatusOr<TiledLayout> Canonicalize() const;

  // Returns the partitioned warp dimensions verbatim.
  std::vector<int64_t> PartitionedWarpDims() const;

  // Returns the partitioned lane dimensions verbatim.
  std::vector<int64_t> PartitionedLaneDims() const;

  // Returns delinearized warp indices for a current thread.
  absl::StatusOr<std::vector<mlir::Value>> WarpIndices(
      mlir::ImplicitLocOpBuilder& builder) const;

  // Returns delinearized lane indices for a current thread.
  absl::StatusOr<std::vector<mlir::Value>> LaneIndices(
      mlir::ImplicitLocOpBuilder& builder) const;

  // Returns the size of the vector dimension. E.g. if the tiling suffix is
  // (..., 4), and vector_dims = {-1}, then the vector length is 4.
  absl::StatusOr<size_t> VectorLength() const;

  template <typename H>
  friend H AbslHashValue(H h, const TiledLayout& layout) {
    return H::combine(std::move(h), layout.tiling_, layout.warp_dims_,
                      layout.lane_dims_, layout.vector_dim_);
  }

  std::string ToString() const;

 private:
  TiledLayout(Tiling tiling, std::vector<Dim> warp_dims,
              std::vector<Dim> lane_dims, int64_t vector_dim)
      : tiling_(std::move(tiling)),
        warp_dims_(std::move(warp_dims)),
        lane_dims_(std::move(lane_dims)),
        vector_dim_(vector_dim) {};

  // Turns the linearized thread index `idx` into a vector of full indices for
  // the given dimensions `dims`.
  absl::StatusOr<std::vector<mlir::Value>> DelinearizeIndex(
      mlir::ImplicitLocOpBuilder& builder, mlir::Value idx,
      const std::vector<TiledLayout::Dim>& dims) const;

  Tiling tiling_;
  std::vector<Dim> warp_dims_;
  std::vector<Dim> lane_dims_;
  int64_t vector_dim_;
};

}  // namespace jax::mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_EXPERIMENTAL_MOSAIC_GPU_CC_TILED_LAYOUT_H_
