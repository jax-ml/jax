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

#include "jaxlib/mosaic/gpu/tiled_layout.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/tsl/platform/statusor.h"

namespace jax::mosaic::gpu {
namespace {

constexpr int64_t WARP_SIZE = 32;
constexpr int64_t WARPGROUP_SIZE = 128;
constexpr int64_t WARPS_IN_WARPGROUP = WARPGROUP_SIZE / WARP_SIZE;

int64_t DimSize(const TiledLayout::Dim& d,
                const std::vector<int64_t>& tiled_shape) {
  if (std::holds_alternative<Replicated>(d)) {
    return std::get<Replicated>(d).times;
  }
  int64_t idx = std::get<int64_t>(d);
  CHECK(idx < 0) << "Dimension index must be negative";
  idx += tiled_shape.size();
  CHECK(idx >= 0 && idx < tiled_shape.size()) << "Dimension index out of range";
  return tiled_shape[idx];
}

std::vector<int64_t> PartitionedDims(
    const std::vector<TiledLayout::Dim>& dims) {
  std::vector<int64_t> result;
  for (const auto& d : dims) {
    if (std::holds_alternative<int64_t>(d)) {
      result.push_back(std::get<int64_t>(d));
    }
  }
  return result;
}

mlir::Value ThreadIdx(mlir::ImplicitLocOpBuilder& builder) {
  mlir::Type i32 = builder.getI32Type();
  auto get_thread_id = [&](mlir::gpu::Dimension dim) {
    mlir::Value tid = mlir::gpu::ThreadIdOp::create(builder, dim);
    return mlir::arith::IndexCastOp::create(builder, i32, tid);
  };
  auto get_block_dim = [&](mlir::gpu::Dimension dim) {
    mlir::Value bdim = mlir::gpu::BlockDimOp::create(builder, dim);
    return mlir::arith::IndexCastOp::create(builder, i32, bdim);
  };

  mlir::Value idx = get_thread_id(mlir::gpu::Dimension::x);
  mlir::Value stride = get_block_dim(mlir::gpu::Dimension::x);
  for (const auto& dim : {mlir::gpu::Dimension::y, mlir::gpu::Dimension::z}) {
    idx = mlir::arith::AddIOp::create(
        builder, idx,
        mlir::arith::MulIOp::create(builder, get_thread_id(dim), stride));
    stride = mlir::arith::MulIOp::create(builder, stride, get_block_dim(dim));
  }

  return idx;
}

}  // namespace

Tiling::Tiling(std::vector<Tile> tiles) : tiles_(std::move(tiles)) {}

absl::StatusOr<Tiling> Tiling::Create(std::vector<Tile> tiles) {
  size_t last_tile_rank = std::numeric_limits<size_t>::max();
  for (const Tile& tile : tiles) {
    if (tile.size() > last_tile_rank) {
      return absl::InvalidArgumentError("Tiles must have a decreasing rank");
    }
    if (tile.empty()) {
      return absl::InvalidArgumentError("Tiles must not be empty");
    }
    if (absl::c_any_of(tile, [](int64_t d) { return d <= 0; })) {
      return absl::InvalidArgumentError(
          "Tile shape must only have positive sizes");
    }
    last_tile_rank = tile.size();
  }

  return Tiling(std::move(tiles));
}

absl::StatusOr<std::vector<int64_t>> Tiling::TileShape(
    const std::vector<int64_t>& shape) const {
  std::vector<int64_t> current_shape = shape;
  for (const Tile& tile : tiles_) {
    if (tile.size() > current_shape.size()) {
      return absl::InvalidArgumentError("Tiling does not apply to shape");
    }
    size_t untiled_rank = current_shape.size() - tile.size();
    std::vector<int64_t> next_shape;
    next_shape.reserve(untiled_rank + 2 * tile.size());
    for (size_t i = 0; i < untiled_rank; ++i) {
      next_shape.push_back(current_shape[i]);
    }
    for (size_t i = 0; i < tile.size(); ++i) {
      int64_t dim = current_shape[untiled_rank + i];
      int64_t t = tile[i];
      if (dim % t != 0) {
        return absl::InvalidArgumentError(
            "Dimension not divisible by tile size");
      }
      next_shape.push_back(dim / t);
    }
    for (int64_t t : tile) {
      next_shape.push_back(t);
    }
    current_shape = std::move(next_shape);
  }
  return current_shape;
}

absl::StatusOr<std::vector<int64_t>> Tiling::UntileShape(
    const std::vector<int64_t>& shape) const {
  std::vector<int64_t> current_shape = shape;
  for (auto it = tiles_.rbegin(); it != tiles_.rend(); ++it) {
    const Tile& tile = *it;
    if (tile.size() * 2 > current_shape.size()) {
      return absl::InvalidArgumentError("Invalid tiled shape");
    }
    size_t untiled_rank = current_shape.size() - 2 * tile.size();
    std::vector<int64_t> next_shape;
    next_shape.reserve(untiled_rank + tile.size());
    for (size_t i = 0; i < untiled_rank; ++i) {
      next_shape.push_back(current_shape[i]);
    }
    for (size_t i = 0; i < tile.size(); ++i) {
      int64_t outer = current_shape[untiled_rank + i];
      int64_t inner = current_shape[untiled_rank + tile.size() + i];
      if (inner != tile[i]) {
        return absl::InvalidArgumentError("Tiling dimension mismatch");
      }
      next_shape.push_back(outer * inner);
    }
    current_shape = std::move(next_shape);
  }
  return current_shape;
}

std::vector<int64_t> Tiling::TileStrides(
    const std::vector<int64_t>& strides) const {
  std::vector<int64_t> current_strides = strides;
  for (const Tile& tile : tiles_) {
    size_t untiled_rank = current_strides.size() - tile.size();
    std::vector<int64_t> next_strides;
    next_strides.reserve(untiled_rank + 2 * tile.size());
    for (size_t i = 0; i < untiled_rank; ++i) {
      next_strides.push_back(current_strides[i]);
    }
    for (size_t i = 0; i < tile.size(); ++i) {
      next_strides.push_back(current_strides[untiled_rank + i] * tile[i]);
    }
    for (size_t i = 0; i < tile.size(); ++i) {
      next_strides.push_back(current_strides[untiled_rank + i]);
    }
    current_strides = std::move(next_strides);
  }
  return current_strides;
}

std::vector<int64_t> Tiling::TileIndices(
    const std::vector<int64_t>& indices) const {
  std::vector<int64_t> current_indices = indices;
  for (const Tile& tile : tiles_) {
    size_t untiled_rank = current_indices.size() - tile.size();
    std::vector<int64_t> next_indices;
    next_indices.reserve(untiled_rank + 2 * tile.size());
    for (size_t i = 0; i < untiled_rank; ++i) {
      next_indices.push_back(current_indices[i]);
    }
    for (size_t i = 0; i < tile.size(); ++i) {
      next_indices.push_back(current_indices[untiled_rank + i] / tile[i]);
    }
    for (size_t i = 0; i < tile.size(); ++i) {
      next_indices.push_back(current_indices[untiled_rank + i] % tile[i]);
    }
    current_indices = std::move(next_indices);
  }
  return current_indices;
}

std::vector<int64_t> Tiling::UntileIndices(
    const std::vector<int64_t>& indices) const {
  std::vector<int64_t> current_indices = indices;
  for (auto it = tiles_.rbegin(); it != tiles_.rend(); ++it) {
    const Tile& tile = *it;
    size_t untiled_rank = current_indices.size() - 2 * tile.size();
    std::vector<int64_t> next_indices;
    next_indices.reserve(untiled_rank + tile.size());
    for (size_t i = 0; i < untiled_rank; ++i) {
      next_indices.push_back(current_indices[i]);
    }
    for (size_t i = 0; i < tile.size(); ++i) {
      int64_t outer = current_indices[untiled_rank + i];
      int64_t inner = current_indices[untiled_rank + tile.size() + i];
      next_indices.push_back(outer * tile[i] + inner);
    }
    current_indices = std::move(next_indices);
  }
  return current_indices;
}

absl::StatusOr<std::pair<std::vector<std::vector<int64_t>>,
                         std::vector<std::vector<int64_t>>>>
Tiling::TileNestedShapeStrides(
    const std::vector<std::vector<int64_t>>& shape,
    const std::vector<std::vector<int64_t>>& strides) const {
  if (shape.size() != strides.size()) {
    return absl::InvalidArgumentError(
        "Shape and strides must have the same length");
  }
  std::vector<std::vector<int64_t>> current_shape = shape;
  std::vector<std::vector<int64_t>> current_strides = strides;

  for (const Tile& tile : tiles_) {
    if (tile.size() > current_shape.size()) {
      return absl::InvalidArgumentError("Tiling does not apply to shape");
    }
    size_t untiled_rank = current_shape.size() - tile.size();
    std::vector<std::vector<int64_t>> next_shape;
    std::vector<std::vector<int64_t>> next_strides;
    next_shape.reserve(untiled_rank + 2 * tile.size());
    next_strides.reserve(untiled_rank + 2 * tile.size());

    for (size_t i = 0; i < untiled_rank; ++i) {
      next_shape.push_back(current_shape[i]);
      next_strides.push_back(current_strides[i]);
    }

    std::vector<std::vector<int64_t>> major_dim_shapes;
    std::vector<std::vector<int64_t>> minor_dim_shapes;
    std::vector<std::vector<int64_t>> major_dim_strides;
    std::vector<std::vector<int64_t>> minor_dim_strides;

    for (size_t i = 0; i < tile.size(); ++i) {
      int64_t t = tile[i];
      const std::vector<int64_t>& dim_shape = current_shape[untiled_rank + i];
      const std::vector<int64_t>& dim_strides =
          current_strides[untiled_rank + i];

      std::vector<int64_t> major_dim_shape_rev, major_dim_stride_rev;
      std::vector<int64_t> minor_dim_shape_rev, minor_dim_stride_rev;

      for (size_t j = 0; j < dim_shape.size(); ++j) {
        size_t idx = dim_shape.size() - 1 - j;
        int64_t d = dim_shape[idx];
        int64_t s = dim_strides[idx];

        if (d < t) {
          if (t % d != 0) {
            return absl::InvalidArgumentError(
                "Dimension not divisible by tile size");
          }
          t /= d;
          minor_dim_shape_rev.push_back(d);
          minor_dim_stride_rev.push_back(s);
        } else if (t != 1) {
          if (d % t != 0) {
            return absl::InvalidArgumentError(
                "Dimension not divisible by tile size");
          }
          minor_dim_shape_rev.push_back(t);
          minor_dim_stride_rev.push_back(s);
          if (d != t) {
            major_dim_shape_rev.push_back(d / t);
            major_dim_stride_rev.push_back(s * t);
          }
          t = 1;
        } else {
          major_dim_shape_rev.push_back(d);
          major_dim_stride_rev.push_back(s);
        }
      }
      if (t != 1) {
        return absl::InvalidArgumentError("Tile size too large for dimension");
      }

      major_dim_shapes.push_back(std::vector<int64_t>(
          major_dim_shape_rev.rbegin(), major_dim_shape_rev.rend()));
      major_dim_strides.push_back(std::vector<int64_t>(
          major_dim_stride_rev.rbegin(), major_dim_stride_rev.rend()));
      minor_dim_shapes.push_back(std::vector<int64_t>(
          minor_dim_shape_rev.rbegin(), minor_dim_shape_rev.rend()));
      minor_dim_strides.push_back(std::vector<int64_t>(
          minor_dim_stride_rev.rbegin(), minor_dim_stride_rev.rend()));
    }
    next_shape.insert(next_shape.end(), major_dim_shapes.begin(),
                      major_dim_shapes.end());
    next_shape.insert(next_shape.end(), minor_dim_shapes.begin(),
                      minor_dim_shapes.end());
    next_strides.insert(next_strides.end(), major_dim_strides.begin(),
                        major_dim_strides.end());
    next_strides.insert(next_strides.end(), minor_dim_strides.begin(),
                        minor_dim_strides.end());
    current_shape = std::move(next_shape);
    current_strides = std::move(next_strides);
  }

  auto normalize = [](std::vector<std::vector<int64_t>>& v) {
    for (std::vector<int64_t>& d : v) {
      if (d.empty()) {
        d.push_back(1);
      }
    }
  };
  normalize(current_shape);
  normalize(current_strides);

  return std::make_pair(std::move(current_shape), std::move(current_strides));
}

absl::StatusOr<std::vector<bool>> Tiling::TileDimension(int dim) const {
  size_t tiling_rank = tiles_[0].size();
  if (dim < 0 || dim >= tiling_rank) {
    return absl::InvalidArgumentError("Invalid dimension");
  }
  std::vector<int64_t> strides(tiling_rank, 1);
  strides[dim] = 0;
  std::vector<int64_t> tiled_strides = TileStrides(strides);
  std::vector<bool> result;
  result.reserve(tiled_strides.size());
  for (int64_t s : tiled_strides) {
    result.push_back(s == 0);
  }
  return result;
}

absl::StatusOr<Tiling> Tiling::RemoveDimension(int dim) const {
  size_t tiling_rank = tiles_[0].size();
  if (dim < 0 || dim >= tiling_rank) {
    return absl::InvalidArgumentError("Invalid dimension");
  }
  int dim_in_tile = dim;
  std::vector<Tile> new_tiles;
  size_t last_tile_rank = tiling_rank;
  for (Tile t : tiles_) {
    if (last_tile_rank < t.size()) {
      return absl::InvalidArgumentError("Rank invariant violated");
    }
    dim_in_tile -= (last_tile_rank - t.size());
    last_tile_rank = t.size();
    if (dim_in_tile >= 0) {
      t.erase(t.begin() + dim_in_tile);
    }
    if (t.empty()) break;
    new_tiles.push_back(std::move(t));
  }
  return Tiling(std::move(new_tiles));
}

Tiling Tiling::Canonicalize() const {
  if (tiles_.size() <= 1) return *this;
  std::vector<Tile> new_tiles;
  new_tiles.push_back(tiles_[0]);
  Tile shape = tiles_[0];
  for (size_t i = 1; i < tiles_.size(); ++i) {
    const Tile& tile = tiles_[i];
    Tile canonical_tile;
    bool found_non_one = false;
    for (size_t j = 0; j < tile.size(); ++j) {
      if (tile[j] != 1) {
        canonical_tile.assign(tile.begin() + j, tile.end());
        found_non_one = true;
        break;
      }
    }
    if (!found_non_one) {
      canonical_tile = {1};
    }

    bool redundant = true;
    if (shape.size() < canonical_tile.size()) {
      redundant = false;
    } else {
      for (size_t k = 0; k < canonical_tile.size(); ++k) {
        if (shape[shape.size() - canonical_tile.size() + k] !=
            canonical_tile[k]) {
          redundant = false;
          break;
        }
      }
    }

    if (redundant) continue;
    shape = canonical_tile;
    new_tiles.push_back(std::move(canonical_tile));
  }
  return Tiling(std::move(new_tiles));
}

std::string Tiling::ToString() const {
  std::stringstream ss;
  ss << "Tiling(";
  for (const Tile& tile : tiles_) {
    ss << "(";
    for (size_t i = 0; i < tile.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << tile[i];
    }
    if (tile.size() == 1) ss << ",";
    ss << ")";
  }
  ss << ")";
  return ss.str();
}

bool Tiling::operator==(const Tiling& other) const {
  return tiles_ == other.tiles_;
}

std::ostream& operator<<(std::ostream& os, const Tiling& tiling) {
  return os << tiling.ToString();
}

std::string Replicated::ToString() const {
  std::stringstream ss;
  ss << "Replicated(" << times << ")";
  return ss.str();
}

absl::StatusOr<TiledLayout> TiledLayout::Create(Tiling tiling,
                                                std::vector<Dim> warp_dims,
                                                std::vector<Dim> lane_dims,
                                                int64_t vector_dim,
                                                bool check_canonical) {
  if (tiling.tiles().empty()) {
    return absl::InvalidArgumentError("Tiling must have at least one tile");
  }
  const Tiling::Tile& min_shape = tiling.tiles()[0];
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> min_tiled_shape,
                      tiling.TileShape(min_shape));

  std::vector<int64_t> partitioned_warp_dims = PartitionedDims(warp_dims);
  std::vector<int64_t> partitioned_lane_dims = PartitionedDims(lane_dims);

  // Keeping the dimensions in a std::vector as the size is small and the extra
  // overhead of a set would be larger.
  std::vector<int64_t> dims_set;
  dims_set.insert(dims_set.end(), partitioned_warp_dims.begin(),
                  partitioned_warp_dims.end());
  dims_set.insert(dims_set.end(), partitioned_lane_dims.begin(),
                  partitioned_lane_dims.end());
  dims_set.push_back(vector_dim);

  for (int64_t d : dims_set) {
    if (d >= 0) {
      return absl::InvalidArgumentError("All dimensions must be negative");
    }

    if (d < -static_cast<int64_t>(min_tiled_shape.size() - min_shape.size())) {
      return absl::InvalidArgumentError("Dimension out of range");
    }
  }

  std::sort(dims_set.begin(), dims_set.end());
  for (size_t i = 1; i < dims_set.size(); ++i) {
    if (dims_set[i] == dims_set[i - 1]) {
      return absl::InvalidArgumentError("Duplicate partitioning dimensions");
    }
  }

  int64_t warp_dims_prod = 1;
  for (const Dim& d : warp_dims) {
    warp_dims_prod *= DimSize(d, min_tiled_shape);
  }
  if (warp_dims_prod != WARPS_IN_WARPGROUP) {
    return absl::InvalidArgumentError(
        "The product of warp dims does not equal the number of warps in a "
        "warpgroup");
  }

  int64_t lane_dims_prod = 1;
  for (const auto& d : lane_dims) {
    lane_dims_prod *= DimSize(d, min_tiled_shape);
  }
  if (lane_dims_prod != WARP_SIZE) {
    return absl::InvalidArgumentError(
        "The product of lane dims does not equal the warp size");
  }

  TiledLayout layout(std::move(tiling), std::move(warp_dims),
                     std::move(lane_dims), vector_dim);
  if (check_canonical) {
    TF_ASSIGN_OR_RETURN(TiledLayout canonical, layout.Canonicalize());
    if (canonical != layout) {
      return absl::InvalidArgumentError("TiledLayout is not canonical");
    }
  }
  return layout;
}

absl::StatusOr<std::vector<int64_t>> TiledLayout::TiledTilingShape() const {
  const Tiling::Tile& min_shape = tiling_.tiles()[0];
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> min_tiled_shape,
                      tiling_.TileShape(min_shape));
  return std::vector<int64_t>(min_tiled_shape.begin() + min_shape.size(),
                              min_tiled_shape.end());
}

absl::StatusOr<TiledLayout> TiledLayout::Canonicalize() const {
  Tiling canonical_tiling = tiling_.Canonicalize();
  const std::vector<int64_t>& s = tiling_.tiles()[0];
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tiled_tiling_shape,
                      TiledTilingShape());

  TF_ASSIGN_OR_RETURN(std::vector<int64_t> canonical_tiled_tiling_shape,
                      canonical_tiling.TileShape(s));
  canonical_tiled_tiling_shape.erase(
      canonical_tiled_tiling_shape.begin(),
      canonical_tiled_tiling_shape.begin() + s.size());

  int64_t offset =
      static_cast<int64_t>(canonical_tiled_tiling_shape.size()) - 1;
  std::vector<bool> rev_removed_dims;
  // Iterate starting from the end in order to eliminate leading dimensions,
  // whenever possible. For instance, say we have
  //
  //   shape=(4, 32, 1, 1, 1, 1, 1)
  //   warp_dims=(-7,),
  //   lane_dims=(-6,)
  //   vector_dim=-1
  //
  // and we want to canonicalize this to
  //
  //   shape=(4, 32, 1)
  //   warp_dims=(-3,),
  //   lane_dims=(-2,)
  //   vector_dim=-1.
  //
  // After the loop below, we end up with
  //
  //   rev_removed_dims=[False, True, True, True, True, False, False]
  //
  // which will yield offsets `4` for `warp_dims[0]`, `4` for `lane_dims[0]`,
  // and `0` for `vector_dim`.
  for (auto it = tiled_tiling_shape.rbegin(); it != tiled_tiling_shape.rend();
       ++it) {
    if (offset >= 0 && *it == canonical_tiled_tiling_shape[offset]) {
      rev_removed_dims.push_back(false);
      offset--;
    } else {
      rev_removed_dims.push_back(true);
    }
  }
  CHECK_EQ(offset, -1);

  std::vector<int64_t> dim_offsets(rev_removed_dims.size());
  int64_t current_sum = 0;
  for (size_t i = 0; i < rev_removed_dims.size(); ++i) {
    if (rev_removed_dims[i]) {
      current_sum++;
    }
    dim_offsets[i] = current_sum;
  }
  std::reverse(dim_offsets.begin(), dim_offsets.end());

  auto replace_tiled_dim = [&](Dim d) -> Dim {
    if (std::holds_alternative<Replicated>(d)) {
      return d;
    }
    int64_t idx = std::get<int64_t>(d);
    CHECK(idx < 0) << "Expected negative index";
    return idx + dim_offsets[idx + tiled_tiling_shape.size()];
  };

  auto is_nontrivial = [&](Dim d) -> bool {
    if (std::holds_alternative<Replicated>(d)) {
      return true;
    }
    int64_t idx = std::get<int64_t>(d);
    CHECK(idx < 0) << "Expected negative index";
    return tiled_tiling_shape[idx + tiled_tiling_shape.size()] != 1;
  };

  std::vector<Dim> new_warp_dims;
  for (const auto& d : warp_dims_) {
    if (is_nontrivial(d)) {
      new_warp_dims.push_back(replace_tiled_dim(d));
    }
  }
  std::vector<Dim> new_lane_dims;
  for (const auto& d : lane_dims_) {
    if (is_nontrivial(d)) {
      new_lane_dims.push_back(replace_tiled_dim(d));
    }
  }
  Dim new_vector_dim_val = replace_tiled_dim(vector_dim_);
  int64_t new_vector_dim = std::get<int64_t>(new_vector_dim_val);

  return TiledLayout(canonical_tiling, new_warp_dims, new_lane_dims,
                     new_vector_dim);
}

std::vector<int64_t> TiledLayout::PartitionedWarpDims() const {
  return PartitionedDims(warp_dims_);
}

std::vector<int64_t> TiledLayout::PartitionedLaneDims() const {
  return PartitionedDims(lane_dims_);
}

absl::StatusOr<std::vector<mlir::Value>> TiledLayout::WarpIndices(
    mlir::ImplicitLocOpBuilder& builder) const {
  mlir::Type i32 = builder.getI32Type();
  mlir::Value c32 = mlir::arith::ConstantOp::create(
      builder, i32, builder.getIntegerAttr(i32, WARP_SIZE));
  mlir::Value c4 = mlir::arith::ConstantOp::create(
      builder, i32, builder.getIntegerAttr(i32, WARPS_IN_WARPGROUP));
  mlir::Value thread_id = ThreadIdx(builder);
  mlir::Value warp_id = mlir::arith::DivUIOp::create(builder, thread_id, c32);
  warp_id = mlir::arith::RemUIOp::create(builder, warp_id, c4);
  return DelinearizeIndex(builder, warp_id, warp_dims_);
}

absl::StatusOr<std::vector<mlir::Value>> TiledLayout::LaneIndices(
    mlir::ImplicitLocOpBuilder& builder) const {
  mlir::Type i32 = builder.getI32Type();
  mlir::Value c32 = mlir::arith::ConstantOp::create(
      builder, i32, builder.getIntegerAttr(i32, WARP_SIZE));
  mlir::Value thread_id = ThreadIdx(builder);
  mlir::Value lane_id = mlir::arith::RemUIOp::create(builder, thread_id, c32);
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tiled_tiling_shape,
                      TiledTilingShape());
  return DelinearizeIndex(builder, lane_id, lane_dims_);
}

absl::StatusOr<size_t> TiledLayout::VectorLength() const {
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tiled_tiling_shape,
                      TiledTilingShape());
  return DimSize(vector_dim_, tiled_tiling_shape);
}

bool TiledLayout::operator==(const TiledLayout& other) const {
  return tiling_ == other.tiling_ && warp_dims_ == other.warp_dims_ &&
         lane_dims_ == other.lane_dims_ && vector_dim_ == other.vector_dim_;
}

std::string TiledLayout::ToString() const {
  std::stringstream ss;
  ss << "TiledLayout(tiling=" << tiling_.ToString() << ", warp_dims=(";
  for (size_t i = 0; i < warp_dims_.size(); ++i) {
    if (i > 0) ss << ", ";
    if (std::holds_alternative<Replicated>(warp_dims_[i])) {
      ss << "Replicated(" << std::get<Replicated>(warp_dims_[i]).times << ")";
    } else {
      ss << std::get<int64_t>(warp_dims_[i]);
    }
  }
  ss << "), lane_dims=(";
  for (size_t i = 0; i < lane_dims_.size(); ++i) {
    if (i > 0) ss << ", ";
    if (std::holds_alternative<Replicated>(lane_dims_[i])) {
      ss << "Replicated(" << std::get<Replicated>(lane_dims_[i]).times << ")";
    } else {
      ss << std::get<int64_t>(lane_dims_[i]);
    }
  }
  ss << "), vector_dim=" << vector_dim_ << ")";
  return ss.str();
}

absl::StatusOr<std::vector<mlir::Value>> TiledLayout::DelinearizeIndex(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value idx,
    const std::vector<TiledLayout::Dim>& dims) const {
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tiled_shape, TiledTilingShape());

  std::vector<int64_t> dims_sizes;
  for (const auto& d : dims) {
    if (std::holds_alternative<Replicated>(d)) {
      dims_sizes.push_back(std::get<Replicated>(d).times);
    } else {
      int64_t idx = std::get<int64_t>(d);
      CHECK(idx < 0) << "Dimension index must be negative";
      dims_sizes.push_back(tiled_shape[idx + tiled_shape.size()]);
    }
  }

  std::vector<int64_t> dims_strides(dims_sizes.size());
  int64_t stride = 1;
  for (int i = dims_sizes.size() - 1; i >= 0; --i) {
    dims_strides[i] = stride;
    stride *= dims_sizes[i];
  }

  mlir::Type i32 = builder.getI32Type();
  std::vector<mlir::Value> dims_indices;
  for (const auto& [dim_stride, dim_size] :
       llvm::zip(dims_strides, dims_sizes)) {
    mlir::Value stride = mlir::arith::ConstantOp::create(
        builder, i32, builder.getIntegerAttr(i32, dim_stride));
    mlir::Value size = mlir::arith::ConstantOp::create(
        builder, i32, builder.getIntegerAttr(i32, dim_size));
    mlir::Value div = mlir::arith::DivUIOp::create(builder, idx, stride);
    dims_indices.push_back(mlir::arith::RemUIOp::create(builder, div, size));
  }

  std::vector<mlir::Value> full_indices(
      tiled_shape.size(), mlir::arith::ConstantOp::create(
                              builder, i32, builder.getIntegerAttr(i32, 0)));
  for (const auto& [dim, dim_idx] : llvm::zip(dims, dims_indices)) {
    if (std::holds_alternative<Replicated>(dim)) {
      continue;
    }
    int64_t d = std::get<int64_t>(dim);
    CHECK(d < 0) << "Dimension index must be negative";
    full_indices[d + tiled_shape.size()] = dim_idx;
  }
  return full_indices;
}

absl::StatusOr<std::vector<int64_t>> TiledLayout::RegistersShape(
    const std::vector<int64_t>& shape) const {
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tiled_shape,
                      tiling_.TileShape(shape));
  for (int64_t d : PartitionedWarpDims()) {
    CHECK(d < 0) << "Expected negative dimension index";
    tiled_shape[d + tiled_shape.size()] = 1;
  }
  for (int64_t d : PartitionedLaneDims()) {
    CHECK(d < 0) << "Expected negative dimension index";
    tiled_shape[d + tiled_shape.size()] = 1;
  }
  CHECK(vector_dim_ < 0) << "Expected negative dimension index";
  tiled_shape[vector_dim_ + tiled_shape.size()] = 1;
  return tiled_shape;
}

absl::StatusOr<std::vector<int64_t>> TiledLayout::ShapeFromRegistersShape(
    const std::vector<int64_t>& shape) const {
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tiled_tiling, TiledTilingShape());
  std::vector<int64_t> tiled_shape = shape;
  for (int64_t d : PartitionedWarpDims()) {
    CHECK(d < 0) << "Expected negative dimension index";
    tiled_shape[d + tiled_shape.size()] = tiled_tiling[d + tiled_tiling.size()];
  }
  for (int64_t d : PartitionedLaneDims()) {
    CHECK(d < 0) << "Expected negative dimension index";
    tiled_shape[d + tiled_shape.size()] = tiled_tiling[d + tiled_tiling.size()];
  }
  CHECK(vector_dim_ < 0) << "Expected negative dimension index";
  tiled_shape[vector_dim_ + tiled_shape.size()] =
      tiled_tiling[vector_dim_ + tiled_tiling.size()];
  return tiling_.UntileShape(tiled_shape);
}

absl::StatusOr<mlir::Type> TiledLayout::RegistersElementType(
    mlir::Type t) const {
  TF_ASSIGN_OR_RETURN(size_t vector_length, VectorLength());
  return mlir::VectorType::get({static_cast<int64_t>(vector_length)}, t);
}

std::vector<int64_t> TiledLayout::BaseTileShape() const {
  return tiling_.tiles()[0];
}

absl::StatusOr<TiledLayout> TiledLayout::RemoveDimension(int64_t dim) const {
  if (dim < 0 || dim >= BaseTileShape().size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Dimension ", dim, " is out of range for ", tiling_.ToString()));
  }

  TF_ASSIGN_OR_RETURN(Tiling new_tiling, tiling_.RemoveDimension(dim));
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> tiled_shape, TiledTilingShape());
  TF_ASSIGN_OR_RETURN(std::vector<bool> removed_dim,
                      tiling_.TileDimension(dim));

  std::vector<int64_t> dim_offsets(removed_dim.size());
  int64_t current_sum = 0;
  for (int i = removed_dim.size() - 1; i >= 0; --i) {
    if (removed_dim[i]) {
      current_sum++;
    }
    dim_offsets[i] = current_sum;
  }

  CHECK(vector_dim_ < 0) << "Expected negative dimension index";
  int64_t offset = removed_dim.size();
  int64_t vector_dim_pos = vector_dim_ + offset;

  int64_t new_vector_dim;
  if (removed_dim[vector_dim_pos]) {
    std::vector<Tiling::Tile> tiles = new_tiling.tiles();
    tiles.push_back({1});
    TF_ASSIGN_OR_RETURN(new_tiling, Tiling::Create(tiles));
    new_vector_dim = -1;
    for (size_t i = 0; i < dim_offsets.size(); ++i) {
      dim_offsets[i]--;  // We inserted an extra dim.
    }
  } else {
    new_vector_dim = vector_dim_ + dim_offsets[vector_dim_pos];
  }

  auto replace_tiled_dim = [&](Dim d, int64_t size) -> Dim {
    if (std::holds_alternative<Replicated>(d)) {
      return d;
    }
    int64_t idx = std::get<int64_t>(d);
    CHECK(idx < 0) << "Expected negative dimension index";
    int64_t pos = idx + offset;
    if (removed_dim[pos]) {
      return Replicated{size};
    } else {
      return idx + dim_offsets[pos];
    }
  };

  std::vector<Dim> new_warp_dims;
  for (const auto& d : warp_dims_) {
    int64_t size = 0;
    if (std::holds_alternative<int64_t>(d)) {
      int64_t pos = std::get<int64_t>(d);
      CHECK(pos < 0) << "Expected negative dimension index";
      size = tiled_shape[pos + tiled_shape.size()];
    }
    new_warp_dims.push_back(replace_tiled_dim(d, size));
  }
  std::vector<Dim> new_lane_dims;
  for (const auto& d : lane_dims_) {
    int64_t size = 0;
    if (std::holds_alternative<int64_t>(d)) {
      int64_t pos = std::get<int64_t>(d);
      CHECK(pos < 0) << "Expected negative dimension index";
      size = tiled_shape[pos + tiled_shape.size()];
    }
    new_lane_dims.push_back(replace_tiled_dim(d, size));
  }

  TF_ASSIGN_OR_RETURN(TiledLayout new_layout,
                      TiledLayout::Create(new_tiling, new_warp_dims,
                                          new_lane_dims, new_vector_dim,
                                          /*check_canonical=*/false));
  return new_layout.Canonicalize();
}

absl::StatusOr<TiledLayout> TiledLayout::Reduce(
    const std::vector<int64_t>& axes) const {
  TiledLayout reduced_layout = *this;
  std::vector<int64_t> sorted_axes = axes;
  std::sort(sorted_axes.rbegin(), sorted_axes.rend());
  for (int a : sorted_axes) {
    TF_ASSIGN_OR_RETURN(reduced_layout, reduced_layout.RemoveDimension(a));
  }
  return reduced_layout;
}

}  // namespace jax::mosaic::gpu
