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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"

namespace jax::mosaic::gpu {

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

}  // namespace jax::mosaic::gpu
