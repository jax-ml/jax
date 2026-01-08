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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace jax::mosaic::gpu {
namespace {

using ::testing::ElementsAre;

TEST(TilingTest, TileNestedShapeStrides) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{64, 64}}));
  std::vector<std::vector<int64_t>> shape = {{128}, {128}};
  std::vector<std::vector<int64_t>> strides = {{128}, {1}};

  ASSERT_OK_AND_ASSIGN((auto [tiled_shape, tiled_strides]),
                       tiling.TileNestedShapeStrides(shape, strides));

  std::vector<std::vector<int64_t>> expected_shape = {{2}, {2}, {64}, {64}};
  std::vector<std::vector<int64_t>> expected_strides = {
      {64 * 128}, {64}, {128}, {1}};
  EXPECT_EQ(tiled_shape, expected_shape);
  EXPECT_EQ(tiled_strides, expected_strides);
}

TEST(TilingTest, TileNestedShapeStridesAlreadySplit) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{64, 64}}));
  std::vector<std::vector<int64_t>> shape = {{2, 64}, {2, 64}};
  std::vector<std::vector<int64_t>> strides = {{64 * 128, 128}, {64, 1}};

  ASSERT_OK_AND_ASSIGN((auto [tiled_shape, tiled_strides]),
                       tiling.TileNestedShapeStrides(shape, strides));

  std::vector<std::vector<int64_t>> expected_shape = {{2}, {2}, {64}, {64}};
  std::vector<std::vector<int64_t>> expected_strides = {
      {64 * 128}, {64}, {128}, {1}};
  EXPECT_EQ(tiled_shape, expected_shape);
  EXPECT_EQ(tiled_strides, expected_strides);
}

TEST(TilingTest, TileNestedShapeStridesMultiLevel) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{64, 64}, {8}}));
  std::vector<std::vector<int64_t>> shape = {{128}, {128}};
  std::vector<std::vector<int64_t>> strides = {{128}, {1}};

  ASSERT_OK_AND_ASSIGN((auto [tiled_shape, tiled_strides]),
                       tiling.TileNestedShapeStrides(shape, strides));

  std::vector<std::vector<int64_t>> expected_shape = {{2}, {2}, {64}, {8}, {8}};
  std::vector<std::vector<int64_t>> expected_strides = {
      {8192}, {64}, {128}, {8}, {1}};
  EXPECT_EQ(tiled_shape, expected_shape);
  EXPECT_EQ(tiled_strides, expected_strides);
}

TEST(TilingTest, TileIndices) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{64, 64}}));
  std::vector<int64_t> indices = {70, 80};

  std::vector<int64_t> tiled_indices = tiling.TileIndices(indices);

  EXPECT_THAT(tiled_indices, ElementsAre(1, 1, 6, 16));
}

TEST(TilingTest, UntileIndices) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{64, 64}}));
  std::vector<int64_t> indices = {1, 1, 6, 16};

  std::vector<int64_t> untiled_indices = tiling.UntileIndices(indices);

  EXPECT_THAT(untiled_indices, ElementsAre(70, 80));
}

TEST(TilingTest, TileIndicesMultiLevel) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{64, 64}, {8}}));
  std::vector<int64_t> indices = {70, 80};

  std::vector<int64_t> tiled_indices = tiling.TileIndices(indices);

  EXPECT_THAT(tiled_indices, ElementsAre(1, 1, 6, 2, 0));
}

TEST(TilingTest, UntileIndicesMultiLevel) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{64, 64}, {8}}));
  std::vector<int64_t> indices = {1, 1, 6, 2, 0};

  auto untiled_indices = tiling.UntileIndices(indices);

  EXPECT_THAT(untiled_indices, ElementsAre(70, 80));
}

}  // namespace
}  // namespace jax::mosaic::gpu
