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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

namespace jax::mosaic::gpu {
namespace {

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Truly;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

// Returns true if the given value is a constant integer equal to zero.
bool IsConstantZero(mlir::Value v) {
  auto op = v.getDefiningOp<mlir::arith::ConstantOp>();
  if (!op) {
    return false;
  }
  auto attr = llvm::dyn_cast<mlir::IntegerAttr>(op.getValue());
  if (!attr) {
    return false;
  }
  return attr.getValue().isZero();
}

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

TEST(TiledLayoutTest, Create) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling,
                       Tiling::Create({{64, 8}, {16, 8}, {8, 8}, {1, 2}}));

  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-8},
                          /*lane_dims=*/{-4, -3},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  EXPECT_THAT(layout.warp_dims(), ElementsAre(-8));
  EXPECT_THAT(layout.lane_dims(), ElementsAre(-4, -3));
  EXPECT_EQ(layout.vector_dim(), -1);
}

TEST(TiledLayoutTest, CreateFailsWithDuplicateDims) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{10, 2}}));

  EXPECT_THAT(TiledLayout::Create(std::move(tiling),
                                  /*warp_dims=*/{-1},
                                  /*lane_dims=*/{-1},
                                  /*vector_dim=*/-2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TiledLayoutTest, CreateFailsWithEmptyTiling) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({}));
  EXPECT_THAT(TiledLayout::Create(std::move(tiling),
                                  /*warp_dims=*/{},
                                  /*lane_dims=*/{},
                                  /*vector_dim=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TiledLayoutTest, CreateFailsWithPositiveDim) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{32, 4}}));
  EXPECT_THAT(TiledLayout::Create(std::move(tiling),
                                  /*warp_dims=*/{1},
                                  /*lane_dims=*/{-2},
                                  /*vector_dim=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TiledLayoutTest, CreateFailsWithOutOfRangeDim) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{32, 4}}));
  EXPECT_THAT(TiledLayout::Create(std::move(tiling),
                                  /*warp_dims=*/{-3},
                                  /*lane_dims=*/{-2},
                                  /*vector_dim=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TiledLayoutTest, CreateFailsWithInvalidWarpDimsProduct) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{32, 4}}));
  EXPECT_THAT(TiledLayout::Create(std::move(tiling),
                                  /*warp_dims=*/{-2},
                                  /*lane_dims=*/{Replicated(32)},
                                  /*vector_dim=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TiledLayoutTest, CreateFailsWithInvalidLaneDimsProduct) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{8, 4, 32}}));
  EXPECT_THAT(TiledLayout::Create(std::move(tiling),
                                  /*warp_dims=*/{-2},
                                  /*lane_dims=*/{-3},
                                  /*vector_dim=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TiledLayoutTest, CreateFailsWithNonCanonicalLayout) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{4, 32}, {1, 1}}));
  EXPECT_THAT(TiledLayout::Create(std::move(tiling),
                                  /*warp_dims=*/{-4},
                                  /*lane_dims=*/{-3},
                                  /*vector_dim=*/-1,
                                  /*check_canonical=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TiledLayoutTest, Canonicalize) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling,
                       Tiling::Create({{4, 32, 1, 1}, {1, 1, 1, 1}}));

  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-8},
                          /*lane_dims=*/{-7},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  ASSERT_OK_AND_ASSIGN(TiledLayout canonical, layout.Canonicalize());

  EXPECT_THAT(canonical.warp_dims(), ElementsAre(-4));
  EXPECT_THAT(canonical.lane_dims(), ElementsAre(-3));
  EXPECT_EQ(canonical.vector_dim(), -1);
}

TEST(TiledLayoutTest, PartitionedDimsReturnAllPartitionedDims) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling,
                       Tiling::Create({{64, 8}, {32, 8}, {8, 8}, {1, 4}}));
  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-8, Replicated(2)},
                          /*lane_dims=*/{-4, -3, Replicated(2)},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  EXPECT_THAT(layout.PartitionedWarpDims(), ElementsAre(-8));
  EXPECT_THAT(layout.PartitionedLaneDims(), ElementsAre(-4, -3));
}

TEST(TiledLayoutTest, VectorLengthReturnsTheSizeOfTheVectorDim) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling,
                       Tiling::Create({{64, 8}, {16, 8}, {8, 8}, {1, 2}}));
  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-8},
                          /*lane_dims=*/{-4, -3},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  EXPECT_THAT(layout.VectorLength(), IsOkAndHolds(2));
}

class TiledLayoutMlirTest : public ::testing::Test {
 public:
  TiledLayoutMlirTest()
      : builder_(mlir::UnknownLoc::get(&context_), &context_),
        module_(mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::ModuleOp::create(builder_.getUnknownLoc(), "module"))) {}

 protected:
  void SetUp() override {
    context_.loadDialect<mlir::arith::ArithDialect, mlir::gpu::GPUDialect>();
    builder_.setInsertionPointToEnd(module_->getBody());
  }

 protected:
  mlir::MLIRContext context_;
  mlir::ImplicitLocOpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

TEST_F(TiledLayoutMlirTest, WarpIndices) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling,
                       Tiling::Create({{64, 8}, {16, 8}, {8, 8}, {1, 2}}));
  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-8},
                          /*lane_dims=*/{-4, -3},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  ASSERT_OK_AND_ASSIGN(std::vector<mlir::Value> indices,
                       layout.WarpIndices(builder_));

  // Make sure warp dim is non-zero.
  EXPECT_EQ(indices.size(), 8);
  EXPECT_FALSE(IsConstantZero(indices[0]));  // 8 - 8 = 0 index
  EXPECT_TRUE(IsConstantZero(indices[1]));
  EXPECT_TRUE(IsConstantZero(indices[2]));
  EXPECT_TRUE(IsConstantZero(indices[3]));
  EXPECT_TRUE(IsConstantZero(indices[4]));
  EXPECT_TRUE(IsConstantZero(indices[5]));
  EXPECT_TRUE(IsConstantZero(indices[6]));
  EXPECT_TRUE(IsConstantZero(indices[7]));
}

TEST_F(TiledLayoutMlirTest, LaneIndices) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling,
                       Tiling::Create({{64, 8}, {16, 8}, {8, 8}, {1, 2}}));
  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-8},
                          /*lane_dims=*/{-4, -3},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  ASSERT_OK_AND_ASSIGN(std::vector<mlir::Value> indices,
                       layout.LaneIndices(builder_));

  // Make sure lane dims are non-zero.
  EXPECT_EQ(indices.size(), 8);
  EXPECT_TRUE(IsConstantZero(indices[0]));
  EXPECT_TRUE(IsConstantZero(indices[1]));
  EXPECT_TRUE(IsConstantZero(indices[2]));
  EXPECT_TRUE(IsConstantZero(indices[3]));
  EXPECT_FALSE(IsConstantZero(indices[4]));  // 8 - 4 = 4 index
  EXPECT_FALSE(IsConstantZero(indices[5]));  // 8 - 3 = 5 index
  EXPECT_TRUE(IsConstantZero(indices[6]));
  EXPECT_TRUE(IsConstantZero(indices[7]));
}

TEST_F(TiledLayoutMlirTest, IndicesWithReplicated) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling,
                       Tiling::Create({{64, 8}, {16, 8}, {8, 8}, {1, 2}}));

  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{Replicated(4)},
                          /*lane_dims=*/{-4, -3},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  ASSERT_OK_AND_ASSIGN(std::vector<mlir::Value> indices,
                       layout.WarpIndices(builder_));

  EXPECT_EQ(indices.size(), 8);
  EXPECT_THAT(indices, Each(Truly(IsConstantZero)));
}

TEST(TiledLayoutTest,
     ApplyingTilingToShapeAndShapeFromRegistersShapeReturnsOriginalShape) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{4, 32, 2}}));
  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-3},
                          /*lane_dims=*/{-2},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  EXPECT_THAT(layout.RegistersShape({8, 128, 2}),
              IsOkAndHolds(ElementsAre(2, 4, 1, 1, 1, 1)));
  EXPECT_THAT(layout.ShapeFromRegistersShape({2, 4, 1, 1, 1, 1}),
              IsOkAndHolds(ElementsAre(8, 128, 2)));
}

TEST_F(TiledLayoutMlirTest, RegistersElementType) {
  ASSERT_OK_AND_ASSIGN(Tiling tiling, Tiling::Create({{4, 32, 2}}));
  ASSERT_OK_AND_ASSIGN(
      TiledLayout layout,
      TiledLayout::Create(std::move(tiling),
                          /*warp_dims=*/{-3},
                          /*lane_dims=*/{-2},
                          /*vector_dim=*/-1, /*check_canonical=*/false));

  mlir::Type f32 = builder_.getF32Type();
  ASSERT_OK_AND_ASSIGN(mlir::Type reg_type, layout.RegistersElementType(f32));

  auto vec_type = llvm::dyn_cast<mlir::VectorType>(reg_type);
  EXPECT_THAT(vec_type.getShape(), ElementsAre(2));
  EXPECT_EQ(vec_type.getElementType(), f32);
}

}  // namespace
}  // namespace jax::mosaic::gpu
