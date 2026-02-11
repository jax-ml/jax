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

#include "jaxlib/mosaic/gpu/transforms.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace jax::mosaic::gpu {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

class TransformsTest : public ::testing::Test {
 public:
  TransformsTest()
      : builder_(mlir::UnknownLoc::get(&context_), &context_),
        module_(mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::ModuleOp::create(builder_.getUnknownLoc(), "module"))) {}

 protected:
  void SetUp() override {
    context_
        .loadDialect<mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
    builder_.setInsertionPointToEnd(module_->getBody());
  }

  mlir::MLIRContext context_;
  mlir::ImplicitLocOpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

TEST_F(TransformsTest, TileTransformRoundsDownShapeCorrectly) {
  TileTransform transform;
  transform.tiling = {32};
  transform.rounding = Rounding::kDown;

  ASSERT_OK_AND_ASSIGN(auto new_shape, transform.TransformShape({100}));

  EXPECT_EQ(new_shape, (std::vector<int64_t>{3, 32}));
}

TEST_F(TransformsTest, TileTransformShape) {
  TileTransform transform;
  transform.tiling = {64, 32};
  std::vector<int64_t> shape = {5, 128, 128};

  ASSERT_OK_AND_ASSIGN(auto new_shape, transform.TransformShape(shape));

  EXPECT_EQ(new_shape, (std::vector<int64_t>{5, 2, 4, 64, 32}));
}

TEST_F(TransformsTest, TileTransformApply) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get({5, 128, 128}, f32);
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  TileTransform transform;
  transform.tiling = {64, 32};
  ASSERT_OK_AND_ASSIGN(mlir::Value result, transform.Apply(builder_, input));

  auto result_type = mlir::cast<mlir::MemRefType>(result.getType());
  EXPECT_EQ(result_type.getShape(), (llvm::ArrayRef<int64_t>{5, 2, 4, 64, 32}));
}

TEST_F(TransformsTest, TileTransformApplyWithStrides) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get(
      {10, 128}, f32, mlir::StridedLayoutAttr::get(&context_, 0, {128, 2}));
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  TileTransform transform;
  transform.tiling = {32};
  ASSERT_OK_AND_ASSIGN(mlir::Value result, transform.Apply(builder_, input));

  auto result_type = mlir::cast<mlir::MemRefType>(result.getType());
  EXPECT_EQ(result_type.getShape(), (llvm::ArrayRef<int64_t>{10, 4, 32}));
  auto [strides, offset] = result_type.getStridesAndOffset();
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(strides, (llvm::ArrayRef<int64_t>{128, 64, 2}));
}

TEST_F(TransformsTest, TileTransformIndex) {
  TileTransform transform;
  transform.tiling = {32};

  auto idx = builder_.create<mlir::arith::ConstantIndexOp>(35);
  ASSERT_OK_AND_ASSIGN(auto transformed_idx,
                       transform.TransformIndex(builder_, {idx}));

  EXPECT_EQ(transformed_idx.size(), 2);
}

TEST_F(TransformsTest, TileTransformIndexMultiDim) {
  TileTransform transform;
  transform.tiling = {2, 4};

  mlir::Value i0 = builder_.create<mlir::arith::ConstantIndexOp>(10);
  mlir::Value i1 = builder_.create<mlir::arith::ConstantIndexOp>(5);
  mlir::Value i2 = builder_.create<mlir::arith::ConstantIndexOp>(7);

  EXPECT_THAT(transform.TransformIndex(builder_, {i0, i1, i2}),
              IsOkAndHolds(SizeIs(5)));
}

TEST_F(TransformsTest, TileTransformStrides) {
  TileTransform transform;
  transform.tiling = {32};

  EXPECT_EQ(transform.TransformStrides(/*strides=*/{1}),
            (std::vector<int64_t>{32, 1}));
}

TEST_F(TransformsTest, TileTransformStridesMultiDim) {
  TileTransform transform;
  transform.tiling = {32, 4};

  EXPECT_EQ(transform.TransformStrides(/*strides=*/{2048, 16, 1}),
            (std::vector<int64_t>{2048, 512, 4, 16, 1}));
}

TEST_F(TransformsTest, TransformShapeFailsOnNonDivisible) {
  TileTransform transform;
  transform.tiling = {32};

  EXPECT_THAT(
      transform.TransformShape({100}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected GMEM slice shape 100 suffix to be a multiple of 32")));
}

TEST_F(TransformsTest,
       ApplyFailsOnNonDivisibleShapeWhenRoundingModeNotSpecified) {  // NOLINT
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get({100}, f32);
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  TileTransform transform;
  transform.tiling = {32};

  EXPECT_THAT(
      transform.Apply(builder_, input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("no rounding mode is specified"),
                     HasSubstr("smaller or a multiple of its tiling"))));
}

TEST_F(TransformsTest, ApplyFailsOnRankMismatch) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get({100}, f32);
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  TileTransform transform;
  transform.tiling = {32, 16};

  EXPECT_THAT(transform.Apply(builder_, input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Shape rank smaller than tiling rank")));
}

TEST_F(TransformsTest, TransformIndexFailsOnRankMismatch) {
  TileTransform transform;
  transform.tiling = {32, 16};
  mlir::Value idx = builder_.create<mlir::arith::ConstantIndexOp>(0);

  EXPECT_THAT(transform.TransformIndex(builder_, {idx}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Shape rank smaller than tiling rank")));
}

}  // namespace
}  // namespace jax::mosaic::gpu
