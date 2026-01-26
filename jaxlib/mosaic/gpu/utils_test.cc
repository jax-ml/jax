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

#include "jaxlib/mosaic/gpu/utils.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace jax::mosaic::gpu {
namespace {

using ::testing::status::StatusIs;

class UtilsTest : public ::testing::Test {
 public:
  UtilsTest()
      : builder_(mlir::UnknownLoc::get(&context_), &context_),
        module_(mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::ModuleOp::create(builder_.getUnknownLoc(), "module"))) {}

 protected:
  void SetUp() override {
    context_.loadDialect<mlir::memref::MemRefDialect>();
    builder_.setInsertionPointToEnd(module_->getBody());
  }

  mlir::MLIRContext context_;
  mlir::ImplicitLocOpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

TEST_F(UtilsTest, MemRefUnfoldUnfoldsCorrectly) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get({128}, f32);
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  ASSERT_OK_AND_ASSIGN(
      mlir::Value result,
      MemRefUnfold(builder_, input, /*dim=*/0, /*factors=*/{32, 4}));

  auto result_type = mlir::cast<mlir::MemRefType>(result.getType());
  EXPECT_EQ(result_type.getShape(), (llvm::ArrayRef<int64_t>{32, 4}));
}

TEST_F(UtilsTest, MemRefUnfoldInfersUnknownDimension) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get({128}, f32);
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  ASSERT_OK_AND_ASSIGN(
      mlir::Value result,
      MemRefUnfold(builder_, input, /*dim=*/0, /*factors=*/{std::nullopt, 4}));

  auto result_type = mlir::cast<mlir::MemRefType>(result.getType());
  EXPECT_EQ(result_type.getShape(), (llvm::ArrayRef<int64_t>{32, 4}));
}

TEST_F(UtilsTest, MemRefUnfoldFailsWithMultipleUnknownDimensions) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get({128}, f32);
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  EXPECT_THAT(MemRefUnfold(builder_, input, /*dim=*/0,
                           /*factors=*/{std::nullopt, std::nullopt}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Can only infer at most one dimension"));
}

TEST_F(UtilsTest, MemRefUnfoldFailsWithNonDivisibleUnfold) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type = mlir::MemRefType::get({128}, f32);
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  auto result = MemRefUnfold(builder_, input, /*dim=*/0, /*factors=*/{33, 4});

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               "Non-divisible unfold"));
}

TEST_F(UtilsTest, MemRefUnfoldWorksWithStridedLayout) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type =
      mlir::MemRefType::get({128, 128}, f32,
                            mlir::StridedLayoutAttr::get(
                                &context_, /*offset=*/0, /*strides=*/{128, 1}));
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  ASSERT_OK_AND_ASSIGN(
      mlir::Value result,
      MemRefUnfold(builder_, input, /*dim=*/0, /*factors=*/{2, 64}));

  auto result_type = mlir::cast<mlir::MemRefType>(result.getType());
  EXPECT_EQ(result_type.getShape(), (llvm::ArrayRef<int64_t>{2, 64, 128}));
  auto [strides, offset] = result_type.getStridesAndOffset();
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(strides, (llvm::ArrayRef<int64_t>{8192, 128, 1}));
}

TEST_F(UtilsTest, MemRefUnfoldWorksWithStridedLayoutAndUnknownDim) {
  mlir::Type f32 = builder_.getF32Type();
  mlir::MemRefType input_type =
      mlir::MemRefType::get({128, 128}, f32,
                            mlir::StridedLayoutAttr::get(
                                &context_, /*offset=*/0, /*strides=*/{128, 1}));
  mlir::Value input = mlir::memref::AllocOp::create(builder_, input_type);

  ASSERT_OK_AND_ASSIGN(
      mlir::Value result,
      MemRefUnfold(builder_, input, /*dim=*/0, /*factors=*/{2, std::nullopt}));

  auto result_type = mlir::cast<mlir::MemRefType>(result.getType());
  EXPECT_EQ(result_type.getShape(), (llvm::ArrayRef<int64_t>{2, 64, 128}));
  auto [strides, offset] = result_type.getStridesAndOffset();
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(strides, (llvm::ArrayRef<int64_t>{8192, 128, 1}));
}

}  // namespace
}  // namespace jax::mosaic::gpu
