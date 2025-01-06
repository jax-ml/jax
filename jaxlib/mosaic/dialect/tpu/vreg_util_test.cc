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

#include "jaxlib/mosaic/dialect/tpu/vreg_util.h"

#include <array>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OwningOpRef.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/DebugStringHelper.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

namespace {

using ::testing::Eq;
using ::testing::Optional;

MATCHER_P2(IsConstantOpWithSplatValue, type, splat_value, "") {
  auto constant_op = dyn_cast<arith::ConstantOp>(arg.getDefiningOp());
  if (constant_op == nullptr) {
    *result_listener << "Expected a constant op, got " << debugString(arg);
    return false;
  }
  auto dense_attr = dyn_cast<DenseElementsAttr>(constant_op.getValue());
  if (dense_attr == nullptr) {
    *result_listener << "Expected a dense elements attr, got "
                     << debugString(arg);
    return false;
  }
  if (dense_attr.getType() != type) {
    *result_listener << "Expected a dense elements attr with type "
                     << debugString(type) << ", got "
                     << debugString(dense_attr.getType());
    return false;
  }
  if (!dense_attr.isSplat()) {
    *result_listener << "Expected a splat dense elements attr, got "
                     << debugString(dense_attr);
    return false;
  }
  if (auto s = dense_attr.template getSplatValue<decltype(splat_value)>();
      s != splat_value) {
    *result_listener << "Expected a splat dense elements attr with value "
                     << splat_value << ", got " << s;
    return false;
  }
  return true;
}

MATCHER_P2(IsVectorTypeWithShape, shape, elem_ty, "") {
  auto vty = dyn_cast<VectorType>(arg);
  if (vty == nullptr) {
    *result_listener << "Expected a vector type, got " << debugString(arg);
    return false;
  }
  if (vty.getShape() != ArrayRef<int64_t>(shape)) {
    *result_listener << "Expected a vector type with shape "
                     << absl::StrJoin(shape, ",") << ", got "
                     << absl::StrJoin(vty.getShape(), ",");
    return false;
  }
  if (vty.getElementType() != elem_ty) {
    *result_listener << "Expected a vector type with element type "
                     << debugString(elem_ty) << ", got "
                     << debugString(vty.getElementType());
    return false;
  }
  return true;
}

class VregUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<arith::ArithDialect, vector::VectorDialect,
                         tpu::TPUDialect>();
    mlir::Location loc = mlir::UnknownLoc::get(&context_);
    mlir::OpBuilder b(&context_);
    module_ = b.create<ModuleOp>(loc);
    builder_ = std::make_unique<mlir::ImplicitLocOpBuilder>(
        module_->getLoc(), module_->getBodyRegion());
  }

  void TearDown() override {
    builder_.reset();
    // Reset the module to prevent memory leaks.
    module_ = nullptr;
  }

  mlir::ImplicitLocOpBuilder& Builder() { return *builder_; }

 private:
  MLIRContext context_;
  std::unique_ptr<mlir::ImplicitLocOpBuilder> builder_;
  OwningOpRef<ModuleOp> module_;
};

TEST_F(VregUtilTest, GetNativeVregOrVmaskTypeBitwidthMismatch) {
  EXPECT_DEATH(getNativeVregOrVmaskType(Builder().getI16Type(),
                                        /*layout_bitwidth=*/8, {2, 4}),
               "");
}

TEST_F(VregUtilTest, GetNativeVregOrVmaskTypeI1) {
  EXPECT_THAT(getNativeVregOrVmaskType(Builder().getI1Type(),
                                       /*layout_bitwidth=*/8, {2, 4}),
              IsVectorTypeWithShape(std::array<int64_t, 3>{2, 4, 4},
                                    Builder().getI1Type()));
}

TEST_F(VregUtilTest, GetNativeVregF32) {
  EXPECT_THAT(getNativeVregType(Builder().getF32Type(), {2, 4}),
              IsVectorTypeWithShape(std::array<int64_t, 2>{2, 4},
                                    Builder().getF32Type()));
}

TEST_F(VregUtilTest, GetNativeVregBf16) {
  EXPECT_THAT(getNativeVregType(Builder().getBF16Type(), {2, 4}),
              IsVectorTypeWithShape(std::array<int64_t, 3>{2, 4, 2},
                                    Builder().getBF16Type()));
}

TEST_F(VregUtilTest, GetFullVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getI32Type());
  TypedValue<VectorType> vec =
      getFullVector(Builder(), vty, Builder().getI32IntegerAttr(0x1));

  EXPECT_THAT(vec, IsConstantOpWithSplatValue(vty, int32_t{0x1}));
}

TEST_F(VregUtilTest, GetFullLikeVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getF32Type());
  TypedValue<VectorType> in_vec = Builder().create<vector::SplatOp>(
      vty, Builder().create<arith::ConstantOp>(
               vty.getElementType(), Builder().getF32FloatAttr(1.0f)));
  TypedValue<VectorType> vec =
      getFullLikeVector(Builder(), in_vec, Builder().getF32FloatAttr(2.0f));

  EXPECT_THAT(vec, IsConstantOpWithSplatValue(vty, float{2.0f}));
}

TEST_F(VregUtilTest, GetZerosVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getI32Type());
  TypedValue<VectorType> vec = getZerosVector(Builder(), vty);

  EXPECT_THAT(vec, IsConstantOpWithSplatValue(vty, int32_t{0}));
}

TEST_F(VregUtilTest, GetZerosLikeVector) {
  VectorType vty = VectorType::get({2, 4}, Builder().getF32Type());
  TypedValue<VectorType> in_vec = Builder().create<vector::SplatOp>(
      vty, Builder().create<arith::ConstantOp>(
               vty.getElementType(), Builder().getF32FloatAttr(1.0f)));
  TypedValue<VectorType> vec = getZerosLikeVector(Builder(), in_vec);

  EXPECT_THAT(vec, IsConstantOpWithSplatValue(vty, float{0.0f}));
}

TEST_F(VregUtilTest, GetX32VmaskByPaddingEndDim0) {
  constexpr std::array<int64_t, 2> kTargetShape = {4, 8};
  FailureOr<TypedValue<VectorType>> vec = getX32VmaskByPaddingEnd(
      Builder(), /*padding=*/1, /*target_shape=*/kTargetShape,
      /*dim=*/0);
  ASSERT_TRUE(succeeded(vec));

  auto cmp_op = dyn_cast<arith::CmpIOp>(vec.value().getDefiningOp());
  ASSERT_TRUE(cmp_op != nullptr);
  EXPECT_EQ(cmp_op.getPredicate(), arith::CmpIPredicate::slt);

  auto iota_op = dyn_cast<tpu::IotaOp>(cmp_op.getLhs().getDefiningOp());
  ASSERT_TRUE(iota_op != nullptr);
  EXPECT_THAT(iota_op.getDimension(), Optional(Eq(0)));

  EXPECT_THAT(
      cmp_op.getRhs(),
      IsConstantOpWithSplatValue(
          VectorType::get(kTargetShape, Builder().getI32Type()), int32_t{3}));
}

TEST_F(VregUtilTest, GetX32VmaskByPaddingEndDim1) {
  constexpr std::array<int64_t, 2> kTargetShape = {4, 8};
  FailureOr<TypedValue<VectorType>> vec = getX32VmaskByPaddingEnd(
      Builder(), /*padding=*/3, /*target_shape=*/kTargetShape,
      /*dim=*/1);
  ASSERT_TRUE(succeeded(vec));

  auto cmp_op = dyn_cast<arith::CmpIOp>(vec.value().getDefiningOp());
  ASSERT_TRUE(cmp_op != nullptr);
  EXPECT_EQ(cmp_op.getPredicate(), arith::CmpIPredicate::slt);

  auto iota_op = dyn_cast<tpu::IotaOp>(cmp_op.getLhs().getDefiningOp());
  ASSERT_TRUE(iota_op != nullptr);
  EXPECT_THAT(iota_op.getDimension(), Optional(Eq(1)));

  EXPECT_THAT(
      cmp_op.getRhs(),
      IsConstantOpWithSplatValue(
          VectorType::get(kTargetShape, Builder().getI32Type()), int32_t{5}));
}

}  // namespace

}  // namespace mlir::tpu
