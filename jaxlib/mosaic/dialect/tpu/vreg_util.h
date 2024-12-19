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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_VREG_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_VREG_UTIL_H_

#include <array>
#include <cstdint>

#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/Types.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "xla/array.h"

namespace mlir::tpu {

// Returns the native vreg or vmask type for the given element type and target
// shape. The layout bitwidth is used for i1 (vmask) elements.
VectorType getNativeVregOrVmaskType(Type elem_ty, int8_t layout_bitwidth,
                                    std::array<int64_t, 2> target_shape);
VectorType getNativeVregType(Type elem_ty, std::array<int64_t, 2> target_shape);

// Returns a zero constant of the same type as `vreg_ty`.
FailureOr<TypedValue<VectorType>> getZerosLikeVreg(
    ImplicitLocOpBuilder &builder, Type vreg_ty);
// Same as above, but takes a `vreg` as input.
FailureOr<TypedValue<VectorType>> getZerosLikeVreg(
    ImplicitLocOpBuilder &builder, Value vreg);

// Returns a constant of the same type as `vreg_ty` with the given `value`.
// `value` must be a splat attribute.
FailureOr<TypedValue<VectorType>> getFullLikeVreg(ImplicitLocOpBuilder &builder,
                                                  Type vreg_ty,
                                                  Attribute value);
// Returns a constant of the same type as `vreg_ty` with the given `value`.
FailureOr<TypedValue<VectorType>> getFullLikeVreg(ImplicitLocOpBuilder &builder,
                                                  Value vreg, Attribute value);

// Creates a vmask with false flags to bottom (dim = 0)
// or right (dim = 1) where the flag count corresponds to the (dim_size -
// padding). If stride is provided, the padding value is sequentially
// increased by the stride value along the dim.
//
// For example, assume vmask shape is (4, 8)
//
// getVmaskByPaddingEnd(padding=3, dim=1) creates:
//  [T, T, T, T, T, F, F, F]
//  [T, T, T, T, T, F, F, F]
//  [T, T, T, T, T, F, F, F]
//  [T, T, T, T, T, F, F, F]
//
// getVmaskByPaddingEnd(padding=3, dim=1, stride=1) creates:
//  [T, T, T, T, T, F, F, F]
//  [T, T, T, T, T, T, F, F]
//  [T, T, T, T, T, T, T, F]
//  [T, T, T, T, T, T, T, T]
FailureOr<TypedValue<VectorType>> getX32VmaskByPaddingEnd(
    ImplicitLocOpBuilder &builder, Value padding,
    std::array<int64_t, 2> target_shape, int64_t dim, int64_t stride = 0);
// Same as above, but takes a constant padding value as input.
FailureOr<TypedValue<VectorType>> getX32VmaskByPaddingEnd(
    ImplicitLocOpBuilder &builder, int64_t padding,
    std::array<int64_t, 2> target_shape, int64_t dim, int64_t stride = 0);

// Masks out the padding in the bottom and right of the vregs. vregs are
// expected to have native tiling, and the masked vregs are mutated in
// `vregs`.
LogicalResult maskNativeTilingVregs(ImplicitLocOpBuilder &builder,
                                    xla::Array<Value> &vregs,
                                    std::array<int64_t, 2> target_shape,
                                    int64_t padding_bottom,
                                    int64_t padding_right);

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_VREG_UTIL_H_
