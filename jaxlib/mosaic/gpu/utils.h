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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_UTILS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_UTILS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace jax::mosaic::gpu {

constexpr int64_t WARP_SIZE = 32;
constexpr int64_t WARPGROUP_SIZE = 128;
constexpr int64_t WARPS_IN_WARPGROUP = WARPGROUP_SIZE / WARP_SIZE;

// Unfolds a single dimension of a MemRef into multiple dimensions.
//
// The product of `factors` must either be equal to the size of the dimension at
// `dim`, or a single factor may be std::nullopt (unset). In the latter case,
// the product of the known factors must divide the dimension size, and the
// unset factor is inferred as the quotient.
//
// Example:
//   ref: memref<128xf32>
//   dim: 0
//   factors: {32, 4}
//   Result: memref<32x4xf32>
//
//   factors: {32, std::nullopt}
//   Result: memref<32x4xf32> (inferred 4)
absl::StatusOr<mlir::Value> MemRefUnfold(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref, int64_t dim,
    const std::vector<std::optional<int64_t>>& factors);

// Slices a MemRef, potentially squeezing unit slice dimensions.
//
// `base_indices` specifies the starting index for each dimension. Each index
// can be a static integer or a dynamic mlir::Value.
// `slice_shape` specifies the size of the slice in each dimension.
// `is_squeezed` specifies which dimensions should be squeezed (removed) from
// the result. If is_squeezed[i] is true, slice_shape[i] must be 1.
//
// Example:
//   ref: memref<10x20xf32>
//   base_indices: {2, 5}
//   slice_shape: {1, 5}
//   is_squeezed: {true, false}
//   Result: memref<5xf32>
absl::StatusOr<mlir::Value> MemRefSlice(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref,
    const std::vector<std::variant<int64_t, mlir::Value>>& base_indices,
    const std::vector<int64_t>& slice_shape,
    const std::vector<bool>& is_squeezed);

// Transposes the dimensions of a MemRef.
//
// The `permutation` vector specifies the permutation of the dimensions: the
// i-th dimension of the result is the permutation[i]-th dimension of the input.
// The rank of the input MemRef must match the size of `permutation` and indices
// must be within [0, rank).
//
// Example:
//   ref: memref<10x20xf32>
//   permutation: {1, 0}
//   Result: memref<20x10xf32>
absl::StatusOr<mlir::Value> MemRefTranspose(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref,
    const std::vector<int64_t>& permutation);

// Creates a constant integer value.
mlir::Value c(mlir::ImplicitLocOpBuilder& b, int64_t val, mlir::Type type);

// Returns the thread index.
mlir::Value ThreadIdx(mlir::ImplicitLocOpBuilder& builder);

// Returns the dot product of two vectors of mlir::Value.
mlir::Value DynDot(mlir::ImplicitLocOpBuilder& builder,
                   const std::vector<mlir::Value>& a,
                   const std::vector<mlir::Value>& b);

// Traverses all elements of a `shape` with the minor-most dim-first order and
// applies `callback` to each of them.
void ForAllNdIndices(const std::vector<int64_t>& shape,
                     std::function<void(const std::vector<int64_t>&)> callback);

}  // namespace jax::mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_UTILS_H_
