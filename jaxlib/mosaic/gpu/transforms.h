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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFORMS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFORMS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace jax::mosaic::gpu {

enum class Rounding { kUp, kDown };

struct TileTransform {
  std::vector<int64_t> tiling;
  std::optional<Rounding> rounding;

  // Tiles a suffix of dimensions of a MemRef.
  //
  // Given a ref with shape s = (b, m, n), a tiling t = (tm, tn), and a
  // rounding function f,
  //   apply(r, t) = transpose((0, 1, 3, 2, 4),
  //                            reshape(r, (b, f(m, tm), tm, f(n, tn), tn))
  //
  // where f is one of
  //   1. exact division (if rounding is unspecified);
  //   2. floordiv (if rounding is `kDown`);
  //   3. ceildiv  (if rounding is `kUp`).
  absl::StatusOr<mlir::Value> Apply(mlir::ImplicitLocOpBuilder& builder,
                                    mlir::Value ref) const;

  // Transforms indices from the original MemRef to the tiled MemRef.
  //
  // Given an index i = (i_b, i_m, i_n) and a tiling t = (tm, tn),
  //   transform_index(i, t) = (i_b, i_m / tm, i_n / tn, i_m % tm, i_n % tn)
  absl::StatusOr<std::vector<mlir::Value>> TransformIndex(
      mlir::ImplicitLocOpBuilder& builder,
      const std::vector<mlir::Value>& idx) const;

  // Computes the shape of the tiled MemRef given the original shape.
  //
  // Given a shape s = (b, m, n), a tiling t = (tm, tn), and a rounding
  // function f,
  //   transform_shape(s, t) = (b, f(m, tm), f(n, tn), tm, tn)
  //
  // where f is one of
  //   1. exact division (if rounding is unspecified);
  //   2. floordiv (if rounding is `kDown`);
  //   3. ceildiv  (if rounding is `kUp`).
  absl::StatusOr<std::vector<int64_t>> TransformShape(
      const std::vector<int64_t>& shape) const;

  // Computes the strides of the tiled MemRef given the original strides.
  //
  // Given strides s = (s_b, s_m, s_n) and a tiling t = (tm, tn),
  //   transform_strides(s, t) = (s_b, s_m * tm, s_n * tn, s_m, s_n)
  std::vector<int64_t> TransformStrides(
      const std::vector<int64_t>& strides) const;
};

}  // namespace jax::mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFORMS_H_
