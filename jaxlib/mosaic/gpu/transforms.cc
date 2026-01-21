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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/gpu/utils.h"
#include "xla/tsl/platform/statusor.h"

namespace jax::mosaic::gpu {

absl::StatusOr<mlir::Value> TileTransform::Apply(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref) const {
  auto ref_ty = mlir::cast<mlir::MemRefType>(ref.getType());
  int64_t untiled_rank = ref_ty.getRank();
  int64_t tiling_rank = tiling.size();
  int64_t tiled_rank = untiled_rank + tiling_rank;
  if (tiling_rank > untiled_rank) {
    return absl::InvalidArgumentError("Shape rank smaller than tiling rank");
  }

  llvm::ArrayRef<int64_t> ref_shape = ref_ty.getShape();
  for (int i = 0; i < tiling_rank; ++i) {
    int64_t d = untiled_rank - 1 - i;
    int64_t t = tiling[tiling_rank - 1 - i];
    ref_ty = mlir::cast<mlir::MemRefType>(ref.getType());
    int64_t s = ref_shape[d];

    if (s > t) {
      if (s % t) {
        if (!rounding.has_value()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "When no rounding mode is specified, dimension ", d,
              " must have size smaller or a multiple of its tiling ", t,
              ", but got ", s));
        }
        if (*rounding == Rounding::kUp) {
          return absl::UnimplementedError("Rounding::kUp not implemented");
        } else if (*rounding == Rounding::kDown) {
          std::vector<std::variant<int64_t, mlir::Value>> base_indices;
          std::vector<int64_t> slice_shape;
          std::vector<bool> is_squeezed;
          base_indices.reserve(d + 1);
          slice_shape.reserve(d + 1);
          is_squeezed.reserve(d + 1);
          for (int k = 0; k < d; ++k) {
            base_indices.push_back(0);
            slice_shape.push_back(ref_ty.getShape()[k]);
            is_squeezed.push_back(false);
          }
          base_indices.push_back(0);
          slice_shape.push_back((s / t) * t);
          is_squeezed.push_back(false);
          for (int k = d + 1; k < ref_ty.getRank(); ++k) {
            base_indices.push_back(0);
            slice_shape.push_back(ref_ty.getShape()[k]);
            is_squeezed.push_back(false);
          }
          TF_ASSIGN_OR_RETURN(ref, MemRefSlice(builder, ref, base_indices,
                                               slice_shape, is_squeezed));
        } else {
          return absl::InvalidArgumentError("Unknown rounding mode");
        }
      }
    } else {
      t = s;
    }
    TF_ASSIGN_OR_RETURN(
        ref, MemRefUnfold(builder, ref, d, /*factors=*/{std::nullopt, t}));
  }

  std::vector<int64_t> permutation;
  permutation.reserve(tiled_rank);
  for (int64_t i = 0; i < untiled_rank - tiling_rank; ++i) {
    permutation.push_back(i);
  }
  for (int64_t i = untiled_rank - tiling_rank; i < tiled_rank; i += 2) {
    permutation.push_back(i);
  }
  for (int64_t i = untiled_rank - tiling_rank + 1; i < tiled_rank; i += 2) {
    permutation.push_back(i);
  }

  return MemRefTranspose(builder, ref, permutation);
}

absl::StatusOr<std::vector<mlir::Value>> TileTransform::TransformIndex(
    mlir::ImplicitLocOpBuilder& builder,
    const std::vector<mlir::Value>& idx) const {
  int64_t tiling_rank = tiling.size();
  if (idx.size() < tiling_rank) {
    return absl::InvalidArgumentError("Shape rank smaller than tiling rank");
  }
  std::vector<mlir::Value> result;
  result.reserve(idx.size() + tiling_rank);

  for (size_t i = 0; i < idx.size() - tiling_rank; ++i) {
    result.push_back(idx[i]);
  }

  for (size_t i = 0; i < tiling_rank; ++i) {
    auto i_val = idx[idx.size() - tiling_rank + i];
    auto t_val = builder.create<mlir::arith::ConstantIndexOp>(tiling[i]);
    result.push_back(builder.create<mlir::arith::DivUIOp>(i_val, t_val));
  }

  for (size_t i = 0; i < tiling_rank; ++i) {
    auto i_val = idx[idx.size() - tiling_rank + i];
    auto t_val = builder.create<mlir::arith::ConstantIndexOp>(tiling[i]);
    result.push_back(builder.create<mlir::arith::RemUIOp>(i_val, t_val));
  }

  return result;
}

absl::StatusOr<std::vector<int64_t>> TileTransform::TransformShape(
    const std::vector<int64_t>& shape) const {
  int64_t tiling_rank = tiling.size();
  if (shape.size() < tiling_rank) {
    return absl::InvalidArgumentError("Shape rank smaller than tiling rank");
  }

  if (!rounding.has_value()) {
    for (size_t i = 0; i < tiling_rank; ++i) {
      int64_t size = shape[shape.size() - tiling_rank + i];
      int64_t tile_size = tiling[i];
      if (size % tile_size != 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected GMEM slice shape ", absl::StrJoin(shape, ", "),
            " suffix to be a multiple of ", absl::StrJoin(tiling, ", "),
            ". If you're using padded async copies, your slice might need to"
            " extend out of bounds of the GMEM buffer (OOB accesses will be"
            " skipped)."));
      }
    }
  } else if (*rounding != Rounding::kDown) {
    return absl::UnimplementedError("Only Rounding::kDown is implemented");
  }

  std::vector<int64_t> result;
  result.reserve(shape.size() + tiling_rank);

  for (size_t i = 0; i < shape.size() - tiling_rank; ++i) {
    result.push_back(shape[i]);
  }

  for (size_t i = 0; i < tiling_rank; ++i) {
    int64_t s = shape[shape.size() - tiling_rank + i];
    int64_t t = tiling[i];
    result.push_back(s / t);
  }

  for (int64_t t : tiling) {
    result.push_back(t);
  }

  return result;
}

std::vector<int64_t> TileTransform::TransformStrides(
    const std::vector<int64_t>& strides) const {
  int64_t tiling_rank = tiling.size();
  std::vector<int64_t> result;
  result.reserve(strides.size() + tiling_rank);

  for (size_t i = 0; i < strides.size() - tiling_rank; ++i) {
    result.push_back(strides[i]);
  }

  for (size_t i = 0; i < tiling_rank; ++i) {
    int64_t s = strides[strides.size() - tiling_rank + i];
    int64_t t = tiling[i];
    result.push_back(s * t);
  }

  for (size_t i = 0; i < tiling_rank; ++i) {
    result.push_back(strides[strides.size() - tiling_rank + i]);
  }

  return result;
}

}  // namespace jax::mosaic::gpu
