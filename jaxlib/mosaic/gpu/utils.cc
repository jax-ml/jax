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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace jax::mosaic::gpu {

absl::StatusOr<mlir::Value> MemRefUnfold(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref, int64_t dim,
    const std::vector<std::optional<int64_t>>& factors) {
  auto ref_ty = mlir::cast<mlir::MemRefType>(ref.getType());
  auto shape = ref_ty.getShape();
  size_t unknown_dims = 0;
  for (std::optional<int64_t> factor : factors) {
    if (!factor.has_value()) {
      unknown_dims++;
    }
  }
  if (unknown_dims > 1) {
    return absl::InvalidArgumentError("Can only infer at most one dimension");
  }
  size_t known_factor_prod = 1;
  for (std::optional<int64_t> factor : factors) {
    if (factor.has_value()) {
      known_factor_prod *= *factor;
    }
  }

  if (known_factor_prod == 0) {
    return absl::UnimplementedError("Factor of 0 is not supported");
  }

  if (shape[dim] % known_factor_prod != 0) {
    return absl::InvalidArgumentError("Non-divisible unfold");
  }

  if (unknown_dims == 0 && shape[dim] != known_factor_prod) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid factor product: ", known_factor_prod,
                     ", expected: ", shape[dim]));
  }

  std::vector<int64_t> new_factors;
  for (int i = 0; i < factors.size(); ++i) {
    if (!factors[i]) {
      new_factors.push_back(shape[dim] / known_factor_prod);
      continue;
    }
    new_factors.push_back(*factors[i]);
  }

  std::vector<int64_t> new_shape;
  new_shape.reserve(shape.size() + new_factors.size() - 1);
  new_shape.insert(new_shape.end(), shape.begin(), shape.begin() + dim);
  new_shape.insert(new_shape.end(), new_factors.begin(), new_factors.end());
  new_shape.insert(new_shape.end(), shape.begin() + dim + 1, shape.end());

  mlir::AffineMapAttr identity =
      mlir::AffineMapAttr::get(mlir::AffineMap::getMultiDimIdentityMap(
          ref_ty.getRank(), builder.getContext()));
  mlir::Attribute contig_strided_1d =
      mlir::parseAttribute("strided<[1]>", builder.getContext());
  mlir::MemRefType new_ref_ty;
  if (ref_ty.getLayout() == identity ||
      ref_ty.getLayout() == contig_strided_1d) {
    auto new_layout =
        mlir::AffineMapAttr::get(mlir::AffineMap::getMultiDimIdentityMap(
            ref_ty.getRank() + new_factors.size() - 1, builder.getContext()));
    new_ref_ty = mlir::MemRefType::get(new_shape, ref_ty.getElementType(),
                                       new_layout, ref_ty.getMemorySpace());
  } else {
    auto [new_strides, offset] = ref_ty.getStridesAndOffset();
    int64_t prev_stride = new_strides[dim];
    std::vector<int64_t> inserted_strides;
    std::vector<int64_t> reversed_factors(new_factors.rbegin(),
                                          new_factors.rend());
    for (int64_t factor : reversed_factors) {
      inserted_strides.push_back(prev_stride);
      prev_stride *= factor;
    }
    new_strides.erase(new_strides.begin() + dim);
    new_strides.insert(new_strides.begin() + dim, inserted_strides.rbegin(),
                       inserted_strides.rend());
    auto new_layout =
        mlir::StridedLayoutAttr::get(builder.getContext(), offset, new_strides);
    new_ref_ty = mlir::MemRefType::get(new_shape, ref_ty.getElementType(),
                                       new_layout, ref_ty.getMemorySpace());
  }

  llvm::SmallVector<mlir::ReassociationIndices> assoc;
  assoc.resize(ref_ty.getRank());

  if (dim == ref_ty.getRank()) {
    for (int64_t i = 0; i < ref_ty.getRank(); ++i) {
      assoc[i].push_back(i);
    }
    for (int i = ref_ty.getRank();
         i < ref_ty.getRank() + new_factors.size() - 1; ++i) {
      assoc.back().push_back(i);
    }
  } else {
    for (int64_t i = 0; i < dim; ++i) {
      assoc[i].push_back(i);
    }

    for (int i = dim; i < dim + new_factors.size(); ++i) {
      assoc[dim].push_back(i);
    }

    for (int64_t i = dim + 1; i < ref_ty.getRank(); ++i) {
      int x = i + new_factors.size() - 1;
      assoc[i].push_back(x);
    }
  }

  CHECK(assoc.size() == ref_ty.getRank());

  return mlir::memref::ExpandShapeOp::create(builder, new_ref_ty, ref, assoc)
      .getResult();
}

absl::StatusOr<mlir::Value> MemRefSlice(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref,
    const std::vector<std::variant<int64_t, mlir::Value>>& base_indices,
    const std::vector<int64_t>& slice_shape,
    const std::vector<bool>& is_squeezed) {
  auto ref_ty = mlir::cast<mlir::MemRefType>(ref.getType());
  if (base_indices.size() != ref_ty.getRank() ||
      slice_shape.size() != ref_ty.getRank() ||
      is_squeezed.size() != ref_ty.getRank()) {
    return absl::InvalidArgumentError("Indices must match memref rank");
  }

  auto [memref_strides, offset] = ref_ty.getStridesAndOffset();
  int64_t dynamic_offset = mlir::ShapedType::kDynamic;
  int64_t new_offset = offset;

  if (new_offset != dynamic_offset) {
    for (const auto& [base_idx, memref_stride] :
         llvm::zip(base_indices, memref_strides)) {
      if (std::holds_alternative<int64_t>(base_idx)) {
        new_offset += std::get<int64_t>(base_idx) * memref_stride;
      } else {
        new_offset = dynamic_offset;
        break;
      }
    }
  }

  std::vector<int64_t> new_strides;
  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < ref_ty.getRank(); ++i) {
    if (!is_squeezed[i]) {
      new_strides.push_back(memref_strides[i]);
      new_shape.push_back(slice_shape[i]);
    } else if (slice_shape[i] != 1) {
      return absl::InvalidArgumentError(
          "Slice shape must be 1 for squeezed dimensions");
    }
  }

  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;

  for (size_t i = 0; i < ref_ty.getRank(); ++i) {
    if (std::holds_alternative<int64_t>(base_indices[i])) {
      offsets.push_back(
          builder.getIndexAttr(std::get<int64_t>(base_indices[i])));
    } else {
      offsets.push_back(std::get<mlir::Value>(base_indices[i]));
    }
    sizes.push_back(builder.getIndexAttr(slice_shape[i]));
    strides.push_back(builder.getIndexAttr(1));
  }

  auto new_layout = mlir::StridedLayoutAttr::get(builder.getContext(),
                                                 new_offset, new_strides);
  auto new_ref_ty = mlir::MemRefType::get(new_shape, ref_ty.getElementType(),
                                          new_layout, ref_ty.getMemorySpace());

  return builder
      .create<mlir::memref::SubViewOp>(new_ref_ty, ref, offsets, sizes, strides)
      .getResult();
}

absl::StatusOr<mlir::Value> MemRefTranspose(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value ref,
    const std::vector<int64_t>& permutation) {
  auto ref_ty = mlir::cast<mlir::MemRefType>(ref.getType());
  if (permutation.size() != ref_ty.getRank()) {
    return absl::InvalidArgumentError("Permutation rank mismatch");
  }

  auto [strides, offset] = ref_ty.getStridesAndOffset();
  std::vector<int64_t> new_strides;
  std::vector<int64_t> new_shape;
  new_strides.reserve(permutation.size());
  new_shape.reserve(permutation.size());

  for (int64_t p : permutation) {
    if (p < 0 || p >= ref_ty.getRank()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid permutation index. Expected 0 <= index < ",
                       ref_ty.getRank(), " but got ", p));
    }
    new_strides.push_back(strides[p]);
    new_shape.push_back(ref_ty.getShape()[p]);
  }

  auto new_layout =
      mlir::StridedLayoutAttr::get(builder.getContext(), offset, new_strides);
  auto new_ty = mlir::MemRefType::get(new_shape, ref_ty.getElementType(),
                                      new_layout, ref_ty.getMemorySpace());

  mlir::AffineMap permutation_map =
      mlir::AffineMap::getPermutationMap(permutation, builder.getContext());
  return builder
      .create<mlir::memref::TransposeOp>(
          new_ty, ref, mlir::AffineMapAttr::get(permutation_map))
      .getResult();
}

mlir::Value c(mlir::ImplicitLocOpBuilder& b, int64_t val, mlir::Type type) {
  CHECK(type.isInteger()) << "Only integer type supported for integer values";
  return mlir::arith::ConstantOp::create(b, type, b.getIntegerAttr(type, val));
}

}  // namespace jax::mosaic::gpu
