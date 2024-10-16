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

#include "jaxlib/mosaic/dialect/tpu/util.h"

#include <cstdint>

#include "llvm/Support/MathExtras.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {
SmallVector<int64_t> ComputeTileStrides(MemRefType memref_ty,
                                        absl::Span<const int64_t> tiling) {
  SmallVector<int64_t> tile_strides(memref_ty.getRank());
  int64_t stride = 1;
  for (int64_t i = 0; i < memref_ty.getRank(); ++i) {
    int64_t idx = memref_ty.getRank() - 1 - i;
    int64_t tiling_idx = tiling.size() - 1 - i;
    tile_strides[idx] = stride;
    if (tiling_idx >= 0) {
      stride *= llvm::divideCeil(memref_ty.getShape()[idx], tiling[tiling_idx]);
    } else {
      stride *= memref_ty.getShape()[idx];
    }
  }
  return tile_strides;
}

DotDimensionNumbersAttr default_dimension_numbers(Builder& builder,
                                                  bool transpose_lhs,
                                                  bool transpose_rhs) {
  return tpu::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhs_contracting_dims=*/{transpose_lhs ? 0 : 1},
      /*rhs_contracting_dims=*/{transpose_rhs ? 1 : 0},
      /*lhs_non_contracting_dims=*/{transpose_lhs ? 1 : 0},
      /*rhs_non_contracting_dims=*/{transpose_rhs ? 0 : 1},
      /*output_dim_order=*/{0, transpose_lhs ? 1 : 0, 1, transpose_rhs ? 0 : 1},
      /*lhs_batch_dims=*/{},
      /*rhs_batch_dims=*/{});
}

std::pair<bool, bool> is_transposed_mkn(DotDimensionNumbersAttr dim_numbers) {
  auto lhs_batch_dims = dim_numbers.getLhsBatchDims();
  auto rhs_batch_dims = dim_numbers.getRhsBatchDims();
  auto lhs_contracting_dims = dim_numbers.getLhsContractingDims();
  auto rhs_contracting_dims = dim_numbers.getRhsContractingDims();
  auto lhs_non_contracting_dims = dim_numbers.getLhsNonContractingDims();
  auto rhs_non_contracting_dims = dim_numbers.getRhsNonContractingDims();

  // Check that batch dimensions are first
  auto are_batch_dims_first = [](const std::vector<int64_t>& batch_dims) {
    for (size_t i = 0; i < batch_dims.size(); ++i) {
      if (batch_dims[i] != static_cast<int64_t>(i)) {
        return false;
      }
    }
    return true;
  };

  CHECK(are_batch_dims_first(lhs_batch_dims) &&
        are_batch_dims_first(rhs_batch_dims));
  CHECK(lhs_batch_dims.size() == rhs_batch_dims.size());
  CHECK(lhs_batch_dims.size() <= 1);
  CHECK(lhs_contracting_dims.size() == 1 && rhs_contracting_dims.size() == 1 &&
        lhs_non_contracting_dims.size() == 1 &&
        rhs_non_contracting_dims.size() == 1);

  int64_t lhs_non_contracting_dim = lhs_non_contracting_dims[0];
  int64_t lhs_contracting_dim = lhs_contracting_dims[0];
  int64_t rhs_non_contracting_dim = rhs_non_contracting_dims[0];
  int64_t rhs_contracting_dim = rhs_contracting_dims[0];

  bool lhs_transposed = lhs_non_contracting_dim > lhs_contracting_dim;

  bool rhs_transposed = rhs_contracting_dim > rhs_non_contracting_dim;

  return {lhs_transposed, rhs_transposed};
}

}  // namespace mlir::tpu
