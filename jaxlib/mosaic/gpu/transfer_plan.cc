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

#include "jaxlib/mosaic/gpu/transfer_plan.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace jax::mosaic::gpu {

std::vector<TransferPlan::IndexTransform>
TrivialTransferPlan::TileIndexTransforms() const {
  return {[](const std::vector<int64_t>& idx) { return idx; }};
}

absl::StatusOr<mlir::Value> TrivialTransferPlan::Select(
    mlir::ImplicitLocOpBuilder& builder,
    const std::vector<mlir::Value>& group_elems) const {
  if (group_elems.size() != 1) {
    return absl::FailedPreconditionError(
        "TrivialTransferPlan expected 1 element");
  }
  return group_elems[0];
}

absl::StatusOr<mlir::Value> TrivialTransferPlan::SelectIfGroup(
    mlir::ImplicitLocOpBuilder& builder, int64_t group_idx, mlir::Value old_val,
    mlir::Value new_val) const {
  if (group_idx != 0) {
    return absl::FailedPreconditionError(
        "TrivialTransferPlan expected group_idx 0");
  }
  return new_val;
}

std::vector<TransferPlan::IndexTransform>
StaggeredTransferPlan::TileIndexTransforms() const {
  auto rotate =
      [this](const std::vector<int64_t>& idx) -> std::vector<int64_t> {
    std::vector<int64_t> new_idx = idx;
    CHECK_GE(dim_, 0);
    CHECK_LT(dim_, idx.size());
    new_idx[dim_] = (idx[dim_] + stagger_) % size_;
    return new_idx;
  };
  return {[](const std::vector<int64_t>& idx) { return idx; }, rotate};
}

absl::StatusOr<mlir::Value> StaggeredTransferPlan::Select(
    mlir::ImplicitLocOpBuilder& builder,
    const std::vector<mlir::Value>& group_elems) const {
  if (group_elems.size() != 2) {
    return absl::FailedPreconditionError(
        "StaggeredTransferPlan expected 2 elements");
  }
  return builder
      .create<mlir::arith::SelectOp>(group_pred_, group_elems[1],
                                     group_elems[0])
      .getResult();
}

absl::StatusOr<mlir::Value> StaggeredTransferPlan::SelectIfGroup(
    mlir::ImplicitLocOpBuilder& builder, int64_t group_idx, mlir::Value old_val,
    mlir::Value new_val) const {
  if (group_idx != 0 && group_idx != 1) {
    return absl::FailedPreconditionError(
        "StaggeredTransferPlan expected group_idx 0 or 1");
  }
  mlir::Value true_val = (group_idx == 0) ? old_val : new_val;
  mlir::Value false_val = (group_idx == 0) ? new_val : old_val;
  return builder.create<mlir::arith::SelectOp>(group_pred_, true_val, false_val)
      .getResult();
}

}  // namespace jax::mosaic::gpu
