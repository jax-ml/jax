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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFER_PLAN_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFER_PLAN_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace jax::mosaic::gpu {

class TransferPlan {
 public:
  using IndexTransform =
      std::function<std::vector<int64_t>(const std::vector<int64_t>&)>;

  virtual ~TransferPlan() = default;

  virtual std::vector<IndexTransform> TileIndexTransforms() const = 0;

  // Selects the value corresponding to the group of the current thread.
  // The argument must be of the same length as `TileIndexTransforms()`.
  virtual absl::StatusOr<mlir::Value> Select(
      mlir::ImplicitLocOpBuilder& builder,
      const std::vector<mlir::Value>& group_elems) const = 0;

  // Returns `new_val` if the current thread belongs to the given group and
  // `old_val` otherwise.
  //
  // `group_idx` must be between 0 and `len(TileIndexTransforms()) - 1`.
  virtual absl::StatusOr<mlir::Value> SelectIfGroup(
      mlir::ImplicitLocOpBuilder& builder, int64_t group_idx,
      mlir::Value old_val, mlir::Value new_val) const = 0;
};

class TrivialTransferPlan : public TransferPlan {
 public:
  std::vector<IndexTransform> TileIndexTransforms() const override;

  absl::StatusOr<mlir::Value> Select(
      mlir::ImplicitLocOpBuilder& builder,
      const std::vector<mlir::Value>& group_elems) const override;

  absl::StatusOr<mlir::Value> SelectIfGroup(mlir::ImplicitLocOpBuilder& builder,
                                            int64_t group_idx,
                                            mlir::Value old_val,
                                            mlir::Value new_val) const override;
};

class StaggeredTransferPlan : public TransferPlan {
 public:
  StaggeredTransferPlan(int64_t stagger, int64_t dim, int64_t size,
                        mlir::Value group_pred)
      : stagger_(stagger), dim_(dim), size_(size), group_pred_(group_pred) {}

  std::vector<IndexTransform> TileIndexTransforms() const override;

  absl::StatusOr<mlir::Value> Select(
      mlir::ImplicitLocOpBuilder& builder,
      const std::vector<mlir::Value>& group_elems) const override;

  absl::StatusOr<mlir::Value> SelectIfGroup(mlir::ImplicitLocOpBuilder& builder,
                                            int64_t group_idx,
                                            mlir::Value old_val,
                                            mlir::Value new_val) const override;

  int64_t stagger() const { return stagger_; }
  int64_t dim() const { return dim_; }
  int64_t size() const { return size_; }
  mlir::Value group_pred() const { return group_pred_; }

 private:
  int64_t stagger_;
  int64_t dim_;
  int64_t size_;
  mlir::Value group_pred_;
};

}  // namespace jax::mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_TRANSFER_PLAN_H_
