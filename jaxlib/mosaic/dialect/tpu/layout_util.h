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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_UTIL_H_

#include <array>
#include <cstdint>

#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"

namespace mlir::tpu {

// Sets the `in_layout` attribute of the given operation.
void setInLayout(Operation *op, ArrayRef<Layout> in);

// Sets the `out_layout` attribute of the given operation.
void setOutLayout(Operation *op, Layout out);

// Same as above, but for multiple outputs.
void setOutLayout(Operation *op, ArrayRef<Layout> out);

// Sets both `in_layout` and `out_layout` attributes of the given operation.
void setLayout(Operation *op, Layout in, Layout out);

// Same as above, but for multiple inputs.
void setLayout(Operation *op, ArrayRef<Layout> in, Layout out);

// Same as above, but for multiple outputs.
void setLayout(Operation *op, Layout in, ArrayRef<Layout> out);

// Same as above, but for multiple inputs and outputs.
void setLayout(Operation *op, ArrayRef<Layout> in, ArrayRef<Layout> out);

// Returns the `in_layout` attribute of the given operation. If the operation
// does not have the attribute, returns an empty vector.
SmallVector<Layout, 4> getInLayout(Operation *op);

// Same as above, but returns a failure if any of input layouts is invalid for
// the target shape.
FailureOr<SmallVector<Layout, 4>> getInLayout(
    Operation *op, std::array<int64_t, 2> target_shape);

// Returns the `out_layout` attribute of the given operation. If the operation
// does not have the attribute, returns an empty vector.
SmallVector<Layout, 4> getOutLayout(Operation *op);

// Same as above, but returns a failure if any of output layouts is invalid for
// the target shape.
FailureOr<SmallVector<Layout, 4>> getOutLayout(
    Operation *op, std::array<int64_t, 2> target_shape);

// Returns the layout of the given value.
// If `force_first_tile_offsets` is true, the out-of-first-tile offset to be
// zero.
Layout getLayout(Value v, bool force_first_tile_offsets = false);

// Returns the layout of the given operation's operands.
SmallVector<Layout, 4> getLayoutFromOperands(Operation *op);

// Returns true if the given layout's tiling and offsets for a value are
// compatible with the target shape.
bool layoutIsValidForValue(const Layout &l, Value v,
                           std::array<int64_t, 2> target_shape);

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_UTIL_H_
