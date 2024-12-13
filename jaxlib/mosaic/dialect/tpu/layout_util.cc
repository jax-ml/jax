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

#include "jaxlib/mosaic/dialect/tpu/layout_util.h"

#include <array>
#include <cstdint>

#include "absl/log/check.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

void setInLayout(Operation *op, ArrayRef<Layout> in) {
  CHECK_EQ(in.size(), op->getNumOperands()) << Print(op);
  SmallVector<Attribute, 4> in_attrs;
  in_attrs.reserve(in.size());
  for (const Layout &p : in) {
    in_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
  }
  op->setAttr("in_layout", ArrayAttr::get(op->getContext(), in_attrs));
}

void setOutLayout(Operation *op, Layout out) {
  setOutLayout(op, ArrayRef<Layout>(out));
}

void setOutLayout(Operation *op, ArrayRef<Layout> out) {
  SmallVector<Attribute, 4> out_attrs;
  out_attrs.reserve(out.size());
  for (const Layout &p : out) {
    out_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
  }
  op->setAttr("out_layout", ArrayAttr::get(op->getContext(), out_attrs));
}

void setLayout(Operation *op, Layout in, Layout out) {
  setLayout(op, ArrayRef<Layout>(in), ArrayRef<Layout>(out));
}

void setLayout(Operation *op, ArrayRef<Layout> in, Layout out) {
  setLayout(op, in, ArrayRef<Layout>(out));
}

void setLayout(Operation *op, Layout in, ArrayRef<Layout> out) {
  setLayout(op, ArrayRef<Layout>(in), out);
}

void setLayout(Operation *op, ArrayRef<Layout> in, ArrayRef<Layout> out) {
  setInLayout(op, in);
  setOutLayout(op, out);
}

SmallVector<Layout, 4> getInLayout(Operation *op) {
  CHECK(op);
  if (!op->hasAttr("in_layout")) {
    return {};
  }
  auto in_attrs = op->getAttrOfType<ArrayAttr>("in_layout").getValue();
  CHECK_EQ(in_attrs.size(), op->getNumOperands());
  SmallVector<Layout, 4> in_layouts;
  in_layouts.reserve(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); ++i) {
    in_layouts.push_back(cast<VectorLayoutAttr>(in_attrs[i]).getLayout());
  }
  return in_layouts;
}

FailureOr<SmallVector<Layout, 4>> getInLayout(
    Operation *op, std::array<int64_t, 2> target_shape) {
  const SmallVector<Layout, 4> in_layouts = getInLayout(op);
  if (in_layouts.size() != op->getNumOperands()) {
    return op->emitOpError("in_layout size does not match number of operands");
  }
  for (const auto [l, operand] :
       llvm::zip_equal(in_layouts, op->getOperands())) {
    if (!layoutIsValidForValue(l, operand, target_shape)) {
      return op->emitOpError("Invalid input layout");
    }
  }
  return in_layouts;
}

SmallVector<Layout, 4> getOutLayout(Operation *op) {
  CHECK(op);
  if (!op->hasAttr("out_layout")) {
    return {};
  }
  auto out_attrs = op->getAttrOfType<ArrayAttr>("out_layout").getValue();
  CHECK_EQ(out_attrs.size(), op->getNumResults());
  SmallVector<Layout, 4> out_layouts;
  out_layouts.reserve(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    out_layouts.push_back(cast<VectorLayoutAttr>(out_attrs[i]).getLayout());
  }
  return out_layouts;
}

FailureOr<SmallVector<Layout, 4>> getOutLayout(
    Operation *op, std::array<int64_t, 2> target_shape) {
  const SmallVector<Layout, 4> out_layouts = getOutLayout(op);
  if (out_layouts.size() != op->getNumResults()) {
    return op->emitOpError("out_layout size does not match number of results");
  }
  for (const auto [l, res] : llvm::zip_equal(out_layouts, op->getResults())) {
    if (!layoutIsValidForValue(l, res, target_shape)) {
      return op->emitOpError("Invalid output layout");
    }
  }
  return out_layouts;
}

Layout getLayout(Value v, bool force_first_tile_offsets) {
  auto op = v.getDefiningOp();
  CHECK(op);
  auto op_result = dyn_cast<OpResult>(v);
  CHECK(op_result);
  auto result_index = op_result.getResultNumber();
  auto out_attrs = op->getAttrOfType<ArrayAttr>("out_layout").getValue();
  CHECK(out_attrs.size() > result_index);
  auto layout = cast<VectorLayoutAttr>(out_attrs[result_index]).getLayout();
  if (force_first_tile_offsets &&
      layout->offsets()[1].value_or(0) >= layout->tiling()[1]) {
    // Force the out-of-first-tile offset to be zero.
    layout = VectorLayout(layout->bitwidth(), {layout->offsets()[0], 0},
                          layout->tiling(), layout->implicit_dim());
  }
  return layout;
}

SmallVector<Layout, 4> getLayoutFromOperands(Operation *op) {
  SmallVector<Layout, 4> layouts;
  layouts.reserve(op->getNumOperands());
  for (const auto &operand : op->getOperands()) {
    if (isa<VectorType>(operand.getType())) {
      layouts.push_back(getLayout(operand));
    } else {
      layouts.push_back(kNoLayout);
    }
  }
  return layouts;
}

bool layoutIsValidForValue(const Layout &l, Value v,
                           std::array<int64_t, 2> target_shape) {
  // l must be non-null iff v is of vector type
  if (const auto vty = dyn_cast<VectorType>(v.getType())) {
    if (!l.has_value()) {
      return false;
    }

    // Vector type should have the same bitwidth as the layout, except for the
    // i1 special case, used for vmasks (see comment for VectorLayout class).
    if (!vty.getElementType().isIntOrFloat()) {
      return false;
    }
    const int8_t bitwidth = vty.getElementTypeBitWidth();
    if (bitwidth != l->bitwidth() && bitwidth != 1) {
      return false;
    }

    return l->isValid(target_shape) && l->layout_rank() <= vty.getRank();
  }
  return !l.has_value();
}

}  // namespace mlir::tpu
