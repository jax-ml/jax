/* Copyright 2023 The JAX Authors.

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

#include "jaxlib/mosaic/dialect/tpu/layout.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "absl/log/check.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

bool RectangularVregBounds::maskVariesAlong(
    const Direction direction,
    const std::array<int64_t, 2> target_shape) const {
  switch (direction) {
    case Direction::kSublanes:
      return starts_[0] != 0 || ends_[0] != target_shape[0];
    case Direction::kLanes:
      return starts_[1] != 0 || ends_[1] != target_shape[1];
    case Direction::kSubelements:
      return false;
  }
}

FailureOr<TypedValue<VectorType>> RectangularVregBounds::getVectorMask(
    OpBuilder& builder, const int /*generation*/,
    const std::array<int64_t, 2> target_shape) const {
  auto boundIdxConst = std::bind(IdxConst, std::placeholders::_1, builder,
                                 builder.getUnknownLoc());
  return cast<TypedValue<VectorType>>(
      builder
          .create<tpu::CreateMaskOp>(
              builder.getUnknownLoc(),
              VectorType::get(target_shape, builder.getI1Type()),
              /*low=*/
              ValueRange{boundIdxConst(starts_[0]), boundIdxConst(starts_[1])},
              /*high=*/
              ValueRange{boundIdxConst(ends_[0]), boundIdxConst(ends_[1])})
          .getResult());
}

DenseBoolArrayAttr RectangularVregBounds::getSublaneMask(
    MLIRContext* mlir_ctxt, const std::array<int64_t, 2> target_shape) const {
  llvm::SmallVector<bool, 8> sublane_mask(target_shape[0], false);
  for (int64_t i = starts_[0]; i < ends_[0]; ++i) {
    sublane_mask[i] = true;
  }
  return DenseBoolArrayAttr::get(mlir_ctxt, sublane_mask);
}

namespace {

mlir::ParseResult parseOffset(llvm::StringRef* data,
                              std::optional<int64_t>* result) {
  int64_t int_result;
  if (data->consume_front("*")) {
    *result = std::nullopt;
    return success();
  }
  if (!data->consumeInteger(10, int_result)) {
    *result = int_result;
    return success();
  }
  return failure();
}

std::array<int64_t, 2> nativeTiling(const int8_t bitwidth,
                                    const std::array<int64_t, 2> target_shape) {
  const int packing = 32 / bitwidth;
  return {target_shape[0] * packing, target_shape[1]};
}

}  // namespace

std::tuple<std::optional<int64_t>, std::optional<int64_t>, int64_t, int64_t,
           int8_t, VectorLayout::ImplicitDim>
VectorLayout::as_tuple() const {
  return std::make_tuple(offsets_[0], offsets_[1], tiling_[0], tiling_[1],
                         bitwidth_, implicit_dim_);
}

bool VectorLayout::operator==(const VectorLayout& other) const {
  return as_tuple() == other.as_tuple();
}

bool VectorLayout::hasNativeTiling(
    const std::array<int64_t, 2> target_shape) const {
  return tiling_ == nativeTiling(bitwidth_, target_shape);
}

llvm::SmallVector<int64_t> VectorLayout::implicitShape(
    ArrayRef<int64_t> shape) const {
  CHECK(!shape.empty());
  switch (implicit_dim_) {
    case ImplicitDim::kNone:
      return llvm::SmallVector<int64_t>(shape);
    case ImplicitDim::kMinor: {
      llvm::SmallVector<int64_t> implicit_shape;
      implicit_shape.reserve(shape.size() + 1);
      implicit_shape.append(shape.begin(), shape.end());
      implicit_shape.push_back(1);
      return implicit_shape;
    }
    case ImplicitDim::kSecondMinor: {
      llvm::SmallVector<int64_t> implicit_shape;
      implicit_shape.reserve(shape.size() + 1);
      implicit_shape.append(shape.begin(), std::prev(shape.end()));
      implicit_shape.push_back(1);
      implicit_shape.push_back(shape.back());
      return implicit_shape;
    }
  }
}

llvm::SmallVector<int64_t> VectorLayout::tileArrayImplicitShape(
    const ArrayRef<int64_t> shape,
    const std::array<int64_t, 2> target_shape) const {
  const std::array<int64_t, 2> vreg_slice = vregSlice(target_shape);
  llvm::SmallVector<int64_t> tiles_shape = implicitShape(shape);
  tiles_shape[tiles_shape.size() - 2] =
      ceilDiv(offsets_[0].value_or(0) + tiles_shape[tiles_shape.size() - 2],
              vreg_slice[0]);
  tiles_shape[tiles_shape.size() - 1] =
      ceilDiv(offsets_[1].value_or(0) + tiles_shape[tiles_shape.size() - 1],
              vreg_slice[1]);
  return tiles_shape;
}

llvm::SmallVector<int64_t> VectorLayout::tileArrayShape(
    const ArrayRef<int64_t> shape,
    const std::array<int64_t, 2> target_shape) const {
  llvm::SmallVector<int64_t> tiles_shape =
      tileArrayImplicitShape(shape, target_shape);
  // Remove the implicit dimension --- it's always of size 1.
  switch (implicit_dim_) {
    case ImplicitDim::kNone:
      break;
    case ImplicitDim::kMinor:
      tiles_shape.pop_back();
      break;
    case ImplicitDim::kSecondMinor:
      tiles_shape.erase(tiles_shape.end() - 1);
      break;
  }
  return tiles_shape;
}

bool VectorLayout::generalizes(
    const VectorLayout& other, ArrayRef<int64_t> shape,
    const std::array<int64_t, 2> target_shape) const {
  if (bitwidth_ != other.bitwidth_) {
    return false;
  }
  for (auto [s, o] : llvm::zip(offsets_, other.offsets_)) {
    if (s.has_value() && s != o) {
      return false;
    }
  }
  if (implicit_dim_ != other.implicit_dim_) {
    // Don't fail yet!
    // If the second-minor dimension is of size 1, then it does not matter
    // whether we have a second minor implicit dim or not.
    if (shape.data() == nullptr) {
      return false;
    }
    const llvm::SmallVector<int64_t> implicit_shape = implicitShape(shape);
    if (!(implicit_shape[implicit_shape.size() - 2] == 1 &&
          ((implicit_dim_ == ImplicitDim::kSecondMinor &&
            other.implicit_dim_ == ImplicitDim::kNone) ||
           (other.implicit_dim_ == ImplicitDim::kSecondMinor &&
            implicit_dim_ == ImplicitDim::kNone)))) {
      return false;
    }
  }
  if (tiling_ != other.tiling_) {
    // Don't fail yet!
    // If there is only one tile in both tilings, then they are equivalent.
    if (shape.data() == nullptr) {
      return false;
    }
    const SmallVector<int64_t> ishape = implicitShape(shape);
    if (!(tiling_[1] == other.tiling_[1] &&
          tiling_[1] == target_shape[1] &&
          offsets_[1].value_or(0) + ishape[ishape.size() - 1] <=
              target_shape[1] &&
          offsets_[0].value_or(0) + ishape[ishape.size() - 2] <=
              std::min(tiling_[0], other.tiling_[0]))) {
      return false;
    }
  }
  return true;
}

template <typename Stream>
void VectorLayout::print(Stream& os) const {
  os << static_cast<int32_t>(bitwidth_) << ",{";
  bool first = true;
  for (auto o : offsets_) {
    if (first) {
      first = false;
    } else {
      os << ',';
    }
    if (!o) {
      os << '*';
    } else {
      os << *o;
    }
  }
  os << "},(" << tiling_[0] << ',' << tiling_[1] << ")";
  if (implicit_dim_ == ImplicitDim::kMinor) {
    os << ",-1";
  } else if (implicit_dim_ == ImplicitDim::kSecondMinor) {
    os << ",-2";
  }
}

std::optional<VectorLayout> VectorLayout::join(const VectorLayout& l,
                                               const VectorLayout& r,
                                               ArrayRef<int64_t> shape) {
  if (l.bitwidth_ != r.bitwidth_ || l.tiling_ != r.tiling_) {
    return std::nullopt;
  }
  if (l.implicit_dim_ != r.implicit_dim_) {
    if (shape.size() < 2) {
      return std::nullopt;
    }
    ImplicitDim dim;
    if (l.implicit_dim_ == ImplicitDim::kNone) {
      dim = r.implicit_dim_;
    } else if (r.implicit_dim_ == ImplicitDim::kNone) {
      dim = l.implicit_dim_;
    } else {
      return std::nullopt;
    }
    if (dim == ImplicitDim::kMinor && shape[shape.size() - 1] == 1) {
      // OK, they are equivalent.
    } else if (dim == ImplicitDim::kSecondMinor &&
               shape[shape.size() - 2] == 1) {
      // OK, they are equivalent.
    } else {
      return std::nullopt;
    }
  }
  LayoutOffsets offsets;
  for (int i = 0; i < 2; ++i) {
    auto lo = l.offsets()[i];
    auto ro = r.offsets()[i];
    if (lo && ro && lo != ro) {
      return std::nullopt;
    }
    offsets[i] = lo.has_value() ? lo : ro;
  }
  return VectorLayout(l.bitwidth_, offsets, l.tiling_, l.implicit_dim_);
}

std::optional<VectorLayout> VectorLayout::parse(llvm::StringRef* data) {
  llvm::StringRef local(*data);
  int8_t bitwidth;
  LayoutOffsets offsets;
  std::array<int64_t, 2> tiling;
  ImplicitDim implicit_dim = ImplicitDim::kNone;
  if (local.consumeInteger(10, bitwidth) || !local.consume_front(",{") ||
      parseOffset(&local, &offsets[0]) || !local.consume_front(",") ||
      parseOffset(&local, &offsets[1]) || !local.consume_front("},(") ||
      local.consumeInteger(10, tiling[0]) || !local.consume_front(",") ||
      local.consumeInteger(10, tiling[1]) || !local.consume_front(")")) {
    return std::nullopt;
  }
  if (local.consume_front(",-1")) {
    implicit_dim = ImplicitDim::kMinor;
  } else if (local.consume_front(",-2")) {
    implicit_dim = ImplicitDim::kSecondMinor;
  }
  *data = local;
  return VectorLayout(bitwidth, offsets, tiling, implicit_dim);
}

namespace {
template <class>
inline constexpr bool false_v = false;

template <typename Stream>
Stream& printLayout(Stream& os, const Layout& v) {
  os << '"';
  if (v.has_value()) {
    v->print(os);
  } else {
    os << "none";
  }
  os << '"';
  return os;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Layout& v) {
  return printLayout<std::ostream>(os, v);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Layout& v) {
  return printLayout<llvm::raw_ostream>(os, v);
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Layout& v) {
  return printLayout<mlir::Diagnostic>(diag, v);
}

llvm::hash_code hash_value(const VectorLayout& layout) {
  return llvm::hash_value(layout.as_tuple());
}

std::optional<Layout> parseLayout(mlir::AsmParser& parser) {
  std::string layout_str;
  if (failed(parser.parseString(&layout_str))) {
    return std::nullopt;
  }
  if (layout_str == "none") {
    return kNoLayout;
  }
  llvm::StringRef ref(layout_str);
  if (auto layout = VectorLayout::parse(&ref); ref.empty()) {
    return *layout;
  }
  return std::nullopt;
}

const Layout kNoLayout = std::nullopt;

}  // namespace mlir::tpu
