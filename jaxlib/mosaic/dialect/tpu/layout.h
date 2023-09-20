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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_

#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <tuple>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tpu {

// TODO(apaszke): Optimize this to encode the optional in the value itself
// and use a narrower type.
// An offset is nullopt when the value is replicated along sublanes or lanes.
using LayoutOffset = std::optional<int64_t>;
using LayoutOffsets = std::array<LayoutOffset, 2>;

enum class Direction { kSublanes, kLanes, kSubelements };

struct VRegDataBounds {
  // TODO(tlongeri): Should get{Vector, Sublane}Mask take a Location?
  virtual ~VRegDataBounds() = default;
  // Determines whether all indices along a direction contain useful data.
  virtual bool maskVariesAlong(Direction direction,
                               std::array<int64_t, 2> target_shape) const = 0;

  bool isComplete(const std::array<int64_t, 2> target_shape) const {
    return maskVariesAlong(Direction::kSublanes, target_shape) ||
           maskVariesAlong(Direction::kLanes, target_shape) ||
           maskVariesAlong(Direction::kSubelements, target_shape);
  }

  // Constructs a vector mask value that is true iff the entry contains useful
  // data.
  //
  // The returned value can be an int32 bitmask too, when the target does not
  // have sufficiently expressive vector masks.
  //
  // Args:
  //   generation: The target TPU generation.
  virtual FailureOr<TypedValue<VectorType>> getVectorMask(
      OpBuilder &builder, Location loc, int generation,
      std::array<int64_t, 2> target_shape) const = 0;

  // Constructs a DenseBoolArrayAttr containing a sublane mask for the vreg.
  //
  // The sublane mask should never have true for sublanes that do not contain
  // useful data, but having an unmasked sublane doesn't imply that all bits
  // in that sublane are used to represent data (relevant for packed layouts).
  virtual DenseBoolArrayAttr getSublaneMask(
      MLIRContext *ctxt, std::array<int64_t, 2> target_shape) const = 0;
};

// Represents a rectangular region of data within a vector register.
//
// This class is very limited in its power and should only be used for 32-bit
// values with native tiling.
//
// Attributes:
//   bounds: A TargetTuple of slices encoding the bounds of the rectangular
//     data region.
// TODO(tlongeri): Can this be removed in favor of the more general
//  TiledRectangularVregBounds?
class RectangularVregBounds : public VRegDataBounds {
 public:
  RectangularVregBounds(const std::array<int64_t, 2> starts,
                        const std::array<int64_t, 2> ends)
      : starts_(starts), ends_(ends) {}

  // See base class.
  bool maskVariesAlong(Direction direction,
                       std::array<int64_t, 2> target_shape) const override;

  // See base class.
  FailureOr<TypedValue<VectorType>> getVectorMask(
      OpBuilder &builder, Location loc, int generation,
      std::array<int64_t, 2> target_shape) const override;

  // See base class.
  DenseBoolArrayAttr getSublaneMask(
      MLIRContext *mlir_ctxt,
      std::array<int64_t, 2> target_shape) const override;

 private:
  std::array<int64_t, 2> starts_;
  std::array<int64_t, 2> ends_;
};

class VectorLayout {
 public:
  enum class ImplicitDim {
    kNone = 0,  // To make if (implicit_dim) work.
    // Also want to do dims[dims.size() - xla::to_underlying(implicit_dim)]
    kMinor = 1,
    kSecondMinor = 2,
  };
  VectorLayout(const int8_t bitwidth, const LayoutOffsets offsets,
               const std::array<int64_t, 2> tiling,
               const ImplicitDim implicit_dim = ImplicitDim::kNone)
      : offsets_(offsets),
        tiling_(tiling),
        bitwidth_(bitwidth),
        implicit_dim_(implicit_dim) {
    // TODO(b/275751535): Allow more bitwidths.
    CHECK(llvm::has_single_bit<unsigned>(bitwidth_) && bitwidth_ <= 32);
    // Offsets should not exceed the tile size. The data always starts within
    // the first tile of a vreg.
    for (auto [o, t] : llvm::zip(offsets_, tiling_)) {
      CHECK(!o || 0 <= *o && *o < t);
    }
  }

  int8_t bitwidth() const { return bitwidth_; }
  const LayoutOffsets &offsets() const { return offsets_; }
  const std::array<int64_t, 2> &tiling() const { return tiling_; }
  ImplicitDim implicit_dim() const { return implicit_dim_; }
  int packing() const { return 32 / bitwidth_; }
  // The number of minormost dimensions tiled by this layout.
  int layout_rank() const { return 1 + (implicit_dim_ == ImplicitDim::kNone); }

  bool operator==(const VectorLayout &other) const;

  // How many tiles fit in each vector register.
  int64_t tilesPerVreg(const std::array<int64_t, 2> target_shape) const {
    const int64_t tile_elems = tiling_[0] * tiling_[1];
    const int64_t vreg_capacity = packing() * target_shape[0] * target_shape[1];
    const auto [tiles_per_vreg, rem] = std::div(vreg_capacity, tile_elems);
    CHECK_EQ(rem, 0);
    return tiles_per_vreg;
  }

  int64_t sublanesPerTile(const std::array<int64_t, 2> target_shape) const {
    auto [sublanes_per_tile, rem] =
        std::div(target_shape[0], tilesPerVreg(target_shape));
    CHECK_EQ(rem, 0);
    return sublanes_per_tile;
  }

  // Returns the size of a window contained in a single vreg.
  //
  // We never reuse the same vector register to store data of multiple rows,
  // so only the minormost dimension can increase.
  std::array<int64_t, 2> vregSlice(std::array<int64_t, 2> target_shape) const {
    return {tiling_[0], tilesPerVreg(target_shape) * tiling_[1]};
  }

  llvm::SmallVector<int64_t> implicitShape(ArrayRef<int64_t> shape) const;

 private:
  llvm::SmallVector<int64_t> tileArrayImplicitShape(
      ArrayRef<int64_t> shape, std::array<int64_t, 2> target_shape) const;

 public:
  // Returns the shape of ndarray of vregs needed to represent a value.
  //
  // All but the last two dimensions are unrolled over vregs. In the last two
  // dims we need as many vregs as indicated by dividing the point at which
  // the value ends (given by the start offset plus the dim size) divided by
  // the respective vreg capacity in that dim (and a ceiling if non-integral).
  // If a value is replicated, then any offset is valid and we pick 0 to
  // minimize the number of vregs.
  //
  // Args:
  // - shape: The shape of the full vector this layout applies to.
  llvm::SmallVector<int64_t> tileArrayShape(
      ArrayRef<int64_t> shape, std::array<int64_t, 2> target_shape) const;

  // Returns the bounds of the given tile that hold useful data.
  //
  // Arguments:
  //   full_shape: The shape of the full vector this layout applies to.
  //   ixs: The indices into an array of tiles representing the full vector
  //     (see tile_array_shape for bounds) selecting the tile for which the
  //     bounds are queried.
  //   allow_replicated: If False, no offset is allowed to be replicated. If
  //     True, offsets are allowed to be replicated, but the bounds will span
  //     the full dimension of the tile (i.e. potentially multiple repeats of
  //     the actual data).
  //
  // Returns:
  //   A TargetTuple of slices, indicating the span of useful data within the
  //   tile selected by idx.
  std::unique_ptr<VRegDataBounds> tileDataBounds(
      MLIRContext *mlir_ctxt, ArrayRef<int64_t> full_shape,
      ArrayRef<int64_t> idxs, std::array<int64_t, 2> target_shape,
      std::array<bool, 2> allow_replicated) const;
  std::unique_ptr<VRegDataBounds> tileDataBounds(
      MLIRContext *mlir_ctxt, ArrayRef<int64_t> full_shape,
      ArrayRef<int64_t> idxs, std::array<int64_t, 2> target_shape,
      bool allow_replicated = false) const {
    return tileDataBounds(mlir_ctxt, full_shape, idxs, target_shape,
                          {allow_replicated, allow_replicated});
  }

  // True if every vector register has a layout without jumps.
  //
  // By without jumps we mean that traversing vregs over (sub)lanes always leads
  // to a contiguous traversal of the (second) minormost dimension of data. This
  // is only true for 32-bit types, since narrower types use two level tiling.
  bool hasNaturalTopology(const std::array<int64_t, 2> target_shape) const {
    return bitwidth_ == 32 && llvm::equal(tiling_, target_shape) &&
           implicit_dim_ == ImplicitDim::kNone;
  }
  // True if every vector register has a natural "packed" topology.
  //
  // This is equivalent to has_natural_topology for 32-bit types, but
  // generalizes it to narrower values with packed layouts too.
  bool hasNativeTiling(std::array<int64_t, 2> target_shape) const;

  // Returns true if the other layout is a special case of this one.
  //
  // In here, other is considered "a special case" when the set of vector
  // register entries that represent a value in that layout is also the set of
  // entries in which this stores the value. This is of course true for layouts
  // that are equivalent, but it does not need to hold both ways. For example,
  // a layout that implies the value does not change along an axis of the vector
  // register is more general than the layout that picks a fixed starting point
  // for the value and does not encode that assumption.
  //
  // The generalization relation is a non-strict partial order. You can think of
  // it as a partial <= on vector layouts, but we don't overload operators since
  // there's no clear way to decide where the bottom and top should be.
  //
  // Args:
  //   other: The layout compared against this.
  //   shape: A optional shape of the vector to which both layouts apply.
  //     If shape.data() == nullptr, then return whether it generalizes across
  //     all shapes.
  //     The generalization relation is larger than usual for some shapes. That
  //     is, if generalizes(other) then also generalizes(other, shape) for any
  //     shape, but that implication does not hold the other way around for some
  //     shapes.
  bool generalizes(const VectorLayout &other, ArrayRef<int64_t> shape,
                   std::array<int64_t, 2> target_shape) const;

  // Returns True if the two layouts are equivalent.
  //
  // That is, when all potential vector entries where the value can be stored
  // (there might be multiple choices for some layouts!) are equal in both
  // self and other.
  //
  // Args:
  //   other: The layout compared against self.
  //   shape: An optional shape of the vector to which both layouts apply. More
  //     layouts are considered equivalent when the shape is specified. Also see
  //     the docstring of the generalizes method.
  bool equivalentTo(const VectorLayout &other, const ArrayRef<int64_t> shape,
                    const std::array<int64_t, 2> target_shape) const {
    return generalizes(other, shape, target_shape) &&
           other.generalizes(*this, shape, target_shape);
  }

  template <typename Stream>
  void print(Stream &os) const;

  static std::optional<VectorLayout> join(const VectorLayout &l,
                                          const VectorLayout &r,
                                          ArrayRef<int64_t> shape);

  static std::optional<VectorLayout> parse(llvm::StringRef *data);

 private:
  std::tuple<std::optional<int64_t>, std::optional<int64_t>, int64_t, int64_t,
             int8_t, ImplicitDim>
  as_tuple() const;

  // Check conditions that depend on the target shape. Invariants that are
  // independent of it are checked in the constructor.
  void verify(const std::array<int64_t, 2> target_shape) const {
    // Tiling should neatly divide the target shape, so that every vector
    // register ends up having the same structure.
    // Also, every tile should occupy a fixed number of sublanes.
    CHECK_EQ((tiling_[0] * tiling_[1]) % (packing() * target_shape[1]), 0);
  }

  friend llvm::hash_code hash_value(const VectorLayout &layout);

  LayoutOffsets offsets_;
  std::array<int64_t, 2> tiling_;
  int8_t bitwidth_;
  ImplicitDim implicit_dim_;
};

using Layout = std::optional<VectorLayout>;
extern const Layout kNoLayout;

std::ostream &operator<<(std::ostream &os, const Layout &v);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Layout &v);
llvm::hash_code hash_value(const VectorLayout &layout);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const Layout &v);

std::optional<Layout> parseLayout(mlir::AsmParser &parser);

}  // namespace mlir::tpu

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_
