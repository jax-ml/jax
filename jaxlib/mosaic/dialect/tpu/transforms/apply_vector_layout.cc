#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"
#include "xla/array.h"
#include "xla/util.h"

// TODO(tlongeri): Prefer returning failure over CHECKs. In particular, be more
// consistent about this for layout null checks in rules.

#define NYI(msg)                            \
  op->emitOpError("not implemented: " msg); \
  return failure();

namespace mlir::tpu {
// TODO(tlongeri): Maybe just roll our own multi-dimensional array instead of
// using XLA's? There's too much glue for going from/to ArrayRef.

#define GEN_PASS_DECL_APPLYVECTORLAYOUTPASS
#define GEN_PASS_DEF_APPLYVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

struct RewriteContext {
  func::FuncOp func;
  OpBuilder &builder;
  // TODO(tlongeri): target_shape should be determined from hardware_generation
  const int hardware_generation;
  const std::array<int64_t, 2> target_shape;
};

RollVectorsOp assemble(RewriteContext &ctx, VectorType vty,
                       const VectorLayout &layout, xla::Array<Value> vals);
FailureOr<xla::Array<Value>> disassemble(RewriteContext &ctx,
                                         const VectorLayout &layout, Value val);
namespace {

FailureOr<VectorType> getNativeVregType(
    Type elem_ty, const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const int8_t bitwidth,
                             getTypeBitwidth<true>(elem_ty));
  if (bitwidth == 32) {
    return VectorType::get(target_shape, elem_ty);
  }
  // bitwidth != 32
  return VectorType::get({target_shape[0], target_shape[1], 32 / bitwidth},
                         elem_ty);
}

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, ArrayRef<Layout>)>;

const llvm::StringMap<rule_type> &rules() {
  static auto rules = new llvm::StringMap<rule_type>{};
  return *rules;
}
}  // namespace

// Get the layout from a VectorLayoutAttr or StringAttr.
mlir::FailureOr<Layout> getLayoutFromAttr(Attribute attr) {
  if (attr == nullptr) {
    return failure();
  }

  if (auto layout_attr = dyn_cast<VectorLayoutAttr>(attr)) {
    return layout_attr.getLayout();
  }

  // TODO(tlongeri): StringAttr support was only added temporarily to avoid
  // having Python bindings for VectorLayoutAttr. Remove this once we get rid
  // of the Python implementation
  if (auto string_attr = dyn_cast<StringAttr>(attr)) {
    StringRef str = string_attr.getValue();
    if (!str.consume_front("#tpu.vpad<\"")) {
      return failure();
    }
    if (str.consume_front("none")) {
      return kNoLayout;
    }
    if (auto layout = VectorLayout::parse(&str)) {
      return layout;
    }
    return failure();
  }

  return failure();
}

// Returns empty vector on null attribute
FailureOr<SmallVector<Layout>> getLayoutArrayFromAttr(const Attribute attr) {
  if (const auto array_attr = dyn_cast_if_present<ArrayAttr>(attr)) {
    SmallVector<Layout> out_layouts;
    out_layouts.reserve(array_attr.size());
    for (const Attribute a : array_attr) {
      FAILUREOR_ASSIGN_OR_RETURN(const Layout layout, getLayoutFromAttr(a));
      out_layouts.push_back(layout);
    }
    return out_layouts;
  }
  return SmallVector<Layout>{};
}

// TODO(tlongeri): Unify with infer_vector_layout.cc's getOutLayout.
FailureOr<SmallVector<Layout>> getOutLayout(Operation &op) {
  // TODO(tlongeri): non-array attribute path should be removed after tests are
  // updated
  FailureOr<Layout> failure_or_layout =
      getLayoutFromAttr(op.getAttr("out_layout"));
  if (succeeded(failure_or_layout)) {
    return SmallVector<Layout>{failure_or_layout.value()};
  }
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> out_layout,
                             getLayoutArrayFromAttr(op.getAttr("out_layout")));
  if (out_layout.size() != op.getNumResults()) {
    return failure();
  }
  return out_layout;
}

FailureOr<SmallVector<Layout>> getInLayout(Operation &op) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> in_layout,
                             getLayoutArrayFromAttr(op.getAttr("in_layout")));
  if (in_layout.size() != op.getNumOperands()) {
    return failure();
  }
  return in_layout;
}

template <typename T>
ArrayRef<T> XlaArrayToFlatArrayRef(xla::Array<T> xla_array) {
  return ArrayRef<T>(xla_array.data(), xla_array.num_elements());
}

template <typename T, typename Range>
xla::Array<T> XlaArrayFromShapeAndValues(ArrayRef<int64_t> sizes, Range vals) {
  // TODO(tlongeri): is there no way to avoid default initialization in the
  // constructor?
  xla::Array<T> arr(sizes);
  arr.SetValues(vals);
  return arr;
}
RollVectorsOp assemble(RewriteContext &ctx, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> vals) {
  CHECK(vals.dimensions() ==
        layout.tileArrayShape(vty.getShape(), ctx.target_shape));
  CHECK_GT(vals.num_elements(), 0);
  Location loc = vals.begin()->getLoc();
  auto op =
      ctx.builder.create<RollVectorsOp>(loc, vty, XlaArrayToFlatArrayRef(vals));
  op->setAttr("out_layout",
              ctx.builder.getAttr<ArrayAttr>(ArrayRef<Attribute>{
                  ctx.builder.getAttr<VectorLayoutAttr>(layout)}));
  return op;
}

// Disassemble an MLIR vector into an ndarray of native vectors.
//
// Args:
//   layout: The layout of val. Used to determine the unrolling into
//     native-shaped vectors.
//   val: Value to disassemble. Must be of type VectorType.
//
// Returns:
//   An ndarray of MLIR values representing the tiling of val given by layout.
FailureOr<xla::Array<Value>> disassemble(RewriteContext &ctx,
                                         const VectorLayout &layout,
                                         const Value val) {
  const auto vty = cast<VectorType>(val.getType());
  const auto op_result = dyn_cast<OpResult>(val);
  if (op_result == nullptr) {
    return failure();
  }
  Operation *const op = op_result.getOwner();
  const unsigned res_idx = op_result.getResultNumber();
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                             getOutLayout(*op));
  const Layout def_layout = def_layouts[res_idx];
  CHECK(def_layout.has_value());
  CHECK(def_layout->equivalentTo(layout, std::nullopt, ctx.target_shape));
  SmallVector<int64_t> layout_shape =
      layout.tileArrayShape(vty.getShape(), ctx.target_shape);
  if (auto roll_vectors_op = dyn_cast<RollVectorsOp>(op)) {
    return XlaArrayFromShapeAndValues<Value>(layout_shape,
                                             roll_vectors_op->getOperands());
  }
  if (auto contraction_op = dyn_cast<vector::ContractionOp>(op)) {
    const int64_t num_vectors = ShapedType::getNumElements(layout_shape);
    FAILUREOR_ASSIGN_OR_RETURN(
        VectorType vreg_ty,
        getNativeVregType(vty.getElementType(), ctx.target_shape));
    // TODO(tlongeri): nicer way of doing ValueTypeRange?
    Operation *const u = ctx.builder.create<UnrollVectorsOp>(
        val.getLoc(), SmallVector<Type>(num_vectors, vreg_ty), val);
    return XlaArrayFromShapeAndValues<Value>(layout_shape, u->getResults());
  }
  return op->emitOpError("unimplemented: ") << val;
}

// Changes the layout of a vector value.
//
// Arguments:
//   v: The value to relayout. Must be of type VectorType.
//   src: The current layout of v.
//   dst: The target layout of v.
//
// Returns:
//   A new MLIR vector value, laid out as requested by dst.
// TODO(apaszke): Test this function properly
FailureOr<Value> relayout(RewriteContext &ctx, Value v, VectorLayout src,
                          const VectorLayout &dst) {
  const int8_t bitwidth = src.bitwidth();
  if (bitwidth != dst.bitwidth()) {
    return emitError(v.getLoc(), "Can't change bitwidth during a relayout");
  }
  const int packing = src.packing();
  VectorType vty = cast<VectorType>(v.getType());
  FAILUREOR_ASSIGN_OR_RETURN(xla::Array<Value> src_tiles,
                             disassemble(ctx, src, v));
  SmallVector<int64_t> dst_tiles_shape =
      dst.tileArrayShape(vty.getShape(), ctx.target_shape);
  if (src.generalizes(dst, vty.getShape(), ctx.target_shape) &&
      src.tilesPerVreg(ctx.target_shape) == 1) {
    return assemble(ctx, vty, dst, std::move(src_tiles)).getResult();
  }
  if (!src.offsets()[0].has_value() && !src.offsets()[1].has_value()) {
    // A fully replicated value is always easy to relayout
    // It would be nice to be able to assert this here, but given replicated
    // values our rules can introduce equivalent expressions.
    // assert all(t is src_tiles_list[0] for t in src_tiles_list)
    xla::Array<Value> dst_tiles(
        /*sizes=*/dst.tileArrayShape(vty.getShape(), ctx.target_shape),
        /*value=*/src_tiles.data()[0]);
    return assemble(ctx, vty, dst, std::move(dst_tiles)).getResult();
  }
  // Try to reconcile differences in implicit dim.
  if (src.implicit_dim() != dst.implicit_dim()) {
    const ArrayRef<int64_t> shape = vty.getShape();
    if (dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
        shape[shape.size() - xla::to_underlying(src.implicit_dim())] == 1) {
      src = VectorLayout(src.bitwidth(), src.offsets(), src.tiling(),
                         VectorLayout::ImplicitDim::kNone);
    }
  }

  // TODO(b/265133506): Generalize retiling to general 16-bit types (might need
  // to use a different unpacking op).
  VectorType vreg_f32 =
      VectorType::get(ctx.target_shape, ctx.builder.getF32Type());
  // (8,128) -> (16,128) tiling change for packed 16-bit types.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      vty.getElementType() == ctx.builder.getBF16Type() &&
      src.offsets() == dst.offsets() &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{16, 128}) {
    const VectorLayout new_src(src.bitwidth(), src.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), ctx.target_shape));
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src_idx[src_idx.size() - 2] *= 2;
      src_idx[src_idx.size() - 1] /= 2;
      Value src_row1 = src_tiles(src_idx);
      if (src_idx[src_idx.size() - 2] + 1 <
          src_tiles.dim(src_tiles.num_dimensions() - 2)) {
        ++src_idx[src_idx.size() - 2];
      }
      Value src_row2 = src_tiles(src_idx);
      const int vreg_part = idx[idx.size() - 1] % 2;
      auto half_row1 = ctx.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row1, vreg_part);
      auto half_row2 = ctx.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row2, vreg_part);
      *tile = ctx.builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_row1.getType(), ValueRange{half_row1, half_row2});
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  }
  // (16, 128) -> (8, 128) tiling change for packed 16-bit types
  if (src.implicit_dim() != VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() != VectorLayout::ImplicitDim::kNone &&
      vty.getElementType() == ctx.builder.getBF16Type() &&
      src.tiling() == std::array<int64_t, 2>{16, 128} &&
      dst.tiling() == std::array<int64_t, 2>{8, 128}) {
    const VectorLayout new_src(src.bitwidth(), src.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), ctx.target_shape));
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src_idx[src_idx.size() - 2] /= 2;
      src_idx[src_idx.size() - 1] *= 2;
      Value src_row1 = src_tiles(src_idx);
      if (src_idx[src_idx.size() - 1] + 1 <
          src_tiles.dim(src_tiles.num_dimensions() - 1)) {
        ++src_idx[src_idx.size() - 1];
      }
      Value src_row2 = src_tiles(src_idx);
      const int vreg_part = idx[idx.size() - 1] % 2;
      auto half_row1 = ctx.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row1, vreg_part);
      auto half_row2 = ctx.builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row2, vreg_part);
      *tile = ctx.builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_row1.getType(), ValueRange{half_row1, half_row2});
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  }
  // Handle retiling from (8, 128) to (1, 128) for 32 bits data.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      getTypeBitwidth(vty.getElementType()) == 32 &&
      (!src.offsets()[0].has_value() || *src.offsets()[0] == 0) &&
      (!src.offsets()[1].has_value() || *src.offsets()[1] == 0) &&
      dst.offsets() == LayoutOffsets{0, 0} &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{1, 128}) {
    const VectorLayout new_src(src.bitwidth(), dst.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), ctx.target_shape));
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      const int64_t dst_row = *(idx.end() - 2);
      const int64_t dst_col = *(idx.end() - 1);
      const int64_t src_row = dst_row / 8;
      const int64_t src_col_0 = dst_col * 8;
      const int64_t src_tile_vreg_count =
          src_col_0 + 8 <= src_tiles.dimensions().back()
              ? 8
              : src_tiles.dimensions().back() % 8;
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      *(src_idx.end() - 2) = src_row;
      *(src_idx.end() - 1) = src_col_0;
      *tile = src_tiles(src_idx);
      for (int64_t i = 1; i < src_tile_vreg_count; ++i) {
        const RectangularVregBounds bounds({i, 0},
                                           {i + 1, ctx.target_shape[1]});

        *(src_idx.end() - 1) = src_col_0 + i;
        Value src_tile = src_tiles(src_idx);
        src_tile = ctx.builder.create<tpu::RotateOp>(
            src_tile.getLoc(), src_tile, /*amount=*/i, /*dimension=*/0);
        const TypedValue<VectorType> mask =
            bounds
                .getVectorMask(ctx.builder, ctx.hardware_generation,
                               ctx.target_shape)
                .value();
        *tile = ctx.builder.create<arith::SelectOp>(mask.getLoc(), mask,
                                                    src_tile, *tile);
      }
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  }
  // TODO(kumudbhandari): Generalize the logic below to handle retiling from
  // (8, 128) to (x, 128) where x=1, 2 or 4.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src.offsets() == dst.offsets() && src.bitwidth() != 16 &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{4, 128}) {
    const VectorLayout retiled_src_layout(src.bitwidth(), src.offsets(),
                                          dst.tiling());
    xla::Array<Value> retiled_src_tiles(
        retiled_src_layout.tileArrayShape(vty.getShape(), ctx.target_shape));

    // Consider a value of type and shape: f32(8, 256). Retiling from (8,128)
    // to (4,128): vreg (tile) array shape (1, 2), with original (8,128)
    // tiling:
    //   vreg_0_0: slice:[0:7, 0:127] vreg_0_1: slice:[0:7, 128:255]

    // vreg (tile) array shape: (2, 1), with (4,128) retiling:
    //   vreg_0_0: slice: [0:3, 0:127], slice: [0:3, 128:255]
    //   vreg_1_0: slice:[4:7, 0:127], slice: [4:7, 128:255]
    const int64_t nd = retiled_src_tiles.num_dimensions();
    retiled_src_tiles.Each(
        [&](absl::Span<const int64_t> retiled_idx, Value *tile) {
          SmallVector<int64_t> src_tile_idx(toArrayRef(retiled_idx));

          // The first src tile, half of which forms the first half of the
          // retiled tile(retiled_row_idx, retiled_col_idx).
          src_tile_idx[nd - 2] /= 2;
          src_tile_idx[nd - 1] *= 2;
          const Value src_tile_1 = src_tiles(src_tile_idx);

          // The second src tile, half of which forms the second half of the
          // retiled tile(retiled_row_idx, retiled_col_idx).
          if (src_tile_idx[nd - 1] + 1 < src_tiles.dim(nd - 1)) {
            // TODO(tlongeri): is this just when the second tile is invalid? Can
            // we just set src_tile_2 to nullptr and not merge it in this
            // situation?
            ++src_tile_idx[nd - 1];
          }
          const Value src_tile_2 = src_tiles(src_tile_idx);

          // Each (retiled_row_idx)th tile is formed from 2 top or 2 bottom half
          // sublanes of the original tile.
          // We need to rotate sublanes of one of the two tiles to push either a
          // top half to the bottom or vice-versa.
          const Value tile_to_merge_1 =
              retiled_idx[nd - 2] % 2 == 0
                  ? src_tile_1
                  : ctx.builder.create<tpu::RotateOp>(
                        v.getLoc(), src_tile_1, /*amount=*/4, /*dimension=*/0);
          const Value tile_to_merge_2 =
              retiled_idx[nd - 2] % 2 != 0
                  ? src_tile_2
                  : ctx.builder.create<tpu::RotateOp>(
                        v.getLoc(), src_tile_2, /*amount=*/4, /*dimension=*/0);
          // Create a mask to select first half from tile 1 and second half of
          // data from tile 2 to be merged.
          const RectangularVregBounds vreg_half_bound(
              {0, 0}, {ctx.target_shape[0] / 2, ctx.target_shape[1]});
          const Value vreg_select_mask =
              vreg_half_bound
                  .getVectorMask(ctx.builder, ctx.hardware_generation,
                                 ctx.target_shape)
                  .value();
          *tile = ctx.builder.create<arith::SelectOp>(
              v.getLoc(), vreg_select_mask, tile_to_merge_1, tile_to_merge_2);
        });
    src = retiled_src_layout;
    src_tiles = std::move(retiled_src_tiles);
  }

  // Fix up the offsets, assuming everything else matches between src and dst.
  if (src.tiling() == dst.tiling() &&
      src.implicit_dim() == dst.implicit_dim()) {
    // TODO(apaszke): Changing an offset might add or remove one vreg.
    if (dst_tiles_shape != src_tiles.dimensions()) {
      return emitError(v.getLoc(), "Offsets changing the vreg array shape");
    }
    xla::Array<Value> dst_tiles = src_tiles;

    // Shifting rows
    int row_diff;
    if (!src.offsets()[0].has_value()) {
      row_diff = 0;
    } else if (!dst.offsets()[0].has_value()) {
      return emitError(v.getLoc(), "Sublane broadcast not implemented");
    } else {
      row_diff = *dst.offsets()[0] - *src.offsets()[0];
    }

    if (row_diff != 0) {  // This is an easy case, because we never need to
                          // combine multiple vregs.
      const SmallVector<int64_t> implicit_shape =
          src.implicitShape(vty.getShape());
      if (implicit_shape[implicit_shape.size() - 2] != 1) {
        return emitError(v.getLoc(), "Row shifts for multi-row values");
      }
      const int64_t src_sublane = *src.offsets()[0] / packing;
      const int64_t dst_sublane = *dst.offsets()[0] / packing;
      if (int64_t sublane_diff = dst_sublane - src_sublane) {
        if (sublane_diff < 0) {
          sublane_diff += ctx.target_shape[0];
        }
        src_tiles.Each([&](absl::Span<const int64_t> idx, Value tile) {
          dst_tiles(idx) = ctx.builder
                               .create<tpu::RotateOp>(v.getLoc(), tile,
                                                      /*amount=*/sublane_diff,
                                                      /*dimension=*/0)
                               .getResult();
        });
      }
      const int src_subelem = src_sublane % packing;
      const int dst_subelem = dst_sublane % packing;
      if (src_subelem != dst_subelem) {
        const int subelem_diff = dst_subelem - src_subelem;
        const int shift_bits = bitwidth * std::abs(subelem_diff);
        VectorType bits_vreg_ty =
            VectorType::get(ctx.target_shape, ctx.builder.getI32Type());
        auto shift_vreg = ctx.builder.create<arith::ConstantOp>(
            v.getLoc(), bits_vreg_ty,
            DenseElementsAttr::get(bits_vreg_ty, shift_bits));
        dst_tiles.Each([&](absl::Span<const int64_t> /*idx*/, Value *tile) {
          auto bit_tile = ctx.builder.create<tpu::BitcastOp>(
              v.getLoc(), bits_vreg_ty, *tile);
          Operation *shift_tile;
          if (subelem_diff > 0) {
            shift_tile = ctx.builder.create<arith::ShLIOp>(v.getLoc(), bit_tile,
                                                           shift_vreg);
          } else {  // subelem_diff < 0
            CHECK_LT(subelem_diff, 0);
            shift_tile = ctx.builder.create<arith::ShRUIOp>(
                v.getLoc(), bit_tile, shift_vreg);
          }
          *tile = shift_tile->getResult(0);
          return absl::OkStatus();
        });
      }
    }
    int64_t col_diff;
    if (!src.offsets()[1].has_value()) {
      col_diff = 0;
    } else if (!dst.offsets()[1].has_value()) {
      return emitError(v.getLoc(), "Not implemented: Lane broadcast");
    } else {
      col_diff = *dst.offsets()[1] - *src.offsets()[1];
    }
    if (col_diff != 0) {
      if (row_diff != 0) {
        return emitError(v.getLoc(),
                         "Not implemented: Both columns and rows are shifted");
      }
      if (col_diff < 0) {
        return emitError(v.getLoc(), "Not implemented: Shifts to the left");
      }
      if (bitwidth != 32) {
        return emitError(
            v.getLoc(), "Not implemented: Only 32-bit column shifts supported");
      }
      const int64_t sublane_diff = col_diff;
      CHECK_GE(src_tiles.num_dimensions(), 1);
      std::optional<tpu::CreateMaskOp> maybe_create_mask;
      if (src_tiles.dimensions()[src_tiles.num_dimensions() - 1] > 1) {
        auto boundIdxConst =
            std::bind(IdxConst, std::placeholders::_1, ctx.builder, v.getLoc());
        maybe_create_mask = ctx.builder.create<tpu::CreateMaskOp>(
            v.getLoc(),
            VectorType::get(ctx.target_shape, ctx.builder.getI1Type()),
            ValueRange{boundIdxConst(0), boundIdxConst(0)},
            ValueRange{boundIdxConst(ctx.target_shape[0]),
                       boundIdxConst(col_diff)});
      }
      src_tiles.Each([&](absl::Span<const int64_t> idx, Value tile) {
        Value rot_tile = ctx.builder
                             .create<tpu::RotateOp>(v.getLoc(), tile,
                                                    /*amount=*/sublane_diff,
                                                    /*dimension=*/1)
                             .getResult();
        if (idx[idx.size() - 1] != 0) {
          SmallVector<int64_t> prev_idx(idx.begin(), idx.end());
          --prev_idx[idx.size() - 1];
          Value prev_rot_tile = dst_tiles(prev_idx);
          rot_tile = ctx.builder.create<arith::SelectOp>(
              v.getLoc(), maybe_create_mask->getResult(), prev_rot_tile, tile);
        }
        dst_tiles(idx) = rot_tile;
      });
    }
    return assemble(ctx, vty, dst, std::move(dst_tiles)).getResult();
  }
  // TODO(apaszke): Implement general relayout
  return emitError(v.getLoc(), "unsupported layout change for ")
         << vty << ": " << src << " -> " << dst;
}

// Rewrites the operation according to its layout annotations.
//
// Args:
//   ctx: The context used for rewriting.
//   op: An MLIR operation to be rewritten.
//
// A valid op is expected to have a layout_in attribute unless it has no
// operands. The layout_in attribute must fulfill the following:
//   - All vector operands originate from an operation (not a BlockArgument)
//   and
//     have a valid layout (Layout1D or Layout2D)
//   - All non-vector operands must have NoLayout.
// TODO(apaszke): Implement a debug mode that inserts additional assertions.
// For example, we should verify that ops that were supposed to generate
// replicated outputs satisfy that requirement.
LogicalResult applyLayoutOp(RewriteContext &ctx, Operation &op) {
  ctx.builder.setInsertionPointAfter(&op);
  // TODO(tlongeri): Once we support all ops, return failure instead.
  if (!rules().contains(op.getName().getStringRef())) {
    return success();
  }

  // When an operation does not have any operands, the layout_in tuple is empty.
  // If one of the operands is not of vector type, the corresponding entry in
  // the layout_in tuple will be None. The same applies to the results of the
  // operation and the layout_out tuple.
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layout_out,
                             getOutLayout(op));
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layout_in,
                             getInLayout(op));
  if (!layout_in.empty()) {
    // Relayout the operands, if their requested input layouts don't match the
    // layouts in which they were produced.
    for (auto [idx, tup] :
         llvm::enumerate(llvm::zip(op.getOperands(), layout_in))) {
      auto [operand, li] = tup;
      auto vty = dyn_cast<VectorType>(operand.getType());
      if ((vty == nullptr) == li.has_value()) {
        return op.emitError(
            "layout should be none iff operand is not a vector");
      }
      if (vty == nullptr) {
        continue;
      }

      // The operand should always be an Operation (and not a BlockArgument)
      // since we expect the FuncOp to have only memrefs and semaphores as
      // arguments.
      auto op_result = dyn_cast<OpResult>(operand);
      if (op_result == nullptr) {
        return op.emitError("expected operand to be an operation result");
      }
      Operation *const def_op = op_result.getOwner();
      CHECK(def_op);
      const unsigned res_idx = op_result.getResultNumber();
      FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                                 getOutLayout(*def_op));
      const Layout lo = def_layouts[res_idx];
      if (!lo.has_value()) {
        return op.emitError() << "vector result should have a defined layout";
      }
      if (lo->generalizes(*li, vty.getShape(), ctx.target_shape)) {
        continue;
      }
      FAILUREOR_ASSIGN_OR_RETURN(Value new_v,
                                 relayout(ctx, operand, /*src=*/*lo,
                                          /*dst=*/*li));
      op.setOperand(idx, new_v);
    }
  }

  const bool no_vector_args =
      llvm::none_of(layout_out,
                    [](Layout layout) { return layout.has_value(); }) &&
      llvm::none_of(layout_in,
                    [](Layout layout) { return layout.has_value(); });
  if (no_vector_args && op.getRegions().empty()) {
    // We don't need to do anything for scalar operations.
    if (!op.getOperands().empty()) {
      op.removeAttr("in_layout");
    }
    if (!op.getResults().empty()) {
      op.removeAttr("out_layout");
    }
    return success();
  }
  if (auto rule_it = rules().find(op.getName().getStringRef());
      rule_it != rules().end()) {
    const rule_type &rule = rule_it->getValue();
    return rule(ctx, op, layout_in, layout_out);
  }
  return op.emitError("Unsupported operation: ") << op.getName();
}

LogicalResult applyLayoutBlock(RewriteContext &ctx, Block &block) {
  // We'll be modifying the block, so use early increment.
  for (Operation &op : make_early_inc_range(block)) {
    if (failed(applyLayoutOp(ctx, op))) {
      return failure();
    }
  }
  return success();
}

// Rewrites the function according to layout annotations of its operations.
//
//   Args:
//     ctx: The context used for rewriting.
//     f: An MLIR function to be rewritten.
LogicalResult applyLayoutFunc(RewriteContext &ctx, func::FuncOp f) {
  if (f->getNumRegions() != 1) {
    return f.emitError("Expected FuncOp to have a single region");
  }
  if (!f.getBody().hasOneBlock()) {
    return f.emitError("Expected FuncOp to have a single block");
  }
  return applyLayoutBlock(ctx, f.getBody().front());
}

struct ApplyVectorLayoutPass
    : public impl::ApplyVectorLayoutPassBase<ApplyVectorLayoutPass> {
  ApplyVectorLayoutPass(int hardware_generation_, int lane_count_,
                        int sublane_count_) {
    hardware_generation = hardware_generation_;
    sublane_count = sublane_count_;
    lane_count = lane_count_;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getBody());
    RewriteContext ctx{
        func, builder, hardware_generation, {sublane_count, lane_count}};
    if (failed(applyLayoutFunc(ctx, func))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    int hardware_generation, int lane_count, int sublane_count) {
  return std::make_unique<ApplyVectorLayoutPass>(hardware_generation,
                                                 lane_count, sublane_count);
}

}  // namespace mlir::tpu
