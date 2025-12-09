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

#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep.
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep.
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.cc.inc"
#include "jaxlib/mosaic/dialect/tpu/tpu_enums.cc.inc"
#include "xla/layout.h"

// This is a bit unclean, but we need to squat the xla namespace to make sure
// that this overload is found via argument-dependent lookup.
namespace xla {

llvm::hash_code hash_value(const ::xla::Tile &p) { return absl::HashOf(p); }

}  // namespace xla

#define GET_ATTRDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_attr_defs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_type_defs.cc.inc"

namespace mlir::tpu {

void TPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "jaxlib/mosaic/dialect/tpu/tpu_attr_defs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "jaxlib/mosaic/dialect/tpu/tpu_type_defs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "jaxlib/mosaic/dialect/tpu/tpu_ops.cc.inc"
      >();
}

Operation *TPUDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

/* static */ std::optional<CoreType> TPUDialect::GetCoreTypeAttr(
    Operation *op) {
  Attribute attr = op->getAttr(GetCoreTypeKey());
  if (attr == nullptr) {
    return std::nullopt;
  }
  if (!mlir::isa<CoreTypeAttr>(attr)) {
    return std::nullopt;
  }
  return mlir::cast<CoreTypeAttr>(attr).getValue();
}

struct MemRefCastEraseLayout : public OpRewritePattern<memref::CastOp> {
  // Set the benefit to 0 to ensure that other patterns that fold in the cast
  // are tried first.
  MemRefCastEraseLayout(MLIRContext* context)
      : OpRewritePattern<memref::CastOp>(context, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(memref::CastOp cast_op,
                                PatternRewriter& rewriter) const final {
    // Push tpu.erase_memref_layout through memref.cast
    auto erase_layout_op = cast_op.getOperand().getDefiningOp<EraseLayoutOp>();
    if (!erase_layout_op) {
      return failure();
    }
    TypedValue<MemRefType> orig_value = erase_layout_op.getOperand();
    const MemRefType orig_type = orig_value.getType();
    const ArrayRef<int64_t> cast_shape = cast_op.getType().getShape();
    MemRefType new_cast_type =
        MemRefType::Builder(orig_type).setShape(cast_shape);
    auto new_cast_op = memref::CastOp::create(rewriter, cast_op.getLoc(),
                                              new_cast_type, orig_value);
    auto new_erase_layout_op =
        EraseLayoutOp::create(rewriter, erase_layout_op.getLoc(), new_cast_op);
    rewriter.replaceOp(cast_op, new_erase_layout_op);
    return success();
  }
};

void TPUDialect::getCanonicalizationPatterns(RewritePatternSet& results) const
/*override*/ {
  results.add<MemRefCastEraseLayout>(getContext());
}

FailureOr<CoreType> GetCoreTypeOfParentFunc(Operation &op) {
  mlir::Operation *func_op = op.getParentOfType<mlir::func::FuncOp>();
  if (func_op == nullptr) {
    return op.emitError() << "Operation " << op.getName()
                          << " is not inside a func.func";
  }
  return TPUDialect::GetCoreTypeAttr(func_op).value_or(CoreType::kTc);
}

void VectorLayoutAttr::print(AsmPrinter &printer) const {
  printer << '<';
  printer << getLayout();
  printer << '>';
}

Attribute VectorLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  if (auto layout = parseLayout(parser);
      layout.has_value() && succeeded(parser.parseGreater())) {
    return get(parser.getContext(), *layout);
  }
  return {};
}

void TiledLayoutAttr::print(AsmPrinter &printer) const {
  printer << '<';
  for (const xla::Tile &tile : getTiles()) {
    printer << tile.ToString();
  }
  printer << ",[";
  for (int i = 0; i < getTileStrides().size(); ++i) {
    if (i > 0) {
      printer << ',';
    }
    printer << getTileStrides()[i];
  }
  printer << "]>";
}

Attribute TiledLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  SmallVector<xla::Tile, 2> tiles;
  int64_t size;
  while (succeeded(parser.parseOptionalLParen())) {
    xla::Tile &tile = tiles.emplace_back();
    bool first = true;
    while (!succeeded(parser.parseOptionalRParen())) {
      if (!first) {
        if (failed(parser.parseComma())) {
          return {};
        }
      }
      first = false;
      if (failed(parser.parseInteger(size))) {
        return {};
      }
      tile.add_dimensions(size);
    }
  }
  SmallVector<int64_t, 2> tile_strides;
  int64_t stride;
  if (failed(parser.parseComma())) {
    return {};
  }
  if (succeeded(parser.parseOptionalLSquare())) {
    bool first = true;
    while (!succeeded(parser.parseOptionalRSquare())) {
      if (!first) {
        if (failed(parser.parseComma())) {
          return {};
        }
      }
      first = false;
      if (failed(parser.parseInteger(stride))) {
        return {};
      }
      tile_strides.push_back(stride);
    }
  } else {
    return {};
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return get(parser.getContext(), tiles, tile_strides);
}

AffineMap TiledLayoutAttr::getAffineMap() const {
  SmallVector<AffineExpr, 8> exprs;
  for (int64_t i = 0; i < getRank(); ++i) {
    exprs.push_back(getAffineDimExpr(i, getContext()));
  }
  for (const xla::Tile& tile : getTiles()) {
    SmallVector<AffineExpr, 8> new_exprs;
    auto dimensions = tile.dimensions();
    int64_t untiled_rank = exprs.size() - dimensions.size();
    assert(untiled_rank >= 0);
    for (int64_t i = 0; i < untiled_rank; ++i) {
      new_exprs.push_back(exprs[i]);
    }
    for (int64_t i = 0; i < dimensions.size(); ++i) {
      new_exprs.push_back(exprs[untiled_rank + i].floorDiv(dimensions[i]));
    }
    for (int64_t i = 0; i < dimensions.size(); ++i) {
      new_exprs.push_back(exprs[untiled_rank + i] % dimensions[i]);
    }
    exprs = std::move(new_exprs);
  }
  int64_t num_symbols = 0;
  AffineExpr result = getAffineConstantExpr(0, getContext());
  SmallVector<int64_t> strides = getExpandedStrides();
  assert(strides.size() == exprs.size());
  for (int64_t i = 0; i < exprs.size(); ++i) {
    AffineExpr stride_expr =
        ShapedType::isDynamic(strides[i])
            ? getAffineSymbolExpr(num_symbols++, getContext())
            : getAffineConstantExpr(strides[i], getContext());
    result = result + exprs[i] * stride_expr;
  }
  return AffineMap::get(getRank(), num_symbols, result);
}

namespace {
int64_t getUntiledRank(ArrayRef<xla::Tile> tiles, const int64_t rank) {
  // Note: This implementation does not assume there is no nested tiling across
  // the first level of tiling, though this is enforced by the verifier.
  int64_t untiled_rank = rank;
  int64_t tiled_rank = rank;
  for (const xla::Tile& tile : tiles) {
    const int64_t tile_ndims = tile.dimensions().size();
    untiled_rank = std::min(untiled_rank, tiled_rank - tile_ndims);
    tiled_rank += tile_ndims;
  }
  return untiled_rank;
}
}  // namespace

int64_t TiledLayoutAttr::getUntiledRank() const {
  return mlir::tpu::getUntiledRank(getTiles(), getRank());
}

namespace {
FailureOr<SmallVector<int64_t>> getExpandedShape(
    const ArrayRef<int64_t> untiled_shape, const ArrayRef<xla::Tile> tiles,
    const bool require_alignment) {
  SmallVector<int64_t> shape(untiled_shape);
  for (const xla::Tile& tile : tiles) {
    const int64_t tile_ndims = tile.dimensions().size();
    const llvm::ArrayRef<int64_t> tiled_shape =
        llvm::ArrayRef(shape).take_back(tile_ndims);
    llvm::SmallVector<int64_t> new_tiled_shape(2 * tile_ndims);
    for (int64_t i = 0; i < tile_ndims; ++i) {
      if (require_alignment && (ShapedType::isDynamic(tiled_shape[i]) ||
                                tiled_shape[i] % tile.dimension(i) != 0)) {
        return failure();
      }
      if (ShapedType::isDynamic(tiled_shape[i])) {
        new_tiled_shape[i] = ShapedType::kDynamic;
      } else {
        new_tiled_shape[i] =
            llvm::divideCeil(tiled_shape[i], tile.dimension(i));
      }
      new_tiled_shape[tile_ndims + i] = tile.dimension(i);
    }
    shape.pop_back_n(tile_ndims);
    shape.append(new_tiled_shape);
  }
  return shape;
}
}  // namespace

SmallVector<int64_t> TiledLayoutAttr::getDefaultTileStrides(
    const ArrayRef<xla::Tile> tiles, const ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size());
  int64_t stride = 1;
  const xla::Tile* const first_tile = tiles.empty() ? nullptr : &tiles.front();
  const int64_t first_tile_rank =
      first_tile == nullptr ? 0 : first_tile->dimensions().size();
  for (int64_t d = shape.size() - 1; d >= 0; --d) {
    assert(!ShapedType::isDynamic(shape[d]));
    strides[d] = stride;
    if (d >= shape.size() - first_tile_rank) {
      assert(first_tile != nullptr);
      const int64_t tile_d = d - (shape.size() - first_tile_rank);
      stride *= llvm::divideCeil(shape[d], first_tile->dimension(tile_d));
    } else {
      stride *= shape[d];
    }
  }
  return strides;
}

bool TiledLayoutAttr::tilesAreKnownContiguous(
    const ArrayRef<int64_t> shape) const {
  const ArrayRef<xla::Tile> tiles = getTiles();
  const ArrayRef<int64_t> tile_strides = getTileStrides();
  int64_t stride = 1;
  const xla::Tile* const first_tile = tiles.empty() ? nullptr : &tiles.front();
  const int64_t first_tile_rank =
      first_tile == nullptr ? 0 : first_tile->dimensions().size();
  for (int64_t d = shape.size() - 1; d >= 0; --d) {
    int64_t size_tiles;
    if (d >= shape.size() - first_tile_rank &&
        shape[d] != ShapedType::kDynamic) {
      assert(first_tile != nullptr);
      const int64_t tile_d = d - (shape.size() - first_tile_rank);
      size_tiles = llvm::divideCeil(shape[d], first_tile->dimension(tile_d));
    } else {
      size_tiles = shape[d];
    }
    // Dimensions with only one element/tile can have any stride.
    if (stride != tile_strides[d] && size_tiles != 1) {
      return false;
    }
    if (d == 0) {
      break;
    }
    // When any dimension other than the leading one has a dynamic size, we
    // cannot guarantee that there are no gaps.
    if (size_tiles == ShapedType::kDynamic) {
      return false;
    }
    stride *= size_tiles;
  }
  return true;
}

SmallVector<int64_t> TiledLayoutAttr::getExpandedShape(
    ArrayRef<int64_t> untiled_shape) const {
  // getExpandedShape should never fail without require_alignment
  return *mlir::tpu::getExpandedShape(untiled_shape, getTiles(),
                                      /*require_alignment=*/false);
}

SmallVector<int64_t> TiledLayoutAttr::getExpandedStrides() const {
  if (getTiles().empty()) {
    return SmallVector<int64_t>(getTileStrides());
  }
  SmallVector<int64_t> strides(getTileStrides());
  // Expand front tile
  const xla::Tile& first_tile = getTiles().front();
  const FailureOr<SmallVector<int64_t>> failure_or_expanded_tile =
      mlir::tpu::getExpandedShape(first_tile.dimensions(),
                                  getTiles().drop_front(),
                                  /*require_alignment=*/true);
  // Verification should ensure this:
  assert(succeeded(failure_or_expanded_tile));
  const SmallVector<int64_t>& expanded_tile = *failure_or_expanded_tile;
  strides.resize_for_overwrite(getRank() + expanded_tile.size());
  int64_t first_tile_size = llvm::product_of(first_tile.dimensions());
  int64_t tile_size = 1;
  for (int64_t d = strides.size() - 1; d >= 0; --d) {
    if (d >= getRank()) {
      const int64_t new_stride = tile_size;
      tile_size *= expanded_tile[d - getRank()];
      strides[d] = new_stride;
    } else {
      strides[d] *= first_tile_size;
    }
  }
  return strides;
}

LogicalResult TiledLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    const llvm::ArrayRef<xla::Tile> tiles,
    const llvm::ArrayRef<int64_t> tile_strides) {
  if (llvm::any_of(tile_strides, ShapedType::isDynamic)) {
    return emitError() << "Not implemented: Dynamic tile strides";
  }
  if (tiles.empty()) {
    return success();
  }
  const int64_t rank = tile_strides.size();
  const xla::Tile& first_tile = tiles.front();
  const int64_t first_tile_rank = first_tile.dimensions().size();
  // The interpretation of tile strides is unclear if there is nested tiling
  // across first tiles (e.g. T(8, 128)(2, 4, 64)), and this has no applications
  // anyway.
  if (mlir::tpu::getUntiledRank(tiles, rank) != rank - first_tile_rank) {
    return emitError() << "Not implemented: Nested tiling across first tiles";
  }
  // Check that nested tiles evenly divide previous tiles (so they don't add any
  // padding or change the tile size)
  if (failed(mlir::tpu::getExpandedShape(first_tile.dimensions(),
                                         tiles.drop_front(),
                                         /*require_alignment=*/true))) {
    return emitError() << "Not implemented: Nested tiles must evenly divide "
                       << "the first tile " << first_tile.ToString()
                       << " but they do not (would add padding)";
  }
  return success();
}

MemRefType getMemRefType(Value value) {
  if (auto erase_op = value.getDefiningOp<tpu::EraseLayoutOp>()) {
    value = erase_op.getOperand();
  }
  return cast<MemRefType>(value.getType());
}

bool isGuaranteedDivisible(Value value, int64_t divisor, int64_t fuel) {
  if (fuel <= 0) {
    return false;
  }
  if (divisor == 1) {
    return true;
  }
  if (auto assume_op = value.getDefiningOp<tpu::AssumeMultipleOp>()) {
    return assume_op.getMultiple() % divisor == 0;
  }
  if (auto mul_op = value.getDefiningOp<arith::MulIOp>()) {
    // We check RHS first, because MLIR canonicalizes constants to the right.
    return isGuaranteedDivisible(mul_op.getRhs(), divisor, fuel / 2) ||
           isGuaranteedDivisible(mul_op.getLhs(), divisor, (fuel + 1) / 2);
  }
  if (auto cst_op = value.getDefiningOp<arith::ConstantOp>()) {
    auto int_attr = dyn_cast<IntegerAttr>(cst_op.getValue());
    return int_attr && int_attr.getInt() % divisor == 0;
  }
  if (auto cast_op = value.getDefiningOp<arith::IndexCastOp>()) {
    return isGuaranteedDivisible(cast_op.getOperand(), divisor, fuel - 1);
  }
  if (auto add_op = value.getDefiningOp<arith::AddIOp>()) {
    return isGuaranteedDivisible(add_op.getRhs(), divisor, fuel / 2) &&
           isGuaranteedDivisible(add_op.getLhs(), divisor, (fuel + 1) / 2);
  }
  return false;
}

DotDimensionNumbersAttr defaultDimensionNumbers(Builder &builder,
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

const ::llvm::fltSemantics& Float8EXMYType::getFloatSemantics() const {
  if (mlir::isa<Float6E3M2FNType>(getUnderlyingType())) {
    return llvm::APFloat::Float6E3M2FN();
  } else if (mlir::isa<Float6E2M3FNType>(getUnderlyingType())) {
    return llvm::APFloat::Float6E2M3FN();
  }
  return cast<FloatType>(getUnderlyingType()).getFloatSemantics();
}

namespace {

struct CommsAnalysisState {
  bool has_communication = false;
  bool has_custom_barrier = false;

  explicit operator bool() { return has_communication && has_custom_barrier; }
};

void analyzeCrossChipCommunication(mlir::Operation *op,
                                   CommsAnalysisState *state) {
  if (auto dma = dyn_cast<tpu::EnqueueDMAOp>(op)) {
    state->has_communication |= dma.getDeviceId() != nullptr;
  } else if (auto signal = dyn_cast<tpu::SemaphoreSignalOp>(op)) {
    state->has_communication |= signal.getDeviceId() != nullptr;
  } else if (auto barrier = dyn_cast<tpu::GetBarrierSemaphoreOp>(op)) {
    state->has_custom_barrier = true;
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        analyzeCrossChipCommunication(&op, state);
        if (*state) {
          return;
        }
      }
    }
  }
}

}  // namespace

std::pair<bool, bool> mightCommunicateBetweenChips(mlir::Operation *op) {
  CommsAnalysisState state;
  analyzeCrossChipCommunication(op, &state);
  return std::make_pair(state.has_communication, state.has_custom_barrier);
}

}  // namespace mlir::tpu
