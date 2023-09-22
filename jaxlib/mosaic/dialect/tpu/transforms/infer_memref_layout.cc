#include "jaxlib/mosaic/dialect/tpu/transforms/infer_memref_layout.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"
#include "xla/layout.h"

namespace mlir::tpu {

// Returns the number of 128-element groups in a tile.
//
// Arguments:
//   num_128s: A number of 128-element groups in the full operand.
//   hardware_generation: An integer indicating the target TPU generation.
//   bitwidth: The bitwidth of the element type of the operand.
int getTilingFactor(const int num_128s, const int hardware_generation,
                    const int8_t bitwidth) {
  CHECK(llvm::isPowerOf2_32(bitwidth));
  CHECK_LE(4, bitwidth);
  CHECK_LE(bitwidth, 32);
  const int packing = 32 / bitwidth;
  const int min_tiling = (1 + (hardware_generation < 4)) * packing;
  const int max_tiling = 8;
  int tiling = min_tiling;
  while (tiling < std::min(num_128s, max_tiling)) {
    tiling *= 2;
  }
  return tiling;
}

FailureOr<TiledLayoutAttr> inferLayout(MemRefType memref,
                                       const int hardware_generation) {
  if (auto tiled_layout_attr = dyn_cast<TiledLayoutAttr>(memref.getLayout())) {
    return tiled_layout_attr;
  }
  if (auto affine_map_attr = dyn_cast<AffineMapAttr>(memref.getLayout())) {
    if (memref.getRank() == 0) {
      return emitError(UnknownLoc::get(memref.getContext()),
                       "0-rank memref not supported");
    }
    if (!affine_map_attr.isIdentity()) {
      return emitError(UnknownLoc::get(memref.getContext()),
                       "Non-identity affine layout");
    }
    if (!memref.isIntOrFloat()) {
      return emitError(UnknownLoc::get(memref.getContext()),
                       "Invalid element type for memref");
    }
    const int8_t bitwidth = memref.getElementTypeBitWidth();
    // Infer the layout
    if (memref.getRank() == 1) {
      const int64_t leading_tile =
          getTilingFactor(llvm::divideCeil(memref.getShape().back(), 128),
                          hardware_generation, bitwidth) *
          128;
      SmallVector<xla::Tile> tiles{xla::Tile({leading_tile})};
      if (bitwidth != 32) {
        if (!llvm::has_single_bit<unsigned>(bitwidth) || bitwidth > 32) {
          return emitError(UnknownLoc::get(memref.getContext()),
                           "Unsupported bitwidth: ")
                 << bitwidth;
        }
        tiles.append({xla::Tile({128}), xla::Tile({32 / bitwidth, 1})});
      }
      return TiledLayoutAttr::get(memref.getContext(), /*rank=*/1, tiles);
    }
    // memref.getRank() > 1
    const ArrayRef<int64_t> shape = memref.getShape();
    const int64_t second_minor = shape[shape.size() - 2];
    const int64_t leading_tile_rows =
        getTilingFactor(second_minor, hardware_generation, bitwidth);
    SmallVector<xla::Tile> tiles{xla::Tile({leading_tile_rows, 128})};
    if (bitwidth != 32) {
      if (!llvm::has_single_bit<unsigned>(bitwidth) || bitwidth > 32) {
        return emitError(UnknownLoc::get(memref.getContext()),
                         "Unsupported bitwidth: ")
               << bitwidth;
      }
      tiles.push_back(xla::Tile({32 / bitwidth, 1}));
    }
    return TiledLayoutAttr::get(memref.getContext(), memref.getRank(), tiles);
  }
  return emitError(UnknownLoc::get(memref.getContext()),
                   "Unrecognized layout annotation");
}

// Make sure only the first tile might introduce padding.
LogicalResult checkTiles(MLIRContext *mlir_ctx,
                         const ArrayRef<xla::Tile> &tiles) {
  SmallVector<int64_t> tiled_dims(tiles.front().dimensions().begin(),
                                  tiles.front().dimensions().end());
  for (const xla::Tile &t : tiles.drop_front()) {
    const int64_t offset = tiled_dims.size() - t.dimensions().size();
    if (offset < 0) {
      return emitError(UnknownLoc::get(mlir_ctx),
                       "Not implemented: layout too complicated");
    }
    for (int i = 0; i < t.dimensions().size(); ++i) {
      auto [d, m] = std::div(tiled_dims[offset + i], t.dimension(i));
      if (m != 0) {
        return emitError(UnknownLoc::get(mlir_ctx),
                         "Not implemented: layout too complicated");
      }
      tiled_dims[offset + i] = d;
    }
    tiled_dims.append(t.dimensions().begin(), t.dimensions().end());
  }
  return success();
}

FailureOr<MemRefType> inferMemref(MemRefType memref,
                                  const int hardware_generation) {
  const Attribute vmem =
      tpu::MemorySpaceAttr::get(memref.getContext(), MemorySpace::vmem);
  const Attribute memory_space =
      memref.getMemorySpace() == nullptr ? vmem : memref.getMemorySpace();
  FAILUREOR_ASSIGN_OR_RETURN(const TiledLayoutAttr layout,
                             inferLayout(memref, hardware_generation));

  const ArrayRef<xla::Tile> tiles = layout.getTiles();
  if (failed(checkTiles(memref.getContext(), tiles))) {
    return failure();
  }
  const xla::Tile &first_tile = tiles.front();
  const int64_t untiled_dims =
      memref.getShape().size() - first_tile.dimensions().size();
  if (untiled_dims < 0) {
    return emitError(UnknownLoc::get(memref.getContext()), "Invalid tiling");
  }
  SmallVector<int64_t> new_shape(memref.getShape());
  for (int i = 0; i < first_tile.dimensions().size(); ++i) {
    new_shape[untiled_dims + i] =
        llvm::alignTo(new_shape[untiled_dims + i], first_tile.dimension(i));
  }
  return MemRefType::get(new_shape, memref.getElementType(), layout,
                         memory_space);
}

}  // namespace mlir::tpu