#include "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h"

#include <cstdint>
#include <vector>

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu.h"

//===----------------------------------------------------------------------===//
// TileTransformAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsATileTransformAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::TileTransformAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuTileTransformAttrGet(MlirContext ctx,
                                                int32_t* tiling,
                                                int32_t tiling_size) {
  return wrap(mosaic_gpu::TileTransformAttr::get(
      unwrap(ctx), llvm::ArrayRef<int32_t>(tiling, tiling_size)));
}

int32_t mlirMosaicGpuTileTransformAttrGetTilingSize(MlirAttribute attr) {
  return mlir::cast<mosaic_gpu::TileTransformAttr>(unwrap(attr))
      .getTiling()
      .size();
}

int32_t mlirMosaicGpuTileTransformAttrGetTiling(MlirAttribute attr,
                                                int32_t index) {
  return mlir::cast<mosaic_gpu::TileTransformAttr>(unwrap(attr))
      .getTiling()[index];
}
//===----------------------------------------------------------------------===//
// TransposeTransformAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsATransposeTransformAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::TransposeTransformAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuTransposeTransformAttrGet(MlirContext ctx,
                                                     int32_t* permutation,
                                                     int32_t permutation_size) {
  return wrap(mosaic_gpu::TransposeTransformAttr::get(
      unwrap(ctx), llvm::ArrayRef<int32_t>(permutation, permutation_size)));
}

int32_t mlirMosaicGpuTransposeTransformAttrGetPermutationSize(
    MlirAttribute attr) {
  return mlir::cast<mosaic_gpu::TransposeTransformAttr>(unwrap(attr))
      .getPermutation()
      .size();
}

int32_t mlirMosaicGpuTransposeTransformAttrGetPermutation(MlirAttribute attr,
                                                          int32_t index) {
  return mlir::cast<mosaic_gpu::TransposeTransformAttr>(unwrap(attr))
      .getPermutation()[index];
}

//===----------------------------------------------------------------------===//
// SwizzleTransformAttr
//===----------------------------------------------------------------------===//
bool mlirMosaicGpuIsASwizzleTransformAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::SwizzleTransformAttr>(unwrap(attr));
}
MlirAttribute mlirMosaicGpuSwizzleTransformAttrGet(MlirContext ctx,
                                                   int32_t swizzle) {
  return wrap(mosaic_gpu::SwizzleTransformAttr::get(
      unwrap(ctx),
      mosaic_gpu::SwizzlingModeAttr::get(
          unwrap(ctx), static_cast<mosaic_gpu::SwizzlingMode>(swizzle))));
}
int32_t mlirMosaicGpuSwizzleTransformAttrGetSwizzle(MlirAttribute attr) {
  return static_cast<int32_t>(
      mlir::cast<mosaic_gpu::SwizzleTransformAttr>(unwrap(attr))
          .getSwizzle()
          .getValue());
}

//===----------------------------------------------------------------------===//
// LayoutAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsALayoutAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::LayoutAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuLayoutAttrGet(MlirContext ctx,
                                         int32_t num_dimensions,
                                         MlirAttribute* transforms,
                                         int32_t transforms_size) {
  std::vector<mlir::Attribute> unwrapped_transforms;
  unwrapped_transforms.reserve(transforms_size);
  for (int i = 0; i < transforms_size; ++i) {
    unwrapped_transforms.push_back(unwrap(transforms[i]));
  }
  return wrap(mosaic_gpu::LayoutAttr::get(unwrap(ctx), num_dimensions,
                                          unwrapped_transforms));
}

int32_t mlirMosaicGpuLayoutAttrGetTransformsSize(MlirAttribute attr) {
  return mlir::cast<mosaic_gpu::LayoutAttr>(unwrap(attr))
      .getTransforms()
      .size();
}

MlirAttribute mlirMosaicGpuLayoutAttrGetTransform(MlirAttribute attr,
                                                  int32_t index) {
  return wrap(
      mlir::cast<mosaic_gpu::LayoutAttr>(unwrap(attr)).getTransforms()[index]);
}