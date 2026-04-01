/* Copyright 2025 The JAX Authors.

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

#include "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h"

#include <cstdint>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
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
  mlir::DenseI32ArrayAttr tiling_attr = mlir::DenseI32ArrayAttr::get(
      unwrap(ctx), llvm::ArrayRef<int32_t>(tiling, tiling_size));
  return wrap(mosaic_gpu::TileTransformAttr::get(unwrap(ctx), tiling_attr));
}

MlirAttribute mlirMosaicGpuTileTransformAttrGetTiling(MlirAttribute attr) {
  return wrap(
      mlir::cast<mosaic_gpu::TileTransformAttr>(unwrap(attr)).getTiling());
}

MlirTypeID mlirMosaicGpuTileTransformAttrGetTypeID() {
  return wrap(mosaic_gpu::TileTransformAttr::getTypeID());
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
  mlir::DenseI32ArrayAttr permutation_attr = mlir::DenseI32ArrayAttr::get(
      unwrap(ctx), llvm::ArrayRef<int32_t>(permutation, permutation_size));
  return wrap(
      mosaic_gpu::TransposeTransformAttr::get(unwrap(ctx), permutation_attr));
}
MlirAttribute mlirMosaicGpuTransposeTransformAttrGetPermutation(
    MlirAttribute attr) {
  return wrap(mlir::cast<mosaic_gpu::TransposeTransformAttr>(unwrap(attr))
                  .getPermutation());
}

MlirTypeID mlirMosaicGpuTransposeTransformAttrGetTypeID() {
  return wrap(mosaic_gpu::TransposeTransformAttr::getTypeID());
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
MlirTypeID mlirMosaicGpuSwizzleTransformAttrGetTypeID() {
  return wrap(mosaic_gpu::SwizzleTransformAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// WGSplatFragLayoutAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsAWGSplatFragLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::WGSplatFragLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuWGSplatFragLayoutAttrGet(MlirContext ctx,
                                                    MlirAttribute shape) {
  return wrap(mosaic_gpu::WGSplatFragLayoutAttr::get(
      unwrap(ctx), mlir::cast<mlir::DenseI64ArrayAttr>(unwrap(shape))));
}

MlirTypeID mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID(void) {
  return wrap(mosaic_gpu::WGSplatFragLayoutAttr::getTypeID());
}

MlirAttribute mlirMosaicGpuWGSplatFragLayoutAttrGetShape(MlirAttribute attr) {
  return wrap(
      mlir::cast<mosaic_gpu::WGSplatFragLayoutAttr>(unwrap(attr)).getShape());
}

//===----------------------------------------------------------------------===//
// WGStridedFragLayoutAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsAWGStridedFragLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::WGStridedFragLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuWGStridedFragLayoutAttrGet(MlirContext ctx,
                                                      MlirAttribute shape,
                                                      int32_t vector_size) {
  return wrap(mosaic_gpu::WGStridedFragLayoutAttr::get(
      unwrap(ctx), mlir::cast<mlir::DenseI64ArrayAttr>(unwrap(shape)),
      vector_size));
}

MlirTypeID mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID(void) {
  return wrap(mosaic_gpu::WGStridedFragLayoutAttr::getTypeID());
}

MlirAttribute mlirMosaicGpuWGStridedFragLayoutAttrGetShape(MlirAttribute attr) {
  return wrap(
      mlir::cast<mosaic_gpu::WGStridedFragLayoutAttr>(unwrap(attr)).getShape());
}

int32_t mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(MlirAttribute attr) {
  return mlir::cast<mosaic_gpu::WGStridedFragLayoutAttr>(unwrap(attr))
      .getVectorSize();
}

//===----------------------------------------------------------------------===//
// ReplicatedAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsAReplicatedAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::ReplicatedAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuReplicatedAttrGet(MlirContext ctx, int32_t times) {
  return wrap(mosaic_gpu::ReplicatedAttr::get(unwrap(ctx), times));
}

MlirTypeID mlirMosaicGpuReplicatedAttrGetTypeID(void) {
  return wrap(mosaic_gpu::ReplicatedAttr::getTypeID());
}

int32_t mlirMosaicGpuReplicatedAttrGetTimes(MlirAttribute attr) {
  return mlir::cast<mosaic_gpu::ReplicatedAttr>(unwrap(attr)).getTimes();
}

//===----------------------------------------------------------------------===//
// TiledLayoutAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsATiledLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::TiledLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuTiledLayoutAttrGet(MlirContext ctx,
                                              MlirAttribute tiling,
                                              MlirAttribute warp_dims,
                                              MlirAttribute lane_dims,
                                              int32_t vector_dim) {
  return wrap(mosaic_gpu::TiledLayoutAttr::get(
      unwrap(ctx), mlir::cast<mlir::ArrayAttr>(unwrap(tiling)),
      mlir::cast<mlir::ArrayAttr>(unwrap(warp_dims)),
      mlir::cast<mlir::ArrayAttr>(unwrap(lane_dims)), vector_dim));
}

MlirTypeID mlirMosaicGpuTiledLayoutAttrGetTypeID(void) {
  return wrap(mosaic_gpu::TiledLayoutAttr::getTypeID());
}

MlirAttribute mlirMosaicGpuTiledLayoutAttrGetTiling(MlirAttribute attr) {
  return wrap(
      mlir::cast<mosaic_gpu::TiledLayoutAttr>(unwrap(attr)).getTiling());
}

MlirAttribute mlirMosaicGpuTiledLayoutAttrGetWarpDims(MlirAttribute attr) {
  return wrap(
      mlir::cast<mosaic_gpu::TiledLayoutAttr>(unwrap(attr)).getWarpDims());
}

MlirAttribute mlirMosaicGpuTiledLayoutAttrGetLaneDims(MlirAttribute attr) {
  return wrap(
      mlir::cast<mosaic_gpu::TiledLayoutAttr>(unwrap(attr)).getLaneDims());
}

int32_t mlirMosaicGpuTiledLayoutAttrGetVectorDim(MlirAttribute attr) {
  return mlir::cast<mosaic_gpu::TiledLayoutAttr>(unwrap(attr)).getVectorDim();
}

//===----------------------------------------------------------------------===//
// CopyPartitionAttrInterface
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsACopyPartitionAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::CopyPartition>(unwrap(attr));
}

//===----------------------------------------------------------------------===//
// CopyReplicatedAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsACopyReplicatedAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::CopyReplicatedAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuCopyReplicatedAttrGet(MlirContext ctx) {
  return wrap(mosaic_gpu::CopyReplicatedAttr::get(unwrap(ctx)));
}

MlirTypeID mlirMosaicGpuCopyReplicatedAttrGetTypeID() {
  return wrap(mosaic_gpu::CopyReplicatedAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// CopyPartitionedAttr
//===----------------------------------------------------------------------===//

bool mlirMosaicGpuIsACopyPartitionedAttr(MlirAttribute attr) {
  return mlir::isa<mosaic_gpu::CopyPartitionedAttr>(unwrap(attr));
}

MlirAttribute mlirMosaicGpuCopyPartitionedAttrGet(MlirContext ctx,
                                                  int32_t axis) {
  return wrap(mosaic_gpu::CopyPartitionedAttr::get(unwrap(ctx), axis));
}

int32_t mlirMosaicGpuCopyPartitionedAttrGetAxis(MlirAttribute attr) {
  return mlir::cast<mosaic_gpu::CopyPartitionedAttr>(unwrap(attr)).getAxis();
}

MlirTypeID mlirMosaicGpuCopyPartitionedAttrGetTypeID() {
  return wrap(mosaic_gpu::CopyPartitionedAttr::getTypeID());
}
