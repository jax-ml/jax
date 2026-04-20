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

#ifndef JAXLIB_MOSAIC_DIALECT_GPU_INTEGRATIONS_C_ATTRIBUTES_H_
#define JAXLIB_MOSAIC_DIALECT_GPU_INTEGRATIONS_C_ATTRIBUTES_H_

#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// TileTransformAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsATileTransformAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirMosaicGpuTileTransformAttrGet(
    MlirContext ctx, int32_t* tiling, int32_t tiling_size);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuTileTransformAttrGetTiling(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuTileTransformAttrGetTypeID();

//===----------------------------------------------------------------------===//
// TransposeTransformAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsATransposeTransformAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirMosaicGpuTransposeTransformAttrGet(
    MlirContext ctx, int32_t* permutation, int32_t permutation_size);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuTransposeTransformAttrGetPermutation(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuTransposeTransformAttrGetTypeID();

//===----------------------------------------------------------------------===//
// SwizzleTransformAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsASwizzleTransformAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuSwizzleTransformAttrGet(MlirContext ctx, int32_t swizzle);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuSwizzleTransformAttrGetSwizzle(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuSwizzleTransformAttrGetTypeID();

//===----------------------------------------------------------------------===//
// WGSplatFragLayoutAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsAWGSplatFragLayoutAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuWGSplatFragLayoutAttrGet(MlirContext ctx, MlirAttribute shape);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuWGSplatFragLayoutAttrGetShape(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// WGStridedFragLayoutAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsAWGStridedFragLayoutAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID
mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute mlirMosaicGpuWGStridedFragLayoutAttrGet(
    MlirContext ctx, MlirAttribute shape, int32_t vector_size);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuWGStridedFragLayoutAttrGetShape(MlirAttribute attr);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ReplicatedAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsAReplicatedAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuReplicatedAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute mlirMosaicGpuReplicatedAttrGet(MlirContext ctx,
                                                                int32_t times);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuReplicatedAttrGetTimes(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TiledLayoutAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsATiledLayoutAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuTiledLayoutAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute mlirMosaicGpuTiledLayoutAttrGet(
    MlirContext ctx, MlirAttribute tiling, MlirAttribute warp_dims,
    MlirAttribute lane_dims, int32_t vector_dim);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuTiledLayoutAttrGetTiling(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuTiledLayoutAttrGetWarpDims(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuTiledLayoutAttrGetLaneDims(MlirAttribute attr);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuTiledLayoutAttrGetVectorDim(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// CopyPartitionAttrInterface
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsACopyPartitionAttr(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// CopyReplicatedAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsACopyReplicatedAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuCopyReplicatedAttrGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuCopyReplicatedAttrGetTypeID();

//===----------------------------------------------------------------------===//
// CopyPartitionedAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsACopyPartitionedAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirMosaicGpuCopyPartitionedAttrGet(MlirContext ctx, int32_t axis);

MLIR_CAPI_EXPORTED int32_t
mlirMosaicGpuCopyPartitionedAttrGetAxis(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuCopyPartitionedAttrGetTypeID();

#ifdef __cplusplus
}
#endif

#endif  // JAXLIB_MOSAIC_DIALECT_GPU_INTEGRATIONS_C_ATTRIBUTES_H_
