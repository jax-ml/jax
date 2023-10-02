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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_

#include "mlir/include/mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "jax/_src/lib/mosaic/dialect/tpu/integrations/c/tpu_passes.capi.h.inc"

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TPU, tpu);

MLIR_CAPI_EXPORTED bool mlirTPUAttributeIsATiledLayoutAttr(MlirAttribute attr);

/// Encodes the tiles as an ArrayAttr of DenseI64ArrayAttrs.
MLIR_CAPI_EXPORTED MlirAttribute
mlirTPUTiledLayoutAttrGetTiles(MlirAttribute attr);

MLIR_CAPI_EXPORTED void mlirTPUAnalyzePotentialCommunication(
    MlirOperation op, bool* has_communication, bool* has_custom_barrier);

#ifdef __cplusplus
}
#endif

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
