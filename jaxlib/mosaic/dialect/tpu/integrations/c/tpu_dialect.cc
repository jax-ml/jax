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

#include "jaxlib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

#include <utility>

#include "llvm/Support/ErrorHandling.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/transforms/serde.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TPU, tpu, mlir::tpu::TPUDialect);

void mlirTPUAnalyzePotentialCommunication(MlirOperation op,
                                          bool *has_communication,
                                          bool *has_custom_barrier) {
  auto result = mlir::tpu::mightCommunicateBetweenChips(unwrap(op));
  *has_communication = result.first;
  *has_custom_barrier = result.second;
}

MLIR_CAPI_EXPORTED void mlirTpuRegisterMosaicSerdePass() {
  mlir::tpu::registerMosaicSerdePass();
}

}  // extern "C"
