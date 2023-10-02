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

#include "jax/_src/lib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

#include "mlir/include/mlir/CAPI/Pass.h"
#include "mlir/include/mlir/CAPI/Registration.h"
#include "mlir/include/mlir/CAPI/Support.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "jax/_src/lib/mosaic/dialect/tpu/tpu_dialect.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TPU, tpu, mlir::tpu::TPUDialect);

bool mlirTPUAttributeIsATiledLayoutAttr(MlirAttribute attr) {
  return llvm::isa<mlir::tpu::TiledLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirTPUTiledLayoutAttrGetTiles(MlirAttribute attr) {
  auto layout_attr = llvm::cast<mlir::tpu::TiledLayoutAttr>(unwrap(attr));
  std::vector<mlir::Attribute> tile_attrs;
  tile_attrs.reserve(layout_attr.getTiles().size());
  mlir::MLIRContext *ctx = layout_attr.getContext();
  for (auto &tile : layout_attr.getTiles()) {
    auto d = tile.dimensions();
    tile_attrs.push_back(mlir::DenseI64ArrayAttr::get(
        ctx, llvm::ArrayRef<int64_t>(d.begin(), d.end())));
  }
  return wrap(mlir::ArrayAttr::get(ctx, tile_attrs));
}

void mlirTPUAnalyzePotentialCommunication(MlirOperation op,
                                          bool *has_communication,
                                          bool *has_custom_barrier) {
  auto result = mlir::tpu::mightCommunicateBetweenChips(unwrap(op));
  *has_communication = result.first;
  *has_custom_barrier = result.second;
}

using namespace mlir::tpu;

#include "jax/_src/lib/mosaic/dialect/tpu/integrations/c/tpu_passes.capi.cc.inc"
}
