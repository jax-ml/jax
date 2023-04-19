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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "jaxlib/mlir/_mlir_libs/passes/jax_passes.h"

namespace jax {

#define GEN_PASS_DEF_STRIPLOCATIONSPASS
#include "jaxlib/mlir/_mlir_libs/passes/jax_passes.h.inc"
#undef GEN_PASS_DEF_STRIPLOCATIONSPASS

namespace {

struct JaxStripLocationsPass
    : public impl::StripLocationsPassBase<JaxStripLocationsPass> {
  void runOnOperation() override {
    mlir::Operation* operation = getOperation();
    auto unknown_loc = mlir::UnknownLoc::get(operation->getContext());
    operation->walk(
        [unknown_loc](mlir::Operation* op) { op->setLoc(unknown_loc); });
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createStripLocationsPass() {
  return std::make_unique<JaxStripLocationsPass>();
}

}  // namespace jax
