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

#include "jaxlib/mosaic/gpu/dump.h"

#include <gtest/gtest.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace {

TEST(DumpTest, GetOrSetDumpOptionsForModuleReturnsConsistentBasenameForModule) {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(
      mlir::UnknownLoc::get(&ctx), /*name=*/"test_module");
  mosaic::gpu::DumpOptions opts1 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module);
  mosaic::gpu::DumpOptions opts2 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module);
  // The module basename should be consistent across calls for the same module.
  EXPECT_EQ(opts1.module_basename, opts2.module_basename);
}

TEST(
    DumpTest,
    GetOrSetDumpOptionsForModuleReturnsConsistentBasenameForDifferentModulesWithTheSameName) {  // NOLINT(whitespace/line_length)
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module1 = mlir::ModuleOp::create(
      mlir::UnknownLoc::get(&ctx), /*name=*/"test_module");
  mlir::OwningOpRef<mlir::ModuleOp> module2 = mlir::ModuleOp::create(
      mlir::UnknownLoc::get(&ctx), /*name=*/"test_module");
  mosaic::gpu::DumpOptions opts1 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module1);
  mosaic::gpu::DumpOptions opts2 =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module2);
  // The module basename should be different for different modules, even though
  // they have the same name.
  EXPECT_NE(opts1.module_basename, opts2.module_basename);
}

}  // namespace
