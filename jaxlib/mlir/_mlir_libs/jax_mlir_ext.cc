/* Copyright 2022 The JAX Authors.

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

// Registers MLIR dialects used by JAX.
// This module is called by mlir/__init__.py during initialization.

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir-c/Dialect/Arith.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/ControlFlow.h"
#include "mlir-c/Dialect/Func.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/GPU.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/LLVM.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/Math.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/MemRef.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/NVGPU.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/NVVM.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/SCF.h"  // IWYU pragma: keep
#include "mlir-c/Dialect/Vector.h"  // IWYU pragma: keep
#include "mlir-c/IR.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "shardy/integrations/c/passes.h"
#include "jaxlib/mosaic/gpu/integrations/c/passes.h"
#include "xla/service/spmd/shardy/integrations/c/passes.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep

namespace nb = ::nanobind;

namespace jax {

absl::StatusOr<std::vector<MlirValue>> InlinedCall(
    MlirOperation c_callee, absl::Span<MlirValue const> c_args, MlirBlock block,
    MlirLocation loc) {
  mlir::Operation* callee = unwrap(c_callee);
  mlir::func::FuncOp func = llvm::cast<mlir::func::FuncOp>(callee);
  mlir::Region& body = func.getBody();
  if (body.getBlocks().size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("expected function to have exactly one block, got %d",
                        body.getBlocks().size()));
  }
  mlir::Block& body_block = body.getBlocks().front();

  mlir::OpBuilder op_builder = mlir::OpBuilder::atBlockEnd(unwrap(block));
  mlir::IRMapping mapping;
  if (body_block.getNumArguments() != c_args.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("expected callee to have %d arguments, got %d",
                        c_args.size(), body_block.getNumArguments()));
  }
  for (auto [arg_value, arg] : llvm::zip(body_block.getArguments(), c_args)) {
    mapping.map(arg_value, unwrap(arg));
  }
  std::optional<std::vector<MlirValue>> results;
  for (mlir::Operation& op : body_block.getOperations()) {
    if (llvm::isa<mlir::func::ReturnOp>(op)) {
      if (results.has_value()) {
        return absl::InternalError(
            "expected function to have exactly one return op");
      }
      results.emplace();
      for (mlir::Value result : op.getOperands()) {
        results->push_back(wrap(mapping.lookup(result)));
      }
    } else {
      op_builder.clone(op, mapping);
    }
  }
  if (!results.has_value()) {
    return absl::InternalError(
        "expected function to have exactly one return op");
  }
  return *results;
}

NB_MODULE(_jax_mlir_ext, m) {
  m.doc() = "Registers upstream MLIR dialects used by JAX.";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
#define REGISTER_DIALECT(name)                                           \
  MlirDialectHandle name##_dialect = mlirGetDialectHandle__##name##__(); \
  mlirDialectHandleInsertDialect(name##_dialect, registry)
    REGISTER_DIALECT(arith);
    REGISTER_DIALECT(func);
    REGISTER_DIALECT(math);
    REGISTER_DIALECT(memref);
    REGISTER_DIALECT(scf);
    REGISTER_DIALECT(vector);
    // For Mosaic GPU
    REGISTER_DIALECT(cf);
    REGISTER_DIALECT(gpu);
    REGISTER_DIALECT(nvgpu);
    REGISTER_DIALECT(nvvm);
    REGISTER_DIALECT(llvm);
#undef REGISTER_DIALECT

    mlirMosaicGpuRegisterPasses();
    mlirRegisterTransformsPasses();
    // For Shardy
    mlirRegisterAllSdyPassesAndPipelines();
    mlirRegisterAllXlaSdyPassesAndPipelines();
    // Transforms used by JAX.
    mlirRegisterTransformsStripDebugInfo();
  });

  m.def("inlined_func_call", xla::ValueOrThrowWrapper(InlinedCall),
        nb::arg("callee"), nb::arg("args"), nb::arg("block"),
        nb::arg("loc").none() = nb::none(),
        "Makes an inlined call to a function containing a single block with a "
        "single return op.");
}

}  // namespace jax
