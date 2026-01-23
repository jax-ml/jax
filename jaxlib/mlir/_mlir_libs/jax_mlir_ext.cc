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
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/integrations/c/passes.h"
#include "jaxlib/mlir/_mlir_libs/traceback_to_location.h"
#include "jaxlib/mosaic/gpu/integrations/c/passes.h"
#include "stablehlo/dialect/VhloOps.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/service/spmd/shardy/integrations/c/passes.h"

namespace nb = ::nanobind;

namespace jax {

namespace {

// Returns true if a location is a NameLoc with a FileLineColLoc child. We
// assume the NameLoc names a function name in a frame in this case.
bool IsFrameNameLocation(mlir::Location location) {
  return mlir::isa<mlir::NameLoc>(location) &&
         mlir::isa<mlir::FileLineColLoc>(
             mlir::cast<mlir::NameLoc>(location).getChildLoc());
}

// Split a location into an operation type and an operation name, and a tail
// location.
void ParseLocation(mlir::Location& location, llvm::StringRef& op_type,
                   llvm::StringRef& op_name) {
  while (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(location)) {
    if (IsFrameNameLocation(name_loc)) {
      break;
    }
    llvm::StringRef name = name_loc.getName().strref();
    if (name.ends_with(":")) {
      op_type = name;
    } else {
      op_name = name;
    }
    location = mlir::cast<mlir::NameLoc>(location).getChildLoc();
  }
}

}  // namespace

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

  mlir::Location parent_base_loc = unwrap(loc);
  llvm::StringRef parent_op_type, parent_op_name;
  ParseLocation(parent_base_loc, parent_op_type, parent_op_name);

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
      mlir::Operation* cloned_op = op_builder.clone(op, mapping);
      cloned_op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
        // Compute a new location for the cloned op.
        // * The name should be "parent_op_name/child_op_name" (assuming both
        //   are present).
        // * We use the op_type of the parent.
        // * We use the traceback of the parent. We want the location of the
        //   equation, not the location of the lowering rule.
        mlir::Location child_loc = op->getLoc();
        llvm::StringRef child_op_type, child_op_name;
        ParseLocation(child_loc, child_op_type, child_op_name);

        child_loc = mlir::CallSiteLoc::get(child_loc, parent_base_loc);
        if (child_op_name.empty()) {
          child_loc = mlir::NameLoc::get(
              op_builder.getStringAttr(parent_op_name), child_loc);
        } else if (parent_op_name.empty()) {
          child_loc = mlir::NameLoc::get(
              op_builder.getStringAttr(child_op_name), child_loc);
        } else {
          std::string name =
              absl::StrCat(static_cast<std::string_view>(parent_op_name), "/",
                           static_cast<std::string_view>(child_op_name));
          child_loc =
              mlir::NameLoc::get(op_builder.getStringAttr(name), child_loc);
        }
        if (!parent_op_type.empty()) {
          child_loc = mlir::NameLoc::get(
              op_builder.getStringAttr(parent_op_type), child_loc);
        }
        op->setLoc(child_loc);
        if (mlir::isa<mlir::sdy::ManualComputationOp>(op)) {
          // Skip `ManualComputationOp`s and their nested operations, they will
          // be handled separately.
          return mlir::WalkResult::skip();
        }
        return mlir::WalkResult::advance();
      });
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
    // TODO(jpienaar): these don't seem to have C API targets known to Bazel
    unwrap(registry)->insert<mlir::shape::ShapeDialect>();
    unwrap(registry)->insert<mlir::tensor::TensorDialect>();
    unwrap(registry)->insert<mlir::vhlo::VhloDialect>();

    // For Mosaic GPU
    REGISTER_DIALECT(cf);
    REGISTER_DIALECT(gpu);
    REGISTER_DIALECT(nvgpu);
    REGISTER_DIALECT(nvvm);
    REGISTER_DIALECT(llvm);
#undef REGISTER_DIALECT

    mlirMosaicGpuRegisterSerdePass();
    mlirRegisterTransformsPasses();
    // For Shardy
    mlirRegisterAllSdyPassesAndPipelines();
    mlirRegisterAllXlaSdyPassesAndPipelines();
    // Transforms used by JAX.
    mlirRegisterTransformsStripDebugInfo();
  });

  m.def("enter_multi_threaded_execution", [](MlirContext context) {
    unwrap(context)->enterMultiThreadedExecution();
  });
  m.def("exit_multi_threaded_execution", [](MlirContext context) {
    unwrap(context)->exitMultiThreadedExecution();
  });

  m.def("inlined_func_call", xla::ValueOrThrowWrapper(InlinedCall),
        nb::arg("callee"), nb::arg("args"), nb::arg("block"),
        nb::arg("loc").none() = nb::none(),
        "Makes an inlined call to a function containing a single block with a "
        "single return op.");

  nb::class_<TracebackToLocationCache>(m, "TracebackToLocationCache")
      .def(
          "__init__",
          [](TracebackToLocationCache* self, nb::callable code_to_filename,
             int frame_limit, MlirContext context) {
            new (self) TracebackToLocationCache(code_to_filename, frame_limit,
                                                unwrap(context));
          },
          nb::arg("code_to_filename"), nb::arg("frame_limit"),
          nb::arg("context").none() = nb::none())
      .def("get", &TracebackToLocationCache::Get);
}

}  // namespace jax
