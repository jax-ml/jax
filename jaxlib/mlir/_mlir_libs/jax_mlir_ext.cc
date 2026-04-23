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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyInsertionPoint;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyLocation;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContextRef;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyOperation;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyOperationRef;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyThreadContextEntry;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyValue;

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

// Converts an mlir::Value to a PyValue python object.
// It is faster to create a PyValue directly than to use the MlirValue type
// caster.
nb::object WrapMlirValue(PyMlirContextRef context_ref, mlir::Value value) {
  mlir::Operation* defining_op = value.getDefiningOp();
  PyOperationRef op_ref = [&]() {
    if (defining_op) {
      return PyOperation::forOperation(context_ref, wrap(defining_op));
    } else {
      auto block_arg = llvm::cast<mlir::BlockArgument>(value);
      mlir::Operation* parent_op = block_arg.getOwner()->getParentOp();
      if (!parent_op) {
        throw nb::value_error("Block argument has no parent operation.");
      }
      return PyOperation::forOperation(context_ref, wrap(parent_op));
    }
  }();
  PyValue py_value(op_ref, wrap(value));
  return py_value.maybeDownCast();
}

// Get the block/operation of the current Python insertion point.
std::pair<mlir::Block*, mlir::Operation*> GetInsertionPoint() {
  PyInsertionPoint* insertion_point =
      PyThreadContextEntry::getDefaultInsertionPoint();
  if (insertion_point == nullptr) {
    throw nb::value_error("No default insertion point found.");
  }
  mlir::Block* block = unwrap(insertion_point->getBlock().get());
  mlir::Operation* ref_op = nullptr;
  if (insertion_point->getRefOperation()) {
    if (auto* py_op = insertion_point->getRefOperation()->get(); py_op) {
      ref_op = unwrap(py_op->get());
    }
  }
  return {block, ref_op};
}

// Optimized version of arith.constant.
nb::object ArithConstant(nb::object value, MlirType type) {
  // The usual pattern for insertion points and locations is to use optional
  // arguments that have default type casters that do the same as the following.
  // Unfortunately they are slow, so we do the same directly.
  auto [block, ref_op] = GetInsertionPoint();

  PyLocation* location = PyThreadContextEntry::getDefaultLocation();
  if (location == nullptr) {
    throw nb::value_error("No default location found.");
  }
  mlir::Location loc = unwrap(location->get());

  mlir::Type mlir_type = unwrap(type);

  mlir::TypedAttr attr;
  if (nb::isinstance<nb::bool_>(value)) {
    attr = mlir::BoolAttr::get(loc.getContext(), nb::cast<bool>(value));
  } else if (nb::isinstance<nb::int_>(value)) {
    attr = mlir::IntegerAttr::get(mlir_type, nb::cast<int64_t>(value));
  } else if (nb::isinstance<nb::float_>(value)) {
    attr = mlir::FloatAttr::get(mlir_type, nb::cast<double>(value));
  } else {
    throw nb::value_error(
        absl::StrCat("Unsupported constant type: ",
                     nb::cast<std::string>(nb::repr(value.type())))
            .c_str());
  }

  mlir::ImplicitLocOpBuilder builder(loc, block, block->end());
  if (ref_op) {
    builder.setInsertionPoint(ref_op);
  }
  auto op = builder.create<mlir::arith::ConstantOp>(mlir_type, attr);

  return WrapMlirValue(location->getContext(), op.getOperation()->getResult(0));
}

}  // namespace

nb::object InlinedCall(nb::object callee_obj, nb::sequence args,
                       nb::object loc_obj) {
  PyOperation& py_callee = nb::cast<PyOperation&>(callee_obj);
  mlir::Operation* callee = unwrap(py_callee.get());
  mlir::func::FuncOp func = llvm::cast<mlir::func::FuncOp>(callee);
  mlir::Region& body = func.getBody();
  if (body.getBlocks().size() != 1) {
    throw nb::value_error("expected function to have exactly one block");
  }
  mlir::Block& body_block = body.getBlocks().front();

  auto [block, ref_op] = GetInsertionPoint();

  mlir::OpBuilder op_builder = mlir::OpBuilder::atBlockEnd(block);
  if (ref_op != nullptr) {
    op_builder.setInsertionPoint(ref_op);
  }

  std::vector<mlir::Value> unwrapped_args;
  unwrapped_args.reserve(nb::len(args));
  for (nb::handle arg : args) {
    unwrapped_args.push_back(unwrap(nb::cast<PyValue&>(arg).get()));
  }

  if (body_block.getNumArguments() != unwrapped_args.size()) {
    throw nb::value_error(
        absl::StrFormat("expected callee to have %zu arguments, got %zu",
                        unwrapped_args.size(), body_block.getNumArguments())
            .c_str());
  }

  mlir::IRMapping mapping;
  for (auto [arg_value, arg] :
       llvm::zip(body_block.getArguments(), unwrapped_args)) {
    mapping.map(arg_value, arg);
  }

  mlir::Location parent_base_loc = [&]() {
    if (loc_obj.is_none()) {
      PyLocation* default_loc = PyThreadContextEntry::getDefaultLocation();
      if (default_loc == nullptr) {
        throw nb::value_error("No default location found.");
      }
      return unwrap(default_loc->get());
    }
    return unwrap(nb::cast<PyLocation&>(loc_obj).get());
  }();

  llvm::StringRef parent_op_type, parent_op_name;
  ParseLocation(parent_base_loc, parent_op_type, parent_op_name);

  std::optional<nb::list> return_values;
  PyLocation* default_loc = PyThreadContextEntry::getDefaultLocation();
  PyMlirContextRef context_ref = default_loc->getContext();

  for (mlir::Operation& op : body_block.getOperations()) {
    if (llvm::isa<mlir::func::ReturnOp>(op)) {
      if (return_values.has_value()) {
        throw nb::value_error(
            "expected function to have exactly one return op");
      }
      return_values.emplace();
      for (mlir::Value result : op.getOperands()) {
        mlir::Value mapped_result = mapping.lookup(result);
        return_values->append(WrapMlirValue(context_ref, mapped_result));
      }
    } else {
      mlir::Operation* cloned_op = op_builder.clone(op, mapping);
      cloned_op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
        // Compute a new location for the cloned op.
        // * The name should be "parent_op_name/child_op_name" (assuming both
        //   are present).
        // * We use the op_type of the parent.
        // * We concatenate the traceback of the parent with the traceback of
        //   the child.
        mlir::Location child_loc = op->getLoc();
        llvm::StringRef child_op_type, child_op_name;
        ParseLocation(child_loc, child_op_type, child_op_name);

        if (mlir::isa<mlir::UnknownLoc>(child_loc)) {
          child_loc = parent_base_loc;
        } else if (!mlir::isa<mlir::UnknownLoc>(parent_base_loc)) {
          child_loc = mlir::CallSiteLoc::get(child_loc, parent_base_loc);
        }
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
  if (!return_values.has_value()) {
    throw nb::value_error("expected function to have exactly one return op");
  }
  return *return_values;
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
    mlirRegisterTransformsStripDebugInfoPass();
  });

  m.def("enter_multi_threaded_execution", [](MlirContext context) {
    unwrap(context)->enterMultiThreadedExecution();
  });
  m.def("exit_multi_threaded_execution", [](MlirContext context) {
    unwrap(context)->exitMultiThreadedExecution();
  });

  m.def("inlined_func_call", &jax::InlinedCall, nb::arg("callee"),
        nb::arg("args"), nb::arg("loc").none() = nb::none(),
        nb::sig("def inlined_func_call(callee: mlir.ir.Operation, args: "
                "collections.abc.Sequence[mlir.ir.Value], loc: "
                "mlir.ir.Location | None = ...) -> list[mlir.ir.Value]"),
        "Makes an inlined call to a function containing a single block with a "
        "single return op.");

  m.def("arith_constant", &ArithConstant, nb::arg("value"), nb::arg("type"),
        nb::sig("def arith_constant(value: int | float | bool, type: "
                "mlir.ir.Type, /) -> mlir.ir.Value"),
        "Creates an arith.constant operation.");

  nb::class_<jax::TracebackToLocationCache>(m, "TracebackToLocationCache")
      .def(
          "__init__",
          [](TracebackToLocationCache* self, nb::callable code_to_filename,
             int frame_limit, MlirContext context) {
            new (self) TracebackToLocationCache(code_to_filename, frame_limit,
                                                unwrap(context));
          },
          nb::arg("code_to_filename"), nb::arg("frame_limit"),
          nb::arg("context").none() = nb::none())
      .def("get", &TracebackToLocationCache::Get,
           nb::sig(
               "def get(self, traceback: Traceback, /) -> mlir.ir.Location"));
}

}  // namespace jax
