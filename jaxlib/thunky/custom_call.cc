/* Copyright 2026 The JAX Authors.

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

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "jaxlib/thunky/dialect/thunky_dialect.h"
#include "jaxlib/thunky/mlir_to_thunks.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

absl::StatusOr<ThunkSequence> ThunkyInlineModuleHandler(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx) {
  ASSIGN_OR_RETURN(xla::ffi::AttributesMap attrs, ctx.GetFfiAttributes());
  auto module_it = attrs.find("module");
  if (module_it == attrs.end() ||
      !std::holds_alternative<std::string>(module_it->second.AsVariant())) {
    return absl::InvalidArgumentError(
        "Expected string 'module' attribute in thunky.inline_module");
  }
  std::string module_str = std::get<std::string>(module_it->second.AsVariant());

  auto get_i64_attr = [&](absl::string_view name) -> absl::StatusOr<int64_t> {
    auto it = attrs.find(std::string(name));
    if (it == attrs.end() ||
        !std::holds_alternative<xla::ffi::Scalar>(it->second.AsVariant())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected scalar '%s' attribute", name));
    }
    const auto& scalar = std::get<xla::ffi::Scalar>(it->second.AsVariant());
    if (std::holds_alternative<int64_t>(scalar.AsVariant())) {
      return std::get<int64_t>(scalar.AsVariant());
    }
    if (std::holds_alternative<int32_t>(scalar.AsVariant())) {
      return static_cast<int64_t>(std::get<int32_t>(scalar.AsVariant()));
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Attribute '%s' must be integer", name));
  };

  ASSIGN_OR_RETURN(int64_t num_aliased_outputs,
                   get_i64_attr("num_aliased_outputs"));
  ASSIGN_OR_RETURN(int64_t num_scratch, get_i64_attr("num_scratch"));

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::func::FuncDialect>();
  mlir_context.loadDialect<mlir::thunky::ThunkyDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(module_str, &mlir_context);
  if (!module) {
    return absl::InvalidArgumentError("Failed to parse thunky MLIR module");
  }

  int64_t num_inputs = instr.operand_count();
  std::vector<BufferAllocation::Slice> input_slices;
  input_slices.reserve(num_inputs);
  for (int64_t i = 0; i < num_inputs; ++i) {
    ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                     ctx.GetOperandAllocationSlice(i, {}));
    input_slices.push_back(slice);
  }

  std::vector<BufferAllocation::Slice> scratch_slices;
  scratch_slices.reserve(num_scratch);
  for (int64_t j = 0; j < num_scratch; ++j) {
    int64_t result_idx = num_aliased_outputs + j;
    ShapeIndex shape_index =
        instr.shape().IsTuple() ? ShapeIndex({result_idx}) : ShapeIndex({});
    ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                     ctx.GetResultAllocationSlice(shape_index));
    scratch_slices.push_back(slice);
  }

  return LowerThunkyModuleToThunkSequence(
      *module, input_slices, scratch_slices,
      [&]() { return ctx.GenerateThunkInfo(); });
}

XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER("thunky.inline_module",
                                            ThunkyInlineModuleHandler);

static absl::Status ThunkyInlineModuleFfiStub(xla::ffi::RemainingArgs,
                                              xla::ffi::RemainingRets,
                                              xla::ffi::Dictionary) {
  return absl::InternalError(
      "thunky.inline_module should be lowered by XLA:GPU ThunkEmitter");
}

XLA_FFI_DEFINE_HANDLER(
    kThunkyInlineModuleFfiStub, ThunkyInlineModuleFfiStub,
    xla::ffi::Ffi::Bind().RemainingArgs().RemainingRets().Attrs());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "thunky.inline_module",
                         "CUDA", kThunkyInlineModuleFfiStub);

}  // namespace
}  // namespace xla::gpu
