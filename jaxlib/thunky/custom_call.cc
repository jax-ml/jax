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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "jaxlib/thunky/proto_to_thunks.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/ffi/attributes.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

absl::StatusOr<ThunkSequence> ThunkyInlineModuleHandler(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx) {
  ABSL_ASSIGN_OR_RETURN(xla::ffi::Attributes attrs, ctx.GetFfiAttributes());
  ABSL_ASSIGN_OR_RETURN(absl::string_view proto_bytes,
                        attrs.Get<absl::string_view>("module"));

  auto get_i64_attr = [&](absl::string_view name) -> absl::StatusOr<int64_t> {
    if (attrs.Contains<int64_t>(name)) {
      return attrs.Get<int64_t>(name);
    }
    if (attrs.Contains<int32_t>(name)) {
      ABSL_ASSIGN_OR_RETURN(int32_t val, attrs.Get<int32_t>(name));
      return static_cast<int64_t>(val);
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Attribute '%s' must be integer", name));
  };

  ABSL_ASSIGN_OR_RETURN(int64_t num_aliased_outputs,
                        get_i64_attr("num_aliased_outputs"));
  ABSL_ASSIGN_OR_RETURN(int64_t num_scratch, get_i64_attr("num_scratch"));

  GpuExecutableProto gpu_proto;
  if (!gpu_proto.ParseFromString(proto_bytes)) {
    return absl::InvalidArgumentError(
        "Failed to parse GpuExecutableProto in thunky.inline_module");
  }

  int64_t num_inputs = instr.operand_count();
  std::vector<BufferAllocation::Slice> input_slices;
  input_slices.reserve(num_inputs);
  for (int64_t i = 0; i < num_inputs; ++i) {
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                          ctx.GetOperandAllocationSlice(i, {}));
    input_slices.push_back(slice);
  }

  std::vector<BufferAllocation::Slice> scratch_slices;
  scratch_slices.reserve(num_scratch);
  for (int64_t j = 0; j < num_scratch; ++j) {
    int64_t result_idx = num_aliased_outputs + j;
    ShapeIndex shape_index =
        instr.shape().IsTuple() ? ShapeIndex({result_idx}) : ShapeIndex({});
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                          ctx.GetResultAllocationSlice(shape_index));
    scratch_slices.push_back(slice);
  }

  return LowerGpuExecutableProtoToThunkSequence(
      gpu_proto, input_slices, scratch_slices,
      [&]() { return ctx.GenerateThunkInfo(); },
      ctx.GetDeviceDescription().gpu_compute_capability());
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
