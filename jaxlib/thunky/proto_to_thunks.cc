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

#include "jaxlib/thunky/proto_to_thunks.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "jaxlib/thunky/thunk_proto_util.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

// Rewrites every `BufferAllocationSliceProto` inside `thunk` (and any nested
// child thunks) from `GpuExecutableProto`'s 0-based virtual allocation index
// `i` to the physical `BufferAllocation::Slice` `operand_slices[i]`, adding the
// physical base offset to any intra-buffer slice offset.
absl::Status RebaseBufferAllocations(
    ThunkProto* thunk,
    absl::Span<const BufferAllocation::Slice> operand_slices) {
  return ForEachBufferSlice(thunk, [&](auto* slice) {
    int64_t idx = slice->buffer_allocation_index();
    if (idx >= 0 && idx < static_cast<int64_t>(operand_slices.size())) {
      const BufferAllocation::Slice& base = operand_slices[idx];
      slice->set_buffer_allocation_index(base.allocation()->index());
      slice->set_offset(base.offset() + slice->offset());
    }
  });
}

// Populates `ThunkInfoProto` (profile annotation and execution stream ID) on
// `tp` and all recursively nested thunks using `make_thunk_info()`. If
// `make_thunk_info()` leaves `thunk_id` at 0 (as in unit tests), preserves any
// non-zero `thunk_id` already assigned in the proto.
void AssignThunkInfoRecursively(
    ThunkProto* tp, absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info) {
  int64_t existing_id = tp->thunk_info().thunk_id();
  Thunk::ThunkInfo info = make_thunk_info();
  *tp->mutable_thunk_info() = info.ToProto();
  if (info.thunk_id.value() == 0 && existing_id != 0) {
    tp->mutable_thunk_info()->set_thunk_id(existing_id);
  }
  if (tp->has_sequential_thunk()) {
    for (auto& child : *tp->mutable_sequential_thunk()->mutable_thunks()) {
      AssignThunkInfoRecursively(&child, make_thunk_info);
    }
  } else if (tp->has_conditional_thunk()) {
    for (auto& branch :
         *tp->mutable_conditional_thunk()->mutable_branch_thunks()) {
      for (auto& child : *branch.mutable_thunks()) {
        AssignThunkInfoRecursively(&child, make_thunk_info);
      }
    }
  } else if (tp->has_while_thunk()) {
    for (auto& child : *tp->mutable_while_thunk()
                            ->mutable_condition_thunk_sequence()
                            ->mutable_thunks()) {
      AssignThunkInfoRecursively(&child, make_thunk_info);
    }
    for (auto& child : *tp->mutable_while_thunk()
                            ->mutable_body_thunk_sequence()
                            ->mutable_thunks()) {
      AssignThunkInfoRecursively(&child, make_thunk_info);
    }
  } else if (tp->has_async_start_thunk()) {
    for (auto& child :
         *tp->mutable_async_start_thunk()->mutable_thunks()->mutable_thunks()) {
      AssignThunkInfoRecursively(&child, make_thunk_info);
    }
  } else if (tp->has_collective_group_thunk()) {
    for (auto& child :
         *tp->mutable_collective_group_thunk()->mutable_thunks()) {
      AssignThunkInfoRecursively(&child, make_thunk_info);
    }
  }
}

}  // namespace

absl::StatusOr<ThunkSequence> LowerGpuExecutableProtoToThunkSequence(
    const GpuExecutableProto& gpu_proto,
    absl::Span<const BufferAllocation::Slice> input_slices,
    absl::Span<const BufferAllocation::Slice> scratch_slices,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    const se::GpuComputeCapability& gpu_compute_capability) {
  // `LowerThunkyModuleToGpuExecutableProto` lays out virtual buffer allocations
  // in `gpu_proto` in three contiguous groups:
  //   1. User input/output buffer parameters: [0, num_user_inputs)
  //   2. Internal scratch buffer parameters:  [num_user_inputs, num_main_args)
  //   3. Materialized constant buffers:       [num_main_args,
  //   total_virtual_allocs)
  int total_virtual_allocs = gpu_proto.buffer_allocations().values_size();
  int num_constants = gpu_proto.constants_size();
  if (total_virtual_allocs < num_constants) {
    return absl::InvalidArgumentError(
        "GpuExecutableProto has fewer buffer allocations than constants");
  }
  size_t num_main_args =
      static_cast<size_t>(total_virtual_allocs - num_constants);
  if (num_main_args < scratch_slices.size()) {
    return absl::InvalidArgumentError(
        "thunky GpuExecutableProto has fewer arguments than scratch_slices");
  }
  size_t num_user_inputs = num_main_args - scratch_slices.size();
  if (input_slices.size() <
      num_user_inputs + static_cast<size_t>(num_constants)) {
    return absl::InvalidArgumentError(
        "thunky LowerGpuExecutableProtoToThunkSequence has fewer input_slices "
        "than user inputs and constants");
  }

  // Build the mapping from `gpu_proto`'s virtual allocation indices to the
  // physical `BufferAllocation::Slice`s supplied by the custom call emitter.
  // Constants are passed by Python as trailing custom-call operands (in
  // `input_slices[num_user_inputs..]`), padded to a uniform size so XLA can
  // pass them; we slice each constant back to its exact virtual allocation
  // size.
  std::vector<BufferAllocation::Slice> virtual_to_physical;
  virtual_to_physical.reserve(total_virtual_allocs);
  for (size_t i = 0; i < num_user_inputs; ++i) {
    virtual_to_physical.push_back(input_slices[i]);
  }
  for (size_t j = 0; j < scratch_slices.size(); ++j) {
    virtual_to_physical.push_back(scratch_slices[j]);
  }
  for (int c = 0; c < num_constants; ++c) {
    BufferAllocation::Slice raw_slice = input_slices[num_user_inputs + c];
    int64_t size =
        gpu_proto.buffer_allocations().values(num_main_args + c).size();
    virtual_to_physical.push_back(BufferAllocation::Slice(
        raw_slice.allocation(), raw_slice.offset(), size));
  }

  // `DeserializeThunkSequenceProto` indexes into a contiguous
  // `absl::Span<const BufferAllocation>` by `allocation->index()`. Since XLA's
  // `BufferAssignment` stores all `BufferAllocation` objects contiguously in a
  // single `std::vector<BufferAllocation>`, we recover the base pointer and
  // span bounds from the `BufferAllocation*` pointers on the physical slices.
  const BufferAllocation* base_alloc = nullptr;
  int64_t max_alloc_index = -1;
  for (const auto& s : input_slices) {
    if (s.allocation() != nullptr) {
      base_alloc = s.allocation() - s.allocation()->index();
      max_alloc_index =
          std::max<int64_t>(max_alloc_index, s.allocation()->index());
    }
  }
  for (const auto& s : scratch_slices) {
    if (s.allocation() != nullptr) {
      base_alloc = s.allocation() - s.allocation()->index();
      max_alloc_index =
          std::max<int64_t>(max_alloc_index, s.allocation()->index());
    }
  }
  absl::Span<const BufferAllocation> buffer_allocations;
  if (base_alloc != nullptr && max_alloc_index >= 0) {
    buffer_allocations = absl::MakeConstSpan(base_alloc, max_alloc_index + 1);
  }

  // Rebase virtual buffer slices to physical allocations and stamp fresh
  // `ThunkInfo` on each thunk before deserializing into runtime `Thunk`
  // objects.
  ThunkSequenceProto seq_proto;
  for (const ThunkProto& orig_thunk : gpu_proto.thunks()) {
    ThunkProto* tp = seq_proto.add_thunks();
    *tp = orig_thunk;
    ABSL_RETURN_IF_ERROR(RebaseBufferAllocations(tp, virtual_to_physical));
    AssignThunkInfoRecursively(tp, make_thunk_info);
  }

  return DeserializeThunkSequenceProto(
      seq_proto, buffer_allocations,
      /*hlo_module=*/nullptr,
      gpu_compute_capability.IsRocm() ? "ROCM" : "CUDA",
      gpu_compute_capability,
      /*gpu_topology=*/std::nullopt);
}

}  // namespace xla::gpu
