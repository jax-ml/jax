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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_PROTO_TO_THUNKS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_PROTO_TO_THUNKS_H_

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Deserializes the thunks in `gpu_proto` into an executable XLA:GPU
// `ThunkSequence`, rebasing `gpu_proto`'s virtual buffer allocations onto the
// caller's physical `BufferAllocation::Slice`s (`input_slices` followed by
// `scratch_slices`).
absl::StatusOr<ThunkSequence> LowerGpuExecutableProtoToThunkSequence(
    const GpuExecutableProto& gpu_proto,
    absl::Span<const BufferAllocation::Slice> input_slices,
    absl::Span<const BufferAllocation::Slice> scratch_slices,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    const se::GpuComputeCapability& gpu_compute_capability);

}  // namespace xla::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_PROTO_TO_THUNKS_H_
