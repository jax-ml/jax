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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_MLIR_TO_PROTO_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_MLIR_TO_PROTO_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/service/gpu/gpu_executable.pb.h"

namespace xla::gpu {

// Extracts a `GpuExecutableProto` from the bytes returned by
// `PjRtExecutable::Serialize` (`compiled.runtime_executable().serialize()`),
// which consists of a size-delimited `SerializedXlaExecutableMetadata` header
// followed by a Riegeli split-proto `ExecutableAndOptionsProto` whose
// `serialized_executable` field is a Riegeli split-proto `GpuExecutableProto`.
absl::StatusOr<GpuExecutableProto> ParseGpuExecutableProto(
    absl::string_view serialized_bytes);

// Lifts an XLA:GPU `GpuExecutableProto` into a `thunky` MLIR `ModuleOp` with a
// `@main` entry function whose arguments correspond to the executable's
// parameter, unaliased output, and internal scratch buffer allocations.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GpuExecutableProtoToThunkyModule(const GpuExecutableProto& gpu_proto,
                                 mlir::MLIRContext* context);

// Lowers a `thunky` MLIR `ModuleOp` (containing a `@main` function of buffer
// arguments and `thunky` dialect operations) into an
// `xla::gpu::GpuExecutableProto` populated with virtual buffer allocations,
// constants, and thunks.
absl::StatusOr<GpuExecutableProto> LowerThunkyModuleToGpuExecutableProto(
    mlir::ModuleOp module);

}  // namespace xla::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_MLIR_TO_PROTO_H_
