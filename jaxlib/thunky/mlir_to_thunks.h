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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_MLIR_TO_THUNKS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_MLIR_TO_THUNKS_H_

#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"

namespace xla::gpu {

absl::StatusOr<ThunkSequence> LowerBlockToThunkSequence(
    mlir::Block& block,
    const llvm::DenseMap<mlir::Value, BufferAllocation::Slice>& value_to_slice,
    llvm::DenseMap<mlir::Value, std::shared_ptr<AsyncExecution>>&
        token_to_async_exec,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info);

absl::StatusOr<ThunkSequence> LowerThunkyModuleToThunkSequence(
    mlir::ModuleOp module,
    absl::Span<const BufferAllocation::Slice> input_slices,
    absl::Span<const BufferAllocation::Slice> scratch_slices,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info);

}  // namespace xla::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_THUNKY_MLIR_TO_THUNKS_H_
