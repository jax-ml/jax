/* Copyright 2024 The JAX Authors.

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

#ifndef JAXLIB_GPU_HYBRID_KERNELS_H_
#define JAXLIB_GPU_HYBRID_KERNELS_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

// The MagmaLookup class is used for dlopening the MAGMA shared library,
// initializing it, and looking up MAGMA symbols.
class MagmaLookup {
 public:
  explicit MagmaLookup() = default;
  ~MagmaLookup();
  absl::StatusOr<void*> FindMagmaInit();
  absl::Status Initialize();
  absl::StatusOr<void*> Find(const char name[]);

 private:
  bool initialized_ = false;
  bool failed_ = false;
  void* handle_ = nullptr;
  std::optional<std::string> lib_path_ = std::nullopt;
  absl::flat_hash_map<std::string, void*> symbols_;
};

XLA_FFI_DECLARE_HANDLER_SYMBOL(kEigReal);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kEigComp);
XLA_FFI_DECLARE_HANDLER_SYMBOL(kGeqp3);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_HYBRID_KERNELS_H_
