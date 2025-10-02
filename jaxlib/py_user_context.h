/* Copyright 2025 The JAX Authors

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

#ifndef JAXLIB_PY_USER_CONTEXT_H_
#define JAXLIB_PY_USER_CONTEXT_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "jaxlib/traceback.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/version.h"

namespace jax {

// IFRT `UserContext` implementation for JAX that captures a Python traceback.
// Can be associated with an IFRT runtime objects such as `xla::ifrt::Array` and
// `xla::ifrt::LoadedExecutable` to track their creation.
//
// All methods are thread-safe.
class PyUserContext
    : public llvm::RTTIExtends<PyUserContext, xla::ifrt::UserContext> {
 public:
  // Creates a `PyUserContext` from a given Python traceback. If `traceback` is
  // `nullopt`, returns `nullptr`.
  static xla::ifrt::UserContextRef Create(std::optional<Traceback> traceback);

  // Creates a `PyUserContext` with a new `Traceback`. If JAX `Traceback` is not
  // enabled, returns `nullptr`.
  static xla::ifrt::UserContextRef Create();

  PyUserContext(const PyUserContext&) = delete;
  PyUserContext& operator=(const PyUserContext&) = delete;

  // Destructor. Does not require GIL.
  ~PyUserContext() override;

  // Returns the traceback captured by this `PyUserContext`.
  // Requires GIL.
  Traceback traceback() const;

  // UserContext implementation.

#if JAX_IFRT_VERSION_NUMBER < 28
  uint64_t Fingerprint() const override { return 1; }
#endif

  xla::ifrt::UserContextId Id() const override;

  // Returns a string representation of the traceback captured by this
  // `PyUserContext`.
  //
  // While GIL is not required to call this method, calling `DebugString()` when
  // the caller already holds GIL is strongly recommended to reduce the overhead
  // of (re)acquiring GIL.
  std::string DebugString() const override;

  // For LLVM RTTI.
  static char ID;  // NOLINT

 private:
  explicit PyUserContext(Traceback traceback);

  xla::ifrt::UserContextId id_;
  Traceback traceback_;

  // Debug string generation can be expensive. Maintain a cache for them.
  mutable absl::Mutex mu_;
  mutable std::optional<std::string> debug_str_ ABSL_GUARDED_BY(mu_);
};

// Retrieves a `Traceback` object from an IFRT `UserContext`. Returns `nullopt`
// if no `Traceback` was captured for a `PyUserContext` or `user_context` is not
// a `PyUserContext`.
//
// Requires GIL.
std::optional<Traceback> GetTraceback(
    const xla::ifrt::UserContext* user_context);

}  // namespace jax

#endif  // JAXLIB_PY_USER_CONTEXT_H_
