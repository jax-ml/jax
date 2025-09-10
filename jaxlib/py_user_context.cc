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

#include "jaxlib/py_user_context.h"

#include <Python.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "jaxlib/python_ref_manager.h"
#include "jaxlib/traceback.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/version.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/fingerprint.h"

namespace jax {

namespace nb = ::nanobind;

// For LLVM RTTI.
char PyUserContext::ID = 0;

xla::ifrt::UserContextRef PyUserContext::Create(
    std::optional<Traceback> traceback) {
  if (traceback.has_value()) {
    return tsl::TakeRef<PyUserContext>(
        new PyUserContext(*std::move(traceback)));
  }
  return {};
}

xla::ifrt::UserContextRef PyUserContext::Create() {
  return Create(Traceback::Get());
}

PyUserContext::PyUserContext(Traceback traceback)
    : traceback_(std::move(traceback)) {}

PyUserContext::~PyUserContext() {
  // The traceback must be destroyed under the GIL.
  GlobalPyRefManager()->AddGarbage(std::move(traceback_));
}

Traceback PyUserContext::traceback() const {
  CHECK(PyGILState_Check());
  return traceback_;
}

uint64_t PyUserContext::Fingerprint() const {
  absl::MutexLock lock(mu_);

  if (fingerprint_.has_value()) {
    return *fingerprint_;
  }

  xla::SlowOperationAlarm slow_gil_alarm(
      absl::Seconds(20),
      "Acquiring the GIL in PyUserContext::Fingerprint took longer than 20s. "
      "This can occur when an operation blocks while holding the GIL.");
  nb::gil_scoped_acquire gil_acquire;
  slow_gil_alarm.cancel();

  try {
    // Make a blob of the content of the traceback and fingerprint them at once.
    // This is faster than `traceback_->ToString()` because it does not resolve
    // the source code line number of each frame and skips most string
    // formatting.
    std::string blob;
    for (const TracebackEntry& frame : traceback_.RawFrames()) {
      absl::StrAppend(
          &blob,
          nb::cast<std::string_view>(nb::handle(frame.code->co_filename)), "\t",
          nb::cast<std::string_view>(nb::handle(frame.code->co_name)), "\t",
          frame.code->co_firstlineno, "\t", frame.lasti, "\n");
    }
    fingerprint_ = tsl::Fingerprint64(blob);
  } catch (std::exception& e) {
    // If the traceback cannot be fingerprinted, we return 0 as a graceful
    // fallback.
    fingerprint_ = 0;
  }
  return *fingerprint_;
}

#if JAX_IFRT_VERSION_NUMBER >= 25
xla::ifrt::UserContextId PyUserContext::Id() const {
  return xla::ifrt::UserContextId(Fingerprint());
}
#endif

std::string PyUserContext::DebugString() const {
  absl::MutexLock lock(mu_);

  if (debug_str_.has_value()) {
    return *debug_str_;
  }

  xla::SlowOperationAlarm slow_gil_alarm(
      absl::Seconds(20),
      "Acquiring the GIL in PyUserContext::DebugString took longer than 20s. "
      "This can occur when an operation blocks while holding the GIL.");
  nb::gil_scoped_acquire gil_acquire;
  slow_gil_alarm.cancel();

  try {
    debug_str_ = traceback_.ToString();
  } catch (std::exception& e) {
    debug_str_ = absl::StrFormat(
        "(traceback could not be converted to a string: %s)", e.what());
  }
  return *debug_str_;
}

std::optional<Traceback> GetTraceback(
    const xla::ifrt::UserContext* user_context) {
  if (const auto* py_user_context =
          llvm::dyn_cast_or_null<PyUserContext>(user_context)) {
    return py_user_context->traceback();
  }
  return std::nullopt;
}

}  // namespace jax
