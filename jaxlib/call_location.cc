/* Copyright 2020 The JAX Authors

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

#include "jaxlib/call_location.h"

#include <atomic>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/traceback.h"
#include "jaxlib/py_user_context.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"

namespace nb = nanobind;

namespace jax {

namespace {

std::atomic<RuntimeTracebackMode> global_runtime_traceback_mode_ =
    RuntimeTracebackMode::kOff;

thread_local std::optional<RuntimeTracebackMode>
    runtime_traceback_mode_thread_local_ = std::nullopt;

RuntimeTracebackMode GetRuntimeTracebackMode() {
  if (runtime_traceback_mode_thread_local_.has_value()) {
    return *runtime_traceback_mode_thread_local_;
  }
  return global_runtime_traceback_mode_.load();
}

static absl::Mutex shared_data_mu;

static absl::NoDestructor<std::vector<std::string>> exclude_paths_from_python
  ABSL_GUARDED_BY(shared_data_mu);

static absl::NoDestructor<absl::flat_hash_map<std::string, bool>>
    known_code_objects ABSL_GUARDED_BY(shared_data_mu);

// Returns true if the code object is an internal JAX frame (Cached)
bool IsJaxInternalFrame(PyCodeObject* code) {
  nb::str file_name = nb::borrow<nb::str>(code->co_filename);
  std::string_view file_name_sv = nb::cast<std::string_view>(file_name);

  absl::MutexLock lock(shared_data_mu);
  auto it = known_code_objects->find(file_name_sv);
  if (it != known_code_objects->end()) {
    return it->second;
  }

  bool is_internal = false;
  for (const auto& prefix : *exclude_paths_from_python) {
    if (absl::StartsWith(file_name_sv, prefix)) {
      is_internal = true;
      break;
    }
  }
  (*known_code_objects)[file_name_sv] = is_internal;
  return is_internal;
}

// Returns the first non-JAX internal frame in the format "file:line"
std::string GetCallLocation(const jax::Traceback& traceback) {
  auto frames = traceback.RawFrames();
  for (const auto& frame : frames) {
    if (!IsJaxInternalFrame(frame.code)) {
      nb::str file_name = nb::borrow<nb::str>(frame.code->co_filename);
      int line_num = PyCode_Addr2Line(frame.code, frame.lasti);
      return absl::StrCat(nb::cast<std::string_view>(file_name), ":", line_num);
    }
  }
  return "";
}

}  // namespace

// Populates the "call_location" field in the execute options if
// jax.config.jax_send_traceback_to_runtime is not 'off'.
// A traceback will be collected from the current user context.
void PopulateCallLocation(xla::ifrt::ExecuteOptions& options,
                          const xla::ifrt::UserContext* user_context) {
  RuntimeTracebackMode mode = GetRuntimeTracebackMode();
  if (mode == RuntimeTracebackMode::kOff) {  // Default case
    return;
  }
  std::optional<Traceback> traceback = GetTraceback(user_context);
  if (!traceback.has_value()) {
    return;
  }

  std::string call_location_str;
  if (mode == RuntimeTracebackMode::kFull) {
    call_location_str = traceback->ToString();
  } else {  // mode == RuntimeTracebackMode::kOn
    call_location_str = GetCallLocation(*traceback);
  }

  if (!call_location_str.empty()) {
    xla::ifrt::AttributeMap::Map attrs_map;
    if (options.custom_options.has_value()) {
      attrs_map = options.custom_options->map();
    }
    attrs_map.insert(
        {std::string(xla::ifrt::PjRtCompatibleLoadedExecutable::kCallLocation),
         xla::ifrt::AttributeMap::StringValue(std::move(call_location_str))});
    options.custom_options.emplace(std::move(attrs_map));
  }
}

// Function to be called from Python to add a single path
void AddExcludePath(std::string path) {
  absl::MutexLock lock(shared_data_mu);
  exclude_paths_from_python->push_back(std::move(path));
  known_code_objects->clear();
}

void SetSendTracebackToRuntimeGlobal(RuntimeTracebackMode mode) {
  global_runtime_traceback_mode_.store(mode);
}

void SetSendTracebackToRuntimeThreadLocal(
    std::optional<RuntimeTracebackMode> mode) {
  runtime_traceback_mode_thread_local_ = mode;
}

}  // namespace jax
