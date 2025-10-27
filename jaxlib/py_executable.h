#include "jaxlib/py_user_context.h"
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

#ifndef JAXLIB_PY_EXECUTABLE_H_
#define JAXLIB_PY_EXECUTABLE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/traceback.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/xla_data.pb.h"

namespace jax {

class PyToken {
 public:
  PyToken() = default;
  explicit PyToken(xla::Future<> future) : future_(std::move(future)) {}

  static PyToken ReadyPyToken() {
    return PyToken(xla::Future<>(absl::OkStatus()));
  }

  absl::Status Await();

 private:
  xla::Future<> future_;
};

// PyShardedToken contains a PyToken for each device's execution.
class PyShardedToken {
 public:
  // Default construction creates a always-ready token.
  PyShardedToken() = default;
  explicit PyShardedToken(std::vector<xla::Future<>> futures)
      : futures_(std::move(futures)) {}

  PyToken GetPyToken(int device_id) const {
    if (futures_.empty()) return PyToken::ReadyPyToken();
    return PyToken(futures_.at(device_id));
  }

  absl::Status Await();

 private:
  std::vector<xla::Future<>> futures_;
};

class PyExecuteResults {
 public:
  PyExecuteResults(const nb_class_ptr<PyClient>& client,
                   std::vector<xla::ifrt::ArrayRef> ifrt_arrays,
                   int num_computations, PyShardedToken token,
                   xla::Future<> result_status = xla::Future<>());

  std::vector<std::vector<PyArray>> DisassembleIntoSingleDeviceArrays();

  std::vector<std::vector<PyArray>> DisassemblePrefixIntoSingleDeviceArrays(
      size_t n);

  std::vector<nanobind::object> ConsumeWithHandlers(
      std::vector<std::variant<const PyArrayResultHandler*, nanobind::object>>
          out_handlers);

  std::vector<xla::ifrt::ArrayRef> Consume();

  PyShardedToken ConsumeToken();

  size_t Size() const {
    CheckNotDisassembled();
    return ifrt_arrays_.size();
  }

  void CheckNotDisassembled() const;

 private:
  bool is_exploded_ = false;
  bool token_consumed_ = false;
  nb_class_ptr<PyClient> client_;
  std::vector<xla::ifrt::ArrayRef> ifrt_arrays_;
  int num_computations_;
  PyShardedToken token_;
  // Only set if the computation has tokens.
  xla::Future<> result_status_;
};

using ExecuteShardedArg = std::variant<PyArray, std::vector<PyArray>>;

// Thin Python wrapper around xla::ifrt::ExecutableRef. We use a wrapper class:
// a) Standardize around xla::ifrt::ExecutableRef, which is
//    std::shared_ptr<xla::ifrt::Executable>.
// b) Concrete subclasses of xla::ifrt::Executable have protected constructors.
class PyExecutable {
 public:
  PyExecutable(xla::ifrt::ExecutableRef ifrt_executable)
      : ifrt_executable_(std::move(ifrt_executable)) {};
  ~PyExecutable() = default;

  // NOTE(dsuo): For now, we only expose the xla::ifrt::Executable members
  // required by the Python bindings.
  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const {
    return ifrt_executable_->GetHloModules();
  }
  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const {
    return ifrt_executable_->GetOutputMemoryKinds();
  }
  std::optional<std::vector<xla::OpSharding>> GetOutputShardings() const {
    return ifrt_executable_->GetOutputShardings();
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const {
    return ifrt_executable_->GetParameterLayouts();
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const {
    return ifrt_executable_->GetOutputLayouts();
  }
  std::optional<std::vector<xla::OpSharding>> GetParameterShardings() const {
    return ifrt_executable_->GetParameterShardings();
  }
  absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats() const {
    return ifrt_executable_->GetCompiledMemoryStats();
  }
  absl::StatusOr<std::string> Serialize() const {
    return ifrt_executable_->Serialize();
  }
  absl::StatusOr<xla::ifrt::AttributeMap> GetCostAnalysis() const {
    return ifrt_executable_->GetCostAnalysis();
  }

 private:
  xla::ifrt::ExecutableRef ifrt_executable_;
};

// Python wrapper around xla::ifrt::LoadedExecutableRef. We use a wrapper class:
// a) to keep the PyClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
class PyLoadedExecutable {
 public:
  PyLoadedExecutable(nb_class_ptr<PyClient> client,
                     xla::ifrt::LoadedExecutableRef ifrt_loaded_executable,
                     std::optional<std::string> fingerprint);
  ~PyLoadedExecutable();

  nb_class_ptr<PyClient> client() const { return client_; }
  xla::ifrt::LoadedExecutable* ifrt_loaded_executable() const {
    return ifrt_loaded_executable_.get();
  }

  xla::ifrt::LoadedExecutableRef shared_ifrt_loaded_executable() {
    return ifrt_loaded_executable_;
  }

  std::vector<nb_class_ptr<PyDevice>> AddressableDevices() const;

  absl::StatusOr<std::string> GetHumanReadableProgramText() const {
    return ifrt_loaded_executable_->GetHumanReadableProgramText();
  }

  int64_t SizeOfGeneratedCodeInBytes() const {
    return ifrt_loaded_executable_->SizeOfGeneratedCodeInBytes();
  }

  absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats() const {
    nanobind::gil_scoped_release scope;
    return ifrt_loaded_executable_->GetCompiledMemoryStats();
  }

  absl::StatusOr<xla::ifrt::AttributeMap> GetCostAnalysis() const {
    return ifrt_loaded_executable_->GetCostAnalysis();
  }

  // Takes args indexed by argid then deviceid, transposes them, and passes to
  // xla::ifrt::LoadedExecutable::Execute. The result is similarly transposed
  // back into the argid,deviceid format. args is [num_args x num_devices].
  absl::StatusOr<PyExecuteResults> ExecuteSharded(
      std::vector<ExecuteShardedArg> args, bool with_tokens);

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> HloModules()
      const;

  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const;

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const;

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const;

  std::optional<std::vector<xla::OpSharding>> GetParameterShardings() const;

  std::optional<std::vector<xla::OpSharding>> GetOutputShardings() const;

  std::optional<Traceback> traceback() {
    return GetTraceback(ifrt_loaded_executable_->user_context().get());
  }

  xla::ifrt::LoadedExecutable* ifrt_executable() const {
    return ifrt_loaded_executable_.get();
  }

  // Short-term escape hatch to get xla::PjRtLoadedExecutable from PyExecutable.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  std::shared_ptr<xla::PjRtLoadedExecutable> shared_ptr_pjrt_executable() {
    auto* exec =
        llvm::dyn_cast_or_null<xla::ifrt::PjRtCompatibleLoadedExecutable>(
            ifrt_loaded_executable_.get());
    if (exec == nullptr) {
      throw xla::XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return exec->shared_ptr_pjrt_loaded_executable();
  }

  // Returns a template of execute options to pass to
  // `ifrt_executable()->Execute()`. Note that the caller may need to override
  // some options such as `launch_id` that change at each execution.
  const xla::ifrt::ExecuteOptions& options() const { return options_; }

  // Returns a unique launch ID to use for the next execution.
  int32_t GetNextLaunchId();

  const std::optional<std::string>& fingerprint() const { return fingerprint_; }

  // Keep `obj` alive as long as PyLoadedExecutable.
  void KeepAlive(nanobind::object obj);

 private:
  friend class PyClient;

  nb_class_ptr<PyClient> client_;
  xla::ifrt::LoadedExecutableRef ifrt_loaded_executable_;

  // Identical executables (i.e. representing the same program) will have the
  // same fingerprint. nullopt on platforms or executables where fingerprints
  // aren't implemented.
  std::optional<std::string> fingerprint_;

  // Launch ID to use for the next execution.
  std::atomic<uint32_t> next_launch_id_;

  // The options to pass to `executable_.Execute`.
  xla::ifrt::ExecuteOptions options_;

  // Python objects to keep alive as requested by user.
  std::vector<nanobind::object> keepalives_;

  // Doubly-linked list of all executables known to the client. Protected by the
  // GIL.
  PyLoadedExecutable* next_;
  PyLoadedExecutable* prev_;
};

}  // namespace jax

#endif  // JAXLIB_PY_EXECUTABLE_H_
