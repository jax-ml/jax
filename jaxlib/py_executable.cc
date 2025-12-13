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

#include "jaxlib/py_executable.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/call_location.h"
#include "jaxlib/guard_lib.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device.h"
#include "jaxlib/py_user_context.h"
#include "jaxlib/traceback.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep
#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "xla/python/version.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/traceme.h"

namespace ifrt = xla::ifrt;

namespace {

uint64_t GetBaseLaunchId(std::optional<std::string> fingerprint,
                         ifrt::LoadedExecutableRef executable) {
  uint64_t ret = 0;
  if (fingerprint.has_value()) {
    ret = tsl::Fingerprint64(*fingerprint);
  }
  // Don't use the device fingerprint for executables running on single process.
  // Pmap and replicated executables for example will only populate the local
  // device to the loaded executable and all devices will have different devices
  // fingerprints.
#if JAX_IFRT_VERSION_NUMBER >= 37
  if (std::optional<ifrt::DeviceListRef> device_list = executable->devices();
      device_list.has_value() && !(*device_list)->IsFullyAddressable()) {
    ret += (*device_list)->fingerprint();
  }
#else
  if (!executable->devices()->IsFullyAddressable()) {
    ret += executable->devices()->fingerprint();
  }
#endif
  VLOG(1) << "Get base launch id: " << ret << " from fingerprint: "
          << (fingerprint.has_value()
                  ? absl::StrCat(tsl::Fingerprint64(*fingerprint))
                  : "<nullopt>");
  return ret;
}

}  // namespace

namespace nb = nanobind;

namespace jax {

// PyToken

absl::Status PyToken::Await() {
  CHECK(future_.IsValid());
  absl::Status status;
  {
    nb::gil_scoped_release gil_release;
    status = future_.Await();
  }
  // `status` originates from `ifrt::ExecuteResult::status`, which can reference
  // an asynchronously propagated `ifrt::UserContext` representing the context
  // of an error. We expand this future result right before returning it to
  // Python (outside of `nb::gil_scoped_release`) so that any attached user
  // context is appended to the status message.
  return xla::ifrt::ExpandUserContexts(std::move(status));
}

void PyToken::Register(nb::module_& m) {
  nb::class_<PyToken> token(m, "Token");
  token.def("block_until_ready",
            [](PyToken& self) { xla::ThrowIfError(self.Await()); });
}

// PyShardedToken

absl::Status PyShardedToken::Await() {
  absl::Status status = absl::OkStatus();
  {
    nb::gil_scoped_release gil_release;
    for (auto& future : futures_) {
      auto s = future.Await();
      if (!s.ok()) status = std::move(s);
    }
  }
  // `status` combines the statuses originating from
  // `ifrt::ExecuteResult::status`, which can reference an asynchronously
  // propagated `ifrt::UserContext` representing the context of an error. We
  // expand this future result right before returning it to Python (outside of
  // `nb::gil_scoped_release`) so that any attached user context is appended to
  // the status message.
  return xla::ifrt::ExpandUserContexts(std::move(status));
}

void PyShardedToken::Register(nb::module_& m) {
  nb::class_<PyShardedToken> sharded_token(m, "ShardedToken");
  sharded_token.def("block_until_ready", [](PyShardedToken& self) {
    xla::ThrowIfError(self.Await());
  });
  sharded_token.def("get_token", &PyShardedToken::GetPyToken);
}

// PyExecuteResults

namespace {

void PopulateExecuteShardedResults(const nb_class_ptr<PyClient>& client,
                                   std::vector<ifrt::ArrayRef> ifrt_arrays,
                                   const xla::Future<>& result_status,
                                   int num_computations,
                                   std::vector<std::vector<PyArray>>& outputs) {
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = ifrt_arrays.size();
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    xla::ifrt::UserContextScope user_context_scope(
        ifrt_arrays[buffer_id]->user_context());
    outputs[buffer_id].reserve(num_computations);
    auto exploded_arrays =
        ifrt_arrays[buffer_id]->DisassembleIntoSingleDeviceArrays(
            ifrt::ArrayCopySemantics::kReuseInput,
            ifrt::SingleDeviceShardSemantics::kAddressableShards);
    TF_CHECK_OK(exploded_arrays.status());
    for (auto& exploded_array : *exploded_arrays) {
      outputs[buffer_id].push_back(PyArray::MakeFromSingleDeviceArray(
          client, std::move(exploded_array), false, true, result_status));
    }
  }
}

}  // namespace

PyExecuteResults::PyExecuteResults(const nb_class_ptr<PyClient>& client,
                                   std::vector<ifrt::ArrayRef> ifrt_arrays,
                                   int num_computations, PyShardedToken token,
                                   xla::Future<> result_status)
    : client_(client),
      ifrt_arrays_(std::move(ifrt_arrays)),
      num_computations_(num_computations),
      token_(std::move(token)),
      result_status_(std::move(result_status)) {}

void PyExecuteResults::CheckNotDisassembled() const {
  if (is_exploded_) {
    throw nb::value_error("ExecuteResults already exploded.");
  }
}

std::vector<ifrt::ArrayRef> PyExecuteResults::Consume() {
  CheckNotDisassembled();
  is_exploded_ = true;
  return std::move(ifrt_arrays_);
}

PyShardedToken PyExecuteResults::ConsumeToken() {
  if (token_consumed_) {
    throw nb::value_error("ExecuteResults token already consumed.");
  }
  token_consumed_ = true;
  return std::move(token_);
}

std::vector<std::vector<PyArray>>
PyExecuteResults::DisassembleIntoSingleDeviceArrays() {
  std::vector<std::vector<PyArray>> outputs;
  PopulateExecuteShardedResults(
      client_, Consume(),
      result_status_.IsValid() ? result_status_ : xla::Future<>(),
      num_computations_, outputs);
  return outputs;
}

std::vector<std::vector<PyArray>>
PyExecuteResults::DisassemblePrefixIntoSingleDeviceArrays(size_t n) {
  CheckNotDisassembled();
  if (n > ifrt_arrays_.size()) {
    throw nb::value_error(
        absl::StrCat("In DisassemblePrefixIntoSingleDeviceArrays: ", n, " > ",
                     ifrt_arrays_.size())
            .c_str());
  }
  std::vector<ifrt::ArrayRef> ifrt_arrays;
  ifrt_arrays.reserve(ifrt_arrays_.size() - n);
  for (size_t i = n; i < ifrt_arrays_.size(); ++i) {
    ifrt_arrays.push_back(std::move(ifrt_arrays_[i]));
  }
  ifrt_arrays_.erase(ifrt_arrays_.begin() + n, ifrt_arrays_.end());
  std::swap(ifrt_arrays_, ifrt_arrays);
  std::vector<std::vector<PyArray>> outputs;
  PopulateExecuteShardedResults(
      client_, std::move(ifrt_arrays),
      result_status_.IsValid() ? result_status_ : xla::Future<>(),
      num_computations_, outputs);
  return outputs;
}

std::vector<nb::object> PyExecuteResults::ConsumeWithHandlers(
    std::vector<std::variant<const PyArrayResultHandler*, nb::object>>
        out_handlers,
    bool strict) {
  std::vector<nb::object> outputs;
  int num_output_buffers = out_handlers.size();
  std::vector<xla::ifrt::ArrayRef> ifrt_arrays;
  if (strict) {
    if (out_handlers.size() != ifrt_arrays_.size()) {
      throw nb::value_error(
          absl::StrCat("Mismatch between out_handlers and num_results: ",
                       out_handlers.size(), " vs ", ifrt_arrays_.size())
              .c_str());
    }
    ifrt_arrays = Consume();
  } else {
    if (out_handlers.size() > ifrt_arrays_.size()) {
      throw nb::value_error(
          absl::StrCat("Mismatch between out_handlers and num_results: ",
                       out_handlers.size(), " > ", ifrt_arrays_.size())
              .c_str());
    }
    CheckNotDisassembled();
    ifrt_arrays.reserve(ifrt_arrays_.size() - num_output_buffers);
    for (size_t i = num_output_buffers; i < ifrt_arrays_.size(); ++i) {
      ifrt_arrays.push_back(std::move(ifrt_arrays_[i]));
    }
    ifrt_arrays_.erase(ifrt_arrays_.begin() + ifrt_arrays_.size(),
                       ifrt_arrays_.end());
    std::swap(ifrt_arrays_, ifrt_arrays);
  }
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    auto& handler = out_handlers[buffer_id];
    xla::ifrt::UserContextScope user_context_scope(
        ifrt_arrays[buffer_id]->user_context());
    if (std::holds_alternative<const PyArrayResultHandler*>(handler)) {
      outputs.push_back(std::get<const PyArrayResultHandler*>(handler)->Call(
          client_, std::move(ifrt_arrays[buffer_id]),
          result_status_.IsValid() ? result_status_ : xla::Future<>()));
    } else {
      tsl::profiler::TraceMe traceme("ConsumeWithHandlers fallback.");
      auto disassembled_arrays =
          ifrt_arrays[buffer_id]->DisassembleIntoSingleDeviceArrays(
              ifrt::ArrayCopySemantics::kReuseInput,
              ifrt::SingleDeviceShardSemantics::kAddressableShards);
      TF_CHECK_OK(disassembled_arrays.status());
      nb::list bufs =
          nb::steal<nb::list>(PyList_New(disassembled_arrays->size()));
      int i = 0;
      for (auto& disassembled_array : *disassembled_arrays) {
        nb::object array = PyArray::MakeFromSingleDeviceArray(
            client_, std::move(disassembled_array), false, true,
            result_status_.IsValid() ? result_status_ : xla::Future<>());
        PyList_SET_ITEM(bufs.ptr(), i, array.release().ptr());
        ++i;
      }
      outputs.push_back(std::get<nb::object>(handler)(std::move(bufs)));
    }
  }
  return outputs;
}

void PyExecuteResults::Register(nb::module_& m) {
  nb::class_<PyExecuteResults>(m, "ExecuteResults")
      .def("__len__", [](PyExecuteResults& results) { return results.Size(); })
      .def("disassemble_into_single_device_arrays",
           &PyExecuteResults::DisassembleIntoSingleDeviceArrays)
      .def("disassemble_prefix_into_single_device_arrays",
           &PyExecuteResults::DisassemblePrefixIntoSingleDeviceArrays)
      .def("consume_with_handlers", &PyExecuteResults::ConsumeWithHandlers,
           nb::arg("out_handlers"), nb::arg("strict") = true)
      .def("consume_token", &PyExecuteResults::ConsumeToken);
}

// PyExecutable

void PyExecutable::Register(nb::module_& m) {
  nb::class_<PyExecutable>(m, "Executable")
      .def("hlo_modules",
           xla::ValueOrThrowWrapper(&PyExecutable::GetHloModules))
      .def("get_output_memory_kinds",
           xla::ValueOrThrowWrapper(&PyExecutable::GetOutputMemoryKinds))
      .def("get_output_shardings", &PyExecutable::GetOutputShardings)
      .def("get_parameter_layouts",
           xla::ValueOrThrowWrapper(&PyExecutable::GetParameterLayouts))
      .def("get_output_layouts",
           xla::ValueOrThrowWrapper(&PyExecutable::GetOutputLayouts))
      .def("get_parameter_shardings", &PyExecutable::GetParameterShardings)
      .def("get_compiled_memory_stats",
           xla::ValueOrThrowWrapper(&PyExecutable::GetCompiledMemoryStats))
      .def("serialize",
           [](const PyExecutable& exec) -> nb::bytes {
             std::string serialized = xla::ValueOrThrow(exec.Serialize());
             return nb::bytes(serialized.data(), serialized.size());
           })
      .def("cost_analysis", [](const PyExecutable& exec) {
        auto attrs = xla::ValueOrThrow(exec.GetCostAnalysis());
        return xla::ifrt::ToPjRtAttributeMap(std::move(attrs));
      });
}

// PyLoadedExecutable

PyLoadedExecutable::PyLoadedExecutable(
    nb_class_ptr<PyClient> client,
    ifrt::LoadedExecutableRef ifrt_loaded_executable,
    std::optional<std::string> fingerprint)
    : client_(std::move(client)),
      ifrt_loaded_executable_(std::move(ifrt_loaded_executable)),
      fingerprint_(std::move(fingerprint)),
      launch_id_key_(GetBaseLaunchId(fingerprint_, ifrt_loaded_executable_)) {
  CHECK(PyGILState_Check());
  if (ifrt_loaded_executable_->user_context() == nullptr &&
      Traceback::IsEnabled()) {
    throw nb::value_error(
        "Expecting an IFRT `LoadedExecutable` to have a user context, but got "
        "a null user context. Use `jax::PyUserContextScope` to set a user "
        "context for operations producing IFRT `LoadedExecutable`s.");
  }
  if (fingerprint_) {
    VLOG(1) << "Fingerprint for executable " << ifrt_loaded_executable_->name()
            << ": " << *fingerprint_;
  }
  nb::ft_lock_guard lock(client_->executables_mutex_);
  next_ = client_->executables_;
  client_->executables_ = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
}

PyLoadedExecutable::~PyLoadedExecutable() {
  CHECK(PyGILState_Check());
  nb::ft_lock_guard lock(client_->executables_mutex_);
  if (client_->executables_ == this) {
    client_->executables_ = next_;
  }
  if (prev_) {
    prev_->next_ = next_;
  }
  if (next_) {
    next_->prev_ = prev_;
  }
}

std::vector<nb_class_ptr<PyDevice>> PyLoadedExecutable::AddressableDevices()
    const {
  std::vector<nb_class_ptr<PyDevice>> devices;
  devices.reserve(ifrt_loaded_executable_->addressable_devices().size());
  for (ifrt::Device* device : ifrt_loaded_executable_->addressable_devices()) {
    devices.push_back(client_->GetPyDevice(device));
  }
  return devices;
}

namespace {

absl::StatusOr<PyExecuteResults> ExecuteShardedOnLocalDevicesInternal(
    const ifrt::ExecuteOptions& options, const nb_class_ptr<PyClient>& client,
    ifrt::LoadedExecutable* ifrt_loaded_executable,
    absl::Span<const PyArray> args,
    std::optional<std::vector<xla::Future<>>>& returned_futures) {
  std::vector<ifrt::ArrayRef> output_arrays;
  std::unique_ptr<tsl::Future<>> returned_future;
  int num_computations = ifrt_loaded_executable->addressable_devices().size();
  xla::Future<> result_status;
  {
    nb::gil_scoped_release gil_release;
    for (const auto& arg : args) {
      if (arg.num_addressable_shards() != num_computations) {
        return xla::InvalidArgument(
            "Expected args to execute_sharded_on_local_devices to have %d "
            "shards, got: [%s]",
            num_computations,
            absl::StrJoin(args, ", ", [](std::string* out, const PyArray& arg) {
              out->append(std::to_string(arg.num_addressable_shards()));
            }));
      }
    }
    std::vector<ifrt::ArrayRef> arg_arrays(args.size());
    absl::c_transform(args, arg_arrays.begin(),
                      [&](const PyArray& arg) mutable {
                        return tsl::FormRef(arg.ifrt_array());
                      });
    TF_ASSIGN_OR_RETURN(auto result, ifrt_loaded_executable->Execute(
                                         absl::MakeSpan(arg_arrays), options,
                                         /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
    // options.fill_status is only supposed to be true when the computation has
    // tokens.
    if (options.fill_status) {
      result_status = result.status;
      if (returned_futures.has_value()) {
        returned_futures->resize(num_computations, std::move(result.status));
      }
    }
  }

  // TODO(b/240696624): Although the PjRt interface require `returned_futures`
  // to be resized correctly if it is not nullopt, some implementation does not
  // implement this. So we have to check whether returned_futures is empty.
  // Remove this check once the implementation is fixed.
  auto py_sharded_token = returned_futures.has_value()
                              ? PyShardedToken(std::move(*returned_futures))
                              : PyShardedToken();

  return PyExecuteResults(client, std::move(output_arrays), num_computations,
                          std::move(py_sharded_token), result_status);
}

}  // namespace

absl::Mutex PyLoadedExecutable::next_launch_id_mutex_(absl::kConstInit);
absl::flat_hash_map<uint64_t, uint32_t>* PyLoadedExecutable::next_launch_id_ =
    new absl::flat_hash_map<uint64_t, uint32_t>();

absl::StatusOr<PyExecuteResults> PyLoadedExecutable::ExecuteSharded(
    std::vector<PyArray> args, bool with_tokens) {
  // Check if the thread guard is active and should prevent execution.
  // Skipped for portable executables.
  if (ifrt_loaded_executable_->devices().has_value()) {
    TF_RETURN_IF_ERROR(CheckThreadGuard(*ifrt_loaded_executable_->devices()));
  }

  xla::ifrt::ExecuteOptions options = options_;
  options.launch_id = GetNextLaunchId();
  options.fill_status = with_tokens;
  options.execution_stream_id = GetExecutionStreamId();
  if (options.execution_stream_id == 0) {
    options.execution_stream_id = tsl::Env::Default()->GetCurrentThreadId();
  }
  PyUserContextScope user_context_scope;
  PopulateCallLocation(options, xla::ifrt::UserContextScope::current().get());
  std::optional<std::vector<xla::Future<>>> returned_futures;
  if (with_tokens) {
    returned_futures.emplace();
  }
  absl::Span<const PyArray> span_args = args;
  return ExecuteShardedOnLocalDevicesInternal(options, client_,
                                              ifrt_loaded_executable_.get(),
                                              span_args, returned_futures);
}

absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>>
PyLoadedExecutable::HloModules() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetHloModules();
}

absl::StatusOr<std::vector<std::vector<std::string_view>>>
PyLoadedExecutable::GetOutputMemoryKinds() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetOutputMemoryKinds();
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
PyLoadedExecutable::GetParameterLayouts() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetParameterLayouts();
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
PyLoadedExecutable::GetOutputLayouts() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetOutputLayouts();
}

std::optional<std::vector<xla::OpSharding>>
PyLoadedExecutable::GetParameterShardings() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetParameterShardings();
}

std::optional<std::vector<xla::OpSharding>>
PyLoadedExecutable::GetOutputShardings() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetOutputShardings();
}

int32_t PyLoadedExecutable::GetNextLaunchId() {
  int32_t launch_id;
  {
    absl::MutexLock lock(next_launch_id_mutex_);
    auto it = next_launch_id_->find(launch_id_key_);
    if (it == next_launch_id_->end()) {
      uint32_t initial_value = static_cast<uint32_t>(launch_id_key_);
      it = next_launch_id_->emplace(launch_id_key_, initial_value).first;
    }
    launch_id = absl::bit_cast<int32_t>(it->second++);
  }
  VLOG(1) << "Launching executable " << ifrt_loaded_executable_->name()
          << " with launch ID: " << launch_id << " key: " << launch_id_key_;
#if JAX_IFRT_VERSION_NUMBER >= 37
  VLOG(2) << "Executable devices for launch ID " << launch_id << ": "
          << (ifrt_loaded_executable_->devices().has_value()
                  ? (*ifrt_loaded_executable_->devices())->DebugString()
                  : "<nullopt>");
#else
  VLOG(2) << "Executable devices for launch ID " << launch_id << ": "
          << ifrt_loaded_executable_->devices()->DebugString();
#endif
  return launch_id;
}

void PyLoadedExecutable::KeepAlive(nb::object obj) {
  keepalives_.push_back(std::move(obj));
}

void PyLoadedExecutable::Register(nb::module_& m) {
  nb::class_<PyLoadedExecutable>(m, "LoadedExecutable")
      .def_prop_ro("client", &PyLoadedExecutable::client)
      .def("local_devices", &PyLoadedExecutable::AddressableDevices)
      .def("get_hlo_text",
           xla::ValueOrThrowWrapper(
               &PyLoadedExecutable::GetHumanReadableProgramText))
      .def("serialize",
           [](const PyLoadedExecutable& exec) -> nb::bytes {
             std::string serialized =
                 xla::ValueOrThrow(exec.ifrt_loaded_executable()->Serialize());
             return nb::bytes(serialized.data(), serialized.size());
           })
      .def("size_of_generated_code_in_bytes",
           &PyLoadedExecutable::SizeOfGeneratedCodeInBytes)
      .def(
          "get_compiled_memory_stats",
          xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetCompiledMemoryStats))
      .def("execute_sharded",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::ExecuteSharded),
           nb::arg("arguments"), nb::arg("with_tokens") = false)
      .def("hlo_modules",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::HloModules))
      .def("get_output_memory_kinds",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetOutputMemoryKinds))
      .def("get_output_shardings", &PyLoadedExecutable::GetOutputShardings)
      .def("get_parameter_layouts",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetParameterLayouts))
      .def("get_output_layouts",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetOutputLayouts))
      .def("get_parameter_shardings",
           &PyLoadedExecutable::GetParameterShardings)
      .def("keep_alive", &PyLoadedExecutable::KeepAlive)
      .def("cost_analysis",
           [](const PyLoadedExecutable& self) {
             auto map = xla::ValueOrThrow(self.GetCostAnalysis());
             return xla::ifrt::ToPjRtAttributeMap(std::move(map));
           })
      .def_prop_ro("traceback", &PyLoadedExecutable::traceback)
      .def_prop_ro("fingerprint", [](PyLoadedExecutable* exec) -> nb::object {
        if (exec->fingerprint().has_value()) {
          return nb::bytes(exec->fingerprint()->data(),
                           exec->fingerprint()->size());
        } else {
          return nb::none();
        }
      });
}

}  // namespace jax
