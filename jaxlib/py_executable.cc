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

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device.h"
#include "jaxlib/traceback.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/tsl/concurrency/ref_count.h"
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

uint32_t GetBaseLaunchId(std::optional<std::string> fingerprint,
                         ifrt::LoadedExecutableRef executable) {
  uint32_t ret = 0;
  if (fingerprint.has_value()) {
    ret = tsl::Fingerprint32(*fingerprint);
  }
  // Don't use the device fingerprint for executables running on single process.
  // Pmap and replicated executables for example will only populate the local
  // device to the loaded executable and all devices will have different devices
  // fingerprints.
  if (!executable->devices()->IsFullyAddressable()) {
    ret += executable->devices()->fingerprint();
  }
  return ret;
}

}  // namespace

namespace nb = nanobind;

namespace jax {

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

PyLoadedExecutable::PyLoadedExecutable(
    nb_class_ptr<PyClient> client,
    ifrt::LoadedExecutableRef ifrt_loaded_executable,
    std::optional<Traceback> traceback, std::optional<std::string> fingerprint)
    : client_(std::move(client)),
      ifrt_loaded_executable_(std::move(ifrt_loaded_executable)),
      traceback_(std::move(traceback)),
      fingerprint_(std::move(fingerprint)),
      next_launch_id_(GetBaseLaunchId(fingerprint_, ifrt_loaded_executable_)) {
  CHECK(PyGILState_Check());
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

static int GetNumDevices(const ExecuteShardedArg& arg) {
  if (std::holds_alternative<PyArray>(arg)) {
    return std::get<PyArray>(arg).num_addressable_shards();
  } else {
    return std::get<std::vector<PyArray>>(arg).size();
  }
}
static ifrt::ArrayRef GetIfRtArray(const ExecuteShardedArg& arg) {
  if (std::holds_alternative<PyArray>(arg)) {
    return tsl::FormRef(std::get<PyArray>(arg).ifrt_array());
  }
  auto& arg_vector = std::get<std::vector<PyArray>>(arg);

  // TODO(hyeontaek): This on-demand Array creation is not efficient and has
  // insufficient information about the shape (a dummy shape is used). This
  // should be removed if possible and only be used in the context where the
  // shape information is unused.
  std::vector<ifrt::ArrayRef> ifrt_arrays;
  ifrt_arrays.reserve(arg_vector.size());
  absl::InlinedVector<ifrt::Device*, 1> devices;
  devices.reserve(arg_vector.size());
  for (auto& arr : arg_vector) {
    CHECK_EQ(arr.ifrt_array()->sharding().devices()->size(), 1)
        << arr.ifrt_array()->sharding().DebugString();
    ifrt_arrays.push_back(tsl::FormRef(arr.ifrt_array()));
    devices.push_back(
        arr.ifrt_array()->sharding().devices()->devices().front());
  }
  CHECK(!ifrt_arrays.empty());
  // Use a dummy shape.
  // TODO(hyeontaek): Find a way to compute a correct shape.
  // TODO(yashkatariya): Plumb sharding or memory_kind here.
  ifrt::Client* client = ifrt_arrays.front()->client();
  absl::StatusOr<ifrt::DeviceListRef> device_list =
      client->MakeDeviceList(devices);
  TF_CHECK_OK(device_list.status());
  absl::Span<xla::ifrt::ArrayRef> arrays = absl::MakeSpan(ifrt_arrays);
  auto ifrt_array = client->AssembleArrayFromSingleDeviceArrays(
      arrays.at(0)->dtype(), std::move(ifrt_arrays.front()->shape()),
      ifrt::OpaqueSharding::Create(*std::move(device_list), ifrt::MemoryKind()),
      arrays, ifrt::ArrayCopySemantics::kReuseInput,
      ifrt::SingleDeviceShardSemantics::kAddressableShards);
  TF_CHECK_OK(ifrt_array.status());
  return *ifrt_array;
}

void PopulateExecuteShardedResults(const nb_class_ptr<PyClient>& client,
                                   std::vector<ifrt::ArrayRef> ifrt_arrays,
                                   const xla::PjRtFuture<>& result_status,
                                   int num_computations,
                                   std::vector<std::vector<PyArray>>& outputs) {
  auto traceback = Traceback::Get();
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = ifrt_arrays.size();
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    outputs[buffer_id].reserve(num_computations);
    auto exploded_arrays =
        ifrt_arrays[buffer_id]->DisassembleIntoSingleDeviceArrays(
            ifrt::ArrayCopySemantics::kReuseInput,
            ifrt::SingleDeviceShardSemantics::kAddressableShards);
    TF_CHECK_OK(exploded_arrays.status());
    for (auto& exploded_array : *exploded_arrays) {
      outputs[buffer_id].push_back(PyArray::MakeFromSingleDeviceArray(
          client, traceback, std::move(exploded_array), false, true,
          result_status));
    }
  }
}

void PopulateCallLocation(xla::ifrt::ExecuteOptions& options) {
  auto traceback = jax::Traceback::Get();
  if (!traceback.has_value()) {
    return;
  }

  std::string call_location_str;
  if (jax::IsFullCallLocationTracebackEnabled()) {
    call_location_str = traceback->ToString();
  } else {
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

absl::StatusOr<PyExecuteResults> ExecuteShardedOnLocalDevicesInternal(
    const ifrt::ExecuteOptions& options, const nb_class_ptr<PyClient>& client,
    ifrt::LoadedExecutable* ifrt_loaded_executable,
    absl::Span<const ExecuteShardedArg> args,
    std::optional<std::vector<xla::PjRtFuture<>>>& returned_futures) {
  std::vector<ifrt::ArrayRef> output_arrays;
  std::unique_ptr<ifrt::Future<>> returned_future;
  int num_computations = ifrt_loaded_executable->addressable_devices().size();
  xla::PjRtFuture<> result_status;
  {
    nb::gil_scoped_release gil_release;
    for (const auto& arg : args) {
      if (GetNumDevices(arg) != num_computations) {
        return xla::InvalidArgument(
            "Expected args to execute_sharded_on_local_devices to have %d "
            "shards, got: [%s]",
            num_computations,
            absl::StrJoin(args, ", ",
                          [](std::string* out, const ExecuteShardedArg& arg) {
                            out->append(std::to_string(GetNumDevices(arg)));
                          }));
      }
    }
    std::vector<ifrt::ArrayRef> arg_arrays(args.size());
    absl::c_transform(args, arg_arrays.begin(),
                      [&](const ExecuteShardedArg& arg) mutable {
                        return GetIfRtArray(arg);
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

PyExecuteResults::PyExecuteResults(const nb_class_ptr<PyClient>& client,
                                   std::vector<ifrt::ArrayRef> ifrt_arrays,
                                   int num_computations, PyShardedToken token,
                                   xla::PjRtFuture<> result_status)
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
      result_status_.IsValid() ? result_status_ : xla::PjRtFuture<>(),
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
      result_status_.IsValid() ? result_status_ : xla::PjRtFuture<>(),
      num_computations_, outputs);
  return outputs;
}

std::vector<nb::object> PyExecuteResults::ConsumeWithHandlers(
    std::vector<std::variant<const PyArrayResultHandler*, nb::object>>
        out_handlers) {
  std::vector<nb::object> outputs;
  auto ifrt_arrays = Consume();
  auto traceback = Traceback::Get();
  int num_output_buffers = ifrt_arrays.size();
  outputs.reserve(num_output_buffers);
  if (out_handlers.size() != num_output_buffers) {
    throw nb::value_error(
        absl::StrCat("Mismatch between out_handlers and num_results: ",
                     out_handlers.size(), " vs ", num_output_buffers)
            .c_str());
  }
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    auto& handler = out_handlers[buffer_id];
    if (std::holds_alternative<const PyArrayResultHandler*>(handler)) {
      outputs.push_back(std::get<const PyArrayResultHandler*>(handler)->Call(
          client_, std::move(ifrt_arrays[buffer_id]),
          result_status_.IsValid() ? result_status_ : xla::PjRtFuture<>()));
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
            client_, traceback, std::move(disassembled_array), false, true,
            result_status_.IsValid() ? result_status_ : xla::PjRtFuture<>());
        PyList_SET_ITEM(bufs.ptr(), i, array.release().ptr());
        ++i;
      }
      outputs.push_back(std::get<nb::object>(handler)(std::move(bufs)));
    }
  }
  return outputs;
}

absl::StatusOr<PyExecuteResults> PyLoadedExecutable::ExecuteSharded(
    std::vector<ExecuteShardedArg> args, bool with_tokens) {
  xla::ifrt::ExecuteOptions options = options_;
  options.launch_id = GetNextLaunchId();
  options.fill_status = with_tokens;
  options.execution_stream_id = GetExecutionStreamId();
  if (options.execution_stream_id == 0) {
    options.execution_stream_id = tsl::Env::Default()->GetCurrentThreadId();
  }
  PopulateCallLocation(options);
  std::optional<std::vector<xla::PjRtFuture<>>> returned_futures;
  if (with_tokens) {
    returned_futures.emplace();
  }
  absl::Span<const ExecuteShardedArg> span_args = args;
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
  return absl::bit_cast<int32_t>(
      next_launch_id_.fetch_add(1, std::memory_order_relaxed));
}

void PyLoadedExecutable::KeepAlive(nb::object obj) {
  keepalives_.push_back(std::move(obj));
}

}  // namespace jax
