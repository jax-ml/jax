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

#include "jaxlib/py_client.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/guard_lib.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/pprof_profile_builder.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_device.h"
#include "jaxlib/py_device_list.h"
#include "jaxlib/py_executable.h"
#include "jaxlib/py_host_callback.h"
#include "jaxlib/py_memory_space.h"
#include "jaxlib/py_user_context.h"
#include "jaxlib/py_values.h"
#include "jaxlib/python_ref_manager.h"
#include "jaxlib/sharding.h"
#include "jaxlib/traceback.h"
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/types.h"
#include "xla/python/version.h"
#include "xla/service/platform_util.h"  // IWYU pragma: keep
#include "xla/service/spmd/shardy/utils.h"  // IWYU pragma: keep
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace ifrt = xla::ifrt;
namespace nb = nanobind;

namespace jax {

/*static*/ nb_class_ptr<PyClient> PyClient::Make(
    std::shared_ptr<ifrt::Client> ifrt_client) {
  auto client = make_nb_class<PyClient>(std::move(ifrt_client));
  Initialize(client);
  return client;
}

PyClient::PyClient(std::shared_ptr<ifrt::Client> ifrt_client)
    : ifrt_client_(std::move(ifrt_client)),
      client_attributes_(ifrt_client_->Attributes()) {
  CHECK(ifrt_client_);
}

/* static */ void PyClient::Initialize(nb_class_ptr<PyClient> client) {
  for (ifrt::Device* device : client->ifrt_client()->devices()) {
    client->devices_[device] = make_nb_class<PyDevice>(client, device);

    for (ifrt::Memory* memory : device->Memories()) {
      auto& py_memory = client->memory_spaces_[memory];
      if (py_memory.get() == nullptr) {
        py_memory = make_nb_class<PyMemorySpace>(client, memory);
      }
    }
  }
}

PyClient::~PyClient() {
  nb::gil_scoped_release gil;
  ifrt_client_ = nullptr;
}

nb_class_ptr<PyDevice> PyClient::GetPyDevice(ifrt::Device* device) {
  auto& py_device = devices_[device];
  if (py_device.get() == nullptr) {
    py_device = make_nb_class<PyDevice>(
        nb::borrow<nb_class_ptr<PyClient>>(nb::find(this)), device);
  }
  return py_device;
}

nb_class_ptr<PyMemorySpace> PyClient::GetPyMemorySpace(
    ifrt::Memory* memory_space) {
  auto& py_memory = memory_spaces_[memory_space];
  if (py_memory.get() == nullptr) {
    py_memory = make_nb_class<PyMemorySpace>(
        nb::borrow<nb_class_ptr<PyClient>>(nb::find(this)), memory_space);
  }
  return py_memory;
}

std::vector<nb_class_ptr<PyDevice>> PyClient::Devices() {
  std::vector<nb_class_ptr<PyDevice>> devices;
  auto span = ifrt_client_->devices();
  devices.reserve(span.size());
  for (ifrt::Device* device : span) {
    devices.push_back(GetPyDevice(device));
  }
  return devices;
}

std::vector<nb_class_ptr<PyDevice>> PyClient::LocalDevices() {
  std::vector<nb_class_ptr<PyDevice>> devices;
  devices.reserve(ifrt_client_->addressable_devices().size());
  for (ifrt::Device* device : ifrt_client_->addressable_devices()) {
    devices.push_back(GetPyDevice(device));
  }
  return devices;
}

std::vector<nb_class_ptr<PyDevice>> PyClient::GetAllDevices() {
  std::vector<nb_class_ptr<PyDevice>> devices;
  devices.reserve(ifrt_client_->GetAllDevices().size());
  for (ifrt::Device* device : ifrt_client_->GetAllDevices()) {
    devices.push_back(GetPyDevice(device));
  }
  return devices;
}

absl::StatusOr<nb_class_ptr<PyDevice>> PyClient::DeviceFromLocalHardwareId(
    int local_hardware_id) {
  TF_ASSIGN_OR_RETURN(ifrt::Device * device,
                      ifrt_client_->LookupAddressableDevice(local_hardware_id));
  return GetPyDevice(device);
}

nb::typed<nb::list, PyLoadedExecutable> PyClient::LiveExecutables() {
  CHECK(PyGILState_Check());
  nb::ft_lock_guard lock(executables_mutex_);
  nb::list executables;
  for (PyLoadedExecutable* exec = executables_; exec; exec = exec->next_) {
    executables.append(nb::find(exec));
  }
  return executables;
}

absl::Status PyClient::Defragment() {
  CHECK(PyGILState_Check());
  if (!llvm::isa<ifrt::PjRtCompatibleClient>(ifrt_client_.get())) {
    return absl::UnimplementedError(
        "Defragmentation is not supported on this runtime.");
  }
  ifrt::PlatformId platform_id = ifrt_client_->platform_id();
  bool is_gpu_client = platform_id == xla::CudaId() ||
                       platform_id == xla::RocmId() ||
                       platform_id == xla::SyclId();

  if (!is_gpu_client) {
    return absl::UnimplementedError(
        "Defragmentation is not supported on this runtime.");
  }

  // TODO(b/399879011): This is a GPU-specific implementation of `Defragment`.
  // Ideally, this would be replaced with some kind of auto-defrag-on-OOM, or at
  // least would not live in this file.

  struct TmpBuffer {
    // Non-empty for buffers found in a PyArray_Storage. Multiple Arrays
    // can reference the same xla::PjRtBuffer.
    std::vector<std::shared_ptr<xla::PjRtBuffer>*> pjrt_buffer_ptrs;
    // TODO(skyewm): maybe use py_buffer's HostValue
    std::shared_ptr<xla::Literal> host_copy;
  };

  // Synchronously copy all buffers to host
  absl::flat_hash_map<xla::PjRtBuffer*, TmpBuffer> pjrt_buf_to_tmp_buffer;

  std::vector<PyArray> arrays = LiveArrays();
  for (const PyArray& array : arrays) {
    // TODO(hyeontaek): Support non-PjRt Arrays.
    // TODO(hyeontaek): Re-construct ifrt::Array with new xla::PjRtBuffer so
    // that std::shared_ptr<xla::PjRtBuffer> does not need to be updated
    // in-place.
    if (array.ifrt_array() == nullptr) {
      continue;
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(array.ifrt_array());
    if (arr == nullptr) {
      throw xla::XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend "
          "only.");
    }
    TF_ASSIGN_OR_RETURN(
        absl::Span<std::shared_ptr<xla::PjRtBuffer>> pjrt_buffers,
        arr->mutable_pjrt_buffers());
    for (int i = 0; i < pjrt_buffers.size(); ++i) {
      std::shared_ptr<xla::PjRtBuffer>& pjrt_buf_ptr = pjrt_buffers[i];
      if (pjrt_buf_ptr->IsDeleted()) {
        continue;
      }
      auto [iter, inserted] =
          pjrt_buf_to_tmp_buffer.insert({pjrt_buf_ptr.get(), TmpBuffer()});
      if (inserted) {
        TF_ASSIGN_OR_RETURN(iter->second.host_copy,
                            pjrt_buf_ptr->ToLiteralSync());
      }
      iter->second.pjrt_buffer_ptrs.push_back(&pjrt_buf_ptr);
    }
  }

  // All buffers successfully copied to host, delete on-device copies.
  //
  // Use blocking delete operation to ensure all memory is actually cleared
  // before we start rewriting buffers.
  //
  // Die instead of returning a bad status because program presumably can't
  // continue if we fail to reconstitute device buffers.
  for (const auto& it : pjrt_buf_to_tmp_buffer) {
    xla::PjRtBuffer* pjrt_buf = it.first;
    TF_CHECK_OK(pjrt_buf
                    ->ReleaseDeviceMemoryOwnership(
                        /*wait_for_operations_to_complete=*/true)
                    .status());
  }

  // Copy host copies back to device and update PyArrays in-place.
  for (auto& it : pjrt_buf_to_tmp_buffer) {
    xla::PjRtBuffer* pjrt_buf = it.first;
    TmpBuffer& tmp_buffer = it.second;
    std::unique_ptr<xla::PjRtBuffer> new_copy =
        pjrt_client()
            ->BufferFromHostLiteral(*tmp_buffer.host_copy,
                                    pjrt_buf->memory_space())
            .value();
    TF_CHECK_OK(new_copy->GetReadyFuture().Await());

    std::shared_ptr<xla::PjRtBuffer> new_pjrt_buf_ptr(new_copy.release());
    for (std::shared_ptr<xla::PjRtBuffer>* pjrt_buffer_ptr :
         tmp_buffer.pjrt_buffer_ptrs) {
      *pjrt_buffer_ptr = new_pjrt_buf_ptr;
    }
  }

  // TODO(skyewm): delete executables?
  return absl::OkStatus();
}

/* static */ absl::StatusOr<nb::object> PyClient::BufferFromPyval(
    nb_class_ptr<PyClient> client, nb::handle argument, ifrt::Device* device,
    bool force_copy, ifrt::Client::HostBufferSemantics host_buffer_semantics) {
  if (device == nullptr) {
    TF_RET_CHECK(!client->ifrt_client_->addressable_devices().empty());
    device = client->ifrt_client_->addressable_devices().front();
  }
  CHECK(device != nullptr);

  auto transfer_guard_formatter = [&argument, dst_device = device] {
    auto type = nb::cast<std::string>(nb::str(argument.type()));
    // Catch exceptions because shape and dtype properties convertible to str
    // are not guaranteed to present in an arbitrary argument.
    std::string shape;
    std::string dtype;
    try {
      shape =
          nb::cast<std::string>(nb::str(nb::object(argument.attr("shape"))));
    } catch (const std::exception& e) {
      shape = "<unknown>";
    }
    try {
      dtype =
          nb::cast<std::string>(nb::str(nb::object(argument.attr("dtype"))));
    } catch (const std::exception& e) {
      dtype = "<unknown>";
    }
    return absl::StrCat("type=", type, ", shape=", shape, ", dtype=", dtype,
                        ", dst_device=", dst_device->DebugString());
  };
  TF_RETURN_IF_ERROR(
      ApplyTransferGuardToHostToDevice(transfer_guard_formatter));

  TF_ASSIGN_OR_RETURN(ifrt::Device * found_device,
                      client->ifrt_client_->LookupDevice(device->Id()));
  if (found_device != device) {
    return xla::InvalidArgument(
        "Cannot copy value to device '%s' with '%s' backend",
        device->DebugString(), client->ifrt_client_->platform_name());
  }
  GlobalPyRefManager()->CollectGarbage();

  PyUserContextScope user_context_scope;
  DevicePutOptions options;
  options.squash_64bit_types = false;
  options.allow_zero_copy =
      (!force_copy && (host_buffer_semantics ==
                       ifrt::Client::HostBufferSemantics::kImmutableZeroCopy));
  TF_ASSIGN_OR_RETURN(DevicePutResult device_put_result,
                      DevicePutWithDevice(argument, client->ifrt_client_.get(),
                                          device, ifrt::MemoryKind(), options));
  TF_ASSIGN_OR_RETURN(ifrt::DeviceListRef device_list,
                      client->ifrt_client()->MakeDeviceList({device}));
  auto sharding =
      make_nb_class<SingleDeviceSharding>(client, std::move(device_list),
                                          /*memory_kind=*/nb::none());
  return PyArray::MakeFromIfrtArrayAndSharding(
      std::move(client), std::move(device_put_result.ifrt_array),
      std::move(sharding),
      /*weak_type=*/false, /*committed=*/false,
      /*skip_checks=*/true);
}

namespace {

// Makes IFRT `CompileOptions` from XLA `CompileOptions` and optional host
// callbacks.
std::unique_ptr<ifrt::CompileOptions> MakeIfrtCompileOptions(
    xla::CompileOptions options, ifrt::DeviceListRef executable_devices,
    std::vector<nb::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that
  // were created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(tsl::FormRef(
        static_cast<ifrt::LoadedHostCallback*>(host_callback.data())));
  }
  return std::make_unique<ifrt::XlaCompileOptions>(
      std::move(options), std::move(executable_devices),
      std::move(ifrt_loaded_host_callbacks));
}

// Makes IFRT `DeserializeExecutableOptions` from `xla::CompileOptions` and
// optional host callbacks.
std::unique_ptr<ifrt::DeserializeExecutableOptions>
MakeIfrtDeserializeExecutableOptions(std::optional<xla::CompileOptions> options,
                                     ifrt::DeviceListRef executable_devices,
                                     std::vector<nb::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that
  // were created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(tsl::FormRef(
        static_cast<ifrt::LoadedHostCallback*>(host_callback.data())));
  }
  return std::make_unique<ifrt::XlaDeserializeExecutableOptions>(
      std::move(options), std::move(executable_devices),
      std::move(ifrt_loaded_host_callbacks));
}

std::unique_ptr<ifrt::DeserializeExecutableOptions>
MakeIfrtDeserializeExecutableOptions(std::optional<xla::CompileOptions> options,
                                     ifrt::DeviceListRef executable_devices,
                                     std::vector<nb::callable> host_callbacks,
                                     ifrt::Client* ifrt_client) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  for (auto& host_callback : host_callbacks) {
    auto callback = tsl::MakeRef<PyFfiLoadedHostCallback>(
        ifrt_client, std::move(host_callback));
    ifrt_loaded_host_callbacks.push_back(callback);
  }
  return std::make_unique<ifrt::XlaDeserializeExecutableOptions>(
      std::move(options), std::move(executable_devices),
      std::move(ifrt_loaded_host_callbacks));
}

}  // namespace

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>>
PyClient::CompileAndLoadIfrtProgram(
    nb_class_ptr<PyClient> client, std::unique_ptr<ifrt::Program> ifrt_program,
    std::unique_ptr<ifrt::CompileOptions> ifrt_options) {
  auto* pjrt_compatible_client =
      llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(
          client->ifrt_client_.get());
  auto* ifrt_xla_options =
      llvm::dyn_cast_or_null<ifrt::XlaCompileOptions>(ifrt_options.get());
  // For XLA programs, pass allocated device memory size to compile options for
  // pjrt compatible backends.
  if (pjrt_compatible_client != nullptr && ifrt_xla_options != nullptr) {
    xla::CompileOptions& options = ifrt_xla_options->compile_options;
    auto addressable_devices =
        pjrt_compatible_client->pjrt_client()->addressable_devices();
    if (!addressable_devices.empty()) {
      int device_ordinal = options.executable_build_options.device_ordinal();
      if (device_ordinal < 0) {
        device_ordinal = 0;
      }
      CHECK_LT(device_ordinal, addressable_devices.size());
      auto stats = addressable_devices[device_ordinal]->GetAllocatorStats();
      if (stats.ok() && stats->bytes_limit) {
        options.executable_build_options.set_device_memory_size(
            *stats->bytes_limit);
      }
    }

    if (pjrt_compatible_client->pjrt_client()->key_value_store().has_value()) {
      options.executable_build_options.set_key_value_store(
          *pjrt_compatible_client->pjrt_client()->key_value_store());
    }
  }

  PyUserContextScope user_context_scope;
  ifrt::LoadedExecutableRef ifrt_loaded_executable;
  std::optional<std::string> fingerprint;
  absl::Status compile_status;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        ifrt_loaded_executable,
        client->ifrt_client_->GetDefaultCompiler()->CompileAndLoad(
            std::move(ifrt_program), std::move(ifrt_options)));
    compile_status = ifrt_loaded_executable->GetReadyFuture().Await();
    if (compile_status.ok()) {
      TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
    }
  }
  if (!compile_status.ok()) {
    // `compile_status.status()` can reference an asynchronously propagated
    // `ifrt::UserContext` representing the context of an error. We expand this
    // future result right before returning it to Python (outside of
    // `nb::gil_scoped_release`) so that any attached user context is appended
    // to the status message.
    return xla::ifrt::ExpandUserContexts(std::move(compile_status));
  }
  return make_nb_class<PyLoadedExecutable>(std::move(client),
                                           std::move(ifrt_loaded_executable),
                                           std::move(fingerprint));
}

/* static */ absl::StatusOr<nb_class_ptr<PyExecutable>> PyClient::Compile(
    nb_class_ptr<PyClient> client, mlir::ModuleOp module,
    ifrt::DeviceListRef executable_devices, xla::CompileOptions options) {
  mlir::OwningOpRef<mlir::ModuleOp> clone(module.clone());
  module = *clone;
  ifrt::ExecutableRef ifrt_executable;
  {
    TF_ASSIGN_OR_RETURN(
        auto topology,
        client->ifrt_client()->GetTopologyForDevices(executable_devices));
    auto xla_options = std::make_unique<ifrt::XlaCompileOptions>(
        options, std::move(executable_devices));
    TF_ASSIGN_OR_RETURN(
        ifrt_executable,
        client->ifrt_client()->GetDefaultCompiler()->Compile(
            std::make_unique<xla::ifrt::HloProgram>(std::move(module)),
            *topology, std::move(xla_options)));
  }
  return make_nb_class<PyExecutable>(ifrt_executable);
}

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>>
PyClient::CompileAndLoad(nb_class_ptr<PyClient> client, mlir::ModuleOp module,
                         ifrt::DeviceListRef executable_devices,
                         xla::CompileOptions options,
                         std::vector<nb::capsule> host_callbacks) {
  mlir::OwningOpRef<mlir::ModuleOp> clone(module.clone());
  module = *clone;
  // TODO(b/420837831): Remove this once we don't need to fall back to GSPMD.
  if (options.executable_build_options.use_shardy_partitioner() &&
      xla::sdy::hasGspmdAttrsOrOps(module)) {
    LOG(WARNING)
        << "Module has GSPMD attrs or ops, but Shardy is enabled. Disabling "
           "Shardy and falling back to using GSPMD propagation.";
    options.executable_build_options.set_use_shardy_partitioner(false);
    if (xla::sdy::hasShardyMesh(module)) {
      // Shardy is not enabled, but the module has shardy ops. Likely due to
      // export loading a GSPMD checkpoint. Fall back to GSPMD.
      TF_RETURN_IF_ERROR(xla::ExportShardyForGSPMD(module));
    }
  }
  options.allow_in_place_mlir_modification = true;  // We just cloned the module
  return CompileAndLoadIfrtProgram(
      client, std::make_unique<xla::ifrt::HloProgram>(std::move(module)),
      MakeIfrtCompileOptions(std::move(options), std::move(executable_devices),
                             std::move(host_callbacks)));
}

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>>
PyClient::CompileAndLoad(nb_class_ptr<PyClient> client, mlir::ModuleOp module,
                         ifrt::DeviceListRef executable_devices,
                         xla::CompileOptions options,
                         std::vector<nb::callable> host_callbacks) {
  mlir::OwningOpRef<mlir::ModuleOp> clone(module.clone());
  module = *clone;
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that
  // were created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()`.
  for (auto& host_callback : host_callbacks) {
    auto callback = tsl::MakeRef<PyFfiLoadedHostCallback>(
        client->ifrt_client(), std::move(host_callback));
    ifrt_loaded_host_callbacks.push_back(callback);
  }
  auto compile_options = std::make_unique<ifrt::XlaCompileOptions>(
      std::move(options), std::move(executable_devices),
      std::move(ifrt_loaded_host_callbacks));
  return CompileAndLoadIfrtProgram(
      client, std::make_unique<xla::ifrt::HloProgram>(module),
      std::move(compile_options));
}

absl::StatusOr<nb::bytes> PyClient::SerializeExecutable(
    const PyLoadedExecutable& executable) const {
  TF_ASSIGN_OR_RETURN(auto serialized,
                      executable.ifrt_loaded_executable()->Serialize());
  return nb::bytes(serialized.data(), serialized.size());
}

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>>
PyClient::DeserializeExecutable(nb_class_ptr<PyClient> client,
                                nb::bytes serialized,
                                ifrt::DeviceListRef executable_devices,
                                std::optional<xla::CompileOptions> options,
                                std::vector<nb::capsule> host_callbacks) {
  ifrt::LoadedExecutableRef ifrt_loaded_executable;
  std::optional<std::string> fingerprint;
  auto ifrt_deserialize_options = MakeIfrtDeserializeExecutableOptions(
      std::move(options), std::move(executable_devices),
      std::move(host_callbacks));
  PyUserContextScope user_context_scope;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        ifrt_loaded_executable,
        client->ifrt_client_->GetDefaultCompiler()->DeserializeLoadedExecutable(
            std::string_view(serialized.c_str(), serialized.size()),
            std::move(ifrt_deserialize_options)));
  }
  TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
  return make_nb_class<PyLoadedExecutable>(std::move(client),
                                           std::move(ifrt_loaded_executable),
                                           std::move(fingerprint));
}

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>>
PyClient::DeserializeExecutable(nb_class_ptr<PyClient> client,
                                nb::bytes serialized,
                                ifrt::DeviceListRef executable_devices,
                                std::optional<xla::CompileOptions> options,
                                std::vector<nb::callable> host_callbacks) {
  ifrt::LoadedExecutableRef ifrt_loaded_executable;
  std::optional<std::string> fingerprint;
  auto ifrt_deserialize_options = MakeIfrtDeserializeExecutableOptions(
      std::move(options), std::move(executable_devices),
      std::move(host_callbacks), client->ifrt_client());
  PyUserContextScope user_context_scope;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        ifrt_loaded_executable,
        client->ifrt_client_->GetDefaultCompiler()->DeserializeLoadedExecutable(
            std::string_view(serialized.c_str(), serialized.size()),
            std::move(ifrt_deserialize_options)));
  }
  TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
  return make_nb_class<PyLoadedExecutable>(std::move(client),
                                           std::move(ifrt_loaded_executable),
                                           std::move(fingerprint));
}

namespace {

struct HeapProfileKey {
  std::optional<Traceback> traceback;
  int64_t size;
  xla::PjRtDevice* device;
  bool operator==(const HeapProfileKey& other) const;
};

bool HeapProfileKey::operator==(const HeapProfileKey& other) const {
  if (size != other.size || device != other.device) {
    return false;
  }
  if ((traceback.has_value()) != (other.traceback.has_value())) {
    return false;
  }
  if (traceback.has_value() && traceback->not_equal(*other.traceback)) {
    return false;
  }
  return true;
}

template <typename H>
H AbslHashValue(H h, const HeapProfileKey& key) {
  if (key.traceback) {
    h = H::combine(std::move(h), nb::hash(*key.traceback));
  }
  h = H::combine(std::move(h), key.size, key.device);
  return h;
}

}  // namespace

absl::StatusOr<nb::bytes> PyClient::HeapProfile() {
  CHECK(PyGILState_Check());
  absl::flat_hash_set<xla::PjRtBuffer*> buffer_set;
  absl::flat_hash_map<HeapProfileKey, int64_t> entries;

  auto add_buffer_to_profile = [&](xla::PjRtBuffer* buffer,
                                   std::optional<Traceback> traceback) {
    // We only wish to count each xla::PjRtBuffer once, even though they may be
    // shared by multiple PyArrays.
    if (!buffer->IsDeleted() && buffer_set.insert(buffer).second) {
      TF_ASSIGN_OR_RETURN(size_t size, buffer->GetOnDeviceSizeInBytes());
      HeapProfileKey key{traceback, static_cast<int64_t>(size),
                         buffer->device()};
      ++entries[key];
    }
    return absl::OkStatus();
  };

  std::vector<PyArray> arrays = LiveArrays();
  for (const PyArray& array : arrays) {
    if (array.ifrt_array() == nullptr) {
      continue;
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(array.ifrt_array());
    // TODO(hyeontaek): Support non-PjRt Arrays.
    if (arr == nullptr) {
      throw xla::XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend "
          "only.");
    }
    for (const auto& buffer : arr->pjrt_buffers()) {
      TF_RETURN_IF_ERROR(
          add_buffer_to_profile(buffer.get(), array.traceback()));
    }
  }

  for (PyLoadedExecutable* executable = executables_; executable;
       executable = executable->next_) {
    HeapProfileKey key{executable->traceback(),
                       executable->SizeOfGeneratedCodeInBytes(), nullptr};
    ++entries[key];
  }

  xla::PprofProfileBuilder builder;
  auto* allocations = builder.profile().add_sample_type();
  allocations->set_type(builder.StringId("allocations"));
  allocations->set_unit(builder.StringId("count"));
  auto* space = builder.profile().add_sample_type();
  space->set_type(builder.StringId("space"));
  space->set_unit(builder.StringId("bytes"));

  const int kind_string_id = builder.StringId("kind");
  const int buffer_string_id = builder.StringId("buffer");
  const int executable_string_id = builder.StringId("executable");
  const int device_string_id = builder.StringId("device");
  for (const auto& entry : entries) {
    auto* sample = builder.profile().add_sample();
    if (entry.first.traceback) {
      for (const auto& frame : entry.first.traceback->RawFrames()) {
        sample->add_location_id(builder.LocationId(frame.code, frame.lasti));
      }
    }
    sample->add_value(entry.second);
    sample->add_value(entry.first.size * entry.second);

    auto* kind_label = sample->add_label();
    kind_label->set_key(kind_string_id);
    if (entry.first.device) {
      kind_label->set_str(buffer_string_id);
      auto* device_label = sample->add_label();
      device_label->set_key(device_string_id);
      std::string device_label_str(entry.first.device->DebugString());
      device_label->set_str(builder.StringId(device_label_str));
    } else {
      kind_label->set_str(executable_string_id);
    }
  }
  std::string serialized = builder.profile().SerializeAsString();
  return nb::bytes(serialized.data(), serialized.size());
}

absl::StatusOr<nb::object> PyClient::MakePythonCallbackUsingHostSendAndRecv(
    nb::callable callable, absl::Span<xla::Shape const> operand_shapes,
    absl::Span<xla::Shape const> result_shapes,
    absl::Span<uint16_t const> send_channel_ids,
    absl::Span<uint16_t const> recv_channel_ids, nb::callable serializer) {
  TF_ASSIGN_OR_RETURN(
      auto loaded_host_callback,
      PyHostSendAndRecvLoadedHostCallback::Create(
          ifrt_client(), std::move(callable), operand_shapes, result_shapes,
          send_channel_ids, recv_channel_ids, std::move(serializer)));
  nb::capsule callback_capsule(
      loaded_host_callback.release(), [](void* ptr) noexcept {
        static_cast<ifrt::LoadedHostCallback*>(ptr)->DropRef();
      });
  return callback_capsule;
}

/* static */ int PyClient::tp_traverse(PyObject* self, visitproc visit,
                                       void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) {
    return 0;
  }
  PyClient* c = nb::inst_ptr<PyClient>(self);
  for (const auto& [ifrt_device, py_device] : c->devices_) {
    Py_VISIT(py_device.ptr());
  }
  for (const auto& [ifrt_memory, py_memory] : c->memory_spaces_) {
    Py_VISIT(py_memory.ptr());
  }
  return 0;
}

/* static */ int PyClient::tp_clear(PyObject* self) {
  PyClient* c = nb::inst_ptr<PyClient>(self);
  absl::flat_hash_map<ifrt::Device*, nb_class_ptr<PyDevice>> devices;
  std::swap(devices, c->devices_);
  absl::flat_hash_map<ifrt::Memory*, nb_class_ptr<PyMemorySpace>> memory_spaces;
  std::swap(memory_spaces, c->memory_spaces_);
  return 0;
}

PyType_Slot PyClient::slots_[] = {
    {Py_tp_traverse, (void*)PyClient::tp_traverse},
    {Py_tp_clear, (void*)PyClient::tp_clear},
    {0, nullptr},
};

/* static */ void PyClient::Register(nb::module_& m) {
  nb::enum_<xla::PjRtClient::HostBufferSemantics>(m, "HostBufferSemantics")
      .value("IMMUTABLE_ONLY_DURING_CALL",
             xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall)
      .value("IMMUTABLE_UNTIL_TRANSFER_COMPLETES",
             xla::PjRtClient::HostBufferSemantics::
                 kImmutableUntilTransferCompletes)
      .value("ZERO_COPY",
             xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy);

  nb::class_<PyClient> py_local_client(m, "Client", nb::is_weak_referenceable(),
                                       nb::type_slots(PyClient::slots_));
  py_local_client.def_prop_ro("platform", &PyClient::platform_name)
      .def_prop_ro("_raw_platform", &PyClient::raw_platform_name)
      .def_prop_ro("platform_version", &PyClient::platform_version)
      .def_prop_ro("runtime_type", &PyClient::runtime_type)
      .def("device_count", &PyClient::device_count)
      .def("local_device_count", &PyClient::addressable_device_count)
      .def("devices", &PyClient::Devices)
      .def("local_devices", &PyClient::LocalDevices)
      // TODO(hyeontaek): Remove this method once we have a unified API for
      // enumerating devices with different criteria.
      .def("_get_all_devices", &PyClient::GetAllDevices)
      .def("device_from_local_hardware_id",
           xla::ValueOrThrowWrapper(&PyClient::DeviceFromLocalHardwareId))
      .def("live_executables", &PyClient::LiveExecutables)
      .def("live_arrays", &PyClient::LiveArrays)
      .def("live_buffers", &PyClient::LiveArrays)
      .def("process_index", &PyClient::process_index)
      .def("host_id", &PyClient::process_index)
      .def("task_id", &PyClient::process_index)
      .def(
          "buffer_from_pyval",
          [](nb_class_ptr<PyClient> client, nb::handle argument,
             PyDevice* device, bool force_copy,
             xla::PjRtClient::HostBufferSemantics host_buffer_semantics) {
            return xla::ValueOrThrow(
                PyClient::BufferFromPyval(std::move(client), argument,
                                          device ? device->device() : nullptr,
                                          force_copy, host_buffer_semantics));
          },
          nb::arg("argument"), nb::arg("device").none() = nullptr,
          nb::arg("force_copy") = false,
          nb::arg("host_buffer_semantics") =
              xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy)
      .def(
          "compile",
          [](nb_class_ptr<PyClient> client, MlirModule mlir_module,
             PyDeviceList& py_executable_devices, xla::CompileOptions options) {
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(py_executable_devices.ifrt_device_list());
            return xla::ValueOrThrow(PyClient::Compile(
                std::move(client), unwrap(mlir_module),
                std::move(executable_devices), std::move(options)));
          },
          nb::arg("computation"), nb::arg("executable_devices"),
          nb::arg("compile_options") = xla::CompileOptions(),
          nb::sig(
              // clang-format off
              "def compile("
              "self, "
              "computation: object, "
              "executable_devices: DeviceList, "
              "compile_options: CompileOptions = ..."
              ") -> Executable"
              // clang-format on
              ))
      .def(
          "compile_and_load",
          [](nb_class_ptr<PyClient> client, MlirModule mlir_module,
             PyDeviceList& py_executable_devices, xla::CompileOptions options,
             std::vector<nb::capsule> host_callbacks) {
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(py_executable_devices.ifrt_device_list());
            return xla::ValueOrThrow(PyClient::CompileAndLoad(
                std::move(client), unwrap(mlir_module),
                std::move(executable_devices), std::move(options),
                std::move(host_callbacks)));
          },
          nb::arg("computation"), nb::arg("executable_devices"),
          nb::arg("compile_options") = xla::CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>(),
          nb::sig(
              // clang-format off
              "def compile_and_load("
              "self, "
              "computation: object, "
              "executable_devices: DeviceList, "
              "compile_options: CompileOptions = ..., "
              "host_callbacks: Sequence[typing_extensions.CapsuleType] = ..."
              ") -> LoadedExecutable"
              // clang-format on
              ))
      .def(
          "compile_and_load",
          [](nb_class_ptr<PyClient> client, MlirModule mlir_module,
             PyDeviceList& py_executable_devices, xla::CompileOptions options,
             std::vector<nb::callable> host_callbacks) {
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(py_executable_devices.ifrt_device_list());
            return xla::ValueOrThrow(PyClient::CompileAndLoad(
                std::move(client), unwrap(mlir_module),
                std::move(executable_devices), std::move(options),
                std::move(host_callbacks)));
          },
          nb::arg("computation"), nb::arg("executable_devices"),
          nb::arg("compile_options") = xla::CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::callable>(),
          nb::sig(
              // clang-format off
              "def compile_and_load("
              "self, "
              "computation: object, "
              "executable_devices: DeviceList, "
              "compile_options: CompileOptions = ..., "
              "host_callbacks: Sequence[Callable[..., typing.Any]] = ..."
              ") -> LoadedExecutable"
              // clang-format on
              ))
      // The following two overloads are for users of deprecated APIs who call
      // `backend.compile` but do not have visibility to `DeviceList`.
      .def(
          "compile_and_load",
          [](nb_class_ptr<PyClient> client, nb::bytes module_str,
             nb::sequence& py_executable_devices, xla::CompileOptions options) {
            mlir::MLIRContext context;
            mlir::OwningOpRef<mlir::ModuleOp> module =
                xla::ValueOrThrow(xla::ParseMlirModuleString(
                    std::string_view(module_str.c_str(), module_str.size()),
                    context));
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(PyDeviceList(nb::tuple(py_executable_devices))
                                      .ifrt_device_list());
            return xla::ValueOrThrow(PyClient::CompileAndLoad(
                std::move(client), *module, std::move(executable_devices),
                std::move(options), std::vector<nb::capsule>()));
          },
          nb::arg("computation"), nb::arg("executable_devices"),
          nb::arg("compile_options") = xla::CompileOptions())
      .def(
          "compile_and_load",
          [](nb_class_ptr<PyClient> client, std::string module_str,
             nb::sequence& py_executable_devices, xla::CompileOptions options) {
            mlir::MLIRContext context;
            mlir::OwningOpRef<mlir::ModuleOp> module = xla::ValueOrThrow(
                xla::ParseMlirModuleString(module_str, context));

            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(PyDeviceList(nb::tuple(py_executable_devices))
                                      .ifrt_device_list());
            return xla::ValueOrThrow(PyClient::CompileAndLoad(
                std::move(client), *module, std::move(executable_devices),
                std::move(options), std::vector<nb::capsule>()));
          },
          nb::arg("computation"), nb::arg("executable_devices"),
          nb::arg("compile_options") = xla::CompileOptions())
      .def("compile_ifrt_program",
           xla::ValueOrThrowWrapper(PyClient::CompileAndLoadIfrtProgram))
      .def("compile_and_load_ifrt_program",
           xla::ValueOrThrowWrapper(PyClient::CompileAndLoadIfrtProgram))
      .def("serialize_executable",
           xla::ValueOrThrowWrapper(&PyClient::SerializeExecutable))
      .def(
          "deserialize_executable",
          [](nb_class_ptr<PyClient> client, nb::bytes serialized,
             PyDeviceList& py_executable_devices,
             std::optional<xla::CompileOptions> options,
             std::vector<nb::capsule> host_callbacks) {
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(py_executable_devices.ifrt_device_list());
            return xla::ValueOrThrow(PyClient::DeserializeExecutable(
                std::move(client), std::move(serialized),
                std::move(executable_devices), std::move(options),
                std::move(host_callbacks)));
          },
          nb::arg("serialized"), nb::arg("executable_devices"),
          nb::arg("compile_options").none() = nb::none(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>())
      .def(
          "deserialize_executable",
          [](nb_class_ptr<PyClient> client, nb::bytes serialized,
             jax::PyDeviceList& py_executable_devices,
             std::optional<xla::CompileOptions> options,
             std::vector<nb::callable> host_callbacks) {
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(py_executable_devices.ifrt_device_list());
            return xla::ValueOrThrow(PyClient::DeserializeExecutable(
                std::move(client), std::move(serialized),
                std::move(executable_devices), std::move(options),
                std::move(host_callbacks)));
          },
          nb::arg("serialized"), nb::arg("executable_devices"),
          nb::arg("compile_options").none() = nb::none(),
          nb::arg("host_callbacks") = std::vector<nb::callable>())
      // The following overload is for users of deprecated APIs who call
      // `deserialize_executable` but do not have visibility to `DeviceList`.
      .def(
          "deserialize_executable",
          [](nb_class_ptr<PyClient> client, nb::bytes serialized,
             nb::sequence& py_executable_devices,
             std::optional<xla::CompileOptions> options) {
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(PyDeviceList(nb::tuple(py_executable_devices))
                                      .ifrt_device_list());
            return xla::ValueOrThrow(PyClient::DeserializeExecutable(
                std::move(client), std::move(serialized),
                std::move(executable_devices), std::move(options),
                std::vector<nb::capsule>()));
          },
          nb::arg("serialized"), nb::arg("executable_devices"),
          nb::arg("compile_options").none() = nb::none())
      .def("heap_profile", xla::ValueOrThrowWrapper(&PyClient::HeapProfile))
      // TODO(zhangqiaorjc): Experimental.
      .def("defragment",
           [](PyClient& self) { xla::ThrowIfError(self.Defragment()); })
      .def("make_python_callback_from_host_send_and_recv",
           xla::ValueOrThrowWrapper(
               &PyClient::MakePythonCallbackUsingHostSendAndRecv),
           nb::arg("callable"), nb::arg("operand_shapes"),
           nb::arg("result_shapes"), nb::arg("send_channel_ids"),
           nb::arg("recv_channel_ids"),
           nb::arg("serializer").none() = nb::none())
      .def(
          "get_default_layout",
          [](PyClient& self, xla::nb_dtype dtype, nb::sequence shard_shape,
             nb_class_ptr<PyDevice> device)
              -> std::shared_ptr<const xla::PjRtLayout> {
            ifrt::DType ifrt_type = xla::ValueOrThrow(DtypeToIfRtDType(dtype));
            std::vector<int64_t> dims =
                xla::SequenceToVector<int64_t>(shard_shape);
            return xla::ValueOrThrow(self.ifrt_client()->GetDefaultPjRtLayout(
                ifrt_type, dims, device->device(), xla::ifrt::MemoryKind()));
          },
          nb::arg("dtype"), nb::arg("shard_shape"), nb::arg("device"))
      .def("__getattr__",
           [](PyClient& client, std::string_view name) -> nb::object {
             auto value =
                 client.Attributes().Get<xla::ifrt::AttributeMap::Value>(
                     std::string(name));
             if (value.ok()) {
               return std::visit([](auto&& v) { return nb::cast(v.value); },
                                 *value);
             }
             throw nb::attribute_error(
                 absl::StrCat("Unknown attribute ", name).c_str());
           });
}

}  // namespace jax
