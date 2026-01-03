/* Copyright 2019 The JAX Authors

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

#include <Python.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "nanobind/nb_defs.h"
#include "nanobind/stl/function.h"  // IWYU pragma: keep
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/set.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/unordered_map.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/ffi.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_program.h"
#include "jaxlib/py_values.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "xla/python/pjrt_ifrt/transfer_server_interface.h"
#include "xla/python/version.h"
#include "xla/tsl/python/lib/core/numpy.h"  // NOLINT

#if defined(__linux__)
#include "gloo/transport/tcp/attr.h"
#include "gloo/transport/tcp/device.h"
#include "jaxlib/py_socket_transfer.h"
#include "xla/backends/cpu/collectives/gloo_collectives.h"
#include "xla/backends/cpu/collectives/gloo_kv_store.h"
#elif defined(__APPLE__)
#include "gloo/transport/uv/device.h"
#include "xla/backends/cpu/collectives/gloo_collectives.h"  // NOLINT
#include "xla/backends/cpu/collectives/gloo_kv_store.h"  // NOLINT
#endif  // defined(__linux__)

#if !defined(_WIN32) && !defined(PLATFORM_GOOGLE)
#include "xla/backends/cpu/collectives/mpi_collectives.h"
#endif  // !_WIN32 && !PLATFORM_GOOGLE

#include "jaxlib/call_location.h"
#include "jaxlib/config.h"
#include "jaxlib/custom_call_sharding.h"
#include "jaxlib/dlpack.h"
#include "jaxlib/guard_lib.h"
#include "jaxlib/jax_jit.h"
#include "jaxlib/mlir.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/partition_spec.h"
#include "jaxlib/pjit.h"
#include "jaxlib/pmap_lib.h"
#include "jaxlib/pprof_profile_builder.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_compile_only_client.h"
#include "jaxlib/py_device.h"
#include "jaxlib/py_device_list.h"
#include "jaxlib/py_executable.h"
#include "jaxlib/py_memory_space.h"
#include "jaxlib/python_ref_manager.h"
#include "jaxlib/pytree.h"
#include "jaxlib/sharding.h"
#include "jaxlib/traceback.h"
#include "jaxlib/xla_compiler.h"
#include "xla/hlo/builder/lib/approx_topk_shape.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/preemption/preemption_sync_manager.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/logging.h"  // IWYU pragma: keep
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/tsl/distributed_runtime/preemption/preemption_sync_manager.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/platform.h"

// TODO(phawkins): remove host_id properties after JAX is update to avoid them.

namespace nb = nanobind;

namespace jax {
namespace {

bool IsOptimizedBuild() {
#if NDEBUG
  return true;
#else
  return false;
#endif  // NDEBUG
}

// Is*san reports whether the build is under that particular sanitizer.
bool IsAsan() {
#if defined(ADDRESS_SANITIZER)
  return true;
#else  // defined(ADDRESS_SANITIZER)
  return false;
#endif
}

bool IsMsan() {
#if defined(MEMORY_SANITIZER)
  return true;
#else  // defined(MEMORY_SANITIZER)
  return false;
#endif
}

bool IsTsan() {
#if defined(THREAD_SANITIZER)
  return true;
#else  // defined(THREAD_SANITIZER)
  return false;
#endif
}

// IsSanitized reports whether the build is under any sanitizer.
bool IsSanitized() { return IsAsan() || IsMsan() || IsTsan(); }

}  // namespace

NB_MODULE(_jax, m) {
  // Initialize ABSL logging because code within XLA uses it.
#ifndef PLATFORM_GOOGLE
  xla::InitializeAbslLogging();
#endif  // PLATFORM_GOOGLE

  // We seem to get a fair number of leak warnings from nanobind. It's unclear
  // whether these are false positives or not.
  nb::set_leak_warnings(false);

  tsl::ImportNumpy();

  // Exceptions
  nb::exception<xla::XlaRuntimeError> xla_runtime_error(m, "JaxRuntimeError",
                                                        PyExc_RuntimeError);
  xla_runtime_error.attr("__doc__") = nb::str(
      "Runtime errors thrown by the JAX runtime. While the JAX runtime may "
      "raise other exceptions as well, most exceptions thrown by the runtime "
      "are instances of this class.");

  // Must be before PyClient.compile.
  xla::BuildXlaCompilerSubmodule(m);

  PyDevice::Register(m);
  PyMemorySpace::Register(m);
  PyClient::Register(m);

  nb::enum_<xla::ifrt::ArrayCopySemantics>(m, "ArrayCopySemantics",
                                           nb::is_arithmetic())
      .value("ALWAYS_COPY", xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
      .value("REUSE_INPUT", xla::ifrt::ArrayCopySemantics::kReuseInput)
      .value("DONATE_INPUT", xla::ifrt::ArrayCopySemantics::kDonateInput);

  nb::class_<xla::PjRtLayout>(m, "PjRtLayout")
      .def("__str__", &xla::PjRtLayout::ToString)
      .def("__eq__",
           [](const xla::PjRtLayout& layout, nb::object other) {
             return nb::isinstance<xla::PjRtLayout>(other) &&
                    layout == nb::cast<const xla::PjRtLayout&>(other);
           })
      .def("__hash__",
           [](const xla::PjRtLayout& layout) { return absl::HashOf(layout); })
      .def("_xla_layout", &xla::PjRtLayout::xla_layout)
      .def("__getstate__",
           [](const xla::PjRtLayout& layout) -> nb::tuple {
             absl::StatusOr<std::string> serialized = layout.Serialize();
             xla::ThrowIfError(serialized.status());
             return nb::make_tuple(
                 nb::bytes(serialized->data(), serialized->size()));
           })
      .def("__setstate__", [](xla::PjRtLayout* self, nb::tuple t) {
        nb::bytes serialized = nb::cast<nb::bytes>(t[0]);
        absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> layout =
            xla::PjRtLayout::Deserialize(
                std::string_view(serialized.c_str(), serialized.size()));
        xla::ThrowIfError(layout.status());
        new (self) xla::PjRtLayout((*layout)->xla_layout());
      });

  nb::class_<xla::cpu::CpuCollectives> cpu_collectives(m, "CpuCollectives");
  cpu_collectives
      .def("Init",
           [](xla::cpu::CpuCollectives*) {
             throw std::runtime_error("Init is not implemented");
           })
      .def("Finalize", [](xla::cpu::CpuCollectives*) {
        throw std::runtime_error("Finalize is not implemented");
      });

  m.def(
      "make_gloo_tcp_collectives",
      [](std::shared_ptr<xla::DistributedRuntimeClient> distributed_client,

         std::optional<std::string> hostname,
         std::optional<std::string> interface)
          -> std::shared_ptr<xla::cpu::CpuCollectives> {
#if defined(__linux__)
        std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;
        if (distributed_client != nullptr) {
          kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"cpu:");
        }
        auto gloo_kv_store =
            std::make_unique<xla::cpu::GlooKeyValueStore>(kv_store);
        auto tcp_attrs = gloo::transport::tcp::attr();
        if (hostname) {
          tcp_attrs.hostname = *hostname;
        }
        if (interface) {
          tcp_attrs.iface = *interface;
        }
        auto tcp_device = gloo::transport::tcp::CreateDevice(tcp_attrs);
        return std::make_shared<xla::cpu::GlooCollectives>(
            std::move(gloo_kv_store), std::move(tcp_device));
#elif defined(__APPLE__)
        std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;
        if (distributed_client != nullptr) {
          kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"cpu:");
        }
        auto gloo_kv_store =
            std::make_unique<xla::cpu::GlooKeyValueStore>(kv_store);
        auto uv_attrs = gloo::transport::uv::attr();
        if (hostname) {
          uv_attrs.hostname = *hostname;
        }
        if (interface) {
          uv_attrs.iface = *interface;
        }
        auto uv_device = gloo::transport::uv::CreateDevice(uv_attrs);
        return std::make_shared<xla::cpu::GlooCollectives>(
            std::move(gloo_kv_store), std::move(uv_device));
#else   // defined(__linux__)
        throw xla::XlaRuntimeError(
            "make_gloo_tcp_collectives only implemented for linux and macos");
#endif  // defined(__linux__)
      },
      nb::arg("distributed_client"), nb::arg("hostname").none() = std::nullopt,
      nb::arg("interface").none() = std::nullopt);

#if !defined(_WIN32) && !defined(PLATFORM_GOOGLE)
  nb::class_<xla::cpu::MpiCollectives> mpi_collectives(m, "MpiCollectives",
                                                       cpu_collectives);
  mpi_collectives.def("Init", &xla::cpu::MpiCollectives::Init);
  mpi_collectives.def("Finalize", &xla::cpu::MpiCollectives::Finalize);
  m.def("make_mpi_collectives",
        []() -> std::shared_ptr<xla::cpu::MpiCollectives> {
          return std::make_shared<xla::cpu::MpiCollectives>();
        });
#else   // !_WIN32 && !PLATFORM_GOOGLE
  m.def("make_mpi_collectives",
        []() -> std::shared_ptr<xla::cpu::CpuCollectives> {
          throw xla::XlaRuntimeError(
              "make_mpi_collectives is not implemented for Windows");
        });
#endif  // !_WIN32 && !PLATFORM_GOOGLE

  m.def(
      "get_tfrt_cpu_client",
      [](bool asynchronous,
         std::shared_ptr<xla::DistributedRuntimeClient> distributed_client,
         int node_id, int num_nodes,
         std::shared_ptr<xla::cpu::CpuCollectives> collectives,
         std::optional<int> num_devices,
         std::optional<int> get_local_topology_timeout_minutes,
         std::optional<int> get_global_topology_timeout_minutes,
         std::optional<xla::ifrt::TransferServerInterfaceFactory>
             transfer_server_factory) -> nb_class_ptr<PyClient> {
        std::unique_ptr<xla::ifrt::PjRtClient> ifrt_client;
        {
          nb::gil_scoped_release gil_release;
          xla::CpuClientOptions options;

          options.asynchronous = asynchronous;
          options.collectives = std::move(collectives);
          options.process_id = node_id;
          options.cpu_device_count = num_devices;
          std::unique_ptr<xla::PjRtClient> client =
              xla::ValueOrThrow(xla::GetPjRtCpuClient(std::move(options)));
          xla::ifrt::PjRtClient::CreateOptions ifrt_options;
          ifrt_options.pjrt_client =
              std::shared_ptr<xla::PjRtClient>(std::move(client));
          if (distributed_client != nullptr) {
            ifrt_options.kv_store =
                GetDistributedKeyValueStore(distributed_client,
                                            /*key_prefix=*/"cpu:");
            ifrt_options.process_id = node_id;
            ifrt_options.num_processes = num_nodes;
          }
          if (get_local_topology_timeout_minutes.has_value()) {
            ifrt_options.get_local_topology_timeout =
                absl::Minutes(*get_local_topology_timeout_minutes);
          }
          if (get_global_topology_timeout_minutes.has_value()) {
            ifrt_options.get_global_topology_timeout =
                absl::Minutes(*get_global_topology_timeout_minutes);
          }
          if (transfer_server_factory.has_value()) {
            ifrt_options.transfer_server_factory =
                std::move(transfer_server_factory->factory_fn);
          }
          ifrt_client = xla::ValueOrThrow(
              xla::ifrt::PjRtClient::Create(std::move(ifrt_options)));
        }
        return PyClient::Make(std::move(ifrt_client));
      },
      nb::arg("asynchronous") = true, nb::arg("distributed_client") = nullptr,
      nb::arg("node_id") = 0, nb::arg("num_nodes") = 1,
      nb::arg("collectives").none() =
          std::shared_ptr<xla::cpu::CpuCollectives>(),
      nb::arg("num_devices").none() = std::nullopt,
      nb::arg("get_local_topology_timeout_minutes").none() = std::nullopt,
      nb::arg("get_global_topology_timeout_minutes").none() = std::nullopt,
      nb::arg("transfer_server_factory").none() = std::nullopt);
  m.def("pjrt_plugin_loaded", [](std::string platform_name) -> bool {
    absl::StatusOr<const PJRT_Api*> pjrt_api = pjrt::PjrtApi(platform_name);
    return pjrt_api.ok();
  });
  m.def(
      "load_pjrt_plugin",
      [](std::string platform_name, std::optional<std::string> library_path,
         std::optional<nb::capsule> c_api) -> nb::capsule {
        if (library_path.has_value()) {
          const PJRT_Api* api = xla::ValueOrThrow(
              pjrt::LoadPjrtPlugin(platform_name, *library_path));
          return nb::capsule(absl::bit_cast<void*>(api), "pjrt_c_api");
        }
        if (std::string_view(c_api->name()) != "pjrt_c_api") {
          throw nb::value_error(
              "c_api argument to load_pjrt_plugin is not a pjrt_c_api "
              "capsule.");
        }
        xla::ThrowIfError(pjrt::SetPjrtApi(
            platform_name, static_cast<const PJRT_Api*>(c_api->data())));
        return *c_api;
      },
      nb::arg("platform_name"), nb::arg("library_path").none() = std::nullopt,
      nb::arg("c_api").none() = std::nullopt);
  m.def("pjrt_plugin_initialized", [](std::string platform_name) -> bool {
    return xla::ValueOrThrow(pjrt::IsPjrtPluginInitialized(platform_name));
  });
  m.def("initialize_pjrt_plugin", [](std::string platform_name) {
    return xla::ThrowIfError(pjrt::InitializePjrtPlugin(platform_name));
  });

  m.def(
      "get_c_api_client",
      [](std::string platform_name,
         const absl::flat_hash_map<std::string, xla::PjRtValueType>& options,
         std::shared_ptr<xla::DistributedRuntimeClient> distributed_client,
         std::optional<xla::ifrt::TransferServerInterfaceFactory>
             transfer_server_factory) -> nb_class_ptr<PyClient> {
        std::unique_ptr<xla::ifrt::PjRtClient> ifrt_client;
        {
          nb::gil_scoped_release gil_release;
          std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;
          if (distributed_client != nullptr) {
            kv_store = GetDistributedKeyValueStore(
                distributed_client,
                /*key_prefix=*/absl::StrCat(platform_name, ":"));
          }
          std::unique_ptr<xla::PjRtClient> c_api_client = xla::ValueOrThrow(
              xla::GetCApiClient(platform_name, options, kv_store));
          xla::ifrt::PjRtClient::CreateOptions ifrt_options;
          ifrt_options.pjrt_client =
              std::shared_ptr<xla::PjRtClient>(std::move(c_api_client));
          ifrt_options.kv_store = kv_store;
          ifrt_options.use_kv_store_for_topology_exchange = false;
          ifrt_options.distributed_client = distributed_client;
          if (transfer_server_factory.has_value()) {
            ifrt_options.transfer_server_factory =
                std::move(transfer_server_factory->factory_fn);
          }
          ifrt_client = xla::ValueOrThrow(
              xla::ifrt::PjRtClient::Create(std::move(ifrt_options)));
        }
        return PyClient::Make(std::move(ifrt_client));
      },
      nb::arg("platform_name"),
      nb::arg("options") =
          absl::flat_hash_map<std::string, xla::PjRtValueType>(),
      nb::arg("distributed_client").none() = nullptr,
      nb::arg("transfer_server_factory").none() = std::nullopt);
  // TODO(b/322357665): Delete this method after TPU plugin changes to use the
  // standard registration.
  m.def("get_default_c_api_topology",
        [](std::string platform_name, std::string topology_name,
           const absl::flat_hash_map<std::string, xla::PjRtValueType>& options)
            -> std::shared_ptr<xla::ifrt::Topology> {
          return std::make_shared<xla::ifrt::PjRtTopology>(xla::ValueOrThrow(
              xla::GetCApiTopology(platform_name, topology_name, options)));
        });
  m.def("get_c_api_topology",
        [](nb::capsule c_api, std::string topology_name,
           const absl::flat_hash_map<std::string, xla::PjRtValueType>& options)
            -> std::shared_ptr<xla::ifrt::Topology> {
          if (std::string_view(c_api.name()) != "pjrt_c_api") {
            throw nb::value_error(
                "Argument to get_c_api_topology was not a pjrt_c_api capsule.");
          }
          return std::make_shared<xla::ifrt::PjRtTopology>(xla::ValueOrThrow(
              xla::GetCApiTopology(static_cast<const PJRT_Api*>(c_api.data()),
                                   topology_name, options)));
        });
  m.def("get_topology_for_devices",
        [](const std::vector<nb_class_ptr<PyDevice>>& py_devices) {
          if (py_devices.empty()) {
            throw nb::value_error(
                "get_topology_for_devices requires >= 1 devices.");
          }
          auto client = py_devices[0]->client();
          absl::InlinedVector<xla::ifrt::Device*, 1> ifrt_devices;
          ifrt_devices.reserve(py_devices.size());
          for (const auto& py_device : py_devices) {
            if (py_device->client().get() != client.get()) {
              throw nb::value_error(
                  "devices passed to get_topology_for_devices come from "
                  "different clients.");
            }
            ifrt_devices.push_back(py_device->device());
          }
          xla::ifrt::DeviceListRef device_list = xla::ValueOrThrow(
              client->ifrt_client()->MakeDeviceList(ifrt_devices));
          return xla::ValueOrThrow(
              client->ifrt_client()->GetTopologyForDevices(device_list));
        });

  TF_CHECK_OK(PyArray::Register(m));
  PyDeviceList::Register(m);
  RegisterSharding(m);

  nb::class_<xla::CompiledMemoryStats>(m, "CompiledMemoryStats")
      .def_rw("generated_code_size_in_bytes",
              &xla::CompiledMemoryStats::generated_code_size_in_bytes)
      .def_rw("argument_size_in_bytes",
              &xla::CompiledMemoryStats::argument_size_in_bytes)
      .def_rw("output_size_in_bytes",
              &xla::CompiledMemoryStats::output_size_in_bytes)
      .def_rw("alias_size_in_bytes",
              &xla::CompiledMemoryStats::alias_size_in_bytes)
      .def_rw("temp_size_in_bytes",
              &xla::CompiledMemoryStats::temp_size_in_bytes)
      .def_rw("host_generated_code_size_in_bytes",
              &xla::CompiledMemoryStats::host_generated_code_size_in_bytes)
      .def_rw("host_argument_size_in_bytes",
              &xla::CompiledMemoryStats::host_argument_size_in_bytes)
      .def_rw("host_output_size_in_bytes",
              &xla::CompiledMemoryStats::host_output_size_in_bytes)
      .def_rw("host_alias_size_in_bytes",
              &xla::CompiledMemoryStats::host_alias_size_in_bytes)
      .def_rw("host_temp_size_in_bytes",
              &xla::CompiledMemoryStats::host_temp_size_in_bytes)
      .def_prop_ro("serialized_buffer_assignment_proto",
                   [](const xla::CompiledMemoryStats& cms) -> nb::bytes {
                     const std::string& s = cms.serialized_buffer_assignment;
                     return nb::bytes(s.data(), s.size());
                   })
      .def_rw("peak_memory_in_bytes",
              &xla::CompiledMemoryStats::peak_memory_in_bytes)
      .def("__str__", &xla::CompiledMemoryStats::DebugString);

  m.def("get_execution_stream_id", []() { return GetExecutionStreamId(); });
  m.def("set_execution_stream_id",
        [](int64_t id) { GetExecutionStreamId() = id; });

  PyLoadedExecutable::Register(m);
  PyExecuteResults::Register(m);
  PyToken::Register(m);
  PyShardedToken::Register(m);
  PyExecutable::Register(m);

  m.def("buffer_to_dlpack_managed_tensor",
        xla::ValueOrThrowWrapper(BufferToDLPackManagedTensor),
        nb::arg("buffer"), nb::arg("stream").none() = nb::none());
  m.def(
      "dlpack_managed_tensor_to_buffer",
      [](const nb::capsule& tensor, nb_class_ptr<PyDevice> device,
         std::optional<std::intptr_t> stream, std::optional<bool> copy) {
        return xla::ValueOrThrow(DLPackManagedTensorToBuffer(
            tensor, device->device(), device->client(), stream, copy));
      },
      nb::arg("dlpack"), nb::arg("device"), nb::arg("stream").none(),
      nb::arg("copy").none() = nb::none(),
      nb::sig(
          // clang-format off
      "def dlpack_managed_tensor_to_buffer("
      "dlpack: typing_extensions.CapsuleType, "
      "device: Device, "
      "stream: int | None, "
      "copy: bool | None = ..."
      ") -> ArrayImpl"
          // clang-format on
          ));
  m.def("cuda_array_interface_to_buffer",
        xla::ValueOrThrowWrapper(CudaArrayInterfaceToBuffer), nb::arg("cai"),
        nb::arg("gpu_backend").none() = nb::none(),
        nb::arg("device_id").none() = nb::none());

  nb::enum_<jax::RuntimeTracebackMode>(m, "RuntimeTracebackMode")
      .value("OFF", jax::RuntimeTracebackMode::kOff)
      .value("ON", jax::RuntimeTracebackMode::kOn)
      .value("FULL", jax::RuntimeTracebackMode::kFull);
  m.def("add_exclude_path", &jax::AddExcludePath,
        "Adds a path to exclude from tracebacks.");
  m.def("set_send_traceback_to_runtime_global",
        &jax::SetSendTracebackToRuntimeGlobal);
  m.def("set_send_traceback_to_runtime_thread_local",
        &jax::SetSendTracebackToRuntimeThreadLocal, nb::arg("mode").none());

  BuildConfigSubmodule(m);
  BuildIfrtProgramsSubmodule(m);
  BuildPytreeSubmodule(m);
  BuildGuardSubmodule(m);
  BuildJaxjitSubmodule(m);
  BuildPmapSubmodule(m);
  BuildPjitSubmodule(m);
  Traceback::Register(m);
  BuildMlirSubmodule(m);
  BuildCustomCallShardingPybindAPI(m);
  RegisterFfiApis(m);
#if defined(__linux__)
  aux::RegisterTransferServerTypes(m);
#endif  // defined(__linux__)

  nb::class_<xla::PreemptionSyncManager> preemption_sync_manager(
      m, "PreemptionSyncManager");
  preemption_sync_manager
      .def(
          "initialize",
          [](xla::PreemptionSyncManager& manager,
             xla::DistributedRuntimeClient* client) {
            xla::CoordinationServiceAgent* agent =
                xla::ValueOrThrow(client->GetCoordinationServiceAgent());
            xla::ThrowIfError(manager.Initialize(agent));
          },
          nb::arg("distributed_client"))
      .def("reached_sync_point",
           [](xla::PreemptionSyncManager& manager, int step_counter) {
             return manager.ReachedSyncPoint(step_counter);
           })
      .def("shutdown", [](xla::PreemptionSyncManager& manager) {
        nb::gil_scoped_release gil_release;
        manager.Shutdown();
      });
  m.def("create_preemption_sync_manager",
        []() { return xla::CreatePreemptionSyncManager(); });

  nb::class_<xla::DistributedRuntimeService> distributed_runtime_service(
      m, "DistributedRuntimeService");
  distributed_runtime_service.def("shutdown",
                                  &xla::DistributedRuntimeService::Shutdown,
                                  nb::call_guard<nb::gil_scoped_release>());
  nb::class_<xla::DistributedRuntimeClient> distributed_runtime_client(
      m, "DistributedRuntimeClient");
  distributed_runtime_client
      .def("connect",
           [](xla::DistributedRuntimeClient& self) {
             nb::gil_scoped_release gil_release;
             xla::ThrowIfError(self.Connect());
           })
      .def("shutdown",
           [](xla::DistributedRuntimeClient& self) {
             nb::gil_scoped_release gil_release;
             xla::ThrowIfError(self.Shutdown());
           })
      // This method assumes that the value is a Python string. Use
      // `blocking_key_value_get_bytes()` if key_value_set() was called with a
      // Python bytes object as its value.
      .def(
          "blocking_key_value_get",
          [](xla::DistributedRuntimeClient& client, std::string key,
             int64_t timeout_in_ms) {
            nb::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.BlockingKeyValueGet(
                key, absl::Milliseconds(timeout_in_ms)));
          },
          nb::arg("key"), nb::arg("timeout_in_ms"))
      // Same as `blocking_key_value_get()`, but retrieves the raw Python byte
      // values explicitly.
      .def(
          "blocking_key_value_get_bytes",
          [](xla::DistributedRuntimeClient& client, std::string key,
             int64_t timeout_in_ms) -> nb::bytes {
            std::string result;
            {
              nb::gil_scoped_release gil_release;
              result = xla::ValueOrThrow(client.BlockingKeyValueGet(
                  key, absl::Milliseconds(timeout_in_ms)));
            }
            return nb::bytes(result.data(), result.size());
          },
          nb::arg("key"), nb::arg("timeout_in_ms"))
      .def(
          "key_value_try_get",
          [](xla::DistributedRuntimeClient& client, std::string key) {
            nb::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.KeyValueTryGet(key));
          },
          nb::arg("key"))
      .def(
          "key_value_try_get_bytes",
          [](xla::DistributedRuntimeClient& client,
             std::string key) -> nb::bytes {
            std::string result;
            {
              nb::gil_scoped_release gil_release;
              result = xla::ValueOrThrow(client.KeyValueTryGet(key));
            }
            return nb::bytes(result.data(), result.size());
          },
          nb::arg("key"))
      .def(
          "key_value_increment",
          [](xla::DistributedRuntimeClient& client, std::string key,
             int64_t increment) {
            nb::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.KeyValueIncrement(key, increment));
          },
          nb::arg("key"), nb::arg("increment"))
      .def(
          "wait_at_barrier",
          [](xla::DistributedRuntimeClient& client, std::string barrier_id,
             int64_t timeout_in_ms,
             std::optional<std::vector<int32_t>> process_ids) {
            nb::gil_scoped_release gil_release;
            xla::ThrowIfError(client.WaitAtBarrier(
                barrier_id, absl::Milliseconds(timeout_in_ms), process_ids));
          },
          nb::arg("barrier_id"), nb::arg("timeout_in_ms"),
          nb::arg("process_ids") = std::nullopt)
      .def(
          "get_live_nodes",
          [](xla::DistributedRuntimeClient& client,
             std::vector<int32_t> process_ids) {
            nb::gil_scoped_release gil_release;
            // Python doesn't understand the IncarnationId type, so we convert
            // to regular integers before returning.
            absl::flat_hash_map<int32_t, tsl::IncarnationId> nodes =
                xla::ValueOrThrow(
                    client.GetLiveNodesWithIncarnations(process_ids));
            absl::flat_hash_map<int32_t, uint64_t> py_nodes;
            for (const auto& [task_id, incarnation_id] : nodes) {
              py_nodes[task_id] = incarnation_id.value();
            }
            return py_nodes;
          },
          nb::arg("process_ids"))
      // The key must be a string, but the value can either be a Python string
      // or bytes object.
      // With Python string values, use `key_value_set()` and
      // `blocking_key_value_get()`.
      // With Python byte object values, use `key_value_set()` and
      // `blocking_key_value_get_bytes()`.
      .def(
          "key_value_set",
          [](xla::DistributedRuntimeClient& client, std::string_view key,
             std::string_view value, bool allow_overwrite) {
            nb::gil_scoped_release gil_release;
            xla::ThrowIfError(client.KeyValueSet(key, value, allow_overwrite));
          },
          nb::arg("key"), nb::arg("value"), nb::arg("allow_overwrite") = false)
      // The key must be a string, but the value must a
      // Python bytes object.
      // Use `key_value_set_bytes()` and `blocking_key_value_get_bytes()`.
      .def(
          "key_value_set_bytes",
          [](xla::DistributedRuntimeClient& client, std::string_view key,
             nb::bytes value, bool allow_overwrite) {
            nb::gil_scoped_release gil_release;
            xla::ThrowIfError(client.KeyValueSet(
                key, std::string_view(value.c_str(), value.size()),
                allow_overwrite));
          },
          nb::arg("key"), nb::arg("value"), nb::arg("allow_overwrite") = false)
      // Assumes that all values in the directory are Python strings.
      .def(
          "key_value_dir_get",
          [](xla::DistributedRuntimeClient& client, std::string_view key) {
            nb::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.KeyValueDirGet(key));
          },
          nb::arg("key"))
      // Assumes that all values in the directory are Python byte objects.
      // Same as `key_value_dir_get()`, but retrieves Python byte values
      // explicitly.
      .def(
          "key_value_dir_get_bytes",
          [](xla::DistributedRuntimeClient& client, std::string_view key)
              -> std::vector<std::pair<std::string, nb::bytes>> {
            std::vector<std::pair<std::string, std::string>> result;
            {
              nb::gil_scoped_release gil_release;
              result = xla::ValueOrThrow(client.KeyValueDirGet(key));
            }
            // Convert std::string values to nb::bytes.
            std::vector<std::pair<std::string, nb::bytes>> kvs;
            kvs.reserve(result.size());
            for (auto& kv : result) {
              kvs.push_back(
                  std::pair(std::move(kv.first),
                            nb::bytes(kv.second.data(), kv.second.size())));
            }
            return kvs;
          },
          nb::arg("key"))
      .def(
          "key_value_delete",
          [](xla::DistributedRuntimeClient& client, std::string_view key) {
            nb::gil_scoped_release gil_release;
            return xla::ThrowIfError(client.KeyValueDelete(key));
          },
          nb::arg("key"));

  m.def(
      "get_distributed_runtime_service",
      [](std::string address, int num_nodes,
         std::optional<int> heartbeat_timeout,
         std::optional<int> cluster_register_timeout,
         std::optional<int> shutdown_timeout)
          -> std::unique_ptr<xla::DistributedRuntimeService> {
        xla::CoordinationServiceImpl::Options options;
        options.num_nodes = num_nodes;
        if (heartbeat_timeout.has_value()) {
          options.heartbeat_timeout = absl::Seconds(*heartbeat_timeout);
        }
        if (cluster_register_timeout.has_value()) {
          options.cluster_register_timeout =
              absl::Seconds(*cluster_register_timeout);
        }
        if (shutdown_timeout.has_value()) {
          options.shutdown_timeout = absl::Seconds(*shutdown_timeout);
        }
        std::unique_ptr<xla::DistributedRuntimeService> service =
            xla::ValueOrThrow(GetDistributedRuntimeService(address, options));
        return service;
      },
      nb::arg("address"), nb::arg("num_nodes"),
      nb::arg("heartbeat_timeout").none() = std::nullopt,
      nb::arg("cluster_register_timeout").none() = std::nullopt,
      nb::arg("shutdown_timeout").none() = std::nullopt);

  m.def(
      "get_distributed_runtime_client",
      [](std::string address, int node_id, std::optional<int> rpc_timeout,
         std::optional<int> init_timeout, std::optional<int> shutdown_timeout,
         std::optional<int> heartbeat_timeout,
         std::optional<nb::callable> missed_heartbeat_callback,
         std::optional<bool> shutdown_on_destruction,
         std::optional<bool> use_compression, std::optional<bool> recoverable)
          -> std::shared_ptr<xla::DistributedRuntimeClient> {
        bool compression = use_compression.value_or(false);
        xla::DistributedRuntimeClient::Options options;
        options.node_id = node_id;
        if (rpc_timeout.has_value()) {
          options.rpc_timeout = absl::Seconds(*rpc_timeout);
        }
        if (init_timeout.has_value()) {
          options.init_timeout = absl::Seconds(*init_timeout);
        }
        if (shutdown_timeout.has_value()) {
          options.shutdown_timeout = absl::Seconds(*shutdown_timeout);
        }
        if (heartbeat_timeout.has_value()) {
          options.heartbeat_timeout = absl::Seconds(*heartbeat_timeout);
        }
        if (missed_heartbeat_callback.has_value()) {
          options.missed_heartbeat_callback =
              nb::cast<std::function<void(absl::Status)>>(
                  *missed_heartbeat_callback);
        }
        if (shutdown_on_destruction.has_value()) {
          options.shutdown_on_destruction = *shutdown_on_destruction;
        }
        if (recoverable.has_value()) {
          options.recoverable = *recoverable;
        }
        return GetDistributedRuntimeClient(address, options, compression);
      },
      nb::arg("address"), nb::arg("node_id"),
      nb::arg("rpc_timeout").none() = std::nullopt,
      nb::arg("init_timeout").none() = std::nullopt,
      nb::arg("shutdown_timeout").none() = std::nullopt,
      nb::arg("heartbeat_timeout").none() = std::nullopt,
      nb::arg("missed_heartbeat_callback").none() = std::nullopt,
      nb::arg("shutdown_on_destruction").none() = std::nullopt,
      nb::arg("use_compression").none() = std::nullopt,
      nb::arg("recoverable").none() = std::nullopt);

  m.def("collect_garbage", []() { GlobalPyRefManager()->CollectGarbage(); });

  m.def("is_optimized_build", &IsOptimizedBuild);

  m.def("json_to_pprof_profile",
        xla::ValueOrThrowWrapper(xla::JsonToPprofProfile),
        "Encodes the JSON representation of a pprof Profile into its binary "
        "protocol buffer encoding.");
  m.def("pprof_profile_to_json",
        xla::ValueOrThrowWrapper(xla::PprofProfileToJson),
        "Decodes an uncompressed pprof Profile protocol buffer into a JSON "
        "representation");

  CompileOnlyPyClient::Register(m);
  nb::class_<xla::ifrt::Topology>(m, "DeviceTopology")
      .def("_make_compile_only_devices",
           [](std::shared_ptr<xla::ifrt::Topology> topology) {
             if (!llvm::isa<xla::ifrt::PjRtTopology>(*topology)) {
               throw xla::XlaRuntimeError("Only PjRtTopologies are supported.");
             }
             return CompileOnlyPyClient::Make(
                        std::dynamic_pointer_cast<xla::ifrt::PjRtTopology>(
                            topology))
                 ->Devices();
           })
      .def_prop_ro("platform",
                   [](xla::ifrt::Topology& topology) {
                     return topology.platform_name();
                   })
      .def_prop_ro("platform_version",
                   [](xla::ifrt::Topology& topology) {
                     return topology.platform_version();
                   })
      .def("serialize",
           [](xla::ifrt::Topology& topology) -> nb::bytes {
             std::string serialized = xla::ValueOrThrow(topology.Serialize());
             return nb::bytes(serialized.data(), serialized.size());
           })
      .def("__getattr__",
           [](xla::ifrt::Topology& topology,
              std::string_view name) -> nb::object {
             auto value =
                 topology.Attributes().Get<xla::ifrt::AttributeMap::Value>(
                     std::string(name));
             if (value.ok()) {
               return std::visit([](auto&& v) { return nb::cast(v.value); },
                                 *value);
             }
             throw nb::attribute_error(
                 absl::StrCat("Unknown attribute ", name).c_str());
           });

  nb::class_<xla::ifrt::TransferServerInterfaceFactory>(
      m, "TransferServerInterfaceFactory");

  m.def("is_asan", IsAsan);
  m.def("is_msan", IsMsan);
  m.def("is_tsan", IsTsan);
  m.def("is_sanitized", IsSanitized);

  m.def(
      "batched_device_put",
      [](nb::object aval, nb::object sharding, std::vector<nb::object> xs,
         std::vector<const PyDevice*> dst_devices, bool committed,
         bool force_copy,
         xla::PjRtClient::HostBufferSemantics host_buffer_semantics,
         std::optional<bool> enable_x64) -> nb::object {
        return xla::ValueOrThrow(PyArray::BatchedDevicePut(
            aval, sharding, std::move(xs), std::move(dst_devices), committed,
            force_copy, host_buffer_semantics,
            enable_x64.has_value() ? *enable_x64 : GetEnableX64()));
      },
      nb::arg("aval"), nb::arg("sharding"), nb::arg("xs"), nb::arg("devices"),
      nb::arg("committed") = true, nb::arg("force_copy") = false,
      nb::arg("host_buffer_semantics") =
          xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
      nb::arg("enable_x64").none() = std::nullopt);
  m.def(
      "reorder_shards",
      [](PyArray x, nb::object dst_sharding,
         xla::ifrt::ArrayCopySemantics array_copy_semantics) {
        return xla::ValueOrThrow(PyArray::ReorderShards(
            std::move(x), std::move(dst_sharding), array_copy_semantics));
      },
      nb::arg("x"), nb::arg("dst_sharding"), nb::arg("array_copy_semantics"));

  m.def("batched_block_until_ready", [](std::vector<nb::object> xs) {
    xla::ThrowIfError(PyArray::BatchedBlockUntilReady(std::move(xs)));
  });

  m.def("check_and_canonicalize_memory_kind", &CheckAndCanonicalizeMemoryKind,
        nb::arg("memory_kind").none(), nb::arg("device_list"));

  m.attr("ifrt_version_number") = JAX_IFRT_VERSION_NUMBER;

  m.def("approx_top_k_reduction_output_size",
        xla::ValueOrThrowWrapper(xla::ApproxTopKReductionOutputSize),
        nb::arg("input_size"), nb::arg("rank"), nb::arg("top_k"),
        nb::arg("recall_target"), nb::arg("aggregate_to_topk") = true,
        nb::arg("input_size_override") = -1);

  m.def("get_internal_device_put_info",
        []() { return DevicePutInfo::GetInfo(); });

  PartitionSpec::Register(m);

  m.def("set_typed_int_type", &SetTypedIntType);
  m.def("set_typed_float_type", &SetTypedFloatType);
  m.def("set_typed_complex_type", &SetTypedComplexType);
  m.def("set_typed_ndarray_type", &SetTypedNdArrayType);
}  // NOLINT(readability/fn_size)

}  // namespace jax
