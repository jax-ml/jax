/* Copyright 2023 The JAX Authors

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

#include "jaxlib/py_compile_only_client.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device_list.h"
#include "jaxlib/py_executable.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/compile_only_ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/version.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace nb = nanobind;

namespace xla {

namespace {

class CompileOnlyPyClient : public PyClient {
 public:
  using PyClient::PyClient;

  static nb_class_ptr<PyClient> Make(
      std::shared_ptr<ifrt::PjRtTopology> topology) {
    auto client =
        nb::borrow<nb_class_ptr<PyClient>>(make_nb_class<CompileOnlyPyClient>(
            std::make_unique<CompileOnlyIfRtClient>(std::move(topology))));
    CompileOnlyPyClient::Initialize(client);
    return client;
  }

  absl::StatusOr<nb_class_ptr<PyExecutable>> CompileUnloaded(
      absl::string_view mlir_module, ifrt::DeviceListRef executable_devices,
      CompileOptions options) {
    ifrt::ExecutableRef ifrt_executable;
    {
      nb::gil_scoped_release gil_release;
      mlir::MLIRContext context;
      TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(mlir_module, context));
      auto* ifrt_client =
          llvm::dyn_cast_or_null<CompileOnlyIfRtClient>(this->ifrt_client());
      CHECK(ifrt_client) << "CompileOnlyPyClient requires ifrt_client be a "
                            "CompileOnlyIfRtClient";

      auto xla_options = std::make_unique<ifrt::XlaCompileOptions>(
          options, std::move(executable_devices));
      TF_ASSIGN_OR_RETURN(auto executable,
                          PjRtCompile(std::move(options), module.get(),
                                      *ifrt_client->topology().description()));
      TF_ASSIGN_OR_RETURN(ifrt_executable,
                          ifrt::PjRtExecutable::Create(std::move(executable)));
    }
    return make_nb_class<PyExecutable>(ifrt_executable);
  }

 private:
  static void Initialize(nb_class_ptr<PyClient> client) {
    PyClient::Initialize(client);
  }
};

}  // namespace

nb_class_ptr<PyClient> MakeCompileOnlyClient(
    std::shared_ptr<ifrt::PjRtTopology> topology) {
  return CompileOnlyPyClient::Make(std::move(topology));
}

void RegisterCompileOnlyClient(nb::module_& m) {
  nb::class_<CompileOnlyPyClient, PyClient>(m, "CompileOnlyPyClient")
      .def(
          "compile",
          [](CompileOnlyPyClient& self, nb::bytes mlir_module,
             jax::PyDeviceList& py_executable_devices, CompileOptions options,
             std::vector<nb::capsule> host_callbacks) {
            ifrt::DeviceListRef executable_devices =
                ValueOrThrow(py_executable_devices.ifrt_device_list());
            return ValueOrThrow(self.CompileUnloaded(
                absl::string_view(mlir_module.c_str(), mlir_module.size()),
                std::move(executable_devices), std::move(options)));
          },
          nb::arg("computation"), nb::arg("executable_devices"),
          nb::arg("compile_options") = CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>())
      .def("compile",
           ValueOrThrowWrapper(&CompileOnlyPyClient::CompileUnloaded),
           nb::arg("computation"), nb::arg("executable_devices"),
           nb::arg("compile_options") = CompileOptions());
}

}  // namespace xla
