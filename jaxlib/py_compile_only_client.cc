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
#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_device_list.h"
#include "jaxlib/py_executable.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/compile_only_ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/version.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/xla_data.pb.h"

namespace ifrt = xla::ifrt;
namespace nb = nanobind;

namespace jax {

nb_class_ptr<PyClient> CompileOnlyPyClient::Make(
    std::shared_ptr<xla::ifrt::PjRtTopology> topology) {
  auto client =
      nb::borrow<nb_class_ptr<PyClient>>(make_nb_class<CompileOnlyPyClient>(
          std::make_unique<xla::CompileOnlyIfRtClient>(std::move(topology))));
  CompileOnlyPyClient::Initialize(client);
  return client;
}

absl::StatusOr<nb_class_ptr<PyExecutable>> CompileOnlyPyClient::CompileUnloaded(
    MlirModule mlir_module, xla::ifrt::DeviceListRef executable_devices,
    xla::CompileOptions options) {
  mlir::ModuleOp module = unwrap(mlir_module);
  mlir::OwningOpRef<mlir::ModuleOp> clone(module.clone());
  module = *clone;
  ifrt::ExecutableRef ifrt_executable;
  {
    nb::gil_scoped_release gil_release;
    auto* ifrt_client =
        llvm::dyn_cast_or_null<xla::CompileOnlyIfRtClient>(this->ifrt_client());
    CHECK(ifrt_client) << "CompileOnlyPyClient requires ifrt_client be a "
                          "xla::CompileOnlyIfRtClient";

    auto xla_options = std::make_unique<ifrt::XlaCompileOptions>(
        options, std::move(executable_devices));
    TF_ASSIGN_OR_RETURN(
        ifrt_executable,
        ifrt_client->GetDefaultCompiler()->Compile(
            std::make_unique<xla::ifrt::HloProgram>(std::move(module)),
            ifrt_client->topology(), std::move(xla_options)));
  }
  return make_nb_class<PyExecutable>(ifrt_executable);
}

void CompileOnlyPyClient::Initialize(nb_class_ptr<PyClient> client) {
  PyClient::Initialize(client);
}

void CompileOnlyPyClient::Register(nb::module_& m) {
  nb::class_<CompileOnlyPyClient, PyClient>(m, "CompileOnlyPyClient")
      .def(
          "compile",
          [](CompileOnlyPyClient& self, MlirModule mlir_module,
             PyDeviceList& py_executable_devices, xla::CompileOptions options,
             std::vector<nb::capsule> host_callbacks) {
            ifrt::DeviceListRef executable_devices =
                xla::ValueOrThrow(py_executable_devices.ifrt_device_list());
            return xla::ValueOrThrow(
                self.CompileUnloaded(mlir_module, std::move(executable_devices),
                                     std::move(options)));
          },
          nb::arg("computation"), nb::arg("executable_devices"),
          nb::arg("compile_options") = xla::CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>(),
          nb::sig(
              // clang-format off
              "def compile("
              "self, "
              "computation: object, "
              "executable_devices: DeviceList, "
              "compile_options: CompileOptions = ..., "
              "host_callbacks: Sequence[typing_extensions.CapsuleType] = ..."
              ") -> Executable"
              // clang-format on
              ));
}

}  // namespace jax
