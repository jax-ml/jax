/* Copyright 2026 The JAX Authors.

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

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "nanobind/nanobind.h"
#include "jaxlib/thunky/dialect/capi.h"
#include "jaxlib/thunky/mlir_to_proto.h"
#include "xla/service/gpu/gpu_executable.pb.h"

namespace nb = nanobind;

namespace xla::gpu {
namespace {

template <typename T>
T ValueOrThrow(absl::StatusOr<T> status_or) {
  if (!status_or.ok()) {
    throw std::runtime_error(std::string(status_or.status().message()));
  }
  return *std::move(status_or);
}

MlirModule JaxExecutableToMlir(nb::bytes serialized_bytes,
                               MlirContext context_c) {
  absl::string_view bytes_view(serialized_bytes.c_str(),
                               serialized_bytes.size());
  GpuExecutableProto gpu_proto =
      ValueOrThrow(ParseGpuExecutableProto(bytes_view));
  mlir::OwningOpRef<mlir::ModuleOp> module = ValueOrThrow(
      GpuExecutableProtoToThunkyModule(gpu_proto, unwrap(context_c)));
  return wrap(module.release());
}

MlirType MlirBufferType(int64_t size, MlirContext context_c) {
  return mlirThunkyBufferTypeGet(context_c, size);
}

MlirType MlirTokenType(MlirContext context_c) {
  return mlirThunkyTokenTypeGet(context_c);
}

nb::bytes MlirModuleToGpuExecutableProto(MlirModule module_c) {
  mlir::ModuleOp module = unwrap(module_c);
  GpuExecutableProto gpu_proto =
      ValueOrThrow(LowerThunkyModuleToGpuExecutableProto(module));
  std::string serialized = gpu_proto.SerializeAsString();
  return nb::bytes(serialized.data(), serialized.size());
}

}  // namespace
}  // namespace xla::gpu

NB_MODULE(_thunky_ext, m) {
  nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"));

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__thunky__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
        mlir::DialectRegistry registry;
        mlir::func::registerInlinerExtension(registry);
        unwrap(context)->appendDialectRegistry(registry);
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("jax_executable_to_mlir", &xla::gpu::JaxExecutableToMlir,
        nb::arg("serialized_bytes"), nb::arg("context") = nb::none());
  m.def("mlir_module_to_gpu_executable_proto",
        &xla::gpu::MlirModuleToGpuExecutableProto, nb::arg("module"));
  m.def("mlir_buffer_type", &xla::gpu::MlirBufferType, nb::arg("size"),
        nb::arg("context") = nb::none());
  m.def("mlir_token_type", &xla::gpu::MlirTokenType,
        nb::arg("context") = nb::none());
}
