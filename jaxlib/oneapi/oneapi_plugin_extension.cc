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

#include "absl/status/status.h"
#include "nanobind/nanobind.h"
#include "jaxlib/gpu/gpu_plugin_extension.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/pjrt/status_casters.h"

namespace nb = nanobind;

namespace jax {
namespace {

nb::dict FfiTypes() { return nb::dict(); }
nb::dict FfiHandlers() { return nb::dict(); }

}  // namespace

NB_MODULE(oneapi_plugin_extension, m) {
  BuildGpuPluginExtension(m);
  m.def("ffi_types", &FfiTypes);
  m.def("ffi_handlers", &FfiHandlers);

  m.def(
      "get_device_ordinal",
      [](std::intptr_t data_value) {
        // TODO(Intel-tf): Implement using SYCL USM pointer query API to retrieve
        // the device ordinal for the given device pointer.
        xla::ThrowIfError(
            absl::UnimplementedError("get_device_ordinal not yet implemented for OneAPI"));
      },
      nb::arg("data_value"));
}
}  // namespace jax
