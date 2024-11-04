/* Copyright 2024 The JAX Authors.

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

// clang-format: off
// pybind11 must be included before mlir/Bindings/Python/PybindAdaptors.h,
// otherwise this code will not build on Windows.
#include "pybind11/pybind11.h"
// clang-format: on

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"  // IWYU pragma: keep
#include "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h"

PYBIND11_MODULE(_mosaic_gpu_ext, m, py::mod_gil_not_used()) {
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__mosaic_gpu__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}
