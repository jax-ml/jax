/* Copyright 2023 The JAX Authors.

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

#include <string>
#include <utility>

#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "jaxlib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

namespace nb = nanobind;

NB_MODULE(_tpu_ext, m) {
  mlirTpuRegisterMosaicSerdePass();

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__tpu__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("private_has_communication", [](MlirOperation op) {
    bool has_communication;
    bool has_custom_barrier;
    mlirTPUAnalyzePotentialCommunication(op, &has_communication,
                                         &has_custom_barrier);
    return std::make_pair(has_communication, has_custom_barrier);
  });

  // TODO(apaszke): All of those should be upstreamed to MLIR Python bindings.
  m.def("private_set_arg_attr",
        [](MlirOperation op, unsigned i, std::string name, MlirAttribute attr) {
          mlirFuncSetArgAttr(
              op, i, mlirStringRefCreateFromCString(name.c_str()), attr);
        });
}
