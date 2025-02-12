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

#include <optional>

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"
#include "jaxlib/triton/triton_dialect_capi.h"

namespace nb = nanobind;

NB_MODULE(_triton_ext, m) {
  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__triton__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  //
  // Types.
  //

  mlir::python::nanobind_adaptors::mlir_type_subclass(m, "PointerType",
                                             mlirTritonIsAPointer)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType pointee_type, int64_t address_space) {
            return cls(mlirTritonPointerTypeGet(pointee_type, address_space));
          },
          nb::arg("cls"), nb::arg("pointee_type"), nb::arg("address_space"),
          "Creates a PointerType type.")
      .def_property_readonly("pointee_type", [](MlirType self) {
        return mlirTritonPointerTypeGetPointeeType(self);
      })
      .def_property_readonly("address_space", [](MlirType self) {
        return mlirTritonPointerTypeGetAddressSpace(self);
      });

  //
  // Attributes.
  //

  m.def("infer_reduce_op_encoding",
        [](MlirAttribute operandEncoding,
           int axis) -> std::optional<MlirAttribute> {
          auto encoding =
              mlirTritonInferReduceOpEncoding(operandEncoding, axis);
          if (mlirAttributeIsNull(encoding)) {
            return std::nullopt;
          }
          return encoding;
        });
}
