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

#ifndef _WIN32

#include <cstdint>
#include <optional>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "nanobind/nanobind.h"
#include "jaxlib/triton/triton_dialect_capi.h"

namespace nb = nanobind;

namespace {

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyType;

struct PointerType : PyConcreteType<PointerType> {
  static constexpr IsAFunctionTy isaFunction = mlirTritonIsAPointer;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTritonPointerTypeGetTypeID;
  static constexpr const char* pyClassName = "PointerType";

  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](PyType& pointee_type, int64_t address_space) {
          return PointerType(
              pointee_type.getContext(),
              mlirTritonPointerTypeGet(pointee_type, address_space));
        },
        nb::arg("pointee_type"), nb::arg("address_space"),
        nb::sig("def get("
                "pointee_type: mlir.ir.Type, "
                "address_space: int"
                ") -> PointerType"),
        "Creates a PointerType type.");
    c.def_prop_ro("pointee_type", [](PointerType& self) {
      return PyType(self.getContext(),
                    mlirTritonPointerTypeGetPointeeType(self))
          .maybeDownCast();
    });
    c.def_prop_ro("address_space", [](PointerType& self) {
      return mlirTritonPointerTypeGetAddressSpace(self);
    });
  }
};

}  // namespace

NB_MODULE(_triton_ext, m) {
  nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"));

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

  PointerType::bind(m);

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

#else  // _WIN32

#include "nanobind/nanobind.h"

NB_MODULE(_triton_ext, m) {}

#endif  // _WIN32
