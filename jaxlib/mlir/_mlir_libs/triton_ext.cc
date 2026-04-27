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

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyAttribute;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyType;

namespace jax {

class PyPointerType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyPointerType> {
 public:
  static constexpr const char* pyClassName = "PointerType";
  static bool isaFunction(MlirType t) { return mlirTritonIsAPointer(t); }
  static constexpr MlirTypeID (*getTypeIdFunction)() =
      mlirTritonPointerTypeGetTypeID;
  using Base::Base;
  static void bindDerived(ClassTy& cls) {
    cls.def_static(
        "get",
        [](PyType& pointee_type, int64_t address_space) {
          MlirContext ctx = mlirTypeGetContext(pointee_type.get());
          return PyPointerType(
              PyMlirContext::forContext(ctx),
              mlirTritonPointerTypeGet(pointee_type.get(), address_space));
        },
        nb::arg("pointee_type"), nb::arg("address_space"));
    cls.def_prop_ro("pointee_type", [](PyPointerType& self) {
      return mlirTritonPointerTypeGetPointeeType(self.get());
    });
    cls.def_prop_ro("address_space", [](PyPointerType& self) {
      return mlirTritonPointerTypeGetAddressSpace(self.get());
    });
  }
};

}  // namespace jax

NB_MODULE(_triton_ext, m) {
  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](PyMlirContext& context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__triton__();
        mlirDialectHandleRegisterDialect(dialect, context.get());
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context.get());
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  //
  // Types.
  //

  jax::PyPointerType::bind(m);

  //
  // Attributes.
  //

  m.def(
      "infer_reduce_op_encoding",
      [](PyAttribute& operandEncoding, int axis) -> std::optional<nb::object> {
        auto encoding =
            mlirTritonInferReduceOpEncoding(operandEncoding.get(), axis);
        if (mlirAttributeIsNull(encoding)) {
          return std::nullopt;
        }
        return nb::cast(
            PyAttribute::createFromCapsule(nanobind::steal<nanobind::object>(
                mlirPythonAttributeToCapsule(encoding))));
      });
}

#else  // _WIN32

#include "nanobind/nanobind.h"

NB_MODULE(_triton_ext, m) {}

#endif  // _WIN32
