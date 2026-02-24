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

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "nanobind/nanobind.h"
#include "jaxlib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

namespace nb = nanobind;

namespace {

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyType;

struct Float8EXMYType : PyConcreteType<Float8EXMYType> {
  static constexpr IsAFunctionTy isaFunction = mlirTpuIsAFloat8EXMYType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTpuFloat8EXMYTypeGetTypeID;
  static constexpr const char* pyClassName = "Float8EXMYType";

  using Base::Base;

  static void bindDerived(ClassTy& c) {
    c.def_static(
        "get",
        [](const PyType& exmy_type, DefaultingPyMlirContext ctx) {
          return Float8EXMYType(
              ctx->getRef(),
              mlirTpuFloat8EXMYTypeGet(ctx.get()->get(), exmy_type));
        },
        nb::arg("exmy_type") = nb::none(), nb::arg("ctx").none() = nb::none(),
        nb::sig("def get("
                "exmy_type: jaxlib.mlir.ir.Type | None = None, "
                "ctx: jaxlib.mlir.ir.Context | None = None"
                ") -> Float8EXMYType"));
    c.def_prop_ro("underlying_type", [](Float8EXMYType& self) {
      return PyType(self.getContext(),
                    mlirTpuFloat8EXMYTypeGetUnderlyingType(self))
          .maybeDownCast();
    });
  }
};

}  // namespace

NB_MODULE(_tpu_ext, m) {
  nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"));

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

  m.def("private_set_arg_attr",
        [](MlirOperation op, unsigned i, std::string name, MlirAttribute attr) {
          mlirFuncSetArgAttr(
              op, i, mlirStringRefCreateFromCString(name.c_str()), attr);
        });

  Float8EXMYType::bind(m);
}
