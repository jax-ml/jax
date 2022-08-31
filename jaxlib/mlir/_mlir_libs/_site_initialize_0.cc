// Registers MLIR dialects used by JAX.
// This module is called by mlir/__init__.py during initialization.

#include "mlir-c/Dialect/Func.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_site_initialize_0, m) {
  m.doc() = "Registers MLIR dialects used by JAX.";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    MlirDialectHandle func_dialect = mlirGetDialectHandle__func__();
    mlirDialectHandleInsertDialect(func_dialect, registry);
  });
}