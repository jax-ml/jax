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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "pybind11/detail/common.h"
#include "pybind11/pytypes.h"
#include "jax/_src/lib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

PYBIND11_MODULE(_tpu_ext, m) {
  mlirRegisterTPUPasses();  // Register all passes on load.

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__tpu__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def("private_is_tiled_layout", [](MlirAttribute attr) {
    return mlirTPUAttributeIsATiledLayoutAttr(attr);
  });
  m.def("private_get_tiles", [](MlirAttribute attr) -> py::object {
    MlirAttribute encoded_tiles = mlirTPUTiledLayoutAttrGetTiles(attr);
    py::tuple py_tiles(mlirArrayAttrGetNumElements(encoded_tiles));
    for (intptr_t i = 0; i < mlirArrayAttrGetNumElements(encoded_tiles); ++i) {
      MlirAttribute tile = mlirArrayAttrGetElement(encoded_tiles, i);
      py::tuple py_tile(mlirDenseArrayGetNumElements(tile));
      for (intptr_t j = 0; j < mlirDenseArrayGetNumElements(tile); ++j) {
        py_tile[j] = mlirDenseI64ArrayGetElement(tile, j);
      }
      py_tiles[i] = py_tile;
    }
    return py_tiles;
  });
  m.def("private_has_communication", [](MlirOperation op) {
    bool has_communication;
    bool has_custom_barrier;
    mlirTPUAnalyzePotentialCommunication(op, &has_communication,
                                         &has_custom_barrier);
    return std::make_pair(has_communication, has_custom_barrier);
  });

  // TODO(apaszke): All of those should be upstreamed to MLIR Python bindings.
  m.def("private_replace_all_uses_with", [](MlirOperation op,
                                            std::vector<MlirValue> vals) {
    if (vals.size() != mlirOperationGetNumResults(op)) {
      throw py::value_error("length mismatch in replace_all_uses_with");
    }
    for (int i = 0; i < vals.size(); ++i) {
      mlirValueReplaceAllUsesOfWith(mlirOperationGetResult(op, i), vals[i]);
    }
  });
  m.def("private_replace_all_uses_except",
        [](MlirValue old, MlirValue new_val, MlirOperation except) {
          for (intptr_t i = 0; i < mlirOperationGetNumOperands(except); ++i) {
            if (mlirValueEqual(mlirOperationGetOperand(except, i), new_val)) {
              throw py::value_error("new val already used in except");
            }
          }
          mlirValueReplaceAllUsesOfWith(old, new_val);
          // Undo the replacement in the except op.
          for (intptr_t i = 0; i < mlirOperationGetNumOperands(except); ++i) {
            if (mlirValueEqual(mlirOperationGetOperand(except, i), new_val)) {
              mlirOperationSetOperand(except, i, old);
            }
          }
        });
  m.def("private_set_operand",
        [](MlirOperation op, int idx, MlirValue new_operand) {
          mlirOperationSetOperand(op, idx, new_operand);
        });
  m.def("private_set_operands", [](MlirOperation op,
                                   std::vector<MlirValue> new_operands) {
    mlirOperationSetOperands(op, new_operands.size(), new_operands.data());
  });
  m.def("private_has_no_memory_space", [](MlirType ty) {
    return mlirAttributeIsNull(mlirMemRefTypeGetMemorySpace(ty));
  });
  m.def("private_is_identity", [](MlirAttribute attr) {
    return mlirAffineMapIsIdentity(mlirAffineMapAttrGetValue(attr));
  });
  m.def("private_insert_argument",
        [](int index, MlirBlock block, MlirType type) -> MlirValue {
          return mlirBlockInsertArgument(
              block, index, type,
              mlirLocationUnknownGet(mlirTypeGetContext(type)));
        });
  m.def("private_set_arg_attr",
        [](MlirOperation op, unsigned i, std::string name, MlirAttribute attr) {
          mlirFuncSetArgAttr(
              op, i, mlirStringRefCreateFromCString(name.c_str()), attr);
        });
  m.def("private_move_all_regions", [](MlirOperation src, MlirOperation dst) {
    if (mlirOperationGetNumRegions(src) != mlirOperationGetNumRegions(dst)) {
      throw py::value_error(
          "Region counts do not match in src operation and dst operations");
    }
    for (intptr_t i = 0; i < mlirOperationGetNumRegions(src); ++i) {
      mlirRegionTakeBody(mlirOperationGetRegion(dst, i),
                         mlirOperationGetRegion(src, i));
    }
  });
}
