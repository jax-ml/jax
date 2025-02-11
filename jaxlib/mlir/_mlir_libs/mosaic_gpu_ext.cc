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

#include <cstdint>
#include <vector>

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h"
#include "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h"

namespace nb = nanobind;

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__mosaic_gpu__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "SwizzleTransformAttr", MosaicGpuIsASwizzleTransformAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, int32_t swizzle, MlirContext ctx) {
            return cls(MosaicGpuSwizzleTransformAttrGet(
                ctx, static_cast<int32_t>(swizzle)));
          },
          nb::arg("cls"), nb::arg("swizzle"),
          nb::arg("context").none() = nb::none(),
          "Creates a SwizzleTransformAttr with the given swizzle.")
      .def_property_readonly("swizzle", [](MlirAttribute self) {
        return MosaicGpuSwizzleTransformAttrGetSwizzle(self);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "LayoutAttr", MosaicGpuIsALayoutAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, int32_t num_dimensions,
             std::vector<MlirAttribute>& transforms, MlirContext ctx) {
            return cls(MosaicGpuLayoutAttrGet(
                ctx, num_dimensions, transforms.data(), transforms.size()));
          },
          nb::arg("cls"), nb::arg("num_dimensions"), nb::arg("transforms"),
          nb::arg("context").none() = nb::none(),
          "Creates a LayoutAttr with the given transforms.")
      .def_property_readonly("transforms", [](MlirAttribute self) {
        std::vector<MlirAttribute> result;
        for (int i = 0; i < MosaicGpuLayoutAttrGetTransformsSize(self); ++i) {
          result.push_back(MosaicGpuLayoutAttrGetTransform(self, i));
        }
        return result;
      });
}
