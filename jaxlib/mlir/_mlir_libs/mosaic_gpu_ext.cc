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

#include "absl/hash/hash.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h"
#include "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h"
#include "jaxlib/mosaic/gpu/tiled_layout.h"

namespace nb = nanobind;
namespace mgpu = jax::mosaic::gpu;

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

  m.def("register_inliner_extensions", [](MlirContext context) {
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    mlirDialectRegistryInsertMosaicGpuInlinerExtensions(registry);
    mlirContextAppendDialectRegistry(context, registry);
    mlirDialectRegistryDestroy(registry);
  });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TileTransformAttr", mlirMosaicGpuIsATileTransformAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<int32_t>& tiling, MlirContext ctx) {
            return cls(mlirMosaicGpuTileTransformAttrGet(ctx, tiling.data(),
                                                         tiling.size()));
          },
          nb::arg("cls"), nb::arg("tiling"),
          nb::arg("context").none() = nb::none(),
          "Creates a TileTransformAttr with the given tiling.")
      .def_property_readonly("tiling", [](MlirAttribute self) {
        std::vector<int32_t> result;
        for (int i = 0; i < mlirMosaicGpuTileTransformAttrGetTilingSize(self);
             ++i) {
          result.push_back(mlirMosaicGpuTileTransformAttrGetTiling(self, i));
        }
        return result;
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TransposeTransformAttr", mlirMosaicGpuIsATransposeTransformAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<int32_t>& permutation,
             MlirContext ctx) {
            return cls(mlirMosaicGpuTransposeTransformAttrGet(
                ctx, permutation.data(), permutation.size()));
          },
          nb::arg("cls"), nb::arg("permutation"),
          nb::arg("context").none() = nb::none(),
          "Creates a TransposeTransformAttr with the given permutation.")
      .def_property_readonly("permutation", [](MlirAttribute self) {
        std::vector<int32_t> result;
        for (int i = 0;
             i < mlirMosaicGpuTransposeTransformAttrGetPermutationSize(self);
             ++i) {
          result.push_back(
              mlirMosaicGpuTransposeTransformAttrGetPermutation(self, i));
        }
        return result;
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TileTransformAttr", mlirMosaicGpuIsATileTransformAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<int32_t>& tiling, MlirContext ctx) {
            return cls(mlirMosaicGpuTileTransformAttrGet(ctx, tiling.data(),
                                                         tiling.size()));
          },
          nb::arg("cls"), nb::arg("tiling"),
          nb::arg("context").none() = nb::none(),
          "Creates a TileTransformAttr with the given tiling.")
      .def_property_readonly("tiling", [](MlirAttribute self) {
        std::vector<int32_t> result;
        for (int i = 0; i < mlirMosaicGpuTileTransformAttrGetTilingSize(self);
             ++i) {
          result.push_back(mlirMosaicGpuTileTransformAttrGetTiling(self, i));
        }
        return result;
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TransposeTransformAttr", mlirMosaicGpuIsATransposeTransformAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<int32_t>& permutation,
             MlirContext ctx) {
            return cls(mlirMosaicGpuTransposeTransformAttrGet(
                ctx, permutation.data(), permutation.size()));
          },
          nb::arg("cls"), nb::arg("permutation"),
          nb::arg("context").none() = nb::none(),
          "Creates a TransposeTransformAttr with the given permutation.")
      .def_property_readonly("permutation", [](MlirAttribute self) {
        std::vector<int32_t> result;
        for (int i = 0;
             i < mlirMosaicGpuTransposeTransformAttrGetPermutationSize(self);
             ++i) {
          result.push_back(
              mlirMosaicGpuTransposeTransformAttrGetPermutation(self, i));
        }
        return result;
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "SwizzleTransformAttr", mlirMosaicGpuIsASwizzleTransformAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, int32_t swizzle, MlirContext ctx) {
            return cls(mlirMosaicGpuSwizzleTransformAttrGet(
                ctx, static_cast<int32_t>(swizzle)));
          },
          nb::arg("cls"), nb::arg("swizzle"),
          nb::arg("context").none() = nb::none(),
          "Creates a SwizzleTransformAttr with the given swizzle.")
      .def_property_readonly("swizzle", [](MlirAttribute self) {
        return mlirMosaicGpuSwizzleTransformAttrGetSwizzle(self);
      });

  nb::class_<mgpu::Tiling>(m, "Tiling")
      .def(
          "__init__",
          [](mgpu::Tiling* self, nb::iterable in_tiles) {
            std::vector<std::vector<int64_t>> tiles;
            for (const auto& tile : in_tiles) {
              tiles.push_back(nb::cast<std::vector<int64_t>>(tile));
            }
            auto result = mgpu::Tiling::Create(tiles);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return new (self) mgpu::Tiling(*result);
          },
          nb::arg("tiles"))
      .def(
          "tile_shape",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& shape) {
            auto result = self.TileShape(shape);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return nb::tuple(nb::cast(*result));
          },
          nb::arg("shape"))
      .def(
          "untile_shape",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& shape) {
            auto result = self.UntileShape(shape);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return nb::tuple(nb::cast(*result));
          },
          nb::arg("shape"))
      .def(
          "tile_strides",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& strides) {
            return nb::tuple(nb::cast(self.TileStrides(strides)));
          },
          nb::arg("strides"))
      .def(
          "tile_indices",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& indices) {
            return nb::tuple(nb::cast(self.TileIndices(indices)));
          },
          nb::arg("indices"))
      .def(
          "untile_indices",
          [](const mgpu::Tiling& self, const std::vector<int64_t>& indices) {
            return nb::tuple(nb::cast(self.UntileIndices(indices)));
          },
          nb::arg("indices"))
      .def(
          "tile_nested_shape_strides",
          [](const mgpu::Tiling& self,
             const std::vector<std::vector<int64_t>>& shape,
             const std::vector<std::vector<int64_t>>& strides) {
            auto result = self.TileNestedShapeStrides(shape, strides);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            auto [tiled_shape, tiled_strides] = *result;
            nb::list shape_list;
            for (const auto& s : tiled_shape) {
              shape_list.append(nb::tuple(nb::cast(s)));
            }
            nb::list strides_list;
            for (const auto& s : tiled_strides) {
              strides_list.append(nb::tuple(nb::cast(s)));
            }
            return nb::make_tuple(nb::tuple(shape_list),
                                  nb::tuple(strides_list));
          },
          nb::arg("shape"), nb::arg("strides"))
      .def(
          "tile_dimension",
          [](const mgpu::Tiling& self, int64_t dim) {
            auto result = self.TileDimension(dim);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return nb::tuple(nb::cast(*result));
          },
          nb::arg("dim"))
      .def(
          "remove_dimension",
          [](const mgpu::Tiling& self, int64_t dim) {
            auto result = self.RemoveDimension(dim);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return *result;
          },
          nb::arg("dim"))
      .def("canonicalize", &mgpu::Tiling::Canonicalize)
      .def_prop_ro("tiles",
                   [](const mgpu::Tiling& self) {
                     nb::list tiles_list;
                     for (const mgpu::Tiling::Tile& tile : self.tiles()) {
                       tiles_list.append(nb::tuple(nb::cast(tile)));
                     }
                     return nb::tuple(tiles_list);
                   })
      .def("__str__", &mgpu::Tiling::ToString)
      .def("__repr__", &mgpu::Tiling::ToString)
      .def(nb::self == nb::self)
      .def("__hash__", [](const mgpu::Tiling& self) {
        return absl::Hash<mgpu::Tiling>{}(self);
      });
}
