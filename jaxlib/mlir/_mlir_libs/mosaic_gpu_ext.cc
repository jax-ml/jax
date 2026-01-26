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
#include <variant>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Block.h"  // IWYU pragma: keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"  // IWYU pragma: keep
#include "mlir/IR/Value.h"  // IWYU pragma: keep
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

namespace {

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyInsertionPoint;

// Returns an ImplicitLocOpBuilder with the given `loc` location and `ip`
// insertion point. Returns invalid argument error if the block is not specified
// in the insertion point.
absl::StatusOr<mlir::ImplicitLocOpBuilder> MlirBuilder(nb::object ip,
                                                       nb::object loc) {
  auto insertion_point = nb::cast<PyInsertionPoint>(ip);
  mlir::Location location = unwrap(nb::cast<MlirLocation>(loc));
  mlir::Block* block = unwrap(insertion_point.getBlock().get());
  if (block == nullptr) {
    return absl::InvalidArgumentError("MLIR block is null");
  }
  mlir::Operation* ref_op = nullptr;
  if (insertion_point.getRefOperation()) {
    if (auto* py_op = insertion_point.getRefOperation()->get(); py_op) {
      ref_op = unwrap(py_op->get());
    }
  }
  mlir::ImplicitLocOpBuilder builder(location, block, block->end());
  if (ref_op != nullptr) {
    builder.setInsertionPoint(ref_op);
  }
  return builder;
}

}  // namespace

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
            new (self) mgpu::Tiling(*result);
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

  nb::class_<mgpu::Replicated>(m, "Replicated")
      .def(nb::init<int64_t>(), nb::arg("times"))
      .def_prop_rw(
          "times", [](const mgpu::Replicated& self) { return self.times; },
          [](mgpu::Replicated& self, int64_t times) { self.times = times; })
      .def("__repr__", &mgpu::Replicated::ToString)
      .def("__hash__",
           [](const mgpu::Replicated& self) {
             return absl::Hash<mgpu::Replicated>{}(self);
           })
      .def("__eq__", [](const mgpu::Replicated& self, nb::object other) {
        if (!nb::isinstance<mgpu::Replicated>(other)) {
          return false;
        }
        return self == nb::cast<mgpu::Replicated>(other);
      });

  nb::class_<mgpu::TiledLayout>(m, "TiledLayout")
      .def(
          "__init__",
          [](mgpu::TiledLayout* self, mgpu::Tiling tiling,
             nb::iterable in_warp_dims, nb::iterable in_lane_dims,
             int64_t vector_dim, bool check_canonical) {
            std::vector<mgpu::TiledLayout::Dim> warp_dims;
            for (const auto& dim : in_warp_dims) {
              if (nb::isinstance<mgpu::Replicated>(dim)) {
                warp_dims.emplace_back(nb::cast<mgpu::Replicated>(dim));
              } else {
                warp_dims.emplace_back(nb::cast<int64_t>(dim));
              }
            }
            std::vector<mgpu::TiledLayout::Dim> lane_dims;
            for (const auto& dim : in_lane_dims) {
              if (nb::isinstance<mgpu::Replicated>(dim)) {
                lane_dims.emplace_back(nb::cast<mgpu::Replicated>(dim));
              } else {
                lane_dims.emplace_back(nb::cast<int64_t>(dim));
              }
            }
            auto result = mgpu::TiledLayout::Create(
                tiling, warp_dims, lane_dims, vector_dim, check_canonical);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            new (self) mgpu::TiledLayout(*result);
          },
          nb::arg("tiling"), nb::arg("warp_dims"), nb::arg("lane_dims"),
          nb::arg("vector_dim"), nb::arg("_check_canonical") = true)
      .def_prop_ro("warp_dims",
                   [](const mgpu::TiledLayout& self) {
                     nb::list l;
                     for (const auto& d : self.warp_dims()) {
                       if (std::holds_alternative<mgpu::Replicated>(d)) {
                         l.append(nb::cast(std::get<mgpu::Replicated>(d)));
                       } else {
                         l.append(nb::cast(std::get<int64_t>(d)));
                       }
                     }
                     return nb::tuple(l);
                   })
      .def_prop_ro("lane_dims",
                   [](const mgpu::TiledLayout& self) {
                     nb::list l;
                     for (const auto& d : self.lane_dims()) {
                       if (std::holds_alternative<mgpu::Replicated>(d)) {
                         l.append(nb::cast(std::get<mgpu::Replicated>(d)));
                       } else {
                         l.append(nb::cast(std::get<int64_t>(d)));
                       }
                     }
                     return nb::tuple(l);
                   })
      .def_prop_ro("partitioned_warp_dims",
                   [](const mgpu::TiledLayout& self) {
                     return nb::tuple(nb::cast(self.PartitionedWarpDims()));
                   })
      .def_prop_ro("partitioned_lane_dims",
                   [](const mgpu::TiledLayout& self) {
                     return nb::tuple(nb::cast(self.PartitionedLaneDims()));
                   })
      .def_prop_ro("vector_length",
                   [](const mgpu::TiledLayout& self) {
                     auto result = self.VectorLength();
                     if (!result.ok()) {
                       throw nb::value_error(result.status().message().data());
                     }
                     return nb::cast(*result);
                   })
      .def_prop_ro("vector_dim", &mgpu::TiledLayout::vector_dim)
      .def_prop_ro("tiling", &mgpu::TiledLayout::tiling)
      .def_prop_ro("tiled_tiling_shape",
                   [](const mgpu::TiledLayout& self) {
                     auto result = self.TiledTilingShape();
                     if (!result.ok()) {
                       throw nb::value_error(result.status().message().data());
                     }
                     return nb::tuple(nb::cast(*self.TiledTilingShape()));
                   })
      .def(
          "warp_indices",
          [](const mgpu::TiledLayout& self, nb::object ip, nb::object loc) {
            auto builder = MlirBuilder(ip, loc);
            if (!builder.ok()) {
              throw nb::value_error(builder.status().message().data());
            }
            auto result = self.WarpIndices(*builder);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            nb::list l;
            for (const auto& v : *result) {
              l.append(nb::cast(wrap(v)));
            }
            return nb::tuple(l);
          },
          nb::arg("ip"), nb::arg("loc"))
      .def(
          "lane_indices",
          [](const mgpu::TiledLayout& self, nb::object ip, nb::object loc) {
            auto builder = MlirBuilder(ip, loc);
            if (!builder.ok()) {
              throw nb::value_error(builder.status().message().data());
            }
            auto result = self.LaneIndices(*builder);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            nb::list l;
            for (const auto& v : *result) {
              l.append(nb::cast(wrap(v)));
            }
            return nb::tuple(l);
          },
          nb::arg("ip"), nb::arg("loc"))

      .def("canonicalize",
           [](const mgpu::TiledLayout& self) {
             auto result = self.Canonicalize();
             if (!result.ok()) {
               throw nb::value_error(result.status().message().data());
             }
             return *result;
           })
      .def(
          "registers_shape",
          [](const mgpu::TiledLayout& self, const std::vector<int64_t>& shape) {
            auto result = self.RegistersShape(shape);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return nb::tuple(nb::cast(*result));
          },
          nb::arg("shape"))
      .def(
          "registers_element_type",
          [](const mgpu::TiledLayout& self, MlirType t) {
            auto result = self.RegistersElementType(unwrap(t));
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return nb::cast(wrap(*result));
          },
          nb::arg("t"))
      .def(
          "shape_from_registers_shape",
          [](const mgpu::TiledLayout& self, const std::vector<int64_t>& shape) {
            auto result = self.ShapeFromRegistersShape(shape);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return nb::tuple(nb::cast(*result));
          },
          nb::arg("shape"))
      .def_prop_ro("base_tile_shape",
                   [](const mgpu::TiledLayout& self) {
                     return nb::tuple(nb::cast(self.BaseTileShape()));
                   })
      .def(
          "remove_dimension",
          [](const mgpu::TiledLayout& self, int64_t dim) {
            auto result = self.RemoveDimension(dim);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return *result;
          },
          nb::arg("dim"))
      .def(
          "reduce",
          [](const mgpu::TiledLayout& self, nb::iterable axes) {
            std::vector<int64_t> axes_vec;
            for (const auto& axis : axes) {
              axes_vec.push_back(nb::cast<int64_t>(axis));
            }
            auto result = self.Reduce(axes_vec);
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return *result;
          },
          nb::arg("axes"))
      .def("thread_idxs",
           [](const mgpu::TiledLayout& self, const std::vector<int64_t>& shape,
              nb::object ip, nb::object loc) {
             auto builder = MlirBuilder(ip, loc);
             if (!builder.ok()) {
               throw nb::value_error(builder.status().message().data());
             }

             auto result = self.ThreadIdxs(*builder, shape);
             if (!result.ok()) {
               throw nb::value_error(result.status().message().data());
             }
             nb::list list;
             for (const auto& row : *result) {
               nb::list inner_list;
               for (const auto& val : row) {
                 inner_list.append(nb::cast(wrap(val)));
               }
               list.append(nb::tuple(inner_list));
             }
             return list;
           })
      .def("__str__", &mgpu::TiledLayout::ToString)
      .def("__repr__", &mgpu::TiledLayout::ToString)
      .def("__hash__",
           [](const mgpu::TiledLayout& self) {
             return absl::Hash<mgpu::TiledLayout>{}(self);
           })
      .def(
          "__eq__",
          [](const mgpu::TiledLayout& self, nb::object other) -> bool {
            if (!nb::isinstance<mgpu::TiledLayout>(other)) {
              return false;
            }
            return self == nb::cast<mgpu::TiledLayout>(other);
          },
          nb::arg("other").none());
}
