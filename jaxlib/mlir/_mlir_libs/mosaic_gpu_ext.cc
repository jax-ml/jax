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
#include <optional>
#include <utility>
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
#include "nanobind/stl/function.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h"
#include "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h"
#include "jaxlib/mosaic/gpu/tiled_layout.h"
#include "jaxlib/mosaic/gpu/transfer_plan.h"
#include "jaxlib/mosaic/gpu/transforms.h"

namespace nb = nanobind;
namespace mgpu = jax::mosaic::gpu;

namespace {

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyInsertionPoint;

// Returns the `mlir_ir_module` and caches its usage in the program. If the
// input module is not specified, returns the already cached module, which
// can be either the default `None` or the module passed in the first call.
static nb::object& MlirIrModule(nb::object mlir_ir_module = nb::none()) {
  static nb::object* mlir_ir = new nb::object(nb::none());
  if (mlir_ir->is_none() && !mlir_ir_module.is_none()) {
    *mlir_ir = mlir_ir_module;
  }
  return *mlir_ir;
}

// Returns an `ImplicitLocOpBuilder` with the current loc location and ip
// insertion point extracted from the `MlirIrModule`. Returns invalid argument
// error if the `mlir.ir` module is not found.
absl::StatusOr<mlir::ImplicitLocOpBuilder> MlirBuilder() {
  nb::object& mlir_ir = MlirIrModule();
  if (mlir_ir.is_none()) {
    return absl::InvalidArgumentError(
        "MLIR IR module has not been initialized. Call init_cc_mlir() before "
        "using other functions.");
  }
  nb::object ip = mlir_ir.attr("InsertionPoint").attr("current");
  nb::object loc = mlir_ir.attr("Location").attr("current");

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

nb::tuple TileIndexTransforms(const mgpu::TransferPlan& plan) {
  auto transforms = plan.TileIndexTransforms();
  nb::list py_transforms;
  for (const auto& transform : transforms) {
    py_transforms.append(nb::cpp_function([transform](nb::tuple idx) {
      auto idx_list = nb::cast<std::vector<int64_t>>(idx);
      nb::list result;
      for (const auto& v : transform(idx_list)) {
        result.append(nb::cast(v));
      }
      return nb::tuple(result);
    }));
  }
  return nb::tuple(py_transforms);
}

mlir::Value Select(const mgpu::TransferPlan& plan, nb::iterable group_elems) {
  auto builder = MlirBuilder();
  if (!builder.ok()) {
    throw nb::value_error(builder.status().message().data());
  }
  std::vector<mlir::Value> elems;
  for (nb::handle elem : group_elems) {
    elems.push_back(unwrap(nb::cast<MlirValue>(elem)));
  }
  auto result = plan.Select(*builder, elems);
  if (!result.ok()) {
    if (absl::IsFailedPrecondition(result.status())) {
      PyErr_SetString(PyExc_AssertionError, result.status().message().data());
      throw nb::python_error();
    }
    throw nb::value_error(result.status().message().data());
  }
  return *result;
}

mlir::Value SelectIfGroup(const mgpu::TransferPlan& plan, int64_t group_idx,
                          MlirValue old_val, MlirValue new_val) {
  auto builder = MlirBuilder();
  if (!builder.ok()) {
    throw nb::value_error(builder.status().message().data());
  }
  auto result =
      plan.SelectIfGroup(*builder, group_idx, unwrap(old_val), unwrap(new_val));
  if (!result.ok()) {
    throw nb::value_error(result.status().message().data());
  }
  return *result;
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

  auto barrier_type =
      mlir::python::nanobind_adaptors::mlir_type_subclass(
          m, "BarrierType", mlirMosaicGpuIsABarrierType,
          mlirMosaicGpuBarrierTypeGetTypeID);
  barrier_type
      .def_staticmethod(
          "get",
          [cls = barrier_type.get_class()](bool orders_tensor_core,
                                           MlirContext ctx) {
            return cls(mlirMosaicGpuBarrierTypeGet(ctx, orders_tensor_core));
          },
          nb::arg("orders_tensor_core") = false,
          nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "orders_tensor_core: bool = False, "
              "context: mlir.ir.Context | None = None"
              ") -> BarrierType"
              // clang-format: on
              ),
          "Creates a BarrierType.")
      .def_property_readonly("orders_tensor_core",
                             mlirMosaicGpuBarrierTypeGetOrdersTensorCore);

  auto tile_transform_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "TileTransformAttr", mlirMosaicGpuIsATileTransformAttr,
          mlirMosaicGpuTileTransformAttrGetTypeID);
  tile_transform_attr
      .def_staticmethod(
          "get",
          [cls = tile_transform_attr.get_class()](std::vector<int32_t>& tiling,
                                                  MlirContext ctx) {
            return cls(mlirMosaicGpuTileTransformAttrGet(ctx, tiling.data(),
                                                         tiling.size()));
          },
          nb::arg("tiling"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "tiling: Sequence[int], "
              "context: mlir.ir.Context | None = None"
              ") -> TileTransformAttr"
              // clang-format: on
              ),
          "Creates a TileTransformAttr with the given tiling.")
      .def_property_readonly(
          "tiling", mlirMosaicGpuTileTransformAttrGetTiling,
          nb::sig("def tiling(self) -> mlir.ir.DenseI32ArrayAttr"));

  auto transpose_transform_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "TransposeTransformAttr", mlirMosaicGpuIsATransposeTransformAttr,
          mlirMosaicGpuTransposeTransformAttrGetTypeID);
  transpose_transform_attr
      .def_staticmethod(
          "get",
          [cls = transpose_transform_attr.get_class()](
              std::vector<int32_t>& permutation, MlirContext ctx) {
            return cls(mlirMosaicGpuTransposeTransformAttrGet(
                ctx, permutation.data(), permutation.size()));
          },
          nb::arg("permutation"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "permutation: Sequence[int], "
              "context: mlir.ir.Context | None = None"
              ") -> TransposeTransformAttr"
              // clang-format: on
              ),
          "Creates a TransposeTransformAttr with the given permutation.")
      .def_property_readonly(
          "permutation", mlirMosaicGpuTransposeTransformAttrGetPermutation,
          nb::sig("def permutation(self) -> mlir.ir.DenseI32ArrayAttr"));

  auto swizzle_transform_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "SwizzleTransformAttr", mlirMosaicGpuIsASwizzleTransformAttr,
          mlirMosaicGpuSwizzleTransformAttrGetTypeID);
  swizzle_transform_attr
      .def_staticmethod(
          "get",
          [cls = swizzle_transform_attr.get_class()](int32_t swizzle,
                                                     MlirContext ctx) {
            return cls(mlirMosaicGpuSwizzleTransformAttrGet(ctx, swizzle));
          },
          nb::arg("swizzle"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "swizzle: int, "
              "context: mlir.ir.Context | None = None"
              ") -> SwizzleTransformAttr"
              // clang-format: on
              ),
          "Creates a SwizzleTransformAttr with the given swizzle.")
      .def_property_readonly("swizzle",
                             mlirMosaicGpuSwizzleTransformAttrGetSwizzle);

  auto splat_fragmented_layout_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "WGSplatFragLayoutAttr", mlirMosaicGpuIsAWGSplatFragLayoutAttr,
          mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID);
  splat_fragmented_layout_attr
      .def_staticmethod(
          "get",
          [cls = splat_fragmented_layout_attr.get_class()](MlirAttribute shape,
                                                           MlirContext ctx) {
            return cls(mlirMosaicGpuWGSplatFragLayoutAttrGet(ctx, shape));
          },
          nb::arg("shape"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "shape: mlir.ir.DenseI64ArrayAttr, "
              "context: mlir.ir.Context | None = None"
              ") -> WGSplatFragLayoutAttr"
              // clang-format: on
              ),
          "Creates a WGSplatFragLayoutAttr with the given shape.")
      .def_property_readonly(
          "shape", mlirMosaicGpuWGSplatFragLayoutAttrGetShape,
          nb::sig("def shape(self) -> mlir.ir.DenseI64ArrayAttr"));

  auto strided_fragmented_layout_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "WGStridedFragLayoutAttr", mlirMosaicGpuIsAWGStridedFragLayoutAttr,
          mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID);
  strided_fragmented_layout_attr
      .def_staticmethod(
          "get",
          [cls = strided_fragmented_layout_attr.get_class()](
              MlirAttribute shape, int32_t vector_size, MlirContext ctx) {
            return cls(mlirMosaicGpuWGStridedFragLayoutAttrGet(ctx, shape,
                                                               vector_size));
          },
          nb::arg("shape"), nb::arg("vector_size"),
          nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "shape: mlir.ir.DenseI64ArrayAttr, "
              "vector_size: int, "
              "context: mlir.ir.Context | None = None"
              ") -> WGStridedFragLayoutAttr"
              // clang-format: on
              ),
          "Creates a WGStridedFragLayoutAttr.")
      .def_property_readonly(
          "shape", mlirMosaicGpuWGStridedFragLayoutAttrGetShape,
          nb::sig("def shape(self) -> mlir.ir.DenseI64ArrayAttr"))
      .def_property_readonly("vector_size",
                             mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize);

  auto replicated_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "ReplicatedAttr", mlirMosaicGpuIsAReplicatedAttr,
          mlirMosaicGpuReplicatedAttrGetTypeID);
  replicated_attr
      .def_staticmethod(
          "get",
          [cls = replicated_attr.get_class()](int32_t times, MlirContext ctx) {
            return cls(mlirMosaicGpuReplicatedAttrGet(ctx, times));
          },
          nb::arg("times"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "times: int, "
              "context: mlir.ir.Context | None = None"
              ") -> ReplicatedAttr"
              // clang-format: on
              ),
          "Creates a ReplicatedAttr.")
      .def_property_readonly("times", mlirMosaicGpuReplicatedAttrGetTimes);

  auto tiled_layout_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "TiledLayoutAttr", mlirMosaicGpuIsATiledLayoutAttr,
          mlirMosaicGpuTiledLayoutAttrGetTypeID);
  tiled_layout_attr
      .def_staticmethod(
          "get",
          [cls = tiled_layout_attr.get_class()](
              MlirAttribute tiling, MlirAttribute warp_dims,
              MlirAttribute lane_dims, int32_t vector_dim, MlirContext ctx) {
            return cls(mlirMosaicGpuTiledLayoutAttrGet(ctx, tiling, warp_dims,
                                                       lane_dims, vector_dim));
          },
          nb::arg("tiling"), nb::arg("warp_dims"), nb::arg("lane_dims"),
          nb::arg("vector_dim"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "tiling: mlir.ir.ArrayAttr, "
              "warp_dims: mlir.ir.ArrayAttr, "
              "lane_dims: mlir.ir.ArrayAttr, "
              "vector_dim: int, "
              "context: mlir.ir.Context | None = None"
              ") -> TiledLayoutAttr"
              // clang-format: on
              ),
          "Creates a TiledLayoutAttr.")
      .def_property_readonly("tiling", mlirMosaicGpuTiledLayoutAttrGetTiling,
                             nb::sig("def tiling(self) -> mlir.ir.ArrayAttr"))
      .def_property_readonly(
          "warp_dims", mlirMosaicGpuTiledLayoutAttrGetWarpDims,
          nb::sig("def warp_dims(self) -> mlir.ir.ArrayAttr"))
      .def_property_readonly(
          "lane_dims", mlirMosaicGpuTiledLayoutAttrGetLaneDims,
          nb::sig("def lane_dims(self) -> mlir.ir.ArrayAttr"))
      .def_property_readonly("vector_dim",
                             mlirMosaicGpuTiledLayoutAttrGetVectorDim);

  auto copy_partition_attr_interface =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "CopyPartitionAttrInterface", mlirMosaicGpuIsACopyPartitionAttr);

  auto copy_replicated_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "CopyReplicatedAttr", mlirMosaicGpuIsACopyReplicatedAttr,
          copy_partition_attr_interface.get_class(),
          mlirMosaicGpuCopyReplicatedAttrGetTypeID);
  copy_replicated_attr.def_staticmethod(
      "get",
      [cls = copy_replicated_attr.get_class()](MlirContext ctx) {
        return cls(mlirMosaicGpuCopyReplicatedAttrGet(ctx));
      },
      nb::arg("context").none() = nb::none(),
      nb::sig(
          // clang-format: off
          "def get("
          "context: mlir.ir.Context | None = None"
          ") -> CopyReplicatedAttr"
          // clang-format: on
          ),
      "Creates a CopyReplicatedAttr.");

  auto copy_partitioned_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "CopyPartitionedAttr", mlirMosaicGpuIsACopyPartitionedAttr,
          copy_partition_attr_interface.get_class(),
          mlirMosaicGpuCopyPartitionedAttrGetTypeID);
  copy_partitioned_attr
      .def_staticmethod(
          "get",
          [cls = copy_partitioned_attr.get_class()](int32_t axis,
                                                    MlirContext ctx) {
            return cls(mlirMosaicGpuCopyPartitionedAttrGet(ctx, axis));
          },
          nb::arg("axis"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "axis: int, "
              "context: mlir.ir.Context | None = None"
              ") -> CopyPartitionedAttr"
              // clang-format: on
              ),
          "Creates a CopyPartitionedAttr.")
      .def_property_readonly("axis", mlirMosaicGpuCopyPartitionedAttrGetAxis);

  m.def("init_cc_mlir", [](nb::object mlir_ir_module) {
    nb::object& mlir_ir = MlirIrModule(mlir_ir_module);
    return !mlir_ir.is_none();
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
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, other: object) -> bool"))
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
                     return *result;
                   })
      .def_prop_ro("vector_dim", &mgpu::TiledLayout::vector_dim)
      .def_prop_ro("tiling", &mgpu::TiledLayout::tiling)
      .def_prop_ro("tiled_tiling_shape",
                   [](const mgpu::TiledLayout& self) {
                     return nb::tuple(nb::cast(self.tiled_tiling_shape()));
                   })
      .def_prop_ro("tiled_tiling_rank",
                   [](const mgpu::TiledLayout& self) {
                     return self.tiled_tiling_rank();
                   })
      .def("warp_indices",
           [](const mgpu::TiledLayout& self) {
             auto builder = MlirBuilder();
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
           })
      .def("lane_indices",
           [](const mgpu::TiledLayout& self) {
             auto builder = MlirBuilder();
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
           })
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
          [](const mgpu::TiledLayout& self, MlirType t) -> MlirType {
            auto result = self.RegistersElementType(unwrap(t));
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return wrap(*result);
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
            auto result = self.Reduce(nb::cast<std::vector<int64_t>>(axes));
            if (!result.ok()) {
              throw nb::value_error(result.status().message().data());
            }
            return *result;
          },
          nb::arg("axes"))
      .def(
          "thread_idxs",
          [](const mgpu::TiledLayout& self, const std::vector<int64_t>& shape) {
            auto builder = MlirBuilder();
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

  nb::enum_<mgpu::Rounding>(m, "Rounding")
      .value("UP", mgpu::Rounding::kUp)
      .value("DOWN", mgpu::Rounding::kDown);

  nb::class_<mgpu::TileTransform>(m, "TileTransform")
      .def(nb::init<std::vector<int64_t>, std::optional<mgpu::Rounding>>(),
           nb::arg("tiling"), nb::arg("rounding") = std::nullopt)
      .def("apply",
           [](const mgpu::TileTransform& self, nb::object ref) {
             auto builder = MlirBuilder();
             if (!builder.ok()) {
               throw nb::value_error(builder.status().message().data());
             }
             auto result =
                 self.Apply(*builder, unwrap(nb::cast<MlirValue>(ref)));
             if (!result.ok()) {
               throw nb::value_error(result.status().message().data());
             }
             return wrap(*result);
           })
      .def("transform_index",
           [](const mgpu::TileTransform& self, nb::iterable idx) {
             auto builder = MlirBuilder();
             if (!builder.ok()) {
               throw nb::value_error(builder.status().message().data());
             }
             std::vector<mlir::Value> idxs;
             for (nb::handle i : idx) {
               idxs.push_back(unwrap(nb::cast<MlirValue>(i)));
             }
             auto result = self.TransformIndex(*builder, idxs);
             if (!result.ok()) {
               throw nb::value_error(result.status().message().data());
             }
             nb::list wrapped_result;
             for (auto v : *result) {
               wrapped_result.append(nb::cast(wrap(v)));
             }
             return nb::tuple(wrapped_result);
           })
      .def("transform_shape",
           [](const mgpu::TileTransform& self, std::vector<int64_t> shape) {
             auto result = self.TransformShape(shape);
             if (!result.ok()) {
               throw nb::value_error(result.status().message().data());
             }
             return nb::tuple(nb::cast(*result));
           })
      .def("transform_strides",
           [](const mgpu::TileTransform& self, std::vector<int64_t> strides) {
             return nb::tuple(nb::cast(self.TransformStrides(strides)));
           });

  nb::class_<mgpu::TrivialTransferPlan>(m, "TrivialTransferPlan")
      .def(nb::init<>())
      .def_prop_ro("tile_index_transforms",
                   [](const mgpu::TrivialTransferPlan& self) {
                     return TileIndexTransforms(self);
                   })
      .def("select",
           [](const mgpu::TrivialTransferPlan& self, nb::iterable group_elems) {
             return wrap(Select(self, group_elems));
           })
      .def("select_if_group",
           [](const mgpu::TrivialTransferPlan& self, int64_t group_idx,
              MlirValue old_val, MlirValue new_val) {
             return wrap(SelectIfGroup(self, group_idx, old_val, new_val));
           });

  nb::class_<mgpu::StaggeredTransferPlan>(m, "StaggeredTransferPlan")
      .def(
          "__init__",
          [](mgpu::StaggeredTransferPlan* self, int64_t stagger, int64_t dim,
             int64_t size, nb::object group_pred) {
            new (self) mgpu::StaggeredTransferPlan(
                stagger, dim, size, unwrap(nb::cast<MlirValue>(group_pred)));
          },
          nb::arg("stagger"), nb::arg("dim"), nb::arg("size"),
          nb::arg("group_pred"))
      .def_prop_ro("stagger", &mgpu::StaggeredTransferPlan::stagger)
      .def_prop_ro("dim", &mgpu::StaggeredTransferPlan::dim)
      .def_prop_ro("size", &mgpu::StaggeredTransferPlan::size)
      .def_prop_ro("group_pred",
                   [](const mgpu::StaggeredTransferPlan& self) {
                     return wrap(self.group_pred());
                   })
      .def_prop_ro("tile_index_transforms",
                   [](const mgpu::StaggeredTransferPlan& self) {
                     return TileIndexTransforms(self);
                   })
      .def("select",
           [](const mgpu::StaggeredTransferPlan& self,
              nb::iterable group_elems) {
             return wrap(Select(self, group_elems));
           })
      .def("select_if_group",
           [](const mgpu::StaggeredTransferPlan& self, int64_t group_idx,
              MlirValue old_val, MlirValue new_val) {
             return wrap(SelectIfGroup(self, group_idx, old_val, new_val));
           });
}
