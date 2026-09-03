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

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/IR/Block.h"  // IWYU pragma: keep
#include "mlir/IR/Location.h"  // IWYU pragma: keep
#include "mlir/IR/Operation.h"  // IWYU pragma: keep
#include "mlir/IR/Value.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h"
#include "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h"
#include "jaxlib/mosaic/gpu/dump.h"

namespace nb = nanobind;

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyAttribute;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyInsertionPoint;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyLocation;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMlirContext;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyThreadContextEntry;
using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyType;

namespace {

#define DEFINE_CONCRETE_ATTR(ClassName, IsAFunc, GetTypeIdFunc, BaseClass)     \
  struct Py##ClassName                                                         \
      : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute< \
            Py##ClassName, BaseClass> {                                        \
    static constexpr const char* pyClassName = #ClassName;                     \
    static bool isaFunction(MlirAttribute a) { return IsAFunc(a); }            \
    static constexpr MlirTypeID (*getTypeIdFunction)() = GetTypeIdFunc;        \
    using Base::Base;                                                          \
    static void bindDerived(ClassTy& cls);                                     \
  };                                                                           \
  void Py##ClassName::bindDerived(Py##ClassName::ClassTy& cls)

struct PyBarrierType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyBarrierType> {
  static constexpr const char* pyClassName = "BarrierType";
  static bool isaFunction(MlirType t) { return mlirMosaicGpuIsABarrierType(t); }
  static constexpr MlirTypeID (*getTypeIdFunction)() =
      mlirMosaicGpuBarrierTypeGetTypeID;
  using Base::Base;
  static void bindDerived(ClassTy& cls) {
    cls.def_static(
        "get",
        [](bool orders_tensor_core, DefaultingPyMlirContext ctx) {
          return PyBarrierType(ctx.resolve().getRef(),
                               mlirMosaicGpuBarrierTypeGet(ctx.resolve().get(),
                                                           orders_tensor_core));
        },
        nb::arg("orders_tensor_core") = false, nb::arg("ctx") = nb::none());
    cls.def_prop_ro("orders_tensor_core", [](PyBarrierType& self) {
      return mlirMosaicGpuBarrierTypeGetOrdersTensorCore(self.get());
    });
  }
};

struct PyB6x16P32Type
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyB6x16P32Type> {
  static constexpr const char* pyClassName = "B6x16P32Type";
  static bool isaFunction(MlirType t) {
    return mlirMosaicGpuIsAB6x16P32Type(t);
  }
  static constexpr MlirTypeID (*getTypeIdFunction)() =
      mlirMosaicGpuB6x16P32TypeGetTypeID;
  using Base::Base;
  static void bindDerived(ClassTy& cls) {
    cls.def_static(
        "get",
        [](PyType elementType, DefaultingPyMlirContext ctx) {
          return PyB6x16P32Type(
              ctx.resolve().getRef(),
              mlirMosaicGpuB6x16P32TypeGet(ctx.resolve().get(),
                                           elementType.get()));
        },
        nb::arg("element_type"),
        nb::arg("ctx") = nb::none());
    cls.def_prop_ro("element_type", [](PyB6x16P32Type& self) {
      return PyType(self.getContext(),
                    mlirMosaicGpuB6x16P32TypeGetElementType(self.get()));
    });
  }
};

struct PyP2B6Type
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyP2B6Type> {
  static constexpr const char* pyClassName = "P2B6Type";
  static bool isaFunction(MlirType t) {
    return mlirMosaicGpuIsAP2B6Type(t);
  }
  static constexpr MlirTypeID (*getTypeIdFunction)() =
      mlirMosaicGpuP2B6TypeGetTypeID;
  using Base::Base;
  static void bindDerived(ClassTy& cls) {
    cls.def_static(
        "get",
        [](PyType elementType, DefaultingPyMlirContext ctx) {
          return PyP2B6Type(
              ctx.resolve().getRef(),
              mlirMosaicGpuP2B6TypeGet(ctx.resolve().get(),
                                       elementType.get()));
        },
        nb::arg("element_type"),
        nb::arg("ctx") = nb::none());
    cls.def_prop_ro("element_type", [](PyP2B6Type& self) {
      return PyType(self.getContext(),
                    mlirMosaicGpuP2B6TypeGetElementType(self.get()));
    });
  }
};

DEFINE_CONCRETE_ATTR(TileTransformAttr, mlirMosaicGpuIsATileTransformAttr,
                     mlirMosaicGpuTileTransformAttrGetTypeID, PyAttribute) {
  cls.def_static(
      "get",
      [](const std::vector<int32_t>& tiling, DefaultingPyMlirContext ctx) {
        MlirAttribute tiling_attr = mlirDenseI32ArrayGet(
            ctx.resolve().get(), tiling.size(), tiling.data());
        MlirAttribute transform_attr =
            mlirMosaicGpuTileTransformAttrGet(ctx.resolve().get(), tiling_attr);
        return PyTileTransformAttr(ctx.resolve().getRef(), transform_attr);
      },
      nb::arg("tiling"), nb::arg("ctx") = nb::none());
  cls.def_prop_ro("tiling", [](PyTileTransformAttr& self) {
    return mlirMosaicGpuTileTransformAttrGetTiling(self.get());
  });
}

DEFINE_CONCRETE_ATTR(SwizzleTransformAttr, mlirMosaicGpuIsASwizzleTransformAttr,
                     mlirMosaicGpuSwizzleTransformAttrGetTypeID, PyAttribute) {
  cls.def_static(
      "get",
      [](int32_t swizzle, DefaultingPyMlirContext ctx) {
        return PySwizzleTransformAttr(
            ctx.resolve().getRef(),
            mlirMosaicGpuSwizzleTransformAttrGet(ctx.resolve().get(), swizzle));
      },
      nb::arg("swizzle"), nb::arg("ctx") = nb::none());
  cls.def_prop_ro("swizzle", [](PySwizzleTransformAttr& self) {
    return mlirMosaicGpuSwizzleTransformAttrGetSwizzle(self.get());
  });
}

DEFINE_CONCRETE_ATTR(WGSplatFragLayoutAttr,
                     mlirMosaicGpuIsAWGSplatFragLayoutAttr,
                     mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID, PyAttribute) {
  cls.def_static(
      "get",
      [](MlirAttribute shape, DefaultingPyMlirContext ctx) {
        return PyWGSplatFragLayoutAttr(
            ctx.resolve().getRef(),
            mlirMosaicGpuWGSplatFragLayoutAttrGet(ctx.resolve().get(), shape));
      },
      nb::arg("shape"), nb::arg("ctx") = nb::none());
  cls.def_prop_ro("shape", [](PyWGSplatFragLayoutAttr& self) {
    return mlirMosaicGpuWGSplatFragLayoutAttrGetShape(self.get());
  });
}

DEFINE_CONCRETE_ATTR(WGStridedFragLayoutAttr,
                     mlirMosaicGpuIsAWGStridedFragLayoutAttr,
                     mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID,
                     PyAttribute) {
  cls.def_static(
      "get",
      [](MlirAttribute shape, int32_t vector_size,
         DefaultingPyMlirContext ctx) {
        return PyWGStridedFragLayoutAttr(
            ctx.resolve().getRef(),
            mlirMosaicGpuWGStridedFragLayoutAttrGet(ctx.resolve().get(), shape,
                                                    vector_size));
      },
      nb::arg("shape"), nb::arg("vector_size"), nb::arg("ctx") = nb::none());
  cls.def_prop_ro("shape", [](PyWGStridedFragLayoutAttr& self) {
    return mlirMosaicGpuWGStridedFragLayoutAttrGetShape(self.get());
  });
  cls.def_prop_ro("vector_size", [](PyWGStridedFragLayoutAttr& self) {
    return mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(self.get());
  });
}

DEFINE_CONCRETE_ATTR(ReplicatedAttr, mlirMosaicGpuIsAReplicatedAttr,
                     mlirMosaicGpuReplicatedAttrGetTypeID, PyAttribute) {
  cls.def_static(
      "get",
      [](int32_t times, DefaultingPyMlirContext ctx) {
        return PyReplicatedAttr(
            ctx.resolve().getRef(),
            mlirMosaicGpuReplicatedAttrGet(ctx.resolve().get(), times));
      },
      nb::arg("times"), nb::arg("ctx") = nb::none());
  cls.def_prop_ro("times", [](PyReplicatedAttr& self) {
    return mlirMosaicGpuReplicatedAttrGetTimes(self.get());
  });
}

DEFINE_CONCRETE_ATTR(TiledLayoutAttr, mlirMosaicGpuIsATiledLayoutAttr,
                     mlirMosaicGpuTiledLayoutAttrGetTypeID, PyAttribute) {
  cls.def_static(
      "get",
      [](MlirAttribute tiling, MlirAttribute warp_dims, MlirAttribute lane_dims,
         int32_t vector_dim, DefaultingPyMlirContext ctx) {
        return PyTiledLayoutAttr(
            ctx.resolve().getRef(),
            mlirMosaicGpuTiledLayoutAttrGet(ctx.resolve().get(), tiling,
                                            warp_dims, lane_dims, vector_dim));
      },
      nb::arg("tiling"), nb::arg("warp_dims"), nb::arg("lane_dims"),
      nb::arg("vector_dim"), nb::arg("ctx") = nb::none());
  cls.def_prop_ro("tiling", [](PyTiledLayoutAttr& self) {
    return mlirMosaicGpuTiledLayoutAttrGetTiling(self.get());
  });
  cls.def_prop_ro("warp_dims", [](PyTiledLayoutAttr& self) {
    return mlirMosaicGpuTiledLayoutAttrGetWarpDims(self.get());
  });
  cls.def_prop_ro("lane_dims", [](PyTiledLayoutAttr& self) {
    return mlirMosaicGpuTiledLayoutAttrGetLaneDims(self.get());
  });
  cls.def_prop_ro("vector_dim", [](PyTiledLayoutAttr& self) {
    return mlirMosaicGpuTiledLayoutAttrGetVectorDim(self.get());
  });
}

struct PyCopyPartitionAttrInterface
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyCopyPartitionAttrInterface> {
  static constexpr const char* pyClassName = "CopyPartitionAttrInterface";
  static bool isaFunction(MlirAttribute a) {
    return mlirMosaicGpuIsACopyPartitionAttr(a);
  }
  using Base::Base;
};

DEFINE_CONCRETE_ATTR(CopyReplicatedAttr, mlirMosaicGpuIsACopyReplicatedAttr,
                     mlirMosaicGpuCopyReplicatedAttrGetTypeID,
                     PyCopyPartitionAttrInterface) {
  cls.def_static(
      "get",
      [](DefaultingPyMlirContext ctx) {
        return PyCopyReplicatedAttr(
            ctx.resolve().getRef(),
            mlirMosaicGpuCopyReplicatedAttrGet(ctx.resolve().get()));
      },
      nb::arg("ctx") = nb::none());
}

DEFINE_CONCRETE_ATTR(CopyPartitionedAttr, mlirMosaicGpuIsACopyPartitionedAttr,
                     mlirMosaicGpuCopyPartitionedAttrGetTypeID,
                     PyCopyPartitionAttrInterface) {
  cls.def_static(
      "get",
      [](int32_t axis, DefaultingPyMlirContext ctx) {
        return PyCopyPartitionedAttr(
            ctx.resolve().getRef(),
            mlirMosaicGpuCopyPartitionedAttrGet(ctx.resolve().get(), axis));
      },
      nb::arg("axis"), nb::arg("ctx") = nb::none());
  cls.def_prop_ro("axis", [](PyCopyPartitionedAttr& self) {
    return mlirMosaicGpuCopyPartitionedAttrGetAxis(self.get());
  });
}

#undef DEFINE_CONCRETE_ATTR

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

  nb::class_<mosaic::gpu::DumpOptions>(m, "DumpOptions")
      .def(nb::init<>())
      .def_ro("mlir_passes", &mosaic::gpu::DumpOptions::mlir_passes)
      .def_ro("ptx", &mosaic::gpu::DumpOptions::ptx)
      .def_ro("ptxas", &mosaic::gpu::DumpOptions::ptxas)
      .def_ro("sass", &mosaic::gpu::DumpOptions::sass)
      .def_ro("sass_ctrl", &mosaic::gpu::DumpOptions::sass_ctrl)
      .def_ro("dump_path", &mosaic::gpu::DumpOptions::dump_path)
      .def_ro("module_basename", &mosaic::gpu::DumpOptions::module_basename);

  m.def(
      "get_or_set_dump_options",
      [](MlirModule c_module) -> mosaic::gpu::DumpOptions {
        return mosaic::gpu::GetOrSetDumpOptionsForModule(unwrap(c_module));
      },
      nb::arg("module"));

  PyBarrierType::bind(m);
  PyB6x16P32Type::bind(m);
  PyP2B6Type::bind(m);
  PyTileTransformAttr::bind(m);
  PySwizzleTransformAttr::bind(m);
  PyWGSplatFragLayoutAttr::bind(m);
  PyWGStridedFragLayoutAttr::bind(m);
  PyReplicatedAttr::bind(m);
  PyTiledLayoutAttr::bind(m);
  PyCopyPartitionAttrInterface::bind(m);
  PyCopyReplicatedAttr::bind(m);
  PyCopyPartitionedAttr::bind(m);
}
