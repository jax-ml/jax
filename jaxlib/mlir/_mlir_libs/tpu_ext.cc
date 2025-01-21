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

#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
// clang-format off
#include "mlir-c/Bindings/Python/Interop.h"
// clang-format on
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"      // IWYU pragma: keep
#include "nanobind/stl/string.h"    // IWYU pragma: keep
#include "nanobind/stl/variant.h"   // IWYU pragma: keep
#include "nanobind/stl/vector.h"    // IWYU pragma: keep
#include "absl/log/check.h"
#include "jaxlib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"
#include "xla/python/nb_numpy.h"
#include "xla/tsl/python/lib/core/numpy.h"

// TODO(tlongeri): Can I add my own return type annotations to functions?
// TODO(tlongeri): I don't understand why MLIR uses the C API to implement
// Python bindings. Do we have a reason to do that?

namespace nb = nanobind;

namespace {
constexpr const char LAYOUT_DEFS_MODULE[] =
    "jax.jaxlib.mosaic.python.layout_defs";
constexpr const char IR_MODULE[] = "jaxlib.mlir.ir";
constexpr MlirTpuI64TargetTuple DEFAULT_TARGET_SHAPE{8, 128};

// TODO(tlongeri): Add type annotations via nanobind once there is
// a release for it (and maybe add a custom Sequence<T> one as well).

// TODO(tlongeri): For our use-case, we don't really need C++ exceptions - just
// setting the exception object and returning NULL to Python should suffice, but
// not sure if this is possible with nanobind.
class NotImplementedException : public std::runtime_error {
  using runtime_error::runtime_error;
};

}  // namespace

template <>
struct nb::detail::type_caster<MlirTpuImplicitDim> {
  NB_TYPE_CASTER(MlirTpuImplicitDim, const_name("ImplicitDim | None"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    if (src.is_none()) {
      value = MlirTpuImplicitDimNone;
      return true;
    }
    auto implicit_dim_cls =
        nb::module_::import_(LAYOUT_DEFS_MODULE).attr("ImplicitDim");
    if (!nb::isinstance(src, implicit_dim_cls)) {
      return false;
    }
    if (src.is(implicit_dim_cls.attr("MINOR"))) {
      value = MlirTpuImplicitDimMinor;
    } else if (src.is(implicit_dim_cls.attr("SECOND_MINOR"))) {
      value = MlirTpuImplicitDimSecondMinor;
    } else {
      return false;
    }
    return true;
  }

  static handle from_cpp(MlirTpuImplicitDim implicit_dim, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    auto implicit_dim_cls =
        nb::module_::import_(LAYOUT_DEFS_MODULE).attr("ImplicitDim");
    switch (implicit_dim) {
      case MlirTpuImplicitDimNone:
        return nb::none().release();
      case MlirTpuImplicitDimMinor:
        return static_cast<nb::object>(implicit_dim_cls.attr("MINOR"))
            .release();
      case MlirTpuImplicitDimSecondMinor:
        return static_cast<nb::object>(implicit_dim_cls.attr("SECOND_MINOR"))
            .release();
    }
  }
};

template <>
struct nb::detail::type_caster<MlirTpuDirection> {
  NB_TYPE_CASTER(MlirTpuDirection, const_name("Direction"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    auto direction_cls =
        nb::module_::import_(LAYOUT_DEFS_MODULE).attr("Direction");
    if (!nb::isinstance(src, direction_cls)) {
      return false;
    }
    if (src.is(direction_cls.attr("LANES"))) {
      value = MlirTpuDirectionLanes;
    } else if (src.is(direction_cls.attr("SUBLANES"))) {
      value = MlirTpuDirectionSublanes;
    } else if (src.is(direction_cls.attr("SUBELEMENTS"))) {
      value = MlirTpuDirectionSubelements;
    } else {
      return false;
    }
    return true;
  }

  static handle from_cpp(MlirTpuDirection direction, rv_policy /* policy */,
                         cleanup_list* /* cleanup */) noexcept {
    auto direction_cls =
        nb::module_::import_(LAYOUT_DEFS_MODULE).attr("ImplicitDim");
    switch (direction) {
      case MlirTpuDirectionLanes:
        return static_cast<nb::object>(direction_cls.attr("LANES")).release();
      case MlirTpuDirectionSublanes:
        return static_cast<nb::object>(direction_cls.attr("SUBLANES"))
            .release();
      case MlirTpuDirectionSubelements:
        return static_cast<nb::object>(direction_cls.attr("SUBELEMENTS"))
            .release();
      default:
        PyErr_Format(PyExc_ValueError, "Invalid MlirTpuDirection: %d",
                     static_cast<int>(direction));
        return nb::handle();
    }
  }
};

template <>
struct nb::detail::type_caster<MlirTpuI64TargetTuple> {
  NB_TYPE_CASTER(MlirTpuI64TargetTuple, const_name("TargetTuple"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    auto target_tuple_cls =
        nb::module_::import_(LAYOUT_DEFS_MODULE).attr("TargetTuple");
    if (!nb::isinstance(src, target_tuple_cls)) {
      return false;
    }
    value = {nb::cast<int64_t>(src.attr("sublanes")),
             nb::cast<int64_t>(src.attr("lanes"))};
    return true;
  }

  static handle from_cpp(MlirTpuI64TargetTuple target_tuple, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    nb::object target_tuple_cls =
        nb::module_::import_(LAYOUT_DEFS_MODULE).attr("TargetTuple");
    return target_tuple_cls(target_tuple.sublane, target_tuple.lane).release();
  }
};

namespace {
// Handler for use with MLIR C API print functions. The 2nd parameter is an
// opaque pointer to "user data" that should always be a string.
void printToString(MlirStringRef c_mlir_str, void* opaque_string) {
  std::string* str = static_cast<std::string*>(opaque_string);
  CHECK(str != nullptr);
  str->append(c_mlir_str.data, c_mlir_str.length);
}

class DiagnosticCapture {
 public:
  DiagnosticCapture(MlirContext ctx)
      : ctx_(ctx),
        id_(mlirContextAttachDiagnosticHandler(ctx, handleDiagnostic, this,
                                               nullptr)) {}

  ~DiagnosticCapture() { mlirContextDetachDiagnosticHandler(ctx_, id_); }

  void throwIfError() {
    if (error_messages_.size() == 1) {
      // Throw NotImplementedException if we got a single diagnostic that
      // contains "Not implemented".
      llvm::StringRef ref = error_messages_.front();
      constexpr llvm::StringRef not_implemented = "Not implemented";
      if (const size_t pos = ref.find(not_implemented);
          pos != llvm::StringRef::npos) {
        // We strip "Not implemented" only if it is a prefix. Sometimes it may
        // come after another prefix (e.g. op prefix), in which case we leave it
        if (pos == 0) {
          ref = ref.drop_front(not_implemented.size());
          ref.consume_front(": ");
        }
        throw NotImplementedException(ref.str());
      }
    }
    if (!error_messages_.empty()) {
      // Note that it is unusual/unexpected to get multiple diagnostics, so we
      // just forward all the error messages.
      throw std::runtime_error(llvm::join(error_messages_, "\n"));
    }
  }

 private:
  static MlirLogicalResult handleDiagnostic(MlirDiagnostic diag,
                                            void* opaque_detector) {
    DiagnosticCapture* detector =
        static_cast<DiagnosticCapture*>(opaque_detector);
    if (mlirDiagnosticGetSeverity(diag) == MlirDiagnosticError) {
      std::string& message = detector->error_messages_.emplace_back();
      mlirDiagnosticPrint(diag, printToString, &message);
    }
    return mlirLogicalResultFailure();  // Propagate to other handlers
  }
  llvm::SmallVector<std::string, 1> error_messages_;
  const MlirContext ctx_;
  const MlirDiagnosticHandlerID id_;
};

}  // namespace

namespace {
nb::object toPyLayoutOffset(int64_t offset) {
  CHECK_GE(offset, -1);
  if (offset == -1) {
    return nb::module_::import_(LAYOUT_DEFS_MODULE).attr("REPLICATED");
  } else {
    return nb::int_(offset);
  }
}

// TODO(tlongeri): Would `type_caster`s let me avoid defining all of these
// to/from functions?
int64_t offsetFromPyOffset(nb::object py_offset) {
  if (nb::isinstance<nb::int_>(py_offset)) {
    int64_t offset = nb::cast<int64_t>(py_offset);
    if (offset < 0) {
      throw nb::value_error("Invalid py layout offset");
    }
    return offset;
  } else if (py_offset.equal(
                 nb::module_::import_(LAYOUT_DEFS_MODULE).attr("REPLICATED"))) {
    return -1;
  } else {
    throw nb::type_error("Invalid layout offset type");
  }
}

template <typename T>
llvm::SmallVector<T> sequenceToSmallVector(nb::sequence seq) {
  llvm::SmallVector<T> out;
  out.reserve(nb::len(seq));
  for (nb::handle elem : seq) {
    out.push_back(nb::cast<T>(elem));
  }
  return out;
}

nb::tuple toPyTuple(const int64_t* data, size_t count) {
  nb::tuple tuple = nb::steal<nb::tuple>(PyTuple_New(count));
  for (size_t i = 0; i < count; ++i) {
    PyTuple_SET_ITEM(tuple.ptr(), i, nb::int_(data[i]).release().ptr());
  }
  return tuple;
}

nb::tuple toPyTuple(MlirTpuI64TargetTuple tuple) {
  return nb::make_tuple(tuple.sublane, tuple.lane);
}

// Unwraps the current default insertion point
// ValueError is raised if default insertion point is not set
MlirTpuInsertionPoint getDefaultInsertionPoint() {
  nb::object insertion_point =
      nb::module_::import_(IR_MODULE).attr("InsertionPoint").attr("current");
  nb::object ref_operation = insertion_point.attr("ref_operation");
  return {nb::cast<MlirBlock>(insertion_point.attr("block")),
          ref_operation.is_none()
              ? MlirOperation{nullptr}
              : nb::cast<MlirOperation>(insertion_point.attr("ref_operation"))};
}

// Unwraps the current default location
// ValueError is raised if default location is not set
MlirLocation getDefaultLocation() {
  return nb::cast<MlirLocation>(
      nb::module_::import_(IR_MODULE).attr("Location").attr("current"));
}

// Unwraps the current default MLIR context
// ValueError is raised if default context is not set
MlirContext getDefaultContext() {
  return nb::cast<MlirContext>(
      nb::module_::import_(IR_MODULE).attr("Context").attr("current"));
}

struct PyTpuVectorLayout {
  PyTpuVectorLayout(MlirTpuVectorLayout layout) : layout(layout) {}
  ~PyTpuVectorLayout() { mlirTpuVectorLayoutDestroy(layout); }
  PyTpuVectorLayout(const PyTpuVectorLayout&) = delete;
  PyTpuVectorLayout& operator=(const PyTpuVectorLayout&) = delete;

  MlirTpuVectorLayout layout;
};

}  // namespace

NB_MODULE(_tpu_ext, m) {
  tsl::ImportNumpy();
  mlirRegisterTPUPasses();  // Register all passes on load.
  mlirTpuRegisterMosaicSerdePass();

  nb::class_<MlirTpuApplyVectorLayoutContext>(m, "ApplyVectorLayoutCtx")
      .def(
          "__init__",
          [](MlirTpuApplyVectorLayoutContext* self, int hardware_generation,
             nb::tuple target_shape, nb::tuple mxu_shape,
             int max_sublanes_in_scratch) {
            if (target_shape.size() != 2) {
              throw nb::value_error("target_shape should be of length 2");
            }
            if (mxu_shape.size() != 2) {
              throw nb::value_error("mxu_shape should be of length 2");
            }
            new (self) MlirTpuApplyVectorLayoutContext{
                .hardware_generation = hardware_generation,
                .target_shape = {nb::cast<int64_t>(target_shape[0]),
                                 nb::cast<int64_t>(target_shape[1])},
                .mxu_shape = {nb::cast<int64_t>(mxu_shape[0]),
                              nb::cast<int64_t>(mxu_shape[1])},
                .max_sublanes_in_scratch = max_sublanes_in_scratch};
          },
          nb::arg("hardware_generation") = -1,
          nb::arg("target_shape") = toPyTuple(DEFAULT_TARGET_SHAPE),
          nb::arg("mxu_shape") = nb::make_tuple(128, 128),
          nb::arg("max_sublanes_in_scratch") = 0);

  nb::class_<MlirTpuVregDataBounds>(m, "VRegDataBounds")
      .def("mask_varies_along",
           [](MlirTpuVregDataBounds self, MlirTpuDirection direction,
              MlirTpuI64TargetTuple target_shape) {
             return mlirTpuVregDataBoundsMaskVariesAlong(self, direction,
                                                         target_shape);
           })
      .def("complete",
           [](MlirTpuVregDataBounds self, MlirTpuI64TargetTuple target_shape) {
             return mlirTpuVregDataBoundsIsComplete(self, target_shape);
           })
      .def("get_vector_mask",
           [](MlirTpuVregDataBounds self, int generation,
              MlirTpuI64TargetTuple target_shape) {
             // TODO: Does this work? Test in Python
             MlirValue mask = mlirTpuVregDataBoundsGetVectorMask(
                 self, getDefaultInsertionPoint(), getDefaultLocation(),
                 generation, target_shape);
             if (mask.ptr == nullptr) {
               throw std::runtime_error("getVectorMask failed");
             }
             return mask;
           })
      .def("get_sublane_mask",
           [](MlirTpuVregDataBounds self, MlirTpuI64TargetTuple target_shape) {
             return mlirTpuVregDataBoundsGetSublaneMask(
                 self, getDefaultContext(), target_shape);
           });

  // TODO(tlongeri): More precise argument type annotations. There currently
  // seems to be no way to define your own?
  nb::class_<PyTpuVectorLayout>(m, "VectorLayout")
      .def(
          "__init__",
          [](PyTpuVectorLayout* self, int bitwidth, nb::tuple offsets,
             nb::tuple tiling, MlirTpuImplicitDim implicit_dim) {
            if (offsets.size() != 2) {
              throw nb::value_error("Offsets should be of length 2");
            }
            if (tiling.size() != 2) {
              throw nb::value_error("Tiling should be of length 2");
            }
            MlirTpuVectorLayout layout = mlirTpuVectorLayoutCreate(
                bitwidth,
                {offsetFromPyOffset(offsets[0]),
                 offsetFromPyOffset(offsets[1])},
                {nb::cast<int64_t>(tiling[0]), nb::cast<int64_t>(tiling[1])},
                implicit_dim);
            new (self) PyTpuVectorLayout(layout);
          },
          nb::arg("bitwidth"), nb::arg("offsets"), nb::arg("tiling"),
          nb::arg("implicit_dim").none())
      .def_prop_ro(
          "bitwidth",
          [](const PyTpuVectorLayout& self) {
            return mlirTpuVectorLayoutGetBitwidth(self.layout);
          },
          "The bitwidth of the stored values.")
      .def_prop_ro(
          "offsets",
          [](const PyTpuVectorLayout& self) {
            MlirTpuLayoutOffsets offsets =
                mlirTpuVectorLayoutGetOffsets(self.layout);
            return nb::make_tuple(toPyLayoutOffset(offsets.sublane),
                                  toPyLayoutOffset(offsets.lane));
          },
          "The coordinates of the first valid element. If an offset is "
          "REPLICATED, then any offset is valid as the value does not vary "
          "across sublanes or lanes respectively.")
      .def_prop_ro(
          "tiling",
          [](const PyTpuVectorLayout& self) {
            return toPyTuple(mlirTpuVectorLayoutGetTiling(self.layout));
          },
          "The tiling used to lay out values (see the XLA docs). For values of "
          "bitwidth < 32, an implicit (32 // bitwidth, 1) tiling is appended "
          "to the one specified as an attribute.")
      .def_prop_ro(
          "implicit_dim",
          [](const PyTpuVectorLayout& self) {
            return mlirTpuVectorLayoutGetImplicitDim(self.layout);
          },
          "If specified, the value has an implicit dim inserted in either "
          "minormost or second minormost position.")
      .def_prop_ro(
          "packing",
          [](const PyTpuVectorLayout& self) {
            return mlirTpuVectorLayoutGetPacking(self.layout);
          },
          "Returns the number of values stored in a vreg entry.")
      .def_prop_ro(
          "layout_rank",
          [](const PyTpuVectorLayout& self) {
            return mlirTpuVectorLayoutGetLayoutRank(self.layout);
          },
          "The number of minormost dimensions tiled by this layout.")
      .def(
          "has_natural_topology",
          [](const PyTpuVectorLayout& self,
             MlirTpuI64TargetTuple target_shape) {
            return mlirTpuVectorLayoutHasNaturalTopology(self.layout,
                                                         target_shape);
          },
          nb::arg("target_shape"),
          "True, if every vector register has a layout without jumps.\n"
          "\n"
          "By without jumps we mean that traversing vregs over (sub)lanes "
          "always leads to a contiguous traversal of the (second) minormost "
          "dimension of data. This is only true for 32-bit types, since "
          "narrower types use two level tiling.")
      .def(
          "has_native_tiling",
          [](const PyTpuVectorLayout& self,
             MlirTpuI64TargetTuple target_shape) {
            return mlirTpuVectorLayoutHasNativeTiling(self.layout,
                                                      target_shape);
          },
          nb::arg("target_shape"),
          "True, if every vector register has a natural \"packed\" topology.\n"
          "\n"
          "This is equivalent to has_natural_topology for 32-bit types, but "
          "generalizes it to narrower values with packed layouts too.")
      .def(
          "tiles_per_vreg",
          [](const PyTpuVectorLayout& self,
             MlirTpuI64TargetTuple target_shape) {
            return mlirTpuVectorLayoutTilesPerVreg(self.layout, target_shape);
          },
          nb::arg("target_shape"),
          "How many tiles fit in each vector register.")
      .def(
          "sublanes_per_tile",
          [](const PyTpuVectorLayout& self,
             MlirTpuI64TargetTuple target_shape) {
            return mlirTpuVectorLayoutSublanesPerTile(self.layout,
                                                      target_shape);
          },
          nb::arg("target_shape"),
          "The number of sublanes necessary to store each tile.")
      .def(
          "vreg_slice",
          [](const PyTpuVectorLayout& self,
             MlirTpuI64TargetTuple target_shape) {
            MlirTpuI64TargetTuple vreg_slice =
                mlirTpuVectorLayoutVregSlice(self.layout, target_shape);
            return nb::module_::import_(LAYOUT_DEFS_MODULE)
                .attr("TargetTuple")(vreg_slice.sublane, vreg_slice.lane);
          },
          nb::arg("target_shape"),
          "Returns the size of a window contained in a single vreg.\n"
          "\n"
          "We never reuse the same vector register to store data of multiple "
          "rows, so only the minormost dimension can increase.")
      .def(
          "implicit_shape",
          [](const PyTpuVectorLayout& self, nb::sequence shape) {
            llvm::SmallVector<int64_t> implicit_shape_vec =
                sequenceToSmallVector<int64_t>(shape);
            MlirTpuI64ArrayRef implicit_shape =
                mlirTpuVectorLayoutImplicitShape(
                    self.layout,
                    {implicit_shape_vec.data(), implicit_shape_vec.size()});
            nb::tuple ret = toPyTuple(implicit_shape.ptr, implicit_shape.size);
            free(implicit_shape.ptr);
            return ret;
          },
          nb::arg("shape"))
      .def(
          "tile_array_shape",
          [](const PyTpuVectorLayout& self, nb::sequence shape,
             MlirTpuI64TargetTuple target_shape) {
            llvm::SmallVector<int64_t> tile_array_shape_vec =
                sequenceToSmallVector<int64_t>(shape);
            MlirTpuI64ArrayRef tile_array_shape =
                mlirTpuVectorLayoutTileArrayShape(
                    self.layout,
                    {tile_array_shape_vec.data(), tile_array_shape_vec.size()},
                    target_shape);
            nb::tuple ret =
                toPyTuple(tile_array_shape.ptr, tile_array_shape.size);
            free(tile_array_shape.ptr);
            return ret;
          },
          nb::arg("shape"), nb::arg("target_shape"),
          "Returns the shape of an ndarray of vregs needed to represent a "
          "value.\n"
          "\n"
          "All but the last two dimensions are unrolled over vregs. In the "
          "last two dims we need as many vregs as indicated by dividing the "
          "point at which the value ends (given by the start offset plus the "
          "dim size) divided by the respective vreg capacity in that dim (and "
          "a ceiling if non-integral). If a value is replicated, then any "
          "offset is valid and we pick 0 to minimize the number of vregs.\n"
          "\n"
          "Args:\n"
          "  shape: The shape of the ndarray to tile.")
      .def(
          "tile_data_bounds",
          [](const PyTpuVectorLayout& self, nb::sequence shape,
             nb::sequence ixs, MlirTpuI64TargetTuple target_shape,
             std::variant<bool, nb::tuple> allow_replicated) {
            llvm::SmallVector<int64_t> shape_vec =
                sequenceToSmallVector<int64_t>(shape);
            llvm::SmallVector<int64_t> ixs_vec =
                sequenceToSmallVector<int64_t>(ixs);
            if (shape_vec.size() != ixs_vec.size()) {
              throw nb::value_error(
                  "Expected shape and ixs to have the same size");
            }
            return std::visit(
                [&](auto ar) {
                  if constexpr (std::is_same_v<decltype(ar), bool>) {
                    return mlirTpuVectorLayoutTileDataBounds(
                        self.layout, getDefaultContext(), shape_vec.data(),
                        ixs_vec.data(), shape_vec.size(), target_shape,
                        {ar, ar});
                  } else {
                    return mlirTpuVectorLayoutTileDataBounds(
                        self.layout, getDefaultContext(), shape_vec.data(),
                        ixs_vec.data(), shape_vec.size(), target_shape,
                        {nb::cast<bool>(ar[0]), nb::cast<bool>(ar[1])});
                  }
                },
                allow_replicated);
          },
          nb::arg("shape"), nb::arg("ixs"), nb::arg("target_shape"),
          nb::arg("allow_replicated") = false,
          "Returns the bounds of the given tile that hold useful data.\n"
          "\n"
          "Arguments:\n"
          "  full_shape: The shape of the full vector this layout applies to.\n"
          "  ixs: The indices into an array of tiles representing the full "
          "vector (see tile_array_shape for bounds) selecting the tile for "
          "which the bounds are queried.\n"
          "  allow_replicated: If False, no offset is allowed to be "
          "REPLICATED. If True, offsets are allowed to be REPLICATED, but the "
          "bounds will span the full dimension of the tile (i.e. potentially "
          "multiple repeats of the actual data).\n"
          "  target_shape: The target shape of the TPU.\n"
          "\n"
          "Returns:\n"
          "  A TargetTuple of slices, indicating the span of useful data "
          "within the tile selected by idx.")
      .def(
          "generalizes",
          [](const PyTpuVectorLayout& self, const PyTpuVectorLayout& other,
             std::optional<nb::sequence> shape,
             MlirTpuI64TargetTuple target_shape) {
            if (shape) {
              llvm::SmallVector<int64_t> shape_vec =
                  sequenceToSmallVector<int64_t>(*shape);
              return mlirTpuVectorLayoutGeneralizes(
                  self.layout, other.layout,
                  {shape_vec.data(), shape_vec.size()}, target_shape);
            }
            return mlirTpuVectorLayoutGeneralizes(self.layout, other.layout,
                                                  {nullptr, 0}, target_shape);
          },
          nb::arg("other"), nb::kw_only(),
          nb::arg("shape").none() = std::nullopt, nb::arg("target_shape"),
          "Returns True if the other layout is a special case of this one.\n"
          "\n"
          "In here, other is considered \"a special case\" when the set of "
          "vector register entries that represent a value in that layout is "
          "also the set of entries in which self stores the value. This is of "
          "course true for layouts that are equivalent, but it does not need "
          "to hold both ways. For example, a layout that implies the value "
          "does not change along an axis of the vector register is more "
          "general than the layout that picks a fixed starting point for the "
          "value and does not encode that assumption.\n"
          "\n"
          "The generalization relation is a non-strict partial order. You can "
          "think of it as a partial <= on vector layouts, but we don't "
          "overload Python operators since there's no clear way to decide "
          "where the bottom and top should be.\n"
          "\n"
          "Args:\n"
          "  other: The layout compared against self.\n"
          "  shape: An optional shape of the vector to which both layouts "
          "apply.\n"
          "    The generalization relation is larger than usual for some "
          "shapes. That is, if self.generalizes(other) then also "
          "self.generalizes(other, shape) for any shape, but that implication "
          "does not hold the other way around for some shapes.\n"
          "  target_shape: The target shape of the TPU.")
      .def(
          "equivalent_to",
          [](const PyTpuVectorLayout& self, const PyTpuVectorLayout& other,
             std::optional<nb::sequence> shape,
             MlirTpuI64TargetTuple target_shape) {
            if (shape) {
              llvm::SmallVector<int64_t> shape_vec =
                  sequenceToSmallVector<int64_t>(*shape);
              return mlirTpuVectorLayoutEquivalentTo(
                  self.layout, other.layout,
                  {shape_vec.data(), shape_vec.size()}, target_shape);
            }
            return mlirTpuVectorLayoutEquivalentTo(self.layout, other.layout,
                                                   {nullptr, 0}, target_shape);
          },
          nb::arg("other"), nb::kw_only(),
          nb::arg("shape").none() = std::nullopt, nb::arg("target_shape"),
          "Returns True if the two layouts are equivalent.\n"
          "\n"
          "That is, when all potential vector entries where the value can be "
          "stored (there might be multiple choices for some layouts!) are "
          "equal in both self and other.\n"
          "\n"
          "Args:\n"
          "  other: The layout compared against self.\n"
          "  shape: An optional shape of the vector to which both layouts "
          "apply. More layouts are considered equivalent when the shape is "
          "specified. Also see the docstring of the generalizes method.\n"
          "  target_shape: The target shape of the TPU.")
      .def("__eq__",
           [](const PyTpuVectorLayout& self, const PyTpuVectorLayout& other) {
             return mlirTpuVectorLayoutEquals(self.layout, other.layout);
           })
      .def("__repr__", [](const PyTpuVectorLayout& self) {
        std::string str;
        mlirTpuVectorLayoutPrint(self.layout, printToString, &str);
        return str;
      });

  // TODO(tlongeri): Can we make the first parameter a VectorType?
  m.def("assemble",
        [](const MlirType ty, const PyTpuVectorLayout& layout,
           nb::object np_arr_obj,
           MlirTpuI64TargetTuple target_shape) -> MlirOperation {
          // TODO(tlongeri): Remove nb::array::c_style, I only added it because
          // I couldn't find a simple way to iterate over array data, but it
          // causes yet another unnecessary copy.
          xla::nb_numpy_ndarray np_arr = xla::nb_numpy_ndarray::ensure(
              np_arr_obj, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
          if (!mlirTypeIsAVector(ty)) {
            throw nb::type_error("Expected vector type");
          }
          llvm::SmallVector<MlirValue> vals(np_arr.size());
          for (int64_t i = 0; i < np_arr.size(); ++i) {
            vals.data()[i] = nb::cast<MlirValue>(nb::handle(
                reinterpret_cast<PyObject* const*>(np_arr.data())[i]));
          }
          llvm::SmallVector<int64_t> shape(np_arr.ndim());
          for (int64_t i = 0; i < np_arr.ndim(); ++i) {
            shape.data()[i] = np_arr.shape()[i];
          }
          return mlirTpuAssemble(
              getDefaultInsertionPoint(), ty, layout.layout,
              MlirTpuValueArray{MlirTpuI64ArrayRef{shape.data(), shape.size()},
                                vals.data()},
              target_shape);
        });
  m.def("disassemble", [](const PyTpuVectorLayout& layout, MlirValue val,
                          MlirTpuI64TargetTuple target_shape) {
    DiagnosticCapture diag_capture(getDefaultContext());
    MlirTpuValueArray val_arr = mlirTpuDisassemble(
        getDefaultInsertionPoint(), layout.layout, val, target_shape);
    if (val_arr.vals == nullptr) {
      diag_capture.throwIfError();
      throw nb::value_error("Failed to disassemble");
    }
    xla::nb_numpy_ndarray np_vals(
        /*dtype=*/xla::nb_dtype("O"),
        /*shape=*/
        absl::Span<int64_t const>(val_arr.shape.ptr, val_arr.shape.size),
        /*strides=*/std::nullopt);
    for (ssize_t i = 0; i < np_vals.size(); ++i) {
      reinterpret_cast<PyObject**>(np_vals.mutable_data())[i] =
          nb::cast(val_arr.vals[i]).release().ptr();
    }
    free(val_arr.shape.ptr);
    free(val_arr.vals);
    return np_vals;
  });

  m.def("apply_layout_op",
        [](MlirTpuApplyVectorLayoutContext ctx, const MlirOperation c_op) {
          DiagnosticCapture diag_capture(getDefaultContext());
          MlirLogicalResult res = mlirTpuApplyLayoutOp(ctx, c_op);
          if (mlirLogicalResultIsFailure(res)) {
            diag_capture.throwIfError();
            throw std::runtime_error("applyLayoutOp failed");
          }
        });
  m.def("relayout", [](MlirValue v, const PyTpuVectorLayout& src,
                       const PyTpuVectorLayout& dst,
                       MlirTpuApplyVectorLayoutContext apply_layout_ctx) {
    DiagnosticCapture diag_capture(getDefaultContext());
    MlirValue new_v = mlirTpuRelayout(getDefaultInsertionPoint(), v, src.layout,
                                      dst.layout, apply_layout_ctx);
    if (new_v.ptr == nullptr) {
      diag_capture.throwIfError();
      throw nb::value_error("Failed to relayout");
    }
    return new_v;
  });
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void*) {
        try {
          if (p) std::rethrow_exception(p);
        } catch (const NotImplementedException& e) {
          PyErr_SetString(PyExc_NotImplementedError, e.what());
        }
      },
      nullptr);

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

  m.def("private_is_tiled_layout", [](MlirAttribute attr) {
    return mlirTPUAttributeIsATiledLayoutAttr(attr);
  });
  m.def("private_get_tiles", [](MlirAttribute attr) -> nb::object {
    MlirAttribute encoded_tiles = mlirTPUTiledLayoutAttrGetTiles(attr);
    nb::tuple py_tiles = nb::steal<nb::tuple>(
        PyTuple_New(mlirArrayAttrGetNumElements(encoded_tiles)));
    for (intptr_t i = 0; i < mlirArrayAttrGetNumElements(encoded_tiles); ++i) {
      MlirAttribute tile = mlirArrayAttrGetElement(encoded_tiles, i);
      nb::tuple py_tile =
          nb::steal<nb::tuple>(PyTuple_New(mlirDenseArrayGetNumElements(tile)));
      for (intptr_t j = 0; j < mlirDenseArrayGetNumElements(tile); ++j) {
        PyTuple_SET_ITEM(
            py_tile.ptr(), j,
            nb::cast(mlirDenseI64ArrayGetElement(tile, j)).release().ptr());
      }
      PyTuple_SET_ITEM(py_tiles.ptr(), i, py_tile.release().ptr());
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
      throw nb::value_error("length mismatch in replace_all_uses_with");
    }
    for (int i = 0; i < vals.size(); ++i) {
      mlirValueReplaceAllUsesOfWith(mlirOperationGetResult(op, i), vals[i]);
    }
  });
  m.def("private_replace_all_uses_except",
        [](MlirValue old, MlirValue new_val, MlirOperation except) {
          for (intptr_t i = 0; i < mlirOperationGetNumOperands(except); ++i) {
            if (mlirValueEqual(mlirOperationGetOperand(except, i), new_val)) {
              throw nb::value_error("new val already used in except");
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
      throw nb::value_error(
          "Region counts do not match in src operation and dst operations");
    }
    for (intptr_t i = 0; i < mlirOperationGetNumRegions(src); ++i) {
      mlirRegionTakeBody(mlirOperationGetRegion(dst, i),
                         mlirOperationGetRegion(src, i));
    }
  });
}
