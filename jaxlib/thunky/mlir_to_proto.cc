/* Copyright 2026 The JAX Authors.

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

#include "jaxlib/thunky/mlir_to_proto.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/thunky/dialect/thunky_dialect.h"
#include "jaxlib/thunky/thunk_proto_util.h"
#include "riegeli/bytes/string_reader.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/reduction_kind.pb.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/attribute_map.pb.h"
#include "xla/ffi/execution_state.pb.h"
#include "xla/literal.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/dense_data_intermediate.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/launch_dimensions.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_slice.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/launch_dim.pb.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {

using ::xla::buffer_assignment::BufferAllocationSliceProto;

//===----------------------------------------------------------------------===//
// Shared Proto <-> MLIR Helpers
//===----------------------------------------------------------------------===//

// Converts an XLA `ReductionKindProto` enum to its `thunky` MLIR string
// attribute representation (`"sum"`, `"prod"`, `"min"`, `"max"`).
std::string ReductionKindToString(xla::ReductionKindProto kind) {
  switch (kind) {
    case xla::ReductionKindProto::REDUCTION_KIND_SUM:
      return "sum";
    case xla::ReductionKindProto::REDUCTION_KIND_PRODUCT:
      return "prod";
    case xla::ReductionKindProto::REDUCTION_KIND_MIN:
      return "min";
    case xla::ReductionKindProto::REDUCTION_KIND_MAX:
      return "max";
    default:
      return "sum";
  }
}

// Converts a `thunky` MLIR reduction kind string attribute into an XLA
// `ReductionKindProto` enum.
absl::StatusOr<xla::ReductionKindProto> StringToReductionKind(
    absl::string_view str) {
  if (str == "sum") return xla::ReductionKindProto::REDUCTION_KIND_SUM;
  if (str == "prod" || str == "product") {
    return xla::ReductionKindProto::REDUCTION_KIND_PRODUCT;
  }
  if (str == "min") return xla::ReductionKindProto::REDUCTION_KIND_MIN;
  if (str == "max") return xla::ReductionKindProto::REDUCTION_KIND_MAX;
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown reduction kind: %s", str));
}

// Converts an XLA `CollectiveOpGroupMode` enum to its `thunky` MLIR string
// attribute representation.
std::string GroupModeToString(CollectiveOpGroupMode mode) {
  switch (mode) {
    case COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA:
      return "cross_replica";
    case COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION:
      return "cross_partition";
    case COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID:
      return "flattened_id";
    case COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION:
      return "cross_replica_and_partition";
    default:
      return "flattened_id";
  }
}

// Converts a `thunky` MLIR collective group mode string attribute into an XLA
// `CollectiveOpGroupMode` enum.
absl::StatusOr<CollectiveOpGroupMode> StringToGroupMode(absl::string_view str) {
  if (str == "flattened_id") return COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID;
  if (str == "cross_replica") return COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  if (str == "cross_partition") return COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION;
  if (str == "cross_replica_and_partition") {
    return COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown collective group mode: %s", str));
}

// Extracts the first operand element type from `config` as an MLIR `TypeAttr`
// (defaulting to `f32` if `operand_element_type` is empty).
mlir::TypeAttr ElemTypeToTypeAttr(mlir::Builder& builder,
                                  const CollectiveConfigProto& config) {
  if (config.operand_element_type_size() > 0) {
    auto statusor_type = xla::ConvertPrimitiveTypeToMlirType(
        static_cast<PrimitiveType>(config.operand_element_type(0)), builder);
    if (statusor_type.ok()) {
      return mlir::TypeAttr::get(*statusor_type);
    }
  }
  return mlir::TypeAttr::get(builder.getF32Type());
}

// Converts the `ReplicaGroup` list in `config` to an MLIR `ArrayAttr` of
// `DenseI64ArrayAttr`s.
mlir::ArrayAttr ReplicaGroupsToAttr(mlir::OpBuilder& builder,
                                    const CollectiveConfigProto& config) {
  llvm::SmallVector<mlir::Attribute> group_attrs;
  for (const auto& rg : config.replica_groups()) {
    llvm::SmallVector<int64_t> ids(rg.replica_ids().begin(),
                                   rg.replica_ids().end());
    group_attrs.push_back(builder.getDenseI64ArrayAttr(ids));
  }
  return builder.getArrayAttr(group_attrs);
}

// Converts an MLIR `ArrayAttr` of replica groups (each element either a
// `DenseI64ArrayAttr` or a nested `ArrayAttr` of `IntegerAttr`) into XLA
// `ReplicaGroup` protos.
std::vector<ReplicaGroup> AttrToReplicaGroups(mlir::ArrayAttr rg_attr) {
  std::vector<ReplicaGroup> result;
  for (mlir::Attribute attr : rg_attr) {
    ReplicaGroup& rg = result.emplace_back();
    if (auto dense_arr = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
      for (int64_t id : dense_arr.asArrayRef()) {
        rg.add_replica_ids(id);
      }
    } else if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      for (mlir::Attribute elem : arr) {
        rg.add_replica_ids(mlir::cast<mlir::IntegerAttr>(elem).getInt());
      }
    }
  }
  return result;
}

// Builds a `CollectiveConfigProto` from a `thunky` collective operation's
// element type, `replica_groups` attribute, and `group_mode` attribute.
absl::StatusOr<CollectiveConfigProto> MakeCollectiveConfigProto(
    mlir::Type elem_type, mlir::ArrayAttr rg_attr,
    absl::string_view group_mode_str) {
  PrimitiveType primitive_type = xla::ConvertMlirTypeToPrimitiveType(elem_type);
  if (primitive_type == xla::PRIMITIVE_TYPE_INVALID) {
    return absl::InvalidArgumentError(
        "Unsupported collective element type in MLIR");
  }
  ABSL_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                        StringToGroupMode(group_mode_str));
  CollectiveConfigProto config;
  config.add_operand_element_type(primitive_type);
  for (ReplicaGroup& rg : AttrToReplicaGroups(rg_attr)) {
    *config.add_replica_groups() = std::move(rg);
  }
  config.set_group_mode(group_mode);
  config.set_use_symmetric_buffer(false);
  return config;
}

// Converts an XLA `ShapeProto` into an MLIR `RankedTensorType` `TypeAttr` for
// `thunky.custom_call` `operand_shapes` / `result_shapes`.
mlir::TypeAttr ShapeProtoToTypeAttr(mlir::OpBuilder& builder,
                                    const xla::ShapeProto& shape) {
  auto statusor_type =
      xla::ConvertPrimitiveTypeToMlirType(shape.element_type(), builder);
  mlir::Type elem_type =
      statusor_type.ok() ? *statusor_type : builder.getF32Type();
  llvm::SmallVector<int64_t> dims(shape.dimensions().begin(),
                                  shape.dimensions().end());
  return mlir::TypeAttr::get(mlir::RankedTensorType::get(dims, elem_type));
}

// Converts an MLIR `TypeAttr` (either a `RankedTensorType` or a scalar element
// type) from `thunky.custom_call` into an `xla::Shape`, falling back to a flat
// `u8[fallback_bytes]` shape when `attr` is null.
absl::StatusOr<xla::Shape> TypeAttrToShape(mlir::Attribute attr,
                                           int64_t fallback_bytes) {
  if (!attr) {
    return xla::ShapeUtil::MakeShape(xla::U8, {fallback_bytes});
  }
  if (auto type_attr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
    mlir::Type type = type_attr.getValue();
    if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      PrimitiveType elem_type =
          xla::ConvertMlirTypeToPrimitiveType(ranked.getElementType());
      if (elem_type == xla::PRIMITIVE_TYPE_INVALID) {
        return absl::InvalidArgumentError(
            "Invalid element type in custom_call shape");
      }
      return xla::ShapeUtil::MakeShape(elem_type, ranked.getShape());
    }
    PrimitiveType elem_type = xla::ConvertMlirTypeToPrimitiveType(type);
    if (elem_type == xla::PRIMITIVE_TYPE_INVALID) {
      return absl::InvalidArgumentError(
          "Invalid element type in custom_call shape");
    }
    return xla::ShapeUtil::MakeShape(elem_type, {});
  }
  return absl::InvalidArgumentError(
      "custom_call operand_shapes and result_shapes must be TypeAttr");
}

// Converts an `xla::ffi::AttributesMapProto` from a `CustomCallThunkProto` into
// an MLIR `DictionaryAttr` suitable for `thunky.custom_call`'s `backend_config`
// (the inverse of `xla::ffi::BuildAttributesMap`).
mlir::DictionaryAttr ConvertAttributesMapProtoToMlirDict(
    mlir::OpBuilder& builder, const xla::ffi::AttributesMapProto& proto) {
  llvm::SmallVector<mlir::NamedAttribute> named_attrs;
  std::vector<std::string> keys;
  keys.reserve(proto.attrs().size());
  for (const auto& [k, _] : proto.attrs()) {
    keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());
  for (const std::string& k : keys) {
    const xla::ffi::AttributeProto& attr = proto.attrs().at(k);
    mlir::Attribute mlir_attr;
    switch (attr.value_case()) {
      case xla::ffi::AttributeProto::kScalar: {
        const auto& s = attr.scalar();
        switch (s.value_case()) {
          case xla::ffi::ScalarProto::kB:
            mlir_attr = builder.getBoolAttr(s.b());
            break;
          case xla::ffi::ScalarProto::kI8:
            mlir_attr = builder.getI8IntegerAttr(s.i8());
            break;
          case xla::ffi::ScalarProto::kI16:
            mlir_attr = builder.getI16IntegerAttr(s.i16());
            break;
          case xla::ffi::ScalarProto::kI32:
            mlir_attr = builder.getI32IntegerAttr(s.i32());
            break;
          case xla::ffi::ScalarProto::kI64:
            mlir_attr = builder.getI64IntegerAttr(s.i64());
            break;
          case xla::ffi::ScalarProto::kU8:
            mlir_attr = builder.getIntegerAttr(
                builder.getIntegerType(8, /*isSigned=*/false), s.u8());
            break;
          case xla::ffi::ScalarProto::kU16:
            mlir_attr = builder.getIntegerAttr(
                builder.getIntegerType(16, /*isSigned=*/false), s.u16());
            break;
          case xla::ffi::ScalarProto::kU32:
            mlir_attr = builder.getIntegerAttr(
                builder.getIntegerType(32, /*isSigned=*/false), s.u32());
            break;
          case xla::ffi::ScalarProto::kU64:
            mlir_attr = builder.getIntegerAttr(
                builder.getIntegerType(64, /*isSigned=*/false), s.u64());
            break;
          case xla::ffi::ScalarProto::kF32:
            mlir_attr = builder.getF32FloatAttr(s.f32());
            break;
          case xla::ffi::ScalarProto::kF64:
            mlir_attr = builder.getF64FloatAttr(s.f64());
            break;
          default:
            break;
        }
        break;
      }
      case xla::ffi::AttributeProto::kStr:
        mlir_attr = builder.getStringAttr(attr.str());
        break;
      case xla::ffi::AttributeProto::kArray: {
        const auto& arr = attr.array();
        switch (arr.value_case()) {
          case xla::ffi::ArrayProto::kI32:
            mlir_attr = builder.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(
                arr.i32().values().data(), arr.i32().values_size()));
            break;
          case xla::ffi::ArrayProto::kI64:
            mlir_attr = builder.getDenseI64ArrayAttr(llvm::ArrayRef<int64_t>(
                arr.i64().values().data(), arr.i64().values_size()));
            break;
          case xla::ffi::ArrayProto::kF32:
            mlir_attr = builder.getDenseF32ArrayAttr(llvm::ArrayRef<float>(
                arr.f32().values().data(), arr.f32().values_size()));
            break;
          case xla::ffi::ArrayProto::kF64:
            mlir_attr = builder.getDenseF64ArrayAttr(llvm::ArrayRef<double>(
                arr.f64().values().data(), arr.f64().values_size()));
            break;
          default:
            break;
        }
        break;
      }
      case xla::ffi::AttributeProto::kDict:
        mlir_attr = ConvertAttributesMapProtoToMlirDict(builder, attr.dict());
        break;
      default:
        break;
    }
    if (mlir_attr) {
      named_attrs.push_back(builder.getNamedAttr(k, mlir_attr));
    }
  }
  return builder.getDictionaryAttr(named_attrs);
}

//===----------------------------------------------------------------------===//
// Proto -> MLIR Helpers
//===----------------------------------------------------------------------===//

// Rewrites `buffer_allocation_index` in every `BufferAllocationSliceProto`
// inside `thunk` according to `mapping` (used to localize a `ThunkProto`'s
// buffer indices to the 0-based operand list of a `thunky.call_thunk_proto`).
absl::Status RemapBufferAllocations(
    ThunkProto* thunk, const absl::flat_hash_map<int64_t, int64_t>& mapping) {
  return ForEachBufferSlice(thunk, [&](auto* slice) {
    auto it = mapping.find(slice->buffer_allocation_index());
    if (it != mapping.end()) {
      slice->set_buffer_allocation_index(it->second);
    }
  });
}

// Reproduces XLA:GPU's constant symbol naming convention (`"buffer_for_..."`)
// so `ConstantInfoProto::symbol_name` can be matched back to its originating
// `HloInstructionProto` in `HloModuleProto`.
std::string SanitizeConstantSymbolName(absl::string_view name) {
  std::string instr_name(name);
  std::replace_if(
      instr_name.begin(), instr_name.end(),
      [](char c) { return c == '.' || c == '-' || c == ';'; }, '_');
  return absl::StrCat("buffer_for_", instr_name);
}

// Indexes all `"constant"` instructions in `gpu_proto`'s `HloModuleProto` by
// their sanitized symbol name (`"buffer_for_<name>"`).
absl::flat_hash_map<std::string, const HloInstructionProto*>
MakeConstantInstructionProtoMap(const GpuExecutableProto& gpu_proto) {
  absl::flat_hash_map<std::string, const HloInstructionProto*> map;
  if (!gpu_proto.has_hlo_module_with_config()) {
    return map;
  }
  for (const auto& comp :
       gpu_proto.hlo_module_with_config().hlo_module().computations()) {
    for (const auto& instr : comp.instructions()) {
      if (instr.opcode() == "constant") {
        map[SanitizeConstantSymbolName(instr.name())] = &instr;
      }
    }
  }
  return map;
}

// Extracts the raw device-layout bytes for `constant_proto`. Uses
// `constant_proto.content()` directly if populated; otherwise reconstructs the
// `Literal` from the corresponding `HloInstructionProto` in `symbol_to_instr`
// (since XLA:GPU clears `ConstantInfoProto::content` when PTX/cubin already
// embeds the constant symbol).
absl::StatusOr<std::string> ExtractConstantBytes(
    const GpuExecutableProto::ConstantInfoProto& constant_proto,
    const absl::flat_hash_map<std::string, const HloInstructionProto*>&
        symbol_to_instr) {
  if (constant_proto.has_content() &&
      !constant_proto.content().data().empty()) {
    return constant_proto.content().data();
  }
  auto it = symbol_to_instr.find(constant_proto.symbol_name());
  if (it == symbol_to_instr.end()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Instruction for constant %s missing in HloModuleProto",
                        constant_proto.symbol_name()));
  }
  ABSL_ASSIGN_OR_RETURN(Literal literal,
                        Literal::CreateFromProto(it->second->literal()));
  ABSL_ASSIGN_OR_RETURN(DenseDataIntermediate dense,
                        LiteralToXlaFormat(literal));
  absl::Span<const uint8_t> span = dense.span();
  return std::string(reinterpret_cast<const char*>(span.data()), span.size());
}

// Appends the distinct `buffer_allocation_index` values referenced by `thunk`
// (in encounter order) to `ordered_indices`.
absl::Status CollectReferencedAllocations(
    ThunkProto* thunk, std::vector<int64_t>& ordered_indices) {
  return ForEachBufferSlice(thunk, [&](auto* slice) {
    int64_t idx = slice->buffer_allocation_index();
    if (std::find(ordered_indices.begin(), ordered_indices.end(), idx) ==
        ordered_indices.end()) {
      ordered_indices.push_back(idx);
    }
  });
}

//===----------------------------------------------------------------------===//
// MLIR -> Proto Helpers
//===----------------------------------------------------------------------===//

// Pairs a `BufferAllocationSliceProto` with an `xla::Shape` into a
// `ShapedSliceProto`.
ShapedSliceProto MakeShapedSliceProto(const BufferAllocationSliceProto& slice,
                                      const xla::Shape& shape) {
  ShapedSliceProto proto;
  *proto.mutable_slice() = slice;
  *proto.mutable_shape() = shape.ToProto();
  return proto;
}

// Constructs a `CollectiveBufferProto` for a single source/destination buffer
// pair of element type `elem_type`, computing element counts from byte sizes.
CollectiveBufferProto MakeCollectiveBufferProto(
    const BufferAllocationSliceProto& src_slice,
    const BufferAllocationSliceProto& dst_slice, PrimitiveType elem_type) {
  int64_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(elem_type);
  int64_t src_elems = src_slice.size() / elem_size;
  int64_t dst_elems = dst_slice.size() / elem_size;
  Shape src_shape = ShapeUtil::MakeShape(elem_type, {src_elems});
  Shape dst_shape = ShapeUtil::MakeShape(elem_type, {dst_elems});
  CollectiveBufferProto proto;
  proto.set_element_count(src_elems);
  *proto.mutable_source_buffer() = MakeShapedSliceProto(src_slice, src_shape);
  *proto.mutable_destination_buffer() =
      MakeShapedSliceProto(dst_slice, dst_shape);
  proto.set_source_memory_space(0);
  proto.set_destination_memory_space(0);
  return proto;
}

// Rewrites the 0-based operand allocation indices inside a `ThunkProto` (from a
// `thunky.call_thunk_proto` op) to the virtual `BufferAllocationSliceProto`s of
// the `thunky` module, adding the operand slice's base offset.
absl::Status RebaseBufferSliceProtos(
    ThunkProto* thunk,
    absl::Span<const BufferAllocationSliceProto> operand_slices) {
  return ForEachBufferSlice(thunk, [&](auto* slice) {
    int64_t idx = slice->buffer_allocation_index();
    if (idx >= 0 && idx < static_cast<int64_t>(operand_slices.size())) {
      const BufferAllocationSliceProto& base = operand_slices[idx];
      slice->set_buffer_allocation_index(base.buffer_allocation_index());
      slice->set_offset(base.offset() + slice->offset());
    }
  });
}

// Recursively lowers the `thunky` operations in `block` into `out_thunks`,
// looking up buffer slices in `value_to_slice`.
absl::Status LowerBlockToThunkProtos(
    mlir::Block& block,
    const llvm::DenseMap<mlir::Value, BufferAllocationSliceProto>&
        value_to_slice,
    llvm::DenseMap<mlir::Value, uint64_t>& token_to_async_id,
    int64_t& next_thunk_id,
    tsl::protobuf::RepeatedPtrField<ThunkProto>* out_thunks) {
  for (mlir::Operation& op_ref : block) {
    mlir::Operation* op = &op_ref;
    if (mlir::isa<mlir::thunky::SliceBufferOp, mlir::thunky::ConstantOp>(op)) {
      continue;
    } else if (auto memzero_op = mlir::dyn_cast<mlir::thunky::MemzeroOp>(op)) {
      const BufferAllocationSliceProto& slice =
          value_to_slice.at(memzero_op.getBuffer());
      Shape shape = ShapeUtil::MakeShape(U8, {slice.size()});
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      *tp.mutable_memzero_thunk()->mutable_dest_buffer() =
          MakeShapedSliceProto(slice, shape);
    } else if (auto memset_op = mlir::dyn_cast<mlir::thunky::Memset32Op>(op)) {
      const BufferAllocationSliceProto& slice =
          value_to_slice.at(memset_op.getBuffer());
      uint32_t val = static_cast<uint32_t>(memset_op.getValueAttr().getInt());
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      *tp.mutable_memset32bit_value_thunk()->mutable_dest_buffer() = slice;
      tp.mutable_memset32bit_value_thunk()->set_value(val);
    } else if (auto copy_op = mlir::dyn_cast<mlir::thunky::CopyOp>(op)) {
      const BufferAllocationSliceProto& src_slice =
          value_to_slice.at(copy_op.getSrc());
      const BufferAllocationSliceProto& dst_slice =
          value_to_slice.at(copy_op.getDst());
      if (src_slice.buffer_allocation_index() ==
              dst_slice.buffer_allocation_index() &&
          src_slice.offset() == dst_slice.offset() &&
          src_slice.size() == dst_slice.size()) {
        continue;
      }
      Shape shape = ShapeUtil::MakeShape(U8, {src_slice.size()});
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* ct = tp.mutable_device_to_device_copy_thunk()->mutable_copy_thunk();
      *ct->mutable_source_buffer() = MakeShapedSliceProto(src_slice, shape);
      *ct->mutable_destination_buffer() =
          MakeShapedSliceProto(dst_slice, shape);
      ct->set_mem_size(src_slice.size());
    } else if (auto ptx_op = mlir::dyn_cast<mlir::thunky::PtxKernelOp>(op)) {
      std::string kernel_name(ptx_op.getKernelName());
      std::string ptx(ptx_op.getPtx());
      llvm::ArrayRef<int64_t> grid = ptx_op.getGridDim();
      llvm::ArrayRef<int64_t> block_dim = ptx_op.getBlockDim();
      int64_t shmem_bytes = ptx_op.getShmemBytesAttr().getInt();

      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* ckt = tp.mutable_custom_kernel_thunk();
      auto args = ptx_op.getBuffers();
      llvm::ArrayRef<bool> written_flags = ptx_op.getWritten();
      for (size_t i = 0; i < args.size(); ++i) {
        const BufferAllocationSliceProto& slice = value_to_slice.at(args[i]);
        Shape shape = ShapeUtil::MakeShape(U8, {slice.size()});
        *ckt->add_args() = MakeShapedSliceProto(slice, shape);
        ckt->add_written(written_flags[i]);
      }
      ckt->set_use_pdl(false);
      auto* ck = ckt->mutable_custom_kernel();
      ck->set_name(kernel_name);
      *ck->mutable_block_dims() =
          se::BlockDim(grid[0], grid[1], grid[2]).ToProto();
      *ck->mutable_thread_dims() =
          se::ThreadDim(block_dim[0], block_dim[1], block_dim[2]).ToProto();
      ck->set_shared_memory_bytes(shmem_bytes);
      auto* spec = ck->mutable_kernel_spec();
      spec->mutable_ptx()->set_data(std::move(ptx));
      spec->set_arity(args.size());
      spec->set_kernel_name(kernel_name);
    } else if (auto kernel_op =
                   mlir::dyn_cast<mlir::thunky::CallThunkProtoOp>(op)) {
      ThunkProto thunk_proto;
      if (!thunk_proto.ParseFromString(
              std::string(kernel_op.getThunkProto()))) {
        return absl::InvalidArgumentError(
            "Failed to parse ThunkProto in thunky.call_thunk_proto");
      }

      auto buffers = kernel_op.getBuffers();
      std::vector<BufferAllocationSliceProto> operand_slices;
      operand_slices.reserve(buffers.size());
      for (mlir::Value buf_val : buffers) {
        operand_slices.push_back(value_to_slice.at(buf_val));
      }

      // `KernelThunkProto` in XLA:GPU references PTX/cubin stored at the
      // module level (`GpuExecutableProto::asm_text` / `binary`). Convert it to
      // a self-contained `CustomKernelThunkProto` carrying its own `KernelSpec`
      // so `DeserializeThunkSequenceProto` can instantiate it without a module
      // binary table.
      if (thunk_proto.has_kernel_thunk()) {
        KernelThunkProto kt = thunk_proto.kernel_thunk();
        auto* ckt = thunk_proto.mutable_custom_kernel_thunk();
        for (int i = 0; i < kt.args_size(); ++i) {
          auto* arg = ckt->add_args();
          *arg->mutable_slice() = kt.args(i);
          if (i < kt.args_shape_size()) {
            *arg->mutable_shape() = kt.args_shape(i);
          } else {
            *arg->mutable_shape() =
                ShapeUtil::MakeShape(U8, {kt.args(i).size()}).ToProto();
          }
        }
        for (bool w : kt.written()) {
          ckt->add_written(w);
        }
        ckt->set_use_pdl(kt.use_pdl());
        for (int64_t z : kt.zeroed_output_buffer_indices()) {
          ckt->add_zeroed_output_buffer_indices(z);
        }
        if (kt.has_tma_metadata()) {
          *ckt->mutable_tma_metadata() = kt.tma_metadata();
        }
        auto* ck = ckt->mutable_custom_kernel();
        ck->set_name(kt.kernel_name());
        *ck->mutable_block_dims() = kt.launch_dimensions().block_counts();
        *ck->mutable_thread_dims() =
            kt.launch_dimensions().thread_counts_per_block();
        if (kt.has_cluster_dim()) {
          *ck->mutable_cluster_dim() = kt.cluster_dim();
        }
        ck->set_shared_memory_bytes(kt.shmem_bytes());
        auto* spec = ck->mutable_kernel_spec();
        spec->set_arity(kt.args_size());
        spec->set_kernel_name(kt.kernel_name());
      }
      if (thunk_proto.has_custom_kernel_thunk()) {
        auto* spec = thunk_proto.mutable_custom_kernel_thunk()
                         ->mutable_custom_kernel()
                         ->mutable_kernel_spec();
        if (!spec->has_cubin() && !spec->has_ptx()) {
          if (!kernel_op.getBinary().empty()) {
            spec->mutable_cubin()->set_data(std::string(kernel_op.getBinary()));
          } else if (!kernel_op.getAsmText().empty()) {
            spec->mutable_ptx()->set_data(std::string(kernel_op.getAsmText()));
          }
        }
      }
      if (thunk_proto.has_cudnn_thunk()) {
        return absl::UnimplementedError(
            "CuDnnThunk lowering is not supported in thunky");
      }

      ABSL_RETURN_IF_ERROR(
          RebaseBufferSliceProtos(&thunk_proto, operand_slices));

      // Flatten any nested `SequentialThunkProto` and drop device-ID thunks
      // (`PartitionIdThunkProto` / `ReplicaIdThunkProto`) that XLA's
      // `DeserializeThunkSequenceProto` does not need when spliced.
      std::function<void(ThunkProto)> emit_leaf_proto = [&](ThunkProto t) {
        if (t.has_sequential_thunk()) {
          for (auto& child : *t.mutable_sequential_thunk()->mutable_thunks()) {
            emit_leaf_proto(std::move(child));
          }
          return;
        }
        if (t.has_partition_id_thunk() || t.has_replica_id_thunk()) {
          return;
        }
        t.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
        *out_thunks->Add() = std::move(t);
      };
      emit_leaf_proto(std::move(thunk_proto));
    } else if (auto custom_call_op =
                   mlir::dyn_cast<mlir::thunky::CustomCallOp>(op)) {
      int64_t num_cc_operands = custom_call_op.getNumOperandsAttr().getInt();
      int64_t num_cc_results = custom_call_op.getNumResultsAttr().getInt();
      auto buffers = custom_call_op.getBuffers();
      auto operand_shapes_attr = custom_call_op.getOperandShapesAttr();
      auto result_shapes_attr = custom_call_op.getResultShapesAttr();

      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* cct = tp.mutable_custom_call_thunk();
      cct->set_target_name(std::string(custom_call_op.getTargetName()));
      cct->set_api_version(CustomCallApiVersion::API_VERSION_TYPED_FFI);
      cct->set_use_pdl(false);

      for (int64_t i = 0; i < num_cc_operands; ++i) {
        const BufferAllocationSliceProto& slice = value_to_slice.at(buffers[i]);
        mlir::Attribute shape_attr = nullptr;
        if (operand_shapes_attr &&
            static_cast<size_t>(i) < operand_shapes_attr.size()) {
          shape_attr = operand_shapes_attr[i];
        }
        ABSL_ASSIGN_OR_RETURN(xla::Shape shape,
                              TypeAttrToShape(shape_attr, slice.size()));
        *cct->add_operands()->mutable_shaped_slice() =
            MakeShapedSliceProto(slice, shape);
      }
      for (int64_t i = 0; i < num_cc_results; ++i) {
        const BufferAllocationSliceProto& slice =
            value_to_slice.at(buffers[num_cc_operands + i]);
        mlir::Attribute shape_attr = nullptr;
        if (result_shapes_attr &&
            static_cast<size_t>(i) < result_shapes_attr.size()) {
          shape_attr = result_shapes_attr[i];
        }
        ABSL_ASSIGN_OR_RETURN(xla::Shape shape,
                              TypeAttrToShape(shape_attr, slice.size()));
        *cct->add_results()->mutable_shaped_slice() =
            MakeShapedSliceProto(slice, shape);
      }
      ABSL_ASSIGN_OR_RETURN(
          xla::ffi::AttributesMap cc_attributes,
          xla::ffi::BuildAttributesMap(custom_call_op.getBackendConfig()));
      *cct->mutable_attributes() = cc_attributes.ToProto();
    } else if (auto cond_op = mlir::dyn_cast<mlir::thunky::CondOp>(op)) {
      const BufferAllocationSliceProto& pred_slice =
          value_to_slice.at(cond_op.getBranchIndex());
      Shape pred_shape = pred_slice.size() == 1 ? ShapeUtil::MakeShape(PRED, {})
                                                : ShapeUtil::MakeShape(S32, {});
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* ct = tp.mutable_conditional_thunk();
      *ct->mutable_branch_index_buffer() =
          MakeShapedSliceProto(pred_slice, pred_shape);
      for (mlir::Region& region : cond_op.getBranches()) {
        ABSL_RETURN_IF_ERROR(LowerBlockToThunkProtos(
            region.front(), value_to_slice, token_to_async_id, next_thunk_id,
            ct->add_branch_thunks()->mutable_thunks()));
      }
    } else if (auto while_op = mlir::dyn_cast<mlir::thunky::WhileOp>(op)) {
      const BufferAllocationSliceProto& cond_slice =
          value_to_slice.at(while_op.getConditionBuffer());
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* wt = tp.mutable_while_thunk();
      *wt->mutable_condition_result_buffer_index() = cond_slice;
      ABSL_RETURN_IF_ERROR(LowerBlockToThunkProtos(
          while_op.getCondRegion().front(), value_to_slice, token_to_async_id,
          next_thunk_id,
          wt->mutable_condition_thunk_sequence()->mutable_thunks()));
      ABSL_RETURN_IF_ERROR(LowerBlockToThunkProtos(
          while_op.getBodyRegion().front(), value_to_slice, token_to_async_id,
          next_thunk_id, wt->mutable_body_thunk_sequence()->mutable_thunks()));
    } else if (auto async_start_op =
                   mlir::dyn_cast<mlir::thunky::AsyncStartOp>(op)) {
      uint64_t stream_id = async_start_op.getStreamId();
      bool is_comm = async_start_op.getIsCommunication();
      ThunkProto& tp = *out_thunks->Add();
      uint64_t async_id = static_cast<uint64_t>(next_thunk_id++);
      tp.mutable_thunk_info()->set_thunk_id(async_id);
      token_to_async_id[async_start_op.getToken()] = async_id;
      auto* ast = tp.mutable_async_start_thunk();
      ast->set_async_execution_id(async_id);
      if (is_comm) {
        ast->set_communication_stream_id(stream_id);
      } else {
        ast->set_computation_stream_id(stream_id);
      }
      ABSL_RETURN_IF_ERROR(LowerBlockToThunkProtos(
          async_start_op.getBodyRegion().front(), value_to_slice,
          token_to_async_id, next_thunk_id,
          ast->mutable_thunks()->mutable_thunks()));
    } else if (auto async_done_op =
                   mlir::dyn_cast<mlir::thunky::AsyncDoneOp>(op)) {
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      tp.mutable_async_done_thunk()->set_async_execution_id(
          token_to_async_id.at(async_done_op.getToken()));
    } else if (auto all_reduce_op =
                   mlir::dyn_cast<mlir::thunky::AllReduceOp>(op)) {
      ABSL_ASSIGN_OR_RETURN(
          CollectiveConfigProto config,
          MakeCollectiveConfigProto(all_reduce_op.getElementType(),
                                    all_reduce_op.getReplicaGroups(),
                                    all_reduce_op.getGroupMode()));
      ABSL_ASSIGN_OR_RETURN(
          xla::ReductionKindProto reduction_kind,
          StringToReductionKind(all_reduce_op.getReductionKind()));
      PrimitiveType elem_type =
          static_cast<PrimitiveType>(config.operand_element_type(0));
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* art = tp.mutable_all_reduce_thunk();
      *art->mutable_collective_config() = std::move(config);
      art->set_reduction_kind(reduction_kind);
      *art->add_buffers() = MakeCollectiveBufferProto(
          value_to_slice.at(all_reduce_op.getSrc()),
          value_to_slice.at(all_reduce_op.getDst()), elem_type);
    } else if (auto all_gather_op =
                   mlir::dyn_cast<mlir::thunky::AllGatherOp>(op)) {
      ABSL_ASSIGN_OR_RETURN(
          CollectiveConfigProto config,
          MakeCollectiveConfigProto(all_gather_op.getElementType(),
                                    all_gather_op.getReplicaGroups(),
                                    all_gather_op.getGroupMode()));
      PrimitiveType elem_type =
          static_cast<PrimitiveType>(config.operand_element_type(0));
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* agt = tp.mutable_all_gather_thunk();
      *agt->mutable_collective_config() = std::move(config);
      agt->set_collectives_mode(DebugOptions::COLLECTIVES_PRIVATE_MEMORY);
      *agt->add_buffers() = MakeCollectiveBufferProto(
          value_to_slice.at(all_gather_op.getSrc()),
          value_to_slice.at(all_gather_op.getDst()), elem_type);
    } else if (auto reduce_scatter_op =
                   mlir::dyn_cast<mlir::thunky::ReduceScatterOp>(op)) {
      ABSL_ASSIGN_OR_RETURN(
          CollectiveConfigProto config,
          MakeCollectiveConfigProto(reduce_scatter_op.getElementType(),
                                    reduce_scatter_op.getReplicaGroups(),
                                    reduce_scatter_op.getGroupMode()));
      ABSL_ASSIGN_OR_RETURN(
          xla::ReductionKindProto reduction_kind,
          StringToReductionKind(reduce_scatter_op.getReductionKind()));
      PrimitiveType elem_type =
          static_cast<PrimitiveType>(config.operand_element_type(0));
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* rst = tp.mutable_reduce_scatter_thunk();
      *rst->mutable_collective_config() = std::move(config);
      rst->set_reduction_kind(reduction_kind);
      *rst->add_buffers() = MakeCollectiveBufferProto(
          value_to_slice.at(reduce_scatter_op.getSrc()),
          value_to_slice.at(reduce_scatter_op.getDst()), elem_type);
    } else if (auto all_to_all_op =
                   mlir::dyn_cast<mlir::thunky::AllToAllOp>(op)) {
      ABSL_ASSIGN_OR_RETURN(
          CollectiveConfigProto config,
          MakeCollectiveConfigProto(all_to_all_op.getElementType(),
                                    all_to_all_op.getReplicaGroups(),
                                    all_to_all_op.getGroupMode()));
      auto src_vals = all_to_all_op.getSrc();
      auto dst_vals = all_to_all_op.getDst();
      if (src_vals.size() != dst_vals.size() || src_vals.empty()) {
        return absl::InvalidArgumentError(
            "all_to_all requires equal non-empty src and dst buffer counts");
      }
      int64_t num_ranks = config.replica_groups().empty()
                              ? 0
                              : config.replica_groups(0).replica_ids_size();
      if (!all_to_all_op.getHasSplitDimension() &&
          static_cast<int64_t>(src_vals.size()) != num_ranks) {
        return absl::InvalidArgumentError(absl::StrCat(
            "all_to_all with has_split_dimension=false requires ", num_ranks,
            " buffer pairs (one per peer rank), got ", src_vals.size()));
      }
      if (all_to_all_op.getHasSplitDimension() && src_vals.size() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("all_to_all with has_split_dimension=true requires 1 "
                         "buffer pair, got ",
                         src_vals.size()));
      }
      PrimitiveType elem_type =
          static_cast<PrimitiveType>(config.operand_element_type(0));
      config.clear_operand_element_type();
      for (size_t i = 0; i < src_vals.size(); ++i) {
        config.add_operand_element_type(elem_type);
      }
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* a2at = tp.mutable_all_to_all_thunk();
      *a2at->mutable_collective_config() = std::move(config);
      a2at->set_has_split_dimension(all_to_all_op.getHasSplitDimension());
      a2at->set_p2p_memcpy_enabled(false);
      for (size_t i = 0; i < src_vals.size(); ++i) {
        *a2at->add_buffers() = MakeCollectiveBufferProto(
            value_to_slice.at(src_vals[i]), value_to_slice.at(dst_vals[i]),
            elem_type);
      }
    } else if (auto collective_permute_op =
                   mlir::dyn_cast<mlir::thunky::CollectivePermuteOp>(op)) {
      PrimitiveType elem_type = xla::ConvertMlirTypeToPrimitiveType(
          collective_permute_op.getElementType());
      if (elem_type == xla::PRIMITIVE_TYPE_INVALID) {
        return absl::InvalidArgumentError(
            "Unsupported collective_permute element type in MLIR");
      }
      ABSL_ASSIGN_OR_RETURN(
          CollectiveOpGroupMode group_mode,
          StringToGroupMode(collective_permute_op.getGroupMode()));
      P2PConfig p2p_config;
      p2p_config.config.operand_element_type = {elem_type};
      p2p_config.config.group_mode = group_mode;
      p2p_config.config.use_symmetric_buffer = false;
      std::set<int64_t> participant_ids;
      for (mlir::Attribute attr :
           collective_permute_op.getSourceTargetPairs()) {
        int64_t src_id = 0;
        int64_t dst_id = 0;
        if (auto dense_arr = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
          src_id = dense_arr.asArrayRef()[0];
          dst_id = dense_arr.asArrayRef()[1];
        } else {
          auto arr = mlir::cast<mlir::ArrayAttr>(attr);
          src_id = mlir::cast<mlir::IntegerAttr>(arr[0]).getInt();
          dst_id = mlir::cast<mlir::IntegerAttr>(arr[1]).getInt();
        }
        p2p_config.id_to_source_target[dst_id].source = src_id;
        p2p_config.id_to_source_target[src_id].target = dst_id;
        participant_ids.insert(src_id);
        participant_ids.insert(dst_id);
      }
      std::set<int64_t> rg_ids;
      if (auto rg_attr = collective_permute_op.getReplicaGroups();
          rg_attr && !rg_attr->empty()) {
        p2p_config.config.replica_groups = AttrToReplicaGroups(*rg_attr);
        for (const auto& rg : p2p_config.config.replica_groups) {
          for (int64_t id : rg.replica_ids()) {
            rg_ids.insert(id);
          }
        }
      }
      if (!std::includes(rg_ids.begin(), rg_ids.end(), participant_ids.begin(),
                         participant_ids.end())) {
        p2p_config.config.replica_groups.clear();
        ReplicaGroup& rg = p2p_config.config.replica_groups.emplace_back();
        for (int64_t id : participant_ids) {
          rg.add_replica_ids(id);
        }
      }
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* cpt = tp.mutable_collective_permute_thunk();
      *cpt->mutable_collective_config() = p2p_config.config.ToProto();
      *cpt->add_buffers() = MakeCollectiveBufferProto(
          value_to_slice.at(collective_permute_op.getSrc()),
          value_to_slice.at(collective_permute_op.getDst()), elem_type);
      std::vector<SourceTarget> source_target_pairs =
          GetSortedSourceTargetPairs(p2p_config.id_to_source_target);
      cpt->mutable_source_target_pairs()->Assign(source_target_pairs.begin(),
                                                 source_target_pairs.end());
      cpt->set_collectives_mode(DebugOptions::COLLECTIVES_PRIVATE_MEMORY);
      cpt->set_connected_components_enabled(false);
    } else if (auto group_op =
                   mlir::dyn_cast<mlir::thunky::CollectiveGroupOp>(op)) {
      ThunkProto& tp = *out_thunks->Add();
      tp.mutable_thunk_info()->set_thunk_id(next_thunk_id++);
      auto* cgt = tp.mutable_collective_group_thunk();
      cgt->set_thunk_kind(Thunk::KindToProto(Thunk::kGroup));
      ABSL_RETURN_IF_ERROR(LowerBlockToThunkProtos(
          group_op.getBodyRegion().front(), value_to_slice, token_to_async_id,
          next_thunk_id, cgt->mutable_thunks()));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<GpuExecutableProto> ParseGpuExecutableProto(
    absl::string_view serialized_bytes) {
  // 1. Strip the size-delimited IFRT `SerializedXlaExecutableMetadata` prefix.
  tsl::protobuf::io::ArrayInputStream array_stream(serialized_bytes.data(),
                                                   serialized_bytes.size());
  tsl::protobuf::io::CodedInputStream coded_stream(&array_stream);
  xla::ifrt::SerializedXlaExecutableMetadata metadata;
  if (!tsl::protobuf::util::ParseDelimitedFromCodedStream(
          &metadata, &coded_stream, nullptr)) {
    return absl::InvalidArgumentError(
        "Failed to parse SerializedXlaExecutableMetadata");
  }
  absl::string_view pjrt_payload =
      serialized_bytes.substr(coded_stream.CurrentPosition());

  // 2. Unpack the PJRT `ExecutableAndOptionsProto` split-proto.
  ExecutableAndOptionsProto exe_and_opts;
  ABSL_RETURN_IF_ERROR(ReadSplitProto(
      std::make_unique<riegeli::StringReader<>>(pjrt_payload), exe_and_opts));

  // 3. Unpack the inner XLA:GPU `GpuExecutableProto` split-proto.
  GpuExecutableProto gpu_exe_proto;
  ABSL_RETURN_IF_ERROR(ReadSplitProto(std::make_unique<riegeli::StringReader<>>(
                                          exe_and_opts.serialized_executable()),
                                      gpu_exe_proto));
  return gpu_exe_proto;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GpuExecutableProtoToThunkyModule(const GpuExecutableProto& gpu_proto,
                                 mlir::MLIRContext* context) {
  context->getOrLoadDialect<mlir::thunky::ThunkyDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(context);
  mlir::Location loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  // Partition `gpu_proto`'s `BufferAllocationProto`s into four categories:
  //   1. `param_alloc_indices`: entry computation parameters, ordered by
  //      `parameter_number`.
  //   2. `fresh_out_alloc_indices`: leaf output buffers (ordered by
  //      `ShapeIndex`) that do not alias an input parameter.
  //   3. `constant_alloc_indices`: constant buffers materialized inside `@main`
  //      via `thunky.constant` rather than passed as `@main` parameters.
  //   4. `internal_alloc_indices`: remaining temp/scratch buffer allocations,
  //      appended as trailing `@main` parameters so the caller can allocate
  //      scratch space for them.
  std::vector<std::pair<int64_t, int64_t>> param_num_and_idx;
  absl::flat_hash_map<int64_t, int64_t> alloc_sizes;
  for (const auto& alloc : gpu_proto.buffer_allocations().values()) {
    alloc_sizes[alloc.index()] = alloc.size();
    if (alloc.is_entry_computation_parameter()) {
      param_num_and_idx.push_back({alloc.parameter_number(), alloc.index()});
    }
  }
  std::sort(param_num_and_idx.begin(), param_num_and_idx.end());

  std::vector<int64_t> param_alloc_indices;
  param_alloc_indices.reserve(param_num_and_idx.size());
  for (const auto& [pnum, idx] : param_num_and_idx) {
    param_alloc_indices.push_back(idx);
  }

  std::vector<int64_t> out_alloc_indices;
  if (!gpu_proto.output_info_map().empty()) {
    ABSL_ASSIGN_OR_RETURN(Shape result_shape,
                          Shape::FromProto(gpu_proto.program_shape().result()));
    std::vector<std::pair<ShapeIndex, int64_t>> out_index_and_alloc;
    for (const auto& entry : gpu_proto.output_info_map()) {
      ShapeIndex shape_index = ShapeIndex::FromProto(entry.shape_index());
      const Shape& subshape = ShapeUtil::GetSubshape(result_shape, shape_index);
      if (!subshape.IsTuple()) {
        out_index_and_alloc.push_back(
            {shape_index, entry.output_info().allocation_index()});
      }
    }
    std::sort(out_index_and_alloc.begin(), out_index_and_alloc.end());
    out_alloc_indices.reserve(out_index_and_alloc.size());
    for (const auto& [shape_idx, alloc_idx] : out_index_and_alloc) {
      out_alloc_indices.push_back(alloc_idx);
    }
  }

  std::set<int64_t> param_alloc_set(param_alloc_indices.begin(),
                                    param_alloc_indices.end());
  std::vector<int64_t> fresh_out_alloc_indices;
  for (int64_t idx : out_alloc_indices) {
    if (param_alloc_set.find(idx) == param_alloc_set.end()) {
      fresh_out_alloc_indices.push_back(idx);
    }
  }

  std::set<int64_t> constant_alloc_indices;
  for (const auto& c : gpu_proto.constants()) {
    if (c.allocation_index() != -1) {
      constant_alloc_indices.insert(c.allocation_index());
    }
  }

  for (const auto& entry : gpu_proto.output_info_map()) {
    if (entry.output_info().has_alias_config()) {
      int64_t pnum = entry.output_info().alias_config().parameter_number();
      int64_t alloc_idx = entry.output_info().allocation_index();
      CHECK_EQ(alloc_idx, param_alloc_indices[pnum]);
    }
  }

  std::set<int64_t> param_or_out_indices(param_alloc_indices.begin(),
                                         param_alloc_indices.end());
  for (int64_t idx : out_alloc_indices) {
    param_or_out_indices.insert(idx);
  }

  std::vector<int64_t> internal_alloc_indices;
  for (const auto& alloc : gpu_proto.buffer_allocations().values()) {
    if (param_or_out_indices.find(alloc.index()) ==
            param_or_out_indices.end() &&
        constant_alloc_indices.find(alloc.index()) ==
            constant_alloc_indices.end()) {
      internal_alloc_indices.push_back(alloc.index());
    }
  }
  std::sort(internal_alloc_indices.begin(), internal_alloc_indices.end());

  llvm::SmallVector<mlir::Type> arg_types;
  for (int64_t idx : param_alloc_indices) {
    arg_types.push_back(
        mlir::thunky::BufferType::get(context, alloc_sizes.at(idx)));
  }
  for (int64_t idx : fresh_out_alloc_indices) {
    arg_types.push_back(
        mlir::thunky::BufferType::get(context, alloc_sizes.at(idx)));
  }
  for (int64_t idx : internal_alloc_indices) {
    arg_types.push_back(
        mlir::thunky::BufferType::get(context, alloc_sizes.at(idx)));
  }

  auto func_type = builder.getFunctionType(arg_types, {});
  auto func = mlir::func::FuncOp::create(builder, loc, "main", func_type);
  mlir::Block* entry_block = func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  absl::flat_hash_map<int64_t, mlir::Value> alloc_to_val;
  for (size_t i = 0; i < param_alloc_indices.size(); ++i) {
    alloc_to_val[param_alloc_indices[i]] = entry_block->getArgument(i);
  }
  for (size_t i = 0; i < fresh_out_alloc_indices.size(); ++i) {
    alloc_to_val[fresh_out_alloc_indices[i]] =
        entry_block->getArgument(param_alloc_indices.size() + i);
  }
  size_t internal_arg_offset =
      param_alloc_indices.size() + fresh_out_alloc_indices.size();
  for (size_t i = 0; i < internal_alloc_indices.size(); ++i) {
    alloc_to_val[internal_alloc_indices[i]] =
        entry_block->getArgument(internal_arg_offset + i);
  }

  // Emit `thunky.constant` ops at the start of `@main` for each constant
  // allocation. If a constant allocation is also an output buffer of the
  // executable, copy the constant into the output buffer argument first.
  if (!gpu_proto.constants().empty()) {
    auto symbol_to_instr = MakeConstantInstructionProtoMap(gpu_proto);
    for (const auto& c : gpu_proto.constants()) {
      if (c.allocation_index() == -1) {
        continue;
      }
      int64_t idx = c.allocation_index();
      ABSL_ASSIGN_OR_RETURN(std::string const_bytes,
                            ExtractConstantBytes(c, symbol_to_instr));
      int64_t base_size = alloc_sizes.at(idx);
      if (static_cast<int64_t>(const_bytes.size()) < base_size) {
        const_bytes.resize(base_size, '\0');
      }
      auto const_op = mlir::thunky::ConstantOp::create(
          builder, loc, mlir::thunky::BufferType::get(context, base_size),
          builder.getStringAttr(const_bytes));
      if (alloc_to_val.contains(idx)) {
        mlir::thunky::CopyOp::create(builder, loc, const_op.getResult(),
                                     alloc_to_val.at(idx));
      }
      alloc_to_val[idx] = const_op.getResult();
    }
  }

  // Returns the MLIR `!thunky.buffer` value for `slice`, emitting a
  // `thunky.slice_buffer` op when `slice` is a sub-range of its parent
  // allocation.
  auto get_slice_val =
      [&](const BufferAllocationSliceProto& slice) -> mlir::Value {
    int64_t idx = slice.buffer_allocation_index();
    mlir::Value base = alloc_to_val.at(idx);
    int64_t base_size = alloc_sizes.at(idx);
    if (slice.offset() == 0 && slice.size() == base_size) {
      return base;
    }
    return mlir::thunky::SliceBufferOp::create(
               builder, loc,
               mlir::thunky::BufferType::get(context, slice.size()), base,
               builder.getI64IntegerAttr(slice.offset()))
        .getResult();
  };

  // Translates each `ThunkProto` into a first-class `thunky` dialect operation
  // when supported, or wraps opaque backend thunks (`KernelThunkProto`,
  // `CustomKernelThunkProto`, `GemmThunkProto`, `CuDnnThunkProto`, etc.) in a
  // `thunky.call_thunk_proto` op with 0-based local buffer indices.
  absl::flat_hash_map<uint64_t, mlir::Value> async_exec_id_to_token;
  std::function<absl::Status(const ThunkProto&)> emit_thunk =
      [&](const ThunkProto& thunk) -> absl::Status {
    if (thunk.has_sequential_thunk()) {
      for (const auto& child : thunk.sequential_thunk().thunks()) {
        ABSL_RETURN_IF_ERROR(emit_thunk(child));
      }
    } else if (thunk.has_async_start_thunk()) {
      const auto& ast = thunk.async_start_thunk();
      bool is_comm = ast.has_communication_stream_id();
      uint64_t stream_id =
          is_comm ? ast.communication_stream_id() : ast.computation_stream_id();
      auto async_start_op = mlir::thunky::AsyncStartOp::create(
          builder, loc, mlir::thunky::TokenType::get(context),
          builder.getI64IntegerAttr(stream_id), builder.getBoolAttr(is_comm));
      async_exec_id_to_token[ast.async_execution_id()] =
          async_start_op.getToken();
      mlir::Block* body_block = &async_start_op.getBodyRegion().emplaceBlock();
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(body_block);
      for (const auto& child : ast.thunks().thunks()) {
        ABSL_RETURN_IF_ERROR(emit_thunk(child));
      }
      mlir::thunky::YieldOp::create(builder, loc);
    } else if (thunk.has_async_done_thunk()) {
      const auto& adt = thunk.async_done_thunk();
      mlir::thunky::AsyncDoneOp::create(
          builder, loc, async_exec_id_to_token.at(adt.async_execution_id()));
    } else if (thunk.has_conditional_thunk()) {
      const auto& ct = thunk.conditional_thunk();
      mlir::Value branch_val = get_slice_val(ct.branch_index_buffer().slice());
      auto cond_op = mlir::thunky::CondOp::create(builder, loc, branch_val,
                                                  ct.branch_thunks_size());
      for (int i = 0; i < ct.branch_thunks_size(); ++i) {
        mlir::Block* branch_block = &cond_op.getBranches()[i].emplaceBlock();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(branch_block);
        for (const auto& child : ct.branch_thunks(i).thunks()) {
          ABSL_RETURN_IF_ERROR(emit_thunk(child));
        }
        mlir::thunky::YieldOp::create(builder, loc);
      }
    } else if (thunk.has_while_thunk()) {
      const auto& wt = thunk.while_thunk();
      mlir::Value cond_val = get_slice_val(wt.condition_result_buffer_index());
      auto while_op = mlir::thunky::WhileOp::create(builder, loc, cond_val);
      {
        mlir::Block* cond_block = &while_op.getCondRegion().emplaceBlock();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(cond_block);
        for (const auto& child : wt.condition_thunk_sequence().thunks()) {
          ABSL_RETURN_IF_ERROR(emit_thunk(child));
        }
        mlir::thunky::YieldOp::create(builder, loc);
      }
      {
        mlir::Block* body_block = &while_op.getBodyRegion().emplaceBlock();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(body_block);
        for (const auto& child : wt.body_thunk_sequence().thunks()) {
          ABSL_RETURN_IF_ERROR(emit_thunk(child));
        }
        mlir::thunky::YieldOp::create(builder, loc);
      }
    } else if (thunk.has_collective_group_thunk()) {
      const auto& cgt = thunk.collective_group_thunk();
      auto group_op = mlir::thunky::CollectiveGroupOp::create(builder, loc);
      mlir::Block* body_block = &group_op.getBodyRegion().emplaceBlock();
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(body_block);
      for (const auto& child : cgt.thunks()) {
        ABSL_RETURN_IF_ERROR(emit_thunk(child));
      }
      mlir::thunky::YieldOp::create(builder, loc);
    } else if (thunk.has_all_reduce_thunk()) {
      const auto& art = thunk.all_reduce_thunk();
      for (const auto& buf : art.buffers()) {
        mlir::thunky::AllReduceOp::create(
            builder, loc, get_slice_val(buf.source_buffer().slice()),
            get_slice_val(buf.destination_buffer().slice()),
            builder.getStringAttr(ReductionKindToString(art.reduction_kind())),
            ElemTypeToTypeAttr(builder, art.collective_config()),
            ReplicaGroupsToAttr(builder, art.collective_config()),
            builder.getStringAttr(
                GroupModeToString(art.collective_config().group_mode())));
      }
    } else if (thunk.has_all_gather_thunk()) {
      const auto& agt = thunk.all_gather_thunk();
      for (const auto& buf : agt.buffers()) {
        mlir::thunky::AllGatherOp::create(
            builder, loc, get_slice_val(buf.source_buffer().slice()),
            get_slice_val(buf.destination_buffer().slice()),
            ElemTypeToTypeAttr(builder, agt.collective_config()),
            ReplicaGroupsToAttr(builder, agt.collective_config()),
            builder.getStringAttr(
                GroupModeToString(agt.collective_config().group_mode())));
      }
    } else if (thunk.has_reduce_scatter_thunk()) {
      const auto& rst = thunk.reduce_scatter_thunk();
      for (const auto& buf : rst.buffers()) {
        mlir::thunky::ReduceScatterOp::create(
            builder, loc, get_slice_val(buf.source_buffer().slice()),
            get_slice_val(buf.destination_buffer().slice()),
            builder.getStringAttr(ReductionKindToString(rst.reduction_kind())),
            ElemTypeToTypeAttr(builder, rst.collective_config()),
            ReplicaGroupsToAttr(builder, rst.collective_config()),
            builder.getStringAttr(
                GroupModeToString(rst.collective_config().group_mode())));
      }
    } else if (thunk.has_all_to_all_thunk()) {
      const auto& a2at = thunk.all_to_all_thunk();
      llvm::SmallVector<mlir::Value> src_vals;
      llvm::SmallVector<mlir::Value> dst_vals;
      for (const auto& buf : a2at.buffers()) {
        src_vals.push_back(get_slice_val(buf.source_buffer().slice()));
        dst_vals.push_back(get_slice_val(buf.destination_buffer().slice()));
      }
      mlir::thunky::AllToAllOp::create(
          builder, loc, src_vals, dst_vals,
          ElemTypeToTypeAttr(builder, a2at.collective_config()),
          ReplicaGroupsToAttr(builder, a2at.collective_config()),
          builder.getStringAttr(
              GroupModeToString(a2at.collective_config().group_mode())),
          builder.getBoolAttr(a2at.has_split_dimension()));
    } else if (thunk.has_collective_permute_thunk()) {
      const auto& cpt = thunk.collective_permute_thunk();
      llvm::SmallVector<mlir::Attribute> pair_attrs;
      for (const auto& pair : cpt.source_target_pairs()) {
        llvm::SmallVector<int64_t> p = {pair.source(), pair.target()};
        pair_attrs.push_back(builder.getDenseI64ArrayAttr(p));
      }
      for (const auto& buf : cpt.buffers()) {
        mlir::thunky::CollectivePermuteOp::create(
            builder, loc, get_slice_val(buf.source_buffer().slice()),
            get_slice_val(buf.destination_buffer().slice()),
            ElemTypeToTypeAttr(builder, cpt.collective_config()),
            builder.getArrayAttr(pair_attrs),
            builder.getStringAttr(
                GroupModeToString(cpt.collective_config().group_mode())),
            ReplicaGroupsToAttr(builder, cpt.collective_config()));
      }
    } else if (thunk.has_memzero_thunk()) {
      mlir::thunky::MemzeroOp::create(
          builder, loc,
          get_slice_val(thunk.memzero_thunk().dest_buffer().slice()));
    } else if (thunk.has_memset32bit_value_thunk()) {
      uint32_t val = thunk.memset32bit_value_thunk().value();
      mlir::thunky::Memset32Op::create(
          builder, loc,
          get_slice_val(thunk.memset32bit_value_thunk().dest_buffer()),
          builder.getI32IntegerAttr(static_cast<int32_t>(val)));
    } else if (thunk.has_device_to_device_copy_thunk()) {
      mlir::thunky::CopyOp::create(
          builder, loc,
          get_slice_val(thunk.device_to_device_copy_thunk()
                            .copy_thunk()
                            .source_buffer()
                            .slice()),
          get_slice_val(thunk.device_to_device_copy_thunk()
                            .copy_thunk()
                            .destination_buffer()
                            .slice()));
    } else if ([&]() -> bool {
                 if (!thunk.has_custom_call_thunk()) return false;
                 const auto& cc = thunk.custom_call_thunk();
                 if (cc.api_version() !=
                     CustomCallApiVersion::API_VERSION_TYPED_FFI) {
                   return false;
                 }
                 if (cc.has_called_computation() ||
                     (cc.has_execution_state() &&
                      !cc.execution_state().type_name().empty())) {
                   return false;
                 }
                 for (const auto& op : cc.operands()) {
                   if (!op.has_shaped_slice() ||
                       !alloc_to_val.contains(op.shaped_slice()
                                                  .slice()
                                                  .buffer_allocation_index())) {
                     return false;
                   }
                 }
                 for (const auto& res : cc.results()) {
                   if (!res.has_shaped_slice() ||
                       !alloc_to_val.contains(res.shaped_slice()
                                                  .slice()
                                                  .buffer_allocation_index())) {
                     return false;
                   }
                 }
                 return true;
               }()) {
      const auto& cc = thunk.custom_call_thunk();
      llvm::SmallVector<mlir::Value> buffers;
      llvm::SmallVector<mlir::Attribute> op_shapes;
      llvm::SmallVector<mlir::Attribute> res_shapes;
      for (const auto& op : cc.operands()) {
        buffers.push_back(get_slice_val(op.shaped_slice().slice()));
        op_shapes.push_back(
            ShapeProtoToTypeAttr(builder, op.shaped_slice().shape()));
      }
      for (const auto& res : cc.results()) {
        buffers.push_back(get_slice_val(res.shaped_slice().slice()));
        res_shapes.push_back(
            ShapeProtoToTypeAttr(builder, res.shaped_slice().shape()));
      }
      mlir::thunky::CustomCallOp::create(
          builder, loc, buffers, builder.getI64IntegerAttr(cc.operands_size()),
          builder.getI64IntegerAttr(cc.results_size()),
          builder.getStringAttr(cc.target_name()),
          ConvertAttributesMapProtoToMlirDict(builder, cc.attributes()),
          builder.getArrayAttr(op_shapes), builder.getArrayAttr(res_shapes));
    } else {
      ThunkProto local_thunk = thunk;
      std::vector<int64_t> ref_indices;
      ABSL_RETURN_IF_ERROR(
          CollectReferencedAllocations(&local_thunk, ref_indices));
      absl::flat_hash_map<int64_t, int64_t> local_mapping;
      llvm::SmallVector<mlir::Value> operands;
      for (size_t j = 0; j < ref_indices.size(); ++j) {
        local_mapping[ref_indices[j]] = static_cast<int64_t>(j);
        operands.push_back(alloc_to_val.at(ref_indices[j]));
      }
      ABSL_RETURN_IF_ERROR(RemapBufferAllocations(&local_thunk, local_mapping));

      std::string asm_text_str = gpu_proto.asm_text();
      std::string binary_str = gpu_proto.binary();
      if (local_thunk.has_custom_kernel_thunk()) {
        const auto& ck = local_thunk.custom_kernel_thunk().custom_kernel();
        if (binary_str.empty() && ck.kernel_spec().has_cubin()) {
          binary_str = ck.kernel_spec().cubin().data();
        }
        if (asm_text_str.empty() && ck.kernel_spec().has_ptx()) {
          asm_text_str = ck.kernel_spec().ptx().data();
        }
      }
      std::string thunk_bytes = local_thunk.SerializeAsString();

      std::string kernel_name =
          local_thunk.has_kernel_thunk()
              ? local_thunk.kernel_thunk().kernel_name()
              : (local_thunk.has_custom_kernel_thunk()
                     ? local_thunk.custom_kernel_thunk().custom_kernel().name()
                     : "thunk");
      mlir::thunky::CallThunkProtoOp::create(
          builder, loc, operands, builder.getStringAttr(kernel_name),
          builder.getStringAttr(thunk_bytes),
          builder.getStringAttr(asm_text_str),
          builder.getStringAttr(binary_str));
    }
    return absl::OkStatus();
  };

  for (const auto& thunk : gpu_proto.thunks()) {
    ABSL_RETURN_IF_ERROR(emit_thunk(thunk));
  }

  mlir::func::ReturnOp::create(builder, loc);
  return module;
}

absl::StatusOr<GpuExecutableProto> LowerThunkyModuleToGpuExecutableProto(
    mlir::ModuleOp module) {
  mlir::func::FuncOp main_func = nullptr;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "main") {
      main_func = func;
    }
  });
  if (!main_func) {
    return absl::InvalidArgumentError(
        "thunky MLIR module missing @main function");
  }

  GpuExecutableProto gpu_proto;
  mlir::Block& entry = main_func.getBody().front();
  llvm::DenseMap<mlir::Value, BufferAllocationSliceProto> value_to_slice;

  // 1. Assign sequential virtual allocation indices `0 .. N-1` to `@main`'s
  //    block arguments (user input/output buffers followed by scratch buffers).
  int64_t next_alloc_index = 0;
  for (mlir::BlockArgument arg : entry.getArguments()) {
    int64_t idx = next_alloc_index++;
    int64_t size =
        mlir::cast<mlir::thunky::BufferType>(arg.getType()).getSize();
    auto* alloc_proto = gpu_proto.mutable_buffer_allocations()->add_values();
    alloc_proto->set_index(idx);
    alloc_proto->set_size(size);

    BufferAllocationSliceProto slice_proto;
    slice_proto.set_buffer_allocation_index(idx);
    slice_proto.set_offset(0);
    slice_proto.set_size(size);
    value_to_slice[arg] = slice_proto;
  }

  // 2. Walk `@main` in pre-order to assign virtual constant allocations to
  //    `thunky.constant` ops and resolve `thunky.slice_buffer` sub-slices
  //    relative to their parent buffer slices.
  int64_t const_count = 0;
  main_func.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (auto const_op = mlir::dyn_cast<mlir::thunky::ConstantOp>(op)) {
      int64_t idx = next_alloc_index++;
      int64_t size =
          mlir::cast<mlir::thunky::BufferType>(const_op.getResult().getType())
              .getSize();
      auto* alloc_proto = gpu_proto.mutable_buffer_allocations()->add_values();
      alloc_proto->set_index(idx);
      alloc_proto->set_size(size);
      alloc_proto->set_is_constant(true);

      auto* const_proto = gpu_proto.add_constants();
      const_proto->set_symbol_name(
          absl::StrCat("thunky_const_", const_count++));
      const_proto->set_allocation_index(idx);
      const_proto->mutable_content()->set_data(
          std::string(const_op.getValue()));

      BufferAllocationSliceProto slice_proto;
      slice_proto.set_buffer_allocation_index(idx);
      slice_proto.set_offset(0);
      slice_proto.set_size(size);
      value_to_slice[const_op.getResult()] = slice_proto;
    } else if (auto slice_op =
                   mlir::dyn_cast<mlir::thunky::SliceBufferOp>(op)) {
      const BufferAllocationSliceProto& parent =
          value_to_slice.at(slice_op.getBuffer());
      int64_t offset = slice_op.getOffset();
      int64_t size =
          mlir::cast<mlir::thunky::BufferType>(slice_op.getResult().getType())
              .getSize();
      BufferAllocationSliceProto slice_proto;
      slice_proto.set_buffer_allocation_index(parent.buffer_allocation_index());
      slice_proto.set_offset(parent.offset() + offset);
      slice_proto.set_size(size);
      value_to_slice[slice_op.getResult()] = slice_proto;
    }
  });

  // 3. Lower `@main`'s operations into `gpu_proto.thunks()`.
  llvm::DenseMap<mlir::Value, uint64_t> token_to_async_id;
  int64_t next_thunk_id = 1;
  ABSL_RETURN_IF_ERROR(LowerBlockToThunkProtos(entry, value_to_slice,
                                               token_to_async_id, next_thunk_id,
                                               gpu_proto.mutable_thunks()));
  return gpu_proto;
}

}  // namespace xla::gpu
