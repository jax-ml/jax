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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "jaxlib/thunky/dialect/capi.h"
#include "jaxlib/thunky/dialect/thunky_dialect.h"
#include "jaxlib/thunky/thunk_proto_util.h"
#include "third_party/riegeli/bytes/string_reader.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/reduction_kind.pb.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/execution_state.pb.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/primitive_util.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/dense_data_intermediate.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/shaped_slice.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace nb = nanobind;

namespace xla::gpu {
namespace {

template <typename T>
T ValueOrThrow(absl::StatusOr<T> status_or) {
  if (!status_or.ok()) {
    throw std::runtime_error(std::string(status_or.status().message()));
  }
  return *std::move(status_or);
}

inline void ThrowIfError(absl::Status status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

absl::StatusOr<GpuExecutableProto> ParseGpuExecutableProto(
    std::string_view serialized_bytes) {
  std::string_view payload = serialized_bytes;
  riegeli::StringReader<> init_reader(payload);
  auto is_split = IsSplitProto(init_reader);
  if (!is_split.ok() || !*is_split) {
    tsl::protobuf::io::ArrayInputStream array_stream(payload.data(),
                                                     payload.size());
    tsl::protobuf::io::CodedInputStream coded_stream(&array_stream);
    xla::ifrt::SerializedXlaExecutableMetadata metadata;
    if (tsl::protobuf::util::ParseDelimitedFromCodedStream(
            &metadata, &coded_stream, nullptr) &&
        !metadata.runtime_name().empty()) {
      payload = payload.substr(coded_stream.CurrentPosition());
    }
  }

  ExecutableAndOptionsProto exe_and_opts;
  auto reader = std::make_unique<riegeli::StringReader<>>(payload);
  auto is_split_exe = IsSplitProto(*reader);
  if (is_split_exe.ok() && *is_split_exe) {
    absl::Status status = ReadSplitProto(std::move(reader), exe_and_opts);
    if (!status.ok()) {
      return status;
    }
  } else {
    if (!exe_and_opts.ParseFromString(payload)) {
      return absl::InvalidArgumentError(
          "Failed to parse ExecutableAndOptionsProto");
    }
  }

  GpuExecutableProto gpu_exe_proto;
  std::string_view gpu_payload = exe_and_opts.serialized_executable();
  auto gpu_reader = std::make_unique<riegeli::StringReader<>>(gpu_payload);
  auto is_split_gpu = IsSplitProto(*gpu_reader);
  if (is_split_gpu.ok() && *is_split_gpu) {
    absl::Status status = ReadSplitProto(std::move(gpu_reader), gpu_exe_proto);
    if (!status.ok()) {
      return status;
    }
  } else {
    if (!gpu_exe_proto.ParseFromString(gpu_payload)) {
      return absl::InvalidArgumentError("Failed to parse GpuExecutableProto");
    }
  }
  return gpu_exe_proto;
}

absl::Status RemapBufferAllocations(
    ThunkProto* thunk, const absl::flat_hash_map<int64_t, int64_t>& mapping) {
  return ForEachBufferSlice(thunk, [&](auto* slice) {
    auto it = mapping.find(slice->buffer_allocation_index());
    if (it != mapping.end()) {
      slice->set_buffer_allocation_index(it->second);
    }
  });
}

bool ExtractEmbeddedThunkMetadata(ThunkProto* thunk,
                                  GpuExecutableProto* embedded_meta) {
  std::string annotation = thunk->thunk_info().profile_annotation();
  constexpr std::string_view kMarker = "||thunky_embedded_meta:";
  size_t pos = annotation.find(kMarker);
  if (pos == std::string::npos) {
    return false;
  }
  std::string meta_b64 = annotation.substr(pos + kMarker.size());
  thunk->mutable_thunk_info()->set_profile_annotation(
      annotation.substr(0, pos));
  std::string meta_bytes;
  if (!absl::Base64Unescape(meta_b64, &meta_bytes)) {
    return false;
  }
  return embedded_meta->ParseFromString(meta_bytes);
}

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

mlir::ArrayAttr MakeReplicaGroupsAttr(mlir::OpBuilder& builder,
                                      const CollectiveConfigProto& config) {
  llvm::SmallVector<mlir::Attribute> group_attrs;
  for (const auto& rg : config.replica_groups()) {
    llvm::SmallVector<int64_t> ids(rg.replica_ids().begin(),
                                   rg.replica_ids().end());
    group_attrs.push_back(builder.getDenseI64ArrayAttr(ids));
  }
  return builder.getArrayAttr(group_attrs);
}

MlirModule JaxExecutableToMlir(nb::bytes serialized_bytes,
                               MlirContext context_c) {
  std::string_view bytes_view(serialized_bytes.c_str(),
                              serialized_bytes.size());
  GpuExecutableProto gpu_proto =
      ValueOrThrow(ParseGpuExecutableProto(bytes_view));

  mlir::MLIRContext* context = unwrap(context_c);
  context->getOrLoadDialect<mlir::thunky::ThunkyDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(context);
  mlir::Location loc = builder.getUnknownLoc();
  mlir::ModuleOp module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

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
  for (const auto& [pnum, idx] : param_num_and_idx) {
    param_alloc_indices.push_back(idx);
  }

  Shape result_shape =
      ValueOrThrow(Shape::FromProto(gpu_proto.program_shape().result()));
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
  std::vector<int64_t> out_alloc_indices;
  out_alloc_indices.reserve(out_index_and_alloc.size());
  for (const auto& [shape_idx, alloc_idx] : out_index_and_alloc) {
    out_alloc_indices.push_back(alloc_idx);
  }

  std::set<int64_t> param_alloc_set(param_alloc_indices.begin(),
                                    param_alloc_indices.end());
  std::vector<int64_t> fresh_out_alloc_indices;
  for (int64_t idx : out_alloc_indices) {
    if (!param_alloc_set.contains(idx)) {
      fresh_out_alloc_indices.push_back(idx);
    }
  }

  std::set<int64_t> constant_alloc_indices;
  for (const auto& c : gpu_proto.constants()) {
    if (c.allocation_index() != -1) {
      constant_alloc_indices.insert(c.allocation_index());
    }
  }

  // Verify that any aliased output matches its alias_config if present.
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
    if (!param_or_out_indices.contains(alloc.index())) {
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

  auto get_slice_val =
      [&](const xla::buffer_assignment::BufferAllocationSliceProto& slice)
      -> mlir::Value {
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

  absl::flat_hash_map<uint64_t, mlir::Value> async_exec_id_to_token;
  auto emit_thunk = [&](const ThunkProto& thunk, auto& emit_thunk_ref) -> void {
    if (thunk.has_sequential_thunk()) {
      for (const auto& child : thunk.sequential_thunk().thunks()) {
        emit_thunk_ref(child, emit_thunk_ref);
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
        emit_thunk_ref(child, emit_thunk_ref);
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
          emit_thunk_ref(child, emit_thunk_ref);
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
          emit_thunk_ref(child, emit_thunk_ref);
        }
        mlir::thunky::YieldOp::create(builder, loc);
      }
      {
        mlir::Block* body_block = &while_op.getBodyRegion().emplaceBlock();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(body_block);
        for (const auto& child : wt.body_thunk_sequence().thunks()) {
          emit_thunk_ref(child, emit_thunk_ref);
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
        emit_thunk_ref(child, emit_thunk_ref);
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
            MakeReplicaGroupsAttr(builder, art.collective_config()),
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
            MakeReplicaGroupsAttr(builder, agt.collective_config()),
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
            MakeReplicaGroupsAttr(builder, rst.collective_config()),
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
          MakeReplicaGroupsAttr(builder, a2at.collective_config()),
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
            MakeReplicaGroupsAttr(builder, cpt.collective_config()));
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
    } else if (thunk.has_device_to_device_copy_thunk() &&
               !constant_alloc_indices.contains(
                   thunk.device_to_device_copy_thunk()
                       .copy_thunk()
                       .source_buffer()
                       .slice()
                       .buffer_allocation_index())) {
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
      ThunkProto cleaned_thunk = thunk;
      GpuExecutableProto embedded_meta;
      bool has_embedded_meta =
          ExtractEmbeddedThunkMetadata(&cleaned_thunk, &embedded_meta);

      std::vector<int64_t> ref_indices;
      ThrowIfError(CollectReferencedAllocations(&cleaned_thunk, ref_indices));
      if (has_embedded_meta) {
        for (const auto& c : embedded_meta.constants()) {
          if (c.allocation_index() != -1) {
            int64_t idx = c.allocation_index();
            if (std::find(ref_indices.begin(), ref_indices.end(), idx) ==
                ref_indices.end()) {
              ref_indices.push_back(idx);
            }
          }
        }
      }
      absl::flat_hash_map<int64_t, int64_t> local_mapping;
      llvm::SmallVector<mlir::Value> operands;
      for (size_t j = 0; j < ref_indices.size(); ++j) {
        local_mapping[ref_indices[j]] = static_cast<int64_t>(j);
        operands.push_back(alloc_to_val.at(ref_indices[j]));
      }
      ThunkProto local_thunk = cleaned_thunk;
      ThrowIfError(RemapBufferAllocations(&local_thunk, local_mapping));
      std::string thunk_bytes = local_thunk.SerializeAsString();

      mlir::StringAttr metadata_attr;
      bool needs_metadata =
          !gpu_proto.constants().empty() ||
          (has_embedded_meta && !embedded_meta.constants().empty()) ||
          (!cleaned_thunk.has_kernel_thunk() &&
           !cleaned_thunk.has_custom_kernel_thunk());
      if (needs_metadata) {
        GpuExecutableProto metadata_proto;
        metadata_proto.mutable_dnn_compiled_graphs()->insert(
            gpu_proto.dnn_compiled_graphs().begin(),
            gpu_proto.dnn_compiled_graphs().end());
        if (has_embedded_meta) {
          metadata_proto.mutable_dnn_compiled_graphs()->insert(
              embedded_meta.dnn_compiled_graphs().begin(),
              embedded_meta.dnn_compiled_graphs().end());
        }
        if (has_embedded_meta && embedded_meta.has_hlo_module_with_config()) {
          *metadata_proto.mutable_hlo_module_with_config() =
              embedded_meta.hlo_module_with_config();
        } else if (gpu_proto.has_hlo_module_with_config()) {
          *metadata_proto.mutable_hlo_module_with_config() =
              gpu_proto.hlo_module_with_config();
        }
        for (const auto& c : gpu_proto.constants()) {
          if (c.allocation_index() == -1 ||
              local_mapping.contains(c.allocation_index())) {
            auto* new_c = metadata_proto.add_constants();
            *new_c = c;
            if (c.allocation_index() != -1) {
              new_c->set_allocation_index(
                  local_mapping.at(c.allocation_index()));
            }
          }
        }
        if (has_embedded_meta) {
          for (const auto& c : embedded_meta.constants()) {
            if (c.allocation_index() == -1 ||
                local_mapping.contains(c.allocation_index())) {
              auto* new_c = metadata_proto.add_constants();
              *new_c = c;
              if (c.allocation_index() != -1) {
                new_c->set_allocation_index(
                    local_mapping.at(c.allocation_index()));
              }
            }
          }
        }
        metadata_attr =
            builder.getStringAttr(metadata_proto.SerializeAsString());
      }

      std::string asm_text_str =
          (has_embedded_meta && !embedded_meta.asm_text().empty())
              ? embedded_meta.asm_text()
              : gpu_proto.asm_text();
      std::string binary_str =
          (has_embedded_meta && !embedded_meta.binary().empty())
              ? embedded_meta.binary()
              : gpu_proto.binary();
      if (cleaned_thunk.has_custom_kernel_thunk()) {
        const auto& ck = cleaned_thunk.custom_kernel_thunk().custom_kernel();
        if (binary_str.empty() && ck.kernel_spec().has_cubin()) {
          binary_str = ck.kernel_spec().cubin().data();
        }
        if (asm_text_str.empty() && ck.kernel_spec().has_ptx()) {
          asm_text_str = ck.kernel_spec().ptx().data();
        }
      }

      std::string kernel_name = cleaned_thunk.has_kernel_thunk()
                                    ? cleaned_thunk.kernel_thunk().kernel_name()
                                    : (cleaned_thunk.has_custom_kernel_thunk()
                                           ? cleaned_thunk.custom_kernel_thunk()
                                                 .custom_kernel()
                                                 .name()
                                           : "thunk");
      mlir::thunky::CallThunkProtoOp::create(
          builder, loc, operands, builder.getStringAttr(kernel_name),
          builder.getStringAttr(thunk_bytes),
          builder.getStringAttr(asm_text_str),
          builder.getStringAttr(binary_str), metadata_attr);
    }
  };

  for (const auto& thunk : gpu_proto.thunks()) {
    emit_thunk(thunk, emit_thunk);
  }

  mlir::func::ReturnOp::create(builder, loc);
  return wrap(module);
}

MlirType MlirBufferType(int64_t size, MlirContext context_c) {
  return mlirThunkyBufferTypeGet(context_c, size);
}

MlirType MlirTokenType(MlirContext context_c) {
  return mlirThunkyTokenTypeGet(context_c);
}

}  // namespace
}  // namespace xla::gpu

NB_MODULE(_thunky_ext, m) {
  nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"));

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__thunky__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("jax_executable_to_mlir", &xla::gpu::JaxExecutableToMlir,
        nb::arg("serialized_bytes"), nb::arg("context") = nb::none());
  m.def("mlir_buffer_type", &xla::gpu::MlirBufferType, nb::arg("size"),
        nb::arg("context") = nb::none());
  m.def("mlir_token_type", &xla::gpu::MlirTokenType,
        nb::arg("context") = nb::none());
}
