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

#include "jaxlib/thunky/mlir_to_thunks.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/thunky/dialect/thunky_dialect.h"
#include "jaxlib/thunky/thunk_proto_util.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/gpu_module_globals.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/service/shaped_slice.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {

absl::StatusOr<ReductionKind> ParseReductionKind(absl::string_view str) {
  if (str == "sum") return ReductionKind::SUM;
  if (str == "prod" || str == "product") return ReductionKind::PRODUCT;
  if (str == "min") return ReductionKind::MIN;
  if (str == "max") return ReductionKind::MAX;
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown reduction kind: %s", str));
}

absl::StatusOr<CollectiveOpGroupMode> ParseGroupMode(absl::string_view str) {
  if (str == "flattened_id") return COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID;
  if (str == "cross_replica") return COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  if (str == "cross_partition") return COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION;
  if (str == "cross_replica_and_partition") {
    return COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown collective group mode: %s", str));
}

std::vector<ReplicaGroup> ParseReplicaGroups(mlir::ArrayAttr rg_attr) {
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

static se::GpuComputeCapability GetGpuComputeCapability() {
  auto platform_or = se::PlatformManager::PlatformWithName("CUDA");
  if (platform_or.ok()) {
    auto executor_or = (*platform_or)->ExecutorForDevice(0);
    if (executor_or.ok()) {
      return (*executor_or)->GetDeviceDescription().gpu_compute_capability();
    }
  }
  return se::CudaComputeCapability{8, 0};
}

absl::StatusOr<CollectiveConfig> ParseCollectiveConfig(
    mlir::Type elem_type, mlir::ArrayAttr rg_attr,
    absl::string_view group_mode_str) {
  PrimitiveType primitive_type =
      xla::ConvertMlirTypeToPrimitiveType(elem_type);
  if (primitive_type == xla::PRIMITIVE_TYPE_INVALID) {
    return absl::InvalidArgumentError(
        "Unsupported collective element type in MLIR");
  }
  ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                   ParseGroupMode(group_mode_str));
  CollectiveConfig config;
  config.operand_element_type = {primitive_type};
  config.replica_groups = ParseReplicaGroups(rg_attr);
  config.group_mode = group_mode;
  config.use_symmetric_buffer = false;
  return config;
}

absl::StatusOr<xla::Shape> ParseShapeAttr(mlir::Attribute attr,
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

std::vector<CollectiveThunk::Buffer> MakeCollectiveBuffers(
    BufferAllocation::Slice src_slice, BufferAllocation::Slice dst_slice,
    PrimitiveType elem_type) {
  int64_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(elem_type);
  int64_t src_elems = src_slice.size() / elem_size;
  int64_t dst_elems = dst_slice.size() / elem_size;
  Shape src_shape = ShapeUtil::MakeShape(elem_type, {src_elems});
  Shape dst_shape = ShapeUtil::MakeShape(elem_type, {dst_elems});
  return {CollectiveThunk::Buffer{
      /*element_count=*/src_elems,
      /*source_buffer=*/ShapedSlice{src_slice, src_shape},
      /*destination_buffer=*/ShapedSlice{dst_slice, dst_shape},
      /*source_memory_space=*/0,
      /*destination_memory_space=*/0,
  }};
}

absl::Status RebaseBufferAllocations(
    ThunkProto* thunk,
    absl::Span<const BufferAllocation::Slice> operand_slices) {
  return ForEachBufferSlice(thunk, [&](auto* slice) {
    int64_t idx = slice->buffer_allocation_index();
    if (idx >= 0 && idx < static_cast<int64_t>(operand_slices.size())) {
      const BufferAllocation::Slice& base = operand_slices[idx];
      slice->set_buffer_allocation_index(base.allocation()->index());
      slice->set_offset(base.offset() + slice->offset());
    }
  });
}

std::optional<absl::flat_hash_map<std::string, const HloInstruction*>>
MakeConstantsMap(const HloModule* debug_module) {
  if (!debug_module) {
    return std::nullopt;
  }
  absl::flat_hash_map<std::string, const HloInstruction*> constants;
  for (const HloComputation* computation :
       debug_module->MakeComputationSorted()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kConstant) {
        continue;
      }
      if (llvm_ir::SanitizeConstantName(*instr) != instr->name()) {
        continue;
      }
      constants.try_emplace(llvm_ir::ConstantHloToGlobalName(*instr), instr);
    }
  }
  return constants;
}

struct ConstantSliceInitializer {
  bool has_slice = false;
  BufferAllocation::Slice slice;
  std::string symbol_name;
  std::vector<uint8_t> content;
};

class EmbeddedThunkWrapper : public Command {
 public:
  EmbeddedThunkWrapper(
      ThunkInfo thunk_info, std::unique_ptr<Thunk> inner_thunk,
      std::string asm_text, std::vector<uint8_t> binary,
      absl::flat_hash_map<std::string, std::string> dnn_compiled_graphs,
      std::shared_ptr<std::vector<BufferAllocation>> owned_allocations,
      std::shared_ptr<HloModule> hlo_module,
      std::vector<ConstantSliceInitializer> constant_slice_initializers)
      : Command(inner_thunk->kind(), inner_thunk->thunk_info()),
        inner_thunk_(std::move(inner_thunk)),
        asm_text_(std::move(asm_text)),
        binary_(std::move(binary)),
        dnn_compiled_graphs_(std::move(dnn_compiled_graphs)),
        owned_allocations_(std::move(owned_allocations)),
        hlo_module_(std::move(hlo_module)),
        constant_slice_initializers_(std::move(constant_slice_initializers)) {}

  ~EmbeddedThunkWrapper() override {
    absl::MutexLock lock(mu_);
    for (auto& [executor, buf] : owned_constant_buffers_) {
      executor->Deallocate(&buf);
    }
  }

  absl::Status Prepare(const PrepareParams& params) override {
    return inner_thunk_->Prepare(params);
  }

  absl::Status EnsureConstantsInitialized(
      se::Stream* stream, const BufferAllocations* buffer_allocations) {
    if (stream == nullptr || constant_slice_initializers_.empty()) {
      return absl::OkStatus();
    }
    se::StreamExecutor* executor = stream->parent();
    {
      absl::MutexLock lock(mu_);
      if (!initialized_executors_.contains(executor)) {
        se::ModuleHandle handle;
        if (!binary_.empty()) {
          se::MultiModuleLoaderSpec module_spec;
          module_spec.AddCudaCubinInMemory(binary_);
          if (auto module_or = executor->LoadModule(module_spec);
              module_or.ok()) {
            handle = *module_or;
            loaded_modules_.emplace(executor,
                                    se::ScopedModuleHandle(executor, handle));
          }
        }
        for (size_t i = 0; i < constant_slice_initializers_.size(); ++i) {
          const auto& init = constant_slice_initializers_[i];
          se::DeviceAddressBase dev_addr;
          if (static_cast<bool>(handle) && !init.symbol_name.empty()) {
            if (auto sym_or = executor->GetSymbol(init.symbol_name, handle);
                sym_or.ok()) {
              dev_addr = *sym_or;
              if (!init.content.empty()) {
                RETURN_IF_ERROR(stream->Memcpy(&dev_addr, init.content.data(),
                                               init.content.size()));
              }
            }
          }
          if (dev_addr.opaque() == nullptr && !init.content.empty()) {
            se::DeviceAddressBase owned_buf =
                executor->Allocate(init.content.size(), /*memory_space=*/0);
            if (owned_buf.opaque() != nullptr) {
              RETURN_IF_ERROR(stream->Memcpy(&owned_buf, init.content.data(),
                                             init.content.size()));
              dev_addr = owned_buf;
              owned_constant_buffers_.emplace_back(executor, owned_buf);
            }
          }
          if (dev_addr.opaque() != nullptr) {
            const_device_addrs_[{executor, i}] = dev_addr;
          }
        }
        initialized_executors_.insert(executor);
      }
    }
    if (buffer_allocations != nullptr) {
      for (size_t i = 0; i < constant_slice_initializers_.size(); ++i) {
        const auto& init = constant_slice_initializers_[i];
        if (!init.has_slice) continue;
        se::DeviceAddressBase dst =
            buffer_allocations->GetDeviceAddress(init.slice);
        se::DeviceAddressBase src;
        {
          absl::MutexLock lock(mu_);
          auto it = const_device_addrs_.find({executor, i});
          if (it != const_device_addrs_.end()) {
            src = it->second;
          }
        }
        if (src.opaque() != nullptr && src.opaque() != dst.opaque()) {
          RETURN_IF_ERROR(stream->MemcpyD2D(
              &dst, src, std::min<size_t>(src.size(), init.slice.size())));
        } else if (src.opaque() == nullptr && !init.content.empty()) {
          RETURN_IF_ERROR(stream->Memcpy(
              &dst, init.content.data(),
              std::min<size_t>(init.content.size(), init.slice.size())));
        }
      }
    }
    return absl::OkStatus();
  }

  absl::Status Initialize(const InitializeParams& params) override {
    RETURN_IF_ERROR(
        EnsureConstantsInitialized(params.stream, params.buffer_allocations));
    InitializeParams child_params = params;
    child_params.src = {asm_text_, binary_, dnn_compiled_graphs_};
    return inner_thunk_->Initialize(child_params);
  }

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    RETURN_IF_ERROR(
        EnsureConstantsInitialized(params.stream, params.buffer_allocations));
    return inner_thunk_->ExecuteOnStream(params);
  }

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override {
    RETURN_IF_ERROR(EnsureConstantsInitialized(
        execute_params.stream, execute_params.buffer_allocations));
    auto* cmd = dynamic_cast<Command*>(inner_thunk_.get());
    if (cmd == nullptr) {
      return absl::UnimplementedError(absl::StrCat(
          "Inner thunk of kind ", Thunk::KindToString(inner_thunk_->kind()),
          " is not a Command"));
    }
    return cmd->Record(execute_params, record_params, std::move(record_action),
                       command_buffer);
  }

  bool requires_update_on_initialize() const override {
    if (auto* cmd = dynamic_cast<const Command*>(inner_thunk_.get())) {
      return cmd->requires_update_on_initialize();
    }
    return false;
  }

  bool requires_warmup() const override {
    if (auto* cmd = dynamic_cast<const Command*>(inner_thunk_.get())) {
      return cmd->requires_warmup();
    }
    return false;
  }

  bool requires_update_on_execute() const override {
    if (auto* cmd = dynamic_cast<const Command*>(inner_thunk_.get())) {
      return cmd->requires_update_on_execute();
    }
    return false;
  }

  bool IsTracedCommand() const override {
    if (auto* cmd = dynamic_cast<const Command*>(inner_thunk_.get())) {
      return cmd->IsTracedCommand();
    }
    return false;
  }

  bool support_loop_unroll() const override {
    if (auto* cmd = dynamic_cast<const Command*>(inner_thunk_.get())) {
      return cmd->support_loop_unroll();
    }
    return true;
  }

  BufferUses buffer_uses() const override {
    return inner_thunk_->buffer_uses();
  }
  ResourceUses resource_uses() const override {
    return inner_thunk_->resource_uses();
  }

  absl::Status WalkNested(Walker pre_order, Walker post_order) override {
    return inner_thunk_->Walk(pre_order, post_order);
  }

  absl::Status WalkNested(ConstWalker pre_order,
                          ConstWalker post_order) const override {
    return static_cast<const Thunk*>(inner_thunk_.get())
        ->Walk(pre_order, post_order);
  }

  absl::StatusOr<ThunkProto> ToProto() const override {
    ASSIGN_OR_RETURN(ThunkProto proto, inner_thunk_->ToProto());
    GpuExecutableProto embedded_meta;
    embedded_meta.set_asm_text(asm_text_);
    embedded_meta.set_binary(binary_.data(), binary_.size());
    embedded_meta.mutable_dnn_compiled_graphs()->insert(
        dnn_compiled_graphs_.begin(), dnn_compiled_graphs_.end());
    if (hlo_module_ != nullptr) {
      *embedded_meta.mutable_hlo_module_with_config() =
          hlo_module_->ToProtoWithConfig();
    }
    for (const auto& init : constant_slice_initializers_) {
      auto* c_proto = embedded_meta.add_constants();
      if (init.has_slice) {
        c_proto->set_symbol_name(absl::StrCat(
            init.symbol_name, "||thunky_slice:", init.slice.offset(), ":",
            init.slice.size()));
      } else {
        c_proto->set_symbol_name(init.symbol_name);
      }
      c_proto->set_allocation_index(
          init.has_slice ? init.slice.allocation()->index() : -1);
      c_proto->mutable_content()->set_data(init.content.data(),
                                           init.content.size());
    }
    std::string orig_annotation = proto.thunk_info().profile_annotation();
    std::string meta_bytes = embedded_meta.SerializeAsString();
    proto.mutable_thunk_info()->set_profile_annotation(absl::StrCat(
        orig_annotation,
        "||thunky_embedded_meta:", absl::Base64Escape(meta_bytes)));
    return proto;
  }

 private:
  std::unique_ptr<Thunk> inner_thunk_;
  std::string asm_text_;
  std::vector<uint8_t> binary_;
  absl::flat_hash_map<std::string, std::string> dnn_compiled_graphs_;
  std::shared_ptr<std::vector<BufferAllocation>> owned_allocations_;
  std::shared_ptr<HloModule> hlo_module_;
  std::vector<ConstantSliceInitializer> constant_slice_initializers_;

  absl::Mutex mu_;
  absl::flat_hash_set<se::StreamExecutor*> initialized_executors_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<se::StreamExecutor*, se::ScopedModuleHandle>
      loaded_modules_ ABSL_GUARDED_BY(mu_);
  std::vector<std::pair<se::StreamExecutor*, se::DeviceAddressBase>>
      owned_constant_buffers_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::pair<se::StreamExecutor*, size_t>,
                      se::DeviceAddressBase>
      const_device_addrs_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

absl::StatusOr<ThunkSequence> LowerBlockToThunkSequence(
    mlir::Block& block,
    const llvm::DenseMap<mlir::Value, BufferAllocation::Slice>& value_to_slice,
    llvm::DenseMap<mlir::Value, std::shared_ptr<AsyncExecution>>&
        token_to_async_exec,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info) {
  ThunkSequence thunks;

  for (mlir::Operation& op_ref : block) {
    mlir::Operation* op = &op_ref;
    if (mlir::isa<mlir::thunky::SliceBufferOp>(op)) {
      continue;
    } else if (auto memzero_op = mlir::dyn_cast<mlir::thunky::MemzeroOp>(op)) {
      BufferAllocation::Slice slice = value_to_slice.at(memzero_op.getBuffer());
      Shape shape = ShapeUtil::MakeShape(U8, {slice.size()});
      thunks.push_back(std::make_unique<MemzeroThunk>(
          make_thunk_info(), ShapedSlice{slice, shape}));
    } else if (auto memset_op = mlir::dyn_cast<mlir::thunky::Memset32Op>(op)) {
      BufferAllocation::Slice slice = value_to_slice.at(memset_op.getBuffer());
      uint32_t val = static_cast<uint32_t>(memset_op.getValueAttr().getInt());
      thunks.push_back(std::make_unique<Memset32BitValueThunk>(
          make_thunk_info(), val, slice));
    } else if (auto copy_op = mlir::dyn_cast<mlir::thunky::CopyOp>(op)) {
      BufferAllocation::Slice src_slice = value_to_slice.at(copy_op.getSrc());
      BufferAllocation::Slice dst_slice = value_to_slice.at(copy_op.getDst());
      if (src_slice == dst_slice) {
        continue;
      }
      Shape shape = ShapeUtil::MakeShape(U8, {src_slice.size()});
      thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
          make_thunk_info(), ShapedSlice{src_slice, shape},
          ShapedSlice{dst_slice, shape}, src_slice.size()));
    } else if (auto ptx_op = mlir::dyn_cast<mlir::thunky::PtxKernelOp>(op)) {
      std::string kernel_name(ptx_op.getKernelName());
      std::string ptx(ptx_op.getPtx());
      llvm::ArrayRef<int64_t> grid = ptx_op.getGridDim();
      llvm::ArrayRef<int64_t> block = ptx_op.getBlockDim();
      int64_t shmem_bytes = ptx_op.getShmemBytesAttr().getInt();

      auto args = ptx_op.getBuffers();
      llvm::ArrayRef<bool> written_flags = ptx_op.getWritten();
      std::vector<emitters::KernelArgument> kargs;
      for (size_t i = 0; i < args.size(); ++i) {
        BufferAllocation::Slice slice = value_to_slice.at(args[i]);
        Shape shape = ShapeUtil::MakeShape(U8, {slice.size()});
        emitters::KernelArgument karg(shape, slice);
        karg.set_written(written_flags[i]);
        kargs.push_back(std::move(karg));
      }
      emitters::KernelArguments kernel_arguments(std::move(kargs));
      se::KernelLoaderSpec kernel_spec =
          se::KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
              std::move(ptx), kernel_name, args.size());
      CustomKernel custom_kernel(kernel_name, std::move(kernel_spec),
                                 se::BlockDim(grid[0], grid[1], grid[2]),
                                 se::ThreadDim(block[0], block[1], block[2]),
                                 shmem_bytes);
      thunks.push_back(std::make_unique<CustomKernelThunk>(
          make_thunk_info(), std::move(custom_kernel), kernel_arguments));
    } else if (auto kernel_op =
                   mlir::dyn_cast<mlir::thunky::CallThunkProtoOp>(op)) {
      ThunkProto thunk_proto;
      if (!thunk_proto.ParseFromString(
              std::string(kernel_op.getThunkProto()))) {
        return absl::InvalidArgumentError(
            "Failed to parse ThunkProto in thunky.call_thunk_proto");
      }
      GpuExecutableProto metadata_proto;
      if (auto meta_attr = kernel_op.getExecutableMetadataProto()) {
        if (!metadata_proto.ParseFromString(std::string(*meta_attr))) {
          return absl::InvalidArgumentError(
              "Failed to parse GpuExecutableProto metadata in "
              "thunky.call_thunk_proto");
        }
      }

      auto buffers = kernel_op.getBuffers();
      std::vector<BufferAllocation::Slice> operand_slices;
      operand_slices.reserve(buffers.size());
      int64_t max_alloc_index = 0;
      for (mlir::Value buf_val : buffers) {
        BufferAllocation::Slice base_slice = value_to_slice.at(buf_val);
        operand_slices.push_back(base_slice);
        max_alloc_index = std::max<int64_t>(max_alloc_index,
                                            base_slice.allocation()->index());
      }

      auto owned_allocations =
          std::make_shared<std::vector<BufferAllocation>>();
      owned_allocations->reserve(max_alloc_index + 1);
      for (int64_t k = 0; k <= max_alloc_index; ++k) {
        int64_t required_size = 1024 * 1024;
        for (const auto& s : operand_slices) {
          if (s.allocation()->index() == k) {
            required_size = std::max(required_size, s.allocation()->size());
            required_size = std::max(required_size, s.offset() + s.size());
          }
        }
        owned_allocations->emplace_back(k, required_size, /*color=*/0);
      }

      std::shared_ptr<HloModule> hlo_module;
      if (metadata_proto.has_hlo_module_with_config()) {
        ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                         HloModule::CreateFromProtoWithConfig(
                             metadata_proto.hlo_module_with_config()));
        hlo_module = std::move(mod);
      }

      std::optional<absl::flat_hash_map<std::string, const HloInstruction*>>
          name_to_const = MakeConstantsMap(hlo_module.get());

      std::vector<ConstantSliceInitializer> constant_slice_initializers;
      for (const auto& constant_proto : metadata_proto.constants()) {
        std::string orig_symbol = constant_proto.symbol_name();
        int64_t saved_offset = 0;
        int64_t saved_size = -1;
        constexpr std::string_view kSliceMarker = "||thunky_slice:";
        size_t marker_pos = orig_symbol.find(kSliceMarker);
        if (marker_pos != std::string::npos) {
          std::string slice_part =
              orig_symbol.substr(marker_pos + kSliceMarker.size());
          orig_symbol = orig_symbol.substr(0, marker_pos);
          size_t colon_pos = slice_part.find(':');
          if (colon_pos != std::string::npos) {
            (void)absl::SimpleAtoi(slice_part.substr(0, colon_pos),
                                   &saved_offset);
            (void)absl::SimpleAtoi(slice_part.substr(colon_pos + 1),
                                   &saved_size);
          }
        }
        std::vector<uint8_t> bytes;
        if (constant_proto.has_content() &&
            !constant_proto.content().data().empty()) {
          const std::string& data = constant_proto.content().data();
          bytes.assign(data.begin(), data.end());
        } else {
          GpuExecutableProto::ConstantInfoProto clean_proto = constant_proto;
          clean_proto.set_symbol_name(orig_symbol);
          ASSIGN_OR_RETURN(
              GpuModuleGlobals::ConstantInfo info,
              GpuModuleGlobals::ConstantInfo::FromProto(
                  clean_proto,
                  name_to_const.has_value() ? &*name_to_const : nullptr));
          bytes.assign(info.content.span().begin(), info.content.span().end());
        }
        int64_t local_idx = constant_proto.allocation_index();
        if (local_idx >= 0 &&
            local_idx < static_cast<int64_t>(operand_slices.size())) {
          BufferAllocation::Slice base = operand_slices[local_idx];
          int64_t slice_offset = base.offset() + saved_offset;
          int64_t slice_size = saved_size >= 0 ? saved_size : base.size();
          BufferAllocation::Slice rebased_slice(
              &(*owned_allocations)[base.allocation()->index()], slice_offset,
              slice_size);
          constant_slice_initializers.push_back(
              ConstantSliceInitializer{/*has_slice=*/true, rebased_slice,
                                       orig_symbol, std::move(bytes)});
        } else {
          constant_slice_initializers.push_back(ConstantSliceInitializer{
              /*has_slice=*/false, BufferAllocation::Slice(), orig_symbol,
              std::move(bytes)});
        }
      }

      RETURN_IF_ERROR(RebaseBufferAllocations(&thunk_proto, operand_slices));

      ThunkSequenceProto seq_proto;
      *seq_proto.add_thunks() = thunk_proto;
      ASSIGN_OR_RETURN(ThunkSequence deserialized_seq,
                       DeserializeThunkSequenceProto(
                           seq_proto, *owned_allocations, hlo_module.get(),
                           "CUDA", GetGpuComputeCapability()));

      absl::flat_hash_map<std::string, std::string> dnn_compiled_graphs(
          metadata_proto.dnn_compiled_graphs().begin(),
          metadata_proto.dnn_compiled_graphs().end());
      std::string asm_text_str(kernel_op.getAsmText());
      std::vector<uint8_t> binary_vec(kernel_op.getBinary().begin(),
                                      kernel_op.getBinary().end());

      std::function<void(std::unique_ptr<Thunk>)> emit_leaf_thunk =
          [&](std::unique_ptr<Thunk> t) {
            if (t->kind() == Thunk::kSequential) {
              auto* seq = static_cast<SequentialThunk*>(t.get());
              for (auto& child : seq->thunks()) {
                emit_leaf_thunk(std::move(child));
              }
              return;
            }
            if (t->kind() == Thunk::kPartitionId ||
                t->kind() == Thunk::kReplicaId) {
              return;
            }
            thunks.push_back(std::make_unique<EmbeddedThunkWrapper>(
                make_thunk_info(), std::move(t), asm_text_str, binary_vec,
                dnn_compiled_graphs, owned_allocations, hlo_module,
                constant_slice_initializers));
          };
      for (auto& t : deserialized_seq) {
        emit_leaf_thunk(std::move(t));
      }
    } else if (auto custom_call_op =
                   mlir::dyn_cast<mlir::thunky::CustomCallOp>(op)) {
      int64_t num_cc_operands = custom_call_op.getNumOperandsAttr().getInt();
      int64_t num_cc_results = custom_call_op.getNumResultsAttr().getInt();
      std::vector<NullableShapedSlice> cc_operands;
      std::vector<NullableShapedSlice> cc_results;
      auto buffers = custom_call_op.getBuffers();
      auto operand_shapes_attr = custom_call_op.getOperandShapesAttr();
      auto result_shapes_attr = custom_call_op.getResultShapesAttr();
      for (int64_t i = 0; i < num_cc_operands; ++i) {
        BufferAllocation::Slice slice = value_to_slice.at(buffers[i]);
        mlir::Attribute shape_attr = nullptr;
        if (operand_shapes_attr &&
            static_cast<size_t>(i) < operand_shapes_attr.size()) {
          shape_attr = operand_shapes_attr[i];
        }
        ASSIGN_OR_RETURN(xla::Shape shape,
                         ParseShapeAttr(shape_attr, slice.size()));
        cc_operands.push_back(NullableShapedSlice(ShapedSlice{slice, shape}));
      }
      for (int64_t i = 0; i < num_cc_results; ++i) {
        BufferAllocation::Slice slice =
            value_to_slice.at(buffers[num_cc_operands + i]);
        mlir::Attribute shape_attr = nullptr;
        if (result_shapes_attr &&
            static_cast<size_t>(i) < result_shapes_attr.size()) {
          shape_attr = result_shapes_attr[i];
        }
        ASSIGN_OR_RETURN(xla::Shape shape,
                         ParseShapeAttr(shape_attr, slice.size()));
        cc_results.push_back(NullableShapedSlice(ShapedSlice{slice, shape}));
      }
      ASSIGN_OR_RETURN(
          xla::ffi::AttributesMap cc_attributes,
          xla::ffi::BuildAttributesMap(custom_call_op.getBackendConfig()));
      ASSIGN_OR_RETURN(
          auto thunk,
          CustomCallThunk::Create(
              make_thunk_info(), std::string(custom_call_op.getTargetName()),
              std::move(cc_operands), std::move(cc_results),
              std::move(cc_attributes),
              /*called_computation=*/nullptr, "CUDA", GetGpuComputeCapability(),
              /*execution_state=*/nullptr, xla::cpu::TargetMachineOptions(),
              /*use_pdl=*/false));
      thunks.push_back(std::move(thunk));
    } else if (auto cond_op = mlir::dyn_cast<mlir::thunky::CondOp>(op)) {
      BufferAllocation::Slice pred_slice =
          value_to_slice.at(cond_op.getBranchIndex());
      Shape pred_shape = pred_slice.size() == 1 ? ShapeUtil::MakeShape(PRED, {})
                                                : ShapeUtil::MakeShape(S32, {});
      std::vector<ThunkSequence> branches;
      for (mlir::Region& region : cond_op.getBranches()) {
        ASSIGN_OR_RETURN(
            ThunkSequence branch_thunks,
            LowerBlockToThunkSequence(region.front(), value_to_slice,
                                      token_to_async_exec, make_thunk_info));
        branches.push_back(std::move(branch_thunks));
      }
      thunks.push_back(std::make_unique<ConditionalThunk>(
          make_thunk_info(), ShapedSlice{pred_slice, pred_shape},
          std::move(branches)));
    } else if (auto while_op = mlir::dyn_cast<mlir::thunky::WhileOp>(op)) {
      BufferAllocation::Slice cond_slice =
          value_to_slice.at(while_op.getConditionBuffer());
      ASSIGN_OR_RETURN(ThunkSequence cond_thunks,
                       LowerBlockToThunkSequence(
                           while_op.getCondRegion().front(), value_to_slice,
                           token_to_async_exec, make_thunk_info));
      ASSIGN_OR_RETURN(ThunkSequence body_thunks,
                       LowerBlockToThunkSequence(
                           while_op.getBodyRegion().front(), value_to_slice,
                           token_to_async_exec, make_thunk_info));
      thunks.push_back(std::make_unique<WhileThunk>(
          make_thunk_info(), cond_slice, std::move(cond_thunks),
          std::move(body_thunks)));
    } else if (auto async_start_op =
                   mlir::dyn_cast<mlir::thunky::AsyncStartOp>(op)) {
      uint64_t stream_id = async_start_op.getStreamId();
      bool is_comm = async_start_op.getIsCommunication();
      ExecutionStreamId exec_stream_id =
          is_comm ? ExecutionStreamId(CommunicationStreamId(stream_id))
                  : ExecutionStreamId(ComputationStreamId(stream_id));
      ASSIGN_OR_RETURN(
          ThunkSequence async_thunks,
          LowerBlockToThunkSequence(async_start_op.getBodyRegion().front(),
                                    value_to_slice, token_to_async_exec,
                                    make_thunk_info));
      Thunk::ThunkInfo start_info = make_thunk_info();
      auto async_exec = std::make_shared<AsyncExecution>(start_info);
      token_to_async_exec[async_start_op.getToken()] = async_exec;
      thunks.push_back(std::make_unique<AsyncStartThunk>(
          start_info, exec_stream_id, std::move(async_thunks), async_exec));
    } else if (auto async_done_op =
                   mlir::dyn_cast<mlir::thunky::AsyncDoneOp>(op)) {
      std::shared_ptr<AsyncExecution> async_exec =
          token_to_async_exec.at(async_done_op.getToken());
      thunks.push_back(
          std::make_unique<AsyncDoneThunk>(make_thunk_info(), async_exec));
    } else if (auto all_reduce_op =
                   mlir::dyn_cast<mlir::thunky::AllReduceOp>(op)) {
      ASSIGN_OR_RETURN(CollectiveConfig config,
                       ParseCollectiveConfig(all_reduce_op.getElementType(),
                                             all_reduce_op.getReplicaGroups(),
                                             all_reduce_op.getGroupMode()));
      ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                       ParseReductionKind(all_reduce_op.getReductionKind()));
      BufferAllocation::Slice src_slice =
          value_to_slice.at(all_reduce_op.getSrc());
      BufferAllocation::Slice dst_slice =
          value_to_slice.at(all_reduce_op.getDst());
      std::vector<CollectiveThunk::Buffer> buffers = MakeCollectiveBuffers(
          src_slice, dst_slice, config.operand_element_type[0]);
      thunks.push_back(std::make_unique<AllReduceThunk>(
          make_thunk_info(), AllReduceConfig{config, reduction_kind},
          std::move(buffers)));
    } else if (auto all_gather_op =
                   mlir::dyn_cast<mlir::thunky::AllGatherOp>(op)) {
      ASSIGN_OR_RETURN(CollectiveConfig config,
                       ParseCollectiveConfig(all_gather_op.getElementType(),
                                             all_gather_op.getReplicaGroups(),
                                             all_gather_op.getGroupMode()));
      BufferAllocation::Slice src_slice =
          value_to_slice.at(all_gather_op.getSrc());
      BufferAllocation::Slice dst_slice =
          value_to_slice.at(all_gather_op.getDst());
      std::vector<CollectiveThunk::Buffer> buffers = MakeCollectiveBuffers(
          src_slice, dst_slice, config.operand_element_type[0]);
      thunks.push_back(std::make_unique<AllGatherThunk>(
          make_thunk_info(), AllGatherConfig{config, /*enable_gxl=*/false},
          std::move(buffers)));
    } else if (auto reduce_scatter_op =
                   mlir::dyn_cast<mlir::thunky::ReduceScatterOp>(op)) {
      ASSIGN_OR_RETURN(
          CollectiveConfig config,
          ParseCollectiveConfig(reduce_scatter_op.getElementType(),
                                reduce_scatter_op.getReplicaGroups(),
                                reduce_scatter_op.getGroupMode()));
      ASSIGN_OR_RETURN(
          ReductionKind reduction_kind,
          ParseReductionKind(reduce_scatter_op.getReductionKind()));
      BufferAllocation::Slice src_slice =
          value_to_slice.at(reduce_scatter_op.getSrc());
      BufferAllocation::Slice dst_slice =
          value_to_slice.at(reduce_scatter_op.getDst());
      std::vector<CollectiveThunk::Buffer> buffers = MakeCollectiveBuffers(
          src_slice, dst_slice, config.operand_element_type[0]);
      thunks.push_back(std::make_unique<ReduceScatterThunk>(
          make_thunk_info(), AllReduceConfig{config, reduction_kind},
          std::move(buffers)));
    } else if (auto all_to_all_op =
                   mlir::dyn_cast<mlir::thunky::AllToAllOp>(op)) {
      ASSIGN_OR_RETURN(CollectiveConfig config,
                       ParseCollectiveConfig(all_to_all_op.getElementType(),
                                             all_to_all_op.getReplicaGroups(),
                                             all_to_all_op.getGroupMode()));
      auto src_vals = all_to_all_op.getSrc();
      auto dst_vals = all_to_all_op.getDst();
      if (src_vals.size() != dst_vals.size() || src_vals.empty()) {
        return absl::InvalidArgumentError(
            "all_to_all requires equal non-empty src and dst buffer counts");
      }
      int64_t num_ranks = config.replica_groups.empty()
                              ? 0
                              : config.replica_groups[0].replica_ids_size();
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
      std::vector<CollectiveThunk::Buffer> buffers;
      buffers.reserve(src_vals.size());
      PrimitiveType elem_type = config.operand_element_type[0];
      for (size_t i = 0; i < src_vals.size(); ++i) {
        BufferAllocation::Slice src_slice = value_to_slice.at(src_vals[i]);
        BufferAllocation::Slice dst_slice = value_to_slice.at(dst_vals[i]);
        auto pair = MakeCollectiveBuffers(src_slice, dst_slice, elem_type);
        buffers.push_back(pair[0]);
      }
      config.operand_element_type.assign(buffers.size(), elem_type);
      thunks.push_back(std::make_unique<AllToAllThunk>(
          make_thunk_info(),
          AllToAllConfig{config, all_to_all_op.getHasSplitDimension()},
          std::move(buffers), /*p2p_memcpy_enabled=*/false));
    } else if (auto collective_permute_op =
                   mlir::dyn_cast<mlir::thunky::CollectivePermuteOp>(op)) {
      PrimitiveType elem_type = xla::ConvertMlirTypeToPrimitiveType(
          collective_permute_op.getElementType());
      if (elem_type == xla::PRIMITIVE_TYPE_INVALID) {
        return absl::InvalidArgumentError(
            "Unsupported collective_permute element type in MLIR");
      }
      ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                       ParseGroupMode(collective_permute_op.getGroupMode()));
      P2PConfig p2p_config;
      p2p_config.config.operand_element_type = {elem_type};
      p2p_config.config.group_mode = group_mode;
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
        p2p_config.config.replica_groups = ParseReplicaGroups(*rg_attr);
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
      BufferAllocation::Slice src_slice =
          value_to_slice.at(collective_permute_op.getSrc());
      BufferAllocation::Slice dst_slice =
          value_to_slice.at(collective_permute_op.getDst());
      std::vector<CollectiveThunk::Buffer> buffers =
          MakeCollectiveBuffers(src_slice, dst_slice, elem_type);
      thunks.push_back(std::make_unique<CollectivePermuteThunk>(
          make_thunk_info(), p2p_config, buffers,
          DebugOptions::COLLECTIVES_PRIVATE_MEMORY,
          /*connected_components_enabled=*/false));
    } else if (auto group_op =
                   mlir::dyn_cast<mlir::thunky::CollectiveGroupOp>(op)) {
      ASSIGN_OR_RETURN(ThunkSequence group_thunks,
                       LowerBlockToThunkSequence(
                           group_op.getBodyRegion().front(), value_to_slice,
                           token_to_async_exec, make_thunk_info));
      thunks.push_back(std::make_unique<CollectiveGroupThunk>(
          make_thunk_info(), Thunk::kGroup, std::move(group_thunks)));
    }
  }

  return thunks;
}

absl::StatusOr<ThunkSequence> LowerThunkyModuleToThunkSequence(
    mlir::ModuleOp module,
    absl::Span<const BufferAllocation::Slice> input_slices,
    absl::Span<const BufferAllocation::Slice> scratch_slices,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info) {
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

  mlir::Block& entry = main_func.getBody().front();
  llvm::DenseMap<mlir::Value, BufferAllocation::Slice> value_to_slice;
  for (size_t i = 0; i < input_slices.size(); ++i) {
    value_to_slice[entry.getArgument(i)] = input_slices[i];
  }
  for (size_t j = 0; j < scratch_slices.size(); ++j) {
    value_to_slice[entry.getArgument(input_slices.size() + j)] =
        scratch_slices[j];
  }

  main_func.walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::thunky::SliceBufferOp slice_op) {
        BufferAllocation::Slice parent =
            value_to_slice.at(slice_op.getBuffer());
        int64_t offset = slice_op.getOffset();
        int64_t size =
            mlir::cast<mlir::thunky::BufferType>(slice_op.getResult().getType())
                .getSize();
        value_to_slice[slice_op.getResult()] = BufferAllocation::Slice(
            parent.allocation(), parent.offset() + offset, size);
      });

  llvm::DenseMap<mlir::Value, std::shared_ptr<AsyncExecution>>
      token_to_async_exec;
  return LowerBlockToThunkSequence(entry, value_to_slice, token_to_async_exec,
                                   make_thunk_info);
}
}  // namespace xla::gpu
