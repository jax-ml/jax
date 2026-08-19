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

// The stock MLIR pipeline lowers gpu.launch_func into a sequence of
// instructions that load the kernel onto the GPU, run it and immediately unload
// it again. This has the correct semantics, but loading the kernel is both
// expensive and forces synchronization, which causes performance issues.
//
// This pass implements an alternative strategy:
// 1. It lowers gpu.launch_func into a call to mosaic_gpu_launch_kernel, which
//    takes a preloaded kernel function handle (CUfunction) passed as an extra
//    first argument to the enclosing host function.
// 2. It extracts the embedded GPU binary (gpu.binary) and kernel configuration
//    metadata (name, dynamic shared memory size, and cluster size) and attaches
//    them as attributes on the root mlir::ModuleOp before erasing gpu.binary.
//
// At compile time, the XLA custom call handler (custom_call.cc) retrieves
// these module attributes and serializes them. At runtime initialization, it
// preloads the GPU module and kernel function directly via CUDA Driver APIs
// and passes the resulting kernel handle to the host function during execution.

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "jaxlib/mosaic/pass_boilerplate.h"

namespace mosaic {
namespace gpu {

namespace {

mlir::Value packKernelArgs(mlir::OpBuilder& builder,
                           mlir::gpu::LaunchFuncOp launch) {
  std::vector<mlir::Type> kernel_operand_types;
  kernel_operand_types.reserve(launch.getNumKernelOperands());
  for (mlir::Value operand : launch.getKernelOperands()) {
    kernel_operand_types.push_back(operand.getType());
  }
  auto kernel_args_struct_ty = mlir::LLVM::LLVMStructType::getLiteral(
      builder.getContext(), kernel_operand_types);
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value c1 = mlir::LLVM::ConstantOp::create(builder, launch.getLoc(),
                                                  builder.getI32Type(),
                                                  builder.getI32IntegerAttr(1));
  mlir::Value kernel_args_struct = mlir::LLVM::AllocaOp::create(
      builder, launch.getLoc(), ptr_ty, kernel_args_struct_ty, c1);
  mlir::Value kernel_args_array = mlir::LLVM::AllocaOp::create(
      builder, launch.getLoc(), ptr_ty,
      mlir::LLVM::LLVMArrayType::get(builder.getI64Type(),
                                     launch.getNumKernelOperands()),
      c1);

  for (auto [i, operand] : llvm::enumerate(launch.getKernelOperands())) {
    mlir::Value storage_ptr = mlir::LLVM::GEPOp::create(
        builder, launch.getLoc(), ptr_ty, kernel_args_struct_ty,
        kernel_args_struct,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                           mlir::LLVM::GEPArg(i)});
    mlir::LLVM::StoreOp::create(builder, launch.getLoc(), operand, storage_ptr);
    mlir::LLVM::GEPArg arr_gep_arg(i);
    mlir::Value array_slot_ptr = mlir::LLVM::GEPOp::create(
        builder, launch.getLoc(), ptr_ty, builder.getI64Type(),
        kernel_args_array, mlir::LLVM::GEPArg(i));
    mlir::LLVM::StoreOp::create(builder, launch.getLoc(), storage_ptr,
                                array_slot_ptr);
  }
  return kernel_args_array;
}

// Emits an llvm array of i32s, initialized to the byte size of each kernel
// operand, 0 for device pointers and byte size for host arguments.
// This assumes that pointer arguments are device pointers and all else is
// host argument to be passed by value.
mlir::Value emitArgBytes(mlir::OpBuilder& builder,
                         mlir::gpu::LaunchFuncOp launch,
                         const mlir::DataLayout& data_layout) {
  const auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  const auto i32 = builder.getI32Type();
  const mlir::Value c1 = mlir::LLVM::ConstantOp::create(
      builder, launch.getLoc(), i32, builder.getI32IntegerAttr(1));
  const auto arr_type =
      mlir::LLVM::LLVMArrayType::get(i32, launch.getNumKernelOperands());
  mlir::Value arr = mlir::LLVM::AllocaOp::create(builder, launch.getLoc(),
                                                 ptr_type, arr_type, c1);
  for (auto [i, operand] : llvm::enumerate(launch.getKernelOperands())) {
    mlir::Type type = operand.getType();
    int32_t bytes = mlir::isa<mlir::LLVM::LLVMPointerType>(type)
                        ? 0
                        : static_cast<int32_t>(
                              data_layout.getTypeSize(type).getFixedValue());
    mlir::Value bytes_val = mlir::LLVM::ConstantOp::create(
        builder, launch.getLoc(), i32, builder.getI32IntegerAttr(bytes));
    mlir::Value slot = mlir::LLVM::GEPOp::create(
        builder, launch.getLoc(), ptr_type, i32, arr, mlir::LLVM::GEPArg(i));
    mlir::LLVM::StoreOp::create(builder, launch.getLoc(), bytes_val, slot);
  }
  return arr;
}

void emitRuntimeDecls(mlir::ModuleOp module) {
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(module.getContext());
  auto i32 = mlir::IntegerType::get(module.getContext(), 32);
  auto decl_builder = mlir::OpBuilder::atBlockBegin(module.getBody());

  mlir::func::FuncOp::create(
      decl_builder, module.getLoc(),
      decl_builder.getStringAttr("mosaic_gpu_build_kernel_spec"),
      mlir::FunctionType::get(module.getContext(),
                              {
                                  ptr_ty,         // MosaicKernelSpec*
                                  i32, i32, i32,  // grid
                                  i32, i32, i32,  // cluster
                                  i32, i32, i32,  // block
                                  i32,            // smem_bytes
                                  i32,            // uses_pdl
                                  i32,            // num_args
                                  ptr_ty,         // arg_ptrs
                                  ptr_ty          // arg_bytes
                              },
                              {}),
      decl_builder.getStringAttr("private"), /*arg_attr=*/nullptr,
      /*res_attrs=*/nullptr);
}

int32_t getClusterSize(mlir::gpu::LaunchFuncOp launch) {
  if (!launch.hasClusterSize()) {
    return 1;
  }
  mlir::gpu::KernelDim3 cluster_shape = launch.getClusterSizeOperandValues();
  assert(cluster_shape.x && cluster_shape.y && cluster_shape.z);
  auto x = mlir::getConstantIntValue(cluster_shape.x);
  auto y = mlir::getConstantIntValue(cluster_shape.y);
  auto z = mlir::getConstantIntValue(cluster_shape.z);
  if (x && y && z) {
    return *x * *y * *z;
  }
  return -1;
}

int32_t getDynamicSmemSize(mlir::gpu::LaunchFuncOp launch) {
  mlir::Value size = launch.getDynamicSharedMemorySize();
  if (!size) {
    return 0;
  }
  if (auto const_smem = mlir::getConstantIntValue(size)) {
    return *const_smem;
  }
  return 0;
}

// Result for prepare launch inputs below.
struct LaunchInputs {
  mlir::gpu::KernelDim3 grid, block, cluster;
  mlir::Value dynamic_smem;
  mlir::Value arg_ptr_array;
  mlir::Value uses_pdl_val;
};

LaunchInputs prepareLaunchInputs(mlir::OpBuilder& builder,
                                 mlir::gpu::LaunchFuncOp launch,
                                 bool uses_pdl) {
  LaunchInputs launch_inputs;
  launch_inputs.dynamic_smem = launch.getDynamicSharedMemorySize();
  if (!launch_inputs.dynamic_smem) {
    launch_inputs.dynamic_smem = mlir::LLVM::ConstantOp::create(
        builder, launch.getLoc(), builder.getI32Type(),
        builder.getI32IntegerAttr(0));
  }
  launch_inputs.arg_ptr_array = packKernelArgs(builder, launch);
  auto as_32bit = [&](mlir::gpu::KernelDim3 dim) {
    dim.x = mlir::LLVM::TruncOp::create(builder, launch.getLoc(),
                                        builder.getI32Type(), dim.x);
    dim.y = mlir::LLVM::TruncOp::create(builder, launch.getLoc(),
                                        builder.getI32Type(), dim.y);
    dim.z = mlir::LLVM::TruncOp::create(builder, launch.getLoc(),
                                        builder.getI32Type(), dim.z);
    return dim;
  };
  launch_inputs.grid = as_32bit(launch.getGridSizeOperandValues());
  launch_inputs.block = as_32bit(launch.getBlockSizeOperandValues());
  if (launch.hasClusterSize()) {
    launch_inputs.cluster = as_32bit(launch.getClusterSizeOperandValues());
  } else {
    launch_inputs.cluster.x = launch_inputs.cluster.y =
        launch_inputs.cluster.z = mlir::LLVM::ConstantOp::create(
            builder, launch.getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(0));
  }
  launch_inputs.uses_pdl_val = mlir::LLVM::ConstantOp::create(
      builder, launch.getLoc(), builder.getI32Type(),
      builder.getI32IntegerAttr(uses_pdl ? 1 : 0));
  return launch_inputs;
}

// Emits a call to mosaic_gpu_build_kernel_spec. cfg is the MosaicKernelSpec*
// out-param.
mlir::LogicalResult emitBuildKernelSpec(mlir::gpu::LaunchFuncOp launch,
                                        mlir::Value cfg, bool uses_pdl,
                                        const mlir::DataLayout& data_layout) {
  mlir::OpBuilder builder(launch);
  LaunchInputs launch_inputs = prepareLaunchInputs(builder, launch, uses_pdl);
  mlir::Value arg_bytes = emitArgBytes(builder, launch, data_layout);
  mlir::Value num_args = mlir::LLVM::ConstantOp::create(
      builder, launch.getLoc(), builder.getI32Type(),
      builder.getI32IntegerAttr(launch.getNumKernelOperands()));
  mlir::func::CallOp::create(
      builder, launch.getLoc(), "mosaic_gpu_build_kernel_spec",
      mlir::TypeRange{},
      mlir::ValueRange{cfg, launch_inputs.grid.x, launch_inputs.grid.y,
                       launch_inputs.grid.z, launch_inputs.cluster.x,
                       launch_inputs.cluster.y, launch_inputs.cluster.z,
                       launch_inputs.block.x, launch_inputs.block.y,
                       launch_inputs.block.z, launch_inputs.dynamic_smem,
                       launch_inputs.uses_pdl_val, num_args,
                       launch_inputs.arg_ptr_array, arg_bytes});
  return mlir::success();
}

class GpuLaunchLoweringPass
    : public jaxlib::mlir::Pass<GpuLaunchLoweringPass, mlir::ModuleOp> {
 public:
  using jaxlib::mlir::Pass<GpuLaunchLoweringPass, mlir::ModuleOp>::Pass;

  static constexpr ::llvm::StringLiteral kArgumentName = "gpu-launch-lowering";
  static constexpr ::llvm::StringLiteral kPassName = "GpuLaunchLoweringPass";

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    emitRuntimeDecls(module);
    bool uses_pdl = module->hasAttr("mosaic_gpu.uses_pdl");
    mlir::DataLayout data_layout(module);
    for (mlir::Operation& op : *module.getBody()) {
      if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(&op)) {
        if (func.isDeclaration() ||
            !func->getAttr(
                mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
          continue;
        }
        bool had_launch = false;
        mlir::Operation* gpu_binary = nullptr;
        auto result = getOperation()->walk([&](mlir::gpu::LaunchFuncOp launch)
                                               -> mlir::WalkResult {
          if (had_launch) {
            launch->emitOpError("Only one launch per function supported.");
            return mlir::WalkResult::interrupt();
          }
          had_launch = true;
          auto binary =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::gpu::BinaryOp>(
                  launch, launch.getKernelModuleName());
          if (!binary) {
            launch.emitError("Failed to find the gpu.binary op for ")
                << launch.getKernelModuleName();
            return mlir::WalkResult::interrupt();
          }
          gpu_binary = binary.getOperation();
          if (binary.getObjects().size() != 1) {
            binary.emitOpError("Expected exactly one object in the binary.");
            return mlir::WalkResult::interrupt();
          }
          mlir::gpu::ObjectAttr object =
              mlir::cast<mlir::gpu::ObjectAttr>(*binary.getObjects().begin());
          if (object.getFormat() != mlir::gpu::CompilationTarget::Fatbin &&
              object.getFormat() != mlir::gpu::CompilationTarget::Binary) {
            binary.emitOpError("Expected a binary or a fatbin object.");
            return mlir::WalkResult::interrupt();
          }

          int32_t smem_bytes = getDynamicSmemSize(launch);
          int32_t cluster_size = getClusterSize(launch);
          module->setAttr("mosaic_gpu.gpu_binary", object.getObject());
          module->setAttr("mosaic_gpu.kernel_name", launch.getKernelName());
          mlir::OpBuilder builder(module);
          module->setAttr("mosaic_gpu.smem_bytes",
                          builder.getI32IntegerAttr(smem_bytes));
          module->setAttr("mosaic_gpu.cluster_size",
                          builder.getI32IntegerAttr(cluster_size));

          mlir::Value kernel_spec = func.getArgument(0);
          mlir::LogicalResult lowered =
              emitBuildKernelSpec(launch, kernel_spec, uses_pdl, data_layout);
          if (lowered.failed()) {
            return mlir::WalkResult::interrupt();
          }
          launch.erase();
          // TODO(apaszke): Generate a destructor function.
          // builder.CreateCall(getModuleUnloadFn(), {moduleObject});

          return mlir::WalkResult::advance();
        });
        if (gpu_binary) {
          // This deletion is load-bearing: the conversion of `gpu.binary` to
          // LLVM is side-effecting, as it creates module constructors and
          // destructors which create an assumption that symbols from the MLIR
          // runtime are available.
          gpu_binary->erase();
        }
        if (result == mlir::WalkResult::interrupt()) {
          signalPassFailure();
        }
      }
    }
  }
};

}  // namespace

void registerGpuLaunchLoweringPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<GpuLaunchLoweringPass>();
  });
}

}  // namespace gpu
}  // namespace mosaic
