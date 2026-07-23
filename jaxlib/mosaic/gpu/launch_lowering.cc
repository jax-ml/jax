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

// This pass implements an alternative strategy, where each function containing
// a gpu.launch_func is split into two functions: one that preloads the kernel
// onto the GPU, and second one that consumes the handle produced by the
// first one. We call the first function at compile-time, while only the
// second one is used at run-time.

// TODO(apaszke): Implement a third function that properly cleans up the
// resources allocated by the first function.

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

void emitRuntimeDecls(mlir::ModuleOp module) {
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(module.getContext());
  auto i32 = mlir::IntegerType::get(module.getContext(), 32);
  auto decl_builder = mlir::OpBuilder::atBlockBegin(module.getBody());
  mlir::func::FuncOp::create(
      decl_builder, module.getLoc(),
      decl_builder.getStringAttr("mosaic_gpu_launch_kernel"),
      mlir::FunctionType::get(module.getContext(),
                              {ptr_ty, i32, i32, i32, i32, i32, i32, i32, i32,
                               i32, i32, i32, ptr_ty, ptr_ty},
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

mlir::LogicalResult launchPreloadedKernel(mlir::func::FuncOp func,
                                          mlir::gpu::LaunchFuncOp launch,
                                          mlir::Value kernel_handle,
                                          bool uses_pdl) {
  // Lower gpu.launch_func to a call to mgpuLaunchKernel.
  mlir::OpBuilder builder(launch);
  mlir::Value dynamic_smem = launch.getDynamicSharedMemorySize();
  if (!dynamic_smem) {
    dynamic_smem = mlir::LLVM::ConstantOp::create(builder, launch.getLoc(),
                                                  builder.getI32Type(),
                                                  builder.getI32IntegerAttr(0));
  }
  mlir::Value arg_ptr_array = packKernelArgs(builder, launch);
  auto as_32bit = [&](mlir::gpu::KernelDim3 dim) {
    dim.x = mlir::LLVM::TruncOp::create(builder, launch.getLoc(),
                                        builder.getI32Type(), dim.x);
    dim.y = mlir::LLVM::TruncOp::create(builder, launch.getLoc(),
                                        builder.getI32Type(), dim.y);
    dim.z = mlir::LLVM::TruncOp::create(builder, launch.getLoc(),
                                        builder.getI32Type(), dim.z);
    return dim;
  };
  mlir::gpu::KernelDim3 grid = as_32bit(launch.getGridSizeOperandValues());
  mlir::gpu::KernelDim3 block = as_32bit(launch.getBlockSizeOperandValues());
  mlir::gpu::KernelDim3 cluster;
  if (launch.hasClusterSize()) {
    cluster = as_32bit(launch.getClusterSizeOperandValues());
  } else {
    cluster.x = cluster.y = cluster.z = mlir::LLVM::ConstantOp::create(
        builder, launch.getLoc(), builder.getI32Type(),
        builder.getI32IntegerAttr(0));
  }
  mlir::Value uses_pdl_val = mlir::LLVM::ConstantOp::create(
      builder, launch.getLoc(), builder.getI32Type(),
      builder.getI32IntegerAttr(uses_pdl ? 1 : 0));
  mlir::Value stream = launch.getAsyncObject();
  mlir::func::CallOp::create(
      builder, launch.getLoc(), "mosaic_gpu_launch_kernel", mlir::TypeRange{},
      mlir::ValueRange{kernel_handle, grid.x, grid.y, grid.z, cluster.x,
                       cluster.y, cluster.z, block.x, block.y, block.z,
                       dynamic_smem, uses_pdl_val, stream, arg_ptr_array});
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
    auto ptr_ty = mlir::LLVM::LLVMPointerType::get(module.getContext());
    emitRuntimeDecls(module);
    bool uses_pdl = module->hasAttr("mosaic_gpu.uses_pdl");
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
          module->setAttr(
              "mosaic_gpu.kernel_name",
              mlir::StringAttr::get(module.getContext(),
                                    launch.getKernelName().getValue()));
          module->setAttr(
              "mosaic_gpu.smem_bytes",
              mlir::OpBuilder(module).getI32IntegerAttr(smem_bytes));
          module->setAttr(
              "mosaic_gpu.cluster_size",
              mlir::OpBuilder(module).getI32IntegerAttr(cluster_size));

          // Add a new function argument for the kernel handle.
          if (func.insertArgument(
                  0, ptr_ty, mlir::DictionaryAttr::get(func.getContext()),
                  mlir::UnknownLoc::get(func.getContext()))
                  .failed()) {
            return mlir::WalkResult::interrupt();
          }
          mlir::Value kernel_handle = func.getArgument(0);
          if (launchPreloadedKernel(func, launch, kernel_handle, uses_pdl)
                  .failed()) {
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
