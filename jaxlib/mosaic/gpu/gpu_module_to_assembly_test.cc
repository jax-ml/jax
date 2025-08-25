/* Copyright 2025 The JAX Authors.

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

#include "jaxlib/mosaic/gpu/gpu_module_to_assembly.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"

namespace {

using ::testing::HasSubstr;

class GpuModuleToAssemblyTest : public ::testing::Test {
 public:
  GpuModuleToAssemblyTest()
      : builder_(&context_),
        module_(mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::ModuleOp::create(builder_.getUnknownLoc(), "module"))) {
    RegisterErrorRecordingHandler();
    mlir::DialectRegistry registry;
    registry.insert<mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                    mlir::ROCDL::ROCDLDialect, mlir::NVVM::NVVMDialect>();
    mlir::registerGPUDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    builder_.setInsertionPointToEnd(module_->getBody());
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();

    mosaic::gpu::registerGpuModuleToAssemblyPass();
  }

  void ExpectLastErrorContains(std::string_view substring) {
    EXPECT_THAT(last_error_message_, HasSubstr(substring));
  }

 protected:
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::string last_error_message_;

 private:
  void RegisterErrorRecordingHandler() {
    // Make sure to make the context single-threaded to avoid race conditions
    // when recording the last error message.
    context_.disableMultithreading();
    mlir::DiagnosticEngine& diagnostic_engine = context_.getDiagEngine();
    diagnostic_engine.registerHandler([&](mlir::Diagnostic& diagnostic) {
      last_error_message_ = diagnostic.str();
    });
  }
};

mlir::gpu::GPUModuleOp CreateGpuModuleWithEmptyFunc(mlir::OpBuilder& b,
                                                    mlir::ArrayAttr targets) {
  mlir::gpu::GPUModuleOp gpu_module = b.create<mlir::gpu::GPUModuleOp>(
      b.getUnknownLoc(), "gpu_module", targets);
  b.setInsertionPointToEnd(gpu_module.getBody());

  mlir::LLVM::LLVMFunctionType func_ty = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(b.getContext()), {});
  mlir::LLVM::LLVMFuncOp func =
      b.create<mlir::LLVM::LLVMFuncOp>(b.getUnknownLoc(), "gpu_func", func_ty);
  b.setInsertionPointToEnd(func.addEntryBlock(b));

  b.create<mlir::LLVM::ReturnOp>(b.getUnknownLoc(), mlir::ValueRange());

  b.setInsertionPointAfter(gpu_module);
  return gpu_module;
}

template <typename T>
mlir::SmallVector<T> GetOpsOfType(mlir::ModuleOp module) {
  mlir::SmallVector<T, 2> ops;
  module.walk([&](T binary) { ops.push_back(binary); });
  return ops;
}

mlir::NVVM::NVVMTargetAttr GetNVVMTargetAttr(mlir::MLIRContext* ctx) {
  return mlir::NVVM::NVVMTargetAttr::get(
      ctx, /*optLevel=*/3, /*triple=*/"nvptx64-nvidia-cuda",
      /*chip=*/"sm_90a", /*features=*/"+ptx87");
}

TEST_F(GpuModuleToAssemblyTest, ConvertGpuModuleWithNVVMAttributeToAssembly) {
  mlir::gpu::GPUModuleOp gpu_module = CreateGpuModuleWithEmptyFunc(
      builder_,
      mlir::ArrayAttr::get(&context_, {GetNVVMTargetAttr(&context_)}));
  EXPECT_TRUE(mlir::succeeded(mosaic::gpu::internal::LowerGpuModuleToAssembly(
      gpu_module, /*libraries_to_link=*/{})));

  EXPECT_EQ(GetOpsOfType<mlir::gpu::GPUModuleOp>(*module_).size(), 0);
  mlir::SmallVector<mlir::gpu::BinaryOp> binary_ops =
      GetOpsOfType<mlir::gpu::BinaryOp>(*module_);
  ASSERT_EQ(binary_ops.size(), 1);

  auto binary = binary_ops.front();
  ASSERT_EQ(binary.getObjects().size(), 1);
  mlir::gpu::ObjectAttr object =
      mlir::cast<mlir::gpu::ObjectAttr>(*binary.getObjects().begin());
  EXPECT_EQ(object.getFormat(), mlir::gpu::CompilationTarget::Assembly);
}

TEST_F(GpuModuleToAssemblyTest,
       EncounteringAGpuModuleWithoutNVVMTargetIsAnError) {
  mlir::gpu::GPUModuleOp gpu_module = CreateGpuModuleWithEmptyFunc(
      builder_, mlir::ArrayAttr::get(
                    &context_, {mlir::ROCDL::ROCDLTargetAttr::get(&context_)}));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(gpu_module)));
  EXPECT_TRUE(mlir::failed(mosaic::gpu::internal::LowerGpuModuleToAssembly(
      gpu_module, /*libraries_to_link=*/{})));
  ExpectLastErrorContains("Target attribute is not of type NVVMTargetAttr");
}

TEST_F(GpuModuleToAssemblyTest,
       EncounteringAGpuModuleWithMultipleTargetsIsAnError) {
  mlir::Attribute target_attr = GetNVVMTargetAttr(&context_);
  mlir::gpu::GPUModuleOp gpu_module = CreateGpuModuleWithEmptyFunc(
      builder_, mlir::ArrayAttr::get(&context_, {target_attr, target_attr}));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(gpu_module)));
  EXPECT_TRUE(mlir::failed(mosaic::gpu::internal::LowerGpuModuleToAssembly(
      gpu_module, /*libraries_to_link=*/{})));
  ExpectLastErrorContains("Expected exactly one target attribute");
}

TEST_F(GpuModuleToAssemblyTest,
       LoweringGpuModuleToAssemblyLinksToLibrariesCorrectly) {
  mlir::Location loc = builder_.getUnknownLoc();
  mlir::Type f32 = builder_.getF32Type();
  mlir::gpu::GPUModuleOp gpu_module = CreateGpuModuleWithEmptyFunc(
      builder_,
      mlir::ArrayAttr::get(&context_, {GetNVVMTargetAttr(&context_)}));
  auto gpu_func = GetOpsOfType<mlir::LLVM::LLVMFuncOp>(*module_).front();

  // Insert a declaration for `__nv_exp2f` defined in libdevice here.
  builder_.setInsertionPointToStart(gpu_module.getBody());
  auto exp2f = builder_.create<mlir::LLVM::LLVMFuncOp>(
      loc, "__nv_exp2f", mlir::LLVM::LLVMFunctionType::get({f32}, f32));
  // Call the function in the entry block of `gpu_func`.
  builder_.setInsertionPointToStart(&gpu_func.getBlocks().front());
  auto constant = builder_.create<mlir::LLVM::ConstantOp>(
      loc, f32, builder_.getF32FloatAttr(1.0));
  builder_.create<mlir::LLVM::CallOp>(loc, exp2f, mlir::ValueRange{constant});

  // Clone the module so that we can check that linking libraries behaves
  // differently than not linking them.
  mlir::OwningOpRef<mlir::ModuleOp> module2 = module_->clone();

  // Without linking to the libraries, we should not be able to resolve the
  // function, and are left with an `extern .func` declaration. Here, we call
  // the pass itself in order to make sure that the pass options are propagated
  // as expected.
  mlir::PassManager pm(module_->getContext());
  auto pass_without_libdevice =
      mlir::parsePassPipeline("builtin.module(mosaic-gpu-module-to-assembly)");
  ASSERT_TRUE(mlir::succeeded(pass_without_libdevice));
  *static_cast<mlir::OpPassManager*>(&pm) = std::move(*pass_without_libdevice);
  EXPECT_TRUE(mlir::succeeded(pm.run(*module_)));
  EXPECT_THAT(mosaic_gpu::MlirToString(*module_), HasSubstr("extern .func"));

  // When linking the libraries, the `extern .func` declaration should
  // disappear.
  std::string libdevice_path =
      ::xla::gpu::nvptx::LibDevicePath("./cuda_sdk_lib");
  auto pass_with_libdevice = mlir::parsePassPipeline(absl::StrCat(
      "builtin.module(mosaic-gpu-module-to-assembly{libraries-to-link=",
      libdevice_path, "})"));
  ASSERT_TRUE(mlir::succeeded(pass_with_libdevice));
  *static_cast<mlir::OpPassManager*>(&pm) = std::move(*pass_with_libdevice);
  EXPECT_TRUE(mlir::succeeded(pm.run(*module2)));
  EXPECT_THAT(mosaic_gpu::MlirToString(*module2),
              Not(HasSubstr("extern .func")));
}

}  // anonymous namespace
