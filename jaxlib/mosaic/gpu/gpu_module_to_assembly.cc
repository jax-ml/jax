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

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "jaxlib/mosaic/pass_boilerplate.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"

namespace mosaic {
namespace gpu {

namespace {

using ::llvm::failure;
using ::llvm::FailureOr;
using ::llvm::LogicalResult;
using ::llvm::SmallVector;
using ::mlir::Attribute;
using ::mlir::gpu::GPUModuleOp;

// A replacement class for the upstream `NVPTXSerializer` and
// `SerializeGpuModuleBase` classes.
class ModuleToAssembly : public mlir::LLVM::ModuleToObject {
 public:
  ModuleToAssembly(gpu::GPUModuleOp gpu_module,
                   ::mlir::NVVM::NVVMTargetAttr target,
                   std::vector<std::string> libraries_to_link)
      : ModuleToObject(*gpu_module, target.getTriple(), target.getChip(),
                       target.getFeatures(), target.getO()),
        libraries_to_link_(std::move(libraries_to_link)) {};

  // Serializes the LLVM module to PTX.
  FailureOr<SmallVector<char, 0>> moduleToObject(
      llvm::Module& llvm_module) override;

  // Loads the bitcode files in `libraries_to_link_`.
  std::optional<SmallVector<std::unique_ptr<llvm::Module>>> loadBitcodeFiles(
      llvm::Module& module) override;

 private:
  std::vector<std::string> libraries_to_link_;
};

FailureOr<SmallVector<char, 0>> ModuleToAssembly::moduleToObject(
    llvm::Module& llvm_module) {
  // Use a debug type compatible with upstream.
#define DEBUG_TYPE "serialize-to-llvm"
  LLVM_DEBUG({ llvm::dbgs() << llvm_module; });
#undef DEBUG_TYPE
  std::optional<llvm::TargetMachine*> machine = getOrCreateTargetMachine();
  if (!machine) {
    return getOperation().emitError()
           << "Target Machine unavailable for "
              "triple "
           << triple << ", can't optimize with LLVM\n";
  }
  llvm::FailureOr<std::string> ptx = translateModuleToISA(
      llvm_module, **machine, [&]() { return getOperation().emitError(); });
  if (failed(ptx)) {
    return getOperation().emitError() << "Failed translating the module"
                                         "to PTX.";
  }

  return SmallVector<char, 0>(ptx->begin(), ptx->end());
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
ModuleToAssembly::loadBitcodeFiles(llvm::Module& llvm_module) {
  llvm::LLVMContext& ctx = llvm_module.getContext();
  llvm::SMDiagnostic err;
  SmallVector<std::unique_ptr<llvm::Module>> loaded_modules;
  loaded_modules.reserve(libraries_to_link_.size());
  for (const std::string& library_path : libraries_to_link_) {
    std::unique_ptr<llvm::Module> library_module =
        xla::gpu::LoadIRModule(library_path, &ctx);
    if (!library_module) {
      getOperation().emitError() << "Failed loading file from " << library_path
                                 << ", error: " << err.getMessage();
      return std::nullopt;
    }
    loaded_modules.push_back(std::move(library_module));
  }
  return loaded_modules;
}

}  // namespace

namespace internal {

// A simplified version of the logic implemented by `NVVMTargetAttrImpl`'s
// `serializeToObject` and `createObject` to deal better with different
// environments.
LogicalResult LowerGpuModuleToAssembly(
    GPUModuleOp gpu_module, const std::vector<std::string>& libraries_to_link) {
  EnsureLLVMNVPTXTargetIsRegistered();
  mlir::gpu::OffloadingLLVMTranslationAttrInterface handler(nullptr);
  mlir::OpBuilder builder(gpu_module->getContext());
  SmallVector<Attribute> objects;
  // Fail if there are no target attributes
  if (gpu_module.getTargetsAttr().size() != 1) {
    return gpu_module.emitError(
               "Expected exactly one target attribute, but got ")
           << gpu_module.getTargetsAttr().size();
  }

  auto target_attr = llvm::dyn_cast<mlir::NVVM::NVVMTargetAttr>(
      gpu_module.getTargetsAttr()[0]);
  if (!target_attr) {
    return gpu_module.emitError(
        "Target attribute is not of type NVVMTargetAttr");
  }

  ModuleToAssembly serializer(gpu_module, target_attr, libraries_to_link);
  std::optional<SmallVector<char, 0>> assembly = serializer.run();
  if (!assembly) {
    gpu_module.emitError("An error happened while serializing the module.");
    return mlir::failure();
  }

  SmallVector<mlir::NamedAttribute> properties{
      builder.getNamedAttr("O", builder.getI32IntegerAttr(target_attr.getO()))};

  Attribute object = builder.getAttr<mlir::gpu::ObjectAttr>(
      target_attr, mlir::gpu::CompilationTarget::Assembly,
      builder.getStringAttr(
          llvm::StringRef(assembly->data(), assembly->size())),
      builder.getDictionaryAttr(properties), /*kernels=*/nullptr);

  if (!object) {
    gpu_module.emitError("An error happened while creating the object.");
    return mlir::failure();
  }

  builder.setInsertionPointAfter(gpu_module);
  mlir::gpu::BinaryOp::create(
      builder, gpu_module.getLoc(), gpu_module.getName(), /*handler=*/nullptr,
      builder.getArrayAttr(SmallVector<Attribute>{object}));
  gpu_module->erase();
  return mlir::success();
}

}  // namespace internal

namespace {

class GpuModuleToAssemblyPass
    : public jaxlib::mlir::Pass<GpuModuleToAssemblyPass, mlir::ModuleOp> {
 public:
  using jaxlib::mlir::Pass<GpuModuleToAssemblyPass, mlir::ModuleOp>::Pass;

  GpuModuleToAssemblyPass() = default;
  GpuModuleToAssemblyPass(const GpuModuleToAssemblyPass&) {};

  static constexpr llvm::StringLiteral kArgumentName =
      "mosaic-gpu-module-to-assembly";
  static constexpr llvm::StringLiteral kPassName = "GpuModuleToAssemblyPass";

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::gpu::GPUModuleOp gpu_module) {
      if (mlir::failed(internal::LowerGpuModuleToAssembly(
              gpu_module, libraries_to_link_))) {
        gpu_module.emitError("Failed to lower GPU module to assembly.");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
  }

 private:
  ListOption<std::string> libraries_to_link_{
      *this, "libraries-to-link",
      llvm::cl::desc("A comma-separated list of bitcode files to link into the "
                     "resulting assembly.")};
};

}  // namespace

void registerGpuModuleToAssemblyPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<GpuModuleToAssemblyPass>();
  });
}

void EnsureLLVMNVPTXTargetIsRegistered() {
  static absl::once_flag register_nvptx_target_flag;
  absl::call_once(register_nvptx_target_flag, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

}  // namespace gpu
}  // namespace mosaic
