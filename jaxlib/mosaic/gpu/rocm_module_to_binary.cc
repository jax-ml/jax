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

#include "jaxlib/mosaic/gpu/rocm_module_to_binary.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "jaxlib/mosaic/pass_boilerplate.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "xla/debug_options_flags.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace mosaic {
namespace gpu {
namespace {

using ::llvm::LogicalResult;
using ::llvm::SmallVector;
using ::mlir::Attribute;
using ::mlir::gpu::GPUModuleOp;

class ModuleToBinary : public mlir::LLVM::ModuleToObject {
 public:
  ModuleToBinary(gpu::GPUModuleOp gpu_module,
                 ::mlir::ROCDL::ROCDLTargetAttr target,
                 const std::string &gcn_arch_name)
      : ModuleToObject(*gpu_module, target.getTriple(), target.getChip(),
                       target.getFeatures(), target.getO()),
        gcn_arch_name_{gcn_arch_name} {};

  llvm::FailureOr<SmallVector<char, 0>> moduleToObject(
      llvm::Module &llvm_module) override {
    stream_executor::GpuComputeCapability cc{
        stream_executor::RocmComputeCapability(gcn_arch_name_)};

    std::vector<uint8_t> hsaco{};
    auto ret = xla::gpu::amdgpu::CompileToHsaco(
        &llvm_module, cc, xla::GetDebugOptionsFromFlags(),
        /*compilation_cache_key*/ ""  // HSACO cache is in-memory only
        // and already uses the content of the module, so since DebugOptions are
        // stable through the process lifetime, we could use empty cache key.
    );
    if (!ret.ok()) {
      return getOperation().emitError()
             << "Failed compiling the module to HSACO.";
    }
    hsaco = std::move(ret.value());

    return SmallVector<char, 0>(hsaco.begin(), hsaco.end());
  }

 private:
  // `this` isn't expected to outlive the reference.
  const std::string &gcn_arch_name_;
};

LogicalResult LowerGpuModuleToBinary(GPUModuleOp gpu_module,
                                     const std::string &gcn_arch_name) {
  mlir::gpu::OffloadingLLVMTranslationAttrInterface handler(nullptr);
  mlir::OpBuilder builder(gpu_module->getContext());
  SmallVector<Attribute> objects;

  // Fail if there are no target attributes
  if (gpu_module.getTargetsAttr().size() != 1) {
    return gpu_module.emitError(
               "Expected exactly one target attribute, but got ")
           << gpu_module.getTargetsAttr().size();
  }

  auto target_attr = llvm::dyn_cast<mlir::ROCDL::ROCDLTargetAttr>(
      gpu_module.getTargetsAttr()[0]);
  if (!target_attr) {
    return gpu_module.emitError(
        "Target attribute is not of type ROCDLTargetAttr");
  }

  ModuleToBinary serializer(gpu_module, target_attr,
                            gcn_arch_name /*, libraries_to_link*/);

  std::optional<SmallVector<char, 0>> binary = serializer.run();
  if (!binary) {
    gpu_module.emitError("An error happened while serializing the module.");
    return mlir::failure();
  }

  SmallVector<mlir::NamedAttribute> properties{
      builder.getNamedAttr("O", builder.getI32IntegerAttr(target_attr.getO()))};

  Attribute object = builder.getAttr<mlir::gpu::ObjectAttr>(
      target_attr, mlir::gpu::CompilationTarget::Binary,
      builder.getStringAttr(llvm::StringRef(binary->data(), binary->size())),
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

class RocmModuleToBinaryPass
    : public jaxlib::mlir::Pass<RocmModuleToBinaryPass, mlir::ModuleOp> {
  using BaseClass = jaxlib::mlir::Pass<RocmModuleToBinaryPass, mlir::ModuleOp>;

 public:
  RocmModuleToBinaryPass() = default;

  // TODO(Arech) for G review: CUDA has a weird implementation of a copy
  // constructor of respective GpuModuleToAssemblyPass: it's just {}, skipping
  // calling the base class c/constructor and not copying libraries_to_link_.
  // libraries_to_link_ is however problematic since one of its base classes
  // prohibits copying, so the fix here isn't obvious. So how a correct copy of
  // this class is expected to occur with such an implemnentation? This class is
  // copied by clonePass() in pass_boilerplate.h. Looks like a code design bug.
  RocmModuleToBinaryPass(const RocmModuleToBinaryPass &o) {};

  static constexpr llvm::StringLiteral kArgumentName =
      "mosaic-rocm-module-to-binary";
  static constexpr llvm::StringLiteral kPassName = "RocmModuleToBinaryPass";

  void runOnOperation() override {
    assert(gcn_arch_name_.hasValue());

    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::gpu::GPUModuleOp gpu_module) {
      if (mlir::failed(LowerGpuModuleToBinary(
              gpu_module,
              gcn_arch_name_.getValue() /*, libraries_to_link_*/))) {
        gpu_module.emitError("Failed to lower GPU module to binary.");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
  }

 private:
  mlir::Pass::Option<std::string> gcn_arch_name_{
      *this, "gcn-arch-name",
      llvm::cl::desc(
          "The GCN architecture name to compile for. Must be in a format "
          "suitable for stream_executor::RocmComputeCapability")};
};

}  // namespace

void registerRocmModuleToBinaryPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<RocmModuleToBinaryPass>();
  });
}

}  // namespace gpu
}  // namespace mosaic
