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

// This pass lowers existing PTX into a `gpu.binary` op using stream executor
// compilation providers. The stock MLIR pipeline uses `ptxas` in a subprocess
// to compile PTX by default. This does not work reliably in all environments,
// and stream executor's compilation providers are meant to remedy this problem.

#include "jaxlib/mosaic/gpu/assembly_to_binary.h"

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "xla/stream_executor/cuda/assemble_compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"

namespace mosaic {
namespace gpu {

namespace {

namespace se = ::stream_executor;

class AssemblyToBinaryPass : public ::mlir::OperationPass<mlir::ModuleOp> {
 public:
  AssemblyToBinaryPass()
      : ::mlir::OperationPass<mlir::ModuleOp>(
            ::mlir::TypeID::get<AssemblyToBinaryPass>()) {}
  AssemblyToBinaryPass(const AssemblyToBinaryPass &other)
      : ::mlir::OperationPass<mlir::ModuleOp>(other) {}
  AssemblyToBinaryPass &operator=(const AssemblyToBinaryPass &) = delete;
  AssemblyToBinaryPass(AssemblyToBinaryPass &&) = delete;
  AssemblyToBinaryPass &operator=(AssemblyToBinaryPass &&) = delete;
  ~AssemblyToBinaryPass() = default;

  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("gpu-assembly-to-binary");
  }
  ::llvm::StringRef getArgument() const override { return getArgumentName(); }
  ::llvm::StringRef getDescription() const override { return ""; }
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("GpuAssemblyToBinaryPass");
  }
  ::llvm::StringRef getName() const override { return getPassName(); }
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<AssemblyToBinaryPass>();
  }
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<AssemblyToBinaryPass>(
        *static_cast<const AssemblyToBinaryPass *>(this));
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssemblyToBinaryPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder b(module);
    mlir::MLIRContext *ctx = module.getContext();

    auto cc_or = GetCudaComputeCapability();
    if (!cc_or.ok()) {
      module.emitError(cc_or.status().message());
      return signalPassFailure();
    }
    se::CudaComputeCapability cc = *cc_or;

    auto compilation_provider_or = GetAssemblyToBinaryCompilationProvider();
    if (!compilation_provider_or.ok()) {
      module.emitError(compilation_provider_or.status().message());
      return signalPassFailure();
    }
    std::unique_ptr<se::cuda::CompilationProvider> compilation_provider =
        std::move(*compilation_provider_or);

    module.walk([&](mlir::gpu::BinaryOp binary) {
      if (binary.getObjects().size() != 1) {
        binary.emitOpError("Expected exactly one object in the binary.");
        return mlir::WalkResult::interrupt();
      }

      mlir::gpu::ObjectAttr object =
          mlir::cast<mlir::gpu::ObjectAttr>(*binary.getObjects().begin());
      if (object.getFormat() != mlir::gpu::CompilationTarget::Assembly) {
        binary.emitOpError("Expected an assembly object.");
        return mlir::WalkResult::interrupt();
      }

      llvm::StringRef assembly_str = object.getObject().getValue();
      absl::StatusOr<se::cuda::Assembly> sass_or =
          compilation_provider->Compile(cc, assembly_str, /*options=*/{});
      if (!sass_or.ok()) {
        binary.emitOpError(sass_or.status().message());
        return mlir::WalkResult::interrupt();
      }
      mlir::StringAttr sass = mlir::StringAttr::get(
          ctx, std::string(sass_or->cubin.begin(), sass_or->cubin.end()));
      b.setInsertionPointAfter(binary);

      mlir::gpu::ObjectAttr new_object = mlir::gpu::ObjectAttr::get(
          ctx, object.getTarget(), mlir::gpu::CompilationTarget::Binary, sass,
          object.getProperties(), object.getKernels());
      binary.setObjectsAttr(mlir::ArrayAttr::get(ctx, {new_object}));

      return mlir::WalkResult::advance();
    });
  }
};

}  // namespace

absl::StatusOr<se::CudaComputeCapability> GetCudaComputeCapability() {
  // Assumes driver has been initialized and a context exists. XLA already has
  // some utilities to query this, but we try to stay runtime-agnostic, so we
  // build our own here.
  CUdevice device;
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
    return absl::InternalError("Failed to get device for current context");
  }
  int major = 0;
  if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           device) != CUDA_SUCCESS) {
    return absl::InternalError("Failed to get major compute capability");
  }
  int minor = 0;
  if (cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           device) != CUDA_SUCCESS) {
    return absl::InternalError("Failed to get minor compute capability");
  }
  // TODO(hebecker): update CudaComputeCapability to embed extensions.
  // Currently, extensions will still be used (but are hardcoded in a util
  // `ShouldUsePtxExtension` instead of being queried).
  return se::CudaComputeCapability(major, minor);
}

absl::StatusOr<std::unique_ptr<se::cuda::CompilationProvider>>
GetAssemblyToBinaryCompilationProvider() {
  // Defaults mostly mirror those used in `xla/debug_options_flags.cc`.
  std::string default_cuda_data_dir = "./cuda_sdk_lib";
  // TODO(bchetioui): this does not mirror the XLA default. Evaluate whether
  // using NvJitLink would work as necessary.
  constexpr se::cuda::CompilationProviderOptions::NvJitLinkMode nvjitlink_mode =
      se::cuda::CompilationProviderOptions::NvJitLinkMode::kDisabled;
  constexpr bool enable_llvm_module_compilation_parallelism = false;
  constexpr bool enable_driver_compilation = false;
  bool enable_libnvptxcompiler = se::IsLibNvPtxCompilerSupported();

  se::cuda::CompilationProviderOptions opts(
      nvjitlink_mode, enable_libnvptxcompiler,
      enable_llvm_module_compilation_parallelism, enable_driver_compilation,
      std::move(default_cuda_data_dir));
  return se::cuda::AssembleCompilationProvider(opts);
}

void registerAssemblyToBinaryPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<AssemblyToBinaryPass>();
  });
}

}  // namespace gpu
}  // namespace mosaic
