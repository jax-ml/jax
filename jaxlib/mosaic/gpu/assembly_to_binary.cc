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
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/gpu/dump.h"
#include "jaxlib/mosaic/pass_boilerplate.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace mosaic {
namespace gpu {

namespace {

namespace se = ::stream_executor;

class AssemblyToBinaryPass
    : public jaxlib::mlir::Pass<AssemblyToBinaryPass, mlir::ModuleOp> {
 public:
  using jaxlib::mlir::Pass<AssemblyToBinaryPass, mlir::ModuleOp>::Pass;

  AssemblyToBinaryPass(
      const se::cuda::CompilationProvider* compilation_provider,
      se::CudaComputeCapability cc)
      : compilation_provider_(std::move(compilation_provider)),
        cc_(std::move(cc)) {}

  static constexpr llvm::StringLiteral kArgumentName =
      "mosaic-gpu-assembly-to-binary";
  static constexpr llvm::StringLiteral kPassName = "GpuAssemblyToBinaryPass";

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder b(module);
    mlir::MLIRContext* ctx = module.getContext();
    DumpOptions dump_opts = GetOrSetDumpOptionsForModule(module);

    se::cuda::CompilationOptions compilation_options;
    compilation_options.dump_compilation_log = dump_opts.ptxas;
    compilation_options.generate_line_info = true;

    mlir::WalkResult result = module.walk([&](mlir::gpu::BinaryOp binary) {
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

      llvm::StringRef ptx_str = object.getObject().getValue();
      if (dump_opts.ptx) {
        DumpToFileOrStdout(ptx_str, dump_opts.module_basename + ".ptx",
                           dump_opts.dump_path);
      }
      absl::StatusOr<se::cuda::Assembly> sass_or =
          compilation_provider_->Compile(cc_, ptx_str, compilation_options);
      if (!sass_or.ok()) {
        binary.emitOpError(sass_or.status().message());
        return mlir::WalkResult::interrupt();
      }

      if (dump_opts.ptxas) {
        if (!sass_or->compilation_log.has_value()) {
          binary.emitOpError("Expected a compilation log to be available.");
          return mlir::WalkResult::interrupt();
        }
        DumpToFileOrStdout(*sass_or->compilation_log,
                           dump_opts.module_basename + ".ptxas",
                           dump_opts.dump_path);
      }

      mlir::StringAttr sass = mlir::StringAttr::get(
          ctx, std::string(sass_or->cubin.begin(), sass_or->cubin.end()));
      b.setInsertionPointAfter(binary);

      mlir::gpu::ObjectAttr new_object = mlir::gpu::ObjectAttr::get(
          ctx, object.getTarget(), mlir::gpu::CompilationTarget::Binary, sass,
          object.getProperties(), object.getKernels());
      binary.setObjectsAttr(mlir::ArrayAttr::get(ctx, {new_object}));

      if (dump_opts.sass || dump_opts.sass_ctrl) {
        DumpSass(binary, dump_opts.dump_path, dump_opts.module_basename,
                 dump_opts.sass_ctrl);
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

 private:
  const se::cuda::CompilationProvider* compilation_provider_;
  se::CudaComputeCapability cc_;
};

}  // namespace

void registerAssemblyToBinaryPass(
    const se::cuda::CompilationProvider* compilation_provider,
    const se::CudaComputeCapability& cc) {
  ::mlir::registerPass(
      [compilation_provider, cc]() -> std::unique_ptr<::mlir::Pass> {
        return std::make_unique<AssemblyToBinaryPass>(compilation_provider,
                                                      std::move(cc));
      });
}

}  // namespace gpu
}  // namespace mosaic
