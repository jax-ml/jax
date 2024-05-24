#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/TargetSelect.h"
#include "mlir/include/mlir/Conversion/Passes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/include/mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/include/mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/include/mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/include/mlir/ExecutionEngine/OptUtils.h"
#include "mlir/include/mlir/IR/AsmState.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/Parser/Parser.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/include/mlir/Transforms/Passes.h"
#include "jaxlib/mosaic/gpu/launch_lowering.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"

namespace {

using MosaicInitFunc = void(void***);
using MosaicHostFunc = void(void**);

std::pair<absl::flat_hash_map<uintptr_t, void*>*, absl::Mutex*>
GetContextCache() {
  static absl::Mutex mutex;
  static auto& context_cache = *new absl::flat_hash_map<uintptr_t, void*>;
  return std::make_pair(&context_cache, &mutex);
}

void InvalidateCache(MosaicInitFunc* init) {
  auto cache = GetContextCache();
  absl::MutexLock lock(cache.second);
  // TODO(apaszke): Free all the resources instead of leaking.
  cache.first->erase(reinterpret_cast<uintptr_t>(init));
}

// Each compiled kernel has a unique init func, and each kernel is used from
// a single HLO module. So it should be safe to not include the CUDA context
// in the key.
void* InitOnce(MosaicInitFunc* init) {
  auto cache_and_mutex = GetContextCache();
  auto* cache = cache_and_mutex.first;
  auto* mutex = cache_and_mutex.second;

  uintptr_t key = reinterpret_cast<uintptr_t>(init);

  {
    // Fast path uses reader lock (as hash map look-up is relatively slow).
    absl::ReaderMutexLock lock(mutex);
    auto it = cache->find(key);
    if (ABSL_PREDICT_TRUE(it != cache->end())) return it->second;
  }

  absl::MutexLock lock(mutex);
  void*& ctx = (*cache)[key];
  // We released the reader lock, another thread might have initialized it.
  if (ctx == nullptr) {
    void** ptr_to_ctx = &ctx;
    init(&ptr_to_ctx);
  }
  return ctx;
}

void InitContext(mlir::MLIRContext* context) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerCanonicalizer();
  mlir::registerCSE();
  mlir::registerStripDebugInfo();
  mlir::registerConvertNVGPUToNVVMPass();
  mlir::registerConvertVectorToSCF();
  mlir::registerSCFToControlFlow();
  mlir::registerConvertNVVMToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerConvertGpuOpsToNVVMOps();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertAffineToStandard();
  mlir::registerReconcileUnrealizedCasts();
  mlir::registerGpuToLLVMConversionPass();
  // TODO(apaszke): Only register the passes we actually use.
  mlir::memref::registerMemRefPasses();
  mlir::registerGPUPasses();
  mosaic::gpu::registerGpuLaunchLoweringPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                  mlir::gpu::GPUDialect, mlir::nvgpu::NVGPUDialect,
                  mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect>();
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

class CompiledKernel {
 public:
  CompiledKernel(std::unique_ptr<mlir::ExecutionEngine> engine,
                 std::unique_ptr<llvm::TargetMachine> target_machine,
                 MosaicInitFunc* init, MosaicHostFunc* host_launch)
      : engine_(std::move(engine)),
        target_machine_(std::move(target_machine)),
        init_(init),
        host_launch_(host_launch) {}

  std::unique_ptr<mlir::ExecutionEngine> engine_;
  std::unique_ptr<llvm::TargetMachine> target_machine_;
  MosaicInitFunc* init_;
  MosaicHostFunc* host_launch_;
};

absl::StatusOr<CompiledKernel> Compile(mlir::ModuleOp module) {
  auto pipeline = mlir::parsePassPipeline(
      R"(
      builtin.module(
        convert-nvgpu-to-nvvm,
        gpu-kernel-outlining{data-layout-str=},
        convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1},
        convert-scf-to-cf,
        convert-nvvm-to-llvm,
        expand-strided-metadata,
        nvvm-attach-target{O=3 chip=sm_90a fast=false features=+ptx80 ftz=false  module= triple=nvptx64-nvidia-cuda},
        lower-affine,
        convert-arith-to-llvm{index-bitwidth=0},
        convert-index-to-llvm{index-bitwidth=64},
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
        cse,
        gpu.module(strip-debuginfo),
        gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false}),
        gpu.module(canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),
        gpu.module(cse),
        gpu.module(reconcile-unrealized-casts),
        gpu-to-llvm{gpu-binary-annotation=gpu.binary use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false},
        gpu-module-to-binary{format=fatbin},
        convert-math-to-llvm{approximate-log1p=true},
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
        cse,
        reconcile-unrealized-casts,
        gpu-launch-lowering,
        convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}
      )
  )");
  if (mlir::failed(pipeline)) {
    return absl::InternalError("Failed to construct pass pipeline");
  }
  mlir::PassManager pm(module.getContext());
  *static_cast<mlir::OpPassManager*>(&pm) = std::move(*pipeline);
  if (mlir::failed(pm.run(module))) {
    return absl::InternalError("Pass pipeline failed");
  }

  llvm::SmallVector<llvm::StringRef> runtime_lib;
  if (const char* lib_path = getenv("MOSAIC_GPU_RUNTIME_LIB_PATH")) {
    runtime_lib.emplace_back(lib_path);
  }
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) {
    return absl::InternalError("Failed to detect host");
  }
  auto tm = builder->createTargetMachine();
  if (!tm) {
    return absl::InternalError("Failed to construct LLVM TargetMachine");
  }

  // Create a transformer to run all LLVM optimization passes at the
  // specified optimization level.
  mlir::ExecutionEngineOptions options;
  options.transformer = mlir::makeOptimizingTransformer(3, 0, tm->get());
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  options.sharedLibPaths = runtime_lib;
  module->dump();
  auto maybe_execution_engine = mlir::ExecutionEngine::create(module, options);
  if (!maybe_execution_engine) {
    return absl::InternalError("Failed to compile kernel");
  }
  std::unique_ptr<mlir::ExecutionEngine> execution_engine =
      std::move(*maybe_execution_engine);
  auto main = execution_engine->lookupPacked("_mlir_ciface_main");
  auto init = execution_engine->lookupPacked("_mlir_ciface_main_init");
  if (!init || !main) {
    return absl::InternalError("Failed to retrieve kernel function");
  }
  return CompiledKernel(std::move(execution_engine),
                        std::move(*tm),
                        reinterpret_cast<MosaicInitFunc*>(*init),
                        reinterpret_cast<MosaicHostFunc*>(*main));
}

void MosaicGPUCustomCall(void* stream, void** buffers, char* opaque,
                         size_t opaque_len, XlaCustomCallStatus* status) {
  std::string error_message = [&]() -> std::string {
    mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
    InitContext(&context);
    mlir::ParserConfig parse_config(&context);
    auto module_op =
        mlir::parseSourceString<mlir::ModuleOp>(opaque, parse_config);
    if (!module_op) {
      return "Failed to parse module";
    }
    auto kernel = Compile(*module_op);
    if (!kernel.ok()) {
      return std::string(kernel.status().message().begin(),
                        kernel.status().message().end());
    }
    auto mptr = kernel->engine_->lookupPacked("_mlir_ciface_main_init");
    if (!mptr) {
      return "No more init?";
    }
    void* ctx = InitOnce(kernel->init_);
    void* args[3] = {&ctx, &stream, &buffers};
    kernel->host_launch_(args);
    return "";
  }();
  if (!error_message.empty()) {
    XlaCustomCallStatusSetFailure(status, error_message.c_str(),
                                  error_message.length());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("mosaic_gpu", &MosaicGPUCustomCall,
                                         "CUDA");

}  // namespace
