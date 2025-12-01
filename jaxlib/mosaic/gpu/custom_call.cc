/* Copyright 2021 The JAX Authors.

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

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include "jaxlib/mosaic/gpu/library_paths.h"
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu.h"
#include "jaxlib/mosaic/gpu/assembly_to_binary.h"
#include "jaxlib/mosaic/gpu/dump.h"
#include "jaxlib/mosaic/gpu/gpu_module_to_assembly.h"
#include "jaxlib/mosaic/gpu/launch_lowering.h"
#include "jaxlib/mosaic/gpu/nvshmem.h"
#include "jaxlib/mosaic/gpu/passes.h"
#include "jaxlib/mosaic/gpu/serde.h"
#include "jaxlib/mosaic/gpu/target.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/stream_executor/cuda/assemble_compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/traceme.h"

namespace {

namespace ffi = xla::ffi;
namespace se = stream_executor;

using MosaicInitFunc = void(void****);
using MosaicHostFunc = void(void**);

// Mirrors `--xla_gpu_cuda_data_dir`'s default value.
constexpr std::string_view kDefaultCudaDataDir = "./cuda_sdk_lib";

absl::StatusOr<std::string> GetPtxIsaVersion(
    const se::cuda::CompilationProvider& compilation_provider) {
  TF_ASSIGN_OR_RETURN(int ptxas_latest_version,
                      compilation_provider.GetLatestPtxIsaVersion());
  // We'd like to target the latest PTX ISA version supported by
  // ptxas. However, it doesn't make sense to ask LLVM to target a PTX
  // ISA that it isn't aware of yet. Find the latest version supported
  // by LLVM and return the minimum of the two versions, one from
  // ptxas and the other from LLVM.
  TF_ASSIGN_OR_RETURN(int llvm_latest_version,
                      mosaic::gpu::GetLatestLlvmPtxIsaVersion());
  int final_version = std::min(ptxas_latest_version, llvm_latest_version);
  return absl::StrFormat("ptx%d", final_version);
}

mlir::FailureOr<mlir::OpPassManager> GetPassPipeline(
    mlir::MLIRContext* ctx,
    const se::cuda::CompilationProvider* compilation_provider,
    const se::CudaComputeCapability& cc, const std::string& sm,
    const std::string& ptx_isa, const std::string& nvshmem_path) {
  static absl::once_flag register_passes_flag;
  absl::call_once(
      register_passes_flag, [&compilation_provider, &cc]() {
        mosaic::gpu::EnsureLLVMNVPTXTargetIsRegistered();

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        mlir::registerCanonicalizer();
        mlir::registerCSE();
        mlir::registerStripDebugInfo();
        mlir::registerConvertNVGPUToNVVMPass();
        mlir::registerConvertVectorToSCF();
        mlir::registerSCFToControlFlowPass();
        mlir::registerConvertNVVMToLLVMPass();
        mlir::registerArithToLLVMConversionPass();
        mlir::registerConvertIndexToLLVMPass();
        mlir::registerConvertGpuOpsToNVVMOps();
        mlir::registerConvertMathToLLVMPass();
        mlir::registerConvertFuncToLLVMPass();
        mlir::registerLowerAffinePass();
        mlir::registerReconcileUnrealizedCastsPass();
        // TODO(apaszke): Only register the passes we actually use.
        mlir::memref::registerMemRefPasses();
        mlir::registerConvertToLLVMPass();
        mlir::registerGPUPasses();
        mlir::registerGpuLaunchSinkIndexComputationsPass();
        mosaic::gpu::registerGpuModuleToAssemblyPass();
        mosaic::gpu::registerAssemblyToBinaryPass(compilation_provider, cc);
        mosaic::gpu::registerGpuLaunchLoweringPass();
        mosaic::gpu::registerConvertGpuToLLVMPass();
        mosaic::gpu::registerByvalInsertionPass();
        mosaic::gpu::registerLLVMAttrInsertionPass();
        mosaic::gpu::registerResolveTrivialLocationsPass();
        mlir::arith::registerArithExpandOpsPass();
        mlir::LLVM::registerDIScopeForLLVMFuncOpPass();
        return true;
      });
  const char* cuda_root = mosaic::gpu::GetCUDARoot();
  if (!cuda_root) {
    return mlir::failure();
  }
  std::vector<std::string> libraries_to_link{
      ::xla::gpu::nvptx::LibDevicePath(kDefaultCudaDataDir)};
  if (!nvshmem_path.empty()) {
    libraries_to_link.push_back(nvshmem_path);
  }
  return mlir::parsePassPipeline(
      absl::StrFormat(R"(
        builtin.module(
          mosaic-gpu-resolve-trivial-locations,
          arith-expand,
          canonicalize,
          gpu-launch-sink-index-computations,
          convert-nvgpu-to-nvvm,
          gpu-kernel-outlining{data-layout-str=},
          convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1},
          convert-scf-to-cf,
          convert-nvvm-to-llvm,
          expand-strided-metadata,
          nvvm-attach-target{O=3 chip=%1$s fast=false features=+%2$s ftz=false  module= triple=nvptx64-nvidia-cuda},
          lower-affine,
          convert-arith-to-llvm{index-bitwidth=0},
          convert-index-to-llvm{index-bitwidth=64},
          canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
          cse,
          gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false}),
          gpu.module(canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
          gpu.module(cse),
          gpu.module(mosaic-byval-insertion),
          gpu.module(mosaic-llvm-attr-insertion),
          gpu.module(reconcile-unrealized-casts),
          mosaic-convert-gpu-to-llvm,
          ensure-debug-info-scope-on-llvm-func{emission-kind=DebugDirectivesOnly},
          mosaic-gpu-module-to-assembly{libraries-to-link=%3$s},
          convert-math-to-llvm{approximate-log1p=true},
          canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
          cse,
          mosaic-gpu-assembly-to-binary,
          gpu-launch-lowering,
          convert-to-llvm,
          reconcile-unrealized-casts
        )
      )",
                      sm, ptx_isa, absl::StrJoin(libraries_to_link, ",")));
}

mlir::LogicalResult RunPasses(mlir::OpPassManager&& passes,
                              mlir::ModuleOp module,
                              const mosaic::gpu::DumpOptions& dump_opts) {
  mlir::PassManager pm(module.getContext());
  *static_cast<mlir::OpPassManager*>(&pm) = std::move(passes);
  std::optional<llvm::raw_fd_ostream> dump_stream;
  if (dump_opts.mlir_passes) {
    if (!dump_opts.dump_path.empty()) {
      std::string path = tsl::io::JoinPath(
          dump_opts.dump_path, dump_opts.module_basename + ".mlir-passes.log");
      std::error_code error;
      dump_stream.emplace(path, error, llvm::sys::fs::OF_None);
      if (error) {
        dump_stream.reset();
        LOG(ERROR) << error.message();
        LOG(ERROR) << "Output will be written to stdout instead.";
        dump_stream = std::nullopt;
      }
    }
    pm.getContext()->disableMultithreading();
    auto print_always = [](mlir::Pass*, mlir::Operation*) { return true; };
    pm.enableIRPrinting(/*shouldPrintBeforePass=*/print_always,
                        /*shouldPrintAfterPass=*/print_always,
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/true,
                        dump_stream.has_value() ? *dump_stream : llvm::outs(),
                        mlir::OpPrintingFlags().enableDebugInfo());
  }
  return pm.run(module);
}

void InitContext(mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect, mlir::gpu::GPUDialect,
                  mlir::nvgpu::NVGPUDialect, mlir::NVVM::NVVMDialect,
                  mlir::LLVM::LLVMDialect, mosaic_gpu::MosaicGPUDialect>();
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

bool is_nvshmem_used(mlir::ModuleOp module) {
  constexpr std::string_view prefix1 = "nvshmem_";
  constexpr std::string_view prefix2 = "nvshmemx_";
  for (mlir::LLVM::LLVMFuncOp llvm_func :
       module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    const auto& func_name = llvm_func.getName();
    if (!func_name.starts_with(prefix1) && !func_name.starts_with(prefix2)) {
      continue;
    }
    auto uses =
        mlir::SymbolTable::getSymbolUses(llvm_func, module.getOperation());
    if (uses && !uses->empty()) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<std::string> get_nvshmem_llvm_lib_path() {
  const char* nvshmem_path_ptr = getenv("MOSAIC_GPU_NVSHMEM_BC_PATH");
  if (!nvshmem_path_ptr)
    return absl::InternalError("Failed to get MOSAIC_GPU_NVSHMEM_BC_PATH");
  return nvshmem_path_ptr;
}

absl::StatusOr<se::cuda::CompilationProvider*>
GetAssemblyToBinaryCompilationProvider() {
  auto create_provider = []() {
    // Defaults mirror those used in `xla/debug_options_flags.cc`.
    constexpr se::cuda::CompilationProviderOptions::NvJitLinkMode
        nvjitlink_mode =
            se::cuda::CompilationProviderOptions::NvJitLinkMode::kAuto;
    constexpr bool enable_llvm_module_compilation_parallelism = false;
    constexpr bool enable_driver_compilation = false;
    bool enable_libnvptxcompiler = se::IsLibNvPtxCompilerSupported();

    se::cuda::CompilationProviderOptions opts(
        nvjitlink_mode, enable_libnvptxcompiler,
        enable_llvm_module_compilation_parallelism, enable_driver_compilation,
        std::string(kDefaultCudaDataDir));

    return absl::NoDestructor(se::cuda::AssembleCompilationProvider(opts));
  };
  static absl::NoDestructor<
      absl::StatusOr<std::unique_ptr<se::cuda::CompilationProvider>>>
      compilation_provider = create_provider();

  if (!compilation_provider->ok()) {
    return compilation_provider->status();
  }
  return (*compilation_provider)->get();
}

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

  TF_ASSIGN_OR_RETURN(std::string sm, mosaic::gpu::GetSmVersion(major, minor));
  bool has_accelerated_features = absl::EndsWith(sm, "a");

  using FeatureExtension = se::CudaComputeCapability::FeatureExtension;
  return se::CudaComputeCapability(major, minor,
                                   has_accelerated_features
                                       ? FeatureExtension::kAcceleratedFeatures
                                       : FeatureExtension::kNone);
}

absl::StatusOr<std::pair<std::unique_ptr<mlir::ExecutionEngine>, bool>> Compile(
    mlir::ModuleOp module) {
  tsl::profiler::TraceMe trace("Compile");
  mosaic::gpu::EnsureLLVMNVPTXTargetIsRegistered();
  TF_ASSIGN_OR_RETURN(se::cuda::CompilationProvider * compilation_provider,
                      GetAssemblyToBinaryCompilationProvider());
  TF_ASSIGN_OR_RETURN(se::CudaComputeCapability cc, GetCudaComputeCapability());
  TF_ASSIGN_OR_RETURN(std::string sm,
                      mosaic::gpu::GetSmVersion(cc.major, cc.minor));
  TF_ASSIGN_OR_RETURN(std::string ptx_isa,
                      GetPtxIsaVersion(*compilation_provider));
  bool is_comm_used = is_nvshmem_used(module);
  std::string nvshmem_path = "";
  if (is_comm_used) {
    TF_ASSIGN_OR_RETURN(nvshmem_path, get_nvshmem_llvm_lib_path());
    if (!mosaic::gpu::NvshmemApi::Default(/*assert_ok=*/false).is_loaded()) {
      return absl::InternalError(
          "Failed to load the NVSHMEM library. Make sure it is installed (e.g. "
          "`pip install nvidia-nvshmem-cu12`).");
    }
  }
  const char* dump_llvm = getenv("MOSAIC_GPU_DUMP_LLVM");
  const char* llvm_debug_only = getenv("MOSAIC_GPU_LLVM_DEBUG_ONLY");
#ifndef NDEBUG
  bool old_debug_state = false;
  std::vector<std::string_view> debug_only_types;
  if (llvm_debug_only) {
    debug_only_types = absl::StrSplit(llvm_debug_only, ',');
  }
  if (dump_llvm) {
    debug_only_types.push_back("serialize-to-llvm");
  }
  if (!debug_only_types.empty()) {
    old_debug_state = llvm::DebugFlag;
    std::vector<const char*> debug_only_types_ptrs;
    debug_only_types_ptrs.reserve(debug_only_types.size());
    for (std::string_view debug_only_type : debug_only_types) {
      debug_only_types_ptrs.push_back(debug_only_type.data());
    }
    llvm::setCurrentDebugTypes(debug_only_types_ptrs.data(),
                               debug_only_types_ptrs.size());
    llvm::DebugFlag = true;
  }
#else
  if (llvm_debug_only || dump_llvm) {
    fprintf(
        stderr,
        "MOSAIC_GPU_LLVM_DEBUG_ONLY or MOSAIC_GPU_DUMP_LLVM is set but LLVM "
        "was built with NDEBUG\n");
    abort();
  }
#endif
  // Use `div.full` for float32 division---this generates better SASS.
  const std::vector<std::string> llvm_cl_options{"-nvptx-prec-divf32=1"};
  // Acquire a lock over the LLVM command line options here. XLA uses this
  // lock to override the default LLVM command line options on a per-client
  // basis. This means that failing to acquire this lock and explicitly
  // setting our own command line options makes compilation dependent on
  // outside state/non-deterministic.
  xla::llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_cl_options);
  auto passes = GetPassPipeline(module.getContext(), compilation_provider, cc,
                                sm, ptx_isa, nvshmem_path);
  if (mlir::failed(passes)) {
    return absl::InternalError("Failed to construct pass pipeline");
  }
  mosaic::gpu::DumpOptions dump_opts =
      mosaic::gpu::GetOrSetDumpOptionsForModule(module);
  if (mlir::failed(RunPasses(std::move(*passes), module, dump_opts))) {
    return absl::InternalError("Pass pipeline failed");
  }
  llvm::SmallVector<llvm::StringRef> runtime_libs;
  if (const char* runtime_lib_path = getenv("MOSAIC_GPU_RUNTIME_LIB_PATH")) {
    runtime_libs.emplace_back(runtime_lib_path);
  }
  if (const char* nvshmem_path = getenv("MOSAIC_GPU_NVSHMEM_SO_PATH")) {
    runtime_libs.emplace_back(nvshmem_path);
  }
  // Create a transformer to run all LLVM optimization passes at the
  // specified optimization level.
  std::function<llvm::Error(llvm::Module*)> transformer =
      [dump_opts](llvm::Module* module) {
        if (getenv("MOSAIC_GPU_DUMP_HOST_LLVM")) {
          std::string ll_str;
          llvm::raw_string_ostream os(ll_str);
          module->print(os, nullptr);
          os.flush();
          mosaic::gpu::DumpToFileOrStdout(
              ll_str, dump_opts.module_basename + ".ll", dump_opts.dump_path);
        }
        return mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr)(module);
      };
  mlir::ExecutionEngineOptions options;
  options.transformer = transformer;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  options.sharedLibPaths = runtime_libs;
  auto maybe_execution_engine = mlir::ExecutionEngine::create(module, options);
#ifndef NDEBUG
  if (llvm_debug_only || dump_llvm) {
    llvm::DebugFlag = old_debug_state;
  }
#endif
  if (!maybe_execution_engine) {
    return absl::InternalError("Failed to compile kernel");
  }
  return std::make_pair(std::move(*maybe_execution_engine), is_comm_used);
}

class CompiledKernel {
 public:
  CompiledKernel(std::unique_ptr<mlir::ExecutionEngine> engine, void* ctx,
                 MosaicHostFunc* host_launch, bool is_comm_used)
      : engine_(std::move(engine)),
        ctx_(ctx),
        host_launch_(host_launch),
        is_comm_used_(is_comm_used) {}

  std::tuple<void*, MosaicHostFunc*, bool> GetHostLaunch() {
    return std::make_tuple(ctx_, host_launch_, is_comm_used_);
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  void* ctx_;  // TODO(apaszke): Destroy this properly
  MosaicHostFunc* host_launch_;
  bool is_comm_used_;
};

absl::StatusOr<std::pair<std::string, std::string>> GetHostAndInitFuncNames(
    mlir::ModuleOp module_op) {
  // We look for two top level C-interface functions:
  // - "host" function with symbol name "_mlir_ciface_<foo>"
  // - "init" function with symbol name "_mlir_ciface_<foo>_init"
  constexpr std::string_view prefix = "_mlir_ciface_";
  std::vector<std::string> names;
  for (mlir::LLVM::LLVMFuncOp llvm_func :
       module_op.getOps<mlir::LLVM::LLVMFuncOp>()) {
    if (llvm_func.getName().starts_with(prefix)) {
      names.push_back(llvm_func.getName().str());
    }
  }
  if (auto size = names.size(); size != 2) {
    return absl::InternalError(absl::StrFormat(
        "Expected to locate 2 symbols with %s prefix in the MLIR module, found "
        "%d instead.",
        prefix, size));
  }
  // _mlir_ciface_<foo>_init now follows _mlir_ciface_<foo>
  std::sort(names.begin(), names.end());

  std::string host_func_name = names[0];
  std::string init_func_name = names[1];

  if (init_func_name != absl::StrCat(host_func_name, "_init")) {
    return absl::InternalError(absl::StrFormat(
        "Expected init function name to equal the concatenation of the host "
        "function name and the \"_init\" suffix, instead got "
        "init_func_name=%s, host_func_name=%s.",
        init_func_name, host_func_name));
  }
  return std::make_pair(host_func_name, init_func_name);
}

absl::StatusOr<CompiledKernel> CompileAndInit(llvm::StringRef module) {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(true);
  InitContext(&context);
  mlir::ParserConfig parse_config(&context);
  auto module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module, parse_config);
  if (!module_op) {
    return absl::InternalError("Failed to parse Mosaic GPU module");
  }
  auto manager = mlir::PassManager::on<mlir::ModuleOp>(module_op->getContext());
  manager.addPass(mosaic::gpu::createSerdePass(
      mosaic::gpu::SerdePassOptions{.serialize = false}));
  if (manager.run(module_op.get()).failed()) {
    return absl::InternalError("Failed to deserialize Mosaic GPU module");
  }
  auto maybe_engine = Compile(*module_op);
  if (!maybe_engine.ok()) {
    return maybe_engine.status();
  }
  mlir::ExecutionEngine* execution_engine = maybe_engine.value().first.get();
  bool is_comm_used = maybe_engine.value().second;

  auto host_and_init_func_names = GetHostAndInitFuncNames(*module_op);
  if (!host_and_init_func_names.ok()) {
    return host_and_init_func_names.status();
  }
  auto [host_name, init_name] = host_and_init_func_names.value();

  auto host = execution_engine->lookupPacked(host_name);
  auto init = execution_engine->lookupPacked(init_name);
  if (!init || !host) {
    return absl::InternalError("Failed to retrieve kernel function");
  }
  void* module_ptr = nullptr;
  void* kernel_ptr = nullptr;
  void** module_ptr_ptr = &module_ptr;
  void** kernel_ptr_ptr = &kernel_ptr;
  void*** init_args[2] = {&module_ptr_ptr, &kernel_ptr_ptr};
  reinterpret_cast<MosaicInitFunc*>(*init)(init_args);
  return CompiledKernel(std::move(maybe_engine.value().first), kernel_ptr,
                        reinterpret_cast<MosaicHostFunc*>(*host), is_comm_used);
}

using KernelHash = std::array<uint64_t, 4>;
using CacheKey = std::pair<KernelHash, uintptr_t>;

struct KernelCache {
  static KernelCache& Global() {
    static absl::NoDestructor<KernelCache> cache;
    return *cache;
  }

  KernelCache() = default;
  // KernelCache is neither copyable nor movable.
  KernelCache(const KernelCache&) = delete;
  KernelCache(KernelCache&&) = delete;

  absl::Mutex mutex;
  absl::flat_hash_map<CacheKey, CompiledKernel> kernels ABSL_GUARDED_BY(mutex);
};

// Each compiled kernel has a unique init func, and each kernel is used from
// a single HLO module. So it should be safe to not include the CUDA context
// in the key.
absl::StatusOr<CompiledKernel*> CachedCompileAndInit(CacheKey key,
                                                     llvm::StringRef module) {
  KernelCache& cache = KernelCache::Global();

  {
    // Fast path uses reader lock (as hash map look-up is relatively slow).
    absl::ReaderMutexLock lock(cache.mutex);
    auto it = cache.kernels.find(key);
    if (ABSL_PREDICT_TRUE(it != cache.kernels.end())) return &it->second;
  }

  absl::MutexLock lock(cache.mutex);
  // We released the reader lock, another thread might have initialized it.
  if (cache.kernels.find(key) == cache.kernels.end()) {
    tsl::profiler::TraceMe trace("Compilation cache miss");
    auto compiled = CompileAndInit(module);
    if (!compiled.ok()) {
      return compiled.status();
    }
    cache.kernels.insert_or_assign(key, std::move(*compiled));
  }
  return &cache.kernels.at(key);
}

void MosaicGPUCustomCall(void* stream, void** buffers, char* opaque,
                         size_t opaque_len, XlaCustomCallStatus* status) {
  // Forward-compatible version using the legacy FFI API
  if (reinterpret_cast<uintptr_t>(opaque) % alignof(KernelHash)) {
    fprintf(stderr, "Misaligned opaque pointer\n");
    abort();
  }
  auto hash = *reinterpret_cast<KernelHash*>(opaque);
  CUcontext ctx;
  if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to get current CUDA context\n");
    abort();
  }
  CacheKey key(hash, reinterpret_cast<uintptr_t>(ctx));
  auto compiled_kernel = CachedCompileAndInit(key, opaque + sizeof(KernelHash));
  if (!compiled_kernel.ok()) {
    XlaCustomCallStatusSetFailure(status,
                                  compiled_kernel.status().message().data(),
                                  compiled_kernel.status().message().size());
    return;
  }
  auto ctx_kernel_comm = (*compiled_kernel)->GetHostLaunch();
  bool is_comm_used = std::get<2>(ctx_kernel_comm);
  void* args[4] = {&std::get<0>(ctx_kernel_comm), &stream, &buffers};
  if (is_comm_used) {
    mosaic::gpu::NvshmemApi::Default().barrier_all_on_stream(
        reinterpret_cast<cudaStream_t>(stream));
  }
  std::get<1>(ctx_kernel_comm)(args);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("mosaic_gpu", &MosaicGPUCustomCall,
                                         "CUDA");

absl::Status MosaicGpuExecute(cudaStream_t stream, ffi::RemainingArgs inputs,
                              ffi::RemainingRets results,
                              std::string_view kernel_hash,
                              std::string_view module,
                              bool use_custom_barrier) {
  // Updated version using the new FFI API supporting custom barrier
  // for distributed kernels
  if (use_custom_barrier) {
    return absl::UnimplementedError("Custom barrier is not supported on GPUs.");
  }
  if (kernel_hash.size() != sizeof(KernelHash)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Kernel hash size is %d bytes, expected %d bytes",
                        kernel_hash.size(), sizeof(KernelHash)));
  }
  KernelHash hash;
  std::memcpy(hash.data(), kernel_hash.data(), sizeof(KernelHash));
  CUcontext ctx;
  if (auto result = cuCtxGetCurrent(&ctx); result != CUDA_SUCCESS) {
    const char* error;
    cuGetErrorString(result, &error);
    return absl::InternalError(
        absl::StrFormat("Failed to get current CUDA context: %s", error));
  }
  CacheKey key(hash, reinterpret_cast<uintptr_t>(ctx));
  TF_ASSIGN_OR_RETURN(auto compiled_kernel,
                      CachedCompileAndInit(key, module));
  auto ctx_kernel_comm = compiled_kernel->GetHostLaunch();
  bool is_comm_used = std::get<2>(ctx_kernel_comm);

  std::vector<void*> buffers;
  buffers.reserve(inputs.size() + results.size());
  for (int i = 0; i < inputs.size(); ++i) {
    buffers.push_back(inputs.get<ffi::AnyBuffer>(i)->untyped_data());
    if (reinterpret_cast<uintptr_t>(buffers.back()) %
        mosaic::gpu::kExpectedHbmAlignment) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input buffer %d is not %d-byte aligned", i,
                          mosaic::gpu::kExpectedHbmAlignment));
    }
  }
  for (int i = 0; i < results.size(); ++i) {
    buffers.push_back((*results.get<ffi::AnyBuffer>(i))->untyped_data());
    if (reinterpret_cast<uintptr_t>(buffers.back()) %
        mosaic::gpu::kExpectedHbmAlignment) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Output buffer %d is not %d-byte aligned", i,
                          mosaic::gpu::kExpectedHbmAlignment));
    }
  }
  void** buffers_ptr = buffers.data();
  void* args[4] = {&std::get<0>(ctx_kernel_comm), &stream, &buffers_ptr};

  if (is_comm_used) {
    mosaic::gpu::NvshmemApi::Default().barrier_all_on_stream(stream);
  }
  std::get<1>(ctx_kernel_comm)(args);
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kMosaicGpuExecute, MosaicGpuExecute,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<std::string_view>("kernel_hash")
                           .Attr<std::string_view>("module")
                           .Attr<bool>("use_custom_barrier"),
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "mosaic_gpu_v2", "CUDA",
                         {
                             /*instantiate=*/nullptr,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kMosaicGpuExecute,
                         });

}  // namespace

extern "C" {

__attribute__((visibility("default"))) void** MosaicGpuCompile(
    const char* module, int num_module_bytes) {
  std::string module_str(module, num_module_bytes);
  auto compiled = CompileAndInit(module_str);
  if (!compiled.ok()) {
    return nullptr;
  }
  auto [ctx, launch, is_comm_used] = compiled->GetHostLaunch();
  auto tuple_ptr = std::unique_ptr<void*>(new void*[3]);
  if (!tuple_ptr) {
    return nullptr;
  }
  tuple_ptr.get()[0] = ctx;
  tuple_ptr.get()[1] = reinterpret_cast<void*>(launch);
  tuple_ptr.get()[2] = new CompiledKernel(std::move(*compiled));
  if (!tuple_ptr.get()[2]) {
    return nullptr;
  }
  return tuple_ptr.release();
}

__attribute__((visibility("default"))) void MosaicGpuUnload(void** tuple_ptr) {
  delete reinterpret_cast<CompiledKernel*>(tuple_ptr[2]);
  delete[] tuple_ptr;
}

}  // extern "C"
