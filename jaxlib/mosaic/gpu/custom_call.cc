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
#include <cstddef>
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
#include <utility>
#include <vector>

#include "jaxlib/mosaic/gpu/library_paths.h"
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
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
#include "third_party/gpus/cuda/include/driver_types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_metadata_thunk.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
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
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/traceme.h"

namespace {

using ::mosaic::gpu::NvshmemApi;

namespace ffi = xla::ffi;
namespace se = stream_executor;

using MosaicInitFunc = void(void****);
using MosaicHostFunc = void(void**);

// Mirrors `--xla_gpu_cuda_data_dir`'s default value.
constexpr std::string_view kDefaultCudaDataDir = "./cuda_sdk_lib";

// Returns the latest PTX ISA version supported by both LLVM and the underlying
// PTX compiler.
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
    const std::string& ptx_isa, const std::string& nvshmem_path,
    bool verify_target) {
  static absl::once_flag register_passes_flag;
  absl::call_once(register_passes_flag, [&compilation_provider, &cc]() {
    mosaic::gpu::EnsureLLVMNVPTXTargetIsRegistered();
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
    mosaic::gpu::registerGpuSinkMemRefDescriptorsPass();
    mlir::arith::registerArithExpandOpsPass();
    mlir::LLVM::registerDIScopeForLLVMFuncOpPass();
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
          mosaic-gpu-sink-memref-descriptors,
          convert-nvgpu-to-nvvm,
          gpu-kernel-outlining{data-layout-str=},
          convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1},
          convert-scf-to-cf,
          convert-nvvm-to-llvm,
          expand-strided-metadata,
          nvvm-attach-target{O=3 chip=%1$s fast=false features=+%2$s ftz=false  module= triple=nvptx64-nvidia-cuda verify-target-arch=%4$v},
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
          mosaic-convert-gpu-to-llvm,
          gpu.module(reconcile-unrealized-casts),
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
                      sm, ptx_isa, absl::StrJoin(libraries_to_link, ","), verify_target));
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

std::string CUDAErrorString(CUresult result) {
  const char* error;
  cuGetErrorString(result, &error);
  return error;
}
// Returns if the CUDA expression returns an error.
#define CUDA_RETURN_IF_ERROR(stmt)                         \
  do {                                                     \
    if (CUresult result = stmt; result != CUDA_SUCCESS) {  \
      return absl::InternalError(CUDAErrorString(result)); \
    }                                                      \
  } while (0)

absl::StatusOr<se::CudaComputeCapability> GetCudaComputeCapability() {
  // Assumes driver has been initialized and a context exists. XLA already has
  // some utilities to query this, but we try to stay runtime-agnostic, so we
  // build our own here.
  CUdevice device;
  CUDA_RETURN_IF_ERROR(cuCtxGetDevice(&device));
  int major = 0;
  CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  int minor = 0;
  CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  TF_ASSIGN_OR_RETURN(std::string sm, mosaic::gpu::GetSmVersion(major, minor));
  bool has_accelerated_features = absl::EndsWith(sm, "a");

  using FeatureExtension = se::CudaComputeCapability::FeatureExtension;
  return se::CudaComputeCapability(major, minor,
                                   has_accelerated_features
                                       ? FeatureExtension::kAcceleratedFeatures
                                       : FeatureExtension::kNone);
}

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

struct CompiledKernel {
  CompiledKernel(std::unique_ptr<mlir::ExecutionEngine> engine,
                 MosaicHostFunc* host_launch, MosaicInitFunc* init,
                 bool is_comm_used)
      : engine(std::move(engine)),
        host_launch(host_launch),
        init(init),
        is_comm_used(is_comm_used) {}

  // CompiledKernel is neither copyable nor movable. We use CompiledKernel* as a
  // key in a cache, so we require pointer stability.
  CompiledKernel(const CompiledKernel&) = delete;
  CompiledKernel(CompiledKernel&& other) = delete;

  std::unique_ptr<mlir::ExecutionEngine> engine;
  MosaicHostFunc* host_launch = nullptr;
  MosaicInitFunc* init = nullptr;
  bool is_comm_used = false;
};

// TODO(b/464203195): Require a compute capability to be passed in.
absl::StatusOr<std::unique_ptr<CompiledKernel>> Compile(
    llvm::StringRef module_str, std::optional<se::CudaComputeCapability> cc) {
  tsl::profiler::TraceMe trace("Compile");
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(true);
  InitContext(&context);
  mlir::ParserConfig parse_config(&context);
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(module_str, parse_config);
  if (!module) {
    return absl::InternalError("Failed to parse Mosaic GPU module");
  }
  auto manager = mlir::PassManager::on<mlir::ModuleOp>(module->getContext());
  manager.addPass(mosaic::gpu::createSerdePass(
      mosaic::gpu::SerdePassOptions{.serialize = false}));
  if (manager.run(module.get()).failed()) {
    return absl::InternalError("Failed to deserialize Mosaic GPU module");
  }

  mosaic::gpu::EnsureLLVMNVPTXTargetIsRegistered();
  TF_ASSIGN_OR_RETURN(se::cuda::CompilationProvider * compilation_provider,
                      GetAssemblyToBinaryCompilationProvider());
  if (!cc.has_value()) {
    TF_ASSIGN_OR_RETURN(cc, GetCudaComputeCapability());
  }
  TF_ASSIGN_OR_RETURN(std::string sm,
                      mosaic::gpu::GetSmVersion(cc->major, cc->minor));
  // Here, it is important to use a PTX ISA version that is supported by both
  // the underlying compilation provider, and by LLVM. Using a version that is
  // newer than what LLVM supports will lead to the indication being ignored by
  // LLVM (potentially causing a version downgrade), while using a version that
  // is newer than what the compilation provider supports will lead to LLVM
  // potentially generating PTX that the compilation provider cannot handle.
  TF_ASSIGN_OR_RETURN(std::string llvm_ptx_isa,
                      GetPtxIsaVersion(*compilation_provider));
  bool is_comm_used = is_nvshmem_used(*module);
  std::string nvshmem_path = "";
  if (is_comm_used) {
    TF_ASSIGN_OR_RETURN(nvshmem_path, get_nvshmem_llvm_lib_path());
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
  // nvbug/5809460: spurious LLVM/MLIR errors with tcgen05+sm_103a; disable
  // verification on sm_103a, sm_110a etc. where we see spurious failures.
  bool verify_target = !((cc->major == 10 && cc->minor > 0) || cc->major == 11);
  // Use `div.full` for float32 division---this generates better SASS.
  const std::vector<std::string> llvm_cl_options{"-nvptx-prec-divf32=1"};
  // Acquire a lock over the LLVM command line options here. XLA uses this
  // lock to override the default LLVM command line options on a per-client
  // basis. This means that failing to acquire this lock and explicitly
  // setting our own command line options makes compilation dependent on
  // outside state/non-deterministic.
  xla::llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_cl_options);
  auto passes = GetPassPipeline(module->getContext(), compilation_provider, *cc,
                                sm, llvm_ptx_isa, nvshmem_path, verify_target);
  if (mlir::failed(passes)) {
    return absl::InternalError("Failed to construct pass pipeline");
  }
  mosaic::gpu::DumpOptions dump_opts =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module);
  if (RunPasses(std::move(*passes), *module, dump_opts).failed()) {
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
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> engine =
      mlir::ExecutionEngine::create(*module, options);
#ifndef NDEBUG
  if (llvm_debug_only || dump_llvm) {
    llvm::DebugFlag = old_debug_state;
  }
#endif
  if (!engine) {
    return absl::InternalError("Failed to compile kernel");
  }
  TF_ASSIGN_OR_RETURN(auto host_and_init_func_names,
                      GetHostAndInitFuncNames(*module));
  auto host = (*engine)->lookupPacked(host_and_init_func_names.first);
  auto init = (*engine)->lookupPacked(host_and_init_func_names.second);
  if (!init || !host) {
    return absl::InternalError("Failed to retrieve kernel function");
  }
  VLOG(5) << "Successfully compiled Mosaic GPU kernel";
  return std::make_unique<CompiledKernel>(
      std::move(*engine), reinterpret_cast<MosaicHostFunc*>(*host),
      reinterpret_cast<MosaicInitFunc*>(*init), is_comm_used);
}

using KernelHash = std::array<uint64_t, 4>;

absl::StatusOr<CompiledKernel*> CachedCompile(
    const KernelHash& kernel_hash, llvm::StringRef module,
    std::optional<se::CudaComputeCapability> cc) {
  struct Cache {
    absl::Mutex mutex;
    absl::flat_hash_map<KernelHash, std::unique_ptr<CompiledKernel>> kernels
        ABSL_GUARDED_BY(mutex);
  };
  static absl::NoDestructor<Cache> cache;

  absl::MutexLock lock(cache->mutex);
  auto it = cache->kernels.find(kernel_hash);
  if (it != cache->kernels.end()) return it->second.get();
  TF_ASSIGN_OR_RETURN(auto kernel, Compile(module, std::move(cc)));
  auto [iter, inserted] =
      cache->kernels.insert_or_assign(kernel_hash, std::move(kernel));
  return iter->second.get();
}

absl::StatusOr<void*> InitKernel(const CompiledKernel& kernel) {
  if (kernel.is_comm_used &&
      !NvshmemApi::Default(/*assert_ok=*/false).is_loaded()) {
    return absl::InternalError(
        "Failed to load the NVSHMEM library. Make sure it is installed (e.g. "
        "`pip install nvidia-nvshmem-cu12`).");
  }
  void* module_ptr = nullptr;
  void* kernel_ptr = nullptr;
  void** module_ptr_ptr = &module_ptr;
  void** kernel_ptr_ptr = &kernel_ptr;
  void*** init_args[2] = {&module_ptr_ptr, &kernel_ptr_ptr};
  kernel.init(init_args);
  VLOG(5) << "Successfully initialized Mosaic GPU kernel";
  return kernel_ptr;
}

// Initializes the kernel in the current CUDA context and return a handle to the
// kernel.
absl::StatusOr<void*> CachedInit(const CompiledKernel* absl_nonnull kernel) {
  using CacheKey = std::pair<const CompiledKernel*, uintptr_t>;
  struct Cache {
    absl::Mutex mutex;
    absl::flat_hash_map<CacheKey, void*> contexts ABSL_GUARDED_BY(mutex);
  };
  static absl::NoDestructor<Cache> cache;

  CUcontext ctx;
  CUDA_RETURN_IF_ERROR(cuCtxGetCurrent(&ctx));
  CacheKey key(kernel, reinterpret_cast<uintptr_t>(ctx));

  absl::MutexLock lock(cache->mutex);
  auto it = cache->contexts.find(key);
  if (it != cache->contexts.end()) return it->second;
  TF_ASSIGN_OR_RETURN(void* context, InitKernel(*kernel));
  cache->contexts.insert_or_assign(key, context);
  return context;
}

absl::Status LegacyCustomCall(cudaStream_t stream, void** buffers,
                              const KernelHash& hash, llvm::StringRef module) {
  TF_ASSIGN_OR_RETURN(auto* kernel,
                      CachedCompile(hash, module, /*cc=*/std::nullopt));
  TF_ASSIGN_OR_RETURN(auto ctx, CachedInit(kernel));
  if (kernel->is_comm_used) {
    NvshmemApi::Default().barrier_all_on_stream(stream);
  }
  void* args[4] = {&ctx, &stream, &buffers};
  kernel->host_launch(args);
  return absl::OkStatus();
}

// TODO(b/464203195): Backward-compatible version using the legacy FFI
// API. Remove once backward compatibility window has passed.
void MosaicGPUCustomCall(void* stream, void** buffers, char* opaque,
                         size_t opaque_len, XlaCustomCallStatus* cc_status) {
  KernelHash hash;
  std::memcpy(hash.data(), opaque, sizeof(KernelHash));
  auto status = LegacyCustomCall(reinterpret_cast<cudaStream_t>(stream),
                                 buffers, hash, opaque + sizeof(KernelHash));
  if (!status.ok()) {
    XlaCustomCallStatusSetFailure(cc_status, status.message().data(),
                                  status.message().size());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("mosaic_gpu", &MosaicGPUCustomCall,
                                         "CUDA");

struct CustomCallResources {
  //  TODO(allanrenucci): Remove explicit constructor after supporting C++20.
  explicit CustomCallResources(CompiledKernel* kernel) : kernel(kernel) {}
  CompiledKernel* kernel = nullptr;

  absl::Mutex mutex;
  // A map from device ID to the CPU version of the collective metadata for TMA
  // initialization.
  absl::flat_hash_map<int, std::vector<char>> cpu_metadata_buffers
      ABSL_GUARDED_BY(mutex);
};

// Validate custom call attributes and compile the kernel.
absl::StatusOr<std::unique_ptr<CustomCallResources>> InstantiateResources(
    const se::GpuComputeCapability* cc, ffi::Dictionary attrs) {
  TF_ASSIGN_OR_RETURN(bool use_custom_barrier,
                      attrs.get<bool>("use_custom_barrier"));
  TF_ASSIGN_OR_RETURN(std::string_view kernel_hash,
                      attrs.get<std::string_view>("kernel_hash"));
  TF_ASSIGN_OR_RETURN(std::string_view module,
                      attrs.get<std::string_view>("module"));
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
  TF_ASSIGN_OR_RETURN(
      CompiledKernel * kernel,
      CachedCompile(hash, module, *cc->cuda_compute_capability()));
  return std::make_unique<CustomCallResources>(kernel);
}

absl::StatusOr<std::vector<ffi::AnyBuffer>> GetBuffers(
    ffi::RemainingArgs inputs, ffi::RemainingRets results) {
  std::vector<ffi::AnyBuffer> buffers;
  buffers.reserve(inputs.size() + results.size());
  for (int i = 0; i < inputs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(auto input_buffer, inputs.get<ffi::AnyBuffer>(i));
    if (reinterpret_cast<uintptr_t>(input_buffer.untyped_data()) %
        mosaic::gpu::kExpectedHbmAlignment) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input buffer %d is not %d-byte aligned", i,
                          mosaic::gpu::kExpectedHbmAlignment));
    }
    buffers.push_back(input_buffer);
  }

  std::vector<ffi::AnyBuffer> result_buffers;
  result_buffers.reserve(results.size());
  for (int i = 0; i < results.size(); ++i) {
    TF_ASSIGN_OR_RETURN(auto result_buffer, results.get<ffi::AnyBuffer>(i));
    if (reinterpret_cast<uintptr_t>(result_buffer->untyped_data()) %
        mosaic::gpu::kExpectedHbmAlignment) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Output buffer %d is not %d-byte aligned", i,
                          mosaic::gpu::kExpectedHbmAlignment));
    }
    buffers.push_back(*result_buffer);
  }
  return buffers;
}

bool ModuleUsesCollectiveMetadata(const xla::ffi::Dictionary& attrs) {
  return attrs.get<bool>("uses_xla_collective_metadata").value_or(false);
}

// TODO(b/477478816): Support replica groups to execute a mosaic kernel on a
// subset of devices.
absl::StatusOr<xla::gpu::GpuCliqueKey> GetCliqueKeyForAllDevices(
    const xla::gpu::CollectiveParams& collective_params) {
  CHECK(collective_params.global_device_id_map != nullptr);

  int num_devices = collective_params.global_device_id_map->size();
  xla::ReplicaGroup group;
  group.mutable_replica_ids()->Reserve(num_devices);
  for (int64_t i = 0; i < num_devices; ++i) {
    group.add_replica_ids(i);
  }

  return GetGpuCliqueKey(
      collective_params, {group},
      xla::CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
      xla::AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE);
}

// Creates a collective metadata, stores the device version in a provided
// location and returns the pointer to the CPU version of the metadata.
absl::StatusOr<std::vector<char>> CreateCollectiveMetadata(
    const xla::gpu::GpuCliqueKey& clique_key, xla::RankId current_rank,
    se::Stream* stream, se::DeviceAddressBase collective_metadata_ptr,
    std::vector<se::DeviceAddressBase>&& parameters) {
  TF_ASSIGN_OR_RETURN(
      std::vector<void*> param_to_peers,
      xla::gpu::CollectiveMetadataThunk::CollectParamToPeers(
          clique_key, current_rank, stream, std::move(parameters)));
  TF_ASSIGN_OR_RETURN(
      auto metadata,
      xla::gpu::CollectiveMetadataThunk::CreateCollectiveMetadata(
          clique_key, current_rank, stream,
          // TODO(patrios): Add multimem support.
          /*multimem=*/nullptr));
  TF_RETURN_IF_ERROR(
      xla::gpu::CollectiveMetadataThunk::CopyCollectiveMetadataToDevice(
          stream, metadata, param_to_peers, collective_metadata_ptr));

  // Metadata contains the current rank, two unused during the CPU lowering
  // pointers and the param_to_peers which is used by TMA.
  constexpr size_t metadata_size = 3 * sizeof(uint64_t);
  std::vector<char> cpu_metadata_buffer(metadata_size +
                                        param_to_peers.size() * sizeof(void*));

  std::memcpy(cpu_metadata_buffer.data(), &current_rank, sizeof(uint64_t));
  std::memcpy(cpu_metadata_buffer.data() + metadata_size, param_to_peers.data(),
              param_to_peers.size() * sizeof(void*));
  return cpu_metadata_buffer;
}

absl::Status MosaicGpuInitialize(
    se::Stream* stream, const xla::gpu::CollectiveParams* collective_params,
    const xla::gpu::CollectiveCliques* collective_cliques,
    ffi::RemainingArgs inputs, ffi::RemainingRets results,
    CustomCallResources* resources, xla::ffi::Dictionary attributes) {
  bool uses_collective_metadata = ModuleUsesCollectiveMetadata(attributes);
  if (!uses_collective_metadata) {
    // If the kernel does not use collective metadata, we can skip the
    // initialization.
    return absl::OkStatus();
  }

  // When several devices handled with a single process we also need to
  // construct the collective metadata and store it to the last buffer.
  TF_ASSIGN_OR_RETURN(std::vector<ffi::AnyBuffer> buffers,
                      GetBuffers(inputs, results));

  std::vector<se::DeviceAddressBase> parameters;
  // Reserve space for input and output buffers, except the
  // collective metadata buffer.
  parameters.reserve(buffers.size() - 1);

  // Add all result buffers except the collective metadata buffer.
  for (int i = 0; i < buffers.size() - 1; ++i) {
    xla::ffi::AnyBuffer input_buffer = buffers[i];
    parameters.push_back(input_buffer.device_memory());
  }

  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCliqueKey clique_key,
                      GetCliqueKeyForAllDevices(*collective_params));

  // The last output buffer is used as the collective metadata buffer.
  se::DeviceAddressBase collective_metadata_ptr =
      buffers.back().device_memory();

  auto current_rank =
      clique_key.rank(collective_params->global_device_id).value();
  TF_ASSIGN_OR_RETURN(
      std::vector<char> cpu_metadata_buffer,
      CreateCollectiveMetadata(clique_key, current_rank, stream,
                               collective_metadata_ptr, std::move(parameters)));

  absl::MutexLock lock(resources->mutex);
  resources->cpu_metadata_buffers[collective_params->global_device_id.value()] =
      std::move(cpu_metadata_buffer);
  return absl::OkStatus();
}

absl::Status MosaicGpuExecute(
    cudaStream_t stream, const xla::gpu::CollectiveParams* collective_params,
    ffi::RemainingArgs inputs, ffi::RemainingRets results,
    CustomCallResources* resources, xla::ffi::Dictionary attributes) {
  std::vector<void*> buffer_ptrs;
  TF_ASSIGN_OR_RETURN(std::vector<ffi::AnyBuffer> buffers,
                      GetBuffers(inputs, results));
  bool uses_collective_metadata = ModuleUsesCollectiveMetadata(attributes);
  buffer_ptrs.reserve(buffers.size() + (uses_collective_metadata ? 1 : 0));
  for (const xla::ffi::AnyBuffer& buffer : buffers) {
    buffer_ptrs.push_back(buffer.untyped_data());
  }

  // Adding a CPU version of the collective metadata for TMA initialization.
  if (uses_collective_metadata) {
    absl::MutexLock lock(resources->mutex);

    std::vector<char>& cpu_metadata_buffer = resources->cpu_metadata_buffers.at(
        collective_params->global_device_id.value());
    buffer_ptrs.push_back(cpu_metadata_buffer.data());
  }

  CompiledKernel* kernel = resources->kernel;
  TF_ASSIGN_OR_RETURN(auto ctx, CachedInit(kernel));
  if (kernel->is_comm_used) {
    NvshmemApi::Default().barrier_all_on_stream(stream);
  }
  void** buffers_data = buffer_ptrs.data();
  void* args[4] = {&ctx, &stream, &buffers_data};
  kernel->host_launch(args);
  return absl::OkStatus();
}

absl::Status MosaicGpuPrepare(
    const xla::gpu::CollectiveParams* absl_nullable collective_params,
    xla::gpu::CollectiveCliqueRequests* absl_nullable clique_requests,
    xla::ffi::Dictionary attributes) {
  if (!ModuleUsesCollectiveMetadata(attributes)) {
    return absl::OkStatus();
  }

  CHECK(collective_params != nullptr);
  CHECK(clique_requests != nullptr);

  // TODO(b/476264413): Add multimem support.
  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCliqueKey clique_key,
                      GetCliqueKeyForAllDevices(*collective_params));

  TF_RETURN_IF_ERROR(clique_requests->RequestClique(clique_key));
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kMosaicGpuPrepare, MosaicGpuPrepare,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliqueRequests>()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(
    kMosaicGPUInstantiate, InstantiateResources,
    ffi::Ffi::BindInstantiate().Ctx<ffi::TargetGpuComputeCapability>().Attrs());

XLA_FFI_DEFINE_HANDLER(kMosaicGpuInitialize, MosaicGpuInitialize,
                       ffi::Ffi::BindInitialize()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliques>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Ctx<xla::ffi::State<CustomCallResources>>()
                           .Attrs(),
                       {ffi::Traits::kCmdBufferCompatible});

//  We expect the following attributes:
// - kernel_hash: a hash of the kernel.
// - module: the serialized MLIR module.
// - use_custom_barrier
// - uses_xla_collective_metadata (optional)
XLA_FFI_DEFINE_HANDLER(kMosaicGpuExecute, MosaicGpuExecute,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::CollectiveParams>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Ctx<xla::ffi::State<CustomCallResources>>()
                           .Attrs(),
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "mosaic_gpu_v2", "CUDA",
                         {
                             /*instantiate=*/kMosaicGPUInstantiate,
                             /*prepare=*/kMosaicGpuPrepare,
                             /*initialize=*/kMosaicGpuInitialize,
                             /*execute=*/kMosaicGpuExecute,
                         });

}  // namespace

extern "C" {

__attribute__((visibility("default"))) void** MosaicGpuCompile(
    const char* module, int num_module_bytes) {
  std::string module_str(module, num_module_bytes);
  auto kernel = Compile(module_str, /*cc=*/std::nullopt);
  if (!kernel.ok()) {
    return nullptr;
  }
  auto ctx = InitKernel(**kernel);
  if (!ctx.ok()) {
    return nullptr;
  }
  auto tuple_ptr = new void*[3];
  tuple_ptr[0] = *ctx;
  tuple_ptr[1] = reinterpret_cast<void*>((*kernel)->host_launch);
  tuple_ptr[2] = (*kernel).release();
  return tuple_ptr;
}

__attribute__((visibility("default"))) void MosaicGpuUnload(void** tuple_ptr) {
  delete reinterpret_cast<CompiledKernel*>(tuple_ptr[2]);
  delete[] tuple_ptr;
}

}  // extern "C"
