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
#include <map>
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
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
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
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu.h"
#include "jaxlib/mosaic/gpu/assembly_to_binary.h"
#include "jaxlib/mosaic/gpu/dump.h"
#include "jaxlib/mosaic/gpu/gpu_module_to_assembly.h"
#include "jaxlib/mosaic/gpu/launch_lowering.h"
#include "jaxlib/mosaic/gpu/mosaic_gpu.pb.h"
#include "jaxlib/mosaic/gpu/nvshmem.h"
#include "jaxlib/mosaic/gpu/passes.h"
#include "jaxlib/mosaic/gpu/serde.h"
#include "jaxlib/mosaic/gpu/target.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_kernel_api.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/type_registry.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/stream_executor/cuda/assemble_compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/traceme.h"

namespace {

using ::mosaic::gpu::NvshmemApi;

namespace ffi = xla::ffi;
namespace se = stream_executor;

using MosaicInitFunc = void(void**, void**);
using MosaicHostFunc = void(void*, void*, void**);
using KernelHash = std::array<uint64_t, 4>;

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

void EnsureNativeLLVMisInitialized() {
  static absl::once_flag init_flag;
  absl::call_once(init_flag, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
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
    EnsureNativeLLVMisInitialized();
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
  return mlir::parsePassPipeline(absl::StrFormat(
      R"(
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
  CompiledKernel(std::unique_ptr<llvm::orc::LLJIT> lljit,
                 MosaicHostFunc* host_launch, MosaicInitFunc* init,
                 bool is_comm_used, std::string object_file,
                 std::string host_func_name, std::string init_func_name)
      : lljit(std::move(lljit)),
        host_launch(host_launch),
        init(init),
        is_comm_used(is_comm_used),
        object_file(std::move(object_file)),
        host_func_name(std::move(host_func_name)),
        init_func_name(std::move(init_func_name)) {}

  // CompiledKernel is neither copyable nor movable. We use CompiledKernel* as a
  // key in a cache, so we require pointer stability.
  CompiledKernel(const CompiledKernel&) = delete;
  CompiledKernel(CompiledKernel&& other) = delete;

  std::unique_ptr<llvm::orc::LLJIT> lljit;
  MosaicHostFunc* host_launch = nullptr;
  MosaicInitFunc* init = nullptr;
  bool is_comm_used = false;
  // The following fields are used for de/serialization of CompiledKernel.
  std::string object_file;
  std::string host_func_name;
  std::string init_func_name;
};

absl::Status RunMlirPasses(mlir::ModuleOp module, se::CudaComputeCapability cc,
                           bool is_comm_used,
                           const mosaic::gpu::DumpOptions& dump_opts) {
  TF_ASSIGN_OR_RETURN(se::cuda::CompilationProvider * compilation_provider,
                      GetAssemblyToBinaryCompilationProvider());
  TF_ASSIGN_OR_RETURN(std::string sm,
                      mosaic::gpu::GetSmVersion(cc.major, cc.minor));
  // Here, it is important to use a PTX ISA version that is supported by both
  // the underlying compilation provider, and by LLVM. Using a version that is
  // newer than what LLVM supports will lead to the indication being ignored by
  // LLVM (potentially causing a version downgrade), while using a version that
  // is newer than what the compilation provider supports will lead to LLVM
  // potentially generating PTX that the compilation provider cannot handle.
  TF_ASSIGN_OR_RETURN(std::string llvm_ptx_isa,
                      GetPtxIsaVersion(*compilation_provider));
  std::string nvshmem_path = "";
  if (is_comm_used) {
    TF_ASSIGN_OR_RETURN(nvshmem_path, get_nvshmem_llvm_lib_path());
  }
  // nvbug/5809460: spurious LLVM/MLIR errors with tcgen05+sm_103a; disable
  // verification on sm_103a, sm_110a etc. where we see spurious failures.
  bool verify_target = !((cc.major == 10 && cc.minor > 0) || cc.major == 11);
  auto passes = GetPassPipeline(module.getContext(), compilation_provider, cc,
                                sm, llvm_ptx_isa, nvshmem_path, verify_target);
  if (mlir::failed(passes)) {
    return absl::InternalError("Failed to construct pass pipeline");
  }
  if (RunPasses(std::move(*passes), module, dump_opts).failed()) {
    return absl::InternalError("Pass pipeline failed");
  }
  return absl::OkStatus();
}

// This function was inspired by mlir::ExecutionEngine::create. It takes an MLIR
// module, compiles the module to LLVM IR, optimizes it using LLVM, and returns
// the compiled object file.
absl::StatusOr<std::unique_ptr<llvm::MemoryBuffer>> CompileModuleToObject(
    mlir::ModuleOp module, const mosaic::gpu::DumpOptions& dump_opts) {
  llvm::LLVMContext llvm_context;
  auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);
  if (!llvm_module) {
    return absl::InternalError("Failed to translate module to LLVM IR");
  }

  auto tm_builder_or_error = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tm_builder_or_error) {
    return absl::InternalError(
        absl::StrFormat("Failed to detect host: %s",
                        llvm::toString(tm_builder_or_error.takeError())));
  }
  tm_builder_or_error->setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);

  auto tm_or_error = tm_builder_or_error->createTargetMachine();
  if (!tm_or_error) {
    return absl::InternalError(
        absl::StrFormat("Failed to create target machine: %s",
                        llvm::toString(tm_or_error.takeError())));
  }
  std::unique_ptr<llvm::TargetMachine> target_machine =
      std::move(tm_or_error.get());

  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvm_module.get(),
                                                        target_machine.get());

  if (getenv("MOSAIC_GPU_DUMP_HOST_LLVM")) {
    std::string ll_str;
    llvm::raw_string_ostream os(ll_str);
    llvm_module->print(os, nullptr);
    os.flush();
    mosaic::gpu::DumpToFileOrStdout(ll_str, dump_opts.module_basename + ".ll",
                                    dump_opts.dump_path);
  }

  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/target_machine.get());
  if (auto err = transformer(llvm_module.get())) {
    return absl::InternalError(absl::StrFormat("Transformer failed: %s",
                                               llvm::toString(std::move(err))));
  }

  std::string result;
  llvm::raw_string_ostream stream(result);
  {
    llvm::buffer_ostream bstream(stream);
    llvm::legacy::PassManager codegen_pm;
    if (target_machine->addPassesToEmitFile(
            codegen_pm, bstream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
      return absl::InternalError(
          "TargetMachine can't emit a file of this type");
    }
    codegen_pm.run(*llvm_module);
  }

  return llvm::MemoryBuffer::getMemBufferCopy(result, "kernel");
}

absl::StatusOr<std::unique_ptr<CompiledKernel>> CreateAndInitJIT(
    std::unique_ptr<llvm::MemoryBuffer> object_file, std::string host_func_name,
    std::string init_func_name, bool is_comm_used) {
  EnsureNativeLLVMisInitialized();
  std::string object_file_str = object_file->getBuffer().str();
  auto lljit_builder = llvm::orc::LLJITBuilder();

  auto tm_builder_or_error = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tm_builder_or_error) {
    return absl::InternalError(
        absl::StrFormat("Failed to detect host: %s",
                        llvm::toString(tm_builder_or_error.takeError())));
  }
  char global_prefix =
      llvm::DataLayout(
          tm_builder_or_error->getTargetTriple().computeDataLayout())
          .getGlobalPrefix();

  lljit_builder.setObjectLinkingLayerCreator(
      [global_prefix](llvm::orc::ExecutionSession& session)
          -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
        auto objectLayer =
            std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
                session, [](llvm::MemoryBufferRef) {
                  return std::make_unique<llvm::SectionMemoryManager>();
                });

        llvm::SmallVector<llvm::StringRef> runtime_libs;
        if (const char* runtime_lib_path =
                getenv("MOSAIC_GPU_RUNTIME_LIB_PATH")) {
          runtime_libs.emplace_back(runtime_lib_path);
        }
        if (const char* nvshmem_path = getenv("MOSAIC_GPU_NVSHMEM_SO_PATH")) {
          runtime_libs.emplace_back(nvshmem_path);
        }

        for (const auto& lib : runtime_libs) {
          llvm::SmallString<128> abs_lib_path(lib);
          if (std::error_code ec = llvm::sys::fs::make_absolute(abs_lib_path)) {
            return llvm::make_error<llvm::StringError>(
                absl::StrFormat("Failed to get absolute path for %s: %s",
                                std::string_view(lib.data(), lib.size()),
                                ec.message()),
                llvm::inconvertibleErrorCode());
          }

          // inspired by mlir::ExecutionEngine::create
          auto mb = llvm::MemoryBuffer::getFile(abs_lib_path);
          if (!mb) {
            return llvm::make_error<llvm::StringError>(
                absl::StrFormat("Failed to create MemoryBuffer for: %s: %s",
                                abs_lib_path.c_str(), mb.getError().message()),
                llvm::inconvertibleErrorCode());
          }

          auto& jd = session.createBareJITDylib(std::string(abs_lib_path));
          auto loaded = llvm::orc::DynamicLibrarySearchGenerator::Load(
              abs_lib_path.c_str(), global_prefix);
          if (!loaded) {
            return loaded.takeError();
          }
          jd.addGenerator(std::move(*loaded));

          if (auto err = objectLayer->add(jd, std::move(mb.get()))) {
            return std::move(err);
          }
        }

        return objectLayer;
      });

  auto lljit_or_error = lljit_builder.create();
  if (auto err = lljit_or_error.takeError()) {
    return absl::InternalError(absl::StrFormat("Failed to create LLJIT: %s",
                                               llvm::toString(std::move(err))));
  }
  std::unique_ptr<llvm::orc::LLJIT> lljit = std::move(*lljit_or_error);

  // Resolve symbols that are linked in the current process. This is necessary
  // to find the host and init functions and resolve all library calls.
  {
    auto& main_jd = lljit->getMainJITDylib();
    auto generator_or_error =
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            global_prefix);
    if (auto err = generator_or_error.takeError()) {
      return absl::InternalError(absl::StrFormat(
          "Failed to create generator: %s", llvm::toString(std::move(err))));
    }
    main_jd.addGenerator(std::move(*generator_or_error));
  }

  // This is the actual linking step. Returns an error if it fails to resolve
  // all symbols from the object file.
  if (auto err = lljit->addObjectFile(std::move(object_file))) {
    return absl::InternalError(absl::StrFormat("Failed to add object file: %s",
                                               llvm::toString(std::move(err))));
  }

  // Run all the static initializers in the object file if there are any.
  if (auto err = lljit->initialize(lljit->getMainJITDylib())) {
    return absl::InternalError(absl::StrFormat("Failed to initialize LLJIT: %s",
                                               llvm::toString(std::move(err))));
  }

  auto host_sym = lljit->lookup(host_func_name);
  if (auto err = host_sym.takeError()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to lookup host symbol: %s", llvm::toString(std::move(err))));
  }

  auto init_sym = lljit->lookup(init_func_name);
  if (auto err = init_sym.takeError()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to lookup init symbol: %s", llvm::toString(std::move(err))));
  }

  VLOG(5) << "Successfully JIT-linked Mosaic GPU kernel";
  return std::make_unique<CompiledKernel>(
      std::move(lljit), host_sym->toPtr<MosaicHostFunc*>(),
      init_sym->toPtr<MosaicInitFunc*>(), is_comm_used,
      std::move(object_file_str), std::move(host_func_name),
      std::move(init_func_name));
}

absl::StatusOr<std::unique_ptr<CompiledKernel>> Compile(
    llvm::StringRef module_str, se::CudaComputeCapability cc) {
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
  mosaic::gpu::EnsureLLVMNVPTXTargetIsRegistered();

  bool is_comm_used = is_nvshmem_used(*module);
  mosaic::gpu::DumpOptions dump_opts =
      mosaic::gpu::GetOrSetDumpOptionsForModule(*module);
  TF_RETURN_IF_ERROR(RunMlirPasses(*module, cc, is_comm_used, dump_opts));

  TF_ASSIGN_OR_RETURN(auto object_file,
                      CompileModuleToObject(*module, dump_opts));
  VLOG(5) << "Successfully compiled Mosaic GPU kernel to object file";

#ifndef NDEBUG
  if (llvm_debug_only || dump_llvm) {
    llvm::DebugFlag = old_debug_state;
  }
#endif

  TF_ASSIGN_OR_RETURN(auto host_and_init_func_names,
                      GetHostAndInitFuncNames(*module));

  return CreateAndInitJIT(
      std::move(object_file), std::move(host_and_init_func_names.first),
      std::move(host_and_init_func_names.second), is_comm_used);
}

struct KernelCache {
  absl::Mutex mutex;
  absl::flat_hash_map<KernelHash, std::unique_ptr<CompiledKernel>> kernels
      ABSL_GUARDED_BY(mutex);
};

static KernelCache& GetKernelCache() {
  static absl::NoDestructor<KernelCache> cache;
  return *cache;
}

absl::StatusOr<CompiledKernel*> GetOrCreateKernel(
    const KernelHash& kernel_hash,
    absl::FunctionRef<absl::StatusOr<std::unique_ptr<CompiledKernel>>()>
        factory) {
  auto& cache = GetKernelCache();
  {
    absl::MutexLock lock(&cache.mutex);
    auto it = cache.kernels.find(kernel_hash);
    if (it != cache.kernels.end()) {
      return it->second.get();
    }
  }
  // Release the lock while compiling the kernel.
  TF_ASSIGN_OR_RETURN(auto kernel, factory());
  absl::MutexLock lock(&cache.mutex);
  auto [iter, inserted] =
      cache.kernels.insert_or_assign(kernel_hash, std::move(kernel));
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
  kernel.init(&module_ptr, &kernel_ptr);
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

  {
    absl::MutexLock lock(&cache->mutex);
    auto it = cache->contexts.find(key);
    if (it != cache->contexts.end()) {
      VLOG(5) << "Found Mosaic GPU kernel in cache";
      return it->second;
    }
  }
  TF_ASSIGN_OR_RETURN(void* context, InitKernel(*kernel));
  absl::MutexLock lock(&cache->mutex);
  cache->contexts.insert_or_assign(key, context);
  return context;
}

// Structure stores data needed during the execution and filled during the
// initialization.
struct DeviceState {
  // Memory addresses of the barrier signal buffers on all participating peers.
  // Used for cross-device synchronization.
  std::vector<se::DeviceAddressBase> peer_barrier_signal_buffers;

  // Memory used to store the current value of the cross-device barrier.
  se::DeviceAddressHandle barrier_signal_value_buffer_handle;

  // Memory used to store the signal buffer for the cross-device barrier.
  se::DeviceAddressHandle barrier_signal_buffer_handle;

  // Serialized collective kernel metadata.
  // Structure has the following layout:
  // [CollectiveKernelMetadata][param_to_peers][multimem_addresses]
  // Note: the collective metadata param to peers and multimem addresses are
  // pointing to the nullptr and should not be used during the lowering.
  std::vector<std::byte> metadata_bytes;

  // The RAII handle of the buffer on the device which stores the structure
  // above.
  se::DeviceAddressHandle metadata_handle;
};

constexpr int kMaxPeers = 8;

}  // namespace

namespace mosaic::gpu {
struct CustomCallResources {
  CompiledKernel* kernel = nullptr;

  // For each participating device store the metadata for the collective
  // operation.
  std::array<DeviceState, kMaxPeers> device_states;
  KernelHash hash;

  static absl::StatusOr<std::string> Serialize(
      const CustomCallResources& resources);
  static absl::StatusOr<std::unique_ptr<CustomCallResources>> Deserialize(
      absl::string_view data);
};
absl::StatusOr<std::string> CustomCallResources::Serialize(
    const CustomCallResources& resources) {
  mosaic::gpu::MosaicGpuKernelProto kernel_proto;
  CompiledKernel* kernel = resources.kernel;
  if (kernel == nullptr) {
    return absl::InternalError(
        "Failed to serialize CustomCallResources: CompiledKernel is null");
  }
  kernel_proto.set_version(1);
  kernel_proto.set_object_file(kernel->object_file);
  kernel_proto.set_is_comm_used(kernel->is_comm_used);
  kernel_proto.set_kernel_hash(resources.hash.data(), sizeof(KernelHash));
  kernel_proto.set_host_func_name(kernel->host_func_name);
  kernel_proto.set_init_func_name(kernel->init_func_name);
  return kernel_proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<CustomCallResources>>
CustomCallResources::Deserialize(absl::string_view data) {
  mosaic::gpu::MosaicGpuKernelProto kernel_proto;
  if (!kernel_proto.ParseFromString(data)) {
    return absl::InternalError("Failed to parse MosaicGpuKernel proto");
  }
  auto resources = std::make_unique<CustomCallResources>();
  if (kernel_proto.kernel_hash().size() != sizeof(KernelHash)) {
    return absl::InternalError("Invalid kernel hash size in proto");
  }
  std::memcpy(resources->hash.data(), kernel_proto.kernel_hash().data(),
              sizeof(KernelHash));

  if (kernel_proto.version() != 1) {
    return absl::InternalError(absl::StrCat(
        "Unsupported Mosaic GPU kernel version: ", kernel_proto.version()));
  }

  std::string host_func_name = kernel_proto.host_func_name();
  std::string init_func_name = kernel_proto.init_func_name();

  TF_ASSIGN_OR_RETURN(
      resources->kernel,
      GetOrCreateKernel(
          resources->hash,
          [&]() -> absl::StatusOr<std::unique_ptr<CompiledKernel>> {
            return CreateAndInitJIT(llvm::MemoryBuffer::getMemBuffer(
                                        kernel_proto.object_file(), "kernel"),
                                    std::move(host_func_name),
                                    std::move(init_func_name),
                                    kernel_proto.is_comm_used());
          }));
  return resources;
}

}  // namespace mosaic::gpu

namespace {

using ::mosaic::gpu::CustomCallResources;

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
      GetOrCreateKernel(
          hash, [&]() -> absl::StatusOr<std::unique_ptr<CompiledKernel>> {
            return Compile(module, *cc->cuda_compute_capability());
          }));
  return std::make_unique<CustomCallResources>(
      CustomCallResources{.kernel = kernel, .hash = hash});
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

absl::StatusOr<std::vector<int64_t>> GetReplicaIds(
    const xla::ffi::Dictionary& attributes) {
  std::string_view replica_ids_str =
      attributes.get<std::string_view>("xla_replica_ids").value_or("");
  if (replica_ids_str.empty()) {
    return absl::InvalidArgumentError("No replica ids found in attributes.");
  }
  std::vector<int64_t> replica_ids;
  for (const std::string_view replica_id_str :
       absl::StrSplit(replica_ids_str, ',')) {
    int64_t replica_id;
    if (!absl::SimpleAtoi(replica_id_str, &replica_id)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to parse replica id: %s", replica_id_str));
    }
    replica_ids.push_back(replica_id);
  }
  return replica_ids;
}

absl::StatusOr<xla::gpu::GpuCliqueKey> GetCliqueKey(
    const xla::gpu::CollectiveParams& collective_params,
    const xla::ffi::Dictionary& attributes) {
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> replica_ids,
                      GetReplicaIds(attributes));

  xla::ReplicaGroup group;
  group.mutable_replica_ids()->Reserve(replica_ids.size());
  for (int64_t replica_id : replica_ids) {
    group.add_replica_ids(replica_id);
  }

  return GetGpuCliqueKey(
      collective_params, {group},
      xla::CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
      xla::AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE);
}

// Returns device groups for a collective operation. Device groups used by
// XLA to decide if collective communicator can be safely split from one of
// the existing clique. Passing incorrect device groups can lead to deadlocks
// or wasted resources from allocating duplicate communicators.
absl::StatusOr<std::vector<std::vector<xla::GlobalDeviceId>>>
GetCliqueDeviceGroups(const xla::gpu::CollectiveParams& collective_params,
                      const xla::ffi::Dictionary& attributes) {
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> replica_ids,
                      GetReplicaIds(attributes));

  std::vector<std::vector<xla::GlobalDeviceId>> device_groups(1);
  device_groups[0].resize(replica_ids.size());
  for (int32_t i = 0; i < replica_ids.size(); ++i) {
    device_groups[0][i] = xla::GlobalDeviceId(replica_ids[i]);
  }

  return device_groups;
}

struct PtrFormatter {
  void operator()(std::string* out, const void* ptr) const {
    absl::StrAppend(out, absl::StrFormat("%p", ptr));
  }
};

struct DeviceAddressFormatter {
  void operator()(std::string* out,
                  const se::DeviceAddressBase& address) const {
    absl::StrAppend(out, absl::StrFormat("DeviceAddress(ptr=%p, size=%d)",
                                         address.opaque(), address.size()));
  }
};

void* AddOffset(void* ptrs, int64_t offset) {
  return reinterpret_cast<void*>(reinterpret_cast<uint64_t>(ptrs) + offset);
}

absl::Status MosaicGpuPrepare(
    const xla::gpu::CollectiveParams* absl_nullable collective_params,
    xla::gpu::CollectiveCliqueRequests* absl_nullable clique_requests,
    CustomCallResources* resources, xla::ffi::Dictionary attributes) {
  // Module initialization calls cuModuleLoadData to load the PTX into the GPU.
  // CUDA has a system-wide mutex around this call which can cause a deadlock
  // if user tries to load the module on one GPU while another GPU is executing
  // the same module. To prevent this we need to initialize all modules used
  // by several devices before the first execution.
  // See more details:
  // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#impact-on-concurrent-kernel-execution
  // This operation should be done at Prepare stage since XLA launches a
  // rendez-vous between Prepare and Initialize, which we need here to make sure
  // that modules were loaded on all devices before the first execution.
  // TODO(b/481949311): Store kernel_ctx in a thunk state to avoid CachedInit
  // call at execution time.
  TF_ASSIGN_OR_RETURN(void* kernel_ctx, CachedInit(resources->kernel));
  CHECK_NOTNULL(kernel_ctx);

  if (!ModuleUsesCollectiveMetadata(attributes)) {
    return absl::OkStatus();
  }

  CHECK(collective_params != nullptr);
  CHECK(clique_requests != nullptr);

  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCliqueKey clique_key,
                      GetCliqueKey(*collective_params, attributes));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<xla::GlobalDeviceId>> device_groups,
      GetCliqueDeviceGroups(*collective_params, attributes));

  TF_RETURN_IF_ERROR(clique_requests->RequestClique(clique_key, device_groups));

  VLOG(6) << "Prepare is done for clique key: " << clique_key;
  return absl::OkStatus();
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

  TF_ASSIGN_OR_RETURN(std::vector<ffi::AnyBuffer> buffers,
                      GetBuffers(inputs, results));
  // Parameters which are going to be exchanged with peer ranks to construct
  // collective metadata.
  std::vector<se::DeviceAddressBase> parameters;
  parameters.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    parameters.push_back(buffers[i].device_memory());
  }

  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCliqueKey clique_key,
                      GetCliqueKey(*collective_params, attributes));
  xla::RankId rank =
      clique_key.rank(collective_params->global_device_id).value();

  CHECK(rank.value() < resources->device_states.size())
      << "Rank id" << rank.value() << " is out of collective metadata bounds: "
      << resources->device_states.size();
  DeviceState& device_state = resources->device_states[rank.value()];

  // Allocate and zero dedicated buffer for cross-device barrier. These buffers
  // can't be a part of the output parameter used for collective metadata
  // because buffer assigner can use the same buffer for different ops and we
  // need to ensure that this buffer is zeroed between all devices for a given
  // operation.
  // Barrier buffers are created and zeroed once during the first custom call
  // operation initialization. During reruns we can reuse the same buffers for
  // the same operation since multi-device barrier can be called multiple times
  // on the same buffers.
  // It's important to zero the buffer synchronously to avoid the situation
  // when peer barrier buffer is not zeroed before the first execution.
  // We can guarantee a zeroed buffers in all participating devices since
  // below we are running rendezvous to exchange peer parameters in the
  // CollectParamToPeers call.
  if (device_state.barrier_signal_buffer_handle.address().is_null()) {
    device_state.barrier_signal_buffer_handle = se::DeviceAddressHandle{
        collective_params->executor,
        collective_params->executor->Allocate(
            xla::gpu::GetMultiGpuBarrierSignalBufferSize())};

    se::DeviceAddressBase barrier_signal_buffer_address =
        device_state.barrier_signal_buffer_handle.address();
    TF_RETURN_IF_ERROR(collective_params->executor->SynchronousMemZero(
        &barrier_signal_buffer_address, barrier_signal_buffer_address.size()));
  }

  if (device_state.metadata_handle.address().is_null()) {
    device_state.barrier_signal_value_buffer_handle = se::DeviceAddressHandle{
        collective_params->executor,
        collective_params->executor->Allocate(
            xla::gpu::GetMultiGpuBarrierSignalValueSize())};
    se::DeviceAddressBase barrier_signal_value_buffer_address =
        device_state.barrier_signal_value_buffer_handle.address();
    TF_RETURN_IF_ERROR(collective_params->executor->SynchronousMemZero(
        &barrier_signal_value_buffer_address,
        barrier_signal_value_buffer_address.size()));
  }

  // Exchange the adresses of the buffer barriers.
  parameters.push_back(device_state.barrier_signal_buffer_handle.address());
  const size_t barrier_parameter_index =
      buffers.size() * clique_key.num_devices();
  TF_ASSIGN_OR_RETURN(std::vector<void*> param_to_peers,
                      xla::gpu::CollectParamToPeers(clique_key, rank, stream,
                                                    std::move(parameters)));

  // Collect addresses of the barrier buffers at the peer devices.
  device_state.peer_barrier_signal_buffers.resize(clique_key.num_devices());
  for (int peer = 0; peer < clique_key.num_devices(); ++peer) {
    device_state.peer_barrier_signal_buffers[peer] =
        se::DeviceAddressBase(param_to_peers[barrier_parameter_index + peer],
                              xla::gpu::GetMultiGpuBarrierSignalBufferSize());
  }

  // Drop the addresses of the barrier buffers from the param_to_peers array,
  // since they are not needed during the execution.
  param_to_peers.resize(param_to_peers.size() - clique_key.num_devices());

  // Construct the collective kernel metadata information.
  CollectiveKernelMetadata metadata;
  metadata.rank = rank.value();
  // See description of DeviceState for more details.
  metadata.param_to_peers = nullptr;
  metadata.param_to_multimem_addresses = nullptr;

  const size_t metadata_size =
      sizeof(CollectiveKernelMetadata) +
      buffers.size() * clique_key.num_devices() * sizeof(void*);
  device_state.metadata_bytes.resize(metadata_size);
  std::memcpy(device_state.metadata_bytes.data(), &metadata,
              sizeof(CollectiveKernelMetadata));
  void* param_to_peers_ptr = AddOffset(device_state.metadata_bytes.data(),
                                       sizeof(CollectiveKernelMetadata));
  std::memcpy(param_to_peers_ptr, param_to_peers.data(),
              param_to_peers.size() * sizeof(void*));

  device_state.metadata_handle = se::DeviceAddressHandle{
      collective_params->executor,
      collective_params->executor->Allocate(metadata_size)};
  // Copy metadata to the device.
  se::DeviceAddressBase metadata_address =
      device_state.metadata_handle.address();
  TF_RETURN_IF_ERROR(stream->Memcpy(&metadata_address,
                                    device_state.metadata_bytes.data(),
                                    device_state.metadata_bytes.size()));

  VLOG(6) << "[" << rank << "] Constructed device state {"
          << " metadata rank: " << metadata.rank << ", param_to_peers: ("
          << absl::StrJoin(param_to_peers, ", ", PtrFormatter{})
          << "), peer_barrier_signal_buffers: ("
          << absl::StrJoin(device_state.peer_barrier_signal_buffers, ", ",
                           DeviceAddressFormatter{})
          << "), copied metadata to the device with address: "
          << metadata_address.opaque() << "}";
  return absl::OkStatus();
}

absl::Status MosaicGpuExecute(
    se::Stream* stream, const xla::gpu::CollectiveParams* collective_params,
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

  CompiledKernel* kernel = resources->kernel;
  TF_ASSIGN_OR_RETURN(void* kernel_ctx, CachedInit(kernel));

  cudaStream_t cuda_stream =
      reinterpret_cast<cudaStream_t>(stream->platform_specific_handle().stream);
  // Adding a CPU version of the collective metadata for TMA initialization.
  if (uses_collective_metadata) {
    TF_ASSIGN_OR_RETURN(xla::gpu::GpuCliqueKey clique_key,
                        GetCliqueKey(*collective_params, attributes));
    auto current_rank =
        clique_key.rank(collective_params->global_device_id).value();
    CHECK(current_rank.value() < resources->device_states.size())
        << "Rank id" << current_rank.value()
        << " is out of collective metadata bounds: "
        << resources->device_states.size();
    DeviceState& device_state = resources->device_states[current_rank.value()];

    se::DeviceAddressBase metadata_address =
        device_state.metadata_handle.address();
    VLOG(5) << "[" << current_rank
            << "] Executing collective with metadata address: "
            << metadata_address.opaque();

    // Appending both the device and the host-side collective metadata.
    // The host-side metadata is needed for TMA initialization.
    buffer_ptrs.push_back(metadata_address.opaque());
    buffer_ptrs.push_back(device_state.metadata_bytes.data());

    VLOG(6) << "[" << current_rank
            << "] Starting multi-GPU barrier with key: " << clique_key;
    TF_RETURN_IF_ERROR(xla::gpu::LaunchMultiGpuBarrier(
        stream, clique_key.num_devices(), current_rank,
        device_state.peer_barrier_signal_buffers,
        device_state.barrier_signal_value_buffer_handle.address()));
    VLOG(6) << "[" << current_rank
            << "] Finished multi-GPU barrier with key: " << clique_key;
  } else if (kernel->is_comm_used) {
    NvshmemApi::Default().barrier_all_on_stream(cuda_stream);
  }

  void** buffers_data = buffer_ptrs.data();
  kernel->host_launch(kernel_ctx, cuda_stream, buffers_data);
  return absl::OkStatus();
}

}  // namespace

namespace xla::ffi {
template <>
struct TypeRegistry::SerDes<mosaic::gpu::CustomCallResources> {
  static constexpr bool value = true;

  static absl::StatusOr<std::string> Serialize(
      const mosaic::gpu::CustomCallResources& resources) {
    return mosaic::gpu::CustomCallResources::Serialize(resources);
  }

  static absl::StatusOr<std::unique_ptr<mosaic::gpu::CustomCallResources>>
  Deserialize(absl::string_view data) {
    return mosaic::gpu::CustomCallResources::Deserialize(data);
  }
};
}  // namespace xla::ffi

namespace {
XLA_FFI_DEFINE_HANDLER(kMosaicGpuPrepare, MosaicGpuPrepare,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliqueRequests>()
                           .Ctx<xla::ffi::State<CustomCallResources>>()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(
    kMosaicGPUInstantiate, InstantiateResources,
    ffi::Ffi::BindInstantiate().Ctx<ffi::TargetGpuComputeCapability>().Attrs());

XLA_FFI_DEFINE_HANDLER(
    kMosaicGpuInitialize, MosaicGpuInitialize,
    ffi::Ffi::BindInitialize()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CollectiveParams>()
        .Ctx<ffi::CollectiveCliques>()
        .RemainingArgs()
        .RemainingRets()
        .Ctx<xla::ffi::State<mosaic::gpu::CustomCallResources>>()
        .Attrs(),
    {ffi::Traits::kCmdBufferCompatible});

//  We expect the following attributes:
// - kernel_hash: a hash of the kernel.
// - module: the serialized MLIR module.
// - use_custom_barrier
// - uses_xla_collective_metadata (optional)
XLA_FFI_DEFINE_HANDLER(
    kMosaicGpuExecute, MosaicGpuExecute,
    ffi::Ffi::BindExecute()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::CollectiveParams>()
        .RemainingArgs()
        .RemainingRets()
        .Ctx<xla::ffi::State<mosaic::gpu::CustomCallResources>>()
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
  auto cc = GetCudaComputeCapability();
  if (!cc.ok()) {
    return nullptr;
  }
  auto kernel = Compile(module_str, *cc);
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

__attribute__((visibility("default"))) void MosaicGpuClearKernelCache() {
  auto& cache = GetKernelCache();
  absl::MutexLock lock(&cache.mutex);
  cache.kernels.clear();
}

}  // extern "C"
