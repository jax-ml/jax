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
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "jaxlib/mosaic/gpu/library_paths.h"
#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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
#include "jaxlib/mosaic/gpu/launch_lowering.h"
#include "jaxlib/mosaic/gpu/nvshmem.h"
#include "jaxlib/mosaic/gpu/passes.h"
#include "jaxlib/mosaic/gpu/serde.h"
#include "jaxlib/mosaic/gpu/target.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
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

struct DumpOptions {
  // Whether to dump the MLIR module before and after each pass.
  bool mlir_passes = false;
  // Whether to dump the PTX resulting from the compilation.
  bool ptx = false;
  // Whether to run ptxas in verbose mode.
  bool ptxas = false;
  // Whether to dump the SASS resulting from the compilation. If both `sass`
  // and `sass_ctrl` are true, a single dump containing both will be
  // generated.
  bool sass = false;
  // Whether to dump the SASS control codes following NervanaSystems/maxas. If
  // both `sass` and `sass_ctrl` are true, a single dump containing both will be
  // generated.
  bool sass_ctrl = false;
  // Where to dump the output files. If empty, dump to stdout.
  std::string dump_path = "";
  // The basename to use when dumping files.
  std::string module_basename;
};

DumpOptions GetDumpOptionsForModule(mlir::ModuleOp module) {
  // Use a static variable in order to ensure that subsequent compilations of
  // modules that share the same name will result in distinct dumps.
  static std::atomic<int> dumped_module_count = 0;
  DumpOptions opts;
  int current_count = dumped_module_count.fetch_add(1);
  if (std::optional<llvm::StringRef> name = module.getName();
      name.has_value()) {
    opts.module_basename = absl::StrCat(name->str(), "_", current_count);
  } else {
    opts.module_basename = absl::StrCat("mosaic_gpu_module_", current_count);
  }

  if (char* dump_to = getenv("MOSAIC_GPU_DUMP_TO"); dump_to != nullptr) {
    // "sponge" is a special value, which if set, will result in the files being
    // dumped to a directory path specified in the `TEST_UNDECLARED_OUTPUTS_DIR`
    // environment variable.
    if (absl::string_view(dump_to) == "sponge") {
      if (char* dump_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
          dump_dir != nullptr) {
        opts.dump_path = dump_dir;
      } else {
        LOG(WARNING) << "\"sponge\" specified as dump directory but "
                        "TEST_UNDECLARED_OUTPUTS_DIR is not set! "
                        "Will dump to stdout instead.";
      }
    } else if (absl::string_view(dump_to) == "-") {
      // Dump to stdout.
      opts.dump_path = "";
    } else {
      opts.dump_path = dump_to;
    }

    opts.mlir_passes = true;
    opts.ptx = true;
    opts.sass = true;
    opts.sass_ctrl = true;
    return opts;
  }

  opts.mlir_passes = getenv("MOSAIC_GPU_DUMP_MLIR_PASSES") != nullptr;
  opts.ptx = getenv("MOSAIC_GPU_DUMP_PTX") != nullptr;
  opts.ptxas = getenv("MOSAIC_GPU_DUMP_PTXAS") != nullptr;
  opts.sass_ctrl = getenv("MOSAIC_GPU_DUMP_SASS_CTRL") != nullptr;
  opts.sass = getenv("MOSAIC_GPU_DUMP_SASS") != nullptr || opts.sass_ctrl;
  return opts;
}

class TemporaryDirectory {
 private:
  TemporaryDirectory(std::string path) : path(std::move(path)) {}
  // TODO(apaszke): Unlink in destructor.

 public:
  static absl::StatusOr<TemporaryDirectory> Create() {
    std::string pattern = "/tmp/mosaic-gpu-XXXXXX";
    if (mkdtemp(pattern.data()) == NULL) {
      return absl::InternalError("Failed to create temporary directory");
    }
    return TemporaryDirectory(std::move(pattern));
  }

  std::string_view GetPath() { return path; }

 private:
  std::string path;
};

absl::StatusOr<std::string> RunCUDATool(const char* tool,
                                        const std::vector<const char*>& args,
                                        bool stderr_to_stdout = true) {
  CHECK(!args.empty() && args.back() == nullptr);
  const char* cuda_path_ptr = mosaic::gpu::GetCUDARoot();
  if (!cuda_path_ptr)
    return absl::InternalError("Failed to get the CUDA toolkit path");
  std::string tool_path(cuda_path_ptr);
  tool_path += "/bin/";
  tool_path += tool;
  int stdout_pipe[2] = {-1, -1};
  pid_t child_pid;
  posix_spawn_file_actions_t file_actions;
  if (posix_spawn_file_actions_init(&file_actions)) {
    return absl::InternalError("Failed to initialize spawn file actions");
  }
  absl::Cleanup file_actions_destroyer = [&file_actions] {
    posix_spawn_file_actions_destroy(&file_actions);
  };
  if (pipe(stdout_pipe) == -1) {
    return absl::InternalError("Failed to set up pipe");
  }
  absl::Cleanup pipe_closer = [&stdout_pipe] {
    if (stdout_pipe[0] != -1) close(stdout_pipe[0]);
    if (stdout_pipe[1] != -1) close(stdout_pipe[1]);
  };
  // close read end in child
  if (posix_spawn_file_actions_addclose(&file_actions, stdout_pipe[0])) {
    return absl::InternalError("Failed to close read end of the pipe in child");
  }
  if (posix_spawn_file_actions_adddup2(&file_actions, stdout_pipe[1],
                                       STDOUT_FILENO)) {
    return absl::InternalError("Failed to redirect stdout to pipe");
  }
  if (stderr_to_stdout && posix_spawn_file_actions_adddup2(
                              &file_actions, STDOUT_FILENO, STDERR_FILENO)) {
    return absl::InternalError("Failed to redirect stderr to stdout");
  }
  // execv is guaranteed by POSIX to not modify the args (other than
  // replacing the whole process image), so the const_cast is valid.
  if (int status =
          posix_spawn(&child_pid, tool_path.c_str(), &file_actions, nullptr,
                      const_cast<char* const*>(args.data()), environ)) {
    return absl::InternalError(
        absl::StrCat("Process spawn failed: ", strerror(status)));
  }
  // Proactively close write end in parent. If we don't do this, read
  // will block since the pipe will have an open write end in the
  // parent process.
  if (close(stdout_pipe[1]) == -1) {
    return absl::InternalError(
        absl::StrCat("Failed to close write end of pipe in parent process: ",
                     strerror(errno)));
  }
  // Mark the write end as successfully closed, so it doesn't get
  // closed a second time by the deferred pipe_closer.
  stdout_pipe[1] = -1;
  std::string stdout;
  char buf[1024];
  while (int bytes_read = read(stdout_pipe[0], buf, sizeof buf)) {
    if (bytes_read == -1) {
      return absl::InternalError(
          absl::StrCat("Failed to read from pipe: ", strerror(errno)));
    }
    stdout.append(buf, bytes_read);
  }
  int status;
  if (waitpid(child_pid, &status, 0) == -1) {
    return absl::InternalError("Failed to wait for CUDA tool invocation");
  }
  if (status != 0) {
    std::string error_message = "CUDA tool failed";
    if (!stdout.empty()) {
      error_message += ": ";
      error_message += stdout;
    }
    return absl::InternalError(error_message);
  }
  return stdout;
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

absl::StatusOr<std::unique_ptr<se::cuda::CompilationProvider>>
GetPtxCompilationProvider() {
  // Defaults mostly mirror those used in `xla/debug_options_flags.cc`.
  std::string default_cuda_data_dir = "./cuda_sdk_lib";
  // TODO(bchetioui): this does not mirror the XLA default. Evaluate whether
  // using NvJitLink would work as necessary.
  constexpr se::cuda::CompilationProviderOptions::NvJitLinkMode nvjitlink_mode =
      se::cuda::CompilationProviderOptions::NvJitLinkMode::kDisabled;
  constexpr bool enable_llvm_module_compilation_parallelism = false;
  constexpr bool enable_driver_compilation = false;
  bool enable_libnvptxcompiler = stream_executor::IsLibNvPtxCompilerSupported();

  se::cuda::CompilationProviderOptions opts(
      nvjitlink_mode, enable_libnvptxcompiler,
      enable_llvm_module_compilation_parallelism, enable_driver_compilation,
      std::move(default_cuda_data_dir));
  return se::cuda::AssembleCompilationProvider(opts);
}

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
  EnsureLLVMNVPTXTargetIsRegistered();
  // TODO(hebecker): update CudaComputeCapability to embed extensions.
  // Currently, extensions will still be used (but are hardcoded in a util
  // `ShouldUsePtxExtension` instead of being queried).
  return se::CudaComputeCapability(major, minor);
}

mlir::FailureOr<mlir::OpPassManager> GetPassPipeline(
    mlir::MLIRContext* ctx, mlir::gpu::CompilationTarget target,
    const std::string& sm, const std::string& ptx_isa, const std::string& nvshmem_path) {
  static absl::once_flag register_passes_flag;
  absl::call_once(register_passes_flag, []() {
    EnsureLLVMNVPTXTargetIsRegistered();

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
    mosaic::gpu::registerGpuLaunchLoweringPass();
    mosaic::gpu::registerConvertGpuToLLVMPass();
    mosaic::gpu::registerByvalInsertionPass();
    mosaic::gpu::registerNvvmAttrInsertionPass();
    mlir::arith::registerArithExpandOpsPass();
    mlir::LLVM::registerDIScopeForLLVMFuncOpPass();
    return true;
  });
  const char *cuda_root = mosaic::gpu::GetCUDARoot();
  if (!cuda_root) {
    return mlir::failure();
  }
  return mlir::parsePassPipeline(absl::StrCat(
      R"(
      builtin.module(
        arith-expand,
        canonicalize,
        gpu-launch-sink-index-computations,
        convert-nvgpu-to-nvvm,
        gpu-kernel-outlining{data-layout-str=},
        convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1},
        convert-scf-to-cf,
        convert-nvvm-to-llvm,
        expand-strided-metadata,
        nvvm-attach-target{O=3 chip=)",
      sm, " fast=false features=+", ptx_isa,
      R"( ftz=false  module= triple=nvptx64-nvidia-cuda},
        lower-affine,
        convert-arith-to-llvm{index-bitwidth=0},
        convert-index-to-llvm{index-bitwidth=64},
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
        cse,
        )",
      R"(
        gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false}),
        gpu.module(canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
        gpu.module(cse),
        gpu.module(mosaic-byval-insertion),
        gpu.module(mosaic-nvvm-attr-insertion),
        gpu.module(reconcile-unrealized-casts),
        mosaic-convert-gpu-to-llvm,
        ensure-debug-info-scope-on-llvm-func{emission-kind=DebugDirectivesOnly},
        gpu-module-to-binary{format=)",
      mlir::gpu::stringifyCompilationTarget(target).str(),
      (!nvshmem_path.empty() ? " l=" + nvshmem_path : ""),
      "  opts=-lineinfo toolkit=", cuda_root,
      R"(},
        convert-math-to-llvm{approximate-log1p=true},
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
        cse,
        )",
      (target != mlir::gpu::CompilationTarget::Assembly ? "gpu-launch-lowering,"
                                                        : ""),
      R"(
        convert-to-llvm,
        reconcile-unrealized-casts
      )
  )"));
}

mlir::LogicalResult RunPasses(mlir::OpPassManager&& passes,
                              mlir::ModuleOp module,
                              const DumpOptions& dump_opts) {
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
                        dump_stream.has_value() ? *dump_stream : llvm::outs());
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

// Parse the SASS and reformat control codes following NervanaSystems/maxas.
std::string FormatSassCtrl(const std::string& sass) {
  std::string result;
  result.reserve(sass.size());
  std::vector<std::string> lines = absl::StrSplit(sass, '\n');
  for (int i = 0; i < lines.size(); ++i) {
    std::string_view line = lines[i];
    if (i + 1 < lines.size()) {
      const std::string& next_line = lines[i + 1];
      size_t first_hex_start = line.rfind("/* 0x");
      size_t first_instr_end = line.rfind(';');
      size_t second_hex_start = next_line.rfind("/* 0x");
      bool second_line_empty = true;
      if (second_hex_start != std::string::npos) {
        for (size_t i = 0; i < second_hex_start; ++i) {
          second_line_empty &= next_line[i] == ' ';
        }
      }
      if (first_hex_start != std::string::npos &&
          first_instr_end != std::string::npos &&
          second_hex_start != std::string::npos &&
          second_line_empty) {
        line = line.substr(0, first_instr_end);
        std::string hex_str = next_line.substr(second_hex_start + 5, 16);
        uint64_t ctrl;
        if (absl::SimpleHexAtoi(hex_str, &ctrl)) {
          uint64_t stall = (ctrl >> 41) & 0xf;
          uint64_t yield = (ctrl >> 45) & 0x1;
          uint64_t write_barrier = (ctrl >> 46) & 0x7;
          uint64_t read_barrier = (ctrl >> 49) & 0x7;
          uint64_t wait_barrier = (ctrl >> 52) & 0x3f;
          std::string wait_barrier_str;
          if (wait_barrier == 0) {
            result += "   -";
          } else if (absl::has_single_bit(wait_barrier)) {
            absl::StrAppendFormat(&result, "   %d",
                                  absl::countr_zero(wait_barrier));
          } else {
            int first_set = absl::countr_zero(wait_barrier);
            uint64_t without_first_set = wait_barrier ^ (1 << first_set);
            if (absl::has_single_bit(without_first_set)) {
              absl::StrAppendFormat(&result, " %d&%d",
                                    absl::countr_zero(without_first_set),
                                    first_set);
            } else {
              absl::StrAppendFormat(&result, "0x%02x", wait_barrier);
            }
          }
          absl::StrAppendFormat(
              &result, ":%c:%c:%c:%02llu",
              read_barrier == 7 ? '-' : ('0' + read_barrier),
              write_barrier == 7 ? '-' : ('0' + write_barrier),
              yield ? 'Y' : '-', stall);
        }
        i++;  // Skip the hex line.
      }
    }
    result += line;
    result.append("\n");
  }
  return result;
}

void DumpToFileOrStdout(absl::string_view content, absl::string_view name,
                        absl::string_view path) {
  if (path.empty()) {
    std::cout << content << std::endl;
    return;
  }
  std::error_code error;
  llvm::raw_fd_ostream out_file(tsl::io::JoinPath(path, name), error,
                                llvm::sys::fs::OF_None);
  if (error) {
    LOG(ERROR) << error.message();
    LOG(ERROR) << "Output will be written to stdout instead.";
    std::cout << content << std::endl;
    return;
  }
  out_file << content << "\n";
}

// TODO(bchetioui): port this to not call ptxas and nvdisasm as subprocesses.
void DumpCompilationOutput(mlir::ModuleOp module, const std::string& sm,
                           const std::string& ptx_isa,
                           const std::string& nvshmem_path,
                           const DumpOptions& dump_opts) {
  if (!dump_opts.ptx && !dump_opts.ptxas && !dump_opts.sass) {
    return;
  }
  std::string module_name = dump_opts.module_basename;
  module = module.clone();  // Prevent accidental modification.
  absl::Cleanup module_destroyer = [module] { module->erase(); };
  auto passes = GetPassPipeline(
      module.getContext(), mlir::gpu::CompilationTarget::Assembly,
      sm, ptx_isa, nvshmem_path);
  if (mlir::failed(passes) ||
      mlir::failed(RunPasses(std::move(*passes), module, dump_opts))) {
    return;
  }
  for (mlir::Operation& op : module.getBody()->getOperations()) {
    auto binary = mlir::dyn_cast<mlir::gpu::BinaryOp>(&op);
    if (!binary) { continue; }
    auto objects = binary.getObjects();
    if (objects.size() != 1) {
      std::cerr << "Multiple objects per gpu.binary unsupported" << std::endl;
      continue;
    }
    auto object = mlir::cast<mlir::gpu::ObjectAttr>(*objects.begin());
    std::string ptx = object.getObject().getValue().str();
    if (dump_opts.ptx) {
      DumpToFileOrStdout(ptx, module_name + ".ptx", dump_opts.dump_path);
    }
    if (!dump_opts.ptxas && !dump_opts.sass) {
      continue;
    }  // We're done.
    auto tmpdir = TemporaryDirectory::Create();
    if (!tmpdir.ok()) {
      std::cerr << "Failed to create a temporary directory" << std::endl;
      continue;
    }
    std::string ptx_path = std::string(tmpdir->GetPath()) + "/kernel.ptx";
    std::string elf_path = std::string(tmpdir->GetPath()) + "/kernel.o";
    // Dump PTX into a file.
    std::ofstream ptx_out(ptx_path.c_str());
    if (!ptx_out) {
      std::cerr << "Failed to write PTX to a file" << std::endl;
      continue;
    }
    ptx_out << ptx << std::endl;
    // Run ptxas to generate SASS.
    std::vector<const char*> ptxas_args = {
        "ptxas",          "--opt-level",   "3",
        "--gpu-name",     sm.c_str(),      "--output-file",
        elf_path.c_str(), ptx_path.c_str()};
    if (dump_opts.ptxas) {
      ptxas_args.push_back("-v");
    }
    ptxas_args.push_back(nullptr);
    if (auto result = RunCUDATool("ptxas", ptxas_args); !result.ok()) {
      std::cerr << "ptxas invocation failed: " << result.status() << std::endl;
      continue;
    } else if (dump_opts.ptxas) {
      DumpToFileOrStdout(*result, module_name + ".ptxas", dump_opts.dump_path);
    }
    if (!dump_opts.sass) {
      continue;  // We're done.
    }
    // Call nvdisasm to pretty-print SASS.
    std::vector<const char*> nvdisasm_args = {
        "nvdisasm", "-ndf", "-c", elf_path.c_str()};
    if (dump_opts.sass) {
      nvdisasm_args.push_back("-hex");
    }
    nvdisasm_args.push_back(nullptr);
    auto result = RunCUDATool("nvdisasm", nvdisasm_args);
    if (!result.ok()) {
      std::cerr << "nvdisasm invocation failed: " << result.status()
                << std::endl;
      continue;
    }

    if (dump_opts.sass_ctrl) {
      DumpToFileOrStdout(FormatSassCtrl(*result), module_name + ".sass_ctrl",
                         dump_opts.dump_path);
    } else {
      // Dump SASS.
      DumpToFileOrStdout(*result, module_name + ".sass", dump_opts.dump_path);
    }
  }
}

bool is_nvshmem_used(mlir::ModuleOp module) {
  constexpr std::string_view prefix1 = "nvshmem_";
  constexpr std::string_view prefix2 = "nvshmemx_";
  for (mlir::LLVM::LLVMFuncOp llvm_func : module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    const auto& func_name = llvm_func.getName();
    if (!func_name.starts_with(prefix1) && !func_name.starts_with(prefix2)) {
      continue;
    }
    auto uses = mlir::SymbolTable::getSymbolUses(llvm_func, module.getOperation());
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

absl::StatusOr<std::pair<std::unique_ptr<mlir::ExecutionEngine>, bool>> Compile(
    mlir::ModuleOp module) {
  tsl::profiler::TraceMe trace("Compile");
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::cuda::CompilationProvider> compilation_provider,
      GetPtxCompilationProvider());
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
  DumpOptions dump_opts = GetDumpOptionsForModule(module);
  DumpCompilationOutput(module, sm, ptx_isa, nvshmem_path, dump_opts);
  // `DumpCompilationOutput` already runs through MLIR passes and may dump them.
  // If that happened, we don't want to dump them again.
  if (dump_opts.ptx || dump_opts.ptxas || dump_opts.sass) {
    dump_opts.mlir_passes = false;
  }

  const char* debug_only = getenv("MOSAIC_GPU_LLVM_DEBUG_ONLY");
#ifndef NDEBUG
  bool old_debug_state = false;
  if (debug_only) {
    old_debug_state = llvm::DebugFlag;
    std::vector<std::string_view> debug_only_types =
        absl::StrSplit(debug_only, ',');
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
  if (debug_only) {
    fprintf(
        stderr,
        "MOSAIC_GPU_LLVM_DEBUG_ONLY is set but LLVM was built with NDEBUG\n");
    abort();
  }
#endif
  auto passes = GetPassPipeline(
      module.getContext(),
      mlir::gpu::CompilationTarget::Binary,
      sm, ptx_isa, nvshmem_path);
  if (mlir::failed(passes)) {
    return absl::InternalError("Failed to construct pass pipeline");
  }
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
  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  mlir::ExecutionEngineOptions options;
  options.transformer = transformer;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  options.sharedLibPaths = runtime_libs;
  auto maybe_execution_engine = mlir::ExecutionEngine::create(module, options);
#ifndef NDEBUG
  if (debug_only) {
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
      : engine_(std::move(engine)), ctx_(ctx), host_launch_(host_launch),
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

using KernelHash = std::array<uint64_t, 4>;
using CacheKey = std::pair<KernelHash, uintptr_t>;

std::pair<absl::flat_hash_map<CacheKey, CompiledKernel>*, absl::Mutex*>
GetKernelCache() {
  static absl::Mutex mutex;
  static auto& context_cache =
      *new absl::flat_hash_map<CacheKey, CompiledKernel>;
  return std::make_pair(&context_cache, &mutex);
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

absl::StatusOr<CompiledKernel> CompileAndInit(const char* module) {
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
                        reinterpret_cast<MosaicHostFunc*>(*host),
                        is_comm_used);
}

// Each compiled kernel has a unique init func, and each kernel is used from
// a single HLO module. So it should be safe to not include the CUDA context
// in the key.
absl::StatusOr<CompiledKernel*> CachedCompileAndInit(
    CacheKey key, const char* module) {
  auto cache_and_mutex = GetKernelCache();
  auto* cache = cache_and_mutex.first;
  auto* mutex = cache_and_mutex.second;

  {
    // Fast path uses reader lock (as hash map look-up is relatively slow).
    absl::ReaderMutexLock lock(mutex);
    auto it = cache->find(key);
    if (ABSL_PREDICT_TRUE(it != cache->end()))
      return &it->second;
  }

  absl::MutexLock lock(mutex);
  // We released the reader lock, another thread might have initialized it.
  if (cache->find(key) == cache->end()) {
    tsl::profiler::TraceMe trace("Compilation cache miss");
    auto compiled = CompileAndInit(module);
    if (!compiled.ok()) {
      return compiled.status();
    }
    cache->insert_or_assign(key, std::move(*compiled));
  }
  return &cache->at(key);
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

absl::Status MosaicGpuExecute(gpuStream_t stream, ffi::RemainingArgs inputs,
                              ffi::RemainingRets results,
                              absl::string_view kernel_hash,
                              absl::string_view module,
                              bool use_custom_barrier,
                              xla::RunId run_id) {
  // Updated version using the new FFI API supporting custom barrier
  // for distributed kernels
  if (use_custom_barrier) {
    fprintf(stderr, "Custom barrier is not supported on GPUs.\n");
    abort();
  }
  if (reinterpret_cast<const uintptr_t>(kernel_hash.data()) %
          alignof(KernelHash) ||
      kernel_hash.size() != sizeof(KernelHash)) {
    fprintf(stderr, "Misaligned opaque pointer\n");
    abort();
  }
  auto hash = *reinterpret_cast<const KernelHash *>(kernel_hash.data());
  CUcontext ctx;
  if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to get current CUDA context\n");
    abort();
  }
  CacheKey key(hash, reinterpret_cast<uintptr_t>(ctx));
  TF_ASSIGN_OR_RETURN(auto compiled_kernel, CachedCompileAndInit(key, module.data()));
  auto ctx_kernel_comm = compiled_kernel->GetHostLaunch();
  bool is_comm_used = std::get<2>(ctx_kernel_comm);

  std::vector<void *> buffers;
  buffers.reserve(inputs.size() + results.size());
  for (int i = 0; i < inputs.size(); ++i) {
    buffers.push_back(inputs.get<ffi::AnyBuffer>(i)->untyped_data());
  }
  for (int i = 0; i < results.size(); ++i) {
    buffers.push_back((*results.get<ffi::AnyBuffer>(i))->untyped_data());
  }
  void **buffers_ptr = buffers.data();
  void *args[4] = {&std::get<0>(ctx_kernel_comm), &stream, &buffers_ptr};

  if (is_comm_used) {
    mosaic::gpu::NvshmemApi::Default().barrier_all_on_stream(
        reinterpret_cast<cudaStream_t>(stream));
  }
  std::get<1>(ctx_kernel_comm)(args);
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kMosaicGpuExecute, MosaicGpuExecute,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
                           .Ctx<xla::ffi::PlatformStream<gpuStream_t>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<absl::string_view>("kernel_hash")
                           .Attr<absl::string_view>("module")
                           .Attr<bool>("use_custom_barrier")
                           .Ctx<xla::RunId>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "mosaic_gpu_v2", "CUDA",
                         {
                             /*instantiate=*/nullptr,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kMosaicGpuExecute,
                         });

}  // namespace

extern "C" {

__attribute__((visibility("default")))
void** MosaicGpuCompile(const char* module) {
  auto compiled = CompileAndInit(module);
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

__attribute__((visibility("default")))
void MosaicGpuUnload(void** tuple_ptr) {
  delete reinterpret_cast<CompiledKernel*>(tuple_ptr[2]);
  delete[] tuple_ptr;
}

}  // extern "C"
