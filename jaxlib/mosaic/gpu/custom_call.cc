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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/TargetSelect.h"
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/include/mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/include/mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/include/mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/include/mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/include/mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/include/mlir/Conversion/Passes.h"
#include "mlir/include/mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Arith/Transforms/Passes.h"
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
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/include/mlir/Transforms/Passes.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu.h"
#include "jaxlib/mosaic/gpu/launch_lowering.h"
#include "jaxlib/mosaic/gpu/passes.h"
#include "jaxlib/mosaic/gpu/serde.h"
#include "jaxlib/mosaic/gpu/target.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"

namespace {

using MosaicInitFunc = void(void****);
using MosaicHostFunc = void(void**);

absl::StatusOr<std::pair<std::string, std::string>> GetSmAndPtxIsaVersion() {
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
  return mosaic::gpu::GetSmAndPtxIsaVersion(major, minor);
}

mlir::FailureOr<mlir::OpPassManager> GetPassPipeline(
    mlir::MLIRContext* ctx, mlir::gpu::CompilationTarget target,
    const std::string& sm, const std::string& ptx_isa) {
  static bool register_once = []() {
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
    // TODO(apaszke): Only register the passes we actually use.
    mlir::memref::registerMemRefPasses();
    mlir::registerConvertToLLVMPass();
    mlir::registerGPUPasses();
    mlir::registerGpuLaunchSinkIndexComputations();
    mosaic::gpu::registerGpuLaunchLoweringPass();
    mosaic::gpu::registerConvertGpuToLLVMPass();
    mosaic::gpu::registerByvalInsertionPass();
    mlir::arith::registerArithExpandOpsPass();
    return true;
  }();
  (void)register_once;
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
      sm, R"( fast=false features=+)", ptx_isa,
      R"( ftz=false  module= triple=nvptx64-nvidia-cuda},
        lower-affine,
        convert-arith-to-llvm{index-bitwidth=0},
        convert-index-to-llvm{index-bitwidth=64},
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
        cse,
        gpu.module(strip-debuginfo),
        gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false}),
        gpu.module(canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
        gpu.module(cse),
        gpu.module(mosaic-byval-insertion),
        gpu.module(reconcile-unrealized-casts),
        mosaic-convert-gpu-to-llvm,
        gpu-module-to-binary{format=)",
      mlir::gpu::stringifyCompilationTarget(target).str(), R"(},
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
                              mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  *static_cast<mlir::OpPassManager*>(&pm) = std::move(passes);
  if (getenv("MOSAIC_GPU_DUMP_MLIR_PASSES") != nullptr) {
    pm.enableIRPrinting();
  }
  return pm.run(module);
}

void InitContext(mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                  mlir::gpu::GPUDialect, mlir::nvgpu::NVGPUDialect,
                  mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect,
                  mosaic_gpu::MosaicGPUDialect>();
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
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

absl::Status RunCUDATool(const char* tool,
                         const std::vector<const char*>& args,
                         bool stderr_to_stdout = false) {
  CHECK(!args.empty() && args.back() == nullptr);
  const char * cuda_path_ptr = getenv("CUDA_ROOT");
  if (!cuda_path_ptr) return absl::InternalError("Failed to get CUDA_ROOT");
  std::string tool_path(cuda_path_ptr);
  tool_path += "/bin/";
  tool_path += tool;
  pid_t child_pid;
  posix_spawn_file_actions_t file_actions;
  if (posix_spawn_file_actions_init(&file_actions)) {
    return absl::InternalError("Failed to initialize spawn file actions");
  }
  if (posix_spawn_file_actions_adddup2(&file_actions, STDOUT_FILENO,
                                       STDERR_FILENO)) {
    return absl::InternalError("Failed to set up spawn file actions");
  }
  // execv is guaranteed by POSIX to not modify the args (other than
  // replacing the whole process image), so the const_cast is valid.
  if (posix_spawn(&child_pid, tool_path.c_str(), &file_actions, nullptr,
                  const_cast<char* const*>(args.data()), environ)) {
    return absl::InternalError("Process spawn failed");
  }
  int status;
  if (waitpid(child_pid, &status, 0) == -1) {
    return absl::InternalError("Failed to wait for CUDA tool invocation");
  }
  if (status != 0) return absl::InternalError("CUDA tool failed");
  if (posix_spawn_file_actions_destroy(&file_actions) != 0) {
    return absl::InternalError("Failed to clean up after posix_spawn");
  }
  return absl::OkStatus();
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

void DumpCompilationOutput(mlir::ModuleOp module, const std::string& sm,
                           const std::string& ptx_isa) {
  bool dump_ptx = getenv("MOSAIC_GPU_DUMP_PTX") != nullptr;
  bool dump_ptxas = getenv("MOSAIC_GPU_DUMP_PTXAS") != nullptr;
  bool dump_sass = getenv("MOSAIC_GPU_DUMP_SASS") != nullptr;
  if (!dump_ptx && !dump_ptxas && !dump_sass) {
    return;
  }

  module = module.clone();  // Prevent accidental modification.
  absl::Cleanup module_destroyer = [module] { module->erase(); };
  auto passes = GetPassPipeline(
      module.getContext(), mlir::gpu::CompilationTarget::Assembly, sm, ptx_isa);
  if (mlir::failed(passes) ||
      mlir::failed(RunPasses(std::move(*passes), module))) {
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
    if (dump_ptx) {
      std::cout << ptx << std::endl;
    }
    if (!dump_ptxas && !dump_sass) { continue; }  // We're done.
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
    if (dump_ptxas) {
      ptxas_args.push_back("-v");
    }
    ptxas_args.push_back(nullptr);
    if (auto status = RunCUDATool("ptxas", ptxas_args); !status.ok()) {
      std::cerr << "ptxas invocation failed: " << status.message() << std::endl;
      continue;
    }
    if (!dump_sass) { continue; }  // We're done.
    // Call nvdisasm to pretty-print SASS.
    if (auto status = RunCUDATool(
            "nvdisasm", {"nvdisasm", "-ndf", "-c", elf_path.c_str(), nullptr});
        !status.ok()) {
      std::cerr << "nvdisasm invocation failed: " << status.message()
                << std::endl;
      continue;
    }
  }
}

absl::StatusOr<std::unique_ptr<mlir::ExecutionEngine>> Compile(
    mlir::ModuleOp module) {
  auto sm_and_ptx_isa = GetSmAndPtxIsaVersion();
  if (!sm_and_ptx_isa.ok()) {
    return sm_and_ptx_isa.status();
  }
  const std::string sm = sm_and_ptx_isa.value().first;
  const std::string ptx_isa = sm_and_ptx_isa.value().second;
  DumpCompilationOutput(module, sm, ptx_isa);
  auto passes = GetPassPipeline(
      module.getContext(), mlir::gpu::CompilationTarget::Binary, sm, ptx_isa);
  if (mlir::failed(passes)) {
    return absl::InternalError("Failed to construct pass pipeline");
  }
  if (mlir::failed(RunPasses(std::move(*passes), module))) {
    return absl::InternalError("Pass pipeline failed");
  }

  llvm::SmallVector<llvm::StringRef> runtime_lib;
  if (const char* lib_path = getenv("MOSAIC_GPU_RUNTIME_LIB_PATH")) {
    runtime_lib.emplace_back(lib_path);
  }
  // Create a transformer to run all LLVM optimization passes at the
  // specified optimization level.
  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  mlir::ExecutionEngineOptions options;
  options.transformer = transformer;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  options.sharedLibPaths = runtime_lib;
  auto maybe_execution_engine = mlir::ExecutionEngine::create(module, options);
  if (!maybe_execution_engine) {
    return absl::InternalError("Failed to compile kernel");
  }
  return std::move(*maybe_execution_engine);
}

class CompiledKernel {
 public:
  CompiledKernel(std::unique_ptr<mlir::ExecutionEngine> engine, void* ctx,
                 MosaicHostFunc* host_launch)
      : engine_(std::move(engine)), ctx_(ctx), host_launch_(host_launch) {}

  std::tuple<void*, MosaicHostFunc*> GetHostLaunch() {
    return std::make_tuple(ctx_, host_launch_);
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  void* ctx_;  // TODO(apaszke): Destroy this properly
  MosaicHostFunc* host_launch_;
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
  mlir::ExecutionEngine* execution_engine = maybe_engine->get();

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
  return CompiledKernel(std::move(*maybe_engine), kernel_ptr,
                        reinterpret_cast<MosaicHostFunc*>(*host));
}

// Each compiled kernel has a unique init func, and each kernel is used from
// a single HLO module. So it should be safe to not include the CUDA context
// in the key.
absl::StatusOr<std::tuple<void*, MosaicHostFunc*>> CachedCompileAndInit(
    CacheKey key, const char* module) {
  auto cache_and_mutex = GetKernelCache();
  auto* cache = cache_and_mutex.first;
  auto* mutex = cache_and_mutex.second;

  {
    // Fast path uses reader lock (as hash map look-up is relatively slow).
    absl::ReaderMutexLock lock(mutex);
    auto it = cache->find(key);
    if (ABSL_PREDICT_TRUE(it != cache->end()))
      return it->second.GetHostLaunch();
  }

  absl::MutexLock lock(mutex);
  // We released the reader lock, another thread might have initialized it.
  if (cache->find(key) == cache->end()) {
    auto compiled = CompileAndInit(module);
    if (!compiled.ok()) {
      return compiled.status();
    }
    cache->insert_or_assign(key, std::move(*compiled));
  }
  return cache->at(key).GetHostLaunch();
}

void MosaicGPUCustomCall(void* stream, void** buffers, char* opaque,
                         size_t opaque_len, XlaCustomCallStatus* status) {
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
  auto ctx_and_kernel = CachedCompileAndInit(key, opaque + sizeof(KernelHash));
  if (!ctx_and_kernel.ok()) {
    XlaCustomCallStatusSetFailure(status,
                                  ctx_and_kernel.status().message().data(),
                                  ctx_and_kernel.status().message().size());
    return;
  }
  void* args[4] = {&std::get<0>(*ctx_and_kernel), &stream, &buffers};
  std::get<1>(*ctx_and_kernel)(args);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("mosaic_gpu", &MosaicGPUCustomCall,
                                         "CUDA");

}  // namespace

extern "C" {

__attribute__((visibility("default")))
void** MosaicGpuCompile(const char* module) {
  auto compiled = CompileAndInit(module);
  if (!compiled.ok()) {
    return nullptr;
  }
  auto [ctx, launch] = compiled->GetHostLaunch();
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
