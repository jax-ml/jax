/* Copyright 2023 The JAX Authors.

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

#include "jaxlib/gpu/triton_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/type_registry.h"
#include "xla/stream_executor/device_description.h"

// Required for absl::c_find_if.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
// Required for absl::CEscape.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "absl/strings/escaping.h"
// Required for absl::StrAppend.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
// Required for absl::StrJoin.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "absl/strings/str_join.h"
// Required for absl::StrSplit.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/triton.pb.h"
#include "jaxlib/gpu/triton_utils.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/service/custom_call_status.h"

#ifdef JAX_GPU_CUDA
#include "xla/stream_executor/cuda/cuda_asm_compiler.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#endif  // JAX_GPU_CUDA

#ifdef JAX_GPU_HIP
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#endif  // JAX_GPU_HIP

#define GPU_RETURN_IF_ERROR(expr) JAX_RETURN_IF_ERROR(JAX_AS_STATUS(expr))

namespace jax::JAX_GPU_NAMESPACE {
namespace {

constexpr float kBenchmarkTimeMillis = 10.;

struct gpuModuleDeleter {
  void operator()(gpuModule_t module) {
    absl::Status status = JAX_AS_STATUS(gpuModuleUnload(module));
    if (!status.ok()) {
      LOG(WARNING) << "Failed to unload GPU module: " << status;
    }
  }
};

using OwnedGPUmodule =
    std::unique_ptr<std::remove_pointer_t<gpuModule_t>, gpuModuleDeleter>;

absl::StatusOr<gpuDevice_t> GetStreamDevice(gpuStream_t stream) {
  gpuDevice_t device;
#ifdef JAX_GPU_HIP
  int device_id = gpuGetStreamDeviceId(stream);
  GPU_RETURN_IF_ERROR(gpuDeviceGet(&device, device_id));
#else  // JAX_GPU_CUDA
  gpuContext_t context;
  GPU_RETURN_IF_ERROR(gpuStreamGetCtx(stream, &context));
  GPU_RETURN_IF_ERROR(gpuCtxPushCurrent(context));
  absl::Cleanup ctx_restorer = [] { gpuCtxPopCurrent(nullptr); };
  GPU_RETURN_IF_ERROR(gpuCtxGetDevice(&device));
#endif
  return device;
}

absl::StatusOr<uint32_t> MaxSharedMemoryPerBlock(gpuDevice_t device) {
  int shared_optin;
  GPU_RETURN_IF_ERROR(gpuDeviceGetAttribute(
      &shared_optin, GPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  return shared_optin;
}

absl::StatusOr<std::vector<uint8_t>> CompileModuleImage(
    std::string_view ptx, int compute_capability) {
  std::vector<uint8_t> module_image;
#ifdef JAX_GPU_HIP  // For HIP/ROCM just read the hsaco file
  std::string result_blob;
  std::string fname{ptx};
  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, fname, &result_blob));
  TF_RETURN_IF_ERROR(env->DeleteFile(fname));
  module_image.assign(result_blob.begin(), result_blob.end());
#else
  // TODO(cjfj): Support `TRITON_PTXAS_PATH` environment variable?
  int cc_major = compute_capability / 10;
  int cc_minor = compute_capability % 10;

  bool has_accelerated_features = cc_major >= 9;
  using FeatureExtension =
      stream_executor::CudaComputeCapability::FeatureExtension;
  const stream_executor::CudaComputeCapability cc(
      cc_major, cc_minor,
      has_accelerated_features ? FeatureExtension::kAcceleratedFeatures
                               : FeatureExtension::kNone);

  // Parse JAX_TRITON_PTXAS_EXTRA_FLAGS (space-separated list).
  std::vector<std::string> ptxas_extra_flags;
  const char* extra_flags_env = std::getenv("JAX_TRITON_PTXAS_EXTRA_FLAGS");
  if (extra_flags_env != nullptr) {
    ptxas_extra_flags = absl::StrSplit(extra_flags_env, ' ', absl::SkipEmpty());
  }

  // Parse JAX_TRITON_PTXAS_DEVICE_DEBUG (boolean: "1" or "true").
  const char* debug_env = std::getenv("JAX_TRITON_PTXAS_DEVICE_DEBUG");
  const bool device_debug =
      debug_env != nullptr && (absl::string_view(debug_env) == "true" ||
                               absl::string_view(debug_env) == "1");
  if (device_debug &&
      absl::c_find_if(ptxas_extra_flags, [](const std::string& flag) {
        return flag == "--device-debug" || flag == "-g";
      }) == ptxas_extra_flags.end()) {
    ptxas_extra_flags.push_back("--device-debug");
  }

  VLOG(1) << absl::StreamFormat(
      "ptxas_extra_flags = {%s}",
      absl::StrJoin(ptxas_extra_flags, ", ",
                    [](std::string* s, absl::string_view v) {
                      absl::StrAppend(s, absl::CEscape(v));
                    }));

  JAX_ASSIGN_OR_RETURN(module_image,
                       stream_executor::CompileGpuAsm(
                           cc, std::string(ptx),
                           stream_executor::GpuAsmOpts(
                               /*disable_gpuasm_optimizations=*/false,
                               /*preferred_cuda_dir=*/"", ptxas_extra_flags)));
#endif
  return module_image;
}

// Creates, and caches, a module image with the given parameters.
// If `pre_compiled_module_image` is not empty, it will be used instead of
// compiling the module image.
absl::StatusOr<ModuleImage*> GetModuleImage(
    std::string_view kernel_name, uint32_t shared_mem_bytes,
    std::string_view ptx, int compute_capability,
    std::string_view pre_compiled_module_image = {}) {
  // Since we want an efficient key lookup, we'd have to suffer a bit.
  using KeyType = std::tuple<std::string, uint32_t, std::string, int>;
  using LookupKeyType =
      std::tuple<std::string_view, uint32_t, std::string_view, int>;

  struct KeyHash {
    using is_transparent = void;

    size_t operator()(const KeyType& v) const {
      return absl::Hash<KeyType>{}(v);
    }
    size_t operator()(const LookupKeyType& v) const {
      return absl::Hash<LookupKeyType>{}(v);
    }
  };

  struct KeyEq {
    using is_transparent = void;

    bool operator()(const KeyType& a, const KeyType& b) const { return a == b; }
    bool operator()(const KeyType& a, const LookupKeyType& b) const {
      return a == b;
    }
    bool operator()(const LookupKeyType& a, const KeyType& b) const {
      return a == b;
    }
  };

  static absl::Mutex cached_images_mutex;
  static auto& module_images =
      *new absl::flat_hash_map<KeyType, std::unique_ptr<ModuleImage>, KeyHash,
                               KeyEq>
          ABSL_GUARDED_BY(cached_images_mutex);

  auto lookup_key = LookupKeyType{std::string_view(kernel_name),
                                  shared_mem_bytes, ptx, compute_capability};
  {
    // Allow threads to lookup in parallel
    absl::ReaderMutexLock reader_lock(cached_images_mutex);
    auto it = module_images.find(lookup_key);
    if (it != module_images.end()) return it->second.get();
  }

  std::vector<uint8_t> module_image;
  if (pre_compiled_module_image.empty()) {
    JAX_ASSIGN_OR_RETURN(module_image,
                         CompileModuleImage(ptx, compute_capability));
  } else {
    module_image.assign(pre_compiled_module_image.begin(),
                        pre_compiled_module_image.end());
  }

  absl::MutexLock writer_lock(cached_images_mutex);
  auto key = KeyType{kernel_name, shared_mem_bytes, ptx, compute_capability};
  // the `key` object must be created before we move kernel_name below.
  auto [it, success] = module_images.try_emplace(std::move(key), nullptr);

  if (success) {
    it->second = std::make_unique<ModuleImage>(
        std::get<0>(it->first), std::move(module_image), shared_mem_bytes,
        compute_capability);
  }
  return it->second.get();
}

absl::StatusOr<float> Benchmark(gpuStream_t stream, KernelCall& kernel_call,
                                void** buffers, int num_iterations) {
  gpuEvent_t start, stop;
  GPU_RETURN_IF_ERROR(gpuEventCreate(&start, /*Flags=*/GPU_EVENT_DEFAULT));
  GPU_RETURN_IF_ERROR(gpuEventCreate(&stop, /*Flags=*/GPU_EVENT_DEFAULT));
  JAX_RETURN_IF_ERROR(kernel_call.Launch(stream, buffers));  // Warm-up.
  GPU_RETURN_IF_ERROR(gpuEventRecord(start, stream));
  for (int i = 0; i < num_iterations; ++i) {
    JAX_RETURN_IF_ERROR(kernel_call.Launch(stream, buffers));
  }
  GPU_RETURN_IF_ERROR(gpuEventRecord(stop, stream));
  GPU_RETURN_IF_ERROR(gpuEventSynchronize(stop));
  float elapsed_ms;
  GPU_RETURN_IF_ERROR(gpuEventElapsedTime(&elapsed_ms, start, stop));
  GPU_RETURN_IF_ERROR(gpuEventDestroy(start));
  GPU_RETURN_IF_ERROR(gpuEventDestroy(stop));
  return elapsed_ms;
}

// Creates, and caches, a kernel using `opaque` as the key.
absl::StatusOr<KernelCall*> GetOrCreateKernelCall(
    std::string_view opaque,
    absl::FunctionRef<absl::StatusOr<KernelCall>()> create_fn) {
  if (opaque.empty()) {
    return absl::InvalidArgumentError("Opaque data is empty.");
  }

  static absl::Mutex kernel_calls_mutex;
  static auto& kernel_calls =
      *new absl::flat_hash_map<std::string,
                               absl::StatusOr<std::unique_ptr<KernelCall>>>
          ABSL_GUARDED_BY(kernel_calls_mutex);

  absl::MutexLock writer_lock(kernel_calls_mutex);
  auto [it, success] = kernel_calls.try_emplace(std::string(opaque),
                                                absl::InternalError("Pending"));
  if (success) {
    it->second = [&]() -> absl::StatusOr<std::unique_ptr<KernelCall>> {
      JAX_ASSIGN_OR_RETURN(KernelCall call, create_fn());
      return std::make_unique<KernelCall>(std::move(call));
    }();
  }

  JAX_RETURN_IF_ERROR(it->second.status());
  return it->second->get();
}

// Creates a `KernelCall` from the given opaque data, and caches it for
// subsequent calls with the same opaque data.
absl::StatusOr<KernelCall*> GetKernelCall(std::string_view opaque,
                                          gpuStream_t stream, void** buffers) {
  return GetOrCreateKernelCall(opaque, [&]() -> absl::StatusOr<KernelCall> {
    JAX_ASSIGN_OR_RETURN(std::string serialized, ZlibUncompress(opaque));

    jax_triton::TritonAnyKernelCall proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError("Failed to parse serialized data.");
    }

    if (proto.has_kernel_call()) {
      return KernelCall::FromProto(proto.kernel_call());
    } else if (proto.has_autotuned_kernel_call()) {
      JAX_ASSIGN_OR_RETURN(
          AutotunedKernelCall autotuned_call,
          AutotunedKernelCall::FromProto(proto.autotuned_kernel_call()));
      return AutotunedKernelCall::Autotune(std::move(autotuned_call), stream,
                                           buffers);
    } else {
      return absl::InvalidArgumentError("Unknown kernel call type.");
    }
  });
}

absl::Status AnnotateModuleLoadStatus(
    const absl::Status& status, std::string_view kernel_name,
    int target_compute_capability, const std::vector<uint8_t>& module_image) {
  int actual_major = 0;
  int actual_minor = 0;
  gpuDevice_t device;
  if (JAX_AS_STATUS(gpuCtxGetDevice(&device)).ok()) {
    JAX_AS_STATUS(gpuDeviceGetAttribute(
                      &actual_major,
                      GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
        .IgnoreError();
    JAX_AS_STATUS(gpuDeviceGetAttribute(
                      &actual_minor,
                      GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
        .IgnoreError();
  }
  bool is_elf = false;
  std::string image_preview = "";
  if (!module_image.empty()) {
    is_elf = (module_image.size() >= 4 && module_image[0] == 0x7f &&
              module_image[1] == 'E' && module_image[2] == 'L' &&
              module_image[3] == 'F');
    image_preview = absl::CEscape(
        std::string_view(reinterpret_cast<const char*>(module_image.data()),
                         std::min(module_image.size(), size_t{256})));
  }
  return absl::Status(
      status.code(),
      absl::StrCat(
          status.message(), "\nFailed to load GPU module for Triton kernel '",
          kernel_name, "'", "\n  Target Compute Capability (from JAX): sm_",
          target_compute_capability, "\n  Actual GPU Compute Capability: sm_",
          actual_major, actual_minor, "\n  Module image size: ",
          module_image.size(), " bytes", "\n  Module image type: ",
          (is_elf ? "ELF Binary (CUBIN/HSACO)" : "Text/Other (PTX/etc.)"),
          "\n  Module image preview (CEscaped): ", image_preview));
}

}  // namespace

class ModuleImage {
 public:
  ModuleImage(std::string_view kernel_name, std::vector<uint8_t> module_image,
              uint32_t shared_mem_bytes, int compute_capability)
      : kernel_name_(kernel_name),
        module_image_(std::move(module_image)),
        shared_mem_bytes_(shared_mem_bytes),
        compute_capability_(compute_capability) {}

  const std::string& kernel_name() const { return kernel_name_; }
  const std::vector<uint8_t>& module_image() const { return module_image_; }
  uint32_t shared_mem_bytes() const { return shared_mem_bytes_; }
  int compute_capability() const { return compute_capability_; }

  absl::StatusOr<gpuFunction_t> GetFunctionForContext(gpuContext_t context) {
    {
      absl::ReaderMutexLock reader_lock(mutex_);
      auto it = functions_.find(context);
      if (ABSL_PREDICT_TRUE(it != functions_.end())) {
        return it->second;
      }
    }

    GPU_RETURN_IF_ERROR(gpuCtxPushCurrent(context));
    absl::Cleanup ctx_restorer = [] {
      absl::Status status = JAX_AS_STATUS(gpuCtxPopCurrent(nullptr));
      if (!status.ok()) {
        LOG(WARNING) << "Failed to pop GPU context: " << status;
      }
    };

    gpuModule_t module;
    absl::Status status =
        JAX_AS_STATUS(gpuModuleLoadData(&module, module_image_.data()));
    if (!status.ok()) {
      return AnnotateModuleLoadStatus(status, kernel_name_, compute_capability_,
                                      module_image_);
    }

    gpuFunction_t function;
    absl::Status function_status = JAX_AS_STATUS(
        gpuModuleGetFunction(&function, module, kernel_name_.c_str()));

    if (!function_status.ok()) {
      JAX_AS_STATUS(gpuModuleUnload(module)).IgnoreError();
      return function_status;
    }

    absl::MutexLock writer_lock(mutex_);
    auto [it, success] = functions_.try_emplace(context, function);
    if (!success) {
      JAX_AS_STATUS(gpuModuleUnload(module)).IgnoreError();
      return it->second;
    }

    modules_.push_back(OwnedGPUmodule(module, gpuModuleDeleter()));

    // The maximum permitted static shared memory allocation in CUDA is 48kB,
    // but we can expose more to the kernel using dynamic shared memory.
    constexpr int kMaxStaticSharedMemBytes = 49152;
    if (shared_mem_bytes_ <= kMaxStaticSharedMemBytes) {
      return function;
    }

    // Set up dynamic shared memory.
    gpuDevice_t device;
    GPU_RETURN_IF_ERROR(gpuCtxGetDevice(&device));

    int shared_optin;
    GPU_RETURN_IF_ERROR(gpuDeviceGetAttribute(
        &shared_optin, GPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));

    if (shared_mem_bytes_ > shared_optin) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shared memory requested (%d b) exceeds device resources (%d b).",
          shared_mem_bytes_, shared_optin));
    }

    if (shared_optin > kMaxStaticSharedMemBytes) {
#ifdef JAX_GPU_CUDA
      GPU_RETURN_IF_ERROR(
          gpuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED));
#endif
      int shared_total;
      GPU_RETURN_IF_ERROR(gpuDeviceGetAttribute(
          &shared_total,
          GPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));
      int shared_static;
      GPU_RETURN_IF_ERROR(gpuFuncGetAttribute(
          &shared_static, GPU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
#ifdef JAX_GPU_CUDA
      GPU_RETURN_IF_ERROR(cuFuncSetAttribute(
          function, GPU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          shared_optin - shared_static));
#endif
    }
    return function;
  }

  static absl::StatusOr<ModuleImage*> FromProto(
      const jax_triton::ModuleImageProto& proto, std::string_view ptx,
      int compute_capability) {
    // Retrieve the ModuleImage via GetModuleImage to ensure caching and reuse
    // across calls. Reusing the cached instance avoids repeating one-time GPU
    // initialization work (e.g., gpuModuleLoadData and gpuModuleGetFunction)
    // on subsequent executions of the same kernel.
    return GetModuleImage(proto.kernel_name(), proto.shared_mem_bytes(), ptx,
                          compute_capability, proto.object_file());
  }

  jax_triton::ModuleImageProto ToProto() const {
    jax_triton::ModuleImageProto proto;
    proto.set_kernel_name(kernel_name_);
    proto.set_object_file(
        std::string(module_image_.begin(), module_image_.end()));
    proto.set_shared_mem_bytes(shared_mem_bytes_);
    return proto;
  }

 private:
  std::string kernel_name_;
  std::vector<uint8_t> module_image_;
  uint32_t shared_mem_bytes_;
  int compute_capability_;

  absl::Mutex mutex_;
  std::vector<OwnedGPUmodule> modules_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<gpuContext_t, gpuFunction_t> functions_
      ABSL_GUARDED_BY(mutex_);
};

// Compiles the kernel proto to machine code (e.g. CUBIN), and updates the proto
// with the results.
absl::Status CompileKernelProto(const stream_executor::GpuComputeCapability* cc,
                                jax_triton::TritonKernel* kernel_proto) {
  int compute_capability = kernel_proto->compute_capability();
  if (cc != nullptr && cc->IsCuda() &&
      cc->cuda_compute_capability() != nullptr) {
    compute_capability = cc->cuda_compute_capability()->major * 10 +
                         cc->cuda_compute_capability()->minor;
  }

  std::string_view pre_compiled =
      kernel_proto->has_module_image()
          ? std::string_view(kernel_proto->module_image().object_file())
          : std::string_view();

  JAX_ASSIGN_OR_RETURN(
      ModuleImage * image,
      GetModuleImage(kernel_proto->kernel_name(),
                     kernel_proto->shared_mem_bytes(), kernel_proto->ptx(),
                     compute_capability, pre_compiled));
  if (image == nullptr) {
    return absl::InternalError("Failed to get module image");
  }

  *kernel_proto->mutable_module_image() = image->ToProto();
  kernel_proto->set_compute_capability(compute_capability);
  return absl::OkStatus();
}

Kernel::Kernel(std::string kernel_name, uint32_t num_warps, uint32_t num_ctas,
               uint32_t shared_mem_bytes, std::string ptx, std::string ttir,
               int compute_capability,
               std::optional<uint32_t> global_scratch_size,
               std::optional<uint32_t> global_scratch_align,
               ModuleImage* module_image)
    : kernel_name_(std::move(kernel_name)),
      block_dim_x_(num_warps * kNumThreadsPerWarp),
      num_ctas_(num_ctas),
      shared_mem_bytes_(shared_mem_bytes),
      ptx_(std::move(ptx)),
      ttir_(std::move(ttir)),
      compute_capability_(compute_capability),
      global_scratch_size_(global_scratch_size),
      global_scratch_align_(global_scratch_align),
      module_image_(module_image) {}

absl::Status Kernel::Launch(gpuStream_t stream, uint32_t grid[3],
                            void** params) {
  if (ABSL_PREDICT_FALSE(module_image_ == nullptr)) {
    JAX_ASSIGN_OR_RETURN(module_image_,
                         GetModuleImage(kernel_name_, shared_mem_bytes_, ptx_,
                                        compute_capability_));
  }

  gpuContext_t context;
#ifdef JAX_GPU_HIP
  int device_id = gpuGetStreamDeviceId(stream);
  gpuDevice_t device;
  GPU_RETURN_IF_ERROR(gpuDeviceGet(&device, device_id));
  GPU_RETURN_IF_ERROR(gpuDevicePrimaryCtxRetain(&context, device));
  JAX_ASSIGN_OR_RETURN(gpuFunction_t kernel,
                       module_image_->GetFunctionForContext(context));
  return JAX_AS_STATUS(gpuLaunchKernel(
      kernel, grid[0], grid[1], grid[2], block_dim_x_,
      /*blockDimY=*/1, /*blockDimZ=*/1, shared_mem_bytes_, stream, params,
      /*extra=*/nullptr));
#else  // JAX_GPU_CUDA
  // TODO(b/324319767): A bug in CUDA prevents us from calling cuStreamGetCtx
  // inside graph capture. We use cuCtxGetCurrent as a workaround here because
  // context is not updated, but we should change it back to cuStreamGetCtx once
  // the bug is fixed.
  gpustreamCaptureStatus_t capture_status;
  GPU_RETURN_IF_ERROR(gpuStreamIsCapturing(stream, &capture_status));
  if (capture_status == GPU_STREAM_CAPTURE_STATUS_ACTIVE) {
    GPU_RETURN_IF_ERROR(gpuCtxGetCurrent(&context));
  } else {
    GPU_RETURN_IF_ERROR(gpuStreamGetCtx(stream, &context));
  }

  JAX_ASSIGN_OR_RETURN(gpuFunction_t kernel,
                       module_image_->GetFunctionForContext(context));
  if (num_ctas_ == 1) {
    return JAX_AS_STATUS(gpuLaunchKernel(
        kernel, grid[0], grid[1], grid[2], block_dim_x_,
        /*blockDimY=*/1, /*blockDimZ=*/1, shared_mem_bytes_, stream, params,
        /*extra=*/nullptr));
  }
  CUlaunchAttribute launch_attrs[2];
  launch_attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launch_attrs[0].value.clusterDim.x = num_ctas_;
  launch_attrs[0].value.clusterDim.y = 1;
  launch_attrs[0].value.clusterDim.z = 1;
  launch_attrs[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  launch_attrs[1].value.clusterSchedulingPolicyPreference =
      CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
  CUlaunchConfig launch_config = {
      /*gridDimX=*/grid[0] * num_ctas_,
      /*gridDimY=*/grid[1],
      /*gridDimZ=*/grid[2],
      /*blockDimX=*/block_dim_x_,
      /*blockDimY=*/1,
      /*blockDimZ=*/1,
      /*sharedMemBytes=*/shared_mem_bytes_,
      /*hStream=*/stream,
      /**attrs=*/launch_attrs,
      /*numAttrs=*/2,
  };
  return JAX_AS_STATUS(
      cuLaunchKernelEx(&launch_config, kernel, params, /*extra=*/nullptr));
#endif
}

/*static*/ absl::StatusOr<Kernel> Kernel::FromProto(
    const jax_triton::TritonKernel& proto) {
  // Use 1 as default value if not specified in already serialized kernels.
  int num_ctas = proto.has_num_ctas() ? proto.num_ctas() : 1;

  ModuleImage* module_image = nullptr;
  if (proto.has_module_image()) {
    JAX_ASSIGN_OR_RETURN(
        module_image, ModuleImage::FromProto(proto.module_image(), proto.ptx(),
                                             proto.compute_capability()));
  }

  std::optional<uint32_t> global_scratch_size;
  if (proto.has_global_scratch_size()) {
    global_scratch_size = proto.global_scratch_size();
  }
  std::optional<uint32_t> global_scratch_align;
  if (proto.has_global_scratch_align()) {
    global_scratch_align = proto.global_scratch_align();
  }

  return Kernel(proto.kernel_name(), proto.num_warps(), num_ctas,
                proto.shared_mem_bytes(), proto.ptx(), proto.ttir(),
                proto.compute_capability(), global_scratch_size,
                global_scratch_align, module_image);
}

jax_triton::TritonKernel Kernel::ToProto() const {
  jax_triton::TritonKernel proto;
  proto.set_kernel_name(kernel_name_);
  proto.set_num_warps(block_dim_x_ / kNumThreadsPerWarp);
  proto.set_num_ctas(num_ctas_);
  proto.set_shared_mem_bytes(shared_mem_bytes_);
  proto.set_ptx(ptx_);
  proto.set_ttir(ttir_);
  proto.set_compute_capability(compute_capability_);
  if (global_scratch_size_.has_value()) {
    proto.set_global_scratch_size(*global_scratch_size_);
  }
  if (global_scratch_align_.has_value()) {
    proto.set_global_scratch_align(*global_scratch_align_);
  }
  if (module_image_ != nullptr) {
    *proto.mutable_module_image() = module_image_->ToProto();
  }
  return proto;
}

/*static*/ absl::StatusOr<KernelCall::Parameter>
KernelCall::Parameter::FromProto(
    const jax_triton::TritonKernelCall_Parameter& proto) {
  using jax_triton::TritonKernelCall_Parameter;
  Parameter param;
  switch (proto.value_case()) {
    case TritonKernelCall_Parameter::kArray:
      param.value = Array{proto.array().bytes_to_zero(),
                          proto.array().ptr_divisibility()};
      break;
    case TritonKernelCall_Parameter::kBool:
      param.value = proto.bool_();
      break;
    case TritonKernelCall_Parameter::kI32:
      param.value = proto.i32();
      break;
    case TritonKernelCall_Parameter::kU32:
      param.value = proto.u32();
      break;
    case TritonKernelCall_Parameter::kI64:
      param.value = proto.i64();
      break;
    case TritonKernelCall_Parameter::kU64:
      param.value = proto.u64();
      break;
    case TritonKernelCall_Parameter::kF32:
      param.value = proto.f32();
      break;
    case TritonKernelCall_Parameter::kF64:
      param.value = proto.f64();
      break;
    case TritonKernelCall_Parameter::kTensorDescriptor: {
      const auto& td = proto.tensor_descriptor();
      if (!td.has_nvidia()) {
        return absl::UnimplementedError(
            "Only NVIDIA TMA tensor descriptors are supported.");
      }
      const auto& d = td.nvidia();
      Parameter::TmaDescriptor desc;
      desc.elem_type = d.elem_type();
      desc.swizzle = d.swizzle();
      desc.shape.assign(d.shape().begin(), d.shape().end());
      desc.strides.assign(d.strides().begin(), d.strides().end());
      desc.block_shape.assign(d.block_shape().begin(), d.block_shape().end());
      desc.oob_fill = d.oob_fill();
      param.value = std::move(desc);
      break;
    }
    default:
      return absl::InvalidArgumentError("Unknown scalar parameter type.");
  }
  return param;
}

bool Kernel::CanLaunchOnDevice(gpuDevice_t device) const {
  return shared_mem_bytes_ <= MaxSharedMemoryPerBlock(device).value_or(0);
}

jax_triton::TritonKernelCall_Parameter KernelCall::Parameter::ToProto() const {
  jax_triton::TritonKernelCall_Parameter proto;
  if (std::holds_alternative<Array>(value)) {
    proto.mutable_array()->set_bytes_to_zero(
        std::get<Array>(value).bytes_to_zero);
    proto.mutable_array()->set_ptr_divisibility(
        std::get<Array>(value).ptr_divisibility);
  } else if (std::holds_alternative<bool>(value)) {
    proto.set_bool_(std::get<bool>(value));
  } else if (std::holds_alternative<int32_t>(value)) {
    proto.set_i32(std::get<int32_t>(value));
  } else if (std::holds_alternative<uint32_t>(value)) {
    proto.set_u32(std::get<uint32_t>(value));
  } else if (std::holds_alternative<int64_t>(value)) {
    proto.set_i64(std::get<int64_t>(value));
  } else if (std::holds_alternative<uint64_t>(value)) {
    proto.set_u64(std::get<uint64_t>(value));
  } else if (std::holds_alternative<float>(value)) {
    proto.set_f32(std::get<float>(value));
  } else if (std::holds_alternative<double>(value)) {
    proto.set_f64(std::get<double>(value));
  } else {
    CHECK(std::holds_alternative<TmaDescriptor>(value));
    const auto& desc = std::get<TmaDescriptor>(value);
    auto* d = proto.mutable_tensor_descriptor()->mutable_nvidia();
    d->set_elem_type(desc.elem_type);
    d->set_swizzle(desc.swizzle);
    d->mutable_shape()->Assign(desc.shape.begin(), desc.shape.end());
    d->mutable_strides()->Assign(desc.strides.begin(), desc.strides.end());
    d->mutable_block_shape()->Assign(desc.block_shape.begin(),
                                     desc.block_shape.end());
    d->set_oob_fill(desc.oob_fill);
  }
  return proto;
}

KernelCall::KernelCall(Kernel kernel, uint32_t grid_0, uint32_t grid_1,
                       uint32_t grid_2, std::vector<Parameter> parameters)
    : kernel_(std::move(kernel)),
      grid_{grid_0, grid_1, grid_2},
      parameters_(std::move(parameters)) {}

namespace {

#if defined(JAX_GPU_CUDA)
absl::StatusOr<uint32_t> GetTmaDataTypeSizeBytes(CUtensorMapDataType type) {
  switch (type) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
      return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      return 8;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported TMA data type: %d.", static_cast<int>(type)));
  }
}

absl::Status EncodeTmaDescriptorTiled(
    const KernelCall::Parameter::TmaDescriptor& desc, void* global_address,
    CUtensorMap* out) {
  if (reinterpret_cast<uintptr_t>(out) % 64 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TMA descriptor output address must be 64-byte aligned, but got %p.",
        out));
  }
  if (reinterpret_cast<uintptr_t>(global_address) % 16 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TMA global base address must be 16-byte aligned, but got %p.",
        global_address));
  }

  const int rank = static_cast<int>(desc.block_shape.size());
  if (rank < 1 || rank > 5) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TMA descriptor rank %d is out of range [1, 5].", rank));
  }
  if (desc.shape.size() != rank || desc.strides.size() != rank) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TMA descriptor rank mismatch: block_shape size %d, shape size %zu, "
        "strides size %zu.",
        rank, desc.shape.size(), desc.strides.size()));
  }

  ABSL_ASSIGN_OR_RETURN(const uint32_t elem_size,
                        GetTmaDataTypeSizeBytes(
                            static_cast<CUtensorMapDataType>(desc.elem_type)));

  for (int i = 0; i < rank; ++i) {
    if (desc.shape[i] < 1 || desc.shape[i] > (1ULL << 32)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TMA shape at dimension %d must be in range [1, 2^32], but got "
          "%llu.",
          i, desc.shape[i]));
    }
    if (desc.block_shape[i] < 1 || desc.block_shape[i] > 256) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TMA block shape at dimension %d must be in range [1, 256], but "
          "got %u.",
          i, desc.block_shape[i]));
    }
  }

  if (desc.strides[rank - 1] != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TMA innermost element stride must be 1, but got %llu.",
        desc.strides[rank - 1]));
  }
  if ((desc.block_shape[rank - 1] * elem_size) % 16 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "TMA innermost block shape in bytes must be divisible by 16, but got "
        "%u * %u = %u bytes.",
        desc.block_shape[rank - 1], elem_size,
        desc.block_shape[rank - 1] * elem_size));
  }

  // Fields in desc are row-major (Triton convention); reverse to column-major
  // for CUDA cuTensorMap.
  uint32_t block_size[5];
  uint64_t shape[5];
  uint64_t strides[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < rank; ++i) {
    block_size[rank - i - 1] = desc.block_shape[i];
    shape[rank - i - 1] = desc.shape[i];
  }
  for (int i = 0; i + 1 < rank; ++i) {
    uint64_t byte_stride = elem_size * desc.strides[i];
    if (byte_stride % 16 != 0 || byte_stride >= (1ULL << 40)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TMA byte stride at dimension %d must be divisible by 16 and less "
          "than 2^40, but got %llu (element stride %llu, element size %u).",
          i, byte_stride, desc.strides[i], elem_size));
    }
    strides[rank - i - 2] = byte_stride;
  }
  strides[rank - 1] =
      shape[rank - 1] * (rank == 1 ? elem_size : strides[rank - 2]);

  CUtensorMapFloatOOBfill fill;
  if (desc.oob_fill == 0) {
    fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  } else if (desc.oob_fill == 1) {
    fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
  } else {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported TMA OOB fill: %u.", desc.oob_fill));
  }

  if (desc.swizzle > 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported TMA swizzle: %u.", desc.swizzle));
  }

  uint32_t element_strides[5] = {1, 1, 1, 1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      out, static_cast<CUtensorMapDataType>(desc.elem_type), rank,
      global_address, shape, strides, block_size, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      static_cast<CUtensorMapSwizzle>(desc.swizzle),
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);
  if (res != CUDA_SUCCESS) {
    const char* str = nullptr;
    cuGetErrorString(res, &str);
    return absl::InternalError(absl::StrCat("Failed to encode TMA descriptor: ",
                                            str ? str : "unknown error"));
  }
  return absl::OkStatus();
}
#endif  // defined(JAX_GPU_CUDA)

}  // namespace

absl::Status KernelCall::Launch(gpuStream_t stream, void** buffers) {
  std::vector<void*> params;
#if defined(JAX_GPU_CUDA)
  // params holds pointers into this container; deque guarantees stability.
  std::deque<CUtensorMap> tma_maps;
#endif
  // +2 accounts for the global scratch buffer and the profiling buffer.
  for (size_t i = 0; i < parameters_.size(); ++i) {
    const Parameter& param = parameters_[i];
    if (std::holds_alternative<Parameter::Array>(param.value)) {
      const auto& array = std::get<Parameter::Array>(param.value);
      void*& ptr = *(buffers++);
      auto cu_ptr = reinterpret_cast<gpuDevicePtr_t>(ptr);

      if (ABSL_PREDICT_FALSE((array.ptr_divisibility != 0) &&
                             ((size_t)cu_ptr % array.ptr_divisibility != 0))) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Parameter %zu (%zu) is not divisible by %d.", i,
                            (size_t)ptr, array.ptr_divisibility));
      }

      if (array.bytes_to_zero > 0) {
        GPU_RETURN_IF_ERROR(
            gpuMemsetD8Async(cu_ptr, 0, array.bytes_to_zero, stream));
      }
      params.push_back(&ptr);
    } else if (std::holds_alternative<Parameter::TmaDescriptor>(param.value)) {
      void* base_ptr = *(buffers++);
#if defined(JAX_GPU_CUDA)
      const auto& desc = std::get<Parameter::TmaDescriptor>(param.value);
      CUtensorMap& map = tma_maps.emplace_back();
      TF_RETURN_IF_ERROR(EncodeTmaDescriptorTiled(desc, base_ptr, &map));
      params.push_back(&map);
#else
      (void)base_ptr;
      return absl::UnimplementedError(
          "Host-side TMA descriptors are only supported on NVIDIA GPUs.");
#endif  // defined(JAX_GPU_CUDA)
    } else {
      params.push_back(const_cast<void*>(std::visit(
          [](auto&& arg) { return reinterpret_cast<const void*>(&arg); },
          param.value)));
    }
  }
  // Allocate per-CTA global scratch buffer if required by the kernel, e.g. for
  // TMA descriptors.
  gpuDevicePtr_t global_scratch = 0;
  if (kernel_.global_scratch_size().has_value()) {
    const uint32_t per_cta_scratch = kernel_.global_scratch_size().value();
    const uint64_t grid_size =
        static_cast<uint64_t>(grid_[0]) * grid_[1] * grid_[2];
    const uint64_t num_ctas = kernel_.num_ctas();
    if (ABSL_PREDICT_FALSE(grid_size != 0 && num_ctas != 0 &&
                           per_cta_scratch >
                               std::numeric_limits<uint64_t>::max() /
                                   grid_size / num_ctas)) {
      return absl::InvalidArgumentError(
          "Triton global scratch buffer size overflow.");
    }
    const uint64_t alloc_size = grid_size * num_ctas * per_cta_scratch;
    if (alloc_size > 0) {
      GPU_RETURN_IF_ERROR(
          gpuMemAllocAsync(&global_scratch, alloc_size, stream));
    }
  }
  absl::Cleanup global_scratch_deleter = [&] {
    if (global_scratch == 0) return;
    absl::Status s = JAX_AS_STATUS(gpuMemFreeAsync(global_scratch, stream));
    if (!s.ok()) {
      LOG(WARNING) << "Failed to free Triton global scratch buffer: " << s;
    }
  };
  if (global_scratch != 0 && kernel_.global_scratch_align().has_value()) {
    const uint32_t align = *kernel_.global_scratch_align();
    if (ABSL_PREDICT_FALSE(align > 1 &&
                           ((uintptr_t)global_scratch % align != 0))) {
      return absl::InternalError(absl::StrFormat(
          "Triton global scratch buffer (%p) is not aligned to %u bytes.",
          (void*)global_scratch, align));
    }
  }
  // Alive until kernel_.Launch returns.
  void* global_scratch_ptr = reinterpret_cast<void*>(global_scratch);
  params.push_back(&global_scratch_ptr);
  void* profiling_buffer = nullptr;  // Alive until kernel_.Launch returns.
  params.push_back(&profiling_buffer);

  return kernel_.Launch(stream, grid_, params.data());
}

/*static*/ absl::StatusOr<KernelCall> KernelCall::FromProto(
    const jax_triton::TritonKernelCall& proto) {
  std::vector<KernelCall::Parameter> parameters;
  for (const jax_triton::TritonKernelCall_Parameter& parameter :
       proto.parameters()) {
    JAX_ASSIGN_OR_RETURN(Parameter p, Parameter::FromProto(parameter));
    parameters.push_back(p);
  }
  JAX_ASSIGN_OR_RETURN(Kernel kernel, Kernel::FromProto(proto.kernel()));
  return KernelCall(std::move(kernel), proto.grid_0(), proto.grid_1(),
                    proto.grid_2(), std::move(parameters));
}

jax_triton::TritonKernelCall KernelCall::ToProto() const {
  jax_triton::TritonKernelCall proto;
  *proto.mutable_kernel() = kernel_.ToProto();
  proto.set_grid_0(grid_[0]);
  proto.set_grid_1(grid_[1]);
  proto.set_grid_2(grid_[2]);
  for (const Parameter& param : parameters_) {
    *proto.add_parameters() = param.ToProto();
  }
  return proto;
}

bool KernelCall::CanLaunchOnDevice(gpuDevice_t device) const {
  return kernel_.CanLaunchOnDevice(device);
}

AutotunedKernelCall::AutotunedKernelCall(
    std::string name, std::vector<Config> configs,
    std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases)
    : name_(std::move(name)),
      configs_(std::move(configs)),
      input_output_aliases_(std::move(input_output_aliases)) {}

/*static*/ absl::StatusOr<AutotunedKernelCall> AutotunedKernelCall::FromProto(
    const jax_triton::TritonAutotunedKernelCall& proto) {
  std::vector<Config> configs;
  for (const jax_triton::TritonAutotunedKernelCall_Config& config :
       proto.configs()) {
    JAX_ASSIGN_OR_RETURN(auto kernel_call,
                         KernelCall::FromProto(config.kernel_call()));
    configs.push_back(Config{std::move(kernel_call), config.description()});
  }

  std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases;
  for (const jax_triton::TritonAutotunedKernelCall_InputOutputAlias& a :
       proto.input_output_aliases()) {
    input_output_aliases.push_back(std::make_tuple(
        a.input_buffer_idx(), a.output_buffer_idx(), a.buffer_size_bytes()));
  }

  return AutotunedKernelCall(proto.name(), std::move(configs),
                             std::move(input_output_aliases));
}

jax_triton::TritonAutotunedKernelCall AutotunedKernelCall::ToProto() const {
  jax_triton::TritonAutotunedKernelCall proto;
  proto.set_name(name_);
  for (const Config& config : configs_) {
    jax_triton::TritonAutotunedKernelCall_Config* c = proto.add_configs();
    *c->mutable_kernel_call() = config.kernel_call.ToProto();
    c->set_description(config.description);
  }
  for (const auto& [input_idx, output_idx, size] : input_output_aliases_) {
    jax_triton::TritonAutotunedKernelCall_InputOutputAlias* a =
        proto.add_input_output_aliases();
    a->set_input_buffer_idx(input_idx);
    a->set_output_buffer_idx(output_idx);
    a->set_buffer_size_bytes(size);
  }
  return proto;
}

/*static*/ absl::StatusOr<KernelCall> AutotunedKernelCall::Autotune(
    AutotunedKernelCall kernel_call, gpuStream_t stream, void** buffers) {
  // Ensure a valid context for driver calls that don't take the stream.
  // gpuContext_t context;
  // GPU_RETURN_IF_ERROR(gpuStreamGetCtx(stream, &context));
  // GPU_RETURN_IF_ERROR(gpuCtxPushCurrent(context));
  // absl::Cleanup ctx_restorer = [] { gpuCtxPopCurrent(nullptr); };

  // Autotuning is not supported if the stream is in graph capture mode.
  gpustreamCaptureStatus_t capture_status;
  GPU_RETURN_IF_ERROR(gpuStreamIsCapturing(stream, &capture_status));
  if (capture_status == GPU_STREAM_CAPTURE_STATUS_ACTIVE) {
    return absl::FailedPreconditionError(
        "Can't autotune Triton kernel when the stream is in graph capture "
        "mode. Autotuning can rely on real data present in input buffers to "
        "use them in address computation, but in graph capture mode buffers "
        "can have arbitrary data");
  }

  // If an input aliases with an output, it will get overwritten during the
  // kernel execution. If the kernel is called repeatedly, as we do during
  // auto-tuning, the final result will be junk, so we take a copy of the
  // input to restore after auto-tuning.
  std::vector<std::pair<size_t, std::vector<uint8_t>>> input_copies;
  for (auto [input_idx, output_idx, size] : kernel_call.input_output_aliases_) {
    if (buffers[input_idx] == buffers[output_idx]) {
      std::vector<uint8_t> input_copy(size);
      GPU_RETURN_IF_ERROR(gpuMemcpyDtoHAsync(
          input_copy.data(),
          reinterpret_cast<gpuDevicePtr_t>(buffers[input_idx]), size, stream));
      input_copies.push_back({input_idx, std::move(input_copy)});
    }
  }

  LOG(INFO) << "Autotuning function: " << kernel_call.name_;
  // First run a single iteration of each to config to determine how many
  // iterations to run for benchmarking.
  float best = std::numeric_limits<float>::infinity();
  JAX_ASSIGN_OR_RETURN(gpuDevice_t device, GetStreamDevice(stream));
  absl::flat_hash_set<Config*> configs_to_skip;
  for (Config& config : kernel_call.configs_) {
    if (!config.kernel_call.CanLaunchOnDevice(device)) {
      configs_to_skip.insert(&config);
      continue;
    }
    JAX_ASSIGN_OR_RETURN(float t,
                         Benchmark(stream, config.kernel_call, buffers, 1));
    LOG(INFO) << config.description << ", ran 1 iter in " << t << " ms";
    best = std::min(best, t);
  }

  int timed_iters = std::max(static_cast<int>(kBenchmarkTimeMillis / best), 1);
  if (timed_iters > 100) {
    timed_iters = 100;
    LOG(INFO) << "Benchmarking with 100 iters (capped at 100)";
  } else {
    timed_iters = std::min(timed_iters, 100);
    LOG(INFO) << "Benchmarking with " << timed_iters
              << " iters (target time: " << kBenchmarkTimeMillis << " ms)";
  }

  best = std::numeric_limits<float>::infinity();
  for (Config& config : kernel_call.configs_) {
    if (configs_to_skip.contains(&config)) {
      LOG(WARNING) << "Unable to launch autotune config on device: "
                   << config.description;
      continue;
    }

    JAX_ASSIGN_OR_RETURN(
        float t, Benchmark(stream, config.kernel_call, buffers, timed_iters));
    LOG(INFO) << config.description << ", ran " << timed_iters << " iters in "
              << t << " ms";

    if (t < best) {
      LOG(INFO) << config.description << " is the new best config";
      best = t;
      std::swap(config, kernel_call.configs_[0]);
    }
  }
  if (std::isinf(best)) {
    LOG(WARNING) << "Finished autotuning function: " << kernel_call.name_
                 << " no valid configs found.";
    return absl::FailedPreconditionError("No launchable configs.");
  }

  LOG(INFO) << "Finished autotuning function: " << kernel_call.name_
            << " best config " << kernel_call.configs_[0].description;

  // Restore aliased inputs to their original values.
  for (const auto& [input_idx, input_copy] : input_copies) {
    GPU_RETURN_IF_ERROR(
        gpuMemcpyHtoDAsync(reinterpret_cast<gpuDevicePtr_t>(buffers[input_idx]),
                           input_copy.data(), input_copy.size(), stream));
  }

  // Synchronize stream to ensure copies are complete before the host copy
  // is deleted.
  GPU_RETURN_IF_ERROR(gpuStreamSynchronize(stream));

  return std::move(kernel_call.configs_[0].kernel_call);
}

void TritonKernelCall(gpuStream_t stream, void** buffers, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  absl::Status result = [=] {
    JAX_ASSIGN_OR_RETURN(
        KernelCall * kernel_call,
        GetKernelCall(std::string_view(opaque, opaque_len), stream, buffers));
    return kernel_call->Launch(stream, buffers);
  }();
  if (!result.ok()) {
    std::string_view msg = result.message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
  }
}

static absl::StatusOr<std::vector<void*>> CombineBuffers(
    ::xla::ffi::RemainingArgs args, ::xla::ffi::RemainingRets rets) {
  std::vector<void*> buffers;
  buffers.reserve(args.size() + rets.size());
  for (size_t i = 0; i < args.size(); ++i) {
    JAX_ASSIGN_OR_RETURN(::xla::ffi::AnyBuffer buf,
                         args.get<::xla::ffi::AnyBuffer>(i));
    buffers.push_back(buf.untyped_data());
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    JAX_ASSIGN_OR_RETURN(::xla::ffi::Result<::xla::ffi::AnyBuffer> buf,
                         rets.get<::xla::ffi::AnyBuffer>(i));
    buffers.push_back(buf->untyped_data());
  }
  return buffers;
}

// Launches the kernel call previously compiled and cached.
absl::Status TritonKernelCallFfi(
    gpuStream_t stream, TritonKernelInitializeResult* initialized_kernel_call,
    ::xla::ffi::RemainingArgs args, ::xla::ffi::RemainingRets rets,
    ::xla::ffi::Dictionary attrs) {
  // The state should always be non-null and have a valid kernel call, but just
  // in case.
  if (initialized_kernel_call == nullptr) {
    return absl::InvalidArgumentError("Initialized kernel call is null.");
  }
  if (initialized_kernel_call->kernel_call == nullptr) {
    return absl::InvalidArgumentError("Kernel call is null.");
  }

  JAX_ASSIGN_OR_RETURN(std::vector<void*> buffers, CombineBuffers(args, rets));
  return initialized_kernel_call->kernel_call->Launch(stream, buffers.data());
}

// Autotunes the kernel if needed, and populates the kernel cache.
// Because of command buffer support, we need to make sure that the kernel
// cache is populated during initialization, and not during execution.
absl::StatusOr<std::unique_ptr<TritonKernelInitializeResult>>
TritonKernelCallFfiInitialize(gpuStream_t stream,
                              TritonKernelInstantiateResult* instantiate_result,
                              ::xla::ffi::RemainingArgs args,
                              ::xla::ffi::RemainingRets rets,
                              ::xla::ffi::Dictionary attrs) {
  // Instantiate always runs before initialize, so this should never be null.
  if (instantiate_result == nullptr) {
    return absl::InvalidArgumentError("State is null.");
  }

  // Creates the KernelCall using GetOrCreateKernelCall so that results are
  // cached.
  auto create_kernel_call = [&]() -> absl::StatusOr<KernelCall> {
    switch (instantiate_result->proto.call_case()) {
      case jax_triton::TritonCustomCallStateProto::kKernelCall: {
        return KernelCall::FromProto(instantiate_result->proto.kernel_call());
      }
      case jax_triton::TritonCustomCallStateProto::
          kAutotuningKernelCandidates: {
        JAX_ASSIGN_OR_RETURN(
            AutotunedKernelCall autotuned_call,
            AutotunedKernelCall::FromProto(
                instantiate_result->proto.autotuning_kernel_candidates()));
        JAX_ASSIGN_OR_RETURN(std::vector<void*> buffers,
                             CombineBuffers(args, rets));
        // The returned KernelCall is fully compiled down to machine code, and
        // thus ready to be executed.
        return AutotunedKernelCall::Autotune(std::move(autotuned_call), stream,
                                             buffers.data());
      }
      default:
        return absl::InvalidArgumentError("Unknown kernel call type.");
    }
  };
  // We only use opaque as a key for the kernel call cache.
  JAX_ASSIGN_OR_RETURN(std::string_view opaque,
                       attrs.get<std::string_view>("opaque"));
  JAX_ASSIGN_OR_RETURN(KernelCall * kernel_call,
                       GetOrCreateKernelCall(opaque, create_kernel_call));
  return std::make_unique<TritonKernelInitializeResult>(kernel_call);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kTritonKernelCallFfi, TritonKernelCallFfi,
    ::xla::ffi::Ffi::Bind()
        .Ctx<::xla::ffi::PlatformStream<gpuStream_t>>()
        .Ctx<::xla::ffi::Initialized<TritonKernelInitializeResult>>()
        .RemainingArgs()
        .RemainingRets()
        .Attrs(),
    {::xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kTritonKernelCallFfiInitialize, TritonKernelCallFfiInitialize,
    ::xla::ffi::Ffi::BindInitialize()
        .Ctx<::xla::ffi::PlatformStream<gpuStream_t>>()
        .Ctx<::xla::ffi::State<TritonKernelInstantiateResult>>()
        .RemainingArgs()
        .RemainingRets()
        .Attrs(),
    {::xla::ffi::Traits::kCmdBufferCompatible});

}  // namespace jax::JAX_GPU_NAMESPACE

namespace xla::ffi {

template <>
struct TypeRegistry::SerDes<
    jax::JAX_GPU_NAMESPACE::TritonKernelInstantiateResult>
    : public std::true_type {
  static absl::StatusOr<std::string> Serialize(
      const jax::JAX_GPU_NAMESPACE::TritonKernelInstantiateResult&
          instantiate_result) {
    return jax::JAX_GPU_NAMESPACE::TritonKernelInstantiateResult::Serialize(
        instantiate_result);
  }
  static absl::StatusOr<
      std::unique_ptr<jax::JAX_GPU_NAMESPACE::TritonKernelInstantiateResult>>
  Deserialize(absl::string_view data) {
    return jax::JAX_GPU_NAMESPACE::TritonKernelInstantiateResult::Deserialize(
        data);
  }
};

}  // namespace xla::ffi

namespace jax::JAX_GPU_NAMESPACE {

// Compiles the kernel down to machine code (e.g. CUBIN) and stores it in the
// TritonKernelInstantiateResult. In case of AutotunedKernels it compiles all
// the candidates.
absl::StatusOr<std::unique_ptr<TritonKernelInstantiateResult>>
TritonKernelCallFfiInstantiate(const stream_executor::GpuComputeCapability* cc,
                               ::xla::ffi::Dictionary attrs) {
  JAX_ASSIGN_OR_RETURN(std::string_view opaque,
                       attrs.get<std::string_view>("opaque"));
  JAX_ASSIGN_OR_RETURN(std::string serialized, ZlibUncompress(opaque));

  jax_triton::TritonAnyKernelCall proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InvalidArgumentError("Failed to parse serialized data.");
  }

  auto instantiate_result = std::make_unique<TritonKernelInstantiateResult>();

  switch (proto.value_case()) {
    case jax_triton::TritonAnyKernelCall::kKernelCall:
      *instantiate_result->proto.mutable_kernel_call() =
          std::move(*proto.mutable_kernel_call());
      JAX_RETURN_IF_ERROR(CompileKernelProto(
          cc,
          instantiate_result->proto.mutable_kernel_call()->mutable_kernel()));
      break;
    case jax_triton::TritonAnyKernelCall::kAutotunedKernelCall:
      *instantiate_result->proto.mutable_autotuning_kernel_candidates() =
          std::move(*proto.mutable_autotuned_kernel_call());

      for (jax_triton::TritonAutotunedKernelCall::Config& config :
           *instantiate_result->proto.mutable_autotuning_kernel_candidates()
                ->mutable_configs()) {
        JAX_RETURN_IF_ERROR(CompileKernelProto(
            cc, config.mutable_kernel_call()->mutable_kernel()));
      }
      break;
    default:
      return absl::InvalidArgumentError("Unknown kernel call type.");
  }

  return instantiate_result;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kTritonKernelCallFfiInstantiate,
                              TritonKernelCallFfiInstantiate,
                              ::xla::ffi::Ffi::BindInstantiate()
                                  .Ctx<::xla::ffi::TargetGpuComputeCapability>()
                                  .Attrs());

}  // namespace jax::JAX_GPU_NAMESPACE
