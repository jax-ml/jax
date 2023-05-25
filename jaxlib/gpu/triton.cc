#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep

#define CUDA_RETURN_IF_ERROR(expr) JAX_RETURN_IF_ERROR(JAX_AS_STATUS(expr))

namespace py = pybind11;

namespace jax::JAX_GPU_NAMESPACE {

// TODO(cjfj): Move this to `gpu_kernel_helpers`?
// Used via JAX_AS_STATUS(expr) macro.
absl::Status AsStatus(CUresult error, const char* file, std::int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_TRUE(error == CUDA_SUCCESS)) {
    return absl::OkStatus();
  }

  const char* str;
  CHECK_EQ(cuGetErrorName(error, &str), CUDA_SUCCESS);
  return absl::InternalError(
      absl::StrFormat("%s:%d: operation %s failed: %s", file, line, expr, str));
}

}  // namespace jax::JAX_GPU_NAMESPACE

namespace jax_triton {
namespace {

constexpr uint32_t kNumThreadsPerWarp = 32;

struct CuModuleDeleter {
  void operator()(CUmodule module) { cuModuleUnload(module); }
};

using OwnedCUmodule =
    std::unique_ptr<std::remove_pointer_t<CUmodule>, CuModuleDeleter>;

class TritonKernel {
 public:
  TritonKernel(std::string module_image, std::string kernel_name,
               uint32_t num_warps, uint32_t shared_mem_bytes)
      : module_image_(std::move(module_image)),
        kernel_name_(std::move(kernel_name)),
        block_dim_x_(num_warps * kNumThreadsPerWarp),
        shared_mem_bytes_(shared_mem_bytes) {}

  absl::Status Launch(CUstream stream, uint32_t grid[3], void** params) {
    CUcontext context;
    CUDA_RETURN_IF_ERROR(cuStreamGetCtx(stream, &context));
    absl::StatusOr<CUfunction> kernel = GetFunctionForContext(context);
    JAX_RETURN_IF_ERROR(kernel.status());
    return JAX_AS_STATUS(cuLaunchKernel(
        *kernel, grid[0], grid[1], grid[2], block_dim_x_,
        /*blockDimY=*/1, /*blockDimZ=*/1, shared_mem_bytes_, stream, params,
        /*extra=*/nullptr));
  }

 private:
  absl::StatusOr<CUfunction> GetFunctionForContext(CUcontext context) {
    absl::MutexLock lock(&mutex_);
    auto it = functions_.find(context);
    if (it != functions_.end()) {
      return it->second;
    }

    CUDA_RETURN_IF_ERROR(cuCtxPushCurrent(context));
    absl::Cleanup ctx_restorer = [] { cuCtxPopCurrent(nullptr); };

    CUmodule module;
    CUDA_RETURN_IF_ERROR(cuModuleLoadData(&module, module_image_.c_str()));
    modules_.push_back(OwnedCUmodule(module, CuModuleDeleter()));

    CUfunction function;
    CUDA_RETURN_IF_ERROR(
        cuModuleGetFunction(&function, module, kernel_name_.c_str()));
    auto [_, success] = functions_.insert({context, function});
    CHECK(success);

    // The maximum permitted static shared memory allocation in CUDA is 48kB,
    // but we can expose more to the kernel using dynamic shared memory.
    constexpr int kMaxStaticSharedMemBytes = 49152;
    if (shared_mem_bytes_ <= kMaxStaticSharedMemBytes) {
      return function;
    }

    // Set up dynamic shared memory.
    CUdevice device;
    CUDA_RETURN_IF_ERROR(cuCtxGetDevice(&device));

    int shared_optin;
    CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
        &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));

    if (shared_optin > kMaxStaticSharedMemBytes) {
      CUDA_RETURN_IF_ERROR(
          cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED));
      int shared_total;
      CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
          &shared_total,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));
      int shared_static;
      CUDA_RETURN_IF_ERROR(cuFuncGetAttribute(
          &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
      CUDA_RETURN_IF_ERROR(cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          shared_optin - shared_static));
    }
    return function;
  }

  std::string module_image_;
  std::string kernel_name_;
  uint32_t block_dim_x_;
  uint32_t shared_mem_bytes_;

  absl::Mutex mutex_;
  std::vector<OwnedCUmodule> modules_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<CUcontext, CUfunction> functions_ ABSL_GUARDED_BY(mutex_);
};

struct TritonKernelCallBase {
  virtual ~TritonKernelCallBase() = default;
  virtual absl::Status Launch(CUstream stream, void** buffers) = 0;
};

class TritonKernelCall : public TritonKernelCallBase {
 public:
  struct ArrayParameter {
    size_t bytes_to_zero;
    bool ptr_must_be_divisible_by_16;
  };

  // Parameters can be either to either arrays or scalars (encoded as uint64).
  using Parameter = std::variant<ArrayParameter, uint64_t>;

  TritonKernelCall(TritonKernel& kernel, uint32_t grid_0, uint32_t grid_1,
                   uint32_t grid_2, std::vector<Parameter> parameters)
      : kernel_(kernel),
        grid_{grid_0, grid_1, grid_2},
        parameters_(std::move(parameters)) {}

  absl::Status Launch(CUstream stream, void** buffers) override final {
    std::vector<void*> params;
    params.reserve(parameters_.size());
    for (size_t i = 0; i < parameters_.size(); ++i) {
      const Parameter& param = parameters_[i];
      if (std::holds_alternative<ArrayParameter>(param)) {
        const ArrayParameter& array = std::get<ArrayParameter>(param);
        void*& ptr = *(buffers++);
        auto cu_ptr = reinterpret_cast<CUdeviceptr>(ptr);

        if (array.ptr_must_be_divisible_by_16 && (cu_ptr % 16 != 0)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Parameter %zu (%p) is not divisible by 16.", i, ptr));
        }

        if (array.bytes_to_zero > 0) {
          CUDA_RETURN_IF_ERROR(
              cuMemsetD8Async(cu_ptr, 0, array.bytes_to_zero, stream));
        }
        params.push_back(&ptr);
      } else {
        params.push_back(const_cast<uint64_t*>(&std::get<uint64_t>(param)));
      }
    }

    return kernel_.Launch(stream, grid_, params.data());
  }

 private:
  TritonKernel& kernel_;
  uint32_t grid_[3];
  std::vector<Parameter> parameters_;
};

class TritonAutotunedKernelCall : public TritonKernelCallBase {
 public:
  struct Config {
    py::object kernel_call;
    std::string description;
  };

  TritonAutotunedKernelCall(
      std::string name, std::vector<Config> configs,
      std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases)
      : name_(std::move(name)),
        configs_(std::move(configs)),
        input_output_aliases_(std::move(input_output_aliases)) {}

  absl::Status Launch(CUstream stream, void** buffers) override {
    absl::call_once(autotune_once_, [=]() {
      if (configs_.size() > 1) {
        autotune_status_ = Autotune(stream, buffers);
      }
    });
    JAX_RETURN_IF_ERROR(autotune_status_);
    auto& kernel_call = py::cast<TritonKernelCall&>(configs_[0].kernel_call);
    return kernel_call.Launch(stream, buffers);
  }

 private:
  static constexpr float kBenchmarkTimeMillis = 10.;

  absl::Status Autotune(CUstream stream, void** buffers) {
    // Ensure a valid context for driver calls that don't take the stream.
    CUcontext context;
    CUDA_RETURN_IF_ERROR(cuStreamGetCtx(stream, &context));
    CUDA_RETURN_IF_ERROR(cuCtxPushCurrent(context));
    absl::Cleanup ctx_restorer = [] { cuCtxPopCurrent(nullptr); };

    // If an input aliases with an output, it will get overwritten during the
    // kernel execution. If the kernel is called repeatedly, as we do during
    // auto-tuning, the final result will be junk, so we take a copy of the
    // input to restore after auto-tuning.
    std::unordered_map<size_t, std::vector<uint8_t>> input_copies;
    for (auto [input_idx, output_idx, size] : input_output_aliases_) {
      if (buffers[input_idx] == buffers[output_idx]) {
        std::vector<uint8_t> input_copy(size);
        CUDA_RETURN_IF_ERROR(cuMemcpyDtoHAsync(
            input_copy.data(),
            reinterpret_cast<CUdeviceptr>(buffers[input_idx]), size, stream));
        input_copies[input_idx] = std::move(input_copy);
      }
    }

    LOG(INFO) << "Autotuning function: " << name_;
    // First run a single iteration of each to config to determine how many
    // iterations to run for benchmarking.
    float best = std::numeric_limits<float>::infinity();
    for (Config& config : configs_) {
      auto& kernel_call = py::cast<TritonKernelCall&>(config.kernel_call);
      absl::StatusOr<float> t = Benchmark(stream, kernel_call, buffers, 1);
      JAX_RETURN_IF_ERROR(t.status());
      LOG(INFO) << config.description << ", ran 1 iter in " << *t << " ms";
      best = std::min(best, *t);
    }

    int timed_iters =
        std::max(static_cast<int>(kBenchmarkTimeMillis / best), 1);
    if (timed_iters > 100) {
      timed_iters = 100;
      LOG(INFO) << "Benchmarking with 100 iters (capped at 100)";
    } else {
      timed_iters = std::min(timed_iters, 100);
      LOG(INFO) << "Benchmarking with " << timed_iters
                << " iters (target time: " << kBenchmarkTimeMillis << " ms)";
    }

    best = std::numeric_limits<float>::infinity();
    for (Config& config : configs_) {
      auto& kernel_call = py::cast<TritonKernelCall&>(config.kernel_call);
      absl::StatusOr<float> t =
          Benchmark(stream, kernel_call, buffers, timed_iters);
      JAX_RETURN_IF_ERROR(t.status());
      LOG(INFO) << config.description << ", ran " << timed_iters << " iters in "
                << *t << " ms";

      if (*t < best) {
        LOG(INFO) << config.description << " is the new best config";
        best = *t;
        std::swap(config, configs_[0]);
      }
    }

    // Discard all but the best config.
    py::gil_scoped_acquire gil;
    configs_.erase(configs_.begin() + 1, configs_.end());

    // Restore aliased inputs to their original values.
    for (auto [input_idx, _, size] : input_output_aliases_) {
      CUDA_RETURN_IF_ERROR(
          cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(buffers[input_idx]),
                            input_copies[input_idx].data(), size, stream));
    }
    // Synchronize stream to ensure copies are complete before the host copy
    // is deleted.
    return JAX_AS_STATUS(cuStreamSynchronize(stream));
  }

  absl::StatusOr<float> Benchmark(CUstream stream,
                                  TritonKernelCall& kernel_call, void** buffers,
                                  int num_iterations) {
    CUevent start, stop;
    CUDA_RETURN_IF_ERROR(cuEventCreate(&start, /*Flags=*/CU_EVENT_DEFAULT));
    CUDA_RETURN_IF_ERROR(cuEventCreate(&stop, /*Flags=*/CU_EVENT_DEFAULT));
    JAX_RETURN_IF_ERROR(kernel_call.Launch(stream, buffers));  // Warm-up.
    CUDA_RETURN_IF_ERROR(cuEventRecord(start, stream));
    for (int i = 0; i < num_iterations; ++i) {
      JAX_RETURN_IF_ERROR(kernel_call.Launch(stream, buffers));
    }
    CUDA_RETURN_IF_ERROR(cuEventRecord(stop, stream));
    CUDA_RETURN_IF_ERROR(cuEventSynchronize(stop));
    float elapsed_ms;
    CUDA_RETURN_IF_ERROR(cuEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_RETURN_IF_ERROR(cuEventDestroy(start));
    CUDA_RETURN_IF_ERROR(cuEventDestroy(stop));
    return elapsed_ms;
  }

  std::string name_;
  // After auto-tuning, all configurations, except the best, will be discarded.
  std::vector<Config> configs_;
  // (input buffer idx, output buffer idx, size)
  std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases_;
  absl::once_flag autotune_once_;
  absl::Status autotune_status_;
};

template <typename CppT, typename PyT>
uint64_t EncodeKernelParameterAs(PyT value) {
  static_assert(sizeof(CppT) <= sizeof(uint64_t));
  union {
    CppT value;
    uint64_t bits;
  } encoded;
  encoded.bits = 0;
  encoded.value = CppT(value);
  return encoded.bits;
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::int_ value,
                                               std::string_view dtype) {
  if ((dtype == "i1") || (dtype == "i8")) {
    return EncodeKernelParameterAs<int8_t>(value);
  } else if (dtype == "u8") {
    return EncodeKernelParameterAs<uint8_t>(value);
  } else if (dtype == "i16") {
    return EncodeKernelParameterAs<int16_t>(value);
  } else if (dtype == "u16") {
    return EncodeKernelParameterAs<uint16_t>(value);
  } else if (dtype == "i32") {
    return EncodeKernelParameterAs<int32_t>(value);
  } else if (dtype == "u32") {
    return EncodeKernelParameterAs<uint32_t>(value);
  } else if (dtype == "i64") {
    return EncodeKernelParameterAs<int64_t>(value);
  } else if (dtype == "u64") {
    return EncodeKernelParameterAs<uint64_t>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::float_ value,
                                               std::string_view dtype) {
  if (dtype == "fp32") {
    return EncodeKernelParameterAs<float>(value);
  } else if (dtype == "fp64") {
    return EncodeKernelParameterAs<double>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::bool_ value,
                                               std::string_view dtype) {
  if ((dtype == "int1") || (dtype == "B")) {
    return EncodeKernelParameterAs<bool>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

}  // namespace

void LaunchTritonKernel(CUstream stream, void** buffers, char* opaque,
                        size_t opaque_len) {
  CHECK_EQ(opaque_len, sizeof(TritonKernelCallBase*));
  TritonKernelCallBase* kernel_call;
  std::memcpy(&kernel_call, opaque, sizeof(TritonKernelCallBase*));
  absl::Status status = kernel_call->Launch(stream, buffers);
  LOG_IF(FATAL, !status.ok()) << status;  // TODO(cjfj): Return the `Status`.
}

PYBIND11_MODULE(_triton, m) {
  py::class_<TritonKernel>(m, "TritonKernel")
      .def(py::init<std::string, std::string, uint32_t, uint32_t>());

  py::class_<TritonKernelCall>(m, "TritonKernelCall")
      .def(py::init<TritonKernel&, uint32_t, uint32_t, uint32_t,
                    std::vector<TritonKernelCall::Parameter>>(),
           py::keep_alive<1, 2>())  // Ensure that the kernel lives long enough.
      .def_property_readonly("descriptor", [](TritonKernelCall& kernel_call) {
        union {
          TritonKernelCall* ptr;
          char bytes[sizeof(TritonKernelCall*)];
        } descriptor;
        descriptor.ptr = &kernel_call;
        return py::bytes(descriptor.bytes, sizeof(TritonKernelCall*));
      });

  py::class_<TritonKernelCall::ArrayParameter>(m, "TritonArrayParameter");

  py::class_<TritonAutotunedKernelCall>(m, "TritonAutotunedKernelCall")
      .def(py::init<>([](std::string name,
                         std::vector<std::pair<py::object, std::string>>
                             calls_and_descriptions,
                         std::vector<std::tuple<size_t, size_t, size_t>>
                             input_output_aliases) {
        std::vector<TritonAutotunedKernelCall::Config> configs;
        configs.reserve(calls_and_descriptions.size());
        for (auto& [kernel_call, desc] : calls_and_descriptions) {
          configs.push_back({std::move(kernel_call), std::move(desc)});
        }
        return std::make_unique<TritonAutotunedKernelCall>(
            std::move(name), std::move(configs),
            std::move(input_output_aliases));
      }))
      .def_property_readonly(
          "descriptor", [](TritonAutotunedKernelCall& kernel_call) {
            union {
              TritonAutotunedKernelCall* ptr;
              char bytes[sizeof(TritonAutotunedKernelCall*)];
            } descriptor;
            descriptor.ptr = &kernel_call;
            return py::bytes(descriptor.bytes,
                             sizeof(TritonAutotunedKernelCall*));
          });

  m.def("get_custom_call", [] {
    return py::capsule(reinterpret_cast<void*>(&LaunchTritonKernel),
                       "xla._CUSTOM_CALL_TARGET");
  });

  m.def("create_array_parameter",
        [](size_t bytes_to_zero, bool ptr_must_be_divisible_by_16) {
          return TritonKernelCall::ArrayParameter{bytes_to_zero,
                                                  ptr_must_be_divisible_by_16};
        });
  m.def("create_scalar_parameter",
        py::overload_cast<py::int_, std::string_view>(&EncodeKernelParameter));
  m.def(
      "create_scalar_parameter",
      py::overload_cast<py::float_, std::string_view>(&EncodeKernelParameter));
  m.def("create_scalar_parameter",
        py::overload_cast<py::bool_, std::string_view>(&EncodeKernelParameter));
  m.def("get_compute_capability", [](int device) -> absl::StatusOr<int> {
    int major, minor;
    CUDA_RETURN_IF_ERROR(cuInit(device));
    CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    return major * 10 + minor;
  });
}

}  // namespace jax_triton
