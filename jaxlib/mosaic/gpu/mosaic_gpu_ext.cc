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

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/absl_status_casters.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/mosaic/gpu/target.h"

namespace jax::cuda {
namespace {

namespace nb = nanobind;

#define THROW(...)                                                 \
  do {                                                             \
    throw std::runtime_error(                                      \
        absl::StrCat("Mosaic GPU profiler error: ", __VA_ARGS__)); \
  } while (0)

#define THROW_IF(expr, ...)       \
  do {                            \
    if (expr) THROW(__VA_ARGS__); \
  } while (0)

#define THROW_IF_CUPTI_ERROR(expr, ...)          \
  do {                                           \
    CUptiResult _result = (expr);                \
    if (_result != CUPTI_SUCCESS) {              \
      const char* s;                             \
      cuptiGetErrorMessage(_result, &s);         \
      THROW(s, ": " __VA_OPT__(, ) __VA_ARGS__); \
    }                                            \
  } while (0)

using CuptiBuffersCallbackRequestFuncV2 = void(CUPTIAPI*)(
    uint8_t**, size_t*, size_t*, void*);
using CuptiBuffersCallbackCompleteFuncV2 = void(CUPTIAPI*)(
    uint8_t*, size_t, size_t, void*);

// V2 availability is determined by weak runtime-symbol checks in
// CanUseCuptiV2(). These aliases only make the weak declarations below match
// the CUPTI headers used to build jaxlib.
#ifdef CUpti_ActivityConfig_STRUCT_SIZE
using CuptiActivityConfigAbi = CUpti_ActivityConfig;
using CuptiBuffersCallbackRequestFuncV2Abi =
    CUpti_BuffersCallbackRequestFunc_v2;
using CuptiBuffersCallbackCompleteFuncV2Abi =
    CUpti_BuffersCallbackCompleteFunc_v2;
#else
using CuptiActivityConfigAbi = void;
using CuptiBuffersCallbackRequestFuncV2Abi = CuptiBuffersCallbackRequestFuncV2;
using CuptiBuffersCallbackCompleteFuncV2Abi =
    CuptiBuffersCallbackCompleteFuncV2;
#endif

#ifdef CUpti_SubscriberParams_STRUCT_SIZE
using CuptiSubscriberParamsAbi = CUpti_SubscriberParams;
constexpr size_t kCuptiSubscriberParamsStructSize =
    CUpti_SubscriberParams_STRUCT_SIZE;
#else
struct CuptiSubscriberParamsAbi {
  size_t structSize;
  const char* subscriberName;
  char* oldSubscriberName;
  size_t oldSubscriberSize;
  uint8_t allowMultipleSubscribers;
  uint8_t padding[7];
};
constexpr size_t kCuptiSubscriberParamsStructSize =
    sizeof(CuptiSubscriberParamsAbi);
#endif

template <typename Params, typename = void>
struct HasAllowMultipleSubscribers : std::false_type {};

template <typename Params>
struct HasAllowMultipleSubscribers<
    Params,
    std::void_t<decltype(std::declval<Params&>().allowMultipleSubscribers)>>
    : std::true_type {};

template <typename Params>
bool SetAllowMultipleSubscribersIfSupported(Params* params) {
  if constexpr (HasAllowMultipleSubscribers<Params>::value) {
    params->allowMultipleSubscribers = 1;
    return true;
  }
  return false;
}

extern "C" {
[[gnu::weak]] CUptiResult cuptiActivityRegisterCallbacks_v2(
    CUpti_SubscriberHandle subscriber,
    CuptiBuffersCallbackRequestFuncV2Abi request,
    CuptiBuffersCallbackCompleteFuncV2Abi complete);
[[gnu::weak]] CUptiResult cuptiActivityEnable_v2(
    CUpti_SubscriberHandle subscriber, CUpti_ActivityKind kind,
    CuptiActivityConfigAbi* config);
[[gnu::weak]] CUptiResult cuptiActivityDisable_v2(
    CUpti_SubscriberHandle subscriber, CUpti_ActivityKind kind,
    CuptiActivityConfigAbi* config);
[[gnu::weak]] CUptiResult cuptiActivityGetNextRecord_v2(
    CUpti_SubscriberHandle subscriber, uint8_t* buffer, size_t valid_size,
    CUpti_Activity** record);
[[gnu::weak]] CUptiResult cuptiGetTimestamp_v2(
    CUpti_SubscriberHandle subscriber, uint64_t* timestamp);
[[gnu::weak]] CUptiResult cuptiSubscribe_v2(
    CUpti_SubscriberHandle* subscriber, CUpti_CallbackFunc callback,
    void* userdata, CuptiSubscriberParamsAbi* params);
}

enum class CuptiApi { kV1, kV2 };

class CuptiSubscriber {
 public:
  CuptiSubscriber() = default;
  CuptiSubscriber(CUpti_SubscriberHandle handle, CuptiApi api)
      : handle_(handle), api_(api) {}
  CuptiSubscriber(const CuptiSubscriber&) = delete;
  CuptiSubscriber& operator=(const CuptiSubscriber&) = delete;
  CuptiSubscriber(CuptiSubscriber&& other) noexcept { *this = std::move(other); }
  CuptiSubscriber& operator=(CuptiSubscriber&& other) noexcept {
    if (this != &other) {
      Close();
      handle_ = std::exchange(other.handle_, nullptr);
      api_ = other.api_;
    }
    return *this;
  }
  ~CuptiSubscriber() { Close(); }

  CUptiResult Close() {
    CUpti_SubscriberHandle handle = std::exchange(handle_, nullptr);
    return handle == nullptr ? CUPTI_SUCCESS : cuptiUnsubscribe(handle);
  }
  CUpti_SubscriberHandle get() const { return handle_; }
  CuptiApi api() const { return api_; }

 private:
  CUpti_SubscriberHandle handle_ = nullptr;
  CuptiApi api_ = CuptiApi::kV1;
};

// Mosaic has one process-global profiler session at a time.
struct {
  CuptiSubscriber subscriber;
  bool active = false;
  std::vector<std::tuple<const char* /*kernel_name*/, double /*ms*/>> timings;
} profiler_state;

bool IsRecoverableV2PreflightFailure(CUptiResult result) {
  return result == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED ||
         result == CUPTI_ERROR_NOT_SUPPORTED || result == CUPTI_ERROR_UNKNOWN;
}

// cuptiActivityGetNextRecord_v2 and cuptiGetTimestamp_v2 were added in CUDA
// 13.3. Requiring them keeps CUDA 13.2's preview V2 APIs on V1. Used by
// V2-specific tests.
bool CanUseCuptiV2() {
  return cuptiSubscribe_v2 != nullptr && cuptiGetTimestamp_v2 != nullptr &&
         cuptiActivityRegisterCallbacks_v2 != nullptr &&
         cuptiActivityEnable_v2 != nullptr &&
         cuptiActivityDisable_v2 != nullptr &&
         cuptiActivityGetNextRecord_v2 != nullptr;
}

enum class CuptiV2SubscribeResult { kSubscribed, kFallBackToV1 };

CuptiV2SubscribeResult SubscribeCuptiV2(CuptiSubscriberParamsAbi* params,
                                         CUpti_SubscriberHandle* handle) {
  CUptiResult result = cuptiSubscribe_v2(
      handle, /*callback=*/nullptr, /*userdata=*/nullptr, params);
  if (IsRecoverableV2PreflightFailure(result)) {
    return CuptiV2SubscribeResult::kFallBackToV1;
  }
  THROW_IF_CUPTI_ERROR(result, "failed to subscribe to V2 CUPTI");
  return CuptiV2SubscribeResult::kSubscribed;
}

void callback_request_impl(uint8_t** buffer, size_t* size,
                           size_t* maxNumRecords) {
  // 10 MiB buffer size is generous but somewhat arbitrary, it's at the upper
  // bound of what's recommended in CUPTI documentation:
  // https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api:~:text=For%20typical%20workloads%2C%20it%E2%80%99s%20suggested%20to%20choose%20a%20size%20between%201%20and%2010%20MB.
  const int buffer_size = 10 * (1 << 20);
  // 8 byte alignment is specified in the official CUPTI code samples, see
  // extras/CUPTI/samples/common/helper_cupti_activity.h in your CUDA
  // installation.
  *buffer = new (std::align_val_t(8)) uint8_t[buffer_size];
  *size = buffer_size;
  *maxNumRecords = 0;
}

void callback_request_v1(uint8_t** buffer, size_t* size, size_t* max_num_records) {
  callback_request_impl(buffer, size, max_num_records);
}

void CUPTIAPI callback_request_v2(uint8_t** buffer, size_t* size,
                                  size_t* max_num_records, void*) {
  callback_request_impl(buffer, size, max_num_records);
}

void process_activity_buffer(uint8_t* buffer, size_t valid_size) {
  // take ownership of the buffer once CUPTI is done using it
  absl::Cleanup cleanup = [buffer]() {
    operator delete[](buffer, std::align_val_t(8));
  };
  CUpti_Activity* record = nullptr;
  while (true) {
    CUptiResult status = profiler_state.subscriber.api() == CuptiApi::kV2
                             ? cuptiActivityGetNextRecord_v2(
                                   profiler_state.subscriber.get(), buffer,
                                   valid_size, &record)
                             : cuptiActivityGetNextRecord(buffer, valid_size,
                                                          &record);
    if (status == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        // TODO(andportnoy) handle multi-GPU
        CUpti_ActivityKernel9* kernel = (CUpti_ActivityKernel9*)record;
        // Convert integer nanoseconds to floating point milliseconds to match
        // the interface of the events-based profiler.
        double duration_ms = (kernel->end - kernel->start) / 1e6;
        const char* kernel_name = kernel->name;
        profiler_state.timings.push_back(
            std::make_tuple(kernel_name, duration_ms));
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // no more records available
      break;
    } else {
      THROW_IF_CUPTI_ERROR(status);
    }
  }

}

// The V1 and V2 completion callbacks have different CUPTI ABIs. Buffer
// parsing is shared above, but these callbacks intentionally remain separate.
void callback_complete_v1(CUcontext context, uint32_t stream_id, uint8_t* buffer,
                          size_t, size_t valid_size) {
  process_activity_buffer(buffer, valid_size);
  size_t num_dropped = 0;
  THROW_IF_CUPTI_ERROR(
      cuptiActivityGetNumDroppedRecords(context, stream_id, &num_dropped),
      "failed to get number of dropped activity records");
  THROW_IF(num_dropped > 0, "activity records were dropped");
}

void CUPTIAPI callback_complete_v2(uint8_t* buffer, size_t, size_t valid_size,
                                   void*) {
  process_activity_buffer(buffer, valid_size);
}

bool InitCuptiV2() {
  if (!CanUseCuptiV2()) {
    return false;
  }

  CuptiSubscriberParamsAbi params = {};
  params.structSize = kCuptiSubscriberParamsStructSize;
  params.subscriberName = "MosaicGpuProfiler";
  if (!SetAllowMultipleSubscribersIfSupported(&params)) {
    return false;
  }
  CUpti_SubscriberHandle handle = nullptr;
  if (SubscribeCuptiV2(&params, &handle) !=
      CuptiV2SubscribeResult::kSubscribed) {
    return false;
  }

  CuptiSubscriber candidate(handle, CuptiApi::kV2);
  uint64_t timestamp = 0;
  CUptiResult result = cuptiGetTimestamp_v2(candidate.get(), &timestamp);
  if (result == CUPTI_SUCCESS) {
    // Publish the subscriber before enabling V2 activity collection: CUPTI
    // can invoke the registered callbacks during enablement.
    profiler_state.subscriber = std::move(candidate);
    result = cuptiActivityRegisterCallbacks_v2(
        profiler_state.subscriber.get(),
        reinterpret_cast<CuptiBuffersCallbackRequestFuncV2Abi>(
            callback_request_v2),
        reinterpret_cast<CuptiBuffersCallbackCompleteFuncV2Abi>(
            callback_complete_v2));
    if (result == CUPTI_SUCCESS) {
      result = cuptiActivityEnable_v2(profiler_state.subscriber.get(),
                                      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                                      nullptr);
    }
    if (result != CUPTI_SUCCESS) {
      profiler_state.subscriber.Close();
      THROW_IF_CUPTI_ERROR(result,
                           "failed to enable V2 CUPTI activity tracing");
    }
    return true;
  }

  CUptiResult unsubscribe_result = candidate.Close();
  if (unsubscribe_result != CUPTI_SUCCESS) {
    THROW_IF_CUPTI_ERROR(unsubscribe_result,
                         "failed to unsubscribe V2 CUPTI subscriber");
  }
  if (!IsRecoverableV2PreflightFailure(result)) {
    THROW_IF_CUPTI_ERROR(result, "failed to initialize V2 CUPTI profiler");
  }
  return false;
}

void InitCuptiV1() {
  // Ok to pass nullptr for the callback here because we don't register any
  // callbacks through cuptiEnableCallback.
  CUpti_SubscriberHandle handle = nullptr;
  auto subscribe_result =
      cuptiSubscribe(&handle, /*callback=*/nullptr, /*userdata=*/nullptr);
  if (subscribe_result == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    THROW(
        "Attempted to subscribe to CUPTI while another subscriber, such as "
        "Nsight Systems or Nsight Compute, is active. CUPTI backend of the "
        "Mosaic GPU profiler cannot be used in that mode since CUPTI does "
        "not support multiple subscribers.");
  }
  THROW_IF_CUPTI_ERROR(subscribe_result, "failed to subscribe to CUPTI");

  CuptiSubscriber candidate(handle, CuptiApi::kV1);
  THROW_IF_CUPTI_ERROR(
      cuptiActivityRegisterCallbacks(callback_request_v1, callback_complete_v1),
      "failed to register CUPTI activity callbacks");
  THROW_IF_CUPTI_ERROR(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
      "failed to enable tracking of kernel activity by CUPTI");
  profiler_state.subscriber = std::move(candidate);
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("_sync_all_devices", []() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess) {
      throw std::runtime_error("Failed to get device count");
    }
    for (int i = 0; i < devices; ++i) {
      if (cudaSetDevice(i) != cudaSuccess) {
        throw std::runtime_error("Failed to set device");
      }
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize device");
      }
    }
  });
  m.def("_cupti_v2_available", [] { return CanUseCuptiV2(); });
  m.def("_cupti_init", []() {
    THROW_IF(profiler_state.active,
             "Nested or concurrent Mosaic CUPTI profiling is not supported.");
    profiler_state.timings.clear();
    // If V2 is unavailable or fails during preflight, clean up any temporary
    // V2 subscriber and continue with the legacy V1 initialization below.
    if (!InitCuptiV2()) {
      InitCuptiV1();
    }
    profiler_state.active = true;
  });
  m.def(
      "_cupti_get_timings",
      [](bool finalize) {
        THROW_IF(!profiler_state.active, "Mosaic CUPTI profiling is not active.");
        CUptiResult first_error = CUPTI_SUCCESS;
        // Continue teardown after an error: throwing immediately could leave
        // the subscriber installed. Preserve the first error to report after
        // the profiler has been made inactive.
        auto record_error = [&first_error](CUptiResult result) {
          if (first_error == CUPTI_SUCCESS && result != CUPTI_SUCCESS) {
            first_error = result;
          }
        };
        if (profiler_state.subscriber.api() == CuptiApi::kV2) {
          record_error(cuptiActivityDisable_v2(
              profiler_state.subscriber.get(),
              CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, nullptr));
        } else {
          record_error(
              cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        }
        record_error(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        if (profiler_state.subscriber.api() == CuptiApi::kV1 && finalize) {
          record_error(cuptiFinalize());
        }
        record_error(profiler_state.subscriber.Close());
        profiler_state.active = false;
        THROW_IF_CUPTI_ERROR(first_error, "failed to stop CUPTI profiling");
        return profiler_state.timings;
      },
      nb::arg("finalize") = true);
  m.def(
      "_get_ptxas_isa_version",
      []() -> int {
        return jax::ValueOrThrow(
            jax::ValueOrThrow(
                mosaic::gpu::GetAssemblyToBinaryCompilationProvider())
                ->GetLatestPtxIsaVersion());
      },
      "Returns the latest PTX ISA version supported by `ptxas`.\n\n"
      "NOTE: This PTX ISA version may not be supported by the LLVM compiler. "
      "LLVM's PTX ISA support should also be checked, unless using inline asm "
      "(which bypasses LLVM).");
}

}  // namespace
}  // namespace jax::cuda
