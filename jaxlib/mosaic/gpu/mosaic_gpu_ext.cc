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
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/absl_status_casters.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/mosaic/gpu/target.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"

// JAX_CUPTI_HAS_MULTI_SUBSCRIBER is defined when the CUPTI V2 multi-subscriber
// APIs are available (CUPTI_API_VERSION >= 130200, i.e. CUDA 13.2+).
#if CUPTI_API_VERSION >= 130200
#define JAX_CUPTI_HAS_MULTI_SUBSCRIBER
#endif

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

// Mosaic keeps a single global profiler state, so only one session may be
// active at a time.
struct {
  CUpti_SubscriberHandle subscriber;
  std::vector<std::tuple<const char* /*kernel_name*/, double /*ms*/>> timings;
} profiler_state;

#ifdef JAX_CUPTI_HAS_MULTI_SUBSCRIBER
void callback_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords,
                      CUpti_BufferCallbackRequestInfo * /*info*/) {
#else
void callback_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
#endif
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

#ifdef JAX_CUPTI_HAS_MULTI_SUBSCRIBER
void callback_complete(uint8_t *buffer, size_t size, size_t validSize,
                       CUpti_BufferCallbackCompleteInfo * /*info*/) {
#else
void callback_complete(CUcontext context, uint32_t streamId, uint8_t *buffer,
                       size_t size, size_t validSize) {
#endif
  // take ownership of the buffer once CUPTI is done using it
  absl::Cleanup cleanup = [buffer]() {
    operator delete[](buffer, std::align_val_t(8));
  };
  CUpti_Activity* record = nullptr;
  while (true) {
    CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
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

#ifndef JAX_CUPTI_HAS_MULTI_SUBSCRIBER
  size_t num_dropped;
  THROW_IF_CUPTI_ERROR(
      cuptiActivityGetNumDroppedRecords(context, streamId, &num_dropped),
      "failed to get number of dropped activity records");
  THROW_IF(num_dropped > 0, "activity records were dropped");
#endif
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("_sync_all_devices", []() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != gpuSuccess) {
      throw std::runtime_error("Failed to get device count");
    }
    for (int i = 0; i < devices; ++i) {
      if (cudaSetDevice(i) != gpuSuccess) {
        throw std::runtime_error("Failed to set device");
      }
      if (cudaDeviceSynchronize() != gpuSuccess) {
        throw std::runtime_error("Failed to synchronize device");
      }
    }
  });
  m.def("_cupti_init", []() {
    profiler_state.timings.clear();
#ifdef JAX_CUPTI_HAS_MULTI_SUBSCRIBER
    // V2 API: allows coexistence with other subscribers, such as jax.profiler.
    CUpti_SubscriberParams params = {};
    params.structSize = CUpti_SubscriberParams_STRUCT_SIZE;
    params.subscriberName = "MosaicGpuProfiler";
    params.allowMultipleSubscribers = 1;
    // Ok to pass nullptr for the callback here because we don't register any
    // callbacks through cuptiEnableCallback.
    THROW_IF_CUPTI_ERROR(cuptiSubscribe_v2(&profiler_state.subscriber,
                                           /*callback=*/nullptr,
                                           /*userdata=*/nullptr, &params),
                         "failed to subscribe to CUPTI");
    THROW_IF_CUPTI_ERROR(
        cuptiActivityRegisterCallbacks_v2(profiler_state.subscriber,
                                          callback_request, callback_complete),
        "failed to register CUPTI activity callbacks");
    CUpti_ActivityConfig act_cfg = {};
    act_cfg.structSize = CUpti_ActivityConfig_STRUCT_SIZE;
    THROW_IF_CUPTI_ERROR(
        cuptiActivityEnable_v2(profiler_state.subscriber,
                               CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, &act_cfg),
        "failed to enable tracking of kernel activity by CUPTI");
#else
    // V1 API: only one CUPTI subscriber allowed at a time.
    // Ok to pass nullptr for the callback here because we don't register any
    // callbacks through cuptiEnableCallback.
    auto subscribe_result = cuptiSubscribe(
        &profiler_state.subscriber, /*callback=*/nullptr, /*userdata=*/nullptr);
    if (subscribe_result == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
      THROW(
          "Attempted to subscribe to CUPTI while another subscriber, such as "
          "Nsight Systems or Nsight Compute, is active. CUPTI backend of the "
          "Mosaic GPU profiler cannot be used in that mode since CUPTI does "
          "not support multiple subscribers.");
    }
    THROW_IF_CUPTI_ERROR(subscribe_result, "failed to subscribe to CUPTI");
    THROW_IF_CUPTI_ERROR(
        cuptiActivityRegisterCallbacks(callback_request, callback_complete),
        "failed to register CUPTI activity callbacks");
    THROW_IF_CUPTI_ERROR(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
        "failed to enable tracking of kernel activity by CUPTI");
#endif
  });
  m.def(
      "_cupti_get_timings",
      [](bool finalize) {
#ifdef JAX_CUPTI_HAS_MULTI_SUBSCRIBER
        (void)finalize;  // unused in the V2 path
        THROW_IF_CUPTI_ERROR(
            cuptiActivityDisable_v2(profiler_state.subscriber,
                                    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                                    nullptr),
            "failed to disable tracking of kernel activity by CUPTI");
        THROW_IF_CUPTI_ERROR(
            cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
            "failed to flush CUPTI activity buffers");
        // The legacy dropped-record query is process-global, so in the V2
        // path it could attribute another subscriber's drops to Mosaic.
        // TODO: In the V2 path, plumb enough context/stream information to
        // query dropped activity records safely with
        // cuptiActivityGetNumDroppedRecords().
        // cuptiUnsubscribe() is sufficient here; cuptiFinalize() tears down
        // global CUPTI state and breaks later V2 re-initialization.
#else
        THROW_IF_CUPTI_ERROR(
            cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
            "failed to disable tracking of kernel activity by CUPTI");
        THROW_IF_CUPTI_ERROR(
            cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
            "failed to flush CUPTI activity buffers");
        // In the V1 path, cuptiFinalize() is required to avoid leaking CUPTI
        // state across repeated measurements.
        if (finalize) {
          THROW_IF_CUPTI_ERROR(cuptiFinalize(), "failed to detach CUPTI");
        }
#endif
        THROW_IF_CUPTI_ERROR(cuptiUnsubscribe(profiler_state.subscriber),
                             "failed to unsubscribe from CUPTI");
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
