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

#include <cstdint>
#include <stdexcept>
#include <string>

#include "nanobind/nanobind.h"
#include "absl/strings/str_cat.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax::cuda {
namespace {

static std::string ToString(CUresult result) {
  const char* error_name;
  if (cuGetErrorName(result, &error_name)) {
    return absl::StrCat("UNKNOWN ERROR (", static_cast<int>(result), ")");
  }
  const char* error_string;
  if (cuGetErrorString(result, &error_string)) {
    return error_name;
  }
  return absl::StrCat(error_name, ": ", error_string);
}

void EventRecordCall(void* stream, void** buffers, char* opaque,
                     size_t opaque_len, XlaCustomCallStatus* status) {
  auto* event = reinterpret_cast<gpuEvent_t**>(opaque);
  if (auto res = gpuEventRecord(**event, reinterpret_cast<gpuStream_t>(stream));
      res) {
    auto message = absl::StrCat("Failed to record event: ", ToString(res));
    XlaCustomCallStatusSetFailure(status, message.c_str(), message.size());
  }
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("_gpu_event_create", []() {
    gpuEvent_t* event = new gpuEvent_t();
    if (auto res = gpuEventCreate(event, GPU_EVENT_DEFAULT); res) {
      throw std::runtime_error(
          absl::StrCat("Failed to create event: ", ToString(res)));
    }
    return reinterpret_cast<uintptr_t>(event);
  });
  m.def("_gpu_event_destroy", [](uintptr_t event) {
    if (auto res = gpuEventDestroy(*reinterpret_cast<gpuEvent_t*>(event));
        res) {
      throw std::runtime_error(
          absl::StrCat("Failed to destroy event: ", ToString(res)));
    }
  });
  m.def("_gpu_event_elapsed", [](uintptr_t start_event, uintptr_t end_event) {
    float elapsed_ms = -1;
    if (auto res = gpuEventElapsedTime(
            &elapsed_ms, *reinterpret_cast<gpuEvent_t*>(start_event),
            *reinterpret_cast<gpuEvent_t*>(end_event));
        res) {
      throw std::runtime_error(absl::StrCat(
          "Failed to get elapsed time between events: ", ToString(res)));
    }
    return elapsed_ms;
  });
  m.def("_record_event_capsule",
        []() { return EncapsulateFunction(EventRecordCall); });
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
}

}  // namespace
}  // namespace jax::cuda
