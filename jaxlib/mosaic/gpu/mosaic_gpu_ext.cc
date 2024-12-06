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

#include <memory>
#include <stdexcept>
#include <string>

#include "nanobind/nanobind.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace jax::cuda {
namespace {

namespace ffi = xla::ffi;
namespace nb = nanobind;

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

// Ensure it is safe to store gpuEvent_t in a uint64_t buffer.
static_assert(sizeof(gpuEvent_t) <= sizeof(uint64_t));

static const auto* kEventRecord =
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Attr<bool>("copy_before")
        .RemainingArgs()
        .Ret<ffi::BufferR0<ffi::U64>>()  // event
        .RemainingRets()
        .To([](gpuStream_t stream, bool copy_before,
               auto remaining_args, auto ret, auto remaining_rets) {
          static auto* event = new gpuEvent_t;
          if (auto res = gpuEventCreate(event, GPU_EVENT_DEFAULT);
              res) {
            return ffi::Error::Internal(
                absl::StrCat("Failed to create event: ", ToString(res)));
          }
          auto do_copy = [&]() {
            gpuMemcpyAsync(ret->untyped_data(), event,
                           sizeof(gpuEvent_t), gpuMemcpyHostToDevice, stream);
          };
          if (copy_before) {
            do_copy();
          }
          if (auto res = gpuEventRecord(*event, stream); res) {
            return ffi::Error::Internal(
                absl::StrCat("Failed to record event: ", ToString(res)));
          }
          if (!copy_before) {
            do_copy();
          }
          return ffi::Error::Success();
        })
        .release();

XLA_FFI_Error* EventRecord(XLA_FFI_CallFrame* call_frame) {
  return kEventRecord->Call(call_frame);
}

static const auto* kEventElapsed =
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Arg<ffi::BufferR0<ffi::U64>>()  // start_event
        .Arg<ffi::BufferR0<ffi::U64>>()  // end_event
        .Ret<ffi::BufferR0<ffi::F32>>()  // elapsed_ms
        .To([](gpuStream_t stream, auto start, auto end, auto out) {
          gpuStreamSynchronize(stream);
          auto start_event = std::make_unique<gpuEvent_t>();
          auto end_event = std::make_unique<gpuEvent_t>();
          absl::MakeCleanup([&]() {
            gpuEventDestroy(*start_event);
            gpuEventDestroy(*end_event);
          });
          gpuMemcpy(start_event.get(), start.untyped_data(), sizeof(gpuEvent_t),
                    gpuMemcpyDeviceToHost);
          gpuMemcpy(end_event.get(), end.untyped_data(), sizeof(gpuEvent_t),
                    gpuMemcpyDeviceToHost);
          float elapsed;
          if (auto res =
                  gpuEventElapsedTime(&elapsed, *start_event, *end_event);
              res) {
            return ffi::Error::Internal(absl::StrCat(
                "Failed to get elapsed time between events: ", ToString(res)));
          }
          gpuMemcpy(out->untyped_data(), &elapsed, sizeof(float),
                    gpuMemcpyHostToDevice);
          return ffi::Error::Success();
        })
        .release();

XLA_FFI_Error* EventElapsed(XLA_FFI_CallFrame* call_frame) {
  return kEventElapsed->Call(call_frame);
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("registrations", []() {
    return nb::make_tuple(
        nb::make_tuple("mgpu_event_record", EncapsulateFunction(EventRecord)),
        nb::make_tuple("mgpu_event_elapsed", EncapsulateFunction(EventElapsed))
    );
  });
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
