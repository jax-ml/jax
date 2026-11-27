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

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cudaProfiler.h"
#include "nanobind/nanobind.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/gpu_plugin_extension.h"
#include "jaxlib/gpu/py_client_gpu.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/pjrt/status_casters.h"

namespace nb = nanobind;

namespace jax {
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

static nb::dict GpuTransposePlanCacheType() {
  auto [type_id, type_info] = cuda::GpuTransposePlanCacheTypeInfo();
  nb::dict d;
  d["type_id"] = nb::capsule(type_id);
  d["type_info"] = nb::capsule(type_info);
  return d;
}

nb::dict FfiTypes() {
  nb::dict dict;
  dict["GpuTransposePlanCache"] = GpuTransposePlanCacheType();
  return dict;
}

nb::dict FfiHandlers() {
  nb::dict dict;
  nb::dict gpu_callback_dict;
  gpu_callback_dict["instantiate"] =
      EncapsulateFfiHandler(cuda::kGpuTransposePlanCacheInstantiate);
  gpu_callback_dict["execute"] =
      EncapsulateFfiHandler(cuda::kXlaFfiPythonGpuCallback);
  dict["xla_ffi_python_gpu_callback"] = gpu_callback_dict;
  dict["xla_ffi_partitioned_python_gpu_callback"] = gpu_callback_dict;
  dict["xla_buffer_python_gpu_callback"] =
      EncapsulateFfiHandler(cuda::kXlaBufferPythonGpuCallback);
  dict["xla_buffer_python_gpu_callback_cmd_buffer"] =
      EncapsulateFfiHandler(cuda::kXlaBufferPythonGpuCallbackCmdBuffer);
  dict["dce_sink"] =
      EncapsulateFfiHandler(cuda::dce_sink_gpu_ffi);
  return dict;
}

// Calls cuProfilerStart (if `start` is true) or cuProfilerStop in the primary
// context of every device whose primary context is active, which are the
// contexts XLA launches its kernels in. The CUDA documentation scopes both
// functions to the current context, so this follows that contract and does not
// depend on the calling thread having a current context. Nsight Systems and
// Nsight Compute treat them as process-wide, so the calls after the first one
// are no-ops.
absl::Status StartOrStopCudaProfiler(bool start) {
  int device_count = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cuDeviceGetCount(&device_count)));
  int num_active_contexts = 0;
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    CUdevice device;
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cuDeviceGet(&device, ordinal)));
    unsigned int flags = 0;
    int active = 0;
    JAX_RETURN_IF_ERROR(
        JAX_AS_STATUS(cuDevicePrimaryCtxGetState(device, &flags, &active)));
    if (!active) {
      continue;
    }
    CUcontext context;
    JAX_RETURN_IF_ERROR(
        JAX_AS_STATUS(cuDevicePrimaryCtxRetain(&context, device)));
    absl::Status status = JAX_AS_STATUS(cuCtxPushCurrent(context));
    if (status.ok()) {
      // These are called directly rather than through a function pointer: the
      // dynamically loaded driver functions carry no type information, so an
      // indirect call fails control-flow integrity checks (-fsanitize=kcfi).
      status = start ? JAX_AS_STATUS(cuProfilerStart())
                     : JAX_AS_STATUS(cuProfilerStop());
      CUcontext popped;
      status.Update(JAX_AS_STATUS(cuCtxPopCurrent(&popped)));
    }
    status.Update(JAX_AS_STATUS(cuDevicePrimaryCtxRelease(device)));
    JAX_RETURN_IF_ERROR(status);
    ++num_active_contexts;
  }
  if (num_active_contexts == 0) {
    return absl::FailedPreconditionError(
        "No active CUDA context found; initialize the GPU backend first");
  }
  return absl::OkStatus();
}

}  // namespace

NB_MODULE(cuda_plugin_extension, m) {
  BuildGpuPluginExtension(m);
  m.def("ffi_types", &FfiTypes);
  m.def("ffi_handlers", &FfiHandlers);

  m.def(
      "get_device_ordinal",
      [](std::intptr_t data_value) {
        if (data_value == 0) {
          return 0;
        }
        int device_ordinal;
        void* data_ptr = reinterpret_cast<void*>(data_value);
        CUresult result =
            cuPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  reinterpret_cast<CUdeviceptr>(data_ptr));
        if (result != CUDA_SUCCESS) {
          xla::ThrowIfError(absl::InvalidArgumentError(absl::StrCat(
              "Not able to get the device_ordinal: ", ToString(result))));
        }
        return device_ordinal;
      },
      nb::arg("data_value"));
  m.def(
      "start_capture",
      []() { xla::ThrowIfError(StartOrStopCudaProfiler(/*start=*/true)); },
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "stop_capture",
      []() { xla::ThrowIfError(StartOrStopCudaProfiler(/*start=*/false)); },
      nb::call_guard<nb::gil_scoped_release>());
}
}  // namespace jax
