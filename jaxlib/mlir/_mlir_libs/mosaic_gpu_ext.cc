#include <cstdint>
#include "nanobind/nanobind.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "jaxlib/mosaic/gpu/integrations/c/passes.h"
#include "xla/service/custom_call_status.h"

namespace jax::cuda {
namespace {

namespace nb = nanobind;
using MosaicHostFunc = void(void**);

void MosaicKernelCall(void* stream, void** buffers, char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  void** static_args = *reinterpret_cast<void***>(opaque);
  MosaicHostFunc* func = reinterpret_cast<MosaicHostFunc*>(static_args[0]);
  void* ctx = static_args[1];
  void* args[3] = {&ctx, &stream, &buffers};
  func(args);
}

void EventRecordCall(void* stream, void** buffers, char* opaque,
                     size_t opaque_len, XlaCustomCallStatus* status) {
  auto* event = reinterpret_cast<gpuEvent_t**>(opaque);
  if (gpuEventRecord(**event, reinterpret_cast<gpuStream_t>(stream)) !=
      gpuSuccess) {
    const char message[] = "Failed to record event";
    XlaCustomCallStatusSetFailure(status, message, sizeof(message));
  }
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("_custom_call_capsule",
        []() { return EncapsulateFunction(MosaicKernelCall); });
  m.def("register_passes", []() { return mlirMosaicGpuRegisterPasses(); });
  m.def("_gpu_event_create", []() {
    gpuEvent_t* event = new gpuEvent_t();
    gpuEventCreate(event, GPU_EVENT_DEFAULT);
    return reinterpret_cast<uintptr_t>(event);
  });
  m.def("_gpu_event_destroy", [](uintptr_t event) {
    gpuEventDestroy(*reinterpret_cast<gpuEvent_t*>(event));
  });
  m.def("_gpu_event_elapsed", [](uintptr_t start_event, uintptr_t end_event) {
    float elapsed_ms = -1;
    if (gpuEventElapsedTime(
            &elapsed_ms, *reinterpret_cast<gpuEvent_t*>(start_event),
            *reinterpret_cast<gpuEvent_t*>(end_event)) != gpuSuccess) {
      throw std::runtime_error("Failed to get elapsed time between events");
    }
    return elapsed_ms;
  });
  m.def("_record_event_capsule",
        []() { return EncapsulateFunction(EventRecordCall); });
}

}  // namespace
}  // namespace jax::cuda
