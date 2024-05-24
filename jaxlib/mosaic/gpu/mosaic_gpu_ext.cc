#include <cstdint>

#include "nanobind/nanobind.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax::cuda {
namespace {

namespace nb = nanobind;
using MosaicInitFunc = void(void***);
using MosaicHostFunc = void(void**);

std::pair<absl::flat_hash_map<uintptr_t, void*>*, absl::Mutex*>
GetContextCache() {
  static absl::Mutex mutex;
  static auto& context_cache = *new absl::flat_hash_map<uintptr_t, void*>;
  return std::make_pair(&context_cache, &mutex);
}

void InvalidateCache(MosaicInitFunc* init) {
  auto cache = GetContextCache();
  absl::MutexLock lock(cache.second);
  // TODO(apaszke): Free all the resources instead of leaking.
  cache.first->erase(reinterpret_cast<uintptr_t>(init));
}

// Each compiled kernel has a unique init func, and each kernel is used from
// a single HLO module. So it should be safe to not include the CUDA context
// in the key.
void* InitOnce(MosaicInitFunc* init) {
  auto cache_and_mutex = GetContextCache();
  auto* cache = cache_and_mutex.first;
  auto* mutex = cache_and_mutex.second;

  uintptr_t key = reinterpret_cast<uintptr_t>(init);

  {
    // Fast path uses reader lock (as hash map look-up is relatively slow).
    absl::ReaderMutexLock lock(mutex);
    auto it = cache->find(key);
    if (ABSL_PREDICT_TRUE(it != cache->end())) return it->second;
  }

  absl::MutexLock lock(mutex);
  void*& ctx = (*cache)[key];
  // We released the reader lock, another thread might have initialized it.
  if (ctx == nullptr) {
    void** ptr_to_ctx = &ctx;
    init(&ptr_to_ctx);
  }
  return ctx;
}

void MosaicKernelCall(void* stream, void** buffers, char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  void** static_args = *reinterpret_cast<void***>(opaque);
  MosaicHostFunc* func = reinterpret_cast<MosaicHostFunc*>(static_args[0]);
  MosaicInitFunc* init = reinterpret_cast<MosaicInitFunc*>(static_args[1]);
  void* ctx = InitOnce(init);
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
  m.def("invalidate_cache", [](uintptr_t init_func_ptr) {
    return InvalidateCache(reinterpret_cast<MosaicInitFunc*>(init_func_ptr));
  });
}

}  // namespace
}  // namespace jax::cuda
