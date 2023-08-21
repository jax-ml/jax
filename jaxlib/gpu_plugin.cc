#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_internal.h"

// extern "C" {
// PJRT_PLUGIN_EXPORTED const PJRT_Api* GetPjrtApi();
// }
const PJRT_Api* GetPjrtApi() { return pjrt::gpu_plugin::GetGpuPjrtApi(); }
