#ifndef JAXLIB_GPU_TRITON_H_
#define JAXLIB_GPU_TRITON_H_

#include "jaxlib/gpu/vendor.h"
#include "xla/service/custom_call_status.h"

void LaunchTritonKernel(CUstream stream, void** buffers, const char* opaque,
                        size_t opaque_len, XlaCustomCallStatus* status);

#endif  // JAXLIB_GPU_TRITON_H_
