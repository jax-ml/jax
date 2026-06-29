/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// OneAPI/SYCL GPU runtime wrappers for JAX.
// Reference: xla/stream_executor/sycl/sycl_gpu_runtime.cc

#include "jaxlib/oneapi/oneapi_gpu_runtime.h"

namespace jax {
namespace oneapi {

absl::Status SyclMemcpyAsync(void *dst, const void *src, size_t byte_count,
                             SyclMemcpyKind /*kind*/, ::sycl::queue *stream) {
  if (byte_count == 0) {
    return absl::OkStatus();
  }
  if (dst == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyAsync: null destination pointer (dst); cannot copy into it");
  }
  if (src == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyAsync: null source pointer (src); cannot copy from it");
  }
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyAsync: null SYCL queue (sycl::queue* stream); cannot submit "
        "the memcpy");
  }
  return TryCatchToStatus([&] { stream->memcpy(dst, src, byte_count); });
}

absl::Status SyclGetLastError() { return absl::OkStatus(); }

absl::Status SyclStreamSynchronize(::sycl::queue *stream) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "SyclStreamSynchronize: null SYCL queue (sycl::queue* stream); cannot "
        "wait for completion");
  }
  return TryCatchToStatus([&] { stream->wait(); });
}

}  // namespace oneapi
}  // namespace jax
