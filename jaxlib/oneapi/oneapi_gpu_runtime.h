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
//
// SYCL wrappers (SyclMemcpyAsync, SyclGetLastError, SyclStreamSynchronize)
// that use SYCL queue APIs and exception handling to return absl::Status.
// On a SYCL exception the descriptive message is embedded in an
// absl::InternalError. vendor.h aliases these to the gpuXxx
// (e.g. gpuMemcpyAsync) abstractions.
//
// Reference: xla/stream_executor/sycl/sycl_gpu_runtime.cc

#ifndef JAXLIB_ONEAPI_ONEAPI_GPU_RUNTIME_H_
#define JAXLIB_ONEAPI_ONEAPI_GPU_RUNTIME_H_

#include <sycl/sycl.hpp>

#include <cstdint>
#include <exception>
#include <type_traits>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace jax {
namespace oneapi {

template <typename Callable>
using InvokeResult = std::invoke_result_t<Callable>;

template <typename Callable>
using StatusOrResult =
    std::conditional_t<std::is_void_v<InvokeResult<Callable>>, absl::Status,
                       absl::StatusOr<InvokeResult<Callable>>>;

// `TryCatchToStatus` converts any thrown exception into an absl error status.
// a void callable returns OkStatus on success, a value callable
// returns its result wrapped in StatusOr.
//
// Usage patterns:
//
//   1. Void SYCL op inside a function returning absl::Status.
//        return JAX_AS_STATUS(TryCatchToStatus([&] {
//          stream->memcpy(dst, src, n);
//        }));
//
//        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(TryCatchToStatus([&] {
//            stream->memcpy(dst, src, byte_count);
//        })));
//
//   2. Capturing a value (e.g. a sycl::event).
//        JAX_ASSIGN_OR_RETURN(sycl::event ev, TryCatchToStatus([&] {
//          return stream->submit(
//              [&](sycl::handler& h) { h.parallel_for(...); });
//        }));
//
//   3. When the status cannot be returned: consume it locally.
//        absl::Status s = TryCatchToStatus(
//            [&] { q->submit([&](handler& h) {...}); });
//        if (!s.ok()) LOG(ERROR) << s.message();
//
template <typename Callable>
StatusOrResult<Callable> TryCatchToStatus(Callable&& func) {
  try {
    if constexpr (std::is_void_v<InvokeResult<Callable>>) {
      std::forward<Callable>(func)();
      return absl::OkStatus();
    } else {
      return std::forward<Callable>(func)();
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("SYCL exception: ", e.what(),
                     " [sycl_code=", e.code().value(), "]"));
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat("std::exception: ", e.what()));
  } catch (...) {
    return absl::InternalError("Unknown (non-std::exception) thrown");
  }
}

// q->memcpy We do not need to specify in which direction the copy is meant to
// happen but we keep the enum for API compatibility with call sites that pass
// gpuMemcpyDeviceToDevice etc. at call sites.
enum SyclMemcpyKind {
  SyclMemcpyHostToHost = 0,
  SyclMemcpyHostToDevice = 1,
  SyclMemcpyDeviceToHost = 2,
  SyclMemcpyDeviceToDevice = 3,
};

// Wraps sycl::queue::memcpy in a try-catch and returns absl::Status.
// The `kind` parameter is accepted for API compatibility but is not used
// by SYCL (memcpy direction is implicit in the source and destination
// pointers).
absl::Status SyclMemcpyAsync(void *dst, const void *src, size_t byte_count,
                             SyclMemcpyKind kind, ::sycl::queue *stream);

// TODO(Intel-tf): Asynchronous error handling.
// Placeholder for gpuGetLastError. SYCL has no equivalent; currently returns
// OkStatus so that call sites can compile without #ifdef guards.
absl::Status SyclGetLastError();

// Stream synchronization wrapper.
// Wraps sycl::queue::wait() in a try-catch and returns absl::Status.
absl::Status SyclStreamSynchronize(::sycl::queue *stream);

}  // namespace oneapi
}  // namespace jax

#endif  // JAXLIB_ONEAPI_ONEAPI_GPU_RUNTIME_H_
