/* Copyright 2022 The JAX Authors

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

#ifndef JAXLIB_UTIL_H_
#define JAXLIB_UTIL_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/python/ifrt/array.h"

namespace jax {

// Waits until future is ready but will cancel if ctrl-c is pressed.
void BlockUntilReadyWithCancel(xla::Future<>& future);

// Requests if given buffers are ready, awaits for results and returns OK if
// all of the buffers are ready or the last non-ok status.
absl::Status AwaitBuffersReady(absl::Span<xla::ifrt::Array* const> ifrt_arrays);

}  // namespace jax

#endif  // JAXLIB_UTIL_H_
