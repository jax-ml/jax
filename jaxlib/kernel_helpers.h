/* Copyright 2019 Google LLC

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

#ifndef JAXLIB_KERNEL_HELPERS_H_
#define JAXLIB_KERNEL_HELPERS_H_

#include <cstddef>
#include <stdexcept>

#include "absl/base/casts.h"
#include "include/pybind11/pybind11.h"

namespace jax {

// Descriptor objects are opaque host-side objects used to pass data from JAX
// to the custom kernel launched by XLA. Currently simply treat host-side
// structures as byte-strings; this is not portable across architectures. If
// portability is needed, we could switch to using a representation such as
// protocol buffers or flatbuffers.

// Packs a descriptor object into a pybind11::bytes structure.
template <typename T>
pybind11::bytes PackDescriptor(const T& descriptor) {
  return pybind11::bytes(absl::bit_cast<const char*>(&descriptor), sizeof(T));
}

// Unpacks a descriptor object from a byte string.
template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid size for linalg operation descriptor.");
  }
  return absl::bit_cast<const T*>(opaque);
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(absl::bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

}  // namespace jax

#endif  // JAXLIB_KERNEL_HELPERS_H_
