/* Copyright 2019 The JAX Authors.

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
#include <string>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"

namespace jax {

// See kernel_nanobind_helpers.h for info on descriptor objects. We separate out
// the functionality that doesn't require nanobind for building CUDA libraries,
// since older versions nvcc don't seem to be able to compile nanobind.

// Packs a descriptor object into a byte string.
template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(absl::bit_cast<const char*>(&descriptor), sizeof(T));
}

// Unpacks a descriptor object from a byte string.
template <typename T>
absl::StatusOr<const T*> UnpackDescriptor(const char* opaque,
                                          std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    return absl::InternalError("Invalid size for operation descriptor.");
  }
  return absl::bit_cast<const T*>(opaque);
}

}  // namespace jax

#endif  // JAXLIB_KERNEL_HELPERS_H_
