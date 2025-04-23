/* Copyright 2020 The JAX Authors

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

// Helpers for converting Python values into buffers.

#ifndef JAXLIB_PY_VALUES_H_
#define JAXLIB_PY_VALUES_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/nb_numpy.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {

struct DevicePutResult {
  DevicePutResult(tsl::RCReference<ifrt::Array> ifrt_array, bool weak_type)
      : ifrt_array(std::move(ifrt_array)), weak_type(weak_type) {}

  // Disallow copy. `DevicePutResult` is expected to be consumed by one user.
  DevicePutResult(const DevicePutResult&) = delete;
  DevicePutResult& operator=(const DevicePutResult&) = delete;
  DevicePutResult(DevicePutResult&&) noexcept = default;
  DevicePutResult& operator=(DevicePutResult&&) noexcept = default;

  // Points to the on-device array.
  tsl::RCReference<ifrt::Array> ifrt_array;
  bool weak_type;
};

// Options for `DevicePut`.
struct DevicePutOptions {
  bool squash_64bit_types = false;
  bool allow_zero_copy = true;
};

// Copies a buffer-like object to be on device. This version is designed for
// creating a single-device array.
//
// If `addressable_shard` is not convertible to a `PjRtBuffer` from C++, an
// error will be returned; float0s are not supported yet.
//
// If the value is known to be a PyBuffer object, py_buffer can be passed as an
// optimization to avoid a Python->C++ cast.
//
// Requires GIL. This function performs Python work inline, and runs expensive
// C++ work with GIL temporarily released.
//
// May throw exceptions from nanobind in addition to failing via an error
// absl::Status. (We could catch these if needed, but there seems little point.)
absl::StatusOr<DevicePutResult> DevicePutWithDevice(
    nanobind::handle addressable_shard, ifrt::Client* ifrt_client,
    ifrt::Device* ifrt_device, ifrt::MemoryKind ifrt_memory_kind,
    const DevicePutOptions& options);

// Copies a buffer-like object to be on device. This version is optimized for
// creating a multi-device array.
//
// `addressable_shards` is a list of buffer-like objects to be copied to
// addressable devices specified in `sharding`.
//
// `shape` and `sharding` determine the shape and sharding of the returned IFRT
// Array.
//
// The size of `addressable_shards` must match the number of addressable devices
// in `sharding`. For a Pmap sharding, there must be at least one addressable
// device.
//
// Requires GIL. This function performs Python work inline, and runs expensive
// C++ work with GIL temporarily released.
//
// See the above `DevicePutWithDevice` for other details.
absl::StatusOr<DevicePutResult> DevicePutWithSharding(
    absl::Span<const nanobind::handle> addressable_shards,
    ifrt::Client* ifrt_client, const nb_dtype& dtype,
    absl::Span<const int64_t> shape, nanobind::handle sharding,
    const DevicePutOptions& options);

// Returns `true` if `arg` is a JAX float0 array.
bool IsFloat0(xla::nb_numpy_ndarray arg);

// Describes the abstract shape and dtype of an argument.
struct PyArgSignature {
  PyArgSignature(PrimitiveType dtype, absl::Span<const int64_t> shape,
                 bool weak_type)
      : dtype(dtype), shape(shape.begin(), shape.end()), weak_type(weak_type) {}
  // This is the XLA dtype of the object.
  const PrimitiveType dtype;
  const absl::InlinedVector<int64_t, 4> shape;
  // JAX arguments can be of weak type, if and only if they are Python scalars
  // or `DeviceArray` values such that `aval.weak_type` is true.
  const bool weak_type;
  bool operator==(const PyArgSignature& other) const {
    return std::tie(dtype, weak_type, shape) ==
           std::tie(other.dtype, other.weak_type, other.shape);
  }
  bool operator!=(const PyArgSignature& other) const {
    return !(*this == other);
  }
  std::string DebugString() const;
};

// Returns the PyArgSignature associated with an argument. Returns an error if
// the argument is not supported.
absl::StatusOr<PyArgSignature> PyArgSignatureOfValue(nanobind::handle arg,
                                                     bool jax_enable_x64);

template <typename H>
H AbslHashValue(H h, const xla::PyArgSignature& s) {
  h = H::combine(std::move(h), s.dtype);
  h = H::combine_contiguous(std::move(h), s.shape.data(), s.shape.size());
  return h;
}

}  // namespace xla

#endif  // JAXLIB_PY_VALUES_H_
