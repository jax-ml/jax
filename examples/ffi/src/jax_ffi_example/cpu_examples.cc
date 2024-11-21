/* Copyright 2024 The JAX Authors.

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

#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// ----------
// Attributes
// ----------
//
// An example demonstrating the different ways that attributes can be passed to
// the FFI.
//
// For example, we can pass arrays, variadic attributes, and user-defined types.
// Full support of user-defined types isn't yet supported by XLA, so that
// example will be added in the future.

ffi::Error ArrayAttrImpl(ffi::Span<const int32_t> array,
                         ffi::ResultBufferR0<ffi::S32> res) {
  int64_t total = 0;
  for (int32_t x : array) {
    total += x;
  }
  res->typed_data()[0] = total;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ArrayAttr, ArrayAttrImpl,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Span<const int32_t>>("array")
                                  .Ret<ffi::BufferR0<ffi::S32>>());

ffi::Error DictionaryAttrImpl(ffi::Dictionary attrs,
                              ffi::ResultBufferR0<ffi::S32> secret,
                              ffi::ResultBufferR0<ffi::S32> count) {
  auto maybe_secret = attrs.get<int64_t>("secret");
  if (maybe_secret.has_error()) {
    return maybe_secret.error();
  }
  secret->typed_data()[0] = maybe_secret.value();
  count->typed_data()[0] = attrs.size();
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DictionaryAttr, DictionaryAttrImpl,
                              ffi::Ffi::Bind()
                                  .Attrs()
                                  .Ret<ffi::BufferR0<ffi::S32>>()
                                  .Ret<ffi::BufferR0<ffi::S32>>());

// -------
// Counter
// -------
//
// An example demonstrating how an FFI call can maintain "state" between calls
//
// In this case, the ``Counter`` call simply accumulates the number of times it
// was executed, but this pattern can also be used for more advanced use cases.
// For example, this pattern is used in jaxlib for:
//
// 1. The GPU solver linear algebra kernels which require an expensive "handler"
//    initialization, and
// 2. The ``triton_call`` function which caches the compiled triton modules
//    after their first use.

ffi::Error CounterImpl(int64_t index, ffi::ResultBufferR0<ffi::S32> out) {
  static std::mutex mutex;
  static auto &cache = *new std::unordered_map<int64_t, int32_t>();
  {
    const std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(index);
    if (it != cache.end()) {
      out->typed_data()[0] = ++it->second;
    } else {
      cache.insert({index, 0});
      out->typed_data()[0] = 0;
    }
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Counter, CounterImpl,
    ffi::Ffi::Bind().Attr<int64_t>("index").Ret<ffi::BufferR0<ffi::S32>>());

// --------
// Aliasing
// --------
//
// This example demonstrates how input-output aliasing works. The handler
// doesn't do anything except to check that the input and output pointers
// address the same data.

ffi::Error AliasingImpl(ffi::AnyBuffer input,
                        ffi::Result<ffi::AnyBuffer> output) {
  if (input.element_type() != output->element_type() ||
      input.element_count() != output->element_count()) {
    return ffi::Error::InvalidArgument(
        "The input and output data types and sizes must match.");
  }
  if (input.untyped_data() != output->untyped_data()) {
    return ffi::Error::InvalidArgument(
        "When aliased, the input and output buffers should point to the same "
        "data.");
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Aliasing, AliasingImpl,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

// Boilerplate for exposing handlers to Python
NB_MODULE(_cpu_examples, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["array_attr"] =
        nb::capsule(reinterpret_cast<void *>(ArrayAttr));
    registrations["dictionary_attr"] =
        nb::capsule(reinterpret_cast<void *>(DictionaryAttr));
    registrations["counter"] = nb::capsule(reinterpret_cast<void *>(Counter));
    registrations["aliasing"] = nb::capsule(reinterpret_cast<void *>(Aliasing));
    return registrations;
  });
}
