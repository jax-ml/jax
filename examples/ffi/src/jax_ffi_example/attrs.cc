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

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

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

NB_MODULE(_attrs, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["array_attr"] =
        nb::capsule(reinterpret_cast<void *>(ArrayAttr));
    registrations["dictionary_attr"] =
        nb::capsule(reinterpret_cast<void *>(DictionaryAttr));
    return registrations;
  });
}
