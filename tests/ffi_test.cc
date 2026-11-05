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
==============================================================================*/

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

namespace {

ffi::Error Execute() { return ffi::Error::Success(); }

ffi::Error Record() {
  return ffi::Error::Internal("FFI record handler invoked");
}

XLA_FFI_DEFINE_HANDLER(kExecute, Execute, ffi::Ffi::Bind());
XLA_FFI_DEFINE_HANDLER(kRecord, Record, ffi::Ffi::BindRecord());

}  // namespace

NB_MODULE(_ffi_test, m) {
  m.def("record_handlers", []() {
    nb::dict handlers;
    handlers["execute"] = nb::capsule(reinterpret_cast<void*>(kExecute));
    handlers["record"] = nb::capsule(reinterpret_cast<void*>(kRecord));
    return handlers;
  });
}
