/* Copyright 2025 The JAX Authors.

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
#include <memory>

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

struct State {
  static xla::ffi::TypeId id;
  explicit State(int32_t value) : value(value) {}
  int32_t value;
};
ffi::TypeId State::id = {};

static ffi::ErrorOr<std::unique_ptr<State>> StateInstantiate() {
  return std::make_unique<State>(42);
}

static ffi::Error StateExecute(cudaStream_t stream, State* state,
                               ffi::ResultBufferR0<ffi::S32> out) {
  cudaMemcpyAsync(out->typed_data(), &state->value, sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kStateInstantiate, StateInstantiate,
                       ffi::Ffi::BindInstantiate());
XLA_FFI_DEFINE_HANDLER(kStateExecute, StateExecute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::State<State>>()
                           .Ret<ffi::BufferR0<ffi::S32>>());

NB_MODULE(_gpu_examples, m) {
  m.def("type_id",
        []() { return nb::capsule(reinterpret_cast<void*>(&State::id)); });
  m.def("state_type", []() {
    // In earlier versions of XLA:FFI, the `MakeTypeInfo` helper was not
    // available. In latest XLF:FFI `TypeInfo` is an alias for C API struct.
#if XLA_FFI_API_MINOR >= 2
    static auto kStateTypeInfo = xla::ffi::MakeTypeInfo<State>();
#else
    static auto kStateTypeInfo = xla::ffi::TypeInfo<State>();
#endif
    nb::dict d;
    d["type_id"] = nb::capsule(reinterpret_cast<void*>(&State::id));
    d["type_info"] = nb::capsule(reinterpret_cast<void*>(&kStateTypeInfo));
    return d;
  });
  m.def("handler", []() {
    nb::dict d;
    d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kStateInstantiate));
    d["execute"] = nb::capsule(reinterpret_cast<void*>(kStateExecute));
    return d;
  });
}
