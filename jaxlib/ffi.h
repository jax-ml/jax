/* Copyright 2025 The JAX Authors

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

#ifndef JAXLIB_XLA_FFI_H_
#define JAXLIB_XLA_FFI_H_

#include <Python.h>

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/pjrt/host_callback.h"
#include "xla/python/nb_numpy.h"
#include "xla/xla_data.pb.h"

namespace jax {

namespace ffi = xla::ffi;
namespace nb = nanobind;

// Wrapper class for XLA FFI execution context.
//
// This class provides a Python interface to the XLA FFI execution context,
// exposing metadata such as the execution stage, device ordinal, and stream.
class PyFfiContext {
 public:
  enum class Stage {
    kInstantiate,
    kPrepare,
    kInitialize,
    kExecute,
  };

  PyFfiContext(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
               XLA_FFI_ExecutionStage stage);
  Stage stage() const;
  absl::StatusOr<uintptr_t> stream() const;

 private:
  const XLA_FFI_Api* api_;
  XLA_FFI_ExecutionContext* ctx_;
  XLA_FFI_ExecutionStage stage_;
};

// Wrapper class for XLA FFI AnyBuffer.
//
// This class provides a Python interface to the XLA FFI `AnyBuffer` class.
// From Python, this object looks like an array (with `.dtype` and `.shape`
// attributes), but it also provides methods zero-copy conversions to standard
// transport formats: `__array__`, `__cuda_array_interface__`, and `__dlpack__`.
class PyFfiAnyBuffer {
 public:
  PyFfiAnyBuffer(DLDeviceType device_type, int32_t device_ordinal, void* data,
                 ffi::Span<int64_t const> dimensions,
                 ffi::DataType element_type, bool writeable);
  PyFfiAnyBuffer(DLDeviceType device_type, int32_t device_ordinal,
                 ffi::AnyBuffer buf);
  PyFfiAnyBuffer(DLDeviceType device_type, int32_t device_ordinal,
                 ffi::Result<ffi::AnyBuffer> buf);

  absl::StatusOr<xla::nb_dtype> dtype() const;
  size_t ndim() const;
  nb::tuple shape() const;
  bool writeable() const;

  absl::StatusOr<xla::nb_numpy_ndarray> NumpyArray() const;
  absl::StatusOr<nb::dict> CudaArrayInterface() const;
  absl::StatusOr<nb::capsule> DLPack() const;
  absl::StatusOr<nb::capsule> DLPackVersioned() const;
  nb::tuple DLPackDevice() const;

 private:
  DLDeviceType device_type_;
  int32_t device_ordinal_;
  void* data_;
  absl::Span<int64_t const> dimensions_;
  xla::PrimitiveType element_type_;
  bool writeable_;
};

template <DLDeviceType DeviceType>
ffi::Error XlaBufferCallback(int32_t device_ordinal, const XLA_FFI_Api* api,
                             XLA_FFI_ExecutionContext* ctx,
                             xla::FfiLoadedHostCallbacks* callbacks,
                             uint64_t index, ffi::RemainingArgs args,
                             ffi::RemainingRets rets) {
  nb::gil_scoped_acquire gil;
  auto callback = nb::borrow<nb::callable>(
      static_cast<PyObject*>(callbacks->callbacks[index]));
  auto nb_args =
      nb::steal<nb::tuple>(PyTuple_New(1 + args.size() + rets.size()));

  jax::PyFfiContext py_ctx(api, ctx, XLA_FFI_ExecutionStage_EXECUTE);
  PyTuple_SET_ITEM(nb_args.ptr(), 0, nb::cast(py_ctx).release().ptr());

  size_t offset = 1;
  for (size_t i = 0; i < args.size(); ++i, ++offset) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    if (arg.has_error()) {
      return arg.error();
    }
    jax::PyFfiAnyBuffer py_buffer(DeviceType, device_ordinal, arg.value());
    PyTuple_SET_ITEM(nb_args.ptr(), offset,
                     nb::cast(py_buffer).release().ptr());
  }

  for (size_t i = 0; i < rets.size(); ++i, ++offset) {
    auto ret = rets.get<ffi::AnyBuffer>(i);
    if (ret.has_error()) {
      return ret.error();
    }
    jax::PyFfiAnyBuffer py_buffer(DeviceType, device_ordinal, ret.value());
    PyTuple_SET_ITEM(nb_args.ptr(), offset,
                     nb::cast(py_buffer).release().ptr());
  }

  xla::EnterHostCallback();
  try {
    callback(*nb::borrow<nb::args>(nb_args));
  } catch (nb::python_error& e) {
    return ffi::Error::Internal(
        absl::StrFormat("Error when calling buffer callback: %s", e.what()));
  }
  xla::LeaveHostCallback();

  return ffi::Error::Success();
}

void BuildFfiSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_XLA_FFI_H_
