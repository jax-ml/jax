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

#include "jaxlib/ffi.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"
#include "nanobind/nanobind.h"
#include "jaxlib/dlpack_support.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace jax {

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace {
const char* const kDlTensorCapsuleName = "dltensor";
const char* const kDlTensorVersionedCapsuleName = "dltensor_versioned";

template <typename ManagedTensor>
struct DLPackTensor {
  std::vector<int64_t> shape;
  ManagedTensor tensor;
};

template <typename ManagedTensor>
void DLPackTensorDeleter(ManagedTensor* t) {
  if (t) {
    delete static_cast<DLPackTensor<ManagedTensor>*>(t->manager_ctx);
  }
}

xla::PrimitiveType PrimitiveTypeForFfiDataType(ffi::DataType dtype) {
  switch (dtype) {
    case ffi::DataType::INVALID:
      return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
    case ffi::PRED:
      return xla::PrimitiveType::PRED;
    case ffi::S1:
      return xla::PrimitiveType::S1;
    case ffi::S2:
      return xla::PrimitiveType::S2;
    case ffi::S4:
      return xla::PrimitiveType::S4;
    case ffi::S8:
      return xla::PrimitiveType::S8;
    case ffi::S16:
      return xla::PrimitiveType::S16;
    case ffi::S32:
      return xla::PrimitiveType::S32;
    case ffi::S64:
      return xla::PrimitiveType::S64;
    case ffi::U1:
      return xla::PrimitiveType::U1;
    case ffi::U2:
      return xla::PrimitiveType::U2;
    case ffi::U4:
      return xla::PrimitiveType::U4;
    case ffi::U8:
      return xla::PrimitiveType::U8;
    case ffi::U16:
      return xla::PrimitiveType::U16;
    case ffi::U32:
      return xla::PrimitiveType::U32;
    case ffi::U64:
      return xla::PrimitiveType::U64;
    case ffi::F16:
      return xla::PrimitiveType::F16;
    case ffi::F32:
      return xla::PrimitiveType::F32;
    case ffi::F64:
      return xla::PrimitiveType::F64;
    case ffi::BF16:
      return xla::PrimitiveType::BF16;
    case ffi::C64:
      return xla::PrimitiveType::C64;
    case ffi::C128:
      return xla::PrimitiveType::C128;
    case ffi::TOKEN:
      return xla::PrimitiveType::TOKEN;
    case ffi::F8E5M2:
      return xla::PrimitiveType::F8E5M2;
    case ffi::F8E4M3:
      return xla::PrimitiveType::F8E4M3;
    case ffi::F8E4M3FN:
      return xla::PrimitiveType::F8E4M3FN;
    case ffi::F8E4M3B11FNUZ:
      return xla::PrimitiveType::F8E4M3B11FNUZ;
    case ffi::F8E5M2FNUZ:
      return xla::PrimitiveType::F8E5M2FNUZ;
    case ffi::F8E4M3FNUZ:
      return xla::PrimitiveType::F8E4M3FNUZ;
    case ffi::F8E3M4:
      return xla::PrimitiveType::F8E3M4;
    case ffi::F4E2M1FN:
      return xla::PrimitiveType::F4E2M1FN;
    case ffi::F8E8M0FNU:
      return xla::PrimitiveType::F8E8M0FNU;
  }
}
}  // namespace

PyFfiContext::PyFfiContext(const XLA_FFI_Api* api,
                           XLA_FFI_ExecutionContext* ctx,
                           XLA_FFI_ExecutionStage stage)
    : api_(api), ctx_(ctx), stage_(stage) {}

PyFfiContext::Stage PyFfiContext::stage() const {
  return static_cast<PyFfiContext::Stage>(stage_);
}

absl::StatusOr<uintptr_t> PyFfiContext::stream() const {
  XLA_FFI_Stream_Get_Args args;
  args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.ctx = ctx_;
  args.stream = nullptr;
  if (XLA_FFI_Error* error = api_->XLA_FFI_Stream_Get(&args)) {
    return ffi::TakeStatus(error);
  }
  return absl::bit_cast<uintptr_t>(args.stream);
}

absl::StatusOr<int32_t> PyFfiContext::device_ordinal() const {
  XLA_FFI_DeviceOrdinal_Get_Args args;
  args.struct_size = XLA_FFI_DeviceOrdinal_Get_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device_ordinal = 0;
  if (XLA_FFI_Error* error = api_->XLA_FFI_DeviceOrdinal_Get(&args)) {
    return ffi::TakeStatus(error);
  }
  return args.device_ordinal;
}

PyFfiAnyBuffer::PyFfiAnyBuffer(DLDeviceType device_type, int32_t device_ordinal,
                               void* data, ffi::Span<int64_t const> dimensions,
                               ffi::DataType element_type, bool writeable)
    : device_type_(device_type),
      device_ordinal_(device_ordinal),
      data_(data),
      dimensions_(dimensions.begin(), dimensions.size()),
      element_type_(PrimitiveTypeForFfiDataType(element_type)),
      writeable_(writeable) {}

PyFfiAnyBuffer::PyFfiAnyBuffer(DLDeviceType device_type, int32_t device_ordinal,
                               ffi::AnyBuffer buf)
    : PyFfiAnyBuffer(device_type, device_ordinal, buf.untyped_data(),
                     buf.dimensions(), buf.element_type(),
                     /*writeable=*/false) {}

PyFfiAnyBuffer::PyFfiAnyBuffer(DLDeviceType device_type, int32_t device_ordinal,
                               ffi::Result<ffi::AnyBuffer> buf)
    : PyFfiAnyBuffer(device_type, device_ordinal, buf->untyped_data(),
                     buf->dimensions(), buf->element_type(),
                     /*writeable=*/true) {}

absl::StatusOr<xla::nb_dtype> PyFfiAnyBuffer::dtype() const {
  return xla::PrimitiveTypeToNbDtype(element_type_);
}

size_t PyFfiAnyBuffer::ndim() const { return dimensions_.size(); }

nb::tuple PyFfiAnyBuffer::shape() const {
  return xla::SpanToNbTuple(dimensions_);
}

bool PyFfiAnyBuffer::writeable() const { return writeable_; }

absl::StatusOr<xla::nb_numpy_ndarray> PyFfiAnyBuffer::NumpyArray() const {
  if (device_type_ != kDLCPU) {
    return absl::UnimplementedError(
        "Buffer.__array__ is only supported on CPU.");
  }

  TF_ASSIGN_OR_RETURN(auto dtype, this->dtype());
  xla::nb_numpy_ndarray array(dtype, dimensions_, /* strides= */ std::nullopt,
                              data_, nb::cast(this));

  // TODO(danfm): We don't seem to be allowed to set this flag like this
  // because the array doesn't own its data.
  // array.attr("flags").attr("writeable") = nb::bool_(writeable_);

  return array;
}

absl::StatusOr<nb::dict> PyFfiAnyBuffer::CudaArrayInterface() const {
  if (device_type_ != kDLCUDA) {
    return absl::UnimplementedError(
        "Buffer.__cuda_array_interface__ is only supported on CUDA.");
  }

  nb::dict result;
  result["shape"] = xla::SpanToNbTuple(dimensions_);
  TF_ASSIGN_OR_RETURN(result["typestr"],
                      TypeDescriptorForPrimitiveType(element_type_));
  result["data"] = nb::make_tuple(
      nb::int_(absl::bit_cast<std::uintptr_t>(data_)), !writeable_);
  result["version"] = nb::int_(2);
  return result;
}

absl::StatusOr<nb::capsule> PyFfiAnyBuffer::DLPack() const {
  auto pack = std::make_unique<DLPackTensor<DLManagedTensor>>();
  pack->tensor.manager_ctx = pack.get();
  pack->tensor.deleter = DLPackTensorDeleter;

  DLTensor& dt = pack->tensor.dl_tensor;
  dt.data = data_;
  dt.device = DLDevice{device_type_, device_ordinal_};
  dt.ndim = dimensions_.size();
  TF_ASSIGN_OR_RETURN(dt.dtype, xla::PrimitiveTypeToDLDataType(element_type_));
  pack->shape = std::vector<int64_t>(dimensions_.begin(), dimensions_.end());
  dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
  dt.strides = nullptr;
  dt.byte_offset = 0;

  // We cannot use nanobind's capsule object constructor because we need to
  // detect if the capsule name has been changed in the deleter, but nanobind
  // hides the underlying Python object from the deleter.
  nb::capsule capsule = nb::steal<nb::capsule>(
      PyCapsule_New(&pack.release()->tensor, kDlTensorCapsuleName,
                    [](PyObject* obj) noexcept {
                      DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(
                          PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
                      if (dlmt) {
                        DLPackTensorDeleter(dlmt);
                      } else {
                        // The tensor has been deleted. Clear any error from
                        // PyCapsule_GetPointer.
                        PyErr_Clear();
                      }
                    }));
  if (!capsule.ptr()) {
    throw nb::python_error();
  }

  return capsule;
}

absl::StatusOr<nb::capsule> PyFfiAnyBuffer::DLPackVersioned() const {
  auto pack = std::make_unique<DLPackTensor<DLManagedTensorVersioned>>();
  pack->tensor.version =
      DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
  pack->tensor.manager_ctx = pack.get();
  pack->tensor.deleter = DLPackTensorDeleter;
  pack->tensor.flags = writeable_ ? 0 : DLPACK_FLAG_BITMASK_READ_ONLY;

  DLTensor& dt = pack->tensor.dl_tensor;
  dt.data = data_;
  dt.device = DLDevice{device_type_, device_ordinal_};
  dt.ndim = dimensions_.size();
  TF_ASSIGN_OR_RETURN(dt.dtype, xla::PrimitiveTypeToDLDataType(element_type_));
  pack->shape = std::vector<int64_t>(dimensions_.begin(), dimensions_.end());
  dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
  dt.strides = nullptr;
  dt.byte_offset = 0;

  // We cannot use nanobind's capsule object constructor because we need to
  // detect if the capsule name has been changed in the deleter, but nanobind
  // hides the underlying Python object from the deleter.
  nb::capsule capsule = nb::steal<nb::capsule>(PyCapsule_New(
      &pack.release()->tensor, kDlTensorVersionedCapsuleName,
      [](PyObject* obj) noexcept {
        DLManagedTensorVersioned* dlmt = static_cast<DLManagedTensorVersioned*>(
            PyCapsule_GetPointer(obj, kDlTensorVersionedCapsuleName));
        if (dlmt) {
          DLPackTensorDeleter(dlmt);
        } else {
          // The tensor has been deleted. Clear any error from
          // PyCapsule_GetPointer.
          PyErr_Clear();
        }
      }));
  if (!capsule.ptr()) {
    throw nb::python_error();
  }

  return capsule;
}

nb::tuple PyFfiAnyBuffer::DLPackDevice() const {
  return nb::make_tuple(static_cast<int32_t>(device_type_), device_ordinal_);
}

void BuildFfiSubmodule(nb::module_& m) {
  tsl::ImportNumpy();

  nb::module_ ffi_module =
      m.def_submodule("ffi", "Python bindings for the XLA FFI.");

  nb::class_<PyFfiAnyBuffer> buffer(ffi_module, "Buffer");
  buffer.def_prop_ro("dtype", xla::ValueOrThrowWrapper(&PyFfiAnyBuffer::dtype));
  buffer.def_prop_ro("ndim", &PyFfiAnyBuffer::ndim);
  buffer.def_prop_ro("shape", &PyFfiAnyBuffer::shape);
  buffer.def_prop_ro("writeable", &PyFfiAnyBuffer::writeable);
  buffer.def(
      "__array__",
      [](PyFfiAnyBuffer self, nb::object dtype, nb::object copy) {
        if (!dtype.is_none()) {
          throw nb::value_error(
              "dtype parameter is not supported by Buffer.__array__.");
        }
        if (!copy.is_none() && nb::cast<bool>(copy)) {
          throw nb::value_error(
              "Buffer.__array__ with copy=True is not supported.");
        }
        return xla::ValueOrThrow(self.NumpyArray());
      },
      nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none());
  buffer.def_prop_ro(
      "__cuda_array_interface__",
      xla::ValueOrThrowWrapper(&PyFfiAnyBuffer::CudaArrayInterface));
  buffer.def(
      "__dlpack__",
      [](PyFfiAnyBuffer self, nb::object stream, nb::object max_version,
         nb::object dl_device, nb::object copy) {
        if (!copy.is_none() && nb::cast<bool>(copy)) {
          throw nb::value_error(
              "Buffer.__dlpack__ with copy=True is not supported.");
        }

        // Fall back on the non-versioned API if unsupported by the requested
        // max_version.
        nb::tuple max_version_tuple;
        int64_t max_version_major;
        if (!nb::try_cast<nb::tuple>(max_version, max_version_tuple) ||
            max_version_tuple.size() < 2 ||
            !nb::try_cast<int64_t>(max_version_tuple[0], max_version_major) ||
            max_version_major < 1) {
          return xla::ValueOrThrow(self.DLPack());
        }

        // TODO(danfm): Handle other optional inputs.
        return xla::ValueOrThrow(self.DLPackVersioned());
      },
      nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
      nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none());
  buffer.def("__dlpack_device__", &PyFfiAnyBuffer::DLPackDevice);

  nb::enum_<PyFfiContext::Stage>(ffi_module, "ExecutionStage")
      .value("INSTANTIATE", PyFfiContext::Stage::kInstantiate)
      .value("PREPARE", PyFfiContext::Stage::kPrepare)
      .value("INITIALIZE", PyFfiContext::Stage::kInitialize)
      .value("EXECUTE", PyFfiContext::Stage::kExecute)
      .export_values();

  nb::class_<PyFfiContext> context(ffi_module, "ExecutionContext");
  context.def_prop_ro("stage", &PyFfiContext::stage);
  context.def_prop_ro("device_ordinal",
                      xla::ValueOrThrowWrapper(&PyFfiContext::device_ordinal));
  context.def_prop_ro("stream",
                      xla::ValueOrThrowWrapper(&PyFfiContext::stream));
}

}  // namespace jax
