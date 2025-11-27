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

#include "jaxlib/dlpack.h"

#include <Python.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_array.h"
#include "jaxlib/py_client.h"
#include "jaxlib/py_user_context.h"
#include "jaxlib/python_ref_manager.h"
#include "jaxlib/util.h"
#include "xla/layout.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/dlpack_types.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/types.h"
#include "xla/python/version.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace ifrt = xla::ifrt;
namespace nb = nanobind;

namespace jax {
namespace {

const char* const kDlTensorCapsuleName = "dltensor";

struct DLPackTensor {
  ~DLPackTensor();

  // `buffer_reference` is populated if we have shared (read-only) access.
  nb::object buffer_reference;

  // `external_reference` is always populated.
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference;

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;
};

DLPackTensor::~DLPackTensor() {
  // We must release the external reference first before deleting the array.
  external_reference.reset();
  if (buffer_reference) {
    GlobalPyRefManager()->AddGarbage(
        absl::MakeSpan(&buffer_reference, /*size=*/1));
  }
}

void DLPackTensorDeleter(DLManagedTensor* t) {
  if (t) {
    delete static_cast<DLPackTensor*>(t->manager_ctx);
  }
}

absl::StatusOr<std::vector<int64_t>> StridesToLayout(
    absl::Span<int64_t const> dims, absl::Span<int64_t const> strides) {
  CHECK_EQ(dims.size(), strides.size());
  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  absl::c_sort(minor_to_major, [&](int a, int b) {
    if (strides[a] < strides[b]) {
      return true;
    }
    if (strides[a] > strides[b]) {
      return false;
    }
    // If two dimensions have the same stride, prefer the major-to-minor
    // interpretation of the ordering, since that's what JAX wants.
    return b < a;
  });
  int64_t stride = 1;
  for (int64_t d : minor_to_major) {
    if (dims[d] > 1 && strides[d] != stride) {
      return xla::Unimplemented(
          "Only DLPack tensors with trivial (compact) striding are supported; "
          "i.e., tensors whose striding represents a transposition of the "
          "underlying buffer but not broadcasting. Dimensions were: [%s], "
          "strides were [%s].",
          absl::StrJoin(dims, ","), absl::StrJoin(strides, ","));
    }
    stride *= dims[d];
  }
  return minor_to_major;
}

absl::StatusOr<DLDeviceType> DLDeviceTypeForDevice(
    const xla::PjRtDevice& device) {
  if (device.client()->platform_id() == xla::CpuId()) {
    return kDLCPU;
  } else if (device.client()->platform_id() == xla::CudaId()) {
    return kDLCUDA;
  } else if (device.client()->platform_id() == xla::RocmId()) {
    return kDLROCM;
  }
  return xla::InvalidArgument("Device %s cannot be used as a DLPack device.",
                              device.DebugString());
}

absl::StatusOr<DLDevice> DLDeviceForDevice(const xla::PjRtDevice& device) {
  DLDevice context;
  TF_ASSIGN_OR_RETURN(context.device_type, DLDeviceTypeForDevice(device));
  context.device_id = device.local_hardware_id().value();
  return context;
}

absl::Status VerifyDType(const DLTensor& dl_tensor) {
  if (dl_tensor.dtype.bits % 8 != 0) {
    return xla::InvalidArgument(
        "Unsupported DLPack tensor dtype: bits should be a multiple of 8, got "
        "%d",
        dl_tensor.dtype.bits);
  }

  if (dl_tensor.dtype.lanes != 1) {
    return xla::InvalidArgument(
        "Unsupported DLPack tensor dtype: lanes should be equal to 1, got %d",
        dl_tensor.dtype.lanes);
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<int64_t>> GetByteStrides(const DLTensor& dl_tensor) {
  TF_RETURN_IF_ERROR(VerifyDType(dl_tensor));

  // Convert element strides from the number of elements to the number of bytes.
  std::vector<int64_t> strides;
  strides.reserve(dl_tensor.ndim);
  for (int i = 0; i < dl_tensor.ndim; ++i) {
    strides.push_back(dl_tensor.strides[i] * dl_tensor.dtype.bits / 8);
  }
  return strides;
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> MakePjrtBuffer(
    xla::PjRtDevice& device, ::DLManagedTensor* dlmt, const xla::Shape& shape,
    xla::PrimitiveType element_type, absl::Span<int64_t const> dimensions,
    std::optional<bool> copy = std::nullopt,
    std::optional<std::intptr_t> stream = std::nullopt) {
  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {
    on_delete_callback = [dlmt]() { dlmt->deleter(dlmt); };
  }

  void* data =
      static_cast<char*>(dlmt->dl_tensor.data) + dlmt->dl_tensor.byte_offset;

  // On CPU, creating a view may fail because of unaligned data buffer
  // in which case we'll fallback to copy. On non-CPU, array-api copy
  // semantics is handled in dlpack._place_array function.
  bool fallback_to_copy =
      !copy.has_value() && dlmt->dl_tensor.device.device_type == kDLCPU;

  // Create a view.
  if (!copy.value_or(false)) {
    auto result = device.client()->CreateViewOfDeviceBuffer(
        data, shape, *device.default_memory_space(), on_delete_callback,
        stream);
    if (!(result.status().code() == absl::StatusCode::kInvalidArgument &&
          fallback_to_copy)) {
      return result;
    }
  }

  // Convert tensor strides (expressed in number of elements) to byte strides.
  std::optional<std::vector<int64_t>> byte_strides;
  if (dlmt->dl_tensor.strides) {
    TF_ASSIGN_OR_RETURN(byte_strides, GetByteStrides(dlmt->dl_tensor));
  }

  TF_ASSIGN_OR_RETURN(auto* memory_space, device.default_memory_space());

  // Create a copy.
  return device.client()->BufferFromHostBuffer(
      data, element_type, dimensions, byte_strides,
      xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy,
      on_delete_callback, memory_space, /*device_layout=*/nullptr);
}

}  // namespace

absl::StatusOr<nb::capsule> BufferToDLPackManagedTensor(
    nb::handle py_buffer, std::optional<std::intptr_t> stream) {
  ifrt::Array* ifrt_array = nb::cast<PyArray>(py_buffer).ifrt_array();
  if (ifrt_array == nullptr) {
    return xla::Unimplemented(
        "BufferToDLPackManagedTensor called on deleted array.");
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr == nullptr) {
    throw xla::XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  xla::PjRtBuffer* pjrt_buffer = arr->pjrt_buffers().front().get();

  if (pjrt_buffer->IsTuple()) {
    return xla::Unimplemented(
        "BufferToDLPackManagedTensor is not implemented for tuple "
        "buffers.");
  }
  if (pjrt_buffer->has_dynamic_dimensions()) {
    return xla::Unimplemented("DynamicShape is not implemented in DLPack.");
  }

  auto pack = std::make_unique<DLPackTensor>();
  DLTensor& dt = pack->tensor.dl_tensor;
  {
    // AcquireExternalReference may block; there are no API guarantees.
    GlobalPyRefManager()->CollectGarbage();
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(pack->external_reference,
                        pjrt_buffer->AcquireExternalReference());
    if (stream) {
      TF_RETURN_IF_ERROR(
          pack->external_reference->WaitUntilBufferReadyOnStream(*stream));
    } else {
      TF_RETURN_IF_ERROR(
          AwaitBuffersReady(absl::MakeConstSpan(&ifrt_array, 1)));
    }
  }
  pack->buffer_reference = nb::borrow<nb::object>(py_buffer);

  dt.data = pack->external_reference->OpaqueDeviceMemoryDataPointer();
  pack->tensor.manager_ctx = pack.get();
  pack->tensor.deleter = DLPackTensorDeleter;
  TF_ASSIGN_OR_RETURN(dt.device, DLDeviceForDevice(*pjrt_buffer->device()));
  dt.device.device_id = pjrt_buffer->device()->local_hardware_id().value();
  dt.ndim = pjrt_buffer->dimensions().size();
  TF_ASSIGN_OR_RETURN(dt.dtype,
                      PrimitiveTypeToDLDataType(pjrt_buffer->element_type()));

  pack->shape = std::vector<int64_t>(pjrt_buffer->dimensions().begin(),
                                     pjrt_buffer->dimensions().end());

  // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
  xla::Layout xla_layout = pjrt_buffer->layout()->xla_layout();
  pack->strides = StridesForShape(pjrt_buffer->element_type(),
                                  pjrt_buffer->dimensions(), xla_layout);

  dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
  dt.strides = reinterpret_cast<std::int64_t*>(pack->strides.data());
  dt.byte_offset = 0;

  // We cannot use nanobind's capsule object constructor because we need to
  // detect if the capsule name has been changed in the deleter, but nanobind
  // hides the underlying Python object from the deleter.
  nb::capsule capsule = nb::steal<nb::capsule>(
      PyCapsule_New(&pack.release()->tensor, kDlTensorCapsuleName,
                    [](PyObject* obj) noexcept {
#if PY_VERSION_HEX < 0x030C0000
                      PyObject *type, *value, *traceback;
                      PyErr_Fetch(&type, &value, &traceback);
#else   // PY_VERSION_HEX < 0x030C0000
                      PyObject* exc = PyErr_GetRaisedException();
#endif  // PY_VERSION_HEX < 0x030C0000
                      DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(
                          PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
                      if (dlmt) {
                        DLPackTensorDeleter(dlmt);
                      }
    // PyCapsule_GetPointer may have raised. Restore the
    // previous exception if there was one.
#if PY_VERSION_HEX < 0x030C0000
                      PyErr_Restore(type, value, traceback);
#else   // PY_VERSION_HEX < 0x030C0000
                      PyErr_SetRaisedException(exc);
#endif  // PY_VERSION_HEX < 0x030C0000
                    }));
  if (!capsule.ptr()) {
    throw nb::python_error();
  }
  return capsule;
}

absl::StatusOr<nb::object> DLPackManagedTensorToBuffer(
    const nb::capsule& tensor, ifrt::Device* ifrt_device,
    nb_class_ptr<PyClient> client, std::optional<std::intptr_t> stream,
    std::optional<bool> copy) {
  ifrt::PjRtDevice* device =
      llvm::dyn_cast_or_null<ifrt::PjRtDevice>(ifrt_device);
  if (device == nullptr) {
    throw xla::XlaRuntimeError(
        "DLPack is supported for PjRt-compatible backends only.");
  }
  if (!device->IsAddressable()) {
    throw xla::XlaRuntimeError(
        "DLPack is only supported for devices addressable by the current "
        "process.");
  }
  if (std::string_view(tensor.name()) != kDlTensorCapsuleName) {
    return xla::InvalidArgument(
        "DLPack tensor must be a capsule with name \"dltensor\", got \"%s\". "
        "Note that a DLPack tensor may be consumed at most once.",
        std::string_view(tensor.name()));
  }
  DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(tensor.data());
  if (dlmt->dl_tensor.ndim < 0) {
    return xla::InvalidArgument(
        "Number of dimensions in DLManagedTensor must be nonnegative, got %d",
        dlmt->dl_tensor.ndim);
  }
  absl::Span<int64_t const> dimensions(
      reinterpret_cast<int64_t*>(dlmt->dl_tensor.shape), dlmt->dl_tensor.ndim);
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType element_type,
                      xla::DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype));

  bool has_custom_layout = dlmt->dl_tensor.strides != nullptr;
  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        reinterpret_cast<int64_t*>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    TF_ASSIGN_OR_RETURN(minor_to_major, StridesToLayout(dimensions, strides));
  } else {
    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      element_type, dimensions, minor_to_major);

  TF_ASSIGN_OR_RETURN(auto pjrt_buffer,
                      MakePjrtBuffer(*device->pjrt_device(), dlmt, shape,
                                     element_type, dimensions, copy, stream));

  // We have taken ownership of the array inside the capsule; make sure the
  // capsule it cannot be used again.
  PyCapsule_SetName(tensor.ptr(), "used_dltensor");
  PyCapsule_SetDestructor(tensor.ptr(), nullptr);

  auto* ifrt_client =
      llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(client->ifrt_client());
  if (ifrt_client == nullptr) {
    throw xla::XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  PyUserContextScope user_context_scope;
  TF_ASSIGN_OR_RETURN(
      auto ifrt_array,
      ifrt_client->CreatePjRtArray(std::move(pjrt_buffer), has_custom_layout));
  return PyArray::MakeFromSingleDeviceArray(std::move(client),
                                            std::move(ifrt_array), false, true);
}

absl::StatusOr<nanobind::dlpack::dtype> PrimitiveTypeToNbDLDataType(
    xla::PrimitiveType type) {
  TF_ASSIGN_OR_RETURN(DLDataType dl_type, PrimitiveTypeToDLDataType(type));

  nanobind::dlpack::dtype nb_type;
  nb_type.lanes = dl_type.lanes;
  nb_type.bits = dl_type.bits;
  nb_type.code = dl_type.code;

  return nb_type;
}

}  // namespace jax
