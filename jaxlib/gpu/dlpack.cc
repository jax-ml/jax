#include "third_party/dlpack/include/dlpack/dlpack.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace ffi = xla::ffi;
namespace nb = nanobind;

absl::StatusOr<DLDataType> DataTypeToDL(ffi::DataType dt) {
  switch (dt) {
    case ffi::DataType::PRED:
      return DLDataType{kDLBool, 8, 1};
    case ffi::DataType::S8:
      return DLDataType{kDLInt, 8, 1};
    case ffi::DataType::S16:
      return DLDataType{kDLInt, 16, 1};
    case ffi::DataType::S32:
      return DLDataType{kDLInt, 32, 1};
    case ffi::DataType::S64:
      return DLDataType{kDLInt, 64, 1};
    case ffi::DataType::U8:
      return DLDataType{kDLUInt, 8, 1};
    case ffi::DataType::U16:
      return DLDataType{kDLUInt, 16, 1};
    case ffi::DataType::U32:
      return DLDataType{kDLUInt, 32, 1};
    case ffi::DataType::U64:
      return DLDataType{kDLUInt, 64, 1};
    case ffi::DataType::F16:
      return DLDataType{kDLFloat, 16, 1};
    case ffi::DataType::F32:
      return DLDataType{kDLFloat, 32, 1};
    case ffi::DataType::F64:
      return DLDataType{kDLFloat, 64, 1};
    case ffi::DataType::BF16:
      return DLDataType{kDLBfloat, 16, 1};
    case ffi::DataType::C64:
      return DLDataType{kDLComplex, 64, 1};
    case ffi::DataType::C128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      std::stringstream ss;
      ss << "FFI type " << dt << " has no DLPack equivalent";
      return absl::UnimplementedError(ss.str());
  }
}

bool DLDataTypesEqual(DLDataType a, DLDataType b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

ffi::Error StatusToError(const absl::Status& status) {
  return ffi::Error{static_cast<XLA_FFI_Error_Code>(status.code()),
                    std::string(status.message())};
}

absl::StatusOr<std::unique_ptr<DLManagedTensor>> BufferBaseToDLTensor(
    ffi::BufferBase buf) {
  auto maybe_dtype = DataTypeToDL(buf.dtype);
  if (!maybe_dtype.ok()) {
    return maybe_dtype.status();
  }
  gpuPointerAttributes attrs;
  CHECK_EQ(gpuPointerGetAttributes(&attrs, buf.data), gpuSuccess)
      << "Failed to gpuPointerGetAttributes";
#if defined(JAX_GPU_CUDA)
  auto device_type = kDLCUDA;
#elif defined(JAX_GPU_HIP)
  auto device_type = kDLROCM;
#endif
  auto dl_tensor = DLTensor{
      .data = buf.data,
      .device = DLDevice{.device_type = device_type, .device_id = attrs.device},
      .ndim = static_cast<int32_t>(buf.dimensions.size()),
      .dtype = maybe_dtype.value(),
      .shape = const_cast<int64_t*>(buf.dimensions.begin()),
      .strides = nullptr,  // row-major
      .byte_offset = 0,
  };
  return std::make_unique<DLManagedTensor>(dl_tensor, /*manager_ctx=*/nullptr,
                                           /*deleter=*/nullptr);
}

nb::capsule WrapDLTensor(std::unique_ptr<DLManagedTensor> tensor) {
  return nb::capsule(
      reinterpret_cast<void*>(tensor.release()), "dltensor",
      [](void* ptr) noexcept { delete static_cast<DLManagedTensor*>(ptr); });
}

DLManagedTensor* UnwrapDLTensor(nb::object o) {
  CHECK(nb::isinstance<nb::capsule>(o));
  auto c = nb::cast<nb::capsule>(o);
  CHECK_EQ(std::string_view(c.name()), "dltensor");
  PyCapsule_SetName(c.ptr(), "used_dltensor");
  PyCapsule_SetDestructor(c.ptr(), nullptr);
  return reinterpret_cast<DLManagedTensor*>(c.data());
}

ffi::Error DlpackCallbackImpl(gpuStream_t stream, PyObject* callback,
                              bool mutable_results,
                              ffi::RemainingArgs packed_args,
                              ffi::RemainingResults packed_results) {
  CHECK_GT(packed_results.size(), 0);

  std::vector<std::unique_ptr<DLManagedTensor>> tensors;
  tensors.reserve(packed_args.size());
  for (int i = 0; i < packed_args.size(); ++i) {
    auto maybe_buf = packed_args.get<ffi::BufferBase>(i);
    CHECK(maybe_buf.has_value()) << "Operand buffer " << i << " is missing";
    auto maybe_tensor = BufferBaseToDLTensor(*maybe_buf);
    if (auto& status = maybe_tensor.status(); !status.ok()) {
      return StatusToError(std::move(status));
    }
    tensors.emplace_back(std::move(*maybe_tensor));
  }

  if (mutable_results) {
    tensors.reserve(tensors.size() + packed_results.size());
    for (int i = 0; i < packed_results.size(); ++i) {
      auto maybe_buf = packed_results.get<ffi::BufferBase>(i);
      CHECK(maybe_buf.has_value()) << "Result buffer " << i << " is missing";
      auto maybe_tensor = BufferBaseToDLTensor(*maybe_buf);
      if (auto& status = maybe_tensor.status(); !status.ok()) {
        return StatusToError(status);
      }
      tensors.emplace_back(std::move(*maybe_tensor));
    }
  }

  nb::gil_scoped_acquire gil;
  auto args = nb::steal<nb::list>(PyList_New(tensors.size()));
  for (int i = 0; i < tensors.size(); ++i) {
    // nanobind::list does not expect nullptr values, so we have to fallback
    // to the C API here.
    PyList_SetItem(args.ptr(), i,
                   WrapDLTensor(std::move(tensors[i])).release().ptr());
  }

  nb::object obj;
  try {
    obj = nb::borrow<nb::object>(callback)(*args);
  } catch (nb::python_error& e) {
    return ffi::Error{XLA_FFI_Error_Code_INTERNAL,
                      absl::StrCat("DlpackCallback error: ", e.what())};
  }

  if (mutable_results) {
    if (!obj.is_none()) {
      return ffi::Error{
          XLA_FFI_Error_Code_INTERNAL,
          "DlpackCallback with mutable_results=True returned a non-None value"};
    }
    return ffi::Error::Success();
  }

  auto results = nb::cast<nb::list>(obj);
  for (int i = 0; i < packed_results.size(); ++i) {
    auto* managed_tensor = UnwrapDLTensor(nb::cast<nb::capsule>(results[i]));
    auto cleanup = absl::MakeCleanup([&managed_tensor] {
      if (managed_tensor->deleter != nullptr)
        managed_tensor->deleter(managed_tensor);
    });
    auto tensor = managed_tensor->dl_tensor;
    auto maybe_buf = packed_results.get<ffi::BufferBase>(i);
    CHECK(maybe_buf.has_value()) << "Result buffer " << i << " is missing";
    auto buf = maybe_buf.value();
    auto expected_dtype = DataTypeToDL(buf.dtype);
    if (!expected_dtype.ok()) {
      return ffi::Error{
          static_cast<XLA_FFI_Error_Code>(expected_dtype.status().code()),
          std::string(expected_dtype.status().message())};
    }
    if (!DLDataTypesEqual(tensor.dtype, expected_dtype.value())) {
      return ffi::Error{
          XLA_FFI_Error_Code_INTERNAL,
          "The callback returned a DLPack tensor with an unexpected dtype"};
    }
    // TODO(slebedev): Ensure that strides are minor-to-major.
    auto expected_shape =
        absl::MakeConstSpan(buf.dimensions.begin(), buf.dimensions.end());
    if (expected_shape != absl::MakeConstSpan(tensor.shape, tensor.ndim)) {
      return ffi::Error{
          XLA_FFI_Error_Code_INTERNAL,
          "The callback returned a DLPack tensor with unexpected shape"};
    }
    size_t size = 1;
    for (int d = 0; d < tensor.ndim; ++d) {
      size *= tensor.shape[d];
    }
    size *= tensor.dtype.bits / 8;
    CHECK_EQ(gpuMemcpyAsync(buf.data, tensor.data, size, gpuMemcpyDefault),
             gpuSuccess)
        << "Failed to gpuMemcpyAsync";
  }

  nb::gil_scoped_release release;
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  return ffi::Error::Success();
}

XLA_FFI_Error* DlpackCallback(XLA_FFI_CallFrame* call_frame) {
  static const auto* kDlpackCallImpl =
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<gpuStream_t>>()
          .Attr<ffi::Pointer<PyObject>>("callback")
          .Attr<bool>("mutable_results")
          .RemainingArgs()
          .RemainingResults()
          .To(DlpackCallbackImpl)
          .release();
  return kDlpackCallImpl->Call(call_frame);
}

NB_MODULE(_dlpack, m) {
  m.def("registrations", []() {
    nb::dict dict;
    dict["dlpack_callback"] =
        nb::capsule(reinterpret_cast<void*>(+DlpackCallback));
    return dict;
  });
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
