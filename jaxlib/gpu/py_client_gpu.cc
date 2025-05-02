/* Copyright 2022 The JAX Authors.

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

#include "jaxlib/gpu/py_client_gpu.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"
#include "nanobind/nanobind.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/ffi.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace nb = nanobind;

namespace jax {
namespace JAX_GPU_NAMESPACE {

struct GpuTransposePlanCache {
  static xla::ffi::TypeId id;
  explicit GpuTransposePlanCache(int capacity) : cache(capacity) {}
  xla::TransposePlanCache cache;
};
xla::ffi::TypeId GpuTransposePlanCache::id = {};

XLA_FFI_REGISTER_TYPE(xla::ffi::GetXlaFfiApi(), "GpuTransposePlanCache",
                      &GpuTransposePlanCache::id);

static xla::ffi::ErrorOr<std::unique_ptr<GpuTransposePlanCache>>
GpuTransposePlanCacheInstantiate(uint64_t index) {
  return std::make_unique<GpuTransposePlanCache>(16);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kGpuTransposePlanCacheInstantiate, GpuTransposePlanCacheInstantiate,
    xla::ffi::Ffi::BindInstantiate().Attr<uint64_t>("index"));
xla::ffi::Error XlaFfiPythonGpuCallback(gpuStream_t stream,
                                        xla::FfiLoadedHostCallbacks* callbacks,
                                        GpuTransposePlanCache* transpose_cache,
                                        uint64_t index,
                                        xla::ffi::RemainingArgs args,
                                        xla::ffi::RemainingRets rets) {
  size_t arity = args.size();
  std::vector<void*> host_input_buffers(arity);
  // Copy input GPU buffers to host
  for (size_t i = 0; i < arity; ++i) {
    auto arg = args.get<xla::ffi::AnyBuffer>(i);
    auto ptype = static_cast<xla::PrimitiveType>(arg->element_type());
    // TODO(b/395428868): Remove this check once we support subbyte types.
    if (ptype == xla::S1 || ptype == xla::U1) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kUnimplemented,
                             absl::StrFormat("Unsupported primitive type: %s",
                                             PrimitiveType_Name(ptype)));
    }
    if (ptype == xla::TOKEN) {
      host_input_buffers[i] = nullptr;
      continue;
    }
    size_t size_bytes = arg->size_bytes();
    // NOTE(dsuo): FFI arguments and return buffers are sized assuming
    // minimum 1-byte element sizes, even if the data itself is packed. We
    // assume that 2-bit and 4-bit types are packed.
    size_t bits_per_element = xla::primitive_util::BitWidth(ptype);
    if (bits_per_element == 2 || bits_per_element == 4) {
      size_bytes = arg->element_count() * bits_per_element / 8;
    }
    host_input_buffers[i] = new char[size_bytes];
    // TODO(b/238441608): Use pinned memory here to speed up the transfer.
    auto gpu_res =
        gpuMemcpyAsync(host_input_buffers[i], arg.value().untyped_data(),
                       size_bytes, gpuMemcpyDeviceToHost, stream);
    CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
  }
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  nb::gil_scoped_acquire gil;
  auto callback = nb::borrow<nb::callable>(
      static_cast<PyObject*>(callbacks->callbacks[index]));
  nb::tuple host_input_arrays = nb::steal<nb::tuple>(PyTuple_New(arity));
  for (size_t i = 0; i < arity; ++i) {
    auto arg = args.get<xla::ffi::AnyBuffer>(i);
    auto ptype = static_cast<xla::PrimitiveType>(arg->element_type());
    if (ptype == xla::TOKEN) {
      PyTuple_SET_ITEM(host_input_arrays.ptr(), i, nb::none().inc_ref().ptr());
      continue;
    }
    auto maybe_dtype = PrimitiveTypeToNbDtype(ptype);
    if (!maybe_dtype.ok()) {
      return xla::ffi::Error::Internal(maybe_dtype.status().ToString());
    }
    auto dtype = maybe_dtype.value();
    auto dims = absl::Span<const int64_t>(arg->dimensions().begin(),
                                          arg->dimensions().size());
    // TODO(b/402422886): Remove this once we form Jax arrays directly instead
    // of packing/unpacking to/from numpy arrays.
    // We pass in data using default numpy layout i.e., std::nullopt.
    size_t bits_per_element = xla::primitive_util::BitWidth(ptype);
    if (bits_per_element == 2 || bits_per_element == 4) {
      // NOTE(dsuo): FFI arguments and return buffers are sized assuming
      // minimum 1-byte element sizes, even if the data itself is packed. We
      // assume that 2-bit and 4-bit types are packed.
      auto size_bytes = arg->element_count() * bits_per_element / 8;
      auto buffer = xla::UnpackIntN(
          bits_per_element, static_cast<const char*>(host_input_buffers[i]),
          size_bytes);
      delete[] static_cast<char*>(host_input_buffers[i]);
      host_input_buffers[i] = buffer.release();
    }
    nb::capsule base(host_input_buffers[i], [](void* ptr) noexcept {
      delete[] static_cast<char*>(ptr);
    });
    auto array = xla::nb_numpy_ndarray(dtype, dims, std::nullopt,
                                       host_input_buffers[i], base);
    array.attr("flags").attr("writeable") = nb::bool_(false);
    PyTuple_SET_ITEM(host_input_arrays.ptr(), i, array.inc_ref().ptr());
  }

  xla::EnterHostCallback();
  // TODO(dsuo): Change this to use the Python vectorcall protocol, which allows
  // you to avoid constructing a tuple for the arguments.
  nb::tuple result_tuple;
  try {
    auto result_object = callback(*nb::borrow<nb::args>(host_input_arrays));
    result_tuple = nb::cast<nb::tuple>(result_object);
  } catch (nb::python_error& e) {
    return xla::ffi::Error::Internal(
        absl::StrFormat("CpuCallback error calling callback: %s", e.what()));
  }
  xla::LeaveHostCallback();

  std::vector<void*> temp_buffers;
  for (size_t i = 0; i < rets.size(); ++i) {
    auto ret = rets.get<xla::ffi::AnyBuffer>(i).value();
    auto ptype = static_cast<xla::PrimitiveType>(ret->element_type());
    // TODO(b/395428868): Remove this check once we support subbyte types.
    if (ptype == xla::S1 || ptype == xla::U1) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kUnimplemented,
                             absl::StrFormat("Unsupported primitive type: %s",
                                             PrimitiveType_Name(ptype)));
    }
    if (ptype == xla::TOKEN) continue;
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    auto array = xla::nb_numpy_ndarray::ensure(std::move(output));
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    // We expect the output to be in default numpy layout.
    auto dims = absl::Span<const int64_t>(ret->dimensions().begin(),
                                          ret->dimensions().size());
    auto maybe_expected_shape = xla::ShapeUtil::MakeValidatedShape(ptype, dims);
    if (!maybe_expected_shape.ok()) {
      return xla::ffi::Error::Internal(
          maybe_expected_shape.status().ToString());
    }
    auto expected_shape = maybe_expected_shape.value();
    auto expected_strides = xla::ByteStridesForShape(expected_shape);

    const void* data = array.data();
    size_t size_bytes = array.size() * array.itemsize();
    if (strides != expected_strides) {
      xla::TransposePlan::Options options;
      options.elem_size_in_bytes = xla::primitive_util::ByteWidth(ptype);
      options.dims = absl::Span<int64_t const>(
          reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
      absl::InlinedVector<int64_t, 4> reversed_layout;
      reversed_layout.resize(expected_shape.dimensions().size());
      absl::c_reverse_copy(expected_shape.layout().minor_to_major(),
                           reversed_layout.begin());
      options.permutation = reversed_layout;
      options.input_layout = xla::TransposePlan::Striding{strides};
      auto maybe_plan = transpose_cache->cache.GetOrCreate(options);
      if (!maybe_plan.ok()) {
        return xla::ffi::Error::Internal(maybe_plan.status().ToString());
      }
      auto plan = maybe_plan.value();
      void* temp = new char[size_bytes];
      temp_buffers.push_back(temp);
      plan->Execute(data, temp);
      data = temp;
    }

    // TODO(b/402422886): Remove this once we form Jax arrays directly instead
    // of packing/unpacking to/from numpy arrays.
    std::unique_ptr<char[]> buffer;
    size_t bits_per_element = xla::primitive_util::BitWidth(ptype);
    if (bits_per_element == 2 || bits_per_element == 4) {
      // NOTE(dsuo): FFI arguments and return buffers are sized assuming
      // minimum 1-byte element sizes, even if the data itself is packed. We
      // assume that 2-bit and 4-bit types are packed.
      buffer = xla::PackIntN(bits_per_element, static_cast<const char*>(data),
                             size_bytes);
      data = buffer.get();
      size_bytes = (size_bytes * bits_per_element) / 8;
    }

    auto gpu_res = gpuMemcpyAsync(ret->untyped_data(), data, size_bytes,
                                  gpuMemcpyHostToDevice, stream);
    CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
  }
  nb::gil_scoped_release release;
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  for (int i = 0; i < temp_buffers.size(); ++i) {
    delete[] static_cast<char*>(temp_buffers[i]);
  }
  return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kXlaFfiPythonGpuCallback, XlaFfiPythonGpuCallback,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<gpuStream_t>>()
        .Ctx<xla::ffi::UserData<xla::FfiLoadedHostCallbacks>>()
        .Ctx<xla::ffi::State<GpuTransposePlanCache>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "xla_ffi_python_gpu_callback",
                         absl::AsciiStrToUpper(JAX_GPU_PLUGIN_NAME),
                         {kGpuTransposePlanCacheInstantiate, nullptr, nullptr,
                          kXlaFfiPythonGpuCallback});
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "xla_ffi_partitioned_python_gpu_callback",
                         absl::AsciiStrToUpper(JAX_GPU_PLUGIN_NAME),
                         {kGpuTransposePlanCacheInstantiate, nullptr, nullptr,
                          kXlaFfiPythonGpuCallback});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kXlaBufferPythonGpuCallback,
#ifdef JAX_GPU_CUDA
    (jax::XlaBufferCallback<kDLCUDA>),
#else
    (jax::XlaBufferCallback<kDLROCM>),
#endif
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::DeviceOrdinal>()
        .Ctx<xla::ffi::FfiApi>()
        .Ctx<xla::ffi::FfiExecutionContext>()
        .Ctx<xla::ffi::UserData<xla::FfiLoadedHostCallbacks>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kXlaBufferPythonGpuCommandBufferCallback,
#ifdef JAX_GPU_CUDA
    (jax::XlaBufferCallback<kDLCUDA>),
#else
    (jax::XlaBufferCallback<kDLROCM>),
#endif
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::DeviceOrdinal>()
        .Ctx<xla::ffi::FfiApi>()
        .Ctx<xla::ffi::FfiExecutionContext>()
        .Ctx<xla::ffi::UserData<xla::FfiLoadedHostCallbacks>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets(),
        {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "xla_buffer_python_gpu_callback",
                         absl::AsciiStrToUpper(JAX_GPU_PLUGIN_NAME),
                         kXlaBufferPythonGpuCallback);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "xla_buffer_python_gpu_command_buffer_callback",
                         absl::AsciiStrToUpper(JAX_GPU_PLUGIN_NAME),
                         kXlaBufferPythonGpuCommandBufferCallback,
                         XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);


}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
