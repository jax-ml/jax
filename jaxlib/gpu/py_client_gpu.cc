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
#include "nanobind/nanobind.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/shape_util.h"
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
    if (ptype == xla::S1 || ptype == xla::S2 || ptype == xla::S4 ||
        ptype == xla::U1 || ptype == xla::U2 || ptype == xla::U4) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kUnimplemented,
                             absl::StrFormat("Unsupported primitive type: %s",
                                             PrimitiveType_Name(ptype)));
    }
    if (ptype == xla::TOKEN) {
      host_input_buffers[i] = nullptr;
      continue;
    }
    void* buf = new char[arg->size_bytes()];
    host_input_buffers[i] = buf;
    // TODO(b/238441608): Use pinned memory here to speed up the transfer.
    auto gpu_res =
        gpuMemcpyAsync(buf, arg.value().untyped_data(), arg->size_bytes(),
                       gpuMemcpyDeviceToHost, stream);
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
    nb::capsule base(host_input_buffers[i], [](void* ptr) noexcept {
      delete[] static_cast<char*>(ptr);
    });
    auto maybe_dtype = PrimitiveTypeToNbDtype(ptype);
    if (!maybe_dtype.ok()) {
      return xla::ffi::Error::Internal(maybe_dtype.status().ToString());
    }
    auto dtype = maybe_dtype.value();
    auto dims = absl::Span<const int64_t>(arg->dimensions().begin(),
                                          arg->dimensions().size());
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
    if (ptype == xla::S1 || ptype == xla::S2 || ptype == xla::S4 ||
        ptype == xla::U1 || ptype == xla::U2 || ptype == xla::U4) {
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
    if (strides == expected_strides) {
      auto gpu_res =
          gpuMemcpyAsync(ret->untyped_data(), array.data(), ret->size_bytes(),
                         gpuMemcpyHostToDevice, stream);
      CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
      continue;
    }
    void* temp = new char[ret->size_bytes()];
    temp_buffers.push_back(temp);
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
    plan->Execute(array.data(), temp);
    auto gpu_res = gpuMemcpyAsync(ret->untyped_data(), temp, ret->size_bytes(),
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
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
