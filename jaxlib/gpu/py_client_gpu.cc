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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_host_callback.h"
#include "xla/python/types.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace nb = nanobind;

namespace jax {
namespace JAX_GPU_NAMESPACE {

void XlaPythonGpuCallback(gpuStream_t stream, void** buffers,
                          const char* opaque, size_t opaque_len,
                          XlaCustomCallStatus* status) {
  // Ignore `descriptor` arg to callback
  buffers += 1;
  uint64_t descriptor;
  if (!absl::SimpleAtoi(opaque, &descriptor)) {
    throw xla::XlaRuntimeError("Invalid callback descriptor");
    return;
  }
  xla::CpuCallback* callback =
      absl::bit_cast<xla::CpuCallback*>(static_cast<uintptr_t>(descriptor));
  size_t arity = callback->num_args();
  std::vector<void*> host_input_buffers(arity);
  // Copy input GPU buffers to host
  for (size_t i = 0; i < arity; ++i) {
    const xla::CpuCallback::Arg& arg = callback->args()[i];
    if (arg.type == xla::TOKEN) {
      host_input_buffers[i] = nullptr;
      continue;
    }
    void* buf = new char[arg.size_in_bytes];
    host_input_buffers[i] = buf;
    // TODO(b/238441608): Use pinned memory here to speed up the transfer.
    auto gpu_res = gpuMemcpyAsync(buf, buffers[i], arg.size_in_bytes,
                                  gpuMemcpyDeviceToHost, stream);
    CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
  }
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  nb::gil_scoped_acquire gil;
  nb::tuple host_input_arrays = nb::steal<nb::tuple>(PyTuple_New(arity));
  for (size_t i = 0; i < arity; ++i) {
    xla::CpuCallback::Arg arg = callback->args()[i];
    if (arg.type == xla::TOKEN) {
      PyTuple_SET_ITEM(host_input_arrays.ptr(), i, nb::none().inc_ref().ptr());
      continue;
    }
    nb::capsule base(host_input_buffers[i], [](void* ptr) noexcept {
      delete[] static_cast<char*>(ptr);
    });
    auto array = xla::nb_numpy_ndarray(arg.dtype, arg.dims, arg.strides,
                                       const_cast<void*>(host_input_buffers[i]),
                                       /*base=*/base);
    array.attr("flags").attr("writeable") = nb::bool_(false);
    PyTuple_SET_ITEM(host_input_arrays.ptr(), i, array.inc_ref().ptr());
  }
  xla::EnterHostCallback();
  absl::StatusOr<nb::tuple> maybe_result_tuple =
      callback->Call(host_input_arrays);
  xla::LeaveHostCallback();
  if (!maybe_result_tuple.ok()) {
    absl::string_view msg = maybe_result_tuple.status().message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
    return;
  }
  nb::tuple result_tuple = maybe_result_tuple.value();
  std::vector<void*> temp_buffers;
  for (size_t i = 0; i < callback->results().size(); ++i) {
    xla::CpuCallback::Result result = callback->results()[i];
    if (result.type == xla::TOKEN) {
      continue;
    }
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    xla::nb_numpy_ndarray array =
        xla::nb_numpy_ndarray::ensure(std::move(output));
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    if (strides == result.expected_strides) {
      auto gpu_res =
          gpuMemcpyAsync(buffers[arity + i], array.data(), result.size_in_bytes,
                         gpuMemcpyHostToDevice, stream);
      CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
    } else {
      void* temp = new char[result.size_in_bytes];
      temp_buffers.push_back(temp);
      xla::TransposePlan::Options options;
      options.elem_size_in_bytes = xla::primitive_util::ByteWidth(result.type);
      options.dims = dims;
      options.permutation = result.reversed_layout;
      options.input_layout = xla::TransposePlan::Striding{strides};
      absl::StatusOr<std::shared_ptr<xla::TransposePlan>> plan =
          callback->transpose_cache().GetOrCreate(options);
      if (!plan.ok()) {
        throw xla::XlaRuntimeError(plan.status().ToString());
      }
      plan.value()->Execute(array.data(), temp);
      auto gpu_res =
          gpuMemcpyAsync(buffers[arity + i], temp, result.size_in_bytes,
                         gpuMemcpyHostToDevice, stream);
      CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
    }
  }
  nb::gil_scoped_release release;
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  for (int i = 0; i < temp_buffers.size(); ++i) {
    delete[] static_cast<char*>(temp_buffers[i]);
  }
}

// TODO(danfm): When compiled as part of a jaxlib plugin, this will register
// the custom call target in the plugin's registry. This won't affect
// registration via the Python API, but we should remove this once we have
// fully migrated to the plugin interface.
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "xla_python_gpu_callback", &XlaPythonGpuCallback,
    absl::AsciiStrToUpper(JAX_GPU_PLUGIN_NAME));

absl::Status XlaFfiPythonGpuCallback(
    gpuStream_t stream,
    std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>* callbacks,
    uint64_t index, xla::ffi::RemainingArgs args,
    xla::ffi::RemainingRets rets) {
  auto loaded_callback = llvm::dyn_cast_or_null<xla::PyCpuLoadedHostCallback>(
      callbacks->at(index).get());
  if (loaded_callback == nullptr) {
    return absl::InternalError(
        "Expected a PyCpuLoadedHostCallback, got something else.");
  }
  xla::CpuCallback* callback = loaded_callback->cpu_callback();
  size_t arity = args.size();
  std::vector<void*> host_input_buffers(arity);
  // Copy input GPU buffers to host
  for (size_t i = 0; i < arity; ++i) {
    auto arg = args.get<xla::ffi::AnyBuffer>(i);
    if (arg->element_type() == xla::TOKEN) {
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
  nb::tuple host_input_arrays = nb::steal<nb::tuple>(PyTuple_New(arity));
  for (size_t i = 0; i < arity; ++i) {
    auto arg = args.get<xla::ffi::AnyBuffer>(i);
    xla::PrimitiveType ptype = arg->element_type();
    if (ptype == xla::TOKEN) {
      PyTuple_SET_ITEM(host_input_arrays.ptr(), i, nb::none().inc_ref().ptr());
    } else {
      nb::capsule base(host_input_buffers[i], [](void* ptr) noexcept {
        delete[] static_cast<char*>(ptr);
      });
      TF_ASSIGN_OR_RETURN(auto dtype, xla::PrimitiveTypeToNbDtype(ptype));
      auto array = xla::nb_numpy_ndarray(dtype, arg->dimensions(), std::nullopt,
                                         host_input_buffers[i], base);
      array.attr("flags").attr("writeable") = nb::bool_(false);
      PyTuple_SET_ITEM(host_input_arrays.ptr(), i, array.inc_ref().ptr());
    }
  }

  xla::EnterHostCallback();
  // TODO(dsuo): Change this to use the Python vectorcall protocol, which allows
  // you to avoid constructing a tuple for the arguments.
  absl::StatusOr<nb::tuple> maybe_result_tuple =
      callback->FfiCall(host_input_arrays);
  xla::LeaveHostCallback();
  TF_ASSIGN_OR_RETURN(auto result_tuple, maybe_result_tuple);

  std::vector<void*> temp_buffers;
  for (size_t i = 0; i < rets.size(); ++i) {
    auto ret = rets.get<xla::ffi::AnyBuffer>(i).value();
    auto ptype = ret->element_type();
    if (ptype == xla::TOKEN) continue;
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    xla::nb_numpy_ndarray array =
        xla::nb_numpy_ndarray::ensure(std::move(output));
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    // We expect the output to be in default numpy layout.
    TF_ASSIGN_OR_RETURN(auto expected_shape, xla::ShapeUtil::MakeValidatedShape(
                                                 ptype, ret->dimensions()));
    auto expected_strides = ByteStridesForShape(expected_shape);
    if (strides == expected_strides) {
      auto gpu_res =
          gpuMemcpyAsync(ret->untyped_data(), array.data(), ret->size_bytes(),
                         gpuMemcpyHostToDevice, stream);
      CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
    } else {
      void* temp = new char[ret->size_bytes()];
      temp_buffers.push_back(temp);
      xla::TransposePlan::Options options;
      options.elem_size_in_bytes = xla::primitive_util::ByteWidth(ptype);
      absl::Span<int64_t const> dims(
          reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
      options.dims = dims;
      absl::InlinedVector<int64_t, 4> reversed_layout;
      reversed_layout.resize(expected_shape.rank());
      absl::c_reverse_copy(expected_shape.layout().minor_to_major(),
                           reversed_layout.begin());
      options.permutation = reversed_layout;
      options.input_layout = xla::TransposePlan::Striding{strides};
      TF_ASSIGN_OR_RETURN(auto plan,
                          callback->transpose_cache().GetOrCreate(options));
      plan->Execute(array.data(), temp);
      auto gpu_res =
          gpuMemcpyAsync(ret->untyped_data(), temp, ret->size_bytes(),
                         gpuMemcpyHostToDevice, stream);
      CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
    }
  }
  nb::gil_scoped_release release;
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  for (int i = 0; i < temp_buffers.size(); ++i) {
    delete[] static_cast<char*>(temp_buffers[i]);
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kXlaFfiPythonGpuCallback, XlaFfiPythonGpuCallback,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<gpuStream_t>>()
        .Ctx<xla::ffi::UserData<
            std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "xla_ffi_python_gpu_callback",
                         absl::AsciiStrToUpper(JAX_GPU_PLUGIN_NAME),
                         kXlaFfiPythonGpuCallback);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
