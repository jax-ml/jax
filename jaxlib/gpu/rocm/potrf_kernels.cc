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

#include "jaxlib/gpu/rocm/potrf_kernels.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/make_batch_pointers.h"
#include "jaxlib/gpu/rocm/potrf.h"
#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

namespace {

template <typename T>
ffi::Error RocPotrfImpl(int64_t batch, int64_t size, gpuStream_t stream,
                        ffi::ScratchAllocator& scratch, bool lower,
                        ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                        ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(size));

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  int out_step = n * n;
  for (auto i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(
        solver::RocPotrf(stream, lower, n, out_data, info_data));
    out_data += out_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <typename T>
ffi::Error RocPotrfBatchedImpl(int64_t batch, int64_t size, gpuStream_t stream,
                               ffi::ScratchAllocator& scratch, bool lower,
                               ffi::AnyBuffer a,
                               ffi::Result<ffi::AnyBuffer> out,
                               ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(size));
  FFI_ASSIGN_OR_RETURN(
      auto batch_ptrs,
      AllocateWorkspace<T*>(scratch, batch, "rocsolver batched potrf"));

  auto a_data = a.untyped_data();
  auto out_data = out->untyped_data();
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  MakeBatchPointersAsync(stream, out_data, batch_ptrs, batch,
                         sizeof(T) * n * n);
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuGetLastError());

  FFI_RETURN_IF_ERROR_STATUS(
      solver::RocPotrfBatched(stream, lower, n, batch_ptrs, info_data, batch));
  return ffi::Error::Success();
}

ffi::Error RocPotrfDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                            bool lower, ffi::AnyBuffer a,
                            ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = a.element_type();
  if (dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "The input and output to rocsolver potrf must have the same element "
        "type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  if (rows != cols) {
    return ffi::Error::InvalidArgument(
        "The input matrix to rocsolver potrf must be square");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out",
                 "rocsolver_potrf"));
  FFI_RETURN_IF_ERROR(
      CheckShape(info->dimensions(), batch, "info", "rocsolver_potrf"));
  if (batch > 1) {
    SOLVER_DISPATCH_IMPL(RocPotrfBatchedImpl, batch, rows, stream, scratch,
                         lower, a, out, info);
  } else {
    SOLVER_DISPATCH_IMPL(RocPotrfImpl, batch, rows, stream, scratch, lower, a,
                         out, info);
  }
  return ffi::Error::InvalidArgument(
      absl::StrFormat("Unsupported dtype %s in rocsolver_potrf",
                      absl::FormatStreamed(dataType)));
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RocPotrfFfi, RocPotrfDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Attr<bool>("lower")
        .Arg<ffi::AnyBuffer>()        // a
        .Ret<ffi::AnyBuffer>()        // out
        .Ret<ffi::Buffer<ffi::S32>>() // info
);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
