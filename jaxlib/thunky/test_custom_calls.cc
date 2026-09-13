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

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

static absl::Status ThunkyTestConcatRowsFfi(
    se::Stream* stream, xla::ffi::BufferR2<xla::F32> x,
    xla::ffi::BufferR2<xla::F32> y, int64_t split_row,
    xla::ffi::Result<xla::ffi::BufferR2<xla::F32>> out) {
  int64_t rows = x.dimensions()[0];
  int64_t cols = x.dimensions()[1];
  if (y.dimensions()[0] != rows || y.dimensions()[1] != cols ||
      out->dimensions()[0] != rows || out->dimensions()[1] != cols) {
    return absl::InvalidArgumentError("Shape mismatch in concat_rows");
  }
  if (split_row < 0 || split_row > rows) {
    return absl::InvalidArgumentError("split_row out of bounds");
  }
  size_t top_bytes = static_cast<size_t>(split_row * cols) * sizeof(float);
  size_t bottom_bytes =
      static_cast<size_t>((rows - split_row) * cols) * sizeof(float);
  if (top_bytes > 0) {
    se::DeviceMemoryBase dst_top = out->device_memory();
    RETURN_IF_ERROR(stream->Memcpy(&dst_top, x.device_memory(), top_bytes));
  }
  if (bottom_bytes > 0) {
    se::DeviceMemoryBase dst_bottom(
        static_cast<char*>(out->device_memory().opaque()) + top_bytes,
        bottom_bytes);
    se::DeviceMemoryBase src_bottom(
        static_cast<char*>(y.device_memory().opaque()) + top_bytes,
        bottom_bytes);
    RETURN_IF_ERROR(stream->Memcpy(&dst_bottom, src_bottom, bottom_bytes));
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kThunkyTestConcatRowsFfi, ThunkyTestConcatRowsFfi,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::Stream>()
                           .Arg<xla::ffi::BufferR2<xla::F32>>()
                           .Arg<xla::ffi::BufferR2<xla::F32>>()
                           .Attr<int64_t>("split_row")
                           .Ret<xla::ffi::BufferR2<xla::F32>>());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "thunky.test_concat_rows_f32", "CUDA",
                         kThunkyTestConcatRowsFfi);

}  // namespace
}  // namespace xla::gpu
