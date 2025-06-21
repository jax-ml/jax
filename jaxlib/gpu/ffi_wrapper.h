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

#ifndef JAXLIB_GPU_FFI_WRAPPER_H_
#define JAXLIB_GPU_FFI_WRAPPER_H_

#include <cstddef>
#include <string_view>
#include <vector>

#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

inline std::vector<void*> CombineBuffers(xla::ffi::RemainingArgs args,
                                         xla::ffi::RemainingRets rets) {
  size_t num_args = args.size();
  size_t num_rets = rets.size();
  std::vector<void*> buffers;
  buffers.reserve(num_args + num_rets);
  for (size_t i = 0; i < args.size(); ++i) {
    buffers.push_back(args.get<xla::ffi::AnyBuffer>(i).value().untyped_data());
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    buffers.push_back(rets.get<xla::ffi::AnyBuffer>(i).value()->untyped_data());
  }
  return buffers;
}

template <typename Fn>
inline xla::ffi::Error WrapLegacyKernel(Fn fn, gpuStream_t stream,
                                        std::string_view opaque,
                                        xla::ffi::RemainingArgs args,
                                        xla::ffi::RemainingRets rets) {
  std::vector<void*> buffers = CombineBuffers(args, rets);
  FFI_RETURN_IF_ERROR_STATUS(
      fn(stream, buffers.data(), opaque.data(), opaque.size()));
  return xla::ffi::Error::Success();
}

#define JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(name, fn)               \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                       \
      name,                                                            \
      [](gpuStream_t stream, std::string_view opaque,                  \
         xla::ffi::RemainingArgs args, xla::ffi::RemainingRets rets) { \
        return WrapLegacyKernel(fn, stream, opaque, args, rets);       \
      },                                                               \
      xla::ffi::Ffi::Bind()                                            \
          .Ctx<xla::ffi::PlatformStream<gpuStream_t>>()                \
          .Attr<std::string_view>("opaque")                            \
          .RemainingArgs()                                             \
          .RemainingRets());

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_FFI_WRAPPER_H_
