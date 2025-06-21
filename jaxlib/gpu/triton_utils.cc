/* Copyright 2023 The JAX Authors.

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

#include "jaxlib/gpu/triton_utils.h"

#include <zlib.h>

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/triton.pb.h"
#include "jaxlib/gpu/vendor.h"

namespace jax::JAX_GPU_NAMESPACE {

absl::StatusOr<std::string> ZlibUncompress(absl::string_view compressed) {
  std::string data;
  uLongf dest_len = 5 * compressed.size();
  while (true) {
    data.resize(dest_len);
    int ret = uncompress(reinterpret_cast<Bytef*>(data.data()), &dest_len,
                         reinterpret_cast<const Bytef*>(compressed.data()),
                         compressed.size());
    if (ret == Z_OK) {
      // `uncompress` overwrites `dest_len` with the uncompressed size.
      data.resize(dest_len);
      break;
    } else if (ret == Z_BUF_ERROR) {
      dest_len *= 2;  // The string buffer wasn't large enough.
    } else {
      return absl::InvalidArgumentError("Failed to uncompress opaque data.");
    }
  }
  return data;
}

absl::StatusOr<std::string> GetTritonKernelCallName(absl::string_view opaque) {
  JAX_ASSIGN_OR_RETURN(std::string serialized, ZlibUncompress(opaque));
  jax_triton::TritonAnyKernelCall proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InvalidArgumentError("Failed to parse serialized data.");
  }
  return proto.name();
}

absl::StatusOr<std::string> GetTritonKernelCallSerializedMetadata(
    absl::string_view opaque) {
  JAX_ASSIGN_OR_RETURN(std::string serialized, ZlibUncompress(opaque));
  jax_triton::TritonAnyKernelCall proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InvalidArgumentError("Failed to parse serialized data.");
  }
  return proto.metadata();
}

}  // namespace jax::JAX_GPU_NAMESPACE
