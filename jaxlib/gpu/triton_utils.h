#ifndef JAXLIB_GPU_TRITON_UTILS_H_
#define JAXLIB_GPU_TRITON_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "jaxlib/gpu/vendor.h"

namespace jax::JAX_GPU_NAMESPACE {

absl::StatusOr<std::string> ZlibUncompress(absl::string_view compressed);
absl::StatusOr<std::string> GetTritonKernelCallName(absl::string_view opaque);
absl::StatusOr<std::string> GetTritonKernelCallSerializedMetadata(
    absl::string_view opaque);

}  // namespace jax::JAX_GPU_NAMESPACE

#endif  // JAXLIB_GPU_TRITON_UTILS_H_
