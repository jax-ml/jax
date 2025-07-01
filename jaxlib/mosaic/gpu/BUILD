# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "nanobind_extension")

package(
    default_applicable_licenses = [],
    # Copybara this out and make it public
    default_visibility = ["@xla//xla/mosaic:mosaic_gpu_users"],
)

py_library(
    name = "mosaic_gpu",
    data = [":libmosaic_gpu_runtime.so"],
    deps = [":_mosaic_gpu_ext"],
)

nanobind_extension(
    name = "_mosaic_gpu_ext",
    srcs = ["mosaic_gpu_ext.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@nanobind",
        "@xla//xla/backends/gpu/runtime:cuda_vendor",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/tsl/cuda:cudart",
    ],
)
