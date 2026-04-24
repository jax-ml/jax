# Copyright 2026 The JAX Authors.
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

# INTEL ONEAPI plugin extension (glue only, no GPU kernels)

load(
    "//jaxlib:jax.bzl",
    "nanobind_extension",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

cc_library(
    name = "oneapi_vendor",
    hdrs = [
        "//jaxlib/gpu:vendor.h",
    ],
    defines = ["JAX_GPU_ONEAPI=1"],
    deps = [
        "@local_config_sycl//sycl:sycl_headers",
    ],
)

nanobind_extension(
    name = "oneapi_plugin_extension",
    srcs = ["oneapi_plugin_extension.cc"],
    module_name = "oneapi_plugin_extension",
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "//jaxlib/gpu:gpu_plugin_extension",
        "@com_google_absl//absl/status",
        "@nanobind",
    ],
)
