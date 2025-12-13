# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//jaxlib:jax.bzl", "nanobind_extension")

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "tiled_layout",
    srcs = ["tiled_layout.cc"],
    hdrs = ["tiled_layout.h"],
    deps = ["@com_google_absl//absl/log:check"],
)

cc_test(
    name = "tiled_layout_test",
    srcs = ["tiled_layout_test.cc"],
    deps = [
        ":tiled_layout",
        "//testing/base/public:gunit_main",
    ],
)

nanobind_extension(
    name = "ext",
    srcs = ["ext.cc"],
    visibility = ["//jax:__subpackages__"],
    deps = [
        ":tiled_layout",
        "@com_google_absl//absl/hash",
        "@nanobind",
    ],
)
