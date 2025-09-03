# Copyright 2025 The JAX Authors.
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

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "roofline_core",
    srcs = [
        "roofline.py",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//jax",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:mesh",
        "//jax/_src:numpy",
        "//jax/_src:partial_eval",
        "//jax/_src:random",
        "//jax/_src:shard_map",
        "//jax/_src:source_info_util",
        "//jax/_src:traceback_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "roofline_rules",
    srcs = [
        "rooflines.py",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":roofline_core",
        "//jax",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:lax_reference",
        "//jax/_src:numpy",
        "//jax/_src:random",
        "//jax/_src:shard_map",
        "//jax/_src:util",
        "//jax/experimental/jax2tf",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "roofline",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//jax/tests:__pkg__"],
    deps = [
        ":roofline_core",
        ":roofline_rules",
    ],
)
