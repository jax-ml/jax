# Copyright 2023 The JAX Authors.
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

# Package for Triton-specific Pallas extensions

load(
    "//jaxlib:jax.bzl",
    "py_deps",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//jax:internal",
    ],
)

pytype_strict_library(
    name = "core",
    srcs = [
        "__init__.py",
        "core.py",
    ],
    deps = ["//jax/_src/pallas"],
)

pytype_strict_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":lowering",
        "//jax",
        "//jax/_src:ad_util",
        "//jax/_src:api_util",
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:mlir",
        "//jax/_src:partial_eval",
        "//jax/_src:source_info_util",
        "//jax/_src:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        "//jax",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:api_util",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:custom_derivatives",
        "//jax/_src:debugging",
        "//jax/_src:lax",
        "//jax/_src:literals",
        "//jax/_src:mlir",
        "//jax/_src:partial_eval",
        "//jax/_src:source_info_util",
        "//jax/_src:state_types",
        "//jax/_src:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":core",
        ":lowering",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:frozen_dict",
        "//jax/_src:mlir",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ],
)
