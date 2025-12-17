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

# Package for Pallas TPU Interpret Mode

load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "py_deps", "pytype_strict_library")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//jax:internal",
    ],
)

py_library(
    name = "interpret_pallas_call",
    srcs = [
        "__init__.py",
        "interpret_pallas_call.py",
    ],
    deps = [
        ":race_detection_state",
        ":shared_memory",
        ":thread_map",
        ":utils",
        ":vector_clock",
        "//jax",
        "//jax/_src:api",
        "//jax/_src:callback",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:frozen_dict",
        "//jax/_src:lax",
        "//jax/_src:mlir",
        "//jax/_src:source_info_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:core",
        "//jax/_src/pallas/mosaic:primitives",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "vector_clock",
    srcs = ["vector_clock.py"],
    deps = py_deps("numpy"),
)

pytype_strict_library(
    name = "shared_memory",
    srcs = ["shared_memory.py"],
    deps = [
        ":race_detection_state",
        ":vector_clock",
        "//jax",
        "//jax/_src:source_info_util",
        "//jax/_src:typing",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:core",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "race_detection_state",
    srcs = ["race_detection_state.py"],
    deps = [
        ":vector_clock",
        "//jax/_src:source_info_util",
    ],
)

pytype_strict_library(
    name = "thread_map",
    srcs = ["thread_map.py"],
    deps = [
        "//jax",
        "//jax/_src:callback",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//jax",
        "//jax/_src:core",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)
