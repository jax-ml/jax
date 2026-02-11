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

# Package for Pallas GPU Interpret Mode

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
        ":gpu_callbacks",
        ":jaxpr_interpret",
        "//jax",
        "//jax/_src:callback",
        "//jax/_src:core",
        "//jax/_src:effects",
        "//jax/_src:lax",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic/interpret:thread_map",
        "//jax/_src/pallas/mosaic/interpret:utils",
        "//jax/_src/pallas/mosaic_gpu:core",
        "//jax/experimental:pallas_mosaic_gpu",
    ],
)

pytype_strict_library(
    name = "gpu_callbacks",
    srcs = ["gpu_callbacks.py"],
    deps = [
        ":shared_memory",
        "//jax",
        "//jax/_src:callback",
        "//jax/_src:lax",
        "//jax/_src:source_info_util",
        "//jax/_src/pallas/mosaic/interpret:race_detection_state",
        "//jax/_src/pallas/mosaic/interpret:utils",
        "//jax/_src/pallas/mosaic/interpret:vector_clock",
        "//jax/_src/pallas/mosaic_gpu:core",
    ] + py_deps([
        "numpy",
    ]),
)

pytype_strict_library(
    name = "jaxpr_interpret",
    srcs = ["jaxpr_interpret.py"],
    deps = [
        ":gpu_callbacks",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:source_info_util",
        "//jax/_src:util",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic/interpret:utils",
        "//jax/_src/pallas/mosaic_gpu:core",
        "//jax/_src/pallas/mosaic_gpu:primitives",
        "//jax/experimental:pallas_mosaic_gpu",
    ],
)

pytype_strict_library(
    name = "shared_memory",
    srcs = ["shared_memory.py"],
    deps = [
        "//jax/_src/pallas/mosaic/interpret:shared_memory",
        "//jax/_src/pallas/mosaic/interpret:utils",
        "//jax/_src/pallas/mosaic/interpret:vector_clock",
    ] + py_deps("absl/logging"),
)
