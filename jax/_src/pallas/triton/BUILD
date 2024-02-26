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
    "py_library_providing_imports_info",
    "pytype_strict_library",
)
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library_providing_imports_info(
    name = "triton",
    srcs = ["__init__.py"],
    lib_rule = pytype_strict_library,
    deps = [
        ":lowering",
        ":pallas_call_registration",
        "//jax/_src/lib",
    ],
)

# TODO(slebedev): Enable pytype for this target.
py_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        "//jax",
    ],
)

pytype_strict_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":lowering",
        "//jax",
        "//jax:config",
        "//jax:mlir",
        "//jax:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ] + py_deps("jax_triton"),
)
