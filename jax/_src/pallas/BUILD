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

load("@rules_python//python:defs.bzl", "py_library")
load(
    "//jaxlib:jax.bzl",
    "py_deps",
)

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//jax:internal",
    ],
)

py_library(
    name = "pallas",
    srcs = [
        "__init__.py",
        "core.py",
        "cost_estimate.py",
        "helpers.py",
        "hlo_interpreter.py",
        "pallas_call.py",
        "primitives.py",
        "utils.py",
    ],
    deps = [
        "//jax/_src:ad",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:api_util",
        "//jax/_src:basearray",
        "//jax/_src:batching",
        "//jax/_src:checkify",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:custom_derivatives",
        "//jax/_src:debugging",
        "//jax/_src:dtypes",
        "//jax/_src:effects",
        "//jax/_src:export",
        "//jax/_src:frozen_dict",
        "//jax/_src:hijax",
        "//jax/_src:lax",
        "//jax/_src:mlir",
        "//jax/_src:numpy",
        "//jax/_src:partial_eval",
        "//jax/_src:pretty_printer",
        "//jax/_src:source_info_util",
        "//jax/_src:tree_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

py_test(
    name = "cost_estimate_test",
    srcs = ["cost_estimate_test.py"],
    deps = [
        ":pallas",
        "//jax",
    ] + py_deps("absl/testing"),
)
