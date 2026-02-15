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

# Package for Mosaic-specific Pallas extensions

load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "py_deps", "pytype_strict_library")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//jax:internal",
    ],
)

py_library(
    name = "core",
    srcs = [
        "__init__.py",
        "core.py",
    ],
    deps = [
        "//jax",
        "//jax/_src/pallas",
        "//jax/extend:backend",
    ],
)

py_library(
    name = "error_handling",
    srcs = ["error_handling.py"],
    deps = [
        "//jax/_src:compiler",
        "//jax/_src:traceback_util",
        "//jax/_src/lib",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":core",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:dtypes",
        "//jax/_src:mlir",
        "//jax/_src:pretty_printer",
        "//jax/_src:tree_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ],
)

py_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":core",
        ":lowering",
        ":sc_lowering",
        "//jax",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:frozen_dict",
        "//jax/_src:lax",
        "//jax/_src:mlir",
        "//jax/_src:sharding_impls",
        "//jax/_src:source_info_util",
        "//jax/_src:tpu_custom_call",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        ":core",
        ":error_handling",
        ":primitives",
        ":random",
        "//jax",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:checkify",
        "//jax/_src:cloud_tpu_init",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:custom_derivatives",
        "//jax/_src:debugging",
        "//jax/_src:dtypes",
        "//jax/_src:export",
        "//jax/_src:lax",
        "//jax/_src:literals",
        "//jax/_src:mesh",
        "//jax/_src:mlir",
        "//jax/_src:partial_eval",
        "//jax/_src:random",
        "//jax/_src:source_info_util",
        "//jax/_src:traceback_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src:xla_bridge",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "sc_lowering",
    srcs = [
        "sc_lowering.py",
    ],
    visibility = ["//jax/experimental:pallas_sc_users"],
    deps = [
        ":core",
        ":lowering",
        ":primitives",
        ":sc_core",
        "//jax/_src:core",
        "//jax/_src:debugging",
        "//jax/_src:lax",
        "//jax/_src:mesh",
        "//jax/_src:mlir",
        "//jax/_src:numpy",
        "//jax/_src:source_info_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic",
    ],
)

pytype_strict_library(
    name = "sc_core",
    srcs = [
        "sc_core.py",
    ],
    visibility = ["//jax/experimental:pallas_sc_users"],
    deps = [
        ":core",
        ":tpu_info",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:partial_eval",
        "//jax/_src:tree_util",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "sc_primitives",
    srcs = [
        "sc_primitives.py",
    ],
    deps = [
        ":core",
        ":lowering",
        ":sc_core",
        ":sc_lowering",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:dtypes",
        "//jax/_src:effects",
        "//jax/_src:lax",
        "//jax/_src:partial_eval",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic",
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":core",
        ":primitives",
        ":tpu_info",
        "//jax",
        "//jax/_src:api_util",
        "//jax/_src:util",
        "//jax/_src/pallas",
        "//jax/experimental:pallas",
        "//jax/extend:backend",
    ] + py_deps("numpy"),
)

py_library(
    name = "random",
    srcs = ["random.py"],
    deps = [
        ":primitives",
        "//jax",
        "//jax/_src:blocked_sampler",
        "//jax/_src:typing",
    ] + py_deps("numpy"),
)

py_library(
    name = "helpers",
    srcs = ["helpers.py"],
    deps = [
        ":core",
        ":primitives",
        "//jax",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "tpu_info",
    srcs = ["tpu_info.py"],
    deps = [
        ":core",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:dtypes",
        "//jax/_src:util",
    ],
)
