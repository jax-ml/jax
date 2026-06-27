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

load("//jaxlib:jax.bzl", "jax_visibility", "py_deps", "pytype_library", "pytype_strict_library")

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
    type_checking = "pyrefly",
    deps = [
        "//jax",
        "//jax/_src:core",
        "//jax/_src:deprecations",
        "//jax/_src:flattree",
        "//jax/_src:frozen_dict",
        "//jax/_src:lax",
        "//jax/_src:partial_eval",
        "//jax/_src:tree_util",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "error_handling",
    srcs = ["error_handling.py"],
    type_checking = "pyrefly",
    deps = [
        "//jax/_src:compiler",
        "//jax/_src:traceback_util",
        "//jax/_src/lib",
    ],
)

pytype_library(
    name = "primitives",
    srcs = ["primitives.py"],
    type_checking = "pyrefly",
    deps = [
        ":core",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:dtypes",
        "//jax/_src:flattree",
        "//jax/_src:mlir",
        "//jax/_src:pretty_printer",
        "//jax/_src:tree_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ],
)

pytype_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    type_checking = "pyrefly",
    deps = [
        ":core",
        ":helpers",
        ":lowering",
        ":sc_core",
        ":sc_lowering",
        ":tpu_info",
        "//jax",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:dtypes",
        "//jax/_src:flattree",
        "//jax/_src:frozen_dict",
        "//jax/_src:lax",
        "//jax/_src:mlir",
        "//jax/_src:partial_eval",
        "//jax/_src:sharding_impls",
        "//jax/_src:source_info_util",
        "//jax/_src:tpu_custom_call",
        "//jax/_src:tree_util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    type_checking = "pyrefly",
    visibility = [
        "//jax:internal",
    ] + jax_visibility("pallas/mosaic/lowering"),
    deps = [
        ":core",
        ":error_handling",
        ":primitives",
        ":random",
        ":sc_core",
        ":tpu_info",
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
    type_checking = "pyrefly",
    visibility = ["//visibility:private"],
    deps = [
        ":core",
        ":lowering",
        ":pipeline",
        ":primitives",
        ":sc_core",
        "//jax/_src:core",
        "//jax/_src:debugging",
        "//jax/_src:lax",
        "//jax/_src:numpy",
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
    type_checking = "pyrefly",
    visibility = ["//jax/experimental:pallas_sc_users"],
    deps = [
        ":core",
        ":tpu_info",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:tree_util",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "sc_primitives",
    srcs = [
        "sc_primitives.py",
    ],
    type_checking = "pyrefly",
    deps = [
        ":core",
        ":lowering",
        ":sc_core",
        ":sc_lowering",
        "//jax",
        "//jax/_src:api_util",
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

pytype_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    type_checking = "pyrefly",
    deps = [
        ":core",
        ":helpers",
        ":primitives",
        ":tpu_info",
        "//jax",
        "//jax/_src:api_util",
        "//jax/_src:lax",
        "//jax/_src:util",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:lowering",
        "//jax/extend:backend",
    ] + py_deps("numpy"),
)

pytype_library(
    name = "random",
    srcs = ["random.py"],
    type_checking = "pyrefly",
    deps = [
        ":primitives",
        "//jax",
        "//jax/_src:blocked_sampler",
        "//jax/_src:typing",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "helpers",
    srcs = ["helpers.py"],
    type_checking = "pyrefly",
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
    type_checking = "pyrefly",
    deps = [
        "//jax",
        "//jax/_src:api",
        "//jax/_src:core",
        "//jax/_src:dtypes",
        "//jax/_src:mesh",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ],
)
