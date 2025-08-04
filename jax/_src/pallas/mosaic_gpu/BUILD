# Copyright 2024 The JAX Authors.
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

py_library(
    name = "mosaic_gpu",
    srcs = ["__init__.py"],
    deps = [
        ":core",
        ":pallas_call_registration",
        ":primitives",
    ],
)

pytype_strict_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":core",
        ":lowering",
        "//jax",
        "//jax:config",
        "//jax:core",
        "//jax:frozen_dict",
        "//jax:mlir",
        "//jax:sharding_impls",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        ":core",
        "//jax",
        "//jax:api",
        "//jax:checkify",
        "//jax:core",
        "//jax:dtypes",
        "//jax:lax",
        "//jax:mesh",
        "//jax:mlir",
        "//jax:partial_eval",
        "//jax:source_info_util",
        "//jax:state_types",
        "//jax:tree_util",
        "//jax:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic_gpu",
        "//jax/experimental:pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "core",
    srcs = ["core.py"],
    tags = [
        # Avoid circular dependency.
        "ignore_for_dep=third_party.py.jax._src.pallas.mosaic_gpu.primitives.wgmma_accumulator_deref",
    ],
    deps = [
        "//jax",
        "//jax:core",
        "//jax:custom_batching",
        "//jax:dtypes",
        "//jax:effects",
        "//jax:frozen_dict",
        "//jax:lax",
        "//jax:pretty_printer",
        "//jax:state_types",
        "//jax:tree_util",
        "//jax:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic_gpu",
        "//jaxlib/mlir:ir",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":core",
        ":lowering",
        "//jax",
        "//jax:core",
        "//jax:dtypes",
        "//jax:lax",
        "//jax:pretty_printer",
        "//jax:state_types",
        "//jax:tree_util",
        "//jax:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":core",
        ":primitives",
        "//jax",
        "//jax:core",
        "//jax:partial_eval",
        "//jax:state_types",
        "//jax:util",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic_gpu",
        "//jax/experimental:pallas",
    ],
)

pytype_strict_library(
    name = "helpers",
    srcs = ["helpers.py"],
    deps = ["//jax"],
)
