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
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:frozen_dict",
        "//jax/_src:mlir",
        "//jax/_src:sharding_impls",
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
        "//jax/_src:api",
        "//jax/_src:checkify",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:debugging",
        "//jax/_src:dtypes",
        "//jax/_src:lax",
        "//jax/_src:literals",
        "//jax/_src:mesh",
        "//jax/_src:mlir",
        "//jax/_src:partial_eval",
        "//jax/_src:source_info_util",
        "//jax/_src:state_types",
        "//jax/_src:tree_util",
        "//jax/_src:util",
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
        "//jax/_src:core",
        "//jax/_src:custom_batching",
        "//jax/_src:dtypes",
        "//jax/_src:effects",
        "//jax/_src:frozen_dict",
        "//jax/_src:lax",
        "//jax/_src:pretty_printer",
        "//jax/_src:state_types",
        "//jax/_src:tree_util",
        "//jax/_src:util",
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
        "//jax/_src:core",
        "//jax/_src:debugging",
        "//jax/_src:dtypes",
        "//jax/_src:lax",
        "//jax/_src:literals",
        "//jax/_src:pretty_printer",
        "//jax/_src:state_types",
        "//jax/_src:tree_util",
        "//jax/_src:util",
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
        "//jax/_src:core",
        "//jax/_src:partial_eval",
        "//jax/_src:state_types",
        "//jax/_src:util",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic_gpu",
        "//jax/experimental:pallas",
    ],
)

pytype_strict_library(
    name = "helpers",
    srcs = ["helpers.py"],
    deps = [
        ":core",
        ":primitives",
        "//jax",
        "//jax/_src:dtypes",
        "//jax/_src:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
        "//jax/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "torch",
    srcs = ["torch.py"],
    deps = [
        "//jax",
        "//jax/_src:dtypes",
        "//jax/_src:util",
        "//jax/_src/lib",
        "//jax/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)
