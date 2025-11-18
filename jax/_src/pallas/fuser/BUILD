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
    name = "fuser",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":block_spec",
        ":custom_evaluate",
        ":custom_fusion",
        ":fusible",
        ":fusion",
        ":jaxpr_fusion",
    ],
)

pytype_strict_library(
    name = "block_spec",
    srcs = [
        "block_spec.py",
    ],
    deps = [
        ":fuser_utils",
        "//jax",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:api_util",
        "//jax/_src:core",
        "//jax/_src:custom_derivatives",
        "//jax/_src:hijax",
        "//jax/_src:lax",
        "//jax/_src:partial_eval",
        "//jax/_src:random",
        "//jax/_src:traceback_util",
        "//jax/_src:tree_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "custom_fusion",
    srcs = [
        "custom_fusion_lib.py",
    ],
    deps = [
        ":block_spec",
        "//jax/_src:api_util",
        "//jax/_src:core",
        "//jax/_src:custom_api_util",
        "//jax/_src:lax",
        "//jax/_src:mlir",
        "//jax/_src:partial_eval",
        "//jax/_src:traceback_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:lowering",
    ],
)

pytype_strict_library(
    name = "fusible",
    srcs = [
        "fusible.py",
    ],
    deps = [
        ":fusion",
        "//jax",
        "//jax/_src:api_util",
        "//jax/_src:batching",
        "//jax/_src:core",
        "//jax/_src:mlir",
        "//jax/_src:partial_eval",
        "//jax/_src:traceback_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
    ],
)

pytype_strict_library(
    name = "fusion",
    srcs = [
        "fusion.py",
    ],
    deps = [
        "//jax",
        "//jax/_src:util",
    ],
)

pytype_strict_library(
    name = "jaxpr_fusion",
    srcs = [
        "jaxpr_fusion.py",
    ],
    deps = [
        ":fusible",
        ":fusible_dtype",
        ":fusion",
        "//jax",
        "//jax/_src:api_util",
        "//jax/_src:core",
        "//jax/_src:partial_eval",
        "//jax/_src:traceback_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
    ],
)

pytype_strict_library(
    name = "fusible_dtype",
    srcs = [
        "fusible_dtype.py",
    ],
    deps = [
        ":block_spec",
        ":fusible",
        "//jax",
        "//jax/_src:api_util",
        "//jax/_src:core",
        "//jax/_src:custom_derivatives",
        "//jax/_src:dtypes",
        "//jax/_src:lax",
        "//jax/_src:partial_eval",
        "//jax/_src:source_info_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "custom_evaluate",
    srcs = ["custom_evaluate.py"],
    deps = [
        ":fuser_utils",
        "//jax",
        "//jax/_src:core",
        "//jax/_src:source_info_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
    ],
)

pytype_strict_library(
    name = "fuser_utils",
    srcs = ["fuser_utils.py"],
    deps = [
        "//jax/_src:api_util",
        "//jax/_src:core",
        "//jax/_src:partial_eval",
        "//jax/_src:tree_util",
    ],
)
