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
        ":fusable",
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
        "//jax:ad_util",
        "//jax:api_util",
        "//jax:core",
        "//jax:partial_eval",
        "//jax:tree_util",
        "//jax:util",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "fusable",
    srcs = [
        "fusable.py",
    ],
    deps = [
        ":fusion",
        "//jax",
        "//jax:api_util",
        "//jax:core",
        "//jax:mlir",
        "//jax:partial_eval",
        "//jax:tree_util",
        "//jax:util",
    ],
)

pytype_strict_library(
    name = "fusion",
    srcs = [
        "fusion.py",
    ],
    deps = [
        "//jax",
        "//jax:util",
    ],
)

pytype_strict_library(
    name = "jaxpr_fusion",
    srcs = [
        "jaxpr_fusion.py",
    ],
    deps = [
        ":fusable",
        ":fusable_dtype",
        ":fusion",
        "//jax",
        "//jax:api_util",
        "//jax:core",
        "//jax:partial_eval",
        "//jax:tree_util",
        "//jax:util",
    ],
)

pytype_strict_library(
    name = "fusable_dtype",
    srcs = [
        "fusable_dtype.py",
    ],
    deps = [
        ":block_spec",
        ":fusable",
        "//jax",
        "//jax:api_util",
        "//jax:core",
        "//jax:dtypes",
        "//jax:partial_eval",
        "//jax:source_info_util",
        "//jax:tree_util",
        "//jax:util",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "custom_evaluate",
    srcs = ["custom_evaluate.py"],
    deps = [
        ":fuser_utils",
        "//jax",
        "//jax:core",
        "//jax:source_info_util",
        "//jax:tree_util",
        "//jax:util",
    ],
)

pytype_strict_library(
    name = "fuser_utils",
    srcs = ["fuser_utils.py"],
    deps = [
        "//jax:api_util",
        "//jax:core",
        "//jax:partial_eval",
        "//jax:tree_util",
    ],
)
