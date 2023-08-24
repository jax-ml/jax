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

load(
    "//jaxlib:jax.bzl",
    "py_deps",
)
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library(
    name = "pallas",
    srcs = glob(
        include = ["**/*.py"],
        exclude = [
            "triton/*.py",
            "mosaic/*.py",
        ],
    ),
    deps = [
        "//jax",
        "//jax:ad_util",
        "//jax:core",
        "//jax:mlir",
        "//jax:partial_eval",
        "//jax:pretty_printer",
        "//jax:tree_util",
        "//jax:util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

py_library(
    name = "gpu",
    visibility = [],
    deps = [
        ":pallas",
        "//jax/_src/pallas/triton",
    ],
)

py_library(
    name = "tpu",
    visibility = [],
    deps = [
        ":pallas",
        "//jax/_src/pallas/mosaic",
    ],
)
