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
    "if_building_jaxlib",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:jax_extend_users"],
)

pytype_strict_library(
    name = "mlir",
    srcs = ["__init__.py"],
    deps = [
        ":ir",
        ":pass_manager",
        "//jax/_src:mlir",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "ir",
    srcs = ["ir.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:ir"]),
)

pytype_strict_library(
    name = "pass_manager",
    srcs = ["passmanager.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:pass_manager"]),
)
