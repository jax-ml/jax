# Copyright 2026 The JAX Authors.
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
    "py_deps",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

pytype_strict_library(
    name = "ragged_dot",
    srcs = ["ragged_dot.py"],
    type_checking = "pyrefly",
    visibility = ["//jax:internal"],
    deps = if_building_jaxlib(
        if_building = [
            "//jax/_src:api",
            "//jax/_src:lax",
            "//jax/_src:core",
            "//jax/_src:tree_util",
            "//jax/_src:typing",
            "//jax/_src:xla_bridge",
            "//jax/_src:numpy",
        ],
        if_not_building = [],
    ) + py_deps("numpy") + [
        "//jax/_src:init",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_triton",
    ],
)
