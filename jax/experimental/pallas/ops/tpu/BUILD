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
    "py_deps",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

pytype_strict_library(
    name = "all_gather",
    srcs = ["all_gather.py"],
    deps = [
        "//jax",
        "//jax/_src:shard_map",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "example_kernel",
    srcs = ["example_kernel.py"],
    deps = [
        "//jax",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ],
)

pytype_strict_library(
    name = "flash_attention",
    srcs = ["flash_attention.py"],
    deps = [
        "//jax",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "matmul",
    srcs = ["matmul.py"],
    deps = [
        "//jax",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ],
)

pytype_strict_library(
    name = "megablox",
    srcs = glob(["megablox/*.py"]),
    deps = [
        "//jax",
        "//jax/_src:dtypes",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "paged_attention",
    srcs = glob(["paged_attention/*.py"]),
    deps = [
        "//jax",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "ragged_paged_attention",
    srcs = glob(["ragged_paged_attention/*.py"]),
    deps = [
        "//jax",
        "//jax/_src:dtypes",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "random",
    srcs = glob(["random/*.py"]),
    deps = [
        "//jax",
        "//jax/_src:dtypes",
        "//jax/_src:random",
        "//jax/_src:shard_map",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "splash_attention",
    srcs = glob(["splash_attention/*.py"]),
    deps = [
        "//jax",
        "//jax/_src:util",
        "//jax/experimental:pallas",
        "//jax/experimental:pallas_tpu",
    ] + py_deps("numpy"),
)
