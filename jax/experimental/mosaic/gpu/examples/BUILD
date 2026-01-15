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

load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "jax_multiplatform_test", "py_deps")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax/experimental:mosaic_gpu_users"],
)

exports_files(
    srcs = [
        "flash_attention.py",
        "matmul.py",
    ],
    visibility = ["//jax:internal"],
)

py_library(
    name = "gpu_examples",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//jax:internal"],
)

py_library(
    name = "matmul",
    srcs = ["matmul.py"],
    deps = [
        "//jax",
        "//jax/experimental:mosaic_gpu",
    ],
)

py_library(
    name = "matmul_blackwell",
    srcs = ["matmul_blackwell.py"],
    deps = [
        "//jax",
        "//jax/experimental:mosaic_gpu",
    ],
)

py_library(
    name = "flash_attention",
    srcs = ["flash_attention.py"],
    deps = [
        "//jax",
        "//jax/experimental:mosaic_gpu",
    ],
)

jax_multiplatform_test(
    name = "run_matmul",
    srcs = ["matmul.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100"],
    main = "matmul.py",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax/experimental:mosaic_gpu",
    ] + py_deps("numpy"),
)
